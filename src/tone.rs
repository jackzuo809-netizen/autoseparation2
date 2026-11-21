//! 渐变图(水彩 / 照片)分色:连续调色版,网点交给 RIP。
//!
//! 每个颜色求「底色 + 最多 4 种油墨」的混合权重,再按印刷顺序换算成各版覆盖率。
//!
//! 空间一致性(让色版成片、平滑,而不是斑驳):
//! 只追求每个像素颜色最准时,相近颜色会在两种相似油墨之间来回切换,色版就碎。
//! 所以求解时再加一项 μ‖w − w̄‖²,w̄ 是邻域的平均权重;迭代几轮后,
//! 一片区域会统一用同一组油墨和相近的深浅。最后去掉低于最小网点的覆盖率。

use rayon::prelude::*;

use crate::color::{Lab, Rgb, dist2, from_u8, lightness, rgb_to_lab};
use crate::mix::{MAXP, Mix, Mixer};
use crate::model::Separation;
use crate::palette::{Rep, choose_inks, kmeans, summarize};

pub struct ToneOpts {
    pub n: Option<usize>,
    pub fixed_inks: Option<Vec<Rgb>>,
    /// 每个像素最多叠几层油墨
    pub max_layers: usize,
    /// 空间一致性强度 μ,0 = 关闭
    pub coherence: f32,
    /// 一致性参考值的网格大小(像素)和平均半径(网格数)
    pub block: usize,
    pub radius: usize,
    /// 迭代轮数
    pub iters: usize,
    /// 被盖住部分的平滑半径(像素),0 = 关闭
    pub hidden_radius: usize,
    /// 最小网点(0..255),低于此覆盖率清零
    pub min_dot: u8,
}

fn key(p: &[u8; 3]) -> u32 {
    ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32
}

fn key_rgb(k: u32) -> Rgb {
    from_u8([(k >> 16) as u8, (k >> 8) as u8, k as u8])
}

fn delta_e(a: Rgb, b: Rgb) -> f32 {
    dist2(rgb_to_lab(a), rgb_to_lab(b)).sqrt()
}

pub fn pick_inks(pixels: &[[u8; 3]], garment: Rgb, underbase: bool, max_layers: usize, n: Option<usize>) -> Vec<Rgb> {
    let stride = (pixels.len() / 200_000).max(1);
    let rgb: Vec<Rgb> = pixels.iter().step_by(stride).map(|p| from_u8(*p)).collect();
    let lab: Vec<_> = rgb.par_iter().map(|c| rgb_to_lab(*c)).collect();
    let (assign, centers) = kmeans(&lab, 64, 20);
    let reps: Vec<Rep> = summarize(&rgb, &assign, centers.len());

    // 候选油墨:代表色 + 加深(更饱和)版本。正常混合下,浅色靠降低网点得到,
    // 油墨要处在颜色范围的「外侧」才能混出更多颜色。
    let g = rgb_to_lab(garment);
    let mut cands: Vec<Rgb> = Vec::new();
    for r in &reps {
        for gamma in [1.0f32, 1.5, 2.2, 3.0] {
            let c = r.rgb.map(|v| v.powf(gamma));
            let l = rgb_to_lab(c);
            // 和底色太接近的颜色印了也看不出来,不作候选
            if dist2(g, l) >= 100.0 && cands.iter().all(|o| dist2(rgb_to_lab(*o), l) > 16.0) {
                cands.push(c);
            }
        }
    }
    let err = |inks: &[Rgb], r: &Rep| {
        let m = Mixer::new(garment, underbase, inks, max_layers);
        delta_e(m.color(&m.solve(r.rgb, u32::MAX)), r.rgb)
    };
    // 自动色数:平均 ΔE 达标,或再加一色收益很小就停
    let (inks, _) = choose_inks(&reps, &cands, n, 16, err, |e, prev| {
        e.mean < 2.0 || prev.is_some_and(|p| p.mean - e.mean < 0.3)
    });
    inks
}

pub fn separate(
    pixels: &[[u8; 3]],
    w: usize,
    h: usize,
    garment: Rgb,
    underbase: bool,
    opts: &ToneOpts,
    gpu: Option<&crate::gpu::Gpu>,
) -> Separation {
    let mut inks = match &opts.fixed_inks {
        Some(v) => v.clone(),
        None => pick_inks(pixels, garment, underbase, opts.max_layers, opts.n),
    };
    // 印刷顺序:浅 → 深(深色在上层)
    inks.sort_by(|a, b| lightness(*b).total_cmp(&lightness(*a)));
    let mixer = Mixer::new(garment, underbase, &inks, opts.max_layers);
    let all = u32::MAX;

    // 按唯一颜色求解
    let mut keys: Vec<u32> = pixels.par_iter().map(key).collect();
    keys.par_sort_unstable();
    keys.dedup();
    let mixes: Vec<Mix> = keys.par_iter().map(|&k| mixer.solve(key_rgb(k), all)).collect();
    let lookup = |p: &[u8; 3]| keys.binary_search(&key(p)).unwrap();
    let uidx: Vec<u32> = pixels.par_iter().map(|p| lookup(p) as u32).collect();
    println!("求解 {} 种颜色", keys.len());

    let n = w * h;
    let q = |v: f32| {
        let b = (v * 255.0).round() as u8;
        if b < opts.min_dot { 0 } else { b }
    };
    let mut plates = vec![vec![0u8; n]; inks.len()];
    let mut under = if underbase { vec![0u8; n] } else { vec![] };

    if opts.coherence <= 0.0 {
        let per_pixel = |i: usize| mixes[uidx[i] as usize];
        write_plates(&mut plates, &mut under, w, h, per_pixel, &mixer, q);
        smooth_hidden(&mut plates, w, h, opts.hidden_radius, opts.min_dot);
        return Separation { inks, plates, under: underbase.then_some(under) };
    }

    // 空间一致性迭代:在低分辨率网格上维护邻域平均权重 w̄
    let f = opts.block.max(1);
    let (lw, lh) = (w.div_ceil(f), h.div_ceil(f));
    // 每个网格的平均颜色(Lab),用来做「按颜色区分邻域」的引导
    let guide: Vec<Lab> = (0..lw * lh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % lw, i / lw);
            let mut acc = [0f32; 3];
            let mut cnt = 0.0;
            for y in (by * f)..((by + 1) * f).min(h) {
                for x in (bx * f)..((bx + 1) * f).min(w) {
                    let l = rgb_to_lab(from_u8(pixels[y * w + x]));
                    for k in 0..3 {
                        acc[k] += l[k];
                    }
                    cnt += 1.0;
                }
            }
            acc.map(|v| v / cnt)
        })
        .collect();
    // 引导色再平滑一圈(3×3 网格),扫描颗粒就不会被当成区域边界
    let guide: Vec<Lab> = (0..lw * lh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % lw, i / lw);
            let mut acc = [0f32; 3];
            let mut cnt = 0.0;
            for y in by.saturating_sub(1)..=(by + 1).min(lh - 1) {
                for x in bx.saturating_sub(1)..=(bx + 1).min(lw - 1) {
                    for k in 0..3 {
                        acc[k] += guide[y * lw + x][k];
                    }
                    cnt += 1.0;
                }
            }
            acc.map(|v| v / cnt)
        })
        .collect();
    let sim = |a: Lab, b: Lab| (-dist2(a, b) / (2.0 * SIGMA * SIGMA)).exp();
    // 像素的「平滑色」:3×3 平均,同样是为了忽略颗粒
    let smooth_lab = |x: usize, y: usize| -> Lab {
        let mut acc = [0f32; 3];
        let mut cnt = 0.0;
        for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let p = pixels[yy * w + xx];
                for k in 0..3 {
                    acc[k] += p[k] as f32;
                }
                cnt += 1.0;
            }
        }
        rgb_to_lab(acc.map(|v| v / cnt / 255.0))
    };
    // 迭代阶段只需要邻域平均,抽样计算即可(每 step×step 取一个像素)
    let accumulate = |solve_px: &(dyn Fn(usize, usize) -> Mix + Sync), step: usize| -> Vec<[f32; MAXP]> {
        let rows: Vec<Vec<[f32; MAXP]>> = (0..lh)
            .into_par_iter()
            .map(|ly| {
                let mut acc = vec![[0f32; MAXP]; lw];
                let mut cnt = vec![0f32; lw];
                for y in ((ly * f)..((ly + 1) * f).min(h)).filter(|y| y % step == 0) {
                    for x in (0..w).step_by(step) {
                        let m = solve_px(x, y);
                        let a = &mut acc[x / f];
                        for p in 0..MAXP {
                            a[p] += m.w[p];
                        }
                        cnt[x / f] += 1.0;
                    }
                }
                for (a, c) in acc.iter_mut().zip(&cnt) {
                    for v in a.iter_mut() {
                        *v /= c.max(1.0);
                    }
                }
                acc
            })
            .collect();
        let mut low: Vec<[f32; MAXP]> = rows.into_iter().flatten().collect();
        bilateral_blur(&mut low, &guide, lw, lh, opts.radius);
        low
    };
    // 取像素的参考权重:周围 4 个网格按距离(双线性)× 颜色相似度加权,
    // 这样边缘处不会混进另一侧区域的油墨
    let sample = |low: &[[f32; MAXP]], x: usize, y: usize, lab: Lab| -> [f32; MAXP] {
        let fx = ((x as f32 + 0.5) / f as f32 - 0.5).clamp(0.0, (lw - 1) as f32);
        let fy = ((y as f32 + 0.5) / f as f32 - 0.5).clamp(0.0, (lh - 1) as f32);
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(lw - 1), (y0 + 1).min(lh - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let cells = [
            (y0 * lw + x0, (1.0 - tx) * (1.0 - ty)),
            (y0 * lw + x1, tx * (1.0 - ty)),
            (y1 * lw + x0, (1.0 - tx) * ty),
            (y1 * lw + x1, tx * ty),
        ];
        let mut out = [0f32; MAXP];
        let mut total = 0.0;
        for (c, bw) in cells {
            let wgt = bw * sim(lab, guide[c]) + 1e-6 * bw;
            for p in 0..MAXP {
                out[p] += low[c][p] * wgt;
            }
            total += wgt;
        }
        out.map(|v| v / total.max(1e-12))
    };
    let mu = opts.coherence;
    let base = mixer.base;
    let n_inks = inks.len();
    // 候选油墨:不加约束时这个颜色用到的油墨 + 邻域里用得多的油墨(缩小求解规模)
    let allowed_at = |i: usize, wbar: &[f32; MAXP]| -> u32 {
        let m0 = &mixes[uidx[i] as usize];
        (0..n_inks).filter(|&k| m0.w[base + k] > 1e-4 || wbar[base + k] > 0.02).fold(0, |acc, k| acc | (1 << k))
    };
    let solve_at = |i: usize, prior: &[[f32; MAXP]]| {
        let t = from_u8(pixels[i]);
        let (x, y) = (i % w, i / w);
        let wbar = sample(prior, x, y, smooth_lab(x, y));
        let m0 = &mixes[uidx[i] as usize];
        // 颜色最准的解已经和邻域一致(比如大面积纯色背景):约束不起作用,直接用
        if (0..MAXP).all(|p| (m0.w[p] - wbar[p]).abs() < 0.02) {
            return *m0;
        }
        mixer.solve_reg(t, allowed_at(i, &wbar), Some((&wbar, mu)))
    };
    let step = (f / 2).max(1);
    let mut low = accumulate(&|x, y| mixes[uidx[y * w + x] as usize], step);

    // GPU:迭代和最后一遍都在显卡上算;失败时退回 CPU
    let gpu_done = gpu.is_some_and(|g| {
        let mixes_flat: Vec<f32> = mixes.iter().flat_map(|m| m.w.iter().copied()).collect();
        let job = crate::gpu::Job {
            w,
            h,
            f,
            lw,
            lh,
            base,
            n_inks,
            max_layers: opts.max_layers,
            step,
            min_dot: opts.min_dot,
            mu,
            sigma: SIGMA,
            sum_w: crate::mix::SUM_WEIGHT as f32,
            pts: &mixer.pts,
            pixels,
            uidx: &uidx,
            mixes: &mixes_flat,
            guide: &guide,
        };
        let res = (|| -> anyhow::Result<()> {
            for _ in 0..opts.iters {
                let mut acc = vec![[0f32; MAXP]; lw * lh];
                let mut cnt = vec![0f32; lw * lh];
                g.run_samples(&job, &low, |x, y, wts| {
                    let c = (y / f) * lw + x / f;
                    for p in 0..MAXP {
                        acc[c][p] += wts[p];
                    }
                    cnt[c] += 1.0;
                })?;
                for (a, c) in acc.iter_mut().zip(&cnt) {
                    for v in a.iter_mut() {
                        *v /= c.max(1.0);
                    }
                }
                bilateral_blur(&mut acc, &guide, lw, lh, opts.radius);
                low = acc;
            }
            let ow = crate::gpu::Gpu::out_words(n_inks);
            g.run_final(&job, &low, |y0, rows, words| {
                let range = y0 * w..(y0 + rows) * w;
                for (k, plate) in plates.iter_mut().enumerate() {
                    plate[range.clone()].par_iter_mut().enumerate().for_each(|(i, v)| {
                        *v = (words[i * ow + k / 4] >> ((k % 4) * 8)) as u8;
                    });
                }
                if !under.is_empty() {
                    under[range.clone()].par_iter_mut().enumerate().for_each(|(i, v)| {
                        *v = (words[i * ow + n_inks / 4] >> ((n_inks % 4) * 8)) as u8;
                    });
                }
            })
        })();
        match res {
            Ok(()) => true,
            Err(e) => {
                eprintln!("GPU 计算失败,改用 CPU:{e:#}");
                false
            }
        }
    });
    if !gpu_done {
        for _ in 0..opts.iters {
            let prior = low;
            low = accumulate(&|x, y| solve_at(y * w + x, &prior), step);
        }
        let solve_final = |i: usize| solve_at(i, &low);
        write_plates(&mut plates, &mut under, w, h, solve_final, &mixer, q);
    }
    smooth_hidden(&mut plates, w, h, opts.hidden_radius, opts.min_dot);
    Separation { inks, plates, under: underbase.then_some(under) }
}

/// 被上层盖住的地方,下层网点值对颜色影响很小,但换算时会被放大成剧烈跳动。
/// 按「露出比例」v 混合:a' = v·a + (1 − v)·邻域平均(a),盖得越多越平滑。
fn smooth_hidden(plates: &mut [Vec<u8>], w: usize, h: usize, r: usize, min_dot: u8) {
    if r == 0 || plates.len() < 2 {
        return;
    }
    let n = w * h;
    // 从最上层往下累计露出比例
    let mut visible = vec![1f32; n];
    for k in (0..plates.len()).rev() {
        if k + 1 < plates.len() {
            let above = &plates[k + 1];
            visible.par_iter_mut().zip(above.par_iter()).for_each(|(v, &a)| *v *= 1.0 - a as f32 / 255.0);
        }
        let mean = box_mean(&plates[k], w, h, r);
        plates[k].par_iter_mut().zip(visible.par_iter()).zip(mean.par_iter()).for_each(|((a, &v), &m)| {
            let x = v * *a as f32 + (1.0 - v) * m;
            let b = x.round() as u8;
            *a = if b < min_dot { 0 } else { b };
        });
    }
}

/// 方框平均(可分离,半径 r)。
fn box_mean(plane: &[u8], w: usize, h: usize, r: usize) -> Vec<f32> {
    let tmp: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
            (a..=b).map(|xx| plane[y * w + xx] as f32).sum::<f32>() / (b - a + 1) as f32
        })
        .collect();
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
            (a..=b).map(|yy| tmp[yy * w + x]).sum::<f32>() / (b - a + 1) as f32
        })
        .collect()
}

/// 颜色相似度的尺度(ΔE):相差超过约 2σ 的邻居基本不参与平均。
const SIGMA: f32 = 10.0;

/// 低分辨率网格上的联合双边平均(可分离近似,半径 r):
/// 只和颜色相近的邻居取平均,区域内部大范围统一,但不越过颜色边界。
fn bilateral_blur(grid: &mut [[f32; MAXP]], guide: &[Lab], w: usize, h: usize, r: usize) {
    if r == 0 {
        return;
    }
    let pass = |src: &[[f32; MAXP]], horizontal: bool| -> Vec<[f32; MAXP]> {
        (0..w * h)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let mut acc = [0f32; MAXP];
                let mut total = 0.0;
                let range = if horizontal {
                    x.saturating_sub(r)..=(x + r).min(w - 1)
                } else {
                    y.saturating_sub(r)..=(y + r).min(h - 1)
                };
                for k in range {
                    let j = if horizontal { y * w + k } else { k * w + x };
                    let wgt = (-dist2(guide[i], guide[j]) / (2.0 * SIGMA * SIGMA)).exp();
                    for p in 0..MAXP {
                        acc[p] += src[j][p] * wgt;
                    }
                    total += wgt;
                }
                acc.map(|v| v / total)
            })
            .collect()
    };
    let tmp = pass(grid, true);
    let out = pass(&tmp, false);
    grid.copy_from_slice(&out);
}

/// 逐行并行求每个像素的配方并写入各色版(不额外占内存)。
fn write_plates(
    plates: &mut [Vec<u8>],
    under: &mut [u8],
    w: usize,
    h: usize,
    mix_at: impl Fn(usize) -> Mix + Sync,
    mixer: &Mixer,
    q: impl Fn(f32) -> u8 + Sync,
) {
    let mut rows: Vec<(Vec<&mut [u8]>, Option<&mut [u8]>)> = Vec::new();
    let mut plate_rows: Vec<std::slice::ChunksMut<u8>> = plates.iter_mut().map(|p| p.chunks_mut(w)).collect();
    let mut under_rows = if under.is_empty() { None } else { Some(under.chunks_mut(w)) };
    for _ in 0..h {
        let row: Vec<&mut [u8]> = plate_rows.iter_mut().map(|it| it.next().unwrap()).collect();
        let u = under_rows.as_mut().and_then(|it| it.next());
        rows.push((row, u));
    }
    rows.into_par_iter().enumerate().for_each(|(y, (mut row, u))| {
        let mut u = u;
        for x in 0..w {
            let m = mix_at(y * w + x);
            for (k, r) in row.iter_mut().enumerate() {
                r[x] = q(mixer.coverage_at(&m, mixer.base + k));
            }
            if let Some(u) = u.as_mut() {
                u[x] = (mixer.coverage_at(&m, 1) * 255.0).round() as u8;
            }
        }
    });
}
