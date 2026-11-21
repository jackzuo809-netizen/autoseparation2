//! 平涂图分色:实地色版(每个像素要么 100% 出墨,要么不出)。
//!
//! 1. 只用「平坦区」像素聚类,自动决定图中有几种颜色(不受抗锯齿边缘干扰);
//! 2. 平坦像素归到最近的颜色;边缘像素只能在附近平坦区出现过的颜色里选,避免出现杂色细线;
//! 3. 去掉小于最小面积的碎块;
//! 4. 选油墨:实地版上层完全盖住下层,所以每种颜色对应一种油墨、白墨或不印(底色);
//! 5. 补漏白:浅色版向相邻深色区外扩。

use rayon::prelude::*;

use crate::color::{Lab, Rgb, dist2, from_u8, lightness, rgb_to_lab};
use crate::imgops::{dilate, remove_small_regions};
use crate::model::Separation;
use crate::palette::{Rep, choose_inks, kmeans, nearest, summarize};

pub struct FlatOpts {
    pub n: Option<usize>,
    pub fixed_inks: Option<Vec<Rgb>>,
    /// 小于此面积(像素)的色块并入周围
    pub min_area: usize,
    /// 补漏白外扩像素
    pub trap: usize,
}

/// 自动决定颜色数:聚类数从 2 往上加,直到 99.5% 的平坦区像素离所属中心 ΔE < 8
/// (用比例而不是平均值,面积小的颜色也不会被并掉)。
fn find_palette(rgb: &[Rgb], lab: &[Lab]) -> (Vec<Lab>, Vec<Rep>) {
    let mut result = None;
    for k in 2..=32 {
        let (assign, centers) = kmeans(lab, k, 15);
        let far = lab.par_iter().zip(assign.par_iter()).filter(|(s, a)| dist2(**s, centers[**a]) > 64.0).count();
        result = Some((assign, centers));
        if (far as f32) < lab.len() as f32 * 0.005 {
            break;
        }
    }
    let (assign, centers) = result.unwrap();
    let mut reps = summarize(rgb, &assign, centers.len());
    // 去掉面积极小的簇(噪点),合并色差 < 6 的簇
    reps.retain(|r| r.weight >= 0.0005);
    reps.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let mut merged: Vec<Rep> = Vec::new();
    for r in reps {
        match merged.iter_mut().find(|m| dist2(m.lab, r.lab) < 36.0) {
            Some(m) => m.weight += r.weight,
            None => merged.push(r),
        }
    }
    (merged.iter().map(|r| r.lab).collect(), merged)
}

pub fn separate(pixels: &[[u8; 3]], w: usize, h: usize, garment: Rgb, underbase: bool, opts: &FlatOpts) -> Separation {
    let n = w * h;
    let lab: Vec<Lab> = pixels.par_iter().map(|p| rgb_to_lab(from_u8(*p))).collect();

    // 边缘像素:与任一 4 邻居色差 > 6
    let edge: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let mut e = false;
            if x > 0 {
                e |= dist2(lab[i], lab[i - 1]) > 36.0;
            }
            if x + 1 < w {
                e |= dist2(lab[i], lab[i + 1]) > 36.0;
            }
            if y > 0 {
                e |= dist2(lab[i], lab[i - w]) > 36.0;
            }
            if y + 1 < h {
                e |= dist2(lab[i], lab[i + w]) > 36.0;
            }
            e
        })
        .collect();

    let flat_idx: Vec<usize> = (0..n).filter(|&i| !edge[i]).collect();
    let stride = (flat_idx.len() / 100_000).max(1);
    let s_rgb: Vec<Rgb> = flat_idx.iter().step_by(stride).map(|&i| from_u8(pixels[i])).collect();
    let s_lab: Vec<Lab> = flat_idx.iter().step_by(stride).map(|&i| lab[i]).collect();
    let (centers, mut palette) = find_palette(&s_rgb, &s_lab);
    println!("图中颜色:{} 种", centers.len());

    // 归类
    let mut labels: Vec<u16> = (0..n)
        .into_par_iter()
        .map(|i| if edge[i] { u16::MAX } else { nearest(&centers, lab[i]).0 as u16 })
        .collect();
    let base = labels.clone();
    labels.par_iter_mut().enumerate().for_each(|(i, l)| {
        if *l != u16::MAX {
            return;
        }
        let (x, y) = (i % w, i / w);
        let mut best = (u16::MAX, f32::MAX);
        for yy in y.saturating_sub(3)..=(y + 3).min(h - 1) {
            for xx in x.saturating_sub(3)..=(x + 3).min(w - 1) {
                let c = base[yy * w + xx];
                if c != u16::MAX {
                    let d = dist2(lab[i], centers[c as usize]);
                    if d < best.1 {
                        best = (c, d);
                    }
                }
            }
        }
        *l = if best.0 != u16::MAX { best.0 } else { nearest(&centers, lab[i]).0 as u16 };
    });

    // 去碎块
    let mut removed = 0;
    for _ in 0..2 {
        removed += remove_small_regions(&mut labels, w, h, opts.min_area);
    }
    println!("去碎斑:合并 {removed} 个小于 {} 像素的碎块", opts.min_area);

    // 按最终面积更新权重
    let mut counts = vec![0usize; palette.len()];
    for &l in &labels {
        counts[l as usize] += 1;
    }
    for (r, c) in palette.iter_mut().zip(&counts) {
        r.weight = *c as f32 / n as f32;
    }

    // 选油墨:候选为图中颜色(与底色接近的不用印)
    let g = rgb_to_lab(garment);
    let white = rgb_to_lab([1.0; 3]);
    let free = |lab: Lab| {
        let mut d = dist2(lab, g);
        if underbase {
            d = d.min(dist2(lab, white));
        }
        d
    };
    let mut inks = match &opts.fixed_inks {
        Some(v) => v.clone(),
        None => {
            let cands: Vec<Rgb> = palette.iter().filter(|r| free(r.lab) > 36.0).map(|r| r.rgb).collect();
            let err = |inks: &[Rgb], r: &Rep| {
                inks.iter().map(|i| dist2(rgb_to_lab(*i), r.lab)).fold(free(r.lab), f32::min).sqrt()
            };
            choose_inks(&palette, &cands, opts.n, cands.len(), err, |e, _| e.max < 6.0).0
        }
    };
    inks.sort_by(|a, b| lightness(*b).total_cmp(&lightness(*a)));
    let ink_labs: Vec<Lab> = inks.iter().map(|c| rgb_to_lab(*c)).collect();

    // 每种颜色的印法:None = 不印 / 白墨,Some(k) = 油墨 k
    let recipes: Vec<(Option<usize>, bool)> = palette
        .iter()
        .map(|r| {
            let best_ink = ink_labs.iter().enumerate().map(|(k, l)| (k, dist2(*l, r.lab))).min_by(|a, b| a.1.total_cmp(&b.1));
            let dg = dist2(r.lab, g);
            let dw = if underbase { dist2(r.lab, white) } else { f32::MAX };
            match best_ink {
                Some((k, d)) if d < dg.min(dw) => (Some(k), underbase),
                _ => (None, dw < dg),
            }
        })
        .collect();
    for (r, (ink, wb)) in palette.iter().zip(&recipes) {
        let how = match (ink, wb) {
            (Some(k), _) => format!("油墨 {}", k + 1),
            (None, true) => "白墨".to_string(),
            (None, false) => "不印(底色)".to_string(),
        };
        println!("  {} 面积 {:>5.1}% → {how}", crate::color::to_hex(r.rgb), r.weight * 100.0);
    }

    let mut plates = vec![vec![0u8; n]; inks.len()];
    let mut under = vec![0u8; n];
    for (idx, &l) in labels.iter().enumerate() {
        let (ink, wb) = recipes[l as usize];
        if wb {
            under[idx] = 255;
        }
        if let Some(k) = ink {
            plates[k][idx] = 255;
        }
    }

    // 补漏白:浅色版外扩到相邻的深色版下面(深色压住浅色,套印偏了也不露底)
    if opts.trap > 0 {
        for k in 0..inks.len() {
            let grown = dilate(&plates[k], w, h, opts.trap);
            let darker: Vec<usize> = ((k + 1)..inks.len()).collect();
            let (head, tail) = plates.split_at_mut(k + 1);
            let pk = &mut head[k];
            pk.par_iter_mut().enumerate().for_each(|(i, v)| {
                if *v == 0 && grown[i] > 0 && darker.iter().any(|&m| tail[m - k - 1][i] == 255) {
                    *v = 255;
                }
            });
        }
    }

    Separation { inks, plates, under: underbase.then_some(under) }
}
