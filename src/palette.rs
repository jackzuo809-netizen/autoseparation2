//! 颜色聚类与油墨选择。
//!
//! 选油墨:从候选色中贪心逐个加入,使还原目标色的加权误差最小;
//! 指定了 N 就选 N 个,否则加到误差达标为止;最后做交换优化。

use rayon::prelude::*;

use crate::color::{Lab, Rgb, dist2, rgb_to_lab};

/// 一个目标色(聚类中心),weight 为面积占比。
#[derive(Clone)]
pub struct Rep {
    pub rgb: Rgb,
    pub lab: Lab,
    pub weight: f32,
}

/// Lab K-means(k-means++ 初始化,确定性)。返回每个样本的归属和中心。
pub fn kmeans(samples: &[Lab], k: usize, iters: usize) -> (Vec<usize>, Vec<Lab>) {
    let k = k.min(samples.len()).max(1);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rand = move || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut centers: Vec<Lab> = vec![samples[samples.len() / 2]];
    let mut d: Vec<f32> = samples.iter().map(|s| dist2(*s, centers[0])).collect();
    while centers.len() < k {
        let total: f64 = d.iter().map(|&v| v as f64).sum();
        if total <= 0.0 {
            break;
        }
        let mut target = rand() * total;
        let mut idx = samples.len() - 1;
        for (n, &v) in d.iter().enumerate() {
            target -= v as f64;
            if target <= 0.0 {
                idx = n;
                break;
            }
        }
        let c = samples[idx];
        centers.push(c);
        d.par_iter_mut().zip(samples.par_iter()).for_each(|(dv, s)| *dv = dv.min(dist2(*s, c)));
    }

    let k = centers.len();
    let mut assign = vec![0usize; samples.len()];
    for _ in 0..iters {
        assign.par_iter_mut().zip(samples.par_iter()).for_each(|(a, s)| *a = nearest(&centers, *s).0);
        let mut sums = vec![[0f64; 4]; k];
        for (s, &a) in samples.iter().zip(&assign) {
            for c in 0..3 {
                sums[a][c] += s[c] as f64;
            }
            sums[a][3] += 1.0;
        }
        for (n, s) in sums.iter().enumerate() {
            if s[3] > 0.0 {
                centers[n] = [(s[0] / s[3]) as f32, (s[1] / s[3]) as f32, (s[2] / s[3]) as f32];
            }
        }
    }
    assign.par_iter_mut().zip(samples.par_iter()).for_each(|(a, s)| *a = nearest(&centers, *s).0);
    (assign, centers)
}

#[inline]
pub fn nearest(centers: &[Lab], s: Lab) -> (usize, f32) {
    let mut best = (0, f32::MAX);
    for (n, c) in centers.iter().enumerate() {
        let d = dist2(s, *c);
        if d < best.1 {
            best = (n, d);
        }
    }
    best
}

/// 把聚类结果汇总成目标色(RGB 取成员平均,避免 Lab→RGB 反算)。
pub fn summarize(rgb: &[Rgb], assign: &[usize], k: usize) -> Vec<Rep> {
    let mut acc = vec![[0f64; 4]; k];
    for (c, &a) in rgb.iter().zip(assign) {
        for i in 0..3 {
            acc[a][i] += c[i] as f64;
        }
        acc[a][3] += 1.0;
    }
    let total = rgb.len() as f32;
    acc.iter()
        .filter(|s| s[3] > 0.0)
        .map(|s| {
            let c = [(s[0] / s[3]) as f32, (s[1] / s[3]) as f32, (s[2] / s[3]) as f32];
            Rep { rgb: c, lab: rgb_to_lab(c), weight: s[3] as f32 / total }
        })
        .collect()
}

/// 油墨组合的还原误差。
#[derive(Clone, Copy, Debug)]
pub struct Eval {
    /// 加权平均 ΔE
    pub mean: f32,
    /// 面积 ≥ 0.2% 的目标色中的最大 ΔE
    pub max: f32,
}

/// `err(inks, rep)`:用这组油墨还原 rep 的 ΔE。
fn evaluate<F: Fn(&[Rgb], &Rep) -> f32>(inks: &[Rgb], reps: &[Rep], err: &F) -> Eval {
    let mut mean = 0.0;
    let mut max = 0.0f32;
    for r in reps {
        let e = err(inks, r);
        mean += r.weight * e;
        if r.weight >= 0.002 {
            max = max.max(e);
        }
    }
    Eval { mean, max }
}

/// 贪心选油墨。`n = None` 时自动决定数量:加到 `done(eval, 上一步)` 为真或到 `max_n`。
pub fn choose_inks<F, D>(reps: &[Rep], cands: &[Rgb], n: Option<usize>, max_n: usize, err: F, done: D) -> (Vec<Rgb>, Eval)
where
    F: Fn(&[Rgb], &Rep) -> f32 + Sync,
    D: Fn(&Eval, Option<&Eval>) -> bool,
{
    let limit = n.unwrap_or(max_n).min(cands.len());
    let mut chosen: Vec<Rgb> = Vec::new();
    let mut eval = evaluate(&chosen, reps, &err);
    let mut prev: Option<Eval> = None;
    println!("  0 色:平均 ΔE {:.2},最大 ΔE {:.1}", eval.mean, eval.max);
    while chosen.len() < limit {
        if n.is_none() && done(&eval, prev.as_ref()) {
            break;
        }
        let best = cands
            .par_iter()
            .filter(|c| !chosen.contains(c))
            .map(|c| {
                let mut set = chosen.clone();
                set.push(*c);
                (evaluate(&set, reps, &err), *c)
            })
            .min_by(|a, b| a.0.mean.total_cmp(&b.0.mean))
            .unwrap();
        chosen.push(best.1);
        prev = Some(eval);
        eval = best.0;
        println!("  {} 色:平均 ΔE {:.2},最大 ΔE {:.1}", chosen.len(), eval.mean, eval.max);
    }

    // 交换优化
    for _ in 0..3 {
        let mut changed = false;
        for slot in 0..chosen.len() {
            let best = cands
                .par_iter()
                .filter(|c| !chosen.contains(c))
                .map(|c| {
                    let mut set = chosen.clone();
                    set[slot] = *c;
                    (evaluate(&set, reps, &err), *c)
                })
                .min_by(|a, b| a.0.mean.total_cmp(&b.0.mean));
            if let Some((e, c)) = best {
                if e.mean < eval.mean - 1e-4 {
                    eval = e;
                    chosen[slot] = c;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (chosen, eval)
}
