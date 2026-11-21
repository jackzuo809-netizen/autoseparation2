//! 求每个颜色由哪些油墨、各占多少权重叠出来(正常混合模型)。
//!
//! 叠出的颜色是「底色 / 白墨打底 / 各油墨」的凸组合,所以问题就是:
//!   min ‖P·w − t‖²   s.t. w ≥ 0, Σw = 1
//! 用 Lawson–Hanson 非负最小二乘(格拉姆形式)求解,和为 1 用加权的附加行近似。
//! 可选加上空间一致性项 μ‖w − w̄‖²,让一片区域用同样的油墨和深浅。
//! RGB 三维 + 和为 1 共 4 个方程,基本解最多 4 个非零权重,
//! 所以最多 4 层叠印是天然满足的;层数上限更小时逐个去掉最小权重的油墨再解。

use crate::color::Rgb;

/// 点的最大数量:底色 + 白墨 + 16 种油墨。
pub const MAXP: usize = 18;
pub const SUM_WEIGHT: f64 = 6.0;

/// 一个颜色的混合权重,下标对应 Mixer 的点(0 = 底色,[1 = 白墨],之后是油墨)。
#[derive(Clone, Copy)]
pub struct Mix {
    pub w: [f32; MAXP],
}

pub struct Mixer {
    pub pts: Vec<Rgb>,
    /// 非油墨点的个数(底色 + 可选白墨)
    pub base: usize,
    pub max_layers: usize,
}

type Mat = [[f64; MAXP]; MAXP];

/// 解 k×k 线性方程组(高斯消元,部分主元,带微小岭值)。
fn solve_linear(k: usize, mut a: Mat, mut b: [f64; MAXP]) -> [f64; MAXP] {
    for i in 0..k {
        a[i][i] += 1e-12;
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-15 {
            continue;
        }
        for r in (col + 1)..k {
            let f = a[r][col] / d;
            if f != 0.0 {
                for c in col..k {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = [0.0; MAXP];
    for i in (0..k).rev() {
        let mut s = b[i];
        for j in (i + 1)..k {
            s -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() < 1e-15 { 0.0 } else { s / a[i][i] };
    }
    x
}

/// Lawson–Hanson NNLS 的格拉姆形式:min ½xᵀGx − cᵀx, x ≥ 0(G = AᵀA + μI, c = Aᵀb + μw̄)。
fn nnls_gram(n: usize, g: &Mat, c: &[f64; MAXP]) -> [f64; MAXP] {
    let tol = 1e-10;
    let mut x = [0.0f64; MAXP];
    let mut passive = [false; MAXP];
    let grad = |x: &[f64; MAXP], j: usize| c[j] - (0..n).map(|i| g[j][i] * x[i]).sum::<f64>();
    let ls = |passive: &[bool; MAXP]| -> [f64; MAXP] {
        let mut idx = [0usize; MAXP];
        let mut k = 0;
        for j in 0..n {
            if passive[j] {
                idx[k] = j;
                k += 1;
            }
        }
        let mut a: Mat = [[0.0; MAXP]; MAXP];
        let mut b = [0.0; MAXP];
        for r in 0..k {
            for q in 0..k {
                a[r][q] = g[idx[r]][idx[q]];
            }
            b[r] = c[idx[r]];
        }
        let z = solve_linear(k, a, b);
        let mut out = [0.0; MAXP];
        for r in 0..k {
            out[idx[r]] = z[r];
        }
        out
    };
    for _ in 0..(3 * n) {
        let mut best = (usize::MAX, tol);
        for j in 0..n {
            if !passive[j] {
                let gj = grad(&x, j);
                if gj > best.1 {
                    best = (j, gj);
                }
            }
        }
        if best.0 == usize::MAX {
            break;
        }
        passive[best.0] = true;
        loop {
            let z = ls(&passive);
            if (0..n).all(|j| !passive[j] || z[j] > tol) {
                x = z;
                break;
            }
            let mut alpha = f64::MAX;
            for j in 0..n {
                if passive[j] && z[j] <= tol {
                    alpha = alpha.min(x[j] / (x[j] - z[j]));
                }
            }
            for j in 0..n {
                if passive[j] {
                    x[j] += alpha * (z[j] - x[j]);
                    if x[j] <= tol {
                        passive[j] = false;
                        x[j] = 0.0;
                    }
                }
            }
            if !passive.iter().any(|&p| p) {
                break;
            }
        }
    }
    x
}

impl Mixer {
    pub fn new(garment: Rgb, underbase: bool, inks: &[Rgb], max_layers: usize) -> Self {
        let mut pts = vec![garment];
        if underbase {
            pts.push([1.0; 3]);
        }
        let base = pts.len();
        pts.extend_from_slice(inks);
        assert!(pts.len() <= MAXP, "油墨最多 {} 种", MAXP - 2);
        Mixer { pts, base, max_layers }
    }

    /// 求混合权重。`allowed`:允许使用的油墨位掩码(底色 / 白墨总是允许)。
    pub fn solve(&self, t: Rgb, allowed: u32) -> Mix {
        self.solve_reg(t, allowed, None)
    }

    /// 带空间一致性约束的求解:额外最小化 μ‖w − w̄‖²,让权重靠近邻域参考值 w̄。
    pub fn solve_reg(&self, t: Rgb, allowed: u32, prior: Option<(&[f32; MAXP], f32)>) -> Mix {
        let s2 = SUM_WEIGHT * SUM_WEIGHT;
        let mut allowed = allowed;
        loop {
            let mut idx = [0usize; MAXP];
            let mut k = 0;
            for p in 0..self.pts.len() {
                if p < self.base || allowed & (1 << (p - self.base)) != 0 {
                    idx[k] = p;
                    k += 1;
                }
            }
            let mut g: Mat = [[0.0; MAXP]; MAXP];
            let mut c = [0.0f64; MAXP];
            for a in 0..k {
                let pa = self.pts[idx[a]];
                for b in 0..k {
                    let pb = self.pts[idx[b]];
                    g[a][b] = (pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]) as f64 + s2;
                }
                c[a] = (pa[0] * t[0] + pa[1] * t[1] + pa[2] * t[2]) as f64 + s2;
                if let Some((wbar, mu)) = prior {
                    g[a][a] += mu as f64;
                    c[a] += (mu * wbar[idx[a]]) as f64;
                }
            }
            let x = nnls_gram(k, &g, &c);
            let sum: f64 = x[..k].iter().sum();
            let mut m = Mix { w: [0.0; MAXP] };
            for a in 0..k {
                m.w[idx[a]] = if sum > 0.0 { (x[a] / sum) as f32 } else { 0.0 };
            }
            if sum <= 0.0 {
                m.w[0] = 1.0;
            }
            // 层数超限:只保留权重最大的 max_layers 种油墨重解
            let mut used: Vec<(usize, f32)> = (self.base..self.pts.len()).filter(|&p| m.w[p] > 1e-4).map(|p| (p, m.w[p])).collect();
            if used.len() <= self.max_layers {
                return m;
            }
            used.sort_by(|a, b| b.1.total_cmp(&a.1));
            allowed = used[..self.max_layers].iter().fold(0, |acc, &(p, _)| acc | (1 << (p - self.base)));
        }
    }

    pub fn color(&self, m: &Mix) -> Rgb {
        let mut c = [0.0f32; 3];
        for (p, pt) in self.pts.iter().enumerate() {
            for k in 0..3 {
                c[k] += m.w[p] * pt[k];
            }
        }
        c
    }

    /// 第 `p` 个点(白墨或油墨)所在图层的覆盖率:本层权重 / (本层及以下权重之和)。
    pub fn coverage_at(&self, m: &Mix, p: usize) -> f32 {
        let below: f32 = m.w[..=p].iter().sum();
        if below > 1e-6 { (m.w[p] / below).clamp(0.0, 1.0) } else { 0.0 }
    }
}
