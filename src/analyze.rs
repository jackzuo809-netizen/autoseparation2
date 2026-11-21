//! 判断图的类型:平涂(色块分明)还是渐变(水彩 / 照片)。
//!
//! 三个指标(在缩小图上计算):
//! - 平坦像素比例:与右、下邻居色差都 < 4 的像素占比;
//! - 颜色集中度:聚成 16 色后,离最近中心 ΔE < 8 的像素占比;
//! - 色块纯度:离最近中心 ΔE < 3 的像素占比。
//! 平涂图三个指标都很高;渐变图(包括干净的数码水彩)颜色连续变化,色块纯度低。

use rayon::prelude::*;

use crate::color::{Lab, dist2, from_u8, rgb_to_lab};
use crate::palette::{kmeans, nearest};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Flat,
    Tone,
}

pub struct Stats {
    pub flat_ratio: f32,
    pub concentration: f32,
    /// 离最近中心 ΔE < 3 的像素占比(区分「真正的色块」和「平滑渐变」)
    pub tight: f32,
}

pub fn analyze(pixels: &[[u8; 3]], w: usize, h: usize) -> (Kind, Stats) {
    let s = (w.max(h) / 1000).max(1);
    let (sw, sh) = (w.div_ceil(s), h.div_ceil(s));
    let small: Vec<Lab> = (0..sw * sh)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i % sw) * s, (i / sw) * s);
            rgb_to_lab(from_u8(pixels[y * w + x]))
        })
        .collect();

    let flat = (0..sw * sh)
        .into_par_iter()
        .filter(|&i| {
            let (x, y) = (i % sw, i / sw);
            let ok_r = x + 1 >= sw || dist2(small[i], small[i + 1]) < 16.0;
            let ok_d = y + 1 >= sh || dist2(small[i], small[i + sw]) < 16.0;
            ok_r && ok_d
        })
        .count();
    let flat_ratio = flat as f32 / small.len() as f32;

    let samples: Vec<Lab> = small.iter().step_by((small.len() / 60_000).max(1)).copied().collect();
    let (_, centers) = kmeans(&samples, 16, 12);
    let close = samples.par_iter().filter(|s| nearest(&centers, **s).1 < 64.0).count();
    let concentration = close as f32 / samples.len() as f32;
    let tight = samples.par_iter().filter(|s| nearest(&centers, **s).1 < 9.0).count() as f32 / samples.len() as f32;

    // 平涂:大部分像素平坦、颜色集中,并且确实落在少数纯色上(数码水彩虽然平滑,
    // 但颜色是连续变化的,色块纯度低)
    let kind = if flat_ratio > 0.6 && concentration > 0.85 && tight > 0.85 { Kind::Flat } else { Kind::Tone };
    (kind, Stats { flat_ratio, concentration, tight })
}
