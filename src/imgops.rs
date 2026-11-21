//! 图像工具:最小/最大值滤波、去小连通域、中值去颗粒、色版粗糙度。

use rayon::prelude::*;

fn filter_1d(src: &[u8], dst: &mut [u8], w: usize, h: usize, r: usize, horizontal: bool, take_max: bool) {
    let pick = |a: u8, b: u8| if take_max { a.max(b) } else { a.min(b) };
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut m = src[y * w + x];
            if horizontal {
                for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                    m = pick(m, src[y * w + xx]);
                }
            } else {
                for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    m = pick(m, src[yy * w + x]);
                }
            }
            row[x] = m;
        }
    });
}

/// 方形窗口最小值(腐蚀,r 像素)。
pub fn erode(plane: &mut [u8], w: usize, h: usize, r: usize) {
    if r == 0 {
        return;
    }
    let mut tmp = vec![0u8; plane.len()];
    filter_1d(plane, &mut tmp, w, h, r, true, false);
    filter_1d(&tmp, plane, w, h, r, false, false);
}

/// 方形窗口最大值(膨胀,r 像素)。
pub fn dilate(plane: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let mut tmp = vec![0u8; plane.len()];
    let mut out = vec![0u8; plane.len()];
    filter_1d(plane, &mut tmp, w, h, r, true, true);
    filter_1d(&tmp, &mut out, w, h, r, false, true);
    out
}

/// 把面积小于 `min_area` 的连通域(4 邻接)并入周围最多的标签。返回被合并的连通域数。
pub fn remove_small_regions(labels: &mut [u16], w: usize, h: usize, min_area: usize) -> usize {
    let n = labels.len();
    let mut seen = vec![false; n];
    let mut stack = Vec::new();
    let mut region = Vec::new();
    let mut merged = 0;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let l = labels[start];
        region.clear();
        stack.push(start);
        seen[start] = true;
        let mut neigh: Vec<(u16, u32)> = Vec::new();
        while let Some(p) = stack.pop() {
            region.push(p);
            let (x, y) = (p % w, p / w);
            let mut visit = |q: usize| {
                if labels[q] == l {
                    if !seen[q] {
                        seen[q] = true;
                        stack.push(q);
                    }
                } else {
                    match neigh.iter_mut().find(|v| v.0 == labels[q]) {
                        Some(v) => v.1 += 1,
                        None => neigh.push((labels[q], 1)),
                    }
                }
            };
            if x > 0 {
                visit(p - 1);
            }
            if x + 1 < w {
                visit(p + 1);
            }
            if y > 0 {
                visit(p - w);
            }
            if y + 1 < h {
                visit(p + w);
            }
        }
        if region.len() < min_area {
            if let Some(&(nl, _)) = neigh.iter().max_by_key(|v| v.1) {
                for &p in &region {
                    labels[p] = nl;
                }
                merged += 1;
            }
        }
    }
    merged
}

/// RGB 每通道 3×3 中值滤波,重复 `passes` 次(去掉扫描颗粒)。
pub fn median3(pixels: &[[u8; 3]], w: usize, h: usize, passes: usize) -> Vec<[u8; 3]> {
    let mut cur = pixels.to_vec();
    for _ in 0..passes {
        let src = cur.clone();
        cur.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let mut win = [[0u8; 9]; 3];
            for x in 0..w {
                let mut n = 0;
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let p = src[yy * w + xx];
                        for c in 0..3 {
                            win[c][n] = p[c];
                        }
                        n += 1;
                    }
                }
                for c in 0..3 {
                    let s = &mut win[c][..n];
                    s.sort_unstable();
                    row[x][c] = s[n / 2];
                }
            }
        });
    }
    cur
}

/// 色版粗糙度:相邻像素(右、下)网点值的平均跳变,按 0–100% 计。越小越干净。
pub fn roughness(plane: &[u8], w: usize, h: usize) -> f64 {
    let sum: u64 = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut s = 0u64;
            for x in 0..w {
                let v = plane[y * w + x] as i32;
                if x + 1 < w {
                    s += (v - plane[y * w + x + 1] as i32).unsigned_abs() as u64;
                }
                if y + 1 < h {
                    s += (v - plane[(y + 1) * w + x] as i32).unsigned_abs() as u64;
                }
            }
            s
        })
        .sum();
    sum as f64 / (w * h) as f64 / 255.0 * 100.0
}
