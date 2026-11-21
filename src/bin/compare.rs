//! 对比工具:读取一组索引色色版 TIFF(调色板 0 = 油墨色,255 = 白,线性过渡),
//! 按与分色程序相同的印刷模型(正常混合,按文件顺序从下往上叠)合成,和原图比色差,并统计每个像素的叠印层数。
//!
//! 用法:compare <原图> <色版1.tif> <色版2.tif> ... [--garment #FFFFFF] [--preview out.png]

#[allow(dead_code)]
#[path = "../color.rs"]
mod color;
#[allow(dead_code)]
#[path = "../imgops.rs"]
mod imgops;

use anyhow::{Context, Result, bail};
use color::{Rgb, dist2, from_u8, parse_hex, rgb_to_lab, to_hex, to_u8};
use rayon::prelude::*;

struct Plate {
    name: String,
    ink: Rgb,
    /// 覆盖率 0..255
    cov: Vec<u8>,
}

fn read_indexed_tiff(path: &str) -> Result<(usize, usize, Plate)> {
    let d = std::fs::read(path).with_context(|| format!("读取 {path}"))?;
    let le = &d[..2] == b"II";
    let u16_at = |o: usize| if le { u16::from_le_bytes([d[o], d[o + 1]]) } else { u16::from_be_bytes([d[o], d[o + 1]]) };
    let u32_at = |o: usize| {
        let b = [d[o], d[o + 1], d[o + 2], d[o + 3]];
        if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) }
    };
    let ifd = u32_at(4) as usize;
    let n = u16_at(ifd) as usize;
    let mut tags = std::collections::HashMap::new();
    for i in 0..n {
        let e = ifd + 2 + 12 * i;
        tags.insert(u16_at(e), (u16_at(e + 2), u32_at(e + 4) as usize, e + 8));
    }
    // 读取某个标签的全部数值(SHORT 或 LONG)
    let values = |tag: u16| -> Vec<usize> {
        let Some(&(ty, count, pos)) = tags.get(&tag) else { return vec![] };
        let size = if ty == 3 { 2 } else { 4 };
        let base = if size * count <= 4 { pos } else { u32_at(pos) as usize };
        (0..count).map(|k| if ty == 3 { u16_at(base + 2 * k) as usize } else { u32_at(base + 4 * k) as usize }).collect()
    };
    let w = values(256)[0];
    let h = values(257)[0];
    if values(262).first() != Some(&3) {
        bail!("{path} 不是索引色 TIFF");
    }
    if values(259).first() != Some(&1) {
        bail!("{path} 是压缩的 TIFF,暂只支持未压缩");
    }
    let cmap = values(320);
    let ink = from_u8([(cmap[0] >> 8) as u8, (cmap[256] >> 8) as u8, (cmap[512] >> 8) as u8]);
    let offsets = values(273);
    let counts = values(279);
    let mut idx = Vec::with_capacity(w * h);
    for (o, c) in offsets.iter().zip(&counts) {
        idx.extend_from_slice(&d[*o..*o + *c]);
    }
    idx.truncate(w * h);
    let cov = idx.iter().map(|v| 255 - v).collect();
    let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
    Ok((w, h, Plate { name, ink, cov }))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut garment = parse_hex("#FFFFFF")?;
    let mut preview_path = None;
    let mut files = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--garment" => garment = parse_hex(&it.next().unwrap())?,
            "--preview" => preview_path = it.next(),
            _ => files.push(a),
        }
    }
    if files.len() < 2 {
        bail!("用法:compare <原图> <色版.tif>... [--garment #FFFFFF] [--preview out.png]");
    }
    let orig = image::open(&files[0])?.to_rgb8();
    let (w, h) = (orig.width() as usize, orig.height() as usize);
    let plates: Vec<Plate> = files[1..]
        .par_iter()
        .map(|f| {
            let (pw, ph, p) = read_indexed_tiff(f)?;
            if (pw, ph) != (w, h) {
                bail!("{f} 尺寸 {pw}×{ph} 与原图 {w}×{h} 不一致");
            }
            Ok(p)
        })
        .collect::<Result<_>>()?;

    let n = w * h;
    println!("色版:");
    for p in &plates {
        let area = p.cov.iter().filter(|&&v| v > 10).count() as f32 / n as f32 * 100.0;
        let max = *p.cov.iter().max().unwrap() as f32 / 2.55;
        let rough = imgops::roughness(&p.cov, w, h);
        println!("  {:<6} {}  面积 {:>5.1}%  最大覆盖 {:>3.0}%  粗糙度 {rough:.2}", p.name, to_hex(p.ink), area, max);
    }

    let rough: f64 = plates.iter().map(|p| imgops::roughness(&p.cov, w, h)).sum::<f64>() / plates.len() as f64;
    println!("平均色版粗糙度 {rough:.2}(越小越干净)");
    let src: Vec<[u8; 3]> = orig.pixels().map(|p| p.0).collect();

    let preview: Vec<[u8; 3]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut c = garment;
            for p in &plates {
                let a = p.cov[i] as f32 / 255.0;
                for k in 0..3 {
                    c[k] = c[k] * (1.0 - a) + p.ink[k] * a;
                }
            }
            to_u8(c)
        })
        .collect();
    let de: f64 = src
        .par_iter()
        .zip(preview.par_iter())
        .map(|(a, b)| dist2(rgb_to_lab(from_u8(*a)), rgb_to_lab(from_u8(*b))).sqrt() as f64)
        .sum::<f64>()
        / n as f64;
    println!("按正常混合、文件顺序(下 → 上)合成,与原图平均色差 ΔE {de:.2}");

    // 叠印层数(覆盖率 > 4% 才算一层)
    let mut layers = vec![0usize; plates.len() + 1];
    let counts: Vec<usize> = (0..n).into_par_iter().map(|i| plates.iter().filter(|p| p.cov[i] > 10).count()).collect();
    for c in counts {
        layers[c] += 1;
    }
    println!("每个像素的叠印层数:");
    for (k, c) in layers.iter().enumerate() {
        if *c > 0 {
            println!("  {k} 层:{:>5.1}%", *c as f32 / n as f32 * 100.0);
        }
    }

    if let Some(p) = preview_path {
        image::RgbImage::from_fn(w as u32, h as u32, |x, y| image::Rgb(preview[y as usize * w + x as usize])).save(&p)?;
        println!("预览 → {p}");
    }
    Ok(())
}
