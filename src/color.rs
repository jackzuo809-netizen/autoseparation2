//! 颜色工具:sRGB(0..1)↔ Lab、色差、十六进制解析。

use anyhow::{Result, bail};

pub type Rgb = [f32; 3];
pub type Lab = [f32; 3];

fn to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn rgb_to_lab(c: Rgb) -> Lab {
    let (r, g, b) = (to_linear(c[0]), to_linear(c[1]), to_linear(c[2]));
    let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / 0.950_47;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b) / 1.088_83;
    let f = |t: f32| {
        if t > 0.008_856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Lab 欧氏距离平方(ΔE76²)。
#[inline]
pub fn dist2(a: Lab, b: Lab) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

pub fn from_u8(p: [u8; 3]) -> Rgb {
    [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]
}

pub fn to_u8(c: Rgb) -> [u8; 3] {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

pub fn parse_hex(s: &str) -> Result<Rgb> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 {
        bail!("颜色格式应为 #RRGGBB: {s}");
    }
    let v = u32::from_str_radix(h, 16)?;
    Ok(from_u8([(v >> 16) as u8, (v >> 8) as u8, v as u8]))
}

pub fn to_hex(c: Rgb) -> String {
    let [r, g, b] = to_u8(c);
    format!("#{r:02X}{g:02X}{b:02X}")
}

pub fn lightness(c: Rgb) -> f32 {
    rgb_to_lab(c)[0]
}
