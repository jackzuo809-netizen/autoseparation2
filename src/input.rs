//! 读取输入图:RGB 图交给 image 库;CMYK TIFF 用 ICC 色彩管理转成 sRGB。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use lcms2::{Flags, Intent, PixelFormat, Profile, Transform};
use tiff::decoder::{Decoder, DecodingResult, ifd::Value};
use tiff::tags::Tag;

pub struct Input {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<[u8; 3]>,
    pub dpi: Option<u32>,
}

/// 文件没有内嵌 ICC 时依次尝试的 CMYK 配置文件(Photoshop 中文版默认 Japan Color 2001 Coated)。
const DEFAULT_CMYK_PROFILES: &[&str] = &[
    "/Library/Application Support/Adobe/Color/Profiles/Recommended/JapanColor2001Coated.icc",
    "/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc",
    "C:\\Program Files (x86)\\Common Files\\Adobe\\Color\\Profiles\\Recommended\\JapanColor2001Coated.icc",
    "C:\\Windows\\System32\\spool\\drivers\\color\\JapanColor2001Coated.icc",
];

pub fn load(path: &Path, cmyk_profile: Option<&Path>) -> Result<Input> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "tif" || ext == "tiff" {
        if let Some(input) = load_cmyk_tiff(path, cmyk_profile)? {
            return Ok(input);
        }
    }
    let img = image::open(path)
        .with_context(|| format!("无法读取 {}", path.display()))?
        .to_rgb8();
    Ok(Input {
        w: img.width() as usize,
        h: img.height() as usize,
        pixels: img.pixels().map(|p| p.0).collect(),
        dpi: None,
    })
}

fn tiff_dpi<R: std::io::Read + std::io::Seek>(dec: &mut Decoder<R>) -> Option<u32> {
    let res = match dec.find_tag(Tag::XResolution).ok()?? {
        Value::Rational(n, d) if d > 0 => n as f64 / d as f64,
        _ => return None,
    };
    let unit = dec.find_tag(Tag::ResolutionUnit).ok().flatten().and_then(|v| v.into_u16().ok()).unwrap_or(2);
    let dpi = if unit == 3 { res * 2.54 } else { res };
    (dpi > 0.0).then(|| dpi.round() as u32)
}

/// 不是 CMYK TIFF 时返回 Ok(None)。
fn load_cmyk_tiff(path: &Path, cmyk_profile: Option<&Path>) -> Result<Option<Input>> {
    let file = std::fs::File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
    let mut dec = Decoder::new(std::io::BufReader::new(file))?;
    let channels = match dec.colortype()? {
        tiff::ColorType::CMYK(8) => 4,
        tiff::ColorType::CMYKA(8) => 5,
        tiff::ColorType::CMYK(_) | tiff::ColorType::CMYKA(_) => bail!("暂只支持 8 位 CMYK TIFF"),
        _ => return Ok(None),
    };
    let (w, h) = dec.dimensions()?;
    let (w, h) = (w as usize, h as usize);
    let dpi = tiff_dpi(&mut dec);
    let embedded = dec.get_tag_u8_vec(Tag::IccProfile).ok();
    let raw = match dec.read_image()? {
        DecodingResult::U8(v) => v,
        _ => bail!("CMYK TIFF 数据格式不支持"),
    };
    let cmyk: Vec<[u8; 4]> = raw.chunks_exact(channels).map(|c| [c[0], c[1], c[2], c[3]]).collect();

    let (profile, source) = if let Some(icc) = embedded {
        (Profile::new_icc(&icc)?, "文件内嵌".to_string())
    } else {
        let candidates: Vec<PathBuf> = cmyk_profile
            .map(|p| vec![p.to_path_buf()])
            .unwrap_or_else(|| DEFAULT_CMYK_PROFILES.iter().map(PathBuf::from).collect());
        match candidates.iter().find(|p| p.exists()) {
            Some(p) => (Profile::new_file(p)?, p.display().to_string()),
            None => {
                if cmyk_profile.is_some() {
                    bail!("找不到 CMYK 配置文件 {}", cmyk_profile.unwrap().display());
                }
                eprintln!("警告:找不到 CMYK 配置文件,使用简单公式换算,颜色可能偏差");
                let pixels = cmyk
                    .iter()
                    .map(|c| {
                        let k = 255 - c[3] as u32;
                        [0, 1, 2].map(|i| ((255 - c[i] as u32) * k / 255) as u8)
                    })
                    .collect();
                return Ok(Some(Input { w, h, pixels, dpi }));
            }
        }
    };
    println!("CMYK → sRGB(配置文件:{source})");
    let t = Transform::new_flags(
        &profile,
        PixelFormat::CMYK_8,
        &Profile::new_srgb(),
        PixelFormat::RGB_8,
        Intent::RelativeColorimetric,
        Flags::BLACKPOINT_COMPENSATION,
    )?;
    let mut pixels = vec![[0u8; 3]; cmyk.len()];
    t.transform_pixels(&cmyk, &mut pixels);
    Ok(Some(Input { w, h, pixels, dpi }))
}
