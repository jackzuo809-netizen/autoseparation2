mod analyze;
mod color;
mod flat;
mod gpu;
mod imgops;
mod input;
mod model;
mod palette;
mod psd;
mod mix;
mod tiffout;
mod tone;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use rayon::prelude::*;

use analyze::Kind;
use color::{Rgb, from_u8, lightness, parse_hex, rgb_to_lab, to_hex, to_u8};

#[derive(Clone, Copy, ValueEnum)]
enum Underbase {
    /// 深色底自动打白底
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum GpuMode {
    /// 有显卡就用
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// 自动判断
    Auto,
    /// 平涂图:实地色版
    Flat,
    /// 渐变图:连续调色版
    Tone,
}

/// 丝网印刷自动分色(专色 + 最多两色叠印 + 白墨打底)
#[derive(Parser)]
struct Args {
    /// 输入图片(PNG / JPG / RGB 或 CMYK TIFF)
    input: PathBuf,
    /// 油墨数量(不含白墨打底);不填则自动决定
    #[arg(short = 'n', long)]
    inks: Option<usize>,
    /// 手动指定油墨色,逗号分隔,如 #D7263D,#1B4F9C
    #[arg(long, value_delimiter = ',')]
    ink_colors: Vec<String>,
    /// 分色模式
    #[arg(long, value_enum, default_value = "auto")]
    mode: Mode,
    /// 衣服 / 承印物颜色
    #[arg(long, default_value = "#FFFFFF")]
    garment: String,
    /// 白墨打底
    #[arg(long, value_enum, default_value = "auto")]
    underbase: Underbase,
    /// 白墨打底内缩(毫米)
    #[arg(long, default_value_t = 0.1)]
    choke_mm: f32,
    /// 平涂:补漏白外扩(毫米)
    #[arg(long, default_value_t = 0.1)]
    trap_mm: f32,
    /// 平涂:小于此边长(毫米)的碎块会被合并
    #[arg(long, default_value_t = 0.3)]
    min_mm: f32,
    /// 渐变:每个像素最多叠几层油墨(1–4)
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(1..=4))]
    max_layers: u8,
    /// 渐变:空间一致性强度(越大色版越成片、平滑,颜色略有损失),0 关闭
    #[arg(long, default_value_t = 0.03)]
    coherence: f32,
    /// 渐变:一致性的作用范围(毫米),越大越能让整块区域归同一个油墨
    #[arg(long, default_value_t = 10.0)]
    coherence_mm: f32,
    /// 渐变:分色前去颗粒(3×3 中值滤波次数),0 关闭
    #[arg(long, default_value_t = 1)]
    denoise: usize,
    /// 渐变:最小网点百分比,低于此值清零
    #[arg(long, default_value_t = 4.0)]
    min_dot: f32,
    /// 输出分辨率 DPI(不填则用原图的,原图没有则 300)
    #[arg(long)]
    dpi: Option<u32>,
    /// 显卡加速(渐变图的逐像素求解)。实测多核 CPU 更快,默认关闭
    #[arg(long, value_enum, default_value = "off")]
    gpu: GpuMode,
    /// CMYK 原图无内嵌 ICC 时使用的配置文件
    #[arg(long)]
    cmyk_profile: Option<PathBuf>,
    /// 输出目录
    #[arg(short, long, default_value = "out")]
    out: PathBuf,
}

fn mm_to_px(mm: f32, dpi: u32) -> usize {
    (mm * dpi as f32 / 25.4).round() as usize
}

fn mean_delta_e(a: &[[u8; 3]], b: &[[u8; 3]]) -> f64 {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| color::dist2(rgb_to_lab(from_u8(*x)), rgb_to_lab(from_u8(*y))).sqrt() as f64)
        .sum::<f64>()
        / a.len() as f64
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    let img = input::load(&args.input, args.cmyk_profile.as_deref())?;
    let (w, h) = (img.w, img.h);
    let dpi = args.dpi.or(img.dpi).unwrap_or(300);
    println!("读取 {w}×{h} @ {dpi} dpi");

    let garment = parse_hex(&args.garment)?;
    let underbase = match args.underbase {
        Underbase::On => true,
        Underbase::Off => false,
        Underbase::Auto => lightness(garment) < 60.0,
    };
    let fixed_inks: Option<Vec<Rgb>> = if args.ink_colors.is_empty() {
        None
    } else {
        Some(args.ink_colors.iter().map(|s| parse_hex(s)).collect::<Result<_>>()?)
    };

    let (kind, stats) = analyze::analyze(&img.pixels, w, h);
    let kind = match args.mode {
        Mode::Auto => kind,
        Mode::Flat => Kind::Flat,
        Mode::Tone => Kind::Tone,
    };
    println!(
        "图类型:{}(平坦像素 {:.0}%,颜色集中度 {:.0}%,色块纯度 {:.0}%)",
        if kind == Kind::Flat { "平涂 → 实地色版" } else { "渐变 → 连续调色版" },
        stats.flat_ratio * 100.0,
        stats.concentration * 100.0,
        stats.tight * 100.0
    );

    // 渐变图先去颗粒;分色误差以送进分色的图为准
    let src = if kind == Kind::Tone && args.denoise > 0 {
        let d = imgops::median3(&img.pixels, w, h, args.denoise);
        println!("去颗粒:与原图平均差 ΔE {:.2}", mean_delta_e(&d, &img.pixels));
        d
    } else {
        img.pixels.clone()
    };
    let mut sep = match kind {
        Kind::Flat => {
            let side = mm_to_px(args.min_mm, dpi).max(2);
            let opts = flat::FlatOpts { n: args.inks, fixed_inks, min_area: side * side, trap: mm_to_px(args.trap_mm, dpi) };
            flat::separate(&src, w, h, garment, underbase, &opts)
        }
        Kind::Tone => {
            let opts = tone::ToneOpts {
                n: args.inks,
                fixed_inks,
                max_layers: args.max_layers as usize,
                coherence: args.coherence,
                block: mm_to_px(0.4, dpi).max(2),
                radius: (args.coherence_mm / 0.4).round().max(1.0) as usize,
                iters: 2,
                hidden_radius: mm_to_px(0.5, dpi).max(1),
                min_dot: (args.min_dot / 100.0 * 255.0).round() as u8,
            };
            let gpu = if args.gpu == GpuMode::Off || opts.coherence <= 0.0 {
                None
            } else {
                let g = gpu::Gpu::new();
                match &g {
                    Some(g) => println!("显卡加速:{}", g.name),
                    None if args.gpu == GpuMode::On => anyhow::bail!("指定了 --gpu on,但没有可用的显卡"),
                    None => println!("没有可用的显卡,用 CPU 计算"),
                }
                g
            };
            tone::separate(&src, w, h, garment, underbase, &opts, gpu.as_ref())
        }
    };
    if let Some(u) = sep.under.as_mut() {
        imgops::erode(u, w, h, mm_to_px(args.choke_mm, dpi));
    }
    println!("油墨 {} 色({:.1}s):", sep.inks.len(), t0.elapsed().as_secs_f32());
    for (k, c) in sep.inks.iter().enumerate() {
        println!("  {:02} {}", k + 1, to_hex(*c));
    }

    let n = w * h;
    let preview = sep.preview(garment, n);
    let mean_de = mean_delta_e(&preview, &src);

    // 输出
    std::fs::create_dir_all(&args.out)?;
    let stem = args.input.file_stem().unwrap().to_string_lossy().to_string();
    // 清掉上次运行留下的同名色版(「名称-1.tif」…或旧版「名称_NN xxx.tif」),免得新旧混在一起
    for entry in std::fs::read_dir(&args.out)?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(body) = name.strip_suffix(".tif") else { continue };
        let new_style = body.strip_prefix(&format!("{stem}-")).is_some_and(|r| r.len() == 1 && r.chars().all(|c| c.is_ascii_alphanumeric()));
        let old_style = body.strip_prefix(&format!("{stem}_")).is_some_and(|r| {
            let b = r.as_bytes();
            b.len() > 3 && b[0].is_ascii_digit() && b[1].is_ascii_digit() && b[2] == b' '
        });
        if new_style || old_style {
            std::fs::remove_file(entry.path())?;
        }
    }

    // 色版列表(按印刷顺序,下 → 上):白墨打底在最下
    let mut plates: Vec<(String, [u8; 3], &Vec<u8>)> = Vec::new();
    if let Some(u) = &sep.under {
        plates.push(("白墨打底".to_string(), [255; 3], u));
    }
    for (k, ink) in sep.inks.iter().enumerate() {
        plates.push((to_hex(*ink), to_u8(*ink), &sep.plates[k]));
    }

    // 分层 PSD:每版一个「正常」图层,不透明度 = 网点覆盖率
    let mut layers = vec![psd::Layer {
        name: format!("底色 {}", to_hex(garment)),
        color: to_u8(garment),
        alpha: vec![255; n],
        blend: *b"norm",
    }];
    for (i, (label, color, plane)) in plates.iter().enumerate() {
        layers.push(psd::Layer {
            name: format!("{}-{} {label}", stem, tiffout::plate_suffix(i)),
            color: *color,
            alpha: (*plane).clone(),
            blend: *b"norm",
        });
    }
    psd::write_psd(&args.out.join(format!("{stem}_分色.psd")), w, h, dpi, &layers, &preview)?;
    drop(layers);

    image::RgbImage::from_fn(w as u32, h as u32, |x, y| image::Rgb(preview[y as usize * w + x as usize]))
        .save(args.out.join(format!("{stem}_预览.png")))?;

    // 每版一张索引色 TIFF:名称-1.tif … 名称-9.tif、名称-a.tif …
    let files: Vec<String> = (0..plates.len()).map(|i| format!("{stem}-{}.tif", tiffout::plate_suffix(i))).collect();
    plates.par_iter().zip(files.par_iter()).try_for_each(|((_, color, plane), file)| {
        tiffout::write_indexed_plate(&args.out.join(file), w, h, dpi, *color, plane)
    })?;

    let mut json = format!(
        "{{\n  \"mode\": \"{}\",\n  \"garment\": \"{}\",\n  \"underbase\": {},\n  \"dpi\": {dpi},\n  \"mean_delta_e\": {mean_de:.2},\n  \"plates\": [\n",
        if kind == Kind::Flat { "flat" } else { "tone" },
        to_hex(garment),
        sep.under.is_some(),
    );
    for (k, ((label, color, plane), file)) in plates.iter().zip(&files).enumerate() {
        let area = plane.iter().filter(|&&v| v > 0).count() as f32 / n as f32 * 100.0;
        let c = format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2]);
        json += &format!(
            "    {{ \"order\": {}, \"file\": \"{file}\", \"name\": \"{label}\", \"color\": \"{c}\", \"area_percent\": {area:.1} }}{}\n",
            k + 1,
            if k + 1 < plates.len() { "," } else { "" }
        );
    }
    json += "  ]\n}\n";
    std::fs::write(args.out.join(format!("{stem}_色版.json")), json)?;

    let rough: f64 = sep.plates.iter().map(|p| imgops::roughness(p, w, h)).sum::<f64>() / sep.plates.len().max(1) as f64;
    println!("分色误差:平均 ΔE {mean_de:.2};色版粗糙度 {rough:.2}(越小越干净)");
    println!("完成({:.1}s)→ {}", t0.elapsed().as_secs_f32(), args.out.display());
    Ok(())
}
