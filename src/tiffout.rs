//! 色版输出:索引色 TIFF(与人工分色文件同一格式)。
//! 调色板 0 = 油墨色,255 = 白,中间线性过渡;像素值 = 255 − 覆盖率。

use std::io::Write;
use std::path::Path;

pub fn write_indexed_plate(path: &Path, w: usize, h: usize, dpi: u32, ink: [u8; 3], coverage: &[u8]) -> std::io::Result<()> {
    // 布局:文件头(8) | IFD | 分辨率(2×8) | 调色板(1536) | 像素
    const N_TAGS: u16 = 13;
    let ifd_off = 8u32;
    let ifd_len = 2 + N_TAGS as u32 * 12 + 4;
    let xres_off = ifd_off + ifd_len;
    let yres_off = xres_off + 8;
    let cmap_off = yres_off + 8;
    let data_off = cmap_off + 768 * 2;
    let n = (w * h) as u32;

    let mut f = Vec::with_capacity(data_off as usize + n as usize);
    f.extend_from_slice(b"II*\0");
    f.extend_from_slice(&ifd_off.to_le_bytes());
    f.extend_from_slice(&N_TAGS.to_le_bytes());
    let mut tag = |t: u16, ty: u16, count: u32, value: u32| {
        f.extend_from_slice(&t.to_le_bytes());
        f.extend_from_slice(&ty.to_le_bytes());
        f.extend_from_slice(&count.to_le_bytes());
        f.extend_from_slice(&value.to_le_bytes());
    };
    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    const RATIONAL: u16 = 5;
    tag(256, LONG, 1, w as u32); // 宽
    tag(257, LONG, 1, h as u32); // 高
    tag(258, SHORT, 1, 8); // 位深
    tag(259, SHORT, 1, 1); // 不压缩
    tag(262, SHORT, 1, 3); // 索引色
    tag(273, LONG, 1, data_off); // 像素数据位置
    tag(277, SHORT, 1, 1); // 每像素 1 个样本
    tag(278, LONG, 1, h as u32); // 单条带
    tag(279, LONG, 1, n); // 条带字节数
    tag(282, RATIONAL, 1, xres_off);
    tag(283, RATIONAL, 1, yres_off);
    tag(296, SHORT, 1, 2); // 英寸
    tag(320, SHORT, 768, cmap_off); // 调色板
    f.extend_from_slice(&0u32.to_le_bytes()); // 没有下一个 IFD
    for _ in 0..2 {
        f.extend_from_slice(&dpi.to_le_bytes());
        f.extend_from_slice(&1u32.to_le_bytes());
    }
    for c in 0..3 {
        for i in 0..256u32 {
            let v = ink[c] as u32 + (255 - ink[c] as u32) * i / 255;
            f.extend_from_slice(&((v * 257) as u16).to_le_bytes());
        }
    }
    f.extend(coverage.iter().map(|&c| 255 - c));
    std::fs::File::create(path)?.write_all(&f)
}

/// 色版编号:1..9, a..z(与人工文件命名一致)。
pub fn plate_suffix(i: usize) -> String {
    let n = i + 1;
    if n <= 9 {
        n.to_string()
    } else {
        ((b'a' + (n - 10) as u8) as char).to_string()
    }
}
