//! 最小可用的分层 PSD 写入器(RGB 8 位,PackBits 压缩,支持中文图层名)。

use std::io::Write;

pub struct Layer {
    pub name: String,
    /// 图层颜色(整层同一颜色)
    pub color: [u8; 3],
    /// 每像素不透明度 = 覆盖率
    pub alpha: Vec<u8>,
    /// 混合模式键:b"norm" / b"mul "
    pub blend: [u8; 4],
}

fn packbits(row: &[u8], out: &mut Vec<u8>) {
    let n = row.len();
    let mut i = 0;
    while i < n {
        let mut run = 1;
        while i + run < n && run < 128 && row[i + run] == row[i] {
            run += 1;
        }
        if run >= 2 {
            out.push((1 - run as i32) as i8 as u8);
            out.push(row[i]);
            i += run;
        } else {
            let start = i;
            let mut len = 0;
            while i < n && len < 128 {
                if i + 1 < n && row[i] == row[i + 1] {
                    break;
                }
                i += 1;
                len += 1;
            }
            out.push((len - 1) as u8);
            out.extend_from_slice(&row[start..start + len]);
        }
    }
}

/// 压缩若干平面:返回 (每行字节数, 数据)。
fn rle_planes(planes: &[&[u8]], w: usize) -> (Vec<u16>, Vec<u8>) {
    let mut counts = Vec::new();
    let mut data = Vec::new();
    for p in planes {
        for row in p.chunks(w) {
            let before = data.len();
            packbits(row, &mut data);
            counts.push((data.len() - before) as u16);
        }
    }
    (counts, data)
}

fn channel_bytes(plane: &[u8], w: usize) -> Vec<u8> {
    let (counts, data) = rle_planes(&[plane], w);
    let mut out = Vec::with_capacity(2 + counts.len() * 2 + data.len());
    out.extend_from_slice(&1u16.to_be_bytes());
    for c in counts {
        out.extend_from_slice(&c.to_be_bytes());
    }
    out.extend_from_slice(&data);
    out
}

fn pascal_name(name: &str) -> Vec<u8> {
    let ascii: Vec<u8> = name.bytes().filter(|b| b.is_ascii_graphic() || *b == b' ').take(255).collect();
    let mut v = vec![ascii.len() as u8];
    v.extend_from_slice(&ascii);
    while v.len() % 4 != 0 {
        v.push(0);
    }
    v
}

fn luni_block(name: &str) -> Vec<u8> {
    let units: Vec<u16> = name.encode_utf16().collect();
    let mut data = Vec::new();
    data.extend_from_slice(&(units.len() as u32).to_be_bytes());
    for u in units {
        data.extend_from_slice(&u.to_be_bytes());
    }
    while data.len() % 4 != 0 {
        data.push(0);
    }
    let mut v = Vec::new();
    v.extend_from_slice(b"8BIMluni");
    v.extend_from_slice(&(data.len() as u32).to_be_bytes());
    v.extend_from_slice(&data);
    v
}

/// `layers` 从底到顶;`merged` 为合成预览(RGB 交错)。
pub fn write_psd(
    path: &std::path::Path,
    w: usize,
    h: usize,
    dpi: u32,
    layers: &[Layer],
    merged: &[[u8; 3]],
) -> std::io::Result<()> {
    let mut f = Vec::<u8>::new();
    // 文件头
    f.extend_from_slice(b"8BPS");
    f.extend_from_slice(&1u16.to_be_bytes());
    f.extend_from_slice(&[0; 6]);
    f.extend_from_slice(&3u16.to_be_bytes());
    f.extend_from_slice(&(h as u32).to_be_bytes());
    f.extend_from_slice(&(w as u32).to_be_bytes());
    f.extend_from_slice(&8u16.to_be_bytes());
    f.extend_from_slice(&3u16.to_be_bytes()); // RGB
    // 颜色模式数据
    f.extend_from_slice(&0u32.to_be_bytes());
    // 图像资源:分辨率
    let mut res = Vec::new();
    res.extend_from_slice(b"8BIM");
    res.extend_from_slice(&0x03EDu16.to_be_bytes());
    res.extend_from_slice(&[0, 0]);
    res.extend_from_slice(&16u32.to_be_bytes());
    for _ in 0..2 {
        res.extend_from_slice(&(dpi << 16).to_be_bytes());
        res.extend_from_slice(&1u16.to_be_bytes());
        res.extend_from_slice(&1u16.to_be_bytes());
    }
    f.extend_from_slice(&(res.len() as u32).to_be_bytes());
    f.extend_from_slice(&res);

    // 图层
    let n = w * h;
    let mut records = Vec::new();
    let mut chdata = Vec::new();
    for layer in layers {
        let planes: Vec<Vec<u8>> = (0..3).map(|c| vec![layer.color[c]; n]).collect();
        let chans: [(i16, Vec<u8>); 4] = [
            (-1, channel_bytes(&layer.alpha, w)),
            (0, channel_bytes(&planes[0], w)),
            (1, channel_bytes(&planes[1], w)),
            (2, channel_bytes(&planes[2], w)),
        ];
        for v in [0i32, 0, h as i32, w as i32] {
            records.extend_from_slice(&v.to_be_bytes());
        }
        records.extend_from_slice(&4u16.to_be_bytes());
        for (id, bytes) in &chans {
            records.extend_from_slice(&id.to_be_bytes());
            records.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        }
        records.extend_from_slice(b"8BIM");
        records.extend_from_slice(&layer.blend);
        records.extend_from_slice(&[255, 0, 0, 0]); // 不透明度、剪贴、标志、填充
        let mut extra = Vec::new();
        extra.extend_from_slice(&0u32.to_be_bytes()); // 图层蒙版
        extra.extend_from_slice(&0u32.to_be_bytes()); // 混合范围
        extra.extend_from_slice(&pascal_name(&layer.name));
        extra.extend_from_slice(&luni_block(&layer.name));
        records.extend_from_slice(&(extra.len() as u32).to_be_bytes());
        records.extend_from_slice(&extra);
        for (_, bytes) in chans {
            chdata.extend_from_slice(&bytes);
        }
    }
    let mut layer_info = Vec::new();
    layer_info.extend_from_slice(&(layers.len() as i16).to_be_bytes());
    layer_info.extend_from_slice(&records);
    layer_info.extend_from_slice(&chdata);
    if layer_info.len() % 2 != 0 {
        layer_info.push(0);
    }
    let section_len = 4 + layer_info.len() + 4;
    f.extend_from_slice(&(section_len as u32).to_be_bytes());
    f.extend_from_slice(&(layer_info.len() as u32).to_be_bytes());
    f.extend_from_slice(&layer_info);
    f.extend_from_slice(&0u32.to_be_bytes()); // 全局蒙版

    // 合成图像
    let planes: Vec<Vec<u8>> = (0..3).map(|c| merged.iter().map(|p| p[c]).collect()).collect();
    let refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
    let (counts, data) = rle_planes(&refs, w);
    f.extend_from_slice(&1u16.to_be_bytes());
    for c in counts {
        f.extend_from_slice(&c.to_be_bytes());
    }
    f.extend_from_slice(&data);

    std::fs::File::create(path)?.write_all(&f)
}
