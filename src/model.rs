//! 印刷模型(与人工分色文件一致):每个色版是一个「正常」混合的图层,
//! 网点覆盖率 = 图层不透明度,按印刷顺序从下往上叠,上层盖住下层。
//!
//! 叠完后的颜色 = 底色和各油墨的凸组合(权重非负、和为 1):
//!   第 k 层权重 w_k = a_k × Π(上面各层的 1 − a_j)。
//! 反过来,给定权重也能唯一换算回每层覆盖率,见 `mix::Mixer::coverages`。

use crate::color::Rgb;

#[inline]
pub fn over(c: &mut Rgb, ink: Rgb, a: f32) {
    for k in 0..3 {
        c[k] = c[k] * (1.0 - a) + ink[k] * a;
    }
}

/// 分色结果:油墨按印刷顺序(下 → 上)、每色版覆盖率(0..255)、白墨打底(最下层)。
pub struct Separation {
    pub inks: Vec<Rgb>,
    pub plates: Vec<Vec<u8>>,
    pub under: Option<Vec<u8>>,
}

impl Separation {
    /// 用最终色版按印刷顺序重新合成预览。
    pub fn preview(&self, garment: Rgb, n: usize) -> Vec<[u8; 3]> {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map(|idx| {
                let mut c = garment;
                if let Some(u) = &self.under {
                    over(&mut c, [1.0; 3], u[idx] as f32 / 255.0);
                }
                for (k, ink) in self.inks.iter().enumerate() {
                    let a = self.plates[k][idx];
                    if a > 0 {
                        over(&mut c, *ink, a as f32 / 255.0);
                    }
                }
                crate::color::to_u8(c)
            })
            .collect()
    }
}
