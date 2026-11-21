//! GPU 加速(wgpu:Windows 走 DX12/Vulkan,Linux 走 Vulkan,Mac 走 Metal)。
//!
//! 只负责最重的「逐像素带约束求解」,逻辑与 tone.rs 的 CPU 版一致(见 solve.wgsl)。
//! 图像按横条分块上传,避免单个缓冲区过大。

use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::color::{Lab, Rgb};
use crate::mix::MAXP;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    y0: u32,
    rows: u32,
    f: u32,
    lw: u32,
    lh: u32,
    npts: u32,
    base: u32,
    n_inks: u32,
    max_layers: u32,
    mode: u32,
    step: u32,
    min_dot: u32,
    halo_y0: u32,
    out_words: u32,
    mu: f32,
    sigma: f32,
    sum_w: f32,
    pad0: f32,
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    max_binding: u64,
    pub name: String,
}

/// 一次求解任务的静态数据。
pub struct Job<'a> {
    pub w: usize,
    pub h: usize,
    pub f: usize,
    pub lw: usize,
    pub lh: usize,
    pub base: usize,
    pub n_inks: usize,
    pub max_layers: usize,
    pub step: usize,
    pub min_dot: u8,
    pub mu: f32,
    pub sigma: f32,
    pub sum_w: f32,
    /// 底色、[白墨]、油墨
    pub pts: &'a [Rgb],
    pub pixels: &'a [[u8; 3]],
    pub uidx: &'a [u32],
    /// 每种颜色不加约束的解,MAXP 个一组
    pub mixes: &'a [f32],
    pub guide: &'a [Lab],
}

/// 每块约 100 万像素:单次显卡任务足够短,不会触发系统看门狗(Windows 默认 2 秒)。
const TILE_PIXELS: usize = 1 << 20;

impl Gpu {
    /// 找一块真正的显卡(跳过软件渲染);找不到返回 None。
    pub fn new() -> Option<Gpu> {
        match Self::try_new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("GPU 不可用:{e:#}");
                None
            }
        }
    }

    fn try_new() -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("没有找到显卡")?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            bail!("只有软件渲染({}),不使用", info.name);
        }
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 8 {
            bail!("显卡存储缓冲区数量不够");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("autoseparation"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .context("无法初始化显卡")?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("solve"),
            source: wgpu::ShaderSource::Wgsl(include_str!("solve.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("solve"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Gpu {
            device,
            queue,
            pipeline,
            max_binding: limits.max_storage_buffer_binding_size.min(limits.max_buffer_size),
            name: format!("{}({:?})", info.name, info.backend),
        })
    }

    fn storage(&self, label: &str, data: &[u8]) -> Result<wgpu::Buffer> {
        if data.len() as u64 > self.max_binding {
            bail!("{label} 需要 {} MB,超过显卡单个缓冲区上限 {} MB", data.len() >> 20, self.max_binding >> 20);
        }
        // 缓冲区不能为空
        let pad = [0u8; 16];
        let contents = if data.is_empty() { &pad[..] } else { data };
        Ok(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }))
    }

    /// 迭代模式:对抽样像素(x、y 都是 step 的倍数)求解,逐个回调 (x, y, 权重)。
    pub fn run_samples(&self, job: &Job, low: &[[f32; MAXP]], mut sink: impl FnMut(usize, usize, &[f32])) -> Result<()> {
        let step = job.step.max(1);
        let sw = job.w.div_ceil(step);
        self.run(job, low, 1, |y0, rows, data| {
            let words: &[f32] = bytemuck::cast_slice(data);
            for sy in 0..rows.div_ceil(step) {
                let y = y0 + sy * step;
                if y >= job.h {
                    break;
                }
                for sx in 0..sw {
                    let o = (sy * sw + sx) * MAXP;
                    sink(sx * step, y, &words[o..o + MAXP]);
                }
            }
        })
    }

    /// 最终模式:每块回调 (起始行, 行数, 打包的网点值)。每像素 out_words 个 u32。
    pub fn run_final(&self, job: &Job, low: &[[f32; MAXP]], sink: impl FnMut(usize, usize, &[u32])) -> Result<()> {
        let mut sink = sink;
        self.run(job, low, 0, |y0, rows, data| sink(y0, rows, bytemuck::cast_slice(data)))
    }

    pub fn out_words(n_inks: usize) -> usize {
        (n_inks + 1).div_ceil(4)
    }

    fn run(&self, job: &Job, low: &[[f32; MAXP]], mode: u32, mut sink: impl FnMut(usize, usize, &[u8])) -> Result<()> {
        let (w, h) = (job.w, job.h);
        let step = job.step.max(1);
        // 每块行数:像素数不超过 TILE_PIXELS,且是 step 的倍数(保证抽样位置是全局对齐的)
        let rows_per_tile = ((TILE_PIXELS / w).max(step) / step * step).min(h.div_ceil(step) * step);
        let out_words = Self::out_words(job.n_inks);
        let out_bytes_per_tile = if mode == 0 {
            rows_per_tile * w * out_words * 4
        } else {
            rows_per_tile.div_ceil(step) * w.div_ceil(step) * MAXP * 4
        };

        let pts: Vec<[f32; 4]> = job.pts.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
        let guide: Vec<[f32; 4]> = job.guide.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
        let low_flat: Vec<f32> = low.iter().flat_map(|a| a.iter().copied()).collect();
        let pts_buf = self.storage("pts", bytemuck::cast_slice(&pts))?;
        let mixes_buf = self.storage("mixes", bytemuck::cast_slice(job.mixes))?;
        let guide_buf = self.storage("guide", bytemuck::cast_slice(&guide))?;
        let low_buf = self.storage("low", bytemuck::cast_slice(&low_flat))?;

        let px_cap = ((rows_per_tile + 2) * w * 4) as u64;
        let uidx_cap = (rows_per_tile * w * 4) as u64;
        let out_cap = out_bytes_per_tile as u64;
        for (label, size) in [("px", px_cap), ("uidx", uidx_cap), ("out", out_cap)] {
            if size > self.max_binding {
                bail!("{label} 分块 {} MB 超过显卡上限", size >> 20);
            }
        }
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let px_buf = mk("px", px_cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let uidx_buf = mk("uidx", uidx_cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let out_buf = mk("out", out_cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let staging = mk("staging", out_cap, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let done_cap = (rows_per_tile * 4).max(16) as u64;
        let done_buf = mk("done", done_cap, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        let done_staging = mk("done_staging", done_cap, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let params_buf = mk("params", std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        let layout = self.pipeline.get_bind_group_layout(0);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("solve"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: px_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uidx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: mixes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: guide_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: low_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: done_buf.as_entire_binding() },
            ],
        });

        // 待算的块(起始行, 行数);某块没算完就对半拆开重算
        let mut todo: Vec<(usize, usize)> =
            (0..h).step_by(rows_per_tile).map(|y0| (y0, rows_per_tile.min(h - y0))).rev().collect();
        let mut retries = 0;
        while let Some((y0, rows)) = todo.pop() {
            let halo_y0 = y0.saturating_sub(1);
            let halo_y1 = (y0 + rows + 1).min(h);
            let px: Vec<u32> = job.pixels[halo_y0 * w..halo_y1 * w]
                .iter()
                .map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32)
                .collect();
            self.queue.write_buffer(&px_buf, 0, bytemuck::cast_slice(&px));
            self.queue.write_buffer(&uidx_buf, 0, bytemuck::cast_slice(&job.uidx[y0 * w..(y0 + rows) * w]));
            let params = Params {
                w: w as u32,
                h: h as u32,
                y0: y0 as u32,
                rows: rows as u32,
                f: job.f as u32,
                lw: job.lw as u32,
                lh: job.lh as u32,
                npts: job.pts.len() as u32,
                base: job.base as u32,
                n_inks: job.n_inks as u32,
                max_layers: job.max_layers as u32,
                mode,
                step: step as u32,
                min_dot: job.min_dot as u32,
                halo_y0: halo_y0 as u32,
                out_words: out_words as u32,
                mu: job.mu,
                sigma: job.sigma,
                sum_w: job.sum_w,
                pad0: 0.0,
            };
            self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

            // 本块的线程网格、输出字节数、每行应完成的数量、行数
            let (gx, gy, used, per_row, n_rows) = if mode == 0 {
                (w.div_ceil(8), rows.div_ceil(8), rows * w * out_words * 4, w, rows)
            } else {
                let (sw, sh) = (w.div_ceil(step), rows.div_ceil(step));
                (sw.div_ceil(8), sh.div_ceil(8), sh * sw * MAXP * 4, sw, sh)
            };
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.clear_buffer(&done_buf, 0, None);
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx as u32, gy as u32, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, used as u64);
            enc.copy_buffer_to_buffer(&done_buf, 0, &done_staging, 0, (n_rows * 4) as u64);
            self.queue.submit([enc.finish()]);

            let slice = staging.slice(..used as u64);
            let done_slice = done_staging.slice(..(n_rows * 4) as u64);
            let (tx, rx) = std::sync::mpsc::channel();
            let tx2 = tx.clone();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            done_slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx2.send(r);
            });
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("等待显卡失败:{e:?}"))?;
            for _ in 0..2 {
                rx.recv()?.map_err(|e| anyhow!("读回显卡结果失败:{e:?}"))?;
            }
            let complete = {
                let view = done_slice.get_mapped_range().map_err(|e| anyhow!("读回显卡结果失败:{e:?}"))?;
                let counts: &[u32] = bytemuck::cast_slice(&view);
                counts.iter().all(|&c| c as usize == per_row)
            };
            done_staging.unmap();
            if complete {
                let view = slice.get_mapped_range().map_err(|e| anyhow!("读回显卡结果失败:{e:?}"))?;
                sink(y0, rows, &view);
            }
            staging.unmap();
            if !complete {
                // 没算完(多半是单次任务太久被系统中止):拆成两半重算
                retries += 1;
                let half = (rows / 2).div_ceil(step) * step;
                if half == 0 || half >= rows || retries > 64 {
                    bail!("显卡第 {y0} 行起的一块计算没有完成");
                }
                eprintln!("显卡第 {y0} 行起的一块没有算完,拆小重算");
                todo.push((y0 + half, rows - half));
                todo.push((y0, half));
            }
        }
        Ok(())
    }
}
