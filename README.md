# autoseparation 丝网印刷自动分色

输入一张图,自动选油墨、分色,输出分层 PSD 和每色一张索引色 TIFF(`名称-1.tif` … `名称-9.tif`、`名称-a.tif` …),格式与人工分色文件一致。

## 编译

```bash
git clone https://github.com/jackzuo809-netizen/autoseparation2.git
```

### Windows

1. 安装 Rust:<https://rustup.rs>,一路默认。
2. 安装 Visual Studio Build Tools(勾选「使用 C++ 的桌面开发」),读 CMYK 图的色彩管理库需要 C 编译器。
3. 在文件夹里打开 PowerShell:

```powershell
cargo build --release
```

编译出来的程序是 `target\release\autoseparation.exe`。

### Linux

```bash
curl https://sh.rustup.rs -sSf | sh     # 安装 Rust
sudo apt install build-essential        # 一般已经有
cargo build --release
```

## 运行

```bash
autoseparation 原图.tif -n 11 -o 输出文件夹
```

默认用 CPU 计算。实测在 44 线程的 Xeon 上,CPU 比显卡(wgpu)快约 9 倍,4K 图 4–5 秒。
显卡版本保留为可选(`--gpu on`),目前的着色器实现还没有针对显卡优化。

常用参数:

| 参数 | 说明 |
|---|---|
| `-n 11` | 油墨数量;不填则自动决定 |
| `--garment "#000000"` | 衣服颜色;深色衣服会自动加白墨打底 |
| `--mode tone` / `--mode flat` | 强制按渐变图 / 平涂图处理(默认自动判断) |
| `--gpu on` | 用显卡计算(默认关闭,实测 CPU 更快) |
| `--coherence 0.03` | 色版成片程度,越大越干净、颜色略偏;0 关闭 |
| `--coherence-mm 10` | 成片的范围(毫米) |
| `--cmyk-profile 文件.icc` | CMYK 原图没有内嵌 ICC 时用的配置文件 |

### CMYK 原图

没有内嵌 ICC 的 CMYK 图默认按 Japan Color 2001 Coated 转换。Windows 装了 Photoshop 会自动找到;
Linux 上请从装有 Photoshop 的电脑拷一份 `JapanColor2001Coated.icc`,用 `--cmyk-profile` 指定。

## 对比工具

```bash
compare 原图.tif 色版-*.tif
```

按印刷顺序合成色版,输出与原图的色差、色版粗糙度和每个像素的叠印层数,人工版和程序版都能比。
