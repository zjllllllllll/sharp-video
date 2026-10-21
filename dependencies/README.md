# 外部源码与安装依赖

本仓库直接保存项目使用的上游模型、几何工具及编解码源码，不再使用 Git 子模块。
普通 `git clone` 和 GitHub 的源码 ZIP 均包含这些文件。

Python/npm 库的安装包仍由包管理器下载，CUDA、驱动和系统工具仍需安装；
这里不是离线运行环境，模型权重也不随仓库上传。

## 已纳入的上游源码

| 路径 | 用途 |
| --- | --- |
| `ml-sharp/` | SHARP 高斯及原生深度 |
| `video-depth-anything/` | 历史视频深度与对比实验 |
| `ml-depth-pro/` | Depth Pro 对比实验 |
| `moge/`、`moge-utils3d/` | MoGe 对比实验与几何工具 |
| `workbench/third_party/pipeline/` | MoGe 声明的 Python 工具依赖 |
| `workbench/third_party/MoDGS/` | 历史共享变形实验 |
| `workbench/third_party/omnimotion/` | 对应关系预处理，含其 RAFT、DINO 源码 |
| `workbench/third_party/splats4d/` | splats4D 原始 Rust 编解码器与网页播放器源码 |
| `workbench/static/compression/` | 工作台使用的 splats4D、fzstd 网页适配代码 |

`sources.json` 记录每份源码的上游地址与提交；各目录保留原许可证。
移除了子模块注册文件及上游示例视频、示例高斯等数据文件，保留代码与文档素材。
本机原有 Git 元数据另存于被忽略的 `.cache/dependency-vendor/git-metadata/`，权重不受转换影响。

## 主流程 Python 环境

`python-main.lock.txt` 锁定当前 Windows x64 / Python 3.13 运行环境的 51 个运行及构建辅助包版本，
包括 PyTorch 2.8.0 / CUDA 12.8、torchvision 0.23.0、gsplat 1.5.3，以及递归运行依赖。
它不是所有操作系统通用的锁文件；Linux 的 NVIDIA 运行包和编译工具由该平台的软件包管理器处理。

在仓库根目录执行 PowerShell：

```powershell
python -m venv .envs/sharp
& .envs/sharp/Scripts/python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& .envs/sharp/Scripts/python.exe -m pip install -r dependencies/python-main.lock.txt
& .envs/sharp/Scripts/python.exe -m pip install --no-deps -e ./ml-sharp
python download_sharp_weights.py
```

本机已使用 Python 3.13.15。创建环境前应确认 `python` 对应 Python 3.13。
安装 SHARP 时的 setuptools 构建依赖由其 `pyproject.toml` 声明，pip 会下载。
不要再执行上游旧版 PyTorch / CUDA 的 requirements 来覆盖这里的运行环境。

历史 VDA worker 使用相同的依赖，可创建独立环境后安装同一锁文件。
VDA 的代码路径由 worker 直接接入，无须从外部 Git 仓库安装。

## 序列压缩环境

```powershell
python -m venv .envs/splats4d
& .envs/splats4d/Scripts/python.exe -m pip install -r workbench/compression-requirements.txt
```

版本锁位于 `python-compression.lock.txt`。`splats4d==0.1.0` 下载对应平台的原生编码器；
需要自行编译时，可使用仓库里的源码与 Cargo 锁文件：

```powershell
cargo build --release --locked --manifest-path workbench/third_party/splats4d/converter/Cargo.toml
```

自行构建需要 Rust/Cargo 和平台的 C/C++ 工具链，构建所需 crates 仍由 Cargo 下载。
将构建出的 `splat4d` 可执行文件加入 PATH 后，现有压缩 worker 可以调用。

## 网页与 GPU 检查依赖

```powershell
npm --prefix workbench ci
node workbench/server.mjs
```

`workbench/package.json` 与 `package-lock.json` 声明并锁定 Three.js 0.180.0、Spark 2.2.0
及其递归依赖；可选性能检查依赖 `webgpu==0.6.2` 也在 devDependencies 中，
包含其 npm 递归版本与完整性信息。普通 `npm ci` 会安装这些包，不再读取本机缓存目录里的包。

```powershell
node --test workbench/tests/*.test.mjs
node workbench/tests/compression_gpu_bench.mjs <输入.splat4d> <报告.json>
```

GPU 检查需要支持 WebGPU 的驱动。只部署网页而不运行检查时可使用 `npm ci --omit=dev`。

## 可选实验

项目自身的轨迹编码和共享变形代码使用：

```powershell
python -m venv .envs/dynamic
& .envs/dynamic/Scripts/python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& .envs/dynamic/Scripts/python.exe -m pip install -r dependencies/python-experiments.requirements.txt
```

MoDGS、OmniMotion 的模块由项目直接从本地源码导入；无需安装它们上游完整训练环境。
其原始 requirements 仍随源码保留，用于上游工程参考。

MoGe 的推理依赖在 `moge/pyproject.toml`、`moge-utils3d/pyproject.toml` 声明，
先在独立环境安装本地包，以使用仓库内的 utils3d 和 pipeline：

```powershell
python -m venv .envs/moge2
& .envs/moge2/Scripts/python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& .envs/moge2/Scripts/python.exe -m pip install -e ./moge-utils3d -e ./workbench/third_party/pipeline
& .envs/moge2/Scripts/python.exe -m pip install -e ./moge
```

MoGe 的上游 Git 依赖已改为对应本地包的版本依赖，见 `sources.json` 的适配记录。
Depth Pro 使用 `ml-depth-pro/pyproject.toml` 的独立依赖；该上游要求 `numpy<2`，
应使用 Python 3.11/3.12 的独立 `.envs/depth-pro` 环境，不要与 Python 3.13 的主环境混装。

## 系统工具与权重

- 主工作台使用 Node.js（本机 24.19.0）、Python 和 FFmpeg。
- FFmpeg 需含 `vidstabdetect` / `vidstabtransform` 滤镜，可用 `SHARP_VIDEO_FFMPEG` 指定路径。
  本机使用 imageio-ffmpeg 0.6.0 随附的 FFmpeg 7.1；其他平台可安装对应构建。
- GPU 推理需要兼容的 NVIDIA 驱动；gsplat 编译及历史实验还需要 CUDA 12.8 和 C/C++ 编译器，
  Windows 本机使用 Visual Studio 2022 Build Tools。
- 主流程 SHARP 权重使用根目录 `download_sharp_weights.py`；对比实验权重按各上游说明下载。

以上系统安装包、Python/npm 安装包、模型权重均不存入 Git。
