# SHARP Video Workbench

本地视频转动态高斯的实验工作台。主流程包含上传、机位检测、视频深度、逐帧 SHARP，以及高斯序列压缩打包与播放。

仓库仅保存源码、源码备份、配置、测试和依赖版本清单，不包含输入视频、运行记录、生成结果、训练权重、预训练权重或安装环境。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `workbench/server.mjs`、`workbench/static/` | 本地 HTTP 服务与浏览器播放器 |
| `workbench/*_worker.py`、`workbench/gaussian_*.py` | 深度、高斯与序列压缩处理 |
| `workbench/tests/` | Python 与 Node 测试 |
| `workbench/trajectory_codec/` | 每轨迹编码、共享属性解码器实验；尚未达到保真重建目标 |
| `backups/` | 替换前的高斯轨迹与中值滤波源码备份，含逐文件 SHA-256 |
| `workbench/static/compression/` | splats4D 编解码适配与整段 GPU 帧缓存播放器 |
| `workbench/dynamic/` | 已停止的共享变形模型实验源码 |
| `ml-sharp/`、`video-depth-anything/` | 模型推理上游源码 |
| `ml-depth-pro/`、`moge/`、`moge-utils3d/` | 深度对比实验的上游源码 |
| `workbench/third_party/` | MoDGS、OmniMotion、RAFT、DINO、pipeline 与 splats4D 上游源码 |

## 获取源码

```sh
git clone https://github.com/zjllllllllll/sharp-video.git
cd sharp-video
```

上游源码已作为普通文件纳入本仓库，无须初始化子模块。来源、固定提交、完整依赖清单和安装说明见 [dependencies/README.md](dependencies/README.md)。各目录保留原许可证。

## 安装和运行

需要 Node.js、Python、NVIDIA CUDA 环境，以及带 `vidstabdetect` / `vidstabtransform` 滤镜的 FFmpeg。GPU 处理依赖较多，请分别建立 Python 环境。

已使用的 GPU 基础组合是 PyTorch 2.8.0 / CUDA 12.8；RTX 50 系列应使用支持该显卡的 PyTorch 构建。不要直接用上游旧 requirements 覆盖这套 GPU 基础包。

Windows PowerShell 示例（在仓库根目录）：

```powershell
python -m venv .envs/sharp
& .envs/sharp/Scripts/python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& .envs/sharp/Scripts/python.exe -m pip install -r dependencies/python-main.lock.txt
& .envs/sharp/Scripts/python.exe -m pip install --no-deps -e ./ml-sharp
python download_sharp_weights.py

python -m venv .envs/video-depth-anything
& .envs/video-depth-anything/Scripts/python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& .envs/video-depth-anything/Scripts/python.exe -m pip install -r dependencies/python-main.lock.txt

python -m venv .envs/splats4d
& .envs/splats4d/Scripts/python.exe -m pip install -r workbench/compression-requirements.txt

npm --prefix workbench ci
node workbench/server.mjs
```

默认打开 <http://127.0.0.1:8775>，也可双击 `workbench/start.cmd`。

以上覆盖主流程依赖。深度对比和历史训练实验按各上游说明安装专用依赖；轨迹编码实验另需 `workbench/trajectory_codec/requirements.txt` 与 gsplat 1.5.3。

## 模型文件与本机路径

权重从对应上游官方项目获取，并遵守各自的使用许可：

| 模型 | 默认本机文件 |
| --- | --- |
| [SHARP](https://github.com/apple-aiml-research/ml-sharp) | `ml-sharp/checkpoints/sharp_2572gikvuh.pt` |
| [Metric Video Depth Anything Base](https://github.com/DepthAnything/Video-Depth-Anything) | `video-depth-anything/checkpoints/metric_video_depth_anything_vitb.pth` |
| MoGe-2（可选） | `moge/checkpoints/moge-2-vits-normal/model.pt` |
| Depth Pro（可选） | `ml-depth-pro/checkpoints/depth_pro.pt` |

可在启动服务前设置下列环境变量：

| 环境变量 | 用途 |
| --- | --- |
| `SHARP_VIDEO_PORT` | HTTP 端口，默认 8775 |
| `SHARP_VIDEO_FFMPEG` | FFmpeg 命令或绝对路径，默认从 PATH 查找 |
| `SHARP_VIDEO_CHECKPOINT` | SHARP 权重路径 |
| `SHARP_VIDEO_SHARP_PYTHON` | SHARP Python，默认 `.envs/sharp` |
| `SHARP_VIDEO_DEPTH_PYTHON` | 视频深度 Python，默认 `.envs/video-depth-anything` |
| `SHARP_VIDEO_MOGE_PYTHON` | MoGe Python，默认 `.envs/moge2` |
| `SHARP_VIDEO_DEPTH_PRO_PYTHON` | Depth Pro Python，默认 `.envs/depth-pro` |
| `SHARP_VIDEO_COMPRESSION_PYTHON` | 压缩 Python，默认 `.envs/splats4d` |
| `SHARP_VIDEO_DYNAMIC_PYTHON` | 训练实验 Python，默认 `.envs/dynamic` |

输入和结果分别写入 `workbench/uploads/`、`workbench/runs/`。这些目录与模型、缓存、日志均已加入 `.gitignore`。

## 验证

在具备相应 Python 依赖的环境中：

```powershell
python -m unittest discover -s workbench/tests -p 'test_*.py' -v
node --test workbench/tests/*.test.mjs
```

主流程、文件格式和实验说明见 [workbench/README.md](workbench/README.md)。

## 使用状态与许可

这是研究和工程实验代码。轨迹编码实验的固定配置已运行过完整序列，但没有达到保真压缩目标；原轨迹计算与中值滤波步骤已移除，源码快照保存在 `backups/gaussian-tracks-2026-10-20.zip`。

各上游组件的许可证及适用限制以其仓库为准。`workbench/dynamic/upstream.py` 包含已注明来源的 Graphdeco / MoDGS 代码，`trajectory_codec/LICENSE.siren-pytorch` 保留 SIREN 实现的 MIT 许可；`workbench/static/compression/` 保留 splats4D 和 fzstd 的 MIT 许可。此仓库未为全部组合代码另行授予统一的开源许可。
