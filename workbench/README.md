# SHARP Video Workbench

主流程共五步：

1. 上传视频。
2. 检测固定机位，或对轻微抖动生成防抖视频。
3. 使用 Metric Video Depth Anything Base 保存 RGB 视频及浮点深度。
4. 每帧 RGB 与对应深度输入 SHARP，生成完整双层 PLY 和规则高斯张量。
5. **压缩高斯**：用 splats4D 将完整 PLY 序列压缩为一个 `.splat4d` 文件，在同一界面准备和播放。

原第五步生命周期高斯轨迹、原第六步中值滤波的入口、生成接口和计算脚本已移除。
替换前源码保存在根目录 `backups/gaussian-tracks-2026-10-20.zip`，并有逐文件 SHA-256。
独立轨迹编码历史实验仍保留旧文件读写兼容逻辑，不参与新主流程。

## 安装和运行

按[根目录安装说明](../README.md)准备主流程环境与 Node 依赖。
压缩使用独立环境：

```powershell
python -m venv .envs/splats4d
& .envs/splats4d/Scripts/python.exe -m pip install -r workbench/compression-requirements.txt
```

从根目录运行 `node workbench/server.mjs`，或双击 `workbench/start.cmd`。
默认端口为 8775，可通过 `SHARP_VIDEO_PORT` 指定。
第五步地址为 `/#compression`；点击第四步的“下一步：压缩高斯”也可进入。

## 第五步

直接读取第四步的全部帧，不再计算光流、轨迹或中值滤波。
PLY 转成上游编码器要求的 32 字节 `.splat` 属性记录，保留顶点顺序、帧数和原视频帧率。
使用官方 `splats4d==0.1.0` 编码器默认配置：位置 2 mm、RGB/透明度 4 个八位级别、
输入网格四元数分量无额外损失、尺度 2%、GOP 30、zstd 3，不启用颜色滤波。
这是有损压缩；编码器误差上界以 `.splat` 为参照，PLY 转 `.splat` 的量化损失另计。

进度条显示输入转换的逐帧进度；上游编码器没有内部百分比输出，编码与完整误差校验阶段
使用不定进度条，不伪造百分比。任务状态保存在磁盘，刷新后可继续查看进度和结果。

完成后在同一界面准备整段播放：解码、协方差计算、初始排序及 GPU 上传在播放前完成。
播放阶段切换已经驻留 GPU 的帧，按原帧率顺序播放，避免实时解码阻塞与主动追时跳帧。
相机移动复用上游深度排序；显示帧率统计源帧更新次数，不是重复绘制次数。
当前 100 帧、每帧约 118 万高斯的序列，GPU 帧缓存和索引约占 4.25 GB。
准备期间保留进度显示；视觉和网页操作由用户验收。

## 输出与接口

| 文件 | 内容 |
| --- | --- |
| `depth_source.mp4`、`depths.npz` | 第三步同步 RGB 与浮点深度 |
| `gaussians/frame_00000.ply` | 第四步完整双层高斯 |
| `gaussians/tensors/frame_00000.pt` | 第四步属性、网格和相机信息 |
| `compressed/sequence.splat4d` | 最终单文件，播放不依赖原 PLY、深度或轨迹 |
| `compressed/compression.json` | 进度、帧率、相机、文件大小及完成状态 |
| `compressed/encoder-report.json` | 编码器完整误差检查报告 |
| `compressed/frames/` | 输入转换缓存，不是最终播放依赖 |

- `POST/GET /api/gaussians`：第四步。
- `POST/GET /api/compression`：新的第五步，JSON / 查询参数以 `filename` 指定视频。
- `GET /api/state`：恢复视频和各阶段状态。
- `GET /results/<runId>/compressed/sequence.splat4d`：文件下载与 HTTP Range 读取。

压缩解释器默认 `../.envs/splats4d`，可用 `SHARP_VIDEO_COMPRESSION_PYTHON` 指定。
编解码和渲染复用 splats4D，来源、许可及适配差异见 `static/compression/README.md`。

## 非视觉验证

```powershell
& .envs/splats4d/Scripts/python.exe -m unittest discover -s workbench/tests -p test_compression.py -v
node --test workbench/tests/*.test.mjs
```

`tests/compression_gpu_bench.mjs` 使用原生 Dawn WebGPU，直接读取最终文件，运行生产解码器、
完整 GPU 帧缓存与 1280×720 绘制性能检查；不打开网页、不截图。
其可选测试依赖 `webgpu==0.6.2` 已加入 `package.json` 的 devDependencies，由 `npm ci` 安装。