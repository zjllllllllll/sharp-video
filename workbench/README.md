# SHARP Video Workbench

当前主流程为逐帧 SHARP、完整高斯参数序列和中值滤波，轨迹拟合已删除。

1. 上传视频。
2. 检测固定机位，或对轻微抖动生成防抖视频。
3. 使用 Metric Video Depth Anything Base 保存 RGB 视频及浮点深度。
4. 每帧 RGB 与对应深度输入 SHARP，生成完整双层 PLY 和规则高斯张量。
5. 用相邻帧光流建立生命周期轨迹，保存包含全部高斯属性的独立参数序列。
6. 总长度少于 5 帧的轨迹全部属性保留原值；其余轨迹使用 5 帧中值滤波处理全部属性，并在同一帧、同一视角切换原始和滤波后的结果。

按[根目录安装说明](../README.md)准备环境后，双击 `start.cmd`，打开 <http://127.0.0.1:8775>。
视频保存在 `uploads/`，结果保存在 `runs/<视频ID>/`；仓库不附带运行记录或模型文件。
共享变形网络训练已停止，当前主流程不调用它。

独立的轨迹编码实验从工作台右上角进入（`/trajectory-codec-experiment.html`）：
每条高斯生命周期保存 64 维编码，全部轨迹共用 Modulated SIREN，拟合第五步未滤波的完整属性。
训练、停止续训、结果下载和原始／重建同步预览见 `trajectory_codec/README.md`。

## 输出

| 阶段 | 文件 | 内容 |
| --- | --- | --- |
| 深度 | `depth_source.mp4`、`depths.npz` | 同步 RGB 与浮点深度 |
| PLY | `gaussians/frame_00000.ply` | 每帧未经抽样的完整双层高斯 |
| 规则张量 | `gaussians/tensors/frame_00000.pt` | SHARP 输出属性与网格、相机信息 |
| 光流 | `flow/` | RAFT Small C_T_V2 相邻帧双向光流及对应记录 |
| 参数序列 | `tracks-lifecycle/animation.4dgs` | 全部帧、生命周期 ID 和 14 项高斯参数 |
| 中值滤波 | `tracks-median/animation.4dgs` | 全属性滤波后的独立参数序列，原始第五步文件保留 |

参数序列可以独立播放，不依赖原 PLY 或深度，使用本项目的自定义容器
`sharp.gaussian_tracks.v5`。

每帧保留两层高斯。光流推进网格位置，重复目标保留一条轨迹，遗漏目标建立新轨迹，
因此后续新出现的高斯也进入参数序列。生命周期结束后槽位可以复用，但不会拼接成同一条轨迹。

每帧保存位置、SH0 颜色、透明度 logit、对数尺度和 wxyz 四元数，共 14 个参数。
大于 2 GiB 的文件按多个 HTTP Range 缓冲读取。

第六步的位置、SH0 颜色、透明度 logit、对数尺度分别使用标准 5 帧中值。
总长度少于 5 帧的轨迹跳过滤波，14 项参数全部直接复制原值；恰好 5 帧的轨迹仍参与滤波。
由于这些编码为单调变换，奇数窗口取中值与解码后的结果等价。起止边界使用
SciPy `ndimage.median_filter(mode='reflect')` 的半采样对称延拓；窗口不跨生命周期或高斯层。
旋转使用 SO(3) 角距离下的 medoid（窗口内样本中，总角距离最小的旋转），距离为
`2*acos(abs(dot(q_i,q_j)))`，计算距离前归一化，输出保留原样本四元数。
这属于离散样本中位数，不是逐分量四元数中值，也不是连续 SO(3) 几何中位数。
距离定义参考 [Rotation Averaging](https://users.cecs.anu.edu.au/~hongdong/ACCV2009.pdf)。
不添加异常阈值或突变分类规则。

## 运行环境与入口

- SHARP Python：默认 `../.envs/sharp`，可用 `SHARP_VIDEO_SHARP_PYTHON` 指定解释器。
- SHARP 源码：`../ml-sharp`
- SHARP 权重：`../ml-sharp/checkpoints/sharp_2572gikvuh.pt`，可用 `SHARP_VIDEO_CHECKPOINT` 指定。
- 光流权重：`checkpoints/raft_small_C_T_V2-01064c6d.pth`。
- 浏览器播放器：Spark 2.2.0、Three.js 0.180.0。
- `POST/GET /api/gaussians`：第四步。
- `POST/GET /api/tracks`：第五步，缺少光流时先生成光流。
- `POST/GET /api/median`：第六步，生成并读取 5 帧全属性中值滤波结果。
- `GET /api/state`：恢复视频及各阶段状态。

请求通过 `filename` 指定已上传的视频。第四至六步显示进度，刷新后读取已保存状态和结果。
点击三维画面后用 WASD 移动、Q/E 升降、Shift 加速，拖动环视，滚轮前后移动。
第六步预先准备原始和滤波后的整段动画，显示实际播放帧率。页面地址为 `/#median`。

## 验证

Python 测试覆盖光流坐标、生命周期对应和双层全部属性。
中值滤波测试涵盖与 SciPy 逐轨迹结果一致、短轨迹和换 ID 边界、阶跃与脉冲、四元数符号不变性。
Node 测试覆盖文件解码、大文件分段读取、参数更新及播放同步。
先运行 Python 测试生成跨语言测试数据，再运行 `node --test workbench/tests/*.test.mjs`。
