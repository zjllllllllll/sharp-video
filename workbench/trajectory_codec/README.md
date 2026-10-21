> 历史独立实验：旧第五步已从主流程移除。本页只读取此前已有的生命周期文件；旧计算脚本见根目录 backups/gaussian-tracks-2026-10-20.zip。

# Gaussian trajectory codec experiment

Entry: `/trajectory-codec-experiment.html` on the local workbench (default port 8775).
This experiment has not achieved faithful reconstruction. Generated data and weights are not included.
The experiment reads only the unfiltered stage-five `tracks-lifecycle/animation.4dgs`.
The six existing stages and their files are unchanged.

## Sources and adaptation

- [Modulated Periodic Activations (ICCV 2021)](https://ishit.github.io/modsine/): latent-conditioned sinusoidal coordinate networks.
- [siren-pytorch 0.1.7](https://github.com/lucidrains/siren-pytorch): community implementation, MIT; see `LICENSE.siren-pytorch`.
- [DeepSDF (CVPR 2019), official code](https://github.com/facebookresearch/DeepSDF): jointly optimize per-instance latent codes and a shared decoder.

`AttributeDecoder` uses the installed upstream `SirenNet` and `Modulator` layers and
initialization. The only forward adaptation is batched latent concatenation on the
last axis and elementwise modulation of corresponding samples. The image-grid
wrapper is not used. This is a Gaussian-attribute adaptation, not an upstream
ready-made Gaussian codec. No SDF distance clamping, shape encoder, meta-training,
spatial deformation field or temporal denoising penalty is introduced.

`config.json` fixes 64 FP32 latent dimensions, five 256-wide synthesis and modulation
layers, frequencies 30/1, 14 output channels, 20 complete epochs, effective batch
65,536, Adam network LR 1e-4 and SparseAdam latent LR 1e-3. Network size: 612,622
parameters. Latents use normal initialization with std 1/sqrt(64). Every observed
Gaussian-frame sample occurs once per epoch, including all short tracks.

## Encoding and loss

Gaussian identity is `cellLifecycleId * layerCount + layer`. Each cell lifecycle
stores int32 `[startFrame, endFrameExclusive, slot]`; the row index is its ID.
Slot reuse never merges separate lifecycle codes. Time is normalized from the
first and last actual timestamps of each lifecycle to [-1,1]; singletons use 0.

The decoder returns 14 attributes in the existing order: XYZ, SH0 RGB, opacity
logit, three log scales, and wxyz quaternion. Nine position/color/log-scale
channels use global sample mean and std (constant channels use divisor 1).
Opacity loss compares sigmoid values, handling source +/-Infinity saturation.
Rotation uses the minimum of the q and -q unit-quaternion squared distances.
The five group means are summed with equal weights. No source samples are dropped.
Inference restores source units and normalizes quaternion outputs.

## Process and files

The Node manager launches `trajectory_codec_worker.py run --run-dir <run>` with the
existing `.envs/dynamic/Scripts/python.exe`. Separate child processes train, decode
and evaluate, releasing GPU allocations between phases. State is persisted in
`runs/<id>/trajectory-codec/state.json`; the page reads the same state after reload.

- `trajectory-codec.pt`: standalone inference package, standard PyTorch serialization
  with tensors and primitive metadata, loaded using `weights_only=True`. Contains
  codes, network, configuration, normalization, lifecycle metadata and the original
  frame/camera header. It is a project-specific codec format, not generic 4DGS.
- `checkpoint.pt`: latest training recovery state, including optimizers, sampler
  position and RNG states. Saved every five epochs and on Stop.
- `prepared.pt`: normalization statistics and lifecycle/slot metadata.
- `loss.jsonl`: reconstruction loss history, no fabricated convergence criterion.
- `animation.4dgs`, `preview.json`: reconstructed v5 playback cache. Regenerable
  from the inference package alone; not counted as compressed output.
- `report.json`: exact full-observation physical attribute MAE/P95/P99, plus all
  original-view frame PSNR/SSIM at the source resolution. Rendering compares raw
  and decoded Gaussian parameters using the same gsplat settings. SH0 is decoded
  directly to sRGB, with no old MoDGS linearRGB transfer.

Stop is cooperative: the current update completes, a checkpoint is atomically
written, then the process exits. Resume regenerates the deterministic sample order
and skips already completed updates. If interrupted after export, the next
unfinished phase resumes without retraining. A stopped decode/evaluation phase
restarts that phase; a completed phase is reused. No task starts automatically on
page refresh or server restart. Refresh reads live supervisor PID and saved state.

The first UI Stop/Resume validation belongs to the same full training run.
Microbatch reduction on CUDA OOM preserves the effective batch through weighted
gradient accumulation. All weights and activations remain FP32.

Standalone decoding, without source PLY/video/trajectory files:

```powershell
& .envs/dynamic/Scripts/python.exe workbench/trajectory_codec_worker.py decode --package <trajectory-codec.pt> --output-dir <preview-directory>
```

API: `POST/GET /api/trajectory-codec`, `POST /api/trajectory-codec/stop`; select the
uploaded video through `filename`. Final model and report download through the
existing `/results/<id>/trajectory-codec/` file server with HTTP Range support.

The page reuses `MedianComparisonViewer` solely for synchronous camera/time A/B
playback. Its internal `filtered` variant displays the **decoded**, unfiltered-target
reconstruction. Neural decoding speed, complete decoding preparation time, browser
cache preparation time and cached playback FPS are shown separately.

## Verification

```powershell
& .envs/dynamic/Scripts/python.exe -m unittest discover -s workbench/tests -p test_trajectory_codec.py -v
```

Checks compare batched outputs and gradients to unmodified upstream calls, cover
q/-q and saturated opacity, verify lifespan/slot/layer ordering and deterministic
resume, and decode an exported fixture in a fresh process after removing its
source. Full-run export checks predictions before and after reload; evaluation
verifies all reconstructed IDs and lifetimes against the original sequence.
