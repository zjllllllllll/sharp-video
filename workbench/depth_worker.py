import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def configure_xformers_for_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12:
        return
    try:
        import xformers.ops

        original_attention = xformers.ops.memory_efficient_attention
        cutlass_forward = xformers.ops.fmha.cutlass.FwOp

        def cutlass_attention(*args, **kwargs):
            kwargs.setdefault("op", (cutlass_forward, None))
            return original_attention(*args, **kwargs)

        xformers.ops.memory_efficient_attention = cutlass_attention
    except ImportError:
        pass


MODEL_CONFIG = {
    "encoder": "vitb",
    "features": 128,
    "out_channels": [96, 192, 384, 768],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Metric Video Depth Anything for the workbench.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"模型权重不存在：{checkpoint}")
    if not input_path.is_file():
        raise FileNotFoundError(f"输入视频不存在：{input_path}")

    sys.path.insert(0, str(repo))
    configure_xformers_for_blackwell()
    from utils.dc_utils import read_video_frames, save_video
    from video_depth_anything.video_depth import VideoDepthAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, source_fps = read_video_frames(str(input_path), -1, -1, 1280)
    if len(frames) == 0:
        raise RuntimeError("输入视频没有可读取的视频帧")

    model = VideoDepthAnything(**MODEL_CONFIG, metric=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    depths, output_fps = model.infer_video_depth(
        frames,
        source_fps,
        input_size=518,
        device=device,
        fp32=False,
    )
    depths = np.asarray(depths, dtype=np.float32)

    source_video = output_dir / "depth_source.mp4"
    depth_video = output_dir / "depth_vis.mp4"
    depth_data = output_dir / "depths.npz"
    save_video(frames, str(source_video), fps=output_fps)
    save_video(depths, str(depth_video), fps=output_fps, is_depths=True)
    np.savez_compressed(depth_data, depths=depths)

    finite = depths[np.isfinite(depths)]
    if finite.size == 0:
        raise RuntimeError("模型输出不包含有效深度值")

    metadata = {
        "status": "complete",
        "model": "Metric Video Depth Anything Base",
        "device": device,
        "frameCount": int(depths.shape[0]),
        "height": int(depths.shape[1]),
        "width": int(depths.shape[2]),
        "fps": float(output_fps),
        "depthMin": float(finite.min()),
        "depthMax": float(finite.max()),
        "depthP50": float(np.percentile(finite, 50)),
        "depthP95": float(np.percentile(finite, 95)),
        "sourceVideo": "depth_source.mp4",
        "visualizationVideo": "depth_vis.mp4",
        "depthData": "depths.npz",
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "depth.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
