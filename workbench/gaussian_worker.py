import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from gaussian_grid import FORMAT, NETWORK_SIZE_WH, grid_shape_from_initializer, save_grid_frame


INTERNAL_SHAPE = NETWORK_SIZE_WH


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a SHARP PLY sequence from RGB-D video.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--depth-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=-1)
    return parser.parse_args()


def write_json_atomic(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


@torch.inference_mode()
def predict_rgbd(predictor, image, depth, focal_px, device, unproject_gaussians):
    height, width = image.shape[:2]
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
        device=device, dtype=torch.float32
    ).permute(2, 0, 1) / 255.0
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).to(
        device=device, dtype=torch.float32
    )[None, None]
    depth_tensor = torch.nan_to_num(depth_tensor, nan=1.0, posinf=100.0, neginf=1.0).clamp_min(1e-3)

    image_resized = F.interpolate(
        image_tensor[None],
        size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
        mode="bilinear",
        align_corners=True,
    )
    depth_resized = F.interpolate(
        depth_tensor,
        size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
        mode="bilinear",
        align_corners=True,
    )
    disparity_factor = torch.tensor([focal_px / width], device=device, dtype=torch.float32)

    gaussians_ndc = predictor(image_resized, disparity_factor, depth_resized)
    intrinsics = torch.tensor(
        [
            [focal_px, 0, width / 2, 0],
            [0, focal_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics[0] *= INTERNAL_SHAPE[0] / width
    intrinsics[1] *= INTERNAL_SHAPE[1] / height
    return unproject_gaussians(
        gaussians_ndc,
        torch.eye(4, device=device),
        intrinsics,
        INTERNAL_SHAPE,
    )


def run(args):
    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    input_video = Path(args.input_video).resolve()
    depth_npz = Path(args.depth_npz).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gaussian.json"

    for required in (checkpoint, input_video, depth_npz):
        if not required.is_file():
            raise FileNotFoundError(f"输入文件不存在：{required}")

    sys.path.insert(0, str(repo / "src"))
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import save_ply, unproject_gaussians
    from sharp.utils.io import convert_focallength

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("逐帧 SHARP 生成需要 CUDA GPU")

    depths = np.load(depth_npz)["depths"]
    reader = imageio.get_reader(input_video)
    video_meta = reader.get_meta_data()
    fps = float(video_meta.get("fps") or 25.0)
    total = len(depths)
    if args.max_frames > 0:
        total = min(total, args.max_frames)
    if total < 1:
        raise RuntimeError("没有可处理的视频帧")

    first = reader.get_data(0)
    height, width = first.shape[:2]
    if depths.shape[1:] != (height, width):
        raise RuntimeError(
            f"RGB 与深度尺寸不一致：RGB={width}x{height}，深度={depths.shape[2]}x{depths.shape[1]}"
        )
    focal_px = float(convert_focallength(width, height, 30.0))

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict, strict=True)
    predictor.eval().to(device)
    grid_shape = grid_shape_from_initializer(predictor.init_model)
    del state_dict

    started = time.time()
    manifest = {
        "status": "running",
        "model": "Apple SHARP",
        "usesExternalDepth": True,
        "currentFrame": 0,
        "totalFrames": total,
        "fps": fps,
        "width": width,
        "height": height,
        "focalLengthPx": focal_px,
        "gaussianGrid": {"format": FORMAT, "shape": list(grid_shape)},
        "frames": [],
        "startedAt": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(manifest_path, manifest)
    emit({"event": "started", "current": 0, "total": total})

    try:
        for index in range(total):
            frame_started = time.time()
            image = first if index == 0 else reader.get_data(index)
            if image.shape[:2] != (height, width):
                raise RuntimeError(f"第 {index + 1} 帧尺寸发生变化")

            gaussians = predict_rgbd(
                predictor,
                image[:, :, :3],
                depths[index],
                focal_px,
                device,
                unproject_gaussians,
            )
            full_name = f"frame_{index:05d}.ply"
            tensor_name = f"tensors/frame_{index:05d}.pt"
            save_grid_frame(
                output_dir / tensor_name, gaussians,
                grid_shape_lhw=grid_shape, image_size_wh=(width, height),
                focal_length_px=focal_px, frame_index=index, fps=fps,
                source={"kind": "prediction", "video": str(input_video), "depth": str(depth_npz)},
            )
            save_ply(gaussians, focal_px, (height, width), output_dir / full_name)

            record = {
                "index": index,
                "full": full_name,
                "gaussianCount": int(gaussians.mean_vectors.shape[1]),
                "fullBytes": (output_dir / full_name).stat().st_size,
                "tensor": tensor_name,
                "tensorBytes": (output_dir / tensor_name).stat().st_size,
                "seconds": round(time.time() - frame_started, 3),
            }
            manifest["frames"].append(record)
            manifest["currentFrame"] = index + 1
            manifest["elapsedSeconds"] = round(time.time() - started, 3)
            write_json_atomic(manifest_path, manifest)
            emit(
                {
                    "event": "progress",
                    "current": index + 1,
                    "total": total,
                    "seconds": record["seconds"],
                }
            )
            del gaussians
            torch.cuda.empty_cache()

        manifest["status"] = "complete"
        manifest["completedAt"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsedSeconds"] = round(time.time() - started, 3)
        manifest["totalFullBytes"] = sum(frame["fullBytes"] for frame in manifest["frames"])
        manifest["totalTensorBytes"] = sum(frame["tensorBytes"] for frame in manifest["frames"])
        write_json_atomic(manifest_path, manifest)
        emit({"event": "complete", "current": total, "total": total})
    finally:
        reader.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "gaussian.json"
    try:
        run(args)
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        failed = {
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failedAt": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                failed = {**previous, **failed}
        except Exception:
            pass
        write_json_atomic(manifest_path, failed)
        emit({"event": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
