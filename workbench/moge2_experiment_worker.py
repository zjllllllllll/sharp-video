import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pointcloud_common import save_depth_preview, write_binary_ply, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))

    from moge.model.v2 import MoGeModel

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()
    prediction = model.infer(image, resolution_level=9, apply_mask=True)
    elapsed = time.perf_counter() - started

    points = prediction["points"].detach().cpu().numpy()
    depth = prediction["depth"].detach().cpu().numpy()
    intrinsics = prediction["intrinsics"].detach().cpu().numpy()
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    if "mask" in prediction:
        valid &= prediction["mask"].detach().cpu().numpy().astype(bool)

    point_count = write_binary_ply(output_dir / "moge2.ply", points[valid], rgb[valid])
    near, far = save_depth_preview(output_dir / "moge2_depth.png", depth, valid)
    height, width = depth.shape
    metadata = {
        "model": "MoGe-2 ViT-S Normal",
        "width": width,
        "height": height,
        "pointCount": point_count,
        "elapsedSeconds": round(elapsed, 3),
        "depthNearP02": near,
        "depthFarP98": far,
        "focalX": float(intrinsics[0, 0] * width),
        "focalY": float(intrinsics[1, 1] * height),
        "principalX": float(intrinsics[0, 2] * width),
        "principalY": float(intrinsics[1, 2] * height),
        "device": str(device),
    }
    write_metadata(output_dir / "moge2.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
