import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pointcloud_common import save_depth_preview, write_binary_ply, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    rgb, _, _ = depth_pro.load_rgb(image_path)
    height, width = rgb.shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
    started = time.perf_counter()
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    image = transform(rgb)
    prediction = model.infer(image, f_px=None)
    elapsed = time.perf_counter() - started

    depth = prediction["depth"].detach().float().cpu().numpy()
    focal = float(prediction["focallength_px"].detach().float().cpu())
    yy, xx = np.indices((height, width), dtype=np.float32)
    points = np.stack(
        ((xx - width / 2.0) * depth / focal,
         (yy - height / 2.0) * depth / focal,
         depth),
        axis=-1,
    )
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    point_count = write_binary_ply(output_dir / "depth_pro.ply", points[valid], rgb[valid])
    near, far = save_depth_preview(output_dir / "depth_pro_depth.png", depth, valid)
    metadata = {
        "model": "Depth Pro",
        "width": width,
        "height": height,
        "pointCount": point_count,
        "elapsedSeconds": round(elapsed, 3),
        "depthNearP02": near,
        "depthFarP98": far,
        "focalX": focal,
        "focalY": focal,
        "principalX": width / 2.0,
        "principalY": height / 2.0,
        "device": str(device),
    }
    write_metadata(output_dir / "depth_pro.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
