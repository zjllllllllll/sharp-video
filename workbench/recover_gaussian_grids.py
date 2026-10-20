"""Recover grid caches from this workbench's original, unmodified SHARP PLYs.

The PLY format itself does not guarantee grid order. This converter relies on
the inspected SHARP composer/exporter and this worker's fixed preprocessing.
It must not be used to infer layout for arbitrary, reordered, or pruned PLYs.
"""

import argparse
import json
import sys
from pathlib import Path

from gaussian_grid import FORMAT, grid_shape_from_initializer, save_grid_frame
from gaussian_worker import write_json_atomic


def recover(repo, gaussian_dir):
    sys.path.insert(0, str(Path(repo).resolve() / "src"))
    from sharp.models.params import PredictorParams
    from sharp.utils.gaussians import load_ply

    gaussian_dir = Path(gaussian_dir).resolve()
    manifest_path = gaussian_dir / "gaussian.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grid_shape = grid_shape_from_initializer(PredictorParams().initializer)
    for record in manifest["frames"]:
        gaussians, _ = load_ply(gaussian_dir / record["full"])
        tensor_name = f"tensors/{Path(record['full']).stem}.pt"
        save_grid_frame(
            gaussian_dir / tensor_name, gaussians,
            grid_shape_lhw=grid_shape,
            image_size_wh=(manifest["width"], manifest["height"]),
            focal_length_px=manifest["focalLengthPx"],
            frame_index=record["index"], fps=manifest["fps"],
            source={"kind": "ply_recovery", "file": record["full"]},
        )
        record["tensor"] = tensor_name
        record["tensorBytes"] = (gaussian_dir / tensor_name).stat().st_size
        write_json_atomic(manifest_path, manifest)
        print(f"Recovered {record['index'] + 1}/{len(manifest['frames'])}: {tensor_name}", flush=True)
    manifest["gaussianGrid"] = {"format": FORMAT, "shape": list(grid_shape)}
    manifest["totalTensorBytes"] = sum(frame["tensorBytes"] for frame in manifest["frames"])
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--gaussian-dir", required=True)
    args = parser.parse_args()
    recover(args.repo, args.gaussian_dir)
