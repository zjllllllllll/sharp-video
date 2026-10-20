"""Compute RGB optical flow and first-layer Gaussian candidates for a run."""

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from gaussian_correspondence import build_correspondence
from gaussian_grid import load_grid_frame
from gaussian_worker import write_json_atomic


WEIGHTS = Raft_Small_Weights.C_T_V2
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/raft_small_C_T_V2-01064c6d.pth"


def prepare_image(image, device):
    image = torch.from_numpy(np.ascontiguousarray(image[..., :3])).permute(2, 0, 1)[None]
    # torchvision's official uint8 RGB -> float [-1,1] transform.
    image, _ = WEIGHTS.transforms()(image, image)
    height, width = image.shape[-2:]
    # Pad only: RGB pixel coordinates and vector magnitudes stay unchanged.
    return F.pad(image.to(device), (0, (-width) % 8, 0, (-height) % 8), mode="replicate")


@torch.inference_mode()
def infer_flow(model, image_source, image_target, image_size_wh):
    width, height = image_size_wh
    return model(image_source, image_target)[-1][0, :, :height, :width].cpu().contiguous()


def error_summary(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"count": 0, "median": None, "p95": None}
    return {"count": int(finite.size), "median": float(np.median(finite)), "p95": float(np.percentile(finite, 95))}


def save_pair(output_dir, source_record, target_record, forward, backward, correspondence):
    source_index, target_index = source_record["index"], target_record["index"]
    forward_name = f"forward_{source_index:05d}.npy"
    backward_name = f"backward_{target_index:05d}.npy"
    candidates_name = f"correspondence_{source_index:05d}_{target_index:05d}.npz"
    np.save(output_dir / forward_name, forward.permute(1, 2, 0).numpy())
    np.save(output_dir / backward_name, backward.permute(1, 2, 0).numpy())
    arrays = {key: value.cpu().numpy() for key, value in correspondence.items()}
    np.savez_compressed(output_dir / candidates_name, **arrays)
    return {
        "sourceFrame": source_index, "targetFrame": target_index,
        "sourceTensor": source_record["tensor"], "targetTensor": target_record["tensor"],
        "forward": forward_name, "backward": backward_name, "correspondence": candidates_name,
        "candidateCount": int(arrays["in_bounds"].size),
        "inBoundsCount": int(arrays["in_bounds"].sum()),
        "fbErrorPx": error_summary(arrays["fb_error_px"]),
        "depthRelativeError": error_summary(arrays["depth_relative_error"]),
    }


def run(args):
    run_dir = Path(args.run_dir).resolve()
    gaussian_dir = run_dir / "gaussians"
    manifest = json.loads((gaussian_dir / "gaussian.json").read_text(encoding="utf-8"))
    depth_meta = json.loads((run_dir / "depth.json").read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "flow"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "flow.json"
    records = manifest["frames"]
    if args.max_pairs > 0:
        records = records[:args.max_pairs + 1]
    size_wh = (manifest["width"], manifest["height"])
    source_video = run_dir / depth_meta["sourceVideo"]
    depth_path = run_dir / depth_meta["depthData"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = {
        "format": "sharp.gaussian_correspondence.v1", "status": "running",
        "model": "torchvision RAFT Small", "weights": WEIGHTS.name,
        "checkpoint": str(Path(args.checkpoint).resolve()), "device": str(device),
        "sourceVideo": str(source_video), "depthData": str(depth_path),
        "gaussianManifest": str(gaussian_dir / "gaussian.json"),
        "fps": manifest["fps"], "width": size_wh[0], "height": size_wh[1],
        "layer": 0, "flowLayout": "H,W,(dx,dy)", "flowUnits": "RGB pixels",
        "gridCoordinates": "grid_to_rgb_sample; zero-based (column,row)",
        "candidateIndex": "nearest target cell on layer 0; row*grid_width+column; -1 outside bounds",
        "depthErrorDefinition": "abs(D_target(q)-D_source(p))/max(D_source(p),D_target(q))",
        "undefinedErrors": "NaN", "qualityFiltering": False,
        "modelConfidence": None, "currentPair": 0, "totalPairs": len(records) - 1,
        "pairs": [], "startedAt": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(state_path, state)
    started = time.time()
    reader = None
    try:
        model = raft_small(weights=None)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        model = model.to(device).eval()
        with np.load(depth_path) as depth_archive:
            depths = depth_archive["depths"]
        reader = imageio.get_reader(source_video)
        source_record = records[0]
        source_grid = load_grid_frame(gaussian_dir / source_record["tensor"])
        source_image = prepare_image(reader.get_data(source_record["index"]), device)
        print(json.dumps({"event": "started", "total": len(records) - 1}), flush=True)
        for pair_number, target_record in enumerate(records[1:], start=1):
            pair_started = time.time()
            target_grid = load_grid_frame(gaussian_dir / target_record["tensor"])
            target_image = prepare_image(reader.get_data(target_record["index"]), device)
            forward = infer_flow(model, source_image, target_image, size_wh)
            backward = infer_flow(model, target_image, source_image, size_wh)
            correspondence = build_correspondence(
                forward, backward,
                torch.from_numpy(depths[source_record["index"]]),
                torch.from_numpy(depths[target_record["index"]]),
                source_grid, target_grid,
            )
            pair = save_pair(output_dir, source_record, target_record, forward, backward, correspondence)
            pair["seconds"] = round(time.time() - pair_started, 3)
            state["pairs"].append(pair)
            state["currentPair"] = pair_number
            state["elapsedSeconds"] = round(time.time() - started, 3)
            write_json_atomic(state_path, state)
            print(json.dumps({"event": "progress", "current": pair_number, "total": len(records) - 1, "seconds": pair["seconds"]}), flush=True)
            source_record, source_grid, source_image = target_record, target_grid, target_image
        state["status"] = "complete"
        state["completedAt"] = datetime.now(timezone.utc).isoformat()
        state["elapsedSeconds"] = round(time.time() - started, 3)
        write_json_atomic(state_path, state)
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json_atomic(state_path, state)
        raise
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-pairs", type=int, default=-1)
    run(parser.parse_args())
