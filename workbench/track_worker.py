"""Build optical-flow indices into the original per-frame, two-layer Gaussians."""

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gaussian_grid import load_grid_frame
from gaussian_tracks import INDEXED_FORMAT, IndexedTrackWriter, IndexedTrajectoryState
from gaussian_worker import write_json_atomic


@torch.inference_mode()
def run(args):
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else run_dir / "tracks-indexed"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "tracks.json"
    state = {
        "format": INDEXED_FORMAT, "positionEncoding": "source_frame_indices", "status": "running",
        "generatedAt": datetime.now(timezone.utc).isoformat(), "currentFrame": 0,
        "validCounts": [], "trackedCounts": [], "qualityFiltering": False,
    }
    write_json_atomic(state_path, state)
    started, writer = time.time(), None
    try:
        flow_dir = run_dir / "flow"
        flow = json.loads((flow_dir / "flow.json").read_text(encoding="utf-8"))
        gaussian_dir = run_dir / "gaussians"
        gaussians = json.loads((gaussian_dir / "gaussian.json").read_text(encoding="utf-8"))
        records = gaussians["frames"]
        grid = load_grid_frame(gaussian_dir / records[0]["tensor"])
        pairs = {(pair["sourceFrame"], pair["targetFrame"]): pair for pair in flow["pairs"]}
        frame_indices = [record["index"] for record in records]
        fps = float(gaussians["fps"])
        layers, height, width = grid["grid_shape_lhw"]
        cell_count = height * width
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state.update({
            "layers": list(range(layers)), "layerCount": layers, "cellCount": cell_count,
            "trackCount": layers * cell_count, "frameCount": len(records), "fps": fps,
            "frameIndices": frame_indices, "gridShapeHW": [height, width],
            "imageSizeWH": grid["image_size_wh"], "intrinsics": grid["intrinsics"].tolist(),
            "coordinateSystem": grid["coordinate_system"], "lengthUnit": "meter",
            "idDefinition": "layer*cellCount+initial_row*grid_width+initial_column; never reassigned",
            "sourceIndexDefinition": "layer*cellCount+sourceCellIndices[t,id%cellCount]; -1 terminates both layers",
            "uvDefinition": "p(t+1)=p(t)+bilinear_sample(F(t),p(t)); float RGB coordinates retained in memory",
            "lookupDefinition": "inverse grid_to_rgb_sample, torch.round (ties to even); same layer, nearest cell",
            "validDefinition": "in-image flow path with an in-bounds target cell; not visibility or correspondence confidence",
            "interpolation": "none; whole source Gaussian record copied at discrete frames",
            "cameraAssumption": "fixed camera and fixed grid/image layout",
            "dynamicAttributes": ["positions", "colors", "scales", "quaternions", "opacities"],
            "sourceFrames": [{"index": record["index"], "full": record["full"]} for record in records],
            "sourceGaussianManifest": str(gaussian_dir / "gaussian.json"),
            "sourceGrid": str(gaussian_dir / records[0]["tensor"]),
            "sourceFlow": str(flow_dir / "flow.json"), "device": device,
        })
        track = IndexedTrajectoryState(grid, device)
        del grid
        writer = IndexedTrackWriter(output, len(records), cell_count, np.asarray(frame_indices, dtype=np.float64) / fps)
        state["arrays"] = writer.descriptors
        write_json_atomic(state_path, state)
        for index, record in enumerate(records):
            (gaussian_dir / record["full"]).stat()
            if index:
                pair = pairs[(frame_indices[index - 1], record["index"])]
                forward = np.load(flow_dir / pair["forward"], allow_pickle=False)
                track.advance(torch.from_numpy(forward).permute(2, 0, 1))
            indices = track.array()
            writer.write_frame(index, indices)
            valid_count = int(np.count_nonzero(indices >= 0)) * layers
            state["validCounts"].append(valid_count)
            state["trackedCounts"].append(valid_count)
            state["currentFrame"] = index + 1
            state["elapsedSeconds"] = round(time.time() - started, 3)
            write_json_atomic(state_path, state)
            print(json.dumps({"frame": index, "total": len(records), "valid": valid_count}), flush=True)
        writer.close()
        writer = None
        state.update(status="complete", elapsedSeconds=round(time.time() - started, 3),
                     completedAt=datetime.now(timezone.utc).isoformat())
        write_json_atomic(state_path, state)
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json_atomic(state_path, state)
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    run(parser.parse_args())
