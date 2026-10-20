"""Export complete per-frame Gaussians with flow-linked trajectory lifetimes."""

import argparse
import json
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gaussian_animation import gather_parameters, read_ply_parameters, write_header
from gaussian_grid import load_grid_frame
from gaussian_lifecycle import FORMAT, LifecycleTrajectoryState, make_lifecycle_header
from gaussian_worker import write_json_atomic


@torch.inference_mode()
def run(args):
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else run_dir / "tracks-lifecycle"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "tracks.json"
    temporary = output / "animation.4dgs.tmp"
    final_path = output / "animation.4dgs"
    state = {"format": FORMAT, "positionEncoding": "full_attribute_samples", "status": "running",
             "stage": "tracks", "currentFrame": 0, "generatedAt": datetime.now(timezone.utc).isoformat(),
             "birthCounts": [], "endedCounts": []}
    started = time.time()

    def progress(stage, frame):
        state.update(stage=stage, currentFrame=frame, elapsedSeconds=round(time.time() - started, 3))
        write_json_atomic(state_path, state)
        print(json.dumps({"stage": stage, "frame": frame, "total": state.get("frameCount")}), flush=True)

    progress("tracks", 0)
    try:
        gaussian_dir, flow_dir = run_dir / "gaussians", run_dir / "flow"
        gaussians = json.loads((gaussian_dir / "gaussian.json").read_text(encoding="utf-8"))
        flow = json.loads((flow_dir / "flow.json").read_text(encoding="utf-8"))
        records = gaussians["frames"]
        pairs = {(p["sourceFrame"], p["targetFrame"]): p for p in flow["pairs"]}
        grid = load_grid_frame(gaussian_dir / records[0]["tensor"])
        layers, height, width = grid["grid_shape_lhw"]
        cells, frames = height * width, len(records)
        frame_indices = [record["index"] for record in records]
        fps = float(gaussians["fps"])
        track = LifecycleTrajectoryState(grid, "cuda" if torch.cuda.is_available() else "cpu")
        state.update(frameCount=frames, trackCount=layers * cells, cellCount=cells, layerCount=layers,
                     layers=list(range(layers)), fps=fps, frameIndices=frame_indices, gridShapeHW=[height, width],
                     imageSizeWH=grid["image_size_wh"], intrinsics=grid["intrinsics"].tolist(),
                     coordinateSystem=grid["coordinate_system"], lengthUnit="meter",
                     cameraAssumption="fixed camera and fixed grid/image layout", validCounts=[layers * cells] * frames)
        del grid

        # Temporary correspondence rows keep the final file self-contained while
        # allowing its header to contain the final lifetime count and offsets.
        with tempfile.TemporaryFile(dir=output) as cache_file:
            cache = np.memmap(cache_file, mode="w+", dtype="<u4", shape=(frames, 2, cells))
            try:
                for frame, record in enumerate(records):
                    born = cells
                    if frame:
                        pair = pairs[(frame_indices[frame - 1], record["index"])]
                        forward = np.load(flow_dir / pair["forward"], allow_pickle=False)
                        born = track.advance(torch.from_numpy(forward).permute(2, 0, 1))
                    cache[frame, 0] = track.ids.flatten().cpu().numpy()
                    cache[frame, 1] = track.array()
                    state["birthCounts"].append(born * layers)
                    state["endedCounts"].append(born * layers if frame else 0)
                    progress("tracks", frame + 1)

                lifetimes = track.lifetimes()
                header = make_lifecycle_header(state, np.asarray(frame_indices) / fps, lifetimes)
                state.update(header, status="running")
                with open(temporary, "wb") as stream:
                    data_offset = write_header(stream, header)
                    for frame, record in enumerate(records):
                        cache[frame, 0].tofile(stream)
                        source = read_ply_parameters(gaussian_dir / record["full"])
                        gather_parameters(source, cache[frame, 1].astype(np.int32), layers).tofile(stream)
                        progress("export", frame + 1)
                    lifetimes.tofile(stream)
            finally:
                cache._mmap.close()

        temporary.replace(final_path)
        state.update(status="complete", completedAt=datetime.now(timezone.utc).isoformat(),
                     file={"file": final_path.name, "bytes": final_path.stat().st_size, "dataOffset": data_offset})
        progress("export", frames)
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json_atomic(state_path, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    run(parser.parse_args())
