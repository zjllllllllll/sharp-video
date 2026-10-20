"""Export indexed trajectories into one independent Gaussian animation file."""

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gaussian_animation import FORMAT, gather_parameters, make_header, read_ply_parameters, write_header
from gaussian_worker import write_json_atomic


def run(args):
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else run_dir / "tracks-animation"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "tracks.json"
    final_path = output / "animation.4dgs"
    temporary = output / "animation.4dgs.tmp"
    state = {"format": FORMAT, "positionEncoding": "full_attribute_samples", "status": "running",
             "stage": "export", "currentFrame": 0, "generatedAt": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(state_path, state)
    started = time.time()
    indices = times = None
    try:
        indexed = run_dir / "tracks-indexed"
        tracks = json.loads((indexed / "tracks.json").read_text(encoding="utf-8"))
        if tracks["status"] != "complete":
            raise ValueError("Generate complete indexed tracks before exporting the animation")
        indices = np.load(indexed / tracks["arrays"]["sourceCellIndices"]["file"], mmap_mode="r", allow_pickle=False)
        times = np.load(indexed / tracks["arrays"]["times"]["file"], mmap_mode="r", allow_pickle=False)
        header = make_header({**tracks, "generatedAt": state["generatedAt"]}, times)
        state.update(header, status="running", stage="export", currentFrame=0)
        write_json_atomic(state_path, state)
        with open(temporary, "wb") as stream:
            data_offset = write_header(stream, header)
            for frame, source in enumerate(tracks["sourceFrames"]):
                attributes = read_ply_parameters(run_dir / "gaussians" / source["full"])
                sample = gather_parameters(attributes, indices[frame], tracks["layerCount"])
                sample.tofile(stream)
                state.update(currentFrame=frame + 1, elapsedSeconds=round(time.time() - started, 3))
                write_json_atomic(state_path, state)
                print(json.dumps({"exportedFrame": frame + 1, "total": tracks["frameCount"]}), flush=True)
        temporary.replace(final_path)
        state.update(status="complete", elapsedSeconds=round(time.time() - started, 3),
                     completedAt=datetime.now(timezone.utc).isoformat(),
                     file={"file": final_path.name, "bytes": final_path.stat().st_size, "dataOffset": data_offset})
        write_json_atomic(state_path, state)
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json_atomic(state_path, state)
        raise
    finally:
        for array in [indices, times]:
            if array is not None:
                array._mmap.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    run(parser.parse_args())
