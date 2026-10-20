"""Subprocess-isolated training, standalone decoding and full evaluation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from trajectory_codec.state import State


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    state = State(run_dir / "trajectory-codec")
    state.update(status="running", pid=os.getpid(), error=None)
    for stage, flag in [("train", "exportComplete"), ("decode", "decodeComplete"), ("evaluate", "evaluationComplete")]:
        state = State(state.output)
        if state.stopped():
            state.update(status="stopped")
            return
        if state.value.get(flag):
            continue
        args = [sys.executable, "-u", __file__, stage]
        if stage == "decode":
            args += ["--package", str(state.output / "trajectory-codec.pt"), "--output-dir", str(state.output)]
        else:
            args += ["--run-dir", str(run_dir)]
        result = subprocess.run(args, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        current = State(state.output)
        if result.returncode or current.value.get("status") in {"failed", "stopped"}:
            if result.returncode and current.value.get("status") != "failed":
                current.update(status="failed", error=f"{stage} exited with code {result.returncode}")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "train", "decode", "evaluate"])
    parser.add_argument("--run-dir")
    parser.add_argument("--package")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output_dir) if args.output_dir else Path(args.run_dir) / "trajectory-codec"
    try:
        if args.command == "run":
            run(args.run_dir)
        elif args.command == "train":
            from trajectory_codec.training import train
            train(args.run_dir)
        elif args.command == "decode":
            from trajectory_codec.decode import decode
            decode(args.package, output, args.device)
        else:
            from trajectory_codec.evaluation import evaluate
            evaluate(args.run_dir)
    except InterruptedError:
        State(output).update(status="stopped")
    except Exception as error:
        State(output).update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)
