"""Standalone decoder: no source tracks, PLY, RGB or depth are read here."""
import time
from pathlib import Path

import numpy as np
import torch

from .legacy_format import write_header
from .model import load_package, local_times, physical_attributes
from .state import State, atomic_json


@torch.no_grad()
def decode(package_path, output_dir, device="cuda"):
    output = Path(output_dir)
    state = State(output)
    state.update(status="running", stage="decoding", currentFrame=0)
    torch.set_num_threads(8)
    package, network = load_package(package_path, device)
    codes = package["codes"].to(device)
    mean, std = package["mean"].to(device), package["std"].to(device)
    life = package["lifecycles"].numpy()
    header = dict(package["header"])
    header["codec"] = {"format": package["format"], "sampleEncoding": "decoded attributes at original timestamps"}
    times = np.asarray(header["timesSeconds"], dtype=np.float64)
    cells, layers = header["cellCount"], header["layerCount"]
    temporary = output / "animation.4dgs.tmp"
    durations = []
    total_started = time.perf_counter()
    micro = package["config"]["micro_batch_size"]
    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    with temporary.open("wb") as stream:
        offset = write_header(stream, header)
        for frame in range(header["frameCount"]):
            if state.stopped():
                raise InterruptedError("Stopped while decoding")
            active = np.flatnonzero((life[:, 0] <= frame) & (frame < life[:, 1]))
            if len(active) != cells or not np.array_equal(np.sort(life[active, 2]), np.arange(cells)):
                raise ValueError(f"Invalid active lifecycle slots at frame {frame}")
            ids = np.empty(cells, dtype="<u4")
            ids[life[active, 2]] = active
            # Output order is layer-major, identical to the source v5 format.
            identities = np.concatenate([ids.astype(np.int64) * layers + layer for layer in range(layers)])
            tau = np.tile(local_times(ids, frame, life, times), (layers, 1))
            attributes = np.empty((cells * layers, 14), dtype="<f4")
            synchronize()
            started = time.perf_counter()
            begin = 0
            while begin < len(identities):
                end = min(begin + micro, len(identities))
                try:
                    code = codes[torch.from_numpy(identities[begin:end]).to(device)]
                    t = torch.from_numpy(tau[begin:end]).to(device)
                    predicted = physical_attributes(network(code, t), mean, std)
                    attributes[begin:end] = predicted.cpu().numpy()
                    begin = end
                except torch.cuda.OutOfMemoryError:
                    if micro == 1:
                        raise
                    code = t = predicted = None
                    torch.cuda.empty_cache()
                    micro = max(1, micro // 2)
            synchronize()
            durations.append(time.perf_counter() - started)
            if not np.isfinite(attributes).all():
                raise FloatingPointError(f"Decoded attributes are not finite at frame {frame}")
            ids.tofile(stream)
            attributes.tofile(stream)
            state.update(currentFrame=frame + 1, frameCount=header["frameCount"], decodeSeconds=sum(durations),
                         decodeFPS=len(durations) / sum(durations))
        lifetimes = np.column_stack((np.arange(len(life)), life[:, :2])).astype("<u4")
        lifetimes.tofile(stream)
    final_path = output / "animation.4dgs"
    temporary.replace(final_path)
    manifest = {**header, "file": {"file": final_path.name, "bytes": final_path.stat().st_size, "dataOffset": offset}}
    atomic_json(output / "preview.json", manifest)
    state.update(decodeComplete=True, preview=manifest, decodeSeconds=sum(durations),
                 decodeFPS=len(durations) / sum(durations), perFrameDecodeSeconds=durations,
                 decodePreparationSeconds=time.perf_counter() - total_started,
                 previewCacheBytes=final_path.stat().st_size, stage="decoded")
