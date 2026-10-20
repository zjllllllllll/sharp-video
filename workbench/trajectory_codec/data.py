"""Read the existing v5 sequence; retain lifecycle identity and original slots."""
import numpy as np
import torch

from gaussian_lifecycle import read_lifecycle_animation
from .model import STANDARD_COLUMNS, local_times


def prepare(source, progress=lambda *_: None, stop=lambda: False):
    header, frames, lifetimes = read_lifecycle_animation(source)
    try:
        count = len(lifetimes)
        if not np.array_equal(lifetimes[:, 0], np.arange(count)):
            raise ValueError("Expected consecutive cell lifecycle IDs")
        life = np.empty((count, 3), dtype=np.int32)
        life[:, :2], life[:, 2] = lifetimes[:, 1:], -1
        sums = np.zeros(9, dtype=np.float64)
        squares = np.zeros(9, dtype=np.float64)
        samples = 0
        for frame in range(header["frameCount"]):
            if stop():
                raise InterruptedError("Stopped while preparing data")
            ids = frames[frame]["ids"].astype(np.int64)
            newborn = life[ids, 0] == frame
            life[ids[newborn], 2] = np.flatnonzero(newborn)
            if not (np.array_equal(life[ids, 2], np.arange(header["cellCount"])) and
                    np.all((life[ids, 0] <= frame) & (frame < life[ids, 1]))):
                raise ValueError(f"Lifecycle/slot mismatch in frame {frame}")
            values = frames[frame]["parameters"]
            for begin in range(0, len(values), 65536):
                block = values[begin:begin + 65536]
                x = block[:, STANDARD_COLUMNS].astype(np.float64)
                if not np.isfinite(x).all() or not np.isfinite(block[:, 10:14]).all() or np.isnan(block[:, 6]).any():
                    raise ValueError(f"Invalid Gaussian attributes in frame {frame}")
                sums += x.sum(0)
                squares += np.square(x).sum(0)
                samples += len(x)
            progress(frame + 1, header["frameCount"])
        if np.any(life[:, 2] < 0):
            raise ValueError("Missing lifecycle birth slot")
        mean = sums / samples
        std = np.sqrt(np.maximum(squares / samples - mean * mean, 0))
        std[std == 0] = 1
        return dict(header=header, lifecycles=torch.from_numpy(life),
                    mean=torch.tensor(mean, dtype=torch.float32),
                    std=torch.tensor(std, dtype=torch.float32), sample_count=samples)
    finally:
        frames._mmap.close()
        lifetimes._mmap.close()


def batches_per_epoch(header, batch_size):
    return header["frameCount"] * ((header["trackCount"] + batch_size - 1) // batch_size)


def epoch_batches(header, frames, lifecycles, config, epoch, skip=0):
    """Exactly one visit per Gaussian-frame sample; stateless order permits resume."""
    times = np.asarray(header["timesSeconds"], dtype=np.float64)
    order = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch])).permutation(header["frameCount"])
    cells, layers = header["cellCount"], header["layerCount"]
    index = 0
    per_frame = (header["trackCount"] + config["batch_size"] - 1) // config["batch_size"]
    for frame in order:
        if index + per_frame <= skip:
            index += per_frame
            continue
        rows = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, int(frame), 1])).permutation(header["trackCount"])
        frame_ids = frames[frame]["ids"]
        parameters = frames[frame]["parameters"]
        for begin in range(0, len(rows), config["batch_size"]):
            if index < skip:
                index += 1
                continue
            selected = rows[begin:begin + config["batch_size"]]
            ids = frame_ids[selected % cells].astype(np.int64)
            gaussian_ids = ids * layers + selected // cells
            tau = local_times(ids, int(frame), lifecycles, times)
            yield index, int(frame), gaussian_ids, tau, parameters[selected].copy()
            index += 1
