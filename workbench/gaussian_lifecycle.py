"""Flow tracks with one owner per target cell and births in uncovered cells."""

import numpy as np
import torch

from gaussian_animation import COLUMNS, RECORD_BYTES, make_header, read_container_header
from gaussian_tracks import IndexedTrajectoryState

FORMAT = "sharp.gaussian_tracks.v5"


class LifecycleTrajectoryState(IndexedTrajectoryState):
    """Keep float UVs for survivors; initialize only newborns at grid centers.

    Render slots are reused, but trajectory IDs are monotonically allocated.
    Both layers share these cell trajectories and have separate attributes.
    """

    def __init__(self, grid, device="cpu"):
        super().__init__(grid, device)
        self.ids = self.indices.to(torch.int64).clone()
        self.birth_frames = torch.zeros_like(self.ids)
        self.next_id = self.ids.numel()
        self.frame = 0
        self.ended = []

    def advance(self, forward):
        super().advance(forward)
        self.frame += 1
        cells, ids = self.indices.flatten().long(), self.ids.flatten()
        valid = cells >= 0
        owner = torch.full_like(ids, self.next_id)
        owner.scatter_reduce_(0, cells[valid], ids[valid], reduce="amin")
        keep = valid & (ids == owner[cells.clamp_min(0)])
        vacant = (~keep).nonzero().flatten()
        missing = (owner == self.next_id).nonzero().flatten()
        count = vacant.numel()

        births = self.birth_frames.flatten()
        self.ended.append(torch.stack((ids[vacant], births[vacant],
                                      torch.full_like(vacant, self.frame)), dim=1).cpu().numpy())
        ids[vacant] = torch.arange(self.next_id, self.next_id + count, device=self.device)
        births[vacant] = self.frame
        self.indices.flatten()[vacant] = missing.to(torch.int32)
        grid_uv = torch.stack((missing % self.width, missing // self.width), dim=1).float()
        self.uv.reshape(-1, 2)[vacant] = grid_uv @ self.matrix[:2, :2].T + self.matrix[:2, 2]
        self.tracked.fill_(True)
        self.next_id += count
        return count

    def lifetimes(self):
        """[cell trajectory ID, start frame, exclusive end frame]."""
        remaining = torch.stack((self.ids.flatten(), self.birth_frames.flatten(),
                                 torch.full_like(self.ids.flatten(), self.frame + 1)), dim=1)
        rows = np.concatenate([*self.ended, remaining.cpu().numpy()]).astype("<u4")
        return rows[np.argsort(rows[:, 0])]


def make_lifecycle_header(metadata, times, lifetimes):
    header = make_header(metadata, times)
    header.pop("reference")
    header.pop("samples")
    parameter_bytes = metadata["trackCount"] * RECORD_BYTES
    parameter_offset = metadata["cellCount"] * 4
    frame_bytes = parameter_offset + parameter_bytes
    header.update({
        "format": FORMAT, "frameBytes": frame_bytes,
        "parameterOffset": parameter_offset, "parameterBytes": parameter_bytes,
        "totalTrackCount": len(lifetimes) * metadata["layerCount"],
        "cellTrackIds": {"offset": 0, "dtype": "<u4", "shape": [metadata["cellCount"]]},
        "parameters": {"offset": parameter_offset, "shape": [metadata["trackCount"], len(COLUMNS)]},
        "lifetimes": {"offset": len(times) * frame_bytes, "dtype": "<u4",
                      "shape": list(lifetimes.shape), "columns": ["cellTrackId", "startFrame", "endFrameExclusive"]},
        "idDefinition": "Gaussian ID = cellTrackIds[slot]*layerCount+layer; IDs never reused",
        "slotDefinition": "parameters[layer*cellCount+slot]; slots reused when a new trajectory is born",
        "collisionRule": "smallest cell trajectory ID survives; other trajectories end before this frame",
        "birthRule": "each uncovered target cell starts a new trajectory, in ascending cell order",
        "visibilityEncoding": "every source Gaussian appears once per frame; no transparent placeholders",
        "validDefinition": "complete source-cell coverage, not verified physical identity or visibility",
        "birthCounts": metadata["birthCounts"], "endedCounts": metadata["endedCounts"],
    })
    return header


def read_lifecycle_animation(path):
    """Read a standalone v5 file as frame records and trajectory lifetimes."""
    header, offset = read_container_header(path)
    if header["format"] != FORMAT:
        raise ValueError("Expected lifecycle animation v5")
    dtype = np.dtype([("ids", "<u4", (header["cellCount"],)),
                      ("parameters", "<f4", (header["trackCount"], len(COLUMNS)))])
    frames = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(header["frameCount"],))
    spec = header["lifetimes"]
    lifetimes = np.memmap(path, dtype="<u4", mode="r", offset=offset + spec["offset"], shape=tuple(spec["shape"]))
    return header, frames, lifetimes
