"""Optical-flow grid indices, plus helpers for legacy RGB-D tracks."""

import json
from pathlib import Path

import numpy as np
import torch

from gaussian_correspondence import image_contains, sample_image


FORMAT = "sharp.gaussian_tracks.v2"
INDEXED_FORMAT = "sharp.gaussian_tracks.v3"
REFERENCE_COLUMNS = ["x", "y", "z", "scale_x", "scale_y", "scale_z",
                     "qw", "qx", "qy", "qz", "r", "g", "b", "opacity"]


class IndexedTrajectoryState:
    """Both layers share float UVs and nearest target cells, never rounded UVs.

    Leaving the RGB image or target grid terminates the path permanently.
    No depth, Gaussian attributes, confidence or quality thresholds are used.
    """

    def __init__(self, grid, device="cpu"):
        self.device = device
        _, self.height, self.width = grid["grid_shape_lhw"]
        self.size_wh = grid["image_size_wh"]
        self.matrix = grid["grid_to_rgb_sample"].to(device=device, dtype=torch.float32)
        self.inverse = torch.linalg.inv(self.matrix[:2, :2])
        row, column = torch.meshgrid(torch.arange(self.height, device=device),
                                     torch.arange(self.width, device=device), indexing="ij")
        cells = torch.stack((column, row), dim=-1).float()
        self.uv = cells @ self.matrix[:2, :2].T + self.matrix[:2, 2]
        self.tracked = torch.ones((self.height, self.width), dtype=torch.bool, device=device)
        self.indices = torch.arange(self.height * self.width, dtype=torch.int32,
                                    device=device).reshape(self.height, self.width)

    def advance(self, forward):
        query = torch.where(self.tracked[..., None], self.uv, 0)
        destination = query + sample_image(forward.to(self.device), query)
        grid_uv = (destination - self.matrix[:2, 2]) @ self.inverse.T
        nearest = torch.round(grid_uv).to(torch.int64)
        self.tracked &= image_contains(destination, self.size_wh)
        self.tracked &= ((nearest[..., 0] >= 0) & (nearest[..., 0] < self.width)
                         & (nearest[..., 1] >= 0) & (nearest[..., 1] < self.height))
        self.indices = torch.where(self.tracked, nearest[..., 1] * self.width + nearest[..., 0], -1).to(torch.int32)
        self.uv = torch.where(self.tracked[..., None], destination, torch.nan)

    def array(self):
        return self.indices.reshape(-1).cpu().numpy()


class IndexedTrackWriter:
    """Only the shared index table and timestamps are persisted for v3."""

    def __init__(self, directory, frame_count, cell_count, times):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.arrays, self.descriptors = {}, {}
        for name, filename, shape, dtype in [
            ("sourceCellIndices", "source_cell_indices.npy", (frame_count, cell_count), "<i4"),
            ("times", "times.npy", (frame_count,), "<f8"),
        ]:
            array = np.lib.format.open_memmap(directory / filename, mode="w+", dtype=dtype, shape=shape)
            self.arrays[name] = array
            self.descriptors[name] = {
                "file": filename, "dtype": array.dtype.str, "shape": list(shape),
                "dataOffset": array.offset, "dataBytes": array.nbytes,
                "frameBytes": array[0].nbytes if name == "sourceCellIndices" else None,
            }
        self.arrays["times"][:] = times

    def write_frame(self, index, indices):
        self.arrays["sourceCellIndices"][index] = indices

    def flush(self):
        for array in self.arrays.values():
            array.flush()

    def close(self):
        self.flush()
        for array in self.arrays.values():
            array._mmap.close()


def unproject(uv, depth, intrinsics):
    z = sample_image(depth[None], uv)[..., 0]
    pixels = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
    xyz = (pixels @ torch.linalg.inv(intrinsics).T) * z[..., None]
    defined = torch.isfinite(xyz).all(dim=-1) & (z > 0)
    return xyz, defined


class TrajectoryState:
    """IDs are layer,row,column array slots and are never reassigned/merged.

    UV tracking survives missing depth, but cannot resume after leaving the
    image or encountering undefined flow. Valid means computable, not reliable
    or visible: occlusion/consistency thresholds are deliberately not applied.
    """

    def __init__(self, grid, depth, device="cpu"):
        _, height, width = grid["grid_shape_lhw"]
        self.size_wh = grid["image_size_wh"]
        self.device = device
        self.intrinsics = grid["intrinsics"].to(device=device, dtype=torch.float32)
        self.initial_means = grid["gaussians"]["mean_vectors"].to(device)
        row, col = torch.meshgrid(torch.arange(height, device=device),
                                  torch.arange(width, device=device), indexing="ij")
        matrix = grid["grid_to_rgb_sample"].to(device=device, dtype=torch.float32)
        self.uv = torch.stack((col, row), dim=-1).float() @ matrix[:2, :2].T + matrix[:2, 2]
        self.tracked = image_contains(self.uv, self.size_wh)
        self.initial_xyz, depth_defined = unproject(self.uv, depth.to(device), self.intrinsics)
        self.reference_defined = self.tracked & depth_defined
        self.finite_means = torch.isfinite(self.initial_means).all(-1)
        self.motion_valid = self.reference_defined.clone()
        self.displacements = torch.where(self.motion_valid[..., None], torch.zeros_like(self.initial_xyz), torch.nan)

    @property
    def valid(self):
        return self.motion_valid[None] & self.finite_means

    @property
    def positions(self):
        return torch.where(self.valid[..., None], self.initial_means + self.displacements[None], torch.nan)

    def advance(self, forward, depth):
        # Keep subpixel positions across frames. Do not snap to the candidate
        # cell indices saved for inspection by the adjacent-pair stage.
        query = torch.where(self.tracked[..., None], self.uv, 0)
        destination = query + sample_image(forward.to(self.device), query)
        self.tracked &= image_contains(destination, self.size_wh)
        self.uv = torch.where(self.tracked[..., None], destination, torch.nan)
        query = torch.where(self.tracked[..., None], self.uv, 0)
        xyz, depth_defined = unproject(query, depth.to(self.device), self.intrinsics)
        self.motion_valid = self.tracked & self.reference_defined & depth_defined
        # Equivalent to accumulating adjacent 3D displacements where defined;
        # retaining the reference also permits recovery after missing depth.
        self.displacements = torch.where(self.motion_valid[..., None], xyz - self.initial_xyz, torch.nan)

    def arrays(self):
        return {
            "uv": self.uv.reshape(-1, 2).cpu().numpy(),
            "displacements": self.displacements.reshape(-1, 3).cpu().numpy(),
            "valid": self.motion_valid.reshape(-1).cpu().numpy(),
        }


def reference_attributes(grid):
    attributes = grid["gaussians"]
    return torch.cat([
        attributes[name].reshape(-1, channels)
        for name, channels in [("mean_vectors", 3), ("singular_values", 3),
                               ("quaternions", 4), ("colors", 3), ("opacities", 1)]
    ], dim=-1).numpy().astype(np.float32)


class TrackWriter:
    """NPY arrays are portable and permit frame-at-a-time reads via mmap/HTTP."""

    def __init__(self, directory, frame_count, track_count, reference, times, *, motion_count):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.arrays = {}
        specs = {
            "ids": ((track_count,), "<i4"), "times": ((frame_count,), "<f8"),
            "reference": ((track_count, 14), "<f4"),
            "uv": ((frame_count, motion_count, 2), "<f4"),
            "displacements": ((frame_count, motion_count, 3), "<f4"),
            "valid": ((frame_count, motion_count), "|b1"),
        }
        self.descriptors = {}
        for name, (shape, dtype) in specs.items():
            array = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            self.arrays[name] = array
            self.descriptors[name] = {
                "file": f"{name}.npy", "dtype": array.dtype.str, "shape": list(shape),
                "dataOffset": array.offset, "dataBytes": array.nbytes,
                "frameBytes": array[0].nbytes if name in ("uv", "displacements", "valid") else None,
            }
        self.arrays["ids"][:] = np.arange(track_count, dtype=np.int32)
        self.arrays["times"][:] = times
        self.arrays["reference"][:] = reference

    def write_frame(self, index, arrays):
        for name, values in arrays.items():
            self.arrays[name][index] = values

    def flush(self):
        for array in self.arrays.values():
            array.flush()


def load_tracks(directory):
    """Load the saved metadata and read-only arrays without loading all frames."""
    directory = Path(directory)
    manifest = json.loads((directory / "tracks.json").read_text(encoding="utf-8"))
    arrays = {name: np.load(directory / spec["file"], mmap_mode="r", allow_pickle=False)
              for name, spec in manifest["arrays"].items()}
    return manifest, arrays
