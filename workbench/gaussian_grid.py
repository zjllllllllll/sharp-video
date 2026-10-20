"""Preserve SHARP's layer/row/column layout and RGB-D coordinate mapping.

This is a per-frame algorithm cache, not a temporal Gaussian ID or 4DGS model.
"""

from pathlib import Path

import torch


NETWORK_SIZE_WH = (1536, 1536)
FORMAT = "sharp.gaussian_grid.v1"
CHANNELS = {
    "mean_vectors": 3,
    "singular_values": 3,
    "quaternions": 4,
    "colors": 3,
    "opacities": None,
}


def grid_shape_from_initializer(initializer, network_size_wh=NETWORK_SIZE_WH):
    width, height = network_size_wh
    return (initializer.num_layers, height // initializer.stride, width // initializer.stride)


def make_grid_frame(
    gaussians, *, grid_shape_lhw, image_size_wh, focal_length_px,
    frame_index, fps, source, network_size_wh=NETWORK_SIZE_WH,
):
    """Reshape without sorting, dropping, or changing Gaussian attributes.

SHARP composer.py flattens [B, C, L, H, W] as [B, L*H*W, C].
unproject_gaussians and save_ply preserve that order.
"""
    layers, grid_height, grid_width = grid_shape_lhw
    width, height = image_size_wh
    network_width, network_height = network_size_wh
    stride_x = network_width / grid_width
    stride_y = network_height / grid_height
    # RGB and depth are resized using bilinear interpolation, align_corners=True.
    # A grid cell represents a stride-sized pooling footprint. Store its nominal
    # sampling center in the ORIGINAL input array's zero-based pixel indices.
    resize_x = (width - 1) / (network_width - 1)
    resize_y = (height - 1) / (network_height - 1)
    grid_to_rgb = torch.tensor([
        [stride_x * resize_x, 0, (stride_x - 1) * 0.5 * resize_x],
        [0, stride_y * resize_y, (stride_y - 1) * 0.5 * resize_y],
        [0, 0, 1],
    ], dtype=torch.float64)
    attributes = {}
    for name, channels in CHANNELS.items():
        shape = (layers, grid_height, grid_width)
        if channels is not None:
            shape += (channels,)
        attributes[name] = getattr(gaussians, name).detach().cpu().reshape(shape).contiguous()
    return {
        "format": FORMAT,
        "source": source,
        "frame_index": int(frame_index),
        "fps": float(fps),
        "time_seconds": frame_index / fps,
        "grid_shape_lhw": list(grid_shape_lhw),
        "flatten_order": "layer,row,column",
        "image_size_wh": [int(width), int(height)],
        "network_size_wh": list(network_size_wh),
        "resize": {"mode": "bilinear", "align_corners": True, "crop": None},
        "grid_to_rgb_sample": grid_to_rgb,
        "grid_to_rgb_sample_convention": "zero_based_array_index; pooling_footprint_center",
        "depth_coordinates": "same_as_rgb; positive_camera_z",
        "coordinate_system": "camera_opencv_x_right_y_down_z_forward",
        "length_unit": "meter",
        "intrinsics": torch.tensor([
            [focal_length_px, 0, width / 2],
            [0, focal_length_px, height / 2],
            [0, 0, 1],
        ], dtype=torch.float64),
        "world_to_camera": torch.eye(4, dtype=torch.float64),
        "quaternion_order": "wxyz",
        "color_space": "linearRGB",
        "scale_encoding": "linear",
        "opacity_encoding": "linear",
        "gaussians": attributes,
    }


def save_grid_frame(path, gaussians, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = make_grid_frame(gaussians, **metadata)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(frame, temporary)
    temporary.replace(path)


def load_grid_frame(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def grid_to_gaussians(frame):
    """Return SHARP's flat structure for its existing renderer/PLY exporter."""
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(**{
        name: value.reshape(1, -1, CHANNELS[name]) if CHANNELS[name] else value.reshape(1, -1)
        for name, value in frame["gaussians"].items()
    })


def flat_index_to_grid(index, grid_shape_lhw):
    """Accept an integer or integer tensor; return (layer, row, column)."""
    _, height, width = grid_shape_lhw
    return index // (height * width), (index // width) % height, index % width


def grid_uv_to_rgb_sample(uv, frame):
    """Map grid (column,row) to the RGB/depth input sampling reference.

This is the owning cell's reference, NOT the final Gaussian's projected center:
SHARP also predicts position offsets. Use project_gaussian_centers for the latter.
"""
    uv = torch.as_tensor(uv, dtype=torch.float64)
    matrix = frame["grid_to_rgb_sample"].to(uv.device)
    return uv @ matrix[:2, :2].T + matrix[:2, 2]


def project_gaussian_centers(frame):
    """Project actual camera-space Gaussian centers using SHARP's intrinsics."""
    xyz = frame["gaussians"]["mean_vectors"]
    camera = frame["intrinsics"].to(device=xyz.device, dtype=xyz.dtype)
    projected = xyz @ camera.T
    return projected[..., :2] / projected[..., 2:3]
