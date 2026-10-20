"""Adjacent-frame candidates for SHARP's first Gaussian layer.

Errors are measurements only: no quality thresholds, confidence scores, or
cross-frame IDs are assigned here. Multiple source cells can share a target.
"""

import torch
import torch.nn.functional as F


def sample_image(field, uv):
    """Bilinearly sample [C,H,W] at zero-based RGB pixel coordinates [...,2]."""
    _, height, width = field.shape
    denominator = uv.new_tensor([width - 1, height - 1])
    normalized = 2 * uv / denominator - 1
    return F.grid_sample(
        field[None], normalized[None], mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )[0].permute(1, 2, 0)


def image_contains(uv, image_size_wh):
    width, height = image_size_wh
    return (
        torch.isfinite(uv).all(dim=-1)
        & (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1)
        & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1)
    )


def build_correspondence(forward, backward, depth_source, depth_target, source_grid, target_grid):
    """Return unfiltered grid candidates and bidirectional/depth residuals.

Flow tensors are [2,H,W] in original RGB pixel units; depth is camera-Z in
meters. Ownership is mapped with the nominal grid sampling reference from
step one, not the displaced Gaussian center's projection.
"""
    _, source_height, source_width = source_grid["grid_shape_lhw"]
    _, target_height, target_width = target_grid["grid_shape_lhw"]
    device, dtype = forward.device, forward.dtype
    row, column = torch.meshgrid(
        torch.arange(source_height, device=device, dtype=dtype),
        torch.arange(source_width, device=device, dtype=dtype), indexing="ij",
    )
    source_uv = torch.stack((column, row), dim=-1)
    source_matrix = source_grid["grid_to_rgb_sample"].to(device=device, dtype=dtype)
    target_matrix = target_grid["grid_to_rgb_sample"].to(device=device, dtype=dtype)
    source_rgb = source_uv @ source_matrix[:2, :2].T + source_matrix[:2, 2]
    sampled_forward = sample_image(forward, source_rgb)
    target_rgb = source_rgb + sampled_forward
    target_uv = (target_rgb - target_matrix[:2, 2]) @ torch.linalg.inv(target_matrix[:2, :2]).T

    # Nearest cell gives a discrete candidate. Keep subpixel target_uv as well
    # so the later trajectory stage need not accumulate nearest-cell rounding.
    nearest = target_uv.round().to(torch.int64)
    rgb_in_bounds = (
        image_contains(source_rgb, source_grid["image_size_wh"])
        & image_contains(target_rgb, target_grid["image_size_wh"])
    )
    in_bounds = (
        rgb_in_bounds & torch.isfinite(target_uv).all(dim=-1)
        & (nearest[..., 0] >= 0) & (nearest[..., 0] < target_width)
        & (nearest[..., 1] >= 0) & (nearest[..., 1] < target_height)
    )
    target_index = (nearest[..., 1] * target_width + nearest[..., 0]).to(torch.int32)
    target_index = torch.where(in_bounds, target_index, -1)

    sampled_backward = sample_image(backward, target_rgb)
    fb_error = torch.linalg.vector_norm(sampled_forward + sampled_backward, dim=-1)
    fb_defined = rgb_in_bounds & torch.isfinite(sampled_backward).all(dim=-1)
    fb_error = torch.where(fb_defined, fb_error, torch.nan)

    source_z = sample_image(depth_source[None], source_rgb)[..., 0]
    target_z = sample_image(depth_target[None], target_rgb)[..., 0]
    depth_defined = (
        rgb_in_bounds & torch.isfinite(source_z) & torch.isfinite(target_z)
        & (source_z > 0) & (target_z > 0)
    )
    # Symmetric relative change, not a geometric ground-truth consistency test:
    # a real object moving towards/away from the camera also changes this value.
    depth_relative_error = (target_z - source_z).abs() / torch.maximum(source_z, target_z)
    depth_relative_error = torch.where(depth_defined, depth_relative_error, torch.nan)
    return {
        "target_grid_uv": target_uv,
        "candidate_target_index": target_index,
        "in_bounds": in_bounds,
        "fb_error_px": fb_error,
        "depth_relative_error": depth_relative_error,
    }
