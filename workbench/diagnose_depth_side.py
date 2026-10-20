import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


INTERNAL_SHAPE = (1536, 1536)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--depth-npz", required=True)
    parser.add_argument("--external-ply", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame", type=int, default=49)
    return parser.parse_args()


def describe(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "min": float(finite.min()),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def linear_to_srgb(colors):
    colors = np.clip(colors, 0.0, 1.0)
    return np.where(
        colors <= 0.0031308,
        12.92 * colors,
        1.055 * np.power(colors, 1.0 / 2.4) - 0.055,
    )


def gaussians_to_cpu(gaussians):
    return type(gaussians)(
        mean_vectors=gaussians.mean_vectors.cpu(),
        singular_values=gaussians.singular_values.cpu(),
        quaternions=gaussians.quaternions.cpu(),
        colors=gaussians.colors.cpu(),
        opacities=gaussians.opacities.cpu(),
    )


def cloud_arrays(gaussians, max_points=180_000):
    xyz = gaussians.mean_vectors[0].cpu().numpy()
    rgb = linear_to_srgb(gaussians.colors[0].cpu().numpy())
    opacity = gaussians.opacities[0].cpu().numpy()
    keep = np.flatnonzero(opacity > 0.05)
    if len(keep) > max_points:
        keep = keep[np.linspace(0, len(keep) - 1, max_points, dtype=np.int64)]
    return xyz[keep], rgb[keep]


def plot_side(ax, gaussians, title):
    xyz, rgb = cloud_arrays(gaussians)
    z_low, z_high = np.percentile(xyz[:, 2], [0.5, 99.5])
    y_low, y_high = np.percentile(-xyz[:, 1], [0.5, 99.5])
    ax.scatter(xyz[:, 2], -xyz[:, 1], c=rgb, s=0.15, alpha=0.35, linewidths=0)
    ax.set_xlim(z_low, z_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("Z / depth (m)")
    ax.set_ylabel("-Y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.15)


def plot_top(ax, gaussians, title):
    xyz, rgb = cloud_arrays(gaussians)
    x_low, x_high = np.percentile(xyz[:, 0], [0.5, 99.5])
    z_low, z_high = np.percentile(xyz[:, 2], [0.5, 99.5])
    ax.scatter(xyz[:, 0], xyz[:, 2], c=rgb, s=0.15, alpha=0.35, linewidths=0)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(z_low, z_high)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z / depth (m)")
    ax.set_title(title)
    ax.grid(alpha=0.15)


def plot_near_side(ax, gaussians, title):
    xyz, rgb = cloud_arrays(gaussians, max_points=300_000)
    near = (xyz[:, 2] >= 1.0) & (xyz[:, 2] <= 8.0)
    xyz = xyz[near]
    rgb = rgb[near]
    y_low, y_high = np.percentile(-xyz[:, 1], [0.5, 99.5])
    ax.scatter(xyz[:, 2], -xyz[:, 1], c=rgb, s=0.35, alpha=0.5, linewidths=0)
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("Z / depth (m)")
    ax.set_ylabel("-Y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.15)


@torch.inference_mode()
def predict_native(predictor, image, focal_px, device, unproject_gaussians):
    height, width = image.shape[:2]
    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(image[:, :, :3]))
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        / 255.0
    )
    image_resized = F.interpolate(
        image_tensor[None],
        size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
        mode="bilinear",
        align_corners=True,
    )
    disparity_factor = torch.tensor([focal_px / width], device=device)
    gaussians_ndc = predictor(image_resized, disparity_factor)
    intrinsics = torch.tensor(
        [
            [focal_px, 0, width / 2, 0],
            [0, focal_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics[0] *= INTERNAL_SHAPE[0] / width
    intrinsics[1] *= INTERNAL_SHAPE[1] / height
    return unproject_gaussians(
        gaussians_ndc,
        torch.eye(4, device=device),
        intrinsics,
        INTERNAL_SHAPE,
    )


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "src"))

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import load_ply, save_ply, unproject_gaussians
    from sharp.utils.io import convert_focallength

    reader = imageio.get_reader(args.video)
    image = reader.get_data(args.frame)[:, :, :3]
    reader.close()
    depths = np.load(args.depth_npz)["depths"].astype(np.float32)
    depth = depths[args.frame]
    height, width = image.shape[:2]
    focal_px = float(convert_focallength(width, height, 30.0))

    native_path = output_dir / f"frame_{args.frame:05d}_sharp_native.ply"
    external_gaussians, _ = load_ply(Path(args.external_ply))
    if native_path.is_file():
        native_gaussians, _ = load_ply(native_path)
    else:
        device = torch.device("cuda")
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict, strict=True)
        predictor.eval().to(device)
        native_gaussians = gaussians_to_cpu(
            predict_native(predictor, image, focal_px, device, unproject_gaussians)
        )
        save_ply(native_gaussians, focal_px, (height, width), native_path)

    external_z = external_gaussians.mean_vectors[0, :, 2].numpy().reshape(2, 768, 768)
    native_z = native_gaussians.mean_vectors[0, :, 2].numpy().reshape(2, 768, 768)
    depth_768 = (
        F.interpolate(
            torch.from_numpy(depth)[None, None],
            size=(768, 768),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        .numpy()
    )
    valid = np.isfinite(depth_768) & np.isfinite(external_z[0]) & np.isfinite(native_z[0])
    temporal_relative_change = np.abs(np.diff(depths, axis=0)) / np.maximum(
        depths[:-1], 1e-3
    )
    metrics = {
        "frame": args.frame,
        "assumedFocalLengthPx": focal_px,
        "inputDepth": describe(depth),
        "externalAlignedPlyLayer0Z": describe(external_z[0]),
        "sharpNativePlyLayer0Z": describe(native_z[0]),
        "inputToExternalZCorrelation": float(
            np.corrcoef(depth_768[valid], external_z[0][valid])[0, 1]
        ),
        "inputToNativeZCorrelation": float(
            np.corrcoef(depth_768[valid], native_z[0][valid])[0, 1]
        ),
        "externalVsNativeZCorrelation": float(
            np.corrcoef(external_z[0][valid], native_z[0][valid])[0, 1]
        ),
        "externalVsNativeMedianAbsRelativeDifference": float(
            np.median(
                np.abs(external_z[0][valid] - native_z[0][valid])
                / np.maximum(native_z[0][valid], 1e-3)
            )
        ),
        "adjacentDepthMedianRelativeChange": float(np.median(temporal_relative_change)),
        "adjacentDepthP95RelativeChange": float(
            np.percentile(temporal_relative_change, 95)
        ),
        "nativePly": native_path.name,
    }
    (output_dir / "depth_side_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"RGB frame {args.frame}")
    axes[0, 0].axis("off")
    depth_image = axes[1, 0].imshow(
        depth, cmap="magma", vmin=np.percentile(depth, 1), vmax=np.percentile(depth, 99)
    )
    axes[1, 0].set_title("Metric Video Depth Anything input")
    axes[1, 0].axis("off")
    figure.colorbar(depth_image, ax=axes[1, 0], fraction=0.046, label="meters")
    plot_side(axes[0, 1], external_gaussians, "Side: external predicted depth alignment")
    plot_side(axes[0, 2], native_gaussians, "Side: SHARP native inference")
    plot_top(axes[1, 1], external_gaussians, "Top: external predicted depth alignment")
    plot_top(axes[1, 2], native_gaussians, "Top: SHARP native inference")
    figure.suptitle(
        "Same RGB frame, all 1,179,648 Gaussians preserved (visualization only is thinned)",
        fontsize=14,
    )
    figure.savefig(output_dir / "depth_side_comparison.png", dpi=180)
    plt.close(figure)

    close_figure, close_axes = plt.subplots(
        1, 2, figsize=(16, 7), constrained_layout=True
    )
    plot_near_side(
        close_axes[0], external_gaussians, "Near-field side: external predicted depth"
    )
    plot_near_side(close_axes[1], native_gaussians, "Near-field side: SHARP native")
    close_figure.suptitle(
        "Near subject/ground geometry, full PLY preserved (display points thinned only)",
        fontsize=14,
    )
    close_figure.savefig(output_dir / "subject_side_comparison.png", dpi=180)
    plt.close(close_figure)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
