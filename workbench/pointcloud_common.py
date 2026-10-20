import json
from pathlib import Path

import numpy as np
from PIL import Image


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> int:
    points = np.asarray(points, dtype="<f4").reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("点坐标与颜色数量不一致")

    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as output:
        output.write(header)
        vertices.tofile(output)
    return len(vertices)


def save_depth_preview(path: Path, depth: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    finite = depth[valid]
    near, far = np.percentile(finite, [2, 98]).astype(float)
    span = max(far - near, 1e-6)
    normalized = np.clip((depth - near) / span, 0, 1)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    preview[valid] = np.round((1.0 - normalized[valid]) * 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)
    return near, far


def write_metadata(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
