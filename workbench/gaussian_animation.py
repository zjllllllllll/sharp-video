"""Standalone, lossless float32 Gaussian parameter sequences (.4dgs).

This is a project-specific container, not an industry-standard 4DGS format.
An initial parameter table is followed by absolute parameter samples, in fixed
ID order. Absolute samples avoid delta subtraction/addition rounding errors.
"""

import json
import struct
from pathlib import Path

import numpy as np

FORMAT = "sharp.gaussian_tracks.v4"
MAGIC = b"SH4DGS01"
COLUMNS = "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
RECORD_BYTES = len(COLUMNS) * 4


def read_ply_parameters(path):
    """Read original binary float32 fields without re-encoding any values."""
    fields, count, vertex, little_endian = [], 0, False, False
    with open(path, "rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            words = line.decode("ascii").strip().split()
            if words == ["end_header"]:
                offset = stream.tell()
                break
            if words[:1] == ["format"]:
                little_endian = words[1:] == ["binary_little_endian", "1.0"]
            if words[:1] == ["element"]:
                vertex = words[1] == "vertex"
                if vertex:
                    count = int(words[2])
            elif vertex and words[:1] == ["property"]:
                if words[1] != "float":
                    raise ValueError("Expected float32 Gaussian vertex properties")
                fields.append(words[2])
    if not little_endian or not count or any(name not in fields for name in COLUMNS):
        raise ValueError("Unsupported Gaussian PLY layout")
    source = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count, len(fields)))
    try:
        return np.ascontiguousarray(source[:, [fields.index(name) for name in COLUMNS]])
    finally:
        source._mmap.close()


def gather_parameters(source, indices, layers):
    """Same whole-record lookup and transparent placeholders as indexed-ply.js."""
    cells = len(indices)
    if source.shape != (layers * cells, len(COLUMNS)):
        raise ValueError("PLY vertex count does not match the trajectory grid")
    if np.any((indices < -1) | (indices >= cells)):
        raise ValueError("Source cell index is out of bounds")
    valid = indices >= 0
    result = np.empty((layers, cells, len(COLUMNS)), dtype="<f4")
    for layer in range(layers):
        result[layer] = source[layer * cells + np.maximum(indices, 0)]
        result[layer, ~valid] = 0
        result[layer, ~valid, 2] = 1  # finite center in front of the camera
        result[layer, ~valid, 6] = -np.inf  # sigmoid(-inf) == 0
        result[layer, ~valid, 10] = 1  # identity quaternion, wxyz
    return result.reshape(-1, len(COLUMNS))


def make_header(tracks, times):
    fields = ["generatedAt", "frameCount", "trackCount", "cellCount", "layerCount", "layers",
              "fps", "frameIndices", "gridShapeHW", "imageSizeWH", "intrinsics", "coordinateSystem",
              "lengthUnit", "idDefinition", "validDefinition", "validCounts", "cameraAssumption"]
    header = {name: tracks[name] for name in fields if name in tracks}
    frame_bytes = tracks["trackCount"] * RECORD_BYTES
    header.update({
        "format": FORMAT, "status": "complete", "positionEncoding": "full_attribute_samples",
        "container": "SH4DGS01; project-specific .4dgs; no external files required",
        "dtype": "<f4", "columns": COLUMNS, "recordBytes": RECORD_BYTES,
        "frameBytes": frame_bytes, "timesSeconds": np.asarray(times).tolist(),
        "reference": {"offset": 0, "shape": [tracks["trackCount"], len(COLUMNS)]},
        "samples": {"offset": frame_bytes, "shape": [tracks["frameCount"] - 1, tracks["trackCount"], len(COLUMNS)]},
        "offsetConvention": "bytes relative to aligned data start; frame t begins at t*frameBytes",
        "sampleEncoding": "absolute values at discrete times; no accumulated deltas or interpolation",
        "parameterEncoding": {"position": "camera XYZ, meter", "color": "PLY SH0 from sRGB",
                              "scale": "natural log", "rotation": "wxyz quaternion", "opacity": "logit"},
        "visibilityEncoding": "stopped IDs use finite placeholders and opacity logit -Infinity",
        "compression": "none", "qualityFiltering": False,
    })
    return header


def write_header(stream, header):
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    stream.write(struct.pack("<8sII", MAGIC, len(encoded), 0))
    stream.write(encoded)
    stream.write(b"\0" * ((-stream.tell()) % 8))
    return stream.tell()


def read_container_header(path):
    """Shared SH4DGS01 prefix; version-specific layouts are described by JSON."""
    with open(path, "rb") as stream:
        magic, length, flags = struct.unpack("<8sII", stream.read(16))
        if magic != MAGIC or flags != 0:
            raise ValueError("Unsupported Gaussian animation container")
        header = json.loads(stream.read(length).decode("utf-8"))
    offset = (16 + length + 7) // 8 * 8
    return header, offset


def read_animation(path):
    """Return self-contained metadata and mmap [time, fixed ID, parameter]."""
    header, offset = read_container_header(path)
    if header["format"] != FORMAT or header["columns"] != COLUMNS or header["dtype"] != "<f4":
        raise ValueError("Unsupported Gaussian animation parameter layout")
    shape = (header["frameCount"], header["trackCount"], len(COLUMNS))
    expected = offset + header["frameCount"] * header["frameBytes"]
    if Path(path).stat().st_size != expected:
        raise ValueError("Gaussian animation file length does not match its header")
    return header, np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=shape)
