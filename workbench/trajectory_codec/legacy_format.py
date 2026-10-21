"""Legacy file I/O for the separate trajectory-codec experiment, not a pipeline step."""
import json
import struct
from pathlib import Path
import numpy as np
FORMAT = "sharp.gaussian_tracks.v4"
MAGIC = b"SH4DGS01"
COLUMNS = "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
RECORD_BYTES = len(COLUMNS) * 4


def make_header(tracks, times):
    fields = ["generatedAt", "frameCount", "trackCount", "cellCount", "layerCount", "layers",
              "fps", "frameIndices", "gridShapeHW", "imageSizeWH", "intrinsics", "coordinateSystem",
              "lengthUnit", "idDefinition", "validDefinition", "validCounts", "cameraAssumption"]
    header = {name: tracks[name] for name in fields if name in tracks}
    frame_bytes = tracks["trackCount"] * RECORD_BYTES
    header.update({
        "format": V4_FORMAT, "status": "complete", "positionEncoding": "full_attribute_samples",
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
    if header["format"] != V4_FORMAT or header["columns"] != COLUMNS or header["dtype"] != "<f4":
        raise ValueError("Unsupported Gaussian animation parameter layout")
    shape = (header["frameCount"], header["trackCount"], len(COLUMNS))
    expected = offset + header["frameCount"] * header["frameBytes"]
    if Path(path).stat().st_size != expected:
        raise ValueError("Gaussian animation file length does not match its header")
    return header, np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=shape)

V4_FORMAT = FORMAT
FORMAT = "sharp.gaussian_tracks.v5"

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
