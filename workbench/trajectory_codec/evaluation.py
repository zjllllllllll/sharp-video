"""Full-sequence attribute statistics and matched gsplat image comparisons."""
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from gaussian_lifecycle import read_lifecycle_animation
from dynamic.upstream import ssim  # Existing, attributed Graphdeco/MoDGS implementation.
from .model import unit_quaternion
from .state import State, atomic_json


def setup_renderer():
    """Reuse the existing Windows gsplat toolchain settings, without a motion model."""
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(root / ".cache" / "dynamic_torch_extensions"))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", "8")
    if os.name == "nt":
        if not os.environ.get("VSCMD_VER"):
            script = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
            output = subprocess.check_output(f'cmd.exe /d /s /c ""{script}" >nul && set"',
                                             text=True, errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
            for line in output.splitlines():
                if "=" in line and not line.startswith("="):
                    name, value = line.split("=", 1)
                    os.environ[name] = value
        cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
        os.environ.setdefault("CUDA_HOME", cuda)
        os.environ["PATH"] = str(Path(cuda) / "bin") + os.pathsep + os.environ["PATH"]
        from torch.utils import cpp_extension
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
        original = cpp_extension._jit_compile
        def compile_windows(*args, **kwargs):
            args = list(args)
            flags = args[2] if len(args) > 2 else kwargs.get("extra_cflags", [])
            flags = ["/O2" if flag == "-O3" else flag for flag in (flags or []) if flag != "-Wno-attributes"]
            if len(args) > 2:
                args[2] = flags
            else:
                kwargs["extra_cflags"] = flags
            return original(*args, **kwargs)
        cpp_extension._jit_compile = compile_windows


@torch.no_grad()
def render(attributes, header, camera_x=0.):
    from gsplat import rasterization
    values = torch.as_tensor(attributes.copy(), device="cuda")
    width, height = header["imageSizeWH"]
    view = torch.eye(4, device="cuda")
    view[0, 3] = -camera_x
    intrinsics = torch.tensor(header["intrinsics"], dtype=torch.float32, device="cuda")
    # Source v5 explicitly stores SH0 of sRGB. Do not apply the old linearRGB output transfer.
    colors = (0.28209479177387814 * values[:, 3:6] + .5).clamp(0, 1)
    image, _, _ = rasterization(values[:, :3].contiguous(), unit_quaternion(values[:, 10:14]).contiguous(),
                                values[:, 7:10].exp().contiguous(), values[:, 6].sigmoid().contiguous(),
                                colors.contiguous(), view[None], intrinsics[None], width, height,
                                render_mode="RGB", packed=True, rasterize_mode="classic")
    return image[0].clamp(0, 1)


def save_image(path, image):
    pixels = image.detach().cpu().numpy()
    Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(path)


def attribute_errors(original, reconstructed):
    """11 physical scalar errors per observation (RGB in [0,1], rotation in degrees)."""
    a = torch.from_numpy(original.copy()).double()
    b = torch.from_numpy(reconstructed.copy()).double()
    dot = (unit_quaternion(a[:, 10:14]) * unit_quaternion(b[:, 10:14])).sum(-1).abs().clamp(0, 1)
    return torch.cat(((a[:, :3] - b[:, :3]).abs(),
                      (a[:, 3:6] - b[:, 3:6]).abs() * 0.28209479177387814,
                      (a[:, 6].sigmoid() - b[:, 6].sigmoid()).abs()[:, None],
                      (a[:, 7:10].exp() - b[:, 7:10].exp()).abs(),
                      (2 * torch.acos(dot) * (180 / math.pi))[:, None]), dim=1).numpy().astype(np.float32)


def evaluate(run_dir):
    run_dir = Path(run_dir)
    output = run_dir / "trajectory-codec"
    state = State(output)
    state.update(status="running", stage="evaluating", currentFrame=0)
    torch.set_num_threads(8)
    setup_renderer()
    header, source, life = read_lifecycle_animation(run_dir / "tracks-lifecycle" / "animation.4dgs")
    decoded_header, decoded, decoded_life = read_lifecycle_animation(output / "animation.4dgs")
    if not np.array_equal(life, decoded_life):
        raise ValueError("Exported lifecycle table differs from source")
    total = header["frameCount"] * header["trackCount"]
    scratch = output / "attribute-errors.tmp"
    errors = np.memmap(scratch, dtype="<f4", mode="w+", shape=(11, total))
    names = ["x_m", "y_m", "z_m", "r_srgb", "g_srgb", "b_srgb", "opacity", "scale_x_m", "scale_y_m", "scale_z_m", "rotation_degrees"]
    rows = []
    started = time.perf_counter()
    try:
        for frame in range(header["frameCount"]):
            if state.stopped():
                raise InterruptedError("Stopped while evaluating")
            if not np.array_equal(source[frame]["ids"], decoded[frame]["ids"]):
                raise ValueError(f"Decoded IDs differ at frame {frame}")
            original, reconstructed = source[frame]["parameters"], decoded[frame]["parameters"]
            begin = frame * header["trackCount"]
            for offset in range(0, header["trackCount"], 65536):
                block = attribute_errors(original[offset:offset + 65536], reconstructed[offset:offset + 65536])
                errors[:, begin + offset:begin + offset + len(block)] = block.T
            with torch.no_grad():
                a, b = render(original, header), render(reconstructed, header)
                mse = (a - b).square().mean().item()
                row = dict(frame=frame, psnr=-10 * math.log10(max(mse, 1e-12)),
                           ssim=ssim(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]).item())
                rows.append(row)
                if frame in {0, 17, 20, 23, 50, header["frameCount"] - 1}:
                    save_image(output / f"original_{frame:03}.png", a)
                    save_image(output / f"decoded_{frame:03}.png", b)
                if frame == 20:
                    for label, displacement in [("left", -.025), ("right", .025)]:
                        save_image(output / f"original_020_{label}.png", render(original, header, displacement))
                        save_image(output / f"decoded_020_{label}.png", render(reconstructed, header, displacement))
            state.update(currentFrame=frame + 1, evaluationSeconds=time.perf_counter() - started,
                         psnr=float(np.mean([r["psnr"] for r in rows])), ssim=float(np.mean([r["ssim"] for r in rows])))
        state.update(stage="statistics")
        metrics = {}
        for column, name in enumerate(names):
            if state.stopped():
                raise InterruptedError("Stopped while computing statistics")
            values = errors[column]
            # Exact percentiles over every original observation, not a sampled estimate.
            average = float(values.mean(dtype=np.float64))
            maximum = float(values.max())
            p95, p99 = np.quantile(values, [.95, .99], overwrite_input=True)
            metrics[name] = dict(mae=average, p95=float(p95), p99=float(p99), maximum=maximum)
        report = dict(attributes=metrics, frames=rows,
                      psnr=float(np.mean([r["psnr"] for r in rows])), ssim=float(np.mean([r["ssim"] for r in rows])),
                      resolution=header["imageSizeWH"], reference="raw v5 Gaussians rendered with the same gsplat settings",
                      color="SH0 decoded directly to sRGB; no linearRGB transfer", evaluationSeconds=time.perf_counter()-started)
        atomic_json(output / "report.json", report)
        state.update(status="complete", stage="complete", evaluationComplete=True, report=report,
                     evaluationSeconds=report["evaluationSeconds"],
                     trainingCacheBytes=sum(p.stat().st_size for p in output.glob("*.pt") if p.name != "trajectory-codec.pt"))
    finally:
        errors._mmap.close()
        scratch.unlink(missing_ok=True)
        for array in (source, life, decoded, decoded_life):
            array._mmap.close()
