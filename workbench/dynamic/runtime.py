"""Windows CUDA setup, native MoDGS field, and shared renderer."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".cache" / "dynamic_torch_extensions"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "dynamic_torch"))
os.environ.setdefault("MAX_JOBS", "8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
sys.path.insert(0, str(ROOT / "ml-sharp" / "src"))
sys.path.insert(0, str(HERE.parent / "third_party" / "MoDGS"))


def setup_cuda():
    if os.name == "nt" and not os.environ.get("VSCMD_VER"):
        script = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
        output = subprocess.check_output(
            f'cmd.exe /d /s /c ""{script}" >nul && set"',
            text=True, errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
        for line in output.splitlines():
            if '=' in line and not line.startswith('='):
                name, value = line.split('=', 1)
                os.environ[name] = value
    cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    os.environ.setdefault("CUDA_HOME", cuda)
    os.environ["PATH"] = str(Path(cuda) / "bin") + os.pathsep + os.environ["PATH"]
    if os.name == "nt":
        from torch.utils import cpp_extension
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
        if not getattr(cpp_extension, "_dynamic_windows_flags", False):
            original = cpp_extension._jit_compile
            def compile_windows(*args, **kwargs):
                args = list(args)
                flags = args[2] if len(args) > 2 else kwargs.get("extra_cflags", [])
                flags = ["/O2" if flag == "-O3" else flag for flag in (flags or [])
                         if flag != "-Wno-attributes"]
                if len(args) > 2:
                    args[2] = flags
                else:
                    kwargs["extra_cflags"] = flags
                return original(*args, **kwargs)
            cpp_extension._jit_compile = compile_windows
            cpp_extension._dynamic_windows_flags = True


import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from model.RealNVP import NVPSimplified
from model.mfn import GaborNet
from upstream import Original_GaussianModel, RGB2SH, ssim


def read_config():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(os.environ.get("SHARP_VIDEO_CHECKPOINT", config["sharp_checkpoint"]))
    config["sharp_checkpoint"] = str(checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path)
    return config


class Deformation(nn.Module):
    """The official observation -> canonical network and its analytic inverse."""
    def __init__(self, chunk=16384):
        super().__init__()
        self.chunk = chunk
        self.feature_mlp = GaborNet(in_size=1, hidden_size=256, n_layers=2,
                                    alpha=4.5, out_size=128).cuda()
        self.deform_mlp = NVPSimplified(n_layers=4, feature_dims=128,
            hidden_size=[256, 256, 256], proj_dims=256, code_proj_hidden_size=[],
            proj_type="fixed_positional_encoding", pe_freq=5,
            normalization=False, affine=False).cuda()
        # Upstream sigma is a plain CUDA tensor; serialize it as a buffer as well.
        for layer in self.deform_mlp.layers1:
            sigma = layer.projection.sigma
            del layer.projection.sigma
            layer.projection.register_buffer("sigma", sigma)

    def _run(self, x, t, inverse):
        feature = self.feature_mlp(t.reshape(1, 1))
        fn = self.deform_mlp.inverse if inverse else self.deform_mlp
        return fn(t.reshape(1, 1), feature, x.unsqueeze(0)).squeeze(0)

    def forward(self, x, time, inverse=False):
        t = torch.as_tensor(time, dtype=x.dtype, device=x.device)
        outputs = []
        for block in x.split(self.chunk):
            if self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._run, block, t, inverse, use_reentrant=False))
            else:
                outputs.append(self._run(block, t, inverse))
        return torch.cat(outputs, dim=0)


def linear_to_srgb(rgb):
    rgb = rgb.clamp_min(0)
    return torch.where(rgb <= 0.0031308, 12.92 * rgb,
                       1.055 * rgb.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def srgb_to_linear(rgb):
    return torch.where(rgb <= 0.04045, rgb / 12.92,
                       ((rgb + 0.055) / 1.055).pow(2.4))


def backproject(depth, K, uv=None):
    if uv is None:
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        uv = np.stack((xx, yy), -1).reshape(-1, 2)
        z = depth.reshape(-1)
    else:
        z = depth
    return np.stack(((uv[:, 0] - K[0, 2]) * z / K[0, 0],
                     (uv[:, 1] - K[1, 2]) * z / K[1, 1], z), -1).astype(np.float32)


def render(gaussians, model, metadata, time, width=None, height=None, camera=None,
           means_override=None, retain_grad=False):
    from gsplat import rasterization
    width, height = width or metadata["width"], height or metadata["height"]
    device = gaussians.get_xyz.device
    scale = metadata["normalization"]["scale"]
    center = torch.tensor(metadata["normalization"]["center"], device=device)
    if means_override is None:
        means_n = model(gaussians.get_xyz, time, inverse=True)
    else:
        means_n = means_override
    means = means_n / scale + center
    K = torch.tensor(metadata["K"], device=device)
    K[0] *= width / metadata["width"]
    K[1] *= height / metadata["height"]
    view = torch.eye(4, device=device)
    if camera:
        # Camera is an OpenCV camera-to-world transform in metres.
        view = torch.linalg.inv(torch.tensor(camera, dtype=torch.float32, device=device).reshape(4, 4))
    colors = (0.28209479177387814 * gaussians._features_dc[:, 0] + 0.5).clamp_min(0)
    rgbd, alpha, info = rasterization(means, gaussians.get_rotation,
        gaussians.get_scaling / scale, gaussians.get_opacity[:, 0], colors,
        view[None], K[None], width, height, render_mode="RGB+ED", packed=False,
        rasterize_mode="classic")
    if retain_grad:
        info["means2d"].retain_grad()
    rgb = linear_to_srgb(rgbd[0, :, :, :3]).clamp(0, 1)
    return rgb, rgbd[0, :, :, 3], alpha[0, :, :, 0], info


def make_gaussians(xyz, color, scales, rotation, opacity):
    g = Original_GaussianModel(0)  # Upstream co-training never raises active SH degree.
    g.spatial_lr_scale = 1.0
    g._xyz = nn.Parameter(xyz.contiguous())
    g._features_dc = nn.Parameter(RGB2SH(color)[:, None].contiguous())
    g._features_rest = nn.Parameter(torch.empty((len(xyz), 0, 3), device=xyz.device))
    g._scaling = nn.Parameter(scales.clamp_min(1e-12).log())
    g._rotation = nn.Parameter(rotation.contiguous())
    g._opacity = nn.Parameter(torch.logit(opacity.reshape(-1, 1).clamp(1e-7, 1-1e-7)))
    g.max_radii2D = torch.zeros(len(xyz), device=xyz.device)
    return g


def save_ply(g, path):
    from plyfile import PlyData, PlyElement
    names = g.construct_list_of_attributes()
    arrays = [g._xyz, torch.zeros_like(g._xyz), g._features_dc[:, 0],
              g._opacity, g._scaling, g._rotation]
    matrix = torch.cat(arrays, dim=1).detach().cpu().numpy().astype('<f4')
    rows = np.empty(len(matrix), dtype=[(name, '<f4') for name in names])
    for index, name in enumerate(names):
        rows[name] = matrix[:, index]
    PlyData([PlyElement.describe(rows, 'vertex')], text=False, byte_order='<',
            comments=['MoDGS SHARP canonical space; colors linearRGB; use deformation.pt']).write(str(path))


def load_package(folder):
    package = torch.load(folder / "deformation.pt", map_location="cuda", weights_only=False)
    model = Deformation(package["config"]["point_chunk"])
    model.load_state_dict(package["state_dict"])
    model.eval()
    g = Original_GaussianModel(0)
    g.load_ply(str(folder / "canonical.ply"))
    return g, model, package["metadata"]
