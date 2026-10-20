"""Batch adapters around siren-pytorch 0.1.7, without changing its layers.

Upstream: https://github.com/lucidrains/siren-pytorch (MIT).
Only modulation broadcasting and concatenation gain a batch dimension.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from siren_pytorch.siren_pytorch import SirenNet, Modulator

FORMAT = "sharp.trajectory_codec.v1"
STANDARD_COLUMNS = [0, 1, 2, 3, 4, 5, 7, 8, 9]
GROUPS = ("position", "color", "scale", "opacity", "rotation")


def default_config():
    return json.loads(Path(__file__).with_name("config.json").read_text())


class AttributeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = SirenNet(dim_in=1, dim_hidden=config["hidden_dim"], dim_out=14,
                            num_layers=config["hidden_layers"], w0=config["w0"],
                            w0_initial=config["w0_initial"])
        self.modulator = Modulator(dim_in=config["latent_dim"],
                                   dim_hidden=config["hidden_dim"],
                                   num_layers=config["hidden_layers"])

    def forward(self, codes, times):
        x, modulation = times, codes
        for layer, mod_layer in zip(self.net.layers, self.modulator.layers):
            mod = mod_layer(modulation)
            x = layer(x) * mod
            modulation = torch.cat((mod, codes), dim=-1)
        return self.net.last_layer(x)


def unit_quaternion(q):
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(norm > 0, q / norm.clamp_min(torch.finfo(q.dtype).tiny), identity)


def group_losses(prediction, target, mean, std):
    normalized = (target[:, STANDARD_COLUMNS] - mean) / std
    scalar = (prediction[:, STANDARD_COLUMNS] - normalized).square()
    q, reference = unit_quaternion(prediction[:, 10:14]), unit_quaternion(target[:, 10:14])
    rotation = torch.minimum((q - reference).square().mean(-1),
                             (q + reference).square().mean(-1)).mean()
    return torch.stack((scalar[:, :3].mean(), scalar[:, 3:6].mean(),
                        scalar[:, 6:9].mean(),
                        (prediction[:, 6].sigmoid() - target[:, 6].sigmoid()).square().mean(),
                        rotation))


def physical_attributes(prediction, mean, std):
    result = prediction.clone()
    result[:, STANDARD_COLUMNS] = prediction[:, STANDARD_COLUMNS] * std + mean
    result[:, 10:14] = unit_quaternion(prediction[:, 10:14])
    return result


def local_times(cell_ids, frame, lifecycles, times):
    """NumPy CPU utility: lifecycle time uses observed timestamps, not frame numbers."""
    import numpy as np
    life = lifecycles[cell_ids]
    start, end = times[life[:, 0]], times[life[:, 1] - 1]
    duration = end - start
    tau = np.zeros(len(cell_ids), dtype=np.float32)
    np.divide(2 * (times[frame] - start), duration, out=tau, where=duration > 0)
    tau[duration > 0] -= 1
    return tau[:, None]


def load_package(path, device="cpu"):
    package = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if package["format"] != FORMAT:
        raise ValueError("Unsupported trajectory codec format")
    model = AttributeDecoder(package["config"]).to(device)
    model.load_state_dict(package["network"])
    model.eval()
    return package, model
