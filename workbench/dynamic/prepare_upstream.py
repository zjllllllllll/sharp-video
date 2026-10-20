"""Extract the upstream Gaussian/SSIM code without unrelated legacy CUDA imports.

Algorithm bodies are copied verbatim. Rerun after intentionally updating the pinned
vendor checkout; attribution and upstream licenses remain in third_party/MoDGS.
"""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "third_party" / "MoDGS"


def extract(relative, names):
    source = (UPSTREAM / relative).read_text(encoding="utf-8")
    tree = ast.parse(source)
    return "\n\n".join(ast.get_source_segment(source, node) for node in tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)


header = '''# Derived verbatim from Mobiuslqm/MoDGS f586bc41bef5539b64cb3a397634725eb0d4db97.
# Copyright (C) 2023, Inria, GRAPHDECO research group. All rights reserved.
# See ../third_party/MoDGS/LICENSE.md and LICENSE for the upstream terms.
import os
from math import exp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from plyfile import PlyData, PlyElement
from typing import NamedTuple
from scipy.spatial import cKDTree

class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def RGB2SH(rgb):
    return (rgb - 0.5) / 0.28209479177387814

def distCUDA2(points):
    # Same mean squared distance to the nearest three other points as SimpleKNN.
    xyz = points.detach().cpu().numpy()
    distances = cKDTree(xyz).query(xyz, k=4, workers=-1)[0][:, 1:]
    return torch.as_tensor(np.square(distances).mean(axis=1), device=points.device, dtype=points.dtype)

'''
general = extract("utils/general_utils.py", {
    "inverse_sigmoid", "get_expon_lr_func", "strip_lowerdiag", "strip_symmetric",
    "build_rotation", "build_scaling_rotation"})
gaussians = extract("scene/PointTrack_gaussian_model.py", {"Original_GaussianModel"})
loss = extract("utils/loss_utils.py", {"gaussian", "create_window", "ssim", "_ssim"})
(HERE / "upstream.py").write_text(header + general + "\n\n" + gaussians + "\n\n" + loss + "\n", encoding="utf-8")
print("Extracted upstream Gaussian optimization, densification and SSIM.")
