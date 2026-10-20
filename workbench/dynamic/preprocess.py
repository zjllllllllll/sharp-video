"""RGB-D preparation and MoDGS/Omnimotion correspondence preprocessing."""
import gc
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from runtime import ROOT, HERE, backproject, make_gaussians, render


def release():
    gc.collect()
    torch.cuda.empty_cache()


def prepare_rgbd(run, cache, report):
    depth_meta = json.loads((run / 'depth.json').read_text(encoding='utf-8'))
    rgb_path, depth_path = cache / 'rgb.npy', cache / 'depth.npy'
    if not depth_path.exists():
        with np.load(run / depth_meta['depthData']) as data:
            np.save(depth_path, data['depths'].astype(np.float32))
    depths = np.load(depth_path, mmap_mode='r')
    count, height, width = depths.shape
    if not rgb_path.exists():
        cap = cv2.VideoCapture(str(run / depth_meta['sourceVideo']))
        temporary = cache / 'rgb.partial.npy'
        rgb = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                      shape=(count, height, width, 3))
        try:
            for index in range(count):
                ok, bgr = cap.read()
                if not ok or bgr.shape[:2] != (height, width):
                    raise RuntimeError(f'RGB 与深度帧不对应：第 {index + 1} 帧')
                rgb[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                report('preparing', index + 1, count, '对齐 RGB 与浮点深度')
            if cap.read()[0]:
                raise RuntimeError('RGB 帧数多于深度帧数')
        finally:
            cap.release()
        rgb.flush()
        del rgb
        temporary.replace(rgb_path)
    rgb = np.load(rgb_path, mmap_mode='r')
    from sharp.utils.io import convert_focallength
    focal = float(convert_focallength(width, height, 30))
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float32)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for index in range(count):
        points = backproject(depths[index], K)
        valid = np.isfinite(points).all(1) & (points[:, 2] > 0)
        lo = np.minimum(lo, points[valid].min(0))
        hi = np.maximum(hi, points[valid].max(0))
    metadata = dict(width=width, height=height, frameCount=count, fps=depth_meta['fps'],
        K=K.tolist(), durationSeconds=(count - 1) / depth_meta['fps'],
        timeNormalization='frameIndex / (frameCount - 1)',
        coordinateSystem='OpenCV: x right, y down, z forward; metres',
        colorSpace='linearRGB', focalSource='SHARP default 30mm equivalent',
        normalization={'center': ((lo + hi) / 2).tolist(), 'scale': float(2 / (hi - lo).max())})
    return rgb, depths, metadata


@torch.no_grad()
def sharp_seed(rgb, depths, metadata, cache, config, report):
    path = cache / 'sharp_seed.pt'
    if path.exists():
        return torch.load(path, weights_only=True, map_location='cpu')
    report('sharp', 0, 1, 'SHARP 首帧初始化')
    from sharp.models import create_predictor, PredictorParams
    from sharp.cli.predict import predict_image
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(torch.load(config['sharp_checkpoint'], weights_only=True, map_location='cpu'))
    predictor.eval().cuda()
    gaussian = predict_image(predictor, np.array(rgb[0]), metadata['K'][0][0], torch.device('cuda'))
    del predictor
    release()
    xyz = gaussian.mean_vectors[0]
    scales = gaussian.singular_values[0]
    rotation = gaussian.quaternions[0]
    color = gaussian.colors[0]
    opacity = gaussian.opacities[0]
    g = make_gaussians(xyz, color, scales, rotation, opacity)
    raw_meta = {**metadata, 'normalization': {'center': [0, 0, 0], 'scale': 1.0}}
    image, rendered_depth, alpha, _ = render(g, None, raw_meta, 0, means_override=xyz)
    target = torch.tensor(np.array(depths[0]), device='cuda')
    valid = (target > 0) & torch.isfinite(target) & (rendered_depth > 0) & (alpha > 0)
    factor = ((rendered_depth[valid] * target[valid]).sum() /
              rendered_depth[valid].square().sum()).item()
    center = torch.tensor(metadata['normalization']['center'], device='cuda')
    scale = metadata['normalization']['scale']
    seed = dict(xyz=((xyz * factor - center) * scale).cpu(),
                scales=(scales * factor * scale).cpu(), rotation=rotation.cpu(),
                color=color.cpu(), opacity=opacity.cpu(), sharpScale=factor)
    if not all(torch.isfinite(seed[key]).all() for key in ('xyz', 'scales', 'rotation', 'color', 'opacity')):
        raise RuntimeError('SHARP 初始化出现非有限数值')
    Image.fromarray((image.cpu().numpy() * 255).round().astype(np.uint8)).save(cache.parent / 'initial.png')
    torch.save(seed, path)
    report('sharp', 1, 1, '首帧高斯已生成', sharpGaussianCount=len(xyz), sharpScale=factor)
    return seed


def keyframes(count, stride):
    return sorted(set(range(0, count, stride)) | {count - 1})


@torch.no_grad()
def compute_features(rgb, keys, cache, report):
    feature_dir = cache / 'features'
    feature_dir.mkdir(exist_ok=True)
    if all((feature_dir / f'{index:05d}.npy').exists() for index in keys):
        return feature_dir
    source = HERE.parent / 'third_party' / 'omnimotion' / 'preprocessing' / 'dino'
    sys.path.insert(0, str(source))
    import vision_transformer
    net = vision_transformer.vit_small(patch_size=16, num_classes=0)
    weights = torch.hub.load_state_dict_from_url(
        'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        map_location='cpu', progress=False)
    net.load_state_dict(weights)
    net.eval().cuda()
    del weights
    # Omnimotion extract_dino_features.py: short side 480, long side multiple of 64.
    h, w = rgb.shape[1:3]
    th, tw = (int((480 * h / w) // 64 * 64), 480) if h > w else (480, int((480 * w / h) // 64 * 64))
    mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
    std = torch.tensor([.228, .224, .225], device='cuda')[:, None, None]
    for ordinal, index in enumerate(keys):
        path = feature_dir / f'{index:05d}.npy'
        if not path.exists():
            image = cv2.resize(np.array(rgb[index]), (tw, th)).astype(np.float32) / 255
            image = torch.from_numpy(image).cuda().permute(2, 0, 1)
            feature = net.get_intermediate_layers(((image - mean) / std)[None], n=1)[0][0, 1:]
            np.save(path, feature.reshape(th // 16, tw // 16, -1).cpu().numpy())
        report('features', ordinal + 1, len(keys), '提取官方过滤流程使用的 DINO 特征')
    del net
    release()
    return feature_dir


@torch.no_grad()
def compute_flow(rgb, keys, cache, config, report):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    folder = cache / 'flow'
    folder.mkdir(exist_ok=True)
    pairs = [(i, j) for i in keys for j in keys if i != j]
    if all((folder / f'{i:05d}_{j:05d}.npy').exists() for i, j in pairs):
        return folder
    weights = Raft_Large_Weights.C_T_V1
    model = raft_large(weights=weights, progress=False).eval().cuda()
    h, w = rgb.shape[1:3]
    for ordinal, (i, j) in enumerate(pairs):
        path = folder / f'{i:05d}_{j:05d}.npy'
        if not path.exists():
            a = torch.from_numpy(np.array(rgb[i])).permute(2, 0, 1)[None].cuda()
            b = torch.from_numpy(np.array(rgb[j])).permute(2, 0, 1)[None].cuda()
            a, b = weights.transforms()(a, b)
            a = F.pad(a, (0, -w % 8, 0, -h % 8), mode='replicate')
            b = F.pad(b, (0, -w % 8, 0, -h % 8), mode='replicate')
            flow = model(a, b, num_flow_updates=config['flow_updates'])[-1][0, :, :h, :w]
            np.save(path, flow.permute(1, 2, 0).cpu().numpy())
            del a, b, flow
        report('flow', ordinal + 1, len(pairs), '计算关键帧双向 RAFT 光流')
    del model
    release()
    return folder


@torch.no_grad()
def correspondences(rgb, depths, metadata, keys, flow_dir, feature_dir, cache, config, report):
    """Port of Omnimotion filter_raft.py: cycle, DINO and valid depth filtering.

    Occluded-only correspondences are excluded from 3D supervision because target
    RGB-D describes the occluder there. They are not valid paired surface depths.
    """
    folder = cache / 'pairs'
    folder.mkdir(exist_ok=True)
    height, width = rgb.shape[1:3]
    yy, xx = torch.meshgrid(torch.arange(height, device='cuda'), torch.arange(width, device='cuda'), indexing='ij')
    grid = torch.stack((xx, yy), -1).float()
    divisor = torch.tensor([width - 1, height - 1], device='cuda')
    grid_norm = grid / divisor * 2 - 1
    K = np.array(metadata['K'], np.float32)
    center = np.array(metadata['normalization']['center'], np.float32)
    scale = metadata['normalization']['scale']
    result = []
    pairs = [(ii, jj, i, j) for ii, i in enumerate(keys) for jj, j in enumerate(keys) if i != j]
    # Feature maps stay on CPU; sample a few image rows at a time (720p x 384
    # descriptors would otherwise allocate multiple gigabytes per pair).
    for ordinal, (ii, jj, i, j) in enumerate(pairs):
        path = folder / f'{i:05d}_{j:05d}.npy'
        if not path.exists():
            fwd = torch.from_numpy(np.load(flow_dir / f'{i:05d}_{j:05d}.npy')).cuda()
            bwd = torch.from_numpy(np.load(flow_dir / f'{j:05d}_{i:05d}.npy')).cuda().permute(2, 0, 1)[None]
            target_uv = grid + fwd
            target_norm = target_uv / divisor * 2 - 1
            inverse = F.grid_sample(bwd, target_norm[None], align_corners=True)[0].permute(1, 2, 0)
            valid = ((fwd + inverse).norm(dim=-1) < config['cycle_threshold_px'])
            valid &= (target_norm.abs() <= 1).all(-1)
            if abs(ii - jj) >= 3:
                fa = torch.from_numpy(np.load(feature_dir / f'{i:05d}.npy')).cuda().permute(2, 0, 1)[None]
                fb = torch.from_numpy(np.load(feature_dir / f'{j:05d}.npy')).cuda().permute(2, 0, 1)[None]
                for start in range(0, height, 16):
                    a = F.grid_sample(fa, grid_norm[None, start:start+16], align_corners=True)
                    b = F.grid_sample(fb, target_norm[None, start:start+16], align_corners=True)
                    valid[start:start+16] &= F.cosine_similarity(a, b, dim=1)[0] > config['dino_similarity']
                del fa, fb, a, b
            z_source = torch.tensor(np.array(depths[i]), device='cuda')
            z_target = F.grid_sample(torch.tensor(np.array(depths[j]), device='cuda')[None, None],
                target_norm[None], align_corners=True)[0, 0]
            valid &= (z_source > 0) & (z_target > 0) & torch.isfinite(z_source) & torch.isfinite(z_target)
            source_xyz = backproject(z_source[valid].cpu().numpy(), K, grid[valid].cpu().numpy())
            target_xyz = backproject(z_target[valid].cpu().numpy(), K, target_uv[valid].cpu().numpy())
            data = np.concatenate(((source_xyz-center)*scale, (target_xyz-center)*scale), 1)
            np.save(path, data)
            del fwd, bwd, inverse, z_source, z_target, target_uv, target_norm
        count = np.load(path, mmap_mode='r').shape[0]
        if count:
            result.append({'path': str(path), 'source': i, 'target': j, 'count': count})
        report('correspondences', ordinal + 1, len(pairs), '将有效光流提升为三维对应', pairCount=len(result))
    if not result:
        raise RuntimeError('没有可用于初始化变形的有效三维对应')
    return result
