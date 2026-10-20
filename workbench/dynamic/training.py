import gc
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from runtime import (Deformation, backproject, make_gaussians, render, save_ply,
                     load_package, srgb_to_linear, ssim)
from preprocess import release


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def train_field(pairs, metadata, cache, config, report):
    model = Deformation(config['point_chunk'])
    output = cache / 'field.pt'
    if output.exists():
        model.load_state_dict(torch.load(output, weights_only=True, map_location='cuda'))
        return model
    optimizer = torch.optim.Adam([
        {'params': model.feature_mlp.parameters(), 'lr': config['init_feature_lr']},
        {'params': model.deform_mlp.parameters(), 'lr': config['init_deform_lr']}])
    rng = np.random.default_rng(config['seed'])
    start = 0
    checkpoint_path = cache / 'initialization_checkpoint.pt'
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=False, map_location='cuda')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
        rng.bit_generator.state = saved['rng']
    dataset = [np.load(pair['path'], mmap_mode='r') for pair in pairs]
    total = config['init_steps']
    started = time.perf_counter()
    model.train()
    with (cache / 'initialization_loss.jsonl').open('a', encoding='utf-8') as log:
        for step in range(start + 1, total + 1):
            report.check_stop()
            index = int(rng.integers(len(pairs)))
            pair = pairs[index]
            count = min(config['points_per_step'], pair['count'])
            selection = rng.choice(pair['count'], count, replace=False)
            data = torch.tensor(np.array(dataset[index][selection]), device='cuda')
            ti = torch.tensor([[pair['source'] / (metadata['frameCount'] - 1)]], device='cuda')
            tj = torch.tensor([[pair['target'] / (metadata['frameCount'] - 1)]], device='cuda')
            decay = config['init_lr_decay'] ** (step-1)
            for group, initial in zip(optimizer.param_groups, [config['init_feature_lr'], config['init_deform_lr']]):
                group['lr'] = initial * decay
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            # Full selected batch, accumulated exactly across blocks, one Adam step.
            for chunk in data.split(config['point_chunk']):
                canonical = model._run(chunk[:, :3], ti, False)
                predicted = model._run(canonical, tj, True)
                loss = (predicted - chunk[:, 3:]).square().sum() / (count * 3)
                loss.backward()
                loss_value += loss.detach().item()
            if not math.isfinite(loss_value):
                raise RuntimeError('变形初始化损失出现非有限数值')
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == total:
                entry = dict(step=step, loss=loss_value, learningRates=[group['lr'] for group in optimizer.param_groups])
                log.write(json.dumps(entry) + '\n')
                log.flush()
                report('initialization', step, total, '学习共享可逆变形',
                       loss=loss_value, secondsPerStep=(time.perf_counter()-started)/(step-start))
            if step % 1000 == 0:
                atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, rng=rng.bit_generator.state), checkpoint_path)
    atomic_torch_save(model.state_dict(), output)
    del optimizer, dataset, data, chunk, canonical, predicted, loss
    release()
    return model


def cell_keys(cells):
    return np.ascontiguousarray(cells).view(np.dtype((np.void, 24))).reshape(-1)


@torch.no_grad()
def fuse_gaussians(seed, model, rgb, depths, metadata, cache, config, report):
    saved_path = cache / 'fused.pt'
    if saved_path.exists():
        state = torch.load(saved_path, weights_only=True, map_location='cuda')
        return make_gaussians(*(state[name] for name in ('xyz', 'color', 'scales', 'rotation', 'opacity')))
    model.eval()
    xyz = model(seed['xyz'].cuda(), 0)
    voxel = config['voxel_size']
    seed_cells = np.unique(np.floor(xyz.cpu().numpy() / voxel).astype(np.int64), axis=0)
    occupied = cell_keys(seed_cells)
    cells = np.empty((0, 3), dtype=np.int64)
    sums = np.empty((0, 6), dtype=np.float64)
    counts = np.empty(0, dtype=np.float64)
    center = np.array(metadata['normalization']['center'], np.float32)
    scale = metadata['normalization']['scale']
    K = np.array(metadata['K'], np.float32)
    for index in range(1, len(rgb)):
        report.check_stop()
        points = backproject(depths[index], K)
        valid = np.isfinite(points).all(1) & (points[:, 2] > 0)
        canonical = model(torch.tensor((points[valid]-center)*scale, device='cuda'), index / (len(rgb)-1))
        canonical = canonical.cpu().numpy()
        colors = srgb_to_linear(torch.from_numpy(np.array(rgb[index]).reshape(-1, 3)[valid]).float()/255).numpy()
        q, inverse = np.unique(np.floor(canonical / voxel).astype(np.int64), axis=0, return_inverse=True)
        local_count = np.bincount(inverse).astype(np.float64)
        values = np.concatenate((canonical, colors), axis=1)
        local_sum = np.stack([np.bincount(inverse, weights=values[:, i]) for i in range(6)], axis=1)
        missing = ~np.isin(cell_keys(q), occupied)
        combined = np.concatenate((cells, q[missing]))
        all_sums = np.concatenate((sums, local_sum[missing]))
        all_counts = np.concatenate((counts, local_count[missing]))
        cells, merge = np.unique(combined, axis=0, return_inverse=True)
        sums = np.stack([np.bincount(merge, weights=all_sums[:, i], minlength=len(cells)) for i in range(6)], axis=1)
        counts = np.bincount(merge, weights=all_counts, minlength=len(cells))
        report('fusion', index, len(rgb)-1, '把后续表面融合到共同空间',
               sharpGaussianCount=len(xyz), addedGaussianCount=len(cells))
    averaged = (sums / counts[:, None]).astype(np.float32)
    added = torch.tensor(averaged[:, :3], device='cuda')
    all_xyz = torch.cat((xyz, added))
    if len(added):
        tree = cKDTree(all_xyz.cpu().numpy())
        distances = tree.query(averaged[:, :3], k=4, workers=-1)[0][:, 1:]
        # Same nearest-three scale initialization used by original 3DGS.
        init_scale = np.sqrt(np.maximum(np.square(distances).mean(1), 1e-7))
        added_scale = torch.tensor(init_scale, dtype=torch.float32, device='cuda')[:, None].expand(-1, 3)
    else:
        added_scale = torch.empty((0, 3), device='cuda')
    rotation = torch.zeros((len(added), 4), device='cuda')
    rotation[:, 0] = 1
    state = dict(xyz=all_xyz, color=torch.cat((seed['color'].cuda(), torch.tensor(averaged[:, 3:], device='cuda'))),
        scales=torch.cat((seed['scales'].cuda(), added_scale)),
        rotation=torch.cat((seed['rotation'].cuda(), rotation)),
        opacity=torch.cat((seed['opacity'].cuda().reshape(-1), torch.full((len(added),), .1, device='cuda'))))
    atomic_torch_save({k: v.cpu() for k, v in state.items()}, saved_path)
    result = make_gaussians(*(state[name] for name in ('xyz', 'color', 'scales', 'rotation', 'opacity')))
    release()
    return result


def depth_order_loss(predicted, target, alpha, config):
    valid = torch.isfinite(target) & (target > 0) & torch.isfinite(predicted) & (alpha > 0)
    truth = target[valid]
    prediction = predicted[valid]
    if truth.numel() < 2:
        return predicted.sum() * 0
    truth = (truth - truth.min()) / (truth.max() - truth.min()).clamp_min(1e-8)
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()).clamp_min(1e-8)
    a = torch.randint(len(truth), (config['depth_pairs'],), device=truth.device)
    b = torch.randint(len(truth), (config['depth_pairs'],), device=truth.device)
    difference = truth[a] - truth[b]
    selected = difference.abs() > config['depth_pair_difference']
    if not selected.any():
        return predicted.sum() * 0
    return (torch.tanh(config['depth_alpha'] * (prediction[a[selected]] - prediction[b[selected]]))
            - difference[selected].sign()).abs().mean()


def train_joint(g, model, rgb, depths, metadata, output, cache, config, report):
    g.training_setup(SimpleNamespace(**config))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['joint_field_lr'])
    rng = np.random.default_rng(config['seed'] + 1)
    start, indices = 0, []
    checkpoint_path = cache / 'joint_checkpoint.pt'
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        g.restore(state['gaussians'], SimpleNamespace(**config))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']
        start, indices = state['step'], state['indices']
    total, started = config['joint_steps'], time.perf_counter()
    model.train()
    with (cache / 'joint_loss.jsonl').open('a', encoding='utf-8') as log:
        for step in range(start+1, total+1):
            report.check_stop()
            if not indices:
                indices = rng.permutation(len(rgb)).tolist()
            index = indices.pop()
            t = index / (len(rgb)-1)
            optimizer.zero_grad(set_to_none=True)
            g.optimizer.zero_grad(set_to_none=True)
            g.update_learning_rate(step-1)
            lr = config['joint_field_lr'] * config['joint_lr_decay'] ** (step-1)
            for group in optimizer.param_groups:
                group['lr'] = lr
            # Checkpoint the whole deformation operation: render detached centres,
            # then recompute each block and apply its exact vector-Jacobian product.
            # Every Gaussian contributes; no point subsampling or image downscale.
            with torch.no_grad():
                means = model(g.get_xyz, t, inverse=True)
            means.requires_grad_(True)
            image, depth, alpha, info = render(g, model, metadata, t, means_override=means, retain_grad=True)
            target = torch.tensor(np.array(rgb[index]), device='cuda').float()/255
            depth_target = torch.tensor(np.array(depths[index]), device='cuda')
            l1 = (image-target).abs().mean()
            structural = 1-ssim(image.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
            ordinal = depth_order_loss(depth, depth_target, alpha, config)
            loss = l1 + config['lambda_ssim']*structural + config['lambda_depth']*ordinal
            if not torch.isfinite(loss):
                raise RuntimeError('联合训练损失出现非有限数值')
            loss.backward()
            upstream_gradient = means.grad.detach()
            time_tensor = torch.tensor(t, device='cuda')
            for offset in range(0, len(means), config['point_chunk']):
                block = g.get_xyz[offset:offset+config['point_chunk']]
                deformed = model._run(block, time_tensor, True)
                deformed.backward(upstream_gradient[offset:offset+len(block)])
            with torch.no_grad():
                if step < config['densify_until']:
                    radii = info['radii'][0]
                    if radii.ndim == 2:
                        radii = radii.max(-1).values
                    visible = radii > 0
                    g.max_radii2D[visible] = torch.maximum(g.max_radii2D[visible], radii[visible].float())
                    # gsplat means2d uses pixels; upstream 3DGS uses NDC derivatives.
                    grads = info['means2d'].grad[0] * torch.tensor([metadata['width']/2, metadata['height']/2], device='cuda')
                    g.xyz_gradient_accum[visible] += grads[visible].norm(dim=-1, keepdim=True)
                    g.denom[visible] += 1
                g.optimizer.step()
                optimizer.step()
                if config['densify_from'] < step < config['densify_until'] and step % config['densify_interval'] == 0:
                    g.densify_and_prune(config['densify_grad_threshold'], config['min_opacity'], 1.0, None)
            if step == 1 or step % 25 == 0 or step == total:
                entry = dict(step=step, frameIndex=index, loss=loss.item(), l1=l1.item(),
                             ssimLoss=structural.item(), depthLoss=ordinal.item(), gaussianCount=len(g.get_xyz))
                log.write(json.dumps(entry) + '\n')
                log.flush()
                report('joint', step, total, '联合优化基础高斯和共享变形',
                       **{k: v for k, v in entry.items() if k != 'step'},
                       secondsPerStep=(time.perf_counter()-started)/(step-start))
            if step % 1000 == 0:
                atomic_torch_save(dict(gaussians=g.capture(), model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=rng.bit_generator.state, indices=indices, step=step), checkpoint_path)
            del image, depth, alpha, info, means, upstream_gradient, loss, l1, structural, ordinal, deformed, block
    model.eval()
    save_ply(g, output / 'canonical.ply')
    package = dict(formatVersion=1, architecture='MoDGS-NVPSimplified-4-Gabor128',
        config=config, metadata=metadata,
        upstream={'MoDGS': 'f586bc41bef5539b64cb3a397634725eb0d4db97',
                  'SHARP': 'aed6527499ef91cba3b54c18d49a870f25947190',
                  'Omnimotion': '43868e5d214b795f00c586b790cb832a6f571a4d'},
        state_dict={k: v.cpu() for k, v in model.state_dict().items()})
    atomic_torch_save(package, output / 'deformation.pt')
    return g, model


@torch.no_grad()
def evaluate(g, model, rgb, metadata, output, report):
    import cv2
    model.eval()
    psnr_values, ssim_values, render_times = [], [], []
    checks = sorted(set([0, len(rgb)//2, len(rgb)-1]))
    reference = {}
    writer = cv2.VideoWriter(str(output / 'preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             metadata['fps'], (metadata['width'], metadata['height']))
    try:
        for index in range(len(rgb)):
            report.check_stop()
            torch.cuda.synchronize()
            started = time.perf_counter()
            image, _, _, _ = render(g, model, metadata, index/(len(rgb)-1))
            torch.cuda.synchronize()
            render_times.append(time.perf_counter()-started)
            truth = torch.tensor(np.array(rgb[index]), device='cuda').float()/255
            mse = (image-truth).square().mean().item()
            psnr_values.append(-10*math.log10(max(mse, 1e-12)))
            ssim_values.append(ssim(image.permute(2, 0, 1)[None], truth.permute(2, 0, 1)[None]).item())
            pixels = (image.cpu().numpy()*255).round().astype(np.uint8)
            writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            if index in checks:
                reference[index] = image.cpu()
                Image.fromarray(pixels).save(output / f'frame_{index:05d}.png')
            report('evaluating', index+1, len(rgb), '生成完整序列和原视角指标')
    finally:
        writer.release()
    # Reload only the delivered two-file package for the export consistency check.
    g2, model2, meta2 = load_package(output)
    errors = []
    for index in checks:
        image, _, _, _ = render(g2, model2, meta2, index/(len(rgb)-1))
        errors.append((image.cpu()-reference[index]).abs().max().item())
    for offset in [-0.025, 0.025]:
        camera = np.eye(4, dtype=np.float32)
        camera[0, 3] = offset
        for index in checks:
            image, _, _, _ = render(g2, model2, meta2, index/(len(rgb)-1), camera=camera.tolist())
            Image.fromarray((image.cpu().numpy()*255).round().astype(np.uint8)).save(
                output / f'view_{offset:+.3f}_{index:05d}.png')
    result = dict(psnr=float(np.mean(psnr_values)), ssim=float(np.mean(ssim_values)),
        perFramePSNR=psnr_values, perFrameSSIM=ssim_values,
        renderFPS=1/float(np.mean(render_times)), exportMaxError=max(errors),
        gaussianCount=len(g.get_xyz), parameterCount=sum(p.numel() for p in model.parameters()),
        plyBytes=(output/'canonical.ply').stat().st_size,
        modelBytes=(output/'deformation.pt').stat().st_size)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if max(errors) > 1e-5:
        raise RuntimeError(f'导出重载渲染不一致：{max(errors)}')
    return result
