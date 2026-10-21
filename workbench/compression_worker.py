"""Convert complete SHARP PLY frames and run the official splats4D encoder."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from plyfile import PlyData


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def pack_ply(path):
    """Upstream gen_splat_frames.py's 32-byte SH0 layout, without reordering."""
    v = PlyData.read(path, mmap='r')['vertex'].data
    rec = np.empty((len(v), 32), dtype=np.uint8)
    xyz = np.column_stack([v[k] for k in ('x', 'y', 'z')]).astype('<f4')
    scales = np.exp(np.column_stack([v[f'scale_{k}'] for k in range(3)])).astype('<f4')
    rgb = np.clip(np.column_stack([v[f'f_dc_{k}'] for k in range(3)]) * 0.28209479177387814 + 0.5, 0, 1)
    alpha = 1 / (1 + np.exp(-np.clip(v['opacity'].astype(np.float64), -88, 88)))
    quat = np.column_stack([v[f'rot_{k}'] for k in range(4)]).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rec[:, :12] = xyz.view(np.uint8).reshape(-1, 12)
    rec[:, 12:24] = scales.view(np.uint8).reshape(-1, 12)
    rec[:, 24:27] = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    rec[:, 27] = np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)
    rec[:, 28:32] = np.clip(np.rint(quat * 128 + 128), 0, 255).astype(np.uint8)
    return rec


def run(run_dir, encoder=None):
    source = run_dir / 'gaussians'
    output = run_dir / 'compressed'
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'compression.json'
    started = time.perf_counter()
    state = {'status': 'running', 'stage': 'preparing', 'progress': 0,
             'currentFrame': 0, 'frameCount': 0, 'pid': os.getpid(),
             'generatedAt': datetime.now(timezone.utc).isoformat(), 'format': 'splat4d.v1'}

    def update(**values):
        state.update(values, elapsedSeconds=round(time.perf_counter() - started, 3))
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    update()
    try:
        meta_path = source / 'gaussian.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('status') != 'complete':
            raise ValueError('请先完成第四步逐帧高斯')
        records = meta['frames']
        frames_dir = output / 'frames'
        frames_dir.mkdir(exist_ok=True)
        encoder = encoder or str(Path(sys.executable).with_name('splat4d.exe' if os.name == 'nt' else 'splat4d'))
        if not Path(encoder).is_file():
            encoder = shutil.which('splat4d') or encoder
        if not Path(encoder).is_file():
            raise FileNotFoundError('未找到 splats4D 编码器，请安装 compression-requirements.txt')
        fps = float(meta['fps'])
        count = int(records[0]['gaussianCount'])
        manifest = {'fps': fps, 'count': len(records), 'num_splats': count, 'frames': []}
        update(stage='converting', frameCount=len(records), fps=fps, gaussianCount=count,
               sourceManifestSha256=hashlib.sha256(meta_path.read_bytes()).hexdigest())
        source_bytes = 0
        for i, record in enumerate(records):
            ply = source / record['full']
            packed = pack_ply(ply)
            if len(packed) != count:
                raise ValueError('splats4D 要求每帧高斯数量一致')
            name = f'frame_{i:04d}.splat'
            packed.tofile(frames_dir / name)
            source_bytes += ply.stat().st_size
            manifest['frames'].append({'file': name, 'timestamp': i / fps})
            update(currentFrame=i + 1, progress=100 * (i + 1) / len(records))
        atomic_json(frames_dir / 'frames.json', manifest)
        camera = {'position': [0, 0, 0], 'target': [0, 0, 1],
                  'width': meta['width'], 'height': meta['height'], 'focalLengthPx': meta['focalLengthPx'],
                  'fov': math.degrees(2 * math.atan(meta['height'] / (2 * meta['focalLengthPx'])))}
        update(stage='encoding', progress=None, sourceBytes=source_bytes, camera=camera)
        temporary = output / 'sequence.pending.splat4d'
        args = [encoder, 'encode', '-i', str(frames_dir), '-o', str(temporary), '--fps', str(fps),
                '--report', str(output / 'encoder-report.json')]
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as child:
            for line in child.stdout:
                print(line.rstrip(), flush=True)
                if line.startswith('times:'):
                    update(stage='verifying', progress=None)
            if child.wait():
                raise RuntimeError(f'splats4D 编码器退出：{child.returncode}')
        report = json.loads((output / 'encoder-report.json').read_text(encoding='utf-8'))
        if not report.get('verify', {}).get('ok'):
            raise RuntimeError('编码器的完整误差检查未通过')
        destination = output / 'sequence.splat4d'
        temporary.replace(destination)
        size = destination.stat().st_size
        update(status='complete', stage='complete', progress=100, file='sequence.splat4d',
               outputBytes=size, compressionRatio=source_bytes / size,
               encoder=report, completedAt=datetime.now(timezone.utc).isoformat())
        return state
    except Exception as error:
        update(status='failed', stage='failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--encoder')
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.encoder)
