import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP / 'dynamic'))
from runtime import setup_cuda, read_config


class Stopped(Exception):
    pass


class Progress:
    def __init__(self, folder):
        self.folder = folder
        self.started = time.time()
        try:
            self.previous_elapsed = json.loads((folder / 'previous_status.json').read_text(encoding='utf-8')).get('elapsedSeconds', 0)
        except FileNotFoundError:
            self.previous_elapsed = 0
        self.value = {'status': 'running', 'pid': os.getpid(), 'startedAt': self.started}

    def check_stop(self):
        if (self.folder / 'stop.request').exists():
            raise Stopped()

    def __call__(self, stage, step, total, message, **extra):
        self.check_stop()
        self.value.update(stage=stage, step=step, total=total, message=message,
            updatedAt=time.time(), elapsedSeconds=self.previous_elapsed+time.time()-self.started, **extra)
        self.write()

    def write(self):
        temp = self.folder / 'status.tmp.json'
        temp.write_text(json.dumps(self.value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, self.folder / 'status.json')
        print(json.dumps(self.value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = run / 'dynamic'
    cache = output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    report = Progress(output)
    try:
        report('preparing', 0, 1, '准备第四步环境')
        setup_cuda()
        import numpy as np
        import torch
        from preprocess import (prepare_rgbd, sharp_seed, release, keyframes,
            compute_features, compute_flow, correspondences)
        from training import train_field, fuse_gaussians, train_joint, evaluate
        config = read_config()
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.set_float32_matmul_precision('high')
        torch.set_num_threads(8)
        (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        rgb, depths, metadata = prepare_rgbd(run, cache, report)
        report.value.update(width=metadata['width'], height=metadata['height'],
            frameCount=metadata['frameCount'], fps=metadata['fps'])
        seed = sharp_seed(rgb, depths, metadata, cache, config, report)
        metadata['sharpScale'] = seed['sharpScale']
        release()
        keys = keyframes(len(rgb), config['keyframe_stride'])
        flow = compute_flow(rgb, keys, cache, config, report)
        features = compute_features(rgb, keys, cache, report)
        pairs = correspondences(rgb, depths, metadata, keys, flow, features, cache, config, report)
        release()
        model = train_field(pairs, metadata, cache, config, report)
        gaussians = fuse_gaussians(seed, model, rgb, depths, metadata, cache, config, report)
        del seed
        release()
        gaussians, model = train_joint(gaussians, model, rgb, depths, metadata, output, cache, config, report)
        metrics = evaluate(gaussians, model, rgb, metadata, output, report)
        cache_bytes = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
        report.value.update(status='complete', stage='complete', metrics=metrics,
            cacheBytes=cache_bytes, elapsedSeconds=report.previous_elapsed+time.time()-report.started,
            completedAt=time.time(), message='动态高斯训练完成', step=1, total=1)
        report.write()
    except Stopped:
        report.value.update(status='stopped', message='训练已停止', updatedAt=time.time())
        report.write()
    except Exception as error:
        traceback.print_exc()
        report.value.update(status='failed', error=str(error), message='第四步运行失败', updatedAt=time.time())
        report.write()
        raise


if __name__ == '__main__':
    main()
