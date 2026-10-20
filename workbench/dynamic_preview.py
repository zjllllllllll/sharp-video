"""Persistent two-file model renderer. stdin/stdout use one JSON object per line."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent / 'dynamic'))
from runtime import setup_cuda, load_package, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    setup_cuda()
    import torch
    import numpy as np
    from PIL import Image
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(8)
    g, model, metadata = load_package(args.model_dir)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            seconds = float(request.get('timeSeconds', 0))
            duration = metadata['durationSeconds']
            time_normalized = max(0, min(1, seconds / duration)) if duration else 0
            width = int(request.get('width') or metadata['width'])
            height = int(request.get('height') or metadata['height'])
            if width < 1 or height < 1:
                raise ValueError('画面尺寸必须为正整数')
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                image, _, _, _ = render(g, model, metadata, time_normalized, width, height, request.get('camera'))
            torch.cuda.synchronize()
            render_ms = (time.perf_counter() - started) * 1000
            pixels = (image.cpu().numpy()*255).round().astype(np.uint8)
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format='PNG', compress_level=1)
            response = dict(id=request['id'], png=base64.b64encode(stream.getvalue()).decode('ascii'), renderMs=render_ms)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = dict(id=request.get('id'), error=str(error))
        print(json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
