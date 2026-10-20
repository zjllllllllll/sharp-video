"""Export a standalone all-attribute median-filtered lifecycle animation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from gaussian_animation import write_header
from gaussian_lifecycle import read_lifecycle_animation
from gaussian_median import FILTER, median_frame


def write_state(path, state):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    output = run_dir / 'tracks-median'
    output.mkdir(exist_ok=True)
    state_path = output / 'median.json'
    temporary = output / 'animation.4dgs.tmp'
    final_path = output / 'animation.4dgs'
    started = time.perf_counter()
    state = {'status': 'running', 'stage': 'median', 'filter': FILTER,
             'generatedAt': datetime.now(timezone.utc).isoformat(), 'currentFrame': 0}
    write_state(state_path, state)
    frames = lifetimes = None
    try:
        header, frames, lifetimes = read_lifecycle_animation(run_dir / 'tracks-lifecycle' / 'animation.4dgs')
        header = {**header, 'filter': FILTER, 'generatedAt': state['generatedAt']}
        state.update(header, status='running', stage='median', currentFrame=0)
        cache = {}
        def read_parameters(index):
            if index not in cache:
                cache[index] = frames[index]['parameters'].copy()
            return cache[index]
        changed = np.zeros(10, dtype=np.int64)
        with temporary.open('wb') as stream:
            offset = write_header(stream, header)
            for frame in range(header['frameCount']):
                ids = frames[frame]['ids']
                result = median_frame(frame, ids, lifetimes, read_parameters, header['layerCount'])
                changed += np.count_nonzero(result[:, :10] != read_parameters(frame)[:, :10], axis=0)
                ids.tofile(stream)
                result.tofile(stream)
                for index in list(cache):
                    if index < frame - 1:
                        del cache[index]
                state.update(currentFrame=frame+1, elapsedSeconds=round(time.perf_counter()-started, 3))
                write_state(state_path, state)
                print(json.dumps({'frame': frame+1, 'total': header['frameCount'], 'elapsedSeconds': state['elapsedSeconds']}), flush=True)
            lifetimes.tofile(stream)
        temporary.replace(final_path)
        state.update(status='complete', completedAt=datetime.now(timezone.utc).isoformat(),
                     elapsedSeconds=round(time.perf_counter()-started, 3),
                     changedScalarCounts=dict(zip(header['columns'][:10], changed.tolist())),
                     file={'file': final_path.name, 'bytes': final_path.stat().st_size, 'dataOffset': offset})
        write_state(state_path, state)
    except Exception as error:
        state.update(status='failed', error=str(error), traceback=traceback.format_exc())
        write_state(state_path, state)
        raise
    finally:
        if frames is not None:
            frames._mmap.close()
        if lifetimes is not None:
            lifetimes._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    run(parser.parse_args().run_dir)
