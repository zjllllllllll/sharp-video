"""Five-sample temporal median, independently within each Gaussian lifetime.

Scalar fields use the ordinary median with scipy.ndimage's half-sample
symmetric ('reflect') endpoint convention. Rotation uses the angular medoid:
argmin_i sum_j 2*acos(abs(dot(q_i,q_j))), restricted to the five input rotations.
This is a sample median, not a componentwise or continuous SO(3) median.
Lifetimes shorter than five frames retain every original attribute.
"""
import numpy as np

WINDOW = 5
MIN_TRACK_FRAMES = 5
FILTER = {
    'method': 'median', 'windowFrames': WINDOW, 'boundary': 'reflect',
    'minTrackFrames': MIN_TRACK_FRAMES, 'shortTracks': 'preserve all original attributes',
    'scope': 'all_attributes',
    'scalarFields': 'x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2'.split(),
    'rotation': 'angular medoid on SO(3); quaternion sign invariant; input sample retained',
    'trajectoryBoundary': 'separate per lifecycle ID and layer',
    'scalarEncoding': 'SH0, log-scale and logit are monotone: odd-window median commutes with decoding',
}


def reflected_times(frame, starts, ends):
    lengths = ends - starts
    offsets = np.arange(-WINDOW // 2 + 1, WINDOW // 2 + 1)[:, None]
    relative = (frame + offsets - starts) % (2 * lengths)
    return starts + np.where(relative < lengths, relative, 2 * lengths - relative - 1)


def rotation_medoid(quaternions):
    """Input [5,N,4] in wxyz; return one original unit rotation per column."""
    q = quaternions.astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    costs = np.zeros(q.shape[:2], dtype=np.float64)
    for i in range(WINDOW):
        for j in range(i):
            angle = 2 * np.arccos(np.clip(np.abs(np.sum(q[i] * q[j], axis=-1)), 0, 1))
            costs[i] += angle
            costs[j] += angle
    # A deterministic middle-first tie order does not add a noise threshold.
    order = np.array([2, 1, 3, 0, 4])
    chosen = order[np.argmin(costs[order], axis=0)]
    return quaternions[chosen, np.arange(q.shape[1])]


def median_frame(frame, frame_ids, lifetimes, read_parameters, layers):
    """read_parameters(t) returns [layers*cells,14]. No cross-ID sampling."""
    cells = len(frame_ids)
    starts = np.asarray(lifetimes[frame_ids, 1], dtype=np.int64)
    ends = np.asarray(lifetimes[frame_ids, 2], dtype=np.int64)
    slots = np.flatnonzero(ends - starts >= MIN_TRACK_FRAMES)
    result = read_parameters(frame).copy()
    if not len(slots):
        return result
    times = reflected_times(frame, starts[slots], ends[slots])
    for layer in range(layers):
        window = np.empty((WINDOW, len(slots), 14), dtype='<f4')
        for k in range(WINDOW):
            for source_time in np.unique(times[k]):
                indices = np.flatnonzero(times[k] == source_time)
                window[k, indices] = read_parameters(int(source_time))[layer * cells + slots[indices]]
        rows = layer * cells + slots
        result[rows, 10:14] = rotation_medoid(window[..., 10:14])
        result[rows, :10] = np.median(window[..., :10], axis=0, overwrite_input=True)
    return result
