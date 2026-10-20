import {setPackedSplat} from '@sparkjsdev/spark';

const SH0 = 0.28209479177387814;

// Update one persistent render object. Input values retain the original PLY
// encodings; no PLY parser, source-index lookup or per-frame mesh is involved.
export function applyAnimationFrame(splats, parameters) {
  const count = parameters.length / 14;
  const packed = splats.ensureSplats(count);
  for (let id = 0; id < count; id++) {
    const p = id * 14;
    setPackedSplat(packed, id,
      parameters[p], parameters[p + 1], parameters[p + 2],
      Math.exp(parameters[p + 7]), Math.exp(parameters[p + 8]), Math.exp(parameters[p + 9]),
      parameters[p + 11], parameters[p + 12], parameters[p + 13], parameters[p + 10],
      1 / (1 + Math.exp(-parameters[p + 6])),
      0.5 + SH0 * parameters[p + 3], 0.5 + SH0 * parameters[p + 4], 0.5 + SH0 * parameters[p + 5],
      splats.splatEncoding);
  }
  splats.numSplats = count;
  splats.needsUpdate = true;
}
