import assert from 'node:assert/strict';
import {test} from 'node:test';
import {PackedSplats} from '@sparkjsdev/spark';
import {Vector3, Quaternion, Color, Group, PerspectiveCamera} from 'three';
import {ANIMATION_COLUMNS, ANIMATION_FORMAT, LIFECYCLE_FORMAT, readAnimationHeader, readAnimationFrame} from '../static/animation-file.js';
import {applyAnimationFrame} from '../static/animation-splats.js';
import {TrackSequenceViewer} from '../static/track-viewer.js';

test('single-file reader uses embedded metadata and supports offsets beyond 4 GiB', async () => {
  const header = {format: ANIMATION_FORMAT, dtype: '<f4', columns: ANIMATION_COLUMNS,
    trackCount: 1179648, frameCount: 100, frameBytes: 1179648 * 56, timesSeconds: Array.from({length: 100}, (_, i) => i / 25)};
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(16);
  prefix.set(new TextEncoder().encode('SH4DGS01'));
  new DataView(prefix.buffer).setUint32(8, encoded.length, true);
  const dataOffset = Math.ceil((16 + encoded.length) / 8) * 8;
  const calls = [];
  const fetcher = async (url, options) => {
    assert.equal(url, 'only-file.4dgs');
    calls.push(options.headers.Range);
    let bytes;
    if (calls.length === 1) bytes = prefix.buffer;
    else if (calls.length === 2) bytes = encoded.buffer;
    else {
      assert.equal(options.headers.Range, `bytes=${dataOffset + 99 * header.frameBytes}-${dataOffset + 100 * header.frameBytes - 1}`);
      bytes = new ArrayBuffer(header.frameBytes);
      new Float32Array(bytes)[0] = 42;
    }
    return {status: 206, arrayBuffer: async () => bytes};
  };
  const loaded = await readAnimationHeader('only-file.4dgs', undefined, fetcher);
  const sample = await readAnimationFrame('only-file.4dgs', loaded, 99, undefined, fetcher);
  assert.equal(sample[0], 42);
  assert.equal(calls.length, 3);
  assert.ok(loaded.dataOffset + 99 * loaded.frameBytes > 2 ** 32);
});

test('parameter samples update all render fields in one persistent buffer, and stopped IDs become transparent', () => {
  const splats = new PackedSplats({maxSplats: 2, lod: false});
  const first = Float32Array.from([1, 2, 3, -.5, .1, .4, 1, -3, -4, -5, 1, 0, 0, 0,
    4, 5, 6, .3, .2, -.2, -.5, -2, -3, -4, .7, .7, 0, 0]);
  const second = Float32Array.from([-1, 3, 5, .6, -.2, .9, -.8, -2, -5, -3, .5, .5, .5, .5,
    0, 0, 1, 0, 0, 0, -Infinity, 0, 0, 0, 1, 0, 0, 0]);
  try {
    applyAnimationFrame(splats, first);
    const buffer = splats.packedArray;
    const old = buffer.slice(0, 8);
    applyAnimationFrame(splats, second);
    assert.equal(splats.packedArray, buffer);
    assert.notDeepEqual(buffer.slice(0, 4), old.slice(0, 4));
    assert.equal(splats.getSplat(1).opacity, 0);
    assert.equal(splats.needsUpdate, true);
    const expected = new PackedSplats({maxSplats: 2, lod: false});
    try {
      for (let id = 0; id < 2; id++) {
        const p = second.subarray(id * 14, (id + 1) * 14);
        expected.setSplat(id, new Vector3(...p.subarray(0, 3)),
          new Vector3(...p.subarray(7, 10)).set(Math.exp(p[7]), Math.exp(p[8]), Math.exp(p[9])),
          new Quaternion(p[11], p[12], p[13], p[10]), 1 / (1 + Math.exp(-p[6])),
          new Color().setRGB(.5 + .28209479177387814 * p[3], .5 + .28209479177387814 * p[4], .5 + .28209479177387814 * p[5]));
      }
      assert.deepEqual(buffer.slice(0, 8), expected.packedArray.slice(0, 8));
    } finally { expected.dispose(); }
  } finally { splats.dispose(); }
});

for (const format of [ANIMATION_FORMAT, LIFECYCLE_FORMAT]) {
test(`${format}: viewer reads only the file and reuses one mesh (no WebGL or browser)`, async () => {
  const parameterOffset = format === LIFECYCLE_FORMAT ? 4 : 0;
  const header = {format, dtype: '<f4', columns: ANIMATION_COLUMNS,
    cellCount: 1, parameterOffset, trackCount: 1, frameCount: 2, frameBytes: 56 + parameterOffset,
    timesSeconds: [0, .04], fps: 25, frameIndices: [0, 1]};
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const dataOffset = Math.ceil((16 + encoded.length) / 8) * 8;
  const bytes = new Uint8Array(dataOffset + 2 * header.frameBytes);
  bytes.set(new TextEncoder().encode('SH4DGS01'));
  new DataView(bytes.buffer).setUint32(8, encoded.length, true);
  bytes.set(encoded, 16);
  new Float32Array(bytes.buffer, dataOffset + parameterOffset, 14)
    .set([1, 2, 3, 0, 0, 0, 1, -3, -3, -3, 1, 0, 0, 0]);
  new Float32Array(bytes.buffer, dataOffset + header.frameBytes + parameterOffset, 14)
    .set([4, 5, 6, .2, .1, -.1, 2, -2, -4, -3, .7, .7, 0, 0]);
  if (parameterOffset) {
    new DataView(bytes.buffer).setUint32(dataOffset, 10, true);
    new DataView(bytes.buffer).setUint32(dataOffset + header.frameBytes, 11, true);
  }
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, options) => {
    assert.equal(url, 'isolated.4dgs');
    requests.push(url);
    const [, from, to] = /^bytes=(\d+)-(\d+)$/.exec(options.headers.Range);
    return {status: 206, arrayBuffer: async () => bytes.slice(Number(from), Number(to) + 1).buffer};
  };
  // Bypass the DOM/WebGL constructor. Exercise only file loading and CPU data.
  const viewer = Object.create(TrackSequenceViewer.prototype);
  viewer.root = new Group();
  viewer.camera = new PerspectiveCamera();
  viewer.onPlayback = viewer.onFrame = () => {};
  try {
    await viewer.load({positionEncoding: 'full_attribute_samples', file: {url: 'isolated.4dgs'}});
    const mesh = viewer.currentMesh;
    assert.equal(viewer.current, 0);
    assert.equal(mesh.packedSplats.getSplat(0).center.x, 1);
    await viewer.show(1);
    assert.equal(viewer.currentMesh, mesh);
    assert.equal(mesh.packedSplats.getSplat(0).center.x, 4);
    assert.equal(viewer.current, 1);
    assert.equal(viewer.root.children.length, 1);
    assert.equal(requests.length, 4); // prefix, header, first sample, second sample
  } finally {
    viewer.clearMeshes();
    globalThis.fetch = originalFetch;
  }
});
}
