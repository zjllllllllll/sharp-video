import assert from 'node:assert/strict';
import {test} from 'node:test';
import {parseGaussianPly, remapGaussianPly} from '../static/indexed-ply.js';

const fields = 'x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3'.split(' ');
function fixture(frame = 0) {
  const header = new TextEncoder().encode('ply\nformat binary_little_endian 1.0\nelement vertex 6\n'
    + fields.map(name => `property float ${name}\n`).join('')
    + 'element version 3\nproperty uchar version\nend_header\n');
  const bytes = new Uint8Array(header.length + 6 * 56 + 3);
  bytes.set(header);
  const view = new DataView(bytes.buffer);
  for (let id = 0; id < 6; id++) {
    for (let field = 0; field < fields.length; field++) {
      // Frame-dependent changes deliberately differ between the two layers.
      view.setFloat32(header.length + id * 56 + field * 4, id * 20 + field + frame * (id + 1) / 4, true);
    }
  }
  bytes.set([1, 0, 0], bytes.length - 3);
  return bytes;
}
function record(bytes, id) {
  const {headerEnd, stride} = parseGaussianPly(bytes);
  return bytes.subarray(headerEnd + id * stride, headerEnd + (id + 1) * stride);
}

test('identity preserves every vertex byte, header and metadata', () => {
  const original = fixture();
  assert.deepEqual(remapGaussianPly(original, Int32Array.of(0, 1, 2), 2), original);
});

test('all attributes follow the current source record, with independent layers and duplicate IDs', () => {
  const indices = Int32Array.of(2, 2, 0);
  const originals = [fixture(0), fixture(1)];
  const snapshots = originals.map(bytes => bytes.slice());
  const outputs = originals.map(bytes => remapGaussianPly(bytes, indices, 2));
  for (let frame = 0; frame < 2; frame++) {
    for (let layer = 0; layer < 2; layer++) {
      for (let cell = 0; cell < 3; cell++) {
        assert.deepEqual(record(outputs[frame], layer * 3 + cell), record(originals[frame], layer * 3 + indices[cell]));
      }
    }
  }
  assert.deepEqual(originals, snapshots);
  const position = (frame, id) => new DataView(record(outputs[frame], id).slice().buffer).getFloat32(0, true);
  assert.notEqual(position(1, 0) - position(0, 0), position(1, 3) - position(0, 3));
  for (let field = 0; field < fields.length; field++) {
    const value = frame => new DataView(record(outputs[frame], 0).slice().buffer).getFloat32(field * 4, true);
    assert.notEqual(value(0), value(1));
  }
});

test('stopped slots use finite geometry and zero opacity, with no stale appearance', () => {
  const source = fixture(1);
  const output = remapGaussianPly(source, Int32Array.of(-1, 1, -1), 2);
  const {headerEnd, vertexCount, stride, offsets} = parseGaussianPly(output);
  assert.deepEqual(output.subarray(0, headerEnd), source.subarray(0, headerEnd));
  assert.deepEqual(output.subarray(headerEnd + vertexCount * stride), source.subarray(headerEnd + vertexCount * stride));
  for (const id of [0, 2, 3, 5]) {
    const view = new DataView(record(output, id).slice().buffer);
    for (const field of fields) {
      const value = view.getFloat32(offsets[field], true);
      if (field === 'opacity') {
        assert.equal(value, -Infinity);
        assert.equal(1 / (1 + Math.exp(-value)), 0);
      } else {
        assert.equal(Number.isFinite(value), true);
        assert.equal(value, field === 'z' || field === 'rot_0' ? 1 : 0);
      }
    }
  }
  assert.deepEqual(record(output, 1), record(source, 1));
  assert.deepEqual(record(output, 4), record(source, 4));
});
