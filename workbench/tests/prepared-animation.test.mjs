import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PackedSplats, SplatGenerator} from '@sparkjsdev/spark';
import {applyAnimationFrame} from '../static/animation-splats.js';
import {packAnimationFrame} from '../static/animation-pack.js';
import {PreparedSplats} from '../static/prepared-animation.js';
import {PlySequenceViewer} from '../static/ply-viewer.js';
import {CodecSequenceViewer} from '../static/codec-viewer.js';

test('worker packing matches Spark byte-for-byte, including edge cases', () => {
  let seed = 247;
  const random = () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 2 ** 32);
  const p = new Float32Array(10000 * 14);
  for (let i = 0; i < p.length; i += 14) {
    for (let j = 0; j < 14; j++) p[i + j] = (random() - .5) * 12;
    p[i + 7] = (random() - .5) * 80;
  }
  p[0] = Infinity; p[1] = -Infinity; p[2] = NaN;
  p[6] = -Infinity;
  p[10] = p[11] = p[12] = p[13] = 0;
  const source = new PackedSplats({maxSplats: 10000, lod: false});
  try {
    applyAnimationFrame(source, p);
    assert.deepEqual(packAnimationFrame(p), source.packedArray);
  } finally { source.dispose(); }
});

test('frame selection reuses pre-uploaded textures and keeps full attributes', () => {
  const source = new PreparedSplats(1);
  const a = Float32Array.from([1, 2, 3, 0, 0, 0, 2, -3, -3, -3, 1, 0, 0, 0]);
  const b = a.slice(); b[0] = 10; b[3] = 1; b[6] = -1;
  const uploads = [];
  const renderer = {initTexture: texture => uploads.push(texture)};
  source.add(0, packAnimationFrame(a), renderer);
  source.add(1, packAnimationFrame(b), renderer);
  try {
    source.select(0);
    assert.equal(source.getSplat(0).center.x, 1);
    assert.equal(source.getTexture(), uploads[0]);
    const version = uploads[0].version;
    source.select(1);
    assert.equal(source.getSplat(0).center.x, 10);
    assert.equal(source.getTexture(), uploads[1]);
    source.select(0);
    assert.equal(source.getTexture().version, version);
    assert.equal(uploads.length, 2);
  } finally { source.dispose(); }
  assert.equal(source.frames.length, 0);
});

test('25 FPS deadlines do not add processing time or skip animation frames', async () => {
  const displayed = [];
  const viewer = Object.create(PlySequenceViewer.prototype);
  Object.assign(viewer, {host: {hidden: false}, playing: true, switching: false,
    fps: 25, nextFrameAt: 40, lastRenderAt: 0, current: 0,
    renderer: {render() {}}, show(index) { this.current = index; displayed.push(index); return Promise.resolve(); }});
  const previous = globalThis.document;
  globalThis.document = {activeElement: null};
  try {
    for (let now = 0; now <= 4000; now += 1000 / 60) {
      viewer.tick(now);
      await Promise.resolve(); await Promise.resolve();
    }
    assert.ok(displayed.length >= 99 && displayed.length <= 100);
    assert.deepEqual(displayed, Array.from({length: displayed.length}, (_, i) => i + 1));
  } finally { globalThis.document = previous; }
});

test('prepared playback uses its timer once, ignores rAF, and cancels on pause', async () => {
  const originalTimeout = globalThis.setTimeout, originalClear = globalThis.clearTimeout;
  const originalDocument = globalThis.document;
  let callback, timerId = 0, renders = 0;
  globalThis.setTimeout = fn => { callback = fn; return ++timerId; };
  globalThis.clearTimeout = () => { callback = undefined; };
  globalThis.document = {activeElement: null, visibilityState: 'visible'};
  const viewer = Object.create(CodecSequenceViewer.prototype);
  Object.assign(viewer, {preparation: {complete: true}, playing: true, host: {hidden: false},
    fps: 25, nextFrameAt: performance.now() - 1, lastRenderAt: performance.now(), current: 0,
    renderer: {render() { renders++; }}, onPlayback() {}, didRenderFrame() {},
    show(index) { this.current = index; return Promise.resolve(); }});
  try {
    viewer.tick(performance.now());
    assert.equal(renders, 0);
    viewer.schedulePreparedFrame();
    callback();
    await Promise.resolve(); await Promise.resolve();
    assert.equal(renders, 1);
    assert.equal(viewer.current, 1);
    assert.equal(timerId, 2);
    viewer.stop();
    assert.equal(viewer.playing, false);
    assert.equal(callback, undefined);
  } finally {
    globalThis.setTimeout = originalTimeout;
    globalThis.clearTimeout = originalClear;
    globalThis.document = originalDocument;
  }
});

test('prepared playback never draws or advances attributes while their sort is pending', async () => {
  const previousDocument = globalThis.document;
  globalThis.document = {activeElement: null, visibilityState: 'visible'};
  const mesh = new SplatGenerator({});
  const initialMappingVersion = mesh.mappingVersion;
  const selected = [], drawn = [];
  let finishSort;
  const viewer = Object.create(CodecSequenceViewer.prototype);
  const spark = {sorting: false, autoUpdate: false, update() {
    this.sorting = true;
    return new Promise(resolve => { finishSort = () => { this.sorting = false; resolve(); }; });
  }};
  Object.assign(viewer, {standalone: true, preparation: {splats: {select: index => selected.push(index)}},
    currentMesh: mesh, frames: [0, 1, 2], current: 0, requestVersion: 0,
    playing: true, switching: false, fps: 25, nextFrameAt: 40, lastRenderAt: 0,
    host: {hidden: false}, scene: {updateMatrixWorld() {}}, camera: {updateMatrixWorld() {}}, spark,
    renderer: {render() { assert.equal(spark.sorting, false); drawn.push(viewer.current); }},
    onFrame() {}, onPlayback() {}, onError(error) { throw error; }, didRenderFrame() {}});
  try {
    PlySequenceViewer.prototype.tick.call(viewer, 40);
    await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
    assert.deepEqual(selected, [1]);
    assert.ok(mesh.mappingVersion > initialMappingVersion);
    assert.deepEqual(drawn, []); // The old path drew here with the previous frame's ordering.
    PlySequenceViewer.prototype.tick.call(viewer, 80);
    assert.deepEqual(selected, [1]); // Slow sorting must not let a second frame replace its input.
    finishSort();
    await viewer.rendering;
    assert.deepEqual(drawn, [1]);
    PlySequenceViewer.prototype.tick.call(viewer, 120);
    assert.deepEqual(selected, [1, 2]);
    finishSort();
    await viewer.rendering;
    assert.deepEqual(drawn, [1, 2]);
  } finally { globalThis.document = previousDocument; }
});
