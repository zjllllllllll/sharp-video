import test from 'node:test';
import assert from 'node:assert/strict';
import {Object3D, Vector3} from 'three';
import {bindViewerControls, updateViewerMovement} from '../static/viewer-controls.js';

// Event callbacks are exercised without a browser, canvas, or actual UI actions.
function surface() {
  const listeners = new Map();
  return {
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) { listeners.get(type)?.delete(fn); },
    emit(type, properties = {}) {
      const event = {target: {closest: () => null}, preventDefault() { this.prevented = true; }, ...properties};
      for (const fn of listeners.get(type) || []) fn(event);
      return event;
    },
    setPointerCapture(id) { this.captured = id; },
    focus() { document.activeElement = this; },
  };
}

function withControls(run) {
  const previousWindow = globalThis.window, previousDocument = globalThis.document;
  const host = surface(), win = surface(), camera = new Object3D();
  camera.rotation.order = 'YXZ';
  const viewer = {host, camera, yaw: 0, pitch: 0, keys: new Set(), drag: null};
  globalThis.window = win;
  globalThis.document = {activeElement: null};
  const unbind = bindViewerControls(viewer);
  try { run(viewer, win, unbind); }
  finally { unbind(); globalThis.window = previousWindow; globalThis.document = previousDocument; }
}

function closeVector(actual, expected) {
  assert.ok(actual.distanceTo(expected) < 1e-12, `${actual.toArray()} != ${expected.toArray()}`);
}

test('shared pointer callbacks retain PLY turning, pan and wheel sensitivity', () => withControls(viewer => {
  const {host, camera} = viewer;
  assert.equal(host.emit('contextmenu').prevented, true);
  host.emit('pointerdown', {clientX: 10, clientY: 20, button: 0, pointerId: 7});
  assert.equal(host.captured, 7);
  assert.equal(document.activeElement, host);
  host.emit('pointermove', {clientX: 110, clientY: 70});
  assert.equal(viewer.yaw, -.3);
  assert.equal(viewer.pitch, -.15);
  assert.equal(camera.rotation.y, -.3);
  assert.equal(camera.rotation.x, -.15);
  host.emit('pointermove', {clientX: 110, clientY: 10000});
  assert.equal(viewer.pitch, -1.55);
  host.emit('pointerup');
  assert.equal(viewer.drag, null);

  camera.rotation.set(.2,-.3,0);
  const expectedPan = new Vector3(-.03,.06,0).applyQuaternion(camera.quaternion);
  host.emit('pointerdown', {clientX: 10, clientY: 20, button: 2, pointerId: 8});
  host.emit('pointermove', {clientX: 20, clientY: 40});
  closeVector(camera.position, expectedPan);
  host.emit('pointercancel');
  assert.equal(viewer.drag, null);
  assert.equal(host.emit('wheel', {deltaY: 25}).prevented, true);
  closeVector(camera.position, expectedPan.clone().add(new Vector3(0,0,.05).applyQuaternion(camera.quaternion)));
}));

test('shared keys retain WASD/QE, Shift speed and focus/blur behavior', () => withControls((viewer, win) => {
  const {host, camera} = viewer;
  const directions = {w:[0,0,-.1], s:[0,0,.1], a:[-.1,0,0], d:[.1,0,0], q:[0,-.1,0], e:[0,.1,0]};
  host.focus();
  for (const [key, direction] of Object.entries(directions)) {
    for (const variant of [key, key.toUpperCase()]) {
      camera.position.set(0,0,0);
      assert.equal(host.emit('keydown', {key: variant}).prevented, true);
      updateViewerMovement(viewer, .1);
      closeVector(camera.position, new Vector3(...direction));
      win.emit('keyup', {key: variant});
      assert.equal(viewer.keys.size, 0);
    }
  }
  camera.position.set(0,0,0);
  host.emit('keydown', {key: 'W'});
  host.emit('keydown', {key: 'Shift'});
  updateViewerMovement(viewer, .1);
  closeVector(camera.position, new Vector3(0,0,-.3));
  document.activeElement = null;
  updateViewerMovement(viewer, .1);
  closeVector(camera.position, new Vector3(0,0,-.3));
  host.emit('blur');
  assert.equal(viewer.keys.size, 0);
  host.emit('keydown', {key: 'w'});
  win.emit('blur');
  assert.equal(viewer.keys.size, 0);
  assert.equal(host.emit('keydown', {key: 'ArrowUp'}).prevented, undefined);
}));

test('shared callbacks leave playback controls alone and can be removed', () => withControls((viewer, win, unbind) => {
  viewer.host.emit('pointerdown', {target: {closest: () => ({})}, clientX: 1, clientY: 1, button: 0});
  assert.equal(viewer.drag, null);
  assert.equal(document.activeElement, null);
  unbind();
  viewer.host.emit('pointerdown', {clientX: 1, clientY: 1, button: 0});
  viewer.host.emit('keydown', {key: 'w'});
  assert.equal(viewer.drag, null);
  assert.equal(viewer.keys.size, 0);
  viewer.keys.add('w');
  win.emit('keyup', {key: 'w'});
  assert.equal(viewer.keys.has('w'), true);
}));
