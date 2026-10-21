import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {Matrix4, Object3D, PerspectiveCamera, Vector3} from 'three';
import {FrameClock} from '../static/compression/frame-clock.js';
import {CompressedSequenceViewer} from '../static/compression/viewer.js';
import {getProjectionMatrix, viewMatrixFromHint} from '../static/compression/camera.js';

function cameraState() {
  const camera = new Object3D();
  camera.rotation.order = 'YXZ';
  return {camera, cameraView: new Matrix4(), yaw: 0, pitch: 0, keys: new Set(), lastRenderAt: 0};
}

function assertClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12, `${i}: ${value} != ${expected[i]}`));
}

test('25 fps on a 60 Hz draw loop visits every source frame in order', () => {
  const clock = new FrameClock(100, 25);
  clock.play(0);
  const shown = [];
  for (let i = 1; i <= 721; i++) {
    const frame = clock.tick(i * 1000 / 60);
    if (frame !== null) shown.push(frame);
  }
  assert.equal(shown.length, 300);
  assert.ok(shown.every((frame, i) => frame === (i + 1) % 100));
});

test('slow preparation/draw never discards frames to catch up with wall time', () => {
  const clock = new FrameClock(100, 25);
  clock.play(0);
  const shown = [];
  for (let t = 103; t < 4000; t += 103) shown.push(clock.tick(t));
  assert.deepEqual(shown, Array.from({length: shown.length}, (_,i) => i + 1));
});

test('pause, seek and loop use the selected source frame', () => {
  const clock = new FrameClock(100, 25);
  clock.play(0); clock.pause();
  assert.equal(clock.tick(1000), null);
  clock.seek(99); clock.play(1000);
  assert.equal(clock.tick(1039), null);
  assert.equal(clock.tick(1040), 0);
});

test('main workflow contains exactly five steps and compression controls replace old steps', async () => {
  const html = await readFile(new URL('../static/index.html', import.meta.url),'utf8');
  assert.equal((html.match(/id="step-/g) || []).length,5);
  assert.ok(html.includes('id="step-compression"'));
  assert.ok(html.includes('id="compression-progress"'));
  assert.ok(html.includes('id="compression-viewer"'));
  assert.ok(!/id="(?:step-tracks|step-median|track-panel|median-panel)"/.test(html));
});

test('prepared production playback switches resident frames without worker decoding or re-upload', () => {
  const viewer = Object.create(CompressedSequenceViewer.prototype);
  const shown = [], drawn = [];
  const identity = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
  Object.assign(viewer,{visible:true, ready:true, host:{hidden:false}, width:1280, height:720, fov:38.94,
    ...cameraState(), view:identity.slice(), viewKey:0,
    clock:new FrameClock(100,25), current:0, state:{frameCount:100}, presentations:[],
    callbacks:{onFrame:frame=>shown.push(frame)},
    worker:{postMessage(){assert.fail('Prepared playback must not request worker decoding/sorting for the unchanged camera');}},
    renderer:{render(sets){drawn.push(sets[0].frame);}},
    sets:Array.from({length:100},(_,frame)=>({viewKey:0,set:{frame,setCamera(){},uploadTexture(){assert.fail('No upload during playback');}}}))});
  const previousDocument = globalThis.document;
  globalThis.document = {activeElement: null};
  try {
    viewer.clock.play(0);
    for(let i=1;i<=241;i++)viewer.tick(i*1000/60);
  } finally { globalThis.document = previousDocument; }
  assert.equal(shown.length,100);
  assert.ok(shown.every((frame,i)=>frame===(i+1)%100));
  assert.ok(drawn.includes(99));
});

test('production loader prepares all resident frames before enabling playback', async () => {
  const previousWorker = globalThis.Worker, previousLocation = globalThis.location;
  const progress = [];
  class WorkerMock {
    emit(data) { queueMicrotask(()=>this.onmessage({data})); }
    postMessage(m) {
      if(m.type==='load') {
        this.emit({type:'meta',n:1,t:2,fps:25});
        this.emit({type:'static',texdata:new Uint32Array(2048*4),texwidth:2048,texheight:1});
      } else if(m.type==='prepare-frame') {
        this.emit({type:'frame',frame:m.frame,band:new ArrayBuffer(2048*16),rows:1,rowStart:0,approximate:false});
        this.emit({type:'sorted',frame:m.frame,viewKey:0,indices:new Uint32Array([0]).buffer,count:1});
      } else if(m.type==='prepare-complete')this.emit({type:'ready',frames:2});
    }
    terminate() {}
  }
  globalThis.Worker=WorkerMock;
  globalThis.location={href:'http://localhost/'};
  try {
    const viewer=Object.create(CompressedSequenceViewer.prototype);
    Object.assign(viewer,{initialized:Promise.resolve(),sets:[],...cameraState(),width:1280,height:720,
      sortBusy:true,callbacks:{onProgress:p=>progress.push(p)},resize(){},
      renderer:{createSet:()=>({uploadTexture(){},setIndices(){},dispose(){}})}});
    await viewer.load({frameCount:2,fps:25,url:'/sequence.splat4d',camera:{position:[0,0,0],target:[0,0,1],fov:38.94}});
    assert.equal(viewer.ready,true);
    assert.equal(viewer.sets.length,2);
    assert.equal(viewer.sortBusy,false);
    assert.equal(progress.at(-1).complete,true);
    assert.equal(progress.at(-1).ready,2);
    viewer.clear();
    assert.equal(viewer.ready,false);
  } finally {globalThis.Worker=previousWorker;globalThis.location=previousLocation;}
});

test('compressed camera preserves the initial OpenCV pose through the shared camera', () => {
  const viewer = Object.assign(Object.create(CompressedSequenceViewer.prototype), cameraState());
  for (const hint of [
    {position: [0,0,0], target: [0,0,1]},
    {position: [1,-.2,.7], target: [.2,.4,3]},
  ]) {
    const view = viewMatrixFromHint(hint);
    viewer.setCameraView(view);
    assertClose(viewer.getCameraView(), view);
    assert.equal(viewer.yaw, viewer.camera.rotation.y);
    assert.equal(viewer.pitch, viewer.camera.rotation.x);
  }
});

test('compressed camera motion projects points in the same directions as the PLY camera', () => {
  const viewer = Object.assign(Object.create(CompressedSequenceViewer.prototype), cameraState());
  const plyCamera = new PerspectiveCamera(70, 1280 / 720, .01, 1000);
  plyCamera.rotation.order = 'YXZ';
  const projection = new Matrix4().fromArray(getProjectionMatrix(
    360 / Math.tan(70 * Math.PI / 360), 360 / Math.tan(70 * Math.PI / 360), 1280, 720));
  for (const pose of [[0,0,0,0,0], [.3,-.2,.5,.2,-.3], [-.2,.4,-.1,-.4,.5]]) {
    for (const camera of [plyCamera, viewer.camera]) {
      camera.position.set(...pose.slice(0,3));
      camera.rotation.set(pose[3], pose[4], 0);
    }
    plyCamera.updateMatrixWorld(true);
    const compressedView = new Matrix4().fromArray(viewer.getCameraView());
    for (const [x,y,z] of [[.2,.3,3], [-1,-.4,4], [.5,-.8,2]]) {
      // Step 4's scene root applies Rx(pi) to the original SHARP points.
      const ply = new Vector3(x,-y,-z).project(plyCamera);
      const compressed = new Vector3(x,y,z).applyMatrix4(compressedView).applyMatrix4(projection);
      assertClose([compressed.x, compressed.y], [ply.x, ply.y]);
    }
  }
});

test('starting compressed playback focuses the same control host as PLY playback', () => {
  let focused = 0;
  const viewer = Object.assign(Object.create(CompressedSequenceViewer.prototype), {
    ready: true, clock: new FrameClock(100,25), callbacks: {}, host: {focus() { focused++; }},
  });
  viewer.toggle();
  assert.equal(viewer.clock.playing, true);
  assert.equal(focused, 1);
  viewer.toggle();
  assert.equal(viewer.clock.playing, false);
});
