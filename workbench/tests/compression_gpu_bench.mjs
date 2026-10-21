// Native Dawn WebGPU benchmark: same production decoder, textures, sorting,
// shader and draw calls. No browser, webpage interaction or image inspection.
import {Worker} from 'node:worker_threads';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {SplatRenderer} from '../static/compression/renderer.js';
import {getProjectionMatrix, multiply4, viewMatrixFromHint} from '../static/compression/camera.js';
import {FrameClock} from '../static/compression/frame-clock.js';

const appDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const file = path.resolve(args[0]);
const output = path.resolve(args[1]);
const {create, globals} = await import(pathToFileURL(path.join(appDir, '../.cache/splats4d-runtime/node_modules/webgpu/index.js')));
Object.assign(globalThis, globals);
const gpu = create([]);
Object.defineProperty(globalThis.navigator, 'gpu', {value: gpu, configurable: true});
const adapter = await gpu.requestAdapter({powerPreference: 'high-performance'});
const info = {vendor: adapter.info.vendor, architecture: adapter.info.architecture,
  device: adapter.info.device, description: adapter.info.description};
console.log('GPU', info);
const width = 1280, height = 720;
let target;
const canvas = {getContext: () => ({
  configure({device, format}) { target = device.createTexture({size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT}); },
  getCurrentTexture() { return target; }
})};
const renderer = await SplatRenderer.create(canvas);
const validation = [];
renderer.device.addEventListener('uncapturederror', event => validation.push(String(event.error.message)));
const sourceMetaPath = path.join(path.dirname(file), 'compression.json');
const state = fs.existsSync(sourceMetaPath) ? JSON.parse(fs.readFileSync(sourceMetaPath, 'utf8')) : null;
const fov = state?.camera?.fov || 38.94;
const fy = height / (2 * Math.tan(fov * Math.PI / 360));
const projection = getProjectionMatrix(fy, fy, width, height);
const view = viewMatrixFromHint({position: [0, 0, 0], target: [0, 0, 1]});
const viewProj = new Float32Array(multiply4(projection, view));
const sets = [];
const worker = new Worker(new URL('./compression-worker-node.mjs', import.meta.url), {workerData: {file}});
let meta, base, tw, th;
let preparationComplete = false;
let resortResolve;
const started = performance.now();
const prepare = frame => worker.postMessage({type: 'prepare-frame', frame, viewKey: 0, viewProj});
await new Promise((resolve, reject) => {
  worker.on('error', reject);
  worker.on('message', async m => {
    try {
      if (m.type === 'node-ready') worker.postMessage({type: 'load', url: 'http://file-only/sequence.splat4d'});
      else if (m.type === 'error') reject(new Error(m.message));
      else if (m.type === 'meta') meta = m;
      else if (m.type === 'static') { base = m.texdata; tw = m.texwidth; th = m.texheight; prepare(0); }
      else if (m.type === 'frame') {
        if (!m.approximate) {
          const set = renderer.createSet();
          if (m.rowStart === 0 && m.rows === th) set.uploadTexture(new Uint32Array(m.band), tw, th);
          else { set.uploadTexture(base, tw, th); set.uploadTexRows(new Uint32Array(m.band), m.rowStart, m.rows); }
          set.setCamera(projection, view, fy, fy, width, height);
          sets[m.frame] = set;
        }
        worker.postMessage({type: 'return', kind: 'band', buffer: m.band}, [m.band]);
      } else if (m.type === 'sorted') {
        sets[m.frame].setIndices(new Uint32Array(m.indices, 0, m.count), m.count);
        worker.postMessage({type: 'return', kind: 'sort', buffer: m.indices}, [m.indices]);
        if (preparationComplete) { resortResolve?.(m); return; }
        if (m.frame === 0) renderer.render([sets[0]]); // Compile/warm up while preparing.
        if ((m.frame + 1) % 10 === 0) console.log('Prepared',m.frame + 1,'/',meta.t);
        await renderer.device.queue.onSubmittedWorkDone();
        if (m.frame + 1 < meta.t) prepare(m.frame + 1);
        else worker.postMessage({type: 'prepare-complete'});
      } else if (m.type === 'ready') { preparationComplete = true; resolve(); }
    } catch (error) { reject(error); }
  });
});
await renderer.device.queue.onSubmittedWorkDone();
const prepareSeconds = (performance.now() - started) / 1000;
base = null;
const timings = [];
for (let pass = 0; pass < 3; pass++) {
  for (let frame = 0; frame < meta.t; frame++) {
    const before = performance.now();
    renderer.render([sets[frame]]);
    await renderer.device.queue.onSubmittedWorkDone();
    timings.push(performance.now() - before);
  }
}
timings.sort((a,b) => a-b);
const mean = timings.reduce((a,b) => a+b,0) / timings.length;
const movedView = viewMatrixFromHint({position: [.025,0,0], target: [0,0,1]});
const resorted = await new Promise(resolve => {
  resortResolve = resolve;
  worker.postMessage({type:'sort-frame', frame:50, viewKey:1,
    viewProj:new Float32Array(multiply4(projection,movedView))});
});
if (resorted.frame !== 50 || resorted.viewKey !== 1 || resorted.count !== meta.n) throw Error('Prepared-camera sort response mismatch');
const clock = new FrameClock(meta.t, meta.fps);
clock.play(0);
const sequence = [];
for (let t = 1000/60; t <= 12000 + 0.001; t += 1000/60) {
  const frame = clock.tick(t);
  if (frame !== null) sequence.push(frame);
}
const ordered = sequence.every((frame, i) => frame === (i + 1) % meta.t);
const report = {gpu: info, resolution: [width, height], frames: meta.t, gaussians: meta.n, fps: meta.fps,
  preparedGpuBytes: sets.reduce((n,s) => n + s.texWidth*s.texHeight*16 + meta.n*4,0), prepareSeconds,
  samples: timings.length, averageDrawMs: mean, p95DrawMs: timings[Math.floor(timings.length * .95)],
  p99DrawMs: timings[Math.floor(timings.length*.99)], maxDrawMs: timings.at(-1),
  frameBudgetMs: 1000/meta.fps, sequentialClock: ordered, presentedIn12s: sequence.length,
  preparedCameraSortMs: resorted.sortMs, validationErrors: validation};
fs.writeFileSync(output, JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
await worker.terminate();
for (const set of sets) set.dispose();
target.destroy(); renderer.device.destroy();
process.exit(validation.length || !ordered || report.p95DrawMs > report.frameBudgetMs ? 1 : 0);
