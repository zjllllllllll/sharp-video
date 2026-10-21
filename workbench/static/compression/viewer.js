import {Matrix4, Object3D} from 'three';
import {SplatRenderer} from './renderer.js';
import {GlSplatRenderer} from './renderer_gl.js';
import {getProjectionMatrix, multiply4, viewMatrixFromHint} from './camera.js';
import {FrameClock} from './frame-clock.js';
import {bindViewerControls, updateViewerMovement} from '../viewer-controls.js';

// Step 4 rotates SHARP's OpenCV world into Three.js with Rx(pi).
// Convert both world and camera axes so these renderers use the same controls.
const cameraAxes = new Matrix4().makeScale(1, -1, -1);

// Decoding, covariance construction and initial sorting finish before Play.
// Each frame owns a resident GPU texture/index buffer; playback selects it.
export class CompressedSequenceViewer {
  constructor(host, callbacks = {}) {
    this.host = host;
    this.callbacks = callbacks;
    this.canvas = document.createElement('canvas');
    host.prepend(this.canvas);
    this.camera = new Object3D();
    this.camera.rotation.order = 'YXZ';
    this.cameraView = new Matrix4();
    this.yaw = this.pitch = 0;
    this.keys = new Set();
    this.drag = null;
    this.lastRenderAt = performance.now();
    this.unbindControls = bindViewerControls(this);
    this.sets = [];
    this.visible = false;
    this.viewKey = 0;
    this.sortBusy = false;
    this.presentations = [];
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.initialized = this.initialize();
  }

  async initialize() {
    if (navigator.gpu) {
      try { this.renderer = await SplatRenderer.create(this.canvas); } catch { /* WebGL2 is the upstream fallback. */ }
    }
    if (!this.renderer) this.renderer = await GlSplatRenderer.create(this.canvas);
    const loop = now => { this.animationId = requestAnimationFrame(loop); this.tick(now); };
    this.animationId = requestAnimationFrame(loop);
  }

  resize() {
    const rect = this.host.getBoundingClientRect();
    this.width = Math.max(1, Math.round(rect.width));
    this.height = Math.max(1, Math.round(rect.height));
    const pixelRatio = Math.min(globalThis.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * pixelRatio);
    this.canvas.height = Math.round(this.height * pixelRatio);
  }

  async load(state) {
    await this.initialized;
    this.clear();
    this.visible = true;
    this.resize();
    this.state = state;
    this.fov = state.camera.fov;
    this.setCameraView(viewMatrixFromHint(state.camera));
    this.view = this.getCameraView();
    this.viewKey = 0;
    this.clock = new FrameClock(state.frameCount, state.fps);
    this.ready = false;
    this.worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});
    const worker = this.worker;
    this.loadPromise = new Promise((resolve, reject) => { this.loadResolve = resolve; this.loadReject = reject; });
    worker.onerror = event => this.fail(new Error(event.message));
    worker.onmessage = ({data: m}) => {
      if (worker !== this.worker) return;
      try {
        if (m.type === 'meta') { this.meta = m; }
        else if (m.type === 'static') {
          this.baseTexture = m.texdata;
          this.texWidth = m.texwidth;
          this.texHeight = m.texheight;
          this.preparationStarted = performance.now();
          this.prepareNext(0);
        } else if (m.type === 'frame') {
          if (!m.approximate) {
            const set = this.renderer.createSet();
            if (m.rowStart === 0 && m.rows === this.texHeight) {
              set.uploadTexture(new Uint32Array(m.band), this.texWidth, this.texHeight);
            } else {
              set.uploadTexture(this.baseTexture, this.texWidth, this.texHeight);
              set.uploadTexRows(new Uint32Array(m.band), m.rowStart, m.rows);
            }
            this.sets[m.frame] = {set, viewKey: 0};
          }
          worker.postMessage({type: 'return', kind: 'band', buffer: m.band}, [m.band]);
        } else if (m.type === 'sorted') {
          const cached = this.sets[m.frame];
          cached.set.setIndices(new Uint32Array(m.indices, 0, m.count), m.count);
          cached.viewKey = m.viewKey;
          worker.postMessage({type: 'return', kind: 'sort', buffer: m.indices}, [m.indices]);
          if (!this.ready) {
            this.callbacks.onProgress?.({ready: m.frame + 1, total: state.frameCount});
            if (m.frame === 0) this.select(0);
            if (m.frame + 1 < state.frameCount) this.prepareNext(m.frame + 1);
            else worker.postMessage({type: 'prepare-complete'});
          } else this.sortBusy = false;
        } else if (m.type === 'ready') {
          this.ready = true;
          this.baseTexture = null;
          this.callbacks.onProgress?.({ready: state.frameCount, total: state.frameCount, complete: true});
          this.loadResolve(this);
        } else if (m.type === 'error') this.fail(new Error(m.message));
      } catch (error) { this.fail(error); }
    };
    this.callbacks.onProgress?.({ready: 0, total: state.frameCount});
    worker.postMessage({type: 'load', url: new URL(state.url, location.href).href});
    return this.loadPromise;
  }

  projection() {
    const fy = 0.5 * this.height / Math.tan(this.fov * Math.PI / 360);
    return {fy, proj: getProjectionMatrix(fy, fy, this.width, this.height)};
  }

  setCameraView(view) {
    this.cameraView.fromArray(view).premultiply(cameraAxes).multiply(cameraAxes).invert()
      .decompose(this.camera.position, this.camera.quaternion, this.camera.scale);
    this.yaw = this.camera.rotation.y;
    this.pitch = this.camera.rotation.x;
  }

  getCameraView() {
    this.camera.updateMatrixWorld(true);
    return this.cameraView.copy(this.camera.matrixWorld).invert()
      .premultiply(cameraAxes).multiply(cameraAxes).toArray();
  }

  prepareNext(frame) {
    const {proj} = this.projection();
    this.worker.postMessage({type: 'prepare-frame', frame, viewKey: 0,
      viewProj: new Float32Array(multiply4(proj, this.view))});
  }

  select(frame) {
    this.clock.seek(frame);
    this.current = frame;
    this.callbacks.onFrame?.(frame, this.state.frameCount);
  }

  toggle() {
    if (!this.ready) return;
    if (this.clock.playing) this.stop();
    else {
      this.presentations = [];
      this.clock.play(performance.now());
      this.callbacks.onPlayback?.(true);
      this.host.focus();
    }
  }
  stop() { this.clock?.pause(); this.callbacks.onPlayback?.(false); }
  show(frame) { this.stop(); if (this.ready) this.select(frame); }
  hide() { this.stop(); this.visible = false; }

  tick(now) {
    const delta = Math.min((now - this.lastRenderAt) / 1000, 0.1);
    this.lastRenderAt = now;
    if (!this.visible || !this.sets.length || !this.renderer || this.host.hidden) return;
    updateViewerMovement(this, delta);
    const view = this.getCameraView();
    if (view.some((v, i) => v !== this.view[i])) { this.view = view.slice(); this.viewKey++; }
    const next = this.ready ? this.clock.tick(now) : null;
    if (next !== null) {
      this.select(next);
      this.presentations.push(now);
      while (this.presentations.length && this.presentations[0] < now - 1000) this.presentations.shift();
      this.callbacks.onPerformance?.(this.presentations.length);
    }
    const cached = this.sets[this.current];
    if (!cached) return;
    const {fy, proj} = this.projection();
    if (this.ready && cached.viewKey !== this.viewKey && !this.sortBusy) {
      this.sortBusy = true;
      this.worker.postMessage({type: 'sort-frame', frame: this.current, viewKey: this.viewKey,
        viewProj: new Float32Array(multiply4(proj, view))});
    }
    cached.set.setCamera(proj, view, fy, fy, this.width, this.height);
    this.renderer.render([cached.set]);
  }

  fail(error) { this.stop(); this.worker?.terminate(); this.loadReject?.(error); this.callbacks.onError?.(error); }
  clear() {
    this.stop(); this.worker?.terminate();
    this.loadReject?.(new DOMException('Aborted', 'AbortError'));
    for (const cached of this.sets) cached?.set.dispose();
    this.sets = []; this.current = -1; this.ready = false;
    this.sortBusy = false; this.presentations = [];
  }
  dispose() { this.clear(); this.unbindControls(); cancelAnimationFrame(this.animationId); this.resizeObserver.disconnect(); this.canvas.remove(); }
}
