import * as THREE from 'three';
import {SparkRenderer, SplatMesh} from '@sparkjsdev/spark';

export class PlySequenceViewer {
  constructor(host, {onFrame, onPlayback, onBuffering, onPerformance, onError} = {}) {
    this.host = host;
    this.onFrame = onFrame || (() => {});
    this.onPlayback = onPlayback || (() => {});
    this.onBuffering = onBuffering || (() => {});
    this.onError = onError || (() => {});
    this.onPerformance = onPerformance || (() => {});
    this.frames = [];
    this.cache = new Map();
    this.current = -1;
    this.currentMesh = null;
    this.playing = false;
    this.switching = false;
    this.nextFrameAt = 0;
    this.fps = 25;
    this.yaw = 0;
    this.pitch = 0;
    this.keys = new Set();
    this.drag = null;
    this.lastRenderAt = performance.now();
    this.initializeScene();
    this.bindControls();
  }

  initializeScene() {
    this.renderer = new THREE.WebGLRenderer({antialias: false});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setClearColor(0x070908);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    const gl = this.renderer.getContext();
    const rendererInfo = gl.getExtension('WEBGL_debug_renderer_info');
    this.renderer.domElement.dataset.renderer = rendererInfo ? gl.getParameter(rendererInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    this.host.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(70, 1, 0.01, 1000);
    this.camera.rotation.order = 'YXZ';
    this.root = new THREE.Group();
    this.root.rotation.x = Math.PI;
    this.scene.add(this.root);
    this.spark = new SparkRenderer({renderer: this.renderer});
    this.scene.add(this.spark);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.host);
    this.resize();

    this.renderer.setAnimationLoop(now => this.tick(now));
  }

  bindControls() {
    this.host.addEventListener('contextmenu', event => event.preventDefault());
    this.host.addEventListener('pointerdown', event => {
      if (event.target.closest('.viewer-controls')) return;
      this.drag = {x: event.clientX, y: event.clientY, button: event.button};
      this.host.setPointerCapture(event.pointerId);
      this.host.focus();
    });
    this.host.addEventListener('pointerup', () => { this.drag = null; });
    this.host.addEventListener('pointercancel', () => { this.drag = null; });
    this.host.addEventListener('pointermove', event => {
      if (!this.drag) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      if (this.drag.button === 2) {
        this.camera.translateX(-dx * 0.003);
        this.camera.translateY(dy * 0.003);
      } else {
        this.yaw -= dx * 0.003;
        this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch - dy * 0.003));
        this.camera.rotation.set(this.pitch, this.yaw, 0);
      }
    });
    this.host.addEventListener('wheel', event => {
      event.preventDefault();
      this.camera.translateZ(Math.max(-0.5, Math.min(0.5, event.deltaY * 0.002)));
    }, {passive: false});
    this.host.addEventListener('keydown', event => {
      if ('wasdqeWASDQE'.includes(event.key) || event.key === 'Shift') {
        this.keys.add(event.key);
        event.preventDefault();
      }
    });
    window.addEventListener('keyup', event => this.keys.delete(event.key));
    window.addEventListener('blur', () => this.keys.clear());
    this.host.addEventListener('blur', () => this.keys.clear());
  }

  resize() {
    const width = Math.max(1, this.host.clientWidth);
    const height = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  async load(manifest) {
    this.stop();
    this.clearMeshes();
    this.frames = manifest.frames || [];
    this.fps = Number(manifest.fps) || 25;
    this.current = -1;
    this.camera.position.set(0, 0, 0);
    this.yaw = 0;
    this.pitch = 0;
    this.camera.rotation.set(0, 0, 0);
    if (!this.frames.length) throw new Error('PLY 序列为空');
    await this.show(0);
    this.preloadFrom(0);
  }

  async ensure(index) {
    const normalized = this.normalize(index);
    const cached = this.cache.get(normalized);
    if (cached) return cached.promise;

    const entry = {mesh: null, promise: null};
    entry.promise = (async () => {
      const mesh = new SplatMesh({url: this.frames[normalized].fullUrl, lod: false});
      mesh.visible = false;
      this.root.add(mesh);
      try {
        await mesh.initialized;
        mesh.splats.prepareFetchSplat();
        if (mesh.packedSplats) this.renderer.initTexture(mesh.packedSplats.getTexture());
        entry.mesh = mesh;
        return mesh;
      } catch (error) {
        this.root.remove(mesh);
        mesh.dispose();
        this.cache.delete(normalized);
        throw error;
      }
    })();
    this.cache.set(normalized, entry);
    return entry.promise;
  }

  normalize(index) {
    const count = this.frames.length;
    return ((Number(index) % count) + count) % count;
  }

  async show(index) {
    const target = this.normalize(index);
    const mesh = await this.ensure(target);
    if (this.currentMesh && this.currentMesh !== mesh) this.currentMesh.visible = false;
    mesh.visible = true;
    this.currentMesh = mesh;
    this.current = target;
    this.onFrame(target, this.frames.length);
    this.preloadFrom(target);
  }

  preloadFrom(index) {
    void (async () => {
      for (let offset = 1; offset < this.frames.length; offset += 1) {
        try { await this.ensure(index + offset); }
        catch (error) { this.onError(error); return; }
      }
    })();
  }

  async fillBuffer() {
    this.onBuffering(true);
    try {
      for (let offset = 1; offset < this.frames.length; offset += 1) {
        await this.ensure(this.current + offset);
      }
    } finally {
      this.onBuffering(false);
    }
  }

  async toggle() {
    if (this.playing) {
      this.stop();
      return;
    }
    await this.fillBuffer();
    this.playing = true;
    this.nextFrameAt = performance.now() + 1000 / this.fps;
    this.onPlayback(true);
    this.host.focus();
  }

  stop() {
    this.playing = false;
    this.onPlayback(false);
  }

  tick(now) {
    const callbackIntervalMs = now - this.lastRenderAt;
    const delta = Math.min((now - this.lastRenderAt) / 1000, 0.1);
    this.lastRenderAt = now;
    if (this.host.hidden) return;

    if (document.activeElement === this.host) {
      const speed = delta * (this.keys.has('Shift') ? 3 : 1);
      if (this.keys.has('w') || this.keys.has('W')) this.camera.translateZ(-speed);
      if (this.keys.has('s') || this.keys.has('S')) this.camera.translateZ(speed);
      if (this.keys.has('a') || this.keys.has('A')) this.camera.translateX(-speed);
      if (this.keys.has('d') || this.keys.has('D')) this.camera.translateX(speed);
      if (this.keys.has('q') || this.keys.has('Q')) this.camera.position.y -= speed;
      if (this.keys.has('e') || this.keys.has('E')) this.camera.position.y += speed;
    }

    if (this.playing && !this.switching && !this.rendering && now >= this.nextFrameAt) {
      this.switching = true;
      // Advance the deadline, rather than adding 40 ms after processing finishes.
      this.nextFrameAt = Math.max(this.nextFrameAt + 1000 / this.fps, now);
      const next = this.current + 1;
      this.show(next).catch(error => {
        this.stop();
        this.onError(error);
      }).finally(() => {
        this.switching = false;
      });
    }
    this.render(now, {callbackIntervalMs, visibility: document.visibilityState});
  }

  render(now, timings = {}) {
    const renderStarted = performance.now();
    this.renderer.render(this.scene, this.camera);
    this.didRenderFrame?.(now, {...timings, renderMs: performance.now() - renderStarted});
  }

  clearMeshes() {
    for (const entry of this.cache.values()) {
      if (!entry.mesh) continue;
      this.root.remove(entry.mesh);
      entry.mesh.dispose();
    }
    this.cache.clear();
    this.currentMesh = null;
  }
}
