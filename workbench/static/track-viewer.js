import * as THREE from 'three';
import {ExtSplats, PackedSplats, SplatMesh} from '@sparkjsdev/spark';
import {PlySequenceViewer} from './ply-viewer.js';
import {remapGaussianPly} from './indexed-ply.js';
import {readAnimationHeader, readAnimationFrame} from './animation-file.js';
import {applyAnimationFrame} from './animation-splats.js';
import {AnimationPreparation} from './prepared-animation.js';

async function readArray(spec, Type, frame = null, signal) {
  const start = spec.dataOffset + (frame === null ? 0 : frame * spec.frameBytes);
  const bytes = frame === null ? spec.dataBytes : spec.frameBytes;
  const response = await fetch(spec.url, {signal, headers: {Range: `bytes=${start}-${start + bytes - 1}`}});
  if (response.status !== 206) throw new Error(`轨迹数据读取失败：${response.status}`);
  return new Type(await response.arrayBuffer());
}

export class TrackSequenceViewer extends PlySequenceViewer {
  async load(manifest) {
    this.stop();
    this.clearMeshes();
    this.requestVersion = (this.requestVersion || 0) + 1;
    const loadVersion = this.requestVersion;
    this.manifest = manifest;
    this.standalone = manifest.positionEncoding === 'full_attribute_samples';
    this.indexed = manifest.positionEncoding === 'source_frame_indices';
    this.sharedMotion = manifest.positionEncoding === 'reference_plus_shared_displacement';
    this.frames = manifest.frameIndices;
    this.fps = manifest.fps;
    this.current = -1;
    this.camera.position.set(0, 0, 0);
    this.yaw = this.pitch = 0;
    this.camera.rotation.set(0, 0, 0);
    // The two-layer trial uses the original PLY viewer's field of view.
    this.camera.fov = this.standalone || this.sharedMotion || this.indexed ? 70 : THREE.MathUtils.radToDeg(2 * Math.atan(manifest.imageSizeWH[1] / (2 * manifest.intrinsics[1][1])));
    this.camera.updateProjectionMatrix();

    if (this.standalone) {
      await this.loadAnimation(manifest.file.url, loadVersion);
      return;
    }

    this.times = await readArray(manifest.arrays.times, Float64Array);
    if (loadVersion !== this.requestVersion) return;
    if (this.indexed) {
      await this.show(0);
      return;
    }

    const reference = await readArray(manifest.arrays.reference, Float32Array);
    if (loadVersion !== this.requestVersion) return;
    this.reference = reference;
    this.referenceValid = new Uint8Array(manifest.trackCount);
    const splats = new ExtSplats({maxSplats: manifest.trackCount, lod: false});
    const center = new THREE.Vector3(), scale = new THREE.Vector3();
    const quaternion = new THREE.Quaternion(), color = new THREE.Color();
    for (let i = 0; i < manifest.trackCount; i++) {
      const offset = i * 14;
      center.fromArray(reference, offset);
      this.referenceValid[i] = Number.isFinite(center.x) && Number.isFinite(center.y) && Number.isFinite(center.z);
      scale.fromArray(reference, offset + 3);
      quaternion.set(reference[offset + 7], reference[offset + 8], reference[offset + 9], reference[offset + 6]);
      color.setRGB(reference[offset + 10], reference[offset + 11], reference[offset + 12]);
      // Match SHARP's PLY export: Spark's stored splat colors are display sRGB.
      color.convertLinearToSRGB();
      splats.setSplat(i, center, scale, quaternion, reference[offset + 13], color);
    }
    splats.updateTextures();
    this.positionWords = splats.extArrays[0];
    this.positionFloats = new Float32Array(this.positionWords.buffer);
    this.opacityWords = new Uint32Array(manifest.trackCount);
    for (let i = 0; i < manifest.trackCount; i++) this.opacityWords[i] = this.positionWords[4 * i + 3];
    this.currentMesh = new SplatMesh({extSplats: splats, lod: false});
    await this.currentMesh.initialized;
    this.root.add(this.currentMesh);
    await this.show(0);
  }

  async show(index) {
    const version = ++this.requestVersion;
    const target = this.normalize(index);
    if (this.standalone) return this.showAnimationFrame(target, version);
    if (this.indexed) {
      this.indexedAbort?.abort();
      const manifest = this.manifest;
      // Decode at most one pending frame. Superseded queued requests skip work.
      this.indexedLoad = (this.indexedLoad || Promise.resolve()).catch(() => {})
        .then(() => this.loadIndexedFrame(manifest, target, version));
      return this.indexedLoad;
    }
    const [positions, valid] = await Promise.all([
      readArray(this.sharedMotion ? this.manifest.arrays.displacements : this.manifest.arrays.positions, Float32Array, target),
      readArray(this.manifest.arrays.valid, Uint8Array, target),
    ]);
    if (version !== this.requestVersion) return;
    // Spark 2.2 ExtSplats stores xyz as float32 bits in words 0..2 and
    // half-float opacity in word 3. Keep array slot == track ID; update only
    // this texture instead of repacking fixed color/scale/rotation each frame.
    for (let i = 0; i < this.manifest.trackCount; i++) {
      const motion = this.sharedMotion ? i % this.manifest.motionCount : i;
      const xyz = motion * 3, rgba = i * 4, reference = i * 14;
      const visible = valid[motion] && this.referenceValid[i];
      for (let axis = 0; axis < 3; axis++) {
        this.positionFloats[rgba + axis] = visible
          ? positions[xyz + axis] + (this.sharedMotion ? this.reference[reference + axis] : 0) : 0;
      }
      this.positionWords[rgba + 3] = visible ? this.opacityWords[i] : 0;
    }
    this.currentMesh.extSplats.textures[0].needsUpdate = true;
    this.currentMesh.needsUpdate = true;
    this.current = target;
    this.onFrame(target, this.frames.length);
  }

  async loadAnimation(url, version) {
    if (['piecewise_chebyshev', 'lifetime_chebyshev'].includes(this.manifest.positionEncoding) && typeof Worker === 'function' && this.renderer) {
      return this.loadPreparedAnimation(url, version);
    }
    const controller = new AbortController();
    this.animationAbort = controller;
    let mesh = null;
    let splats = null;
    try {
      const header = await readAnimationHeader(url, controller.signal);
      const parameters = await readAnimationFrame(url, header, 0, controller.signal);
      if (version !== this.requestVersion) return;
      splats = new PackedSplats({maxSplats: header.trackCount, lod: false});
      applyAnimationFrame(splats, parameters);
      mesh = new SplatMesh({packedSplats: splats, lod: false});
      await mesh.initialized;
      if (version !== this.requestVersion) return;
      this.animationUrl = url;
      this.animationHeader = header;
      this.manifest = {...header, file: {url}};
      this.frames = header.frameIndices;
      this.fps = header.fps;
      this.times = Float64Array.from(header.timesSeconds);
      this.root.add(mesh);
      this.currentMesh = mesh;
      mesh = splats = null;
      this.current = 0;
      this.onFrame(0, header.frameCount);
    } catch (error) {
      if (version === this.requestVersion && error.name !== 'AbortError') throw error;
    } finally {
      if (this.animationAbort === controller) this.animationAbort = null;
      if (mesh) mesh.dispose();
      else splats?.dispose();
    }
  }

  async loadPreparedAnimation(url, version) {
    this.preparing = true;
    // Explicitly finish each frame's depth sort before drawing its attributes.
    this.spark.autoUpdate = false;
    const preparation = new AnimationPreparation(url, this.renderer, {
      onProgress: progress => {
        if (this.preparation !== preparation) return;
        this.preparing = !progress.complete;
        this.onBuffering({...progress, preparing: this.preparing});
      },
      onError: error => {
        if (this.preparation === preparation) this.onError(error);
      }
    });
    this.preparation = preparation;
    try {
      await preparation.first;
      if (version !== this.requestVersion) return;
      const header = preparation.header;
      const mesh = new SplatMesh({packedSplats: preparation.splats, lod: false});
      await mesh.initialized;
      if (version !== this.requestVersion) { mesh.dispose(); return; }
      this.animationUrl = url;
      this.animationHeader = header;
      this.manifest = {...header, file: {url}};
      this.frames = header.frameIndices;
      this.fps = header.fps;
      this.times = Float64Array.from(header.timesSeconds);
      this.root.add(mesh);
      this.currentMesh = mesh;
      this.current = 0;
      this.onFrame(0, header.frameCount);
    } catch (error) {
      if (version === this.requestVersion && error.name !== 'AbortError') throw error;
    }
  }

  async showAnimationFrame(target, version) {
    if (this.preparation) {
      this.preparation.splats.select(target);
      this.currentMesh.updateMappingVersion();
      this.current = target;
      this.onFrame(target, this.frames.length);
      return;
    }
    this.animationAbort?.abort();
    const controller = new AbortController();
    this.animationAbort = controller;
    try {
      const parameters = await readAnimationFrame(this.animationUrl, this.animationHeader, target, controller.signal);
      if (version !== this.requestVersion) return;
      applyAnimationFrame(this.currentMesh.packedSplats, parameters);
      this.currentMesh.needsUpdate = true;
      this.current = target;
      this.onFrame(target, this.frames.length);
    } catch (error) {
      if (version === this.requestVersion && error.name !== 'AbortError') throw error;
    } finally {
      if (this.animationAbort === controller) this.animationAbort = null;
    }
  }

  async loadIndexedFrame(manifest, target, version) {
    if (version !== this.requestVersion) return;
    const controller = new AbortController();
    this.indexedAbort = controller;
    let mesh = null;
    try {
      const frame = manifest.sourceFrames[target];
      const [indices, original] = await Promise.all([
        readArray(manifest.arrays.sourceCellIndices, Int32Array, target, controller.signal),
        fetch(frame.fullUrl, {signal: controller.signal}).then(async response => {
          if (!response.ok) throw new Error(`原帧 PLY 读取失败：${response.status}`);
          return response.arrayBuffer();
        }),
      ]);
      if (version !== this.requestVersion) return;
      const bytes = remapGaussianPly(original, indices, manifest.layerCount);
      mesh = new SplatMesh({fileBytes: bytes, fileName: frame.full, lod: false});
      await mesh.initialized;
      if (version !== this.requestVersion) return;
      const previous = this.currentMesh;
      this.root.add(mesh);
      this.currentMesh = mesh;
      mesh = null; // Ownership transferred to the displayed frame.
      if (previous) {
        this.root.remove(previous);
        previous.dispose();
      }
      this.current = target;
      this.onFrame(target, this.frames.length);
    } catch (error) {
      if (version === this.requestVersion && error.name !== 'AbortError') throw error;
    } finally {
      controller.abort();
      if (this.indexedAbort === controller) this.indexedAbort = null;
      mesh?.dispose();
    }
  }

  async fillBuffer() { await this.preparation?.ready; }

  async toggle() {
    if (!this.playing) this.presentationSamples = [];
    await super.toggle();
    if (this.playing && this.preparation) this.schedulePreparedFrame();
  }

  schedulePreparedFrame() {
    clearTimeout(this.playbackTimer);
    if (!this.playing || !this.preparation) return;
    this.playbackTimer = setTimeout(async () => {
      if (!this.playing || !this.preparation) return;
      const now = performance.now();
      if (now >= this.nextFrameAt) super.tick(now);
      await this.rendering;
      this.schedulePreparedFrame();
    }, Math.max(0, this.nextFrameAt - performance.now()));
  }

  tick(now) {
    // Some embedded browsers throttle rAF to 1 Hz while the page is visible.
    // Prepared animation uses a deadline timer; rAF still drives paused views.
    if (this.playing && this.preparation) return;
    super.tick(now);
  }

  render(now, timings = {}) {
    if (!this.preparation || !this.currentMesh) return super.render(now, timings);
    if (this.rendering) return;
    const preparation = this.preparation;
    this.scene.updateMatrixWorld();
    this.camera.updateMatrixWorld();
    this.rendering = this.spark.update({scene: this.scene, camera: this.camera})
      .then(() => {
        if (this.preparation === preparation) super.render(performance.now(), timings);
      })
      .catch(error => { this.stop(); this.onError(error); })
      .finally(() => { this.rendering = null; });
  }

  didRenderFrame(now, timings = {}) {
    if (!this.playing || !this.preparation || !this.currentMesh) return;
    // Count a new animation state only once it is in Spark's displayed mapping.
    const mapping = this.spark.display.mapping.find(item => item.node === this.currentMesh);
    if (mapping?.version !== this.currentMesh.version) return;
    const samples = this.presentationSamples || (this.presentationSamples = []);
    if (samples.at(-1)?.frame === this.current) return;
    samples.push({frame: this.current, time: now, ...timings});
    if (samples.length > this.frames.length) samples.shift();
    if (samples.length > 1) {
      const milliseconds = (now - samples[0].time) / (samples.length - 1);
      this.onPerformance({fps: 1000 / milliseconds, milliseconds, samples});
    }
  }

  stop() {
    super.stop();
    clearTimeout(this.playbackTimer);
    this.requestVersion = (this.requestVersion || 0) + 1;
    this.indexedAbort?.abort();
    this.animationAbort?.abort();
  }

  clearMeshes() {
    this.requestVersion = (this.requestVersion || 0) + 1;
    this.indexedAbort?.abort();
    this.animationAbort?.abort();
    this.animationHeader = this.animationUrl = null;
    this.preparation?.dispose();
    this.preparation = null;
    if (this.spark) this.spark.autoUpdate = true;
    this.preparing = false;
    this.presentationSamples = [];
    this.reference = this.referenceValid = null;
    if (this.currentMesh) {
      this.root.remove(this.currentMesh);
      this.currentMesh.dispose();
      this.currentMesh = null;
    }
  }
}
