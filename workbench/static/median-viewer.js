import {SplatMesh} from '@sparkjsdev/spark';
import {TrackSequenceViewer} from './track-viewer.js';
import {AnimationPreparation} from './prepared-animation.js';

// One camera and timeline; both variants stay prepared for immediate A/B switching.
export class MedianComparisonViewer extends TrackSequenceViewer {
  async loadPair(original, filtered) {
    this.stop();
    this.clearMeshes();
    const version = this.pairVersion;
    this.variants = {};
    this.source = 'filtered';
    this.current = -1;
    this.standalone = true;
    this.preparing = true;
    this.camera.position.set(0, 0, 0);
    this.yaw = this.pitch = 0;
    this.camera.rotation.set(0, 0, 0);
    this.camera.fov = 70;
    this.camera.updateProjectionMatrix();
    this.spark.autoUpdate = false;
    const total = filtered.frameCount * 2;
    const progress = {original: 0, filtered: 0};
    const start = async (name, manifest) => {
      const state = {};
      this.variants[name] = state;
      state.preparation = new AnimationPreparation(manifest.file.url, this.renderer, {
        onProgress: p => {
          if (version !== this.pairVersion) return;
          progress[name] = p.ready;
          this.preparing = !Object.values(this.variants).every(s => s.preparation?.complete)
            || Object.keys(this.variants).length < 2;
          this.onBuffering({ready: progress.original + progress.filtered, total, preparing: this.preparing});
        },
        onError: error => { if (version === this.pairVersion) this.onError(error); }
      });
      await state.preparation.first;
      if (version !== this.pairVersion) return;
      const mesh = new SplatMesh({packedSplats: state.preparation.splats, lod: false});
      await mesh.initialized;
      if (version !== this.pairVersion) { mesh.dispose(); return; }
      state.mesh = mesh;
      state.manifest = {...state.preparation.header, file: manifest.file};
      mesh.visible = name === this.source;
      this.root.add(mesh);
      if (name === this.source) {
        this.preparation = state.preparation;
        this.currentMesh = mesh;
        this.manifest = state.manifest;
        this.animationHeader = state.preparation.header;
        this.frames = this.manifest.frameIndices;
        this.fps = this.manifest.fps;
        this.times = Float64Array.from(this.manifest.timesSeconds);
        this.current = 0;
        this.onFrame(0, this.frames.length);
      }
    };
    await Promise.all([start('filtered', filtered), start('original', original)]);
  }

  selectSource(name) {
    const state = this.variants?.[name];
    if (!state?.mesh || !state.preparation.complete || this.preparing) return;
    this.stop();
    this.source = name;
    for (const [key, item] of Object.entries(this.variants)) item.mesh.visible = key === name;
    this.preparation = state.preparation;
    this.currentMesh = state.mesh;
    this.manifest = state.manifest;
    this.animationHeader = state.preparation.header;
    this.preparation.splats.select(this.current);
    this.currentMesh.updateMappingVersion();
    this.presentationSamples = [];
    this.onFrame(this.current, this.frames.length);
  }

  clearMeshes() {
    this.pairVersion = (this.pairVersion || 0) + 1;
    for (const state of Object.values(this.variants || {})) {
      if (state.preparation !== this.preparation) state.preparation?.dispose();
      if (state.mesh && state.mesh !== this.currentMesh) {
        this.root.remove(state.mesh);
        state.mesh.dispose();
      }
    }
    this.variants = {};
    super.clearMeshes();
  }
}
