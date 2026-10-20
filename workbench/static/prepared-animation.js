import {PackedSplats} from '@sparkjsdev/spark';

// Stable source/uniforms: a frame change only selects a pre-uploaded texture.
export class PreparedSplats extends PackedSplats {
  constructor(count) {
    super({maxSplats: count, lod: false});
    this.count = count;
    this.frames = [];
    this.active = null;
  }

  add(index, packed, renderer) {
    const frame = new PackedSplats({packedArray: packed, numSplats: this.count, lod: false});
    this.frames[index] = frame;
    const texture = frame.getTexture();
    renderer?.initTexture(texture);
  }

  select(index) {
    const frame = this.frames[index];
    if (!frame) throw new Error('该帧尚未准备完成');
    this.active = frame;
    this.packedArray = frame.packedArray;
    this.numSplats = frame.numSplats;
    this.maxSplats = frame.maxSplats;
    this.needsUpdate = false;
  }

  getTexture() { return this.active ? this.active.getTexture() : super.getTexture(); }

  dispose() {
    for (const frame of this.frames) frame?.dispose();
    this.frames = [];
    this.active = null;
    super.dispose();
  }
}

export class AnimationPreparation {
  constructor(url, renderer, {onProgress = () => {}, onError = () => {}} = {}) {
    this.worker = new Worker(new URL('./animation-worker.js', import.meta.url), {type: 'module'});
    this.first = new Promise((resolve, reject) => { this.firstResolve = resolve; this.firstReject = reject; });
    this.ready = new Promise((resolve, reject) => { this.readyResolve = resolve; this.readyReject = reject; });
    this.ready.catch(() => {}); // Errors also go through onError before Play is pressed.
    this.worker.onerror = event => this.fail(new Error(event.message), onError);
    this.worker.onmessage = ({data}) => {
      try {
        if (data.type === 'header') {
          this.header = data.header;
          this.splats = new PreparedSplats(data.header.trackCount);
          onProgress({ready: 0, total: data.header.frameCount});
        } else if (data.type === 'frame') {
          this.splats.add(data.index, data.packed, renderer);
          if (data.index === 0) { this.splats.select(0); this.firstResolve(this); }
          onProgress({ready: data.index + 1, total: this.header.frameCount});
          this.worker.postMessage({type: 'next'});
        } else if (data.type === 'complete') {
          this.complete = true;
          this.worker.terminate();
          this.readyResolve(this);
          onProgress({ready: this.header.frameCount, total: this.header.frameCount, complete: true});
        } else if (data.type === 'error') this.fail(new Error(data.message), onError);
      } catch (error) { this.fail(error, onError); }
    };
    this.worker.postMessage({type: 'load', url});
  }

  fail(error, onError) {
    this.worker.terminate();
    this.firstReject(error);
    this.readyReject(error);
    onError(error);
  }

  dispose() {
    this.worker.terminate();
    const error = new DOMException('Aborted', 'AbortError');
    this.firstReject(error);
    this.readyReject(error);
    this.splats?.dispose();
  }
}
