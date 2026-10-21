import {CompressedSequenceViewer} from './compression/viewer.js';

const bytes = n => n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(1)} MB`;
const metric = (name, value) => `<div class="metric"><span>${name}</span><strong>${value}</strong></div>`;

export class CompressionStep {
  constructor({showStage, showError, isActive, setMessage}) {
    this.showError = showError;
    this.isActive = isActive;
    this.setMessage = setMessage;
    for (const name of ['step', 'next', 'panel', 'result', 'metrics', 'note', 'run', 'processing', 'progress', 'percent',
      'title', 'viewer', 'play', 'timeline', 'frame', 'performance', 'download']) {
      const id = name === 'step' ? 'step-compression' : name === 'next' ? 'next-compression' : `compression-${name}`;
      this[name] = document.getElementById(id);
    }
    this.token = 0;
    this.step.addEventListener('click', () => showStage(5));
    this.next.addEventListener('click', () => showStage(5));
    this.run.addEventListener('click', () => void this.start());
    this.play.addEventListener('click', () => this.player?.toggle());
    this.timeline.addEventListener('input', () => this.player?.show(Number(this.timeline.value)));
  }

  setVideo(video) {
    this.token++;
    this.player?.clear();
    this.loadedKey = null;
    this.video = video;
    this.updateAvailability();
    if (video.compression?.status === 'running') void this.poll(this.token);
  }

  updateAvailability() {
    this.step.disabled = this.video?.gaussians?.status !== 'complete';
    this.step.classList.toggle('muted', this.step.disabled);
    this.step.classList.toggle('done', this.video?.compression?.status === 'complete');
    this.next.hidden = this.step.disabled;
  }

  hide() { this.panel.hidden = true; this.processing.hidden = true; this.viewer.hidden = true; this.player?.hide(); }
  show() { this.panel.hidden = false; this.render(); }

  render() {
    this.updateAvailability();
    if (!this.isActive()) return;
    const state = this.video?.compression;
    const status = state?.status || 'idle';
    this.panel.className = `analysis-panel ${status}`;
    this.run.hidden = status === 'running' || status === 'complete';
    this.run.disabled = status === 'running';
    this.run.textContent = ['failed', 'interrupted'].includes(status) ? '重新压缩高斯' : '开始压缩高斯';
    this.download.hidden = status !== 'complete';
    this.viewer.hidden = status !== 'complete';
    if (status === 'running') {
      this.processing.hidden = false;
      const labels = {preparing: '正在准备压缩…', converting: '正在转换高斯属性…', encoding: '正在压缩打包…', verifying: '正在校验压缩文件…'};
      this.title.textContent = labels[state.stage] || '正在压缩高斯…';
      if (state.progress === null) this.progress.removeAttribute('value');
      else this.progress.value = state.progress || 0;
      this.percent.textContent = state.stage === 'converting'
        ? `${state.currentFrame} / ${state.frameCount} 帧 · ${Math.round(state.progress)}%`
        : state.stage === 'verifying' ? '逐高斯检查重建误差…' : '编码器正在处理完整序列…';
      this.result.textContent = this.title.textContent;
      this.setMessage('正在将完整 PLY 序列压缩为一个文件。');
      return;
    }
    if (status === 'complete') {
      this.result.textContent = '压缩完成';
      this.metrics.innerHTML = [metric('序列', `${state.frameCount} 帧 · ${state.fps} fps`),
        metric('原始 PLY', bytes(state.sourceBytes)), metric('压缩文件', bytes(state.outputBytes)),
        metric('压缩倍率', `${state.compressionRatio.toFixed(2)}×`)].join('');
      this.download.href = state.url;
      this.download.download = 'sequence.splat4d';
      this.note.textContent = '播放前准备整段高斯帧，完成后直接切换 GPU 中的帧。';
      this.player && (this.player.visible = true);
      const key = `${state.url}:${state.completedAt}`;
      if (this.loadedKey !== key) void this.prepare(state, key);
      else this.processing.hidden = !this.player || this.player.ready;
      return;
    }
    this.processing.hidden = true;
    this.play.disabled = this.timeline.disabled = true;
    this.metrics.innerHTML = '';
    this.result.textContent = status === 'failed' ? '压缩失败' : status === 'interrupted' ? '任务已中断' : '尚未压缩';
    this.note.textContent = state?.error || '将第四步的完整 PLY 序列压缩打包为一个 splat4d 文件。';
    this.setMessage('压缩完成后可在这里直接播放，并下载单文件。');
  }

  async prepare(state, key) {
    this.loadedKey = key;
    const token = this.token;
    this.processing.hidden = false;
    this.play.disabled = this.timeline.disabled = true;
    this.title.textContent = '正在准备流畅播放…';
    this.performance.textContent = '';
    if (!this.player) this.player = new CompressedSequenceViewer(this.viewer, {
      onFrame: (frame, total) => { this.frame.textContent = `${frame + 1} / ${total}`; this.timeline.value = frame; },
      onPlayback: playing => { this.play.textContent = playing ? '⏸' : '▶'; },
      onPerformance: fps => { this.performance.textContent = `${fps} 帧/秒`; },
      onProgress: p => {
        this.progress.value = 100 * p.ready / p.total;
        this.percent.textContent = `准备播放 ${p.ready} / ${p.total} 帧`;
        this.processing.hidden = !this.isActive() || Boolean(p.complete);
        this.play.disabled = this.timeline.disabled = !p.complete;
        if (p.complete) this.setMessage('已准备完整序列，可播放、拖动时间轴或移动相机。');
      },
      onError: error => { this.processing.hidden = true; this.showError(error.message); }
    });
    this.timeline.max = state.frameCount - 1;
    try {
      await this.player.load(state);
      if (token !== this.token) return;
      this.processing.hidden = true;
    } catch (error) {
      if (token === this.token && error.name !== 'AbortError') {
        this.loadedKey = null; this.processing.hidden = true; this.showError(`压缩文件读取失败：${error.message}`);
      }
    }
  }

  async poll(token) {
    while (token === this.token) {
      await new Promise(resolve => setTimeout(resolve, 700));
      if (token !== this.token) return;
      try {
        const response = await fetch(`/api/compression?filename=${encodeURIComponent(this.video.name)}`, {cache: 'no-store'});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || '无法读取压缩进度');
        if (token !== this.token) return;
        this.video.compression = result.compression;
        this.render();
        if (result.compression?.status !== 'running') return;
      } catch (error) { if (token === this.token) this.showError(error.message); return; }
    }
  }

  async start() {
    if (!this.video || this.video.compression?.status === 'running') return;
    this.showError(null);
    this.run.disabled = true;
    const token = ++this.token;
    try {
      const response = await fetch('/api/compression', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({filename: this.video.name})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '无法启动压缩');
      if (token !== this.token) return;
      this.video.compression = result.compression;
      this.render();
      if (result.compression?.status === 'running') void this.poll(token);
    } catch (error) { if (token === this.token) { this.run.disabled = false; this.showError(error.message); } }
  }
}
