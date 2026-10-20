import {MedianComparisonViewer} from './median-viewer.js';

export class MedianStep {
  constructor({showStage, showError, setMessage}) {
    this.showError = showError;
    this.setMessage = setMessage;
    for (const name of ['step','next','panel','result','metrics','note','run','processing','progress','percent','viewer','play','timeline','frame','performance','original','filtered','switch']) {
      const id = name === 'step' ? 'step-median' : name === 'next' ? 'next-median' : `median-${name}`;
      this[name] = document.getElementById(id);
    }
    this.token = 0;
    this.step.addEventListener('click', () => showStage(6));
    this.next.addEventListener('click', () => showStage(6));
    this.run.addEventListener('click', () => void this.start());
    this.play.addEventListener('click', () => void this.player?.toggle().catch(e => showError(e.message)));
    this.timeline.addEventListener('input', () => {
      this.player?.stop();
      void this.player?.show(Number(this.timeline.value)).catch(e => showError(e.message));
    });
    for (const name of ['original', 'filtered']) this[name].addEventListener('click', () => {
      this.player?.selectSource(name);
      this.original.classList.toggle('active', name === 'original');
      this.filtered.classList.toggle('active', name === 'filtered');
      this.performance.textContent = '';
    });
  }

  setVideo(video) {
    ++this.token;
    this.video = video;
    this.loadedKey = null;
    this.player?.stop();
    this.player?.clearMeshes();
    this.updateAvailability();
    if (video.median?.status === 'running') void this.poll(video, this.token);
  }

  updateAvailability() {
    this.step.disabled = this.video?.tracks?.status !== 'complete' || this.video?.tracks?.format !== 'sharp.gaussian_tracks.v5';
    this.step.classList.toggle('muted', this.step.disabled);
    this.next.hidden = this.step.disabled;
    this.step.classList.toggle('done', this.video?.median?.status === 'complete');
    this.step.classList.toggle('failed', this.video?.median?.status === 'failed');
  }

  hide() {
    this.active = false;
    this.step.classList.remove('active');
    this.panel.hidden = this.viewer.hidden = this.processing.hidden = true;
    this.player?.stop();
  }

  show() {
    this.active = true;
    this.step.classList.add('active');
    this.render();
  }

  render() {
    this.updateAvailability();
    if (!this.active) return;
    const state = this.video.median;
    const status = state?.status || 'idle';
    this.panel.hidden = false;
    this.panel.className = `analysis-panel ${status}`;
    this.processing.hidden = status !== 'running';
    this.run.hidden = status === 'complete' || status === 'running';
    this.run.disabled = status === 'running';
    this.run.textContent = status === 'failed' ? '重新进行中值滤波' : '开始中值滤波';
    this.switch.hidden = status !== 'complete';
    this.result.textContent = status === 'complete' ? '全属性中值滤波完成' : status === 'running' ? '正在处理完整序列' : status === 'failed' ? '滤波失败' : '尚未滤波';
    this.note.textContent = status === 'failed' ? state.error : '5 帧时间窗口，分别处理每条轨迹及两层高斯。位置、颜色、尺度、透明度取中值；旋转按角距离取中位样本。';
    if (status !== 'failed' && (state?.filter?.minTrackFrames === 5 || status === 'idle')) {
      this.note.textContent += ' 总长度少于 5 帧的轨迹，全部属性保留原值，不做滤波。';
    }
    this.metrics.replaceChildren();
    if (status === 'complete') {
      const metrics = [['窗口', '5 帧'], ['范围', '全部高斯属性'], ['耗时', `${state.elapsedSeconds.toFixed(1)} s`], ['文件大小', `${(state.file.bytes/1024**3).toFixed(2)} GiB`]];
      for (const [label, value] of metrics) {
        const node = document.createElement('div'); node.className = 'metric';
        const span = document.createElement('span'); span.textContent = label;
        const strong = document.createElement('strong'); strong.textContent = value;
        node.append(span, strong); this.metrics.append(node);
      }
      void this.load();
    } else {
      this.viewer.hidden = true;
      this.progress.value = 100 * (state?.currentFrame || 0) / (state?.frameCount || 1);
      this.percent.textContent = `${state?.currentFrame || 0} / ${state?.frameCount || this.video.tracks.frameCount} 帧`;
      this.setMessage(status === 'running' ? '正在沿每条轨迹进行全属性中值滤波…' : '对第五步的完整参数序列进行中值滤波，完成后可切换查看前后效果。');
    }
  }

  async load() {
    const video = this.video;
    const key = `${video.name}:${video.median.generatedAt}`;
    if (!this.player) this.player = new MedianComparisonViewer(this.viewer, {
      onFrame: (index, count) => {
        this.timeline.max = String(count - 1); this.timeline.value = String(index);
        this.frame.textContent = `${index+1} / ${count} · ${(index/this.video.median.fps).toFixed(2)} s`;
      },
      onPlayback: playing => {
        this.play.textContent = playing ? '❚❚' : '▶';
        this.play.setAttribute('aria-label', playing ? '暂停滤波对照' : '播放滤波对照');
      },
      onBuffering: p => {
        this.play.disabled = this.timeline.disabled = this.original.disabled = this.filtered.disabled = p.preparing;
        this.performance.textContent = p.preparing ? `准备 ${p.ready}/${p.total}` : '已就绪';
        if (this.active) this.setMessage(p.preparing ? `正在准备原始和滤波后的序列：${p.ready} / ${p.total} 帧。`
          : '可播放或拖动时间轴；切换「原始序列 / 中值滤波后」会保持同一帧和同一视角。');
      },
      onPerformance: p => { this.performance.textContent = `${p.fps.toFixed(1)} FPS`; },
      onError: e => this.showError(e.message)
    });
    this.viewer.hidden = false;
    this.player.resize();
    if (this.loadedKey === key) return;
    this.loadedKey = key;
    this.play.disabled = this.timeline.disabled = this.original.disabled = this.filtered.disabled = true;
    this.original.classList.remove('active'); this.filtered.classList.add('active');
    try { await this.player.loadPair(video.tracks, video.median); }
    catch (e) { this.loadedKey = null; this.showError(e.message); }
  }

  async start() {
    const video = this.video;
    const token = ++this.token;
    this.run.disabled = true;
    this.showError(null);
    try {
      const response = await fetch('/api/median', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({filename: video.name})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '中值滤波启动失败');
      if (video !== this.video || token !== this.token) return;
      video.median = data.median; this.render();
      if (data.median?.status === 'running') await this.poll(video, token);
    } catch (e) { this.run.disabled = false; this.showError(e.message); }
  }

  async poll(video, token) {
    while (video === this.video && token === this.token) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      if (video !== this.video || token !== this.token) return;
      try {
        const response = await fetch(`/api/median?filename=${encodeURIComponent(video.name)}`, {cache:'no-store'});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '无法读取滤波进度');
        if (video !== this.video || token !== this.token) return;
        video.median = data.median; this.render();
        if (data.median?.status !== 'running') return;
      } catch (e) { this.showError(e.message); return; }
    }
  }
}
