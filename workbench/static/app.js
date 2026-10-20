import {PlySequenceViewer} from './ply-viewer.js';
import {TrackSequenceViewer} from './track-viewer.js';
import {MedianStep} from './median-step.js';

const fileInput = document.querySelector('#video-file');
const dropZone = document.querySelector('#drop-zone');
const emptyState = document.querySelector('#empty-state');
const preview = document.querySelector('#video-preview');
const uploading = document.querySelector('#uploading');
const analyzing = document.querySelector('#analyzing');
const depthProcessing = document.querySelector('#depth-processing');
const gaussianProcessing = document.querySelector('#gaussian-processing');
const gaussianProgress = document.querySelector('#gaussian-progress');
const gaussianPercent = document.querySelector('#gaussian-percent');
const progress = document.querySelector('#upload-progress');
const percent = document.querySelector('#upload-percent');
const fileMeta = document.querySelector('#file-meta');
const message = document.querySelector('#message');
const replaceAction = document.querySelector('#replace-action');
const errorBox = document.querySelector('#error');
const stageIndex = document.querySelector('#stage-index');
const stageTitle = document.querySelector('#stage-title');
const stepUpload = document.querySelector('#step-upload');
const stepMotion = document.querySelector('#step-motion');
const stepDepth = document.querySelector('#step-depth');
const stepGaussians = document.querySelector('#step-gaussians');
const analysisPanel = document.querySelector('#analysis-panel');
const analysisResult = document.querySelector('#analysis-result');
const motionMetrics = document.querySelector('#motion-metrics');
const analysisNote = document.querySelector('#analysis-note');
const viewSwitch = document.querySelector('#view-switch');
const viewOriginal = document.querySelector('#view-original');
const viewStabilized = document.querySelector('#view-stabilized');
const depthPanel = document.querySelector('#depth-panel');
const depthResult = document.querySelector('#depth-result');
const depthMetrics = document.querySelector('#depth-metrics');
const depthNote = document.querySelector('#depth-note');
const depthViewSwitch = document.querySelector('#depth-view-switch');
const viewDepthSource = document.querySelector('#view-depth-source');
const viewDepthMap = document.querySelector('#view-depth-map');
const runDepthButton = document.querySelector('#run-depth');
const downloadDepth = document.querySelector('#download-depth');
const gaussianPanel = document.querySelector('#gaussian-panel');
const gaussianResult = document.querySelector('#gaussian-result');
const gaussianMetrics = document.querySelector('#gaussian-metrics');
const gaussianNote = document.querySelector('#gaussian-note');
const runGaussiansButton = document.querySelector('#run-gaussians');
const splatViewerHost = document.querySelector('#splat-viewer');
const viewerPlay = document.querySelector('#viewer-play');
const viewerTimeline = document.querySelector('#viewer-timeline');
const viewerFrame = document.querySelector('#viewer-frame');
const stepTracks = document.querySelector('#step-tracks');
const nextTracksButton = document.querySelector('#next-tracks');
const trackPanel = document.querySelector('#track-panel');
const trackResult = document.querySelector('#track-result');
const trackMetrics = document.querySelector('#track-metrics');
const trackNote = document.querySelector('#track-note');
const runTracksButton = document.querySelector('#run-tracks');
const trackProcessing = document.querySelector('#track-processing');
const trackProgressTitle = document.querySelector('#track-progress-title');
const trackProgress = document.querySelector('#track-progress');
const trackPercent = document.querySelector('#track-percent');
const trackViewerHost = document.querySelector('#track-viewer');
const trackPlay = document.querySelector('#track-play');
const trackTimeline = document.querySelector('#track-timeline');
const trackFrame = document.querySelector('#track-frame');
const trackPerformance = document.querySelector('#track-performance');
let currentVideo = null;
let currentStage = 1;
let analysisRunning = false;
let depthRunning = false;
let gaussianRunning = false;
let gaussianPollToken = 0;
let plyViewer = null;
let loadedGaussianKey = null;
let trackRunning = false;
let trackPollToken = 0;
let trackViewer = null;
let loadedTrackKey = null;
let trackLoadPromise = null;
const medianStep = new MedianStep({showStage, showError, setMessage: text => { message.textContent = text; }});

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function showError(error) {
  errorBox.hidden = !error;
  errorBox.textContent = error || '';
}

function setPreview(url, activeButton = null) {
  splatViewerHost.hidden = true;
  trackViewerHost.hidden = true;
  plyViewer?.stop();
  trackViewer?.stop();
  preview.hidden = false;
  const time = preview.currentTime || 0;
  const paused = preview.paused;
  preview.src = url;
  preview.addEventListener('loadedmetadata', () => {
    preview.currentTime = Math.min(time, preview.duration || time);
    if (!paused) preview.play().catch(() => {});
  }, {once: true});
  for (const button of [viewOriginal, viewStabilized, viewDepthSource, viewDepthMap]) {
    button.classList.toggle('active', button === activeButton);
  }
}

function setPreviewIfDifferent(url) {
  const target = new URL(url, location.href);
  const current = preview.currentSrc ? new URL(preview.currentSrc) : null;
  if (!current || current.pathname !== target.pathname) setPreview(url);
}

function originalUrl() {
  if (!currentVideo) return '';
  return `${currentVideo.url}?v=${encodeURIComponent(currentVideo.modifiedAt)}`;
}

function showStage(stage) {
  if (stage === 2 && !currentVideo) return;
  if (stage === 3 && !currentVideo?.analysis?.allowed) return;
  if (stage === 4 && currentVideo?.depth?.status !== 'complete') return;
  if (stage === 5 && currentVideo?.gaussians?.status !== 'complete') return;
  if (stage === 6 && medianStep.step.disabled) return;
  medianStep.hide();
  trackViewer?.stop();
  trackViewerHost.hidden = true;
  currentStage = stage;
  history.replaceState(null, '', `${location.pathname}${location.search}${stage === 6 ? '#median' : stage === 5 ? '#tracks' : ''}`);
  trackPanel.hidden = true;
  trackProcessing.hidden = true;
  if (stage !== 5) {
    trackViewerHost.hidden = true;
    trackViewer?.stop();
  }
  if (stage !== 4) {
    splatViewerHost.hidden = true;
    preview.hidden = false;
    plyViewer?.stop();
  }
  stepUpload.classList.toggle('active', stage === 1);
  stepUpload.classList.toggle('done', stage > 1 && !!currentVideo);
  stepMotion.classList.toggle('active', stage === 2);
  stepDepth.classList.toggle('active', stage === 3);
  stepDepth.classList.toggle('done', stage > 3 && !!currentVideo?.depth);
  stepGaussians.classList.toggle('active', stage === 4);
  stepGaussians.classList.toggle('done', stage !== 4 && currentVideo?.gaussians?.status === 'complete');
  stepTracks.classList.toggle('active', stage === 5);
  stepTracks.classList.toggle('done', currentVideo?.tracks?.status === 'complete');

  if (stage === 1) {
    stageIndex.textContent = 'STEP 01';
    stageTitle.textContent = '上传视频';
    analysisPanel.hidden = true;
    depthPanel.hidden = true;
    gaussianPanel.hidden = true;
    analyzing.hidden = true;
    depthProcessing.hidden = true;
    gaussianProcessing.hidden = true;
    if (currentVideo) {
      setPreview(originalUrl());
      message.textContent = '视频已保存到本地，可以更换输入素材。';
    } else {
      message.textContent = '上传后会在这里直接预览视频，方便确认开发输入。';
    }
    return;
  }

  if (stage === 2) {
    stageIndex.textContent = 'STEP 02';
    stageTitle.textContent = '机位检测';
    depthPanel.hidden = true;
    gaussianPanel.hidden = true;
    depthProcessing.hidden = true;
    gaussianProcessing.hidden = true;
    analyzing.hidden = !analysisRunning;
    if (currentVideo.analysis) renderAnalysis(currentVideo.analysis);
    else {
      analysisPanel.hidden = true;
      message.textContent = '正在判断机位是否固定…';
    }
    return;
  }

  if (stage === 3) {
    stageIndex.textContent = 'STEP 03';
    stageTitle.textContent = '预测深度视频';
    gaussianPanel.hidden = true;
    gaussianProcessing.hidden = true;
    analysisPanel.hidden = true;
    analyzing.hidden = true;
    depthProcessing.hidden = !depthRunning;
    renderDepth(currentVideo.depth);
    return;
  }

  if (stage === 6) {
    stageIndex.textContent = 'STEP 06';
    stageTitle.textContent = '中值滤波';
    analysisPanel.hidden = depthPanel.hidden = gaussianPanel.hidden = true;
    analyzing.hidden = depthProcessing.hidden = gaussianProcessing.hidden = true;
    preview.pause();
    preview.hidden = true;
    medianStep.show();
    return;
  }

  if (stage === 5) {
    stageIndex.textContent = 'STEP 05';
    stageTitle.textContent = '高斯轨迹';
    analysisPanel.hidden = depthPanel.hidden = gaussianPanel.hidden = true;
    analyzing.hidden = depthProcessing.hidden = gaussianProcessing.hidden = true;
    renderTracks(currentVideo.tracks);
    return;
  }

  stageIndex.textContent = 'STEP 04';
  stageTitle.textContent = '逐帧高斯视频';
  analysisPanel.hidden = true;
  depthPanel.hidden = true;
  analyzing.hidden = true;
  depthProcessing.hidden = true;
  renderGaussians(currentVideo.gaussians);
}

function showVideo(video, stage = 2) {
  plyViewer?.stop();
  trackViewer?.stop();
  trackPollToken++;
  trackRunning = video.tracks?.status === 'running';
  loadedTrackKey = null;
  currentVideo = video;
  medianStep.setVideo(video);
  preview.src = originalUrl();
  preview.hidden = false;
  emptyState.hidden = true;
  replaceAction.hidden = false;
  fileMeta.textContent = `${video.name} · ${formatBytes(video.size)}`;
  stepMotion.disabled = false;
  stepMotion.classList.remove('muted', 'failed');
  stepMotion.classList.toggle('done', !!video.analysis?.allowed);
  stepDepth.disabled = !video.analysis?.allowed;
  stepDepth.classList.toggle('muted', !video.analysis?.allowed);
  stepDepth.classList.remove('failed', 'done');
  stepGaussians.disabled = video.depth?.status !== 'complete';
  stepGaussians.classList.toggle('muted', video.depth?.status !== 'complete');
  stepGaussians.classList.remove('failed', 'done');
  stepTracks.disabled = video.gaussians?.status !== 'complete';
  stepTracks.classList.toggle('muted', stepTracks.disabled);
  stepTracks.classList.toggle('failed', video.tracks?.status === 'failed');
  loadedGaussianKey = null;
  stepDepth.classList.toggle('done', video.depth?.status === 'complete');
  stepGaussians.classList.toggle('done', video.gaussians?.status === 'complete');
  showStage(stage === 6 && !medianStep.step.disabled ? 6 : stage === 5 && !stepTracks.disabled ? 5 : 2);

  if (!video.analysis) runAnalysis(video);
  if (video.gaussians?.status === 'running') {
    gaussianRunning = true;
    const token = ++gaussianPollToken;
    void pollGaussians(video, token);
  }
  if (trackRunning) void pollTracks(video, trackPollToken);
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderAnalysis(analysis) {
  analyzing.hidden = true;
  analysisPanel.hidden = currentStage !== 2;
  analysisPanel.className = `analysis-panel ${analysis.status}`;
  stepMotion.classList.remove('failed');
  stepDepth.disabled = !analysis.allowed;
  stepDepth.classList.toggle('muted', !analysis.allowed);
  const metrics = analysis.metrics || {};
  motionMetrics.innerHTML = metrics.reliable ? [
    metric('P95 平移', `${metrics.translationP95Px.toFixed(2)} px`),
    metric('最大平移', `${metrics.translationMaxPx.toFixed(2)} px`),
    metric('P95 旋转', `${metrics.rotationP95Deg.toFixed(3)}°`),
    metric('检测帧数', String(metrics.frameCount))
  ].join('') : metric('检测帧数', String(metrics.frameCount || 0));

  viewSwitch.hidden = true;
  if (analysis.status === 'fixed') {
    analysisResult.textContent = '固定机位 · 已通过';
    analysisNote.textContent = '相机运动低于固定机位阈值，后续流程可直接使用原视频。';
    if (currentStage === 2) message.textContent = '检测完成：视频为固定机位。';
    stepMotion.classList.add('done');
  } else if (analysis.status === 'stabilized') {
    analysisResult.textContent = '轻微抖动 · 防抖后通过';
    analysisNote.textContent = '检测到轻微抖动，已使用固定缩放生成防抖视频；后续流程应统一使用防抖版本。';
    if (currentStage === 2) message.textContent = '检测完成：已生成防抖版本。';
    stepMotion.classList.add('done');
    viewSwitch.hidden = false;
    if (currentStage === 2) setPreview(`${analysis.outputUrl}?v=${encodeURIComponent(analysis.analyzedAt)}`, viewStabilized);
  } else if (analysis.status === 'rejected') {
    analysisResult.textContent = '明显运镜 · 已拒绝';
    analysisNote.textContent = '相机运动超过轻微抖动范围，已停止后续流程。';
    if (currentStage === 2) message.textContent = '检测完成：该视频不能进入后续步骤。';
    stepMotion.classList.add('failed');
  } else {
    analysisResult.textContent = '无法可靠判断 · 已停止';
    analysisNote.textContent = '没有获得足够的有效运动测量，已停止后续流程。';
    if (currentStage === 2) message.textContent = '机位检测没有获得可靠结果。';
    stepMotion.classList.add('failed');
  }
}

function depthInputUrl() {
  const analysis = currentVideo?.analysis;
  if (analysis?.status === 'stabilized' && analysis.outputUrl) {
    return `${analysis.outputUrl}?v=${encodeURIComponent(analysis.analyzedAt)}`;
  }
  return originalUrl();
}

function renderDepth(depth) {
  depthPanel.hidden = currentStage !== 3;
  if (currentStage !== 3) return;

  runDepthButton.disabled = depthRunning;
  downloadDepth.hidden = true;
  depthViewSwitch.hidden = true;
  depthMetrics.innerHTML = '';
  depthPanel.className = 'analysis-panel';
  if (!depth) {
    depthResult.textContent = depthRunning ? '正在生成' : '尚未生成';
    depthNote.textContent = depthRunning
      ? '模型正在逐帧预测时序一致的公制深度。'
      : '使用 Metric Video Depth Anything Base 生成时序一致的深度。';
    runDepthButton.hidden = depthRunning;
    setPreview(depthInputUrl(), viewDepthSource);
    message.textContent = depthRunning ? '正在预测深度视频…' : '机位检测已通过，可以开始预测深度。';
    return;
  }

  depthPanel.classList.add('complete');
  depthResult.textContent = '深度视频已生成';
  depthMetrics.innerHTML = [
    metric('视频帧数', String(depth.frameCount)),
    metric('输出尺寸', `${depth.width} × ${depth.height}`),
    metric('深度中位数', `${depth.depthP50.toFixed(2)} m`),
    metric('深度范围', `${depth.depthMin.toFixed(2)} – ${depth.depthMax.toFixed(2)} m`)
  ].join('');
  depthNote.textContent = `${depth.model} · ${depth.device.toUpperCase()} · ${depth.fps.toFixed(2)} FPS`;
  depthViewSwitch.hidden = false;
  runDepthButton.hidden = true;
  downloadDepth.hidden = false;
  downloadDepth.href = depth.dataUrl;
  downloadDepth.download = 'depths.npz';
  setPreview(`${depth.outputUrl}?v=${encodeURIComponent(depth.createdAt)}`, viewDepthMap);
  message.textContent = '深度预测完成，可以预览深度视频或下载逐帧浮点深度。';
  stepDepth.classList.add('done');
  stepGaussians.disabled = false;
  stepGaussians.classList.remove('muted');
}

function ensurePlyViewer() {
  if (plyViewer) return plyViewer;
  plyViewer = new PlySequenceViewer(splatViewerHost, {
    onFrame: (index, count) => {
      viewerTimeline.max = String(Math.max(0, count - 1));
      viewerTimeline.value = String(index);
      viewerFrame.textContent = `${index + 1} / ${count}`;
    },
    onPlayback: playing => { viewerPlay.textContent = playing ? '❚❚' : '▶'; },
    onBuffering: buffering => {
      viewerPlay.disabled = buffering;
      if (buffering) viewerPlay.textContent = '…';
      else viewerPlay.textContent = plyViewer?.playing ? '❚❚' : '▶';
    },
    onError: error => showError(`PLY 播放失败：${error.message}`)
  });
  return plyViewer;
}

async function loadGaussianViewer(gaussians) {
  const video = currentVideo;
  const key = `${gaussians.completedAt}:${gaussians.frames.length}`;
  if (loadedGaussianKey === key) {
    splatViewerHost.hidden = false;
    preview.hidden = true;
    return;
  }
  loadedGaussianKey = key;
  try {
    const viewer = ensurePlyViewer();
    await viewer.load(gaussians);
    if (currentStage !== 4 || currentVideo !== video) return;
    splatViewerHost.hidden = false;
    preview.hidden = true;
    splatViewerHost.focus();
  } catch (error) {
    loadedGaussianKey = null;
    showError(`PLY 播放器载入失败：${error.message}`);
  }
}

function renderGaussians(gaussians) {
  stepTracks.disabled = gaussians?.status !== 'complete';
  stepTracks.classList.toggle('muted', stepTracks.disabled);
  nextTracksButton.hidden = stepTracks.disabled;
  gaussianPanel.hidden = currentStage !== 4;
  if (currentStage !== 4) return;

  const status = gaussians?.status || 'idle';
  const current = Number(gaussians?.currentFrame || 0);
  const total = Number(gaussians?.totalFrames || currentVideo?.depth?.frameCount || 0);
  const ratio = total ? Math.min(1, current / total) : 0;
  gaussianProgress.value = Math.round(ratio * 100);
  gaussianPercent.textContent = total
    ? `${current} / ${total} 帧 · ${Math.round(ratio * 100)}%`
    : '准备 SHARP 模型…';
  gaussianProcessing.hidden = status !== 'running';
  runGaussiansButton.hidden = status === 'running' || status === 'complete';
  runGaussiansButton.disabled = status === 'running';
  gaussianPanel.className = 'analysis-panel';
  gaussianMetrics.innerHTML = '';
  splatViewerHost.hidden = true;
  preview.hidden = false;

  if (status === 'running') {
    gaussianResult.textContent = `正在生成 ${current} / ${total}`;
    gaussianNote.textContent = '每帧使用对应 RGB 与公制深度调用 Apple SHARP，并写入未经抽样的完整 PLY。';
    message.textContent = 'SHARP 正在逐帧生成 PLY 序列…';
    if (currentVideo.depth?.outputUrl) setPreviewIfDifferent(`${currentVideo.depth.outputUrl}?v=${encodeURIComponent(currentVideo.depth.createdAt)}`);
    return;
  }

  if (status === 'complete') {
    gaussianPanel.classList.add('complete');
    gaussianResult.textContent = 'PLY 序列已生成';
    gaussianMetrics.innerHTML = [
      metric('序列帧数', String(gaussians.totalFrames)),
      metric('每帧高斯', Number(gaussians.frames[0]?.gaussianCount || 0).toLocaleString()),
      metric('生成耗时', `${Number(gaussians.elapsedSeconds || 0).toFixed(1)} s`)
    ].join('');
    gaussianNote.textContent = '网页播放器直接加载每一帧未经抽样的完整 PLY。';
    message.textContent = '点击播放器后可使用 WASD 移动、Q/E 升降，并用鼠标拖动视角。';
    stepGaussians.classList.add('done');
    void loadGaussianViewer(gaussians);
    return;
  }

  if (status === 'failed') {
    gaussianResult.textContent = '生成失败';
    gaussianNote.textContent = gaussians.error || 'SHARP 生成失败，可以重试。';
    stepGaussians.classList.add('failed');
    if (gaussians.error) showError(gaussians.error);
  } else if (status === 'incomplete') {
    gaussianResult.textContent = `序列不完整 · ${current} / ${gaussians.expectedFrames || total}`;
    gaussianNote.textContent = '检测到测试或中断结果，点击按钮会重新生成完整序列。';
  } else {
    gaussianResult.textContent = '尚未生成';
    gaussianNote.textContent = '每帧使用对应 RGB 与公制深度调用 Apple SHARP。';
  }
  if (currentVideo.depth?.outputUrl) setPreviewIfDifferent(`${currentVideo.depth.outputUrl}?v=${encodeURIComponent(currentVideo.depth.createdAt)}`);
  message.textContent = '深度结果已准备好，可以开始生成逐帧 PLY。';
}

function updateTrackMetrics(tracks, index = 0) {
  if (tracks.format === 'sharp.gaussian_tracks.v5') {
    trackMetrics.innerHTML = [
      metric('两层高斯 / 当前帧', Number(tracks.trackCount).toLocaleString()),
      metric('原帧覆盖', '100%'),
      metric('本帧新轨迹', Number(tracks.birthCounts[index]).toLocaleString()),
      metric('整段轨迹 ID', Number(tracks.totalTrackCount).toLocaleString())
    ].join('');
    return;
  }
  trackMetrics.innerHTML = [
    metric(tracks.layerCount === 2 ? '两层高斯 / 固定 ID' : '固定 ID', Number(tracks.trackCount).toLocaleString()),
    metric('轨迹帧数', String(tracks.frameCount)),
    metric(['sharp.gaussian_tracks.v3', 'sharp.gaussian_tracks.v4'].includes(tracks.format) ? '有效对应' : '当前有效', Number(tracks.validCounts[index]).toLocaleString()),
    metric('时间轴', `${tracks.fps} FPS`)
  ].join('');
}

function ensureTrackViewer() {
  if (trackViewer) return trackViewer;
  trackViewer = new TrackSequenceViewer(trackViewerHost, {
    onFrame(index, count) {
      trackTimeline.max = String(count - 1);
      trackTimeline.value = String(index);
      trackFrame.textContent = `${index + 1} / ${count} · ${trackViewer.times[index].toFixed(2)} s`;
      if (currentStage === 5) updateTrackMetrics(trackViewer.manifest, index);
    },
    onPlayback(playing) {
      trackPlay.textContent = playing ? '❚❚' : '▶';
      trackPlay.setAttribute('aria-label', playing ? '暂停高斯轨迹' : '播放高斯轨迹');
    },
    onBuffering(progress) {
      if (currentStage !== 5 || typeof progress !== 'object') return;
      trackPlay.disabled = trackTimeline.disabled = progress.preparing;
      trackPerformance.textContent = progress.preparing ? `准备 ${progress.ready}/${progress.total}` : '已就绪 · 目标 25 FPS';
      message.textContent = progress.preparing
        ? `正在后台准备流畅播放：${progress.ready} / ${progress.total} 帧。首帧可移动查看，准备完成后即可播放。`
        : '全部帧已准备好。播放时直接切换 GPU 数据，右下角显示实测动画帧率。';
    },
    onPerformance(stats) {
      if (currentStage !== 5) return;
      trackPerformance.textContent = `${stats.fps.toFixed(1)} FPS · ${stats.milliseconds.toFixed(1)} ms`;
      trackPerformance.dataset.samples = JSON.stringify(stats.samples);
    },
    onError: error => showError(`轨迹播放失败：${error.message}`)
  });
  return trackViewer;
}

async function loadTrackViewer(tracks) {
  const video = currentVideo;
  const stage = 5;
  const key = `${video.name}:${tracks.format}:${tracks.generatedAt || tracks.elapsedSeconds}:${tracks.frameCount}`;
  trackPlay.disabled = trackTimeline.disabled = true;
  try {
    if (loadedTrackKey !== key) {
      trackPerformance.textContent = '';
      delete trackPerformance.dataset.samples;
      loadedTrackKey = key;
      trackLoadPromise = ensureTrackViewer().load(tracks);
    }
    await trackLoadPromise;
    if (currentStage !== stage || currentVideo !== video || loadedTrackKey !== key) return;
    if (trackViewer.current < 0) {
      // Leaving this step can cancel initial file loading before its header is
      // available. Re-enter by loading the file again, not by querying no header.
      if (trackViewer.standalone && !trackViewer.animationHeader) await trackViewer.load(tracks);
      else await trackViewer.show(0);
    }
    if (currentStage !== stage || currentVideo !== video || loadedTrackKey !== key) return;
    preview.pause();
    preview.hidden = true;
    trackViewerHost.hidden = false;
    trackPlay.disabled = trackTimeline.disabled = !!trackViewer.preparing;
    updateTrackMetrics(tracks, trackViewer.current);
  } catch (error) {
    if (loadedTrackKey === key) loadedTrackKey = null;
    if (currentStage === stage && currentVideo === video) showError(`轨迹播放器载入失败：${error.message}`);
  }
}

function renderTracks(tracks) {
  medianStep.updateAvailability();
  const status = tracks?.status || 'idle';
  const indexed = tracks?.format === 'sharp.gaussian_tracks.v3';
  const lifecycle = tracks?.format === 'sharp.gaussian_tracks.v5';
  const standalone = lifecycle || tracks?.format === 'sharp.gaussian_tracks.v4';
  stepTracks.classList.toggle('done', status === 'complete');
  stepTracks.classList.toggle('failed', status === 'failed');
  trackPanel.hidden = currentStage !== 5;
  if (currentStage !== 5) return;

  trackPanel.className = `analysis-panel ${status}`;
  trackProcessing.hidden = status !== 'running';
  runTracksButton.hidden = status === 'running' || (status === 'complete' && lifecycle);
  runTracksButton.disabled = status === 'running';
  runTracksButton.textContent = status === 'complete' && !lifecycle ? '重建完整覆盖轨迹'
    : status === 'failed' ? '重新生成高斯轨迹' : '开始生成高斯轨迹';
  trackMetrics.innerHTML = '';

  if (status === 'complete') {
    trackResult.textContent = lifecycle ? '完整覆盖 · 轨迹新增与结束' : standalone ? '独立动画文件 · 全属性更新' : indexed ? '光流对应 · 全属性更新'
      : tracks.layerCount === 2 ? '两层高斯轨迹已生成 · 共享位移' : '高斯轨迹已生成';
    updateTrackMetrics(tracks);
    trackNote.textContent = lifecycle
      ? '每帧原始高斯各显示一次：重复对应保留一条，遗漏建立新轨迹。全部属性保存在单个 .4dgs 文件中。轨迹 ID 可在后续帧新增，不保证跨遮挡身份连续。'
      : standalone
      ? '直接读取单个 .4dgs 文件：首帧高斯与逐帧完整参数均已保存，播放不依赖原始 PLY。时间轴为原始帧率，实际速度取决于读取与更新耗时。'
      : indexed
      ? '两层共用像素轨迹，分别读取对应原帧高斯的位置、颜色、尺度、旋转和透明度。允许重复对应，出界后停止；有效数不表示对应一定正确。'
      : tracks.positionEncoding === 'reference_plus_shared_displacement'
      ? '两层同网格高斯共用一份 RGB-D 位移，各自保留第一帧的位置偏移、颜色、尺度、旋转和透明度。未按误差筛选；有效数只表示位置可计算。'
      : '同一组高斯按固定 ID 更新位置，颜色、尺度和旋转沿用第一帧。未按光流或深度误差筛选；有效数只表示位置可计算。';
    message.textContent = '轨迹已接入当前视频流程，可直接播放、拖动时间轴或切换前面的步骤。';
    void loadTrackViewer(tracks);
    return;
  }

  trackViewerHost.hidden = true;
  trackViewer?.stop();
  preview.hidden = false;
  if (currentVideo.depth?.sourceUrl) setPreviewIfDifferent(currentVideo.depth.sourceUrl);
  if (status === 'running') {
    const flow = tracks.stage === 'flow';
    const exporting = tracks.stage === 'export';
    const current = Number(flow ? tracks.currentPair || 0 : tracks.currentFrame || 0);
    const total = Number(flow ? tracks.totalPairs || currentVideo.depth.frameCount - 1 : tracks.frameCount || currentVideo.depth.frameCount);
    trackProgress.value = total ? 100 * current / total : 0;
    trackProgressTitle.textContent = flow ? '正在计算相邻帧光流…' : exporting ? '正在写入独立高斯动画文件…' : '正在连接轨迹并补齐遗漏…';
    trackPercent.textContent = `${current} / ${total} ${flow ? '对' : '帧'}`;
    trackResult.textContent = flow ? '生成中 · 光流与候选对应' : exporting ? '生成中 · 全属性与轨迹起止时间' : '生成中 · 每帧完整覆盖';
    trackNote.textContent = exporting ? '将两层高斯的全部属性、轨迹 ID 和起止时间写入一个文件。' : '沿光流连接已有轨迹，重复对应保留一条，未覆盖网格建立新轨迹。';
    message.textContent = trackProgressTitle.textContent;
  } else {
    trackResult.textContent = status === 'failed' ? '轨迹生成失败' : '尚未生成';
    trackNote.textContent = status === 'failed' ? tracks.error || '生成失败，可以重试。' : '复用已有光流，连接轨迹、去除重复引用并补齐遗漏，将每帧全部高斯保存为一个独立文件。';
    message.textContent = '逐帧高斯已准备好，可以继续生成轨迹并重放。';
  }
}

async function pollTracks(video, token) {
  while (token === trackPollToken && currentVideo === video) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    if (token !== trackPollToken || currentVideo !== video) return;
    try {
      const response = await fetch(`/api/tracks?filename=${encodeURIComponent(video.name)}`, {cache: 'no-store'});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '无法读取轨迹生成进度');
      if (token !== trackPollToken || currentVideo !== video) return;
      video.tracks = result.tracks;
      trackRunning = result.tracks?.status === 'running';
      renderTracks(result.tracks);
      if (!trackRunning) return;
    } catch (error) {
      trackRunning = false;
      trackProcessing.hidden = true;
      showError(error.message);
      return;
    }
  }
}

async function runTracks() {
  if (currentVideo?.gaussians?.status !== 'complete' || trackRunning) return;
  const video = currentVideo;
  const token = ++trackPollToken;
  showError(null);
  trackRunning = true;
  runTracksButton.disabled = true;
  try {
    const response = await fetch('/api/tracks', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: video.name})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '轨迹生成启动失败');
    if (token !== trackPollToken || currentVideo !== video) return;
    video.tracks = result.tracks;
    trackRunning = result.tracks?.status === 'running';
    renderTracks(result.tracks);
    if (trackRunning) await pollTracks(video, token);
  } catch (error) {
    if (token !== trackPollToken || currentVideo !== video) return;
    trackRunning = false;
    runTracksButton.disabled = false;
    trackProcessing.hidden = true;
    showError(error.message);
  }
}

async function runAnalysis(video) {
  showError(null);
  analysisRunning = true;
  analyzing.hidden = currentStage !== 2;
  analysisPanel.hidden = true;
  message.textContent = '正在判断机位是否固定…';
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: video.name})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '机位检测失败');
    video.analysis = result.analysis;
    analysisRunning = false;
    renderAnalysis(result.analysis);
  } catch (error) {
    analysisRunning = false;
    analyzing.hidden = true;
    showError(error.message);
    message.textContent = '机位检测失败。';
  }
}

async function runDepth() {
  if (!currentVideo?.analysis?.allowed || depthRunning) return;
  showError(null);
  depthRunning = true;
  depthProcessing.hidden = currentStage !== 3;
  renderDepth(null);
  try {
    const response = await fetch('/api/depth', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: currentVideo.name})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '深度预测失败');
    currentVideo.depth = result.depth;
    depthRunning = false;
    depthProcessing.hidden = true;
    renderDepth(result.depth);
  } catch (error) {
    depthRunning = false;
    depthProcessing.hidden = true;
    runDepthButton.hidden = false;
    runDepthButton.disabled = false;
    depthResult.textContent = '生成失败';
    message.textContent = '深度预测失败。';
    showError(error.message);
  }
}

async function fetchGaussianStatus(video) {
  const response = await fetch(`/api/gaussians?filename=${encodeURIComponent(video.name)}`, {
    cache: 'no-store'
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '无法读取 PLY 生成进度');
  video.gaussians = result.gaussians;
  return result.gaussians;
}

async function pollGaussians(video, token) {
  while (token === gaussianPollToken && currentVideo === video) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    if (token !== gaussianPollToken || currentVideo !== video) return;
    try {
      const state = await fetchGaussianStatus(video);
      gaussianRunning = state?.status === 'running';
      renderGaussians(state);
      if (!gaussianRunning) return;
    } catch (error) {
      gaussianRunning = false;
      gaussianProcessing.hidden = true;
      showError(error.message);
      return;
    }
  }
}

async function runGaussians() {
  if (!currentVideo?.depth || gaussianRunning) return;
  showError(null);
  gaussianRunning = true;
  const video = currentVideo;
  const token = ++gaussianPollToken;
  try {
    const response = await fetch('/api/gaussians', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: video.name})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'PLY 序列生成启动失败');
    video.gaussians = result.gaussians;
    gaussianRunning = result.gaussians?.status === 'running';
    renderGaussians(result.gaussians);
    if (gaussianRunning) await pollGaussians(video, token);
  } catch (error) {
    gaussianRunning = false;
    gaussianProcessing.hidden = true;
    runGaussiansButton.hidden = false;
    showError(error.message);
  }
}

async function restoreLatest() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error('无法读取工作台状态');
    const state = await response.json();
    if (state.videos?.length) showVideo(state.videos[0], location.hash === '#median' ? 6 : location.hash === '#tracks' ? 5 : 2);
  } catch (error) {
    showError(error.message);
  }
}

function upload(file) {
  if (!file) return;
  if (!file.type.startsWith('video/') && !/\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name)) {
    showError('请选择视频文件');
    return;
  }

  showError(null);
  analysisPanel.hidden = true;
  analyzing.hidden = true;
  uploading.hidden = false;
  progress.value = 0;
  percent.textContent = '0%';

  const request = new XMLHttpRequest();
  request.open('POST', '/api/upload');
  request.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
  request.setRequestHeader('X-File-Name', encodeURIComponent(file.name));

  request.upload.addEventListener('progress', event => {
    if (!event.lengthComputable) return;
    const value = Math.round(event.loaded / event.total * 100);
    progress.value = value;
    percent.textContent = `${value}%`;
  });

  request.addEventListener('load', () => {
    uploading.hidden = true;
    let result;
    try { result = JSON.parse(request.responseText); } catch { result = {}; }
    if (request.status < 200 || request.status >= 300) {
      showError(result.error || '上传失败');
      return;
    }
    showVideo(result.video);
  });

  request.addEventListener('error', () => {
    uploading.hidden = true;
    showError('上传失败：无法连接本地服务');
  });

  request.send(file);
}

viewOriginal.addEventListener('click', () => {
  if (currentVideo) setPreview(`${currentVideo.url}?v=${encodeURIComponent(currentVideo.modifiedAt)}`, viewOriginal);
});

viewStabilized.addEventListener('click', () => {
  const analysis = currentVideo?.analysis;
  if (analysis?.outputUrl) setPreview(`${analysis.outputUrl}?v=${encodeURIComponent(analysis.analyzedAt)}`, viewStabilized);
});

viewDepthSource.addEventListener('click', () => {
  const depth = currentVideo?.depth;
  if (depth?.sourceUrl) setPreview(`${depth.sourceUrl}?v=${encodeURIComponent(depth.createdAt)}`, viewDepthSource);
  else setPreview(depthInputUrl(), viewDepthSource);
});

viewDepthMap.addEventListener('click', () => {
  const depth = currentVideo?.depth;
  if (depth?.outputUrl) setPreview(`${depth.outputUrl}?v=${encodeURIComponent(depth.createdAt)}`, viewDepthMap);
});

stepUpload.addEventListener('click', () => showStage(1));
stepMotion.addEventListener('click', () => showStage(2));
stepDepth.addEventListener('click', () => showStage(3));
stepGaussians.addEventListener('click', () => showStage(4));
stepTracks.addEventListener('click', () => showStage(5));
nextTracksButton.addEventListener('click', () => showStage(5));
runDepthButton.addEventListener('click', runDepth);
runGaussiansButton.addEventListener('click', runGaussians);
runTracksButton.addEventListener('click', runTracks);
trackPlay.addEventListener('click', () => {
  if (trackViewer) void trackViewer.toggle().catch(error => showError(error.message));
});
trackTimeline.addEventListener('input', () => {
  if (!trackViewer) return;
  trackViewer.stop();
  void trackViewer.show(Number(trackTimeline.value)).catch(error => showError(error.message));
});
viewerPlay.addEventListener('click', () => {
  if (plyViewer) void plyViewer.toggle();
});
viewerTimeline.addEventListener('change', () => {
  if (!plyViewer) return;
  plyViewer.stop();
  void plyViewer.show(Number(viewerTimeline.value));
});

fileInput.addEventListener('change', () => {
  upload(fileInput.files?.[0]);
  fileInput.value = '';
});

for (const eventName of ['dragenter', 'dragover']) {
  dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.add('dragging');
  });
}

for (const eventName of ['dragleave', 'drop']) {
  dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.remove('dragging');
  });
}

dropZone.addEventListener('drop', event => upload(event.dataTransfer?.files?.[0]));
restoreLatest();

