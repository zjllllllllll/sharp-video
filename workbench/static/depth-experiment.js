import {PointCloudViewer} from './pointcloud-viewer.js';

const runButton = document.querySelector('#run-experiment');
const fileLabel = document.querySelector('#experiment-file');
const progress = document.querySelector('#experiment-progress');
const status = document.querySelector('#experiment-status');
const sourcePanel = document.querySelector('#source-panel');
const sourceFrame = document.querySelector('#source-frame');
const sourceLabel = document.querySelector('#source-label');
const comparison = document.querySelector('#comparison');
const errorBox = document.querySelector('#experiment-error');
const mogeStatus = document.querySelector('#moge-status');
const depthProStatus = document.querySelector('#depthpro-status');
let video = null;
let mogeViewer = null;
let depthProViewer = null;

function metrics(model) {
  return `<span>焦距 <strong>${model.focalX.toFixed(1)} px</strong></span>` +
    `<span>深度 P02–P98 <strong>${model.depthNearP02.toFixed(2)}–${model.depthFarP98.toFixed(2)} m</strong></span>`;
}

async function request(url, options) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || '请求失败');
  return value;
}

function showError(error) {
  errorBox.hidden = !error;
  errorBox.textContent = error || '';
}

async function showResult(result) {
  progress.hidden = true;
  sourcePanel.hidden = false;
  comparison.hidden = false;
  sourceFrame.src = result.frameUrl;
  sourceLabel.textContent = `第 ${result.frameIndex + 1} 帧`;
  document.querySelector('#moge-depth').src = result.models.moge2.depthUrl;
  document.querySelector('#depthpro-depth').src = result.models.depthPro.depthUrl;
  document.querySelector('#moge-metrics').innerHTML = metrics(result.models.moge2);
  document.querySelector('#depthpro-metrics').innerHTML = metrics(result.models.depthPro);
  mogeViewer ||= new PointCloudViewer(document.querySelector('#moge-viewer'), mogeStatus);
  depthProViewer ||= new PointCloudViewer(document.querySelector('#depthpro-viewer'), depthProStatus);
  await Promise.all([
    mogeViewer.load(result.models.moge2.pointCloudUrl, result.models.moge2),
    depthProViewer.load(result.models.depthPro.pointCloudUrl, result.models.depthPro)
  ]);
}

async function poll() {
  const {experiment} = await request(`/api/depth-experiment?filename=${encodeURIComponent(video.name)}`);
  if (!experiment) {
    progress.hidden = false;
    progress.querySelector('.spinner').hidden = true;
    status.textContent = '尚未运行';
    return;
  }
  if (experiment.status === 'complete') {
    runButton.disabled = false;
    await showResult(experiment);
    return;
  }
  if (experiment.status === 'failed') throw new Error(experiment.error || '实验失败');
  const labels = {extracting: '正在提取中间帧…', moge2: '正在运行 MoGe-2…', depthPro: '正在运行 Depth Pro…'};
  progress.hidden = false;
  progress.querySelector('.spinner').hidden = false;
  status.textContent = labels[experiment.stage] || '正在准备模型…';
  setTimeout(() => poll().catch(showError), 1200);
}

async function run() {
  if (!video) return;
  showError('');
  runButton.disabled = true;
  progress.hidden = false;
  progress.querySelector('.spinner').hidden = false;
  status.textContent = '正在启动实验…';
  await request('/api/depth-experiment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: video.name})
  });
  await poll();
}

runButton.addEventListener('click', () => run().catch(error => {
  runButton.disabled = false;
  showError(error.message);
}));

try {
  const state = await request('/api/state');
  video = state.videos?.[0] || null;
  if (!video) throw new Error('工作台里还没有视频');
  fileLabel.textContent = video.name;
  await poll();
} catch (error) {
  runButton.disabled = true;
  showError(error.message);
}
