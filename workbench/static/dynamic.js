export function createDynamicView(showError, formatBytes) {
  const el = name => document.querySelector(`#dynamic-${name}`);
  const panel = el('panel'), source = el('rgb'), image = el('image');
  let video, state, active = false, timer, playing = false, rendering = false;
  let frame = 0, pending = false, imageUrl, lastDisplay = 0, yaw = 0, pitch = 0;
  let position = [0, 0, 0], lastPointer;
  const phases = {preparing:'准备数据', sharp:'首帧 SHARP', flow:'双向光流',
    features:'提取对应特征', correspondences:'筛选三维对应', initialization:'可逆变形初始化',
    fusion:'多帧表面融合', joint:'联合优化', evaluating:'生成预览与指标', complete:'训练完成'};
  const metric = (label, value) => {
    const node = document.createElement('div'); node.className = 'metric';
    const labelNode = document.createElement('span'), valueNode = document.createElement('strong');
    labelNode.textContent = label; valueNode.textContent = value;
    node.append(labelNode, valueNode); return node;
  };
  function camera() {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return [cy, sy*sp, sy*cp, position[0], 0, cp, -sp, position[1],
      -sy, cy*sp, cy*cp, position[2], 0, 0, 0, 1];
  }
  function move(x, y, z) {
    const m = camera();
    for (let i=0; i<3; i++) position[i] += m[i*4]*x + m[i*4+1]*y + m[i*4+2]*z;
    requestRender();
  }
  async function api(path, method = 'GET') {
    const response = await fetch(`/api/dynamic${path}${method === 'GET' ? `?filename=${encodeURIComponent(video.name)}` : ''}`,
      method === 'GET' ? {cache:'no-store'} : {method, headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:video.name})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '第四步请求失败');
    return result.dynamic;
  }
  function display(next) {
    state = next;
    const busy = ['running','stopping'].includes(state?.status), complete = state?.status === 'complete';
    el('status').textContent = state ? (state.status === 'failed' ? '运行失败' : state.status === 'stopped' ? '已停止' : phases[state.stage] || state.message) : '尚未训练';
    el('note').textContent = state ? `${state.message || ''}${busy && state.total ? ` · ${state.step} / ${state.total}` : ''}` : '使用完整视频 RGB-D 和首帧 SHARP 高斯，训练一套共享动态场景。';
    el('start').hidden = busy || complete;
    el('start').textContent = state ? '继续训练' : '开始训练';
    el('stop').hidden = !busy; el('stop').disabled = state?.status === 'stopping';
    el('progress').hidden = !busy; el('progress').max = state?.total || 1; el('progress').value = state?.step || 0;
    document.querySelector('#step-dynamic').classList.toggle('done', complete);
    el('metrics').replaceChildren();
    if (state?.frameCount) el('metrics').append(metric('训练视频', `${state.frameCount} 帧 · ${state.width} × ${state.height}`));
    if (state?.elapsedSeconds) el('metrics').append(metric('累计耗时', `${(state.elapsedSeconds/3600).toFixed(2)} 小时`));
    if (state?.secondsPerStep && busy) el('metrics').append(metric('当前阶段', `${state.secondsPerStep.toFixed(2)} 秒 / 步`));
    if (state?.loss != null && busy) el('metrics').append(metric('当前损失', state.loss.toFixed(6)));
    if (complete) {
      const m = state.metrics;
      for (const [label,value] of [['PSNR',`${m.psnr.toFixed(2)} dB`],['SSIM',m.ssim.toFixed(4)],
        ['基础高斯',m.gaussianCount.toLocaleString()],['变形参数',m.parameterCount.toLocaleString()],
        ['基础 PLY',formatBytes(m.plyBytes)],['变形模型',formatBytes(m.modelBytes)],
        ['训练缓存',formatBytes(state.cacheBytes)],['本机渲染',`${m.renderFPS.toFixed(2)} FPS`]]) el('metrics').append(metric(label,value));
      el('ply').href = state.plyUrl; el('model').href = state.modelUrl; el('report').href = state.metricsUrl;
    }
    el('downloads').hidden = !complete;
    el('controls').hidden = !complete; el('camera-note').hidden = !complete;
    el('comparison').hidden = !complete && !state?.initialUrl;
    el('image-label').textContent = complete ? '动态高斯' : '首帧 SHARP 初始化';
    if (state?.initialUrl && !complete && !image.src.endsWith(state.initialUrl)) image.src = state.initialUrl;
    el('time').max = (state?.frameCount || video.depth.frameCount)-1;
    if (state?.error) showError(state.error);
    if (complete && !imageUrl) requestRender();
  }
  async function refresh() {
    if (!active) return;
    try { display(await api('')); } catch (error) { showError(error.message); }
    if (active) timer = setTimeout(refresh, 2000);
  }
  function setPlaying(value) { playing = value; el('play').textContent = value ? '暂停' : '播放'; }
  function syncTime() {
    const fps = state?.fps || video.depth.fps;
    el('time').value = frame;
    el('time-label').textContent = `${frame+1} / ${Number(el('time').max)+1} · ${(frame/fps).toFixed(2)} s`;
    if (source.readyState) source.currentTime = frame/fps;
  }
  async function requestRender() {
    if (!active || state?.status !== 'complete') return;
    if (rendering) { pending = true; return; }
    rendering = true; pending = false;
    const requestedFrame = frame, started = performance.now();
    try {
      const response = await fetch('/api/dynamic/render', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({filename:video.name, timeSeconds:requestedFrame/state.fps, camera:camera(), width:state.width, height:state.height})});
      if (!response.ok) throw new Error((await response.json()).error || '渲染失败');
      const url = URL.createObjectURL(await response.blob());
      if (!active) { URL.revokeObjectURL(url); return; }
      image.src = url;
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      imageUrl = url;
      if (frame === requestedFrame) syncTime();
      const now = performance.now();
      el('fps').textContent = `${(1000/(playing && lastDisplay ? now-lastDisplay : now-started)).toFixed(2)} 实际 FPS`;
      lastDisplay = now;
    } catch (error) { showError(error.message); setPlaying(false); }
    finally { rendering = false; }
    if (pending) { requestRender(); return; }
    if (playing && active) {
      frame = (frame+1) % (Number(el('time').max)+1);
      setTimeout(() => { if (playing && active) requestRender(); }, Math.max(0,1000/state.fps-(performance.now()-started)));
    }
  }
  el('start').addEventListener('click', async () => {
    el('start').disabled = true; showError(null);
    try { display(await api('', 'POST')); } catch(error) { showError(error.message); }
    finally { el('start').disabled = false; }
  });
  el('stop').addEventListener('click', async () => {
    el('stop').disabled = true;
    try { display(await api('/stop','POST')); } catch(error) { showError(error.message); el('stop').disabled = false; }
  });
  el('play').addEventListener('click', () => { setPlaying(!playing); if (playing) {lastDisplay = 0; requestRender();} });
  el('time').addEventListener('input', () => { setPlaying(false); frame = Number(el('time').value); requestRender(); });
  el('reset').addEventListener('click', () => { position = [0,0,0]; yaw = pitch = 0; requestRender(); });
  const surface = el('camera');
  surface.addEventListener('contextmenu', event => event.preventDefault());
  surface.addEventListener('pointerdown', event => { surface.focus(); surface.setPointerCapture(event.pointerId); lastPointer = {x:event.clientX,y:event.clientY,button:event.button}; });
  surface.addEventListener('pointerup', () => { lastPointer = null; });
  surface.addEventListener('pointercancel', () => { lastPointer = null; });
  surface.addEventListener('pointermove', event => {
    if (!lastPointer) return;
    const dx = event.clientX-lastPointer.x, dy = event.clientY-lastPointer.y;
    if (lastPointer.button === 2) move(-dx*.001, -dy*.001, 0);
    else { yaw += dx*.002; pitch += dy*.002; requestRender(); }
    lastPointer.x = event.clientX; lastPointer.y = event.clientY;
  });
  surface.addEventListener('wheel', event => { event.preventDefault(); move(0,0,-event.deltaY*.0005); }, {passive:false});
  surface.addEventListener('keydown', event => {
    const key = event.key.toLowerCase(), step = .025*(event.shiftKey ? 4 : 1);
    const moves = {w:[0,0,step],s:[0,0,-step],a:[-step,0,0],d:[step,0,0],q:[0,step,0],e:[0,-step,0]};
    if (moves[key]) { event.preventDefault(); move(...moves[key]); }
  });
  return {show(visible, selected) {
    active = visible; panel.hidden = !visible; clearTimeout(timer);
    if (!visible) { setPlaying(false); return; }
    if (video?.name !== selected.name) {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      imageUrl = null; frame = 0; position = [0,0,0]; yaw = pitch = 0;
      source.src = selected.depth.sourceUrl;
    }
    video = selected; display(video.dynamic); refresh();
  }};
}
