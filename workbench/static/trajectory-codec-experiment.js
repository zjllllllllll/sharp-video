import {CodecComparisonViewer} from './codec-comparison-viewer.js';
const $ = id => document.getElementById(id);
const stages = {preparing:'读取原始轨迹与归一化统计',training:'训练编码和共享解码器',checkpoint:'正在保存检查点',exporting:'导出独立编码文件',exported:'编码文件已导出',decoding:'本机解码完整序列',decoded:'解码完成',evaluating:'渲染对照并计算误差',statistics:'统计全部属性误差',complete:'实验完成'};
let videos=[], selected=null, state=null, player=null, loaded=null, busy=false, preparationStart=0, generation=0;
const number = value => Number(value).toLocaleString('zh-CN');
const seconds = value => `${Number(value).toFixed(1)} s`;
const gb = value => `${(value/1e9).toFixed(3)} GB`;
function error(message) { $('error').hidden=!message; $('error').textContent=message||''; }
async function api(path, body) {
  const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{cache:'no-store'});
  const data=await response.json(); if(!response.ok) throw new Error(data.error||`请求失败 ${response.status}`); return data;
}
function metric(label,value) {const box=document.createElement('div');const title=document.createElement('span');title.textContent=label;const text=document.createElement('strong');text.textContent=value;box.append(title,text);$('metrics').append(box);}
function plot(rows=[]) {
  const canvas=$('loss'), ratio=devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*ratio);canvas.height=180*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const w=canvas.clientWidth,h=180;ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#95a28f';ctx.font='12px sans-serif';ctx.fillText('总损失',0,14);
  if(!rows.length)return;const max=Math.max(...rows.map(r=>r.loss),1e-9);ctx.strokeStyle='#b5d790';ctx.lineWidth=1.6;ctx.beginPath();
  rows.forEach((r,i)=>{const x=45+(w-55)*r.step/(state?.totalSteps||rows.at(-1).step);const y=24+(h-42)*(1-r.loss/max);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  ctx.fillText(max.toPrecision(3),0,30);ctx.fillText('0',22,h-15);ctx.fillText(`${rows.at(-1).step} 次更新`,Math.max(50,w-120),h-2);
}
function render() {
  const running=state?.status==='running', complete=state?.status==='complete';
  $('start').disabled=busy||running||complete||!selected;
  $('start').textContent=complete?'训练已完成':state?.resumable?'继续实验':'开始训练';
  $('stop').disabled=busy||!running||state?.stopRequested;
  $('stop').textContent=state?.stopRequested?'正在停止并保存…':'停止并保存';
  $('video').disabled=running;
  const suffix=state?.status==='stopped'?' · 已停止':state?.status==='interrupted'?' · 进程已中断，可继续':state?.status==='failed'?' · 失败':'';
  $('status').textContent=state?(stages[state.stage]||'准备任务')+suffix:'尚未训练';
  let percent=0, label='';
  if(['training','checkpoint','exporting','exported'].includes(state?.stage)) {percent=100*(state.globalStep||0)/(state.totalSteps||1);label=`${state.globalStep||0} / ${state.totalSteps||36000} 次更新 · ${(state.epochFraction??state.epoch??0).toFixed(2)} / 20 轮`;}
  else if(state) {percent=100*(state.currentFrame||0)/(state.frameCount||100);label=`${state.currentFrame||0} / ${state.frameCount||100} 帧`;}
  if(complete){percent=100;label='100 帧 · 20 轮';}
  $('progress').value=percent;$('progress-text').textContent=label;
  $('detail').textContent=state?.stopRequested?'当前计算结束后保存进度，之后可继续。':state?'刷新页面会恢复任务进度。训练结束后自动导出、解码并计算误差。':'尚无重建结果。开始后在这里查看进度，完成后可对照原始和重建画面。';
  error(state?.error);
  $('metrics').replaceChildren();
  if(state?.gaussianTrackCount)metric('高斯生命周期',number(state.gaussianTrackCount));
  if(state?.networkParameters)metric('共享网络参数',number(state.networkParameters));
  if(state?.trainingSeconds!=null)metric('训练耗时',seconds(state.trainingSeconds));
  if(state?.modelBytes)metric('编码模型',gb(state.modelBytes));
  if(state?.compressionRatio)metric('压缩倍数',`${state.compressionRatio.toFixed(2)}×`);
  if(state?.trainingCacheBytes||state?.checkpointBytes)metric('训练缓存',gb(state.trainingCacheBytes||state.checkpointBytes));
  if(state?.previewCacheBytes)metric('解码预览缓存',gb(state.previewCacheBytes));
  if(state?.decodeFPS)metric('本机解码速度',`${state.decodeFPS.toFixed(2)} FPS`);
  if(state?.decodePreparationSeconds)metric('完整解码准备',seconds(state.decodePreparationSeconds));
  if(state?.report){metric('原视角 PSNR',`${state.report.psnr.toFixed(2)} dB`);metric('原视角 SSIM',state.report.ssim.toFixed(4));}
  plot(state?.lossHistory);
  if(state?.losses)$('loss-detail').textContent=Object.entries(state.losses).map(([k,v])=>`${({position:'位置',color:'颜色',scale:'尺度',opacity:'透明度',rotation:'旋转'})[k]} ${v.toPrecision(4)}`).join(' · ');
  $('result').hidden=!state?.modelUrl;$('download').href=state?.modelUrl||'#';$('load-preview').disabled=!state?.preview||Boolean(loaded);
  $('report').hidden=!state?.reportUrl;$('report').href=state?.reportUrl||'#';
  $('accuracy').hidden=!state?.report;
  if(state?.report){$('error-rows').replaceChildren();for(const [name,v] of Object.entries(state.report.attributes)){const tr=document.createElement('tr');for(const value of [name,v.mae,v.p95,v.p99]){const td=document.createElement('td');td.textContent=typeof value==='number'?value.toPrecision(5):value;tr.append(td);}$('error-rows').append(tr);}}
}
async function refresh() {if(!selected)return;const token=generation;const data=await api(`/api/trajectory-codec?filename=${encodeURIComponent(selected.name)}`);if(token!==generation)return;state=data.codec;render();}
async function action(stop=false) {if(!selected)return;busy=true;render();let failure;try{state=(await api(`/api/trajectory-codec${stop?'/stop':''}`,{filename:selected.name})).codec;}catch(e){failure=e.message;}finally{busy=false;render();if(failure)error(failure);}}
$('start').onclick=()=>action();$('stop').onclick=()=>action(true);
$('video').onchange=async()=>{generation++;selected=videos.find(v=>v.name===$('video').value);state=null;loaded=null;player?.stop();player?.clearMeshes();$('viewer').hidden=true;$('load-preview').disabled=false;render();await refresh();};
$('load-preview').onclick=async()=>{
  if(!state?.preview||loaded)return;loaded=selected.name;preparationStart=performance.now();$('load-preview').disabled=true;$('viewer').hidden=false;
  if(!player)player=new CodecComparisonViewer($('viewer'),{
    onFrame:(index,count)=>{$('timeline').max=count-1;$('timeline').value=index;$('frame').textContent=`${index+1} / ${count} · ${(index/(state?.preview?.fps||25)).toFixed(2)} s`;},
    onPlayback:playing=>{$('play').textContent=playing?'暂停':'播放';},
    onBuffering:p=>{for(const id of ['original','decoded','play','timeline','reset'])$(id).disabled=p.preparing;$('preview-status').textContent=p.preparing?`网页缓存准备 ${p.ready} / ${p.total} 帧`:`网页缓存准备完成，用时 ${((performance.now()-preparationStart)/1000).toFixed(1)} s。切换保持同一帧、同一视角。`;$('fps').textContent=p.preparing?'':'已就绪';},
    onPerformance:p=>{$('fps').textContent=`缓存播放 ${p.fps.toFixed(1)} FPS`;},onError:e=>error(e.message)
  });
  player.resize();try{await player.loadPair(selected.tracks,state.preview);}catch(e){loaded=null;$('load-preview').disabled=false;error(e.message);}
};
for(const [id,key] of [['original','original'],['decoded','filtered']])$(id).onclick=()=>{player?.selectSource(key);$('original').classList.toggle('active',id==='original');$('decoded').classList.toggle('active',id==='decoded');};
$('play').onclick=()=>player?.toggle().catch(e=>error(e.message));
$('timeline').oninput=()=>{player?.stop();player?.show(Number($('timeline').value)).catch(e=>error(e.message));};
$('reset').onclick=()=>{if(player){player.camera.position.set(0,0,0);player.yaw=player.pitch=0;player.camera.rotation.set(0,0,0);}};
try {
  videos=(await api('/api/state')).videos.filter(v=>v.tracks?.status==='complete'&&v.tracks?.format==='sharp.gaussian_tracks.v5');
  for(const video of videos){const option=document.createElement('option');option.value=option.textContent=video.name;$('video').append(option);}
  selected=videos.find(v=>v.name===new URLSearchParams(location.search).get('filename'))||videos[0];
  if(selected){$('video').value=selected.name;await refresh();}else{$('status').textContent='请先在工作台完成第五步高斯轨迹';}
}catch(e){error(e.message);}
setInterval(()=>{if(!busy)refresh().catch(e=>error(e.message));},3000);
