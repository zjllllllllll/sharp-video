import {createReadStream, createWriteStream} from 'node:fs';
import {mkdir, readFile, readdir, rename, stat, unlink, writeFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {extname, join, relative, resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {codecJobs} from './trajectory-codec-jobs.mjs';

const appDir = fileURLToPath(new URL('.', import.meta.url));
const trajectoryCodec = codecJobs(appDir);
const staticDir = join(appDir, 'static');
const uploadDir = join(appDir, 'uploads');
const runsDir = join(appDir, 'runs');
const vendorDir = join(appDir, 'node_modules');
const environmentPython = name => join(appDir, '..', '.envs', name,
  ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
const depthRepo = join(appDir, '..', 'video-depth-anything');
const depthPython = process.env.SHARP_VIDEO_DEPTH_PYTHON || environmentPython('video-depth-anything');
const depthWorker = join(appDir, 'depth_worker.py');
const depthCheckpoint = join(depthRepo, 'checkpoints', 'metric_video_depth_anything_vitb.pth');
const sharpRepo = join(appDir, '..', 'ml-sharp');
const sharpPython = process.env.SHARP_VIDEO_SHARP_PYTHON || environmentPython('sharp');
const sharpWorker = join(appDir, 'gaussian_worker.py');
const flowWorker = join(appDir, 'flow_worker.py');
const lifecycleWorker = join(appDir, 'lifecycle_worker.py');
const medianWorker = join(appDir, 'median_worker.py');
const sharpCheckpoint = process.env.SHARP_VIDEO_CHECKPOINT || join(sharpRepo, 'checkpoints', 'sharp_2572gikvuh.pt');
const mogeRepo = join(appDir, '..', 'moge');
const mogePython = process.env.SHARP_VIDEO_MOGE_PYTHON || environmentPython('moge2');
const mogeWorker = join(appDir, 'moge2_experiment_worker.py');
const mogeCheckpoint = join(mogeRepo, 'checkpoints', 'moge-2-vits-normal', 'model.pt');
const depthProRepo = join(appDir, '..', 'ml-depth-pro');
const depthProPython = process.env.SHARP_VIDEO_DEPTH_PRO_PYTHON || environmentPython('depth-pro');
const depthProWorker = join(appDir, 'depth_pro_experiment_worker.py');
const depthProCheckpoint = join(depthProRepo, 'checkpoints', 'depth_pro.pt');
const ffmpeg = process.env.SHARP_VIDEO_FFMPEG || 'ffmpeg';
const host = '127.0.0.1';
const port = Number(process.env.SHARP_VIDEO_PORT || 8775);
const thresholds = Object.freeze({
  fixedTranslationP95Px: 3,
  fixedRotationP95Deg: 0.05,
  shakyTranslationP95Px: 20,
  shakyRotationP95Deg: 0.5
});
const analysisJobs = new Map();
const depthJobs = new Map();
const gaussianJobs = new Map();
const trackJobs = new Map();
const medianJobs = new Map();
const experimentJobs = new Map();

await mkdir(uploadDir, {recursive: true});
await mkdir(runsDir, {recursive: true});

const mimeTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.m4v': 'video/x-m4v',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.mp4': 'video/mp4',
  '.png': 'image/png',
  '.ply': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.webm': 'video/webm'
};

function json(response, statusCode, value) {
  const body = Buffer.from(JSON.stringify(value));
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': body.length,
    'Cache-Control': 'no-store'
  });
  response.end(body);
}

function safeName(value) {
  let decoded = String(value || 'video');
  try { decoded = decodeURIComponent(decoded); } catch {}
  const base = decoded.split(/[\\/]/).pop() || 'video';
  const cleaned = base.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').trim();
  return cleaned || 'video';
}

function uniqueName(original) {
  const extension = extname(original);
  const stem = original.slice(0, original.length - extension.length) || 'video';
  const stamp = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14);
  return `${stem}-${stamp}${extension.toLowerCase()}`;
}

function inside(root, candidate) {
  const child = relative(resolve(root), resolve(candidate));
  return child === '' || (!child.startsWith('..') && !child.includes(':'));
}

async function listVideos() {
  const entries = await readdir(uploadDir, {withFileTypes: true});
  const files = await Promise.all(entries
    .filter(entry => entry.isFile() && !entry.name.startsWith('.'))
    .map(async entry => {
      const info = await stat(join(uploadDir, entry.name));
      const analysis = await readAnalysis(entry.name);
      const depth = await readDepth(entry.name);
      const gaussians = await readGaussians(entry.name, depth);
      const tracks = await readTracks(entry.name);
      return {
        name: entry.name,
        size: info.size,
        modifiedAt: info.mtime.toISOString(),
        url: `/uploads/${encodeURIComponent(entry.name)}`,
        analysis,
        depth,
        gaussians,
        tracks,
        median: await readMedian(entry.name),
      };
    }));
  return files.sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt));
}

function runIdFor(filename) {
  return Buffer.from(filename, 'utf8').toString('base64url');
}

async function readAnalysis(filename) {
  try {
    return JSON.parse(await readFile(join(runsDir, runIdFor(filename), 'analysis.json'), 'utf8'));
  } catch {
    return null;
  }
}

async function readDepth(filename) {
  try {
    const runId = runIdFor(filename);
    const depth = JSON.parse(await readFile(join(runsDir, runId, 'depth.json'), 'utf8'));
    return {
      ...depth,
      sourceUrl: `/results/${runId}/${depth.sourceVideo}`,
      outputUrl: `/results/${runId}/${depth.visualizationVideo}`,
      dataUrl: `/results/${runId}/${depth.depthData}`
    };
  } catch {
    return null;
  }
}

function decorateGaussians(filename, value) {
  if (!value) return null;
  const runId = runIdFor(filename);
  return {
    ...value,
    frames: (value.frames || []).map(frame => ({
      ...frame,
      fullUrl: `/results/${runId}/gaussians/${frame.full}`
    }))
  };
}

async function readGaussians(filename, depth = null) {
  const memory = gaussianJobs.get(filename);
  let saved = null;
  try {
    saved = JSON.parse(await readFile(join(runsDir, runIdFor(filename), 'gaussians', 'gaussian.json'), 'utf8'));
  } catch {}

  const expectedFrames = depth?.frameCount ?? (await readDepth(filename))?.frameCount ?? null;
  if (saved?.status === 'complete' && expectedFrames && saved.totalFrames !== expectedFrames) {
    saved = {...saved, status: 'incomplete', expectedFrames};
  }
  if (!memory) return decorateGaussians(filename, saved);
  return decorateGaussians(filename, {
    ...(saved || {}),
    status: memory.status,
    currentFrame: memory.current,
    totalFrames: memory.total,
    error: memory.error || saved?.error
  });
}

async function readTracks(filename) {
  const runId = runIdFor(filename);
  const job = trackJobs.get(filename);
  let tracks = null;
  let directory = 'tracks-lifecycle';
  for (const candidate of ['tracks-lifecycle', 'tracks-animation', 'tracks-indexed', 'tracks']) {
    try {
      tracks = JSON.parse(await readFile(join(runsDir, runId, candidate, 'tracks.json'), 'utf8'));
      directory = candidate;
      break;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  }
  if (job) {
    if (tracks?.format !== 'sharp.gaussian_tracks.v5') tracks = job.initial;
    tracks = {...tracks, status: job.status, stage: job.stage === 'flow' ? 'flow' : tracks?.stage || job.stage,
      error: job.error || tracks?.error};
    if (job.stage === 'flow') {
      try {
        const flow = JSON.parse(await readFile(join(runsDir, runId, 'flow', 'flow.json'), 'utf8'));
        tracks.currentPair = flow.currentPair;
        tracks.totalPairs = flow.totalPairs;
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
      }
    }
  }
  for (const spec of Object.values(tracks?.arrays || {})) {
    spec.url = `/results/${runId}/${directory}/${spec.file}`;
  }
  if (tracks?.sourceFrames) tracks.sourceFrames = tracks.sourceFrames.map(frame => ({
    ...frame, fullUrl: `/results/${runId}/gaussians/${frame.full}`
  }));
  if (tracks?.file) tracks.file.url = `/results/${runId}/${directory}/${tracks.file.file}`;
  return tracks;
}

async function readMedian(filename) {
  const runId = runIdFor(filename);
  let state = medianJobs.get(filename)?.initial || null;
  try { state = JSON.parse(await readFile(join(runsDir, runId, 'tracks-median', 'median.json'), 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (state?.file) state.file.url = `/results/${runId}/tracks-median/${state.file.file}`;
  return state;
}

async function startMedian(filename) {
  const tracks = await readTracks(filename);
  if (tracks?.status !== 'complete' || tracks.format !== 'sharp.gaussian_tracks.v5') {
    throw new Error('请先完成第五步高斯轨迹');
  }
  const previous = await readMedian(filename);
  if (previous?.status === 'complete' || medianJobs.has(filename)) return previous;
  const runDir = join(runsDir, runIdFor(filename));
  const outputDir = join(runDir, 'tracks-median');
  await mkdir(outputDir, {recursive: true});
  if (medianJobs.has(filename)) return readMedian(filename);
  const initial = {status: 'running', stage: 'median', currentFrame: 0,
    frameCount: tracks.frameCount, generatedAt: new Date().toISOString()};
  const job = {initial};
  medianJobs.set(filename, job);
  job.promise = (async () => {
    await writeFile(join(outputDir, 'median.json'), JSON.stringify(initial), 'utf8');
    await runProcess(sharpPython, [medianWorker, '--run-dir', runDir], appDir);
  })().catch(async error => {
    const saved = await readMedian(filename).catch(() => initial);
    const temporary = join(outputDir, 'median.failed.tmp');
    await writeFile(temporary, JSON.stringify({...saved, status: 'failed', error: error.message}), 'utf8');
    await rename(temporary, join(outputDir, 'median.json'));
  }).finally(() => medianJobs.delete(filename));
  return initial;
}

function decorateExperiment(filename, value) {
  if (!value) return null;
  const runId = runIdFor(filename);
  const base = `/results/${runId}/depth-experiment`;
  return {
    ...value,
    frameUrl: value.frameFile ? `${base}/${value.frameFile}` : null,
    models: value.models ? {
      moge2: {...value.models.moge2, pointCloudUrl: `${base}/moge2.ply`, depthUrl: `${base}/moge2_depth.png`},
      depthPro: {...value.models.depthPro, pointCloudUrl: `${base}/depth_pro.ply`, depthUrl: `${base}/depth_pro_depth.png`}
    } : null
  };
}

async function readExperiment(filename) {
  const memory = experimentJobs.get(filename);
  let saved = null;
  try {
    saved = JSON.parse(await readFile(
      join(runsDir, runIdFor(filename), 'depth-experiment', 'experiment.json'),
      'utf8'
    ));
  } catch {}
  if (!memory) return decorateExperiment(filename, saved);
  return decorateExperiment(filename, {
    ...(saved || {}),
    status: memory.status,
    stage: memory.stage,
    frameIndex: memory.frameIndex,
    error: memory.error || saved?.error
  });
}

function runProcess(executable, args, cwd) {
  return new Promise((resolveProcess, rejectProcess) => {
    const child = spawn(executable, args, {cwd, windowsHide: true});
    child.stdout.resume();
    const stderr = [];
    child.stderr.on('data', chunk => {
      stderr.push(chunk.toString());
      if (stderr.length > 80) stderr.shift();
    });
    child.once('error', rejectProcess);
    child.once('close', code => {
      if (code === 0) resolveProcess();
      else rejectProcess(new Error(stderr.join('').trim() || `进程退出码 ${code}`));
    });
  });
}

function runGaussianProcess(filename, args) {
  const job = gaussianJobs.get(filename);
  return new Promise((resolveProcess, rejectProcess) => {
    const child = spawn(sharpPython, args, {cwd: appDir, windowsHide: true});
    const stderr = [];
    let pending = '';
    child.stdout.on('data', chunk => {
      pending += chunk.toString();
      const lines = pending.split(/\r?\n/);
      pending = lines.pop() || '';
      for (const line of lines) {
        try {
          const event = JSON.parse(line);
          if (Number.isFinite(event.current)) job.current = event.current;
          if (Number.isFinite(event.total)) job.total = event.total;
          if (event.event === 'failed') {
            job.status = 'failed';
            job.error = event.error;
          }
        } catch {}
      }
    });
    child.stderr.on('data', chunk => {
      stderr.push(chunk.toString());
      if (stderr.length > 120) stderr.shift();
    });
    child.once('error', rejectProcess);
    child.once('close', code => {
      if (code === 0) resolveProcess();
      else rejectProcess(new Error(stderr.join('').trim() || `SHARP 进程退出码 ${code}`));
    });
  });
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)];
}

async function parseGlobalMotion(path) {
  const text = await readFile(path, 'utf8');
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const rows = lines
    .filter(line => !line.startsWith('#'))
    .map(line => line.split(/\s+/).map(Number))
    .filter(parts => parts.length >= 6 && parts.slice(0, 6).every(Number.isFinite));
  const noFields = lines.filter(line => line === '# no fields').length;
  if (rows.length < 2 || noFields >= rows.length) {
    return {reliable: false, frameCount: rows.length};
  }

  const translations = rows.slice(1).map(parts => Math.hypot(parts[1], parts[2]));
  const rotations = rows.slice(1).map(parts => Math.abs(parts[3] * 180 / Math.PI));
  return {
    reliable: true,
    frameCount: rows.length,
    translationP95Px: percentile(translations, 0.95),
    translationMaxPx: Math.max(...translations),
    rotationP95Deg: percentile(rotations, 0.95),
    rotationMaxDeg: Math.max(...rotations)
  };
}

async function analyzeVideo(filename) {
  const source = join(uploadDir, filename);
  if (!inside(uploadDir, source)) throw new Error('非法视频路径');
  await stat(source);

  const runId = runIdFor(filename);
  const runDir = join(runsDir, runId);
  await mkdir(runDir, {recursive: true});

  await runProcess(ffmpeg, [
    '-hide_banner', '-loglevel', 'warning', '-i', source,
    '-vf', 'vidstabdetect=shakiness=5:accuracy=15:tripod=1:result=transforms.trf',
    '-f', 'null', 'NUL'
  ], runDir);

  await runProcess(ffmpeg, [
    '-hide_banner', '-loglevel', 'warning', '-i', source,
    '-vf', 'vidstabtransform=input=transforms.trf:tripod=1:debug=1',
    '-f', 'null', 'NUL'
  ], runDir);

  const metrics = await parseGlobalMotion(join(runDir, 'global_motions.trf'));
  let status = 'uncertain';
  let allowed = false;
  let outputUrl = null;

  if (metrics.reliable) {
    if (metrics.translationP95Px <= thresholds.fixedTranslationP95Px &&
        metrics.rotationP95Deg <= thresholds.fixedRotationP95Deg) {
      status = 'fixed';
      allowed = true;
    } else if (metrics.translationP95Px <= thresholds.shakyTranslationP95Px &&
               metrics.rotationP95Deg <= thresholds.shakyRotationP95Deg) {
      status = 'stabilized';
      allowed = true;
      await runProcess(ffmpeg, [
        '-y', '-hide_banner', '-loglevel', 'warning', '-i', source,
        '-vf', 'vidstabtransform=input=transforms.trf:tripod=1:optzoom=1',
        '-map', '0:v:0', '-map', '0:a?',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', 'stabilized.mp4'
      ], runDir);
      outputUrl = `/results/${runId}/stabilized.mp4`;
    } else {
      status = 'rejected';
    }
  }

  const analysis = {
    status,
    allowed,
    metrics,
    thresholds,
    outputUrl,
    analyzedAt: new Date().toISOString()
  };
  await writeFile(join(runDir, 'analysis.json'), JSON.stringify(analysis, null, 2), 'utf8');
  return analysis;
}

async function generateDepth(filename) {
  const analysis = await readAnalysis(filename);
  if (!analysis?.allowed) throw new Error('视频未通过机位检测，不能预测深度');

  const runId = runIdFor(filename);
  const runDir = join(runsDir, runId);
  const source = analysis.status === 'stabilized'
    ? join(runDir, 'stabilized.mp4')
    : join(uploadDir, filename);
  if (!inside(runsDir, runDir) || !inside(uploadDir, join(uploadDir, filename))) {
    throw new Error('非法视频路径');
  }
  await stat(source);
  await mkdir(runDir, {recursive: true});
  await Promise.all([stat(depthPython), stat(depthCheckpoint)]);
  await runProcess(depthPython, [
    depthWorker,
    '--repo', depthRepo,
    '--checkpoint', depthCheckpoint,
    '--input', source,
    '--output-dir', runDir
  ], appDir);

  const depth = await readDepth(filename);
  if (!depth) throw new Error('深度预测完成，但没有生成结果描述文件');
  return depth;
}

async function startGaussianGeneration(filename) {
  const depth = await readDepth(filename);
  if (depth?.status !== 'complete') throw new Error('请先完成深度视频预测');

  const existing = await readGaussians(filename, depth);
  if (existing?.status === 'complete' && existing.totalFrames === depth.frameCount) return existing;
  if (gaussianJobs.has(filename)) return readGaussians(filename, depth);

  const runId = runIdFor(filename);
  const runDir = join(runsDir, runId);
  const gaussianDir = join(runDir, 'gaussians');
  const inputVideo = join(runDir, depth.sourceVideo);
  const depthData = join(runDir, depth.depthData);
  if (!inside(runsDir, gaussianDir)) throw new Error('非法结果路径');
  await Promise.all([
    stat(sharpPython), stat(sharpWorker), stat(sharpCheckpoint),
    stat(inputVideo), stat(depthData)
  ]);
  await mkdir(gaussianDir, {recursive: true});

  const job = {status: 'running', current: 0, total: depth.frameCount, error: null};
  gaussianJobs.set(filename, job);
  const args = [
    sharpWorker,
    '--repo', sharpRepo,
    '--checkpoint', sharpCheckpoint,
    '--input-video', inputVideo,
    '--depth-npz', depthData,
    '--output-dir', gaussianDir
  ];
  job.promise = runGaussianProcess(filename, args)
    .then(() => { job.status = 'complete'; job.current = job.total; })
    .catch(error => { job.status = 'failed'; job.error = error.message; })
    .finally(() => {
      setTimeout(() => {
        if (gaussianJobs.get(filename) === job) gaussianJobs.delete(filename);
      }, 5000);
    });
  return readGaussians(filename, depth);
}

async function startTrackGeneration(filename) {
  const gaussians = await readGaussians(filename);
  if (gaussians?.status !== 'complete') throw new Error('请先完成逐帧高斯生成');
  const existing = await readTracks(filename);
  if ((existing?.status === 'complete' && existing.format === 'sharp.gaussian_tracks.v5') || trackJobs.has(filename)) return existing;

  const runDir = join(runsDir, runIdFor(filename));
  const outputDir = join(runDir, 'tracks-lifecycle');
  await mkdir(outputDir, {recursive: true});
  // Recheck after asynchronous reads so simultaneous clicks share one job.
  if (trackJobs.has(filename)) return readTracks(filename);
  const job = {status: 'running', stage: 'tracks', error: null};
  trackJobs.set(filename, job);
  job.promise = (async () => {
    const starting = job.initial = {
      format: 'sharp.gaussian_tracks.v5', positionEncoding: 'full_attribute_samples',
      status: 'running', stage: 'tracks', currentFrame: 0, frameCount: gaussians.frames.length,
      generatedAt: new Date().toISOString()
    };
    const temporary = join(outputDir, 'tracks.json.tmp');
    await writeFile(temporary, JSON.stringify(starting, null, 2), 'utf8');
    await rename(temporary, join(outputDir, 'tracks.json'));
    let flow = null;
    try { flow = JSON.parse(await readFile(join(runDir, 'flow', 'flow.json'), 'utf8')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (flow?.status !== 'complete') {
      job.stage = 'flow';
      await runProcess(sharpPython, [flowWorker, '--run-dir', runDir], appDir);
    }
    job.stage = 'tracks';
    await runProcess(sharpPython, [lifecycleWorker, '--run-dir', runDir], appDir);
    job.status = 'complete';
  })().catch(async error => {
    job.status = 'failed';
    job.error = error.message;
    let previous = null;
    try { previous = await readTracks(filename); } catch {}
    const failed = {...previous, format: 'sharp.gaussian_tracks.v5', positionEncoding: 'full_attribute_samples',
      status: 'failed', stage: job.stage, error: error.message};
    try {
      const temporary = join(outputDir, 'tracks.json.tmp');
      await writeFile(temporary, JSON.stringify(failed, null, 2), 'utf8');
      await rename(temporary, join(outputDir, 'tracks.json'));
    } catch (saveError) {
      console.error('无法保存轨迹失败状态：', saveError.message);
    }
  }).finally(() => {
    if (trackJobs.get(filename) === job) trackJobs.delete(filename);
  });
  return readTracks(filename);
}

async function startDepthExperiment(filename) {
  const existing = await readExperiment(filename);
  if (existing?.status === 'complete') return existing;
  if (experimentJobs.has(filename)) return existing;

  const runId = runIdFor(filename);
  const runDir = join(runsDir, runId);
  const outputDir = join(runDir, 'depth-experiment');
  const depth = await readDepth(filename);
  const analysis = await readAnalysis(filename);
  const source = depth?.sourceVideo
    ? join(runDir, depth.sourceVideo)
    : analysis?.status === 'stabilized'
      ? join(runDir, 'stabilized.mp4')
      : join(uploadDir, filename);
  const frameIndex = Math.max(0, Math.floor(((depth?.frameCount || 1) - 1) / 2));
  const frameFile = `frame_${String(frameIndex).padStart(5, '0')}.png`;
  const framePath = join(outputDir, frameFile);
  if (!inside(runsDir, outputDir)) throw new Error('非法结果路径');
  await Promise.all([
    stat(source), stat(mogePython), stat(mogeWorker), stat(mogeCheckpoint),
    stat(depthProPython), stat(depthProWorker), stat(depthProCheckpoint)
  ]);
  await mkdir(outputDir, {recursive: true});

  const job = {status: 'running', stage: 'extracting', frameIndex, error: null};
  experimentJobs.set(filename, job);
  job.promise = (async () => {
    await runProcess(ffmpeg, [
      '-y', '-hide_banner', '-loglevel', 'error',
      '-ss', String(frameIndex / (depth?.fps || 25)), '-i', source,
      '-frames:v', '1', framePath
    ], appDir);
    job.stage = 'moge2';
    await runProcess(mogePython, [
      mogeWorker, '--repo', mogeRepo, '--checkpoint', mogeCheckpoint,
      '--image', framePath, '--output-dir', outputDir
    ], appDir);
    job.stage = 'depthPro';
    await runProcess(depthProPython, [
      depthProWorker, '--checkpoint', depthProCheckpoint,
      '--image', framePath, '--output-dir', outputDir
    ], depthProRepo);
    const models = {
      moge2: JSON.parse(await readFile(join(outputDir, 'moge2.json'), 'utf8')),
      depthPro: JSON.parse(await readFile(join(outputDir, 'depth_pro.json'), 'utf8'))
    };
    const manifest = {
      status: 'complete', frameIndex, frameFile, models,
      completedAt: new Date().toISOString()
    };
    await writeFile(join(outputDir, 'experiment.json'), JSON.stringify(manifest, null, 2), 'utf8');
    job.status = 'complete';
    job.stage = 'complete';
  })().catch(async error => {
    job.status = 'failed';
    job.error = error.message;
    const manifest = {status: 'failed', frameIndex, frameFile, error: error.message};
    await writeFile(join(outputDir, 'experiment.json'), JSON.stringify(manifest, null, 2), 'utf8');
  }).finally(() => {
    setTimeout(() => {
      if (experimentJobs.get(filename) === job) experimentJobs.delete(filename);
    }, 5000);
  });
  return readExperiment(filename);
}

function readJson(request) {
  return new Promise((resolveBody, rejectBody) => {
    const chunks = [];
    let bytes = 0;
    request.on('data', chunk => {
      bytes += chunk.length;
      if (bytes > 64 * 1024) {
        rejectBody(new Error('请求内容过大'));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.once('end', () => {
      try { resolveBody(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { rejectBody(new Error('无效请求')); }
    });
    request.once('error', rejectBody);
  });
}

async function sendFile(request, response, path) {
  const info = await stat(path);
  if (!info.isFile()) throw new Error('not found');

  const mime = mimeTypes[extname(path).toLowerCase()] || 'application/octet-stream';
  const range = request.headers.range;
  const commonHeaders = {
    'Content-Type': mime,
    'Accept-Ranges': 'bytes',
    'Cache-Control': path.startsWith(uploadDir) ? 'no-cache' : 'no-store'
  };

  if (!range) {
    response.writeHead(200, {...commonHeaders, 'Content-Length': info.size});
    createReadStream(path).pipe(response);
    return;
  }

  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match) {
    response.writeHead(416, {'Content-Range': `bytes */${info.size}`});
    response.end();
    return;
  }

  const start = match[1] ? Number(match[1]) : 0;
  const end = match[2] ? Math.min(Number(match[2]), info.size - 1) : info.size - 1;
  if (start > end || start >= info.size) {
    response.writeHead(416, {'Content-Range': `bytes */${info.size}`});
    response.end();
    return;
  }

  response.writeHead(206, {
    ...commonHeaders,
    'Content-Length': end - start + 1,
    'Content-Range': `bytes ${start}-${end}/${info.size}`
  });
  createReadStream(path, {start, end}).pipe(response);
}

async function receiveUpload(request, response) {
  const original = safeName(request.headers['x-file-name']);
  const extension = extname(original).toLowerCase();
  const contentType = String(request.headers['content-type'] || '');
  const videoExtensions = new Set(['.mp4', '.mov', '.m4v', '.webm', '.mkv']);
  if (!contentType.startsWith('video/') && !videoExtensions.has(extension)) {
    request.resume();
    json(response, 400, {error: '请选择视频文件'});
    return;
  }

  const filename = uniqueName(original);
  const finalPath = join(uploadDir, filename);
  const temporaryPath = `${finalPath}.uploading`;
  const output = createWriteStream(temporaryPath, {flags: 'wx'});

  let bytes = 0;
  request.on('data', chunk => { bytes += chunk.length; });
  request.pipe(output);

  const failed = new Promise((_, reject) => {
    request.once('error', reject);
    output.once('error', reject);
  });
  const finished = new Promise(resolveFinished => output.once('finish', resolveFinished));

  try {
    await Promise.race([finished, failed]);
    await rename(temporaryPath, finalPath);
    json(response, 201, {
      video: {
        name: filename,
        size: bytes,
        modifiedAt: new Date().toISOString(),
        url: `/uploads/${encodeURIComponent(filename)}`
      }
    });
  } catch (error) {
    await unlink(temporaryPath).catch(() => {});
    json(response, 500, {error: `上传失败：${error.message}`});
  }
}

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host || `${host}:${port}`}`);

    if (request.method === 'GET' && url.pathname === '/api/state') {
      json(response, 200, {videos: await listVideos()});
      return;
    }

    if (request.method === 'POST' && url.pathname === '/api/upload') {
      await receiveUpload(request, response);
      return;
    }

    if (request.method === 'POST' && url.pathname === '/api/analyze') {
      const body = await readJson(request);
      const filename = safeName(body.filename);
      let task = analysisJobs.get(filename);
      if (!task) {
        task = analyzeVideo(filename).finally(() => analysisJobs.delete(filename));
        analysisJobs.set(filename, task);
      }
      json(response, 200, {analysis: await task});
      return;
    }

    if (request.method === 'POST' && url.pathname === '/api/depth') {
      const body = await readJson(request);
      const filename = safeName(body.filename);
      let task = depthJobs.get(filename);
      if (!task) {
        task = generateDepth(filename).finally(() => depthJobs.delete(filename));
        depthJobs.set(filename, task);
      }
      json(response, 200, {depth: await task});
      return;
    }

    if (url.pathname === '/api/gaussians' && request.method === 'POST') {
      const body = await readJson(request);
      const filename = safeName(body.filename);
      json(response, 202, {gaussians: await startGaussianGeneration(filename)});
      return;
    }

    if (url.pathname === '/api/gaussians' && request.method === 'GET') {
      const filename = safeName(url.searchParams.get('filename'));
      json(response, 200, {gaussians: await readGaussians(filename)});
      return;
    }

    if (url.pathname === '/api/tracks' && request.method === 'POST') {
      const body = await readJson(request);
      json(response, 202, {tracks: await startTrackGeneration(safeName(body.filename))});
      return;
    }

    if (url.pathname === '/api/tracks' && request.method === 'GET') {
      json(response, 200, {tracks: await readTracks(safeName(url.searchParams.get('filename')))});
      return;
    }

    if (url.pathname === '/api/median' && request.method === 'POST') {
      const body = await readJson(request);
      json(response, 202, {median: await startMedian(safeName(body.filename))});
      return;
    }

    if (url.pathname === '/api/median' && request.method === 'GET') {
      json(response, 200, {median: await readMedian(safeName(url.searchParams.get('filename')))});
      return;
    }

    if (url.pathname === '/api/trajectory-codec' && request.method === 'GET') {
      json(response, 200, {codec: await trajectoryCodec.read(safeName(url.searchParams.get('filename')))});
      return;
    }
    if (url.pathname === '/api/trajectory-codec' && request.method === 'POST') {
      const body = await readJson(request);
      json(response, 202, {codec: await trajectoryCodec.start(safeName(body.filename))});
      return;
    }
    if (url.pathname === '/api/trajectory-codec/stop' && request.method === 'POST') {
      const body = await readJson(request);
      json(response, 200, {codec: await trajectoryCodec.stop(safeName(body.filename))});
      return;
    }

    if (url.pathname === '/api/depth-experiment' && request.method === 'POST') {
      const body = await readJson(request);
      const filename = safeName(body.filename);
      json(response, 202, {experiment: await startDepthExperiment(filename)});
      return;
    }

    if (url.pathname === '/api/depth-experiment' && request.method === 'GET') {
      const filename = safeName(url.searchParams.get('filename'));
      json(response, 200, {experiment: await readExperiment(filename)});
      return;
    }

    if (request.method === 'GET' && url.pathname.startsWith('/uploads/')) {
      const name = safeName(url.pathname.slice('/uploads/'.length));
      const path = join(uploadDir, name);
      if (!inside(uploadDir, path)) throw new Error('not found');
      await sendFile(request, response, path);
      return;
    }

    if (request.method === 'GET' && url.pathname.startsWith('/results/')) {
      const parts = url.pathname.split('/').filter(Boolean);
      const allowedResults = new Set(['stabilized.mp4', 'depth_source.mp4', 'depth_vis.mp4', 'depths.npz']);
      let path;
      if (parts.length === 3 && allowedResults.has(parts[2])) {
        path = join(runsDir, safeName(parts[1]), parts[2]);
      } else if (parts.length === 4 && parts[2] === 'gaussians' &&
                 /^frame_\d{5}(?:_preview)?\.ply$/.test(parts[3])) {
        path = join(runsDir, safeName(parts[1]), 'gaussians', parts[3]);
      } else if (parts.length === 4 && parts[2] === 'tracks' &&
                 /^(?:ids|times|reference|uv|positions|displacements|valid)\.npy$/.test(parts[3])) {
        path = join(runsDir, safeName(parts[1]), 'tracks', parts[3]);
      } else if (parts.length === 4 && parts[2] === 'tracks-indexed' &&
                 /^(?:source_cell_indices|times)\.npy$/.test(parts[3])) {
        path = join(runsDir, safeName(parts[1]), 'tracks-indexed', parts[3]);
      } else if (parts.length === 4 && ['tracks-animation', 'tracks-lifecycle', 'tracks-median'].includes(parts[2]) && parts[3] === 'animation.4dgs') {
        path = join(runsDir, safeName(parts[1]), parts[2], parts[3]);
      } else if (parts.length === 4 && parts[2] === 'trajectory-codec' &&
                 ['trajectory-codec.pt', 'animation.4dgs', 'report.json'].includes(parts[3])) {
        path = join(runsDir, safeName(parts[1]), parts[2], parts[3]);
      } else if (parts.length === 4 && parts[2] === 'depth-experiment' &&
                 /^(?:frame_\d{5}|moge2_depth|depth_pro_depth)\.png$|^(?:moge2|depth_pro)\.ply$/.test(parts[3])) {
        path = join(runsDir, safeName(parts[1]), 'depth-experiment', parts[3]);
      } else {
        throw new Error('not found');
      }
      if (!inside(runsDir, path)) throw new Error('not found');
      await sendFile(request, response, path);
      return;
    }

    if (request.method === 'GET' && url.pathname.startsWith('/vendor/')) {
      const relative = url.pathname.slice('/vendor/'.length);
      const path = join(vendorDir, relative);
      if (!inside(vendorDir, path) || !['.js', '.mjs', '.wasm'].includes(extname(path))) {
        throw new Error('not found');
      }
      await sendFile(request, response, path);
      return;
    }

    if (request.method === 'GET' && url.pathname === '/tracks.html') {
      response.writeHead(302, {Location: '/#tracks', 'Cache-Control': 'no-store'});
      response.end();
      return;
    }

    if (request.method === 'GET') {
      const relative = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
      const path = join(staticDir, relative);
      if (!inside(staticDir, path)) throw new Error('not found');
      await sendFile(request, response, path);
      return;
    }

    json(response, 405, {error: 'method not allowed'});
  } catch (error) {
    const isApi = request.url?.startsWith('/api/');
    json(response, isApi ? 500 : 404, {error: isApi ? error.message : 'not found'});
  }
});

server.listen(port, host, () => {
  console.log(`SHARP Video Workbench: http://${host}:${port}`);
});

