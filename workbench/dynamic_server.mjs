import {createWriteStream} from 'node:fs';
import {readFile, writeFile, mkdir, open, unlink, stat} from 'node:fs/promises';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';

export function createDynamicManager(appDir, runsDir, runIdFor, readDepth) {
  const python = process.env.SHARP_VIDEO_DYNAMIC_PYTHON || join(appDir, '..', '.envs', 'dynamic',
    ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
  const jobs = new Map();
  let renderer = null;
  let sequence = 0;
  const runDir = filename => join(runsDir, runIdFor(filename));
  const outputDir = filename => join(runDir(filename), 'dynamic');
  const alive = pid => {
    if (!Number.isSafeInteger(pid) || pid <= 0) return false;
    try { process.kill(pid, 0); return true; } catch { return false; }
  };

  async function read(filename) {
    let result;
    try { result = JSON.parse(await readFile(join(outputDir(filename), 'status.json'), 'utf8')); }
    catch { return null; }
    if (['running', 'stopping'].includes(result.status) && !alive(result.pid)) {
      result = {...result, status: 'stopped', message: '计算进程已结束'};
    }
    if (result.status === 'running') {
      try { await stat(join(outputDir(filename), 'stop.request')); result = {...result, status: 'stopping', message: '正在停止，将在当前计算块结束后退出'}; } catch {}
    }
    const base = `/results/${runIdFor(filename)}/dynamic`;
    let initialUrl = null;
    try { await stat(join(outputDir(filename), 'initial.png')); initialUrl = `${base}/initial.png`; } catch {}
    return {...result, initialUrl,
      plyUrl: result.status === 'complete' ? `${base}/canonical.ply` : null,
      modelUrl: result.status === 'complete' ? `${base}/deformation.pt` : null,
      metricsUrl: result.status === 'complete' ? `${base}/metrics.json` : null};
  }

  function closeRenderer() {
    if (renderer) renderer.child.kill();
    renderer = null;
  }

  async function start(filename) {
    const depth = await readDepth(filename);
    if (depth?.status !== 'complete') throw new Error('请先完成第三步深度预测');
    const previous = await read(filename);
    if (previous?.status === 'complete' || ['running', 'stopping'].includes(previous?.status)) return previous;
    const folder = outputDir(filename);
    await mkdir(folder, {recursive: true});
    const lock = join(folder, 'launch.lock');
    // Both local ports share this directory. The lock covers the launch race only.
    let handle;
    try { handle = await open(lock, 'wx'); }
    catch (error) {
      if (error.code === 'EEXIST') throw new Error('训练正在启动，请稍候刷新');
      throw error;
    }
    try {
      await stat(python);
      if (previous) await writeFile(join(folder, 'previous_status.json'), JSON.stringify(previous), 'utf8');
      await unlink(join(folder, 'stop.request')).catch(error => { if (error.code !== 'ENOENT') throw error; });
      closeRenderer();
      const log = createWriteStream(join(folder, 'worker.log'), {flags: 'a'});
      const child = spawn(python, ['-u', join(appDir, 'dynamic_worker.py'), '--run-dir', runDir(filename)],
        {cwd: appDir, windowsHide: true, env: {...process.env, PYTHONUTF8: '1'}});
      jobs.set(filename, child);
      child.stdout.pipe(log, {end: false});
      child.stderr.pipe(log, {end: false});
      const initial = {status: 'running', stage: 'preparing', step: 0, total: 1,
        pid: child.pid, message: '正在启动第四步', width: depth.width, height: depth.height,
        frameCount: depth.frameCount, fps: depth.fps, startedAt: Date.now()/1000};
      await writeFile(join(folder, 'status.json'), JSON.stringify(initial), 'utf8');
      child.once('error', async error => {
        await writeFile(join(folder, 'status.json'), JSON.stringify({...initial, status: 'failed', error: error.message}));
      });
      child.once('close', () => { jobs.delete(filename); log.end(); });
      return initial;
    } finally {
      await handle.close();
      await unlink(lock);
    }
  }

  async function stop(filename) {
    const state = await read(filename);
    if (!['running', 'stopping'].includes(state?.status)) return state;
    await writeFile(join(outputDir(filename), 'stop.request'), 'stop', 'utf8');
    // Cooperative stop also works when the task was launched from the other port.
    return {...state, status: 'stopping', message: '正在停止，将在当前计算块结束后退出'};
  }

  async function renderFrame(filename, request) {
    const state = await read(filename);
    if (state?.status !== 'complete') throw new Error('模型尚未训练完成');
    if (!Number.isFinite(Number(request.timeSeconds ?? 0))) throw new Error('无效播放时间');
    if (request.camera && (!Array.isArray(request.camera) || request.camera.length !== 16 ||
        !request.camera.every(Number.isFinite))) throw new Error('相机需要 16 个有限数值');
    if (!renderer || renderer.filename !== filename) {
      closeRenderer();
      const child = spawn(python, ['-u', join(appDir, 'dynamic_preview.py'), '--model-dir', outputDir(filename)],
        {cwd: appDir, windowsHide: true, env: {...process.env, PYTHONUTF8: '1'}});
      const instance = {filename, child, pending: new Map(), stderr: ''};
      renderer = instance;
      child.stderr.on('data', data => { instance.stderr = (instance.stderr + data).slice(-12000); });
      createInterface({input: child.stdout}).on('line', line => {
        let result;
        try { result = JSON.parse(line); } catch { return; }
        const task = instance.pending.get(result.id);
        if (!task) return;
        instance.pending.delete(result.id);
        if (result.error) task.reject(new Error(result.error));
        else task.resolve({bytes: Buffer.from(result.png, 'base64'), renderMs: result.renderMs});
      });
      const failed = error => {
        for (const task of instance.pending.values()) task.reject(error);
        instance.pending.clear();
        if (renderer === instance) renderer = null;
      };
      child.once('error', failed);
      child.once('close', code => failed(new Error(instance.stderr || `预览进程退出：${code}`)));
    }
    const instance = renderer;
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      instance.pending.set(id, {resolve, reject});
      instance.child.stdin.write(JSON.stringify({...request, id}) + '\n', error => {
        if (error) { instance.pending.delete(id); reject(error); }
      });
    });
  }
  process.once('exit', closeRenderer);
  return {read, start, stop, renderFrame};
}
