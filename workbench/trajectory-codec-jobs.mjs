import {createWriteStream} from 'node:fs';
import {mkdir, readFile, rename, stat, unlink, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {spawn} from 'node:child_process';

export function codecJobs(appDir) {
  const starts = new Map();
  const idFor = name => Buffer.from(name, 'utf8').toString('base64url');
  const directory = name => join(appDir, 'runs', idFor(name), 'trajectory-codec');
  const alive = pid => { if (!pid) return false; try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } };
  const exists = path => stat(path).then(() => true, () => false);
  async function saved(name) {
    try { return JSON.parse(await readFile(join(directory(name), 'state.json'), 'utf8')); }
    catch (e) { if (e.code === 'ENOENT') return null; throw e; }
  }
  async function save(name, value) {
    const path = join(directory(name), 'state.json');
    const temp = `${path}.server.tmp`;
    await writeFile(temp, JSON.stringify(value), 'utf8');
    await rename(temp, path);
  }
  async function read(name) {
    const state = await saved(name);
    if (!state) return null;
    const base = `/results/${idFor(name)}/trajectory-codec`;
    if (state.status === 'running' && !alive(state.pid) && !starts.has(name)) state.status = 'interrupted';
    const result = {...state, resumable: Boolean(state.exportComplete) || await exists(join(directory(name), 'checkpoint.pt'))};
    result.stopRequested = state.status === 'running' && await exists(join(directory(name), 'stop.request'));
    if (state.modelFile) result.modelUrl = `${base}/${state.modelFile}`;
    if (state.preview) result.preview = {...state.preview, file: {...state.preview.file, url: `${base}/animation.4dgs`}};
    if (state.report) result.reportUrl = `${base}/report.json`;
    try {
      result.lossHistory = (await readFile(join(directory(name), 'loss.jsonl'), 'utf8'))
        .split('\n').filter(Boolean).flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } });
    } catch (e) { if (e.code !== 'ENOENT') throw e; }
    return result;
  }
  async function launch(name) {
    const runDir = join(appDir, 'runs', idFor(name));
    const source = JSON.parse(await readFile(join(runDir, 'tracks-lifecycle', 'tracks.json'), 'utf8'));
    if (source.status !== 'complete' || source.format !== 'sharp.gaussian_tracks.v5') throw new Error('请先完成第五步高斯轨迹');
    const previous = await saved(name);
    if (previous?.status === 'complete' || (previous?.status === 'running' && alive(previous.pid))) return read(name);
    const out = directory(name);
    await mkdir(out, {recursive: true});
    await unlink(join(out, 'stop.request')).catch(e => { if (e.code !== 'ENOENT') throw e; });
    const initial = {...previous, status: 'running', stage: previous?.stage || 'preparing', error: null, pid: process.pid,
      frameCount: source.frameCount, generatedAt: previous?.generatedAt || new Date().toISOString()};
    await save(name, initial);
    const log = createWriteStream(join(out, 'worker.log'), {flags: 'a'});
    const python = process.env.SHARP_VIDEO_DYNAMIC_PYTHON || join(appDir, '..', '.envs', 'dynamic',
      ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
    const child = spawn(python, ['-u', join(appDir, 'trajectory_codec_worker.py'), 'run', '--run-dir', runDir],
      {cwd: appDir, windowsHide: true, env: {...process.env, PYTHONUTF8: '1'}});
    child.stdout.pipe(log, {end: false});
    child.stderr.pipe(log, {end: false});
    child.once('error', async error => { await save(name, {...await saved(name), status: 'failed', error: error.message}); log.end(); });
    child.once('close', async code => {
      const final = await saved(name);
      if (final?.status === 'running' && [child.pid, process.pid].includes(final.pid)) {
        await save(name, {...final, status: 'failed', error: `计算进程退出（${code}），可从检查点继续。`});
      }
      log.end();
    });
    // The supervisor writes its own PID before importing torch. Avoid racing its state updates.
    return {...initial, pid: child.pid};
  }
  function start(name) {
    if (!starts.has(name)) starts.set(name, launch(name).finally(() => starts.delete(name)));
    return starts.get(name);
  }
  async function stop(name) {
    const current = await read(name);
    if (current?.status === 'running') await writeFile(join(directory(name), 'stop.request'), 'stop\n');
    return read(name);
  }
  return {read, start, stop};
}
