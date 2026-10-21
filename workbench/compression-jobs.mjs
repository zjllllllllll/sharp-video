import {createWriteStream} from 'node:fs';
import {mkdir, readFile, rename, stat, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {spawn} from 'node:child_process';

export function compressionJobs(appDir) {
  const starts = new Map();
  const idFor = name => Buffer.from(name, 'utf8').toString('base64url');
  const directory = name => join(appDir, 'runs', idFor(name), 'compressed');
  const alive = pid => { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } };
  async function saved(name) {
    try { return JSON.parse(await readFile(join(directory(name), 'compression.json'), 'utf8')); }
    catch (e) { if (e.code === 'ENOENT') return null; throw e; }
  }
  async function save(name, value) {
    const path = join(directory(name), 'compression.json');
    await writeFile(`${path}.server.tmp`, JSON.stringify(value), 'utf8');
    await rename(`${path}.server.tmp`, path);
  }
  async function read(name) {
    const state = await saved(name);
    if (!state) return null;
    if (state.status === 'running' && !starts.has(name) && (!state.pid || !alive(state.pid))) {
      state.status = 'interrupted';
    }
    if (state.file) state.url = `/results/${idFor(name)}/compressed/${state.file}`;
    return state;
  }
  async function launch(name) {
    const previous = await read(name);
    if (previous?.status === 'complete' || previous?.status === 'running') return previous;
    const runDir = join(appDir, 'runs', idFor(name));
    const source = JSON.parse(await readFile(join(runDir, 'gaussians', 'gaussian.json'), 'utf8'));
    if (source.status !== 'complete') throw new Error('请先完成第四步逐帧高斯');
    const python = process.env.SHARP_VIDEO_COMPRESSION_PYTHON || join(appDir, '..', '.envs', 'splats4d',
      ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
    await stat(python);
    await mkdir(directory(name), {recursive: true});
    const initial = {status: 'running', stage: 'preparing', progress: 0, currentFrame: 0,
      frameCount: source.frames.length, pid: process.pid, generatedAt: new Date().toISOString()};
    await save(name, initial);
    const log = createWriteStream(join(directory(name), 'worker.log'), {flags: 'a'});
    const child = spawn(python, ['-u', join(appDir, 'compression_worker.py'), '--run-dir', runDir],
      {cwd: appDir, windowsHide: true, env: {...process.env, PYTHONUTF8: '1'}});
    child.stdout.pipe(log, {end: false});
    child.stderr.pipe(log, {end: false});
    child.once('error', async error => {
      await save(name, {...initial, status: 'failed', error: error.message}); log.end();
    });
    child.once('close', async code => {
      const final = await saved(name);
      if (final?.status === 'running') await save(name, {...final, status: 'failed', error: `压缩进程退出（${code}）`});
      log.end();
    });
    return initial;
  }
  function start(name) {
    if (!starts.has(name)) starts.set(name, launch(name).finally(() => starts.delete(name)));
    return starts.get(name);
  }
  return {read, start};
}
