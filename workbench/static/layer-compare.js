import {SplatMesh} from '@sparkjsdev/spark';
import {PlySequenceViewer} from './ply-viewer.js';

// SHARP writes layer,row,column order. Slice the original vertex bytes before
// Spark decodes them; retain every first-layer attribute and all PLY metadata.
export function firstLayerPly(buffer, count) {
  const bytes = new Uint8Array(buffer);
  const prefix = new TextDecoder().decode(bytes.subarray(0, 4096));
  const end = prefix.indexOf('end_header\n') + 'end_header\n'.length;
  const header = prefix.slice(0, end);
  const originalCount = Number(header.match(/element vertex (\d+)/)[1]);
  const properties = header.split('element vertex ')[1].split('\n').slice(1).join('\n').split('element ')[0];
  // This project's SHARP PLY uses scalar float32 vertex properties only.
  const stride = (properties.match(/^property float /gm) || []).length * 4;
  const newHeader = new TextEncoder().encode(header.replace(`element vertex ${originalCount}`, `element vertex ${count}`));
  const vertices = bytes.subarray(end, end + count * stride);
  const metadata = bytes.subarray(end + originalCount * stride);
  const output = new Uint8Array(newHeader.length + vertices.length + metadata.length);
  output.set(newHeader);
  output.set(vertices, newHeader.length);
  output.set(metadata, newHeader.length + vertices.length);
  return output;
}

async function startComparison() {
  const state = await (await fetch('/api/state')).json();
  const video = state.videos[0];
  const frame = video.gaussians.frames[0];
  const shape = video.gaussians.gaussianGrid.shape;
  const firstCount = shape[1] * shape[2];
  const response = await fetch(frame.fullUrl);
  if (!response.ok) throw new Error(`PLY 读取失败：${response.status}`);
  const original = await response.arrayBuffer();
  const first = firstLayerPly(original, firstCount);
  const left = new PlySequenceViewer(document.querySelector('#first-view'));
  const right = new PlySequenceViewer(document.querySelector('#full-view'));
  left.renderer.setAnimationLoop(null);
  right.renderer.setAnimationLoop(null);

  // Both panes literally use the same camera, with shared drag angles.
  right.camera = left.camera;
  for (const name of ['yaw', 'pitch']) Object.defineProperty(right, name, {
    get: () => left[name], set: value => { left[name] = value; }
  });
  left.resize();
  right.resize();
  const firstMesh = new SplatMesh({fileBytes: first, fileName: frame.full, lod: false});
  const fullMesh = new SplatMesh({fileBytes: original, fileName: frame.full, lod: false});
  await Promise.all([firstMesh.initialized, fullMesh.initialized]);
  left.root.add(firstMesh);
  right.root.add(fullMesh);
  left.renderer.setAnimationLoop(now => {
    const rightActive = document.activeElement === right.host;
    (rightActive ? right : left).tick(now);
    (rightActive ? left : right).tick(now);
  });
  document.querySelector('#first-count').textContent = `${firstMesh.numSplats.toLocaleString()} 个`;
  document.querySelector('#full-count').textContent = `${fullMesh.numSplats.toLocaleString()} 个`;
  document.querySelector('#compare-status').textContent = `${frame.full} · 第 ${frame.index + 1} 帧 · 同一相机 / 70° 视角 / 原始 PLY 属性`;
}

startComparison().catch(error => {
  document.querySelector('#compare-status').textContent = '载入失败';
  const output = document.querySelector('#compare-error');
  output.hidden = false;
  output.textContent = error.message;
});
