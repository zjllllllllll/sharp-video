// The only external resource needed by this reader is the .4dgs file itself.
export const ANIMATION_FORMAT = 'sharp.gaussian_tracks.v4';
export const LIFECYCLE_FORMAT = 'sharp.gaussian_tracks.v5';
export const ANIMATION_COLUMNS = 'x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3'.split(' ');

async function readRange(url, start, length, signal, fetcher) {
  const response = await fetcher(url, {signal, headers: {Range: `bytes=${start}-${start + length - 1}`}});
  if (response.status !== 206) throw new Error(`动画文件读取失败：${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== length) throw new Error('动画文件数据不完整');
  return bytes;
}

export async function readAnimationHeader(url, signal, fetcher = fetch) {
  const prefix = await readRange(url, 0, 16, signal, fetcher);
  const view = new DataView(prefix);
  if (new TextDecoder().decode(new Uint8Array(prefix, 0, 8)) !== 'SH4DGS01' || view.getUint32(12, true) !== 0) {
    throw new Error('不支持的高斯动画文件');
  }
  const length = view.getUint32(8, true);
  const bytes = await readRange(url, 16, length, signal, fetcher);
  const header = JSON.parse(new TextDecoder().decode(bytes));
  const parameterOffset = header.format === LIFECYCLE_FORMAT ? header.cellCount * 4 : 0;
  if (![ANIMATION_FORMAT, LIFECYCLE_FORMAT].includes(header.format) || header.dtype !== '<f4'
      || JSON.stringify(header.columns) !== JSON.stringify(ANIMATION_COLUMNS)
      || header.frameBytes !== parameterOffset + header.trackCount * 56
      || (header.format === LIFECYCLE_FORMAT && header.parameterOffset !== parameterOffset)) {
    throw new Error('不支持的高斯动画属性布局');
  }
  return {...header, dataOffset: Math.ceil((16 + length) / 8) * 8};
}

export async function readAnimationFrame(url, header, index, signal, fetcher = fetch) {
  if (!Number.isInteger(index) || index < 0 || index >= header.frameCount) throw new Error('动画帧号超出范围');
  // Number arithmetic deliberately supports offsets beyond 4 GiB.
  // Lifecycle IDs precede each v5 sample. Rendering only needs the live parameters.
  const start = header.dataOffset + index * header.frameBytes + (header.parameterOffset || 0);
  const bytes = await readRange(url, start, header.trackCount * 56, signal, fetcher);
  return new Float32Array(bytes);
}
