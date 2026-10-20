// Keep source PLY row numbers authoritative: remap before Spark decoding.
export function parseGaussianPly(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const decoder = new TextDecoder();
  let lineStart = 0, headerEnd = -1, vertexCount = 0, inVertices = false;
  let littleEndian = false;
  const fields = [];
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] !== 10) continue;
    const line = decoder.decode(bytes.subarray(lineStart, i)).trim();
    lineStart = i + 1;
    if (line === 'end_header') { headerEnd = lineStart; break; }
    if (line.startsWith('format ')) littleEndian = line === 'format binary_little_endian 1.0';
    if (line.startsWith('element ')) {
      const parts = line.split(/\s+/);
      inVertices = parts[1] === 'vertex';
      if (inVertices) vertexCount = Number(parts[2]);
    } else if (inVertices && line.startsWith('property ')) {
      const parts = line.split(/\s+/);
      if (parts[1] !== 'float') throw new Error('当前轨迹仅支持 SHARP 的 float32 PLY 顶点');
      fields.push(parts[2]);
    }
  }
  const stride = fields.length * 4;
  if (headerEnd < 0 || !littleEndian || !vertexCount || !stride || headerEnd + vertexCount * stride > bytes.length) {
    throw new Error('无法读取 SHARP PLY 顶点布局');
  }
  const offsets = Object.fromEntries(fields.map((name, index) => [name, index * 4]));
  for (const name of ['x', 'y', 'z', 'opacity', 'rot_0']) {
    if (offsets[name] === undefined) throw new Error(`PLY 缺少属性：${name}`);
  }
  return {bytes, headerEnd, vertexCount, stride, offsets};
}

export function remapGaussianPly(buffer, cellIndices, layerCount) {
  const {bytes, headerEnd, vertexCount, stride, offsets} = parseGaussianPly(buffer);
  const cellCount = cellIndices.length;
  if (vertexCount !== layerCount * cellCount) throw new Error('PLY 数量与轨迹网格不一致');
  // Copying the whole file also preserves its header and all metadata elements.
  const output = new Uint8Array(bytes);
  const view = new DataView(output.buffer);
  for (let cell = 0; cell < cellCount; cell++) {
    const source = cellIndices[cell];
    if (!Number.isInteger(source) || source < -1 || source >= cellCount) throw new Error('轨迹索引超出原帧范围');
    for (let layer = 0; layer < layerCount; layer++) {
      const destinationStart = headerEnd + (layer * cellCount + cell) * stride;
      if (source >= 0) {
        const sourceStart = headerEnd + (layer * cellCount + source) * stride;
        output.set(bytes.subarray(sourceStart, sourceStart + stride), destinationStart);
      } else {
        // A finite, identity-oriented placeholder with exactly zero opacity.
        output.fill(0, destinationStart, destinationStart + stride);
        view.setFloat32(destinationStart + offsets.z, 1, true);
        view.setFloat32(destinationStart + offsets.rot_0, 1, true);
        view.setFloat32(destinationStart + offsets.opacity, -Infinity, true);
      }
    }
  }
  return output;
}
