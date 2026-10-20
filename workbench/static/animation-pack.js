/* Spark 2.2 compatible packing, usable in a Worker without THREE/import maps.
 * Adapted from Spark (MIT), Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions: The above copyright notice and this
 * permission notice shall be included in all copies or substantial portions of
 * the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
 * EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const f16 = typeof Float16Array === 'function' ? new Float16Array(1) : null;
const u16 = f16 ? new Uint16Array(f16.buffer) : null;
function half(value) {
  if (f16) { f16[0] = value; return u16[0]; }
  f32[0] = value;
  const bits = u32[0], sign = (bits >>> 31) << 15;
  const exponent = (bits >>> 23) & 255, fraction = bits & 8388607;
  if (exponent === 255) return sign | (fraction ? 32767 : 31744);
  const next = exponent - 112;
  if (next >= 31) return sign | 31744;
  if (next <= 0) return next < -10 ? sign : sign | ((fraction | 8388608) >> (14 - next));
  return sign | (next << 10) | (fraction >> 13);
}
const byte = value => Math.max(0, Math.min(255, Math.round(value * 255)));

function quaternion(x, y, z, w) {
  const norm = Math.sqrt(x * x + y * y + z * z + w * w);
  if (norm === 0) { x = y = z = 0; w = 1; }
  else { const inverse = 1 / norm; x *= inverse; y *= inverse; z *= inverse; w *= inverse; }
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const angle = 2 * Math.acos(w);
  const xyz = Math.sqrt(x * x + y * y + z * z);
  if (xyz < 1e-6) { x = 1; y = z = 0; }
  else { const inverse = 1 / xyz; x *= inverse; y *= inverse; z *= inverse; }
  const sum = Math.abs(x) + Math.abs(y) + Math.abs(z);
  let px = x / sum, py = y / sum;
  if (z < 0) {
    const previous = px;
    px = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    py = (1 - Math.abs(previous)) * (py >= 0 ? 1 : -1);
  }
  return Math.round(angle * (255 / Math.PI)) << 16
    | Math.round((py * .5 + .5) * 255) << 8 | Math.round((px * .5 + .5) * 255);
}

export function packAnimationFrame(parameters) {
  const count = parameters.length / 14;
  const height = Math.max(1, Math.min(2048, Math.ceil(count / 2048)));
  const capacity = 2048 * height * Math.ceil(count / (2048 * height));
  const packed = new Uint32Array(capacity * 4);
  const scale = log => {
    // Preserve Spark's exact operation order and existing quantization.
    const value = Math.exp(log);
    return value < Math.exp(-30) ? 0 : Math.min(255, Math.max(1, Math.round((Math.log(value) + 12) * (254 / 21)) + 1));
  };
  for (let id = 0; id < count; id++) {
    const p = id * 14, i = id * 4;
    const q = quaternion(parameters[p + 11], parameters[p + 12], parameters[p + 13], parameters[p + 10]);
    const r = byte(.5 + .28209479177387814 * parameters[p + 3]);
    const g = byte(.5 + .28209479177387814 * parameters[p + 4]);
    const b = byte(.5 + .28209479177387814 * parameters[p + 5]);
    const a = byte(1 / (1 + Math.exp(-parameters[p + 6])));
    packed[i] = r | g << 8 | b << 16 | a << 24;
    packed[i + 1] = half(parameters[p]) | half(parameters[p + 1]) << 16;
    packed[i + 2] = half(parameters[p + 2]) | (q & 255) << 16 | ((q >>> 8) & 255) << 24;
    packed[i + 3] = scale(parameters[p + 7]) | scale(parameters[p + 8]) << 8
      | scale(parameters[p + 9]) << 16 | ((q >>> 16) & 255) << 24;
  }
  return packed;
}
