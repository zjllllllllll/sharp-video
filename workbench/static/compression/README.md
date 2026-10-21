# Compressed Gaussian playback

Decoder, camera math, WebGPU/WebGL2 renderer and sorting are adapted from
[adamraudonis/splats4D](https://github.com/adamraudonis/splats4D), snapshot
`bf7c41ac2ef59a311899887bd7e032239b819829`, under `LICENSE.splats4D`.
`fzstd.js` is fzstd 0.1.1 under `LICENSE.fzstd`.

Local changes:

- TypeScript types are erased; module imports are relative for direct serving.
- fzstd's two numeric-sentinel comparisons use strict equality. Loose comparison
  coerces large Uint8Array output buffers to strings and crashes GOP decoding.
- The worker prepares every frame before playback, retaining positions and alpha
  for camera sorting. The main thread uploads each complete frame to its own GPU
  texture and index buffer. Decoded GOP caches are released after preparation.
- Playing selects resident textures; it does not decode or rebuild covariance.
  Camera changes use the original counting sort on retained positions.
- The frame clock advances one source frame at a time, never selecting a distant
  frame merely to catch up to wall time. Reported playback FPS counts source
  frame presentations rather than redraws of the same frame.

The current 100-frame clip prepares approximately 4.25 GB of GPU textures and
indices. Preparation is visible in the workbench progress bar. No Gaussian
subsampling, temporal denoising or extra representation quantization is added.
Packing PLY to the encoder's required 32-byte `.splat` input is itself lossy;
encoder error bounds are relative to that input, not the original FP32 PLY.

`tests/compression_gpu_bench.mjs` runs the production worker and renderer using
native Dawn WebGPU, without opening a webpage or inspecting any rendered image.
Its optional Node `webgpu` package is for performance checks, not browser playback.
