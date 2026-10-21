// Run the production browser worker in a Node worker thread, with file-backed
// range reads. This verifies file-only playback without opening a webpage.
import {parentPort, workerData} from 'node:worker_threads';
import {open} from 'node:fs/promises';
const file = await open(workerData.file, 'r');
const {size} = await file.stat();
globalThis.self = globalThis;
globalThis.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);
globalThis.fetch = async (_url, options) => {
  const match = /bytes=(\d+)-(\d+)/.exec(options.headers.Range);
  const start = Number(match[1]), end = Math.min(size - 1, Number(match[2]));
  const buffer = Buffer.alloc(end - start + 1);
  const {bytesRead} = await file.read(buffer, 0, buffer.length, start);
  if (bytesRead !== buffer.length) throw new Error('Incomplete range read');
  return new Response(buffer, {status: 206, headers: {'Content-Range': `bytes ${start}-${end}/${size}`, 'Content-Length': buffer.length}});
};
await import('../static/compression/worker.js');
parentPort.on('message', data => self.onmessage({data}));
parentPort.postMessage({type: 'node-ready'});
