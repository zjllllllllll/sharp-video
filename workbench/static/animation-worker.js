import {readAnimationHeader, readAnimationFrame} from './animation-file.js';
import {packAnimationFrame} from './animation-pack.js';

let acknowledge;
self.onmessage = event => {
  if (event.data.type === 'next') { acknowledge?.(); return; }
  if (event.data.type !== 'load') return;
  void prepare(event.data.url).catch(error => self.postMessage({type: 'error', message: error.message}));
};

async function prepare(url) {
  const header = await readAnimationHeader(url);
  self.postMessage({type: 'header', header});
  for (let index = 0; index < header.frameCount; index++) {
    const parameters = await readAnimationFrame(url, header, index);
    const packed = packAnimationFrame(parameters);
    // Transfer ownership, and wait for the GPU upload before preparing more.
    const uploaded = new Promise(resolve => { acknowledge = resolve; });
    self.postMessage({type: 'frame', index, packed}, [packed.buffer]);
    await uploaded;
  }
  self.postMessage({type: 'complete'});
}
