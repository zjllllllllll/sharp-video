// Shared first-person controls for the PLY and compressed sequence viewers.
export function bindViewerControls(viewer) {
  const {host} = viewer;
  const listeners = [];
  const listen = (target, type, callback, options) => {
    target.addEventListener(type, callback, options);
    listeners.push(() => target.removeEventListener(type, callback, options));
  };
  listen(host, 'contextmenu', event => event.preventDefault());
  listen(host, 'pointerdown', event => {
    if (event.target.closest('.viewer-controls')) return;
    viewer.drag = {x: event.clientX, y: event.clientY, button: event.button};
    host.setPointerCapture(event.pointerId);
    host.focus();
  });
  listen(host, 'pointerup', () => { viewer.drag = null; });
  listen(host, 'pointercancel', () => { viewer.drag = null; });
  listen(host, 'pointermove', event => {
    if (!viewer.drag) return;
    const dx = event.clientX - viewer.drag.x;
    const dy = event.clientY - viewer.drag.y;
    viewer.drag.x = event.clientX;
    viewer.drag.y = event.clientY;
    if (viewer.drag.button === 2) {
      viewer.camera.translateX(-dx * 0.003);
      viewer.camera.translateY(dy * 0.003);
    } else {
      viewer.yaw -= dx * 0.003;
      viewer.pitch = Math.max(-1.55, Math.min(1.55, viewer.pitch - dy * 0.003));
      viewer.camera.rotation.set(viewer.pitch, viewer.yaw, 0);
    }
  });
  listen(host, 'wheel', event => {
    event.preventDefault();
    viewer.camera.translateZ(Math.max(-0.5, Math.min(0.5, event.deltaY * 0.002)));
  }, {passive: false});
  listen(host, 'keydown', event => {
    if ('wasdqeWASDQE'.includes(event.key) || event.key === 'Shift') {
      viewer.keys.add(event.key);
      event.preventDefault();
    }
  });
  listen(window, 'keyup', event => viewer.keys.delete(event.key));
  listen(window, 'blur', () => viewer.keys.clear());
  listen(host, 'blur', () => viewer.keys.clear());
  return () => { for (const remove of listeners) remove(); };
}

export function updateViewerMovement(viewer, delta) {
  if (document.activeElement !== viewer.host) return;
  const {camera, keys} = viewer;
  const speed = delta * (keys.has('Shift') ? 3 : 1);
  if (keys.has('w') || keys.has('W')) camera.translateZ(-speed);
  if (keys.has('s') || keys.has('S')) camera.translateZ(speed);
  if (keys.has('a') || keys.has('A')) camera.translateX(-speed);
  if (keys.has('d') || keys.has('D')) camera.translateX(speed);
  if (keys.has('q') || keys.has('Q')) camera.position.y -= speed;
  if (keys.has('e') || keys.has('E')) camera.position.y += speed;
}
