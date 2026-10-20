import * as THREE from 'three';
import {PLYLoader} from 'three/addons/loaders/PLYLoader.js';

export class PointCloudViewer {
  constructor(host, status) {
    this.host = host;
    this.status = status;
    this.keys = new Set();
    this.drag = null;
    this.yaw = 0;
    this.pitch = 0;
    this.target = new THREE.Vector3(0, 0, -5);
    this.orbitRadius = 5;
    this.lastTime = performance.now();
    this.renderer = new THREE.WebGLRenderer({antialias: false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.setClearColor(0x070908);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.host.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(65, 1, 0.005, 10000);
    this.camera.rotation.order = 'YXZ';
    this.root = new THREE.Group();
    this.root.rotation.x = Math.PI;
    this.scene.add(this.root);
    this.bindControls();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
    this.renderer.setAnimationLoop(now => this.render(now));
  }

  bindControls() {
    this.host.addEventListener('contextmenu', event => event.preventDefault());
    this.host.addEventListener('pointerdown', event => {
      this.drag = {x: event.clientX, y: event.clientY, button: event.button};
      this.host.setPointerCapture(event.pointerId);
      this.host.focus();
    });
    this.host.addEventListener('pointerup', () => { this.drag = null; });
    this.host.addEventListener('pointercancel', () => { this.drag = null; });
    this.host.addEventListener('pointermove', event => {
      if (!this.drag) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      if (this.drag.button === 2) {
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
        const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
        const offset = right.multiplyScalar(-dx * this.orbitRadius * 0.001)
          .add(up.multiplyScalar(dy * this.orbitRadius * 0.001));
        this.target.add(offset);
        this.camera.position.add(offset);
      } else {
        this.yaw -= dx * 0.004;
        this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + dy * 0.004));
        this.updateOrbit();
      }
    });
    this.host.addEventListener('wheel', event => {
      event.preventDefault();
      this.orbitRadius = Math.max(0.05, this.orbitRadius * Math.exp(event.deltaY * 0.001));
      this.updateOrbit();
    }, {passive: false});
    this.host.addEventListener('keydown', event => {
      if ('wasdqeWASDQE'.includes(event.key) || event.key === 'Shift') {
        this.keys.add(event.key);
        event.preventDefault();
      }
    });
    window.addEventListener('keyup', event => this.keys.delete(event.key));
    window.addEventListener('blur', () => this.keys.clear());
  }

  resize() {
    const width = Math.max(1, this.host.clientWidth);
    const height = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  async load(url, metadata) {
    this.status.textContent = '正在加载点云…';
    const geometry = await new PLYLoader().loadAsync(url);
    if (this.points) {
      this.root.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
    }
    const pointSize = Math.max(0.002, Number(metadata.depthNearP02 || 1) * 0.0022);
    const material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      vertexColors: geometry.hasAttribute('color')
    });
    this.points = new THREE.Points(geometry, material);
    this.root.add(this.points);
    const fy = Number(metadata.focalY);
    if (fy > 0) {
      this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(metadata.height / (2 * fy)));
      this.camera.updateProjectionMatrix();
    }
    this.camera.position.set(0, 0, 0);
    this.yaw = 0;
    this.pitch = 0;
    this.orbitRadius = Math.sqrt(Number(metadata.depthNearP02) * Number(metadata.depthFarP98));
    this.target.set(0, 0, -this.orbitRadius);
    this.updateOrbit();
    this.status.textContent = `${Number(metadata.pointCount).toLocaleString()} 点 · ${metadata.elapsedSeconds}s`;
  }

  updateOrbit() {
    const cosPitch = Math.cos(this.pitch);
    this.camera.position.set(
      this.target.x + this.orbitRadius * Math.sin(this.yaw) * cosPitch,
      this.target.y + this.orbitRadius * Math.sin(this.pitch),
      this.target.z + this.orbitRadius * Math.cos(this.yaw) * cosPitch
    );
    this.camera.lookAt(this.target);
  }

  render(now) {
    const delta = Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;
    if (document.activeElement === this.host) {
      const speed = delta * (this.keys.has('Shift') ? 3 : 1);
      const offset = new THREE.Vector3();
      const forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      if (this.keys.has('w') || this.keys.has('W')) offset.addScaledVector(forward, speed);
      if (this.keys.has('s') || this.keys.has('S')) offset.addScaledVector(forward, -speed);
      if (this.keys.has('a') || this.keys.has('A')) offset.addScaledVector(right, -speed);
      if (this.keys.has('d') || this.keys.has('D')) offset.addScaledVector(right, speed);
      if (this.keys.has('q') || this.keys.has('Q')) offset.y -= speed;
      if (this.keys.has('e') || this.keys.has('E')) offset.y += speed;
      this.camera.position.add(offset);
      this.target.add(offset);
    }
    this.renderer.render(this.scene, this.camera);
  }
}
