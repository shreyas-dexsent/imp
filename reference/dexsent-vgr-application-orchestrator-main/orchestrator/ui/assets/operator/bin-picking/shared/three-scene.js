import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createSceneViewport(container, { background = 0x101214 } = {}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(background);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(2.5, -3.2, 2.2);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = false;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0.45);

  const grid = new THREE.GridHelper(4, 40, 0x40505a, 0x263039);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x20242a, 1.4));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
  keyLight.position.set(3, -4, 5);
  scene.add(keyLight);

  let animationId = 0;
  const resize = () => {
    const rect = container.getBoundingClientRect();
    camera.aspect = rect.width / Math.max(rect.height, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(rect.width, rect.height, false);
  };
  const animate = () => {
    animationId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  const frameObject = (object) => {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 0.5);
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(radius * 1.4, -radius * 1.8, radius * 1.2));
    camera.near = Math.max(radius / 1000, 0.001);
    camera.far = Math.max(radius * 20, 20);
    camera.updateProjectionMatrix();
    controls.update();
  };
  const dispose = () => {
    cancelAnimationFrame(animationId);
    window.removeEventListener('resize', resize);
    ro.disconnect();
    renderer.dispose();
    renderer.domElement.remove();
  };

  resize();
  window.addEventListener('resize', resize);
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  animate();

  return { scene, camera, renderer, controls, grid, resize, frameObject, dispose };
}
