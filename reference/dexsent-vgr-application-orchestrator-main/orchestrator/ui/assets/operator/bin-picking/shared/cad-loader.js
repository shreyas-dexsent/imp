import * as THREE from 'three';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const visualMaterial = new THREE.MeshStandardMaterial({
  color: 0xd8d2c6,
  metalness: 0.08,
  roughness: 0.55,
});

const OCCT_EXTENSIONS = new Set(['step', 'stp', 'iges', 'igs', 'brep']);

let occt = null;

export async function loadCadLikeFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  const stlLoader = new STLLoader();
  const daeLoader = new ColladaLoader();
  const objLoader = new OBJLoader();

  if (ext === 'stl') {
    const geometry = stlLoader.parse(await file.arrayBuffer());
    geometry.computeVertexNormals();
    return new THREE.Mesh(geometry, visualMaterial.clone());
  }
  if (ext === 'obj') {
    const object = objLoader.parse(await file.text());
    object.traverse((child) => {
      if (child.isMesh) normalizeVisualMaterial(child);
    });
    return object;
  }
  if (ext === 'dae') {
    const url = URL.createObjectURL(file);
    try {
      const collada = await daeLoader.loadAsync(url);
      collada.scene.rotation.set(0, 0, 0);
      collada.scene.traverse((child) => {
        if (child.isMesh) normalizeVisualMaterial(child);
      });
      return collada.scene;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  if (OCCT_EXTENSIONS.has(ext)) {
    return loadOcctFile(file, ext);
  }
  throw new Error(`Unsupported file extension .${ext}`);
}

export function normalizeObjectToOrigin(object) {
  object.traverse((child) => {
    if (!child.isMesh) return;
    child.castShadow = true;
    child.receiveShadow = true;
  });
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxSize = Math.max(size.x, size.y, size.z);
  if (maxSize > 20) object.scale.multiplyScalar(1 / 1000);
  object.position.sub(center.multiply(object.scale));
  object.updateMatrixWorld(true);
}

async function loadOcctFile(file, ext) {
  const runtime = await getOcct();
  const content = new Uint8Array(await file.arrayBuffer());
  const params = {
    linearUnit: 'meter',
    linearDeflectionType: 'bounding_box_ratio',
    linearDeflection: 0.001,
    angularDeflection: 0.5,
  };
  let result;
  if (ext === 'step' || ext === 'stp') result = runtime.ReadStepFile(content, params);
  if (ext === 'iges' || ext === 'igs') result = runtime.ReadIgesFile(content, params);
  if (ext === 'brep') result = runtime.ReadBrepFile(content, params);
  if (!result?.success) throw new Error(result?.error ?? 'CAD import failed');
  return buildOcctObject(result);
}

async function getOcct() {
  if (occt) return occt;
  // OpenCascade is heavy, so keep it out of the initial app bundle and load it
  // only when a precise CAD format is imported.
  if (!window.occtimportjs) await loadScript('/ui/assets/operator/bin-picking/vendor/occt/occt-import-js.js');
  occt = await window.occtimportjs({
    locateFile: (path) => `/ui/assets/operator/bin-picking/vendor/occt/${path}`,
  });
  return occt;
}

function buildOcctObject(result) {
  const root = new THREE.Group();
  root.name = result.root?.name ?? 'cad_import';
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0x111111,
    transparent: true,
    opacity: 0.45,
  });

  for (const resultMesh of result.meshes ?? []) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(resultMesh.attributes.position.array, 3));
    if (resultMesh.attributes.normal) {
      geometry.setAttribute('normal', new THREE.Float32BufferAttribute(resultMesh.attributes.normal.array, 3));
    } else {
      geometry.computeVertexNormals();
    }
    geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(resultMesh.index.array), 1));
    const color = resultMesh.color ? new THREE.Color(...resultMesh.color) : new THREE.Color(0xd8d2c6);
    const material = new THREE.MeshStandardMaterial({ color, metalness: 0.02, roughness: 0.55 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = resultMesh.name ?? 'cad_mesh';
    root.add(mesh);
    root.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 35), edgeMaterial));
  }
  return root;
}

function normalizeVisualMaterial(mesh) {
  if (!mesh.material) {
    mesh.material = visualMaterial.clone();
    return;
  }
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (const material of materials) {
    material.side = THREE.FrontSide;
    material.needsUpdate = true;
    if (material.map) material.map.colorSpace = THREE.SRGBColorSpace;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      existing.addEventListener('load', resolve, { once: true });
      if (window.occtimportjs) resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}
