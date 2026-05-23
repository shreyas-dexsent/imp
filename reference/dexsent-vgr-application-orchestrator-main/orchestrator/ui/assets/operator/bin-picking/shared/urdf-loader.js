import * as THREE from 'three';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

/**
 * Minimal URDF loader sufficient for the gripper studio:
 *
 * - Parses <link>, <joint>, <visual>, <collision>, <origin>, <geometry>.
 * - Mesh refs use the <mesh filename="..."/> attribute, resolved relative to
 *   the URDF URL. Both .dae and .stl/.obj are supported.
 * - URDF rpy is roll-pitch-yaw such that R = Rz(yaw) · Ry(pitch) · Rx(roll),
 *   which is THREE.Euler order 'ZYX' with the per-axis angles (roll, pitch, yaw).
 * - Joints are placed at home position (joint angle = 0). Joint axes/limits
 *   are stored on each joint group as user-data so a future joint slider can
 *   actuate them; mimic joints likewise carry a reference for later use.
 *
 * Returned root has tree:
 *   <robot Group>
 *     <link Group root_link>
 *       [visual meshes]
 *       <joint Group joint_name>
 *         <link Group child_link>
 *           ...
 */

const DEFAULT_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0xd8d2c6, metalness: 0.08, roughness: 0.55,
});

const textCache = new Map();
const meshCache = new Map();

export async function loadUrdf(urdfUrl, { loadCollision = false } = {}) {
  const absoluteUrdfUrl = new URL(urdfUrl, window.location.href).href;
  let text = textCache.get(absoluteUrdfUrl);
  if (!text) {
    text = await (await fetch(absoluteUrdfUrl, { cache: 'no-store' })).text();
    textCache.set(absoluteUrdfUrl, text);
  }
  const xml = new DOMParser().parseFromString(text, 'application/xml');
  const parseError = xml.querySelector('parsererror');
  if (parseError) throw new Error(`URDF parse error: ${parseError.textContent}`);

  const robot = xml.querySelector('robot');
  if (!robot) throw new Error('No <robot> element in URDF');

  const links = new Map();
  for (const link of robot.querySelectorAll(':scope > link')) {
    links.set(link.getAttribute('name'), link);
  }

  const childToParent = new Map();
  const childrenOf = new Map();
  for (const j of robot.querySelectorAll(':scope > joint')) {
    const parent = j.querySelector(':scope > parent')?.getAttribute('link');
    const child = j.querySelector(':scope > child')?.getAttribute('link');
    if (!parent || !child) continue;
    childToParent.set(child, j);
    if (!childrenOf.has(parent)) childrenOf.set(parent, []);
    childrenOf.get(parent).push({ child, joint: j });
  }

  const root = [...links.keys()].find((name) => !childToParent.has(name));
  if (!root) throw new Error('Could not determine root link');

  const baseDir = absoluteUrdfUrl.replace(/[^/]+$/, '');
  const linkGroups = new Map(); // linkName -> Group, exposed for TF lookup later
  const jointMap = new Map();   // jointName -> joint descriptor (see below)

  const robotRoot = new THREE.Group();
  robotRoot.name = robot.getAttribute('name') ?? 'robot';
  robotRoot.userData.urdf = { rootLink: root, links: linkGroups, joints: jointMap };

  await buildLink(root, robotRoot, { baseDir, links, childrenOf, linkGroups, jointMap, loadCollision });

  // Master actuators = movable joints without a <mimic> dependency
  const movable = new Set(['revolute', 'continuous', 'prismatic']);
  robotRoot.userData.urdf.actuators = [...jointMap.values()].filter(
    (j) => movable.has(j.type) && !j.mimic,
  );
  robotRoot.userData.urdf.setJointValue = (name, value) => setJointValue(jointMap, name, value);

  return robotRoot;
}

function setJointValue(jointMap, name, value) {
  const j = jointMap.get(name);
  if (!j) return;
  applyJointValue(j, value);
  for (const other of jointMap.values()) {
    if (other.mimic?.joint === name) {
      applyJointValue(other, value * other.mimic.multiplier + other.mimic.offset);
    }
  }
}

function applyJointValue(j, value) {
  j.value = value;
  if (j.type === 'revolute' || j.type === 'continuous') {
    const q = new THREE.Quaternion().setFromAxisAngle(j.axis, value);
    j.group.quaternion.copy(j.originQuat).multiply(q);
    j.group.position.copy(j.originPos);
  } else if (j.type === 'prismatic') {
    const offset = j.axis.clone().multiplyScalar(value).applyQuaternion(j.originQuat);
    j.group.position.copy(j.originPos).add(offset);
    j.group.quaternion.copy(j.originQuat);
  }
}

async function buildLink(name, parent, ctx) {
  const linkGroup = new THREE.Group();
  linkGroup.name = `link:${name}`;
  linkGroup.userData.urdfLink = name;
  parent.add(linkGroup);
  ctx.linkGroups.set(name, linkGroup);

  const link = ctx.links.get(name);
  if (link) {
    for (const visual of link.querySelectorAll(':scope > visual')) {
      const node = await buildGeometry(visual, ctx.baseDir, /*opacity*/ 1.0);
      if (node) {
        applyOrigin(node, visual.querySelector(':scope > origin'));
        linkGroup.add(node);
      }
    }
    if (ctx.loadCollision) {
      for (const coll of link.querySelectorAll(':scope > collision')) {
        const node = await buildGeometry(coll, ctx.baseDir, /*opacity*/ 0.35);
        if (node) {
          applyOrigin(node, coll.querySelector(':scope > origin'));
          node.userData.collision = true;
          linkGroup.add(node);
        }
      }
    }
  }

  for (const { child, joint } of ctx.childrenOf.get(name) ?? []) {
    const jointGroup = new THREE.Group();
    const jointName = joint.getAttribute('name');
    jointGroup.name = `joint:${jointName}`;
    applyOrigin(jointGroup, joint.querySelector(':scope > origin'));

    const limit = joint.querySelector(':scope > limit');
    const mimicEl = joint.querySelector(':scope > mimic');
    const axis = parseTriple(joint.querySelector(':scope > axis')?.getAttribute('xyz') ?? '1 0 0', [1, 0, 0]);
    const descriptor = {
      name: jointName,
      type: joint.getAttribute('type'),
      axis: new THREE.Vector3(axis[0], axis[1], axis[2]).normalize(),
      lower: limit ? Number(limit.getAttribute('lower') ?? 0) : 0,
      upper: limit ? Number(limit.getAttribute('upper') ?? 0) : 0,
      mimic: mimicEl ? {
        joint: mimicEl.getAttribute('joint'),
        multiplier: Number(mimicEl.getAttribute('multiplier') ?? 1),
        offset: Number(mimicEl.getAttribute('offset') ?? 0),
      } : null,
      parentLink: name,
      childLink: child,
      group: jointGroup,
      originPos: jointGroup.position.clone(),
      originQuat: jointGroup.quaternion.clone(),
      value: 0,
    };
    ctx.jointMap.set(jointName, descriptor);
    jointGroup.userData.urdfJoint = descriptor;

    linkGroup.add(jointGroup);
    await buildLink(child, jointGroup, ctx);
  }
}

async function buildGeometry(elementWithGeometry, baseDir, opacity) {
  const geom = elementWithGeometry.querySelector(':scope > geometry');
  if (!geom) return null;
  const meshNode = geom.querySelector(':scope > mesh');
  if (meshNode) {
    const file = meshNode.getAttribute('filename');
    if (!file) return null;
    const url = resolveMeshUrl(file, baseDir);
    try {
      const mesh = await loadMeshUrl(url);
      if (opacity < 1) {
        mesh.traverse((c) => {
          if (c.isMesh && c.material) {
            c.material = c.material.clone();
            c.material.transparent = true;
            c.material.opacity = opacity;
          }
        });
      }
      const scale = parseTriple(meshNode.getAttribute('scale') ?? '1 1 1', [1, 1, 1]);
      mesh.scale.set(scale[0], scale[1], scale[2]);
      return mesh;
    } catch (err) {
      console.warn(`URDF mesh load failed for ${url}:`, err);
      return null;
    }
  }
  const box = geom.querySelector(':scope > box');
  if (box) {
    const size = parseTriple(box.getAttribute('size') ?? '0.01 0.01 0.01', [0.01, 0.01, 0.01]);
    return new THREE.Mesh(
      new THREE.BoxGeometry(size[0], size[1], size[2]),
      DEFAULT_MATERIAL.clone(),
    );
  }
  const cyl = geom.querySelector(':scope > cylinder');
  if (cyl) {
    const r = Number(cyl.getAttribute('radius')) || 0.01;
    const l = Number(cyl.getAttribute('length')) || 0.02;
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, l, 32), DEFAULT_MATERIAL.clone());
    m.rotation.x = Math.PI / 2; // URDF cylinders are along Z
    return m;
  }
  const sphere = geom.querySelector(':scope > sphere');
  if (sphere) {
    const r = Number(sphere.getAttribute('radius')) || 0.01;
    return new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), DEFAULT_MATERIAL.clone());
  }
  return null;
}

async function loadMeshUrl(url) {
  const cached = meshCache.get(url);
  if (cached) return cached.clone(true);
  const ext = url.split('?')[0].split('.').pop().toLowerCase();
  let loaded = null;
  if (ext === 'stl') {
    const buf = await (await fetch(url, { cache: 'no-store' })).arrayBuffer();
    const geometry = new STLLoader().parse(buf);
    geometry.computeVertexNormals();
    loaded = new THREE.Mesh(geometry, DEFAULT_MATERIAL.clone());
  } else if (ext === 'obj') {
    const text = await (await fetch(url, { cache: 'no-store' })).text();
    const obj = new OBJLoader().parse(text);
    obj.traverse((c) => { if (c.isMesh) c.material = DEFAULT_MATERIAL.clone(); });
    loaded = obj;
  } else if (ext === 'dae') {
    // We need the DAE's <up_axis> to normalize the mesh to URDF Z_UP. ColladaLoader
    // by default rotates Z_UP → Y_UP (Three.js canonical) and leaves Y_UP alone,
    // so we have to undo that and apply our own Y→Z rotation when needed.
    const text = await (await fetch(url, { cache: 'no-store' })).text();
    const upAxis = (text.match(/<up_axis>\s*(\w+)\s*<\/up_axis>/i)?.[1] ?? 'Y_UP').toUpperCase();

    const collada = await new ColladaLoader().parse(text, url.replace(/[^/]+$/, ''));
    loaded = collada.scene;
    // Strip ColladaLoader's auto-rotation — we handle up_axis ourselves below.
    loaded.rotation.set(0, 0, 0);
    loaded.traverse((c) => {
      if (c.isMesh && (!c.material || !c.material.isMeshStandardMaterial)) {
        c.material = DEFAULT_MATERIAL.clone();
      }
    });
    if (upAxis === 'Y_UP') {
      // Y_UP → Z_UP correction. A parent-group wrapper would be overwritten by
      // applyOrigin() (which does obj.position.set / obj.rotation.set), so we
      // bake the full world matrix of every mesh into its geometry instead.
      // Step 1: compute world matrices relative to `loaded` (which has no transform).
      loaded.updateWorldMatrix(true, true);
      const yupToZup = new THREE.Matrix4().makeRotationX(Math.PI / 2);
      const flatMeshes = [];
      loaded.traverse((c) => { if (c.isMesh) flatMeshes.push(c); });
      for (const mesh of flatMeshes) {
        // Bake: world matrix of this mesh (relative to loaded root) + Y→Z correction.
        const world = mesh.matrixWorld.clone().premultiply(yupToZup);
        mesh.geometry = mesh.geometry.clone();
        mesh.geometry.applyMatrix4(world);
        // The geometry now contains all transforms; zero out the node itself.
        mesh.position.set(0, 0, 0);
        mesh.quaternion.identity();
        mesh.scale.set(1, 1, 1);
        mesh.updateMatrix();
        // Re-parent to loaded root with add() so no extra transform is applied.
        loaded.add(mesh);
      }
    }
  } else {
    throw new Error(`Unsupported mesh format: .${ext}`);
  }
  meshCache.set(url, loaded);
  return loaded.clone(true);
}

function applyOrigin(obj, originEl) {
  if (!originEl) return;
  const xyz = parseTriple(originEl.getAttribute('xyz') ?? '0 0 0', [0, 0, 0]);
  const rpy = parseTriple(originEl.getAttribute('rpy') ?? '0 0 0', [0, 0, 0]);
  obj.position.set(xyz[0], xyz[1], xyz[2]);
  // URDF <origin rpy="roll pitch yaw"> means R = Rz(yaw) * Ry(pitch) * Rx(roll).
  // In three.js that is Euler order 'ZYX' with the per-axis angles (roll, pitch, yaw):
  //   Euler(roll, pitch, yaw, 'ZYX')  ->  Rz(yaw) * Ry(pitch) * Rx(roll).
  // (The old code used 'XYZ' which gives Rx*Ry*Rz — wrong whenever two or more of
  //  roll/pitch/yaw are nonzero, e.g. the franka finger origin "3.14159 0 -1.5708".)
  obj.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');
}

function parseTriple(str, fallback) {
  const parts = String(str).trim().split(/\s+/).map(Number);
  if (parts.length === 3 && parts.every((v) => Number.isFinite(v))) return parts;
  return fallback;
}

function resolveMeshUrl(filename, baseDir) {
  // URDF "package://pkg/path" → strip prefix and resolve as relative
  if (filename.startsWith('package://')) {
    return new URL(filename.replace(/^package:\/\/[^/]+\//, ''), baseDir).href;
  }
  return new URL(filename, baseDir).href;
}
