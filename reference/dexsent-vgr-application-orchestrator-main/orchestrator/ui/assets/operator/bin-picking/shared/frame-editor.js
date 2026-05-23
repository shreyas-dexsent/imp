import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const DEG = Math.PI / 180;

const DEFAULT_FRAMES = [
  { name: 'flange', color: 0xffa533 },
  { name: 'tcp',    color: 0x33d6ff },
];

/**
 * Multi-frame editor backed by TransformControls + numeric inputs.
 *
 * Configurable `frames` lets callers reuse the editor for the gripper's
 * (flange + TCP) pair, the object studio's single object frame, or any other
 * combination of named frames.
 *
 * DOM contract — inputs/buttons matched by data attribute:
 *   <input data-frame="<name>" data-axis="px|py|pz|rx|ry|rz" />
 *   <input type="radio" name="gs-edit-frame" value="<name>" />   (only when >1 frame)
 *   <input type="radio" name="gsm-edit-frame" value="<name>" />  (gripper studio)
 *   <button data-mode="translate|rotate" />
 *
 * Frame state: { position: [x,y,z] mm, rotation: [rx,ry,rz] degrees }
 *
 * Public API:
 *   setFrames(state)                  — { name: { position, rotation } }
 *   getFrames()                       — same shape
 *   setActiveFrame(name)
 *   setMode('translate'|'rotate')
 *   resizeForBounds(box)
 *   onChange(callback)
 *   dispose()
 */
export function createFrameEditor({
  scene, camera, renderer, controls, container,
  frames = DEFAULT_FRAMES,
}) {
  const indicators = {};
  const colors = {};
  const initialFrame = { position: [0, 0, 0], rotation: [0, 0, 0] };

  for (const f of frames) {
    colors[f.name] = f.color;
    indicators[f.name] = buildIndicator(0.05, f.color);
    scene.add(indicators[f.name]);
  }

  const gizmo = new TransformControls(camera, renderer.domElement);
  gizmo.setSize(0.9);
  gizmo.setSpace('local');
  scene.add(gizmo);
  gizmo.addEventListener('dragging-changed', (e) => {
    if (controls) controls.enabled = !e.value;
  });

  const state = {
    activeFrame: frames[0].name,
    mode: 'translate',
    frames: Object.fromEntries(frames.map((f) => [f.name, structuredClone(initialFrame)])),
  };

  const listeners = new Set();
  const emitChange = () => { for (const fn of listeners) fn(state.frames); };

  // Numeric inputs
  for (const input of container.querySelectorAll('input[data-frame][data-axis]')) {
    input.addEventListener('input', () => {
      const name = input.dataset.frame;
      const f = state.frames[name];
      if (!f) return;
      const v = Number(input.value) || 0;
      ({ px: () => f.position[0] = v, py: () => f.position[1] = v, pz: () => f.position[2] = v,
         rx: () => f.rotation[0] = v, ry: () => f.rotation[1] = v, rz: () => f.rotation[2] = v }[input.dataset.axis] ?? (() => {}))();
      placeIndicator(indicators[name], f);
      attachGizmoToActive();
      emitChange();
    });
  }

  // Edit-frame radios (only when >1 frame)
  for (const r of container.querySelectorAll('input[name="gs-edit-frame"], input[name="gsm-edit-frame"]')) {
    r.addEventListener('change', () => {
      if (r.checked) setActiveFrame(r.value);
    });
  }

  // Gizmo mode buttons
  const modeButtons = container.querySelectorAll('button[data-mode]');
  for (const btn of modeButtons) {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  }

  // Gizmo drag — sync state ← gizmo transform
  gizmo.addEventListener('objectChange', () => {
    const target = indicators[state.activeFrame];
    const f = state.frames[state.activeFrame];
    f.position = [round2(target.position.x * 1000), round2(target.position.y * 1000), round2(target.position.z * 1000)];
    f.rotation = [round2(target.rotation.x / DEG), round2(target.rotation.y / DEG), round2(target.rotation.z / DEG)];
    syncFormFromState(f, state.activeFrame);
    emitChange();
  });

  attachGizmoToActive();
  syncCardEditingState();

  function setActiveFrame(name) {
    if (!indicators[name]) return;
    state.activeFrame = name;
    attachGizmoToActive();
    syncCardEditingState();
  }
  function setMode(mode) {
    state.mode = mode;
    gizmo.setMode(mode);
    for (const btn of modeButtons) btn.classList.toggle('active', btn.dataset.mode === mode);
  }
  function attachGizmoToActive() { gizmo.attach(indicators[state.activeFrame]); }
  function syncCardEditingState() {
    for (const card of container.querySelectorAll('.gs-frame-card[data-frame]')) {
      card.dataset.editing = String(card.dataset.frame === state.activeFrame);
    }
  }
  function setFrames(input) {
    for (const f of frames) {
      if (input?.[f.name]) state.frames[f.name] = normalizeFrame(input[f.name]);
      placeIndicator(indicators[f.name], state.frames[f.name]);
      syncFormFromState(state.frames[f.name], f.name);
    }
    attachGizmoToActive();
  }
  function getFrames() { return structuredClone(state.frames); }
  function resizeForBounds(box) {
    if (!box || box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const scale = Math.max(size.x, size.y, size.z, 0.04) * 0.22;
    for (const f of frames) {
      rebuildIndicator(indicators[f.name], scale, f.color);
      placeIndicator(indicators[f.name], state.frames[f.name]);
    }
    gizmo.setSize(Math.max(scale * 6, 0.7));
    attachGizmoToActive();
  }
  function syncFormFromState(frame, name) {
    setInput(name, 'px', frame.position[0]);
    setInput(name, 'py', frame.position[1]);
    setInput(name, 'pz', frame.position[2]);
    setInput(name, 'rx', frame.rotation[0]);
    setInput(name, 'ry', frame.rotation[1]);
    setInput(name, 'rz', frame.rotation[2]);
  }
  function setInput(name, axis, value) {
    const input = container.querySelector(`input[data-frame="${name}"][data-axis="${axis}"]`);
    if (input && document.activeElement !== input) input.value = String(value);
  }
  function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  function dispose() {
    gizmo.detach();
    scene.remove(gizmo);
    for (const f of frames) scene.remove(indicators[f.name]);
    listeners.clear();
  }

  return { setFrames, getFrames, setActiveFrame, setMode, resizeForBounds, onChange, dispose };
}

/**
 * Build a fixed (non-editable) reference frame indicator (e.g. CAD origin).
 * Returns a Group the caller can add and reposition; pass to disposeReference
 * for cleanup.
 */
export function buildReferenceFrame(scale = 0.05, color = 0xffffff) {
  return buildIndicator(scale, color);
}
export function rebuildReferenceFrame(group, scale, color = 0xffffff) {
  rebuildIndicator(group, scale, color);
}

function buildIndicator(size, originColor) {
  const group = new THREE.Group();
  appendArrows(group, size, originColor);
  return group;
}
function rebuildIndicator(group, size, originColor) {
  group.clear();
  appendArrows(group, size, originColor);
}
function appendArrows(group, size, originColor) {
  const axes = [
    [new THREE.Vector3(1, 0, 0), 0xff4444],
    [new THREE.Vector3(0, 1, 0), 0x44cc66],
    [new THREE.Vector3(0, 0, 1), 0x4488ff],
  ];
  for (const [dir, color] of axes) {
    group.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), size, color, size * 0.28, size * 0.16));
  }
  group.add(new THREE.Mesh(
    new THREE.SphereGeometry(size * 0.09, 12, 12),
    new THREE.MeshStandardMaterial({ color: originColor, emissive: originColor, emissiveIntensity: 0.4 }),
  ));
}
function placeIndicator(group, frame) {
  group.position.set(frame.position[0] / 1000, frame.position[1] / 1000, frame.position[2] / 1000);
  group.rotation.set(frame.rotation[0] * DEG, frame.rotation[1] * DEG, frame.rotation[2] * DEG);
}
function normalizeFrame(frame) {
  return {
    position: [Number(frame.position?.[0]) || 0, Number(frame.position?.[1]) || 0, Number(frame.position?.[2]) || 0],
    rotation: [Number(frame.rotation?.[0]) || 0, Number(frame.rotation?.[1]) || 0, Number(frame.rotation?.[2]) || 0],
  };
}
function round2(v) { return Math.round(v * 100) / 100; }
