import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { createSceneViewport } from "./operator/bin-picking/shared/three-scene.js";
import { loadUrdf } from "./operator/bin-picking/shared/urdf-loader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";

const editors = [];
const DEG = Math.PI / 180.0;
const RAD = 180.0 / Math.PI;
const DEFAULT_MESH_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0xd8d2c6,
  metalness: 0.08,
  roughness: 0.55,
});

function resolveField(field) {
  if (!field) return null;
  if (typeof field === "string") return document.getElementById(field);
  return field;
}

function numberFromInput(input, fallback = 0) {
  const value = Number(input?.value);
  return Number.isFinite(value) ? value : fallback;
}

function writeInputSilent(input, value, digits) {
  if (!input) return;
  const text = Number(value || 0).toFixed(digits);
  if (input.value !== text) input.value = text;
}

function buildFrameIndicator(size = 0.06, color = 0x33d6ff) {
  const group = new THREE.Group();
  const axes = [
    [new THREE.Vector3(1, 0, 0), 0xff4444],
    [new THREE.Vector3(0, 1, 0), 0x44cc66],
    [new THREE.Vector3(0, 0, 1), 0x4488ff],
  ];
  for (const [dir, c] of axes) {
    group.add(
      new THREE.ArrowHelper(dir, new THREE.Vector3(), size, c, size * 0.28, size * 0.18)
    );
  }
  group.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(size * 0.12, 16, 12),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.4,
      })
    )
  );
  return group;
}

async function loadMeshUrl(url) {
  const ext = url.split("?")[0].split(".").pop().toLowerCase();
  if (ext === "stl") {
    const buf = await (await fetch(url)).arrayBuffer();
    const geometry = new STLLoader().parse(buf);
    geometry.computeVertexNormals();
    return new THREE.Mesh(geometry, DEFAULT_MESH_MATERIAL.clone());
  }
  if (ext === "obj") {
    const text = await (await fetch(url)).text();
    const obj = new OBJLoader().parse(text);
    obj.traverse((c) => {
      if (c.isMesh) c.material = DEFAULT_MESH_MATERIAL.clone();
    });
    return obj;
  }
  if (ext === "dae") {
    const collada = await new ColladaLoader().loadAsync(url);
    const root = collada.scene;
    root.traverse((c) => {
      if (c.isMesh && (!c.material || !c.material.isMeshStandardMaterial)) {
        c.material = DEFAULT_MESH_MATERIAL.clone();
      }
    });
    return root;
  }
  throw new Error(`Unsupported mesh format: .${ext}`);
}

function mountFrameEditor(options = {}) {
  const root =
    typeof options.root === "string"
      ? document.getElementById(options.root)
      : options.root || document.getElementById(options.rootId);
  if (!root || root.dataset.gizmoMounted === "1") return null;
  root.dataset.gizmoMounted = "1";

  const fields = {
    x: resolveField(options.fields?.x),
    y: resolveField(options.fields?.y),
    z: resolveField(options.fields?.z),
    roll: resolveField(options.fields?.roll),
    pitch: resolveField(options.fields?.pitch),
    yaw: resolveField(options.fields?.yaw),
  };

  const shell = document.createElement("div");
  shell.className = "gizmo-editor";
  shell.innerHTML = `
    <div class="gizmo-editor-toolbar">
      <div class="gizmo-editor-title"></div>
      <div class="gizmo-editor-actions">
        <button type="button" data-mode="translate">Move</button>
        <button type="button" data-mode="rotate">Rotate</button>
        <button type="button" data-reset>Zero</button>
        <button type="button" data-fit>Fit</button>
      </div>
    </div>
    <div class="gizmo-editor-viewport"></div>
    <div class="gizmo-editor-status muted"></div>
    <div class="gizmo-editor-readout"></div>
  `;
  root.appendChild(shell);

  shell.querySelector(".gizmo-editor-title").textContent =
    options.title || "Frame Gizmo";
  const viewportEl = shell.querySelector(".gizmo-editor-viewport");
  const statusEl = shell.querySelector(".gizmo-editor-status");
  const readout = shell.querySelector(".gizmo-editor-readout");

  const viewport = createSceneViewport(viewportEl, { background: 0x101820 });
  viewport.controls.target.set(0, 0, 0.05);
  viewport.camera.position.set(0.45, -0.65, 0.45);
  viewport.controls.update();

  // Asset (URDF/mesh) shown as visual context — not interactable.
  const assetGroup = new THREE.Group();
  assetGroup.name = "gizmo-asset";
  viewport.scene.add(assetGroup);

  // Frame indicator: this is what the gizmo edits.
  const indicator = buildFrameIndicator(options.indicatorSize ?? 0.06);
  viewport.scene.add(indicator);

  const transform = new TransformControls(
    viewport.camera,
    viewport.renderer.domElement
  );
  transform.setSize(0.95);
  transform.setSpace("local");
  transform.setMode("translate");
  transform.setTranslationSnap(options.translationSnap ?? null);
  transform.setRotationSnap(
    options.rotationSnapDeg ? options.rotationSnapDeg * DEG : null
  );
  viewport.scene.add(transform);
  transform.attach(indicator);

  transform.addEventListener("dragging-changed", (event) => {
    viewport.controls.enabled = !event.value;
  });

  const writeFields = () => {
    writeInputSilent(fields.x, indicator.position.x, 6);
    writeInputSilent(fields.y, indicator.position.y, 6);
    writeInputSilent(fields.z, indicator.position.z, 6);
    writeInputSilent(fields.roll, indicator.rotation.x * RAD, 3);
    writeInputSilent(fields.pitch, indicator.rotation.y * RAD, 3);
    writeInputSilent(fields.yaw, indicator.rotation.z * RAD, 3);
    updateReadout();
    if (typeof options.onChange === "function") {
      options.onChange(readFrame());
    }
  };

  const readFrame = () => ({
    position_m: [
      numberFromInput(fields.x),
      numberFromInput(fields.y),
      numberFromInput(fields.z),
    ],
    rotation_rpy_deg: [
      numberFromInput(fields.roll),
      numberFromInput(fields.pitch),
      numberFromInput(fields.yaw),
    ],
  });

  const syncFromFields = () => {
    const frame = readFrame();
    indicator.position.set(
      frame.position_m[0],
      frame.position_m[1],
      frame.position_m[2]
    );
    indicator.rotation.set(
      frame.rotation_rpy_deg[0] * DEG,
      frame.rotation_rpy_deg[1] * DEG,
      frame.rotation_rpy_deg[2] * DEG
    );
    updateReadout();
  };

  function updateReadout() {
    readout.textContent =
      `xyz(m): ${indicator.position.x.toFixed(4)}, ${indicator.position.y.toFixed(4)}, ${indicator.position.z.toFixed(4)} | ` +
      `rpy(deg): ${(indicator.rotation.x * RAD).toFixed(2)}, ${(indicator.rotation.y * RAD).toFixed(2)}, ${(indicator.rotation.z * RAD).toFixed(2)}`;
  }

  function setStatus(text, isError = false) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("err", !!isError);
  }

  function clearAsset() {
    while (assetGroup.children.length > 0) {
      const child = assetGroup.children[0];
      assetGroup.remove(child);
      child.traverse?.((c) => {
        if (c.isMesh) {
          c.geometry?.dispose?.();
          if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose?.());
          else c.material?.dispose?.();
        }
      });
    }
  }

  function fitToAsset() {
    const box = new THREE.Box3().setFromObject(assetGroup);
    if (!box.isEmpty()) {
      viewport.frameObject(assetGroup);
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.05);
      transform.setSize(Math.max(radius * 1.6, 0.7));
      const indicatorScale = Math.max(radius * 0.18, 0.04);
      indicator.scale.setScalar(1);
      // Re-place arrows of indicator already sized; just bump the gizmo size.
    } else {
      viewport.controls.target.set(0, 0, 0.05);
      viewport.camera.position.set(0.45, -0.65, 0.45);
      viewport.controls.update();
      transform.setSize(0.95);
    }
  }

  async function loadAsset(asset) {
    clearAsset();
    if (!asset) {
      setStatus("");
      fitToAsset();
      return null;
    }
    try {
      let obj = null;
      if (asset.urdfUrl) {
        setStatus(`Loading URDF: ${asset.urdfUrl}`);
        obj = await loadUrdf(asset.urdfUrl);
      } else if (asset.meshUrl) {
        setStatus(`Loading mesh: ${asset.meshUrl}`);
        obj = await loadMeshUrl(asset.meshUrl);
        if (Number.isFinite(Number(asset.meshScale))) {
          obj.scale.setScalar(Number(asset.meshScale));
        }
      }
      if (obj) {
        assetGroup.add(obj);
        setStatus(asset.label || "Asset loaded.");
        fitToAsset();
      } else {
        setStatus("No URDF/mesh URL provided.");
      }
      return obj;
    } catch (err) {
      console.error("[gizmo-editor] asset load failed:", err);
      setStatus(`Asset load failed: ${err?.message || err}`, true);
      return null;
    }
  }

  // Wire input ↔ gizmo (no synthetic dispatch from gizmo side; only user typing
  // dispatches input/change → syncFromFields is the canonical apply path).
  transform.addEventListener("objectChange", writeFields);
  Object.values(fields).forEach((input) => {
    if (!input) return;
    input.addEventListener("input", syncFromFields);
    input.addEventListener("change", syncFromFields);
  });

  shell.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      transform.setMode(button.dataset.mode);
      shell.querySelectorAll("[data-mode]").forEach((b) =>
        b.classList.toggle("active", b.dataset.mode === button.dataset.mode)
      );
    });
  });
  shell.querySelector("[data-mode='translate']")?.classList.add("active");

  shell.querySelector("[data-reset]")?.addEventListener("click", () => {
    indicator.position.set(0, 0, 0);
    indicator.rotation.set(0, 0, 0);
    writeFields();
  });
  shell.querySelector("[data-fit]")?.addEventListener("click", fitToAsset);

  syncFromFields();
  fitToAsset();

  const editor = {
    root,
    sync: syncFromFields,
    read: readFrame,
    loadAsset,
    clearAsset,
    setStatus,
    fit: fitToAsset,
    setMode: (mode) => transform.setMode(mode),
    dispose() {
      transform.detach();
      viewport.scene.remove(transform);
      viewport.scene.remove(indicator);
      clearAsset();
      viewport.scene.remove(assetGroup);
      viewport.dispose();
      root.dataset.gizmoMounted = "";
      shell.remove();
    },
  };
  editors.push(editor);
  return editor;
}

function syncAll() {
  editors.forEach((editor) => editor.sync());
}

function loadAssetForAll(asset) {
  return Promise.all(editors.map((e) => e.loadAsset(asset)));
}

window.impGizmoEditors = { mountFrameEditor, syncAll, loadAssetForAll };
window.dispatchEvent(new Event("imp:gizmo-ready"));
