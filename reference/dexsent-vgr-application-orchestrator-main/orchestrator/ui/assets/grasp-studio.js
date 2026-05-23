import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { createSceneViewport } from "./operator/bin-picking/shared/three-scene.js";
import { loadCadLikeFile } from "./operator/bin-picking/shared/cad-loader.js";
import { loadUrdf } from "./operator/bin-picking/shared/urdf-loader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const DEG = Math.PI / 180.0;
const RAD = 180.0 / Math.PI;
const DEFAULT_MAT = new THREE.MeshStandardMaterial({
  color: 0xc8cfd6,
  metalness: 0.05,
  roughness: 0.65,
});

const HTML = `
  <div class="gp-root">
    <aside class="gp-side">
      <div>
        <h3>Object</h3>
        <div id="gp-obj-name"><strong>—</strong></div>
        <div id="gp-status" class="gp-status">Loading…</div>
      </div>

      <div class="gp-upload">
        <h3 style="margin:0">Object Frame</h3>
        <div id="gp-frame-status" class="gp-status">Loading object frame…</div>
      </div>

      <div class="gp-upload">
        <h3 style="margin:0">Object CAD</h3>
        <div id="gp-mesh-status" class="gp-status">Loading from the object folder.</div>
      </div>

      <div class="gp-upload">
        <h3 style="margin:0">Gripper Preview</h3>
        <div id="gp-gripper-status" class="gp-status">Loading selected gripper.</div>
      </div>

      <div>
        <h3>Gripping Points</h3>
        <div id="gp-list" class="gp-list"></div>
        <button id="gp-add" style="margin-top:6px;padding:6px 10px;border-radius:6px;border:1px solid rgba(170,237,246,0.18);background:#161e2a;color:#c8d8e4;cursor:pointer;">+ Add Point</button>
      </div>

      <div id="gp-editor" hidden>
        <h3>Edit Point</h3>
        <label style="font-size:11px;color:#5a7080">ID
          <input id="gp-field-id" type="text" style="display:block;width:100%;padding:4px 6px;border:1px solid rgba(170,237,246,0.15);border-radius:4px;margin-top:2px;background:#0f151e;color:#c8d8e4;" />
        </label>
        <div style="margin-top:8px;font-size:11px;color:#5a7080">Position [mm]</div>
        <div class="gp-grid3">
          <label>X<input id="gp-px" type="number" step="0.01" value="0" /></label>
          <label>Y<input id="gp-py" type="number" step="0.01" value="0" /></label>
          <label>Z<input id="gp-pz" type="number" step="0.01" value="0" /></label>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#5a7080">Rotation [°]</div>
        <div class="gp-grid3">
          <label>R<input id="gp-rx" type="number" step="0.1" value="0" /></label>
          <label>P<input id="gp-ry" type="number" step="0.1" value="0" /></label>
          <label>Y<input id="gp-rz" type="number" step="0.1" value="0" /></label>
        </div>
        <div style="margin-top:6px;font-size:10px;color:#5a7080;line-height:1.5;background:rgba(170,237,246,0.04);border-radius:4px;padding:5px 7px;border:1px solid rgba(170,237,246,0.08);">
          <span style="color:#4488ff">■</span> Gripper <b>+Z</b> = approach direction &nbsp;·&nbsp;
          <span style="color:#44cc66">■</span> Gripper <b>+Y</b> = jaw opening axis
        </div>
        <div style="margin-top:8px;font-size:11px;color:#5a7080">Stroke [mm]</div>
        <div class="gp-stroke-row">
          <input id="gp-stroke" type="number" min="0" max="500" step="0.5" value="80" />
          <input id="gp-stroke-slider" type="range" min="0" max="160" step="0.5" value="80" />
        </div>
        <div style="margin-top:8px;font-size:11px;color:#5a7080">Grasp Family</div>
        <div class="gp-grasp-family-row">
          <label class="gp-family-opt">
            <input type="radio" name="gp-family" value="external" checked />
            <span class="gp-family-label">External</span>
            <span class="gp-family-hint">max → min (close on object)</span>
          </label>
          <label class="gp-family-opt">
            <input type="radio" name="gp-family" value="internal" />
            <span class="gp-family-label">Internal</span>
            <span class="gp-family-hint">min → max (expand inside bore)</span>
          </label>
        </div>
        <div class="gp-divider"></div>
        <label class="gp-inv-toggle">
          <input id="gp-inv-enabled" type="checkbox" />
          <span>Enable rotation invariance</span>
        </label>
        <div id="gp-inv-body" hidden>
          <div class="gp-mode-bar gp-inv-mode-tabs">
            <button type="button" class="active" data-inv-tab="basic">Basic</button>
            <button type="button" data-inv-tab="advanced">Advanced</button>
          </div>
          <div id="gp-inv-basic-body">
            <div style="margin-top:8px;font-size:11px;color:#5a7080">Invariant Axis</div>
            <div class="gp-axis-radios">
              <label><input type="radio" name="gp-inv-axis" value="x" /> X</label>
              <label><input type="radio" name="gp-inv-axis" value="y" /> Y</label>
              <label><input type="radio" name="gp-inv-axis" value="z" checked /> Z</label>
            </div>
            <div class="gp-grid3">
              <label>Lower<input id="gp-inv-lower" type="number" step="1" value="0" /></label>
              <label>Upper<input id="gp-inv-upper" type="number" step="1" value="180" /></label>
              <label>Steps<input id="gp-inv-steps" type="number" min="1" max="36" step="1" value="8" /></label>
            </div>
          </div>
          <div id="gp-inv-adv-body" hidden>
            <div style="margin-top:8px;font-size:11px;color:#5a7080">Axis Position [mm]</div>
            <div class="gp-grid3">
              <label>X<input id="gp-inv-ax" type="number" step="0.01" value="0" /></label>
              <label>Y<input id="gp-inv-ay" type="number" step="0.01" value="0" /></label>
              <label>Z<input id="gp-inv-az" type="number" step="0.01" value="0" /></label>
            </div>
            <div class="gp-step-actions">
              <button id="gp-inv-enable-all" type="button">Enable All</button>
              <button id="gp-inv-disable-all" type="button">Disable All</button>
            </div>
            <div id="gp-inv-badges" class="gp-step-badges"></div>
          </div>
          <div style="margin-top:8px;font-size:11px;color:#5a7080">Preview Step <span id="gp-inv-slider-val">0</span></div>
          <input id="gp-inv-slider" class="gp-stroke-row-full" type="range" min="0" max="8" step="1" value="0" />
        </div>
        <div style="margin-top:10px;font-size:11px;color:#5a7080">Gizmo Mode</div>
        <div class="gp-mode-bar">
          <button class="active" data-mode="translate">Translate</button>
          <button data-mode="rotate">Rotate</button>
        </div>
        <hr class="gp-divider" />
        <button id="gp-delete" style="padding:6px 10px;border-radius:6px;border:1px solid rgba(224,82,82,0.5);color:#e05252;background:rgba(224,82,82,0.08);cursor:pointer;">Delete Point</button>
      </div>

      <hr class="gp-divider" />
      <div class="gp-actions">
        <button id="gp-reload">Reload</button>
        <button id="gp-save" class="primary">Save Grasps</button>
      </div>
    </aside>
    <div class="gp-viewport-shell">
      <div class="gp-viewport"></div>
      <div class="gp-hud"><span id="gp-hud-name">No mesh loaded</span></div>
    </div>
  </div>
`;

export function mountGraspStudio(container, options = {}) {
  if (!container) return null;
  container.innerHTML = HTML;

  const processId = options.processId;
  const objectId = options.objectId;
  const els = {
    objName: container.querySelector("#gp-obj-name"),
    status: container.querySelector("#gp-status"),
    frameStatus: container.querySelector("#gp-frame-status"),
    meshStatus: container.querySelector("#gp-mesh-status"),
    gripperStatus: container.querySelector("#gp-gripper-status"),
    list: container.querySelector("#gp-list"),
    add: container.querySelector("#gp-add"),
    editor: container.querySelector("#gp-editor"),
    id: container.querySelector("#gp-field-id"),
    px: container.querySelector("#gp-px"),
    py: container.querySelector("#gp-py"),
    pz: container.querySelector("#gp-pz"),
    rx: container.querySelector("#gp-rx"),
    ry: container.querySelector("#gp-ry"),
    rz: container.querySelector("#gp-rz"),
    stroke: container.querySelector("#gp-stroke"),
    strokeSlider: container.querySelector("#gp-stroke-slider"),
    familyRadios: container.querySelectorAll("input[name='gp-family']"),
    invEnabled: container.querySelector("#gp-inv-enabled"),
    invBody: container.querySelector("#gp-inv-body"),
    invBasicBody: container.querySelector("#gp-inv-basic-body"),
    invAdvBody: container.querySelector("#gp-inv-adv-body"),
    invLower: container.querySelector("#gp-inv-lower"),
    invUpper: container.querySelector("#gp-inv-upper"),
    invSteps: container.querySelector("#gp-inv-steps"),
    invAx: container.querySelector("#gp-inv-ax"),
    invAy: container.querySelector("#gp-inv-ay"),
    invAz: container.querySelector("#gp-inv-az"),
    invBadges: container.querySelector("#gp-inv-badges"),
    invEnableAll: container.querySelector("#gp-inv-enable-all"),
    invDisableAll: container.querySelector("#gp-inv-disable-all"),
    invSlider: container.querySelector("#gp-inv-slider"),
    invSliderVal: container.querySelector("#gp-inv-slider-val"),
    del: container.querySelector("#gp-delete"),
    save: container.querySelector("#gp-save"),
    reload: container.querySelector("#gp-reload"),
    hud: container.querySelector("#gp-hud-name"),
    viewport: container.querySelector(".gp-viewport"),
  };

  els.objName.innerHTML = `<strong>${objectId}</strong>`;

  const vp = createSceneViewport(els.viewport, { background: 0x101820 });
  vp.controls.target.set(0, 0, 0.05);
  vp.camera.position.set(0.4, -0.5, 0.35);
  vp.controls.update();

  const meshGroup = new THREE.Group();
  vp.scene.add(meshGroup);

  const gripperPreview = new THREE.Group();
  gripperPreview.name = "gripper_preview";
  gripperPreview.visible = false;
  vp.scene.add(gripperPreview);

  // objectFrameGroup is positioned at the object frame origin in CAD/world space.
  // All grasp indicators and markers live inside it so their local coords = object frame coords.
  const objectFrameGroup = new THREE.Group();
  objectFrameGroup.name = "object_frame";
  vp.scene.add(objectFrameGroup);

  // Small axis indicator at the object frame origin (not interactive).
  const objectFrameOrigin = makeGripIndicator(0.03, 0xffdd00);
  objectFrameGroup.add(objectFrameOrigin);

  // Indicator the gizmo is attached to (the active grasp point).
  const indicator = makeGripIndicator(0.04);
  indicator.visible = false;
  objectFrameGroup.add(indicator);

  const transform = new TransformControls(vp.camera, vp.renderer.domElement);
  transform.setSize(0.95);
  transform.setSpace("local");
  transform.setMode("translate");
  vp.scene.add(transform);
  transform.addEventListener("dragging-changed", (e) => {
    vp.controls.enabled = !e.value;
  });

  // Visualization layer: little markers per grasp.
  const markers = new Map(); // id -> THREE.Group
  function placeMarker(g) {
    let m = markers.get(g.id);
    if (!m) {
      m = makeGripIndicator(0.025, 0xff8844);
      markers.set(g.id, m);
      objectFrameGroup.add(m);
    }
    m.position.set(g.position[0] / 1000, g.position[1] / 1000, g.position[2] / 1000);
    m.rotation.set(g.rotation[0] * DEG, g.rotation[1] * DEG, g.rotation[2] * DEG);
  }
  function clearMarkers() {
    markers.forEach((m) => objectFrameGroup.remove(m));
    markers.clear();
  }

  function activeGrasp() {
    return state.grasps.find((g) => g.id === state.activeId) || null;
  }

  function defaultInvariance() {
    return {
      enabled: false,
      axis: "z",
      lowerLimit: 0,
      upperLimit: 180,
      steps: 8,
      axisPos: [0, 0, 0],
      enabledSteps: Array.from({ length: 9 }, (_, i) => i),
    };
  }

  function ensureInvariance(g) {
    if (!g.invariance || typeof g.invariance !== "object") {
      g.invariance = defaultInvariance();
    }
    const inv = g.invariance;
    inv.axisPos = Array.isArray(inv.axisPos) ? inv.axisPos : [0, 0, 0];
    inv.steps = Math.max(1, Math.min(36, Math.round(Number(inv.steps ?? 8))));
    inv.enabledSteps = Array.isArray(inv.enabledSteps)
      ? inv.enabledSteps
      : Array.from({ length: inv.steps + 1 }, (_, i) => i);
    inv.axis = ["x", "y", "z"].includes(inv.axis) ? inv.axis : "z";
    inv.lowerLimit = Number(inv.lowerLimit ?? 0);
    inv.upperLimit = Number(inv.upperLimit ?? 180);
    return inv;
  }

  function syncInvarianceEditor(g) {
    const inv = ensureInvariance(g);
    els.invEnabled.checked = !!inv.enabled;
    els.invBody.hidden = !inv.enabled;
    els.invLower.value = String(inv.lowerLimit);
    els.invUpper.value = String(inv.upperLimit);
    els.invSteps.value = String(inv.steps);
    els.invAx.value = String(inv.axisPos[0] || 0);
    els.invAy.value = String(inv.axisPos[1] || 0);
    els.invAz.value = String(inv.axisPos[2] || 0);
    container.querySelectorAll("input[name='gp-inv-axis']").forEach((radio) => {
      radio.checked = radio.value === inv.axis;
    });
    els.invSlider.max = String(inv.steps);
    state.previewStep = Math.min(state.previewStep, inv.steps);
    els.invSlider.value = String(state.previewStep);
    els.invSliderVal.textContent = String(state.previewStep);
    renderBadges(g);
  }

  function renderBadges(g) {
    const inv = ensureInvariance(g);
    const enabled = new Set(inv.enabledSteps);
    els.invBadges.replaceChildren();
    for (let i = 0; i <= inv.steps; i += 1) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "gp-step-badge";
      btn.textContent = String(i);
      btn.classList.toggle("enabled", enabled.has(i));
      btn.addEventListener("click", () => {
        if (enabled.has(i)) enabled.delete(i);
        else enabled.add(i);
        inv.enabledSteps = Array.from(enabled).sort((a, b) => a - b);
        btn.classList.toggle("enabled", enabled.has(i));
      });
      els.invBadges.appendChild(btn);
    }
  }

  const ringGroup = new THREE.Group();
  objectFrameGroup.add(ringGroup);

  const state = {
    grasps: [],
    activeId: null,
    gripper: null,
    gripperTcpFrame: null,
    maxStroke: 160,
    invMode: "basic",
    previewStep: 0,
  };

  function setStatus(text, isError = false) {
    els.status.textContent = text || "";
    els.status.classList.toggle("err", !!isError);
  }

  function renderList() {
    els.list.replaceChildren();
    if (!state.grasps.length) {
      const empty = document.createElement("div");
      empty.className = "gp-status";
      empty.textContent = "No grasps yet — click + Add Point.";
      els.list.appendChild(empty);
      return;
    }
    for (const g of state.grasps) {
      const row = document.createElement("div");
      row.className = "gp-list-row" + (g.id === state.activeId ? " active" : "");
      const family = g.grasp_family || "external";
      row.innerHTML = `
        <span style="flex:1">${escapeHtml(g.id)}</span>
        <span class="gp-tag gp-tag-${family}">${(g.stroke ?? 80).toFixed(1)} mm ${family === "internal" ? "int" : "ext"}${ensureInvariance(g).enabled ? " inv" : ""}</span>
      `;
      row.addEventListener("click", () => selectGrasp(g.id));
      els.list.appendChild(row);
    }
  }

  function renderInvarianceRing() {
    ringGroup.clear();
    const g = activeGrasp();
    if (!g || !ensureInvariance(g).enabled) return;
    const inv = ensureInvariance(g);
    const center = new THREE.Vector3(
      Number(inv.axisPos[0] || 0) / 1000,
      Number(inv.axisPos[1] || 0) / 1000,
      Number(inv.axisPos[2] || 0) / 1000
    );
    const radius = 0.08;
    const torus = new THREE.Mesh(
      new THREE.TorusGeometry(radius, radius * 0.018, 16, 72),
      new THREE.MeshStandardMaterial({
        color: 0xff3333,
        transparent: true,
        opacity: 0.58,
        depthWrite: false,
      })
    );
    torus.position.copy(center);
    if (inv.axis === "x") torus.rotation.y = Math.PI / 2;
    else if (inv.axis === "y") torus.rotation.x = Math.PI / 2;
    ringGroup.add(torus);
  }

  function syncEditorFromActive() {
    const g = state.grasps.find((x) => x.id === state.activeId);
    if (!g) {
      els.editor.hidden = true;
      indicator.visible = false;
      gripperPreview.visible = false;
      transform.detach();
      renderInvarianceRing();
      return;
    }
    els.editor.hidden = false;
    els.id.value = g.id;
    els.px.value = g.position[0].toFixed(2);
    els.py.value = g.position[1].toFixed(2);
    els.pz.value = g.position[2].toFixed(2);
    els.rx.value = g.rotation[0].toFixed(1);
    els.ry.value = g.rotation[1].toFixed(1);
    els.rz.value = g.rotation[2].toFixed(1);
    g.stroke = Math.max(0, Math.min(state.maxStroke, Number(g.stroke ?? state.maxStroke)));
    els.stroke.value = g.stroke.toFixed(1);
    els.strokeSlider.value = String(g.stroke);
    const family = g.grasp_family || "external";
    els.familyRadios.forEach((r) => { r.checked = r.value === family; });
    syncInvarianceEditor(g);
    applyGripperStroke(g.stroke);
    indicator.visible = true;
    indicator.position.set(g.position[0] / 1000, g.position[1] / 1000, g.position[2] / 1000);
    indicator.rotation.set(g.rotation[0] * DEG, g.rotation[1] * DEG, g.rotation[2] * DEG);
    transform.attach(indicator);
    renderInvarianceRing();
    repositionGripperPreview();
  }

  function selectGrasp(id) {
    state.activeId = id;
    syncEditorFromActive();
    renderList();
  }

  function pushIndicatorToActive() {
    const g = state.grasps.find((x) => x.id === state.activeId);
    if (!g) return;
    g.position = [
      Math.round(indicator.position.x * 100000) / 100,
      Math.round(indicator.position.y * 100000) / 100,
      Math.round(indicator.position.z * 100000) / 100,
    ];
    g.rotation = [
      Math.round(indicator.rotation.x * RAD * 10) / 10,
      Math.round(indicator.rotation.y * RAD * 10) / 10,
      Math.round(indicator.rotation.z * RAD * 10) / 10,
    ];
    placeMarker(g);
    syncEditorFromActive();
  }

  transform.addEventListener("objectChange", pushIndicatorToActive);

  function pushFieldsToActive() {
    const g = state.grasps.find((x) => x.id === state.activeId);
    if (!g) return;
    const newId = (els.id.value || "").trim();
    if (newId && newId !== g.id) {
      const existing = markers.get(g.id);
      if (existing) {
        markers.delete(g.id);
        markers.set(newId, existing);
      }
      g.id = newId;
      state.activeId = newId;
    }
    g.position = [
      Number(els.px.value) || 0,
      Number(els.py.value) || 0,
      Number(els.pz.value) || 0,
    ];
    g.rotation = [
      Number(els.rx.value) || 0,
      Number(els.ry.value) || 0,
      Number(els.rz.value) || 0,
    ];
    g.stroke = Math.max(0, Math.min(state.maxStroke, Number(els.stroke.value) || 0));
    els.stroke.value = g.stroke.toFixed(1);
    els.strokeSlider.value = String(g.stroke);
    const checkedFamily = Array.from(els.familyRadios).find((r) => r.checked);
    g.grasp_family = checkedFamily ? checkedFamily.value : "external";
    indicator.position.set(g.position[0] / 1000, g.position[1] / 1000, g.position[2] / 1000);
    indicator.rotation.set(g.rotation[0] * DEG, g.rotation[1] * DEG, g.rotation[2] * DEG);
    placeMarker(g);
    applyGripperStroke(g.stroke);
    renderInvarianceRing();
    repositionGripperPreview();
    renderList();
  }

  ["px", "py", "pz", "rx", "ry", "rz", "stroke", "id"].forEach((k) => {
    els[k].addEventListener("input", pushFieldsToActive);
    els[k].addEventListener("change", pushFieldsToActive);
  });
  els.familyRadios.forEach((r) => r.addEventListener("change", pushFieldsToActive));
  els.strokeSlider.addEventListener("input", () => {
    els.stroke.value = Number(els.strokeSlider.value || 0).toFixed(1);
    pushFieldsToActive();
  });
  els.strokeSlider.addEventListener("change", () => {
    els.stroke.value = Number(els.strokeSlider.value || 0).toFixed(1);
    pushFieldsToActive();
  });

  els.invEnabled.addEventListener("change", () => {
    const g = activeGrasp();
    if (!g) return;
    ensureInvariance(g).enabled = !!els.invEnabled.checked;
    els.invBody.hidden = !g.invariance.enabled;
    renderInvarianceRing();
    repositionGripperPreview();
    renderList();
  });
  container.querySelectorAll("[data-inv-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.invMode = btn.dataset.invTab || "basic";
      container.querySelectorAll("[data-inv-tab]").forEach((b) => b.classList.toggle("active", b === btn));
      els.invBasicBody.hidden = state.invMode !== "basic";
      els.invAdvBody.hidden = state.invMode !== "advanced";
    });
  });
  container.querySelectorAll("input[name='gp-inv-axis']").forEach((radio) => {
    radio.addEventListener("change", () => {
      const g = activeGrasp();
      if (!g || !radio.checked) return;
      ensureInvariance(g).axis = radio.value;
      renderInvarianceRing();
      repositionGripperPreview();
    });
  });
  [
    [els.invLower, (inv, v) => { inv.lowerLimit = v; }],
    [els.invUpper, (inv, v) => { inv.upperLimit = v; }],
    [els.invAx, (inv, v) => { inv.axisPos[0] = v; }],
    [els.invAy, (inv, v) => { inv.axisPos[1] = v; }],
    [els.invAz, (inv, v) => { inv.axisPos[2] = v; }],
  ].forEach(([input, apply]) => {
    input.addEventListener("input", () => {
      const g = activeGrasp();
      if (!g) return;
      apply(ensureInvariance(g), Number(input.value) || 0);
      renderInvarianceRing();
      repositionGripperPreview();
    });
  });
  els.invSteps.addEventListener("input", () => {
    const g = activeGrasp();
    if (!g) return;
    const inv = ensureInvariance(g);
    const steps = Math.max(1, Math.min(36, Math.round(Number(els.invSteps.value) || 8)));
    inv.steps = steps;
    inv.enabledSteps = Array.from({ length: steps + 1 }, (_, i) => i);
    els.invSlider.max = String(steps);
    state.previewStep = Math.min(state.previewStep, steps);
    els.invSlider.value = String(state.previewStep);
    els.invSliderVal.textContent = String(state.previewStep);
    renderBadges(g);
    repositionGripperPreview();
  });
  els.invEnableAll.addEventListener("click", () => {
    const g = activeGrasp();
    if (!g) return;
    const inv = ensureInvariance(g);
    inv.enabledSteps = Array.from({ length: inv.steps + 1 }, (_, i) => i);
    renderBadges(g);
  });
  els.invDisableAll.addEventListener("click", () => {
    const g = activeGrasp();
    if (!g) return;
    ensureInvariance(g).enabledSteps = [];
    renderBadges(g);
  });
  els.invSlider.addEventListener("input", () => {
    state.previewStep = Number(els.invSlider.value) || 0;
    els.invSliderVal.textContent = String(state.previewStep);
    repositionGripperPreview();
  });

  els.add.addEventListener("click", () => {
    const id = `g${state.grasps.length + 1}`;
    const grasp = {
      id,
      position: [0, 0, 0],
      rotation: [0, 0, 0],
      stroke: Math.min(80, state.maxStroke),
      grasp_family: "external",
      type: "parallel_jaw",
    };
    state.grasps.push(grasp);
    placeMarker(grasp);
    selectGrasp(id);
  });

  els.del.addEventListener("click", () => {
    const idx = state.grasps.findIndex((g) => g.id === state.activeId);
    if (idx < 0) return;
    const removed = state.grasps.splice(idx, 1)[0];
    const m = markers.get(removed.id);
    if (m) objectFrameGroup.remove(m);
    markers.delete(removed.id);
    state.activeId = state.grasps[0]?.id || null;
    syncEditorFromActive();
    renderList();
    repositionGripperPreview();
  });

  container.querySelectorAll(".gp-mode-bar button[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      transform.setMode(btn.dataset.mode);
      container
        .querySelectorAll(".gp-mode-bar button[data-mode]")
        .forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  els.save.addEventListener("click", saveGrasps);
  els.reload.addEventListener("click", loadAll);

  async function loadMesh(url) {
    try {
      while (meshGroup.children.length) meshGroup.remove(meshGroup.children[0]);
      const ext = url.split("?")[0].split(".").pop().toLowerCase();
      let obj = null;
      if (ext === "stl") {
        const buf = await (await fetch(url)).arrayBuffer();
        const geo = new STLLoader().parse(buf);
        geo.computeVertexNormals();
        obj = new THREE.Mesh(geo, DEFAULT_MAT.clone());
      } else if (ext === "obj") {
        const text = await (await fetch(url)).text();
        obj = new OBJLoader().parse(text);
        obj.traverse((c) => {
          if (c.isMesh) c.material = DEFAULT_MAT.clone();
        });
      } else {
        // best-effort fallback via cad-loader (handles step/iges via OCCT)
        const blob = await (await fetch(url)).blob();
        const file = new File([blob], "mesh." + ext, { type: blob.type });
        obj = await loadCadLikeFile(file);
      }
      // Mesh in mm — bin-picking convention. Scale to meters for scene:
      obj.scale.setScalar(0.001);
      meshGroup.add(obj);
      vp.frameObject(meshGroup);
      els.hud.textContent = `Mesh loaded (${ext.toUpperCase()})`;
      els.meshStatus.textContent = `Loaded ${url.split("/").pop() || "mesh"} from object folder.`;
    } catch (err) {
      console.error(err);
      els.hud.textContent = `Mesh load failed: ${err.message}`;
      els.meshStatus.textContent = `Mesh load failed: ${err.message}`;
    }
  }

  async function loadGripperPreview() {
    try {
      const res = await fetch("/bin-picking/gripper-assets");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const gripper = Array.isArray(data?.grippers) ? data.grippers[0] : null;
      if (!gripper?.urdf_url) {
        els.gripperStatus.textContent = "No gripper URDF found.";
        return;
      }
      const maxStroke = Number(gripper.manifest?.stroke_width ?? 160);
      state.maxStroke = Number.isFinite(maxStroke) && maxStroke > 0 ? maxStroke : 160;
      state.gripperTcpFrame = gripper.frames?.tcp || null;
      els.stroke.max = String(state.maxStroke);
      els.strokeSlider.max = String(state.maxStroke);
      while (gripperPreview.children.length) gripperPreview.remove(gripperPreview.children[0]);
      const root = await loadUrdf(gripper.urdf_url, { loadCollision: false });
      root.name = "grasp_gripper_preview_model";
      root.traverse((node) => {
        if (node.isMesh) {
          node.material = node.material?.clone?.() || new THREE.MeshStandardMaterial();
          node.material.transparent = true;
          node.material.opacity = 0.62;
          node.material.depthWrite = true;
        }
      });
      gripperPreview.add(root);
      state.gripper = gripper;
      els.gripperStatus.textContent = `Loaded ${gripper.name || gripper.id || "gripper"} preview.`;
      applyGripperStroke(Number(els.stroke.value || 80));
      repositionGripperPreview();
    } catch (err) {
      console.error(err);
      els.gripperStatus.textContent = `Gripper preview failed: ${err.message}`;
    }
  }

  function applyGripperStroke(strokeMm) {
    const root = gripperPreview.children[0];
    const urdf = root?.userData?.urdf;
    if (!urdf?.actuators?.length || typeof urdf.setJointValue !== "function") return;
    const maxStroke = Number(state.maxStroke || 160);
    const clampedMm = Math.max(0, Math.min(maxStroke, Number(strokeMm) || 0));
    const perFingerM = clampedMm / 2000.0;
    urdf.actuators.forEach((joint) => {
      urdf.setJointValue(joint.name, perFingerM);
    });
  }

  function tcpFrameMatrix() {
    const tcp = state.gripperTcpFrame || { position: [0, 0, 0], rotation: [0, 0, 0] };
    const pos = Array.isArray(tcp.position) ? tcp.position : [0, 0, 0];
    const rot = Array.isArray(tcp.rotation) ? tcp.rotation : [0, 0, 0];
    return new THREE.Matrix4().compose(
      new THREE.Vector3(
        Number(pos[0] || 0) / 1000,
        Number(pos[1] || 0) / 1000,
        Number(pos[2] || 0) / 1000
      ),
      new THREE.Quaternion().setFromEuler(
        new THREE.Euler(
          Number(rot[0] || 0) * DEG,
          Number(rot[1] || 0) * DEG,
          Number(rot[2] || 0) * DEG,
          "XYZ"
        )
      ),
      new THREE.Vector3(1, 1, 1)
    );
  }

  function graspTcpWorldMatrix(g) {
    const inv = ensureInvariance(g);
    // Build the grasp matrix in object frame local coords.
    const localBase = new THREE.Matrix4().compose(
      new THREE.Vector3(
        Number(g.position[0] || 0) / 1000,
        Number(g.position[1] || 0) / 1000,
        Number(g.position[2] || 0) / 1000
      ),
      new THREE.Quaternion().setFromEuler(
        new THREE.Euler(
          Number(g.rotation[0] || 0) * DEG,
          Number(g.rotation[1] || 0) * DEG,
          Number(g.rotation[2] || 0) * DEG,
          "XYZ"
        )
      ),
      new THREE.Vector3(1, 1, 1)
    );
    let localMat = localBase;
    if (inv.enabled && state.previewStep > 0) {
      const angle = THREE.MathUtils.lerp(inv.lowerLimit, inv.upperLimit, state.previewStep / inv.steps) * DEG;
      const axis = {
        x: new THREE.Vector3(1, 0, 0),
        y: new THREE.Vector3(0, 1, 0),
        z: new THREE.Vector3(0, 0, 1),
      }[inv.axis] || new THREE.Vector3(0, 0, 1);
      const center = new THREE.Vector3(
        Number(inv.axisPos[0] || 0) / 1000,
        Number(inv.axisPos[1] || 0) / 1000,
        Number(inv.axisPos[2] || 0) / 1000
      );
      const rotationAroundAxis = new THREE.Matrix4()
        .makeTranslation(center.x, center.y, center.z)
        .multiply(new THREE.Matrix4().makeRotationAxis(axis, angle))
        .multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
      localMat = rotationAroundAxis.multiply(localBase);
    }
    // Transform from object frame to world by composing with objectFrameGroup world matrix.
    objectFrameGroup.updateMatrixWorld();
    return objectFrameGroup.matrixWorld.clone().multiply(localMat);
  }

  function repositionGripperPreview() {
    const g = activeGrasp();
    const model = gripperPreview.children[0];
    if (!g || !model) {
      gripperPreview.visible = false;
      return;
    }
    const worldTcp = graspTcpWorldMatrix(g);
    const rootToTcpInv = tcpFrameMatrix().invert();
    const worldRoot = worldTcp.multiply(rootToTcpInv);
    gripperPreview.matrixAutoUpdate = false;
    gripperPreview.matrix.copy(worldRoot);
    gripperPreview.updateMatrixWorld(true);
    gripperPreview.visible = true;
    applyGripperStroke(g.stroke ?? state.maxStroke);
  }

  async function loadAll() {
    setStatus("Loading…");
    try {
      // Fetch object frame (defined relative to CAD frame).
      const frameRes = await fetch(
        `/processes/${encodeURIComponent(processId)}/objects/${encodeURIComponent(objectId)}/frame`
      );
      if (frameRes.ok) {
        const frameData = await frameRes.json();
        const f = frameData?.frame || {};
        const pos = Array.isArray(f.position_m) ? f.position_m : [0, 0, 0];
        const rpy = Array.isArray(f.rotation_rpy_deg) ? f.rotation_rpy_deg : [0, 0, 0];
        objectFrameGroup.position.set(Number(pos[0]) || 0, Number(pos[1]) || 0, Number(pos[2]) || 0);
        objectFrameGroup.rotation.set(
          (Number(rpy[0]) || 0) * DEG,
          (Number(rpy[1]) || 0) * DEG,
          (Number(rpy[2]) || 0) * DEG
        );
        els.frameStatus.textContent =
          `Object frame: [${pos.map((v) => Number(v).toFixed(3)).join(", ")}] m, ` +
          `[${rpy.map((v) => Number(v).toFixed(1)).join(", ")}]°`;
      } else {
        objectFrameGroup.position.set(0, 0, 0);
        objectFrameGroup.rotation.set(0, 0, 0);
        els.frameStatus.textContent = "No object frame defined — using CAD frame.";
      }
      objectFrameGroup.updateMatrixWorld();

      // Try to fetch existing mesh metadata
      const metaRes = await fetch(
        `/processes/${encodeURIComponent(processId)}/objects/${encodeURIComponent(objectId)}/cad`
      );
      if (metaRes.ok) {
        const data = await metaRes.json();
        if (data?.url) await loadMesh(data.url);
      } else {
        els.meshStatus.textContent = "No CAD found in the object folder.";
      }
      // Load grasps
      const gRes = await fetch(
        `/processes/${encodeURIComponent(processId)}/objects/${encodeURIComponent(objectId)}/grasps`
      );
      if (gRes.ok) {
        const data = await gRes.json();
        state.grasps = Array.isArray(data?.grasps) ? data.grasps : [];
        clearMarkers();
        for (const g of state.grasps) {
          g.position = (g.position || [0, 0, 0]).slice(0, 3).map(Number);
          g.rotation = (g.rotation || [0, 0, 0]).slice(0, 3).map(Number);
          g.stroke = Math.max(0, Math.min(state.maxStroke, Number(g.stroke ?? 80)));
          g.grasp_family = (g.grasp_family === "internal" || g.grasp_family === "external") ? g.grasp_family : "external";
          placeMarker(g);
        }
      } else {
        state.grasps = [];
      }
      state.activeId = state.grasps[0]?.id || null;
      renderList();
      syncEditorFromActive();
      renderInvarianceRing();
      repositionGripperPreview();
      setStatus(state.grasps.length ? `Loaded ${state.grasps.length} grasp(s).` : "No grasps yet.");
    } catch (err) {
      setStatus(`Load failed: ${err.message}`, true);
    }
  }

  async function saveGrasps() {
    setStatus("Saving…");
    try {
      const res = await fetch(
        `/processes/${encodeURIComponent(processId)}/objects/${encodeURIComponent(objectId)}/grasps`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ grasps: state.grasps }),
        }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatus(`Saved ${state.grasps.length} grasp(s).`);
    } catch (err) {
      setStatus(`Save failed: ${err.message}`, true);
    }
  }

  loadGripperPreview().finally(loadAll);

  return {
    save: saveGrasps,
    reload: loadAll,
    dispose: () => {
      transform.detach();
      vp.scene.remove(transform);
      vp.dispose?.();
    },
  };
}

function makeGripIndicator(size = 0.04, originColor = 0x33d6ff) {
  const group = new THREE.Group();
  const axes = [
    [new THREE.Vector3(1, 0, 0), 0xff4444],
    [new THREE.Vector3(0, 1, 0), 0x44cc66],
    [new THREE.Vector3(0, 0, 1), 0x4488ff],
  ];
  for (const [dir, color] of axes) {
    group.add(
      new THREE.ArrowHelper(dir, new THREE.Vector3(), size, color, size * 0.28, size * 0.18)
    );
  }
  group.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(size * 0.12, 16, 12),
      new THREE.MeshStandardMaterial({
        color: originColor,
        emissive: originColor,
        emissiveIntensity: 0.4,
      })
    )
  );
  return group;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}
