import * as THREE from "three";
import { createSceneViewport } from "./operator/bin-picking/shared/three-scene.js";
import { loadUrdf } from "./operator/bin-picking/shared/urdf-loader.js";
import { createFrameEditor } from "./operator/bin-picking/shared/frame-editor.js";

const HTML = `
  <div class="gs-mini">
    <div class="gs-mini-config">
      <div class="gs-mini-header">
        <strong id="gsm-current-name">No gripper selected</strong>
        <span id="gsm-load-status" class="muted gsm-status">Loading available gripper assets.</span>
      </div>
      <label class="workflow-field gsm-asset-picker-wrap">
        <span>Asset Gripper</span>
        <select id="gsm-asset-picker"></select>
      </label>

      <section class="gs-section">
        <div class="gs-section-title">Stroke Width</div>
        <div class="gsm-stroke-row">
          <input id="gsm-stroke-width" type="number" min="0" max="500" step="0.5" value="70" />
          <input id="gsm-stroke-slider" type="range" min="0" max="160" step="0.5" value="70" />
        </div>
      </section>

      <section class="gs-section gs-frame-card" data-frame="flange">
        <div class="gs-frame-header">
          <span class="gs-frame-dot gs-dot-flange"></span>
          <span class="gs-section-title gs-frame-name">Flange Frame</span>
          <label class="gs-edit-toggle">
            <input type="radio" name="gsm-edit-frame" value="flange" checked />
            <span>Edit on CAD</span>
          </label>
        </div>
        <div class="gs-input-group-label">Position [mm]</div>
        <div class="gs-inputs-grid">
          <label class="workflow-field"><span>X</span><input data-frame="flange" data-axis="px" type="number" step="0.01" value="0" /></label>
          <label class="workflow-field"><span>Y</span><input data-frame="flange" data-axis="py" type="number" step="0.01" value="0" /></label>
          <label class="workflow-field"><span>Z</span><input data-frame="flange" data-axis="pz" type="number" step="0.01" value="0" /></label>
        </div>
        <div class="gs-input-group-label">Rotation [°]</div>
        <div class="gs-inputs-grid">
          <label class="workflow-field"><span>R</span><input data-frame="flange" data-axis="rx" type="number" step="0.1" value="0" /></label>
          <label class="workflow-field"><span>P</span><input data-frame="flange" data-axis="ry" type="number" step="0.1" value="0" /></label>
          <label class="workflow-field"><span>Y</span><input data-frame="flange" data-axis="rz" type="number" step="0.1" value="0" /></label>
        </div>
      </section>

      <section class="gs-section gs-frame-card" data-frame="tcp">
        <div class="gs-frame-header">
          <span class="gs-frame-dot gs-dot-tcp"></span>
          <span class="gs-section-title gs-frame-name">TCP Frame</span>
          <label class="gs-edit-toggle">
            <input type="radio" name="gsm-edit-frame" value="tcp" />
            <span>Edit on CAD</span>
          </label>
        </div>
        <div class="gs-input-group-label">Position [mm]</div>
        <div class="gs-inputs-grid">
          <label class="workflow-field"><span>X</span><input data-frame="tcp" data-axis="px" type="number" step="0.01" value="0" /></label>
          <label class="workflow-field"><span>Y</span><input data-frame="tcp" data-axis="py" type="number" step="0.01" value="0" /></label>
          <label class="workflow-field"><span>Z</span><input data-frame="tcp" data-axis="pz" type="number" step="0.01" value="0" /></label>
        </div>
        <div class="gs-input-group-label">Rotation [°]</div>
        <div class="gs-inputs-grid">
          <label class="workflow-field"><span>R</span><input data-frame="tcp" data-axis="rx" type="number" step="0.1" value="0" /></label>
          <label class="workflow-field"><span>P</span><input data-frame="tcp" data-axis="ry" type="number" step="0.1" value="0" /></label>
          <label class="workflow-field"><span>Y</span><input data-frame="tcp" data-axis="rz" type="number" step="0.1" value="0" /></label>
        </div>
      </section>

      <section class="gs-section">
        <div class="gs-section-title">Gizmo Mode</div>
        <div class="segmented gsm-mode">
          <button type="button" class="active" data-mode="translate">Translate</button>
          <button type="button" data-mode="rotate">Rotate</button>
        </div>
      </section>

      <div class="gs-save-row">
        <button type="button" class="primary gsm-save">Save Frames</button>
        <button type="button" class="secondary gsm-reload">Reload</button>
      </div>
    </div>
    <div class="gs-mini-viewport-shell">
      <div class="gs-mini-viewport"></div>
      <div class="gs-mini-hud"><span class="gsm-model-name">No gripper loaded</span></div>
    </div>
  </div>
`;

export function mountGripperStudio(container, options = {}) {
  if (!container) return null;
  if (container.dataset.gripperStudioMounted === "1") return null;
  container.dataset.gripperStudioMounted = "1";

  container.classList.add("gripper-studio-host");
  container.innerHTML = HTML;

  const viewportEl = container.querySelector(".gs-mini-viewport");
  const statusEl = container.querySelector("#gsm-load-status");
  const nameEl = container.querySelector("#gsm-current-name");
  const modelNameEl = container.querySelector(".gsm-model-name");
  const saveBtn = container.querySelector(".gsm-save");
  const reloadBtn = container.querySelector(".gsm-reload");
  const assetPicker = container.querySelector("#gsm-asset-picker");
  const assetPickerWrap = container.querySelector(".gsm-asset-picker-wrap");
  const strokeWidth = container.querySelector("#gsm-stroke-width");
  const strokeSlider = container.querySelector("#gsm-stroke-slider");

  const vp = createSceneViewport(viewportEl, { background: 0x101820 });
  vp.controls.target.set(0, 0, 0.05);
  vp.camera.position.set(0.45, -0.65, 0.45);
  vp.controls.update();

  const editor = createFrameEditor({
    scene: vp.scene,
    camera: vp.camera,
    renderer: vp.renderer,
    controls: vp.controls,
    container,
  });

  const state = {
    processId: options.processId || null,
    gripperInfo: null,
    model: null,
    availableGrippers: [],
    strokeWidth: 70,
    maxStroke: 160,
  };

  function setStatus(text, isError = false) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("err", !!isError);
  }

  function replaceModel(obj, name) {
    if (state.model) vp.scene.remove(state.model);
    state.model = obj;
    if (obj) {
      vp.scene.add(obj);
      vp.frameObject(obj);
      editor.resizeForBounds(new THREE.Box3().setFromObject(obj));
      applyGripperStroke(state.strokeWidth);
    }
    modelNameEl.textContent = name || "No gripper loaded";
  }

  async function loadAsset(processId) {
    state.processId = processId;
    if (!processId) {
      await loadAvailableGrippers();
      return;
    }
    setStatus("Loading gripper…");
    try {
      const res = await fetch(
        `/processes/${encodeURIComponent(processId)}/bin-picking/assets`,
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const gripper = data?.gripper;
      if (!gripper) {
        await loadAvailableGrippers();
        return;
      }
      await loadGripperInfo(gripper, processId);
    } catch (err) {
      console.error(err);
      setStatus(`Load failed: ${err.message}`, true);
    }
  }

  async function loadAvailableGrippers(preferredAssetId = "") {
    setStatus("Scanning data/stations for gripper assets…");
    try {
      const res = await fetch("/bin-picking/gripper-assets", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.availableGrippers = Array.isArray(data?.grippers) ? data.grippers : [];
      renderAssetPicker(preferredAssetId);
      const chosen =
        state.availableGrippers.find((item) => item.asset_id === preferredAssetId) ||
        state.availableGrippers[0];
      if (!chosen) {
        setStatus("No gripper folders found under data/stations.", true);
        replaceModel(null);
        return;
      }
      await loadGripperInfo(chosen, chosen.asset_id || state.processId);
    } catch (err) {
      console.error(err);
      setStatus(`Gripper scan failed: ${err.message}`, true);
    }
  }

  function renderAssetPicker(preferredAssetId = "") {
    if (!assetPicker || !assetPickerWrap) return;
    assetPicker.replaceChildren();
    state.availableGrippers.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.asset_id || "";
      opt.textContent = `${item.asset_id || "asset"} / ${item.name || item.id || "gripper"}`;
      assetPicker.appendChild(opt);
    });
    assetPickerWrap.hidden = state.availableGrippers.length <= 1;
    if (preferredAssetId) assetPicker.value = preferredAssetId;
  }

  async function loadGripperInfo(gripper, processIdForFrames = "") {
    state.processId = processIdForFrames || state.processId;
    state.gripperInfo = gripper;
    nameEl.textContent = gripper.name || gripper.id || "Gripper";
    if (assetPicker && gripper.asset_id) assetPicker.value = gripper.asset_id;
    if (gripper.urdf_url) {
      const obj = await loadUrdf(gripper.urdf_url);
      replaceModel(obj, gripper.name || gripper.id);
    } else {
      replaceModel(null);
      setStatus("Gripper has no URDF - frames editable, no preview.");
    }
    const frames =
      gripper.frames ||
      (state.processId ? await fetchFrames(state.processId) : null) ||
      defaultFrames();
    const maxStroke = Number(gripper.manifest?.stroke_width ?? 160);
    state.maxStroke = Number.isFinite(maxStroke) && maxStroke > 0 ? Math.max(maxStroke, 1) : 160;
    setStrokeWidth(Number(frames.strokeWidth ?? gripper.manifest?.stroke_width ?? state.strokeWidth ?? 70), false);
    editor.setFrames(frames);
    setStatus(
      gripper.has_frames
        ? `Loaded ${gripper.name || gripper.id || "gripper"} from ${gripper.asset_path || "asset folder"}.`
        : "No saved frames - start at default."
    );
    pushTcpToCalibration(frames.tcp);
    pushFramesToCalibration(frames);
  }

  async function fetchFrames(processId) {
    try {
      const res = await fetch(
        `/processes/${encodeURIComponent(processId)}/bin-picking/assets/gripper/frames`
      );
      if (!res.ok) return null;
      const { frames } = await res.json();
      return frames;
    } catch {
      return null;
    }
  }

  async function save() {
    if (!state.processId) {
      setStatus("No process selected.", true);
      return;
    }
    const frames = {
      ...editor.getFrames(),
      strokeWidth: Number(state.strokeWidth || 0),
    };
    setStatus("Saving frames…");
    try {
      const res = await fetch(
        `/processes/${encodeURIComponent(state.processId)}/bin-picking/assets/gripper/frames`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frames }),
        }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatus("Saved frames to gripper asset.");
      pushTcpToCalibration(frames.tcp);
      pushFramesToCalibration(frames);
      if (typeof options.onSave === "function") options.onSave(frames);
    } catch (err) {
      setStatus(`Save failed: ${err.message}`, true);
    }
  }

  function pushTcpToCalibration(tcp) {
    if (typeof options.onTcpChange === "function" && tcp) {
      options.onTcpChange({
        position_mm: tcp.position,
        rotation_deg: tcp.rotation,
        position_m: [
          (tcp.position[0] || 0) / 1000,
          (tcp.position[1] || 0) / 1000,
          (tcp.position[2] || 0) / 1000,
        ],
      });
    }
  }

  function pushFramesToCalibration(frames) {
    if (typeof options.onFramesChange === "function" && frames) {
      options.onFramesChange(frames);
    }
  }

  editor.onChange((frames) => {
    pushTcpToCalibration(frames.tcp);
    pushFramesToCalibration(frames);
  });

  function setStrokeWidth(value, updateInputs = true) {
    const clamped = Math.max(0, Math.min(Number(state.maxStroke || 160), Number(value) || 0));
    state.strokeWidth = clamped;
    if (updateInputs) {
      strokeWidth.value = clamped.toFixed(1);
      strokeSlider.value = String(clamped);
    } else {
      strokeWidth.max = String(state.maxStroke);
      strokeSlider.max = String(state.maxStroke);
      strokeWidth.value = clamped.toFixed(1);
      strokeSlider.value = String(clamped);
    }
    applyGripperStroke(clamped);
  }

  function applyGripperStroke(strokeMm) {
    const urdf = state.model?.userData?.urdf;
    if (!urdf?.actuators?.length || typeof urdf.setJointValue !== "function") return;
    const perFingerM = Math.max(0, Math.min(Number(state.maxStroke || 160), Number(strokeMm) || 0)) / 2000.0;
    urdf.actuators.forEach((joint) => {
      urdf.setJointValue(joint.name, perFingerM);
    });
  }

  strokeWidth.addEventListener("input", () => setStrokeWidth(strokeWidth.value));
  strokeWidth.addEventListener("change", () => setStrokeWidth(strokeWidth.value));
  strokeSlider.addEventListener("input", () => setStrokeWidth(strokeSlider.value));
  strokeSlider.addEventListener("change", () => setStrokeWidth(strokeSlider.value));
  saveBtn.addEventListener("click", save);
  reloadBtn.addEventListener("click", () => loadAsset(state.processId));
  assetPicker?.addEventListener("change", async () => {
    const chosen = state.availableGrippers.find((item) => item.asset_id === assetPicker.value);
    if (chosen) await loadGripperInfo(chosen, chosen.asset_id || state.processId);
  });
  container.querySelectorAll('input[name="gsm-edit-frame"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.checked) editor.setActiveFrame(radio.value);
    });
    radio.closest(".gs-edit-toggle")?.addEventListener("click", () => {
      radio.checked = true;
      editor.setActiveFrame(radio.value);
    });
  });
  container.querySelectorAll(".gsm-mode button").forEach((btn) => {
    btn.addEventListener("click", () => {
      container.querySelectorAll(".gsm-mode button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });

  function defaultFrames() {
    return {
      flange: { position: [0, 0, 0], rotation: [0, 0, 0] },
      tcp: { position: [0, 0, 0], rotation: [0, 0, 0] },
      strokeWidth: state.strokeWidth,
    };
  }

  if (state.processId) {
    loadAsset(state.processId);
  }

  return {
    loadAsset,
    save,
    getFrames: () => editor.getFrames(),
    setFrames: (f) => editor.setFrames(f),
    dispose: () => {
      editor.dispose?.();
      vp.dispose?.();
      container.dataset.gripperStudioMounted = "";
      container.innerHTML = "";
    },
  };
}
