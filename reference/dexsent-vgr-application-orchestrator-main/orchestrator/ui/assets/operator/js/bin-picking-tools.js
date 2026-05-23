import * as THREE from "three";
import { createSceneViewport } from "../bin-picking/shared/three-scene.js";
import { loadUrdf } from "../bin-picking/shared/urdf-loader.js";
import { loadCadLikeFile } from "../bin-picking/shared/cad-loader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const KIND_LABEL = {
  robot: "Robot",
  gripper: "Gripper",
};

const cadObjectCache = new Map();

// Read a global only if it's actually a string. On index.html, `window.currentTaskType`
// is a *function*, so a naive `window.currentTaskType || ...` would yield garbage.
function gstr(name) {
  const v = window[name];
  return typeof v === "string" ? v : "";
}

function context() {
  const shellState = window.operatorShell?.state || {};
  const selectedTaskId = document.getElementById("taskSelect")?.value || "";
  const selectedTaskType = document.getElementById("taskTypeSelect")?.value || "";
  if (window.operatorShell && typeof window.operatorShell.getContext === "function") {
    return {
      ...window.operatorShell.getContext(),
      taskId: shellState.currentTaskId || gstr("currentTaskId") || selectedTaskId || "",
      taskType: shellState.currentTaskType || gstr("currentTaskType") || selectedTaskType || "",
      runId: shellState.currentRunId || gstr("currentRunId") || "",
    };
  }
  const assetId = gstr("currentProcessId") || gstr("currentAssetId") || "";
  const stationId = gstr("currentStationId") || "station-1";
  return {
    stationId,
    assetId,
    // Prefer the live DOM selects (index.html keeps these authoritative).
    taskId: selectedTaskId || gstr("currentTaskId") || "",
    taskType: selectedTaskType || gstr("currentTaskType") || "",
    runId: gstr("currentRunId") || "",
  };
}

function isDummyTestingTask() {
  const ctx = context();
  return String(ctx.taskType || "").trim().toLowerCase() === "dummy_testing";
}

function robotVizObjectQuery(ctx) {
  const resolved = ctx || context();
  const taskType = String(resolved.taskType || "").trim().toLowerCase();
  const taskId = String(resolved.taskId || "").trim();
  const params = new URLSearchParams();
  if (taskType === "dummy_testing") params.set("object_id", "dummy_testing_none");
  if (taskType) params.set("task_type", taskType);
  if (taskId) params.set("task_id", taskId);
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

function api(path, options = {}, cfg = {}) {
  if (window.operatorApi) return window.operatorApi(path, options, cfg);
  return fetch(path, options).then(async (res) => ({
    ok: res.ok,
    body: await res.json().catch(() => null),
  }));
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  });
  children.forEach((child) => node.appendChild(typeof child === "string" ? document.createTextNode(child) : child));
  return node;
}

function emptyTool(title, message) {
  const shell = el("div", { class: "tool-empty" }, [
    el("h3", { text: title }),
    el("p", { text: message }),
  ]);
  window.operatorModal.open({ title, content: shell });
}

async function openSelection(kind) {
  const ctx = context();
  if (!ctx.assetId) {
    emptyTool(`${KIND_LABEL[kind]} Selection`, "Select an asset first.");
    return;
  }
  const body = el("div", { class: "asset-selection-tool" }, [
    el("div", { class: "hint", text: "Loading catalog..." }),
  ]);
  window.operatorModal.open({ title: `${KIND_LABEL[kind]} Selection`, content: body });

  const catalogRes = await api("/bin-picking/catalog", {}, { silent: true });
  body.replaceChildren();
  if (!catalogRes.ok || !catalogRes.body) {
    body.appendChild(el("div", { class: "hint", text: "Catalog unavailable." }));
    return;
  }

  const items = kind === "robot" ? catalogRes.body.robots || [] : catalogRes.body.grippers || [];
  if (!items.length) {
    body.appendChild(el("div", { class: "hint", text: `No ${kind} catalog items found.` }));
    return;
  }

  const grid = el("div", { class: "tool-card-grid" });
  items.forEach((item) => {
    const manifest = item.manifest || {};
    const card = el("article", { class: "tool-card" }, [
      el("h3", { text: item.name || item.id || item.catalog_path }),
      el("p", { text: [manifest.manufacturer, manifest.model, manifest.family].filter(Boolean).join(" | ") || item.catalog_path }),
      el("button", {
        class: "primary",
        type: "button",
        text: `Use ${KIND_LABEL[kind]}`,
        onclick: async () => {
          const res = await api(
            `/processes/${encodeURIComponent(ctx.assetId)}/bin-picking/assets/${kind}/select`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ catalog_path: item.catalog_path }),
            }
          );
          if (!res.ok) {
            card.appendChild(el("div", { class: "hint", text: "Selection failed." }));
            return;
          }
          openRobotVisualizer();
        },
      }),
    ]);
    grid.appendChild(card);
  });
  body.appendChild(grid);
}

async function loadDataGrippers() {
  const res = await api("/bin-picking/gripper-assets", {}, { silent: true });
  if (!res.ok || !res.body) return [];
  return Array.isArray(res.body.grippers) ? res.body.grippers : [];
}

function addGripperAssetPicker(parent, grippers, onSelect) {
  const wrap = el("div", { class: "asset-picker" });
  const select = el("select");
  grippers.forEach((item) => {
    select.appendChild(
      el("option", {
        value: `${item.asset_id || ""}`,
        text: `${item.asset_id || item.name} - ${item.name || "gripper"}`,
      })
    );
  });
  const loadBtn = el("button", {
    class: "secondary",
    type: "button",
    text: "Load Gripper",
    onclick: () => {
      const chosen = grippers.find((item) => String(item.asset_id || "") === select.value) || grippers[0];
      if (chosen) onSelect(chosen);
    },
  });
  wrap.appendChild(select);
  wrap.appendChild(loadBtn);
  parent.appendChild(wrap);
  if (grippers.length === 1) onSelect(grippers[0]);
}

function addAxisMarker(scene) {
  const axes = new THREE.AxesHelper(0.25);
  axes.name = "world_axes";
  scene.add(axes);
}

// FR3 gripper finger joint names and per-finger stroke limit (m).
const FR3_FINGER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"];
const FR3_FINGER_MAX_M = 0.04; // each finger travels 0–0.04 m; total width = 2× finger pos

function applyRobotState(root, robotState) {
  const urdf = root && root.userData ? root.userData.urdf : null;
  if (!urdf || typeof urdf.setJointValue !== "function") return;
  const joints = robotState && (robotState.joints || robotState.joint_positions || robotState.joint_state || robotState.q);
  const applied = {};
  if (Array.isArray(joints)) {
    (urdf.actuators || []).forEach((joint, idx) => {
      if (idx < joints.length) {
        const value = Number(joints[idx]) || 0;
        urdf.setJointValue(joint.name, value);
        applied[joint.name] = value;
      }
    });
    updateJointControls(root, applied);
  } else if (joints && typeof joints === "object") {
    Object.entries(joints).forEach(([name, value]) => {
      const numeric = Number(value) || 0;
      urdf.setJointValue(name, numeric);
      applied[name] = numeric;
    });
    updateJointControls(root, applied);
  }
  // Drive gripper fingers from real gripper_width_m (total width = 2× per-finger travel).
  if (robotState && robotState.gripper_width_m != null) {
    const fingerPos = Math.max(0, Math.min(FR3_FINGER_MAX_M, robotState.gripper_width_m / 2));
    FR3_FINGER_JOINTS.forEach((name) => urdf.setJointValue(name, fingerPos));
  } else if (robotState) {
    // Fallback: use gripper_open bool — open = max, closed = 0
    const fingerPos = robotState.gripper_open ? FR3_FINGER_MAX_M : 0;
    FR3_FINGER_JOINTS.forEach((name) => urdf.setJointValue(name, fingerPos));
  }
}

async function openRobotVisualizer() {
  const ctx = context();
  if (!ctx.assetId) {
    emptyTool("Robot Digital Twin", "Select an asset first.");
    return;
  }
  await _buildRobotViz(ctx, null);
}

async function mountRobotVisualizer(mountEl) {
  const ctx = context();
  if (!ctx.assetId) return;
  if (mountEl) mountEl.replaceChildren();
  await _buildRobotViz(ctx, mountEl);
}

async function _buildRobotViz(ctx, mountEl) {
  const isInline = !!mountEl;

  const toolbarChildren = isInline
    ? [
        // Run-tab inline: just reload + asset label. No engineering controls.
        el("button", { class: "ghost", type: "button", text: "Reload", onclick: () => mountRobotVisualizer(mountEl) }),
        el("span", { class: "hint", text: `Digital twin · ${ctx.assetId}` }),
        // Hidden placeholders so the rest of the wiring code can still querySelector them.
        el("button", { id: "robotVizSync", style: "display:none", type: "button" }),
        el("button", { id: "robotVizCheck", style: "display:none", type: "button" }),
        el("button", { id: "robotVizPlan", style: "display:none", type: "button" }),
        el("select", { id: "robotVizObstacleSelect", style: "display:none" }),
        el("button", { id: "robotVizObstacleMove", style: "display:none", type: "button" }),
        el("button", { id: "robotVizObstacleRotate", style: "display:none", type: "button" }),
        el("button", { id: "robotVizObstacleSave", style: "display:none", type: "button" }),
        el("label", { style: "display:none" }, [
          el("input", { type: "file", id: "robotVizObstacleFile", accept: ".stl,.obj,.dae,.ply,.glb,.gltf", style: "display:none" }),
        ]),
      ]
    : [
        // Popup engineering view: full toolbar.
        el("button", { class: "secondary", type: "button", text: "Select Robot", onclick: () => openSelection("robot") }),
        el("button", { class: "secondary", type: "button", text: "Select Gripper", onclick: () => openSelection("gripper") }),
        el("button", { class: "secondary", type: "button", text: "Sync Live", id: "robotVizSync" }),
        el("button", { class: "secondary", type: "button", text: "Check Collision", id: "robotVizCheck" }),
        el("button", { class: "secondary", type: "button", text: "Plan To Grasp", id: "robotVizPlan" }),
        el("select", { id: "robotVizObstacleSelect", title: "Collision obstacle" }),
        el("button", { class: "secondary", type: "button", text: "Move Obstacle", id: "robotVizObstacleMove" }),
        el("button", { class: "secondary", type: "button", text: "Rotate Obstacle", id: "robotVizObstacleRotate" }),
        el("button", { class: "primary", type: "button", text: "Save Obstacle Pose", id: "robotVizObstacleSave" }),
        el("label", { class: "secondary file-button" }, [
          document.createTextNode("Import Collision CAD"),
          el("input", { type: "file", id: "robotVizObstacleFile", accept: ".stl,.obj,.dae,.ply,.glb,.gltf", style: "display:none" }),
        ]),
        el("button", { class: "ghost", type: "button", text: "Reload", onclick: () => openRobotVisualizer() }),
        el("span", { class: "hint", text: `Digital twin | Asset ${ctx.assetId}` }),
      ];

  const shell = el("div", { class: "robot-viz-tool" }, [
    el("div", { class: "robot-viz-toolbar" }, toolbarChildren),
    el("div", { class: "robot-viz-layout" }, [
      el("div", { class: "robot-viz-viewport", id: "robotVizViewport" }),
      el("aside", { class: "robot-viz-side", id: "robotVizSide" }, [
        el("div", { class: "robot-viz-side-toggle", id: "robotVizSideToggle", title: "Toggle panel" }, [
          el("span", { text: "‹" }),
        ]),
        el("div", { class: "robot-viz-side-body" }, [
        el("h3", { text: "Run Status" }),
        el("div", { class: "robot-viz-runstatus", id: "robotVizRunStatus",
                    style: "font-size:12px;line-height:1.5;padding:6px 8px;border-radius:4px;background:rgba(255,255,255,0.05)" },
           [ el("div", { id: "robotVizRunStage", style: "font-weight:600", text: "—" }),
             el("div", { id: "robotVizRunDetail", style: "opacity:0.8", text: "no active run" }) ]),
        el("h3", { text: "Displays" }),
        el("label", { class: "check" }, [checkboxPersist("robotVizVisual", true), document.createTextNode(" Robot visual meshes")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizCollision", false), document.createTextNode(" Robot collision meshes")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizObstacles", true), document.createTextNode(" Imported obstacle meshes")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizTf", true), document.createTextNode(" TF frames")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizGrid", true), document.createTextNode(" Grid")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizTcp", true), document.createTextNode(" TCP frame")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizCamera", true), document.createTextNode(" Camera frame")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizDetectedPoses", true), document.createTextNode(" Detected object poses")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizPointCloud", true), document.createTextNode(" Point cloud")]),
        el("label", { class: "check" }, [checkboxPersist("robotVizCyclePath", true), document.createTextNode(" Cycle path")]),
        el("div", { class: "robot-viz-legend" }, [
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#ffaa00" }), document.createTextNode(" TCP (wrt flange)")]),
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#44ddff" }), document.createTextNode(" Camera (wrt flange)")]),
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#ffd166" }), document.createTextNode(" Planned TCP path")]),
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#24d17e" }), document.createTextNode(" Robot-engine path")]),
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#e040fb" }), document.createTextNode(" Detected object poses")]),
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#76ff03" }), document.createTextNode(" Point cloud")]),
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#ff9800" }), document.createTextNode(" Cycle path")]),
          el("div", { class: "robot-viz-legend-row" }, [el("span", { class: "robot-viz-swatch", style: "background:#ff4444" }), document.createTextNode(" X"), el("span", { class: "robot-viz-swatch", style: "background:#44cc66;margin-left:4px" }), document.createTextNode(" Y"), el("span", { class: "robot-viz-swatch", style: "background:#4488ff;margin-left:4px" }), document.createTextNode(" Z")]),
        ]),
        el("h3", { text: "Obstacle" }),
        el("div", { class: "robot-viz-obstacle", id: "robotVizObstaclePanel", style: "display:flex;flex-direction:column;gap:6px;font-size:11px" }, [
          el("select", { id: "robotVizObstacleSelectPanel", title: "Collision obstacle", style: "width:100%" }),
          el("label", { style: "display:flex;align-items:center;gap:6px" }, [checkbox("robotVizObstacleEdit", false), document.createTextNode(" Edit pose (unchecked = locked)")]),
          el("div", { style: "display:flex;gap:4px" }, [
            el("button", { type: "button", id: "robotVizObstacleModeTranslate", text: "Translate gizmo", style: "flex:1;font-size:10px" }),
            el("button", { type: "button", id: "robotVizObstacleModeRotate", text: "Rotate gizmo", style: "flex:1;font-size:10px" }),
          ]),
          el("div", { style: "display:grid;grid-template-columns:auto 1fr auto 1fr;gap:3px 6px;align-items:center" }, [
            el("span", { text: "X" }), el("input", { type: "number", step: "0.001", id: "robotVizObsX", style: "width:100%" }),
            el("span", { text: "R°" }), el("input", { type: "number", step: "0.5", id: "robotVizObsR", style: "width:100%" }),
            el("span", { text: "Y" }), el("input", { type: "number", step: "0.001", id: "robotVizObsY", style: "width:100%" }),
            el("span", { text: "P°" }), el("input", { type: "number", step: "0.5", id: "robotVizObsP", style: "width:100%" }),
            el("span", { text: "Z" }), el("input", { type: "number", step: "0.001", id: "robotVizObsZ", style: "width:100%" }),
            el("span", { text: "Yaw°" }), el("input", { type: "number", step: "0.5", id: "robotVizObsYaw", style: "width:100%" }),
          ]),
          el("button", { type: "button", id: "robotVizObstacleSavePanel", text: "Save to scene.yaml", style: "font-size:10px" }),
        ]),
        el("h3", { text: "Joints" }),
        el("div", { class: "robot-viz-joints", id: "robotVizJoints" }),
        el("h3", { text: "Live" }),
        el("div", { class: "robot-viz-live", id: "robotVizLive", text: "Waiting for robot state..." }),
        el("h3", { text: "Planning" }),
        el("canvas", { class: "robot-viz-plot", id: "robotVizPlot", width: "260", height: "150" }),
        el("div", { class: "robot-viz-planning", id: "robotVizPlanning", text: "Waiting for planning events..." }),
        el("h3", { text: "Robot Engine" }),
        el("div", { class: "mini-log", id: "robotVizInfo", text: "Loading..." }),
        ]), // end robot-viz-side-body
      ]),
    ]),
  ]);

  if (isInline) {
    mountEl.appendChild(shell);
  } else {
    window.operatorModal.open({ title: "Robot Digital Twin", content: shell, className: "wide" });
  }

  const viewport = shell.querySelector("#robotVizViewport");
  const info = shell.querySelector("#robotVizInfo");
  const jointsEl = shell.querySelector("#robotVizJoints");
  const liveEl = shell.querySelector("#robotVizLive");
  const planningEl = shell.querySelector("#robotVizPlanning");
  const runStageEl = shell.querySelector("#robotVizRunStage");
  const runDetailEl = shell.querySelector("#robotVizRunDetail");
  const plotCanvas = shell.querySelector("#robotVizPlot");
  const vp = createSceneViewport(viewport, { background: 0x111820 });
  addAxisMarker(vp.scene);
  const obstacleTransform = new TransformControls(vp.camera, vp.renderer.domElement);
  obstacleTransform.setSize(0.9);
  obstacleTransform.setSpace("local");
  obstacleTransform.setMode("translate");
  obstacleTransform.addEventListener("dragging-changed", (event) => {
    vp.controls.enabled = !event.value;
  });
  obstacleTransform.addEventListener("objectChange", () => {
    syncSelectedObstacleFromObject();
    refreshPanelFieldsFromObject();
  });
  vp.scene.add(obstacleTransform);
  const sceneGroups = {
    robot: null,
    gripper: null,
    object: null,
    tcpFrame: null,
    cameraFrame: null,
    obstacles: new THREE.Group(),
    grasps: new THREE.Group(),
    trajectory: new THREE.Group(),
    plannedPath: new THREE.Group(),
    detectedPoses: new THREE.Group(),
    pointCloud: new THREE.Group(),
    cyclePath: new THREE.Group(),
  };
  vp.scene.add(
    sceneGroups.obstacles, sceneGroups.grasps, sceneGroups.trajectory, sceneGroups.plannedPath,
    sceneGroups.detectedPoses, sceneGroups.pointCloud, sceneGroups.cyclePath
  );
  let pollTimer = null;
  let evalTimer = null;
  let timelineTimer = null;
  let visionTimer = null;
  let runStatusTickTimer = null;
  let stateEventSource = null;
  let currentDummyObstacles = [];
  let activeScene = null;
  // Current bin-picking stage for the Run Status strip: {stage, detail, sinceMs, runId}.
  let runStatus = { stage: "—", detail: "no active run", sinceMs: 0, runId: "" };
  // Camera frame is rigidly attached to the (moving) robot TCP. Detected object
  // poses / point clouds are captured at the capture pose and are static in the
  // world — so we snapshot T_world_camera once per vision capture and freeze it,
  // instead of re-reading the live (moving) camera frame on every redraw.
  let frozenCameraMatrix = null;
  let liveInFlight = false;
  let evalInFlight = false;
  let timelineInFlight = false;
  let visionInFlight = false;
  let lastTimelineSignature = "";
  let lastVisionSignature = "";
  let lastVisionCycle = -1;
  let handeye = null;

  const cleanup = () => {
    if (pollTimer) clearInterval(pollTimer);
    if (evalTimer) clearInterval(evalTimer);
    if (timelineTimer) clearInterval(timelineTimer);
    if (visionTimer) clearInterval(visionTimer);
    if (runStatusTickTimer) clearInterval(runStatusTickTimer);
    if (stateEventSource) { stateEventSource.close(); stateEventSource = null; }
    obstacleTransform.detach();
    obstacleTransform.dispose?.();
    vp.dispose?.();
  };
  if (!isInline) {
    const oldClose = window.operatorModal.close;
    window.operatorModal.close = () => {
      cleanup();
      window.operatorModal.close = oldClose;
      oldClose();
    };
  }

  const displayControls = {
    visual: shell.querySelector("#robotVizVisual"),
    collision: shell.querySelector("#robotVizCollision"),
    obstacles: shell.querySelector("#robotVizObstacles"),
    tf: shell.querySelector("#robotVizTf"),
    grid: shell.querySelector("#robotVizGrid"),
    tcp: shell.querySelector("#robotVizTcp"),
    camera: shell.querySelector("#robotVizCamera"),
    detectedPoses: shell.querySelector("#robotVizDetectedPoses"),
    pointCloud: shell.querySelector("#robotVizPointCloud"),
    cyclePath: shell.querySelector("#robotVizCyclePath"),
  };

  Object.values(displayControls).forEach((input) => input?.addEventListener("change", () => {
    // Persist each checkbox state to localStorage when changed.
    if (input?.id) localStorage.setItem(`rviz_cb_${input.id}`, input.checked ? "1" : "0");
    applyDisplayFlags(sceneGroups, displayControls, vp);
  }));

  // Collapsible side panel with persisted state.
  const sidePanel = shell.querySelector("#robotVizSide");
  const sideToggle = shell.querySelector("#robotVizSideToggle");
  const layoutDiv = shell.querySelector(".robot-viz-layout");
  const SIDE_KEY = "rviz_side_collapsed";
  function setSideCollapsed(collapsed) {
    if (sidePanel) sidePanel.classList.toggle("collapsed", collapsed);
    if (sideToggle) sideToggle.querySelector("span").textContent = collapsed ? "›" : "‹";
    localStorage.setItem(SIDE_KEY, collapsed ? "1" : "0");
  }
  // Default to collapsed (overlay panel shouldn't open by default)
  const storedSide = localStorage.getItem(SIDE_KEY);
  setSideCollapsed(storedSide === null ? true : storedSide === "1");
  sideToggle?.addEventListener("click", () => setSideCollapsed(!sidePanel?.classList.contains("collapsed")));

  shell.querySelector("#robotVizSync")?.addEventListener("click", syncLiveState);
  shell.querySelector("#robotVizCheck")?.addEventListener("click", runEvaluation);
  shell.querySelector("#robotVizPlan")?.addEventListener("click", runEvaluation);
  shell.querySelector("#robotVizObstacleFile")?.addEventListener("change", uploadObstacleCad);
  shell.querySelector("#robotVizObstacleSelect")?.addEventListener("change", () => { onObstacleSelectionChanged(); });
  shell.querySelector("#robotVizObstacleMove")?.addEventListener("click", () => {
    obstacleTransform.setMode("translate");
    attachSelectedObstacle();
  });
  shell.querySelector("#robotVizObstacleRotate")?.addEventListener("click", () => {
    obstacleTransform.setMode("rotate");
    attachSelectedObstacle();
  });
  shell.querySelector("#robotVizObstacleSave")?.addEventListener("click", saveObstacleScene);

  // --- Obstacle pose panel (X/Y/Z + R/P/Y, edit lock, gizmo mode) ---
  const obsFieldIds = ["robotVizObsX", "robotVizObsY", "robotVizObsZ", "robotVizObsR", "robotVizObsP", "robotVizObsYaw"];
  const obsEditCheck = shell.querySelector("#robotVizObstacleEdit");
  shell.querySelector("#robotVizObstacleSelectPanel")?.addEventListener("change", () => { onObstacleSelectionChanged(); });
  obsEditCheck?.addEventListener("change", () => { setObstacleEditState(!!obsEditCheck.checked); });
  obsFieldIds.forEach((id) => {
    shell.querySelector(`#${id}`)?.addEventListener("input", () => { applyPanelFieldsToObject(); });
  });
  shell.querySelector("#robotVizObstacleModeTranslate")?.addEventListener("click", () => {
    obstacleTransform.setMode("translate");
    if (obsEditCheck) { obsEditCheck.checked = true; }
    setObstacleEditState(true);
  });
  shell.querySelector("#robotVizObstacleModeRotate")?.addEventListener("click", () => {
    obstacleTransform.setMode("rotate");
    if (obsEditCheck) { obsEditCheck.checked = true; }
    setObstacleEditState(true);
  });
  shell.querySelector("#robotVizObstacleSavePanel")?.addEventListener("click", saveObstacleScene);

  function obstacleEditEnabled() {
    return !!obsEditCheck?.checked;
  }

  function onObstacleSelectionChanged() {
    // Keep both selects in sync.
    const panelSel = shell.querySelector("#robotVizObstacleSelectPanel");
    const toolbarSel = shell.querySelector("#robotVizObstacleSelect");
    const value = panelSel?.value || toolbarSel?.value || "";
    if (panelSel && panelSel.value !== value) panelSel.value = value;
    if (toolbarSel && toolbarSel.value !== value) toolbarSel.value = value;
    refreshPanelFieldsFromObject();
    attachSelectedObstacle();
  }

  function setObstacleFieldsReadonly(readonly) {
    obsFieldIds.forEach((id) => {
      const inp = shell.querySelector(`#${id}`);
      if (inp) inp.readOnly = !!readonly;
    });
    const savePanel = shell.querySelector("#robotVizObstacleSavePanel");
    if (savePanel) savePanel.disabled = !!readonly;
    ["robotVizObstacleModeTranslate", "robotVizObstacleModeRotate"].forEach((id) => {
      const b = shell.querySelector(`#${id}`);
      if (b) b.disabled = !!readonly;
    });
  }

  function setObstacleEditState(editing) {
    setObstacleFieldsReadonly(!editing);
    if (editing) {
      attachSelectedObstacle();
    } else {
      obstacleTransform.detach();
    }
    console.log(`[robot-viz] obstacle edit ${editing ? "ENABLED" : "LOCKED"}`);
  }

  function refreshPanelFieldsFromObject() {
    const object = selectedObstacleObject();
    const vals = object
      ? [object.position.x, object.position.y, object.position.z,
         THREE.MathUtils.radToDeg(object.rotation.x), THREE.MathUtils.radToDeg(object.rotation.y), THREE.MathUtils.radToDeg(object.rotation.z)]
      : [0, 0, 0, 0, 0, 0];
    obsFieldIds.forEach((id, i) => {
      const inp = shell.querySelector(`#${id}`);
      if (inp && document.activeElement !== inp) inp.value = roundPoseNumber(vals[i]);
    });
  }

  function applyPanelFieldsToObject() {
    if (!obstacleEditEnabled()) return;
    const object = selectedObstacleObject();
    if (!object) return;
    const num = (id) => Number(shell.querySelector(`#${id}`)?.value || 0) || 0;
    object.position.set(num("robotVizObsX"), num("robotVizObsY"), num("robotVizObsZ"));
    object.rotation.set(
      num("robotVizObsR") * Math.PI / 180,
      num("robotVizObsP") * Math.PI / 180,
      num("robotVizObsYaw") * Math.PI / 180,
      "XYZ"
    );
    syncSelectedObstacleFromObject();
  }

  try {
    const sceneQuery = robotVizObjectQuery(ctx);
    console.log(`[robot-viz] loading scene: asset=${ctx.assetId} taskId=${ctx.taskId || "(none)"} taskType=${ctx.taskType || "(none)"} query=${sceneQuery || "(none)"}`);
    const sceneRes = await api(`/processes/${encodeURIComponent(ctx.assetId)}/robot-digital-twin/scene${sceneQuery}`, {}, { silent: true });
    if (!sceneRes.ok || !sceneRes.body?.scene) throw new Error("digital twin scene unavailable");
    const scene = sceneRes.body.scene;
    activeScene = scene;
    const meta = sceneRes.body.meta || {};
    console.log(`[robot-viz] scene loaded: scene_path=${meta.dummy_testing?.scene_path || "(task-type default)"} obstacles=${JSON.stringify((meta.dummy_testing?.obstacles || []).map((o) => ({ id: o.id, mesh: o.mesh, url: o.url })))} collision_objects=${(scene.collision_objects || []).length}`);
    const robot = meta.robot;
    const gripper = meta.gripper;
    const loaded = [];

    if (robot && robot.urdf_url) {
      const robotRoot = await loadUrdf(robot.urdf_url, { loadCollision: true });
      robotRoot.name = "selected_robot";
      vp.scene.add(robotRoot);
      sceneGroups.robot = robotRoot;
      loaded.push(`robot: ${robot.name || robot.id}`);
      buildJointControls(robotRoot, jointsEl);
      applyRobotState(robotRoot, meta.robot_state || {});
    } else {
      loaded.push("robot: not selected");
    }

    if (gripper && gripper.urdf_url) {
      const gripperRoot = await loadUrdf(gripper.urdf_url, { loadCollision: true });
      gripperRoot.name = "selected_gripper";
      attachOrPlaceGripper(sceneGroups.robot, gripperRoot, robot?.manifest, gripper?.frames);
      sceneGroups.gripper = gripperRoot;
      loaded.push(`gripper: ${gripper.name || gripper.id}`);

      // TCP frame marker — wrt flange, so child of gripperRoot at position 0,0,0 of flange.
      const tcpFrameData = gripper?.frames?.tcp || null;
      sceneGroups.tcpFrame = drawFlangeFrame(gripperRoot, tcpFrameData, 0xffaa00, "tcp_frame");

      // Camera frame — from handeye calibration.
      // Handeye is T_custom_tcp_camera: camera expressed in the custom TCP frame.
      // The custom TCP frame in the viz is sceneGroups.tcpFrame (child of gripperRoot
      // which already applies flange.rotation, then tcp.position on top).
      // Parent the camera frame directly to tcpFrame so the viz matches the solver frame.
      try {
        const heRes = await api("/calibration/handeye", {}, { silent: true });
        if (heRes.ok && heRes.body) {
          handeye = heRes.body;
          const he = heRes.body;
          const camPos = (he.translation_m || [0, 0, 0]).map((v) => Number(v) * 1000); // m → mm
          const camRot = he.rotation_rpy_deg || [0, 0, 0];
          // Parent to TCP frame — handeye is T_custom_tcp_camera.
          const parentGroup = sceneGroups.tcpFrame || gripperRoot;
          sceneGroups.cameraFrame = drawFlangeFrame(
            parentGroup,
            { position: camPos, rotation: camRot },
            0x44ddff,
            "camera_frame"
          );
          loaded.push(`camera frame: loaded (parent: ${parentGroup.name || "tcp_frame"})`);
        } else {
          loaded.push("camera frame: no handeye calibration");
        }
      } catch {
        loaded.push("camera frame: fetch failed");
      }
    }

    // Object CAD is shown per-detection via drawDetectedPoses (vision-detected poses
    // in robot base frame) — not as a static scene object.

    drawGrasps(sceneGroups.grasps, scene.grasp_candidates || []);
    await loadDummyObstacles(meta.dummy_testing?.obstacles || []);
    applyDisplayFlags(sceneGroups, displayControls, vp);
    vp.frameObject(sceneGroups.robot || sceneGroups.object || sceneGroups.gripper || sceneGroups.grasps);
    info.textContent = `${loaded.join("\n")}\nscene objects: ${scene.collision_objects?.length || 0}\ngrasps: ${scene.grasp_candidates?.length || 0}`;
    // Robot state via SSE push — eliminates polling latency.
    // The server streams state changes at ~50 Hz; we apply each event immediately.
    stateEventSource = new EventSource("/robot/state/stream");
    stateEventSource.onmessage = (ev) => {
      if (!sceneGroups.robot) return;
      try {
        const state = JSON.parse(ev.data);
        applyRobotState(sceneGroups.robot, state);
        const q = Array.isArray(state.q) ? state.q : [];
        if (liveEl) {
          liveEl.textContent = [
            `connected: ${state.connected ? "yes" : "no"}`,
            `mode: ${state.mode || state.robot_mode || "n/a"}`,
            `stream: sse`,
            `q: ${q.length ? q.map((v) => Number(v).toFixed(3)).join(", ") : "n/a"}`,
          ].join("\n");
        }
      } catch (_) {}
    };
    stateEventSource.onerror = () => {
      // SSE failed — fall back to polling at 50 ms.
      if (stateEventSource) { stateEventSource.close(); stateEventSource = null; }
      if (!pollTimer) pollTimer = setInterval(syncLiveState, 50);
    };
    evalTimer = setInterval(runEvaluation, 3000);
    timelineTimer = setInterval(refreshPlanningTimeline, 900);
    visionTimer = setInterval(syncVisionResult, 1500);
    runStatusTickTimer = setInterval(renderRunStatus, 200);  // smooth elapsed-seconds between polls
    renderRunStatus();
    await syncLiveState();
    await refreshPlanningTimeline();
    await runEvaluation();
    await syncVisionResult();
  } catch (err) {
    info.textContent = `Visualizer failed: ${err.message || err}`;
  }

  async function syncLiveState() {
    if (!sceneGroups.robot || liveInFlight) return;
    liveInFlight = true;
    const started = performance.now();
    try {
      const robotState = await api("/robot/state", {}, { silent: true });
      if (robotState.ok && robotState.body) {
        applyRobotState(sceneGroups.robot, robotState.body);
        const q = Array.isArray(robotState.body.q) ? robotState.body.q : [];
        const dt = Math.round(performance.now() - started);
        if (liveEl) {
          liveEl.textContent = [
            `connected: ${robotState.body.connected ? "yes" : "no"}`,
            `mode: ${robotState.body.mode || robotState.body.robot_mode || "n/a"}`,
            `latency: ${dt} ms`,
            `q: ${q.length ? q.map((v) => Number(v).toFixed(3)).join(", ") : "n/a"}`,
          ].join("\n");
        }
      } else if (liveEl) {
        liveEl.textContent = "robot state unavailable";
      }
    } catch (err) {
      if (liveEl) liveEl.textContent = `robot state failed: ${err.message || err}`;
    } finally {
      liveInFlight = false;
    }
  }

  async function syncVisionResult() {
    if (visionInFlight) return;
    visionInFlight = true;
    try {
      const res = await api("/vision/latest", {}, { silent: true });
      if (!res.ok || !res.body || res.body.status === "pending") return;

      // /vision/latest wraps the vision payload in res.body.result
      const visionPayload = res.body.result || res.body;
      // matches[] is the list of pose detections in camera frame
      const matches = visionPayload.matches || visionPayload.poses || visionPayload.all_poses || [];

      const frameId = res.body.frame_id || res.body.timestamp_ns || "";
      const signature = JSON.stringify([matches.length, frameId]);
      if (signature === lastVisionSignature) return;
      lastVisionSignature = signature;

      // New vision capture: if we don't already have a frozen camera matrix for the
      // current cycle, snapshot it now (robot is ~at the capture pose). Reuse the
      // frozen one so the cloud/poses stay put when the robot moves afterwards.
      if (!frozenCameraMatrix) snapshotCameraMatrix();
      const T_world_camera = vizCameraMatrix();

      // Build pose list: each match has pose_matrix (4×4 row-major, camera frame)
      const posesForDraw = matches.map((m) => ({
        matrix: m.pose_matrix || m.initial_pose_matrix || null,
      })).filter((p) => p.matrix);

      const objectId = matches[0]?.object_id || visionPayload.object_id || null;
      drawDetectedPoses(sceneGroups.detectedPoses, posesForDraw, T_world_camera, ctx.assetId, objectId);

      // Point cloud: prefer the full scene PLY from debug_paths (dense, colored).
      // Fall back to sparse safety_pcd neighbor points if PLY not available.
      const primaryMatch = matches[0] || {};
      const vis3d = primaryMatch.visualization_3d || {};
      const dbg = primaryMatch.debug_paths || {};
      const plyPath = String(
        vis3d.scene_point_cloud_ply_path || dbg.scene_point_cloud_ply_path || ""
      ).trim();

      let drawnPly = false;
      if (plyPath) {
        drawnPly = await loadAndDrawScenePly(sceneGroups.pointCloud, plyPath, T_world_camera);
      }
      if (!drawnPly) {
        // Fallback: sparse safety_pcd points (camera frame)
        const pcdPoints = [];
        for (const m of matches) {
          const pcd = m.safety_pcd || {};
          pcdPoints.push(...(pcd.target_points_camera_m || []));
          pcdPoints.push(...(pcd.neighbor_points_camera_m || []));
        }
        if (pcdPoints.length) drawPointCloud(sceneGroups.pointCloud, pcdPoints, T_world_camera);
      }
      applyDisplayFlags(sceneGroups, displayControls, vp);
    } catch (err) {
      console.warn("vision result sync failed", err);
    } finally {
      visionInFlight = false;
    }
  }

  function cameraWorldMatrix() {
    // Returns a THREE.Matrix4 representing the *live* T_world_camera (tracks robot).
    if (sceneGroups.cameraFrame) {
      sceneGroups.cameraFrame.updateWorldMatrix(true, false);
      return sceneGroups.cameraFrame.matrixWorld.clone();
    }
    return new THREE.Matrix4();
  }

  function snapshotCameraMatrix() {
    // Capture the current camera world matrix and freeze it for vision rendering.
    frozenCameraMatrix = cameraWorldMatrix();
    return frozenCameraMatrix;
  }

  function vizCameraMatrix() {
    // T_world_camera to use when placing detected poses / point clouds: the frozen
    // capture-time matrix if we have one, else the live camera frame.
    return frozenCameraMatrix ? frozenCameraMatrix.clone() : cameraWorldMatrix();
  }

  async function runEvaluation() {
    if (evalInFlight) return;
    evalInFlight = true;
    try {
      const res = await api(`/processes/${encodeURIComponent(ctx.assetId)}/robot-digital-twin/evaluate${robotVizObjectQuery(ctx)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ distance: true, collision: true }),
      }, { silent: true });
      if (!res.ok || !res.body) {
        info.textContent = `Robot engine evaluation failed.`;
        return;
      }
      drawTrajectory(sceneGroups.trajectory, res.body.motion);
      const evalResult = res.body.evaluation || {};
      const collision = evalResult.collision;
      const distances = evalResult.distance_results || [];
      const errors = evalResult.errors || [];
      const debug = await refreshCollisionDebug();
      const debugObjects = debug?.objects || [];
      const unavailableCount = debugObjects.filter((object) => object.backend === "exact_unavailable").length;
      const collidingIds = new Set((debug?.collision?.colliding_pairs || []).flatMap((pair) => Array.isArray(pair) ? pair.map(String) : []));
      const collidingSources = debugObjects
        .filter((object) => collidingIds.has(String(object.object_id)))
        .map((object) => `${object.object_id} <- ${shortAssetPath(object.asset_path) || "primitive"}`);
      const minDistance = distances
        .map((d) => Number(d.distance))
        .filter((d) => Number.isFinite(d))
        .sort((a, b) => a - b)[0];
      info.textContent = [
        `asset: ${ctx.assetId}`,
        `scene ok: ${evalResult.ok ? "yes" : "no"}`,
        `collision: ${collision?.collision ? "yes" : "no"}`,
        `pairs: ${(collision?.colliding_pairs || []).length}`,
        `collision objects: ${debugObjects.length}`,
        `exact mesh unavailable: ${unavailableCount}`,
        collidingSources.length ? `pair sources:\n${collidingSources.slice(0, 4).join("\n")}` : "pair sources: none",
        `min distance: ${Number.isFinite(minDistance) ? minDistance.toFixed(4) + " m" : "n/a"}`,
        `motion: ${res.body.motion?.success ? "planned" : (res.body.motion?.rejection_reason || "not planned")}`,
        errors.length ? `errors: ${errors.map((e) => e.code).join(", ")}` : "errors: none",
      ].join("\n");
    } catch (err) {
      info.textContent = `Robot engine evaluation failed: ${err.message || err}`;
    } finally {
      evalInFlight = false;
    }
  }

  async function refreshCollisionDebug() {
    try {
      const res = await api(`/processes/${encodeURIComponent(ctx.assetId)}/robot-digital-twin/collision-debug${robotVizObjectQuery(ctx)}`, {}, { silent: true });
      if (!res.ok || !res.body?.debug) return null;
      return res.body.debug;
    } catch (err) {
      console.warn("collision debug refresh failed", err);
      return null;
    }
  }

  async function refreshPlanningTimeline() {
    if (timelineInFlight) return;
    const runId = await resolveActiveRunId();
    if (!runId) {
      if (planningEl) planningEl.textContent = "No active/recent run for this task.";
      drawJointPlot(plotCanvas, []);
      runStatus = { stage: "—", detail: "no active run", sinceMs: 0, snapAt: Date.now(), runId: "" };
      renderRunStatus();
      return;
    }
    timelineInFlight = true;
    try {
      const res = await api(`/runs/${encodeURIComponent(runId)}/timeline?limit=240`, {}, { silent: true });
      const events = Array.isArray(res.body?.events) ? res.body.events : [];
      const lastEvent = events.length ? events[events.length - 1] : null;
      // Always refresh the Run Status strip (cheap), even when nothing else changed.
      runStatus = { ...deriveRunStage(events, runId), snapAt: Date.now() };
      renderRunStatus();
      const signature = `${runId}:${events.length}:${lastEvent?.timestamp_ns || ""}`;
      if (signature === lastTimelineSignature) return;
      lastTimelineSignature = signature;
      const plan = latestPlannerEvent(events);
      const moveEvents = events.filter((event) => event.event === "DUMMY_ROBOT_API_REQUEST" && event.api === "robot.movej" && Array.isArray(event.q));
      const moveResponses = events.filter((event) => event.event === "DUMMY_ROBOT_API_RESPONSE" && event.api === "robot.movej");
      const sceneEval = [...events].reverse().find((event) => event.event === "DUMMY_SCENE_EVALUATE_RESPONSE");
      const qWaypoints = Array.isArray(plan?.q_waypoints) && plan.q_waypoints.length
        ? plan.q_waypoints
        : moveEvents.map((event) => event.q);
      drawJointPlot(plotCanvas, qWaypoints);
      await drawJointTcpPath(sceneGroups.plannedPath, qWaypoints, activeScene, plan?.planner || "timeline");

      // Bin picking cycle path: collect all movej waypoints in order across the full cycle.
      const cycleWaypoints = extractCycleWaypoints(events);
      await drawCyclePath(sceneGroups.cyclePath, cycleWaypoints, activeScene);

      // Detect latest PICK_PLACE_MATCH cycle — force vision re-sync when cycle advances.
      const latestPickMatch = [...events].reverse().find((e) => e.event === "PICK_PLACE_MATCH" && typeof e.cycle === "number");
      const latestCycle = latestPickMatch?.cycle ?? -1;
      if (latestCycle !== lastVisionCycle) {
        lastVisionCycle = latestCycle;
        lastVisionSignature = ""; // invalidate so syncVisionResult redraws
        // New capture cycle: freeze the camera matrix now (robot is at the capture
        // pose). All draws for this cycle reuse it, so the cloud/poses don't drift.
        snapshotCameraMatrix();
        // If the PICK_PLACE_MATCH event carries inline match data, draw it immediately.
        if (latestPickMatch?.match) {
          const matchObj = latestPickMatch.match;
          const T_world_camera = vizCameraMatrix();
          const posesForDraw = [];
          if (matchObj.pose_matrix) posesForDraw.push({ matrix: matchObj.pose_matrix });
          if (matchObj.initial_pose_matrix) posesForDraw.push({ matrix: matchObj.initial_pose_matrix });
          if (posesForDraw.length) {
            drawDetectedPoses(sceneGroups.detectedPoses, posesForDraw, T_world_camera, ctx.assetId, matchObj.object_id || null);
          }
          // Prefer full scene PLY over sparse safety_pcd
          const vis3d = matchObj.visualization_3d || {};
          const dbg = matchObj.debug_paths || {};
          const plyPath = String(
            vis3d.scene_point_cloud_ply_path || dbg.scene_point_cloud_ply_path || ""
          ).trim();
          let drawnPly = false;
          if (plyPath) {
            drawnPly = await loadAndDrawScenePly(sceneGroups.pointCloud, plyPath, T_world_camera);
          }
          if (!drawnPly) {
            const pcd = matchObj.safety_pcd || {};
            const pcdPoints = [
              ...(pcd.target_points_camera_m || []),
              ...(pcd.neighbor_points_camera_m || []),
            ];
            if (pcdPoints.length) drawPointCloud(sceneGroups.pointCloud, pcdPoints, T_world_camera);
          }
          applyDisplayFlags(sceneGroups, displayControls, vp);
        }
      }

      if (planningEl) {
        const details = plan?.details || {};
        planningEl.textContent = [
          `run: ${runId}`,
          `planner: ${plan?.planner || "n/a"}`,
          `planned: ${plan?.ok ? "yes" : "no"}`,
          `waypoints: ${qWaypoints.length}`,
          `reason: ${details.reason || plan?.reason || "n/a"}`,
          `robot move req/resp: ${moveEvents.length}/${moveResponses.length}`,
          `collision: ${sceneEval?.collision?.collision ? "yes" : "no"}`,
          `pairs: ${(sceneEval?.collision?.colliding_pairs || []).length}`,
          `cycle waypoints: ${cycleWaypoints.length}`,
          `pick cycle: ${latestCycle >= 0 ? latestCycle : "n/a"}`,
        ].join("\n");
      }
    } catch (err) {
      if (planningEl) planningEl.textContent = `planning timeline failed: ${err.message || err}`;
    } finally {
      timelineInFlight = false;
    }
  }

  async function resolveActiveRunId() {
    const freshCtx = context();
    if (freshCtx.runId) return freshCtx.runId;
    if (!freshCtx.taskId) return "";
    const res = await api(`/tasks/${encodeURIComponent(freshCtx.taskId)}/runs`, {}, { silent: true });
    const runs = Array.isArray(res.body?.runs) ? res.body.runs : [];
    const active = runs.find((run) => ["running", "created", "paused"].includes(String(run.state || "")));
    return String((active || runs[0] || {}).run_id || "");
  }

  function latestPlannerEvent(events) {
    return [...events].reverse().find((event) => (
      event.event === "DUMMY_PLANNER_RESPONSE" &&
      (Array.isArray(event.q_waypoints) || event.planner || event.reason || event.details)
    )) || null;
  }

  // Map bin-picking timeline events to a coarse "what is it doing right now" stage.
  // Returns {stage, detail, sinceMs} from the most recent meaningful event.
  function deriveRunStage(events, runId) {
    if (!Array.isArray(events) || !events.length) {
      return { stage: "—", detail: "no events yet", sinceMs: 0, runId };
    }
    const tNs = (e) => Number(e?.timestamp_ns || 0);
    const last = events[events.length - 1];
    // Walk backwards to the first event that defines the current stage.
    const STAGE_BY_EVENT = {
      RUN_CREATED:                    ["queued", "run created"],
      RUN_START:                      ["starting", "run started"],
      BIN_PICKING_PREWARM:            ["prewarming vision", "loading model"],
      PICK_PLACE_CAPTURE_MOVE:        ["moving to capture pose", ""],
      PICK_PLACE_MOVE_TO_POSE_BEGIN:  ["moving robot", ""],
      PICK_PLACE_MOVE_TO_POSE_COMPLETE: ["at capture pose", ""],
      PICK_PLACE_CAPTURE_REACHED:     ["at capture pose", "ready to look"],
      PICK_PLACE_INTERMEDIATE_MOVE:   ["moving intermediate", ""],
      PICK_PLACE_VISION_START:        ["pose estimation", "running vision"],
      VISION_SESSION_START:           ["pose estimation", "running vision"],
      PICK_PLACE_LOOKING:             ["pose estimation", "running vision"],
      PICK_PLACE_MATCH:               ["grasp planning", "object found, selecting grasp"],
      VISION_MATCH:                   ["grasp planning", "object found, selecting grasp"],
      DEBUG_PICK_POSES:               ["grasp planning", "scoring grasp candidates"],
      PICK_PLACE_GRIPPER_PREGRASP:    ["opening gripper", "pre-grasp"],
      PICK_PLACE_APPROACH:            ["planning / IK", "solving approach"],
      PICK_PLACE_MOVE_BEGIN:          ["robot moving", ""],
      PICK_PLACE_MOVE_COMPLETE:       ["robot move done", ""],
      PICK_PLACE_GRASP:               ["grasping", "closing gripper"],
      PICK_PLACE_GRIPPER_ON:          ["grasping", "vacuum/jaw on"],
      PICK_PLACE_RETREAT:             ["retreating", ""],
      PICK_DONE:                      ["pick done", "object grasped"],
      DEBUG_PLACE_PLAN_RESOLVED:      ["planning place", ""],
      DEBUG_PLACE_POSES:              ["planning place", ""],
      DEBUG_PLACE_POSITIONS:          ["planning place", ""],
      PICK_PLACE_PLACE_APPROACH:      ["moving to place", ""],
      PICK_PLACE_PLACE_MOVE:          ["placing", ""],
      PICK_PLACE_GRIPPER_OFF:         ["releasing", ""],
      PICK_PLACE_PLACE_RETREAT:       ["retreating from place", ""],
      PLACE_DONE:                     ["place done", ""],
      PICK_PLACE_DONE:                ["cycle done", "preparing next"],
      PICK_PLACE_VISION_REJECTED:     ["retrying", "no grasp candidate — re-capturing"],
      PICK_PLACE_IK_FAILED:           ["retrying", "IK failed — re-capturing"],
      PICK_PLACE_ATTEMPT_FAILED:      ["retrying", "attempt failed"],
      PICK_PLACE_RETRY:               ["retrying", "re-capturing"],
      RUN_STOPPING:                   ["stopping", "stop requested"],
      RUN_ABORTED:                    ["aborted", ""],
      RUN_FAILED:                     ["failed", ""],
      RUN_DONE:                       ["done", ""],
      RUN_COMPLETED:                  ["done", ""],
    };
    // Prefer the most recent event that has a mapping.
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      const m = STAGE_BY_EVENT[ev?.event];
      if (!m) continue;
      let [stage, detail] = m;
      // Enrich a couple of stages with extra context if present on the event.
      const cyc = (typeof ev.cycle === "number") ? ` · cycle ${ev.cycle}` : "";
      const att = (typeof ev.attempt === "number") ? ` att ${ev.attempt}` : "";
      if (ev.event === "PICK_PLACE_IK_FAILED" || ev.event === "PICK_PLACE_ATTEMPT_FAILED") {
        const err = String(ev.error || "").split(":")[0];
        if (err) detail = `${err}${cyc}${att}`;
      } else if (detail || cyc || att) {
        detail = `${detail || ""}${detail ? "" : ""}${cyc}${att}`.trim();
      }
      // sinceMs from this event's timestamp (ns) to now.
      const nowNs = Date.now() * 1e6;
      const sinceMs = tNs(ev) > 0 ? Math.max(0, (nowNs - tNs(ev)) / 1e6) : 0;
      return { stage, detail, sinceMs, runId };
    }
    // Fallback: use the very last event's name.
    const nowNs = Date.now() * 1e6;
    return {
      stage: String(last.event || "?").toLowerCase().replace(/_/g, " "),
      detail: "",
      sinceMs: tNs(last) > 0 ? Math.max(0, (nowNs - tNs(last)) / 1e6) : 0,
      runId,
    };
  }

  function renderRunStatus() {
    if (!runStageEl) return;
    const s = runStatus || {};
    const secs = (s.sinceMs || 0) / 1000.0;
    // sinceMs in runStatus is the snapshot from the last timeline poll; add the
    // wall-clock drift since then so the seconds tick smoothly between polls.
    const drift = s.snapAt ? (Date.now() - s.snapAt) / 1000.0 : 0;
    const elapsed = secs + drift;
    runStageEl.textContent = `${s.stage || "—"}${elapsed > 0 ? `  ·  ${elapsed.toFixed(1)}s` : ""}`;
    if (runDetailEl) {
      const parts = [];
      if (s.detail) parts.push(s.detail);
      if (s.runId) parts.push(s.runId);
      runDetailEl.textContent = parts.join("  ·  ") || (s.runId ? s.runId : "no active run");
    }
  }

  async function uploadObstacleCad(event) {
    const file = event?.target?.files?.[0];
    if (!file) return;
    const res = await api(`/processes/${encodeURIComponent(ctx.assetId)}/dummy-testing/obstacles/upload?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    }, { silent: true });
    event.target.value = "";
    if (!res.ok) {
      info.textContent = `Obstacle upload failed.`;
      return;
    }
    await loadDummyObstacles(res.body?.scene?.obstacles || []);
    await runEvaluation();
  }

  async function loadDummyObstacles(obstacles) {
    sceneGroups.obstacles.clear();
    obstacleTransform.detach();
    currentDummyObstacles = (obstacles || []).map((obstacle) => ({
      ...obstacle,
      pose: { ...(obstacle.pose || {}) },
    }));
    console.log(`[robot-viz] loadDummyObstacles: ${(obstacles || []).length} obstacle(s)`);
    for (const obstacle of obstacles || []) {
      const obstacleUrl = obstacle?.url || (obstacle?.mesh ? `/processes/${encodeURIComponent(ctx.assetId)}/dummy-testing/files/${obstacle.mesh}` : "");
      if (!obstacleUrl) {
        console.warn(`[robot-viz]   skip obstacle '${obstacle?.id || obstacle?.name}': no URL/mesh`);
        continue;
      }
      const object = await loadCadUrl(obstacleUrl, 0xff6b6b, 0.45);
      if (!object) {
        console.warn(`[robot-viz]   FAILED to load obstacle '${obstacle?.id || obstacle?.name}' from ${obstacleUrl}`);
        continue;
      }
      console.log(`[robot-viz]   loaded obstacle '${obstacle?.id || obstacle?.name}' from ${obstacleUrl}`);
      const obstacleId = String(obstacle.id || obstacle.name || "obstacle");
      object.name = `dummy:${obstacleId}`;
      object.userData.obstacleId = obstacleId;
      object.userData.importedObstacle = true;
      const pose = obstacle.pose || {};
      const pos = Array.isArray(pose.position_m) ? pose.position_m : [0, 0, 0];
      const rot = Array.isArray(pose.rotation_rpy_deg) ? pose.rotation_rpy_deg : [0, 0, 0];
      object.position.set(Number(pos[0] || 0), Number(pos[1] || 0), Number(pos[2] || 0));
      object.rotation.set(
        Number(rot[0] || 0) * Math.PI / 180,
        Number(rot[1] || 0) * Math.PI / 180,
        Number(rot[2] || 0) * Math.PI / 180,
        "XYZ"
      );
      sceneGroups.obstacles.add(object);
    }
    populateObstacleSelect();
    if (obsEditCheck) obsEditCheck.checked = false;
    setObstacleEditState(false);
    refreshPanelFieldsFromObject();
    applyDisplayFlags(sceneGroups, displayControls, vp);
  }

  function populateObstacleSelect() {
    const selects = [
      shell.querySelector("#robotVizObstacleSelect"),
      shell.querySelector("#robotVizObstacleSelectPanel"),
    ].filter(Boolean);
    if (!selects.length) return;
    const previous = selects[0].value || (selects[1] && selects[1].value) || "";
    selects.forEach((select) => {
      select.innerHTML = "";
      currentDummyObstacles.forEach((obstacle) => {
        const id = String(obstacle.id || obstacle.name || "");
        if (!id) return;
        select.appendChild(el("option", { value: id, text: obstacle.name || id }));
      });
      if (previous && Array.from(select.options).some((opt) => opt.value === previous)) {
        select.value = previous;
      } else if (select.options.length) {
        select.selectedIndex = 0;
      }
    });
  }

  function obstacleSelectValue() {
    const panel = shell.querySelector("#robotVizObstacleSelectPanel");
    if (panel && panel.value) return String(panel.value);
    const toolbar = shell.querySelector("#robotVizObstacleSelect");
    return String(toolbar?.value || "");
  }

  function selectedObstacleObject() {
    const id = obstacleSelectValue();
    if (!id) return null;
    return sceneGroups.obstacles.children.find((object) => object.userData?.obstacleId === id) || null;
  }

  function attachSelectedObstacle() {
    const object = selectedObstacleObject();
    if (!object || !obstacleEditEnabled()) {
      obstacleTransform.detach();
      return;
    }
    obstacleTransform.attach(object);
  }

  function syncSelectedObstacleFromObject() {
    const object = selectedObstacleObject();
    if (!object) return;
    const id = object.userData.obstacleId;
    const obstacle = currentDummyObstacles.find((item) => String(item.id || item.name || "") === id);
    if (!obstacle) return;
    obstacle.pose = {
      ...(obstacle.pose || {}),
      position_m: [
        roundPoseNumber(object.position.x),
        roundPoseNumber(object.position.y),
        roundPoseNumber(object.position.z),
      ],
      rotation_rpy_deg: [
        roundPoseNumber(THREE.MathUtils.radToDeg(object.rotation.x)),
        roundPoseNumber(THREE.MathUtils.radToDeg(object.rotation.y)),
        roundPoseNumber(THREE.MathUtils.radToDeg(object.rotation.z)),
      ],
    };
  }

  function roundPoseNumber(value) {
    return Number((Number(value) || 0).toFixed(6));
  }

  function obstacleForSave(obstacle) {
    const out = { ...obstacle, pose: { ...(obstacle.pose || {}) } };
    delete out.url;
    return out;
  }

  async function saveObstacleScene() {
    syncSelectedObstacleFromObject();
    const payload = { obstacles: currentDummyObstacles.map(obstacleForSave) };
    // When a task is selected, save into that task's scene.yaml; otherwise fall
    // back to the legacy dummy_testing/scene.yaml.
    const taskId = String(ctx.taskId || "").trim();
    const savePath = taskId
      ? `/processes/${encodeURIComponent(ctx.assetId)}/tasks/${encodeURIComponent(taskId)}/scene`
      : `/processes/${encodeURIComponent(ctx.assetId)}/dummy-testing/scene`;
    console.log(`[robot-viz] saving obstacle scene -> ${savePath} (${payload.obstacles.length} obstacle(s))`);
    const res = await api(savePath, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }, { silent: true });
    if (!res.ok) {
      info.textContent = "Obstacle pose save failed.";
      console.warn("[robot-viz] obstacle save failed", res?.body);
      return;
    }
    // Re-fetch the full digital-twin scene so obstacle URLs/poses come back
    // through the same (per-task) path that loaded them.
    const sceneRes = await api(`/processes/${encodeURIComponent(ctx.assetId)}/robot-digital-twin/scene${robotVizObjectQuery(ctx)}`, {}, { silent: true });
    const obstacles = sceneRes.ok ? (sceneRes.body?.meta?.dummy_testing?.obstacles || []) : (res.body?.scene?.obstacles || payload.obstacles);
    await loadDummyObstacles(obstacles);
    await runEvaluation();
  }
}

function checkbox(id, checked) {
  const input = document.createElement("input");
  input.type = "checkbox";
  input.id = id;
  input.checked = !!checked;
  return input;
}

function checkboxPersist(id, defaultChecked) {
  const stored = localStorage.getItem(`rviz_cb_${id}`);
  const checked = stored === null ? defaultChecked : stored === "1";
  return checkbox(id, checked);
}

function attachOrPlaceGripper(robotRoot, gripperRoot, robotManifest = {}, gripperFrames = {}) {
  const mount = robotManifest?.mount_link || robotManifest?.tcp_link || robotManifest?.planning_groups?.arm?.tip_link;
  const links = robotRoot?.userData?.urdf?.links;
  const mountGroup = mount && links?.get?.(mount);
  if (mountGroup) {
    mountGroup.add(gripperRoot);
  } else {
    gripperRoot.position.set(0.55, 0, 0.1);
    robotRoot?.parent?.add(gripperRoot);
  }
  const flange = gripperFrames?.flange;
  if (flange?.position || flange?.rotation) {
    const p = flange.position || [0, 0, 0];
    const r = flange.rotation || [0, 0, 0];
    gripperRoot.position.set(Number(p[0] || 0) / 1000, Number(p[1] || 0) / 1000, Number(p[2] || 0) / 1000);
    gripperRoot.rotation.set(Number(r[0] || 0) * Math.PI / 180, Number(r[1] || 0) * Math.PI / 180, Number(r[2] || 0) * Math.PI / 180, "XYZ");
  }
}

function drawFlangeFrame(parent, frameData, color, name) {
  const DEG = Math.PI / 180;
  const group = new THREE.Group();
  group.name = name;
  const pos = Array.isArray(frameData?.position) ? frameData.position : [0, 0, 0];
  const rot = Array.isArray(frameData?.rotation) ? frameData.rotation : [0, 0, 0];
  // position is in mm, rotation in degrees — same convention as gripper frames.json
  group.position.set(Number(pos[0] || 0) / 1000, Number(pos[1] || 0) / 1000, Number(pos[2] || 0) / 1000);
  group.rotation.set(Number(rot[0] || 0) * DEG, Number(rot[1] || 0) * DEG, Number(rot[2] || 0) * DEG, "XYZ");
  const size = 0.07;
  group.add(new THREE.AxesHelper(size));
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(size * 0.13, 12, 8),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5 })
  );
  sphere.userData.frameMarker = true;
  group.add(sphere);
  parent.add(group);
  return group;
}

function applyDisplayFlags(groups, controls, vp) {
  const showVisual = !!controls.visual?.checked;
  const showCollision = !!controls.collision?.checked;
  const showTf = !!controls.tf?.checked;
  if (vp?.grid) vp.grid.visible = !!controls.grid?.checked;
  [groups.robot, groups.gripper].forEach((root) => {
    root?.traverse?.((node) => {
      if (node.type === "AxesHelper") node.visible = showTf;
      if (node.isMesh) {
        const isCollision = !!node.userData?.collision || !!node.parent?.userData?.collision;
        // Don't hide frame marker spheres (they carry userData.frameMarker).
        if (node.userData?.frameMarker) return;
        node.visible = isCollision ? showCollision : showVisual;
      }
    });
  });
  if (groups.object) groups.object.visible = showVisual;
  if (groups.obstacles) groups.obstacles.visible = !!controls.obstacles?.checked;
  if (groups.tcpFrame) groups.tcpFrame.visible = !!controls.tcp?.checked;
  if (groups.cameraFrame) groups.cameraFrame.visible = !!controls.camera?.checked;
  if (groups.detectedPoses) groups.detectedPoses.visible = controls.detectedPoses ? !!controls.detectedPoses.checked : true;
  if (groups.pointCloud) groups.pointCloud.visible = controls.pointCloud ? !!controls.pointCloud.checked : true;
  if (groups.cyclePath) groups.cyclePath.visible = controls.cyclePath ? !!controls.cyclePath.checked : true;
}

function buildJointControls(root, parent) {
  if (!parent) return;
  const urdf = root?.userData?.urdf;
  parent.replaceChildren();
  if (urdf) urdf.liveControls = new Map();
  const actuators = urdf?.actuators || [];
  if (!actuators.length) {
    parent.textContent = "No actuated joints.";
    return;
  }
  actuators.forEach((joint) => {
    const row = el("label", { class: "joint-row" });
    const title = el("span", { class: "joint-header", text: joint.name });
    const slider = el("input", { type: "range" });
    const out = el("output", { text: "0.00" });
    const lower = Number.isFinite(joint.lower) && joint.lower !== joint.upper ? joint.lower : -Math.PI;
    const upper = Number.isFinite(joint.upper) && joint.lower !== joint.upper ? joint.upper : Math.PI;
    slider.min = String(lower);
    slider.max = String(upper);
    slider.step = joint.type === "prismatic" ? "0.001" : "0.01";
    slider.value = String(Number(joint.value || 0));
    slider.addEventListener("pointerdown", () => {
      slider.dataset.dragging = "1";
    });
    slider.addEventListener("pointerup", () => {
      slider.dataset.dragging = "";
    });
    slider.addEventListener("pointercancel", () => {
      slider.dataset.dragging = "";
    });
    slider.addEventListener("input", () => {
      const value = Number(slider.value) || 0;
      urdf.setJointValue(joint.name, value);
      out.textContent = joint.type === "prismatic" ? `${value.toFixed(3)} m` : `${value.toFixed(2)} rad`;
    });
    urdf.liveControls?.set?.(joint.name, { slider, out, type: joint.type });
    row.append(title, slider, out);
    parent.appendChild(row);
  });
}

function updateJointControls(root, joints) {
  const controls = root?.userData?.urdf?.liveControls;
  if (!controls || !joints) return;
  Object.entries(joints).forEach(([name, value]) => {
    const control = controls.get(name);
    if (!control) return;
    const numeric = Number(value) || 0;
    if (control.slider?.dataset?.dragging !== "1") {
      control.slider.value = String(numeric);
    }
    if (control.out) {
      control.out.textContent = control.type === "prismatic" ? `${numeric.toFixed(3)} m` : `${numeric.toFixed(3)} rad`;
    }
  });
}

async function loadObjectCad(assetId, objectId) {
  try {
    const meta = await api(`/processes/${encodeURIComponent(assetId)}/objects/${encodeURIComponent(objectId)}/cad`, {}, { silent: true });
    if (!meta.ok || !meta.body?.url) return null;
    const url = meta.body.url;
    const object = await loadCadUrl(url, 0x8fd3ff, 0.62);
    object.name = `object:${objectId}`;
    object.scale.setScalar(0.001);

    // The detected pose is at the bin_picking_frame (defined in metadata.json), not the
    // CAD origin. Shift the mesh by -bin_picking_frame so its bin_picking_frame lands at
    // the detected origin when we later apply T_world_obj. Wrap in a parent Group so the
    // shift composes cleanly with the 0.001 scale already on `object`.
    let bpPosM = [0, 0, 0];
    let bpRpyDeg = [0, 0, 0];
    try {
      const frameRes = await api(
        `/processes/${encodeURIComponent(assetId)}/objects/${encodeURIComponent(objectId)}/frame`,
        {},
        { silent: true },
      );
      if (frameRes.ok && frameRes.body?.frame) {
        const f = frameRes.body.frame;
        if (Array.isArray(f.position_m)) bpPosM = f.position_m.map((v) => Number(v) || 0);
        if (Array.isArray(f.rotation_rpy_deg)) bpRpyDeg = f.rotation_rpy_deg.map((v) => Number(v) || 0);
      }
    } catch {
      // Metadata may be missing — fall back to no shift.
    }

    const isShifted =
      Math.abs(bpPosM[0]) > 1e-9 || Math.abs(bpPosM[1]) > 1e-9 || Math.abs(bpPosM[2]) > 1e-9 ||
      Math.abs(bpRpyDeg[0]) > 1e-9 || Math.abs(bpRpyDeg[1]) > 1e-9 || Math.abs(bpRpyDeg[2]) > 1e-9;
    if (!isShifted) return object;

    // CAD vertex p_in_bp = R_bp^-1 · (p_in_cad − bp_pos). Equivalently: place the loaded
    // CAD-frame `object` under a parent transformed by (R_bp^-1, R_bp^-1·(−bp_pos)) — i.e.
    // the inverse of the bp→cad transform. Then applying T_world_obj to the parent puts
    // the bin_picking_frame at the detected pose, with the CAD origin at the right offset.
    const DEG = Math.PI / 180.0;
    const T_cad_bp = new THREE.Matrix4().compose(
      new THREE.Vector3(bpPosM[0], bpPosM[1], bpPosM[2]),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(bpRpyDeg[0] * DEG, bpRpyDeg[1] * DEG, bpRpyDeg[2] * DEG, "XYZ")),
      new THREE.Vector3(1, 1, 1),
    );
    const T_bp_cad = new THREE.Matrix4().copy(T_cad_bp).invert();
    const wrapper = new THREE.Group();
    wrapper.name = `object_bp:${objectId}`;
    wrapper.matrixAutoUpdate = false;
    wrapper.matrix.copy(T_bp_cad);
    wrapper.add(object);
    return wrapper;
  } catch (err) {
    console.warn("object CAD load failed", err);
    return null;
  }
}

async function loadCadUrl(url, color = 0x8fd3ff, opacity = 0.62) {
  const cacheKey = `${url}|${color}|${opacity}`;
  const cached = cadObjectCache.get(cacheKey);
  if (cached) return cached.clone(true);
  const ext = url.split("?")[0].split(".").pop().toLowerCase();
  let object = null;
  const material = () => new THREE.MeshStandardMaterial({ color, transparent: opacity < 1, opacity });
  if (ext === "stl") {
    const buf = await (await fetch(url, { cache: "no-store" })).arrayBuffer();
    const geo = new STLLoader().parse(buf);
    geo.computeVertexNormals();
    object = new THREE.Mesh(geo, material());
  } else if (ext === "obj") {
    const text = await (await fetch(url, { cache: "no-store" })).text();
    object = new OBJLoader().parse(text);
    object.traverse((c) => {
      if (c.isMesh) c.material = material();
    });
  } else if (ext === "dae") {
    object = (await new ColladaLoader().loadAsync(url)).scene;
  } else {
    const blob = await (await fetch(url, { cache: "no-store" })).blob();
    object = await loadCadLikeFile(new File([blob], `object.${ext}`, { type: blob.type }));
    object.traverse?.((c) => {
      if (c.isMesh) c.material = material();
    });
  }
  cadObjectCache.set(cacheKey, object);
  return object;
}

function shortAssetPath(path) {
  const text = String(path || "");
  const marker = "/data/stations/";
  const idx = text.indexOf(marker);
  if (idx >= 0) return text.slice(idx + 1);
  return text.split("/").slice(-4).join("/");
}

function drawGrasps(group, grasps) {
  group.clear();
  grasps.slice(0, 120).forEach((grasp, idx) => {
    const matrix = grasp?.tcp_in_object?.matrix;
    if (!matrix) return;
    const marker = makeFrameMarker(idx === 0 ? 0xffd166 : 0x66e3ff, idx === 0 ? 0.055 : 0.035);
    const m = new THREE.Matrix4().fromArray(matrix.flat());
    marker.applyMatrix4(m);
    group.add(marker);
  });
}

function makeFrameMarker(color, size) {
  const group = new THREE.Group();
  group.add(new THREE.AxesHelper(size));
  group.add(new THREE.Mesh(
    new THREE.SphereGeometry(size * 0.12, 12, 8),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35 })
  ));
  return group;
}

function drawTrajectory(group, motion) {
  group.clear();
  const points = [];
  const frames = motion?.cartesian_waypoints || motion?.generated_frames || [];
  frames.forEach((frame) => {
    const matrix = frame?.matrix;
    if (!matrix) return;
    points.push(new THREE.Vector3(Number(matrix[0][3] || 0), Number(matrix[1][3] || 0), Number(matrix[2][3] || 0)));
  });
  if (points.length < 2) return;
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group.add(new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color: 0x24d17e, linewidth: 2 })
  ));
}

function drawJointPlot(canvas, qWaypoints) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (height * i) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  const rows = (qWaypoints || []).filter((q) => Array.isArray(q) && q.length);
  if (rows.length < 2) {
    ctx.fillStyle = "#9aa8b5";
    ctx.font = "12px IBM Plex Mono, monospace";
    ctx.fillText("No path yet", 12, 24);
    return;
  }
  const dof = Math.min(7, Math.max(...rows.map((q) => q.length)));
  const colors = ["#24d17e", "#44ddff", "#ffaa00", "#ff6b6b", "#b58cff", "#f5e663", "#d4dde6"];
  for (let j = 0; j < dof; j += 1) {
    const values = rows.map((q) => Number(q[j] || 0));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, 0.001);
    ctx.strokeStyle = colors[j % colors.length];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    values.forEach((value, idx) => {
      const x = rows.length === 1 ? 0 : (idx / (rows.length - 1)) * width;
      const y = height - ((value - min) / span) * (height - 18) - 9;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#d4dde6";
  ctx.font = "11px IBM Plex Mono, monospace";
  ctx.fillText(`${rows.length} waypoints`, 10, 18);
}

async function drawJointTcpPath(group, qWaypoints, scene, plannerName) {
  const rows = (qWaypoints || []).filter((q) => Array.isArray(q) && q.length);
  const lastRow = rows.length ? rows[rows.length - 1] : null;
  const signature = `${plannerName}:${rows.length}:${rows[0]?.join(",") || ""}:${lastRow?.join(",") || ""}`;
  if (group.userData.lastSignature === signature) return;
  group.userData.lastSignature = signature;
  group.clear();
  if (!scene?.chain || rows.length < 2) return;
  const tcpFrame = scene.chain.tcp?.transform?.child_frame || scene.chain.tip_frame;
  const jointNames = (scene.chain.joints || [])
    .filter((joint) => joint.joint_type !== "fixed")
    .map((joint) => joint.name);
  if (!jointNames.length) return;
  const stride = Math.max(1, Math.ceil(rows.length / 80));
  const sampled = rows.filter((_, idx) => idx % stride === 0 || idx === rows.length - 1);
  const points = [];
  for (const q of sampled) {
    const jointPositions = {};
    jointNames.forEach((name, idx) => {
      if (idx < q.length) jointPositions[name] = Number(q[idx]) || 0;
    });
    try {
      const res = await api("/robot-engine/fk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chain: scene.chain,
          joint_positions: jointPositions,
          target_frame: tcpFrame,
        }),
      }, { silent: true });
      const tf = res.body?.result?.transforms?.[tcpFrame];
      const matrix = tf?.matrix;
      if (matrix) {
        points.push(new THREE.Vector3(
          Number(matrix[0][3] || 0),
          Number(matrix[1][3] || 0),
          Number(matrix[2][3] || 0)
        ));
      }
    } catch (err) {
      console.warn("FK path point failed", err);
    }
  }
  if (points.length < 2) return;
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  group.add(new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color: 0xffd166, linewidth: 3 })
  ));
  points.forEach((point, idx) => {
    if (idx !== 0 && idx !== points.length - 1 && idx % Math.max(1, Math.floor(points.length / 12)) !== 0) return;
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(idx === 0 || idx === points.length - 1 ? 0.012 : 0.007, 10, 8),
      new THREE.MeshStandardMaterial({ color: idx === 0 ? 0x44ddff : idx === points.length - 1 ? 0x24d17e : 0xffd166 })
    );
    marker.position.copy(point);
    group.add(marker);
  });
}

// ---------------------------------------------------------------------------
// Detected poses (vision result → robot frame)
// ---------------------------------------------------------------------------

async function drawDetectedPoses(group, poses, T_world_camera, assetId, objectId) {
  group.clear();
  if (!poses || !poses.length) return;

  // Load the object CAD once for cloning.
  let cadTemplate = null;
  if (assetId && objectId) {
    cadTemplate = await loadObjectCad(assetId, objectId).catch(() => null);
  }

  const T_wc = T_world_camera; // THREE.Matrix4 world←camera

  poses.slice(0, 20).forEach((pose, idx) => {
    // pose.matrix is a 4×4 row-major list-of-lists in camera frame.
    const mat = pose?.matrix || pose?.pose_matrix || pose?.T_camera_object;
    if (!mat) return;

    // THREE.Matrix4.fromArray() reads column-major, but pose_matrix is row-major.
    // fromArray(flat) would silently transpose — build from rows explicitly instead.
    const r = mat;
    const T_cam_obj = new THREE.Matrix4().set(
      r[0][0], r[0][1], r[0][2], r[0][3],
      r[1][0], r[1][1], r[1][2], r[1][3],
      r[2][0], r[2][1], r[2][2], r[2][3],
      r[3][0], r[3][1], r[3][2], r[3][3],
    );

    // T_world_obj = T_world_camera @ T_camera_object
    const T_world_obj = T_wc.clone().multiply(T_cam_obj);

    if (cadTemplate) {
      const cad = cadTemplate.clone(true);
      cad.applyMatrix4(T_world_obj);
      // Tint first/best pose differently.
      cad.traverse((c) => {
        if (c.isMesh) {
          c.material = c.material.clone();
          c.material.color.setHex(idx === 0 ? 0xe040fb : 0x9c27b0);
          c.material.opacity = idx === 0 ? 0.75 : 0.45;
          c.material.transparent = true;
        }
      });
      group.add(cad);
    } else {
      // Fallback: axis marker at object pose.
      const marker = makeFrameMarker(idx === 0 ? 0xe040fb : 0x9c27b0, idx === 0 ? 0.06 : 0.04);
      marker.applyMatrix4(T_world_obj);
      group.add(marker);
    }
  });
}

// ---------------------------------------------------------------------------
// Point cloud (safety_pcd in camera frame → robot frame)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scene PLY loader — parses binary-little-endian PLY with float x,y,z and
// optional uchar r,g,b,a, downsamples to ≤MAX_PCD_POINTS, transforms to
// world frame, and builds colored THREE.Points.
// ---------------------------------------------------------------------------

const MAX_PCD_POINTS = 60000;

function parsePlyBinary(buffer) {
  const decoder = new TextDecoder("ascii");
  const bytes = new Uint8Array(buffer);

  // Parse header (ASCII lines up to "end_header\n")
  let headerEnd = 0;
  for (let i = 0; i < bytes.length - 10; i++) {
    if (bytes[i] === 0x65 && decoder.decode(bytes.slice(i, i + 11)) === "end_header\n") {
      headerEnd = i + 11;
      break;
    }
  }
  if (!headerEnd) return null;

  const header = decoder.decode(bytes.slice(0, headerEnd));
  const isBinaryLE = /format binary_little_endian/i.test(header);
  if (!isBinaryLE) return null; // ASCII PLY not handled

  // Parse vertex count and property list from header
  const vertexMatch = header.match(/element vertex (\d+)/);
  if (!vertexMatch) return null;
  const vertexCount = parseInt(vertexMatch[1], 10);

  // Extract ordered properties between "element vertex" and next "element" or end_header
  const vertexSection = header.slice(header.indexOf("element vertex"));
  const propLines = vertexSection.split("\n").slice(1).filter((l) => l.startsWith("property "));
  const nextElem = propLines.findIndex((l) => l.startsWith("element "));
  const ownProps = nextElem >= 0 ? propLines.slice(0, nextElem) : propLines;

  const propSizes = { float: 4, double: 8, int: 4, uint: 4, short: 2, ushort: 2, uchar: 1, char: 1, int32: 4, uint32: 4, int16: 2, uint16: 2, int8: 1, uint8: 1 };
  let stride = 0;
  const fields = [];
  for (const line of ownProps) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] !== "property" || parts.length < 3) continue;
    const type = parts[1].toLowerCase();
    const name = parts[2].toLowerCase();
    const size = propSizes[type] || 4;
    fields.push({ name, type, offset: stride, size });
    stride += size;
  }

  const xField = fields.find((f) => f.name === "x");
  const yField = fields.find((f) => f.name === "y");
  const zField = fields.find((f) => f.name === "z");
  const rField = fields.find((f) => f.name === "red" || f.name === "r");
  const gField = fields.find((f) => f.name === "green" || f.name === "g");
  const bField = fields.find((f) => f.name === "blue" || f.name === "b");
  if (!xField || !yField || !zField) return null;
  const hasColors = !!(rField && gField && bField);

  const view = new DataView(buffer, headerEnd);
  const total = Math.min(vertexCount, Math.floor((buffer.byteLength - headerEnd) / stride));

  // Stride-based downsampling
  const step = total > MAX_PCD_POINTS ? Math.ceil(total / MAX_PCD_POINTS) : 1;
  const kept = Math.ceil(total / step);
  const positions = new Float32Array(kept * 3);
  const colors = hasColors ? new Float32Array(kept * 3) : null;
  let wi = 0;
  for (let i = 0; i < total; i += step) {
    const base = i * stride;
    positions[wi * 3] = view.getFloat32(base + xField.offset, true);
    positions[wi * 3 + 1] = view.getFloat32(base + yField.offset, true);
    positions[wi * 3 + 2] = view.getFloat32(base + zField.offset, true);
    if (colors) {
      colors[wi * 3] = view.getUint8(base + rField.offset) / 255;
      colors[wi * 3 + 1] = view.getUint8(base + gField.offset) / 255;
      colors[wi * 3 + 2] = view.getUint8(base + bField.offset) / 255;
    }
    wi++;
  }
  return { positions, colors, count: wi };
}

async function loadAndDrawScenePly(group, plyPath, T_world_camera) {
  try {
    const url = `/debug/file?path=${encodeURIComponent(plyPath)}&t=${Date.now()}`;
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) return false;
    const parsed = parsePlyBinary(await resp.arrayBuffer());
    if (!parsed || !parsed.count) return false;

    const { positions, colors, count } = parsed;
    // Transform positions from camera frame to world frame in-place.
    const vec = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      vec.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      vec.applyMatrix4(T_world_camera);
      positions[i * 3] = vec.x;
      positions[i * 3 + 1] = vec.y;
      positions[i * 3 + 2] = vec.z;
    }

    group.clear();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    if (colors) geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.002,
      sizeAttenuation: true,
      vertexColors: !!colors,
      color: colors ? 0xffffff : 0x76ff03,
    });
    group.add(new THREE.Points(geo, mat));
    return true;
  } catch (err) {
    console.warn("scene PLY load failed", err);
    return false;
  }
}

function drawPointCloud(group, pointsCamera, T_world_camera) {
  group.clear();
  if (!pointsCamera || !pointsCamera.length) return;

  const flat = [];
  // pointsCamera is [[x,y,z], ...] in metres (safety_pcd fallback).
  for (const pt of pointsCamera) {
    if (!pt || pt.length < 3) continue;
    const v = new THREE.Vector3(Number(pt[0]), Number(pt[1]), Number(pt[2]));
    v.applyMatrix4(T_world_camera);
    flat.push(v.x, v.y, v.z);
  }
  if (!flat.length) return;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(flat, 3));
  const mat = new THREE.PointsMaterial({ color: 0x76ff03, size: 0.006, sizeAttenuation: true });
  group.add(new THREE.Points(geo, mat));
}

// ---------------------------------------------------------------------------
// Bin picking cycle path (all robot moves in timeline order)
// ---------------------------------------------------------------------------

function extractCycleWaypoints(events) {
  // Collect all movej joint configs from timeline, in event order.
  // Covers capture → approach → pick → retreat → place.
  return events
    .filter((e) => (
      (e.event === "DUMMY_ROBOT_API_REQUEST" || e.event === "BIN_PICKING_ROBOT_MOVE") &&
      Array.isArray(e.q) && e.q.length
    ) || (
      e.event === "PLANNER_WAYPOINTS" && Array.isArray(e.q_waypoints) && e.q_waypoints.length
    ))
    .flatMap((e) => e.q ? [e.q] : (e.q_waypoints || []));
}

async function drawCyclePath(group, qWaypoints, scene) {
  const rows = (qWaypoints || []).filter((q) => Array.isArray(q) && q.length);
  const lastRow = rows.length ? rows[rows.length - 1] : null;
  const sig = `cycle:${rows.length}:${rows[0]?.join(",") || ""}:${lastRow?.join(",") || ""}`;
  if (group.userData.lastCycleSig === sig) return;
  group.userData.lastCycleSig = sig;
  group.clear();
  if (!scene?.chain || rows.length < 2) return;

  const tcpFrame = scene.chain.tcp?.transform?.child_frame || scene.chain.tip_frame;
  const jointNames = (scene.chain.joints || [])
    .filter((j) => j.joint_type !== "fixed")
    .map((j) => j.name);
  if (!jointNames.length) return;

  const stride = Math.max(1, Math.ceil(rows.length / 120));
  const sampled = rows.filter((_, i) => i % stride === 0 || i === rows.length - 1);
  const points = [];

  for (const q of sampled) {
    const jointPositions = {};
    jointNames.forEach((name, idx) => { if (idx < q.length) jointPositions[name] = Number(q[idx]) || 0; });
    try {
      const res = await api("/robot-engine/fk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chain: scene.chain, joint_positions: jointPositions, target_frame: tcpFrame }),
      }, { silent: true });
      const tf = res.body?.result?.transforms?.[tcpFrame];
      const matrix = tf?.matrix;
      if (matrix) {
        points.push(new THREE.Vector3(Number(matrix[0][3] || 0), Number(matrix[1][3] || 0), Number(matrix[2][3] || 0)));
      }
    } catch (_) {}
  }
  if (points.length < 2) return;

  group.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: 0xff9800, linewidth: 3 })
  ));
  // Mark start (cyan) and end (green) of the cycle.
  [points[0], points[points.length - 1]].forEach((pt, idx) => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.013, 10, 8),
      new THREE.MeshStandardMaterial({ color: idx === 0 ? 0x44ddff : 0x24d17e })
    );
    m.position.copy(pt);
    group.add(m);
  });
}

function openToolPlaceholder(name) {
  const ctx = context();
  const message = ctx.assetId
    ? `${name} will use asset-scoped data under data/stations/.../assets/${ctx.assetId}.`
    : "Select an asset first.";
  emptyTool(name, message);
}

function bind() {
  const bindings = [
    ["openRobotVizBtn", openRobotVisualizer],
    ["openRobotSelectionBtn", () => openSelection("robot")],
    ["openGripperSelectionBtn", () => openSelection("gripper")],
    ["openObjectStudioBtn", () => openToolPlaceholder("Object Studio")],
    ["openBinPickingStudioBtn", () => openToolPlaceholder("Bin Picking Studio")],
  ];
  bindings.forEach(([id, fn]) => {
    const button = document.getElementById(id);
    if (button) button.addEventListener("click", fn);
  });

  // Auto-mount inline into monitor run tab if the mount point exists.
  const monitorMount = document.getElementById("monitorVizMount");
  if (monitorMount) {
    // Try immediately (shell may have already loaded from URL state), then retry after a delay.
    let mounted = false;
    let lastMountKey = "";
    function mountKey() {
      const ctx = context();
      return `${ctx.assetId || ""}|${ctx.taskId || ""}`;
    }
    async function tryMonitorMount(reason, force) {
      const ctx = context();
      if (!ctx.assetId) return;
      const key = mountKey();
      if (mounted && !force && key === lastMountKey) return;
      console.log(`[robot-viz] (re)mounting monitor viz — reason=${reason || "init"} key=${key}`);
      mounted = true;
      lastMountKey = key;
      await mountRobotVisualizer(monitorMount);
    }
    function remountIfChanged(reason) {
      // Only re-mount when asset/task actually changed — events fire often.
      if (mounted && mountKey() === lastMountKey) return;
      mounted = false;
      tryMonitorMount(reason);
    }
    tryMonitorMount("init");
    // Retry when asset selection changes.
    const assetSel = document.getElementById("assetSelect");
    if (assetSel) {
      assetSel.addEventListener("change", () => remountIfChanged("assetSelect-change"));
    }
    // Re-mount when the task selector (form tab) changes.
    const taskSel = document.getElementById("taskSelect");
    if (taskSel) {
      taskSel.addEventListener("change", () => remountIfChanged("taskSelect-change"));
    }
    // Re-mount when the Run Control task / Refresh fires (index.html dispatches this).
    window.addEventListener("imp:task-changed", (e) => {
      const reason = e?.detail?.reason || "?";
      // Explicit Refresh button: force a reload even if task id is unchanged
      // (scene.yaml on disk may have been edited).
      if (e?.detail?.force) {
        mounted = false;
        tryMonitorMount(`task-changed:${reason}`, true);
        return;
      }
      remountIfChanged(`task-changed:${reason}`);
    });
    // Retry when the run tab becomes visible (main UI dispatches this event).
    window.addEventListener("imp:run-tab-activated", () => remountIfChanged("run-tab-activated"));
    // Re-mount when the process/asset changes (more reliable than the assetSelect event).
    window.addEventListener("imp:process-changed", () => remountIfChanged("process-changed"));
    // Fallback: poll for asset/task changes for a while (covers async task-context load).
    const mountRetry = setInterval(async () => {
      remountIfChanged("poll");
    }, 800);
    // Stop polling after 30 s regardless.
    setTimeout(() => clearInterval(mountRetry), 30000);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bind);
} else {
  bind();
}

window.operatorBinPickingTools = {
  openRobotVisualizer,
  mountRobotVisualizer,
  openRobotSelection: () => openSelection("robot"),
  openGripperSelection: () => openSelection("gripper"),
};
