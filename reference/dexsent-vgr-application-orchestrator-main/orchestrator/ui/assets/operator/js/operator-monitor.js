(function () {
  const state = {
    tasks: [],
    currentTaskId: "",
    currentRunId: "",
    lastRunState: "idle",
    lastVisionRequestId: "",
    activeCameraId: "",
    runtimeCameraIds: [],
    latestVisionSummary: "",
    latestPickAnnotation: "",
    pollTimer: null,
    feedTimer: null,
    busy: false,
    feedBusy: false,
    liveSource: "camera",
    liveFrameId: "",
    liveObjectUrl: "",
    pendingLiveToken: "",
  };

  const liveBuffer = new Image();

  function byId(id) {
    return document.getElementById(id);
  }

  function getContext() {
    if (!window.operatorShell || typeof window.operatorShell.getContext !== "function") {
      return {
        stationId: "",
        assetId: "",
        stations: [],
        assets: [],
        cameraIds: [],
      };
    }
    return window.operatorShell.getContext();
  }

  function setInfo(text) {
    if (window.operatorShell && typeof window.operatorShell.setInfo === "function") {
      window.operatorShell.setInfo(text || "");
      return;
    }
    const host = byId("pageInfo");
    if (host) host.textContent = text || "";
  }

  function setText(id, text) {
    const el = byId(id);
    if (el) el.textContent = text;
  }

  function assetId(item) {
    if (!item || typeof item !== "object") return "";
    if (window.operatorShell && typeof window.operatorShell.assetId === "function") {
      return window.operatorShell.assetId(item);
    }
    return String(item.asset_id || item.process_id || "").trim();
  }

  function stationName(stationId) {
    const ctx = getContext();
    const station = (ctx.stations || []).find((item) => item.station_id === stationId);
    return (station && (station.name || station.station_id)) || stationId || "-";
  }

  function assetName(assetKey) {
    const ctx = getContext();
    const asset = (ctx.assets || []).find((item) => assetId(item) === assetKey);
    return (asset && (asset.name || assetId(asset))) || assetKey || "-";
  }

  function taskName(taskId) {
    const task = state.tasks.find((item) => item.task_id === taskId);
    return (task && (task.name || task.task_id)) || taskId || "-";
  }

  function fillSelect(id, items, getValue, getLabel, selectedValue) {
    const sel = byId(id);
    if (!sel) return;
    const current = selectedValue || sel.value || "";
    sel.innerHTML = "";
    (items || []).forEach((item) => {
      const opt = document.createElement("option");
      opt.value = getValue(item);
      opt.textContent = getLabel(item);
      sel.appendChild(opt);
    });
    if (current && Array.from(sel.options).some((opt) => opt.value === current)) {
      sel.value = current;
      return;
    }
    if (sel.options.length) {
      sel.selectedIndex = 0;
    }
  }

  function runLedMode(value) {
    const text = String(value || "").toLowerCase();
    if (text === "running" || text === "created" || text === "starting") {
      return "on";
    }
    if (text === "failed" || text === "aborted" || text === "error" || text === "stopped") {
      return "error";
    }
    return "off";
  }

  function setRunStateUi(value, phase) {
    const text = String(value || "idle").toLowerCase();
    const phaseText = String(phase || "").toLowerCase();
    const displayText = phaseText && text === "running" ? phaseText : text;
    state.lastRunState = text;
    const runLed = byId("runLed");
    setText("runStateText", displayText);
    setText("metaRunState", displayText);
    if (runLed) {
      runLed.classList.remove("on", "off", "error");
      runLed.classList.add(runLedMode(text));
    }
  }

  function setMeta() {
    const ctx = getContext();
    setText("metaStation", stationName(ctx.stationId));
    setText("metaAsset", assetName(ctx.assetId));
    setText("metaTask", taskName(state.currentTaskId));
    setText("metaRunId", state.currentRunId || "-");
  }

  function stationCameraIds() {
    const ctx = getContext();
    const station = (ctx.stations || []).find((item) => item.station_id === ctx.stationId);
    return station && Array.isArray(station.camera_ids) ? station.camera_ids : [];
  }

  function assetCameraIds() {
    const ctx = getContext();
    const asset = (ctx.assets || []).find((item) => assetId(item) === ctx.assetId);
    return asset && Array.isArray(asset.camera_ids) ? asset.camera_ids : [];
  }

  function refreshCameraSelector() {
    const ctx = getContext();
    const all = [
      ...stationCameraIds(),
      ...assetCameraIds(),
      ...(ctx.cameraIds || []),
      ...(state.runtimeCameraIds || []),
    ]
      .map((item) => String(item || "").trim())
      .filter(Boolean);

    const unique = [];
    all.forEach((item) => {
      if (!unique.includes(item)) unique.push(item);
    });

    fillSelect("cameraIdSelect", unique, (v) => v, (v) => v, state.activeCameraId);
    const sel = byId("cameraIdSelect");
    const next = (sel && sel.value) || "";
    if (next !== state.activeCameraId) {
      state.activeCameraId = next;
      state.liveFrameId = "";
      void refreshFeeds();
    }
  }

  async function refreshRuntimeCameras() {
    const res = await window.operatorApi("/camera/cameras", {}, { silent: true });
    if (!res.ok || !res.body) {
      state.runtimeCameraIds = [];
      return;
    }
    const cams = Array.isArray(res.body.cameras) ? res.body.cameras : [];
    state.runtimeCameraIds = cams.map((item) => String(item || "").trim()).filter(Boolean);
  }

  async function refreshTasks() {
    const ctx = getContext();
    if (!ctx.assetId) {
      state.tasks = [];
      state.currentTaskId = "";
      fillSelect("taskSelect", [], () => "", () => "", "");
      setMeta();
      return;
    }

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/tasks`,
      {},
      { silent: true }
    );
    if (!res.ok || !res.body) {
      state.tasks = [];
      state.currentTaskId = "";
      fillSelect("taskSelect", [], () => "", () => "", "");
      setInfo("Failed loading tasks.");
      setMeta();
      return;
    }

    state.tasks = Array.isArray(res.body.tasks) ? res.body.tasks : [];
    fillSelect(
      "taskSelect",
      state.tasks,
      (task) => task.task_id,
      (task) => task.name || task.task_id,
      state.currentTaskId
    );
    const sel = byId("taskSelect");
    state.currentTaskId = (sel && sel.value) || "";
    setMeta();
  }

  function pickLatestRun(runs) {
    if (!Array.isArray(runs) || !runs.length) return null;
    const sorted = runs.slice().sort((a, b) => {
      const ta = Date.parse(String(a.updated_at || a.created_at || a.started_at || a.ended_at || ""));
      const tb = Date.parse(String(b.updated_at || b.created_at || b.started_at || b.ended_at || ""));
      return (Number.isFinite(tb) ? tb : 0) - (Number.isFinite(ta) ? ta : 0);
    });
    return sorted[0];
  }

  async function refreshLatestRunForTask() {
    if (!state.currentTaskId) {
      state.currentRunId = "";
      state.lastVisionRequestId = "";
      state.latestPickAnnotation = "";
      setRunStateUi("idle");
      setLiveFeedAnnotation("");
      setMeta();
      return;
    }

    const res = await window.operatorApi(
      `/tasks/${encodeURIComponent(state.currentTaskId)}/runs`,
      {},
      { silent: true }
    );
    if (!res.ok || !res.body) return;

    const latest = pickLatestRun(res.body.runs || []);
    if (!latest || !latest.run_id) return;
    if (!state.currentRunId) {
      state.currentRunId = String(latest.run_id || "");
    }
    setMeta();
  }

  function shortEventLine(event) {
    const type = String(event.event || "event");
    const stage = String(event.stage || "");
    const message = String(event.message || event.detail || "");
    return [type, stage ? `(${stage})` : "", message].filter(Boolean).join(" ");
  }

  async function refreshRunState() {
    if (!state.currentRunId) {
      setRunStateUi("idle");
      setText("runEventLog", "No run events yet.");
      state.latestPickAnnotation = "";
      setLiveFeedAnnotation("");
      setMeta();
      return;
    }

    const runRes = await window.operatorApi(
      `/runs/${encodeURIComponent(state.currentRunId)}`,
      {},
      { silent: true }
    );
    if (!runRes.ok || !runRes.body) {
      setRunStateUi("idle");
      state.lastVisionRequestId = "";
      state.latestPickAnnotation = "";
      setLiveFeedAnnotation("");
      return;
    }

    setRunStateUi(String(runRes.body.state || "idle"), runRes.body.phase || "");
    state.lastVisionRequestId = String(
      runRes.body.vision_request_id || runRes.body.last_vision_request_id || ""
    );

    const timelineRes = await window.operatorApi(
      `/runs/${encodeURIComponent(state.currentRunId)}/timeline?limit=32`,
      {},
      { silent: true }
    );
    if (timelineRes.ok && timelineRes.body) {
      const events = Array.isArray(timelineRes.body.events) ? timelineRes.body.events : [];
      const recentEvents = events.slice(-8);
      setText("runEventLog", recentEvents.length ? recentEvents.map(shortEventLine).join("\n") : "No run events yet.");
      state.latestPickAnnotation = extractLatestPickAnnotation(events);
    } else {
      state.latestPickAnnotation = "";
    }
    if (state.liveSource === "vision") {
      setLiveFeedAnnotation(state.latestPickAnnotation);
    }
    setMeta();
  }

  function safeNumber(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function formatMeters(value) {
    return safeNumber(value, 0).toFixed(3);
  }

  function formatDeg(value) {
    return safeNumber(value, 0).toFixed(1);
  }

  function formatVec3(value) {
    if (!Array.isArray(value) || value.length < 3) return "[ -, -, - ]";
    return `[${formatMeters(value[0])}, ${formatMeters(value[1])}, ${formatMeters(value[2])}]`;
  }

  function setLiveFeedAnnotation(text) {
    setText("liveFeedAnnotation", text || "");
  }

  function buildPickAnnotation(match) {
    const pickContact = match && typeof match === "object" ? match.pick_contact : null;
    if (!pickContact || typeof pickContact !== "object") return "";
    const label = String(pickContact.label || pickContact.id || "contact").trim();
    return [
      `pick ${label}`,
      `point_b ${formatVec3(pickContact.point_base_m)} m`,
      `normal_b ${formatVec3(pickContact.normal_base)}`,
    ].join("\n");
  }

  function extractLatestPickAnnotation(events) {
    if (!Array.isArray(events) || !events.length) return "";
    for (let idx = events.length - 1; idx >= 0; idx -= 1) {
      const event = events[idx];
      if (!event || event.event !== "PICK_PLACE_MATCH") continue;
      const annotation = buildPickAnnotation(event.match);
      if (annotation) return annotation;
    }
    return "";
  }

  function radToDeg(value) {
    return value * (180 / Math.PI);
  }

  function quatToRpyDeg(quat) {
    if (!Array.isArray(quat) || quat.length < 4) return [0, 0, 0];
    const x = safeNumber(quat[0], 0);
    const y = safeNumber(quat[1], 0);
    const z = safeNumber(quat[2], 0);
    const w = safeNumber(quat[3], 1);

    const sinrCosp = 2 * (w * x + y * z);
    const cosrCosp = 1 - 2 * (x * x + y * y);
    const roll = Math.atan2(sinrCosp, cosrCosp);

    const sinp = 2 * (w * y - z * x);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * (Math.PI / 2) : Math.asin(sinp);

    const sinyCosp = 2 * (w * z + x * y);
    const cosyCosp = 1 - 2 * (y * y + z * z);
    const yaw = Math.atan2(sinyCosp, cosyCosp);

    return [radToDeg(roll), radToDeg(pitch), radToDeg(yaw)];
  }

  function readRpyFromPose(pose, fallback) {
    if (!pose) return fallback || [0, 0, 0];
    const rpyDeg = pose.rotation_rpy_deg || pose.rpy_deg || pose.rpy;
    if (Array.isArray(rpyDeg) && rpyDeg.length >= 3) {
      return [safeNumber(rpyDeg[0], 0), safeNumber(rpyDeg[1], 0), safeNumber(rpyDeg[2], 0)];
    }
    return quatToRpyDeg(pose.quat_xyzw || [0, 0, 0, 1]);
  }

  async function refreshRobotState() {
    const res = await window.operatorApi("/robot/state", {}, { silent: true });
    if (!res.ok || !res.body) {
      setText("robotMode", "offline");
      setText("robotError", "state unavailable");
      return;
    }

    const robotState = res.body;
    // Display custom_tcp_pose (fingertip wrt base) — falls back to raw EE if unavailable.
    const displayPose = robotState.custom_tcp_pose || robotState.tcp_pose || {};
    const position = Array.isArray(displayPose.position_m) ? displayPose.position_m : [0, 0, 0];
    const rpy = readRpyFromPose(displayPose);

    setText("robotX", formatMeters(position[0]));
    setText("robotY", formatMeters(position[1]));
    setText("robotZ", formatMeters(position[2]));
    setText("robotR", formatDeg(rpy[0]));
    setText("robotP", formatDeg(rpy[1]));
    setText("robotYaw", formatDeg(rpy[2]));
    setText("robotMode", String(robotState.mode || "idle"));
    setText("robotError", String(robotState.last_error || "").trim() || "none");
  }

  async function refreshVisionInfo() {
    state.latestVisionSummary = "";
    if (!state.lastVisionRequestId) {
      setLiveFeedAnnotation("");
      return;
    }

    const params = new URLSearchParams();
    params.set("request_id", state.lastVisionRequestId);
    params.set("include_image", "false");
    const res = await window.operatorApi(`/vision/latest?${params.toString()}`, {}, { silent: true });
    if (!res.ok || !res.body || String(res.body.status || "").toLowerCase() === "pending") return;

    const matches = Array.isArray(res.body.result && res.body.result.matches)
      ? res.body.result.matches.length
      : 0;
    const frameId = String(res.body.frame_id || "-");
    state.latestVisionSummary = `matches ${matches} | frame ${frameId}`;
    if (state.liveSource === "vision") {
      setLiveFeedAnnotation(state.latestPickAnnotation);
    }
  }

  function revokeUrl(url) {
    if (!url) return;
    try {
      URL.revokeObjectURL(url);
    } catch (_) {
      // Ignore URL revoke failures.
    }
  }

  async function fetchFrame(url) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) return null;
      const frameId = String(res.headers.get("X-Frame-Id") || "");
      const blob = await res.blob();
      if (!blob || blob.size === 0) return null;
      return { frameId, blob };
    } catch (_) {
      return null;
    }
  }

  function renderLiveFrame(source, frameId, blob, infoText) {
    const img = byId("liveFeed");
    if (!img) return;
    if (frameId && frameId === state.liveFrameId && source === state.liveSource) return;

    const nextUrl = URL.createObjectURL(blob);
    const token = `${source}-${frameId || Date.now()}`;
    state.pendingLiveToken = token;
    liveBuffer.onload = () => {
      if (state.pendingLiveToken !== token) {
        revokeUrl(nextUrl);
        return;
      }
      state.pendingLiveToken = "";
      revokeUrl(state.liveObjectUrl);
      state.liveObjectUrl = nextUrl;
      state.liveFrameId = frameId || token;
      state.liveSource = source;
      img.src = nextUrl;
      setText("liveFeedMode", source);
      setText("liveFeedInfo", infoText || `${source} live`);
      setLiveFeedAnnotation(source === "vision" ? state.latestPickAnnotation : "");
    };
    liveBuffer.onerror = () => {
      revokeUrl(nextUrl);
      if (state.pendingLiveToken === token) state.pendingLiveToken = "";
    };
    liveBuffer.src = nextUrl;
  }

  async function refreshUnifiedFeed() {
    if (state.lastVisionRequestId) {
      const visionParams = new URLSearchParams();
      visionParams.set("request_id", state.lastVisionRequestId);
      visionParams.set("t", String(Date.now()));
      const visionFrame = await fetchFrame(`/vision/frame?${visionParams.toString()}`);
      if (visionFrame) {
        const visionInfo = state.latestVisionSummary || "annotation live";
        renderLiveFrame("vision", visionFrame.frameId, visionFrame.blob, visionInfo);
        return;
      }
    }

    if (!state.activeCameraId) {
      setText("liveFeedMode", "camera");
      setText("liveFeedInfo", "camera id not selected");
      setLiveFeedAnnotation("");
      return;
    }

    const cameraParams = new URLSearchParams();
    cameraParams.set("camera_id", state.activeCameraId);
    cameraParams.set("fmt", "jpg");
    cameraParams.set("quality", "72");
    cameraParams.set("t", String(Date.now()));
    const cameraFrame = await fetchFrame(`/camera/frame?${cameraParams.toString()}`);
    if (!cameraFrame) {
      setText("liveFeedMode", "camera");
      setText("liveFeedInfo", `waiting for ${state.activeCameraId}`);
      setLiveFeedAnnotation("");
      return;
    }
    renderLiveFrame("camera", cameraFrame.frameId, cameraFrame.blob, `camera ${state.activeCameraId}`);
  }

  async function refreshFeeds() {
    if (state.feedBusy) return;
    state.feedBusy = true;
    try {
      await refreshUnifiedFeed();
    } finally {
      state.feedBusy = false;
    }
  }

  async function startRun() {
    if (!state.currentTaskId) {
      setInfo("Select a task first.");
      return;
    }
    const res = await window.operatorApi(
      `/tasks/${encodeURIComponent(state.currentTaskId)}/runs/start`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params: {} }),
      }
    );
    if (!res.ok || !res.body) {
      setInfo("Run start failed.");
      return;
    }
    state.currentRunId = String(res.body.run_id || "");
    state.lastVisionRequestId = "";
    state.latestPickAnnotation = "";
    state.liveFrameId = "";
    setLiveFeedAnnotation("");
    setRunStateUi(String(res.body.state || "created"));
    await refreshRunState();
    await refreshVisionInfo();
    await refreshFeeds();
    setInfo("Run started.");
  }

  async function pauseRun() {
    if (!state.currentRunId) {
      setInfo("No run selected.");
      return;
    }
    const res = await window.operatorApi(
      `/runs/${encodeURIComponent(state.currentRunId)}/pause`,
      { method: "POST" }
    );
    if (!res.ok) {
      setInfo("Run pause failed.");
      return;
    }
    await refreshRunState();
    setInfo("Run paused.");
  }

  async function stopRun() {
    if (!state.currentRunId) {
      setInfo("No run selected.");
      return;
    }
    const res = await window.operatorApi(
      `/runs/${encodeURIComponent(state.currentRunId)}/stop`,
      { method: "POST" }
    );
    if (!res.ok) {
      setInfo("Run stop failed.");
      return;
    }
    await refreshRunState();
    setInfo("Run stop requested.");
  }

  async function pollAll() {
    if (state.busy) return;
    state.busy = true;
    try {
      await refreshRuntimeCameras();
      refreshCameraSelector();
      await refreshRunState();
      await refreshRobotState();
      await refreshVisionInfo();
      setMeta();
    } finally {
      state.busy = false;
    }
  }

  async function refreshAll() {
    await refreshTasks();
    await refreshRuntimeCameras();
    refreshCameraSelector();
    await refreshLatestRunForTask();
    await pollAll();
    await refreshFeeds();
  }

  function bindEvents() {
    const taskSelect = byId("taskSelect");
    if (taskSelect) {
      taskSelect.addEventListener("change", async (event) => {
        state.currentTaskId = String(event.target.value || "");
        state.currentRunId = "";
        state.lastVisionRequestId = "";
        state.latestVisionSummary = "";
        state.latestPickAnnotation = "";
        state.liveFrameId = "";
        setLiveFeedAnnotation("");
        await refreshLatestRunForTask();
        await refreshRunState();
        await refreshVisionInfo();
        await refreshFeeds();
        setMeta();
      });
    }

    const cameraSelect = byId("cameraIdSelect");
    if (cameraSelect) {
      cameraSelect.addEventListener("change", async (event) => {
        state.activeCameraId = String(event.target.value || "");
        state.liveFrameId = "";
        await refreshFeeds();
      });
    }

    const startBtn = byId("startRunBtn");
    if (startBtn) startBtn.addEventListener("click", startRun);

    const pauseBtn = byId("pauseRunBtn");
    if (pauseBtn) pauseBtn.addEventListener("click", pauseRun);

    const stopBtn = byId("stopRunBtn");
    if (stopBtn) stopBtn.addEventListener("click", stopRun);

    const refreshBtn = byId("refreshMonitorBtn");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", async () => {
        if (window.operatorShell) {
          await window.operatorShell.refreshContext();
        }
        await refreshAll();
        setInfo("Monitor refreshed.");
      });
    }
  }

  async function init() {
    bindEvents();
    setLiveFeedAnnotation("");

    if (window.operatorShell && typeof window.operatorShell.init === "function") {
      await window.operatorShell.init({
        onContextChanged: async () => {
          await refreshTasks();
          await refreshRuntimeCameras();
          refreshCameraSelector();
          setMeta();
        },
      });
    }

    await refreshAll();
    setInfo("Monitor ready.");

    state.pollTimer = setInterval(() => {
      void pollAll();
    }, 1300);
    state.feedTimer = setInterval(() => {
      void refreshFeeds();
    }, 420);
  }

  window.addEventListener("beforeunload", () => {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    if (state.feedTimer) {
      clearInterval(state.feedTimer);
      state.feedTimer = null;
    }
    revokeUrl(state.liveObjectUrl);
    state.liveObjectUrl = "";
  });

  document.addEventListener("DOMContentLoaded", init);
})();
