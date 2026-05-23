(function () {
  const STORAGE_KEY = "dexsent_operator_context_v1";

  const state = {
    initialized: false,
    stations: [],
    assets: [],
    cameraIds: [],
    currentStationId: "",
    currentAssetId: "",
    listeners: [],
    statusTimer: null,
    boundSelectors: false,
    boundVisionTransport: false,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function readStoredContext() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      state.currentStationId = String(parsed.station_id || "");
      state.currentAssetId = String(parsed.asset_id || "");
    } catch (_) {
      // Ignore local storage failures.
    }
  }

  function persistContext() {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          station_id: state.currentStationId,
          asset_id: state.currentAssetId,
        })
      );
    } catch (_) {
      // Ignore local storage failures.
    }
  }

  function assetId(item) {
    if (!item || typeof item !== "object") return "";
    return String(item.asset_id || item.process_id || "").trim();
  }

  function fillSelect(id, items, getValue, getLabel, selectedValue) {
    const sel = byId(id);
    if (!sel) return;
    sel.innerHTML = "";
    (items || []).forEach((item) => {
      const opt = document.createElement("option");
      opt.value = getValue(item);
      opt.textContent = getLabel(item);
      sel.appendChild(opt);
    });
    if (selectedValue && Array.from(sel.options).some((o) => o.value === selectedValue)) {
      sel.value = selectedValue;
      return;
    }
    if (sel.options.length) {
      sel.selectedIndex = 0;
    }
  }

  function setLed(id, mode) {
    const el = byId(id);
    if (!el) return;
    el.classList.remove("on", "off", "error");
    el.classList.add(mode || "off");
  }

  function setText(id, text) {
    const el = byId(id);
    if (el) el.textContent = text;
  }

  function setInfo(text) {
    const uiInfo = byId("uiInfo");
    if (uiInfo) {
      uiInfo.textContent = text || "";
      return;
    }
    const pageInfo = byId("pageInfo");
    if (pageInfo) {
      pageInfo.textContent = text || "";
    }
  }

  function highlightNav() {
    const page = String(document.body.getAttribute("data-page") || "");
    document.querySelectorAll("[data-nav]").forEach((el) => {
      const nav = String(el.getAttribute("data-nav") || "");
      el.classList.toggle("active", nav === page);
    });
  }

  function getContext() {
    return {
      stationId: state.currentStationId,
      assetId: state.currentAssetId,
      stations: state.stations.slice(),
      assets: state.assets.slice(),
      cameraIds: state.cameraIds.slice(),
    };
  }

  function notifyContext() {
    const snapshot = getContext();
    state.listeners.forEach((listener) => {
      try {
        listener(snapshot);
      } catch (_) {
        // Ignore listener errors.
      }
    });
  }

  function onContextChanged(listener) {
    if (typeof listener === "function") {
      state.listeners.push(listener);
    }
  }

  async function refreshStations() {
    const res = await window.operatorApi("/stations", {}, { silent: true });
    if (!res.ok || !res.body) {
      setInfo("Cannot reach station list.");
      state.stations = [];
      fillSelect("stationSelect", [], () => "", () => "", "");
      state.currentStationId = "";
      return;
    }

    state.stations = Array.isArray(res.body.stations) ? res.body.stations : [];

    const stationSelect = byId("stationSelect");
    fillSelect(
      "stationSelect",
      state.stations,
      (station) => station.station_id,
      (station) => station.name || station.station_id,
      state.currentStationId
    );

    const stationExists = state.stations.some(
      (station) => station.station_id === state.currentStationId
    );

    if (stationSelect) {
      state.currentStationId = stationSelect.value || "";
    } else if (!stationExists) {
      state.currentStationId = state.stations.length ? state.stations[0].station_id : "";
    }

    persistContext();
  }

  async function refreshAssets() {
    if (!state.currentStationId) {
      state.assets = [];
      state.currentAssetId = "";
      fillSelect("assetSelect", [], () => "", () => "", "");
      persistContext();
      return;
    }

    const res = await window.operatorApi(
      `/stations/${encodeURIComponent(state.currentStationId)}/processes`,
      {},
      { silent: true }
    );

    if (!res.ok || !res.body) {
      setInfo("Failed loading assets.");
      state.assets = [];
      state.currentAssetId = "";
      fillSelect("assetSelect", [], () => "", () => "", "");
      persistContext();
      return;
    }

    state.assets = Array.isArray(res.body.processes) ? res.body.processes : [];

    const assetSelect = byId("assetSelect");
    fillSelect(
      "assetSelect",
      state.assets,
      (asset) => assetId(asset),
      (asset) => asset.name || assetId(asset),
      state.currentAssetId
    );

    const assetExists = state.assets.some((asset) => assetId(asset) === state.currentAssetId);
    if (assetSelect) {
      state.currentAssetId = assetSelect.value || "";
    } else if (!assetExists) {
      state.currentAssetId = state.assets.length ? assetId(state.assets[0]) : "";
    }

    persistContext();
  }

  async function refreshContext() {
    await refreshStations();
    await refreshAssets();
    notifyContext();
  }

  async function refreshSystemStatus() {
    const cam = await window.operatorApi("/camera/cameras", {}, { silent: true });
    if (cam.ok && cam.body) {
      const cams = Array.isArray(cam.body.cameras) ? cam.body.cameras : [];
      state.cameraIds = cams;
      setLed("cameraLed", cams.length ? "on" : "off");
      setText("cameraText", cams.length ? "on" : "off");
    } else {
      state.cameraIds = [];
      setLed("cameraLed", "error");
      setText("cameraText", "off");
    }

    const robot = await window.operatorApi("/robot/state", {}, { silent: true });
    if (!robot.ok || !robot.body) {
      setLed("robotLed", "error");
      setText("robotText", "off");
    } else {
      const mode = String(robot.body.mode || "").toUpperCase();
      const lastError = String(robot.body.last_error || "").toUpperCase();
      const unhealthy =
        !robot.body.connected ||
        mode.includes("DISCONNECT") ||
        mode.includes("ERROR") ||
        mode.includes("FAULT") ||
        mode.includes("COLLISION") ||
        lastError.includes("ERROR") ||
        lastError.includes("DISCONNECT") ||
        lastError.includes("COLLISION");
      setLed("robotLed", unhealthy ? "error" : "on");
      setText("robotText", unhealthy ? "off" : "on");
    }

    const vision = await window.operatorApi("/vision/cameras", {}, { silent: true });
    if (vision.ok && vision.body) {
      const running = !!vision.body.engine_running;
      setLed("visionLed", running ? "on" : "off");
      const transport = String(vision.body.transport || "").toLowerCase();
      setText("visionText", running ? (transport === "websocket" ? "blackwell" : "running") : "off");
    } else {
      setLed("visionLed", "error");
      setText("visionText", "off");
    }

    await refreshVisionTransport();

    const health = await window.operatorApi("/health", {}, { silent: true });
    const healthStatus = health && health.body ? health.body.status : "";
    if (health.ok && String(healthStatus || "").toLowerCase() === "alive") {
      setLed("serverLed", "on");
      setText("serverText", "on");
    } else {
      setLed("serverLed", "error");
      setText("serverText", "off");
    }
  }

  async function refreshVisionTransport() {
    const select = byId("visionTransportSelect");
    if (!select) return;
    const res = await window.operatorApi("/runtime/vision-transport", {}, { silent: true });
    if (!res.ok || !res.body) return;
    const transport = String(res.body.transport || "zmq").toLowerCase();
    select.value = transport === "websocket" ? "websocket" : "zmq";
  }

  function bindContextSelectors() {
    if (state.boundSelectors) return;

    const stationSelect = byId("stationSelect");
    if (stationSelect) {
      stationSelect.addEventListener("change", async (evt) => {
        state.currentStationId = String(evt.target.value || "");
        state.currentAssetId = "";
        persistContext();
        await refreshAssets();
        notifyContext();
      });
    }

    const assetSelect = byId("assetSelect");
    if (assetSelect) {
      assetSelect.addEventListener("change", (evt) => {
        state.currentAssetId = String(evt.target.value || "");
        persistContext();
        notifyContext();
      });
    }

    state.boundSelectors = true;
  }

  function bindVisionTransportSelector() {
    if (state.boundVisionTransport) return;
    const select = byId("visionTransportSelect");
    if (!select) return;
    select.addEventListener("change", async (event) => {
      const transport = String(event.target.value || "zmq");
      const res = await window.operatorApi("/runtime/vision-transport", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transport }),
      });
      if (!res.ok) {
        await refreshVisionTransport();
        setInfo("Vision transport update failed.");
        return;
      }
      setInfo(transport === "websocket" ? "Vision mode: blackwell." : "Vision mode: local.");
      await refreshSystemStatus();
    });
    state.boundVisionTransport = true;
  }

  async function init(options) {
    const opts = options || {};
    if (typeof opts.onContextChanged === "function") {
      onContextChanged(opts.onContextChanged);
    }

    if (!state.initialized) {
      readStoredContext();
      highlightNav();
      bindContextSelectors();
      bindVisionTransportSelector();
      state.initialized = true;
    }

    await refreshContext();
    await refreshSystemStatus();

    if (!state.statusTimer) {
      state.statusTimer = setInterval(() => {
        refreshSystemStatus();
      }, 3000);
    }
  }

  window.addEventListener("beforeunload", () => {
    if (state.statusTimer) {
      clearInterval(state.statusTimer);
      state.statusTimer = null;
    }
  });

  window.operatorShell = {
    state,
    byId,
    fillSelect,
    assetId,
    setText,
    setInfo,
    init,
    refreshContext,
    refreshStations,
    refreshAssets,
    refreshSystemStatus,
    getContext,
    onContextChanged,
  };
})();
