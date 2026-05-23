(function () {
  const statusBox = () => document.getElementById("systemStatus");
  const reconnectState = {
    started: false,
    wasOffline: false,
    startupToken: null,
    timer: null,
  };

  async function api(path, options, cfg) {
    const opts = options || {};
    const meta = cfg || {};
    const method = opts.method || "GET";
    const silent = meta.silent === true;
    try {
      if (!silent && statusBox()) statusBox().textContent = `request ${method}`;
      const res = await fetch(path, opts);
      const text = await res.text();
      let body = null;
      try {
        body = text ? JSON.parse(text) : null;
      } catch (_) {
        body = { raw: text };
      }
      if (!silent && statusBox()) {
        statusBox().textContent = res.ok ? "ready" : `error ${res.status}`;
      }
      return { ok: res.ok, status: res.status, body };
    } catch (err) {
      if (!silent && statusBox()) statusBox().textContent = "offline";
      return { ok: false, status: 0, body: null, error: String(err) };
    }
  }

  async function probeReady() {
    try {
      const res = await fetch("/ready", { cache: "no-store" });
      if (!res.ok) {
        reconnectState.wasOffline = true;
        return;
      }
      const body = await res.json();
      const token = String(body && body.startup_token || "");
      if (!token) {
        reconnectState.wasOffline = false;
        return;
      }
      if (!reconnectState.startupToken) {
        reconnectState.startupToken = token;
        reconnectState.wasOffline = false;
        return;
      }
      if (reconnectState.startupToken !== token) {
        window.location.reload();
        return;
      }
      reconnectState.wasOffline = false;
    } catch (_) {
      reconnectState.wasOffline = true;
    }
  }

  function startReconnectWatcher() {
    if (reconnectState.started) return;
    reconnectState.started = true;
    void probeReady();
    reconnectState.timer = window.setInterval(() => {
      void probeReady();
    }, 2000);
    window.addEventListener("online", () => {
      void probeReady();
    });
  }

  window.operatorApi = api;
  startReconnectWatcher();
})();
