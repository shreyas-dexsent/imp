/**
 * operator-robotviz.js
 * Page controller for /ui/operator/robot-viz.
 * Waits for an asset to be selected, then mounts the robot visualizer inline
 * (not as a modal) inside #robotVizPageMount.
 */

(function () {
  "use strict";

  let vizMounted = false;
  let lastAssetId = "";
  let lastTaskId = "";
  let tasksLoadedForAsset = "";

  function currentAssetId() {
    return String(
      window.operatorShell?.state?.currentAssetId
      || document.getElementById("assetSelect")?.value
      || ""
    ).trim();
  }

  function currentTaskId() {
    return String(document.getElementById("taskSelect")?.value || "").trim();
  }

  async function loadTasks(assetId) {
    const select = document.getElementById("taskSelect");
    if (!select || !assetId) return;
    if (tasksLoadedForAsset === assetId && select.options.length) return;
    let tasks = [];
    try {
      const res = await (window.operatorApi
        ? window.operatorApi(`/processes/${encodeURIComponent(assetId)}/tasks`, {}, { silent: true })
        : fetch(`/processes/${encodeURIComponent(assetId)}/tasks`).then(async (r) => ({ ok: r.ok, body: await r.json().catch(() => null) })));
      if (res.ok && res.body && Array.isArray(res.body.tasks)) tasks = res.body.tasks;
    } catch (_) {
      tasks = [];
    }
    const previous = select.value;
    select.innerHTML = "";
    const noneOpt = document.createElement("option");
    noneOpt.value = "";
    noneOpt.textContent = "(none — task-type defaults)";
    select.appendChild(noneOpt);
    tasks.forEach((task) => {
      const opt = document.createElement("option");
      opt.value = String(task.task_id || "");
      opt.textContent = String(task.name || task.task_id || "");
      opt.dataset.taskType = String(task.task_type || "");
      select.appendChild(opt);
    });
    if (previous && Array.from(select.options).some((o) => o.value === previous)) {
      select.value = previous;
    } else if (select.options.length > 1) {
      select.selectedIndex = 1;
    }
    tasksLoadedForAsset = assetId;
    syncGlobalsFromTaskSelect();
  }

  function syncGlobalsFromTaskSelect() {
    const select = document.getElementById("taskSelect");
    const opt = select?.selectedOptions?.[0];
    window.currentTaskId = String(select?.value || "");
    window.currentTaskType = String(opt?.dataset?.taskType || "");
  }

  async function mountVizWhenReady() {
    const assetId = currentAssetId();
    if (!assetId) return;
    await loadTasks(assetId);
    syncGlobalsFromTaskSelect();
    const taskId = currentTaskId();
    if (assetId === lastAssetId && taskId === lastTaskId && vizMounted) return;
    lastAssetId = assetId;
    lastTaskId = taskId;
    vizMounted = false;
    if (window.operatorBinPickingTools?.mountRobotVisualizer) {
      window.operatorBinPickingTools.mountRobotVisualizer(
        document.getElementById("robotVizPageMount")
      );
      vizMounted = true;
    }
  }

  function resetAndMount() {
    vizMounted = false;
    lastAssetId = "";
    lastTaskId = "";
    tasksLoadedForAsset = "";
    mountVizWhenReady();
  }

  function remountForTaskChange() {
    syncGlobalsFromTaskSelect();
    vizMounted = false;
    lastAssetId = "";
    lastTaskId = "";
    mountVizWhenReady();
  }

  async function init() {
    const assetSelect = document.getElementById("assetSelect");
    if (assetSelect) {
      assetSelect.addEventListener("change", resetAndMount);
    }
    const taskSelect = document.getElementById("taskSelect");
    if (taskSelect) {
      taskSelect.addEventListener("change", remountForTaskChange);
    }
    document.addEventListener("operatorContextReady", mountVizWhenReady);
    document.addEventListener("operatorAssetChanged", resetAndMount);
    document.addEventListener("operatorTaskChanged", resetAndMount);
    if (window.operatorShell?.init) {
      try {
        await window.operatorShell.init({ onContextChanged: () => { resetAndMount(); } });
      } catch (_) {
        // Shell init failures shouldn't block the viz from retrying below.
      }
    } else if (window.operatorShell?.onContextChanged) {
      window.operatorShell.onContextChanged(() => { resetAndMount(); });
    }
    // Retry in case bin-picking-tools.js module hasn't initialised yet.
    setTimeout(mountVizWhenReady, 400);
    setTimeout(mountVizWhenReady, 1200);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
