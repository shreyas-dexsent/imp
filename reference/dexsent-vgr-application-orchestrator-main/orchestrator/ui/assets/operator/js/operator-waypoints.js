(function () {
  const state = {
    poses: [],
    poseTimer: null,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function context() {
    if (!window.operatorShell || typeof window.operatorShell.getContext !== "function") {
      return { assetId: "" };
    }
    return window.operatorShell.getContext();
  }

  function log(message) {
    const host = byId("waypointLog");
    if (!host) return;
    const stamp = new Date().toLocaleTimeString();
    host.textContent = `[${stamp}] ${message}\n${host.textContent || ""}`.slice(0, 5000);
  }

  function safeNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  }

  function formatFixed(value, digits) {
    return String(safeNumber(value, 0).toFixed(digits));
  }

  function quatToRpyDeg(quat) {
    const x = safeNumber(quat && quat[0], 0);
    const y = safeNumber(quat && quat[1], 0);
    const z = safeNumber(quat && quat[2], 0);
    const w = safeNumber(quat && quat[3], 1);

    const sinrCosp = 2 * (w * x + y * z);
    const cosrCosp = 1 - 2 * (x * x + y * y);
    const roll = Math.atan2(sinrCosp, cosrCosp);

    const sinp = 2 * (w * y - z * x);
    const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * (Math.PI / 2) : Math.asin(sinp);

    const sinyCosp = 2 * (w * z + x * y);
    const cosyCosp = 1 - 2 * (y * y + z * z);
    const yaw = Math.atan2(sinyCosp, cosyCosp);

    return [
      (roll * 180) / Math.PI,
      (pitch * 180) / Math.PI,
      (yaw * 180) / Math.PI,
    ];
  }

  function rpyDegToQuat(rpyDeg) {
    const roll = safeNumber(rpyDeg && rpyDeg[0], 0) * Math.PI / 180.0;
    const pitch = safeNumber(rpyDeg && rpyDeg[1], 0) * Math.PI / 180.0;
    const yaw = safeNumber(rpyDeg && rpyDeg[2], 0) * Math.PI / 180.0;
    const cy = Math.cos(yaw * 0.5);
    const sy = Math.sin(yaw * 0.5);
    const cp = Math.cos(pitch * 0.5);
    const sp = Math.sin(pitch * 0.5);
    const cr = Math.cos(roll * 0.5);
    const sr = Math.sin(roll * 0.5);
    const qw = cr * cp * cy + sr * sp * sy;
    const qx = sr * cp * cy - cr * sp * sy;
    const qy = cr * sp * cy + sr * cp * sy;
    const qz = cr * cp * sy - sr * sp * cy;
    const norm = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) || 1;
    return [qx / norm, qy / norm, qz / norm, qw / norm];
  }

  function setPoseInputs(tcpPose) {
    if (!tcpPose) return;
    const pos = Array.isArray(tcpPose.position_m) ? tcpPose.position_m : [0, 0, 0];
    const rpy = Array.isArray(tcpPose.rotation_rpy_deg) && tcpPose.rotation_rpy_deg.length >= 3
      ? tcpPose.rotation_rpy_deg
      : quatToRpyDeg(tcpPose.quat_xyzw || [0, 0, 0, 1]);
    if (byId("poseXInput")) byId("poseXInput").value = formatFixed(pos[0], 4);
    if (byId("poseYInput")) byId("poseYInput").value = formatFixed(pos[1], 4);
    if (byId("poseZInput")) byId("poseZInput").value = formatFixed(pos[2], 4);
    if (byId("poseRollInput")) byId("poseRollInput").value = formatFixed(rpy[0], 2);
    if (byId("posePitchInput")) byId("posePitchInput").value = formatFixed(rpy[1], 2);
    if (byId("poseYawInput")) byId("poseYawInput").value = formatFixed(rpy[2], 2);
  }

  function readPoseInputs() {
    return {
      position_m: [
        safeNumber(byId("poseXInput") && byId("poseXInput").value, 0),
        safeNumber(byId("poseYInput") && byId("poseYInput").value, 0),
        safeNumber(byId("poseZInput") && byId("poseZInput").value, 0),
      ],
      rotation_rpy_deg: [
        safeNumber(byId("poseRollInput") && byId("poseRollInput").value, 0),
        safeNumber(byId("posePitchInput") && byId("posePitchInput").value, 0),
        safeNumber(byId("poseYawInput") && byId("poseYawInput").value, 0),
      ],
    };
  }

  function angleStepDeg() {
    return safeNumber(byId("angleStep") && byId("angleStep").value, 5);
  }

  async function refreshCurrentPose(silent = false) {
    const robot = await window.operatorApi("/robot/state", {}, { silent: true });
    if (!robot.ok || !robot.body) {
      if (!silent) log("Robot state unavailable.");
      return null;
    }
    const displayPose = robot.body.custom_tcp_pose;
    if (!displayPose || !Array.isArray(displayPose.position_m)) {
      if (!silent) log(robot.body.custom_tcp_error || "Custom TCP pose unavailable.");
      return null;
    }
    setPoseInputs(displayPose);
    return robot.body;
  }

  function renderPoses() {
    const select = byId("poseSelect");
    if (!select) return;
    const current = select.value;
    select.innerHTML = "";
    state.poses.forEach((pose) => {
      const option = document.createElement("option");
      option.value = pose.name;
      option.textContent = pose.name;
      select.appendChild(option);
    });
    if (current && Array.from(select.options).some((opt) => opt.value === current)) {
      select.value = current;
    }
  }

  async function refreshPoses() {
    const ctx = context();
    if (!ctx.assetId) {
      state.poses = [];
      renderPoses();
      log("Select asset to load poses.");
      return;
    }

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/poses`,
      {},
      { silent: true }
    );

    if (!res.ok || !res.body) {
      state.poses = [];
      renderPoses();
      log("Failed loading poses.");
      return;
    }

    state.poses = Array.isArray(res.body.poses) ? res.body.poses : [];
    renderPoses();
    log(`Loaded ${state.poses.length} pose(s).`);
  }

  async function nudge(axis, dir) {
    const robot = await window.operatorApi("/robot/state", {}, { silent: true });
    if (!robot.ok || !robot.body || !robot.body.custom_tcp_pose || !robot.body.custom_tcp_pose.position_m) {
      log(robot.body?.custom_tcp_error || "Custom TCP pose unavailable.");
      return;
    }

    const step = Number((byId("stepSize") && byId("stepSize").value) || 0.01);
    const pos = robot.body.custom_tcp_pose.position_m.slice();
    const idx = axis === "x" ? 0 : axis === "y" ? 1 : 2;
    pos[idx] += step * dir;

    const payload = {
      position_m: pos,
      quat_xyzw: robot.body.custom_tcp_pose.quat_xyzw || [0, 0, 0, 1],
      frame: robot.body.custom_tcp_pose.frame || "base",
      profile: "slow",
    };

    const res = await window.operatorApi("/robot/movel_custom_tcp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      log("Robot jog failed.");
      return;
    }
    // Refresh display in custom_tcp_pose frame.
    await refreshCurrentPose(true);
    log(`Jog ${axis.toUpperCase()} ${dir > 0 ? "+" : "-"}${step.toFixed(3)} m`);
  }

  async function nudgeRotation(axis, dir) {
    const robot = await window.operatorApi("/robot/state", {}, { silent: true });
    if (!robot.ok || !robot.body || !robot.body.custom_tcp_pose || !robot.body.custom_tcp_pose.position_m) {
      log(robot.body?.custom_tcp_error || "Custom TCP pose unavailable.");
      return;
    }

    const pose = robot.body.custom_tcp_pose;
    const rpy = Array.isArray(pose.rotation_rpy_deg) && pose.rotation_rpy_deg.length >= 3
      ? pose.rotation_rpy_deg.slice(0, 3)
      : quatToRpyDeg(pose.quat_xyzw || [0, 0, 0, 1]);
    const step = angleStepDeg();
    const idx = axis === "roll" ? 0 : axis === "pitch" ? 1 : 2;
    rpy[idx] += step * dir;

    const payload = {
      position_m: Array.isArray(pose.position_m) ? pose.position_m.slice() : [0, 0, 0],
      quat_xyzw: rpyDegToQuat(rpy),
      frame: pose.frame || "base",
      profile: "slow",
    };

    const res = await window.operatorApi("/robot/movel_custom_tcp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      log("Robot jog failed.");
      return;
    }
    // Refresh display in custom_tcp_pose frame.
    await refreshCurrentPose(true);
    log(`Jog ${axis} ${dir > 0 ? "+" : "-"}${step.toFixed(1)} deg`);
  }

  async function openGripper() {
    const res = await window.operatorApi("/robot/gripper/open", { method: "POST" });
    log(res.ok ? "Gripper open command sent." : "Gripper open failed.");
  }

  async function closeGripper() {
    const res = await window.operatorApi("/robot/gripper/close", { method: "POST" });
    log(res.ok ? "Gripper close command sent." : "Gripper close failed.");
  }

  async function stopRobot() {
    const res = await window.operatorApi("/robot/stop", { method: "POST" });
    log(res.ok ? "Robot stop command sent." : "Robot stop failed.");
  }

  async function moveManualPose() {
    const manual = readPoseInputs();
    // Inputs are in custom_tcp frame — backend converts to EE before movel.
    const payload = {
      position_m: manual.position_m,
      quat_xyzw: rpyDegToQuat(manual.rotation_rpy_deg),
      frame: "base",
      profile: "slow",
    };
    const res = await window.operatorApi("/robot/movel_custom_tcp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      log("Move manual TCP failed.");
      return;
    }
    await refreshCurrentPose(true);
    log("Moved robot to manual custom TCP target.");
  }

  async function savePose() {
    const ctx = context();
    if (!ctx.assetId) {
      log("Select asset before saving pose.");
      return;
    }

    const name = (byId("poseName") && byId("poseName").value.trim()) || "";
    const mode = (byId("poseMode") && byId("poseMode").value) || "auto";

    if (!name) {
      log("Pose name required.");
      return;
    }

    if (!window.confirm(`Save pose "${name}" in ${mode} mode?`)) {
      return;
    }

    // Don't pass position — let the backend read current robot state and apply the
    // custom TCP offset. This ensures the saved pose is always in the custom_tcp frame.
    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/poses`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, mode }),
      }
    );

    if (!res.ok) {
      const detail = res.body && (res.body.detail || res.body.error || res.body.raw);
      log(`Save pose failed${detail ? `: ${detail}` : "."}`);
      return;
    }

    log(`Pose saved: ${name}`);
    await refreshPoses();
  }

  async function moveToPose() {
    const name = (byId("poseSelect") && byId("poseSelect").value) || "";
    if (!name) {
      log("Select pose to move.");
      return;
    }

    const pose = state.poses.find((item) => item.name === name);
    if (!pose) {
      log("Pose not found in cache.");
      return;
    }

    if (pose.tcp_pose && Array.isArray(pose.tcp_pose.position_m)) {
      const movel = await window.operatorApi("/robot/movel_custom_tcp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          position_m: pose.tcp_pose.position_m,
          quat_xyzw: pose.tcp_pose.quat_xyzw || [0, 0, 0, 1],
          frame: pose.tcp_pose.frame || "base",
          profile: "slow",
        }),
      });
      log(movel.ok ? `MoveL to ${name}` : `MoveL failed for ${name}`);
      return;
    }

    if (Array.isArray(pose.joints) && pose.joints.length) {
      const movej = await window.operatorApi("/robot/movej", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ joints: pose.joints, profile: "slow" }),
      });
      log(movej.ok ? `MoveJ to ${name}` : `MoveJ failed for ${name}`);
      return;
    }

    log("Pose has no joints or tcp_pose.");
  }

  async function renamePose() {
    const ctx = context();
    if (!ctx.assetId) {
      log("Select asset first.");
      return;
    }
    const currentName = (byId("poseSelect") && byId("poseSelect").value) || "";
    if (!currentName) {
      log("Select pose to rename.");
      return;
    }

    const nextName = window.prompt("New pose name:", currentName);
    if (!nextName || !nextName.trim() || nextName.trim() === currentName) {
      return;
    }

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/poses/${encodeURIComponent(currentName)}/rename`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: nextName.trim() }),
      }
    );

    if (!res.ok) {
      log("Rename pose failed.");
      return;
    }

    log(`Pose renamed: ${currentName} -> ${nextName.trim()}`);
    await refreshPoses();
  }

  async function deletePose() {
    const ctx = context();
    if (!ctx.assetId) {
      log("Select asset first.");
      return;
    }

    const name = (byId("poseSelect") && byId("poseSelect").value) || "";
    if (!name) {
      log("Select pose to delete.");
      return;
    }

    if (!window.confirm(`Delete pose "${name}"?`)) {
      return;
    }

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/poses/${encodeURIComponent(name)}`,
      { method: "DELETE" }
    );

    if (!res.ok) {
      log("Delete pose failed.");
      return;
    }

    log(`Pose deleted: ${name}`);
    await refreshPoses();
  }

  function bindEvents() {
    const map = [
      ["nudgeXp", () => nudge("x", 1)],
      ["nudgeYp", () => nudge("y", 1)],
      ["nudgeZp", () => nudge("z", 1)],
      ["nudgeXn", () => nudge("x", -1)],
      ["nudgeYn", () => nudge("y", -1)],
      ["nudgeZn", () => nudge("z", -1)],
      ["nudgeRollP", () => nudgeRotation("roll", 1)],
      ["nudgePitchP", () => nudgeRotation("pitch", 1)],
      ["nudgeYawP", () => nudgeRotation("yaw", 1)],
      ["nudgeRollN", () => nudgeRotation("roll", -1)],
      ["nudgePitchN", () => nudgeRotation("pitch", -1)],
      ["nudgeYawN", () => nudgeRotation("yaw", -1)],
      ["gripperOpenBtn", openGripper],
      ["gripperCloseBtn", closeGripper],
      ["robotStopBtn", stopRobot],
      ["moveManualPoseBtn", moveManualPose],
      ["savePoseBtn", savePose],
      ["refreshPoseBtn", refreshPoses],
      ["movePoseBtn", moveToPose],
      ["renamePoseBtn", renamePose],
      ["deletePoseBtn", deletePose],
    ];

    map.forEach(([id, fn]) => {
      const el = byId(id);
      if (el) el.addEventListener("click", fn);
    });

    const refreshContextBtn = byId("refreshContextBtn");
    if (refreshContextBtn && window.operatorShell) {
      refreshContextBtn.addEventListener("click", async () => {
        await window.operatorShell.refreshContext();
        await refreshPoses();
      });
    }
  }

  async function init() {
    bindEvents();

    if (window.operatorShell && typeof window.operatorShell.init === "function") {
      await window.operatorShell.init({
        onContextChanged: async () => {
          await refreshPoses();
        },
      });
    }

    await refreshPoses();
    await refreshCurrentPose(true);
    state.poseTimer = window.setInterval(() => {
      void refreshCurrentPose(true);
    }, 2000);
    if (window.operatorShell) {
      window.operatorShell.setInfo("Robot waypoint manager ready.");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
