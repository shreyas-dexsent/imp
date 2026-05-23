(function () {
  const state = {
    objects: [],
    captureImage: null,
    captureProposals: [],
    selectedProposalIndex: null,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function context() {
    if (!window.operatorShell || typeof window.operatorShell.getContext !== "function") {
      return { assetId: "", cameraIds: [] };
    }
    return window.operatorShell.getContext();
  }

  function log(message) {
    const host = byId("objectLog");
    if (!host) return;
    const stamp = new Date().toLocaleTimeString();
    host.textContent = `[${stamp}] ${message}\n${host.textContent || ""}`.slice(0, 6000);
  }

  function clearCanvas() {
    const canvas = byId("snapshot");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawCapture() {
    const canvas = byId("snapshot");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (!state.captureImage) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      state.captureProposals.forEach((proposal, idx) => {
        const bbox = proposal.bbox_xywh || [0, 0, 0, 0];
        const x = Number(bbox[0] || 0);
        const y = Number(bbox[1] || 0);
        const w = Number(bbox[2] || 0);
        const h = Number(bbox[3] || 0);
        ctx.strokeStyle = idx === state.selectedProposalIndex ? "#12c6cf" : "#115a9b";
        ctx.lineWidth = idx === state.selectedProposalIndex ? 3 : 2;
        if (proposal.obb_points && proposal.obb_points.length === 4) {
          ctx.beginPath();
          ctx.moveTo(proposal.obb_points[0][0], proposal.obb_points[0][1]);
          ctx.lineTo(proposal.obb_points[1][0], proposal.obb_points[1][1]);
          ctx.lineTo(proposal.obb_points[2][0], proposal.obb_points[2][1]);
          ctx.lineTo(proposal.obb_points[3][0], proposal.obb_points[3][1]);
          ctx.closePath();
          ctx.stroke();
        } else {
          ctx.strokeRect(x, y, w, h);
        }
      });
    };
    img.src = state.captureImage;
  }

  function computeObbPoints(bbox, yawDeg) {
    const x = Number(bbox[0] || 0);
    const y = Number(bbox[1] || 0);
    const w = Number(bbox[2] || 0);
    const h = Number(bbox[3] || 0);

    const cx = x + w / 2;
    const cy = y + h / 2;
    const hw = w / 2;
    const hh = h / 2;
    const rad = (Number(yawDeg) || 0) * (Math.PI / 180);
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const points = [
      [-hw, -hh],
      [hw, -hh],
      [hw, hh],
      [-hw, hh],
    ];

    return points.map(([dx, dy]) => [
      Math.round(cx + dx * cos - dy * sin),
      Math.round(cy + dx * sin + dy * cos),
    ]);
  }

  function pointInPolygon(x, y, points) {
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const xi = points[i][0];
      const yi = points[i][1];
      const xj = points[j][0];
      const yj = points[j][1];
      const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 0.00001) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function selectProposal(index) {
    state.selectedProposalIndex = index;
    const selected = state.captureProposals[index];
    if (selected && selected.bbox_xywh) {
      const bbox = selected.bbox_xywh;
      byId("bboxX").value = Math.round(Number(bbox[0] || 0));
      byId("bboxY").value = Math.round(Number(bbox[1] || 0));
      byId("bboxW").value = Math.round(Number(bbox[2] || 0));
      byId("bboxH").value = Math.round(Number(bbox[3] || 0));
      byId("bboxYaw").value = selected.yaw_deg ? Number(selected.yaw_deg).toFixed(1) : "0";
    }
    drawCapture();
  }

  function renderProposals() {
    const host = byId("proposalList");
    if (!host) return;
    host.innerHTML = "";

    if (!state.captureProposals.length) {
      host.textContent = "Capture a frame to see proposals.";
      return;
    }

    state.captureProposals.forEach((proposal, idx) => {
      const row = document.createElement("div");
      row.className = "proposal-item";

      const left = document.createElement("span");
      const score = proposal.score ? Number(proposal.score).toFixed(0) : "0";
      const yaw = proposal.yaw_deg ? Number(proposal.yaw_deg).toFixed(1) : "0.0";
      left.textContent = `#${idx + 1} yaw=${yaw} score=${score}`;
      row.appendChild(left);

      const selectBtn = document.createElement("button");
      selectBtn.className = "secondary";
      selectBtn.textContent = "Select";
      selectBtn.type = "button";
      selectBtn.addEventListener("click", () => selectProposal(idx));
      row.appendChild(selectBtn);

      host.appendChild(row);
    });
  }

  async function cropCapture(bbox) {
    const x = Math.max(0, Math.round(Number(bbox[0] || 0)));
    const y = Math.max(0, Math.round(Number(bbox[1] || 0)));
    const w = Math.max(1, Math.round(Number(bbox[2] || 0)));
    const h = Math.max(1, Math.round(Number(bbox[3] || 0)));
    if (!state.captureImage || w <= 0 || h <= 0) return null;

    return await new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const maxW = img.naturalWidth || img.width;
        const maxH = img.naturalHeight || img.height;
        const cropX = Math.min(maxW - 1, x);
        const cropY = Math.min(maxH - 1, y);
        const cropW = Math.min(maxW - cropX, w);
        const cropH = Math.min(maxH - cropY, h);
        const off = document.createElement("canvas");
        off.width = cropW;
        off.height = cropH;
        const ctx = off.getContext("2d");
        ctx.drawImage(img, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
        resolve(off.toDataURL("image/png"));
      };
      img.onerror = () => resolve(null);
      img.src = state.captureImage;
    });
  }

  async function captureSegment() {
    const cameraId = (byId("captureCameraId") && byId("captureCameraId").value.trim()) || "";
    if (!cameraId) {
      log("Camera ID is required.");
      return;
    }

    const payload = {
      camera_id: cameraId,
      module: "object_proposals",
      timeout_s: 3,
      params: {
        method: "auto",
        color_method: "background",
        combine_mode: "depth",
        min_area_px: 800,
        max_area_px: 300000,
        max_proposals: 6,
        min_extent: 0.25,
        min_solidity: 0.65,
        nms_iou_threshold: 0.35,
        depth_plane_source: "border",
        depth_plane_fit: true,
        depth_plane_bins: 140,
        depth_plane_border_px: 24,
        depth_smooth_ksize: 3,
        depth_object_min_height_m: 0.008,
        depth_object_max_height_m: 0.2,
        bg_delta: 18,
        bg_delta_min: 12,
        bg_delta_scale: 2.5,
        bg_border_px: 24,
        include_image: true,
        format: "jpg",
        quality: 85,
      },
    };

    const res = await window.operatorApi("/vision/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok || !res.body || !res.body.result) {
      log("Capture failed.");
      return;
    }

    const result = res.body.result;
    const fmt = result.format || "jpg";
    state.captureImage = result.image_b64 ? `data:image/${fmt};base64,${result.image_b64}` : null;
    state.captureProposals = Array.isArray(result.proposals) ? result.proposals : [];
    state.selectedProposalIndex = null;

    renderProposals();
    drawCapture();
    log(`Capture complete. proposals=${state.captureProposals.length}`);
  }

  function clearCapture() {
    state.captureImage = null;
    state.captureProposals = [];
    state.selectedProposalIndex = null;
    renderProposals();
    clearCanvas();
    log("Capture cleared.");
  }

  function applyBoxEdit() {
    if (state.selectedProposalIndex === null) return;
    const x = Number(byId("bboxX").value);
    const y = Number(byId("bboxY").value);
    const w = Number(byId("bboxW").value);
    const h = Number(byId("bboxH").value);
    const yaw = Number(byId("bboxYaw").value || 0);
    if ([x, y, w, h].some((v) => Number.isNaN(v))) {
      log("Invalid bbox values.");
      return;
    }

    const bbox = [
      Math.max(0, Math.round(x)),
      Math.max(0, Math.round(y)),
      Math.max(1, Math.round(w)),
      Math.max(1, Math.round(h)),
    ];

    const proposal = state.captureProposals[state.selectedProposalIndex];
    proposal.bbox_xywh = bbox;
    proposal.yaw_deg = Number.isNaN(yaw) ? 0 : yaw;
    proposal.obb_size = [bbox[2], bbox[3]];
    proposal.obb_points = computeObbPoints(bbox, yaw);
    drawCapture();
    renderProposals();
    log("BBox updated.");
  }

  async function readFileAsDataUrl(file) {
    return await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => resolve("");
      reader.readAsDataURL(file);
    });
  }

  async function saveTemplate() {
    const ctx = context();
    if (!ctx.assetId) {
      log("Select asset before saving template.");
      return;
    }

    const objectId = (byId("objectId") && byId("objectId").value.trim()) || "";
    const templateName = (byId("templateName") && byId("templateName").value.trim()) || "";
    const fileInput = byId("templateFile");
    const file = fileInput && fileInput.files && fileInput.files.length ? fileInput.files[0] : null;

    if (!objectId || !templateName) {
      log("Object ID and template name are required.");
      return;
    }

    let imageData = "";
    if (file) {
      imageData = await readFileAsDataUrl(file);
    } else {
      if (state.captureImage && state.selectedProposalIndex !== null) {
        const selected = state.captureProposals[state.selectedProposalIndex];
        const bbox = selected && selected.bbox_xywh ? selected.bbox_xywh : null;
        if (bbox) {
          imageData = (await cropCapture(bbox)) || "";
        }
      }
    }

    if (!imageData) {
      log("Capture and select a proposal, or upload an image.");
      return;
    }

    const payload = {
      object_id: objectId,
      template_name: templateName,
      image_b64: imageData,
      ext: "png",
    };

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects/templates/upload`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    if (!res.ok) {
      log("Template save failed.");
      return;
    }

    log(`Template saved: ${objectId}/${templateName}`);
    await refreshObjects();
  }

  function renderObjects() {
    const host = byId("objectLibrary");
    if (!host) return;
    host.innerHTML = "";

    if (!state.objects.length) {
      host.textContent = "No templates yet.";
      return;
    }

    state.objects.forEach((objectItem) => {
      const objectRow = document.createElement("div");
      objectRow.className = "object-item";

      const left = document.createElement("span");
      const templateCount = Array.isArray(objectItem.templates) ? objectItem.templates.length : 0;
      left.textContent = `${objectItem.object_id} (${templateCount})`;
      objectRow.appendChild(left);

      const controls = document.createElement("div");
      controls.className = "actions";

      const renameObjectBtn = document.createElement("button");
      renameObjectBtn.className = "ghost";
      renameObjectBtn.type = "button";
      renameObjectBtn.textContent = "Rename";
      renameObjectBtn.addEventListener("click", () => renameObject(objectItem.object_id));
      controls.appendChild(renameObjectBtn);

      const deleteObjectBtn = document.createElement("button");
      deleteObjectBtn.className = "ghost";
      deleteObjectBtn.type = "button";
      deleteObjectBtn.textContent = "Delete";
      deleteObjectBtn.addEventListener("click", () => deleteObject(objectItem.object_id));
      controls.appendChild(deleteObjectBtn);

      objectRow.appendChild(controls);
      host.appendChild(objectRow);

      (objectItem.templates || []).forEach((template) => {
        const templateRow = document.createElement("div");
        templateRow.className = "template-item";

        const templateName = template.filename || `${template.name}.${template.ext}`;
        const nameEl = document.createElement("span");
        nameEl.textContent = templateName;
        templateRow.appendChild(nameEl);

        const templateControls = document.createElement("div");
        templateControls.className = "actions";

        const renameTemplateBtn = document.createElement("button");
        renameTemplateBtn.className = "ghost";
        renameTemplateBtn.type = "button";
        renameTemplateBtn.textContent = "Rename";
        renameTemplateBtn.addEventListener("click", () => renameTemplate(objectItem.object_id, templateName));
        templateControls.appendChild(renameTemplateBtn);

        const deleteTemplateBtn = document.createElement("button");
        deleteTemplateBtn.className = "ghost";
        deleteTemplateBtn.type = "button";
        deleteTemplateBtn.textContent = "Delete";
        deleteTemplateBtn.addEventListener("click", () => deleteTemplate(objectItem.object_id, templateName));
        templateControls.appendChild(deleteTemplateBtn);

        templateRow.appendChild(templateControls);
        host.appendChild(templateRow);
      });
    });
  }

  function updateGeometryObjectSelect() {
    const select = byId("geometryObjectSelect");
    if (!select) return;

    const current = select.value;
    select.innerHTML = "";
    state.objects.forEach((objectItem) => {
      const option = document.createElement("option");
      option.value = objectItem.object_id;
      option.textContent = objectItem.object_id;
      select.appendChild(option);
    });

    if (current && Array.from(select.options).some((opt) => opt.value === current)) {
      select.value = current;
    } else if (select.options.length) {
      select.selectedIndex = 0;
    }
  }

  async function refreshObjects() {
    const ctx = context();
    if (!ctx.assetId) {
      state.objects = [];
      renderObjects();
      updateGeometryObjectSelect();
      return;
    }

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects`,
      {},
      { silent: true }
    );

    if (!res.ok || !res.body) {
      state.objects = [];
      renderObjects();
      updateGeometryObjectSelect();
      log("Failed loading object templates.");
      return;
    }

    state.objects = Array.isArray(res.body.objects) ? res.body.objects : [];
    renderObjects();
    updateGeometryObjectSelect();
    log(`Loaded ${state.objects.length} object definition(s).`);
  }

  async function renameObject(objectId) {
    const ctx = context();
    if (!ctx.assetId) return;
    const next = window.prompt("New object id:", objectId);
    if (!next || !next.trim() || next.trim() === objectId) return;

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects/rename`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ object_id: objectId, new_id: next.trim() }),
      }
    );

    if (!res.ok) {
      log("Object rename failed.");
      return;
    }

    log(`Object renamed: ${objectId} -> ${next.trim()}`);
    await refreshObjects();
  }

  async function deleteObject(objectId) {
    const ctx = context();
    if (!ctx.assetId) return;
    if (!window.confirm(`Delete object "${objectId}" and all templates?`)) return;

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects/${encodeURIComponent(objectId)}`,
      { method: "DELETE" }
    );

    if (!res.ok) {
      log("Object delete failed.");
      return;
    }

    log(`Object deleted: ${objectId}`);
    await refreshObjects();
  }

  async function renameTemplate(objectId, templateName) {
    const ctx = context();
    if (!ctx.assetId) return;
    const next = window.prompt("New template name:", templateName || "");
    if (!next || !next.trim() || next.trim() === templateName) return;

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects/templates/rename`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          object_id: objectId,
          template_name: templateName,
          new_name: next.trim(),
        }),
      }
    );

    if (!res.ok) {
      log("Template rename failed.");
      return;
    }

    log(`Template renamed: ${templateName} -> ${next.trim()}`);
    await refreshObjects();
  }

  async function deleteTemplate(objectId, templateName) {
    const ctx = context();
    if (!ctx.assetId) return;
    if (!window.confirm(`Delete template "${templateName}"?`)) return;

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects/${encodeURIComponent(objectId)}/templates/${encodeURIComponent(templateName)}`,
      { method: "DELETE" }
    );

    if (!res.ok) {
      log("Template delete failed.");
      return;
    }

    log(`Template deleted: ${templateName}`);
    await refreshObjects();
  }

  function showGeometryStatus(message, isError) {
    const host = byId("geometryStatus");
    if (!host) return;
    host.textContent = message;
    host.style.color = isError ? "#115a9b" : "var(--muted)";
  }

  async function saveGeometry() {
    const ctx = context();
    const objectId = (byId("geometryObjectSelect") && byId("geometryObjectSelect").value) || "";
    if (!ctx.assetId || !objectId) {
      showGeometryStatus("Select asset and object first.", true);
      return;
    }

    const L = parseFloat((byId("geometryL") && byId("geometryL").value) || 0);
    const B = parseFloat((byId("geometryB") && byId("geometryB").value) || 0);
    const H = parseFloat((byId("geometryH") && byId("geometryH").value) || 0);
    const axisConvention = (byId("geometryAxisConvention") && byId("geometryAxisConvention").value) || "LBH_XYZ";

    if (L <= 0 || B <= 0 || H <= 0) {
      showGeometryStatus("All dimensions must be > 0", true);
      return;
    }

    const payload = {
      L_mm: L,
      B_mm: B,
      H_mm: H,
      axis_convention: axisConvention,
    };

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects/${encodeURIComponent(objectId)}/geometry`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    if (!res.ok) {
      showGeometryStatus("Failed to save geometry.", true);
      return;
    }

    showGeometryStatus(`Geometry saved: ${L}x${B}x${H} mm`, false);
    log(`Geometry saved for ${objectId}`);
  }

  async function loadGeometry() {
    const ctx = context();
    const objectId = (byId("geometryObjectSelect") && byId("geometryObjectSelect").value) || "";
    if (!ctx.assetId || !objectId) {
      showGeometryStatus("Select asset and object first.", true);
      return;
    }

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(ctx.assetId)}/objects/${encodeURIComponent(objectId)}/geometry`,
      {},
      { silent: true }
    );

    if (!res.ok || !res.body) {
      showGeometryStatus("No geometry found for this object.", true);
      return;
    }

    const geometry = res.body.geometry || res.body;
    byId("geometryL").value = geometry.L_mm || 0;
    byId("geometryB").value = geometry.B_mm || 0;
    byId("geometryH").value = geometry.H_mm || 0;
    byId("geometryAxisConvention").value = geometry.axis_convention || "LBH_XYZ";

    showGeometryStatus(`Loaded: ${geometry.L_mm}x${geometry.B_mm}x${geometry.H_mm} mm`, false);
  }

  function clearGeometry() {
    byId("geometryL").value = "";
    byId("geometryB").value = "";
    byId("geometryH").value = "";
    byId("geometryAxisConvention").value = "LBH_XYZ";
    showGeometryStatus("Geometry form cleared.", false);
  }

  function bindCanvasSelection() {
    const canvas = byId("snapshot");
    if (!canvas) return;
    canvas.addEventListener("click", (event) => {
      if (!state.captureProposals.length) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (event.clientX - rect.left) * scaleX;
      const y = (event.clientY - rect.top) * scaleY;

      let hitIndex = null;
      state.captureProposals.forEach((proposal, idx) => {
        if (proposal.obb_points && proposal.obb_points.length === 4) {
          if (pointInPolygon(x, y, proposal.obb_points)) {
            hitIndex = idx;
          }
          return;
        }

        const bbox = proposal.bbox_xywh || [0, 0, 0, 0];
        const bx = Number(bbox[0] || 0);
        const by = Number(bbox[1] || 0);
        const bw = Number(bbox[2] || 0);
        const bh = Number(bbox[3] || 0);
        if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
          hitIndex = idx;
        }
      });

      if (hitIndex !== null) {
        selectProposal(hitIndex);
      }
    });
  }

  function bindEvents() {
    const map = [
      ["captureSegmentBtn", captureSegment],
      ["clearCaptureBtn", clearCapture],
      ["saveTemplateBtn", saveTemplate],
      ["applyBoxBtn", applyBoxEdit],
      ["refreshTemplatesBtn", refreshObjects],
      ["saveGeometryBtn", saveGeometry],
      ["loadGeometryBtn", loadGeometry],
      ["clearGeometryBtn", clearGeometry],
    ];

    map.forEach(([id, fn]) => {
      const el = byId(id);
      if (el) el.addEventListener("click", fn);
    });

    const refreshContextBtn = byId("refreshContextBtn");
    if (refreshContextBtn && window.operatorShell) {
      refreshContextBtn.addEventListener("click", async () => {
        await window.operatorShell.refreshContext();
        await refreshObjects();
      });
    }

    bindCanvasSelection();
  }

  async function init() {
    bindEvents();

    if (window.operatorShell && typeof window.operatorShell.init === "function") {
      await window.operatorShell.init({
        onContextChanged: async () => {
          await refreshObjects();
        },
      });
    }

    const cameraIds = context().cameraIds || [];
    if (cameraIds.length && byId("captureCameraId") && !byId("captureCameraId").value.trim()) {
      byId("captureCameraId").value = cameraIds[0];
    }

    await refreshObjects();
    if (window.operatorShell) {
      window.operatorShell.setInfo("Object definition manager ready.");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
