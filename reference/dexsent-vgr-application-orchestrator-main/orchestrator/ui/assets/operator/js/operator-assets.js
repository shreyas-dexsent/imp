(function () {
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
      };
    }
    return window.operatorShell.getContext();
  }

  function assetId(item) {
    if (!item || typeof item !== "object") return "";
    if (window.operatorShell && typeof window.operatorShell.assetId === "function") {
      return window.operatorShell.assetId(item);
    }
    return String(item.asset_id || item.process_id || "").trim();
  }

  function setInfo(text) {
    if (window.operatorShell && typeof window.operatorShell.setInfo === "function") {
      window.operatorShell.setInfo(text || "");
      return;
    }
    const host = byId("pageInfo");
    if (host) host.textContent = text || "";
  }

  function currentAsset() {
    const ctx = getContext();
    return (ctx.assets || []).find((item) => assetId(item) === ctx.assetId) || null;
  }

  function setRenameInputFromSelection() {
    const input = byId("renameAssetName");
    if (!input) return;
    const item = currentAsset();
    input.value = (item && item.name) || "";
  }

  function selectAsset(nextAssetId) {
    const select = byId("assetSelect");
    if (!select || !nextAssetId) return;
    if (!Array.from(select.options).some((opt) => opt.value === nextAssetId)) return;
    select.value = nextAssetId;
    select.dispatchEvent(new Event("change"));
    setRenameInputFromSelection();
  }

  function renderAssetList() {
    const host = byId("assetList");
    if (!host) return;

    const ctx = getContext();
    const items = Array.isArray(ctx.assets) ? ctx.assets : [];
    if (!items.length) {
      host.textContent = "No assets loaded.";
      return;
    }

    host.innerHTML = "";
    items.forEach((asset) => {
      const id = assetId(asset);
      const row = document.createElement("div");
      row.className = "object-item";

      const name = document.createElement("span");
      name.textContent = asset.name ? `${asset.name} (${id})` : id;
      row.appendChild(name);

      const controls = document.createElement("div");
      controls.className = "actions";

      const selectBtn = document.createElement("button");
      selectBtn.className = "ghost";
      selectBtn.type = "button";
      selectBtn.textContent = "Select";
      selectBtn.addEventListener("click", () => selectAsset(id));
      controls.appendChild(selectBtn);

      const renameBtn = document.createElement("button");
      renameBtn.className = "ghost";
      renameBtn.type = "button";
      renameBtn.textContent = "Rename";
      renameBtn.addEventListener("click", () => renameAsset(id, asset.name || ""));
      controls.appendChild(renameBtn);

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "ghost";
      deleteBtn.type = "button";
      deleteBtn.textContent = "Delete";
      deleteBtn.addEventListener("click", () => deleteAsset(id, asset.name || id));
      controls.appendChild(deleteBtn);

      row.appendChild(controls);
      host.appendChild(row);
    });
  }

  async function createAsset() {
    const ctx = getContext();
    if (!ctx.stationId) {
      setInfo("Select a station first.");
      return;
    }

    const name = String((byId("newAssetName") && byId("newAssetName").value) || "").trim();
    const inputAssetId = String(
      (byId("newAssetId") && byId("newAssetId").value) || ""
    ).trim();

    const payload = { name: name || "New Asset" };
    if (inputAssetId) payload.asset_id = inputAssetId;

    const res = await window.operatorApi(
      `/stations/${encodeURIComponent(ctx.stationId)}/processes`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );
    if (!res.ok || !res.body) {
      setInfo("Asset create failed.");
      return;
    }

    const createdAssetId = String(res.body.asset_id || res.body.process_id || "").trim();
    if (byId("newAssetName")) byId("newAssetName").value = "";
    if (byId("newAssetId")) byId("newAssetId").value = "";

    if (window.operatorShell) {
      await window.operatorShell.refreshContext();
    }
    renderAssetList();
    if (createdAssetId) {
      selectAsset(createdAssetId);
    }
    setInfo("Asset created.");
  }

  async function renameAsset(assetIdOverride, currentNameOverride) {
    const ctx = getContext();
    const targetAssetId = String(assetIdOverride || ctx.assetId || "").trim();
    if (!targetAssetId) {
      setInfo("Select an asset first.");
      return;
    }

    const inputName = String(
      (byId("renameAssetName") && byId("renameAssetName").value) || ""
    ).trim();
    let nextName = inputName;
    if (!nextName && assetIdOverride) {
      nextName = String(
        window.prompt("New asset name:", currentNameOverride || targetAssetId) || ""
      ).trim();
    }
    if (!nextName) {
      setInfo("Asset name is required.");
      return;
    }

    const res = await window.operatorApi(
      `/processes/${encodeURIComponent(targetAssetId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: nextName }),
      }
    );
    if (!res.ok) {
      setInfo("Asset rename failed.");
      return;
    }

    if (window.operatorShell) {
      await window.operatorShell.refreshContext();
    }
    renderAssetList();
    setRenameInputFromSelection();
    setInfo("Asset renamed.");
  }

  async function deleteAsset(assetIdOverride, labelOverride) {
    const ctx = getContext();
    const targetAssetId = String(assetIdOverride || ctx.assetId || "").trim();
    if (!targetAssetId) {
      setInfo("Select an asset first.");
      return;
    }

    const assets = Array.isArray(ctx.assets) ? ctx.assets : [];
    if (assets.length <= 1) {
      setInfo("At least one asset must remain.");
      return;
    }

    const label = String(labelOverride || targetAssetId);
    if (!window.confirm(`Delete asset "${label}"?`)) {
      return;
    }

    const res = await window.operatorApi(`/processes/${encodeURIComponent(targetAssetId)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      setInfo("Asset delete failed.");
      return;
    }

    if (window.operatorShell) {
      await window.operatorShell.refreshContext();
    }
    renderAssetList();
    setRenameInputFromSelection();
    setInfo("Asset deleted.");
  }

  function bindEvents() {
    const refreshBtn = byId("refreshContextBtn");
    if (refreshBtn && window.operatorShell) {
      refreshBtn.addEventListener("click", async () => {
        await window.operatorShell.refreshContext();
        renderAssetList();
        setRenameInputFromSelection();
        setInfo("Context refreshed.");
      });
    }

    const createBtn = byId("createAssetBtn");
    if (createBtn) createBtn.addEventListener("click", createAsset);

    const renameBtn = byId("renameAssetBtn");
    if (renameBtn) renameBtn.addEventListener("click", () => renameAsset());

    const deleteBtn = byId("deleteAssetBtn");
    if (deleteBtn) deleteBtn.addEventListener("click", () => deleteAsset());

    const assetSelect = byId("assetSelect");
    if (assetSelect) {
      assetSelect.addEventListener("change", () => {
        setRenameInputFromSelection();
      });
    }
  }

  async function init() {
    bindEvents();

    if (window.operatorShell && typeof window.operatorShell.init === "function") {
      await window.operatorShell.init({
        onContextChanged: () => {
          renderAssetList();
          setRenameInputFromSelection();
        },
      });
    }

    renderAssetList();
    setRenameInputFromSelection();
    setInfo("Select asset and open the required manager page.");
  }

  document.addEventListener("DOMContentLoaded", init);
})();
