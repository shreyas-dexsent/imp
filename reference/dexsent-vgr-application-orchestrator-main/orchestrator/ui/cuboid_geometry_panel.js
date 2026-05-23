/**
 * Cuboid Geometry Panel Component
 * Manages UI for setting and viewing cuboid object dimensions (L, B, H)
 */

class CuboidGeometryPanel {
  constructor(containerId, apiBase = '/api') {
    this.container = document.getElementById(containerId);
    this.apiBase = apiBase;
    this.currentObjectId = null;
    this.currentProcessId = null;
    this.init();
  }

  init() {
    this.render();
    this.attachEventListeners();
  }

  render() {
    this.container.innerHTML = `
      <div class="cuboid-geometry-panel">
        <h3>Cuboid Object Geometry</h3>
        
        <div class="form-group">
          <label>Length (L, mm):</label>
          <input type="number" id="input-L" min="10" step="1" placeholder="100" />
        </div>
        
        <div class="form-group">
          <label>Breadth (B, mm):</label>
          <input type="number" id="input-B" min="10" step="1" placeholder="80" />
        </div>
        
        <div class="form-group">
          <label>Height (H, mm):</label>
          <input type="number" id="input-H" min="10" step="1" placeholder="60" />
        </div>
        
        <div class="form-group">
          <label>Axis Convention:</label>
          <div class="axis-convention">
            <label>
              L axis: 
              <select id="select-L-axis">
                <option value="x">X</option>
                <option value="y">Y</option>
                <option value="z">Z</option>
              </select>
            </label>
            <label>
              B axis: 
              <select id="select-B-axis">
                <option value="x">X</option>
                <option value="y" selected>Y</option>
                <option value="z">Z</option>
              </select>
            </label>
            <label>
              H axis: 
              <select id="select-H-axis">
                <option value="x">X</option>
                <option value="y">Y</option>
                <option value="z" selected>Z</option>
              </select>
            </label>
          </div>
        </div>
        
        <div class="button-group">
          <button id="btn-save-geometry" class="btn btn-primary">Save Geometry</button>
          <button id="btn-load-geometry" class="btn btn-secondary">Load Geometry</button>
          <button id="btn-enable-6d" class="btn btn-success">Enable 6D Tracking</button>
        </div>
        
        <div id="status-message" class="status-message"></div>
        
        <div class="tracking-status">
          <h4>6D Pose Tracking Status</h4>
          <div class="status-row">
            <span>State:</span>
            <span id="status-state" class="status-value">INIT</span>
          </div>
          <div class="status-row">
            <span>Confidence:</span>
            <span id="status-confidence" class="status-value">0.00</span>
          </div>
          <div class="status-row">
            <span>Position (m):</span>
            <span id="status-position" class="status-value">-</span>
          </div>
          <div class="status-row">
            <span>Inliers:</span>
            <span id="status-inliers" class="status-value">0</span>
          </div>
          <div class="status-row">
            <span>Residual (mm):</span>
            <span id="status-residual" class="status-value">-</span>
          </div>
          <div class="form-group">
            <label>
              <input type="checkbox" id="toggle-overlay" />
              Show Overlay
            </label>
          </div>
        </div>
      </div>
    `;
  }

  attachEventListeners() {
    document.getElementById('btn-save-geometry')
      .addEventListener('click', () => this.saveGeometry());
    document.getElementById('btn-load-geometry')
      .addEventListener('click', () => this.loadGeometry());
    document.getElementById('btn-enable-6d')
      .addEventListener('click', () => this.enable6DTracking());
    document.getElementById('toggle-overlay')
      .addEventListener('change', (e) => this.toggleOverlay(e.target.checked));
  }

  setObject(processId, objectId) {
    this.currentProcessId = processId;
    this.currentObjectId = objectId;
    this.loadGeometry();
  }

  async saveGeometry() {
    const L = parseFloat(document.getElementById('input-L').value);
    const B = parseFloat(document.getElementById('input-B').value);
    const H = parseFloat(document.getElementById('input-H').value);
    const L_axis = document.getElementById('select-L-axis').value;
    const B_axis = document.getElementById('select-B-axis').value;
    const H_axis = document.getElementById('select-H-axis').value;

    if (!this.currentProcessId || !this.currentObjectId) {
      this.showStatus('No object selected', 'error');
      return;
    }

    if (isNaN(L) || isNaN(B) || isNaN(H)) {
      this.showStatus('Please enter valid dimensions', 'error');
      return;
    }

    try {
      const url = `${this.apiBase}/processes/${this.currentProcessId}/objects/${this.currentObjectId}/geometry`;
      const response = await fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          L_mm: L,
          B_mm: B,
          H_mm: H,
          axis_convention: {
            L_axis: L_axis,
            B_axis: B_axis,
            H_axis: H_axis
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      this.showStatus('Geometry saved successfully', 'success');
    } catch (error) {
      this.showStatus(`Save failed: ${error.message}`, 'error');
    }
  }

  async loadGeometry() {
    if (!this.currentProcessId || !this.currentObjectId) {
      return;
    }

    try {
      const url = `${this.apiBase}/processes/${this.currentProcessId}/objects/${this.currentObjectId}/geometry`;
      const response = await fetch(url);

      if (!response.ok) {
        if (response.status === 404) {
          // No geometry set yet, show defaults
          this.setDefaults();
          return;
        }
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();
      const geom = result.geometry;

      document.getElementById('input-L').value = geom.L_mm || 100;
      document.getElementById('input-B').value = geom.B_mm || 80;
      document.getElementById('input-H').value = geom.H_mm || 60;

      const convention = geom.axis_convention || {};
      document.getElementById('select-L-axis').value = convention.L_axis || 'x';
      document.getElementById('select-B-axis').value = convention.B_axis || 'y';
      document.getElementById('select-H-axis').value = convention.H_axis || 'z';
    } catch (error) {
      console.warn('Failed to load geometry:', error);
      this.setDefaults();
    }
  }

  setDefaults() {
    document.getElementById('input-L').value = 100;
    document.getElementById('input-B').value = 80;
    document.getElementById('input-H').value = 60;
    document.getElementById('select-L-axis').value = 'x';
    document.getElementById('select-B-axis').value = 'y';
    document.getElementById('select-H-axis').value = 'z';
  }

  enable6DTracking() {
    // This would typically send a control command to the vision engine
    this.showStatus('6D cuboid tracking enabled', 'info');
  }

  toggleOverlay(enabled) {
    if (enabled) {
      this.showStatus('Overlay enabled - cuboid edges will be drawn', 'info');
    } else {
      this.showStatus('Overlay disabled', 'info');
    }
    // Emit event for parent component to handle
    window.dispatchEvent(new CustomEvent('cuboid-overlay-toggle', { detail: { enabled } }));
  }

  updateTrackingStatus(poseResult) {
    if (!poseResult) {
      return;
    }

    document.getElementById('status-state').textContent = poseResult.state || 'UNKNOWN';
    document.getElementById('status-confidence').textContent = 
      (poseResult.confidence || 0).toFixed(2);

    if (poseResult.pose_cam) {
      const t = poseResult.pose_cam.t_m;
      document.getElementById('status-position').textContent = 
        `[${t[0].toFixed(3)}, ${t[1].toFixed(3)}, ${t[2].toFixed(3)}]`;
    }

    const quality = poseResult.quality || {};
    document.getElementById('status-inliers').textContent = 
      quality.depth_inliers || 0;
    document.getElementById('status-residual').textContent = 
      ((quality.depth_residual_m || 0) * 1000).toFixed(1);
  }

  showStatus(message, type = 'info') {
    const elem = document.getElementById('status-message');
    elem.textContent = message;
    elem.className = `status-message status-${type}`;
    
    if (type !== 'error') {
      setTimeout(() => {
        elem.textContent = '';
        elem.className = 'status-message';
      }, 3000);
    }
  }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CuboidGeometryPanel;
}
