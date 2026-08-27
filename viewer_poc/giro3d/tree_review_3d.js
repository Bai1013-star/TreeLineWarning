(function () {
  'use strict';

  const FRAME_WINDOW = 24;
  const SLOW_FRAME_MS = 27;
  const MIN_QUALITY = 0.35;

  function finite(value, fallback) {
    return Number.isFinite(Number(value)) ? Number(value) : fallback;
  }

  function xyzOf(layer) {
    return layer && Array.isArray(layer.xyz) ? layer.xyz : [];
  }

  function parseColor(THREE, value, fallback) {
    const color = new THREE.Color(fallback);
    try {
      if (typeof value === 'string' || typeof value === 'number') color.set(value);
    } catch (_) {
      color.set(fallback);
    }
    return color;
  }

  class TreeReviewViewer3D {
    constructor(host, labelOverlay, options) {
      const G = window.P3AGiro3D;
      if (!G || !G.THREE || !G.MapControls) throw new Error('本地 Giro3D / Three.js 渲染资源不可用');
      this.THREE = G.THREE;
      this.host = host;
      this.labelOverlay = labelOverlay;
      this.options = options || {};
      this.source = null;
      this.origin = [0, 0, 0];
      this.bounds = null;
      this.maxDim = 1;
      this.treePoints = null;
      this.treeIds = [];
      this.treeXYZ = [];
      this.treeBaseColors = null;
      this.previewKeys = new Set();
      this.previewTreeId = null;
      this.selection = new Set();
      this.qualityFraction = 1;
      this.frameSamples = [];
      this.renderPending = false;
      this.dampingActive = false;
      this.pointerStart = null;
      this.hoverPending = false;

      host.textContent = '';
      this.canvas = document.createElement('canvas');
      this.canvas.className = 'tree-review-canvas';
      this.canvas.setAttribute('aria-label', 'Giro3D 单木点云复核画布');
      host.appendChild(this.canvas);

      this.scene = new this.THREE.Scene();
      this.scene.background = new this.THREE.Color(0x071015);
      this.content = new this.THREE.Group();
      this.scene.add(this.content);
      this.camera = new this.THREE.PerspectiveCamera(52, 1, 0.01, 1000000);
      this.camera.up.set(0, 0, 1);
      this.renderer = new this.THREE.WebGLRenderer({canvas: this.canvas, antialias: true, alpha: false});
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
      this.controls = new G.MapControls(this.camera, this.canvas);
      this.controls.mouseButtons.LEFT = this.THREE.MOUSE.ROTATE;
      this.controls.mouseButtons.MIDDLE = this.THREE.MOUSE.PAN;
      this.controls.mouseButtons.RIGHT = this.THREE.MOUSE.PAN;
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.11;
      this.controls.screenSpacePanning = true;
      this.controls.addEventListener('start', () => { this.dampingActive = true; });
      this.controls.addEventListener('change', () => {
        this.requestRender(true);
        this.renderLabels();
      });
      this.controls.addEventListener('end', () => { this.dampingActive = true; this.requestRender(true); });
      this.raycaster = new this.THREE.Raycaster();
      this.bindPicking();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(host);
      this.resize();
    }

    disposeObject(object) {
      if (!object) return;
      object.traverse(item => {
        if (item.geometry) item.geometry.dispose();
        const materials = item.material ? (Array.isArray(item.material) ? item.material : [item.material]) : [];
        materials.forEach(material => material.dispose());
      });
    }

    clearContent() {
      this.scene.remove(this.content);
      this.disposeObject(this.content);
      this.content = new this.THREE.Group();
      this.scene.add(this.content);
      this.treePoints = null;
    }

    resize() {
      const width = Math.max(1, this.host.clientWidth);
      const height = Math.max(1, this.host.clientHeight);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.requestRender();
      this.renderLabels();
    }

    requestRender(interacting) {
      if (interacting) this.dampingActive = true;
      if (this.renderPending) return;
      this.renderPending = true;
      requestAnimationFrame(timestamp => {
        const started = performance.now();
        this.renderPending = false;
        const changed = this.controls.enabled ? this.controls.update() : false;
        this.renderer.render(this.scene, this.camera);
        const elapsed = performance.now() - started;
        if (this.dampingActive || changed) this.recordFrame(elapsed);
        if (changed) this.requestRender(true);
        else this.dampingActive = false;
        if (typeof this.options.onRendered === 'function') this.options.onRendered(timestamp, elapsed);
      });
    }

    recordFrame(elapsed) {
      this.frameSamples.push(elapsed);
      if (this.frameSamples.length > FRAME_WINDOW) this.frameSamples.shift();
      if (this.frameSamples.length < FRAME_WINDOW) return;
      const sorted = [...this.frameSamples].sort((a, b) => a - b);
      const p90 = sorted[Math.floor(sorted.length * 0.9)];
      if (p90 > SLOW_FRAME_MS && this.qualityFraction > MIN_QUALITY) {
        this.qualityFraction = Math.max(MIN_QUALITY, this.qualityFraction * 0.72);
        this.frameSamples = [];
        this.rebuildTrees();
        if (typeof this.options.onQualityChange === 'function') {
          this.options.onQualityChange({quality: this.qualityFraction, p90_ms: p90, visible_points: this.visiblePointCount()});
        }
      }
    }

    visiblePointCount() {
      return this.treePoints ? this.treePoints.geometry.getAttribute('position').count : 0;
    }

    computeBounds(data) {
      const layers = [data.background, data.wire, data.wires3d, data.tower, data.trees3d, data.danger_points];
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      let count = 0;
      layers.forEach(layer => xyzOf(layer).forEach(point => {
        if (!Array.isArray(point) || point.length < 3) return;
        for (let axis = 0; axis < 3; axis += 1) {
          const value = Number(point[axis]);
          if (!Number.isFinite(value)) return;
          min[axis] = Math.min(min[axis], value);
          max[axis] = Math.max(max[axis], value);
        }
        count += 1;
      }));
      if (!count) return null;
      this.origin = min.map((value, axis) => (value + max[axis]) / 2);
      this.bounds = {min, max};
      this.maxDim = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1);
      return this.bounds;
    }

    positions(points, rows) {
      const chosen = rows || points.map((_, index) => index);
      const values = new Float32Array(chosen.length * 3);
      chosen.forEach((sourceIndex, index) => {
        const point = points[sourceIndex];
        values[index * 3] = Number(point[0]) - this.origin[0];
        values[index * 3 + 1] = Number(point[1]) - this.origin[1];
        values[index * 3 + 2] = Number(point[2]) - this.origin[2];
      });
      return values;
    }

    addPoints(points, colorSource, options) {
      if (!points.length) return null;
      const geometry = new this.THREE.BufferGeometry();
      geometry.setAttribute('position', new this.THREE.BufferAttribute(this.positions(points), 3));
      const colors = new Float32Array(points.length * 3);
      const color = new this.THREE.Color();
      points.forEach((_, index) => {
        const value = typeof colorSource === 'function' ? colorSource(index) : colorSource;
        if (Array.isArray(value)) color.setRGB(finite(value[0], 160) / 255, finite(value[1], 180) / 255, finite(value[2], 200) / 255);
        else color.copy(parseColor(this.THREE, value, options.fallback));
        colors[index * 3] = color.r; colors[index * 3 + 1] = color.g; colors[index * 3 + 2] = color.b;
      });
      geometry.setAttribute('color', new this.THREE.BufferAttribute(colors, 3));
      geometry.computeBoundingSphere();
      const material = new this.THREE.PointsMaterial({
        size: options.size,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: options.opacity < 1,
        opacity: options.opacity,
        depthTest: true,
        depthWrite: options.opacity >= 1,
      });
      const object = new this.THREE.Points(geometry, material);
      object.userData.role = options.role;
      this.content.add(object);
      return object;
    }

    addWireCurve(points) {
      if (points.length < 2) return;
      const geometry = new this.THREE.BufferGeometry();
      geometry.setAttribute('position', new this.THREE.BufferAttribute(this.positions(points), 3));
      const line = new this.THREE.Line(geometry, new this.THREE.LineBasicMaterial({color: 0xff1744}));
      line.userData.role = 'WIRE_CURVE';
      this.content.add(line);
    }

    setData(data) {
      const started = performance.now();
      this.source = data || {};
      this.clearContent();
      if (!this.computeBounds(this.source)) {
        this.host.dataset.empty = 'true';
        this.requestRender();
        return {gpu_viewer_init_ms: performance.now() - started, visible_tree_points: 0};
      }
      delete this.host.dataset.empty;
      const background = xyzOf(this.source.background);
      const rgb = Array.isArray(this.source.background && this.source.background.rgb) ? this.source.background.rgb : [];
      this.addPoints(background, index => rgb[index] || [145, 165, 180], {role: 'BACKGROUND', size: 1, opacity: 0.22, fallback: 0x78909c});
      this.addPoints(xyzOf(this.source.wire), '#ff1744', {role: 'WIRE', size: 2.2, opacity: 1, fallback: 0xff1744});
      this.addWireCurve(xyzOf(this.source.wires3d));
      this.addPoints(xyzOf(this.source.tower), '#ff9800', {role: 'TOWER', size: 7, opacity: 1, fallback: 0xff9800});
      this.addPoints(xyzOf(this.source.danger_points), '#ff0000', {role: 'DANGER_POINT', size: 5.2, opacity: 1, fallback: 0xff0000});
      const trees = this.source.trees3d || {};
      this.treeXYZ = Array.isArray(trees.xyz) ? trees.xyz : [];
      this.treeIds = Array.isArray(trees.tree_id_by_index) ? trees.tree_id_by_index.map(Number) : [];
      this.treeBaseColors = (Array.isArray(trees.color) ? trees.color : []).map(value => parseColor(this.THREE, value, 0x62d96b));
      this.rebuildTrees();
      this.camera.near = Math.max(0.01, this.maxDim / 100000);
      this.camera.far = this.maxDim * 100;
      this.camera.updateProjectionMatrix();
      this.resetView();
      this.renderLabels();
      return {gpu_viewer_init_ms: performance.now() - started, visible_tree_points: this.visiblePointCount()};
    }

    sampledRows() {
      const count = Math.min(this.treeXYZ.length, this.treeIds.length);
      if (this.qualityFraction >= 0.999 || count < 2) return Array.from({length: count}, (_, index) => index);
      const target = Math.max(1, Math.floor(count * this.qualityFraction));
      const rows = new Array(target);
      for (let index = 0; index < target; index += 1) rows[index] = Math.min(count - 1, Math.floor(index * count / target));
      return rows;
    }

    rebuildTrees() {
      if (this.treePoints) {
        this.content.remove(this.treePoints);
        this.disposeObject(this.treePoints);
      }
      const rows = this.sampledRows();
      if (!rows.length) { this.treePoints = null; this.requestRender(); return; }
      const geometry = new this.THREE.BufferGeometry();
      geometry.setAttribute('position', new this.THREE.BufferAttribute(this.positions(this.treeXYZ, rows), 3));
      geometry.setAttribute('sourceIndex', new this.THREE.Uint32BufferAttribute(rows, 1));
      geometry.setAttribute('color', new this.THREE.BufferAttribute(new Float32Array(rows.length * 3), 3));
      geometry.computeBoundingSphere();
      const material = new this.THREE.PointsMaterial({size: 2.1, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.96, depthTest: true, depthWrite: true});
      this.treePoints = new this.THREE.Points(geometry, material);
      this.treePoints.userData.role = 'TREE';
      this.content.add(this.treePoints);
      this.updateTreeColors();
      this.requestRender();
    }

    pointKey(point) {
      return point.map(value => Number(value).toPrecision(15)).join(',');
    }

    updateTreeColors() {
      if (!this.treePoints) return;
      const colorAttr = this.treePoints.geometry.getAttribute('color');
      const sourceAttr = this.treePoints.geometry.getAttribute('sourceIndex');
      const hasSelection = this.selection.size > 0;
      const red = new this.THREE.Color(0xff0000);
      const selectedColor = new this.THREE.Color(0xffd43b);
      const fallback = new this.THREE.Color(0x62d96b);
      const muted = new this.THREE.Color();
      const previewAllTrees = Number(this.previewTreeId) < 0;
      for (let index = 0; index < colorAttr.count; index += 1) {
        const sourceIndex = sourceAttr.getX(index);
        const treeId = Number(this.treeIds[sourceIndex]);
        const point = this.treeXYZ[sourceIndex];
        let color = this.treeBaseColors[sourceIndex] || fallback;
        if ((previewAllTrees || Number(treeId) === Number(this.previewTreeId)) && this.previewKeys.has(this.pointKey(point))) color = red;
        else if (hasSelection && !this.selection.has(treeId)) color = muted.copy(color).multiplyScalar(0.24);
        else if (this.selection.has(treeId)) color = selectedColor;
        colorAttr.setXYZ(index, color.r, color.g, color.b);
      }
      colorAttr.needsUpdate = true;
      this.requestRender();
    }

    setSelection(treeIds) {
      this.selection = new Set([...treeIds].map(Number));
      this.updateTreeColors();
      this.renderLabels();
    }

    setPreview(points, treeId) {
      this.previewKeys = new Set((Array.isArray(points) ? points : []).map(point => this.pointKey(point)));
      this.previewTreeId = this.previewKeys.size ? Number(treeId) : null;
      this.updateTreeColors();
    }

    setLabelsVisible(visible) {
      this.labelsVisible = Boolean(visible);
      this.renderLabels();
    }

    renderLabels() {
      if (!this.labelOverlay || !this.source) return;
      this.labelOverlay.innerHTML = '';
      const labels = Array.isArray(this.source.trees3d && this.source.trees3d.labels) ? this.source.trees3d.labels : [];
      labels.forEach(label => {
        const treeId = Number(label.tree_id);
        if (!this.labelsVisible && !this.selection.has(treeId)) return;
        const point = label.xyz;
        if (!Array.isArray(point) || point.length < 3) return;
        const projected = new this.THREE.Vector3(Number(point[0]) - this.origin[0], Number(point[1]) - this.origin[1], Number(point[2]) - this.origin[2]).project(this.camera);
        if (![projected.x, projected.y, projected.z].every(Number.isFinite) || projected.z < -1 || projected.z > 1 || Math.abs(projected.x) > 1.2 || Math.abs(projected.y) > 1.2) return;
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'tree-label';
        if (this.selection.has(treeId)) button.classList.add('selected');
        button.textContent = String(treeId);
        button.title = `选择树 ${treeId}`;
        button.setAttribute('aria-label', `选择树 ${treeId}`);
        button.style.left = `${(projected.x + 1) * 50}%`;
        button.style.top = `${(1 - projected.y) * 50}%`;
        button.onclick = event => { event.stopPropagation(); if (typeof this.options.onTreePick === 'function') this.options.onTreePick(treeId, true); };
        this.labelOverlay.appendChild(button);
      });
    }

    frame(direction, up) {
      if (!this.bounds) return;
      const half = this.maxDim / 2;
      const distance = half / Math.tan(this.camera.fov * Math.PI / 360) * 1.45;
      const length = Math.hypot(...direction) || 1;
      const dir = direction.map(value => value / length);
      this.camera.position.set(dir[0] * distance, dir[1] * distance, dir[2] * distance);
      const cameraUp = up || [0, 0, 1];
      this.camera.up.set(...cameraUp);
      this.controls.target.set(0, 0, 0);
      this.camera.lookAt(0, 0, 0);
      this.controls.minDistance = this.maxDim * 0.003;
      this.controls.maxDistance = this.maxDim * 50;
      this.controls.update();
      this.requestRender();
      this.renderLabels();
    }

    resetView() { this.frame([-0.72, -0.72, 0.52]); }
    topView() { this.frame([0, 0, 1], [0, 1, 0]); }
    sideView() { this.frame((this.bounds.max[0] - this.bounds.min[0]) >= (this.bounds.max[1] - this.bounds.min[1]) ? [0, -1, 0.08] : [-1, 0, 0.08]); }

    setControlsEnabled(enabled) {
      this.controls.enabled = Boolean(enabled);
      this.canvas.style.cursor = enabled ? 'grab' : 'crosshair';
      if (enabled) this.requestRender();
    }

    currentViewProjection() {
      this.camera.updateMatrixWorld(true);
      const model = new this.THREE.Matrix4().makeTranslation(-this.origin[0], -this.origin[1], -this.origin[2]);
      const matrix = new this.THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse).multiply(model);
      return Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => matrix.elements[column * 4 + row]));
    }

    canvasRect() { return this.canvas.getBoundingClientRect(); }

    hitAt(clientX, clientY) {
      if (!this.treePoints) return null;
      const rect = this.canvasRect();
      const ndc = new this.THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
      this.raycaster.setFromCamera(ndc, this.camera);
      const distance = this.camera.position.distanceTo(this.controls.target);
      this.raycaster.params.Points.threshold = Math.max(this.maxDim * 0.0008, distance * Math.tan(this.camera.fov * Math.PI / 360) * 8 / Math.max(1, rect.height));
      const hits = this.raycaster.intersectObject(this.treePoints, false);
      if (!hits.length) return null;
      const sourceAttr = this.treePoints.geometry.getAttribute('sourceIndex');
      const sourceIndex = sourceAttr.getX(hits[0].index);
      return {treeId: Number(this.treeIds[sourceIndex]), xyz: this.treeXYZ[sourceIndex], renderedPointIndex: hits[0].index};
    }

    bindPicking() {
      this.canvas.addEventListener('pointerdown', event => { this.pointerStart = {x: event.clientX, y: event.clientY, button: event.button}; });
      this.canvas.addEventListener('pointercancel', () => { this.pointerStart = null; });
      this.canvas.addEventListener('pointerup', event => {
        const start = this.pointerStart;
        this.pointerStart = null;
        if (!this.controls.enabled || !start || start.button !== 0 || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
        const hit = this.hitAt(event.clientX, event.clientY);
        if (!hit) return;
        if (typeof this.options.onPointPick === 'function' && this.options.onPointPick(hit) === true) return;
        if (typeof this.options.onTreePick === 'function') this.options.onTreePick(hit.treeId, true);
      });
    }

    rendererInfo() {
      const gl = this.renderer.getContext();
      const extension = gl.getExtension('WEBGL_debug_renderer_info');
      return {
        three_version: this.THREE.REVISION,
        webgl_version:
          typeof WebGL2RenderingContext !== 'undefined' &&
          gl instanceof WebGL2RenderingContext
            ? 2
            : 1,
        renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : 'masked',
        visible_tree_points: this.visiblePointCount(),
        quality: this.qualityFraction,
      };
    }
  }

  window.TreeLineTreeReview3D = TreeReviewViewer3D;
})();