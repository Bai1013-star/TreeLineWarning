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

  function viewerRuntime() {
    return window.TreeLineViewerRuntime || null;
  }

  function errorRecord(error) {
    return {
      name: error && error.name ? String(error.name) : 'Error',
      message: error && error.message ? String(error.message) : String(error || ''),
      stack: error && error.stack ? String(error.stack) : '',
    };
  }

  class TreeReviewViewer3D {
    constructor(host, labelOverlay, options) {
      const G = window.P3AGiro3D;
      const runtime = viewerRuntime();
      if (!G || !G.THREE || !G.MapControls) {
        const missing = new Error('本地 Giro3D / Three.js 渲染资源不可用');
        if (runtime && typeof runtime.notifyFailure === 'function') {
          throw runtime.notifyFailure(missing, {
            phase: 'resource',
            // A missing Three/Giro3D global means the packaged viewer script
            // did not initialize, even when the browser has no script error
            // event (for example a CSP or global-evaluation failure).
            resourceMissing: true,
          });
        }
        throw missing;
      }
      this.THREE = G.THREE;
      this.host = host;
      this.labelOverlay = labelOverlay;
      this.options = options || {};
      this.runtime = runtime;
      this.source = null;
      this.origin = [0, 0, 0];
      this.bounds = null;
      this.maxDim = 1;
      this.treePoints = null;
      this.treeIds = [];
      this.treeSourceIds = [];
      this.treeOriginalIndices = [];
      this.treeVisible = [];
      this.treeXYZ = [];
      this.treeBaseColors = null;
      this.treeInitialColors = null;
      this.displayBaseId = null;
      this.editorRevision = 0;
      this.activeTreeBufferRows = 0;
      this.previewKeys = new Set();
      this.previewTreeId = null;
      this.selection = new Set();
      this.qualityFraction = 1;
      this.frameSamples = [];
      this.renderPending = false;
      this.dampingActive = false;
      this.pointerStart = null;
      this.hoverPending = false;
      this.renderer = null;
      this.rendererContext = null;
      this.canvas = null;
      this.controls = null;
      this.scene = null;
      this.content = null;
      this.camera = null;
      this.raycaster = null;
      this.resizeObserver = null;
      this.rafId = null;
      this.disposed = false;
      this.runtimeFailure = null;
      this.rendererAttempts = [];
      this._contextLostHandler = null;
      this._resizeHandler = null;
      this._onPointerDown = null;
      this._onPointerCancel = null;
      this._onPointerUp = null;
      this._onControlStart = null;
      this._onControlChange = null;
      this._onControlEnd = null;

      try {
        host.textContent = '';
        this.scene = new this.THREE.Scene();
        this.scene.background = new this.THREE.Color(0x071015);
        this.content = new this.THREE.Group();
        this.scene.add(this.content);
        this.camera = new this.THREE.PerspectiveCamera(52, 1, 0.01, 1000000);
        this.camera.up.set(0, 0, 1);
        this.initializeRenderer(G);
        this.controls = new G.MapControls(this.camera, this.canvas);
        this.controls.mouseButtons.LEFT = this.THREE.MOUSE.ROTATE;
        this.controls.mouseButtons.MIDDLE = this.THREE.MOUSE.PAN;
        this.controls.mouseButtons.RIGHT = this.THREE.MOUSE.PAN;
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.11;
        this.controls.screenSpacePanning = true;
        this._onControlStart = () => { this.dampingActive = true; };
        this._onControlChange = () => {
          this.requestRender(true);
          this.renderLabels();
        };
        this._onControlEnd = () => { this.dampingActive = true; this.requestRender(true); };
        this.controls.addEventListener('start', this._onControlStart);
        this.controls.addEventListener('change', this._onControlChange);
        this.controls.addEventListener('end', this._onControlEnd);
        this.raycaster = new this.THREE.Raycaster();
        this.bindPicking();
        this._contextLostHandler = event => {
          if (event && typeof event.preventDefault === 'function') event.preventDefault();
          this.handleRuntimeFailure(new Error(
            `WebGL context lost${event && event.statusMessage ? `: ${event.statusMessage}` : ''}`
          ), 'context-lost');
        };
        if (this.canvas && typeof this.canvas.addEventListener === 'function') {
          this.canvas.addEventListener('webglcontextlost', this._contextLostHandler, false);
        }
        this._resizeHandler = () => {
          try { this.resize(); }
          catch (error) { this.handleRuntimeFailure(error, 'resize'); }
        };
        if (typeof ResizeObserver === 'function') {
          this.resizeObserver = new ResizeObserver(this._resizeHandler);
          this.resizeObserver.observe(host);
        }
        this.resize();
      } catch (error) {
        this.dispose();
        const context = error && error.viewerFailureContext
          ? error.viewerFailureContext
          : {phase: error && error.viewerPhase || 'renderer'};
        context.capabilities = context.capabilities || this.capabilities;
        context.attempts = this.rendererAttempts;
        const wrapped = runtime && typeof runtime.notifyFailure === 'function'
          ? runtime.notifyFailure(error, context) : error;
        if (wrapped && wrapped !== error && wrapped.viewerDiagnostic) this.runtimeFailure = wrapped;
        throw wrapped;
      }
    }

    createCanvas() {
      const canvas = document.createElement('canvas');
      canvas.className = 'tree-review-canvas';
      canvas.setAttribute('aria-label', 'Giro3D 单木点云复核画布');
      this.host.appendChild(canvas);
      return canvas;
    }

    initializeRenderer(G) {
      const runtime = this.runtime;
      const environment = runtime && typeof runtime.getCapabilities === 'function'
        ? runtime.getCapabilities() : null;
      const safeFirst = Boolean(runtime && runtime.isLowCapability && runtime.isLowCapability(environment));
      // Weak devices stay on the compatibility configuration for both
      // bounded attempts.  A successful normal attempt is kept at the normal
      // pixel ratio; only an actual normal failure gets a fresh safe canvas.
      const modes = safeFirst ? ['safe', 'safe'] : ['normal', 'safe'];
      let lastError = null;
      for (let index = 0; index < modes.length; index += 1) {
        const mode = modes[index];
        const canvas = this.createCanvas();
        const options = runtime && typeof runtime.rendererOptions === 'function'
          ? runtime.rendererOptions(mode, environment)
          : {
            antialias: mode === 'normal',
            alpha: false,
            depth: true,
            pixelRatio: Math.min(window.devicePixelRatio || 1, mode === 'normal' ? 1.75 : 1),
            mode,
          };
        const record = runtime && typeof runtime.probeCanvas === 'function'
          ? runtime.probeCanvas(canvas, {
            alpha: false,
            antialias: Boolean(options.antialias),
            depth: true,
            stencil: false,
          }) : null;
        const capabilities = record && record.capabilities ? record.capabilities : record;
        const webgl2 = capabilities && capabilities.webgl2;
        const attempt = {mode, options: {...options}, contextCreationError: capabilities && capabilities.contextCreationError || null};
        if (capabilities && capabilities.contextAttributes) {
          attempt.actualContextAttributes = {...capabilities.contextAttributes};
        }
        if (webgl2 && !webgl2.contextAvailable) {
          if (record && record.contextError) attempt.error = errorRecord(record.contextError);
          attempt.code = runtime && runtime.classify
            ? runtime.classify(null, {capabilities, phase: 'context'})
            : 'WEBGL_CONTEXT_CREATION_FAILED';
          this.rendererAttempts.push(attempt);
          lastError = record && record.contextError
            ? record.contextError : new Error('WebGL2 context creation failed');
          lastError.viewerFailureContext = {phase: 'context', capabilities};
          this.releaseRendererAttempt(canvas, null);
          continue;
        }
        if (webgl2 && webgl2.contextAvailable && capabilities.contextAttributes
          && mode === 'normal' && capabilities.contextAttributes.antialias === false) {
          // Context attributes are immutable after creation.  A browser that
          // ignores an antialias=true request must go through the explicit
          // safe path with a fresh context rather than claiming normal mode.
          attempt.code = 'CONTEXT_ATTRIBUTES_MISMATCH';
          this.rendererAttempts.push(attempt);
          lastError = new Error('WebGL2 context antialias attribute mismatch');
          lastError.viewerFailureContext = {phase: 'context', capabilities};
          this.releaseRendererAttempt(canvas, null);
          continue;
        }
        try {
          const rendererOptions = {
            canvas,
            antialias: Boolean(options.antialias),
            alpha: false,
            depth: true,
          };
          // Three.js r180 accepts an existing WebGL2 context.  Passing the
          // retained probe context avoids creating a second context on clean
          // machines while keeping context capability detection separate from
          // renderer success.
          if (record && record.context) rendererOptions.context = record.context;
          this.renderer = new this.THREE.WebGLRenderer(rendererOptions);
          this.renderer.setPixelRatio(options.pixelRatio);
          this.canvas = canvas;
          this.capabilities = runtime && typeof runtime.snapshot === 'function'
            ? runtime.snapshot(record) : capabilities;
          if (this.capabilities && this.capabilities.contextAttributes) {
            attempt.actualContextAttributes = {...this.capabilities.contextAttributes};
          }
          this.rendererContext = record && record.context
            ? record.context
            : this.safeRendererContext();
          this.setRendererSizeForInitialization();
          if (!this.renderer || typeof this.renderer.render !== 'function') {
            throw new Error('THREE.WebGLRenderer.render is unavailable');
          }
          // A successful context probe is not enough: exercise the actual
          // Three.js constructor and one initial render before proceeding.
          this.renderer.render(this.scene, this.camera);
          attempt.code = 'OK';
          this.rendererMode = mode;
          this.rendererAttempts.push(attempt);
          if (runtime && typeof runtime.reportSuccess === 'function') {
            runtime.reportSuccess({
              phase: 'initialization',
              rendererMode: mode,
              capabilities: this.capabilities,
              attempts: this.rendererAttempts,
            });
          }
          return;
        } catch (error) {
          attempt.error = errorRecord(error);
          attempt.code = runtime && runtime.classify
            ? runtime.classify(error, {capabilities, phase: 'renderer'})
            : 'THREE_RENDERER_FAILED';
          this.rendererAttempts.push(attempt);
          lastError = error;
          this.releaseRendererAttempt(canvas, this.renderer);
          this.renderer = null;
          this.canvas = null;
          this.rendererContext = null;
        }
      }
      if (!lastError) lastError = new Error('3D renderer initialization failed');
      if (!lastError.viewerFailureContext) {
        lastError.viewerFailureContext = {
          phase: 'renderer', capabilities: this.capabilities || environment,
        };
      }
      throw lastError;
    }

    safeRendererContext() {
      if (!this.renderer || typeof this.renderer.getContext !== 'function') return null;
      try { return this.renderer.getContext(); } catch (_) { return null; }
    }

    setRendererSizeForInitialization() {
      const width = Math.max(1, Number(this.host.clientWidth) || 1);
      const height = Math.max(1, Number(this.host.clientHeight) || 1);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }

    releaseRendererAttempt(canvas, renderer) {
      if (this.runtime && typeof this.runtime.releaseCanvas === 'function') {
        this.runtime.releaseCanvas(canvas, renderer);
        return;
      }
      try { if (renderer && typeof renderer.dispose === 'function') renderer.dispose(); } catch (_) {}
      if (canvas && canvas.parentNode && typeof canvas.parentNode.removeChild === 'function') {
        try { canvas.parentNode.removeChild(canvas); } catch (_) {}
      }
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
      this.editSeedMarkers = null;
      if (!this.scene || !this.content) return;
      this.scene.remove(this.content);
      this.disposeObject(this.content);
      this.content = new this.THREE.Group();
      this.scene.add(this.content);
      this.treePoints = null;
    }

    resize() {
      if (this.disposed || !this.renderer || !this.camera) return;
      const width = Math.max(1, this.host.clientWidth);
      const height = Math.max(1, this.host.clientHeight);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.requestRender();
      this.renderLabels();
    }

    requestRender(interacting) {
      if (this.disposed || this.runtimeFailure || !this.renderer || !this.controls) return;
      if (interacting) this.dampingActive = true;
      if (this.renderPending) return;
      this.renderPending = true;
      const schedule = typeof requestAnimationFrame === 'function'
        ? requestAnimationFrame
        : callback => setTimeout(() => callback(performance.now()), 0);
      this.rafId = schedule(timestamp => {
        const started = performance.now();
        this.rafId = null;
        this.renderPending = false;
        try {
          if (this.disposed || this.runtimeFailure || !this.renderer) return;
          const changed = this.controls.enabled ? this.controls.update() : false;
          this.renderer.render(this.scene, this.camera);
          const elapsed = performance.now() - started;
          if (this.dampingActive || changed) this.recordFrame(elapsed);
          if (changed) this.requestRender(true);
          else this.dampingActive = false;
          if (typeof this.options.onRendered === 'function') this.options.onRendered(timestamp, elapsed);
        } catch (error) {
          this.handleRuntimeFailure(error, 'render');
        }
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
      return this.treePoints ? this.activeTreeBufferRows : 0;
    }

    computeBounds(data) {
      const layers = [data.background, data.wire, data.wires3d, data.tower, data.infrastructure, data.trees3d, data.danger_points];
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

    addWireCurve(points, role = 'WIRE_CURVE', color = 0xff1744) {
      if (points.length < 2) return;
      const geometry = new this.THREE.BufferGeometry();
      geometry.setAttribute('position', new this.THREE.BufferAttribute(this.positions(points), 3));
      const line = new this.THREE.Line(geometry, new this.THREE.LineBasicMaterial({color}));
      line.userData.role = role;
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
      const fittedSegments = this.source.wires3d && this.source.wires3d.segments;
      if (Array.isArray(fittedSegments)) fittedSegments.forEach(segment => this.addWireCurve(xyzOf(segment)));
      else this.addWireCurve(xyzOf(this.source.wires3d)); // Compatibility for raw DBH candidate axes.
      this.addPoints(xyzOf(this.source.tower), '#ff9800', {role: 'TOWER', size: 7, opacity: 1, fallback: 0xff9800});
      this.addPoints(xyzOf(this.source.infrastructure), '#ffb74d', {role: 'PROTECTED_INFRASTRUCTURE', size: 1.8, opacity: 1, fallback: 0xffb74d});
      (this.source.infrastructure?.pole_segments || []).forEach(pole => this.addWireCurve(xyzOf(pole), 'PROTECTED_POLE_FIT', 0xff9800));
      this.addPoints(xyzOf(this.source.danger_points), '#ff0000', {role: 'DANGER_POINT', size: 5.2, opacity: 1, fallback: 0xff0000});
      const trees = this.source.trees3d || {};
      this.treeXYZ = Array.isArray(trees.xyz) ? trees.xyz : [];
      this.treeIds = Array.isArray(trees.tree_id_by_index) ? trees.tree_id_by_index.map(Number) : [];
      this.treeSourceIds = Array.isArray(trees.source_tree_id_by_index)
        ? trees.source_tree_id_by_index.map(Number) : [...this.treeIds];
      this.treeOriginalIndices = Array.isArray(trees.original_index_by_index)
        ? trees.original_index_by_index.map(Number) : this.treeXYZ.map((_, index) => index);
      this.treeVisible = Array.isArray(trees.visible_by_index)
        ? trees.visible_by_index.map(Boolean) : this.treeIds.map(treeId => Number(treeId) >= 0);
      this.treeBaseColors = (Array.isArray(trees.color) ? trees.color : []).map(value => parseColor(this.THREE, value, 0x62d96b));
      this.treeInitialColors = this.treeBaseColors.map(color => color.clone());
      this.displayBaseId = data && data.display_base_id ? String(data.display_base_id) : null;
      this.editorRevision = Number(data && data.editor_revision || 0);
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
      const visible = [];
      for (let index = 0; index < count; index += 1) {
        if (this.treeVisible[index] !== false && Number(this.treeIds[index]) >= 0) visible.push(index);
      }
      if (this.qualityFraction >= 0.999 || visible.length < 2) return visible;
      const target = Math.max(1, Math.floor(visible.length * this.qualityFraction));
      const rows = new Array(target);
      for (let index = 0; index < target; index += 1) {
        rows[index] = visible[Math.min(visible.length - 1, Math.floor(index * visible.length / target))];
      }
      return rows;
    }

    rebuildTrees() {
      const rows = this.sampledRows();
      const capacity = Math.max(1, this.treeXYZ.length);
      let geometry = this.treePoints && this.treePoints.geometry;
      const existing = geometry && geometry.getAttribute('sourceIndex');
      if (!geometry || !existing || existing.count < capacity) {
        if (this.treePoints) {
          this.content.remove(this.treePoints);
          this.disposeObject(this.treePoints);
        }
        geometry = new this.THREE.BufferGeometry();
        geometry.setAttribute('position', new this.THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
        geometry.setAttribute('sourceIndex', new this.THREE.Uint32BufferAttribute(new Uint32Array(capacity), 1));
        geometry.setAttribute('color', new this.THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
        const material = new this.THREE.PointsMaterial({size: 2.1, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.96, depthTest: true, depthWrite: true});
        this.treePoints = new this.THREE.Points(geometry, material);
        this.treePoints.userData.role = 'TREE';
        this.content.add(this.treePoints);
      }
      const positions = this.positions(this.treeXYZ, rows);
      const positionAttr = geometry.getAttribute('position');
      const sourceAttr = geometry.getAttribute('sourceIndex');
      positionAttr.array.set(positions, 0);
      sourceAttr.array.set(rows, 0);
      positionAttr.needsUpdate = true;
      sourceAttr.needsUpdate = true;
      geometry.setDrawRange(0, rows.length);
      this.activeTreeBufferRows = rows.length;
      if (rows.length) geometry.computeBoundingSphere();
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
      for (let index = 0; index < this.activeTreeBufferRows; index += 1) {
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

    applyDelta(delta) {
      const started = performance.now();
      if (!delta || delta.available !== true || String(delta.base_id || '') !== String(this.displayBaseId || '')) {
        throw new Error('EDITOR_DISPLAY_DELTA_BASE_MISMATCH');
      }
      if (Number(delta.from_revision) !== Number(this.editorRevision)) {
        throw new Error('EDITOR_DISPLAY_DELTA_REVISION_MISMATCH');
      }
      const mapping = delta.tree_id_by_source || {};
      const removed = new Set((delta.removed_original_index || []).map(Number));
      const clearanceCurrent = delta.clearance_current === true;
      const exactIds = delta.tree_id_by_index;
      const exactVisible = delta.visible_by_index;
      const exact = Array.isArray(exactIds) && Array.isArray(exactVisible);
      if (exact && (exactIds.length !== this.treeXYZ.length || exactVisible.length !== this.treeXYZ.length)) {
        throw new Error('EDITOR_DISPLAY_DELTA_SHAPE_MISMATCH');
      }
      for (let index = 0; index < this.treeXYZ.length; index += 1) {
        const sourceId = Number(this.treeSourceIds[index]);
        const mapped = exact ? Number(exactIds[index])
          : Object.prototype.hasOwnProperty.call(mapping, String(sourceId))
            ? Number(mapping[String(sourceId)]) : -1;
        this.treeIds[index] = mapped;
        this.treeVisible[index] = mapped >= 0 && (exact ? Boolean(exactVisible[index])
          : !removed.has(Number(this.treeOriginalIndices[index])));
        this.treeBaseColors[index] = clearanceCurrent && this.treeInitialColors[index]
          ? this.treeInitialColors[index].clone()
          : parseColor(this.THREE, `hsl(${(Math.max(0, mapped) * 137) % 360},72%,58%)`, 0x62d96b);
      }
      this.editorRevision = Number(delta.revision);
      if (this.source) {
        this.source.editor_revision = this.editorRevision;
        this.source.trees = Array.isArray(delta.trees) ? delta.trees : [];
        this.source.operations = Array.isArray(delta.operations) ? delta.operations : [];
        if (this.source.trees3d) {
          this.source.trees3d.tree_id_by_index = this.treeIds;
          this.source.trees3d.visible_by_index = this.treeVisible;
          this.source.trees3d.labels = Array.isArray(delta.labels) ? delta.labels : [];
        }
      }
      if (this.content) {
        this.content.children.forEach(object => {
          if (object.userData && object.userData.role === 'DANGER_POINT') {
            object.visible = clearanceCurrent;
          }
        });
      }
      const currentIds = new Set((delta.trees || []).map(tree => Number(tree.tree_id)));
      this.selection = new Set([...this.selection].filter(treeId => currentIds.has(Number(treeId))));
      this.previewKeys.clear();
      this.previewTreeId = null;
      this.rebuildTrees();
      this.renderLabels();
      return {
        delta_apply_ms: performance.now() - started,
        visible_tree_points: this.visiblePointCount(),
        buffer_reused: true,
      };
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

    setEditSeeds(points) {
      if (this.editSeedMarkers) {
        this.content.remove(this.editSeedMarkers);
        this.disposeObject(this.editSeedMarkers);
        this.editSeedMarkers = null;
      }
      if (points.length) {
        const geometry = new this.THREE.BufferGeometry();
        geometry.setAttribute('position', new this.THREE.Float32BufferAttribute(points.flatMap(p => p.map((v, i) => Number(v) - this.origin[i])), 3));
        const material = new this.THREE.PointsMaterial({color: 0x25eaff, size: 14, sizeAttenuation: false, depthTest: false});
        this.editSeedMarkers = new this.THREE.Points(geometry, material);
        this.editSeedMarkers.renderOrder = 20;
        this.content.add(this.editSeedMarkers);
      }
      this.requestRender();
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
        button.onclick = event => { event.stopPropagation(); if (typeof this.options.onTreePick === 'function') this.options.onTreePick(treeId, event); };
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
      this._onPointerDown = event => { this.pointerStart = {x: event.clientX, y: event.clientY, button: event.button}; };
      this._onPointerCancel = () => { this.pointerStart = null; };
      this._onPointerUp = event => {
        const start = this.pointerStart;
        this.pointerStart = null;
        if (!this.controls.enabled || !start || start.button !== 0 || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
        const hit = this.hitAt(event.clientX, event.clientY);
        if (!hit) return;
        if (typeof this.options.onPointPick === 'function' && this.options.onPointPick(hit, event) === true) return;
        if (typeof this.options.onTreePick === 'function') this.options.onTreePick(hit.treeId, event);
      };
      this.canvas.addEventListener('pointerdown', this._onPointerDown);
      this.canvas.addEventListener('pointercancel', this._onPointerCancel);
      this.canvas.addEventListener('pointerup', this._onPointerUp);
    }

    handleRuntimeFailure(error, phase) {
      if (this.runtimeFailure || this.disposed) return this.runtimeFailure;
      const context = {
        phase: phase || 'renderer',
        capabilities: this.capabilities,
        attempts: this.rendererAttempts,
      };
      this.runtimeFailure = this.runtime && typeof this.runtime.notifyFailure === 'function'
        ? this.runtime.notifyFailure(error, context) : error;
      try {
        if (typeof this.options.onInitFailure === 'function') this.options.onInitFailure(this.runtimeFailure);
      } catch (_) {}
      this.dispose();
      return this.runtimeFailure;
    }

    disposeObjectTree(object) {
      if (!object) return;
      try { this.disposeObject(object); } catch (_) {}
    }

    dispose() {
      if (this.disposed) return;
      this.disposed = true;
      this.renderPending = false;
      if (this.rafId != null) {
        try {
          if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(this.rafId);
          else clearTimeout(this.rafId);
        } catch (_) {}
        this.rafId = null;
      }
      if (this.resizeObserver && typeof this.resizeObserver.disconnect === 'function') {
        try { this.resizeObserver.disconnect(); } catch (_) {}
      }
      this.resizeObserver = null;
      if (this.controls) {
        try {
          if (this._onControlStart && typeof this.controls.removeEventListener === 'function') this.controls.removeEventListener('start', this._onControlStart);
          if (this._onControlChange && typeof this.controls.removeEventListener === 'function') this.controls.removeEventListener('change', this._onControlChange);
          if (this._onControlEnd && typeof this.controls.removeEventListener === 'function') this.controls.removeEventListener('end', this._onControlEnd);
        } catch (_) {}
        try { if (typeof this.controls.dispose === 'function') this.controls.dispose(); } catch (_) {}
      }
      this.controls = null;
      if (this.canvas && typeof this.canvas.removeEventListener === 'function') {
        try {
          if (this._onPointerDown) this.canvas.removeEventListener('pointerdown', this._onPointerDown);
          if (this._onPointerCancel) this.canvas.removeEventListener('pointercancel', this._onPointerCancel);
          if (this._onPointerUp) this.canvas.removeEventListener('pointerup', this._onPointerUp);
          if (this._contextLostHandler) this.canvas.removeEventListener('webglcontextlost', this._contextLostHandler);
        } catch (_) {}
      }
      this.disposeObjectTree(this.content);
      this.content = null;
      if (this.renderer) {
        try { if (typeof this.renderer.dispose === 'function') this.renderer.dispose(); } catch (_) {}
        try { if (typeof this.renderer.forceContextLoss === 'function') this.renderer.forceContextLoss(); } catch (_) {}
      }
      const canvas = this.canvas;
      const renderer = this.renderer;
      if (this.runtime && typeof this.runtime.releaseCanvas === 'function') {
        try { this.runtime.releaseCanvas(canvas, null); } catch (_) {}
      }
      this.renderer = null;
      this.rendererContext = null;
      this.canvas = null;
      if (canvas && canvas.parentNode && typeof canvas.parentNode.removeChild === 'function') {
        try { canvas.parentNode.removeChild(canvas); } catch (_) {}
      }
      this.scene = null;
      this.camera = null;
      this.raycaster = null;
    }

    rendererInfo() {
      const caps = this.capabilities || {};
      return {
        three_version: this.THREE.REVISION,
        webgl_version: caps.webgl2 && caps.webgl2.contextAvailable ? 2 : 1,
        renderer: caps.renderer || 'masked',
        vendor: caps.vendor || 'masked',
        max_texture_size: caps.maxTextureSize == null ? null : caps.maxTextureSize,
        device_pixel_ratio: caps.devicePixelRatio == null ? null : caps.devicePixelRatio,
        capabilities: caps,
        renderer_mode: this.rendererMode || 'unknown',
        visible_tree_points: this.visiblePointCount(),
        quality: this.qualityFraction,
      };
    }
  }

  window.TreeLineTreeReview3D = TreeReviewViewer3D;
})();
