(function () {
  'use strict';

  function now() { return performance.now(); }
  function finite(value, fallback) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }

  class TowerPointCloudViewer {
    constructor(canvas, options) {
      const G = window.P3AGiro3D;
      if (!G || !G.THREE || !G.MapControls) throw new Error('本地三维渲染资源不可用');
      this.THREE = G.THREE;
      this.canvas = canvas;
      this.options = options || {};
      this.origin = [0, 0, 0];
      this.bounds = null;
      this.maxDim = 1;
      this.pointSize = finite(this.options.pointSize, 2);
      this.mode = this.options.mode || 'height';
      this.points = null;
      this.source = null;
      this.markerGroup = new this.THREE.Group();
      this.scene = new this.THREE.Scene();
      this.scene.background = new this.THREE.Color(0x071015);
      this.scene.add(this.markerGroup);
      this.camera = new this.THREE.PerspectiveCamera(54, 1, 0.01, 1000000);
      this.camera.up.set(0, 0, 1);
      this.renderer = new this.THREE.WebGLRenderer({canvas: canvas, antialias: true, alpha: false});
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.controls = new G.MapControls(this.camera, canvas);
      this.controls.mouseButtons.LEFT = this.THREE.MOUSE.ROTATE;
      this.controls.mouseButtons.MIDDLE = this.THREE.MOUSE.PAN;
      this.controls.mouseButtons.RIGHT = this.THREE.MOUSE.PAN;
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.12;
      this.controls.screenSpacePanning = true;
      this.controls.addEventListener('change', () => this.requestRender());
      this.raycaster = new this.THREE.Raycaster();
      this.pointerStart = null;
      this.renderPending = false;
      this.firstVisibleResolve = null;
      this.firstVisible = new Promise(resolve => { this.firstVisibleResolve = resolve; });
      this.bindPicking();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(canvas);
      this.resize();
    }

    disposeObject(object) {
      if (!object) return;
      object.traverse(item => {
        if (item.geometry) item.geometry.dispose();
        if (item.material) {
          const materials = Array.isArray(item.material) ? item.material : [item.material];
          materials.forEach(material => {
            if (material.map) material.map.dispose();
            material.dispose();
          });
        }
      });
    }

    resize() {
      const width = Math.max(1, this.canvas.clientWidth);
      const height = Math.max(1, this.canvas.clientHeight);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.requestRender();
    }

    requestRender() {
      if (this.renderPending) return;
      this.renderPending = true;
      requestAnimationFrame(() => {
        this.renderPending = false;
        const changed = this.controls.update();
        this.renderer.render(this.scene, this.camera);
        if (this.firstVisibleResolve && this.points) {
          this.firstVisibleResolve(now());
          this.firstVisibleResolve = null;
        }
        if (changed) this.requestRender();
      });
    }

    setData(payload) {
      const started = now();
      this.firstVisible = new Promise(resolve => { this.firstVisibleResolve = resolve; });
      if (this.points) {
        this.scene.remove(this.points);
        this.disposeObject(this.points);
      }
      const xyz = payload.xyz || [];
      if (!xyz.length) throw new Error('三维点云没有可显示点');
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      xyz.forEach(point => {
        for (let axis = 0; axis < 3; axis += 1) {
          min[axis] = Math.min(min[axis], Number(point[axis]));
          max[axis] = Math.max(max[axis], Number(point[axis]));
        }
      });
      this.origin = min.map((value, axis) => (value + max[axis]) / 2);
      this.bounds = {min: min, max: max};
      this.maxDim = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1);
      const positions = new Float32Array(xyz.length * 3);
      for (let i = 0; i < xyz.length; i += 1) {
        positions[i * 3] = Number(xyz[i][0]) - this.origin[0];
        positions[i * 3 + 1] = Number(xyz[i][1]) - this.origin[1];
        positions[i * 3 + 2] = Number(xyz[i][2]) - this.origin[2];
      }
      const geometry = new this.THREE.BufferGeometry();
      geometry.setAttribute('position', new this.THREE.BufferAttribute(positions, 3));
      geometry.computeBoundingSphere();
      const material = new this.THREE.PointsMaterial({
        size: this.pointSize,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: false,
        depthTest: true,
        depthWrite: true,
      });
      this.points = new this.THREE.Points(geometry, material);
      this.source = payload;
      this.scene.add(this.points);
      this.setMode(this.mode);
      this.camera.near = Math.max(0.01, this.maxDim / 100000);
      this.camera.far = this.maxDim * 100;
      this.camera.updateProjectionMatrix();
      this.resetView();
      return {gpu_viewer_init_ms: Math.round((now() - started) * 1000) / 1000};
    }

    colorArray(mode) {
      const xyz = this.source.xyz;
      const rgb = this.source.rgb || [];
      const intensity = this.source.intensity || [];
      const colors = new Float32Array(xyz.length * 3);
      const zMin = this.bounds.min[2];
      const zSpan = Math.max(0.001, this.bounds.max[2] - zMin);
      let intensityMax = 1;
      for (let i = 0; i < intensity.length; i += 1) intensityMax = Math.max(intensityMax, Number(intensity[i]));
      const color = new this.THREE.Color();
      for (let i = 0; i < xyz.length; i += 1) {
        if (mode === 'height') {
          const t = Math.max(0, Math.min(1, (Number(xyz[i][2]) - zMin) / zSpan));
          color.setHSL((0.68 - 0.68 * t), 0.86, 0.56);
          colors[i * 3] = color.r; colors[i * 3 + 1] = color.g; colors[i * 3 + 2] = color.b;
        } else if (mode === 'intensity') {
          const value = 0.16 + 0.84 * Math.sqrt(Math.max(0, Number(intensity[i]) || 0) / intensityMax);
          colors[i * 3] = value; colors[i * 3 + 1] = value; colors[i * 3 + 2] = value;
        } else {
          const sourceColor = rgb[i] || [165, 185, 205];
          colors[i * 3] = Number(sourceColor[0]) / 255;
          colors[i * 3 + 1] = Number(sourceColor[1]) / 255;
          colors[i * 3 + 2] = Number(sourceColor[2]) / 255;
        }
      }
      return colors;
    }

    setMode(mode) {
      this.mode = ['rgb', 'height', 'intensity'].includes(mode) ? mode : 'height';
      if (!this.points) return;
      const old = this.points.geometry.getAttribute('color');
      if (old) this.points.geometry.deleteAttribute('color');
      this.points.geometry.setAttribute('color', new this.THREE.BufferAttribute(this.colorArray(this.mode), 3));
      this.points.geometry.attributes.color.needsUpdate = true;
      this.requestRender();
    }

    setPointSize(size) {
      this.pointSize = Math.max(1, Math.min(8, finite(size, 2)));
      if (this.points) this.points.material.size = this.pointSize;
      this.requestRender();
    }

    frame(direction, up) {
      if (!this.bounds) return;
      const half = this.maxDim / 2;
      const distance = half / Math.tan(this.camera.fov * Math.PI / 360) * 1.35;
      const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
      const dir = direction.map(value => value / length);
      this.camera.position.set(dir[0] * distance, dir[1] * distance, dir[2] * distance);
      const cameraUp = up || [0, 0, 1];
      this.camera.up.set(cameraUp[0], cameraUp[1], cameraUp[2]);
      this.controls.target.set(0, 0, 0);
      this.camera.lookAt(0, 0, 0);
      this.controls.minDistance = this.maxDim * 0.002;
      this.controls.maxDistance = this.maxDim * 50;
      this.controls.update();
      this.requestRender();
    }

    resetView() { this.frame([-0.72, -0.72, 0.52]); }
    perspectiveView() { this.resetView(); }
    topView() { this.frame([0, 0, 1], [0, 1, 0]); }
    sideView() {
      const xSpan = this.bounds.max[0] - this.bounds.min[0];
      const ySpan = this.bounds.max[1] - this.bounds.min[1];
      this.frame(xSpan >= ySpan ? [0, -1, 0.08] : [-1, 0, 0.08]);
    }

    makeLabel(text) {
      const canvas = document.createElement('canvas');
      canvas.width = 160; canvas.height = 54;
      const context = canvas.getContext('2d');
      context.fillStyle = 'rgba(7,16,21,.88)'; context.fillRect(0, 0, canvas.width, canvas.height);
      context.strokeStyle = '#ffffff'; context.lineWidth = 3; context.strokeRect(2, 2, 156, 50);
      context.fillStyle = '#ffbd4a'; context.font = 'bold 30px Microsoft YaHei';
      context.textAlign = 'center'; context.textBaseline = 'middle'; context.fillText(text, 80, 28);
      const texture = new this.THREE.CanvasTexture(canvas);
      const sprite = new this.THREE.Sprite(new this.THREE.SpriteMaterial({map: texture, depthTest: false}));
      const scale = Math.max(this.maxDim * 0.018, 2);
      sprite.scale.set(scale * 2.7, scale, 1);
      return sprite;
    }

    setMarkers(towers) {
      this.scene.remove(this.markerGroup);
      this.disposeObject(this.markerGroup);
      this.markerGroup = new this.THREE.Group();
      (towers || []).forEach((tower, index) => {
        const local = [Number(tower.x) - this.origin[0], Number(tower.y) - this.origin[1], Number(tower.z) - this.origin[2]];
        const radius = Math.max(this.maxDim * 0.003, 0.45);
        const marker = new this.THREE.Mesh(
          new this.THREE.SphereGeometry(radius, 14, 10),
          new this.THREE.MeshBasicMaterial({color: 0xff9f1c, depthTest: false})
        );
        marker.position.set(local[0], local[1], local[2]);
        this.markerGroup.add(marker);
        const label = this.makeLabel(tower.display_name || `T${String(index + 1).padStart(3, '0')}`);
        label.position.set(local[0], local[1], local[2] + Math.max(Number(tower.height_m) || 0, radius * 4));
        this.markerGroup.add(label);
      });
      this.scene.add(this.markerGroup);
      this.requestRender();
    }

    bindPicking() {
      if (!this.options.onPick) return;
      this.canvas.addEventListener('pointerdown', event => {
        this.pointerStart = {x: event.clientX, y: event.clientY, button: event.button};
      });
      this.canvas.addEventListener('pointercancel', () => { this.pointerStart = null; });
      this.canvas.addEventListener('pointerup', event => {
        const start = this.pointerStart;
        this.pointerStart = null;
        if (!start || start.button !== 0 || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
        const rect = this.canvas.getBoundingClientRect();
        const ndc = new this.THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          -((event.clientY - rect.top) / rect.height) * 2 + 1
        );
        this.raycaster.setFromCamera(ndc, this.camera);
        const distance = this.camera.position.distanceTo(this.controls.target);
        this.raycaster.params.Points.threshold = Math.max(
          this.maxDim * 0.0005,
          distance * Math.tan(this.camera.fov * Math.PI / 360) * (this.pointSize + 5) / Math.max(1, rect.height)
        );
        const hits = this.raycaster.intersectObject(this.points, false);
        if (!hits.length) return;
        const point = hits[0].point;
        this.options.onPick({
          x: point.x + this.origin[0],
          y: point.y + this.origin[1],
          z: point.z + this.origin[2],
          rendered_point_index: hits[0].index,
        });
      });
    }
  }

  window.TreeLineTowerViewer3D = TowerPointCloudViewer;
})();
