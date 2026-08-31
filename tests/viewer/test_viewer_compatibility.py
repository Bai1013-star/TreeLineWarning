"""Focused clean-PC compatibility checks for the packaged Tree Review Viewer."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_JS = ROOT / "viewer_poc" / "giro3d" / "viewer_runtime.js"
VIEWER_JS = ROOT / "viewer_poc" / "giro3d" / "tree_review_3d.js"
APP_JS = ROOT / "viewer_poc" / "giro3d" / "tree_review_app.js"


_NODE_CHECK = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const runtimeSource = fs.readFileSync(process.argv[1], 'utf8');
const viewerSource = fs.readFileSync(process.argv[2], 'utf8');

function harness(options = {}) {
  const panels = new Map();
  const removed = [];
  const rendererOptions = [];
  const contexts = [];
  let rendererCount = 0;
  const failures = Array.isArray(options.ctorFailures) ? options.ctorFailures.slice() : [];
const renderFailures = Array.isArray(options.renderFailures)
  ? options.renderFailures.slice() : [];

  function context(attrs) {
    const gl = {
      VENDOR: 0x1f00, RENDERER: 0x1f01, MAX_TEXTURE_SIZE: 0x0d33,
      getParameter(value) {
        if (value === 0x1f00) return 'Test vendor';
        if (value === 0x1f01) return 'ANGLE (Test GPU)';
        if (value === 0x0d33) return options.maxTexture || 8192;
        return null;
      },
      getContextAttributes() {
        return {antialias: Boolean(attrs.antialias), alpha: false, depth: true};
      },
      getExtension(name) {
        if (name === 'WEBGL_debug_renderer_info') {
          if (options.debugThrows) throw new Error('debug extension denied');
          return {UNMASKED_VENDOR_WEBGL: 0x9245, UNMASKED_RENDERER_WEBGL: 0x9246};
        }
        if (name === 'WEBGL_lose_context') {
          return {loseContext() { gl.lost = true; }};
        }
        return null;
      },
    };
    contexts.push(gl);
    return gl;
  }

  function canvas() {
    const canvasListeners = {};
    const element = {
      style: {}, className: '', parentNode: null,
      setAttribute() {},
      addEventListener(name, callback) { canvasListeners[name] = callback; },
      removeEventListener() {},
      getBoundingClientRect() { return {left: 0, top: 0, width: 100, height: 100}; },
      getContext(kind, attrs) {
        if (kind === 'webgl2') {
          if (options.webgl2 === false) return null;
          return context(attrs || {});
        }
        if (kind === 'webgl' || kind === 'experimental-webgl') {
          if (options.webgl1 === false) return null;
          return context(attrs || {});
        }
        return null;
      },
      _listeners: canvasListeners,
    };
    return element;
  }

  const document = {
    createElement(name) {
      if (name === 'canvas') return canvas();
      return {style: {}, dataset: {}, setAttribute() {}, appendChild() {}, addEventListener() {}};
    },
    getElementById(id) {
      if (!panels.has(id)) panels.set(id, {
        style: {}, dataset: {}, hidden: true, value: '', textContent: '',
      });
      return panels.get(id);
    },
    execCommand() { return true; },
  };
  const host = {
    clientWidth: 100, clientHeight: 100, textContent: '', children: [],
    appendChild(child) { this.children.push(child); child.parentNode = this; },
    removeChild(child) {
      this.children = this.children.filter(item => item !== child);
      child.parentNode = null;
      removed.push(child);
    },
  };

  const THREE = {
    REVISION: '180', MOUSE: {ROTATE: 0, PAN: 1},
    Scene: class {
      constructor() { this.children = []; }
      add(value) { this.children.push(value); }
      remove() {}
    },
    Color: class {
      constructor() { this.r = 1; this.g = 1; this.b = 1; }
      set() { return this; }
      setRGB(r, g, b) { this.r = r; this.g = g; this.b = b; return this; }
      copy(value) { this.r = value.r; this.g = value.g; this.b = value.b; return this; }
      multiplyScalar() { return this; }
    },
    Group: class {
      constructor() { this.children = []; this.userData = {}; }
      add(value) { this.children.push(value); }
      remove() {}
      traverse(callback) { this.children.forEach(callback); }
    },
    PerspectiveCamera: class {
      constructor() {
        this.up = {set() {}};
        this.position = {set() {}, distanceTo() { return 10; }};
      }
      updateProjectionMatrix() {}
      lookAt() {}
    },
    WebGLRenderer: class {
      constructor(value) {
        rendererOptions.push(value);
        rendererCount += 1;
        if (failures.shift()) throw new Error('renderer constructor failed');
        this.context = value.context || contexts[contexts.length - 1];
      }
      setPixelRatio(value) { this.pixelRatio = value; }
      setSize() {}
      render() { if (renderFailures.shift()) throw new Error('initial render failed'); }
      getContext() { return this.context; }
      dispose() { this.disposed = true; }
      forceContextLoss() { this.forced = true; }
    },
    Raycaster: class {}, Vector2: class {}, Vector3: class {}, Matrix4: class {},
    BufferGeometry: class {}, BufferAttribute: class {}, Uint32BufferAttribute: class {},
    PointsMaterial: class {}, Points: class {}, LineBasicMaterial: class {}, Line: class {},
  };
  THREE.Vector3.prototype.project = function () { return this; };

  class Controls {
    constructor() { this.enabled = true; this.target = {set() {}}; this.mouseButtons = {}; }
    addEventListener() {}
    removeEventListener() {}
    update() { return false; }
    dispose() { this.disposed = true; }
  }
  const sandbox = {
    window: null, document, navigator: {
      userAgent: 'Edg/151.0.4129.107',
      deviceMemory: options.deviceMemory === undefined ? 8 : options.deviceMemory,
      hardwareConcurrency: options.cores === undefined ? 8 : options.cores,
      clipboard: {writeText: async value => { sandbox.copied = value; }},
    },
    devicePixelRatio: 2, performance: {now: () => 0}, console,
    fetch: () => Promise.reject(new Error('native endpoint unavailable')),
    setTimeout, clearTimeout, requestAnimationFrame: () => 1, cancelAnimationFrame() {},
    ResizeObserver: class { observe() {} disconnect() {} },
    WebGL2RenderingContext: options.noWebglConstructors ? undefined : class {},
    WebGLRenderingContext: options.noWebglConstructors ? undefined : class {},
    P3AGiro3D: {THREE, MapControls: Controls},
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(runtimeSource, sandbox, {filename: 'viewer_runtime.js'});
  vm.runInContext(viewerSource, sandbox, {filename: 'tree_review_3d.js'});
  return {
    sandbox, host, panels, removed, rendererOptions, contexts,
    get rendererCount() { return rendererCount; },
  };
}

function newViewer(options) {
  const test = harness(options);
  const viewer = new test.sandbox.TreeLineTreeReview3D(test.host, null, {});
  return {test, viewer};
}

// Normal WebGL2 path: one renderer construction, one initial render, and the
// existing 1.75 device-pixel cap.
{
  const {test, viewer} = newViewer({});
  assert.strictEqual(viewer.rendererMode, 'normal');
  assert.strictEqual(test.rendererOptions[0].antialias, true);
  assert.strictEqual(viewer.renderer.pixelRatio, 1.75);
  assert.strictEqual(viewer.rendererAttempts.length, 1);
  viewer.dispose();
  assert.strictEqual(test.host.children.length, 0);
  assert.ok(test.contexts.some(item => item.lost), 'dispose must release the WebGL context');
}

// A normal constructor failure gets one fresh safe canvas/context.
{
  const {test, viewer} = newViewer({ctorFailures: [true, false]});
  assert.strictEqual(viewer.rendererMode, 'safe');
  assert.strictEqual(viewer.rendererAttempts.map(item => item.mode).join(','), 'normal,safe');
  assert.strictEqual(test.rendererOptions[0].antialias, true);
  assert.strictEqual(test.rendererOptions[1].antialias, false);
  assert.strictEqual(test.rendererOptions[1].alpha, false);
  assert.strictEqual(viewer.renderer.pixelRatio, 1);
  viewer.dispose();
  assert.strictEqual(test.host.children.length, 0);
  assert.ok(test.removed.length >= 2, 'failed and disposed canvases must be removed');
}

// Low-capability devices use safe parameters on the first renderer attempt.
{
  const {test, viewer} = newViewer({maxTexture: 2048});
  assert.strictEqual(viewer.rendererMode, 'safe');
  assert.strictEqual(test.rendererOptions[0].antialias, false);
  assert.strictEqual(viewer.renderer.pixelRatio, 1);
  viewer.dispose();
}

// Two renderer failures produce a structured diagnostic and leave no canvas.
{
  const test = harness({ctorFailures: [true, true]});
  assert.throws(
    () => new test.sandbox.TreeLineTreeReview3D(test.host, null, {}),
    error => error.viewerInitFailure && error.code === 'THREE_RENDERER_FAILED',
  );
  assert.strictEqual(test.rendererCount, 2);
  assert.strictEqual(test.host.children.length, 0);
  assert.ok(test.panels.get('viewer-init-details').value.includes('THREE_RENDERER_FAILED'));
}

// No WebGL API and WebGL2 context creation failure are distinct codes.
{
  const noApi = harness({webgl2: false, webgl1: false, noWebglConstructors: true});
  assert.throws(
    () => new noApi.sandbox.TreeLineTreeReview3D(noApi.host, null, {}),
    error => error.code === 'WEBGL_UNAVAILABLE',
  );
  const nullContext = harness({webgl2: false, webgl1: false});
  assert.throws(
    () => new nullContext.sandbox.TreeLineTreeReview3D(nullContext.host, null, {}),
    error => error.code === 'WEBGL_CONTEXT_CREATION_FAILED',
  );
}

// Debug extension failures are nonfatal, and a missing global is a packaged
// resource failure rather than an editor-data/network failure.
{
  const debug = newViewer({debugThrows: true});
  assert.strictEqual(debug.viewer.rendererInfo().renderer, 'ANGLE (Test GPU)');
  debug.viewer.dispose();
  const missing = harness();
  missing.sandbox.P3AGiro3D = null;
  assert.throws(
    () => new missing.sandbox.TreeLineTreeReview3D(missing.host, null, {}),
    error => error.code === 'VIEWER_RESOURCE_MISSING',
  );
}

// The runtime panel contains capabilities, failure code, and original stack;
// copy uses the clipboard without changing application state.
{
  const test = harness();
  const original = new Error('renderer exploded');
  original.stack = 'STACK: renderer exploded';
  const error = test.sandbox.TreeLineViewerRuntime.notifyFailure(original, {
    phase: 'renderer', capabilities: test.sandbox.TreeLineViewerRuntime.getCapabilities(),
  });
  assert.strictEqual(test.panels.get('viewer-init-failure').hidden, false);
  assert.ok(test.panels.get('viewer-init-details').value.includes('STACK: renderer exploded'));
  assert.ok(test.panels.get('viewer-init-details').value.includes('WebGL2 API'));
  test.sandbox.TreeLineViewerRuntime.copyDiagnostic(error).then(() => {
    assert.ok(String(test.sandbox.copied).includes('renderer exploded'));
  });
}

console.log('clean-PC viewer compatibility checks PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_clean_pc_viewer_compatibility_paths() -> None:
    result = subprocess.run(
        [shutil.which("node") or "node", "-e", _NODE_CHECK, str(RUNTIME_JS), str(VIEWER_JS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


def test_app_wires_structured_viewer_failure_retry_and_copy() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    for token in (
        "viewer-init-failure",
        "viewer-init-summary",
        "viewer-init-details",
        "btn-viewer-retry",
        "btn-viewer-copy",
        "retryViewer",
        "displayRefreshUncertain",
        "viewerRuntime",
    ):
        assert token in source
