"""P3A-I5 static viewer-deliverable checks (no browser required).

Asserts the POC viewer files exist, reference only the three served roots, and
that every Potree asset the page includes actually resolves on disk — so the
loopback-served page has no missing-asset 404s before a browser is involved.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.serve_p3a_potree import DATASET_ROOT, POTREE_ROOT, VIEWER_ROOT

INDEX = VIEWER_ROOT / "index.html"
VIEWER_JS = VIEWER_ROOT / "viewer.js"
VIEWER_PERF = VIEWER_ROOT / "viewer_perf.js"
CONFIG = VIEWER_ROOT / "poc_config.json"

_REF_RE = re.compile(r'(?:src|href)=["\']([^"\']+)["\']')
_ALLOWED_PREFIXES = (
    "potree/",
    "viewer.js",
    "viewer_util.js",
    "viewer_perf.js",
    "poc_config.json",
)


def _refs(html: str) -> list[str]:
    return _REF_RE.findall(html)


def test_deliverables_exist() -> None:
    assert INDEX.is_file()
    assert VIEWER_JS.is_file()
    assert VIEWER_PERF.is_file()
    assert CONFIG.is_file()


def test_index_references_stay_within_served_roots() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for ref in _refs(html):
        assert ref.startswith(_ALLOWED_PREFIXES), f"reference escapes served roots: {ref}"
        assert not ref.startswith("../"), f"parent traversal in reference: {ref}"


def test_every_potree_asset_reference_resolves() -> None:
    html = INDEX.read_text(encoding="utf-8")
    refs = [r for r in _refs(html) if r.startswith("potree/")]
    assert refs, "expected at least one potree/ asset reference"
    for ref in refs:
        target = POTREE_ROOT / ref[len("potree/"):]
        assert target.is_file(), f"missing Potree asset on disk: {ref}"


def test_config_is_valid_json() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["pointcloudUrl"].endswith("metadata.json")
    assert cfg["pointcloudName"]
    assert cfg["viewer"]["activeAttribute"] in ("rgb", "rgba")
    assert cfg["viewer"]["pointBudget"] > 0


def test_viewer_js_hooks_into_potree_and_config() -> None:
    js = VIEWER_JS.read_text(encoding="utf-8")
    assert "Potree.loadPointCloud" in js
    assert "poc_config.json" in js
    assert "potree_render_area" in js


def test_perf_script_loaded_before_viewer_and_init_is_fail_open() -> None:
    """P3A perf diagnostics are an OPTIONAL enhancement.

    The script is referenced before viewer.js and viewer.js only calls
    p3aPerf.init() behind a capability guard, so a missing/corrupt perf helper
    can never break the viewer (fail-open), unlike the REQUIRED startup helpers
    which fail closed."""
    html = INDEX.read_text(encoding="utf-8")
    assert html.index("viewer_perf.js") < html.index("viewer.js")
    js = VIEWER_JS.read_text(encoding="utf-8")
    assert "window.p3aPerf" in js
    assert 'typeof window.p3aPerf.init === "function"' in js
    assert "p3aPerf.init({" in js
    # The init call is AFTER the successful load (pointcloud attached + fit
    # registered), so diagnostics never touch a half-initialized scene.
    assert js.index("p3aPerf.init({") > js.index("viewer.scene.addPointCloud(pointcloud)")


@pytest.mark.skipif(
    not (DATASET_ROOT / "p3a_roi" / "metadata.json").is_file(),
    reason="ROI octree not converted",
)
def test_pointcloud_target_resolves() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    url = cfg["pointcloudUrl"]  # e.g. pointclouds/p3a_roi/metadata.json
    target = DATASET_ROOT / url.split("/", 1)[1]
    assert target.is_file(), f"pointcloud target missing: {url}"


# ---------------------------------------------------------------------------
# P3A-004: visible dataset-load diagnostics (static, no browser required)
# ---------------------------------------------------------------------------


def _viewer_js() -> str:
    return VIEWER_JS.read_text(encoding="utf-8")


def test_metadata_preflight_exists() -> None:
    js = _viewer_js()
    assert "preflightMetadata" in js
    assert "fetch(pointcloudUrl" in js
    assert "response.ok" in js
    assert "metadata.attributes" in js


def test_preflight_gates_potree_load_and_failure_routes_to_banner() -> None:
    js = _viewer_js()
    # preflight runs before Potree startup in the same try block, so a failed
    # preflight jumps to the shared catch that calls showError() and
    # Potree.loadPointCloud is never reached.
    assert js.index("preflightMetadata(cfg.pointcloudUrl)") < js.index(
        "Potree.loadPointCloud("
    )
    assert "showError(" in js


def test_global_error_handlers_route_to_banner() -> None:
    js = _viewer_js()
    assert 'addEventListener("error"' in js
    assert 'addEventListener("unhandledrejection"' in js


def test_load_timeout_bounds_the_load() -> None:
    js = _viewer_js()
    assert "withTimeout(" in js
    assert "timed out" in js
    assert "resolveLoadTimeoutMs(" in js
    assert "loadTimeoutMs" in CONFIG.read_text(encoding="utf-8")


def test_startup_guard_precedes_helper_use() -> None:
    """P3A-010: error UI + global handlers are installed before any helper use,
    and the helper is dereferenced only after the startup guard.

    Anchored on code tokens (not docstring mentions) so the check reflects the
    actual control flow."""
    js = _viewer_js()
    show_error_idx = js.index("function showError")
    error_handler_idx = js.index('window.addEventListener("error"')
    unhandled_idx = js.index('window.addEventListener("unhandledrejection"')
    guard_idx = js.index("const decide = window.p3aDecideViewerStartup")
    util_use_idx = js.index("decide(window.p3aViewerUtil")
    resolver_use_idx = js.index("util.resolveLoadTimeoutMs(")

    # Error UI + global handlers are established before the guard runs.
    assert show_error_idx < guard_idx
    assert error_handler_idx < guard_idx
    assert unhandled_idx < guard_idx
    # The helper global is only passed to the guard, and the resolver is only
    # used after the guard passes.
    assert util_use_idx > guard_idx
    assert resolver_use_idx > util_use_idx
    # No early top-level dereference of the helper before error infrastructure.
    assert "window.p3aViewerUtil.resolveLoadTimeoutMs" not in js
    assert "window.p3aViewerUtil.DEFAULT_LOAD_TIMEOUT_MS" not in js


# ---------------------------------------------------------------------------
# P3A-008: timeout validation (executes the ACTUAL helper via Node)
# ---------------------------------------------------------------------------

VIEWER_UTIL = VIEWER_ROOT / "viewer_util.js"


def test_viewer_uses_shared_timeout_helper() -> None:
    assert VIEWER_UTIL.is_file()
    js = _viewer_js()
    assert "resolveLoadTimeoutMs" in js
    assert "p3aDecideViewerStartup" in js
    html = INDEX.read_text(encoding="utf-8")
    assert html.index("viewer_util.js") < html.index("viewer.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_timeout_validation_behavior() -> None:
    util_path = str(VIEWER_UTIL).replace("\\", "/")
    template = r"""
const fs = require('fs');
eval(fs.readFileSync("__PATH__", 'utf8'));
const u = globalThis.p3aViewerUtil;
if (!u) { throw new Error('p3aViewerUtil not exposed'); }
const cases = [
  [undefined, 30000],
  [-1, 30000],
  [0, 30000],
  ['bad', 30000],
  ['30000', 30000],
  [null, 30000],
  [false, 30000],
  [{}, 30000],
  [[], 30000],
  [NaN, 30000],
  [Infinity, 30000],
  [-Infinity, 30000],
  [1, 1],
  [30000, 30000],
  [45000, 45000],
];
let failed = 0;
for (const [input, expected] of cases) {
  const got = u.resolveLoadTimeoutMs(input);
  if (got !== expected) {
    failed += 1;
    console.log('FAIL input=' + String(input) + ' got=' + got + ' expected=' + expected);
  }
}
if (u.DEFAULT_LOAD_TIMEOUT_MS !== 30000) {
  failed += 1;
  console.log('FAIL default != 30000');
}
if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A-008 helper tests PASS');
"""
    result = subprocess.run(
        ["node", "-e", template.replace("__PATH__", util_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"node helper test failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# P3A-009: displayed point count matches the current sample
# ---------------------------------------------------------------------------


def test_config_description_has_no_stale_sample_count() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    description = cfg["description"]
    # The active acceptance dataset is the dense ROI; the old global stride
    # sample count (299,303) must not be shown.
    assert "300,159" not in description
    assert "299,303" not in description


@pytest.mark.skipif(
    not (DATASET_ROOT.parent / "roi-provenance.json").is_file(),
    reason="ROI provenance not present",
)
def test_config_description_matches_roi_provenance_count() -> None:
    prov = json.loads(
        (DATASET_ROOT.parent / "roi-provenance.json").read_text(encoding="utf-8")
    )
    actual = prov["sample"]["point_count"]
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    # The description renders the count comma-grouped (e.g. "982,504").
    assert f"{actual:,}" in cfg["description"]


# ---------------------------------------------------------------------------
# P3A-010: startup guard — helper must be validated before any dereference
# ---------------------------------------------------------------------------

_DECIDE_SCRIPT = r"""
const fs = require('fs');
eval(fs.readFileSync("__UTIL_PATH__", 'utf8'));
const decide = globalThis.p3aDecideViewerStartup;
const util = globalThis.p3aViewerUtil;
let failed = 0;
function check(cond, msg) {
  if (!cond) { failed += 1; console.log('FAIL: ' + msg); }
}
check(typeof decide === 'function', 'p3aDecideViewerStartup is exposed');
check(decide(undefined).ok === false, 'undefined -> not ok');
check(decide(null).ok === false, 'null -> not ok');
check(decide({}).ok === false, 'empty object -> not ok');
check(decide({ resolveLoadTimeoutMs: 5 }).ok === false, 'resolver not a function -> not ok');
const ok = decide(util);
check(ok.ok === true, 'valid util -> ok');
check(ok.util === util, 'valid util -> util returned');
if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A-010 decideViewerStartup tests PASS');
"""

_VM_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');

const UTIL_PATH = "__UTIL_PATH__";
const VIEWER_PATH = "__VIEWER_PATH__";

const utilCode = fs.readFileSync(UTIL_PATH, 'utf8');
const viewerCode = fs.readFileSync(VIEWER_PATH, 'utf8');

function makeCalls() {
  return {
    potreeLoad: 0, fetchCount: 0, errors: [], handlers: {},
    orbitSet: false, fitToScreen: 0, appendChild: 0,
    createdEls: [], clickHandler: null, lastPointcloud: null,
  };
}

function MockBox3() {
  this.min = { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } };
  this.max = { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } };
}

function runScenario(opts) {
  const calls = makeCalls();
  const renderer = {
    insertAdjacentHTML(pos, html) { calls.errors.push(html); },
    appendChild() { calls.appendChild += 1; },
  };
  const document = {
    title: '',
    getElementById(id) {
      if (id === 'potree_render_area') return renderer;
      if (id === 'p3a_loading_banner') return { remove() {} };
      return null;
    },
    createElement(tag) {
      const el = {
        tagName: tag, id: '', style: {}, textContent: '', innerHTML: '',
        addEventListener(type, fn) { if (type === 'click') { calls.clickHandler = fn; } },
        remove() {},
      };
      calls.createdEls.push(el);
      return el;
    },
  };
  const Potree = {
    Viewer: function () {
      const self = {
        orbitControls: { enabled: false },
        setControls(c) { calls.orbitSet = (c === self.orbitControls); },
        fitToScreen() { calls.fitToScreen += 1; },
        setFOV() {}, setEDLEnabled() {}, setPointBudget() {},
        setDescription() {}, loadSettingsFromURL() {}, setLanguage() {},
        loadGUI(cb) { if (cb) cb(); },
        scene: { addPointCloud() {} },
      };
      return self;
    },
    PointSizeType: { ADAPTIVE: 0 },
    loadPointCloud(url, name, cb) {
      calls.potreeLoad += 1;
      const pointcloud = {
        material: {},
        updateMatrixWorld() {},
        matrixWorld: {},
        pcoGeometry: {
          boundingBox: { clone() { return new MockBox3(); } },
        },
      };
      calls.lastPointcloud = pointcloud;
      if (cb) { cb({ pointcloud: pointcloud }); }
      return Promise.resolve({ pointcloud: pointcloud });
    },
  };
  const fetch = function (url) {
    calls.fetchCount += 1;
    if (String(url).indexOf('poc_config.json') !== -1) {
      return Promise.resolve({
        ok: true,
        json: function () { return Promise.resolve({
          title: 'P3A test',
          pointcloudUrl: 'pointclouds/p3a_roi/metadata.json',
          pointcloudName: 'p3a_roi',
          description: 'test',
          viewer: {
            fov: 60, edlEnabled: false, pointBudget: 1000000,
            activeAttribute: 'rgba', size: 1, minSize: 2, maxSize: 5,
            loadTimeoutMs: 30000,
          },
          controlHelp: { left: '左键拖动: 旋转', right: '右键拖动: 平移', wheel: '滚轮: 缩放' },
        }); },
      });
    }
    return Promise.resolve({
      ok: true,
      json: function () { return Promise.resolve({
        attributes: [{ name: 'position', min: [0, 0, 0], max: [10, 10, 10] }],
        boundingBox: { min: [0, 0, 0], max: [20, 20, 20] },
      }); },
    });
  };

  const sandbox = {
    document: document,
    Potree: Potree,
    fetch: fetch,
    console: console,
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
    requestAnimationFrame: function (cb) { cb(); return 1; },
    addEventListener: function (type) { calls.handlers[type] = true; },
    window: {},
  };
  sandbox.window = sandbox;

  if (opts.loadUtil) {
    vm.createContext(sandbox);
    vm.runInContext(utilCode, sandbox, { filename: 'viewer_util.js' });
  } else {
    vm.createContext(sandbox);
  }
  if (opts.p3aViewerUtil !== undefined) { sandbox.p3aViewerUtil = opts.p3aViewerUtil; }
  if (opts.p3aDecideViewerStartup !== undefined) { sandbox.p3aDecideViewerStartup = opts.p3aDecideViewerStartup; }

  vm.runInContext(viewerCode, sandbox, { filename: 'viewer.js' });
  return new Promise(function (resolve) { setTimeout(function () { resolve(calls); }, 30); });
}

(async function () {
  let failed = 0;
  function check(cond, msg) {
    if (!cond) { failed += 1; console.log('FAIL: ' + msg); }
  }

  // Scenario 1: viewer_util.js completely absent -> visible error, no Potree.
  {
    const calls = await runScenario({ loadUtil: false });
    check(calls.potreeLoad === 0, 'helper-absent: Potree.loadPointCloud must not be called');
    check(calls.fetchCount === 0, 'helper-absent: config/metadata fetch must not be called');
    check(calls.errors.length >= 1, 'helper-absent: a visible error banner must be shown');
    check(/required viewer utility failed to load/.test(calls.errors.join('\n')),
      'helper-absent: error identifies the missing utility');
  }

  // Scenario 2: util present but corrupt (no resolveLoadTimeoutMs function).
  {
    const calls = await runScenario({ loadUtil: true, p3aViewerUtil: {} });
    check(calls.potreeLoad === 0, 'resolver-missing: Potree.loadPointCloud must not be called');
    check(calls.fetchCount === 0, 'resolver-missing: config/metadata fetch must not be called');
    check(calls.errors.length >= 1, 'resolver-missing: a visible error banner must be shown');
  }

  // Scenario 3: happy path with the real helper -> Potree load proceeds.
  {
    const calls = await runScenario({ loadUtil: true });
    check(calls.potreeLoad === 1, 'happy: Potree.loadPointCloud must be called once');
    check(calls.fetchCount >= 2, 'happy: config + metadata preflight must be fetched');
    check(calls.errors.length === 0, 'happy: no error banner shown');
    check(calls.handlers['error'] === true, 'happy: window error handler installed');
    check(calls.handlers['unhandledrejection'] === true, 'happy: unhandledrejection handler installed');

    // VISUAL-002: orbit controls explicitly selected via setControls.
    check(calls.orbitSet === true, 'happy: orbitControls explicitly selected via setControls');
    // VISUAL-001: tight bounds set from the metadata position attribute and the
    // official fit path invoked only after a successful load.
    check(calls.lastPointcloud !== null &&
          calls.lastPointcloud.pcoGeometry.tightBoundingBox !== undefined,
      'happy: tight bounding box set from metadata position');
    check(calls.fitToScreen >= 1, 'happy: initial tight fit called after load');
    // Fit/Reset button + control help are appended.
    check(calls.appendChild >= 2, 'happy: control help + Fit/Reset appended');
    const helpEl = calls.createdEls.find(function (e) { return e.id === 'p3a_control_help'; });
    check(helpEl !== undefined && helpEl.textContent.indexOf('旋转') !== -1,
      'happy: control help shows rotate');
    const fitBtn = calls.createdEls.find(function (e) { return e.id === 'p3a_fit_reset'; });
    check(fitBtn !== undefined, 'happy: Fit/Reset button created');
    // Clicking Fit/Reset re-frames through the same Potree fit path.
    if (calls.clickHandler) { calls.clickHandler(); }
    check(calls.fitToScreen >= 2, 'happy: Fit/Reset click re-frames');
  }

  if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
  console.log('P3A-010 viewer startup control-flow tests PASS');
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_decide_viewer_startup_behavior() -> None:
    util_path = str(VIEWER_UTIL).replace("\\", "/")
    result = subprocess.run(
        ["node", "-e", _DECIDE_SCRIPT.replace("__UTIL_PATH__", util_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_startup_control_flow() -> None:
    util_path = str(VIEWER_UTIL).replace("\\", "/")
    viewer_path = str(VIEWER_JS).replace("\\", "/")
    script = (
        _VM_SCRIPT.replace("__UTIL_PATH__", util_path).replace(
            "__VIEWER_PATH__", viewer_path
        )
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# VISUAL-001 / VISUAL-002: camera + controls config (static)
# ---------------------------------------------------------------------------


def test_orbit_controls_explicitly_selected() -> None:
    js = _viewer_js()
    assert "viewer.setControls(viewer.orbitControls)" in js


def test_initial_fit_only_after_successful_load() -> None:
    js = _viewer_js()
    # The tight-bounds fit is registered only inside the load-success path,
    # after the point cloud has been attached to the scene.
    add_idx = js.index("viewer.scene.addPointCloud(pointcloud)")
    rAF_idx = js.index("requestAnimationFrame(fit)")
    assert add_idx < rAF_idx
    assert "frameToPointcloud(viewer, pointcloud, metadata)" in js


def test_fit_reset_control_invokes_framing_path() -> None:
    js = _viewer_js()
    assert "addFitResetButton(fit)" in js
    assert "frameToPointcloud" in js
    assert "viewer.fitToScreen()" in js


def test_tight_bounds_from_metadata_position() -> None:
    js = _viewer_js()
    assert "tightBoundingBox" in js
    assert 'a.name === "position"' in js
    assert "metadata.boundingBox.min" in js
    assert "positionAttr.min" in js


def test_control_help_matches_actual_controls() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    help_text = "".join(cfg.get("controlHelp", {}).values())
    assert "旋转" in help_text
    assert "平移" in help_text
    assert "缩放" in help_text
    js = _viewer_js()
    assert "addControlHelp" in js
    # The code path matches the documented Potree 1.8.2 control mapping.
    assert "setControls(viewer.orbitControls)" in js
