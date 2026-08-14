"""P3A Giro3D + COPC viewer (Candidate B) static deliverable checks.

No browser is required.  Verifies the required files exist, the configuration
is valid and matches the ROI, the pure decision helpers (LAS-header bounds
parsing, framing math, click->XYZ plumbing) behave correctly under Node, and
the viewer performs no automatic scientific-pipeline work.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.serve_p3a_giro3d import DATASET_ROOT, VIEWER_ROOT, VENDOR_ROOT, LAZPERF_ROOT

# Comment stripping for source-level checks (docstrings describe what the viewer
# does NOT do; the checks must assert about CODE).
_COMMENT_RE_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_RE_LINE = re.compile(r"//[^\n]*")

VIEWER_JS = VIEWER_ROOT / "viewer.js"
INDEX = VIEWER_ROOT / "index.html"
CONFIG = VIEWER_ROOT / "poc_config.json"
BUNDLE = VENDOR_ROOT / "giro3d.bundle.js"
WASM = LAZPERF_ROOT / "laz-perf.wasm"
COPC = DATASET_ROOT / "p3a_roi.copc.laz"


def _viewer_src() -> str:
    return VIEWER_JS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Static required files
# ---------------------------------------------------------------------------


def test_required_files_exist() -> None:
    assert INDEX.is_file()
    assert VIEWER_JS.is_file()
    assert CONFIG.is_file()
    assert BUNDLE.is_file(), "Giro3D bundle must be vendored under viewer_poc/giro3d/vendor/"
    assert WASM.is_file(), "laz-perf.wasm must be vendored for local LAZ decompression"


def test_index_references_served_roots_only() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert "vendor/giro3d.bundle.js" in html
    assert "viewer.js" in html
    for ref in ("../", "http://", "https://"):
        assert ref not in html, f"external/escape reference in index: {ref}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_valid_and_matches_roi() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["dataset"] == "p3a_roi"
    assert cfg["pointCount"] == 982504
    assert cfg["copcUrl"].endswith("p3a_roi.copc.laz")
    assert cfg["lazPerfPath"].startswith("/")
    assert cfg["viewer"]["pointBudget"] > 0


def test_config_points_match_active_copc_header() -> None:
    """The configured point count must match the ACTIVE COPC LAS header."""
    assert COPC.is_file(), "active COPC missing (run prepare_p3a_giro3d_copc.py)"
    hdr = COPC.read_bytes()[:375]
    # LAS header: point count is an 8-byte little-endian value at offset 107.
    count = int.from_bytes(hdr[107:115], "little")
    assert count == 982504
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["pointCount"] == count


FULL_COPC = Path(__file__).resolve().parents[2] / "cache" / "giro3d_full_source" / "dataset" / "p3a_full.copc.laz"
FULL_CONFIG = VIEWER_ROOT / "poc_config_full.json"


def test_full_source_config_valid() -> None:
    cfg = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["dataset"] == "p3a_full"
    assert cfg["pointCount"] == 105055350
    assert cfg["copcUrl"].endswith("p3a_full.copc.laz")
    assert cfg["lazPerfPath"].startswith("/")
    assert cfg["viewer"]["pointBudget"] > 0


@pytest.mark.skipif(not FULL_COPC.is_file(), reason="full-source COPC not converted")
def test_full_source_copc_header_matches_source_count() -> None:
    """The full-source COPC LAS header must report the complete source count."""
    hdr = FULL_COPC.read_bytes()[:375]
    count = int.from_bytes(hdr[107:115], "little")
    assert count == 105055350
    cfg = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["pointCount"] == count


# ---------------------------------------------------------------------------
# Pure decision helpers (Node)
# ---------------------------------------------------------------------------

_HELPER_SCRIPT = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = {
  window: {},
  document: { getElementById: () => null },
  fetch: () => Promise.reject(new Error('no fetch')),
  console: console,
  setTimeout: setTimeout, clearTimeout: clearTimeout, setInterval: () => 0,
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const u = sandbox.p3aGiro3dUtil;
if (!u) { console.log('FAIL: p3aGiro3dUtil not exposed'); process.exit(1); }
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

// LAS-header bounds parsing (synthetic header)
const buf = new ArrayBuffer(375);
const dv = new DataView(buf);
dv.setFloat64(187, 100.0, true); dv.setFloat64(203, 200.0, true); dv.setFloat64(219, 300.0, true);
dv.setFloat64(179, 110.0, true); dv.setFloat64(195, 210.0, true); dv.setFloat64(211, 310.0, true);
const b = u.parseLasHeaderBounds(new Uint8Array(buf));
check(b.min.x === 100 && b.min.y === 200 && b.min.z === 300, 'bounds min parsed');
check(b.max.x === 110 && b.max.y === 210 && b.max.z === 310, 'bounds max parsed');

// Framing math
const f = u.computeFitCamera({ x: 0, y: 0, z: 0 }, { x: 100, y: 100, z: 100 }, 60);
check(Math.abs(f.maxDim - 100) < 1e-9, 'fit maxDim');
check(Math.abs(f.target.x - 50) < 1e-9 && Math.abs(f.target.z - 50) < 1e-9, 'fit target is box center');
// The default direction frames from above-and-to-the-side (z above the target),
// so the corridor/towers are recognizable immediately — not a top-center dot.
check(f.position.z > f.target.z, 'fit camera is above the target');
check(f.position.x !== f.target.x || f.position.y !== f.target.y, 'fit camera is off-axis (angled view)');
// Default direction frames the occupied box at a useful distance.
const dist = Math.sqrt((f.position.x - f.target.x) ** 2 + (f.position.y - f.target.y) ** 2 + (f.position.z - f.target.z) ** 2);
check(dist > 40 && dist < 200, 'fit camera distance is useful, got ' + dist.toFixed(1));

// Click -> XYZ plumbing
const xyz = u.pickResultToXyz([{ point: { x: 1.5, y: -2.5, z: 3.25 } }]);
check(xyz && xyz.x === 1.5 && xyz.y === -2.5 && xyz.z === 3.25, 'pick xyz extracted');
check(u.pickResultToXyz([]) === null, 'empty pick -> null');
check(u.pickResultToXyz(null) === null, 'null pick -> null');
check(u.pickResultToXyz([{}]) === null, 'pick without point -> null');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('P3A giro3d helpers PASS');
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_pure_decision_helpers() -> None:
    result = subprocess.run(
        ["node", "-e", _HELPER_SCRIPT.replace("__VIEWER_JS__", str(VIEWER_JS).replace("\\", "/"))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# No automatic scientific pipeline
# ---------------------------------------------------------------------------


def test_viewer_never_runs_scientific_pipeline() -> None:
    # Strip comments first — the viewer docstring legitimately says "no
    # prediction / no tower review"; the check is about CODE, not documentation.
    js = _COMMENT_RE_BLOCK.sub(" ", _viewer_src())
    js = _COMMENT_RE_LINE.sub(" ", js)
    for token in (
        "prepare_p3a_potree_sample",
        "prepare_p3a_giro3d_copc",
        "PotreeConverter",
        "run_manual_single_span",
        "SourceIdentity",
        "predict",
        "tower",
    ):
        assert token not in js, f"scientific/prediction token present in viewer code: {token}"
    # The viewer only loads the configured COPC URL (no conversion/processing).
    assert "COPCSource" in js
    assert "cfg.copcUrl" in js


def test_viewer_has_required_features() -> None:
    js = _viewer_src()
    for token in (
        "computeFitCamera",
        "parseLasHeaderBounds",
        "pickResultToXyz",
        "MapControls",
        "enableDamping",
        "pointCloud.pick",
        "setLazPerfPath",
        "fitBtn.addEventListener",
    ):
        assert token in js, f"missing viewer feature: {token}"
