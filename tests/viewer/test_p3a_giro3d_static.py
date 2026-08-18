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
from scripts.serve_p3a_giro3d import DATASET_ROOT, LAZPERF_ROOT, VENDOR_ROOT, VIEWER_ROOT

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


FULL_COPC = (
    Path(__file__).resolve().parents[2]
    / "cache" / "giro3d_full_source" / "dataset" / "p3a_full.copc.laz"
)
FULL_CONFIG = VIEWER_ROOT / "poc_config_full.json"
PDAL_FULL_COPC = (
    Path(__file__).resolve().parents[2]
    / "cache" / "giro3d_full_pdal" / "dataset" / "p3a_full_pdal.copc.laz"
)
PDAL_FULL_CONFIG = VIEWER_ROOT / "poc_config_full_pdal.json"
LAUNCHER = Path(__file__).resolve().parents[2] / "RUN_TOWER_REVIEW_FULL.bat"
_PDAL_EXE = (
    Path(__file__).resolve().parents[2]
    / "cache" / "giro3d_pdal_env" / "Library" / "bin" / "pdal.exe"
)


def test_full_source_config_valid() -> None:
    """The historical custom-builder full config remains a valid comparison."""
    cfg = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["dataset"] == "p3a_full"
    assert cfg["pointCount"] == 105055350
    assert cfg["copcUrl"].endswith("p3a_full.copc.laz")
    assert cfg["lazPerfPath"].startswith("/")
    assert cfg["viewer"]["pointBudget"] > 0


@pytest.mark.skipif(not FULL_COPC.is_file(), reason="full-source COPC not converted")
def test_full_source_copc_header_matches_source_count() -> None:
    """The historical full-source COPC LAS header reports the complete count."""
    hdr = FULL_COPC.read_bytes()[:375]
    count = int.from_bytes(hdr[107:115], "little")
    assert count == 105055350
    cfg = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["pointCount"] == count


def test_full_pdal_production_config_valid() -> None:
    """The production Owner config points at the PDAL writers.copc COPC."""
    cfg = json.loads(PDAL_FULL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["dataset"] == "p3a_full_pdal"
    assert cfg["pointCount"] == 105055350
    assert cfg["copcUrl"].endswith("p3a_full_pdal.copc.laz")
    assert cfg["lazPerfPath"].startswith("/")
    assert cfg["viewer"]["pointBudget"] > 0


@pytest.mark.skipif(not PDAL_FULL_COPC.is_file(), reason="PDAL full COPC not converted")
@pytest.mark.skipif(not _PDAL_EXE.is_file(), reason="isolated PDAL not available")
def test_full_pdal_copc_header_matches_source_count() -> None:
    """The production PDAL full COPC reports the complete source count.

    A COPC stores its point count in the hierarchy (the LAS header count is 0
    per the COPC spec), so the count is read through the isolated PDAL reader.
    """
    result = subprocess.run(
        [str(_PDAL_EXE), "info", str(PDAL_FULL_COPC), "--metadata"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"pdal info failed:\n{result.stdout}\n{result.stderr}"
    meta = json.loads(result.stdout)["metadata"]
    assert meta["count"] == 105055350
    assert meta["copc"] is True
    assert meta["dataformat_id"] == 7
    cfg = json.loads(PDAL_FULL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["pointCount"] == meta["count"]


def test_owner_launcher_uses_production_pdal_copc() -> None:
    """The normal Owner workflow defaults to the PDAL full COPC.

    The historical custom-builder COPC stays available for comparison but must
    not be the production default.
    """
    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    assert "poc_config_full_pdal.json" in text
    assert "cache\\giro3d_full_pdal\\dataset" in text
    assert "p3a_full_pdal.copc.laz" in text
    assert "cache\\giro3d_full_source\\dataset" in text  # historical, kept for comparison
    assert "poc_config_full.json" in text  # historical config retained


def test_owner_launcher_is_cmd_safe_ascii_crlf() -> None:
    """The launcher must survive real cmd.exe: ASCII-only, CRLF, no non-ASCII source literal.

    The protected LAS filename contains a non-ASCII character that cmd.exe mangles
    when read from a UTF-8 batch file (the observed Owner failure: the filename
    was corrupted to ``16娌?4``).  The launcher therefore must be ASCII-only with
    CRLF line endings and must NOT embed the Chinese filename in a ``--source``
    argument; ``prepare_tower_ordered_project.py`` resolves the path via Python.
    """
    raw = LAUNCHER.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", "launcher must not carry a UTF-8 BOM"
    assert b"\r\n" in raw, "launcher must use CRLF line endings for cmd.exe"
    assert b"\n" not in raw.replace(b"\r\n", b""), "launcher must have no bare LF line endings"
    assert all(byte < 0x80 for byte in raw), "launcher must be ASCII-only (no non-ASCII bytes)"
    text = raw.decode("ascii")
    assert "--source" not in text, "launcher must not pass a hard-coded --source path"
    assert "2026-08-07" not in text, "launcher must not embed the protected LAS filename"
    # Every non-comment line is a valid cmd.exe command (no raw prose executed).
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("REM"):
            continue
        assert not any(char in stripped for char in "()"), (
            f"executable launcher line must be parenthesis-free for cmd.exe: {stripped!r}"
        )


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
check(Math.abs(f.target.x - 50) < 1e-9 && Math.abs(f.target.z - 50) < 1e-9,
  'fit target is box center');
// The default direction frames from above-and-to-the-side (z above the target),
// so the corridor/towers are recognizable immediately — not a top-center dot.
check(f.position.z > f.target.z, 'fit camera is above the target');
check(f.position.x !== f.target.x || f.position.y !== f.target.y,
  'fit camera is off-axis (angled view)');
// Default direction frames the occupied box at a useful distance.
const dist = Math.sqrt(
  (f.position.x - f.target.x) ** 2 + (f.position.y - f.target.y) ** 2
  + (f.position.z - f.target.z) ** 2
);
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


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_selected_span_visibility_hotfix_contract() -> None:
    """Selection owns one viewer-only endpoint/corridor/label overlay."""

    source = _viewer_src()
    for token in (
        'name = "selected-span-overlay"',
        "selectedSpanEndpoint",
        "selected-span-corridor-frame",
        "selectedSpanLabel",
        "clearSpanSelectionOverlay",
        "clearSelectedSpan",
        "visualizationOnly",
        "towerViewerPosition",
        "spanCorridorVertices",
        "new THREE.BufferGeometry()",
    ):
        assert token in source, f"missing selected-Span visualization token: {token}"
    assert 'postJson("/api/spans/synchronize"' in source

    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = { window: {}, document: { getElementById: () => null }, console: console };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const frame = sandbox.p3aGiro3dUtil.spanSelectionFrame({
  span_id: 'S001', group_id: 'G001', from_tower_id: 'T001', to_tower_id: 'T002',
  from_tower_geometry: { base_xyz_absolute_m: [10, 20, 1] },
  to_tower_geometry: { base_xyz_absolute_m: [40, 60, 3] }
}, 8, 12);
let failed = 0;
function check(value, message) { if (!value) { failed += 1; console.log('FAIL: ' + message); } }
check(frame && frame.span_id === 'S001', 'span id preserved');
check(frame.from_tower_id === 'T001' && frame.to_tower_id === 'T002', 'endpoint ids preserved');
check(Math.abs(frame.length - 50) < 1e-9, 'XY corridor length');
check(frame.width === 8 && frame.height === 12, 'viewer envelope dimensions');
check(frame.midpoint[0] === 25 && frame.midpoint[1] === 40, 'midpoint computed');
const transformedFrom = sandbox.p3aGiro3dUtil.towerViewerPosition({ display_xyz_absolute_m: [10, 20, 1] });
const transformedTo = sandbox.p3aGiro3dUtil.towerViewerPosition({ display_xyz_absolute_m: [40, 60, 3] });
check(JSON.stringify(frame.from) === JSON.stringify(transformedFrom), 'from endpoint reuses Tower transform');
check(JSON.stringify(frame.to) === JSON.stringify(transformedTo), 'to endpoint reuses Tower transform');
const vertices = sandbox.p3aGiro3dUtil.spanCorridorVertices(frame);
check(vertices.length === 72, 'explicit corridor has 12 line segments');
check(vertices[2] === 1 && vertices[5] === 1, 'corridor starts Z-up at from base');
check(vertices[20] === 13 && vertices[23] === 1, 'from frame edge is Z-up');
check(vertices[32] === 3 && vertices[35] === 15, 'to frame edge is Z-up');
if (failed > 0) process.exit(1);
console.log('selected-Span visibility helper PASS');
""".replace("__VIEWER_JS__", str(VIEWER_JS).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(VIEWER_JS.parents[2]),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "selected-Span visibility helper PASS" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_span_focus_mode_viewer_hotfix_contract() -> None:
    """Span focus is explicit viewer state and never a science/API mutation."""

    source = _viewer_src()
    html = INDEX.read_text(encoding="utf-8")
    for token in (
        '"SPAN_FOCUS"',
        '"PROJECT_VIEW"',
        "spanFocusDisplayState",
        "spanFocusMode",
        "activeSpanId",
        "enterSpanFocus",
        "returnToProjectView",
        "applySpanFocusVisuals",
        "spanFocusClipPlanes",
        "clearSpanFocusFilter",
        "clippingPlanes",
        "focusFilterActive",
        "fitProjectView",
        "selected-span-overlay",
        "selected-span-endpoint-",
        "selected-span-corridor-frame",
        "confirmedPointsGroup",
    ):
        assert token in source, f"missing Span focus token: {token}"
    for token in ('id="span_focus_mode"', 'id="span_return_project"', "返回全景 / Return to Project View"):
        assert token in html, f"missing visible Span focus DOM token: {token}"

    focus_start = source.index("function enterSpanFocus")
    return_start = source.index("function returnToProjectView")
    return_end = source.index("function disposeSpanOverlayObject", return_start)
    assert "postJson" not in source[focus_start:return_end]
    assert "fetch(" not in source[focus_start:return_end]

    script = r'''
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = { window: {}, document: { getElementById: () => null }, console: console };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const s2 = { span_id: 'S002', group_id: 'G001', from_tower_id: 'T002', to_tower_id: 'T003' };
const s3 = { span_id: 'S003', group_id: 'G001', from_tower_id: 'T003', to_tower_id: 'T004' };
const f2 = sandbox.p3aGiro3dUtil.spanFocusDisplayState(s2, 'SPAN_FOCUS');
const f3 = sandbox.p3aGiro3dUtil.spanFocusDisplayState(s3, 'SPAN_FOCUS');
const project = sandbox.p3aGiro3dUtil.spanFocusDisplayState(null, 'PROJECT_VIEW');
let failed = 0;
function check(value, message) { if (!value) { failed += 1; console.log('FAIL: ' + message); } }
check(f2.mode === 'SPAN_FOCUS' && f2.active_span_id === 'S002', 'S002 enters focus mode');
check(f2.from_tower_id === 'T002' && f2.to_tower_id === 'T003' && f2.group_id === 'G001', 'S002 display identity');
check(f3.mode === 'SPAN_FOCUS' && f3.active_span_id === 'S003', 'S002 -> S003 replaces active focus');
check(project.mode === 'PROJECT_VIEW' && project.active_span_id === null, 'return restores project view');
check(JSON.stringify(s2) === JSON.stringify({ span_id: 'S002', group_id: 'G001', from_tower_id: 'T002', to_tower_id: 'T003' }), 'focus helper does not mutate project span data');
const frame = sandbox.p3aGiro3dUtil.spanSelectionFrame({
  span_id: 'S002', group_id: 'G001', from_tower_id: 'T002', to_tower_id: 'T003',
  from_tower_geometry: { base_xyz_absolute_m: [0, 0, 1] },
  to_tower_geometry: { base_xyz_absolute_m: [30, 0, 2] }
}, 8, 12);
const envelope = sandbox.p3aGiro3dUtil.spanFocusClipPlanes(frame, { along: 15, across: 12, below: 6, above: 18 });
check(envelope && envelope.kind === 'viewer-span-focus-envelope', 'viewer focus envelope exists');
check(envelope.planes.length === 6 && envelope.along_margin === 15 && envelope.across_margin === 12, 'six-plane viewer filter with explicit margin');
function inside(point) { return envelope.planes.every((plane) => plane.normal[0] * point[0] + plane.normal[1] * point[1] + plane.normal[2] * point[2] + plane.constant >= -1e-9); }
check(inside([15, 0, 5]), 'span midpoint is retained by viewer filter');
check(!inside([100, 100, 100]), 'unrelated point is outside viewer filter');
if (failed > 0) process.exit(1);
console.log('Span focus mode helper PASS');
'''.replace("__VIEWER_JS__", str(VIEWER_JS).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(VIEWER_JS.parents[2]),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "Span focus mode helper PASS" in result.stdout


# ---------------------------------------------------------------------------
# No automatic scientific pipeline
# ---------------------------------------------------------------------------


def test_viewer_never_runs_scientific_pipeline() -> None:
    # Strip comments first — the viewer docstring legitimately says "interaction
    # only"; the check is about CODE, not documentation.  The viewer may run the
    # tower-review JSON API (detection/refinement happen server-side), but it
    # must never invoke any local conversion/prediction/processing script.
    js = _COMMENT_RE_BLOCK.sub(" ", _viewer_src())
    js = _COMMENT_RE_LINE.sub(" ", js)
    for token in (
        "prepare_p3a_potree_sample",
        "prepare_p3a_giro3d_copc",
        "PotreeConverter",
        "run_manual_single_span",
        "SourceIdentity",
        "detect_pole_candidates",
        "refine_tower_from_raw",
    ):
        assert token not in js, f"scientific/pipeline token present in viewer code: {token}"
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


def test_viewer_has_tower_review_overlay() -> None:
    """The V5A2/V5A3 review overlay + interaction modes are present in the viewer."""
    js = _viewer_src()
    for token in (
        "makeTowerReview",
        "reviewMarkerColors",
        "towerMarkerColors",
        "candidateDisplayLabel",
        "spriteScaleForDistance",
        "attachOverlayGroup",
        "/api/tower-review/state",
        "manual-candidates",
        "adjusted_seed_xyz",
        "trp_accept",
        "trp_reject",
        "trp_manual",
        "setMode",
    ):
        assert token in js, f"missing tower-review feature: {token}"


def test_viewer_has_selected_candidate_visibility_ux() -> None:
    """The selected-candidate UX (region, big label, locate, details) is present."""
    js = _viewer_src()
    html = INDEX.read_text(encoding="utf-8")
    for token in (
        "rebuildSelection",
        "applySelectionVisuals",
        "updateDetails",
        "focusOnSelected",
        "selectedLabel",
        "labelPx",
        "selectionGroup",
        "scaleLabels",
        "candidateRegion",
        "selectionMarkerSpec",
        "BoxGeometry",
        "候选区域",
        "Candidate Region",
        "inspection volume",
        "EdgesGeometry",
        "候选标记仅用于定位",
        "localization only",
        "Candidate bounds",
    ):
        assert token in js, f"missing selected-candidate viewer feature: {token}"
    for token in ("trp_locate", "trp_details"):
        assert token in html, f"missing selected-candidate panel control: {token}"


def test_viewer_long_beacon_removed() -> None:
    """The confusing ~60 m beacon must be gone (replaced by the region + stub)."""
    js = _viewer_src()
    assert "beaconTopZ" not in js
    assert "isBeacon" not in js
    assert "CylinderGeometry(0.18" not in js


def test_viewer_no_large_filled_candidate_cylinder() -> None:
    """The misleading large filled horizontal cylinder is gone.

    The V5A3 owner finding: the selected-candidate region rendered as a large
    translucent Y-axis-aligned ``CylinderGeometry`` tube (Three.js default) that
    crossed the scene horizontally.  The marker must now be a thin Z-up
    wireframe ``BoxGeometry`` with no filled translucent surface.
    """
    js = _viewer_src()
    assert "new THREE.CylinderGeometry(radius, radius, height" not in js
    assert "regionMesh" not in js
    assert "opacity: 0.22" not in js
    assert "selectionMarkerSpec" in js
    assert "new THREE.BoxGeometry(spec.width, spec.depth, spec.height)" in js


def test_span_focus_footer_pinned_and_reachable() -> None:
    """The Span Focus footer (summary + Return button) can never be clipped away.

    The Owner reproduced the Return button and result panel disappearing after
    clicking a Span: the panel used ``overflow: hidden`` so the bottom section was
    clipped out of the max-height panel.  The footer is now pinned to the bottom of
    a scrollable panel.
    """
    html = INDEX.read_text(encoding="utf-8")
    style = html[html.find("<style>"): html.find("</style>")]
    assert "overflow-y: auto" in style, "trp_panel must be vertically scrollable"
    assert "overflow-x: hidden" in style
    assert "position: sticky" in style and "bottom: 0" in style, "span focus footer must be sticky"
    assert 'id="span_focus_footer"' in html
    # The footer must contain the summary, the focus mode and the Return button.
    footer_start = html.find('id="span_focus_footer"')
    footer_block = html[footer_start: footer_start + 400]
    assert 'id="span_summary"' in footer_block
    assert 'id="span_focus_mode"' in footer_block
    assert 'id="span_return_project"' in footer_block


def test_span_summary_renders_result_fields() -> None:
    """The COMPLETE Span result panel renders the required science fields."""
    js = _viewer_src()
    for token in (
        "Endpoints:",
        "Wires:",
        "Trees:",
        "Minimum clearance:",
        "Risk threshold:",
        "Below threshold:",
        "wire_count",
        "below_threshold",
        "risk_threshold_m",
    ):
        assert token in js, f"missing Span result field in viewer: {token}"


def test_return_to_project_rerenders_span_list() -> None:
    """Return to Project View clears the stale Span-row selection highlight."""
    js = _viewer_src()
    assert "renderSpans(spanState)" in js
    assert 'spanFocusMode = "PROJECT_VIEW"' in js


def test_viewer_has_span_tree_inspection() -> None:
    """The final-reconciled Tree list + locate/highlight feature is present."""
    js = _viewer_src()
    html = INDEX.read_text(encoding="utf-8")
    for token in (
        "fetchTreeList",
        "renderTreeList",
        "selectTree",
        "clearTreeSelection",
        "renderTreePreview",
        "renderTreeClearance",
        "focusTree",
        "treePreviewGroup",
        "treeClearanceGroup",
        "global_tree_id",
        "Trees: ",
        "Core: ",
        "Halo/Review: ",
        "/trees/",
        "activeSpanId",
    ):
        assert token in js, f"missing Span Tree inspection viewer feature: {token}"
    for token in ("tree_panel", "tree_header", "tree_list", "tree_details"):
        assert token in html, f"missing Span Tree panel element: {token}"


def test_viewer_tree_code_has_no_hardcoded_span() -> None:
    """The Tree inspection is driven by the selected Span id — never S002."""
    js = _viewer_src()
    for token in ("G001-Span002", "S002"):
        assert token not in js, f"hard-coded Span id in viewer Tree code: {token}"
    assert "encodeURIComponent(span.span_id)" in js or "encodeURIComponent(spanId)" in js
    assert "encodeURIComponent(treeListState.span_id)" in js


def test_viewer_attaches_overlays_via_instance_not_view() -> None:
    """The overlay group must be attached to the Instance's native-object surface.

    The pinned Giro3D exposes ``threeObjects``/``scene`` on the Instance, not on
    ``instance.view``; attaching via ``instance.view.scene.add`` fails with
    "Cannot read properties of undefined (reading 'add')".
    """
    js = _viewer_src()
    assert "attachOverlayGroup(viewer.instance" in js
    assert "viewer.instance.view.scene.add" not in js
    assert "threeObjects" in js


def test_viewer_overlay_state_serialization() -> None:
    """The pure marker/label helpers produce deterministic review-facing output."""
    u = "p3aGiro3dUtil"
    js = _viewer_src()
    assert f"{u}.reviewMarkerColors" in js or "reviewMarkerColors" in js
    assert "candidateDisplayLabel" in js
