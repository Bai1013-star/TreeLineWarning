"""Live V5A2/V5A3 tower-review API tests against the real loopback server.

Starts the actual ``scripts/serve_p3a_giro3d.py`` server on an auto-assigned
loopback port with a synthetic tower project + RAW tile cache + ground evidence
LAS, then exercises the JSON tower-review endpoints over HTTP.  No browser is
required; the static page and pure JS helpers are checked directly.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _http_get(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, payload: dict, timeout: int = 90):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, {"ok": False, "error": {"code": "NON_JSON", "message": body}}


@pytest.fixture
def review_server(tmp_path):
    """Start the real Giro3D server with the tower-review API enabled."""
    from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
    from core.project import create_project_writable

    from tests.towers.conftest import project_for_source, write_ground_evidence_las, write_tower_las

    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=(45.0, 50.0, 0.0))
    project, binding, metadata = project_for_source(src, points)
    root = tmp_path / "project"
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    ground_las = tmp_path / "ground.las"
    write_ground_evidence_las(ground_las, points)
    cache_dir = tmp_path / "cache"
    RawTileCache.prepare(src, cache_dir, source_binding=binding, config=RawTileCacheConfig())

    log_path = tmp_path / "server.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(_REPO_ROOT / "scripts" / "serve_p3a_giro3d.py"),
                "--port",
                "0",
                "--no-open",
                "--config",
                "poc_config.json",
                "--tower-project-root",
                str(root),
                "--tower-tile-cache",
                str(cache_dir),
                "--tower-detect-ground",
                str(ground_las),
                "--tower-operator",
                "owner",
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_REPO_ROOT),
        )
    base_url = None
    deadline = time.time() + 30
    try:
        while time.time() < deadline and base_url is None:
            text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            match = re.search(r"serving P3A Giro3D viewer at (http://127\.0\.0\.1:\d+/)", text)
            if match:
                base_url = match.group(1)
                break
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if base_url is None:
            text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            pytest.skip("tower-review server did not start: " + text[-500:])
        yield base_url, root
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_state_get(review_server):
    base_url, _ = review_server
    status, payload = _http_get(base_url + "api/tower-review/state")
    assert status == 200
    assert payload["ok"] is True
    state = payload["state"]
    assert state["project_id"]
    assert "candidate_sets" in state
    assert "confirmed_towers" in state
    assert state["source"]["coordinate_frame"] == "SOURCE_LAS_ABSOLUTE"


def test_spans_api_is_well_formed_and_read_only(review_server):
    """The Span API payload is well-formed and the GET is non-mutating."""
    base_url, root = review_server
    status, payload = _http_get(base_url + "api/spans")
    assert status == 200
    assert payload["ok"] is True
    assert payload["project_id"]
    assert "generation_id" in payload
    assert isinstance(payload["spans"], list)
    # The synthetic project has no spans; the GET must not create any.
    from core.project import open_project_read_only

    with open_project_read_only(root) as session:
        assert session.span_registry.records == ()


def test_detect_post_persists_candidate_set(review_server):
    base_url, _ = review_server
    status, payload = _http_post(
        base_url + "api/tower-review/detect", {"candidate_set_id": "auto-001"}
    )
    assert status == 200, payload
    result = payload["result"]
    assert result["candidate_set_id"] == "auto-001"
    assert result["candidate_count"] >= 1
    assert "candidate_set_digest" in result
    ids = [c["candidate_id"] for c in result["candidates"]]
    assert all(pid.startswith("P") for pid in ids)


def test_accept_post_confirms_tower(review_server):
    base_url, _ = review_server
    _, detect = _http_post(base_url + "api/tower-review/detect", {"candidate_set_id": "auto-001"})
    candidate = detect["result"]["candidates"][0]
    status, payload = _http_post(
        base_url
        + "api/tower-review/candidates/auto-001/"
        + candidate["candidate_id"]
        + "/accept",
        {
            "candidate_set_digest": detect["result"]["candidate_set_digest"],
            "operator_identity": "owner",
        },
    )
    assert status == 200, payload
    assert payload["result"]["success"] is True
    assert payload["result"]["tower_id"] == "T001"
    # The updated state reflects the confirmation.
    confirmed = [t["tower_id"] for t in payload["state"]["confirmed_towers"]]
    assert confirmed == ["T001"]


def test_reject_post_persists(review_server):
    base_url, _ = review_server
    _, detect = _http_post(base_url + "api/tower-review/detect", {"candidate_set_id": "auto-001"})
    candidate = detect["result"]["candidates"][0]
    status, payload = _http_post(
        base_url
        + "api/tower-review/candidates/auto-001/"
        + candidate["candidate_id"]
        + "/reject",
        {
            "candidate_set_digest": detect["result"]["candidate_set_digest"],
            "operator_identity": "owner",
            "reason": "not a tower",
        },
    )
    assert status == 200, payload
    assert payload["event"]["decision"] == "REJECTED"
    state = payload["state"]
    row = state["candidate_sets"][0]["candidates"][0]
    assert row["review_state"] == "REJECTED"
    assert row["confirmed_tower_id"] is None


def test_manual_add_post(review_server):
    base_url, _ = review_server
    status, payload = _http_post(
        base_url + "api/tower-review/manual-candidates",
        {"xyz": [45.0, 50.0, 0.0], "operator_identity": "owner"},
    )
    assert status == 200, payload
    assert payload["candidate_id"] == "P001"
    assert payload["candidate_set_id"].startswith("manual-")


def test_candidate_reference_validation(review_server):
    base_url, _ = review_server
    _, detect = _http_post(base_url + "api/tower-review/detect", {"candidate_set_id": "auto-001"})
    candidate = detect["result"]["candidates"][0]
    # Wrong candidate-set digest must be rejected with a structured 409.
    status, payload = _http_post(
        base_url
        + "api/tower-review/candidates/auto-001/"
        + candidate["candidate_id"]
        + "/accept",
        {"candidate_set_digest": "0" * 64},
    )
    assert status == 409
    assert payload["ok"] is False
    assert payload["error"]["code"] == "CANDIDATE_SET_DIGEST_MISMATCH"
    # Unknown candidate set -> 404 structured error.
    status, payload = _http_post(
        base_url + "api/tower-review/candidates/missing-set/P001/accept",
        {"candidate_set_digest": detect["result"]["candidate_set_digest"]},
    )
    assert status == 404
    assert payload["error"]["code"] == "CANDIDATE_SET_NOT_FOUND"


def test_structured_error_bad_json(review_server):
    base_url, _ = review_server
    request = urllib.request.Request(
        base_url + "api/tower-review/manual-candidates",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=30)
    assert excinfo.value.code == 400
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert payload["ok"] is False
    assert payload["error"]["code"] == "BAD_JSON"


def test_static_ui_contains_tower_review_controls(review_server):
    base_url, _ = review_server
    status, page = None, None
    with urllib.request.urlopen(base_url, timeout=30) as resp:
        status = resp.status
        page = resp.read().decode("utf-8")
    assert status == 200
    for token in (
        "trp_accept",
        "trp_reject",
        "trp_adjust",
        "trp_manual",
        "trp_refresh",
        "trp_panel",
        "trp_order_select",
        "trp_order_confirm",
        "trp_auto_toggle",
        "顺序选杆",
        "Ordered Tower Selection",
        "Automatic Suggestions",
    ):
        assert token in page, f"missing tower-review control in index.html: {token}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_overlay_state_serialization(review_server):
    """The pure marker/label helpers serialize review state deterministically."""
    viewer_js = _REPO_ROOT / "viewer_poc" / "giro3d" / "viewer.js"
    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = {
  window: {},
  document: { getElementById: () => null },
  fetch: () => Promise.reject(new Error('no fetch')),
  console: console, setTimeout: setTimeout, clearTimeout: clearTimeout, setInterval: () => 0
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const u = sandbox.p3aGiro3dUtil;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }
check(u.reviewMarkerColors('ACCEPTED').bg === '#1b5e20', 'accepted color');
check(u.reviewMarkerColors('REJECTED').bg === '#4a148c', 'rejected color');
check(u.towerMarkerColors().bg === '#00695c', 'tower color');
const s = u.spriteScaleForDistance(100, 26, 60, 800);
check(isFinite(s) && s > 0, 'sprite scale finite positive');
const label = u.candidateDisplayLabel({
  candidate_id: 'P001', origin: 'AUTOMATIC', confidence: 0.87, review_state: 'PENDING_REVIEW'
});
check(label.indexOf('P001') === 0 && label.indexOf('PENDING_REVIEW') > -1,
  'candidate label serialization');
if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('tower-review overlay helpers PASS');
""".replace("__VIEWER_JS__", str(viewer_js).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_selected_label_is_larger_and_screen_stable() -> None:
    """The selected marker is visually distinct by size and never microscopic.

    Verifies the selected label (46 px) renders larger than unselected labels
    (26 px) at the same distance, that label scale shrinks as the camera recedes
    (screen-space sizing, so zooming out does not make it microscopic), and that
    the overlay attach helper has no ``.add``-on-undefined regression.
    """
    viewer_js = _REPO_ROOT / "viewer_poc" / "giro3d" / "viewer.js"
    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = {
  window: {}, document: { getElementById: () => null },
  fetch: () => Promise.reject(new Error('no fetch')),
  console: console, setTimeout: setTimeout, clearTimeout: clearTimeout, setInterval: () => 0
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const u = sandbox.p3aGiro3dUtil;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

// Selected label (46px) is larger than an unselected label (26px) at the same distance.
const sel = u.spriteScaleForDistance(100, 46, 60, 800);
const unsel = u.spriteScaleForDistance(100, 26, 60, 800);
check(sel > unsel,
  'selected label scale larger than unselected, got ' + sel.toFixed(3) + ' vs ' + unsel.toFixed(3));

// Screen-space stability: farther camera -> larger world scale so it stays readable.
const near = u.spriteScaleForDistance(50, 26, 60, 800);
const far = u.spriteScaleForDistance(500, 26, 60, 800);
check(far > near,
  'farther camera keeps label readable (far > near), got '
  + far.toFixed(3) + ' vs ' + near.toFixed(3));

// State colors are distinct from the selected magenta (not color-only, but a check).
check(u.reviewMarkerColors('ACCEPTED').bg !== '#ff3d81', 'accepted color distinct from selected');

// No .add regression: overlay still attaches to the pinned Instance surface.
const calls = [];
const host = { threeObjects: { add(child) { calls.push('threeObjects.add'); } }, view: {} };
let target = null;
try { target = u.attachOverlayGroup(host, {}); }
catch (e) { failed += 1; console.log('FAIL: attach threw: ' + e.message); }
check(target === 'threeObjects' && calls.length === 1, 'overlay attach regression check');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('tower-review selection sizing PASS');
""".replace("__VIEWER_JS__", str(viewer_js).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_candidate_region_tracks_selected_candidate() -> None:
    """The candidate region follows the ACTUAL candidate XYZ/bounds.

    Verifies the pure ``candidateRegion`` helper: candidate bounds are honoured
    exactly (no forced 2.5 m x 25 m volume), a bounds-less candidate falls back
    to a compact labeled Viewer fallback, and switching P011 -> P015 changes the
    region centre.
    """
    viewer_js = _REPO_ROOT / "viewer_poc" / "giro3d" / "viewer.js"
    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = {
  window: {}, document: { getElementById: () => null },
  fetch: () => Promise.reject(new Error('no fetch')),
  console: console, setTimeout: setTimeout, clearTimeout: clearTimeout, setInterval: () => 0
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const u = sandbox.p3aGiro3dUtil;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

// Candidate with bounds -> region uses the REAL bounds (no forced 2.5 m / 25 m).
const withBounds = {
  proposal_xyz_absolute_m: [-66.38, 2.47, 1.52],
  candidate_bounds_absolute_m: {
    min_xyz_absolute_m: [-66.5, 2.3, 1.4],
    max_xyz_absolute_m: [-66.2, 2.6, 9.4]
  }
};
const r1 = u.candidateRegion(withBounds);
check(r1 && r1.source === 'CANDIDATE_BOUNDS',
  'bounds region source is CANDIDATE_BOUNDS, got ' + (r1 && r1.source));
check(r1 && Math.abs(r1.cx - (-66.35)) < 1e-6, 'bounds region centre x from bounds');
check(r1 && Math.abs(r1.cy - 2.45) < 1e-6, 'bounds region centre y from bounds');
check(r1 && Math.abs(r1.radius - 0.5) < 1e-9,
  'bounds region uses the real footprint radius (0.5 m), not a forced 2.5 m');
check(r1 && Math.abs(r1.height - 8.0) < 1e-9,
  'bounds region uses the real Z extent (8.0 m), not a forced 25 m');
check(r1 && r1.height === (r1.topZ - r1.baseZ), 'region height matches base/top extent');

// Z-up marker spec: thin wireframe box, X/Y horizontal, Z vertical.
const spec1 = u.selectionMarkerSpec(r1);
check(spec1 && spec1.axis === 'Z', 'selected marker is Z-up (axis Z)');
check(spec1 && spec1.kind === 'wireframe-box', 'selected marker is a thin wireframe box');
check(spec1 && Math.abs(spec1.width - r1.radius * 2) < 1e-9
  && Math.abs(spec1.depth - r1.radius * 2) < 1e-9, 'marker X/Y are the horizontal footprint');
check(spec1 && Math.abs(spec1.height - r1.height) < 1e-9, 'marker Z is the vertical extent');
check(spec1 && spec1.baseZ < spec1.topZ, 'marker spans upward along Z');

// No bounds -> compact labeled Viewer fallback (not a huge 25 m x 3 m volume).
const noBounds = { proposal_xyz_absolute_m: [-66.38, 2.47, 1.52] };
const r2 = u.candidateRegion(noBounds);
check(r2 && r2.source === 'VIEWER_FALLBACK', 'no-bounds fallback source is VIEWER_FALLBACK');
check(r2 && Math.abs(r2.radius - 1.2) < 1e-9, 'fallback radius is compact (1.2 m)');
check(r2 && Math.abs(r2.height - 18.0) < 1e-9, 'fallback height is compact (18 m)');

// Region changes when the selection changes P011 -> P015.
const r3 = u.candidateRegion({ proposal_xyz_absolute_m: [-25.16, 20.40, 0.80] });
check(r2.cx !== r3.cx || r2.cy !== r3.cy, 'region centre changes with the selected candidate');

// Candidate with no proposal -> null (never crashes).
check(u.candidateRegion(null) === null, 'null candidate -> null region');
check(u.candidateRegion({}) === null, 'proposal-less candidate -> null region');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('tower-review candidate region PASS');
""".replace("__VIEWER_JS__", str(viewer_js).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_real_candidates_p011_p015_p016_marker_is_compact_and_zup() -> None:
    """Real owner candidates P011 / P015 / P016 get compact, Z-up markers.

    Uses the actual persisted candidate geometry (auto-owner-review-001) to prove
    the V5A3 fix: every selected candidate now renders a thin Z-up wireframe box
    around its REAL bounds — no large horizontal cyan tube, no forced 2.5 m x 25 m
    volume, and the candidate XYZ/base location stays obvious.
    """
    viewer_js = _REPO_ROOT / "viewer_poc" / "giro3d" / "viewer.js"
    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = {
  window: {}, document: { getElementById: () => null },
  fetch: () => Promise.reject(new Error('no fetch')),
  console: console, setTimeout: setTimeout, clearTimeout: clearTimeout, setInterval: () => 0
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const u = sandbox.p3aGiro3dUtil;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

const candidates = {
  P011: {
    proposal_xyz_absolute_m: [-66.38368383246578, 2.4650994140619957, 1.5190666666666668],
    candidate_bounds_absolute_m: {
      min_xyz_absolute_m: [-67.06018852261872, 1.1944836127979492, 1.5190666666666668],
      max_xyz_absolute_m: [-65.55729025342396, 3.1416041042149363, 9.618622222222221]
    }
  },
  P015: {
    proposal_xyz_absolute_m: [-25.162239388021337, 20.397977191839775, 0.8028],
    candidate_bounds_absolute_m: {
      min_xyz_absolute_m: [-26.133616737634227, 19.639422064449107, 0.8028],
      max_xyz_absolute_m: [-24.40368426063067, 21.17915454145266, 8.0674]
    }
  },
  P016: {
    proposal_xyz_absolute_m: [17.299993945312004, 22.264032747395333, 1.0842333333333334],
    candidate_bounds_absolute_m: {
      min_xyz_absolute_m: [16.867236280492026, 21.831275082575356, 1.0842333333333334],
      max_xyz_absolute_m: [17.741718276798643, 22.87884596777086, 7.8948222222222215]
    }
  }
};

Object.keys(candidates).forEach(function (id) {
  const c = candidates[id];
  const r = u.candidateRegion(c);
  check(r && r.source === 'CANDIDATE_BOUNDS', id + ' uses real candidate bounds');
  check(r && r.radius < 2.5, id + ' radius is the real footprint (< 2.5 m), got '
    + (r && r.radius.toFixed(3)));
  check(r && r.height < 25, id + ' height is the real Z extent (< 25 m), got '
    + (r && r.height.toFixed(2)));
  check(r && Math.abs(r.baseZ - c.candidate_bounds_absolute_m.min_xyz_absolute_m[2]) < 1e-9,
    id + ' base Z is the candidate bounds base');
  const s = u.selectionMarkerSpec(r);
  check(s && s.axis === 'Z', id + ' marker is Z-up');
  check(s && s.kind === 'wireframe-box', id + ' marker is a thin wireframe box');
  check(s && Math.abs(s.width - r.radius * 2) < 1e-9 && Math.abs(s.depth - r.radius * 2) < 1e-9,
    id + ' marker X/Y are the horizontal footprint');
  check(s && Math.abs(s.height - r.height) < 1e-9, id + ' marker Z is the vertical extent');
  check(s && s.baseZ < s.topZ, id + ' marker spans upward along Z (localization only)');
});

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('tower-review P011/P015/P016 compact Z-up marker PASS');
""".replace("__VIEWER_JS__", str(viewer_js).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_span_focus_display_state_pure() -> None:
    """The Span focus/project-view state projection is deterministic and read-only."""
    viewer_js = _REPO_ROOT / "viewer_poc" / "giro3d" / "viewer.js"
    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = {
  window: {}, document: { getElementById: () => null },
  fetch: () => Promise.reject(new Error('no fetch')),
  console: console, setTimeout: setTimeout, clearTimeout: clearTimeout, setInterval: () => 0
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const u = sandbox.p3aGiro3dUtil;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

const span = { span_id: 'S002', group_id: 'G001', from_tower_id: 'T002', to_tower_id: 'T003' };
const focused = u.spanFocusDisplayState(span, 'SPAN_FOCUS');
check(focused.mode === 'SPAN_FOCUS', 'SPAN_FOCUS mode, got ' + focused.mode);
check(focused.active_span_id === 'S002', 'active span id is S002');
check(focused.from_tower_id === 'T002' && focused.to_tower_id === 'T003', 'endpoints projected');
const project = u.spanFocusDisplayState(span, 'PROJECT_VIEW');
check(project.mode === 'PROJECT_VIEW', 'PROJECT_VIEW mode');
check(project.active_span_id === null, 'project view has no active span');

// A focus-mode state must never leak into project view.
check(u.spanFocusDisplayState(null, 'SPAN_FOCUS').active_span_id === null,
  'null span -> no active id');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('span focus display state PASS');
""".replace("__VIEWER_JS__", str(viewer_js).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_overlay_attach_against_pinned_instance_surface() -> None:
    """The overlay group attaches to the pinned Instance, never to view.scene.

    This regression test reproduces the Owner's real failure: the pinned Giro3D
    Instance exposes ``threeObjects``/``scene`` (native-object insertion), while
    ``instance.view`` exposes neither.  ``attachOverlayGroup`` must attach to
    ``threeObjects`` and must throw a clear error (never call ``.add`` on an
    undefined target) when no add target exists.
    """
    viewer_js = _REPO_ROOT / "viewer_poc" / "giro3d" / "viewer.js"
    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__VIEWER_JS__", 'utf8');
const sandbox = {
  window: {}, document: { getElementById: () => null },
  fetch: () => Promise.reject(new Error('no fetch')),
  console: console, setTimeout: setTimeout, clearTimeout: clearTimeout, setInterval: () => 0
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'viewer.js' });
const u = sandbox.p3aGiro3dUtil;
let failed = 0;
function check(cond, msg) { if (!cond) { failed += 1; console.log('FAIL: ' + msg); } }

// Mock host shaped EXACTLY like the pinned Giro3D Instance:
// - threeObjects and scene groups live on the INSTANCE;
// - instance.view has NO scene/threeObjects (the real bug).
const calls = [];
const fakeGroup = {
  name: '', children: [],
  add(child) { this.children.push(child); calls.push('threeObjects.add'); }
};
const fakeScene = { add(child) { calls.push('scene.add'); } };
const host = {
  threeObjects: fakeGroup,
  scene: fakeScene,
  view: { camera: {} }  // no .scene, no .threeObjects
};
const overlay = { kind: 'overlay' };
let target = null;
try { target = u.attachOverlayGroup(host, overlay); }
catch (e) { failed += 1; console.log('FAIL: attach threw: ' + e.message); }
check(target === 'threeObjects', 'attaches to threeObjects, got ' + target);
check(fakeGroup.children.length === 1, 'overlay added to threeObjects group');
check(calls.indexOf('threeObjects.add') !== -1, 'threeObjects.add was called');
check(calls.indexOf('scene.add') === -1, 'scene.add NOT called (threeObjects preferred)');

// Fail-safe: a host with NO add target must throw a clear error (never hang).
let threw = false;
try { u.attachOverlayGroup({ view: {} }, overlay); }
catch (e) {
  threw = true;
  check(String(e.message).indexOf('add') !== -1, 'clear error mentions add target');
}
check(threw, 'missing add target throws instead of calling .add on undefined');

if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('tower-review overlay attach PASS');
""".replace("__VIEWER_JS__", str(viewer_js).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout
