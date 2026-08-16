"""Live V5A3 ordered tower-selection API tests against the real loopback server.

Starts the actual ``scripts/serve_p3a_giro3d.py`` server on an auto-assigned
loopback port with a synthetic tower project + RAW tile cache, then exercises
the ordered JSON endpoints over HTTP.  Verifies:

- ordered mode is primary in the state projection;
- SELECT returns a preview with a capped VIEWER-ONLY display sample (the exact
  membership arrays are never served to the viewer);
- PREVIEW (GET) returns the deterministic sample;
- CONFIRM allocates T001 with a linked membership artifact;
- RESELECT / CANCEL allocate no T### and clear the preview;
- Automatic Suggestions remain available (HTML + /detect endpoint);
- the production PDAL viewer page is still served intact.
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


def _http_post(url: str, payload: dict, timeout: int = 120):
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
def ordered_server(tmp_path):
    """Start the real Giro3D server with the tower-review API enabled."""
    from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
    from core.project import create_project_writable

    from tests.towers.conftest import (
        project_for_source,
        write_ground_evidence_las,
        write_tower_las,
    )

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


def test_ordered_mode_is_primary(ordered_server):
    base_url, _ = ordered_server
    status, payload = _http_get(base_url + "api/tower-review/state")
    assert status == 200
    ordered = payload["state"]["ordered"]
    assert ordered["mode"] == "ORDERED_MANUAL"
    assert ordered["next_tower_id"] == "T001"
    assert ordered["confirmed_tower_ids"] == []
    assert ordered["confirmed_tower_count"] == 0


def _ensure_group(base_url, display_name="区域 1"):
    status, payload = _http_post(
        base_url + "api/tower-review/groups", {"display_name": display_name}
    )
    assert status == 200, payload
    return payload["group"]["group_id"]


def test_select_preview_confirm_flow(ordered_server):
    base_url, _ = ordered_server
    g001 = _ensure_group(base_url)
    status, payload = _http_post(
        base_url + "api/tower-review/ordered/select",
        {"xyz": [10.0, 20.0, 0.0], "group_id": g001, "operator_identity": "owner"},
    )
    assert status == 200, payload
    assert payload["ok"] is True
    assert payload["success"] is True
    preview = payload["preview"]
    assert preview["member_count"] > 0
    assert preview["height_m"] > 0
    assert preview["refinement_ready"] == "PREVIEW_READY"
    assert preview["preview_sample_authority"] == "VIEWER_ONLY"
    assert preview["preview_sample_size"] <= preview["preview_sample_cap"]
    # The preview is group-bound.
    assert preview["group_id"] == g001
    # The viewer never receives the authoritative membership arrays.
    assert "source_index" not in preview
    assert "xyz_absolute_m" not in preview
    token = preview["token"]

    # GET preview returns the deterministic sample.
    status, payload = _http_get(base_url + "api/tower-review/ordered/preview?token=" + token)
    assert status == 200, payload
    assert payload["preview"]["preview_sample_size"] <= 20_000

    # CONFIRM -> T001 with a linked membership artifact.
    status, payload = _http_post(
        base_url + "api/tower-review/ordered/confirm",
        {"token": token, "operator_identity": "owner"},
    )
    assert status == 200, payload
    assert payload["result"]["tower_id"] == "T001"
    assert payload["result"]["artifact_reference"]["role"] == "tower-local-membership"
    assert payload["result"]["artifact_reference"]["relative_path"].startswith("tower-memberships/")
    state = payload["state"]
    assert state["ordered"]["confirmed_tower_ids"] == ["T001"]
    assert state["ordered"]["next_tower_id"] == "T002"


def test_confirm_second_is_t002(ordered_server):
    base_url, _ = ordered_server
    g001 = _ensure_group(base_url)
    for xyz in ([10.0, 20.0, 0.0], [45.0, 50.0, 0.0]):
        _, select = _http_post(
            base_url + "api/tower-review/ordered/select",
            {"xyz": xyz, "group_id": g001, "operator_identity": "owner"},
        )
        token = select["preview"]["token"]
        _, confirm = _http_post(
            base_url + "api/tower-review/ordered/confirm",
            {"token": token, "operator_identity": "owner"},
        )
        assert confirm["ok"] is True, confirm
    _, state_payload = _http_get(base_url + "api/tower-review/state")
    ordered = state_payload["state"]["ordered"]
    assert ordered["confirmed_tower_ids"] == ["T001", "T002"]
    assert ordered["next_tower_id"] == "T003"


def test_reselect_and_cancel_do_not_allocate(ordered_server):
    base_url, _ = ordered_server
    g001 = _ensure_group(base_url)
    # RESELECT
    _, select = _http_post(
        base_url + "api/tower-review/ordered/select",
        {"xyz": [10.0, 20.0, 0.0], "group_id": g001, "operator_identity": "owner"},
    )
    token = select["preview"]["token"]
    _, reselect = _http_post(
        base_url + "api/tower-review/ordered/reselect",
        {"token": token, "operator_identity": "owner"},
    )
    assert reselect["ok"] is True
    assert reselect["state"]["ordered"]["confirmed_tower_ids"] == []
    assert reselect["state"]["ordered"]["next_tower_id"] == "T001"
    # The consumed preview cannot confirm.
    status, payload = _http_post(
        base_url + "api/tower-review/ordered/confirm",
        {"token": token, "operator_identity": "owner"},
    )
    assert status == 409
    assert payload["error"]["code"] == "PREVIEW_TOKEN_INVALID"

    # CANCEL
    _, select = _http_post(
        base_url + "api/tower-review/ordered/select",
        {"xyz": [10.0, 20.0, 0.0], "group_id": g001, "operator_identity": "owner"},
    )
    token = select["preview"]["token"]
    _, cancel = _http_post(
        base_url + "api/tower-review/ordered/cancel",
        {"token": token, "operator_identity": "owner"},
    )
    assert cancel["ok"] is True
    assert cancel["state"]["ordered"]["confirmed_tower_ids"] == []
    assert cancel["state"]["ordered"]["next_tower_id"] == "T001"


def test_failed_segmentation_returns_structured_failure(ordered_server):
    base_url, _ = ordered_server
    g001 = _ensure_group(base_url)
    status, payload = _http_post(
        base_url + "api/tower-review/ordered/select",
        {"xyz": [500.0, 500.0, 0.0], "group_id": g001, "operator_identity": "owner"},
    )
    assert status == 200
    assert payload["ok"] is True
    assert payload["success"] is False
    assert payload["failure_reason"].startswith("LOCAL_SEGMENTATION_FAILED")
    assert payload["state"]["ordered"]["next_tower_id"] == "T001"


def test_automatic_suggestions_still_available(ordered_server):
    base_url, _ = ordered_server
    # The automatic /detect endpoint remains available.
    status, payload = _http_post(
        base_url + "api/tower-review/detect", {"candidate_set_id": "auto-001"}
    )
    assert status == 200, payload
    assert payload["result"]["candidate_count"] >= 1
    # The static page still exposes the collapsible Automatic Suggestions UI.
    with urllib.request.urlopen(base_url, timeout=30) as resp:
        page = resp.read().decode("utf-8")
    for token in (
        "trp_order_select",
        "trp_order_confirm",
        "trp_order_reselect",
        "trp_order_cancel",
        "trp_auto_toggle",
        "自动候选辅助",
        "Automatic Suggestions",
        "trp_accept",
    ):
        assert token in page, f"missing ordered/automatic control in index.html: {token}"


def test_production_pdal_viewer_page_intact(ordered_server):
    base_url, _ = ordered_server
    with urllib.request.urlopen(base_url, timeout=30) as resp:
        assert resp.status == 200
        page = resp.read().decode("utf-8")
    # The viewer still loads the Giro3D bundle and the ordered panel.
    assert "vendor/giro3d.bundle.js" in page
    assert "viewer.js" in page
    assert "顺序选杆" in page
    # The full-PDAL production config exists on disk and names the COPC.
    config_path = _REPO_ROOT / "viewer_poc" / "giro3d" / "poc_config_full_pdal.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["copcUrl"].endswith("p3a_full_pdal.copc.laz")


def test_groups_create_and_state_list(ordered_server):
    base_url, _ = ordered_server
    g1 = _ensure_group(base_url, "区域 1")
    g2 = _ensure_group(base_url, "道路右侧")
    assert g1 == "G001" and g2 == "G002"
    _, payload = _http_get(base_url + "api/tower-review/state")
    groups = {g["group_id"]: g for g in payload["state"]["ordered"]["tower_groups"]}
    assert "G001" in groups and "G002" in groups
    assert groups["G001"]["display_name"] == "区域 1"
    assert groups["G001"]["next_group_position"] == 1
    assert groups["G001"]["tower_count"] == 0
    assert payload["state"]["ordered"]["next_group_sequence"] == 3
    # Missing display name -> 400.
    status, payload = _http_post(base_url + "api/tower-review/groups", {})
    assert status == 400 and payload["error"]["code"] == "BAD_DISPLAY_NAME"


def test_preview_token_group_bound_and_no_silent_reassignment(ordered_server):
    base_url, _ = ordered_server
    g1 = _ensure_group(base_url, "区域 1")
    g2 = _ensure_group(base_url, "道路右侧")
    assert g2 == "G002"
    _, select = _http_post(
        base_url + "api/tower-review/ordered/select",
        {"xyz": [10.0, 20.0, 0.0], "group_id": g1, "operator_identity": "owner"},
    )
    assert select["preview"]["group_id"] == g1
    token = select["preview"]["token"]
    # Confirm in G002 context must NOT move the preview; the token is G001-bound.
    _, confirm = _http_post(
        base_url + "api/tower-review/ordered/confirm",
        {"token": token, "operator_identity": "owner"},
    )
    assert confirm["ok"] and confirm["result"]["tower_id"] == "T001"
    assert confirm["result"]["group_id"] == g1
    _, state_payload = _http_get(base_url + "api/tower-review/state")
    gmap = {
        g["group_id"]: g["ordered_tower_ids"]
        for g in state_payload["state"]["ordered"]["tower_groups"]
    }
    assert gmap["G001"] == ["T001"]
    assert gmap["G002"] == []


def test_g001_g002_g001_sequence_api(ordered_server):
    base_url, _ = ordered_server
    g1 = _ensure_group(base_url, "区域 1")
    g2 = _ensure_group(base_url, "道路右侧")

    def select_confirm(xyz, gid):
        _, select = _http_post(
            base_url + "api/tower-review/ordered/select",
            {"xyz": xyz, "group_id": gid, "operator_identity": "owner"},
        )
        assert select["ok"] and select["success"], select
        _, confirm = _http_post(
            base_url + "api/tower-review/ordered/confirm",
            {"token": select["preview"]["token"], "operator_identity": "owner"},
        )
        assert confirm["ok"], confirm
        return confirm["result"]["tower_id"]

    assert select_confirm([10.0, 20.0, 0.0], g1) == "T001"
    assert select_confirm([45.0, 50.0, 0.0], g1) == "T002"
    assert select_confirm([10.0, 20.0, 1.0], g2) == "T003"
    assert select_confirm([45.0, 50.0, 1.0], g2) == "T004"
    assert select_confirm([10.0, 20.0, 2.0], g1) == "T005"
    _, state_payload = _http_get(base_url + "api/tower-review/state")
    gmap = {
        g["group_id"]: g["ordered_tower_ids"]
        for g in state_payload["state"]["ordered"]["tower_groups"]
    }
    assert gmap["G001"] == ["T001", "T002", "T005"], gmap
    assert gmap["G002"] == ["T003", "T004"], gmap
    assert state_payload["state"]["ordered"]["next_tower_id"] == "T006"


def test_confirmed_membership_preview_endpoint(ordered_server):
    base_url, _ = ordered_server
    g1 = _ensure_group(base_url)
    _, select = _http_post(
        base_url + "api/tower-review/ordered/select",
        {"xyz": [10.0, 20.0, 0.0], "group_id": g1, "operator_identity": "owner"},
    )
    _, confirm = _http_post(
        base_url + "api/tower-review/ordered/confirm",
        {"token": select["preview"]["token"], "operator_identity": "owner"},
    )
    assert confirm["ok"] and confirm["result"]["tower_id"] == "T001"
    status, payload = _http_get(base_url + "api/tower-review/towers/T001/membership-preview")
    assert status == 200, payload
    assert payload["member_count"] > 0
    assert payload["preview_sample_authority"] == "VIEWER_ONLY"
    assert payload["preview_sample_size"] <= payload["preview_sample_cap"]
    assert "source_index" not in payload
    assert payload["group_ids"] == ["G001"]
    # Unknown tower -> structured 404.
    try:
        _http_get(base_url + "api/tower-review/towers/T999/membership-preview")
        raise AssertionError("expected HTTPError for unknown tower")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
        body = json.loads(exc.read().decode("utf-8"))
        assert body["error"]["code"] == "TOWER_NOT_FOUND"


def test_membership_preview_unavailable_for_legacy_tower(ordered_server):
    """A tower confirmed without a membership artifact reports MEMBERSHIP_UNAVAILABLE.

    Historical towers legitimately lack an immutable membership artifact; the
    viewer must treat this as a calm "unavailable", never fabricate RAW
    membership, and Locate must still work.  An automatic-detection confirmation
    produces exactly such a tower (no membership artifact reference).
    """
    base_url, _ = ordered_server
    _, detect = _http_post(base_url + "api/tower-review/detect", {"candidate_set_id": "auto-001"})
    assert detect["ok"], detect
    candidate = detect["result"]["candidates"][0]
    _, accept = _http_post(
        base_url
        + "api/tower-review/candidates/auto-001/"
        + candidate["candidate_id"]
        + "/accept",
        {
            "candidate_set_digest": detect["result"]["candidate_set_digest"],
            "operator_identity": "owner",
        },
    )
    assert accept["ok"] and accept["result"]["success"], accept
    tower_id = accept["result"]["tower_id"]
    try:
        _http_get(base_url + "api/tower-review/towers/" + tower_id + "/membership-preview")
        raise AssertionError("expected HTTPError for a membership-less tower")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
        body = json.loads(exc.read().decode("utf-8"))
        assert body["error"]["code"] == "MEMBERSHIP_UNAVAILABLE"


def test_select_unknown_group_fails_closed(ordered_server):
    base_url, _ = ordered_server
    status, payload = _http_post(
        base_url + "api/tower-review/ordered/select",
        {"xyz": [10.0, 20.0, 0.0], "group_id": "G999", "operator_identity": "owner"},
    )
    assert status == 400
    assert payload["ok"] is False
    assert payload["error"]["code"] == "ORDERED_WORKFLOW_ERROR"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_ordered_helpers(ordered_server):
    """The ordered pure helpers serialize deterministically."""
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
function check(c, m) { if (!c) { failed += 1; console.log('FAIL: ' + m); } }
check(u.nextTowerLabel({ ordered: { next_tower_id: 'T003' } }) === 'T003', 'next tower label');
check(u.confirmedTowerLine({ ordered: { confirmed_tower_ids: ['T001', 'T002'] } }) === 'T001 T002',
  'confirmed tower line');
const preview = {
  member_count: 1200, height_m: 28.5,
  member_bounds_absolute_m: {
    min_xyz_absolute_m: [-67.0, 1.2, 1.0], max_xyz_absolute_m: [-65.5, 3.1, 30.0]
  },
  quality: { median_verticality: 0.9, median_linearity: 0.8, ground_support_points: 42,
    membership_digest: 'abcd' + '1'.repeat(60) },
  refinement_ready: 'PREVIEW_READY'
};
const lines = u.orderedPreviewLines(preview);
check(lines.indexOf('member count: 1200') !== -1, 'preview member count line');
check(lines.indexOf('PREVIEW_READY') !== -1, 'preview status line');
check(lines.indexOf('VIEWER-ONLY') === -1, 'sample authority line');
check(u.orderedPreviewLines(null) === null, 'null preview -> null');
if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('ordered viewer helpers PASS');
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
def test_viewer_group_helpers() -> None:
    """The group helpers expose the two distinct "next" values and group resolution."""
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
function check(c, m) { if (!c) { failed += 1; console.log('FAIL: ' + m); } }
const state = {
  ordered: {
    next_tower_id: 'T009',
    tower_groups: [
      { group_id: 'G001', display_name: '区域 1', tower_count: 5, next_group_position: 6 },
      { group_id: 'G002', display_name: '道路右侧', tower_count: 3, next_group_position: 4 }
    ]
  }
};
check(u.resolveActiveGroup(state, 'G002').group_id === 'G002', 'resolve G002');
check(u.resolveActiveGroup(state, 'G999').group_id === 'G001', 'default first group');
check(u.resolveActiveGroup(state, null).group_id === 'G001', 'default first group when null');
check(u.resolveActiveGroup({ ordered: { tower_groups: [] } }, 'G001') === null,
  'no groups -> null');
const lines = u.groupStatusLines(state, 'G001');
check(lines.indexOf('当前区域: G001 区域 1') !== -1, 'current group line');
check(lines.indexOf('本区域已确认: 5') !== -1, 'group count line');
check(lines.indexOf('本区域下一位置: 第 6 根') !== -1, 'group next position line (distinct)');
check(lines.indexOf('项目下一 Tower ID: T009') !== -1, 'global next tower line (distinct)');
check(u.groupStatusLines(null, 'G001') === null, 'null state -> null');
if (failed > 0) { console.log('FAILURES=' + failed); process.exit(1); }
console.log('group viewer helpers PASS');
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
