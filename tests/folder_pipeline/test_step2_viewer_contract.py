# ruff: noqa: E501
"""Step-2 viewer payload-contract regressions.

Regression for the blank Step-2 3D viewer.  The browser threw
``Cannot read properties of undefined (reading 'length')`` because the
Step-2 inline JS read the tree 3D point cloud from ``data.trees`` — which the
backend populates with the right-panel tree LIST (``snap["trees"]``) — instead
of ``data.trees3d``.  ``data.trees.xyz`` was therefore ``undefined`` and
``trees.xyz.length`` aborted ``loadEditorData()`` before ``Plotly.newPlot``.

These tests pin the contract on BOTH sides so the class of bug cannot come
back without a test failing:

* backend: ``/api/editor_data`` schema is unambiguous — ``trees`` is the tree
  list, ``trees3d`` is the 3D cloud ``{xyz,color,tree_id_by_index}``, and every
  display layer always carries an ``xyz`` array;
* frontend: the Step-2 inline JS reads ``data.trees3d`` and guards every
  display layer with ``Array.isArray(x.xyz)`` so an empty/missing layer renders
  as an empty trace instead of blanking the whole viewer;
* JS execution (Node, skipped if absent): the real inline script is run against
  a populated and an empty payload and must never throw.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
from algorithms.folder_pipeline.html import step2_html
from algorithms.folder_pipeline.server import FolderServer
from tests.folder_pipeline.test_restart_hydration import _build_completed_workspace


def _extract_step2_js() -> str:
    html = step2_html()
    scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert len(scripts) == 1, "step2_html must contain exactly one inline script"
    return scripts[0]


def _extract_step3_js() -> str:
    from algorithms.folder_pipeline.html import step3_html

    scripts = re.findall(r"<script>([\s\S]*?)</script>", step3_html())
    assert len(scripts) == 1
    return scripts[0]


# ---------------------------------------------------------------------------
# backend payload contract
# ---------------------------------------------------------------------------


def test_editor_data_schema_is_unambiguous(tmp_path):
    """`trees` is the tree list and `trees3d` is the 3D cloud — never confused."""
    workspace_root, workspace = _build_completed_workspace(tmp_path)
    server = FolderServer(workspace)
    payload = server._build_editor_display()
    assert set(payload) == {"trees", "trees3d", "background", "wire", "wires3d", "tower"}

    # `trees` is a LIST of tree dicts (right-panel table) — it must not carry xyz.
    assert isinstance(payload["trees"], list)
    assert len(payload["trees"]) > 0
    for tree in payload["trees"]:
        assert "tree_id" in tree
        assert "xyz" not in tree, "trees (the list) must not be the 3D point cloud"

    # `trees3d` is the 3D point cloud: xyz/color/tree_id_by_index aligned.
    t3d = payload["trees3d"]
    assert set(t3d) == {"xyz", "color", "tree_id_by_index"}
    assert len(t3d["xyz"]) == len(t3d["color"]) == len(t3d["tree_id_by_index"]) > 0
    assert isinstance(t3d["xyz"][0], (list, tuple))

    # Every optional display layer is a dict that ALWAYS carries `xyz` so the
    # frontend can rely on the shape (empty is expressed as `[]`, never missing).
    for key in ("background", "wire", "wires3d", "tower"):
        assert isinstance(payload[key], dict)
        assert "xyz" in payload[key], f"{key} must always carry an xyz array"
        assert isinstance(payload[key]["xyz"], list)


def test_step3_final_display_schema_matches_frontend_contract(tmp_path):
    """Step-3 display layers use the same dict-with-xyz contract as Step 2."""
    workspace_root, workspace = _build_completed_workspace(tmp_path)
    server = FolderServer(workspace)
    # `_build_final_display` needs a reconciled/risk runtime; here we only pin
    # the shared builder shapes so the frontend's guards line up with the backend.
    bg = server._background_display()
    assert set(bg) == {"xyz", "rgb"}
    tower = server._tower_display()
    assert set(tower) == {"xyz", "labels"}
    wire = server._wire_curves_display()
    assert set(wire) == {"xyz"}


# ---------------------------------------------------------------------------
# frontend JS contract (static, no runtime needed)
# ---------------------------------------------------------------------------


def test_step2_js_reads_trees3d_and_guards_every_display_layer():
    """The Step-2 JS consumes `trees3d` (not `trees`) and guards each layer."""
    js = _extract_step2_js()
    # The 3D cloud must be read from `data.trees3d`, never from `data.trees`.
    assert "trees3d=data.trees3d" in js
    # The bare pattern that crashed the viewer must not exist anywhere.
    assert "trees.xyz.length" not in js
    assert "trees && trees.xyz" not in js
    # Every optional display layer must be guarded with Array.isArray so a
    # missing/empty `.xyz` degrades to an empty layer instead of a TypeError.
    assert js.count("Array.isArray(") >= 5  # background / wire / wires3d / tower / trees3d


def test_step3_js_guards_display_dict_layers():
    """Step-3 final JS guards the dict-`xyz` layers instead of calling `.length`."""
    js = _extract_step3_js()
    assert "disp.background" in js
    assert "disp.wire3d" in js
    assert "disp.towers" in js
    # Step 3 must not treat the dict layers as plain arrays (silent skip bug).
    assert "data.display.background.length" not in js
    assert "data.display.wire3d.length" not in js
    assert "data.display.towers.length" not in js


# ---------------------------------------------------------------------------
# JS execution (Node) — proves the initialization script no longer throws
# ---------------------------------------------------------------------------


_HARNESS = r"""
const vm = require('vm');
const fs = require('fs');
const jsFile = process.argv[2], payloadFile = process.argv[3], stateFile = process.argv[4];
const scriptSrc = fs.readFileSync(jsFile, 'utf8');
const editorData = JSON.parse(fs.readFileSync(payloadFile, 'utf8'));
const state = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
function makeElement(tag) {
  const el = { tag, textContent: '', innerHTML: '', className: '', style: {}, dataset: {},
    disabled: false, onclick: null, on(t, cb) { el._on = cb; return el; },
    addEventListener() {}, appendChild() {}, removeChild() {},
    getBoundingClientRect() { return { left: 0, top: 0 }; } };
  el.querySelector = () => makeElement('child');
  return el;
}
const documentStub = {
  getElementById(id) { return (documentStub._els[id] ||= makeElement(id)); },
  querySelector(sel) { return (documentStub._qs[sel] ||= makeElement(sel)); },
  createElement(tag) { return makeElement(tag); },
  _els: {}, _qs: {},
};
const tracesSeen = [];
const sandbox = {
  console, Map, Set, Math, Number, String, JSON, Array, Object, parseInt, parseFloat,
  document: documentStub,
  fetch: async (path) => {
    const body = String(path).includes('/state') ? state : editorData;
    return { json: async () => body, ok: true };
  },
  Plotly: { newPlot(el, traces) { tracesSeen.push(...traces.map(t => t.name)); return Promise.resolve(); } },
  setInterval() { return 1; }, clearInterval() {}, setTimeout() { return 1; },
  window: { location: { set href(v) {} }, addEventListener() {} }, location: { href: '' }, Headers: class {},
};
sandbox.window.document = documentStub;
(async () => {
  let error = null;
  try {
    vm.runInNewContext(scriptSrc, sandbox, { filename: 'step2.js', timeout: 15000 });
    await new Promise(r => setTimeout(r, 400));
  } catch (e) { error = String(e && e.message ? e.message : e); }
  console.log(JSON.stringify({ exception: error || 'NONE', traces: tracesSeen }));
  process.exit(error ? 1 : 0);
})();
"""


def _run_node_harness(tmp_path, js: str, editor_data: dict, state: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    js_path = tmp_path / "step2.js"
    payload_path = tmp_path / "editor_data.json"
    state_path = tmp_path / "state.json"
    js_path.write_text(js, encoding="utf-8")
    payload_path.write_text(json.dumps(editor_data), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    result = subprocess.run(
        [node, str(harness), str(js_path), str(payload_path), str(state_path)],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(f"node harness failed: {result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip())


def test_step2_js_executes_against_populated_payload(tmp_path):
    """The real Step-2 inline JS runs against a populated editor payload without throwing."""
    workspace_root, workspace = _build_completed_workspace(tmp_path)
    server = FolderServer(workspace)
    editor_data = server._build_editor_display()
    state = {
        "step": "TREE_EDITING", "step_label": "STEP 2", "current_message": None,
        "error": None,
        "tree_snapshot": {"original_tree_count": 2, "current_tree_count": 2,
                          "trees": editor_data["trees"], "selected_ids": []},
    }
    result = _run_node_harness(tmp_path, _extract_step2_js(), editor_data, state)
    assert result["exception"] == "NONE"
    names = set(result["traces"])
    # All five layers come from the SAME payload the tree list is built from.
    assert "背景" in names and "单木" in names and "导线" in names and "杆塔" in names


def test_step2_js_tolerates_empty_and_missing_display_layers(tmp_path):
    """An empty/missing display layer must render as nothing, never crash the viewer."""
    js = _extract_step2_js()
    state = {"step": "TREE_EDITING", "step_label": "STEP 2", "current_message": None,
             "error": None, "tree_snapshot": {"trees": [], "current_tree_count": 0}}

    empty = {
        "trees": [],
        "trees3d": {"xyz": [], "color": [], "tree_id_by_index": []},
        "background": {"xyz": [], "rgb": []},
        "wire": {"xyz": []},
        "wires3d": {"xyz": []},
        "tower": {"xyz": [], "labels": []},
    }
    result = _run_node_harness(tmp_path, js, empty, state)
    assert result["exception"] == "NONE"
    assert result["traces"] == []

    # Missing optional layers (the classic undefined.x.length crash).
    missing = {"trees": [], "background": {"xyz": [], "rgb": []}}
    result = _run_node_harness(tmp_path, js, missing, state)
    assert result["exception"] == "NONE"
    assert result["traces"] == []

    # Every display layer null.
    nulls = {"trees": None, "trees3d": None, "background": None,
             "wire": None, "wires3d": None, "tower": None}
    result = _run_node_harness(tmp_path, js, nulls, state)
    assert result["exception"] == "NONE"
    assert result["traces"] == []


# ---------------------------------------------------------------------------
# editor vs source tree count distinction
# ---------------------------------------------------------------------------


def test_state_tree_snapshot_refreshes_after_edit(tmp_path):
    """`/state.tree_snapshot` follows edits instead of flashing the stale snapshot.

    The Step-2 right-panel list is re-rendered on an interval from
    ``/state.tree_snapshot``; if that snapshot is only written at
    ``build_editor`` time, a delete/merge is visually reverted on the next poll.
    """
    workspace_root, workspace = _build_completed_workspace(tmp_path)
    server = FolderServer(workspace)
    orch = server.orchestrator
    assert orch.build_editor().snapshot()["current_tree_count"] == 2

    orch.editor_select([0])
    orch.editor_delete()
    assert orch.state.snapshot()["tree_snapshot"]["current_tree_count"] == 1
    assert orch.state.snapshot()["tree_edit_count"] == 1

    orch.editor_undo()
    assert orch.state.snapshot()["tree_snapshot"]["current_tree_count"] == 2
    assert orch.state.snapshot()["tree_edit_count"] == 0


def test_editor_tree_count_distinct_from_source_algorithm_count(tmp_path):
    """Editor tree count (2, span-owned) stays distinct from source count (3)."""
    workspace_root, workspace = _build_completed_workspace(tmp_path)
    manifest = json.loads(
        (workspace_root / "02_tree_analysis" / "tree_segmentation_original.json")
        .read_text(encoding="utf-8")
    )
    editor = manifest["algorithm_tree_count"]
    source = manifest["source_algorithm_tree_count"]
    assert editor == 2
    assert source == 3
    assert editor != source, "editor count must stay distinct from source algorithm count"

    server = FolderServer(workspace)
    snap = server.state.snapshot()
    ts = snap["tree_snapshot"]
    # The editor reports the EDITOR count (2), never the source algorithm count (3).
    assert ts["original_tree_count"] == editor
    assert ts["current_tree_count"] == editor
    assert len(server.orchestrator.original_trees.records) == editor
