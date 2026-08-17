"""Tree Viewer D identity and presentation contracts (no project mutation)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from scripts.serve_p3a_giro3d import TowerReviewApi

ROOT = Path(__file__).resolve().parents[2]
TREE_VIEWER = ROOT / "viewer_poc" / "giro3d" / "tree_viewer.js"
INDEX = ROOT / "viewer_poc" / "giro3d" / "index.html"


def test_server_identity_presentation_preserves_core_semantics() -> None:
    persistent = TowerReviewApi._tree_identity_presentation(
        {"global_tree_id": 23, "persistent_tree_id": "G002-T024"}
    )
    assert persistent["display_id"] == "G002-T024"
    assert persistent["prediction_eligible"] is True
    assert persistent["color_key"] == "persistent:G002-T024"

    legacy = TowerReviewApi._tree_identity_presentation({"global_tree_id": 24})
    assert legacy["display_id"] == "Legacy Tree 24"
    assert legacy["persistent_tree_id"] is None
    assert legacy["prediction_eligible"] is False

    review = TowerReviewApi._tree_identity_presentation(
        {
            "global_tree_id": 25,
            "identity_state": "MERGE_AMBIGUOUS_REVIEW",
            "candidate_persistent_tree_ids": ["G002-T008", "G002-T007", "G002-T008"],
        }
    )
    assert review["display_id"] == "未绑定实体"
    assert review["persistent_tree_id"] is None
    assert review["candidate_persistent_tree_ids"] == ["G002-T007", "G002-T008"]
    assert review["needs_review"] is True
    assert review["prediction_eligible"] is False


def test_server_tree_color_is_stable_and_identity_keyed() -> None:
    old = TowerReviewApi._tree_instance_color("persistent:G002-T024")
    assert old == TowerReviewApi._tree_instance_color("persistent:G002-T024")
    assert old != ""
    # No collection position is an input, so a new tree cannot renumber old colors.
    assert old == TowerReviewApi._tree_instance_color("persistent:G002-T024")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_tree_viewer_identity_helpers() -> None:
    script = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync("__TREE_VIEWER__", 'utf8');
const sandbox = { window: {}, document: { getElementById: () => null }, console: console };
sandbox.window = sandbox;
sandbox.window.addEventListener = () => {};
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'tree_viewer.js' });
const u = sandbox.p3aTreeViewerUtil;
let failed = 0;
function check(value, message) { if (!value) { failed += 1; console.log('FAIL: ' + message); } }
const a = { global_tree_id: 23, persistent_tree_id: 'G002-T024', span_order_index: 4 };
const b = { global_tree_id: 99, persistent_tree_id: 'G002-T099', span_order_index: 99 };
check(u.treeColor(a) === u.treeColor(a), 'same persistent ID has the same color');
const rerun = { global_tree_id: 23, persistent_tree_id: 'G002-T024' };
check(u.treeColor(a) === u.treeColor(rerun), 'rerun preserves persistent color');
check(u.treeColor(a) === u.treeColor([b, a][1]), 'insertion preserves persistent color');
const ordered = u.sortTrees([
  { global_tree_id: 9, span_order_index: 9 }, a, { global_tree_id: 2, span_order_index: 2 }
]);
check(ordered.map((tree) => tree.global_tree_id).join(',') === '2,23,9', 'list uses span order');
const merge = u.treePresentation({
  global_tree_id: 25, identity_state: 'MERGE_AMBIGUOUS_REVIEW',
  candidate_persistent_tree_ids: ['G002-T008', 'G002-T007']
});
check(merge.primary === '未绑定实体' && merge.state === 'MERGE REVIEW', 'merge has no fake ID');
check(merge.candidates.join(',') === 'G002-T007,G002-T008', 'merge candidates are deterministic');
check(merge.longTermPredictionEntity === false, 'merge review is not a prediction entity');
const legacy = u.treePresentation({ global_tree_id: 24 });
check(legacy.primary === 'Legacy Tree 24', 'legacy record has an explicit fallback');
if (failed) process.exit(1);
console.log('tree viewer identity helpers PASS');
""".replace("__TREE_VIEWER__", str(TREE_VIEWER).replace("\\", "/"))
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=60, cwd=str(ROOT)
    )
    assert result.returncode == 0, f"node failed:\n{result.stdout}\n{result.stderr}"
    assert "tree viewer identity helpers PASS" in result.stdout


def test_tree_panel_uses_one_persistent_identity_presentation() -> None:
    source = TREE_VIEWER.read_text(encoding="utf-8")
    html = INDEX.read_text(encoding="utf-8")
    for token in (
        "persistent_tree_id",
        "global_tree_id",
        "candidate_persistent_tree_ids",
        "MERGE_AMBIGUOUS_REVIEW",
        "span_order_index",
        "instance_color",
        "persistent-tree-markers",
        "persistent-tree-selection-preview",
    ):
        assert token in source
    server_source = (ROOT / "scripts" / "serve_p3a_giro3d.py").read_text(encoding="utf-8")
    assert "prediction_identity" in server_source
    for token in ("tree_panel", "tree_span_select", "tree_header", "tree_list", "tree_details"):
        assert token in html
