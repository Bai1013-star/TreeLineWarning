"""Read-only Span final-reconciled Tree list + bounded preview API tests.

Uses the real ordered project when present (gitignored).  These are
GENERIC-span tests: every assertion is driven by an arbitrary Span ID resolved
through the project's authoritative run artifacts — never a hard-coded S002
branch.  Science is never mutated (read-only session + digest-checked reads).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORDERED_PROJECT = _REPO_ROOT / "cache" / "tower_ordered_project"

from scripts.serve_p3a_giro3d import TowerReviewApi, TowerReviewConfig  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (_ORDERED_PROJECT / "project.json").is_file(),
    reason="real ordered project not present",
)


@pytest.fixture(scope="module")
def api():
    return TowerReviewApi(
        TowerReviewConfig(
            project_root=_ORDERED_PROJECT,
            tile_cache=_ORDERED_PROJECT / ".tile-cache-unused",
            operator_identity="owner",
        )
    )


def _trees(api, span_id):
    return api.spans_trees(span_id)


# ---------------------------------------------------------------------------
# S002 real-data acceptance fixture — INVARIANT-based.  The current S002 run is
# the corrected HIGH-RECALL run (no broad wire exclusion), so no hard-coded tree
# count / clearance value is asserted here (the vegetation fix deliberately
# changed them).  Only structural invariants that hold for ANY final reconciled
# run are checked.
# ---------------------------------------------------------------------------


def test_s002_final_tree_list_consistent(api):
    payload = _trees(api, "S002")
    assert payload["span_id"] == "S002"
    assert payload["tree_count"] == len(payload["trees"])
    assert payload["tree_count"] > 0
    assert payload["core_count"] + payload["halo_review_count"] == payload["tree_count"]
    for tree in payload["trees"]:
        assert tree["external_name"].startswith("G001-Span002-Tree")


def test_s002_clearance_from_authoritative_artifact(api):
    payload = _trees(api, "S002")
    for tree in payload["trees"]:
        pair = tree["clearance"]
        if pair is None or pair["minimum_clearance_m"] is None:
            continue
        # The exact winning pair distance equals the persisted clearance.
        import math

        d = math.dist(pair["tree_xyz_absolute_m"], pair["wire_xyz_absolute_m"])
        assert d == pytest.approx(pair["minimum_clearance_m"], abs=1e-6)
        assert tree["minimum_clearance_m"] == pair["minimum_clearance_m"]


def test_no_zero_member_final_trees(api):
    payload = _trees(api, "S002")
    for tree in payload["trees"]:
        assert tree["full_resolution_point_count"] > 0
        assert tree["full_resolution_point_count"] == (
            tree["core_point_count"] + tree["halo_point_count"]
        )


def test_core_halo_review_mapping(api):
    payload = _trees(api, "S002")
    for tree in payload["trees"]:
        expected = (
            "CORE" if (tree["core_fraction"] > 0.0 and not tree["review"]) else "HALO / REVIEW"
        )
        assert tree["category"] == expected, (
            f"{tree['external_name']}: category {tree['category']} != {expected} "
            f"(core_fraction={tree['core_fraction']}, review={tree['review']})"
        )
    assert payload["core_count"] == sum(1 for t in payload["trees"] if t["category"] == "CORE")
    assert payload["halo_review_count"] == sum(
        1 for t in payload["trees"] if t["category"] == "HALO / REVIEW"
    )


def test_s002_clearance_pair_visualization_fields(api):
    payload = _trees(api, "S002")
    trees_with_clearance = [t for t in payload["trees"] if t["clearance"] is not None]
    assert trees_with_clearance  # the current run has distance-ready trees
    for tree in trees_with_clearance:
        pair = tree["clearance"]
        assert len(pair["tree_xyz_absolute_m"]) == 3
        assert len(pair["wire_xyz_absolute_m"]) == 3


# ---------------------------------------------------------------------------
# Generic resolution: every Span resolves its own final trees
# ---------------------------------------------------------------------------


def test_generic_arbitrary_span_resolution(api):
    # S001/S002/S003/S004/S005 all use the exact same code path.
    for span_id in ("S001", "S002", "S003", "S004", "S005"):
        payload = _trees(api, span_id)
        assert payload["span_id"] == span_id
        assert payload["tree_count"] == len(payload["trees"])
        assert all(
            isinstance(tree["external_name"], str) and tree["external_name"]
            for tree in payload["trees"]
        )


def test_s005_many_trees_list_usable(api):
    payload = _trees(api, "S005")
    assert payload["tree_count"] >= 10
    # The aggregate counts always reconcile to the tree list.
    assert payload["core_count"] + payload["halo_review_count"] == payload["tree_count"]


def test_s005_viewer_layers_share_one_verified_run(api):
    span = api.spans_get("S005")["span"]
    trees = api.spans_trees("S005")
    wire = api.spans_wire("S005")
    danger = api.spans_danger_points("S005")
    run_id = span["active_run_id"]
    assert run_id == trees["run_id"] == wire["run_id"] == danger["run_id"]
    assert span["group_id"] == "G002"
    assert (span["from_tower_id"], span["to_tower_id"]) == ("T006", "T007")
    assert trees["tree_count"] == 22
    assert wire["authoritative"] is True and wire["strand_count"] == 3
    assert danger["authoritative"] is True
    assert danger["danger_point_count"] == 38_589
    assert danger["danger_tree_count"] == 4


def test_no_hardcoded_span_branch_in_api(api):
    source = inspect.getsource(api.spans_trees) + inspect.getsource(api.spans_tree_preview)
    for token in ("S002", "G001-Span002", "G001-Span002-Tree001"):
        assert token not in source, f"hard-coded Span/Tree branch in API: {token}"
    assert "span_id" in source  # driven by the resolved Span id


# ---------------------------------------------------------------------------
# Bounded preview (S005 holds tens of millions of members)
# ---------------------------------------------------------------------------


def test_tree_preview_bounded(api):
    # Any tree with more members than the display cap is sampled to exactly the cap.
    payload = _trees(api, "S002")
    big = max(payload["trees"], key=lambda t: t["full_resolution_point_count"])
    assert big["full_resolution_point_count"] > 40_000
    preview = api.spans_tree_preview("S002", str(big["global_tree_id"]))
    assert preview["full_member_count"] == big["full_resolution_point_count"]
    assert preview["sample_count"] == 40_000  # deterministic hard cap
    assert len(preview["sampled_xyz_absolute_m"]) == 40_000
    assert preview["bounds_absolute_m"] is not None
    assert len(preview["centroid_xyz_absolute_m"]) == 3


def test_small_tree_preview_not_over_sampled(api):
    # A small tree is returned complete (under the cap).
    payload = _trees(api, "S002")
    small = min(payload["trees"], key=lambda t: t["full_resolution_point_count"])
    assert small["full_resolution_point_count"] < 40_000
    preview = api.spans_tree_preview("S002", str(small["global_tree_id"]))
    assert preview["full_member_count"] == small["full_resolution_point_count"]
    assert preview["sample_count"] == small["full_resolution_point_count"]


def test_s005_large_tree_preview_bounded(api):
    payload = _trees(api, "S005")
    biggest = max(payload["trees"], key=lambda t: t["full_resolution_point_count"])
    assert biggest["full_resolution_point_count"] > 1_000_000
    preview = api.spans_tree_preview("S005", str(biggest["global_tree_id"]))
    assert preview["full_member_count"] == biggest["full_resolution_point_count"]
    assert preview["sample_count"] == 40000  # hard display cap
    assert len(preview["sampled_xyz_absolute_m"]) == 40000


def test_unknown_tree_preview_fails_closed(api):
    with pytest.raises(Exception) as excinfo:
        api.spans_tree_preview("S002", "999")
    assert excinfo.value.status == 404  # structured 404, no fabricated preview


# ---------------------------------------------------------------------------
# No science mutation
# ---------------------------------------------------------------------------


def test_tree_api_does_not_mutate_project(api):
    from core.project import open_project_read_only

    before = None
    with open_project_read_only(_ORDERED_PROJECT) as session:
        before = session.generation_id
    # Read-only tree list + previews must not advance the generation.
    _trees(api, "S002")
    api.spans_tree_preview("S002", "1")
    with open_project_read_only(_ORDERED_PROJECT) as session:
        assert session.generation_id == before
        assert len(session.span_registry.records) == 5
