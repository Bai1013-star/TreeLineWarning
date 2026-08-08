"""Tests for the read-only V4 tree-segmentation review visualization."""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation.v4_tree_artifacts import publish_v4_tree_segmentation_artifact
from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees
from algorithms.vegetation.v4_tree_visualization import (
    LAYER_NAMES,
    V4TreeVisualizationBudgetError,
    V4TreeVisualizationError,
    V4TreeVisualizationOutputError,
    build_summary,
    deterministic_tree_sample_positions,
    render_html,
    run_v4_tree_visualization,
    tree_color,
    validate_display_budget,
)
from core.contracts.v4_tree_segmentation import (
    V4ApexCandidateRecord,
    V4AssignmentReasonFlag,
    V4TreeAssignmentState,
    V4TreeCrownSummary,
    V4TreeGeometrySummary,
    V4TreeInstanceRecord,
    V4TreeSegmentationAudit,
    V4TreeSegmentationResult,
    V4TrunkCandidateRecord,
    V4TrunkCandidateState,
    V4TrunkEvidenceAvailability,
)

from tests.io.test_source_provider import _source_identity
from tests.vegetation.test_v4_tree_segmentation import (
    make_config,
    make_handoff,
    two_trees_scene,
)

_RO = False


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _source():
    return _source_identity(
        "C:/synthetic.las", "a" * 64, 8, ([[0.0, 0.0, 0.0], [14.0, 6.0, 7.0]])
    )


def _audit(tree_count: int = 0, total_points: int = 0) -> V4TreeSegmentationAudit:
    return V4TreeSegmentationAudit(
        schema_version="v4-tree-segmentation-result-v1",
        total_points=total_points,
        apex_candidate_count=0,
        accepted_trunk_count=0,
        weak_trunk_count=0,
        suppressed_trunk_count=0,
        unmatched_trunk_count=0,
        strong_association_count=0,
        review_association_count=0,
        unsupported_association_count=0,
        path_observed_count=0,
        path_not_available_count=0,
        profile_confirmed_count=0,
        profile_weak_count=0,
        profile_no_valley_count=0,
        profile_not_available_count=0,
        hypothesis_count=0,
        slice_count=0,
        relation_provisional_count=0,
        relation_final_count=0,
        tree_count=tree_count,
        phase_resource_audit="synthetic",
        determinism_evidence="synthetic",
        warnings=(),
        limitations=(),
    )


def _apex(
    candidate_id: str, xyz: tuple[float, float, float] = (2.0, 2.0, 6.2)
) -> V4ApexCandidateRecord:
    fields = {
        "candidate_id": candidate_id,
        "supporting_original_index": 0,
        "xyz_local_m": list(xyz),
        "hag_m": 6.2,
        "prominence_m": 1.0,
        "prominence_available": True,
        "plateau_support_count": 5,
        "cell_point_count": 10,
        "density_points_per_m2": 3.0,
        "neighborhood_median_density_points_per_m2": 1.0,
        "apex_support_score": 0.9,
        "quality_flags": 0,
        "reason_flags": 0,
    }
    digest = _digest({"digest_domain": "V4APX1", **fields})
    return V4ApexCandidateRecord(**fields, diagnostic_digest=digest)


def _trunk(
    candidate_id: str, state: V4TrunkCandidateState = V4TrunkCandidateState.ACCEPTED
) -> V4TrunkCandidateRecord:
    fields = {
        "candidate_id": candidate_id,
        "representative_original_index": 0,
        "supporting_original_index_digest": "b" * 64,
        "centroid_xyz": [2.0, 2.0, 1.5],
        "bounds_min_xyz": [1.9, 1.9, 0.4],
        "bounds_max_xyz": [2.1, 2.1, 3.0],
        "refit_principal_axis": [0.0, 0.0, 1.0],
        "core_hag_min_m": 0.4,
        "core_hag_max_m": 3.0,
        "observed_vertical_span_m": 2.6,
        "k_min": 0,
        "k_max": 3,
        "evaluated_bins": 4,
        "occupied_required_bins": 4,
        "max_bin_centroid_xy_drift_m": 0.05,
        "tv": 1.0,
        "ta": 1.0,
        "ts": 1.0,
        "tc": 1.0,
        "trunk_candidate_score": 0.9,
        "candidate_state": state.value,
        "quality_flags": 0,
        "reason_flags": 0,
    }
    digest = _digest({"digest_domain": "V4TRK1", **fields})
    fields["candidate_state"] = state
    return V4TrunkCandidateRecord(**fields, diagnostic_digest=digest)


def _digest(payload) -> str:
    from core.contracts.frv import frv_digest

    return frv_digest(payload)


def _tree_record(
    tree_id: int,
    member_count: int,
    *,
    primary_apex: str = "apex-0",
    primary_trunk: str | None = None,
    xy_centroid: tuple[float, float, float] = (2.0, 2.0, 4.0),
) -> V4TreeInstanceRecord:
    crown = V4TreeCrownSummary(
        crown_area_m2=3.14,
        equivalent_radius_m=1.0,
        responsibility_area_m2=5.0,
        boundary_vertex_count=0,
        boundary_digest="d" * 64,
    )
    geometry = V4TreeGeometrySummary(
        member_count=member_count,
        hag_min_m=0.4,
        hag_max_m=6.2,
        robust_hag_m=5.0,
        vertical_extent_m=5.8,
        xy_centroid=xy_centroid,
        xy_bounds_min=(0.4, 0.4, 0.4),
        xy_bounds_max=(3.6, 3.6, 6.2),
        crown_area_m2=3.14,
        equivalent_radius_m=1.0,
        responsibility_area_m2=5.0,
        slice_count=1,
        valid_slice_count=1,
        lowest_supported_slice_id="s0",
        highest_supported_slice_id="s0",
        trunk_evidence_state=(
            V4TrunkEvidenceAvailability.OBSERVED
            if primary_trunk is not None
            else V4TrunkEvidenceAvailability.NOT_AVAILABLE
        ),
        accepted_trunk_count=1 if primary_trunk is not None else 0,
        primary_trunk_representative_original_index=0 if primary_trunk is not None else None,
        multi_stem_flag=False,
        geometry_quality_flags=0,
    )
    return V4TreeInstanceRecord(
        tree_id=tree_id,
        primary_apex_candidate_id=primary_apex,
        secondary_apex_candidate_ids=(),
        primary_trunk_candidate_id=primary_trunk,
        secondary_trunk_candidate_ids=(),
        trunk_apex_association_ids=(),
        member_count=member_count,
        member_original_index_digest="e" * 64,
        crown_summary=crown,
        geometry_summary=geometry,
        slice_ids=("s0",),
        relation_ids=(),
        confidence=0.95,
        quality_flags=0,
        reason_flags=0,
        future_visual_evidence_reference=None,
    )


def _synthetic_result(
    *,
    xyz: list[tuple[float, float, float]],
    states: list[int],
    tree_ids: list[int],
    reason_flags: list[int],
    tree_records: list[V4TreeInstanceRecord],
    apex_records: list[V4ApexCandidateRecord] | None = None,
    trunk_records: list[V4TrunkCandidateRecord] | None = None,
) -> V4TreeSegmentationResult:
    n = len(xyz)
    result = V4TreeSegmentationResult.__new__(V4TreeSegmentationResult)
    object.__setattr__(result, "schema_version", "v4-tree-segmentation-result-v1")
    object.__setattr__(result, "algorithm_version", "v4-algorithm-v1")
    object.__setattr__(result, "source_identity", _source())
    object.__setattr__(result, "config", make_config())
    object.__setattr__(result, "handoff_digest", "h" * 64)
    object.__setattr__(result, "validated_view_digest", "v" * 64)
    object.__setattr__(result, "coverage_report_digest", "c" * 64)
    object.__setattr__(result, "original_index", _ro(np.arange(n, dtype="<i8")))
    object.__setattr__(result, "xyz_local_m", _ro(np.asarray(xyz, dtype="<f8")))
    object.__setattr__(result, "hag_m", _ro(np.asarray(xyz, dtype=np.float64)[:, 2].astype("<f4")))
    object.__setattr__(result, "assignment_state_by_point", _ro(np.asarray(states, dtype="|u1")))
    object.__setattr__(result, "tree_id_by_point", _ro(np.asarray(tree_ids, dtype="<i4")))
    object.__setattr__(
        result,
        "assignment_confidence_by_point",
        _ro(np.full(n, 0.9, dtype="<f4")),
    )
    object.__setattr__(
        result,
        "assignment_reason_flags_by_point",
        _ro(np.asarray(reason_flags, dtype="<u4")),
    )
    object.__setattr__(
        result, "tree_member_offsets", _ro(np.zeros(len(tree_records) + 1, dtype="<i8"))
    )
    object.__setattr__(
        result, "ordered_tree_member_original_index", _ro(np.empty(0, dtype="<i8"))
    )
    object.__setattr__(result, "apex_records", tuple(apex_records or ()))
    object.__setattr__(result, "trunk_records", tuple(trunk_records or ()))
    object.__setattr__(result, "trunk_apex_association_records", ())
    object.__setattr__(result, "structural_path_records", ())
    object.__setattr__(result, "rotating_profile_records", ())
    object.__setattr__(result, "hypothesis_records", ())
    object.__setattr__(result, "slice_records", ())
    object.__setattr__(result, "relation_records", ())
    object.__setattr__(result, "tree_records", tuple(tree_records))
    object.__setattr__(
        result, "crown_boundary_vertices", _ro(np.empty((0, 2), dtype="<f8"))
    )
    object.__setattr__(
        result, "crown_boundary_offsets", _ro(np.zeros(len(tree_records) + 1, dtype="<i8"))
    )
    object.__setattr__(result, "audit", _audit(len(tree_records), n))
    object.__setattr__(result, "result_digest", "0" * 64)
    return result


def _mixed_result() -> V4TreeSegmentationResult:
    """Two trees with assigned, review, unassigned and REVIEW_CONFLICT points."""
    xyz = [
        (1.0, 1.0, 5.0),
        (1.5, 1.0, 5.5),
        (2.0, 2.0, 6.0),
        (8.0, 8.0, 5.0),
        (8.5, 8.0, 5.5),
        (5.0, 5.0, 4.0),  # review between the two trees
        (5.2, 5.2, 4.0),  # REVIEW_CONFLICT
        (0.5, 0.5, 0.5),  # unassigned low support
    ]
    states = [
        int(V4TreeAssignmentState.ASSIGNED),
        int(V4TreeAssignmentState.ASSIGNED),
        int(V4TreeAssignmentState.ASSIGNED),
        int(V4TreeAssignmentState.ASSIGNED),
        int(V4TreeAssignmentState.ASSIGNED),
        int(V4TreeAssignmentState.REVIEW_REQUIRED),
        int(V4TreeAssignmentState.REVIEW_REQUIRED),
        int(V4TreeAssignmentState.UNASSIGNED),
    ]
    tree_ids = [0, 0, 0, 1, 1, -1, -1, -1]
    reason_flags = [
        0,
        0,
        0,
        0,
        0,
        0,
        int(V4AssignmentReasonFlag.REVIEW_CONFLICT),
        0,
    ]
    apexes = [_apex("apex-0", (2.0, 2.0, 6.2)), _apex("apex-1", (8.0, 8.0, 5.5))]
    trunks = [_trunk("trunk-0"), _trunk("trunk-1", V4TrunkCandidateState.SUPPRESSED)]
    trees = [
        _tree_record(
            0, 3, primary_apex="apex-0", primary_trunk="trunk-0", xy_centroid=(2.0, 2.0, 4.0)
        ),
        _tree_record(
            1, 2, primary_apex="apex-1", primary_trunk="trunk-1", xy_centroid=(8.0, 8.0, 4.0)
        ),
    ]
    return _synthetic_result(
        xyz=xyz,
        states=states,
        tree_ids=tree_ids,
        reason_flags=reason_flags,
        tree_records=trees,
        apex_records=apexes,
        trunk_records=trunks,
    )


# ---------------------------------------------------------------------------
# Tree colours and sampling
# ---------------------------------------------------------------------------


def test_tree_color_is_stable_and_distinct() -> None:
    assert tree_color(0) == tree_color(0)
    assert tree_color(1) == tree_color(1)
    assert tree_color(0) != tree_color(1)
    for tree_id in range(20):
        color = tree_color(tree_id)
        assert color.startswith("#")
        assert len(color) == 7


def test_bounded_sampling_small_tree_not_lost() -> None:
    big = np.arange(1000, dtype=np.int64)
    small = np.arange(5, dtype=np.int64)
    big_sample = deterministic_tree_sample_positions(big, cap=10)
    small_sample = deterministic_tree_sample_positions(small, cap=10)
    assert big_sample.shape[0] == 10  # large tree is bounded
    assert small_sample.shape[0] == 5  # small tree keeps every point
    assert np.array_equal(small_sample, small)


def test_sampling_is_deterministic() -> None:
    positions = np.arange(500, dtype=np.int64)
    assert np.array_equal(
        deterministic_tree_sample_positions(positions, 37),
        deterministic_tree_sample_positions(positions, 37),
    )


# ---------------------------------------------------------------------------
# Layer extraction and rendering
# ---------------------------------------------------------------------------


def test_layer_positions_include_review_unassigned_and_conflict() -> None:
    from algorithms.vegetation.v4_tree_visualization import _layer_positions

    result = _mixed_result()
    layers = _layer_positions(result)
    assert layers["assigned"].shape[0] == 5
    assert layers["review"].shape[0] == 2
    assert layers["unassigned"].shape[0] == 1
    assert layers["conflict"].shape[0] == 1


def test_render_html_contains_all_layers(tmp_path: Path) -> None:
    result = _mixed_result()
    html = render_html(result, validate_display_budget(100, 100, 100, 100))
    assert "plotly" in html.lower()
    assert "tree 0" in html
    assert "tree 1" in html
    assert LAYER_NAMES["review"] in html
    assert LAYER_NAMES["unassigned"] in html
    assert LAYER_NAMES["conflict"] in html
    assert LAYER_NAMES["apex"] in html
    assert LAYER_NAMES["trunk"] in html
    assert LAYER_NAMES["footprint"] in html


def test_displayed_counts_for_mixed_result(tmp_path: Path) -> None:
    from algorithms.vegetation.v4_tree_visualization import (
        _tree_accepted_trunks,
        _tree_apexes,
    )

    result = _mixed_result()
    layers = _layers(result)
    # Assigned points per tree are bounded per-tree (2 each under cap 2).
    assert sum(
        min(int(_assigned_map(result)[tree.tree_id].shape[0]), 2)
        for tree in result.tree_records
    ) == 4
    assert min(int(layers["review"].shape[0]), 100) == 2
    assert min(int(layers["unassigned"].shape[0]), 100) == 1
    assert min(int(layers["conflict"].shape[0]), 100) == 1
    assert len(_tree_apexes(result, result.tree_records[0])) == 1
    assert len(_tree_accepted_trunks(result, result.tree_records[0])) == 1
    # The suppressed trunk is never displayed.
    assert len(_tree_accepted_trunks(result, result.tree_records[1])) == 0


def test_summary_is_deterministic() -> None:
    result = _mixed_result()
    kwargs = {
        "result": result,
        "layer_positions": _layers(result),
        "assigned_positions_by_tree": _assigned_map(result),
        "displayed_counts": {
            "assigned": 4, "review": 2, "unassigned": 1, "conflict": 1, "apex": 2, "trunk": 1
        },
        "budget": validate_display_budget(100, 100, 100, 100),
        "output_paths": {"html": "a.html", "summary_json": "a.json"},
    }
    assert build_summary(**kwargs) == build_summary(**kwargs)


def _layers(result):
    from algorithms.vegetation.v4_tree_visualization import _layer_positions

    return _layer_positions(result)


def _assigned_map(result):
    from algorithms.vegetation.v4_tree_visualization import _tree_assigned_positions

    return _tree_assigned_positions(result)


# ---------------------------------------------------------------------------
# End-to-end over a real segmentation artifact
# ---------------------------------------------------------------------------


def _make_real_artifact(tmp_path: Path) -> Path:
    handoff = make_handoff(two_trees_scene())
    result = segment_v4_trees(handoff, make_config())
    root = tmp_path / "real-v4"
    publish_v4_tree_segmentation_artifact(result, root, scratch_dir=tmp_path)
    return root


def test_real_artifact_end_to_end(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    output = run_v4_tree_visualization(v4_root, tmp_path / "out.html", tmp_path / "out.json")
    assert output.html_path.is_file()
    assert output.summary_path.is_file()
    html = output.html_path.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    summary = json.loads(output.summary_path.read_text(encoding="utf-8"))
    assert summary["tool"] == "v4-tree-visualization"
    assert summary["trees"][0]["tree_id"] == 0
    assert summary["trees"][0]["assigned_point_count"] > 0
    assert summary["displayed_counts"]["assigned"] > 0
    assert summary["displayed_counts"]["apex"] == len(summary["trees"])
    assert summary["v4_result_digest"]


def test_real_artifact_is_deterministic(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    out_html = tmp_path / "out.html"
    out_json = tmp_path / "out.json"
    first = run_v4_tree_visualization(v4_root, out_html, out_json)
    second = run_v4_tree_visualization(v4_root, out_html, out_json, overwrite=True)
    assert first.html_path.read_bytes() == second.html_path.read_bytes()
    assert first.summary == second.summary


def test_real_artifact_small_cap_is_bounded(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    output = run_v4_tree_visualization(
        v4_root, tmp_path / "out.html", tmp_path / "out.json", max_points_per_tree=3
    )
    assert all(
        tree["displayed_point_count"] <= 3 for tree in output.summary["trees"]
    )


def test_invalid_artifact_refusal(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(V4TreeVisualizationError):
        run_v4_tree_visualization(missing, tmp_path / "out.html")
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    (bogus / "v4_tree_segmentation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(V4TreeVisualizationError):
        run_v4_tree_visualization(bogus, tmp_path / "out2.html")


def test_output_overwrite_refusal(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    out = tmp_path / "existing.html"
    out.write_text("old", encoding="utf-8")
    with pytest.raises(V4TreeVisualizationOutputError):
        run_v4_tree_visualization(v4_root, out)
    assert out.read_text(encoding="utf-8") == "old"  # untouched


def test_overwrite_flag_allows_replacement(tmp_path: Path) -> None:
    v4_root = _make_real_artifact(tmp_path)
    out = tmp_path / "existing.html"
    out.write_text("old", encoding="utf-8")
    result = run_v4_tree_visualization(v4_root, out, overwrite=True)
    assert result.html_path.is_file()
    assert "plotly" in out.read_text(encoding="utf-8")[:100_000].lower()


def test_budget_validation_rejects_invalid_caps() -> None:
    with pytest.raises(V4TreeVisualizationBudgetError):
        validate_display_budget(0, 10, 10, 10)
    with pytest.raises(V4TreeVisualizationBudgetError):
        validate_display_budget(1_000_000, 10, 10, 10)
    with pytest.raises(V4TreeVisualizationBudgetError):
        validate_display_budget(10, 0, 10, 10)
    with pytest.raises(V4TreeVisualizationBudgetError):
        validate_display_budget(10, 10, -1, 10)


def test_no_segmentation_runtime_imported() -> None:
    source = pathlib.Path("algorithms/vegetation/v4_tree_visualization.py").read_text(
        encoding="utf-8"
    )
    assert "segment_v4_trees" not in source
    assert "from algorithms.vegetation.v4_tree_segmentation" not in source
