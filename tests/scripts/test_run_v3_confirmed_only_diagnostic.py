from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scripts.run_v3_confirmed_only_diagnostic as runner
from core.contracts.tree_inputs import TreeSegmentationInputDisposition
from core.contracts.vegetation_instances import VegetationClusterStatus

from tests.vegetation.test_tree_segmentation import _pipeline


def _source() -> object:
    source, _model, _candidates, _segmentation = _pipeline(
        xy=[(0.0, 0.0), (0.5, 0.0), (2.0, 0.0), (2.5, 0.0)],
        hag=[4.0, 3.5, 5.0, 4.5],
        statuses=[
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.REVIEW_REQUIRED,
        ],
        cluster_ids=[10, 10, 20, 20],
        minimum_peak_hag_m=0.0,
    )
    return source


def _arguments(output_dir: Path, maximum: str = "10") -> list[str]:
    return [
        "--v2-npz",
        "v2.npz",
        "--v2-json",
        "v2.json",
        "--a2a-npz",
        "a2a.npz",
        "--a2a-json",
        "a2a.json",
        "--a2b-npz",
        "a2b.npz",
        "--a2b-json",
        "a2b.json",
        "--output-dir",
        str(output_dir),
        "--max-3d-points",
        maximum,
    ]


def _reviewed_summary_payload(source: object, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_identity": source.identity.to_dict(),
        "source_point_count": 4,
        "assigned_point_count": 3,
        "unassigned_point_count": 1,
        "candidate_count": 2,
        "segment_record_count": 2,
        "chm_shape": [2, 3],
        "ambiguous_cell_count": 0,
        "conflict_pixel_count": 7,
    }
    payload.update(overrides)
    return payload


def _reconciliation_metrics() -> dict[str, int | list[int]]:
    return {
        "source_point_count": 4,
        "assigned_point_count": 3,
        "unassigned_point_count": 1,
        "candidate_count": 2,
        "record_count": 2,
        "chm_shape": [2, 3],
        "ambiguity_cell_count": 0,
        "topdown_conflict_pixel_count": 7,
    }


def _partitioned_source() -> object:
    source, _model, _candidates, _segmentation = _pipeline(
        xy=[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
            (7.0, 0.0),
        ],
        hag=[4.0] * 8,
        statuses=[
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.PROVISIONAL,
            VegetationClusterStatus.PROVISIONAL,
            VegetationClusterStatus.REJECTED_AS_INSTANCE,
            VegetationClusterStatus.REJECTED_AS_INSTANCE,
        ],
        cluster_ids=[10, 10, 20, 20, 30, 30, 40, 40],
        minimum_peak_hag_m=0.0,
    )
    return source


def test_parse_args_is_repository_independent(tmp_path: Path) -> None:
    args = runner.parse_args(_arguments(tmp_path / "result", "15"))
    assert args.output_dir == tmp_path / "result"
    assert args.max_3d_points == 15
    assert not args.verify_determinism


@pytest.mark.parametrize("maximum", ["-1", "0", "1.5", "200001"])
def test_parse_args_rejects_invalid_maximum(maximum: str, tmp_path: Path) -> None:
    with pytest.raises(runner.DiagnosticCliError, match="max-3d-points"):
        runner.parse_args(_arguments(tmp_path / "result", maximum))


def test_output_directory_is_contained_and_refuses_existing_target(tmp_path: Path) -> None:
    temporary_target = tmp_path / "new-diagnostic"
    assert runner._validate_output_dir(temporary_target) == temporary_target.resolve()
    temporary_target.mkdir()
    with pytest.raises(runner.DiagnosticCliError, match="already exists"):
        runner._validate_output_dir(temporary_target)
    with pytest.raises(runner.DiagnosticCliError, match="output-dir"):
        runner._validate_output_dir(Path("data") / "forbidden")


def test_projection_allocates_only_status_changes_and_preserves_source() -> None:
    source = _source()
    original_point_status = source.vegetation_points.cluster_status_by_point.copy()
    original_cluster_status = source.cluster_status.copy()

    projected = runner.confirmed_only_projection(source)

    assert (
        projected.vegetation_points.cluster_status_by_point
        is not source.vegetation_points.cluster_status_by_point
    )
    assert projected.cluster_status is not source.cluster_status
    assert projected.cluster_id is source.cluster_id
    assert projected.geometry_features is source.geometry_features
    assert projected.heights_above_ground is source.heights_above_ground
    assert projected.identity is source.identity
    assert not projected.vegetation_points.cluster_status_by_point.flags.writeable
    assert not projected.cluster_status.flags.writeable
    assert source.review_only_cluster_ids == (20,)
    assert projected.review_only_cluster_ids == ()
    assert projected.excluded_cluster_ids == (20,)
    assert np.array_equal(source.vegetation_points.cluster_status_by_point, original_point_status)
    assert np.array_equal(source.cluster_status, original_cluster_status)
    assert np.all(
        projected.vegetation_points.cluster_status_by_point[2:]
        == int(VegetationClusterStatus.REJECTED_AS_INSTANCE)
    )


def test_projection_partitions_complete_status_set() -> None:
    source = _partitioned_source()
    projected = runner.confirmed_only_projection(source)

    dispositions = {
        int(row.cluster_id): (row.disposition, row.reason)
        for row in projected.cluster_dispositions
    }
    assert dispositions[10] == (
        TreeSegmentationInputDisposition.ELIGIBLE,
        "confirmed_distance_ready",
    )
    for cluster_id in (20, 30, 40):
        assert dispositions[cluster_id] == (
            TreeSegmentationInputDisposition.EXCLUDED,
            "rejected_as_instance",
        )
    assert projected.review_only_cluster_ids == ()
    assert projected.excluded_cluster_ids == (20, 30, 40)
    assert source.eligible_cluster_ids == (10,)


def test_projection_preserves_pre_existing_excluded_membership() -> None:
    source = _partitioned_source()
    projected = runner.confirmed_only_projection(source)

    source_dispositions = {
        int(row.cluster_id): (row.disposition, row.reason)
        for row in source.cluster_dispositions
    }
    assert source_dispositions[40] == (
        TreeSegmentationInputDisposition.EXCLUDED,
        "rejected_as_instance",
    )
    member_40 = source.vegetation_points.cluster_id_by_point == 40
    assert np.all(
        source.vegetation_points.cluster_status_by_point[member_40]
        == int(VegetationClusterStatus.REJECTED_AS_INSTANCE)
    )
    assert np.all(
        projected.vegetation_points.cluster_status_by_point[member_40]
        == int(VegetationClusterStatus.REJECTED_AS_INSTANCE)
    )
    assert np.all(
        projected.cluster_status[source.cluster_id == 40]
        == int(VegetationClusterStatus.REJECTED_AS_INSTANCE)
    )
    assert 40 in projected.excluded_cluster_ids


def test_projected_input_passes_v3_and_never_creates_permanent_ids() -> None:
    source = _source()
    projected = runner.confirmed_only_projection(source)

    _chm, _candidates, _segmentation, result, _fingerprint = runner._run_pipeline(projected)

    assert np.all(result.tree_id_by_point == -1)
    assert not np.any(result.tree_distance_ready_by_point)


def test_metrics_distinguish_raw_chm_and_assigned_counts() -> None:
    source = _source()
    canopy, candidates, segmentation, result, fingerprint = runner._run_pipeline(source)
    metrics = runner.collect_metrics(
        variant="production_like",
        tree_input=source,
        original_input=source,
        canopy_model=canopy,
        candidates=candidates,
        segmentation=segmentation,
        result=result,
        fingerprint=fingerprint,
        determinism_verified=True,
    )

    assert metrics["raw_selected_member_point_count"] == 4
    assert metrics["chm_included_point_count"] <= metrics["raw_selected_member_point_count"]
    assert metrics["assigned_point_count"] <= metrics["chm_included_point_count"]
    assert metrics["deterministic_fingerprint_sha256"] == fingerprint
    assert metrics["nearest_candidate_distance_m"]["count"] >= 0


def test_split_and_merge_statistics_use_selected_full_arrays() -> None:
    source = _source()
    split_result = SimpleNamespace(segment_id_by_point=np.array([0, 1, 1, 1], dtype=np.int32))
    records = (
        SimpleNamespace(source_cluster_ids=(10,)),
        SimpleNamespace(source_cluster_ids=(10, 20)),
    )

    assert runner._split_cluster_count(source, split_result) == 1
    assert sum(len(record.source_cluster_ids) > 1 for record in records) == 1


def test_repeated_pipeline_fingerprints_match() -> None:
    source = _source()
    first = runner._run_pipeline(source)[-1]
    second = runner._run_pipeline(source)[-1]
    assert first == second


def test_verify_determinism_reports_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    calls = {"count": 0}

    def flaky_pipeline(tree_input: object) -> tuple[object, object, object, object, str]:
        calls["count"] += 1
        fingerprint = "a" * 64 if calls["count"] == 1 else "b" * 64
        return (tree_input, tree_input, tree_input, tree_input, fingerprint)

    monkeypatch.setattr(runner, "_run_pipeline", flaky_pipeline)

    with pytest.raises(runner.DiagnosticCliError, match="deterministic fingerprint mismatch"):
        runner._run_variant("production_like", source, source, verify_determinism=True)


def test_reconciliation_without_reviewed_summary_is_not_requested(tmp_path: Path) -> None:
    source = _source()
    args = runner.parse_args(_arguments(tmp_path / "result"))
    assert args.reviewed_summary is None

    reconciliation = runner._reconcile_baseline(args=args, source=source, metrics={})

    assert reconciliation == {
        "requested": False,
        "status": "not_requested",
        "reviewed_summary": None,
        "identity_verified": False,
        "comparisons": {},
    }


def test_reconciliation_has_no_scene_simple_implicit_discovery(tmp_path: Path) -> None:
    source = _source()
    scene_simple = replace(source, identity=replace(source.identity, source_id="scene_simple"))
    args = runner.parse_args(_arguments(tmp_path / "result"))
    assert args.reviewed_summary is None

    reconciliation = runner._reconcile_baseline(args=args, source=scene_simple, metrics={})

    assert reconciliation["requested"] is False
    assert reconciliation["status"] == "not_requested"
    assert reconciliation["reviewed_summary"] is None
    assert not hasattr(runner, "_reviewed_summary_path")


def test_reconciliation_accepts_valid_explicit_complete_identity(tmp_path: Path) -> None:
    source = _source()
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(
        json.dumps(_reviewed_summary_payload(source)),
        encoding="utf-8",
    )
    args = runner.parse_args(
        [*_arguments(tmp_path / "result"), "--reviewed-summary", str(reviewed_path)]
    )

    reconciliation = runner._reconcile_baseline(
        args=args, source=source, metrics=_reconciliation_metrics()
    )

    assert reconciliation["requested"] is True
    assert reconciliation["status"] == "passed"
    assert reconciliation["identity_verified"] is True
    assert reconciliation["reviewed_summary"] == str(reviewed_path)
    assert all(entry["matches"] for entry in reconciliation["comparisons"].values())
    assert reconciliation["comparisons"]["topdown_conflict_pixel_count"]["matches"]


def test_reconciliation_rejects_non_object_summary(tmp_path: Path) -> None:
    source = _source()
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    args = runner.parse_args(
        [*_arguments(tmp_path / "result"), "--reviewed-summary", str(reviewed_path)]
    )

    with pytest.raises(runner.DiagnosticCliError, match="JSON object"):
        runner._reconcile_baseline(args=args, source=source, metrics={})


def test_reconciliation_requires_source_identity(tmp_path: Path) -> None:
    source = _source()
    payload = _reviewed_summary_payload(source)
    del payload["source_identity"]
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(json.dumps(payload), encoding="utf-8")
    args = runner.parse_args(
        [*_arguments(tmp_path / "result"), "--reviewed-summary", str(reviewed_path)]
    )

    with pytest.raises(runner.DiagnosticCliError, match="source_identity"):
        runner._reconcile_baseline(args=args, source=source, metrics={})


def test_reconciliation_rejects_identity_with_missing_digest(tmp_path: Path) -> None:
    source = _source()
    payload = _reviewed_summary_payload(source)
    identity = dict(payload["source_identity"])
    del identity["a2a_npz_sha256"]
    payload["source_identity"] = identity
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(json.dumps(payload), encoding="utf-8")
    args = runner.parse_args(
        [*_arguments(tmp_path / "result"), "--reviewed-summary", str(reviewed_path)]
    )

    with pytest.raises(runner.DiagnosticCliError, match="source_identity"):
        runner._reconcile_baseline(args=args, source=source, metrics={})


def test_reconciliation_reports_sha256_identity_mismatch(tmp_path: Path) -> None:
    source = _source()
    payload = _reviewed_summary_payload(source)
    identity = dict(payload["source_identity"])
    identity["v2_npz_sha256"] = "b" * 64
    payload["source_identity"] = identity
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(json.dumps(payload), encoding="utf-8")
    args = runner.parse_args(
        [*_arguments(tmp_path / "result"), "--reviewed-summary", str(reviewed_path)]
    )

    with pytest.raises(runner.DiagnosticCliError, match="v2_npz_sha256") as excinfo:
        runner._reconcile_baseline(args=args, source=source, metrics={})

    message = str(excinfo.value)
    assert "b" * 64 not in message
    assert "a" * 64 not in message


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_reconciliation_rejects_nonstandard_json_constants(
    tmp_path: Path, constant: str
) -> None:
    source = _source()
    payload = _reviewed_summary_payload(source)
    payload["ignored_decorative_field"] = "__CONSTANT__"
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_path.write_text(
        json.dumps(payload).replace('"__CONSTANT__"', constant),
        encoding="utf-8",
    )
    args = runner.parse_args(
        [*_arguments(tmp_path / "result"), "--reviewed-summary", str(reviewed_path)]
    )

    with pytest.raises(runner.DiagnosticCliError, match="reviewed summary is not valid JSON"):
        runner._reconcile_baseline(
            args=args, source=source, metrics=_reconciliation_metrics()
        )


def test_run_writes_exact_layout_and_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source()
    monkeypatch.setattr(runner, "load_tree_segmentation_input_set", lambda *args: source)
    output_dir = tmp_path / "v3-control"

    paths = runner.run(runner.parse_args(_arguments(output_dir)))

    assert output_dir.exists()
    assert {
        "comparison.csv",
        "comparison.json",
        "run_manifest.json",
        "confirmed_only",
        "production_like",
    } == {path.name for path in output_dir.iterdir()}
    for variant in ("production_like", "confirmed_only"):
        assert {
            "tree_segmentation_3d.html",
            "tree_segmentation_diagnostic.png",
            "tree_segmentation_summary.json",
            "tree_segmentation_topdown.png",
        } == {path.name for path in (output_dir / variant).iterdir()}
    assert set(paths) == {
        "production_like_summary",
        "confirmed_only_summary",
        "comparison_json",
        "comparison_csv",
        "run_manifest",
    }


def test_failure_before_publish_leaves_no_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source()
    monkeypatch.setattr(runner, "load_tree_segmentation_input_set", lambda *args: source)

    def fail_render(*args: object) -> object:
        raise OSError("boom")

    monkeypatch.setattr(runner, "_render_variant", fail_render)
    output_dir = tmp_path / "failed-control"

    with pytest.raises(OSError, match="boom"):
        runner.run(runner.parse_args(_arguments(output_dir)))

    assert not output_dir.exists()
