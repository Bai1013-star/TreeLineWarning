"""Run a non-production confirmed-only control against the V3 tree pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.vegetation.canopy_height_model import (  # noqa: E402
    CanopyHeightModelError,
    build_canopy_height_model,
)
from algorithms.vegetation.canopy_segmentation import (  # noqa: E402
    CanopySegmentationError,
    segment_canopy_regions,
)
from algorithms.vegetation.tree_inputs import (  # noqa: E402
    TreeSegmentationInputError,
    load_tree_segmentation_input_set,
)
from algorithms.vegetation.tree_segmentation import (  # noqa: E402
    TreeResultConstructionError,
    build_tree_segmentation_result,
)
from algorithms.vegetation.tree_visualization import (  # noqa: E402
    render_3d_html,
    render_diagnostic_png,
    render_topdown_png,
    topdown_pixel_statistics,
    validate_input_result_alignment,
    validate_maximum_3d_points,
)
from algorithms.vegetation.treetop_candidates import (  # noqa: E402
    TreetopDetectionError,
    detect_treetop_candidates,
)
from core.contracts.tree_inputs import (  # noqa: E402
    TreeSegmentationInputClusterDisposition,
    TreeSegmentationInputDisposition,
    TreeSegmentationInputIdentity,
    TreeSegmentationInputSet,
)
from core.contracts.vegetation_instances import VegetationClusterStatus  # noqa: E402

_DIAGNOSTIC_ROOT = _REPOSITORY_ROOT / "outputs" / "diagnostics"
_DISTANCE_BIN_EDGES_M = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
_RECONCILIATION_FIELDS = (
    ("source_point_count", "source_point_count"),
    ("assigned_point_count", "assigned_point_count"),
    ("unassigned_point_count", "unassigned_point_count"),
    ("candidate_count", "candidate_count"),
    ("record_count", "segment_record_count"),
    ("chm_shape", "chm_shape"),
    ("ambiguity_cell_count", "ambiguous_cell_count"),
    ("topdown_conflict_pixel_count", "conflict_pixel_count"),
)


class DiagnosticCliError(RuntimeError):
    """Expected, concise failure for the bounded diagnostic CLI."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve(strict=False)
    diagnostic_root = _DIAGNOSTIC_ROOT.resolve(strict=False)
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=False)
    permitted = _is_relative_to(resolved, diagnostic_root) or _is_relative_to(
        resolved, temporary_root
    )
    if not permitted:
        raise DiagnosticCliError(
            "--output-dir must be under outputs/diagnostics or the system temporary directory"
        )
    if resolved.exists():
        raise DiagnosticCliError("--output-dir already exists; refusing to overwrite diagnostics")
    return resolved


def _parse_maximum_3d_points(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise DiagnosticCliError(
            "--max-3d-points must be an integer from 1 through 200000"
        ) from error
    try:
        return validate_maximum_3d_points(parsed)
    except ValueError as error:
        raise DiagnosticCliError(f"--max-3d-points {error}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-npz", type=Path, required=True)
    parser.add_argument("--v2-json", type=Path, required=True)
    parser.add_argument("--a2a-npz", type=Path, required=True)
    parser.add_argument("--a2a-json", type=Path, required=True)
    parser.add_argument("--a2b-npz", type=Path, required=True)
    parser.add_argument("--a2b-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed-summary", type=Path)
    parser.add_argument("--max-3d-points", default="200000")
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args(argv)
    args.max_3d_points = _parse_maximum_3d_points(args.max_3d_points)
    return args


def confirmed_only_projection(source: TreeSegmentationInputSet) -> TreeSegmentationInputSet:
    """Return an ephemeral contract-valid view that excludes review-only clusters.

    ``REJECTED_AS_INSTANCE`` is used solely as the already-supported excluded state
    required by the current V3 contracts.  This function never changes or writes
    the loaded V2 arrays and must not be used as a V2 reclassification operation.
    """
    if type(source) is not TreeSegmentationInputSet:
        raise TypeError("source must be a TreeSegmentationInputSet")

    review_ids = np.asarray(source.review_only_cluster_ids, dtype=np.int32)
    point_status = np.array(source.vegetation_points.cluster_status_by_point, copy=True)
    cluster_status = np.array(source.cluster_status, copy=True)
    if len(review_ids):
        point_status[np.isin(source.vegetation_points.cluster_id_by_point, review_ids)] = int(
            VegetationClusterStatus.REJECTED_AS_INSTANCE
        )
        cluster_status[np.isin(source.cluster_id, review_ids)] = int(
            VegetationClusterStatus.REJECTED_AS_INSTANCE
        )
    point_status.setflags(write=False)
    cluster_status.setflags(write=False)

    excluded_ids = tuple(sorted((*source.excluded_cluster_ids, *source.review_only_cluster_ids)))
    excluded_set = set(excluded_ids)
    dispositions = tuple(
        TreeSegmentationInputClusterDisposition(
            cluster_id=int(cluster_id),
            disposition=(
                TreeSegmentationInputDisposition.EXCLUDED
                if int(cluster_id) in excluded_set
                else TreeSegmentationInputDisposition.ELIGIBLE
            ),
            reason=(
                "rejected_as_instance"
                if int(cluster_id) in excluded_set
                else "confirmed_distance_ready"
            ),
        )
        for cluster_id in sorted(int(value) for value in source.cluster_id)
    )
    projected_points = replace(source.vegetation_points, cluster_status_by_point=point_status)
    return TreeSegmentationInputSet(
        vegetation_points=projected_points,
        geometry_features=source.geometry_features,
        heights_above_ground=source.heights_above_ground,
        cluster_id=source.cluster_id,
        cluster_status=cluster_status,
        cluster_vegetation_id=source.cluster_vegetation_id,
        cluster_member_offsets=source.cluster_member_offsets,
        ordered_cluster_member_row_index=source.ordered_cluster_member_row_index,
        identity=source.identity,
        eligible_cluster_ids=source.eligible_cluster_ids,
        review_only_cluster_ids=(),
        excluded_cluster_ids=excluded_ids,
        cluster_dispositions=dispositions,
        ground_usable_by_point=source.ground_usable_by_point,
        usable_ground_member_count_by_cluster=source.usable_ground_member_count_by_cluster,
    )


def _hash_array(digest: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))


def _record_payload(record: Any) -> dict[str, Any]:
    return {
        "tree_id": record.tree_id,
        "segment_id": record.segment_id,
        "status": record.status.value,
        "source_cluster_ids": list(record.source_cluster_ids),
        "source_cluster_statuses": [status.name for status in record.source_cluster_statuses],
        "source_cluster_distance_ready": list(record.source_cluster_distance_ready),
        "tree_distance_ready": record.tree_distance_ready,
        "confidence": record.confidence,
        "quality_flags": int(record.quality_flags),
    }


def deterministic_fingerprint(
    canopy_model: Any, candidates: Any, segmentation: Any, result: Any
) -> str:
    """Hash the required deterministic V3 stage outputs in a canonical order."""
    digest = hashlib.sha256()
    for prefix, value in (
        ("chm", canopy_model),
        ("candidates", candidates),
        ("segmentation", segmentation),
    ):
        for field in fields(value):
            field_value = getattr(value, field.name)
            if isinstance(field_value, np.ndarray):
                _hash_array(digest, f"{prefix}.{field.name}", field_value)
    for name in (
        "segment_id_by_point",
        "tree_id_by_point",
        "tree_distance_ready_by_point",
        "tree_member_offsets",
        "ordered_tree_member_row_index",
    ):
        _hash_array(digest, f"result.{name}", getattr(result, name))
    records = [_record_payload(record) for record in result.records]
    digest.update(json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(len(data)),
        "minimum": float(np.min(data)),
        "p25": float(np.percentile(data, 25)),
        "median": float(np.percentile(data, 50)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def nearest_candidate_distance_distribution(candidates: Any) -> dict[str, Any]:
    """Summarize each candidate's nearest other candidate in local metres."""
    count = len(candidates.candidate_id)
    if count < 2:
        return {
            **_distribution([]),
            "bin_edges_m": list(_DISTANCE_BIN_EDGES_M),
            "final_bin": ">=10.0",
            "bin_counts": [0] * 6,
        }
    xy = candidates.local_xy.astype(np.float64, copy=False)
    delta = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    histogram_edges = np.asarray((*_DISTANCE_BIN_EDGES_M, np.inf), dtype=np.float64)
    counts, _ = np.histogram(nearest, bins=histogram_edges)
    return {
        **_distribution(nearest),
        "bin_edges_m": list(_DISTANCE_BIN_EDGES_M),
        "final_bin": ">=10.0",
        "bin_counts": [int(value) for value in counts],
    }


def _selected_cluster_ids(tree_input: TreeSegmentationInputSet) -> tuple[int, ...]:
    return (*tree_input.eligible_cluster_ids, *tree_input.review_only_cluster_ids)


def _split_cluster_count(tree_input: TreeSegmentationInputSet, result: Any) -> int:
    point_clusters = tree_input.vegetation_points.cluster_id_by_point
    segment_ids = result.segment_id_by_point
    count = 0
    for cluster_id in _selected_cluster_ids(tree_input):
        assigned = segment_ids[(point_clusters == cluster_id) & (segment_ids >= 0)]
        if len(np.unique(assigned)) > 1:
            count += 1
    return count


def _segment_distributions(
    tree_input: TreeSegmentationInputSet, result: Any
) -> dict[str, dict[str, Any]]:
    segment_ids = result.segment_id_by_point
    hag = tree_input.heights_above_ground.height_agl_m
    xy = tree_input.vegetation_points.xyz_local[:, :2].astype(np.float64, copy=False)
    point_counts: list[float] = []
    segment_hag: list[float] = []
    horizontal_extent: list[float] = []
    for segment_id in range(len(result.records)):
        positions = np.flatnonzero(segment_ids == segment_id)
        point_counts.append(float(len(positions)))
        if not len(positions):
            continue
        segment_hag.extend(float(value) for value in hag[positions])
        extent = np.max(xy[positions], axis=0) - np.min(xy[positions], axis=0)
        horizontal_extent.append(float(np.hypot(extent[0], extent[1])))
    return {
        "segment_point_count": _distribution(point_counts),
        "segment_hag_m": _distribution(segment_hag),
        "segment_horizontal_bbox_diagonal_m": _distribution(horizontal_extent),
    }


def collect_metrics(
    *,
    variant: str,
    tree_input: TreeSegmentationInputSet,
    original_input: TreeSegmentationInputSet,
    canopy_model: Any,
    candidates: Any,
    segmentation: Any,
    result: Any,
    fingerprint: str,
    determinism_verified: bool,
) -> dict[str, Any]:
    """Collect diagnostic-only metrics from complete, unsampled runtime arrays."""
    source_count = validate_input_result_alignment(tree_input, result)
    selected_ids = np.asarray(_selected_cluster_ids(tree_input), dtype=np.int32)
    point_clusters = tree_input.vegetation_points.cluster_id_by_point
    raw_selected = int(np.count_nonzero(np.isin(point_clusters, selected_ids)))
    original_review = np.asarray(original_input.review_only_cluster_ids, dtype=np.int32)
    control_excluded_points = (
        int(np.count_nonzero(np.isin(point_clusters, original_review)))
        if variant == "confirmed_only"
        else 0
    )
    segment_ids = result.segment_id_by_point
    assigned = int(np.count_nonzero(segment_ids >= 0))
    audit = result.input_provenance.get("mapping_audit", {})
    topdown = topdown_pixel_statistics(tree_input, result)
    distributions = _segment_distributions(tree_input, result)
    return {
        "variant": variant,
        "projection_label": (
            "diagnostic_excluded_review_only" if variant == "confirmed_only" else "production_like"
        ),
        "source_point_count": source_count,
        "included_v2_cluster_count": int(len(selected_ids)),
        "raw_selected_member_point_count": raw_selected,
        "review_only_cluster_count_excluded_by_control": (
            len(original_input.review_only_cluster_ids) if variant == "confirmed_only" else 0
        ),
        "review_only_member_point_count_excluded_by_control": control_excluded_points,
        "chm_included_point_count": int(canopy_model.included_point_count),
        "chm_valid_cell_count": int(np.count_nonzero(canopy_model.occupied_mask)),
        "chm_shape": [int(value) for value in canopy_model.canopy_height_m.shape],
        "candidate_count": int(len(candidates.candidate_id)),
        "candidate_peak_below_2m_count": int(np.count_nonzero(candidates.peak_height_m < 2.0)),
        "nearest_candidate_distance_m": nearest_candidate_distance_distribution(candidates),
        "assigned_point_count": assigned,
        "unassigned_point_count": source_count - assigned,
        "record_count": int(len(result.records)),
        "ambiguity_cell_count": int(segmentation.ambiguous_cell_count),
        "ambiguity_assigned_point_count": int(audit.get("ambiguous_assigned_point_count", 0)),
        "topdown_conflict_pixel_count": int(topdown["conflict_pixel_count"]),
        "selected_source_clusters_split_across_v3_segments": _split_cluster_count(
            tree_input, result
        ),
        "v3_records_with_multiple_selected_source_clusters": int(
            sum(len(record.source_cluster_ids) > 1 for record in result.records)
        ),
        "distributions": distributions,
        "deterministic_fingerprint_sha256": fingerprint,
        "determinism_verified": determinism_verified,
        "topdown_pixel_statistics": {name: int(value) for name, value in topdown.items()},
    }


def _run_pipeline(tree_input: TreeSegmentationInputSet) -> tuple[Any, Any, Any, Any, str]:
    canopy_model = build_canopy_height_model(tree_input)
    candidates = detect_treetop_candidates(canopy_model)
    segmentation = segment_canopy_regions(canopy_model, candidates)
    result = build_tree_segmentation_result(tree_input, canopy_model, candidates, segmentation)
    validate_input_result_alignment(tree_input, result)
    if np.any(result.tree_id_by_point != -1) or np.any(result.tree_distance_ready_by_point):
        raise DiagnosticCliError("diagnostic result unexpectedly created permanent tree readiness")
    fingerprint = deterministic_fingerprint(canopy_model, candidates, segmentation, result)
    return canopy_model, candidates, segmentation, result, fingerprint


def _run_variant(
    variant: str,
    tree_input: TreeSegmentationInputSet,
    original_input: TreeSegmentationInputSet,
    verify_determinism: bool,
) -> dict[str, Any]:
    canopy_model, candidates, segmentation, result, fingerprint = _run_pipeline(tree_input)
    if verify_determinism:
        repeated = _run_pipeline(tree_input)[-1]
        if fingerprint != repeated:
            raise DiagnosticCliError(f"{variant} deterministic fingerprint mismatch")
    metrics = collect_metrics(
        variant=variant,
        tree_input=tree_input,
        original_input=original_input,
        canopy_model=canopy_model,
        candidates=candidates,
        segmentation=segmentation,
        result=result,
        fingerprint=fingerprint,
        determinism_verified=verify_determinism,
    )
    return {
        "tree_input": tree_input,
        "canopy_model": canopy_model,
        "candidates": candidates,
        "segmentation": segmentation,
        "result": result,
        "metrics": metrics,
    }


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "v2_npz": args.v2_npz,
        "v2_json": args.v2_json,
        "a2a_npz": args.a2a_npz,
        "a2a_json": args.a2a_json,
        "a2b_npz": args.a2b_npz,
        "a2b_json": args.a2b_json,
    }


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN/Infinity/-Infinity so the summary parses as strict JSON.

    Python's ``json.loads`` accepts these non-standard constants by default.
    ``parse_constant`` is invoked for every such token anywhere in the
    document, so raising here rejects them even in ignored top-level fields.
    """
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _reconcile_baseline(
    *, args: argparse.Namespace, source: TreeSegmentationInputSet, metrics: dict[str, Any]
) -> dict[str, Any]:
    """Reconcile the production-like run against an explicit reviewed baseline.

    Reconciliation is opt-in: without ``--reviewed-summary`` the returned state
    is ``not_requested`` and no source is consulted, so no scene id has any
    special implicit behavior.  With a summary, its ``source_identity`` must
    parse into a complete typed identity and match ``source.identity`` before
    any metric comparison is attempted.  The summary itself must be strict
    JSON; non-standard constants such as NaN and Infinity are rejected.
    """
    reviewed_path = args.reviewed_summary
    if reviewed_path is None:
        return {
            "requested": False,
            "status": "not_requested",
            "reviewed_summary": None,
            "identity_verified": False,
            "comparisons": {},
        }
    try:
        reviewed_text = reviewed_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DiagnosticCliError(
            f"cannot read reviewed baseline summary: {reviewed_path}"
        ) from error
    try:
        reviewed = json.loads(reviewed_text, parse_constant=_reject_nonstandard_json_constant)
    except ValueError as error:
        raise DiagnosticCliError("reviewed summary is not valid JSON") from error
    if not isinstance(reviewed, dict):
        raise DiagnosticCliError("reviewed baseline summary must be a JSON object")
    identity_data = reviewed.get("source_identity")
    if not isinstance(identity_data, dict):
        raise DiagnosticCliError("reviewed baseline summary is missing source_identity")
    try:
        reviewed_identity = TreeSegmentationInputIdentity.from_dict(identity_data)
    except (TypeError, ValueError) as error:
        raise DiagnosticCliError(
            f"reviewed baseline source_identity is invalid: {error}"
        ) from error
    mismatched = sorted(
        name
        for name in TreeSegmentationInputIdentity.__dataclass_fields__
        if getattr(reviewed_identity, name) != getattr(source.identity, name)
    )
    if mismatched:
        raise DiagnosticCliError(
            "reviewed baseline identity does not match the loaded source: "
            + ", ".join(mismatched)
        )
    comparisons: dict[str, dict[str, Any]] = {}
    passed = True
    for actual_key, reviewed_key in _RECONCILIATION_FIELDS:
        actual = metrics[actual_key]
        expected = reviewed.get(reviewed_key)
        matches = actual == expected
        comparisons[actual_key] = {"expected": expected, "actual": actual, "matches": matches}
        passed = passed and matches
    if not passed:
        raise DiagnosticCliError("production-like run did not reproduce the reviewed baseline")
    return {
        "requested": True,
        "status": "passed",
        "reviewed_summary": str(reviewed_path),
        "identity_verified": True,
        "comparisons": comparisons,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_variant(
    variant_dir: Path,
    final_dir: Path,
    run_data: dict[str, Any],
    maximum_3d_points: int,
) -> dict[str, Any]:
    paths = {
        "topdown_png": variant_dir / "tree_segmentation_topdown.png",
        "diagnostic_png": variant_dir / "tree_segmentation_diagnostic.png",
        "interactive_html": variant_dir / "tree_segmentation_3d.html",
        "summary_json": variant_dir / "tree_segmentation_summary.json",
    }
    render_topdown_png(run_data["tree_input"], run_data["result"], paths["topdown_png"])
    render_diagnostic_png(
        run_data["canopy_model"],
        run_data["candidates"],
        run_data["segmentation"],
        paths["diagnostic_png"],
    )
    _, displayed = render_3d_html(
        run_data["tree_input"],
        run_data["result"],
        paths["interactive_html"],
        maximum_points=maximum_3d_points,
    )
    summary = {
        "diagnostic_control": True,
        "not_a_production_reclassification": True,
        "html_displayed_point_count": displayed,
        "html_maximum_point_count": maximum_3d_points,
        "output_paths": {
            name: str(final_dir / path.relative_to(variant_dir.parent))
            for name, path in paths.items()
        },
        "metrics": run_data["metrics"],
    }
    _write_json(paths["summary_json"], summary)
    return summary


def _comparison_rows(production: dict[str, Any], confirmed: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = (
        "included_v2_cluster_count",
        "raw_selected_member_point_count",
        "review_only_cluster_count_excluded_by_control",
        "review_only_member_point_count_excluded_by_control",
        "chm_included_point_count",
        "chm_valid_cell_count",
        "candidate_count",
        "candidate_peak_below_2m_count",
        "assigned_point_count",
        "unassigned_point_count",
        "record_count",
        "ambiguity_cell_count",
        "ambiguity_assigned_point_count",
        "topdown_conflict_pixel_count",
        "selected_source_clusters_split_across_v3_segments",
        "v3_records_with_multiple_selected_source_clusters",
    )
    return [
        {
            "metric": name,
            "production_like": production[name],
            "confirmed_only": confirmed[name],
            "delta_confirmed_minus_production": confirmed[name] - production[name],
        }
        for name in metrics
    ]


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["metric"])
        writer.writeheader()
        writer.writerows(rows)


def _manifest(
    source: TreeSegmentationInputSet,
    args: argparse.Namespace,
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "diagnostic_name": "v3_confirmed_only_control",
        "source_identity": source.identity.to_dict(),
        "source_artifact_paths": {name: str(path) for name, path in _source_paths(args).items()},
        "verify_determinism": bool(args.verify_determinism),
        "projection": {
            "label": "diagnostic_excluded_review_only",
            "ephemeral": True,
            "not_a_production_reclassification": True,
            "serialized_as_production_artifact": False,
        },
        "interpretation_limits": [
            "This is a diagnostic control, not a production reclassification.",
            "No segmentation-accuracy claim is made without manual labels.",
            "Fewer segments are not automatically better.",
            "A cleaner image is not proof of correct single-tree segmentation.",
            "No thresholds or formal production semantics were changed.",
        ],
        "production_like_reconciliation": reconciliation,
    }


def _publish(stage_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise DiagnosticCliError("final diagnostic directory appeared during execution")
    os.replace(stage_dir, final_dir)


def run(args: argparse.Namespace) -> dict[str, Path]:
    final_dir = _validate_output_dir(args.output_dir)
    try:
        maximum_3d_points = validate_maximum_3d_points(args.max_3d_points)
    except ValueError as error:
        raise DiagnosticCliError(f"--max-3d-points {error}") from error
    source = load_tree_segmentation_input_set(
        args.v2_npz,
        args.v2_json,
        args.a2a_npz,
        args.a2a_json,
        args.a2b_npz,
        args.a2b_json,
    )
    control = confirmed_only_projection(source)
    production = _run_variant("production_like", source, source, args.verify_determinism)
    confirmed = _run_variant("confirmed_only", control, source, args.verify_determinism)
    reconciliation = _reconcile_baseline(args=args, source=source, metrics=production["metrics"])

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=final_dir.parent))
    try:
        production_summary = _render_variant(
            stage_dir / "production_like", final_dir, production, maximum_3d_points
        )
        confirmed_summary = _render_variant(
            stage_dir / "confirmed_only", final_dir, confirmed, maximum_3d_points
        )
        rows = _comparison_rows(production["metrics"], confirmed["metrics"])
        comparison = {
            "production_like": production_summary,
            "confirmed_only": confirmed_summary,
            "numeric_comparison": rows,
            "interpretation": {
                "accuracy_claim": "none without manual labels",
                "fewer_segments_automatically_better": False,
                "cleaner_images_prove_correct_trees": False,
            },
            "production_like_reconciliation": reconciliation,
        }
        _write_json(stage_dir / "comparison.json", comparison)
        _write_comparison_csv(stage_dir / "comparison.csv", rows)
        _write_json(stage_dir / "run_manifest.json", _manifest(source, args, reconciliation))
        _publish(stage_dir, final_dir)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise
    return {
        "production_like_summary": final_dir / "production_like" / "tree_segmentation_summary.json",
        "confirmed_only_summary": final_dir / "confirmed_only" / "tree_segmentation_summary.json",
        "comparison_json": final_dir / "comparison.json",
        "comparison_csv": final_dir / "comparison.csv",
        "run_manifest": final_dir / "run_manifest.json",
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        paths = run(args)
    except (
        DiagnosticCliError,
        TreeSegmentationInputError,
        CanopyHeightModelError,
        TreetopDetectionError,
        CanopySegmentationError,
        TreeResultConstructionError,
        OSError,
    ) as error:
        print(f"V3 confirmed-only diagnostic failed: {error}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
