"""Search and gate a crown-like scan_large ROI before an unchanged FAST V4 run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import plotly.graph_objects as go

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.vegetation.fast_multi_tree_roi import (  # noqa: E402
    MULTI_TREE_ROI_SCHEMA,
    MultiTreeRoiCandidate,
    assess_retained_vegetation_purity,
    crown_evidence_mask,
    rank_multi_tree_roi_candidates,
    write_multi_tree_candidate_preview,
)
from algorithms.vegetation.fast_research import (  # noqa: E402
    FastResearchError,
    run_fast_ground_hag,
    run_fast_v4,
)
from algorithms.vegetation.fast_vegetation_ab import (  # noqa: E402
    FastVegetationEvidenceConfig,
    _pdal_covariance_features,
)
from algorithms.vegetation.fast_weighted_preprocess import (  # noqa: E402
    FAST_WEIGHTED_MARKER,
    WeightedWorkingSet,
    build_weighted_fast_vegetation,
    write_weighted_roi_las,
)
from algorithms.vegetation.v4_tree_segmentation import V4ApexQualityFlag  # noqa: E402
from algorithms.vegetation.v4_tree_visualization import run_v4_tree_visualization  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "selected"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fast_research/scan_large_multi_tree_roi"),
    )
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        default=Path("artifacts/fast_research/scan_large_preprocess_weighted_v4/preprocess"),
    )
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--confirm-visual-gate", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), "utf-8")
    os.replace(temporary, path)


def _working_set(preprocess_root: Path) -> WeightedWorkingSet:
    summary = preprocess_root / "scan_large_preprocessing_summary.json"
    payload = json.loads(summary.read_text("utf-8"))
    for key in (
        "source_bounds_min_xyz",
        "source_bounds_max_xyz",
        "source_scales",
        "source_offsets",
    ):
        payload[key] = tuple(float(value) for value in payload[key])
    return WeightedWorkingSet(**payload)


def _search_arrays(ground_las: Path) -> dict[str, np.ndarray]:
    las = laspy.read(ground_las)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    hag = np.asarray(las.HeightAboveGround, dtype=np.float32)
    classification = np.asarray(las.classification, dtype=np.uint8)
    raw_weight = np.asarray(las.raw_count, dtype=np.uint32)
    feature_rows = (classification != 2) & np.isfinite(hag) & (hag >= 0.3)
    features = {
        "linearity": np.zeros(len(xyz), dtype=np.float32),
        "planarity": np.zeros(len(xyz), dtype=np.float32),
        "scattering": np.zeros(len(xyz), dtype=np.float32),
        "verticality": np.zeros(len(xyz), dtype=np.float32),
    }
    if int(np.count_nonzero(feature_rows)) >= 16:
        measured, _elapsed = _pdal_covariance_features(
            xyz[feature_rows], config=FastVegetationEvidenceConfig()
        )
        for name, values in measured.items():
            features[name][feature_rows] = values
    return {
        "xyz_absolute": xyz,
        "hag_m": hag,
        "classification": classification,
        "raw_weight": raw_weight,
        **features,
    }


def _candidate_from_dict(payload: dict[str, Any]) -> MultiTreeRoiCandidate:
    values = dict(payload)
    values["xy_bounds_min"] = tuple(float(value) for value in values["xy_bounds_min"])
    values["xy_bounds_max"] = tuple(float(value) for value in values["xy_bounds_max"])
    values["crown_peak_xy"] = tuple(
        tuple(float(value) for value in pair) for pair in values["crown_peak_xy"]
    )
    return MultiTreeRoiCandidate(**values)


def _run_search(output: Path, working: WeightedWorkingSet) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    search_path = output / "scan_large_multi_tree_roi_search.json"
    if search_path.is_file():
        return json.loads(search_path.read_text("utf-8"))
    ground_status, ground = run_fast_ground_hag(
        Path(working.representative_las_path), output / "compact_ground"
    )
    arrays = _search_arrays(Path(ground.output_las_path))
    candidates, evaluated_count = rank_multi_tree_roi_candidates(
        xyz_absolute=arrays["xyz_absolute"],
        hag_m=arrays["hag_m"],
        raw_weight=arrays["raw_weight"],
        classification=arrays["classification"],
        linearity=arrays["linearity"],
        planarity=arrays["planarity"],
        scattering=arrays["scattering"],
    )
    evidence = crown_evidence_mask(
        hag_m=arrays["hag_m"],
        classification=arrays["classification"],
        linearity=arrays["linearity"],
        planarity=arrays["planarity"],
        scattering=arrays["scattering"],
    )
    previews: list[str] = []
    for candidate in candidates[:3]:
        preview = output / f"scan_large_roi_candidate_{candidate.rank:02d}.html"
        write_multi_tree_candidate_preview(
            candidate,
            xyz_absolute=arrays["xyz_absolute"],
            hag_m=arrays["hag_m"],
            classification=arrays["classification"],
            crown_evidence=evidence,
            output_html=preview,
        )
        previews.append(str(preview))
    payload = {
        "schema_version": MULTI_TREE_ROI_SCHEMA,
        "non_certified_marker": FAST_WEIGHTED_MARKER,
        "preprocessing_architecture": "REUSED_VALIDATED_WEIGHTED_WORKING_SET",
        "v4_semantics_changed": False,
        "compact_ground_status": ground_status,
        "compact_ground": ground.to_dict(),
        "candidate_count_evaluated": evaluated_count,
        "candidate_count_ranked": len(candidates),
        "crown_evidence_formula": (
            "NON_GROUND AND HAG_GE_1.5 AND SCATTERING_GE_0.18 "
            "AND LINEARITY_LE_0.70 AND PLANARITY_LE_0.75"
        ),
        "ranking": "MULTI_CROWN_LIKELIHOOD_WITH_RIGID_GEOMETRY_PENALTY",
        "candidates": [candidate.to_dict() for candidate in candidates],
        "candidate_previews": previews,
    }
    _write_json(search_path, payload)
    return payload


def _selected_roi_payload(
    *, working: WeightedWorkingSet, candidate: MultiTreeRoiCandidate
) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_absolute"], dtype=np.float64)
        ids = np.asarray(stream["representative_id"], dtype=np.uint32)
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    lower = np.asarray(candidate.xy_bounds_min, dtype=np.float64)
    upper = np.asarray(candidate.xy_bounds_max, dtype=np.float64)
    selected = (
        (xyz[:, 0] >= lower[0])
        & (xyz[:, 0] < upper[0])
        & (xyz[:, 1] >= lower[1])
        & (xyz[:, 1] < upper[1])
    )
    selected_ids = ids[selected]
    if len(selected_ids) != candidate.representative_count:
        raise FastResearchError("FAST_MULTI_TREE_SELECTED_ROI_CORRESPONDENCE_INVALID")
    payload = {
        "schema_version": MULTI_TREE_ROI_SCHEMA,
        "non_certified_marker": FAST_WEIGHTED_MARKER,
        "selected_candidate_rank": candidate.rank,
        "xy_bounds_min": list(candidate.xy_bounds_min),
        "xy_bounds_max": list(candidate.xy_bounds_max),
        "representative_count": int(np.count_nonzero(selected)),
        "raw_weight_count": int(np.sum(raw_count[selected], dtype=np.int64)),
        "visible_crown_estimate": f"approximately {candidate.crown_peak_count} crown-like peaks",
        "selection_reason": (
            "highest deterministic multi-crown-likelihood candidate passing the "
            "low-rigidity, non-elongation automatic gate; visually reviewed before V4"
        ),
        "candidate_metrics": candidate.to_dict(),
    }
    return payload, selected_ids


def _write_vegetation_preview(diagnostic_npz: Path, output_html: Path) -> Path:
    with np.load(diagnostic_npz, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
    figure = go.Figure()
    labels = {
        1: ("ground excluded", "#8c8c8c"),
        4: ("rigid linear excluded", "#756bb1"),
        5: ("rigid surface excluded", "#54278f"),
        6: ("no vegetation evidence", "#bdbdbd"),
        7: ("retained vegetation", "#31a354"),
    }
    for code, (label, color) in labels.items():
        rows = np.flatnonzero(state == code)
        if len(rows) > 30_000:
            rows = rows[np.linspace(0, len(rows) - 1, 30_000, dtype=np.int64)]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[rows, 0],
                y=xyz[rows, 1],
                z=xyz[rows, 2],
                mode="markers",
                name=label,
                marker={"size": 2, "color": color, "opacity": 0.7},
            )
        )
    figure.update_layout(
        title="Selected multi-tree ROI — vegetation purity gate",
        scene={"aspectmode": "data"},
    )
    figure.write_html(output_html, include_plotlyjs=True, full_html=True)
    return output_html


def _write_clean_v4_overview(v4_root: Path, output_html: Path) -> Path:
    with np.load(v4_root / "v4_tree_segmentation.npz", allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        tree = np.asarray(stream["tree_id_by_point"], dtype=np.int32)
    payload = json.loads((v4_root / "v4_tree_segmentation.json").read_text("utf-8"))
    figure = go.Figure()
    for tree_id in sorted(int(value) for value in np.unique(tree) if value >= 0):
        rows = np.flatnonzero(tree == tree_id)
        figure.add_trace(
            go.Scatter3d(
                x=xyz[rows, 0],
                y=xyz[rows, 1],
                z=xyz[rows, 2],
                mode="markers",
                name=f"tree {tree_id}",
                marker={"size": 2},
            )
        )
    apex = [record["xyz_local_m"] for record in payload["apex_records"]]
    if apex:
        apex_xyz = np.asarray(apex, dtype=np.float64)
        figure.add_trace(
            go.Scatter3d(
                x=apex_xyz[:, 0],
                y=apex_xyz[:, 1],
                z=apex_xyz[:, 2],
                mode="markers",
                name="apex candidates",
                marker={"size": 7, "color": "#e31a1c", "symbol": "diamond"},
            )
        )
    figure.update_layout(
        title="Selected multi-tree ROI — current RAW-WEIGHTED V4",
        scene={"aspectmode": "data"},
    )
    figure.write_html(output_html, include_plotlyjs=True, full_html=True)
    return output_html


def _v4_diagnostics(
    *,
    candidate: MultiTreeRoiCandidate,
    v4_root: Path,
    purity: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads((v4_root / "v4_tree_segmentation.json").read_text("utf-8"))
    records = payload["tree_records"]
    suppressed_flag = int(V4ApexQualityFlag.SUPPRESSED)
    apex_records = payload["apex_records"]
    accepted = sum(
        not bool(int(record["quality_flags"]) & suppressed_flag) for record in apex_records
    )
    overmerge = any(
        len(record["secondary_apex_candidate_ids"]) > 0
        or max(
            float(record["geometry_summary"]["xy_bounds_max"][axis])
            - float(record["geometry_summary"]["xy_bounds_min"][axis])
            for axis in (0, 1)
        )
        > 12.0
        for record in records
    )
    centroids = [record["geometry_summary"]["xy_centroid"][:2] for record in records]
    oversplit = any(
        np.hypot(first[0] - second[0], first[1] - second[1]) < 2.0
        for index, first in enumerate(centroids)
        for second in centroids[index + 1 :]
    )
    if purity["quality"] == "CLEAN" and (overmerge or oversplit):
        justified = "YES"
    elif purity["quality"] == "CLEAN":
        justified = "NO"
    else:
        justified = "INCONCLUSIVE"
    return {
        "visible_crown_estimate": f"approximately {candidate.crown_peak_count} crown-like peaks",
        "apex_candidates": len(apex_records),
        "accepted_apices": accepted,
        "suppressed_apices": len(apex_records) - accepted,
        "v4_tree_count": len(records),
        "overmerge_suspected": overmerge,
        "oversplit_suspected": oversplit,
        "v4_algorithm_change_justified": justified,
        "v4_summary": {
            "tree_records": records,
            "result_digest": payload["result_digest"],
        },
    }


def _run_selected(
    output: Path, working: WeightedWorkingSet, args: argparse.Namespace
) -> dict[str, Any]:
    if not args.confirm_visual_gate:
        raise FastResearchError("FAST_MULTI_TREE_VISUAL_GATE_REQUIRED")
    search = json.loads((output / "scan_large_multi_tree_roi_search.json").read_text("utf-8"))
    matched = [item for item in search["candidates"] if item["rank"] == args.candidate_rank]
    if len(matched) != 1:
        raise FastResearchError("FAST_MULTI_TREE_CANDIDATE_RANK_INVALID")
    candidate = _candidate_from_dict(matched[0])
    if not candidate.automatic_gate_pass:
        raise FastResearchError("FAST_MULTI_TREE_AUTOMATIC_GATE_FAILED")
    selected_bounds, representative_ids = _selected_roi_payload(
        working=working, candidate=candidate
    )
    bounds_path = output / "selected_multi_tree_roi_bounds.json"
    if bounds_path.exists():
        existing = json.loads(bounds_path.read_text("utf-8"))
        if existing != selected_bounds:
            raise FastResearchError("FAST_MULTI_TREE_SELECTED_BOUNDS_EXISTS")
    else:
        _write_json(bounds_path, selected_bounds)
    roi = {
        "representative_ids": [int(value) for value in representative_ids],
        "xy_bounds_min": list(candidate.xy_bounds_min),
        "xy_bounds_max": list(candidate.xy_bounds_max),
        "z_bounds_min": float(working.source_bounds_min_xyz[2]),
        "z_bounds_max": float(working.source_bounds_max_xyz[2]),
    }
    roi_las = write_weighted_roi_las(working, roi, output / "selected_computation")
    _ground_status, ground = run_fast_ground_hag(roi_las, output / "selected_ground")
    vegetation, vegetation_manifest = build_weighted_fast_vegetation(
        working, roi, ground, output / "selected_vegetation"
    )
    vegetation_preview = _write_vegetation_preview(
        output / "selected_vegetation" / "scan_large_fast_vegetation_diagnostics.npz",
        output / "selected_multi_tree_vegetation_review.html",
    )
    with np.load(
        output / "selected_vegetation" / "scan_large_fast_vegetation_diagnostics.npz",
        allow_pickle=False,
    ) as stream:
        purity = assess_retained_vegetation_purity(
            diagnostic_state=np.asarray(stream["diagnostic_state"], dtype=np.uint8),
            linearity=np.asarray(stream["linearity"], dtype=np.float32),
            planarity=np.asarray(stream["planarity"], dtype=np.float32),
            scattering=np.asarray(stream["scattering"], dtype=np.float32),
        )
    diagnostics_path = output / "selected_multi_tree_v4_diagnostics.json"
    common = {
        "schema_version": MULTI_TREE_ROI_SCHEMA,
        "non_certified_marker": FAST_WEIGHTED_MARKER,
        "selected_roi": selected_bounds,
        "vegetation_manifest": vegetation_manifest,
        "vegetation_quality": purity,
        "v4_semantics_changed": False,
        "outputs": {"vegetation_review": str(vegetation_preview)},
    }
    if purity["quality"] == "HEAVILY_CONTAMINATED":
        _write_json(
            diagnostics_path,
            {
                **common,
                "status": "BLOCKED_HEAVILY_CONTAMINATED_BEFORE_V4",
                "v4_algorithm_change_justified": "INCONCLUSIVE",
            },
        )
        return json.loads(diagnostics_path.read_text("utf-8"))
    _v4_status, v4_root, _summary = run_fast_v4(
        vegetation,
        output / "selected_v4",
        source_bounds_min_xyz=working.source_bounds_min_xyz,
        source_bounds_max_xyz=working.source_bounds_max_xyz,
    )
    v4_review = output / "selected_multi_tree_v4_review.html"
    v4_review_json = output / "selected_multi_tree_v4_review.json"
    run_v4_tree_visualization(v4_root, v4_review, v4_review_json, overwrite=False)
    clean_overview = _write_clean_v4_overview(
        v4_root, output / "selected_multi_tree_v4_clean_overview.html"
    )
    diagnostics = {
        **common,
        "status": "PASS",
        **_v4_diagnostics(candidate=candidate, v4_root=v4_root, purity=purity),
        "outputs": {
            "vegetation_review": str(vegetation_preview),
            "v4_review": str(v4_review),
            "v4_clean_overview": str(clean_overview),
            "v4_review_json": str(v4_review_json),
        },
    }
    _write_json(diagnostics_path, diagnostics)
    return diagnostics


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    output = args.output.resolve()
    working = _working_set(args.preprocess_root.resolve())
    if args.phase == "search":
        return _run_search(output, working)
    return _run_selected(output, working, args)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
