"""POLE_SPAN_MATURE_BASELINE: pole-first / span-based FAST mature orchestration.

The pipeline follows the electrical-corridor topology end to end:

    FULL-RESOLUTION LAS
        -> coarse 0.25 m representatives (routing / Ground/HAG only)
        -> pole-like candidates + coarse wire evidence
        -> pole <-> wire mutual validation -> confirmed/likely poles
        -> corridor topology, deterministic line ordering, spans (core+halo)
        -> per-span wire reconstruction (PCA -> DBSCAN -> strand -> curve -> W1)
        -> per-span tree segmentation (CHM -> smoothing -> adaptive tops ->
           marker watershed -> 3D membership -> full-resolution back-map)
        -> cross-span reconciliation + global tree ids
        -> exact Tree-Wire clearance (reuses the reviewed exact mathematics)
        -> per-span risk report + 16 review visualizations

Every stage is digest-checkpointed (source SHA-256, source identity, config
digest, schema version, upstream digest) so a downstream failure never forces
re-running validated upstream stages on the 105M-point source.  This pipeline
is FAST / NON_CERTIFIED / research demonstration; it never claims Formal
authority.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import (
    FAST_BASELINE_SCHEMA_VERSION,
    NON_CERTIFIED_MARKER,
    FastBaselineConfig,
    canonical_digest,
    default_fast_baseline_config,
)

from algorithms.poles.candidates import (
    PoleCandidateResult,
    detect_pole_candidates,
    pole_candidates_to_dict,
)
from algorithms.poles.evidence import (
    GroundEvidence,
    load_ground_evidence,
)
from algorithms.poles.topology import (
    SpanTopologyResult,
    build_span_topology,
    topology_to_dict,
)
from algorithms.poles.validation import (
    PoleValidationResult,
    pole_validation_to_dict,
    validate_poles,
)
from algorithms.poles.wire_evidence import (
    CoarseWireResult,
    coarse_wire_to_dict,
    detect_coarse_wire_evidence,
)
from algorithms.spans.checkpoint import (
    read_checkpoint,
    write_checkpoint,
)
from algorithms.spans.clearance import (
    SpanClearanceResult,
    run_span_tree_wire_clearance,
)
from algorithms.spans.reconcile import (
    TreeReconciliationResult,
    reconcile_trees,
    reconciliation_to_dict,
)
from algorithms.spans.region import subset_raw_region
from algorithms.spans.tree_runner import (
    load_span_tree_outcome,
    run_span_tree_pipeline,
)
from algorithms.spans.wire import (
    load_span_wire_result,
    run_span_wire_pipeline,
)
from algorithms.vegetation.coarse_to_fine import (
    recover_vegetation_region_raw,
)
from algorithms.vegetation.fast_research import (
    FastResearchError,
    sha256_file,
)
from algorithms.vegetation.fast_weighted_preprocess import (
    WeightedWorkingSet,
    build_weighted_fast_vegetation,
    build_weighted_working_set,
)

_Progress = Callable[[int, str, bool, float | None], None]
_BLOCKED_INSUFFICIENT_POLES = "SPAN_PIPELINE:BLOCKED_INSUFFICIENT_POLES"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
    )
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _report(
    progress: _Progress | None,
    stage: int,
    message: str,
    done: bool,
    elapsed: float | None,
) -> None:
    if progress is not None:
        progress(stage, message, done, elapsed)


def _array_digest(values: np.ndarray) -> str:
    return canonical_digest(
        {
            "shape": [int(value) for value in values.shape],
            "dtype": str(values.dtype),
            "sha256": _sha256_bytes(values.tobytes()),
        }
    )


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def run_fast_baseline_pipeline(
    source: Path | str,
    run_root: Path | str,
    *,
    progress: _Progress | None = None,
    config: FastBaselineConfig | None = None,
    voxel_size_m: float | None = None,
) -> dict[str, Any]:
    """Run the pole-first span-based mature baseline in an isolated namespace."""

    source_path = Path(source).resolve()
    root = Path(run_root).resolve()
    if not source_path.is_file():
        raise FastResearchError(f"FAST_BASELINE_INPUT_MISSING: {source_path}")
    if root.exists():
        # The one-click CLI writes run.log before orchestration; it is the sole
        # permitted pre-existing file so a partial run never resumes silently.
        unexpected = [path for path in root.iterdir() if path.name != "run.log"]
        if unexpected:
            raise FastResearchError(f"FAST_BASELINE_OUTPUT_EXISTS: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
    for stage in (
        "00_input",
        "01_preprocess",
        "02_ground",
        "03_poles",
        "04_topology",
        "05_wire",
        "06_tree",
        "07_clearance",
        "08_report",
    ):
        (root / stage).mkdir(parents=True, exist_ok=True)

    cfg = config or default_fast_baseline_config()
    total_started = time.perf_counter()
    input_sha = sha256_file(source_path)
    source_id = f"new_las_{input_sha[:16]}"
    inspection = _inspect_source(source_path, input_sha, source_id)
    bounds_min = np.asarray(inspection["bounds_min_xyz"], dtype=np.float64)
    bounds_max = np.asarray(inspection["bounds_max_xyz"], dtype=np.float64)
    xy_area_m2 = float(
        max(0.0, bounds_max[0] - bounds_min[0])
        * max(0.0, bounds_max[1] - bounds_min[1])
    )
    coarse_voxel_size_m = (
        choose_representative_voxel_size(
            int(inspection["point_count"]), xy_area_m2=xy_area_m2
        )
        if voxel_size_m is None
        else float(voxel_size_m)
    )
    inspection["coarse_voxel_size_m"] = coarse_voxel_size_m
    _write_json(root / "00_input" / "source_inspection.json", inspection)
    timings: dict[str, float] = {}
    quality: dict[str, str] = {}
    base_upstream = canonical_digest(
        {
            "schema": FAST_BASELINE_SCHEMA_VERSION,
            "source_sha256": input_sha,
            "source_id": source_id,
            "config_digest": cfg.computed_digest(),
            "coarse_voxel_size_m": coarse_voxel_size_m,
        }
    )

    # ------------------------------------------------------------------ stage 1
    _report(progress, 1, "读取点云并建立粗点云", False, None)
    started = time.perf_counter()
    working = _stage_preprocess(
        source_path, root, cfg, input_sha, source_id, coarse_voxel_size_m, base_upstream
    )
    timings["preprocess"] = time.perf_counter() - started
    _report(progress, 1, "粗点云建立完成", True, timings["preprocess"])
    upstream = working.config_digest

    # ------------------------------------------------------------------ stage 2
    _report(progress, 2, "地面分离与高度计算", False, None)
    started = time.perf_counter()
    ground = _stage_ground(working, root, cfg, input_sha, source_id, upstream)
    timings["ground"] = time.perf_counter() - started
    _report(progress, 2, "地面分离与高度计算完成", True, timings["ground"])
    from algorithms.vegetation.fast_research import fast_ground_content_digest

    upstream = fast_ground_content_digest(ground)

    # ------------------------------------------------------------------ stage 3
    _report(progress, 3, "检测电杆/杆塔候选", False, None)
    started = time.perf_counter()
    origin_xyz = tuple(float(value) for value in working.source_bounds_min_xyz)
    poles = _stage_poles(
        ground, root, cfg, input_sha, source_id, upstream, origin_xyz=origin_xyz
    )
    timings["poles"] = time.perf_counter() - started
    _report(progress, 3, "电杆/杆塔候选检测完成", True, timings["poles"])
    upstream = poles["digest"]

    # ------------------------------------------------------------------ stage 4
    _report(progress, 4, "建立线路拓扑并划分档距", False, None)
    started = time.perf_counter()
    topology = _stage_topology(
        poles, root, cfg, input_sha, source_id, upstream
    )
    timings["topology"] = time.perf_counter() - started
    _report(progress, 4, "线路拓扑与档距划分完成", True, timings["topology"])
    upstream = topology["digest"]

    if topology["topology"].span_count == 0:
        # Still render the pole / topology review pages so the owner can inspect
        # exactly why no corridor could be established.
        from algorithms.spans.visualization import generate_blocked_review_pages

        generate_blocked_review_pages(root, poles, topology["topology"])
        return _build_blocked_summary(
            source_path, root, cfg, inspection, working, ground, poles,
            topology, quality, timings, total_started,
            reason="SPAN_PIPELINE:BLOCKED_INSUFFICIENT_POLES",
        )

    # ------------------------------------------------------------------ stage 5
    _report(progress, 5, "逐档识别和重建导线", False, None)
    started = time.perf_counter()
    wire_outcomes = _stage_span_wires(
        source_path, working, ground, poles, topology, root, cfg,
        input_sha, source_id, upstream,
    )
    timings["wire"] = time.perf_counter() - started
    _report(progress, 5, "逐档导线重建完成", True, timings["wire"])
    upstream = _stage_digest(
        [outcome["digest"] for outcome in wire_outcomes]
    )

    # ------------------------------------------------------------------ stage 6
    _report(progress, 6, "逐档进行单木分割", False, None)
    started = time.perf_counter()
    tree_outcomes = _stage_span_trees(
        source_path, working, ground, poles, topology, root, cfg,
        input_sha, source_id, upstream,
    )
    timings["tree"] = time.perf_counter() - started
    _report(progress, 6, "逐档单木分割完成", True, timings["tree"])
    upstream = _stage_digest([outcome["digest"] for outcome in tree_outcomes])

    # ------------------------------------------------------------------ stage 7
    _report(progress, 7, "全分辨率回映与树线距离计算", False, None)
    started = time.perf_counter()
    reconciliation, clearance = _stage_reconcile_clearance(
        tree_outcomes,
        wire_outcomes,
        root,
        cfg,
        input_sha,
        source_id,
        upstream,
    )
    timings["clearance"] = time.perf_counter() - started
    _report(progress, 7, "全分辨率回映与树线距离计算完成", True, timings["clearance"])

    # ------------------------------------------------------------------ stage 8
    _report(progress, 8, "生成结果与报告", False, None)
    started = time.perf_counter()
    summary = _build_summary(
        source_path, root, cfg, inspection, working, ground, poles, topology,
        wire_outcomes, tree_outcomes, reconciliation, clearance, quality,
        timings, total_started,
    )
    from algorithms.spans.visualization import generate_all_review_pages

    generate_all_review_pages(
        root, summary, poles, topology, wire_outcomes, tree_outcomes,
        reconciliation, clearance,
    )
    timings["report"] = time.perf_counter() - started
    _report(progress, 8, "结果与报告生成完成", True, timings["report"])
    return summary


def choose_representative_voxel_size(
    raw_point_count: int, xy_area_m2: float | None = None
) -> float:
    """Deterministic coarse-routing scale from source size and scene extent.

    A compact scene (small XY area) keeps the reviewed 0.25 m coarse scale even
    when the raw point count is very large, because the number of occupied
    coarse voxels is bounded by the area, not the point count.  Larger scenes
    coarsen only when they genuinely have many occupied voxels.
    """

    if xy_area_m2 is not None and xy_area_m2 <= 300_000:
        return 0.25
    if raw_point_count <= 5_000_000:
        return 0.25
    if raw_point_count <= 20_000_000:
        return 0.35
    if raw_point_count <= 50_000_000:
        return 0.5
    return 0.75


def _stage_preprocess(
    source: Path,
    root: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    coarse_voxel_size_m: float,
    upstream: str,
) -> WeightedWorkingSet:
    stage_dir = root / "01_preprocess"
    cached = read_checkpoint(
        stage_dir, "preprocess",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
    )
    if cached is not None:
        _summary, _digest = cached
        return _load_working_set_from_files(stage_dir, source, coarse_voxel_size_m)
    _status, working = build_weighted_working_set(
        source,
        stage_dir,
        voxel_size_m=coarse_voxel_size_m,
        artifact_stem="new_las",
    )
    digest = working.config_digest
    write_checkpoint(
        stage_dir, "preprocess",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
        summary={
            "representative_count": working.representative_count,
            "source_point_count": working.source_point_count,
            "status": _status,
        },
        output_digest=digest,
    )
    return working


def _load_working_set_from_files(
    stage_dir: Path, source: Path, coarse_voxel_size_m: float
) -> WeightedWorkingSet:
    import laspy

    from algorithms.vegetation.fast_research import canonical_digest

    reps_npz = stage_dir / "working_set_representatives.npz"
    reps_las = stage_dir / "working_set_representatives.las"
    mapping = stage_dir / "raw_to_representative_id.npy"
    with laspy.open(source) as reader:
        source_count = int(reader.header.point_count)
        mins = np.asarray(reader.header.mins, dtype=np.float64)
        maxs = np.asarray(reader.header.maxs, dtype=np.float64)
        scales = np.asarray(reader.header.scales, dtype=np.float64)
        offsets = np.asarray(reader.header.offsets, dtype=np.float64)
    with np.load(reps_npz, allow_pickle=False) as stream:
        representative_count = int(len(stream["representative_id"]))
    return WeightedWorkingSet(
        schema_version="coarse-weighted-preprocess-v1",
        non_certified_marker=NON_CERTIFIED_MARKER,
        source_las_path=str(source),
        source_las_sha256=sha256_file(source),
        source_point_count=source_count,
        source_bounds_min_xyz=tuple(float(value) for value in mins),
        source_bounds_max_xyz=tuple(float(value) for value in maxs),
        source_scales=tuple(float(value) for value in scales),
        source_offsets=tuple(float(value) for value in offsets),
        voxel_size_m=coarse_voxel_size_m,
        representative_count=representative_count,
        raw_count_sum=source_count,
        representatives_npz_path=str(reps_npz),
        representative_las_path=str(reps_las),
        raw_to_representative_npy_path=str(mapping),
        classification_counts={},
        elapsed_seconds=0.0,
        config_digest=canonical_digest(
            {
                "schema_version": "coarse-weighted-preprocess-v1",
                "source_las_sha256": sha256_file(source),
                "voxel_size_m": float(coarse_voxel_size_m),
            }
        ),
    )


def _stage_ground(
    working: WeightedWorkingSet,
    root: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    upstream: str,
) -> Any:
    from algorithms.vegetation.fast_research import (
        FastGroundResult,
        fast_ground_content_digest,
        run_fast_ground_hag,
    )

    stage_dir = root / "02_ground"
    cached = read_checkpoint(
        stage_dir, "ground",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
    )
    if cached is not None:
        _summary, _digest = cached
        from algorithms.vegetation.fast_research import FastGroundResult

        payload = json.loads((stage_dir / "ground_hag_result.json").read_text("utf-8"))
        return FastGroundResult.from_dict(payload)
    _status, ground = run_fast_ground_hag(
        Path(working.representative_las_path),
        stage_dir,
    )
    digest = fast_ground_content_digest(ground)
    write_checkpoint(
        stage_dir, "ground",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
        summary={
            "status": _status,
            "ground_class_2_count": ground.ground_class_2_count,
            "hag_valid_count": ground.hag_valid_count,
        },
        output_digest=digest,
    )
    return ground


def _stage_poles(
    ground: Any,
    root: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    upstream: str,
    *,
    origin_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    stage_dir = root / "03_poles"
    cached = read_checkpoint(
        stage_dir, "poles",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
    )
    if cached is not None:
        summary, digest = cached
        evidence = _load_ground_evidence_from_cache(ground, cfg, origin_xyz=origin_xyz)
        with np.load(stage_dir / "pole_evidence.npz", allow_pickle=False) as stream:
            candidate_id_by_rep = np.asarray(
                stream["candidate_id_by_representative"], dtype=np.int32
            )
            seed_mask = np.asarray(stream["seed_mask"], dtype=bool)
            component_label = np.asarray(stream["component_label"], dtype=np.int32)
            wire_candidate_mask = np.asarray(stream["wire_candidate_mask"], dtype=bool)
            wire_segment_id_by_rep = np.asarray(stream["wire_segment_id_by_rep"], dtype=np.int32)
            wire_component_label = np.asarray(
                stream["wire_component_label"], dtype=np.int32
            )
        candidates = PoleCandidateResult.from_records_dict(
            summary["candidates"],
            candidate_id_by_representative=candidate_id_by_rep,
            seed_mask=seed_mask,
            component_label_by_representative=component_label,
        )
        wire_evidence = CoarseWireResult.from_segments_dict(
            summary["wire_evidence"],
            candidate_mask=wire_candidate_mask,
            segment_id_by_representative=wire_segment_id_by_rep,
            component_label_by_representative=wire_component_label,
        )
        validation = PoleValidationResult.from_records_dict(summary["validation"])
        return {
            "evidence": evidence,
            "candidates": candidates,
            "wire_evidence": wire_evidence,
            "validation": validation,
            "digest": digest,
        }
    evidence = load_ground_evidence(
        Path(ground.output_las_path),
        cfg,
        origin_xyz=origin_xyz,
    )
    candidates = detect_pole_candidates(evidence, cfg)
    wire_evidence = detect_coarse_wire_evidence(evidence, cfg)
    validation = validate_poles(candidates, wire_evidence, evidence, cfg)
    payload = {
        "candidates": pole_candidates_to_dict(candidates),
        "wire_evidence": coarse_wire_to_dict(wire_evidence),
        "validation": pole_validation_to_dict(validation),
    }
    _write_json(stage_dir / "pole_candidates.json", payload["candidates"])
    _write_json(stage_dir / "coarse_wire_evidence.json", payload["wire_evidence"])
    _write_json(stage_dir / "pole_validation.json", payload["validation"])
    _write_npz(
        stage_dir / "pole_evidence.npz",
        candidate_id_by_representative=candidates.candidate_id_by_representative,
        seed_mask=candidates.seed_mask,
        component_label=candidates.component_label_by_representative,
        wire_candidate_mask=wire_evidence.candidate_mask,
        wire_segment_id_by_rep=wire_evidence.segment_id_by_representative,
        wire_component_label=wire_evidence.component_label_by_representative,
    )
    digest = canonical_digest(payload)
    write_checkpoint(
        stage_dir, "poles",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
        summary=payload,
        output_digest=digest,
    )
    return {
        "evidence": evidence,
        "candidates": candidates,
        "wire_evidence": wire_evidence,
        "validation": validation,
        "digest": digest,
    }


def _load_ground_evidence_from_cache(
    ground: Any, cfg: FastBaselineConfig, *, origin_xyz: tuple[float, float, float]
) -> GroundEvidence:
    return load_ground_evidence(
        Path(ground.output_las_path),
        cfg,
        origin_xyz=origin_xyz,
    )


def _stage_topology(
    poles: dict[str, Any],
    root: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    upstream: str,
) -> dict[str, Any]:
    stage_dir = root / "04_topology"
    cached = read_checkpoint(
        stage_dir, "topology",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
    )
    if cached is not None:
        summary, digest = cached
        return {
            "topology": SpanTopologyResult.from_dict(summary),
            "digest": digest,
        }
    topology = build_span_topology(
        poles["candidates"],
        poles["validation"],
        poles["wire_evidence"],
        poles["evidence"],
        cfg,
    )
    payload = topology_to_dict(topology)
    _write_json(stage_dir / "topology.json", payload)
    digest = canonical_digest(payload)
    write_checkpoint(
        stage_dir, "topology",
        source_sha256=input_sha, source_id=source_id,
        config_digest=cfg.computed_digest(), upstream_digest=upstream,
        summary=payload,
        output_digest=digest,
    )
    return {"topology": topology, "digest": digest}


def _stage_span_wires(
    source: Path,
    working: WeightedWorkingSet,
    ground: Any,
    poles: dict[str, Any],
    topology: dict[str, Any],
    root: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    upstream: str,
) -> list[dict[str, Any]]:
    stage_dir = root / "05_wire"
    wire_candidate_rep_ids = np.unique(
        poles["evidence"].representative_id[poles["wire_evidence"].candidate_mask]
    )
    wire_region_dir = stage_dir / "wire_region"
    wire_region = _recover_wire_region(
        source,
        working,
        wire_candidate_rep_ids,
        wire_region_dir,
        cfg,
        input_sha,
        source_id,
        upstream,
    )
    origin_xyz = tuple(float(value) for value in working.source_bounds_min_xyz)
    outcomes: list[dict[str, Any]] = []
    for position, span in enumerate(topology["topology"].spans):
        span_dir = stage_dir / f"span_{position + 1:03d}"
        cached = read_checkpoint(
            span_dir, "wire",
            source_sha256=input_sha, source_id=source_id,
            config_digest=cfg.computed_digest(), upstream_digest=upstream,
        )
        if cached is not None:
            summary, digest = cached
            wire_result = load_span_wire_result(span_dir)
        else:
            subset = subset_raw_region(
                wire_region,
                span.halo_bounds_min_xyz,
                span.halo_bounds_max_xyz,
                span_dir / "recovered_region",
            )
            wire_result = run_span_wire_pipeline(
                span,
                subset,
                cfg,
                origin_xyz=origin_xyz,
                output_dir=span_dir,
                authoritative_source_point_count=working.source_point_count,
            )
            payload = {
                "strand_count": wire_result.strand_count,
                "strands": [strand.to_dict() for strand in wire_result.strands],
            }
            _write_json(span_dir / "wire_summary.json", payload)
            digest = canonical_digest(payload)
            write_checkpoint(
                span_dir, "wire",
                source_sha256=input_sha, source_id=source_id,
                config_digest=cfg.computed_digest(), upstream_digest=upstream,
                summary=payload,
                output_digest=digest,
            )
        outcomes.append(
            {
                "span": span,
                "wire": wire_result,
                "digest": canonical_digest(
                    {"strand_count": wire_result.strand_count}
                ),
            }
        )
    return outcomes


def _recover_wire_region(
    source: Path,
    working: WeightedWorkingSet,
    wire_candidate_rep_ids: np.ndarray,
    output_dir: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    upstream: str,
) -> Any:
    output_dir = Path(output_dir)
    sidecar = output_dir / "region_sidecar.json"
    if sidecar.is_file():
        from algorithms.vegetation.coarse_to_fine import load_raw_region

        return load_raw_region(sidecar)
    return recover_vegetation_region_raw(
        source,
        working,
        wire_candidate_rep_ids,
        output_dir,
    )


def _stage_span_trees(
    source: Path,
    working: WeightedWorkingSet,
    ground: Any,
    poles: dict[str, Any],
    topology: dict[str, Any],
    root: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    upstream: str,
) -> list[dict[str, Any]]:
    stage_dir = root / "06_tree"
    from algorithms.vegetation.coarse_to_fine import (
        load_coarse_ground_representatives,
    )

    coarse_ground = load_coarse_ground_representatives(ground)
    all_scene_roi = {
        "xy_bounds_min": [
            float(working.source_bounds_min_xyz[0]),
            float(working.source_bounds_min_xyz[1]),
        ],
        "xy_bounds_max": [
            float(working.source_bounds_max_xyz[0]),
            float(working.source_bounds_max_xyz[1]),
        ],
        "z_bounds_min": float(working.source_bounds_min_xyz[2]),
        "z_bounds_max": float(working.source_bounds_max_xyz[2]),
    }
    wire_candidate_rep_ids = np.unique(
        poles["evidence"].representative_id[poles["wire_evidence"].candidate_mask]
    )
    _coarse_vegetation, coarse_manifest = build_weighted_fast_vegetation(
        working,
        all_scene_roi,
        ground,
        stage_dir / "coarse_routing",
        source_id=source_id,
        wire_geometry_representative_ids=wire_candidate_rep_ids,
        artifact_stem="new_las",
    )
    with np.load(coarse_manifest["diagnostics_npz_path"], allow_pickle=False) as stream:
        diag_state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
        diag_ids = np.asarray(stream["representative_id"], dtype=np.uint32)
    from algorithms.vegetation.fast_vegetation_v3 import FastV3DiagnosticState

    retained_ids = diag_ids[diag_state == int(FastV3DiagnosticState.VEGETATION_RETAINED)]
    veg_region_dir = stage_dir / "vegetation_region"
    veg_region = _recover_vegetation_region(
        source, working, retained_ids, veg_region_dir
    )
    origin_xyz = tuple(float(value) for value in working.source_bounds_min_xyz)
    outcomes: list[dict[str, Any]] = []
    for position, span in enumerate(topology["topology"].spans):
        span_dir = stage_dir / f"span_{position + 1:03d}"
        cached = read_checkpoint(
            span_dir, "tree",
            source_sha256=input_sha, source_id=source_id,
            config_digest=cfg.computed_digest(), upstream_digest=upstream,
        )
        if cached is not None:
            summary, digest = cached
            outcome = load_span_tree_outcome(span, span_dir)
        else:
            outcome = run_span_tree_pipeline(
                span,
                veg_region,
                coarse_ground,
                cfg,
                output_dir=span_dir,
                origin_xyz=origin_xyz,
            )
            payload = outcome.to_manifest()
            _write_json(span_dir / "tree_summary.json", payload)
            digest = canonical_digest(payload)
            write_checkpoint(
                span_dir, "tree",
                source_sha256=input_sha, source_id=source_id,
                config_digest=cfg.computed_digest(), upstream_digest=upstream,
                summary=payload,
                output_digest=digest,
            )
        outcomes.append(
            {
                "span": span,
                "outcome": outcome,
                "digest": canonical_digest(
                    {"tree_count": len(outcome.tree_records)}
                ),
            }
        )
    return outcomes


def _recover_vegetation_region(
    source: Path,
    working: WeightedWorkingSet,
    retained_ids: np.ndarray,
    output_dir: Path,
) -> Any:
    output_dir = Path(output_dir)
    sidecar = output_dir / "region_sidecar.json"
    if sidecar.is_file():
        from algorithms.vegetation.coarse_to_fine import load_raw_region

        return load_raw_region(sidecar)
    return recover_vegetation_region_raw(
        source,
        working,
        retained_ids,
        output_dir,
    )


def _stage_reconcile_clearance(
    tree_outcomes: list[dict[str, Any]],
    wire_outcomes: list[dict[str, Any]],
    root: Path,
    cfg: FastBaselineConfig,
    input_sha: str,
    source_id: str,
    upstream: str,
) -> tuple[TreeReconciliationResult, SpanClearanceResult]:
    stage_dir = root / "07_clearance"
    # Reconcile + clearance are cheap (in-memory over already-persisted span
    # outcomes); they are always recomputed from the validated span artifacts.
    span_outcomes = [entry["outcome"] for entry in tree_outcomes]
    reconciliation = reconcile_trees(
        span_outcomes,
        cfg,
        origin_xyz=tuple(0.0 for _ in range(3)),
    )
    wire_results = [entry["wire"] for entry in wire_outcomes]
    clearance = run_span_tree_wire_clearance(reconciliation, wire_results, cfg)
    _write_json(
        stage_dir / "reconciliation.json", reconciliation_to_dict(reconciliation)
    )
    _write_json(
        stage_dir / "tree_wire_clearance.json",
        {
            "status": clearance.status,
            "global_minimum_clearance_m": clearance.global_minimum_clearance_m,
            "records": [record.to_dict() for record in clearance.records],
            "edge_count": clearance.edge_count,
            "diagnostics": dict(clearance.diagnostics),
        },
    )
    return reconciliation, clearance


def clearance_to_dict(clearance: SpanClearanceResult) -> dict[str, Any]:
    return {
        "status": clearance.status,
        "global_minimum_clearance_m": clearance.global_minimum_clearance_m,
        "edge_count": clearance.edge_count,
        "records": [record.to_dict() for record in clearance.records],
        "diagnostics": dict(clearance.diagnostics),
    }


def _stage_digest(digests: list[str]) -> str:
    return canonical_digest({"stage_outputs": sorted(digests)})


def _inspect_source(source: Path, input_sha: str, source_id: str) -> dict[str, Any]:
    import laspy

    started = time.perf_counter()
    with laspy.open(source) as reader:
        header = reader.header
        payload = {
            "input_file": str(source),
            "input_sha256": input_sha,
            "source_id": source_id,
            "point_count": int(header.point_count),
            "bounds_min_xyz": [float(value) for value in header.mins],
            "bounds_max_xyz": [float(value) for value in header.maxs],
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
            "point_format": int(header.point_format.id),
            "version": str(header.version),
        }
    if payload["point_count"] <= 0:
        raise FastResearchError("NEW_LAS_DEMO_EMPTY_SOURCE")
    payload["inspection_runtime_seconds"] = time.perf_counter() - started
    return payload


def _build_blocked_summary(
    source: Path,
    root: Path,
    cfg: FastBaselineConfig,
    inspection: dict[str, Any],
    working: WeightedWorkingSet,
    ground: Any,
    poles: dict[str, Any],
    topology: dict[str, Any],
    quality: dict[str, str],
    timings: dict[str, float],
    total_started: float,
    *,
    reason: str,
) -> dict[str, Any]:
    summary = {
        "RUN_STATUS": "BLOCKED",
        "POLE_SPAN_MATURE_BASELINE": "BLOCKED_OWNER_DECISION",
        "SPAN_PIPELINE": reason,
        "NON_CERTIFIED": True,
        "NON_CERTIFIED_MARKER": NON_CERTIFIED_MARKER,
        "INPUT_FILE": str(source),
        "INPUT_SHA256": inspection["input_sha256"],
        "SOURCE_ID": inspection["source_id"],
        "OUTPUT_DIRECTORY": str(root),
        "RAW_POINTS": working.source_point_count,
        "COARSE_REPRESENTATIVES": working.representative_count,
        "POLE_CANDIDATES": poles["candidates"].candidate_count,
        "CONFIRMED_POLES": len(poles["validation"].confirmed_pole_ids),
        "LINE_COUNT": 0,
        "SPAN_COUNT": 0,
        "TOTAL_RUNTIME": time.perf_counter() - total_started,
        "PEAK_RSS_GIB": None,
        "REASON": reason,
        "TREE_WIRE_RESULT_STATUS": "BLOCKED",
    }
    _write_summary(root, summary)
    return summary


def _build_summary(
    source: Path,
    root: Path,
    cfg: FastBaselineConfig,
    inspection: dict[str, Any],
    working: WeightedWorkingSet,
    ground: Any,
    poles: dict[str, Any],
    topology: dict[str, Any],
    wire_outcomes: list[dict[str, Any]],
    tree_outcomes: list[dict[str, Any]],
    reconciliation: TreeReconciliationResult,
    clearance: SpanClearanceResult,
    quality: dict[str, str],
    timings: dict[str, float],
    total_started: float,
) -> dict[str, Any]:
    confirmed_pole_ids = poles["validation"].confirmed_pole_ids
    likely_pole_ids = poles["validation"].likely_pole_ids
    raw_local_maxima = sum(entry["outcome"].raw_local_maxima for entry in tree_outcomes)
    accepted_tops = sum(entry["outcome"].accepted_tree_tops for entry in tree_outcomes)
    final_trees = [
        record for record in reconciliation.global_tree_records if not record.review
    ]
    review_trees = [
        record for record in reconciliation.global_tree_records if record.review
    ]
    wire_candidate_points = sum(
        int(entry["wire"].diagnostics.get("wire_candidate_point_count", 0))
        for entry in wire_outcomes
    )
    wire_clusters = sum(
        int(entry["wire"].diagnostics.get("wire_cluster_count", 0))
        for entry in wire_outcomes
    )
    wire_strands = sum(entry["wire"].strand_count for entry in wire_outcomes)
    w1_polyline_count = sum(entry["wire"].strand_count for entry in wire_outcomes)
    w1_segments = sum(
        int(entry["wire"].distance_ready_segments) for entry in wire_outcomes
    )
    pole_quality = _pole_quality(
        poles["validation"].confirmed_pole_ids,
        poles["validation"].likely_pole_ids,
        poles["candidates"].candidate_count,
    )
    tree_quality = _tree_quality(final_trees, raw_local_maxima, accepted_tops)
    wire_quality = _wire_quality(wire_strands, len(wire_outcomes))
    span_topology_quality = _topology_quality(topology["topology"])
    quality = {
        "POLE_DETECTION_QUALITY": pole_quality,
        "SPAN_TOPOLOGY_QUALITY": span_topology_quality,
        "TREE_SEGMENTATION_QUALITY": tree_quality["tree"],
        "TREE_OVERMERGE_SUSPECTED": tree_quality["overmerge"],
        "TREE_OVERSPLIT_SUSPECTED": tree_quality["oversplit"],
        "WIRE_EXTRACTION_QUALITY": wire_quality,
        "TREE_WIRE_RESULT_STATUS": (
            "USABLE_FOR_DEMO"
            if clearance.status == "VALID"
            else "SUSPICIOUS"
            if clearance.status == "NO_DISTANCE_READY_EDGE"
            else "BLOCKED"
        ),
    }
    summary = {
        "RUN_STATUS": (
            "PASS" if clearance.status == "VALID" else "PASS_WITH_TREE_WIRE_BLOCKED"
        ),
        "POLE_SPAN_MATURE_BASELINE": "PASS",
        "METHOD": "POLE_FIRST_SPAN_BASED",
        "NON_CERTIFIED": True,
        "NON_CERTIFIED_MARKER": NON_CERTIFIED_MARKER,
        "INPUT_FILE": str(source),
        "INPUT_SHA256": inspection["input_sha256"],
        "SOURCE_ID": inspection["source_id"],
        "OUTPUT_DIRECTORY": str(root),
        "RAW_POINTS": working.source_point_count,
        "COARSE_REPRESENTATIVES": working.representative_count,
        "POLE_CANDIDATES": poles["candidates"].candidate_count,
        "CONFIRMED_POLES": len(confirmed_pole_ids),
        "LIKELY_POLES": len(likely_pole_ids),
        "LINE_COUNT": topology["topology"].line_count,
        "SPAN_COUNT": topology["topology"].span_count,
        "RAW_LOCAL_MAXIMA": int(raw_local_maxima),
        "ACCEPTED_TREE_TOPS": int(accepted_tops),
        "FINAL_TREE_COUNT": len(final_trees),
        "REVIEW_TREE_COUNT": len(review_trees),
        "FULLRES_ASSIGNED_TREE_POINTS": len(reconciliation.final_original_index),
        "WIRE_CANDIDATE_POINTS": int(wire_candidate_points),
        "WIRE_CLUSTER_COUNT": int(wire_clusters),
        "WIRE_STRAND_COUNT": int(wire_strands),
        "W1_POLYLINE_COUNT": int(w1_polyline_count),
        "W1_DISTANCE_READY_SEGMENTS": int(w1_segments),
        "TREE_WIRE_TREE_POINT_SOURCE": "FULL_RESOLUTION_RAW",
        "TREE_WIRE_STATUS": clearance.status,
        "TREE_WIRE_RESULT_STATUS": quality["TREE_WIRE_RESULT_STATUS"],
        "GLOBAL_MINIMUM_CLEARANCE": clearance.global_minimum_clearance_m,
        "PREPROCESS_RUNTIME": timings.get("preprocess", 0.0),
        "GROUND_RUNTIME": timings.get("ground", 0.0),
        "POLES_RUNTIME": timings.get("poles", 0.0),
        "TOPOLOGY_RUNTIME": timings.get("topology", 0.0),
        "WIRE_RUNTIME": timings.get("wire", 0.0),
        "TREE_RUNTIME": timings.get("tree", 0.0),
        "CLEARANCE_RUNTIME": timings.get("clearance", 0.0),
        "REPORT_RUNTIME": timings.get("report", 0.0),
        "TOTAL_RUNTIME": time.perf_counter() - total_started,
        "PEAK_RSS_GIB": None,
        **quality,
        "TREES": [record.to_dict() for record in reconciliation.global_tree_records],
        "SPANS": [span.to_dict() for span in topology["topology"].spans],
        "LINES": [line.to_dict() for line in topology["topology"].lines],
        "CLEARANCE_RECORDS": [record.to_dict() for record in clearance.records],
    }
    _write_summary(root, summary)
    return summary


def _pole_quality(
    confirmed: list[int], likely: list[int], candidates: int
) -> str:
    usable = len(confirmed) + len(likely)
    if candidates == 0:
        return "POOR"
    if usable >= 4 and len(confirmed) >= max(2, usable * 0.4):
        return "GOOD"
    if usable >= 2:
        return "PARTIAL"
    return "POOR"


def _topology_quality(topology: SpanTopologyResult) -> str:
    if topology.span_count == 0:
        return "POOR"
    if topology.span_count >= 2 and topology.line_count >= 1:
        return "GOOD"
    return "PARTIAL"


def _tree_quality(
    final_trees: list[Any], raw_maxima: int, accepted_tops: int
) -> dict[str, str]:
    tree = "PARTIAL"
    overmerge = "NO"
    oversplit = "NO"
    if len(final_trees) == 0:
        tree = "POOR"
    elif accepted_tops > 0:
        ratio = len(final_trees) / max(1, accepted_tops)
        if ratio < 0.3:
            oversplit = "YES"
        if (
            len(final_trees) == 1
            and final_trees
            and final_trees[0].full_resolution_point_count > 200_000
        ):
            overmerge = "YES"
        if len(final_trees) >= 2:
            tree = "GOOD"
    return {"tree": tree, "overmerge": overmerge, "oversplit": oversplit}


def _wire_quality(wire_strands: int, span_count: int) -> str:
    if span_count == 0:
        return "POOR"
    per_span = wire_strands / span_count
    if per_span >= 1.5:
        return "GOOD"
    if per_span >= 0.5:
        return "PARTIAL"
    return "POOR"


def _write_summary(root: Path, summary: dict[str, Any]) -> None:
    report_dir = root / "08_report"
    _write_json(report_dir / "run_summary.json", summary)
    _write_text(report_dir / "run_summary.txt", render_summary_text(summary))


def render_summary_text(summary: dict[str, Any]) -> str:
    labels = {
        "RUN_STATUS": "运行状态",
        "RAW_POINTS": "原始点数",
        "COARSE_REPRESENTATIVES": "粗路由代表点",
        "POLE_CANDIDATES": "电杆候选",
        "CONFIRMED_POLES": "确认电杆",
        "LINE_COUNT": "线路数",
        "SPAN_COUNT": "档距数",
        "RAW_LOCAL_MAXIMA": "原始局部极大值",
        "ACCEPTED_TREE_TOPS": "接受树顶",
        "FINAL_TREE_COUNT": "最终单木数",
        "FULLRES_ASSIGNED_TREE_POINTS": "已分配全分辨率树点数",
        "WIRE_CANDIDATE_POINTS": "导线候选点数",
        "WIRE_CLUSTER_COUNT": "导线簇数",
        "WIRE_STRAND_COUNT": "导线根数",
        "W1_POLYLINE_COUNT": "W1 导线数量",
        "W1_DISTANCE_READY_SEGMENTS": "W1 距离就绪线段",
        "TREE_WIRE_STATUS": "树线距离状态",
        "GLOBAL_MINIMUM_CLEARANCE": "场景最小距离（米）",
        "POLE_DETECTION_QUALITY": "电杆检测质量",
        "SPAN_TOPOLOGY_QUALITY": "档距拓扑质量",
        "TREE_SEGMENTATION_QUALITY": "单木分割质量",
        "TREE_OVERMERGE_SUSPECTED": "疑似过合并",
        "TREE_OVERSPLIT_SUSPECTED": "疑似过分割",
        "WIRE_EXTRACTION_QUALITY": "导线提取质量",
        "TREE_WIRE_RESULT_STATUS": "树线结果状态",
        "NON_CERTIFIED": "非认证研究演示",
    }
    lines = ["POLE_SPAN_MATURE_BASELINE 运行摘要", "=" * 44]
    for key, label in labels.items():
        lines.append(f"{label}: {summary.get(key)}")
    lines.append("")
    lines.append(f"结果目录: {summary.get('OUTPUT_DIRECTORY')}")
    return "\n".join(lines) + "\n"


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


__all__ = [
    "_BLOCKED_INSUFFICIENT_POLES",
    "run_fast_baseline_pipeline",
    "choose_representative_voxel_size",
]
