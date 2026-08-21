"""Source-shared role-index preparation for production Stage-B batches.

The canonical per-Span executor deliberately consumes two different source
views.  Wire keeps the reviewed coarse-wire admission route, while vegetation
is high recall and therefore must not inherit a broad wire exclusion.  This
module materializes those views once per source binding and publishes the
role-specific tile indexes expected by the executor.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from algorithms.spans.project_adapter import SpanAnalysisInput


@dataclass(frozen=True, slots=True)
class RoleIndexPreparation:
    wire_reference: str
    vegetation_reference: str
    source_point_count: int
    source_full_pass_count: int
    cache_hit: bool
    details: Mapping[str, Any]


def _source_path(source_identity: Mapping[str, Any]) -> Path:
    value = (
        source_identity.get("resolved_access_path")
        or source_identity.get("point_cloud_source")
        or source_identity.get("path")
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ROLE_INDEX_SOURCE_PATH_MISSING")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError("ROLE_INDEX_SOURCE_PATH_INVALID")
    return path


def _load_region(path: Path) -> Any | None:
    sidecar = path / "region_sidecar.json"
    if not sidecar.is_file():
        return None
    from algorithms.vegetation.coarse_to_fine import load_raw_region

    return load_raw_region(sidecar)


def _reference(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _representative_voxel_size(raw_point_count: int, xy_area_m2: float) -> float:
    """Frozen FAST coarse-routing scale without importing report backends."""

    if xy_area_m2 <= 300_000:
        return 0.25
    if raw_point_count <= 5_000_000:
        return 0.25
    if raw_point_count <= 20_000_000:
        return 0.35
    if raw_point_count <= 50_000_000:
        return 0.5
    return 0.75


def _span_admission_digest(span_inputs: tuple[SpanAnalysisInput, ...]) -> str:
    """Identity of the authoritative topology used by Wire PATH-B rescue."""

    from core.config.fast_baseline import canonical_digest

    return canonical_digest(
        [
            {
                "group_id": item.group_id,
                "span_id": item.span_id,
                "from_tower_id": item.from_tower_id,
                "to_tower_id": item.to_tower_id,
                "from_endpoint_geometry": dict(item.from_endpoint_geometry),
                "to_endpoint_geometry": dict(item.to_endpoint_geometry),
                "span_geometry": dict(item.span_geometry),
            }
            for item in sorted(span_inputs, key=lambda value: (value.group_id, value.span_id))
        ]
    )


def ensure_role_tile_indexes(
    project_root: Path,
    preparation_root: Path,
    source_identity: Mapping[str, Any],
    *,
    span_inputs: tuple[SpanAnalysisInput, ...],
    wire_configuration_digest: str,
    vegetation_configuration_digest: str,
) -> RoleIndexPreparation:
    """Create or validate the frozen Wire and high-recall Vegetation indexes.

    The weighted working set and coarse Ground/HAG are existing reviewed
    algorithms.  They are used only to admit the source-wide Wire role.  The
    Vegetation role retains every source row; the per-Span oriented query and
    exact confirmed-conductor subtraction remain downstream authorities.
    """

    from algorithms.poles.evidence import load_ground_evidence
    from algorithms.poles.wire_evidence import (
        detect_coarse_wire_evidence,
        rescue_span_aligned_wire_evidence,
    )
    from algorithms.spans.canonical_executor import _span_record
    from algorithms.spans.role_tiles import (
        ROLE_TILE_CELL_SIZE_M,
        RoleSpecificTileCache,
    )
    from algorithms.vegetation.coarse_to_fine import recover_vegetation_region_raw
    from algorithms.vegetation.fast_research import run_fast_ground_hag
    from algorithms.vegetation.fast_weighted_preprocess import (
        build_weighted_working_set,
    )

    from core.config.fast_baseline import FastBaselineConfig

    project_root = Path(project_root).resolve()
    preparation_root = Path(preparation_root).resolve()
    preparation_root.mkdir(parents=True, exist_ok=True)
    source = _source_path(source_identity)
    if not span_inputs:
        raise ValueError("ROLE_INDEX_SPAN_INPUTS_EMPTY")
    admission_digest = _span_admission_digest(span_inputs)

    import laspy

    with laspy.open(source) as reader:
        point_count = int(reader.header.point_count)
        mins = np.asarray(reader.header.mins, dtype=np.float64)
        maxs = np.asarray(reader.header.maxs, dtype=np.float64)
    if point_count <= 0:
        raise ValueError("ROLE_INDEX_SOURCE_EMPTY")

    # PATH-B admission is topology-bound.  Digest-suffixed Wire roots prevent
    # an index prepared for a subset from being silently reused for a different
    # project topology.  Vegetation is source-wide HIGH RECALL and topology-free.
    wire_root = preparation_root / (
        f"wire-role-tile-index-path-ab-rescue-v2-{admission_digest[:12]}"
    )
    vegetation_root = preparation_root / "vegetation-role-tile-index-high-recall-v3"
    expected = (
        ("wire", wire_root, wire_configuration_digest),
        ("vegetation", vegetation_root, vegetation_configuration_digest),
    )
    validated: dict[str, Any] = {}
    for role, root, digest in expected:
        try:
            validated[role] = RoleSpecificTileCache.load(
                root,
                role=role,
                source_identity=source_identity,
                config_digest=digest,
                source_point_count=point_count,
                expected_cell_size_m=ROLE_TILE_CELL_SIZE_M,
            )
        except Exception:  # absent/stale indexes are rebuilt by the path below
            validated = {}
            break
    if len(validated) == 2:
        return RoleIndexPreparation(
            wire_reference=_reference(project_root, wire_root),
            vegetation_reference=_reference(project_root, vegetation_root),
            source_point_count=point_count,
            source_full_pass_count=0,
            cache_hit=True,
            details={
                "schema": "treeline-stage-b-role-preparation-v1",
                "status": "REUSED_VALIDATED_ROLE_INDEXES",
                "wire_point_count": int(validated["wire"].metadata["totals"]["point_count"]),
                "vegetation_point_count": int(
                    validated["vegetation"].metadata["totals"]["point_count"]
                ),
            },
        )

    started = time.perf_counter()
    xy_area = float(maxs[0] - mins[0]) * float(maxs[1] - mins[1])
    voxel_size = _representative_voxel_size(point_count, xy_area)
    preprocess_status, working = build_weighted_working_set(
        source,
        preparation_root / "role-preprocess",
        voxel_size_m=voxel_size,
        artifact_stem="source_role",
    )
    source_full_pass_count = 0 if preprocess_status == "SKIPPED_FROM_CACHE" else 3

    ground_status, ground = run_fast_ground_hag(
        Path(working.representative_las_path),
        preparation_root / "role-ground",
    )
    config = FastBaselineConfig()
    evidence = load_ground_evidence(
        Path(ground.output_las_path),
        config,
        origin_xyz=tuple(float(value) for value in working.source_bounds_min_xyz),
    )
    wire = detect_coarse_wire_evidence(evidence, config)
    wire_admission_mask = np.asarray(wire.candidate_mask, dtype=bool).copy()
    rescue_details: dict[str, Any] = {}
    for item in sorted(span_inputs, key=lambda value: (value.group_id, value.span_id)):
        span = _span_record(item, config)
        rescued, diagnostics = rescue_span_aligned_wire_evidence(
            evidence,
            wire,
            span,
            config,
            origin_xyz=tuple(float(value) for value in working.source_bounds_min_xyz),
        )
        wire_admission_mask |= rescued
        rescue_details[f"{item.group_id}/{item.span_id}"] = dict(diagnostics)
    wire_representative_ids = np.unique(
        evidence.representative_id[wire_admission_mask]
    ).astype(np.uint32, copy=False)
    if not len(wire_representative_ids):
        raise ValueError("ROLE_INDEX_WIRE_ADMISSION_EMPTY")

    wire_region_root = preparation_root / (
        f"wire-region-span-rescue-{admission_digest[:12]}"
    )
    wire_region = _load_region(wire_region_root)
    if wire_region is None:
        wire_region = recover_vegetation_region_raw(
            source,
            working,
            wire_representative_ids,
            wire_region_root,
        )
        source_full_pass_count += 1

    vegetation_region_root = preparation_root / "vegetation-region-high-recall"
    vegetation_region = _load_region(vegetation_region_root)
    if vegetation_region is None:
        # HIGH RECALL means no broad source-role exclusion.  Span geometry and
        # exact confirmed-conductor subtraction are applied by the executor.
        vegetation_region = recover_vegetation_region_raw(
            source,
            working,
            np.arange(working.representative_count, dtype=np.uint32),
            vegetation_region_root,
        )
        source_full_pass_count += 1

    if not (wire_root / "role-tile-cache.json").is_file():
        RoleSpecificTileCache.prepare_from_raw_region(
            wire_region,
            wire_root,
            role="wire",
            source_identity=source_identity,
            config_digest=wire_configuration_digest,
            source_point_count=point_count,
            cell_size_m=ROLE_TILE_CELL_SIZE_M,
        )
    if not (vegetation_root / "role-tile-cache.json").is_file():
        RoleSpecificTileCache.prepare_from_raw_region(
            vegetation_region,
            vegetation_root,
            role="vegetation",
            source_identity=source_identity,
            config_digest=vegetation_configuration_digest,
            source_point_count=point_count,
            cell_size_m=ROLE_TILE_CELL_SIZE_M,
        )

    # Load once through the production validator before publishing references.
    wire_cache = RoleSpecificTileCache.load(
        wire_root,
        role="wire",
        source_identity=source_identity,
        config_digest=wire_configuration_digest,
        source_point_count=point_count,
        expected_cell_size_m=ROLE_TILE_CELL_SIZE_M,
    )
    vegetation_cache = RoleSpecificTileCache.load(
        vegetation_root,
        role="vegetation",
        source_identity=source_identity,
        config_digest=vegetation_configuration_digest,
        source_point_count=point_count,
        expected_cell_size_m=ROLE_TILE_CELL_SIZE_M,
    )
    details = {
        "schema": "treeline-stage-b-role-preparation-v1",
        "status": "PREPARED_AND_VALIDATED_ROLE_INDEXES",
        "runtime_seconds": time.perf_counter() - started,
        "source_path": str(source),
        "source_point_count": point_count,
        "coarse_voxel_size_m": voxel_size,
        "weighted_preprocess_status": preprocess_status,
        "coarse_ground_status": ground_status,
        "representative_count": int(working.representative_count),
        "wire_representative_count": int(len(wire_representative_ids)),
        "wire_strict_representative_count": int(
            np.count_nonzero(wire.candidate_mask)
        ),
        "wire_span_rescue_representative_count": int(
            np.count_nonzero(wire_admission_mask & ~wire.candidate_mask)
        ),
        "wire_admission": "STRICT_PATH_A_UNION_SPAN_ALIGNED_PATH_B_RESCUE",
        "wire_span_admission_digest": admission_digest,
        "wire_span_count": len(span_inputs),
        "wire_span_rescue_diagnostics": rescue_details,
        "wire_point_count": int(wire_cache.metadata["totals"]["point_count"]),
        "vegetation_point_count": int(
            vegetation_cache.metadata["totals"]["point_count"]
        ),
        "vegetation_admission": "HIGH_RECALL_ALL_SOURCE_NO_BROAD_WIRE_EXCLUSION",
        "source_full_pass_count": source_full_pass_count,
    }
    temporary = preparation_root / "role-preparation.json.tmp"
    temporary.write_text(
        json.dumps(details, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(preparation_root / "role-preparation.json")
    return RoleIndexPreparation(
        wire_reference=_reference(project_root, wire_root),
        vegetation_reference=_reference(project_root, vegetation_root),
        source_point_count=point_count,
        source_full_pass_count=source_full_pass_count,
        cache_hit=False,
        details=details,
    )


__all__ = ["RoleIndexPreparation", "ensure_role_tile_indexes"]
