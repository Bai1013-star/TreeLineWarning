"""FAST-only preprocess-first working sets with lossless raw-row back-mapping.

The adapter deliberately operates on a source LAS rather than Formal A1 sample
artifacts.  It reuses A1's deterministic voxel-representative principle while
adding the one property A1 does not need for its normal sample workflow: an
exclusive raw-row -> representative mapping.  A source row's original index is
its zero-based LAS row position, so the mapping is a compact ``uint32`` array
indexed by that original index.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from algorithms.vegetation.fast_research import (
    FAST_RESEARCH_SCHEMA_VERSION,
    NON_CERTIFIED_INPUT_MARKER,
    FastGroundResult,
    FastResearchError,
    FastVegetationInputResult,
    canonical_digest,
    fast_ground_content_digest,
    sha256_file,
    validate_fast_v4_arrays,
)
from algorithms.vegetation.fast_vegetation_ab import (
    FastVegetationEvidenceConfig,
    _pdal_covariance_features,
    classify_plan_b_geometry,
)
from algorithms.vegetation.fast_vegetation_v3 import (
    FastV3DiagnosticState,
    FastVegetationV3Config,
    _rigid_linear_exclusion,
    _rigid_surface_exclusion,
)

FAST_WEIGHTED_PREPROCESS_SCHEMA = "fast-weighted-preprocess-v1"
FAST_WEIGHTED_MARKER = "NON_CERTIFIED_FAST_PREPROCESS_WEIGHTED"


@dataclass(slots=True)
class _VoxelAccumulator:
    raw_count: int
    representative_original_index: int
    representative_xyz: tuple[float, float, float]
    representative_classification: int
    min_z: float
    max_z: float
    sum_z: float


@dataclass(frozen=True, slots=True)
class WeightedWorkingSet:
    """Persistent compact source representation and raw-row membership map."""

    schema_version: str
    non_certified_marker: str
    source_las_path: str
    source_las_sha256: str
    source_point_count: int
    source_bounds_min_xyz: tuple[float, float, float]
    source_bounds_max_xyz: tuple[float, float, float]
    source_scales: tuple[float, float, float]
    source_offsets: tuple[float, float, float]
    voxel_size_m: float
    representative_count: int
    raw_count_sum: int
    representatives_npz_path: str
    representative_las_path: str
    raw_to_representative_npy_path: str
    classification_counts: dict[str, int]
    elapsed_seconds: float
    config_digest: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "source_bounds_min_xyz",
            "source_bounds_max_xyz",
            "source_scales",
            "source_offsets",
        ):
            payload[key] = list(payload[key])
        return payload


def build_weighted_working_set(
    input_las: Path,
    output_root: Path,
    *,
    voxel_size_m: float,
    chunk_size: int = 1_000_000,
) -> tuple[str, WeightedWorkingSet]:
    """Build a deterministic representative set and exclusive raw membership map.

    The first streaming pass selects the first real source point in every voxel
    and accumulates its raw population.  The second streaming pass writes a
    length-N map from every source ``original_index`` to that representative.
    This is lossless for membership while never putting N raw points into V4.
    """
    input_las = Path(input_las).resolve()
    output_root = Path(output_root).resolve()
    if not input_las.is_file() or not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_INPUT_INVALID")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_CHUNK_INVALID")
    with laspy.open(input_las) as reader:
        header = reader.header
        source_count = int(header.point_count)
        if source_count <= 0:
            raise FastResearchError("FAST_WEIGHTED_PREPROCESS_EMPTY_SOURCE")
        mins = np.asarray(header.mins, dtype=np.float64)
        maxs = np.asarray(header.maxs, dtype=np.float64)
        scales = np.asarray(header.scales, dtype=np.float64)
        offsets = np.asarray(header.offsets, dtype=np.float64)
    source_sha = sha256_file(input_las)
    config_digest = canonical_digest(
        {
            "schema_version": FAST_WEIGHTED_PREPROCESS_SCHEMA,
            "source_las_sha256": source_sha,
            "voxel_size_m": float(voxel_size_m),
            "chunk_size": chunk_size,
            "representative": "first_source_row_per_xyz_voxel",
            "raw_membership": "original_index_position_to_representative_id",
        }
    )
    summary_path = output_root / "scan_large_preprocessing_summary.json"
    reps_npz = output_root / "working_set_representatives.npz"
    reps_las = output_root / "working_set_representatives.las"
    raw_mapping = output_root / "raw_to_representative_id.npy"
    cached = _load_cached_working_set(
        summary_path, reps_npz, reps_las, raw_mapping, source_sha, config_digest
    )
    if cached is not None:
        return "SKIPPED_FROM_CACHE", cached
    if any(path.exists() for path in (summary_path, reps_npz, reps_las, raw_mapping)):
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_OUTPUT_EXISTS")

    started = time.perf_counter()
    cell_shape = np.floor((maxs - mins) / voxel_size_m).astype(np.int64) + 2
    if np.any(cell_shape <= 0):
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_GRID_INVALID")
    voxels: dict[int, _VoxelAccumulator] = {}
    class_counts = np.zeros(256, dtype=np.int64)
    offset = 0
    with laspy.open(input_las) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = _chunk_xyz(chunk)
            count = len(xyz)
            original = np.arange(offset, offset + count, dtype=np.int64)
            classification = np.asarray(chunk.classification, dtype=np.uint8)
            class_counts += np.bincount(classification, minlength=256)
            keys = _voxel_key(xyz, mins, voxel_size_m, cell_shape)
            unique, first, inverse, counts = np.unique(
                keys, return_index=True, return_inverse=True, return_counts=True
            )
            z = xyz[:, 2]
            sums = np.bincount(inverse, weights=z, minlength=len(unique))
            min_z = np.full(len(unique), np.inf, dtype=np.float64)
            max_z = np.full(len(unique), -np.inf, dtype=np.float64)
            np.minimum.at(min_z, inverse, z)
            np.maximum.at(max_z, inverse, z)
            for row, key_value in enumerate(unique):
                key = int(key_value)
                accumulator = voxels.get(key)
                if accumulator is None:
                    source_position = int(first[row])
                    point = xyz[source_position]
                    voxels[key] = _VoxelAccumulator(
                        raw_count=int(counts[row]),
                        representative_original_index=int(original[source_position]),
                        representative_xyz=(float(point[0]), float(point[1]), float(point[2])),
                        representative_classification=int(classification[source_position]),
                        min_z=float(min_z[row]),
                        max_z=float(max_z[row]),
                        sum_z=float(sums[row]),
                    )
                else:
                    accumulator.raw_count += int(counts[row])
                    accumulator.min_z = min(accumulator.min_z, float(min_z[row]))
                    accumulator.max_z = max(accumulator.max_z, float(max_z[row]))
                    accumulator.sum_z += float(sums[row])
            offset += count
    if offset != source_count:
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_SOURCE_COUNT_CHANGED")
    ordered_keys = np.asarray(sorted(voxels), dtype=np.int64)
    if len(ordered_keys) > np.iinfo(np.uint32).max:
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_REPRESENTATIVE_LIMIT")
    arrays = _representative_arrays(voxels, ordered_keys, cell_shape)
    if int(np.sum(arrays["raw_count"], dtype=np.int64)) != source_count:
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_RAW_COUNT_CONSERVATION")
    _write_representative_las(
        reps_las,
        xyz=arrays["xyz_absolute"],
        classification=arrays["classification"],
        raw_count=arrays["raw_count"],
        representative_id=arrays["representative_id"],
        source_original_index=arrays["original_index"],
        scales=scales,
        offsets=offsets,
    )
    _write_npz_atomic(reps_npz, **arrays)
    _write_raw_mapping(
        input_las,
        raw_mapping,
        source_count=source_count,
        ordered_keys=ordered_keys,
        mins=mins,
        voxel_size_m=voxel_size_m,
        cell_shape=cell_shape,
        chunk_size=chunk_size,
        expected_raw_count=arrays["raw_count"],
    )
    result = WeightedWorkingSet(
        schema_version=FAST_WEIGHTED_PREPROCESS_SCHEMA,
        non_certified_marker=FAST_WEIGHTED_MARKER,
        source_las_path=str(input_las),
        source_las_sha256=source_sha,
        source_point_count=source_count,
        source_bounds_min_xyz=tuple(float(value) for value in mins),
        source_bounds_max_xyz=tuple(float(value) for value in maxs),
        source_scales=tuple(float(value) for value in scales),
        source_offsets=tuple(float(value) for value in offsets),
        voxel_size_m=float(voxel_size_m),
        representative_count=len(ordered_keys),
        raw_count_sum=int(np.sum(arrays["raw_count"], dtype=np.int64)),
        representatives_npz_path=str(reps_npz),
        representative_las_path=str(reps_las),
        raw_to_representative_npy_path=str(raw_mapping),
        classification_counts={
            str(index): int(value) for index, value in enumerate(class_counts) if value
        },
        elapsed_seconds=time.perf_counter() - started,
        config_digest=config_digest,
    )
    _write_json_atomic(summary_path, result.to_dict())
    return "PASS", result


def select_weighted_multitree_roi(
    working_set: WeightedWorkingSet,
    output_root: Path,
    *,
    coarse_cell_m: float = 2.5,
    window_m: float = 25.0,
) -> dict[str, Any]:
    """Choose a deterministic, height-rich multi-structure ROI without crown drawing."""
    if coarse_cell_m <= 0 or window_m < coarse_cell_m or not math.isfinite(window_m):
        raise FastResearchError("FAST_WEIGHTED_ROI_CONFIG_INVALID")
    output_root = Path(output_root).resolve()
    output_path = output_root / "scan_large_preprocessed_roi_bounds.json"
    with np.load(working_set.representatives_npz_path, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_absolute"], dtype=np.float64)
        raw_count = np.asarray(stream["raw_count"], dtype=np.uint32)
        representative_id = np.asarray(stream["representative_id"], dtype=np.uint32)
    if len(xyz) != working_set.representative_count:
        raise FastResearchError("FAST_WEIGHTED_ROI_REPRESENTATIVE_MISMATCH")
    origin = np.min(xyz[:, :2], axis=0)
    cell_xy = np.floor((xyz[:, :2] - origin) / coarse_cell_m).astype(np.int32)
    cols = int(cell_xy[:, 0].max()) + 1
    rows = int(cell_xy[:, 1].max()) + 1
    cell_id = cell_xy[:, 1].astype(np.int64) * cols + cell_xy[:, 0]
    max_z = np.full(rows * cols, -np.inf, dtype=np.float64)
    raw_weight = np.zeros(rows * cols, dtype=np.int64)
    np.maximum.at(max_z, cell_id, xyz[:, 2])
    np.add.at(raw_weight, cell_id, raw_count.astype(np.int64))
    observed_max = max_z[np.isfinite(max_z)]
    high_threshold = float(np.quantile(observed_max, 0.80))
    high = np.isfinite(max_z) & (max_z >= high_threshold)
    window_cells = max(2, int(round(window_m / coarse_cell_m)))
    best: tuple[float, int, int, int, int] | None = None
    for lower_row in range(max(1, rows - window_cells + 1)):
        for lower_col in range(max(1, cols - window_cells + 1)):
            upper_row = min(rows, lower_row + window_cells)
            upper_col = min(cols, lower_col + window_cells)
            ids = np.asarray(
                [
                    row * cols + col
                    for row in range(lower_row, upper_row)
                    for col in range(lower_col, upper_col)
                ],
                dtype=np.int64,
            )
            high_ids = ids[high[ids]]
            if len(high_ids) < 5:
                continue
            component_count = _high_cell_component_count(high_ids, cols)
            weight = int(np.sum(raw_weight[ids], dtype=np.int64))
            score = component_count * 1_000_000_000_000 + len(high_ids) * 1_000_000 + weight
            candidate = (float(score), component_count, len(high_ids), lower_row, lower_col)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise FastResearchError("FAST_WEIGHTED_ROI_NO_MULTITREE_REGION")
    _score, component_count, high_cells, lower_row, lower_col = best
    minimum_xy = origin + np.asarray((lower_col, lower_row), dtype=np.float64) * coarse_cell_m
    maximum_xy = origin + np.asarray(
        (min(cols, lower_col + window_cells), min(rows, lower_row + window_cells)), dtype=np.float64
    ) * coarse_cell_m
    roi_mask = (
        (xyz[:, 0] >= minimum_xy[0])
        & (xyz[:, 0] < maximum_xy[0])
        & (xyz[:, 1] >= minimum_xy[1])
        & (xyz[:, 1] < maximum_xy[1])
    )
    roi_ids = representative_id[roi_mask]
    payload = {
        "schema_version": FAST_WEIGHTED_PREPROCESS_SCHEMA,
        "non_certified_marker": FAST_WEIGHTED_MARKER,
        "selection": "deterministic_coarse_height_components_window",
        "coarse_cell_m": coarse_cell_m,
        "window_m": window_m,
        "xy_bounds_min": [float(value) for value in minimum_xy],
        "xy_bounds_max": [float(value) for value in maximum_xy],
        "z_bounds_min": float(np.min(xyz[roi_mask, 2])),
        "z_bounds_max": float(np.max(xyz[roi_mask, 2])),
        "height_threshold_z_m": high_threshold,
        "high_structure_component_count": component_count,
        "high_structure_cell_count": high_cells,
        "roi_representative_count": int(np.count_nonzero(roi_mask)),
        "roi_raw_point_count": int(np.sum(raw_count[roi_mask], dtype=np.int64)),
        "representative_ids": [int(value) for value in roi_ids],
        "why": (
            "Maximum-height coarse cells are automatically grouped; the highest-scoring "
            "25 m window contains multiple disconnected high structures rather than a "
            "hand-picked single crown."
        ),
    }
    _write_json_atomic(output_path, payload)
    return payload


def write_weighted_roi_las(
    working_set: WeightedWorkingSet, roi: dict[str, Any], output_root: Path
) -> Path:
    """Write the compact representative subset used for Ground/HAG and V4."""
    output_root = Path(output_root).resolve()
    output_path = output_root / "scan_large_roi_computation_set.las"
    roi_npz = output_root / "scan_large_roi_computation_set.npz"
    if output_path.exists() or roi_npz.exists():
        raise FastResearchError("FAST_WEIGHTED_ROI_OUTPUT_EXISTS")
    selected_ids = np.asarray(roi["representative_ids"], dtype=np.uint32)
    with np.load(working_set.representatives_npz_path, allow_pickle=False) as stream:
        ids = np.asarray(stream["representative_id"], dtype=np.uint32)
        selected = np.isin(ids, selected_ids)
        arrays = {name: np.asarray(stream[name])[selected] for name in stream.files}
    if int(np.count_nonzero(selected)) != len(selected_ids):
        raise FastResearchError("FAST_WEIGHTED_ROI_ID_MISMATCH")
    _write_representative_las(
        output_path,
        xyz=np.asarray(arrays["xyz_absolute"], dtype=np.float64),
        classification=np.asarray(arrays["classification"], dtype=np.uint8),
        raw_count=np.asarray(arrays["raw_count"], dtype=np.uint32),
        representative_id=np.asarray(arrays["representative_id"], dtype=np.uint32),
        source_original_index=np.asarray(arrays["original_index"], dtype=np.int64),
        scales=np.asarray(working_set.source_scales, dtype=np.float64),
        offsets=np.asarray(working_set.source_offsets, dtype=np.float64),
    )
    _write_npz_atomic(roi_npz, **arrays)
    return output_path


def build_weighted_fast_vegetation(
    working_set: WeightedWorkingSet,
    roi: dict[str, Any],
    ground: FastGroundResult,
    output_root: Path,
) -> tuple[FastVegetationInputResult, dict[str, Any]]:
    """Build V3-style positive vegetation evidence on compact representatives.

    ``scan_large`` has no native W1 artifact, so this function never invents a
    wire model.  It records that condition explicitly and applies the existing
    PDAL covariance evidence plus V3 rigid component exclusions.
    """
    output_root = Path(output_root).resolve()
    output_npz = output_root / "fast_v4_input.npz"
    diagnostic_npz = output_root / "scan_large_fast_vegetation_diagnostics.npz"
    manifest_path = output_root / "scan_large_fast_vegetation_diagnostics.json"
    if any(path.exists() for path in (output_npz, diagnostic_npz, manifest_path)):
        raise FastResearchError("FAST_WEIGHTED_VEGETATION_OUTPUT_EXISTS")
    started = time.perf_counter()
    las = laspy.read(Path(ground.output_las_path))
    xyz_absolute = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    hag = np.asarray(las.HeightAboveGround, dtype=np.float32)
    classification = np.asarray(las.classification, dtype=np.uint8)
    original = np.asarray(las.source_original_index, dtype=np.int64)
    raw_count = np.asarray(las.raw_count, dtype=np.uint32)
    representative_id = np.asarray(las.representative_id, dtype=np.uint32)
    count = len(original)
    if any(
        values.shape != (count,)
        for values in (hag, classification, raw_count, representative_id)
    ):
        raise FastResearchError("FAST_WEIGHTED_VEGETATION_ALIGNMENT_INVALID")
    if np.any(raw_count == 0) or len(np.unique(original)) != count:
        raise FastResearchError("FAST_WEIGHTED_VEGETATION_MEMBERSHIP_INVALID")
    origin = np.asarray(working_set.source_bounds_min_xyz, dtype=np.float64)
    xyz_local = xyz_absolute - origin
    ground_mask = classification == 2
    hag_valid = np.isfinite(hag) & (hag >= 0.3)
    geometry_candidate = ~ground_mask & hag_valid
    features = {
        "linearity": np.zeros(count, dtype=np.float32),
        "planarity": np.zeros(count, dtype=np.float32),
        "scattering": np.zeros(count, dtype=np.float32),
        "verticality": np.zeros(count, dtype=np.float32),
    }
    feature_elapsed = 0.0
    candidate_rows = np.flatnonzero(geometry_candidate)
    if len(candidate_rows) >= 16:
        measured, feature_elapsed = _pdal_covariance_features(
            xyz_local[candidate_rows], config=FastVegetationEvidenceConfig()
        )
        for name, values in measured.items():
            features[name][candidate_rows] = values
    positive_evidence = np.zeros(count, dtype=bool)
    if len(candidate_rows):
        positive_evidence[candidate_rows] = classify_plan_b_geometry(
            features["scattering"][candidate_rows], config=FastVegetationEvidenceConfig()
        )
    base = geometry_candidate & positive_evidence
    v3_config = FastVegetationV3Config()
    rigid_linear, linear_component, linear_summaries = _rigid_linear_exclusion(
        xyz_local,
        base,
        hag=hag,
        linearity=features["linearity"],
        scattering=features["scattering"],
        config=v3_config,
    )
    rigid_surface, surface_component, surface_summaries = _rigid_surface_exclusion(
        xyz_local,
        base & ~rigid_linear,
        hag=hag,
        planarity=features["planarity"],
        scattering=features["scattering"],
        config=v3_config,
    )
    retained = base & ~rigid_linear & ~rigid_surface
    state = np.full(
        count,
        int(FastV3DiagnosticState.NO_POSITIVE_VEGETATION_EVIDENCE),
        dtype=np.uint8,
    )
    state[retained] = int(FastV3DiagnosticState.VEGETATION_RETAINED)
    state[rigid_surface] = int(FastV3DiagnosticState.RIGID_SURFACE_EXCLUDED)
    state[rigid_linear] = int(FastV3DiagnosticState.RIGID_LINEAR_EXCLUDED)
    state[ground_mask] = int(FastV3DiagnosticState.GROUND_EXCLUDED)
    clean_original = original[retained]
    clean_xyz = xyz_local[retained]
    clean_hag = hag[retained]
    clean_weight = raw_count[retained]
    clean_original, clean_xyz, clean_hag, clean_weight = _order_v4_input_arrays(
        clean_original, clean_xyz, clean_hag, clean_weight
    )
    validate_fast_v4_arrays(clean_original, clean_xyz, clean_hag)
    if len(clean_original) == 0:
        raise FastResearchError("FAST_WEIGHTED_VEGETATION_EMPTY")
    config_payload = {
        "schema_version": FAST_WEIGHTED_PREPROCESS_SCHEMA,
        "ground_digest": fast_ground_content_digest(ground),
        "working_set_config_digest": working_set.config_digest,
        "roi_bounds": roi["xy_bounds_min"] + roi["xy_bounds_max"],
        "positive_evidence": FastVegetationEvidenceConfig().to_dict(),
        "rigid_config": asdict(v3_config),
        "wire_mode": "NO_NATIVE_W1_AVAILABLE",
    }
    config_digest = canonical_digest(config_payload)
    _write_npz_atomic(
        output_npz,
        original_index=clean_original.astype("<i8", copy=False),
        xyz_local_m=clean_xyz.astype("<f8", copy=False),
        hag_m=clean_hag.astype("<f4", copy=False),
        raw_count=clean_weight.astype("<u4", copy=False),
        candidate_reason_flags=np.zeros(len(clean_original), dtype="<u4"),
    )
    _write_npz_atomic(
        diagnostic_npz,
        original_index=original.astype("<i8", copy=False),
        representative_id=representative_id.astype("<u4", copy=False),
        xyz_local_m=xyz_local.astype("<f8", copy=False),
        hag_m=hag.astype("<f4", copy=False),
        raw_count=raw_count.astype("<u4", copy=False),
        diagnostic_state=state,
        positive_evidence=positive_evidence,
        linearity=features["linearity"],
        planarity=features["planarity"],
        scattering=features["scattering"],
        verticality=features["verticality"],
        rigid_linear_component_id=linear_component.astype("<i4", copy=False),
        rigid_surface_component_id=surface_component.astype("<i4", copy=False),
    )
    vegetation = FastVegetationInputResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_INPUT_MARKER,
        ground_result_digest=fast_ground_content_digest(ground),
        config_digest=config_digest,
        output_npz_path=str(output_npz),
        output_npz_sha256=sha256_file(output_npz),
        point_count=len(clean_original),
        source_id="scan_large_preprocessed",
        source_las_path=working_set.source_las_path,
        source_point_count=working_set.source_point_count,
        source_file_sha256=working_set.source_las_sha256,
        origin_context=tuple(float(value) for value in origin),
        selected_bounds_min_xyz=(
            float(roi["xy_bounds_min"][0]),
            float(roi["xy_bounds_min"][1]),
            float(roi["z_bounds_min"]),
        ),
        selected_bounds_max_xyz=(
            float(roi["xy_bounds_max"][0]),
            float(roi["xy_bounds_max"][1]),
            float(roi["z_bounds_max"]),
        ),
        minimum_hag_m=0.3,
        admission_voxel_size_m=working_set.voxel_size_m,
        original_index_sha256=_array_digest(clean_original),
        xyz_sha256=_array_digest(clean_xyz),
        hag_sha256=_array_digest(clean_hag),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json_atomic(output_root / "fast_v4_input.json", vegetation.to_dict())
    manifest = {
        "schema_version": FAST_WEIGHTED_PREPROCESS_SCHEMA,
        "non_certified_marker": FAST_WEIGHTED_MARKER,
        "wire_mode": "NO_NATIVE_W1_AVAILABLE",
        "config_digest": config_digest,
        "formula": (
            "NON_GROUND AND HAG_READY AND POSITIVE_PDAL_COVARIANCE_EVIDENCE "
            "AND NOT_RIGID_COMPONENT"
        ),
        "counts": {
            "input_representatives": count,
            "input_raw_weight": int(np.sum(raw_count, dtype=np.int64)),
            "ground_excluded": int(
                np.count_nonzero(state == FastV3DiagnosticState.GROUND_EXCLUDED)
            ),
            "exact_wire_excluded": 0,
            "wire_geometry_excluded": 0,
            "rigid_linear_excluded": int(np.count_nonzero(rigid_linear)),
            "rigid_surface_excluded": int(np.count_nonzero(rigid_surface)),
            "no_positive_vegetation_evidence_excluded": int(
                np.count_nonzero(state == FastV3DiagnosticState.NO_POSITIVE_VEGETATION_EVIDENCE)
            ),
            "final_vegetation_representatives": int(np.count_nonzero(retained)),
            "final_vegetation_raw_weight": int(np.sum(clean_weight, dtype=np.int64)),
        },
        "feature_runtime_seconds": feature_elapsed,
        "linear_components": linear_summaries,
        "surface_components": surface_summaries,
        "vegetation_input": vegetation.to_dict(),
    }
    _write_json_atomic(manifest_path, manifest)
    return vegetation, manifest


def backmap_representative_tree_ids(
    working_set: WeightedWorkingSet,
    roi_root: Path,
    vegetation_npz: Path,
    v4_npz: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Persist the exact representative -> raw membership tree back-map."""
    roi_root = Path(roi_root)
    with np.load(roi_root / "scan_large_roi_computation_set.npz", allow_pickle=False) as stream:
        roi_original = np.asarray(stream["original_index"], dtype=np.int64)
        roi_ids = np.asarray(stream["representative_id"], dtype=np.uint32)
        roi_weight = np.asarray(stream["raw_count"], dtype=np.uint32)
    with np.load(vegetation_npz, allow_pickle=False) as stream:
        vegetation_original = np.asarray(stream["original_index"], dtype=np.int64)
        vegetation_weight = np.asarray(stream["raw_count"], dtype=np.uint32)
    with np.load(v4_npz, allow_pickle=False) as stream:
        v4_original = np.asarray(stream["original_index"], dtype=np.int64)
        v4_tree = np.asarray(stream["tree_id_by_point"], dtype=np.int32)
    order = np.argsort(vegetation_original, kind="stable")
    positions = np.searchsorted(vegetation_original[order], v4_original)
    if np.any(positions >= len(order)) or not np.array_equal(
        vegetation_original[order][positions], v4_original
    ):
        raise FastResearchError("FAST_WEIGHTED_BACKMAP_V4_INDEX_MISMATCH")
    tree_by_original = np.full(len(roi_original), -1, dtype=np.int32)
    roi_order = np.argsort(roi_original, kind="stable")
    roi_positions = np.searchsorted(roi_original[roi_order], v4_original)
    if np.any(roi_positions >= len(roi_order)) or not np.array_equal(
        roi_original[roi_order][roi_positions], v4_original
    ):
        raise FastResearchError("FAST_WEIGHTED_BACKMAP_ROI_INDEX_MISMATCH")
    tree_by_original[roi_order[roi_positions]] = v4_tree
    if int(np.sum(vegetation_weight, dtype=np.int64)) <= 0:
        raise FastResearchError("FAST_WEIGHTED_BACKMAP_EMPTY_WEIGHT")
    _write_npz_atomic(
        output_path,
        representative_id=roi_ids,
        source_original_index=roi_original,
        raw_count=roi_weight,
        tree_id_by_representative=tree_by_original,
    )
    assigned = tree_by_original >= 0
    return {
        "representative_count": int(len(roi_ids)),
        "raw_count_sum": int(np.sum(roi_weight, dtype=np.int64)),
        "v4_input_raw_weight": int(np.sum(vegetation_weight, dtype=np.int64)),
        "assigned_raw_weight": int(np.sum(roi_weight[assigned], dtype=np.int64)),
        "unassigned_raw_weight": int(np.sum(roi_weight[~assigned], dtype=np.int64)),
        "assigned_representatives": int(np.count_nonzero(assigned)),
        "unassigned_representatives": int(np.count_nonzero(~assigned)),
        "mapping_path": str(output_path),
        "mapping_semantics": (
            "raw_original_index is the position into raw_to_representative_id.npy; every "
            "source row inherits its exclusive representative tree_id."
        ),
    }


def _chunk_xyz(chunk: Any) -> np.ndarray:
    return np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64, copy=False)


def _order_v4_input_arrays(
    original_index: np.ndarray,
    xyz_local_m: np.ndarray,
    hag_m: np.ndarray,
    raw_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Order compact rows by source position without breaking their alignment."""

    order = np.argsort(original_index, kind="stable")
    return (
        original_index[order],
        xyz_local_m[order],
        hag_m[order],
        raw_count[order],
    )


def _voxel_key(
    xyz: np.ndarray, mins: np.ndarray, voxel_size_m: float, cell_shape: np.ndarray
) -> np.ndarray:
    cell = np.floor((xyz - mins) / voxel_size_m).astype(np.int64)
    if np.any(cell < 0) or np.any(cell >= cell_shape):
        raise FastResearchError("FAST_WEIGHTED_PREPROCESS_VOXEL_OUT_OF_RANGE")
    return (cell[:, 0] * cell_shape[1] + cell[:, 1]) * cell_shape[2] + cell[:, 2]


def _representative_arrays(
    voxels: dict[int, _VoxelAccumulator], ordered_keys: np.ndarray, cell_shape: np.ndarray
) -> dict[str, np.ndarray]:
    count = len(ordered_keys)
    xyz = np.empty((count, 3), dtype=np.float64)
    original = np.empty(count, dtype=np.int64)
    raw_count = np.empty(count, dtype=np.uint32)
    classification = np.empty(count, dtype=np.uint8)
    minimum_z = np.empty(count, dtype=np.float32)
    maximum_z = np.empty(count, dtype=np.float32)
    mean_z = np.empty(count, dtype=np.float32)
    for row, key_value in enumerate(ordered_keys):
        accumulator = voxels[int(key_value)]
        xyz[row] = accumulator.representative_xyz
        original[row] = accumulator.representative_original_index
        raw_count[row] = accumulator.raw_count
        classification[row] = accumulator.representative_classification
        minimum_z[row] = accumulator.min_z
        maximum_z[row] = accumulator.max_z
        mean_z[row] = accumulator.sum_z / accumulator.raw_count
    x_cell = ordered_keys // (cell_shape[1] * cell_shape[2])
    remainder = ordered_keys % (cell_shape[1] * cell_shape[2])
    y_cell = remainder // cell_shape[2]
    z_cell = remainder % cell_shape[2]
    return {
        "representative_id": np.arange(count, dtype=np.uint32),
        "original_index": original,
        "xyz_absolute": xyz,
        "raw_count": raw_count,
        "classification": classification,
        "voxel_key": np.column_stack((x_cell, y_cell, z_cell)).astype(np.int32),
        "raw_min_z": minimum_z,
        "raw_max_z": maximum_z,
        "raw_mean_z": mean_z,
    }


def _write_raw_mapping(
    input_las: Path,
    output_path: Path,
    *,
    source_count: int,
    ordered_keys: np.ndarray,
    mins: np.ndarray,
    voxel_size_m: float,
    cell_shape: np.ndarray,
    chunk_size: int,
    expected_raw_count: np.ndarray,
) -> None:
    temporary = output_path.with_name(output_path.stem + ".tmp.npy")
    mapping = np.lib.format.open_memmap(temporary, mode="w+", dtype="<u4", shape=(source_count,))
    observed = np.zeros(len(ordered_keys), dtype=np.int64)
    offset = 0
    try:
        with laspy.open(input_las) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                xyz = _chunk_xyz(chunk)
                keys = _voxel_key(xyz, mins, voxel_size_m, cell_shape)
                positions = np.searchsorted(ordered_keys, keys)
                if np.any(positions >= len(ordered_keys)) or not np.array_equal(
                    ordered_keys[positions], keys
                ):
                    raise FastResearchError("FAST_WEIGHTED_PREPROCESS_MAPPING_KEY_MISSING")
                mapping[offset : offset + len(xyz)] = positions.astype(np.uint32, copy=False)
                observed += np.bincount(positions, minlength=len(ordered_keys)).astype(np.int64)
                offset += len(xyz)
        mapping.flush()
        if offset != source_count or not np.array_equal(
            observed, expected_raw_count.astype(np.int64)
        ):
            raise FastResearchError("FAST_WEIGHTED_PREPROCESS_MAPPING_CONSERVATION")
        # Windows keeps the mmap's file handle open until the object is released.
        # Close it before atomically publishing the completed mapping.
        del mapping
        mapping = None
        os.replace(temporary, output_path)
    except Exception:
        if mapping is not None:
            del mapping
        temporary.unlink(missing_ok=True)
        raise


def _write_representative_las(
    path: Path,
    *,
    xyz: np.ndarray,
    classification: np.ndarray,
    raw_count: np.ndarray,
    representative_id: np.ndarray,
    source_original_index: np.ndarray,
    scales: np.ndarray,
    offsets: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = scales
    header.offsets = offsets
    header.add_extra_dim(laspy.ExtraBytesParams(name="raw_count", type=np.uint32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="representative_id", type=np.uint32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="source_original_index", type=np.uint64))
    data = laspy.LasData(header)
    data.x, data.y, data.z = xyz.T
    data.classification = classification
    data.raw_count = raw_count
    data.representative_id = representative_id
    data.source_original_index = source_original_index.astype(np.uint64, copy=False)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    data.write(temporary)
    os.replace(temporary, path)


def _high_cell_component_count(high_ids: np.ndarray, cols: int) -> int:
    remaining = {int(value) for value in high_ids}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            row, col = divmod(current, cols)
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    if drow == 0 and dcol == 0:
                        continue
                    neighbor_row = row + drow
                    neighbor_col = col + dcol
                    if neighbor_row < 0 or neighbor_col < 0 or neighbor_col >= cols:
                        continue
                    neighbor = neighbor_row * cols + neighbor_col
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return components


def _array_digest(values: np.ndarray) -> str:
    array = np.asarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8")
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _load_cached_working_set(
    summary_path: Path,
    reps_npz: Path,
    reps_las: Path,
    raw_mapping: Path,
    source_sha: str,
    config_digest: str,
) -> WeightedWorkingSet | None:
    if not all(path.is_file() for path in (summary_path, reps_npz, reps_las, raw_mapping)):
        return None
    try:
        payload = json.loads(summary_path.read_text("utf-8"))
        for key in (
            "source_bounds_min_xyz",
            "source_bounds_max_xyz",
            "source_scales",
            "source_offsets",
        ):
            payload[key] = tuple(float(value) for value in payload[key])
        result = WeightedWorkingSet(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if result.source_las_sha256 != source_sha or result.config_digest != config_digest:
        return None
    if (
        Path(result.representatives_npz_path) != reps_npz
        or Path(result.representative_las_path) != reps_las
        or Path(result.raw_to_representative_npy_path) != raw_mapping
    ):
        return None
    return result
