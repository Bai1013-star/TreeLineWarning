"""A2B deterministic grid ground estimation and height-above-ground baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import (
    CancellationToken,
    FeatureQualityFlag,
    GroundEstimationConfig,
    GroundEstimationResult,
    GroundEstimationStatistics,
    GroundQualityFlag,
    GroundSurfaceModel,
    HeightAboveGroundSet,
    PipelineProgress,
    ProgressReporter,
    UnifiedError,
)
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "a2b-grid-quantile-idw-v1"
OUTPUT_BASENAME = "scene_simple_ground_hag"
FORMULAS: dict[str, str] = {
    "grid_origin_xy": "floor(min_xy / grid_size_m) * grid_size_m",
    "grid_index": "ix=floor((x-origin_x)/grid_size_m), iy=floor((y-origin_y)/grid_size_m)",
    "ground_cell_id": "iy * grid_width + ix",
    "ground_control_z": "quantile(cell_z, ground_quantile, method='linear')",
    "idw": "sum(z_i / d_i**idw_power) / sum(1 / d_i**idw_power)",
    "ground_distance_m": "grid-cell-center distance to nearest direct control cell center",
    "raw_height_agl_m": "xyz_local[:, 2] - ground_z_m",
    "height_agl_m": "maximum(raw_height_agl_m, 0)",
}


class GroundEstimationError(Exception):
    """A recoverable A2B failure with a structured project error."""

    def __init__(self, error: UnifiedError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True)
class _A2BInput:
    a1_npz_path: Path
    a1_json_path: Path
    a2a_npz_path: Path
    a2a_json_path: Path
    a1_npz_sha256: str
    a1_json_sha256: str
    a2a_npz_sha256: str
    a2a_json_sha256: str
    source_id: str
    manifest_source_id: str
    xyz_local: np.ndarray
    origin_xyz: np.ndarray
    original_index: np.ndarray
    sample_index: np.ndarray
    classification: np.ndarray
    a2a_quality_flags: np.ndarray


@dataclass(frozen=True)
class _ComputedGround:
    surface: GroundSurfaceModel
    hag: HeightAboveGroundSet
    classification_validation: Mapping[str, Any]


def run_ground_height_estimation(
    a1_npz_path: Path | str,
    a1_json_path: Path | str,
    a2a_npz_path: Path | str,
    a2a_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: GroundEstimationConfig | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
) -> GroundEstimationResult:
    """Estimate a grid ground surface from A1/A2A artifacts and write A2B output."""

    effective_config = config or GroundEstimationConfig()
    if effective_config.smoothing_radius_cells != 0:
        raise _ground_error(
            code="GROUND_SMOOTHING_UNSUPPORTED",
            message="A2B v1 only supports smoothing_radius_cells=0",
            details={"smoothing_radius_cells": effective_config.smoothing_radius_cells},
        )
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output = Path(output_dir).resolve()
    _validate_output_dir(output, project_root=root)
    output_npz = output / f"{OUTPUT_BASENAME}.npz"
    output_json = output / f"{OUTPUT_BASENAME}.json"
    _validate_outputs_absent((output_npz, output_json), overwrite=overwrite)
    _check_cancelled(cancellation_token, "before A2B input validation")

    start_time = time.perf_counter()
    process = _current_process() if effective_config.record_memory else None
    rss_start = _rss_bytes(process)
    rss_peak = rss_start
    input_data = _load_inputs(
        Path(a1_npz_path),
        Path(a1_json_path),
        Path(a2a_npz_path),
        Path(a2a_json_path),
        config=effective_config,
    )
    _report(reporter, input_data.source_id, "validated_inputs", 0, len(input_data.xyz_local))
    _check_cancelled(cancellation_token, "before A2B grid construction")
    rss_peak = _max_rss(rss_peak, process)

    computed = _compute_ground(
        input_data,
        config=effective_config,
        reporter=reporter,
        cancellation_token=cancellation_token,
    )
    rss_peak = _max_rss(rss_peak, process)
    _validate_output_arrays(computed.surface, computed.hag)
    elapsed_seconds = time.perf_counter() - start_time
    statistics = _build_statistics(
        computed.surface,
        computed.hag,
        config=effective_config,
        elapsed_seconds=elapsed_seconds,
        rss_start=rss_start,
        rss_peak=rss_peak,
        rss_end=_rss_bytes(process),
        classification_validation=computed.classification_validation,
    )
    _check_cancelled(cancellation_token, "before A2B output write")
    result = _write_outputs_atomic(
        output_npz,
        output_json,
        input_data=input_data,
        surface=computed.surface,
        hag=computed.hag,
        config=effective_config,
        statistics=statistics,
        classification_validation=computed.classification_validation,
        overwrite=overwrite,
        cancellation_token=cancellation_token,
    )
    _report(
        reporter,
        input_data.source_id,
        "complete",
        computed.hag.point_count,
        computed.hag.point_count,
        status="complete",
    )
    return result


def _load_inputs(
    a1_npz_path: Path,
    a1_json_path: Path,
    a2a_npz_path: Path,
    a2a_json_path: Path,
    *,
    config: GroundEstimationConfig,
) -> _A2BInput:
    paths = (a1_npz_path, a1_json_path, a2a_npz_path, a2a_json_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise _ground_error(
            code="GROUND_INPUT_MISSING",
            message="A2B requires existing A1 and A2A NPZ/JSON inputs",
            details={"paths": missing},
        )
    try:
        a1_metadata = json.loads(a1_json_path.read_text(encoding="utf-8"))
        a2a_metadata = json.loads(a2a_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _ground_error(
            code="GROUND_INPUT_JSON_INVALID",
            message="A2B input JSON could not be loaded",
            details={},
            cause=repr(error),
        ) from error
    _validate_metadata_paths(a1_metadata, a1_npz_path, a1_json_path, stage="A1")
    _validate_metadata_paths(a2a_metadata, a2a_npz_path, a2a_json_path, stage="A2A")
    try:
        with np.load(a1_npz_path, allow_pickle=False) as data:
            a1 = {name: data[name] for name in data.files}
        with np.load(a2a_npz_path, allow_pickle=False) as data:
            a2a = {name: data[name] for name in data.files}
    except (OSError, ValueError) as error:
        raise _ground_error(
            code="GROUND_INPUT_NPZ_INVALID",
            message="A2B input NPZ could not be loaded",
            details={},
            cause=repr(error),
        ) from error
    _validate_input_arrays(a1, a2a, a1_metadata, a2a_metadata, config=config)
    source_id = _scalar_text(a1["source_id"], "A1 source_id")
    manifest_source_id = _scalar_text(a1["manifest_source_id"], "A1 manifest_source_id")
    return _A2BInput(
        a1_npz_path=a1_npz_path.resolve(),
        a1_json_path=a1_json_path.resolve(),
        a2a_npz_path=a2a_npz_path.resolve(),
        a2a_json_path=a2a_json_path.resolve(),
        a1_npz_sha256=_sha256_file(a1_npz_path),
        a1_json_sha256=_sha256_file(a1_json_path),
        a2a_npz_sha256=_sha256_file(a2a_npz_path),
        a2a_json_sha256=_sha256_file(a2a_json_path),
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        xyz_local=a1["xyz_local"],
        origin_xyz=a1["origin_xyz"],
        original_index=a1["original_index"],
        sample_index=a1["sample_index"],
        classification=a1["classification"],
        a2a_quality_flags=a2a["quality_flags"],
    )


def _validate_metadata_paths(
    metadata: Mapping[str, Any], npz_path: Path, json_path: Path, *, stage: str
) -> None:
    required = {"source_id", "output_npz_path", "output_json_path", "origin_xyz", "arrays"}
    if stage == "A2A":
        required |= {
            "manifest_source_id",
            "input_a1_npz_sha256",
            "input_a1_json_sha256",
            "output_npz_sha256",
        }
    missing = sorted(required - set(metadata))
    if missing:
        raise _ground_error(
            code="GROUND_INPUT_METADATA_INCOMPLETE",
            message=f"{stage} JSON metadata is missing required fields",
            details={"missing": missing},
        )
    if (
        Path(str(metadata["output_npz_path"])).resolve() != npz_path.resolve()
        or Path(str(metadata["output_json_path"])).resolve() != json_path.resolve()
    ):
        raise _ground_error(
            code="GROUND_INPUT_PATH_MISMATCH",
            message=f"{stage} JSON output paths do not match supplied input paths",
            details={"npz_path": str(npz_path), "json_path": str(json_path)},
        )


def _validate_input_arrays(
    a1: Mapping[str, np.ndarray],
    a2a: Mapping[str, np.ndarray],
    a1_metadata: Mapping[str, Any],
    a2a_metadata: Mapping[str, Any],
    *,
    config: GroundEstimationConfig,
) -> None:
    a1_required = {
        "source_id",
        "manifest_source_id",
        "xyz_local",
        "origin_xyz",
        "original_index",
        "sample_index",
        "classification",
    }
    a2a_required = {
        "source_id",
        "manifest_source_id",
        "xyz_local",
        "origin_xyz",
        "original_index",
        "sample_index",
        "planarity",
        "verticality",
        "scattering",
        "normal",
        "height_m",
        "quality_flags",
    }
    for stage, arrays, required in (("A1", a1, a1_required), ("A2A", a2a, a2a_required)):
        missing = sorted(required - set(arrays))
        if missing:
            raise _ground_error(
                code="GROUND_INPUT_ARRAY_MISSING",
                message=f"{stage} NPZ is missing required arrays",
                details={"missing": missing},
            )
    xyz = a1["xyz_local"]
    point_count = len(xyz)
    expected_a1 = {
        "xyz_local": ((point_count, 3), np.dtype(np.float32)),
        "origin_xyz": ((3,), np.dtype(np.float64)),
        "original_index": ((point_count,), np.dtype(np.int64)),
        "sample_index": ((point_count,), np.dtype(np.int64)),
        "classification": ((point_count,), np.dtype(np.uint8)),
    }
    expected_a2a = {
        "xyz_local": ((point_count, 3), np.dtype(np.float32)),
        "origin_xyz": ((3,), np.dtype(np.float64)),
        "original_index": ((point_count,), np.dtype(np.int64)),
        "sample_index": ((point_count,), np.dtype(np.int64)),
        "planarity": ((point_count,), np.dtype(np.float32)),
        "verticality": ((point_count,), np.dtype(np.float32)),
        "scattering": ((point_count,), np.dtype(np.float32)),
        "normal": ((point_count, 3), np.dtype(np.float32)),
        "height_m": ((point_count,), np.dtype(np.float32)),
        "quality_flags": ((point_count,), np.dtype(np.uint16)),
    }
    for stage, arrays, expected in (("A1", a1, expected_a1), ("A2A", a2a, expected_a2a)):
        for name, (shape, dtype) in expected.items():
            value = arrays[name]
            if value.shape != shape or value.dtype != dtype:
                raise _ground_error(
                    code="GROUND_INPUT_ARRAY_SHAPE_DTYPE",
                    message=f"{stage} input array has unexpected shape or dtype",
                    details={"array": name, "shape": list(value.shape), "dtype": str(value.dtype)},
                )
    if point_count == 0:
        raise _ground_error(
            code="GROUND_INPUT_EMPTY", message="A2B input point cloud is empty", details={}
        )
    a1_source = _scalar_text(a1["source_id"], "A1 source_id")
    a2a_source = _scalar_text(a2a["source_id"], "A2A source_id")
    a1_manifest = _scalar_text(a1["manifest_source_id"], "A1 manifest_source_id")
    a2a_manifest = _scalar_text(a2a["manifest_source_id"], "A2A manifest_source_id")
    if a1_source != a2a_source or a1_source != config.source_id or a1_manifest != a2a_manifest:
        raise _ground_error(
            code="GROUND_INPUT_SOURCE_MISMATCH",
            message="A1/A2A source provenance is inconsistent",
            details={"a1_source": a1_source, "a2a_source": a2a_source},
        )
    if str(a1_metadata["source_id"]) != a1_source or str(a2a_metadata["source_id"]) != a1_source:
        raise _ground_error(
            code="GROUND_INPUT_METADATA_SOURCE_MISMATCH",
            message="A1/A2A JSON source_id is inconsistent with NPZ",
            details={},
        )
    if str(a2a_metadata["manifest_source_id"]) != a1_manifest:
        raise _ground_error(
            code="GROUND_INPUT_MANIFEST_MISMATCH",
            message="A2A JSON manifest source is inconsistent with A1",
            details={},
        )
    for name in ("xyz_local", "origin_xyz", "original_index", "sample_index"):
        if not np.array_equal(a1[name], a2a[name]):
            raise _ground_error(
                code="GROUND_INPUT_ALIGNMENT_MISMATCH",
                message="A1 and A2A arrays are not element-wise aligned",
                details={"array": name},
            )
    numeric_arrays = (
        a1["xyz_local"],
        a1["origin_xyz"],
        a2a["planarity"],
        a2a["verticality"],
        a2a["scattering"],
        a2a["normal"],
        a2a["height_m"],
    )
    if not all(np.isfinite(value).all() for value in numeric_arrays):
        raise _ground_error(
            code="GROUND_INPUT_NONFINITE",
            message="A1/A2A input arrays contain NaN or Inf",
            details={},
        )
    if np.any(a1["original_index"] < 0) or np.any(a1["sample_index"] < 0):
        raise _ground_error(
            code="GROUND_INPUT_INDEX_NEGATIVE",
            message="A1 index mappings must be non-negative",
            details={},
        )
    if (
        len(np.unique(a1["original_index"])) != point_count
        or len(np.unique(a1["sample_index"])) != point_count
    ):
        raise _ground_error(
            code="GROUND_INPUT_INDEX_NONUNIQUE",
            message="A1 index mappings must be unique",
            details={},
        )
    if np.any(a2a["quality_flags"] & int(FeatureQualityFlag.NUMERIC_INVALID)):
        raise _ground_error(
            code="GROUND_A2A_NUMERIC_INVALID",
            message="A2A contains numeric-invalid points and cannot seed A2B",
            details={
                "count": int(
                    np.count_nonzero(a2a["quality_flags"] & int(FeatureQualityFlag.NUMERIC_INVALID))
                )
            },
        )
    actual_a1_npz = _sha256_file(Path(str(a1_metadata["output_npz_path"])))
    actual_a1_json = _sha256_file(Path(str(a1_metadata["output_json_path"])))
    if actual_a1_npz != str(a2a_metadata["input_a1_npz_sha256"]) or actual_a1_json != str(
        a2a_metadata["input_a1_json_sha256"]
    ):
        raise _ground_error(
            code="GROUND_INPUT_SHA_MISMATCH",
            message="A2A declared A1 SHA-256 does not match supplied A1 artifacts",
            details={},
        )
    if _sha256_file(Path(str(a2a_metadata["output_npz_path"]))) != str(
        a2a_metadata["output_npz_sha256"]
    ):
        raise _ground_error(
            code="GROUND_A2A_SHA_MISMATCH",
            message="A2A NPZ SHA-256 does not match its JSON declaration",
            details={},
        )


def _compute_ground(
    input_data: _A2BInput,
    *,
    config: GroundEstimationConfig,
    reporter: ProgressReporter | None,
    cancellation_token: CancellationToken | None,
) -> _ComputedGround:
    xyz = input_data.xyz_local
    xy = xyz[:, :2].astype(np.float64, copy=False)
    z = xyz[:, 2].astype(np.float64, copy=False)
    grid_origin = np.floor(np.min(xy, axis=0) / config.grid_size_m) * config.grid_size_m
    relative = (xy - grid_origin) / config.grid_size_m
    tolerance = 1e-6
    if np.any(relative < -tolerance) or not np.isfinite(relative).all():
        raise _ground_error(
            code="GROUND_GRID_COORDINATE_INVALID",
            message="Point coordinates cannot be assigned to the ground grid",
            details={},
        )
    indices = np.floor(np.maximum(relative, 0.0)).astype(np.int64)
    grid_width = int(indices[:, 0].max()) + 1
    grid_height = int(indices[:, 1].max()) + 1
    total_cells = grid_width * grid_height
    if total_cells > np.iinfo(np.int32).max:
        raise _ground_error(
            code="GROUND_GRID_TOO_LARGE",
            message="Ground grid cannot be represented by int32 ground_cell_id",
            details={"total_cells": total_cells},
        )
    point_cell_id = indices[:, 1] * grid_width + indices[:, 0]
    if np.any(point_cell_id < 0) or np.any(point_cell_id >= total_cells):
        raise _ground_error(
            code="GROUND_GRID_INDEX_INVALID",
            message="Ground cell index is outside the computed grid",
            details={},
        )
    counts_flat = np.bincount(point_cell_id, minlength=total_cells).astype(np.int32, copy=False)
    cell_ground_flat = np.full(total_cells, np.nan, dtype=np.float32)
    cell_flags_flat = np.zeros(total_cells, dtype=np.uint16)
    empty = counts_flat == 0
    sparse = (counts_flat > 0) & (counts_flat < config.min_points_per_cell)
    controls = counts_flat >= config.min_points_per_cell
    cell_flags_flat[empty] |= int(GroundQualityFlag.EMPTY_GROUND_CELL)
    cell_flags_flat[sparse] |= int(GroundQualityFlag.SPARSE_GROUND_CELL)
    if not np.any(controls):
        raise _ground_error(
            code="GROUND_CONTROL_EMPTY",
            message="A2B found no grid cells with enough points for ground controls",
            details={"min_points_per_cell": config.min_points_per_cell},
        )
    order = np.argsort(point_cell_id, kind="stable")
    boundaries = np.concatenate(([0], np.cumsum(counts_flat, dtype=np.int64)))
    for cell_id in np.flatnonzero(controls):
        _check_cancelled(cancellation_token, "while calculating ground quantiles")
        members = order[boundaries[cell_id] : boundaries[cell_id + 1]]
        cell_ground_flat[cell_id] = np.float32(
            np.quantile(z[members], config.ground_quantile, method=config.quantile_method)
        )
    _report(reporter, input_data.source_id, "ground_controls", len(xyz), len(xyz))
    _check_cancelled(cancellation_token, "before A2B IDW interpolation")

    grid_ids = np.arange(total_cells, dtype=np.int64)
    grid_ix = grid_ids % grid_width
    grid_iy = grid_ids // grid_width
    centers = np.column_stack(
        (
            grid_origin[0] + (grid_ix + 0.5) * config.grid_size_m,
            grid_origin[1] + (grid_iy + 0.5) * config.grid_size_m,
        )
    )
    control_ids = np.flatnonzero(controls)
    control_xy = centers[control_ids]
    control_z = cell_ground_flat[control_ids]
    tree = cKDTree(control_xy)
    cell_distance_flat = np.full(total_cells, np.nan, dtype=np.float32)
    cell_distance_flat[controls] = 0.0
    noncontrol_ids = np.flatnonzero(~controls)
    requested_k = min(config.idw_neighbors, len(control_ids))
    distances, neighbors = tree.query(centers[noncontrol_ids], k=requested_k)
    distances = np.asarray(distances, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if requested_k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    for target, row_distance, row_neighbor in zip(
        noncontrol_ids, distances, neighbors, strict=True
    ):
        _check_cancelled(cancellation_token, "while calculating finite-distance IDW")
        valid = (
            np.isfinite(row_distance)
            & (row_distance >= 0.0)
            & (row_neighbor >= 0)
            & (row_neighbor < len(control_z))
        )
        if not np.any(valid):
            cell_flags_flat[target] |= int(GroundQualityFlag.EXTRAPOLATION_TOO_FAR)
            cell_flags_flat[target] |= int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
            continue
        distance = row_distance[valid]
        neighbor = row_neighbor[valid]
        nearest = float(np.min(distance))
        cell_distance_flat[target] = np.float32(nearest)
        if nearest > config.interpolation_max_distance_m:
            cell_flags_flat[target] |= int(GroundQualityFlag.EXTRAPOLATION_TOO_FAR)
            cell_flags_flat[target] |= int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
            continue
        zero = distance == 0.0
        if np.any(zero):
            estimate = float(control_z[neighbor[np.flatnonzero(zero)[0]]])
        else:
            weights = 1.0 / np.power(distance, config.idw_power)
            weight_sum = float(np.sum(weights))
            if not math.isfinite(weight_sum) or weight_sum <= 0.0:
                cell_flags_flat[target] |= int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
                continue
            estimate = float(np.sum(weights * control_z[neighbor]) / weight_sum)
        if not math.isfinite(estimate):
            cell_flags_flat[target] |= int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
            continue
        cell_ground_flat[target] = np.float32(estimate)
        cell_flags_flat[target] |= int(GroundQualityFlag.INTERPOLATED_GROUND)
    _mark_high_gradients(
        cell_ground_flat,
        cell_flags_flat,
        grid_width=grid_width,
        grid_height=grid_height,
        config=config,
    )
    _check_cancelled(cancellation_token, "after A2B grid diagnostics")

    point_flags = cell_flags_flat[point_cell_id].astype(np.uint16, copy=True)
    ground_z = cell_ground_flat[point_cell_id].astype(np.float32, copy=True)
    ground_distance = cell_distance_flat[point_cell_id].astype(np.float32, copy=True)
    valid_ground = np.isfinite(ground_z) & (
        point_flags & int(GroundQualityFlag.INVALID_GROUND_ESTIMATE) == 0
    )
    if not np.any(valid_ground):
        raise _ground_error(
            code="GROUND_ALL_POINTS_INVALID",
            message="A2B produced no valid ground estimates for input points",
            details={},
        )
    invalid_ground = ~valid_ground
    point_flags[invalid_ground] |= int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    raw_hag = np.full(len(xyz), np.nan, dtype=np.float32)
    height_agl = np.full(len(xyz), np.nan, dtype=np.float32)
    raw_hag[valid_ground] = (z[valid_ground] - ground_z[valid_ground]).astype(np.float32)
    height_agl[valid_ground] = np.maximum(raw_hag[valid_ground], 0.0)
    below = valid_ground & (raw_hag < -config.below_ground_tolerance_m)
    extreme = valid_ground & (raw_hag > config.max_reasonable_hag_m)
    point_flags[below] |= int(GroundQualityFlag.BELOW_GROUND_TOLERANCE)
    point_flags[extreme] |= int(GroundQualityFlag.EXTREME_HEIGHT_AGL)
    class_ground = input_data.classification == config.source_ground_code
    disagreement = (
        class_ground
        & valid_ground
        & (np.abs(raw_hag) > config.classification_disagreement_tolerance_m)
    )
    point_flags[disagreement] |= int(GroundQualityFlag.SOURCE_CLASSIFICATION_DISAGREES)

    cell_shape = (grid_height, grid_width)
    surface = GroundSurfaceModel(
        grid_origin_xy=np.asarray(grid_origin, dtype=np.float64),
        grid_size_m=config.grid_size_m,
        grid_shape=cell_shape,
        cell_point_count=counts_flat.reshape(cell_shape),
        cell_ground_z_m=cell_ground_flat.reshape(cell_shape),
        cell_control_mask=controls.reshape(cell_shape),
        cell_quality_flags=cell_flags_flat.reshape(cell_shape),
        control_cell_xy=np.asarray(control_xy, dtype=np.float32),
        control_ground_z_m=np.asarray(control_z, dtype=np.float32),
    )
    hag = HeightAboveGroundSet(
        source_id=input_data.source_id,
        manifest_source_id=input_data.manifest_source_id,
        original_index=input_data.original_index.copy(),
        sample_index=input_data.sample_index.copy(),
        ground_z_m=ground_z,
        raw_height_agl_m=raw_hag,
        height_agl_m=height_agl,
        ground_cell_id=point_cell_id.astype(np.int32),
        ground_distance_m=ground_distance,
        quality_flags=point_flags,
    )
    classification_validation = {
        "source_ground_code": config.source_ground_code,
        "classification_ground_point_count": int(np.count_nonzero(class_ground)),
        "classification_disagreement_point_count": int(np.count_nonzero(disagreement)),
        "applicable": bool(np.any(class_ground)),
        "reason": (
            "no_points_with_source_ground_code"
            if not np.any(class_ground)
            else "applied_as_auxiliary_check"
        ),
    }
    return _ComputedGround(
        surface=surface, hag=hag, classification_validation=classification_validation
    )


def _mark_high_gradients(
    ground_z: np.ndarray,
    flags: np.ndarray,
    *,
    grid_width: int,
    grid_height: int,
    config: GroundEstimationConfig,
) -> None:
    finite = np.isfinite(ground_z)
    for iy in range(grid_height):
        for ix in range(grid_width):
            cell_id = iy * grid_width + ix
            if not finite[cell_id]:
                continue
            for dx, dy in ((1, 0), (0, 1)):
                next_x, next_y = ix + dx, iy + dy
                if next_x >= grid_width or next_y >= grid_height:
                    continue
                neighbor_id = next_y * grid_width + next_x
                if not finite[neighbor_id]:
                    continue
                gradient = (
                    abs(float(ground_z[cell_id]) - float(ground_z[neighbor_id]))
                    / config.grid_size_m
                )
                if gradient > config.high_ground_gradient_threshold_m_per_m:
                    flags[cell_id] |= int(GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION)
                    flags[neighbor_id] |= int(GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION)


def _build_statistics(
    surface: GroundSurfaceModel,
    hag: HeightAboveGroundSet,
    *,
    config: GroundEstimationConfig,
    elapsed_seconds: float,
    rss_start: int | None,
    rss_peak: int | None,
    rss_end: int | None,
    classification_validation: Mapping[str, Any],
) -> GroundEstimationStatistics:
    cell_flags = surface.cell_quality_flags
    point_flags = hag.quality_flags
    valid = (point_flags & int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)) == 0
    return GroundEstimationStatistics(
        input_point_count=hag.point_count,
        output_point_count=hag.point_count,
        total_cell_count=int(surface.cell_point_count.size),
        nonempty_cell_count=int(np.count_nonzero(surface.cell_point_count > 0)),
        direct_control_cell_count=surface.control_cell_count,
        sparse_cell_count=int(
            np.count_nonzero(cell_flags & int(GroundQualityFlag.SPARSE_GROUND_CELL))
        ),
        empty_cell_count=int(
            np.count_nonzero(cell_flags & int(GroundQualityFlag.EMPTY_GROUND_CELL))
        ),
        interpolated_cell_count=int(
            np.count_nonzero(cell_flags & int(GroundQualityFlag.INTERPOLATED_GROUND))
        ),
        extrapolation_too_far_cell_count=int(
            np.count_nonzero(cell_flags & int(GroundQualityFlag.EXTRAPOLATION_TOO_FAR))
        ),
        valid_ground_point_count=int(np.count_nonzero(valid)),
        invalid_ground_point_count=int(np.count_nonzero(~valid)),
        high_gradient_cell_count=int(
            np.count_nonzero(cell_flags & int(GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION))
        ),
        high_gradient_point_count=int(
            np.count_nonzero(point_flags & int(GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION))
        ),
        raw_negative_hag_count=int(np.count_nonzero(hag.raw_height_agl_m[valid] < 0.0)),
        below_ground_tolerance_count=int(
            np.count_nonzero(point_flags & int(GroundQualityFlag.BELOW_GROUND_TOLERANCE))
        ),
        extreme_hag_count=int(
            np.count_nonzero(point_flags & int(GroundQualityFlag.EXTREME_HEIGHT_AGL))
        ),
        classification_ground_count=int(
            classification_validation["classification_ground_point_count"]
        ),
        classification_disagreement_count=int(
            classification_validation["classification_disagreement_point_count"]
        ),
        elapsed_seconds=elapsed_seconds,
        rss_start_bytes=rss_start,
        rss_peak_bytes=rss_peak,
        rss_end_bytes=rss_end,
        control_tree_coordinate_bytes=surface.control_cell_xy.nbytes
        if config.record_memory
        else None,
    )


def _write_outputs_atomic(
    npz_path: Path,
    json_path: Path,
    *,
    input_data: _A2BInput,
    surface: GroundSurfaceModel,
    hag: HeightAboveGroundSet,
    config: GroundEstimationConfig,
    statistics: GroundEstimationStatistics,
    classification_validation: Mapping[str, Any],
    overwrite: bool,
    cancellation_token: CancellationToken | None,
) -> GroundEstimationResult:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp = _temporary_path(npz_path)
    json_tmp = _temporary_path(json_path)
    published: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        arrays = {
            "source_id": np.asarray(hag.source_id),
            "manifest_source_id": np.asarray(hag.manifest_source_id),
            "origin_xyz": input_data.origin_xyz,
            "grid_origin_xy": surface.grid_origin_xy,
            "grid_size_m": np.asarray(surface.grid_size_m, dtype=np.float64),
            "grid_shape": np.asarray(surface.grid_shape, dtype=np.int32),
            "cell_point_count": surface.cell_point_count,
            "cell_ground_z_m": surface.cell_ground_z_m,
            "cell_control_mask": surface.cell_control_mask,
            "cell_quality_flags": surface.cell_quality_flags,
            "control_cell_xy": surface.control_cell_xy,
            "control_ground_z_m": surface.control_ground_z_m,
            "original_index": hag.original_index,
            "sample_index": hag.sample_index,
            "ground_z_m": hag.ground_z_m,
            "raw_height_agl_m": hag.raw_height_agl_m,
            "height_agl_m": hag.height_agl_m,
            "ground_cell_id": hag.ground_cell_id,
            "ground_distance_m": hag.ground_distance_m,
            "quality_flags": hag.quality_flags,
        }
        _check_cancelled(cancellation_token, "before A2B NPZ write")
        np.savez_compressed(npz_tmp, **arrays)
        output_sha256 = _sha256_file(npz_tmp)
        output_bytes = npz_tmp.stat().st_size
        surface_metadata = surface.array_metadata()
        point_metadata = hag.array_metadata()
        result = GroundEstimationResult(
            source_id=hag.source_id,
            manifest_source_id=hag.manifest_source_id,
            input_a1_npz_path=input_data.a1_npz_path,
            input_a1_json_path=input_data.a1_json_path,
            input_a1_npz_sha256=input_data.a1_npz_sha256,
            input_a1_json_sha256=input_data.a1_json_sha256,
            input_a2a_npz_path=input_data.a2a_npz_path,
            input_a2a_json_path=input_data.a2a_json_path,
            input_a2a_npz_sha256=input_data.a2a_npz_sha256,
            input_a2a_json_sha256=input_data.a2a_json_sha256,
            output_npz_path=npz_path,
            output_json_path=json_path,
            output_npz_sha256=output_sha256,
            output_npz_bytes=output_bytes,
            origin_xyz=tuple(float(value) for value in input_data.origin_xyz),
            config=config,
            statistics=statistics,
            grid={
                "grid_origin_xy": [float(value) for value in surface.grid_origin_xy],
                "grid_shape": list(surface.grid_shape),
                "grid_size_m": surface.grid_size_m,
                "total_cell_count": int(surface.cell_point_count.size),
            },
            arrays={"surface": surface_metadata, "points": point_metadata},
            quality_flag_counts=_quality_flag_counts(hag.quality_flags),
            distributions={
                "ground_z_m": _distribution(hag.ground_z_m),
                "raw_height_agl_m": _distribution(hag.raw_height_agl_m),
                "height_agl_m": _distribution(hag.height_agl_m),
            },
            formulas=FORMULAS,
            classification_validation=classification_validation,
            algorithm_version=ALGORITHM_VERSION,
            created_at=datetime.now(UTC).isoformat(),
        )
        json_tmp.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _check_cancelled(cancellation_token, "before A2B output publication")
        backups = _publish_pair(
            ((npz_tmp, npz_path), (json_tmp, json_path)), overwrite=overwrite, published=published
        )
        for backup in backups.values():
            if backup.exists():
                backup.unlink()
        return result
    except Exception:
        _restore_after_publish_failure(published, backups)
        raise
    finally:
        for path in (npz_tmp, json_tmp):
            if path.exists():
                path.unlink()


def _validate_output_arrays(surface: GroundSurfaceModel, hag: HeightAboveGroundSet) -> None:
    invalid = (hag.quality_flags & int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)) != 0
    for value in (hag.ground_z_m, hag.raw_height_agl_m, hag.height_agl_m, hag.ground_distance_m):
        if not np.isfinite(value[~invalid]).all():
            raise _ground_error(
                code="GROUND_OUTPUT_NONFINITE",
                message="A2B output contains unflagged NaN or Inf",
                details={},
            )
    if np.any(hag.height_agl_m[~invalid] < 0.0):
        raise _ground_error(
            code="GROUND_OUTPUT_HAG_NEGATIVE",
            message="A2B clamped height_agl_m cannot be negative",
            details={},
        )
    if not np.isfinite(surface.control_ground_z_m).all():
        raise _ground_error(
            code="GROUND_OUTPUT_CONTROL_NONFINITE",
            message="A2B direct ground control heights must be finite",
            details={},
        )


def _quality_flag_counts(flags: np.ndarray) -> dict[str, int]:
    result = {"NONE": int(np.count_nonzero(flags == int(GroundQualityFlag.NONE)))}
    for flag in GroundQualityFlag:
        if flag is GroundQualityFlag.NONE:
            continue
        result[flag.name] = int(np.count_nonzero(flags & int(flag)))
    return result


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    percentiles = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _publish_pair(
    pairs: tuple[tuple[Path, Path], ...], *, overwrite: bool, published: list[Path]
) -> dict[Path, Path]:
    backups: dict[Path, Path] = {}
    if overwrite:
        for _temporary, final in pairs:
            if final.exists():
                backup = final.with_name(f"{final.stem}.backup.{uuid.uuid4().hex}{final.suffix}")
                os.replace(final, backup)
                backups[final] = backup
    else:
        _validate_outputs_absent(tuple(final for _temporary, final in pairs), overwrite=False)
    for temporary, final in pairs:
        os.replace(temporary, final)
        published.append(final)
    return backups


def _restore_after_publish_failure(published: list[Path], backups: Mapping[Path, Path]) -> None:
    for final in published:
        if final.exists():
            final.unlink()
    for final, backup in backups.items():
        if backup.exists():
            os.replace(backup, final)


def _validate_output_dir(output_dir: Path, *, project_root: Path) -> None:
    allowed = (project_root / "outputs" / "features").resolve(strict=False)
    if not _is_relative_to(output_dir.resolve(strict=False), allowed):
        raise _ground_error(
            code="GROUND_OUTPUT_FORBIDDEN",
            message="A2B output must be under outputs/features",
            details={"output_dir": str(output_dir)},
        )


def _validate_outputs_absent(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _ground_error(
            code="GROUND_OUTPUT_EXISTS",
            message="A2B output already exists and overwrite is disabled",
            details={"paths": existing},
        )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _report(
    reporter: ProgressReporter | None,
    source_id: str,
    stage: str,
    processed_points: int,
    total_points: int,
    *,
    status: str = "running",
) -> None:
    if reporter is None:
        return
    reporter.update(
        PipelineProgress(
            task_id=f"{source_id}:ground-hag",
            stage=stage,
            status=status,  # type: ignore[arg-type]
            processed_points=min(processed_points, total_points),
            total_points=total_points,
            percent=100.0
            if total_points == 0
            else min(processed_points / total_points * 100.0, 100.0),
            message=f"{stage}: {processed_points} of {total_points} points",
        )
    )


def _check_cancelled(token: CancellationToken | None, context: str) -> None:
    if token is not None and token.is_cancelled():
        raise _ground_error(
            code="GROUND_CANCELLED",
            message=f"A2B ground estimation cancelled {context}",
            details={"context": context},
        )


def _scalar_text(value: np.ndarray, name: str) -> str:
    if value.shape != ():
        raise _ground_error(
            code="GROUND_INPUT_SCALAR_INVALID",
            message="A2B scalar metadata has an invalid shape",
            details={"array": name, "shape": list(value.shape)},
        )
    result = str(value.item())
    if not result:
        raise _ground_error(
            code="GROUND_INPUT_SCALAR_INVALID",
            message="A2B scalar metadata must be non-empty",
            details={"array": name},
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ground_error(
    *, code: str, message: str, details: Mapping[str, Any], cause: str | None = None
) -> GroundEstimationError:
    return GroundEstimationError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="algorithms.features.ground",
            recoverable=True,
            details=details,
            cause=cause,
        )
    )


def _current_process() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process()


def _rss_bytes(process: Any | None) -> int | None:
    return None if process is None else int(process.memory_info().rss)


def _max_rss(current: int | None, process: Any | None) -> int | None:
    value = _rss_bytes(process)
    if value is None:
        return current
    return value if current is None else max(current, value)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
