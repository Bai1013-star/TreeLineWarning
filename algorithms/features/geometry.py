"""A2A controlled-memory local PCA geometry feature extraction."""

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
    GeometryFeatureConfig,
    GeometryFeatureResult,
    GeometryFeatureStatistics,
    PipelineProgress,
    PointFeatureSet,
    ProgressReporter,
    UnifiedError,
)
from scipy.spatial import cKDTree

OUTPUT_BASENAME = "scene_simple_geometry_features"
ALGORITHM_VERSION = "a2a-local-pca-v1"

FORMULAS: dict[str, str] = {
    "covariance": "C=(1/n)*sum((p_i-mean(p))@(p_i-mean(p)).T)",
    "eigenvalue_order": "lambda_1 >= lambda_2 >= lambda_3 >= 0",
    "linearity": "(lambda_1-lambda_2)/max(lambda_1, feature_epsilon_m2)",
    "planarity": "(lambda_2-lambda_3)/max(lambda_1, feature_epsilon_m2)",
    "scattering": "lambda_3/max(lambda_1, feature_epsilon_m2)",
    "verticality": "1-abs(normal_z), normal is the lambda_3 eigenvector",
    "local_density": "neighbor_count/((4/3)*pi*r_k^3), units points/m^3",
    "height_m": "xyz_local[:, 2], relative to the A1 local origin and not HAG",
}


class GeometryFeatureError(Exception):
    """Exception carrying a JSON-safe geometry feature error payload."""

    def __init__(self, error: UnifiedError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True)
class _A1Input:
    npz_path: Path
    json_path: Path
    source_id: str
    manifest_source_id: str
    xyz_local: np.ndarray
    origin_xyz: np.ndarray
    original_index: np.ndarray
    sample_index: np.ndarray
    npz_sha256: str
    json_sha256: str


def run_geometry_feature_extraction(
    a1_npz_path: Path | str,
    a1_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: GeometryFeatureConfig | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
) -> GeometryFeatureResult:
    """Create aligned A2A local PCA features from an A1 NPZ/JSON artifact pair."""
    cfg = config or GeometryFeatureConfig()
    input_npz = Path(a1_npz_path).resolve()
    input_json = Path(a1_json_path).resolve()
    output = Path(output_dir).resolve()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    _validate_output_dir(output, project_root=root)
    output_npz = output / f"{OUTPUT_BASENAME}.npz"
    output_json = output / f"{OUTPUT_BASENAME}.json"
    _validate_outputs_absent((output_npz, output_json), overwrite=overwrite)
    _check_cancelled(cancellation_token, "before A2A input validation")

    started = time.perf_counter()
    process = _current_process() if cfg.record_memory else None
    rss_start = _rss_bytes(process)
    rss_peak = rss_start
    a1 = _load_a1_input(input_npz, input_json, config=cfg)
    _check_cancelled(cancellation_token, "before KDTree construction")

    features, computation = _compute_features(
        a1,
        config=cfg,
        reporter=reporter,
        cancellation_token=cancellation_token,
        process=process,
        rss_peak=rss_peak,
    )
    rss_peak = computation["rss_peak"]
    _validate_output_arrays(features)
    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes(process)
    rss_peak = _max_rss(rss_peak, process)
    statistics = GeometryFeatureStatistics(
        input_point_count=a1.xyz_local.shape[0],
        output_point_count=features.point_count,
        pca_computed_count=computation["pca_computed_count"],
        insufficient_neighbor_count=computation["insufficient_neighbor_count"],
        degenerate_covariance_count=computation["degenerate_covariance_count"],
        duplicate_neighbor_coordinate_count=computation["duplicate_neighbor_coordinate_count"],
        negative_eigenvalue_clipped_count=computation["negative_eigenvalue_clipped_count"],
        normal_direction_ambiguous_count=computation["normal_direction_ambiguous_count"],
        principal_direction_ambiguous_count=computation["principal_direction_ambiguous_count"],
        density_undefined_count=computation["density_undefined_count"],
        numeric_invalid_count=computation["numeric_invalid_count"],
        requested_k=min(cfg.k_neighbors + 1, features.point_count),
        query_batch_size=cfg.query_batch_size,
        elapsed_seconds=elapsed,
        rss_start_bytes=rss_start,
        rss_peak_bytes=rss_peak,
        rss_end_bytes=rss_end,
        kd_tree_coordinate_bytes=int(a1.xyz_local.shape[0] * 3 * np.dtype(np.float64).itemsize),
    )
    quality_counts = _quality_flag_counts(features.quality_flags)
    result = _write_outputs_atomic(
        output_npz,
        output_json,
        features,
        input_a1=a1,
        config=cfg,
        statistics=statistics,
        quality_flag_counts=quality_counts,
        overwrite=overwrite,
        cancellation_token=cancellation_token,
    )
    _report(
        reporter,
        features.source_id,
        "geometry-features",
        features.point_count,
        features.point_count,
        status="completed",
    )
    return result


def _load_a1_input(
    npz_path: Path,
    json_path: Path,
    *,
    config: GeometryFeatureConfig,
) -> _A1Input:
    if not npz_path.is_file() or not json_path.is_file():
        raise _feature_error(
            code="FEATURE_INPUT_MISSING",
            message="A1 NPZ and JSON inputs must both exist",
            details={"a1_npz_path": str(npz_path), "a1_json_path": str(json_path)},
        )
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _feature_error(
            code="FEATURE_INPUT_JSON_INVALID",
            message="A1 JSON metadata cannot be parsed",
            details={"a1_json_path": str(json_path)},
            cause=str(exc),
        ) from exc
    _validate_a1_metadata(metadata, npz_path=npz_path, json_path=json_path, config=config)
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
    except Exception as exc:
        raise _feature_error(
            code="FEATURE_INPUT_NPZ_INVALID",
            message="A1 NPZ artifact cannot be loaded without pickle support",
            details={"a1_npz_path": str(npz_path)},
            cause=str(exc),
        ) from exc
    _validate_a1_arrays(arrays, metadata, config=config)
    return _A1Input(
        npz_path=npz_path,
        json_path=json_path,
        source_id=_scalar_text(arrays["source_id"], "source_id"),
        manifest_source_id=_scalar_text(arrays["manifest_source_id"], "manifest_source_id"),
        xyz_local=np.asarray(arrays["xyz_local"], dtype=np.float32),
        origin_xyz=np.asarray(arrays["origin_xyz"], dtype=np.float64),
        original_index=np.asarray(arrays["original_index"], dtype=np.int64),
        sample_index=np.asarray(arrays["sample_index"], dtype=np.int64),
        npz_sha256=_sha256_file(npz_path),
        json_sha256=_sha256_file(json_path),
    )


def _validate_a1_metadata(
    metadata: Any,
    *,
    npz_path: Path,
    json_path: Path,
    config: GeometryFeatureConfig,
) -> None:
    if not isinstance(metadata, Mapping):
        raise _feature_error(
            code="FEATURE_INPUT_METADATA_INVALID",
            message="A1 JSON root must be an object",
            details={"a1_json_path": str(json_path)},
        )
    required = {
        "source_id",
        "manifest_source_id",
        "output_npz_path",
        "output_json_path",
        "origin_xyz",
        "arrays",
        "statistics",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise _feature_error(
            code="FEATURE_INPUT_METADATA_MISSING_FIELDS",
            message="A1 JSON is missing required metadata fields",
            details={"missing_fields": missing},
        )
    if Path(str(metadata["output_npz_path"])).resolve() != npz_path:
        raise _feature_error(
            code="FEATURE_INPUT_NPZ_PATH_MISMATCH",
            message="A1 JSON output_npz_path does not match the provided NPZ input",
            details={
                "declared_path": str(metadata["output_npz_path"]),
                "input_path": str(npz_path),
            },
        )
    if Path(str(metadata["output_json_path"])).resolve() != json_path:
        raise _feature_error(
            code="FEATURE_INPUT_JSON_PATH_MISMATCH",
            message="A1 JSON output_json_path does not match the provided JSON input",
            details={
                "declared_path": str(metadata["output_json_path"]),
                "input_path": str(json_path),
            },
        )
    if str(metadata["source_id"]) != config.source_id:
        raise _feature_error(
            code="FEATURE_INPUT_SOURCE_ID_MISMATCH",
            message="A1 source_id does not match GeometryFeatureConfig.source_id",
            details={
                "a1_source_id": str(metadata["source_id"]),
                "config_source_id": config.source_id,
            },
        )
    if not str(metadata["manifest_source_id"]):
        raise _feature_error(
            code="FEATURE_INPUT_MANIFEST_SOURCE_ID_INVALID",
            message="A1 manifest_source_id must be non-empty",
            details={},
        )


def _validate_a1_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    config: GeometryFeatureConfig,
) -> None:
    required = {
        "xyz_local",
        "origin_xyz",
        "original_index",
        "sample_index",
        "source_id",
        "manifest_source_id",
        "attribute_names",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise _feature_error(
            code="FEATURE_INPUT_ARRAY_MISSING",
            message="A1 NPZ is missing required arrays",
            details={"missing_arrays": missing},
        )
    xyz = arrays["xyz_local"]
    origin = arrays["origin_xyz"]
    original_index = arrays["original_index"]
    sample_index = arrays["sample_index"]
    if xyz.shape == (0, 3):
        raise _feature_error(
            code="FEATURE_EMPTY_INPUT",
            message="A1 NPZ contains zero points",
            details={},
        )
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype != np.float32:
        raise _feature_error(
            code="FEATURE_INPUT_XYZ_INVALID",
            message="A1 xyz_local must be finite float32 with shape (M, 3)",
            details={"shape": list(xyz.shape), "dtype": str(xyz.dtype)},
        )
    if origin.shape != (3,) or origin.dtype != np.float64 or not np.isfinite(origin).all():
        raise _feature_error(
            code="FEATURE_INPUT_ORIGIN_INVALID",
            message="A1 origin_xyz must be finite float64 with shape (3,)",
            details={"shape": list(origin.shape), "dtype": str(origin.dtype)},
        )
    if not np.isfinite(xyz).all():
        raise _feature_error(
            code="FEATURE_INPUT_XYZ_NONFINITE",
            message="A1 xyz_local contains NaN or Inf",
            details={},
        )
    point_count = xyz.shape[0]
    for name, value in (("original_index", original_index), ("sample_index", sample_index)):
        if value.shape != (point_count,) or value.dtype != np.int64:
            raise _feature_error(
                code="FEATURE_INPUT_INDEX_INVALID",
                message="A1 traceability indexes must be int64 with shape (M,)",
                details={"array": name, "shape": list(value.shape), "dtype": str(value.dtype)},
            )
        if np.any(value < 0) or np.unique(value).size != point_count:
            raise _feature_error(
                code="FEATURE_INPUT_INDEX_TRACEABILITY_INVALID",
                message="A1 traceability indexes must be non-negative and unique",
                details={"array": name, "point_count": point_count},
            )
    if _scalar_text(arrays["source_id"], "source_id") != str(metadata["source_id"]):
        raise _feature_error(
            code="FEATURE_INPUT_SOURCE_ID_MISMATCH",
            message="A1 NPZ and JSON source_id values differ",
            details={},
        )
    if _scalar_text(arrays["source_id"], "source_id") != config.source_id:
        raise _feature_error(
            code="FEATURE_INPUT_SOURCE_ID_MISMATCH",
            message="A1 NPZ source_id does not match GeometryFeatureConfig.source_id",
            details={"config_source_id": config.source_id},
        )
    if _scalar_text(arrays["manifest_source_id"], "manifest_source_id") != str(
        metadata["manifest_source_id"]
    ):
        raise _feature_error(
            code="FEATURE_INPUT_MANIFEST_SOURCE_ID_MISMATCH",
            message="A1 NPZ and JSON manifest_source_id values differ",
            details={},
        )
    if not np.array_equal(origin, np.asarray(metadata["origin_xyz"], dtype=np.float64)):
        raise _feature_error(
            code="FEATURE_INPUT_ORIGIN_MISMATCH",
            message="A1 NPZ and JSON origin_xyz values differ",
            details={},
        )
    statistics = metadata["statistics"]
    if (
        not isinstance(statistics, Mapping)
        or statistics.get("downsampled_point_count") != point_count
    ):
        raise _feature_error(
            code="FEATURE_INPUT_POINT_COUNT_MISMATCH",
            message="A1 JSON downsampled point count does not match NPZ",
            details={"npz_point_count": point_count},
        )
    _validate_a1_array_metadata(arrays, metadata["arrays"])


def _validate_a1_array_metadata(arrays: Mapping[str, np.ndarray], metadata: Any) -> None:
    if not isinstance(metadata, Mapping):
        raise _feature_error(
            code="FEATURE_INPUT_ARRAY_METADATA_INVALID",
            message="A1 JSON arrays metadata must be an object",
            details={},
        )
    attribute_names = tuple(str(value) for value in arrays["attribute_names"].tolist())
    required = {
        "xyz_local",
        "origin_xyz",
        "original_index",
        "sample_index",
        "voxel_key",
        "voxel_point_count",
        "attributes",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise _feature_error(
            code="FEATURE_INPUT_ARRAY_METADATA_INVALID",
            message="A1 JSON arrays metadata is incomplete",
            details={"missing_fields": missing},
        )
    for name in (
        "xyz_local",
        "origin_xyz",
        "original_index",
        "sample_index",
        "voxel_key",
        "voxel_point_count",
    ):
        _validate_shape_dtype_metadata(name, arrays[name], metadata[name])
    attribute_metadata = metadata["attributes"]
    if not isinstance(attribute_metadata, Mapping) or set(attribute_metadata) != set(
        attribute_names
    ):
        raise _feature_error(
            code="FEATURE_INPUT_ATTRIBUTE_METADATA_MISMATCH",
            message="A1 attribute metadata does not match attribute_names",
            details={},
        )
    for name in attribute_names:
        if name not in arrays:
            raise _feature_error(
                code="FEATURE_INPUT_ATTRIBUTE_MISSING",
                message="A1 attribute_names references an absent NPZ array",
                details={"attribute": name},
            )
        _validate_shape_dtype_metadata(name, arrays[name], attribute_metadata[name])


def _validate_shape_dtype_metadata(name: str, array: np.ndarray, metadata: Any) -> None:
    if not isinstance(metadata, Mapping):
        raise _feature_error(
            code="FEATURE_INPUT_ARRAY_METADATA_INVALID",
            message="A1 array metadata entry must be an object",
            details={"array": name},
        )
    if metadata.get("shape") != list(array.shape) or metadata.get("dtype") != str(array.dtype):
        raise _feature_error(
            code="FEATURE_INPUT_ARRAY_METADATA_MISMATCH",
            message="A1 array shape or dtype does not match JSON metadata",
            details={"array": name, "npz_shape": list(array.shape), "npz_dtype": str(array.dtype)},
        )


def _compute_features(
    a1: _A1Input,
    *,
    config: GeometryFeatureConfig,
    reporter: ProgressReporter | None,
    cancellation_token: CancellationToken | None,
    process: Any | None,
    rss_peak: int | None,
) -> tuple[PointFeatureSet, dict[str, Any]]:
    point_count = a1.xyz_local.shape[0]
    tree_xyz = np.asarray(a1.xyz_local, dtype=np.float64)
    tree = cKDTree(tree_xyz, copy_data=False)
    linearity = np.zeros(point_count, dtype=np.float32)
    planarity = np.zeros(point_count, dtype=np.float32)
    scattering = np.zeros(point_count, dtype=np.float32)
    verticality = np.zeros(point_count, dtype=np.float32)
    local_density = np.zeros(point_count, dtype=np.float32)
    height_m = a1.xyz_local[:, 2].astype(np.float32, copy=True)
    neighbor_count = np.zeros(point_count, dtype=np.uint16)
    eigenvalues = np.zeros((point_count, 3), dtype=np.float32)
    principal_direction = np.zeros((point_count, 3), dtype=np.float32)
    normal = np.zeros((point_count, 3), dtype=np.float32)
    quality_flags = np.zeros(point_count, dtype=np.uint16)
    requested_k = min(config.k_neighbors + 1, point_count)

    for start in range(0, point_count, config.query_batch_size):
        end = min(start + config.query_batch_size, point_count)
        distances, indexes = tree.query(
            tree_xyz[start:end],
            k=requested_k,
            workers=config.workers,
        )
        batch_count = end - start
        if requested_k == 1:
            distances = np.asarray(distances).reshape(batch_count, 1)
            indexes = np.asarray(indexes).reshape(batch_count, 1)
        else:
            distances = np.asarray(distances)
            indexes = np.asarray(indexes)
        selected_indexes, selected_distances, counts = _select_neighbors(
            distances,
            indexes,
            start=start,
            point_count=point_count,
            k_neighbors=config.k_neighbors,
        )
        neighbor_count[start:end] = counts
        _populate_batch_features(
            tree_xyz,
            selected_indexes,
            selected_distances,
            counts,
            start=start,
            config=config,
            linearity=linearity,
            planarity=planarity,
            scattering=scattering,
            verticality=verticality,
            local_density=local_density,
            eigenvalues=eigenvalues,
            principal_direction=principal_direction,
            normal=normal,
            quality_flags=quality_flags,
        )
        del distances, indexes, selected_indexes, selected_distances, counts
        rss_peak = _max_rss(rss_peak, process)
        _report(reporter, a1.source_id, "geometry-knn-pca", end, point_count)
        _check_cancelled(cancellation_token, "during KNN PCA feature extraction")

    flags = quality_flags.astype(np.uint16, copy=False)
    result = PointFeatureSet(
        source_id=a1.source_id,
        manifest_source_id=a1.manifest_source_id,
        xyz_local=a1.xyz_local,
        origin_xyz=a1.origin_xyz,
        original_index=a1.original_index,
        sample_index=a1.sample_index,
        linearity=linearity,
        planarity=planarity,
        scattering=scattering,
        verticality=verticality,
        local_density=local_density,
        height_m=height_m,
        neighbor_count=neighbor_count,
        eigenvalues=eigenvalues,
        principal_direction=principal_direction,
        normal=normal,
        quality_flags=flags,
    )
    return result, {
        "rss_peak": rss_peak,
        "pca_computed_count": int(
            np.count_nonzero((flags & int(FeatureQualityFlag.INSUFFICIENT_NEIGHBORS)) == 0)
        ),
        "insufficient_neighbor_count": _flag_count(
            flags, FeatureQualityFlag.INSUFFICIENT_NEIGHBORS
        ),
        "degenerate_covariance_count": _flag_count(flags, FeatureQualityFlag.DEGENERATE_COVARIANCE),
        "duplicate_neighbor_coordinate_count": _flag_count(
            flags, FeatureQualityFlag.DUPLICATE_NEIGHBOR_COORDINATES
        ),
        "negative_eigenvalue_clipped_count": _flag_count(
            flags, FeatureQualityFlag.NEGATIVE_EIGENVALUE_CLIPPED
        ),
        "normal_direction_ambiguous_count": _flag_count(
            flags, FeatureQualityFlag.NORMAL_DIRECTION_AMBIGUOUS
        ),
        "principal_direction_ambiguous_count": _flag_count(
            flags, FeatureQualityFlag.PRINCIPAL_DIRECTION_AMBIGUOUS
        ),
        "density_undefined_count": _flag_count(flags, FeatureQualityFlag.DENSITY_UNDEFINED),
        "numeric_invalid_count": _flag_count(flags, FeatureQualityFlag.NUMERIC_INVALID),
    }


def _select_neighbors(
    distances: np.ndarray,
    indexes: np.ndarray,
    *,
    start: int,
    point_count: int,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_count = distances.shape[0]
    selected_indexes = np.full((batch_count, k_neighbors), -1, dtype=np.int64)
    selected_distances = np.full((batch_count, k_neighbors), np.inf, dtype=np.float64)
    counts = np.zeros(batch_count, dtype=np.uint16)
    for row in range(batch_count):
        query_index = start + row
        row_indexes = np.asarray(indexes[row], dtype=np.int64)
        row_distances = np.asarray(distances[row], dtype=np.float64)
        valid = (
            (row_indexes >= 0)
            & (row_indexes < point_count)
            & (row_indexes != query_index)
            & np.isfinite(row_distances)
        )
        row_indexes = row_indexes[valid]
        row_distances = row_distances[valid]
        if row_indexes.size:
            order = np.lexsort((row_indexes, row_distances))
            kept = min(k_neighbors, row_indexes.size)
            row_indexes = row_indexes[order[:kept]]
            row_distances = row_distances[order[:kept]]
            selected_indexes[row, :kept] = row_indexes
            selected_distances[row, :kept] = row_distances
            counts[row] = kept
    return selected_indexes, selected_distances, counts


def _populate_batch_features(
    tree_xyz: np.ndarray,
    selected_indexes: np.ndarray,
    selected_distances: np.ndarray,
    counts: np.ndarray,
    *,
    start: int,
    config: GeometryFeatureConfig,
    linearity: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
    verticality: np.ndarray,
    local_density: np.ndarray,
    eigenvalues: np.ndarray,
    principal_direction: np.ndarray,
    normal: np.ndarray,
    quality_flags: np.ndarray,
) -> None:
    batch_indices = np.arange(start, start + len(counts))
    insufficient = counts < config.min_neighbors
    if np.any(insufficient):
        target = batch_indices[insufficient]
        quality_flags[target] |= int(FeatureQualityFlag.INSUFFICIENT_NEIGHBORS)
        quality_flags[target] |= int(FeatureQualityFlag.DENSITY_UNDEFINED)

    usable = ~insufficient
    if not np.any(usable):
        return
    usable_rows = np.flatnonzero(usable)
    usable_counts = counts[usable_rows].astype(np.intp)
    max_count = int(usable_counts.max())
    # This is a bounded batch buffer, never an M x K retention structure.
    compact_indexes = selected_indexes[usable_rows, :max_count]
    compact_distances = selected_distances[usable_rows, :max_count]
    position_mask = np.arange(max_count)[None, :] < usable_counts[:, None]
    neighborhood = tree_xyz[compact_indexes]
    means = np.sum(neighborhood * position_mask[:, :, None], axis=1) / usable_counts[:, None]
    centered = (neighborhood - means[:, None, :]) * position_mask[:, :, None]
    covariance = np.einsum("bni,bnj->bij", centered, centered) / usable_counts[:, None, None]
    if not np.isfinite(covariance).all():
        raise _feature_error(
            code="FEATURE_COVARIANCE_NONFINITE",
            message="Local covariance contains NaN or Inf",
            details={},
        )
    # NumPy's batched eigh path has been unstable on the supported Windows build.
    # Keep the bounded covariance batch but dispatch independent 3x3 decompositions.
    values_ascending = np.empty((len(covariance), 3), dtype=np.float64)
    vectors_ascending = np.empty((len(covariance), 3, 3), dtype=np.float64)
    for row, matrix in enumerate(covariance):
        values_ascending[row], vectors_ascending[row] = np.linalg.eigh(matrix)
    values = values_ascending[:, ::-1]
    vectors = vectors_ascending[:, :, ::-1]
    values, clipped = _sanitize_eigenvalues(
        values, config=config, batch_start=start, rows=usable_rows
    )
    targets = batch_indices[usable_rows]
    if np.any(clipped):
        quality_flags[targets[clipped]] |= int(FeatureQualityFlag.NEGATIVE_EIGENVALUE_CLIPPED)
    eigenvalues[targets] = values.astype(np.float32)

    lambda_1 = values[:, 0]
    lambda_2 = values[:, 1]
    lambda_3 = values[:, 2]
    denominator = np.maximum(lambda_1, config.feature_epsilon_m2)
    degenerate = lambda_1 <= config.feature_epsilon_m2
    if np.any(degenerate):
        degenerate_targets = targets[degenerate]
        quality_flags[degenerate_targets] |= int(FeatureQualityFlag.DEGENERATE_COVARIANCE)
        quality_flags[degenerate_targets] |= int(FeatureQualityFlag.NORMAL_DIRECTION_AMBIGUOUS)
        quality_flags[degenerate_targets] |= int(FeatureQualityFlag.PRINCIPAL_DIRECTION_AMBIGUOUS)
    valid_geometry = ~degenerate
    if np.any(valid_geometry):
        valid_targets = targets[valid_geometry]
        linearity[valid_targets] = ((lambda_1 - lambda_2) / denominator)[valid_geometry]
        planarity[valid_targets] = ((lambda_2 - lambda_3) / denominator)[valid_geometry]
        scattering[valid_targets] = (lambda_3 / denominator)[valid_geometry]
        principal_gap = (lambda_1 - lambda_2) / denominator
        normal_gap = (lambda_2 - lambda_3) / denominator
        principal_ambiguous = valid_geometry & (
            principal_gap <= config.direction_gap_relative_tolerance
        )
        normal_ambiguous = valid_geometry & (normal_gap <= config.direction_gap_relative_tolerance)
        if np.any(principal_ambiguous):
            quality_flags[targets[principal_ambiguous]] |= int(
                FeatureQualityFlag.PRINCIPAL_DIRECTION_AMBIGUOUS
            )
        if np.any(normal_ambiguous):
            quality_flags[targets[normal_ambiguous]] |= int(
                FeatureQualityFlag.NORMAL_DIRECTION_AMBIGUOUS
            )
        principal_valid = valid_geometry & ~principal_ambiguous
        normal_valid = valid_geometry & ~normal_ambiguous
        if np.any(principal_valid):
            principal_direction[targets[principal_valid]] = _canonicalize_directions(
                vectors[principal_valid, :, 0]
            ).astype(np.float32)
        if np.any(normal_valid):
            canonical_normals = _canonicalize_directions(vectors[normal_valid, :, 2])
            normal[targets[normal_valid]] = canonical_normals.astype(np.float32)
            verticality[targets[normal_valid]] = (1.0 - np.abs(canonical_normals[:, 2])).astype(
                np.float32
            )

    farthest = compact_distances[np.arange(len(usable_rows)), usable_counts - 1]
    volume = (4.0 / 3.0) * math.pi * np.power(farthest, 3)
    density_valid = (
        np.isfinite(farthest)
        & (farthest > config.density_epsilon_m)
        & np.isfinite(volume)
        & (volume > 0.0)
    )
    if np.any(density_valid):
        local_density[targets[density_valid]] = (
            usable_counts[density_valid] / volume[density_valid]
        ).astype(np.float32)
    if np.any(~density_valid):
        quality_flags[targets[~density_valid]] |= int(FeatureQualityFlag.DENSITY_UNDEFINED)
    duplicate = np.any(
        (compact_distances <= config.density_epsilon_m) & position_mask,
        axis=1,
    )
    if np.any(duplicate):
        quality_flags[targets[duplicate]] |= int(FeatureQualityFlag.DUPLICATE_NEIGHBOR_COORDINATES)


def _sanitize_eigenvalues(
    values: np.ndarray,
    *,
    config: GeometryFeatureConfig,
    batch_start: int,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lambda_1 = values[:, 0]
    tolerance = np.maximum(
        config.eigenvalue_absolute_tolerance_m2,
        config.eigenvalue_relative_tolerance * np.maximum(lambda_1, 0.0),
    )
    invalid = values < -tolerance[:, None]
    if np.any(invalid):
        first = np.argwhere(invalid)[0]
        row, column = int(first[0]), int(first[1])
        raise _feature_error(
            code="FEATURE_EIGENVALUE_NEGATIVE",
            message="Covariance eigendecomposition produced a materially negative eigenvalue",
            details={
                "point_index": batch_start + int(rows[row]),
                "eigenvalue_column": column,
                "eigenvalue": float(values[row, column]),
                "tolerance": float(tolerance[row]),
            },
        )
    clipped = np.any(values < 0.0, axis=1)
    return np.maximum(values, 0.0), clipped


def _canonicalize_directions(directions: np.ndarray) -> np.ndarray:
    result = np.asarray(directions, dtype=np.float64).copy()
    largest = np.argmax(np.abs(result), axis=1)
    signs = np.sign(result[np.arange(len(result)), largest])
    signs[signs == 0.0] = 1.0
    return result * signs[:, None]


def _write_outputs_atomic(
    npz_path: Path,
    json_path: Path,
    features: PointFeatureSet,
    *,
    input_a1: _A1Input,
    config: GeometryFeatureConfig,
    statistics: GeometryFeatureStatistics,
    quality_flag_counts: Mapping[str, int],
    overwrite: bool,
    cancellation_token: CancellationToken | None,
) -> GeometryFeatureResult:
    npz_tmp = _temporary_path(npz_path)
    json_tmp = _temporary_path(json_path)
    published: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        _check_cancelled(cancellation_token, "before A2A output write")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (npz_tmp, json_tmp):
            if path.exists():
                path.unlink()
        arrays = {
            "source_id": np.asarray(features.source_id),
            "manifest_source_id": np.asarray(features.manifest_source_id),
            "xyz_local": features.xyz_local,
            "origin_xyz": features.origin_xyz,
            "original_index": features.original_index,
            "sample_index": features.sample_index,
            "linearity": features.linearity,
            "planarity": features.planarity,
            "scattering": features.scattering,
            "verticality": features.verticality,
            "local_density": features.local_density,
            "height_m": features.height_m,
            "neighbor_count": features.neighbor_count,
            "eigenvalues": features.eigenvalues,
            "principal_direction": features.principal_direction,
            "normal": features.normal,
            "quality_flags": features.quality_flags,
        }
        np.savez_compressed(npz_tmp, **arrays)
        output_sha256 = _sha256_file(npz_tmp)
        output_bytes = npz_tmp.stat().st_size
        result = GeometryFeatureResult(
            source_id=features.source_id,
            manifest_source_id=features.manifest_source_id,
            input_a1_npz_path=input_a1.npz_path,
            input_a1_json_path=input_a1.json_path,
            input_a1_npz_sha256=input_a1.npz_sha256,
            input_a1_json_sha256=input_a1.json_sha256,
            output_npz_path=npz_path,
            output_json_path=json_path,
            output_npz_sha256=output_sha256,
            output_npz_bytes=output_bytes,
            origin_xyz=tuple(float(value) for value in features.origin_xyz),
            config=config,
            statistics=statistics,
            arrays=features.array_metadata(),
            quality_flag_counts=quality_flag_counts,
            formulas=FORMULAS,
            algorithm_version=ALGORITHM_VERSION,
            created_at=datetime.now(UTC).isoformat(),
        )
        json_tmp.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _check_cancelled(cancellation_token, "before A2A output publication")
        backups = _publish_pair(
            ((npz_tmp, npz_path), (json_tmp, json_path)),
            overwrite=overwrite,
            published=published,
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


def _publish_pair(
    pairs: tuple[tuple[Path, Path], ...],
    *,
    overwrite: bool,
    published: list[Path],
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


def _validate_output_arrays(features: PointFeatureSet) -> None:
    values = (
        features.xyz_local,
        features.origin_xyz,
        features.linearity,
        features.planarity,
        features.scattering,
        features.verticality,
        features.local_density,
        features.height_m,
        features.eigenvalues,
        features.principal_direction,
        features.normal,
    )
    if not all(np.isfinite(value).all() for value in values):
        raise _feature_error(
            code="FEATURE_OUTPUT_NONFINITE",
            message="A2A output arrays contain NaN or Inf",
            details={},
        )
    for value in (
        features.linearity,
        features.planarity,
        features.scattering,
        features.verticality,
    ):
        if np.any(value < 0.0) or np.any(value > 1.0 + 1e-6):
            raise _feature_error(
                code="FEATURE_OUTPUT_RANGE_INVALID",
                message="A2A normalized geometry features are outside [0, 1]",
                details={},
            )
    if np.any(features.local_density < 0.0):
        raise _feature_error(
            code="FEATURE_OUTPUT_DENSITY_INVALID",
            message="A2A local_density contains a negative value",
            details={},
        )


def _quality_flag_counts(flags: np.ndarray) -> dict[str, int]:
    result = {"NONE": int(np.count_nonzero(flags == int(FeatureQualityFlag.NONE)))}
    for flag in FeatureQualityFlag:
        if flag is FeatureQualityFlag.NONE:
            continue
        result[flag.name] = _flag_count(flags, flag)
    return result


def _flag_count(flags: np.ndarray, flag: FeatureQualityFlag) -> int:
    return int(np.count_nonzero((flags & int(flag)) != 0))


def _scalar_text(value: np.ndarray, name: str) -> str:
    if value.shape != ():
        raise _feature_error(
            code="FEATURE_INPUT_SCALAR_INVALID",
            message="A1 scalar metadata array has an invalid shape",
            details={"array": name, "shape": list(value.shape)},
        )
    result = str(value.item())
    if not result:
        raise _feature_error(
            code="FEATURE_INPUT_SCALAR_INVALID",
            message="A1 scalar metadata array must be non-empty",
            details={"array": name},
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_output_dir(output_dir: Path, *, project_root: Path) -> None:
    allowed = (project_root / "outputs" / "features").resolve(strict=False)
    if not _is_relative_to(output_dir.resolve(strict=False), allowed):
        raise _feature_error(
            code="FEATURE_OUTPUT_FORBIDDEN",
            message="Geometry feature output must be under outputs/features",
            details={"output_dir": str(output_dir)},
        )


def _validate_outputs_absent(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _feature_error(
            code="FEATURE_OUTPUT_EXISTS",
            message="Geometry feature output already exists and overwrite is disabled",
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
            task_id=f"{source_id}:geometry-features",
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
        raise _feature_error(
            code="FEATURE_CANCELLED",
            message=f"Geometry feature extraction cancelled {context}",
            details={"context": context},
        )


def _feature_error(
    *,
    code: str,
    message: str,
    details: Mapping[str, Any],
    cause: str | None = None,
) -> GeometryFeatureError:
    return GeometryFeatureError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="algorithms.features.geometry",
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
