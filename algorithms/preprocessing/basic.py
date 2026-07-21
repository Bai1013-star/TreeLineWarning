"""A1 basic preprocessing and algorithm point cloud generation."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import (
    CancellationToken,
    CropBounds,
    DownsampledPointCloud,
    PipelineProgress,
    PreprocessingConfig,
    PreprocessingResult,
    ProcessingStatistics,
    ProgressReporter,
)

from algorithms.io import iter_las_chunks, read_las_header_summary, validate_scene_sample_artifacts
from algorithms.io.headers import LasIoError, _error

ALGORITHM_NAME = "basic_preprocessing_voxel_downsample"
ALGORITHM_VERSION = "0.1"
OUTPUT_BASENAME = "scene_simple_preprocessed"
PRESERVED_ATTRIBUTE_NAMES = (
    "red",
    "green",
    "blue",
    "intensity",
    "classification",
    "NormalX",
    "NormalY",
    "NormalZ",
)


@dataclass
class _VoxelAccumulator:
    count: int
    sum_xyz: np.ndarray


@dataclass
class _VoxelCandidate:
    distance_sq: float
    sample_index: int
    original_index: int
    xyz_local: np.ndarray
    attributes: dict[str, Any]


@dataclass
class _QualityCounts:
    processed: int = 0
    valid: int = 0
    nan_inf_removed: int = 0
    abnormal_removed: int = 0


def run_basic_preprocessing(
    sample_path: Path | str,
    index_map_path: Path | str,
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    config: PreprocessingConfig | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
) -> PreprocessingResult:
    """Generate a local-coordinate, voxel-downsampled algorithm point cloud.

    The implementation reads LAS points only through ``iter_las_chunks()``. It
    performs two streaming voxel passes so the selected representative remains a
    real source point with both sample-row and raw-LAS index traceability.
    """
    cfg = config or PreprocessingConfig()
    sample = Path(sample_path).resolve()
    index_map = Path(index_map_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    _validate_output_dir(output, project_root=root)
    npz_path = output / f"{OUTPUT_BASENAME}.npz"
    json_path = output / f"{OUTPUT_BASENAME}.json"
    _validate_outputs_absent((npz_path, json_path), overwrite=overwrite)
    _check_cancelled(cancellation_token, "before preprocessing")

    started = time.perf_counter()
    process = _current_process() if cfg.record_memory else None
    rss_start = _rss_bytes(process)
    rss_peak = rss_start

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    bounds = CropBounds.from_dict(manifest["crop_bounds"])
    validation = validate_scene_sample_artifacts(
        sample_path=sample,
        index_map_path=index_map,
        manifest_path=manifest_file,
        bounds=bounds,
        chunk_size=cfg.chunk_size,
    )
    header = read_las_header_summary(sample, source_id=cfg.source_id)
    if header.point_count == 0:
        raise _preprocessing_error(
            code="PREPROCESSING_EMPTY_INPUT",
            message="Preprocessing input LAS contains zero points",
            details={"sample_path": str(sample)},
        )

    raw_original_index = _load_raw_original_index(index_map, expected_length=header.point_count)
    origin = _resolve_origin(sample, header.min_xyz, config=cfg)
    sor_allowed_sample_index, sor_removed, sor_skipped_reason = _prepare_sor_filter(
        sample,
        raw_original_index,
        origin,
        config=cfg,
        cancellation_token=cancellation_token,
    )

    voxels: dict[tuple[int, int, int], _VoxelAccumulator] = {}
    counts = _QualityCounts()
    for chunk_xyz, sample_index, _raw_index, _attributes in _iter_prepared_chunks(
        sample,
        raw_original_index,
        origin,
        config=cfg,
        cancellation_token=cancellation_token,
        sor_allowed_sample_index=sor_allowed_sample_index,
    ):
        counts.processed += len(sample_index)
        valid_mask, nan_count, abnormal_count = _quality_mask(chunk_xyz, config=cfg)
        counts.nan_inf_removed += nan_count
        counts.abnormal_removed += abnormal_count
        if not bool(np.any(valid_mask)):
            continue
        valid_xyz = chunk_xyz[valid_mask]
        counts.valid += len(valid_xyz)
        keys = _voxel_keys(valid_xyz, cfg.voxel_size_m)
        unique_keys, inverse, voxel_counts = np.unique(
            keys,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(sums, inverse, valid_xyz.astype(np.float64, copy=False))
        for key_array, point_count, xyz_sum in zip(unique_keys, voxel_counts, sums, strict=True):
            key = _key_tuple(key_array)
            existing = voxels.get(key)
            if existing is None:
                voxels[key] = _VoxelAccumulator(
                    count=int(point_count),
                    sum_xyz=xyz_sum.copy(),
                )
            else:
                existing.count += int(point_count)
                existing.sum_xyz += xyz_sum
        rss_peak = _max_rss(rss_peak, process)
        _report(
            reporter,
            cfg.source_id,
            "preprocess-voxel-accumulate",
            counts.processed,
            _effective_total(header.point_count, cfg.max_input_points),
        )
        _check_cancelled(cancellation_token, "during voxel accumulation")

    if counts.valid == 0:
        raise _preprocessing_error(
            code="PREPROCESSING_ALL_POINTS_FILTERED",
            message="All input points were removed by preprocessing quality filters",
            details={"sample_path": str(sample), "config": cfg.to_dict()},
        )
    if not voxels:
        raise _preprocessing_error(
            code="PREPROCESSING_NO_VOXELS",
            message="Voxel downsampling produced zero voxels",
            details={"sample_path": str(sample), "voxel_size_m": cfg.voxel_size_m},
        )
    if cfg.max_output_points is not None and len(voxels) > cfg.max_output_points:
        raise _preprocessing_error(
            code="PREPROCESSING_OUTPUT_POINT_LIMIT_EXCEEDED",
            message="Voxel output exceeds max_output_points; increase voxel_size_m",
            details={
                "voxel_count": len(voxels),
                "max_output_points": cfg.max_output_points,
                "voxel_size_m": cfg.voxel_size_m,
            },
        )

    centroids = {
        key: accumulator.sum_xyz / accumulator.count for key, accumulator in voxels.items()
    }
    candidates: dict[tuple[int, int, int], _VoxelCandidate] = {}
    second_pass_count = 0
    for chunk_xyz, sample_index, raw_index, attributes in _iter_prepared_chunks(
        sample,
        raw_original_index,
        origin,
        config=cfg,
        cancellation_token=cancellation_token,
        sor_allowed_sample_index=sor_allowed_sample_index,
    ):
        second_pass_count += len(sample_index)
        valid_mask, _nan_count, _abnormal_count = _quality_mask(chunk_xyz, config=cfg)
        if not bool(np.any(valid_mask)):
            continue
        valid_xyz = chunk_xyz[valid_mask]
        valid_sample_index = sample_index[valid_mask]
        valid_raw_index = raw_index[valid_mask]
        valid_attributes = {
            name: np.asarray(value)[valid_mask] for name, value in attributes.items()
        }
        keys = _voxel_keys(valid_xyz, cfg.voxel_size_m)
        for row, key_array in enumerate(keys):
            key = _key_tuple(key_array)
            if key not in centroids:
                continue
            if cfg.voxel_representative == "first":
                distance_sq = 0.0
            else:
                delta = valid_xyz[row].astype(np.float64, copy=False) - centroids[key]
                distance_sq = float(np.dot(delta, delta))
            sample_idx = int(valid_sample_index[row])
            existing = candidates.get(key)
            if existing is not None and not _candidate_is_better(
                distance_sq,
                sample_idx,
                existing,
                representative=cfg.voxel_representative,
            ):
                continue
            candidates[key] = _VoxelCandidate(
                distance_sq=distance_sq,
                sample_index=sample_idx,
                original_index=int(valid_raw_index[row]),
                xyz_local=valid_xyz[row].copy(),
                attributes={
                    name: np.asarray(values[row]).copy()
                    for name, values in valid_attributes.items()
                },
            )
        rss_peak = _max_rss(rss_peak, process)
        _report(
            reporter,
            cfg.source_id,
            "preprocess-voxel-select",
            second_pass_count,
            _effective_total(header.point_count, cfg.max_input_points),
        )
        _check_cancelled(cancellation_token, "during voxel representative selection")

    cloud = _build_cloud(cfg.source_id, candidates, voxels, dtype=cfg.coordinate_dtype)
    _validate_traceability(
        cloud,
        sample_point_count=header.point_count,
        raw_source_point_count=int(manifest["source_point_count"]),
    )
    elapsed = time.perf_counter() - started
    rss_end = _rss_bytes(process)
    rss_peak = _max_rss(rss_peak, process)
    statistics = ProcessingStatistics(
        input_point_count=header.point_count,
        processed_input_point_count=counts.processed,
        valid_point_count=counts.valid,
        nan_inf_removed_count=counts.nan_inf_removed,
        abnormal_coord_removed_count=counts.abnormal_removed,
        outlier_removed_count=sor_removed,
        downsampled_point_count=cloud.point_count,
        downsample_ratio=cloud.point_count / counts.valid,
        chunk_size=cfg.chunk_size,
        voxel_size_m=cfg.voxel_size_m,
        elapsed_seconds=elapsed,
        rss_start_bytes=rss_start,
        rss_peak_bytes=rss_peak,
        rss_end_bytes=rss_end,
        estimated_input_array_bytes=(
            header.point_count * 3 * np.dtype(cfg.coordinate_dtype).itemsize
        ),
        estimated_output_array_bytes=(
            cloud.point_count * 3 * np.dtype(cfg.coordinate_dtype).itemsize
        ),
        sor_skipped_reason=sor_skipped_reason,
        warnings=tuple(_warnings(cfg, validation, sor_skipped_reason)),
    )
    result = PreprocessingResult(
        source_id=cfg.source_id,
        input_path=sample,
        index_map_path=index_map,
        manifest_path=manifest_file,
        output_npz_path=npz_path,
        output_json_path=json_path,
        config=cfg,
        statistics=statistics,
        manifest_source_id=str(manifest["source_id"]),
        origin_xyz=tuple(float(value) for value in origin),
        arrays=cloud.array_metadata(),
        warnings=statistics.warnings,
        created_at=datetime.now(UTC).isoformat(),
    )
    _write_outputs_atomic(npz_path, json_path, cloud, result, overwrite=overwrite)
    _report(
        reporter,
        cfg.source_id,
        "preprocess",
        statistics.processed_input_point_count,
        _effective_total(header.point_count, cfg.max_input_points),
        status="completed",
    )
    return result


def _iter_prepared_chunks(
    sample_path: Path,
    raw_original_index: np.ndarray,
    origin: np.ndarray,
    *,
    config: PreprocessingConfig,
    cancellation_token: CancellationToken | None,
    sor_allowed_sample_index: set[int] | None,
) -> Any:
    total_yielded = 0
    dtype = np.dtype(config.coordinate_dtype)
    for chunk in iter_las_chunks(
        sample_path,
        source_id=config.source_id,
        chunk_size=config.chunk_size,
        cancellation_token=cancellation_token,
    ):
        _check_cancelled(cancellation_token, "during chunk read")
        sample_index = np.asarray(chunk.original_index, dtype=np.int64)
        if config.max_input_points is not None:
            remaining = config.max_input_points - total_yielded
            if remaining <= 0:
                break
            if len(sample_index) > remaining:
                sample_index = sample_index[:remaining]
                xyz_source = np.asarray(chunk.xyz)[:remaining]
                attributes = {
                    name: np.asarray(value)[:remaining]
                    for name, value in chunk.attributes.items()
                    if name in PRESERVED_ATTRIBUTE_NAMES
                }
            else:
                xyz_source = np.asarray(chunk.xyz)
                attributes = {
                    name: np.asarray(value)
                    for name, value in chunk.attributes.items()
                    if name in PRESERVED_ATTRIBUTE_NAMES
                }
        else:
            xyz_source = np.asarray(chunk.xyz)
            attributes = {
                name: np.asarray(value)
                for name, value in chunk.attributes.items()
                if name in PRESERVED_ATTRIBUTE_NAMES
            }
        total_yielded += len(sample_index)
        if sor_allowed_sample_index is not None:
            sor_mask = np.fromiter(
                (int(index) in sor_allowed_sample_index for index in sample_index),
                dtype=bool,
                count=len(sample_index),
            )
            sample_index = sample_index[sor_mask]
            xyz_source = xyz_source[sor_mask]
            attributes = {name: value[sor_mask] for name, value in attributes.items()}
        if len(sample_index) == 0:
            continue
        raw_index = raw_original_index[sample_index]
        xyz_local = (xyz_source.astype(np.float64, copy=False) - origin).astype(dtype, copy=False)
        yield xyz_local, sample_index, raw_index, attributes


def _resolve_origin(
    sample_path: Path,
    header_min_xyz: tuple[float, float, float],
    *,
    config: PreprocessingConfig,
) -> np.ndarray:
    if config.origin_mode == "header_min":
        return np.asarray(header_min_xyz, dtype=np.float64)
    if config.origin_mode == "explicit":
        return np.asarray(config.origin_xyz, dtype=np.float64)
    for chunk in iter_las_chunks(
        sample_path,
        source_id=config.source_id,
        chunk_size=config.chunk_size,
    ):
        xyz = np.asarray(chunk.xyz, dtype=np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        if bool(np.any(finite)):
            return xyz[finite][0].astype(np.float64)
    raise _preprocessing_error(
        code="PREPROCESSING_ORIGIN_UNAVAILABLE",
        message="Could not determine first valid point origin",
        details={"sample_path": str(sample_path)},
    )


def _quality_mask(
    xyz_local: np.ndarray,
    *,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, int, int]:
    finite = np.isfinite(xyz_local).all(axis=1)
    nan_inf_removed = int(len(xyz_local) - np.count_nonzero(finite))
    if not config.reject_nan_inf and nan_inf_removed:
        raise _preprocessing_error(
            code="PREPROCESSING_NONFINITE_COORDINATES",
            message="Input contains NaN or Inf coordinates and reject_nan_inf is disabled",
            details={"nonfinite_count": nan_inf_removed},
        )
    mask = finite
    abnormal_removed = 0
    if config.max_abs_local_coord_m is not None:
        abnormal = np.any(np.abs(xyz_local) > config.max_abs_local_coord_m, axis=1)
        abnormal_removed = int(np.count_nonzero(mask & abnormal))
        mask = mask & ~abnormal
    return mask, nan_inf_removed, abnormal_removed


def _voxel_keys(xyz_local: np.ndarray, voxel_size_m: float) -> np.ndarray:
    return np.floor(xyz_local.astype(np.float64, copy=False) / voxel_size_m).astype(np.int64)


def _key_tuple(value: np.ndarray) -> tuple[int, int, int]:
    return (int(value[0]), int(value[1]), int(value[2]))


def _candidate_is_better(
    distance_sq: float,
    sample_index: int,
    existing: _VoxelCandidate,
    *,
    representative: str,
) -> bool:
    if representative == "first":
        return sample_index < existing.sample_index
    if distance_sq < existing.distance_sq:
        return True
    return math.isclose(distance_sq, existing.distance_sq) and sample_index < existing.sample_index


def _build_cloud(
    source_id: str,
    candidates: dict[tuple[int, int, int], _VoxelCandidate],
    voxels: dict[tuple[int, int, int], _VoxelAccumulator],
    *,
    dtype: str,
) -> DownsampledPointCloud:
    ordered_keys = sorted(candidates)
    xyz_local = np.asarray(
        [candidates[key].xyz_local for key in ordered_keys],
        dtype=np.dtype(dtype),
    ).reshape((-1, 3))
    origin_xyz = np.zeros(3, dtype=np.float64)
    original_index = np.asarray(
        [candidates[key].original_index for key in ordered_keys],
        dtype=np.int64,
    )
    sample_index = np.asarray(
        [candidates[key].sample_index for key in ordered_keys],
        dtype=np.int64,
    )
    voxel_key = np.asarray(ordered_keys, dtype=np.int64).reshape((-1, 3))
    voxel_point_count = np.asarray([voxels[key].count for key in ordered_keys], dtype=np.int32)
    attributes: dict[str, np.ndarray] = {}
    names = sorted({name for key in ordered_keys for name in candidates[key].attributes})
    for name in names:
        attributes[name] = np.asarray([candidates[key].attributes[name] for key in ordered_keys])
    if {"red", "green", "blue"}.issubset(attributes):
        attributes["rgb"] = np.column_stack(
            (attributes["red"], attributes["green"], attributes["blue"])
        ).astype(np.uint16, copy=False)
    return DownsampledPointCloud(
        source_id=source_id,
        xyz_local=xyz_local,
        origin_xyz=origin_xyz,
        original_index=original_index,
        sample_index=sample_index,
        voxel_key=voxel_key,
        voxel_point_count=voxel_point_count,
        attributes=attributes,
    )


def _write_outputs_atomic(
    npz_path: Path,
    json_path: Path,
    cloud: DownsampledPointCloud,
    result: PreprocessingResult,
    *,
    overwrite: bool,
) -> None:
    npz_tmp = _temporary_path(npz_path)
    json_tmp = _temporary_path(json_path)
    try:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (npz_tmp, json_tmp):
            if path.exists():
                path.unlink()
        arrays = {
            "xyz_local": cloud.xyz_local,
            "origin_xyz": np.asarray(result.origin_xyz, dtype=np.float64),
            "original_index": cloud.original_index,
            "sample_index": cloud.sample_index,
            "voxel_key": cloud.voxel_key,
            "voxel_point_count": cloud.voxel_point_count,
            "source_id": np.asarray(result.source_id),
            "manifest_source_id": np.asarray(result.manifest_source_id),
            "attribute_names": np.asarray(sorted(cloud.attributes)),
        }
        arrays.update(cloud.attributes)
        np.savez_compressed(npz_tmp, **arrays)
        json_tmp.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _replace(npz_tmp, npz_path, overwrite=overwrite)
        _replace(json_tmp, json_path, overwrite=overwrite)
    except Exception:
        for path in (npz_tmp, json_tmp):
            if path.exists():
                path.unlink()
        raise


def _prepare_sor_filter(
    sample_path: Path,
    raw_original_index: np.ndarray,
    origin: np.ndarray,
    *,
    config: PreprocessingConfig,
    cancellation_token: CancellationToken | None,
) -> tuple[set[int] | None, int, str | None]:
    if not config.enable_statistical_outlier_filter:
        return None, 0, "disabled_by_default_to_preserve_thin_wires"
    effective_total = _effective_total(len(raw_original_index), config.max_input_points)
    if effective_total > config.sor_max_points:
        raise _preprocessing_error(
            code="PREPROCESSING_SOR_POINT_LIMIT_EXCEEDED",
            message="SOR is enabled but input exceeds sor_max_points",
            details={"point_count": effective_total, "sor_max_points": config.sor_max_points},
        )
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]
    except Exception as exc:
        raise _preprocessing_error(
            code="PREPROCESSING_SOR_DEPENDENCY_UNAVAILABLE",
            message=(
                "SOR requires scipy in the current environment; dependencies were not installed"
            ),
            details={"dependency": "scipy"},
            cause=str(exc),
        ) from exc

    xyz_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    for xyz_local, sample_index, _raw_index, _attributes in _iter_prepared_chunks(
        sample_path,
        raw_original_index,
        origin,
        config=PreprocessingConfig(
            **{
                **config.to_dict(),
                "enable_statistical_outlier_filter": False,
            }
        ),
        cancellation_token=cancellation_token,
        sor_allowed_sample_index=None,
    ):
        mask, _nan_count, _abnormal_count = _quality_mask(xyz_local, config=config)
        if bool(np.any(mask)):
            xyz_parts.append(xyz_local[mask].astype(np.float64, copy=False))
            sample_parts.append(sample_index[mask])
    if not xyz_parts:
        return set(), 0, None
    xyz = np.concatenate(xyz_parts)
    sample_index = np.concatenate(sample_parts)
    k = min(config.sor_k + 1, len(xyz))
    if k <= 1:
        return set(int(value) for value in sample_index), 0, None
    tree = cKDTree(xyz)
    distances, _indexes = tree.query(xyz, k=k)
    mean_distance = np.mean(distances[:, 1:], axis=1)
    threshold = float(mean_distance.mean() + config.sor_std_ratio * mean_distance.std())
    keep = mean_distance <= threshold
    removed = int(np.count_nonzero(~keep))
    return set(int(value) for value in sample_index[keep]), removed, None


def _load_raw_original_index(index_map_path: Path, *, expected_length: int) -> np.ndarray:
    with np.load(index_map_path, allow_pickle=False) as data:
        original_index = np.asarray(data["original_index"], dtype=np.int64)
    if len(original_index) != expected_length:
        raise _preprocessing_error(
            code="PREPROCESSING_INDEX_MAP_LENGTH_MISMATCH",
            message="index_map original_index length does not match sample LAS point count",
            details={"index_length": len(original_index), "sample_point_count": expected_length},
        )
    return original_index


def _validate_traceability(
    cloud: DownsampledPointCloud,
    *,
    sample_point_count: int,
    raw_source_point_count: int,
) -> None:
    original_index = np.asarray(cloud.original_index, dtype=np.int64)
    sample_index = np.asarray(cloud.sample_index, dtype=np.int64)
    if len(np.unique(original_index)) != len(original_index):
        raise _preprocessing_error(
            code="PREPROCESSING_ORIGINAL_INDEX_DUPLICATE",
            message="Downsampled original_index contains duplicates",
            details={"point_count": len(original_index)},
        )
    if len(np.unique(sample_index)) != len(sample_index):
        raise _preprocessing_error(
            code="PREPROCESSING_SAMPLE_INDEX_DUPLICATE",
            message="Downsampled sample_index contains duplicates",
            details={"point_count": len(sample_index)},
        )
    original_out_of_range = len(original_index) and (
        int(original_index.min()) < 0 or int(original_index.max()) >= raw_source_point_count
    )
    if original_out_of_range:
        raise _preprocessing_error(
            code="PREPROCESSING_ORIGINAL_INDEX_OUT_OF_RANGE",
            message="Downsampled original_index is outside raw LAS point range",
            details={"raw_source_point_count": raw_source_point_count},
        )
    sample_out_of_range = len(sample_index) and (
        int(sample_index.min()) < 0 or int(sample_index.max()) >= sample_point_count
    )
    if sample_out_of_range:
        raise _preprocessing_error(
            code="PREPROCESSING_SAMPLE_INDEX_OUT_OF_RANGE",
            message="Downsampled sample_index is outside sample LAS point range",
            details={"sample_point_count": sample_point_count},
        )
    if not np.isfinite(np.asarray(cloud.xyz_local)).all():
        raise _preprocessing_error(
            code="PREPROCESSING_NONFINITE_OUTPUT",
            message="Downsampled xyz_local contains non-finite coordinates",
            details={"point_count": cloud.point_count},
        )


def _validate_output_dir(output_dir: Path, *, project_root: Path) -> None:
    output_resolved = output_dir.resolve(strict=False)
    allowed = (project_root / "outputs" / "pointcloud").resolve(strict=False)
    if not _is_relative_to(output_resolved, allowed):
        raise _preprocessing_error(
            code="PREPROCESSING_OUTPUT_FORBIDDEN",
            message="Preprocessing output must be under outputs/pointcloud",
            details={"output_dir": str(output_dir)},
        )


def _validate_outputs_absent(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _preprocessing_error(
            code="PREPROCESSING_OUTPUT_EXISTS",
            message="Preprocessing output already exists and overwrite is disabled",
            details={"paths": existing},
        )


def _replace(temp_path: Path, final_path: Path, *, overwrite: bool) -> None:
    if final_path.exists() and not overwrite:
        raise _preprocessing_error(
            code="PREPROCESSING_OUTPUT_EXISTS",
            message="Preprocessing output already exists and overwrite is disabled",
            details={"path": str(final_path)},
        )
    if final_path.exists():
        final_path.unlink()
    os.replace(temp_path, final_path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _effective_total(point_count: int, max_input_points: int | None) -> int:
    if max_input_points is None:
        return point_count
    return min(point_count, max_input_points)


def _warnings(
    config: PreprocessingConfig,
    validation: dict[str, Any],
    sor_skipped_reason: str | None,
) -> list[str]:
    warnings = []
    if config.max_input_points is not None:
        warnings.append(f"Processed only first {config.max_input_points} input points")
    if sor_skipped_reason is not None:
        warnings.append(f"SOR skipped: {sor_skipped_reason}")
    manifest_warnings = validation.get("manifest", {}).get("warnings", [])
    warnings.extend(str(value) for value in manifest_warnings)
    return warnings


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
    percent = 100.0 if total_points == 0 else processed_points / total_points * 100.0
    reporter.update(
        PipelineProgress(
            task_id=f"{source_id}:basic-preprocessing",
            stage=stage,
            status=status,  # type: ignore[arg-type]
            processed_points=min(processed_points, total_points),
            total_points=total_points,
            percent=min(percent, 100.0),
            message=f"{stage}: {processed_points} of {total_points} points",
        )
    )


def _check_cancelled(token: CancellationToken | None, context: str) -> None:
    if token is not None and token.is_cancelled():
        raise _preprocessing_error(
            code="PREPROCESSING_CANCELLED",
            message=f"Preprocessing cancelled {context}",
            details={"context": context},
        )


def _preprocessing_error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    cause: str | None = None,
) -> LasIoError:
    return _error(
        code=code,
        message=message,
        details=details,
        recoverable=True,
        cause=cause,
    )


def _current_process() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process()


def _rss_bytes(process: Any | None) -> int | None:
    if process is None:
        return None
    return int(process.memory_info().rss)


def _max_rss(current: int | None, process: Any | None) -> int | None:
    value = _rss_bytes(process)
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
