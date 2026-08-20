"""W1 confirmed-wire centreline reconstruction from A4 artifacts only."""

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
    PipelineProgress,
    ProgressReporter,
    UnifiedError,
    WireModelMethod,
    WireModelQualityFlag,
    WireModelSet,
    WireModelStatus,
    WireModelSummary,
    WireReconstructionConfig,
    WireReconstructionResult,
    WireReconstructionStatistics,
    WireSeparationResult,
    WireTrackStatus,
)
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "w1-confirmed-wire-centreline-v3"
SCHEMA_VERSION = "wire-models-v2"


class WireReconstructionError(Exception):
    def __init__(self, error: UnifiedError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass
class _Segment:
    wire_id: int
    index: int
    point_indices: np.ndarray
    representative_xyz: np.ndarray
    representative_t: np.ndarray
    method: WireModelMethod
    flags: int
    ready: bool
    sampled_xyz: np.ndarray
    sampled_t: np.ndarray
    control_xyz: np.ndarray
    knots: np.ndarray
    degree: int
    support_residual: tuple[float, float, float, float]
    nearest_sample_residual: tuple[float, float, float, float]
    representative_residual: tuple[float, float, float, float]
    observation_residual: tuple[float, float, float, float]
    turn_angle: tuple[float, float]
    endpoint_uncertainty: float
    reason: str
    sample_step: tuple[float, float, float]
    representative_arc_length: float
    candidate_arc_length: float
    arc_length_inflation_ratio: float
    warnings: tuple[str, ...]
    smoothing_value: float | None
    spline_candidates: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _SplineCandidate:
    smoothing_value: float
    sampled_xyz: np.ndarray
    sampled_t: np.ndarray
    control_xyz: np.ndarray
    knots: np.ndarray
    degree: int
    support_residual: tuple[float, float, float, float]
    nearest_sample_residual: tuple[float, float, float, float]
    representative_residual: tuple[float, float, float, float]
    observation_residual: tuple[float, float, float, float]
    turn_angle: tuple[float, float]
    sample_step: tuple[float, float, float]
    representative_arc_length: float
    candidate_arc_length: float
    arc_length_inflation_ratio: float
    warnings: tuple[str, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]


def run_wire_reconstruction(
    a4_npz_path: Path | str,
    a4_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: WireReconstructionConfig | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
    generate_preview: bool = True,
) -> WireReconstructionResult:
    """Reconstruct only A4 confirmed wires; no LAS, A3, or global distance matrix is used."""

    config = config or WireReconstructionConfig()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output_dir = Path(output_dir).resolve()
    _validate_output_dir(output_dir, root / "outputs" / "wire_models")
    output_npz = output_dir / f"{config.source_id}_wire_models.npz"
    output_json = output_dir / f"{config.source_id}_wire_models.json"
    _validate_outputs((output_npz, output_json), overwrite)
    _check_cancelled(cancellation_token, "before input validation")
    started = time.perf_counter()
    process = _current_process() if config.record_memory else None
    rss_start = _rss(process)
    a4_npz, a4_json = Path(a4_npz_path).resolve(), Path(a4_json_path).resolve()
    data, a4_result, hashes = _load_a4(a4_npz, a4_json, config)
    confirmed = np.flatnonzero(data["wire_id_by_point"] >= 0)
    _report(reporter, config.source_id, "validated_a4", 0, len(confirmed))
    wire_ids = np.unique(data["wire_id_by_point"][confirmed]).astype(np.int32)
    wire_records: list[dict[str, Any]] = []
    all_segments: list[_Segment] = []
    gaps: list[dict[str, Any]] = []
    for position, wire_id_value in enumerate(wire_ids):
        _check_cancelled(cancellation_token, f"while reconstructing wire {int(wire_id_value)}")
        record, segments, wire_gaps = _reconstruct_wire(data, a4_result, int(wire_id_value), config)
        wire_records.append(record)
        all_segments.extend(segments)
        gaps.extend(wire_gaps)
        _report(reporter, config.source_id, "reconstructed_wires", position + 1, len(wire_ids))
    arrays, summaries, segment_summaries = _arrays_and_summaries(wire_records, all_segments, gaps)
    model_set = WireModelSet(
        arrays["wire_id"],
        arrays["source_track_id"],
        arrays["wire_status"],
        arrays["wire_method"],
        arrays["wire_quality_flags"],
        arrays["wire_has_distance_ready_segments"],
        arrays["wire_continuous_distance_ready"],
        arrays["wire_segment_offsets"],
        arrays["segment_wire_id"],
        arrays["segment_index"],
        arrays["segment_method"],
        arrays["segment_quality_flags"],
        arrays["segment_distance_ready"],
        arrays["segment_parameter_domain"],
        arrays["sampled_curve_xyz"],
        arrays["sampled_curve_parameter"],
        arrays["sampled_curve_offsets"],
    )
    del model_set  # Construction validates the flattened numerical layout before publication.
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = WireReconstructionStatistics(
        input_point_count=len(data["xyz_local"]),
        confirmed_wire_point_count=len(confirmed),
        wire_count=len(wire_ids),
        segment_count=len(all_segments),
        gap_count=len(gaps),
        distance_ready_segment_count=int(np.count_nonzero(arrays["segment_distance_ready"])),
        elapsed_seconds=time.perf_counter() - started,
        rss_start_bytes=rss_start,
        rss_peak_bytes=_rss(process),
        rss_end_bytes=_rss(process),
    )

    def relative(path: Path) -> str:
        return _relative(root, path)

    placeholder = "0" * 64
    result = WireReconstructionResult(
        SCHEMA_VERSION,
        config.source_id,
        a4_result.manifest_source_id,
        relative(a4_npz),
        relative(a4_json),
        hashes["npz"],
        hashes["json"],
        relative(output_npz),
        relative(output_json),
        placeholder,
        0,
        config,
        stats,
        _array_metadata(arrays),
        tuple(summaries),
        tuple(segment_summaries),
        tuple(gaps),
        (),
        ALGORITHM_VERSION,
        datetime.now(UTC).isoformat(),
    )
    result = _publish(arrays, result, output_npz, output_json, started, process)
    if generate_preview:
        preview = root / "outputs" / "previews" / f"{config.source_id}_wire_models_preview.html"
        generate_wire_models_preview(
            output_npz,
            output_json,
            preview,
            overwrite=overwrite,
            project_root=root,
        )
    _report(reporter, config.source_id, "published", len(confirmed), len(confirmed), "completed")
    return result


def load_wire_models(
    model_npz_path: Path | str, model_json_path: Path | str
) -> tuple[WireReconstructionResult, dict[str, np.ndarray]]:
    """Load a complete manifest-last W1 pair without any upstream artifact."""

    npz_path, json_path = Path(model_npz_path), Path(model_json_path)
    if not npz_path.is_file() or not json_path.is_file():
        raise _error("WIRE_MODEL_INPUT_MISSING", "W1 NPZ and JSON are both required")
    try:
        result = WireReconstructionResult.from_dict(json.loads(json_path.read_text("utf-8")))
        with np.load(npz_path, allow_pickle=False) as stream:
            arrays = {name: stream[name] for name in stream.files}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise _error(
            "WIRE_MODEL_INPUT_LOAD", "unable to load W1 result", cause=str(error)
        ) from error
    if _sha(npz_path) != result.output_npz_sha256:
        raise _error("WIRE_MODEL_INPUT_SHA", "W1 NPZ SHA-256 does not match JSON")
    if result.schema_version != SCHEMA_VERSION:
        raise _error("WIRE_MODEL_SCHEMA", "W1 JSON schema version is unsupported")
    _validate_model_arrays(arrays)
    return result, arrays


def generate_wire_models_preview(
    model_npz_path: Path | str,
    model_json_path: Path | str,
    output_html_path: Path | str,
    *,
    overwrite: bool = False,
    project_root: Path | str | None = None,
) -> Path:
    """Write a standalone W1 Plotly preview with one line trace per observed segment."""

    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output = Path(output_html_path).resolve()
    _validate_output_dir(output.parent, root / "outputs" / "previews")
    if output.exists() and not overwrite:
        raise _error(
            "WIRE_MODEL_PREVIEW_EXISTS", "wire-model preview exists and overwrite is disabled"
        )
    result, arrays = load_wire_models(model_npz_path, model_json_path)
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise _error(
            "WIRE_MODEL_PREVIEW_DEPENDENCY", "Plotly is unavailable", cause=str(error)
        ) from error
    figure = go.Figure()
    palette = [
        "#dc2626",
        "#2563eb",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
    ]
    input_offsets = arrays["wire_input_offsets"]
    segment_offsets = arrays["wire_segment_offsets"]
    sample_offsets = arrays["sampled_curve_offsets"]
    for wire_position, summary in enumerate(result.wires):
        wire_id = summary.wire_id
        color = palette[wire_id % len(palette)]
        begin, end = input_offsets[wire_position : wire_position + 2]
        observed = arrays["ordered_input_xyz"][begin:end]
        figure.add_trace(
            go.Scatter3d(
                x=observed[:, 0],
                y=observed[:, 1],
                z=observed[:, 2],
                mode="markers",
                name=f"wire {wire_id} observations",
                legendgroup=f"wire-{wire_id}",
                marker={"size": 2.1, "color": color, "opacity": 0.55},
                hovertemplate=f"wire={wire_id}<br>confirmed observation<extra></extra>",
            )
        )
        first_segment, last_segment = segment_offsets[wire_position : wire_position + 2]
        for segment_position in range(first_segment, last_segment):
            sample_start, sample_end = sample_offsets[segment_position : segment_position + 2]
            sample = arrays["sampled_curve_xyz"][sample_start:sample_end]
            method = WireModelMethod(int(arrays["segment_method"][segment_position])).name
            ready = bool(arrays["segment_distance_ready"][segment_position])
            segment_label = int(arrays["segment_index"][segment_position])
            figure.add_trace(
                go.Scatter3d(
                    x=sample[:, 0],
                    y=sample[:, 1],
                    z=sample[:, 2],
                    mode="lines+markers",
                    name=f"wire {wire_id} segment {segment_label} {method}",
                    legendgroup=f"wire-{wire_id}",
                    line={
                        "width": 6 if ready else 3,
                        "color": color,
                        "dash": "solid" if ready else "dash",
                    },
                    marker={"size": 1.2, "color": color},
                    hovertemplate=(
                        f"wire={wire_id}<br>segment=%{{text}}<br>method={method}"
                        f"<br>distance-ready={ready}<extra></extra>"
                    ),
                    text=[int(arrays["segment_index"][segment_position])] * len(sample),
                )
            )
    figure.update_layout(
        title="W1 Confirmed Wire Models (observed segments only)",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(output)
    try:
        figure.write_html(temporary, include_plotlyjs=True, full_html=True)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _load_a4(
    npz_path: Path, json_path: Path, config: WireReconstructionConfig
) -> tuple[dict[str, np.ndarray], WireSeparationResult, dict[str, str]]:
    if not npz_path.is_file() or not json_path.is_file():
        raise _error("WIRE_MODEL_A4_MISSING", "A4 NPZ and JSON inputs are required")
    try:
        raw_json = json.loads(json_path.read_text("utf-8"))
        result = WireSeparationResult.from_dict(raw_json)
        with np.load(npz_path, allow_pickle=False) as stream:
            data = {name: stream[name] for name in stream.files}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise _error("WIRE_MODEL_A4_LOAD", "unable to load A4 input", cause=str(error)) from error
    hashes = {"npz": _sha(npz_path), "json": _sha(json_path)}
    if result.output_npz_sha256 != hashes["npz"]:
        raise _error("WIRE_MODEL_A4_SHA", "A4 NPZ SHA-256 does not match JSON")
    if (
        result.source_id != config.source_id
        or str(data.get("source_id", np.asarray("")).item()) != result.source_id
    ):
        raise _error("WIRE_MODEL_A4_SOURCE", "A4 source identifiers are inconsistent")
    required = {
        "xyz_local": ((None, 3), np.float32),
        "original_index": ((None,), np.int64),
        "sample_index": ((None,), np.int64),
        "track_id_by_point": ((None,), np.int32),
        "wire_id_by_point": ((None,), np.int32),
        "assignment_score": ((None,), np.float32),
        "assignment_quality_flags": ((None,), np.uint16),
        "assignment_reason_flags": ((None,), np.uint16),
        "ordered_wire_point_indices": ((None,), np.int64),
        "wire_offsets": ((None,), np.int64),
        "track_status": ((None,), np.uint8),
        "track_wire_id": ((None,), np.int32),
        "track_actual_main_direction": ((None, 3), np.float32),
    }
    for name, (shape, dtype) in required.items():
        value = data.get(name)
        if (
            value is None
            or value.dtype != dtype
            or len(value.shape) != len(shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(value.shape, shape, strict=True)
            )
        ):
            raise _error("WIRE_MODEL_A4_SHAPE", "A4 array shape or dtype is invalid", array=name)
    count = len(data["xyz_local"])
    if any(
        len(data[name]) != count
        for name in (
            "original_index",
            "sample_index",
            "track_id_by_point",
            "wire_id_by_point",
            "assignment_score",
            "assignment_quality_flags",
            "assignment_reason_flags",
        )
    ):
        raise _error("WIRE_MODEL_A4_ALIGNMENT", "A4 full-length arrays are not aligned")
    if count != result.statistics.input_point_count or not np.isfinite(data["xyz_local"]).all():
        raise _error("WIRE_MODEL_A4_COORDINATES", "A4 coordinate count or values are invalid")
    if not np.isfinite(data["assignment_score"]).all() or not np.all(
        (data["assignment_score"] >= 0) & (data["assignment_score"] <= 1)
    ):
        raise _error("WIRE_MODEL_A4_SCORE", "A4 assignment scores are invalid")
    for name in ("original_index", "sample_index"):
        if np.any(data[name] < 0) or len(np.unique(data[name])) != count:
            raise _error(
                "WIRE_MODEL_A4_INDEX", "A4 indices must be unique and non-negative", array=name
            )
    confirmed = np.flatnonzero(data["wire_id_by_point"] >= 0)
    if np.any(data["track_id_by_point"][confirmed] < 0):
        raise _error("WIRE_MODEL_A4_MAPPING", "confirmed A4 points require a track")
    ordered = data["ordered_wire_point_indices"]
    offsets = data["wire_offsets"]
    if (
        offsets.ndim != 1
        or len(offsets) != result.statistics.confirmed_wire_count + 1
        or offsets[0] != 0
        or offsets[-1] != len(ordered)
        or np.any(np.diff(offsets) < 0)
    ):
        raise _error("WIRE_MODEL_A4_OFFSETS", "A4 wire offsets are invalid")
    if len(np.unique(ordered)) != len(ordered) or not np.array_equal(np.sort(ordered), confirmed):
        raise _error(
            "WIRE_MODEL_A4_CONFIRMED_SET", "A4 ordered confirmed set does not match wire IDs"
        )
    wire_ids = np.unique(data["wire_id_by_point"][confirmed])
    if len(wire_ids) != result.statistics.confirmed_wire_count:
        raise _error("WIRE_MODEL_A4_WIRE_COUNT", "A4 confirmed wire count is inconsistent")
    for wire_id in wire_ids:
        rows = confirmed[data["wire_id_by_point"][confirmed] == wire_id]
        tracks = np.unique(data["track_id_by_point"][rows])
        if len(tracks) != 1:
            raise _error(
                "WIRE_MODEL_A4_MAPPING", "A4 wire maps to multiple tracks", wire_id=int(wire_id)
            )
        track = int(tracks[0])
        if (
            data["track_status"][track] != int(WireTrackStatus.CONFIRMED)
            or data["track_wire_id"][track] != wire_id
        ):
            raise _error(
                "WIRE_MODEL_A4_STATUS", "A4 confirmed mapping is inconsistent", wire_id=int(wire_id)
            )
    return data, result, hashes


def _reconstruct_wire(
    data: Mapping[str, np.ndarray],
    a4: WireSeparationResult,
    wire_id: int,
    config: WireReconstructionConfig,
) -> tuple[dict[str, Any], list[_Segment], list[dict[str, Any]]]:
    rows = np.flatnonzero(data["wire_id_by_point"] == wire_id)
    track_id = int(data["track_id_by_point"][rows[0]])
    xyz = data["xyz_local"][rows].astype(np.float64)
    direction, direction_flag = _direction(data["track_actual_main_direction"][track_id], xyz)
    anchor = xyz.mean(axis=0)
    projection = (xyz - anchor) @ direction
    order = np.lexsort((data["sample_index"][rows], data["original_index"][rows], projection))
    rows, xyz, projection = rows[order], xyz[order], projection[order]
    reps, rep_t, rep_point_groups = _representatives(
        xyz, projection, data["assignment_score"][rows], config
    )
    threshold = _gap_threshold(rep_t, config)
    boundaries = np.flatnonzero(np.diff(rep_t) > threshold) + 1
    parts = np.split(np.arange(len(reps)), boundaries)
    segments: list[_Segment] = []
    gaps: list[dict[str, Any]] = []
    for segment_index, part in enumerate(parts):
        input_positions = (
            np.concatenate([rep_point_groups[value] for value in part])
            if len(part)
            else np.empty(0, dtype=np.int64)
        )
        segment_rows = rows[np.unique(input_positions)]
        segment = _fit_segment(
            wire_id, segment_index, segment_rows, reps[part], rep_t[part], xyz, projection, config
        )
        segments.append(segment)
    for index in range(len(parts) - 1):
        left, right = parts[index], parts[index + 1]
        gap_length = float(rep_t[right[0]] - rep_t[left[-1]])
        gaps.append(
            {
                "wire_id": wire_id,
                "left_segment_index": index,
                "right_segment_index": index + 1,
                "gap_start_xyz": reps[left[-1]].tolist(),
                "gap_end_xyz": reps[right[0]].tolist(),
                "projected_gap_length_m": gap_length,
                "bridgeable": False,
                "quality_flags": int(
                    WireModelQualityFlag.LARGE_GAP | WireModelQualityFlag.FRAGMENTED_COVERAGE
                ),
                "reason": "unbridgeable_observation_gap",
                "effective_gap_threshold_m": threshold,
            }
        )
    segment_ready = [segment.ready for segment in segments]
    any_ready = any(segment_ready)
    has_large_gap = bool(gaps)
    has_shape_warnings = any(segment.warnings for segment in segments)
    flags = direction_flag
    if has_large_gap:
        flags |= int(WireModelQualityFlag.LARGE_GAP | WireModelQualityFlag.FRAGMENTED_COVERAGE)
    if any(segment.method != WireModelMethod.CUBIC_BSPLINE for segment in segments):
        flags |= int(WireModelQualityFlag.FALLBACK_USED)
    if has_shape_warnings:
        flags |= int(WireModelQualityFlag.OUTLIER_HEAVY)
    if not any_ready:
        status = WireModelStatus.FAILED
        flags |= int(WireModelQualityFlag.RECONSTRUCTION_FAILURE)
    elif has_large_gap or has_shape_warnings or any(not segment.ready for segment in segments):
        status = WireModelStatus.REVIEW_REQUIRED
    elif any(segment.method != WireModelMethod.CUBIC_BSPLINE for segment in segments):
        status = WireModelStatus.FALLBACK_MODEL
    else:
        status = WireModelStatus.CONFIRMED_MODEL
    methods = {segment.method for segment in segments}
    method = next(iter(methods)) if len(methods) == 1 else WireModelMethod.MIXED
    observed_spans = [
        float(segment.representative_t[-1] - segment.representative_t[0])
        if len(segment.representative_t) > 1
        else 0.0
        for segment in segments
    ]
    envelope = float(rep_t[-1] - rep_t[0]) if len(rep_t) > 1 else 0.0
    total_span = float(sum(observed_spans))
    coverage = 1.0 if envelope <= 1e-12 else min(total_span / envelope, 1.0)
    continuous = bool(any_ready and not has_large_gap and all(segment_ready))
    record = {
        "wire_id": wire_id,
        "track_id": track_id,
        "rows": rows,
        "xyz": xyz.astype(np.float32),
        "original_index": data["original_index"][rows],
        "sample_index": data["sample_index"][rows],
        "score": data["assignment_score"][rows],
        "quality": data["assignment_quality_flags"][rows],
        "representatives": reps.astype(np.float32),
        "representative_t": rep_t.astype(np.float32),
        "segments": segments,
        "gaps": gaps,
        "status": status,
        "method": method,
        "flags": flags,
        "has_ready": any_ready,
        "continuous": continuous,
        "direction": direction,
        "observed_total_span": total_span,
        "envelope": envelope,
        "gap_span": float(sum(item["projected_gap_length_m"] for item in gaps)),
        "coverage": coverage,
        "arc_length": float(sum(_arc_length(segment.sampled_xyz) for segment in segments)),
        "threshold": threshold,
    }
    return record, segments, gaps


def _direction(a4_direction: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, int]:
    candidate = np.asarray(a4_direction, dtype=np.float64)
    norm = float(np.linalg.norm(candidate))
    flag = 0
    if not math.isfinite(norm) or norm <= 1e-12:
        flag |= int(WireModelQualityFlag.UNSTABLE_DIRECTION)
        centered = xyz - xyz.mean(axis=0)
        _, values, vectors = np.linalg.svd(centered, full_matrices=False)
        candidate = vectors[0] if len(values) and values[0] > 1e-12 else np.array([1.0, 0.0, 0.0])
    candidate = candidate / np.linalg.norm(candidate)
    centered = xyz - xyz.mean(axis=0)
    try:
        _, values, vectors = np.linalg.svd(centered, full_matrices=False)
        if len(values) and values[0] > 1e-12:
            local = vectors[0]
            if local @ candidate < 0:
                local = -local
            if abs(float(local @ candidate)) >= 0.50:
                candidate = local
            else:
                flag |= int(WireModelQualityFlag.UNSTABLE_DIRECTION)
    except np.linalg.LinAlgError:
        flag |= int(WireModelQualityFlag.UNSTABLE_DIRECTION)
    return candidate / np.linalg.norm(candidate), flag


def _representatives(
    xyz: np.ndarray, projection: np.ndarray, score: np.ndarray, config: WireReconstructionConfig
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    bin_id = np.floor((projection - projection[0]) / config.representative_bin_m + 1e-9).astype(
        np.int64
    )
    representatives, parameters, groups = [], [], []
    for value in np.unique(bin_id):
        positions = np.flatnonzero(bin_id == value)
        weights = np.maximum(score[positions].astype(np.float64), 0.05)
        representatives.append(np.average(xyz[positions], axis=0, weights=weights))
        parameters.append(float(np.average(projection[positions], weights=weights)))
        groups.append(positions)
    return np.asarray(representatives), np.asarray(parameters), groups


def _gap_threshold(rep_t: np.ndarray, config: WireReconstructionConfig) -> float:
    spacing = np.diff(rep_t)
    positive = spacing[spacing > 1e-9]
    local = float(np.median(positive)) if len(positive) else config.representative_bin_m
    return float(
        np.clip(
            local * config.gap_relative_multiplier,
            config.gap_absolute_min_m,
            config.gap_absolute_max_m,
        )
    )


def _fit_segment(
    wire_id: int,
    index: int,
    point_indices: np.ndarray,
    reps: np.ndarray,
    rep_t: np.ndarray,
    ordered_xyz: np.ndarray,
    ordered_t: np.ndarray,
    config: WireReconstructionConfig,
) -> _Segment:
    flags = 0
    reason = ""
    span = float(rep_t[-1] - rep_t[0]) if len(rep_t) > 1 else 0.0
    if len(reps) < 2:
        return _failed_segment(
            wire_id, index, point_indices, reps, rep_t, "insufficient_representatives"
        )
    if span < config.minimum_segment_span_m:
        flags |= int(WireModelQualityFlag.INSUFFICIENT_SPAN)
    if len(reps) < config.minimum_segment_points:
        flags |= int(WireModelQualityFlag.INSUFFICIENT_POINTS)
    local_mask = (ordered_t >= rep_t[0] - 1e-9) & (ordered_t <= rep_t[-1] + 1e-9)
    local_xyz = ordered_xyz[local_mask]
    local_t = ordered_t[local_mask]
    representative_arc_length = _arc_length(reps)
    candidate_diagnostics: list[dict[str, Any]] = []
    accepted_candidates: list[_SplineCandidate] = []
    if not flags:
        parameter = (rep_t - rep_t[0]) / max(span, 1e-12)
        sample_t = np.linspace(0.0, 1.0, _sample_count(reps, config), dtype=np.float64)
        for factor in config.smoothing_candidates:
            try:
                tck, _ = splprep(
                    reps.T,
                    u=parameter,
                    k=3,
                    s=factor * len(reps),
                    nest=min(len(reps) + 4, 128),
                )
                sampled = np.asarray(splev(sample_t, tck)).T
                support_residual, representative_residual, observation_residual = (
                    _spline_support_residuals(
                        tck, parameter, reps, local_t, local_xyz, rep_t[0], span
                    )
                )
                nearest_sample_residual = _nearest_sample_residual(local_xyz, sampled)
                turn_angle = _turn_angle(sampled)
                sample_step = _sample_step_statistics(sampled)
                candidate_arc_length = _arc_length(sampled)
                arc_length_inflation_ratio = candidate_arc_length / max(
                    representative_arc_length, 1e-12
                )
                accepted, rejection_reasons, warnings = _spline_acceptance(
                    sampled,
                    sample_t,
                    support_residual,
                    turn_angle,
                    sample_step,
                    arc_length_inflation_ratio,
                    reps,
                    config,
                )
                candidate = _SplineCandidate(
                    factor,
                    sampled,
                    rep_t[0] + sample_t * span,
                    np.asarray(tck[1]).T,
                    np.asarray(tck[0]),
                    int(tck[2]),
                    support_residual,
                    nearest_sample_residual,
                    representative_residual,
                    observation_residual,
                    turn_angle,
                    sample_step,
                    representative_arc_length,
                    candidate_arc_length,
                    arc_length_inflation_ratio,
                    warnings,
                    accepted,
                    rejection_reasons,
                )
                candidate_diagnostics.append(_candidate_diagnostic(candidate))
                if accepted:
                    accepted_candidates.append(candidate)
            except Exception as error:  # SciPy failures must remain auditable per candidate.
                candidate_diagnostics.append(
                    {
                        "smoothing_value": factor,
                        "accepted": False,
                        "rejection_reasons": [f"spline_exception:{type(error).__name__}"],
                    }
                )
        if accepted_candidates:
            # Candidates are validated as increasing; this is the least-smoothed acceptable curve.
            selected = accepted_candidates[0]
            return _segment(
                wire_id,
                index,
                point_indices,
                reps,
                rep_t,
                WireModelMethod.CUBIC_BSPLINE,
                0,
                True,
                selected.sampled_xyz,
                selected.sampled_t,
                selected.control_xyz,
                selected.knots,
                selected.degree,
                selected.support_residual,
                selected.nearest_sample_residual,
                selected.representative_residual,
                selected.observation_residual,
                selected.turn_angle,
                _endpoint_uncertainty(reps),
                "",
                selected.sample_step,
                selected.representative_arc_length,
                selected.candidate_arc_length,
                selected.arc_length_inflation_ratio,
                selected.warnings,
                selected.smoothing_value,
                tuple(candidate_diagnostics),
            )
        if candidate_diagnostics:
            last_reasons = candidate_diagnostics[-1].get("rejection_reasons", [])
            reason = f"spline_rejected_{last_reasons[0]}" if last_reasons else "spline_rejected"
            all_reasons = {
                item_reason
                for item in candidate_diagnostics
                for item_reason in item.get("rejection_reasons", [])
            }
            if "high_residual" in all_reasons:
                flags |= int(WireModelQualityFlag.HIGH_RESIDUAL)
            if "unsupported_oscillation" in all_reasons or "longitudinal_reversal" in all_reasons:
                flags |= int(WireModelQualityFlag.HIGH_TURN_ANGLE)
    else:
        reason = "insufficient_support_for_cubic_spline"
    smooth = _polyline_samples(reps, rep_t, config)
    support_residual, representative_residual, observation_residual = _linear_support_residuals(
        reps, rep_t, local_t, local_xyz, smooth[1], smooth[0]
    )
    nearest_sample_residual = _nearest_sample_residual(local_xyz, smooth[0])
    turn_angle = _turn_angle(smooth[0])
    sample_step = _sample_step_statistics(smooth[0])
    candidate_arc_length = _arc_length(smooth[0])
    arc_length_inflation_ratio = candidate_arc_length / max(representative_arc_length, 1e-12)
    fallback_accepted, _, warnings = _spline_acceptance(
        smooth[0],
        (smooth[1] - rep_t[0]) / max(span, 1e-12),
        support_residual,
        turn_angle,
        sample_step,
        arc_length_inflation_ratio,
        reps,
        config,
    )
    ready = fallback_accepted and not (
        flags
        & int(WireModelQualityFlag.INSUFFICIENT_SPAN | WireModelQualityFlag.INSUFFICIENT_POINTS)
    )
    if len(reps) >= 3:
        return _segment(
            wire_id,
            index,
            point_indices,
            reps,
            rep_t,
            WireModelMethod.SMOOTHED_POLYLINE,
            flags | int(WireModelQualityFlag.FALLBACK_USED),
            ready,
            smooth[0],
            smooth[1],
            reps,
            np.empty(0, dtype=np.float64),
            1,
            support_residual,
            nearest_sample_residual,
            representative_residual,
            observation_residual,
            turn_angle,
            _endpoint_uncertainty(reps),
            reason or "spline_unavailable",
            sample_step,
            representative_arc_length,
            candidate_arc_length,
            arc_length_inflation_ratio,
            warnings,
            None,
            tuple(candidate_diagnostics),
        )
    return _segment(
        wire_id,
        index,
        point_indices,
        reps,
        rep_t,
        WireModelMethod.PIECEWISE_LINEAR,
        flags | int(WireModelQualityFlag.FALLBACK_USED),
        ready,
        smooth[0],
        smooth[1],
        reps,
        np.empty(0, dtype=np.float64),
        1,
        support_residual,
        nearest_sample_residual,
        representative_residual,
        observation_residual,
        turn_angle,
        _endpoint_uncertainty(reps),
        reason or "two_point_segment",
        sample_step,
        representative_arc_length,
        candidate_arc_length,
        arc_length_inflation_ratio,
        warnings,
        None,
        tuple(candidate_diagnostics),
    )


def _failed_segment(
    wire_id: int,
    index: int,
    point_indices: np.ndarray,
    reps: np.ndarray,
    rep_t: np.ndarray,
    reason: str,
) -> _Segment:
    xyz = reps if len(reps) else np.zeros((1, 3), dtype=np.float64)
    parameters = rep_t if len(rep_t) else np.zeros(1, dtype=np.float64)
    return _segment(
        wire_id,
        index,
        point_indices,
        reps,
        rep_t,
        WireModelMethod.NONE,
        int(WireModelQualityFlag.INSUFFICIENT_POINTS | WireModelQualityFlag.RECONSTRUCTION_FAILURE),
        False,
        xyz,
        parameters,
        np.empty((0, 3)),
        np.empty(0),
        0,
        (1e30, 1e30, 1e30, 1e30),
        (1e30, 1e30, 1e30, 1e30),
        (1e30, 1e30, 1e30, 1e30),
        (1e30, 1e30, 1e30, 1e30),
        (1e30, 1e30),
        1e30,
        reason,
        (0.0, 0.0, 0.0),
        0.0,
        0.0,
        0.0,
        (),
    )


def _segment(
    wire_id: int,
    index: int,
    points: np.ndarray,
    reps: np.ndarray,
    rep_t: np.ndarray,
    method: WireModelMethod,
    flags: int,
    ready: bool,
    sampled: np.ndarray,
    sampled_t: np.ndarray,
    control: np.ndarray,
    knots: np.ndarray,
    degree: int,
    support_residual: tuple[float, float, float, float],
    nearest_sample_residual: tuple[float, float, float, float],
    representative_residual: tuple[float, float, float, float],
    observation_residual: tuple[float, float, float, float],
    turn_angle: tuple[float, float],
    uncertainty: float,
    reason: str,
    sample_step: tuple[float, float, float] | None = None,
    representative_arc_length: float = 0.0,
    candidate_arc_length: float = 0.0,
    arc_length_inflation_ratio: float = 0.0,
    warnings: tuple[str, ...] = (),
    smoothing_value: float | None = None,
    spline_candidates: tuple[Mapping[str, Any], ...] = (),
) -> _Segment:
    return _Segment(
        wire_id,
        index,
        points,
        reps,
        rep_t,
        method,
        flags,
        ready,
        np.asarray(sampled, dtype=np.float64),
        np.asarray(sampled_t, dtype=np.float64),
        np.asarray(control, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        degree,
        support_residual,
        nearest_sample_residual,
        representative_residual,
        observation_residual,
        turn_angle,
        uncertainty,
        reason,
        (0.0, 0.0, 0.0) if sample_step is None else sample_step,
        representative_arc_length,
        candidate_arc_length,
        arc_length_inflation_ratio,
        warnings,
        smoothing_value,
        spline_candidates,
    )


def _polyline_samples(
    reps: np.ndarray, rep_t: np.ndarray, config: WireReconstructionConfig
) -> tuple[np.ndarray, np.ndarray]:
    count = _sample_count(reps, config)
    parameters = np.linspace(rep_t[0], rep_t[-1], count)
    return np.column_stack(
        [np.interp(parameters, rep_t, reps[:, axis]) for axis in range(3)]
    ), parameters


def _sample_count(xyz: np.ndarray, config: WireReconstructionConfig) -> int:
    length = _arc_length(xyz)
    return int(
        np.clip(
            math.ceil(length / config.sample_spacing_m) + 1, 2, config.maximum_samples_per_segment
        )
    )


def _residual_statistics(values: np.ndarray) -> tuple[float, float, float, float]:
    if not len(values) or not np.isfinite(values).all():
        return (math.inf, math.inf, math.inf, math.inf)
    return (
        float(math.sqrt(np.mean(values * values))),
        float(np.median(values)),
        float(np.percentile(values, 95)),
        float(np.max(values)),
    )


def _nearest_sample_residual(
    points: np.ndarray, sampled: np.ndarray
) -> tuple[float, float, float, float]:
    if not len(points) or not len(sampled):
        return (math.inf, math.inf, math.inf, math.inf)
    return _residual_statistics(cKDTree(sampled).query(points, k=1)[0])


def _spline_support_residuals(
    tck: tuple[np.ndarray, np.ndarray, int],
    representative_parameter: np.ndarray,
    representatives: np.ndarray,
    observation_t: np.ndarray,
    observations: np.ndarray,
    start_t: float,
    span: float,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    representative_values = np.linalg.norm(
        np.asarray(splev(representative_parameter, tck)).T - representatives, axis=1
    )
    observation_parameter = (observation_t - start_t) / max(span, 1e-12)
    observation_values = np.linalg.norm(
        np.asarray(splev(observation_parameter, tck)).T - observations, axis=1
    )
    return (
        _residual_statistics(np.concatenate((representative_values, observation_values))),
        _residual_statistics(representative_values),
        _residual_statistics(observation_values),
    )


def _linear_support_residuals(
    representatives: np.ndarray,
    representative_t: np.ndarray,
    observation_t: np.ndarray,
    observations: np.ndarray,
    model_t: np.ndarray,
    model_xyz: np.ndarray,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    def evaluate(parameters: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.interp(parameters, model_t, model_xyz[:, axis]) for axis in range(3)]
        )

    representative_values = np.linalg.norm(evaluate(representative_t) - representatives, axis=1)
    observation_values = np.linalg.norm(evaluate(observation_t) - observations, axis=1)
    return (
        _residual_statistics(np.concatenate((representative_values, observation_values))),
        _residual_statistics(representative_values),
        _residual_statistics(observation_values),
    )


def _turn_angle(xyz: np.ndarray) -> tuple[float, float]:
    """Return p95 and maximum consecutive-step turning angles in radians, not curvature."""
    segments = np.diff(xyz, axis=0)
    norms = np.linalg.norm(segments, axis=1)
    good = norms > 1e-12
    vectors = segments[good] / norms[good, None]
    if len(vectors) < 2:
        return (0.0, 0.0)
    angles = np.arccos(np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1.0, 1.0))
    return (
        float(np.percentile(angles, 95)) if len(angles) else 0.0,
        float(np.max(angles)) if len(angles) else 0.0,
    )


def _sample_step_statistics(xyz: np.ndarray) -> tuple[float, float, float]:
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if not len(steps):
        return (0.0, 0.0, 0.0)
    return (float(np.median(steps)), float(np.percentile(steps, 95)), float(np.max(steps)))


def _spline_acceptance(
    sampled: np.ndarray,
    parameter: np.ndarray,
    support_residual: tuple[float, float, float, float],
    turn_angle: tuple[float, float],
    sample_step: tuple[float, float, float],
    arc_length_inflation_ratio: float,
    representatives: np.ndarray,
    config: WireReconstructionConfig,
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    reasons: list[str] = []
    warnings: list[str] = []
    nonfinite_geometry = not np.isfinite(sampled).all() or not np.isfinite(sampled[[0, -1]]).all()
    invalid_parameterization = (
        len(parameter) < 2
        or not np.isfinite(parameter).all()
        or np.any(np.diff(parameter) <= 0)
        or parameter[0] < -1e-12
        or parameter[-1] > 1.0 + 1e-12
    )
    if nonfinite_geometry:
        reasons.append("nonfinite_or_unstable_endpoint")
    if invalid_parameterization:
        reasons.append("invalid_parameterization")
    if support_residual[2] > config.maximum_residual_m:
        reasons.append("high_residual")
    turn_angle_spike = turn_angle[1] > config.maximum_turn_angle_rad or (
        turn_angle[0] > 1e-12
        and turn_angle[1] > turn_angle[0] * config.maximum_turn_angle_spike_multiplier
    )
    step_limit = max(
        config.sample_spacing_m * config.maximum_sample_step_multiplier,
        sample_step[0] * config.maximum_sample_step_multiplier,
        sample_step[1] * config.maximum_sample_step_multiplier,
    )
    sample_step_spike = sample_step[2] > step_limit
    unsupported_arc_inflation = (
        not math.isfinite(arc_length_inflation_ratio)
        or arc_length_inflation_ratio > config.maximum_arc_length_inflation_ratio
    )
    direction = (
        representatives[-1] - representatives[0] if len(representatives) >= 2 else np.zeros(3)
    )
    direction_norm = float(np.linalg.norm(direction))
    longitudinal_reversal = bool(
        direction_norm > 1e-12
        and np.any(np.diff(sampled, axis=0) @ (direction / direction_norm) < -1e-9)
    )
    if turn_angle_spike:
        warnings.append("turn_angle_spike_detected")
    if sample_step_spike:
        warnings.append("sample_step_spike_detected")
    if unsupported_arc_inflation:
        warnings.append("unsupported_arc_inflation_detected")
    if longitudinal_reversal:
        warnings.append("longitudinal_reversal_detected")
        reasons.append("longitudinal_reversal")
    if (turn_angle_spike or sample_step_spike) and unsupported_arc_inflation:
        reasons.append("unsupported_oscillation")
    return not reasons, tuple(reasons), tuple(warnings)


def _candidate_diagnostic(candidate: _SplineCandidate) -> dict[str, Any]:
    return {
        "smoothing_value": candidate.smoothing_value,
        "support_residual_p95_m": candidate.support_residual[2],
        "support_residual_max_m": candidate.support_residual[3],
        "representative_residual_p95_m": candidate.representative_residual[2],
        "observation_residual_p95_m": candidate.observation_residual[2],
        "nearest_sample_residual_p95_m": candidate.nearest_sample_residual[2],
        "turn_angle_p95_rad": candidate.turn_angle[0],
        "turn_angle_max_rad": candidate.turn_angle[1],
        "sample_step_median_m": candidate.sample_step[0],
        "sample_step_p95_m": candidate.sample_step[1],
        "sample_step_max_m": candidate.sample_step[2],
        "representative_polyline_arc_length_m": candidate.representative_arc_length,
        "candidate_arc_length_m": candidate.candidate_arc_length,
        "arc_length_inflation_ratio": candidate.arc_length_inflation_ratio,
        "warnings": list(candidate.warnings),
        "accepted": candidate.accepted,
        "rejection_reasons": list(candidate.rejection_reasons),
    }


def _endpoint_uncertainty(reps: np.ndarray) -> float:
    return (
        float(max(np.linalg.norm(reps[1] - reps[0]), np.linalg.norm(reps[-1] - reps[-2])))
        if len(reps) >= 2
        else math.inf
    )


def _arc_length(xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


def _arrays_and_summaries(
    records: list[dict[str, Any]], segments: list[_Segment], gaps: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], list[WireModelSummary], list[dict[str, Any]]]:
    records.sort(key=lambda item: item["wire_id"])
    segments.sort(key=lambda item: (item.wire_id, item.index))
    wire_offsets = _offsets([record["segments"] for record in records])
    input_offsets = _offsets([record["rows"] for record in records])
    rep_offsets = _offsets([record["representatives"] for record in records])
    sample_offsets = _offsets([segment.sampled_xyz for segment in segments])
    control_offsets = _offsets([segment.control_xyz for segment in segments])
    knot_offsets = _offsets([segment.knots for segment in segments])
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "wire_id": np.asarray([record["wire_id"] for record in records], dtype=np.int32),
        "source_track_id": np.asarray([record["track_id"] for record in records], dtype=np.int32),
        "wire_status": np.asarray([int(record["status"]) for record in records], dtype=np.uint8),
        "wire_method": np.asarray([int(record["method"]) for record in records], dtype=np.uint8),
        "wire_quality_flags": np.asarray([record["flags"] for record in records], dtype=np.uint16),
        "wire_has_distance_ready_segments": np.asarray(
            [record["has_ready"] for record in records], dtype=bool
        ),
        "wire_continuous_distance_ready": np.asarray(
            [record["continuous"] for record in records], dtype=bool
        ),
        "wire_segment_offsets": wire_offsets,
        "wire_input_offsets": input_offsets,
        "wire_representative_offsets": rep_offsets,
        "ordered_input_xyz": _concat([record["xyz"] for record in records], (0, 3), np.float32),
        "ordered_input_row_index": _concat([record["rows"] for record in records], (0,), np.int64),
        "ordered_input_original_index": _concat(
            [record["original_index"] for record in records], (0,), np.int64
        ),
        "ordered_input_sample_index": _concat(
            [record["sample_index"] for record in records], (0,), np.int64
        ),
        "ordered_input_score": _concat([record["score"] for record in records], (0,), np.float32),
        "ordered_input_quality_flags": _concat(
            [record["quality"] for record in records], (0,), np.uint16
        ),
        "representative_xyz": _concat(
            [record["representatives"] for record in records], (0, 3), np.float32
        ),
        "representative_parameter": _concat(
            [record["representative_t"] for record in records], (0,), np.float32
        ),
        "segment_wire_id": np.asarray([segment.wire_id for segment in segments], dtype=np.int32),
        "segment_index": np.asarray([segment.index for segment in segments], dtype=np.int32),
        "segment_method": np.asarray([int(segment.method) for segment in segments], dtype=np.uint8),
        "segment_quality_flags": np.asarray(
            [segment.flags for segment in segments], dtype=np.uint16
        ),
        "segment_distance_ready": np.asarray([segment.ready for segment in segments], dtype=bool),
        "segment_parameter_domain": np.asarray(
            [
                [
                    segment.representative_t[0] if len(segment.representative_t) else 0.0,
                    segment.representative_t[-1] if len(segment.representative_t) else 0.0,
                ]
                for segment in segments
            ],
            dtype=np.float64,
        ),
        "segment_start_xyz": np.asarray(
            [segment.sampled_xyz[0] for segment in segments], dtype=np.float32
        ),
        "segment_end_xyz": np.asarray(
            [segment.sampled_xyz[-1] for segment in segments], dtype=np.float32
        ),
        "segment_observed_span_m": np.asarray(
            [
                segment.representative_t[-1] - segment.representative_t[0]
                if len(segment.representative_t) > 1
                else 0.0
                for segment in segments
            ],
            dtype=np.float32,
        ),
        "segment_arc_length_m": np.asarray(
            [_arc_length(segment.sampled_xyz) for segment in segments], dtype=np.float32
        ),
        "segment_support_residual_stats_m": np.asarray(
            [segment.support_residual for segment in segments], dtype=np.float32
        ),
        "segment_nearest_sample_residual_stats_m": np.asarray(
            [segment.nearest_sample_residual for segment in segments], dtype=np.float32
        ),
        "segment_representative_residual_stats_m": np.asarray(
            [segment.representative_residual for segment in segments], dtype=np.float32
        ),
        "segment_observation_residual_stats_m": np.asarray(
            [segment.observation_residual for segment in segments], dtype=np.float32
        ),
        "segment_turn_angle_stats_rad": np.asarray(
            [segment.turn_angle for segment in segments], dtype=np.float32
        ),
        "segment_endpoint_uncertainty_m": np.asarray(
            [segment.endpoint_uncertainty for segment in segments], dtype=np.float32
        ),
        "segment_sample_step_stats_m": np.asarray(
            [segment.sample_step for segment in segments], dtype=np.float32
        ),
        "segment_representative_polyline_arc_length_m": np.asarray(
            [segment.representative_arc_length for segment in segments], dtype=np.float32
        ),
        "segment_candidate_arc_length_m": np.asarray(
            [segment.candidate_arc_length for segment in segments], dtype=np.float32
        ),
        "segment_arc_length_inflation_ratio": np.asarray(
            [segment.arc_length_inflation_ratio for segment in segments], dtype=np.float32
        ),
        "segment_smoothing_value": np.asarray(
            [
                segment.smoothing_value if segment.smoothing_value is not None else -1.0
                for segment in segments
            ],
            dtype=np.float32,
        ),
        "sampled_curve_xyz": _concat(
            [segment.sampled_xyz for segment in segments], (0, 3), np.float32
        ),
        "sampled_curve_parameter": _concat(
            [segment.sampled_t for segment in segments], (0,), np.float32
        ),
        "sampled_curve_offsets": sample_offsets,
        "control_xyz": _concat([segment.control_xyz for segment in segments], (0, 3), np.float32),
        "control_offsets": control_offsets,
        "knot_values": _concat([segment.knots for segment in segments], (0,), np.float64),
        "knot_offsets": knot_offsets,
        "spline_degree": np.asarray([segment.degree for segment in segments], dtype=np.uint8),
        "gap_wire_id": np.asarray([item["wire_id"] for item in gaps], dtype=np.int32),
        "gap_left_segment_index": np.asarray(
            [item["left_segment_index"] for item in gaps], dtype=np.int32
        ),
        "gap_right_segment_index": np.asarray(
            [item["right_segment_index"] for item in gaps], dtype=np.int32
        ),
        "gap_start_xyz": np.asarray(
            [item["gap_start_xyz"] for item in gaps], dtype=np.float32
        ).reshape((-1, 3)),
        "gap_end_xyz": np.asarray([item["gap_end_xyz"] for item in gaps], dtype=np.float32).reshape(
            (-1, 3)
        ),
        "gap_projected_length_m": np.asarray(
            [item["projected_gap_length_m"] for item in gaps], dtype=np.float32
        ),
        "gap_bridgeable": np.asarray([item["bridgeable"] for item in gaps], dtype=bool),
        "gap_quality_flags": np.asarray([item["quality_flags"] for item in gaps], dtype=np.uint16),
    }
    summaries = []
    for record in records:
        starts = [segment.sampled_xyz[0] for segment in record["segments"]]
        ends = [segment.sampled_xyz[-1] for segment in record["segments"]]
        summaries.append(
            WireModelSummary(
                record["wire_id"],
                record["track_id"],
                record["status"],
                record["method"],
                record["flags"],
                record["has_ready"],
                record["continuous"],
                len(record["segments"]),
                len(record["rows"]),
                len(record["representatives"]),
                record["observed_total_span"],
                record["envelope"],
                record["gap_span"],
                record["coverage"],
                record["arc_length"],
                tuple(float(value) for value in starts[0]),
                tuple(float(value) for value in ends[-1]),
                tuple(float(value) for value in record["direction"]),
                "unbridgeable_observation_gap"
                if record["gaps"]
                else (
                    next((segment.reason for segment in record["segments"] if segment.reason), "")
                ),
            )
        )
    segment_summaries = [
        {
            "wire_id": segment.wire_id,
            "segment_index": segment.index,
            "method": segment.method.name,
            "quality_flags": segment.flags,
            "segment_distance_ready": segment.ready,
            "point_count": len(segment.point_indices),
            "representative_count": len(segment.representative_xyz),
            "observed_span_m": float(segment.representative_t[-1] - segment.representative_t[0])
            if len(segment.representative_t) > 1
            else 0.0,
            "arc_length_m": _arc_length(segment.sampled_xyz),
            "support_residual_definition": (
                "combined_representative_and_observation_parameter_matched"
            ),
            "support_residual_rmse_m": segment.support_residual[0],
            "support_residual_median_m": segment.support_residual[1],
            "support_residual_p95_m": segment.support_residual[2],
            "support_residual_max_m": segment.support_residual[3],
            "representative_residual_stats_m": list(segment.representative_residual),
            "observation_residual_stats_m": list(segment.observation_residual),
            "nearest_sample_residual_stats_m": list(segment.nearest_sample_residual),
            "turn_angle_p95_rad": segment.turn_angle[0],
            "turn_angle_max_rad": segment.turn_angle[1],
            "endpoint_uncertainty_m": segment.endpoint_uncertainty,
            "sample_step_median_m": segment.sample_step[0],
            "sample_step_p95_m": segment.sample_step[1],
            "sample_step_max_m": segment.sample_step[2],
            "representative_polyline_arc_length_m": segment.representative_arc_length,
            "candidate_arc_length_m": segment.candidate_arc_length,
            "arc_length_inflation_ratio": segment.arc_length_inflation_ratio,
            "warnings": list(segment.warnings),
            "selected_smoothing_value": segment.smoothing_value,
            "spline_candidates": [dict(item) for item in segment.spline_candidates],
            "reason": segment.reason,
        }
        for segment in segments
    ]
    return arrays, summaries, segment_summaries


def _offsets(groups: list[Any]) -> np.ndarray:
    return np.asarray(
        [0, *np.cumsum([len(group) for group in groups], dtype=np.int64)], dtype=np.int64
    )


def _concat(
    groups: list[np.ndarray], empty_shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.ndarray:
    return (
        np.concatenate(groups).astype(dtype, copy=False)
        if groups
        else np.empty(empty_shape, dtype=dtype)
    )


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _validate_model_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "schema_version",
        "wire_id",
        "source_track_id",
        "wire_status",
        "wire_method",
        "wire_quality_flags",
        "wire_has_distance_ready_segments",
        "wire_continuous_distance_ready",
        "wire_segment_offsets",
        "segment_wire_id",
        "segment_index",
        "segment_method",
        "segment_quality_flags",
        "segment_distance_ready",
        "segment_parameter_domain",
        "sampled_curve_xyz",
        "sampled_curve_parameter",
        "sampled_curve_offsets",
    }
    if missing := required - set(arrays):
        raise _error("WIRE_MODEL_ARRAY_MISSING", "W1 array is missing", arrays=sorted(missing))
    if str(np.asarray(arrays["schema_version"]).item()) != SCHEMA_VERSION:
        raise _error("WIRE_MODEL_SCHEMA", "W1 NPZ schema version is unsupported")
    WireModelSet(
        arrays["wire_id"],
        arrays["source_track_id"],
        arrays["wire_status"],
        arrays["wire_method"],
        arrays["wire_quality_flags"],
        arrays["wire_has_distance_ready_segments"],
        arrays["wire_continuous_distance_ready"],
        arrays["wire_segment_offsets"],
        arrays["segment_wire_id"],
        arrays["segment_index"],
        arrays["segment_method"],
        arrays["segment_quality_flags"],
        arrays["segment_distance_ready"],
        arrays["segment_parameter_domain"],
        arrays["sampled_curve_xyz"],
        arrays["sampled_curve_parameter"],
        arrays["sampled_curve_offsets"],
    )


def _publish(
    arrays: Mapping[str, np.ndarray],
    result: WireReconstructionResult,
    output_npz: Path,
    output_json: Path,
    started: float,
    process: Any | None,
) -> WireReconstructionResult:
    temporary_npz, temporary_json = _temporary(output_npz), _temporary(output_json)
    npz_published = False
    try:
        np.savez_compressed(temporary_npz, **arrays)
        with np.load(temporary_npz, allow_pickle=False) as stream:
            _validate_model_arrays({name: stream[name] for name in stream.files})
        values = [
            value
            for value in (
                result.statistics.rss_start_bytes,
                result.statistics.rss_peak_bytes,
                _rss(process),
            )
            if value is not None
        ]
        result = WireReconstructionResult.from_dict(
            result.to_dict()
            | {
                "output_npz_sha256": _sha(temporary_npz),
                "output_npz_bytes": temporary_npz.stat().st_size,
                "statistics": result.statistics.to_dict()
                | {
                    "elapsed_seconds": time.perf_counter() - started,
                    "rss_peak_bytes": max(values) if values else None,
                    "rss_end_bytes": _rss(process),
                },
            }
        )
        temporary_json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        WireReconstructionResult.from_dict(json.loads(temporary_json.read_text("utf-8")))
        os.replace(temporary_npz, output_npz)
        npz_published = True
        os.replace(temporary_json, output_json)  # Manifest last: formal completion marker.
        return result
    except Exception:
        temporary_npz.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
        if npz_published:
            output_npz.unlink(missing_ok=True)
        raise


def _validate_output_dir(output: Path, allowed: Path) -> None:
    try:
        output.relative_to(allowed.resolve())
    except ValueError as error:
        raise _error(
            "WIRE_MODEL_OUTPUT_FORBIDDEN",
            "output must be under the approved directory",
            output=str(output),
        ) from error


def _validate_outputs(paths: tuple[Path, Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _error(
            "WIRE_MODEL_OUTPUT_EXISTS", "output exists and overwrite is disabled", paths=existing
        )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise _error(
            "WIRE_MODEL_PATH", "path must remain under project root", path=str(path)
        ) from error


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(code: str, message: str, **details: Any) -> WireReconstructionError:
    cause = details.pop("cause", None)
    return WireReconstructionError(
        UnifiedError(code, message, "error", "algorithms.wires.reconstruct", False, details, cause)
    )


def _check_cancelled(token: CancellationToken | None, context: str) -> None:
    if token is not None and token.is_cancelled():
        raise _error(
            "WIRE_MODEL_CANCELLED", f"W1 reconstruction cancelled {context}", context=context
        )


def _current_process() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process()


def _rss(process: Any | None) -> int | None:
    return None if process is None else int(process.memory_info().rss)


def _report(
    reporter: ProgressReporter | None,
    source: str,
    stage: str,
    done: int,
    total: int,
    status: str = "running",
) -> None:
    if reporter is not None:
        reporter.update(
            PipelineProgress(
                f"{source}:wire-reconstruction",
                stage,
                status,
                done,
                total,
                100.0 if total == 0 else 100.0 * done / total,
                f"{stage}: {done} of {total}",
            )
        )
