"""A3 explainable rule/manual wire-candidate baseline; it never reads LAS."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import (
    CancellationToken,
    CandidateEvidenceSet,
    CandidateLevel,
    CandidatePointSet,
    CandidateReasonFlag,
    CandidateSource,
    FeatureQualityFlag,
    GroundQualityFlag,
    ManualSelectionConfig,
    PipelineProgress,
    ProgressReporter,
    UnifiedError,
    WireCandidateConfig,
    WireCandidateQualityFlag,
    WireCandidateResult,
    WireCandidateStatistics,
)

ALGORITHM_VERSION = "a3-rule-manual-wire-candidates-v1"
_BASE_NAME = "scene_simple_wire_candidates"
_FORMULAS = {
    "geometry_guard": "linearity >= 0.60 and scattering <= 0.10 and not NUMERIC_INVALID",
    "linearity_score": "clip((linearity-0.60)/(0.90-0.60), 0, 1)",
    "scattering_score": "clip((0.10-scattering)/(0.10-0.01), 0, 1)",
    "planarity_score": "clip((0.60-planarity)/(0.60-0.20), 0, 1)",
    "height_score": (
        "HAG normally; 0.35*HAG + 0.65*height_m for high ground variation; "
        "height_m fallback for invalid, below-ground, extrapolated, or extreme HAG"
    ),
    "density_score": "0 <=50, ramp 50-150, 1 for 150-1500, ramp 1500-3000, 0 >=3000 points/m^3",
    "rule_score": "normalized weighted component scores multiplied by (1-quality_penalty)",
    "candidate_score": (
        "manual include uses max(rule_score, manual_include_score); "
        "manual exclude sets score to zero"
    ),
}


class WireCandidateError(Exception):
    """A structured, recoverable A3 candidate-selection failure."""

    def __init__(self, error: UnifiedError) -> None:
        super().__init__(error.message)
        self.error = error


def run_wire_candidate_selection(
    a1_npz_path: Path | str,
    a1_json_path: Path | str,
    a2a_npz_path: Path | str,
    a2a_json_path: Path | str,
    a2b_npz_path: Path | str,
    a2b_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: WireCandidateConfig | None = None,
    manual_selection: ManualSelectionConfig | None = None,
    mode: str = "rule",
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
    generate_preview: bool = True,
) -> WireCandidateResult:
    """Generate full-length A3 arrays from verified A1, A2A, and A2B artifacts."""

    config = config or WireCandidateConfig()
    manual_selection = manual_selection or ManualSelectionConfig()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output_dir = Path(output_dir).resolve()
    _validate_output_dir(output_dir, root / "outputs" / "candidates")
    _validate_mode(mode, manual_selection)

    output_npz_path = output_dir / f"{_BASE_NAME}.npz"
    output_json_path = output_dir / f"{_BASE_NAME}.json"
    _validate_outputs_absent((output_npz_path, output_json_path), overwrite=overwrite)
    _check_cancelled(cancellation_token, "before input validation")
    pipeline_started = time.perf_counter()
    process = _current_process() if config.record_memory else None
    rss_start = _rss_bytes(process)
    rss_peak = rss_start
    paths = _input_paths(
        a1_npz_path,
        a1_json_path,
        a2a_npz_path,
        a2a_json_path,
        a2b_npz_path,
        a2b_json_path,
    )
    data, metadata, input_hashes = _load_and_validate_inputs(paths, config)
    _report(reporter, config.source_id, "validated_inputs", 0, len(data["a1"]["xyz_local"]))

    try:
        arrays, points, evidence, statistics = _score_candidates(
            data,
            config,
            manual_selection,
            mode,
            cancellation_token,
            process,
            rss_start,
            rss_peak,
            pipeline_started,
        )
        rss_peak = max(rss_peak or 0, statistics.rss_peak_bytes or 0) or None
        _report(
            reporter,
            config.source_id,
            "scored_candidates",
            points.point_count,
            points.point_count,
        )
        _check_cancelled(cancellation_token, "before output publication")
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays_metadata = points.array_metadata() | evidence.array_metadata()
        result = _write_outputs(
            arrays,
            points,
            evidence,
            paths,
            input_hashes,
            output_npz_path,
            output_json_path,
            config,
            manual_selection,
            mode,
            statistics,
            arrays_metadata,
            pipeline_started,
            process,
        )
    except Exception:
        raise

    if generate_preview:
        preview_path = root / "outputs" / "previews" / f"{_BASE_NAME}_preview.html"
        try:
            generate_wire_candidate_preview(
                paths["a1_npz"],
                output_npz_path,
                preview_path,
                overwrite=overwrite,
                project_root=root,
            )
        except Exception as error:  # Preview is deliberately non-fatal after core publication.
            result = replace(result, warnings=(f"preview generation failed: {error}",))
    _report(
        reporter,
        config.source_id,
        "completed",
        points.point_count,
        points.point_count,
        status="completed",
    )
    return result


def generate_wire_candidate_preview(
    a1_npz_path: Path | str,
    candidate_npz_path: Path | str,
    output_html_path: Path | str,
    *,
    overwrite: bool = False,
    project_root: Path | str | None = None,
    max_background_points: int = 40_000,
    max_candidate_points: int = 50_000,
) -> Path:
    """Write a deterministic, lightweight 3D candidate preview from NPZ only."""

    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output_html_path = Path(output_html_path).resolve()
    _validate_output_dir(output_html_path.parent, root / "outputs" / "previews")
    if output_html_path.exists() and not overwrite:
        raise _error(
            "CANDIDATE_PREVIEW_EXISTS",
            "A3 preview already exists and overwrite is disabled",
            paths=[str(output_html_path)],
        )
    if max_background_points <= 0 or max_candidate_points <= 0:
        raise ValueError("preview point limits must be positive")
    with np.load(Path(a1_npz_path), allow_pickle=False) as stream:
        xyz = stream["xyz_local"]
    with np.load(Path(candidate_npz_path), allow_pickle=False) as stream:
        mask = stream["candidate_mask"]
        score = stream["candidate_score"]
        level = stream["candidate_level"]
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise _error("CANDIDATE_PREVIEW_INPUT", "A1 xyz_local has invalid shape or dtype")
    point_count = len(xyz)
    for name, value, dtype in (
        ("candidate_mask", mask, np.bool_),
        ("candidate_score", score, np.float32),
        ("candidate_level", level, np.uint8),
    ):
        if value.shape != (point_count,) or value.dtype != dtype:
            raise _error(
                "CANDIDATE_PREVIEW_INPUT", "candidate preview array is invalid", array=name
            )
    background_source = np.flatnonzero(~mask)
    if not len(background_source):
        background_source = np.arange(point_count, dtype=np.int64)
    background_index = _evenly_spaced_indices(background_source, max_background_points)
    candidate_source = np.flatnonzero(mask)
    candidate_index = _top_candidate_indices(candidate_source, score, max_candidate_points)

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise _error(
            "CANDIDATE_PREVIEW_DEPENDENCY", "Plotly is unavailable", cause=str(error)
        ) from error

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[background_index, 0],
            y=xyz[background_index, 1],
            z=xyz[background_index, 2],
            mode="markers",
            name="background",
            marker={"size": 1.4, "color": "#6b7280", "opacity": 0.16},
            hoverinfo="skip",
        )
    )
    for candidate_level, label, color in (
        (CandidateLevel.LOW, "low", "#f59e0b"),
        (CandidateLevel.MEDIUM, "medium", "#2563eb"),
        (CandidateLevel.HIGH, "high", "#dc2626"),
    ):
        index = candidate_index[level[candidate_index] == int(candidate_level)]
        if not len(index):
            continue
        figure.add_trace(
            go.Scatter3d(
                x=xyz[index, 0],
                y=xyz[index, 1],
                z=xyz[index, 2],
                mode="markers",
                name=f"{label} candidates",
                marker={"size": 2.6, "color": color, "opacity": 0.9},
                customdata=np.column_stack((index, score[index])),
                hovertemplate=(
                    "input order=%{customdata[0]}<br>score=%{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Scene Simple A3 Wire Candidates",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"itemsizing": "constant"},
    )
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(output_html_path)
    try:
        figure.write_html(temporary, include_plotlyjs="cdn", full_html=True)
        os.replace(temporary, output_html_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output_html_path


def _score_candidates(
    data: Mapping[str, Mapping[str, np.ndarray]],
    config: WireCandidateConfig,
    manual: ManualSelectionConfig,
    mode: str,
    cancellation_token: CancellationToken | None,
    process: Any | None,
    rss_start: int | None,
    rss_peak: int | None,
    pipeline_started: float,
) -> tuple[dict[str, np.ndarray], CandidatePointSet, CandidateEvidenceSet, WireCandidateStatistics]:
    a1, a2a, a2b = data["a1"], data["a2a"], data["a2b"]
    point_count = len(a1["xyz_local"])
    _check_cancelled(cancellation_token, "before scoring")
    linearity = a2a["linearity"]
    scattering = a2a["scattering"]
    planarity = a2a["planarity"]
    density = a2a["local_density"]
    height_m = a2a["height_m"]
    geometry_flags = a2a["quality_flags"]
    ground_flags = a2b["quality_flags"]
    height_agl = a2b["height_agl_m"]

    linearity_score = _clip(
        (linearity - config.linearity_guard) / (config.linearity_full - config.linearity_guard)
    )
    scattering_score = _clip(
        (config.scattering_guard - scattering) / (config.scattering_guard - config.scattering_good)
    )
    planarity_score = _clip(
        (config.planarity_bad - planarity) / (config.planarity_bad - config.planarity_good)
    )
    density_score = _density_score(density, config)
    local_height_score = _height_score(height_m, config)
    hag_score = _height_score(height_agl, config)

    invalid_ground = _has_flag(ground_flags, GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    high_gradient = _has_flag(ground_flags, GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION)
    below_ground = _has_flag(ground_flags, GroundQualityFlag.BELOW_GROUND_TOLERANCE)
    extrapolated = _has_flag(ground_flags, GroundQualityFlag.EXTRAPOLATION_TOO_FAR)
    extreme_hag = _has_flag(ground_flags, GroundQualityFlag.EXTREME_HEIGHT_AGL)
    hag_available = np.isfinite(height_agl) & ~invalid_ground
    local_height_available = np.isfinite(height_m)
    fallback = invalid_ground | below_ground | extrapolated | extreme_hag | ~hag_available
    mixed = high_gradient & ~fallback
    normal = ~mixed & ~fallback
    height_score = np.zeros(point_count, dtype=np.float32)
    height_score[normal & hag_available] = hag_score[normal & hag_available]
    both_available = mixed & hag_available & local_height_available
    height_score[both_available] = (
        config.high_gradient_hag_weight * hag_score[both_available]
        + (1.0 - config.high_gradient_hag_weight) * local_height_score[both_available]
    )
    hag_only = mixed & hag_available & ~local_height_available
    local_only = (mixed & ~hag_available & local_height_available) | (
        fallback & local_height_available
    )
    height_score[hag_only] = hag_score[hag_only]
    height_score[local_only] = local_height_score[local_only]
    height_score = _clip(height_score)

    numeric_invalid = _has_flag(geometry_flags, FeatureQualityFlag.NUMERIC_INVALID)
    principal_ambiguous = _has_flag(
        geometry_flags, FeatureQualityFlag.PRINCIPAL_DIRECTION_AMBIGUOUS
    )
    normal_ambiguous = _has_flag(geometry_flags, FeatureQualityFlag.NORMAL_DIRECTION_AMBIGUOUS)
    geometry_degraded = (geometry_flags != 0) & ~normal_ambiguous & ~numeric_invalid
    geometry_degraded |= principal_ambiguous
    extreme_density = (density <= config.density_zero_low) | (density >= config.density_zero_high)
    hag_unavailable = fallback & ~local_height_available

    quality_flags = np.zeros(point_count, dtype=np.uint16)
    quality_flags[geometry_degraded] |= int(WireCandidateQualityFlag.GEOMETRY_QUALITY_DEGRADED)
    quality_flags[high_gradient] |= int(WireCandidateQualityFlag.GROUND_QUALITY_DEGRADED)
    quality_flags[hag_unavailable] |= int(WireCandidateQualityFlag.HAG_UNAVAILABLE)
    quality_flags[below_ground] |= int(WireCandidateQualityFlag.BELOW_GROUND_INPUT)
    quality_flags[extreme_density] |= int(WireCandidateQualityFlag.EXTREME_DENSITY)
    quality_flags[principal_ambiguous] |= int(
        WireCandidateQualityFlag.AMBIGUOUS_PRINCIPAL_DIRECTION
    )
    quality_flags[numeric_invalid] |= int(WireCandidateQualityFlag.INVALID_INPUT_FEATURE)

    quality_penalty = np.zeros(point_count, dtype=np.float32)
    quality_penalty += geometry_degraded.astype(np.float32) * config.penalty_geometry
    quality_penalty += high_gradient.astype(np.float32) * config.penalty_ground_gradient
    quality_penalty += hag_unavailable.astype(np.float32) * config.penalty_hag_unavailable
    quality_penalty += below_ground.astype(np.float32) * config.penalty_below_ground
    quality_penalty += extreme_hag.astype(np.float32) * config.penalty_extreme_hag
    quality_penalty += principal_ambiguous.astype(np.float32) * config.penalty_principal_ambiguity
    quality_penalty += extreme_density.astype(np.float32) * config.penalty_extreme_density
    quality_penalty = _clip(np.minimum(quality_penalty, config.penalty_cap))

    geometry_guard = (
        (linearity >= config.linearity_guard)
        & (scattering <= config.scattering_guard)
        & ~numeric_invalid
    )
    weights = np.array(
        [
            config.weight_linearity,
            config.weight_scattering,
            config.weight_planarity,
            config.weight_height,
            config.weight_density,
        ],
        dtype=np.float32,
    )
    weighted_score = (
        weights[0] * linearity_score
        + weights[1] * scattering_score
        + weights[2] * planarity_score
        + weights[3] * height_score
        + weights[4] * density_score
    ) / weights.sum()
    rule_score = _clip(weighted_score * (1.0 - quality_penalty))
    rule_score[numeric_invalid] = 0.0
    rule_mask = geometry_guard & (rule_score >= config.rule_min_score)

    evidence_flags = np.zeros(point_count, dtype=np.uint16)
    evidence_flags[linearity >= config.evidence_linearity] |= int(
        CandidateReasonFlag.HIGH_LINEARITY
    )
    evidence_flags[scattering <= config.evidence_scattering] |= int(
        CandidateReasonFlag.LOW_SCATTERING
    )
    evidence_flags[planarity <= config.evidence_planarity] |= int(
        CandidateReasonFlag.ACCEPTABLE_PLANARITY
    )
    evidence_flags[height_score >= config.evidence_height_score] |= int(
        CandidateReasonFlag.SUFFICIENT_HEIGHT
    )
    evidence_flags[density_score >= config.evidence_density_score] |= int(
        CandidateReasonFlag.ACCEPTABLE_DENSITY
    )
    evidence_flags[mixed | fallback] |= int(CandidateReasonFlag.HEIGHT_FALLBACK_USED)

    include_mask, exclude_mask = _manual_masks(a1, manual)
    include_mask &= ~numeric_invalid | manual.allow_numeric_invalid_override
    evidence_flags[include_mask] |= int(CandidateReasonFlag.MANUAL_INCLUDE)
    evidence_flags[exclude_mask] |= int(CandidateReasonFlag.MANUAL_EXCLUDE)
    candidate_score = rule_score.copy()
    candidate_score[include_mask] = np.maximum(
        candidate_score[include_mask], manual.manual_include_score
    )
    candidate_score[exclude_mask] = 0.0
    candidate_mask = _select_mode(rule_mask, include_mask, exclude_mask, mode)
    candidate_level = _candidate_level(candidate_mask, candidate_score, config)
    quality_flags[candidate_mask & (candidate_score < config.medium_score)] |= int(
        WireCandidateQualityFlag.LOW_CONFIDENCE
    )
    candidate_source = _candidate_source(candidate_mask, rule_mask, include_mask)

    points = CandidatePointSet(
        source_id=_scalar_text(a1["source_id"], "A1 source_id"),
        manifest_source_id=_scalar_text(a1["manifest_source_id"], "A1 manifest_source_id"),
        xyz_local=a1["xyz_local"],
        origin_xyz=a1["origin_xyz"],
        original_index=a1["original_index"],
        sample_index=a1["sample_index"],
        principal_direction=a2a["principal_direction"],
        rule_score=rule_score,
        candidate_score=candidate_score,
        candidate_mask=candidate_mask,
        candidate_level=candidate_level,
        candidate_source=candidate_source,
        evidence_flags=evidence_flags,
        quality_flags=quality_flags,
        geometry_quality_flags=geometry_flags,
        ground_quality_flags=ground_flags,
    )
    evidence = CandidateEvidenceSet(
        linearity_score=linearity_score,
        scattering_score=scattering_score,
        planarity_score=planarity_score,
        height_score=height_score,
        density_score=density_score,
        quality_penalty=quality_penalty,
        geometry_guard_mask=geometry_guard,
    )
    arrays = _output_arrays(points, evidence)
    rss_peak = _max_rss(rss_peak, process)
    statistics = WireCandidateStatistics(
        input_point_count=point_count,
        candidate_count=int(candidate_mask.sum()),
        low_count=int(np.count_nonzero(candidate_level == int(CandidateLevel.LOW))),
        medium_count=int(np.count_nonzero(candidate_level == int(CandidateLevel.MEDIUM))),
        high_count=int(np.count_nonzero(candidate_level == int(CandidateLevel.HIGH))),
        rule_count=int(np.count_nonzero(candidate_source == int(CandidateSource.RULE))),
        manual_count=int(np.count_nonzero(candidate_source == int(CandidateSource.MANUAL))),
        hybrid_count=int(np.count_nonzero(candidate_source == int(CandidateSource.HYBRID))),
        hag_normal_count=int(normal.sum()),
        hag_mixed_count=int(mixed.sum()),
        hag_fallback_count=int(fallback.sum()),
        elapsed_seconds=time.perf_counter() - pipeline_started,
        rss_start_bytes=rss_start,
        rss_peak_bytes=rss_peak,
        rss_end_bytes=_rss_bytes(process),
    )
    _check_cancelled(cancellation_token, "after scoring")
    return arrays, points, evidence, statistics


def _write_outputs(
    arrays: Mapping[str, np.ndarray],
    points: CandidatePointSet,
    evidence: CandidateEvidenceSet,
    paths: Mapping[str, Path],
    input_hashes: Mapping[str, str],
    output_npz_path: Path,
    output_json_path: Path,
    config: WireCandidateConfig,
    manual_selection: ManualSelectionConfig,
    mode: str,
    statistics: WireCandidateStatistics,
    arrays_metadata: Mapping[str, Any],
    pipeline_started: float,
    process: Any | None,
) -> WireCandidateResult:
    temporary_npz = _temporary_path(output_npz_path)
    temporary_json = _temporary_path(output_json_path)
    published_npz = False
    try:
        np.savez_compressed(temporary_npz, **arrays)
        output_hash = _sha256_file(temporary_npz)
        statistics = replace(
            statistics,
            elapsed_seconds=time.perf_counter() - pipeline_started,
            rss_peak_bytes=_max_rss(statistics.rss_peak_bytes, process),
            rss_end_bytes=_rss_bytes(process),
        )
        result = WireCandidateResult(
            source_id=points.source_id,
            manifest_source_id=points.manifest_source_id,
            input_paths={name: str(path.resolve()) for name, path in paths.items()},
            input_sha256=dict(input_hashes),
            output_npz_path=output_npz_path,
            output_json_path=output_json_path,
            output_npz_sha256=output_hash,
            output_npz_bytes=temporary_npz.stat().st_size,
            origin_xyz=tuple(float(value) for value in points.origin_xyz),
            config=config,
            manual_selection=manual_selection,
            mode=mode,
            statistics=statistics,
            arrays=arrays_metadata,
            evidence_counts=_flag_counts(points.evidence_flags, CandidateReasonFlag),
            quality_counts=_flag_counts(points.quality_flags, WireCandidateQualityFlag),
            formulas=_FORMULAS,
            warnings=(),
            algorithm_version=ALGORITHM_VERSION,
            created_at=datetime.now(UTC).isoformat(),
        )
        temporary_json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_npz, output_npz_path)
        published_npz = True
        os.replace(temporary_json, output_json_path)
        return result
    except Exception:
        temporary_npz.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
        if published_npz:
            output_npz_path.unlink(missing_ok=True)
        raise


def _load_and_validate_inputs(
    paths: Mapping[str, Path], config: WireCandidateConfig
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Mapping[str, Any]], dict[str, str]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise _error("CANDIDATE_INPUT_MISSING", "Six A1/A2A/A2B inputs are required", paths=missing)
    metadata = {
        "a1": _load_json(paths["a1_json"]),
        "a2a": _load_json(paths["a2a_json"]),
        "a2b": _load_json(paths["a2b_json"]),
    }
    data = {
        "a1": _load_npz(paths["a1_npz"]),
        "a2a": _load_npz(paths["a2a_npz"]),
        "a2b": _load_npz(paths["a2b_npz"]),
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    _validate_provenance(paths, metadata, hashes, config)
    _validate_arrays(data)
    return data, metadata, hashes


def _validate_provenance(
    paths: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, Any]],
    hashes: Mapping[str, str],
    config: WireCandidateConfig,
) -> None:
    for stage, json_key in (("a1", "a1_json"), ("a2a", "a2a_json"), ("a2b", "a2b_json")):
        payload = metadata[stage]
        for key in (
            "source_id",
            "manifest_source_id",
            "origin_xyz",
            "output_npz_path",
            "output_json_path",
        ):
            if key not in payload:
                raise _error(
                    "CANDIDATE_METADATA_MISSING",
                    "required metadata is absent",
                    stage=stage,
                    key=key,
                )
        if Path(payload["output_npz_path"]).resolve() != paths[f"{stage}_npz"].resolve():
            raise _error(
                "CANDIDATE_METADATA_PATH", "metadata NPZ path does not match input", stage=stage
            )
        if Path(payload["output_json_path"]).resolve() != paths[json_key].resolve():
            raise _error(
                "CANDIDATE_METADATA_PATH", "metadata JSON path does not match input", stage=stage
            )
    source_ids = {metadata[stage]["source_id"] for stage in metadata}
    manifest_ids = {metadata[stage]["manifest_source_id"] for stage in metadata}
    origins = [np.asarray(metadata[stage]["origin_xyz"], dtype=np.float64) for stage in metadata]
    if source_ids != {config.source_id}:
        raise _error("CANDIDATE_INPUT_SOURCE", "source_id is inconsistent with configuration")
    if (
        len(manifest_ids) != 1
        or any(origin.shape != (3,) for origin in origins)
        or not all(np.array_equal(origin, origins[0]) for origin in origins[1:])
    ):
        raise _error(
            "CANDIDATE_INPUT_PROVENANCE", "manifest source or origin metadata is inconsistent"
        )
    required_hashes = {
        ("a2a", "input_a1_npz_sha256", "a1_npz"),
        ("a2a", "input_a1_json_sha256", "a1_json"),
        ("a2a", "output_npz_sha256", "a2a_npz"),
        ("a2b", "input_a1_npz_sha256", "a1_npz"),
        ("a2b", "input_a1_json_sha256", "a1_json"),
        ("a2b", "input_a2a_npz_sha256", "a2a_npz"),
        ("a2b", "input_a2a_json_sha256", "a2a_json"),
        ("a2b", "output_npz_sha256", "a2b_npz"),
    }
    for stage, field, file_key in required_hashes:
        if metadata[stage].get(field) != hashes[file_key]:
            raise _error(
                "CANDIDATE_INPUT_SHA",
                "input provenance hash is inconsistent",
                stage=stage,
                field=field,
            )


def _validate_arrays(data: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    a1, a2a, a2b = data["a1"], data["a2a"], data["a2b"]
    _require_array(a1, "xyz_local", None, np.float32, dimensions=2)
    point_count = len(a1["xyz_local"])
    _require_array(a1, "origin_xyz", (3,), np.float64)
    for name in ("original_index", "sample_index"):
        _require_array(a1, name, (point_count,), np.int64)
    for name in ("source_id", "manifest_source_id"):
        _require_scalar_text(a1, name)
    if not np.isfinite(a1["xyz_local"]).all() or not np.isfinite(a1["origin_xyz"]).all():
        raise _error("CANDIDATE_INPUT_NONFINITE", "A1 coordinates are not finite")
    for name in ("original_index", "sample_index"):
        values = a1[name]
        if np.any(values < 0) or len(np.unique(values)) != point_count:
            raise _error(
                "CANDIDATE_INPUT_INDEX", "A1 indexes must be unique and non-negative", array=name
            )
    for name in ("xyz_local", "origin_xyz", "original_index", "sample_index"):
        _require_array(a2a, name, a1[name].shape, a1[name].dtype)
        if not np.array_equal(a2a[name], a1[name]):
            raise _error("CANDIDATE_INPUT_ALIGNMENT", "A1/A2A point alignment differs", array=name)
    for name in ("source_id", "manifest_source_id"):
        _require_scalar_text(a2a, name)
        if _scalar_text(a2a[name], name) != _scalar_text(a1[name], name):
            raise _error("CANDIDATE_INPUT_SOURCE", "A1/A2A source metadata differs", array=name)
    for name in ("linearity", "scattering", "planarity", "local_density", "height_m"):
        _require_array(a2a, name, (point_count,), np.float32)
        if not np.isfinite(a2a[name]).all():
            raise _error("CANDIDATE_INPUT_NONFINITE", "A2A feature is not finite", array=name)
    _require_array(a2a, "principal_direction", (point_count, 3), np.float32)
    _require_array(a2a, "quality_flags", (point_count,), np.uint16)
    if not np.isfinite(a2a["principal_direction"]).all():
        raise _error("CANDIDATE_INPUT_NONFINITE", "A2A principal direction is not finite")
    for name in ("original_index", "sample_index"):
        _require_array(a2b, name, (point_count,), np.int64)
        if not np.array_equal(a2b[name], a1[name]):
            raise _error("CANDIDATE_INPUT_ALIGNMENT", "A1/A2B index alignment differs", array=name)
    for name in ("source_id", "manifest_source_id"):
        _require_scalar_text(a2b, name)
        if _scalar_text(a2b[name], name) != _scalar_text(a1[name], name):
            raise _error("CANDIDATE_INPUT_SOURCE", "A1/A2B source metadata differs", array=name)
    for name in ("height_agl_m", "raw_height_agl_m"):
        _require_array(a2b, name, (point_count,), np.float32)
    _require_array(a2b, "quality_flags", (point_count,), np.uint16)
    valid_ground = ~_has_flag(a2b["quality_flags"], GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    for name in ("height_agl_m", "raw_height_agl_m"):
        if not np.isfinite(a2b[name][valid_ground]).all():
            raise _error("CANDIDATE_INPUT_NONFINITE", "valid A2B HAG is not finite", array=name)


def _manual_masks(
    a1: Mapping[str, np.ndarray], manual: ManualSelectionConfig
) -> tuple[np.ndarray, np.ndarray]:
    original_index = a1["original_index"]
    configured = (*manual.include_original_index, *manual.exclude_original_index)
    missing = [int(value) for value in configured if not np.any(original_index == value)]
    if missing:
        raise _error(
            "CANDIDATE_MANUAL_INDEX", "manual original_index is not in A1", indexes=missing
        )
    include = np.isin(original_index, np.asarray(manual.include_original_index, dtype=np.int64))
    for lower, upper in manual.include_boxes:
        include |= np.all((a1["xyz_local"] >= lower) & (a1["xyz_local"] <= upper), axis=1)
    exclude = np.isin(original_index, np.asarray(manual.exclude_original_index, dtype=np.int64))
    return include, exclude


def _select_mode(
    rule_mask: np.ndarray, include_mask: np.ndarray, exclude_mask: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "rule":
        candidate_mask = rule_mask.copy()
    elif mode == "union":
        candidate_mask = rule_mask | include_mask
    elif mode == "intersection":
        candidate_mask = rule_mask & include_mask
    else:
        candidate_mask = include_mask.copy()
    candidate_mask &= ~exclude_mask
    return candidate_mask


def _candidate_level(
    candidate_mask: np.ndarray, candidate_score: np.ndarray, config: WireCandidateConfig
) -> np.ndarray:
    level = np.zeros(len(candidate_mask), dtype=np.uint8)
    level[candidate_mask & (candidate_score < config.medium_score)] = int(CandidateLevel.LOW)
    level[
        candidate_mask
        & (candidate_score >= config.medium_score)
        & (candidate_score < config.high_score)
    ] = int(CandidateLevel.MEDIUM)
    level[candidate_mask & (candidate_score >= config.high_score)] = int(CandidateLevel.HIGH)
    return level


def _candidate_source(
    candidate_mask: np.ndarray, rule_mask: np.ndarray, include_mask: np.ndarray
) -> np.ndarray:
    source = np.zeros(len(candidate_mask), dtype=np.uint8)
    source[candidate_mask & rule_mask & ~include_mask] = int(CandidateSource.RULE)
    source[candidate_mask & ~rule_mask & include_mask] = int(CandidateSource.MANUAL)
    source[candidate_mask & rule_mask & include_mask] = int(CandidateSource.HYBRID)
    return source


def _output_arrays(
    points: CandidatePointSet, evidence: CandidateEvidenceSet
) -> dict[str, np.ndarray]:
    return {
        "source_id": np.asarray(points.source_id),
        "manifest_source_id": np.asarray(points.manifest_source_id),
        "xyz_local": points.xyz_local,
        "origin_xyz": points.origin_xyz,
        "original_index": points.original_index,
        "sample_index": points.sample_index,
        "principal_direction": points.principal_direction,
        "rule_score": points.rule_score,
        "candidate_score": points.candidate_score,
        "candidate_mask": points.candidate_mask,
        "candidate_level": points.candidate_level,
        "candidate_source": points.candidate_source,
        "evidence_flags": points.evidence_flags,
        "quality_flags": points.quality_flags,
        "geometry_quality_flags": points.geometry_quality_flags,
        "ground_quality_flags": points.ground_quality_flags,
        "linearity_score": evidence.linearity_score,
        "scattering_score": evidence.scattering_score,
        "planarity_score": evidence.planarity_score,
        "height_score": evidence.height_score,
        "density_score": evidence.density_score,
        "quality_penalty": evidence.quality_penalty,
        "geometry_guard_mask": evidence.geometry_guard_mask,
    }


def _density_score(density: np.ndarray, config: WireCandidateConfig) -> np.ndarray:
    return np.where(
        density <= config.density_zero_low,
        0.0,
        np.where(
            density < config.density_full_low,
            (density - config.density_zero_low)
            / (config.density_full_low - config.density_zero_low),
            np.where(
                density <= config.density_full_high,
                1.0,
                np.where(
                    density < config.density_zero_high,
                    (config.density_zero_high - density)
                    / (config.density_zero_high - config.density_full_high),
                    0.0,
                ),
            ),
        ),
    ).astype(np.float32)


def _height_score(height: np.ndarray, config: WireCandidateConfig) -> np.ndarray:
    return _clip((height - config.height_low_m) / (config.height_full_m - config.height_low_m))


def _clip(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip(values, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def _has_flag(values: np.ndarray, flag: int) -> np.ndarray:
    return (values & int(flag)) != 0


def _flag_counts(values: np.ndarray, flag_type: type[Any]) -> dict[str, int]:
    return {
        member.name: int(np.count_nonzero(values & int(member)))
        for member in flag_type
        if int(member) != 0
    }


def _input_paths(
    a1_npz_path: Path | str,
    a1_json_path: Path | str,
    a2a_npz_path: Path | str,
    a2a_json_path: Path | str,
    a2b_npz_path: Path | str,
    a2b_json_path: Path | str,
) -> dict[str, Path]:
    return {
        "a1_npz": Path(a1_npz_path).resolve(),
        "a1_json": Path(a1_json_path).resolve(),
        "a2a_npz": Path(a2a_npz_path).resolve(),
        "a2a_json": Path(a2a_json_path).resolve(),
        "a2b_npz": Path(a2b_npz_path).resolve(),
        "a2b_json": Path(a2b_json_path).resolve(),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stream:
            return {name: stream[name] for name in stream.files}
    except (OSError, ValueError) as error:
        raise _error(
            "CANDIDATE_INPUT_NPZ", "unable to load NPZ input", path=str(path), cause=str(error)
        ) from error


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _error(
            "CANDIDATE_INPUT_JSON", "unable to load JSON input", path=str(path), cause=str(error)
        ) from error
    if not isinstance(data, Mapping):
        raise _error("CANDIDATE_INPUT_JSON", "input JSON must contain an object", path=str(path))
    return data


def _require_array(
    data: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...] | None,
    dtype: type[np.generic[Any]],
    *,
    dimensions: int | None = None,
) -> None:
    value = data.get(name)
    if value is None or value.dtype != np.dtype(dtype):
        raise _error(
            "CANDIDATE_INPUT_SHAPE", "input array has invalid dtype or is missing", array=name
        )
    if shape is not None and value.shape != shape:
        raise _error("CANDIDATE_INPUT_SHAPE", "input array has invalid shape", array=name)
    if dimensions is not None and value.ndim != dimensions:
        raise _error("CANDIDATE_INPUT_SHAPE", "input array has invalid dimensions", array=name)


def _require_scalar_text(data: Mapping[str, np.ndarray], name: str) -> None:
    if name not in data:
        raise _error("CANDIDATE_INPUT_SCALAR", "input scalar metadata is missing", array=name)
    _scalar_text(data[name], name)


def _scalar_text(value: np.ndarray, name: str) -> str:
    if value.shape != ():
        raise _error(
            "CANDIDATE_INPUT_SCALAR", "input scalar metadata has invalid shape", array=name
        )
    text = str(value.item())
    if not text:
        raise _error(
            "CANDIDATE_INPUT_SCALAR", "input scalar metadata must be non-empty", array=name
        )
    return text


def _validate_mode(mode: str, manual: ManualSelectionConfig) -> None:
    if mode not in {"rule", "union", "intersection", "manual_override"}:
        raise _error("CANDIDATE_MODE_INVALID", "candidate mode is invalid", mode=mode)
    if mode == "rule" and (manual.has_include or manual.exclude_original_index):
        raise _error("CANDIDATE_MANUAL_FOR_RULE", "rule mode requires empty manual selection")
    if mode in {"intersection", "manual_override"} and not manual.has_include:
        raise _error("CANDIDATE_MANUAL_INCLUDE_REQUIRED", "selected mode requires manual include")


def _validate_output_dir(output_dir: Path, allowed_root: Path) -> None:
    try:
        output_dir.relative_to(allowed_root.resolve())
    except ValueError as error:
        raise _error(
            "CANDIDATE_OUTPUT_FORBIDDEN",
            "output must be under the approved output directory",
            output_dir=str(output_dir),
        ) from error


def _validate_outputs_absent(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _error(
            "CANDIDATE_OUTPUT_EXISTS",
            "A3 output already exists and overwrite is disabled",
            paths=existing,
        )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_cancelled(token: CancellationToken | None, context: str) -> None:
    if token is not None and token.is_cancelled():
        raise _error(
            "CANDIDATE_CANCELLED",
            f"A3 candidate selection cancelled {context}",
            context=context,
        )


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
            task_id=f"{source_id}:wire-candidates",
            stage=stage,
            status=status,  # type: ignore[arg-type]
            processed_points=processed_points,
            total_points=total_points,
            percent=100.0 if total_points == 0 else 100.0 * processed_points / total_points,
            message=f"{stage}: {processed_points} of {total_points} points",
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


def _evenly_spaced_indices(index: np.ndarray, limit: int) -> np.ndarray:
    if len(index) <= limit:
        return index
    return index[np.linspace(0, len(index) - 1, limit, dtype=np.int64)]


def _top_candidate_indices(index: np.ndarray, score: np.ndarray, limit: int) -> np.ndarray:
    if len(index) <= limit:
        return index
    order = np.lexsort((index, -score[index]))
    return index[order[:limit]]


def _error(code: str, message: str, **details: Any) -> WireCandidateError:
    cause = details.pop("cause", None)
    return WireCandidateError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="algorithms.candidates.wire",
            recoverable=True,
            details=details,
            cause=cause,
        )
    )
