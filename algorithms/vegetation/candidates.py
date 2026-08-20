# ruff: noqa: E501
"""V1 full-M explainable woody-vegetation candidate selection."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import (
    CancellationToken,
    FeatureQualityFlag,
    GroundQualityFlag,
    PipelineProgress,
    ProgressReporter,
    VegetationCandidateConfig,
    VegetationCandidateLevel,
    VegetationCandidatePointSet,
    VegetationCandidateResult,
    VegetationCandidateStatistics,
    VegetationEvidenceFlag,
    VegetationHagState,
    VegetationManualAction,
    VegetationManualProvenanceFlag,
    VegetationManualSelectionConfig,
    VegetationQualityFlag,
    VegetationReasonFlag,
)
from core.contracts.errors import UnifiedError
from core.contracts.wires import WireSeparationResult

SCHEMA_VERSION = "vegetation-candidates-v1"
ALGORITHM_VERSION = "v1-explainable-woody-candidates-v1"
_BASE = "{source_id}_vegetation_candidates"


class VegetationCandidateError(RuntimeError):
    def __init__(self, error: UnifiedError):
        super().__init__(error.message)
        self.error = error


def run_vegetation_candidate_selection(
    a1_npz_path: Path | str,
    a1_json_path: Path | str,
    a2a_npz_path: Path | str,
    a2a_json_path: Path | str,
    a2b_npz_path: Path | str,
    a2b_json_path: Path | str,
    a4_npz_path: Path | str,
    a4_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: VegetationCandidateConfig | None = None,
    manual_selection: VegetationManualSelectionConfig | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
    generate_preview: bool = True,
) -> VegetationCandidateResult:
    """Score explainable V1 vegetation candidates from verified stage artifacts."""
    cfg = config or VegetationCandidateConfig()
    manual = manual_selection or VegetationManualSelectionConfig()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output = Path(output_dir).resolve()
    _validate_output_dir(output, root / "outputs" / "vegetation_candidates")
    npz_out = output / f"{_BASE.format(source_id=cfg.source_id)}.npz"
    json_out = output / f"{_BASE.format(source_id=cfg.source_id)}.json"
    _validate_outputs((npz_out, json_out), overwrite)
    _check_cancelled(cancellation_token, "before input validation")
    started = time.perf_counter()
    process = _current_process() if cfg.record_memory else None
    rss_start = _rss(process)
    paths = _paths(
        a1_npz_path,
        a1_json_path,
        a2a_npz_path,
        a2a_json_path,
        a2b_npz_path,
        a2b_json_path,
        a4_npz_path,
        a4_json_path,
    )
    data, metadata, hashes = _load_and_validate_inputs(paths, cfg)
    count = len(data["a1"]["original_index"])
    _report(reporter, cfg.source_id, "validated_inputs", 0, count)
    _check_cancelled(cancellation_token, "before scoring")
    arrays, points, statistics, counts, formulas = _score(
        data, cfg, manual, started, cancellation_token, process, rss_start
    )
    _report(reporter, cfg.source_id, "scored_candidates", count, count)
    _check_cancelled(cancellation_token, "before output publication")
    output.mkdir(parents=True, exist_ok=True)
    result = _publish(
        arrays,
        points,
        npz_out,
        json_out,
        cfg,
        manual,
        paths,
        hashes,
        statistics,
        counts,
        formulas,
        root,
    )
    if generate_preview:
        try:
            generate_vegetation_candidate_preview(
                npz_out,
                json_out,
                root
                / "outputs"
                / "previews"
                / f"{_BASE.format(source_id=cfg.source_id)}_preview.html",
                overwrite=overwrite,
                project_root=root,
            )
        except Exception as error:  # core artifact remains valid if optional visualisation fails
            result = replace(result, warnings=(f"preview generation failed: {error}",))
    _report(reporter, cfg.source_id, "completed", count, count, "completed")
    return result


def load_vegetation_candidates(
    model_npz_path: Path | str | None = None, model_json_path: Path | str | None = None
) -> tuple[VegetationCandidateResult, dict[str, np.ndarray]]:
    """Load a relocated V1 NPZ/JSON sibling pair without upstream artifacts."""
    if model_json_path is None:
        if model_npz_path is None:
            raise _error("VEGETATION_LOAD_PATH", "NPZ and JSON paths, or a JSON path, are required")
        json_path = Path(model_npz_path)
        if json_path.suffix.lower() != ".json":
            raise _error("VEGETATION_LOAD_PATH", "single-path loading requires a JSON manifest")
        manifest = _load_json(json_path)
        filename = manifest.get("output_npz_filename")
        if not isinstance(filename, str):
            raise _error("VEGETATION_LOAD_MANIFEST", "manifest lacks output NPZ filename")
        npz_path = json_path.with_name(filename)
    else:
        npz_path, json_path = Path(model_npz_path), Path(model_json_path)
    if not npz_path.is_file() or not json_path.is_file():
        raise _error("VEGETATION_LOAD_MISSING", "V1 NPZ and JSON are both required")
    result = VegetationCandidateResult.from_dict(_load_json(json_path))
    if npz_path.name != result.output_npz_filename:
        raise _error(
            "VEGETATION_LOAD_FILENAME", "NPZ filename differs from manifest sibling reference"
        )
    if _sha(npz_path) != result.output_npz_sha256:
        raise _error("VEGETATION_LOAD_SHA", "NPZ SHA-256 differs from manifest")
    arrays = _load_npz(npz_path)
    if result.schema_version != SCHEMA_VERSION:
        raise _error("VEGETATION_LOAD_SCHEMA", "unsupported V1 schema")
    _validate_output_arrays(arrays)
    return result, arrays


def generate_vegetation_candidate_preview(
    model_npz_path: Path | str,
    model_json_path: Path | str,
    output_html_path: Path | str,
    *,
    w1_npz_path: Path | str | None = None,
    w1_json_path: Path | str | None = None,
    overwrite: bool = False,
    project_root: Path | str | None = None,
    background_limit: int = 20_000,
    candidate_limit: int = 20_000,
    warning_limit: int = 5_000,
) -> Path:
    """Write a standalone Plotly V1 review preview without LAS/A3 access."""
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output = Path(output_html_path).resolve()
    _validate_output_dir(output.parent, root / "outputs" / "previews")
    if output.exists() and not overwrite:
        raise _error("VEGETATION_PREVIEW_EXISTS", "preview exists and overwrite is disabled")
    result, arrays = load_vegetation_candidates(model_npz_path, model_json_path)
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise _error(
            "VEGETATION_PREVIEW_DEPENDENCY", "Plotly is unavailable", cause=str(error)
        ) from error
    xyz = arrays["xyz_local"]
    count = len(xyz)
    background = (
        np.linspace(0, count - 1, min(count, background_limit), dtype=np.int64)
        if count
        else np.empty(0, dtype=np.int64)
    )
    levels = arrays["vegetation_level"]
    hover = _hover_text(arrays)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[background, 0],
            y=xyz[background, 1],
            z=xyz[background, 2],
            mode="markers",
            name="base cloud",
            marker={"size": 1, "color": "#94a3b8", "opacity": 0.18},
            hoverinfo="skip",
        )
    )
    colors = {
        VegetationCandidateLevel.LOW: "#f59e0b",
        VegetationCandidateLevel.MEDIUM: "#22c55e",
        VegetationCandidateLevel.HIGH: "#047857",
    }
    candidate_count = int(np.count_nonzero(levels != int(VegetationCandidateLevel.REJECTED)))
    for level in (
        VegetationCandidateLevel.LOW,
        VegetationCandidateLevel.MEDIUM,
        VegetationCandidateLevel.HIGH,
    ):
        selected = np.flatnonzero(levels == int(level))
        if candidate_count > candidate_limit and len(selected):
            limit = max(1, round(candidate_limit * len(selected) / candidate_count))
            selected = selected[np.linspace(0, len(selected) - 1, limit, dtype=np.int64)]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[selected, 0],
                y=xyz[selected, 1],
                z=xyz[selected, 2],
                mode="markers",
                name=f"vegetation {level.name.lower()}",
                marker={"size": 2.4, "color": colors[level], "opacity": 0.82},
                text=hover[selected],
                hovertemplate="%{text}<extra></extra>",
                visible="legendonly" if level is VegetationCandidateLevel.LOW else True,
            )
        )
    wire = np.flatnonzero(arrays["wire_id_by_point"] >= 0)
    figure.add_trace(
        go.Scatter3d(
            x=xyz[wire, 0],
            y=xyz[wire, 1],
            z=xyz[wire, 2],
            mode="markers",
            name="confirmed wires",
            marker={"size": 2.6, "color": "#dc2626", "opacity": 0.95},
            text=hover[wire],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    warning = np.flatnonzero(
        (arrays["vegetation_quality_flags"] != 0) | (arrays["vegetation_reason_flags"] != 0)
    )
    if len(warning) > warning_limit:
        warning = warning[np.linspace(0, len(warning) - 1, warning_limit, dtype=np.int64)]
    figure.add_trace(
        go.Scatter3d(
            x=xyz[warning, 0],
            y=xyz[warning, 1],
            z=xyz[warning, 2],
            mode="markers",
            name="HAG / quality warnings",
            marker={"size": 2.0, "color": "#a855f7", "symbol": "x"},
            text=hover[warning],
            hovertemplate="%{text}<extra></extra>",
            visible="legendonly",
        )
    )
    manual = np.flatnonzero(arrays["manual_provenance_flags"] != 0)
    figure.add_trace(
        go.Scatter3d(
            x=xyz[manual, 0],
            y=xyz[manual, 1],
            z=xyz[manual, 2],
            mode="markers",
            name="manual actions",
            marker={"size": 3.0, "color": "#0f172a", "symbol": "diamond"},
            text=hover[manual],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    if (w1_npz_path is None) != (w1_json_path is None):
        raise _error("VEGETATION_PREVIEW_W1", "W1 NPZ and JSON must be supplied together")
    if w1_npz_path is not None:
        from algorithms.wires.reconstruct import load_wire_models

        _w1_result, w1 = load_wire_models(w1_npz_path, w1_json_path)
        for pos in range(len(w1["segment_wire_id"])):
            if not bool(w1["segment_distance_ready"][pos]):
                continue
            start, end = w1["sampled_curve_offsets"][pos : pos + 2]
            curve = w1["sampled_curve_xyz"][start:end]
            figure.add_trace(
                go.Scatter3d(
                    x=curve[:, 0],
                    y=curve[:, 1],
                    z=curve[:, 2],
                    mode="lines",
                    name=f"W1 wire {int(w1['segment_wire_id'][pos])}",
                    line={"color": "#7f1d1d", "width": 5},
                    visible="legendonly",
                )
            )
    figure.update_layout(
        title=f"V1 explainable vegetation candidates — {result.source_id}",
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


def _paths(*values: Path | str) -> dict[str, Path]:
    names = ("a1_npz", "a1_json", "a2a_npz", "a2a_json", "a2b_npz", "a2b_json", "a4_npz", "a4_json")
    return {name: Path(value).resolve() for name, value in zip(names, values, strict=True)}


def _load_and_validate_inputs(
    paths: Mapping[str, Path], cfg: VegetationCandidateConfig
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Mapping[str, Any]], dict[str, str]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise _error(
            "VEGETATION_INPUT_MISSING", "mandatory A1/A2A/A2B/A4 inputs are required", paths=missing
        )
    metadata = {stage: _load_json(paths[f"{stage}_json"]) for stage in ("a1", "a2a", "a2b", "a4")}
    data = {stage: _load_npz(paths[f"{stage}_npz"]) for stage in ("a1", "a2a", "a2b", "a4")}
    hashes = {name: _sha(path) for name, path in paths.items()}
    _validate_provenance(paths, metadata, data, hashes, cfg)
    _validate_inputs(data)
    return data, metadata, hashes


def _validate_provenance(
    paths: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, Any]],
    data: Mapping[str, Mapping[str, np.ndarray]],
    hashes: Mapping[str, str],
    cfg: VegetationCandidateConfig,
) -> None:
    for stage in ("a1", "a2a", "a2b"):
        required = {
            "source_id",
            "manifest_source_id",
            "origin_xyz",
            "output_npz_path",
            "output_json_path",
        }
        if not required.issubset(metadata[stage]):
            raise _error(
                "VEGETATION_INPUT_METADATA", "stage metadata lacks required fields", stage=stage
            )
        if (
            Path(str(metadata[stage]["output_npz_path"])).resolve() != paths[f"{stage}_npz"]
            or Path(str(metadata[stage]["output_json_path"])).resolve() != paths[f"{stage}_json"]
        ):
            raise _error(
                "VEGETATION_INPUT_PATH",
                "stage metadata path does not match supplied input",
                stage=stage,
            )
    try:
        a4_result = WireSeparationResult.from_dict(metadata["a4"])
    except (ValueError, KeyError) as error:
        raise _error(
            "VEGETATION_INPUT_A4_CONTRACT", "A4 JSON contract is invalid", cause=str(error)
        ) from error
    if a4_result.output_npz_sha256 != hashes["a4_npz"] or a4_result.source_id != cfg.source_id:
        raise _error("VEGETATION_INPUT_A4_HASH", "A4 JSON does not identify supplied A4 NPZ")
    source_ids = {str(metadata[name].get("source_id", "")) for name in metadata}
    manifests = {str(metadata[name].get("manifest_source_id", "")) for name in metadata}
    if source_ids != {cfg.source_id} or len(manifests) != 1 or "" in manifests:
        raise _error("VEGETATION_INPUT_SOURCE", "source or manifest identifiers are inconsistent")
    origins = [
        np.asarray(metadata[name]["origin_xyz"], dtype=np.float64) for name in ("a1", "a2a", "a2b")
    ]
    if any(value.shape != (3,) for value in origins) or not all(
        np.array_equal(origins[0], value) for value in origins[1:]
    ):
        raise _error("VEGETATION_INPUT_ORIGIN", "stage origins are inconsistent")
    expected = (
        ("a2a", "input_a1_npz_sha256", "a1_npz"),
        ("a2a", "input_a1_json_sha256", "a1_json"),
        ("a2a", "output_npz_sha256", "a2a_npz"),
        ("a2b", "input_a1_npz_sha256", "a1_npz"),
        ("a2b", "input_a1_json_sha256", "a1_json"),
        ("a2b", "input_a2a_npz_sha256", "a2a_npz"),
        ("a2b", "input_a2a_json_sha256", "a2a_json"),
        ("a2b", "output_npz_sha256", "a2b_npz"),
    )
    for stage, field, path_key in expected:
        if metadata[stage].get(field) != hashes[path_key]:
            raise _error(
                "VEGETATION_INPUT_HASH",
                "upstream provenance hash is inconsistent",
                stage=stage,
                field=field,
            )
    for stage, prefix in (
        ("a2a", "a2a-local-pca-"),
        ("a2b", "a2b-grid-quantile-"),
        ("a4", "a4-sliced-hungarian-"),
    ):
        version = metadata[stage].get("algorithm_version")
        if not isinstance(version, str) or not version.startswith(prefix):
            raise _error(
                "VEGETATION_INPUT_VERSION", "unsupported stage algorithm metadata", stage=stage
            )
    for name in ("source_id", "manifest_source_id"):
        if name not in data["a1"] or str(data["a1"][name].item()) != str(metadata["a1"][name]):
            raise _error(
                "VEGETATION_INPUT_SCALAR", "A1 scalar metadata differs from JSON", array=name
            )
    if str(data["a4"].get("source_id", np.asarray("")).item()) != cfg.source_id:
        raise _error("VEGETATION_INPUT_A4_SOURCE", "A4 NPZ source identifier is inconsistent")


def _require(
    data: Mapping[str, np.ndarray], name: str, shape: tuple[int | None, ...], dtype: Any
) -> np.ndarray:
    value = data.get(name)
    if (
        value is None
        or value.dtype != np.dtype(dtype)
        or value.ndim != len(shape)
        or any(
            expected is not None and expected != actual
            for expected, actual in zip(shape, value.shape, strict=True)
        )
    ):
        raise _error("VEGETATION_INPUT_ARRAY", "array shape or dtype is invalid", array=name)
    return value


def _validate_inputs(data: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    a1, a2a, a2b, a4 = (data[name] for name in ("a1", "a2a", "a2b", "a4"))
    xyz = _require(a1, "xyz_local", (None, 3), np.float32)
    count = len(xyz)
    _require(a1, "origin_xyz", (3,), np.float64)
    for name in ("original_index", "sample_index"):
        value = _require(a1, name, (count,), np.int64)
        if np.any(value < 0) or len(np.unique(value)) != count:
            raise _error(
                "VEGETATION_INPUT_MAPPING",
                "A1 mappings must be unique and non-negative",
                array=name,
            )
    if not np.isfinite(xyz).all() or not np.isfinite(a1["origin_xyz"]).all():
        raise _error("VEGETATION_INPUT_FINITE", "A1 coordinates must be finite")
    for stage in (a2a, a2b, a4):
        for name in ("original_index", "sample_index"):
            value = _require(stage, name, (count,), np.int64)
            if not np.array_equal(value, a1[name]):
                raise _error(
                    "VEGETATION_INPUT_ALIGNMENT", "stable mappings differ across stages", array=name
                )
    for stage in (a2a, a4):
        if not np.array_equal(_require(stage, "xyz_local", (count, 3), np.float32), xyz):
            raise _error("VEGETATION_INPUT_ALIGNMENT", "local coordinates differ across stages")
    for name in (
        "linearity",
        "planarity",
        "scattering",
        "verticality",
        "local_density",
        "height_m",
    ):
        value = _require(a2a, name, (count,), np.float32)
        if not np.isfinite(value).all():
            raise _error("VEGETATION_INPUT_FINITE", "A2A feature must be finite", array=name)
    _require(a2a, "neighbor_count", (count,), np.uint16)
    _require(a2a, "quality_flags", (count,), np.uint16)
    for name in ("ground_z_m", "raw_height_agl_m", "height_agl_m", "ground_distance_m"):
        _require(a2b, name, (count,), np.float32)
    _require(a2b, "quality_flags", (count,), np.uint16)
    for name, dtype in (("wire_id_by_point", np.int32), ("track_id_by_point", np.int32)):
        value = _require(a4, name, (count,), dtype)
        if np.any(value < -1):
            raise _error("VEGETATION_INPUT_A4_IDS", "A4 IDs must be -1 or non-negative", array=name)
    for name, dtype, shape in (("rgb", np.uint16, (count, 3)), ("intensity", np.uint16, (count,))):
        if name in a1:
            _require(a1, name, shape, dtype)


def _score(
    data: Mapping[str, Mapping[str, np.ndarray]],
    cfg: VegetationCandidateConfig,
    manual: VegetationManualSelectionConfig,
    started: float,
    token: CancellationToken | None,
    process: Any | None = None,
    rss_start: int | None = None,
) -> tuple[
    dict[str, np.ndarray],
    VegetationCandidatePointSet,
    VegetationCandidateStatistics,
    dict[str, Any],
    dict[str, Any],
]:
    a1, a2a, a2b, a4 = (data[name] for name in ("a1", "a2a", "a2b", "a4"))
    count = len(a1["original_index"])
    _check_cancelled(token, "during score setup")
    geometry_flags, ground_flags = a2a["quality_flags"], a2b["quality_flags"]
    geo_invalid = _has(geometry_flags, FeatureQualityFlag.NUMERIC_INVALID)
    geometry_degraded = (geometry_flags != 0) & ~geo_invalid
    density_unusable = (
        _has(geometry_flags, FeatureQualityFlag.DENSITY_UNDEFINED)
        | _has(geometry_flags, FeatureQualityFlag.INSUFFICIENT_NEIGHBORS)
        | (a2a["neighbor_count"] < 8)
        | ~np.isfinite(a2a["local_density"])
        | (a2a["local_density"] < 0)
    )
    hag_state, _fallback_state, hag_uncertain, hag_invalid, strong_negative = _hag_states(a2b, cfg)
    structural_invalid = geo_invalid | ~np.isfinite(a1["xyz_local"]).all(axis=1)
    confirmed_wire = a4["wire_id_by_point"] >= 0
    reliable_near_ground = (hag_state == int(VegetationHagState.RELIABLE)) & (
        a2b["height_agl_m"] <= cfg.near_ground_hag_m
    )
    hard = structural_invalid | confirmed_wire | reliable_near_ground
    hag, fallback_used, fallback_degenerate = _hag_evidence(
        a2a["height_m"], a2b["height_agl_m"], hag_state, structural_invalid, cfg
    )
    volumetric = _clip(
        (a2a["scattering"] - cfg.scattering_low) / (cfg.scattering_high - cfg.scattering_low)
    )
    planarity = _clip(
        (a2a["planarity"] - cfg.planarity_low) / (cfg.planarity_high - cfg.planarity_low)
    )
    linearity = _clip(
        (a2a["linearity"] - cfg.linearity_low) / (cfg.linearity_high - cfg.linearity_low)
    ) * (1.0 - volumetric)
    density = _density_evidence(a2a["local_density"], structural_invalid | density_unusable)
    auxiliary = np.zeros(count, dtype=np.float32)
    denominator = (
        cfg.height_weight + cfg.volumetric_weight + cfg.density_weight + cfg.auxiliary_weight
    )
    evidence = (
        cfg.height_weight * hag
        + cfg.volumetric_weight * volumetric
        + cfg.density_weight * density
        + cfg.auxiliary_weight * auxiliary
    ) / denominator
    penalty = np.minimum(
        cfg.maximum_penalty,
        cfg.planarity_penalty_weight * planarity + cfg.linearity_penalty_weight * linearity,
    ).astype(np.float32)
    score = _clip(evidence - penalty)
    geometry_confidence = np.where(geometry_degraded, 0.60, 1.00).astype(np.float32)
    hag_confidence = np.select(
        [
            hag_state == int(VegetationHagState.RELIABLE),
            hag_state == int(VegetationHagState.UNCERTAIN),
            hag_state == int(VegetationHagState.STRONGLY_NEGATIVE),
            fallback_used,
        ],
        [1.00, 0.55, 0.40, 0.30],
        default=0.15,
    ).astype(np.float32)
    ground_distance = a2b["ground_distance_m"]
    ground_support = np.where(
        np.isfinite(ground_distance),
        _clip((cfg.ground_support_distance_m - ground_distance) / cfg.ground_support_distance_m),
        0.0,
    ).astype(np.float32)
    density_confidence = np.where(
        density_unusable, 0.30, np.where((a2a["neighbor_count"] < 16) | (density <= 0.0), 0.60, 1.0)
    ).astype(np.float32)
    confidence = _clip(
        cfg.geometry_confidence_weight * geometry_confidence
        + cfg.hag_confidence_weight * hag_confidence
        + cfg.ground_support_confidence_weight * ground_support
        + cfg.density_confidence_weight * density_confidence
    )
    automatic = _levels(score, confidence, cfg)
    automatic[hard] = int(VegetationCandidateLevel.REJECTED)
    provenance, action = _manual_arrays(a1, manual, confirmed_wire)
    final = automatic.copy()
    excluded = (provenance & int(VegetationManualProvenanceFlag.MANUAL_EXCLUDE_MATCHED)) != 0
    include = (provenance & int(VegetationManualProvenanceFlag.MANUAL_INCLUDE_MATCHED)) != 0
    final[excluded] = int(VegetationCandidateLevel.REJECTED)
    final[include & ~excluded & ~confirmed_wire & ~structural_invalid] = np.maximum(
        final[include & ~excluded & ~confirmed_wire & ~structural_invalid],
        int(VegetationCandidateLevel.LOW),
    )
    final[hard] = int(VegetationCandidateLevel.REJECTED)
    mask = final != int(VegetationCandidateLevel.REJECTED)
    evidence_flags = np.zeros(count, dtype=np.uint16)
    evidence_flags[hag > 0] |= int(VegetationEvidenceFlag.HEIGHT_EVIDENCE)
    evidence_flags[fallback_used] |= int(VegetationEvidenceFlag.HEIGHT_FALLBACK_USED)
    evidence_flags[volumetric > 0] |= int(VegetationEvidenceFlag.VOLUMETRIC_EVIDENCE)
    evidence_flags[density > 0] |= int(VegetationEvidenceFlag.DENSITY_SUPPORT)
    evidence_flags[(a4["track_id_by_point"] >= 0) & ~confirmed_wire] |= int(
        VegetationEvidenceFlag.A4_NONCONFIRMED_TRACK_ASSOCIATION
    )
    reasons = np.zeros(count, dtype=np.uint16)
    reasons[confirmed_wire] |= int(VegetationReasonFlag.CONFIRMED_WIRE_HARD_EXCLUSION)
    reasons[structural_invalid] |= int(VegetationReasonFlag.STRUCTURAL_INVALID_HARD_EXCLUSION)
    reasons[reliable_near_ground] |= int(VegetationReasonFlag.RELIABLE_NEAR_GROUND_HARD_EXCLUSION)
    reasons[strong_negative] |= int(VegetationReasonFlag.STRONGLY_NEGATIVE_RAW_HAG)
    reasons[(confidence < cfg.medium_confidence) & ~hard] |= int(
        VegetationReasonFlag.REVIEW_REQUIRED_LOW_CONFIDENCE
    )
    reasons[excluded] |= int(VegetationReasonFlag.MANUAL_EXCLUSION)
    quality = np.zeros(count, dtype=np.uint16)
    quality[geometry_degraded] |= int(VegetationQualityFlag.GEOMETRY_DEGRADED)
    quality[hag_uncertain] |= int(VegetationQualityFlag.HAG_UNCERTAIN)
    quality[hag_invalid] |= int(VegetationQualityFlag.HAG_INVALID_OR_UNAVAILABLE)
    quality[density_unusable] |= int(VegetationQualityFlag.DENSITY_UNUSABLE)
    quality[fallback_degenerate] |= int(VegetationQualityFlag.HEIGHT_FALLBACK_DEGENERATE)
    if "rgb" not in a1:
        quality |= int(VegetationQualityFlag.RGB_UNAVAILABLE_OR_UNUSABLE)
    if "intensity" not in a1:
        quality |= int(VegetationQualityFlag.INTENSITY_UNAVAILABLE_OR_UNUSABLE)
    points = VegetationCandidatePointSet(
        str(a1["source_id"].item()),
        str(a1["manifest_source_id"].item()),
        a1["xyz_local"],
        a1["origin_xyz"],
        a1["original_index"],
        a1["sample_index"],
        a4["wire_id_by_point"],
        hag_state,
        score,
        confidence,
        automatic.astype(np.uint8),
        final.astype(np.uint8),
        mask.astype(bool),
        evidence_flags,
        reasons,
        quality,
        provenance,
        action,
        geometry_flags,
        ground_flags,
    )
    arrays = _output_arrays(
        points,
        {
            "hag_contribution": hag,
            "volumetric_contribution": volumetric,
            "planarity_penalty": planarity,
            "linearity_penalty": linearity.astype(np.float32),
            "density_contribution": density,
            "auxiliary_contribution": auxiliary,
            "geometry_confidence": geometry_confidence,
            "hag_confidence": hag_confidence,
            "ground_support_confidence": ground_support,
            "density_confidence": density_confidence,
        },
    )
    level_counts = {
        level.name: int(np.count_nonzero(final == int(level))) for level in VegetationCandidateLevel
    }
    rss_end = _rss(process)
    rss_values = [value for value in (rss_start, rss_end) if value is not None]
    stats = VegetationCandidateStatistics(
        count,
        int(mask.sum()),
        level_counts["REJECTED"],
        level_counts["LOW"],
        level_counts["MEDIUM"],
        level_counts["HIGH"],
        int(confirmed_wire.sum()),
        int(reliable_near_ground.sum()),
        int(structural_invalid.sum()),
        time.perf_counter() - started,
        rss_start,
        max(rss_values) if rss_values else None,
        rss_end,
    )
    counts = {
        "hag_states": {
            state.name: int(np.count_nonzero(hag_state == int(state)))
            for state in VegetationHagState
        },
        "height_fallback_count": int(fallback_used.sum()),
        "automatic_levels": {
            level.name: int(np.count_nonzero(automatic == int(level)))
            for level in VegetationCandidateLevel
        },
        "final_levels": level_counts,
        "evidence_flags": _flag_counts(evidence_flags, VegetationEvidenceFlag),
        "reason_flags": _flag_counts(reasons, VegetationReasonFlag),
        "quality_flags": _flag_counts(quality, VegetationQualityFlag),
        "manual_provenance": _flag_counts(provenance, VegetationManualProvenanceFlag),
    }
    formulas = {
        "likelihood": "clip01(weighted height/scattering/density/auxiliary evidence - capped planarity/linearity penalty)",
        "confidence": "clip01(0.35 geometry + 0.35 HAG + 0.15 ground support + 0.15 density)",
        "rgb_intensity_default_contribution": 0.0,
    }
    _check_cancelled(token, "during scoring")
    return arrays, points, stats, counts, formulas


def _hag_states(
    a2b: Mapping[str, np.ndarray], cfg: VegetationCandidateConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flags, raw, hag, distance = (
        a2b["quality_flags"],
        a2b["raw_height_agl_m"],
        a2b["height_agl_m"],
        a2b["ground_distance_m"],
    )
    invalid = (
        _has(flags, GroundQualityFlag.INVALID_GROUND_ESTIMATE)
        | _has(flags, GroundQualityFlag.EXTRAPOLATION_TOO_FAR)
        | ~np.isfinite(raw)
        | ~np.isfinite(hag)
        | ~np.isfinite(distance)
    )
    strong = ~invalid & (raw <= -cfg.strong_negative_raw_hag_m)
    uncertain_flags = (
        GroundQualityFlag.SPARSE_GROUND_CELL,
        GroundQualityFlag.EMPTY_GROUND_CELL,
        GroundQualityFlag.INTERPOLATED_GROUND,
        GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION,
        GroundQualityFlag.BELOW_GROUND_TOLERANCE,
        GroundQualityFlag.SOURCE_CLASSIFICATION_DISAGREES,
        GroundQualityFlag.EXTREME_HEIGHT_AGL,
    )
    uncertain = (
        ~invalid & ~strong & np.logical_or.reduce([_has(flags, flag) for flag in uncertain_flags])
    )
    state = np.full(len(flags), int(VegetationHagState.RELIABLE), dtype=np.uint8)
    state[uncertain] = int(VegetationHagState.UNCERTAIN)
    state[strong] = int(VegetationHagState.STRONGLY_NEGATIVE)
    state[invalid] = int(VegetationHagState.INVALID_OR_UNAVAILABLE)
    return (
        state,
        state == int(VegetationHagState.INVALID_OR_UNAVAILABLE),
        uncertain,
        invalid,
        strong,
    )


def _hag_evidence(
    height_m: np.ndarray,
    hag: np.ndarray,
    state: np.ndarray,
    structural_invalid: np.ndarray,
    cfg: VegetationCandidateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.zeros(len(hag), dtype=np.float32)
    normal = (state == int(VegetationHagState.RELIABLE)) | (
        state == int(VegetationHagState.UNCERTAIN)
    )
    result[normal] = _clip(
        (hag[normal] - cfg.near_ground_hag_m) / (cfg.height_full_hag_m - cfg.near_ground_hag_m)
    )
    fallback = state == int(VegetationHagState.INVALID_OR_UNAVAILABLE)
    valid = np.isfinite(height_m) & ~structural_invalid
    source = height_m[valid]
    used = np.zeros(len(hag), dtype=bool)
    degenerate = np.zeros(len(hag), dtype=bool)
    if source.size:
        low, high = np.percentile(source, (10, 90))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            use = fallback & valid
            result[use] = 0.5 * _clip((height_m[use] - low) / (high - low))
            used[use] = True
        else:
            degenerate[fallback] = True
    else:
        degenerate[fallback] = True
    return result, used, degenerate


def _density_evidence(density: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    result = np.zeros(len(density), dtype=np.float32)
    valid = ~invalid & np.isfinite(density) & (density >= 0)
    if np.any(valid):
        transformed = np.log1p(density[valid].astype(np.float64))
        low, high = np.percentile(transformed, (10, 90))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            result[valid] = _clip((transformed - low) / (high - low))
    return result


def _levels(
    score: np.ndarray, confidence: np.ndarray, cfg: VegetationCandidateConfig
) -> np.ndarray:
    level = np.full(len(score), int(VegetationCandidateLevel.REJECTED), dtype=np.uint8)
    level[score >= cfg.low_score] = int(VegetationCandidateLevel.LOW)
    level[score >= cfg.medium_score] = int(VegetationCandidateLevel.MEDIUM)
    level[score >= cfg.high_score] = int(VegetationCandidateLevel.HIGH)
    level[(level == int(VegetationCandidateLevel.HIGH)) & (confidence < cfg.high_confidence)] = int(
        VegetationCandidateLevel.MEDIUM
    )
    level[
        (level >= int(VegetationCandidateLevel.MEDIUM)) & (confidence < cfg.medium_confidence)
    ] = int(VegetationCandidateLevel.LOW)
    return level


def _manual_arrays(
    a1: Mapping[str, np.ndarray],
    manual: VegetationManualSelectionConfig,
    confirmed_wire: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    original, xyz = a1["original_index"], a1["xyz_local"]
    requested = (*manual.include_original_index, *manual.exclude_original_index)
    missing = [item for item in requested if not np.any(original == item)]
    if missing:
        raise _error(
            "VEGETATION_MANUAL_INDEX", "manual original_index is not present", indexes=missing
        )
    include = np.isin(original, np.asarray(manual.include_original_index, dtype=np.int64))
    exclude = np.isin(original, np.asarray(manual.exclude_original_index, dtype=np.int64))
    for lower, upper in manual.include_boxes:
        include |= np.all((xyz >= lower) & (xyz <= upper), axis=1)
    for lower, upper in manual.exclude_boxes:
        exclude |= np.all((xyz >= lower) & (xyz <= upper), axis=1)
    flags = np.zeros(len(original), dtype=np.uint8)
    flags[include] |= int(VegetationManualProvenanceFlag.MANUAL_INCLUDE_MATCHED)
    flags[exclude] |= int(VegetationManualProvenanceFlag.MANUAL_EXCLUDE_MATCHED)
    flags[include & exclude] |= int(VegetationManualProvenanceFlag.MANUAL_CONFLICT)
    flags[include & confirmed_wire] |= int(VegetationManualProvenanceFlag.MANUAL_WIRE_BLOCKED)
    action = np.full(len(original), int(VegetationManualAction.NONE), dtype=np.uint8)
    action[include] = int(VegetationManualAction.INCLUDE)
    action[exclude] = int(VegetationManualAction.EXCLUDE)
    action[include & confirmed_wire] = int(VegetationManualAction.WIRE_BLOCKED)
    action[include & exclude] = int(VegetationManualAction.EXCLUDE)
    return flags, action


def _output_arrays(
    points: VegetationCandidatePointSet, audits: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    arrays = {
        "source_id": np.asarray(points.source_id),
        "manifest_source_id": np.asarray(points.manifest_source_id),
    }
    arrays.update(
        {
            name: getattr(points, name)
            for name in points.__dataclass_fields__
            if name not in {"source_id", "manifest_source_id"}
        }
    )
    arrays.update(audits)
    return arrays


def _validate_output_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    try:
        point_fields = {
            name: arrays[name]
            for name in VegetationCandidatePointSet.__dataclass_fields__
            if name not in {"source_id", "manifest_source_id"}
        }
        points = VegetationCandidatePointSet(
            str(arrays["source_id"].item()),
            str(arrays["manifest_source_id"].item()),
            **point_fields,
        )
    except (KeyError, ValueError) as error:
        raise _error(
            "VEGETATION_OUTPUT_ARRAYS", "V1 output arrays are invalid", cause=str(error)
        ) from error
    for name in (
        "hag_contribution",
        "volumetric_contribution",
        "planarity_penalty",
        "linearity_penalty",
        "density_contribution",
        "auxiliary_contribution",
        "geometry_confidence",
        "hag_confidence",
        "ground_support_confidence",
        "density_confidence",
    ):
        value = arrays.get(name)
        if (
            value is None
            or value.shape != (points.point_count,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or not np.all((value >= 0) & (value <= 1))
        ):
            raise _error("VEGETATION_OUTPUT_AUDIT", "V1 audit array is invalid", array=name)


def _publish(
    arrays: Mapping[str, np.ndarray],
    points: VegetationCandidatePointSet,
    npz_out: Path,
    json_out: Path,
    cfg: VegetationCandidateConfig,
    manual: VegetationManualSelectionConfig,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    statistics: VegetationCandidateStatistics,
    counts: Mapping[str, Any],
    formulas: Mapping[str, Any],
    root: Path,
) -> VegetationCandidateResult:
    temp_npz, temp_json = _temporary(npz_out), _temporary(json_out)
    published_npz = False
    try:
        np.savez_compressed(temp_npz, **arrays)
        _validate_output_arrays(_load_npz(temp_npz))
        provenance = {
            name: {"logical_path": _relative_or_name(root, path), "sha256": hashes[name]}
            for name, path in paths.items()
        }
        result = VegetationCandidateResult(
            SCHEMA_VERSION,
            points.source_id,
            points.manifest_source_id,
            npz_out.name,
            _sha(temp_npz),
            temp_npz.stat().st_size,
            cfg,
            manual,
            provenance,
            statistics,
            points.array_metadata()
            | {
                name: {"shape": [points.point_count], "dtype": "float32", "unit": "unitless"}
                for name in arrays
                if name.endswith("contribution") or name.endswith("confidence")
            },
            counts,
            formulas,
            (),
            ALGORITHM_VERSION,
            datetime.now(timezone.utc).isoformat(),
        )
        temp_json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        VegetationCandidateResult.from_dict(_load_json(temp_json))
        os.replace(temp_npz, npz_out)
        published_npz = True
        os.replace(temp_json, json_out)
        return result
    except Exception:
        temp_npz.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)
        if published_npz:
            npz_out.unlink(missing_ok=True)
        raise


def _hover_text(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    names = {member.value: member.name for member in VegetationCandidateLevel}
    states = {member.value: member.name for member in VegetationHagState}
    return np.asarray(
        [
            f"original_index={int(arrays['original_index'][i])}<br>sample_index={int(arrays['sample_index'][i])}<br>score={float(arrays['vegetation_rule_score'][i]):.3f}<br>confidence={float(arrays['vegetation_confidence'][i]):.3f}<br>automatic={names[int(arrays['automatic_vegetation_level'][i])]}<br>final={names[int(arrays['vegetation_level'][i])]}<br>HAG={states[int(arrays['hag_state'][i])]}<br>wire_id={int(arrays['wire_id_by_point'][i])}<br>evidence={int(arrays['vegetation_evidence_flags'][i])}<br>reason={int(arrays['vegetation_reason_flags'][i])}<br>quality={int(arrays['vegetation_quality_flags'][i])}<br>manual={int(arrays['manual_provenance_flags'][i])}"
            for i in range(len(arrays["original_index"]))
        ],
        dtype=str,
    )


def _clip(value: np.ndarray | float) -> np.ndarray:
    return np.nan_to_num(np.clip(value, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def _has(values: np.ndarray, flag: int) -> np.ndarray:
    return (values & int(flag)) != 0


def _flag_counts(values: np.ndarray, flag_type: Any) -> dict[str, int]:
    return {
        item.name: int(np.count_nonzero(values & int(item))) for item in flag_type if int(item) != 0
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stream:
            arrays = {name: stream[name] for name in stream.files}
    except (OSError, ValueError) as error:
        raise _error(
            "VEGETATION_INPUT_NPZ",
            "unable to load NPZ without pickle",
            path=str(path),
            cause=str(error),
        ) from error
    if any(value.dtype == object for value in arrays.values()):
        raise _error("VEGETATION_INPUT_OBJECT", "object arrays are forbidden", path=str(path))
    return arrays


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _error(
            "VEGETATION_INPUT_JSON",
            "unable to load JSON manifest",
            path=str(path),
            cause=str(error),
        ) from error
    if not isinstance(value, Mapping):
        raise _error("VEGETATION_INPUT_JSON", "JSON manifest must be an object", path=str(path))
    return value


def _validate_output_dir(output: Path, allowed: Path) -> None:
    try:
        output.resolve().relative_to(allowed.resolve())
    except ValueError as error:
        raise _error(
            "VEGETATION_OUTPUT_FORBIDDEN",
            "output must remain in approved directory",
            output=str(output),
        ) from error


def _validate_outputs(paths: tuple[Path, Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _error(
            "VEGETATION_OUTPUT_EXISTS", "output exists and overwrite is disabled", paths=existing
        )


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _relative_or_name(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_process() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process()


def _rss(process: Any | None) -> int | None:
    try:
        return int(process.memory_info().rss) if process is not None else None
    except Exception:
        return None


def _check_cancelled(token: CancellationToken | None, stage: str) -> None:
    if token is not None and token.is_cancelled():
        raise _error("VEGETATION_CANCELLED", "V1 operation was cancelled", stage=stage)


def _report(
    reporter: ProgressReporter | None,
    task_id: str,
    stage: str,
    processed: int,
    total: int,
    status: str = "running",
) -> None:
    if reporter is not None:
        reporter.update(
            PipelineProgress(
                task_id=task_id,
                stage=stage,
                status=status,
                processed_points=processed,
                total_points=total,
                percent=100.0 * processed / max(total, 1),
            )
        )


def _error(code: str, message: str, **details: Any) -> VegetationCandidateError:
    cause = details.pop("cause", None)
    return VegetationCandidateError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="algorithms.vegetation.candidates",
            recoverable=False,
            details=details,
            cause=str(cause) if cause is not None else None,
        )
    )
