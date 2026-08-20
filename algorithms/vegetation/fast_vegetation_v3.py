"""FAST-only V3 positive vegetation admission and diagnostic review support.

V3 is intentionally isolated from FGC/FRV.  It reuses the V2 PDAL covariance
artifact, W1 geometry, and positive V1 evidence to construct a stricter input
for the unchanged V4 implementation.  All memberships remain source-index
preserving; W1's pre-existing members are never matched by XYZ proximity.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.vegetation.fast_research import (
    FAST_RESEARCH_SCHEMA_VERSION,
    NON_CERTIFIED_INPUT_MARKER,
    FastGroundResult,
    FastResearchError,
    FastVegetationInputResult,
    _array_digest,
    _write_json_atomic,
    _write_npz_atomic,
    canonical_digest,
    fast_ground_content_digest,
    sha256_file,
    validate_fast_v4_arrays,
)
from algorithms.wires.reconstruct import load_wire_models

FAST_V3_SCHEMA_VERSION = "fast-vegetation-v3-v1"
FAST_V3_MARKER = "NON_CERTIFIED_FAST_VEGETATION_V3"
_DENSE_COMPONENT_LABEL_MAX_BYTES = 256 * 1024 * 1024
_NEIGHBOR_OFFSETS_26_HALF = tuple(
    (dx, dy, dz)
    for dx in range(-1, 2)
    for dy in range(-1, 2)
    for dz in range(-1, 2)
    if (dx, dy, dz) > (0, 0, 0)
)


class FastV3DiagnosticState(IntEnum):
    """Exclusive FAST-research state recorded for every V2-admission point."""

    GROUND_EXCLUDED = 1
    EXACT_WIRE_EXCLUDED = 2
    WIRE_GEOMETRY_EXCLUDED = 3
    RIGID_LINEAR_EXCLUDED = 4
    RIGID_SURFACE_EXCLUDED = 5
    NO_POSITIVE_VEGETATION_EVIDENCE = 6
    VEGETATION_RETAINED = 7


@dataclass(frozen=True, slots=True)
class FastVegetationV3Config:
    """Scene-inspected V3 admission settings, all explicitly persisted."""

    # 0.6 m is the scene-calibrated V1 continuity radius: it preserves the
    # sparse but supported canopy members needed for V4 while retaining an
    # explicit positive-evidence requirement (rather than a distance-only
    # admission rule).
    v1_support_radius_m: float = 0.6
    wire_candidate_distance_m: float = 0.6
    wire_component_cell_m: float = 0.1
    wire_min_component_points: int = 15
    wire_max_v1_support_fraction: float = 0.15
    wire_max_median_distance_m: float = 0.12
    wire_max_p95_distance_m: float = 0.2
    wire_min_median_linearity: float = 0.35
    wire_max_median_scattering: float = 0.25
    wire_min_tangent_alignment: float = 0.95
    wire_min_w1_span_m: float = 0.3
    wire_min_continuous_bins: int = 6
    wire_max_z_span_m: float = 0.3
    rigid_linear_cell_m: float = 0.15
    rigid_linear_max_hag_m: float = 1.0
    rigid_linear_min_points: int = 30
    rigid_linear_min_linearity: float = 0.35
    rigid_linear_max_scattering: float = 0.25
    rigid_linear_max_vertical_axis: float = 0.4
    rigid_linear_min_aspect_ratio: float = 3.0
    rigid_linear_max_z_span_m: float = 0.6
    rigid_surface_cell_m: float = 0.1
    rigid_surface_min_points: int = 100
    rigid_surface_max_hag_m: float = 2.5
    rigid_surface_min_planarity: float = 0.65
    rigid_surface_max_scattering: float = 0.16
    rigid_surface_max_vertical_axis: float = 0.55
    rigid_surface_max_aspect_ratio: float = 5.0
    rigid_surface_min_flatness_ratio: float = 1.2
    rigid_surface_max_z_span_m: float = 1.5
    rigid_surface_min_xy_span_m: float = 0.6

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise FastResearchError(f"FAST_V3_CONFIG_INVALID_{name.upper()}")
            if float(value) <= 0.0:
                raise FastResearchError(f"FAST_V3_CONFIG_INVALID_{name.upper()}")
        if self.wire_max_v1_support_fraction > 1.0:
            raise FastResearchError("FAST_V3_CONFIG_INVALID_WIRE_MAX_V1_SUPPORT_FRACTION")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _W1Metrics:
    distance_m: np.ndarray
    nearest_segment: np.ndarray
    wire_id: np.ndarray
    tangent_xyz: np.ndarray
    along_wire_m: np.ndarray


def build_fast_vegetation_v3_input(
    admission: FastVegetationInputResult,
    ground: FastGroundResult,
    output_root: Path,
    *,
    v2_feature_npz: Path,
    w1_npz: Path,
    w1_json: Path,
    reference_a2a_npz: Path,
    reference_v1_npz: Path,
    config: FastVegetationV3Config | None = None,
) -> tuple[str, FastVegetationInputResult, dict[str, Any]]:
    """Build a positive-evidence, structure-aware FAST V3 input for V4."""

    config = config or FastVegetationV3Config()
    output_root = Path(output_root).resolve()
    admission_npz = Path(admission.output_npz_path)
    v2_feature_npz = Path(v2_feature_npz).resolve()
    w1_npz, w1_json = Path(w1_npz).resolve(), Path(w1_json).resolve()
    reference_a2a_npz, reference_v1_npz = (
        Path(reference_a2a_npz).resolve(),
        Path(reference_v1_npz).resolve(),
    )
    for path, label in (
        (admission_npz, "ADMISSION"),
        (v2_feature_npz, "V2_FEATURE"),
        (w1_npz, "W1_NPZ"),
        (w1_json, "W1_JSON"),
        (reference_a2a_npz, "A2A"),
        (reference_v1_npz, "V1"),
    ):
        if not path.is_file():
            raise FastResearchError(f"FAST_V3_{label}_MISSING")
    if sha256_file(admission_npz) != admission.output_npz_sha256:
        raise FastResearchError("FAST_V3_ADMISSION_SHA_MISMATCH")
    config_digest = _v3_config_digest(
        admission,
        ground,
        v2_feature_npz=v2_feature_npz,
        w1_npz=w1_npz,
        w1_json=w1_json,
        reference_a2a_npz=reference_a2a_npz,
        reference_v1_npz=reference_v1_npz,
        config=config,
    )
    clean_npz = output_root / "fast_v4_input.npz"
    diagnostics_npz = output_root / "fast_v3_diagnostics.npz"
    diagnostics_json = output_root / "fast_v3_diagnostics.json"
    cached = _load_v3_cache(diagnostics_json, clean_npz, diagnostics_npz, config_digest)
    if cached is not None:
        return "SKIPPED_FROM_CACHE", _fast_input_from_dict(cached["vegetation_input"]), cached

    started = time.perf_counter()
    original, xyz, hag = _load_admission(admission_npz)
    features = _load_v2_features(v2_feature_npz, original)
    classification = np.asarray(features["classification"], dtype=np.uint8)
    v2_retained = np.asarray(features["retained_mask"], dtype=bool)
    exact_wire = np.asarray(features["known_wire_exact_mask"], dtype=bool)
    w1_result, w1_arrays = load_wire_models(w1_npz, w1_json)
    if w1_result.source_id != admission.source_id:
        raise FastResearchError("FAST_V3_W1_SOURCE_ID_MISMATCH")
    known_wire_original = np.asarray(w1_arrays["ordered_input_original_index"], dtype="<i8")
    expected_exact = np.isin(original, known_wire_original)
    if not np.array_equal(exact_wire, expected_exact):
        raise FastResearchError("FAST_V3_EXACT_W1_CORRESPONDENCE_INVALID")

    positive_v1_support, v1_distance = _positive_v1_support(
        xyz, source_id=admission.source_id, reference_v1_npz=reference_v1_npz, config=config
    )
    w1_metrics = _w1_metrics(xyz, w1_arrays)
    point_tangent_alignment = _a2a_tangent_alignment(
        original,
        xyz,
        reference_a2a_npz=reference_a2a_npz,
        tangent_xyz=w1_metrics.tangent_xyz,
    )
    wire_geometry, wire_component_id, wire_summaries, component_alignment = (
        _wire_geometry_exclusion(
            xyz,
            v2_retained & ~exact_wire,
            hag=hag,
            linearity=features["linearity"],
            scattering=features["scattering"],
            positive_v1_support=positive_v1_support,
            w1=w1_metrics,
            config=config,
        )
    )
    tangent_alignment = point_tangent_alignment.copy()
    tangent_alignment[wire_geometry] = component_alignment[wire_geometry]

    rigid_linear, linear_component_id, linear_summaries = _rigid_linear_exclusion(
        xyz,
        v2_retained & ~exact_wire & ~wire_geometry,
        hag=hag,
        linearity=features["linearity"],
        scattering=features["scattering"],
        config=config,
    )
    rigid_surface, surface_component_id, surface_summaries = _rigid_surface_exclusion(
        xyz,
        v2_retained & ~exact_wire & ~wire_geometry & ~rigid_linear,
        hag=hag,
        planarity=features["planarity"],
        scattering=features["scattering"],
        config=config,
    )

    ground_mask = classification == 2
    positive_base = v2_retained & positive_v1_support
    retained = _positive_v1_admission(
        v2_retained=v2_retained,
        positive_v1_support=positive_v1_support,
        ground_mask=ground_mask,
        exact_wire=exact_wire,
        wire_geometry=wire_geometry,
        rigid_linear=rigid_linear,
        rigid_surface=rigid_surface,
    )
    state = _diagnostic_states(
        ground_mask=ground_mask,
        exact_wire=exact_wire,
        wire_geometry=wire_geometry,
        rigid_linear=rigid_linear,
        rigid_surface=rigid_surface,
        retained=retained,
    )
    _assert_v3_selection(
        state=state,
        retained=retained,
        ground_mask=ground_mask,
        exact_wire=exact_wire,
        wire_geometry=wire_geometry,
        rigid_linear=rigid_linear,
        rigid_surface=rigid_surface,
    )
    clean_original, clean_xyz, clean_hag = original[retained], xyz[retained], hag[retained]
    validate_fast_v4_arrays(clean_original, clean_xyz, clean_hag)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_npz_atomic(
        clean_npz,
        original_index=clean_original,
        xyz_local_m=clean_xyz,
        hag_m=clean_hag,
        candidate_reason_flags=np.zeros(len(clean_original), dtype=np.uint32),
    )
    _write_npz_atomic(
        diagnostics_npz,
        original_index=original,
        xyz_local_m=xyz,
        hag_m=hag,
        diagnostic_state=state.astype(np.uint8),
        v2_retained_mask=v2_retained,
        positive_v1_support_mask=positive_v1_support,
        v1_positive_distance_m=v1_distance.astype("<f4"),
        exact_wire_mask=exact_wire,
        wire_geometry_mask=wire_geometry,
        rigid_linear_mask=rigid_linear,
        rigid_surface_mask=rigid_surface,
        nearest_w1_distance_m=w1_metrics.distance_m.astype("<f4"),
        nearest_w1_segment=w1_metrics.nearest_segment.astype("<i4"),
        nearest_w1_wire_id=w1_metrics.wire_id.astype("<i4"),
        tangent_alignment=tangent_alignment.astype("<f4"),
        wire_component_id=wire_component_id.astype("<i4"),
        rigid_linear_component_id=linear_component_id.astype("<i4"),
        rigid_surface_component_id=surface_component_id.astype("<i4"),
        linearity=np.asarray(features["linearity"], dtype="<f4"),
        planarity=np.asarray(features["planarity"], dtype="<f4"),
        scattering=np.asarray(features["scattering"], dtype="<f4"),
        verticality=np.asarray(features["verticality"], dtype="<f4"),
    )
    vegetation = FastVegetationInputResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_INPUT_MARKER,
        ground_result_digest=fast_ground_content_digest(ground),
        config_digest=config_digest,
        output_npz_path=str(clean_npz),
        output_npz_sha256=sha256_file(clean_npz),
        point_count=int(len(clean_original)),
        source_id=admission.source_id,
        source_las_path=admission.source_las_path,
        source_point_count=admission.source_point_count,
        source_file_sha256=admission.source_file_sha256,
        origin_context=admission.origin_context,
        selected_bounds_min_xyz=admission.selected_bounds_min_xyz,
        selected_bounds_max_xyz=admission.selected_bounds_max_xyz,
        minimum_hag_m=admission.minimum_hag_m,
        admission_voxel_size_m=admission.admission_voxel_size_m,
        original_index_sha256=_array_digest(clean_original),
        xyz_sha256=_array_digest(clean_xyz),
        hag_sha256=_array_digest(clean_hag),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json_atomic(output_root / "fast_v4_input.json", vegetation.to_dict())
    counts = {
        "raw_admission_points": int(len(original)),
        "v2_retained_previous": int(np.count_nonzero(v2_retained)),
        "positive_v1_base_count": int(np.count_nonzero(positive_base)),
        "current_v2_points_with_positive_v1_evidence": int(np.count_nonzero(positive_base)),
        "current_v2_points_without_positive_v1_evidence": int(
            np.count_nonzero(v2_retained & ~positive_v1_support)
        ),
        "ground_excluded": int(np.count_nonzero(state == FastV3DiagnosticState.GROUND_EXCLUDED)),
        "exact_wire_excluded": int(
            np.count_nonzero(state == FastV3DiagnosticState.EXACT_WIRE_EXCLUDED)
        ),
        "wire_geometry_excluded": int(
            np.count_nonzero(state == FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED)
        ),
        "rigid_linear_excluded": int(
            np.count_nonzero(state == FastV3DiagnosticState.RIGID_LINEAR_EXCLUDED)
        ),
        "rigid_surface_excluded": int(
            np.count_nonzero(state == FastV3DiagnosticState.RIGID_SURFACE_EXCLUDED)
        ),
        "no_positive_vegetation_evidence_excluded": int(
            np.count_nonzero(state == FastV3DiagnosticState.NO_POSITIVE_VEGETATION_EVIDENCE)
        ),
        "final_v3_vegetation_retained": int(np.count_nonzero(retained)),
        "ground_overlap": 0,
        "exact_known_wire_overlap": 0,
        "wire_geometry_overlap": 0,
        "rigid_linear_overlap": 0,
        "rigid_surface_overlap": 0,
    }
    manifest = {
        "schema_version": FAST_V3_SCHEMA_VERSION,
        "non_certified_marker": FAST_V3_MARKER,
        "config_digest": config_digest,
        "config": config.to_dict(),
        "formula": (
            "V2_GEOMETRY_EVIDENCE_PASS AND POSITIVE_V1_COMPONENT_SUPPORT AND "
            "NOT_GROUND AND NOT_EXACT_W1_MEMBER AND NOT_WIRE_GEOMETRY AND "
            "NOT_RIGID_LINEAR AND NOT_RIGID_SURFACE"
        ),
        "source_artifacts": {
            "v2_feature_npz": str(v2_feature_npz),
            "v2_feature_npz_sha256": sha256_file(v2_feature_npz),
            "w1_npz": str(w1_npz),
            "w1_npz_sha256": sha256_file(w1_npz),
            "reference_a2a_npz": str(reference_a2a_npz),
            "reference_a2a_npz_sha256": sha256_file(reference_a2a_npz),
            "reference_v1_npz": str(reference_v1_npz),
            "reference_v1_npz_sha256": sha256_file(reference_v1_npz),
        },
        "correspondence": {
            "original_index_preserved": True,
            "w1_exact_membership_method": "original_index_only",
            "w1_fuzzy_xyz_membership_used": False,
            "w1_known_member_count": int(len(known_wire_original)),
        },
        "calibration": {
            "v1_positive_support_distance_quantiles_m": _quantiles(v1_distance[v2_retained]),
            "exact_w1_distance_quantiles_m": _quantiles(w1_metrics.distance_m[exact_wire]),
            "exact_w1_linearity_quantiles": _quantiles(features["linearity"][exact_wire]),
            "exact_w1_scattering_quantiles": _quantiles(features["scattering"][exact_wire]),
        },
        "counts": counts,
        "wire_geometry_components": wire_summaries,
        "rigid_linear_components": linear_summaries,
        "rigid_surface_components": surface_summaries,
        "machine_pre_v4_gate": {
            "status": "PASS",
            "ground_overlap": 0,
            "exact_known_wire_overlap": 0,
            "wire_geometry_overlap": 0,
            "rigid_linear_overlap": 0,
            "rigid_surface_overlap": 0,
            "positive_v1_retained_fraction": (
                0.0
                if counts["positive_v1_base_count"] == 0
                else counts["final_v3_vegetation_retained"] / counts["positive_v1_base_count"]
            ),
        },
        "vegetation_input": vegetation.to_dict(),
        "diagnostics_npz_path": str(diagnostics_npz),
        "diagnostics_npz_sha256": sha256_file(diagnostics_npz),
        "runtime_seconds": vegetation.elapsed_seconds,
    }
    _write_json_atomic(diagnostics_json, manifest)
    return "PASS", vegetation, manifest


def generate_fast_vegetation_review_v3(manifest: dict[str, Any], output_html: Path) -> Path:
    """Render all required V3 categories before V4 may run."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise FastResearchError("FAST_V3_REVIEW_DEPENDENCY_MISSING") from exc
    values = _load_diagnostics_arrays(manifest)
    state = values["diagnostic_state"]
    labels = (
        (FastV3DiagnosticState.VEGETATION_RETAINED, "retained vegetation", "#228b22"),
        (FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED, "wire geometry excluded", "#dc2626"),
        (FastV3DiagnosticState.RIGID_LINEAR_EXCLUDED, "rigid linear excluded", "#f59e0b"),
        (FastV3DiagnosticState.RIGID_SURFACE_EXCLUDED, "rigid surface excluded", "#9333ea"),
        (FastV3DiagnosticState.EXACT_WIRE_EXCLUDED, "exact W1 member excluded", "#7f1d1d"),
        (FastV3DiagnosticState.GROUND_EXCLUDED, "ground excluded", "#8b5e3c"),
        (
            FastV3DiagnosticState.NO_POSITIVE_VEGETATION_EVIDENCE,
            "no positive vegetation evidence",
            "#9ca3af",
        ),
    )
    figure = go.Figure()
    xyz = values["xyz_local_m"]
    original = values["original_index"]
    hag = values["hag_m"]
    for code, label_name, color in labels:
        selected = np.flatnonzero(state == int(code))
        selected = _even_indices(selected, 60_000)
        if not len(selected):
            continue
        figure.add_trace(
            go.Scatter3d(
                x=xyz[selected, 0],
                y=xyz[selected, 1],
                z=xyz[selected, 2],
                mode="markers",
                name=f"{label_name} ({len(np.flatnonzero(state == int(code))):,})",
                marker={"size": 1.9, "color": color, "opacity": 0.72},
                customdata=np.column_stack((original[selected], hag[selected])),
                hovertemplate=(
                    "original_index=%{customdata[0]}<br>HAG=%{customdata[1]:.2f} m<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=(
            "FAST vegetation V3 pre-V4 review — positive V1 support + W1 geometry + "
            "rigid-component exclusions"
        ),
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        legend={"orientation": "h"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return _write_html_atomic(figure, output_html)


def generate_minimum_clearance_local_review_v3(
    clearance: Any, manifest: dict[str, Any], output_html: Path
) -> tuple[Path, dict[str, Any]]:
    """Render and return the required V3 diagnostic for a suspicious closest pair."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise FastResearchError("FAST_V3_REVIEW_DEPENDENCY_MISSING") from exc
    valid = [record for record in clearance.records if record.minimum_clearance_m is not None]
    if not valid:
        raise FastResearchError("FAST_V3_CLEARANCE_NO_VALID_RECORD")
    record = min(valid, key=lambda item: float(item.minimum_clearance_m))
    values = _load_diagnostics_arrays(manifest)
    original = values["original_index"]
    position = int(np.searchsorted(original, record.nearest_tree_original_index))
    if position >= len(original) or int(original[position]) != record.nearest_tree_original_index:
        raise FastResearchError("FAST_V3_CLEARANCE_DIAGNOSTIC_INDEX_MISSING")
    state = FastV3DiagnosticState(int(values["diagnostic_state"][position])).name
    component_type, component_id = _point_component_type(values, position)
    diagnostic = {
        "tree_id": int(record.tree_id),
        "nearest_wire_id": int(record.nearest_wire_id),
        "minimum_clearance_m": float(record.minimum_clearance_m),
        "nearest_tree_original_index": int(record.nearest_tree_original_index),
        "nearest_tree_xyz_local_m": [float(value) for value in values["xyz_local_m"][position]],
        "nearest_tree_hag_m": float(values["hag_m"][position]),
        "v3_diagnostic_state": state,
        "v1_positive_support": bool(values["positive_v1_support_mask"][position]),
        "linearity": float(values["linearity"][position]),
        "planarity": float(values["planarity"][position]),
        "scattering": float(values["scattering"][position]),
        "distance_to_w1_m": float(values["nearest_w1_distance_m"][position]),
        "tangent_alignment": _finite_or_none(values["tangent_alignment"][position]),
        "component_type": component_type,
        "component_id": component_id,
    }
    tree = np.asarray(record.nearest_tree_xyz_local_m, dtype=np.float64)
    wire = np.asarray(record.nearest_wire_xyz_local_m, dtype=np.float64)
    figure = go.Figure(
        go.Scatter3d(
            x=[tree[0], wire[0]],
            y=[tree[1], wire[1]],
            z=[tree[2], wire[2]],
            mode="lines+markers+text",
            text=["nearest tree point", "nearest W1 curve point"],
            textposition="top center",
            marker={"size": 6, "color": ["#228b22", "#dc2626"]},
            line={"width": 7, "color": "#111827"},
            name="unchanged TreeWire closest pair",
        )
    )
    figure.add_annotation(
        text=(
            f"idx={diagnostic['nearest_tree_original_index']} | "
            f"HAG={diagnostic['nearest_tree_hag_m']:.3f} m"
            f" | V1 support={diagnostic['v1_positive_support']} | L/P/S="
            f"{diagnostic['linearity']:.3f}/{diagnostic['planarity']:.3f}/{diagnostic['scattering']:.3f}"
            f" | W1 distance={diagnostic['distance_to_w1_m']:.3f} m"
        ),
        x=0.5,
        y=0.02,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_layout(
        title=(
            "FAST V3 minimum-clearance diagnostic — "
            f"{diagnostic['minimum_clearance_m']:.9f} m ({state})"
        ),
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 60},
    )
    return _write_html_atomic(figure, output_html), diagnostic


def _v3_config_digest(
    admission: FastVegetationInputResult,
    ground: FastGroundResult,
    *,
    v2_feature_npz: Path,
    w1_npz: Path,
    w1_json: Path,
    reference_a2a_npz: Path,
    reference_v1_npz: Path,
    config: FastVegetationV3Config,
) -> str:
    return canonical_digest(
        {
            "schema_version": FAST_V3_SCHEMA_VERSION,
            "admission_sha256": admission.output_npz_sha256,
            "ground_digest": fast_ground_content_digest(ground),
            "v2_feature_sha256": sha256_file(v2_feature_npz),
            "w1_npz_sha256": sha256_file(w1_npz),
            "w1_json_sha256": sha256_file(w1_json),
            "a2a_sha256": sha256_file(reference_a2a_npz),
            "v1_sha256": sha256_file(reference_v1_npz),
            "config": config.to_dict(),
        }
    )


def _load_admission(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        original = np.asarray(stream["original_index"], dtype="<i8")
        xyz = np.asarray(stream["xyz_local_m"], dtype="<f8")
        hag = np.asarray(stream["hag_m"], dtype="<f4")
    validate_fast_v4_arrays(original, xyz, hag)
    return original, xyz, hag


def _load_v2_features(path: Path, original: np.ndarray) -> dict[str, np.ndarray]:
    required = (
        "original_index",
        "classification",
        "known_wire_exact_mask",
        "retained_mask",
        "linearity",
        "planarity",
        "scattering",
        "verticality",
    )
    try:
        with np.load(path, allow_pickle=False) as stream:
            values = {name: np.asarray(stream[name]) for name in required}
    except (OSError, ValueError, KeyError) as exc:
        raise FastResearchError("FAST_V3_V2_FEATURE_INVALID") from exc
    if not np.array_equal(np.asarray(values.pop("original_index"), dtype="<i8"), original):
        raise FastResearchError("FAST_V3_V2_FEATURE_CORRESPONDENCE_INVALID")
    count = len(original)
    for name in ("classification", "known_wire_exact_mask", "retained_mask"):
        if values[name].shape != (count,):
            raise FastResearchError("FAST_V3_V2_FEATURE_SHAPE_INVALID")
    for name in ("linearity", "planarity", "scattering", "verticality"):
        if values[name].shape != (count,) or not np.isfinite(values[name]).all():
            raise FastResearchError("FAST_V3_V2_FEATURE_SHAPE_INVALID")
    return values


def _positive_v1_support(
    xyz: np.ndarray, *, source_id: str, reference_v1_npz: Path, config: FastVegetationV3Config
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        with np.load(reference_v1_npz, allow_pickle=False) as stream:
            artifact_source = str(np.asarray(stream["source_id"]).item())
            v1_xyz = np.asarray(stream["xyz_local"], dtype=np.float64)
            vegetation_mask = np.asarray(stream["vegetation_mask"], dtype=bool)
    except (ImportError, OSError, ValueError, KeyError) as exc:
        raise FastResearchError("FAST_V3_V1_EVIDENCE_INVALID") from exc
    if (
        artifact_source != source_id
        or v1_xyz.shape[1:] != (3,)
        or len(v1_xyz) != len(vegetation_mask)
    ):
        raise FastResearchError("FAST_V3_V1_EVIDENCE_CORRESPONDENCE_INVALID")
    positive = v1_xyz[vegetation_mask]
    if len(positive) == 0:
        raise FastResearchError("FAST_V3_V1_POSITIVE_EVIDENCE_EMPTY")
    distance, _indices = cKDTree(positive).query(xyz, workers=-1)
    if not np.isfinite(distance).all():
        raise FastResearchError("FAST_V3_V1_DISTANCE_INVALID")
    return distance <= config.v1_support_radius_m, np.asarray(distance, dtype=np.float64)


def _w1_metrics(xyz: np.ndarray, w1_arrays: dict[str, np.ndarray]) -> _W1Metrics:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise FastResearchError("FAST_V3_SCIPY_UNAVAILABLE") from exc
    curve = np.asarray(w1_arrays["sampled_curve_xyz"], dtype=np.float64)
    offsets = np.asarray(w1_arrays["sampled_curve_offsets"], dtype=np.int64)
    if curve.ndim != 2 or curve.shape[1] != 3 or offsets.ndim != 1 or len(offsets) < 2:
        raise FastResearchError("FAST_V3_W1_CURVE_INVALID")
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    wire_ids: list[np.ndarray] = []
    starts_s: list[np.ndarray] = []
    for wire_id, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        if end - begin < 2:
            continue
        rows = np.arange(begin, end - 1, dtype=np.int64)
        first, second = curve[rows], curve[rows + 1]
        step = np.linalg.norm(second - first, axis=1)
        if np.any(step <= 0.0):
            raise FastResearchError("FAST_V3_W1_ZERO_LENGTH_EDGE")
        starts.append(first)
        ends.append(second)
        wire_ids.append(np.full(len(rows), wire_id, dtype=np.int32))
        starts_s.append(np.concatenate(([0.0], np.cumsum(step)[:-1])))
    if not starts:
        raise FastResearchError("FAST_V3_W1_NO_EDGE")
    start, end = np.concatenate(starts), np.concatenate(ends)
    wire_id = np.concatenate(wire_ids)
    along_start = np.concatenate(starts_s)
    vector = end - start
    length = np.linalg.norm(vector, axis=1)
    midpoint = (start + end) * 0.5
    nearest = cKDTree(midpoint).query(xyz, k=min(8, len(midpoint)), workers=-1)[1]
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    a, v = start[nearest], vector[nearest]
    delta = xyz[:, None, :] - a
    t = np.clip(np.sum(delta * v, axis=2) / np.sum(v * v, axis=2), 0.0, 1.0)
    projection = a + v * t[..., None]
    distance = np.linalg.norm(xyz[:, None, :] - projection, axis=2)
    row = np.arange(len(xyz), dtype=np.int64)
    local_choice = np.argmin(distance, axis=1)
    segment = nearest[row, local_choice]
    nearest_distance = distance[row, local_choice]
    nearest_t = t[row, local_choice]
    tangent = vector[segment] / length[segment, None]
    return _W1Metrics(
        distance_m=np.asarray(nearest_distance, dtype=np.float64),
        nearest_segment=np.asarray(segment, dtype=np.int32),
        wire_id=np.asarray(wire_id[segment], dtype=np.int32),
        tangent_xyz=np.asarray(tangent, dtype=np.float64),
        along_wire_m=np.asarray(
            along_start[segment] + nearest_t * length[segment], dtype=np.float64
        ),
    )


def _a2a_tangent_alignment(
    original: np.ndarray,
    xyz: np.ndarray,
    *,
    reference_a2a_npz: Path,
    tangent_xyz: np.ndarray,
) -> np.ndarray:
    del xyz  # Alignment is allowed only for exact A2A source-row correspondence.
    try:
        with np.load(reference_a2a_npz, allow_pickle=False) as stream:
            a2a_original = np.asarray(stream["original_index"], dtype="<i8")
            principal = np.asarray(stream["principal_direction"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise FastResearchError("FAST_V3_A2A_DIRECTION_INVALID") from exc
    if principal.shape != (len(a2a_original), 3) or not np.isfinite(principal).all():
        raise FastResearchError("FAST_V3_A2A_DIRECTION_INVALID")
    order = np.argsort(a2a_original)
    sorted_original = a2a_original[order]
    position = np.searchsorted(sorted_original, original)
    safe = np.minimum(position, len(sorted_original) - 1)
    matched = (position < len(sorted_original)) & (sorted_original[safe] == original)
    alignment = np.full(len(original), np.nan, dtype=np.float64)
    directions = principal[order][position[matched]]
    norms = np.linalg.norm(directions, axis=1)
    valid = norms > 0.0
    match_rows = np.flatnonzero(matched)
    alignment[match_rows[valid]] = np.abs(
        np.sum(directions[valid] / norms[valid, None] * tangent_xyz[match_rows[valid]], axis=1)
    )
    return alignment


def _wire_geometry_exclusion(
    xyz: np.ndarray,
    candidate: np.ndarray,
    *,
    hag: np.ndarray,
    linearity: np.ndarray,
    scattering: np.ndarray,
    positive_v1_support: np.ndarray,
    w1: _W1Metrics,
    config: FastVegetationV3Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    seed = candidate & (w1.distance_m <= config.wire_candidate_distance_m)
    labels, _cell, _origin = _grid_component_labels(xyz, seed, config.wire_component_cell_m)
    excluded = np.zeros(len(xyz), dtype=bool)
    component_id = np.full(len(xyz), -1, dtype=np.int32)
    alignment = np.full(len(xyz), np.nan, dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for label_id in np.unique(labels[labels > 0]):
        rows = np.flatnonzero(labels == label_id)
        summary = _wire_component_summary(
            int(label_id),
            rows,
            xyz=xyz,
            hag=hag,
            linearity=linearity,
            scattering=scattering,
            positive_v1_support=positive_v1_support,
            w1=w1,
            config=config,
        )
        if bool(summary["wire_geometry_excluded"]):
            excluded[rows] = True
            component_id[rows] = int(label_id)
            alignment[rows] = float(summary["median_tangent_alignment"])
            summaries.append(summary)
    return excluded, component_id, summaries, alignment


def _wire_component_summary(
    component_id: int,
    rows: np.ndarray,
    *,
    xyz: np.ndarray,
    hag: np.ndarray,
    linearity: np.ndarray,
    scattering: np.ndarray,
    positive_v1_support: np.ndarray,
    w1: _W1Metrics,
    config: FastVegetationV3Config,
) -> dict[str, Any]:
    points = xyz[rows]
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    axis = eigenvectors[:, -1]
    tangent_alignment = np.abs(w1.tangent_xyz[rows] @ axis)
    span_by_wire: list[float] = []
    run_by_wire: list[int] = []
    for wire_id in np.unique(w1.wire_id[rows]):
        along = w1.along_wire_m[rows][w1.wire_id[rows] == wire_id]
        span_by_wire.append(float(np.max(along) - np.min(along)))
        bins = np.unique(np.floor(along / 0.05).astype(np.int64))
        run_by_wire.append(_longest_contiguous_run(bins))
    z_span = float(np.ptp(points[:, 2]))
    summary = {
        "component_id": component_id,
        "point_count": int(len(rows)),
        "median_hag_m": float(np.median(hag[rows])),
        "median_distance_to_w1_m": float(np.median(w1.distance_m[rows])),
        "p95_distance_to_w1_m": float(np.quantile(w1.distance_m[rows], 0.95)),
        "median_linearity": float(np.median(linearity[rows])),
        "median_scattering": float(np.median(scattering[rows])),
        "median_tangent_alignment": float(np.median(tangent_alignment)),
        "max_w1_span_m": max(span_by_wire, default=0.0),
        "max_continuous_w1_bins": max(run_by_wire, default=0),
        "z_span_m": z_span,
        "positive_v1_support_fraction": float(np.mean(positive_v1_support[rows])),
    }
    summary["wire_geometry_excluded"] = bool(
        summary["point_count"] >= config.wire_min_component_points
        and summary["positive_v1_support_fraction"] <= config.wire_max_v1_support_fraction
        and summary["median_distance_to_w1_m"] <= config.wire_max_median_distance_m
        and summary["p95_distance_to_w1_m"] <= config.wire_max_p95_distance_m
        and summary["median_linearity"] >= config.wire_min_median_linearity
        and summary["median_scattering"] <= config.wire_max_median_scattering
        and summary["median_tangent_alignment"] >= config.wire_min_tangent_alignment
        and summary["max_w1_span_m"] >= config.wire_min_w1_span_m
        and summary["max_continuous_w1_bins"] >= config.wire_min_continuous_bins
        and summary["z_span_m"] <= config.wire_max_z_span_m
    )
    return summary


def _rigid_linear_exclusion(
    xyz: np.ndarray,
    candidate: np.ndarray,
    *,
    hag: np.ndarray,
    linearity: np.ndarray,
    scattering: np.ndarray,
    config: FastVegetationV3Config,
    component_grid_diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seed = (
        candidate
        & (hag <= config.rigid_linear_max_hag_m)
        & (linearity >= config.rigid_linear_min_linearity)
        & (scattering <= config.rigid_linear_max_scattering)
    )
    labels, _cell, _origin = _grid_component_labels(
        xyz,
        seed,
        config.rigid_linear_cell_m,
        diagnostics=component_grid_diagnostics,
    )
    excluded = np.zeros(len(xyz), dtype=bool)
    component_id = np.full(len(xyz), -1, dtype=np.int32)
    summaries: list[dict[str, Any]] = []
    for label_id in np.unique(labels[labels > 0]):
        rows = np.flatnonzero(labels == label_id)
        summary = _rigid_component_summary(
            int(label_id),
            rows,
            xyz=xyz,
            hag=hag,
            linearity=linearity,
            planarity=None,
            scattering=scattering,
        )
        summary["rigid_linear_excluded"] = bool(
            summary["point_count"] >= config.rigid_linear_min_points
            and summary["median_hag_m"] <= config.rigid_linear_max_hag_m
            and summary["median_linearity"] >= config.rigid_linear_min_linearity
            and summary["median_scattering"] <= config.rigid_linear_max_scattering
            and summary["vertical_axis_abs"] <= config.rigid_linear_max_vertical_axis
            and summary["aspect_ratio"] >= config.rigid_linear_min_aspect_ratio
            and summary["z_span_m"] <= config.rigid_linear_max_z_span_m
        )
        if bool(summary["rigid_linear_excluded"]):
            excluded[rows] = True
            component_id[rows] = int(label_id)
            summaries.append(summary)
    return excluded, component_id, summaries


def _rigid_surface_exclusion(
    xyz: np.ndarray,
    candidate: np.ndarray,
    *,
    hag: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
    config: FastVegetationV3Config,
    component_grid_diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seed = (
        candidate
        & (planarity >= config.rigid_surface_min_planarity)
        & (scattering <= config.rigid_surface_max_scattering)
    )
    labels, _cell, _origin = _grid_component_labels(
        xyz,
        seed,
        config.rigid_surface_cell_m,
        diagnostics=component_grid_diagnostics,
    )
    excluded = np.zeros(len(xyz), dtype=bool)
    component_id = np.full(len(xyz), -1, dtype=np.int32)
    summaries: list[dict[str, Any]] = []
    for label_id in np.unique(labels[labels > 0]):
        rows = np.flatnonzero(labels == label_id)
        summary = _rigid_component_summary(
            int(label_id),
            rows,
            xyz=xyz,
            hag=hag,
            linearity=None,
            planarity=planarity,
            scattering=scattering,
        )
        summary["rigid_surface_excluded"] = bool(
            summary["point_count"] >= config.rigid_surface_min_points
            and summary["median_hag_m"] <= config.rigid_surface_max_hag_m
            and summary["median_planarity"] >= config.rigid_surface_min_planarity
            and summary["median_scattering"] <= config.rigid_surface_max_scattering
            and summary["vertical_axis_abs"] <= config.rigid_surface_max_vertical_axis
            and summary["aspect_ratio"] <= config.rigid_surface_max_aspect_ratio
            and summary["flatness_ratio"] >= config.rigid_surface_min_flatness_ratio
            and summary["z_span_m"] <= config.rigid_surface_max_z_span_m
            and summary["min_xy_span_m"] >= config.rigid_surface_min_xy_span_m
        )
        if bool(summary["rigid_surface_excluded"]):
            excluded[rows] = True
            component_id[rows] = int(label_id)
            summaries.append(summary)
    return excluded, component_id, summaries


def _rigid_component_summary(
    component_id: int,
    rows: np.ndarray,
    *,
    xyz: np.ndarray,
    hag: np.ndarray,
    linearity: np.ndarray | None,
    planarity: np.ndarray | None,
    scattering: np.ndarray,
) -> dict[str, Any]:
    points = xyz[rows]
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    span = np.ptp(points, axis=0)
    summary: dict[str, Any] = {
        "component_id": component_id,
        "point_count": int(len(rows)),
        "median_hag_m": float(np.median(hag[rows])),
        "median_scattering": float(np.median(scattering[rows])),
        "vertical_axis_abs": float(abs(eigenvectors[2, -1])),
        "aspect_ratio": float(math.sqrt(max(eigenvalues[-1], 0.0) / max(eigenvalues[-2], 1e-9))),
        "flatness_ratio": float(math.sqrt(max(eigenvalues[-2], 0.0) / max(eigenvalues[-3], 1e-9))),
        "xy_span_m": [float(span[0]), float(span[1])],
        "min_xy_span_m": float(min(span[0], span[1])),
        "z_span_m": float(span[2]),
    }
    if linearity is not None:
        summary["median_linearity"] = float(np.median(linearity[rows]))
    if planarity is not None:
        summary["median_planarity"] = float(np.median(planarity[rows]))
    return summary


def _grid_component_labels(
    xyz: np.ndarray,
    seed: np.ndarray,
    cell_size_m: float,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label occupied voxels with sparse 26-connected components.

    The previous implementation materialized the full scene bounding box as a
    boolean array and then asked ``scipy.ndimage.label`` for an equally sized
    integer label array.  This preserves the same ``generate_binary_structure(3,
    3)`` connectivity while storing only occupied cells and their local sparse
    graph.  Component IDs are canonicalized by the first C-order voxel, exactly
    matching the historical dense-label ordering.
    """

    if len(xyz) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.zeros(3, dtype=np.float64),
        )
    origin = np.min(xyz, axis=0)
    if seed.shape != (len(xyz),) or not np.any(seed):
        _record_component_grid_diagnostics(
            diagnostics,
            cell_size_m=cell_size_m,
            shape=None,
            seed_point_count=0,
            occupied_cell_count=0,
            sparse_edge_count=0,
            sparse_estimated_working_bytes=0,
        )
        return np.zeros(len(xyz), dtype=np.int32), np.empty((0, 3), dtype=np.int32), origin

    cell = np.floor((xyz - origin) / cell_size_m).astype(np.int32)
    shape = np.max(cell, axis=0) + 1
    occupied, inverse = np.unique(cell[seed], axis=0, return_inverse=True)
    occupied_labels, sparse_edge_count = _sparse_26_component_labels(occupied, shape)
    labels = np.zeros(len(xyz), dtype=np.int32)
    labels[seed] = occupied_labels[inverse]
    sparse_estimated_working_bytes = _estimate_sparse_component_working_bytes(
        point_count=len(xyz),
        occupied_cells=occupied,
        inverse=inverse,
        sparse_edge_count=sparse_edge_count,
    )
    _record_component_grid_diagnostics(
        diagnostics,
        cell_size_m=cell_size_m,
        shape=shape,
        seed_point_count=int(np.count_nonzero(seed)),
        occupied_cell_count=len(occupied),
        sparse_edge_count=sparse_edge_count,
        sparse_estimated_working_bytes=sparse_estimated_working_bytes,
    )
    return labels, cell, origin


def _dense_grid_component_labels_reference(
    xyz: np.ndarray,
    seed: np.ndarray,
    cell_size_m: float,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Guarded legacy reference used only by small equivalence tests.

    It must never be used in production: the guard rejects any scene whose
    dense integer label grid would exceed the conservative memory budget.
    """

    del diagnostics
    try:
        from scipy.ndimage import generate_binary_structure, label
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise FastResearchError("FAST_V3_SCIPY_UNAVAILABLE") from exc
    if len(xyz) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.zeros(3, dtype=np.float64),
        )
    origin = np.min(xyz, axis=0)
    if seed.shape != (len(xyz),) or not np.any(seed):
        return np.zeros(len(xyz), dtype=np.int32), np.empty((0, 3), dtype=np.int32), origin
    cell = np.floor((xyz - origin) / cell_size_m).astype(np.int32)
    shape = np.max(cell, axis=0) + 1
    _require_dense_component_grid_safe(shape)
    occupied = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    occupied[tuple(cell[seed].T)] = True
    labels_grid, _count = label(occupied, structure=generate_binary_structure(3, 3))
    labels = np.zeros(len(xyz), dtype=np.int32)
    labels[seed] = labels_grid[tuple(cell[seed].T)]
    return labels, cell, origin


def _sparse_26_component_labels(
    occupied: np.ndarray, shape: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return dense-compatible labels over lexicographically sorted occupied cells."""

    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise FastResearchError("FAST_V3_SCIPY_UNAVAILABLE") from exc
    count = len(occupied)
    if count == 0:
        return np.empty(0, dtype=np.int32), 0
    keys = _pack_sparse_cell_keys(occupied, shape)
    if keys is None:
        return _sparse_26_component_labels_dict(occupied)

    edge_rows: list[np.ndarray] = []
    edge_columns: list[np.ndarray] = []
    shape64 = np.asarray(shape, dtype=np.int64)
    occupied64 = np.asarray(occupied, dtype=np.int64)
    for offset in _NEIGHBOR_OFFSETS_26_HALF:
        target = occupied64 + np.asarray(offset, dtype=np.int64)
        valid = np.all((target >= 0) & (target < shape64), axis=1)
        if not np.any(valid):
            continue
        target_keys = _pack_sparse_cell_keys(target[valid], shape)
        if target_keys is None:  # pragma: no cover - protected by the source-key check
            return _sparse_26_component_labels_dict(occupied)
        positions = np.searchsorted(keys, target_keys)
        found = positions < count
        found[found] = keys[positions[found]] == target_keys[found]
        if not np.any(found):
            continue
        source_rows = np.flatnonzero(valid)[found].astype(np.int32, copy=False)
        edge_rows.append(source_rows)
        edge_columns.append(positions[found].astype(np.int32, copy=False))
    if edge_rows:
        row = np.concatenate(edge_rows)
        column = np.concatenate(edge_columns)
        graph = coo_matrix(
            (np.ones(len(row), dtype=np.uint8), (row, column)),
            shape=(count, count),
            dtype=np.uint8,
        ).tocsr()
        sparse_edge_count = int(len(row))
    else:
        graph = coo_matrix((count, count), dtype=np.uint8).tocsr()
        sparse_edge_count = 0
    component_count, component = connected_components(graph, directed=False, return_labels=True)
    first_cell_by_component = np.full(component_count, count, dtype=np.int64)
    np.minimum.at(first_cell_by_component, component, np.arange(count, dtype=np.int64))
    component_order = np.argsort(first_cell_by_component, kind="stable")
    dense_compatible = np.empty(component_count, dtype=np.int32)
    dense_compatible[component_order] = np.arange(1, component_count + 1, dtype=np.int32)
    return dense_compatible[component], sparse_edge_count


def _sparse_26_component_labels_dict(occupied: np.ndarray) -> tuple[np.ndarray, int]:
    """Overflow-safe sparse fallback; each lookup is one of 26 voxel neighbours."""

    lookup = {tuple(int(value) for value in cell): index for index, cell in enumerate(occupied)}
    labels = np.zeros(len(occupied), dtype=np.int32)
    component_id = 0
    edge_count = 0
    all_offsets = tuple(
        (dx, dy, dz)
        for dx in range(-1, 2)
        for dy in range(-1, 2)
        for dz in range(-1, 2)
        if (dx, dy, dz) != (0, 0, 0)
    )
    for start, _cell in enumerate(occupied):
        if labels[start]:
            continue
        component_id += 1
        labels[start] = component_id
        stack = [start]
        while stack:
            current = stack.pop()
            x, y, z = (int(value) for value in occupied[current])
            for dx, dy, dz in all_offsets:
                neighbor = lookup.get((x + dx, y + dy, z + dz))
                if neighbor is None:
                    continue
                if current < neighbor:
                    edge_count += 1
                if labels[neighbor] == 0:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
    return labels, edge_count


def _pack_sparse_cell_keys(cells: np.ndarray, shape: np.ndarray) -> np.ndarray | None:
    """Pack non-negative relative coordinates without relying on global allocation."""

    sx, sy, sz = (int(value) for value in shape)
    maximum = np.iinfo(np.int64).max
    if sy <= 0 or sz <= 0 or sx > maximum // sy // sz:
        return None
    values = np.asarray(cells, dtype=np.int64)
    return (values[:, 0] * sy + values[:, 1]) * sz + values[:, 2]


def _projected_dense_component_label_bytes(shape: np.ndarray) -> int:
    dimensions = tuple(int(value) for value in shape)
    if not dimensions or any(value <= 0 for value in dimensions):
        return 0
    return math.prod(dimensions) * np.dtype(np.intp).itemsize


def _require_dense_component_grid_safe(shape: np.ndarray) -> int:
    projected = _projected_dense_component_label_bytes(shape)
    if projected > _DENSE_COMPONENT_LABEL_MAX_BYTES:
        raise FastResearchError(
            "FAST_V3_DENSE_COMPONENT_GRID_FORBIDDEN: "
            f"projected_label_bytes={projected} limit={_DENSE_COMPONENT_LABEL_MAX_BYTES}"
        )
    return projected


def _estimate_sparse_component_working_bytes(
    *,
    point_count: int,
    occupied_cells: np.ndarray,
    inverse: np.ndarray,
    sparse_edge_count: int,
) -> int:
    """Conservative O(occupied) working-memory estimate for diagnostics only."""

    occupied_count = len(occupied_cells)
    coordinate_bytes = int(occupied_cells.nbytes) + point_count * 3 * np.dtype(np.int32).itemsize
    label_bytes = (
        point_count * np.dtype(np.int32).itemsize
        + occupied_count * 2 * np.dtype(np.int32).itemsize
    )
    key_bytes = occupied_count * np.dtype(np.int64).itemsize
    inverse_bytes = int(inverse.nbytes)
    sparse_graph_bytes = sparse_edge_count * 2 * np.dtype(np.int32).itemsize * 3
    return coordinate_bytes + label_bytes + key_bytes + inverse_bytes + sparse_graph_bytes


def _record_component_grid_diagnostics(
    diagnostics: dict[str, Any] | None,
    *,
    cell_size_m: float,
    shape: np.ndarray | None,
    seed_point_count: int,
    occupied_cell_count: int,
    sparse_edge_count: int,
    sparse_estimated_working_bytes: int,
) -> None:
    if diagnostics is None:
        return
    dense_shape = None if shape is None else [int(value) for value in shape]
    dense_label_bytes = 0 if shape is None else _projected_dense_component_label_bytes(shape)
    diagnostics.update(
        {
            "representation": "SPARSE_OCCUPIED_VOXELS_26_CONNECTED",
            "connectivity": "26_CONNECTED",
            "cell_size_m": float(cell_size_m),
            "dense_shape_reference_only": dense_shape,
            "dense_cell_count_reference_only": (
                0 if shape is None else int(math.prod(int(value) for value in shape))
            ),
            "dense_label_bytes_reference_only": dense_label_bytes,
            "dense_label_gib_reference_only": dense_label_bytes / 1024**3,
            "seed_point_count": int(seed_point_count),
            "occupied_cell_count": int(occupied_cell_count),
            "sparse_edge_count": int(sparse_edge_count),
            "sparse_estimated_working_bytes": int(sparse_estimated_working_bytes),
            "sparse_estimated_working_mib": sparse_estimated_working_bytes / 1024**2,
        }
    )


def _diagnostic_states(
    *,
    ground_mask: np.ndarray,
    exact_wire: np.ndarray,
    wire_geometry: np.ndarray,
    rigid_linear: np.ndarray,
    rigid_surface: np.ndarray,
    retained: np.ndarray,
) -> np.ndarray:
    count = len(retained)
    state = np.full(
        count, int(FastV3DiagnosticState.NO_POSITIVE_VEGETATION_EVIDENCE), dtype=np.uint8
    )
    state[retained] = int(FastV3DiagnosticState.VEGETATION_RETAINED)
    state[rigid_surface] = int(FastV3DiagnosticState.RIGID_SURFACE_EXCLUDED)
    state[rigid_linear] = int(FastV3DiagnosticState.RIGID_LINEAR_EXCLUDED)
    state[wire_geometry] = int(FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED)
    state[exact_wire] = int(FastV3DiagnosticState.EXACT_WIRE_EXCLUDED)
    state[ground_mask] = int(FastV3DiagnosticState.GROUND_EXCLUDED)
    return state


def _positive_v1_admission(
    *,
    v2_retained: np.ndarray,
    positive_v1_support: np.ndarray,
    ground_mask: np.ndarray,
    exact_wire: np.ndarray,
    wire_geometry: np.ndarray,
    rigid_linear: np.ndarray,
    rigid_surface: np.ndarray,
) -> np.ndarray:
    """Keep only positively supported V2 candidates outside every hard exclusion."""
    return (
        v2_retained
        & positive_v1_support
        & ~ground_mask
        & ~exact_wire
        & ~wire_geometry
        & ~rigid_linear
        & ~rigid_surface
    )


def _assert_v3_selection(
    *,
    state: np.ndarray,
    retained: np.ndarray,
    ground_mask: np.ndarray,
    exact_wire: np.ndarray,
    wire_geometry: np.ndarray,
    rigid_linear: np.ndarray,
    rigid_surface: np.ndarray,
) -> None:
    forbidden = ground_mask | exact_wire | wire_geometry | rigid_linear | rigid_surface
    if np.any(retained & forbidden):
        raise FastResearchError("FAST_V3_EXCLUSION_LEAK")
    if np.any(state[retained] != int(FastV3DiagnosticState.VEGETATION_RETAINED)):
        raise FastResearchError("FAST_V3_RETAINED_STATE_INVALID")


def _load_v3_cache(
    manifest_path: Path, clean_npz: Path, diagnostics_npz: Path, config_digest: str
) -> dict[str, Any] | None:
    if not manifest_path.is_file() or not clean_npz.is_file() or not diagnostics_npz.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text("utf-8"))
        vegetation = _fast_input_from_dict(payload["vegetation_input"])
        if (
            payload.get("schema_version") != FAST_V3_SCHEMA_VERSION
            or payload.get("non_certified_marker") != FAST_V3_MARKER
            or payload.get("config_digest") != config_digest
            or vegetation.output_npz_sha256 != sha256_file(clean_npz)
            or payload.get("diagnostics_npz_sha256") != sha256_file(diagnostics_npz)
        ):
            return None
        _load_diagnostics_arrays(payload)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, FastResearchError):
        return None
    return payload


def _load_diagnostics_arrays(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(manifest["diagnostics_npz_path"]))
    if not path.is_file() or sha256_file(path) != manifest["diagnostics_npz_sha256"]:
        raise FastResearchError("FAST_V3_DIAGNOSTICS_INVALID")
    try:
        with np.load(path, allow_pickle=False) as stream:
            values = {name: np.asarray(stream[name]) for name in stream.files}
    except (OSError, ValueError, KeyError) as exc:
        raise FastResearchError("FAST_V3_DIAGNOSTICS_INVALID") from exc
    count = len(values.get("original_index", ()))
    required = (
        "xyz_local_m",
        "hag_m",
        "diagnostic_state",
        "positive_v1_support_mask",
        "nearest_w1_distance_m",
        "tangent_alignment",
        "wire_component_id",
        "rigid_linear_component_id",
        "rigid_surface_component_id",
        "linearity",
        "planarity",
        "scattering",
    )
    if any(name not in values or values[name].shape[0] != count for name in required):
        raise FastResearchError("FAST_V3_DIAGNOSTICS_SHAPE_INVALID")
    return values


def _fast_input_from_dict(payload: Any) -> FastVegetationInputResult:
    if not isinstance(payload, dict):
        raise FastResearchError("FAST_V3_CACHE_INVALID")
    return FastVegetationInputResult.from_dict(payload)


def _longest_contiguous_run(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    breaks = np.flatnonzero(np.diff(values) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(values)]
    return int(np.max(ends - starts))


def _quantiles(values: np.ndarray) -> list[float] | None:
    if len(values) == 0:
        return None
    return [float(value) for value in np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))]


def _even_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]


def _point_component_type(values: dict[str, np.ndarray], position: int) -> tuple[str, int | None]:
    for name, label in (
        ("wire_component_id", "WIRE_GEOMETRY"),
        ("rigid_linear_component_id", "RIGID_LINEAR"),
        ("rigid_surface_component_id", "RIGID_SURFACE"),
    ):
        component_id = int(values[name][position])
        if component_id >= 0:
            return label, component_id
    return "NONE", None


def _finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _write_html_atomic(figure: Any, output_html: Path) -> Path:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_html.with_name(output_html.name + ".tmp")
    figure.write_html(temporary, include_plotlyjs=True, full_html=True)
    os.replace(temporary, output_html)
    return output_html
