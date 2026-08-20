"""FAST-only vegetation A/B evidence and clean V4-admission preparation.

This module is deliberately outside the formal FGC/FRV authority chain.  It
uses compiled PDAL covariance features on the already explicit FAST candidate
set, applies only exact-source membership exclusions, and publishes a separate
non-certified V4 input.  It never changes V4, W1, or TreeWire semantics.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
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

FAST_VEGETATION_AB_SCHEMA_VERSION = "fast-vegetation-ab-v1"
PLAN_B_BACKEND = "PDAL filters.covariancefeatures (KNN=16, Dimensionality)"
PLAN_B_CLASSIFIER = "scattering_ge_0.060_with_exact_hard_exclusions"


@dataclass(frozen=True, slots=True)
class FastVegetationEvidenceConfig:
    """Fixed, deterministic Plan-B configuration.

    The scattering gate is intentionally conservative: on the supplied
    exact-index V1 reference subset it retains the overwhelming majority of
    V1 vegetation evidence, while planar/linear low-scattering structures are
    excluded.  Confirmed W1 members are always excluded independently of this
    soft geometry evidence.
    """

    knn: int = 16
    minimum_scattering: float = 0.06

    def __post_init__(self) -> None:
        if type(self.knn) is not int or self.knn < 3:
            raise FastResearchError("FAST_VEGETATION_CONFIG_INVALID_KNN")
        if not math.isfinite(self.minimum_scattering) or not (
            0.0 <= self.minimum_scattering <= 1.0
        ):
            raise FastResearchError("FAST_VEGETATION_CONFIG_INVALID_SCATTERING")

    def to_dict(self) -> dict[str, Any]:
        return {
            "knn": self.knn,
            "minimum_scattering": self.minimum_scattering,
            "classifier": PLAN_B_CLASSIFIER,
        }


@dataclass(frozen=True, slots=True)
class FastVegetationCleanSelection:
    """In-memory exact-membership selection used before V4 handoff."""

    retained_mask: np.ndarray
    ground_mask: np.ndarray
    known_wire_mask: np.ndarray
    vegetation_evidence_mask: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.retained_mask)
        for name in (
            "retained_mask",
            "ground_mask",
            "known_wire_mask",
            "vegetation_evidence_mask",
        ):
            value = getattr(self, name)
            if value.dtype != bool or value.shape != (count,):
                raise FastResearchError(f"FAST_VEGETATION_SELECTION_INVALID_{name.upper()}")
        expected = ~self.ground_mask & ~self.known_wire_mask & self.vegetation_evidence_mask
        if not np.array_equal(self.retained_mask, expected):
            raise FastResearchError("FAST_VEGETATION_SELECTION_FORMULA_MISMATCH")
        if np.any(self.retained_mask & (self.ground_mask | self.known_wire_mask)):
            raise FastResearchError("FAST_VEGETATION_HARD_EXCLUSION_LEAK")


def fast_vegetation_config_digest(
    candidate: FastVegetationInputResult,
    ground: FastGroundResult,
    *,
    w1_npz: Path,
    w1_json: Path,
    reference_a2a_npz: Path | None,
    reference_v1_npz: Path | None,
    config: FastVegetationEvidenceConfig,
) -> str:
    """Return a cache key that changes with every evidence or source input."""

    return canonical_digest(
        {
            "schema_version": FAST_VEGETATION_AB_SCHEMA_VERSION,
            "candidate_npz_sha256": candidate.output_npz_sha256,
            "ground_result_digest": fast_ground_content_digest(ground),
            "w1_npz_sha256": sha256_file(w1_npz),
            "w1_json_sha256": sha256_file(w1_json),
            "reference_a2a_npz_sha256": (
                sha256_file(reference_a2a_npz)
                if reference_a2a_npz is not None and reference_a2a_npz.is_file()
                else None
            ),
            "reference_v1_npz_sha256": (
                sha256_file(reference_v1_npz)
                if reference_v1_npz is not None and reference_v1_npz.is_file()
                else None
            ),
            "config": config.to_dict(),
        }
    )


def classify_plan_b_geometry(
    scattering: np.ndarray, *, config: FastVegetationEvidenceConfig
) -> np.ndarray:
    """Return the deterministic volumetric-evidence gate without Python KNN loops."""

    values = np.asarray(scattering)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise FastResearchError("FAST_VEGETATION_GEOMETRY_INVALID")
    return np.asarray(values >= config.minimum_scattering, dtype=bool)


def benchmark_plan_b_representative(
    xyz_local_m: np.ndarray,
    *,
    config: FastVegetationEvidenceConfig,
    maximum_points: int = 100_000,
) -> dict[str, Any]:
    """Time the compiled Plan-B backend on a deterministic 50k--200k subset."""

    if type(maximum_points) is not int or not 50_000 <= maximum_points <= 200_000:
        raise FastResearchError("FAST_VEGETATION_REPRESENTATIVE_SIZE_INVALID")
    xyz = np.asarray(xyz_local_m, dtype="<f8")
    if len(xyz) == 0:
        raise FastResearchError("FAST_VEGETATION_REPRESENTATIVE_EMPTY")
    positions = _even_display_indices(len(xyz), maximum_points)
    _features, elapsed = _pdal_covariance_features(xyz[positions], config=config)
    return {
        "point_count": int(len(positions)),
        "selection": "deterministic_even_rows_from_fast_admission",
        "runtime_seconds": elapsed,
        "backend": PLAN_B_BACKEND,
    }


def probe_plan_a_open3d_ml() -> dict[str, Any]:
    """Perform a no-install eligibility probe for the time-boxed Plan A route."""

    started = time.perf_counter()
    try:
        import open3d
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        return {
            "status": "ENVIRONMENT_BLOCKED",
            "backend": "Open3D-ML pretrained semantic route (Toronto3D-class model family)",
            "model_dataset": "NOT_RUN: Open3D unavailable",
            "license_status": "Open3D-ML model/data-weight reuse not verified",
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"Open3D import failed: {exc}",
        }
    try:
        import open3d.ml.torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific dependency
        return {
            "status": "ENVIRONMENT_BLOCKED",
            "backend": "Open3D-ML pretrained semantic route (Toronto3D-class model family)",
            "model_dataset": "NOT_RUN: PyTorch Open3D-ML semantic backend unavailable",
            "open3d_version": open3d.__version__,
            "license_status": (
                "Open3D code is MIT; pretrained model/data-weight reuse license was not "
                "verified in this timebox"
            ),
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"open3d.ml.torch import failed: {type(exc).__name__}: {exc}",
        }
    return {
        "status": "FAIL",
        "backend": "Open3D-ML pretrained semantic route (Toronto3D-class model family)",
        "model_dataset": "Backend import passed, but no compatible pretrained model was run",
        "open3d_version": open3d.__version__,
        "license_status": "Model/data-weight reuse license requires separate verification",
        "runtime_seconds": time.perf_counter() - started,
        "reason": "Backend import passed but this probe does not claim semantic accuracy",
    }


def apply_fast_vegetation_hard_exclusions(
    *,
    classification: np.ndarray,
    original_index: np.ndarray,
    known_wire_original_index: np.ndarray,
    vegetation_evidence_mask: np.ndarray,
) -> FastVegetationCleanSelection:
    """Apply the required exact formula before V4.

    Membership is exclusively by ``original_index``.  No XYZ proximity,
    rounding, or fuzzy correspondence is accepted for W1 members.
    """

    indices = np.asarray(original_index, dtype="<i8")
    classes = np.asarray(classification)
    evidence = np.asarray(vegetation_evidence_mask, dtype=bool)
    if (
        indices.ndim != 1
        or classes.shape != indices.shape
        or evidence.shape != indices.shape
        or np.any(indices < 0)
        or (len(indices) > 1 and not np.all(np.diff(indices) > 0))
    ):
        raise FastResearchError("FAST_VEGETATION_CORRESPONDENCE_INVALID")
    wire_indices = np.asarray(known_wire_original_index, dtype="<i8")
    if wire_indices.ndim != 1 or np.any(wire_indices < 0):
        raise FastResearchError("FAST_VEGETATION_W1_MEMBERSHIP_INVALID")
    ground_mask = np.asarray(classes == 2, dtype=bool)
    known_wire_mask = np.isin(indices, np.unique(wire_indices), assume_unique=False)
    retained_mask = ~ground_mask & ~known_wire_mask & evidence
    return FastVegetationCleanSelection(
        retained_mask=retained_mask,
        ground_mask=ground_mask,
        known_wire_mask=known_wire_mask,
        vegetation_evidence_mask=evidence,
    )


def build_fast_vegetation_ab_input(
    candidate: FastVegetationInputResult,
    ground: FastGroundResult,
    output_root: Path,
    *,
    w1_npz: Path,
    w1_json: Path,
    reference_a2a_npz: Path | None,
    reference_v1_npz: Path | None,
    config: FastVegetationEvidenceConfig | None = None,
) -> tuple[str, FastVegetationInputResult, dict[str, Any]]:
    """Build/reuse a clean FAST V4 input and its transparent Plan-B evidence.

    ``candidate`` is an existing non-certified, HAG-ready admission set.  This
    function re-reads the classified HAG LAS at those exact original indices so
    that the pre-V4 ground-overlap assertion is evidenced rather than assumed.
    """

    config = config or FastVegetationEvidenceConfig()
    output_root = Path(output_root).resolve()
    candidate_npz = Path(candidate.output_npz_path)
    if not candidate_npz.is_file() or sha256_file(candidate_npz) != candidate.output_npz_sha256:
        raise FastResearchError("FAST_VEGETATION_CANDIDATE_INPUT_INVALID")
    w1_npz, w1_json = Path(w1_npz).resolve(), Path(w1_json).resolve()
    if not w1_npz.is_file() or not w1_json.is_file():
        raise FastResearchError("FAST_VEGETATION_W1_INPUT_MISSING")
    if ground.non_certified_marker == "":
        raise FastResearchError("FAST_VEGETATION_GROUND_RESULT_INVALID")
    digest = fast_vegetation_config_digest(
        candidate,
        ground,
        w1_npz=w1_npz,
        w1_json=w1_json,
        reference_a2a_npz=reference_a2a_npz,
        reference_v1_npz=reference_v1_npz,
        config=config,
    )
    output_npz = output_root / "fast_v4_input.npz"
    evidence_npz = output_root / "plan_b_geometry_features.npz"
    manifest_path = output_root / "fast_vegetation_cleaning.json"
    cached = _load_clean_cache(manifest_path, output_npz, evidence_npz, digest)
    if cached is not None:
        vegetation = _fast_input_from_dict(cached["vegetation_input"])
        return "SKIPPED_FROM_CACHE", vegetation, cached

    started = time.perf_counter()
    with np.load(candidate_npz, allow_pickle=False) as stream:
        original_index = np.asarray(stream["original_index"], dtype="<i8")
        xyz_local_m = np.asarray(stream["xyz_local_m"], dtype="<f8")
        hag_m = np.asarray(stream["hag_m"], dtype="<f4")
    validate_fast_v4_arrays(original_index, xyz_local_m, hag_m)
    classification = _classifications_at_original_indices(
        Path(ground.output_las_path), original_index
    )
    _w1_result, w1_arrays = load_wire_models(w1_npz, w1_json)
    known_wire_indices = np.asarray(w1_arrays["ordered_input_original_index"], dtype="<i8")
    if len(np.unique(known_wire_indices)) != len(known_wire_indices):
        raise FastResearchError("FAST_VEGETATION_W1_MEMBERSHIP_NOT_UNIQUE")
    features, feature_elapsed = _pdal_covariance_features(xyz_local_m, config=config)
    reference = _load_reference_v1(
        original_index,
        reference_a2a_npz=reference_a2a_npz,
        reference_v1_npz=reference_v1_npz,
    )
    evidence = classify_plan_b_geometry(features["scattering"], config=config)
    selection = apply_fast_vegetation_hard_exclusions(
        classification=classification,
        original_index=original_index,
        known_wire_original_index=known_wire_indices,
        vegetation_evidence_mask=evidence,
    )
    _assert_pre_v4_clean(selection)
    retained = selection.retained_mask
    clean_original = original_index[retained]
    clean_xyz = xyz_local_m[retained]
    clean_hag = hag_m[retained]
    validate_fast_v4_arrays(clean_original, clean_xyz, clean_hag)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_npz_atomic(
        output_npz,
        original_index=clean_original,
        xyz_local_m=clean_xyz,
        hag_m=clean_hag,
        candidate_reason_flags=np.zeros(len(clean_original), dtype=np.uint32),
    )
    _write_npz_atomic(
        evidence_npz,
        original_index=original_index,
        classification=np.asarray(classification, dtype=np.uint8),
        known_wire_exact_mask=selection.known_wire_mask,
        vegetation_evidence_mask=selection.vegetation_evidence_mask,
        retained_mask=selection.retained_mask,
        linearity=np.asarray(features["linearity"], dtype="<f4"),
        planarity=np.asarray(features["planarity"], dtype="<f4"),
        scattering=np.asarray(features["scattering"], dtype="<f4"),
        verticality=np.asarray(features["verticality"], dtype="<f4"),
        reference_a2a_v1_match_mask=reference["match_mask"],
        reference_v1_vegetation_mask=reference["vegetation_mask"],
    )
    vegetation = FastVegetationInputResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_INPUT_MARKER,
        ground_result_digest=fast_ground_content_digest(ground),
        config_digest=digest,
        output_npz_path=str(output_npz),
        output_npz_sha256=sha256_file(output_npz),
        point_count=int(len(clean_original)),
        source_id=candidate.source_id,
        source_las_path=candidate.source_las_path,
        source_point_count=candidate.source_point_count,
        source_file_sha256=candidate.source_file_sha256,
        origin_context=candidate.origin_context,
        selected_bounds_min_xyz=candidate.selected_bounds_min_xyz,
        selected_bounds_max_xyz=candidate.selected_bounds_max_xyz,
        minimum_hag_m=candidate.minimum_hag_m,
        admission_voxel_size_m=candidate.admission_voxel_size_m,
        original_index_sha256=_array_digest(clean_original),
        xyz_sha256=_array_digest(clean_xyz),
        hag_sha256=_array_digest(clean_hag),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json_atomic(output_root / "fast_v4_input.json", vegetation.to_dict())
    reference_match = reference["match_mask"]
    v1_vegetation = reference["vegetation_mask"]
    agreement_denominator = int(np.count_nonzero(reference_match & v1_vegetation))
    agreement_numerator = int(
        np.count_nonzero(reference_match & v1_vegetation & selection.vegetation_evidence_mask)
    )
    manifest = {
        "schema_version": FAST_VEGETATION_AB_SCHEMA_VERSION,
        "non_certified_marker": NON_CERTIFIED_INPUT_MARKER,
        "config_digest": digest,
        "plan_b_backend": PLAN_B_BACKEND,
        "plan_b_classifier": PLAN_B_CLASSIFIER,
        "config": config.to_dict(),
        "formula": "NON_GROUND AND NOT_KNOWN_WIRE AND VEGETATION_EVIDENCE_PASS",
        "correspondence": {
            "method": "exact_original_index_only",
            "fuzzy_xyz_matching_used": False,
            "candidate_original_index_sha256": candidate.original_index_sha256,
            "w1_known_wire_member_count": int(len(known_wire_indices)),
        },
        "counts": {
            "candidate_input_count": int(len(original_index)),
            "candidate_ground_overlap_before_clean": int(np.count_nonzero(selection.ground_mask)),
            "ground_excluded_source_count": int(ground.ground_class_2_count),
            "known_wire_excluded_count": int(np.count_nonzero(selection.known_wire_mask)),
            "vegetation_evidence_rejected_count": int(
                np.count_nonzero(~selection.vegetation_evidence_mask)
            ),
            "other_non_vegetation_excluded_count": int(
                np.count_nonzero(
                    ~selection.ground_mask
                    & ~selection.known_wire_mask
                    & ~selection.vegetation_evidence_mask
                )
            ),
            "final_vegetation_retained_count": int(np.count_nonzero(selection.retained_mask)),
            "final_tree_ground_overlap": 0,
            "final_tree_known_wire_overlap": 0,
        },
        "reference_a2a_v1": {
            "a2a_path": None if reference_a2a_npz is None else str(reference_a2a_npz),
            "v1_path": None if reference_v1_npz is None else str(reference_v1_npz),
            "exact_match_count": int(np.count_nonzero(reference_match)),
            "v1_vegetation_reference_count": agreement_denominator,
            "plan_b_retained_v1_vegetation_count": agreement_numerator,
            "plan_b_retained_v1_vegetation_fraction": (
                None if agreement_denominator == 0 else agreement_numerator / agreement_denominator
            ),
        },
        "runtime_seconds": vegetation.elapsed_seconds,
        "feature_runtime_seconds": feature_elapsed,
        "vegetation_input": vegetation.to_dict(),
        "output_npz_sha256": vegetation.output_npz_sha256,
        "feature_npz_path": str(evidence_npz),
        "feature_npz_sha256": sha256_file(evidence_npz),
    }
    _write_json_atomic(manifest_path, manifest)
    return "PASS", vegetation, manifest


def generate_plan_b_geometry_review(manifest: dict[str, Any], output_html: Path) -> Path:
    """Render an inspectable Plan-B feature/evidence review from the NPZ."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise FastResearchError("FAST_VEGETATION_REVIEW_DEPENDENCY_MISSING") from exc
    feature_path = Path(str(manifest["feature_npz_path"]))
    if not feature_path.is_file() or sha256_file(feature_path) != manifest["feature_npz_sha256"]:
        raise FastResearchError("FAST_VEGETATION_REVIEW_FEATURE_INVALID")
    with np.load(feature_path, allow_pickle=False) as stream:
        original = np.asarray(stream["original_index"], dtype="<i8")
        scattering = np.asarray(stream["scattering"], dtype="<f4")
        linearity = np.asarray(stream["linearity"], dtype="<f4")
        planarity = np.asarray(stream["planarity"], dtype="<f4")
        wires = np.asarray(stream["known_wire_exact_mask"], dtype=bool)
        retained = np.asarray(stream["retained_mask"], dtype=bool)
        reference = np.asarray(stream["reference_a2a_v1_match_mask"], dtype=bool)
        reference_vegetation = np.asarray(stream["reference_v1_vegetation_mask"], dtype=bool)
    order = _even_display_indices(len(original), 80_000)
    figure = go.Figure()
    groups = (
        ("retained geometry evidence", retained, "#228b22"),
        ("exact known W1 members (excluded)", wires, "#d62728"),
        ("geometry-rejected non-wire", ~retained & ~wires, "#7f7f7f"),
        (
            "exact-index V1 vegetation reference",
            reference & reference_vegetation,
            "#1f77b4",
        ),
    )
    for name, mask, color in groups:
        selected = order[mask[order]]
        if len(selected):
            figure.add_trace(
                go.Scattergl(
                    x=linearity[selected],
                    y=scattering[selected],
                    mode="markers",
                    name=name,
                    marker={"size": 4, "color": color, "opacity": 0.55},
                    customdata=np.column_stack((original[selected], planarity[selected])),
                    hovertemplate=(
                        "original_index=%{customdata[0]}<br>linearity=%{x:.3f}"
                        "<br>scattering=%{y:.3f}<br>planarity=%{customdata[1]:.3f}<extra></extra>"
                    ),
                )
            )
    threshold = manifest["config"]["minimum_scattering"]
    figure.add_hline(y=threshold, line_dash="dash", line_color="#111827")
    figure.update_layout(
        title=(
            "FAST Plan B — PDAL covariance geometry (exact W1 hard exclusion; "
            "V1 shown only as exact-index reference)"
        ),
        xaxis_title="PDAL Linearity",
        yaxis_title="PDAL Scattering",
        legend={"orientation": "h"},
        margin={"l": 55, "r": 20, "t": 60, "b": 50},
    )
    return _write_html_atomic(figure, output_html)


def generate_fast_vegetation_review(
    vegetation: FastVegetationInputResult, manifest: dict[str, Any], output_html: Path
) -> Path:
    """Render the retained FAST vegetation set that is eligible for V4."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise FastResearchError("FAST_VEGETATION_REVIEW_DEPENDENCY_MISSING") from exc
    npz_path = Path(vegetation.output_npz_path)
    if not npz_path.is_file() or sha256_file(npz_path) != vegetation.output_npz_sha256:
        raise FastResearchError("FAST_VEGETATION_REVIEW_INPUT_INVALID")
    with np.load(npz_path, allow_pickle=False) as stream:
        original = np.asarray(stream["original_index"], dtype="<i8")
        xyz = np.asarray(stream["xyz_local_m"], dtype="<f8")
        hag = np.asarray(stream["hag_m"], dtype="<f4")
    sample = _even_display_indices(len(original), 80_000)
    figure = go.Figure(
        go.Scatter3d(
            x=xyz[sample, 0],
            y=xyz[sample, 1],
            z=xyz[sample, 2],
            mode="markers",
            name="V4-eligible vegetation evidence",
            marker={"size": 1.8, "color": hag[sample], "colorscale": "Viridis", "opacity": 0.7},
            customdata=original[sample],
            hovertemplate=(
                "original_index=%{customdata}<br>HAG=%{marker.color:.2f} m<extra></extra>"
            ),
        )
    )
    counts = manifest["counts"]
    figure.update_layout(
        title=(
            "FAST vegetation input v2 — NON_GROUND AND NOT_KNOWN_WIRE AND "
            "VEGETATION_EVIDENCE_PASS "
            f"(retained={counts['final_vegetation_retained_count']:,})"
        ),
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    return _write_html_atomic(figure, output_html)


def generate_minimum_clearance_local_review(clearance: Any, output_html: Path) -> Path:
    """Render the exact closest pair for mandatory owner inspection.

    This is a local evidence viewer, not a reimplementation of TreeWire
    distance semantics.  It uses only coordinates already published by the
    frozen clearance result.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise FastResearchError("FAST_VEGETATION_REVIEW_DEPENDENCY_MISSING") from exc
    valid = [record for record in clearance.records if record.minimum_clearance_m is not None]
    if not valid:
        raise FastResearchError("FAST_VEGETATION_CLEARANCE_REVIEW_NO_VALID_RECORD")
    record = min(valid, key=lambda item: float(item.minimum_clearance_m))
    tree = np.asarray(record.nearest_tree_xyz_local_m, dtype=np.float64)
    wire = np.asarray(record.nearest_wire_xyz_local_m, dtype=np.float64)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=[tree[0], wire[0]],
            y=[tree[1], wire[1]],
            z=[tree[2], wire[2]],
            mode="lines+markers+text",
            text=["nearest tree point", "nearest W1 curve point"],
            textposition="top center",
            marker={"size": 6, "color": ["#228b22", "#d62728"]},
            line={"width": 7, "color": "#111827"},
            name="frozen TreeWire closest pair",
            hovertemplate="local (%{x:.4f}, %{y:.4f}, %{z:.4f})<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            "FAST minimum-clearance local inspection — "
            f"tree {record.tree_id}, wire {record.nearest_wire_id}, "
            f"distance {float(record.minimum_clearance_m):.9f} m"
        ),
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    return _write_html_atomic(figure, output_html)


def _pdal_covariance_features(
    xyz_local_m: np.ndarray, *, config: FastVegetationEvidenceConfig
) -> tuple[dict[str, np.ndarray], float]:
    try:
        import pdal
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise FastResearchError("FAST_VEGETATION_PDAL_BACKEND_UNAVAILABLE") from exc
    xyz = np.asarray(xyz_local_m, dtype="<f8")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise FastResearchError("FAST_VEGETATION_PDAL_INPUT_INVALID")
    source = np.empty(len(xyz), dtype=[("X", "<f8"), ("Y", "<f8"), ("Z", "<f8")])
    source["X"], source["Y"], source["Z"] = xyz.T
    definition = {
        "pipeline": [
            {
                "type": "filters.covariancefeatures",
                "knn": config.knn,
                "feature_set": "Dimensionality",
            }
        ]
    }
    started = time.perf_counter()
    try:
        pipeline = pdal.Pipeline(json.dumps(definition), arrays=[source])
        count = int(pipeline.execute())
        result = pipeline.arrays[0]
    except Exception as exc:  # pragma: no cover - native backend failure
        raise FastResearchError(f"FAST_VEGETATION_PDAL_FEATURES_FAILED: {exc}") from exc
    elapsed = time.perf_counter() - started
    if count != len(xyz):
        raise FastResearchError("FAST_VEGETATION_PDAL_POINT_COUNT_CHANGED")
    fields = {
        name.lower(): np.asarray(result[name], dtype="<f4")
        for name in ("Linearity", "Planarity", "Scattering", "Verticality")
    }
    if any(
        values.shape != (len(xyz),) or not np.isfinite(values).all()
        for values in fields.values()
    ):
        raise FastResearchError("FAST_VEGETATION_PDAL_FEATURE_OUTPUT_INVALID")
    return fields, elapsed


def _classifications_at_original_indices(las_path: Path, original_index: np.ndarray) -> np.ndarray:
    """Read only the selected row classifications while preserving exact order."""

    import laspy

    if not las_path.is_file():
        raise FastResearchError("FAST_VEGETATION_CLASSIFICATION_SOURCE_MISSING")
    requested = np.asarray(original_index, dtype="<i8")
    if requested.ndim != 1 or (len(requested) > 1 and not np.all(np.diff(requested) > 0)):
        raise FastResearchError("FAST_VEGETATION_CLASSIFICATION_INDEX_INVALID")
    output = np.empty(len(requested), dtype=np.uint8)
    cursor = 0
    offset = 0
    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(500_000):
            upper = offset + len(chunk)
            end = int(np.searchsorted(requested, upper, side="left"))
            if end > cursor:
                positions = requested[cursor:end] - offset
                output[cursor:end] = np.asarray(chunk.classification, dtype=np.uint8)[positions]
                cursor = end
            offset = upper
    if cursor != len(requested):
        raise FastResearchError("FAST_VEGETATION_CLASSIFICATION_INDEX_OUT_OF_RANGE")
    return output


def _load_reference_v1(
    original_index: np.ndarray,
    *,
    reference_a2a_npz: Path | None,
    reference_v1_npz: Path | None,
) -> dict[str, np.ndarray]:
    empty = {
        "match_mask": np.zeros(len(original_index), dtype=bool),
        "vegetation_mask": np.zeros(len(original_index), dtype=bool),
    }
    if reference_a2a_npz is None or reference_v1_npz is None:
        return empty
    a2a_path, v1_path = Path(reference_a2a_npz), Path(reference_v1_npz)
    if not a2a_path.is_file() or not v1_path.is_file():
        return empty
    try:
        with np.load(a2a_path, allow_pickle=False) as stream:
            a2a_original = np.asarray(stream["original_index"], dtype="<i8")
        with np.load(v1_path, allow_pickle=False) as stream:
            v1_original = np.asarray(stream["original_index"], dtype="<i8")
            v1_vegetation = np.asarray(stream["vegetation_mask"], dtype=bool)
    except (OSError, ValueError, KeyError) as exc:
        raise FastResearchError("FAST_VEGETATION_REFERENCE_V1_INVALID") from exc
    if (
        a2a_original.shape != v1_original.shape
        or v1_vegetation.shape != v1_original.shape
        or len(np.unique(a2a_original)) != len(a2a_original)
        or len(np.unique(v1_original)) != len(v1_original)
        or not np.array_equal(a2a_original, v1_original)
    ):
        raise FastResearchError("FAST_VEGETATION_REFERENCE_A2A_V1_CORRESPONDENCE_INVALID")
    order = np.argsort(v1_original)
    sorted_original = v1_original[order]
    positions = np.searchsorted(sorted_original, original_index)
    safe = np.minimum(positions, len(sorted_original) - 1)
    matched = (positions < len(sorted_original)) & (sorted_original[safe] == original_index)
    vegetation = np.zeros(len(original_index), dtype=bool)
    vegetation[matched] = v1_vegetation[order][positions[matched]]
    return {"match_mask": matched, "vegetation_mask": vegetation}


def _assert_pre_v4_clean(selection: FastVegetationCleanSelection) -> None:
    if np.count_nonzero(selection.retained_mask & selection.ground_mask):
        raise FastResearchError("FAST_VEGETATION_PRE_V4_GROUND_OVERLAP")
    if np.count_nonzero(selection.retained_mask & selection.known_wire_mask):
        raise FastResearchError("FAST_VEGETATION_PRE_V4_WIRE_OVERLAP")


def _load_clean_cache(
    manifest_path: Path, output_npz: Path, evidence_npz: Path, config_digest: str
) -> dict[str, Any] | None:
    if not manifest_path.is_file() or not output_npz.is_file() or not evidence_npz.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text("utf-8"))
        vegetation = _fast_input_from_dict(payload["vegetation_input"])
        if (
            payload.get("schema_version") != FAST_VEGETATION_AB_SCHEMA_VERSION
            or payload.get("non_certified_marker") != NON_CERTIFIED_INPUT_MARKER
            or payload.get("config_digest") != config_digest
            or vegetation.output_npz_sha256 != sha256_file(output_npz)
            or payload.get("feature_npz_sha256") != sha256_file(evidence_npz)
        ):
            return None
        with np.load(output_npz, allow_pickle=False) as stream:
            validate_fast_v4_arrays(
                np.asarray(stream["original_index"], dtype="<i8"),
                np.asarray(stream["xyz_local_m"], dtype="<f8"),
                np.asarray(stream["hag_m"], dtype="<f4"),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, FastResearchError):
        return None
    return payload


def _fast_input_from_dict(payload: Any) -> FastVegetationInputResult:
    if not isinstance(payload, dict):
        raise FastResearchError("FAST_VEGETATION_CACHE_INVALID")
    return FastVegetationInputResult.from_dict(payload)


def _even_display_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _write_html_atomic(figure: Any, output_html: Path) -> Path:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_html.with_name(output_html.name + ".tmp")
    figure.write_html(temporary, include_plotlyjs=True, full_html=True)
    os.replace(temporary, output_html)
    return output_html
