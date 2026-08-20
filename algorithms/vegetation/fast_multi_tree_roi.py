"""Deterministic FAST-only search for crown-like multi-tree diagnostic ROIs.

This module deliberately does not invoke V4.  It ranks compact representative
windows from Ground/HAG and local covariance evidence, emits reviewable
candidate metrics, and keeps the eventual V4 decision behind a separate visual
and vegetation-purity gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from algorithms.vegetation.fast_research import FastResearchError

MULTI_TREE_ROI_SCHEMA = "fast-multi-tree-roi-search-v1"
_CROWN_MIN_HAG_M = 1.5
_CROWN_MIN_SCATTERING = 0.18
_CROWN_MAX_LINEARITY = 0.70
_CROWN_MAX_PLANARITY = 0.75
_PEAK_MIN_HAG_M = 4.0


@dataclass(frozen=True, slots=True)
class MultiTreeRoiCandidate:
    """A ranked, non-certified candidate region measured before V4."""

    rank: int
    score: float
    xy_bounds_min: tuple[float, float]
    xy_bounds_max: tuple[float, float]
    representative_count: int
    raw_weight_count: int
    crown_evidence_representative_count: int
    crown_evidence_raw_weight_count: int
    crown_peak_count: int
    crown_peak_xy: tuple[tuple[float, float], ...]
    occupancy_component_count: int
    hag_p05_m: float
    hag_p95_m: float
    hag_range_m: float
    median_linearity: float
    median_planarity: float
    median_scattering: float
    rigid_geometry_fraction: float
    elongation_ratio: float
    horizontal_spread_m: float
    automatic_gate_pass: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["xy_bounds_min"] = list(self.xy_bounds_min)
        payload["xy_bounds_max"] = list(self.xy_bounds_max)
        payload["crown_peak_xy"] = [list(value) for value in self.crown_peak_xy]
        return payload


def rank_multi_tree_roi_candidates(
    *,
    xyz_absolute: np.ndarray,
    hag_m: np.ndarray,
    raw_weight: np.ndarray,
    classification: np.ndarray,
    linearity: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
    window_width_m: float = 12.0,
    window_height_m: float = 18.0,
    stride_m: float = 2.0,
    maximum_candidates: int = 6,
) -> tuple[list[MultiTreeRoiCandidate], int]:
    """Rank low-rigidity windows by independently supported crown-like peaks."""

    arrays = _validated_arrays(
        xyz_absolute=xyz_absolute,
        hag_m=hag_m,
        raw_weight=raw_weight,
        classification=classification,
        linearity=linearity,
        planarity=planarity,
        scattering=scattering,
    )
    if (
        not math.isfinite(window_width_m)
        or not math.isfinite(window_height_m)
        or not math.isfinite(stride_m)
        or window_width_m < 4.0
        or window_height_m < 4.0
        or stride_m <= 0.0
        or maximum_candidates <= 0
    ):
        raise FastResearchError("FAST_MULTI_TREE_ROI_CONFIG_INVALID")

    xyz = arrays["xyz_absolute"]
    hag = arrays["hag_m"]
    raw = arrays["raw_weight"]
    classes = arrays["classification"]
    linear = arrays["linearity"]
    planar = arrays["planarity"]
    scatter = arrays["scattering"]
    source_origin = np.min(xyz[:, :2], axis=0)
    cell_m = 1.0
    grid_xy = np.floor((xyz[:, :2] - source_origin) / cell_m).astype(np.int32)
    cols = int(np.max(grid_xy[:, 0])) + 1
    rows = int(np.max(grid_xy[:, 1])) + 1
    crown_evidence = (
        (classes != 2)
        & (hag >= _CROWN_MIN_HAG_M)
        & (scatter >= _CROWN_MIN_SCATTERING)
        & (linear <= _CROWN_MAX_LINEARITY)
        & (planar <= _CROWN_MAX_PLANARITY)
    )
    eligible_peaks = _crown_peak_records(
        xyz=xyz,
        hag=hag,
        linearity=linear,
        scattering=scatter,
        crown_evidence=crown_evidence,
        grid_xy=grid_xy,
        source_origin=source_origin,
        rows=rows,
        cols=cols,
        cell_m=cell_m,
    )
    width_cells = int(round(window_width_m / cell_m))
    height_cells = int(round(window_height_m / cell_m))
    stride_cells = int(round(stride_m / cell_m))
    if width_cells <= 0 or height_cells <= 0 or stride_cells <= 0:
        raise FastResearchError("FAST_MULTI_TREE_ROI_GRID_CONFIG_INVALID")

    provisional: list[MultiTreeRoiCandidate] = []
    for lower_y in range(0, max(1, rows - height_cells + 1), stride_cells):
        for lower_x in range(0, max(1, cols - width_cells + 1), stride_cells):
            upper_x = min(cols, lower_x + width_cells)
            upper_y = min(rows, lower_y + height_cells)
            bounds_min = source_origin + np.asarray((lower_x, lower_y), dtype=np.float64)
            bounds_max = source_origin + np.asarray((upper_x, upper_y), dtype=np.float64)
            inside = (
                (xyz[:, 0] >= bounds_min[0])
                & (xyz[:, 0] < bounds_max[0])
                & (xyz[:, 1] >= bounds_min[1])
                & (xyz[:, 1] < bounds_max[1])
            )
            evidence = inside & crown_evidence
            if int(np.count_nonzero(evidence)) < 300:
                continue
            result = _candidate_metrics(
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                inside=inside,
                crown_evidence=evidence,
                crown_peaks=eligible_peaks,
                xyz=xyz,
                hag=hag,
                raw_weight=raw,
                linearity=linear,
                planarity=planar,
                scattering=scatter,
                classes=classes,
                grid_xy=grid_xy,
                lower_xy=(lower_x, lower_y),
                width_cells=upper_x - lower_x,
                height_cells=upper_y - lower_y,
            )
            if result is not None:
                provisional.append(result)
    provisional.sort(key=lambda value: (-value.score, value.xy_bounds_min))
    selected: list[MultiTreeRoiCandidate] = []
    for candidate in provisional:
        if all(_bounds_iou(candidate, other) < 0.35 for other in selected):
            selected.append(candidate)
            if len(selected) == maximum_candidates:
                break
    return [
        _with_rank(candidate, rank=index + 1) for index, candidate in enumerate(selected)
    ], len(provisional)


def write_multi_tree_candidate_preview(
    candidate: MultiTreeRoiCandidate,
    *,
    xyz_absolute: np.ndarray,
    hag_m: np.ndarray,
    classification: np.ndarray,
    crown_evidence: np.ndarray,
    output_html: Path,
) -> Path:
    """Write a compact two-panel preview for the mandatory pre-V4 visual gate."""

    try:
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise FastResearchError("FAST_MULTI_TREE_ROI_PREVIEW_DEPENDENCY_MISSING") from exc
    xyz = np.asarray(xyz_absolute, dtype=np.float64)
    hag = np.asarray(hag_m, dtype=np.float32)
    classes = np.asarray(classification, dtype=np.uint8)
    evidence = np.asarray(crown_evidence, dtype=bool)
    lower = np.asarray(candidate.xy_bounds_min, dtype=np.float64)
    upper = np.asarray(candidate.xy_bounds_max, dtype=np.float64)
    inside = (
        (xyz[:, 0] >= lower[0])
        & (xyz[:, 0] < upper[0])
        & (xyz[:, 1] >= lower[1])
        & (xyz[:, 1] < upper[1])
        & (classes != 2)
        & np.isfinite(hag)
        & (hag >= 0.3)
    )
    all_rows = _display_rows(np.flatnonzero(inside), maximum=25_000)
    crown_rows = _display_rows(np.flatnonzero(inside & evidence), maximum=25_000)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("non-ground HAG-ready", "crown-likelihood evidence"),
    )
    for column, rows_to_plot, marker_size in ((1, all_rows, 3), (2, crown_rows, 4)):
        figure.add_trace(
            go.Scattergl(
                x=xyz[rows_to_plot, 0],
                y=xyz[rows_to_plot, 1],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": hag[rows_to_plot],
                    "colorscale": "Viridis",
                    "cmin": 0.0,
                    "cmax": float(np.max(hag[inside])) if np.any(inside) else 1.0,
                    "showscale": column == 2,
                    "colorbar": {"title": "HAG (m)"},
                },
                name="representatives" if column == 1 else "crown evidence",
                hovertemplate="x=%{x:.2f}; y=%{y:.2f}; HAG=%{marker.color:.2f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        if candidate.crown_peak_xy:
            peak_xy = np.asarray(candidate.crown_peak_xy, dtype=np.float64)
            figure.add_trace(
                go.Scattergl(
                    x=peak_xy[:, 0],
                    y=peak_xy[:, 1],
                    mode="markers",
                    marker={"size": 10, "color": "#e31a1c", "symbol": "diamond"},
                    name="crown-like peak",
                    hovertemplate="crown-like local peak<extra></extra>",
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(range=[float(lower[0]), float(upper[0])], row=1, col=column)
        figure.update_yaxes(
            range=[float(lower[1]), float(upper[1])], scaleanchor=f"x{column}", row=1, col=column
        )
    figure.update_layout(
        title=(
            f"Candidate {candidate.rank}: peaks={candidate.crown_peak_count}; "
            f"rigid={candidate.rigid_geometry_fraction:.1%}; "
            f"elongation={candidate.elongation_ratio:.2f}"
        ),
        legend={"orientation": "h"},
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_html, include_plotlyjs=True, full_html=True)
    return output_html


def assess_retained_vegetation_purity(
    *,
    diagnostic_state: np.ndarray,
    linearity: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
) -> dict[str, Any]:
    """Classify retained admission rows before their V4 handoff."""

    state = np.asarray(diagnostic_state, dtype=np.uint8)
    linear = np.asarray(linearity, dtype=np.float32)
    planar = np.asarray(planarity, dtype=np.float32)
    scatter = np.asarray(scattering, dtype=np.float32)
    if any(values.shape != state.shape for values in (linear, planar, scatter)):
        raise FastResearchError("FAST_MULTI_TREE_PURITY_ALIGNMENT_INVALID")
    retained = state == 7  # FastV3DiagnosticState.VEGETATION_RETAINED
    if not np.any(retained):
        return {
            "quality": "HEAVILY_CONTAMINATED",
            "retained_representative_count": 0,
            "rigid_like_fraction": 1.0,
            "median_linearity": None,
            "median_planarity": None,
            "median_scattering": None,
        }
    rigid_like = (
        (linear[retained] > 0.65)
        | (planar[retained] > 0.75)
        | (scatter[retained] < 0.06)
    )
    fraction = float(np.mean(rigid_like))
    quality = "CLEAN" if fraction <= 0.05 else "MODERATELY_CONTAMINATED"
    if fraction > 0.20:
        quality = "HEAVILY_CONTAMINATED"
    return {
        "quality": quality,
        "retained_representative_count": int(np.count_nonzero(retained)),
        "rigid_like_fraction": fraction,
        "median_linearity": float(np.median(linear[retained])),
        "median_planarity": float(np.median(planar[retained])),
        "median_scattering": float(np.median(scatter[retained])),
    }


def crown_evidence_mask(
    *,
    hag_m: np.ndarray,
    classification: np.ndarray,
    linearity: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
) -> np.ndarray:
    """Return the fixed, non-V4 evidence mask used by the ROI search."""

    hag = np.asarray(hag_m, dtype=np.float32)
    classes = np.asarray(classification, dtype=np.uint8)
    linear = np.asarray(linearity, dtype=np.float32)
    planar = np.asarray(planarity, dtype=np.float32)
    scatter = np.asarray(scattering, dtype=np.float32)
    if any(values.shape != hag.shape for values in (classes, linear, planar, scatter)):
        raise FastResearchError("FAST_MULTI_TREE_CROWN_MASK_ALIGNMENT_INVALID")
    return np.asarray(
        (classes != 2)
        & np.isfinite(hag)
        & (hag >= _CROWN_MIN_HAG_M)
        & (scatter >= _CROWN_MIN_SCATTERING)
        & (linear <= _CROWN_MAX_LINEARITY)
        & (planar <= _CROWN_MAX_PLANARITY),
        dtype=bool,
    )


def _validated_arrays(**values: np.ndarray) -> dict[str, np.ndarray]:
    xyz = np.asarray(values["xyz_absolute"], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise FastResearchError("FAST_MULTI_TREE_XYZ_INVALID")
    result = {"xyz_absolute": xyz}
    for name, dtype in (
        ("hag_m", np.float32),
        ("raw_weight", np.uint32),
        ("classification", np.uint8),
        ("linearity", np.float32),
        ("planarity", np.float32),
        ("scattering", np.float32),
    ):
        array = np.asarray(values[name], dtype=dtype)
        if array.shape != (len(xyz),) or not np.isfinite(array).all():
            raise FastResearchError(f"FAST_MULTI_TREE_{name.upper()}_INVALID")
        result[name] = array
    if np.any(result["raw_weight"] == 0):
        raise FastResearchError("FAST_MULTI_TREE_RAW_WEIGHT_INVALID")
    return result


def _crown_peak_records(
    *,
    xyz: np.ndarray,
    hag: np.ndarray,
    linearity: np.ndarray,
    scattering: np.ndarray,
    crown_evidence: np.ndarray,
    grid_xy: np.ndarray,
    source_origin: np.ndarray,
    rows: int,
    cols: int,
    cell_m: float,
) -> tuple[tuple[float, float], ...]:
    cell_max = np.full((rows, cols), -np.inf, dtype=np.float64)
    np.maximum.at(
        cell_max,
        (grid_xy[crown_evidence, 1], grid_xy[crown_evidence, 0]),
        hag[crown_evidence],
    )
    local = (cell_max >= _PEAK_MIN_HAG_M) & (
        cell_max == ndimage.maximum_filter(cell_max, size=5, mode="constant", cval=-np.inf)
    )
    labels, label_count = ndimage.label(local, structure=np.ones((3, 3), dtype=np.uint8))
    result: list[tuple[float, float]] = []
    for label in range(1, label_count + 1):
        cells = np.argwhere(labels == label)
        heights = cell_max[cells[:, 0], cells[:, 1]]
        row, col = cells[int(np.argmax(heights))]
        center = source_origin + (np.asarray((col, row), dtype=np.float64) + 0.5) * cell_m
        distance = np.hypot(xyz[:, 0] - center[0], xyz[:, 1] - center[1])
        support = crown_evidence & (distance <= 2.0)
        if int(np.count_nonzero(support)) < 30:
            continue
        xy = xyz[support, :2]
        eigen = np.linalg.eigvalsh(np.cov(xy, rowvar=False))
        elongation = float(math.sqrt(float(eigen[-1]) / max(float(eigen[0]), 1e-6)))
        hspan = float(np.quantile(hag[support], 0.95) - np.quantile(hag[support], 0.05))
        if (
            elongation <= 3.25
            and hspan >= 1.25
            and float(np.median(scattering[support])) >= _CROWN_MIN_SCATTERING
            and float(np.median(linearity[support])) <= 0.60
        ):
            result.append((float(center[0]), float(center[1])))
    return tuple(sorted(result))


def _candidate_metrics(
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    inside: np.ndarray,
    crown_evidence: np.ndarray,
    crown_peaks: tuple[tuple[float, float], ...],
    xyz: np.ndarray,
    hag: np.ndarray,
    raw_weight: np.ndarray,
    linearity: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
    classes: np.ndarray,
    grid_xy: np.ndarray,
    lower_xy: tuple[int, int],
    width_cells: int,
    height_cells: int,
) -> MultiTreeRoiCandidate | None:
    evidence = crown_evidence
    rows = np.flatnonzero(evidence)
    if len(rows) < 300:
        return None
    xy = xyz[rows, :2]
    covariance = np.cov(xy, rowvar=False)
    eigen = np.linalg.eigvalsh(covariance)
    elongation = float(math.sqrt(float(eigen[-1]) / max(float(eigen[0]), 1e-6)))
    xspan = float(np.quantile(xy[:, 0], 0.95) - np.quantile(xy[:, 0], 0.05))
    yspan = float(np.quantile(xy[:, 1], 0.95) - np.quantile(xy[:, 1], 0.05))
    horizontal_spread = float(math.sqrt(max(xspan * yspan, 0.0)))
    peak_xy = tuple(
        peak
        for peak in crown_peaks
        if bounds_min[0] <= peak[0] < bounds_max[0]
        and bounds_min[1] <= peak[1] < bounds_max[1]
    )
    local_cells = grid_xy[rows] - np.asarray(lower_xy, dtype=np.int32)
    occupancy = np.zeros((height_cells, width_cells), dtype=bool)
    valid = (
        (local_cells[:, 0] >= 0)
        & (local_cells[:, 0] < width_cells)
        & (local_cells[:, 1] >= 0)
        & (local_cells[:, 1] < height_cells)
    )
    occupancy[local_cells[valid, 1], local_cells[valid, 0]] = True
    labels, label_count = ndimage.label(occupancy, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(labels.ravel(), minlength=label_count + 1)[1:]
    component_count = int(np.count_nonzero(component_sizes >= 3))
    all_non_ground = inside & (classes != 2) & (hag >= 0.3)
    if not np.any(all_non_ground):
        return None
    rigid_geometry = (
        (linearity[all_non_ground] > 0.65)
        | (planarity[all_non_ground] > 0.75)
        | (scattering[all_non_ground] < 0.06)
    )
    rigid_fraction = float(np.mean(rigid_geometry))
    h05 = float(np.quantile(hag[rows], 0.05))
    h95 = float(np.quantile(hag[rows], 0.95))
    hspan = h95 - h05
    score = (
        20.0 * min(len(peak_xy), 8)
        + 4.0 * min(component_count, 8)
        + 2.0 * hspan
        + 0.25 * horizontal_spread
        - 4.0 * max(elongation - 2.0, 0.0)
        - 30.0 * rigid_fraction
    )
    gate = (
        len(peak_xy) >= 4
        and component_count >= 2
        and elongation <= 3.5
        and rigid_fraction <= 0.15
    )
    return MultiTreeRoiCandidate(
        rank=0,
        score=float(score),
        xy_bounds_min=(float(bounds_min[0]), float(bounds_min[1])),
        xy_bounds_max=(float(bounds_max[0]), float(bounds_max[1])),
        representative_count=int(np.count_nonzero(inside)),
        raw_weight_count=int(np.sum(raw_weight[inside], dtype=np.int64)),
        crown_evidence_representative_count=len(rows),
        crown_evidence_raw_weight_count=int(np.sum(raw_weight[rows], dtype=np.int64)),
        crown_peak_count=len(peak_xy),
        crown_peak_xy=peak_xy,
        occupancy_component_count=component_count,
        hag_p05_m=h05,
        hag_p95_m=h95,
        hag_range_m=hspan,
        median_linearity=float(np.median(linearity[rows])),
        median_planarity=float(np.median(planarity[rows])),
        median_scattering=float(np.median(scattering[rows])),
        rigid_geometry_fraction=rigid_fraction,
        elongation_ratio=elongation,
        horizontal_spread_m=horizontal_spread,
        automatic_gate_pass=gate,
    )


def _with_rank(candidate: MultiTreeRoiCandidate, *, rank: int) -> MultiTreeRoiCandidate:
    return replace(candidate, rank=rank)


def _bounds_iou(first: MultiTreeRoiCandidate, second: MultiTreeRoiCandidate) -> float:
    first_min = np.asarray(first.xy_bounds_min)
    first_max = np.asarray(first.xy_bounds_max)
    second_min = np.asarray(second.xy_bounds_min)
    second_max = np.asarray(second.xy_bounds_max)
    overlap = np.maximum(0.0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min))
    overlap_area = float(overlap[0] * overlap[1])
    first_area = float(np.prod(first_max - first_min))
    second_area = float(np.prod(second_max - second_min))
    return overlap_area / (first_area + second_area - overlap_area)


def _display_rows(rows: np.ndarray, *, maximum: int) -> np.ndarray:
    if len(rows) <= maximum:
        return rows
    return rows[np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)]
