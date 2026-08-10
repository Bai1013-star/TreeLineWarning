"""Per-span FAST mature tree pipeline: CHM + smoothing + adaptive tree tops +
marker-controlled watershed + 3D membership + full-resolution back-map.

Every span is processed locally: broad vegetation raw rows inside the span
core+halo are re-voxelized to 0.10 m fine vegetation representatives, a
canopy-height model is built from HAG (never absolute Z), the CHM is smoothed
to suppress branch/leaf local maxima, tree tops are detected with a
height-adaptive suppression window, and a deterministic priority-flood
marker-controlled watershed assigns crown regions.  The 2D crown ids map back
to fine 3D vegetation with HAG consistency and then to the full-resolution raw
source rows via ``original_index``.

This module is FAST, NON_CERTIFIED research code; it never claims Formal
authority.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import (
    FastBaselineConfig,
    height_to_suppression_radius_m,
)

from algorithms.poles.evidence import PoleEvidenceError

SPAN_TREE_SCHEMA = "fast-baseline-span-tree-v1"


@dataclass(frozen=True, slots=True)
class SpanChm:
    """Span-local canopy height model over HAG."""

    height_m: np.ndarray  # float32 (rows, cols); NaN where unoccupied
    occupied: np.ndarray  # bool (rows, cols)
    point_count_by_cell: np.ndarray  # int32 (rows, cols)
    origin_xy: tuple[float, float]
    cell_size_m: float
    minimum_hag_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.height_m.shape


@dataclass(frozen=True, slots=True)
class TreeTopRecord:
    """One accepted tree top with its adaptive suppression radius."""

    tree_top_id: int
    row: int
    col: int
    xy_local_m: tuple[float, float]
    height_m: float
    prominence_m: float
    suppression_radius_m: float
    support_cell_count: int
    edge_limited: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_top_id": self.tree_top_id,
            "row": self.row,
            "col": self.col,
            "xy_local_m": list(self.xy_local_m),
            "height_m": self.height_m,
            "prominence_m": self.prominence_m,
            "suppression_radius_m": self.suppression_radius_m,
            "support_cell_count": self.support_cell_count,
            "edge_limited": self.edge_limited,
        }


@dataclass(frozen=True, slots=True)
class SpanTreeTops:
    """Accepted adaptive tree tops plus the raw local-maxima diagnostic."""

    accepted: tuple[TreeTopRecord, ...]
    raw_local_maxima_count: int
    accepted_count: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SpanCrownSegmentation:
    """Marker-controlled watershed crowns over the span CHM."""

    region_label_by_cell: np.ndarray  # int32 (rows, cols); 0 unassigned
    crown_cell_count_by_top: np.ndarray  # int64 (K,)
    ambiguity_mask: np.ndarray  # bool (rows, cols)
    diagnostics: dict[str, Any]


# ---------------------------------------------------------------------------
# CHM build
# ---------------------------------------------------------------------------


def build_span_chm(
    xyz_local_m: np.ndarray,
    hag_m: np.ndarray,
    config: FastBaselineConfig,
) -> SpanChm:
    """Build a span-local CHM from fine vegetation HAG values.

    Empty cells are explicitly NaN (never implicitly zero), non-finite HAG and
    cells below ``chm_minimum_hag_m`` are excluded, and the per-cell height is
    the exact maximum HAG (no interpolation).
    """

    xyz = np.asarray(xyz_local_m, dtype=np.float64)
    hag = np.asarray(hag_m, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or hag.shape != (len(xyz),):
        raise PoleEvidenceError("SPAN_CHM_ALIGNMENT_INVALID")
    res = config.chm_resolution_m
    if res <= 0.0:
        raise PoleEvidenceError("SPAN_CHM_RESOLUTION_INVALID")
    valid = np.isfinite(hag) & (hag >= config.chm_minimum_hag_m)
    if not np.any(valid):
        return SpanChm(
            height_m=np.full((0, 0), np.nan, dtype=np.float32),
            occupied=np.zeros((0, 0), dtype=bool),
            point_count_by_cell=np.zeros((0, 0), dtype=np.int32),
            origin_xy=(0.0, 0.0),
            cell_size_m=res,
            minimum_hag_m=config.chm_minimum_hag_m,
        )
    selected_xy = xyz[valid, :2]
    origin_x = float(np.floor(np.min(selected_xy[:, 0]) / res) * res)
    origin_y = float(np.floor(np.min(selected_xy[:, 1]) / res) * res)
    cell = np.floor((selected_xy - np.asarray((origin_x, origin_y))) / res).astype(np.int64)
    rows = int(np.max(cell[:, 1])) + 1
    cols = int(np.max(cell[:, 0])) + 1
    if rows * cols > 12_000_000:
        raise PoleEvidenceError("SPAN_CHM_GRID_TOO_LARGE")
    flat = cell[:, 1] * cols + cell[:, 0]
    height = np.full(rows * cols, -np.inf, dtype=np.float32)
    point_count = np.zeros(rows * cols, dtype=np.int32)
    selected_hag = hag[valid]
    np.maximum.at(height, flat, selected_hag)
    np.add.at(point_count, flat, 1)
    height = height.reshape(rows, cols)
    point_count = point_count.reshape(rows, cols)
    occupied = np.isfinite(height) & (point_count > 0)
    height[~occupied] = np.nan
    return SpanChm(
        height_m=height,
        occupied=occupied,
        point_count_by_cell=point_count,
        origin_xy=(origin_x, origin_y),
        cell_size_m=res,
        minimum_hag_m=config.chm_minimum_hag_m,
    )


# ---------------------------------------------------------------------------
# CHM smoothing (normalized Gaussian over occupied cells)
# ---------------------------------------------------------------------------


def smooth_span_chm(chm: SpanChm, config: FastBaselineConfig) -> np.ndarray:
    """Return a smoothed CHM where occupied cells are blurred in place.

    Uses a normalized convolution so empty cells never dilute crown heights at
    the canopy boundary: the Gaussian blur is divided by the blur of the
    occupancy mask.  Unoccupied cells remain NaN.
    """

    from scipy.ndimage import gaussian_filter

    if chm.shape == (0, 0):
        return chm.height_m.copy()
    sigma_cells = max(0.0, config.chm_smooth_sigma_m / chm.cell_size_m)
    if sigma_cells <= 0.0:
        return chm.height_m.copy()
    fill = np.where(chm.occupied, chm.height_m, 0.0)
    blurred = gaussian_filter(fill, sigma=sigma_cells, mode="constant", cval=0.0)
    weight = gaussian_filter(chm.occupied.astype(np.float64), sigma=sigma_cells, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(
            weight > 1e-9, blurred / weight, chm.height_m
        ).astype(np.float32)
    smoothed[~chm.occupied] = np.nan
    return smoothed


# ---------------------------------------------------------------------------
# Adaptive tree-top detection
# ---------------------------------------------------------------------------


def _neighbor_cells(rows: int, cols: int, row: int, col: int, radius: float) -> np.ndarray:
    """Return all occupied-cell offsets within a physical circular radius."""

    radius_cells = int(math.ceil(radius))
    offsets = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr == 0 and dc == 0:
                continue
            if math.hypot(dr, dc) <= radius + 1e-9:
                offsets.append((dr, dc))
    return np.asarray(offsets, dtype=np.int64)


def _annulus_median_height(
    height: np.ndarray, occupied: np.ndarray, row: int, col: int,
    inner_radius: float, outer_radius: float, cell_size_m: float,
) -> float | None:
    outer_cells = int(math.ceil(outer_radius / cell_size_m))
    rows, cols = height.shape
    values = []
    for dr in range(-outer_cells, outer_cells + 1):
        for dc in range(-outer_cells, outer_cells + 1):
            distance = math.hypot(dr, dc) * cell_size_m
            if distance < inner_radius - 1e-9 or distance > outer_radius + 1e-9:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols and occupied[r, c]:
                values.append(float(height[r, c]))
    if not values:
        return None
    return float(np.median(values))


def detect_span_tree_tops(
    chm: SpanChm,
    smoothed: np.ndarray,
    config: FastBaselineConfig,
) -> SpanTreeTops:
    """Detect accepted tree tops with height-adaptive suppression.

    Local maxima are the cells that dominate their small local window.  Each is
    scored by prominence (peak vs. surrounding annulus) and local canopy
    support.  Peaks are then suppressed with an adaptive radius that grows with
    the height of the taller (accepted) peak, so broad tall crowns absorb small
    branch peaks while distinct neighbouring crowns survive.
    """

    height = chm.height_m
    occupied = chm.occupied
    rows, cols = height.shape
    started = time.perf_counter()
    raw_count = 0
    candidates: list[dict[str, Any]] = []
    local_radius_cells = max(1, int(math.ceil(config.tree_top_local_radius_m / chm.cell_size_m)))
    for row in range(rows):
        for col in range(cols):
            if not occupied[row, col]:
                continue
            peak = float(height[row, col])
            if peak < config.tree_minimum_hag_m:
                continue
            is_max = True
            for dr in range(-local_radius_cells, local_radius_cells + 1):
                for dc in range(-local_radius_cells, local_radius_cells + 1):
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < rows and 0 <= c < cols and occupied[r, c]:
                        if float(height[r, c]) > peak + 1e-9:
                            is_max = False
                            break
                if not is_max:
                    break
            if not is_max:
                continue
            raw_count += 1
            prominence = _annulus_median_height(
                height,
                occupied,
                row,
                col,
                config.tree_top_local_radius_m,
                config.tree_top_prominence_annulus_outer_m,
                chm.cell_size_m,
            )
            prominence_value = peak - prominence if prominence is not None else 0.0
            support_radius = height_to_suppression_radius_m(config, peak)
            radius_cells = int(math.ceil(support_radius / chm.cell_size_m))
            r0 = max(0, row - radius_cells)
            r1 = min(rows, row + radius_cells + 1)
            c0 = max(0, col - radius_cells)
            c1 = min(cols, col + radius_cells + 1)
            support = int(np.count_nonzero(occupied[r0:r1, c0:c1]))
            annulus_cells = int(
                math.ceil(config.tree_top_prominence_annulus_outer_m / chm.cell_size_m)
            )
            edge_limited = (
                row < annulus_cells
                or col < annulus_cells
                or row >= rows - annulus_cells
                or col >= cols - annulus_cells
            )
            candidates.append(
                {
                    "row": row,
                    "col": col,
                    "height": peak,
                    "prominence": prominence_value,
                    "support": support,
                    "radius": support_radius,
                    "edge_limited": edge_limited,
                }
            )

    # Sort by height desc, prominence desc, then (row, col) for determinism.
    candidates.sort(
        key=lambda item: (
            -item["height"],
            -item["prominence"],
            item["row"],
            item["col"],
        )
    )
    accepted: list[TreeTopRecord] = []
    accepted_positions: list[tuple[float, float]] = []  # xy
    accepted_radii: list[float] = []
    for item in candidates:
        xy = (
            chm.origin_xy[0] + (item["col"] + 0.5) * chm.cell_size_m,
            chm.origin_xy[1] + (item["row"] + 0.5) * chm.cell_size_m,
        )
        dominated = False
        for accepted_xy, accepted_radius in zip(
            accepted_positions, accepted_radii, strict=True
        ):
            distance = math.hypot(xy[0] - accepted_xy[0], xy[1] - accepted_xy[1])
            if distance <= accepted_radius + 1e-9:
                dominated = True
                break
        if dominated:
            continue
        required_prominence = max(
            config.tree_top_min_prominence_m,
            config.tree_top_min_prominence_fraction * item["height"],
        )
        if item["prominence"] < required_prominence:
            continue
        if item["support"] < config.tree_top_min_support_points:
            continue
        radius_cells = max(1, int(math.ceil(item["radius"] / chm.cell_size_m)))
        r0 = max(0, item["row"] - radius_cells)
        r1 = min(rows, item["row"] + radius_cells + 1)
        c0 = max(0, item["col"] - radius_cells)
        c1 = min(cols, item["col"] + radius_cells + 1)
        window_occupied = int(np.count_nonzero(occupied[r0:r1, c0:c1]))
        window_total = (r1 - r0) * (c1 - c0)
        if window_total and window_occupied / window_total < 0.20:
            continue
        accepted.append(
            TreeTopRecord(
                tree_top_id=len(accepted),
                row=item["row"],
                col=item["col"],
                xy_local_m=xy,
                height_m=item["height"],
                prominence_m=item["prominence"],
                suppression_radius_m=item["radius"],
                support_cell_count=item["support"],
                edge_limited=item["edge_limited"],
            )
        )
        accepted_positions.append(xy)
        accepted_radii.append(item["radius"])
    return SpanTreeTops(
        accepted=tuple(accepted),
        raw_local_maxima_count=raw_count,
        accepted_count=len(accepted),
        diagnostics={
            "schema": SPAN_TREE_SCHEMA,
            "raw_local_maxima_count": raw_count,
            "accepted_tree_top_count": len(accepted),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


# ---------------------------------------------------------------------------
# Marker-controlled watershed (deterministic max-bottleneck priority flood)
# ---------------------------------------------------------------------------


def _watershed_domain(chm: SpanChm, config: FastBaselineConfig) -> np.ndarray:
    return chm.occupied & np.isfinite(chm.height_m) & (
        chm.height_m >= config.chm_minimum_hag_m
    )


def segment_span_crowns(
    chm: SpanChm,
    smoothed: np.ndarray,
    tops: SpanTreeTops,
    config: FastBaselineConfig,
) -> SpanCrownSegmentation:
    """Marker-controlled watershed over the span canopy surface.

    Accepted tree tops are the markers.  Each domain cell is assigned to the
    marker that maximizes the minimum height along the path (a deterministic
    priority-flood over the smoothed CHM).  Crown regions therefore stop at
    canopy gaps, low valleys, watershed ridges and non-vegetation cells.
    """

    rows, cols = chm.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    if rows == 0 or cols == 0 or not tops.accepted:
        return SpanCrownSegmentation(
            region_label_by_cell=labels,
            crown_cell_count_by_top=np.zeros(len(tops.accepted), dtype=np.int64),
            ambiguity_mask=np.zeros((rows, cols), dtype=bool),
            diagnostics={
                "schema": SPAN_TREE_SCHEMA,
                "marker_count": len(tops.accepted),
                "domain_cell_count": 0,
                "assigned_cell_count": 0,
            },
        )
    domain = _watershed_domain(chm, config)
    height = np.where(domain, smoothed, np.nan).astype(np.float64)
    bottleneck = np.full((rows, cols), np.nan, dtype=np.float64)
    marker_seed = np.zeros((rows, cols), dtype=bool)
    heap: list[tuple[float, int, int]] = []  # (-bottleneck, marker, flat)
    for top in tops.accepted:
        row, col = top.row, top.col
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        marker = top.tree_top_id
        labels[row, col] = marker + 1
        marker_seed[row, col] = True
        bottleneck[row, col] = float(height[row, col])
        heapq.heappush(heap, (-bottleneck[row, col], marker, row * cols + col))
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    assigned = 0
    while heap:
        neg_bottleneck, marker, flat = heapq.heappop(heap)
        row, col = divmod(flat, cols)
        if labels[row, col] != marker + 1:
            continue  # stale entry after a deterministic relabel
        flood_bottleneck = -neg_bottleneck
        for dr, dc in neighbors:
            r, c = row + dr, col + dc
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if not domain[r, c]:
                continue
            cell_height = float(height[r, c])
            candidate = min(flood_bottleneck, cell_height)
            current_label = labels[r, c]
            if current_label == 0:
                labels[r, c] = marker + 1
                bottleneck[r, c] = candidate
                assigned += 1
                heapq.heappush(heap, (-candidate, marker, r * cols + c))
            else:
                if marker_seed[r, c]:
                    continue  # a marker seed always keeps its own crown id
                current_marker = current_label - 1
                current_bottleneck = bottleneck[r, c]
                wider = candidate > current_bottleneck + 1e-12
                equal_tie = (
                    abs(candidate - current_bottleneck) <= 1e-12
                    and marker < current_marker
                )
                if wider or equal_tie:
                    labels[r, c] = marker + 1
                    bottleneck[r, c] = candidate
                    heapq.heappush(heap, (-candidate, marker, r * cols + c))
    crown_cells = np.zeros(len(tops.accepted), dtype=np.int64)
    counts = np.bincount(labels.ravel(), minlength=1)
    for marker in range(len(tops.accepted)):
        crown_cells[marker] = int(counts[marker + 1])
    return SpanCrownSegmentation(
        region_label_by_cell=labels,
        crown_cell_count_by_top=crown_cells,
        ambiguity_mask=np.zeros((rows, cols), dtype=bool),
        diagnostics={
            "schema": SPAN_TREE_SCHEMA,
            "marker_count": len(tops.accepted),
            "domain_cell_count": int(np.count_nonzero(domain)),
            "assigned_cell_count": int(assigned),
        },
    )


# ---------------------------------------------------------------------------
# 3D membership and full-resolution back-map
# ---------------------------------------------------------------------------


def assign_span_tree_membership(
    fine_xyz_local_m: np.ndarray,
    fine_hag_m: np.ndarray,
    chm: SpanChm,
    smoothed: np.ndarray,
    crowns: SpanCrownSegmentation,
    config: FastBaselineConfig,
) -> np.ndarray:
    """Map 2D watershed crown ids onto 3D fine vegetation representatives.

    A fine representative belongs to a crown when its XY cell carries that
    crown label and its HAG is consistent with the local canopy surface
    (``hag <= surface_height + tolerance`` and ``hag >= tree_minimum_hag_m``).
    """

    xyz = np.asarray(fine_xyz_local_m, dtype=np.float64)
    if len(xyz) == 0:
        return np.full(0, -1, dtype=np.int32)
    res = chm.cell_size_m
    cell = np.floor(
        (xyz[:, :2] - np.asarray(chm.origin_xy, dtype=np.float64)) / res
    ).astype(np.int64)
    rows, cols = chm.shape
    inside = (
        (cell[:, 0] >= 0) & (cell[:, 0] < cols) & (cell[:, 1] >= 0) & (cell[:, 1] < rows)
    )
    flat = cell[:, 1] * cols + cell[:, 0]
    label = np.full(len(xyz), -1, dtype=np.int32)
    hag = np.asarray(fine_hag_m, dtype=np.float32)
    crown_label = crowns.region_label_by_cell.ravel()
    surface = np.asarray(smoothed, dtype=np.float32).ravel()
    # Reps outside the CHM grid must never index it; clip for the lookup and
    # mask with ``inside`` so the out-of-bounds value is never used.
    safe_flat = np.where(inside, flat, 0)
    candidate = crown_label[safe_flat] > 0
    surface_at = surface[safe_flat]
    mask = (
        inside
        & candidate
        & (hag >= config.tree_minimum_hag_m)
        & (np.isfinite(surface_at) & (hag <= surface_at + config.tree_membership_hag_tolerance_m))
    )
    label[mask] = crown_label[safe_flat[mask]] - 1
    # Drop crowns with too few final members (leaf-scale noise).
    if len(label):
        counts = np.bincount(label[label >= 0], minlength=0)
        small = np.flatnonzero(counts < config.tree_min_final_members)
        for tree_id in small:
            label[label == tree_id] = -1
    return label


def build_span_tree_result(
    span_subset_region: Any,
    fine_reps_npz_path: Any,
    fine_original_index: np.ndarray,
    tree_id_by_fine: np.ndarray,
    tops: SpanTreeTops,
    crowns: SpanCrownSegmentation,
    chm: SpanChm,
    config: FastBaselineConfig,
) -> dict[str, Any]:
    """Back-map fine tree membership to raw source rows and summarize."""

    from algorithms.vegetation.coarse_to_fine import map_region_rows_to_tree

    raw_tree = map_region_rows_to_tree(
        span_subset_region,
        fine_reps_npz_path,
        fine_original_index,
        tree_id_by_fine.astype(np.int32),
    )
    tree_ids = np.asarray(tree_id_by_fine, dtype=np.int32)
    records: list[dict[str, Any]] = []
    for top in tops.accepted:
        members = np.count_nonzero(tree_ids == top.tree_top_id)
        raw_members = int(np.count_nonzero(raw_tree == top.tree_top_id))
        records.append(
            {
                "tree_id": top.tree_top_id,
                "tree_top_id": top.tree_top_id,
                "apex_xy_local_m": list(top.xy_local_m),
                "apex_height_m": top.height_m,
                "prominence_m": top.prominence_m,
                "suppression_radius_m": top.suppression_radius_m,
                "fine_representative_count": int(members),
                "raw_point_count": raw_members,
                "crown_cell_count": int(crowns.crown_cell_count_by_top[top.tree_top_id])
                if top.tree_top_id < len(crowns.crown_cell_count_by_top)
                else 0,
            }
        )
    return {
        "schema": SPAN_TREE_SCHEMA,
        "tree_count": len(records),
        "trees": records,
        "raw_tree_id": raw_tree,
        "fine_tree_id": tree_ids,
        "raw_local_maxima": tops.raw_local_maxima_count,
        "accepted_tree_tops": tops.accepted_count,
        "chm_origin_xy": list(chm.origin_xy),
        "chm_cell_size_m": chm.cell_size_m,
    }


__all__ = [
    "SPAN_TREE_SCHEMA",
    "SpanChm",
    "TreeTopRecord",
    "SpanTreeTops",
    "SpanCrownSegmentation",
    "build_span_chm",
    "smooth_span_chm",
    "detect_span_tree_tops",
    "segment_span_crowns",
    "assign_span_tree_membership",
    "build_span_tree_result",
]
