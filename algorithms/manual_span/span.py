"""Manual pole refinement and the single manual ``SpanRecord``.

The teacher clicks two approximate pole positions on a deterministic sample
of the ORIGINAL source cloud.  Each click is refined inside a bounded local
neighbourhood of the coarse routing representatives: the pole shaft is scored
for vertical continuity, small horizontal footprint, shaft-like geometry and
ground/base support — explicitly never blindly snapped to a nearby tree trunk.
A defensible refinement yields base / top / attachment geometry; otherwise the
failure reason is returned so the UI can ask for a re-click (a UI correction,
not a development handoff).

The two refined poles then define exactly ONE ``SpanRecord`` (core + halo)
in the same scene-local frame the span pipelines expect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.manual_span.config import ManualSpanConfig
from algorithms.poles.evidence import GroundEvidence
from algorithms.poles.topology import SpanRecord


@dataclass(frozen=True, slots=True)
class ManualRefinedPole:
    """One defensible pole refined from an approximate teacher click."""

    label: str  # "A" | "B"
    clicked_xyz_absolute_m: tuple[float, float, float]
    base_xyz_local_m: tuple[float, float, float]
    base_xyz_absolute_m: tuple[float, float, float]
    top_xyz_local_m: tuple[float, float, float]
    top_xyz_absolute_m: tuple[float, float, float]
    height_m: float
    xy_footprint_m: float
    median_verticality: float
    median_linearity: float
    representative_count: int
    ground_support_reps: int
    defensible: bool
    reason: str | None
    refinement_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "clicked_xyz_absolute_m": list(self.clicked_xyz_absolute_m),
            "base_xyz_local_m": list(self.base_xyz_local_m),
            "base_xyz_absolute_m": list(self.base_xyz_absolute_m),
            "top_xyz_local_m": list(self.top_xyz_local_m),
            "top_xyz_absolute_m": list(self.top_xyz_absolute_m),
            "height_m": self.height_m,
            "xy_footprint_m": self.xy_footprint_m,
            "median_verticality": self.median_verticality,
            "median_linearity": self.median_linearity,
            "representative_count": self.representative_count,
            "ground_support_reps": self.ground_support_reps,
            "defensible": self.defensible,
            "reason": self.reason,
            "refinement_diagnostics": dict(self.refinement_diagnostics),
        }


def _unit_xy(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector[:2]))
    if norm <= 1e-12:
        return np.asarray((1.0, 0.0), dtype=np.float64)
    return np.asarray(vector[:2], dtype=np.float64) / norm


def refine_pole_from_click(
    label: str,
    clicked_xyz_absolute_m: tuple[float, float, float],
    evidence: GroundEvidence,
    cfg: FastBaselineConfig,
    manual: ManualSpanConfig,
) -> ManualRefinedPole:
    """Refine one approximate click into a defensible pole (or a clear failure).

    *evidence* carries the coarse non-ground representatives plus per-point
    dimensionality features and HAG in the shared scene-local frame.  The
    clicked coordinates are absolute; they are shifted to local internally.
    """

    origin = np.asarray(evidence.origin_xyz, dtype=np.float64)
    click = np.asarray(clicked_xyz_absolute_m, dtype=np.float64)
    click_local = click - origin

    xy = evidence.xyz_local_m[:, :2]
    xy_delta = xy - click_local[:2][None, :]
    xy_dist = np.linalg.norm(xy_delta, axis=1)
    in_radius = xy_dist <= manual.refine_search_radius_m
    hag = np.asarray(evidence.hag_m, dtype=np.float32)
    in_height = hag <= manual.refine_z_max_hag_m
    rows = np.flatnonzero(in_radius & in_height & (hag >= 0.0))
    non_ground_rows = rows[evidence.classification[rows] != 2]

    def failure(reason: str) -> ManualRefinedPole:
        return ManualRefinedPole(
            label=label,
            clicked_xyz_absolute_m=clicked_xyz_absolute_m,
            base_xyz_local_m=tuple(float(value) for value in click_local),
            base_xyz_absolute_m=clicked_xyz_absolute_m,
            top_xyz_local_m=tuple(float(value) for value in click_local),
            top_xyz_absolute_m=clicked_xyz_absolute_m,
            height_m=0.0,
            xy_footprint_m=0.0,
            median_verticality=0.0,
            median_linearity=0.0,
            representative_count=0,
            ground_support_reps=0,
            defensible=False,
            reason=reason,
        )

    if len(non_ground_rows) < manual.refine_min_support_reps:
        return failure("LOCAL_POINT_CLOUD_TOO_SPARSE")

    # Extract the dense shaft core: the pole is the tightest vertical cluster in
    # the local neighbourhood (surrounding canopy branches are iteratively
    # trimmed by median-centre + bounded radius, never blindly snapped to a
    # nearby tree trunk because the shaft-like and continuity gates still run
    # on the extracted core).
    core_rows = non_ground_rows.copy()
    for radius in (1.6, 1.3, 1.1, 0.9, 0.75):
        if len(core_rows) < manual.refine_min_support_reps:
            break
        centre = np.median(evidence.xyz_local_m[core_rows, :2], axis=0)
        dists = np.linalg.norm(
            evidence.xyz_local_m[core_rows, :2] - centre[None, :], axis=1
        )
        keep = dists <= radius
        if np.count_nonzero(keep) < manual.refine_min_support_reps:
            break
        core_rows = core_rows[keep]

    points = evidence.xyz_local_m[core_rows]
    hag_rows = hag[core_rows]
    sorted_by_z = np.argsort(hag_rows, kind="stable")
    points = points[sorted_by_z]
    hag_rows = hag_rows[sorted_by_z]
    core_rows = core_rows[sorted_by_z]

    min_hag = float(np.min(hag_rows))
    max_hag = float(np.max(hag_rows))
    height_m = max_hag - min_hag
    if height_m < manual.refine_min_pole_height_m:
        return failure(
            f"INSUFFICIENT_VERTICAL_EXTENT height={height_m:.2f}m < "
            f"{manual.refine_min_pole_height_m}m"
        )

    # Horizontal footprint of the extracted shaft core (robust spread).
    p_low = float(np.quantile(points[:, 0], 0.10))
    p_high = float(np.quantile(points[:, 0], 0.90))
    q_low = float(np.quantile(points[:, 1], 0.10))
    q_high = float(np.quantile(points[:, 1], 0.90))
    footprint = float(max(p_high - p_low, q_high - q_low))
    if footprint > manual.refine_max_footprint_m:
        return failure(
            f"FOOTPRINT_TOO_WIDE xy_footprint={footprint:.2f}m > "
            f"{manual.refine_max_footprint_m}m"
        )

    # Vertical continuity: split the column into 1 m HAG bands and require the
    # XY centroid to remain stable while the bands stay occupied up the shaft.
    band = manual.refine_band_height_m
    band_start = int(math.floor(min_hag / band))
    band_stop = int(math.floor(max_hag / band))
    centroids: list[tuple[float, float]] = []
    occupied_bands: list[int] = []
    for band_index in range(band_start, band_stop + 1):
        lo = band_index * band
        hi = lo + band
        band_mask = (hag_rows >= lo) & (hag_rows < hi)
        count = int(np.count_nonzero(band_mask))
        if count == 0:
            continue
        occupied_bands.append(band_index)
        centroids.append(
            (
                float(np.mean(points[band_mask, 0])),
                float(np.mean(points[band_mask, 1])),
            )
        )
    if not occupied_bands:
        return failure("NO_OCCUPIED_HEIGHT_BANDS")
    max_gap = occupied_bands[-1] - occupied_bands[0] + 1 - len(occupied_bands)
    if max_gap > manual.refine_max_vertical_gap_m / band:
        return failure(
            f"VERTICAL_CONTINUITY_BROKEN max_band_gap={max_gap * band:.2f}m"
        )
    centroids_arr = np.asarray(centroids, dtype=np.float64)
    drift = float(np.max(np.linalg.norm(np.diff(centroids_arr, axis=0), axis=1)))
    if len(centroids_arr) >= 2 and drift > manual.refine_max_band_drift_m:
        return failure(
            f"SHAFT_DRIFT_TOO_LARGE band_centroid_drift={drift:.2f}m"
        )

    # Shaft-like geometry from the frozen dimensionality features.
    median_verticality = float(np.median(evidence.verticality[core_rows]))
    median_linearity = float(np.median(evidence.linearity[core_rows]))
    vertical_ok = median_verticality >= manual.refine_verticality_min
    linear_ok = median_linearity >= manual.refine_linearity_min
    if not (vertical_ok or linear_ok):
        return failure(
            f"NOT_SHAFT_LIKE verticality={median_verticality:.2f} "
            f"linearity={median_linearity:.2f}"
        )

    # Ground/base support: the pole genuinely reaches the ground.  Under dense
    # canopy the SMRF ground class may be patchy, so the pole is accepted when
    # EITHER the shaft core descends to near-ground HAG (min_hag <= 1.5 m) OR
    # ground representatives exist within the contact radius of the shaft.
    base_xy = np.asarray((float(np.median(points[:, 0])), float(np.median(points[:, 1]))))
    ground_rows = rows[evidence.classification[rows] == 2]
    support_reps = 0
    if len(ground_rows):
        ground_xy = evidence.xyz_local_m[ground_rows, :2]
        ground_dist = np.linalg.norm(ground_xy - base_xy[None, :], axis=1)
        support_reps = int(np.count_nonzero(ground_dist <= manual.refine_ground_contact_radius_m))
    reaches_ground = float(min_hag) <= 1.5
    if support_reps == 0 and not reaches_ground:
        return failure("NO_GROUND_SUPPORT")

    # Refined geometry: base at ground Z under the shaft; top at a robust
    # high quantile of the upper points (a single isolated noise point never
    # inflates the pole height).
    ground_z = float(
        np.median(evidence.xyz_local_m[ground_rows, 2])
        if len(ground_rows)
        else float(np.min(points[:, 2]))
    )
    top_quantile = manual.refine_top_quantile
    top_z = float(np.quantile(hag_rows, top_quantile))
    top_xy = base_xy
    base_xyz_local = (float(base_xy[0]), float(base_xy[1]), ground_z)
    top_xyz_local = (float(top_xy[0]), float(top_xy[1]), base_xyz_local[2] + top_z)
    base_xyz_absolute = tuple(float(value) for value in np.asarray(base_xyz_local) + origin)
    top_xyz_absolute = tuple(float(value) for value in np.asarray(top_xyz_local) + origin)
    height_used = float(top_xyz_local[2] - base_xyz_local[2])

    return ManualRefinedPole(
        label=label,
        clicked_xyz_absolute_m=clicked_xyz_absolute_m,
        base_xyz_local_m=base_xyz_local,
        base_xyz_absolute_m=base_xyz_absolute,
        top_xyz_local_m=top_xyz_local,
        top_xyz_absolute_m=top_xyz_absolute,
        height_m=height_used,
        xy_footprint_m=footprint,
        median_verticality=median_verticality,
        median_linearity=median_linearity,
        representative_count=int(len(core_rows)),
        ground_support_reps=support_reps,
        defensible=True,
        reason=None,
    )


def refine_pole_from_raw_cloud(
    label: str,
    clicked_xyz_absolute_m: tuple[float, float, float],
    raw_xyz_absolute: np.ndarray,
    origin_xyz: tuple[float, float, float],
    cfg: FastBaselineConfig,
    manual: ManualSpanConfig,
) -> ManualRefinedPole:
    """Refine one approximate click from nearby RAW source LAS points.

    The click is treated only as an approximate XY seed.  The ORIGINAL
    (pre-denoise / pre-ground) points in the local box are evaluated for a
    plausible vertical pole structure: dense-shaft extraction, vertical extent,
    horizontal footprint, vertical continuity, shaft-like dimensionality and
    ground/base support.  The returned ``ManualRefinedPole`` keeps the same
    output contract and always carries ``refinement_diagnostics`` with at least
    the evaluated vertical extent and the refined XY.
    """

    click = np.asarray(clicked_xyz_absolute_m, dtype=np.float64)
    origin = np.asarray(origin_xyz, dtype=np.float64)
    points = np.asarray(raw_xyz_absolute, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return _raw_failure(
            label, clicked_xyz_absolute_m, "LOCAL_POINT_CLOUD_TOO_SPARSE", {},
        )
    # Memory-safe circular filter: the previous full-N
    # `np.linalg.norm(points[:, :2] - click, axis=1)` materialized an (N,) and an
    # (N,2) float64 pair (~33 MiB + 67 MiB for N=4.2M), which OOM'd on a dense
    # real Pole-B neighborhood.  A cheap square pre-filter (boolean masks only)
    # reduces N, then the circular distance is computed in bounded chunks so the
    # full-N float64 distance array is never materialized.  Semantics are
    # identical (the same circular radius refine_search_radius_m).
    radius = float(manual.refine_search_radius_m)
    cx, cy = float(click[0]), float(click[1])
    inside_square = (
        (points[:, 0] >= cx - radius)
        & (points[:, 0] <= cx + radius)
        & (points[:, 1] >= cy - radius)
        & (points[:, 1] <= cy + radius)
    )
    square_rows = np.flatnonzero(inside_square)
    if len(square_rows) == 0:
        local = points[inside_square]
    else:
        square_xy = points[square_rows, :2]
        circle_mask = _within_radius_chunked(square_xy, cx, cy, radius)
        local = points[square_rows][circle_mask]
    # Diagnostics always carry the evaluated vertical extent and refined XY even
    # when the structure is too weak to be accepted.
    local_extent = float(np.ptp(local[:, 2])) if len(local) else 0.0
    local_xy = (
        [float(v) for v in np.median(local[:, :2], axis=0)]
        if len(local)
        else [float(click[0]), float(click[1])]
    )
    base_diag: dict[str, Any] = {
        "local_point_count": int(len(local)),
        "vertical_extent_m": local_extent,
        "refined_xy_absolute_m": local_xy,
        "clicked_xy_absolute_m": [float(click[0]), float(click[1])],
    }
    if len(local) < manual.refine_min_support_reps:
        return _raw_failure(
            label, clicked_xyz_absolute_m, "LOCAL_POINT_CLOUD_TOO_SPARSE",
            dict(base_diag),
        )
    # Isolate the plausible vertical pole column.  In the RAW local cloud a pole
    # appears as a small-XY column whose points span a large vertical range, so
    # we voxelize XY into small cells and pick the cell with the largest vertical
    # extent (with sufficient support) as the shaft seed, then take the points
    # within a tight radius of that seed as the core.  This never snaps to a
    # broad tree crown and a surrounding tree trunk fails the footprint gate.
    _SHAFT_CELL_M = 0.30
    keys = np.floor(local[:, :2] / _SHAFT_CELL_M).astype(np.int64)
    order = np.lexsort((local[:, 2], keys[:, 0], keys[:, 1]))
    sorted_keys = keys[order]
    sorted_z = local[order, 2]
    boundaries = np.flatnonzero(
        np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    ) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    best_cell: tuple[int, int] | None = None
    best_score = -1.0
    best_extent_fallback = 0.0
    fallback_cell: tuple[int, int] | None = None
    for group in range(len(starts)):
        if int(ends[group] - starts[group]) < manual.refine_min_support_reps:
            continue
        zvals = sorted_z[starts[group]: ends[group]]
        extent = float(zvals[-1] - zvals[0])
        occupied_bands = sorted(set(int(value // 1.0) for value in zvals))
        max_gap = max(
            (
                occupied_bands[pos + 1] - occupied_bands[pos]
                for pos in range(len(occupied_bands) - 1)
            ),
            default=0,
        )
        # Score prefers CONTINUOUS tall columns (a pole) over a fragmented stack
        # of low structure + high canopy (a tree), which has a large extent but
        # a large vertical gap.
        score = extent if max_gap <= manual.refine_max_vertical_gap_m else 0.0
        if score > best_score:
            best_score = score
            best_cell = (
                int(sorted_keys[starts[group], 0]),
                int(sorted_keys[starts[group], 1]),
            )
        if extent > best_extent_fallback:
            best_extent_fallback = extent
            fallback_cell = (
                int(sorted_keys[starts[group], 0]),
                int(sorted_keys[starts[group], 1]),
            )
    if best_cell is None:
        return _raw_failure(
            label, clicked_xyz_absolute_m, "LOCAL_POINT_CLOUD_TOO_SPARSE",
            dict(base_diag),
        )
    if best_score <= 0.0:
        # No continuous column found; use the tallest (for a clear continuity
        # failure diagnostic) rather than silently accepting the nearest point.
        best_cell = fallback_cell
    seed_xy = (np.asarray(best_cell, dtype=np.float64) + 0.5) * _SHAFT_CELL_M
    # A utility pole is a thin column (~0.3 m radius); a tight core radius keeps
    # the shaft features from being diluted by dense surrounding vegetation.
    core_mask = (
        np.linalg.norm(local[:, :2] - seed_xy[None, :], axis=1) <= 0.35
    )
    core = local[core_mask]
    if len(core) < manual.refine_min_support_reps:
        return _raw_failure(
            label, clicked_xyz_absolute_m, "LOCAL_POINT_CLOUD_TOO_SPARSE",
            dict(base_diag),
        )
    min_z = float(np.min(core[:, 2]))
    max_z = float(np.max(core[:, 2]))
    height_m = max_z - min_z
    diag: dict[str, Any] = {
        **base_diag,
        "core_point_count": int(len(core)),
        "vertical_extent_m": height_m,
        "refined_xy_absolute_m": [float(v) for v in np.median(core[:, :2], axis=0)],
    }
    if height_m < manual.refine_min_pole_height_m:
        return _raw_failure(
            label,
            clicked_xyz_absolute_m,
            f"INSUFFICIENT_VERTICAL_EXTENT height={height_m:.2f}m < "
            f"{manual.refine_min_pole_height_m}m",
            diag,
        )
    p_low = float(np.quantile(core[:, 0], 0.10))
    p_high = float(np.quantile(core[:, 0], 0.90))
    q_low = float(np.quantile(core[:, 1], 0.10))
    q_high = float(np.quantile(core[:, 1], 0.90))
    footprint = float(max(p_high - p_low, q_high - q_low))
    diag["xy_footprint_m"] = footprint
    if footprint > manual.refine_max_footprint_m:
        return _raw_failure(
            label,
            clicked_xyz_absolute_m,
            f"FOOTPRINT_TOO_WIDE xy_footprint={footprint:.2f}m > "
            f"{manual.refine_max_footprint_m}m",
            diag,
        )
    # Vertical continuity: stable XY centroids across 1 m height bands.
    band = manual.refine_band_height_m
    band_start = int(math.floor(min_z / band))
    band_stop = int(math.floor(max_z / band))
    centroids: list[tuple[float, float]] = []
    occupied: list[int] = []
    for band_index in range(band_start, band_stop + 1):
        lo = band_index * band
        hi = lo + band
        band_mask = (core[:, 2] >= lo) & (core[:, 2] < hi)
        if not np.any(band_mask):
            continue
        occupied.append(band_index)
        centroids.append(
            (float(np.mean(core[band_mask, 0])), float(np.mean(core[band_mask, 1])))
        )
    if not occupied:
        return _raw_failure(label, clicked_xyz_absolute_m, "NO_OCCUPIED_HEIGHT_BANDS", diag)
    max_gap = occupied[-1] - occupied[0] + 1 - len(occupied)
    if max_gap > manual.refine_max_vertical_gap_m / band:
        return _raw_failure(
            label, clicked_xyz_absolute_m,
            f"VERTICAL_CONTINUITY_BROKEN max_band_gap={max_gap * band:.2f}m", diag,
        )
    centroids_arr = np.asarray(centroids, dtype=np.float64)
    drift = float(np.max(np.linalg.norm(np.diff(centroids_arr, axis=0), axis=1)))
    if len(centroids_arr) >= 2 and drift > manual.refine_max_band_drift_m:
        return _raw_failure(
            label, clicked_xyz_absolute_m,
            f"SHAFT_DRIFT_TOO_LARGE band_centroid_drift={drift:.2f}m", diag,
        )
    # Shaft-like dimensionality from the CORE (the isolated pole column), not the
    # whole local box which surrounding vegetation would dilute.
    from algorithms.poles.evidence import local_pca_features

    features = local_pca_features(core, neighbor_count=cfg.wire_pca_neighbor_count)
    median_verticality = float(np.median(features["verticality"]))
    median_linearity = float(np.median(features["linearity"]))
    diag["median_verticality"] = median_verticality
    diag["median_linearity"] = median_linearity
    if not (
        median_verticality >= manual.refine_verticality_min
        or median_linearity >= manual.refine_linearity_min
    ):
        return _raw_failure(
            label, clicked_xyz_absolute_m,
            f"NOT_SHAFT_LIKE verticality={median_verticality:.2f} "
            f"linearity={median_linearity:.2f}",
            diag,
        )
    # Ground/base support: the shaft descends to near its own base elevation.
    reaches_ground = float(min_z - np.min(core[:, 2])) <= 1.5
    base_xy = np.median(core[:, :2], axis=0)
    bottom = core[core[:, 2] <= min_z + 1.5]
    support_reps = int(np.count_nonzero(
        np.linalg.norm(bottom[:, :2] - base_xy[None, :], axis=1)
        <= manual.refine_ground_contact_radius_m
    ))
    diag["ground_support_reps"] = support_reps
    if not reaches_ground and support_reps == 0:
        return _raw_failure(label, clicked_xyz_absolute_m, "NO_GROUND_SUPPORT", diag)
    base_xyz_absolute = (float(base_xy[0]), float(base_xy[1]), min_z)
    top_z = float(np.quantile(core[:, 2] - min_z, manual.refine_top_quantile))
    top_xyz_absolute = (float(base_xy[0]), float(base_xy[1]), min_z + top_z)
    diag["refined_xy_absolute_m"] = [float(base_xy[0]), float(base_xy[1])]
    diag["vertical_extent_m"] = height_m
    return ManualRefinedPole(
        label=label,
        clicked_xyz_absolute_m=clicked_xyz_absolute_m,
        base_xyz_local_m=tuple(float(v) for v in np.asarray(base_xyz_absolute) - origin),
        base_xyz_absolute_m=base_xyz_absolute,
        top_xyz_local_m=tuple(float(v) for v in np.asarray(top_xyz_absolute) - origin),
        top_xyz_absolute_m=top_xyz_absolute,
        height_m=float(top_xyz_absolute[2] - base_xyz_absolute[2]),
        xy_footprint_m=footprint,
        median_verticality=median_verticality,
        median_linearity=median_linearity,
        representative_count=int(len(core)),
        ground_support_reps=support_reps,
        defensible=True,
        reason=None,
        refinement_diagnostics=diag,
    )


def _within_radius_chunked(
    xy: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    chunk_size: int = 250_000,
) -> np.ndarray:
    """Boolean mask of *xy* points within *radius* of (cx, cy), computed in chunks.

    The squared distance is built one bounded chunk at a time, so a dense
    neighborhood (millions of points) never materializes a full-length float64
    distance array.  Produces the exact same mask as
    ``np.linalg.norm(xy - (cx, cy), axis=1) <= radius``.
    """

    radius2 = float(radius * radius)
    keep = np.zeros(len(xy), dtype=bool)
    for start in range(0, len(xy), chunk_size):
        stop = min(len(xy), start + chunk_size)
        block = xy[start:stop]
        d2 = (block[:, 0] - cx) ** 2 + (block[:, 1] - cy) ** 2
        keep[start:stop] = d2 <= radius2
    return keep


def _raw_failure(
    label: str,
    clicked_xyz_absolute_m: tuple[float, float, float],
    reason: str,
    diag: dict[str, Any],
) -> ManualRefinedPole:
    return ManualRefinedPole(
        label=label,
        clicked_xyz_absolute_m=clicked_xyz_absolute_m,
        base_xyz_local_m=clicked_xyz_absolute_m,
        base_xyz_absolute_m=clicked_xyz_absolute_m,
        top_xyz_local_m=clicked_xyz_absolute_m,
        top_xyz_absolute_m=clicked_xyz_absolute_m,
        height_m=0.0,
        xy_footprint_m=0.0,
        median_verticality=0.0,
        median_linearity=0.0,
        representative_count=0,
        ground_support_reps=0,
        defensible=False,
        reason=reason,
        refinement_diagnostics=diag,
    )


def _span_bounds(
    start_xy: np.ndarray,
    direction_xy: np.ndarray,
    length: float,
    half_width: float,
    s_lo: float,
    s_hi: float,
    z_lo: float,
    z_hi: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    perp = np.asarray((-direction_xy[1], direction_xy[0]), dtype=np.float64)
    corners_s = np.asarray((s_lo, s_lo, s_hi, s_hi), dtype=np.float64)
    corners_u = np.asarray((-half_width, half_width, -half_width, half_width), dtype=np.float64)
    corners = start_xy[:2][None, :] + corners_s[:, None] * direction_xy[None, :] + (
        corners_u[:, None] * perp[None, :]
    )
    minimum = np.asarray((np.min(corners[:, 0]), np.min(corners[:, 1]), z_lo))
    maximum = np.asarray((np.max(corners[:, 0]), np.max(corners[:, 1]), z_hi))
    return tuple(float(v) for v in minimum), tuple(float(v) for v in maximum)


def build_manual_span_record(
    pole_a: ManualRefinedPole,
    pole_b: ManualRefinedPole,
    cfg: FastBaselineConfig,
    manual: ManualSpanConfig,
    origin_xyz: tuple[float, float, float],
    *,
    span_id: int = 0,
) -> SpanRecord:
    """Build the single authoritative ``SpanRecord`` from two refined poles.

    Coordinates follow the existing topology contract: bounds are scene-local
    (origin ``origin_xyz``), ``*_absolute_m`` fields are in the source LAS
    frame.  The construction mirrors ``build_span_topology`` for one span.
    """

    origin = np.asarray(origin_xyz, dtype=np.float64)
    start_local = np.asarray(pole_a.base_xyz_local_m, dtype=np.float64)
    end_local = np.asarray(pole_b.base_xyz_local_m, dtype=np.float64)
    start_xy = start_local[:2]
    end_xy = end_local[:2]
    length = float(np.linalg.norm(end_xy - start_xy))
    if length < manual.span_min_length_m:
        raise ValueError(
            f"MANUAL_SPAN_TOO_SHORT length={length:.2f}m < "
            f"{manual.span_min_length_m}m"
        )
    direction_xy = _unit_xy(end_xy - start_xy)
    half_width = manual.span_half_width_m
    halo = manual.span_halo_m
    z_lo = min(float(start_local[2]), float(end_local[2])) - manual.span_z_ground_clearance_m
    z_hi = (
        max(float(pole_a.top_xyz_local_m[2]), float(pole_b.top_xyz_local_m[2]))
        + manual.span_z_top_margin_m
    )
    core_min, core_max = _span_bounds(
        start_xy, direction_xy, length, half_width, -0.5, length + 0.5, z_lo, z_hi
    )
    halo_min, halo_max = _span_bounds(
        start_xy, direction_xy, length, half_width, -halo, length + halo, z_lo, z_hi
    )
    line_name = "ManualLine"
    span_name = f"{line_name}-Span{span_id + 1:03d}"
    return SpanRecord(
        line_id=0,
        line_name=line_name,
        span_id=span_id,
        span_name=span_name,
        start_pole_candidate_id=-1,
        end_pole_candidate_id=-2,
        start_xyz_absolute_m=tuple(
            float(value) for value in (start_local + origin)
        ),
        end_xyz_absolute_m=tuple(float(value) for value in (end_local + origin)),
        start_pole_top_z_m=float(pole_a.top_xyz_absolute_m[2]),
        end_pole_top_z_m=float(pole_b.top_xyz_absolute_m[2]),
        length_m=length,
        direction_xy=(float(direction_xy[0]), float(direction_xy[1]), 0.0),
        half_width_m=half_width,
        halo_m=halo,
        core_bounds_min_xyz=core_min,
        core_bounds_max_xyz=core_max,
        halo_bounds_min_xyz=halo_min,
        halo_bounds_max_xyz=halo_max,
        quality="MANUAL_CONFIRMED_SPAN",
    )


def manual_span_selection_payload(
    pole_a: ManualRefinedPole,
    pole_b: ManualRefinedPole,
    span: SpanRecord,
    source_sha256: str,
    source_identity: str,
    origin_xyz: tuple[float, float, float],
    *,
    selection_version: str,
) -> dict[str, Any]:
    """Deterministic persisted manual selection (stale selection rejection)."""

    payload = {
        "schema_version": selection_version,
        "non_certified_marker": "NON_CERTIFIED_RESEARCH_DEMONSTRATION",
        "source_sha256": source_sha256,
        "source_identity": source_identity,
        "coordinate_frame": "SOURCE_LAS_ABSOLUTE",
        "pole_a": pole_a.to_dict(),
        "pole_b": pole_b.to_dict(),
        "span": span.to_dict(),
        "origin_xyz": list(origin_xyz),
        "version": 1,
    }
    return payload


def selection_digest(payload: dict[str, Any]) -> str:
    from algorithms.manual_span.config import canonical_digest

    identity = {
        "source_sha256": payload.get("source_sha256"),
        "pole_a": payload.get("pole_a"),
        "pole_b": payload.get("pole_b"),
        "span": payload.get("span"),
        "origin_xyz": payload.get("origin_xyz"),
    }
    return canonical_digest(identity)


__all__ = [
    "ManualRefinedPole",
    "refine_pole_from_click",
    "refine_pole_from_raw_cloud",
    "build_manual_span_record",
    "manual_span_selection_payload",
    "selection_digest",
]
