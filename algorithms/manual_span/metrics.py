"""Per-tree structural metrics from full-resolution raw members.

One shared ground KD-tree is used for a single chunked HAG pass over all
assigned full-resolution tree points.  Each tree then reports:

- ``tree_height_m`` — ground-relative HAG, robust high-end statistic (the 98th
  percentile of member HAG, so one isolated noise point never inflates it);
- ``tree_top_xyz`` / ``tree_centroid_xyz``;
- crown widths from horizontal XY geometry (major / minor / mean / ellipse
  area) — never a 3D box diagonal;
- DBH from the breast-height band stem evidence (see ``dbh``).

All outputs are scene-local; consumers add the origin for the source frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithms.manual_span.config import ManualSpanConfig
from algorithms.manual_span.dbh import DbhEvidence, classify_dbh
from algorithms.spans.reconcile import TreeReconciliationResult

TREE_HEIGHT_STATISTIC = "p98_of_ground_relative_hag_fullres"
CROWN_WIDTH_STATISTIC = "xy_p95_minus_p05_along_principal_axes"


@dataclass(frozen=True, slots=True)
class TreeMetrics:
    """Full set of per-tree structural metrics."""

    global_tree_id: int
    tree_height_m: float | None
    dbh: DbhEvidence
    crown_major_width_m: float
    crown_minor_width_m: float
    crown_mean_width_m: float
    crown_area_m2: float
    tree_top_xyz_local_m: tuple[float, float, float] | None
    tree_centroid_xyz_local_m: tuple[float, float, float] | None
    fullres_point_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_tree_id": self.global_tree_id,
            "tree_height_m": self.tree_height_m,
            "tree_height_statistic": TREE_HEIGHT_STATISTIC,
            "dbh_m": self.dbh.dbh_m,
            "dbh_quality": self.dbh.dbh_quality,
            "crown_major_width_m": self.crown_major_width_m,
            "crown_minor_width_m": self.crown_minor_width_m,
            "crown_mean_width_m": self.crown_mean_width_m,
            "crown_area_m2": self.crown_area_m2,
            "tree_top_xyz_local_m": (
                None if self.tree_top_xyz_local_m is None else list(self.tree_top_xyz_local_m)
            ),
            "tree_centroid_xyz_local_m": (
                None
                if self.tree_centroid_xyz_local_m is None
                else list(self.tree_centroid_xyz_local_m)
            ),
            "fullres_point_count": self.fullres_point_count,
        }


def _hag_chunked(
    xyz_local_m: np.ndarray,
    ground_tree: Any,
    ground_xyz_absolute: np.ndarray,
    origin_z: float,
    chunk_size: int,
) -> np.ndarray:
    out = np.empty(len(xyz_local_m), dtype=np.float32)
    for start in range(0, len(xyz_local_m), chunk_size):
        stop = min(len(xyz_local_m), start + chunk_size)
        block = xyz_local_m[start:stop]
        neighbors = ground_tree.query(block[:, :2], k=8)[1]
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        ground_z = np.median(ground_xyz_absolute[neighbors, 2], axis=1)
        out[start:stop] = (block[:, 2] + origin_z) - ground_z
    return out


def compute_tree_metrics(
    reconciled: TreeReconciliationResult,
    coarse_ground_xyz_absolute: np.ndarray,
    origin_xyz: tuple[float, float, float],
    manual: ManualSpanConfig,
) -> tuple[TreeMetrics, ...]:
    """Compute all structural metrics for every final tree (one HAG pass)."""

    from scipy.spatial import cKDTree

    ground = np.asarray(coarse_ground_xyz_absolute, dtype=np.float64)
    ground_tree = cKDTree(ground[:, :2])
    origin = np.asarray(origin_xyz, dtype=np.float64)
    final_xyz = np.asarray(reconciled.final_xyz_local_m, dtype=np.float64)
    final_tree = np.asarray(reconciled.final_tree_id, dtype=np.int32)
    tree_ids = sorted(set(int(value) for value in final_tree if value >= 0))
    results: list[TreeMetrics] = []
    for tree_id in tree_ids:
        rows = np.flatnonzero(final_tree == tree_id)
        points = final_xyz[rows]
        count = int(len(points))
        hag = _hag_chunked(points, ground_tree, ground, float(origin[2]), 1_000_000)

        height_m: float | None = None
        top_xyz_local: tuple[float, float, float] | None = None
        if count:
            height_m = float(np.quantile(hag, 0.98))
            high = hag >= float(np.quantile(hag, 0.98))
            if np.any(high):
                top_xyz_local = tuple(
                    float(value) for value in np.median(points[high], axis=0)
                )
        centroid_xyz_local = tuple(float(value) for value in np.mean(points, axis=0))

        # Crown geometry from horizontal XY (principal axes).
        xy = points[:, :2]
        centred = xy - np.mean(xy, axis=0)
        try:
            covariance = np.cov(centred, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            major_axis = eigenvectors[:, order[0]]
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate guard
            major_axis = np.asarray((1.0, 0.0))
        projection = centred @ major_axis
        minor_projection = centred @ np.asarray((-major_axis[1], major_axis[0]))
        major_width = float(
            np.quantile(projection, 0.95) - np.quantile(projection, 0.05)
        )
        minor_width = float(
            np.quantile(minor_projection, 0.95) - np.quantile(minor_projection, 0.05)
        )
        major = max(major_width, minor_width)
        minor = min(major_width, minor_width)
        mean_width = 0.5 * (major + minor)
        area = np.pi * (major / 2.0) * (minor / 2.0)

        # DBH from the breast-height band.
        base_z = float(np.quantile(points[:, 2], 0.02))
        trunk_mask = points[:, 2] <= base_z + 1.0
        base_xy = (
            np.median(points[trunk_mask, :2], axis=0)
            if np.count_nonzero(trunk_mask) >= 3
            else np.median(points[:, :2], axis=0)
        )
        band_mask = (hag >= manual.dbh_band_center_m - manual.dbh_band_half_width_m) & (
            hag <= manual.dbh_band_center_m + manual.dbh_band_half_width_m
        )
        band_count = int(np.count_nonzero(band_mask))
        if band_count:
            stem_xy = points[band_mask, :2]
            stem_xy = stem_xy[
                np.linalg.norm(stem_xy - base_xy[None, :], axis=1)
                <= manual.dbh_stem_radius_m
            ]
        else:
            stem_xy = None
        dbh = classify_dbh(stem_xy, band_count, manual)

        results.append(
            TreeMetrics(
                global_tree_id=tree_id,
                tree_height_m=height_m,
                dbh=dbh,
                crown_major_width_m=float(major),
                crown_minor_width_m=float(minor),
                crown_mean_width_m=float(mean_width),
                crown_area_m2=float(area),
                tree_top_xyz_local_m=top_xyz_local,
                tree_centroid_xyz_local_m=centroid_xyz_local,
                fullres_point_count=count,
            )
        )
    return tuple(results)


__all__ = [
    "TREE_HEIGHT_STATISTIC",
    "CROWN_WIDTH_STATISTIC",
    "TreeMetrics",
    "compute_tree_metrics",
]
