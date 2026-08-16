"""Bounded local RAW tower refinement.

An accepted candidate (or its review-adjusted seed) is refined exclusively from
the exact full-resolution RAW points returned by the local tile-cache query.
The Giro3D displayed points, COPC LOD nodes, and coarse representatives are
never authoritative here; only ``FULL_RESOLUTION_RAW`` geometry may feed the
defensible ``TowerGeometry``.

The algorithm preserves the validated manual pole-refinement concepts:

- the click/seed is only an approximate XY seed;
- a dense shaft is isolated from the RAW local box by voxelizing XY into small
  cells and keeping the cell with the largest *continuous* vertical extent;
- a tight core radius keeps shaft features from being diluted by vegetation;
- gates: minimum height, horizontal footprint, vertical continuity across 1 m
  bands, band-centroid drift, shaft-like PCA dimensionality, ground/base support;
- the base is the shaft XY at the core base elevation; the top is a robust high
  quantile of the core so an isolated noise point never inflates the height.

On any failed gate the refinement fails closed (``RAW_REFINEMENT_FAILED``): it
never invents geometry from insufficient evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import canonical_digest
from core.contracts.project import SourceIdentityBinding, TowerGeometry
from core.contracts.tower_candidates import TowerCandidateRawRefinementEvidence

from algorithms.towers.raw_access import RawTileQueryResult

TOWER_RAW_REFINEMENT_VERSION = "raw-refinement-v1"
TOWER_RAW_REFINEMENT_METHOD = "full-resolution-raw"


class TowerRawRefinementError(RuntimeError):
    """Closed failure raised by the RAW tower refinement boundary."""


@dataclass(frozen=True, slots=True)
class TowerRawRefinementConfig:
    """Fixed RAW tower-refinement parameters (sealed by a configuration digest)."""

    search_radius_m: float = 3.5
    z_context_below_m: float = 20.0
    z_context_above_m: float = 120.0
    min_pole_height_m: float = 5.0
    max_footprint_m: float = 1.6
    band_height_m: float = 1.0
    max_band_drift_m: float = 0.7
    max_vertical_gap_m: float = 3.0
    min_support_points: int = 5
    verticality_min: float = 0.50
    linearity_min: float = 0.30
    ground_contact_radius_m: float = 1.8
    top_quantile: float = 0.95
    shaft_cell_m: float = 0.30
    core_radius_m: float = 0.35
    base_support_band_m: float = 1.5

    def __post_init__(self) -> None:
        for name in (
            "search_radius_m",
            "z_context_below_m",
            "z_context_above_m",
            "min_pole_height_m",
            "max_footprint_m",
            "band_height_m",
            "max_band_drift_m",
            "max_vertical_gap_m",
            "ground_contact_radius_m",
            "top_quantile",
            "shaft_cell_m",
            "core_radius_m",
            "base_support_band_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if name == "top_quantile":
                if not 0.0 < float(value) < 1.0:
                    raise ValueError("top_quantile must be strictly inside (0, 1)")
            elif name in {"z_context_below_m", "z_context_above_m", "max_band_drift_m"}:
                if float(value) < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            else:
                if float(value) <= 0.0:
                    raise ValueError(f"{name} must be positive")
        if type(self.min_support_points) is not int or self.min_support_points <= 0:
            raise ValueError("min_support_points must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_radius_m": self.search_radius_m,
            "z_context_below_m": self.z_context_below_m,
            "z_context_above_m": self.z_context_above_m,
            "min_pole_height_m": self.min_pole_height_m,
            "max_footprint_m": self.max_footprint_m,
            "band_height_m": self.band_height_m,
            "max_band_drift_m": self.max_band_drift_m,
            "max_vertical_gap_m": self.max_vertical_gap_m,
            "min_support_points": self.min_support_points,
            "verticality_min": self.verticality_min,
            "linearity_min": self.linearity_min,
            "ground_contact_radius_m": self.ground_contact_radius_m,
            "top_quantile": self.top_quantile,
            "shaft_cell_m": self.shaft_cell_m,
            "core_radius_m": self.core_radius_m,
            "base_support_band_m": self.base_support_band_m,
        }

    def configuration_digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class TowerRawRefinementResult:
    """Result of one bounded RAW refinement attempt (success or closed failure)."""

    success: bool
    geometry: TowerGeometry | None
    evidence: TowerCandidateRawRefinementEvidence | None
    failure_reason: str | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "geometry": self.geometry.to_dict() if self.geometry is not None else None,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "failure_reason": self.failure_reason,
            "diagnostics": dict(self.diagnostics),
        }


def _failure(reason: str, diagnostics: dict[str, Any]) -> TowerRawRefinementResult:
    return TowerRawRefinementResult(
        success=False,
        geometry=None,
        evidence=None,
        failure_reason=reason,
        diagnostics=dict(diagnostics),
    )


def build_tower_raw_refinement_evidence(
    source_binding: SourceIdentityBinding,
    *,
    details: dict[str, Any],
    algorithm_version: str = TOWER_RAW_REFINEMENT_VERSION,
) -> TowerCandidateRawRefinementEvidence:
    """Build the closed RAW refinement evidence bound to the authoritative source."""
    if type(source_binding) is not SourceIdentityBinding:
        raise ValueError("source_binding must be a SourceIdentityBinding")
    return TowerCandidateRawRefinementEvidence(
        method=TOWER_RAW_REFINEMENT_METHOD,
        algorithm_version=algorithm_version,
        source_content_identity_digest=source_binding.source_content_identity_digest,
        source_identity_record_digest=source_binding.source_identity_record_digest,
        details=details,
    )


def _select_shaft_cell(
    local: np.ndarray,
    *,
    shaft_cell_m: float,
    min_support_points: int,
    max_vertical_gap_m: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Pick the XY cell with the largest continuous vertical extent as shaft seed."""
    keys = np.floor(local[:, :2] / shaft_cell_m).astype(np.int64)
    order = np.lexsort((local[:, 2], keys[:, 0], keys[:, 1]))
    sorted_keys = keys[order]
    sorted_z = local[order, 2]
    boundaries = np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    best_cell: tuple[int, int] | None = None
    best_score = -1.0
    best_extent_fallback = 0.0
    fallback_cell: tuple[int, int] | None = None
    for group in range(len(starts)):
        if int(ends[group] - starts[group]) < min_support_points:
            continue
        zvals = sorted_z[starts[group] : ends[group]]
        extent = float(zvals[-1] - zvals[0])
        occupied_bands = sorted(set(int(value // 1.0) for value in zvals))
        max_gap = max(
            (
                occupied_bands[pos + 1] - occupied_bands[pos]
                for pos in range(len(occupied_bands) - 1)
            ),
            default=0,
        )
        # Prefer CONTINUOUS tall columns (a tower) over a fragmented stack of low
        # structure + high canopy (a tree), which has a large extent but a gap.
        score = extent if max_gap <= max_vertical_gap_m else 0.0
        if score > best_score:
            best_score = score
            best_cell = (int(sorted_keys[starts[group], 0]), int(sorted_keys[starts[group], 1]))
        if extent > best_extent_fallback:
            best_extent_fallback = extent
            fallback_cell = (int(sorted_keys[starts[group], 0]), int(sorted_keys[starts[group], 1]))
    if best_cell is None:
        return None, {"shaft_seed": None}
    if best_score <= 0.0:
        best_cell = fallback_cell
    seed_xy = (np.asarray(best_cell, dtype=np.float64) + 0.5) * shaft_cell_m
    return seed_xy, {
        "shaft_seed": [float(value) for value in seed_xy],
        "shaft_cell_m": shaft_cell_m,
    }


def refine_tower_from_raw(
    query: RawTileQueryResult,
    source_binding: SourceIdentityBinding,
    config: TowerRawRefinementConfig | None = None,
) -> TowerRawRefinementResult:
    """Refine one approximate seed against bounded exact RAW points.

    ``query`` must already be a source-validated local RAW query around the
    seed.  The returned geometry is in ``SOURCE_LAS_ABSOLUTE`` metres; the
    evidence binds the refinement to the authoritative source identity.
    """
    if type(query) is not RawTileQueryResult:
        raise ValueError("query must be a RawTileQueryResult")
    if type(source_binding) is not SourceIdentityBinding:
        raise ValueError("source_binding must be a SourceIdentityBinding")
    if config is None:
        config = TowerRawRefinementConfig()
    if type(config) is not TowerRawRefinementConfig:
        raise ValueError("config must be a TowerRawRefinementConfig")

    seed = np.asarray(query.seed_xyz_absolute_m, dtype=np.float64)
    points = query.xyz
    indices = query.source_index
    base_diag: dict[str, Any] = {
        "raw_query": {
            "seed_xyz_absolute_m": list(query.seed_xyz_absolute_m),
            "xy_radius_m": query.xy_radius_m,
            "z_window_min_m": query.z_window_min_m,
            "z_window_max_m": query.z_window_max_m,
            "tiles_used": list(query.tiles),
            "raw_point_count": query.point_count,
        },
        "configuration_digest": config.configuration_digest(),
    }
    if query.point_count == 0:
        return _failure("LOCAL_POINT_CLOUD_TOO_SPARSE", base_diag)
    local_mask = (
        np.linalg.norm(points[:, :2] - seed[:2][None, :], axis=1)
        <= config.search_radius_m
    )
    local = points[local_mask]
    local_idx = indices[local_mask]
    base_diag["local_point_count"] = int(len(local))
    base_diag["vertical_extent_local_m"] = float(np.ptp(local[:, 2])) if len(local) else 0.0
    if len(local) < config.min_support_points:
        return _failure("LOCAL_POINT_CLOUD_TOO_SPARSE", base_diag)

    seed_xy, shaft_diag = _select_shaft_cell(
        local,
        shaft_cell_m=config.shaft_cell_m,
        min_support_points=config.min_support_points,
        max_vertical_gap_m=config.max_vertical_gap_m,
    )
    base_diag.update(shaft_diag)
    if seed_xy is None:
        return _failure("LOCAL_POINT_CLOUD_TOO_SPARSE", base_diag)
    core_mask = (
        np.linalg.norm(local[:, :2] - seed_xy[None, :], axis=1) <= config.core_radius_m
    )
    core = local[core_mask]
    core_idx = local_idx[core_mask]
    if len(core) < config.min_support_points:
        return _failure("LOCAL_POINT_CLOUD_TOO_SPARSE", base_diag)
    min_z = float(np.min(core[:, 2]))
    max_z = float(np.max(core[:, 2]))
    height_m = max_z - min_z
    base_diag["core_point_count"] = int(len(core))
    base_diag["vertical_extent_m"] = height_m
    if height_m < config.min_pole_height_m:
        return _failure(
            f"INSUFFICIENT_VERTICAL_EXTENT height={height_m:.2f}m < "
            f"{config.min_pole_height_m}m",
            base_diag,
        )
    p_low = float(np.quantile(core[:, 0], 0.10))
    p_high = float(np.quantile(core[:, 0], 0.90))
    q_low = float(np.quantile(core[:, 1], 0.10))
    q_high = float(np.quantile(core[:, 1], 0.90))
    footprint = float(max(p_high - p_low, q_high - q_low))
    base_diag["xy_footprint_m"] = footprint
    if footprint > config.max_footprint_m:
        return _failure(
            f"FOOTPRINT_TOO_WIDE xy_footprint={footprint:.2f}m > {config.max_footprint_m}m",
            base_diag,
        )

    band = config.band_height_m
    band_start = int(math.floor(min_z / band))
    band_stop = int(math.floor(max_z / band))
    centroids: list[np.ndarray] = []
    occupied: list[int] = []
    for band_index in range(band_start, band_stop + 1):
        lo = band_index * band
        hi = lo + band
        band_mask = (core[:, 2] >= lo) & (core[:, 2] < hi)
        if not np.any(band_mask):
            continue
        occupied.append(band_index)
        centroids.append(np.mean(core[band_mask, :2], axis=0))
    if not occupied:
        return _failure("NO_OCCUPIED_HEIGHT_BANDS", base_diag)
    max_band_gap = occupied[-1] - occupied[0] + 1 - len(occupied)
    base_diag["vertical_continuity_max_band_gap_m"] = max_band_gap * band
    if max_band_gap > config.max_vertical_gap_m / band:
        return _failure(
            f"VERTICAL_CONTINUITY_BROKEN max_band_gap={max_band_gap * band:.2f}m",
            base_diag,
        )
    centroids_arr = np.asarray(centroids, dtype=np.float64)
    drift = float(np.max(np.linalg.norm(np.diff(centroids_arr, axis=0), axis=1)))
    base_diag["shaft_band_drift_m"] = drift
    if len(centroids_arr) >= 2 and drift > config.max_band_drift_m:
        return _failure(
            f"SHAFT_DRIFT_TOO_LARGE band_centroid_drift={drift:.2f}m", base_diag
        )

    from algorithms.poles.evidence import local_pca_features

    features = local_pca_features(core, neighbor_count=16)
    median_verticality = float(np.median(features["verticality"]))
    median_linearity = float(np.median(features["linearity"]))
    base_diag["median_verticality"] = median_verticality
    base_diag["median_linearity"] = median_linearity
    if not (
        median_verticality >= config.verticality_min
        or median_linearity >= config.linearity_min
    ):
        return _failure(
            f"NOT_SHAFT_LIKE verticality={median_verticality:.2f} "
            f"linearity={median_linearity:.2f}",
            base_diag,
        )

    base_xy = np.median(core[:, :2], axis=0)
    bottom = core[core[:, 2] <= min_z + config.base_support_band_m]
    support_reps = int(
        np.count_nonzero(
            np.linalg.norm(bottom[:, :2] - base_xy[None, :], axis=1)
            <= config.ground_contact_radius_m
        )
    )
    base_diag["ground_support_points"] = support_reps
    if support_reps == 0:
        return _failure("NO_GROUND_SUPPORT", base_diag)

    base_xyz_absolute = (float(base_xy[0]), float(base_xy[1]), min_z)
    top_z = float(np.quantile(core[:, 2] - min_z, config.top_quantile))
    top_xyz_absolute = (float(base_xy[0]), float(base_xy[1]), min_z + top_z)
    geometry = TowerGeometry(
        base_xyz_absolute_m=base_xyz_absolute,
        top_xyz_absolute_m=top_xyz_absolute,
        height_m=float(top_xyz_absolute[2] - base_xyz_absolute[2]),
    )

    # Preserve exact original source row identities for evidence (bounded sample;
    # the frozen evidence contract caps metadata sequences at 128 entries).
    core_indices = [int(value) for value in core_idx[:128]]
    details: dict[str, Any] = {
        **base_diag,
        "method": TOWER_RAW_REFINEMENT_METHOD,
        "algorithm_version": TOWER_RAW_REFINEMENT_VERSION,
        "shaft": {
            "shaft_cell_m": config.shaft_cell_m,
            "core_radius_m": config.core_radius_m,
            "core_point_count": int(len(core)),
        },
        "base_xyz_absolute_m": list(base_xyz_absolute),
        "top_xyz_absolute_m": list(top_xyz_absolute),
        "height_m": geometry.height_m,
        "top_quantile": config.top_quantile,
        "source_original_index_count": int(len(core_idx)),
        "source_original_indices": core_indices,
    }
    evidence = build_tower_raw_refinement_evidence(source_binding, details=details)
    diagnostics = dict(details)
    diagnostics["success"] = True
    return TowerRawRefinementResult(
        success=True,
        geometry=geometry,
        evidence=evidence,
        failure_reason=None,
        diagnostics=diagnostics,
    )


__all__ = [
    "TOWER_RAW_REFINEMENT_METHOD",
    "TOWER_RAW_REFINEMENT_VERSION",
    "TowerRawRefinementConfig",
    "TowerRawRefinementError",
    "TowerRawRefinementResult",
    "build_tower_raw_refinement_evidence",
    "refine_tower_from_raw",
]
