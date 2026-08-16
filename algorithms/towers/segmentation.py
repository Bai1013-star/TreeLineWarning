"""V5A3 local exact RAW tower membership segmentation (seed-driven, V1).

The Owner clicks an approximate tower position; this module isolates the
*actual* full-resolution RAW member points belonging to that local structure and
returns them as an exact ``TowerLocalMembership``.  Only
``FULL_RESOLUTION_RAW`` points (the exact float64 XYZ and original source row
identities returned by the bounded tile-cache query) are ever authoritative
here.  Giro3D / COPC LOD nodes and coarse representatives never contribute
membership.

V1 deliberately keeps segmentation simple and seed-driven:

1. the click/seed is only an approximate XY prior;
2. ground / obvious low clutter are rejected by seeding the connectivity growth
   from the shaft core *above* the base band and growing strictly upward;
3. the vertical shaft/support nearest the Owner seed is found through the proven
   continuous-vertical-extent cell search (seed-biased so a taller disconnected
   clutter column elsewhere does not win);
4. the existing defensive gates (vertical extent, PCA verticality, compact
   footprint, continuity, drift, base support) decide whether a defensible
   structure exists at all;
5. a 3D voxel connectivity growth from the shaft core retains the structurally
   connected shaft and attached upper support / cross-arm points and rejects
   disconnected vegetation / canopy / ground / clutter.

On any failed gate the segmentation fails closed (``LOCAL_SEGMENTATION_FAILED``):
it never invents membership from insufficient evidence, never allocates a
``T###``, and never fabricates fallback geometry.

The result is deliberately independent of persistence: ``TowerLocalMembership``
is a pure in-memory scientific result that the ordered workflow may preview and
later persist as an immutable content-addressed artifact.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import canonical_digest
from core.contracts.project import SourceIdentityBinding

from algorithms.towers.raw_access import RawTileQueryResult

TOWER_LOCAL_SEGMENTATION_VERSION = "tower-local-segmentation-v1"
TOWER_LOCAL_SEGMENTATION_SCHEMA = "treeline-tower-local-membership-v1"
TOWER_LOCAL_SEGMENTATION_FAILURE = "LOCAL_SEGMENTATION_FAILED"


class TowerLocalSegmentationError(RuntimeError):
    """Closed failure raised by the local tower-membership segmentation boundary."""


@dataclass(frozen=True, slots=True)
class TowerLocalSegmentationConfig:
    """Fixed seed-driven membership-segmentation parameters (sealed by a digest)."""

    search_radius_m: float = 3.5
    z_context_below_m: float = 20.0
    z_context_above_m: float = 120.0
    shaft_cell_m: float = 0.30
    core_radius_m: float = 0.35
    voxel_m: float = 0.25
    min_pole_height_m: float = 5.0
    max_footprint_m: float = 1.6
    band_height_m: float = 1.0
    max_band_drift_m: float = 0.7
    max_vertical_gap_m: float = 3.0
    min_support_points: int = 5
    verticality_min: float = 0.50
    linearity_min: float = 0.30
    ground_contact_radius_m: float = 1.8
    base_support_band_m: float = 1.5
    seed_elevation_m: float = 1.0
    top_quantile: float = 0.95
    max_member_radius_m: float = 2.5
    max_member_height_m: float = 80.0
    max_crossarm_extension_m: float = 15.0

    def __post_init__(self) -> None:
        for name in (
            "search_radius_m",
            "z_context_below_m",
            "z_context_above_m",
            "shaft_cell_m",
            "core_radius_m",
            "voxel_m",
            "min_pole_height_m",
            "max_footprint_m",
            "band_height_m",
            "max_band_drift_m",
            "max_vertical_gap_m",
            "ground_contact_radius_m",
            "base_support_band_m",
            "seed_elevation_m",
            "top_quantile",
            "max_member_radius_m",
            "max_member_height_m",
            "max_crossarm_extension_m",
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
            "schema_version": TOWER_LOCAL_SEGMENTATION_SCHEMA,
            "search_radius_m": self.search_radius_m,
            "z_context_below_m": self.z_context_below_m,
            "z_context_above_m": self.z_context_above_m,
            "shaft_cell_m": self.shaft_cell_m,
            "core_radius_m": self.core_radius_m,
            "voxel_m": self.voxel_m,
            "min_pole_height_m": self.min_pole_height_m,
            "max_footprint_m": self.max_footprint_m,
            "band_height_m": self.band_height_m,
            "max_band_drift_m": self.max_band_drift_m,
            "max_vertical_gap_m": self.max_vertical_gap_m,
            "min_support_points": self.min_support_points,
            "verticality_min": self.verticality_min,
            "linearity_min": self.linearity_min,
            "ground_contact_radius_m": self.ground_contact_radius_m,
            "base_support_band_m": self.base_support_band_m,
            "seed_elevation_m": self.seed_elevation_m,
            "top_quantile": self.top_quantile,
            "max_member_radius_m": self.max_member_radius_m,
            "max_member_height_m": self.max_member_height_m,
            "max_crossarm_extension_m": self.max_crossarm_extension_m,
        }

    def configuration_digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class TowerLocalMembership:
    """Exact full-resolution RAW member points for one local tower structure.

    ``source_index`` carries the exact original source row identities and
    ``xyz_absolute_m`` the exact SOURCE_LAS_ABSOLUTE XYZ of the same rows, so
    every authoritative member point corresponds one-to-one to RAW source
    points.  ``member_bounds_absolute_m`` is the inclusive axis-aligned bounds
    of the member points.  The refined base/top geometry comes from the
    defensible shaft core (never from Giro3D / COPC LOD geometry).
    """

    source_binding: SourceIdentityBinding
    segmentation_config: TowerLocalSegmentationConfig
    config_digest: str
    source_index: np.ndarray  # int64 (n,)
    xyz_absolute_m: np.ndarray  # float64 (n, 3)
    member_count: int
    member_bounds_absolute_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    base_xyz_absolute_m: tuple[float, float, float]
    top_xyz_absolute_m: tuple[float, float, float]
    height_m: float
    raw_tile_cache_identity: dict[str, Any]
    quality_diagnostics: dict[str, Any]
    algorithm_version: str = TOWER_LOCAL_SEGMENTATION_VERSION

    def __post_init__(self) -> None:
        if type(self.source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        if type(self.segmentation_config) is not TowerLocalSegmentationConfig:
            raise ValueError("segmentation_config must be a TowerLocalSegmentationConfig")
        if not isinstance(self.source_index, np.ndarray) or self.source_index.dtype != np.int64:
            raise ValueError("source_index must be an int64 numpy array")
        if self.source_index.ndim != 1:
            raise ValueError("source_index must be a 1-D array")
        if (
            not isinstance(self.xyz_absolute_m, np.ndarray)
            or self.xyz_absolute_m.dtype != np.float64
        ):
            raise ValueError("xyz_absolute_m must be a float64 numpy array")
        if self.xyz_absolute_m.ndim != 2 or self.xyz_absolute_m.shape[1] != 3:
            raise ValueError("xyz_absolute_m must be an (n, 3) array")
        if len(self.source_index) != len(self.xyz_absolute_m):
            raise ValueError("source_index and xyz_absolute_m must have equal length")
        if type(self.member_count) is not int or self.member_count != len(self.source_index):
            raise ValueError("member_count must exactly match the number of member points")
        if not np.isfinite(self.source_index).all():
            raise ValueError("source_index must be finite")
        if not np.isfinite(self.xyz_absolute_m).all():
            raise ValueError("xyz_absolute_m must be finite")
        minimum, maximum = self.member_bounds_absolute_m
        if (
            any(lower > upper for lower, upper in zip(minimum, maximum, strict=True))
            or any(
                not math.isfinite(value) or math.isnan(value)
                for value in (*minimum, *maximum)
            )
        ):
            raise ValueError("member_bounds_absolute_m must be finite inclusive bounds")
        if self.member_count > 0:
            computed_min = tuple(float(value) for value in np.min(self.xyz_absolute_m, axis=0))
            computed_max = tuple(float(value) for value in np.max(self.xyz_absolute_m, axis=0))
            if tuple(minimum) != computed_min or tuple(maximum) != computed_max:
                raise ValueError("member_bounds_absolute_m must match the member point extents")
        if not all(
            math.isfinite(value) and value == value
            for value in (*self.base_xyz_absolute_m, *self.top_xyz_absolute_m)
        ):
            raise ValueError("base/top geometry must be finite")
        if type(self.height_m) is not float or self.height_m <= 0.0:
            raise ValueError("height_m must be a positive finite float")
        if self.top_xyz_absolute_m[2] - self.base_xyz_absolute_m[2] != self.height_m:
            raise ValueError("height_m must equal the base-to-top Z difference")
        if type(self.config_digest) is not str or len(self.config_digest) != 64:
            raise ValueError("config_digest must be a SHA-256 digest")

    @property
    def member_bounds_min_xyz(self) -> tuple[float, float, float]:
        return self.member_bounds_absolute_m[0]

    @property
    def member_bounds_max_xyz(self) -> tuple[float, float, float]:
        return self.member_bounds_absolute_m[1]

    def to_digest_dict(self) -> dict[str, Any]:
        """Deterministic digest payload: config + source binding + exact arrays."""
        order = np.argsort(self.source_index, kind="stable")
        return {
            "schema_version": TOWER_LOCAL_SEGMENTATION_SCHEMA,
            "algorithm_version": self.algorithm_version,
            "config_digest": self.config_digest,
            "source_content_identity_digest": (
                self.source_binding.source_content_identity_digest
            ),
            "source_identity_record_digest": self.source_binding.source_identity_record_digest,
            "member_count": self.member_count,
            "source_index": self.source_index[order].tolist(),
            "xyz_absolute_m": self.xyz_absolute_m[order].tolist(),
        }

    def membership_digest(self) -> str:
        """Content-addressed SHA-256 over the exact membership (self-excluding)."""
        payload = json_dumps_bytes(self.to_digest_dict())
        return hashlib.sha256(payload).hexdigest()


def json_dumps_bytes(value: Any) -> bytes:
    import json

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TowerLocalSegmentationResult:
    """Outcome of one local segmentation attempt (success or closed failure)."""

    success: bool
    membership: TowerLocalMembership | None
    failure_reason: str | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure_reason": self.failure_reason,
            "diagnostics": dict(self.diagnostics),
        }


def _failure(reason: str, diagnostics: dict[str, Any]) -> TowerLocalSegmentationResult:
    return TowerLocalSegmentationResult(
        success=False,
        membership=None,
        failure_reason=reason,
        diagnostics=dict(diagnostics),
    )


def _sparse_failure(diagnostics: dict[str, Any]) -> TowerLocalSegmentationResult:
    return _failure(
        f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:LOCAL_POINT_CLOUD_TOO_SPARSE",
        diagnostics,
    )


def _select_shaft_cell_seed_biased(
    local: np.ndarray,
    seed_xy: np.ndarray,
    *,
    shaft_cell_m: float,
    min_support_points: int,
    max_vertical_gap_m: float,
    min_height_m: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Pick the continuous vertical cell nearest the Owner seed.

    Every candidate cell with a continuous vertical extent >= ``min_height_m``
    is scored by (extent, seed distance); the nearest tall column wins so a
    taller *disconnected* clutter column elsewhere in the search box does not
    override the Owner's chosen structure.  Returns the cell XY.
    """
    keys = np.floor(local[:, :2] / shaft_cell_m).astype(np.int64)
    order = np.lexsort((local[:, 2], keys[:, 0], keys[:, 1]))
    sorted_keys = keys[order]
    sorted_z = local[order, 2]
    boundaries = np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))

    best_cell: tuple[int, int] | None = None
    best_score: tuple[float, float] | None = None
    best_extent = 0.0
    for group in range(len(starts)):
        if int(ends[group] - starts[group]) < min_support_points:
            continue
        zvals = sorted_z[starts[group] : ends[group]]
        extent = float(zvals[-1] - zvals[0])
        if extent < min_height_m:
            continue
        cell_xy = (np.asarray(sorted_keys[starts[group]], dtype=np.float64) + 0.5) * shaft_cell_m
        distance = float(np.hypot(cell_xy[0] - seed_xy[0], cell_xy[1] - seed_xy[1]))
        # Prefer the nearest tall column; tie-break by larger extent.
        score = (distance, -extent)
        if best_score is None or score < best_score:
            best_score = score
            best_cell = (int(sorted_keys[starts[group], 0]), int(sorted_keys[starts[group], 1]))
            best_extent = extent
    if best_cell is None:
        return None, {"shaft_seed": None}
    seed_xy_result = (np.asarray(best_cell, dtype=np.float64) + 0.5) * shaft_cell_m
    return seed_xy_result, {
        "shaft_seed": [float(value) for value in seed_xy_result],
        "shaft_cell_m": shaft_cell_m,
        "shaft_extent_m": best_extent,
    }


def _upward_connected_members(
    local: np.ndarray,
    local_idx: np.ndarray,
    seed_local_indices: np.ndarray,
    base_xy: np.ndarray,
    *,
    voxel_m: float,
    base_z: float,
    seed_floor_z: float,
    max_member_radius_m: float,
    max_member_height_m: float,
    voxel_origin: np.ndarray,
) -> np.ndarray:
    """3D voxel connectivity grown strictly upward from the elevated shaft core.

    Returns the local point indices that are structurally connected to the shaft
    core through the upward expansion.  Ground below the base band is never
    reached (the seed starts above it and expansion never descends across the
    band), and disconnected vegetation / canopy / clutter is not connected
    through space to the shaft.
    """
    voxel_floor = np.floor((local - voxel_origin) / voxel_m).astype(np.int64)
    voxel_map: dict[tuple[int, int, int], list[int]] = {}
    for index in range(len(local)):
        key = (int(voxel_floor[index, 0]), int(voxel_floor[index, 1]), int(voxel_floor[index, 2]))
        voxel_map.setdefault(key, []).append(index)
    # Sort each voxel's point list by source identity for deterministic growth.
    for key in voxel_map:
        voxel_map[key].sort(key=lambda index: int(local_idx[index]))

    members: set[int] = set()
    queue: list[tuple[int, int, int]] = []
    seed_keys: set[tuple[int, int, int]] = set()
    for index in seed_local_indices:
        if local[index, 2] < seed_floor_z:
            continue
        members.add(int(index))
        key = (int(voxel_floor[index, 0]), int(voxel_floor[index, 1]), int(voxel_floor[index, 2]))
        if key not in seed_keys:
            seed_keys.add(key)
            queue.append(key)
    queue_iter = 0
    # 26-neighbour offsets in a fixed deterministic order.
    offsets: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    seen: set[tuple[int, int, int]] = set(seed_keys)
    radius_sq = max_member_radius_m * max_member_radius_m
    while queue_iter < len(queue):
        key = queue[queue_iter]
        queue_iter += 1
        for offset in offsets:
            neighbor = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
            if neighbor in seen:
                continue
            points = voxel_map.get(neighbor)
            if points is None:
                continue
            seen.add(neighbor)
            accepted_any = False
            for index in points:
                if index in members:
                    accepted_any = True
                    continue
                z = float(local[index, 2])
                # Strictly upward growth: never descend across the base band.
                if z < seed_floor_z - voxel_m:
                    continue
                if z > base_z + max_member_height_m:
                    continue
                horizontal = float(
                    (local[index, 0] - base_xy[0]) ** 2 + (local[index, 1] - base_xy[1]) ** 2
                )
                if horizontal > radius_sq:
                    continue
                members.add(int(index))
                accepted_any = True
            if accepted_any:
                queue.append(neighbor)
    if not members:
        return np.empty(0, dtype=np.int64)
    return np.asarray(sorted(members), dtype=np.int64)


def _local_pca_shaft_like(core: np.ndarray, config: TowerLocalSegmentationConfig) -> dict[str, Any]:
    from algorithms.poles.evidence import local_pca_features

    features = local_pca_features(core, neighbor_count=16)
    median_verticality = float(np.median(features["verticality"]))
    median_linearity = float(np.median(features["linearity"]))
    return {
        "median_verticality": median_verticality,
        "median_linearity": median_linearity,
        "verticality_min": config.verticality_min,
        "linearity_min": config.linearity_min,
    }


def segment_tower_membership(
    query: RawTileQueryResult,
    source_binding: SourceIdentityBinding,
    config: TowerLocalSegmentationConfig | None = None,
) -> TowerLocalSegmentationResult:
    """Segment the local exact RAW membership around one approximate Owner seed.

    ``query`` must already be a source-validated bounded RAW query around the
    seed.  All returned member points are exact FULL_RESOLUTION_RAW points with
    their original source row identities preserved.
    """
    if type(query) is not RawTileQueryResult:
        raise ValueError("query must be a RawTileQueryResult")
    if type(source_binding) is not SourceIdentityBinding:
        raise ValueError("source_binding must be a SourceIdentityBinding")
    if config is None:
        config = TowerLocalSegmentationConfig()
    if type(config) is not TowerLocalSegmentationConfig:
        raise ValueError("config must be a TowerLocalSegmentationConfig")

    import time as _time

    started = _time.perf_counter()
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
        "algorithm_version": TOWER_LOCAL_SEGMENTATION_VERSION,
    }
    if query.point_count == 0:
        return _sparse_failure(base_diag)
    if not np.isfinite(points).all() or not np.isfinite(indices).all():
        return _failure(f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:NON_FINITE_RAW_POINTS", base_diag)

    local_mask = (
        np.linalg.norm(points[:, :2] - seed[:2][None, :], axis=1) <= config.search_radius_m
    )
    local = points[local_mask]
    local_idx = indices[local_mask]
    base_diag["local_point_count"] = int(len(local))
    if len(local) < config.min_support_points:
        return _sparse_failure(base_diag)

    seed_xy, shaft_diag = _select_shaft_cell_seed_biased(
        local,
        seed[:2],
        shaft_cell_m=config.shaft_cell_m,
        min_support_points=config.min_support_points,
        max_vertical_gap_m=config.max_vertical_gap_m,
        min_height_m=config.min_pole_height_m,
    )
    base_diag.update(shaft_diag)
    if seed_xy is None:
        return _failure(f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:NO_VALID_SHAFT_CELL", base_diag)

    core_mask = (
        np.linalg.norm(local[:, :2] - seed_xy[None, :], axis=1) <= config.core_radius_m
    )
    core = local[core_mask]
    core_local_positions = np.flatnonzero(core_mask)
    if len(core) < config.min_support_points:
        return _sparse_failure(base_diag)
    min_z = float(np.min(core[:, 2]))
    max_z = float(np.max(core[:, 2]))
    height_m = max_z - min_z
    base_diag["core_point_count"] = int(len(core))
    base_diag["vertical_extent_m"] = height_m
    if height_m < config.min_pole_height_m:
        return _failure(
            f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:INSUFFICIENT_VERTICAL_EXTENT "
            f"height={height_m:.2f}m < {config.min_pole_height_m}m",
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
            f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:FOOTPRINT_TOO_WIDE "
            f"xy_footprint={footprint:.2f}m > {config.max_footprint_m}m",
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
        return _failure(f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:NO_OCCUPIED_HEIGHT_BANDS", base_diag)
    max_band_gap = occupied[-1] - occupied[0] + 1 - len(occupied)
    base_diag["vertical_continuity_max_band_gap_m"] = max_band_gap * band
    if max_band_gap > config.max_vertical_gap_m / band:
        return _failure(
            f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:VERTICAL_CONTINUITY_BROKEN "
            f"max_band_gap={max_band_gap * band:.2f}m",
            base_diag,
        )
    centroids_arr = np.asarray(centroids, dtype=np.float64)
    drift = float(np.max(np.linalg.norm(np.diff(centroids_arr, axis=0), axis=1)))
    base_diag["shaft_band_drift_m"] = drift
    if len(centroids_arr) >= 2 and drift > config.max_band_drift_m:
        return _failure(
            f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:SHAFT_DRIFT_TOO_LARGE "
            f"band_centroid_drift={drift:.2f}m",
            base_diag,
        )

    pca = _local_pca_shaft_like(core, config)
    base_diag.update(pca)
    if not (
        pca["median_verticality"] >= config.verticality_min
        or pca["median_linearity"] >= config.linearity_min
    ):
        return _failure(
            f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:NOT_SHAFT_LIKE "
            f"verticality={pca['median_verticality']:.2f} linearity={pca['median_linearity']:.2f}",
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
        return _failure(f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:NO_GROUND_SUPPORT", base_diag)

    seed_floor_z = min_z + config.seed_elevation_m
    voxel_origin = np.min(local, axis=0)
    member_indices = _upward_connected_members(
        local,
        local_idx,
        seed_local_indices=core_local_positions,
        base_xy=base_xy,
        voxel_m=config.voxel_m,
        base_z=min_z,
        seed_floor_z=seed_floor_z,
        max_member_radius_m=config.max_member_radius_m,
        max_member_height_m=config.max_member_height_m,
        voxel_origin=voxel_origin,
    )
    if len(member_indices) < config.min_support_points:
        return _failure(
            f"{TOWER_LOCAL_SEGMENTATION_FAILURE}:MEMBERSHIP_TOO_SPARSE",
            {**base_diag, "member_point_count": int(len(member_indices))},
        )
    member_xyz = local[member_indices]
    member_idx = local_idx[member_indices]
    member_min = tuple(float(value) for value in np.min(member_xyz, axis=0))
    member_max = tuple(float(value) for value in np.max(member_xyz, axis=0))
    # Member points ABOVE the shaft core top are attached upper support/cross-arm
    # structure (structurally connected through the upward growth).
    crossarm_above = int(np.count_nonzero(member_xyz[:, 2] > max_z))
    base_diag["member_point_count"] = int(len(member_indices))
    base_diag["crossarm_above_core_points"] = crossarm_above
    base_diag["member_bounds_min_xyz"] = list(member_min)
    base_diag["member_bounds_max_xyz"] = list(member_max)

    top_z = float(np.quantile(core[:, 2] - min_z, config.top_quantile))
    base_xyz_absolute = (float(base_xy[0]), float(base_xy[1]), min_z)
    top_xyz_absolute = (float(base_xy[0]), float(base_xy[1]), min_z + top_z)

    membership = TowerLocalMembership(
        source_binding=source_binding,
        segmentation_config=config,
        config_digest=config.configuration_digest(),
        source_index=member_idx.copy(),
        xyz_absolute_m=member_xyz.copy(),
        member_count=int(len(member_indices)),
        member_bounds_absolute_m=(member_min, member_max),
        base_xyz_absolute_m=base_xyz_absolute,
        top_xyz_absolute_m=top_xyz_absolute,
        height_m=float(top_z),
        raw_tile_cache_identity={
            "tiles_used": list(query.tiles),
            "seed_xyz_absolute_m": list(query.seed_xyz_absolute_m),
            "xy_radius_m": query.xy_radius_m,
            "z_window_min_m": query.z_window_min_m,
            "z_window_max_m": query.z_window_max_m,
            "raw_point_count": query.point_count,
        },
        quality_diagnostics=dict(base_diag),
        algorithm_version=TOWER_LOCAL_SEGMENTATION_VERSION,
    )
    diagnostics = dict(base_diag)
    diagnostics["success"] = True
    diagnostics["membership_digest"] = membership.membership_digest()
    diagnostics["runtime_seconds"] = _time.perf_counter() - started
    return TowerLocalSegmentationResult(
        success=True,
        membership=membership,
        failure_reason=None,
        diagnostics=diagnostics,
    )


__all__ = [
    "TOWER_LOCAL_SEGMENTATION_FAILURE",
    "TOWER_LOCAL_SEGMENTATION_SCHEMA",
    "TOWER_LOCAL_SEGMENTATION_VERSION",
    "TowerLocalMembership",
    "TowerLocalSegmentationConfig",
    "TowerLocalSegmentationError",
    "TowerLocalSegmentationResult",
    "segment_tower_membership",
]
