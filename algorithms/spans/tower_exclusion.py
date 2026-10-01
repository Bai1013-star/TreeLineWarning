"""Derive exact tower point identities from registered positions and local RAW.

The registered position is a search prior, not a deletion cylinder.  A tower
shaft must have ground contact, continuous vertical support, a compact core,
and shaft-like local geometry.  Only points connected upward to that proven
core are excluded from tree input.  The source LAS and role caches are untouched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.towers.segmentation import (
    TowerLocalSegmentationConfig,
    _local_pca_shaft_like,
    _upward_connected_members,
)
from algorithms.vegetation.coarse_to_fine import RawRegion

TOWER_TREE_EXCLUSION_VERSION = "registered-tower-connected-members-v2"
_LOCAL_CHUNK = 1_000_000


@dataclass(frozen=True, slots=True)
class TowerTreeExclusion:
    source_index: np.ndarray
    towers: tuple[dict[str, Any], ...]
    source_index_by_tower: tuple[np.ndarray, ...]


def publish_tower_membership(
    result: TowerTreeExclusion,
    output_root: Path,
    *,
    source_identity: Mapping[str, Any],
    region_origin_xyz: Sequence[float],
) -> dict[str, Any]:
    """Atomically save per-tower and union source identities with an audit."""
    output_root.mkdir(parents=True, exist_ok=True)
    archive = output_root / "tower_membership.npz"
    temporary = output_root / "tower_membership.tmp.npz"
    np.savez_compressed(
        temporary,
        source_index=result.source_index,
        **{
            f"tower_{index}_source_index": indices
            for index, indices in enumerate(result.source_index_by_tower)
        },
        schema=np.asarray([TOWER_TREE_EXCLUSION_VERSION]),
    )
    temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "schema": TOWER_TREE_EXCLUSION_VERSION,
        "source_identity": dict(source_identity),
        "coordinate_frame": "SOURCE_LAS_ABSOLUTE_METRE",
        "region_origin_xyz": [float(value) for value in region_origin_xyz],
        "source_index_artifact": archive.name,
        "source_index_sha256": digest,
        "source_index_count": int(len(result.source_index)),
        "towers": [
            {**tower, "source_index_array": f"tower_{index}_source_index"}
            for index, tower in enumerate(result.towers)
        ],
    }
    manifest_tmp = output_root / "tower_membership.tmp.json"
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(output_root / "tower_membership.json")
    return manifest


def load_tower_membership(
    output_root: Path,
    *,
    source_identity: Mapping[str, Any],
    registered_towers: Sequence[Mapping[str, Any]],
    region_origin_xyz: Sequence[float],
) -> TowerTreeExclusion | None:
    """Read a previously audited exact exclusion, rejecting stale identities."""
    manifest_path = output_root / "tower_membership.json"
    archive_path = output_root / "tower_membership.npz"
    if not manifest_path.is_file() or not archive_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != TOWER_TREE_EXCLUSION_VERSION
            or manifest.get("source_identity") != dict(source_identity)
            or manifest.get("coordinate_frame") != "SOURCE_LAS_ABSOLUTE_METRE"
            or manifest.get("region_origin_xyz")
            != [float(value) for value in region_origin_xyz]
            or manifest.get("source_index_artifact") != archive_path.name
            or hashlib.sha256(archive_path.read_bytes()).hexdigest()
            != manifest.get("source_index_sha256")
        ):
            return None
        with np.load(archive_path, allow_pickle=False) as archive:
            if list(archive["schema"]) != [TOWER_TREE_EXCLUSION_VERSION]:
                return None
            indices = np.asarray(archive["source_index"], dtype=np.int64)
            towers = tuple(manifest["towers"])
            parts = tuple(
                np.asarray(archive[tower["source_index_array"]], dtype=np.int64) for tower in towers
            )
        if len(towers) != len(registered_towers):
            return None
        for stored, registered in zip(towers, registered_towers, strict=True):
            tower_id, base, top = _registered_geometry(registered)
            if (
                stored.get("tower_id") != tower_id
                or stored.get("registered_base_xyz_absolute_m") != base.tolist()
                or stored.get("registered_top_xyz_absolute_m") != top.tolist()
            ):
                return None
        union = (
            np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        )
        if (
            len(indices) != manifest.get("source_index_count")
            or not np.array_equal(indices, union)
            or np.any(indices < 0)
            or any(
                len(part) != tower.get("excluded_point_count")
                for tower, part in zip(towers, parts, strict=True)
            )
        ):
            return None
        return TowerTreeExclusion(indices, towers, parts)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _registered_geometry(value: Mapping[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    tower_id = str(value["tower_id"])
    base = np.asarray(value["base_xyz_absolute_m"], dtype=np.float64)
    top = np.asarray(value["top_xyz_absolute_m"], dtype=np.float64)
    if not tower_id or base.shape != (3,) or top.shape != (3,):
        raise ValueError("registered tower geometry is invalid")
    if not np.isfinite(base).all() or not np.isfinite(top).all() or top[2] <= base[2]:
        raise ValueError("registered tower geometry is invalid")
    return tower_id, base, top


def _nearby_raw(
    region: RawRegion, base: np.ndarray, top: np.ndarray, radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Read a bounded neighbourhood, retaining original source identities."""
    xyz = region.xyz()
    source = region.original_index()
    origin = np.asarray(region.origin_xyz, dtype=np.float64)
    local_seed = base - origin
    points: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    for start in range(0, region.count, _LOCAL_CHUNK):
        stop = min(region.count, start + _LOCAL_CHUNK)
        block = np.asarray(xyz[start:stop], dtype=np.float64)
        near = (
            (np.sum((block[:, :2] - local_seed[:2]) ** 2, axis=1) <= radius_m**2)
            & (block[:, 2] >= base[2] - origin[2] - 0.5)
            & (block[:, 2] <= top[2] - origin[2] + 1.0)
        )
        if np.any(near):
            points.append(block[near] + origin)
            identities.append(np.asarray(source[start:stop], dtype=np.int64)[near])
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
    return np.concatenate(points), np.concatenate(identities)


def _shaft_seed(
    points: np.ndarray, base: np.ndarray, top: np.ndarray, config: TowerLocalSegmentationConfig
) -> tuple[np.ndarray | None, str]:
    """Find one shaft; reject a second plausible vertical stem nearby."""
    cell_m = config.shaft_cell_m
    # Local coordinates avoid integer overflow on large projected coordinates
    # and make registration translation-invariant.
    cells = np.floor((points[:, :2] - base[:2]) / cell_m).astype(np.int64)
    bands = np.floor((points[:, 2] - base[2]) / config.band_height_m).astype(np.int64)
    keys = np.column_stack((cells, bands))
    occupied, counts = np.unique(keys, axis=0, return_counts=True)
    candidate_cells, inverse = np.unique(occupied[:, :2], axis=0, return_inverse=True)
    candidates: list[tuple[float, np.ndarray]] = []
    minimum_bands = max(5, min(8, int((top[2] - base[2]) * 0.6)))
    for index, cell in enumerate(candidate_cells):
        xy = base[:2] + (cell.astype(np.float64) + 0.5) * cell_m
        distance = float(np.linalg.norm(xy - base[:2]))
        if distance > 1.5:
            continue
        selected = inverse == index
        cell_bands = occupied[selected, 2]
        band_counts = counts[selected]
        supported = cell_bands[band_counts >= config.min_support_points]
        if len(supported) < minimum_bands or np.min(supported) > 1:
            continue
        if np.max(supported) < minimum_bands - 1:
            continue
        # The shaft must be present through the lower and middle structure.
        # Density distinguishes the actual pole from sparse cells crossed by
        # a cable or canopy near the operator's approximate click.
        candidates.append((float(np.sum(band_counts)), xy))
    if not candidates:
        return None, "NO_CONTINUOUS_SHAFT_NEAR_REGISTERED_POSITION"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, best_xy = candidates[0]
    # Distinct strong columns cannot be assigned confidently by density alone:
    # a nearby tree trunk can contain more returns than the registered pole.
    # Cells within 0.75 m are treated as one pole cross-section; a second
    # viable ground-to-midheight column beyond that makes attribution unsafe,
    # even when it is much sparser than a neighbouring tree trunk.
    if any(
        np.linalg.norm(xy - best_xy) > 0.75 for _support, xy in candidates[1:]
    ):
        return None, "AMBIGUOUS_NEARBY_VERTICAL_SHAFTS"
    return best_xy, ""


def derive_registered_tower_exclusion(
    region: RawRegion, towers: Sequence[Mapping[str, Any]]
) -> TowerTreeExclusion:
    """Return only geometrically supported tower members in ``region``.

    A failed tower remains in the audit with zero members.  It does not trigger
    a broader deletion radius.  The result contains exact original LAS row IDs.
    """
    config = TowerLocalSegmentationConfig()
    parts: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    by_tower: list[np.ndarray] = []
    for raw in towers:
        tower_id, base, top = _registered_geometry(raw)
        row: dict[str, Any] = {
            "tower_id": tower_id,
            "registered_base_xyz_absolute_m": base.tolist(),
            "registered_top_xyz_absolute_m": top.tolist(),
            "method": TOWER_TREE_EXCLUSION_VERSION,
            "coverage": "UNAVAILABLE",
            "excluded_point_count": 0,
        }
        points, identities = _nearby_raw(region, base, top, config.search_radius_m)
        row["local_raw_point_count"] = int(len(points))
        if len(points) < config.min_support_points:
            row["reason"] = "NO_LOCAL_RAW_SUPPORT"
            audit.append(row)
            by_tower.append(np.empty(0, dtype=np.int64))
            continue
        seed, seed_reason = _shaft_seed(points, base, top, config)
        if seed is None:
            row["reason"] = seed_reason
            audit.append(row)
            by_tower.append(np.empty(0, dtype=np.int64))
            continue
        core_mask = np.sum((points[:, :2] - seed) ** 2, axis=1) <= config.core_radius_m**2
        core = points[core_mask]
        if len(core) < config.min_support_points:
            row["reason"] = "SHAFT_CORE_SPARSE"
            audit.append(row)
            by_tower.append(np.empty(0, dtype=np.int64))
            continue
        min_z = float(np.min(core[:, 2]))
        height = float(np.max(core[:, 2]) - min_z)
        bands = np.floor((core[:, 2] - min_z) / config.band_height_m).astype(np.int32)
        occupied = np.unique(bands)
        gap = int(occupied[-1] - occupied[0] + 1 - len(occupied))
        if (
            height < config.min_pole_height_m
            or gap * config.band_height_m > config.max_vertical_gap_m
        ):
            row["reason"] = "SHAFT_HEIGHT_OR_CONTINUITY_FAILED"
            audit.append(row)
            by_tower.append(np.empty(0, dtype=np.int64))
            continue
        band_centers = np.asarray([np.mean(core[bands == b, :2], axis=0) for b in occupied])
        drift = (
            float(np.max(np.linalg.norm(np.diff(band_centers, axis=0), axis=1)))
            if len(band_centers) > 1
            else 0.0
        )
        if drift > config.max_band_drift_m:
            row["reason"] = "SHAFT_DRIFT_FAILED"
            audit.append(row)
            by_tower.append(np.empty(0, dtype=np.int64))
            continue
        pca = _local_pca_shaft_like(core, config)
        if (
            pca["median_verticality"] < config.verticality_min
            and pca["median_linearity"] < config.linearity_min
        ):
            row["reason"] = "SHAFT_GEOMETRY_FAILED"
            audit.append(row)
            by_tower.append(np.empty(0, dtype=np.int64))
            continue
        base_xy = np.median(core[:, :2], axis=0)
        selected = _upward_connected_members(
            points,
            identities,
            np.flatnonzero(core_mask),
            base_xy,
            voxel_m=config.voxel_m,
            base_z=min_z,
            seed_floor_z=min_z + config.seed_elevation_m,
            max_member_radius_m=config.max_member_radius_m,
            max_member_height_m=min(config.max_member_height_m, float(top[2] - base[2]) + 1.0),
            voxel_origin=np.min(points, axis=0),
        )
        if len(selected) < config.min_support_points:
            row["reason"] = "CONNECTED_MEMBERSHIP_SPARSE"
            audit.append(row)
            by_tower.append(np.empty(0, dtype=np.int64))
            continue
        member_source = np.unique(identities[selected])
        parts.append(member_source)
        by_tower.append(member_source)
        row.update(
            {
                "coverage": "PARTIAL_CONNECTED_STRUCTURE",
                "excluded_point_count": int(len(member_source)),
                "shaft_seed_xy_absolute_m": seed.tolist(),
                "shaft_base_xy_absolute_m": base_xy.tolist(),
                "shaft_height_m": height,
                "shaft_seed_offset_m": float(np.linalg.norm(seed - base[:2])),
                "maximum_member_radius_m": config.max_member_radius_m,
                "median_verticality": pca["median_verticality"],
                "median_linearity": pca["median_linearity"],
            }
        )
        audit.append(row)
    source_index = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
    return TowerTreeExclusion(source_index, tuple(audit), tuple(by_tower))


__all__ = [
    "TOWER_TREE_EXCLUSION_VERSION",
    "TowerTreeExclusion",
    "derive_registered_tower_exclusion",
]
