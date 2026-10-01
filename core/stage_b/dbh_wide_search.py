"""Bounded all-source stem search inside a current crown footprint.

The source-wide vegetation tile cache retains every LAS row. Occupancy only
proposes search locations; each proposal is subsequently re-queried and must
pass independent circles, crown identity, and global source ownership.
"""

from __future__ import annotations

import numpy as np
from algorithms.manual_span.metrics import _hag_chunked
from algorithms.spans.role_tiles import ROLE_TILE_RECORD_DTYPE

from core.stage_b.dbh_evidence import confined, read_json, sha256, strong_source

CELL_M = 0.25
HEIGHT_BIN_M = 0.30
MAX_TILES = 16
MAX_TILE_ROWS = 6_000_000
MAX_BOUNDS_M = 20.0


def current_crown_wide_seeds(
    cache_root, source, crown_xyz, ground, ground_tree, origin,
    *, excluded_indices=(), existing_seeds=(), limit=4,
):
    """Find vertically persistent raw columns near this surviving crown."""
    cache = cache_root.resolve()
    crown = np.asarray(crown_xyz)
    origin = np.asarray(origin)
    if crown.ndim != 2 or crown.shape[1:] != (3,) or not len(crown):
        return [], {"status": "NO_CURRENT_CROWN"}
    if limit <= 0:
        return [], {"status": "ROI_BUDGET_EXHAUSTED"}
    lower = np.min(crown[:, :2], axis=0) + origin[:2] - 0.6
    upper = np.max(crown[:, :2], axis=0) + origin[:2] + 0.6
    if np.any(upper - lower > MAX_BOUNDS_M):
        return [], {"status": "CROWN_BOUNDS_OVER_BUDGET", "bounds_m": (upper-lower).tolist()}
    if not (cache / "role-tile-cache.json").is_file():
        return [], {"status": "ALL_SOURCE_TILE_CACHE_UNAVAILABLE",
                    "review_hint": "无法读取逐树原始点宽范围索引，请检查点云缓存并手动复核。"}
    metadata = read_json(cache / "role-tile-cache.json")
    if (metadata.get("role") != "vegetation"
            or metadata.get("admission_semantics") != "HIGH_RECALL"
            or metadata.get("totals", {}).get("point_count") != source["source_point_count"]
            or strong_source(metadata.get("source_identity", {})) != strong_source(source)):
        raise ValueError("DBH_WIDE_SEARCH_SOURCE_NOT_ALL_RAW")
    reference = metadata["artifact_references"]["tile_index"]
    index_path = confined(cache, reference["relative_path"])
    if sha256(index_path) != reference["sha256"]:
        raise ValueError("DBH_WIDE_SEARCH_INDEX_HASH_MISMATCH")
    index = read_json(index_path)
    nearby_ground = ground[
        (ground[:, 0] >= lower[0] - 3) & (ground[:, 0] <= upper[0] + 3)
        & (ground[:, 1] >= lower[1] - 3) & (ground[:, 1] <= upper[1] + 3)
    ]
    if len(nearby_ground) < 6:
        return [], {"status": "LOCAL_GROUND_SPARSE"}
    z_low = float(np.min(nearby_ground[:, 2]) - 0.5)
    z_high = float(np.max(nearby_ground[:, 2]) + 3.5)
    tiles = []
    for item in index["tiles"].values():
        tile_low, tile_high = np.asarray(item["min_xyz"]), np.asarray(item["max_xyz"])
        if (np.all(tile_high[:2] >= lower) and np.all(tile_low[:2] <= upper)
                and tile_high[2] >= z_low and tile_low[2] <= z_high):
            tiles.append(item)
    count = sum(int(item["point_count"]) for item in tiles)
    if len(tiles) > MAX_TILES or count > MAX_TILE_ROWS:
        return [], {"status": "SEARCH_BUDGET_EXHAUSTED", "tile_count": len(tiles),
                    "tile_rows": count, "tile_limit": MAX_TILES,
                    "row_limit": MAX_TILE_ROWS,
                    "review_hint": "胸径自动宽范围搜索超出有界读取预算，请在原始点云中逐树选干复核。"}
    excluded = np.sort(np.asarray(excluded_indices, dtype=np.int64))
    blocks = []
    for item in tiles:
        path = confined(cache / "tiles", item["file"])
        if path.stat().st_size != int(item["point_count"]) * ROLE_TILE_RECORD_DTYPE.itemsize:
            raise ValueError("DBH_WIDE_SEARCH_TILE_SIZE_MISMATCH")
        if sha256(path) != item["sha256"]:
            raise ValueError("DBH_WIDE_SEARCH_TILE_HASH_MISMATCH")
        data = np.fromfile(path, dtype=ROLE_TILE_RECORD_DTYPE)
        xyz = np.column_stack((data["x"], data["y"], data["z"]))
        keep = (np.all((xyz[:, :2] >= lower) & (xyz[:, :2] <= upper), axis=1)
                & (xyz[:, 2] >= z_low) & (xyz[:, 2] <= z_high))
        if len(excluded):
            keep &= ~np.isin(data["source_index"], excluded)
        if keep.any():
            blocks.append(xyz[keep])
    if not blocks:
        return [], {"status": "NO_RAW_IN_CROWN_BOUNDS"}
    xyz = np.concatenate(blocks)
    hag = _hag_chunked(xyz - origin, ground_tree, ground, origin, 250_000)
    mask = (hag >= 0.85) & (hag < 3.25)
    xyz, hag = xyz[mask], hag[mask]
    if not len(xyz):
        return [], {"status": "NO_LOW_RAW_IN_CROWN_BOUNDS"}
    # Grid in the survey's local frame so a global coordinate translation
    # cannot silently change which quarter-metre column proposes a stem.
    seeds, occupancy = _propose_columns(xyz, hag, origin, existing_seeds, limit)
    return seeds, {"status": occupancy["status"], "tile_count": len(tiles), "tile_rows": count,
                   "low_raw_points": len(xyz), "proposed_count": len(seeds),
                   "maximum_occupied_height_bins": occupancy["maximum_occupied_height_bins"],
                   "minimum_points_per_height_bin": 30,
                   "minimum_occupied_height_bins": 6,
                   "bounds_xy_absolute_m": [lower.tolist(), upper.tolist()]}


def _propose_columns(xyz, hag, origin, existing_seeds, limit):
    """Use low raw occupancy only to locate reviewable proposals, never DBH."""
    xyz = np.asarray(xyz)
    hag = np.asarray(hag)
    origin = np.asarray(origin)
    cell = np.floor((xyz[:, :2] - origin[:2]) / CELL_M).astype(np.int64)
    height_bin = np.floor((hag - 0.85) / HEIGHT_BIN_M).astype(np.int64)
    keys, counts = np.unique(np.column_stack((cell, height_bin)), axis=0, return_counts=True)
    xy_keys, inverse = np.unique(keys[:, :2], axis=0, return_inverse=True)
    occupied = np.bincount(inverse, weights=(counts >= 30).astype(int), minlength=len(xy_keys))
    total = np.bincount(inverse, weights=counts, minlength=len(xy_keys))
    order = np.lexsort((xy_keys[:, 1], xy_keys[:, 0], -total, -occupied))
    seeds = []
    prior = [np.asarray(seed["xy"], dtype=float) for seed in existing_seeds]
    for position in order:
        if occupied[position] < 6:
            break
        center = (xy_keys[position] + 0.5) * CELL_M
        if any(np.linalg.norm(center - old) <= 0.60 for old in [*prior, *(np.asarray(s["xy"]) for s in seeds)]):
            continue
        seeds.append({"xy": center.tolist(), "source": "ALL_SOURCE_VERTICAL_OCCUPANCY",
                      "support": int(total[position]), "occupied_height_bins": int(occupied[position])})
        if len(seeds) == limit:
            break
    status = "PASS" if seeds else (
        "LOW_DENSITY_VERTICAL_OCCUPANCY" if int(np.max(occupied, initial=0)) < 6
        else "NO_DISTINCT_VERTICAL_COLUMN"
    )
    return seeds, {"status": status,
                   "maximum_occupied_height_bins": int(np.max(occupied, initial=0))}


__all__ = ["current_crown_wide_seeds"]
