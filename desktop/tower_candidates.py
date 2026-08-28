"""Bounded presentation-only tower candidate ranking for local point clouds.

Candidates help an operator find a likely tower near a registered GIS seed.
They never create or confirm a Stage-A Tower and are never consumed directly by
scientific analysis. Confirmation must still pass through the real RAW local
refinement workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TowerCandidate:
    candidate_id: str
    xyz: tuple[float, float, float]
    height_m: float
    score: float
    local_point_count: int
    distance_to_seed_m: float
    vertical_continuity: float
    authority: str = "PRESENTATION_ASSISTANCE_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "xyz": list(self.xyz),
            "height_m": self.height_m,
            "score": self.score,
            "local_point_count": self.local_point_count,
            "distance_to_seed_m": self.distance_to_seed_m,
            "vertical_continuity": self.vertical_continuity,
            "authority": self.authority,
        }


def _finite_xyz(xyz: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(xyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("xyz must be an N×3 coordinate array")
    if len(values) == 0:
        return values
    return values[np.all(np.isfinite(values), axis=1)]


def _continuity(z: np.ndarray, base_z: float, top_z: float) -> float:
    if top_z <= base_z:
        return 0.0
    edges = np.linspace(base_z, top_z, 9)
    occupied = np.histogram(z, bins=edges)[0] > 0
    return float(np.count_nonzero(occupied) / len(occupied))


def rank_tower_candidates(
    xyz: Sequence[Sequence[float]],
    seed_xy: Sequence[float],
    *,
    maximum_candidates: int = 3,
    grid_size_m: float = 2.5,
    neighborhood_radius_m: float = 5.0,
    minimum_height_m: float = 6.0,
    merge_distance_m: float = 5.0,
) -> tuple[TowerCandidate, ...]:
    """Rank a few tower-like vertical structures around one registered GIS seed."""

    if maximum_candidates <= 0:
        raise ValueError("maximum_candidates must be positive")
    for value, name in (
        (grid_size_m, "grid_size_m"),
        (neighborhood_radius_m, "neighborhood_radius_m"),
        (minimum_height_m, "minimum_height_m"),
        (merge_distance_m, "merge_distance_m"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if len(seed_xy) < 2:
        raise ValueError("seed_xy must contain x and y")
    seed = np.asarray((float(seed_xy[0]), float(seed_xy[1])), dtype=np.float64)
    if not np.all(np.isfinite(seed)):
        raise ValueError("seed_xy must be finite")

    points = _finite_xyz(xyz)
    if len(points) < 8:
        return ()
    global_base = float(np.quantile(points[:, 2], 0.05))
    high = points[points[:, 2] >= global_base + float(minimum_height_m)]
    if len(high) < 3:
        return ()

    cells = np.floor((high[:, :2] - seed) / float(grid_size_m)).astype(np.int64)
    unique_cells, counts = np.unique(cells, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    raw: list[TowerCandidate] = []
    radius_sq = float(neighborhood_radius_m) ** 2
    for cell_index in order:
        cell = unique_cells[cell_index]
        members = high[np.all(cells == cell, axis=1)]
        center = np.median(members[:, :2], axis=0)
        radial_sq = np.sum((points[:, :2] - center) ** 2, axis=1)
        local = points[radial_sq <= radius_sq]
        if len(local) < 8:
            continue
        base_z, top_z = np.quantile(local[:, 2], (0.05, 0.99))
        height = float(top_z - base_z)
        if height < float(minimum_height_m):
            continue
        continuity = _continuity(local[:, 2], float(base_z), float(top_z))
        distance = float(np.linalg.norm(center - seed))
        median_radius = float(np.median(np.sqrt(radial_sq[radial_sq <= radius_sq])))
        height_score = min(height / 30.0, 1.0)
        density_score = min(math.log1p(len(local)) / 8.0, 1.0)
        compactness = 1.0 / (1.0 + median_radius)
        seed_score = 1.0 / (1.0 + distance / max(float(neighborhood_radius_m), 1.0))
        score = (
            0.40 * height_score
            + 0.25 * continuity
            + 0.15 * density_score
            + 0.10 * compactness
            + 0.10 * seed_score
        )
        raw.append(
            TowerCandidate(
                candidate_id="pending",
                xyz=(float(center[0]), float(center[1]), float(base_z)),
                height_m=height,
                score=float(score),
                local_point_count=int(len(local)),
                distance_to_seed_m=distance,
                vertical_continuity=continuity,
            )
        )

    ranked = sorted(raw, key=lambda item: (-item.score, item.distance_to_seed_m))
    selected: list[TowerCandidate] = []
    for candidate in ranked:
        if any(
            math.hypot(
                candidate.xyz[0] - existing.xyz[0],
                candidate.xyz[1] - existing.xyz[1],
            )
            < float(merge_distance_m)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= maximum_candidates:
            break
    return tuple(
        TowerCandidate(
            candidate_id=f"candidate-{index + 1}",
            xyz=item.xyz,
            height_m=item.height_m,
            score=item.score,
            local_point_count=item.local_point_count,
            distance_to_seed_m=item.distance_to_seed_m,
            vertical_continuity=item.vertical_continuity,
        )
        for index, item in enumerate(selected)
    )


__all__ = ["TowerCandidate", "rank_tower_candidates"]
