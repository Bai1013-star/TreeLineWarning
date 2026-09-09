"""Structured sampler: one label-agnostic path from any example to fixed-size input.

Contract:

* identical code path for GOOD and every defect class — there is no ``if label``;
* 4096 points primary, 2048 points robustness mode;
* four features per point: locally normalised X, Y, Z and the candidate membership
  flag. Nothing else is ever exposed, so no provenance or class metadata can leak;
* the selection is a spatially stratified draw, which keeps stem, crown, the
  candidate/context boundary and the corruption evidence in the sample instead of
  letting a dense crown swamp a sparse stem.

Local normalisation is a similarity transform: the cloud is centred on the candidate
centroid and scaled by one isotropic factor, so relative geometry and the membership
boundary are preserved exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POINTS_PRIMARY = 4096
POINTS_ROBUSTNESS = 2048
FEATURES = ('local_x', 'local_y', 'local_z', 'candidate_membership')
FEATURE_COUNT = len(FEATURES)
SAMPLER_VERSION = 'GATE_D_STRUCTURED_SAMPLER_V1'

# Grid resolution: cell count scales with sqrt(N) so the mean occupancy stays roughly
# constant across clouds of very different size. Fixed formula, no class dependence.
GRID_BASE = 8
GRID_SCALE = 0.75


class SamplerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sample:
    """One fixed-size model input."""

    features: np.ndarray      # (points, 4) float32
    indices: np.ndarray       # (points,) int64 indices into the source cloud
    source_count: int
    points: int
    version: str = SAMPLER_VERSION


def local_normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre on the centroid and scale by one isotropic factor."""
    points = np.asarray(points, dtype=np.float32)
    centre = points.mean(axis=0)
    centred = points - centre
    scale = float(np.sqrt((centred.astype(np.float64) ** 2).sum(axis=1).mean()))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (centred / scale).astype(np.float32), centre, scale


def _grid_cells(local: np.ndarray, resolution: int) -> np.ndarray:
    """Deterministic integer cell index per point over the local XY footprint.

    The grid is 2D on purpose. Tree TLS clouds are surface samples, so a 3D lattice
    leaves most cells empty and a per-cell floor would then waste quota on the sparse
    shell; the XY footprint keeps cells populated and lets proportional allocation
    track the source composition. Z is not discarded — it is a sampled coordinate.
    """
    low = local[:, :2].min(axis=0)
    high = local[:, :2].max(axis=0)
    span = np.maximum(high - low, 1e-9)
    normalised = (local[:, :2] - low) / span
    cell = np.minimum((normalised * resolution).astype(np.int64), resolution - 1)
    return cell[:, 0] * resolution + cell[:, 1]


def _resolution_for(count: int, points: int) -> int:
    return max(4, int(round(GRID_BASE + GRID_SCALE * np.sqrt(count / max(points, 1)))))


def stratified_indices(points: np.ndarray, count: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Spatially stratified selection of `count` indices.

    Quotas are allocated **proportionally** to each occupied cell's size by the
    largest-remainder method. Proportional allocation is what keeps the sampler
    unbiased: a hard per-cell cap would systematically under-sample a dense cluster
    (for example a compact candidate crown inside a spread-out neighbouring tree), and
    a per-cell floor would over-sample whichever class occupies more cells. Both would
    shift the membership fraction by class. Within a cell the draw is uniform, so the
    expected membership fraction of the sample equals the source's, and every occupied
    cell still receives a fractional share that survives rounding, which keeps sparse
    stem and boundary regions represented.

    The result is sorted so the sample itself carries no ordering signal.
    """
    total = points.shape[0]
    if count >= total:
        return np.arange(total, dtype=np.int64)
    if count <= 0:
        raise SamplerError('sample size must be positive')

    local, _, _ = local_normalise(points)
    cells = _grid_cells(local, _resolution_for(total, count))
    order = np.argsort(cells, kind='stable')
    sorted_cells = cells[order]
    boundaries = np.flatnonzero(np.diff(sorted_cells)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [total]])
    sizes = (stops - starts).astype(np.int64)
    occupied = sizes.size

    # Proportional quota by largest remainder. No per-cell floor and no per-cell cap:
    # both distort the sampled class composition.
    quotas = np.zeros(occupied, dtype=np.int64)
    exact = sizes.astype(np.float64) * count / float(sizes.sum())
    quotas = np.floor(exact).astype(np.int64)
    left = count - int(quotas.sum())
    if left > 0:
        remainder = exact - quotas
        take = np.argsort(-remainder, kind='stable')[:left]
        quotas[take] += 1
    quotas = np.minimum(quotas, sizes)

    picked: list[np.ndarray] = []
    for index in range(occupied):
        start, stop = int(starts[index]), int(stops[index])
        available = order[start:stop]
        quota = int(quotas[index])
        if quota <= 0:
            continue
        if quota >= available.size:
            picked.append(available)
        else:
            chosen = rng.choice(available.size, size=quota, replace=False)
            picked.append(available[np.sort(chosen)])
    if not picked:
        raise SamplerError('stratified selection produced no points')
    selected = np.concatenate(picked)
    if selected.size < count:
        mask = np.ones(total, dtype=bool)
        mask[selected] = False
        residual = np.flatnonzero(mask)
        extra = rng.choice(residual.size, size=count - selected.size, replace=False)
        selected = np.concatenate([selected, residual[np.sort(extra)]])
    selected = np.unique(selected)
    if selected.size > count:
        # Trim deterministically while keeping spatial spread.
        trim = np.linspace(0, selected.size - 1, count).astype(np.int64)
        selected = selected[trim]
    if selected.size != count:
        raise SamplerError(f'expected {count} points, selected {selected.size}')
    return selected.astype(np.int64)


def sample(points: np.ndarray, membership: np.ndarray, spec_seed: int,
           points_target: int = POINTS_PRIMARY) -> Sample:
    """Sample one cloud into the fixed model input. No label argument exists."""
    points = np.asarray(points, dtype=np.float32)
    membership = np.asarray(membership, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise SamplerError(f'points must be (N, 3), got {points.shape}')
    if membership.shape[0] != points.shape[0]:
        raise SamplerError('membership must align with points')
    if points.shape[0] == 0:
        raise SamplerError('cannot sample an empty cloud')
    if points_target <= 0:
        raise SamplerError('points_target must be positive')

    rng = np.random.default_rng(np.uint64(spec_seed) + np.uint64(0x9E3779B97F4A7C15))
    # A cloud smaller than the target is upsampled by a deterministic repeat of the
    # stratified draw, never by inventing coordinates.
    if points.shape[0] < points_target:
        chosen = stratified_indices(points, points.shape[0], rng)
        repeats = int(np.ceil(points_target / chosen.size))
        chosen = np.concatenate([chosen] * repeats)[:points_target]
    else:
        chosen = stratified_indices(points, points_target, rng)

    local, _, _ = local_normalise(points)
    features = np.empty((chosen.size, FEATURE_COUNT), dtype=np.float32)
    features[:, :3] = local[chosen]
    features[:, 3] = membership[chosen].astype(np.float32)
    return Sample(features=features, indices=chosen, source_count=int(points.shape[0]),
                  points=int(chosen.size))


def membership_fraction(sample_result: Sample) -> float:
    return float(sample_result.features[:, 3].mean()) if sample_result.points else 0.0


def spatial_coverage(sample_result: Sample, points: np.ndarray) -> float:
    """Fraction of source grid cells represented in the sample (sampler quality probe)."""
    source_local, _, _ = local_normalise(points)
    resolution = _resolution_for(points.shape[0], sample_result.points)
    source_cells = np.unique(_grid_cells(source_local, resolution))
    sample_cells = np.unique(_grid_cells(source_local[sample_result.indices], resolution))
    if source_cells.size == 0:
        return 0.0
    return float(np.intersect1d(source_cells, sample_cells).size / source_cells.size)
