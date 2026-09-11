"""Structured sampler: one label-agnostic path from a scene to a fixed-size input.

Contract:

* identical code path for GOOD and every defect class — ``sample`` has no class or label
  parameter and never sees ``gt_source_id``;
* 4096 points primary, 2048 points robustness mode;
* four features: local metric X, Y, Z and ``current_candidate_membership``;
* **metric scale is preserved.** The only normalisation is a translation to a local
  origin; there is no per-example RMS or unit-sphere scaling, so a 12 m tree stays 12 m.
  A fixed, research-wide constant may be applied for numerical stabilisation, and
  ``METRIC_SCALE_CONSTANT`` is that constant — it is identical for every example.

Coverage guarantees are explicit and class-independent:

* the candidate/context quota is fixed by the publisher GT split of the scene, which is
  the same for every class, so no class can shift it;
* height bands and spatial cells each receive a quota floor, so stem, crown and the
  candidate/context boundary are represented;
* changed-membership evidence gets a reserved quota, taken **inside** those fixed group
  budgets, so it cannot change the candidate/context ratio;
* the reserved quota is never filled by duplicating points. If fewer changed points
  exist than the reservation, the shortfall is recorded and redistributed; the sampler
  reports ``changed_evidence_available`` and ``changed_evidence_sampled``.

**Seed decoupling.** ``sampling_seed`` derives the sampler RNG from a key that provably
contains no class, label or corruption information, while the corruption RNG keeps using
``ExampleSpec.seed``. Before this rule the sampler was fed the corruption seed, which is
indexed by the synthetic class (``10_007 * class_index``), so although ``sample`` has no
class parameter its RNG still depended on the class and produced a deterministic,
class-correlated XYZ artifact. ``sample`` asserts its seed contract and rejects a seed
that was not produced by :func:`sampling_seed`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

POINTS_PRIMARY = 4096
POINTS_ROBUSTNESS = 2048
FEATURES = ('local_x', 'local_y', 'local_z', 'current_candidate_membership')
FEATURE_COUNT = len(FEATURES)
SAMPLER_VERSION = 'GATE_D_STRUCTURED_SAMPLER_V4'
SAMPLING_SEED_VERSION = 'GATE_D_SAMPLING_SEED_V1'
# Domain separation for the seed preimage: the corruption RNG and the sampling RNG must
# never share a seed even if every other key component is equal.
SAMPLING_SEED_DOMAIN = 'gate_d.sampling_seed'

# Fixed research-wide numerical constant. It is NOT fitted per example: every scene is
# divided by the same number, so metric scale and cross-example comparability survive.
METRIC_SCALE_CONSTANT = 10.0

# Coverage floors. Fixed for every class and severity.
MIN_CHANGED_EVIDENCE = 32
HEIGHT_BANDS = 4
SPATIAL_RESOLUTION = 10
MIN_PER_HEIGHT_BAND = 24
MIN_PER_SPATIAL_CELL = 1
# Candidate/context group budgets are **proportional** to the scene's own publisher GT
# split, which is identical for every class of the same candidate. A fixed fraction was
# tried and rejected: it over-sampled the candidate on scenes where context dominates,
# shifting the sampled membership fraction by up to +0.43 and creating a class-dependent
# fingerprint. Coverage is provided by the per-stratum floor and the changed-evidence
# reservation *inside* these proportional budgets, not by re-weighting the groups.
MIN_CANDIDATE_QUOTA = 32
MIN_CONTEXT_QUOTA = 32


class SamplerError(RuntimeError):
    pass


def sampling_seed(partition: str, candidate_id: str, severity: str,
                  points_target: int = POINTS_PRIMARY, repeat: int = 0,
                  seed_base: int = 0) -> int:
    """Deterministic sampler RNG seed that contains no class or label information.

    The key is hashed from exactly these components:

    * ``partition``       — model-selection split
    * ``candidate_id``    — which tree the scene belongs to
    * ``severity``        — corruption strength
    * ``points_target``   — 4096 primary / 2048 robustness
    * ``repeat``          — repeat index within a cell
    * ``seed_base``       — research-wide base offset
    * ``SAMPLING_SEED_VERSION`` — domain separation from the corruption RNG

    ``example_class``, ``class_index``, the label, the corruption mechanism and the
    changed-membership count are deliberately **absent**. Two specs that differ only in
    class therefore receive the identical sampling seed, so the synthetic class can only
    express itself through ``current_candidate_membership`` and never through which
    points happen to be drawn.
    """
    key = (f'{SAMPLING_SEED_DOMAIN}|{SAMPLING_SEED_VERSION}|{partition}'
           f'|{candidate_id}|{severity}|{int(points_target)}|{int(repeat)}'
           f'|{int(seed_base)}|{SAMPLER_VERSION}')
    digest = hashlib.sha256(key.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], 'little')
    _ISSUED_SAMPLING_SEEDS.add(seed)
    return seed


# Every seed handed out by :func:`sampling_seed`. ``sample`` refuses anything else, so a
# caller cannot silently fall back to the corruption seed (which is class-indexed) or to
# any other arbitrary value.
_ISSUED_SAMPLING_SEEDS: set[int] = set()


def is_sampling_seed(seed: int) -> bool:
    """Whether ``seed`` came from :func:`sampling_seed`."""
    return int(seed) in _ISSUED_SAMPLING_SEEDS


@dataclass(frozen=True)
class Sample:
    """One fixed-size model input plus sampler diagnostics."""

    features: np.ndarray        # (points, 4) float32
    indices: np.ndarray         # (points,) int64 indices into the scene
    source_count: int
    points: int
    version: str = SAMPLER_VERSION
    diagnostics: dict = None


def local_translate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to a local origin and apply the fixed metric constant.

    Translation only: no rotation, no scaling by the cloud's own extent, so metric size
    is preserved exactly.
    """
    points = np.asarray(points, dtype=np.float32)
    origin = points.mean(axis=0)
    local = ((points - origin) / np.float32(METRIC_SCALE_CONSTANT)).astype(np.float32)
    return local, origin


def _height_bands(points: np.ndarray, bands: int = HEIGHT_BANDS) -> np.ndarray:
    """Band index per point over the scene's own Z extent (relative, not rescaled)."""
    z = points[:, 2]
    low, high = float(z.min()), float(z.max())
    if high - low <= 1e-9:
        return np.zeros(points.shape[0], dtype=np.int64)
    index = np.floor((z - low) / (high - low) * bands).astype(np.int64)
    return np.clip(index, 0, bands - 1)


def _spatial_cells(points: np.ndarray,
                   resolution: int = SPATIAL_RESOLUTION) -> np.ndarray:
    low = points[:, :2].min(axis=0)
    span = np.maximum(points[:, :2].max(axis=0) - low, 1e-9)
    cell = np.floor((points[:, :2] - low) / span * resolution).astype(np.int64)
    cell = np.clip(cell, 0, resolution - 1)
    return cell[:, 0] * resolution + cell[:, 1]


def _select_within(pool: np.ndarray, keys: np.ndarray, budget: int,
                   rng: np.random.Generator, floor: int) -> np.ndarray:
    """Choose exactly `budget` indices from `pool`, spread over `keys`.

    Every occupied stratum gets `floor` points when the budget allows, and the remainder
    is allocated proportionally to stratum size by largest remainder. The floor is what
    guarantees coverage of sparse structure; it is applied identically to every class.

    When there are more occupied strata than the budget can give a floor to, the largest
    strata are funded first and the rest receive nothing. That branch resets `allocated`
    while clearing `quotas`, which an earlier revision failed to do: it left `allocated`
    at the pre-clear total, so the funding loop broke immediately, every quota stayed 0
    and the group contributed no structured points at all. The invariant is therefore
    asserted directly: the returned size is exactly `min(budget, pool.size)`.
    """
    if budget <= 0 or pool.size == 0:
        return np.zeros(0, dtype=np.int64)
    budget = min(budget, pool.size)
    if keys.shape[0] != pool.size:
        raise SamplerError(
            f'keys must align with pool, got {keys.shape[0]} and {pool.size}')
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, boundaries)
    sizes = np.array([g.size for g in groups], dtype=np.int64)
    occupied = len(groups)

    quotas = np.minimum(sizes, floor)
    allocated = int(quotas.sum())
    if allocated > budget:
        # Not enough budget for every floor: fund the largest strata first. `quotas` is
        # cleared, so `allocated` must be reset with it, otherwise the loop exits at once
        # and the group contributes nothing.
        priority = np.argsort(-sizes, kind='stable')
        quotas = np.zeros(occupied, dtype=np.int64)
        allocated = 0
        for index in priority:
            if allocated >= budget:
                break
            give = min(floor, int(sizes[index]), budget - allocated)
            if give <= 0:
                continue
            quotas[index] = give
            allocated += give

    remaining = budget - allocated
    if remaining > 0:
        headroom = sizes - quotas
        weight = headroom.astype(np.float64) / max(headroom.sum(), 1)
        extra = np.floor(weight * remaining).astype(np.int64)
        extra = np.minimum(extra, headroom)
        left = remaining - int(extra.sum())
        if left > 0:
            remainder = weight * remaining - extra
            take = np.argsort(-remainder, kind='stable')[:left]
            extra[take] += 1
            extra = np.minimum(extra, headroom)
        quotas += extra

    picked = []
    for group, quota in zip(groups, quotas, strict=True):
        if quota <= 0:
            continue
        if quota >= group.size:
            picked.append(pool[group])
        else:
            chosen = rng.choice(group.size, size=int(quota), replace=False)
            picked.append(pool[group[np.sort(chosen)]])
    if not picked:
        raise SamplerError(
            f'stratified selection produced no points from a pool of {pool.size} '
            f'with budget {budget}')
    selected = np.concatenate(picked)
    if selected.size != budget:
        raise SamplerError(
            f'stratified selection returned {selected.size} points, expected {budget}')
    return selected


def _matched_drop(droppable: np.ndarray, add: np.ndarray, height: np.ndarray,
                  membership: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Choose dropped points that match the added points on height band and membership.

    Changed points are spatially clustered *and* carry a class-characteristic membership
    value by construction (a claimed neighbour point is 1, a dropped crown point is 0), so
    a free choice of which selected point to drop would shift the sampled marginals in a
    class-dependent direction.

    The two criteria are deliberately prioritised: **height band first**, because a
    vertical shift is the one thing that must never appear in the XYZ marginals the
    corruption cannot legitimately move, and that gate is the hard one; **membership
    value second**, because the membership channel is the injected defect and is allowed
    to vary. Measurements: band-first gives an XYZ artifact separation of 0.0970 and
    membership-first gives 0.1463 with no improvement in the membership gap, so band-first
    is the correct priority.
    """
    if add.size == 0:
        return np.zeros(0, dtype=np.int64)
    add_keys = (height[add].astype(np.int64) * 2 + membership[add].astype(np.int64))
    drop_keys = (height[droppable].astype(np.int64) * 2
                 + membership[droppable].astype(np.int64))
    available: dict[int, list[int]] = {}
    for key in np.unique(drop_keys):
        bucket = list(np.flatnonzero(drop_keys == key))
        rng.shuffle(bucket)
        available[int(key)] = bucket

    chosen: list[int] = []
    for key in add_keys:
        bucket = available.get(int(key))
        if bucket:
            chosen.append(bucket.pop())
    if len(chosen) < add.size:
        # Exact key exhausted: fall back to the same height band with any membership value,
        # which still protects the vertical profile.
        taken = set(chosen)
        for key in add_keys[len(chosen):]:
            band = int(key) // 2
            candidates = [index for index in np.flatnonzero(height[droppable] == band)
                          if index not in taken and index not in chosen]
            if candidates:
                pick = int(candidates[rng.integers(0, len(candidates))])
                chosen.append(pick)
                taken.add(pick)
    if len(chosen) < add.size:
        taken = set(chosen)
        leftovers = [index for index in range(droppable.size) if index not in taken]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:add.size - len(chosen)])
    return droppable[np.asarray(chosen[:add.size], dtype=np.int64)]


def sample(points: np.ndarray, membership: np.ndarray, changed: np.ndarray,
           candidate_mask: np.ndarray, sampling_seed_value: int,
           points_target: int = POINTS_PRIMARY) -> Sample:
    """Sample one scene into the fixed model input.

    ``changed`` is the set of points whose membership differs from the GOOD baseline; it
    is used only to reserve evidence inside the fixed candidate/context budgets. There is
    no class or label parameter.

    ``sampling_seed_value`` must come from :func:`sampling_seed`; a corruption seed is
    rejected, because the corruption seed is indexed by the synthetic class and would
    reintroduce the class-correlated draw artifact this contract exists to prevent.
    """
    if not is_sampling_seed(sampling_seed_value):
        raise SamplerError(
            'sampling seed must be produced by sampling_seed(); a corruption seed is '
            'class-indexed and must never drive the sampler RNG')
    points = np.asarray(points, dtype=np.float32)
    membership = np.asarray(membership, dtype=bool)
    changed = np.asarray(changed, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise SamplerError(f'points must be (N, 3), got {points.shape}')
    for name, array in (('membership', membership), ('changed', changed),
                        ('candidate_mask', candidate_mask)):
        if array.shape[0] != points.shape[0]:
            raise SamplerError(f'{name} must align with points')
    if points.shape[0] == 0:
        raise SamplerError('cannot sample an empty scene')
    if points_target <= 0:
        raise SamplerError('points_target must be positive')

    # Golden-ratio mixing, masked to uint64 so a large seed cannot overflow (numpy would
    # otherwise emit a RuntimeWarning and rely on wraparound).
    mixed = (int(sampling_seed_value) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.default_rng(np.uint64(mixed))
    total = points.shape[0]
    height = _height_bands(points)
    cell = _spatial_cells(points)

    # Proportional, class-independent group budgets. The split follows the scene's own
    # publisher GT composition, which is the same for every class of this candidate.
    candidate_pool = np.flatnonzero(candidate_mask)
    context_pool = np.flatnonzero(~candidate_mask)
    candidate_share = candidate_pool.size / float(total)
    candidate_budget = int(round(points_target * candidate_share))
    context_budget = points_target - candidate_budget
    if candidate_pool.size == 0:
        candidate_budget, context_budget = 0, points_target
    elif context_pool.size == 0:
        candidate_budget, context_budget = points_target, 0
    else:
        # Guarantee a minimum for both groups without exceeding the total.
        candidate_budget = int(np.clip(candidate_budget, MIN_CANDIDATE_QUOTA,
                                       points_target - MIN_CONTEXT_QUOTA))
        context_budget = points_target - candidate_budget

    chosen: list[np.ndarray] = []
    diagnostics: dict = {}

    def group_select(pool: np.ndarray, budget: int, label: str) -> np.ndarray:
        if pool.size == 0 or budget <= 0:
            return np.zeros(0, dtype=np.int64)
        strata = cell[pool] * HEIGHT_BANDS + height[pool]
        picked = _select_within(pool, strata, budget, rng, MIN_PER_SPATIAL_CELL)
        diagnostics[f'{label}_pool'] = int(pool.size)
        diagnostics[f'{label}_budget'] = int(budget)
        diagnostics[f'{label}_sampled'] = int(picked.size)
        return picked

    candidate_picked = group_select(candidate_pool, candidate_budget, 'candidate')
    context_picked = group_select(context_pool, context_budget, 'context')

    # Changed-membership evidence is reserved *inside* the fixed group budgets, so it
    # can never move the candidate/context ratio. Points are never duplicated.
    changed_pool = np.flatnonzero(changed)
    selected = np.concatenate([p for p in (candidate_picked, context_picked) if p.size]) \
        if (candidate_picked.size or context_picked.size) else np.zeros(0, dtype=np.int64)
    have = int(changed[selected].sum()) if selected.size else 0
    reservation = max(0, MIN_CHANGED_EVIDENCE - have)
    diagnostics['changed_evidence_available'] = int(changed_pool.size)
    diagnostics['changed_evidence_before_reserve'] = have
    diagnostics['changed_evidence_reservation'] = int(reservation)
    if reservation > 0:
        for pool, picked, budget, label in (
                (candidate_pool, candidate_picked, candidate_budget, 'candidate'),
                (context_pool, context_picked, context_budget, 'context')):
            if budget <= 0 or picked.size == 0:
                continue
            missing = np.setdiff1d(pool[changed[pool]], picked)
            droppable = picked[~changed[picked]]
            take = int(min(reservation, missing.size, droppable.size))
            if take <= 0:
                continue
            # Height-band matched exchange. Changed points are spatially clustered by
            # construction (crown tops, stem bottoms), so a free choice of which selected
            # point to drop would shift the sampled Z distribution in a class-dependent
            # direction and create the very XYZ artifact this module must avoid. Pairing
            # each added point with a dropped point from the same height band keeps the
            # vertical profile intact while still guaranteeing the evidence.
            add = missing[rng.permutation(missing.size)[:take]]
            drop = _matched_drop(droppable, add, height, membership, rng)
            updated = np.union1d(np.setdiff1d(picked, drop), add)
            diagnostics[f'{label}_sampled'] = int(updated.size)
            if label == 'candidate':
                candidate_picked = updated
            else:
                context_picked = updated
            reservation -= take
    chosen = [candidate_picked, context_picked]
    selected = np.concatenate([p for p in chosen if p.size]) \
        if any(p.size for p in chosen) else np.zeros(0, dtype=np.int64)

    selected = np.unique(selected)
    if selected.size > points_target:
        trim = np.linspace(0, selected.size - 1, points_target).astype(np.int64)
        selected = selected[trim]
    if selected.size < points_target:
        missing = np.setdiff1d(np.arange(total, dtype=np.int64), selected)
        if missing.size:
            add = missing[rng.permutation(missing.size)[
                :min(points_target - selected.size, missing.size)]]
            selected = np.union1d(selected, add)
    diagnostics['changed_evidence_sampled'] = int(changed[selected].sum())
    diagnostics['changed_evidence_shortfall'] = int(
        max(0, MIN_CHANGED_EVIDENCE - int(changed[selected].sum())))
    diagnostics['membership_sampled'] = round(float(membership[selected].mean()), 6)
    diagnostics['candidate_sampled'] = int(candidate_mask[selected].sum())
    diagnostics['context_sampled'] = int((~candidate_mask[selected]).sum())
    diagnostics['height_bands_sampled'] = sorted(
        {int(v) for v in height[selected]})
    diagnostics['spatial_cells_sampled'] = int(np.unique(cell[selected]).size)
    diagnostics['spatial_cells_available'] = int(np.unique(cell).size)
    diagnostics['points_target'] = int(points_target)
    diagnostics['source_count'] = int(total)
    diagnostics['candidate_share'] = round(float(candidate_share), 6)
    diagnostics['metric_scale_constant'] = METRIC_SCALE_CONSTANT
    diagnostics['per_example_scaling'] = False

    if selected.size != points_target:
        raise SamplerError(
            f'expected {points_target} points, selected {selected.size}')

    local, origin = local_translate(points)
    features = np.empty((selected.size, FEATURE_COUNT), dtype=np.float32)
    features[:, :3] = local[selected]
    features[:, 3] = membership[selected].astype(np.float32)
    diagnostics['origin'] = [round(float(v), 6) for v in origin]
    return Sample(features=features, indices=selected.astype(np.int64),
                  source_count=int(total), points=int(selected.size),
                  diagnostics=diagnostics)


def membership_fraction(sample_result: Sample) -> float:
    return (float(sample_result.features[:, 3].mean())
            if sample_result.points else 0.0)


def spatial_coverage(sample_result: Sample, points: np.ndarray) -> float:
    """Fraction of scene spatial cells represented in the sample."""
    cells = _spatial_cells(np.asarray(points, dtype=np.float32))
    available = np.unique(cells)
    if available.size == 0:
        return 0.0
    sampled = np.unique(cells[sample_result.indices])
    return float(np.intersect1d(available, sampled).size / available.size)


def height_coverage(sample_result: Sample, points: np.ndarray) -> float:
    bands = _height_bands(np.asarray(points, dtype=np.float32))
    available = np.unique(bands)
    if available.size == 0:
        return 0.0
    sampled = np.unique(bands[sample_result.indices])
    return float(np.intersect1d(available, sampled).size / available.size)
