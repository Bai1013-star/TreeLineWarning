"""Gate D synthetic tree-QC generator.

Every example is the **same** kind of object: a ``BaseScene`` of real publisher geometry
whose ``current_candidate_membership`` channel is corrupted in a class-specific way. The
point cloud stays as close as possible to identical across classes; what changes is
which points the *current* single-tree segmentation claims.

That matches the real TreeLineWarning question: given this pile of real points, did the
current single-tree membership choose correctly?

Classes:

* ``GOOD``                   — publisher GT membership, unchanged;
* ``MERGED_MULTI_TREE``      — a whole neighbouring tree inside the query is claimed;
* ``NEIGHBOR_CONTAMINATION`` — a contiguous neighbouring region near the boundary is
  claimed;
* ``OVER_SEGMENTED``         — a spatially contiguous part of the candidate is dropped;
* ``CROWN_INCOMPLETE``       — the contiguous crown region is dropped;
* ``STEM_MISSING``           — the contiguous stem/lower region is dropped;
* ``NOISE``                  — scattered, non-structural context points are claimed.

Two properties are load-bearing and are asserted by test:

1. **No ground-truth leakage.** ``current_candidate_membership`` means "the current
   segmentation selected this point", never "this is the answer". Merged and contaminating
   neighbour points are therefore 1, and dropped candidate points are 0.
2. **No geometry invention.** Every coordinate comes from the scene; no point is deleted,
   no class-specific refill exists, and no external source outside the candidate's frozen
   surviving context is ever used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from gate_d_provenance import MODEL_SELECTION_PARTITIONS
from gate_d_scene import BaseScene, SceneError, build_base_scene, scene_source_union

GOOD = 'GOOD'
MERGED_MULTI_TREE = 'MERGED_MULTI_TREE'
NEIGHBOR_CONTAMINATION = 'NEIGHBOR_CONTAMINATION'
OVER_SEGMENTED = 'OVER_SEGMENTED'
CROWN_INCOMPLETE = 'CROWN_INCOMPLETE'
STEM_MISSING = 'STEM_MISSING'
NOISE = 'NOISE'

CLASSES = (GOOD, MERGED_MULTI_TREE, NEIGHBOR_CONTAMINATION, OVER_SEGMENTED,
           CROWN_INCOMPLETE, STEM_MISSING, NOISE)
DEFECT_CLASSES = CLASSES[1:]
BAD_CLASSES = DEFECT_CLASSES
CLAIM_CLASSES = (MERGED_MULTI_TREE, NEIGHBOR_CONTAMINATION, NOISE)
DROP_CLASSES = (OVER_SEGMENTED, CROWN_INCOMPLETE, STEM_MISSING)

LOW = 'LOW'
MEDIUM = 'MEDIUM'
HIGH = 'HIGH'
SEVERITIES = (LOW, MEDIUM, HIGH)
SEVERITY_LEVEL = {LOW: 0, MEDIUM: 1, HIGH: 2}

LABEL_PASS = 'PASS'
LABEL_REVIEW = 'REVIEW_REQUIRED'
LABELS = (LABEL_PASS, LABEL_REVIEW)

GENERATOR_VERSION = 'GATE_D_SYNTHETIC_GENERATOR_V2'
MEMBERSHIP_CHANNEL = 'current_candidate_membership'

FRACTION_RANGE = (0.06, 0.34)
SEVERITY_MULTIPLIER = {LOW: 0.55, MEDIUM: 1.0, HIGH: 1.65}
FRACTION_CEILING = 0.72

# STEM_MISSING applicability: the candidate must show observable, contiguous lower
# evidence before a missing-stem defect is meaningful. Thresholds are fixed and applied
# to every candidate identically.
STEM_MIN_LOWER_POINTS = 24
STEM_MIN_VERTICAL_SPAN_M = 1.5
STEM_MAX_RADIUS_M = 0.75
# The "lower" region is the bottom 45 % of the candidate's height range. It is a
# definition of where stem evidence lives, not a tunable per-candidate threshold.
STEM_LOWER_HEIGHT_FRACTION = 0.45

STEM_APPLICABLE = 'APPLICABLE'
STEM_NOT_APPLICABLE = 'STEM_MISSING_NOT_APPLICABLE'


class GeneratorError(RuntimeError):
    pass


class NotApplicableError(GeneratorError):
    """Raised when a defect is not supported by the candidate's real geometry."""

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass(frozen=True)
class ExampleSpec:
    """Frozen, serialisable recipe. Reproduces one example byte-for-byte."""

    candidate_id: str
    partition: str
    example_class: str
    severity: str
    seed: int

    def __post_init__(self) -> None:
        if self.example_class not in CLASSES:
            raise GeneratorError(f'unknown class {self.example_class!r}')
        if self.severity not in SEVERITIES:
            raise GeneratorError(f'unknown severity {self.severity!r}')
        if self.partition not in MODEL_SELECTION_PARTITIONS:
            raise GeneratorError(f'partition {self.partition!r} is not model-selection')

    @property
    def key(self) -> str:
        return (f'{self.partition}|{self.candidate_id}|{self.example_class}'
                f'|{self.severity}|{self.seed}')

    def as_row(self) -> dict:
        return {'candidate_id': self.candidate_id, 'partition': self.partition,
                'class': self.example_class, 'severity': self.severity,
                'seed': self.seed, 'key': self.key}

    @classmethod
    def from_row(cls, row: dict) -> ExampleSpec:
        return cls(candidate_id=row['candidate_id'], partition=row['partition'],
                   example_class=row['class'], severity=row['severity'],
                   seed=int(row['seed']))


@dataclass
class SyntheticExample:
    """One generated example plus its complete audit provenance."""

    spec: ExampleSpec
    points: np.ndarray                 # (N, 3) float32 metres, real geometry
    membership: np.ndarray             # (N,) bool current_candidate_membership
    label: str
    changed_membership: np.ndarray     # (N,) bool points whose flag differs from GOOD
    source_ids: tuple[str, ...]
    provenance: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def donors(self) -> tuple[str, ...]:
        return tuple(self.provenance.get('donors', ()))


def _rng(spec: ExampleSpec) -> np.random.Generator:
    material = f'{GENERATOR_VERSION}|{spec.key}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8],
                                                'little'))


def _draw_fraction(rng: np.random.Generator, severity: str) -> float:
    base = float(rng.uniform(*FRACTION_RANGE))
    return float(min(FRACTION_CEILING, base * SEVERITY_MULTIPLIER[severity]))


def _unit_direction(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return vector / norm


def _take(count: int, want: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic uniform index draw of `want` from `count` (sorted)."""
    want = max(1, min(int(want), count))
    if want >= count:
        return np.arange(count)
    return np.sort(rng.permutation(count)[:want])


def _candidate_indices(scene: BaseScene) -> np.ndarray:
    return np.flatnonzero(scene.candidate_mask)


def _context_indices(scene: BaseScene) -> np.ndarray:
    return np.flatnonzero(~scene.candidate_mask)


def _context_sources(scene: BaseScene) -> list[int]:
    """Scene source indices that are context (not the candidate)."""
    return [index for index, source_id in enumerate(scene.source_ids)
            if source_id != scene.candidate_id]


def _stem_axis_xy(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] == 0:
        return 0.0, 0.0
    low = points[:, 2] <= np.quantile(points[:, 2], 0.10)
    subset = points[low] if low.any() else points
    return float(np.median(subset[:, 0])), float(np.median(subset[:, 1]))


def stem_evidence(scene: BaseScene) -> dict:
    """Observable lower/stem-like contiguous evidence in the candidate's own points.

    A missing-stem defect is only meaningful when the candidate actually shows a
    stem-like column: enough points in the lower part of its height range, within a
    narrow horizontal radius of the trunk axis, spanning a non-trivial height. The check
    is applied identically to every candidate.
    """
    candidate = scene.points[_candidate_indices(scene)]
    if candidate.shape[0] == 0:
        return {'applicable': False, 'reason': 'no_candidate_points'}
    z_low, z_high = float(candidate[:, 2].min()), float(candidate[:, 2].max())
    span = z_high - z_low
    if span <= 1e-6:
        return {'applicable': False, 'reason': 'degenerate_height_range',
                'z_span': round(span, 6)}
    axis_x, axis_y = _stem_axis_xy(candidate)
    horizontal = np.hypot(candidate[:, 0] - axis_x, candidate[:, 1] - axis_y)
    radius = float(np.quantile(horizontal, 0.35))
    z_cut = z_low + span * STEM_LOWER_HEIGHT_FRACTION
    lower = ((candidate[:, 2] <= z_cut)
             & (horizontal <= min(radius, STEM_MAX_RADIUS_M)))
    count = int(lower.sum())
    lower_span = (float(candidate[lower, 2].max() - candidate[lower, 2].min())
                  if count else 0.0)
    detail = {
        'stem_axis_xy': [round(axis_x, 6), round(axis_y, 6)],
        'radius_m': round(min(radius, STEM_MAX_RADIUS_M), 6),
        'z_cut': round(z_cut, 6),
        'z_span': [round(z_low, 6), round(z_high, 6)],
        'lower_points': count,
        'lower_span_m': round(lower_span, 6),
        'min_lower_points': STEM_MIN_LOWER_POINTS,
        'min_vertical_span_m': STEM_MIN_VERTICAL_SPAN_M,
    }
    if count < STEM_MIN_LOWER_POINTS:
        detail.update(applicable=False, reason='insufficient_lower_points')
        return detail
    if lower_span < STEM_MIN_VERTICAL_SPAN_M:
        detail.update(applicable=False, reason='insufficient_vertical_span')
        return detail
    detail.update(applicable=True, reason=None, lower_mask=lower)
    return detail


def _claim_from_sources(scene: BaseScene, source_indices, fraction: float,
                        rng: np.random.Generator, local_only: bool) -> np.ndarray:
    """Indices whose membership flips 0 -> 1, taken from approved context sources."""
    context = _context_indices(scene)
    if context.size == 0:
        raise NotApplicableError('no_context_points',
                                 {'context_points': 0})
    if local_only:
        centre = scene.points[_candidate_indices(scene)].mean(axis=0)
        distance = np.linalg.norm(scene.points[context] - centre, axis=1)
        radius = float(np.quantile(distance, 0.5))
        near = context[distance <= radius]
        pool = near if near.size else context
    else:
        pool = context
    if source_indices is not None:
        allowed = np.isin(scene.gt_source_id[pool], np.asarray(source_indices))
        restricted = pool[allowed]
        pool = restricted if restricted.size else pool
    want = int(round(pool.size * fraction))
    if want < 1:
        raise NotApplicableError('claim_below_one_point', {'pool': int(pool.size)})
    return pool[_take(pool.size, want, rng)]


def _scattered_selection(scene: BaseScene, pool: np.ndarray, want: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Non-structural selection: at most one point per spatial cell.

    NOISE must not look like a contiguous neighbouring region, otherwise it would be
    indistinguishable from contamination and would differ only by class name. Points are
    spread across cells instead. The cell size is refined until the lattice can hold the
    requested count, so the defect is visible rather than throttled by the grid.
    """
    want = max(1, min(int(want), pool.size))
    points = scene.points[pool]
    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1e-9)
    for resolution in (6, 10, 16, 24, 32, 48):
        cells = np.floor((points - low) / (span / resolution)).astype(np.int64)
        keys = (cells[:, 0] * (resolution + 2) + cells[:, 1]) * (resolution + 2) \
            + cells[:, 2]
        unique = np.unique(keys)
        if unique.size >= want:
            order = np.argsort(keys, kind='stable')
            sorted_keys = keys[order]
            boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
            groups = np.split(order, boundaries)
            chosen = rng.permutation(len(groups))[:want]
            return np.asarray([pool[groups[g][rng.integers(0, groups[g].size)]]
                               for g in chosen], dtype=np.int64)
    # The lattice cannot separate enough points: fall back to a uniform draw, which is
    # still non-contiguous in expectation and is reported.
    return pool[_take(pool.size, want, rng)]


def generate_example(spec: ExampleSpec, geometry, provenance) -> SyntheticExample:
    """Generate one example from the shared scene. Raises on any violation."""
    try:
        scene = build_base_scene(spec.candidate_id, spec.partition, geometry, provenance)
    except SceneError as exc:
        raise GeneratorError(str(exc)) from exc

    rng = _rng(spec)
    membership = scene.membership_gt()
    gt_membership = membership.copy()
    detail: dict = {'fraction': 0.0, 'drawn_fraction': 0.0,
                    'mechanism': 'unchanged', 'claimed_points': 0, 'dropped_points': 0}
    donors: list[str] = []
    donors_detail: dict[str, int] = {}

    if spec.example_class == GOOD:
        pass

    elif spec.example_class == MERGED_MULTI_TREE:
        # A whole neighbouring tree inside the query is claimed.
        sources = _context_sources(scene)
        if not sources:
            raise NotApplicableError('no_context_source', {})
        order = rng.permutation(len(sources))
        chosen_source = sources[int(order[0])]
        source_id = scene.source_ids[chosen_source]
        pool = _context_indices(scene)
        pool = pool[scene.gt_source_id[pool] == chosen_source]
        if pool.size == 0:
            raise NotApplicableError('donor_source_clipped_empty',
                                     {'donor': source_id})
        fraction = _draw_fraction(rng, spec.severity)
        want = int(round(pool.size * fraction))
        if want < 1:
            raise NotApplicableError('merged_below_one_point', {'pool': int(pool.size)})
        claimed = pool[_take(pool.size, want, rng)]
        membership[claimed] = True
        donors = [source_id]
        donors_detail[source_id] = int(claimed.size)
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'claim_whole_neighbour',
                  'claimed_points': int(claimed.size), 'dropped_points': 0,
                  'donor_source': source_id,
                  'donor_claimed_fraction_of_clipped': round(
                      float(claimed.size / pool.size), 6)}

    elif spec.example_class == NEIGHBOR_CONTAMINATION:
        # A contiguous neighbouring region adjacent to the candidate boundary.
        sources = _context_sources(scene)
        if not sources:
            raise NotApplicableError('no_context_source', {})
        fraction = _draw_fraction(rng, spec.severity)
        claimed = _claim_from_sources(scene, sources, fraction, rng, local_only=True)
        membership[claimed] = True
        for source_index in np.unique(scene.gt_source_id[claimed]):
            source_id = scene.source_ids[int(source_index)]
            donors.append(source_id)
            donors_detail[source_id] = int(
                (scene.gt_source_id[claimed] == source_index).sum())
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'claim_contiguous_neighbour_region',
                  'claimed_points': int(claimed.size), 'dropped_points': 0,
                  'claim_radius_fraction': 0.5}

    elif spec.example_class == NOISE:
        # Scattered, non-structural context points are claimed.
        sources = _context_sources(scene)
        if not sources:
            raise NotApplicableError('no_context_source', {})
        fraction = _draw_fraction(rng, spec.severity)
        pool = _context_indices(scene)
        want = int(round(pool.size * fraction))
        if want < 1:
            raise NotApplicableError('noise_below_one_point', {'pool': int(pool.size)})
        claimed = _scattered_selection(scene, pool, want, rng)
        membership[claimed] = True
        for source_index in np.unique(scene.gt_source_id[claimed]):
            source_id = scene.source_ids[int(source_index)]
            donors.append(source_id)
            donors_detail[source_id] = int(
                (scene.gt_source_id[claimed] == source_index).sum())
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'claim_scattered_context_points',
                  'claimed_points': int(claimed.size), 'dropped_points': 0,
                  'distinct_cells': int(claimed.size)}

    elif spec.example_class == OVER_SEGMENTED:
        # A spatially contiguous part of the candidate is no longer selected.
        candidate = _candidate_indices(scene)
        fraction = _draw_fraction(rng, spec.severity)
        axis = _unit_direction(rng)
        projection = scene.points[candidate] @ axis
        threshold = float(np.quantile(projection, fraction))
        drop = candidate[projection <= threshold]
        if drop.size < 1:
            raise NotApplicableError('over_segmented_below_one_point',
                                     {'candidate_points': int(candidate.size)})
        membership[drop] = False
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'drop_contiguous_candidate_region',
                  'claimed_points': 0, 'dropped_points': int(drop.size),
                  'axis': [round(float(v), 6) for v in axis],
                  'dropped_fraction_of_candidate': round(
                      float(drop.size / candidate.size), 6)}

    elif spec.example_class == CROWN_INCOMPLETE:
        # The contiguous crown region is no longer selected. The real crown points stay
        # in the scene: the segmentation did not pick them, the tree still has them.
        candidate = _candidate_indices(scene)
        fraction = _draw_fraction(rng, spec.severity)
        z = scene.points[candidate, 2]
        z_low, z_high = float(z.min()), float(z.max())
        threshold = z_high - (z_high - z_low) * fraction
        drop = candidate[z > threshold]
        if drop.size < 1:
            raise NotApplicableError('crown_incomplete_below_one_point',
                                     {'candidate_points': int(candidate.size)})
        membership[drop] = False
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'drop_contiguous_crown_region',
                  'claimed_points': 0, 'dropped_points': int(drop.size),
                  'z_cut': round(threshold, 6),
                  'z_span': [round(z_low, 6), round(z_high, 6)],
                  'dropped_fraction_of_candidate': round(
                      float(drop.size / candidate.size), 6)}

    elif spec.example_class == STEM_MISSING:
        evidence = stem_evidence(scene)
        if not evidence['applicable']:
            raise NotApplicableError(STEM_NOT_APPLICABLE, evidence)
        candidate = _candidate_indices(scene)
        fraction = _draw_fraction(rng, spec.severity)
        z_low, z_high = evidence['z_span']
        z_cut = z_low + (z_high - z_low) * fraction
        axis_x, axis_y = evidence['stem_axis_xy']
        horizontal = np.hypot(scene.points[candidate, 0] - axis_x,
                              scene.points[candidate, 1] - axis_y)
        drop = candidate[(scene.points[candidate, 2] <= z_cut)
                         & (horizontal <= evidence['radius_m'])]
        if drop.size < 1:
            raise NotApplicableError('stem_missing_below_one_point',
                                     {'candidate_points': int(candidate.size)})
        membership[drop] = False
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'mechanism': 'drop_contiguous_stem_region',
                  'claimed_points': 0, 'dropped_points': int(drop.size),
                  'z_cut': round(z_cut, 6), 'stem_axis_xy': evidence['stem_axis_xy'],
                  'stem_radius_m': evidence['radius_m'],
                  'stem_evidence': {k: v for k, v in evidence.items()
                                    if k != 'lower_mask'}}

    else:  # pragma: no cover - guarded by ExampleSpec
        raise GeneratorError(f'unhandled class {spec.example_class}')

    changed = membership != gt_membership
    donors = sorted(set(donors))
    union = scene_source_union(scene)
    # Unified fail-closed check over every source that contributed a point.
    authorised_union = provenance.assert_scene_sources(spec.candidate_id, union)
    if tuple(authorised_union) != union:
        raise GeneratorError('source union normalization mismatch')

    label = LABEL_PASS if spec.example_class == GOOD else LABEL_REVIEW
    record = {
        'generator_version': GENERATOR_VERSION,
        'membership_channel': MEMBERSHIP_CHANNEL,
        'key': spec.key,
        'candidate_id': spec.candidate_id,
        'partition': spec.partition,
        'site': scene.site,
        'class': spec.example_class,
        'severity': spec.severity,
        'label': label,
        'seed': spec.seed,
        'scene_version': scene.audit.get('scene_version'),
        'query_box': scene.audit.get('query_box'),
        'query_pad_m': scene.audit.get('query_pad_m'),
        'donors': donors,
        'donor_point_counts': donors_detail,
        'context_ids': list(scene.source_ids[1:]),
        'point_count': scene.point_count,
        'actual_point_source_ids': list(union),
        'actual_point_source_counts': scene.source_counts(),
        'actual_point_source_aliases': {
            source_id: list(provenance.source_aliases_of.get(source_id, ()))
            for source_id in union},
        'candidate_points': scene.candidate_points,
        'context_points': scene.context_points,
        'membership_fraction': round(float(membership.mean()), 6),
        'gt_membership_fraction': round(float(gt_membership.mean()), 6),
        'changed_membership_points': int(changed.sum()),
        'changed_membership_fraction': round(float(changed.mean()), 6),
        'claimed_points': int(detail.get('claimed_points', 0)),
        'dropped_points': int(detail.get('dropped_points', 0)),
        'source_union': list(union),
        'corruption': detail,
    }
    return SyntheticExample(spec=spec, points=scene.points, membership=membership,
                            label=label, changed_membership=changed,
                            source_ids=scene.source_ids, provenance=record)


def enumerate_specs(candidate_ids, partition: str, classes=CLASSES,
                    severities=SEVERITIES, seed_base: int = 0,
                    repeats: int = 1) -> list[ExampleSpec]:
    """Deterministic, order-stable spec list for a bounded diagnostic or a full run."""
    specs: list[ExampleSpec] = []
    for index, candidate_id in enumerate(candidate_ids):
        for class_index, example_class in enumerate(classes):
            for severity_index, severity in enumerate(severities):
                for repeat in range(repeats):
                    seed = (seed_base + 1_000_003 * index + 10_007 * class_index
                            + 101 * severity_index + repeat)
                    specs.append(ExampleSpec(candidate_id=candidate_id,
                                             partition=partition,
                                             example_class=example_class,
                                             severity=severity, seed=seed))
    return specs
