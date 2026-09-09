"""Gate D synthetic tree-QC generator.

Seven classes, three severities, generated online from **real publisher geometry**
only. No Gaussian point cloud is ever invented to stand in for a segmentation defect:

* ``GOOD``                 — the official observation, membership untouched;
* ``MERGED_MULTI_TREE``    — real neighbouring trees moved in whole/partial;
* ``NEIGHBOR_CONTAMINATION`` — real neighbouring boundary geometry added;
* ``OVER_SEGMENTED``       — a spatially contiguous subset of the same tree;
* ``CROWN_INCOMPLETE``     — a spatially contiguous crown (top) region removed;
* ``STEM_MISSING``         — a spatially contiguous stem/lower region removed;
* ``NOISE``                — real non-member plot geometry dominates the cloud.

Every donor comes from the candidate's own surviving context and is re-authorised
here, so the generator fails closed even if a caller hands it an illegal spec.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from gate_d_geometry import CandidateGeometry, TreeGeometry
from gate_d_provenance import MODEL_SELECTION_PARTITIONS

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

LOW = 'LOW'
MEDIUM = 'MEDIUM'
HIGH = 'HIGH'
SEVERITIES = (LOW, MEDIUM, HIGH)
SEVERITY_LEVEL = {LOW: 0, MEDIUM: 1, HIGH: 2}

# Target binary label. PASS = the observation is a usable single-tree instance.
LABEL_PASS = 'PASS'
LABEL_REVIEW = 'REVIEW_REQUIRED'
LABELS = (LABEL_PASS, LABEL_REVIEW)

GENERATOR_VERSION = 'GATE_D_SYNTHETIC_GENERATOR_V1'

# Corruption magnitude per severity. The *fraction* is always drawn from the same
# uniform range for every severity, so severity carries the strength of the defect
# without becoming a class fingerprint. These are the only hand-set numbers.
FRACTION_RANGE = (0.06, 0.34)
SEVERITY_MULTIPLIER = {LOW: 0.55, MEDIUM: 1.0, HIGH: 1.65}
FRACTION_CEILING = 0.72

# Every cloud is normalised to a size drawn from one class-independent distribution
# before sampling. Without this the cloud size alone reveals the class: MERGED appends
# whole neighbouring trees, NOISE prepends background geometry and OVER_SEGMENTED drops
# a region, so the raw count is a construction artefact rather than a property of the
# defect. Trees in this benchmark range from ~4.8 k to ~5.8 M points, so the target is a
# range rather than a single value; the draw is uniform for every class and severity.
NORMALISE_RANGE = (30_000, 70_000)


class GeneratorError(RuntimeError):
    pass


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
    """One generated example plus its complete provenance."""

    spec: ExampleSpec
    points: np.ndarray            # (N, 3) float32 metres, real geometry
    membership: np.ndarray        # (N,) bool official GT membership flag
    label: str                    # PASS | REVIEW_REQUIRED
    donors: tuple[str, ...]       # canonical ids whose geometry is present
    provenance: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.spec.key


def _rng(spec: ExampleSpec) -> np.random.Generator:
    """Seed stream derived from the spec only; no hidden state, no clock."""
    material = f'{GENERATOR_VERSION}|{spec.key}'.encode()
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def _draw_fraction(rng: np.random.Generator, severity: str) -> float:
    base = float(rng.uniform(*FRACTION_RANGE))
    return float(min(FRACTION_CEILING, base * SEVERITY_MULTIPLIER[severity]))


def _unit_direction(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return vector / norm


def _subsample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic uniform draw of `count` indices.

    Uses a partial permutation rather than ``choice(replace=False)``, which is
    pathologically slow once the array is large.
    """
    count = max(1, min(int(count), points.shape[0]))
    if count >= points.shape[0]:
        return np.arange(points.shape[0])
    return np.sort(rng.permutation(points.shape[0])[:count])


def _stem_axis_xy(points: np.ndarray) -> tuple[float, float]:
    """Robust trunk axis: median XY of the lowest decile of the cloud."""
    if points.shape[0] == 0:
        return 0.0, 0.0
    low = points[:, 2] <= np.quantile(points[:, 2], 0.10)
    subset = points[low]
    if subset.shape[0] == 0:
        subset = points
    return float(np.median(subset[:, 0])), float(np.median(subset[:, 1]))


def _background_in_region(geometry: CandidateGeometry, mask_fn,
                          rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Real non-member geometry inside a region, for filling a vacated volume.

    A missing region must not be left empty: a point-only model cannot see a hole, and
    a hole is not what a segmentation defect looks like. What the tree loses, the plot
    actually contains — ground, understorey or a neighbouring tree — so the region is
    refilled with real geometry carrying membership 0.
    """
    if geometry.background.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), 0
    inside = mask_fn(geometry.background)
    if not inside.any():
        return np.zeros((0, 3), dtype=np.float32), 0
    selected = geometry.background[inside]
    return np.asarray(selected, dtype=np.float32), int(np.unique(
        geometry.background_label[inside]).size)


def _select_donors(geometry: CandidateGeometry, rng: np.random.Generator,
                   wanted: int) -> tuple[TreeGeometry, ...]:
    """Pick real context trees deterministically from the seeded stream."""
    available = [t for t in geometry.context if t.points.shape[0] > 0]
    if not available:
        return ()
    order = rng.permutation(len(available))
    return tuple(available[i] for i in order[:max(1, min(wanted, len(available)))])


def _normalise_count(points: np.ndarray, membership: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalise the cloud to a size drawn from the shared class-independent range.

    A cloud larger than the target is reduced by a uniform deterministic draw; a
    smaller cloud is upsampled by repeating points from the same uniform draw. Repeats
    are real coordinates, so no synthetic geometry is introduced, and the draw is
    identical in distribution for every class and severity.
    """
    count = points.shape[0]
    target = int(rng.integers(NORMALISE_RANGE[0], NORMALISE_RANGE[1] + 1))
    if count == target:
        return points, membership, {
            'action': 'exact', 'before': int(count), 'after': int(count)}
    if count > target:
        keep = np.sort(rng.permutation(count)[:target])
        return points[keep], membership[keep], {
            'action': 'reduced', 'before': int(count), 'after': int(keep.size),
            'target': target}
    keep = rng.permutation(count)
    repeats = int(np.ceil(target / count))
    chosen = np.concatenate([keep] * repeats)[:target]
    return points[chosen], membership[chosen], {
        'action': 'upsampled', 'before': int(count), 'after': int(chosen.size),
        'target': target, 'unique_sources': int(count)}


def generate_example(spec: ExampleSpec, geometry: CandidateGeometry,
                     provenance) -> SyntheticExample:
    """Generate one example. Raises on any provenance or geometry violation."""
    if geometry.candidate_id != spec.candidate_id:
        raise GeneratorError(
            f'geometry is for {geometry.candidate_id}, spec asks {spec.candidate_id}')
    candidate = geometry.candidate
    if candidate.points.shape[0] == 0:
        raise GeneratorError(f'{spec.candidate_id} has no stored geometry')

    rng = _rng(spec)
    points = np.asarray(candidate.points, dtype=np.float32)
    membership = np.asarray(candidate.membership, dtype=bool)
    donors: tuple[str, ...] = ()
    # `drawn_fraction` comes from the one shared distribution every class and severity
    # uses. `fraction` is the realised magnitude after the class-specific geometry
    # mapping, which necessarily differs between a crown cut and a boundary injection.
    detail: dict = {'fraction': 0.0, 'drawn_fraction': 0.0}

    if spec.example_class == GOOD:
        # Official GT membership is preserved exactly.
        pass

    elif spec.example_class == MERGED_MULTI_TREE:
        chosen = _select_donors(geometry, rng, 2)
        if not chosen:
            raise GeneratorError(f'{spec.candidate_id} has no context donor for MERGED')
        fraction = _draw_fraction(rng, spec.severity)
        extra_points, extra_membership, used = [], [], []
        for donor in chosen:
            take = _subsample(donor.points, int(donor.points.shape[0] * fraction), rng)
            extra_points.append(donor.points[take])
            extra_membership.append(np.asarray(donor.membership[take], dtype=bool))
            used.append(donor.canonical_tree_id)
        points = np.concatenate([points, *extra_points], axis=0)
        membership = np.concatenate([membership, *extra_membership], axis=0)
        donors = tuple(used)
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'donors_used': list(used),
                  'donor_points': int(sum(p.shape[0] for p in extra_points))}

    elif spec.example_class == NEIGHBOR_CONTAMINATION:
        chosen = _select_donors(geometry, rng, 2)
        if not chosen:
            raise GeneratorError(
                f'{spec.candidate_id} has no context donor for CONTAMINATION')
        fraction = _draw_fraction(rng, spec.severity)
        centre = points.mean(axis=0)
        radius = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * 0.5
        extra_points, used = [], []
        for donor in chosen:
            distance = np.linalg.norm(donor.points - centre, axis=1)
            near = np.flatnonzero(distance <= radius)
            if near.size == 0:
                near = np.argsort(distance)[:max(1, donor.points.shape[0] // 100)]
            take = near[_subsample(donor.points[near], int(near.size * fraction), rng)]
            extra_points.append(donor.points[take])
            used.append(donor.canonical_tree_id)
        points = np.concatenate([points, *extra_points], axis=0)
        membership = np.concatenate(
            [membership, np.zeros(sum(p.shape[0] for p in extra_points), dtype=bool)])
        donors = tuple(used)
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'donors_used': list(used),
                  'boundary_radius_m': radius,
                  'donor_points': int(sum(p.shape[0] for p in extra_points))}

    elif spec.example_class == OVER_SEGMENTED:
        fraction = _draw_fraction(rng, spec.severity)
        axis = _unit_direction(rng)
        projection = points @ axis
        threshold = float(np.quantile(projection, fraction))
        keep = projection > threshold
        if keep.sum() < 8 or (~keep).sum() < 8:
            raise GeneratorError(f'{spec.candidate_id} cannot be over-segmented')
        points, membership = points[keep], membership[keep]
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'axis': [round(float(v), 6) for v in axis],
                  'dropped_points': int((~keep).sum())}

    elif spec.example_class == CROWN_INCOMPLETE:
        fraction = _draw_fraction(rng, spec.severity)
        z_low, z_high = float(points[:, 2].min()), float(points[:, 2].max())
        threshold = z_high - (z_high - z_low) * fraction
        keep = points[:, 2] <= threshold
        if keep.sum() < 8 or (~keep).sum() < 8:
            raise GeneratorError(f'{spec.candidate_id} cannot lose a crown region')
        removed = points[~keep]
        fill, fill_instances = _background_in_region(
            geometry, lambda xyz: xyz[:, 2] > threshold, rng)
        points = np.concatenate([points[keep], fill], axis=0)
        membership = np.concatenate(
            [membership[keep], np.zeros(fill.shape[0], dtype=bool)])
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'z_cut': round(threshold, 6),
                  'z_span': [round(z_low, 6), round(z_high, 6)],
                  'removed_points': int((~keep).sum()),
                  'fill_points': int(fill.shape[0]),
                  'fill_instances': fill_instances,
                  'removed_centroid_z': round(float(removed[:, 2].mean()), 6)}

    elif spec.example_class == STEM_MISSING:
        fraction = _draw_fraction(rng, spec.severity)
        axis_x, axis_y = _stem_axis_xy(points)
        z_low, z_high = float(points[:, 2].min()), float(points[:, 2].max())
        z_cut = z_low + (z_high - z_low) * fraction
        horizontal = np.hypot(points[:, 0] - axis_x, points[:, 1] - axis_y)
        radius = float(np.quantile(horizontal, 0.35))
        keep = ~((points[:, 2] <= z_cut) & (horizontal <= radius))
        if keep.sum() < 8 or (~keep).sum() < 8:
            raise GeneratorError(f'{spec.candidate_id} cannot lose a stem region')
        # The publisher plot floor sits a few decimetres below the tree base, so the
        # refill region is the whole vertical column under the stem, not only the
        # cylinder that the tree itself occupied.
        fill_radius = max(radius, 1.5)
        fill, fill_instances = _background_in_region(
            geometry,
            lambda xyz: ((xyz[:, 2] <= z_cut)
                         & (np.hypot(xyz[:, 0] - axis_x, xyz[:, 1] - axis_y)
                            <= fill_radius)),
            rng)
        points = np.concatenate([points[keep], fill], axis=0)
        membership = np.concatenate(
            [membership[keep], np.zeros(fill.shape[0], dtype=bool)])
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'z_cut': round(z_cut, 6),
                  'stem_radius_m': round(radius, 6),
                  'fill_radius_m': round(fill_radius, 6),
                  'stem_axis_xy': [round(axis_x, 6), round(axis_y, 6)],
                  'removed_points': int((~keep).sum()),
                  'fill_points': int(fill.shape[0]),
                  'fill_instances': fill_instances}

    elif spec.example_class == NOISE:
        if geometry.background.shape[0] == 0:
            raise GeneratorError(f'{spec.candidate_id} has no real non-member geometry')
        fraction = _draw_fraction(rng, spec.severity)
        background = np.asarray(geometry.background, dtype=np.float32)
        take = _subsample(background, int(background.shape[0] * max(fraction, 0.5)), rng)
        noise_points = background[take]
        keep = _subsample(points, int(points.shape[0] * (1.0 - fraction)), rng)
        points = np.concatenate([noise_points, points[keep]], axis=0)
        membership = np.concatenate(
            [np.zeros(noise_points.shape[0], dtype=bool), membership[keep]])
        detail = {'fraction': fraction, 'drawn_fraction': fraction,
                  'noise_points': int(noise_points.shape[0]),
                  'noise_source_trees': int(np.unique(
                      geometry.background_label[take]).size),
                  'surviving_candidate_points': int(keep.size)}

    else:  # pragma: no cover - guarded by ExampleSpec
        raise GeneratorError(f'unhandled class {spec.example_class}')

    raw_point_count = int(points.shape[0])
    points, membership, normalisation = _normalise_count(points, membership, rng)
    if points.shape[0] == 0:
        raise GeneratorError(f'{spec.candidate_id} produced an empty cloud')

    donors = tuple(donors)
    # Fail closed on the real contract, using the same gate the dataset uses.
    provenance.assert_example_sources(spec.candidate_id, donors)

    label = LABEL_PASS if spec.example_class == GOOD else LABEL_REVIEW
    record = {
        'generator_version': GENERATOR_VERSION,
        'key': spec.key,
        'candidate_id': spec.candidate_id,
        'partition': spec.partition,
        'site': candidate.site,
        'class': spec.example_class,
        'severity': spec.severity,
        'label': label,
        'seed': spec.seed,
        'donors': list(donors),
        'context_ids': list(geometry.context_ids),
        'point_count': int(points.shape[0]),
        'raw_point_count': raw_point_count,
        'normalisation': normalisation,
        'membership_fraction': round(float(membership.mean()), 6)
        if membership.size else 0.0,
        'source_union': sorted({spec.candidate_id, *donors}),
        'corruption': detail,
    }
    return SyntheticExample(spec=spec, points=np.asarray(points, dtype=np.float32),
                            membership=np.asarray(membership, dtype=bool),
                            label=label, donors=donors, provenance=record)


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
