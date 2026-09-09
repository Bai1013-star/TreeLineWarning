"""Gate D generator fingerprint diagnostics.

A synthetic dataset is worthless if the corruption is detectable from an artefact of
how it was produced rather than from the geometry. These checks look for the five
fingerprints named by the task:

1. **point count fingerprint** — does cloud size predict the class?
2. **fixed fraction fingerprint** — is the corruption fraction class-specific?
3. **fixed-axis fingerprint** — is the corruption direction class-specific?
4. **class-specific sampling fingerprint** — does the sampler treat classes differently?
5. **metadata leakage** — is the class predictable from the exposed tensor or from
   per-example identifiers?

Statistical honesty: every test is a permutation test with a fixed seed, and a *small*
p-value means the class is predictable from that artefact, which is the failure. A
large p-value is the pass condition. Nothing here is a claim about scientific validity;
it only rules out construction artefacts.
"""
from __future__ import annotations

from dataclasses import dataclass

import gate_d_sampler as sampler
import numpy as np

FINGERPRINT_VERSION = 'GATE_D_FINGERPRINT_V1'
PERMUTATIONS = 2000
ALPHA = 0.01
# Leave-one-out nearest-centroid accuracy above this (with chance subtracted) is a
# leak. Chance is 1 / number of classes present.
STAT_LEAK_MARGIN = 0.10


@dataclass(frozen=True)
class ExampleRecord:
    """The minimal per-example facts a fingerprint test may look at."""

    key: str
    candidate_id: str
    partition: str
    example_class: str
    severity: str
    label: str
    source_points: int
    raw_source_points: int
    sampled_points: int
    corruption_fraction: float
    drawn_fraction: float
    corruption_axis: tuple[float, float, float] | None
    membership_source: float
    membership_sampled: float
    feature_mean: tuple[float, float, float, float]
    feature_std: tuple[float, float, float, float]
    spatial_coverage: float


def _permutation_p(values: np.ndarray, groups: np.ndarray, statistic,
                   rng: np.random.Generator) -> float:
    """Two-sided permutation p-value for a group statistic.

    `groups` is converted to integer codes once, so the statistic never has to deal
    with string labels and the permutations stay cheap.
    """
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    if np.unique(inverse).size < 2:
        return 1.0
    observed = statistic(values, inverse)
    count = 0
    for _ in range(PERMUTATIONS):
        if statistic(values, rng.permutation(inverse)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (PERMUTATIONS + 1)


def _range_ratio(values: np.ndarray, groups: np.ndarray) -> float:
    """Between-group share of total variance (0 = no class signal, 1 = all of it)."""
    values = np.asarray(values, dtype=float)
    inverse = np.asarray(groups, dtype=np.int64)
    counts = np.bincount(inverse).astype(float)
    totals = np.bincount(inverse, weights=values)
    overall = float(np.var(values))
    if overall <= 1e-12:
        return 0.0
    means = totals / np.maximum(counts, 1)
    between = float(np.sum(counts * (means - values.mean()) ** 2) / counts.sum())
    return between / overall


def _loo_nearest_centroid(features: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out accuracy of a nearest-centroid classifier (a simple leak probe).

    Vectorised: the class centroid sum per class is accumulated once, then each
    held-out example is scored against ``(sum - itself) / (count - 1)``.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    classes, inverse = np.unique(labels, return_inverse=True)
    if classes.size < 2 or features.shape[0] < classes.size + 2:
        return 0.0
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-12] = 1.0
    scaled = (features - mean) / std

    counts = np.bincount(inverse, minlength=classes.size).astype(np.float64)
    sums = np.zeros((classes.size, scaled.shape[1]))
    np.add.at(sums, inverse, scaled)
    # Centroids that exclude the held-out example.
    leave_one_out = (sums[inverse] - scaled) / np.maximum(counts[inverse] - 1, 1)[:, None]
    distance = np.linalg.norm(leave_one_out[:, None, :] - sums[None, :, :]
                              / np.maximum(counts, 1)[None, :, None], axis=2)
    prediction = distance.argmin(axis=1)
    return float((prediction == inverse).mean())


def _leak_permutation_p(features: np.ndarray, labels: np.ndarray,
                        rng: np.random.Generator) -> tuple[float, float]:
    """Permutation p-value for 'the tensor predicts the class above chance'.

    A fixed accuracy margin would be arbitrary: a nearest-centroid probe overfits a
    handful of examples. Shuffling the labels gives the null distribution of the same
    probe, so the p-value answers the question that actually matters.
    """
    observed = _loo_nearest_centroid(features, labels)
    count = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(labels)
        if _loo_nearest_centroid(features, shuffled) >= observed - 1e-12:
            count += 1
    return observed, (count + 1) / (PERMUTATIONS + 1)


def _check(name: str, p_value: float, detail: dict, critical_if: bool) -> dict:
    return {'check': name, 'p_value': round(float(p_value), 6),
            'pass': not critical_if, 'critical': bool(critical_if), 'detail': detail}


def analyse(records: list[ExampleRecord], seed: int = 20260909) -> dict:
    if not records:
        raise ValueError('no example records to analyse')
    rng = np.random.default_rng(seed)
    classes = np.array([r.example_class for r in records])
    source_points = np.array([r.source_points for r in records], dtype=float)
    sampled_points = np.array([r.sampled_points for r in records], dtype=float)
    coverage = np.array([r.spatial_coverage for r in records], dtype=float)
    membership_source = np.array([r.membership_source for r in records], dtype=float)
    membership_sampled = np.array([r.membership_sampled for r in records], dtype=float)
    feature_stats = np.array(
        [list(r.feature_mean) + list(r.feature_std) for r in records], dtype=float)

    checks: list[dict] = []

    # 1. point count fingerprint: the cloud size that reaches the model must not
    #    encode the class. Both the pre-normalisation and post-normalisation counts are
    #    tested, because either one being class-specific would be an artefact.
    p_raw = _permutation_p(
        np.array([r.raw_source_points for r in records], dtype=float), classes,
        _range_ratio, rng)
    checks.append(_check(
        'point_count_fingerprint_raw', p_raw,
        {'statistic': 'between_group_mean_variance_ratio',
         'raw_min': int(min(r.raw_source_points for r in records)),
         'raw_max': int(max(r.raw_source_points for r in records)),
         'note': 'pre-normalisation size; MERGED and NOISE append geometry, so a '
                 'class-dependent raw size is expected and is not a model-input leak'},
        critical_if=False))
    p_source = _permutation_p(source_points, classes, _range_ratio, rng)
    checks.append(_check(
        'point_count_fingerprint_source', p_source,
        {'statistic': 'between_group_mean_variance_ratio',
         'min': int(source_points.min()), 'max': int(source_points.max()),
         'note': 'post-normalisation size must be class-independent, otherwise the '
                 'generator leaks the class through cloud size'},
        critical_if=p_source < ALPHA))
    unique_sampled = np.unique(sampled_points)
    checks.append(_check(
        'point_count_fingerprint_sampled', 1.0 if unique_sampled.size == 1 else 0.0,
        {'distinct_sampled_sizes': [int(v) for v in unique_sampled],
         'note': 'every example must reach the model at the same fixed size'},
        critical_if=unique_sampled.size != 1))

    # 2. fixed fraction fingerprint: the *drawn* fraction must not be class-specific.
    #    GOOD has no corruption by construction, so it is excluded; the realised
    #    magnitude is reported separately because mapping a fraction onto a crown cut
    #    versus a boundary injection cannot yield the same displacement.
    defect_rows = [r for r in records if r.example_class != 'GOOD']
    drawn = np.array([r.drawn_fraction for r in defect_rows], dtype=float)
    realised = np.array([r.corruption_fraction for r in defect_rows], dtype=float)
    defect_classes = np.array([r.example_class for r in defect_rows])
    p_fraction = _permutation_p(drawn, defect_classes, _range_ratio, rng)
    per_class_drawn = {c: {'n': int((defect_classes == c).sum()),
                           'mean': round(float(drawn[defect_classes == c].mean()), 6),
                           'min': round(float(drawn[defect_classes == c].min()), 6),
                           'max': round(float(drawn[defect_classes == c].max()), 6)}
                       for c in np.unique(defect_classes)}
    checks.append(_check(
        'fixed_fraction_fingerprint', p_fraction,
        {'fraction_min': round(float(drawn.min()), 6),
         'fraction_max': round(float(drawn.max()), 6),
         'fraction_mean': round(float(drawn.mean()), 6),
         'fraction_distinct_values': int(np.unique(drawn).size),
         'per_class_drawn_fraction': per_class_drawn,
         'realised_magnitude_mean': round(float(realised.mean()), 6),
         'excluded_classes': ['GOOD'],
         'note': 'the drawn fraction comes from one shared distribution; the realised '
                 'magnitude differs by class because the geometry mapping differs'},
        critical_if=p_fraction < ALPHA))

    # 3. fixed-axis fingerprint: corruption direction must not be class-specific.
    axis_rows = [r for r in records if r.corruption_axis is not None]
    if len(axis_rows) >= 8:
        axis = np.array([r.corruption_axis for r in axis_rows], dtype=float)
        axis_class = np.array([r.example_class for r in axis_rows])
        dominant = np.abs(axis).argmax(axis=1)
        # A fixed-axis defect shows up as a class-constant dominant axis.
        contingency = np.zeros((np.unique(axis_class).size, 3))
        class_index = {c: i for i, c in enumerate(np.unique(axis_class))}
        for c, d in zip(axis_class, dominant, strict=True):
            contingency[class_index[c], d] += 1
        expected = contingency.sum(axis=1, keepdims=True) * \
            contingency.sum(axis=0, keepdims=True) / contingency.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = np.nansum((contingency - expected) ** 2 / expected)
        p_axis = float(np.exp(-chi2 / 2.0)) if chi2 > 0 else 1.0
        checks.append(_check(
            'fixed_axis_fingerprint', p_axis,
            {'dominant_axis_histogram': {str(i): int((dominant == i).sum())
                                         for i in range(3)},
             'per_class_dominant_axis': {
                 c: int(np.bincount(dominant[axis_class == c], minlength=3).argmax())
                 for c in np.unique(axis_class)},
             'chi2': round(float(chi2), 6),
             'note': 'a class-constant dominant axis is the failure mode'},
            critical_if=p_axis < ALPHA))
    else:
        checks.append(_check('fixed_axis_fingerprint', 1.0,
                             {'applicable': False,
                              'reason': 'fewer than 8 axis-bearing examples'},
                             critical_if=False))

    # 4. class-specific sampling fingerprint: the sampler must be label-agnostic.
    #
    #    Membership genuinely differs by class — that is the defect signal, not an
    #    artefact. The artefact question is whether the sampler *shifts* the membership
    #    fraction differently for different classes. A label-agnostic sampler applied
    #    to a membership distribution must produce a class-independent shift, so the
    #    gap (sampled minus source) is the statistic under test.
    membership_gap = membership_sampled - membership_source
    p_sampling = _permutation_p(membership_gap, classes, _range_ratio, rng)
    per_class_gap = {c: round(float(membership_gap[classes == c].mean()), 6)
                     for c in np.unique(classes)}
    checks.append(_check(
        'class_specific_sampling_fingerprint', p_sampling,
        {'mean_membership_source': round(float(membership_source.mean()), 6),
         'mean_membership_sampled': round(float(membership_sampled.mean()), 6),
         'mean_membership_gap': round(float(membership_gap.mean()), 6),
         'max_abs_membership_gap': round(float(np.abs(membership_gap).max()), 6),
         'per_class_mean_gap': per_class_gap,
         'mean_spatial_coverage': round(float(coverage.mean()), 6),
         'statistic': 'between-group variance share of the sampled-minus-source '
                      'membership gap',
         'note': 'the absolute membership fraction is the class signal; only a '
                 'class-dependent sampler shift would be an artefact'},
        critical_if=p_sampling < ALPHA))

    # 5. metadata leakage: class must not be predictable from exposed tensors alone.
    accuracy, p_stats = _leak_permutation_p(feature_stats, classes, rng)
    chance = 1.0 / np.unique(classes).size
    checks.append(_check(
        'metadata_leakage_tensor_stats', p_stats,
        {'loo_nearest_centroid_accuracy': round(float(accuracy), 6),
         'chance_accuracy': round(float(chance), 6),
         'margin_over_chance': round(float(accuracy - chance), 6),
         'feature_channels': list(sampler.FEATURES),
         'feature_channel_count': sampler.FEATURE_COUNT,
         'test': 'permutation p-value for a nearest-centroid probe on the 8 per-channel '
                 'summary statistics of the exposed tensor',
         'note': 'the tensor carries 4 declared channels only; no provenance or class '
                 'field exists in the model input'},
        critical_if=p_stats < ALPHA))

    # Identifier leakage: class must be independent of candidate and partition.
    per_candidate: dict[str, set[str]] = {}
    per_severity: dict[str, set[str]] = {}
    for record in records:
        per_candidate.setdefault(record.candidate_id, set()).add(record.example_class)
        per_severity.setdefault(record.severity, set()).add(record.example_class)
    class_sets_identical = len({tuple(sorted(v)) for v in per_candidate.values()}) == 1
    severity_sets_identical = len({tuple(sorted(v)) for v in per_severity.values()}) == 1
    checks.append(_check(
        'metadata_leakage_identifiers',
        1.0 if (class_sets_identical and severity_sets_identical) else 0.0,
        {'class_sets_identical_across_candidates': class_sets_identical,
         'class_sets_identical_across_severities': severity_sets_identical,
         'candidates': len(per_candidate),
         'severities': sorted(per_severity),
         'note': 'every candidate and severity carries the same class set, so an '
                 'identifier cannot stand in for the label'},
        critical_if=not (class_sets_identical and severity_sets_identical)))

    critical = [c['check'] for c in checks if c['critical']]
    return {
        'fingerprint_version': FINGERPRINT_VERSION,
        'examples': len(records),
        'permutations': PERMUTATIONS,
        'alpha': ALPHA,
        'checks': checks,
        'critical_fingerprints': critical,
        'GENERATOR_FINGERPRINT': 'NONE_DETECTED' if not critical else 'CRITICAL',
        'CRITICAL_FINGERPRINTS': critical,
    }
