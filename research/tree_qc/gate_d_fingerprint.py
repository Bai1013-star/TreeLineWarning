"""Gate D generator fingerprint diagnostics.

A synthetic dataset is worthless if the corruption is detectable from an artefact of how
it was produced rather than from the geometry. These checks look for the five
fingerprints named by the task:

1. **point count fingerprint** — does cloud size predict the class?
2. **fixed fraction fingerprint** — is the corruption fraction class-specific?
3. **fixed-axis fingerprint** — is the corruption direction class-specific?
4. **class-specific sampling fingerprint** — does the sampler treat classes differently?
5. **metadata leakage** — is the class predictable from the exposed tensor or from
   per-example identifiers?

The leak probe is an honest leave-one-out classifier: for each sample it builds class
references **from the other samples only**, predicts that sample's class, and only then
compares with the truth. The sample's own label is never used to pick a reference.
Two negative controls are mandatory:

* **random control** — features independent of the label. The detector must land near
  chance; a near-perfect score here means the detector is broken, not that a leak exists.
* **implanted shortcut** — one perfectly class-coded scalar added to the features. The
  detector must find it, or the gate is blind.

If either control misbehaves, the fingerprint gate itself FAILS.
"""
from __future__ import annotations

from dataclasses import dataclass

import gate_d_sampler as sampler
import numpy as np

FINGERPRINT_VERSION = 'GATE_D_FINGERPRINT_V2'
PERMUTATIONS = 2000
ALPHA = 0.01
# A single feature independent of the class scores ~1/k in expectation, but the control
# takes the maximum over all features, which inflates the null. The threshold is set
# above that selection effect and far below a class-coded scalar's ~1.0.
RANDOM_CONTROL_MAX_SEPARATION = 0.25
IMPLANTED_CONTROL_MIN_SEPARATION = 0.90
# A membership-only rule above this out-of-sample accuracy would be a real shortcut.
MEMBERSHIP_SHORTCUT_LIMIT = 0.80


@dataclass(frozen=True)
class ExampleRecord:
    """The minimal per-example facts a fingerprint test may look at."""

    key: str
    candidate_id: str
    partition: str
    example_class: str
    severity: str
    label: str
    point_count: int
    sampled_points: int
    membership_fraction: float
    gt_membership_fraction: float
    intended_membership_shift: float
    changed_membership_points: int
    changed_membership_fraction: float
    candidate_points: int
    context_points: int
    claimed_points: int
    dropped_points: int
    source_count: int
    corruption_fraction: float
    drawn_fraction: float
    corruption_axis: tuple[float, float, float] | None
    membership_sampled: float
    candidate_sampled: int
    context_sampled: int
    changed_evidence_sampled: int
    spatial_coverage: float
    height_coverage: float
    feature_mean: tuple[float, float, float, float]
    feature_std: tuple[float, float, float, float]
    bbox_extent: tuple[float, float, float]
    centroid_norm: float
    duplicate_rate: float

    def audit_features(self) -> tuple[float, ...]:
        """Audit-only scalars used to hunt for geometric shortcuts.

        These never enter the model input; they exist so the diagnostic can show whether
        the generator leaves a shortcut in the geometry itself.
        """
        return (float(self.point_count), float(self.membership_fraction),
                float(self.gt_membership_fraction),
                float(self.intended_membership_shift),
                float(self.changed_membership_points),
                float(self.changed_membership_fraction),
                float(self.candidate_points), float(self.context_points),
                float(self.claimed_points), float(self.dropped_points),
                float(self.source_count), float(self.corruption_fraction),
                float(self.drawn_fraction), float(self.membership_sampled),
                float(self.candidate_sampled), float(self.context_sampled),
                float(self.changed_evidence_sampled), float(self.spatial_coverage),
                float(self.height_coverage), *[float(v) for v in self.bbox_extent],
                float(self.centroid_norm), float(self.duplicate_rate),
                *[float(v) for v in self.feature_mean],
                *[float(v) for v in self.feature_std])


def _standardise(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def max_feature_class_separation(features: np.ndarray, labels: np.ndarray) -> float:
    """Largest per-feature between-class share of variance, over all features.

    A perfectly class-coded scalar scores ~1.0; a feature independent of the class scores
    ~1/k in expectation. Unlike an L2 nearest-centroid probe on a mixed matrix, this is
    not diluted by unrelated dimensions, so it is the decisive detector for the
    "implanted shortcut" control.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    _, inverse = np.unique(labels, return_inverse=True)
    best = 0.0
    for column in range(features.shape[1]):
        best = max(best, _range_ratio(features[:, column], inverse))
    return float(best)


def loo_nearest_centroid(features: np.ndarray, labels: np.ndarray) -> float:
    """Honest leave-one-out accuracy of a nearest-centroid classifier.

    For each sample the class references **and the standardisation statistics** are built
    from the remaining samples only. Using the held-out sample's own value in the mean or
    standard deviation would leak it back into the prediction, which is exactly the
    mistake this version fixes.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size < 2 or features.shape[0] < classes.size + 2:
        return 0.0
    correct = 0
    for index in range(features.shape[0]):
        train = np.delete(features, index, axis=0)
        train_labels = np.delete(labels, index)
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std[std < 1e-12] = 1.0
        centroids = []
        for name in classes:
            members = train[train_labels == name]
            if members.shape[0] == 0:
                break
            centroids.append(((members - mean) / std).mean(axis=0))
        else:
            scaled = (features[index] - mean) / std
            distances = np.linalg.norm(np.stack(centroids) - scaled, axis=1)
            if classes[int(distances.argmin())] == labels[index]:
                correct += 1
    return correct / features.shape[0]


def _permutation_p(values: np.ndarray, groups: np.ndarray, statistic,
                   rng: np.random.Generator) -> float:
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


def _check(name: str, p_value: float, detail: dict, critical_if: bool) -> dict:
    return {'check': name, 'p_value': round(float(p_value), 6),
            'pass': not critical_if, 'critical': bool(critical_if), 'detail': detail}


def leave_one_candidate_out_accuracy(values: np.ndarray, binary: np.ndarray,
                                     candidate_ids: np.ndarray) -> float:
    """Accuracy of a single-threshold rule fitted without the evaluated candidate.

    A membership-fraction threshold is only a real shortcut if it transfers across trees.
    Fitting on other candidates and testing on the held-out candidate is the honest
    version of that question.
    """
    values = np.asarray(values, dtype=float)
    binary = np.asarray(binary, dtype=int)
    candidate_ids = np.asarray(candidate_ids)
    correct = 0
    total = 0
    for candidate in np.unique(candidate_ids):
        test = candidate_ids == candidate
        train = ~test
        if len(np.unique(binary[train])) < 2 or not test.any():
            continue
        split = best_binary_split(values[train], binary[train])
        threshold, direction = split['threshold'], split['direction']
        prediction = (direction * values[test] >= direction * threshold).astype(int)
        correct += int((prediction == binary[test]).sum())
        total += int(test.sum())
    return correct / total if total else 0.0


def cross_partition_threshold_accuracy(values: np.ndarray, binary: np.ndarray,
                                       partitions: np.ndarray) -> dict:
    """Fit a membership-only threshold on one partition, test on the other.

    This is the deployment-shaped question: a shortcut that only works because the
    candidate's own baseline membership fraction is known is not a shortcut a real
    operator could use on an unseen tree.
    """
    values = np.asarray(values, dtype=float)
    binary = np.asarray(binary, dtype=int)
    partitions = np.asarray(partitions)
    unique = sorted(set(partitions.tolist()))
    if len(unique) < 2:
        return {'applicable': False, 'reason': 'single partition'}
    result = {'applicable': True}
    scores = []
    for fit_on in unique:
        test_on = [p for p in unique if p != fit_on][0]
        fit = partitions == fit_on
        test = partitions == test_on
        if len(np.unique(binary[fit])) < 2 or not test.any():
            continue
        split = best_binary_split(values[fit], binary[fit])
        threshold, direction = split['threshold'], split['direction']
        prediction = (direction * values[test] >= direction * threshold).astype(int)
        truth = binary[test]
        sensitivity = (float((prediction[truth == 1] == 1).mean())
                       if (truth == 1).any() else 0.0)
        specificity = (float((prediction[truth == 0] == 0).mean())
                       if (truth == 0).any() else 0.0)
        balanced = 0.5 * (sensitivity + specificity)
        scores.append(balanced)
        result[f'fit_{fit_on}_test_{test_on}_balanced_accuracy'] = round(balanced, 6)
    result['mean_balanced_accuracy'] = round(float(np.mean(scores)), 6) if scores else 0.0
    return result


def loo_threshold_accuracy(values: np.ndarray, binary: np.ndarray) -> float:
    """Leave-one-out balanced accuracy of a single-threshold rule.

    The threshold is chosen on the other samples only, then applied to the held-out one,
    so an in-sample threshold that merely carves off one outlier cannot inflate the
    score. Folds whose training split contains only one class are skipped: a threshold
    learned from one class is not a rule.
    """
    values = np.asarray(values, dtype=float)
    binary = np.asarray(binary, dtype=int)
    if values.size < 3 or len(np.unique(binary)) < 2:
        return 0.0
    tp = fn = tn = fp = 0
    for index in range(values.size):
        train = np.delete(values, index)
        train_labels = np.delete(binary, index)
        if len(np.unique(train_labels)) < 2:
            continue
        split = best_binary_split(train, train_labels)
        threshold, direction = split['threshold'], split['direction']
        prediction = int(direction * values[index] >= direction * threshold)
        if binary[index] == 1:
            tp += prediction == 1
            fn += prediction == 0
        else:
            tn += prediction == 0
            fp += prediction == 1
    if (tp + fn) == 0 or (tn + fp) == 0:
        return 0.0
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return 0.5 * (sensitivity + specificity)


def best_binary_split(values: np.ndarray, binary: np.ndarray) -> dict:
    """Best single-threshold **balanced accuracy** against a binary target.

    Balanced accuracy is used because the diagnostic is deliberately class-imbalanced
    (6 GOOD vs 36 defect examples per candidate); raw accuracy would report the
    majority-class rate and look like a shortcut when nothing was learned.

    Computed by sorting once and accumulating prefix counts, so it is O(n log n) rather
    than O(n^2) over every candidate threshold.
    """
    values = np.asarray(values, dtype=float)
    binary = np.asarray(binary, dtype=int)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    sorted_binary = binary[order]
    total = sorted_binary.size
    positives = int(sorted_binary.sum())
    negatives = total - positives
    if total == 0 or positives == 0 or negatives == 0:
        # A single-class target has no meaningful balanced accuracy; the constant
        # predictor is trivially correct on the class that exists.
        return {'best_balanced_accuracy': 1.0 if total else 0.0,
                'best_accuracy': 1.0 if total else 0.0,
                'threshold': 0.0, 'direction': 1, 'cut_index': 0}

    # Cut index `k` splits the sorted values: the first `k` are predicted negative and
    # the rest positive for direction +1 (>=), and the reverse for direction -1 (<=).
    # Both constant predictors are reachable: k=0 predicts all positive, k=total all
    # negative, so a constant feature can only reach 50% balanced accuracy.
    below_pos = np.concatenate([[0], np.cumsum(sorted_binary)])
    below_neg = np.concatenate([[0], np.cumsum(1 - sorted_binary)])
    true_pos_ge = positives - below_pos
    true_neg_ge = below_neg
    balanced_ge = 0.5 * (true_pos_ge / positives + true_neg_ge / negatives)
    balanced_le = 0.5 * (below_pos / positives
                         + (negatives - below_neg) / negatives)
    best_ge = int(np.argmax(balanced_ge))
    best_le = int(np.argmax(balanced_le))
    if balanced_ge[best_ge] >= balanced_le[best_le]:
        index, score, direction = best_ge, float(balanced_ge[best_ge]), 1
    else:
        index, score, direction = best_le, float(balanced_le[best_le]), -1
    if direction == 1:
        threshold = (float(sorted_values[-1]) if index >= total
                     else float(sorted_values[index]))
    else:
        threshold = (float(sorted_values[0]) if index <= 0
                     else float(sorted_values[index - 1]))
    raw = (float(true_pos_ge[index] + true_neg_ge[index]) / total
           if direction == 1
           else float(below_pos[index] + negatives - below_neg[index]) / total)
    return {'best_balanced_accuracy': round(score, 6),
            'best_accuracy': round(raw, 6),
            'threshold': round(threshold, 6), 'direction': direction,
            'cut_index': int(index)}


def _controls(feature_matrix: np.ndarray, labels: np.ndarray,
              rng: np.random.Generator) -> dict:
    """Mandatory negative controls for the leak detector itself.

    Two probes are reported for each control: the honest LOO nearest-centroid accuracy
    and the decisive per-feature class separation. The random control uses a matrix of
    the same shape as the real features, so it exercises the detector at the real sample
    size. The implanted control carries a perfectly class-coded scalar.
    """
    classes, inverse = np.unique(labels, return_inverse=True)
    chance = 1.0 / classes.size
    random_features = rng.normal(size=np.asarray(feature_matrix).shape)
    random_accuracy = loo_nearest_centroid(random_features, labels)
    random_separation = max_feature_class_separation(random_features, labels)
    implanted = np.concatenate(
        [np.asarray(feature_matrix, dtype=float), inverse[:, None].astype(float)], axis=1)
    implanted_accuracy = loo_nearest_centroid(implanted, labels)
    implanted_separation = max_feature_class_separation(implanted, labels)
    random_ok = random_separation <= RANDOM_CONTROL_MAX_SEPARATION
    implanted_ok = implanted_separation >= IMPLANTED_CONTROL_MIN_SEPARATION
    return {
        'chance_accuracy': round(float(chance), 6),
        'random_control_accuracy': round(float(random_accuracy), 6),
        'random_control_separation': round(float(random_separation), 6),
        'random_control_max_separation': RANDOM_CONTROL_MAX_SEPARATION,
        'random_control_pass': bool(random_ok),
        'implanted_shortcut_accuracy': round(float(implanted_accuracy), 6),
        'implanted_shortcut_separation': round(float(implanted_separation), 6),
        'implanted_shortcut_min_separation': IMPLANTED_CONTROL_MIN_SEPARATION,
        'implanted_shortcut_pass': bool(implanted_ok),
        'detector_verified': bool(random_ok and implanted_ok),
    }


def analyse(records: list[ExampleRecord], seed: int = 20260909) -> dict:
    if not records:
        raise ValueError('no example records to analyse')
    rng = np.random.default_rng(seed)
    classes = np.array([r.example_class for r in records])
    sampled_points = np.array([r.sampled_points for r in records], dtype=float)
    coverage = np.array([r.spatial_coverage for r in records], dtype=float)
    membership_sampled = np.array([r.membership_sampled for r in records], dtype=float)
    feature_stats = np.array(
        [list(r.feature_mean) + list(r.feature_std) for r in records], dtype=float)

    controls = _controls(feature_stats, classes, rng)
    checks: list[dict] = []

    # 1. point count fingerprint. The scene for one candidate is the same object across
    #    all its classes, so its size must not vary with the class; across candidates the
    #    size legitimately differs because each candidate has its own real context.
    scene_sizes: dict[str, set[int]] = {}
    for record in records:
        scene_sizes.setdefault(record.candidate_id, set()).add(record.point_count)
    varying = {cid: sorted(sizes) for cid, sizes in scene_sizes.items()
               if len(sizes) != 1}
    checks.append(_check(
        'point_count_fingerprint_source',
        1.0 if not varying else 0.0,
        {'candidates': len(scene_sizes),
         'candidates_with_class_dependent_scene_size': varying,
         'note': 'one candidate = one shared scene for every class; only the membership '
                 'channel changes'},
        critical_if=bool(varying)))
    unique_sampled = np.unique(sampled_points)
    checks.append(_check(
        'point_count_fingerprint_sampled',
        1.0 if unique_sampled.size == 1 else 0.0,
        {'distinct_sampled_sizes': [int(v) for v in unique_sampled]},
        critical_if=unique_sampled.size != 1))

    # 2. fixed fraction fingerprint over the defect classes.
    defect_rows = [r for r in records if r.example_class != 'GOOD']
    drawn = np.array([r.drawn_fraction for r in defect_rows], dtype=float)
    defect_classes = np.array([r.example_class for r in defect_rows])
    p_fraction = _permutation_p(drawn, defect_classes, _range_ratio, rng)
    checks.append(_check(
        'fixed_fraction_fingerprint', p_fraction,
        {'fraction_min': round(float(drawn.min()), 6),
         'fraction_max': round(float(drawn.max()), 6),
         'fraction_mean': round(float(drawn.mean()), 6),
         'per_class_drawn_fraction': {
             c: round(float(drawn[defect_classes == c].mean()), 6)
             for c in np.unique(defect_classes)},
         'excluded_classes': ['GOOD']},
        critical_if=p_fraction < ALPHA))

    # 3. fixed-axis fingerprint.
    axis_rows = [r for r in records if r.corruption_axis is not None]
    if len(axis_rows) >= 8:
        axis = np.array([r.corruption_axis for r in axis_rows], dtype=float)
        axis_class = np.array([r.example_class for r in axis_rows])
        dominant = np.abs(axis).argmax(axis=1)
        p_axis = 1.0
        unique_axis_classes = np.unique(axis_class)
        if unique_axis_classes.size >= 2:
            contingency = np.zeros((unique_axis_classes.size, 3))
            index_of = {c: i for i, c in enumerate(unique_axis_classes)}
            for c, d in zip(axis_class, dominant, strict=True):
                contingency[index_of[c], d] += 1
            expected = (contingency.sum(axis=1, keepdims=True)
                        * contingency.sum(axis=0, keepdims=True) / contingency.sum())
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2 = float(np.nansum((contingency - expected) ** 2 / expected))
            p_axis = float(np.exp(-chi2 / 2.0)) if chi2 > 0 else 1.0
        checks.append(_check(
            'fixed_axis_fingerprint', p_axis,
            {'dominant_axis_histogram': {str(i): int((dominant == i).sum())
                                         for i in range(3)},
             'classes_with_axis': list(unique_axis_classes)},
            critical_if=p_axis < ALPHA))
    else:
        checks.append(_check('fixed_axis_fingerprint', 1.0,
                             {'applicable': False}, critical_if=False))

    # 4. class-specific sampling fingerprint.
    #
    #    The generator's intended membership shift is class-specific — that IS the defect
    #    signal. The artefact question is whether the sampler preserves the scene's own
    #    membership fraction, which a label-agnostic sampler must do for every class. The
    #    statistic is therefore sampled-minus-scene, not sampled-minus-GOOD-baseline.
    membership_gap = membership_sampled - np.array(
        [r.membership_fraction for r in records], dtype=float)
    p_sampling = _permutation_p(membership_gap, classes, _range_ratio, rng)
    intended = np.array([r.intended_membership_shift for r in records], dtype=float)
    checks.append(_check(
        'class_specific_sampling_fingerprint', p_sampling,
        {'mean_sampler_gap': round(float(membership_gap.mean()), 6),
         'max_abs_sampler_gap': round(float(np.abs(membership_gap).max()), 6),
         'per_class_mean_sampler_gap': {
             c: round(float(membership_gap[classes == c].mean()), 6)
             for c in np.unique(classes)},
         'per_class_mean_intended_shift': {
             c: round(float(intended[classes == c].mean()), 6)
             for c in np.unique(classes)},
         'mean_spatial_coverage': round(float(coverage.mean()), 6),
         'note': 'the sampler must reproduce the scene membership fraction for every '
                 'class; the intended shift is the generator signal and is reported '
                 'separately'},
        critical_if=p_sampling < ALPHA))

    # 5. metadata leakage: the exposed tensor must not predict the class.
    accuracy = loo_nearest_centroid(feature_stats, classes)
    chance = 1.0 / np.unique(classes).size
    p_stats = _permutation_p(feature_stats[:, 0], classes, _range_ratio, rng)
    checks.append(_check(
        'metadata_leakage_tensor_stats', p_stats,
        {'loo_nearest_centroid_accuracy': round(float(accuracy), 6),
         'chance_accuracy': round(float(chance), 6),
         'feature_channels': list(sampler.FEATURES),
         'feature_channel_count': sampler.FEATURE_COUNT,
         'detector_verified_by_controls': controls['detector_verified']},
        critical_if=p_stats < ALPHA))

    # 5b. audit-only shortcut scan. The scalars that *define* the label (how many points
    #     changed membership) are excluded, because "no point changed" is what GOOD
    #     means; including them would only restate the label.
    binary = (classes == 'GOOD').astype(int)
    label_defining = {3, 4}  # changed_membership_points, changed_membership_fraction
    shortcut_rows = []
    for index in range(feature_stats.shape[1]):
        split = best_binary_split(feature_stats[:, index], binary)
        split['label_defining'] = index in label_defining
        shortcut_rows.append({'feature': index, **split})
    scored = [r for r in shortcut_rows if not r['label_defining']]
    best_shortcut = max(scored, key=lambda r: r['best_accuracy'])

    # 5c. membership-only shortcut: can the model input's fourth channel alone reveal the
    #     label? This is the shortcut the Supervisor asked to be measured. Accuracy is
    #     leave-one-out, so a threshold that merely carves off one outlier cannot inflate
    #     it. The distribution overlap is reported alongside.
    good_fraction = np.array([r.membership_sampled for r in records
                              if r.example_class == 'GOOD'], dtype=float)
    bad_fraction = np.array([r.membership_sampled for r in records
                             if r.example_class != 'GOOD'], dtype=float)
    membership_values = np.array([r.membership_sampled for r in records], dtype=float)
    zero_counts = np.array([r.sampled_points - int(round(
        r.membership_sampled * r.sampled_points)) for r in records], dtype=float)
    one_counts = np.array([int(round(r.membership_sampled * r.sampled_points))
                           for r in records], dtype=float)
    membership_only = {
        'membership_fraction_loo_balanced_accuracy': round(
            float(loo_threshold_accuracy(membership_values, binary)), 6),
        'membership_fraction_leave_one_candidate_out_balanced_accuracy': round(
            float(leave_one_candidate_out_accuracy(
                membership_values, binary,
                np.array([r.candidate_id for r in records]))), 6),
        'membership_fraction_cross_partition': cross_partition_threshold_accuracy(
            membership_values, binary, np.array([r.partition for r in records])),
        'membership_fraction_in_sample_balanced_accuracy': best_binary_split(
            membership_values, binary)['best_balanced_accuracy'],
        'membership_zero_count_loo_balanced_accuracy': round(
            float(loo_threshold_accuracy(zero_counts, binary)), 6),
        'membership_one_count_loo_balanced_accuracy': round(
            float(loo_threshold_accuracy(one_counts, binary)), 6),
        'good_fraction_range': [round(float(good_fraction.min()), 6),
                                round(float(good_fraction.max()), 6)]
        if good_fraction.size else None,
        'bad_fraction_range': [round(float(bad_fraction.min()), 6),
                               round(float(bad_fraction.max()), 6)]
        if bad_fraction.size else None,
        'majority_class_rate': round(float(max(
            (binary == 1).mean(), (binary == 0).mean())), 6),
        'any_membership_zero': {
            'balanced_accuracy': round(float(0.5 * (
                (membership_values[membership_values < 1.0] < 1.0).mean()
                + (membership_values[membership_values >= 1.0] >= 1.0).mean()
                if (membership_values < 1.0).any()
                and (membership_values >= 1.0).any() else 0.0)), 6),
            'examples_with_a_zero_flag': int((membership_values < 1.0).sum()),
            'examples': int(membership_values.size),
            'note': 'a naive any(membership == 0) rule; every example has one',
        },
    }
    cross_partition = membership_only['membership_fraction_cross_partition']
    cross_score = (cross_partition.get('mean_balanced_accuracy', 0.0)
                   if cross_partition.get('applicable') else 0.0)
    membership_best = max(cross_score,
                          membership_only['membership_zero_count_loo_balanced_accuracy'],
                          membership_only['membership_one_count_loo_balanced_accuracy'])
    checks.append(_check(
        'membership_only_shortcut', 1.0 if membership_best < MEMBERSHIP_SHORTCUT_LIMIT
        else 0.0,
        {'best_membership_only_balanced_accuracy': round(float(membership_best), 6),
         'limit': MEMBERSHIP_SHORTCUT_LIMIT,
         'detail': membership_only,
         'note': 'balanced accuracy is used because the diagnostic is deliberately '
                 'imbalanced (6 GOOD vs 36 defect examples per candidate); GOOD always '
                 'contains membership-0 context, so the presence of a zero flag cannot '
                 'itself identify the class; the headline number fits a threshold on one '
                 'partition and applies it to the other'},
        critical_if=membership_best >= MEMBERSHIP_SHORTCUT_LIMIT))

    per_candidate = {}
    per_severity = {}
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
         'best_single_audit_scalar': best_shortcut,
         'note': 'audit scalars never enter the model; this scan only shows whether the '
                 'generator leaves a trivially separable geometric artefact'},
        critical_if=not (class_sets_identical and severity_sets_identical)))

    critical = [c['check'] for c in checks if c['critical']]
    if not controls['detector_verified']:
        critical.append('fingerprint_detector_control_failed')
    return {
        'fingerprint_version': FINGERPRINT_VERSION,
        'examples': len(records),
        'permutations': PERMUTATIONS,
        'alpha': ALPHA,
        'checks': checks,
        'controls': controls,
        'audit_shortcut_scan': {
            'best_single_scalar': best_shortcut,
            'per_feature': shortcut_rows,
        },
        'critical_fingerprints': critical,
        'GENERATOR_FINGERPRINT': 'NONE_DETECTED' if not critical else 'CRITICAL',
        'CRITICAL_FINGERPRINTS': critical,
    }
