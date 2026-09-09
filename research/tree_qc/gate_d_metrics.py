"""Evaluation metrics and frozen threshold selection for Tree QC.

Label semantics: index 0 = ``PASS`` (the observation is a usable single tree),
index 1 = ``REVIEW_REQUIRED`` (a segmentation defect). The model emits a defect
score; a frozen threshold maps it to a decision.

Threshold rules, fixed before any model result exists:

* the threshold is chosen on the **training-site validation partition only**;
* held-out trees can never influence the threshold, a hyper-parameter or a model
  choice — :func:`select_threshold` refuses a held-out partition by construction;
* the selection objective is bad recall subject to a bounded good false-review rate,
  so a model cannot buy recall by flagging everything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS = 0
REVIEW_REQUIRED = 1
LABEL_NAMES = ('PASS', 'REVIEW_REQUIRED')

# Future three-way decisions. Gate E round 1 ships binary score + frozen threshold;
# this mapping exists so the interface does not have to change later.
DECISION_PASS = 'PASS'
DECISION_REVIEW = 'REVIEW_REQUIRED'
DECISION_UNCERTAIN = 'UNCERTAIN'
DECISIONS = (DECISION_PASS, DECISION_REVIEW, DECISION_UNCERTAIN)

MODEL_SELECTION_PARTITIONS = ('train', 'validation')
DEFAULT_GOOD_FALSE_REVIEW_BUDGET = 0.25
THRESHOLD_GRID = tuple(round(v, 4) for v in np.linspace(0.0, 1.0, 201))


class ThresholdError(RuntimeError):
    pass


@dataclass(frozen=True)
class Confusion:
    true_pass_pred_pass: int
    true_pass_pred_review: int
    true_review_pred_pass: int
    true_review_pred_review: int

    def as_row(self) -> dict:
        return {
            'tn_pass_pass': self.true_pass_pred_pass,
            'fp_pass_as_review': self.true_pass_pred_review,
            'fn_review_as_pass': self.true_review_pred_pass,
            'tp_review_review': self.true_review_pred_review,
        }


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Confusion:
    labels = np.asarray(labels).astype(np.int64).ravel()
    predictions = (np.asarray(scores, dtype=np.float64).ravel() >= threshold).astype(np.int64)
    if labels.shape != predictions.shape:
        raise ThresholdError('labels and scores must have the same shape')
    return Confusion(
        true_pass_pred_pass=int(((labels == PASS) & (predictions == PASS)).sum()),
        true_pass_pred_review=int(((labels == PASS) & (predictions == REVIEW_REQUIRED)).sum()),
        true_review_pred_pass=int(((labels == REVIEW_REQUIRED) & (predictions == PASS)).sum()),
        true_review_pred_review=int(
            ((labels == REVIEW_REQUIRED) & (predictions == REVIEW_REQUIRED)).sum()))


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """The five metrics named by the task, plus supporting rates."""
    matrix = confusion(labels, scores, threshold)
    row = matrix.as_row()
    good_total = row['tn_pass_pass'] + row['fp_pass_as_review']
    bad_total = row['fn_review_as_pass'] + row['tp_review_review']
    total = good_total + bad_total
    bad_recall = (row['tp_review_review'] / bad_total) if bad_total else 0.0
    good_specificity = (row['tn_pass_pass'] / good_total) if good_total else 0.0
    good_false_review = (row['fp_pass_as_review'] / good_total) if good_total else 0.0
    bad_false_pass = (row['fn_review_as_pass'] / bad_total) if bad_total else 0.0
    return {
        'threshold': round(float(threshold), 6),
        'examples': int(total),
        'good_examples': int(good_total),
        'bad_examples': int(bad_total),
        'BAD_RECALL': round(float(bad_recall), 6),
        'BAD_FALSE_PASS': round(float(bad_false_pass), 6),
        'GOOD_FALSE_REVIEW': round(float(good_false_review), 6),
        'GOOD_SPECIFICITY': round(float(good_specificity), 6),
        'BALANCED_ACCURACY': round(float((bad_recall + good_specificity) / 2.0), 6),
        'ACCURACY': round(float((row['tn_pass_pass'] + row['tp_review_review'])
                                / total), 6) if total else 0.0,
        'CONFUSION_MATRIX': row,
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray, partition: str,
                     good_false_review_budget: float = DEFAULT_GOOD_FALSE_REVIEW_BUDGET,
                     grid=THRESHOLD_GRID) -> dict:
    """Choose the frozen operating threshold on model-selection data only."""
    if partition not in MODEL_SELECTION_PARTITIONS:
        raise ThresholdError(
            f'threshold selection is forbidden on partition {partition!r}')
    labels = np.asarray(labels).astype(np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape != scores.shape or labels.size == 0:
        raise ThresholdError('labels and scores must be non-empty and aligned')
    if not np.isfinite(scores).all():
        raise ThresholdError('scores contain non-finite values')

    candidates = []
    for threshold in grid:
        row = metrics(labels, scores, float(threshold))
        if row['GOOD_FALSE_REVIEW'] <= good_false_review_budget + 1e-12:
            candidates.append(row)
    if not candidates:
        # No threshold satisfies the budget: fall back to the least bad one and say so.
        scored = [metrics(labels, scores, float(t)) for t in grid]
        best = min(scored, key=lambda r: (r['GOOD_FALSE_REVIEW'], -r['BAD_RECALL']))
        best['budget_satisfied'] = False
        best['partition'] = partition
        best['good_false_review_budget'] = good_false_review_budget
        best['selection_rule'] = 'fallback_min_good_false_review'
        return best
    best = max(candidates, key=lambda r: (r['BAD_RECALL'], r['BALANCED_ACCURACY'],
                                          -r['GOOD_FALSE_REVIEW']))
    best['budget_satisfied'] = True
    best['partition'] = partition
    best['good_false_review_budget'] = good_false_review_budget
    best['selection_rule'] = 'max_bad_recall_within_good_false_review_budget'
    return best


def three_way_decisions(scores: np.ndarray, threshold: float,
                        uncertain_band: float = 0.10) -> np.ndarray:
    """Future PASS / REVIEW_REQUIRED / UNCERTAIN mapping around the frozen threshold."""
    if uncertain_band < 0:
        raise ThresholdError('uncertain_band must be non-negative')
    scores = np.asarray(scores, dtype=np.float64).ravel()
    low, high = threshold - uncertain_band, threshold + uncertain_band
    out = np.empty(scores.shape, dtype=object)
    out[scores >= high] = DECISION_REVIEW
    out[scores <= low] = DECISION_PASS
    out[(scores > low) & (scores < high)] = DECISION_UNCERTAIN
    return out


def freeze_threshold(selected: dict, dataset_digest: str, checkpoint_digest: str,
                     seed: int) -> dict:
    """Immutable record of the operating point, for Gate E to consume verbatim."""
    if 'threshold' not in selected:
        raise ThresholdError('selected threshold record is incomplete')
    if selected.get('partition') not in MODEL_SELECTION_PARTITIONS:
        raise ThresholdError('a frozen threshold must come from model-selection data')
    return {
        'frozen': True,
        'threshold': float(selected['threshold']),
        'partition': selected['partition'],
        'selection_rule': selected.get('selection_rule'),
        'good_false_review_budget': selected.get('good_false_review_budget'),
        'budget_satisfied': selected.get('budget_satisfied'),
        'validation_metrics': selected,
        'dataset_digest': dataset_digest,
        'checkpoint_digest': checkpoint_digest,
        'seed': int(seed),
        'held_out_used': False,
    }
