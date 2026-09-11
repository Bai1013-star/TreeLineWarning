"""Scoring, per-defect breakdown and robustness evaluation for the Public Pilot.

Two rules shape this module:

1. **Metrics come from the frozen Gate D metric core.** ``BAD_RECALL``,
   ``BAD_FALSE_PASS``, ``GOOD_FALSE_REVIEW``, ``BALANCED_ACCURACY`` and the confusion
   matrix are computed by ``gate_d_metrics.metrics`` and never re-derived here. The only
   metric this layer adds is ``F1_REVIEW``, which the reporting layer is explicitly allowed
   to compute; nothing here touches the frozen metric core.
2. **Threshold selection is impossible on held-out data.** :func:`score_partition` takes a
   *frozen* threshold as an argument. It has no selection path, and
   ``gate_d_metrics.select_threshold`` independently refuses any partition outside
   ``train``/``validation``. So a held-out score cannot leak back into an operating point
   even by mistake.

The three-way view (``PASS`` / ``UNCERTAIN`` / ``REVIEW_REQUIRED``) is reported alongside
the binary decision but never replaces it: the binary metrics are always computed from the
frozen binary threshold, and an uncertain outcome is never counted as a detected defect.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import efg_gates as gates
import gate_d_metrics as metrics_module
import numpy as np

EVALUATOR_VERSION = 'EFG_EVALUATOR_V1'
SEVERITIES = gates.SEVERITIES
PRIMARY_SEVERITIES = gates.PRIMARY_SEVERITIES
DEFECT_CLASSES = gates.DEFECT_CLASSES
GOOD_CLASS = gates.GOOD_CLASS
UNCERTAIN_BAND = gates.UNCERTAIN_BAND
PRIMARY_GATE = gates.PRIMARY_GATE
LEARNABILITY_GATE = gates.LEARNABILITY_GATE
MODEL_STABILITY_MAX_SPREAD = gates.MODEL_STABILITY_MAX_SPREAD
SAMPLING_MAX_RECALL_DROP = gates.SAMPLING_MAX_RECALL_DROP


class EvaluationError(RuntimeError):
    pass


# --- scoring ---------------------------------------------------------------

def score_dataset(model, dataset, device, batch_size: int = 8,
                  strict_exclusion_rate: float = 0.05) -> dict:
    """Run the frozen network over every example and return scores and labels.

    ``dataset`` is any object exposing ``__len__`` and ``__getitem__ -> item`` with
    ``.features``, ``.label``, ``.key`` and ``.provenance``. Both the frozen
    ``GateDDataset`` (through the EFG source wrapper) and the held-out adapter satisfy that
    shape.

    A declared spec whose defect the candidate's geometry cannot support raises
    ``NotApplicableExample``; such an example is **removed from the metric population** and
    counted, never substituted. If more than ``strict_exclusion_rate`` of the partition is
    excluded, the run stops rather than reporting a metric over a silently shrunken
    population.
    """
    import torch
    from efg_dataset import NotApplicableExample

    model.eval()
    keys: list[str] = []
    classes: list[str] = []
    severities: list[str] = []
    labels: list[int] = []
    scores: list[np.ndarray] = []
    excluded: list[dict] = []
    started = time.time()
    total = len(dataset)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch_items = []
            for index in range(start, stop):
                try:
                    batch_items.append(dataset[index])
                except NotApplicableExample as exc:
                    excluded.append({'index': index, 'key': exc.key,
                                     'reason': exc.reason, 'detail': exc.detail})
            if not batch_items:
                continue
            features = np.stack([item.features for item in batch_items], axis=0)
            tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
            tensor = tensor.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                logits = model(tensor)
            if not torch.isfinite(logits).all():
                raise EvaluationError(
                    f'non-finite logits at examples {start}..{stop - 1}')
            probability = torch.softmax(logits.float(), dim=1)[:, 1]
            scores.append(probability.detach().cpu().numpy())
            for item in batch_items:
                keys.append(item.key)
                labels.append(int(item.label))
                classes.append(item.provenance['class'])
                severities.append(item.provenance['severity'])
    if not labels:
        raise EvaluationError('empty evaluation set')
    exclusion_rate = len(excluded) / max(1, total)
    if exclusion_rate > strict_exclusion_rate:
        raise EvaluationError(
            f'{len(excluded)} of {total} declared examples '
            f'({exclusion_rate:.4%}) are not applicable, above the '
            f'{strict_exclusion_rate:.2%} limit; refusing to report metrics over a '
            f'shrunken population')
    reasons: dict[str, int] = {}
    for entry in excluded:
        reasons[entry['reason']] = reasons.get(entry['reason'], 0) + 1
    return {
        'evaluator_version': EVALUATOR_VERSION,
        'keys': keys,
        'classes': classes,
        'severities': severities,
        'labels': np.asarray(labels, dtype=np.int64),
        'scores': np.concatenate(scores).astype(np.float64),
        'examples': len(labels),
        'declared_examples': int(total),
        'excluded_examples': len(excluded),
        'excluded_fraction': round(exclusion_rate, 6),
        'excluded_by_reason': dict(sorted(reasons.items())),
        'excluded_detail': excluded[:50],
        'seconds': round(time.time() - started, 3),
        'device': str(device),
        'batch_size': int(batch_size),
    }


# --- derived metrics -------------------------------------------------------

def f1_review(metrics: dict) -> float:
    """F1 of the REVIEW_REQUIRED (positive) class, from the frozen confusion matrix."""
    return gates.f1_review(metrics)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    row = metrics_module.metrics(labels, scores, threshold)
    row['F1_REVIEW'] = gates.f1_review(row)
    return row


def three_way_counts(scores: np.ndarray, threshold: float,
                     band: float = UNCERTAIN_BAND) -> dict:
    decisions = metrics_module.three_way_decisions(scores, threshold, band)
    counts = {'PASS': 0, 'UNCERTAIN': 0, 'REVIEW_REQUIRED': 0}
    for decision in decisions:
        counts[str(decision)] += 1
    total = max(1, int(scores.size))
    return {'uncertain_band': float(band),
            'counts': counts,
            'rates': {k: round(v / total, 6) for k, v in counts.items()},
            'note': 'UNCERTAIN is never counted as a detected defect; the binary primary '
                    'metrics use the frozen binary threshold independently'}


def score_partition(result: dict, threshold: float,
                    uncertain_band: float = UNCERTAIN_BAND) -> dict:
    """Binary + per-defect + three-way metrics for one scored partition.

    ``threshold`` is always supplied by the caller from the freeze. This function cannot
    choose one.
    """
    labels = result['labels']
    scores = result['scores']
    classes = np.asarray(result['classes'])
    severities = np.asarray(result['severities'])
    summary = binary_metrics(labels, scores, threshold)

    primary_mask = np.isin(severities, PRIMARY_SEVERITIES)
    primary = (binary_metrics(labels[primary_mask], scores[primary_mask], threshold)
               if primary_mask.any() else None)

    per_severity = {}
    for severity in SEVERITIES:
        mask = severities == severity
        if mask.any():
            per_severity[severity] = binary_metrics(labels[mask], scores[mask], threshold)

    per_class = {}
    for example_class in (GOOD_CLASS, *DEFECT_CLASSES):
        mask = classes == example_class
        if not mask.any():
            continue
        entry = binary_metrics(labels[mask], scores[mask], threshold)
        entry['defect_class'] = example_class
        entry['severity_breakdown'] = {
            severity: binary_metrics(labels[mask & (severities == severity)],
                                     scores[mask & (severities == severity)], threshold)
            for severity in SEVERITIES if (mask & (severities == severity)).any()}
        per_class[example_class] = entry

    defect_primary = {}
    defect_all = {}
    for example_class in DEFECT_CLASSES:
        mask = classes == example_class
        if mask.any():
            defect_all[example_class] = binary_metrics(labels[mask], scores[mask],
                                                       threshold)
        mask_primary = mask & primary_mask
        if mask_primary.any():
            defect_primary[example_class] = binary_metrics(labels[mask_primary],
                                                           scores[mask_primary], threshold)

    return {
        'overall': summary,
        'primary_moderate_high': primary,
        'per_severity': per_severity,
        'per_class': per_class,
        'per_defect_primary': defect_primary,
        'per_defect_all_severities': defect_all,
        'three_way': three_way_counts(scores, threshold, uncertain_band),
        'partition': result.get('partition'),
        'partition_digest': result.get('partition_digest'),
        'examples': int(labels.size),
        'good_examples': int((labels == 0).sum()),
        'bad_examples': int((labels == 1).sum()),
    }


def empty_metrics() -> dict:
    return {'BAD_RECALL': None, 'BAD_FALSE_PASS': None, 'GOOD_FALSE_REVIEW': None,
            'BALANCED_ACCURACY': None, 'F1_REVIEW': None, 'examples': 0}


def apply_primary_gate(metrics: dict, gate: dict | None = None) -> dict:
    """Evaluate the pre-declared continuation targets. Never relaxes them."""
    gate = dict(gate or PRIMARY_GATE)
    if not metrics or metrics.get('BAD_RECALL') is None:
        return {'evaluated': False, 'reason': 'no metrics', 'gate': gate}
    checks = {
        'BAD_RECALL': metrics['BAD_RECALL'] >= gate['BAD_RECALL_min'],
        'BAD_FALSE_PASS': metrics['BAD_FALSE_PASS'] <= gate['BAD_FALSE_PASS_max'],
        'GOOD_FALSE_REVIEW': metrics['GOOD_FALSE_REVIEW'] <= gate['GOOD_FALSE_REVIEW_max'],
        'BALANCED_ACCURACY': (metrics['BALANCED_ACCURACY']
                              >= gate['BALANCED_ACCURACY_min']),
    }
    return {'evaluated': True, 'gate': gate, 'checks': checks,
            'passed': all(checks.values()),
            'failed': sorted(k for k, ok in checks.items() if not ok),
            'observed': {k: metrics[k] for k in checks}}


def learnability_verdict(validation_metrics: dict,
                         gate: dict | None = None) -> dict:
    """The Phase E1 gate: is the task learnable at all from train/validation alone?"""
    gate = dict(gate or LEARNABILITY_GATE)
    if not validation_metrics or validation_metrics.get('BAD_RECALL') is None:
        return {'evaluated': False, 'reason': 'no validation metrics', 'gate': gate}
    balanced = validation_metrics['BAD_RECALL'] is not None and (
        validation_metrics['BALANCED_ACCURACY'] or 0.0)
    bad_recall = validation_metrics['BAD_RECALL'] or 0.0
    passed = (balanced >= gate['BALANCED_ACCURACY_min']
              and bad_recall >= gate['BAD_RECALL_min'])
    return {'evaluated': True, 'gate': gate,
            'observed': {'BALANCED_ACCURACY': balanced, 'BAD_RECALL': bad_recall},
            'passed': bool(passed),
            'verdict': 'LEARNABLE' if passed else 'PUBLIC_PILOT_NOT_LEARNABLE'}


# --- multi-seed aggregation ------------------------------------------------

def aggregate_seed_metrics(per_seed: dict, metric_names=None) -> dict:
    """mean / min / max / std across seeds. No seed is dropped or favoured."""
    metric_names = list(metric_names or ('BAD_RECALL', 'BAD_FALSE_PASS',
                                         'GOOD_FALSE_REVIEW', 'BALANCED_ACCURACY',
                                         'F1_REVIEW'))
    summary: dict = {'seeds': sorted(int(s) for s in per_seed), 'metrics': {}}
    for name in metric_names:
        values = [float(per_seed[seed][name]) for seed in sorted(per_seed, key=int)
                  if per_seed[seed].get(name) is not None]
        if not values:
            summary['metrics'][name] = {'mean': None, 'min': None, 'max': None,
                                        'std': None, 'n': 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        summary['metrics'][name] = {
            'mean': round(float(array.mean()), 6),
            'min': round(float(array.min()), 6),
            'max': round(float(array.max()), 6),
            'std': round(float(array.std(ddof=0)), 6),
            'n': int(array.size),
            'values': [round(float(v), 6) for v in values],
        }
    recall = summary['metrics']['BAD_RECALL']
    spread = (None if recall['max'] is None else round(recall['max'] - recall['min'], 6))
    summary['bad_recall_spread'] = spread
    summary['stability_limit'] = MODEL_STABILITY_MAX_SPREAD
    summary['MODEL_STABILITY'] = (
        'NOT_EVALUATED' if spread is None
        else ('PASS' if spread <= MODEL_STABILITY_MAX_SPREAD else 'FAIL'))
    return summary


def robustness_summary(primary_recall: float, per_config: dict) -> dict:
    """Compare each robustness configuration's BAD_RECALL against the 4096 primary."""
    rows = {}
    worst = 0.0
    for label, entry in per_config.items():
        recall = entry.get('BAD_RECALL')
        if recall is None:
            rows[label] = {'BAD_RECALL': None, 'drop_pp': None}
            continue
        drop = round(float(primary_recall) - float(recall), 6)
        worst = max(worst, drop)
        rows[label] = {'BAD_RECALL': round(float(recall), 6),
                       'BAD_RECALL_drop': drop,
                       'BAD_RECALL_drop_pp': round(drop * 100.0, 3),
                       'sampling_seed': entry.get('sampling_seed'),
                       'points': entry.get('points'),
                       'examples': entry.get('examples')}
    verdict = 'PASS' if worst <= SAMPLING_MAX_RECALL_DROP else 'FAIL'
    return {'primary_bad_recall': round(float(primary_recall), 6),
            'configurations': rows,
            'max_bad_recall_drop': round(worst, 6),
            'max_bad_recall_drop_pp': round(worst * 100.0, 3),
            'limit_pp': SAMPLING_MAX_RECALL_DROP * 100.0,
            'SAMPLING_ROBUSTNESS': verdict,
            'verdict': verdict if verdict == 'PASS' else 'SAMPLING_TOO_SENSITIVE'}


# --- persisted evaluation records ------------------------------------------

@dataclass
class EvaluationRecord:
    """One scored partition plus the digests that make it reproducible."""

    name: str
    partition: str
    threshold: float
    metrics: dict
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        payload = {'evaluator_version': EVALUATOR_VERSION, 'name': self.name,
                   'partition': self.partition, 'threshold': float(self.threshold),
                   'metrics': self.metrics, **self.extras}
        payload['record_digest'] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(',', ':'),
                       default=str).encode('utf-8')).hexdigest()
        return payload


def write_evaluation_record(directory: Path | str, record: EvaluationRecord) -> Path:
    import efg_state as state_module
    directory = Path(directory)
    path = directory / f'{record.name}.json'
    state_module.atomic_write_json(path, record.as_row(), with_digest=False)
    return path
