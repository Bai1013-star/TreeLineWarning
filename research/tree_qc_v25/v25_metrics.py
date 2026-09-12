"""Metrics and the GO / NO-GO decision for the V2.5 round.

Every quantity the round asked to be reported is computed here, and the verdict is
derived from the five declared criteria rather than from a judgement call:

* criteria 1-3 (bad proposals not made worse, good proposals not damaged) decide
  between ``DEEP_LEARNING_NO_GO`` and ``DEEP_LEARNING_BORDERLINE``;
* criteria 4-5 (auto-edit reliability and coverage) decide between
  ``DEEP_LEARNING_BORDERLINE`` and ``DEEP_LEARNING_GO``;
* all five are evaluated on the cross-site configuration only, because an in-site
  number is not generalisation evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import v25_config as cfg


def _subset(rows: list[dict], which: str) -> list[dict]:
    if which == 'GOOD':
        return [row for row in rows if row['current_iou'] >= cfg.GOOD_IOU_THRESHOLD]
    if which == 'BAD':
        return [row for row in rows if row['current_iou'] < cfg.GOOD_IOU_THRESHOLD]
    return rows


def _block(rows: list[dict], refined_field: str, delta_field: str) -> dict:
    if not rows:
        return {'rows': 0}
    current = np.array([row['current_iou'] for row in rows])
    refined = np.array([row[refined_field] for row in rows])
    delta = np.array([row[delta_field] for row in rows])
    return {
        'rows': len(rows),
        'current_iou_mean': float(current.mean()),
        'current_iou_median': float(np.median(current)),
        'refined_iou_mean': float(refined.mean()),
        'refined_iou_median': float(np.median(refined)),
        'delta_iou_mean': float(delta.mean()),
        'delta_iou_median': float(np.median(delta)),
        'improved_fraction': float((delta > 1e-9).mean()),
        'unchanged_fraction': float((np.abs(delta) <= 1e-9).mean()),
        'degraded_fraction': float((delta < -1e-9).mean()),
        'improved_gt_threshold': float(
            (delta > cfg.BAD_IMPROVEMENT_IOU).mean()),
        'degraded_gt_threshold': float(
            (delta < -cfg.GOOD_DEGRADATION_IOU).mean()),
    }


def summarise(rows: list[dict], label: str) -> dict:
    """Result B: the delta refiner with no gate."""
    if not rows:
        return {'label': label, 'rows': 0}
    changed = np.array([not row['unchanged'] for row in rows])
    delta = np.array([row['delta_iou'] for row in rows])
    good = np.array([row['current_iou'] >= cfg.GOOD_IOU_THRESHOLD for row in rows])
    out = {
        'label': label, 'rows': len(rows),
        'ALL': _block(rows, 'refined_iou', 'delta_iou'),
        'GOOD': _block(_subset(rows, 'GOOD'), 'refined_iou', 'delta_iou'),
        'BAD': _block(_subset(rows, 'BAD'), 'refined_iou', 'delta_iou'),
        'GOOD_MEAN_DELTA_IOU': float(delta[good].mean()) if good.any() else None,
        'GOOD_DEGRADATION_RATE': float((delta[good] < -cfg.GOOD_DEGRADATION_IOU).mean())
        if good.any() else None,
        'BAD_MEAN_DELTA_IOU': float(delta[~good].mean()) if (~good).any() else None,
        'BAD_IMPROVEMENT_RATE': float((delta[~good] > cfg.BAD_IMPROVEMENT_IOU).mean())
        if (~good).any() else None,
        'EDIT_RATE': float(changed.mean()),
        'NO_OP_RATE': float(1.0 - changed.mean()),
        'edited_success_rate': float((delta[changed] >= cfg.GATE_BENEFIT_IOU).mean())
        if changed.any() else None,
        'edited_rows': int(changed.sum()),
        'change_region_f1_mean': float(np.mean([row['change_region_f1']
                                                for row in rows])),
        'change_region_iou_mean': float(np.mean([row['change_region_iou']
                                                 for row in rows])),
        'by_corruption': _by(rows, 'corruption', 'refined_iou', 'delta_iou'),
        'by_proposal_kind': _by(rows, 'proposal_kind', 'refined_iou', 'delta_iou'),
    }
    out.update(_point_metrics(rows))
    return out


def summarise_gated(rows: list[dict], label: str) -> dict:
    """Result C: the delta refiner behind the conservative gate."""
    if not rows:
        return {'label': label, 'rows': 0}
    auto = np.array([bool(row['auto_edit']) for row in rows])
    delta = np.array([row['gated_delta_iou'] for row in rows])
    good = np.array([row['current_iou'] >= cfg.GOOD_IOU_THRESHOLD for row in rows])
    out = {
        'label': label, 'rows': len(rows),
        'ALL': _block(rows, 'gated_refined_iou', 'gated_delta_iou'),
        'GOOD': _block(_subset(rows, 'GOOD'), 'gated_refined_iou', 'gated_delta_iou'),
        'BAD': _block(_subset(rows, 'BAD'), 'gated_refined_iou', 'gated_delta_iou'),
        'AUTO_EDIT_COVERAGE': float(auto.mean()),
        'AUTO_EDIT_COUNT': int(auto.sum()),
        'AUTO_EDIT_SUCCESS_RATE': float(
            (delta[auto] >= cfg.GATE_BENEFIT_IOU).mean()) if auto.any() else None,
        'AUTO_EDIT_HARM_RATE': float(
            (delta[auto] < -cfg.GOOD_DEGRADATION_IOU).mean()) if auto.any() else None,
        'AUTO_EDIT_MEAN_DELTA_IOU': float(delta[auto].mean()) if auto.any() else None,
        'NO_OP_RATE': float(1.0 - auto.mean()),
        'GOOD_MEAN_DELTA_IOU': float(delta[good].mean()) if good.any() else None,
        'GOOD_DEGRADATION_RATE': float(
            (delta[good] < -cfg.GOOD_DEGRADATION_IOU).mean()) if good.any() else None,
        'BAD_MEAN_DELTA_IOU': float(delta[~good].mean()) if (~good).any() else None,
        'BAD_IMPROVEMENT_RATE': float(
            (delta[~good] > cfg.BAD_IMPROVEMENT_IOU).mean()) if (~good).any() else None,
        'GOOD_AUTO_EDIT_COVERAGE': float(auto[good].mean()) if good.any() else None,
        'BAD_AUTO_EDIT_COVERAGE': float(auto[~good].mean()) if (~good).any() else None,
        'by_corruption': _by_gated(rows, 'corruption'),
        'by_proposal_kind': _by_gated(rows, 'proposal_kind'),
    }
    out.update(_point_metrics(rows, auto=auto))
    return out


def _point_metrics(rows: list[dict], auto: np.ndarray | None = None) -> dict:
    """ADD / REMOVE precision, recall and F1, optionally over auto-edited rows."""
    if auto is not None:
        rows = [row for row, take in zip(rows, auto, strict=True) if take]
    if not rows:
        return {'remove_precision': None, 'remove_recall': None, 'remove_f1': None,
                'add_precision': None, 'add_recall': None, 'add_f1': None}
    def scores(prefix: str) -> dict:
        tp = sum(row[f'{prefix}_true_positive'] for row in rows)
        fp = sum(row[f'{prefix}_false_positive'] for row in rows)
        fn = sum(row[f'{prefix}_false_negative'] for row in rows)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        title = prefix.upper()
        return {f'{title}_PRECISION': precision, f'{title}_RECALL': recall,
                f'{title}_F1': f1, f'{title}_TP': tp, f'{title}_FP': fp,
                f'{title}_FN': fn}
    return {**scores('remove'), **scores('add')}


def _by(rows: list[dict], key: str, refined_field: str, delta_field: str) -> dict:
    out = {}
    for value in sorted({str(row.get(key)) for row in rows}):
        group = [row for row in rows if str(row.get(key)) == value]
        delta = np.array([row[delta_field] for row in group])
        out[value] = {
            'rows': len(group),
            'current_iou_mean': float(np.mean([row['current_iou'] for row in group])),
            'refined_iou_mean': float(np.mean([row[refined_field] for row in group])),
            'delta_iou_mean': float(delta.mean()),
            'delta_iou_median': float(np.median(delta)),
            'improved_fraction': float((delta > 1e-9).mean()),
            'no_op_rate': float(np.mean([row['unchanged'] for row in group])),
        }
    return out


def _by_gated(rows: list[dict], key: str) -> dict:
    out = {}
    for value in sorted({str(row.get(key)) for row in rows}):
        group = [row for row in rows if str(row.get(key)) == value]
        auto = np.array([bool(row['auto_edit']) for row in group])
        delta = np.array([row['gated_delta_iou'] for row in group])
        out[value] = {
            'rows': len(group),
            'current_iou_mean': float(np.mean([row['current_iou'] for row in group])),
            'gated_refined_iou_mean': float(
                np.mean([row['gated_refined_iou'] for row in group])),
            'gated_delta_iou_mean': float(delta.mean()),
            'auto_edit_coverage': float(auto.mean()),
            'auto_edit_success_rate': float(
                (delta[auto] >= cfg.GATE_BENEFIT_IOU).mean()) if auto.any() else None,
        }
    return out


def evaluate_criteria(cross_site_gated: dict) -> dict:
    """Apply the five declared criteria to the cross-site gated result."""
    observed = {
        'CROSS_SITE_BAD_MEAN_DELTA_IOU': cross_site_gated.get('BAD_MEAN_DELTA_IOU'),
        'CROSS_SITE_GOOD_MEAN_DELTA_IOU': cross_site_gated.get('GOOD_MEAN_DELTA_IOU'),
        'CROSS_SITE_GOOD_DEGRADATION_RATE': cross_site_gated.get('GOOD_DEGRADATION_RATE'),
        'CROSS_SITE_AUTO_EDIT_SUCCESS_RATE': cross_site_gated.get('AUTO_EDIT_SUCCESS_RATE'),
        'CROSS_SITE_AUTO_EDIT_COVERAGE': cross_site_gated.get('AUTO_EDIT_COVERAGE'),
    }
    out = {}
    for name, rule in cfg.CRITERIA.items():
        value = observed.get(name)
        if value is None:
            passed = False
        elif rule['op'] == '>':
            passed = bool(value > rule['value'])
        elif rule['op'] == '>=':
            passed = bool(value >= rule['value'])
        else:
            passed = bool(value <= rule['value'])
        out[name] = {'observed': value, 'rule': f"{rule['op']} {rule['value']}",
                     'passed': passed}
    return out


def _fold_headline(fold: dict) -> dict:
    """The handful of numbers the final summary quotes for one configuration."""
    gated = fold['test']['gated']
    return {
        'BAD_MEAN_DELTA_IOU': gated.get('BAD_MEAN_DELTA_IOU'),
        'BAD_IMPROVEMENT_RATE': gated.get('BAD_IMPROVEMENT_RATE'),
        'GOOD_MEAN_DELTA_IOU': gated.get('GOOD_MEAN_DELTA_IOU'),
        'GOOD_DEGRADATION_RATE': gated.get('GOOD_DEGRADATION_RATE'),
        'AUTO_EDIT_COVERAGE': gated.get('AUTO_EDIT_COVERAGE'),
        'AUTO_EDIT_SUCCESS_RATE': gated.get('AUTO_EDIT_SUCCESS_RATE'),
        'delta_iou_mean_gated': gated['ALL']['delta_iou_mean'],
        'delta_iou_mean_ungated': fold['test']['ungated']['ALL']['delta_iou_mean'],
        'no_op_rate_gated': gated.get('NO_OP_RATE'),
        'no_op_rate_ungated': fold['test']['ungated'].get('NO_OP_RATE'),
    }


def verdict(payload: dict) -> dict:
    """The round's final decision, derived from the criteria and nothing else."""
    folds = {fold['configuration']: fold for fold in payload.get('folds', [])}
    cross = folds.get('cross_site_test_wytham')
    in_site = folds.get('in_site_test_litchfield')
    out = {'evaluated_on': 'cross_site_test_wytham',
           'criteria': {}, 'core_passed': None, 'edit_passed': None}
    if cross is None:
        out.update({'verdict': 'DEEP_LEARNING_NO_GO',
                    'reason': 'the cross-site configuration did not run',
                    'TREE_QC_DEEP_LEARNING_ROUTE': 'STOPPED'})
        return out
    criteria = evaluate_criteria(cross['test']['gated'])
    out['criteria'] = criteria
    core_ok = all(criteria[name]['passed'] for name in cfg.CORE_CRITERIA)
    edit_ok = all(criteria[name]['passed'] for name in cfg.EDIT_CRITERIA)
    out['core_passed'] = core_ok
    out['edit_passed'] = edit_ok
    if not core_ok:
        out['verdict'] = 'DEEP_LEARNING_NO_GO'
    elif edit_ok:
        out['verdict'] = 'DEEP_LEARNING_GO'
    else:
        out['verdict'] = 'DEEP_LEARNING_BORDERLINE'
    out['TREE_QC_DEEP_LEARNING_ROUTE'] = (
        'STOPPED' if out['verdict'] == 'DEEP_LEARNING_NO_GO' else 'CONTINUE')
    out['cross_site'] = _fold_headline(cross)
    if in_site is not None:
        out['in_site'] = _fold_headline(in_site)
    baseline = payload.get('result_a_v2_frozen_full_mask', {})
    cross_baseline = (baseline.get('configurations', {})
                      .get('cross_site_test_wytham', {}).get('models', {}))
    if cross_baseline:
        best = max(cross_baseline.values(), key=lambda entry: entry['delta_iou_mean'])
        out['result_a_v2_baseline_cross_site_delta'] = best['delta_iou_mean']
        out['result_a_v2_baseline_name'] = best['name']
    if out['verdict'] == 'DEEP_LEARNING_NO_GO':
        out['recommended_next_route'] = [
            'CROWN_STEM_HYBRID as the primary tree algorithm',
            'deterministic geometry / structural QC instead of a learned refiner',
            'human review for the cases those two cannot settle',
        ]
    return out


def write_rows(path: Path, test_rows, test_gated, validation_rows,
               validation_gated) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'test_rows': test_rows, 'test_gated': test_gated,
                                'validation_rows': validation_rows,
                                'validation_gated': validation_gated},
                               indent=2, default=str), encoding='utf-8')
