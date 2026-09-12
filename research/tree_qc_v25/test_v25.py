"""Tests for the Tree QC V2.5 GO / NO-GO round.

These cover the properties a wrong implementation would break while still
producing plausible numbers, and the properties this round is *about*:

* the delta heads are supervised on disjoint point sets, so neither can answer a
  question it will not be asked;
* zero confidence on both heads reproduces the current mask exactly - the NO-OP
  default is structural, not hoped for;
* the GOOD/BAD sampling reaches its declared share rather than drifting;
* every declared threshold is frozen before the test is scored, and the test path
  reads them from the frozen file instead of choosing them;
* the verdict follows the five declared criteria and nothing else, including the
  case where a gate that refuses everything would otherwise look safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_data as data
import v25_delta_model as model_module
import v25_metrics as metrics
import v25_train as train

# --- the delta task -----------------------------------------------------------

def test_heads_are_supervised_on_disjoint_point_sets():
    index = data.load_library()
    sys.path.insert(0, str(cfg.V2_DIR))
    import v2_refiner_train as v2_train
    split = v2_train.split_by_tree(index, test_site='wytham', seed=cfg.SEED)
    dataset = data.DeltaDataset(index, data.build_examples(index, split.train)[:4],
                                512, cfg.SEED, augment=False)
    for i in range(len(dataset)):
        item = dataset[i]
        current = item['current'] > 0.5
        remove_active = item['remove_target'] >= 0
        add_active = item['add_target'] >= 0
        assert bool((remove_active == current).all()), 'REMOVE must be the mask'
        assert bool((add_active == ~current).all()), 'ADD must be outside the mask'
        assert not bool((remove_active & add_active).any())


def test_zero_confidence_reproduces_the_current_mask_exactly():
    """The NO-OP default must be structural, not a learned tendency."""
    current = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])
    zeros = torch.zeros_like(current)
    applied = model_module.apply_delta(current, zeros, zeros)
    assert bool((applied['refined'] == (current > 0.5)).all())
    assert int(applied['removed'].sum()) == 0
    assert int(applied['added'].sum()) == 0


def test_the_refined_mask_can_only_change_where_the_heads_fire():
    current = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    remove = torch.tensor([[1.0, 0.0, 1.0, 0.0]])   # the third point is not in the mask
    add = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    applied = model_module.apply_delta(current, remove, add)
    assert applied['refined'].tolist() == [[False, True, False, True]]
    # A REMOVE above threshold on an unmasked point cannot delete anything.
    assert not bool(applied['removed'][0, 2])
    # An ADD above threshold on a masked point cannot add anything.
    add_on_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    other = model_module.apply_delta(current, torch.zeros_like(current), add_on_mask)
    assert other['refined'].tolist() == [[True, True, False, False]]


def test_masked_bce_ignores_points_where_the_head_does_not_apply():
    """The logit at an inactive point must not reach the loss.

    Both calls use the same target, so the active set is identical; only the logit
    at the inactive point changes, between a value that would be harmless and one
    that would be badly wrong if it were scored. The loss must not move.
    """
    target = torch.tensor([[1.0, 0.0, -1.0]])
    mild = torch.tensor([[5.0, -5.0, 0.0]])
    extreme = torch.tensor([[5.0, -5.0, 50.0]])
    assert torch.allclose(train.masked_bce(mild, target),
                          train.masked_bce(extreme, target))
    # A row whose points are all inactive contributes nothing at all.
    assert float(train.masked_bce(mild, torch.full((1, 3), -1.0))) == 0.0


def test_masked_bce_honours_the_importance_weights():
    """The loss must be exactly the weighted mean of the per-point losses.

    Checked against hand-computed values rather than against another configuration:
    shifting weight between two points leaves a mean where it is, so comparing two
    weightings would not test anything. The per-point losses are chosen to be very
    different - logit 8 on the true class is confident and cheap, logit 8 on the
    false class is confident and expensive - so a wrong weighting shows up plainly.
    """
    import math
    logits = torch.tensor([[8.0, 8.0]])
    target = torch.tensor([[1.0, 0.0]])
    correct, wrong = math.log1p(math.exp(-8.0)), math.log1p(math.exp(8.0))

    # Weight only the correct point: the mean is its own loss.
    only_correct = train.masked_bce(logits, target, torch.tensor([[2.0, 0.0]]))
    assert abs(float(only_correct) - correct) < 1e-5
    # Weight only the wrong point: the mean is the expensive one.
    only_wrong = train.masked_bce(logits, target, torch.tensor([[0.0, 2.0]]))
    assert abs(float(only_wrong) - wrong) < 1e-4
    # Equal weights: the plain mean of the two, and equal to the unweighted call.
    equal = train.masked_bce(logits, target, torch.ones(1, 2))
    assert abs(float(equal) - (correct + wrong) / 2) < 1e-4
    assert torch.allclose(equal, train.masked_bce(logits, target))
    # A zero weight must be the same as marking the point inactive.
    zeroed = train.masked_bce(logits, target, torch.tensor([[1.0, 0.0]]))
    dropped = train.masked_bce(logits, torch.tensor([[1.0, -1.0]]))
    assert torch.allclose(zeroed, dropped)
    # The two per-point losses really are far apart, so the test has power.
    assert wrong > 100 * correct


def test_masked_bce_is_finite_when_one_class_is_absent():
    logits = torch.tensor([[0.5, -0.5, 0.1]])
    all_negative = torch.tensor([[0.0, 0.0, 0.0]])
    all_positive = torch.tensor([[1.0, 1.0, 1.0]])
    assert torch.isfinite(train.masked_bce(logits, all_negative))
    assert torch.isfinite(train.masked_bce(logits, all_positive))
    empty = torch.full((3,), -1.0)
    assert float(train.masked_bce(logits, empty)) == 0.0


# --- GOOD proposals as explicit training rows ---------------------------------

def _fake_example(key: str, iou: float) -> data.Example:
    return data.Example(key=key, site='s', target_id='t#0', proposal_id=key,
                        proposal_kind='SIMULATED_PROPOSAL', corruption='X',
                        severity='LOW', current_iou=iou)


def test_sampling_reaches_its_declared_good_share():
    """The realised GOOD share must match the target, not drift below it."""
    examples = ([_fake_example(f'good{i}', 0.95) for i in range(100)]
                + [_fake_example(f'bad{i}', 0.40) for i in range(900)])
    for target in (0.5, 0.3, 0.25):
        _order, report = data.sample_training_set(examples, cfg.SEED,
                                                  good_fraction=target,
                                                  max_good_repeat=10)
        assert abs(report.good_fraction - target) <= 0.05, (target, report.good_fraction)
        assert report.good_rows > 0
        assert report.as_dict()['good_rows_are_explicit_training_samples'] is True


def test_sampling_reports_a_shortfall_instead_of_hiding_it():
    """A capped repeat that cannot reach the target must say so."""
    examples = ([_fake_example(f'good{i}', 0.95) for i in range(2)]
                + [_fake_example(f'bad{i}', 0.40) for i in range(998)])
    _order, report = data.sample_training_set(examples, cfg.SEED,
                                              good_fraction=0.5, max_good_repeat=1)
    assert report.good_repeat == 1
    assert report.good_fraction < 0.5
    assert report.as_dict()['target_good_fraction'] == 0.5


def test_sampling_keeps_every_bad_row():
    examples = ([_fake_example(f'good{i}', 0.95) for i in range(50)]
                + [_fake_example(f'bad{i}', 0.40) for i in range(150)])
    order, report = data.sample_training_set(examples, cfg.SEED)
    assert report.bad_rows == 150
    assert sum(1 for row in order if not row.is_good) == 150


# --- the conservative gate ----------------------------------------------------

def _gate_rows(count: int, helped_pattern) -> list[dict]:
    rows = []
    for i in range(count):
        helped = helped_pattern(i)
        confidence = 0.9 if helped else 0.2
        rows.append({
            'current_iou': 0.5, 'refined_iou': 0.6 if helped else 0.4,
            'delta_iou': 0.1 if helped else -0.1, 'unchanged': False,
            'changed_points': 10, 'region_points': 100,
            'removed_points': 5, 'added_points': 5,
            'remove_true_positive': 0, 'remove_false_positive': 0,
            'remove_false_negative': 0, 'add_true_positive': 0,
            'add_false_positive': 0, 'add_false_negative': 0,
            'change_region_f1': 0.5, 'change_region_iou': 0.5,
            'gate_remove_top': confidence, 'gate_remove_mean': confidence,
            'gate_add_top': confidence, 'gate_add_mean': confidence,
            'gate_change_fraction': 0.1, 'gate_remove_fraction': 0.05,
            'gate_add_fraction': 0.05, 'gate_change_score': confidence,
        })
    return rows


def test_the_gate_refuses_everything_when_nothing_was_edited():
    rows = _gate_rows(20, lambda i: True)
    for row in rows:
        row['unchanged'] = True
    gate = train.fit_gate(rows, verbose=False)
    assert gate['effective_threshold'] >= cfg.GATE_MIN_PROBABILITY
    assert gate['coverage'] == 0.0
    gated = train.apply_gate(rows, gate)
    assert all(not row['auto_edit'] for row in gated)
    # A refusal must leave the current mask standing, not silently apply the edit.
    assert all(row['gated_refined_iou'] == row['current_iou'] for row in gated)
    assert all(row['gated_delta_iou'] == 0.0 for row in gated)


def test_the_gate_accepts_edits_that_helped_and_refuses_those_that_did_not():
    rows = _gate_rows(60, lambda i: i % 2 == 0)
    gate = train.fit_gate(rows, verbose=False)
    gated = train.apply_gate(rows, gate)
    accepted = [row for row in gated if row['auto_edit']]
    assert accepted, 'a separable validation set should let the gate accept something'
    assert all(row['delta_iou'] > 0 for row in accepted)


def test_frozen_thresholds_are_on_disk_and_reapplied_not_rechosen():
    rows = _gate_rows(40, lambda i: i % 3 != 0)
    gate = train.fit_gate(rows, verbose=False)
    path = cfg.REPORT_DIR / 'FROZEN_GATE_test.json'
    digest = train.freeze_gate(gate, path)
    try:
        frozen = json.loads(path.read_text(encoding='utf-8'))
        assert frozen['frozen_before_wytham_test'] is True
        assert frozen['gate']['effective_threshold'] == gate['effective_threshold']
        assert frozen['frozen_thresholds']['tuned_on'] == 'train + validation only'
        assert len(digest) == 64
        # Applying the frozen gate must equal applying the fitted one.
        assert ([r['auto_edit'] for r in train.apply_gate(rows, frozen['gate'])]
                == [r['auto_edit'] for r in train.apply_gate(rows, gate)])
    finally:
        path.unlink(missing_ok=True)


def test_the_gate_uses_no_ground_truth_feature():
    """The gate must not be able to read the answer it is predicting."""
    forbidden = {'delta_iou', 'refined_iou', 'current_iou', 'truth', 'gt',
                 'current_mask', 'added_points', 'removed_points'}
    assert not (set(train.GATE_FEATURE_ORDER) & forbidden)
    assert all(name.startswith('gate_') for name in train.GATE_FEATURE_ORDER)


# --- the verdict --------------------------------------------------------------

def _gated(bad: float, good: float, degradation: float, coverage: float,
           success) -> dict:
    return {'BAD_MEAN_DELTA_IOU': bad, 'BAD_IMPROVEMENT_RATE': 0.0,
            'GOOD_MEAN_DELTA_IOU': good, 'GOOD_DEGRADATION_RATE': degradation,
            'AUTO_EDIT_COVERAGE': coverage, 'AUTO_EDIT_SUCCESS_RATE': success,
            'ALL': {'delta_iou_mean': 0.0},
            'GOOD': {'delta_iou_mean': good}, 'BAD': {'delta_iou_mean': bad}}


def _payload(bad: float, good: float, degradation: float, coverage: float,
             success) -> dict:
    fold = {'configuration': 'cross_site_test_wytham',
            'test': {'gated': _gated(bad, good, degradation, coverage, success),
                     'ungated': _gated(-0.5, -0.7, 0.98, None, None)},
            'gate': {'coverage': 0.3, 'success_rate': 0.1}}
    return {'folds': [fold], 'result_a_v2_frozen_full_mask': {'configurations': {}}}


def test_verdict_is_go_only_when_all_five_criteria_pass():
    payload = _payload(0.05, 0.0, 0.0, 0.5, 0.9)
    verdict = metrics.verdict(payload)
    assert verdict['verdict'] == 'DEEP_LEARNING_GO'
    assert verdict['TREE_QC_DEEP_LEARNING_ROUTE'] == 'CONTINUE'


def test_verdict_is_no_go_when_a_core_criterion_fails():
    payload = _payload(bad=-0.01, good=0.0, degradation=0.0, coverage=0.5, success=0.9)
    verdict = metrics.verdict(payload)
    assert verdict['verdict'] == 'DEEP_LEARNING_NO_GO'
    assert verdict['TREE_QC_DEEP_LEARNING_ROUTE'] == 'STOPPED'
    assert 'recommended_next_route' in verdict


def test_verdict_is_borderline_when_core_passes_but_edits_do_not():
    payload = _payload(bad=0.05, good=0.0, degradation=0.0, coverage=0.02, success=0.9)
    verdict = metrics.verdict(payload)
    assert verdict['verdict'] == 'DEEP_LEARNING_BORDERLINE'
    assert verdict['core_passed'] is True
    assert verdict['edit_passed'] is False


def test_a_gate_that_refuses_everything_cannot_buy_a_pass():
    """The round's own criterion 5 exists for this, so it must fail here."""
    payload = _payload(bad=0.0, good=0.0, degradation=0.0, coverage=0.0, success=None)
    verdict = metrics.verdict(payload)
    assert verdict['verdict'] == 'DEEP_LEARNING_NO_GO'
    assert verdict['criteria']['CROSS_SITE_AUTO_EDIT_COVERAGE']['passed'] is False
    assert verdict['criteria']['CROSS_SITE_AUTO_EDIT_SUCCESS_RATE']['passed'] is False
    # BAD_MEAN_DELTA_IOU of exactly zero does not satisfy "> 0".
    assert verdict['criteria']['CROSS_SITE_BAD_MEAN_DELTA_IOU']['passed'] is False


# --- declared thresholds and configuration ------------------------------------

def test_declared_criteria_match_the_round_specification():
    assert cfg.CRITERIA['CROSS_SITE_BAD_MEAN_DELTA_IOU'] == {'op': '>', 'value': 0.0}
    assert cfg.CRITERIA['CROSS_SITE_GOOD_MEAN_DELTA_IOU']['value'] == -0.005
    assert cfg.CRITERIA['CROSS_SITE_GOOD_DEGRADATION_RATE']['value'] == 0.05
    assert cfg.CRITERIA['CROSS_SITE_AUTO_EDIT_SUCCESS_RATE']['value'] == 0.80
    assert cfg.CRITERIA['CROSS_SITE_AUTO_EDIT_COVERAGE']['value'] == 0.10
    assert cfg.MAX_EPOCHS == 15
    assert cfg.PATIENCE == 3
    assert cfg.CORE_CRITERIA + cfg.EDIT_CRITERIA == tuple(cfg.CRITERIA)


def test_the_backbone_is_the_frozen_v2_module():
    """Same backbone must be a fact about the code, not a claim in a report."""
    model = model_module.build_model(5)
    assert type(model.backbone).__module__.startswith('v2_models')
    breakdown = model_module.parameter_breakdown(5)
    v2_b = breakdown['v2_same_backbone_parameter_count']['MODEL_B_POINTNEXT_S']
    assert breakdown['parameters_backbone'] < v2_b
    assert breakdown['parameters_heads'] > 0
    assert breakdown['parameters_total'] == (
        breakdown['parameters_backbone'] + breakdown['parameters_heads'])


def test_config_digest_covers_every_frozen_threshold():
    digest = cfg.config_digest()
    assert len(digest) == 64
    assert digest == cfg.config_digest()
    frozen = cfg.frozen_thresholds()
    assert frozen['frozen_before_wytham'] is True
    assert frozen['tuned_on'] == 'train + validation only'


# --- the run artefacts, when present ------------------------------------------

def test_the_real_run_satisfies_its_own_protocol():
    path = cfg.REPORT_DIR / 'V25_GO_NO_GO.json'
    if not path.exists():
        pytest.skip('the round has not been run')
    run = json.loads(path.read_text(encoding='utf-8'))
    assert run['base_head'] == cfg.BASE_HEAD
    assert run['result_a_v2_frozen_full_mask']['baseline_is_retrained_this_round'] is False
    configurations = [fold['configuration'] for fold in run['folds']]
    assert 'cross_site_test_wytham' in configurations
    for fold in run['folds']:
        assert fold['thresholds_frozen_before_test'] is True
        assert fold['gate']['fitted_on'] == 'validation rows only'
        assert len(fold['gate_frozen_sha256']) == 64
        # The gate is applied with the frozen threshold, never a fresh one.
        frozen = json.loads(Path(fold['gate_frozen_file']).read_text(encoding='utf-8'))
        assert frozen['gate']['effective_threshold'] == \
            fold['gate']['effective_threshold']
        training = fold['training']
        assert training['epochs_run'] <= cfg.MAX_EPOCHS
        assert training['seed'] == cfg.SEED
        assert training['sampling']['good_rows'] > 0
        assert training['sampling']['good_rows_are_explicit_training_samples'] is True


def test_the_real_run_respects_the_round_prohibitions():
    path = cfg.REPORT_DIR / 'V25_GO_NO_GO.json'
    if not path.exists():
        pytest.skip('the round has not been run')
    report_path = cfg.REPO_ROOT / 'TREE_QC_V25_FINAL_GO_NO_GO_REPORT.json'
    if not report_path.exists():
        pytest.skip('the report has not been generated')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    prohibitions = report['prohibitions']
    for key in ('local_20_trees_accessed', 'local_8_bad_accessed',
                'product_integration', 'onnx', 'crown_stem_hybrid_modified',
                'new_forest_downloaded', 'reverse_fold_run', 'test_site_changed',
                'fourth_model_added', 'thresholds_tuned_on_wytham',
                'hyperparameter_search', 'epochs_increased_after_results'):
        assert prohibitions[key] == 'NO', key
    assert report['v2_code_untouched']['v2_files_modified'] == []
    assert report['provenance']['can_the_result_be_called_product_generalisation'] is False
    assert report['provenance']['simulated_share_of_training_rows'] is not None


def test_the_real_verdict_matches_its_own_criteria():
    report_path = cfg.REPO_ROOT / 'TREE_QC_V25_FINAL_GO_NO_GO_REPORT.json'
    if not report_path.exists():
        pytest.skip('the report has not been generated')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    verdict = report['verdict']
    core = all(verdict['criteria'][name]['passed'] for name in cfg.CORE_CRITERIA)
    edit = all(verdict['criteria'][name]['passed'] for name in cfg.EDIT_CRITERIA)
    if not core:
        assert verdict['verdict'] == 'DEEP_LEARNING_NO_GO'
    elif edit:
        assert verdict['verdict'] == 'DEEP_LEARNING_GO'
    else:
        assert verdict['verdict'] == 'DEEP_LEARNING_BORDERLINE'
    expected_route = ('STOPPED' if verdict['verdict'] == 'DEEP_LEARNING_NO_GO'
                      else 'CONTINUE')
    assert verdict['TREE_QC_DEEP_LEARNING_ROUTE'] == expected_route
    markdown = (cfg.REPO_ROOT / 'TREE_QC_V25_FINAL_GO_NO_GO_REPORT.md').read_text(
        encoding='utf-8')
    assert verdict['verdict'] in markdown
    assert expected_route in markdown
    assert 'SIMULATED_PROPOSAL' in markdown
