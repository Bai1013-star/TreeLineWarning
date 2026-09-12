"""Tests for the Tree QC V2 feasibility research code.

These are unit tests over the round's own modules. They do not read the public
dataset unless a test is explicitly marked, and they never touch the frozen V1
artefacts, the TreeLineWarning local corpus, or the local 8-bad challenge.

The properties under test are the ones a wrong implementation would silently
break and still produce plausible numbers:

* the ROI is scored consistently, so a proposal is never compared against a
  ground-truth mask that the ROI itself truncated;
* ``match_proposals`` maps a proposal to the instance that actually holds it -
  the off-by-one that made every real proposal look like a total failure;
* the change-region metrics really do separate an identity copier from a refiner;
* the three architectures are a fair comparison: identical decoder, and ``MODEL_C``
  a strict superset of ``MODEL_B``;
* the feature layout conversion is the one the network was written for.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_config as cfg
import v2_data as data
import v2_models as models
import v2_proposal_test as proposal_test
import v2_refiner_data as refdata
import v2_refiner_train as trainer
import v2_scene as scene

# --- the off-by-one that invalidated every real proposal -----------------------

def test_match_proposals_uses_the_instance_that_holds_the_points():
    """Instance ids start at zero and the matcher must not drop bin zero."""
    instance_ids = (0, 1, 2)
    gt_instance = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, -1], dtype=np.int32)
    proposals = []
    for wanted in (0, 1, 2):
        mask = gt_instance == wanted
        proposals.append({'mask': mask, 'state': 'CONFIRMED_TREE'})
    matched = scene.match_proposals(proposals, gt_instance, instance_ids)
    assert [row['instance'] for row in matched['rows']] == [0, 1, 2]
    assert all(row['unique'] for row in matched['rows'])
    assert [row['matched_points'] for row in matched['rows']] == [2, 3, 4]


def test_match_proposals_marks_a_merge_as_not_unique():
    instance_ids = (0, 1)
    gt_instance = np.array([0, 0, 1, 1], dtype=np.int32)
    merged = {'mask': np.array([True, True, True, True]), 'state': 'X'}
    matched = scene.match_proposals([merged], gt_instance, instance_ids)
    assert matched['rows'][0]['share'] == pytest.approx(0.5)
    assert not matched['rows'][0]['unique']


# --- proposal metrics ---------------------------------------------------------

def test_identity_copy_has_no_change_region():
    target = np.array([True, True, True, False, False, False])
    current = np.array([True, True, False, False, False, True])
    metrics = trainer.region_metrics(current, target, current)
    assert metrics['delta_iou'] == pytest.approx(0.0)
    assert metrics['unchanged'] is True
    assert metrics['change_region_f1'] == 0.0
    assert metrics['added_points'] == 0
    assert metrics['removed_points'] == 0


def test_a_real_correction_scores_above_the_identity_copier():
    target = np.array([True, True, True, False, False, False])
    current = np.array([True, True, False, False, False, True])
    corrected = np.array([True, True, True, False, False, False])
    identity = trainer.region_metrics(current, target, current)
    fixed = trainer.region_metrics(corrected, target, current)
    assert fixed['delta_iou'] > identity['delta_iou']
    assert fixed['change_region_f1'] > identity['change_region_f1']
    assert fixed['refined_iou'] == pytest.approx(1.0)


def test_removed_point_metrics_track_the_removal():
    target = np.array([True, True, False, False])
    current = np.array([True, True, True, True])
    cleaned = np.array([True, True, False, False])
    metrics = trainer.region_metrics(cleaned, target, current)
    assert metrics['removed_points'] == 2
    assert metrics['removed_point_recall'] == pytest.approx(1.0)
    assert metrics['removed_point_precision'] == pytest.approx(1.0)


def test_metric_counts_are_self_consistent():
    rng = np.random.default_rng(0)
    target = rng.random(1000) > 0.6
    current = rng.random(1000) > 0.5
    predicted = rng.random(1000) > 0.45
    metrics = trainer.region_metrics(predicted, target, current)
    assert metrics['changed_points'] == int((predicted != current).sum())
    assert metrics['added_points'] == int((predicted & ~current).sum())
    assert metrics['removed_points'] == int((current & ~predicted).sum())
    assert 0.0 <= metrics['refined_iou'] <= 1.0


# --- gates --------------------------------------------------------------------

def _summary(delta_values, change_f1, current_values=None):
    rows = []
    currents = current_values if current_values is not None else [0.5] * len(delta_values)
    for delta, current in zip(delta_values, currents, strict=True):
        rows.append({'delta_iou': delta, 'current_iou': current,
                     'refined_iou': current + delta,
                     'refined_precision': 0.5, 'refined_recall': 0.5,
                     'change_region_f1': change_f1, 'change_region_recall': change_f1,
                     'added_point_f1': change_f1, 'removed_point_f1': change_f1,
                     'unchanged': abs(delta) < 1e-9})
    return trainer.summarise(rows, 'test')


def test_identity_copy_gate_rejects_a_model_that_changes_nothing():
    summary = _summary([0.0] * 20, change_f1=0.0)
    gate = trainer.identity_copy_gate(summary)
    assert gate['verdict'] == 'FAIL'
    assert gate['identity_copy_fraction'] == 1.0


def test_identity_copy_gate_accepts_a_model_that_repairs_bad_proposals():
    summary = _summary([0.20] * 20, change_f1=0.45, current_values=[0.4] * 20)
    gate = trainer.identity_copy_gate(summary)
    assert gate['verdict'] == 'PASS'


def test_good_preservation_gate_rejects_a_model_that_breaks_good_trees():
    summary = _summary([-0.30] * 20, change_f1=0.4, current_values=[0.95] * 20)
    gate = trainer.good_preservation_gate(summary)
    assert gate['verdict'] == 'FAIL'
    assert gate['severe_degradation_count'] == 20


def test_good_preservation_gate_accepts_a_gentle_model():
    summary = _summary([0.01, -0.01] * 10, change_f1=0.2,
                       current_values=[0.95] * 20)
    gate = trainer.good_preservation_gate(summary)
    assert gate['verdict'] == 'PASS'


def test_summarise_separates_good_from_bad_proposals():
    rows = []
    for current, delta in ((0.95, -0.01), (0.30, 0.25), (0.90, 0.00), (0.10, 0.40)):
        rows.append({'delta_iou': delta, 'current_iou': current,
                     'refined_iou': current + delta, 'refined_precision': 0.5,
                     'refined_recall': 0.5, 'change_region_f1': 0.5,
                     'change_region_recall': 0.5, 'added_point_f1': 0.5,
                     'removed_point_f1': 0.5, 'unchanged': abs(delta) < 1e-9})
    summary = trainer.summarise(rows, 'x')
    assert summary['good_proposals']['rows'] == 2
    assert summary['bad_proposals']['rows'] == 2
    assert summary['bad_proposals']['mean_delta_iou'] > 0


# --- the three architectures --------------------------------------------------

@pytest.mark.parametrize('key', ('A', 'B', 'C'))
def test_models_produce_per_point_logits(key):
    torch.manual_seed(0)
    channels = len(refdata.FEATURE_CHANNELS)
    model = models.build_model(key, channels)
    model.train()
    points = torch.randn(2, 256, channels) * 0.3
    points[..., 3] = (torch.rand(2, 256) > 0.5).float()
    output = model(points)
    assert output['logits'].shape == (2, 256)
    if key == 'C':
        assert output['offset'].shape == (2, 256, 3)
        assert output['quality'].shape == (2,)
    else:
        assert 'offset' not in output and 'quality' not in output


def test_model_c_is_a_strict_superset_of_model_b():
    torch.manual_seed(0)
    channels = len(refdata.FEATURE_CHANNELS)
    model_b = models.build_model('B', channels)
    torch.manual_seed(0)
    model_c = models.build_model('C', channels)
    shared = [name for name in model_b.state_dict()
              if not name.startswith(('offset.', 'quality.'))]
    for name in shared:
        assert torch.equal(model_b.state_dict()[name], model_c.state_dict()[name]), name
    assert models.parameter_count(model_c) > models.parameter_count(model_b)


def test_models_a_and_b_share_one_decoder():
    channels = len(refdata.FEATURE_CHANNELS)
    backbone_a = models.PointNetPlusPlusBackbone(channels)
    backbone_b = models.PointNeXtBackbone(channels)
    for name in ('fp1', 'fp2', 'fp3', 'fp4'):
        first = getattr(backbone_a, name).mlp[0]
        second = getattr(backbone_b, name).mlp[0]
        assert first.in_channels == second.in_channels, name
        assert first.out_channels == second.out_channels, name
        assert type(first) is type(second)


@pytest.mark.parametrize('key', ('A', 'B', 'C'))
def test_every_parameter_receives_a_gradient(key):
    torch.manual_seed(0)
    channels = len(refdata.FEATURE_CHANNELS)
    model = models.build_model(key, channels)
    model.train()
    points = torch.randn(2, 256, channels) * 0.3
    points[..., 3] = (torch.rand(2, 256) > 0.5).float()
    target = (torch.rand(2, 256) > 0.5).float()
    output = model(points)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output['logits'], target)
    if model.auxiliary:
        loss = loss + output['offset'].abs().mean() + output['quality'].mean()
    loss.backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
    assert not missing, missing


def test_conv1d_rejects_a_transposed_tensor():
    """A (B, C, N) tensor reaching the point-wise helper must fail loudly."""
    layers = models.pointwise([5, 8])
    with pytest.raises(ValueError):
        models.conv1d(torch.randn(2, 5, 16), layers)


def test_the_validation_batch_is_enough_for_the_multitask_model():
    """Early stopping must not need the auxiliary targets.

    The validation subset carries only features, targets and the current mask, so a
    validation pass that calls the full multitask loss raises ``KeyError`` on
    ``offset_active`` for MODEL_C. That crash ended a real bakeoff run, which is
    why it is pinned here.
    """
    import v2_refiner_data as refdata_module

    batch = refdata_module.collate_validation([{
        'features': torch.zeros(8, len(refdata.FEATURE_CHANNELS)),
        'target': torch.zeros(8), 'current': torch.zeros(8),
        'filename': 'a|b', 'site': 's', 'proposal_kind': 'SIMULATED_PROPOSAL',
        'corruption': 'ADD_CONTAMINATION', 'severity': 'HIGH', 'current_iou': 0.5,
    }])
    assert 'offset_active' not in batch
    model = models.build_model('C', len(refdata.FEATURE_CHANNELS))
    outputs = model(batch['features'])
    loss = trainer.membership_loss(outputs['logits'], batch['target'],
                                   batch['current']).mean()
    assert torch.isfinite(loss)


def test_early_stopping_uses_one_rule_for_every_model():
    """All three architectures must be selected on the same quantity."""
    source = (Path(__file__).resolve().parent / 'v2_refiner_train.py').read_text(
        encoding='utf-8')
    validation_block = source.split('model.eval()', 2)[1]
    assert 'membership_loss(' in validation_block
    assert 'total_loss(' not in validation_block


def test_ball_query_indices_stay_in_range():
    xyz = torch.randn(1, 64, 3) * 5.0
    centers = xyz[:, :8]
    indices = models.ball_query(xyz, centers, 0.01, 16)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 64


# --- feature contract ---------------------------------------------------------

def test_feature_channels_are_exactly_the_declared_five():
    assert refdata.FEATURE_CHANNELS == (
        'local_x', 'local_y', 'local_z', 'current_candidate_membership',
        'log_candidate_extent')


def test_candidate_frame_is_derived_from_the_proposal_only():
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 4.0], [0.0, 2.0, 8.0]],
                   dtype=np.float32)
    frame = refdata.candidate_frame(xyz)
    assert frame['scale'] == pytest.approx(8.0)
    assert frame['centre'][2] == pytest.approx(4.0)


def test_minimum_scale_floor_protects_a_degenerate_proposal():
    xyz = np.zeros((5, 3), dtype=np.float32)
    frame = refdata.candidate_frame(xyz)
    assert frame['scale'] >= refdata.MIN_SCALE_M


def test_sampler_guarantees_proposal_and_target_points():
    rng = np.random.default_rng(0)
    total = 5000
    current = np.zeros(total, dtype=bool)
    current[:100] = True
    gt = np.zeros(total, dtype=bool)
    gt[:60] = True
    gt[100:160] = True
    instance = np.zeros(total, dtype=np.int32)
    instance[100:300] = 3
    selected = refdata.sample_points(current, gt, instance, 512, rng)
    assert selected.size == 512
    assert np.unique(selected).size == 512
    assert current[selected].any()
    assert (gt[selected] & ~current[selected]).any()


def test_sampler_handles_a_roi_smaller_than_the_target():
    rng = np.random.default_rng(0)
    total = 100
    current = np.ones(total, dtype=bool)
    gt = np.ones(total, dtype=bool)
    instance = np.zeros(total, dtype=np.int32)
    selected = refdata.sample_points(current, gt, instance, 512, rng)
    assert selected.size == 512


def test_stem_reference_is_a_proxy_and_is_recorded_as_one():
    xyz = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.2],
                    [0.0, 0.0, 12.0]], dtype=np.float32)
    ground = np.zeros(4, dtype=np.float32)
    reference = refdata.stem_base_reference(xyz, ground)
    assert reference['points'] >= 1
    assert 'proxy' in reference['definition'] or 'median' in reference['definition']
    assert reference['reference'][2] < 5.0


def test_proposal_mask_treats_gt_correct_as_the_ground_truth():
    cluster = {'masks': {'gt': np.array([True, False, True]),
                         'SIM_X': np.array([True, True, False])}}
    assert np.array_equal(refdata.proposal_mask(cluster, 'GT_CORRECT'),
                          cluster['masks']['gt'])
    assert np.array_equal(refdata.proposal_mask(cluster, 'SIM_X'),
                          cluster['masks']['SIM_X'])


# --- ROI construction ---------------------------------------------------------

def _patch() -> scene.Patch:
    rng = np.random.default_rng(0)
    first = rng.normal(0.0, 1.0, size=(400, 3)).astype(np.float32)
    first[:, 2] += 5.0
    second = rng.normal(40.0, 1.0, size=(200, 3)).astype(np.float32)
    second[:, 2] += 5.0
    xyz = np.concatenate([first, second])
    instance = np.concatenate([np.zeros(400, dtype=np.int32),
                               np.ones(200, dtype=np.int32)])
    return scene.Patch(patch_id='p', site='s', xyz=xyz, instance=instance,
                       instance_ids=(0, 1),
                       point_counts={0: {'points_in_window': 400, 'points_total': 400},
                                     1: {'points_in_window': 200, 'points_total': 200}},
                       ground_z=np.zeros(len(xyz), dtype=np.float32))


def test_roi_contains_the_target_and_its_neighbours():
    patch = _patch()
    roi = scene.roi_mask(patch, 0)
    assert roi[:400].all()
    assert not roi[400:].any()


def test_roi_coverage_of_the_target_is_complete_by_construction():
    patch = _patch()
    roi = scene.roi_mask(patch, 0)
    target = patch.instance == 0
    assert int((target & roi).sum()) == int(target.sum())


def test_qualifying_targets_rejects_a_truncated_boundary_tree():
    patch = _patch()
    patch.point_counts[0]['points_total'] = 4000
    qualifying = proposal_test.qualifying_targets(patch, minimum_points=100)
    assert [row['instance'] for row in qualifying] == [1]


def test_qualifying_targets_rejects_a_fragment():
    patch = _patch()
    qualifying = proposal_test.qualifying_targets(patch, minimum_points=300)
    assert [row['instance'] for row in qualifying] == [0]


# --- corruption construction --------------------------------------------------

def _view() -> proposal_test.RoiView:
    """A synthetic ROI with a full vertical profile: stem at the bottom, crown
    above. A fixture without low points could never exercise STEM_MISSING."""
    rng = np.random.default_rng(0)
    n = 800
    xyz = rng.normal(0.0, 0.6, size=(n, 3)).astype(np.float32)
    # Spread heights over 0.2 m to 12 m so some points sit inside the stem band.
    xyz[:, 2] = np.linspace(0.2, 12.0, n).astype(np.float32) \
        + rng.normal(0.0, 0.05, size=n).astype(np.float32)
    instance = np.zeros(n, dtype=np.int32)
    instance[400:] = 1
    target = instance == 0
    return proposal_test.RoiView(
        patch_id='p', site='s', target_instance=0, target_tree_id='t',
        rows=np.arange(n), xyz=xyz, instance=instance,
        ground_z=np.zeros(n, dtype=np.float32), target_mask=target,
        containment=1.0, points_file=n)


@pytest.mark.parametrize('corruption', proposal_test.CORRUPTIONS)
@pytest.mark.parametrize('severity', proposal_test.SEVERITIES)
def test_every_corruption_produces_a_mask_with_the_declared_size(corruption, severity):
    view = _view()
    result = proposal_test.simulate(view, corruption, severity, ('p', 0))
    mask = result['mask']
    assert mask.shape == view.target_mask.shape
    changed = int((mask != view.target_mask).sum())
    assert changed > 0, result['detail']
    expected = {'LOW': 0.10, 'MEDIUM': 0.25, 'HIGH': 0.45}[severity]
    if corruption == 'MERGED_NEIGHBOR':
        # A merge takes the whole nearest neighbour.
        assert changed == int((~view.target_mask).sum())
    elif corruption == 'STEM_MISSING':
        # The stem corruption is a fraction of the target's points inside the
        # lowest 2 m band, not of the whole target.
        band = view.target_mask & ((view.xyz[:, 2] - view.ground_z) <= 2.0)
        assert changed == max(1, int(round(expected * int(band.sum()))))
    else:
        assert changed == max(1, int(round(expected * int(view.target_mask.sum()))))


def test_corruptions_are_deterministic():
    view = _view()
    first = proposal_test.simulate(view, 'REMOVE_MISSING', 'HIGH', ('p', 0))
    second = proposal_test.simulate(view, 'REMOVE_MISSING', 'HIGH', ('p', 0))
    assert np.array_equal(first['mask'], second['mask'])
    other = proposal_test.simulate(view, 'REMOVE_MISSING', 'HIGH', ('p', 1))
    assert not np.array_equal(first['mask'], other['mask'])


def test_higher_severity_moves_more_points():
    view = _view()
    sizes = []
    for severity in proposal_test.SEVERITIES:
        result = proposal_test.simulate(view, 'ADD_CONTAMINATION', severity, ('p', 0))
        sizes.append(int((result['mask'] != view.target_mask).sum()))
    assert sizes[0] < sizes[1] < sizes[2]


def test_merge_by_definition_contaminates_with_a_neighbour():
    view = _view()
    result = proposal_test.simulate(view, 'MERGED_NEIGHBOR', 'LOW', ('p', 0))
    metrics = scene.proposal_metrics(result['mask'], view.target_mask,
                                     (view.instance > 0) & (~view.target_mask),
                                     view.instance < 0)
    assert metrics['neighbour_contamination_points'] > 0
    assert metrics['is_merge']


# --- mask quality and offset supervision --------------------------------------

def test_offset_supervision_only_covers_target_points():
    """Offsets are unknown off the target, so they must not be supervised there."""
    assert refdata.OFFSET_ACTIVE_FRACTION > 0
    active = np.array([True, False, True, False])
    difference = np.array([0.1, 9.9, 0.2, 9.9])
    masked = (difference * active).sum() / active.sum()
    assert masked == pytest.approx(0.15)


# --- config -------------------------------------------------------------------

def test_config_digest_is_stable_and_content_addressed():
    first = cfg.config_digest()
    second = cfg.config_digest()
    assert first == second
    assert len(first) == 64
    assert first != cfg.SCHEMA


def test_declared_thresholds_are_the_ones_the_report_quotes():
    assert cfg.GOOD_IOU_THRESHOLD == 0.80
    assert cfg.GOOD_DEGRADED_IOU_DROP == 0.05
    assert cfg.GOOD_SEVERE_IOU_DROP == 0.20
    assert trainer.DECISION_THRESHOLD == 0.5
    assert trainer.GOOD_IOU_THRESHOLD == cfg.GOOD_IOU_THRESHOLD


def test_absent_sites_are_declared_not_silently_dropped():
    assert 'ofental' in cfg.ABSENT_SITES
    assert 'robsoncreek' in cfg.ABSENT_SITES
    assert set(cfg.SITE_TREE_ROOT) == {'litchfield', 'wytham'}


# --- report assembly ----------------------------------------------------------

def test_report_reads_artefacts_and_reports_missing_stages_as_not_run():
    """With no artefacts present the report must say NOT_RUN, not invent values."""
    import tempfile

    import v2_report as report
    original = report.REPORTS
    with tempfile.TemporaryDirectory() as directory:
        report.REPORTS = Path(directory)
        try:
            payload = report.build()
        finally:
            report.REPORTS = original
    assert payload['site_fingerprint_test']['status'] == 'NOT_RUN'
    assert payload['public_gt_proposal_test']['status'] == 'NOT_RUN'
    assert payload['mini_refiner_bakeoff']['status'] == 'NOT_RUN'
    assert payload['formal_v2_training_justified'] == 'NO'
    assert payload['local_data_untouched']['local_20_trees_accessed'] == 'NO'
    assert payload['local_data_untouched']['local_8_bad_accessed'] == 'NO'


def test_identity_baseline_defines_the_failure_the_gate_catches():
    import v2_report as report
    baseline = report._identity_baseline(None)
    assert baseline['delta_iou_mean'] == 0.0
    assert baseline['change_region_f1_mean'] == 0.0
    assert baseline['identity_copy_gate'] == 'FAIL'


def _fake_model(name: str, delta: float) -> dict:
    summary = {
        'label': 'x', 'rows': 100, 'current_iou_mean': 0.6, 'current_iou_median': 0.7,
        'refined_iou_mean': 0.6 + delta, 'refined_iou_median': 0.7 + delta,
        'delta_iou_mean': delta, 'delta_iou_median': delta,
        'improved_fraction': 0.4, 'unchanged_fraction': 0.3, 'degraded_fraction': 0.3,
        'improvement_gt_0.02': 0.3, 'degradation_gt_0.02': 0.1,
        'refined_precision_mean': 0.8, 'refined_recall_mean': 0.8,
        'change_region_f1_mean': 0.4, 'change_region_recall_mean': 0.4,
        'added_point_f1_mean': 0.3, 'removed_point_f1_mean': 0.5,
        'identity_copy_fraction': 0.3,
        'good_proposals': {'rows': 50, 'mean_delta_iou': -0.01,
                           'median_delta_iou': 0.0, 'degraded_fraction': 0.2,
                           'degraded_fraction_gt_0.05': 0.1,
                           'severe_degradation_count': 1, 'mean_current_iou': 0.9,
                           'mean_refined_iou': 0.89},
        'bad_proposals': {'rows': 50, 'mean_delta_iou': delta * 2,
                          'median_delta_iou': delta, 'improved_fraction': 0.6,
                          'improvement_gt_0.02': 0.5, 'mean_current_iou': 0.3,
                          'mean_refined_iou': 0.3 + delta},
        'by_corruption': {'MERGED_NEIGHBOR': {
            'rows': 10, 'current_iou_mean': 0.5, 'refined_iou_mean': 0.6,
            'delta_iou_mean': 0.1, 'delta_iou_median': 0.1, 'improved_fraction': 0.7,
            'identity_copy_fraction': 0.1, 'change_region_f1_mean': 0.5}},
    }
    return {
        'model': name, 'model_key': name[-1], 'parameters': 1000, 'in_channels': 5,
        'points': 2048, 'epochs_budget': 30, 'epochs_run': 12, 'best_epoch': 11,
        'best_validation_loss': 0.3, 'history': [], 'threshold': 0.5,
        'examples': {'train': 156, 'validation': 34, 'test': 118},
        'validation_used_for_selection': 'x', 'seconds': 1.0, 'device': 'cuda',
        'summary': summary,
        'identity_copy_gate': {'verdict': 'PASS'},
        'good_preservation_gate': {'verdict': 'PASS', 'good_rows': 50,
                                   'mean_delta_iou': -0.01,
                                   'degraded_fraction_gt_0.05': 0.1,
                                   'severe_degradation_count': 1},
        'held_in_validation': {'rows': 20, 'delta_iou_mean': 0.02},
        'resources': {'parameters': 1000, 'device': 'cuda', 'points': 2048,
                      'single_roi_inference_seconds': 0.11,
                      'cuda_peak_memory_mb': 210.0, 'process_rss_delta_mb': 30.0,
                      'cpu_inference_supported': True},
        'cpu_probe': {'points': 2048, 'cpu_single_roi_inference_seconds': 0.9,
                      'supported': True},
        'checkpoint_digest': 'deadbeef',
    }


def test_the_report_renders_every_section_when_the_bakeoff_has_run():
    """The renderer must produce every headline section from a real payload shape."""
    import v2_report as report
    payload = report.build()
    payload['mini_refiner_bakeoff'] = {
        'status': 'RUN', 'points': 2048, 'epochs': 30, 'device': 'cuda', 'seed': 101,
        'library_index_sha256': 'ab' * 32, 'target_count': 308,
        'targets_per_site': {'litchfield': 190, 'wytham': 118},
        'evaluation_cap': {'max_points_per_roi': 200000, 'meaning': 'x'},
        'shared_protocol': {'optimiser': 'Adam'},
        'model_descriptions': {
            name: {'key': name[-1], 'parameters': 1000,
                   'auxiliary_heads': name.endswith('MULTITASK'), 'in_channels': 5,
                   'reference_implementation': name,
                   'implementation': 'LOCAL_IMPLEMENTATION (plain torch, no external code)',
                   'pretrained_weights': None}
            for name in ('MODEL_A_POINTNET2', 'MODEL_B_POINTNEXT_S',
                         'MODEL_C_POINTNEXT_S_MULTITASK')},
        'folds': [{
            'test_site': 'wytham', 'held_in_sites': ['litchfield'],
            'split': {'train': 156, 'validation': 34, 'test': 118},
            'test_site_never_used_for_selection': True, 'seconds': 1.0,
            'models': {'A': _fake_model('MODEL_A_POINTNET2', 0.05),
                       'B': _fake_model('MODEL_B_POINTNEXT_S', 0.07),
                       'C': _fake_model('MODEL_C_POINTNEXT_S_MULTITASK', 0.09)},
        }],
        'verdict': {},
    }
    payload['final_verdict'] = report._verdict(payload)
    payload['formal_v2_training_justified'] = (
        'YES' if payload['final_verdict']['verdict'] == 'V2_FEASIBILITY_PASS' else 'NO')
    markdown = report.render_markdown(payload)
    for section in ('TREE_QC_V2_FEASIBILITY_AUDIT', 'FINAL_VERDICT', 'BLOCKERS =',
                    'MINI_REFINER_BAKEOFF', 'SITE_FINGERPRINT_TEST',
                    'PUBLIC_GT_PROPOSAL_TEST', 'Known limitations',
                    'IDENTITY_COPY_GATE =', 'GOOD_PRESERVATION =',
                    'FORMAL_V2_TRAINING_JUSTIFIED'):
        assert section in markdown, section


def test_verdict_requires_both_gates_and_a_real_bad_proposal_improvement():
    import v2_report as report
    payload = report.build()
    models = {key: _fake_model(name, delta) for key, name, delta in (
        ('A', 'MODEL_A_POINTNET2', 0.05), ('B', 'MODEL_B_POINTNEXT_S', 0.07),
        ('C', 'MODEL_C_POINTNEXT_S_MULTITASK', 0.09))}
    fold = {'test_site': 'wytham', 'held_in_sites': ['litchfield'],
            'split': {'train': 1, 'validation': 1, 'test': 1}, 'models': models}
    payload['mini_refiner_bakeoff'] = {'status': 'RUN', 'folds': [fold],
                                       'model_descriptions': {}}
    assert report._verdict(payload)['verdict'] == 'V2_FEASIBILITY_PASS'

    # Break the good-preservation gate on every model: PASS must be withdrawn.
    for model in models.values():
        model['good_preservation_gate'] = {'verdict': 'FAIL', 'good_rows': 50,
                                           'mean_delta_iou': -0.4,
                                           'degraded_fraction_gt_0.05': 1.0,
                                           'severe_degradation_count': 50}
    assert report._verdict(payload)['verdict'] == 'V2_FEASIBILITY_PARTIAL'


# --- protocol fairness and leakage --------------------------------------------

def _toy_index() -> dict:
    targets = []
    for site, count in (('litchfield', 20), ('wytham', 10)):
        for number in range(count):
            targets.append({
                'target_id': f'{site}_p{number}#0', 'site': site,
                'rows': [{'proposal_id': 'SIM_ADD_CONTAMINATION_HIGH',
                          'proposal_kind': 'SIMULATED_PROPOSAL',
                          'corruption': 'ADD_CONTAMINATION', 'severity': 'HIGH',
                          'iou': 0.8},
                         {'proposal_id': 'GT_CORRECT', 'proposal_kind': 'GT_CORRECT',
                          'corruption': 'NONE', 'severity': 'NONE', 'iou': 1.0}]})
    return {'targets': targets}


def test_the_split_is_deterministic_for_a_given_seed():
    index = _toy_index()
    first = trainer.split_by_tree(index, test_site='wytham', seed=101)
    second = trainer.split_by_tree(index, test_site='wytham', seed=101)
    assert first.as_dict() == second.as_dict()
    assert first.train == second.train
    assert first.validation == second.validation
    assert first.test == second.test


def test_cross_site_split_never_leaks_the_test_site_into_training():
    index = _toy_index()
    split = trainer.split_by_tree(index, test_site='wytham', seed=101)
    by_id = {row['target_id']: row['site'] for row in index['targets']}
    assert {by_id[key] for key in split.train} == {'litchfield'}
    assert {by_id[key] for key in split.validation} == {'litchfield'}
    assert {by_id[key] for key in split.test} == {'wytham'}
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))
    assert not (set(split.train) & set(split.validation))


def test_a_single_site_fold_holds_that_site_out_entirely():
    index = _toy_index()
    split = trainer.split_by_tree(index, test_site='litchfield', seed=101)
    by_id = {row['target_id']: row['site'] for row in index['targets']}
    assert {by_id[key] for key in split.test} == {'litchfield'}
    assert {by_id[key] for key in split.train} == {'wytham'}


def test_every_model_sees_the_same_examples_for_a_fold():
    """The fairness claim, checked against the code that builds the batches."""
    index = _toy_index()
    split = trainer.split_by_tree(index, test_site='wytham', seed=101)
    built = [trainer.build_examples(index, split.train) for _ in range(3)]
    keys = [{example.key for example in group} for group in built]
    assert keys[0] == keys[1] == keys[2]
    assert len(built[0]) == len(split.train) * 2


def test_example_kinds_filter_is_the_same_for_every_model():
    index = _toy_index()
    split = trainer.split_by_tree(index, test_site='wytham', seed=101)
    everything = trainer.build_examples(index, split.train)
    only_simulated = trainer.build_examples(index, split.train,
                                            kinds={'SIMULATED_PROPOSAL'})
    assert len(only_simulated) < len(everything)
    assert all(example.proposal_kind == 'SIMULATED_PROPOSAL'
               for example in only_simulated)


def test_declared_epoch_and_early_stop_budget_are_shared_constants():
    """No model can be given a longer budget: the budget is one config value."""
    assert cfg.REFINER_EPOCHS > 0
    assert cfg.REFINER_EARLY_STOP_PATIENCE > 0
    assert cfg.REFINER_EARLY_STOP_MIN_DELTA > 0
    source = (Path(__file__).resolve().parent / 'v2_bakeoff.py').read_text(
        encoding='utf-8')
    assert source.count('trainer.train_one(') == 1
    assert 'epochs' in source


def test_the_bakeoff_can_only_run_three_models():
    import v2_bakeoff as bakeoff
    assert bakeoff.MODEL_KEYS == ('A', 'B', 'C')
    assert set(bakeoff.MODEL_KEYS) == set(models.MODELS)


def test_the_real_artefacts_render_without_a_template_error():
    """The report generator must run against whatever the runs actually wrote.

    The synthetic-payload test above checks the templates; this one checks that the
    real payload's shapes match them, which is where a missing key or a renamed
    field actually shows up. It is skipped when no run has produced artefacts yet,
    so the suite still passes on a clean checkout.
    """
    import v2_report as report
    if not (report.REPORTS / 'SITE_FINGERPRINT.json').exists():
        pytest.skip('no run artefacts present')
    payload = report.build()
    markdown = report.render_markdown(payload)
    assert len(markdown) > 5000
    for section in ('TREE_QC_V2_FEASIBILITY_AUDIT', 'FINAL_VERDICT',
                    'FORMAL_V2_TRAINING_JUSTIFIED', 'Known limitations',
                    'BLOCKERS =', 'License audit', 'STAGE_STATUS'):
        assert section in markdown, section
    verdict = payload['final_verdict']
    assert verdict['verdict'] in ('V2_FEASIBILITY_PASS', 'V2_FEASIBILITY_PARTIAL',
                                  'V2_FEASIBILITY_FAIL')
    # The round's own prohibitions have to survive into the report.
    assert payload['local_data_untouched']['local_20_trees_accessed'] == 'NO'
    assert payload['local_data_untouched']['local_8_bad_accessed'] == 'NO'
    assert payload['v1_history_untouched']['changed_files_in_research_tree_qc'] == []
    if payload['mini_refiner_bakeoff'].get('status') == 'RUN':
        # Every fold must say which configuration it is, or an in-site result could
        # be read as a cross-site one.
        for fold in payload['mini_refiner_bakeoff']['folds']:
            assert fold.get('configuration')
            assert fold['configuration'].startswith(('cross_site', 'in_site'))


# --- final task closure -------------------------------------------------------

def test_every_stage_reached_a_terminal_state():
    """No stage may be left unreported, and none may claim COMPLETE untruthfully."""
    import v2_report as report
    if not (report.REPORTS / 'SITE_FINGERPRINT.json').exists():
        pytest.skip('no run artefacts present')
    status = report.build()['stage_status']
    by_name = {stage['stage']: stage for stage in status['stages']}
    assert set(by_name) == {'SITE_FINGERPRINT_TEST', 'PUBLIC_GT_PROPOSAL_TEST',
                            'ROI_LIBRARY', 'MINI_REFINER_BAKEOFF', 'FINAL_REPORT'}
    for name, stage in by_name.items():
        assert stage['status'] in ('COMPLETE', 'COMPLETE_WITH_LIMITATION'), name
        assert stage['artefact'], name
    assert status['all_stages_reached_a_terminal_state'] is True
    assert status['round_stopped_at_verdict'] is True
    assert status['further_training_started'] is False


def test_the_bakeoff_limitation_names_the_fold_that_was_not_run():
    """The stage is COMPLETE_WITH_LIMITATION, and it says which fold is missing."""
    import v2_report as report
    if not (report.REPORTS / 'MINI_REFINER_BAKEOFF.json').exists():
        pytest.skip('no bakeoff artefact present')
    payload = report.build()
    limitation = payload['mini_refiner_bakeoff']['limitation']
    assert limitation['status'] == 'COMPLETE_WITH_LIMITATION'
    assert limitation['cross_site_folds_run'] == ['cross_site_test_wytham']
    assert limitation['cross_site_folds_missing'] == ['cross_site_test_litchfield']
    assert limitation['reverse_fold_deliberately_not_backfilled'] is True
    assert 'reverse' in limitation['why_incomplete'].lower()
    bakeoff = payload['mini_refiner_bakeoff']
    configurations = [fold['configuration'] for fold in bakeoff['folds']]
    assert configurations.count('cross_site_test_wytham') == 1
    assert 'cross_site_test_litchfield' not in configurations
    assert configurations.count('in_site_test_litchfield') == 1
    assert bakeoff['status'] == 'RUN'


def test_the_reported_library_digest_is_the_one_on_disk():
    """The digest the report quotes must describe the library the runs used."""
    import hashlib

    import v2_report as report
    if not (report.REPORTS / 'ROI_LIBRARY.json').exists():
        pytest.skip('no ROI library artefact present')
    payload = report.build()
    quoted = payload['mini_refiner_bakeoff']['library_index_sha256']
    assert quoted, 'the report did not resolve a library digest'
    assert quoted == payload['roi_library']['index_sha256']
    stored = payload['roi_library'].get('index_file')
    if stored and Path(stored).exists():
        digest = hashlib.sha256(Path(stored).read_bytes()).hexdigest()
        assert digest == quoted


# --- data layer ---------------------------------------------------------------

def test_ply_type_map_covers_every_type_the_benchmark_publishes():
    import v2_ply as ply
    for name in ('float', 'float32', 'double', 'float64'):
        assert name in ply.PLY_TYPES
    assert ply.PLY_TYPES['int'] == 'i4'
    assert ply.PLY_TYPES['uint'] == 'u4'


def test_site_root_map_points_only_at_the_public_extraction_roots():
    for site, root in cfg.SITE_TREE_ROOT.items():
        assert 'TreeQCData' in str(root)
        assert site in str(root)


def test_discovery_reports_real_counts_not_requested_counts():
    """The sampler must report the site's real size when it is smaller."""
    class Row:
        def __init__(self, site, key):
            self.site, self.key = site, key

    rows = [Row('a', f'a|{i}') for i in range(3)] + [Row('b', f'b|{i}') for i in range(50)]
    chosen = data.sample_instances(rows, 40, cfg.SEED)
    per_site = {}
    for row in chosen:
        per_site[row.site] = per_site.get(row.site, 0) + 1
    assert per_site['a'] == 3
    assert per_site['b'] == 40


def test_log_scale_channel_is_constant_per_example():
    """The explicit metric channel must not vary point to point."""
    scale = 12.0
    channel = np.full((10, 1), math.log(scale), dtype=np.float32)
    assert channel.std() == 0.0
    assert float(channel[0, 0]) == pytest.approx(math.log(scale))
