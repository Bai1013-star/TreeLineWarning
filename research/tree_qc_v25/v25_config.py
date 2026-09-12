"""Frozen configuration for TREE_QC_V25_FINAL_DEEP_LEARNING_GO_NO_GO.

The one rule this module exists to enforce: **every threshold is decided on train
and validation only, before the Wytham test is looked at once.** The values here
are declared in code, committed, and read back by the report, so a threshold that
moved after unblinding would be visible as a changed config digest rather than as a
quietly better number.

Nothing in this round retrains, re-scores or edits anything under
``research/tree_qc_v2/``. The V2 frozen results are read from their artefacts and
used as the old-method baseline exactly as they stand.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'tree_qc_v25_go_no_go_v1'
ROUND = 'TREE_QC_V25_FINAL_DEEP_LEARNING_GO_NO_GO'

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = REPO_ROOT / 'research' / 'tree_qc_v2'
RESEARCH_DIR = REPO_ROOT / 'research' / 'tree_qc_v25'
REPORT_DIR = RESEARCH_DIR / 'reports'

DATA_ROOT = Path(os.environ.get('TREE_QC_V2_DATA_ROOT', r'D:\TreeQCData'))
WORK_ROOT = DATA_ROOT / 'v2_feasibility'
ROI_DIR = WORK_ROOT / 'roi_library'

#: The frozen V2 commit this round starts from. Recorded so the report can prove
#: the baseline it compares against is the one that was accepted.
BASE_HEAD = '169d42dccd9b1173193d17167937bbde1a6638c0'

#: One seed, no hyper-parameter search, no best-of-N.
SEED = 101
MAX_EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
POINTS_PER_EXAMPLE = 2048
#: Evaluation cap, inherited from V2 so the two rounds are scored identically.
EVAL_MAX_POINTS = 200_000
#: Scoring batch. Identical for every configuration, so no variant is scored at a
#: different resolution or with a different amount of padding waste.
EVAL_BATCH_SIZE = 8

#: PointNeXt-S at the same width V2 used, so the backbone is not a variable.
MODEL_WIDTH = 32

# --- GOOD proposals must participate in training -------------------------------

#: A proposal is GOOD at or above this IoU, exactly as in V2.
GOOD_IOU_THRESHOLD = 0.80
#: Target share of GOOD examples in the training batch. The library carries far more
#: bad than good rows, so without this the model never learns "do not touch a tree
#: that is already right", which is the whole point of this round.
TARGET_GOOD_FRACTION = 0.50
#: A GOOD example may be repeated at most this many times. Recording the realised
#: ratio is mandatory; hiding it would misrepresent what was trained on.
MAX_GOOD_REPEAT = 3

# --- Conservative gate ---------------------------------------------------------

#: An edit counts as worth making when it improves the mask by at least this much.
GATE_BENEFIT_IOU = 0.005
#: Gate probability above which the round is willing to edit automatically.
GATE_MIN_PROBABILITY = 0.70
#: Coverage floor from the round's own criteria. If no threshold reaches it, the
#: criterion fails and is reported as failed; the threshold is not lowered.
GATE_MIN_COVERAGE = 0.10
#: Coverage target used when searching for the threshold on validation.
GATE_COVERAGE_TARGET = 0.30

# --- Protocol revision ---------------------------------------------------------

#: Revision 2 importance-weights the loss by the stratified sampler's inverse
#: inclusion probability. Revision 1 did not, which fitted the heads to the
#: stratified sample's class balance instead of the ROI's: measured over 200
#: training rows, the ADD-head positive rate was 0.1298 as sampled against 0.0908
#: in the population. The revision is on the record because it is a change to the
#: protocol, not a tweak made after seeing a good number: revision 1 failed with
#: the removal head collapsing to "always remove", and that failure is what
#: identified the missing correction. Revision 1 reports are kept on disk under
#: ``PILOT_V1_UNWEIGHTED_*``.
PROTOCOL_REVISION = 2
REVISION_NOTES = {
    2: ("loss is importance-weighted by the stratified sampler's inverse inclusion "
        "probability, so the effective training distribution matches the ROI"),
    1: ("pilot attempt with an unweighted loss; retained as PILOT_V1_UNWEIGHTED_* "
        "for audit"),
}

# --- Verdict thresholds, from the round's own criteria -------------------------

CRITERIA = {
    'CROSS_SITE_BAD_MEAN_DELTA_IOU': {'op': '>', 'value': 0.0},
    'CROSS_SITE_GOOD_MEAN_DELTA_IOU': {'op': '>=', 'value': -0.005},
    'CROSS_SITE_GOOD_DEGRADATION_RATE': {'op': '<=', 'value': 0.05},
    'CROSS_SITE_AUTO_EDIT_SUCCESS_RATE': {'op': '>=', 'value': 0.80},
    'CROSS_SITE_AUTO_EDIT_COVERAGE': {'op': '>=', 'value': 0.10},
}
#: Criteria 1-3 decide between NO_GO and BORDERLINE; 4-5 between BORDERLINE and GO.
CORE_CRITERIA = ('CROSS_SITE_BAD_MEAN_DELTA_IOU',
                 'CROSS_SITE_GOOD_MEAN_DELTA_IOU',
                 'CROSS_SITE_GOOD_DEGRADATION_RATE')
EDIT_CRITERIA = ('CROSS_SITE_AUTO_EDIT_SUCCESS_RATE',
                 'CROSS_SITE_AUTO_EDIT_COVERAGE')

#: A GOOD proposal counts as degraded when the edit costs it at least this much IoU.
GOOD_DEGRADATION_IOU = 0.01
#: A BAD proposal counts as improved when the edit gains it at least this much IoU.
BAD_IMPROVEMENT_IOU = 0.01


def config_digest() -> str:
    payload = {
        'schema': SCHEMA, 'round': ROUND, 'base_head': BASE_HEAD,
        'seed': SEED, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
        'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE,
        'points_per_example': POINTS_PER_EXAMPLE,
        'eval_max_points': EVAL_MAX_POINTS,
        'eval_batch_size': EVAL_BATCH_SIZE, 'model_width': MODEL_WIDTH,
        'good_iou_threshold': GOOD_IOU_THRESHOLD,
        'target_good_fraction': TARGET_GOOD_FRACTION,
        'max_good_repeat': MAX_GOOD_REPEAT,
        'gate_benefit_iou': GATE_BENEFIT_IOU,
        'gate_min_probability': GATE_MIN_PROBABILITY,
        'gate_min_coverage': GATE_MIN_COVERAGE,
        'gate_coverage_target': GATE_COVERAGE_TARGET,
        'criteria': CRITERIA,
        'core_criteria': list(CORE_CRITERIA),
        'edit_criteria': list(EDIT_CRITERIA),
        'good_degradation_iou': GOOD_DEGRADATION_IOU,
        'bad_improvement_iou': BAD_IMPROVEMENT_IOU,
        'protocol_revision': PROTOCOL_REVISION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()


def frozen_thresholds() -> dict:
    """The thresholds the report must show were fixed before unblinding."""
    return {
        'decision_threshold_on_heads': 0.5,
        'gate_min_probability': GATE_MIN_PROBABILITY,
        'gate_min_coverage': GATE_MIN_COVERAGE,
        'gate_coverage_target': GATE_COVERAGE_TARGET,
        'gate_benefit_iou': GATE_BENEFIT_IOU,
        'good_iou_threshold': GOOD_IOU_THRESHOLD,
        'good_degradation_iou': GOOD_DEGRADATION_IOU,
        'bad_improvement_iou': BAD_IMPROVEMENT_IOU,
        'protocol_revision': PROTOCOL_REVISION,
        'frozen_before_wytham': True,
        'tuned_on': 'train + validation only',
    }
