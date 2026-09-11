"""Pre-declared Public Pilot gates, thresholds and decision constants.

These values are declared once, before any result exists, and every other EFG module
imports them rather than restating them. That is the point: a gate that is written down in
more than one place can drift, and a drifted gate is a changed experiment.

Nothing here can be edited in response to a result. The orchestrator freezes these values
into ``PUBLIC_EXPERIMENT_FREEZE.json`` before held-out is unlocked, and reports the frozen
copy rather than recomputing targets after the fact.
"""
from __future__ import annotations

GATES_VERSION = 'EFG_GATES_V1'

# --- label semantics (frozen Gate D) ---------------------------------------
PASS = 0
REVIEW_REQUIRED = 1
LABEL_NAMES = ('PASS', 'REVIEW_REQUIRED')

GOOD_CLASS = 'GOOD'
DEFECT_CLASSES = ('MERGED_MULTI_TREE', 'NEIGHBOR_CONTAMINATION', 'OVER_SEGMENTED',
                  'CROWN_INCOMPLETE', 'STEM_MISSING', 'NOISE')
CLASSES = (GOOD_CLASS, *DEFECT_CLASSES)
SEVERITIES = ('LOW', 'MEDIUM', 'HIGH')
PRIMARY_SEVERITIES = ('MEDIUM', 'HIGH')

# --- training budget (frozen plan) -----------------------------------------
EPOCHS = 30
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
POINTS_PRIMARY = 4096
POINTS_ROBUSTNESS = 2048
AMP_WHEN_CUDA = True
AMP_DTYPE = 'float16'
GRAD_CLIP = 1.0
WORKERS = 0
GOOD_FALSE_REVIEW_BUDGET = 0.25

MODEL_SEEDS = (101, 202, 303)
COMPARISON_SEED = 101
ARCHITECTURES = ('pointnet', 'pointnet2_ssg')
PRIMARY_ARCHITECTURE_CANDIDATE = 'pointnet2_ssg'

# --- metric gates ----------------------------------------------------------
PRIMARY_GATE = {
    'BAD_RECALL_min': 0.85,
    'BAD_FALSE_PASS_max': 0.15,
    'GOOD_FALSE_REVIEW_max': 0.25,
    'BALANCED_ACCURACY_min': 0.80,
}
LEARNABILITY_GATE = {'BALANCED_ACCURACY_min': 0.70, 'BAD_RECALL_min': 0.60}
MODEL_STABILITY_MAX_SPREAD = 0.10
SAMPLING_MAX_RECALL_DROP = 0.10
NON_INFERIORITY_MARGIN = 0.01
UNCERTAIN_BAND = 0.10

# Sampling seeds for the robustness sweep. Declared here, frozen into the experiment
# freeze, and never chosen after seeing a held-out number.
ROBUSTNESS_SAMPLING_SEEDS = (11, 23, 37)
ROBUSTNESS_POINTS = (POINTS_PRIMARY, POINTS_ROBUSTNESS)

# --- resource guards -------------------------------------------------------
DISK_GUARD_GIB = 35
PROBE_BATCHES = (16, 8, 4, 2, 1)

# --- verdict vocabulary ----------------------------------------------------
VERDICT_NOT_LEARNABLE = 'PUBLIC_PILOT_NOT_LEARNABLE'
VERDICT_UNSTABLE = 'MODEL_UNSTABLE'
VERDICT_SAMPLING_TOO_SENSITIVE = 'SAMPLING_TOO_SENSITIVE'
VERDICT_READY = 'READY_FOR_GATE_H_LOCAL_BLIND'
PRODUCT_READY_FORBIDDEN = 'PRODUCT_READY'


def f1_review(metrics: dict) -> float:
    """F1 of the REVIEW_REQUIRED class, from the frozen confusion matrix.

    The reporting layer is explicitly allowed to compute this; the frozen Gate D metric
    core is not modified to add it.
    """
    row = metrics.get('CONFUSION_MATRIX') or {}
    true_positive = int(row.get('tp_review_review', 0))
    false_negative = int(row.get('fn_review_as_pass', 0))
    false_positive = int(row.get('fp_pass_as_review', 0))
    denominator = 2 * true_positive + false_negative + false_positive
    return round(float(2 * true_positive / denominator), 6) if denominator else 0.0


def public_pilot_gates() -> dict:
    """The frozen gate block that goes into the experiment freeze and the final report."""
    return {
        'gates_version': GATES_VERSION,
        'primary_metrics': ['BAD_RECALL', 'BAD_FALSE_PASS', 'GOOD_FALSE_REVIEW',
                            'BALANCED_ACCURACY', 'F1_REVIEW'],
        'targets': dict(PRIMARY_GATE),
        'learnability_gate': dict(LEARNABILITY_GATE),
        'primary_severities': list(PRIMARY_SEVERITIES),
        'model_stability_max_bad_recall_spread': MODEL_STABILITY_MAX_SPREAD,
        'sampling_max_bad_recall_drop': SAMPLING_MAX_RECALL_DROP,
        'uncertain_band': UNCERTAIN_BAND,
        'uncertain_policy': 'UNCERTAIN is reported alongside the binary decision and is '
                            'never counted as a detected defect',
        'required_defect_classes': list(DEFECT_CLASSES),
        'required_meaningful_defects': ['MERGED_MULTI_TREE', 'NEIGHBOR_CONTAMINATION',
                                        'OVER_SEGMENTED'],
        'note': 'targets are continuation criteria for Gate H, not product claims',
    }


def robustness_plan() -> dict:
    """The sampling robustness configuration, fixed before held-out is opened."""
    configurations = []
    for points in ROBUSTNESS_POINTS:
        seeds = ROBUSTNESS_SAMPLING_SEEDS if points == POINTS_ROBUSTNESS else (0,)
        for seed in seeds:
            configurations.append({'label': f'{points}_sampling_seed_{seed}',
                                   'points': int(points), 'sampling_seed': int(seed),
                                   'primary': points == POINTS_PRIMARY and seed == 0})
    return {
        'points_primary': POINTS_PRIMARY,
        'points_robustness': POINTS_ROBUSTNESS,
        'sampling_seeds': list(ROBUSTNESS_SAMPLING_SEEDS),
        'configurations': configurations,
        'max_bad_recall_drop': SAMPLING_MAX_RECALL_DROP,
        'rule': 'a BAD_RECALL drop greater than 10 percentage points against the 4096 '
                'primary configuration is SAMPLING_TOO_SENSITIVE',
        'declared_before_heldout': True,
    }
