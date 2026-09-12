"""Frozen knobs for the Tree QC V2 feasibility audit.

Everything a later stage reads is declared here once, so the report can quote the
exact configuration that produced it. Nothing in this module touches the frozen
V1 Gate D / EFG artefacts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'tree_qc_v2_feasibility_v1'

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / 'research' / 'tree_qc_v2'
REPORT_DIR = RESEARCH_DIR / 'reports'

DATA_ROOT = Path(os.environ.get('TREE_QC_V2_DATA_ROOT', r'D:\TreeQCData'))

#: Published per-tree instance clouds, verified against the official MD5 of the
#: pinned Zenodo record. Only these two sites are present on this machine.
SITE_TREE_ROOT = {
    'litchfield': DATA_ROOT / 'extracted_manual_verified' / 'litchfield' / 'trees',
    'wytham': DATA_ROOT / 'extracted_manual_verified' / 'wytham' / 'trees',
}

#: Plot-level clouds, needed only for the ground/reference frame.
SITE_PLOT_ROOT = {
    'litchfield': DATA_ROOT / 'extracted_manual_verified' / 'litchfield',
    'wytham': DATA_ROOT / 'extracted_manual_verified' / 'wytham',
}

#: Sites the audit was asked to cover but which are not present locally.
ABSENT_SITES = ('ofental', 'robsoncreek')

WORK_ROOT = DATA_ROOT / 'v2_feasibility'
CACHE_DIR = WORK_ROOT / 'cache'
RUN_DIR = WORK_ROOT / 'runs'

#: Where the ROI library for Task C is written.
ROI_DIR = WORK_ROOT / 'roi_library'
#: Window grid for the ROI library: several local windows per site instead of one.
ROI_WINDOW_M = 12.0
ROI_WINDOW_GRID_M = 20.0
ROI_WINDOWS_PER_SITE = 22
#: A target keeps all points above this count; larger clouds are strided down for
#: the library, and the stride is recorded per row.
ROI_STORE_MAX_POINTS = 120_000

#: Every experiment uses this seed unless a per-model seed is declared.
SEED = 101

# --- Task A: site fingerprint -------------------------------------------------

#: Point budget per tree for the site classifier.
SITE_CLASSIFIER_POINTS = 1024
#: Trees sampled per site per split, so the classifier cannot learn a class prior.
SITE_CLASSIFIER_TREES_PER_SITE = 300
SITE_CLASSIFIER_EPOCHS = 60
SITE_CLASSIFIER_LR = 1e-3
SITE_CLASSIFIER_HIDDEN = 128
SITE_CLASSIFIER_BATCH = 32
#: Train/validation/test split of the good-tree pool, by tree, seeded.
SITE_CLASSIFIER_SPLIT = (0.6, 0.2, 0.2)

# --- Task B: proposal construction -------------------------------------------

#: Per-site tree instances entering the local proposal test.
PROPOSAL_TREES_PER_SITE = 40
#: Side of the local forest window the real algorithm is run on.
PROPOSAL_WINDOW_M = 12.0
#: An instance must hold at least this many points inside the window to be a
#: meaningful refinement target. Small fragments are excluded and counted.
PROPOSAL_MIN_TARGET_POINTS = 8000
#: A target must hold at least this fraction of its own instance inside the
#: window, otherwise it is a boundary fragment and its mask is not the truth.
PROPOSAL_MIN_TARGET_CONTAINMENT = 0.90
#: ROI: the candidate's own bounding box grown by these metres in XY.
ROI_XY_PAD_M = 1.5
#: ROI floor as a fraction of candidate height below the candidate minimum Z.
ROI_Z_FLOOR_FRACTION = 0.10
ROI_Z_CEILING_PAD_M = 0.5
#: Ground estimate grid for a public plot (no product ground classifier exists here).
GROUND_GRID_M = 1.0

# --- Task C: mini refiner bakeoff ---------------------------------------------

#: Points per ROI fed to a refiner during training.
REFINER_POINTS = 2048
#: Epoch budget. Shared by every architecture: this is the number the fairness
#: claim in the protocol refers to, so no model may be given a different one.
REFINER_EPOCHS = 20
#: Cap on the points a single ROI contributes to the *evaluation* pass. The
#: largest ROIs in the library reach 1.5 M points and a full 18-proposal sweep over
#: 118 targets at that size is hours of GPU time for no extra information, so the
#: evaluation applies one deterministic spatially-sorted stride per ROI. The stride
#: and the pre-stride point count are recorded on every scored row, so a reader can
#: see exactly what was scored and re-run any row uncapped.
REFINER_EVAL_MAX_POINTS = 200_000
REFINER_EPOCHS = 40
REFINER_LR = 1e-3
REFINER_BATCH = 8
REFINER_EARLY_STOP_PATIENCE = 8
REFINER_EARLY_STOP_MIN_DELTA = 1e-4

#: Cross-site configuration the audit protocol fixes.
CROSS_SITE_TRAIN = ('litchfield', 'wytham')
CROSS_SITE_TEST = ('wytham', 'litchfield')

#: GOOD-preservation thresholds, declared before the runs.
GOOD_IOU_THRESHOLD = 0.80
GOOD_DEGRADED_IOU_DROP = 0.05
GOOD_SEVERE_IOU_DROP = 0.20

#: Identity-copy gate: an identity-copying refiner has near-zero change-region
#: recall. Declared before the runs.
IDENTITY_COPY_CHANGE_F1_FLOOR = 0.10
IDENTITY_COPY_IMPROVEMENT_IOU_FLOOR = 0.02


def config_digest() -> str:
    payload = {
        'schema': SCHEMA,
        'seed': SEED,
        'site_classifier': {
            'points': SITE_CLASSIFIER_POINTS,
            'trees_per_site': SITE_CLASSIFIER_TREES_PER_SITE,
            'epochs': SITE_CLASSIFIER_EPOCHS,
            'lr': SITE_CLASSIFIER_LR,
            'hidden': SITE_CLASSIFIER_HIDDEN,
            'batch': SITE_CLASSIFIER_BATCH,
            'split': list(SITE_CLASSIFIER_SPLIT),
        },
        'proposal': {
            'trees_per_site': PROPOSAL_TREES_PER_SITE,
            'roi_xy_pad_m': ROI_XY_PAD_M,
            'roi_z_floor_fraction': ROI_Z_FLOOR_FRACTION,
            'roi_z_ceiling_pad_m': ROI_Z_CEILING_PAD_M,
            'ground_grid_m': GROUND_GRID_M,
        },
        'refiner': {
            'points': REFINER_POINTS,
            'epochs': REFINER_EPOCHS,
            'lr': REFINER_LR,
            'batch': REFINER_BATCH,
            'patience': REFINER_EARLY_STOP_PATIENCE,
            'min_delta': REFINER_EARLY_STOP_MIN_DELTA,
        },
        'cross_site_train': list(CROSS_SITE_TRAIN),
        'good_iou_threshold': GOOD_IOU_THRESHOLD,
        'good_degraded_iou_drop': GOOD_DEGRADED_IOU_DROP,
        'good_severe_iou_drop': GOOD_SEVERE_IOU_DROP,
        'identity_copy_change_f1_floor': IDENTITY_COPY_CHANGE_F1_FLOOR,
        'identity_copy_improvement_iou_floor': IDENTITY_COPY_IMPROVEMENT_IOU_FLOOR,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
