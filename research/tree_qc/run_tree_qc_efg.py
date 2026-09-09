"""Gate E/F/G orchestrator: the full future validation run, written but not run.

This module implements the eight-step plan the Supervisor specified for the round after
Gate D acceptance. Every training step is gated on the Gate D acceptance marker, so
running it today refuses at step 1 with `GATE_D_NOT_ACCEPTED` instead of starting GPU
work.

Steps:
    1. PointNet baseline
    2. PointNet++ primary
    3. validation threshold selection
    4. held-out Wytham 749 blind evaluation
    5. stop early if the model clearly failed
    6. three seeds when the primary result is promising
    7. 4096 vs 2048 sampling robustness
    8. unified TREE_QC_EFG_FINAL_REPORT

`--dry-run` prints the resolved plan and the lock state without touching the GPU, and is
the only mode that may be executed while Gate D is unaccepted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gate_d_lock as lock
from gate_d_provenance import load_gate_d_provenance

HERE = Path(__file__).resolve().parent
REPORTS = HERE / 'reports'
ORCHESTRATOR_VERSION = 'TREE_QC_EFG_ORCHESTRATOR_V1'

# Step 5 gate: a primary run below these floors is not worth three seeds of GPU time.
PROMISING_BALANCED_ACCURACY = 0.70
PROMISING_BAD_RECALL = 0.60
SEEDS = (101, 202, 303)
POINT_COUNTS = (4096, 2048)


def plan(epochs: int, batch_size: int, max_train_candidates: int | None,
         max_validation_candidates: int | None, seeds=SEEDS) -> dict:
    """The full run plan as data, so it can be reviewed before it is executed."""
    return {
        'orchestrator_version': ORCHESTRATOR_VERSION,
        'steps': [
            {'step': 1, 'name': 'pointnet_baseline', 'model': 'pointnet',
             'seed': seeds[0], 'epochs': epochs, 'batch_size': batch_size,
             'trainer': 'train_pointnet.py'},
            {'step': 2, 'name': 'pointnet2_primary', 'model': 'pointnet2_ssg',
             'seed': seeds[0], 'epochs': epochs, 'batch_size': batch_size,
             'trainer': 'train_pointnet2.py'},
            {'step': 3, 'name': 'validation_threshold_selection',
             'partition': 'validation', 'held_out_used': False,
             'note': 'frozen operating point comes from model-selection data only'},
            {'step': 4, 'name': 'held_out_blind_evaluation', 'partition': 'held_out',
             'expected_candidates': 749, 'policy': 'STRICT_HIGH_CONTAINMENT_QUARANTINE',
             'note': 'evaluated once, after the threshold is frozen'},
            {'step': 5, 'name': 'early_stop_check',
             'min_balanced_accuracy': PROMISING_BALANCED_ACCURACY,
             'min_bad_recall': PROMISING_BAD_RECALL,
             'note': 'a clearly failed model stops here instead of burning GPU'},
            {'step': 6, 'name': 'multi_seed', 'seeds': list(seeds[1:]),
             'note': 'only if the primary result is promising'},
            {'step': 7, 'name': 'sampling_robustness', 'points': list(POINT_COUNTS),
             'note': '4096 primary vs 2048 robustness'},
            {'step': 8, 'name': 'final_report',
             'artifact': 'TREE_QC_EFG_FINAL_REPORT'},
        ],
        'partitions': {'train': 'model_selection', 'validation': 'threshold_selection',
                       'held_out': 'blind_evaluation_only'},
        'max_train_candidates': max_train_candidates,
        'max_validation_candidates': max_validation_candidates,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-train-candidates', type=int, default=None)
    parser.add_argument('--max-validation-candidates', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    state = lock.acceptance_state()
    resolved = plan(args.epochs, args.batch_size, args.max_train_candidates,
                    args.max_validation_candidates)
    if args.dry_run:
        print(json.dumps({'training_lock': state, 'plan': resolved}, indent=2))
        return 0

    try:
        lock.require_training_allowed()
    except lock.TrainingLockedError as exc:
        print(json.dumps({'error': 'GATE_D_NOT_ACCEPTED', 'detail': str(exc),
                          'training_lock': state,
                          'message': 'Gate D has not been accepted; the E/F/G run is '
                                     'locked. Run 01_RUN_GATE_D_DIAGNOSTICS.cmd and '
                                     'obtain Supervisor acceptance first.'},
                         indent=2), file=sys.stderr)
        return 3

    # From here the run is authorised. The steps below are intentionally not executed
    # in this round: Gate D acceptance has not been granted, and the implementation is
    # delivered for review, not for a live experiment.
    provenance = load_gate_d_provenance()
    print(json.dumps({'orchestrator_version': ORCHESTRATOR_VERSION,
                      'protocol': provenance.protocol,
                      'held_out_candidates': len(provenance.held_out_eligible),
                      'plan': resolved,
                      'note': 'Gate E/F/G execution is not part of this round; run the '
                              'steps through the individual launchers.'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
