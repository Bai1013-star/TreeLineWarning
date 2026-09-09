"""PointNet baseline training entry point (binary PASS vs REVIEW_REQUIRED).

This is a formal training entry point, not a diagnostic. It refuses to run until the
Supervisor accepts Gate D, and this round did not run it.

The model is the shared pointwise-MLP + symmetric max pooling baseline from
``models.py`` (no T-Net). The binary head is primary; :func:`issue_head_interface`
documents the future optional per-issue head so the interface does not have to change.
"""
from __future__ import annotations

import argparse
import json
import sys

import gate_d_lock as lock
import gate_d_train as engine
from gate_d_provenance import load_gate_d_provenance

MODEL = 'pointnet'
ISSUE_CLASSES = ('MERGED_MULTI_TREE', 'NEIGHBOR_CONTAMINATION', 'OVER_SEGMENTED',
                 'CROWN_INCOMPLETE', 'STEM_MISSING', 'NOISE')


def issue_head_interface() -> dict:
    """Declared interface for the future optional issue head (not trained here)."""
    return {
        'status': 'reserved_not_implemented',
        'binary_head': {'name': 'qc_binary', 'classes': ['PASS', 'REVIEW_REQUIRED']},
        'future_issue_head': {'name': 'qc_issue', 'classes': list(ISSUE_CLASSES),
                              'shares_backbone': True,
                              'loss': 'auxiliary cross-entropy weighted below binary'},
        'note': 'Gate E round 1 is binary only; the head is additive, not a rewrite.',
    }


def build_config(args) -> engine.TrainConfig:
    return engine.TrainConfig(
        model=MODEL, points=args.points, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, amp=not args.no_amp,
        workers=args.workers, max_train_candidates=args.max_train_candidates,
        max_validation_candidates=args.max_validation_candidates,
        repeats=args.repeats, output_root=args.output_root, cache_root=args.cache_root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--points', type=int, default=4096, choices=[4096, 2048])
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-train-candidates', type=int, default=None)
    parser.add_argument('--max-validation-candidates', type=int, default=None)
    parser.add_argument('--output-root', default='D:/TreeQCTraining/gate_d_runs')
    parser.add_argument('--cache-root', default='D:/TreeQCTraining/gate_d_cache')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the resolved plan and the lock state, then exit')
    args = parser.parse_args(argv)

    state = lock.acceptance_state()
    if args.dry_run:
        config = build_config(args)
        print(json.dumps({'training_lock': state, 'config': config.as_row(),
                          'config_digest': config.digest(),
                          'issue_head': issue_head_interface()}, indent=2))
        return 0
    try:
        provenance = load_gate_d_provenance()
        report = engine.train(build_config(args), provenance,
                              resume_from=args.resume)
    except lock.TrainingLockedError as exc:
        print(json.dumps({'error': 'GATE_D_NOT_ACCEPTED', 'detail': str(exc),
                          'training_lock': state}, indent=2), file=sys.stderr)
        return 3
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
