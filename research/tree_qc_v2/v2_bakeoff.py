"""Run the MINI_REFINER_BAKEOFF: three architectures, one protocol.

What is shared, and therefore cannot explain a difference:

* the same train, validation and test candidates;
* the same ROI point clouds and the same proposal pairs;
* the same sampler, augmentation, point target and seed;
* the same epoch budget, early-stop policy, optimiser family and learning rate;
* the same evaluation code and decision threshold.

What differs is the architecture.

The cross-site configuration the protocol fixes is: train on the held-in site's
trees, validate on held-in trees only, and test on a site the model never saw. The
test site is never used for early stopping, normalisation, augmentation or
thresholds. Two folds are run - each site held out in turn - and both are reported.

Every reported IoU is computed over the **whole ROI**, not the 2048-point training
sample, so a sampled subset cannot decide the verdict for a 300 000-point region.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import v2_config as cfg
import v2_models as models
import v2_refiner_data as refdata
import v2_refiner_train as trainer
import v2_roi_library as library

MODEL_KEYS = ('A', 'B', 'C')


def _checkpoint_digest(model) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:32]


def _release_gpu(model=None) -> None:
    """Drop a model and return its memory to the driver before the next one.

    Without this, the caching allocator stays fragmented after a model is
    discarded: the next architecture then runs with only a few hundred megabytes
    of headroom, and on an 8 GB card its epoch time was measured at 8.5 minutes
    against 47 seconds for the same architecture in a clean process. Releasing the
    cache between models is what makes the shared epoch budget a fair comparison
    rather than a statement about allocator state.
    """
    import gc
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def train_and_evaluate(index: dict, test_site: str, points: int, epochs: int,
                       seed: int, device: str, verbose: bool = True,
                       kinds=None, split=None, label: str = None) -> dict:
    """Train every architecture on one configuration, then score it on full ROIs.

    ``split`` can be supplied to run the in-site configuration; otherwise the
    cross-site split that holds ``test_site`` out is used.
    """
    started = time.time()
    if split is None:
        split = trainer.split_by_tree(index, test_site=test_site, seed=seed)
    held_in = sorted({row['site'] for row in index['targets']
                      if row['target_id'] in set(split.train)})
    if label is None:
        label = f'cross_site_test_{test_site}'
    configuration = label
    if verbose:
        print(f'[C] {configuration}: train {len(split.train)} '
              f'validation {len(split.validation)} test {len(split.test)} '
              f'(held-in sites {held_in}, test site {test_site})', flush=True)

    fold = {
        'configuration': configuration,
        'test_site': test_site, 'held_in_sites': held_in,
        'split': split.as_dict(),
        'test_site_never_used_for_selection': True,
        'models': {},
    }
    for key in MODEL_KEYS:
        if verbose:
            print(f'  --- training {key}', flush=True)
        _release_gpu()
        torch.manual_seed(seed)
        np.random.seed(seed)
        trained, model = trainer.train_one(key, index, split, points, epochs, seed,
                                           device, verbose=verbose, kinds=kinds)
        if verbose:
            print(f'      scoring {len(split.test)} test target(s) over full ROIs',
                  flush=True)
        rows = trainer.evaluate_full_roi(model, index, split.test, kinds, device,
                                         seed=seed)
        rows_held_in = trainer.evaluate_full_roi(model, index, split.validation, kinds,
                                                 device, seed=seed)
        summary = trainer.summarise(rows, f'{key}:{configuration}')
        summary['by_corruption'] = trainer.by_group(rows, 'corruption')
        summary['by_severity'] = trainer.by_group(rows, 'severity')
        summary['by_proposal_kind'] = trainer.by_group(rows, 'proposal_kind')
        entry = {
            **trained,
            'summary': summary,
            'identity_copy_gate': trainer.identity_copy_gate(summary),
            'good_preservation_gate': trainer.good_preservation_gate(summary),
            'held_in_validation': trainer.summarise(rows_held_in, 'held_in'),
            'resources': trainer.measure_resources(model, device, points=points),
            'cpu_probe': trainer.cpu_inference_probe(model, points=points),
            'checkpoint_digest': _checkpoint_digest(model),
        }
        # Per-row detail stays on disk; the aggregate payload stays readable.
        entry['rows_file'] = f'BAKEOFF_ROWS_{test_site}_{key}.json'
        fold['models'][key] = entry
        if verbose:
            gate = entry['identity_copy_gate']
            good = entry['good_preservation_gate']
            print(f'      delta_iou mean {summary["delta_iou_mean"]:+.4f} '
                  f'median {summary["delta_iou_median"]:+.4f} '
                  f'change_f1 {summary["change_region_f1_mean"]:.4f} '
                  f'identity_copy={gate["verdict"]} '
                  f'good_preservation={good["verdict"]}', flush=True)
        _write_rows(rows, rows_held_in, configuration, key)
        _release_gpu(model)
    fold['seconds'] = round(time.time() - started, 2)
    return fold


def _write_rows(rows, held_in_rows, configuration: str, key: str) -> None:
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = cfg.REPORT_DIR / f'BAKEOFF_ROWS_{configuration}_{key}.json'
    target.write_text(json.dumps({'configuration': configuration, 'model_key': key,
                                  'test_rows': rows, 'held_in_rows': held_in_rows},
                                 indent=2), encoding='utf-8')


def verdict(folds: list[dict], model_keys=MODEL_KEYS) -> dict:
    """Which candidates survive both gates on an unseen site, for both folds."""
    out = {'per_model': {}, 'per_fold': {}}
    for key in model_keys:
        scores = []
        for fold in folds:
            entry = fold['models'][key]
            summary = entry['summary']
            bad = summary.get('bad_proposals') or {}
            scores.append({
                'test_site': fold['test_site'],
                'delta_iou_mean': summary['delta_iou_mean'],
                'bad_mean_delta_iou': bad.get('mean_delta_iou'),
                'bad_improvement_gt_0.02': bad.get('improvement_gt_0.02'),
                'change_region_f1_mean': summary['change_region_f1_mean'],
                'identity_copy_gate': entry['identity_copy_gate']['verdict'],
                'good_preservation_gate': entry['good_preservation_gate']['verdict'],
                'parameters': entry['parameters'],
            })
        good_folds = [s for s in scores
                      if s['identity_copy_gate'] == 'PASS'
                      and s['good_preservation_gate'] == 'PASS'
                      and (s['bad_mean_delta_iou'] or 0) > 0]
        out['per_model'][key] = {
            'folds': scores,
            'folds_passing_both_gates_with_bad_improvement': len(good_folds),
            'passes': bool(len(good_folds) == len(scores)),
        }
    for fold in folds:
        out['per_fold'][fold['test_site']] = {
            key: {'delta_iou_mean': fold['models'][key]['summary']['delta_iou_mean'],
                  'change_region_f1_mean':
                      fold['models'][key]['summary']['change_region_f1_mean'],
                  'identity_copy': fold['models'][key]['identity_copy_gate']['verdict'],
                  'good_preservation':
                      fold['models'][key]['good_preservation_gate']['verdict']}
            for key in model_keys}
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=cfg.REFINER_POINTS)
    parser.add_argument('--epochs', type=int, default=cfg.REFINER_EPOCHS)
    parser.add_argument('--seed', type=int, default=cfg.SEED)
    parser.add_argument('--folds', nargs='*', default=['wytham', 'litchfield'])
    parser.add_argument('--in-site', nargs='*', default=['litchfield'],
                        help='sites for the in-site diagnostic configuration')
    parser.add_argument('--no-in-site', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    points = 512 if args.quick else args.points
    epochs = 4 if args.quick else args.epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index = library.load_library()
    payload = {
        'schema': cfg.SCHEMA, 'task': 'MINI_REFINER_BAKEOFF',
        'config_digest': cfg.config_digest(), 'seed': args.seed,
        'points': points, 'epochs': epochs, 'device': device,
        'library_index_sha256': index.get('index_sha256'),
        'target_count': index['target_count'],
        'targets_per_site': index['targets_per_site'],
        'feature_channels': list(refdata.FEATURE_CHANNELS),
        'model_descriptions': models.describe_models(len(refdata.FEATURE_CHANNELS)),
        'shared_protocol': {
            'optimiser': 'Adam', 'learning_rate': cfg.REFINER_LR,
            'scheduler': 'cosine annealing over the epoch budget',
            'batch_size': cfg.REFINER_BATCH,
            'early_stop_patience': cfg.REFINER_EARLY_STOP_PATIENCE,
            'early_stop_min_delta': cfg.REFINER_EARLY_STOP_MIN_DELTA,
            'decision_threshold': trainer.DECISION_THRESHOLD,
            'selection_metric': 'validation loss on held-in trees only',
            'augmentation': 'random Z rotation + 1 cm jitter, identical for all models',
            'sampler': 'stratified proposal / target-only / other-tree / background',
            'evaluation': 'full ROI, not the training sample',
            'loss': ('balanced BCE; MODEL_C adds offset L1 on target points '
                     'and mask-quality MSE'),
        },
        'folds': [],
        'evaluation_cap': {
            'max_points_per_roi': cfg.REFINER_EVAL_MAX_POINTS,
            'meaning': ('each ROI is scored up to this many points by one '
                        'deterministic spatially-sorted stride; the per-row stride '
                        'and pre-stride point count are recorded on every row'),
        },
    }
    for test_site in args.folds:
        payload['folds'].append(
            train_and_evaluate(index, test_site, points, epochs, args.seed, device,
                               verbose=True))
    # The in-site diagnostic: the same architectures and the same budget on one
    # forest, holding out trees from that forest. It separates "cannot refine at
    # all" from "cannot transfer", and it is reported next to the cross-site folds
    # rather than instead of them.
    if not args.no_in_site:
        for site in args.in_site:
            payload['folds'].append(train_and_evaluate(
                index, site, points, epochs, args.seed, device, verbose=True,
                split=trainer.split_within_site(index, site, args.seed),
                label=f'in_site_test_{site}'))
    payload['verdict'] = verdict(payload['folds'])
    payload['cross_site_folds'] = [fold['configuration'] for fold in payload['folds']
                                   if fold['configuration'].startswith('cross_site')]
    payload['in_site_folds'] = [fold['configuration'] for fold in payload['folds']
                                if fold['configuration'].startswith('in_site')]
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = '_QUICK' if args.quick else ''
    out = cfg.REPORT_DIR / f'MINI_REFINER_BAKEOFF{suffix}.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
