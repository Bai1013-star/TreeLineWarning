"""Run TREE_QC_V25_FINAL_DEEP_LEARNING_GO_NO_GO.

Three results, and only three:

``A``  the frozen V2 full-mask refiner, read from the V2 artefacts, not retrained
``B``  the PointNeXt-S delta refiner, no gate
``C``  the same delta refiner behind the conservative gate

The order of operations is the research claim of this round, so it is fixed in
code: train, score validation, fit the gate on validation, **freeze and digest the
thresholds**, and only then score the test split with thresholds re-read from the
frozen file.
"""
from __future__ import annotations

import json

# Reuse the frozen V2 split function so the splits are provably the same ones.
import sys
import time

import v25_config as cfg
import v25_delta_data as data
import v25_metrics as metrics
import v25_train as train

if str(cfg.V2_DIR) not in sys.path:
    sys.path.insert(0, str(cfg.V2_DIR))
import v2_refiner_train as v2_train  # noqa: E402


def v2_frozen_baseline() -> dict:
    """Result A: read the V2 numbers from the V2 artefacts. Nothing is re-run."""
    path = cfg.V2_DIR / 'reports' / 'MINI_REFINER_BAKEOFF.json'
    if not path.exists():
        return {'status': 'MISSING', 'path': str(path)}
    payload = json.loads(path.read_text(encoding='utf-8'))
    out = {'status': 'OK', 'source': 'research/tree_qc_v2/reports/'
                                     'MINI_REFINER_BAKEOFF.json',
           'baseline_is_retrained_this_round': False,
           'base_head': cfg.BASE_HEAD, 'configurations': {}}
    for fold in payload['folds']:
        configuration = fold['configuration']
        models = {}
        for key, model in sorted(fold['models'].items()):
            summary = model['summary']
            models[key] = {
                'name': model['model'],
                'parameters': model['parameters'],
                'checkpoint_digest': model['checkpoint_digest'],
                'delta_iou_mean': summary['delta_iou_mean'],
                'delta_iou_median': summary['delta_iou_median'],
                'current_iou_mean': summary['current_iou_mean'],
                'refined_iou_mean': summary['refined_iou_mean'],
                'change_region_f1_mean': summary['change_region_f1_mean'],
                'good_mean_delta_iou': summary['good_proposals']['mean_delta_iou'],
                'good_degradation_rate': summary['good_proposals']['degraded_fraction'],
                'bad_mean_delta_iou': summary['bad_proposals']['mean_delta_iou'],
                'bad_improvement_rate': summary['bad_proposals']['improved_fraction'],
                'identity_copy_fraction': summary['identity_copy_fraction'],
            }
        out['configurations'][configuration] = {'models': models}
    return out


def run_fold(index: dict, label: str, split, verbose: bool = True) -> dict:
    started = time.time()
    print(f'[V25] {label}: train {len(split.train)} validation '
          f'{len(split.validation)} test {len(split.test)}', flush=True)
    trained = train.train_delta(index, split, verbose=verbose)
    model, device = trained['model'], trained['device']

    print('   scoring validation (gate fitting data)', flush=True)
    validation_rows = train.score_rows(
        model, index, data.build_examples(index, split.validation), device,
        verbose=verbose)
    gate = train.fit_gate(validation_rows)
    frozen_path = cfg.REPORT_DIR / f'FROZEN_GATE_{label}.json'
    digest = train.freeze_gate(gate, frozen_path)
    print(f'   thresholds frozen to {frozen_path.name} sha256={digest[:16]}',
          flush=True)
    # Re-read from disk: the test path may not use the in-memory copy.
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if frozen['gate']['effective_threshold'] != gate['effective_threshold']:
        raise RuntimeError('frozen gate does not match the fitted gate')

    print('   scoring test (thresholds already frozen)', flush=True)
    test_rows = train.score_rows(
        model, index, data.build_examples(index, split.test), device, verbose=verbose)

    validation_gated = train.apply_gate(validation_rows, frozen['gate'])
    test_gated = train.apply_gate(test_rows, frozen['gate'])

    fold = {
        'label': label,
        'configuration': label,
        'split': split.as_dict(),
        'training': trained['training'],
        'gate': frozen['gate'],
        'gate_frozen_sha256': digest,
        'gate_frozen_file': str(frozen_path),
        'thresholds_frozen_before_test': True,
        'frozen_thresholds': frozen['frozen_thresholds'],
        'validation': {
            'ungated': metrics.summarise(validation_rows, 'validation_ungated'),
            'gated': metrics.summarise_gated(validation_gated, 'validation_gated'),
        },
        'test': {
            'ungated': metrics.summarise(test_rows, 'test_ungated'),
            'gated': metrics.summarise_gated(test_gated, 'test_gated'),
        },
        'seconds': round(time.time() - started, 2),
        'device': device,
    }
    metrics.write_rows(cfg.REPORT_DIR / f'V25_ROWS_{label}.json', test_rows,
                       test_gated, validation_rows, validation_gated)
    print(f'   done in {fold["seconds"]:.0f}s', flush=True)
    return fold


def _archive_previous_run() -> str | None:
    """Move an earlier attempt's reports aside instead of overwriting them.

    The round has exactly one protocol revision on the record. The first attempt
    trained the heads without correcting the stratified sampler's class balance,
    which biases the decision toward the over-sampled class; its reports are kept
    under a PILOT name so the revision is auditable rather than invisible.
    """
    naming = {
        'V25_GO_NO_GO.json': 'PILOT_V1_UNWEIGHTED_GO_NO_GO.json',
        'V25_OVERFIT_PROBE.json': 'PILOT_V1_UNWEIGHTED_OVERFIT_PROBE.json',
    }
    moved = []
    for source, target in naming.items():
        path = cfg.REPORT_DIR / source
        if path.exists() and not (cfg.REPORT_DIR / target).exists():
            path.replace(cfg.REPORT_DIR / target)
            moved.append(target)
    for label in ('cross_site_test_wytham', 'in_site_test_litchfield'):
        path = cfg.REPORT_DIR / f'V25_ROWS_{label}.json'
        target = cfg.REPORT_DIR / f'PILOT_V1_UNWEIGHTED_ROWS_{label}.json'
        if path.exists() and not target.exists():
            path.replace(target)
            moved.append(target.name)
    return moved[0] if moved else None


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--configurations', nargs='*',
                        default=['cross_site_test_wytham', 'in_site_test_litchfield'])
    parser.add_argument('--skip-insite', action='store_true')
    args = parser.parse_args()

    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    archived = _archive_previous_run()
    if archived:
        print(f'previous attempt archived as {archived}', flush=True)
    index = data.load_library()
    # Compute the digest of the index file itself rather than reading a field out of
    # it: the V2 artefact records that digest, the live index does not, and a value
    # taken from the thing being described is not a check on it.
    library_digest = data.file_digest(cfg.ROI_DIR / 'index.json')
    print(f'library: {index["target_count"]} targets, index digest '
          f'{library_digest[:16]}', flush=True)

    payload = {
        'schema': cfg.SCHEMA, 'round': cfg.ROUND, 'config_digest': cfg.config_digest(),
        'base_head': cfg.BASE_HEAD,
        'frozen_thresholds': cfg.frozen_thresholds(),
        'criteria': cfg.CRITERIA,
        'model': {'architecture': 'PointNeXt-S backbone (imported from the frozen '
                                  'V2 code) + REMOVE head + ADD head',
                  'parameter_breakdown': None},
        'library': {'target_count': index['target_count'],
                    'targets_per_site': index['targets_per_site'],
                    'index_sha256': library_digest,
                    'index_path': str(cfg.ROI_DIR / 'index.json')},
        'result_a_v2_frozen_full_mask': v2_frozen_baseline(),
        'folds': [],
    }

    labels = list(args.configurations)
    for label in labels:
        if label.startswith('cross_site'):
            site = label.replace('cross_site_test_', '')
            split = v2_train.split_by_tree(index, test_site=site, seed=cfg.SEED)
        else:
            site = label.replace('in_site_test_', '')
            split = v2_train.split_within_site(index, site, cfg.SEED)
        if args.skip_insite and label.startswith('in_site'):
            continue
        payload['folds'].append(run_fold(index, label, split))

    payload['verdict'] = metrics.verdict(payload)
    out = cfg.REPORT_DIR / 'V25_GO_NO_GO.json'
    out.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(f'wrote {out}')
    print(json.dumps(payload['verdict'], indent=2)[:1600])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
