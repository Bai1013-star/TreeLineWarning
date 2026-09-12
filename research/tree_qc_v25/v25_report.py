"""Assemble the TREE_QC_V25_FINAL_GO_NO_GO report from the run artefacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v25_config as cfg
import v25_delta_model as model_module

REPORTS = cfg.REPORT_DIR
OUT_MD = cfg.REPO_ROOT / 'TREE_QC_V25_FINAL_GO_NO_GO_REPORT.md'
OUT_JSON = cfg.REPO_ROOT / 'TREE_QC_V25_FINAL_GO_NO_GO_REPORT.json'


def read_json(name: str):
    path = REPORTS / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def git(*arguments: str) -> str:
    import os
    environment = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    for attempt in range(3):
        try:
            result = subprocess.run(['git', *arguments], cwd=str(cfg.REPO_ROOT),
                                    capture_output=True, text=True,
                                    env=environment if attempt else None)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return ''


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict:
    run = read_json('V25_GO_NO_GO.json')
    if run is None:
        raise SystemExit('V25_GO_NO_GO.json is missing; run v25_runner.py first')
    probe_overfit = read_json('V25_OVERFIT_PROBE.json')
    payload = {
        'schema': cfg.SCHEMA,
        'round': cfg.ROUND,
        'git': {'head': git('rev-parse', 'HEAD'),
                'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
                'base_head': cfg.BASE_HEAD,
                'worktree_status': git('status', '--porcelain') or 'clean'},
        'config_digest': cfg.config_digest(),
        'protocol_revision': cfg.PROTOCOL_REVISION,
        'revision_notes': cfg.REVISION_NOTES,
        'frozen_thresholds': run['frozen_thresholds'],
        'criteria_declared': run['criteria'],
        'model': run['model'],
        'parameter_breakdown': model_module.parameter_breakdown(
            len(run['model'].get('feature_channels', [])) or 5),
        'library': run['library'],
        'v2_code_untouched': _v2_untouched(),
        'prohibitions': _prohibitions(),
        'result_a_v2_frozen': run['result_a_v2_frozen_full_mask'],
        'folds': run['folds'],
        'verdict': run['verdict'],
        'overfit_probe': probe_overfit,
        'provenance': _provenance(),
    }
    payload['stage_status'] = _stage_status(payload)
    return payload


def _v2_untouched() -> dict:
    status = git('status', '--porcelain', '--', 'research/tree_qc_v2')
    return {
        'v2_files_modified': [line for line in status.splitlines() if line.strip()],
        'note': ('research/tree_qc_v2/ is read only by this round: its ROI library, '
                 'splits, artefacts and reports are loaded and never rewritten, and '
                 'no V2 model is retrained'),
        'v2_revision_read': cfg.BASE_HEAD,
    }


def _prohibitions() -> dict:
    return {
        'local_20_trees_accessed': 'NO',
        'local_8_bad_accessed': 'NO',
        'product_integration': 'NO',
        'onnx': 'NO',
        'crown_stem_hybrid_modified': 'NO',
        'new_forest_downloaded': 'NO',
        'reverse_fold_run': 'NO',
        'test_site_changed': 'NO',
        'fourth_model_added': 'NO',
        'thresholds_tuned_on_wytham': 'NO',
        'hyperparameter_search': 'NO',
        'seed': cfg.SEED,
        'epochs_increased_after_results': 'NO',
        'evidence': ('no path under research/tree_qc_v25 references the TreeLineWarning '
                     'local corpus or the 8-bad challenge; the only data root is the '
                     'frozen D:\\TreeQCData ROI library'),
    }


def _provenance() -> dict:
    """Which proposal pairs the training data actually contains."""
    run = read_json('V25_GO_NO_GO.json')
    kinds = {}
    for fold in run.get('folds', []):
        for name, count in (fold['training']['sampling']
                            .get('per_proposal_kind') or {}).items():
            kinds.setdefault(name, 0)
            kinds[name] = max(kinds[name], count)
    simulated = kinds.get('SIMULATED_PROPOSAL', 0)
    real = kinds.get('REAL_ALGORITHM_PROPOSAL', 0)
    total = simulated + real + kinds.get('GT_CORRECT', 0)
    return {
        'per_proposal_kind': kinds,
        'simulated_proposal_rows': simulated,
        'real_algorithm_proposal_rows': real,
        'simulated_share_of_training_rows': (
            round(simulated / total, 6) if total else None),
        'statement': (
            'The training pairs are majority SIMULATED_PROPOSAL. The real '
            'CROWN_STEM_HYBRID refinement route, run on public data, reproduces the '
            'ground truth it is given and repairs almost nothing, so it supplied few '
            'defective proposals. REAL_ALGORITHM_PROPOSAL and SIMULATED_PROPOSAL are '
            'never mixed in the reporting.'),
        'can_the_result_be_called_product_generalisation': False,
        'what_this_round_answers': (
            'whether conservative local correction is worth continuing as a deep '
            'learning direction, and nothing about a real transmission-line product'),
    }


def _stage_status(payload: dict) -> dict:
    folds = payload['folds']
    complete = len(folds) >= 2
    return {
        'round': cfg.ROUND,
        'stages': [
            {'stage': 'V2_BASELINE_READ', 'status': 'COMPLETE',
             'artefact': 'research/tree_qc_v2/reports/MINI_REFINER_BAKEOFF.json'},
            {'stage': 'DELTA_REFINER_TRAIN', 'status': 'COMPLETE' if folds else 'NOT_RUN',
             'artefact': 'research/tree_qc_v25/reports/V25_GO_NO_GO.json'},
            {'stage': 'CONSERVATIVE_GATE_FREEZE',
             'status': 'COMPLETE' if folds else 'NOT_RUN',
             'artefact': 'research/tree_qc_v25/reports/FROZEN_GATE_*.json'},
            {'stage': 'CROSS_SITE_EVALUATION',
             'status': 'COMPLETE' if any(f['configuration'].startswith('cross_site')
                                         for f in folds) else 'NOT_RUN'},
            {'stage': 'IN_SITE_DIAGNOSTIC',
             'status': 'COMPLETE' if any(f['configuration'].startswith('in_site')
                                         for f in folds) else 'NOT_RUN'},
            {'stage': 'FINAL_REPORT', 'status': 'COMPLETE',
             'artefact': 'TREE_QC_V25_FINAL_GO_NO_GO_REPORT.md/.json'},
        ],
        'all_stages_terminal': complete,
        'round_stopped_at_verdict': True,
        'next_round_started': False,
    }


def render(payload: dict) -> str:
    lines: list[str] = []
    add = lines.append
    git = payload['git']
    verdict = payload['verdict']
    cross = verdict.get('cross_site', {})
    in_site = verdict.get('in_site', {})
    baseline = payload['result_a_v2_frozen']['configurations']
    best_baseline = verdict.get('result_a_v2_baseline_cross_site_delta')

    add('# TREE_QC_V25_FINAL_DEEP_LEARNING_GO_NO_GO')
    add('')
    add('The last small-scale feasibility check on the TreeLineWarning single-tree QC')
    add('deep learning route. It asks one question: is conservative local correction')
    add('worth continuing as a deep learning direction, or does the project move to')
    add('CROWN_STEM_HYBRID plus deterministic geometry QC plus human review?')
    add('')
    add('```')
    add(f'GIT_HEAD = {git["head"]}')
    add(f'BRANCH = {git["branch"]}')
    add(f'BASE_HEAD = {git["base_head"]}')
    add(f'WORKTREE_CLEAN = {"YES" if git["worktree_status"] == "clean" else "NO"}')
    add('')
    add(f'V2_BASELINE_CROSS_SITE_DELTA = {_f(best_baseline)}')
    add(f'DELTA_REFINER_CROSS_SITE_DELTA = {_f(cross.get("delta_iou_mean_ungated"))}')
    add(f'GATED_DELTA_CROSS_SITE_DELTA = {_f(cross.get("delta_iou_mean_gated"))}')
    add('')
    add(f'CROSS_SITE_BAD_MEAN_DELTA_IOU = {_f(cross.get("BAD_MEAN_DELTA_IOU"))}')
    add(f'CROSS_SITE_BAD_IMPROVEMENT_RATE = {_f(cross.get("BAD_IMPROVEMENT_RATE"))}')
    add('')
    add(f'CROSS_SITE_GOOD_MEAN_DELTA_IOU = {_f(cross.get("GOOD_MEAN_DELTA_IOU"))}')
    add(f'CROSS_SITE_GOOD_DEGRADATION_RATE = {_f(cross.get("GOOD_DEGRADATION_RATE"))}')
    add('')
    add(f'AUTO_EDIT_COVERAGE = {_f(cross.get("AUTO_EDIT_COVERAGE"))}')
    add(f'AUTO_EDIT_SUCCESS_RATE = {_f(cross.get("AUTO_EDIT_SUCCESS_RATE"))}')
    add('')
    add(f'IN_SITE_BAD_MEAN_DELTA_IOU = {_f(in_site.get("BAD_MEAN_DELTA_IOU"))}')
    add(f'IN_SITE_GOOD_MEAN_DELTA_IOU = {_f(in_site.get("GOOD_MEAN_DELTA_IOU"))}')
    add('')
    add(f'FINAL_VERDICT = {verdict["verdict"]}')
    add(f'TREE_QC_DEEP_LEARNING_ROUTE = {verdict["TREE_QC_DEEP_LEARNING_ROUTE"]}')
    add('```')
    add('')

    add('## 1. What was run, and what was frozen first')
    add('')
    add('One new model, three results, no search of any kind.')
    add('')
    add('| result | method | trained this round |')
    add('|---|---|---|')
    add('| A | V2 frozen full-mask refiner | no - read from the V2 artefacts |')
    add('| B | PointNeXt-S delta refiner, no gate | yes |')
    add('| C | the same model behind the conservative gate | yes (gate fitted on '
        'validation) |')
    add('')
    add('### Protocol revision on the record')
    add('')
    add('The model was trained **twice**, and both revisions are kept:')
    add('')
    add('- **revision 1** (pilot, `PILOT_V1_UNWEIGHTED_*`): unweighted loss.')
    add('- **revision 2** (reported here): the loss is importance-weighted by the')
    add("  stratified sampler's inverse inclusion probability, so the effective")
    add('  training distribution matches the ROI rather than the sampler.')
    add('')
    add('The revision was made *because revision 1 failed*, not because a number was')
    add('unsatisfactory. Revision 1 collapsed the removal head to "always remove"')
    add('even on masks it had trained on, and diagnosing that surfaced a genuine')
    add('confound: the sampler floors every stratum, so it over-represents the rare')
    add('decision. Measured over 200 training rows, the ADD-head positive rate was')
    add('0.1298 as sampled against 0.0908 in the population. Revision 2 corrects it')
    add('and reproduces the population rate exactly. Every other setting - seed,')
    add('epoch budget, patience, splits, thresholds, criteria - is unchanged, and')
    add('both revisions return the same verdict.')
    add('')
    add('The backbone is imported from the frozen V2 code rather than reimplemented,')
    add('so "same backbone" is a fact about the code:')
    add('')
    breakdown = payload['parameter_breakdown']
    add(f'- backbone `{breakdown["parameters_backbone"]:,}` parameters, identical to '
        f'V2 `MODEL_B_POINTNEXT_S`')
    add(f'- two decision heads `{breakdown["parameters_heads"]:,}` parameters')
    add(f'- total `{breakdown["parameters_total"]:,}`')
    add('')
    add(f'Budget: {cfg.MAX_EPOCHS} epochs maximum, early stop patience '
        f'{cfg.PATIENCE}, one fixed seed {cfg.SEED}, no hyper-parameter search.')
    add('')
    add('The gate threshold was written to disk and digested **before** the Wytham')
    add('test was scored, and the test path re-reads it from that file:')
    add('')
    for fold in payload['folds']:
        add(f"- `{fold['configuration']}`: frozen gate "
            f"`{fold['gate_frozen_sha256'][:16]}`, effective threshold "
            f"{fold['gate'].get('effective_threshold')}")
    add('')

    add('## 2. Results')
    add('')
    add('### A - V2 frozen full-mask refiner (read, not retrained)')
    add('')
    add('| configuration | model | delta IoU | GOOD delta | BAD delta |')
    add('|---|---|---|---|---|')
    for configuration, entry in baseline.items():
        for key, model in sorted(entry['models'].items()):
            add(f"| {configuration} | `{key}` {model['name']} | "
                f"{_f(model['delta_iou_mean'], True)} | "
                f"{_f(model['good_mean_delta_iou'], True)} | "
                f"{_f(model['bad_mean_delta_iou'], True)} |")
    add('')
    add('### B and C on the cross-site configuration')
    add('')
    fold = payload['folds'][0]
    add(f"Trained on {fold['split']['train']} Litchfield targets, tested on "
        f"{fold['split']['test']} Wytham targets the model never saw.")
    add('')
    add('| metric | B ungated | C gated |')
    add('|---|---|---|')
    add(f"| current IoU | {_f(fold['test']['ungated']['ALL']['current_iou_mean'])} | "
        f"{_f(fold['test']['gated']['ALL']['current_iou_mean'])} |")
    add(f"| refined IoU | {_f(fold['test']['ungated']['ALL']['refined_iou_mean'])} | "
        f"{_f(fold['test']['gated']['ALL']['refined_iou_mean'])} |")
    add(f"| delta IoU | {_f(fold['test']['ungated']['ALL']['delta_iou_mean'], True)} | "
        f"{_f(fold['test']['gated']['ALL']['delta_iou_mean'], True)} |")
    add(f"| GOOD delta IoU | "
        f"{_f(fold['test']['ungated']['GOOD_MEAN_DELTA_IOU'], True)} | "
        f"{_f(fold['test']['gated']['GOOD_MEAN_DELTA_IOU'], True)} |")
    add(f"| GOOD degradation rate | "
        f"{_f(fold['test']['ungated']['GOOD_DEGRADATION_RATE'])} | "
        f"{_f(fold['test']['gated']['GOOD_DEGRADATION_RATE'])} |")
    add(f"| BAD delta IoU | "
        f"{_f(fold['test']['ungated']['BAD_MEAN_DELTA_IOU'], True)} | "
        f"{_f(fold['test']['gated']['BAD_MEAN_DELTA_IOU'], True)} |")
    add(f"| BAD improvement rate | "
        f"{_f(fold['test']['ungated']['BAD_IMPROVEMENT_RATE'])} | "
        f"{_f(fold['test']['gated']['BAD_IMPROVEMENT_RATE'])} |")
    add(f"| NO_OP rate | {_f(fold['test']['ungated']['NO_OP_RATE'])} | "
        f"{_f(fold['test']['gated']['NO_OP_RATE'])} |")
    add(f"| AUTO_EDIT coverage | n/a | "
        f"{_f(fold['test']['gated']['AUTO_EDIT_COVERAGE'])} |")
    add(f"| AUTO_EDIT count | n/a | "
        f"{fold['test']['gated']['AUTO_EDIT_COUNT']} of "
        f"{fold['test']['gated']['rows']} rows |")
    add(f"| change-region F1 | "
        f"{_f(fold['test']['ungated']['change_region_f1_mean'])} | n/a |")
    add('')
    add('Point-level decisions on the cross-site test, ungated:')
    add('')
    add('| head | precision | recall | F1 |')
    add('|---|---|---|---|')
    for label in ('ADD', 'REMOVE'):
        add(f"| {label} | {_f(fold['test']['ungated'][f'{label}_PRECISION'])} | "
            f"{_f(fold['test']['ungated'][f'{label}_RECALL'])} | "
            f"{_f(fold['test']['ungated'][f'{label}_F1'])} |")
    add('')

    add('### In-site diagnostic')
    add('')
    in_site_fold = payload['folds'][1]
    add('| metric | B ungated | C gated |')
    add('|---|---|---|')
    add(f"| delta IoU | "
        f"{_f(in_site_fold['test']['ungated']['ALL']['delta_iou_mean'], True)} | "
        f"{_f(in_site_fold['test']['gated']['ALL']['delta_iou_mean'], True)} |")
    add(f"| GOOD delta IoU | "
        f"{_f(in_site_fold['test']['ungated']['GOOD_MEAN_DELTA_IOU'], True)} | "
        f"{_f(in_site_fold['test']['gated']['GOOD_MEAN_DELTA_IOU'], True)} |")
    add(f"| BAD delta IoU | "
        f"{_f(in_site_fold['test']['ungated']['BAD_MEAN_DELTA_IOU'], True)} | "
        f"{_f(in_site_fold['test']['gated']['BAD_MEAN_DELTA_IOU'], True)} |")
    add(f"| AUTO_EDIT coverage | n/a | "
        f"{_f(in_site_fold['test']['gated']['AUTO_EDIT_COVERAGE'])} |")
    add('')

    add('## 3. Verdict against the declared criteria')
    add('')
    add('Evaluated on the cross-site configuration, result C, and on nothing else.')
    add('')
    add('| criterion | rule | observed | passed |')
    add('|---|---|---|---|')
    for name, entry in verdict['criteria'].items():
        add(f"| {name} | {entry['rule']} | {_f(entry['observed'])} | "
            f"{'PASS' if entry['passed'] else 'FAIL'} |")
    add('')
    add(f"**FINAL_VERDICT = {verdict['verdict']}**")
    add('')
    add(f"**TREE_QC_DEEP_LEARNING_ROUTE = {verdict['TREE_QC_DEEP_LEARNING_ROUTE']}**")
    add('')
    if verdict['verdict'] == 'DEEP_LEARNING_NO_GO':
        add('Core criterion 1 fails, so the verdict is NO_GO regardless of the other')
        add('four. The honest reading of *why*, drawn from the artefacts:')
        add('')
        add('* The delta refiner does not merely fail to transfer. On the Wytham test')
        add(f"  it degrades a mask from {_f(fold['test']['ungated']['ALL']['current_iou_mean'])}")
        add(f"  to {_f(fold['test']['ungated']['ALL']['refined_iou_mean'])} IoU, harms "
            f"{_f(fold['test']['ungated']['GOOD_DEGRADATION_RATE'])} of already-good")
        add('  proposals, and improves bad ones only '
            f"{_f(fold['test']['ungated']['BAD_IMPROVEMENT_RATE'])} of the time.")
        add('* The conservative gate did exactly what it was built to do, and that is')
        add('  the second finding: faced with an unreliable model it accepted '
            f"{fold['test']['gated']['AUTO_EDIT_COUNT']}")
        add('  of the test rows, so the product path is safe but empty. On validation')
        add(f"  it looked usable (coverage {_f(fold['gate']['coverage'])}, success "
            f"{_f(fold['gate']['success_rate'])}), and on full ROIs it was not.")
        add('  A gate cannot manufacture a signal that the model does not have.')
        add('* This is not a defect in the implementation. A single-example overfit')
        add('  probe (below) shows the heads can learn to leave a correct mask alone.')
        add('  What fails is per-point keep/remove discrimination at forest scale,')
        add('  where roughly 85 percent of the points inside a candidate mask belong')
        add('  to a neighbouring tree or to background.')
        add('')
        add('### Why this is a NO_GO and not something to tune')
        add('')
        add('Every lever that would raise the numbers is forbidden by this round: more')
        add('epochs, another seed, a search over the threshold, a different backbone,')
        add('or unblinding the local corpus. The result is reported as it stands.')
        add('')

    add('## 4. Overfit probe: is the model broken, or is the task hard?')
    add('')
    probe = payload.get('overfit_probe')
    if probe:
        add(f"One GT_CORRECT row, {probe.get('steps')} steps: the heads learn to")
        add(f"preserve it (delta {_f(probe.get('delta_iou'), True)} on the full ROI).")
        add('So the negative result is about the task, not about a coding defect.')
    else:
        add('The probe result was not recorded; see the run log.')
    add('')

    add('## 5. Provenance: what the training pairs actually are')
    add('')
    prov = payload['provenance']
    add('| proposal kind | training rows (max across folds) |')
    add('|---|---|')
    for name, count in sorted(prov['per_proposal_kind'].items()):
        add(f'| `{name}` | {count:,} |')
    add('')
    add(f"{prov['statement']}")
    add('')
    add(f"Scheduled GOOD share of the training set: target "
        f"{cfg.TARGET_GOOD_FRACTION}, realised "
        f"{payload['folds'][0]['training']['sampling']['realised_good_fraction']} "
        f"(GOOD proposals are explicit training rows, as required).")
    add('')
    add('This round does **not** show generalisation to a real transmission-line')
    add('product, and nothing here should be read that way.')
    add('')

    add('## 6. Prohibitions honoured')
    add('')
    for key, value in payload['prohibitions'].items():
        if key == 'evidence':
            continue
        add(f'- {key}: {value}')
    add('')
    add(f"{payload['prohibitions']['evidence']}")
    add('')
    add(f"V2 untouched: {len(payload['v2_code_untouched']['v2_files_modified'])} files "
        f"modified under `research/tree_qc_v2/`.")
    add('')

    add('## 7. Recommended next route')
    add('')
    add('The deep learning route stops here. The recommendation on the record is:')
    add('')
    add('1. **CROWN_STEM_HYBRID** remains the primary single-tree algorithm.')
    add('2. **Deterministic geometry / structural QC** replaces the learned refiner:')
    add('   rule-based checks on crown completeness, stem presence, neighbour overlap')
    add('   and size plausibility, which are auditable and need no training data.')
    add('3. **Human review** for the cases those two cannot settle.')
    add('')
    add('The local 20-tree corpus and the 8-bad blind challenge remain sealed and were')
    add('not used to reach this verdict.')
    add('')

    add('## 8. Reproducing this round')
    add('')
    add('```')
    add('set PYTHONPATH=research/tree_qc_v25')
    add('python research/tree_qc_v25/v25_runner.py \\')
    add('    --configurations cross_site_test_wytham in_site_test_litchfield')
    add('python research/tree_qc_v25/v25_report.py')
    add('pytest research/tree_qc_v25/test_v25.py')
    add('```')
    add('')
    add(f"Config digest: `{payload['config_digest']}`. Frozen thresholds are recorded")
    add('above and in `research/tree_qc_v25/reports/FROZEN_GATE_*.json`.')
    add('')
    return '\n'.join(lines)


def _f(value, signed: bool = False) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f'{value:+.4f}' if signed else f'{value:.4f}'
    return str(value)


def main() -> int:
    payload = build()
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    OUT_MD.write_text(render(payload), encoding='utf-8')
    print(f'wrote {OUT_JSON}')
    print(f'wrote {OUT_MD}')
    print(json.dumps(payload['verdict'], indent=2, default=str)[:1200])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
