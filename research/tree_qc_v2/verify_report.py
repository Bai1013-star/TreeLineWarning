"""Verify the final report against the artefacts it claims to be generated from.

This does not re-run anything. It reads the artefacts, reads the report, and
recomputes the headline numbers from the raw per-row data, so a report that
drifted from its runs fails here rather than being shipped.

Exit code 0 means every check passed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPORT_JSON = REPO / 'TREE_QC_V2_FEASIBILITY_REPORT.json'
REPORT_MD = REPO / 'TREE_QC_V2_FEASIBILITY_REPORT.md'
REPORTS = HERE / 'reports'

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    print('=== 1. report presence and tracking')
    check(REPORT_JSON.exists(), f'missing {REPORT_JSON.name}')
    check(REPORT_MD.exists(), f'missing {REPORT_MD.name}')
    if not (REPORT_JSON.exists() and REPORT_MD.exists()):
        report_failures()
        return 1
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch',
                              str(REPORT_MD.relative_to(REPO)),
                              str(REPORT_JSON.relative_to(REPO))],
                             cwd=REPO, capture_output=True, text=True)
    check(tracked.returncode == 0, 'reports are not tracked by git')
    print(f'   report json {REPORT_JSON.stat().st_size:,} bytes, '
          f'markdown {REPORT_MD.stat().st_size:,} bytes, tracked = '
          f'{tracked.returncode == 0}')

    report = load(REPORT_JSON)
    markdown = REPORT_MD.read_text(encoding='utf-8')

    print('=== 2. every stage is reported and none is NOT_RUN')
    fingerprint_artefact = load(REPORTS / 'SITE_FINGERPRINT.json')
    proposal_artefact = load(REPORTS / 'PUBLIC_GT_PROPOSAL.json')
    library_artefact = load(REPORTS / 'ROI_LIBRARY.json')
    bakeoff_artefact = load(REPORTS / 'MINI_REFINER_BAKEOFF.json')
    check(report['site_fingerprint_test'].get('status') != 'NOT_RUN',
          'SITE_FINGERPRINT_TEST reported as NOT_RUN')
    check(report['public_gt_proposal_test'].get('status') != 'NOT_RUN',
          'PUBLIC_GT_PROPOSAL_TEST reported as NOT_RUN')
    check(report['mini_refiner_bakeoff'].get('status') == 'RUN',
          'MINI_REFINER_BAKEOFF reported as NOT_RUN')
    check(report['mini_refiner_bakeoff'].get('target_count')
          == library_artefact['target_count'],
          'the ROI library target count drifted from the library artefact')
    check(report['mini_refiner_bakeoff'].get('library_index_sha256')
          == library_artefact['index_sha256'],
          'the ROI library index digest drifted from the library artefact')
    print('   three stages present in the report')
    print(f"   ROI library: {library_artefact['target_count']} targets, index "
          f"{str(library_artefact['index_sha256'])[:16]}")

    print('=== 3. Task A numbers match the artefact')
    for name, entry in fingerprint_artefact['representations'].items():
        reported = report['site_fingerprint_test']['representations'].get(name)
        check(reported is not None, f'{name} missing from the report')
        if reported is None:
            continue
        check(abs(reported['test_site_accuracy']
                  - entry['test']['site_accuracy']) < 1e-12,
              f'{name} test accuracy drifted')
        check(reported['parameters'] == entry['parameters'],
              f'{name} parameter count drifted')
        check(reported['in_channels'] == entry['in_channels'],
              f'{name} channel count drifted')
    print(f"   {len(fingerprint_artefact['representations'])} representations match")

    print('=== 4. Task B numbers match the artefact')
    for key in ('real_pairs_above_min_iou', 'real_pairs_per_site',
                'simulated_rows_total', 'gt_correct_rows_total'):
        check(report['public_gt_proposal_test'][key] == proposal_artefact[key],
              f'Task B {key} drifted')
    check(report['public_gt_proposal_test']['verdict']
          == proposal_artefact['PUBLIC_GT_PAIR_CONSTRUCTION'],
          'Task B verdict drifted')
    check(report['public_gt_proposal_test']['roi_coverage']['mean']
          == proposal_artefact['roi_coverage']['mean'],
          'Task B ROI coverage drifted')
    print(f"   real pairs {proposal_artefact['real_pairs_above_min_iou']}, "
          f"simulated {proposal_artefact['simulated_rows_total']}")

    print('=== 5. Task C headline numbers recomputed from the raw per-row data')
    for fold in bakeoff_artefact['folds']:
        configuration = fold['configuration']
        for key, model in sorted(fold['models'].items()):
            reported = report['mini_refiner_bakeoff']['folds']
            match = [f for f in reported if f['configuration'] == configuration]
            check(len(match) == 1, f'{configuration} missing or duplicated in report')
            if not match:
                continue
            reported_model = match[0]['models'].get(key)
            check(reported_model is not None, f'{configuration}/{key} missing')
            if reported_model is None:
                continue
            for field in ('delta_iou_mean', 'change_region_f1_mean',
                          'current_iou_mean'):
                check(abs(reported_model['summary'][field]
                          - model['summary'][field]) < 1e-12,
                      f'{configuration}/{key} {field} drifted')
            check(reported_model['parameters'] == model['parameters'],
                  f'{configuration}/{key} parameter count drifted')
            # Independent recomputation straight from the stored rows.
            rows_file = REPORTS / f'BAKEOFF_ROWS_{configuration}_{key}.json'
            check(rows_file.exists(), f'missing raw rows {rows_file.name}')
            if not rows_file.exists():
                continue
            rows = load(rows_file)['test_rows']
            raw_delta = sum(r['delta_iou'] for r in rows) / len(rows)
            raw_f1 = sum(r['change_region_f1'] for r in rows) / len(rows)
            raw_current = sum(r['current_iou'] for r in rows) / len(rows)
            check(abs(raw_delta - model['summary']['delta_iou_mean']) < 1e-9,
                  f'{configuration}/{key} raw delta_iou recomputation disagrees')
            check(abs(raw_f1 - model['summary']['change_region_f1_mean']) < 1e-9,
                  f'{configuration}/{key} raw change F1 recomputation disagrees')
            check(abs(raw_current - model['summary']['current_iou_mean']) < 1e-9,
                  f'{configuration}/{key} raw current IoU recomputation disagrees')
    print(f"   {len(bakeoff_artefact['folds'])} configurations, "
          f"{sum(len(f['models']) for f in bakeoff_artefact['folds'])} model runs "
          f'recomputed from raw rows')

    print('=== 6. the limitation is recorded, and only one cross-site fold exists')
    configurations = [f['configuration'] for f in bakeoff_artefact['folds']]
    cross = [c for c in configurations if c.startswith('cross_site')]
    in_site = [c for c in configurations if c.startswith('in_site')]
    check(len(cross) == 1, f'expected exactly one cross-site fold, found {cross}')
    check(cross == ['cross_site_test_wytham'],
          f'expected cross_site_test_wytham, found {cross}')
    check(len(in_site) == 1, f'expected exactly one in-site fold, found {in_site}')
    check('cross_site_test_litchfield' not in configurations,
          'a reverse cross-site fold is present but was not supposed to be run')
    limitation = report['mini_refiner_bakeoff'].get('limitation') or {}
    check(limitation.get('status') == 'COMPLETE_WITH_LIMITATION',
          'the bakeoff limitation status is not recorded')
    check('reverse' in json.dumps(limitation).lower(),
          'the missing reverse fold is not named in the limitation')
    check('COMPLETE_WITH_LIMITATION' in markdown,
          'the limitation status is missing from the markdown report')
    print(f'   cross-site folds: {cross}')
    print(f'   in-site folds:    {in_site}')
    print(f"   limitation: {limitation.get('status')}")

    print('=== 7. verdict is consistent with the gates it sits on')
    verdict = report['final_verdict']
    checks = {
        'identity_copy': all(
            value == 'PASS'
            for values in (verdict.get('identity_copy_gate') or {}).values()
            for value in values),
        'good_preservation_any_pass': any(
            value == 'PASS'
            for values in (verdict.get('good_preservation') or {}).values()
            for value in values),
    }
    check(checks['identity_copy'],
          'the identity-copy gate did not pass everywhere but the report says it did')
    check(not checks['good_preservation_any_pass'],
          'a GOOD-preservation pass appeared without a re-run')
    check(verdict['verdict'] == 'V2_FEASIBILITY_PARTIAL',
          f"unexpected verdict {verdict['verdict']}")
    check(report['formal_v2_training_justified'] == 'NO',
          'FORMAL_V2_TRAINING_JUSTIFIED should be NO for a PARTIAL verdict')
    print(f"   verdict {verdict['verdict']}, "
          f"identity-copy all PASS = {checks['identity_copy']}, "
          f"good-preservation any PASS = {checks['good_preservation_any_pass']}")

    print('=== 8. the round prohibitions survived into the report')
    check(report['local_data_untouched']['local_20_trees_accessed'] == 'NO',
          'local 20 trees marked as accessed')
    check(report['local_data_untouched']['local_8_bad_accessed'] == 'NO',
          'local 8-bad marked as accessed')
    check(report['v1_history_untouched']['changed_files_in_research_tree_qc'] == [],
          'V1 history files were changed')
    check(report['v1_history_untouched']['changed_files_in_algorithms'] == [],
          'algorithms/ files were changed')
    check(report['v1_history_untouched']['changed_files_in_core'] == [],
          'core/ files were changed')
    for marker in ('LOCAL_20_TREES_ACCESSED = NO', 'LOCAL_8_BAD_ACCESSED = NO',
                   'V1_HISTORY_UNTOUCHED = YES',
                   'FORMAL_V2_TRAINING_JUSTIFIED = NO'):
        check(marker in markdown, f'{marker} missing from the markdown report')
    print('   local data untouched, V1 untouched, formal training not justified')

    print('=== 9. the report is regenerable without changing a number')
    before = load(REPORT_JSON)
    # The child needs the real environment: a scrubbed one leaves `git` unable to
    # resolve the repository, and the regenerated report then carries an empty
    # GIT_HEAD. That is checked below rather than discovered later.
    regenerated = subprocess.run(
        [sys.executable, str(HERE / 'v2_report.py')],
        cwd=REPO, capture_output=True, text=True, env=os.environ.copy())
    if regenerated.returncode != 0:
        check(False, f'report regeneration failed: {regenerated.stderr[-400:]}')
    else:
        after = load(REPORT_JSON)
        check(bool(after['git']['head']) and len(after['git']['head']) == 40,
              'the regenerated report has no GIT_HEAD')
        check(bool(after['git']['branch']),
              'the regenerated report has no BRANCH')
        differing = sorted(
            key for key in set(before) | set(after)
            if json.dumps(before.get(key), sort_keys=True)
            != json.dumps(after.get(key), sort_keys=True))
        # Only the git block may move between two reads, and only inside itself:
        # HEAD and worktree state describe the repository, not the experiments.
        unexpected = [key for key in differing if key != 'git']
        check(not unexpected,
              f'regeneration changed more than git state: {unexpected}')
        if differing == ['git']:
            git_keys = sorted(
                key for key in set(before['git']) | set(after['git'])
                if json.dumps(before['git'].get(key), sort_keys=True)
                != json.dumps(after['git'].get(key), sort_keys=True))
            detail = {key: {'before': before['git'].get(key),
                            'after': after['git'].get(key)} for key in git_keys}
            check(set(git_keys) <= {'head', 'worktree_status'},
                  f'regeneration changed git fields beyond head/status: {detail}')
            print(f'   regeneration differs only in git.{git_keys} '
                  f'(HEAD and worktree state)')
            print(f'   before head {str(before["git"]["head"])[:12]}, '
                  f'after head {str(after["git"]["head"])[:12]}, '
                  f'branch {after["git"]["branch"]!r}')
        elif not differing:
            print('   regeneration reproduced the report byte for byte')
        else:
            print(f'   regeneration differs only in: {differing}')

    report_failures()
    return 1 if FAILURES else 0


def report_failures() -> None:
    print()
    if FAILURES:
        print(f'FAILED {len(FAILURES)} of {CHECKS} checks:')
        for failure in FAILURES:
            print(f'  - {failure}')
    else:
        print(f'ALL {CHECKS} CHECKS PASSED')


if __name__ == '__main__':
    raise SystemExit(main())
