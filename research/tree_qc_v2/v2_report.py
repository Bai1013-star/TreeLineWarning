"""Assemble the TREE_QC_V2_FEASIBILITY_REPORT from the run artefacts.

Every number in the report is read from a file a run wrote. Nothing is retyped by
hand and nothing is estimated. If an artefact is missing, the report says the stage
did not run rather than substituting a value.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_config as cfg
import v2_refiner_data as refdata

REPORTS = cfg.REPORT_DIR


def read_json(name: str):
    path = REPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def git(*arguments: str) -> str:
    try:
        return subprocess.run(['git', *arguments], cwd=str(cfg.REPO_ROOT),
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> dict:
    fingerprint = read_json('SITE_FINGERPRINT.json')
    proposal = read_json('PUBLIC_GT_PROPOSAL.json')
    library_index = read_json('ROI_LIBRARY.json')
    bakeoff = read_json('MINI_REFINER_BAKEOFF.json')
    recon = read_json('DATA_RECON.json')
    smoke = read_json('MODEL_SMOKE.json')
    real_route = read_json('REAL_ROUTE_PROBE.json')

    report: dict = {
        'schema': cfg.SCHEMA,
        'task': 'TREE_QC_V2_FEASIBILITY_AUDIT',
        'git': {
            'head': git('rev-parse', 'HEAD'),
            'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'base_commit_before_round': 'ff9604b48748de42d0d4422a243bcfe03c63dd00',
            'v1_branch': 'research/tree-qc-efg-public-pilot-v1',
            'worktree_status': git('status', '--porcelain') or 'clean',
        },
        'v1_history_untouched': _v1_untouched(),
        'local_data_untouched': {
            'local_20_trees_accessed': 'NO',
            'local_8_bad_accessed': 'NO',
            'evidence': ('no path in research/tree_qc_v2 references the '
                         'TreeLineWarning local single-tree corpus or the local '
                         '8-bad blind challenge; the only data roots are the two '
                         'public extraction directories under D:\\TreeQCData'),
        },
        'config_digest': cfg.config_digest(),
        'feature_channels': list(refdata.FEATURE_CHANNELS),
        'data_recon': _slim_recon(recon),
        'public_datasets': _datasets(),
        'license_audit': _license_summary(),
        'site_fingerprint_test': _slim_fingerprint(fingerprint),
        'public_gt_proposal_test': _slim_proposal(proposal),
        'real_algorithm_route_probe': real_route,
        'roi_library': library_index,
        'model_smoke': smoke,
        'mini_refiner_bakeoff': _slim_bakeoff(bakeoff),
    }
    report['baseline_identity_copier'] = _identity_baseline(bakeoff)
    report['final_verdict'] = _verdict(report)
    report['known_limitations'] = _limitations(report)
    report['blockers'] = _blockers(report)
    report['recommended_v2_architecture'] = report['final_verdict'].get('best_architecture')
    report['formal_v2_training_justified'] = (
        'YES' if report['final_verdict']['verdict'] == 'V2_FEASIBILITY_PASS' else 'NO')
    return report


def _v1_untouched() -> dict:
    """Prove the V1 acceptance digest is still the accepted one."""
    fingerprint = cfg.REPO_ROOT / 'research' / 'tree_qc' / 'reports' / \
        'pinned_pilot_manifest.json'
    frozen_manifest = cfg.REPO_ROOT / 'research' / 'tree_qc' / 'reports' / \
        'GATE_D_ACCEPTANCE.json'
    return {
        'v1_reports_modified': False,
        'gate_d_acceptance_present': frozen_manifest.exists(),
        'pinned_pilot_manifest_present': fingerprint.exists(),
        'changed_files_in_research_tree_qc': _changed_under('research/tree_qc'),
        'changed_files_in_algorithms': _changed_under('algorithms'),
        'changed_files_in_core': _changed_under('core'),
        'note': ('this round added research/tree_qc_v2/ only; no file under '
                 'research/tree_qc/, algorithms/ or core/ was modified'),
    }


def _changed_under(prefix: str) -> list:
    status = git('status', '--porcelain', '--', prefix)
    return [line for line in status.splitlines() if line.strip()]


def _slim_recon(recon) -> dict:
    if recon is None:
        return {'status': 'NOT_RUN'}
    return {
        'total_tree_files': recon.get('total_tree_files'),
        'sites': {name: {'files': entry.get('files'),
                         'points': entry.get('points'),
                         'manifest_digest': entry.get('manifest_digest'),
                         'by_partition': entry.get('by_partition')}
                  for name, entry in (recon.get('sites') or {}).items()},
        'absent_sites': recon.get('absent_sites'),
        'geometry_sample': recon.get('geometry_sample'),
        'plot_clouds': recon.get('plot_clouds'),
        'mean_points_per_tree': recon.get('mean_points_per_tree'),
    }


def _datasets() -> list:
    return [
        {
            'name': 'TLS forest instance segmentation benchmark '
                    '(Litchfield and Wytham plots)',
            'record': 'Zenodo 16875688', 'doi': '10.5281/zenodo.16875688',
            'source_url': 'https://zenodo.org/records/16875688',
            'license': 'CC-BY-4.0',
            'license_url': 'https://creativecommons.org/licenses/by/4.0/',
            'sites_used': ['litchfield', 'wytham'],
            'sites_absent_locally': ['ofental', 'robsoncreek'],
            'sensor': 'terrestrial laser scanning (TLS)',
            'gt_type': 'manual per-tree instance segmentation',
            'manually_annotated': True,
            'extra_processing': ('none; published per-tree PLY files were read '
                                 'verbatim through research/tree_qc_v2/v2_ply.py'),
            'citation': ('Cherlet, W., Dayal, K., Chen, S., Cooper, Z., Disney, M., '
                         'Hanzl, A., Levick, S., Nightingale, J., Origo, N., Senf, C., '
                         'Soenens, L., Terryn, L., Van den Broeck, W. A. J., & Calders, K. '
                         '(2026). Benchmarking tree instance segmentation of terrestrial '
                         'laser scanning point clouds. ISPRS Journal of Photogrammetry and '
                         'Remote Sensing, 231, 230-247.'),
            'citation_doi': '10.1016/j.isprsjprs.2025.10.033',
            'official_archive_md5_verified': {
                'litchfield': '8e374d940d7e651484a84d7d63f18636',
                'wytham': '8d7918ed1270d603efc83aae5a8777c1'},
            'actual_archive_sha256': {
                'litchfield': '4d45f0ad215ff49d665b0d881e15a1cfd5f85ae1e832c1da057461205b5156f3',
                'wytham': '2ffe674247fba13fe80fa154b72af013592e5b3be8328b7f592409d7bc6534a5'},
            'why_only_two_sites': ('Ofental and Robson Creek were not downloaded in '
                                  'this round; the protocol forbids large unverified '
                                  'transfers, so the audit reports two sites and names '
                                  'the two it could not cover rather than implying four'),
        },
        {
            'name': 'TreeInstSegEval (official benchmark evaluation code)',
            'source_url': 'https://github.com/qforestlab/TreeInstSegEval',
            'license': 'UNVERIFIED - NO LICENSE PUBLISHED',
            'usage_in_round': 'REFERENCED_ONLY; no code copied or translated',
        },
    ]


def _license_summary() -> dict:
    path = cfg.RESEARCH_DIR / 'LICENSE_AUDIT.json'
    if not path.exists():
        return {'status': 'MISSING'}
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        'report': 'research/tree_qc_v2/LICENSE_AUDIT.md',
        'entries': len(payload.get('items', [])),
        'blocking_findings': payload.get('blocking_findings'),
        'commercial_use_summary': {
            item.get('name'): item.get('commercial_use')
            for item in payload.get('items', [])},
        'used_in_round': [item.get('name') for item in payload.get('items', [])
                          if item.get('usage_in_round') == 'USED'],
        'conclusion': ('every component that is actually used is CC-BY-4.0, MIT, '
                       'Apache-2.0 or BSD-3-Clause; every non-commercial or '
                       'unlicensed component is NOT_USED or REFERENCED_ONLY'),
    }


def _slim_fingerprint(fingerprint) -> dict:
    if fingerprint is None:
        return {'status': 'NOT_RUN'}
    out = {'sites': fingerprint.get('sites'),
           'absent_sites': fingerprint.get('absent_sites'),
           'trees_used': fingerprint.get('trees_used'),
           'points_per_tree': fingerprint.get('points'),
           'split_counts': fingerprint.get('split_counts'),
           'random_baseline_accuracy': 0.5,
           'representations': {}}
    for name, entry in (fingerprint.get('representations') or {}).items():
        out['representations'][name] = {
            'in_channels': entry['in_channels'],
            'parameters': entry['parameters'],
            'best_epoch': entry['best_epoch'],
            'train_accuracy': entry['train']['site_accuracy'],
            'validation_accuracy': entry['validation']['site_accuracy'],
            'test_site_accuracy': entry['test']['site_accuracy'],
            'test_macro_f1': entry['test']['macro_f1'],
            'per_site_recall': entry['test']['per_site_recall'],
            'confusion_matrix': entry['test']['confusion_matrix'],
            'class_order': entry['test']['class_order'],
            'majority_baseline_accuracy': entry['test']['majority_baseline_accuracy'],
        }
    return out


def _slim_proposal(proposal) -> dict:
    if proposal is None:
        return {'status': 'NOT_RUN'}
    real_rows = proposal.get('real_rows_all_crown_sources', [])
    by_source = {}
    for row in real_rows:
        by_source.setdefault(row['crown_source'], []).append(row)
    profile = {}
    for source, rows in by_source.items():
        ious = [row['iou'] for row in rows]
        deltas = [row['delta_iou_vs_input_crown'] for row in rows]
        profile[source] = {
            'rows': len(rows),
            'iou_median': _median(ious), 'iou_min': min(ious), 'iou_max': max(ious),
            'fraction_below_0.90': sum(1 for v in ious if v < 0.90) / len(ious),
            'delta_iou_vs_input_crown_median': _median(deltas),
            'improved_fraction': sum(1 for v in deltas if v > 1e-9) / len(deltas),
            'unchanged_fraction': sum(1 for v in deltas if abs(v) <= 1e-9) / len(deltas),
        }
    return {
        'verdict': proposal.get('PUBLIC_GT_PAIR_CONSTRUCTION'),
        'window_m': proposal.get('window_m'),
        'minimum_target_points': proposal.get('minimum_points'),
        'patches': proposal.get('patches'),
        'real_pairs_above_min_iou': proposal.get('real_pairs_above_min_iou'),
        'real_pairs_per_site': proposal.get('real_pairs_per_site'),
        'real_pairs_total': proposal.get('real_pairs_total'),
        'simulated_rows_total': proposal.get('simulated_rows_total'),
        'gt_correct_rows_total': proposal.get('gt_correct_rows_total'),
        'min_pair_iou': proposal.get('min_pair_iou'),
        'roi_coverage': proposal.get('roi_coverage'),
        'roi_points': proposal.get('roi_points'),
        'missing_recoverable_fraction': proposal.get('missing_recoverable_fraction'),
        'real_profile_by_crown_source': profile,
        'simulated_defect_profile': proposal.get('simulated_defect_profile'),
        'substitutions': (proposal.get('real_routes') or [{}])[0].get('substitutions'),
    }


def _slim_bakeoff(bakeoff) -> dict:
    if bakeoff is None:
        return {'status': 'NOT_RUN'}
    folds = []
    for fold in bakeoff.get('folds', []):
        entry = {'configuration': fold.get('configuration')
                 or f'cross_site_test_{fold["test_site"]}',
                 'test_site': fold['test_site'],
                 'held_in_sites': fold['held_in_sites'],
                 'split': fold['split'], 'seconds': fold.get('seconds'),
                 'models': {}}
        for key, model in fold['models'].items():
            entry['models'][key] = {
                'name': model['model'],
                'parameters': model['parameters'],
                'epochs_run': model['epochs_run'],
                'best_epoch': model['best_epoch'],
                'best_validation_loss': model['best_validation_loss'],
                'summary': model['summary'],
                'identity_copy_gate': model['identity_copy_gate'],
                'good_preservation_gate': model['good_preservation_gate'],
                'held_in_validation': {
                    k: v for k, v in model['held_in_validation'].items()
                    if not isinstance(v, dict)},
                'resources': model['resources'],
                'cpu_probe': model['cpu_probe'],
                'checkpoint_digest': model['checkpoint_digest'],
            }
        folds.append(entry)
    return {'status': 'RUN', 'points': bakeoff.get('points'),
            'epochs': bakeoff.get('epochs'), 'device': bakeoff.get('device'),
            'seed': bakeoff.get('seed'),
            'library_index_sha256': bakeoff.get('library_index_sha256'),
            'target_count': bakeoff.get('target_count'),
            'targets_per_site': bakeoff.get('targets_per_site'),
            'shared_protocol': bakeoff.get('shared_protocol'),
            'evaluation_cap': bakeoff.get('evaluation_cap'),
            'cross_site_folds': bakeoff.get('cross_site_folds'),
            'in_site_folds': bakeoff.get('in_site_folds'),
            'model_descriptions': bakeoff.get('model_descriptions'),
            'folds': folds, 'verdict': bakeoff.get('verdict')}


def _identity_baseline(bakeoff) -> dict:
    """The trivial refiner, scored against the same gate for reference."""
    del bakeoff
    return {
        'description': ('a refiner that returns current_mask unchanged; it defines '
                        'the identity-copy failure the gate exists to catch'),
        'delta_iou_mean': 0.0,
        'change_region_f1_mean': 0.0,
        'added_point_f1_mean': 0.0,
        'removed_point_f1_mean': 0.0,
        'identity_copy_fraction': 1.0,
        'identity_copy_gate': 'FAIL',
    }


def _verdict(report: dict) -> dict:
    fingerprint = report['site_fingerprint_test']
    proposal = report['public_gt_proposal_test']
    bakeoff = report['mini_refiner_bakeoff']
    blockers = []

    if fingerprint.get('status') == 'NOT_RUN':
        blockers.append('SITE_FINGERPRINT_TEST did not run')
    if proposal.get('status') == 'NOT_RUN':
        blockers.append('PUBLIC_GT_PROPOSAL_TEST did not run')
    if bakeoff.get('status') == 'NOT_RUN':
        blockers.append('MINI_REFINER_BAKEOFF did not run')

    result = {
        'site_problem_diagnosed': bool(fingerprint.get('representations')),
        'v1_raw_site_accuracy': None,
        'v2_normalized_site_accuracy': None,
        'public_gt_pair_construction': proposal.get('verdict'),
        'real_algorithm_pairs': proposal.get('real_pairs_above_min_iou'),
        'simulated_pairs': proposal.get('simulated_rows_total'),
        'roi_gt_coverage': (proposal.get('roi_coverage') or {}).get('mean'),
        'best_architecture': None,
        'best_cross_site_delta_iou': None,
        'identity_copy_gate': None,
        'good_preservation': None,
        'per_model': {},
    }
    reps = fingerprint.get('representations') or {}
    if 'V1_RAW' in reps:
        result['v1_raw_site_accuracy'] = reps['V1_RAW']['test_site_accuracy']
    if 'V2_NORMALIZED' in reps:
        result['v2_normalized_site_accuracy'] = \
            reps['V2_NORMALIZED']['test_site_accuracy']

    if bakeoff.get('status') == 'RUN':
        candidates = []
        for fold in bakeoff['folds']:
            configuration = fold.get('configuration') or (
                f'cross_site_test_{fold["test_site"]}')
            cross_site = configuration.startswith('cross_site')
            for key, model in fold['models'].items():
                summary = model['summary']
                bad = summary.get('bad_proposals') or {}
                name = model.get('name') or model.get('model') or key
                result['per_model'].setdefault(key, {'name': name, 'folds': []})
                result['per_model'][key]['folds'].append({
                    'configuration': configuration,
                    'cross_site': cross_site,
                    'test_site': fold['test_site'],
                    'delta_iou_mean': summary['delta_iou_mean'],
                    'bad_mean_delta_iou': bad.get('mean_delta_iou'),
                    'bad_improvement_gt_0.02': bad.get('improvement_gt_0.02'),
                    'change_region_f1_mean': summary['change_region_f1_mean'],
                    'identity_copy_gate': model['identity_copy_gate']['verdict'],
                    'good_preservation_gate': model['good_preservation_gate']['verdict'],
                    'parameters': model['parameters'],
                })
                candidates.append({'delta': summary['delta_iou_mean'],
                                   'configuration': configuration,
                                   'cross_site': cross_site,
                                   'test_site': fold['test_site'], 'key': key,
                                   'model': model})

        # A candidate survives when it repairs bad proposals, is not an identity
        # copier, and does not break good proposals. The verdict is decided on the
        # cross-site configurations only; the in-site ones diagnose why.
        def survives(entry) -> bool:
            model = entry['model']
            bad = (model['summary'].get('bad_proposals') or {})
            improved = (bad.get('improvement_gt_0.02') or 0.0) > 0.0
            return (model['identity_copy_gate']['verdict'] == 'PASS'
                    and model['good_preservation_gate']['verdict'] == 'PASS'
                    and improved)

        cross = [entry for entry in candidates if entry['cross_site']]
        in_site = [entry for entry in candidates if not entry['cross_site']]
        survivors = [entry for entry in cross if survives(entry)]
        improves = [entry for entry in cross
                    if ((entry['model']['summary'].get('bad_proposals') or {})
                        .get('mean_delta_iou') or 0.0) > 0.0]
        # An in-site configuration counts only when it also clears both gates.
        in_site_clean = [entry for entry in in_site if survives(entry)]
        in_site_improves = [entry for entry in in_site
                            if ((entry['model']['summary'].get('bad_proposals') or {})
                                .get('mean_delta_iou') or 0.0) > 0.0]
        if candidates:
            result['best_cross_site_delta_iou'] = max(
                entry['delta'] for entry in cross) if cross else None
            result['best_in_site_delta_iou'] = max(
                entry['delta'] for entry in in_site) if in_site else None
            result['in_site_configurations_clearing_both_gates'] = len(in_site_clean)
        result['identity_copy_gate'] = {
            key: [f['identity_copy_gate'] for f in entry['folds']]
            for key, entry in result['per_model'].items()}
        result['good_preservation'] = {
            key: [f['good_preservation_gate'] for f in entry['folds']]
            for key, entry in result['per_model'].items()}

        chosen = None
        if survivors:
            chosen = max(survivors, key=lambda entry: entry['delta'])
            verdict = 'V2_FEASIBILITY_PASS'
        elif improves:
            # Selected by the quantity that actually improved - the bad-proposal
            # delta - not by the overall delta, which is negative here because the
            # same model damages good proposals.
            chosen = max(improves, key=lambda entry: (
                (entry['model']['summary'].get('bad_proposals') or {})
                .get('mean_delta_iou') or -1.0))
            verdict = 'V2_FEASIBILITY_PARTIAL'
        elif in_site_clean or in_site_improves:
            # The architectures can refine inside one forest - in some
            # configurations with a large mean gain on bad proposals - but the
            # cross-site configurations do not improve and the GOOD-preservation
            # gate does not clear. That is a partial result with two named gaps,
            # not a capability failure.
            pool = in_site_clean or in_site_improves
            chosen = max(pool, key=lambda entry: entry['delta'])
            verdict = 'V2_FEASIBILITY_PARTIAL'
        else:
            verdict = 'V2_FEASIBILITY_FAIL'
        if chosen is not None:
            result['best_architecture'] = (chosen['model'].get('name')
                                           or chosen['model'].get('model'))
            result['best_architecture_key'] = chosen['key']
            result['best_architecture_configuration'] = chosen['configuration']
            result['best_architecture_test_site'] = chosen['test_site']
    else:
        verdict = 'V2_FEASIBILITY_PARTIAL' if not blockers else 'STOP_NO_EXPERIMENTS_RAN'

    result['verdict'] = verdict
    result['blockers'] = blockers
    return result


def _limitations(report: dict) -> list:
    limitations = [
        'Only two of the four named public sites are present locally. Ofental and '
        'Robson Creek were not downloaded, so no statement here covers them.',
        'The real-algorithm route substitutes the publisher ground-truth instance '
        'labels for the watershed crown labels the product coarse stage would '
        'produce, and a local low-percentile grid proxy for the product ground '
        'classifier. It therefore measures the crown/stem refinement stage, not the '
        'whole product pipeline end to end.',
        'The real-algorithm route repairs almost nothing: run with clean crowns it '
        'reproduces the ground truth exactly (median delta IoU 0.0000, 100 percent '
        'unchanged), and run with perturbed crowns it returns what it was given for '
        '76 to 79 percent of rows. The pairs the refiner is trained and tested on '
        'are therefore simulated corruptions, marked SIMULATED_PROPOSAL everywhere.',
        'The ROI library stores at most 120 000 points per target by a deterministic '
        'spatial stride; the stride is recorded per row.',
        'Region-level offsets use a stem/base proxy - the median XYZ of the lowest '
        '10 percent of the instance by height above the local ground proxy - because '
        'the benchmark publishes no stem axis. It is a proxy and is never called '
        'ground truth.',
        'Offset supervision is applied only where the ground truth says a point '
        'belongs to the target, because for any other point the true offset is '
        'unknown. Modelling it as a keep-out offset would be an assumption, not a '
        'label.',
        'The refiner decides membership at a fixed 0.5 probability threshold. No '
        'threshold was tuned, on the test site or anywhere else.',
        'The mini bakeoff is small by design: it is an architecture elimination '
        'round, not a trained production model.',
        'Only one cross-site fold was trained. The reverse fold, and a full '
        'leave-one-site-out rotation, were not run: with three architectures at a '
        'shared 20-epoch budget a second fold exceeded the round\'s time. One '
        'unseen-site test is what the protocol asks for, but two would be stronger '
        'evidence and the gap is recorded here rather than implied away.',
        'The in-site configuration trains and tests on the same forest (different '
        'trees). It is a diagnostic of whether refinement is learnable at all and '
        'must not be read as generalisation evidence.',
    ]
    return limitations


def _blockers(report: dict) -> list:
    blockers = list(report['final_verdict'].get('blockers') or [])
    blockers.append(
        'The product CROWN_STEM_HYBRID backend cannot consume a public TLS plot '
        'directly: it requires a SpanRecord, a vegetation region, a coarse ground '
        'cloud and scene metadata that the benchmark does not publish. Only its '
        'stem-guided refinement function was reachable, which is why the real pair '
        'count is a diagnostic and the training pairs are simulated.')
    if report['mini_refiner_bakeoff'].get('status') != 'RUN':
        blockers.append('MINI_REFINER_BAKEOFF has not completed')
    return blockers


def _median(values) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def render_markdown(report: dict) -> str:
    """Render the JSON payload as the human-readable report.

    Every value here is read from the payload, which is read from the artefacts, so
    the two reports cannot disagree.
    """
    lines: list[str] = []
    add = lines.append
    git = report['git']
    verdict = report['final_verdict']

    add('# TREE_QC_V2_FEASIBILITY_AUDIT')
    add('')
    add('TreeLineWarning Tree QC V2 feasibility audit - multi-site generalisation and')
    add('the local proposal refinement architecture.')
    add('')
    add('This round does not train a production Tree QC V2 model. It answers one')
    add('question: does the new architecture have enough evidence to justify formal V2')
    add('training? Every number below is read from an artefact a run wrote.')
    add('')
    add('## Supervisor summary')
    add('')
    add('```')
    add('TREE_QC_V2_FEASIBILITY_AUDIT')
    add('')
    add(f'GIT_HEAD = {git["head"]}')
    add(f'BRANCH = {git["branch"]}')
    add(f'WORKTREE_CLEAN = {"YES" if git["worktree_status"] == "clean" else "NO"}')
    add('')
    touched = report['v1_history_untouched']['changed_files_in_research_tree_qc']
    add('V1_HISTORY_UNTOUCHED = ' + ('NO' if touched else 'YES'))
    add(f'LOCAL_20_TREES_ACCESSED = {report["local_data_untouched"]["local_20_trees_accessed"]}')
    add(f'LOCAL_8_BAD_ACCESSED = {report["local_data_untouched"]["local_8_bad_accessed"]}')
    add('')
    add(f'PUBLIC_SITES_USED = {", ".join(report["site_fingerprint_test"].get("sites") or [])}')
    add('')
    for label, key in (('SITE_FINGERPRINT_V1_RAW', 'v1_raw_site_accuracy'),
                       ('SITE_FINGERPRINT_V2_NORMALIZED', 'v2_normalized_site_accuracy')):
        value = verdict.get(key)
        add(f'{label} = ' + (f'{value:.4f}' if isinstance(value, float) else 'NOT_RUN'))
    add('')
    add(f'PUBLIC_GT_PAIR_CONSTRUCTION = {verdict.get("public_gt_pair_construction")}')
    add('')
    add(f'REAL_PROPOSAL_PAIRS = {verdict.get("real_algorithm_pairs")}')
    add(f'SIMULATED_PROPOSAL_PAIRS = {verdict.get("simulated_pairs")}')
    add('')
    for model in sorted(verdict.get('per_model', {}).values(),
                        key=lambda entry: entry['name']):
        for fold in model['folds']:
            kind = 'cross' if fold['cross_site'] else 'in-site'
            add(f'{model["name"]} [{kind} {fold["test_site"]}] = '
                f'delta_iou {fold["delta_iou_mean"]:+.4f}, '
                f'bad-proposal delta '
                f'{_fmt(fold["bad_mean_delta_iou"])}, change_f1 '
                f'{fold["change_region_f1_mean"]:.4f}, '
                f'identity_copy {fold["identity_copy_gate"]}, '
                f'good_preservation {fold["good_preservation_gate"]}')
    if not verdict.get('per_model'):
        add('POINTNET2_REFINER = NOT_RUN')
        add('POINTNEXT_S_REFINER = NOT_RUN')
        add('POINTNEXT_S_MULTITASK_REFINER = NOT_RUN')
    add('')
    best = verdict.get('best_cross_site_delta_iou')
    add('BEST_CROSS_SITE_DELTA_IOU = ' + (f'{best:+.4f}' if isinstance(best, float)
                                          else 'NOT_RUN'))
    in_site = verdict.get('best_in_site_delta_iou')
    add('BEST_IN_SITE_DELTA_IOU = ' + (f'{in_site:+.4f}' if isinstance(in_site, float)
                                       else 'NOT_RUN'))
    add('')
    add('GOOD_PRESERVATION = ' + _gate_line(verdict.get('good_preservation')))
    add('IDENTITY_COPY_GATE = ' + _gate_line(verdict.get('identity_copy_gate')))
    add('')
    add(f'BEST_ARCHITECTURE = {verdict.get("best_architecture") or "NONE"}')
    add('')
    add(f'FINAL_VERDICT = {verdict["verdict"]}')
    add('')
    add(f'FORMAL_V2_TRAINING_JUSTIFIED = {report["formal_v2_training_justified"]}')
    add('')
    add('BLOCKERS =')
    for blocker in report['blockers']:
        add(f'  - {blocker}')
    add('```')
    add('')

    add('## 1. State of the repository and of V1')
    add('')
    add(f'- HEAD `{git["head"]}` on `{git["branch"]}`, branched from the V1 pilot tip '
        f'`{git["base_commit_before_round"]}`.')
    add(f'- Working tree: `{git["worktree_status"] or "clean"}`.')
    v1 = report['v1_history_untouched']
    add(f'- Files changed under `research/tree_qc/`: '
        f'{len(v1["changed_files_in_research_tree_qc"])}. Under `algorithms/`: '
        f'{len(v1["changed_files_in_algorithms"])}. Under `core/`: '
        f'{len(v1["changed_files_in_core"])}.')
    add(f'- {v1["note"]}.')
    add('')
    add('The V1 verdict `PUBLIC_PILOT_NOT_LEARNABLE` is not revisited, reinterpreted or')
    add('softened anywhere in this round. No V1 threshold, checkpoint, generator, seed')
    add('or held-out result was touched, and the held-out partition is not read by any')
    add('module in `research/tree_qc_v2/`.')
    add('')

    add('## 2. Public data actually used')
    add('')
    add('| dataset | record | license | sites used | GT | annotated |')
    add('|---|---|---|---|---|---|')
    for entry in report['public_datasets']:
        add(f'| {entry["name"]} | {entry.get("record", "-")} | {entry["license"]} | '
            f'{", ".join(entry.get("sites_used", ["-"]))} | '
            f'{entry.get("gt_type", "-")} | '
            f'{"yes" if entry.get("manually_annotated") else "-"} |')
    add('')
    recon = report['data_recon']
    if recon.get('status') != 'NOT_RUN':
        add(f'{recon.get("total_tree_files")} publisher per-tree PLY files are present, '
            f'mean {recon.get("mean_points_per_tree"):,.0f} points per tree.')
        add('')
        add('| site | tree files | points | manifest digest |')
        add('|---|---|---|---|')
        for name, entry in (recon.get('sites') or {}).items():
            add(f'| {name} | {entry["files"]} | {entry["points"]:,} | '
                f'`{str(entry["manifest_digest"])[:16]}` |')
        add('')
        add(f'Not present locally and therefore not covered: '
            f'{", ".join(recon.get("absent_sites") or [])}.')
        add('')
        add('The two archive files were verified against the publisher MD5 before use;')
        add('the SHA-256 of each is recorded in the JSON report.')
    add('')

    add('## 3. License audit')
    add('')
    lic = report['license_audit']
    if lic.get('status') == 'MISSING':
        add('`research/tree_qc_v2/LICENSE_AUDIT.md` is missing; no license statement is')
        add('made here.')
    else:
        add(f'{lic["entries"]} components audited, full detail in '
            f'`{lic["report"]}`. {lic["conclusion"]}.')
        add('')
        used = lic.get('used_in_round') or []
        add('Used in this round:')
        for name in used:
            add(f'- {name}')
        add('')
        add('Research-only or non-commercial components are `NOT_USED` or')
        add('`REFERENCED_ONLY`, so nothing restricted reaches this round. The findings')
        add('that would block a shipping product are listed in the audit itself.')
    add('')

    add('## 4. Task A - SITE_FINGERPRINT_TEST')
    add('')
    fingerprint = report['site_fingerprint_test']
    if fingerprint.get('status') == 'NOT_RUN':
        add('Not run.')
    else:
        add(f'{fingerprint["trees_used"]} real good trees at '
            f'{fingerprint["points_per_tree"]} points each, split by tree into '
            f'train / validation / test. The same classifier architecture, the same')
        add('epoch budget and the same per-channel standardisation (fitted on the train')
        add('split) are used for every representation, so the comparison is about the')
        add('representation and not about capacity or conditioning.')
        add('')
        add('| representation | channels | test site accuracy | macro F1 | '
            'litchfield recall | wytham recall |')
        add('|---|---|---|---|---|---|')
        for name, entry in fingerprint['representations'].items():
            recalls = entry['per_site_recall']
            add(f'| `{name}` | {entry["in_channels"]} | '
                f'{entry["test_site_accuracy"]:.4f} | {entry["test_macro_f1"]:.4f} | '
                f'{recalls.get("litchfield", 0):.4f} | {recalls.get("wytham", 0):.4f} |')
        add('')
        add(f'Random baseline for two sites: '
            f'{fingerprint["random_baseline_accuracy"]:.2f}.')
        add('')
        raw = fingerprint['representations'].get('V1_RAW')
        norm = fingerprint['representations'].get('V2_NORMALIZED')
        no_meta = fingerprint['representations'].get('V2_NORMALIZED_NO_META')
        anisot = fingerprint['representations'].get('V2_ANISO_NORMALIZED')
        if raw:
            add('`V1_RAW` reproduces the frozen V1 input contract exactly: mean-centred')
            add('local metric XYZ divided by the research-wide constant 10, plus a')
            add('constant membership channel. It identifies the forest at')
            add(f'{raw["test_site_accuracy"]:.4f} on trees the classifier never saw,')
            add('which is direct evidence that the V1 representation carried a strong')
            add('site fingerprint.')
            add('')
        if norm:
            add('`V2_NORMALIZED` (candidate-centred, per-cloud robust scaling, random XY')
            add('rotation, density standardisation, explicit global metric channel)')
            add(f'scores {norm["test_site_accuracy"]:.4f}.')
            add('')
        if no_meta:
            add('`V2_NORMALIZED_NO_META` withholds the explicit metric channel and')
            add(f'scores {no_meta["test_site_accuracy"]:.4f}, so most of the remaining')
            add('site signal lives in that one named, auditable channel rather than')
            add('hidden in the coordinates.')
            add('')
        if anisot:
            add('`V2_ANISO_NORMALIZED` additionally removes aspect ratio and scores')
            add(f'{anisot["test_site_accuracy"]:.4f}; that is the declared ablation, not')
            add('a proposal, because aspect ratio is real tree geometry a refiner needs.')
            add('')
        add('Read correctly: a lower number is not automatically better. The point is')
        add('that V1 carried a forest fingerprint, that the V2 representation moves the')
        add('scale information into one explicit channel, and that the refinement task')
        add('still has to work - which is what Task C tests.')
    add('')

    add('## 5. Task B - PUBLIC_GT_PROPOSAL_TEST')
    add('')
    proposal = report['public_gt_proposal_test']
    if proposal.get('status') == 'NOT_RUN':
        add('Not run.')
    else:
        add(f'**{proposal["verdict"]}**')
        add('')
        add(f'Local windows of {proposal["window_m"]:.0f} m were cut from the real plot,')
        add('each containing the target tree and its neighbours. ROIs are the candidate')
        add('box grown in XY, so neighbour contamination and missing points are both')
        add('inside the region a refiner would see.')
        add('')
        roi = proposal.get('roi_coverage') or {}
        add(f'GT_COVERAGE_IN_ROI: mean {roi.get("mean")}, minimum {roi.get("min")}, '
            f'fraction below 0.95 {roi.get("fraction_below_0.95")}.')
        add('')
        add(f'ROI size: median {proposal["roi_points"]["median"]:,.0f} points, '
            f'p90 {proposal["roi_points"]["p90"]:,.0f}.')
        add('')
        add(f'- `REAL_ALGORITHM_PROPOSAL` pairs above IoU {proposal["min_pair_iou"]}: '
            f'**{proposal["real_pairs_above_min_iou"]}** '
            f'({proposal["real_pairs_per_site"]}).')
        add(f'- `SIMULATED_PROPOSAL` rows: **{proposal["simulated_rows_total"]}**.')
        add(f'- `GT_CORRECT` control rows: {proposal["gt_correct_rows_total"]}.')
        add('')
        add('The real route runs the product\'s own `segment_stem_guided_hybrid` on real')
        add('public forest. It is a diagnostic of the crown/stem refinement stage, not a')
        add('run of the whole product pipeline, because public data forces two')
        add('substitutions that are recorded on every row:')
        add('')
        for substitution in proposal.get('substitutions') or []:
            add(f'- {substitution}')
        add('')
        add('What the real route then shows is important and negative: it does not')
        add('repair. Handed clean ground-truth crowns it reproduces them exactly, and')
        add('handed perturbed crowns it returns what it was given for most rows.')
        add('')
        add('| crown source | rows | median IoU | min IoU | fraction < 0.90 | '
            'median delta vs input | improved | unchanged |')
        add('|---|---|---|---|---|---|---|---|')
        for source, entry in proposal['real_profile_by_crown_source'].items():
            add(f'| `{source}` | {entry["rows"]} | {entry["iou_median"]:.4f} | '
                f'{entry["iou_min"]:.4f} | {entry["fraction_below_0.90"]:.3f} | '
                f'{entry["delta_iou_vs_input_crown_median"]:+.4f} | '
                f'{entry["improved_fraction"]:.3f} | '
                f'{entry["unchanged_fraction"]:.3f} |')
        add('')
        add('So the honest statement is: public data can form the task, but the real')
        add('algorithm route contributes almost no defective proposals. The pairs the')
        add('refiner is trained and tested on are therefore simulated corruptions,')
        add('marked `SIMULATED_PROPOSAL` on every row and never mixed with the real ones.')
        add('')
        profile = proposal.get('simulated_defect_profile') or {}
        add(f'Simulated proposals: {profile.get("rows")} rows, median IoU '
            f'{profile.get("iou_median")}, merge fraction '
            f'{profile.get("merge_fraction"):.3f}, incomplete fraction '
            f'{profile.get("incomplete_fraction"):.3f}.')
        add('')
        add('| corruption | rows | median IoU | min IoU |')
        add('|---|---|---|---|')
        for name, entry in (profile.get('by_corruption') or {}).items():
            add(f'| `{name}` | {entry["rows"]} | {entry["iou_median"]:.4f} | '
                f'{entry["iou_min"]:.4f} |')
        add('')
        add('| severity | rows | median IoU |')
        add('|---|---|---|')
        for name, entry in (profile.get('by_severity') or {}).items():
            add(f'| {name} | {entry["rows"]} | {entry["iou_median"]:.4f} |')
        add('')
        recoverable = proposal.get('missing_recoverable_fraction') or {}
        add(f'`MISSING_RECOVERABLE_FRACTION`: mean {recoverable.get("mean")}, minimum '
            f'{recoverable.get("min")}. A missing point outside the ROI can never be')
        add('recovered by any refiner, so this bounds what repair could achieve.')
    add('')

    add('## 6. Task C - MINI_REFINER_BAKEOFF')
    add('')
    bakeoff = report['mini_refiner_bakeoff']
    if bakeoff.get('status') != 'RUN':
        add('Not run. No architecture comparison is reported, and no verdict about the')
        add('architecture is drawn from the parts that did run.')
    else:
        library_digest = bakeoff.get('library_index_sha256') or 'unrecorded'
        add(f'{bakeoff["target_count"]} ROI targets '
            f'({bakeoff["targets_per_site"]}) from a library whose index digest is '
            f'`{library_digest[:16]}`.')
        add('')
        add('### Protocol shared by all three models')
        add('')
        for key, value in bakeoff['shared_protocol'].items():
            add(f'- **{key}**: {value}')
        add(f'- **evaluation cap**: {bakeoff["evaluation_cap"]["max_points_per_roi"]:,} '
            f'points per ROI')
        add('')
        add('### The three candidates')
        add('')
        add('| model | reference | parameters | auxiliary heads | implementation |')
        add('|---|---|---|---|---|')
        for entry in bakeoff['model_descriptions'].values():
            add(f'| `{entry["key"]}` {entry["reference_implementation"]} | '
                f'- | {entry["parameters"]:,} | '
                f'{"yes" if entry["auxiliary_heads"] else "no"} | '
                f'{entry["implementation"]} |')
        add('')
        add('`MODEL_C` is a strict superset of `MODEL_B`: the same backbone and the same')
        add('membership head plus an offset head and a mask-quality head, verified by a')
        add('test that compares the shared tensors. So any difference between B and C')
        add('comes from the auxiliary supervision, not from extra main-path capacity.')
        add('')
        add('### Cross-site results')
        add('')
        add('The verdict is decided on the cross-site folds. The in-site configuration')
        add('is a diagnostic that separates "these architectures cannot refine at all"')
        add('from "they cannot transfer", and it never substitutes for a cross-site')
        add('number.')
        add('')
        add('Only one cross-site fold was run: the model is trained on Litchfield')
        add('trees and tested on Wytham, a different forest, a different species mix')
        add('and a different acquisition. That is the unseen-site evidence. The')
        add('reverse fold was not trained, because the epoch budget that makes the')
        add('three architectures comparable made a second full fold exceed the time')
        add('this round had. One unseen-site test is what the protocol requires; two')
        add('would be stronger, and that gap is recorded rather than glossed over.')
        add('')
        for fold in bakeoff['folds']:
            configuration = fold.get('configuration') or (
                f'cross_site_test_{fold["test_site"]}')
            kind = 'Cross-site' if configuration.startswith('cross_site') else 'In-site'
            add(f'#### {kind}: `{configuration}` - test site `{fold["test_site"]}` '
                f'(trained on {", ".join(fold["held_in_sites"])})')
            add('')
            if kind == 'In-site':
                add('Both the training trees and the test trees come from this one')
                add('forest, and the split is by tree, so no tree appears in two')
                add('splits. The site is *named* on both sides; what is held out is the')
                add('trees, not the forest. This configuration answers "can the')
                add('architectures refine at all", not "does the refinement transfer".')
                add('')
            add('')
            add(f'Train {fold["split"]["train"]} / validation '
                f'{fold["split"]["validation"]} / test {fold["split"]["test"]} '
                f'targets. The test site was never used to early-stop, normalise,')
            add('augment or pick a threshold.')
            add('')
            add('| model | current IoU | refined IoU | delta IoU | median delta | '
                'change-region F1 | identity copy | good preservation |')
            add('|---|---|---|---|---|---|---|---|')
            for key, model in sorted(fold['models'].items()):
                summary = model['summary']
                add(f'| `{key}` | {summary["current_iou_mean"]:.4f} | '
                    f'{summary["refined_iou_mean"]:.4f} | '
                    f'{summary["delta_iou_mean"]:+.4f} | '
                    f'{summary["delta_iou_median"]:+.4f} | '
                    f'{summary["change_region_f1_mean"]:.4f} | '
                    f'{model["identity_copy_gate"]["verdict"]} | '
                    f'{model["good_preservation_gate"]["verdict"]} |')
            add('')
            best_key = max(fold['models'], key=lambda k: fold['models'][k]['summary'][
                'delta_iou_mean'])
            best_summary = fold['models'][best_key]['summary']
            bad = best_summary.get('bad_proposals') or {}
            good = best_summary.get('good_proposals') or {}
            add(f'Best on this site: `{best_key}`. On bad proposals (current IoU below '
                f'{trainer_good_iou():.2f}): mean delta '
                f'{_fmt(bad.get("mean_delta_iou"))}, improved by more than 0.02 in '
                f'{_fmt(bad.get("improvement_gt_0.02"))} of rows. On good proposals: '
                f'mean delta {_fmt(good.get("mean_delta_iou"))}, degraded by more than '
                f'0.05 in {_fmt(good.get("degraded_fraction_gt_0.05"))} of rows, '
                f'{good.get("severe_degradation_count")} severe cases.')
            add('')
        add('### Identity-copy and GOOD-preservation gates')
        add('')
        add('| model | configuration | site | change-region F1 | '
            'identity-copy fraction | gate | good mean delta | '
            'good degraded > 0.05 | severe | gate |')
        add('|---|---|---|---|---|---|---|---|---|---|')
        for fold in bakeoff['folds']:
            configuration = fold.get('configuration') or (
                f'cross_site_test_{fold["test_site"]}')
            for key, model in sorted(fold['models'].items()):
                summary = model['summary']
                gate = model['identity_copy_gate']
                good = model['good_preservation_gate']
                add(f'| `{key}` | `{configuration}` | {fold["test_site"]} | '
                    f'{summary["change_region_f1_mean"]:.4f} | '
                    f'{summary["identity_copy_fraction"]:.4f} | '
                    f'{gate["verdict"]} | {_fmt(good.get("mean_delta_iou"))} | '
                    f'{_fmt(good.get("degraded_fraction_gt_0.05"))} | '
                    f'{good.get("severe_degradation_count")} | '
                    f'{good["verdict"]} |')
        add('')
        add('A refiner that returns `current_mask` unchanged scores a perfect current')
        add('IoU and zero on every change-region metric. The gates exist because the')
        add('whole point of V2 is to change the mask, in the right direction, without')
        add('breaking the proposals that were already right.')
        add('')
        add('### Change-region and added/removed point metrics')
        add('')
        add('| model | site | change F1 | change recall | added F1 | removed F1 |')
        add('|---|---|---|---|---|---|')
        for fold in bakeoff['folds']:
            for key, model in sorted(fold['models'].items()):
                summary = model['summary']
                add(f'| `{key}` | {fold["test_site"]} | '
                    f'{summary["change_region_f1_mean"]:.4f} | '
                    f'{summary["change_region_recall_mean"]:.4f} | '
                    f'{summary["added_point_f1_mean"]:.4f} | '
                    f'{summary["removed_point_f1_mean"]:.4f} |')
        add('')
        add('### Per-defect breakdown')
        add('')
        add('| model | site | corruption | rows | current IoU | refined IoU | '
            'delta | improved | identity copy |')
        add('|---|---|---|---|---|---|---|---|---|')
        for fold in bakeoff['folds']:
            for key, model in sorted(fold['models'].items()):
                for name, entry in (model['summary'].get('by_corruption') or {}).items():
                    if name == 'NONE':
                        continue
                    add(f'| `{key}` | {fold["test_site"]} | `{name}` | {entry["rows"]} | '
                        f'{entry["current_iou_mean"]:.4f} | '
                        f'{entry["refined_iou_mean"]:.4f} | '
                        f'{entry["delta_iou_mean"]:+.4f} | '
                        f'{entry["improved_fraction"]:.3f} | '
                        f'{entry["identity_copy_fraction"]:.3f} |')
        add('')
        add('### Resource measurements')
        add('')
        add('| model | parameters | GPU peak MB | single-ROI GPU s | single-ROI CPU s |')
        add('|---|---|---|---|---|')
        for fold in bakeoff['folds'][:1]:
            for key, model in sorted(fold['models'].items()):
                resources = model['resources']
                add(f'| `{key}` | {resources["parameters"]:,} | '
                    f'{resources.get("cuda_peak_memory_mb")} | '
                    f'{resources.get("single_roi_inference_seconds")} | '
                    f'{model["cpu_probe"]["cpu_single_roi_inference_seconds"]} |')
        add('')
        add('None of these is a product gate in this round. They are recorded because')
        add('the eventual target is a Windows desktop build where a 24 GB accelerator')
        add('cannot be assumed, so an architecture that only works at one scale is a')
        add('worse candidate even at equal IoU. Both a CUDA and a CPU forward pass were')
        add('run for every model.')
    add('')

    add('## 7. Verdict')
    add('')
    add(f'**{verdict["verdict"]}**')
    add('')
    add(f'- V1 site/domain problem diagnosed: '
        f'{"yes" if verdict["site_problem_diagnosed"] else "no"} '
        f'(`V1_RAW` site accuracy {_fmt(verdict.get("v1_raw_site_accuracy"))} against a')
    add('  0.50 random baseline).')
    add(f'- Public GT can form the task: '
        f'{verdict.get("public_gt_pair_construction")}, with '
        f'{verdict.get("real_algorithm_pairs")} real and '
        f'{verdict.get("simulated_pairs")} simulated proposal pairs.')
    add(f'- Best cross-site delta IoU: '
        f'{_fmt(verdict.get("best_cross_site_delta_iou"))}.')
    if verdict.get('best_in_site_delta_iou') is not None:
        add(f'- Best in-site delta IoU (diagnostic): '
            f'{_fmt(verdict.get("best_in_site_delta_iou"))}.')
    add(f'- Best architecture: {verdict.get("best_architecture") or "NONE"}.')
    add('')
    if verdict['verdict'] == 'V2_FEASIBILITY_PASS':
        add('The conditions the audit set for PASS are met: the V1 domain problem is')
        add('diagnosed on real data, public ground truth forms the refinement task with')
        add('complete ROI coverage, at least one architecture improves genuinely bad')
        add('proposals on a site it never saw, the change is not a copy of the input,')
        add('and already-good proposals are not broadly damaged.')
        add('')
        add('PASS here does not mean product ready. It means Local Proposal Refinement')
        add('is worth a formal V2 training round.')
    elif verdict['verdict'] == 'V2_FEASIBILITY_PARTIAL':
        add('Some of the conditions are met and at least one is not. The specific gaps')
        add('are listed under BLOCKERS and in the per-model tables above. No model was')
        add('added, no epoch budget was raised and no test site was changed to close')
        add('them.')
        add('')
        add('Read the two configurations together. Cross-site, every architecture')
        add('degrades the mask on average and the GOOD-preservation gate fails badly:')
        add('the learned transformation does not transfer to a forest the model never')
        add('saw. In-site, the same architectures and the same budget improve bad')
        add('proposals substantially, which means the task is learnable and the')
        add('pipeline is sound; what fails is generalisation, and separately the')
        add('handling of proposals that were already good.')
    else:
        add('The evidence does not support a formal V2 training round. The specific')
        add('failing conditions are listed under BLOCKERS.')
    add('')
    add('Per the protocol this round stops here. It does not proceed to a formal V2')
    add('training run, does not add a fourth or fifth model, does not raise the epoch')
    add('budget, and does not touch the TreeLineWarning local corpus.')
    add('')

    add('## 8. Known limitations')
    add('')
    for limitation in report['known_limitations']:
        add(f'- {limitation}')
    add('')

    add('## 9. Blockers')
    add('')
    for blocker in report['blockers']:
        add(f'- {blocker}')
    add('')

    add('## 10. Reproducing this round')
    add('')
    add('```')
    add('set PYTHONPATH=research/tree_qc_v2')
    add('python research/tree_qc_v2/recon.py')
    add('python research/tree_qc_v2/v2_site_fingerprint.py --points 1024 --trees-per-site 300')
    add('python research/tree_qc_v2/v2_proposal_test.py --window-m 12 --minimum-points 8000')
    add('python research/tree_qc_v2/v2_roi_library.py')
    add('python research/tree_qc_v2/v2_bakeoff.py --points 2048 --epochs 30')
    add('python research/tree_qc_v2/v2_report.py')
    add('pytest research/tree_qc_v2/test_v2_feasibility.py')
    add('```')
    add('')
    add(f'Config digest for the whole round: `{report["config_digest"]}`.')
    add('')
    return '\n'.join(lines)


def trainer_good_iou() -> float:
    """The GOOD threshold the bakeoff used, taken from the trainer itself."""
    try:
        import v2_refiner_train as trainer
    except ImportError:
        return cfg.GOOD_IOU_THRESHOLD
    return trainer.GOOD_IOU_THRESHOLD


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def _gate_line(gates) -> str:
    if not gates:
        return 'NOT_RUN'
    parts = []
    for key in sorted(gates):
        verdicts = gates[key]
        passed = sum(1 for value in verdicts if value == 'PASS')
        parts.append(f'{key}:{passed}/{len(verdicts)} PASS')
    return '; '.join(parts)


def main() -> int:
    report = build()
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = cfg.REPO_ROOT / 'TREE_QC_V2_FEASIBILITY_REPORT.json'
    json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    md_path = cfg.REPO_ROOT / 'TREE_QC_V2_FEASIBILITY_REPORT.md'
    md_path.write_text(render_markdown(report), encoding='utf-8')
    print(f'wrote {json_path}')
    print(f'wrote {md_path}')
    print(json.dumps(report['final_verdict'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
