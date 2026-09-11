"""Public Pilot reporting: the frozen field list, verdicts and stop reports.

Every number in the final report is copied from an artifact that already exists on disk —
``PUBLIC_EXPERIMENT_FREEZE.json``, the per-seed threshold files, the held-out evaluation
records — or is ``NOT_RUN`` with a cause. Nothing is recomputed after the fact, and no
metric is ever filled in from memory, because a report that can invent a number is not
evidence.

The verdict vocabulary is closed. ``READY_FOR_GATE_H_LOCAL_BLIND`` requires *all* of:
primary binary targets met on held-out moderate/high defects, acceptable multi-seed
stability, sampling recall drop within 10 percentage points, no critical leakage and a
clean held-out protocol. Anything else is a stop code or one of the three negative
verdicts. ``PRODUCT_READY`` is not in the vocabulary.
"""
from __future__ import annotations

import json
from pathlib import Path

import efg_gates as gates
import efg_state as state_module

REPORT_VERSION = 'EFG_FINAL_REPORT_V1'
NOT_RUN = 'NOT_RUN'
UNDEFINED = None


def _value(payload, *path, default=NOT_RUN):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def decide_verdict(stop_code, heldout_accessed: bool, primary_gate: dict | None,
                   stability: dict | None, robustness: dict | None,
                   leakage: dict | None, learnability: dict | None,
                   heldout_complete: bool = True) -> dict:
    """The single place a Public Pilot verdict is produced."""
    if stop_code:
        return {'verdict': f'STOP_{stop_code}' if not str(stop_code).startswith('STOP_')
                           else str(stop_code),
                'basis': [f'fail-closed stop: {stop_code}'], 'terminal': True}
    if learnability and learnability.get('evaluated') and not learnability.get('passed'):
        return {'verdict': gates.VERDICT_NOT_LEARNABLE,
                'basis': ['validation learnability gate not met',
                          json.dumps(learnability.get('observed'))],
                'terminal': True}
    if not heldout_accessed:
        return {'verdict': NOT_RUN, 'basis': ['held-out was never accessed'],
                'terminal': False}
    bases: list[str] = []
    if primary_gate is None or not primary_gate.get('evaluated'):
        return {'verdict': NOT_RUN, 'basis': ['held-out primary metrics unavailable'],
                'terminal': False}
    if not heldout_complete:
        # A partial held-out evaluation can never be a continuation verdict, however good
        # the numbers on the scored subset look.
        return {'verdict': 'STOP_PARTIAL_HELDOUT_EVALUATION',
                'basis': ['held-out was evaluated on a proper subset of the frozen '
                          'candidates; this is a rehearsal, not a scientific result'],
                'terminal': True}
    bases.append('primary gate: ' + ('PASS' if primary_gate['passed'] else
                                     'FAIL ' + ','.join(primary_gate['failed'])))
    gate_ok = bool(primary_gate['passed'])

    stability_ok = bool(stability and stability.get('MODEL_STABILITY') == 'PASS')
    bases.append(f"model stability: {stability.get('MODEL_STABILITY') if stability else NOT_RUN}")

    robustness_ok = bool(robustness and robustness.get('SAMPLING_ROBUSTNESS') == 'PASS')
    bases.append('sampling robustness: '
                 f"{robustness.get('SAMPLING_ROBUSTNESS') if robustness else NOT_RUN}")

    leakage_ok = bool(leakage and leakage.get('critical_leakage') == 'none')
    bases.append('critical leakage: '
                 f"{(leakage or {}).get('critical_leakage', NOT_RUN)}")
    protocol_ok = bool(leakage and leakage.get('heldout_protocol_clean') is True)
    bases.append('held-out protocol: '
                 f"{(leakage or {}).get('heldout_protocol', NOT_RUN)}")

    meaningful = (leakage or {}).get('meaningful_defects') or {}
    bases.append('meaningful defects: '
                 + (', '.join(f'{k}={v}' for k, v in sorted(meaningful.items()))
                    if meaningful else NOT_RUN))

    if not gate_ok:
        return {'verdict': gates.VERDICT_NOT_LEARNABLE,
                'basis': bases + ['held-out primary targets not met on moderate/high '
                                  'defects: ' + ','.join(primary_gate.get('failed') or [])],
                'terminal': True}
    if not stability_ok:
        return {'verdict': gates.VERDICT_UNSTABLE, 'basis': bases, 'terminal': True}
    if not robustness_ok:
        return {'verdict': gates.VERDICT_SAMPLING_TOO_SENSITIVE, 'basis': bases,
                'terminal': True}
    if not (leakage_ok and protocol_ok):
        return {'verdict': 'STOP_CRITICAL_LEAKAGE', 'basis': bases, 'terminal': True}
    if not meaningful or not all(bool(v) for v in meaningful.values()):
        return {'verdict': gates.VERDICT_NOT_LEARNABLE,
                'basis': bases + ['the critical defect classes (MERGED_MULTI_TREE, '
                                  'NEIGHBOR_CONTAMINATION, OVER_SEGMENTED) are not '
                                  'meaningfully detected'],
                'terminal': True}
    return {'verdict': gates.VERDICT_READY, 'basis': bases, 'terminal': True}


def _primary_failure_verdict(primary_gate: dict) -> str:
    failed = set(primary_gate.get('failed') or [])
    return 'PUBLIC_PILOT_PRIMARY_GATE_FAILED:' + ','.join(sorted(failed))


def build_report(context: dict) -> dict:
    """Assemble TREE_QC_EFG_FINAL_REPORT from the frozen artifacts in ``context``."""
    freeze = context.get('freeze') or {}
    heldout = context.get('heldout') or {}
    comparison = context.get('comparison') or {}
    runs = context.get('runs') or {}
    tests = context.get('tests') or {}
    resources = context.get('resources') or {}
    calibration = context.get('calibration') or {}
    verdict = context.get('verdict') or {}
    primary = (heldout.get('primary_moderate_high') or {})
    multi = heldout.get('multi_seed') or {}
    stability = multi.get('metrics', {}).get('BAD_RECALL', {}) if multi else {}
    robustness = heldout.get('robustness') or {}
    leakage = heldout.get('leakage') or {}

    report = {
        'report_version': REPORT_VERSION,
        'generated_at': state_module.local_timestamp(),
        'run_id': context.get('run_id'),
        'BASE_ACCEPTED_HEAD': context.get('base_accepted_head'),
        'IMPLEMENTATION_HEAD': context.get('implementation_head'),
        'ORCHESTRATOR_VERSION': context.get('orchestrator_version'),
        'GATE_D_ACCEPTANCE': context.get('gate_d_acceptance', NOT_RUN),
        'GATE_D_GENERATOR_DIGEST': _value(freeze, 'gate_d_binding',
                                          'generator_digest',
                                          default=context.get('gate_d_generator_digest',
                                                              NOT_RUN)),
        'GATE_D_PROTOCOL_DIGEST': _value(freeze, 'gate_d_binding', 'protocol_digest',
                                         default=context.get('gate_d_protocol_digest',
                                                             NOT_RUN)),
        'GATE_D_CORE_CHANGED': context.get('gate_d_core_changed', NOT_RUN),
        'AUTOPILOT': context.get('autopilot', NOT_RUN),
        'RESUME_TEST': tests.get('RESUME_TEST', NOT_RUN),
        'SAFE_BATCH_POINTNET': _value(calibration, 'models', 'pointnet', 'safe_batch',
                                      default=NOT_RUN),
        'SAFE_BATCH_POINTNET2': _value(calibration, 'models', 'pointnet2_ssg',
                                       'safe_batch', default=NOT_RUN),
        'COMMON_TRAIN_BATCH': calibration.get('safe_common_batch', NOT_RUN),
        'POINTNET_SEED101': comparison.get('pointnet', NOT_RUN),
        'POINTNET2_SEED101': comparison.get('pointnet2_ssg', NOT_RUN),
        'SELECTED_ARCHITECTURE': freeze.get('selected_architecture', NOT_RUN),
        'ARCHITECTURE_SELECTION_BASIS': freeze.get('architecture_selection_basis',
                                                   NOT_RUN),
        'MODEL_SEEDS': freeze.get('model_seeds', list(gates.MODEL_SEEDS)),
        'ALL_MODEL_SEEDS_FROZEN_BEFORE_HELDOUT': context.get(
            'all_model_seeds_frozen_before_heldout', NOT_RUN),
        'DEPLOYMENT_CANDIDATE_SEED': freeze.get('deployment_candidate_seed', NOT_RUN),
        'PUBLIC_EXPERIMENT_FREEZE_SHA256': freeze.get('freeze_sha256', NOT_RUN),
        'HELDOUT_ADAPTER_PARITY': _value(heldout, 'adapter_parity',
                                         'HELDOUT_ADAPTER_PARITY', default=NOT_RUN),
        'HELDOUT_ACCESSED_BEFORE_FREEZE': context.get('heldout_accessed_before_freeze',
                                                       'NO'),
        'HELDOUT_ACCESSED': 'YES' if heldout.get('accessed') else 'NO',
        'HELDOUT_CANDIDATES': heldout.get('candidates', NOT_RUN),
        'HELDOUT_EVALUATED_CANDIDATES': heldout.get('evaluated_candidates', NOT_RUN),
        'HELDOUT_COMPLETE': heldout.get('complete', NOT_RUN),
        'HELDOUT_EXCLUSIONS': heldout.get('exclusions', NOT_RUN),
        'HELDOUT_PRIMARY_BAD_RECALL': primary.get('BAD_RECALL', NOT_RUN),
        'HELDOUT_PRIMARY_BAD_FALSE_PASS': primary.get('BAD_FALSE_PASS', NOT_RUN),
        'HELDOUT_PRIMARY_GOOD_FALSE_REVIEW': primary.get('GOOD_FALSE_REVIEW', NOT_RUN),
        'HELDOUT_PRIMARY_BALANCED_ACCURACY': primary.get('BALANCED_ACCURACY', NOT_RUN),
        'HELDOUT_PRIMARY_F1_REVIEW': primary.get('F1_REVIEW', NOT_RUN),
        'HELDOUT_PRIMARY_EXAMPLES': primary.get('examples', NOT_RUN),
        'HELDOUT_PER_DEFECT_METRICS': heldout.get('per_defect_primary', NOT_RUN),
        'HELDOUT_PER_DEFECT_ALL_SEVERITIES': heldout.get('per_defect_all_severities',
                                                         NOT_RUN),
        'HELDOUT_PER_SEVERITY': heldout.get('per_severity', NOT_RUN),
        'HELDOUT_THREE_WAY': heldout.get('three_way', NOT_RUN),
        'HELDOUT_OVERALL_BINARY': heldout.get('overall', NOT_RUN),
        'PER_DEFECT_METRICS': heldout.get('per_defect_primary', NOT_RUN),
        'MULTI_SEED_BAD_RECALL_MEAN': stability.get('mean', NOT_RUN),
        'MULTI_SEED_BAD_RECALL_MIN': stability.get('min', NOT_RUN),
        'MULTI_SEED_BAD_RECALL_MAX': stability.get('max', NOT_RUN),
        'MULTI_SEED_BAD_RECALL_STD': stability.get('std', NOT_RUN),
        'MULTI_SEED_METRICS': multi.get('metrics', NOT_RUN),
        'MULTI_SEED_PER_SEED': heldout.get('per_seed', NOT_RUN),
        'MODEL_STABILITY': multi.get('MODEL_STABILITY', NOT_RUN),
        'ROBUSTNESS_4096': robustness.get('configurations', {}).get('4096_sampling_seed_0',
                                                                    NOT_RUN),
        'ROBUSTNESS_2048': {k: v for k, v in
                            (robustness.get('configurations') or {}).items()
                            if str(k).startswith('2048')} or NOT_RUN,
        'ROBUSTNESS_CONFIGURATIONS': robustness.get('configurations', NOT_RUN),
        'MAX_BAD_RECALL_DROP': robustness.get('max_bad_recall_drop_pp', NOT_RUN),
        'SAMPLING_ROBUSTNESS': robustness.get('SAMPLING_ROBUSTNESS', NOT_RUN),
        'VALIDATION_METRICS_PER_SEED': freeze.get('validation_metrics_per_seed',
                                                   NOT_RUN),
        'COMPARISON_EVIDENCE': comparison.get('evidence', NOT_RUN),
        'LEAKAGE_AUDIT': leakage,
        'LOCAL_8_BAD_CHALLENGE_ACCESSED': 'NO',
        'ONNX_STARTED': 'NO',
        'PRODUCT_INTEGRATION': 'NO',
        'LOCAL_8_BAD_CHALLENGE_READS': 0,
        'GATE_D_TESTS': tests.get('GATE_D_TESTS', NOT_RUN),
        'GATE_C2_TESTS': tests.get('GATE_C2_TESTS', NOT_RUN),
        'EFG_TESTS': tests.get('EFG_TESTS', NOT_RUN),
        'RUFF': tests.get('RUFF', NOT_RUN),
        'WORKTREE_CLEAN': tests.get('WORKTREE_CLEAN', NOT_RUN),
        'ENVIRONMENT': {
            'gpu': _value(resources, 'gpu', 'device_name', default=NOT_RUN),
            'vram_total_bytes': _value(resources, 'gpu', 'vram_total_bytes',
                                       default=NOT_RUN),
            'ram_total_bytes': resources.get('ram_total_bytes', NOT_RUN),
            'torch': _value(resources, 'gpu', 'torch', default=NOT_RUN),
            'driver': _value(resources, 'driver', 'nvidia_smi', default=NOT_RUN),
            'disks': resources.get('disks', NOT_RUN),
        },
        'FROZEN_THRESHOLDS': freeze.get('frozen_thresholds', NOT_RUN),
        'TRAINING_CONFIG': freeze.get('training_config', NOT_RUN),
        'RUNS': {str(seed): {k: v for k, v in (runs.get(seed) or {}).items()
                             if k not in ('history',)}
                 for seed in sorted(runs, key=lambda s: int(s))} if runs else NOT_RUN,
        'PUBLIC_PILOT_STATUS': context.get('public_pilot_status', NOT_RUN),
        'CURRENT_STAGE': context.get('current_stage', NOT_RUN),
        'RESUME_READY': context.get('resume_ready', NOT_RUN),
        'STOP': context.get('stop'),
        'API_INTERFACE_RESERVED': interface_contract(),
        'VERDICT': verdict.get('verdict', NOT_RUN),
        'VERDICT_BASIS': verdict.get('basis', []),
    }
    report['REPORT_SHA256'] = state_module.sha256_text(state_module.canonical(report))
    return report


def interface_contract() -> dict:
    """The frozen output contract a future product consumes. No product code here."""
    return {
        'status': 'reserved_not_implemented',
        'purpose': 'let a future TreeRepairEngine map REVIEW_REQUIRED to a local '
                   'resegmentation attempt without changing the QC model',
        'fields': {
            'persistent_tree_id': 'canonical tree identity / evaluation key',
            'qc_score': 'defect probability in [0, 1]',
            'binary_decision': 'PASS | REVIEW_REQUIRED at the frozen threshold',
            'three_way_decision': 'PASS | UNCERTAIN | REVIEW_REQUIRED at the frozen '
                                  'uncertain band',
            'threshold': 'the frozen per-seed operating point',
            'model_name': 'pointnet | pointnet2_ssg',
            'model_version': 'trainer version plus checkpoint digest',
            'checkpoint_digest': 'SHA-256 of the deployed checkpoint',
            'sampler_version': 'frozen sampler version',
            'points': 'sampling target, 4096 primary',
            'evidence_metadata': 'per-example sampler diagnostics and provenance row',
        },
        'future_repair_contract': {
            'AUTO_RESEGMENT_MAX_ATTEMPTS': 1,
            'note': 'no QC -> resegment -> QC -> resegment loop is permitted',
            'implemented_this_round': False,
        },
        'implemented_this_round': False,
    }


def render_markdown(report: dict) -> str:
    """Compact human-readable companion to the JSON report."""
    lines = ['# TREE_QC_EFG_FINAL_REPORT', '',
             f"- run_id: `{report.get('run_id')}`",
             f"- generated_at: `{report.get('generated_at')}`",
             f"- verdict: **{report.get('VERDICT')}**",
             f"- public pilot status: `{report.get('PUBLIC_PILOT_STATUS')}`",
             f"- current stage: `{report.get('CURRENT_STAGE')}`", '',
             '## Binding', '',
             f"- BASE_ACCEPTED_HEAD: `{report.get('BASE_ACCEPTED_HEAD')}`",
             f"- IMPLEMENTATION_HEAD: `{report.get('IMPLEMENTATION_HEAD')}`",
             f"- GATE_D_ACCEPTANCE: `{report.get('GATE_D_ACCEPTANCE')}`",
             f"- GATE_D_GENERATOR_DIGEST: `{report.get('GATE_D_GENERATOR_DIGEST')}`",
             f"- GATE_D_PROTOCOL_DIGEST: `{report.get('GATE_D_PROTOCOL_DIGEST')}`",
             f"- GATE_D_CORE_CHANGED: `{report.get('GATE_D_CORE_CHANGED')}`",
             f"- PUBLIC_EXPERIMENT_FREEZE_SHA256: "
             f"`{report.get('PUBLIC_EXPERIMENT_FREEZE_SHA256')}`", '',
             '## Training', '',
             f"- selected architecture: `{report.get('SELECTED_ARCHITECTURE')}`",
             f"- model seeds: `{report.get('MODEL_SEEDS')}`",
             f"- deployment candidate seed: `{report.get('DEPLOYMENT_CANDIDATE_SEED')}`",
             f"- safe batch pointnet / pointnet2 / common: "
             f"`{report.get('SAFE_BATCH_POINTNET')}` / "
             f"`{report.get('SAFE_BATCH_POINTNET2')}` / "
             f"`{report.get('COMMON_TRAIN_BATCH')}`", '',
             '## Held-out', '',
             f"- HELDOUT_ACCESSED: `{report.get('HELDOUT_ACCESSED')}`",
             f"- HELDOUT_CANDIDATES: `{report.get('HELDOUT_CANDIDATES')}`",
             f"- HELDOUT_ADAPTER_PARITY: `{report.get('HELDOUT_ADAPTER_PARITY')}`",
             f"- BAD_RECALL: `{report.get('HELDOUT_PRIMARY_BAD_RECALL')}`",
             f"- BAD_FALSE_PASS: `{report.get('HELDOUT_PRIMARY_BAD_FALSE_PASS')}`",
             f"- GOOD_FALSE_REVIEW: `{report.get('HELDOUT_PRIMARY_GOOD_FALSE_REVIEW')}`",
             f"- BALANCED_ACCURACY: `{report.get('HELDOUT_PRIMARY_BALANCED_ACCURACY')}`",
             f"- F1_REVIEW: `{report.get('HELDOUT_PRIMARY_F1_REVIEW')}`",
             f"- MODEL_STABILITY: `{report.get('MODEL_STABILITY')}`",
             f"- SAMPLING_ROBUSTNESS: `{report.get('SAMPLING_ROBUSTNESS')}` "
             f"(max drop {report.get('MAX_BAD_RECALL_DROP')} pp)", '',
             '## Per-defect (primary severities)', '',
             '```json',
             json.dumps(report.get('PER_DEFECT_METRICS'), indent=2, sort_keys=True,
                        default=str),
             '```', '',
             '## Boundaries', '',
             f"- LOCAL_8_BAD_CHALLENGE_ACCESSED: "
             f"`{report.get('LOCAL_8_BAD_CHALLENGE_ACCESSED')}`",
             f"- ONNX_STARTED: `{report.get('ONNX_STARTED')}`",
             f"- PRODUCT_INTEGRATION: `{report.get('PRODUCT_INTEGRATION')}`",
             f"- GATE_D_TESTS / GATE_C2_TESTS / EFG_TESTS: "
             f"`{report.get('GATE_D_TESTS')}` / `{report.get('GATE_C2_TESTS')}` / "
             f"`{report.get('EFG_TESTS')}`",
             f"- RUFF: `{report.get('RUFF')}`", '',
             '## Verdict basis', '']
    for item in report.get('VERDICT_BASIS') or []:
        lines.append(f'- {item}')
    lines.extend(['', '## Verdict vocabulary', '',
                  'This phase can only report one of: '
                  '`PUBLIC_PILOT_NOT_LEARNABLE`, `MODEL_UNSTABLE`, '
                  '`SAMPLING_TOO_SENSITIVE`, `READY_FOR_GATE_H_LOCAL_BLIND`, or a '
                  '`STOP_<CONCRETE_BLOCKER>` code. `PRODUCT_READY` is not issued.', ''])
    return '\n'.join(lines)


def write_report(directory: Path | str, report: dict) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / 'TREE_QC_EFG_FINAL_REPORT.json'
    md_path = directory / 'TREE_QC_EFG_FINAL_REPORT.md'
    state_module.atomic_write_json(json_path, report, with_digest=False)
    state_module.atomic_write_text(md_path, render_markdown(report))
    return {'json': str(json_path), 'md': str(md_path)}


def write_stop_report(directory: Path | str, stop: dict, state: dict) -> dict:
    """A stop produces a readable report, not a bare traceback."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        'report_version': REPORT_VERSION,
        'kind': 'EFG_AUTOPILOT_STOP_REPORT',
        'generated_at': state_module.local_timestamp(),
        'run_id': state.get('run_id'),
        'stop_code': stop.get('code'),
        'stop_detail': stop.get('detail'),
        'stages': {name: {'status': entry.get('status'),
                          'updated_at': entry.get('updated_at')}
                   for name, entry in (state.get('stages') or {}).items()},
        'heldout': state.get('heldout'),
        'freeze': state.get('freeze'),
        'calibration': state.get('calibration'),
        'next_action': next_action(stop.get('code')),
    }
    path = directory / 'EFG_STOP_REPORT.json'
    state_module.atomic_write_json(path, payload, with_digest=False)
    return {'json': str(path), 'payload': payload}


def next_action(code: str | None) -> str:
    actions = {
        'STOP_GATE_D_BINDING_MISMATCH': 'the accepted Gate D binding no longer matches '
                                        'the code; restore the frozen files or obtain a '
                                        'new Supervisor acceptance',
        'STOP_ACCEPTANCE_INVALID': 'the acceptance marker is missing or invalid; the '
                                   'Supervisor must sign Gate D again',
        'STOP_ENVIRONMENT_NOT_READY': 'repair the existing training environment manually; '
                                      'the pilot never installs or upgrades anything',
        'STOP_DISK_GUARD': 'free space on the required drive and rerun; nothing is '
                           'deleted automatically',
        'STOP_DATASET_FINGERPRINT_CHANGED': 'the frozen split or a cached example '
                                            'changed; do not resume, investigate first',
        'STOP_RUNTIME_OOM_AFTER_FREEZE': 'the frozen safe batch OOMed; the batch is not '
                                         'reduced automatically, report to the Supervisor',
        'STOP_NONFINITE_TRAINING': 'a non-finite loss or logit was observed; the last '
                                   'valid checkpoint is kept and training is not resumed',
        'STOP_CHECKPOINT_CORRUPTION': 'a checkpoint is missing or its digest changed',
        'STOP_CONFIG_MISMATCH': 'the training configuration changed after the freeze',
        'STOP_RESUME_MISMATCH': 'the recorded run does not match the current config or '
                                'dataset; a fresh training is required',
        'STOP_HELDOUT_BEFORE_FREEZE': 'held-out was requested before a valid freeze',
        'STOP_HELDOUT_ADAPTER_DRIFT': 'the held-out adapter no longer reproduces the '
                                      'frozen pipeline; nothing was unblinded',
        'STOP_HELDOUT_MANIFEST_CHANGED': 'the frozen provenance changed or the held-out '
                                         'candidate count moved',
        'STOP_FREEZE_MUTATED_AFTER_UNBLIND': 'the experiment freeze changed after '
                                             'held-out was opened',
    }
    return actions.get(code or '', 'inspect the stop detail and the state file')


def git_state(repo_root: Path | str) -> dict:
    """Implementation HEAD and cleanliness, recorded rather than assumed."""
    import subprocess
    root = str(repo_root)
    try:
        head = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'],
                                       text=True, timeout=30).strip()
        branch = subprocess.check_output(['git', '-C', root, 'rev-parse',
                                          '--abbrev-ref', 'HEAD'], text=True,
                                        timeout=30).strip()
        status = subprocess.check_output(['git', '-C', root, 'status', '--porcelain'],
                                         text=True, timeout=30).strip()
        return {'implementation_head': head, 'branch': branch,
                'worktree_clean': 'YES' if not status else 'NO',
                'dirty_entries': status.splitlines()[:40]}
    except Exception as exc:
        return {'implementation_head': NOT_RUN, 'branch': NOT_RUN,
                'worktree_clean': NOT_RUN, 'error': str(exc)}
