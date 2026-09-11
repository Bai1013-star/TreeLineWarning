"""Public experiment freeze: the scientific lock between validation and held-out.

Everything that could be tuned by looking at held-out data is decided here, before any
held-out geometry is loaded:

* the architecture, from validation metrics only, under a pre-declared rule;
* the deployment candidate seed, from validation metrics only, under a pre-declared rule;
* one frozen threshold per model seed, each chosen on that seed's own validation scores
  under the frozen ``max BAD_RECALL subject to GOOD_FALSE_REVIEW <= 0.25`` rule;
* the robustness plan — ``4096`` primary, ``2048`` robustness, and a pre-declared list of
  sampling seeds that is fixed before the first held-out score exists;
* the uncertain band, pre-declared at the repository's existing ``+-0.10``.

The freeze file is written once and carries its own SHA-256. From the moment held-out is
first touched, any change to the file is ``STOP_FREEZE_MUTATED_AFTER_UNBLIND``.

Architecture selection (validation only, deterministic):

1. eligible architectures are those with ``GOOD_FALSE_REVIEW <= 0.25``;
2. prefer the higher ``BAD_RECALL``;
3. then the higher ``BALANCED_ACCURACY``;
4. if the two are within ``0.01`` on both ``BAD_RECALL`` and ``BALANCED_ACCURACY``, the
   pre-declared non-inferiority policy prefers the simpler ``pointnet``;
5. the lower seed index is the final deterministic tie-break.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import efg_gates as gates
import efg_state as state_module

FREEZE_VERSION = 'EFG_PUBLIC_EXPERIMENT_FREEZE_V1'
FREEZE_FILE = 'PUBLIC_EXPERIMENT_FREEZE.json'
ARCHITECTURES = gates.ARCHITECTURES
NON_INFERIORITY_MARGIN = gates.NON_INFERIORITY_MARGIN
GOOD_FALSE_REVIEW_BUDGET = gates.GOOD_FALSE_REVIEW_BUDGET
POINTS_PRIMARY = gates.POINTS_PRIMARY
POINTS_ROBUSTNESS = gates.POINTS_ROBUSTNESS
ROBUSTNESS_POINTS = gates.ROBUSTNESS_POINTS
ROBUSTNESS_SAMPLING_SEEDS = gates.ROBUSTNESS_SAMPLING_SEEDS
MODEL_SEEDS = gates.MODEL_SEEDS
UNCERTAIN_BAND = gates.UNCERTAIN_BAND
SAMPLING_MAX_RECALL_DROP = gates.SAMPLING_MAX_RECALL_DROP
MODEL_STABILITY_MAX_SPREAD = gates.MODEL_STABILITY_MAX_SPREAD


class FreezeError(RuntimeError):
    """Raised with a STOP_* code when the freeze would be invalid or was mutated."""

    def __init__(self, code: str, detail):
        super().__init__(f'{code}: {detail}')
        self.code = code
        self.detail = detail


# --- validation-only decisions ---------------------------------------------

def select_architecture(evidence: dict) -> dict:
    """Apply the pre-declared rule to validation evidence for both architectures.

    ``evidence`` maps a model name to that model's validation metrics at its selected
    threshold. The result is fully determined by those numbers.
    """
    missing = [name for name in ARCHITECTURES if name not in evidence]
    if missing:
        raise FreezeError('STOP_ARCHITECTURE_EVIDENCE_INCOMPLETE', {
            'missing': missing, 'have': sorted(evidence)})
    rows = []
    for name in ARCHITECTURES:
        metrics = evidence[name]
        for field in ('BAD_RECALL', 'BALANCED_ACCURACY', 'GOOD_FALSE_REVIEW'):
            if metrics.get(field) is None:
                raise FreezeError('STOP_ARCHITECTURE_EVIDENCE_INCOMPLETE',
                                  {'model': name, 'field': field})
        rows.append({'model': name, 'BAD_RECALL': float(metrics['BAD_RECALL']),
                     'BALANCED_ACCURACY': float(metrics['BALANCED_ACCURACY']),
                     'GOOD_FALSE_REVIEW': float(metrics['GOOD_FALSE_REVIEW']),
                     'budget_satisfied': (float(metrics['GOOD_FALSE_REVIEW'])
                                          <= GOOD_FALSE_REVIEW_BUDGET)})

    eligible = [row for row in rows if row['budget_satisfied']]
    if not eligible:
        # Neither architecture met the good-false-review budget on validation. The rule
        # cannot be applied, and the pilot does not get to invent a new one.
        raise FreezeError('STOP_ARCHITECTURE_BUDGET_UNSATISFIED', {
            'message': 'no architecture satisfied GOOD_FALSE_REVIEW <= '
                       f'{GOOD_FALSE_REVIEW_BUDGET} on validation',
            'evidence': rows})
    ranked = sorted(eligible, key=lambda row: (-row['BAD_RECALL'],
                                               -row['BALANCED_ACCURACY'],
                                               ARCHITECTURES.index(row['model'])))
    winner = ranked[0]
    basis = {'rule': ['GOOD_FALSE_REVIEW <= 0.25', 'max BAD_RECALL',
                      'max BALANCED_ACCURACY',
                      'pointnet preferred within 0.01 non-inferiority',
                      'lowest model seed as final tie-break'],
             'evidence': rows, 'eligible': [row['model'] for row in eligible],
             'ranked': [row['model'] for row in ranked]}

    if len(ranked) > 1:
        runner_up = ranked[1]
        if (abs(winner['BAD_RECALL'] - runner_up['BAD_RECALL'])
                <= NON_INFERIORITY_MARGIN
                and abs(winner['BALANCED_ACCURACY'] - runner_up['BALANCED_ACCURACY'])
                <= NON_INFERIORITY_MARGIN):
            pair = {winner['model'], runner_up['model']}
            if pair == set(ARCHITECTURES):
                basis['non_inferiority_applied'] = {
                    'margin': NON_INFERIORITY_MARGIN,
                    'BAD_RECALL_delta': round(abs(winner['BAD_RECALL']
                                                  - runner_up['BAD_RECALL']), 6),
                    'BALANCED_ACCURACY_delta': round(
                        abs(winner['BALANCED_ACCURACY']
                            - runner_up['BALANCED_ACCURACY']), 6),
                    'reason': 'architectures are non-inferior on validation; the simpler '
                              'and cheaper-to-deploy pointnet is selected'}
                selected = 'pointnet'
                basis['selected'] = selected
                return {'selected_architecture': selected, 'basis': basis,
                        'non_inferiority_applied': True}
    basis['selected'] = winner['model']
    return {'selected_architecture': winner['model'], 'basis': basis,
            'non_inferiority_applied': False}


def select_deployment_candidate(per_seed_validation: dict, model_seeds=MODEL_SEEDS) -> dict:
    """Deterministic deployment seed choice from validation metrics only.

    Rule: budget satisfied -> max BAD_RECALL -> max BALANCED_ACCURACY -> lowest seed.
    """
    missing = [seed for seed in model_seeds if seed not in per_seed_validation]
    if missing:
        raise FreezeError('STOP_DEPLOYMENT_SEED_EVIDENCE_INCOMPLETE',
                          {'missing_seeds': missing})
    rows = []
    for seed in model_seeds:
        metrics = per_seed_validation[seed]
        rows.append({'seed': int(seed), 'BAD_RECALL': float(metrics['BAD_RECALL']),
                     'BALANCED_ACCURACY': float(metrics['BALANCED_ACCURACY']),
                     'GOOD_FALSE_REVIEW': float(metrics['GOOD_FALSE_REVIEW']),
                     'budget_satisfied': (float(metrics['GOOD_FALSE_REVIEW'])
                                          <= GOOD_FALSE_REVIEW_BUDGET)})
    eligible = [row for row in rows if row['budget_satisfied']]
    if not eligible:
        raise FreezeError('STOP_DEPLOYMENT_SEED_BUDGET_UNSATISFIED', {
            'message': 'no seed satisfied the good-false-review budget on validation',
            'evidence': rows})
    ranked = sorted(eligible, key=lambda row: (-row['BAD_RECALL'],
                                               -row['BALANCED_ACCURACY'], row['seed']))
    return {'deployment_candidate_seed': int(ranked[0]['seed']),
            'rule': ['GOOD_FALSE_REVIEW <= 0.25', 'max BAD_RECALL',
                     'max BALANCED_ACCURACY', 'lowest seed'],
            'evidence': rows, 'eligible': [row['seed'] for row in eligible],
            'ranked': [row['seed'] for row in ranked]}


def robustness_plan() -> dict:
    """The sampling robustness configuration, fixed before held-out is opened."""
    return gates.robustness_plan()


def public_pilot_gates() -> dict:
    return gates.public_pilot_gates()


# --- freeze construction ----------------------------------------------------

def build_freeze(gate_d_binding: dict, selected: dict, comparison_evidence: dict,
                 training_config: dict, model_seeds, runs: dict,
                 validation_dataset: dict, per_seed_validation: dict,
                 deployment: dict, freeze_timestamp: str,
                 robustness: dict | None = None,
                 orchestrator_version: str | None = None) -> dict:
    """Assemble PUBLIC_EXPERIMENT_FREEZE.json. Missing evidence raises, never defaults."""
    for seed in model_seeds:
        entry = runs.get(seed) or {}
        for field in ('checkpoint_path', 'checkpoint_sha256', 'config_sha256',
                      'dataset_digest', 'threshold', 'threshold_path',
                      'threshold_sha256'):
            if not entry.get(field):
                raise FreezeError('STOP_FREEZE_EVIDENCE_INCOMPLETE',
                                  {'seed': seed, 'field': field})
    payload = {
        'freeze_version': FREEZE_VERSION,
        'freeze_timestamp': freeze_timestamp,
        'orchestrator_version': orchestrator_version,
        'gate_d_binding': {
            'generator_version': gate_d_binding.get('generator_version'),
            'generator_digest': gate_d_binding.get('generator_digest'),
            'protocol_digest': gate_d_binding.get('protocol_digest'),
            'gate_d_accepted': True,
        },
        'selected_architecture': selected['selected_architecture'],
        'architecture_selection_basis': selected['basis'],
        'non_inferiority_applied': selected.get('non_inferiority_applied', False),
        'model_comparison_evidence': comparison_evidence,
        'training_config': training_config,
        'model_seeds': [int(seed) for seed in model_seeds],
        'runs': {str(seed): dict(runs[seed]) for seed in model_seeds},
        'validation_dataset': validation_dataset,
        'validation_metrics_per_seed': {str(seed): per_seed_validation[seed]
                                        for seed in model_seeds},
        'frozen_thresholds': {str(seed): runs[seed]['threshold'] for seed in model_seeds},
        'threshold_rule': 'max BAD_RECALL subject to GOOD_FALSE_REVIEW <= '
                          f'{GOOD_FALSE_REVIEW_BUDGET}, chosen on validation only',
        'deployment_candidate_seed': int(deployment['deployment_candidate_seed']),
        'deployment_seed_basis': deployment,
        'points_primary': POINTS_PRIMARY,
        'points_robustness': POINTS_ROBUSTNESS,
        'robustness_plan': robustness or robustness_plan(),
        'uncertain_policy': public_pilot_gates()['uncertain_policy'],
        'uncertain_band': UNCERTAIN_BAND,
        'public_pilot_metric_gates': public_pilot_gates(),
        'held_out_evaluated': False,
        'onnx_started': False,
        'product_integration': False,
        'local_8_bad_challenge_accessed': False,
    }
    return payload


def freeze_digest(payload: dict) -> str:
    """Digest of the freeze, excluding its own recorded digest field."""
    body = {k: v for k, v in payload.items() if k != 'freeze_sha256'}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(',', ':'),
                   default=str).encode('utf-8')).hexdigest()


def write_freeze(path: Path | str, payload: dict) -> tuple[Path, str]:
    """Write the freeze once. An existing freeze is never overwritten in place."""
    path = Path(path)
    if path.exists():
        existing = state_module.read_json(path, verify=False)
        if existing and 'freeze_sha256' in existing:
            return path, existing['freeze_sha256']
        raise FreezeError('STOP_FREEZE_ALREADY_EXISTS', {
            'message': f'{path} exists without a digest; refusing to overwrite',
            'path': str(path)})
    body = dict(payload)
    body['freeze_sha256'] = freeze_digest(body)
    state_module.atomic_write_json(path, body, with_digest=False)
    return path, body['freeze_sha256']


def load_freeze(path: Path | str) -> dict:
    payload = state_module.read_json(path, verify=False)
    if not payload:
        raise FreezeError('STOP_HELDOUT_BEFORE_FREEZE', {
            'message': f'no freeze at {path}', 'path': str(path)})
    return payload


def freeze_state(path: Path | str) -> dict:
    """Whether the freeze exists, is self-consistent, and matches its recorded digest."""
    path = Path(path)
    state = {'path': str(path), 'exists': path.exists(), 'valid': False,
             'sha256': None, 'reason': None}
    if not path.exists():
        state['reason'] = 'freeze file absent'
        return state
    try:
        payload = state_module.read_json(path, verify=False)
    except state_module.StateError as exc:
        state['reason'] = f'unreadable freeze: {exc}'
        return state
    recorded = payload.get('freeze_sha256')
    if not recorded:
        state['reason'] = 'freeze has no recorded digest'
        return state
    actual = freeze_digest(payload)
    if recorded != actual:
        state['reason'] = (f'freeze digest mismatch: recorded {recorded[:16]}... != '
                           f'computed {actual[:16]}...')
        return state
    for field in ('gate_d_binding', 'selected_architecture', 'model_seeds', 'runs',
                  'frozen_thresholds', 'deployment_candidate_seed', 'points_primary',
                  'points_robustness', 'robustness_plan', 'public_pilot_metric_gates'):
        if field not in payload:
            state['reason'] = f'freeze is missing required field {field!r}'
            return state
    if payload.get('held_out_evaluated') is not False:
        state['reason'] = 'freeze was written with held_out_evaluated set; refusing'
        return state
    state.update(valid=True, sha256=recorded, payload=payload)
    return state


def require_valid_freeze(path: Path | str) -> dict:
    state = freeze_state(path)
    if not state['valid']:
        raise FreezeError('STOP_HELDOUT_BEFORE_FREEZE', {
            'message': 'held-out evaluation requires a valid, unmodified '
                       'PUBLIC_EXPERIMENT_FREEZE.json',
            'freeze_state': {k: v for k, v in state.items() if k != 'payload'}})
    return state


def check_freeze_immutable(path: Path | str, unblind_audit: dict | None) -> dict:
    """STOP_FREEZE_MUTATED_AFTER_UNBLIND when the freeze changed after unblinding."""
    state = freeze_state(path)
    audit = unblind_audit or {}
    if not audit.get('accessed'):
        return {'checked': False, 'reason': 'held-out has not been accessed',
                'freeze_sha256': state.get('sha256')}
    recorded = audit.get('freeze_sha256_at_unblind')
    if not state['valid']:
        raise FreezeError('STOP_FREEZE_MUTATED_AFTER_UNBLIND', {
            'message': 'the freeze is no longer valid after held-out was accessed',
            'reason': state['reason'], 'recorded_at_unblind': recorded})
    if recorded != state['sha256']:
        raise FreezeError('STOP_FREEZE_MUTATED_AFTER_UNBLIND', {
            'message': 'PUBLIC_EXPERIMENT_FREEZE.json changed after held-out was opened',
            'recorded_at_unblind': recorded, 'current': state['sha256']})
    return {'checked': True, 'freeze_sha256': state['sha256'], 'unchanged': True}


def require_all_frozen(runs: dict, model_seeds) -> dict:
    """Every seed must have a frozen threshold and a live, digest-verified checkpoint."""
    problems = []
    for seed in model_seeds:
        entry = runs.get(int(seed)) or runs.get(str(seed)) or {}
        if not entry:
            problems.append({'seed': seed, 'issue': 'no run record'})
            continue
        if not entry.get('threshold_frozen'):
            problems.append({'seed': seed, 'issue': 'threshold not frozen'})
        checkpoint = entry.get('checkpoint_path')
        if not checkpoint or not Path(checkpoint).exists():
            problems.append({'seed': seed, 'issue': 'checkpoint missing',
                             'path': checkpoint})
            continue
        actual = state_module.digest_file(checkpoint)
        if actual != entry.get('checkpoint_sha256'):
            problems.append({'seed': seed, 'issue': 'checkpoint digest mismatch',
                             'recorded': entry.get('checkpoint_sha256'),
                             'actual': actual})
    if problems:
        raise FreezeError('STOP_CHECKPOINT_CORRUPTION', {
            'message': 'not every selected model seed has a valid frozen artifact',
            'problems': problems})
    return {'seeds': [int(seed) for seed in model_seeds], 'all_frozen': True}


def checkpoint_digest(path: Path | str) -> str:
    return state_module.digest_file(path)
