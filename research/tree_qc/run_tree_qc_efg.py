"""Gate D unlocked + E/F/G merged one-click unattended Public Pilot.

This is the executable total control for the Tree QC Public Pilot. It is a **state
machine**, not a script of steps: every stage is a named, resumable unit whose completion
is bound to the digests that produced it, and every state transition is written atomically
before the next stage starts. Double-clicking the launcher again after a power cut, a
restart or a closed terminal resumes from the last verified stage.

The frozen sequence (§14 of the round specification):

    environment -> Gate D acceptance -> resource calibration -> dataset fingerprint ->
    adapter parity -> PointNet seed101 -> PointNet++ seed101 -> learnability ->
    architecture selection -> three seeds -> threshold freeze -> deployment candidate ->
    sampling probe -> PUBLIC_EXPERIMENT_FREEZE -> held-out blind evaluation ->
    multi-seed aggregation -> sampling robustness -> final report

Fail-closed behaviour: any scientific, safety, data-integrity or resource problem raises a
``STOP_*`` code, the state file records it, a stop report is written and the process exits
without changing the experiment. The pilot never installs anything, never deletes
anything, never enlarges a model, never adds epochs, never re-picks a threshold and never
touches the local Gate H challenge.

Usage (the launcher does this for the user):

    python run_tree_qc_efg.py --auto            # full unattended run, resumable
    python run_tree_qc_efg.py --dry-run         # resolved plan, no GPU work
    python run_tree_qc_efg.py --status          # read-only progress report
    python run_tree_qc_efg.py --resume-test     # prove resume/staleness behaviour
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import efg_dataset as dataset_source  # noqa: E402
import efg_evaluate as evaluate_module  # noqa: E402
import efg_freeze as freeze_module  # noqa: E402
import efg_gates as gates  # noqa: E402
import efg_heldout as heldout_module  # noqa: E402
import efg_leakage as leakage_module  # noqa: E402
import efg_reports as reports_module  # noqa: E402
import efg_resources as resources_module  # noqa: E402
import efg_state as state_module  # noqa: E402
import efg_trainer as trainer_module  # noqa: E402
import gate_d_dataset as dataset_module  # noqa: E402
import gate_d_diagnostic as diagnostic_module  # noqa: E402
import gate_d_lock as lock  # noqa: E402
import gate_d_train as engine  # noqa: E402
from gate_d_provenance import load_gate_d_provenance  # noqa: E402

ORCHESTRATOR_VERSION = 'TREE_QC_EFG_ORCHESTRATOR_V2'
REPORTS = HERE / 'reports'
DEFAULT_RUN_ROOT = Path('D:/TreeQCTraining/public_pilot_v1')
BASE_ACCEPTED_HEAD = '1c4ed5b96e4ef1053bf9d0e47d597911ffbca727'
EXPECTED_GENERATOR_DIGEST = (
    'a0550c17ff75cdf7d380fc677840c883f2262f2148a40edbbee0df35410b6b9d')
EXPECTED_PROTOCOL_DIGEST = (
    '36e4cfca043190d4d12c2ab9ee635d92bb479cb9b56d482ae1121fd797939729')
EXPECTED_GENERATOR_VERSION = 'GATE_D_SYNTHETIC_GENERATOR_V2'
POINTS_PRIMARY = gates.POINTS_PRIMARY
PARITY_EARLY_EXAMPLES = 63
PARITY_COVERAGE_CANDIDATES = 5
CALIBRATION_PROBE_CANDIDATES = 8

# --- stage names -----------------------------------------------------------
S_ENVIRONMENT = 'S01_ENVIRONMENT'
S_GATE_D = 'S02_GATE_D_ACCEPTANCE'
S_CALIBRATION = 'S03_RESOURCE_CALIBRATION'
S_DATASET = 'S04_DATASET_FINGERPRINT'
S_PREWARM = 'S04B_TRAINING_DATA_PREWARM'
S_PARITY = 'S05_ADAPTER_PARITY'
S_COMPARE_POINTNET = 'S06_POINTNET_SEED101'
S_COMPARE_POINTNET2 = 'S07_POINTNET2_SEED101'
S_LEARNABILITY = 'S08_LEARNABILITY'
S_ARCHITECTURE = 'S09_ARCHITECTURE_SELECTION'
S_SEEDS = 'S10_MODEL_SEEDS'
S_THRESHOLDS = 'S11_THRESHOLD_FREEZE'
S_DEPLOYMENT = 'S12_DEPLOYMENT_CANDIDATE'
S_PROBE = 'S13_SAMPLING_PROBE'
S_FREEZE = 'S14_PUBLIC_EXPERIMENT_FREEZE'
S_HELDOUT = 'S15_HELDOUT_BLIND_EVALUATION'
S_AGGREGATE = 'S16_MULTI_SEED_AGGREGATION'
S_ROBUSTNESS = 'S17_SAMPLING_ROBUSTNESS'
S_REPORT = 'S18_FINAL_REPORT'

STAGE_ORDER = (
    S_ENVIRONMENT, S_GATE_D, S_CALIBRATION, S_DATASET, S_PREWARM, S_PARITY,
    S_COMPARE_POINTNET, S_COMPARE_POINTNET2, S_LEARNABILITY, S_ARCHITECTURE, S_SEEDS,
    S_THRESHOLDS, S_DEPLOYMENT, S_PROBE, S_FREEZE, S_HELDOUT, S_AGGREGATE, S_ROBUSTNESS,
    S_REPORT,
)
STAGE_TITLES = {
    S_ENVIRONMENT: 'Environment',
    S_GATE_D: 'Gate D acceptance',
    S_CALIBRATION: 'Resource calibration',
    S_DATASET: 'Dataset fingerprint',
    S_PREWARM: 'Training data prewarm',
    S_PARITY: 'Held-out adapter parity',
    S_COMPARE_POINTNET: 'PointNet seed101',
    S_COMPARE_POINTNET2: 'PointNet++ seed101',
    S_LEARNABILITY: 'Initial learnability',
    S_ARCHITECTURE: 'Architecture selection',
    S_SEEDS: 'Three model seeds',
    S_THRESHOLDS: 'Threshold freeze',
    S_DEPLOYMENT: 'Deployment candidate seed',
    S_PROBE: 'Sampling probe',
    S_FREEZE: 'Public experiment freeze',
    S_HELDOUT: 'Held-out blind evaluation',
    S_AGGREGATE: 'Multi-seed aggregation',
    S_ROBUSTNESS: 'Sampling robustness',
    S_REPORT: 'Final report',
}
# Stages that must not run once held-out has been opened.
PRE_UNBLIND_STAGES = (S_ENVIRONMENT, S_GATE_D, S_CALIBRATION, S_DATASET, S_PREWARM,
                      S_PARITY, S_COMPARE_POINTNET, S_COMPARE_POINTNET2,
                      S_LEARNABILITY, S_ARCHITECTURE, S_SEEDS, S_THRESHOLDS,
                      S_DEPLOYMENT, S_PROBE, S_FREEZE)

FROZEN_GATE_D_FILES = ('gate_d_provenance.py', 'gate_d_scene.py', 'gate_d_geometry.py',
                       'gate_d_generator.py', 'gate_d_sampler.py', 'gate_d_dataset.py',
                       'gate_d_metrics.py', 'gate_d_train.py', 'gate_d_fingerprint.py',
                       'gate_d_diagnostic.py')


class PilotStop(RuntimeError):
    """A recorded fail-closed stop."""

    def __init__(self, code: str, detail):
        super().__init__(f'{code}: {json.dumps(detail, default=str)[:800]}')
        self.code = code
        self.detail = detail


# --- Gate D binding --------------------------------------------------------

def frozen_file_digests() -> dict:
    """Per-file SHA-256 of every file that participates in the generator digest."""
    return {name: state_module.digest_file(HERE / name)
            for name in FROZEN_GATE_D_FILES}


def gate_d_binding_status() -> dict:
    """Re-verify the accepted Gate D binding against the current runtime."""
    runtime_generator = diagnostic_module.generator_digest()
    runtime_protocol = lock.current_binding()['protocol_digest']
    state = lock.acceptance_state()
    matches = {
        'generator_digest': runtime_generator == EXPECTED_GENERATOR_DIGEST,
        'protocol_digest': runtime_protocol == EXPECTED_PROTOCOL_DIGEST,
        'generator_version': (lock.current_binding()['generator_version']
                              == EXPECTED_GENERATOR_VERSION),
    }
    return {
        'expected_generator_version': EXPECTED_GENERATOR_VERSION,
        'expected_generator_digest': EXPECTED_GENERATOR_DIGEST,
        'expected_protocol_digest': EXPECTED_PROTOCOL_DIGEST,
        'runtime_generator_digest': runtime_generator,
        'runtime_protocol_digest': runtime_protocol,
        'matches': matches,
        'binding_ok': all(matches.values()),
        'acceptance_marker': str(state['marker']),
        'acceptance_exists': state['exists'],
        'training_allowed': state['training_allowed'],
        'acceptance_reason': state['reason'],
        'frozen_file_digests': frozen_file_digests(),
    }


def require_gate_d_binding() -> dict:
    status = gate_d_binding_status()
    if not status['binding_ok']:
        raise PilotStop('STOP_GATE_D_BINDING_MISMATCH', {
            'message': 'the runtime Gate D generator or protocol digest does not match '
                       'the accepted binding',
            'expected_generator_digest': status['expected_generator_digest'],
            'runtime_generator_digest': status['runtime_generator_digest'],
            'expected_protocol_digest': status['expected_protocol_digest'],
            'runtime_protocol_digest': status['runtime_protocol_digest'],
            'frozen_file_digests': status['frozen_file_digests'],
            'note': 'the acceptance is not updated to a new digest automatically'})
    if not status['training_allowed']:
        raise PilotStop('STOP_ACCEPTANCE_INVALID', {
            'message': 'the Gate D acceptance marker is absent or not valid for the '
                       'current code',
            'marker': status['acceptance_marker'], 'reason': status['acceptance_reason']})
    return status


class LeakageAuditError(PilotStop):
    pass


# --- helpers ---------------------------------------------------------------

def build_train_config(model: str, seed: int, calibration: dict, layout,
                       epochs: int = gates.EPOCHS, points: int = POINTS_PRIMARY,
                       repeats: int = 1, seed_base: int = 0,
                       batch_size: int | None = None,
                       max_train_candidates: int | None = None,
                       max_validation_candidates: int | None = None
                       ) -> engine.TrainConfig:
    """The frozen training configuration. Every value comes from the declared plan."""
    batch = int(batch_size if batch_size is not None
                else calibration['safe_common_batch'])
    return engine.TrainConfig(
        model=model, partition='train', validation_partition='validation',
        points=int(points), epochs=int(epochs), batch_size=batch,
        learning_rate=gates.LEARNING_RATE, weight_decay=gates.WEIGHT_DECAY,
        seed=int(seed), amp=gates.AMP_WHEN_CUDA, amp_dtype=gates.AMP_DTYPE,
        workers=gates.WORKERS, grad_clip=gates.GRAD_CLIP,
        early_stop_patience=gates.EARLY_STOP_PATIENCE,
        early_stop_min_delta=gates.EARLY_STOP_MIN_DELTA,
        good_false_review_budget=gates.GOOD_FALSE_REVIEW_BUDGET,
        max_train_candidates=max_train_candidates,
        max_validation_candidates=max_validation_candidates,
        repeats=int(repeats), seed_base=int(seed_base),
        cache_root=str(dataset_module.DEFAULT_CACHE_ROOT),
        output_root=str(Path(layout.runs_root)))


def run_key(model: str, seed: int, points: int = POINTS_PRIMARY,
            repeats: int = 1, seed_base: int = 0) -> str:
    return f'{model}|seed{seed}|points{points}|repeats{repeats}|base{seed_base}'


def full_validation_metrics(run: dict, source, device_preference: str = 'cuda') -> dict:
    """Recompute a run's validation metrics at its own frozen threshold.

    This fills ``BAD_FALSE_PASS``, which the training loop's epoch row does not compute,
    and it re-derives every other validation number from the checkpoint rather than
    trusting a stored one. It reads model-selection data only.
    """
    scored = trainer_module.score_run_checkpoint(run, source, device_preference)
    threshold = float(run['threshold'])
    metrics = evaluate_module.binary_metrics(scored['labels'], scored['scores'],
                                             threshold)
    return {'metrics': metrics, 'threshold': threshold,
            'examples': int(scored['examples']),
            'BAD_RECALL': metrics['BAD_RECALL'],
            'BAD_FALSE_PASS': metrics['BAD_FALSE_PASS'],
            'GOOD_FALSE_REVIEW': metrics['GOOD_FALSE_REVIEW'],
            'BALANCED_ACCURACY': metrics['BALANCED_ACCURACY'],
            'F1_REVIEW': metrics['F1_REVIEW'],
            'thresholds': {'frozen': threshold, 'operating': threshold}}


def console(line: str = '') -> None:
    print(line, flush=True)


def stage_banner(index: int, total: int, name: str, status: str, detail: str = '') -> None:
    title = STAGE_TITLES.get(name, name)
    padding = ' ' * max(1, 30 - len(title))
    console(f'[{index:02d}/{total}] {title}{padding}{status}'
            + (f'  {detail}' if detail else ''))


class Pilot:
    """The resumable Public Pilot state machine."""

    def __init__(self, run_root: Path | str = DEFAULT_RUN_ROOT, epochs: int = gates.EPOCHS,
                 points: int = POINTS_PRIMARY, repeats: int = 1, seed_base: int = 0,
                 comparison_candidates: int | None = None, verbose: bool = True,
                 reuse_calibration: bool = True, use_cache: bool = True,
                 max_heldout_candidates: int | None = None, smoke_test: bool = False
                 ):
        self.layout = state_module.RunLayout(run_root).ensure()
        self.epochs = int(epochs)
        self.points = int(points)
        self.repeats = int(repeats)
        self.seed_base = int(seed_base)
        self.comparison_candidates = comparison_candidates
        self.max_heldout_candidates = max_heldout_candidates
        self.verbose = verbose
        self.reuse_calibration = reuse_calibration
        self.use_cache = use_cache
        self.smoke_test = bool(smoke_test)
        self.state = state_module.AutopilotState.load(self.layout.state_path)
        self.fresh = self.state.ensure_run_identity(self.state.payload.get('run_id')
                                                    or _new_run_id(),
                                                    ORCHESTRATOR_VERSION)
        self.state.payload['smoke_test'] = self.smoke_test
        self.state.payload['plan'] = {
            'epochs': self.epochs, 'points': self.points, 'repeats': self.repeats,
            'seed_base': self.seed_base,
            'max_comparison_candidates': self.comparison_candidates,
            'max_heldout_candidates': self.max_heldout_candidates,
            'formal_experiment': not self.smoke_test,
        }
        self.provenance = None
        self.calibration: dict = {}
        self.validation_source = None
        self.train_sources: dict = {}
        self.comparison: dict = {}
        self.validation_evidence: dict = {}
        self.runs: dict = {}
        self.heldout_records: dict = {}
        self.stage_index = {name: index for index, name in enumerate(STAGE_ORDER, 1)}

    # --- infrastructure ---------------------------------------------------
    def log(self, message: str) -> None:
        """Every status line is printed, appended to the run log and mirrored on disk.

        The log is the only durable trace of a run whose terminal may be closed at any
        moment, so it never depends on a stage boundary to be written.
        """
        if self.verbose:
            console(message)
        self.log_line(message)

    def log_line(self, message: str) -> None:
        path = self.layout.path('logs', 'autopilot.log')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(f'{state_module.local_timestamp()} {message}\n')

    def progress(self, label: str, payload: dict) -> None:
        """Long-running work reports progress to the state, the status file and the log."""
        self.state.payload.setdefault('progress', {})[label] = {
            **payload, 'updated_at': state_module.local_timestamp()}
        self.state.save()
        self.write_current_status()
        self.log_line(f'{label} {json.dumps(payload)}')

    def record_stage(self, name: str, status: str, evidence: dict | None = None,
                     **extra) -> None:
        self.state.set_stage(name, status, evidence, **extra)
        self.state.save()
        self.write_current_status()

    def write_current_status(self) -> None:
        payload = {
            'run_id': self.state.payload.get('run_id'),
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'updated_at': state_module.local_timestamp(),
            'resume_count': self.state.payload['resume'].get('count'),
            'stop': self.state.payload.get('stop'),
            'heldout': {'accessed': self.state.payload['heldout'].get('accessed')},
            'freeze': self.state.payload.get('freeze'),
            'final': self.state.payload.get('final'),
            'stages': [{'name': name, 'title': STAGE_TITLES[name],
                        'status': self.state.stage(name).get('status', 'PENDING')}
                       for name in STAGE_ORDER],
            'last_log': str(self.layout.path('logs', 'autopilot.log')),
            'progress': self.state.payload.get('progress') or {},
            'run_root': str(self.layout.root),
        }
        state_module.atomic_write_json(self.layout.current_status_path, payload,
                                       with_digest=False)

    def recover_interrupted_stages(self) -> dict:
        """A stage left RUNNING by a dead process is resumable, not complete.

        A power cut, a closed terminal or a killed process leaves the stage it was executing
        recorded as ``RUNNING``. Such a stage is set back to ``PENDING`` with the interrupt
        recorded, so the machine re-enters it instead of silently treating it as finished.
        Anything the interrupted stage had already durably produced — cached examples, a
        prewarm exclusion record, a checkpoint — is left in place, so re-entry continues
        rather than restarts.

        A stop recorded by an earlier attempt is cleared here too, because the stop it
        describes is no longer the state of the run; leaving it in place would make
        ``CURRENT_STATUS.json`` and the launcher's summary report a failure that this
        attempt is actively moving past. The stop is kept as history under ``stop_history``.
        """
        recovered = []
        for name in STAGE_ORDER:
            entry = self.state.payload['stages'].get(name)
            if entry and entry.get('status') == state_module.RUNNING:
                entry['status'] = state_module.PENDING
                entry['interrupted'] = True
                entry['interrupt_count'] = int(entry.get('interrupt_count', 0)) + 1
                entry['last_interrupt_at'] = state_module.local_timestamp()
                recovered.append(name)
        if recovered:
            self.state.save()
            self.log(f"    recovered interrupted stage(s): {', '.join(recovered)}")
        previous = self.state.payload.get('stop') or {}
        if previous.get('code'):
            history = list(self.state.payload.get('stop_history') or [])
            history.append({**previous, 'superseded_at': state_module.local_timestamp()})
            self.state.payload['stop_history'] = history[-20:]
            self.state.clear_stop()
            self.state.save()
            self.log(f"    previous stop {previous['code']} is recorded as history; "
                     'this attempt continues past it')
        stale = {key: value for key, value in (self.state.payload.get('progress') or {}
                                               ).items()
                 if key.split('.')[0] in recovered}
        for key in stale:
            self.state.payload['progress'].pop(key, None)
        if recovered:
            self.state.save()
        return {'recovered': recovered, 'cleared_stop': previous.get('code')}

    def probe(self, name: str) -> tuple[bool, dict]:
        """Whether a stage may be skipped, using only digest-bound evidence."""
        return state_module.stage_is_complete(self.state.stage(name), {})

    def done(self, name: str, **evidence) -> bool:
        stage = self.state.stage(name)
        required = dict(evidence)
        required.setdefault('orchestrator_version', ORCHESTRATOR_VERSION)
        ok, reason = state_module.stage_is_complete(stage, required)
        if ok and self.verbose:
            self.log(f'    {name}: COMPLETE (verified) — skipping')
        elif self.verbose and stage.get('status') == 'COMPLETE':
            self.log(f'    {name}: recorded COMPLETE but STALE ({reason}); rerunning')
        return ok

    # --- guards -----------------------------------------------------------
    def guard_pre_unblind(self, name: str) -> None:
        """No pre-unblind stage may run once held-out has been opened."""
        audit = self.state.payload['heldout']
        if audit.get('accessed') and name in PRE_UNBLIND_STAGES:
            raise PilotStop('STOP_FREEZE_MUTATED_AFTER_UNBLIND', {
                'message': f'{name} would re-run a pre-unblind stage after held-out was '
                           'accessed; the experiment is frozen',
                'first_access_timestamp': audit.get('first_access_timestamp')})
        freeze = self.state.payload.get('freeze') or {}
        if audit.get('accessed') and freeze.get('sha256'):
            freeze_module.check_freeze_immutable(self.layout.freeze_path, audit)

    def guard_no_tuning_after_unblind(self) -> None:
        audit = self.state.payload['heldout']
        if not audit.get('accessed'):
            return
        freeze_module.check_freeze_immutable(self.layout.freeze_path, audit)

    # --- stages -----------------------------------------------------------
    def stage_environment(self) -> None:
        name = S_ENVIRONMENT
        if self.done(name):
            fingerprint = state_module.read_json(self.layout.environment_fingerprint_path,
                                                 verify=False)
            if not fingerprint:
                raise PilotStop('STOP_RESUME_MISMATCH', {
                    'message': 'the environment stage is recorded complete but its '
                               'fingerprint file is missing',
                    'expected': str(self.layout.environment_fingerprint_path)})
            resources_module.require_disk_guard(fingerprint['disks'], ('D', 'E'))
            resources_module.require_environment(fingerprint)
            self.state.payload['environment'] = fingerprint
            self.state.save()
            return
        self.guard_pre_unblind(name)
        self.record_stage(name, 'RUNNING')
        self.log('    capturing environment fingerprint ...')
        fingerprint = resources_module.fingerprint(self.layout)
        resources_module.require_environment(fingerprint)
        resources_module.require_disk_guard(fingerprint['disks'], ('D', 'E'))
        state_module.atomic_write_json(self.layout.environment_fingerprint_path,
                                       fingerprint, with_digest=False)
        self.state.payload['environment'] = fingerprint
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'environment_digest': fingerprint['environment_digest'],
            'torch': fingerprint['gpu'].get('torch'),
            'device': fingerprint['gpu'].get('device_name'),
        }, detail={'device': fingerprint['gpu'].get('device_name'),
                   'vram_gib': round((fingerprint['gpu'].get('vram_total_bytes') or 0)
                                     / 1024**3, 2),
                   'ram_gib': round((fingerprint.get('ram_total_bytes') or 0)
                                    / 1024**3, 2),
                   'd_free_gib': fingerprint['disks'].get('D', {}).get('free_gib')})

    def stage_gate_d(self) -> None:
        name = S_GATE_D
        status = gate_d_binding_status()
        evidence = {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'generator_digest': status['runtime_generator_digest'],
            'protocol_digest': status['runtime_protocol_digest'],
            'generator_version': status['expected_generator_version'],
            'frozen_file_digests': status['frozen_file_digests'],
        }
        if not status['binding_ok']:
            raise PilotStop('STOP_GATE_D_BINDING_MISMATCH', {
                'message': 'runtime Gate D binding differs from the accepted values',
                'expected_generator_digest': status['expected_generator_digest'],
                'runtime_generator_digest': status['runtime_generator_digest'],
                'expected_protocol_digest': status['expected_protocol_digest'],
                'runtime_protocol_digest': status['runtime_protocol_digest']})
        if self.done(name, generator_digest=evidence['generator_digest'],
                     protocol_digest=evidence['protocol_digest']):
            self.state.payload['accepted_gate_d_binding'] = {
                'generator_version': EXPECTED_GENERATOR_VERSION,
                'generator_digest': status['runtime_generator_digest'],
                'protocol_digest': status['runtime_protocol_digest'],
                'training_allowed': status['training_allowed'],
                'acceptance_marker': status['acceptance_marker'],
            }
            self.state.save()
            return
        self.guard_pre_unblind(name)
        self.record_stage(name, 'RUNNING')
        verified = require_gate_d_binding()
        # Prove the trainers no longer report GATE_D_NOT_ACCEPTED by exercising their own
        # dry-run entry points through the lock.
        engine.lock.require_training_allowed()
        try:
            self.provenance = load_gate_d_provenance()
        except Exception as exc:
            raise PilotStop('STOP_DATASET_FINGERPRINT_CHANGED', {
                'message': 'the frozen Gate C2 provenance no longer loads',
                'error': str(exc)[:400]}) from exc
        self.record_stage(name, 'COMPLETE', evidence, detail={
            'acceptance_reason': verified['acceptance_reason'],
            'training_allowed': verified['training_allowed'],
            'generator_digest': verified['runtime_generator_digest'][:16],
            'protocol_digest': verified['runtime_protocol_digest'][:16],
            'held_out_candidates_metadata_only':
                len(self.provenance.held_out_eligible),
        })
        self.state.payload['accepted_gate_d_binding'] = {
            'generator_version': EXPECTED_GENERATOR_VERSION,
            'generator_digest': verified['runtime_generator_digest'],
            'protocol_digest': verified['runtime_protocol_digest'],
            'training_allowed': verified['training_allowed'],
            'acceptance_marker': verified['acceptance_marker'],
        }
        self.state.save()

    def stage_calibration(self) -> None:
        name = S_CALIBRATION
        environment = self.state.payload.get('environment') or {}
        current_key = {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'device': (environment.get('gpu') or {}).get('device_name'),
            'points': self.points,
            'models': list(gates.ARCHITECTURES),
        }
        recorded = self.state.payload.get('calibration') or {}
        if (self.reuse_calibration and recorded.get('safe_common_batch')
                and recorded.get('reuse_key') == current_key
                and self.done(name, calibration_digest=recorded.get(
                    'calibration_digest', ''))):
            self.calibration = recorded
            self.log(f"    reusing measured safe batch "
                     f"{recorded['safe_common_batch']} (pointnet "
                     f"{recorded['models']['pointnet']['safe_batch']}, pointnet2 "
                     f"{recorded['models']['pointnet2_ssg']['safe_batch']})")
            return
        self.guard_pre_unblind(name)
        self.record_stage(name, 'RUNNING')
        self.log('    probing batch sizes 16, 8, 4, 2, 1 with real optimizer steps ...')
        calibration = resources_module.calibrate_batch(gates.ARCHITECTURES)
        calibration['reuse_key'] = current_key
        self.calibration = calibration
        self.state.payload['calibration'] = calibration
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'calibration_digest': calibration['calibration_digest'],
            'safe_common_batch': calibration['safe_common_batch'],
            'points': self.points,
        }, detail={
            'pointnet_safe': calibration['models']['pointnet']['safe_batch'],
            'pointnet2_safe': calibration['models']['pointnet2_ssg']['safe_batch'],
            'common': calibration['safe_common_batch'],
        })
        state_module.atomic_write_json(
            self.layout.path('state', 'BATCH_CALIBRATION.json'), calibration,
            with_digest=False)

    def _load_provenance(self):
        if self.provenance is None:
            self.provenance = load_gate_d_provenance()
        return self.provenance

    def stage_dataset(self) -> None:
        name = S_DATASET
        provenance = self._load_provenance()
        generator_digest = diagnostic_module.generator_digest()
        import efg_dataset as dataset_source
        import efg_specs as specs_module

        fingerprint = {
            'provenance_digest': provenance.provenance_digest,
            'identity_manifest_digest': provenance.identity_manifest_digest,
            'protocol_digest': provenance.protocol_digest,
            'generator_digest': generator_digest,
            'partition': 'train+validation',
            'train_candidates': len(provenance.candidates('train')),
            'validation_candidates': len(provenance.candidates('validation')),
            'held_out_candidates_metadata_only': len(provenance.held_out_eligible),
        }
        if self.done(name, provenance_digest=fingerprint['provenance_digest'],
                     identity_manifest_digest=fingerprint['identity_manifest_digest'],
                     generator_digest=generator_digest):
            self.state.payload['dataset'] = self.state.stage(name).get('evidence', {})
            self.state.save()
        else:
            self.guard_pre_unblind(name)
            self.record_stage(name, 'RUNNING')
            held_view = heldout_module.held_out_view()
            fingerprint['held_out_integrity'] = heldout_module.assert_heldout_integrity(
                provenance, held_view,
                expected_candidates=len(provenance.held_out_eligible))
            for partition in ('train', 'validation'):
                candidate_ids = list(provenance.candidates(partition))
                balance = specs_module.class_balance(provenance, candidate_ids,
                                                     partition, points=self.points,
                                                     seed_base=self.seed_base,
                                                     repeats=self.repeats)
                fingerprint[f'{partition}_examples_declared'] = balance[
                    'declared_examples']
                fingerprint[f'{partition}_spec_list_digest'] = balance[
                    'spec_list_digest']
            self.state.payload['dataset'] = fingerprint
            self.record_stage(name, 'COMPLETE', {
                'orchestrator_version': ORCHESTRATOR_VERSION,
                **{k: v for k, v in fingerprint.items() if k != 'held_out_integrity'},
            }, detail={
                'train_candidates': fingerprint['train_candidates'],
                'validation_candidates': fingerprint['validation_candidates'],
                'train_examples': fingerprint['train_examples_declared'],
                'validation_examples': fingerprint['validation_examples_declared'],
                'held_out_candidates_metadata_only':
                    fingerprint['held_out_candidates_metadata_only'],
            })
        self.log('    preloading model-selection geometry ...')
        train_config = self._config_for('pointnet', gates.COMPARISON_SEED)
        validation_config = engine.TrainConfig.from_row(
            {**train_config.as_row(), 'partition': 'validation'})
        self.train_sources['train'] = dataset_source.build_model_selection_source(
            train_config, provenance, cache_root=self.layout.cache_root,
            use_cache=self.use_cache, preload=True, generator_digest=generator_digest)
        self.validation_source = dataset_source.build_model_selection_source(
            validation_config, provenance, cache_root=self.layout.cache_root,
            use_cache=self.use_cache, preload=True, generator_digest=generator_digest)
        self.log(f"    dataset digests: train "
                 f"{self.train_sources['train'].digest()[:16]}  validation "
                 f"{self.validation_source.digest()[:16]}")
        self.log(f"    declared examples: train "
                 f"{len(self.train_sources['train'])}  validation "
                 f"{len(self.validation_source)}  "
                 f"(candidate cap {self.candidate_cap})")

    def stage_parity(self) -> None:
        name = S_PARITY
        provenance = self._load_provenance()
        if self.done(name, heldout_adapter_parity='PASS'):
            recorded = state_module.read_json(
                self.layout.path('evaluation', 'HELDOUT_ADAPTER_PARITY.json'),
                verify=False)
            if recorded:
                self.state.payload['parity'] = recorded
                self.state.save()
                return
            self.record_stage(name, 'STALE', {'orchestrator_version':
                                              ORCHESTRATOR_VERSION})
        self.guard_pre_unblind(name)
        self.record_stage(name, 'RUNNING')
        self.log('    parity: frozen pipeline vs held-out adapter on real validation '
                 'candidates, all 7 classes x 3 severities ...')
        report = heldout_module.require_parity(provenance,
                                               points=self.points)
        coverage = heldout_module.parity_report(
            provenance, candidate_ids=list(provenance.candidates('validation'))[:5],
            points=gates.POINTS_ROBUSTNESS)
        report['robustness_points_coverage'] = {
            'points': gates.POINTS_ROBUSTNESS,
            'examples': coverage['examples'],
            'HELDOUT_ADAPTER_PARITY': coverage['HELDOUT_ADAPTER_PARITY'],
            'partitions': 'validation only; no held-out geometry read',
        }
        if coverage['HELDOUT_ADAPTER_PARITY'] != 'PASS':
            raise heldout_module.AdapterDrift({
                'message': 'adapter parity failed at 2048 points',
                'mismatches': coverage['mismatches'][:5]})
        state_module.atomic_write_json(
            self.layout.path('evaluation', 'HELDOUT_ADAPTER_PARITY.json'), report,
            with_digest=False)
        self.state.payload['parity'] = report
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'heldout_adapter_parity': report['HELDOUT_ADAPTER_PARITY'],
            'parity_digest': report['parity_digest'],
            'examples': report['examples'],
            'points': report['points'],
        }, detail={'examples': report['examples'],
                   'mismatches': report['mismatch_count'],
                   'held_out_geometry_read': False})

    def _config_for(self, model: str, seed: int, points: int | None = None,
                    repeats: int | None = None, seed_base: int | None = None,
                    max_candidates: int | None = None) -> engine.TrainConfig:
        return build_train_config(
            model, seed, self.calibration, self.layout, epochs=self.epochs,
            points=self.points if points is None else points,
            repeats=self.repeats if repeats is None else repeats,
            seed_base=self.seed_base if seed_base is None else seed_base,
            max_train_candidates=self.candidate_cap if max_candidates is None
            else max_candidates,
            max_validation_candidates=self.candidate_cap if max_candidates is None
            else max_candidates)

    @property
    def candidate_cap(self) -> int | None:
        """The bounded candidate count, or None for the formal full-data run.

        A smoke test must bound the training and validation sets *together*: bounding only
        one side would train on one population and select a threshold on another. The
        formal run has this property return ``None``, and the CLI refuses to start it with
        any bounded option set.
        """
        return self.comparison_candidates

    def _cache_root(self):
        return self.layout.cache_root

    def _sources_for(self, model: str, seed: int) -> tuple:
        """Build (train, validation) sources for one (model, seed) run.

        The run is bound to the candidate cap declared for the formal experiment (``None``)
        or for the smoke test. The cap is applied to train and validation together, so a
        model is never trained on one population and threshold-selected on another.
        Preloading is left to the prewarm stage, which iterates every index anyway.
        """
        config = self._config_for(model, seed)
        train = dataset_source.build_model_selection_source(
            config, self._load_provenance(), cache_root=self.layout.cache_root,
            use_cache=self.use_cache, preload=False)
        validation_config = engine.TrainConfig.from_row(
            {**config.as_row(), 'partition': 'validation'})
        validation = dataset_source.build_model_selection_source(
            validation_config, self._load_provenance(), cache_root=self.layout.cache_root,
            use_cache=self.use_cache, preload=False)
        return train, validation

    def prepare_training_data(self, model: str) -> dict:
        """Prewarm and vet the training data once, before any GPU time is spent.

        This measures how many declared specs the frozen pipeline can actually construct.
        ``gate_d_generator`` refuses a defect the candidate's real geometry cannot support,
        and the frozen dataset propagates that refusal instead of substituting an example.
        The count, the reasons and the per-class breakdown are recorded and must stay inside
        the declared exclusion limit, otherwise the run stops here rather than training on a
        quietly shrunken population.

        It is also what makes the epoch affordable: after one generation pass every declared
        example is a verified cache read, so each epoch is a memory-mapped load rather than
        12 915 fresh scene reconstructions, and the loader's applicability mask can be built
        from the recorded exclusions instead of regenerating the population.
        """
        name = S_PREWARM
        evidence = self.state.stage(name).get('evidence', {})
        self.prewarm_summary = None
        train, validation = self._sources_for(model, gates.COMPARISON_SEED)
        if not self.use_cache:
            self.log('    cache disabled: examples are regenerated every epoch')
            return {'train': train, 'validation': validation,
                    'prewarm': {'skipped': 'cache disabled'}}
        if self.done(name, train_dataset_digest=train.digest(),
                     validation_dataset_digest=validation.digest()):
            self.prewarm_summary = state_module.read_json(
                self.layout.path('state', 'TRAINING_DATA_PREWARM.json'), verify=False)
            return {'train': train, 'validation': validation,
                    'prewarm': evidence}
        self.guard_pre_unblind(name)
        self.record_stage(name, 'RUNNING')
        self.log(f'    prewarming {len(train)} train and {len(validation)} validation '
                 'examples once (each is then a verified cache read) ...')
        tracker_train = trainer_module.NotApplicableTracker()
        tracker_validation = trainer_module.NotApplicableTracker()
        train_result = train.prewarm(
            tracker_train,
            progress=lambda row: self.progress(f'{name}.train', row))
        self.log(f"    train prewarm done: {train_result['generated']} generated, "
                 f"{train_result['excluded']} not applicable "
                 f"({train_result['seconds']}s)")
        validation_result = validation.prewarm(
            tracker_validation,
            progress=lambda row: self.progress(f'{name}.validation', row))
        summary = {
            'train': {**train_result,
                      'not_applicable': tracker_train.enforce(len(train.dataset.specs))},
            'validation': {**validation_result,
                           'not_applicable': tracker_validation.enforce(
                               len(validation.dataset.specs))},
        }
        self.prewarm_summary = summary
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'train_dataset_digest': train.digest(),
            'validation_dataset_digest': validation.digest(),
            'train_excluded': summary['train']['excluded'],
            'validation_excluded': summary['validation']['excluded'],
        }, detail=summary)
        state_module.atomic_write_json(
            self.layout.path('state', 'TRAINING_DATA_PREWARM.json'), summary,
            with_digest=False)
        self.log(f"    train: {train_result['generated']} generated / "
                 f"{train_result['excluded']} not applicable "
                 f"({train_result['seconds']}s); validation: "
                 f"{validation_result['generated']} / {validation_result['excluded']} "
                 f"({validation_result['seconds']}s)")
        return {'train': train, 'validation': validation, 'prewarm': summary}

    def run_or_reuse(self, model: str, seed: int, stage: str,
                     max_candidates: int | None = None,
                     condition: str = 'comparison') -> dict:
        """Train one (model, seed), or re-use a completed run that still matches.

        A recorded run is re-used only when its config digest, dataset digest, checkpoint
        digest and frozen threshold all still match. When the caller passes a stage, that
        stage's recorded evidence must match too; when it passes ``None`` (the three-seed
        stage covers all seeds at once), the run-level digests are the whole test. Either
        way, a completed and verified run is never retrained: the earlier behaviour of
        retraining a matching run because no stage pointer existed wasted hours of GPU time
        for exactly the work this contract exists to avoid.
        """
        key = run_key(model, seed) + (f'|cap{max_candidates}'
                                      if max_candidates else '')
        config = self._config_for(model, seed, max_candidates=max_candidates)
        recorded = self.state.model_run(key)
        if recorded and recorded.get('status') == 'COMPLETE':
            ok, reason = trainer_module.completed_run_is_valid(
                recorded, config, recorded.get('dataset_digest', ''))
            stage_ok = True
            if stage is not None:
                stage_ok, stage_reason = state_module.stage_is_complete(
                    self.state.stage(stage),
                    {'orchestrator_version': ORCHESTRATOR_VERSION,
                     'checkpoint_sha256': recorded.get('checkpoint_sha256'),
                     'config_digest': recorded.get('config_digest')})
                if not stage_ok:
                    reason = stage_reason
            if ok and stage_ok:
                self.log(f'    {key}: COMPLETE and digest-verified — not retrained')
                return recorded
            self.log(f'    {key}: recorded COMPLETE but STALE ({reason})')
            self.state.set_model_run(key, {'status': 'STALE', 'stale_reason': reason})
            self.state.save()
        self.guard_pre_unblind(stage)
        self.state.set_model_run(key, {'status': 'RUNNING', 'model': model,
                                       'seed': seed, 'condition': condition,
                                       'config_digest': config.digest()})
        self.state.save()
        expected_checkpoint = self.layout.runs_root / f'{model}_seed{seed}' / 'last.pt'
        if expected_checkpoint.exists():
            self.log(f'    found {expected_checkpoint}; the trainer will resume it only '
                     'if its config and dataset digests still match')
        self.log(f'    training {model} seed {seed} '
                 f'(batch {config.batch_size}, points {config.points}, '
                 f'epochs {config.epochs}, max candidates {max_candidates}) ...')
        started = time.time()

        def progress(row):
            self.log_line(f'{key} epoch {row["epoch"]} train_loss '
                          f'{row["train_loss"]} val_BA {row["validation_BALANCED_ACCURACY"]} '
                          f'val_BR {row["validation_BAD_RECALL"]}')
            if self.verbose:
                console(f'        epoch {row["epoch"]:>3}  train_loss '
                        f'{row["train_loss"]:.4f}  val_BA '
                        f'{row["validation_BALANCED_ACCURACY"]:.4f}  val_BAD_RECALL '
                        f'{row["validation_BAD_RECALL"]:.4f}  thr '
                        f'{row["validation_threshold"]:.3f}')

        run = trainer_module.train_run(
            config, self._load_provenance(), cache_root=self._cache_root(),
            use_cache=self.use_cache, resume=True,
            audit_dir=self.layout.path('runs', 'audit'), progress=progress,
            prewarmed=bool(self.prewarm_summary))
        run['status'] = 'COMPLETE'
        run['condition'] = condition
        run['seconds_wall'] = round(time.time() - started, 3)
        run['gate_d_generator_digest'] = diagnostic_module.generator_digest()
        if self.validation_source is not None:
            run['full_validation'] = full_validation_metrics(run, self.validation_source)
        self.state.set_model_run(key, run)
        self.state.save()
        self.log(f"    {key}: COMPLETE  best epoch {run['best']['epoch']}  "
                 f"threshold {run['threshold']:.4f}  "
                 f"checkpoint {run['checkpoint_sha256'][:16]}")
        return run

    def stage_compare(self, model: str, stage: str) -> None:
        if self.done(stage, model=model, seed=gates.COMPARISON_SEED):
            key = run_key(model, gates.COMPARISON_SEED)
            self.comparison[model] = 'COMPLETE'
            self.runs.setdefault(gates.COMPARISON_SEED,
                                 self.state.model_run(key))
            return
        self.record_stage(stage, 'RUNNING')
        run = self.run_or_reuse(model, gates.COMPARISON_SEED, stage, max_candidates=None)
        self.comparison[model] = 'COMPLETE'
        self.runs[gates.COMPARISON_SEED] = run
        self.validation_evidence[model] = run['full_validation']['metrics']
        self.record_stage(stage, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'model': model, 'seed': gates.COMPARISON_SEED,
            'config_digest': run['config_digest'],
            'dataset_digest': run['dataset_digest'],
            'checkpoint_sha256': run['checkpoint_sha256'],
        }, detail={'best_epoch': run['best']['epoch'],
                   'threshold': run['threshold'],
                   'validation_BALANCED_ACCURACY':
                       run['full_validation']['BALANCED_ACCURACY'],
                   'validation_BAD_RECALL': run['full_validation']['BAD_RECALL'],
                   'validation_BAD_FALSE_PASS':
                       run['full_validation']['BAD_FALSE_PASS'],
                   'validation_GOOD_FALSE_REVIEW':
                       run['full_validation']['GOOD_FALSE_REVIEW']})

    def stage_learnability(self) -> None:
        name = S_LEARNABILITY
        evidence = self.state.stage(name).get('evidence', {})
        if self.done(name, **{k: evidence.get(k) for k in
                              ('pointnet_balanced_accuracy',)}):
            return
        missing = [model for model in gates.ARCHITECTURES
                   if model not in self.validation_evidence]
        for model in missing:
            key = run_key(model, gates.COMPARISON_SEED)
            run = self.state.model_run(key)
            if not run:
                raise PilotStop('STOP_RESUME_MISMATCH', {
                    'message': f'the seed-101 comparison run for {model} is missing'})
            self.validation_evidence[model] = (
                run['full_validation']['metrics'] if run.get('full_validation')
                else full_validation_metrics(run, self.validation_source))
        verdicts = {model: evaluate_module.learnability_verdict(
            self.validation_evidence[model], learnability_limits(self.smoke_test))
            for model in gates.ARCHITECTURES}
        learnable = all(entry['passed'] for entry in verdicts.values())
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'pointnet_balanced_accuracy':
                self.validation_evidence['pointnet']['BALANCED_ACCURACY'],
            'pointnet_bad_recall':
                self.validation_evidence['pointnet']['BAD_RECALL'],
            'pointnet2_balanced_accuracy':
                self.validation_evidence['pointnet2_ssg']['BALANCED_ACCURACY'],
            'pointnet2_bad_recall':
                self.validation_evidence['pointnet2_ssg']['BAD_RECALL'],
            'learnable': learnable,
        }, detail={'verdicts': verdicts, 'held_out_accessed': False})
        self.log(f"    learnability: pointnet BA "
                 f"{self.validation_evidence['pointnet']['BALANCED_ACCURACY']:.4f} / "
                 f"BR {self.validation_evidence['pointnet']['BAD_RECALL']:.4f}, "
                 f"pointnet2 BA "
                 f"{self.validation_evidence['pointnet2_ssg']['BALANCED_ACCURACY']:.4f} "
                 f"/ BR {self.validation_evidence['pointnet2_ssg']['BAD_RECALL']:.4f} "
                 f"-> {'LEARNABLE' if learnable else 'NOT_LEARNABLE'}")
        if not learnable:
            raise PilotStop('PUBLIC_PILOT_NOT_LEARNABLE', {
                'message': 'neither architecture learned the task above the validation '
                           'floors; held-out stays locked and the budget is not changed',
                'validation_evidence': {model: self.validation_evidence[model]
                                        for model in gates.ARCHITECTURES},
                'verdicts': verdicts,
                'gate': learnability_limits(self.smoke_test),
                'forbidden_actions': ['more epochs', 'larger model', 'threshold change',
                                      'opening held-out']})

    def stage_architecture(self) -> None:
        name = S_ARCHITECTURE
        evidence = self.state.stage(name).get('evidence', {})
        if self.done(name, selected_architecture=evidence.get('selected_architecture')):
            self.state.payload['selected_architecture'] = evidence.get(
                'selected_architecture')
            self.state.save()
            return
        for model in gates.ARCHITECTURES:
            if model not in self.validation_evidence:
                key = run_key(model, gates.COMPARISON_SEED)
                run = self.state.model_run(key)
                if run.get('full_validation'):
                    self.validation_evidence[model] = run['full_validation']['metrics']
        try:
            selected = freeze_module.select_architecture(self.validation_evidence)
        except freeze_module.FreezeError as exc:
            raise PilotStop(exc.code, exc.detail) from exc
        self.state.payload['selected_architecture'] = selected['selected_architecture']
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'selected_architecture': selected['selected_architecture'],
            'selection_basis_digest': state_module.sha256_text(
                state_module.canonical(selected['basis'])),
        }, detail=selected)
        self.log(f"    selected architecture: {selected['selected_architecture']}"
                 + ('  (non-inferiority policy chose the simpler model)'
                    if selected.get('non_inferiority_applied') else ''))

    def stage_seeds(self) -> None:
        name = S_SEEDS
        selected = self.state.payload.get('selected_architecture')
        if not selected:
            raise PilotStop('STOP_CONFIG_MISMATCH',
                            {'message': 'no architecture selected before seed training'})
        evidence = self.state.stage(name).get('evidence', {})
        if self.done(name, selected_architecture=selected,
                     runs_digest=evidence.get('runs_digest')):
            for seed in gates.MODEL_SEEDS:
                self.runs[seed] = self.state.model_run(run_key(selected, seed))
            return
        self.record_stage(name, 'RUNNING')
        for seed in gates.MODEL_SEEDS:
            key = run_key(selected, seed)
            recorded = self.state.model_run(key)
            run = self.run_or_reuse(selected, seed, None, condition='final_seed')
            self.runs[seed] = run
            run['threshold_path'] = str(trainer_module.write_threshold_file(
                self.layout.path('freeze', 'thresholds'), run, run['dataset_digest']))
            run['threshold_sha256'] = state_module.digest_file(run['threshold_path'])
            self.state.set_model_run(key, run)
            self.state.save()
            if recorded and recorded.get('checkpoint_sha256') == run.get(
                    'checkpoint_sha256'):
                self.log(f'    seed {seed}: re-used the identical checkpoint')
        runs_digest = state_module.sha256_text(state_module.canonical(
            {str(seed): {'checkpoint_sha256': self.runs[seed]['checkpoint_sha256'],
                         'config_digest': self.runs[seed]['config_digest'],
                         'dataset_digest': self.runs[seed]['dataset_digest'],
                         'threshold': self.runs[seed]['threshold']}
             for seed in gates.MODEL_SEEDS}))
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'selected_architecture': selected,
            'runs_digest': runs_digest,
        }, detail={'seeds': list(gates.MODEL_SEEDS), 'runs_digest': runs_digest})

    def stage_thresholds(self) -> None:
        name = S_THRESHOLDS
        selected = self.state.payload['selected_architecture']
        evidence = self.state.stage(name).get('evidence', {})
        if self.done(name, thresholds_digest=evidence.get('thresholds_digest')):
            return
        thresholds = {}
        for seed in gates.MODEL_SEEDS:
            run = self.runs.get(seed) or self.state.model_run(run_key(selected, seed))
            if not run:
                raise PilotStop('STOP_RESUME_MISMATCH',
                                {'message': f'seed {seed} run record missing'})
            path = self.layout.path('freeze', 'thresholds',
                                    f'seed{seed}_threshold.json')
            if not path.exists():
                path = trainer_module.write_threshold_file(
                    self.layout.path('freeze', 'thresholds'), run,
                    run['dataset_digest'])
            payload = state_module.read_json(path, verify=False)
            if payload.get('threshold') != run['threshold']:
                raise PilotStop('STOP_CONFIG_MISMATCH', {
                    'message': f'seed {seed} threshold file disagrees with the run',
                    'file': str(path), 'file_threshold': payload.get('threshold'),
                    'run_threshold': run['threshold']})
            if payload.get('epoch_selection_rule') != 'best_epoch_validation_selection':
                raise PilotStop('STOP_CONFIG_MISMATCH', {
                    'message': f'seed {seed} best epoch was not chosen by the frozen '
                               'validation-only rule',
                    'epoch_selection_rule': payload.get('epoch_selection_rule')})
            if payload.get('operating_point_rule') not in (
                    'max_bad_recall_within_good_false_review_budget',
                    'fallback_min_good_false_review'):
                raise PilotStop('STOP_CONFIG_MISMATCH', {
                    'message': f'seed {seed} threshold was not chosen by the frozen '
                               'operating-point rule',
                    'operating_point_rule': payload.get('operating_point_rule')})
            thresholds[str(seed)] = {'threshold': payload['threshold'],
                                     'path': str(path),
                                     'sha256': state_module.digest_file(path),
                                     'epoch_selection_rule':
                                         payload['epoch_selection_rule'],
                                     'operating_point_rule':
                                         payload['operating_point_rule'],
                                     'threshold_frozen': True}
        digest = state_module.sha256_text(state_module.canonical(thresholds))
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'thresholds_digest': digest,
        }, detail={'thresholds': thresholds})
        self.log('    thresholds frozen on validation only: '
                 + ', '.join(f"seed{k}={v['threshold']:.4f}"
                             for k, v in thresholds.items()))

    def _validation_evidence_for_seed(self, seed: int) -> dict:
        selected = self.state.payload['selected_architecture']
        run = self.runs.get(seed) or self.state.model_run(run_key(selected, seed))
        if run.get('full_validation'):
            return run['full_validation']['metrics']
        metrics = full_validation_metrics(run, self.validation_source)
        run['full_validation'] = metrics
        self.runs[seed] = run
        self.state.set_model_run(run_key(selected, seed), run)
        self.state.save()
        return metrics['metrics']

    def stage_deployment(self) -> None:
        name = S_DEPLOYMENT
        evidence = self.state.stage(name).get('evidence', {})
        if self.done(name, deployment_candidate_seed=evidence.get(
                'deployment_candidate_seed')):
            self.state.payload['deployment_candidate_seed'] = evidence.get(
                'deployment_candidate_seed')
            self.state.save()
            return
        per_seed = {seed: self._validation_evidence_for_seed(seed)
                    for seed in gates.MODEL_SEEDS}
        try:
            decision = freeze_module.select_deployment_candidate(per_seed)
        except freeze_module.FreezeError as exc:
            raise PilotStop(exc.code, exc.detail) from exc
        self.state.payload['deployment_candidate_seed'] = decision[
            'deployment_candidate_seed']
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'deployment_candidate_seed': decision['deployment_candidate_seed'],
            'validation_only': True,
        }, detail=decision)
        self.log(f"    deployment candidate seed: "
                 f"{decision['deployment_candidate_seed']} (validation only)")

    def stage_probe(self) -> None:
        """A short measured sampling probe on validation data, before the freeze."""
        name = S_PROBE
        selected = self.state.payload['selected_architecture']
        deployment = int(self.state.payload['deployment_candidate_seed'])
        run = self.runs.get(deployment) or self.state.model_run(
            run_key(selected, deployment))
        if self.done(name, checkpoint_sha256=run.get('checkpoint_sha256'),
                     points=gates.POINTS_ROBUSTNESS):
            return
        self.guard_pre_unblind(name)
        self.record_stage(name, 'RUNNING')
        import efg_dataset as dataset_source
        config = engine.TrainConfig.from_row({
            **build_train_config(selected, deployment, self.calibration, self.layout,
                                 epochs=self.epochs,
                                 points=gates.POINTS_ROBUSTNESS,
                                 repeats=self.repeats,
                                 seed_base=self.seed_base).as_row(),
            'partition': 'validation'})
        source = dataset_source.build_model_selection_source(
            config, self._load_provenance(), cache_root=self.layout.cache_root,
            use_cache=self.use_cache, preload=True)
        scored = trainer_module.score_run_checkpoint(run, source)
        metrics = evaluate_module.binary_metrics(scored['labels'], scored['scores'],
                                                 float(run['threshold']))
        primary = evaluate_module.binary_metrics(
            scored['labels'], scored['scores'], float(run['threshold']))
        payload = {
            'partition': 'validation',
            'points': gates.POINTS_ROBUSTNESS,
            'checkpoint_sha256': run['checkpoint_sha256'],
            'threshold': float(run['threshold']),
            'metrics': metrics,
            'primary_points_metrics': primary,
            'note': 'validation-only probe; no held-out data is involved and no result '
                    'here can change a frozen decision',
        }
        state_module.atomic_write_json(
            self.layout.path('robustness', 'VALIDATION_SAMPLING_PROBE.json'), payload,
            with_digest=False)
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'checkpoint_sha256': run['checkpoint_sha256'],
            'points': gates.POINTS_ROBUSTNESS,
            'bad_recall': metrics['BAD_RECALL'],
        }, detail={'validation_BAD_RECALL_2048': metrics['BAD_RECALL'],
                   'validation_BALANCED_ACCURACY': metrics['BALANCED_ACCURACY']})

    def stage_freeze(self) -> None:
        name = S_FREEZE
        selected = self.state.payload['selected_architecture']
        state = freeze_module.freeze_state(self.layout.freeze_path)
        if state['valid']:
            ok, reason = state_module.stage_is_complete(
                self.state.stage(name), {'orchestrator_version': ORCHESTRATOR_VERSION,
                                         'freeze_sha256': state['sha256']})
            if ok:
                self.state.payload['freeze'] = {'created': True, 'sha256': state['sha256'],
                                                'path': str(self.layout.freeze_path)}
                self.state.save()
                return
            self.log(f'    freeze exists but the stage record is stale ({reason}); '
                     'revalidating')
        self.guard_pre_unblind(name)
        self.record_stage(name, 'RUNNING')
        for seed in gates.MODEL_SEEDS:
            key = run_key(selected, seed)
            run = self.runs.get(seed) or self.state.model_run(key)
            if not run or run.get('status') != 'COMPLETE':
                raise PilotStop('STOP_RESUME_MISMATCH', {
                    'message': f'seed {seed} is not a completed, frozen run'})
            threshold_path = self.layout.path('freeze', 'thresholds',
                                              f'seed{seed}_threshold.json')
            run['threshold_path'] = str(threshold_path)
            run['threshold_sha256'] = (state_module.digest_file(threshold_path)
                                       if threshold_path.exists() else None)
            run['threshold_frozen'] = True
            self.runs[seed] = run
        try:
            freeze_module.require_all_frozen(self.runs, gates.MODEL_SEEDS)
        except freeze_module.FreezeError as exc:
            raise PilotStop(exc.code, exc.detail) from exc

        comparison = {
            'pointnet': {
                'validation_metrics': self.validation_evidence.get(
                    'pointnet', self._validation_evidence_for_seed(
                        gates.COMPARISON_SEED)),
                'complete': self.comparison.get('pointnet', 'COMPLETE'),
            },
            'pointnet2_ssg': {
                'validation_metrics': self.validation_evidence.get(
                    'pointnet2_ssg', self._validation_evidence_for_seed(
                        gates.COMPARISON_SEED)),
                'complete': self.comparison.get('pointnet2_ssg', 'COMPLETE'),
            },
        }
        comparison['summary'] = {
            'same_train_candidates': True, 'same_validation_candidates': True,
            'same_examples': True, 'same_sampler': True, 'same_points': True,
            'same_corruption': True, 'same_budget': True, 'same_effective_batch': True,
            'same_threshold_rule': True, 'same_early_stopping_rule': True,
            'only_difference': 'network architecture',
        }
        training_config = self._config_for(selected, gates.COMPARISON_SEED).as_row()
        validation_dataset = {
            'partition': 'validation',
            'digest': self.validation_source.digest(),
            'points': self.points,
            'candidates': len(self._load_provenance().candidates('validation')),
            'source': 'frozen GateDDataset, validation partition only',
        }
        deployment = {
            'deployment_candidate_seed': int(
                self.state.payload['deployment_candidate_seed']),
            'basis': self.state.stage(S_DEPLOYMENT).get('detail', {}),
        }
        payload = freeze_module.build_freeze(
            gate_d_binding=self.state.payload['accepted_gate_d_binding'],
            selected={'selected_architecture': selected,
                      'basis': self.state.stage(S_ARCHITECTURE).get('detail', {}).get(
                          'basis', {}),
                      'non_inferiority_applied': self.state.stage(
                          S_ARCHITECTURE).get('detail', {}).get(
                              'non_inferiority_applied', False)},
            comparison_evidence=comparison,
            training_config=training_config,
            model_seeds=gates.MODEL_SEEDS,
            runs=self.runs,
            validation_dataset=validation_dataset,
            per_seed_validation={seed: self._validation_evidence_for_seed(seed)
                                 for seed in gates.MODEL_SEEDS},
            deployment=deployment,
            freeze_timestamp=state_module.local_timestamp(),
            robustness=gates.robustness_plan(),
            orchestrator_version=ORCHESTRATOR_VERSION)
        payload['dataset_fingerprint'] = self.state.payload.get('dataset')
        payload['adapter_parity'] = {
            'HELDOUT_ADAPTER_PARITY': self.state.payload.get('parity', {}).get(
                'HELDOUT_ADAPTER_PARITY'),
            'parity_digest': self.state.payload.get('parity', {}).get('parity_digest'),
        }
        payload['heldout_manifest_expectation'] = {
            'expected_candidates': len(self._load_provenance().held_out_eligible),
            'source': 'frozen provenance runtime count, read before unblinding',
        }
        path, digest = freeze_module.write_freeze(self.layout.freeze_path, payload)
        state = freeze_module.require_valid_freeze(self.layout.freeze_path)
        self.state.payload['freeze'] = {'created': True, 'sha256': digest,
                                        'path': str(path)}
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'freeze_sha256': digest,
            'selected_architecture': selected,
            'model_seeds': list(gates.MODEL_SEEDS),
            'deployment_candidate_seed': payload['deployment_candidate_seed'],
            'heldout_candidates_expected': payload[
                'heldout_manifest_expectation']['expected_candidates'],
        }, detail={'freeze_sha256': digest,
                   'held_out_locked': True,
                   'held_out_accessed': False})
        self.log(f'    PUBLIC_EXPERIMENT_FREEZE written: {digest[:16]}  '
                 f'held-out locked = YES')

    def load_frozen_runs(self, selected: str) -> dict:
        """Reload the three frozen runs from persisted state, thresholds included.

        ``self.runs`` is in-memory and empty on a restart, so every stage that runs after
        the three-seed stage must rebuild it from the state file. The threshold files are
        re-read and re-marked frozen here: a restart must arrive at exactly the same
        precondition the first attempt had, otherwise the run fails its own guard for a
        reason that has nothing to do with the experiment.
        """
        for seed in gates.MODEL_SEEDS:
            run = self.runs.get(seed) or self.state.model_run(run_key(selected, seed))
            if not run:
                raise PilotStop('STOP_RESUME_MISMATCH', {
                    'message': f'no recorded run for {selected} seed {seed}',
                    'seed': seed, 'model': selected})
            path = self.layout.path('freeze', 'thresholds', f'seed{seed}_threshold.json')
            if not path.exists():
                path = trainer_module.write_threshold_file(
                    self.layout.path('freeze', 'thresholds'), run,
                    run['dataset_digest'])
            payload = state_module.read_json(path, verify=False)
            if payload.get('threshold') != run['threshold']:
                raise PilotStop('STOP_CONFIG_MISMATCH', {
                    'message': f'seed {seed} threshold file disagrees with the run',
                    'file': str(path), 'file_threshold': payload.get('threshold'),
                    'run_threshold': run['threshold']})
            run['threshold_path'] = str(path)
            run['threshold_sha256'] = state_module.digest_file(path)
            run['threshold_frozen'] = True
            self.runs[seed] = run
        return self.runs

    def stage_heldout(self) -> None:
        name = S_HELDOUT
        audit = self.state.payload['heldout']
        freeze_state = freeze_module.require_valid_freeze(self.layout.freeze_path)
        if self.done(name, freeze_sha256=freeze_state['sha256'],
                     heldout_accessed='YES'):
            self.heldout_records = leakage_module.read_records(
                self.layout.path('evaluation', 'heldout'))
            if self.heldout_records:
                return
            raise PilotStop('STOP_HELDOUT_MANIFEST_CHANGED', {
                'message': 'the held-out stage is recorded complete but no held-out '
                           'evaluation record exists',
                'directory': str(self.layout.path('evaluation', 'heldout'))})
        self.guard_no_tuning_after_unblind()
        provenance = self._load_provenance()
        selected = freeze_state['payload']['selected_architecture']
        thresholds = freeze_state['payload']['frozen_thresholds']
        self.load_frozen_runs(selected)

        # Every precondition, checked in order, before one held-out point is read.
        parity = self.state.payload.get('parity') or {}
        if parity.get('HELDOUT_ADAPTER_PARITY') != 'PASS':
            raise PilotStop('STOP_HELDOUT_ADAPTER_DRIFT', {
                'message': 'held-out evaluation requires a passing adapter parity gate',
                'parity': parity.get('HELDOUT_ADAPTER_PARITY')})
        try:
            freeze_module.require_all_frozen(self.runs, gates.MODEL_SEEDS)
        except freeze_module.FreezeError as exc:
            raise PilotStop(exc.code, exc.detail) from exc
        held_view = heldout_module.held_out_view()
        manifest_check = heldout_module.require_manifest_count(
            held_view,
            freeze_state['payload']['heldout_manifest_expectation'][
                'expected_candidates'])
        integrity = heldout_module.assert_heldout_integrity(
            provenance, held_view,
            expected_candidates=freeze_state['payload'][
                'heldout_manifest_expectation']['expected_candidates'])

        self.record_stage(name, 'RUNNING', detail={
            'unblinding': True, 'freeze_sha256': freeze_state['sha256'],
            'manifest_check': manifest_check, 'integrity': integrity})
        audit = self.state.mark_heldout_accessed(freeze_state['sha256'])
        self.state.payload['heldout']['manifest_check'] = manifest_check
        self.state.payload['heldout']['parity_at_unblind'] = parity.get(
            'HELDOUT_ADAPTER_PARITY')
        self.state.save()
        self.log(f"    HELDOUT_ACCESSED = YES  first access "
                 f"{audit['first_access_timestamp']}  freeze "
                 f"{freeze_state['sha256'][:16]}  candidates "
                 f"{manifest_check['actual']}")
        state_module.atomic_write_json(self.layout.unblind_audit_path, {
            'HELDOUT_ACCESSED': 'YES',
            'first_access_timestamp': audit['first_access_timestamp'],
            'PUBLIC_EXPERIMENT_FREEZE_SHA256': freeze_state['sha256'],
            'manifest_check': manifest_check,
            'integrity': integrity,
            'adapter_parity': parity.get('HELDOUT_ADAPTER_PARITY'),
            'thresholds_used': thresholds,
            'selected_architecture': selected,
            'note': 'irreversible; after this point no model, seed, threshold or '
                    'hyper-parameter may change',
        }, with_digest=False)

        for seed in gates.MODEL_SEEDS:
            run = self.runs.get(seed) or self.state.model_run(run_key(selected, seed))
            threshold = float(freeze_state['payload']['frozen_thresholds'][str(seed)])
            if abs(threshold - float(run['threshold'])) > 1e-12:
                raise PilotStop('STOP_CONFIG_MISMATCH', {
                    'message': f'seed {seed} threshold differs from the frozen value',
                    'frozen': threshold, 'run': run['threshold']})
            record = self.evaluate_heldout(selected, seed, run, threshold,
                                           points=self.points, sampling_seed=0,
                                           label=f'seed{seed}_primary')
            self.heldout_records[record['name']] = record
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'freeze_sha256': freeze_state['sha256'],
            'heldout_accessed': 'YES',
            'seeds': list(gates.MODEL_SEEDS),
        }, detail={'heldout_accessed': True,
                   'records': sorted(self.heldout_records)})

    def evaluate_heldout(self, model: str, seed: int, run: dict, threshold: float,
                         points: int, sampling_seed: int, label: str) -> dict:
        """One held-out scoring pass. Requires a valid freeze; never selects anything."""
        freeze_state = freeze_module.require_valid_freeze(self.layout.freeze_path)
        provenance = self._load_provenance()
        held_view = heldout_module.held_out_view()
        candidate_ids = None
        if self.max_heldout_candidates is not None:
            candidate_ids = list(held_view.eligible)[:self.max_heldout_candidates]
        dataset = heldout_module.HeldOutDataset(
            provenance, held_view, candidate_ids=candidate_ids, points=points,
            repeats=1, seed_base=self.seed_base,
            cache_root=self.layout.path('cache', 'heldout_geometry'))
        dataset.preload()
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        payload = engine.load_checkpoint(run['checkpoint_path'], map_location=device)
        model_object = engine.build_model(model).to(device)
        model_object.load_state_dict(payload['model_state'])
        scored = evaluate_module.score_dataset(
            model_object, dataset, device,
            batch_size=int(run['config']['batch_size']),
            progress=lambda row: self.progress(f'heldout.{label}', row),
            progress_every=40)
        del model_object
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        scored['partition'] = 'held_out'
        metrics = evaluate_module.score_partition(scored, threshold)
        metrics['partition'] = 'held_out'

        source_union_counts: dict[str, int] = {}
        for index in range(len(dataset.specs)):
            spec = dataset.specs[index]
            scene_sources = held_view.context.get(spec.candidate_id, ())
            source_union_counts[spec.candidate_id] = source_union_counts.get(
                spec.candidate_id, 0) + 1
            for source in scene_sources:
                source_union_counts[source] = source_union_counts.get(source, 0) + 1
        leakage = leakage_module.select_leakage_audit(
            {label: {'source_union_counts': source_union_counts}}, provenance)
        record = {
            'name': label,
            'partition': 'held_out',
            'model': model,
            'seed': int(seed),
            'points': int(points),
            'sampling_seed': int(sampling_seed),
            'threshold': float(threshold),
            'checkpoint_sha256': run['checkpoint_sha256'],
            'config_digest': run['config_digest'],
            'dataset_digest': run['dataset_digest'],
            'freeze_sha256': freeze_state['sha256'],
            'held_out_candidates': len(held_view.eligible),
            'held_out_evaluated_candidates': len(dataset.candidate_ids),
            'held_out_complete': len(dataset.candidate_ids) == len(held_view.eligible),
            'examples': int(scored['examples']),
            'declared_examples': scored.get('declared_examples'),
            'excluded_examples': scored.get('excluded_examples'),
            'excluded_by_reason': scored.get('excluded_by_reason'),
            'metrics': metrics,
            'per_defect_primary': metrics['per_defect_primary'],
            'per_defect_all_severities': metrics['per_defect_all_severities'],
            'per_severity': metrics['per_severity'],
            'three_way': metrics['three_way'],
            'overall': metrics['overall'],
            'primary_moderate_high': metrics['primary_moderate_high'],
            'leakage': leakage,
            'source_union_counts': source_union_counts,
            'seconds': scored['seconds'],
            'device': scored['device'],
        }
        payload_path = self.layout.path('evaluation', 'heldout', f'{label}.json')
        state_module.atomic_write_json(payload_path, record, with_digest=False)
        record['record_path'] = str(payload_path)
        # Per-example raw scores and labels are written alongside the metrics so the
        # operating point can be audited and re-thresholded after the fact without
        # re-running the evaluation. They are the evidence behind every reported number.
        raw_path = self.layout.path('evaluation', 'heldout', f'{label}_scores.npz')
        np.savez_compressed(
            raw_path,
            scores=np.asarray(scored['scores'], dtype=np.float64),
            labels=np.asarray(scored['labels'], dtype=np.int64),
            classes=np.asarray(scored['classes'], dtype='U32'),
            severities=np.asarray(scored['severities'], dtype='U8'),
            keys=np.asarray(scored['keys'], dtype='U200'))
        record['scores_path'] = str(raw_path)
        record['scores_sha256'] = state_module.digest_file(raw_path)
        primary = record['primary_moderate_high'] or {}
        self.log(f"    {label}: examples {record['examples']}  BAD_RECALL "
                 f"{primary.get('BAD_RECALL')}  BAD_FALSE_PASS "
                 f"{primary.get('BAD_FALSE_PASS')}  GOOD_FALSE_REVIEW "
                 f"{primary.get('GOOD_FALSE_REVIEW')}  BALANCED_ACCURACY "
                 f"{primary.get('BALANCED_ACCURACY')}  ({scored['seconds']}s)")
        return record

    def stage_aggregate(self) -> None:
        name = S_AGGREGATE
        per_seed = {}
        for seed in gates.MODEL_SEEDS:
            label = f'seed{seed}_primary'
            record = self.heldout_records.get(label) or state_module.read_json(
                self.layout.path('evaluation', 'heldout', f'{label}.json'),
                verify=False)
            if not record:
                raise PilotStop('STOP_HELDOUT_MANIFEST_CHANGED', {
                    'message': f'held-out record {label} is missing'})
            self.heldout_records[label] = record
            per_seed[seed] = record['primary_moderate_high']
        multi = evaluate_module.aggregate_seed_metrics(per_seed)
        multi['per_seed_primary'] = {str(seed): per_seed[seed]
                                     for seed in sorted(per_seed)}
        primary_gate = evaluate_module.apply_primary_gate(
            per_seed[int(self.state.payload['deployment_candidate_seed'])])
        self.state.payload['heldout']['multi_seed'] = multi
        self.state.payload['heldout']['primary_gate'] = primary_gate
        self.state.save()
        state_module.atomic_write_json(
            self.layout.path('evaluation', 'MULTI_SEED_AGGREGATE.json'),
            {'multi_seed': multi, 'primary_gate': primary_gate}, with_digest=False)
        if not self.done(name, multi_seed_digest=state_module.sha256_text(
                state_module.canonical(multi))):
            self.record_stage(name, 'COMPLETE', {
                'orchestrator_version': ORCHESTRATOR_VERSION,
                'multi_seed_digest': state_module.sha256_text(
                    state_module.canonical(multi)),
                'model_stability': multi['MODEL_STABILITY'],
            }, detail={'BAD_RECALL_mean': multi['metrics']['BAD_RECALL']['mean'],
                       'BAD_RECALL_min': multi['metrics']['BAD_RECALL']['min'],
                       'BAD_RECALL_max': multi['metrics']['BAD_RECALL']['max'],
                       'BAD_RECALL_std': multi['metrics']['BAD_RECALL']['std'],
                       'MODEL_STABILITY': multi['MODEL_STABILITY'],
                       'primary_gate': primary_gate})

    def stage_robustness(self) -> None:
        name = S_ROBUSTNESS
        freeze_state = freeze_module.require_valid_freeze(self.layout.freeze_path)
        selected = freeze_state['payload']['selected_architecture']
        deployment = int(freeze_state['payload']['deployment_candidate_seed'])
        plan = freeze_state['payload']['robustness_plan']
        self.load_frozen_runs(selected)
        run = self.runs.get(deployment) or self.state.model_run(
            run_key(selected, deployment))
        threshold = float(freeze_state['payload']['frozen_thresholds'][str(deployment)])
        primary_label = f'seed{deployment}_primary'
        primary_record = self.heldout_records.get(primary_label)
        if not primary_record:
            raise PilotStop('STOP_HELDOUT_MANIFEST_CHANGED', {
                'message': f'primary held-out record {primary_label} is missing'})
        primary_recall = primary_record['primary_moderate_high']['BAD_RECALL']
        configurations = {}
        for entry in plan['configurations']:
            label = entry['label']
            if label == primary_label or (entry['points'] == self.points
                                          and entry['sampling_seed'] == 0):
                configurations[label] = {
                    'BAD_RECALL': primary_recall,
                    'sampling_seed': 0, 'points': int(self.points),
                    'examples': primary_record['primary_moderate_high'].get('examples'),
                    'source': primary_label,
                }
                continue
            if entry['points'] == self.points and entry['sampling_seed'] == 0:
                continue
            if self.done(name, points=entry['points'],
                         sampling_seed=entry['sampling_seed'],
                         freeze_sha256=freeze_state['sha256']):
                record = state_module.read_json(
                    self.layout.path('robustness', f'{label}.json'), verify=False)
            else:
                self.log(f"    robustness {label} ...")
                record = self.evaluate_heldout(
                    selected, deployment, run, threshold, points=entry['points'],
                    sampling_seed=entry['sampling_seed'], label=label)
                state_module.atomic_write_json(
                    self.layout.path('robustness', f'{label}.json'), record,
                    with_digest=False)
            metrics = record['primary_moderate_high']
            configurations[label] = {
                'BAD_RECALL': metrics['BAD_RECALL'],
                'BAD_FALSE_PASS': metrics['BAD_FALSE_PASS'],
                'GOOD_FALSE_REVIEW': metrics['GOOD_FALSE_REVIEW'],
                'BALANCED_ACCURACY': metrics['BALANCED_ACCURACY'],
                'F1_REVIEW': metrics['F1_REVIEW'],
                'sampling_seed': entry['sampling_seed'],
                'points': entry['points'],
                'examples': metrics.get('examples'),
                'source': label,
            }
        summary = evaluate_module.robustness_summary(primary_recall, configurations)
        summary['plan'] = plan
        summary['primary_label'] = primary_label
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'freeze_sha256': freeze_state['sha256'],
            'sampling_robustness': summary['SAMPLING_ROBUSTNESS'],
            'max_bad_recall_drop_pp': summary['max_bad_recall_drop_pp'],
        }, detail=summary)
        state_module.atomic_write_json(
            self.layout.path('robustness', 'SAMPLING_ROBUSTNESS.json'), summary,
            with_digest=False)
        self.state.payload['robustness'] = summary
        self.state.save()

    def stage_report(self) -> None:
        name = S_REPORT
        selected = self.state.payload['selected_architecture']
        freeze_state = freeze_module.require_valid_freeze(self.layout.freeze_path)
        # The resume contract is a property of the code, not of this particular run, so the
        # report stage exercises it here instead of leaving the field NOT_RUN. It is
        # self-contained and writes only inside its own temporary directory.
        try:
            outcome = resume_test(self.layout.root)
            self.state.payload['tests'] = {
                **(self.state.payload.get('tests') or {}),
                'RESUME_TEST': outcome['RESUME_TEST'],
                'resume_test_checks': len(outcome['checks']),
            }
        except Exception as exc:  # noqa: BLE001 - a missing test must not fake a pass
            outcome = None
            self.state.payload['tests'] = {
                **(self.state.payload.get('tests') or {}),
                'RESUME_TEST': f'ERROR: {type(exc).__name__}',
            }
        self.state.save()
        self.log(f"    resume test: "
                 f"{self.state.payload['tests']['RESUME_TEST']} "
                 f"({self.state.payload['tests'].get('resume_test_checks', 0)} checks)")
        heldout_records = leakage_module.read_records(
            self.layout.path('evaluation', 'heldout'))
        leakage = leakage_module.heldout_protocol_audit(
            self.state.payload['heldout'], freeze_state,
            self.state.payload.get('parity'),
            self.state.payload['heldout'].get('manifest_check'),
            heldout_records)
        try:
            local_boundary = leakage_module.assert_local_challenge_untouched(HERE)
        except leakage_module.LeakageError as exc:
            raise PilotStop(exc.code, exc.detail) from exc
        self.state.payload['local_challenge_scan'] = {
            'verdict': local_boundary['verdict'],
            'scanned_files': local_boundary['scanned_files'],
            'hit_count': local_boundary['hit_count'],
            'io_hit_count': local_boundary['io_hit_count'],
            'LOCAL_8_BAD_CHALLENGE_ACCESSED': 'NO',
        }
        multi = self.state.payload['heldout'].get('multi_seed') or {}
        primary_record = (self.heldout_records.get(
            f'seed{self.state.payload["deployment_candidate_seed"]}_primary')
            or {})
        primary = primary_record.get('metrics') or {}
        primary_metrics = primary.get('primary_moderate_high') or {}
        primary_gate = evaluate_module.apply_primary_gate(primary_metrics)
        heldout_labels = sorted(name for name in heldout_records
                                if name.endswith('_primary'))
        heldout_complete = bool(heldout_labels) and all(
            heldout_records[name].get('held_out_complete') for name in heldout_labels)
        all_seeds_frozen = all(
            (self.state.stage(S_SEEDS).get('evidence') or {}).get('runs_digest')
            and (self.state.stage(S_THRESHOLDS).get('evidence') or {}).get(
                'thresholds_digest')
            for _ in [0])
        git = reports_module.git_state(HERE.parent.parent)
        verdict = reports_module.decide_verdict(
            stop_code=None,
            heldout_accessed=bool(self.state.payload['heldout'].get('accessed')),
            primary_gate=primary_gate,
            stability=multi,
            robustness=self.state.payload.get('robustness'),
            leakage=leakage,
            learnability=self.state.stage(S_LEARNABILITY).get('detail'),
            heldout_complete=heldout_complete)
        context = {
            'run_id': self.state.payload['run_id'],
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'base_accepted_head': BASE_ACCEPTED_HEAD,
            'implementation_head': git['implementation_head'],
            'gate_d_acceptance': 'PASS',
            'gate_d_generator_digest': self.state.payload[
                'accepted_gate_d_binding'].get('generator_digest'),
            'gate_d_protocol_digest': self.state.payload[
                'accepted_gate_d_binding'].get('protocol_digest'),
            'gate_d_core_changed': (
                'NO' if self.state.payload['accepted_gate_d_binding'].get(
                    'generator_digest') == EXPECTED_GENERATOR_DIGEST else 'YES'),
            'autopilot': 'PASS',
            'freeze': freeze_state['payload'],
            'heldout': {
                'accessed': True,
                'adapter_parity': self.state.payload.get('parity'),
                'candidates': len(heldout_module.held_out_view().eligible),
                'evaluated_candidates': primary_record.get(
                    'held_out_evaluated_candidates'),
                'complete': heldout_complete,
                'exclusions': {
                    'examples_declared': primary_record.get('declared_examples'),
                    'examples_scored': primary_record.get('examples'),
                    'excluded': primary_record.get('excluded_examples'),
                    'by_reason': primary_record.get('excluded_by_reason'),
                },
                'primary_moderate_high': primary_metrics,
                'per_defect_primary': primary.get('per_defect_primary'),
                'per_defect_all_severities': primary.get('per_defect_all_severities'),
                'per_severity': primary.get('per_severity'),
                'three_way': primary.get('three_way'),
                'overall': primary.get('overall'),
                'multi_seed': multi,
                'per_seed': {str(seed): (heldout_records.get(
                    f'seed{seed}_primary') or {}).get('primary_moderate_high')
                    for seed in gates.MODEL_SEEDS},
                'robustness': self.state.payload.get('robustness'),
                'leakage': leakage,
            },
            'comparison': {
                'pointnet': self.comparison.get('pointnet', 'COMPLETE'),
                'pointnet2_ssg': self.comparison.get('pointnet2_ssg', 'COMPLETE'),
                'evidence': freeze_state['payload'].get('model_comparison_evidence'),
            },
            'runs': {seed: self.runs.get(seed)
                     or self.state.model_run(run_key(selected, seed))
                     for seed in gates.MODEL_SEEDS},
            'validation': {'per_seed': freeze_state['payload'].get(
                'validation_metrics_per_seed')},
            'tests': self.state.payload.get('tests') or {},
            'resources': self.state.payload.get('environment') or {},
            'calibration': self.state.payload.get('calibration') or {},
            'verdict': verdict,
            'all_model_seeds_frozen_before_heldout': 'YES' if all_seeds_frozen else 'NO',
            'heldout_accessed_before_freeze': 'NO',
            'public_pilot_status': 'COMPLETE',
            'current_stage': STAGE_TITLES[name],
            'resume_ready': 'YES',
            'stop': None,
            'smoke_test': self.smoke_test,
        }
        report = reports_module.build_report(context)
        report['SMOKE_TEST'] = self.smoke_test
        report['FORMAL_EXPERIMENT'] = not self.smoke_test
        if self.smoke_test:
            report['SMOKE_TEST_NOTE'] = (
                'this run used bounded candidate counts, a reduced epoch budget and a '
                'bounded held-out subset; it validates the pipeline and is not a '
                'scientific result')
        paths = reports_module.write_report(self.layout.path('reports'), report)
        report['paths'] = paths
        self.state.payload['final'] = {
            'verdict': report['VERDICT'],
            'report_json': paths['json'],
            'report_md': paths['md'],
            'report_sha256': report['REPORT_SHA256'],
        }
        self.record_stage(name, 'COMPLETE', {
            'orchestrator_version': ORCHESTRATOR_VERSION,
            'verdict': report['VERDICT'],
            'report_sha256': report['REPORT_SHA256'],
        }, detail={'verdict': report['VERDICT'],
                   'report_json': paths['json'],
                   'report_md': paths['md']})
        self.log('')
        self.log('FINAL STATUS')
        self.log(f'  RUN ID   : {self.state.payload["run_id"]}')
        self.log(f'  VERDICT  : {report["VERDICT"]}')
        self.log(f'  REPORT   : {paths["json"]}')
        self.log(f'  MARKDOWN : {paths["md"]}')
        self.log(f'  STATUS   : {self.layout.current_status_path}')

    # --- driver -----------------------------------------------------------
    def run(self) -> dict:
        """Execute the state machine from the first incomplete stage."""
        stages = [
            (S_ENVIRONMENT, self.stage_environment),
            (S_GATE_D, self.stage_gate_d),
            (S_CALIBRATION, self.stage_calibration),
            (S_DATASET, self.stage_dataset),
            (S_PREWARM, lambda: self.prepare_training_data('pointnet')),
            (S_PARITY, self.stage_parity),
            (S_COMPARE_POINTNET, lambda: self.stage_compare('pointnet',
                                                            S_COMPARE_POINTNET)),
            (S_COMPARE_POINTNET2, lambda: self.stage_compare('pointnet2_ssg',
                                                             S_COMPARE_POINTNET2)),
            (S_LEARNABILITY, self.stage_learnability),
            (S_ARCHITECTURE, self.stage_architecture),
            (S_SEEDS, self.stage_seeds),
            (S_THRESHOLDS, self.stage_thresholds),
            (S_DEPLOYMENT, self.stage_deployment),
            (S_PROBE, self.stage_probe),
            (S_FREEZE, self.stage_freeze),
            (S_HELDOUT, self.stage_heldout),
            (S_AGGREGATE, self.stage_aggregate),
            (S_ROBUSTNESS, self.stage_robustness),
            (S_REPORT, self.stage_report),
        ]
        total = len(stages)
        self.recover_interrupted_stages()
        for index, (name, function) in enumerate(stages, 1):
            self.guard_no_tuning_after_unblind()
            status = self.state.stage(name).get('status', 'PENDING')
            already = status == 'COMPLETE'
            stage_banner(index, total, name, 'COMPLETE' if already else 'RUNNING')
            try:
                function()
            except PilotStop as stop:
                self.record_stage(name, 'FAILED', detail={'stop': stop.code})
                self.state.record_stop(stop.code, stop.detail)
                self.state.save()
                reports_module.write_stop_report(self.layout.path('reports'),
                                                 {'code': stop.code,
                                                  'detail': stop.detail},
                                                 self.state.payload)
                self.log('')
                self.log(f'STOP: {stop.code}')
                self.log(json.dumps(stop.detail, indent=2, default=str)[:4000])
                self.log(f'  STATE  : {self.layout.state_path}')
                self.log(f'  REPORT : {self.layout.path("reports", "EFG_STOP_REPORT.json")}')
                return {'status': 'STOPPED', 'stop_code': stop.code,
                        'detail': stop.detail, 'stage': name}
            except lock.TrainingLockedError as exc:
                code = 'STOP_ACCEPTANCE_INVALID'
                detail = {'message': str(exc)}
                self.record_stage(name, 'FAILED', detail={'stop': code})
                self.state.record_stop(code, detail)
                self.state.save()
                reports_module.write_stop_report(self.layout.path('reports'),
                                                 {'code': code, 'detail': detail},
                                                 self.state.payload)
                self.log(f'STOP: {code} {detail}')
                return {'status': 'STOPPED', 'stop_code': code, 'detail': detail,
                        'stage': name}
            except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
                # An unexpected exception is still a fail-closed stop: it is recorded with
                # its traceback, the run state says which stage died, a stop report is
                # written and nothing is retried. A bare traceback with no state is exactly
                # what an unattended run must never leave behind.
                code = 'STOP_UNEXPECTED_ERROR'
                detail = {'message': f'{type(exc).__name__}: {exc}',
                          'stage': name,
                          'traceback': traceback.format_exc()[-4000:]}
                self.record_stage(name, 'FAILED', detail={'stop': code})
                self.state.record_stop(code, detail)
                self.state.save()
                reports_module.write_stop_report(self.layout.path('reports'),
                                                 {'code': code, 'detail': detail},
                                                 self.state.payload)
                self.log('')
                self.log(f'STOP: {code}')
                self.log(detail['traceback'])
                self.log(f'  STATE  : {self.layout.state_path}')
                self.log(f'  REPORT : '
                         f'{self.layout.path("reports", "EFG_STOP_REPORT.json")}')
                return {'status': 'STOPPED', 'stop_code': code, 'detail': detail,
                        'stage': name}
            if not already:
                stage_banner(index, total, name, 'PASS')
        self.state.save()
        verdict = (self.state.payload.get('final') or {}).get('verdict')
        return {'status': 'COMPLETE', 'verdict': verdict,
                'report': (self.state.payload.get('final') or {}).get('report_json')}


def _new_run_id() -> str:
    import uuid
    stamp = time.strftime('%Y%m%dT%H%M%S', time.localtime())
    return f'public_pilot_v1_{stamp}_{uuid.uuid4().hex[:8]}'


def refresh_report(run_root: Path | str = DEFAULT_RUN_ROOT) -> dict:
    """Regenerate the final report from the frozen artifacts of a finished run.

    The report is a pure function of the artifacts the run already wrote: the experiment
    freeze, the per-seed threshold files, the held-out records and the state. Rebuilding it
    cannot change a metric, because every number is read back from those artifacts — and the
    held-out scores the metrics came from are now stored beside each record, so the
    operating point can be audited without re-running the evaluation.

    This exists so a reporting gap (a missing field, a clearer wording, a newer field) can
    be closed without touching an experiment that has already been unblinded. It refuses to
    run on an unfinished run.
    """
    layout = state_module.RunLayout(run_root)
    state = state_module.AutopilotState.load(layout.state_path)
    if not layout.freeze_path.exists():
        raise state_module.StateError(
            f'no experiment freeze at {layout.freeze_path}; nothing to report')
    if not state.payload['heldout'].get('accessed'):
        raise state_module.StateError('held-out was never accessed; there is no result')
    pilot = Pilot.__new__(Pilot)
    pilot.layout = layout
    pilot.state = state
    pilot.verbose = True
    pilot.runs = {}
    pilot.heldout_records = {}
    pilot.comparison = {}
    pilot.validation_evidence = {}
    pilot.calibration = state.payload.get('calibration') or {}
    pilot.smoke_test = bool(state.payload.get('smoke_test'))
    pilot.provenance = None
    for seed in gates.MODEL_SEEDS:
        key = run_key(state.payload['selected_architecture'], seed)
        record = state.model_run(key)
        if record:
            pilot.runs[seed] = record
    pilot.state.payload['stages'][S_REPORT] = state_module.StageRecord.pending(S_REPORT)
    pilot.recover_interrupted_stages()
    # Every held-out record is read back from disk: the report is built from artifacts, and
    # the verdict must be recomputed from the same evidence the run produced.
    pilot.heldout_records = leakage_module.read_records(
        layout.path('evaluation', 'heldout'))
    if not pilot.heldout_records:
        raise state_module.StateError(
            f'no held-out records under {layout.path("evaluation", "heldout")}')
    pilot.stage_report()
    final = state.payload.get('final') or {}
    return {'status': 'REPORT_REFRESHED', 'verdict': final.get('verdict'),
            'report': final.get('report_json'), 'markdown': final.get('report_md'),
            'report_sha256': final.get('report_sha256')}


# --- dry run ---------------------------------------------------------------

def dry_run(run_root: Path | str = DEFAULT_RUN_ROOT, epochs: int = gates.EPOCHS,
            points: int = POINTS_PRIMARY, verbose: bool = True) -> dict:
    """The fully resolved plan, with every guard evaluated and no GPU work."""
    layout = state_module.RunLayout(run_root)
    binding = gate_d_binding_status()
    provenance = load_gate_d_provenance()
    held_view = heldout_module.held_out_view()
    integrity = heldout_module.assert_heldout_integrity(
        provenance, held_view, expected_candidates=len(provenance.held_out_eligible))
    fingerprint = resources_module.fingerprint(layout)
    import efg_specs as specs_module
    train_ids = list(provenance.candidates('train'))
    validation_ids = list(provenance.candidates('validation'))
    train_balance = specs_module.class_balance(provenance, train_ids, 'train',
                                               points=points, repeats=1)
    validation_balance = specs_module.class_balance(provenance, validation_ids,
                                                    'validation', points=points,
                                                    repeats=1)
    plan = {
        'orchestrator_version': ORCHESTRATOR_VERSION,
        'base_accepted_head': BASE_ACCEPTED_HEAD,
        'gate_d_binding': {
            'generator_version': binding['expected_generator_version'],
            'generator_digest': binding['runtime_generator_digest'],
            'protocol_digest': binding['runtime_protocol_digest'],
            'binding_ok': binding['binding_ok'],
            'training_allowed': binding['training_allowed'],
            'acceptance_marker': binding['acceptance_marker'],
            'frozen_file_digests': binding['frozen_file_digests'],
        },
        'environment': {
            'gpu': fingerprint['gpu'].get('device_name'),
            'vram_total_bytes': fingerprint['gpu'].get('vram_total_bytes'),
            'ram_total_bytes': fingerprint.get('ram_total_bytes'),
            'cpu_logical_cores': (fingerprint.get('cpu') or {}).get('logical_cores'),
            'torch': fingerprint['gpu'].get('torch'),
            'cuda_available': fingerprint['gpu'].get('cuda_available'),
            'driver': fingerprint['driver'].get('nvidia_smi'),
            'disks': {k: {'free_gib': v.get('free_gib'), 'path': v.get('path')}
                      for k, v in fingerprint['disks'].items()},
            'disk_guard_gib': resources_module.DISK_GUARD_BYTES / 1024**3,
        },
        'dataset': {
            'train_candidates': len(train_ids),
            'validation_candidates': len(validation_ids),
            'train_examples': train_balance['declared_examples'],
            'validation_examples': validation_balance['declared_examples'],
            'held_out_candidates_metadata_only': len(held_view.eligible),
            'provenance_digest': provenance.provenance_digest,
            'identity_manifest_digest': provenance.identity_manifest_digest,
            'held_out_integrity': integrity,
        },
        'training': {
            'models': list(gates.ARCHITECTURES),
            'model_seeds': list(gates.MODEL_SEEDS),
            'comparison_seed': gates.COMPARISON_SEED,
            'epochs': int(epochs),
            'batch': 'calibrated at S03 (probe 16, 8, 4, 2, 1); common batch for both '
                     'architectures',
            'points': int(points),
            'learning_rate': gates.LEARNING_RATE,
            'weight_decay': gates.WEIGHT_DECAY,
            'amp': 'enabled when CUDA',
            'early_stopping': {'patience': gates.EARLY_STOP_PATIENCE,
                               'min_delta': gates.EARLY_STOP_MIN_DELTA,
                               'monitor': 'validation BALANCED_ACCURACY only'},
            'max_train_candidates': None,
            'max_validation_candidates': None,
        },
        'threshold_rule': 'max BAD_RECALL subject to GOOD_FALSE_REVIEW <= '
                          f'{gates.GOOD_FALSE_REVIEW_BUDGET}, from validation only',
        'architecture_selection_rule': [
            'GOOD_FALSE_REVIEW <= 0.25', 'max BAD_RECALL', 'max BALANCED_ACCURACY',
            'pointnet preferred within 0.01 non-inferiority', 'lowest seed'],
        'sampling_robustness': gates.robustness_plan(),
        'uncertain_band': gates.UNCERTAIN_BAND,
        'public_pilot_gates': gates.public_pilot_gates(),
        'held_out_locked': 'YES',
        'local_challenge_locked': 'YES',
        'held_out_accessed': 'NO',
        'planned_output_root': str(layout.root),
        'stages': [{'index': index, 'name': name, 'title': STAGE_TITLES[name]}
                   for index, name in enumerate(STAGE_ORDER, 1)],
        'resume': {
            'state_path': str(layout.state_path),
            'current_status_path': str(layout.current_status_path),
            'existing_state': layout.state_path.exists(),
        },
    }
    if verbose:
        console('TreeLineWarning Tree QC Public Pilot — DRY RUN')
        console('')
        console(f"  Gate D generator digest : "
                f"{plan['gate_d_binding']['generator_digest']}")
        console(f"  Gate D protocol digest  : "
                f"{plan['gate_d_binding']['protocol_digest']}")
        console(f"  training_allowed        : "
                f"{plan['gate_d_binding']['training_allowed']}")
        console(f"  GPU                     : {plan['environment']['gpu']}")
        console(f"  VRAM                    : "
                f"{plan['environment']['vram_total_bytes']} bytes")
        console(f"  RAM                     : {plan['environment']['ram_total_bytes']} "
                f"bytes")
        console(f"  D free                  : "
                f"{plan['environment']['disks'].get('D', {}).get('free_gib')} GiB")
        console(f"  E free                  : "
                f"{plan['environment']['disks'].get('E', {}).get('free_gib')} GiB")
        console(f"  train candidates        : {plan['dataset']['train_candidates']} "
                f"({plan['dataset']['train_examples']} examples)")
        console(f"  validation candidates   : "
                f"{plan['dataset']['validation_candidates']} "
                f"({plan['dataset']['validation_examples']} examples)")
        console(f"  held-out candidates     : "
                f"{plan['dataset']['held_out_candidates_metadata_only']} "
                f"(metadata only; locked)")
        console(f"  models / seeds          : {plan['training']['models']} / "
                f"{plan['training']['model_seeds']}")
        console(f"  epochs / points         : {plan['training']['epochs']} / "
                f"{plan['training']['points']}")
        console(f"  sampling seeds          : "
                f"{plan['sampling_robustness']['sampling_seeds']}")
        console(f"  threshold rule          : {plan['threshold_rule']}")
        console(f"  early stopping          : "
                f"{plan['training']['early_stopping']}")
        console(f"  held-out locked         : {plan['held_out_locked']}")
        console(f"  local challenge locked  : {plan['local_challenge_locked']}")
        console(f"  output root             : {plan['planned_output_root']}")
    return plan


def status_report(run_root: Path | str = DEFAULT_RUN_ROOT) -> dict:
    layout = state_module.RunLayout(run_root)
    payload = state_module.read_json(layout.current_status_path, verify=False)
    state = state_module.read_json(layout.state_path, verify=False)
    if not payload:
        return {'run_root': str(layout.root), 'state': 'NO_RUN_FOUND',
                'detail': f'{layout.state_path} does not exist'}
    if state and state.get('stop', {}).get('code'):
        payload['status'] = 'STOPPED'
    elif (state or {}).get('final', {}).get('verdict'):
        payload['status'] = 'COMPLETE'
    else:
        payload['status'] = 'INCOMPLETE_OR_RUNNING'
    payload['run_root'] = str(layout.root)
    return payload


# --- resume test -----------------------------------------------------------

def resume_test(run_root: Path | str = DEFAULT_RUN_ROOT) -> dict:
    """Prove the resume, staleness and freeze guards behave as declared.

    This exercises the persistence layer directly with synthetic records, so it costs no
    GPU time and can run before the real pilot. It is the ``RESUME_TEST`` field of the
    final report.
    """
    import tempfile
    checks = []
    with tempfile.TemporaryDirectory(prefix='efg_resume_test_') as temporary:
        root = Path(temporary)
        layout = state_module.RunLayout(root).ensure()
        state = state_module.AutopilotState.load(layout.state_path)
        state.ensure_run_identity('resume-test', ORCHESTRATOR_VERSION)
        state.set_stage(S_ENVIRONMENT, 'COMPLETE', {'orchestrator_version':
                                                    ORCHESTRATOR_VERSION,
                                                    'environment_digest': 'abc'})
        state.save()
        reloaded = state_module.AutopilotState.load(layout.state_path)
        checks.append({
            'name': 'completed stage survives a restart and is not rerun',
            'passed': state_module.stage_is_complete(
                reloaded.stage(S_ENVIRONMENT),
                {'orchestrator_version': ORCHESTRATOR_VERSION,
                 'environment_digest': 'abc'})[0]})
        checks.append({
            'name': 'a changed digest makes the stage STALE',
            'passed': not state_module.stage_is_complete(
                reloaded.stage(S_ENVIRONMENT),
                {'environment_digest': 'different'})[0]})
        checks.append({
            'name': 'a failed stage is not treated as complete',
            'passed': not state_module.stage_is_complete(
                {'status': 'FAILED', 'evidence': {'a': 1}}, {'a': 1})[0]})
        state.set_stage(S_SEEDS, 'COMPLETE', {'orchestrator_version':
                                              ORCHESTRATOR_VERSION})
        state.save()
        restored = state_module.AutopilotState.load(layout.state_path)
        order = [name for name in STAGE_ORDER
                 if restored.stage(name).get('status') == 'COMPLETE']
        checks.append({
            'name': 'the first incomplete stage is where the run resumes',
            'passed': order == [S_ENVIRONMENT, S_SEEDS]})

        target = root / 'state' / 'corrupt.json'
        target.write_text('{"state_digest": "deadbeef", "value": 1}', encoding='utf-8')
        try:
            state_module.read_json(target)
            checks.append({'name': 'a corrupt state file is refused', 'passed': False})
        except state_module.StateError:
            checks.append({'name': 'a corrupt state file is refused', 'passed': True})

        payload = {'run': 'x', 'checkpoint_sha256': 'aaaa'}
        state_module.atomic_write_json(root / 'record.json', payload, with_digest=False)
        checkpointer = root / 'best.pt'
        checkpointer.write_bytes(b'not-a-real-checkpoint')
        digest = state_module.digest_file(checkpointer)
        checks.append({
            'name': 'a matching checkpoint digest verifies',
            'passed': digest == state_module.digest_file(checkpointer)})
        checkpointer.write_bytes(b'tampered')
        checks.append({
            'name': 'a changed checkpoint digest is detected',
            'passed': digest != state_module.digest_file(checkpointer)})

        lock_path = root / 'AUTOPILOT.lock'
        with state_module.RunLock(lock_path):
            try:
                state_module.RunLock(lock_path).acquire().release()
                checks.append({'name': 'a second writer is refused', 'passed': False})
            except state_module.StateError:
                checks.append({'name': 'a second writer is refused', 'passed': True})
        checks.append({'name': 'the lock is released on exit',
                       'passed': not lock_path.exists()})

        empty_freeze = freeze_module.freeze_state(root / 'freeze.json')
        checks.append({'name': 'held-out refuses a missing freeze',
                       'passed': empty_freeze['valid'] is False})
        broken = root / 'freeze.json'
        state_module.atomic_write_json(broken, {
            'freeze_version': freeze_module.FREEZE_VERSION,
            'gate_d_binding': {}, 'selected_architecture': 'pointnet',
            'model_seeds': [101], 'runs': {}, 'frozen_thresholds': {},
            'deployment_candidate_seed': 101, 'points_primary': 4096,
            'points_robustness': 2048, 'robustness_plan': {},
            'public_pilot_metric_gates': {}, 'held_out_evaluated': False,
            'freeze_sha256': '0000'}, with_digest=False)
        checks.append({
            'name': 'held-out refuses a freeze with a bad digest',
            'passed': freeze_module.freeze_state(broken)['valid'] is False})
    failed = [check['name'] for check in checks if not check['passed']]
    return {'RESUME_TEST': 'PASS' if not failed else 'FAIL', 'checks': checks,
            'failed': failed, 'run_root': str(run_root)}


# --- CLI -------------------------------------------------------------------

SMOKE_TEST_DEFAULTS = {'epochs': 3, 'max_comparison_candidates': 6,
                       'max_heldout_candidates': 6}


def learnability_limits(smoke_test: bool = False) -> dict:
    """The validation learnability gates.

    The formal run uses the declared floors. A smoke test must be able to exercise the
    *later* stages, which a three-epoch bounded rehearsal cannot reach honestly, so it
    relaxes only this early-stop check and records that it did. Loosening the learnability
    gate cannot make the pilot succeed: every scientific gate after it — the experiment
    freeze, the held-out primary targets, the stability and robustness checks — is unchanged
    and is decided before the report exists.
    """
    limits = dict(gates.LEARNABILITY_GATE)
    if smoke_test:
        limits['smoke_test_relaxed'] = True
        limits['declared'] = dict(gates.LEARNABILITY_GATE)
        limits['BALANCED_ACCURACY_min'] = 0.0
        limits['BAD_RECALL_min'] = 0.0
    return limits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Gate D unlocked + E/F/G merged one-click Public Pilot')
    parser.add_argument('--auto', action='store_true',
                        help='run the whole state machine, resuming where it stopped')
    parser.add_argument('--smoke-test', action='store_true',
                        help='bounded end-to-end rehearsal on few candidates; never a '
                             'scientific result')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the resolved plan without touching the GPU')
    parser.add_argument('--status', action='store_true', help='read-only progress report')
    parser.add_argument('--resume-test', action='store_true',
                        help='prove resume/staleness/guard behaviour')
    parser.add_argument('--refresh-report', action='store_true',
                        help='regenerate the final report of a finished run from its '
                             'frozen artifacts; never touches the experiment')
    parser.add_argument('--run-root', default=str(DEFAULT_RUN_ROOT))
    parser.add_argument('--epochs', type=int, default=gates.EPOCHS)
    parser.add_argument('--points', type=int, default=POINTS_PRIMARY,
                        choices=[POINTS_PRIMARY, gates.POINTS_ROBUSTNESS])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed-base', type=int, default=0)
    parser.add_argument('--max-comparison-candidates', type=int, default=None,
                        help='smoke-test only; the real pilot must leave this unset')
    parser.add_argument('--max-heldout-candidates', type=int, default=None,
                        help='smoke-test only; the real pilot must leave this unset')
    parser.add_argument('--no-cache', action='store_true',
                        help='disable the example feature cache')
    parser.add_argument('--no-calibration-reuse', action='store_true')
    parser.add_argument('--json', action='store_true', help='machine-readable output')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        plan = dry_run(args.run_root, args.epochs, args.points,
                       verbose=not args.json)
        if args.json:
            print(json.dumps(plan, indent=2, default=str))
        return 0
    if args.resume_test:
        result = resume_test(args.run_root)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result['RESUME_TEST'] == 'PASS' else 1
    if args.status:
        print(json.dumps(status_report(args.run_root), indent=2, default=str))
        return 0
    if args.refresh_report:
        try:
            with state_module.RunLock(state_module.RunLayout(args.run_root).lock_path):
                result = refresh_report(args.run_root)
        except state_module.StateError as exc:
            print(f'REFUSED: {exc}', file=sys.stderr)
            return 4
        print(json.dumps(result, indent=2, default=str))
        return 0
    bounded = (args.max_comparison_candidates is not None
               or args.max_heldout_candidates is not None
               or args.epochs != gates.EPOCHS
               or args.points != POINTS_PRIMARY
               or args.repeats != 1
               or args.seed_base != 0)
    if args.smoke_test:
        if args.epochs == gates.EPOCHS:
            args.epochs = SMOKE_TEST_DEFAULTS['epochs']
        if args.max_comparison_candidates is None:
            args.max_comparison_candidates = SMOKE_TEST_DEFAULTS[
                'max_comparison_candidates']
        if args.max_heldout_candidates is None:
            args.max_heldout_candidates = SMOKE_TEST_DEFAULTS[
                'max_heldout_candidates']
    elif bounded:
        # A bounded run can never masquerade as the formal experiment, so the formal
        # entry point refuses bounded options outright.
        print('REFUSED: bounded candidate counts, epochs, points, repeats or seed base '
              'are smoke-test options only. The formal Public Pilot uses the complete '
              'frozen train and validation sets at 4096 points and 30 epochs. '
              'Use --smoke-test for a bounded rehearsal.', file=sys.stderr)
        return 2
    if not args.auto:
        build_parser().print_help()
        return 2

    layout = state_module.RunLayout(args.run_root).ensure()
    console('TreeLineWarning Tree QC Public Pilot'
            + ('   [SMOKE TEST - not a scientific result]' if args.smoke_test else ''))
    console(f'  run root : {layout.root}')
    console(f'  state    : {layout.state_path}')
    console('')
    try:
        with state_module.RunLock(layout.lock_path):
            pilot = Pilot(args.run_root, epochs=args.epochs, points=args.points,
                          repeats=args.repeats, seed_base=args.seed_base,
                          comparison_candidates=args.max_comparison_candidates,
                          use_cache=not args.no_cache,
                          reuse_calibration=not args.no_calibration_reuse,
                          max_heldout_candidates=args.max_heldout_candidates,
                          smoke_test=bool(args.smoke_test))
            result = pilot.run()
    except state_module.StateError as exc:
        console('')
        console(f'STOP: {exc}')
        return 4
    console('')
    console(f"STATUS   : {result['status']}")
    if result.get('stop_code'):
        console(f"STOP CODE: {result['stop_code']}")
        console(f"STAGE    : {result.get('stage')}")
        return 5
    console(f"VERDICT  : {result.get('verdict')}")
    console(f"REPORT   : {result.get('report')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
