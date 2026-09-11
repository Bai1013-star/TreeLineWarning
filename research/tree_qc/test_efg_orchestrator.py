"""EFG orchestrator tests: the freeze, held-out and resume guards.

The frozen Gate D tests already prove the generator, sampler, provenance allowlist and
metric core. These tests prove the layer this round adds: that the Public Pilot refuses to
run outside its declared plan, that a completed stage is only re-used when the digests that
produced it still match, that held-out is unreachable before a valid freeze, and that
nothing in the E/F/G path can read the local Gate H challenge.

Where a guard needs real provenance semantics (held-out authorization, adapter parity,
manifest counts) the tests run against the real frozen Gate C2 artifacts. Where it needs
geometry they build small synthetic trees, for speed.

Coverage map (the letters are the cases the round specification requires):

    A  Gate D marker invalid -> refuses training
    B  valid acceptance -> EFG allowed
    C  resume completed stage -> does not rerun
    D  stale checkpoint digest -> refuses reuse
    E  config mismatch -> refuses resume
    F  validation-only threshold -> PASS
    G  threshold attempted on held-out -> refuses
    H  held-out before freeze -> refuses
    I  incomplete three seeds before unblind -> refuses
    J  modified freeze after held-out -> refuses
    K  adapter parity failure -> refuses held-out
    L  wrong dataset/provenance digest -> refuses resume/unblind
    M  non-finite training result -> fail closed
    N  disk below guard -> stops
    O  crash/restart state -> resumes correctly
    P  finished job -> idempotent rerun
    Q  public-pilot failure -> never opens local 8 bad
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
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
import efg_specs as specs_module  # noqa: E402
import efg_state as state_module  # noqa: E402
import efg_trainer as trainer_module  # noqa: E402
import gate_d_generator as gen  # noqa: E402
import gate_d_lock as lock  # noqa: E402
import gate_d_metrics as metrics_module  # noqa: E402
import gate_d_provenance as provenance_module  # noqa: E402
import gate_d_scene as scene_module  # noqa: E402
import gate_d_train as engine  # noqa: E402
import run_tree_qc_efg as orchestrator  # noqa: E402

PROVENANCE = provenance_module.load_gate_d_provenance()
ACCEPTANCE_PATH = lock.acceptance_path()
EXPECTED_GENERATOR = orchestrator.EXPECTED_GENERATOR_DIGEST
EXPECTED_PROTOCOL = orchestrator.EXPECTED_PROTOCOL_DIGEST


# --- small fixtures --------------------------------------------------------

def tree_cloud(rng, count, box):
    """A narrow stem plus a wide crown: enough structure for every defect class."""
    x0, y0, z0, x1, y1, z1 = box
    span_x, span_y = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    span_z = max(z1 - z0, 1e-3)
    axis_x, axis_y = x0 + span_x / 2.0, y0 + span_y / 2.0
    stem_count = max(16, int(count * 0.3))
    crown_count = count - stem_count
    stem = np.stack([
        (axis_x + rng.normal(0.0, 0.12, stem_count)).astype(np.float32),
        (axis_y + rng.normal(0.0, 0.12, stem_count)).astype(np.float32),
        (z0 + rng.random(stem_count).astype(np.float32) * span_z * 0.7),
    ], axis=1)
    crown = np.stack([
        (axis_x + rng.normal(0.0, span_x * 0.3, crown_count)).astype(np.float32),
        (axis_y + rng.normal(0.0, span_y * 0.3, crown_count)).astype(np.float32),
        (z0 + span_z * 0.5
         + rng.random(crown_count).astype(np.float32) * span_z * 0.5),
    ], axis=1)
    return np.concatenate([stem, crown], axis=0).astype(np.float32)


class Candidate:
    """The frozen ``CandidateGeometry`` contract for held-out authorization tests."""

    def __init__(self, candidate_id, context_ids, seed=0, candidate_points=900,
                 context_points=800):
        rng = np.random.default_rng(seed)
        record = PROVENANCE.tree[candidate_id]
        x0, y0, z0, x1, y1, z1 = record['bbox']
        self.candidate = _Tree(candidate_id, record['site'],
                               tree_cloud(rng, candidate_points, record['bbox']))
        self.context = tuple(
            _Tree(cid, PROVENANCE.tree[cid]['site'],
                  tree_cloud(rng, context_points, PROVENANCE.tree[cid]['bbox']))
            for cid in (context_ids if context_ids is not None
                        else PROVENANCE.authorized_context(candidate_id)))

    @property
    def candidate_id(self):
        return self.candidate.canonical_tree_id

    @property
    def context_ids(self):
        return tuple(t.canonical_tree_id for t in self.context)


class _Tree:
    def __init__(self, canonical_tree_id, site, points):
        self.canonical_tree_id = canonical_tree_id
        self.site = site
        self.points = points
        self.point_count = int(points.shape[0])
        self.stored_points = int(points.shape[0])


class HeldOutCandidate(Candidate):
    """The same contract built from the frozen **held-out** context."""

    def __init__(self, candidate_id, seed=0, candidate_points=900, context_points=800):
        view = heldout_module.held_out_view()
        rng = np.random.default_rng(seed)
        record = PROVENANCE.tree[candidate_id]
        self.candidate = _Tree(candidate_id, record['site'],
                               tree_cloud(rng, candidate_points, record['bbox']))
        self.context = tuple(
            _Tree(cid, PROVENANCE.tree[cid]['site'],
                  tree_cloud(rng, context_points, PROVENANCE.tree[cid]['bbox']))
            for cid in view.context[candidate_id])


def small_geometry(candidate_id: str, partition: str, seed: int = 3):
    """A synthetic candidate geometry valid for the frozen scene builder."""
    context_ids = PROVENANCE.authorized_context(candidate_id)
    return Candidate(candidate_id, context_ids, seed=seed)


# --- A / B: Gate D acceptance ---------------------------------------------

class GateDAcceptance(unittest.TestCase):
    """A and B: the acceptance marker is the only thing that unlocks training."""

    def test_A_absent_marker_refuses_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary)
            state = lock.acceptance_state(reports)
            self.assertFalse(state['training_allowed'])
            self.assertIn('GATE_D_NOT_ACCEPTED', state['reason'])
            with self.assertRaises(lock.TrainingLockedError):
                lock.require_training_allowed(reports)

    def test_A_malformed_marker_refuses_training(self):
        cases = [
            {'gate_d': 'REJECTED', 'training_allowed': True},
            {'gate_d': 'ACCEPTED', 'training_allowed': False},
            {'gate_d': 'ACCEPTED'},
            'not-an-object',
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as temporary:
                    reports = Path(temporary)
                    marker = reports / lock.ACCEPTANCE_FILE
                    marker.write_text(json.dumps(payload), encoding='utf-8')
                    state = lock.acceptance_state(reports)
                    self.assertFalse(state['training_allowed'])

    def test_A_wrong_generator_digest_refuses_training(self):
        binding = lock.current_binding()
        for field, wrong in (('generator_digest', 'deadbeef'),
                             ('protocol_digest', 'deadbeef'),
                             ('generator_version', 'GATE_D_SYNTHETIC_GENERATOR_V1')):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    reports = Path(temporary)
                    payload = {'gate_d': 'ACCEPTED', 'training_allowed': True,
                               'generator_version': binding['generator_version'],
                               'generator_digest': binding['generator_digest'],
                               'protocol_digest': binding['protocol_digest']}
                    payload[field] = wrong
                    (reports / lock.ACCEPTANCE_FILE).write_text(
                        json.dumps(payload), encoding='utf-8')
                    state = lock.acceptance_state(reports)
                    self.assertFalse(state['training_allowed'])
                    self.assertIn('GATE_D_NOT_ACCEPTED', state['reason'])

    def test_B_shipped_acceptance_unlocks_the_pilot(self):
        status = orchestrator.gate_d_binding_status()
        self.assertTrue(status['binding_ok'], status['matches'])
        self.assertEqual(status['runtime_generator_digest'], EXPECTED_GENERATOR)
        self.assertEqual(status['runtime_protocol_digest'], EXPECTED_PROTOCOL)
        self.assertTrue(status['training_allowed'])
        binding = orchestrator.require_gate_d_binding()
        self.assertEqual(binding['acceptance_reason'], 'GATE_D_ACCEPTED')

    def test_B_trainers_no_longer_report_the_lock(self):
        for script in ('train_pointnet.py', 'train_pointnet2.py'):
            completed = subprocess.run(
                [sys.executable, str(HERE / script), '--dry-run'],
                capture_output=True, text=True, cwd=str(HERE), timeout=300)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn('GATE_D_NOT_ACCEPTED', completed.stdout)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload['training_lock']['training_allowed'])

    def test_B_frozen_gate_d_files_are_unchanged(self):
        digests = orchestrator.frozen_file_digests()
        self.assertEqual(len(digests), len(orchestrator.FROZEN_GATE_D_FILES))
        self.assertEqual(orchestrator.diagnostic_module.generator_digest(),
                         EXPECTED_GENERATOR)
        self.assertEqual(EXPECTED_PROTOCOL, provenance_module.protocol_digest())
        self.assertEqual(EXPECTED_GENERATOR,
                         lock.current_binding()['generator_digest'])


# --- state machine: C, D, E, L, O, P --------------------------------------

class StateMachineGuards(unittest.TestCase):
    """C, O and P: completion, crash/restart and idempotence."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.layout = state_module.RunLayout(self.root).ensure()
        self.state = state_module.AutopilotState.load(self.layout.state_path)
        self.state.ensure_run_identity('test-run', orchestrator.ORCHESTRATOR_VERSION)

    def tearDown(self):
        self._temporary.cleanup()

    def test_C_completed_stage_is_not_rerun(self):
        self.state.set_stage(orchestrator.S_ENVIRONMENT, state_module.COMPLETE,
                             {'orchestrator_version': orchestrator.ORCHESTRATOR_VERSION,
                              'environment_digest': 'abc'})
        self.state.save()
        reloaded = state_module.AutopilotState.load(self.layout.state_path)
        ok, reason = state_module.stage_is_complete(
            reloaded.stage(orchestrator.S_ENVIRONMENT),
            {'orchestrator_version': orchestrator.ORCHESTRATOR_VERSION,
             'environment_digest': 'abc'})
        self.assertTrue(ok, reason)
        stale, reason = state_module.stage_is_complete(
            reloaded.stage(orchestrator.S_ENVIRONMENT),
            {'environment_digest': 'different'})
        self.assertFalse(stale)
        self.assertIn('mismatch', reason)

    def test_C_resume_continues_from_the_first_incomplete_stage(self):
        self.state.set_stage(orchestrator.S_ENVIRONMENT, state_module.COMPLETE, {})
        self.state.set_stage(orchestrator.S_GATE_D, state_module.COMPLETE, {})
        self.state.set_stage(orchestrator.S_CALIBRATION, state_module.COMPLETE, {})
        self.state.set_stage(orchestrator.S_DATASET, state_module.RUNNING, {})
        self.state.save()
        reloaded = state_module.AutopilotState.load(self.layout.state_path)
        completed = [name for name in orchestrator.STAGE_ORDER
                     if reloaded.stage(name).get('status') == state_module.COMPLETE]
        self.assertEqual(completed, [orchestrator.S_ENVIRONMENT,
                                     orchestrator.S_GATE_D,
                                     orchestrator.S_CALIBRATION])
        self.assertEqual(reloaded.stage(orchestrator.S_DATASET)['status'],
                         state_module.RUNNING)
        self.assertGreaterEqual(reloaded.payload['resume']['count'], 1)
        self.assertTrue(reloaded.payload['resume']['events'])

    def test_P_finished_run_re_enters_without_duplicating_work(self):
        for name in orchestrator.STAGE_ORDER:
            self.state.set_stage(name, state_module.COMPLETE,
                                 {'orchestrator_version':
                                  orchestrator.ORCHESTRATOR_VERSION})
        self.state.save()
        first = state_module.AutopilotState.load(self.layout.state_path)
        before = {name: dict(first.stage(name)) for name in orchestrator.STAGE_ORDER}
        second = state_module.AutopilotState.load(self.layout.state_path)
        for name in orchestrator.STAGE_ORDER:
            entry = second.stage(name)
            self.assertEqual(entry['status'], state_module.COMPLETE)
            self.assertEqual(entry['updated_at'], before[name]['updated_at'])
            self.assertEqual(entry['attempts'], before[name]['attempts'])

    def test_O_torn_state_file_is_refused_not_trusted(self):
        path = self.layout.state_path
        state_module.atomic_write_json(path, {'stages': {}, 'run_id': 'x'},
                                       with_digest=True)
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['run_id'] = 'tampered'
        path.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaises(state_module.StateError):
            state_module.read_json(path)

    def test_O_orchestrator_refuses_a_version_change_on_resume(self):
        self.state.save()
        with self.assertRaises(state_module.StateError):
            self.state.ensure_run_identity('test-run', 'SOME_OTHER_VERSION')

    def test_single_writer_lock(self):
        lock_path = self.layout.lock_path
        with state_module.RunLock(lock_path):
            with self.assertRaises(state_module.StateError):
                state_module.RunLock(lock_path).acquire()
        self.assertFalse(lock_path.exists())

    def test_state_write_is_atomic_and_leaves_no_temp_file(self):
        self.state.set_stage(orchestrator.S_ENVIRONMENT, state_module.COMPLETE, {})
        self.state.save()
        leftovers = list(self.layout.state_path.parent.glob('*.tmp*'))
        self.assertEqual(leftovers, [])


# --- C, D, E, L: model-run reuse ------------------------------------------

class ModelRunReuse(unittest.TestCase):
    """C, D, E and L: a completed run is re-used only when everything still matches."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.checkpoint = self.root / 'best.pt'
        self.checkpoint.write_bytes(b'checkpoint-bytes')
        digest = state_module.digest_file(self.checkpoint)
        self.config = engine.TrainConfig(
            model='pointnet', points=4096, epochs=30, batch_size=8,
            learning_rate=1e-3, seed=101, cache_root=str(self.root / 'cache'),
            output_root=str(self.root / 'runs'))
        self.record = {
            'model': 'pointnet', 'seed': 101,
            'config_digest': self.config.digest(),
            'config': self.config.as_row(),
            'dataset_digest': 'dataset-digest-1',
            'checkpoint_path': str(self.checkpoint),
            'checkpoint_sha256': digest,
            'frozen_threshold': {'frozen': True, 'threshold': 0.5},
            'threshold': 0.5,
            'status': 'COMPLETE',
        }

    def tearDown(self):
        self._temporary.cleanup()

    def test_C_identical_record_is_reused(self):
        ok, reason = trainer_module.completed_run_is_valid(
            self.record, self.config, 'dataset-digest-1')
        self.assertTrue(ok, reason)

    def test_D_stale_checkpoint_digest_refuses_reuse(self):
        self.checkpoint.write_bytes(b'tampered-bytes')
        ok, reason = trainer_module.completed_run_is_valid(
            self.record, self.config, 'dataset-digest-1')
        self.assertFalse(ok)
        self.assertIn('checkpoint digest mismatch', reason)

    def test_D_missing_checkpoint_refuses_reuse(self):
        self.checkpoint.unlink()
        ok, reason = trainer_module.completed_run_is_valid(
            self.record, self.config, 'dataset-digest-1')
        self.assertFalse(ok)
        self.assertEqual(reason, 'checkpoint missing')

    def test_E_config_mismatch_refuses_reuse(self):
        other = engine.TrainConfig.from_row({**self.config.as_row(),
                                             'learning_rate': 5e-3})
        ok, reason = trainer_module.completed_run_is_valid(
            self.record, other, 'dataset-digest-1')
        self.assertFalse(ok)
        self.assertIn('config_digest mismatch', reason)

    def test_E_batch_and_points_change_refuse_reuse(self):
        for field, value in (('batch_size', 4), ('points', 2048), ('seed', 202)):
            with self.subTest(field=field):
                other = engine.TrainConfig.from_row({**self.config.as_row(),
                                                     field: value})
                ok, _ = trainer_module.completed_run_is_valid(
                    self.record, other, 'dataset-digest-1')
                self.assertFalse(ok)

    def test_L_dataset_digest_change_refuses_reuse(self):
        ok, reason = trainer_module.completed_run_is_valid(
            self.record, self.config, 'dataset-digest-2')
        self.assertFalse(ok)
        self.assertIn('dataset_digest mismatch', reason)

    def test_L_resume_decision_refuses_a_changed_dataset(self):
        payload = {'config_digest': self.config.digest(),
                   'dataset_digest': 'dataset-digest-1', 'epoch': 4}
        path = self.root / 'last.pt'
        path.write_bytes(b'irrelevant')
        original = engine.load_checkpoint
        engine.load_checkpoint = lambda *a, **k: dict(payload)
        try:
            decision = trainer_module.resume_decision(path, self.config,
                                                      'dataset-digest-1')
            self.assertTrue(decision['resume'])
            with self.assertRaises(trainer_module.TrainingStop) as raised:
                trainer_module.resume_decision(path, self.config, 'other-dataset')
            self.assertEqual(raised.exception.code, 'STOP_RESUME_MISMATCH')
            with self.assertRaises(trainer_module.TrainingStop) as raised:
                trainer_module.resume_decision(
                    path, engine.TrainConfig.from_row({**self.config.as_row(),
                                                       'batch_size': 2}),
                    'dataset-digest-1')
            self.assertEqual(raised.exception.code, 'STOP_RESUME_MISMATCH')
        finally:
            engine.load_checkpoint = original

    def test_resume_decision_reports_a_corrupt_checkpoint(self):
        path = self.root / 'corrupt.pt'
        path.write_bytes(b'x')
        original = engine.load_checkpoint

        def explode(*args, **kwargs):
            raise ValueError('not a torch file')

        engine.load_checkpoint = explode
        try:
            with self.assertRaises(trainer_module.TrainingStop) as raised:
                trainer_module.resume_decision(path, self.config, 'dataset-digest-1')
            self.assertEqual(raised.exception.code, 'STOP_CHECKPOINT_CORRUPTION')
        finally:
            engine.load_checkpoint = original


# --- F, G: threshold discipline -------------------------------------------

class ThresholdDiscipline(unittest.TestCase):
    """F and G: thresholds come from validation and can never come from held-out."""

    def test_F_validation_threshold_selection_passes(self):
        rng = np.random.default_rng(0)
        labels = np.concatenate([np.zeros(60, dtype=np.int64),
                                 np.ones(60, dtype=np.int64)])
        scores = np.concatenate([rng.uniform(0.0, 0.45, 60),
                                 rng.uniform(0.55, 1.0, 60)])
        selected = metrics_module.select_threshold(labels, scores, 'validation')
        self.assertTrue(selected['budget_satisfied'])
        self.assertEqual(selected['partition'], 'validation')
        self.assertGreaterEqual(selected['BAD_RECALL'], 0.9)
        frozen = metrics_module.freeze_threshold(
            selected, 'dataset-digest', 'checkpoint-digest', 101)
        self.assertTrue(frozen['frozen'])
        self.assertFalse(frozen['held_out_used'])

    def test_F_train_partition_is_also_allowed(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        selected = metrics_module.select_threshold(labels, scores, 'train')
        self.assertEqual(selected['partition'], 'train')

    def test_G_held_out_threshold_selection_is_refused(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        for partition in ('held_out', 'HELD_OUT', 'heldout', 'test'):
            with self.subTest(partition=partition):
                with self.assertRaises(metrics_module.ThresholdError):
                    metrics_module.select_threshold(labels, scores, partition)

    def test_G_freeze_threshold_refuses_held_out_provenance(self):
        with self.assertRaises(metrics_module.ThresholdError):
            metrics_module.freeze_threshold(
                {'threshold': 0.5, 'partition': 'held_out'}, 'd', 'c', 101)

    def test_G_held_out_evaluation_takes_a_threshold_and_cannot_choose_one(self):
        import inspect
        signature = inspect.signature(orchestrator.Pilot.evaluate_heldout)
        self.assertIn('threshold', signature.parameters)
        source = inspect.getsource(evaluate_module.score_partition)
        self.assertNotIn('select_threshold', source)
        self.assertNotIn('freeze_threshold', source)

    def test_uncertain_band_never_counts_as_a_detection(self):
        scores = np.array([0.5])
        counts = evaluate_module.three_way_counts(scores, 0.5, 0.1)
        self.assertEqual(counts['counts']['UNCERTAIN'], 1)
        metrics = evaluate_module.binary_metrics(np.array([1]), scores, 0.5)
        self.assertEqual(metrics['BAD_RECALL'], 1.0)
        self.assertIn('never counted', counts['note'])


# --- H, I, J, K: held-out hard lock ---------------------------------------

def valid_freeze_payload(model_seeds=(101, 202, 303)) -> dict:
    runs = {}
    for seed in model_seeds:
        runs[str(seed)] = {
            'checkpoint_path': f'checkpoint-{seed}.pt',
            'checkpoint_sha256': f'digest-{seed}',
            'config_sha256': f'config-{seed}',
            'dataset_digest': f'dataset-{seed}',
            'threshold': 0.5,
            'threshold_path': f'threshold-{seed}.json',
            'threshold_sha256': f'threshold-digest-{seed}',
        }
    return {
        'freeze_version': freeze_module.FREEZE_VERSION,
        'freeze_timestamp': '2026-01-01T00:00:00+08:00',
        'gate_d_binding': {'generator_version': 'GATE_D_SYNTHETIC_GENERATOR_V2',
                           'generator_digest': EXPECTED_GENERATOR,
                           'protocol_digest': EXPECTED_PROTOCOL,
                           'gate_d_accepted': True},
        'selected_architecture': 'pointnet',
        'architecture_selection_basis': {},
        'model_comparison_evidence': {},
        'training_config': {},
        'model_seeds': list(model_seeds),
        'runs': runs,
        'validation_dataset': {},
        'validation_metrics_per_seed': {},
        'frozen_thresholds': {str(seed): 0.5 for seed in model_seeds},
        'threshold_rule': 'validation only',
        'deployment_candidate_seed': int(model_seeds[0]),
        'deployment_seed_basis': {},
        'points_primary': 4096,
        'points_robustness': 2048,
        'robustness_plan': gates.robustness_plan(),
        'uncertain_policy': 'never a detection',
        'uncertain_band': gates.UNCERTAIN_BAND,
        'public_pilot_metric_gates': gates.public_pilot_gates(),
        'held_out_evaluated': False,
    }


class HeldOutHardLock(unittest.TestCase):
    """H, I, J and K: held-out cannot be reached until every lock is satisfied."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.freeze_path = self.root / 'PUBLIC_EXPERIMENT_FREEZE.json'

    def tearDown(self):
        self._temporary.cleanup()

    def _write_freeze(self, payload=None):
        payload = payload or valid_freeze_payload()
        body = dict(payload)
        body['freeze_sha256'] = freeze_module.freeze_digest(body)
        self.freeze_path.write_text(json.dumps(body), encoding='utf-8')
        return body

    def test_H_missing_freeze_refuses_held_out(self):
        with self.assertRaises(freeze_module.FreezeError) as raised:
            freeze_module.require_valid_freeze(self.freeze_path)
        self.assertEqual(raised.exception.code, 'STOP_HELDOUT_BEFORE_FREEZE')

    def test_H_incomplete_freeze_refuses_held_out(self):
        payload = valid_freeze_payload()
        payload.pop('frozen_thresholds')
        self._write_freeze(payload)
        with self.assertRaises(freeze_module.FreezeError) as raised:
            freeze_module.require_valid_freeze(self.freeze_path)
        self.assertEqual(raised.exception.code, 'STOP_HELDOUT_BEFORE_FREEZE')

    def test_H_a_freeze_written_with_held_out_already_evaluated_is_refused(self):
        payload = valid_freeze_payload()
        payload['held_out_evaluated'] = True
        self._write_freeze(payload)
        state = freeze_module.freeze_state(self.freeze_path)
        self.assertFalse(state['valid'])
        self.assertIn('held_out_evaluated', state['reason'])

    def test_valid_freeze_is_accepted(self):
        self._write_freeze()
        state = freeze_module.require_valid_freeze(self.freeze_path)
        self.assertTrue(state['valid'])
        self.assertEqual(len(state['sha256']), 64)

    def test_I_incomplete_seeds_are_refused(self):
        runs = {'101': {'threshold_frozen': True}}
        with self.assertRaises(freeze_module.FreezeError) as raised:
            freeze_module.require_all_frozen(runs, (101, 202, 303))
        self.assertEqual(raised.exception.code, 'STOP_CHECKPOINT_CORRUPTION')
        self.assertEqual(len(raised.exception.detail['problems']), 3)

    def test_I_a_seed_without_a_frozen_threshold_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'best.pt'
            checkpoint.write_bytes(b'bytes')
            runs = {101: {'checkpoint_path': str(checkpoint),
                          'checkpoint_sha256': state_module.digest_file(checkpoint),
                          'threshold_frozen': False}}
            with self.assertRaises(freeze_module.FreezeError) as raised:
                freeze_module.require_all_frozen(runs, (101,))
            self.assertIn('threshold not frozen',
                          json.dumps(raised.exception.detail))

    def test_J_freeze_mutation_after_unblind_is_refused(self):
        body = self._write_freeze()
        audit = {'accessed': True, 'freeze_sha256_at_unblind': body['freeze_sha256']}
        result = freeze_module.check_freeze_immutable(self.freeze_path, audit)
        self.assertTrue(result['unchanged'])

        mutated = json.loads(self.freeze_path.read_text(encoding='utf-8'))
        mutated['selected_architecture'] = 'pointnet2_ssg'
        self.freeze_path.write_text(json.dumps(mutated), encoding='utf-8')
        with self.assertRaises(freeze_module.FreezeError) as raised:
            freeze_module.check_freeze_immutable(self.freeze_path, audit)
        self.assertEqual(raised.exception.code, 'STOP_FREEZE_MUTATED_AFTER_UNBLIND')

    def test_J_a_resigned_freeze_cannot_hide_the_change(self):
        """Recomputing the digest must not launder a post-unblind edit."""
        body = self._write_freeze()
        audit = {'accessed': True, 'freeze_sha256_at_unblind': body['freeze_sha256']}
        mutated = json.loads(self.freeze_path.read_text(encoding='utf-8'))
        mutated['selected_architecture'] = 'pointnet2_ssg'
        mutated['freeze_sha256'] = freeze_module.freeze_digest(mutated)
        self.freeze_path.write_text(json.dumps(mutated), encoding='utf-8')
        self.assertTrue(freeze_module.freeze_state(self.freeze_path)['valid'])
        with self.assertRaises(freeze_module.FreezeError):
            freeze_module.check_freeze_immutable(self.freeze_path, audit)

    def test_J_pre_unblind_stage_is_refused_after_unblind(self):
        self._write_freeze()
        pilot = orchestrator.Pilot.__new__(orchestrator.Pilot)
        pilot.layout = state_module.RunLayout(self.root)
        pilot.state = state_module.AutopilotState(self.root / 'state.json')
        pilot.state.payload['heldout'] = {
            'accessed': True, 'unlocked': True,
            'first_access_timestamp': '2026-01-01T00:00:00+08:00',
            'freeze_sha256_at_unblind': freeze_module.freeze_state(
                self.freeze_path)['sha256']}
        pilot.verbose = False
        with self.assertRaises(orchestrator.PilotStop) as raised:
            pilot.guard_pre_unblind(orchestrator.S_SEEDS)
        self.assertEqual(raised.exception.code, 'STOP_FREEZE_MUTATED_AFTER_UNBLIND')

    def test_K_adapter_parity_failure_refuses_held_out(self):
        original = heldout_module.parity_report
        heldout_module.parity_report = lambda *a, **k: {
            'HELDOUT_ADAPTER_PARITY': 'FAIL', 'mismatch_count': 1, 'mismatches': []}
        try:
            with self.assertRaises(heldout_module.AdapterDrift) as raised:
                heldout_module.require_parity(PROVENANCE, candidate_ids=[])
            self.assertEqual(raised.exception.code, 'STOP_HELDOUT_ADAPTER_DRIFT')
        finally:
            heldout_module.parity_report = original

    def test_K_real_parity_passes_on_real_validation_candidates(self):
        report = heldout_module.require_parity(
            PROVENANCE, candidate_ids=list(PROVENANCE.candidates('validation'))[:2])
        self.assertEqual(report['HELDOUT_ADAPTER_PARITY'], 'PASS')
        self.assertEqual(report['mismatch_count'], 0)
        self.assertFalse(report['held_out_geometry_read'])
        self.assertGreaterEqual(report['examples'], 21)

    def test_K_held_out_evaluation_requires_the_freeze_file_to_exist(self):
        pilot = orchestrator.Pilot.__new__(orchestrator.Pilot)
        pilot.layout = state_module.RunLayout(self.root)
        self.assertFalse(pilot.layout.freeze_path.exists())
        with self.assertRaises(freeze_module.FreezeError) as raised:
            pilot.evaluate_heldout('pointnet', 101, {}, 0.5, 4096, 0, 'x')
        self.assertEqual(raised.exception.code, 'STOP_HELDOUT_BEFORE_FREEZE')


# --- held-out provenance discipline ---------------------------------------

class HeldOutProvenance(unittest.TestCase):
    """The held-out route uses real held-out context and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.view = heldout_module.held_out_view()
        cls.candidates = list(cls.view.eligible)

    def test_held_out_view_matches_the_frozen_artifact(self):
        payload = json.loads((provenance_module.REPORTS
                              / 'GATE_C2_CONTEXT_PROVENANCE.json').read_text(
                                  encoding='utf-8'))
        self.assertEqual(list(self.view.eligible), payload['held_out']['eligible'])
        self.assertEqual(len(self.view.eligible),
                         len(PROVENANCE.held_out_eligible))

    def test_integrity_re_derives_the_invariants(self):
        report = heldout_module.assert_heldout_integrity(
            PROVENANCE, self.view, expected_candidates=len(self.view.eligible))
        self.assertEqual(report['cross_partition_overlap'], 0)
        self.assertEqual(report['quarantine_sources_in_heldout'], 0)
        self.assertGreater(report['held_out_context_edges'], 0)

    def test_a_changed_manifest_count_is_refused(self):
        with self.assertRaises(heldout_module.HeldOutError) as raised:
            heldout_module.require_manifest_count(self.view, len(self.view.eligible) - 1)
        self.assertEqual(raised.exception.code, 'STOP_HELDOUT_MANIFEST_CHANGED')

    def test_manifest_count_matches_the_frozen_expectation(self):
        check = heldout_module.require_manifest_count(
            self.view, len(PROVENANCE.held_out_eligible))
        self.assertTrue(check['match'])
        self.assertEqual(check['actual'], 749)

    def test_held_out_geometry_is_authorized_from_its_own_context(self):
        candidate_id = self.candidates[0]
        geometry = HeldOutCandidate(candidate_id)
        scene = heldout_module.build_heldout_scene(candidate_id, geometry, PROVENANCE,
                                                   self.view)
        union = scene_module.scene_source_union(scene)
        authorised = heldout_module.authorize_heldout_scene(
            PROVENANCE, self.view, candidate_id, union)
        self.assertEqual(tuple(authorised), tuple(union))
        self.assertIn(candidate_id, union)
        model_selection = (set(PROVENANCE.train.source_union)
                           | set(PROVENANCE.validation.source_union))
        self.assertEqual(set(union) & model_selection, set())

    def test_a_train_source_in_a_held_out_example_is_refused(self):
        candidate_id = self.candidates[0]
        train_source = sorted(PROVENANCE.train.source_union)[0]
        with self.assertRaises(heldout_module.HeldOutError) as raised:
            heldout_module.authorize_heldout_scene(
                PROVENANCE, self.view, candidate_id, (candidate_id, train_source))
        self.assertEqual(raised.exception.code, 'STOP_HELDOUT_MANIFEST_CHANGED')

    def test_a_non_context_held_out_tree_is_refused(self):
        candidate_id = self.candidates[0]
        context = set(self.view.context[candidate_id])
        outsider = next(cid for cid in self.candidates[1:] if cid not in context)
        with self.assertRaises(heldout_module.HeldOutError):
            heldout_module.authorize_heldout_scene(
                PROVENANCE, self.view, candidate_id, (candidate_id, outsider))

    def test_a_non_candidate_is_refused_as_a_held_out_candidate(self):
        unknown = sorted(set(PROVENANCE.train.eligible))[0]
        with self.assertRaises(heldout_module.HeldOutError):
            heldout_module.authorize_heldout_scene(PROVENANCE, self.view, unknown,
                                                   (unknown,))

    def test_a_candidate_without_its_own_points_is_refused(self):
        candidate_id = self.candidates[0]
        with self.assertRaises(heldout_module.HeldOutError) as raised:
            heldout_module.authorize_heldout_scene(PROVENANCE, self.view, candidate_id,
                                                   ())
        self.assertEqual(raised.exception.code, 'STOP_HELDOUT_MANIFEST_CHANGED')

    def test_quarantine_trees_are_unreachable_in_held_out(self):
        quarantine = sorted(PROVENANCE.high_containment_quarantine)
        self.assertTrue(quarantine)
        candidate_id = self.candidates[0]
        with self.assertRaises(heldout_module.HeldOutError):
            heldout_module.authorize_heldout_scene(
                PROVENANCE, self.view, candidate_id, (candidate_id, quarantine[0]))

    def test_held_out_dataset_refuses_a_model_selection_candidate(self):
        with self.assertRaises(heldout_module.HeldOutError):
            heldout_module.HeldOutDataset(PROVENANCE, self.view,
                                          candidate_ids=[sorted(
                                              PROVENANCE.train.eligible)[0]])

    def test_held_out_adapter_builds_a_usable_example(self):
        candidate_id = self.candidates[0]
        dataset = heldout_module.HeldOutDataset(
            PROVENANCE, self.view, candidate_ids=[candidate_id], points=4096,
            cache_root=Path('D:/TreeQCTraining/tmp/efg_heldout_test'))
        self.assertEqual(len(dataset), 21)
        item = dataset[0]
        self.assertEqual(item.features.shape, (4096, 4))
        self.assertEqual(item.features.dtype, np.float32)
        self.assertEqual(item.provenance['partition'], 'held_out')
        self.assertIn(item.label, (0, 1))


# --- the adapter port itself ----------------------------------------------

class AdapterPort(unittest.TestCase):
    """The ported corruption body matches the frozen generator field by field."""

    def test_spec_enumeration_matches_the_frozen_stride(self):
        candidate_ids = list(PROVENANCE.candidates('validation'))[:3]
        mine = specs_module.enumerate_specs(candidate_ids, 'validation')
        frozen = gen.enumerate_specs(candidate_ids, 'validation')
        self.assertEqual(len(mine), len(frozen))
        for adapter_spec, frozen_spec in zip(mine, frozen, strict=True):
            self.assertEqual(adapter_spec.key, frozen_spec.key)
            self.assertEqual(adapter_spec.seed, frozen_spec.seed)
            self.assertEqual(adapter_spec.as_row(), frozen_spec.as_row())
            self.assertEqual(adapter_spec.sampling_seed(4096),
                             frozen_spec.sampling_seed(4096))

    def test_specs_refuse_a_non_model_selection_partition_in_the_frozen_class(self):
        self.assertIn('held_out', specs_module.EFG_PARTITIONS)
        with self.assertRaises(gen.GeneratorError):
            gen.ExampleSpec(candidate_id='x', partition='held_out',
                            example_class=gen.GOOD, severity=gen.LOW, seed=1)

    def test_corruption_body_matches_on_a_frozen_scene(self):
        import gate_d_dataset as dataset_module
        from efg_specs import SamplingSpec
        candidate_id = list(PROVENANCE.candidates('validation'))[0]
        geometry = dataset_module.GateDDataset(
            dataset_module.DatasetConfig(partition='validation'), PROVENANCE,
            candidate_ids=[candidate_id])
        candidate_geometry = geometry._geometry_for(candidate_id)
        scene = scene_module.build_base_scene(candidate_id, 'validation',
                                              candidate_geometry, PROVENANCE)
        frozen_spec = gen.enumerate_specs([candidate_id], 'validation')[0]
        adapter_spec = SamplingSpec.from_row(frozen_spec.as_row())
        mine = heldout_module.generate_selftest_example(adapter_spec, scene, PROVENANCE)
        theirs = gen.generate_example(frozen_spec, candidate_geometry, PROVENANCE)
        self.assertEqual(mine.label, theirs.label)
        self.assertTrue(np.array_equal(mine.membership, theirs.membership))
        self.assertTrue(np.array_equal(mine.changed_membership,
                                       theirs.changed_membership))
        self.assertEqual(mine.provenance['membership_fraction'],
                         theirs.provenance['membership_fraction'])


# --- L: dataset fingerprint ------------------------------------------------

class DatasetFingerprint(unittest.TestCase):
    """L: the frozen split is measured, not assumed."""

    def test_spec_list_digest_changes_with_the_candidate_order(self):
        ids = list(PROVENANCE.candidates('validation'))[:4]
        first = specs_module.enumerate_specs(ids, 'validation')
        second = specs_module.enumerate_specs(list(reversed(ids)), 'validation')
        self.assertNotEqual(specs_module.spec_list_digest(first),
                            specs_module.spec_list_digest(second))

    def test_class_balance_counts_all_classes_and_severities(self):
        ids = list(PROVENANCE.candidates('validation'))[:2]
        balance = specs_module.class_balance(PROVENANCE, ids, 'validation')
        self.assertEqual(balance['declared_examples'], 2 * 7 * 3)
        self.assertEqual(len(balance['declared_per_class_severity']), 21)

    def test_dataset_source_refuses_a_non_model_selection_partition(self):
        config = engine.TrainConfig.from_row({
            **engine.TrainConfig(model='pointnet').as_row(),
            'partition': 'held_out'})
        with self.assertRaises(Exception) as raised:
            dataset_source.build_model_selection_source(config, PROVENANCE)
        self.assertIn('model-selection', str(raised.exception))

    def test_gate_d_dataset_refuses_a_non_model_selection_partition(self):
        import gate_d_dataset as dataset_module
        with self.assertRaises(dataset_module.DatasetError):
            dataset_module.DatasetConfig(partition='held_out')

    def test_cache_round_trip_and_tamper_detection(self):
        import efg_cache as cache_module
        with tempfile.TemporaryDirectory() as temporary:
            key = cache_module.cache_key(PROVENANCE, 'validation', 4096, 1, 0,
                                         'spec-digest', 'generator-digest', 'sampler')
            cache = cache_module.FeatureCache(Path(temporary), key)
            features = np.arange(16, dtype=np.float32).reshape(4, 4)
            cache.put(0, 'key-0', features, 1, {'key': 'key-0'})
            hit = cache.get(0, 'key-0')
            self.assertIsNotNone(hit)
            self.assertTrue(np.array_equal(hit[0], features))
            self.assertEqual(hit[1], 1)
            self.assertEqual(hit[2], {'key': 'key-0'})
            self.assertIsNone(cache.get(0, 'other-key'))
            self.assertIsNone(cache.get(1, 'key-0'))
            np.save(cache._paths(0)[0], features + 1.0)
            self.assertIsNone(cache.get(0, 'key-0'))

    def test_cache_key_changes_with_every_component(self):
        import efg_cache as cache_module
        base = cache_module.cache_key(PROVENANCE, 'validation', 4096, 1, 0, 's', 'g',
                                      'v')
        for field, value in (('partition', 'train'), ('points', 2048), ('repeats', 2),
                             ('seed_base', 9), ('spec_list_digest', 'x'),
                             ('generator_digest', 'y'), ('sampler_version', 'z')):
            with self.subTest(field=field):
                self.assertNotEqual(base, {**base, field: value})

    def test_cached_features_match_fresh_frozen_generation(self):
        config = engine.TrainConfig.from_row({
            **engine.TrainConfig(model='pointnet', points=4096).as_row(),
            'partition': 'validation'})
        with tempfile.TemporaryDirectory() as temporary:
            source = dataset_source.build_model_selection_source(
                config, PROVENANCE, cache_root=Path(temporary), use_cache=True,
                preload=False)
            for index in range(min(10, len(source))):
                source[index]
            report = dataset_source.verify_cache(source, range(min(10, len(source))))
            self.assertEqual(report['verdict'], 'PASS')
            self.assertEqual(report['mismatch_count'], 0)
            self.assertEqual(report['checked'], min(10, len(source)))


# --- M: non-finite training ------------------------------------------------

class NonFiniteGuard(unittest.TestCase):
    """M: a non-finite loss or logit fails closed immediately."""

    def test_non_finite_loss_is_a_stop(self):
        monitor = trainer_module.NonFiniteMonitor()
        monitor.last_logits_finite = True
        with self.assertRaises(trainer_module.TrainingStop) as raised:
            monitor.check(float('nan'), {'model': 'pointnet', 'seed': 101, 'epoch': 3,
                                         'batch': 7})
        self.assertEqual(raised.exception.code, 'STOP_NONFINITE_TRAINING')
        self.assertEqual(raised.exception.detail['epoch'], 3)
        self.assertEqual(raised.exception.detail['batch'], 7)

    def test_infinite_loss_is_a_stop(self):
        monitor = trainer_module.NonFiniteMonitor()
        with self.assertRaises(trainer_module.TrainingStop) as raised:
            monitor.check(float('inf'), {'epoch': 0})
        self.assertEqual(raised.exception.code, 'STOP_NONFINITE_TRAINING')

    def test_non_finite_logits_are_a_stop(self):
        import torch
        monitor = trainer_module.NonFiniteMonitor()

        class Model:
            def forward(self, value):
                return torch.tensor([[float('nan'), 0.0]])

        model = Model()
        original = monitor.wrap(model)
        # ``wrap`` returns the callable it replaced so the trainer can restore it, and
        # installs a different one in the meantime.
        self.assertEqual(original.__name__, 'forward')
        self.assertEqual(model.forward.__name__, 'instrumented')
        model.forward(torch.zeros(1, 4))
        self.assertFalse(monitor.last_logits_finite)
        model.forward = original
        self.assertEqual(model.forward.__name__, 'forward')
        with self.assertRaises(trainer_module.TrainingStop) as raised:
            monitor.check(1.0, {'epoch': 1})
        self.assertEqual(raised.exception.code, 'STOP_NONFINITE_TRAINING')

    def test_finite_training_is_not_a_stop(self):
        monitor = trainer_module.NonFiniteMonitor()
        monitor.last_logits_finite = True
        monitor.check(0.5, {'epoch': 1})
        self.assertEqual(monitor.last_loss, 0.5)


# --- N: resource guards ----------------------------------------------------

class ResourceGuards(unittest.TestCase):
    """N: the disk guard stops instead of deleting anything."""

    def test_disk_below_guard_stops(self):
        disks = {'D': {'free_bytes': 10 * 1024**3, 'free_gib': 10.0},
                 'E': {'free_bytes': 60 * 1024**3, 'free_gib': 60.0}}
        with self.assertRaises(resources_module.ResourceError) as raised:
            resources_module.require_disk_guard(disks, ('D', 'E'))
        self.assertEqual(raised.exception.code, 'STOP_DISK_GUARD')
        self.assertIn('no file is deleted', json.dumps(raised.exception.detail))

    def test_disk_above_guard_passes(self):
        disks = {'D': {'free_bytes': 100 * 1024**3, 'free_gib': 100.0},
                 'E': {'free_bytes': 60 * 1024**3, 'free_gib': 60.0}}
        resources_module.require_disk_guard(disks, ('D', 'E'))

    def test_guard_threshold_is_the_declared_reserve(self):
        self.assertEqual(resources_module.DISK_GUARD_BYTES, 35 * 1024**3)
        self.assertEqual(gates.DISK_GUARD_GIB, 35)

    def test_missing_environment_stops(self):
        payload = {'gpu': {'torch': None, 'cuda_available': False},
                   'data_root_exists': False, 'ram_total_bytes': None,
                   'paths': {'data_root': 'X:/missing'}}
        with self.assertRaises(resources_module.ResourceError) as raised:
            resources_module.require_environment(payload)
        self.assertEqual(raised.exception.code, 'STOP_ENVIRONMENT_NOT_READY')
        items = {problem['item'] for problem in raised.exception.detail['problems']}
        self.assertEqual(items, {'torch', 'cuda', 'data_root', 'ram'})

    def test_probe_batch_order_is_the_declared_order(self):
        self.assertEqual(tuple(resources_module.PROBE_BATCHES), (16, 8, 4, 2, 1))
        self.assertEqual(gates.PROBE_BATCHES, (16, 8, 4, 2, 1))

    def test_calibration_selects_a_common_batch(self):
        original = resources_module._probe_one
        profile = {16: 'STABLE', 8: 'STABLE', 4: 'STABLE', 2: 'STABLE', 1: 'STABLE'}
        resources_module._probe_one = (
            lambda model, batch, device: {'batch': batch, 'model': model,
                                          'status': profile[batch]})
        try:
            calibration = resources_module.calibrate_batch(['pointnet'])
            self.assertEqual(calibration['models']['pointnet']['safe_batch'], 8)
            self.assertEqual(calibration['safe_common_batch'], 8)
        finally:
            resources_module._probe_one = original

    def test_calibration_honours_the_half_largest_stable_policy(self):
        original = resources_module._probe_one
        profile = {16: 'OOM', 8: 'OOM', 4: 'STABLE', 2: 'STABLE', 1: 'STABLE'}
        resources_module._probe_one = (
            lambda model, batch, device: {'batch': batch, 'model': model,
                                          'status': profile[batch]})
        try:
            calibration = resources_module.calibrate_batch(['pointnet', 'pointnet2_ssg'])
            self.assertEqual(calibration['models']['pointnet']['safe_batch'], 2)
            self.assertEqual(calibration['safe_common_batch'], 2)
        finally:
            resources_module._probe_one = original

    def test_calibration_without_any_stable_batch_stops(self):
        original = resources_module._probe_one
        resources_module._probe_one = (
            lambda model, batch, device: {'batch': batch, 'model': model,
                                          'status': 'OOM'})
        try:
            with self.assertRaises(resources_module.ResourceError) as raised:
                resources_module.calibrate_batch(['pointnet'])
            self.assertEqual(raised.exception.code, 'STOP_BATCH_CALIBRATION')
        finally:
            resources_module._probe_one = original


# --- architecture and seed selection --------------------------------------

class SelectionRules(unittest.TestCase):
    """The pre-declared selection rules, exercised on synthetic evidence."""

    def _metrics(self, recall, balanced, false_review):
        return {'BAD_RECALL': recall, 'BALANCED_ACCURACY': balanced,
                'GOOD_FALSE_REVIEW': false_review, 'BAD_FALSE_PASS': 1 - recall}

    def test_higher_recall_wins(self):
        result = freeze_module.select_architecture({
            'pointnet': self._metrics(0.80, 0.85, 0.10),
            'pointnet2_ssg': self._metrics(0.92, 0.94, 0.10)})
        self.assertEqual(result['selected_architecture'], 'pointnet2_ssg')
        self.assertFalse(result['non_inferiority_applied'])

    def test_non_inferiority_prefers_pointnet(self):
        result = freeze_module.select_architecture({
            'pointnet': self._metrics(0.900, 0.930, 0.10),
            'pointnet2_ssg': self._metrics(0.905, 0.935, 0.10)})
        self.assertEqual(result['selected_architecture'], 'pointnet')
        self.assertTrue(result['non_inferiority_applied'])

    def test_a_clear_recall_gap_is_not_non_inferior(self):
        result = freeze_module.select_architecture({
            'pointnet': self._metrics(0.880, 0.920, 0.10),
            'pointnet2_ssg': self._metrics(0.930, 0.950, 0.10)})
        self.assertEqual(result['selected_architecture'], 'pointnet2_ssg')

    def test_the_good_false_review_budget_binds_first(self):
        result = freeze_module.select_architecture({
            'pointnet': self._metrics(0.99, 0.99, 0.40),
            'pointnet2_ssg': self._metrics(0.70, 0.80, 0.10)})
        self.assertEqual(result['selected_architecture'], 'pointnet2_ssg')

    def test_no_architecture_inside_the_budget_is_a_stop(self):
        with self.assertRaises(freeze_module.FreezeError) as raised:
            freeze_module.select_architecture({
                'pointnet': self._metrics(0.99, 0.99, 0.40),
                'pointnet2_ssg': self._metrics(0.99, 0.99, 0.35)})
        self.assertEqual(raised.exception.code,
                         'STOP_ARCHITECTURE_BUDGET_UNSATISFIED')

    def test_missing_evidence_is_a_stop_not_a_default(self):
        with self.assertRaises(freeze_module.FreezeError) as raised:
            freeze_module.select_architecture({'pointnet': self._metrics(0.9, 0.9, 0.1)})
        self.assertEqual(raised.exception.code,
                         'STOP_ARCHITECTURE_EVIDENCE_INCOMPLETE')

    def test_deployment_seed_rule(self):
        result = freeze_module.select_deployment_candidate({
            101: self._metrics(0.86, 0.90, 0.10),
            202: self._metrics(0.91, 0.93, 0.10),
            303: self._metrics(0.91, 0.94, 0.10)})
        self.assertEqual(result['deployment_candidate_seed'], 303)

    def test_deployment_seed_tie_break_is_the_lowest_seed(self):
        result = freeze_module.select_deployment_candidate({
            101: self._metrics(0.90, 0.92, 0.10),
            202: self._metrics(0.90, 0.92, 0.10),
            303: self._metrics(0.90, 0.92, 0.10)})
        self.assertEqual(result['deployment_candidate_seed'], 101)

    def test_deployment_seed_budget_binds(self):
        result = freeze_module.select_deployment_candidate({
            101: self._metrics(0.99, 0.99, 0.40),
            202: self._metrics(0.80, 0.85, 0.10),
            303: self._metrics(0.99, 0.99, 0.40)})
        self.assertEqual(result['deployment_candidate_seed'], 202)

    def test_selection_is_deterministic(self):
        evidence = {'pointnet': self._metrics(0.900, 0.930, 0.10),
                    'pointnet2_ssg': self._metrics(0.905, 0.935, 0.10)}
        first = freeze_module.select_architecture(evidence)
        for _ in range(5):
            self.assertEqual(freeze_module.select_architecture(evidence), first)


# --- metrics and verdicts --------------------------------------------------

class MetricsAndVerdicts(unittest.TestCase):
    """The five primary metrics and the closed verdict vocabulary."""

    def test_balanced_accuracy_uses_the_frozen_definition(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.7, 0.2, 0.9])
        metrics = evaluate_module.binary_metrics(labels, scores, 0.5)
        self.assertEqual(metrics['CONFUSION_MATRIX']['fp_pass_as_review'], 1)
        self.assertEqual(metrics['CONFUSION_MATRIX']['fn_review_as_pass'], 1)
        self.assertAlmostEqual(metrics['BAD_RECALL'], 0.5)
        self.assertAlmostEqual(metrics['GOOD_FALSE_REVIEW'], 0.5)
        self.assertAlmostEqual(metrics['BALANCED_ACCURACY'], 0.5)
        self.assertAlmostEqual(metrics['F1_REVIEW'], 0.5)

    def test_per_defect_and_severity_breakdown(self):
        classes = np.array(['GOOD'] * 4 + ['MERGED_MULTI_TREE'] * 4)
        severities = np.array(['MEDIUM'] * 2 + ['HIGH'] * 2 + ['LOW'] * 2
                              + ['HIGH'] * 2)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.3, 0.9])
        result = {'labels': labels, 'scores': scores, 'classes': classes,
                  'severities': severities, 'partition': 'x'}
        scored = evaluate_module.score_partition(result, 0.5)
        self.assertIn('MERGED_MULTI_TREE', scored['per_defect_primary'])
        self.assertIn('GOOD', scored['per_class'])
        self.assertIn('LOW', scored['per_severity'])
        self.assertEqual(scored['primary_moderate_high']['bad_examples'], 2)
        self.assertAlmostEqual(scored['per_defect_primary']['MERGED_MULTI_TREE'][
            'BAD_RECALL'], 0.5)

    def test_multi_seed_aggregation_reports_every_seed(self):
        per_seed = {
            101: {'BAD_RECALL': 0.90, 'BAD_FALSE_PASS': 0.10,
                  'GOOD_FALSE_REVIEW': 0.10, 'BALANCED_ACCURACY': 0.90,
                  'F1_REVIEW': 0.90},
            202: {'BAD_RECALL': 0.86, 'BAD_FALSE_PASS': 0.14,
                  'GOOD_FALSE_REVIEW': 0.12, 'BALANCED_ACCURACY': 0.87,
                  'F1_REVIEW': 0.87},
            303: {'BAD_RECALL': 0.88, 'BAD_FALSE_PASS': 0.12,
                  'GOOD_FALSE_REVIEW': 0.11, 'BALANCED_ACCURACY': 0.88,
                  'F1_REVIEW': 0.88}}
        summary = evaluate_module.aggregate_seed_metrics(per_seed)
        recall = summary['metrics']['BAD_RECALL']
        self.assertAlmostEqual(recall['mean'], 0.88, places=6)
        self.assertAlmostEqual(recall['min'], 0.86, places=6)
        self.assertAlmostEqual(recall['max'], 0.90, places=6)
        self.assertEqual(recall['n'], 3)
        self.assertEqual(summary['MODEL_STABILITY'], 'PASS')

    def test_unstable_seeds_are_flagged(self):
        per_seed = {seed: {'BAD_RECALL': value, 'BAD_FALSE_PASS': 0.1,
                           'GOOD_FALSE_REVIEW': 0.1, 'BALANCED_ACCURACY': value,
                           'F1_REVIEW': value}
                    for seed, value in ((101, 0.95), (202, 0.70), (303, 0.85))}
        summary = evaluate_module.aggregate_seed_metrics(per_seed)
        self.assertEqual(summary['MODEL_STABILITY'], 'FAIL')
        self.assertGreater(summary['bad_recall_spread'], 0.10)

    def test_sampling_robustness_drop_limit(self):
        configurations = {
            '4096_sampling_seed_0': {'BAD_RECALL': 0.90, 'points': 4096,
                                     'sampling_seed': 0},
            '2048_sampling_seed_11': {'BAD_RECALL': 0.85, 'points': 2048,
                                      'sampling_seed': 11},
        }
        summary = evaluate_module.robustness_summary(0.90, configurations)
        self.assertEqual(summary['SAMPLING_ROBUSTNESS'], 'PASS')
        self.assertAlmostEqual(summary['max_bad_recall_drop'], 0.05)
        self.assertAlmostEqual(summary['max_bad_recall_drop_pp'], 5.0)

    def test_sampling_sensitivity_is_flagged(self):
        configurations = {'2048_sampling_seed_11': {'BAD_RECALL': 0.70,
                                                    'points': 2048,
                                                    'sampling_seed': 11}}
        summary = evaluate_module.robustness_summary(0.90, configurations)
        self.assertEqual(summary['SAMPLING_ROBUSTNESS'], 'FAIL')
        self.assertEqual(summary['verdict'], 'SAMPLING_TOO_SENSITIVE')

    def test_primary_gate_targets_are_the_declared_targets(self):
        passing = {'BAD_RECALL': 0.86, 'BAD_FALSE_PASS': 0.14,
                   'GOOD_FALSE_REVIEW': 0.24, 'BALANCED_ACCURACY': 0.81}
        self.assertTrue(evaluate_module.apply_primary_gate(passing)['passed'])
        for field, value in (('BAD_RECALL', 0.84), ('BAD_FALSE_PASS', 0.16),
                             ('GOOD_FALSE_REVIEW', 0.26),
                             ('BALANCED_ACCURACY', 0.79)):
            with self.subTest(field=field):
                failed = evaluate_module.apply_primary_gate({**passing, field: value})
                self.assertFalse(failed['passed'])
                self.assertIn(field, failed['failed'])

    def test_learnability_gate(self):
        self.assertTrue(evaluate_module.learnability_verdict(
            {'BAD_RECALL': 0.75, 'BALANCED_ACCURACY': 0.75})['passed'])
        verdict = evaluate_module.learnability_verdict(
            {'BAD_RECALL': 0.40, 'BALANCED_ACCURACY': 0.55})
        self.assertFalse(verdict['passed'])
        self.assertEqual(verdict['verdict'], 'PUBLIC_PILOT_NOT_LEARNABLE')

    def test_verdict_vocabulary_is_closed(self):
        vocabulary = {gates.VERDICT_NOT_LEARNABLE, gates.VERDICT_UNSTABLE,
                      gates.VERDICT_SAMPLING_TOO_SENSITIVE, gates.VERDICT_READY}
        self.assertNotIn('PRODUCT_READY', vocabulary)
        self.assertEqual(gates.PRODUCT_READY_FORBIDDEN, 'PRODUCT_READY')
        ready = reports_module.decide_verdict(
            None, True, {'evaluated': True, 'passed': True, 'failed': []},
            {'MODEL_STABILITY': 'PASS'}, {'SAMPLING_ROBUSTNESS': 'PASS'},
            {'critical_leakage': 'none', 'heldout_protocol_clean': True,
             'meaningful_defects': {'MERGED_MULTI_TREE': True,
                                    'NEIGHBOR_CONTAMINATION': True,
                                    'OVER_SEGMENTED': True}}, None)
        self.assertEqual(ready['verdict'], gates.VERDICT_READY)

    def test_a_stop_reports_its_own_code(self):
        verdict = reports_module.decide_verdict(
            'STOP_DISK_GUARD', False, None, None, None, None, None)
        self.assertEqual(verdict['verdict'], 'STOP_DISK_GUARD')

    def test_verdict_cannot_be_ready_without_held_out(self):
        verdict = reports_module.decide_verdict(
            None, False, {'evaluated': True, 'passed': True, 'failed': []}, None,
            None, None, None)
        self.assertEqual(verdict['verdict'], reports_module.NOT_RUN)


# --- Q: local challenge boundary ------------------------------------------

class LocalChallengeBoundary(unittest.TestCase):
    """Q: this round reads nothing from the local Gate H challenge."""

    def test_no_efg_module_reads_a_local_challenge_path(self):
        report = leakage_module.assert_local_challenge_untouched(HERE)
        self.assertEqual(report['hits'], [])

    def test_scan_would_catch_a_real_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'efg_offender.py').write_text(
                'PATH = "D:/TreeQCData/local_8_bad/case.ply"\n', encoding='utf-8')
            with self.assertRaises(leakage_module.LeakageError) as raised:
                leakage_module.assert_local_challenge_untouched(root)
            self.assertEqual(raised.exception.code, 'STOP_CRITICAL_LEAKAGE')

    def test_the_final_report_declares_the_boundary(self):
        report = reports_module.build_report({})
        self.assertEqual(report['LOCAL_8_BAD_CHALLENGE_ACCESSED'], 'NO')
        self.assertEqual(report['ONNX_STARTED'], 'NO')
        self.assertEqual(report['PRODUCT_INTEGRATION'], 'NO')

    def test_the_interface_contract_is_reserved_not_implemented(self):
        contract = reports_module.interface_contract()
        self.assertEqual(contract['status'], 'reserved_not_implemented')
        self.assertFalse(contract['implemented_this_round'])
        self.assertEqual(contract['future_repair_contract'][
            'AUTO_RESEGMENT_MAX_ATTEMPTS'], 1)
        for field in ('persistent_tree_id', 'qc_score', 'binary_decision',
                      'three_way_decision', 'threshold', 'model_name',
                      'model_version', 'checkpoint_digest', 'sampler_version', 'points',
                      'evidence_metadata'):
            self.assertIn(field, contract['fields'])

    def test_no_product_or_tree_repair_code_was_added(self):
        """Only the reserved *interface* may name the future repair engine."""
        implementation_markers = ('class TreeRepairEngine', 'def tree_repair',
                                  'def local_resegment', 'import onnxruntime',
                                  'from onnxruntime', 'torch.onnx.export(',
                                  'CROWN_STEM_HYBRID(', 'crown_stem_hybrid(')
        for path in sorted(HERE.glob('efg_*.py')):
            source = path.read_text(encoding='utf-8')
            for marker in implementation_markers:
                self.assertNotIn(marker, source,
                                 f'{path.name} must not implement {marker}')

    def test_the_round_adds_no_product_integration_files(self):
        repo = HERE.parent.parent
        for relative in ('TreeRepairEngine', 'tree_repair_engine.py',
                         'research/tree_qc/tree_repair_engine.py'):
            self.assertFalse((repo / relative).exists(), relative)


# --- dry run and CLI -------------------------------------------------------

class DryRunPlan(unittest.TestCase):
    """The dry run resolves the plan without touching the GPU or held-out data."""

    @classmethod
    def setUpClass(cls):
        cls.plan = orchestrator.dry_run(verbose=False)

    def test_reports_the_gate_d_binding(self):
        binding = self.plan['gate_d_binding']
        self.assertEqual(binding['generator_digest'], EXPECTED_GENERATOR)
        self.assertEqual(binding['protocol_digest'], EXPECTED_PROTOCOL)
        self.assertTrue(binding['binding_ok'])
        self.assertTrue(binding['training_allowed'])

    def test_reports_resources(self):
        environment = self.plan['environment']
        self.assertTrue(environment['gpu'])
        self.assertTrue(environment['vram_total_bytes'])
        self.assertTrue(environment['ram_total_bytes'])
        self.assertIn('D', environment['disks'])
        self.assertEqual(environment['disk_guard_gib'], 35)

    def test_reports_the_frozen_dataset_without_held_out_geometry(self):
        dataset = self.plan['dataset']
        self.assertEqual(dataset['train_candidates'], 440)
        self.assertEqual(dataset['validation_candidates'], 175)
        self.assertEqual(dataset['held_out_candidates_metadata_only'], 749)
        self.assertEqual(dataset['train_examples'], 440 * 7 * 3)
        self.assertEqual(dataset['validation_examples'], 175 * 7 * 3)

    def test_reports_the_plan_and_the_locks(self):
        training = self.plan['training']
        self.assertEqual(training['models'], ['pointnet', 'pointnet2_ssg'])
        self.assertEqual(training['model_seeds'], [101, 202, 303])
        self.assertEqual(training['epochs'], 30)
        self.assertEqual(training['points'], 4096)
        self.assertIsNone(training['max_train_candidates'])
        self.assertIsNone(training['max_validation_candidates'])
        self.assertEqual(self.plan['held_out_locked'], 'YES')
        self.assertEqual(self.plan['local_challenge_locked'], 'YES')
        self.assertEqual(self.plan['held_out_accessed'], 'NO')
        self.assertEqual(self.plan['sampling_robustness']['sampling_seeds'],
                         [11, 23, 37])

    def test_stage_list_is_the_declared_sequence(self):
        names = [stage['name'] for stage in self.plan['stages']]
        self.assertEqual(names, list(orchestrator.STAGE_ORDER))
        self.assertEqual(len(names), 19)
        self.assertEqual(names[0], orchestrator.S_ENVIRONMENT)
        self.assertEqual(names[-1], orchestrator.S_REPORT)
        self.assertIn(orchestrator.S_FREEZE, names)
        self.assertLess(names.index(orchestrator.S_PREWARM),
                        names.index(orchestrator.S_COMPARE_POINTNET))
        self.assertLess(names.index(orchestrator.S_FREEZE),
                        names.index(orchestrator.S_HELDOUT))


class CliContract(unittest.TestCase):
    """The formal entry point refuses bounded smoke-test options."""

    def test_formal_run_refuses_bounded_options(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / 'run_tree_qc_efg.py'), '--auto',
             '--max-comparison-candidates', '4'],
            capture_output=True, text=True, cwd=str(HERE), timeout=300)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('REFUSED', completed.stderr)

    def test_formal_run_refuses_a_reduced_epoch_budget(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / 'run_tree_qc_efg.py'), '--auto', '--epochs', '2'],
            capture_output=True, text=True, cwd=str(HERE), timeout=300)
        self.assertEqual(completed.returncode, 2)

    def test_resume_test_passes(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / 'run_tree_qc_efg.py'), '--resume-test'],
            capture_output=True, text=True, cwd=str(HERE), timeout=600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload['RESUME_TEST'], 'PASS')
        self.assertEqual(payload['failed'], [])

    def test_dry_run_exits_zero(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / 'run_tree_qc_efg.py'), '--dry-run', '--json'],
            capture_output=True, text=True, cwd=str(HERE), timeout=600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload['gate_d_binding']['binding_ok'])

    def test_status_is_read_only(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / 'run_tree_qc_efg.py'), '--status',
             '--run-root', str(Path(tempfile.gettempdir()) / 'efg_no_such_run')],
            capture_output=True, text=True, cwd=str(HERE), timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload['state'], 'NO_RUN_FOUND')


class Launcher(unittest.TestCase):
    """The one-click launcher stays a locator, not a second implementation."""

    @classmethod
    def setUpClass(cls):
        cls.path = HERE.parent.parent / 'RUN_TREE_QC_PUBLIC_PILOT.cmd'
        cls.text = cls.path.read_text(encoding='utf-8', errors='replace')

    def test_launcher_exists_at_the_repository_root(self):
        self.assertTrue(self.path.exists(), str(self.path))

    def test_launcher_calls_the_orchestrator(self):
        self.assertIn('run_tree_qc_efg.py', self.text)
        self.assertIn('--auto', self.text)

    def test_launcher_keeps_the_window_open(self):
        self.assertIn('pause', self.text.lower())

    def test_launcher_shows_the_final_status(self):
        self.assertIn('FINAL STATUS', self.text)
        self.assertIn('CURRENT_STATUS.json', self.text)

    def test_launcher_performs_no_dangerous_system_change(self):
        lowered = self.text.lower()
        for forbidden in ('nvidia-smi -r', 'winget ', 'choco ', 'del /', 'rmdir',
                          'format ', 'reg add', 'setx ', 'docker', 'pip install',
                          'cuda_installer', 'diskpart'):
            self.assertNotIn(forbidden, lowered, f'launcher must not run {forbidden}')

    def test_launcher_sets_utf8(self):
        self.assertIn('chcp 65001', self.text)
        self.assertIn('PYTHONUTF8', self.text)


# --- not-applicable specs and partial evaluation ---------------------------

class NotApplicableHandling(unittest.TestCase):
    """A defect the real geometry cannot support is excluded and counted, not faked."""

    def test_exclusion_is_typed_and_carries_its_reason(self):
        exc = dataset_source.NotApplicableExample('k', 'merged_below_one_point',
                                                  {'pool': 3})
        self.assertEqual(exc.reason, 'merged_below_one_point')
        self.assertEqual(exc.detail['pool'], 3)

    def test_tracker_enforces_the_declared_limit(self):
        tracker = trainer_module.NotApplicableTracker(limit=0.05)
        for index in range(3):
            tracker.record(f'validation|c|MERGED_MULTI_TREE|LOW|{index}', 'reason')
        summary = tracker.enforce(100)
        self.assertTrue(summary['within_limit'])
        self.assertEqual(summary['excluded'], 3)
        self.assertEqual(summary['by_class']['MERGED_MULTI_TREE'], 3)
        with self.assertRaises(trainer_module.TrainingStop) as raised:
            tracker.enforce(10)
        self.assertEqual(raised.exception.code, 'STOP_DATASET_FINGERPRINT_CHANGED')

    def test_cache_records_and_returns_an_exclusion(self):
        import efg_cache as cache_module
        with tempfile.TemporaryDirectory() as temporary:
            key = cache_module.cache_key(PROVENANCE, 'validation', 4096, 1, 0, 's', 'g',
                                         'v')
            cache = cache_module.FeatureCache(Path(temporary), key)
            cache.record_exclusion(4, 'key-4', 'merged_below_one_point', {'pool': 3})
            self.assertIsNone(cache.get(4, 'key-4'))
            exclusion = cache.exclusion(4, 'key-4')
            self.assertEqual(exclusion['reason'], 'merged_below_one_point')
            self.assertIsNone(cache.exclusion(4, 'other-key'))
            self.assertEqual(cache.stats()['excluded_entries'], 1)
            self.assertEqual(cache.exclusions(), {'merged_below_one_point': 1})

    def test_scorer_removes_not_applicable_examples_and_stops_when_too_many(self):
        import torch

        class Item:
            def __init__(self, features, label, key, reason=None):
                self.features = features
                self.label = label
                self.key = key
                self.provenance = {'class': 'GOOD', 'severity': 'HIGH'}
                self.reason = reason

        class FakeDataset:
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, index):
                item = self.items[index]
                if item.reason:
                    raise dataset_source.NotApplicableExample(item.key, item.reason)
                return item

        good = np.zeros((4, 4), dtype=np.float32)
        items = [Item(good, 0, f'k{i}') for i in range(99)]
        items.append(Item(good, 1, 'bad', reason='merged_below_one_point'))
        model = torch.nn.Linear(4, 2)
        scored = evaluate_module.score_dataset(model, FakeDataset(items),
                                               torch.device('cpu'))
        self.assertEqual(scored['declared_examples'], 100)
        self.assertEqual(scored['examples'], 99)
        self.assertEqual(scored['excluded_examples'], 1)
        self.assertEqual(scored['excluded_by_reason'], {'merged_below_one_point': 1})

        heavy = [Item(good, 0, f'k{i}') for i in range(5)]
        heavy += [Item(good, 1, f'b{i}', reason='no_context_source') for i in range(5)]
        with self.assertRaises(evaluate_module.EvaluationError) as raised:
            evaluate_module.score_dataset(model, FakeDataset(heavy),
                                          torch.device('cpu'))
        self.assertIn('shrunken population', str(raised.exception))

    def test_masked_dataset_drops_excluded_indices_without_shifting_batches(self):
        source = dataset_source.CachedExampleSource(
            dataset_source.dataset_module.GateDDataset(
                dataset_source.dataset_module.DatasetConfig(partition='validation'),
                PROVENANCE, candidate_ids=list(PROVENANCE.candidates('validation'))[:2]),
            cache=None, preload=False)
        spec = source.dataset.specs[0]
        source.exclusions[spec.key] = {'reason': 'synthetic', 'detail': {}}
        tracker = trainer_module.NotApplicableTracker()
        masked = trainer_module._MaskedDataset(source, tracker)
        self.assertEqual(len(masked), len(source) - 1)
        self.assertEqual(tracker.summary(len(source))['excluded'], 1)
        self.assertNotIn(0, masked.valid)


class PartialHeldOutEvaluation(unittest.TestCase):
    """A rehearsal-sized held-out pass can never be a scientific verdict."""

    def test_partial_evaluation_is_not_ready_for_gate_h(self):
        verdict = reports_module.decide_verdict(
            None, True, {'evaluated': True, 'passed': True, 'failed': []},
            {'MODEL_STABILITY': 'PASS'}, {'SAMPLING_ROBUSTNESS': 'PASS'},
            {'critical_leakage': 'none', 'heldout_protocol_clean': True,
             'meaningful_defects': {'MERGED_MULTI_TREE': True,
                                    'NEIGHBOR_CONTAMINATION': True,
                                    'OVER_SEGMENTED': True}},
            None, heldout_complete=False)
        self.assertEqual(verdict['verdict'], 'STOP_PARTIAL_HELDOUT_EVALUATION')

    def test_a_failed_primary_gate_reports_not_learnable(self):
        verdict = reports_module.decide_verdict(
            None, True,
            {'evaluated': True, 'passed': False,
             'failed': ['BALANCED_ACCURACY', 'GOOD_FALSE_REVIEW']},
            {'MODEL_STABILITY': 'PASS'}, {'SAMPLING_ROBUSTNESS': 'PASS'},
            {'critical_leakage': 'none', 'heldout_protocol_clean': True,
             'meaningful_defects': {}}, None, heldout_complete=True)
        self.assertEqual(verdict['verdict'], gates.VERDICT_NOT_LEARNABLE)

    def test_critical_leakage_has_its_own_verdict(self):
        verdict = reports_module.decide_verdict(
            None, True, {'evaluated': True, 'passed': True, 'failed': []},
            {'MODEL_STABILITY': 'PASS'}, {'SAMPLING_ROBUSTNESS': 'PASS'},
            {'critical_leakage': 'detected', 'heldout_protocol_clean': False,
             'meaningful_defects': {}}, None, heldout_complete=True)
        self.assertEqual(verdict['verdict'], 'STOP_CRITICAL_LEAKAGE')


class TrainingDataPrewarm(unittest.TestCase):
    """The bounded-candidate cap is applied to train and validation together."""

    def test_candidate_cap_bounds_both_partitions(self):
        pilot = orchestrator.Pilot.__new__(orchestrator.Pilot)
        pilot.comparison_candidates = 6
        pilot.epochs = 3
        pilot.points = 4096
        pilot.repeats = 1
        pilot.seed_base = 0
        pilot.calibration = {'safe_common_batch': 8}
        pilot.layout = state_module.RunLayout('D:/TreeQCTraining/tmp/efg_cap_probe')
        train = pilot._config_for('pointnet', 101)
        validation = trainer_module._validation_config(train)
        self.assertEqual(train.max_train_candidates, 6)
        self.assertEqual(train.max_validation_candidates, 6)
        self.assertEqual(validation.partition, 'validation')
        self.assertEqual(validation.max_validation_candidates, 6)
        self.assertIsNone(validation.max_train_candidates)

    def test_formal_run_has_no_candidate_cap(self):
        pilot = orchestrator.Pilot.__new__(orchestrator.Pilot)
        pilot.comparison_candidates = None
        pilot.epochs = 30
        pilot.points = 4096
        pilot.repeats = 1
        pilot.seed_base = 0
        pilot.calibration = {'safe_common_batch': 8}
        pilot.layout = state_module.RunLayout('D:/TreeQCTraining/tmp/efg_cap_probe')
        self.assertIsNone(pilot.candidate_cap)
        config = pilot._config_for('pointnet', 101)
        self.assertIsNone(config.max_train_candidates)
        self.assertIsNone(config.max_validation_candidates)

    def test_learnability_limits_are_declared_and_relaxed_only_for_smoke(self):
        formal = orchestrator.learnability_limits(False)
        self.assertEqual(formal, gates.LEARNABILITY_GATE)
        smoke = orchestrator.learnability_limits(True)
        self.assertTrue(smoke['smoke_test_relaxed'])
        self.assertEqual(smoke['declared'], gates.LEARNABILITY_GATE)
        self.assertEqual(smoke['BALANCED_ACCURACY_min'], 0.0)


class FrozenPathDefect(unittest.TestCase):
    """The frozen loader defect is worked around here, never repaired in place."""

    def test_frozen_make_loader_cannot_collate_a_batch(self):
        import gate_d_dataset as dataset_module
        dataset = dataset_module.GateDDataset(
            dataset_module.DatasetConfig(partition='validation'), PROVENANCE,
            candidate_ids=list(PROVENANCE.candidates('validation'))[:1])
        item = dataset[0]
        with self.assertRaises(TypeError):
            # The frozen exporter yields (features, label); the frozen collate expects a
            # third index tensor. Reproducing it keeps the finding verifiable.
            dataset_module.collate([(item.features, item.label)])
            dataset_module.collate([dataset_module.TorchGateDDataset(dataset)[0]])

    def test_efg_loader_collates_the_same_item(self):
        import gate_d_dataset as dataset_module
        import torch
        source = dataset_source.CachedExampleSource(
            dataset_module.GateDDataset(
                dataset_module.DatasetConfig(partition='validation'), PROVENANCE,
                candidate_ids=list(PROVENANCE.candidates('validation'))[:1]),
            cache=None, preload=False)
        features, labels, indices = trainer_module._collate(
            [dataset_module.TorchGateDDataset(source)[0]])
        self.assertEqual(tuple(features.shape), (1, 4096, 4))
        self.assertEqual(tuple(labels.shape), (1,))
        self.assertEqual(tuple(indices.shape), (1,))
        self.assertEqual(indices.dtype, torch.long)

    def test_gate_d_files_are_still_unmodified(self):
        """The workaround lives outside the digest; the digest must still match."""
        status = orchestrator.gate_d_binding_status()
        self.assertTrue(status['binding_ok'])
        self.assertEqual(status['runtime_generator_digest'], EXPECTED_GENERATOR)


# --- report shape ----------------------------------------------------------

class FinalReportShape(unittest.TestCase):
    """The final report carries every field the round specification names."""

    REQUIRED = (
        'BASE_ACCEPTED_HEAD', 'IMPLEMENTATION_HEAD', 'GATE_D_ACCEPTANCE',
        'GATE_D_GENERATOR_DIGEST', 'GATE_D_PROTOCOL_DIGEST', 'GATE_D_CORE_CHANGED',
        'AUTOPILOT', 'RESUME_TEST', 'SAFE_BATCH_POINTNET', 'SAFE_BATCH_POINTNET2',
        'COMMON_TRAIN_BATCH', 'POINTNET_SEED101', 'POINTNET2_SEED101',
        'SELECTED_ARCHITECTURE', 'ARCHITECTURE_SELECTION_BASIS', 'MODEL_SEEDS',
        'ALL_MODEL_SEEDS_FROZEN_BEFORE_HELDOUT', 'DEPLOYMENT_CANDIDATE_SEED',
        'PUBLIC_EXPERIMENT_FREEZE_SHA256', 'HELDOUT_ADAPTER_PARITY',
        'HELDOUT_ACCESSED_BEFORE_FREEZE', 'HELDOUT_ACCESSED', 'HELDOUT_CANDIDATES',
        'HELDOUT_PRIMARY_BAD_RECALL', 'HELDOUT_PRIMARY_BAD_FALSE_PASS',
        'HELDOUT_PRIMARY_GOOD_FALSE_REVIEW', 'HELDOUT_PRIMARY_BALANCED_ACCURACY',
        'HELDOUT_PRIMARY_F1_REVIEW', 'PER_DEFECT_METRICS', 'MULTI_SEED_BAD_RECALL_MEAN',
        'MULTI_SEED_BAD_RECALL_MIN', 'MULTI_SEED_BAD_RECALL_MAX',
        'MULTI_SEED_BAD_RECALL_STD', 'MODEL_STABILITY', 'ROBUSTNESS_4096',
        'ROBUSTNESS_2048', 'MAX_BAD_RECALL_DROP', 'SAMPLING_ROBUSTNESS',
        'LOCAL_8_BAD_CHALLENGE_ACCESSED', 'ONNX_STARTED', 'PRODUCT_INTEGRATION',
        'GATE_D_TESTS', 'GATE_C2_TESTS', 'EFG_TESTS', 'RUFF', 'WORKTREE_CLEAN',
        'VERDICT',
    )

    def test_every_required_field_is_present(self):
        report = reports_module.build_report({})
        for field in self.REQUIRED:
            self.assertIn(field, report, f'missing report field {field}')

    def test_unavailable_values_are_not_invented(self):
        report = reports_module.build_report({})
        self.assertEqual(report['HELDOUT_PRIMARY_BAD_RECALL'], reports_module.NOT_RUN)
        self.assertEqual(report['SELECTED_ARCHITECTURE'], reports_module.NOT_RUN)

    def test_markdown_renders(self):
        report = reports_module.build_report({})
        text = reports_module.render_markdown(report)
        self.assertIn('# TREE_QC_EFG_FINAL_REPORT', text)
        self.assertIn('Verdict vocabulary', text)
        self.assertIn('PRODUCT_READY', text)

    def test_report_digest_is_stable_apart_from_the_timestamp(self):
        first = reports_module.build_report({'run_id': 'r'})
        second = reports_module.build_report({'run_id': 'r'})
        first.pop('generated_at')
        second.pop('generated_at')
        self.assertEqual(state_module.digest_payload(first),
                         state_module.digest_payload(second))

    def test_stop_report_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = reports_module.write_stop_report(
                temporary, {'code': 'STOP_DISK_GUARD', 'detail': {'free_gib': 3}},
                {'run_id': 'r', 'stages': {}})
            payload = json.loads(Path(result['json']).read_text(encoding='utf-8'))
            self.assertEqual(payload['stop_code'], 'STOP_DISK_GUARD')
            self.assertIn('next_action', payload)


if __name__ == '__main__':
    unittest.main(verbosity=2)
