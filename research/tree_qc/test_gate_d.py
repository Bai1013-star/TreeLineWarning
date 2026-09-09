"""Gate D tests: donor authorization, fail-closed provenance, determinism, sampler.

Fixture geometry is synthetic *for speed*, but every provenance test runs against the
real frozen Gate C2 artifacts, because the contract is what is being tested. Tests that
need real publisher geometry are marked and skipped when the verified extraction root is
absent.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import gate_d_dataset as dataset_module
import gate_d_fingerprint as fingerprint
import gate_d_generator as gen
import gate_d_geometry as geometry
import gate_d_metrics as metrics
import gate_d_provenance as provenance_module
import gate_d_sampler as sampler
import numpy as np

TMP_ROOT = Path('D:/TreeQCTraining/tmp')
DATA_ROOT = Path('D:/TreeQCData/extracted_manual_verified')
HAS_REAL_DATA = DATA_ROOT.is_dir()
PROVENANCE = provenance_module.load_gate_d_provenance()


def fixture_geometry(candidate_id: str, context_ids, seed: int = 0,
                     background_points: int = 4000):
    """A small, entirely synthetic geometry bundle for generator/sampler tests."""
    rng = np.random.default_rng(seed)
    def tree(tree_id, count, origin):
        points = (rng.random((count, 3)).astype(np.float32) * 2.0
                  + np.asarray(origin, dtype=np.float32))
        return geometry.TreeGeometry(canonical_tree_id=tree_id, site='litchfield',
                                     points=points,
                                     membership=np.ones(count, dtype=bool),
                                     dominant_instance=1, point_count=count,
                                     stored_points=count)
    candidate = tree(candidate_id, 600, (0.0, 0.0, 0.0))
    context = tuple(tree(cid, 400, (3.0 * (i + 1), 0.0, 0.0))
                    for i, cid in enumerate(context_ids))
    background = (rng.random((background_points, 3)).astype(np.float32) * 3.0
                  + np.asarray((0.5, 0.5, -1.0), dtype=np.float32))
    label = rng.integers(0, 3, size=background_points).astype(np.int32)
    return geometry.CandidateGeometry(candidate=candidate, context=context,
                                      background=background, background_label=label,
                                      background_ids=tuple(context_ids[:1]))


class ProvenanceFixture(unittest.TestCase):
    def setUp(self):
        self.candidate = PROVENANCE.train.eligible[0]
        self.context = PROVENANCE.authorized_context(self.candidate)
        self.held_out = PROVENANCE.held_out_eligible[0]
        self.quarantine = sorted(PROVENANCE.high_containment_quarantine)[0]


class DonorAuthorization(ProvenanceFixture):
    def test_own_surviving_context_is_authorized(self):
        for donor in self.context:
            PROVENANCE.authorize_donor(self.candidate, donor)

    def test_self_donor_rejected(self):
        with self.assertRaises(provenance_module.SelfDonorError):
            PROVENANCE.authorize_donor(self.candidate, self.candidate)

    def test_held_out_donor_rejected(self):
        # The held-out check fires before the allowlist check, so the refusal is
        # specifically a held-out access violation rather than a generic one.
        with self.assertRaises(provenance_module.HeldOutAccessError):
            PROVENANCE.authorize_donor(self.candidate, self.held_out)

    def test_quarantine_donor_rejected(self):
        with self.assertRaises(provenance_module.QuarantineError):
            PROVENANCE.authorize_donor(self.candidate, self.quarantine)

    def test_unknown_tree_rejected(self):
        with self.assertRaises(provenance_module.UnknownTreeError):
            PROVENANCE.authorize_donor(self.candidate, 'nope:not_a_tree')

    def test_non_context_donor_rejected(self):
        # A real train tree that is not in this candidate's context.
        allowed = set(self.context) | {self.candidate}
        other = next(cid for cid in PROVENANCE.train.eligible if cid not in allowed)
        with self.assertRaises(provenance_module.UnauthorizedDonorError):
            PROVENANCE.authorize_donor(self.candidate, other)

    def test_cross_partition_donor_rejected(self):
        validation = PROVENANCE.validation.eligible
        train_only = [c for c in validation
                      if c not in PROVENANCE.train.source_union]
        donor = train_only[0]
        # A validation candidate may never donate to a train candidate even if some
        # malformed provenance listed it: the allowlist is checked first, then the
        # partition rule.
        with self.assertRaises(provenance_module.ProvenanceViolation):
            PROVENANCE.authorize_donor(self.candidate, donor)

    def test_held_out_candidate_rejected(self):
        with self.assertRaises(provenance_module.HeldOutAccessError):
            PROVENANCE.partition_of_candidate(self.held_out)

    def test_boundary_quarantine_may_not_be_a_candidate(self):
        boundary = sorted(PROVENANCE.boundary_quarantine)[0]
        # A boundary-quarantine tree is a legal context donor but never a candidate.
        self.assertNotIn(boundary, PROVENANCE.train.eligible)
        self.assertNotIn(boundary, PROVENANCE.validation.eligible)
        self.assertNotIn(boundary, PROVENANCE.held_out_eligible)
        with self.assertRaises(provenance_module.CrossPartitionError):
            PROVENANCE.assert_not_boundary_candidate(boundary)
        with self.assertRaises(provenance_module.HeldOutAccessError):
            PROVENANCE.partition_of_candidate(boundary)

    def test_missing_provenance_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError, OSError)):
                provenance_module.load_gate_d_provenance(Path(tmp))

    def test_frozen_counts_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (provenance_module.PROVENANCE_FILE,
                         provenance_module.MANIFEST_FILE,
                         provenance_module.REMEDIATION_FILE):
                (root / name).write_bytes(
                    (provenance_module.REPORTS / name).read_bytes())
            payload = json.loads((root / provenance_module.PROVENANCE_FILE)
                                 .read_text(encoding='utf-8'))
            payload['train']['eligible'] = payload['train']['eligible'][:-1]
            (root / provenance_module.PROVENANCE_FILE).write_text(
                json.dumps(payload), encoding='utf-8')
            with self.assertRaises(provenance_module.ManifestMismatchError):
                provenance_module.load_gate_d_provenance(root)

    def test_alias_of_quarantined_tree_rejected(self):
        alias = sorted(PROVENANCE.high_containment_aliases)[0]
        owner = next(cid for cid, aliases in PROVENANCE.source_aliases_of.items()
                     if alias in aliases)
        with self.assertRaises(provenance_module.QuarantineError):
            PROVENANCE.assert_alias_clean(owner)

    def test_example_sources_accepts_candidate_and_own_context(self):
        PROVENANCE.assert_example_sources(self.candidate, self.context)

    def test_example_sources_rejects_quarantine_in_union(self):
        with self.assertRaises(provenance_module.ProvenanceViolation):
            PROVENANCE.assert_example_sources(self.candidate,
                                              [*self.context, self.quarantine])


class TrainingLock(unittest.TestCase):
    def test_lock_is_active(self):
        import gate_d_lock as lock
        self.assertFalse(lock.TRAINING_ALLOWED)
        state = lock.acceptance_state()
        self.assertFalse(state['training_allowed'])
        self.assertEqual(state['reason'], 'GATE_D_NOT_ACCEPTED')

    def test_require_training_allowed_raises(self):
        import gate_d_lock as lock
        with self.assertRaises(lock.TrainingLockedError):
            lock.require_training_allowed()

    def test_forged_marker_without_acceptance_is_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / lock.ACCEPTANCE_FILE
            path.write_text(json.dumps({'gate_d': 'PENDING'}), encoding='utf-8')
            self.assertFalse(lock.acceptance_state(Path(tmp))['training_allowed'])
            path.write_text(json.dumps({'gate_d': 'ACCEPTED',
                                        'training_allowed': False}),
                            encoding='utf-8')
            self.assertFalse(lock.acceptance_state(Path(tmp))['training_allowed'])
            path.write_text(json.dumps({'gate_d': 'ACCEPTED',
                                        'training_allowed': True,
                                        'generator_digest': 'a'}), encoding='utf-8')
            self.assertFalse(lock.acceptance_state(Path(tmp), 'b')['training_allowed'])
            self.assertTrue(lock.acceptance_state(Path(tmp), 'a')['training_allowed'])

    def test_trainers_refuse_without_acceptance(self):
        import train_pointnet
        import train_pointnet2
        self.assertEqual(train_pointnet.main(['--epochs', '1']), 3)
        self.assertEqual(train_pointnet2.main(['--epochs', '1']), 3)


class GeneratorSemantics(unittest.TestCase):
    def setUp(self):
        self.candidate = PROVENANCE.train.eligible[0]
        self.context = PROVENANCE.authorized_context(self.candidate)
        self.geometry = fixture_geometry(self.candidate, self.context, seed=3)

    def spec(self, example_class, severity='MEDIUM', seed=11):
        return gen.ExampleSpec(self.candidate, 'train', example_class, severity, seed)

    def test_all_classes_and_severities_generate(self):
        for example_class in gen.CLASSES:
            for severity in gen.SEVERITIES:
                example = gen.generate_example(self.spec(example_class, severity),
                                               self.geometry, PROVENANCE)
                self.assertGreater(example.points.shape[0], 0)
                self.assertEqual(example.membership.shape[0], example.points.shape[0])
                self.assertTrue(np.isfinite(example.points).all())

    def test_good_preserves_official_membership(self):
        example = gen.generate_example(self.spec(gen.GOOD), self.geometry, PROVENANCE)
        self.assertTrue(example.membership.all())
        self.assertEqual(example.label, gen.LABEL_PASS)
        self.assertEqual(example.donors, ())

    def test_defect_classes_are_review_required(self):
        for example_class in gen.DEFECT_CLASSES:
            example = gen.generate_example(self.spec(example_class), self.geometry,
                                           PROVENANCE)
            self.assertEqual(example.label, gen.LABEL_REVIEW)

    def test_merged_uses_real_context_trees(self):
        example = gen.generate_example(self.spec(gen.MERGED_MULTI_TREE),
                                       self.geometry, PROVENANCE)
        self.assertTrue(example.donors)
        for donor in example.donors:
            self.assertIn(donor, self.context)
        # Every donor point is real context geometry, not an invention.
        donor_points = np.concatenate(
            [t.points for t in self.geometry.context if t.canonical_tree_id in example.donors])
        source = np.unique(example.points, axis=0)
        self.assertTrue(np.isin(source, donor_points).any() or
                        np.isin(source, self.geometry.candidate.points).any())

    def test_contamination_adds_non_member_points(self):
        example = gen.generate_example(self.spec(gen.NEIGHBOR_CONTAMINATION),
                                       self.geometry, PROVENANCE)
        self.assertTrue(example.donors)
        self.assertLess(example.membership.mean(), 1.0)
        self.assertEqual(example.provenance['corruption']['donor_points'] > 0, True)

    def test_over_segmented_is_a_spatial_subset_of_the_candidate(self):
        example = gen.generate_example(self.spec(gen.OVER_SEGMENTED),
                                       self.geometry, PROVENANCE)
        detail = example.provenance['corruption']
        # Membership stays official (1.0): the defect is missing geometry, not foreign
        # geometry. The kept points must be a subset of the candidate's own points.
        self.assertTrue(example.membership.all())
        self.assertGreater(detail['dropped_points'], 0)
        unique = np.unique(example.points, axis=0)
        self.assertTrue(np.isin(unique, self.geometry.candidate.points).all())
        self.assertLess(unique.shape[0], self.geometry.candidate.points.shape[0])

    def test_crown_incomplete_removes_top_and_refills_with_real_geometry(self):
        example = gen.generate_example(self.spec(gen.CROWN_INCOMPLETE),
                                       self.geometry, PROVENANCE)
        detail = example.provenance['corruption']
        self.assertGreater(detail['removed_points'], 0)
        self.assertGreater(detail['fill_points'], 0)
        self.assertLess(example.membership.mean(), 1.0)
        member_max = example.points[example.membership, 2].max()
        self.assertLessEqual(member_max, detail['z_cut'] + 1e-6)

    def test_stem_missing_removes_lower_region_and_refills(self):
        example = gen.generate_example(self.spec(gen.STEM_MISSING),
                                       self.geometry, PROVENANCE)
        detail = example.provenance['corruption']
        self.assertGreater(detail['removed_points'], 0)
        self.assertGreater(detail['fill_points'], 0)

    def test_noise_is_dominated_by_real_non_member_geometry(self):
        example = gen.generate_example(self.spec(gen.NOISE), self.geometry, PROVENANCE)
        detail = example.provenance['corruption']
        self.assertGreater(detail['noise_points'], detail['surviving_candidate_points'])
        self.assertLess(example.membership.mean(), 0.5)
        # The noise points come from the background bundle, which is real geometry.
        source = np.unique(example.points, axis=0)
        self.assertTrue(np.isin(source, self.geometry.background).any())

    def test_no_class_uses_invented_gaussian_geometry(self):
        """Every generated coordinate must come from the real geometry bundle."""
        allowed = np.unique(np.concatenate([
            self.geometry.candidate.points,
            *[t.points for t in self.geometry.context],
            self.geometry.background]), axis=0)
        for example_class in gen.DEFECT_CLASSES:
            example = gen.generate_example(self.spec(example_class), self.geometry,
                                           PROVENANCE)
            self.assertTrue(np.isin(example.points, allowed).all(), example_class)

    def test_severity_controls_the_drawn_fraction(self):
        """Severity scales the drawn fraction; the draw itself stays class-independent."""
        drawn = {}
        for severity in gen.SEVERITIES:
            values = [gen.generate_example(
                self.spec(gen.OVER_SEGMENTED, severity, seed=seed), self.geometry,
                PROVENANCE).provenance['corruption']['drawn_fraction']
                for seed in range(40)]
            drawn[severity] = float(np.mean(values))
        self.assertGreater(drawn[gen.HIGH], drawn[gen.MEDIUM])
        self.assertGreater(drawn[gen.MEDIUM], drawn[gen.LOW])

    def test_unknown_class_and_severity_rejected(self):
        with self.assertRaises(gen.GeneratorError):
            gen.ExampleSpec(self.candidate, 'train', 'NOT_A_CLASS', 'LOW', 1)
        with self.assertRaises(gen.GeneratorError):
            gen.ExampleSpec(self.candidate, 'train', gen.GOOD, 'EXTREME', 1)
        with self.assertRaises(gen.GeneratorError):
            gen.ExampleSpec(self.candidate, 'held_out', gen.GOOD, 'LOW', 1)

    def test_geometry_mismatch_rejected(self):
        other = PROVENANCE.train.eligible[1]
        with self.assertRaises(gen.GeneratorError):
            gen.generate_example(gen.ExampleSpec(other, 'train', gen.GOOD, 'LOW', 1),
                                 self.geometry, PROVENANCE)


class Determinism(unittest.TestCase):
    def setUp(self):
        self.candidate = PROVENANCE.train.eligible[0]
        self.context = PROVENANCE.authorized_context(self.candidate)
        self.geometry = fixture_geometry(self.candidate, self.context, seed=5)

    def test_same_spec_same_bytes(self):
        spec = gen.ExampleSpec(self.candidate, 'train', gen.MERGED_MULTI_TREE,
                               'HIGH', 99)
        first = gen.generate_example(spec, self.geometry, PROVENANCE)
        second = gen.generate_example(spec, self.geometry, PROVENANCE)
        self.assertTrue(np.array_equal(first.points, second.points))
        self.assertTrue(np.array_equal(first.membership, second.membership))
        self.assertEqual(first.provenance, second.provenance)

    def test_different_seed_changes_the_example(self):
        low = gen.generate_example(gen.ExampleSpec(self.candidate, 'train',
                                                   gen.OVER_SEGMENTED, 'HIGH', 1),
                                   self.geometry, PROVENANCE)
        high = gen.generate_example(gen.ExampleSpec(self.candidate, 'train',
                                                    gen.OVER_SEGMENTED, 'HIGH', 2),
                                    self.geometry, PROVENANCE)
        self.assertFalse(np.array_equal(low.points, high.points))

    def test_spec_round_trips_through_json(self):
        spec = gen.ExampleSpec(self.candidate, 'train', gen.NOISE, 'LOW', 7)
        self.assertEqual(gen.ExampleSpec.from_row(spec.as_row()), spec)

    def test_enumerate_specs_is_stable(self):
        first = gen.enumerate_specs([self.candidate], 'train', repeats=2)
        second = gen.enumerate_specs([self.candidate], 'train', repeats=2)
        self.assertEqual([s.key for s in first], [s.key for s in second])
        self.assertEqual(len(first), 7 * 3 * 2)


class SamplerContract(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.points = rng.random((20000, 3)).astype(np.float32)
        self.membership = rng.random(20000) > 0.3

    def test_primary_and_robustness_sizes(self):
        for target in (sampler.POINTS_PRIMARY, sampler.POINTS_ROBUSTNESS):
            sample = sampler.sample(self.points, self.membership, 1, target)
            self.assertEqual(sample.features.shape, (target, sampler.FEATURE_COUNT))
            self.assertEqual(sample.features.shape[1], 4)

    def test_feature_channels_are_exactly_declared(self):
        sample = sampler.sample(self.points, self.membership, 1)
        self.assertEqual(tuple(sampler.FEATURES),
                         ('local_x', 'local_y', 'local_z', 'candidate_membership'))
        self.assertTrue(np.isin(sample.features[:, 3], [0.0, 1.0]).all())

    def test_local_normalisation_preserves_relative_geometry(self):
        sample = sampler.sample(self.points, self.membership, 3)
        local, centre, scale = sampler.local_normalise(self.points)
        original = self.points[sample.indices]
        expected = (original - centre) / scale
        self.assertTrue(np.allclose(sample.features[:, :3], expected, atol=1e-5))

    def test_sampler_has_no_label_parameter(self):
        import inspect
        signature = inspect.signature(sampler.sample)
        self.assertEqual(list(signature.parameters),
                         ['points', 'membership', 'spec_seed', 'points_target'])

    def test_small_cloud_is_upsampled_without_new_coordinates(self):
        tiny = self.points[:100]
        sample = sampler.sample(tiny, self.membership[:100], 1)
        self.assertEqual(sample.points, sampler.POINTS_PRIMARY)
        self.assertTrue(np.isin(sample.features[:, :3],
                                sampler.local_normalise(tiny)[0]).all())

    def test_deterministic_for_same_seed(self):
        first = sampler.sample(self.points, self.membership, 42)
        second = sampler.sample(self.points, self.membership, 42)
        self.assertTrue(np.array_equal(first.features, second.features))
        self.assertTrue(np.array_equal(first.indices, second.indices))

    def test_spatial_coverage_is_high(self):
        sample = sampler.sample(self.points, self.membership, 7)
        self.assertGreater(sampler.spatial_coverage(sample, self.points), 0.5)

    def test_misaligned_membership_rejected(self):
        with self.assertRaises(sampler.SamplerError):
            sampler.sample(self.points, self.membership[:10], 1)

    def test_empty_cloud_rejected(self):
        with self.assertRaises(sampler.SamplerError):
            sampler.sample(np.zeros((0, 3), np.float32), np.zeros(0, bool), 1)

    def test_bad_target_rejected(self):
        with self.assertRaises(sampler.SamplerError):
            sampler.sample(self.points, self.membership, 1, 0)


class DatasetContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_root = Path(tempfile.mkdtemp(prefix='gate_d_ds_', dir=str(TMP_ROOT)))
        cls.candidates = list(PROVENANCE.train.eligible[:2])

    def make(self):
        return dataset_module.GateDDataset(
            dataset_module.DatasetConfig(partition='train', repeats=1),
            PROVENANCE, candidate_ids=self.candidates, cache_root=self.cache_root)

    @unittest.skipUnless(HAS_REAL_DATA, 'verified extraction root not present')
    def test_dataset_is_deterministic_and_covers_all_classes(self):
        first = self.make()
        second = self.make()
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(len(first), len(self.candidates) * 7 * 3)
        for index in (0, 5, 20):
            self.assertTrue(np.array_equal(first[index].features,
                                           second[index].features))
        balance = first.class_balance()
        self.assertEqual(len(balance), 21)

    @unittest.skipUnless(HAS_REAL_DATA, 'verified extraction root not present')
    def test_held_out_candidates_are_refused(self):
        with self.assertRaises(provenance_module.HeldOutAccessError):
            dataset_module.GateDDataset(
                dataset_module.DatasetConfig(partition='train'), PROVENANCE,
                candidate_ids=[PROVENANCE.held_out_eligible[0]],
                cache_root=self.cache_root)

    def test_held_out_partition_is_refused(self):
        with self.assertRaises(dataset_module.DatasetError):
            dataset_module.DatasetConfig(partition='held_out')

    def test_restart_state_is_stable(self):
        first = self.make()
        self.assertEqual(first.restart_state()['digest'], first.digest())

    def test_manifest_carries_full_provenance(self):
        dataset = self.make()
        row = dataset.manifest()[0]
        for field in ('candidate_id', 'partition', 'class', 'severity', 'seed', 'key'):
            self.assertIn(field, row)


class MetricsAndThreshold(unittest.TestCase):
    def test_metrics_on_a_known_confusion(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        row = metrics.metrics(labels, scores, 0.5)
        self.assertEqual(row['CONFUSION_MATRIX']['tn_pass_pass'], 1)
        self.assertEqual(row['CONFUSION_MATRIX']['fp_pass_as_review'], 1)
        self.assertEqual(row['CONFUSION_MATRIX']['fn_review_as_pass'], 1)
        self.assertEqual(row['CONFUSION_MATRIX']['tp_review_review'], 1)
        self.assertAlmostEqual(row['BAD_RECALL'], 0.5)
        self.assertAlmostEqual(row['GOOD_FALSE_REVIEW'], 0.5)
        self.assertAlmostEqual(row['BALANCED_ACCURACY'], 0.5)

    def test_perfect_separation(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.01, 0.02, 0.98, 0.99])
        row = metrics.metrics(labels, scores, 0.5)
        self.assertAlmostEqual(row['BAD_RECALL'], 1.0)
        self.assertAlmostEqual(row['GOOD_FALSE_REVIEW'], 0.0)
        self.assertAlmostEqual(row['BALANCED_ACCURACY'], 1.0)

    def test_threshold_selection_respects_the_budget(self):
        rng = np.random.default_rng(0)
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.concatenate([rng.uniform(0, 0.6, 100), rng.uniform(0.4, 1, 100)])
        selected = metrics.select_threshold(labels, scores, 'validation',
                                            good_false_review_budget=0.2)
        self.assertTrue(selected['budget_satisfied'])
        self.assertLessEqual(selected['GOOD_FALSE_REVIEW'], 0.2)

    def test_threshold_selection_refuses_held_out(self):
        labels = np.array([0, 1])
        scores = np.array([0.1, 0.9])
        for partition in ('held_out', 'test', 'train_held_out'):
            with self.assertRaises(metrics.ThresholdError):
                metrics.select_threshold(labels, scores, partition)

    def test_three_way_mapping_is_between_pass_and_review(self):
        decisions = metrics.three_way_decisions(np.array([0.0, 0.5, 1.0]), 0.5, 0.1)
        self.assertEqual(list(decisions),
                         [metrics.DECISION_PASS, metrics.DECISION_UNCERTAIN,
                          metrics.DECISION_REVIEW])

    def test_frozen_threshold_requires_model_selection_partition(self):
        with self.assertRaises(metrics.ThresholdError):
            metrics.freeze_threshold({'threshold': 0.5, 'partition': 'held_out'},
                                     'd', 'c', 1)
        frozen = metrics.freeze_threshold(
            {'threshold': 0.5, 'partition': 'validation'}, 'd', 'c', 1)
        self.assertTrue(frozen['frozen'])
        self.assertFalse(frozen['held_out_used'])


class FingerprintChecks(unittest.TestCase):
    def _record(self, example_class, severity, source_points, sampled=4096,
                fraction=0.2, drawn=0.2, axis=None, membership=0.5):
        return fingerprint.ExampleRecord(
            key=f'{example_class}|{severity}|{source_points}', candidate_id='c1',
            partition='train', example_class=example_class, severity=severity,
            label='PASS' if example_class == 'GOOD' else 'REVIEW_REQUIRED',
            source_points=source_points, raw_source_points=source_points,
            sampled_points=sampled, corruption_fraction=fraction,
            drawn_fraction=drawn, corruption_axis=axis,
            membership_source=membership, membership_sampled=membership,
            feature_mean=(0.0, 0.0, 0.0, membership),
            feature_std=(1.0, 1.0, 1.0, 0.1), spatial_coverage=0.9)

    def test_fixed_sample_size_is_enforced(self):
        records = [self._record(c, s, 50000) for c in gen.CLASSES
                   for s in gen.SEVERITIES]
        report = fingerprint.analyse(records, seed=1)
        check = next(c for c in report['checks']
                     if c['check'] == 'point_count_fingerprint_sampled')
        self.assertTrue(check['pass'])

    def test_varying_sample_size_is_critical(self):
        records = [self._record(c, s, 50000, sampled=4096 + i)
                   for i, (c, s) in enumerate((c, s) for c in gen.CLASSES
                                              for s in gen.SEVERITIES)]
        report = fingerprint.analyse(records, seed=1)
        check = next(c for c in report['checks']
                     if c['check'] == 'point_count_fingerprint_sampled')
        self.assertTrue(check['critical'])
        self.assertIn('point_count_fingerprint_sampled',
                      report['CRITICAL_FINGERPRINTS'])

    def test_class_specific_fraction_is_detected(self):
        records = []
        for index, example_class in enumerate(gen.DEFECT_CLASSES):
            for severity in gen.SEVERITIES:
                records.append(self._record(example_class, severity, 50000,
                                            drawn=0.05 + 0.05 * index))
        report = fingerprint.analyse(records, seed=1)
        check = next(c for c in report['checks']
                     if c['check'] == 'fixed_fraction_fingerprint')
        self.assertTrue(check['critical'])

    def test_identical_class_sets_pass_identifier_check(self):
        records = [self._record(c, s, 50000) for c in gen.CLASSES
                   for s in gen.SEVERITIES]
        report = fingerprint.analyse(records, seed=1)
        check = next(c for c in report['checks']
                     if c['check'] == 'metadata_leakage_identifiers')
        self.assertTrue(check['pass'])


class GeneratorDigest(unittest.TestCase):
    def test_digest_is_stable_and_hex(self):
        import gate_d_diagnostic
        first = gate_d_diagnostic.generator_digest()
        second = gate_d_diagnostic.generator_digest()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)


class GeometryAuthorization(unittest.TestCase):
    def test_background_trees_exclude_quarantine_and_held_out(self):
        candidate = PROVENANCE.train.eligible[0]
        chosen = geometry.allowed_background_trees(PROVENANCE, candidate)
        for tree_id in chosen:
            self.assertNotIn(tree_id, PROVENANCE.high_containment_quarantine)
            self.assertNotEqual(PROVENANCE.partition_of[tree_id], 'held_out')
            self.assertNotIn(tree_id, PROVENANCE.authorized_context(candidate))

    def test_background_trees_are_deterministic(self):
        candidate = PROVENANCE.train.eligible[0]
        self.assertEqual(geometry.allowed_background_trees(PROVENANCE, candidate),
                         geometry.allowed_background_trees(PROVENANCE, candidate))

    def test_isolated_candidate_still_gets_background(self):
        """Every train/validation candidate must be able to generate NOISE."""
        missing = []
        for partition in ('train', 'validation'):
            for candidate in PROVENANCE.candidates(partition):
                if not geometry.allowed_background_trees(PROVENANCE, candidate):
                    missing.append(candidate)
        self.assertEqual(missing, [])

    @unittest.skipUnless(HAS_REAL_DATA, 'verified extraction root not present')
    def test_real_batch_loads_and_is_authorised(self):
        candidates = list(PROVENANCE.train.eligible[:2])
        loaded = geometry.build_candidate_geometry_batch(candidates, PROVENANCE)
        for candidate in candidates:
            bundle = loaded[candidate]
            self.assertEqual(bundle.candidate_id, candidate)
            self.assertTrue(bundle.candidate.membership.all())
            for tree in bundle.context:
                self.assertFalse(tree.membership.any())
            self.assertEqual(bundle.context_ids,
                             PROVENANCE.authorized_context(candidate))


if __name__ == '__main__':
    unittest.main()
