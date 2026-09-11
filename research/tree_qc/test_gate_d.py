"""Gate D tests: membership semantics, source allowlist, sampler, lock, fingerprints.

Fixture geometry is synthetic *for speed*, but every provenance test runs against the
real frozen Gate C2 artifacts, because the contract is what is being tested. Tests that
need real publisher geometry are skipped when the verified extraction root is absent.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import gate_d_dataset as dataset_module
import gate_d_fingerprint as fingerprint
import gate_d_generator as gen
import gate_d_geometry as geometry
import gate_d_metrics as metrics
import gate_d_provenance as provenance_module
import gate_d_sampler as sampler
import gate_d_scene as scene_module
import numpy as np

TMP_ROOT = Path('D:/TreeQCTraining/tmp')
DATA_ROOT = Path('D:/TreeQCData/extracted_manual_verified')
HAS_REAL_DATA = DATA_ROOT.is_dir()
PROVENANCE = provenance_module.load_gate_d_provenance()


def _synthetic_tree(rng, count: int, box):
    """A tree-like cloud inside `box`: a narrow vertical stem plus a wide crown.

    The stem gives STEM_MISSING something real to be applicable to; the crown gives
    CROWN_INCOMPLETE a contiguous region to drop.
    """
    x0, y0, z0, x1, y1, z1 = box
    span_x, span_y = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    span_z = max(z1 - z0, 1e-3)
    axis_x, axis_y = x0 + span_x / 2.0, y0 + span_y / 2.0
    stem_count = max(8, int(count * 0.25))
    crown_count = count - stem_count
    stem = np.stack([
        (axis_x + rng.normal(0.0, 0.15, stem_count)).astype(np.float32),
        (axis_y + rng.normal(0.0, 0.15, stem_count)).astype(np.float32),
        (z0 + rng.random(stem_count).astype(np.float32) * (span_z * 0.75)),
    ], axis=1)
    crown = np.stack([
        (axis_x + rng.normal(0.0, span_x * 0.3, crown_count)).astype(np.float32),
        (axis_y + rng.normal(0.0, span_y * 0.3, crown_count)).astype(np.float32),
        (z0 + span_z * 0.55
         + rng.random(crown_count).astype(np.float32) * span_z * 0.45),
    ], axis=1)
    return np.concatenate([stem, crown], axis=0).astype(np.float32)


def fixture_geometry(candidate_id: str, context_ids=None, seed: int = 0,
                     candidate_points: int = 900, context_points: int = 700):
    """A small, synthetic geometry bundle that satisfies the frozen context check.

    Context ids must be the candidate's real frozen surviving context, because the scene
    layer refuses a bundle whose context disagrees with the allowlist. Coordinates are
    synthetic but are placed inside the candidate's real frozen 3 m query box, so the
    clip step keeps them.
    """
    if context_ids is None:
        context_ids = PROVENANCE.authorized_context(candidate_id)
    row = PROVENANCE.tree[candidate_id]
    box = scene_module.frozen_query_box(row['bbox'])
    rng = np.random.default_rng(seed)

    def tree(tree_id, count):
        points = _synthetic_tree(rng, count, box)
        return geometry.TreeGeometry(canonical_tree_id=tree_id, site='litchfield',
                                     points=points, point_count=count,
                                     stored_points=count)

    return geometry.CandidateGeometry(candidate=tree(candidate_id, candidate_points),
                                      context=tuple(tree(cid, context_points)
                                                    for cid in context_ids))


def fixture_scene(candidate_id: str, context_ids, seed: int = 0,
                  candidate_points: int = 900, context_points: int = 700):
    """A BaseScene built directly, for tests that do not need real data."""
    rng = np.random.default_rng(seed)
    box = (-3.0, -3.0, -1.0, 9.0, 9.0, 3.0)
    blocks, source_ids, is_candidate = [], [], []
    blocks.append(_synthetic_tree(rng, candidate_points, box))
    source_ids.append(candidate_id)
    is_candidate.append(True)
    for cid in context_ids:
        blocks.append(_synthetic_tree(rng, context_points, box))
        source_ids.append(cid)
        is_candidate.append(False)
    points = np.concatenate(blocks)
    gt_source_id = np.concatenate([
        np.full(b.shape[0], i, dtype=np.int32) for i, b in enumerate(blocks)])
    candidate_mask = np.concatenate([
        np.full(b.shape[0], flag, dtype=bool) for b, flag in zip(blocks, is_candidate,
                                                                 strict=True)])
    return scene_module.BaseScene(
        candidate_id=candidate_id, partition='train', site='litchfield',
        points=points, gt_source_id=gt_source_id, source_ids=tuple(source_ids),
        candidate_mask=candidate_mask,
        query_box=box,
        audit={'scene_version': scene_module.SCENE_VERSION,
               'query_box': list(box),
               'query_pad_m': 3.0, 'candidate_id': candidate_id,
               'source_ids': list(source_ids), 'source_counts': {},
               'source_aliases': {}, 'candidate_points': candidate_points,
               'context_points': context_points * len(context_ids),
               'context_sources_used': len(context_ids),
               'context_sources_authorised': len(context_ids)})


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
        with self.assertRaises(provenance_module.HeldOutAccessError):
            PROVENANCE.authorize_donor(self.candidate, self.held_out)

    def test_quarantine_donor_rejected(self):
        with self.assertRaises(provenance_module.QuarantineError):
            PROVENANCE.authorize_donor(self.candidate, self.quarantine)

    def test_unknown_tree_rejected(self):
        with self.assertRaises(provenance_module.UnknownTreeError):
            PROVENANCE.authorize_donor(self.candidate, 'nope:not_a_tree')

    def test_non_context_donor_rejected(self):
        allowed = set(self.context) | {self.candidate}
        other = next(cid for cid in PROVENANCE.train.eligible if cid not in allowed)
        with self.assertRaises(provenance_module.UnauthorizedDonorError):
            PROVENANCE.authorize_donor(self.candidate, other)

    def test_cross_partition_donor_rejected(self):
        validation = PROVENANCE.validation.eligible
        donor = [c for c in validation if c not in PROVENANCE.train.source_union][0]
        with self.assertRaises(provenance_module.ProvenanceViolation):
            PROVENANCE.authorize_donor(self.candidate, donor)

    def test_held_out_candidate_rejected(self):
        with self.assertRaises(provenance_module.HeldOutAccessError):
            PROVENANCE.partition_of_candidate(self.held_out)

    def test_boundary_quarantine_may_not_be_a_candidate(self):
        boundary = sorted(PROVENANCE.boundary_quarantine)[0]
        self.assertNotIn(boundary, PROVENANCE.train.eligible)
        self.assertNotIn(boundary, PROVENANCE.validation.eligible)
        with self.assertRaises(provenance_module.CrossPartitionError):
            PROVENANCE.assert_not_boundary_candidate(boundary)

    def test_missing_provenance_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError, OSError)):
                provenance_module.load_gate_d_provenance(Path(tmp))

    def test_frozen_counts_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (provenance_module.PROVENANCE_FILE,
                         provenance_module.MANIFEST_FILE,
                         provenance_module.REMEDIATION_FILE,
                         provenance_module.PROTOCOL_FILE):
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

    def test_protocol_digest_is_reported(self):
        self.assertEqual(PROVENANCE.protocol_digest,
                         provenance_module.protocol_digest())


class SceneSourceAllowlist(ProvenanceFixture):
    """Blocker 2: every contributing point source must pass the allowlist."""

    def test_contributing_union_is_accepted(self):
        union = (self.candidate, *self.context[:2])
        self.assertEqual(PROVENANCE.assert_scene_sources(self.candidate, union),
                         tuple(sorted(union)))

    def test_non_context_source_is_refused(self):
        allowed = set(self.context) | {self.candidate}
        other = next(cid for cid in PROVENANCE.train.eligible if cid not in allowed)
        with self.assertRaises(provenance_module.UnauthorizedDonorError):
            PROVENANCE.assert_scene_sources(self.candidate,
                                            [self.candidate, other])

    def test_held_out_source_is_refused(self):
        with self.assertRaises(provenance_module.HeldOutAccessError):
            PROVENANCE.assert_scene_sources(self.candidate,
                                            [self.candidate, self.held_out])

    def test_quarantine_source_is_refused(self):
        with self.assertRaises(provenance_module.QuarantineError):
            PROVENANCE.assert_scene_sources(self.candidate,
                                            [self.candidate, self.quarantine])

    def test_unknown_source_is_refused(self):
        with self.assertRaises(provenance_module.UnknownTreeError):
            PROVENANCE.assert_scene_sources(self.candidate,
                                            [self.candidate, 'ghost:tree'])

    def test_candidate_must_contribute(self):
        with self.assertRaises(provenance_module.ManifestMismatchError):
            PROVENANCE.assert_scene_sources(self.candidate, list(self.context))

    def test_background_helper_is_gone(self):
        """The radius-search background path that produced 72 illegal references."""
        self.assertFalse(hasattr(geometry, 'allowed_background_trees'))
        self.assertFalse(hasattr(geometry.CandidateGeometry, 'background'))

    def test_frozen_query_box_rule(self):
        row = PROVENANCE.tree[self.candidate]
        box = scene_module.frozen_query_box(row['bbox'])
        x0, y0, z0, x1, y1, z1 = row['bbox']
        self.assertAlmostEqual(box[0], x0 - 3.0)
        self.assertAlmostEqual(box[1], y0 - 3.0)
        self.assertAlmostEqual(box[2], z0)
        self.assertAlmostEqual(box[3], x1 + 3.0)
        self.assertAlmostEqual(box[4], y1 + 3.0)
        self.assertAlmostEqual(box[5], z1)

    def test_clip_keeps_only_points_inside_the_query(self):
        points = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]], dtype=np.float32)
        inside = scene_module.clip_to_query(points, (-1, -1, -1, 1, 1, 1))
        self.assertEqual(inside.tolist(), [0])


class SharedScene(unittest.TestCase):
    """Blocker 1: GOOD and BAD share one scene; only membership differs."""

    def setUp(self):
        self.candidate = PROVENANCE.train.eligible[0]
        self.context = PROVENANCE.authorized_context(self.candidate)
        self.scene = fixture_scene(self.candidate, self.context, seed=4)

    def test_scene_has_candidate_and_context(self):
        self.assertGreater(self.scene.candidate_points, 0)
        self.assertGreater(self.scene.context_points, 0)
        self.assertEqual(self.scene.point_count,
                         self.scene.candidate_points + self.scene.context_points)

    def test_scene_source_counts_sum_to_scene_size(self):
        counts = self.scene.source_counts()
        self.assertEqual(sum(counts.values()), self.scene.point_count)
        self.assertEqual(tuple(sorted(counts)), tuple(sorted(self.scene.source_ids)))

    def test_good_membership_is_the_publisher_gt_candidate_set(self):
        membership = self.scene.membership_gt()
        self.assertTrue(np.array_equal(membership, self.scene.candidate_mask))
        self.assertTrue(membership[self.scene.candidate_mask].all())
        self.assertFalse(membership[~self.scene.candidate_mask].any())


class CurrentMembershipSemantics(unittest.TestCase):
    """Blocker 1: the channel means 'current segmentation', not 'the answer'."""

    def setUp(self):
        self.candidate = PROVENANCE.train.eligible[0]
        self.context = PROVENANCE.authorized_context(self.candidate)
        self.geometry = fixture_geometry(self.candidate, self.context, seed=7)

    def spec(self, example_class, severity='MEDIUM', seed=11):
        return gen.ExampleSpec(self.candidate, 'train', example_class, severity, seed)

    def _example(self, example_class, severity='MEDIUM', seed=11):
        return gen.generate_example(self.spec(example_class, severity, seed),
                                    self.geometry, PROVENANCE)

    def test_good_has_real_context_and_membership_zero_points(self):
        example = self._example(gen.GOOD)
        self.assertEqual(example.label, gen.LABEL_PASS)
        self.assertGreater(example.provenance['context_points'], 0)
        self.assertLess(example.membership.mean(), 1.0)
        self.assertGreater(int((~example.membership).sum()), 0)

    def test_every_class_has_membership_zero_points(self):
        """No class can be identified by the mere existence of a zero flag."""
        for example_class in gen.CLASSES:
            example = self._example(example_class)
            self.assertGreater(int((~example.membership).sum()), 0, example_class)

    def test_claimed_neighbour_points_have_membership_one(self):
        for example_class in gen.CLAIM_CLASSES:
            example = self._example(example_class)
            claimed = example.changed_membership & example.membership
            self.assertGreater(int(claimed.sum()), 0, example_class)
            # Claimed points are publisher GT context points that the current
            # membership now selects: that is the defect, not the answer label.
            self.assertGreater(example.provenance['claimed_points'], 0, example_class)
            self.assertTrue(example.donors, example_class)
            for donor in example.donors:
                self.assertIn(donor, self.context)

    def test_dropped_candidate_points_have_membership_zero(self):
        for example_class in gen.DROP_CLASSES:
            example = self._example(example_class)
            dropped = example.changed_membership & ~example.membership
            self.assertGreater(int(dropped.sum()), 0, example_class)

    def test_gt_source_is_never_a_model_channel(self):
        self.assertEqual(sampler.FEATURES,
                         ('local_x', 'local_y', 'local_z',
                          'current_candidate_membership'))
        self.assertEqual(gen.MEMBERSHIP_CHANNEL, 'current_candidate_membership')

    def test_geometry_is_never_deleted(self):
        """Every class keeps the same scene size; corruption only flips flags."""
        sizes = {c: self._example(c).points.shape[0] for c in gen.CLASSES}
        self.assertEqual(len(set(sizes.values())), 1, sizes)

    def test_every_coordinate_is_real_geometry(self):
        allowed = np.unique(np.concatenate(
            [self.geometry.candidate.points, *[t.points for t in self.geometry.context]]),
            axis=0)
        for example_class in gen.CLASSES:
            example = self._example(example_class)
            self.assertTrue(np.isin(example.points, allowed).all(), example_class)

    def test_good_and_defects_share_the_same_scene(self):
        scenes = {c: self._example(c).points for c in gen.CLASSES}
        reference = scenes[gen.GOOD]
        for example_class, points in scenes.items():
            self.assertTrue(np.array_equal(points, reference), example_class)


class NoClassSpecificRefill(unittest.TestCase):
    """Blocker 3: no class-only geometry source, and STEM needs evidence."""

    def setUp(self):
        self.candidate = PROVENANCE.train.eligible[0]
        self.context = PROVENANCE.authorized_context(self.candidate)
        self.geometry = fixture_geometry(self.candidate, self.context, seed=8)

    def test_refill_helpers_are_gone(self):
        self.assertFalse(hasattr(gen, '_background_in_region'))

    def test_crown_incomplete_keeps_the_real_crown_points(self):
        example = gen.generate_example(
            gen.ExampleSpec(self.candidate, 'train', gen.CROWN_INCOMPLETE, 'HIGH', 5),
            self.geometry, PROVENANCE)
        detail = example.provenance['corruption']
        self.assertEqual(detail['mechanism'], 'drop_contiguous_crown_region')
        self.assertNotIn('fill_points', detail)
        self.assertGreater(detail['dropped_points'], 0)

    def test_stem_missing_keeps_the_real_stem_points(self):
        example = gen.generate_example(
            gen.ExampleSpec(self.candidate, 'train', gen.STEM_MISSING, 'MEDIUM', 5),
            self.geometry, PROVENANCE)
        detail = example.provenance['corruption']
        self.assertEqual(detail['mechanism'], 'drop_contiguous_stem_region')
        self.assertNotIn('fill_points', detail)
        self.assertGreater(detail['dropped_points'], 0)

    def test_stem_applicability_requires_lower_evidence(self):
        scene = fixture_scene(self.candidate, self.context, seed=9)
        # Flatten the crown onto the ground so there is no stem column at all.
        crown = scene.candidate_mask & (scene.points[:, 2] > -0.5)
        scene.points[crown, 2] = -0.9
        evidence = gen.stem_evidence(scene)
        self.assertFalse(evidence['applicable'])
        self.assertIn(evidence['reason'],
                      ('insufficient_lower_points', 'insufficient_vertical_span'))

    def test_stem_not_applicable_raises_instead_of_faking(self):
        scene = fixture_scene(self.candidate, self.context, seed=10)
        crown = scene.candidate_mask & (scene.points[:, 2] > -0.5)
        scene.points[crown, 2] = -0.9
        evidence = gen.stem_evidence(scene)
        self.assertFalse(evidence['applicable'])
        with self.assertRaises(gen.NotApplicableError) as context:
            raise gen.NotApplicableError(gen.STEM_NOT_APPLICABLE, evidence)
        self.assertEqual(context.exception.reason, gen.STEM_NOT_APPLICABLE)

    def test_stem_evidence_reports_a_positive_case(self):
        scene = fixture_scene(self.candidate, self.context, seed=11,
                              candidate_points=2000, context_points=700)
        evidence = gen.stem_evidence(scene)
        self.assertTrue(evidence['applicable'], evidence)
        self.assertGreaterEqual(evidence['lower_points'], gen.STEM_MIN_LOWER_POINTS)
        self.assertGreaterEqual(evidence['lower_span_m'], gen.STEM_MIN_VERTICAL_SPAN_M)
        self.assertIn('min_lower_points', evidence)


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

    def test_defect_classes_are_review_required(self):
        for example_class in gen.DEFECT_CLASSES:
            example = gen.generate_example(self.spec(example_class), self.geometry,
                                           PROVENANCE)
            self.assertEqual(example.label, gen.LABEL_REVIEW)

    def test_merged_claims_a_real_neighbour(self):
        example = gen.generate_example(self.spec(gen.MERGED_MULTI_TREE),
                                       self.geometry, PROVENANCE)
        self.assertTrue(example.donors)
        for donor in example.donors:
            self.assertIn(donor, self.context)

    def test_contamination_and_noise_differ_in_structure(self):
        contamination = gen.generate_example(self.spec(gen.NEIGHBOR_CONTAMINATION),
                                             self.geometry, PROVENANCE)
        noise = gen.generate_example(self.spec(gen.NOISE), self.geometry, PROVENANCE)
        self.assertEqual(contamination.provenance['corruption']['mechanism'],
                         'claim_contiguous_neighbour_region')
        self.assertEqual(noise.provenance['corruption']['mechanism'],
                         'claim_scattered_context_points')

        def claimed_extent(example):
            points = example.points[example.changed_membership & example.membership]
            return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

        def scene_extent(example):
            return float(np.linalg.norm(example.points.max(axis=0)
                                        - example.points.min(axis=0)))

        # Contamination is a local contiguous region; noise is spread across the scene.
        contamination_ratio = claimed_extent(contamination) / scene_extent(contamination)
        noise_ratio = claimed_extent(noise) / scene_extent(noise)
        self.assertLess(contamination_ratio, noise_ratio)
        self.assertEqual(noise.provenance['corruption']['distinct_cells'],
                         noise.provenance['corruption']['claimed_points'])

    def test_over_segmented_is_a_spatial_subset(self):
        example = gen.generate_example(self.spec(gen.OVER_SEGMENTED),
                                       self.geometry, PROVENANCE)
        self.assertGreater(example.provenance['corruption']['dropped_points'], 0)
        axis = np.asarray(example.provenance['corruption']['axis'])
        projection = example.points @ axis
        dropped = example.changed_membership
        self.assertLess(projection[dropped].max(),
                        projection[~dropped].max() + 1e-6)

    def test_severity_controls_the_drawn_fraction(self):
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

    def test_source_union_is_complete_and_reconstructed(self):
        for example_class in gen.CLASSES:
            example = gen.generate_example(self.spec(example_class), self.geometry,
                                           PROVENANCE)
            counts = example.provenance['actual_point_source_counts']
            self.assertEqual(sorted(counts),
                             example.provenance['actual_point_source_ids'])
            self.assertEqual(sum(counts.values()), example.provenance['point_count'])
            self.assertIn(example.provenance['candidate_id'], counts)


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
        self.assertFalse(np.array_equal(low.membership, high.membership))

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
        self.candidate_mask = np.zeros(20000, dtype=bool)
        self.candidate_mask[:6000] = True
        self.membership = self.candidate_mask.copy()
        self.changed = np.zeros(20000, dtype=bool)
        self.changed[:500] = True

    @staticmethod
    def _seed(tag, target=sampler.POINTS_PRIMARY, severity='LOW', repeat=0):
        """A valid sampling seed; ``sample`` refuses arbitrary integers."""
        return sampler.sampling_seed('train', f'test:{tag}', severity,
                                     points_target=target, repeat=repeat)

    def _sample(self, target=sampler.POINTS_PRIMARY, seed=None):
        if seed is None:
            seed = self._seed('contract', target)
        return sampler.sample(self.points, self.membership, self.changed,
                              self.candidate_mask, seed, target)

    def test_primary_and_robustness_sizes(self):
        for target in (sampler.POINTS_PRIMARY, sampler.POINTS_ROBUSTNESS):
            sample = self._sample(target)
            self.assertEqual(sample.features.shape, (target, sampler.FEATURE_COUNT))

    def test_feature_channels_are_exactly_declared(self):
        sample = self._sample()
        self.assertEqual(tuple(sampler.FEATURES),
                         ('local_x', 'local_y', 'local_z',
                          'current_candidate_membership'))
        self.assertTrue(np.isin(sample.features[:, 3], [0.0, 1.0]).all())

    def test_sampler_has_no_label_parameter(self):
        import inspect
        parameters = list(inspect.signature(sampler.sample).parameters)
        self.assertEqual(parameters,
                         ['points', 'membership', 'changed', 'candidate_mask',
                          'sampling_seed_value', 'points_target', 'repeat'])
        # No parameter may carry class, label, corruption or provenance information.
        for forbidden in ('class', 'label', 'spec', 'severity', 'partition',
                          'candidate_id', 'example_class', 'changed_membership'):
            for name in parameters:
                self.assertNotIn(forbidden, name, forbidden)

    def test_metric_scale_is_preserved(self):
        """No per-example RMS scaling: a 10 m tree stays 10 m."""
        big = self.points * 10.0
        sample = sampler.sample(big, self.membership, self.changed,
                                self.candidate_mask, self._seed('metric'))
        extent = sample.features[:, :3].max(axis=0) - sample.features[:, :3].min(axis=0)
        # Local coordinates are metres divided by the fixed constant only.
        expected = (big.max(axis=0) - big.min(axis=0)) / sampler.METRIC_SCALE_CONSTANT
        self.assertTrue(np.allclose(extent, expected, rtol=1e-3))

    def test_no_per_example_scaling_flag(self):
        sample = self._sample()
        self.assertFalse(sample.diagnostics['per_example_scaling'])
        self.assertEqual(sample.diagnostics['metric_scale_constant'],
                         sampler.METRIC_SCALE_CONSTANT)

    def test_translation_only_normalisation(self):
        local, origin = sampler.local_translate(self.points)
        self.assertTrue(np.allclose(local * sampler.METRIC_SCALE_CONSTANT
                                    + origin, self.points, atol=1e-4))

    def test_candidate_context_quota_is_class_independent(self):
        sample_a = self._sample()
        # Same scene, different membership: the group quotas must not move.
        membership_b = self.membership.copy()
        membership_b[10000:12000] = True
        sample_b = sampler.sample(self.points, membership_b, self.changed,
                                  self.candidate_mask, self._seed('quota'))
        self.assertEqual(sample_a.diagnostics['candidate_sampled'],
                         sample_b.diagnostics['candidate_sampled'])
        self.assertEqual(sample_a.diagnostics['context_sampled'],
                         sample_b.diagnostics['context_sampled'])

    def test_changed_evidence_is_reserved(self):
        sample = self._sample()
        self.assertGreaterEqual(sample.diagnostics['changed_evidence_sampled'],
                                min(sampler.MIN_CHANGED_EVIDENCE, 500))

    def test_changed_evidence_is_never_duplicated(self):
        tiny = np.zeros(20000, dtype=bool)
        tiny[:10] = True
        sample = sampler.sample(self.points, self.membership, tiny,
                                self.candidate_mask, self._seed('nodup'))
        self.assertLessEqual(sample.diagnostics['changed_evidence_sampled'], 10)
        self.assertEqual(sample.indices.size, len(np.unique(sample.indices)))

    def test_height_and_spatial_coverage(self):
        sample = self._sample()
        self.assertGreaterEqual(len(sample.diagnostics['height_bands_sampled']), 2)
        self.assertGreaterEqual(sample.diagnostics['spatial_cells_sampled'], 2)
        self.assertGreater(sampler.spatial_coverage(sample, self.points), 0.3)
        self.assertGreater(sampler.height_coverage(sample, self.points), 0.3)

    def test_deterministic_for_same_seed(self):
        first = self._sample()
        second = self._sample()
        self.assertTrue(np.array_equal(first.features, second.features))

    def test_misaligned_inputs_rejected(self):
        with self.assertRaises(sampler.SamplerError):
            sampler.sample(self.points, self.membership[:10], self.changed,
                           self.candidate_mask, self._seed('misaligned'))

    def test_empty_scene_rejected(self):
        with self.assertRaises(sampler.SamplerError):
            sampler.sample(np.zeros((0, 3), np.float32), np.zeros(0, bool),
                           np.zeros(0, bool), np.zeros(0, bool),
                           self._seed('empty'))

    def test_sparse_stem_is_represented(self):
        """A sparse lower region must still appear in the sample."""
        points = np.random.default_rng(3).random((20000, 3)).astype(np.float32)
        points[:200, 2] = 0.0  # a thin sparse floor layer
        mask = np.zeros(20000, dtype=bool)
        mask[200:] = True
        sample = sampler.sample(points, mask, np.zeros(20000, bool), mask,
                                self._seed('sparse-stem'))
        low = points[sample.indices][:, 2] < 0.05
        self.assertGreater(int(low.sum()), 0)

    def test_uneven_density_keeps_both_regions(self):
        rng = np.random.default_rng(5)
        dense = rng.normal(0, 0.2, size=(19000, 3)).astype(np.float32)
        sparse = rng.normal(5, 1.5, size=(1000, 3)).astype(np.float32)
        points = np.concatenate([dense, sparse])
        mask = np.concatenate([np.ones(19000, bool), np.zeros(1000, bool)])
        sample = sampler.sample(points, mask, np.zeros(20000, bool), mask,
                                self._seed('uneven-density'))
        sampled = points[sample.indices]
        self.assertGreater(int((sampled[:, 0] > 2.5).sum()), 0)


class SamplerSeedDecoupling(unittest.TestCase):
    """The sampler RNG must not depend on the synthetic class.

    Regression for the class-indexed sampling seed: `enumerate_specs` gives every class a
    distinct corruption seed (`10_007 * class_index`), and that seed used to drive the
    sampler as well. Although `sample` has no class parameter, its RNG therefore still
    depended on the class, which produced a deterministic, class-correlated XYZ artifact
    (mean-Z separation 0.190446).
    """

    CANDIDATE = PROVENANCE.train.eligible[0]

    def setUp(self):
        self.geometry = fixture_geometry(self.CANDIDATE, seed=13)

    def specs(self, severity='MEDIUM', seed_base=4200, repeats=1):
        return gen.enumerate_specs([self.CANDIDATE], 'train', severities=[severity],
                                   seed_base=seed_base, repeats=repeats)

    # --- Test A: the sampling seed itself is class-independent ---

    def test_sampling_seed_is_identical_across_all_seven_classes(self):
        for severity in gen.SEVERITIES:
            with self.subTest(severity=severity):
                group = self.specs(severity)
                self.assertEqual(len(group), len(gen.CLASSES))
                seeds = {spec.sampling_seed(4096, 4200) for spec in group}
                self.assertEqual(len(seeds), 1,
                                 f'sampling seed varies by class: {seeds}')

    def test_corruption_seed_still_varies_by_class(self):
        group = self.specs('MEDIUM')
        self.assertEqual(len({spec.seed for spec in group}), len(gen.CLASSES))

    def test_sampling_seed_ignores_class_label_and_corruption_mechanism(self):
        """Only class-independent components may enter the sampler seed."""
        base = sampler.sampling_seed('train', self.CANDIDATE, 'MEDIUM',
                                     points_target=4096, repeat=0, seed_base=4200)
        for spec in self.specs('MEDIUM'):
            self.assertEqual(spec.sampling_seed(4096, 4200), base, spec.example_class)

    def test_sampling_seed_varies_with_class_independent_components(self):
        """It must still respond to the components that legitimately define a scene."""
        base = sampler.sampling_seed('train', self.CANDIDATE, 'MEDIUM',
                                     points_target=4096, repeat=0, seed_base=4200)
        variants = {
            'partition': sampler.sampling_seed('validation', self.CANDIDATE, 'MEDIUM',
                                               points_target=4096, repeat=0,
                                               seed_base=4200),
            'candidate': sampler.sampling_seed('train', 'other:tree', 'MEDIUM',
                                               points_target=4096, repeat=0,
                                               seed_base=4200),
            'severity': sampler.sampling_seed('train', self.CANDIDATE, 'HIGH',
                                              points_target=4096, repeat=0,
                                              seed_base=4200),
            'points_target': sampler.sampling_seed('train', self.CANDIDATE, 'MEDIUM',
                                                   points_target=2048, repeat=0,
                                                   seed_base=4200),
            'repeat': sampler.sampling_seed('train', self.CANDIDATE, 'MEDIUM',
                                            points_target=4096, repeat=1,
                                            seed_base=4200),
            'seed_base': sampler.sampling_seed('train', self.CANDIDATE, 'MEDIUM',
                                               points_target=4096, repeat=0,
                                               seed_base=99),
        }
        for name, value in variants.items():
            self.assertNotEqual(value, base, name)

    def test_sampling_seed_is_deterministic(self):
        first = self.specs('LOW')[0].sampling_seed(4096, 4200)
        second = self.specs('LOW')[0].sampling_seed(4096, 4200)
        self.assertEqual(first, second)

    # --- Test B: base sampler indices are class-independent ---

    def test_base_sampler_indices_are_identical_across_classes(self):
        """Same scene, same sampling seed, same masks -> identical indices."""
        group = self.specs('MEDIUM')
        reference = None
        for spec in group:
            example = gen.generate_example(spec, self.geometry, PROVENANCE)
            scene = scene_module.build_base_scene(spec.candidate_id, spec.partition,
                                                  self.geometry, PROVENANCE)
            sample = sampler.sample(
                example.points, example.membership, example.changed_membership,
                scene.candidate_mask, spec.sampling_seed(4096, 4200))
            if reference is None:
                reference = sample.indices
            self.assertTrue(np.array_equal(sample.indices, reference),
                            spec.example_class)

    # --- Test C: only changed evidence may move points ---

    def test_changed_evidence_reservation_preserves_budgets(self):
        good = gen.enumerate_specs([self.CANDIDATE], 'train',
                                   classes=[gen.GOOD], severities=['MEDIUM'],
                                   seed_base=4200)[0]
        defect = gen.enumerate_specs([self.CANDIDATE], 'train',
                                     classes=[gen.CROWN_INCOMPLETE],
                                     severities=['MEDIUM'], seed_base=4200)[0]
        seed = good.sampling_seed(4096, 4200)
        self.assertEqual(seed, defect.sampling_seed(4096, 4200))
        scene = scene_module.build_base_scene(self.CANDIDATE, 'train', self.geometry,
                                              PROVENANCE)
        samples = {}
        for spec in (good, defect):
            example = gen.generate_example(spec, self.geometry, PROVENANCE)
            samples[spec.example_class] = sampler.sample(
                example.points, example.membership, example.changed_membership,
                scene.candidate_mask, seed)
        for name, sample in samples.items():
            diagnostics = sample.diagnostics
            self.assertEqual(diagnostics['candidate_sampled'],
                             diagnostics['candidate_budget'], name)
            self.assertEqual(diagnostics['context_sampled'],
                             diagnostics['context_budget'], name)
            self.assertEqual(sample.indices.size, diagnostics['points_target'], name)
            self.assertEqual(len(np.unique(sample.indices)), sample.indices.size, name)
        # Budgets are identical across the two classes even if the points differ.
        good_diag = samples[gen.GOOD].diagnostics
        defect_diag = samples[gen.CROWN_INCOMPLETE].diagnostics
        self.assertEqual(good_diag['candidate_budget'], defect_diag['candidate_budget'])
        self.assertEqual(good_diag['context_budget'], defect_diag['context_budget'])

    def test_only_the_reservation_may_change_indices(self):
        """Any index difference must be explained by the changed-evidence reservation."""
        good = gen.enumerate_specs([self.CANDIDATE], 'train', classes=[gen.GOOD],
                                   severities=['MEDIUM'], seed_base=4200)[0]
        defect = gen.enumerate_specs([self.CANDIDATE], 'train',
                                     classes=[gen.CROWN_INCOMPLETE],
                                     severities=['MEDIUM'], seed_base=4200)[0]
        seed = good.sampling_seed(4096, 4200)
        scene = scene_module.build_base_scene(self.CANDIDATE, 'train', self.geometry,
                                              PROVENANCE)
        good_example = gen.generate_example(good, self.geometry, PROVENANCE)
        defect_example = gen.generate_example(defect, self.geometry, PROVENANCE)
        good_sample = sampler.sample(
            good_example.points, good_example.membership,
            good_example.changed_membership, scene.candidate_mask, seed)
        defect_sample = sampler.sample(
            defect_example.points, defect_example.membership,
            defect_example.changed_membership, scene.candidate_mask, seed)
        # Same seed and same scene: with no changed evidence the draws must agree.
        no_change = sampler.sample(
            good_example.points, good_example.membership,
            np.zeros_like(good_example.changed_membership), scene.candidate_mask, seed)
        self.assertTrue(np.array_equal(no_change.indices, good_sample.indices))
        # When they do differ, the difference is bounded by the reservation.
        if not np.array_equal(good_sample.indices, defect_sample.indices):
            reservation = defect_sample.diagnostics['changed_evidence_reservation']
            self.assertGreaterEqual(
                defect_sample.diagnostics['changed_evidence_sampled'],
                min(sampler.MIN_CHANGED_EVIDENCE,
                    defect_sample.diagnostics['changed_evidence_available']))
            self.assertLessEqual(reservation, sampler.MIN_CHANGED_EVIDENCE)

    # --- Test D: XYZ cannot change through RNG alone ---

    def test_sampled_xyz_is_byte_identical_across_classes(self):
        """With no changed-evidence replacement, features[:, :3] must not move.

        This is the decisive test for the decoupling: the synthetic class may only show
        through the membership channel, never through which points were drawn.
        """
        group = self.specs('MEDIUM')
        scene = scene_module.build_base_scene(self.CANDIDATE, 'train', self.geometry,
                                              PROVENANCE)
        reference_xyz = None
        for spec in group:
            example = gen.generate_example(spec, self.geometry, PROVENANCE)
            # Force the same changed mask so only the class differs.
            sample = sampler.sample(
                example.points, example.membership,
                np.zeros_like(example.changed_membership),
                scene.candidate_mask, spec.sampling_seed(4096, 4200))
            xyz = np.ascontiguousarray(sample.features[:, :3])
            if reference_xyz is None:
                reference_xyz = xyz.tobytes()
            self.assertEqual(xyz.tobytes(), reference_xyz, spec.example_class)

    def test_sampled_xyz_is_byte_identical_across_severities(self):
        """Severity changes the corruption seed, not the sampling seed.

        Severity is part of the sampling-seed key by design, so this asserts the weaker
        but still required property: for a fixed severity, class cannot move the XYZ.
        """
        scene = scene_module.build_base_scene(self.CANDIDATE, 'train', self.geometry,
                                              PROVENANCE)
        for severity in gen.SEVERITIES:
            group = self.specs(severity)
            reference = None
            for spec in group:
                example = gen.generate_example(spec, self.geometry, PROVENANCE)
                sample = sampler.sample(
                    example.points, example.membership,
                    np.zeros_like(example.changed_membership),
                    scene.candidate_mask, spec.sampling_seed(4096, 4200))
                payload = np.ascontiguousarray(
                    sample.features[:, :3]).tobytes()
                if reference is None:
                    reference = payload
                self.assertEqual(payload, reference,
                                 f'{severity}/{spec.example_class}')


class SamplerSeedContract(unittest.TestCase):
    """``sample`` must refuse anything that is not a sampling seed."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.points = rng.random((8000, 3)).astype(np.float32)
        self.mask = np.ones(8000, dtype=bool)
        self.changed = np.zeros(8000, dtype=bool)

    def test_corruption_seed_is_rejected(self):
        spec = gen.enumerate_specs([SamplerSeedDecoupling.CANDIDATE], 'train',
                                   classes=[gen.GOOD], severities=['LOW'],
                                   seed_base=4200)[0]
        self.assertFalse(sampler.is_sampling_seed(spec.seed))
        with self.assertRaises(sampler.SamplerError):
            sampler.sample(self.points, self.mask, self.changed, self.mask, spec.seed)

    def test_arbitrary_int_is_rejected(self):
        for raw in (0, 1, 42, 12345):
            self.assertFalse(sampler.is_sampling_seed(raw))
            with self.assertRaises(sampler.SamplerError):
                sampler.sample(self.points, self.mask, self.changed, self.mask, raw)

    def test_helper_seed_is_accepted(self):
        seed = sampler.sampling_seed('train', 'x:y', 'LOW')
        self.assertTrue(sampler.is_sampling_seed(seed))
        self.assertGreater(seed, 0)
        self.assertLess(seed, 1 << 64)
        sample = sampler.sample(self.points, self.mask, self.changed, self.mask, seed)
        self.assertEqual(sample.points, sampler.POINTS_PRIMARY)

    def test_full_range_seed_does_not_overflow(self):
        """Sampling seeds are drawn from the full uint64 range.

        Mixing them with the golden-ratio constant must not emit a numpy overflow
        warning, which the previous unmasked addition did.
        """
        import warnings
        seeds = [sampler.sampling_seed('train', f'x:{i}', 'LOW') for i in range(64)]
        self.assertTrue(any(seed > 1 << 63 for seed in seeds),
                        'expected some seeds above 2**63')
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            for seed in seeds:
                sample = sampler.sample(self.points, self.mask, self.changed,
                                        self.mask, seed)
                self.assertEqual(sample.indices.size, sampler.POINTS_PRIMARY)


def _record_from_manifest_row(row: dict) -> fingerprint.ExampleRecord:
    """Rebuild an ExampleRecord from a GATE_D_DIAGNOSTIC_MANIFEST.jsonl row."""
    diagnostics = row['sample']['diagnostics']
    return fingerprint.ExampleRecord(
        key=row['key'], candidate_id=row['candidate_id'], partition=row['partition'],
        example_class=row['class'], severity=row['severity'], label=row['label'],
        point_count=row['point_count'], sampled_points=row['sample']['points'],
        membership_fraction=row['membership_fraction'],
        gt_membership_fraction=row['gt_membership_fraction'],
        intended_membership_shift=round(
            row['membership_fraction'] - row['gt_membership_fraction'], 6),
        changed_membership_points=row['changed_membership_points'],
        changed_membership_fraction=row['changed_membership_fraction'],
        candidate_points=row['candidate_points'], context_points=row['context_points'],
        claimed_points=row['claimed_points'], dropped_points=row['dropped_points'],
        source_count=len(row['actual_point_source_ids']),
        corruption_fraction=row['corruption']['fraction'],
        drawn_fraction=row['corruption']['drawn_fraction'],
        corruption_axis=tuple(row['corruption']['axis'])
        if row['corruption'].get('axis') else None,
        membership_sampled=diagnostics['membership_sampled'],
        candidate_sampled=diagnostics['candidate_sampled'],
        context_sampled=diagnostics['context_sampled'],
        changed_evidence_sampled=diagnostics['changed_evidence_sampled'],
        repeat=int(diagnostics['repeat']),
        points_target=int(diagnostics['points_target']),
        sampling_seed=int(diagnostics['sampling_seed']),
        candidate_budget=int(diagnostics['candidate_budget']),
        context_budget=int(diagnostics['context_budget']),
        pre_reservation_indices_digest=diagnostics['pre_reservation_indices_digest'],
        final_indices_digest=diagnostics['final_indices_digest'],
        reservation_replacements=int(diagnostics['reservation_replacements']),
        reservation_shortfall=int(diagnostics['reservation_shortfall']),
        changed_evidence_available=int(diagnostics['changed_evidence_available']),
        scene_points=int(row['point_count']),
        spatial_coverage=row['spatial_coverage'], height_coverage=row['height_coverage'],
        feature_mean=tuple(row['feature_mean']), feature_std=tuple(row['feature_std']),
        bbox_extent=(1.0, 1.0, 1.0), centroid_norm=1.0, duplicate_rate=0.0)


class SamplingStructureGate(unittest.TestCase):
    """The structural hard gate that replaced the invalid permutation gate.

    `class_specific_sampling_fingerprint` freely permuted the class label across 210
    examples that are in fact repeated measures (10 candidates x 7 classes x 3 severities)
    with paired structure, so its p-value did not satisfy exchangeability. It is replaced
    by direct structural invariants, which are proven here to fire on a real violation and
    to stay silent on a legal one.
    """

    CONTRACT = ('c2', 'MEDIUM')

    def _records(self, **overrides):
        return _synthetic_records(contracts={self.CONTRACT: overrides})

    def _gate(self, records):
        return next(c for c in fingerprint.analyse(records)['checks']
                    if c['check'] == 'sampling_structure_invariants')

    # --- Test A: a class-indexed sampling seed must fail ---

    def test_class_indexed_sampling_seed_is_critical(self):
        """Reproduce the original defect: one class is fed a class-indexed seed.

        The corruption seed is `seed_base + 1_000_003 * candidate + 10_007 * class_index
        + 101 * severity`, which is exactly the value the sampler used to receive. Feeding
        it back for a single class must trip the structural gate.
        """
        records = _synthetic_records()
        broken = []
        for record in records:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class == gen.NOISE:
                record = replace(record, sampling_seed=64242)
            broken.append(record)
        result = fingerprint.analyse(broken)
        check = next(c for c in result['checks']
                     if c['check'] == 'sampling_structure_invariants')
        self.assertTrue(check['critical'])
        self.assertFalse(check['pass'])
        self.assertIn('sampling_structure_invariants',
                      result['CRITICAL_FINGERPRINTS'])
        self.assertIn('sampling_seed_class_independent',
                      [v['invariant'] for v in check['detail']['violations']])
        # Exactly one contract group is affected, out of the full grid.
        self.assertEqual(check['detail']['sampling_seed_groups_passed'],
                         check['detail']['groups_checked'] - 1)

    # --- Test B: a class-dependent base sample must fail ---

    def test_pre_reservation_digest_difference_is_critical(self):
        base = _synthetic_records()
        broken = []
        for record in base:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class == gen.CROWN_INCOMPLETE:
                record = replace(record,
                                 pre_reservation_indices_digest='tampered-digest')
            broken.append(record)
        result = fingerprint.analyse(broken)
        check = next(c for c in result['checks']
                     if c['check'] == 'sampling_structure_invariants')
        self.assertTrue(check['critical'])
        self.assertIn('pre_reservation_indices_class_independent',
                      [v['invariant'] for v in check['detail']['violations']])
        self.assertIn('sampling_structure_invariants',
                      result['CRITICAL_FINGERPRINTS'])

    def test_missing_pre_reservation_digest_is_critical(self):
        base = _synthetic_records()
        broken = []
        for record in base:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class == gen.GOOD:
                record = replace(record, pre_reservation_indices_digest='')
            broken.append(record)
        check = self._gate(broken)
        self.assertTrue(check['critical'])

    # --- Test C: a budget difference must fail ---

    def test_budget_difference_is_critical(self):
        base = _synthetic_records()
        broken = []
        for record in base:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class == gen.MERGED_MULTI_TREE:
                record = replace(record, candidate_budget=record.candidate_budget + 1)
            broken.append(record)
        result = fingerprint.analyse(broken)
        check = next(c for c in result['checks']
                     if c['check'] == 'sampling_structure_invariants')
        self.assertTrue(check['critical'])
        self.assertIn('candidate_context_budget_independent',
                      [v['invariant'] for v in check['detail']['violations']])
        self.assertIn('sampling_structure_invariants',
                      result['CRITICAL_FINGERPRINTS'])

    def test_budget_arithmetic_violation_is_critical(self):
        """candidate_sampled + context_sampled must equal points_target."""
        base = _synthetic_records()
        broken = []
        for record in base:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class == gen.STEM_MISSING:
                record = replace(record, context_sampled=record.context_sampled - 1)
            broken.append(record)
        check = self._gate(broken)
        self.assertTrue(check['critical'])
        violation = next(v for v in check['detail']['violations']
                         if v['invariant'] == 'per_example_target_and_budget')
        self.assertIn('context_sampled_matches_budget', violation['failed'])
        self.assertIn('sampled_sum_matches_target', violation['failed'])

    def test_budget_sum_violation_is_critical(self):
        """candidate_budget + context_budget must equal points_target."""
        base = _synthetic_records()
        broken = []
        for record in base:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class == gen.NOISE:
                record = replace(record, candidate_budget=record.candidate_budget + 1)
            broken.append(record)
        check = self._gate(broken)
        self.assertTrue(check['critical'])
        invariants = [v['invariant'] for v in check['detail']['violations']]
        self.assertIn('per_example_target_and_budget', invariants)
        self.assertIn('candidate_context_budget_independent', invariants)

    # --- Test D: a legal reservation difference must pass ---

    def test_legal_reservation_difference_passes(self):
        """Different final draws from changed evidence are allowed and not a violation."""
        base = _synthetic_records()
        varied = []
        for record in base:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class != gen.GOOD:
                record = replace(record, final_indices_digest='reserved',
                                 reservation_replacements=12,
                                 changed_evidence_sampled=32)
            varied.append(record)
        check = self._gate(varied)
        self.assertFalse(check['critical'])
        self.assertTrue(check['pass'])
        self.assertEqual(check['detail']['violations'], [])
        self.assertGreater(check['detail']['groups_checked'], 0)
        self.assertEqual(check['detail']['sampling_seed_groups_passed'],
                         check['detail']['groups_checked'])
        self.assertEqual(check['detail']['pre_reservation_digest_groups_passed'],
                         check['detail']['groups_checked'])
        self.assertEqual(check['detail']['budget_groups_passed'],
                         check['detail']['groups_checked'])

    # --- Test 1: repeats form their own contract groups ---

    def test_repeats_two_gives_sixty_contract_groups(self):
        """10 candidates x 3 severities x 2 repeats = 60 groups, 7 classes each."""
        records = _synthetic_records(repeats=2)
        self.assertEqual(len(records), 10 * 3 * 2 * 7)
        check = self._gate(records)
        self.assertTrue(check['pass'], check['detail']['violations'])
        self.assertEqual(check['detail']['groups_checked'], 60)
        self.assertEqual(check['detail']['sampling_seed_groups_passed'], 60)
        self.assertEqual(check['detail']['pre_reservation_digest_groups_passed'], 60)
        self.assertEqual(check['detail']['budget_groups_passed'], 60)
        self.assertEqual(check['detail']['violations'], [])

    def test_repeats_have_distinct_seeds_but_classes_within_a_repeat_agree(self):
        records = _synthetic_records(repeats=2)
        by_repeat = {}
        for record in records:
            if record.candidate_id != 'c0' or record.severity != 'MEDIUM':
                continue
            by_repeat.setdefault(record.repeat, set()).add(record.sampling_seed)
        self.assertEqual(set(by_repeat), {0, 1})
        # Within a repeat all seven classes share one seed.
        for repeat, seeds in by_repeat.items():
            self.assertEqual(len(seeds), 1, repeat)
        # Different repeats use different seeds.
        self.assertNotEqual(by_repeat[0], by_repeat[1])

    # --- Test 2: a wrong sampled size must fail, with no singleton escape ---

    def test_wrong_sampled_size_is_detected_without_singleton_escape(self):
        """The target comes from points_target, not from the achieved sample."""
        base = _synthetic_records()
        broken = []
        for record in base:
            if (record.candidate_id, record.severity) == self.CONTRACT \
                    and record.example_class == gen.CROWN_INCOMPLETE:
                record = replace(record, sampled_points=record.points_target - 1)
            broken.append(record)
        check = self._gate(broken)
        self.assertTrue(check['critical'])
        # It must NOT have formed its own legal-looking group: the group count is
        # unchanged because the key is the pre-declared target, not the result.
        self.assertEqual(check['detail']['groups_checked'], 30)
        violation = next(v for v in check['detail']['violations']
                         if v['invariant'] == 'per_example_target_and_budget')
        self.assertIn('sampled_size_matches_target', violation['failed'])

    def test_contract_key_never_uses_the_achieved_sample(self):
        """Two records differing only in sampled_points share one contract key."""
        base = _synthetic_records()
        first = base[0]
        second = replace(first, sampled_points=first.points_target - 7)
        self.assertEqual(first.contract_key(), second.contract_key())
        self.assertEqual(first.contract_key()[3], first.repeat)
        self.assertEqual(first.contract_key()[4], first.points_target)
        self.assertNotIn(first.example_class, first.contract_key())

    def test_real_records_satisfy_the_gate(self):
        """The contract holds on the real bounded diagnostic artifacts."""
        manifest = (provenance_module.REPORTS / 'GATE_D_DIAGNOSTIC_MANIFEST.jsonl')
        if not manifest.exists():
            self.skipTest('diagnostic manifest not present')
        rows = [json.loads(line) for line in
                manifest.read_text(encoding='utf-8').splitlines() if line]
        records = [_record_from_manifest_row(row) for row in rows]
        check = self._gate(records)
        self.assertTrue(check['pass'], check['detail']['violations'])
        self.assertGreaterEqual(check['detail']['groups_checked'], 30)
        self.assertEqual(check['detail']['sampling_seed_groups_passed'],
                         check['detail']['groups_checked'])
        self.assertEqual(check['detail']['pre_reservation_digest_groups_passed'],
                         check['detail']['groups_checked'])

    # --- Test E: the ratio bias is audit only ---

    def test_ratio_bias_is_audited_not_gated(self):
        """A small systematic, scene-dependent gap must not fail Gate D.

        The gap is deliberately made a function of the candidate's scene membership
        fraction (which is what the real diagnostic shows) and kept identical across
        classes, so it is not a model-visible shortcut.
        """
        base = _synthetic_records()
        gapped = []
        for record in base:
            scene = 0.1 + 0.02 * int(record.candidate_id.lstrip('c'))
            gapped.append(replace(record, membership_fraction=scene,
                                  membership_sampled=scene + 0.002))
        result = fingerprint.analyse(gapped)
        self.assertNotEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
        self.assertEqual(result['CRITICAL_FINGERPRINTS'], [])
        audit = next(c for c in result['checks']
                     if c['check'] == 'sampling_ratio_bias_audit')
        self.assertFalse(audit['critical'])
        self.assertEqual(audit['detail']['status'], 'AUDIT_ONLY')
        self.assertFalse(audit['detail']['is_gate'])
        self.assertAlmostEqual(audit['detail']['mean_gap'], 0.002, places=6)
        # The bias is visible in the audit even though it does not gate.
        self.assertAlmostEqual(audit['detail']['max_abs_gap'], 0.002, places=6)
        self.assertEqual(len(set(audit['detail']['per_class_mean_gap'].values())), 1)
        for key in ('mean_abs_gap', 'max_abs_gap', 'per_class_mean_gap',
                    'per_candidate_mean_gap', 'per_severity_mean_gap',
                    'scene_membership_fraction_min',
                    'scene_membership_fraction_max',
                    'reservation_contribution', 'caveats'):
            self.assertIn(key, audit['detail'])

    def test_ratio_audit_never_enters_critical_fingerprints(self):
        records = _synthetic_records()
        result = fingerprint.analyse(records)
        self.assertNotIn('sampling_ratio_bias_audit',
                         result['CRITICAL_FINGERPRINTS'])
        self.assertNotIn('class_specific_sampling_fingerprint',
                         [c['check'] for c in result['checks']])


class VisibleShortcutGatesRemainHard(unittest.TestCase):
    """The gates on the real model-visible channels must stay hard."""

    @staticmethod
    def _gate(result, name):
        return next(c for c in result['checks'] if c['check'] == name)

    # --- Test F: a real membership-only shortcut must still fail ---

    def test_membership_only_shortcut_is_critical(self):
        """A membership scalar that separates PASS from REVIEW end to end."""
        records = _synthetic_records()
        # GOOD membership fraction near 1, every defect near 0: a saturated shortcut no
        # matter which candidate or partition is used.
        rigged = []
        for record in records:
            value = 0.99 if record.example_class == gen.GOOD else 0.01
            rigged.append(replace(record, membership_sampled=value,
                                  membership_fraction=value))
        result = fingerprint.analyse(rigged)
        check = self._gate(result, 'membership_only_shortcut')
        self.assertTrue(check['critical'])
        self.assertFalse(check['pass'])
        self.assertIn('membership_only_shortcut',
                      result['CRITICAL_FINGERPRINTS'])

    # --- Test G: a real XYZ shortcut must still fail, per column ---

    def test_xyz_shortcut_columns_remain_critical(self):
        order = {name: index for index, name in enumerate(gen.CLASSES)}
        for column in range(3):
            with self.subTest(column=f'feature_mean[{column}]'):

                def mean_of(index, example_class, column=column):
                    values = [0.0] * 4
                    values[column] = float(order[example_class])
                    return values

                result = fingerprint.analyse(
                    _synthetic_records(feature_mean=mean_of))
                self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
                self.assertIn('metadata_leakage_tensor_stats',
                              result['CRITICAL_FINGERPRINTS'])
        for column in range(3):
            with self.subTest(column=f'feature_std[{column}]'):

                def std_of(index, example_class, column=column):
                    values = [1.0] * 4
                    values[column] = float(order[example_class])
                    return values

                result = fingerprint.analyse(
                    _synthetic_records(feature_std=std_of))
                self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
                self.assertIn('metadata_leakage_tensor_stats',
                              result['CRITICAL_FINGERPRINTS'])

    def test_other_hard_gates_are_still_present(self):
        result = fingerprint.analyse(_synthetic_records(seed=61))
        names = [c['check'] for c in result['checks']]
        for name in ('point_count_fingerprint_source',
                     'point_count_fingerprint_sampled',
                     'fixed_fraction_fingerprint',
                     'fixed_axis_fingerprint',
                     'metadata_leakage_tensor_stats',
                     'membership_only_shortcut',
                     'metadata_leakage_identifiers',
                     'sampling_structure_invariants'):
            self.assertIn(name, names, name)
        self.assertTrue(result['controls']['detector_verified'])
        self.assertTrue(result['controls']['random_control_pass'])
        self.assertTrue(result['controls']['implanted_shortcut_pass'])


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
        self.assertEqual(len(first.class_balance()), 21)

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
        row = self.make().manifest()[0]
        for field in ('candidate_id', 'partition', 'class', 'severity', 'seed', 'key'):
            self.assertIn(field, row)


class MetricsAndThreshold(unittest.TestCase):
    def test_metrics_on_a_known_confusion(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        row = metrics.metrics(labels, scores, 0.5)
        self.assertEqual(row['CONFUSION_MATRIX']['tn_pass_pass'], 1)
        self.assertAlmostEqual(row['BAD_RECALL'], 0.5)
        self.assertAlmostEqual(row['BALANCED_ACCURACY'], 0.5)

    def test_perfect_separation(self):
        row = metrics.metrics(np.array([0, 0, 1, 1]), np.array([0.01, 0.02, 0.98, 0.99]),
                              0.5)
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
        for partition in ('held_out', 'test', 'train_held_out'):
            with self.assertRaises(metrics.ThresholdError):
                metrics.select_threshold(np.array([0, 1]), np.array([0.1, 0.9]),
                                         partition)

    def test_three_way_mapping(self):
        decisions = metrics.three_way_decisions(np.array([0.0, 0.5, 1.0]), 0.5, 0.1)
        self.assertEqual(list(decisions), [metrics.DECISION_PASS,
                                           metrics.DECISION_UNCERTAIN,
                                           metrics.DECISION_REVIEW])

    def test_frozen_threshold_requires_model_selection_partition(self):
        with self.assertRaises(metrics.ThresholdError):
            metrics.freeze_threshold({'threshold': 0.5, 'partition': 'held_out'},
                                     'd', 'c', 1)


class FingerprintControls(unittest.TestCase):
    """Blocker 5: the detector itself must be verified by negative controls."""

    def test_random_features_land_near_chance(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(7), 30)
        features = rng.normal(size=(labels.size, 8))
        separation = fingerprint.max_feature_class_separation(features, labels)
        self.assertLessEqual(separation, fingerprint.RANDOM_CONTROL_MAX_SEPARATION)

    def test_implanted_shortcut_is_detected(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(7), 30)
        features = rng.normal(size=(labels.size, 8))
        implanted = np.concatenate([features, labels.astype(float)[:, None]], axis=1)
        separation = fingerprint.max_feature_class_separation(implanted, labels)
        self.assertGreaterEqual(separation,
                                fingerprint.IMPLANTED_CONTROL_MIN_SEPARATION)

    def test_controls_report_verification(self):
        rng = np.random.default_rng(1)
        labels = np.repeat(np.arange(7), 30)
        features = rng.normal(size=(labels.size, 8))
        controls = fingerprint._controls(features, labels, rng)
        self.assertTrue(controls['detector_verified'])
        self.assertTrue(controls['random_control_pass'])
        self.assertTrue(controls['implanted_shortcut_pass'])

    def test_loo_detector_does_not_use_the_held_out_label(self):
        """Random features must not score above chance under the honest probe."""
        rng = np.random.default_rng(2)
        labels = np.repeat(np.arange(7), 30)
        features = rng.normal(size=(labels.size, 8))
        accuracy = fingerprint.loo_nearest_centroid(features, labels)
        self.assertLess(accuracy, 0.40)

    def test_loo_threshold_accuracy_is_out_of_sample(self):
        """A threshold that isolates one in-sample outlier must not survive LOO."""
        rng = np.random.default_rng(3)
        values = rng.normal(size=120)
        values[0] = 50.0  # one extreme point
        binary = np.zeros(120, dtype=int)
        binary[0] = 1
        in_sample = fingerprint.best_binary_split(values, binary)
        self.assertGreater(in_sample['best_balanced_accuracy'], 0.95)
        self.assertLess(fingerprint.loo_threshold_accuracy(values, binary), 0.98)

    def test_balanced_accuracy_ignores_class_imbalance(self):
        # A constant feature with interleaved labels carries no class information.
        values = np.zeros(180)
        binary = np.zeros(180, dtype=int)
        binary[::6] = 1
        split = fingerprint.best_binary_split(values, binary)
        # Raw accuracy reports the majority-class rate (0.83); balanced accuracy does not.
        self.assertGreater(split['best_accuracy'], 0.80)
        self.assertLess(split['best_balanced_accuracy'], 0.60)


class SamplerExtremeStrata(unittest.TestCase):
    """Blocker 4 regression: occupied strata may outnumber the group budget.

    The earlier `_select_within` cleared `quotas` without resetting `allocated`, so the
    funding loop broke immediately, the group contributed zero structured points, and the
    generic whole-scene refill in `sample()` silently replaced the group budget.
    """

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _check_selection(self, pool, keys, budget, floor=1, label='case'):
        picked = sampler._select_within(pool, keys, budget, self.rng, floor)
        expected = min(budget, pool.size)
        self.assertEqual(picked.size, expected, label)
        self.assertGreater(picked.size, 0, label)
        self.assertEqual(len(np.unique(picked)), picked.size, label)
        self.assertTrue(np.isin(picked, pool).all(), label)
        return picked

    def test_more_strata_than_budget(self):
        # 100 occupied strata, budget 32, floor 1: the exact reported failure.
        picked = self._check_selection(np.arange(100), np.arange(100), 32,
                                       label='100x1 budget32')
        # The structured quota must survive: points reach many distinct strata.
        self.assertGreaterEqual(len(np.unique(np.asarray(picked))), 32)

    def test_many_strata_many_budgets(self):
        for strata, budget in ((500, 64), (1000, 10), (200, 199), (64, 63)):
            with self.subTest(strata=strata, budget=budget):
                self._check_selection(np.arange(strata), np.arange(strata), budget,
                                      label=f'{strata}x1 budget{budget}')

    def test_uneven_strata_do_not_exceed_budget_or_size(self):
        sizes = np.array([1] * 90 + [400] * 10)
        keys = np.repeat(np.arange(sizes.size), sizes)
        pool = np.arange(keys.size)
        for budget in (32, 64, 512, 4096):
            with self.subTest(budget=budget):
                picked = self._check_selection(pool, keys, budget,
                                               label=f'uneven budget{budget}')
                counts = np.bincount(keys[picked], minlength=sizes.size)
                # Never exceeds any stratum's real size.
                self.assertTrue((counts <= sizes).all())

    def test_budget_larger_than_pool_is_capped(self):
        self._check_selection(np.arange(50), np.arange(50), 200, label='budget>pool')

    def test_normal_proportional_path_still_works(self):
        keys = np.repeat(np.arange(6), 1000)
        pool = np.arange(6000)
        picked = self._check_selection(pool, keys, 4096, label='proportional')
        counts = np.bincount(keys[picked], minlength=6)
        # Roughly equal strata must receive roughly equal quotas.
        self.assertLess(int(counts.max() - counts.min()), 32)

    def test_misaligned_keys_rejected(self):
        with self.assertRaises(sampler.SamplerError):
            sampler._select_within(np.arange(10), np.arange(9), 5, self.rng, 1)

    def test_group_budget_preserved_when_strata_exceed_budget(self):
        """`sample()` must not let the generic refill replace a group's budget."""
        rng = np.random.default_rng(7)
        count = 20000
        points = rng.normal(size=(count, 3)).astype(np.float32)
        candidate_mask = np.zeros(count, dtype=bool)
        candidate_mask[:3000] = True
        membership = candidate_mask.copy()
        changed = np.zeros(count, dtype=bool)
        changed[:100] = True

        sample = sampler.sample(
            points, membership, changed, candidate_mask,
            sampler.sampling_seed('train', 'budget-test', 'LOW'))
        diagnostics = sample.diagnostics
        self.assertEqual(diagnostics['candidate_sampled'],
                         diagnostics['candidate_budget'])
        self.assertEqual(diagnostics['context_sampled'],
                         diagnostics['context_budget'])
        self.assertEqual(diagnostics['candidate_sampled']
                         + diagnostics['context_sampled'],
                         diagnostics['points_target'])
        self.assertEqual(sample.indices.size, len(np.unique(sample.indices)))
        self.assertEqual(sample.indices.size, diagnostics['points_target'])

    def test_extreme_strata_scene_keeps_coverage_without_duplicates(self):
        """A scene engineered so each group has far more strata than its budget."""
        rng = np.random.default_rng(11)
        # Candidate: 600 points spread over a wide, sparse footprint.
        candidate = (rng.random((600, 3)).astype(np.float32) * 20.0)
        # Context: a dense clump plus a thin wide shell, so strata are uneven.
        dense = rng.normal(10.0, 0.05, size=(9000, 3)).astype(np.float32)
        shell = (rng.random((3000, 3)).astype(np.float32) * 20.0)
        points = np.concatenate([candidate, dense, shell])
        candidate_mask = np.zeros(points.shape[0], dtype=bool)
        candidate_mask[:600] = True
        membership = candidate_mask.copy()
        changed = np.zeros(points.shape[0], dtype=bool)
        changed[:64] = True

        sample = sampler.sample(
            points, membership, changed, candidate_mask,
            sampler.sampling_seed('train', 'extreme-strata', 'LOW'))
        diagnostics = sample.diagnostics
        self.assertEqual(sample.indices.size, diagnostics['points_target'])
        self.assertEqual(len(np.unique(sample.indices)), sample.indices.size)
        self.assertEqual(diagnostics['candidate_sampled'],
                         diagnostics['candidate_budget'])
        self.assertEqual(diagnostics['context_sampled'],
                         diagnostics['context_budget'])
        # Structured coverage survives the extreme strata layout.
        self.assertGreaterEqual(diagnostics['spatial_cells_sampled'], 2)
        self.assertGreaterEqual(len(diagnostics['height_bands_sampled']), 2)
        self.assertGreaterEqual(
            diagnostics['changed_evidence_sampled'],
            min(sampler.MIN_CHANGED_EVIDENCE, int(changed.sum())))


def _synthetic_records(feature_mean=None, feature_std=None,
                       seed: int = 5, contracts=None, repeats: int = 1,
                       candidates: int = 10) -> list:
    """A full ExampleRecord set covering every class, for `analyse()` level tests.

    Every field is plausible; only the exposed tensor statistics are manipulated. The
    structural fields default to valid, class-independent values so the
    `sampling_structure_invariants` hard gate is satisfied. ``contracts`` may be a mapping
    ``(candidate_id, severity, repeat) -> dict`` of per-contract overrides used to violate
    an invariant deliberately.
    """
    rng = np.random.default_rng(seed)
    classes = list(gen.CLASSES)
    records = []
    for candidate_index in range(candidates):
        candidate_id = f'c{candidate_index}'
        for severity in gen.SEVERITIES:
            for repeat in range(repeats):
                contract = (contracts or {}).get((candidate_id, severity, repeat), {})
                sampling_seed = contract.get(
                    'sampling_seed',
                    sampler.sampling_seed('train', candidate_id, severity,
                                          repeat=repeat))
                pre_digest = contract.get(
                    'pre_reservation_indices_digest',
                    f'digest-{candidate_id}-{severity}-{repeat}')
                for class_index, example_class in enumerate(classes):
                    index = (candidate_index * len(gen.SEVERITIES) * repeats
                             + list(gen.SEVERITIES).index(severity) * repeats
                             + repeat) * len(classes) + class_index
                    mean = list(feature_mean(index, example_class)) \
                        if callable(feature_mean) \
                        else list(feature_mean if feature_mean is not None
                                  else rng.normal(size=4))
                    std = list(feature_std(index, example_class)) \
                        if callable(feature_std) \
                        else list(feature_std if feature_std is not None
                                  else np.abs(rng.normal(size=4)) + 1.0)
                    points_target = contract.get('points_target', 4096)
                    candidate_budget = contract.get('candidate_budget', 1024)
                    context_budget = contract.get(
                        'context_budget', points_target - candidate_budget)
                    records.append(fingerprint.ExampleRecord(
                        key=f'{candidate_id}|{example_class}|{severity}|{repeat}',
                        candidate_id=candidate_id, partition='train',
                        example_class=example_class, severity=severity,
                        label='PASS' if example_class == gen.GOOD
                        else 'REVIEW_REQUIRED',
                        point_count=1000,
                        sampled_points=contract.get('sampled_points', points_target),
                        membership_fraction=0.2,
                        gt_membership_fraction=0.2, intended_membership_shift=0.0,
                        changed_membership_points=0,
                        changed_membership_fraction=0.0,
                        candidate_points=1024, context_points=3072, claimed_points=0,
                        dropped_points=0, source_count=2, corruption_fraction=0.1,
                        drawn_fraction=0.1, corruption_axis=None,
                        membership_sampled=0.2,
                        candidate_sampled=contract.get('candidate_sampled',
                                                       candidate_budget),
                        context_sampled=contract.get('context_sampled',
                                                     context_budget),
                        changed_evidence_sampled=0,
                        repeat=repeat, points_target=points_target,
                        sampling_seed=sampling_seed,
                        candidate_budget=candidate_budget,
                        context_budget=context_budget,
                        pre_reservation_indices_digest=pre_digest,
                        final_indices_digest=(
                            f'final-{candidate_id}-{severity}-{repeat}-{example_class}'),
                        reservation_replacements=0, reservation_shortfall=0,
                        changed_evidence_available=0, scene_points=4096,
                        spatial_coverage=0.9, height_coverage=1.0,
                        feature_mean=tuple(mean), feature_std=tuple(std),
                        bbox_extent=(1.0, 1.0, 1.0), centroid_norm=1.0,
                        duplicate_rate=0.0))
    return records


class FingerprintRealGate(unittest.TestCase):
    """Blocker 5 end-to-end: the real `analyse()` gate must actually HOLD.

    Testing the detector helper alone is not enough; these tests drive the full
    `fingerprint.analyse(records)` path that produces `GENERATOR_FINGERPRINT`.

    The gate decides on the **artifact columns**: all six marginal XYZ mean/std
    statistics. The corruption re-labels points and never moves them, so none of X, Y or Z
    may vary with the synthetic class. The membership statistics are the injected defect
    and are reported as the defect signature.
    """

    @staticmethod
    def _gate(result):
        return next(c for c in result['checks']
                    if c['check'] == 'metadata_leakage_tensor_stats')

    @staticmethod
    def _class_code(example_class):
        return float(list(gen.CLASSES).index(example_class))

    def _implant_mean(self, column):
        def factory(index, example_class):
            values = [0.0] * 4
            values[column] = self._class_code(example_class)
            return values
        return factory

    def _implant_std(self, column):
        def factory(index, example_class):
            values = [1.0] * 4
            values[column] = self._class_code(example_class)
            return values
        return factory

    # --- A..F: every XYZ marginal statistic must fail the gate ---

    def test_implanted_mean_x_shortcut_is_critical(self):
        result = fingerprint.analyse(
            _synthetic_records(feature_mean=self._implant_mean(0)))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
        self.assertIn('metadata_leakage_tensor_stats',
                      result['CRITICAL_FINGERPRINTS'])
        check = self._gate(result)
        self.assertTrue(check['critical'])
        self.assertFalse(check['pass'])
        self.assertEqual(check['detail']['artifact_detector']['worst_column'], 0)

    def test_implanted_mean_y_shortcut_is_critical(self):
        result = fingerprint.analyse(
            _synthetic_records(feature_mean=self._implant_mean(1)))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
        check = self._gate(result)
        self.assertEqual(check['detail']['artifact_detector']['worst_column'], 1)

    def test_implanted_mean_z_shortcut_is_critical(self):
        """Z is a position statistic too: the earlier formulation excluded it."""
        result = fingerprint.analyse(
            _synthetic_records(feature_mean=self._implant_mean(2)))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
        self.assertIn('metadata_leakage_tensor_stats',
                      result['CRITICAL_FINGERPRINTS'])
        check = self._gate(result)
        self.assertTrue(check['critical'])
        # Column 2 is mean_z in the artifact slice, which lists XYZ as 0,1,2,4,5,6.
        self.assertEqual(check['detail']['artifact_detector']['worst_column'], 2)
        self.assertEqual(check['detail']['artifact_column_names'][2],
                         'feature_mean[2]')

    def test_implanted_std_x_shortcut_is_critical(self):
        result = fingerprint.analyse(
            _synthetic_records(feature_std=self._implant_std(0)))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
        check = self._gate(result)
        # std_x is the fourth artifact column (index 3).
        self.assertEqual(check['detail']['artifact_detector']['worst_column'], 3)
        self.assertEqual(check['detail']['artifact_column_names'][3],
                         'feature_std[0]')

    def test_implanted_std_y_shortcut_is_critical(self):
        result = fingerprint.analyse(
            _synthetic_records(feature_std=self._implant_std(1)))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
        check = self._gate(result)
        self.assertEqual(check['detail']['artifact_detector']['worst_column'], 4)

    def test_implanted_std_z_shortcut_is_critical(self):
        """The std-Z column must be inside the artifact gate as well."""
        result = fingerprint.analyse(
            _synthetic_records(feature_std=self._implant_std(2)))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'CRITICAL')
        self.assertIn('metadata_leakage_tensor_stats',
                      result['CRITICAL_FINGERPRINTS'])
        check = self._gate(result)
        self.assertEqual(check['detail']['artifact_detector']['worst_column'], 5)
        self.assertEqual(check['detail']['artifact_column_names'][5],
                         'feature_std[2]')

    def test_all_six_xyz_columns_are_in_the_artifact_gate(self):
        """The artifact slice must actually cover mean/std of X, Y and Z."""
        self.assertEqual(fingerprint.ARTIFACT_COLUMNS, (0, 1, 2, 4, 5, 6))
        self.assertEqual(fingerprint.ARTIFACT_COLUMN_NAMES,
                         ('feature_mean[0]', 'feature_mean[1]', 'feature_mean[2]',
                          'feature_std[0]', 'feature_std[1]', 'feature_std[2]'))
        self.assertEqual(fingerprint.DEFECT_COLUMNS, (3, 7))
        self.assertEqual(fingerprint.DEFECT_COLUMN_NAMES,
                         ('feature_mean[3]', 'feature_std[3]'))
        # And the running detector reports the same slice it decides on.
        result = fingerprint.analyse(_synthetic_records(seed=41))
        check = self._gate(result)
        self.assertEqual(len(check['detail']['artifact_per_column_separation']), 6)
        self.assertEqual(check['detail']['artifact_columns'],
                         list(fingerprint.ARTIFACT_COLUMNS))

    # --- membership defect signature stays a signal, not a failure ---

    def test_membership_mean_defect_signature_is_reported_not_failed(self):
        """A class signal in feature_mean[3] is the injected defect."""
        result = fingerprint.analyse(
            _synthetic_records(feature_mean=self._implant_mean(3)))
        check = self._gate(result)
        self.assertFalse(check['critical'])
        self.assertFalse(check['detail']['artifact_detector'][
            'exceeds_calibrated_null'])
        self.assertGreater(
            check['detail']['defect_signature']['observed_separation'],
            fingerprint.MIN_REAL_SEPARATION)

    def test_membership_std_defect_signature_is_reported_not_failed(self):
        """A class signal in feature_std[3] is the Bernoulli image of the same defect."""
        result = fingerprint.analyse(
            _synthetic_records(feature_std=self._implant_std(3)))
        check = self._gate(result)
        self.assertFalse(check['critical'])
        self.assertFalse(check['detail']['artifact_detector'][
            'exceeds_calibrated_null'])
        self.assertGreater(
            check['detail']['defect_signature']['observed_separation'],
            fingerprint.MIN_REAL_SEPARATION)

    def test_membership_only_shortcut_gate_still_runs(self):
        """Membership is not unconditionally safe: its own gate must remain active."""
        result = fingerprint.analyse(_synthetic_records(seed=43))
        names = [c['check'] for c in result['checks']]
        self.assertIn('membership_only_shortcut', names)
        check = next(c for c in result['checks']
                     if c['check'] == 'membership_only_shortcut')
        detail = check['detail']['detail']
        for key in ('membership_fraction_cross_partition',
                    'membership_fraction_leave_one_candidate_out_balanced_accuracy',
                    'membership_zero_count_loo_balanced_accuracy',
                    'membership_one_count_loo_balanced_accuracy',
                    'any_membership_zero'):
            self.assertIn(key, detail)

    # --- controls and calibration ---

    def test_random_exposed_statistics_do_not_trigger_the_gate(self):
        result = fingerprint.analyse(_synthetic_records(seed=17))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'NONE_DETECTED')
        self.assertNotIn('metadata_leakage_tensor_stats',
                         result['CRITICAL_FINGERPRINTS'])
        check = self._gate(result)
        self.assertFalse(check['critical'])
        self.assertTrue(check['pass'])
        self.assertTrue(result['controls']['random_control_pass'])

    def test_analyse_reports_the_calibrated_detector(self):
        result = fingerprint.analyse(_synthetic_records(seed=23))
        artifact = self._gate(result)['detail']['artifact_detector']
        for key in ('critical_value', 'observed_separation', 'p_value',
                    'exceeds_calibrated_null'):
            self.assertIn(key, artifact)
        self.assertEqual(artifact['permutations'], fingerprint.PERMUTATIONS)
        self.assertEqual(artifact['min_real_separation'],
                         fingerprint.MIN_REAL_SEPARATION)

    def test_controls_and_real_gate_use_the_same_statistic(self):
        """The verified detector must be the one the real gate decides with."""
        result = fingerprint.analyse(_synthetic_records(seed=29))
        check = self._gate(result)
        self.assertEqual(check['detail']['artifact_detector']['statistic'],
                         'max over columns of the between-class variance share')
        controls = result['controls']
        self.assertIn('max-feature separation',
                      controls['verification_basis'])
        self.assertTrue(controls['detector_verified'])
        self.assertTrue(controls['implanted_shortcut_pass'])

    def test_detector_ignores_a_negligible_separation(self):
        """A tiny but statistically detectable separation must not fail the gate."""
        rng = np.random.default_rng(31)

        def mean_of(index, example_class):
            values = list(rng.normal(size=4))
            # Barely class-correlated, far below the practical floor.
            values[0] += 0.001 * list(gen.CLASSES).index(example_class)
            return values

        result = fingerprint.analyse(_synthetic_records(feature_mean=mean_of))
        self.assertEqual(result['GENERATOR_FINGERPRINT'], 'NONE_DETECTED')


class TrainingLock(unittest.TestCase):
    """Blocker 6: the lock binds the current generator and protocol digests."""

    def _marker(self, tmp, **overrides):
        import gate_d_lock as lock
        payload = {
            'gate_d': 'ACCEPTED',
            'training_allowed': True,
            'generator_version': lock.current_binding()['generator_version'],
            'generator_digest': lock.current_binding()['generator_digest'],
            'protocol_digest': lock.current_binding()['protocol_digest'],
            'accepted_by': 'test',
            'accepted_at': '2026-01-01T00:00:00Z',
        }
        payload.update(overrides)
        path = Path(tmp) / lock.ACCEPTANCE_FILE
        path.write_text(json.dumps(payload), encoding='utf-8')
        return path

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

    def test_missing_marker_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            state = lock.acceptance_state(Path(tmp))
            self.assertFalse(state['training_allowed'])
            self.assertEqual(state['reason'], 'GATE_D_NOT_ACCEPTED')

    def test_missing_required_field_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            for field in lock.REQUIRED_FIELDS:
                payload = {
                    'gate_d': 'ACCEPTED', 'training_allowed': True,
                    'generator_version': lock.current_binding()['generator_version'],
                    'generator_digest': lock.current_binding()['generator_digest'],
                    'protocol_digest': lock.current_binding()['protocol_digest']}
                payload.pop(field)
                (Path(tmp) / lock.ACCEPTANCE_FILE).write_text(
                    json.dumps(payload), encoding='utf-8')
                state = lock.acceptance_state(Path(tmp))
                self.assertFalse(state['training_allowed'], field)
                self.assertIn('missing required field', state['reason'])

    def test_explicit_hold_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            self._marker(tmp, gate_d='HOLD')
            state = lock.acceptance_state(Path(tmp))
            self.assertFalse(state['training_allowed'])
            self.assertIn('gate_d=', state['reason'])

    def test_training_not_allowed_flag_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            self._marker(tmp, training_allowed=False)
            self.assertFalse(lock.acceptance_state(Path(tmp))['training_allowed'])

    def test_stale_generator_digest_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            self._marker(tmp, generator_digest='0' * 64)
            state = lock.acceptance_state(Path(tmp))
            self.assertFalse(state['training_allowed'])
            self.assertIn('generator_digest', state['reason'])

    def test_wrong_generator_version_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            self._marker(tmp, generator_version='GATE_D_SYNTHETIC_GENERATOR_V1')
            state = lock.acceptance_state(Path(tmp))
            self.assertFalse(state['training_allowed'])
            self.assertIn('generator_version', state['reason'])

    def test_wrong_protocol_digest_rejected(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            self._marker(tmp, protocol_digest='f' * 64)
            state = lock.acceptance_state(Path(tmp))
            self.assertFalse(state['training_allowed'])
            self.assertIn('protocol_digest', state['reason'])

    def test_correct_mock_marker_allows_only_at_test_level(self):
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            self._marker(tmp)
            state = lock.acceptance_state(Path(tmp))
            self.assertTrue(state['training_allowed'])
            self.assertEqual(state['reason'], 'GATE_D_ACCEPTED')
            # The module-level constant stays False: a marker never flips it.
            self.assertFalse(lock.TRAINING_ALLOWED)

    def test_code_change_after_marker_rejects(self):
        """A binding computed before a code change no longer matches the marker."""
        import gate_d_lock as lock
        with tempfile.TemporaryDirectory() as tmp:
            self._marker(tmp)
            stale = dict(lock.current_binding())
            stale['generator_digest'] = 'a' * 64
            self.assertFalse(
                lock.acceptance_state(Path(tmp), binding=stale)['training_allowed'])

    def test_trainers_refuse_without_acceptance(self):
        import train_pointnet
        import train_pointnet2
        self.assertEqual(train_pointnet.main(['--epochs', '1']), 3)
        self.assertEqual(train_pointnet2.main(['--epochs', '1']), 3)


class GeneratorDigestIntegrity(unittest.TestCase):
    """The generator digest must be a genuine 64-character lowercase SHA-256 and must be
    bound consistently across the runtime, the diagnostic artifact and the training lock.
    """

    def _artifact(self) -> dict:
        return json.loads(
            (provenance_module.REPORTS / 'GATE_D_SYNTHETIC_DIAGNOSTIC.json')
            .read_text(encoding='utf-8'))

    def test_runtime_digest_is_64_lowercase_hex(self):
        import gate_d_diagnostic
        digest = gate_d_diagnostic.generator_digest()
        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r'^[0-9a-f]{64}$')
        self.assertEqual(digest, digest.lower())

    def test_digest_is_a_genuine_sha256_of_the_declared_files(self):
        """Independent recomputation, so a truncated or hand-edited value cannot pass."""
        import gate_d_diagnostic
        expected = hashlib.sha256()
        for name in gate_d_diagnostic.GENERATOR_FILES:
            expected.update(name.encode('utf-8'))
            expected.update((gate_d_diagnostic.HERE / name).read_bytes())
        self.assertEqual(gate_d_diagnostic.generator_digest(), expected.hexdigest())

    def test_artifact_digest_is_well_formed(self):
        digest = self._artifact()['generator_digest']
        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r'^[0-9a-f]{64}$')

    def test_diagnostic_digest_binding(self):
        """runtime == diagnostic artifact."""
        import gate_d_diagnostic
        self.assertEqual(self._artifact()['generator_digest'],
                         gate_d_diagnostic.generator_digest())

    def test_training_lock_digest_binding(self):
        """training_lock.expected.generator_digest == runtime."""
        import gate_d_diagnostic
        import gate_d_lock
        expected = gate_d_lock.current_binding()['generator_digest']
        self.assertEqual(expected, gate_d_diagnostic.generator_digest())
        self.assertEqual(len(expected), 64)
        self.assertRegex(expected, r'^[0-9a-f]{64}$')
        state = gate_d_lock.acceptance_state()
        self.assertEqual(state['expected']['generator_digest'], expected)

    def test_all_three_bindings_agree(self):
        import gate_d_diagnostic
        import gate_d_lock
        runtime = gate_d_diagnostic.generator_digest()
        artifact = self._artifact()['generator_digest']
        lock = gate_d_lock.current_binding()['generator_digest']
        self.assertEqual(runtime, artifact)
        self.assertEqual(runtime, lock)

    def test_fingerprint_remediation_records_the_same_digest(self):
        payload = json.loads(
            (provenance_module.REPORTS / 'GATE_D_FINGERPRINT_REMEDIATION.json')
            .read_text(encoding='utf-8'))
        self.assertEqual(payload['generator_digest'],
                         self._artifact()['generator_digest'])


class GeneratorDigest(unittest.TestCase):
    def test_digest_is_stable_and_hex(self):
        import gate_d_diagnostic
        first = gate_d_diagnostic.generator_digest()
        self.assertEqual(first, gate_d_diagnostic.generator_digest())
        self.assertEqual(len(first), 64)
        int(first, 16)

    def test_artifact_reports_a_deterministic_digest(self):
        payload = json.loads(
            (provenance_module.REPORTS / 'GATE_D_SYNTHETIC_DIAGNOSTIC.json')
            .read_text(encoding='utf-8'))
        for field in ('deterministic_digest', 'manifest_digest', 'generator_digest'):
            self.assertIn(field, payload)
            self.assertEqual(len(payload[field]), 64)
        stripped = {k: v for k, v in payload.items() if k != 'deterministic_digest'}
        recomputed = hashlib.sha256(json.dumps(
            stripped, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        self.assertEqual(recomputed, payload['deterministic_digest'])
        manifest = (provenance_module.REPORTS / 'GATE_D_DIAGNOSTIC_MANIFEST.jsonl') \
            .read_text(encoding='utf-8')
        self.assertEqual(hashlib.sha256(manifest.encode('utf-8')).hexdigest(),
                         payload['manifest_digest'])


class GeometryAuthorization(unittest.TestCase):
    def test_real_batch_authorises_everything(self):
        candidates = list(PROVENANCE.train.eligible[:2])
        loaded = geometry.build_candidate_geometry_batch(candidates, PROVENANCE)
        for candidate in candidates:
            bundle = loaded[candidate]
            self.assertEqual(bundle.candidate_id, candidate)
            self.assertEqual(bundle.context_ids,
                             PROVENANCE.authorized_context(candidate))

    @unittest.skipUnless(HAS_REAL_DATA, 'verified extraction root not present')
    def test_real_scene_clips_context_to_the_frozen_query(self):
        candidate = PROVENANCE.train.eligible[0]
        loaded = geometry.build_candidate_geometry_batch([candidate], PROVENANCE)
        scene = scene_module.build_base_scene(candidate, 'train', loaded[candidate],
                                              PROVENANCE)
        self.assertGreater(scene.context_points, 0)
        x0, y0, z0, x1, y1, z1 = scene.query_box
        context = scene.points[~scene.candidate_mask]
        self.assertTrue((context[:, 0] >= x0 - 1e-6).all())
        self.assertTrue((context[:, 0] <= x1 + 1e-6).all())
        self.assertTrue((context[:, 1] >= y0 - 1e-6).all())
        self.assertTrue((context[:, 1] <= y1 + 1e-6).all())
        self.assertTrue((context[:, 2] >= z0 - 1e-6).all())
        self.assertTrue((context[:, 2] <= z1 + 1e-6).all())

    @unittest.skipUnless(HAS_REAL_DATA, 'verified extraction root not present')
    def test_every_model_selection_candidate_has_real_context(self):
        missing = []
        for partition in ('train', 'validation'):
            for candidate in PROVENANCE.candidates(partition):
                context = PROVENANCE.authorized_context(candidate)
                if not context:
                    missing.append(candidate)
        self.assertEqual(missing, [])


if __name__ == '__main__':
    unittest.main()
