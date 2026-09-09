"""Focused identity tests. Synthetic fixtures only; no publisher data required."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import canonical_identity as ci
import numpy as np

TMP_ROOT = Path('D:/TreeQCTraining/tmp')


def write_ply(path, points, instance=None, semantic=None):
    """Minimal binary little-endian PLY vertex element."""
    points = np.asarray(points, dtype=np.float32)
    fields = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if instance is not None:
        fields.append(('instance', 'i4'))
    if semantic is not None:
        fields.append(('semantic', 'i4'))
    data = np.zeros(points.shape[0], dtype=np.dtype(fields))
    data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], points[:, 2]
    if instance is not None:
        data['instance'] = np.asarray(instance, dtype=np.int32)
    if semantic is not None:
        data['semantic'] = np.asarray(semantic, dtype=np.int32)
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {points.shape[0]}']
    for name, kind in fields:
        ply_kind = np.dtype(kind).name.replace('int32', 'int').replace('float32', 'float')
        header.append(f'property {ply_kind} {name}')
    header.append('end_header')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        handle.write(('\n'.join(header) + '\n').encode('ascii'))
        handle.write(data.tobytes())


def cloud(seed, count=200, origin=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return rng.random((count, 3)).astype(np.float32) * 2.0 + np.asarray(origin, dtype=np.float32)


def tree_dir(root, site, folder, basename, points, instance=None, semantic=None):
    path = root / site / 'trees' / folder / f'{basename}.ply'
    write_ply(path, points, instance=instance, semantic=semantic)
    return path


def manifest_for(root):
    sources = ci.discover_tree_sources(root, keep_keys=True)
    pairs = {}
    for site in sorted({s.site for s in sources}):
        pairs.update(ci.shared_cell_pairs([s for s in sources if s.site == site]))
    rows, summary, evidence = ci.canonical_manifest(sources, pairs)
    return rows, summary, evidence


def canonical_rows(root):
    sources = ci.discover_tree_sources(root, keep_keys=True)
    return ci.canonical_manifest(sources, {})[0]


class IdentityFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='gate_c2_identity_', dir=str(TMP_ROOT)))

    def rows_by_basename(self, rows, basename):
        return [r for r in rows if r['basenames'] == [basename]]


class ExactAlias(IdentityFixture):
    def test_exact_duplicate_across_folders_is_one_canonical_tree(self):
        points = cloud(1)
        tree_dir(self.root, 'litchfield', 'train', 'alpha', points, instance=7)
        tree_dir(self.root, 'litchfield', 'val', 'alpha', points, instance=7)
        rows, summary, _ = manifest_for(self.root)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['occurrence_count'], 2)
        self.assertEqual(row['alias_evidence'], ci.LEVEL_A)
        self.assertEqual(row['identity_status'], ci.IDENTITY_EXACT)
        self.assertEqual(summary['boundary_duplicate_groups'], 1)
        self.assertEqual(sorted(row['folder_partitions']), ['train', 'val'])

    def test_different_instance_ids_same_coordinates_stay_one_tree(self):
        points = cloud(2)
        tree_dir(self.root, 'litchfield', 'train', 'beta', points, instance=11)
        tree_dir(self.root, 'litchfield', 'val', 'beta', points, instance=42)
        rows, _, _ = manifest_for(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['local_instance_id'], [11, 42])
        self.assertEqual(rows[0]['identity_status'], ci.IDENTITY_EXACT)

    def test_exact_aliases_without_basename_agreement_are_unresolved(self):
        points = cloud(3)
        tree_dir(self.root, 'litchfield', 'train', 'gamma', points)
        tree_dir(self.root, 'litchfield', 'val', 'delta', points)
        rows, summary, _ = manifest_for(self.root)
        self.assertEqual(summary['basename_identity_conflicts'], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['identity_status'], ci.IDENTITY_UNRESOLVED)
        self.assertEqual(len(rows[0]['basenames']), 2)


class LocalInstanceIds(IdentityFixture):
    def test_same_instance_id_different_trees_are_distinct(self):
        tree_dir(self.root, 'litchfield', 'train', 'eps', cloud(4, origin=(0, 0, 0)), instance=5)
        tree_dir(self.root, 'litchfield', 'train', 'zeta', cloud(5, origin=(50, 50, 0)),
                 instance=5)
        rows, _, _ = manifest_for(self.root)
        self.assertEqual(len(rows), 2)
        seen = {}
        for row in rows:
            for instance in row['local_instance_id']:
                seen.setdefault(instance, set()).add(row['canonical_tree_id'])
        self.assertEqual(len(seen[5]), 2, 'one local id must not collapse two trees')


class PartialOverlap(IdentityFixture):
    def test_shared_cells_are_not_an_alias(self):
        base = cloud(6, count=400)
        first = base[:300]
        second = np.concatenate([base[280:400], cloud(7, count=120, origin=(0, 0, 0))])
        tree_dir(self.root, 'litchfield', 'train', 'eta', first)
        tree_dir(self.root, 'litchfield', 'val', 'theta', second)
        rows, summary, evidence = manifest_for(self.root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary['strong_alias_groups'], 0)
        self.assertEqual(summary['partial_overlap_nonaliases'], 1)
        self.assertEqual(evidence[0]['level'], ci.LEVEL_C)
        self.assertLess(evidence[0]['containment_of_smaller'], ci.STRONG_CONTAINMENT)

    def test_containment_above_threshold_is_review_not_union(self):
        base = cloud(8, count=400)
        tree_dir(self.root, 'litchfield', 'train', 'iota', base)
        tree_dir(self.root, 'litchfield', 'val', 'iota', base[:390])
        rows, summary, evidence = manifest_for(self.root)
        self.assertEqual(summary['strong_alias_groups'], 1)
        self.assertEqual(evidence[0]['level'], ci.LEVEL_B)
        self.assertEqual(len(rows), 2, 'LEVEL_B must not auto-union')


class Partitioning(IdentityFixture):
    def test_boundary_duplicate_receives_exactly_one_partition(self):
        import gate_c2_audit as audit
        points = cloud(9, origin=(0, 0, 0))
        tree_dir(self.root, 'litchfield', 'train', 'kappa', points, instance=1)
        tree_dir(self.root, 'litchfield', 'val', 'kappa', points, instance=1)
        rows, _, _ = manifest_for(self.root)
        row = rows[0]
        assigned = audit.assign_partition(row, boundary=10.0)
        self.assertEqual(assigned, 'train')
        self.assertEqual(audit.assign_partition(row, boundary=-10.0), 'validation')
        self.assertEqual(audit.assign_partition(row, boundary=1.0), 'quarantine')

    def test_held_out_site_and_unresolved_are_never_in_train(self):
        import gate_c2_audit as audit
        row = {'site': 'wytham', 'identity_status': ci.IDENTITY_EXACT,
               'bbox': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}
        self.assertEqual(audit.assign_partition(row, boundary=5.0), 'held_out')
        row['identity_status'] = ci.IDENTITY_UNRESOLVED
        self.assertEqual(audit.assign_partition(row, boundary=5.0), 'held_out')
        row['site'] = 'litchfield'
        self.assertEqual(audit.assign_partition(row, boundary=5.0), 'quarantine')

    def test_pools_are_disjoint_for_aliased_trees(self):
        import gate_c2_audit as audit
        points = cloud(10)
        tree_dir(self.root, 'litchfield', 'train', 'mu', points)
        tree_dir(self.root, 'litchfield', 'val', 'mu', points)
        rows, _, _ = manifest_for(self.root)
        for row in rows:
            row['partition'] = audit.assign_partition(row, boundary=10.0)
        pools = {}
        for row in rows:
            pools.setdefault(row['partition'], []).append(row['canonical_tree_id'])
        self.assertEqual(set(pools.get('train', [])) & set(pools.get('validation', [])), set())


class ContextIndex(IdentityFixture):
    def test_alias_occurrence_is_not_a_neighbour(self):
        points = cloud(11, origin=(0, 0, 0))
        tree_dir(self.root, 'litchfield', 'train', 'nu', points, instance=1)
        tree_dir(self.root, 'litchfield', 'val', 'nu', points, instance=1)
        rows, _, _ = manifest_for(self.root)
        index = ci.build_context_index(rows, pad=3.0)
        self.assertEqual(index[rows[0]['canonical_tree_id']], [])

    def test_real_neighbour_is_retained(self):
        tree_dir(self.root, 'litchfield', 'train', 'xi', cloud(12, origin=(0, 0, 0)))
        tree_dir(self.root, 'litchfield', 'train', 'omicron',
                 cloud(13, origin=(1.0, 1.0, 0.0)))
        rows, _, _ = manifest_for(self.root)
        index = ci.build_context_index(rows, pad=3.0)
        self.assertEqual(len(index[rows[0]['canonical_tree_id']]), 1)
        self.assertEqual(len(index[rows[1]['canonical_tree_id']]), 1)

    def test_distant_tree_is_not_a_neighbour(self):
        tree_dir(self.root, 'litchfield', 'train', 'pi', cloud(14, origin=(0, 0, 0)))
        tree_dir(self.root, 'litchfield', 'train', 'rho', cloud(15, origin=(90, 90, 0)))
        rows, _, _ = manifest_for(self.root)
        index = ci.build_context_index(rows, pad=3.0)
        self.assertEqual(index[rows[0]['canonical_tree_id']], [])


class Determinism(IdentityFixture):
    def test_canonical_ids_and_digest_are_stable_across_reruns(self):
        points = cloud(16)
        tree_dir(self.root, 'litchfield', 'train', 'sigma', points)
        tree_dir(self.root, 'litchfield', 'val', 'sigma', points)
        tree_dir(self.root, 'wytham', 'train', 'tau', cloud(17, origin=(5, 5, 0)))
        first_rows, _, _ = manifest_for(self.root)
        second_rows, _, _ = manifest_for(self.root)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(ci.manifest_digest(first_rows), ci.manifest_digest(second_rows))

    def test_digest_ignores_row_order(self):
        rows = [{'canonical_tree_id': 'a'}, {'canonical_tree_id': 'b'}]
        shuffled = list(rows)
        random.Random(0).shuffle(shuffled)
        self.assertEqual(ci.manifest_digest(rows), ci.manifest_digest(shuffled))

    def test_manifest_payload_is_json_serialisable(self):
        tree_dir(self.root, 'litchfield', 'train', 'upsilon', cloud(18))
        rows, _, _ = manifest_for(self.root)
        self.assertEqual(json.loads(json.dumps(rows)), rows)

    def test_shared_cell_pairs_is_independent_of_source_order(self):
        base = cloud(19, count=200)
        tree_dir(self.root, 'litchfield', 'train', 'phi', base)
        tree_dir(self.root, 'litchfield', 'val', 'chi', base[:120])
        sources = ci.discover_tree_sources(self.root, keep_keys=True)
        forward = ci.shared_cell_pairs(list(sources))
        reverse = ci.shared_cell_pairs(list(reversed(sources)))
        self.assertEqual(forward, reverse)
        self.assertEqual(list(forward.values()), [120])
        key = next(iter(forward))
        self.assertLess(key[0], key[1])


class PlyReading(IdentityFixture):
    def test_signed_zero_cells_are_equal(self):
        positive = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        negative = np.array([[-0.0, 1.0, 2.0]], dtype=np.float32)
        self.assertEqual(ci.fingerprint(ci.keys_of(positive)),
                         ci.fingerprint(ci.keys_of(negative)))

    def test_header_rejects_list_property(self):
        path = self.root / 'bad.ply'
        path.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 1\n'
                         b'property list uchar int idx\nend_header\n')
        with self.assertRaises(ValueError):
            ci.read_ply_header(path)

    def test_membership_matches_exact_cells_only(self):
        plot = self.root / 'plot.ply'
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        write_ply(plot, points, instance=[3, 4], semantic=[1, 1])
        keys, instance = ci.plot_records(plot)
        query = ci.keys_of(np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float32))
        result = ci.match_keys(query, keys, instance)
        self.assertEqual(result['matched_cells'], 1)
        self.assertEqual(result['instance_ids'], [3])
        self.assertEqual(result['instance_counts'], [1])


if __name__ == '__main__':
    unittest.main()
