"""Gate C2 canonical identity audit.

Read-only over the verified extraction root. Produces derived research artifacts:
an identity manifest, alias evidence, plot membership, leakage-safe pools and a
representative case review. No publisher data is modified, no model is trained,
and no biological identity is unioned on weak evidence.

Usage:
    python gate_c2_audit.py identity
    python gate_c2_audit.py membership
    python gate_c2_audit.py pools
    python gate_c2_audit.py cases
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import canonical_identity as ci
import numpy as np

DATA_ROOT = Path('D:/TreeQCData/extracted_manual_verified')
REPORTS = Path(__file__).resolve().parent / 'reports'
SITE_ROLES = {'litchfield': 'model_selection', 'wytham': 'held_out'}
QUERY_PAD_M = 3.0
MIN_SUPPORT = 30


def save(name: str, payload) -> Path:
    path = REPORTS / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    return path


def load(name: str):
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))


def sites() -> list[str]:
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


def phase_identity() -> None:
    started = time.time()
    sources = ci.discover_tree_sources(DATA_ROOT, keep_keys=False)
    per_site_pairs: dict[str, dict] = {}
    for site in sorted({s.site for s in sources}):
        site_sources = [s for s in sources if s.site == site]
        pairs = ci.shared_cell_pairs(site_sources)
        per_site_pairs[site] = pairs
        print(f'{site}: {len(site_sources)} sources, {len(pairs)} sharing pairs', flush=True)

    rows: list[dict] = []
    summary: dict = {}
    evidence: list[dict] = []
    for site in sorted({s.site for s in sources}):
        site_sources = sorted((s for s in sources if s.site == site),
                              key=lambda s: s.relpath)
        site_rows, site_summary, site_evidence = ci.canonical_manifest(
            site_sources, per_site_pairs[site])
        rows.extend(site_rows)
        evidence.extend(site_evidence)
        for key, value in site_summary.items():
            if key in ('source_file_occurrences', 'source_basenames',
                       'exact_alias_groups', 'exact_alias_extra_occurrences',
                       'strong_alias_groups', 'unresolved_groups',
                       'boundary_duplicate_groups', 'partial_overlap_nonaliases',
                       'basename_identity_conflicts', 'cross_basename_strong_candidates',
                       'canonical_tree_count'):
                summary[key] = summary.get(key, 0) + value
            else:
                summary[key] = max(summary.get(key, 0), value)
    rows.sort(key=lambda r: (r['site'], r['canonical_tree_id']))
    evidence.sort(key=lambda e: (-e['shared_cells'], e['a'], e['b']))
    summary['manifest_digest'] = ci.manifest_digest(rows)
    summary['manifest_digest_full'] = ci.full_manifest_digest(rows)
    summary['seconds'] = round(time.time() - started, 3)
    summary['source_rows'] = [s.as_row() for s in sources]
    save('GATE_C2_IDENTITY_MANIFEST.json', rows)
    save('GATE_C2_ALIAS_EVIDENCE.json', evidence)
    save('GATE_C2_IDENTITY_SUMMARY.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_rows'}, indent=2))


def phase_membership() -> None:
    started = time.time()
    records = []
    all_sources = ci.discover_tree_sources(DATA_ROOT, keep_keys=False)
    for site in sites():
        site_root = DATA_ROOT / site
        sources = [source for source in all_sources if source.site == site]
        if not sources:
            raise RuntimeError(f'no tree sources discovered for site {site}')
        for plot in sorted(site_root.glob('*.ply')):
            t0 = time.time()
            keys, instance = ci.plot_records(plot)
            print(f'{site}/{plot.name}: {keys.size} cells in {time.time()-t0:.1f}s',
                  flush=True)
            for source in sources:
                data = ci.read_points(Path(source.path))
                query = np.unique(ci.keys_of(ci.xyz_native(data)))
                result = ci.match_keys(query, keys, instance)
                top = sorted(zip(result['instance_ids'], result['instance_counts'],
                                 strict=True), key=lambda x: (-x[1], x[0]))[:8]
                dominant = top[0] if top else (None, 0)
                records.append({
                    'site': site,
                    'source': source.relpath,
                    'canonical_hint': f'{site}:{source.basename}',
                    'plot': plot.name,
                    'native_instance_ids': list(source.native_instance_ids),
                    'point_count': source.point_count,
                    'unique_cells': int(query.size),
                    'matched_cells': result['matched_cells'],
                    'matched_rows': result['matched_rows'],
                    'match_fraction': result['matched_cell_fraction'],
                    'distinct_instances': len(result['instance_ids']),
                    'dominant_instance': int(dominant[0]) if top else None,
                    'dominant_points': int(dominant[1]),
                    'dominant_share': (round(dominant[1] / result['matched_rows'], 6)
                                       if result['matched_rows'] else 0.0),
                    'top_instances': [{'instance': int(i), 'points': int(c)}
                                      for i, c in top],
                })
            del keys, instance
    if not records:
        raise RuntimeError('membership produced no records')
    save('GATE_C2_PLOT_MEMBERSHIP.json', records)
    print(f'membership records: {len(records)} in {time.time()-started:.1f}s')


def phase_instances() -> None:
    """Per-plot instance-id namespaces and the merged-plot namespace per site."""
    payload = {}
    for site in sites():
        site_root = DATA_ROOT / site
        union: set[int] = set()
        per_plot = {}
        for plot in sorted(site_root.glob('*.ply')):
            ids: set[int] = set()
            rows = 0
            for block in ci.iter_points(plot):
                ids.update(int(v) for v in np.unique(block['instance']))
                rows += block.size
            trees = sorted(v for v in ids if v > 0)
            per_plot[plot.name] = {
                'rows': rows,
                'ground_present': -1 in ids,
                'tree_ids': len(trees),
                'tree_id_min': trees[0] if trees else None,
                'tree_id_max': trees[-1] if trees else None,
            }
            union |= set(trees)
        ordered = sorted(union)
        payload[site] = {
            'plots': per_plot,
            'merged_tree_ids': len(ordered),
            'merged_id_min': ordered[0] if ordered else None,
            'merged_id_max': ordered[-1] if ordered else None,
            'merged_ids_sequential_from_1': ordered == list(range(1, len(ordered) + 1)),
            'merged_id_list': ordered,
        }
        print(f'{site}: merged ids {len(ordered)} '
              f'{ordered[0] if ordered else None}..{ordered[-1] if ordered else None}',
              flush=True)
    save('GATE_C2_INSTANCE_NAMESPACE.json', payload)


def phase_pools() -> None:
    manifest = load('GATE_C2_IDENTITY_MANIFEST.json')
    membership = load('GATE_C2_PLOT_MEMBERSHIP.json')
    evidence = load('GATE_C2_ALIAS_EVIDENCE.json')
    by_source: dict[tuple[str, str], list[dict]] = {}
    for row in membership:
        by_source.setdefault((row['site'], row['source']), []).append(row)

    # Dominant plot instance per canonical tree, with cross-plot agreement.
    # Aliases are exact coordinate duplicates, so their membership records are
    # identical; only the representative occurrence is counted to avoid double
    # counting one physical observation.
    for row in manifest:
        votes: dict[int, int] = {}
        support = 0
        for record in by_source.get((row['site'], row['source_file']), []):
            if record['match_fraction'] > 0 and record['dominant_instance'] is not None:
                instance = record['dominant_instance']
                votes[instance] = votes.get(instance, 0) + record['dominant_points']
                support += record['matched_cells']
        row['plot_instance_votes'] = dict(sorted(votes.items(), key=lambda x: -x[1]))
        row['plot_support_cells'] = support
        row['dominant_plot_instance'] = (max(votes, key=lambda k: votes[k])
                                         if votes else None)
        native = row['local_instance_id']
        row['native_equals_dominant'] = (len(native) == 1
                                         and row['dominant_plot_instance'] == native[0])

    # A local instance id must never name two canonical trees.
    collisions: dict[tuple[str, int], list[str]] = {}
    for row in manifest:
        for instance in row['local_instance_id']:
            collisions.setdefault((row['site'], instance), []).append(
                row['canonical_tree_id'])
    collisions = {f'{s}:{i}': sorted(set(ids)) for (s, i), ids in collisions.items()
                  if len(set(ids)) > 1}

    # Cross-site identity leakage: identical coordinate fingerprints across sites.
    cross = sorted(f"{left['canonical_tree_id']}|{right['canonical_tree_id']}"
                   for left in manifest for right in manifest
                   if left['site'] < right['site']
                   and left['coordinate_fingerprint'] == right['coordinate_fingerprint'])

    litch = [r for r in manifest if r['site'] == 'litchfield']
    centroids = sorted(r['centroid'][0] for r in litch)
    index = int(0.7 * (len(centroids) - 1))
    boundary = centroids[index]
    for row in manifest:
        row['partition'] = assign_partition(row, boundary)

    rows_by_id = {row['canonical_tree_id']: row for row in manifest}
    by_relpath_row = {alias: row for row in manifest for alias in row['source_aliases']}
    index_by_id = ci.build_context_index(manifest, QUERY_PAD_M)
    points_cache: dict[str, np.ndarray] = {}

    def cached_points(relpath: str, site: str) -> np.ndarray:
        if relpath not in points_cache:
            data = ci.read_points(DATA_ROOT / site / relpath)
            points_cache[relpath] = ci.xyz_native(data)
        return points_cache[relpath]

    for row in manifest:
        box = ci.query_box(row, QUERY_PAD_M)
        neighbours = index_by_id[row['canonical_tree_id']]
        support_points = 0
        with_points = []
        for neighbour_id in neighbours:
            other = rows_by_id[neighbour_id]
            pts = cached_points(other['source_file'], other['site'])
            inside = ((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[3])
                      & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[4])
                      & (pts[:, 2] >= box[2]) & (pts[:, 2] <= box[5]))
            count = int(np.count_nonzero(inside))
            if count:
                support_points += count
                with_points.append(neighbour_id)
        row['neighbour_count'] = len(neighbours)
        row['neighbours_with_points'] = len(with_points)
        row['neighbour_support_points'] = support_points
        row['neighbour_ids'] = with_points
        points_cache.clear()

    for row in manifest:
        core_blockers = [name for name, failed in (
            ('not_held_out_site', row['site'] != 'wytham'),
            ('identity_not_exact', row['identity_status'] != ci.IDENTITY_EXACT),
            ('no_neighbour_points', row['neighbours_with_points'] <= 0),
            ('too_few_points', row['point_count'] < 4096)) if failed]
        row['held_out_core_eligible'] = not core_blockers
        row['held_out_blockers'] = core_blockers
        if row['plot_support_cells'] <= 0:
            row['held_out_blockers'] = core_blockers + ['no_plot_support']
        row['held_out_eligible'] = not row['held_out_blockers']

    pools: dict[str, list[str]] = {'train': [], 'validation': [], 'held_out': [],
                                   'quarantine': []}
    for row in manifest:
        pools[row['partition']].append(row['canonical_tree_id'])
    held_out = sorted(r['canonical_tree_id'] for r in manifest if r['held_out_eligible'])
    held_out_core = sorted(r['canonical_tree_id'] for r in manifest
                           if r['held_out_core_eligible'])
    wytham = [r for r in manifest if r['site'] == 'wytham']
    high_pairs = [e for e in evidence if e['containment_of_smaller'] >= 0.30]
    affected = sorted({f"{'wytham'}:{Path(name).stem}"
                       for item in high_pairs for name in (item['a'], item['b'])})
    held_out_strict = [cid for cid in held_out if cid not in set(affected)]
    held_out_core_strict = [cid for cid in held_out_core if cid not in set(affected)]
    payload = {
        'site_roles': SITE_ROLES,
        'query_pad_m': QUERY_PAD_M,
        'minimum_support': MIN_SUPPORT,
        'litchfield_boundary_x_m': boundary,
        'litchfield_boundary_index': index,
        'litchfield_canonical_count': len(centroids),
        'pools': {k: sorted(v) for k, v in pools.items()},
        'pool_sizes': {k: len(v) for k, v in pools.items()},
        'leakage': {
            'train_validation_overlap': sorted(set(pools['train'])
                                               & set(pools['validation'])),
            'train_heldout_overlap': sorted(set(pools['train']) & set(pools['held_out'])),
            'validation_heldout_overlap': sorted(set(pools['validation'])
                                                 & set(pools['held_out'])),
            'cross_site_fingerprint_matches': cross,
        },
        'instance_id_collisions': collisions,
        'native_matches_dominant': {
            'total': len(manifest),
            'agree': sum(1 for r in manifest if r['native_equals_dominant']),
            'disagree': sum(1 for r in manifest
                            if r['local_instance_id'] and not r['native_equals_dominant']),
            'no_native_id': sum(1 for r in manifest if not r['local_instance_id']),
        },
        'held_out_pool': {
            'eligible': held_out,
            'eligible_count': len(held_out),
            'eligible_core_count': len(held_out_core),
            'wytham_canonical_count': len(wytham),
            'blocked_count': sum(1 for r in wytham if not r['held_out_eligible']),
            'blockers': _counter(b for r in wytham if not r['held_out_eligible']
                                 for b in r['held_out_blockers']),
            'core_blockers': _counter(b for r in wytham if not r['held_out_core_eligible']
                                      for b in r['held_out_blockers']
                                      if b != 'no_plot_support'),
            'high_containment_affected': affected,
            'eligible_excluding_high_containment': held_out_strict,
            'eligible_excluding_high_containment_count': len(held_out_strict),
            'eligible_core_excluding_high_containment_count': len(held_out_core_strict),
        },
        'high_containment_pairs': [dict(item, **{
            'a_dominant_instance': by_relpath_row[item['a']]['dominant_plot_instance'],
            'b_dominant_instance': by_relpath_row[item['b']]['dominant_plot_instance'],
            'publisher_labels_agree': (
                by_relpath_row[item['a']]['dominant_plot_instance'] is not None
                and by_relpath_row[item['a']]['dominant_plot_instance']
                == by_relpath_row[item['b']]['dominant_plot_instance']),
        }) for item in high_pairs],
    }
    save('GATE_C2_IDENTITY_MANIFEST.json', manifest)
    save('GATE_C2_PARTITION_POOLS.json', payload)
    save('DEDUP_REPORT.json', {
        'protocol': 'GATE_C_CANONICAL_IDENTITY_PROTOCOL_V2',
        'superseded_protocol': 'GATE_C_SPATIAL_PROTOCOL_V1 (retained unchanged)',
        'source_file_occurrences': load('GATE_C2_IDENTITY_SUMMARY.json')[
            'source_file_occurrences'],
        'canonical_tree_count': len(manifest),
        'exact_alias_groups': load('GATE_C2_IDENTITY_SUMMARY.json')['exact_alias_groups'],
        'boundary_duplicate_groups': load('GATE_C2_IDENTITY_SUMMARY.json')[
            'boundary_duplicate_groups'],
        'unresolved_groups': load('GATE_C2_IDENTITY_SUMMARY.json')['unresolved_groups'],
        'pool_sizes': payload['pool_sizes'],
        'leakage': {k: len(v) for k, v in payload['leakage'].items()},
        'minimum_support': payload['minimum_support'],
        'held_out_pool': {
            'eligible_count': payload['held_out_pool']['eligible_count'],
            'eligible_excluding_high_containment_count':
                payload['held_out_pool']['eligible_excluding_high_containment_count'],
        },
        'manifest_digest': load('GATE_C2_IDENTITY_SUMMARY.json')['manifest_digest'],
    })
    print(json.dumps(payload['pool_sizes'], indent=2))
    print('held_out eligible', len(held_out), 'of', len(wytham))
    print('boundary', boundary, 'collisions', len(collisions))


def _counter(values) -> dict:
    out: dict = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def assign_partition(row: dict, boundary: float) -> str:
    """Whole-canonical-tree partition assignment with a fixed 3 m buffer."""
    if SITE_ROLES.get(row['site']) == 'held_out':
        return 'held_out'
    if row['site'] != 'litchfield':
        return 'quarantine'
    if row['identity_status'] != ci.IDENTITY_EXACT:
        return 'quarantine'
    if row['bbox'][3] + QUERY_PAD_M < boundary:
        return 'train'
    if row['bbox'][0] - QUERY_PAD_M > boundary:
        return 'validation'
    return 'quarantine'


def phase_cases() -> None:
    manifest = load('GATE_C2_IDENTITY_MANIFEST.json')
    evidence = load('GATE_C2_ALIAS_EVIDENCE.json')
    membership = load('GATE_C2_PLOT_MEMBERSHIP.json')
    by_source: dict[tuple[str, str], list[dict]] = {}
    for row in membership:
        by_source.setdefault((row['site'], row['source']), []).append(row)
    rows_by_id = {row['canonical_tree_id']: row for row in manifest}

    cases = {'A_exact_alias': [], 'B_partial_overlap_nonalias': [],
             'C_boundary_duplicate': [], 'D_unresolved': []}
    for row in manifest:
        if row['occurrence_count'] > 1 and len(cases['A_exact_alias']) < 5:
            cases['A_exact_alias'].append(summarize(row, by_source))
    for item in evidence:
        if item['level'] == ci.LEVEL_C and len(cases['B_partial_overlap_nonalias']) < 5:
            cases['B_partial_overlap_nonalias'].append(item)
    for row in manifest:
        if len(row['folder_partitions']) > 1 and len(cases['C_boundary_duplicate']) < 5:
            cases['C_boundary_duplicate'].append(summarize(row, by_source))
    def row_of(relpath: str) -> dict:
        basename = Path(relpath).stem
        for candidate in (f'wytham:{basename}', f'litchfield:{basename}'):
            if candidate in rows_by_id:
                return rows_by_id[candidate]
        raise KeyError(f'no canonical tree for {relpath}')

    for item in sorted(evidence, key=lambda e: -e['containment_of_smaller'])[:5]:
        cases['D_unresolved'].append({
            'kind': 'high_partial_overlap_pair',
            'note': ('Not an exact alias. Kept as two independent canonical trees '
                     'because the publisher labels them as distinct instances; the '
                     'archive alone does not prove either verdict.'),
            'pair': item,
            'a_summary': summarize(row_of(item['a']), by_source),
            'b_summary': summarize(row_of(item['b']), by_source),
        })
    for row in sorted(manifest, key=lambda r: r['plot_support_cells'])[:3]:
        if len(cases['D_unresolved']) < 8:
            cases['D_unresolved'].append({
                'kind': 'weakest_plot_support', 'summary': summarize(row, by_source)})
    save('GATE_C2_CASE_REVIEW.json', cases)
    print(json.dumps({k: len(v) for k, v in cases.items() if isinstance(v, list)},
                     indent=2))


def summarize(row: dict, by_source) -> dict:
    points = ci.xyz_native(ci.read_points(DATA_ROOT / row['site'] / row['source_file']))
    return {
        'canonical_tree_id': row['canonical_tree_id'],
        'identity_status': row['identity_status'],
        'alias_evidence': row['alias_evidence'],
        'occurrence_count': row['occurrence_count'],
        'folder_partitions': row['folder_partitions'],
        'source_aliases': row['source_aliases'],
        'point_count': row['point_count'],
        'unique_cells': row['unique_cells'],
        'bbox': [round(v, 4) for v in row['bbox']],
        'centroid': [round(v, 4) for v in row['centroid']],
        'coordinate_stats': {
            'x': [round(float(points[:, 0].min()), 4), round(float(points[:, 0].max()), 4),
                  round(float(points[:, 0].mean()), 4), round(float(points[:, 0].std()), 4)],
            'y': [round(float(points[:, 1].min()), 4), round(float(points[:, 1].max()), 4),
                  round(float(points[:, 1].mean()), 4), round(float(points[:, 1].std()), 4)],
            'z': [round(float(points[:, 2].min()), 4), round(float(points[:, 2].max()), 4),
                  round(float(points[:, 2].mean()), 4), round(float(points[:, 2].std()), 4)],
            'non_finite': int(points.size - np.isfinite(points).sum()),
            'duplicate_cells': int(points.shape[0] - row['unique_cells']),
        },
        'local_instance_id': row['local_instance_id'],
        'dominant_plot_instance': row.get('dominant_plot_instance'),
        'plot_support_cells': row.get('plot_support_cells'),
        'neighbour_count': row.get('neighbour_count'),
        'neighbours_with_points': row.get('neighbours_with_points'),
        'neighbour_support_points': row.get('neighbour_support_points'),
        'membership': by_source.get((row['site'], row['source_file']), []),
    }


def phase_verify() -> None:
    """Independent re-computation of a deterministic sample plus invariant checks."""
    manifest = load('GATE_C2_IDENTITY_MANIFEST.json')
    pools = load('GATE_C2_PARTITION_POOLS.json')
    checks = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({'check': name, 'pass': bool(passed), 'detail': detail})

    step = max(1, len(manifest) // 40)
    mismatches = []
    for row in manifest[::step]:
        source = ci.describe_source(row['site'], DATA_ROOT / row['site'] / row['source_file'],
                                    DATA_ROOT / row['site'], keep_keys=False)
        if (source.fingerprint_f32 != row['coordinate_fingerprint']
                or source.point_count != row['point_count']):
            mismatches.append(row['canonical_tree_id'])
    check('sampled_fingerprint_recompute', not mismatches,
          {'sampled': len(manifest[::step]), 'mismatches': mismatches})

    alias_bad = [r['canonical_tree_id'] for r in manifest if r['occurrence_count'] > 1
                 and r['alias_evidence'] != ci.LEVEL_A]
    check('every_multi_occurrence_group_is_level_a', not alias_bad, alias_bad)

    duplicate_ids = len(manifest) - len({r['canonical_tree_id'] for r in manifest})
    check('canonical_ids_unique', duplicate_ids == 0, {'duplicates': duplicate_ids})

    alias_paths = [p for r in manifest for p in r['source_aliases']]
    check('every_source_occurrence_accounted_for',
          len(alias_paths) == len(set(alias_paths))
          and len(alias_paths) == load('GATE_C2_IDENTITY_SUMMARY.json')['source_file_occurrences'],
          {'alias_paths': len(alias_paths), 'distinct': len(set(alias_paths))})

    leak = pools['leakage']
    check('no_canonical_leakage',
          not (leak['train_validation_overlap'] or leak['train_heldout_overlap']
               or leak['validation_heldout_overlap']
               or leak['cross_site_fingerprint_matches']),
          {k: len(v) for k, v in leak.items()})
    check('minimum_support_preserved',
          pools['minimum_support'] == 30
          and pools['held_out_pool']['eligible_count'] >= 30,
          {'minimum': pools['minimum_support'],
           'eligible': pools['held_out_pool']['eligible_count']})
    check('no_instance_id_collisions', not pools['instance_id_collisions'],
          list(pools['instance_id_collisions'])[:5])
    check('digest_matches_summary',
          ci.manifest_digest(manifest)
          == load('GATE_C2_IDENTITY_SUMMARY.json')['manifest_digest'],
          ci.manifest_digest(manifest))

    save('GATE_C2_VERIFICATION.json', checks)
    print(json.dumps(checks, indent=2))
    print('ALL PASS' if all(c['pass'] for c in checks) else 'FAILURES PRESENT')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['identity', 'membership', 'instances',
                                          'pools', 'cases', 'verify'])
    args = parser.parse_args()
    {'identity': phase_identity, 'membership': phase_membership,
     'instances': phase_instances, 'pools': phase_pools,
     'cases': phase_cases, 'verify': phase_verify}[args.phase]()


if __name__ == '__main__':
    main()
