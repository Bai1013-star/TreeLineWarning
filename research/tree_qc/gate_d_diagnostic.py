"""Gate D bounded synthetic diagnostic.

Runs the generator and sampler over a small, representative set of train and validation
trees, covering all 7 classes x 3 severities, and writes the Gate D artifacts plus
representative visualisations. It never trains and never touches held-out trees.

Usage:
    python gate_d_diagnostic.py --per-partition 5 --repeats 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gate_d_fingerprint as fingerprint
import gate_d_generator as gen
import gate_d_geometry as geometry
import gate_d_lock as lock
import gate_d_sampler as sampler
import gate_d_scene as scene_module
import numpy as np
from gate_d_provenance import load_gate_d_provenance

HERE = Path(__file__).resolve().parent
REPORTS = HERE / 'reports'
DIAGNOSTIC_VERSION = 'GATE_D_SYNTHETIC_DIAGNOSTIC_V2'
GENERATOR_FILES = ('gate_d_provenance.py', 'gate_d_scene.py', 'gate_d_geometry.py',
                   'gate_d_generator.py', 'gate_d_sampler.py', 'gate_d_dataset.py',
                   'gate_d_metrics.py', 'gate_d_train.py', 'gate_d_fingerprint.py',
                   'gate_d_diagnostic.py')


def generator_digest() -> str:
    """Stable digest of the generator implementation, for the acceptance marker."""
    digest = hashlib.sha256()
    for name in GENERATOR_FILES:
        digest.update(name.encode('utf-8'))
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def select_representatives(provenance, partition: str, count: int) -> list[str]:
    """Evenly spaced candidates by point count: spans small to large real trees."""
    eligible = sorted(provenance.candidates(partition),
                      key=lambda c: (provenance.tree[c]['point_count'], c))
    if count >= len(eligible):
        return eligible
    positions = np.linspace(0, len(eligible) - 1, count).round().astype(int)
    return [eligible[int(p)] for p in dict.fromkeys(positions.tolist())]


def _candidate_stats(scene) -> dict:
    """Per-candidate scene facts that do not depend on the class.

    Computed once per candidate instead of once per example: the scene is identical
    across all classes of a candidate, and these are the expensive full-array probes.
    """
    points = scene.points
    extent = points.max(axis=0) - points.min(axis=0)
    _, unique_counts = np.unique(points, axis=0, return_counts=True)
    return {
        'extent': tuple(round(float(v), 6) for v in extent),
        'centroid_norm': round(float(np.linalg.norm(points.mean(axis=0))), 6),
        'duplicate_rate': round(float(1.0 - unique_counts.size / points.shape[0]), 6),
    }


def _record(spec, example, sample, coverage, height_cov,
            stats: dict) -> fingerprint.ExampleRecord:
    corruption = example.provenance['corruption']
    diagnostics = sample.diagnostics
    points = example.points
    return fingerprint.ExampleRecord(
        key=spec.key, candidate_id=spec.candidate_id, partition=spec.partition,
        example_class=spec.example_class, severity=spec.severity,
        label=example.label, point_count=int(points.shape[0]),
        sampled_points=int(sample.points),
        membership_fraction=round(float(example.membership.mean()), 6),
        gt_membership_fraction=example.provenance['gt_membership_fraction'],
        intended_membership_shift=round(
            float(example.membership.mean())
            - example.provenance['gt_membership_fraction'], 6),
        changed_membership_points=example.provenance['changed_membership_points'],
        changed_membership_fraction=example.provenance['changed_membership_fraction'],
        candidate_points=example.provenance['candidate_points'],
        context_points=example.provenance['context_points'],
        claimed_points=example.provenance['claimed_points'],
        dropped_points=example.provenance['dropped_points'],
        source_count=len(example.provenance['actual_point_source_ids']),
        corruption_fraction=float(corruption.get('fraction', 0.0)),
        drawn_fraction=float(corruption.get('drawn_fraction', 0.0)),
        corruption_axis=tuple(corruption['axis']) if corruption.get('axis') else None,
        membership_sampled=diagnostics['membership_sampled'],
        candidate_sampled=diagnostics['candidate_sampled'],
        context_sampled=diagnostics['context_sampled'],
        changed_evidence_sampled=diagnostics['changed_evidence_sampled'],
        repeat=int(spec.repeat),
        points_target=int(diagnostics['points_target']),
        sampling_seed=int(diagnostics['sampling_seed']),
        candidate_budget=int(diagnostics['candidate_budget']),
        context_budget=int(diagnostics['context_budget']),
        pre_reservation_indices_digest=diagnostics['pre_reservation_indices_digest'],
        final_indices_digest=diagnostics['final_indices_digest'],
        reservation_replacements=int(diagnostics['reservation_replacements']),
        reservation_shortfall=int(diagnostics['reservation_shortfall']),
        changed_evidence_available=int(diagnostics['changed_evidence_available']),
        scene_points=int(points.shape[0]),
        spatial_coverage=round(float(coverage), 6),
        height_coverage=round(float(height_cov), 6),
        feature_mean=tuple(float(v) for v in sample.features.mean(axis=0)),
        feature_std=tuple(float(v) for v in sample.features.std(axis=0)),
        bbox_extent=stats['extent'],
        centroid_norm=stats['centroid_norm'],
        duplicate_rate=stats['duplicate_rate'])


def run(per_partition: int = 5, repeats: int = 1, points: int = sampler.POINTS_PRIMARY,
        cache_root: Path | str = 'D:/TreeQCTraining/gate_d_cache',
        write_visuals: bool = True, seed_base: int = 4200) -> dict:
    lock.assert_not_training()
    started = time.time()
    provenance = load_gate_d_provenance()

    selected = {p: select_representatives(provenance, p, per_partition)
                for p in ('train', 'validation')}
    all_ids = [cid for ids in selected.values() for cid in ids]
    geometries = geometry.load_cached_geometry_batch(cache_root, provenance, all_ids)

    manifest_rows: list[dict] = []
    records: list[fingerprint.ExampleRecord] = []
    per_class_severity: dict[str, int] = {}
    violations: list[str] = []
    quarantine_hits = 0
    donor_usage: dict[str, int] = {}
    visual_payload: list[tuple] = []
    not_applicable: list[dict] = []
    stem_applicable = 0
    stem_rejected = 0
    stem_rejection_reasons: dict[str, int] = {}
    good_without_context = 0
    unauthorized_sources = 0
    source_union_mismatches = 0

    for partition, candidate_ids in selected.items():
        for candidate_id in candidate_ids:
            candidate_geometry = geometries[candidate_id]
            scene = scene_module.build_base_scene(candidate_id, partition,
                                                  candidate_geometry, provenance)
            stats = _candidate_stats(scene)
            for spec in gen.enumerate_specs(candidate_ids=[candidate_id],
                                             partition=partition,
                                             repeats=repeats, seed_base=seed_base):
                if spec.example_class == gen.STEM_MISSING:
                    evidence = gen.stem_evidence(scene)
                    if evidence['applicable']:
                        stem_applicable += 1
                    else:
                        stem_rejected += 1
                        reason = evidence.get('reason', 'unknown')
                        stem_rejection_reasons[reason] = \
                            stem_rejection_reasons.get(reason, 0) + 1
                try:
                    example = gen.generate_example(spec, candidate_geometry, provenance)
                except gen.NotApplicableError as exc:
                    not_applicable.append({'key': spec.key, 'class': spec.example_class,
                                           'severity': spec.severity,
                                           'reason': exc.reason,
                                           'detail': {k: v for k, v in exc.detail.items()
                                                      if k != 'lower_mask'}})
                    continue
                # Corruption RNG uses spec.seed (class-indexed by design); the sampler RNG
                # uses the class-independent sampling seed.
                sample_seed = spec.sampling_seed(points, seed_base)
                sample = sampler.sample(example.points, example.membership,
                                        example.changed_membership, scene.candidate_mask,
                                        sample_seed, points, spec.repeat)
                coverage = sampler.spatial_coverage(sample, example.points)
                height_cov = sampler.height_coverage(sample, example.points)
                record = _record(spec, example, sample, coverage, height_cov, stats)
                records.append(record)
                key = f'{spec.example_class}|{spec.severity}'
                per_class_severity[key] = per_class_severity.get(key, 0) + 1
                for donor in example.donors:
                    donor_usage[donor] = donor_usage.get(donor, 0) + 1
                    if donor in provenance.high_containment_quarantine:
                        quarantine_hits += 1
                row = dict(example.provenance)
                row['sample'] = {'points': sample.points,
                                 'source_points': sample.source_count,
                                 'sampling_seed': int(sample_seed),
                                 'corruption_seed': int(spec.seed),
                                 'repeat': int(spec.repeat),
                                 'points_target': int(
                                     sample.diagnostics['points_target']),
                                 'features': list(sampler.FEATURES),
                                 'diagnostics': sample.diagnostics}
                row['spatial_coverage'] = record.spatial_coverage
                row['height_coverage'] = record.height_coverage
                row['feature_mean'] = [round(v, 6) for v in record.feature_mean]
                row['feature_std'] = [round(v, 6) for v in record.feature_std]
                manifest_rows.append(row)
                if example.provenance['class'] == gen.GOOD and \
                        example.provenance['context_points'] == 0:
                    good_without_context += 1
                try:
                    provenance.assert_scene_sources(
                        spec.candidate_id, example.provenance['actual_point_source_ids'])
                except Exception:
                    unauthorized_sources += 1
                # Independent reconstruction: the reported source union must equal the
                # set of sources that actually contributed points, and the reported
                # counts must sum to the scene size.
                counts = example.provenance['actual_point_source_counts']
                reconstructed = tuple(sorted(counts))
                if (list(reconstructed) != list(
                        example.provenance['actual_point_source_ids'])
                        or sum(counts.values()) != example.provenance['point_count']):
                    source_union_mismatches += 1
                for aliases in example.provenance[
                        'actual_point_source_aliases'].values():
                    if set(aliases) & provenance.high_containment_aliases:
                        unauthorized_sources += 1
                if len(visual_payload) < 21 and spec.severity == gen.MEDIUM:
                    visual_payload.append((spec, example, sample))

    report_fingerprint = fingerprint.analyse(records)
    if report_fingerprint['critical_fingerprints']:
        violations.append('critical_generator_fingerprint')
    if quarantine_hits:
        violations.append('quarantine_usage')
    if good_without_context:
        violations.append('good_without_real_context')
    if unauthorized_sources:
        violations.append('unauthorized_point_source')
    if source_union_mismatches:
        violations.append('source_union_mismatch')

    summary = {
        'diagnostic_version': DIAGNOSTIC_VERSION,
        'generator_version': gen.GENERATOR_VERSION,
        'generator_digest': generator_digest(),
        'sampler_version': sampler.SAMPLER_VERSION,
        'scene_version': scene_module.SCENE_VERSION,
        'membership_channel': gen.MEMBERSHIP_CHANNEL,
        'protocol': provenance.protocol,
        'protocol_digest': provenance.protocol_digest,
        'identity_manifest_digest': provenance.identity_manifest_digest,
        'provenance_digest': provenance.provenance_digest,
        'training_lock': lock.acceptance_state(),
        'TRAINING_LOCK_ACTIVE': not lock.acceptance_state()['training_allowed'],
        'points_primary': sampler.POINTS_PRIMARY,
        'points_robustness': sampler.POINTS_ROBUSTNESS,
        'points_used': points,
        'metric_scale_constant': sampler.METRIC_SCALE_CONSTANT,
        'per_example_rms_scaling': False,
        'classes': list(gen.CLASSES),
        'severities': list(gen.SEVERITIES),
        'partitions': list(selected),
        'candidates': {p: ids for p, ids in selected.items()},
        'examples': len(records),
        'examples_per_class_severity': dict(sorted(per_class_severity.items())),
        'not_applicable_examples': not_applicable,
        'not_applicable_count': len(not_applicable),
        'stem_applicable': stem_applicable,
        'stem_rejected': stem_rejected,
        'stem_rejection_reasons': dict(sorted(stem_rejection_reasons.items())),
        'good_without_real_context': good_without_context,
        'donor_usage_examples': len(donor_usage),
        'quarantine_donor_uses': quarantine_hits,
        'held_out_examples': 0,
        'cross_partition_leakage': 0,
        'unauthorized_point_sources': unauthorized_sources,
        'source_union_mismatches': source_union_mismatches,
        'violations': violations,
        'fingerprint': report_fingerprint,
    }
    summary['manifest_digest'] = hashlib.sha256(
        ('\n'.join(json.dumps(row, sort_keys=True) for row in manifest_rows)
         + '\n').encode('utf-8')).hexdigest()

    REPORTS.mkdir(parents=True, exist_ok=True)
    if write_visuals:
        summary['visualisations'] = write_visualisations(visual_payload)
    # The committed artifact is fully deterministic: two runs on the same frozen inputs
    # produce the same file, so re-running the diagnostic never leaves the worktree
    # dirty. The digest covers the finished summary, so it is self-verifying. Wall-clock
    # timing is volatile run telemetry and goes to a separate gitignored file.
    summary['deterministic_digest'] = hashlib.sha256(json.dumps(
        summary, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'GATE_D_SYNTHETIC_DIAGNOSTIC.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    with (REPORTS / 'GATE_D_DIAGNOSTIC_MANIFEST.jsonl').open('w', encoding='utf-8') as out:
        for row in manifest_rows:
            out.write(json.dumps(row, sort_keys=True) + '\n')
    (REPORTS / 'GATE_D_FINGERPRINT_REMEDIATION.json').write_text(
        json.dumps({
            'fingerprint': report_fingerprint,
            'controls': report_fingerprint['controls'],
            'audit_shortcut_scan': report_fingerprint['audit_shortcut_scan'],
            'examples': len(records),
            'generator_digest': summary['generator_digest'],
            'membership_channel': gen.MEMBERSHIP_CHANNEL,
            'violations': violations,
        }, indent=2), encoding='utf-8')
    telemetry = {'seconds': round(time.time() - started, 3),
                 'deterministic_digest': summary['deterministic_digest'],
                 'manifest_digest': summary['manifest_digest'],
                 'examples': len(records), 'violations': violations}
    (REPORTS / 'GATE_D_DIAGNOSTIC_RUNTIME.json').write_text(
        json.dumps(telemetry, indent=2), encoding='utf-8')
    return summary


def write_visualisations(payload) -> list[str]:
    """Small representative PNGs: scene membership and the sampled model input."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = REPORTS / 'gate_d_visuals'
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for spec, example, sample in payload:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=110)
        member = example.membership
        axes[0].scatter(example.points[~member, 0], example.points[~member, 1],
                        s=0.4, c='#c0392b', label='membership 0')
        axes[0].scatter(example.points[member, 0], example.points[member, 1],
                        s=0.4, c='#2f7d32', label='membership 1')
        axes[0].set_title(f'{spec.example_class} / {spec.severity}\n'
                          f'{spec.candidate_id}', fontsize=8)
        axes[0].set_aspect('equal', adjustable='datalim')
        axes[0].legend(markerscale=8, fontsize=6, loc='upper right')
        flag = sample.features[:, 3] > 0.5
        axes[1].scatter(sample.features[~flag, 0], sample.features[~flag, 1],
                        s=3, c='#c0392b')
        axes[1].scatter(sample.features[flag, 0], sample.features[flag, 1],
                        s=3, c='#2f7d32')
        axes[1].set_title(f'sampled input: {sample.points} pts, '
                          f'membership {sample.features[:, 3].mean():.2f}', fontsize=8)
        axes[1].set_aspect('equal', adjustable='datalim')
        figure.tight_layout()
        name = f'{spec.example_class}_{spec.severity}_' \
               f'{spec.candidate_id.replace(":", "_")}.png'
        path = out_dir / name
        figure.savefig(path, bbox_inches='tight')
        plt.close(figure)
        written.append(str(path.relative_to(REPORTS)))
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-partition', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--points', type=int, default=sampler.POINTS_PRIMARY,
                        choices=[sampler.POINTS_PRIMARY, sampler.POINTS_ROBUSTNESS])
    parser.add_argument('--cache-root', default='D:/TreeQCTraining/gate_d_cache')
    parser.add_argument('--no-visuals', action='store_true')
    args = parser.parse_args(argv)
    summary = run(args.per_partition, args.repeats, args.points, args.cache_root,
                  not args.no_visuals)
    print(json.dumps({k: v for k, v in summary.items() if k != 'fingerprint'},
                     indent=2))
    print(json.dumps(summary['fingerprint']['CRITICAL_FINGERPRINTS']))
    return 1 if summary['violations'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
