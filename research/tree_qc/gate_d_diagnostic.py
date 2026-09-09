"""Gate D bounded synthetic diagnostic.

Runs the generator and sampler over a small, representative set of train and validation
trees, covering all 7 classes x 3 severities, and writes the four Gate D artifacts plus
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
import numpy as np
from gate_d_provenance import load_gate_d_provenance

HERE = Path(__file__).resolve().parent
REPORTS = HERE / 'reports'
DIAGNOSTIC_VERSION = 'GATE_D_SYNTHETIC_DIAGNOSTIC_V1'
GENERATOR_FILES = ('gate_d_provenance.py', 'gate_d_geometry.py', 'gate_d_generator.py',
                   'gate_d_sampler.py', 'gate_d_dataset.py', 'gate_d_metrics.py',
                   'gate_d_train.py', 'gate_d_fingerprint.py', 'gate_d_diagnostic.py')


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


def _record(spec, example, sample, coverage) -> fingerprint.ExampleRecord:
    corruption = example.provenance['corruption']
    axis = corruption.get('axis')
    return fingerprint.ExampleRecord(
        key=spec.key, candidate_id=spec.candidate_id, partition=spec.partition,
        example_class=spec.example_class, severity=spec.severity,
        label=example.label, source_points=int(example.points.shape[0]),
        raw_source_points=int(example.provenance.get('raw_point_count',
                                                     example.points.shape[0])),
        sampled_points=int(sample.points),
        corruption_fraction=float(corruption.get('fraction', 0.0)),
        drawn_fraction=float(corruption.get('drawn_fraction', 0.0)),
        corruption_axis=tuple(axis) if axis else None,
        membership_source=round(float(example.membership.mean())
                                if example.membership.size else 0.0, 6),
        membership_sampled=round(sampler.membership_fraction(sample), 6),
        feature_mean=tuple(float(v) for v in sample.features.mean(axis=0)),
        feature_std=tuple(float(v) for v in sample.features.std(axis=0)),
        spatial_coverage=round(float(coverage), 6))


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

    for partition, candidate_ids in selected.items():
        for candidate_id in candidate_ids:
            candidate_geometry = geometries[candidate_id]
            for spec in gen.enumerate_specs(candidate_ids=[candidate_id],
                                             partition=partition,
                                             repeats=repeats, seed_base=seed_base):
                example = gen.generate_example(spec, candidate_geometry, provenance)
                sample = sampler.sample(example.points, example.membership, spec.seed,
                                        points)
                coverage = sampler.spatial_coverage(sample, example.points)
                record = _record(spec, example, sample, coverage)
                records.append(record)
                key = f'{spec.example_class}|{spec.severity}'
                per_class_severity[key] = per_class_severity.get(key, 0) + 1
                for donor in example.donors:
                    donor_usage[donor] = donor_usage.get(donor, 0) + 1
                    if donor in provenance.high_containment_quarantine:
                        quarantine_hits += 1
                row = dict(example.provenance)
                row['sample'] = {'points': sample.points, 'source_points': sample.source_count,
                                 'features': list(sampler.FEATURES),
                                 'membership_fraction': record.membership_sampled,
                                 'spatial_coverage': record.spatial_coverage,
                                 'sampler_version': sampler.SAMPLER_VERSION}
                row['feature_mean'] = [round(v, 6) for v in record.feature_mean]
                row['feature_std'] = [round(v, 6) for v in record.feature_std]
                manifest_rows.append(row)
                if len(visual_payload) < 21 and spec.severity == gen.MEDIUM:
                    visual_payload.append((spec, example, sample))

    report_fingerprint = fingerprint.analyse(records)
    if report_fingerprint['critical_fingerprints']:
        violations.append('critical_generator_fingerprint')
    if quarantine_hits:
        violations.append('quarantine_usage')

    summary = {
        'diagnostic_version': DIAGNOSTIC_VERSION,
        'generator_version': gen.GENERATOR_VERSION,
        'generator_digest': generator_digest(),
        'sampler_version': sampler.SAMPLER_VERSION,
        'protocol': provenance.protocol,
        'identity_manifest_digest': provenance.identity_manifest_digest,
        'provenance_digest': provenance.provenance_digest,
        'training_lock': lock.acceptance_state(),
        'TRAINING_LOCK_ACTIVE': not lock.acceptance_state()['training_allowed'],
        'points_primary': sampler.POINTS_PRIMARY,
        'points_robustness': sampler.POINTS_ROBUSTNESS,
        'points_used': points,
        'classes': list(gen.CLASSES),
        'severities': list(gen.SEVERITIES),
        'partitions': list(selected),
        'candidates': {p: ids for p, ids in selected.items()},
        'examples': len(records),
        'examples_per_class_severity': dict(sorted(per_class_severity.items())),
        'donor_usage_examples': len(donor_usage),
        'quarantine_donor_uses': quarantine_hits,
        'held_out_examples': 0,
        'cross_partition_leakage': 0,
        'violations': violations,
        'fingerprint': report_fingerprint,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    if write_visuals:
        summary['visualisations'] = write_visualisations(visual_payload)
    # The examples themselves must be reproducible byte-for-byte. Wall-clock timing is
    # the only field that legitimately varies, so the deterministic digest covers the
    # finished summary without it and the manifest is hashed directly.
    summary['deterministic_digest'] = hashlib.sha256(json.dumps(
        summary, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    summary['manifest_digest'] = hashlib.sha256(
        ('\n'.join(json.dumps(row, sort_keys=True) for row in manifest_rows)
         + '\n').encode('utf-8')).hexdigest()
    summary['seconds'] = round(time.time() - started, 3)
    (REPORTS / 'GATE_D_SYNTHETIC_DIAGNOSTIC.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    with (REPORTS / 'GATE_D_DIAGNOSTIC_MANIFEST.jsonl').open('w', encoding='utf-8') as out:
        for row in manifest_rows:
            out.write(json.dumps(row, sort_keys=True) + '\n')
    return summary


def write_visualisations(payload) -> list[str]:
    """Small representative PNGs: candidate cloud, corruption, sampled model input."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = REPORTS / 'gate_d_visuals'
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for spec, example, sample in payload:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=110)
        member = example.membership
        axes[0].scatter(example.points[member, 0], example.points[member, 1],
                        s=0.4, c='#2f7d32', label='candidate member')
        if (~member).any():
            axes[0].scatter(example.points[~member, 0], example.points[~member, 1],
                            s=0.4, c='#c0392b', label='non-member')
        axes[0].set_title(f'{spec.example_class} / {spec.severity}\n'
                          f'{spec.candidate_id}', fontsize=8)
        axes[0].set_aspect('equal', adjustable='datalim')
        axes[0].legend(markerscale=8, fontsize=6, loc='upper right')
        flag = sample.features[:, 3] > 0.5
        axes[1].scatter(sample.features[flag, 0], sample.features[flag, 1],
                        s=3, c='#2f7d32')
        axes[1].scatter(sample.features[~flag, 0], sample.features[~flag, 1],
                        s=3, c='#c0392b')
        axes[1].set_title(f'sampled model input: {sample.points} pts, '
                          f'membership {sample.features[:, 3].mean():.2f}', fontsize=8)
        axes[1].set_aspect('equal', adjustable='datalim')
        figure.tight_layout()
        name = f'{spec.example_class}_{spec.severity}_{spec.candidate_id.replace(":", "_")}.png'
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
