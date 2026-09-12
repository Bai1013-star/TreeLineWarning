"""Reconnaissance: what public tree instances actually exist, and how big.

Writes ``reports/DATA_RECON.json``. No training, no proposals, no model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_config as cfg
import v2_data as data
import v2_ply as ply


def main() -> int:
    started = time.time()
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sites = {}
    all_rows = []
    for site in sorted(cfg.SITE_TREE_ROOT):
        root = cfg.SITE_TREE_ROOT[site]
        if not root.exists():
            sites[site] = {'present': False, 'root': str(root)}
            continue
        rows = data.discover_tree_instances(sites=(site,))
        by_partition: dict[str, dict] = {}
        for partition in sorted({row.partition for row in rows}):
            group = [row for row in rows if row.partition == partition]
            by_partition[partition] = {
                'files': len(group),
                'points': int(sum(r.point_count for r in group)),
                'subfolders': sorted({r.subfolder for r in group}),
                'bytes': int(sum(r.byte_size for r in group)),
            }
        sites[site] = {
            'present': True, 'root': str(root), 'files': len(rows),
            'points': int(sum(r.point_count for r in rows)),
            'bytes': int(sum(r.byte_size for r in rows)),
            'by_partition': by_partition,
            'manifest_digest': data.manifest_digest(rows),
        }
        all_rows.extend(rows)
        print(f'{site}: {len(rows)} tree files, '
              f'{sum(r.point_count for r in rows):,} points', flush=True)

    # Geometry sample: per-site point-count and extent distribution from real headers.
    sample = data.sample_instances(all_rows, 60, cfg.SEED)
    geometry = {}
    for site in sorted({r.site for r in sample}):
        group = [r for r in sample if r.site == site]
        counts, heights, areas = [], [], []
        for row in group[:40]:
            xyz = ply.read_xyz(row.path)
            span = xyz.max(axis=0) - xyz.min(axis=0)
            counts.append(int(xyz.shape[0]))
            heights.append(float(span[2]))
            areas.append(float(span[0] * span[1]))
        geometry[site] = {
            'sampled': len(counts),
            'points_median': float(np.median(counts)),
            'points_p10': float(np.percentile(counts, 10)),
            'points_p90': float(np.percentile(counts, 90)),
            'height_median_m': float(np.median(heights)),
            'height_p10_m': float(np.percentile(heights, 10)),
            'height_p90_m': float(np.percentile(heights, 90)),
            'bbox_area_median_m2': float(np.median(areas)),
        }

    # Plot clouds: are the region files usable at all?
    plots = {}
    for site in sorted(cfg.SITE_PLOT_ROOT):
        entry = {}
        for partition in ('train', 'val', 'test'):
            path = cfg.SITE_PLOT_ROOT[site] / f'{site}_{partition}.ply'
            if not path.exists():
                entry[partition] = None
                continue
            header = ply.read_ply_header(path)
            entry[partition] = {'bytes': int(path.stat().st_size),
                                'points': int(header.count),
                                'properties': [n for n, _ in header.properties]}
        plots[site] = entry

    payload = {
        'schema': cfg.SCHEMA,
        'config_digest': cfg.config_digest(),
        'sites': sites,
        'absent_sites': list(cfg.ABSENT_SITES),
        'geometry_sample': geometry,
        'plot_clouds': plots,
        'total_tree_files': len(all_rows),
        'seconds': round(time.time() - started, 2),
    }
    slow = np.mean([r.point_count for r in all_rows]) if all_rows else 0.0
    payload['mean_points_per_tree'] = float(slow)
    out = cfg.REPORT_DIR / 'DATA_RECON.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items()
                      if k in ('sites', 'geometry_sample', 'total_tree_files')}, indent=2))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
