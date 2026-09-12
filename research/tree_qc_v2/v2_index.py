"""Cached bounding-box index over the public tree instances.

Building a local ROI needs to know which trees are near a window. Reading every
cloud to find out costs minutes per patch; reading 64 bytes of header plus a
bounds reduction per file is enough, and the result is cached with a digest so a
later run can prove it is using the same index.

The index stores only publisher-derived facts: file path, point count, byte size
and the XYZ bounding box of that file.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import v2_config as cfg
import v2_data as data
import v2_ply as ply

INDEX_VERSION = 'tree_qc_v2_bbox_index_v1'


def index_path(site: str) -> Path:
    return cfg.CACHE_DIR / f'bbox_index_{site}.json'


def _row_bounds(row, box=None):
    if box is None:
        xyz = ply.read_xyz(row.path, dtype=np.float32)
        box = np.array([xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 2].min(),
                        xyz[:, 0].max(), xyz[:, 1].max(), xyz[:, 2].max()],
                       dtype=np.float64)
    return box


def build_site_index(site: str, workers: int = 12, verbose: bool = True) -> dict:
    """Header + bounding box for every tree file of one site, cached to JSON."""
    rows = data.discover_tree_instances(sites=(site,))
    started = time.time()
    bounds = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, box in zip(range(len(rows)), pool.map(_row_bounds, rows), strict=True):
            bounds[index] = box
    entries = []
    for row, box in zip(rows, bounds, strict=True):
        entries.append({
            'key': row.key, 'tree_id': row.tree_id, 'site': site,
            'partition': row.partition, 'source_file': row.source_file,
            'path': str(row.path), 'point_count': int(row.point_count),
            'byte_size': int(row.byte_size),
            'bbox': [float(v) for v in box],
        })
    digest = hashlib.sha256(json.dumps(
        [[e['key'], e['point_count'], [round(v, 6) for v in e['bbox']]] for e in entries],
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    payload = {'version': INDEX_VERSION, 'site': site, 'entries': entries,
               'count': len(entries), 'digest': digest,
               'points': int(sum(e['point_count'] for e in entries)),
               'seconds': round(time.time() - started, 2)}
    target = index_path(site)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding='utf-8')
    if verbose:
        print(f'  index {site}: {len(entries)} trees, '
              f'{payload["points"]:,} points, {payload["seconds"]}s', flush=True)
    return payload


def load_site_index(site: str, rebuild: bool = False) -> dict:
    target = index_path(site)
    if target.exists() and not rebuild:
        payload = json.loads(target.read_text(encoding='utf-8'))
        if payload.get('version') == INDEX_VERSION:
            return payload
    return build_site_index(site)


def window_members(index: dict, bounds, cell_m: float = 20.0) -> list[dict]:
    """Index entries whose bounding box intersects an XY window."""
    grid: dict[tuple[int, int], list[dict]] = {}
    for entry in index['entries']:
        x0, y0, _, x1, y1, _ = entry['bbox']
        for cx in range(int(np.floor(x0 / cell_m)), int(np.floor(x1 / cell_m)) + 1):
            for cy in range(int(np.floor(y0 / cell_m)), int(np.floor(y1 / cell_m)) + 1):
                grid.setdefault((cx, cy), []).append(entry)
    found: dict[str, dict] = {}
    for cx in range(int(np.floor(bounds[0] / cell_m)),
                    int(np.floor(bounds[2] / cell_m)) + 1):
        for cy in range(int(np.floor(bounds[1] / cell_m)),
                        int(np.floor(bounds[3] / cell_m)) + 1):
            for entry in grid.get((cx, cy), ()):
                x0, y0, _, x1, y1, _ = entry['bbox']
                if x0 <= bounds[2] and x1 >= bounds[0] and y0 <= bounds[3] and y1 >= bounds[1]:
                    found[entry['key']] = entry
    return sorted(found.values(), key=lambda e: e['key'])


def densest_window(index: dict, window_m: float, cell_m: float = 20.0) -> tuple:
    """XY window over the densest 20 m cell of a site, as a real piece of forest."""
    from collections import Counter
    density: Counter = Counter()
    for entry in index['entries']:
        x0, y0, _, x1, y1, _ = entry['bbox']
        density[(int(np.floor((x0 + x1) / 2 / cell_m)),
                 int(np.floor((y0 + y1) / 2 / cell_m)))] += 1
    (cx, cy), _ = max(density.items(), key=lambda item: (item[1], item[0]))
    centre_x, centre_y = (cx + 0.5) * cell_m, (cy + 0.5) * cell_m
    return (centre_x - window_m / 2, centre_y - window_m / 2,
            centre_x + window_m / 2, centre_y + window_m / 2)


def available_sites() -> list[str]:
    return [site for site, root in sorted(cfg.SITE_TREE_ROOT.items()) if root.exists()]
