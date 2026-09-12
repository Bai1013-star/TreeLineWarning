"""Public tree-instance discovery, manifest, and neighbourhood geometry.

Tree instances are the publisher's per-tree PLY files under
``trees/{train,val,test}/...`` in the pinned Zenodo record. This module never
invents an instance: a row exists only because a file exists, and the manifest
records the file, its partition folder, its byte size and its point count.

Nothing here reads the TreeLineWarning local 20-tree corpus or the local 8-bad
blind challenge; those paths are not reachable from this module at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import v2_config as cfg
import v2_ply as ply


@dataclass(frozen=True)
class TreeInstance:
    """One publisher-labelled individual tree."""

    tree_id: str
    site: str
    partition: str
    subfolder: str
    source_file: str
    path: Path
    point_count: int
    byte_size: int
    xyz_min: tuple[float, float, float]
    xyz_max: tuple[float, float, float]

    @property
    def key(self) -> str:
        return f'{self.site}|{self.partition}|{self.tree_id}'

    def as_row(self) -> dict:
        return {
            'tree_id': self.tree_id, 'site': self.site, 'partition': self.partition,
            'subfolder': self.subfolder, 'source_file': self.source_file,
            'point_count': self.point_count, 'byte_size': self.byte_size,
            'xyz_min': list(self.xyz_min), 'xyz_max': list(self.xyz_max),
        }


def discover_tree_instances(sites=None, partitions=None) -> list[TreeInstance]:
    """Enumerate publisher per-tree PLY files with a cheap header + bbox pass."""
    sites = tuple(sites) if sites else tuple(cfg.SITE_TREE_ROOT)
    rows: list[TreeInstance] = []
    for site in sites:
        root = cfg.SITE_TREE_ROOT[site]
        if not root.exists():
            continue
        for path in sorted(root.rglob('*.ply')):
            rel = path.relative_to(root)
            parts = rel.parts
            partition = parts[0] if len(parts) > 1 else 'unknown'
            subfolder = '/'.join(parts[1:-1])
            if partitions is not None and partition not in partitions:
                continue
            header = ply.read_ply_header(path)
            rows.append(TreeInstance(
                tree_id=path.stem, site=site, partition=partition,
                subfolder=subfolder, source_file=str(rel).replace('\\', '/'),
                path=path, point_count=int(header.count),
                byte_size=int(path.stat().st_size),
                xyz_min=(0.0, 0.0, 0.0), xyz_max=(0.0, 0.0, 0.0)))
    return rows


def with_bounds(rows: list[TreeInstance], progress=None) -> list[TreeInstance]:
    """Fill each row's bounding box by reading only the XYZ channels."""
    out = []
    for index, row in enumerate(rows):
        xyz = ply.read_xyz(row.path, dtype=np.float32)
        out.append(TreeInstance(
            tree_id=row.tree_id, site=row.site, partition=row.partition,
            subfolder=row.subfolder, source_file=row.source_file, path=row.path,
            point_count=row.point_count, byte_size=row.byte_size,
            xyz_min=tuple(float(v) for v in xyz.min(axis=0)),
            xyz_max=tuple(float(v) for v in xyz.max(axis=0))))
        if progress is not None and (index + 1) % 50 == 0:
            progress(index + 1, len(rows))
    return out


def manifest_digest(rows: list[TreeInstance]) -> str:
    payload = [row.as_row() for row in rows]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()


def sample_instances(rows: list[TreeInstance], per_site: int, seed: int,
                     eligible=None) -> list[TreeInstance]:
    """Deterministic per-site sample, so a small run is reproducible.

    Selection is by a seeded permutation of the site's sorted tree ids, before
    any geometry or label is inspected. Top-up rules are not applied: if a site
    has fewer instances than requested, the real count is what is reported.
    """
    chosen: list[TreeInstance] = []
    by_site: dict[str, list[TreeInstance]] = {}
    for row in rows:
        if eligible is not None and not eligible(row):
            continue
        by_site.setdefault(row.site, []).append(row)
    for site in sorted(by_site):
        group = sorted(by_site[site], key=lambda r: r.key)
        order = np.random.default_rng(
            np.random.SeedSequence([seed, int(hashlib.sha256(site.encode()).hexdigest()[:8], 16)])
        ).permutation(len(group))
        for index in order[:per_site]:
            chosen.append(group[int(index)])
    return sorted(chosen, key=lambda r: r.key)


# --- neighbourhood geometry ---------------------------------------------------

def build_ground_model(xyz: np.ndarray, grid_m: float = None) -> dict:
    """Grid-minimum ground proxy for a plot cloud.

    The public benchmark publishes no ground classification for these plots, and
    this audit must not import the product ground classifier. The proxy is the
    minimum Z per XY cell, which is the standard bounded fallback for TLS forest
    plots and is recorded as a proxy, never as a product ground surface.
    """
    grid_m = float(grid_m if grid_m is not None else cfg.GROUND_GRID_M)
    origin = np.floor(xyz[:, :2].min(axis=0) / grid_m) * grid_m
    cells = np.floor((xyz[:, :2] - origin) / grid_m).astype(np.int64)
    cols = int(cells[:, 0].max()) + 1
    keys = cells[:, 1] * cols + cells[:, 0]
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    sorted_z = xyz[order, 2]
    unique, starts = np.unique(sorted_keys, return_index=True)
    minima = np.minimum.reduceat(sorted_z, starts)
    return {'origin': origin.tolist(), 'grid_m': grid_m, 'cols': cols,
            'keys': unique, 'min_z': minima, 'kind': 'GRID_MINIMUM_PROXY'}


def ground_height(model: dict, xy: np.ndarray) -> np.ndarray:
    """Ground Z for arbitrary XY, from the nearest populated cell in cell space."""
    origin = np.asarray(model['origin'], dtype=np.float64)
    grid_m = float(model['grid_m'])
    cols = int(model['cols'])
    cells = np.floor((xy - origin) / grid_m).astype(np.int64)
    keys = cells[:, 1] * cols + cells[:, 0]
    index = np.searchsorted(model['keys'], keys)
    index = np.clip(index, 0, len(model['keys']) - 1)
    return model['min_z'][index]


def read_plot_xyz(site: str, partitions=('train', 'val', 'test'),
                  chunk: int = ply.CHUNK_POINTS, progress=None) -> np.ndarray:
    """Full plot XYZ for a site, concatenated from the published region clouds."""
    blocks = []
    for partition in partitions:
        path = cfg.SITE_PLOT_ROOT[site] / f'{site}_{partition}.ply'
        if not path.exists():
            continue
        for block in ply.iter_points(path, chunk=chunk):
            xyz = np.empty((block.size, 3), dtype=np.float32)
            xyz[:, 0] = block['x']
            xyz[:, 1] = block['y']
            xyz[:, 2] = block['z']
            blocks.append(xyz)
        if progress is not None:
            progress(partition, int(sum(len(b) for b in blocks)))
    if not blocks:
        raise ply.PlyError(f'no plot cloud for {site}')
    return np.concatenate(blocks, axis=0)


def roi_from_candidate(candidate_xyz: np.ndarray, context_xyz: np.ndarray) -> np.ndarray:
    """Boolean mask over ``context_xyz`` for the candidate's ROI.

    The ROI is the candidate bounding box grown in XY and extended downward from
    the candidate's own floor. A neighbour is admitted when it has points inside
    that box, which is what makes neighbour contamination and missing points
    both repairable: both populations are inside the ROI by construction.
    """
    lo = candidate_xyz.min(axis=0).astype(np.float64)
    hi = candidate_xyz.max(axis=0).astype(np.float64)
    height = max(float(hi[2] - lo[2]), 0.1)
    low_z = lo[2] - cfg.ROI_Z_FLOOR_FRACTION * height
    high_z = hi[2] + cfg.ROI_Z_CEILING_PAD_M
    pad = cfg.ROI_XY_PAD_M
    inside = (
        (context_xyz[:, 0] >= lo[0] - pad) & (context_xyz[:, 0] <= hi[0] + pad)
        & (context_xyz[:, 1] >= lo[1] - pad) & (context_xyz[:, 1] <= hi[1] + pad)
        & (context_xyz[:, 2] >= low_z) & (context_xyz[:, 2] <= high_z)
    )
    return np.asarray(inside, dtype=bool)
