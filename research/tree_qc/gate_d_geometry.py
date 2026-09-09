"""Bounded geometry cache for Gate D scenes.

Design constraints from the task:

* never rescan the 105 M-point source clouds per sample;
* never scan a whole publisher cloud when a bounded lookup suffices;
* **every** point source must pass the Gate C2 source authorization.

The cache stores, for one candidate, exactly two kinds of real geometry:

* ``candidate`` — the candidate tree's own points;
* ``context``   — the candidate's frozen surviving context trees.

There is no separate "background" source. An earlier revision invented one by searching
for nearby trees outside the frozen surviving context, which produced 72 unauthorized
candidate→background references. That path is deleted: external geometry may only come
from the candidate's own surviving context, and the scene layer clips it to the frozen
3 m query.

Publisher per-tree files are the membership authority. The merged plot clouds are not
used for membership: exact-cell cross-matching between per-tree files and plot clouds is
mostly empty for this benchmark (median match fraction 0), so plot-derived per-point
membership would be unsound.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import canonical_identity as ci
import numpy as np

DATA_ROOT = Path('D:/TreeQCData/extracted_manual_verified')
CACHE_VERSION = 'gate_d_geometry_cache_v5'
# Bounded per-tree storage cap, applied by a deterministic stride over a spatially
# sorted order so the vertical profile (stem through crown) stays intact.
MAX_TREE_POINTS = 400_000
CHUNK_POINTS = 2_000_000

PLOT_BY_SITE = {
    'litchfield': ('litchfield_train.ply', 'litchfield_val.ply', 'litchfield_test.ply'),
    'wytham': ('wytham_train.ply', 'wytham_val.ply', 'wytham_test.ply'),
}


class GeometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class TreeGeometry:
    """One tree's points. Membership is a scene concept, not a tree concept."""

    canonical_tree_id: str
    site: str
    points: np.ndarray
    point_count: int
    stored_points: int


@dataclass(frozen=True)
class CandidateGeometry:
    """The authorised geometry for one candidate: itself plus its surviving context."""

    candidate: TreeGeometry
    context: tuple[TreeGeometry, ...]

    @property
    def candidate_id(self) -> str:
        return self.candidate.canonical_tree_id

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(t.canonical_tree_id for t in self.context)


def _stride_cap(points: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic spatially-spread subsample, sorted by height then strided."""
    return points[_stride_order(points, cap)]


def _stride_order(points: np.ndarray, cap: int) -> np.ndarray:
    if points.shape[0] <= cap:
        return np.arange(points.shape[0])
    order = np.lexsort((points[:, 1], points[:, 0], points[:, 2]))
    stride = int(np.ceil(points.shape[0] / cap))
    return order[::stride][:cap]


def _authorize_candidate(provenance, candidate_id: str) -> tuple[str, ...]:
    provenance.assert_not_quarantined(candidate_id, 'candidate')
    provenance.assert_not_boundary_candidate(candidate_id)
    context_ids = provenance.authorized_context(candidate_id)
    for neighbour in context_ids:
        provenance.authorize_donor(candidate_id, neighbour)
        provenance.assert_alias_clean(neighbour)
    provenance.assert_alias_clean(candidate_id)
    return context_ids


def _read_tree(tree_id: str, provenance, root: Path,
               max_tree_points: int) -> TreeGeometry:
    record = provenance.tree[tree_id]
    xyz = np.asarray(
        ci.xyz_native(ci.read_points(root / record['site'] / record['source_file'])),
        dtype=np.float32)
    stored = _stride_cap(xyz, max_tree_points)
    return TreeGeometry(canonical_tree_id=tree_id, site=record['site'], points=stored,
                        point_count=int(xyz.shape[0]), stored_points=int(stored.shape[0]))


def build_candidate_geometry_batch(candidate_ids, provenance, data_root: Path | None = None,
                                   max_tree_points: int = MAX_TREE_POINTS
                                   ) -> dict[str, CandidateGeometry]:
    """Authorise and load many candidates.

    Every candidate and every context tree is authorised *before* any geometry is read,
    so a single illegal id aborts the whole batch instead of producing a partially-legal
    dataset.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    ids = list(candidate_ids)
    authorised = {cid: _authorize_candidate(provenance, cid) for cid in ids}
    results: dict[str, CandidateGeometry] = {}
    for candidate_id in ids:
        candidate = _read_tree(candidate_id, provenance, root, max_tree_points)
        context = tuple(_read_tree(n, provenance, root, max_tree_points)
                        for n in authorised[candidate_id])
        results[candidate_id] = CandidateGeometry(candidate=candidate, context=context)
    return {cid: results[cid] for cid in ids}


def build_candidate_geometry(candidate_id: str, provenance, data_root: Path | None = None,
                             max_tree_points: int = MAX_TREE_POINTS) -> CandidateGeometry:
    """Authorise, then load, one candidate and its surviving context."""
    return build_candidate_geometry_batch([candidate_id], provenance, data_root,
                                          max_tree_points)[candidate_id]


# --- on-disk cache ---------------------------------------------------------

def _cache_key(provenance, candidate_id: str, max_tree_points: int) -> str:
    payload = json.dumps({
        'version': CACHE_VERSION,
        'identity_manifest_digest': provenance.identity_manifest_digest,
        'provenance_digest': provenance.provenance_digest,
        'candidate': candidate_id,
        'context': list(provenance.authorized_context(candidate_id)),
        'max_tree_points': max_tree_points,
    }, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]


def cache_path(cache_root: Path | str, provenance, candidate_id: str,
               max_tree_points: int = MAX_TREE_POINTS) -> Path:
    return Path(cache_root) / (_cache_key(provenance, candidate_id, max_tree_points)
                               + '.npz')


def write_cache(path: Path, geometry: CandidateGeometry) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    members = [geometry.candidate, *geometry.context]
    payload = {
        'candidate_id': np.array(geometry.candidate_id),
        'context_ids': np.array([m.canonical_tree_id for m in geometry.context]),
        'meta': np.array(json.dumps({
            'version': CACHE_VERSION,
            'members': [{'id': m.canonical_tree_id, 'site': m.site,
                         'point_count': int(m.point_count),
                         'stored_points': int(m.stored_points)} for m in members],
        })),
    }
    for index, member in enumerate(members):
        payload[f'p{index}'] = np.asarray(member.points, dtype=np.float32)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)
    return path


def _read_cache(path: Path, candidate_id: str) -> CandidateGeometry:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        if meta.get('version') != CACHE_VERSION:
            raise GeometryError('cache version mismatch')
        stored_candidate = str(data['candidate_id'])
        if stored_candidate != candidate_id:
            raise GeometryError(f'cache holds {stored_candidate}, requested {candidate_id}')
        context_ids = [str(v) for v in data['context_ids']]
        members = [TreeGeometry(canonical_tree_id=info['id'], site=info['site'],
                                points=data[f'p{index}'],
                                point_count=int(info['point_count']),
                                stored_points=int(info['stored_points']))
                   for index, info in enumerate(meta['members'])]
        if len(members) != len(context_ids) + 1:
            raise GeometryError('cache member count mismatch')
        return CandidateGeometry(candidate=members[0], context=tuple(members[1:]))


def load_cached_geometry(cache_root: Path | str, provenance, candidate_id: str,
                         data_root: Path | None = None,
                         max_tree_points: int = MAX_TREE_POINTS) -> CandidateGeometry:
    """Return cached geometry, building and persisting it on first use."""
    path = cache_path(cache_root, provenance, candidate_id, max_tree_points)
    if path.exists():
        try:
            return _read_cache(path, candidate_id)
        except (OSError, KeyError, ValueError, GeometryError):
            path.unlink(missing_ok=True)
    geometry = build_candidate_geometry(candidate_id, provenance, data_root,
                                        max_tree_points)
    write_cache(path, geometry)
    return geometry


def load_cached_geometry_batch(cache_root: Path | str, provenance, candidate_ids,
                               data_root: Path | None = None,
                               max_tree_points: int = MAX_TREE_POINTS
                               ) -> dict[str, CandidateGeometry]:
    """Cache-aware batch load. Misses are built together."""
    root = Path(cache_root)
    results: dict[str, CandidateGeometry] = {}
    missing: list[str] = []
    for candidate_id in candidate_ids:
        path = cache_path(root, provenance, candidate_id, max_tree_points)
        if path.exists():
            try:
                results[candidate_id] = _read_cache(path, candidate_id)
                continue
            except (OSError, KeyError, ValueError, GeometryError):
                path.unlink(missing_ok=True)
        missing.append(candidate_id)
    if missing:
        built = build_candidate_geometry_batch(missing, provenance, data_root,
                                               max_tree_points)
        for candidate_id, geometry in built.items():
            write_cache(cache_path(root, provenance, candidate_id, max_tree_points),
                        geometry)
            results[candidate_id] = geometry
    return {candidate_id: results[candidate_id] for candidate_id in candidate_ids}
