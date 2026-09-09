"""Bounded geometry index and cache for Gate D examples.

Design constraints from the task:

* never rescan the 105 M-point source clouds per sample;
* never scan a whole publisher cloud when a bounded lookup suffices;
* the generator must not be able to reach geometry it was not authorised to use.

**Membership definition.** The publisher ships two related but non-identical views: the
per-tree files and the merged plot clouds. Exact-cell cross-matching between them is
mostly empty for this benchmark (median match fraction 0), so per-point membership
cannot be derived from the plot clouds. Gate D therefore uses the publisher's own
per-tree annotation: every point of the candidate tree carries membership 1 and every
point contributed by context or background geometry carries membership 0. That is the
official GT label for the tree instance, it needs no cross-file inference, and it makes
the fourth model channel meaningful for exactly the defects that move foreign geometry
into a candidate (MERGED, CONTAMINATION, NOISE, and the refilled CROWN/STEM regions).

For one candidate the cache stores exactly the point sets the generator may touch:

* ``candidate``  — the candidate tree's own points, membership 1;
* ``context``    — every surviving context tree's points, membership 0;
* ``background`` — real points from other **allowed** trees of the same site, used only
  by the NOISE class and to refill vacated CROWN/STEM volumes.

Background trees are allowlisted: same site, not high-containment quarantined, not
held-out, and not part of the candidate's own context. Every id is re-authorised by the
provenance gate before it is read.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import canonical_identity as ci
import numpy as np

DATA_ROOT = Path('D:/TreeQCData/extracted_manual_verified')
CACHE_VERSION = 'gate_d_geometry_cache_v4'
# Bounded per-candidate storage caps. Applied by a deterministic stride over a
# spatially sorted order, never by truncation, so coverage is preserved.
MAX_TREE_POINTS = 400_000
MAX_BACKGROUND_POINTS = 150_000
MAX_BACKGROUND_TREES = 8
CHUNK_POINTS = 2_000_000
# Horizontal search radii for background trees, tried in order. An isolated tree can
# have no neighbour inside the first radius, and a NOISE example needs real non-member
# geometry, so the radius escalates deterministically rather than failing.
BACKGROUND_RADII_M = (12.0, 20.0, 35.0, 50.0)

PLOT_BY_SITE = {
    'litchfield': ('litchfield_train.ply', 'litchfield_val.ply', 'litchfield_test.ply'),
    'wytham': ('wytham_train.ply', 'wytham_val.ply', 'wytham_test.ply'),
}


class GeometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class TreeGeometry:
    """One tree's points plus its membership label."""

    canonical_tree_id: str
    site: str
    points: np.ndarray           # (N, 3) float32 metres
    membership: np.ndarray       # (N,) bool: publisher annotation for this instance
    dominant_instance: int
    point_count: int
    stored_points: int


@dataclass(frozen=True)
class CandidateGeometry:
    """Everything the generator needs for one candidate example."""

    candidate: TreeGeometry
    context: tuple[TreeGeometry, ...]
    background: np.ndarray        # (M, 3) float32 real non-member geometry
    background_label: np.ndarray  # (M,) int32 id of the tree each point came from
    background_ids: tuple[str, ...] = ()

    @property
    def candidate_id(self) -> str:
        return self.candidate.canonical_tree_id

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(t.canonical_tree_id for t in self.context)


def _stride_cap(points: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic spatially-spread subsample.

    Sorting by height before striding keeps the vertical profile (stem through crown)
    intact instead of dropping the top of a tall crown.
    """
    return points[_stride_order(points, cap)]


def _stride_order(points: np.ndarray, cap: int) -> np.ndarray:
    """Indices of the deterministic strided selection (identity when under the cap)."""
    if points.shape[0] <= cap:
        return np.arange(points.shape[0])
    order = np.lexsort((points[:, 1], points[:, 0], points[:, 2]))
    stride = int(np.ceil(points.shape[0] / cap))
    return order[::stride][:cap]


def _cells(xyz: np.ndarray) -> np.ndarray:
    """Exact float32 cell keys, matching the Gate C2 identity rule."""
    return ci.keys_of(np.asarray(xyz, dtype=np.float32))


def _authorize_candidate(provenance, candidate_id: str) -> tuple[str, ...]:
    provenance.assert_not_quarantined(candidate_id, 'candidate')
    provenance.assert_not_boundary_candidate(candidate_id)
    context_ids = provenance.authorized_context(candidate_id)
    for neighbour in context_ids:
        provenance.authorize_donor(candidate_id, neighbour)
    return context_ids


def allowed_background_trees(provenance, candidate_id: str,
                             radii=BACKGROUND_RADII_M,
                             limit: int = MAX_BACKGROUND_TREES) -> tuple[str, ...]:
    """Same-site real trees that may supply non-member geometry to this candidate.

    Excluded: the candidate itself, its context, every high-containment quarantine
    tree, every held-out tree, and any tree whose bbox does not come within the radius.
    The radii are tried in order and the first tier with at least one tree wins, so an
    isolated candidate still gets real geometry instead of failing. The result is
    deterministic (ordered by centroid distance, then id).
    """
    site = provenance.site(candidate_id)
    context = set(provenance.authorized_context(candidate_id))
    centre = np.asarray(provenance.tree[candidate_id]['centroid'], dtype=float)
    scored: list[tuple[float, str]] = []
    for tree_id, row in provenance.tree.items():
        if tree_id == candidate_id or tree_id in context:
            continue
        if row['site'] != site:
            continue
        if tree_id in provenance.high_containment_quarantine:
            continue
        if row['partition'] == 'held_out':
            continue
        centroid = np.asarray(row['centroid'], dtype=float)
        scored.append((float(np.linalg.norm(centroid - centre)), tree_id))
    scored.sort()
    for radius in radii:
        tier = [(d, t) for d, t in scored if d <= radius]
        if tier:
            return tuple(t for _, t in tier[:limit])
    return ()


def _read_tree(tree_id: str, provenance, root: Path,
               max_tree_points: int) -> tuple[np.ndarray, int]:
    record = provenance.tree[tree_id]
    xyz = np.asarray(ci.xyz_native(ci.read_points(root / record['site'] / record['source_file'])),
                     dtype=np.float32)
    return _stride_cap(xyz, max_tree_points), int(xyz.shape[0])


def _tree_geometry(tree_id: str, provenance, root: Path, membership: bool,
                   dominant: int, max_tree_points: int) -> TreeGeometry:
    stored, total = _read_tree(tree_id, provenance, root, max_tree_points)
    return TreeGeometry(canonical_tree_id=tree_id, site=provenance.tree[tree_id]['site'],
                        points=stored,
                        membership=np.full(stored.shape[0], membership, dtype=bool),
                        dominant_instance=dominant, point_count=total,
                        stored_points=int(stored.shape[0]))


def build_candidate_geometry_batch(candidate_ids, provenance, data_root: Path | None = None,
                                   max_tree_points: int = MAX_TREE_POINTS,
                                   max_background_points: int = MAX_BACKGROUND_POINTS,
                                   background_radii=BACKGROUND_RADII_M
                                   ) -> dict[str, CandidateGeometry]:
    """Authorise and load many candidates.

    Every candidate, context tree and background tree is authorised *before* any
    geometry is read, so a single illegal id aborts the whole batch instead of
    producing a partially-legal dataset.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    ids = list(candidate_ids)
    authorised = {cid: _authorize_candidate(provenance, cid) for cid in ids}
    background_ids = {
        cid: allowed_background_trees(provenance, cid, background_radii)
        for cid in ids}
    for chosen in background_ids.values():
        for tree_id in chosen:
            provenance.assert_not_quarantined(tree_id, 'background')
            if provenance.partition_of[tree_id] == 'held_out':
                from gate_d_provenance import HeldOutAccessError
                raise HeldOutAccessError(f'held-out background tree {tree_id}')
            provenance.assert_alias_clean(tree_id)

    results: dict[str, CandidateGeometry] = {}
    for candidate_id in ids:
        dominant = provenance.tree[candidate_id].get('dominant_plot_instance')
        candidate = _tree_geometry(candidate_id, provenance, root, True,
                                   int(dominant) if dominant is not None else -1,
                                   max_tree_points)
        context = tuple(
            _tree_geometry(n, provenance, root, False,
                           int(provenance.tree[n].get('dominant_plot_instance') or -1),
                           max_tree_points)
            for n in authorised[candidate_id])
        chosen = background_ids[candidate_id]
        if chosen:
            pieces, labels = [], []
            for index, tree_id in enumerate(chosen):
                points, _ = _read_tree(tree_id, provenance, root, max_tree_points)
                if points.shape[0] == 0:
                    continue
                pieces.append(points)
                labels.append(np.full(points.shape[0], index, dtype=np.int32))
            if pieces:
                xyz = np.concatenate(pieces).astype(np.float32)
                label = np.concatenate(labels)
                order = _stride_order(xyz, max_background_points)
                xyz, label = xyz[order], label[order]
            else:
                xyz = np.zeros((0, 3), dtype=np.float32)
                label = np.zeros(0, dtype=np.int32)
        else:
            xyz = np.zeros((0, 3), dtype=np.float32)
            label = np.zeros(0, dtype=np.int32)
        results[candidate_id] = CandidateGeometry(
            candidate=candidate, context=context, background=xyz,
            background_label=label, background_ids=chosen)
    return {cid: results[cid] for cid in ids}


def build_candidate_geometry(candidate_id: str, provenance, data_root: Path | None = None,
                             max_tree_points: int = MAX_TREE_POINTS,
                             max_background_points: int = MAX_BACKGROUND_POINTS
                             ) -> CandidateGeometry:
    """Authorise, then load, one candidate and its surviving context."""
    return build_candidate_geometry_batch([candidate_id], provenance, data_root,
                                          max_tree_points, max_background_points
                                          )[candidate_id]


# --- on-disk cache ---------------------------------------------------------

def _cache_key(provenance, candidate_id: str, max_tree_points: int,
               max_background_points: int) -> str:
    payload = json.dumps({
        'version': CACHE_VERSION,
        'identity_manifest_digest': provenance.identity_manifest_digest,
        'provenance_digest': provenance.provenance_digest,
        'candidate': candidate_id,
        'context': list(provenance.authorized_context(candidate_id)),
        'background': list(allowed_background_trees(provenance, candidate_id)),
        'max_tree_points': max_tree_points,
        'max_background_points': max_background_points,
    }, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]


def cache_path(cache_root: Path | str, provenance, candidate_id: str,
               max_tree_points: int = MAX_TREE_POINTS,
               max_background_points: int = MAX_BACKGROUND_POINTS) -> Path:
    return Path(cache_root) / (_cache_key(provenance, candidate_id, max_tree_points,
                                          max_background_points) + '.npz')


def write_cache(path: Path, geometry: CandidateGeometry) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    members = [geometry.candidate, *geometry.context]
    payload = {
        'candidate_id': np.array(geometry.candidate_id),
        'context_ids': np.array([m.canonical_tree_id for m in geometry.context]),
        'background_ids': np.array(list(geometry.background_ids)),
        'meta': np.array(json.dumps({
            'version': CACHE_VERSION,
            'members': [{'id': m.canonical_tree_id, 'site': m.site,
                         'dominant_instance': int(m.dominant_instance),
                         'point_count': int(m.point_count),
                         'stored_points': int(m.stored_points)} for m in members],
            'background_points': int(geometry.background.shape[0]),
        })),
        'background': np.asarray(geometry.background, dtype=np.float32),
        'background_label': np.asarray(geometry.background_label, dtype=np.int32),
    }
    for index, member in enumerate(members):
        payload[f'p{index}'] = np.asarray(member.points, dtype=np.float32)
        payload[f'm{index}'] = np.asarray(member.membership, dtype=bool)
    tmp = path.with_suffix('.npz.tmp')
    with open(tmp, 'wb') as handle:
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
        background_ids = tuple(str(v) for v in data['background_ids'])
        members = []
        for index, info in enumerate(meta['members']):
            members.append(TreeGeometry(
                canonical_tree_id=info['id'], site=info['site'],
                points=data[f'p{index}'], membership=data[f'm{index}'],
                dominant_instance=int(info['dominant_instance']),
                point_count=int(info['point_count']),
                stored_points=int(info['stored_points'])))
        if len(members) != len(context_ids) + 1:
            raise GeometryError('cache member count mismatch')
        return CandidateGeometry(candidate=members[0], context=tuple(members[1:]),
                                 background=data['background'],
                                 background_label=data['background_label'],
                                 background_ids=background_ids)


def load_cached_geometry(cache_root: Path | str, provenance, candidate_id: str,
                         data_root: Path | None = None,
                         max_tree_points: int = MAX_TREE_POINTS,
                         max_background_points: int = MAX_BACKGROUND_POINTS
                         ) -> CandidateGeometry:
    """Return cached geometry, building and persisting it on first use."""
    path = cache_path(cache_root, provenance, candidate_id, max_tree_points,
                      max_background_points)
    if path.exists():
        try:
            return _read_cache(path, candidate_id)
        except (OSError, KeyError, ValueError, GeometryError):
            path.unlink(missing_ok=True)
    geometry = build_candidate_geometry(candidate_id, provenance, data_root,
                                        max_tree_points, max_background_points)
    write_cache(path, geometry)
    return geometry


def load_cached_geometry_batch(cache_root: Path | str, provenance, candidate_ids,
                               data_root: Path | None = None,
                               max_tree_points: int = MAX_TREE_POINTS,
                               max_background_points: int = MAX_BACKGROUND_POINTS
                               ) -> dict[str, CandidateGeometry]:
    """Cache-aware batch load. Misses are built together in one pass."""
    root = Path(cache_root)
    results: dict[str, CandidateGeometry] = {}
    missing: list[str] = []
    for candidate_id in candidate_ids:
        path = cache_path(root, provenance, candidate_id, max_tree_points,
                          max_background_points)
        if path.exists():
            try:
                results[candidate_id] = _read_cache(path, candidate_id)
                continue
            except (OSError, KeyError, ValueError, GeometryError):
                path.unlink(missing_ok=True)
        missing.append(candidate_id)
    if missing:
        built = build_candidate_geometry_batch(missing, provenance, data_root,
                                               max_tree_points, max_background_points)
        for candidate_id, geometry in built.items():
            write_cache(cache_path(root, provenance, candidate_id, max_tree_points,
                                   max_background_points), geometry)
            results[candidate_id] = geometry
    return {candidate_id: results[candidate_id] for candidate_id in candidate_ids}
