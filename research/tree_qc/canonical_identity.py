"""Canonical tree identity for the pinned two-site TLS benchmark.

Research layer only. This module reads the publisher PLY files under the verified
extraction root, derives order-independent coordinate fingerprints, groups file
occurrences into canonical tree identities with explicit alias evidence, and emits
a derived manifest. It never modifies publisher data, never unions biological
identities on weak evidence, and never trains a model.

Identity is deliberately conservative:

* LEVEL_A_EXACT_ALIAS requires multiset equality of the full native coordinate set.
* LEVEL_B_STRONG_ALIAS_CANDIDATE requires high mutual membership plus agreeing
  bbox/centroid; it is reported for review and is never auto-unioned.
* LEVEL_C_PARTIAL_OVERLAP_ONLY covers shared XYZ cells or bbox intersection only;
  these sources stay independent and unresolved.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# PLY scalar type -> numpy type. `int` in PLY is 32-bit; numpy's default `int`
# is platform sized (64-bit on Windows), so an explicit map is mandatory.
PLY_TYPES = {
    'char': 'i1', 'int8': 'i1',
    'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2',
    'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4',
    'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
}

# Exact-equality key for a float32 XYZ cell, compared on raw bit patterns.
KEY_DTYPE = np.dtype([('x', '<u4'), ('y', '<u4'), ('z', '<u4')])

IDENTITY_EXACT = 'EXACT'
IDENTITY_STRONG_VERIFIED = 'STRONG_VERIFIED'
IDENTITY_UNRESOLVED = 'UNRESOLVED'

LEVEL_A = 'LEVEL_A_EXACT_ALIAS'
LEVEL_B = 'LEVEL_B_STRONG_ALIAS_CANDIDATE'
LEVEL_C = 'LEVEL_C_PARTIAL_OVERLAP_ONLY'

# Predeclared thresholds. Fixed before any pool statistics are computed.
STRONG_CONTAINMENT = 0.95
STRONG_CENTROID_M = 0.25
CHUNK_POINTS = 2_000_000


@dataclass(frozen=True)
class PlyHeader:
    offset: int
    count: int
    dtype: np.dtype
    properties: tuple[tuple[str, str], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return self.dtype.names or ()


def read_ply_header(path: Path) -> PlyHeader:
    with Path(path).open('rb') as handle:
        raw = handle.read(65536)
    index = raw.find(b'end_header')
    if index < 0:
        raise ValueError(f'PLY header without end_header: {path}')
    newline = raw.find(b'\n', index)
    if newline < 0:
        raise ValueError(f'PLY header without terminator: {path}')
    offset = newline + 1
    lines = raw[:offset].decode('ascii').splitlines()
    if not lines or lines[0] != 'ply':
        raise ValueError(f'not a PLY file: {path}')
    fmt = None
    count = None
    properties: list[tuple[str, str]] = []
    for line in lines[1:]:
        if line.startswith('format '):
            fmt = line.split()[1]
        elif line.startswith('element vertex '):
            count = int(line.split()[2])
        elif line.startswith('property '):
            parts = line.split()
            if parts[1] == 'list':
                raise ValueError(f'list property unsupported: {path}')
            if parts[1] not in PLY_TYPES:
                raise ValueError(f'unknown PLY type {parts[1]!r}: {path}')
            properties.append((parts[2], PLY_TYPES[parts[1]]))
    if fmt != 'binary_little_endian':
        raise ValueError(f'unsupported PLY format {fmt!r}: {path}')
    if count is None or not properties:
        raise ValueError(f'incomplete PLY header: {path}')
    return PlyHeader(offset=offset, count=count, dtype=np.dtype(properties),
                     properties=tuple(properties))


def iter_points(path: Path, chunk: int = CHUNK_POINTS):
    """Yield structured-array chunks of a binary little-endian PLY vertex element."""
    header = read_ply_header(path)
    remaining = header.count
    with Path(path).open('rb') as handle:
        handle.seek(header.offset)
        while remaining > 0:
            block = np.fromfile(handle, dtype=header.dtype, count=min(chunk, remaining))
            if block.size == 0:
                raise ValueError(f'truncated PLY payload: {path}')
            remaining -= block.size
            yield block


def read_points(path: Path) -> np.ndarray:
    header = read_ply_header(path)
    with Path(path).open('rb') as handle:
        handle.seek(header.offset)
        data = np.fromfile(handle, dtype=header.dtype, count=header.count)
    if data.size != header.count:
        raise ValueError(f'truncated PLY payload: {path}')
    return data


def _xyz_block(data: np.ndarray) -> np.ndarray:
    if not {'x', 'y', 'z'} <= set(data.dtype.names or ()):
        raise ValueError('PLY lacks x/y/z')
    block = np.empty((data.size, 3), dtype=np.float64)
    block[:, 0] = data['x']
    block[:, 1] = data['y']
    block[:, 2] = data['z']
    return block


def xyz_native(data: np.ndarray) -> np.ndarray:
    """Native-precision XYZ as float64 for bounds/centroid reductions."""
    return _xyz_block(data)


def cells_f32(xyz: np.ndarray) -> np.ndarray:
    """(N, 3) float32 view of XYZ with signed zero normalised."""
    block = np.ascontiguousarray(xyz, dtype=np.float32)
    zero = block == 0.0
    if zero.any():
        block = block.copy()
        block[zero] = np.float32(0.0)
    return block


def keys_of(xyz: np.ndarray) -> np.ndarray:
    """Exact float32 cell keys; equality is bitwise equality of normalised cells."""
    block = cells_f32(xyz)
    if not np.isfinite(block).all():
        raise ValueError('non-finite coordinate present')
    raw = block.view(np.uint32).reshape(-1, 3)
    out = np.empty(raw.shape[0], dtype=KEY_DTYPE)
    out['x'] = raw[:, 0]
    out['y'] = raw[:, 1]
    out['z'] = raw[:, 2]
    return out


def fingerprint(keys: np.ndarray, unique: bool = False) -> str:
    """Order-independent SHA256 over exact float32 cell keys."""
    ordered = np.sort(keys)
    if unique:
        ordered = ordered[np.concatenate(([True], ordered[1:] != ordered[:-1]))]
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(ordered).tobytes())
    return digest.hexdigest()


def native_fingerprint(xyz: np.ndarray) -> str:
    """Order-independent SHA256 over the exact native float64 XYZ multiset."""
    block = np.ascontiguousarray(xyz, dtype=np.float64)
    order = np.lexsort((block[:, 2], block[:, 1], block[:, 0]))
    return hashlib.sha256(block[order].tobytes()).hexdigest()


def unique_cell_count(keys: np.ndarray) -> int:
    ordered = np.sort(keys)
    if ordered.size == 0:
        return 0
    return int(1 + np.count_nonzero(ordered[1:] != ordered[:-1]))


@dataclass
class TreeSource:
    """One publisher tree file occurrence (a candidate alias of a canonical tree)."""

    site: str
    relpath: str
    basename: str
    folder_partition: str
    point_count: int
    unique_cells: int
    bbox: tuple[float, float, float, float, float, float]
    centroid: tuple[float, float, float]
    native_dtype: str
    native_instance_ids: tuple[int, ...]
    fingerprint_f32: str
    fingerprint_native: str
    keys: np.ndarray = field(repr=False, default=None)
    xyz_min: tuple[float, float, float] = ()
    xyz_max: tuple[float, float, float] = ()
    path: str = ''

    @property
    def size_m(self) -> tuple[float, float, float]:
        return tuple(self.xyz_max[i] - self.xyz_min[i] for i in range(3))

    def as_row(self) -> dict:
        return {
            'site': self.site, 'relpath': self.relpath, 'basename': self.basename,
            'folder_partition': self.folder_partition, 'point_count': self.point_count,
            'unique_cells': self.unique_cells, 'bbox': list(self.bbox),
            'centroid': list(self.centroid), 'native_dtype': self.native_dtype,
            'native_instance_ids': list(self.native_instance_ids),
            'fingerprint_f32': self.fingerprint_f32,
            'fingerprint_native': self.fingerprint_native,
        }


def folder_partition_of(relpath: str) -> str:
    """Map a tree path below `<site>/trees/` onto a publisher folder partition."""
    parts = Path(relpath).parts
    if parts[0] != 'trees':
        return 'unknown'
    if parts[1] in ('train', 'val'):
        return parts[1]
    if parts[1] == 'test':
        if len(parts) > 2 and parts[2].startswith('in_plot'):
            return 'test_in_plot'
        if len(parts) > 2 and parts[2].startswith('out_plot'):
            return 'test_out_plot'
        return 'test'
    return 'unknown'


def describe_source(site: str, path: Path, site_root: Path,
                    keep_keys: bool = True) -> TreeSource:
    header = read_ply_header(path)
    data = read_points(path)
    xyz = xyz_native(data)
    keys = keys_of(xyz)
    names = set(header.names)
    ids: tuple[int, ...] = ()
    if 'instance' in names:
        ids = tuple(sorted(int(v) for v in np.unique(data['instance'])))
    relpath = str(path.relative_to(site_root)).replace('\\', '/')
    return TreeSource(
        site=site,
        relpath=relpath,
        basename=path.stem,
        folder_partition=folder_partition_of(relpath),
        point_count=int(data.size),
        unique_cells=unique_cell_count(keys),
        bbox=(float(xyz[:, 0].min()), float(xyz[:, 1].min()), float(xyz[:, 2].min()),
              float(xyz[:, 0].max()), float(xyz[:, 1].max()), float(xyz[:, 2].max())),
        centroid=(float(xyz[:, 0].mean()), float(xyz[:, 1].mean()), float(xyz[:, 2].mean())),
        native_dtype=str(header.dtype),
        native_instance_ids=ids,
        fingerprint_f32=fingerprint(keys),
        fingerprint_native=native_fingerprint(xyz),
        keys=keys if keep_keys else None,
        xyz_min=(float(xyz[:, 0].min()), float(xyz[:, 1].min()), float(xyz[:, 2].min())),
        xyz_max=(float(xyz[:, 0].max()), float(xyz[:, 1].max()), float(xyz[:, 2].max())),
        path=str(path),
    )


def discover_tree_sources(root: Path, keep_keys: bool = True) -> list[TreeSource]:
    """Every publisher tree PLY occurrence, deterministically ordered."""
    sources: list[TreeSource] = []
    for site_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        trees = site_dir / 'trees'
        if not trees.is_dir():
            continue
        for path in sorted(trees.rglob('*.ply')):
            sources.append(describe_source(site_dir.name, path, site_dir, keep_keys=keep_keys))
    return sources


def bbox_overlap(a: TreeSource, b: TreeSource, pad: float = 0.0) -> bool:
    for axis in range(3):
        if a.xyz_max[axis] + pad < b.xyz_min[axis] - pad:
            return False
        if b.xyz_max[axis] + pad < a.xyz_min[axis] - pad:
            return False
    return True


PAIR_DTYPE = np.dtype([('x', '<u4'), ('y', '<u4'), ('z', '<u4'), ('s', '<u4')])


def shared_cell_pairs(sources: list[TreeSource], min_shared: int = 1) -> dict:
    """Exact shared-cell counts for every source pair that shares at least one cell.

    A single sorted pass over every tree point of a site. Cell equality is exact
    float32 bit equality, so a shared cell is a genuine duplicated observation,
    not a quantisation artefact. Keys are relative paths, so the result does not
    depend on any caller's source ordering.
    """
    total = sum(source.point_count for source in sources)
    record = np.empty(total, dtype=PAIR_DTYPE)
    offset = 0
    names: list[str] = []
    for index, source in enumerate(sources):
        data = read_points(Path(source.path))
        keys = keys_of(xyz_native(data))
        count = keys.size
        record['x'][offset:offset + count] = keys['x']
        record['y'][offset:offset + count] = keys['y']
        record['z'][offset:offset + count] = keys['z']
        record['s'][offset:offset + count] = index
        offset += count
        names.append(source.relpath)
    record.sort()
    change = np.zeros(record.size, dtype=bool)
    if record.size > 1:
        change[1:] = ((record['x'][1:] != record['x'][:-1])
                      | (record['y'][1:] != record['y'][:-1])
                      | (record['z'][1:] != record['z'][:-1]))
    # Index 0 always starts the first run; `change[i]` marks a later run start.
    starts = np.concatenate(([0], np.flatnonzero(change))) if record.size else np.array([],
                                                                                      dtype=np.int64)
    ends = np.append(starts[1:], record.size)
    lengths = ends - starts
    counts: dict[tuple[str, str], int] = {}
    for run in np.flatnonzero(lengths > 1):
        unique, _ = np.unique(record['s'][starts[run]:ends[run]], return_counts=True)
        for i in range(unique.size):
            for j in range(i + 1, unique.size):
                left, right = names[int(unique[i])], names[int(unique[j])]
                key = (left, right) if left < right else (right, left)
                counts[key] = counts.get(key, 0) + 1
    return {key: value for key, value in counts.items() if value >= min_shared}


def pair_evidence(left: TreeSource, right: TreeSource, shared: int) -> dict:
    """Alias evidence for one pair. Never performs a union."""
    denominator = min(left.unique_cells, right.unique_cells)
    fraction = shared / denominator if denominator else 0.0
    gap = float(np.linalg.norm(np.asarray(left.centroid) - np.asarray(right.centroid)))
    exact = left.fingerprint_f32 == right.fingerprint_f32
    evidence = {
        'site': left.site,
        'a': left.relpath,
        'b': right.relpath,
        'same_basename': left.basename == right.basename,
        'point_count_a': left.point_count,
        'point_count_b': right.point_count,
        'shared_cells': shared,
        'containment_of_smaller': round(fraction, 6),
        'centroid_gap_m': round(gap, 6),
        'exact_multiset_equal': bool(exact),
        'bbox_overlap': bbox_overlap(left, right),
    }
    if exact:
        evidence['level'] = LEVEL_A
        if not evidence['same_basename']:
            evidence['conflict'] = 'exact coordinate multiset under differing basenames'
    elif (fraction >= STRONG_CONTAINMENT and gap <= STRONG_CENTROID_M
          and bbox_overlap(left, right)):
        evidence['level'] = LEVEL_B
    else:
        evidence['level'] = LEVEL_C
    return evidence


def canonical_manifest(sources: list[TreeSource], pairs: dict | None = None
                       ) -> tuple[list[dict], dict, list[dict]]:
    """Group occurrences into canonical trees with explicit alias evidence.

    No biological union happens without an exact coordinate multiset match;
    strong candidates stay flagged for review, partial overlaps stay independent.
    """
    pairs = pairs or {}
    by_site: dict[str, list[TreeSource]] = {}
    for source in sources:
        by_site.setdefault(source.site, []).append(source)

    rows: list[dict] = []
    evidence: list[dict] = []
    summary = {
        'source_file_occurrences': len(sources),
        'source_basenames': len({(s.site, s.basename) for s in sources}),
        'exact_alias_groups': 0,
        'exact_alias_extra_occurrences': 0,
        'strong_alias_groups': 0,
        'unresolved_groups': 0,
        'boundary_duplicate_groups': 0,
        'partial_overlap_nonaliases': 0,
        'basename_identity_conflicts': 0,
        'cross_basename_strong_candidates': 0,
        'max_partial_containment': 0.0,
    }

    for site in sorted(by_site):
        ordered = sorted(by_site[site], key=lambda s: s.relpath)
        by_relpath = {source.relpath: source for source in ordered}
        group_by_fp: dict[str, list[TreeSource]] = {}
        for source in ordered:
            group_by_fp.setdefault(source.fingerprint_f32, []).append(source)

        for (left_name, right_name), shared in sorted(pairs.items()):
            left, right = by_relpath.get(left_name), by_relpath.get(right_name)
            if left is None or right is None:
                continue  # pair belongs to another site
            item = pair_evidence(left, right, shared)
            if item['level'] == LEVEL_A:
                continue  # exact groups are reported through the manifest
            evidence.append(item)
            summary['max_partial_containment'] = max(summary['max_partial_containment'],
                                                     item['containment_of_smaller'])
            if item['level'] == LEVEL_B and not item['same_basename']:
                item['cross_basename_strong'] = True
                summary['cross_basename_strong_candidates'] += 1

        for fp in sorted(group_by_fp):
            members = sorted(group_by_fp[fp], key=lambda s: s.relpath)
            basenames = sorted({m.basename for m in members})
            partitions = sorted({m.folder_partition for m in members})
            boundary = len(partitions) > 1
            conflict = len(basenames) > 1
            if conflict:
                summary['basename_identity_conflicts'] += 1
                summary['unresolved_groups'] += 1
            status = IDENTITY_EXACT if not conflict else IDENTITY_UNRESOLVED
            if len(members) > 1:
                summary['exact_alias_groups'] += 1
                summary['exact_alias_extra_occurrences'] += len(members) - 1
            if boundary:
                summary['boundary_duplicate_groups'] += 1
            canonical_id = (f'{site}:{basenames[0]}' if len(basenames) == 1
                            else f'{site}:{basenames[0]}+{len(basenames) - 1}#{fp[:12]}')
            rows.append({
                'canonical_tree_id': canonical_id,
                'site': site,
                'identity_status': status,
                'alias_evidence': LEVEL_A if len(members) > 1 else 'SINGLE_OCCURRENCE',
                'source_aliases': [m.relpath for m in members],
                'source_file': members[0].relpath,
                'folder_partition': members[0].folder_partition,
                'folder_partitions': partitions,
                'basenames': basenames,
                'local_instance_id': sorted({i for m in members
                                             for i in m.native_instance_ids}),
                'point_count': members[0].point_count,
                'unique_cells': members[0].unique_cells,
                'bbox': list(members[0].bbox),
                'centroid': list(members[0].centroid),
                'coordinate_fingerprint': fp,
                'occurrence_count': len(members),
            })

    rows.sort(key=lambda r: (r['site'], r['canonical_tree_id']))
    evidence.sort(key=lambda e: (-e['shared_cells'], e['a'], e['b']))
    summary['strong_alias_groups'] = sum(1 for e in evidence if e['level'] == LEVEL_B)
    summary['partial_overlap_nonaliases'] = sum(1 for e in evidence
                                                if e['level'] == LEVEL_C)
    summary['canonical_tree_count'] = len(rows)
    summary['max_partial_containment'] = round(summary['max_partial_containment'], 6)
    return rows, summary, evidence


MANIFEST_IDENTITY_FIELDS = (
    'canonical_tree_id', 'site', 'identity_status', 'alias_evidence', 'source_aliases',
    'source_file', 'folder_partition', 'folder_partitions', 'basenames',
    'local_instance_id', 'point_count', 'unique_cells', 'bbox', 'centroid',
    'coordinate_fingerprint', 'occurrence_count',
)


def manifest_digest(rows: list[dict], fields: tuple[str, ...] = MANIFEST_IDENTITY_FIELDS) -> str:
    """Stable digest of the identity payload, independent of ordering or enrichment.

    Only identity fields are hashed, so adding derived membership/context columns
    later cannot change the digest of an otherwise identical manifest.
    """
    projection = [{key: row.get(key) for key in fields} for row in rows]
    payload = json.dumps(projection, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def full_manifest_digest(rows: list[dict]) -> str:
    """Digest over every field of the rows as written."""
    payload = json.dumps(rows, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# --- plot membership -------------------------------------------------------

PLOT_RECORD_DTYPE = np.dtype([('x', '<u4'), ('y', '<u4'), ('z', '<u4'), ('i', '<i4')])


def plot_records(path: Path, chunk: int = CHUNK_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Sorted exact float32 cell keys and their plot instance labels."""
    parts = []
    for block in iter_points(path, chunk):
        if 'instance' not in (block.dtype.names or ()):
            raise ValueError(f'plot cloud lacks instance field: {path}')
        keys = keys_of(xyz_native(block))
        record = np.empty(keys.size, dtype=PLOT_RECORD_DTYPE)
        record['x'] = keys['x']
        record['y'] = keys['y']
        record['z'] = keys['z']
        record['i'] = np.asarray(block['instance'], dtype=np.int32)
        parts.append(record)
    record = np.concatenate(parts)
    del parts
    record.sort()
    keys = np.empty(record.size, dtype=KEY_DTYPE)
    keys['x'] = record['x']
    keys['y'] = record['y']
    keys['z'] = record['z']
    instance = record['i'].copy()
    return keys, instance


def query_box(row: dict, pad: float) -> tuple[float, float, float, float, float, float]:
    """Candidate XY bbox expanded by `pad` metres; Z extent kept native."""
    x0, y0, z0, x1, y1, z1 = row['bbox']
    return (x0 - pad, y0 - pad, z0, x1 + pad, y1 + pad, z1)


def boxes_intersect(a, b) -> bool:
    return not (a[3] < b[0] or b[3] < a[0] or a[4] < b[1]
                or b[4] < a[1] or a[5] < b[2] or b[5] < a[2])


def build_context_index(rows: list[dict], pad: float) -> dict[str, list[str]]:
    """Provenance-bound neighbour index over canonical trees.

    A canonical tree is never its own neighbour, so a duplicate file occurrence
    (alias) of the same physical observation can never be counted as context.
    """
    by_site: dict[str, list[dict]] = {}
    for row in rows:
        by_site.setdefault(row['site'], []).append(row)
    index: dict[str, list[str]] = {}
    for members in by_site.values():
        for row in members:
            box = query_box(row, pad)
            index[row['canonical_tree_id']] = sorted(
                other['canonical_tree_id'] for other in members
                if other['canonical_tree_id'] != row['canonical_tree_id']
                and boxes_intersect(box, other['bbox']))
    return index


def match_keys(query: np.ndarray, plot_keys: np.ndarray,
               plot_instance: np.ndarray) -> dict:
    """Histogram of plot instance labels over cells equal to each query cell.

    The plot index must come from :func:`plot_records`. Membership is exact
    float32 cell equality; no nearest-neighbour or ID-based union is performed.
    """
    lo = np.searchsorted(plot_keys, query, side='left')
    hi = np.searchsorted(plot_keys, query, side='right')
    counts = (hi - lo).astype(np.int64)
    matched_cells = int(np.count_nonzero(counts))
    rows = int(counts.sum())
    if rows:
        base = np.cumsum(counts) - counts
        index = np.repeat(lo, counts) + (np.arange(rows, dtype=np.int64)
                                         - np.repeat(base, counts))
        labels, tally = np.unique(plot_instance[index], return_counts=True)
    else:
        labels, tally = np.array([], dtype=np.int32), np.array([], dtype=np.int64)
    return {
        'query_cells': int(query.size),
        'matched_cells': matched_cells,
        'matched_cell_fraction': round(matched_cells / query.size, 6) if query.size else 0.0,
        'matched_rows': rows,
        'instance_ids': [int(v) for v in labels],
        'instance_counts': [int(v) for v in tally],
    }
