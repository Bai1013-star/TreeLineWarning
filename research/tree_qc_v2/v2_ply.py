"""Bounded binary PLY reader for the V2 feasibility round.

Read-only reuse of the accepted identity parser: the same type map, the same
``end_header`` scan, the same structured-array read. Nothing here writes to or
modifies the frozen V1 tree, and nothing here imports the frozen generator.

Only ``binary_little_endian`` vertex elements without list properties are
supported, which is what the public benchmark publishes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

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

CHUNK_POINTS = 2_000_000


class PlyError(RuntimeError):
    pass


class PlyHeader:
    __slots__ = ('offset', 'count', 'dtype', 'properties', 'text')

    def __init__(self, offset, count, dtype, properties, text):
        self.offset = offset
        self.count = count
        self.dtype = dtype
        self.properties = properties
        self.text = text

    @property
    def names(self):
        return self.dtype.names or ()


def read_ply_header(path: Path) -> PlyHeader:
    with Path(path).open('rb') as handle:
        raw = handle.read(65536)
    index = raw.find(b'end_header')
    if index < 0:
        raise PlyError(f'PLY header without end_header: {path}')
    newline = raw.find(b'\n', index)
    if newline < 0:
        raise PlyError(f'PLY header without terminator: {path}')
    offset = newline + 1
    lines = raw[:offset].decode('ascii').splitlines()
    if not lines or lines[0] != 'ply':
        raise PlyError(f'not a PLY file: {path}')
    fmt, count = None, None
    properties = []
    for line in lines[1:]:
        if line.startswith('format '):
            fmt = line.split()[1]
        elif line.startswith('element vertex '):
            count = int(line.split()[2])
        elif line.startswith('property '):
            parts = line.split()
            if parts[1] == 'list':
                raise PlyError(f'list property unsupported: {path}')
            if parts[1] not in PLY_TYPES:
                raise PlyError(f'unknown PLY type {parts[1]!r}: {path}')
            properties.append((parts[2], PLY_TYPES[parts[1]]))
    if fmt != 'binary_little_endian':
        raise PlyError(f'unsupported PLY format {fmt!r}: {path}')
    if count is None or not properties:
        raise PlyError(f'incomplete PLY header: {path}')
    return PlyHeader(offset, count, np.dtype(properties), tuple(properties), ''.join(lines))


def iter_points(path: Path, chunk: int = CHUNK_POINTS):
    header = read_ply_header(path)
    remaining = header.count
    with Path(path).open('rb') as handle:
        handle.seek(header.offset)
        while remaining > 0:
            block = np.fromfile(handle, dtype=header.dtype, count=min(chunk, remaining))
            if block.size == 0:
                raise PlyError(f'truncated PLY payload: {path}')
            remaining -= block.size
            yield block


def read_points(path: Path) -> np.ndarray:
    header = read_ply_header(path)
    with Path(path).open('rb') as handle:
        handle.seek(header.offset)
        data = np.fromfile(handle, dtype=header.dtype, count=header.count)
    if data.size != header.count:
        raise PlyError(f'truncated PLY payload: {path}')
    return data


def read_xyz(path: Path, dtype=np.float32) -> np.ndarray:
    """(N, 3) XYZ, no other property touched."""
    data = read_points(path)
    names = set(data.dtype.names or ())
    if not {'x', 'y', 'z'} <= names:
        raise PlyError(f'PLY lacks x/y/z: {path}')
    block = np.empty((data.size, 3), dtype=dtype)
    block[:, 0] = data['x']
    block[:, 1] = data['y']
    block[:, 2] = data['z']
    return block


def read_xyz_extra(path: Path, extra=()):
    """(N, 3) XYZ plus the requested optional scalar properties.

    Returns ``(xyz, extras)`` where ``extras`` maps each found property name to a
    1-D array and each missing one to ``None``.
    """
    data = read_points(path)
    names = set(data.dtype.names or ())
    if not {'x', 'y', 'z'} <= names:
        raise PlyError(f'PLY lacks x/y/z: {path}')
    xyz = np.empty((data.size, 3), dtype=np.float32)
    xyz[:, 0] = data['x']
    xyz[:, 1] = data['y']
    xyz[:, 2] = data['z']
    found = {}
    for name in extra:
        found[name] = None if name not in names else np.asarray(data[name])
    return xyz, found
