"""Synthetic LAS scene builder and formal-source loader for FGC tests.

Synthetic fixtures only; never imported by production code and never required
for the real scan_small runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import laspy
import numpy as np

from tests.io.test_source_provider import (
    _configuration,
    _load,
    _provider_config,
    _reference,
    _source_identity,
)


class SyntheticScene:
    """A written synthetic LAS with a validated formal source result."""

    def __init__(self, path: Path, reference, config, configuration, result) -> None:
        self.path = path
        self.reference = reference
        self.config = config
        self.configuration = configuration
        self.result = result

    @property
    def count(self) -> int:
        return self.result.source_identity.source_point_count


def write_synthetic_las(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray | None = None,
    point_format: int = 7,
    scales: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
) -> None:
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = scales if scales is not None else np.array([0.01, 0.01, 0.01])
    header.offsets = offsets if offsets is not None else np.array([100.0, 200.0, 0.0])
    las = laspy.LasData(header)
    las.x = np.asarray(x, dtype=np.float64)
    las.y = np.asarray(y, dtype=np.float64)
    las.z = np.asarray(z, dtype=np.float64)
    if classification is None:
        classification = np.zeros(len(las.x), dtype=np.uint8)
    las.classification = np.asarray(classification, dtype=np.uint8)
    las.write(path)
    return Path(path)


def load_scene(
    path: Path,
    *,
    chunk_size: int = 50_000,
    source_id: str = "synthetic",
    pass_through_dimensions=(),
) -> SyntheticScene:
    """Write-time-independent loader: builds the formal reference from a LAS."""

    las_path = Path(path)
    with laspy.open(las_path, mode="r") as reader:
        las = reader.read()
    count = int(las.header.point_count)
    min_xyz = (float(las.header.mins[0]), float(las.header.mins[1]), float(las.header.mins[2]))
    max_xyz = (float(las.header.maxs[0]), float(las.header.maxs[1]), float(las.header.maxs[2]))
    file_sha = hashlib.sha256(las_path.read_bytes()).hexdigest()
    config = _provider_config(pass_through_dimensions=pass_through_dimensions)
    configuration = _configuration()
    identity = _source_identity(str(las_path), file_sha, count, (list(min_xyz), list(max_xyz)))
    reference = _reference(
        las_path,
        identity,
        config,
        configuration,
        component_sha=file_sha,
        component_size=las_path.stat().st_size,
    )
    session = _load(reference, config, configuration, chunk_size=chunk_size)
    for _ in session.iter_chunks():
        pass
    result = session.finalize()
    return SyntheticScene(las_path, reference, config, configuration, result)


def make_flat_scene(
    path: Path, *, side_m: float = 10.0, spacing_m: float = 0.5, z: float = 0.0
) -> SyntheticScene:
    gx, gy = np.meshgrid(
        np.arange(0.0, side_m, spacing_m),
        np.arange(0.0, side_m, spacing_m),
    )
    zs = np.full(gx.ravel().shape, z, dtype=np.float64)
    write_synthetic_las(path, x=100.0 + gx.ravel(), y=200.0 + gy.ravel(), z=zs)
    return load_scene(path)
