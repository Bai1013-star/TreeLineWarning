"""Shared fixtures/helpers for the Folder pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import (
    OriginalTree,
    OriginalTrees,
)


def write_synthetic_las(path: Path, xyz: np.ndarray, *, seed: int = 7) -> Path:
    """Write a small synthetic LAS with real coordinates (for display checks)."""

    import laspy

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.asarray((0.0, 0.0, 0.0))
    las = laspy.LasData(header)
    las.x = np.asarray(xyz[:, 0], dtype=np.float64)
    las.y = np.asarray(xyz[:, 1], dtype=np.float64)
    las.z = np.asarray(xyz[:, 2], dtype=np.float64)
    rng = np.random.default_rng(seed)
    rgb8 = rng.integers(0, 255, (len(xyz), 3))
    las.red = (rgb8[:, 0].astype(np.uint16) * 256).astype(np.uint16)
    las.green = (rgb8[:, 1].astype(np.uint16) * 256).astype(np.uint16)
    las.blue = (rgb8[:, 2].astype(np.uint16) * 256).astype(np.uint16)
    las.write(path)
    return path


def make_workspace(tmp_path: Path, xyz: np.ndarray | None = None) -> Path:
    """Create a minimal valid Folder workspace with one synthetic LAS."""

    workspace = tmp_path / "workspace"
    pointcloud = workspace / "input" / "pointcloud"
    pointcloud.mkdir(parents=True, exist_ok=True)
    if xyz is None:
        rng = np.random.default_rng(3)
        xyz = rng.uniform([1000.0, 2000.0, 2.0], [1400.0, 2200.0, 30.0], (2000, 3))
    write_synthetic_las(pointcloud / "source.las", xyz)
    return workspace


def build_original_trees_fixture(
    *,
    tree_sizes: list[int] | None = None,
    tree_count: int = 4,
) -> OriginalTrees:
    """Build an immutable OriginalTrees with N distinct trees."""

    sizes = tree_sizes or [100, 120, 140, 160][:tree_count]
    parts_index: list[np.ndarray] = []
    parts_tree: list[np.ndarray] = []
    parts_xyz: list[np.ndarray] = []
    next_index = 0
    records: list[OriginalTree] = []
    for tree_id, size in enumerate(sizes):
        rng = np.random.default_rng(tree_id + 1)
        xy = np.column_stack(
            (
                np.full(size, 10.0 * tree_id) + rng.normal(0, 0.8, size),
                np.full(size, 50.0 + 3.0 * tree_id) + rng.normal(0, 0.8, size),
            )
        )
        z = np.linspace(2.0, 12.0 + tree_id, size) + rng.normal(0, 0.3, size)
        xyz = np.column_stack((xy, z))
        idx = np.arange(next_index, next_index + size, dtype=np.int64)
        next_index += size
        parts_index.append(idx)
        parts_tree.append(np.full(size, tree_id, dtype=np.int32))
        parts_xyz.append(xyz)
        records.append(
            OriginalTree(
                tree_id=tree_id,
                external_name=f"Line-Span001-Tree{tree_id + 1:03d}",
                point_count=size,
                apex_xy_local_m=(float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1])), float(z[-1])),
                apex_height_m=float(z[-1]),
            )
        )
    return OriginalTrees(
        records=tuple(records),
        original_index=np.concatenate(parts_index),
        tree_id=np.concatenate(parts_tree),
        xyz_local_m=np.concatenate(parts_xyz),
    )


@pytest.fixture
def original_trees() -> OriginalTrees:
    return build_original_trees_fixture()
