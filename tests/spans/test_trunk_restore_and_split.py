from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from algorithms.folder_pipeline.tree_editor import (
    OriginalTree,
    OriginalTrees,
    TreeEditSession,
)
from algorithms.spans.tree_runner import _restore_validated_trunk_membership
from algorithms.vegetation.coarse_to_fine import RawRegion


def _raw_region(tmp_path, xyz: np.ndarray, classification: np.ndarray) -> RawRegion:
    xyz_path = tmp_path / "xyz.bin"
    index_path = tmp_path / "index.bin"
    class_path = tmp_path / "class.bin"
    np.asarray(xyz, dtype="<f4").tofile(xyz_path)
    np.arange(len(xyz), dtype="<i8").tofile(index_path)
    np.asarray(classification, dtype="<u1").tofile(class_path)
    return RawRegion(
        xyz_local_m_path=str(xyz_path),
        original_index_path=str(index_path),
        classification_path=str(class_path),
        count=len(xyz),
        bounds_min_xyz=tuple(np.min(xyz, axis=0)),
        bounds_max_xyz=tuple(np.max(xyz, axis=0)),
        origin_xyz=(0.0, 0.0, 0.0),
    )


def test_restore_trunk_preserves_low_source_rows_but_not_noise(tmp_path):
    trunk_z = np.linspace(0.10, 2.20, 22)
    trunk = np.column_stack((np.full(22, 0.03), np.zeros(22), trunk_z))
    unrelated = np.asarray([[2.0, 2.0, 1.3], [0.02, 0.01, 1.3], [0.02, 0.01, 0.0]])
    xyz = np.vstack((trunk, unrelated))
    # One nearby source point is LAS noise and one is ground: neither may be restored.
    classification = np.asarray([1] * 22 + [1, 7, 2], dtype=np.uint8)
    region = _raw_region(tmp_path, xyz, classification)
    raw_tree = np.full(len(xyz), -1, dtype=np.int32)
    stem = SimpleNamespace(
        stem_seed_id=5,
        xy_local_m=(0.0, 0.0),
        cross_section_radius_m=0.12,
        stem_vertical_extent_m=2.2,
    )
    candidate = SimpleNamespace(
        tree_id=3,
        quality_state=SimpleNamespace(value="CONFIRMED_TREE"),
        stem_seed_ids=(5,),
    )
    hybrid = SimpleNamespace(candidates=(candidate,), validated_stems=(stem,))
    ground = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )

    restored, diagnostics = _restore_validated_trunk_membership(
        region, raw_tree, hybrid, ground, (0.0, 0.0, 0.0)
    )

    assert np.all(restored[:22] == 3)
    assert restored[22:].tolist() == [-1, -1, -1]
    assert diagnostics["restored_raw_points"] == 22
    assert diagnostics["dbh_reference"] == "ORIGINAL_GROUND_PLUS_1.30_M"


def test_split_connected_tree_keeps_every_original_point():
    rng = np.random.default_rng(20260930)
    left = np.column_stack(
        (rng.normal(-0.8, 0.10, 300), rng.normal(0.0, 0.10, 300), rng.uniform(0.1, 8.0, 300))
    )
    right = np.column_stack(
        (rng.normal(0.8, 0.10, 300), rng.normal(0.0, 0.10, 300), rng.uniform(0.1, 8.0, 300))
    )
    xyz = np.vstack((left, right))
    original = OriginalTrees(
        records=(OriginalTree(4, "T-1-004", len(xyz), (0.0, 0.0, 8.0), 8.0),),
        original_index=np.arange(len(xyz), dtype=np.int64),
        tree_id=np.full(len(xyz), 4, dtype=np.int32),
        xyz_local_m=xyz,
    )
    session = TreeEditSession(original)
    session.select([4])

    child_ids = session.split_selected()
    first, _ = session.current_membership(child_ids[0])
    second, _ = session.current_membership(child_ids[1])

    assert not np.intersect1d(first, second).size
    assert np.array_equal(np.union1d(first, second), original.original_index)
    assert len(first) >= 100 and len(second) >= 100
    assert all(row["split"] for row in session.snapshot()["trees"])

