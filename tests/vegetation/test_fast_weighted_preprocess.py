"""Focused tests for FAST-only raw-population-preserving V4 preprocessing."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
from algorithms.vegetation.fast_research import _fast_v4_config
from algorithms.vegetation.fast_weighted_preprocess import (
    FAST_WEIGHTED_MARKER,
    FAST_WEIGHTED_PREPROCESS_SCHEMA,
    WeightedWorkingSet,
    backmap_representative_tree_ids,
    build_weighted_working_set,
)
from algorithms.vegetation.v4_tree_segmentation import (
    _build_grid,
    _NormalPopulation,
    derive_algorithm_scales,
)


def _write_las(path: Path, xyz: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    data = laspy.LasData(header)
    data.x, data.y, data.z = xyz.T
    data.classification = np.zeros(len(xyz), dtype=np.uint8)
    data.write(path)


def test_weighted_working_set_conserves_all_raw_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    _write_las(
        source,
        np.asarray(
            (
                (0.01, 0.01, 0.0),
                (0.10, 0.10, 0.1),
                (0.60, 0.10, 0.2),
                (0.61, 0.10, 0.3),
                (1.10, 0.10, 0.4),
            ),
            dtype=np.float64,
        ),
    )
    status, working = build_weighted_working_set(
        source, tmp_path / "working", voxel_size_m=0.5, chunk_size=2
    )
    assert status == "PASS"
    assert working.source_point_count == 5
    assert working.representative_count == 3
    assert working.raw_count_sum == 5
    mapping = np.load(working.raw_to_representative_npy_path, mmap_mode="r")
    with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
        raw_count = np.asarray(stream["raw_count"], dtype=np.int64)
        representative_original = np.asarray(stream["original_index"], dtype=np.int64)
    assert np.array_equal(np.bincount(mapping, minlength=len(raw_count)), raw_count)
    assert representative_original.tolist() == [0, 2, 4]


def test_raw_weights_change_density_without_changing_representative_layout() -> None:
    config = _fast_v4_config(maximum_points=200_000)
    count = 2
    population = _NormalPopulation(
        source_identity=None,  # _build_grid only consumes measured representative arrays.
        original_index=np.asarray((10, 20), dtype=np.int64),
        xyz_local_m=np.asarray(((0.1, 0.1, 2.0), (0.2, 0.2, 2.2)), dtype=np.float64),
        hag_m=np.asarray((2.0, 2.2), dtype=np.float32),
        membership_state=np.ones(count, dtype=np.uint8),
        hag_evidence_state=np.ones(count, dtype=np.uint8),
        segmentation_hag_readiness=np.ones(count, dtype=np.uint8),
        candidate_reason_flags=np.zeros(count, dtype=np.uint32),
        raw_weight=np.asarray((120, 5), dtype=np.uint32),
        handoff_digest="0" * 64,
        validated_view_digest="1" * 64,
        coverage_report_digest="2" * 64,
        point_index_by_original_index={10: 0, 20: 1},
    )
    grid = _build_grid(population, derive_algorithm_scales(config), config)
    summary = next(iter(grid.cell_id_to_summary.values()))
    assert summary.count == 2
    assert summary.density == 125.0 / (config.xy_cell_size_m**2)


def test_backmapping_reports_raw_weighted_assignment(tmp_path: Path) -> None:
    roi_root = tmp_path / "roi"
    roi_root.mkdir()
    np.savez(
        roi_root / "scan_large_roi_computation_set.npz",
        representative_id=np.asarray((3, 8, 11), dtype=np.uint32),
        original_index=np.asarray((30, 80, 110), dtype=np.int64),
        raw_count=np.asarray((12, 7, 3), dtype=np.uint32),
    )
    vegetation = tmp_path / "vegetation.npz"
    np.savez(
        vegetation,
        original_index=np.asarray((30, 110), dtype=np.int64),
        raw_count=np.asarray((12, 3), dtype=np.uint32),
    )
    v4 = tmp_path / "v4.npz"
    np.savez(
        v4,
        original_index=np.asarray((30, 110), dtype=np.int64),
        tree_id_by_point=np.asarray((2, -1), dtype=np.int32),
    )
    working = WeightedWorkingSet(
        schema_version=FAST_WEIGHTED_PREPROCESS_SCHEMA,
        non_certified_marker=FAST_WEIGHTED_MARKER,
        source_las_path=str(tmp_path / "source.las"),
        source_las_sha256="0" * 64,
        source_point_count=22,
        source_bounds_min_xyz=(0.0, 0.0, 0.0),
        source_bounds_max_xyz=(1.0, 1.0, 1.0),
        source_scales=(0.001, 0.001, 0.001),
        source_offsets=(0.0, 0.0, 0.0),
        voxel_size_m=0.25,
        representative_count=12,
        raw_count_sum=22,
        representatives_npz_path=str(tmp_path / "reps.npz"),
        representative_las_path=str(tmp_path / "reps.las"),
        raw_to_representative_npy_path=str(tmp_path / "mapping.npy"),
        classification_counts={"0": 22},
        elapsed_seconds=0.0,
        config_digest="1" * 64,
    )
    result = backmap_representative_tree_ids(
        working, roi_root, vegetation, v4, tmp_path / "backmap.npz"
    )
    assert result["assigned_raw_weight"] == 12
    assert result["unassigned_raw_weight"] == 10
    with np.load(tmp_path / "backmap.npz", allow_pickle=False) as stream:
        assert stream["tree_id_by_representative"].tolist() == [2, -1, -1]
