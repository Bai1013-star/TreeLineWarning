# ruff: noqa: E501
"""Structural regressions for the parallel stem-guided tree candidate."""

from __future__ import annotations

import numpy as np
from algorithms.spans.stem_guided_hybrid import (
    StemEvidence,
    StemGuidedHybridConfig,
    TreeQualityState,
    _deduplicate_crown_stems,
    segment_stem_guided_hybrid,
)
from algorithms.vegetation.coarse_to_fine import (
    RawRegion,
    build_fine_representatives,
    map_region_rows_to_tree,
)

CFG = StemGuidedHybridConfig(
    min_weighted_stem_support=3,
    min_crown_high_points=3,
    stem_dedup_radius_m=1.1,
    independent_stem_min_separation_m=1.4,
)


def _stem(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    h = np.arange(0.7, 5.3, 0.45)
    xyz = np.column_stack((np.full(len(h), x), np.full(len(h), y), h))
    return xyz, h


def _crown(x: float, y: float, *, radius: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([(x + dx, y + dy) for dx in (-radius, 0.0, radius) for dy in (-radius, 0.0, radius)])
    h = np.linspace(6.0, 8.5, len(xy))
    return np.column_stack((xy, h)), h


def _run(parts: list[tuple[np.ndarray, np.ndarray, int]], **kwargs):
    xyz = np.concatenate([part[0] for part in parts])
    hag = np.concatenate([part[1] for part in parts])
    labels = np.concatenate([np.full(len(part[1]), part[2], dtype=np.int32) for part in parts])
    return segment_stem_guided_hybrid(xyz, hag, labels, config=CFG, **kwargs)


def _states(result):
    return [candidate.quality_state for candidate in result.candidates]


def test_case_a_clear_stem_and_crown_is_confirmed_and_deterministic():
    stem = _stem(0.0, 0.0)
    crown = _crown(0.0, 0.0)
    one = _run([(stem[0], stem[1], 0), (crown[0], crown[1], 0)])
    two = _run([(stem[0], stem[1], 0), (crown[0], crown[1], 0)])
    assert _states(one) == [TreeQualityState.CONFIRMED_TREE]
    assert len(one.validated_stems) == 1
    assert np.array_equal(one.fine_tree_id, two.fine_tree_id)
    assert one.diagnostics["membership_duplicates"] == 0


def test_cases_b_c_h_multi_stem_splits_but_one_stem_does_not():
    s1, s2 = _stem(-1.0, 0.0), _stem(1.0, 0.0)
    shared = _crown(0.0, 0.0, radius=1.6)
    split = _run(
        [(s1[0], s1[1], 0), (s2[0], s2[1], 0), (shared[0], shared[1], 0)],
        crown_peak_count_by_id={0: 2},
    )
    assert _states(split).count(TreeQualityState.CONFIRMED_TREE) == 2
    assert split.diagnostics["multi_stem_split_count"] == 1
    s3 = _stem(0.0, 0.0)
    shoulder_a, shoulder_b = _crown(-1.0, 0.0), _crown(1.0, 0.0)
    unsplit = _run([(s3[0], s3[1], 0), (shoulder_a[0], shoulder_a[1], 0), (shoulder_b[0], shoulder_b[1], 0)])
    assert _states(unsplit).count(TreeQualityState.CONFIRMED_TREE) == 1
    assert unsplit.diagnostics["multi_stem_split_count"] == 0


def test_cases_d_e_f_g_small_real_pole_and_crown_only_semantics():
    tiny_xyz = np.array([[10.0, 10.0, 6.0], [10.1, 10.0, 6.1], [9.9, 10.0, 6.2]])
    tiny = _run([(tiny_xyz, tiny_xyz[:, 2], 0)])
    assert _states(tiny) == [TreeQualityState.REJECTED_SMALL_FRAGMENT]
    small_stem, small_crown = _stem(0.0, 0.0), _crown(0.0, 0.0, radius=0.35)
    small_real = _run([(small_stem[0], small_stem[1], 0), (small_crown[0], small_crown[1], 0)])
    assert _states(small_real) == [TreeQualityState.CONFIRMED_TREE]
    pole_stem = _stem(4.0, 0.0)
    pole = _run([(pole_stem[0], pole_stem[1], -1)])
    assert _states(pole) == [TreeQualityState.REJECTED_NO_CROWN]
    crown = _crown(8.0, 0.0)
    crown_only = _run([(crown[0], crown[1], 0)])
    assert _states(crown_only) == [TreeQualityState.CROWN_ONLY_REVIEW]


def test_cases_i_j_attachment_requires_real_3d_contact():
    stem, crown = _stem(0.0, 0.0), _crown(0.0, 0.0)
    touching = np.array([[1.25, 1.2, 8.5], [1.30, 1.2, 8.5], [1.25, 1.25, 8.45]])
    attached = _run([(stem[0], stem[1], 0), (crown[0], crown[1], 0), (touching, touching[:, 2], 1)])
    assert TreeQualityState.ATTACHED_FRAGMENT in _states(attached)
    separated = touching + np.array([3.0, 0.0, 0.0])
    not_attached = _run([(stem[0], stem[1], 0), (crown[0], crown[1], 0), (separated, separated[:, 2], 1)])
    assert TreeQualityState.ATTACHED_FRAGMENT not in _states(not_attached)
    assert TreeQualityState.REJECTED_SMALL_FRAGMENT in _states(not_attached)


def test_full_resolution_original_index_backmap_is_exclusive(tmp_path):
    stem, crown = _stem(0.0, 0.0), _crown(0.0, 0.0, radius=0.35)
    xyz = np.vstack((stem[0], crown[0])).astype("<f4")
    # Multiple raw points can share a fine voxel; every source row remains
    # traceable to no more than one output tree id.
    raw = np.repeat(xyz, 2, axis=0)
    original = np.arange(1000, 1000 + len(raw), dtype="<i8")
    xyz_path, original_path, class_path = (
        tmp_path / "xyz.bin",
        tmp_path / "original.bin",
        tmp_path / "class.bin",
    )
    raw.tofile(xyz_path)
    original.tofile(original_path)
    np.zeros(len(raw), dtype="<u1").tofile(class_path)
    region = RawRegion(
        xyz_local_m_path=str(xyz_path),
        original_index_path=str(original_path),
        classification_path=str(class_path),
        count=len(raw),
        bounds_min_xyz=tuple(raw.min(axis=0)),
        bounds_max_xyz=tuple(raw.max(axis=0)),
        origin_xyz=(0.0, 0.0, 0.0),
    )
    fine_path = tmp_path / "fine.npz"
    fine = build_fine_representatives(region, fine_path, voxel_size_m=0.10)
    result = segment_stem_guided_hybrid(
        fine["xyz_local_m"],
        fine["xyz_local_m"][:, 2],
        np.zeros(len(fine["original_index"]), dtype=np.int32),
        raw_count=fine["raw_count"],
        config=CFG,
    )
    raw_tree = map_region_rows_to_tree(
        region, fine_path, fine["original_index"], result.fine_tree_id
    )
    assigned_original = original[raw_tree >= 0]
    assert len(assigned_original) == len(np.unique(assigned_original))
    assert np.all(raw_tree[raw_tree >= 0] == 0)


def test_cases_k_l_m_n_stem_dedup_and_strict_split_gate():
    fragments = tuple(
        StemEvidence(
            stem_seed_id=index,
            xy_local_m=(0.08 * index, 0.0),
            stem_vertical_extent_m=3.5,
            occupied_height_bins=4,
            vertical_gap_ratio=0.0,
            pca_verticality=0.95,
            cross_section_radius_m=0.2,
            point_support=20,
            xy_drift_m=0.1,
            local_vegetation_connectivity=1.0,
            stem_confidence=0.9,
        )
        for index in range(5)
    )
    assert len(_deduplicate_crown_stems(fragments, CFG)) == 1  # Case K
    trunk = _stem(0.0, 0.0)
    foliage = [_stem(x, 0.2) for x in (-0.45, -0.2, 0.2, 0.45)]
    crown = _crown(0.0, 0.0, radius=1.5)
    l_parts = [(trunk[0], trunk[1], 0), (crown[0], crown[1], 0)]
    l_parts.extend((item[0], item[1], 0) for item in foliage)
    foliage_crown = _run(l_parts, crown_peak_count_by_id={0: 2})
    assert _states(foliage_crown).count(TreeQualityState.CONFIRMED_TREE) == 1  # Case L
    left, right = _stem(-1.4, 0.0), _stem(1.4, 0.0)
    split_crown = _run(
        [(left[0], left[1], 0), (right[0], right[1], 0), (crown[0], crown[1], 0)],
        crown_peak_count_by_id={0: 2},
    )
    assert split_crown.diagnostics["multi_stem_split_count"] == 1  # Case M
    branch_like = _stem(2.0, 0.0)
    no_split = _run(
        [(trunk[0], trunk[1], 0), (branch_like[0], branch_like[1], 0), (crown[0], crown[1], 0)]
    )
    assert no_split.diagnostics["multi_stem_split_count"] == 0  # Case N
