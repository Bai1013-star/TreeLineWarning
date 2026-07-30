from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from typing import Any

import algorithms.vegetation as vegetation
import algorithms.vegetation.tree_segmentation as construction
import numpy as np
import pytest
from algorithms.vegetation.canopy_height_model import (
    CanopyHeightModelConfig,
    build_canopy_height_model,
)
from algorithms.vegetation.canopy_segmentation import (
    CanopySegmentationConfig,
    segment_canopy_regions,
)
from algorithms.vegetation.tree_segmentation import (
    TreeResultConstructionConfig,
    TreeResultConstructionError,
    TreeResultConstructionErrorCode,
    build_tree_segmentation_result,
)
from algorithms.vegetation.treetop_candidates import (
    TreetopDetectionConfig,
    detect_treetop_candidates,
)
from core.contracts import (
    TREE_SEGMENTATION_ALGORITHM_VERSION,
    TREE_SEGMENTATION_SCHEMA_VERSION,
    TreeSegmentationInputClusterDisposition,
    TreeSegmentationInputDisposition,
    TreeSegmentationStatus,
    VegetationClusterStatus,
)

from tests.vegetation.test_canopy_height_model import _tree_input


def _canonicalize_input(tree_input: Any) -> Any:
    xyz = np.array(tree_input.vegetation_points.xyz_local, dtype=np.float32, order="C", copy=True)
    xyz.setflags(write=False)
    object.__setattr__(tree_input.vegetation_points, "xyz_local", xyz)
    origin = np.array(tree_input.identity.origin_xyz, dtype=np.float64, order="C", copy=True)
    origin.setflags(write=False)
    object.__setattr__(tree_input.geometry_features, "origin_xyz", origin)
    status_rules = {
        VegetationClusterStatus.CONFIRMED: (
            TreeSegmentationInputDisposition.ELIGIBLE,
            "confirmed_distance_ready",
        ),
        VegetationClusterStatus.REVIEW_REQUIRED: (
            TreeSegmentationInputDisposition.REVIEW_ONLY,
            "review_required",
        ),
        VegetationClusterStatus.PROVISIONAL: (
            TreeSegmentationInputDisposition.REVIEW_ONLY,
            "provisional",
        ),
        VegetationClusterStatus.REJECTED_AS_INSTANCE: (
            TreeSegmentationInputDisposition.EXCLUDED,
            "rejected_as_instance",
        ),
    }
    rows = sorted(
        (
            int(cluster_id),
            *status_rules[VegetationClusterStatus(int(status))],
        )
        for cluster_id, status in zip(
            tree_input.cluster_id,
            tree_input.cluster_status,
            strict=True,
        )
    )
    object.__setattr__(
        tree_input,
        "cluster_dispositions",
        tuple(
            TreeSegmentationInputClusterDisposition(cluster_id, disposition, reason)
            for cluster_id, disposition, reason in rows
        ),
    )
    return tree_input


def _pipeline(
    *,
    xy: list[tuple[float, float]],
    hag: list[float],
    statuses: list[VegetationClusterStatus] | None = None,
    cluster_ids: list[int] | None = None,
    wire_ids: list[int] | None = None,
    ground_usable: list[bool] | None = None,
    minimum_hag_m: float = 0.0,
    minimum_peak_hag_m: float = 0.0,
) -> tuple[Any, Any, Any, Any]:
    tree_input = _canonicalize_input(
        _tree_input(
            xy=xy,
            hag=hag,
            statuses=statuses,
            cluster_ids=cluster_ids,
            wire_ids=wire_ids,
            ground_usable=ground_usable,
        )
    )
    model = build_canopy_height_model(
        tree_input,
        CanopyHeightModelConfig(cell_size_m=1.0, minimum_hag_m=minimum_hag_m),
    )
    candidates = detect_treetop_candidates(
        model,
        TreetopDetectionConfig(
            minimum_peak_hag_m=minimum_peak_hag_m,
            local_max_radius_m=0.0,
            prominence_radius_m=1.5,
        ),
    )
    segmentation = segment_canopy_regions(model, candidates, CanopySegmentationConfig())
    return tree_input, model, candidates, segmentation


def _build(scene: tuple[Any, Any, Any, Any], **config: Any) -> Any:
    return build_tree_segmentation_result(
        *scene,
        config=TreeResultConstructionConfig(**config) if config else None,
    )


def _frozen_array(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _identity_with_point_count(identity: Any, point_count: int) -> Any:
    values = {name: getattr(identity, name) for name in identity.__dataclass_fields__}
    values["point_count"] = point_count
    return type(identity)(**values)


def test_public_api_signature_exports_and_frozen_config() -> None:
    expected = {
        "TreeResultConstructionConfig",
        "TreeResultConstructionErrorCode",
        "TreeResultConstructionError",
        "build_tree_segmentation_result",
    }
    assert expected <= set(vegetation.__all__)
    for name in expected:
        assert getattr(vegetation, name) is getattr(construction, name)
    assert set(construction.__all__) == expected
    assert tuple(inspect.signature(build_tree_segmentation_result).parameters) == (
        "tree_input",
        "canopy_model",
        "treetop_candidates",
        "canopy_segmentation",
        "config",
    )
    config = TreeResultConstructionConfig()
    assert config == TreeResultConstructionConfig(
        maximum_source_points=10_000_000,
        maximum_grid_cells=10_000_000,
        maximum_candidates=100_000,
        maximum_source_clusters=1_000_000,
        maximum_grouped_member_rows=10_000_000,
        maximum_additional_bytes=2_147_483_648,
        chunk_point_count=250_000,
    )
    with pytest.raises(FrozenInstanceError):
        config.maximum_candidates = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    tuple(TreeResultConstructionConfig.__dataclass_fields__),
)
def test_config_accepts_python_and_numpy_integers(field: str) -> None:
    assert getattr(TreeResultConstructionConfig(**{field: 3}), field) == 3
    value = getattr(TreeResultConstructionConfig(**{field: np.int64(4)}), field)
    assert value == 4
    assert type(value) is int


@pytest.mark.parametrize("value", [0, -1, True, np.bool_(False), 1.0, "1"])
def test_config_rejects_nonpositive_bool_and_noninteger(value: object) -> None:
    with pytest.raises(ValueError):
        TreeResultConstructionConfig(chunk_point_count=value)  # type: ignore[arg-type]


def test_config_representability_limits() -> None:
    assert TreeResultConstructionConfig(maximum_candidates=2_147_483_646)
    assert TreeResultConstructionConfig(maximum_source_clusters=2_147_483_647)
    with pytest.raises(ValueError):
        TreeResultConstructionConfig(maximum_candidates=2_147_483_647)
    with pytest.raises(ValueError):
        TreeResultConstructionConfig(maximum_source_clusters=2_147_483_648)


def test_error_enum_and_exception_contract() -> None:
    assert [value.value for value in TreeResultConstructionErrorCode] == [
        "resource_limit",
        "allocation_failed",
        "input_bundle_mismatch",
        "point_mapping_mismatch",
        "internal_invariant",
    ]
    error = TreeResultConstructionError(
        TreeResultConstructionErrorCode.POINT_MAPPING_MISMATCH,
        " retained message ",
        source_point_count=3,
        chunk_start=1,
        chunk_end=2,
    )
    assert str(error) == " retained message "
    assert error.source_point_count == 3
    assert error.chunk_start == 1
    assert error.chunk_end == 2
    assert not hasattr(error, "details")
    with pytest.raises(ValueError):
        TreeResultConstructionError("resource_limit", "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TreeResultConstructionError(
            TreeResultConstructionErrorCode.RESOURCE_LIMIT,
            "x",
            source_point_count=np.int64(1),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        TreeResultConstructionError(
            TreeResultConstructionErrorCode.RESOURCE_LIMIT,
            "x",
            chunk_start=2,
            chunk_end=1,
        )


@pytest.mark.parametrize("position", range(4))
def test_public_arguments_require_exact_types(position: int) -> None:
    scene = list(_pipeline(xy=[], hag=[]))
    scene[position] = object()
    with pytest.raises(TypeError):
        build_tree_segmentation_result(*scene)


def test_empty_result_is_canonical_and_zero_copy() -> None:
    result = _build(_pipeline(xy=[], hag=[]))
    assert result.schema_version == TREE_SEGMENTATION_SCHEMA_VERSION
    assert result.algorithm_version == TREE_SEGMENTATION_ALGORITHM_VERSION
    assert result.records == ()
    assert result.statistics.input_point_count == 0
    assert result.statistics.record_count == 0
    assert result.segment_id_by_point.tolist() == []
    assert result.tree_member_offsets.tolist() == [0]
    assert result.ordered_tree_member_row_index.tolist() == []
    for value in (
        result.segment_id_by_point,
        result.tree_id_by_point,
        result.tree_distance_ready_by_point,
        result.tree_member_offsets,
        result.ordered_tree_member_row_index,
    ):
        assert type(value) is np.ndarray
        assert value.flags.owndata
        assert value.flags.c_contiguous
        assert not value.flags.writeable


def test_one_marker_point_constructs_one_provisional_record() -> None:
    result = _build(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    assert result.segment_id_by_point.tolist() == [0]
    assert result.tree_id_by_point.tolist() == [-1]
    assert result.tree_distance_ready_by_point.tolist() == [False]
    assert result.tree_member_offsets.tolist() == [0, 1]
    assert result.ordered_tree_member_row_index.tolist() == [0]
    assert len(result.records) == 1
    record = result.records[0]
    assert record.segment_id == 0
    assert record.tree_id == -1
    assert record.status is TreeSegmentationStatus.PROVISIONAL
    assert record.source_cluster_ids == (0,)
    assert record.source_cluster_statuses == (VegetationClusterStatus.CONFIRMED,)
    assert record.source_cluster_distance_ready == (True,)
    assert record.tree_distance_ready is False
    assert record.confidence == 0.0
    assert record.geometry_summary is None
    assert record.crown_summary is None
    assert int(record.quality_flags) == 0


def test_all_excluded_and_all_eligible_unseeded() -> None:
    excluded = _build(
        _pipeline(
            xy=[(0.1, 0.1)],
            hag=[2.0],
            statuses=[VegetationClusterStatus.REJECTED_AS_INSTANCE],
        )
    )
    assert excluded.segment_id_by_point.tolist() == [-1]
    assert excluded.records == ()
    assert excluded.input_provenance["mapping_audit"]["excluded_point_count"] == 1

    unseeded = _build(
        _pipeline(
            xy=[(0.1, 0.1)],
            hag=[2.0],
            minimum_peak_hag_m=3.0,
        )
    )
    assert unseeded.segment_id_by_point.tolist() == [-1]
    assert unseeded.records == ()
    assert unseeded.warnings == (
        "eligible source points remain unassigned to a candidate-linked region",
    )
    assert unseeded.input_provenance["mapping_audit"]["unseeded_domain_point_count"] == 1


@pytest.mark.parametrize(
    ("changes", "assigned", "source_status"),
    [
        ({}, True, VegetationClusterStatus.CONFIRMED),
        (
            {"statuses": [VegetationClusterStatus.REVIEW_REQUIRED]},
            True,
            VegetationClusterStatus.REVIEW_REQUIRED,
        ),
        (
            {"statuses": [VegetationClusterStatus.PROVISIONAL]},
            True,
            VegetationClusterStatus.PROVISIONAL,
        ),
        (
            {"statuses": [VegetationClusterStatus.REJECTED_AS_INSTANCE]},
            False,
            None,
        ),
        ({"wire_ids": [4]}, False, None),
        ({"ground_usable": [False]}, False, None),
        ({"hag": [0.49], "minimum_hag_m": 0.5}, False, None),
        ({"hag": [0.5], "minimum_hag_m": 0.5}, True, VegetationClusterStatus.CONFIRMED),
        (
            {
                "statuses": [VegetationClusterStatus.UNASSIGNED],
                "cluster_ids": [-1],
            },
            False,
            None,
        ),
    ],
)
def test_exact_v3b1_eligibility_gate(
    changes: dict[str, Any],
    assigned: bool,
    source_status: VegetationClusterStatus | None,
) -> None:
    values: dict[str, Any] = {"xy": [(0.1, 0.1)], "hag": [2.0]}
    values.update(changes)
    result = _build(_pipeline(**values))
    assert result.segment_id_by_point.tolist() == ([0] if assigned else [-1])
    if assigned:
        assert result.records[0].source_cluster_statuses == (source_status,)
    else:
        assert result.records == ()


def test_cross_cluster_traceability_and_warning() -> None:
    result = _build(
        _pipeline(
            xy=[(0.1, 0.1), (0.2, 0.2)],
            hag=[2.0, 1.0],
            cluster_ids=[7, 3],
        )
    )
    assert result.ordered_tree_member_row_index.tolist() == [0, 1]
    assert result.records[0].source_cluster_ids == (3, 7)
    assert result.records[0].source_cluster_distance_ready == (True, True)
    assert result.warnings == (
        "one or more canopy segments contain points from multiple V2 source clusters",
    )


def test_ambiguous_positive_label_retains_owner_and_warning() -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    segmentation = scene[3]
    ambiguity = np.ones((1, 1), dtype=np.bool_)
    ambiguity.setflags(write=False)
    counts = np.ones(1, dtype=np.int64)
    counts.setflags(write=False)
    object.__setattr__(segmentation, "ambiguity_mask", ambiguity)
    object.__setattr__(segmentation, "ambiguous_cell_count_by_candidate", counts)
    object.__setattr__(segmentation, "ambiguous_cell_count", 1)
    result = _build(tuple(scene))
    assert result.segment_id_by_point.tolist() == [0]
    assert result.warnings[0] == (
        "ambiguous assigned source points are retained in their V3B3 regions"
    )


def test_different_chunk_sizes_are_value_identical() -> None:
    scene = _pipeline(
        xy=[(0.1, 0.1), (0.2, 0.2), (1.1, 0.1), (1.2, 0.2)],
        hag=[2.0, 1.0, 3.0, 2.0],
        cluster_ids=[5, 5, 9, 9],
    )
    first = _build(scene, chunk_point_count=1)
    second = _build(scene, chunk_point_count=3)
    for name in (
        "segment_id_by_point",
        "tree_id_by_point",
        "tree_distance_ready_by_point",
        "tree_member_offsets",
        "ordered_tree_member_row_index",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.records == second.records
    assert first.input_provenance["mapping_audit"] == second.input_provenance["mapping_audit"]


def test_membership_duplicate_is_detected() -> None:
    scene = list(
        _pipeline(
            xy=[(0.1, 0.1), (0.2, 0.2)],
            hag=[2.0, 1.0],
            cluster_ids=[4, 4],
        )
    )
    rows = np.array([0, 0], dtype=np.int64)
    rows.setflags(write=False)
    object.__setattr__(scene[0], "ordered_cluster_member_row_index", rows)
    with pytest.raises(ValueError, match="more than once"):
        _build(tuple(scene), chunk_point_count=1)


@pytest.mark.parametrize(
    "corruption",
    ["offset_start", "offset_end", "empty_cluster", "readiness", "vegetation_id"],
)
def test_membership_offset_readiness_and_assignment_corruption(corruption: str) -> None:
    if corruption == "empty_cluster":
        scene = list(
            _pipeline(
                xy=[(0.1, 0.1), (1.1, 0.1)],
                hag=[2.0, 2.0],
                cluster_ids=[3, 7],
            )
        )
        object.__setattr__(
            scene[0],
            "cluster_member_offsets",
            _frozen_array([0, 0, 2], np.int64),
        )
    else:
        scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
        if corruption == "offset_start":
            object.__setattr__(
                scene[0],
                "cluster_member_offsets",
                _frozen_array([1, 1], np.int64),
            )
        elif corruption == "offset_end":
            object.__setattr__(
                scene[0],
                "cluster_member_offsets",
                _frozen_array([0, 0], np.int64),
            )
        elif corruption == "readiness":
            object.__setattr__(
                scene[0].vegetation_points,
                "distance_ready_by_point",
                _frozen_array([False], np.bool_),
            )
        else:
            object.__setattr__(
                scene[0],
                "cluster_vegetation_id",
                _frozen_array([-1], np.int32),
            )
    with pytest.raises(ValueError):
        _build(tuple(scene))


@pytest.mark.parametrize("field", ["original_index", "sample_index", "xyz_local"])
def test_source_row_alignment_corruption(field: str) -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    if field == "xyz_local":
        object.__setattr__(
            scene[0].geometry_features,
            field,
            _frozen_array([[0.2, 0.1, 100.0]], np.float32),
        )
    else:
        object.__setattr__(
            scene[0].heights_above_ground,
            field,
            _frozen_array([999], np.int64),
        )
    with pytest.raises(TreeResultConstructionError) as caught:
        _build(tuple(scene))
    assert caught.value.code is TreeResultConstructionErrorCode.INPUT_BUNDLE_MISMATCH


@pytest.mark.parametrize(
    ("owner_index", "field"),
    [
        (0, "identity"),
        (1, "source_identity"),
        (2, "source_identity"),
        (3, "source_identity"),
    ],
)
def test_each_identity_point_count_must_equal_actual_p(owner_index: int, field: str) -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    owner = scene[owner_index]
    identity = getattr(owner, field)
    object.__setattr__(owner, field, _identity_with_point_count(identity, 2))
    with pytest.raises(ValueError, match="actual source point count"):
        _build(tuple(scene))


def test_jointly_wrong_four_way_identity_point_count_is_rejected() -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    wrong_identity = _identity_with_point_count(scene[0].identity, 2)
    object.__setattr__(scene[0], "identity", wrong_identity)
    for owner in scene[1:]:
        object.__setattr__(owner, "source_identity", wrong_identity)
    with pytest.raises(ValueError, match="actual source point count"):
        _build(tuple(scene))


@pytest.mark.parametrize(
    ("field", "value", "dtype", "candidate_field"),
    [
        ("point_count_by_cell", [[2]], np.int32, "source_cell_point_count"),
        ("canopy_height_m", [[3.0]], np.float32, "peak_height_m"),
        ("occupied_mask", [[False]], np.bool_, None),
        ("support_flags_by_cell", [[2]], np.uint8, "source_support_flags"),
        ("peak_support_flags_by_cell", [[2]], np.uint8, "source_peak_support_flags"),
    ],
)
def test_full_chm_raster_audit_detects_independent_corruption(
    field: str,
    value: Any,
    dtype: Any,
    candidate_field: str | None,
) -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    object.__setattr__(scene[1], field, _frozen_array(value, dtype))
    if candidate_field is not None:
        object.__setattr__(
            scene[2],
            candidate_field,
            _frozen_array(np.asarray(value).reshape(-1), dtype),
        )
    with pytest.raises((TreeResultConstructionError, ValueError)):
        _build(tuple(scene))


@pytest.mark.parametrize(
    "corruption",
    ["candidate_id", "candidate_cell", "marker", "region_label", "domain", "region_count"],
)
def test_candidate_and_segmentation_bundle_corruption(corruption: str) -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    if corruption == "candidate_id":
        object.__setattr__(scene[2], "candidate_id", _frozen_array([1], np.int32))
    elif corruption == "candidate_cell":
        object.__setattr__(scene[2], "row_index", _frozen_array([1], np.int64))
    elif corruption == "marker":
        object.__setattr__(
            scene[3],
            "marker_flat_cell_index_by_candidate",
            _frozen_array([1], np.int64),
        )
    elif corruption == "region_label":
        object.__setattr__(
            scene[3],
            "region_label_by_cell",
            _frozen_array([[2]], np.int32),
        )
    elif corruption == "domain":
        object.__setattr__(
            scene[3],
            "segmentation_domain_mask",
            _frozen_array([[False]], np.bool_),
        )
    else:
        object.__setattr__(
            scene[3],
            "region_cell_count_by_candidate",
            _frozen_array([2], np.int64),
        )
    with pytest.raises((TreeResultConstructionError, ValueError)):
        _build(tuple(scene))


def test_identity_and_geometry_mismatches_are_typed() -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    identity = scene[2].source_identity
    values = {name: getattr(identity, name) for name in identity.__dataclass_fields__}
    values["source_id"] = "other"
    object.__setattr__(scene[2], "source_identity", type(identity)(**values))
    with pytest.raises(TreeResultConstructionError) as caught:
        _build(tuple(scene))
    assert caught.value.code is TreeResultConstructionErrorCode.INPUT_BUNDLE_MISMATCH

    scene = list(_pipeline(xy=[(0.1, 0.1)], hag=[2.0]))
    for owner, name in (
        (scene[1], "grid_origin_xy"),
        (scene[2], "source_grid_origin_xy"),
        (scene[3], "source_grid_origin_xy"),
    ):
        object.__setattr__(owner, name, (1.0, 0.0))
    with pytest.raises(TreeResultConstructionError) as caught:
        _build(tuple(scene))
    assert caught.value.code is TreeResultConstructionErrorCode.POINT_MAPPING_MISMATCH


def test_resource_cap_precedes_content_scan() -> None:
    scene = list(_pipeline(xy=[(0.1, 0.1), (0.2, 0.2)], hag=[2.0, 1.0]))
    xyz = np.array(scene[0].vegetation_points.xyz_local, copy=True)
    xyz[0, 0] = np.nan
    xyz.setflags(write=False)
    object.__setattr__(scene[0].vegetation_points, "xyz_local", xyz)
    with pytest.raises(TreeResultConstructionError) as caught:
        _build(tuple(scene), maximum_source_points=1)
    assert caught.value.code is TreeResultConstructionErrorCode.RESOURCE_LIMIT
    assert caught.value.source_point_count == 2


def test_cap_failure_precedes_target_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = _pipeline(xy=[(0.1, 0.1), (0.2, 0.2)], hag=[2.0, 1.0])
    calls: list[Any] = []
    original = construction.np.zeros

    def spy(*args: Any, **kwargs: Any) -> np.ndarray:
        calls.append(args[0] if args else kwargs.get("shape"))
        return original(*args, **kwargs)

    monkeypatch.setattr(construction.np, "zeros", spy)
    with pytest.raises(TreeResultConstructionError):
        _build(scene, maximum_source_points=1)
    assert calls == []


def _old_formal_constructor_peak(
    *,
    p: int,
    k: int,
    assigned: int,
    incidence: int,
) -> int:
    fixed = 1_048_576
    q = min(65_536, max(p, assigned, k + 1))
    reserve = 2_048 * k + 96 * incidence
    return fixed + 9 * p + 8 * assigned + 8 * (k + 1) + reserve + 32 * k + 32 * q


def test_formal_constructor_peak_tracks_every_live_array_dimension() -> None:
    values = {
        "p": 100,
        "n": 50,
        "k": 10,
        "c": 20,
        "allowed_count": 20,
        "b": 10,
        "assigned": 50,
        "incidence": 30,
    }
    base = construction._phase_peaks(**values)["formal_constructor"]
    expected_deltas = {
        "c": 9,
        "k": 2_104,
        "incidence": 102,
        "p": 41,
        "assigned": 8,
    }
    for name, expected_delta in expected_deltas.items():
        changed = dict(values)
        changed[name] += 1
        peak = construction._phase_peaks(**changed)["formal_constructor"]
        assert peak - base == expected_delta


def test_old_estimate_gap_is_rejected_by_corrected_formal_phase() -> None:
    values = {
        "p": 100,
        "n": 50,
        "k": 10,
        "c": 20,
        "allowed_count": 20,
        "b": 10,
        "assigned": 50,
        "incidence": 30,
    }
    peaks = construction._phase_peaks(**values)
    corrected = peaks["formal_constructor"]
    old = _old_formal_constructor_peak(
        p=values["p"],
        k=values["k"],
        assigned=values["assigned"],
        incidence=values["incidence"],
    )
    limit = old + (corrected - old) // 2
    assert old < limit < corrected
    config = TreeResultConstructionConfig(maximum_additional_bytes=limit)
    with pytest.raises(TreeResultConstructionError) as caught:
        construction._check_phase(
            "formal_constructor",
            peaks,
            config,
            p=values["p"],
            n=values["n"],
            k=values["k"],
            c=values["c"],
            assigned=values["assigned"],
        )
    assert caught.value.code is TreeResultConstructionErrorCode.RESOURCE_LIMIT


def test_corrected_formal_phase_exact_byte_boundary() -> None:
    values = {
        "p": 100,
        "n": 50,
        "k": 10,
        "c": 20,
        "allowed_count": 20,
        "b": 10,
        "assigned": 50,
        "incidence": 30,
    }
    peaks = construction._phase_peaks(**values)
    corrected = peaks["formal_constructor"]
    construction._check_phase(
        "formal_constructor",
        peaks,
        TreeResultConstructionConfig(maximum_additional_bytes=corrected),
        p=values["p"],
        n=values["n"],
        k=values["k"],
        c=values["c"],
        assigned=values["assigned"],
    )
    with pytest.raises(TreeResultConstructionError) as caught:
        construction._check_phase(
            "formal_constructor",
            peaks,
            TreeResultConstructionConfig(maximum_additional_bytes=corrected - 1),
            p=values["p"],
            n=values["n"],
            k=values["k"],
            c=values["c"],
            assigned=values["assigned"],
        )
    assert caught.value.code is TreeResultConstructionErrorCode.RESOURCE_LIMIT


def test_representative_corrected_peak_and_default_admission() -> None:
    peaks = construction._phase_peaks(
        p=10_000_000,
        n=10_000_000,
        k=100_000,
        c=1_000_000,
        allowed_count=1_000_000,
        b=250_000,
        assigned=10_000_000,
        incidence=10_000_000,
    )
    assert peaks["formal_constructor"] == 1_412_545_744
    assert max(peaks.values()) == 1_412_545_744
    assert max(peaks.values()) < TreeResultConstructionConfig().maximum_additional_bytes


def test_formal_phase_check_precedes_formal_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _pipeline(xy=[(0.1, 0.1)], hag=[2.0])
    operations: list[str] = []
    original_check = construction._check_phase
    original_constructor = construction.TreeSegmentationResult

    def checked(*args: Any, **kwargs: Any) -> None:
        if args[0] == "formal_constructor":
            operations.append("formal_check")
        original_check(*args, **kwargs)

    def constructor_spy(*args: Any, **kwargs: Any) -> Any:
        operations.append("constructor")
        return original_constructor(*args, **kwargs)

    monkeypatch.setattr(construction, "_check_phase", checked)
    monkeypatch.setattr(construction, "TreeSegmentationResult", constructor_spy)
    _build(scene)
    assert operations == ["formal_check", "constructor"]


def test_equal_chm_total_with_wrong_cell_distribution_is_rejected() -> None:
    scene = list(
        _pipeline(
            xy=[(0.1, 0.1), (0.2, 0.2), (1.1, 0.1)],
            hag=[3.0, 2.0, 4.0],
        )
    )
    wrong_counts = _frozen_array([[1, 2]], np.int32)
    object.__setattr__(scene[1], "point_count_by_cell", wrong_counts)
    rows = scene[2].row_index
    columns = scene[2].column_index
    object.__setattr__(
        scene[2],
        "source_cell_point_count",
        _frozen_array(wrong_counts[rows, columns], np.int32),
    )
    with pytest.raises(TreeResultConstructionError) as caught:
        _build(tuple(scene))
    assert caught.value.code is TreeResultConstructionErrorCode.POINT_MAPPING_MISMATCH


def test_one_source_row_duplicated_across_formal_clusters_is_rejected() -> None:
    scene = list(
        _pipeline(
            xy=[(0.1, 0.1), (1.1, 0.1)],
            hag=[2.0, 2.0],
            cluster_ids=[3, 7],
        )
    )
    object.__setattr__(
        scene[0],
        "ordered_cluster_member_row_index",
        _frozen_array([0, 0], np.int64),
    )
    with pytest.raises(ValueError, match="more than once"):
        _build(tuple(scene), chunk_point_count=1)


def test_complex_stamp_reset_and_cross_chunk_incidence() -> None:
    result = _build(
        _pipeline(
            xy=[(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)],
            hag=[4.0, 3.0, 2.0, 1.0],
            cluster_ids=[5, 5, 3, 7],
        ),
        chunk_point_count=1,
    )
    assert result.records[0].source_cluster_ids == (3, 5, 7)
    assert result.ordered_tree_member_row_index.tolist() == [0, 1, 2, 3]
    assert result.input_provenance["mapping_audit"]["cross_cluster_record_count"] == 1


def test_maximum_additional_bytes_uses_conservative_ledger() -> None:
    scene = _pipeline(xy=[(0.1, 0.1)], hag=[2.0])
    with pytest.raises(TreeResultConstructionError) as caught:
        _build(scene, maximum_additional_bytes=1)
    assert caught.value.code is TreeResultConstructionErrorCode.RESOURCE_LIMIT
    assert caught.value.estimated_additional_bytes is not None
    assert caught.value.estimated_additional_bytes > 1


def test_memory_error_is_translated_and_chained(monkeypatch: pytest.MonkeyPatch) -> None:
    scene = _pipeline(xy=[(0.1, 0.1)], hag=[2.0])

    def fail(*args: Any, **kwargs: Any) -> np.ndarray:
        raise MemoryError("synthetic")

    monkeypatch.setattr(construction.np, "argsort", fail)
    with pytest.raises(TreeResultConstructionError) as caught:
        _build(scene)
    assert caught.value.code is TreeResultConstructionErrorCode.ALLOCATION_FAILED
    assert isinstance(caught.value.__cause__, MemoryError)


def test_statistics_provenance_context_warnings_and_limitations() -> None:
    result = _build(_pipeline(xy=[(-0.1, -0.1)], hag=[2.0]))
    assert result.statistics.to_dict() == {
        "input_point_count": 1,
        "record_count": 1,
        "confirmed_tree_count": 0,
        "review_required_count": 0,
        "provisional_count": 1,
        "rejected_count": 0,
        "unresolved_count": 0,
        "elapsed_seconds": 0.0,
        "phase_timings_seconds": {},
    }
    assert tuple(result.input_provenance) == tuple(
        sorted(
            (
                "source_identity",
                "canopy_height_model",
                "treetop_candidates",
                "canopy_segmentation",
                "result_construction",
                "mapping_audit",
                "id_mapping",
            )
        )
    )
    grid = result.coordinate_context["grid"]
    assert result.coordinate_context["frame"] == "LOCAL_CARTESIAN_XYZ"
    assert grid["origin_xy"] == (-1.0, -1.0)
    assert grid["shape"] == (1, 1)
    assert grid["row_axis"] == "local_y"
    assert grid["column_axis"] == "local_x"
    assert result.warnings == ()
    assert result.limitations == (
        "segments are provisional canopy-region hypotheses and are not confirmed botanical trees",
        "formal tree IDs are unavailable and remain -1",
        "tree geometry and crown summaries are not constructed",
        "excluded and eligible-unassigned source points share segment_id -1",
        "point-level ambiguity is represented only by aggregate audit metadata",
        "tree-to-wire distance and risk outputs are not computed",
    )
