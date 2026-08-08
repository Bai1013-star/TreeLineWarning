"""FRV-VIS-A1 visualization module tests (PLAN-FROZEN)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import core.contracts.frv as frv
import numpy as np
import pytest
from algorithms.vegetation import build_v4_input_handoff
from algorithms.vegetation import frv_v4_visualization as vis
from core.contracts.frv_membership import (
    FrvMembershipExecutionResult,
    FrvMembershipExecutionState,
)

from tests.vegetation.test_frv_membership import (
    run_membership_ready,
    run_provider_chain,
)

_NONZERO_ORIGIN = (500000.0, 4200000.0, 125.0)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ready(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("vis_ready"))
    packed = run_membership_ready(root)
    result, las_path, _col, _ground, reference, source_result, configuration, *_ = packed
    v4_root = root / "v4out"
    build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=v4_root,
        source_chunk_size=40,
    )
    return result, las_path, v4_root


@pytest.fixture(scope="module")
def ready_origin(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("vis_ready_origin"))
    packed = run_membership_ready(root, offsets=_NONZERO_ORIGIN)
    result, las_path, _col, _ground, reference, source_result, configuration, *_ = packed
    v4_root = root / "v4out_origin"
    build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=v4_root,
        source_chunk_size=40,
    )
    return result, las_path, v4_root


def _copy_roots(frv_root: Path, v4_root: Path, tmp_path: Path) -> tuple[Path, Path]:
    frv_copy = tmp_path / "frv"
    v4_copy = tmp_path / "v4"
    shutil.copytree(frv_root, frv_copy)
    shutil.copytree(v4_root, v4_copy)
    return frv_copy, v4_copy


def _run_vis(
    frv_root: Path,
    v4_root: Path,
    las: Path,
    tmp_path: Path,
    **kwargs,
) -> vis.FrvV4VisualizationResult:
    return vis.run_frv_v4_visualization(
        source=las,
        frv_artifact_root=frv_root,
        v4_input_root=v4_root,
        output_html=tmp_path / "out.html",
        summary_json=tmp_path / "out.json",
        review_checklist=tmp_path / "out.md",
        **kwargs,
    )


def _run_ready(ready, tmp_path: Path, **kwargs) -> vis.FrvV4VisualizationResult:
    result, las_path, v4_root = ready
    return _run_vis(
        result.output_reference.root_path,
        v4_root,
        las_path,
        tmp_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Synthetic layer helpers
# ---------------------------------------------------------------------------


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _make_layer_source(
    name: str,
    indices: list[int],
    *,
    membership_state: int = 1,
    reason: int = 0,
    quality: int = 0,
    wire_relation: int = 0,
    clamped: float = 2.0,
    raw: float = 2.0,
    priority: bool = False,
) -> vis._LayerSource:
    n = len(indices)
    return vis._LayerSource(
        name=name,
        original_index=np.asarray(indices, dtype=np.int64),
        membership_state=np.full(n, membership_state, dtype=np.uint8),
        hag_evidence_state=np.full(n, 1, dtype=np.uint8),
        hag_readiness=np.full(n, 1, dtype=np.uint8),
        reason_flags=np.full(n, reason, dtype=np.uint32),
        quality_flags=np.full(n, quality, dtype=np.uint32),
        raw_hag=np.full(n, raw, dtype=np.float64),
        clamped_hag=np.full(n, clamped, dtype=np.float64),
        wire_relation_state=np.full(n, wire_relation, dtype=np.uint8),
        nearest_wire_id=np.full(n, -1, dtype=np.int32),
        centerline_distance_m=np.zeros(n, dtype=np.float64),
        priority_flagged=np.full(n, priority, dtype=bool),
        xyz=np.zeros((n, 3), dtype=np.float64),
    )


def _alloc(
    layers: dict[str, vis._LayerSource],
    *,
    source_count: int,
    total: int = 250000,
    layer_cap: int = 100000,
    background: int = 50000,
    seed: int = 0,
) -> vis.DisplayAllocation:
    full = {
        name: layers.get(name, _make_layer_source(name, []))
        for name in vis._MEASURED_LAYERS
    }
    return vis.allocate_display_budget(
        full,
        source_point_count=source_count,
        budget=vis.validate_display_budget(total, layer_cap, background, seed),
    )


# ---------------------------------------------------------------------------
# Budget validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "layer", "background"),
    [
        (0, 100000, 50000),
        (500001, 100000, 50000),
        (250000, 0, 50000),
        (250000, 100001, 50000),
        (250000, 100000, 0),
        (250000, 100000, 100001),
        (True, 100000, 50000),
        (250000, "100000", 50000),
    ],
)
def test_invalid_display_budget_rejected(total, layer, background) -> None:
    with pytest.raises(ValueError):
        vis.validate_display_budget(total, layer, background)


@pytest.mark.parametrize(
    ("total", "layer", "background"),
    [
        (1, 1, 1),
        (250000, 100000, 50000),
        (500000, 100000, 100000),
    ],
)
def test_valid_display_budget_accepted(total, layer, background) -> None:
    budget = vis.validate_display_budget(total, layer, background)
    assert budget.max_total_points == total
    assert budget.max_points_per_layer == layer
    assert budget.max_background_points == background


def test_sampling_seed_must_be_non_negative_integer() -> None:
    with pytest.raises(ValueError):
        vis.validate_display_budget(250000, 100000, 50000, -1)
    with pytest.raises(ValueError):
        vis.validate_display_budget(250000, 100000, 50000, True)
    with pytest.raises(ValueError):
        vis.validate_display_budget(250000, 100000, 50000, 1.5)


# ---------------------------------------------------------------------------
# Deterministic sampling / allocation
# ---------------------------------------------------------------------------


def test_sampling_deterministic_same_seed() -> None:
    normal = _make_layer_source(
        vis.LAYER_NORMAL, list(range(300000)), clamped=2.0
    )
    layers = {vis.LAYER_NORMAL: normal}
    first = _alloc(layers, source_count=600000, total=10000, seed=7)
    second = _alloc(layers, source_count=600000, total=10000, seed=7)
    assert np.array_equal(
        first.layers[0].displayed_index, second.layers[0].displayed_index
    )


def test_sampling_row_order_independent() -> None:
    """Identical points in different published block row orders sample the same.

    The mapping normalises block row order (ascending original index) before
    the display allocation, so a shuffled block row order never changes the
    displayed ``original_index`` set for identical inputs, seeds and caps.
    """
    payload = _payload_arrays(
        list(range(0, 2000)),
        [1] * 1000 + [3] * 1000,
        readiness=[1] * 1000 + [2] * 1000,
    )
    normal_a = _normal_arrays(list(range(0, 1000)))
    normal_b = _normal_arrays(list(range(0, 1000))[::-1])
    review_a = _review_arrays(list(range(1000, 2000)), membership_state=[3] * 1000)
    review_b = _review_arrays(list(range(1000, 2000))[::-1], membership_state=[3] * 1000)
    layers_a = vis.map_layers(
        normal=normal_a,
        review=review_a,
        payload=payload,
        source_point_count=50000,
    )
    layers_b = vis.map_layers(
        normal=normal_b,
        review=review_b,
        payload=payload,
        source_point_count=50000,
    )
    alloc_a = _alloc(layers_a, source_count=50000, total=100, seed=3)
    alloc_b = _alloc(layers_b, source_count=50000, total=100, seed=3)
    by_a = {result.name: result.displayed_index for result in alloc_a.layers}
    by_b = {result.name: result.displayed_index for result in alloc_b.layers}
    for name in (vis.LAYER_NORMAL, vis.LAYER_MEMBERSHIP_REVIEW):
        assert np.array_equal(by_a[name], by_b[name])


def test_displayed_never_exceeds_total_cap() -> None:
    layers = {
        vis.LAYER_NORMAL: _make_layer_source(vis.LAYER_NORMAL, list(range(200000))),
        vis.LAYER_MEMBERSHIP_REVIEW: _make_layer_source(
            vis.LAYER_MEMBERSHIP_REVIEW, list(range(200000, 300000))
        ),
        vis.LAYER_EXCLUDED: _make_layer_source(
            vis.LAYER_EXCLUDED, list(range(300000, 400000))
        ),
    }
    for total in (1, 100, 250000, 500000):
        allocation = _alloc(layers, source_count=1000000, total=total)
        assert allocation.total_displayed_point_count <= total
        for result in allocation.layers:
            assert result.displayed_count <= result.requested_cap


def test_total_cap_one_priority() -> None:
    layers = {
        vis.LAYER_NORMAL: _make_layer_source(vis.LAYER_NORMAL, list(range(10, 20))),
        vis.LAYER_EXCLUDED: _make_layer_source(
            vis.LAYER_EXCLUDED, list(range(5, 10))
        ),
        vis.LAYER_HAG_REVIEW: _make_layer_source(
            vis.LAYER_HAG_REVIEW, list(range(3, 5))
        ),
        vis.LAYER_MEMBERSHIP_REVIEW: _make_layer_source(
            vis.LAYER_MEMBERSHIP_REVIEW, list(range(0, 3))
        ),
    }
    allocation = _alloc(layers, source_count=100, total=1)
    by_name = {result.name: result for result in allocation.layers}
    assert by_name[vis.LAYER_MEMBERSHIP_REVIEW].displayed_count == 1
    assert by_name[vis.LAYER_HAG_REVIEW].displayed_count == 0
    assert by_name[vis.LAYER_EXCLUDED].displayed_count == 0
    assert by_name[vis.LAYER_NORMAL].displayed_count == 0


def test_priority_allocation_when_over_cap() -> None:
    layers = {
        vis.LAYER_MEMBERSHIP_REVIEW: _make_layer_source(
            vis.LAYER_MEMBERSHIP_REVIEW, list(range(0, 3))
        ),
        vis.LAYER_HAG_REVIEW: _make_layer_source(
            vis.LAYER_HAG_REVIEW, list(range(3, 5))
        ),
        vis.LAYER_EXCLUDED: _make_layer_source(
            vis.LAYER_EXCLUDED, list(range(5, 9))
        ),
        vis.LAYER_NORMAL: _make_layer_source(
            vis.LAYER_NORMAL, list(range(9, 14))
        ),
    }
    allocation = _alloc(layers, source_count=100, total=6)
    by_name = {result.name: result for result in allocation.layers}
    assert by_name[vis.LAYER_MEMBERSHIP_REVIEW].displayed_count == 3
    assert by_name[vis.LAYER_HAG_REVIEW].displayed_count == 2
    assert by_name[vis.LAYER_EXCLUDED].displayed_count == 1
    assert by_name[vis.LAYER_NORMAL].displayed_count == 0


def test_flagged_point_gets_priority_over_excluded() -> None:
    layers = {
        vis.LAYER_NORMAL: _make_layer_source(
            vis.LAYER_NORMAL, [0, 1, 2], priority=True
        ),
        vis.LAYER_EXCLUDED: _make_layer_source(
            vis.LAYER_EXCLUDED, [3, 4, 5, 6]
        ),
        vis.LAYER_MEMBERSHIP_REVIEW: _make_layer_source(
            vis.LAYER_MEMBERSHIP_REVIEW, []
        ),
        vis.LAYER_HAG_REVIEW: _make_layer_source(vis.LAYER_HAG_REVIEW, []),
    }
    allocation = _alloc(layers, source_count=100, total=1)
    by_name = {result.name: result for result in allocation.layers}
    assert by_name[vis.LAYER_NORMAL].displayed_count == 1
    assert by_name[vis.LAYER_EXCLUDED].displayed_count == 0
    # The single displayed normal point must be a priority-flagged point.
    normal_index = by_name[vis.LAYER_NORMAL].displayed_index
    assert normal_index.tolist() == [0]


def test_layer_cap_boundaries() -> None:
    for full_count, cap in ((0, 1), (1, 1), (100000, 100000), (200000, 100000)):
        layers = {
            vis.LAYER_NORMAL: _make_layer_source(
                vis.LAYER_NORMAL, list(range(full_count))
            )
        }
        allocation = _alloc(layers, source_count=500000, total=500000, layer_cap=cap)
        result = allocation.layers[0]
        assert result.full_count == full_count
        assert result.displayed_count == min(full_count, cap)
        assert result.dropped_display_count == max(0, full_count - cap)


def test_empty_normal_and_empty_review() -> None:
    layers = {
        vis.LAYER_NORMAL: _make_layer_source(vis.LAYER_NORMAL, []),
        vis.LAYER_MEMBERSHIP_REVIEW: _make_layer_source(
            vis.LAYER_MEMBERSHIP_REVIEW, list(range(5))
        ),
        vis.LAYER_HAG_REVIEW: _make_layer_source(vis.LAYER_HAG_REVIEW, []),
        vis.LAYER_EXCLUDED: _make_layer_source(vis.LAYER_EXCLUDED, []),
    }
    allocation = _alloc(layers, source_count=100, total=10)
    by_name = {result.name: result for result in allocation.layers}
    assert by_name[vis.LAYER_NORMAL].full_count == 0
    assert by_name[vis.LAYER_NORMAL].displayed_count == 0
    assert by_name[vis.LAYER_HAG_REVIEW].full_count == 0
    assert by_name[vis.LAYER_MEMBERSHIP_REVIEW].displayed_count == 5


def test_full_stats_unchanged_by_sampling() -> None:
    layers = {
        vis.LAYER_NORMAL: _make_layer_source(vis.LAYER_NORMAL, list(range(200000))),
        vis.LAYER_MEMBERSHIP_REVIEW: _make_layer_source(
            vis.LAYER_MEMBERSHIP_REVIEW, list(range(200000, 210000))
        ),
    }
    allocation = _alloc(layers, source_count=500000, total=5000)
    by_name = {result.name: result for result in allocation.layers}
    assert by_name[vis.LAYER_NORMAL].full_count == 200000
    assert by_name[vis.LAYER_MEMBERSHIP_REVIEW].full_count == 10000
    # Strict priority: membership review consumes the whole budget before normal.
    assert by_name[vis.LAYER_MEMBERSHIP_REVIEW].displayed_count == 5000
    assert by_name[vis.LAYER_NORMAL].displayed_count == 0


def test_background_sampling_and_cap() -> None:
    layers = {
        vis.LAYER_NORMAL: _make_layer_source(vis.LAYER_NORMAL, list(range(0, 500))),
        vis.LAYER_MEMBERSHIP_REVIEW: _make_layer_source(
            vis.LAYER_MEMBERSHIP_REVIEW, []
        ),
        vis.LAYER_HAG_REVIEW: _make_layer_source(vis.LAYER_HAG_REVIEW, []),
        vis.LAYER_EXCLUDED: _make_layer_source(vis.LAYER_EXCLUDED, []),
    }
    allocation = _alloc(
        layers, source_count=100000, total=1000, background=100
    )
    assert allocation.background.full_count == 99500
    assert allocation.background.displayed_count == 100
    # Background indices are never measured indices.
    measured = set(range(500))
    assert measured.isdisjoint(set(allocation.background.displayed_index.tolist()))
    assert allocation.total_displayed_point_count <= 1000


# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------


def _payload_arrays(
    original_index: list[int],
    membership_state: list[int],
    readiness: list[int] | None = None,
) -> dict[str, np.ndarray]:
    n = len(original_index)
    if readiness is None:
        readiness = [1] * n
    return {
        "original_index": np.asarray(original_index, dtype=np.int64),
        "membership_state": np.asarray(membership_state, dtype=np.uint8),
        "reason_flags": np.zeros(n, dtype=np.uint32),
        "input_quality_flags": np.zeros(n, dtype=np.uint32),
        "hag_evidence_state": np.ones(n, dtype=np.uint8),
        "segmentation_hag_readiness": np.asarray(readiness, dtype=np.uint8),
        "raw_hag": np.full(n, 2.0, dtype=np.float64),
        "clamped_hag": np.full(n, 2.0, dtype=np.float64),
    }


def _normal_arrays(original_index: list[int]) -> dict[str, np.ndarray]:
    n = len(original_index)
    return {
        "original_index": np.asarray(original_index, dtype=np.int64),
        "xyz_local_m": np.zeros((n, 3), dtype=np.float64),
        "hag_m": np.full(n, 2.0, dtype=np.float32),
        "membership_state": np.full(n, 1, dtype=np.uint8),
        "hag_evidence_state": np.ones(n, dtype=np.uint8),
        "segmentation_hag_readiness": np.ones(n, dtype=np.uint8),
        "candidate_reason_flags": np.zeros(n, dtype=np.uint32),
    }


def _review_arrays(
    original_index: list[int],
    membership_state: list[int] | None = None,
    readiness: list[int] | None = None,
    primary_route: list[int] | None = None,
) -> dict[str, np.ndarray]:
    n = len(original_index)
    if membership_state is None:
        membership_state = [3] * n
    if readiness is None:
        readiness = [2] * n
    if primary_route is None:
        primary_route = [2] * n
    return {
        "original_index": np.asarray(original_index, dtype=np.int64),
        "xyz_local_m": np.zeros((n, 3), dtype=np.float64),
        "membership_state": np.asarray(membership_state, dtype=np.uint8),
        "hag_evidence_state": np.ones(n, dtype=np.uint8),
        "segmentation_hag_readiness": np.asarray(readiness, dtype=np.uint8),
        "raw_hag_m": np.full(n, 2.0, dtype=np.float64),
        "clamped_hag_m": np.full(n, 2.0, dtype=np.float64),
        "reason_flags": np.zeros(n, dtype=np.uint32),
        "quality_flags": np.zeros(n, dtype=np.uint32),
        "wire_provider_state": np.zeros(n, dtype=np.uint8),
        "wire_relation_state": np.zeros(n, dtype=np.uint8),
        "nearest_wire_id": np.full(n, -1, dtype=np.int32),
        "centerline_distance_m": np.zeros(n, dtype=np.float64),
        "primary_route": np.asarray(primary_route, dtype=np.uint8),
    }


def test_route_mapping_assigns_correct_layers() -> None:
    # Evaluated payload: 1..6 admitted (1), 7..8 excluded (2), 9 review (3),
    # 10 admitted (1) but not HAG-ready (2).
    payload = _payload_arrays(
        list(range(1, 11)),
        [1, 1, 1, 1, 1, 1, 2, 2, 3, 1],
        readiness=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    )
    normal = _normal_arrays([1, 2, 3, 4, 5, 6])
    # Review split: indices 9 (non-admitted -> membership_review) and
    # 10 (admitted but not HAG ready -> hag_review).
    review = _review_arrays([9, 10], membership_state=[3, 1])
    layers = vis.map_layers(
        normal=normal,
        review=review,
        payload=payload,
        source_point_count=100,
    )
    assert layers[vis.LAYER_NORMAL].original_index.tolist() == [1, 2, 3, 4, 5, 6]
    assert layers[vis.LAYER_MEMBERSHIP_REVIEW].original_index.tolist() == [9]
    assert layers[vis.LAYER_HAG_REVIEW].original_index.tolist() == [10]
    assert layers[vis.LAYER_EXCLUDED].original_index.tolist() == [7, 8]


def test_mapping_mutual_exclusion_and_conservation() -> None:
    # States: indices 0-3 verified, 4-6 excluded, 7-9 review (not ready).
    payload = _payload_arrays(
        list(range(10)),
        [1] * 4 + [2] * 3 + [3] * 3,
        readiness=[1] * 4 + [2] * 6,
    )
    normal = _normal_arrays(list(range(4)))
    review = _review_arrays(list(range(7, 10)), membership_state=[3] * 3)
    layers = vis.map_layers(
        normal=normal, review=review, payload=payload, source_point_count=100
    )
    layer_names = (
        vis.LAYER_NORMAL,
        vis.LAYER_MEMBERSHIP_REVIEW,
        vis.LAYER_HAG_REVIEW,
        vis.LAYER_EXCLUDED,
    )
    sets = {
        name: set(int(v) for v in layers[name].original_index)
        for name in layer_names
    }
    assert sets[vis.LAYER_NORMAL] == {0, 1, 2, 3}
    assert sets[vis.LAYER_EXCLUDED] == {4, 5, 6}
    assert sets[vis.LAYER_MEMBERSHIP_REVIEW] == {7, 8, 9}
    assert sets[vis.LAYER_HAG_REVIEW] == set()
    assert sets[vis.LAYER_NORMAL].isdisjoint(sets[vis.LAYER_MEMBERSHIP_REVIEW])
    assert sets[vis.LAYER_NORMAL].isdisjoint(sets[vis.LAYER_EXCLUDED])
    assert sets[vis.LAYER_MEMBERSHIP_REVIEW].isdisjoint(sets[vis.LAYER_EXCLUDED])
    union = set().union(*sets.values())
    assert union == set(range(10))


def test_mapping_rejects_overlap_and_conservation_break() -> None:
    # Index 1 is formally normal (admitted and HAG-ready) but also appears in
    # the review block: the per-point cross-validation fails closed first.
    payload = _payload_arrays(list(range(5)), [1] * 5)
    normal = _normal_arrays([0, 1, 2])
    review = _review_arrays([1, 3, 4], membership_state=[3, 3, 3])
    with pytest.raises(vis.FrvV4ValidationError, match="actually a normal"):
        vis.map_layers(
            normal=normal, review=review, payload=payload, source_point_count=100
        )


def test_mapping_is_row_order_independent() -> None:
    payload = _payload_arrays(
        list(range(10)),
        [1] * 4 + [2] * 3 + [3] * 3,
        readiness=[1] * 4 + [2] * 6,
    )
    normal = _normal_arrays(list(range(4))[::-1])
    review = _review_arrays(list(range(7, 10))[::-1], membership_state=[3] * 3)
    layers = vis.map_layers(
        normal=normal, review=review, payload=payload, source_point_count=100
    )
    for name in (vis.LAYER_NORMAL, vis.LAYER_MEMBERSHIP_REVIEW, vis.LAYER_EXCLUDED):
        assert layers[name].original_index.tolist() == sorted(
            layers[name].original_index.tolist()
        )


# ---------------------------------------------------------------------------
# Per-point route cross-validation (FRV-VIS-A1-ROUTE-001)
# ---------------------------------------------------------------------------


def test_swap_normal_review_indices_rejected() -> None:
    """Same-count normal/review original_index swap must fail closed."""
    payload = _payload_arrays(
        list(range(10)),
        [1] * 4 + [3] * 3 + [2] * 3,
        readiness=[1] * 4 + [2] * 6,
    )
    # The normal block carries authoritative review indices and vice versa.
    normal = _normal_arrays([4, 5, 6])
    review = _review_arrays([0, 1, 2], membership_state=[1, 1, 1])
    with pytest.raises(vis.FrvV4ValidationError):
        vis.map_layers(
            normal=normal, review=review, payload=payload, source_point_count=100
        )


def test_normal_row_membership_not_admitted_rejected() -> None:
    payload = _payload_arrays(list(range(4)), [1] * 4)
    normal = _normal_arrays([0, 1, 2])
    normal["membership_state"][1] = 3  # a row claims a non-admitted state
    with pytest.raises(vis.FrvV4ValidationError, match="not an admitted state"):
        vis.map_layers(
            normal=normal, review=_review_arrays([]), payload=payload,
            source_point_count=100,
        )


def test_normal_row_hag_not_ready_rejected() -> None:
    payload = _payload_arrays(list(range(4)), [1] * 4)
    normal = _normal_arrays([0, 1, 2])
    normal["segmentation_hag_readiness"][0] = 2  # a row is not formally ready
    with pytest.raises(vis.FrvV4ValidationError, match="not formally ready"):
        vis.map_layers(
            normal=normal, review=_review_arrays([]), payload=payload,
            source_point_count=100,
        )


def test_normal_row_authoritative_route_not_normal_rejected() -> None:
    """A normal row whose authoritative route is not NORMAL must fail closed."""
    payload = _payload_arrays(
        list(range(6)), [1, 1, 2, 2, 1, 1], readiness=[1, 1, 2, 2, 1, 1]
    )
    normal = _normal_arrays([0, 2])  # index 2 is authoritative excluded
    with pytest.raises(vis.FrvV4ValidationError, match="not formally admitted"):
        vis.map_layers(
            normal=normal, review=_review_arrays([]), payload=payload,
            source_point_count=100,
        )


def test_review_row_primary_route_not_measured_rejected() -> None:
    payload = _payload_arrays(
        list(range(6)), [3] * 3 + [1] * 3, readiness=[2] * 6
    )
    review = _review_arrays(
        [0, 1, 2], membership_state=[3, 3, 3], primary_route=[3, 2, 2]
    )
    with pytest.raises(vis.FrvV4ValidationError, match="primary route"):
        vis.map_layers(
            normal=_normal_arrays([]), review=review, payload=payload,
            source_point_count=100,
        )


def test_row_reason_field_disagrees_with_payload_rejected() -> None:
    payload = _payload_arrays(
        list(range(6)), [3] * 3 + [1] * 3, readiness=[2] * 6
    )
    review = _review_arrays([0, 1, 2], membership_state=[3, 3, 3])
    review["reason_flags"][1] = int(frv.FrvReasonFlag.INSUFFICIENT_EVIDENCE)
    with pytest.raises(vis.FrvV4ValidationError, match="reason flags"):
        vis.map_layers(
            normal=_normal_arrays([]), review=review, payload=payload,
            source_point_count=100,
        )


def test_valid_rows_pass_cross_validation() -> None:
    payload = _payload_arrays(
        list(range(10)),
        [1] * 4 + [3] * 3 + [2] * 3,
        readiness=[1] * 4 + [2] * 6,
    )
    normal = _normal_arrays([0, 1, 2, 3])
    review = _review_arrays([4, 5, 6], membership_state=[3] * 3)
    layers = vis.map_layers(
        normal=normal, review=review, payload=payload, source_point_count=100
    )
    assert layers[vis.LAYER_NORMAL].original_index.tolist() == [0, 1, 2, 3]
    assert layers[vis.LAYER_MEMBERSHIP_REVIEW].original_index.tolist() == [4, 5, 6]
    assert layers[vis.LAYER_EXCLUDED].original_index.tolist() == [7, 8, 9]


# ---------------------------------------------------------------------------
# LAS display reader
# ---------------------------------------------------------------------------


def test_las_reader_local_xyz_with_nonzero_origin(ready_origin) -> None:
    _result, las_path, _v4_root = ready_origin
    source_identity = _result.source_identity
    las = vis.read_las_display(
        las_path,
        needed=np.arange(5, dtype=np.int64),
        origin=source_identity.origin_context,
        expected_count=source_identity.source_point_count,
        expected_sha256=source_identity.source_file_sha256,
        expected_bounds_min=source_identity.source_bounds_min_xyz,
        expected_bounds_max=source_identity.source_bounds_max_xyz,
    )
    assert las.extracted_index.tolist() == [0, 1, 2, 3, 4]
    origin = np.asarray(_NONZERO_ORIGIN, dtype=np.float64)
    # Column points sit at absolute (500000+5, 4200000+5, 125+z).
    assert np.allclose(las.xyz_local[0], [5.0, 5.0, 0.0], atol=1e-4)
    # local == absolute - origin, and origin is subtracted exactly once.
    assert np.allclose(
        las.xyz_local[0],
        np.asarray([500000.0 + 5.0, 4200000.0 + 5.0, 125.0 + 0.0]) - origin,
        atol=1e-6,
    )


def test_las_reader_rejects_mismatched_sha(ready) -> None:
    _result, las_path, _v4_root = ready
    with pytest.raises(vis.FrvV4ValidationError, match="SHA-256"):
        vis.read_las_display(
            las_path,
            needed=np.empty(0, dtype=np.int64),
            origin=(0.0, 0.0, 0.0),
            expected_count=_result.source_identity.source_point_count,
            expected_sha256="0" * 64,
            expected_bounds_min=_result.source_identity.source_bounds_min_xyz,
            expected_bounds_max=_result.source_identity.source_bounds_max_xyz,
        )


def test_las_reader_rejects_mismatched_count(ready) -> None:
    _result, las_path, _v4_root = ready
    with pytest.raises(vis.FrvV4ValidationError, match="point count"):
        vis.read_las_display(
            las_path,
            needed=np.empty(0, dtype=np.int64),
            origin=(0.0, 0.0, 0.0),
            expected_count=_result.source_identity.source_point_count + 1,
            expected_sha256=_result.source_identity.source_file_sha256,
            expected_bounds_min=_result.source_identity.source_bounds_min_xyz,
            expected_bounds_max=_result.source_identity.source_bounds_max_xyz,
        )


def test_las_reader_rejects_laz(ready, tmp_path: Path) -> None:
    _result, las_path, _v4_root = ready
    laz_copy = tmp_path / "copy.laz"
    shutil.copyfile(las_path, laz_copy)
    with pytest.raises(vis.FrvV4ValidationError, match=".laz"):
        vis.read_las_display(
            laz_copy,
            needed=np.empty(0, dtype=np.int64),
            origin=(0.0, 0.0, 0.0),
            expected_count=1,
            expected_sha256="0" * 64,
            expected_bounds_min=(0.0, 0.0, 0.0),
            expected_bounds_max=(1.0, 1.0, 1.0),
        )


# ---------------------------------------------------------------------------
# End-to-end ready-mode run
# ---------------------------------------------------------------------------


def test_ready_end_to_end_outputs(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    frv_root = result.output_reference.root_path
    vis_result = _run_vis(frv_root, v4_root, las_path, tmp_path)
    assert vis_result.mode == "ready"
    assert vis_result.html_path.is_file()
    assert vis_result.summary_path.is_file()
    assert vis_result.checklist_path.is_file()
    summary = json.loads(vis_result.summary_path.read_text("utf-8"))
    assert summary["mode"] == "ready"
    assert summary["layers"][vis.LAYER_NORMAL]["full_count"] > 0
    assert summary["layers"][vis.LAYER_MEMBERSHIP_REVIEW]["full_count"] > 0
    assert summary["layers"][vis.LAYER_EXCLUDED]["full_count"] > 0
    assert all(
        value is True for value in summary["identity_validation"].values()
    )


def test_ready_excluded_coordinates_recovered_from_source(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    budget = vis.validate_display_budget(250000, 100000, 50000, 0)
    data = vis.load_frv_v4_input(
        source=las_path,
        frv_artifact_root=result.output_reference.root_path,
        v4_input_root=v4_root,
    )
    allocation = vis.allocate_display_budget(
        data.layers,
        source_point_count=data.source_identity.source_point_count,
        budget=budget,
    )
    excluded_allocation = next(
        r for r in allocation.layers if r.name == vis.LAYER_EXCLUDED
    )
    needed = np.sort(
        np.union1d(
            excluded_allocation.displayed_index,
            allocation.background.displayed_index,
        )
    )
    las = vis.read_las_display(
        las_path,
        needed=needed,
        origin=data.source_identity.origin_context,
        expected_count=data.source_identity.source_point_count,
        expected_sha256=data.source_identity.source_file_sha256,
        expected_bounds_min=data.source_identity.source_bounds_min_xyz,
        expected_bounds_max=data.source_identity.source_bounds_max_xyz,
    )
    import laspy

    with laspy.open(las_path) as reader:
        points = reader.read()
    absolute = np.column_stack(
        (
            np.asarray(points.x, dtype=np.float64),
            np.asarray(points.y, dtype=np.float64),
            np.asarray(points.z, dtype=np.float64),
        )
    )
    origin = np.asarray(data.source_identity.origin_context, dtype=np.float64)
    positions = np.searchsorted(
        las.extracted_index, excluded_allocation.displayed_index
    )
    recovered = las.xyz_local[positions]
    expected = absolute[excluded_allocation.displayed_index] - origin
    assert np.allclose(recovered, expected, atol=1e-6)


def test_identity_mismatch_rejected(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    frv_root, v4_copy = _copy_roots(result.output_reference.root_path, v4_root, tmp_path)
    manifest_path = v4_copy / "handoff_manifest.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["source_content_identity_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(vis.FrvV4ValidationError):
        _run_vis(frv_root, v4_copy, las_path, tmp_path / "out")


def test_ready_manifest_v1_read_compatibility(ready, tmp_path: Path) -> None:
    """FRV-VIS-A1 keeps read compatibility with persisted v1 manifests."""

    result, las_path, v4_root = ready
    frv_root, v4_copy = _copy_roots(result.output_reference.root_path, v4_root, tmp_path)
    manifest_path = v4_copy / "handoff_manifest.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["schema_version"] = "v4-input-handoff-v1"
    for key in (
        "source_identity_record_digest",
        "origin_context",
        "normal_row_count",
        "review_row_count",
        "normal_block_paths",
        "review_block_paths",
        "normal_block_digests",
        "review_block_digests",
        "normal_set_digest",
        "review_set_digest",
        "handoff_digest",
        "manifest_digest",
        "source_proof_digest",
        "hag_proof_digest",
        "geometry_proof_digest",
        "wire_proof_digest",
    ):
        payload.pop(key)
    manifest_path.write_text(json.dumps(payload), "utf-8")
    # v1 has no persisted block digests, so the viewer reads it as before.
    data = _run_vis(frv_root, v4_copy, las_path, tmp_path / "out")
    assert data.mode == "ready"


def test_ready_v2_block_hash_verified(ready, tmp_path: Path) -> None:
    """A v2 manifest verifies every published block SHA-256 on read."""

    result, las_path, v4_root = ready
    _frv, v4_copy = _copy_roots(result.output_reference.root_path, v4_root, tmp_path)
    block = next((v4_copy / "normal").glob("*.v4blk"))
    data = bytearray(block.read_bytes())
    data[-1] ^= 0xFF
    block.write_bytes(bytes(data))
    with pytest.raises(vis.FrvV4ValidationError):
        _run_vis(result.output_reference.root_path, v4_copy, las_path, tmp_path / "out")


def test_corrupt_normal_block_rejected(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    _frv, v4_copy = _copy_roots(result.output_reference.root_path, v4_root, tmp_path)
    block = next((v4_copy / "normal").glob("*.v4blk"))
    block.write_bytes(b"corrupted")
    with pytest.raises(vis.FrvV4ValidationError):
        _run_vis(result.output_reference.root_path, v4_copy, las_path, tmp_path / "out")


def test_corrupt_frv_payload_block_rejected(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    frv_copy, _v4 = _copy_roots(result.output_reference.root_path, v4_root, tmp_path)
    block = next((frv_copy / "blocks").glob("*.frvblk"))
    block.write_bytes(b"corrupted")
    with pytest.raises(vis.FrvV4ValidationError):
        _run_vis(frv_copy, v4_root, las_path, tmp_path / "out")


def test_inputs_not_overwritten_and_digests_unchanged(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    frv_root = result.output_reference.root_path

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = {
        "frv": snapshot(frv_root),
        "v4": snapshot(v4_root),
        "las": las_path.read_bytes(),
    }
    _run_vis(frv_root, v4_root, las_path, tmp_path)
    assert snapshot(frv_root) == before["frv"]
    assert snapshot(v4_root) == before["v4"]
    assert las_path.read_bytes() == before["las"]


def test_no_partial_output_on_failure(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    _frv, v4_copy = _copy_roots(result.output_reference.root_path, v4_root, tmp_path)
    block = next((v4_copy / "normal").glob("*.v4blk"))
    block.write_bytes(b"corrupted")
    out_dir = tmp_path / "nested" / "out"
    with pytest.raises(vis.FrvV4ValidationError):
        vis.run_frv_v4_visualization(
            source=las_path,
            frv_artifact_root=result.output_reference.root_path,
            v4_input_root=v4_copy,
            output_html=out_dir / "out.html",
        )
    assert not (out_dir / "out.html").exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_invalid_cap_produces_no_output(tmp_path: Path) -> None:
    with pytest.raises(vis.FrvV4BudgetError):
        vis.run_frv_v4_visualization(
            source=tmp_path / "missing.las",
            frv_artifact_root=tmp_path / "frv",
            v4_input_root=tmp_path / "v4",
            output_html=tmp_path / "out.html",
            max_total_points=0,
        )
    assert list(tmp_path.glob("*.html")) == []
    assert list(tmp_path.rglob("*.tmp")) == []


def test_output_overwrite_refused(ready, tmp_path: Path) -> None:
    result, las_path, v4_root = ready
    output = tmp_path / "existing.html"
    output.write_text("x", "utf-8")
    with pytest.raises(vis.FrvV4OutputError, match="refusing to overwrite"):
        vis.run_frv_v4_visualization(
            source=las_path,
            frv_artifact_root=result.output_reference.root_path,
            v4_input_root=v4_root,
            output_html=output,
        )


# ---------------------------------------------------------------------------
# HTML renderer and summary
# ---------------------------------------------------------------------------


def test_html_contains_layers_and_hover_fields(ready, tmp_path: Path) -> None:
    vis_result = _run_ready(ready, tmp_path)
    html = vis_result.html_path.read_text("utf-8")
    assert "plotly" in html.lower()
    assert "Normal admitted" in html
    assert "Membership review" in html
    assert "HAG review" in html
    assert "Excluded audit" in html
    assert "Original background" in html
    assert "Unrecovered regions" in html
    assert "Wire review overlay" in html
    assert "Ground low-HAG overlay" in html
    for field in (
        "original_index",
        "clamped_hag",
        "wire_relation",
        "nearest_wire_id",
        "hag_readiness",
    ):
        assert field in html


def test_summary_schema(ready, tmp_path: Path) -> None:
    vis_result = _run_ready(ready, tmp_path)
    summary = vis_result.summary
    assert summary["tool"] == vis.TOOL_NAME
    assert summary["summary_schema_version"] == vis.SUMMARY_SCHEMA_VERSION
    for key in (
        "source",
        "status",
        "identity_validation",
        "routes",
        "layers",
        "regions",
        "display_budget",
        "denominators",
        "outputs",
    ):
        assert key in summary
    budget = summary["display_budget"]
    assert budget["requested_total_cap"] == 250000
    assert budget["hard_total_cap"] == vis.HARD_TOTAL_CAP
    assert budget["sampling_seed"] == 0
    assert budget["allocation_priority"] == list(vis.ALLOCATION_PRIORITY)
    assert budget["total_displayed_point_count"] <= budget["requested_total_cap"]
    denominator = summary["denominators"]["point_ratio_denominator"]
    assert denominator == summary["source"]["source_point_count"]
    assert summary["regions"]["unrecovered_region_count"] == 0
    # Full counts are never changed by display sampling.
    layer = summary["layers"][vis.LAYER_NORMAL]
    assert layer["full_count"] >= layer["displayed_count"]
    assert layer["requested_cap"] == 100000
    assert "dropped_display_count" in layer


def test_summary_overlay_counts_recorded(ready, tmp_path: Path) -> None:
    vis_result = _run_ready(ready, tmp_path)
    summary = vis_result.summary
    for name in (vis.LAYER_WIRE_OVERLAY, vis.LAYER_GROUND_OVERLAY, vis.LAYER_UNRECOVERED):
        assert name in summary["layers"]


def test_checklist_markdown_contains_rows_and_conclusions(ready, tmp_path: Path) -> None:
    vis_result = _run_ready(ready, tmp_path)
    text = vis_result.checklist_path.read_text("utf-8")
    assert "FRV-VIS-A1 Human Acceptance Checklist" in text
    for conclusion in (
        "PASS_READY_FOR_V4",
        "PASS_WITH_REVIEW_NOTES",
        "FRV_RULE_ADJUSTMENT_REQUIRED",
        "DATA_OR_ARTIFACT_BLOCKED",
    ):
        assert conclusion in text
    for item, _focus in vis._CHECKLIST_ROWS:
        assert item in text


# ---------------------------------------------------------------------------
# Blocked mode
# ---------------------------------------------------------------------------


def _blocked_coverage_record(key: str, source_digest: str) -> frv.FRVScopeCoverageRecord:
    payload = {
        "record_schema_version": frv.FRV_SCOPE_COVERAGE_RECORD_SCHEMA_VERSION,
        "source_content_identity_digest": source_digest,
        "canonical_inventory_key": key,
        "canonical_inventory_record_digest": _sha("rec-" + key),
        "canonical_inventory_digest": _sha("inv"),
        "inventory_disposition": int(frv.FrvInventoryDisposition.REQUIRED_FRV),
        "required_inventory_record_digest": _sha("req-" + key),
        "required_inventory_digest": _sha("req"),
        "approved_hard_rejection_evidence": None,
        "upstream_candidate_reference": key,
        "upstream_candidate_type": "v1_point",
        "upstream_status": "1",
        "upstream_evidence_reference": key,
        "scope_id": None,
        "scope_construction_state": int(frv.FrvScopeConstructionState.NOT_CREATED),
        "scope_coverage_state": int(frv.FrvScopeCoverageState.SCOPE_BLOCKED),
        "failure_reason_flags": int(
            frv.FrvReasonFlag.PROVIDER_UNCERTAINTY | frv.FrvReasonFlag.REVIEW_REQUIRED
        ),
        "review_required": True,
        "safety_completeness_effect": "BLOCKS_SAFETY_COMPLETENESS",
        "diagnostic_reference": key,
        "out_of_policy_safety_approval_digest": None,
        "record_digest": "",
    }
    record = frv.FRVScopeCoverageRecord.__new__(frv.FRVScopeCoverageRecord)
    for name, value in payload.items():
        object.__setattr__(record, name, value)
    object.__setattr__(record, "record_digest", record.computed_digest())
    return record


def _blocked_manifest(source_identity) -> frv.FRVScopeCoverageManifest:
    source_digest = source_identity.source_content_identity_digest
    records = [
        _blocked_coverage_record("V1:00000000000000000001", source_digest),
        _blocked_coverage_record("V1:00000000000000000002", source_digest),
    ]
    payload = {
        "manifest_schema_version": frv.FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "source_content_identity_digest": source_digest,
        "configuration_digest": _sha("cfg"),
        "scope_construction_version": "scope-1",
        "canonical_inventory_digest": _sha("inv"),
        "required_inventory_digest": _sha("req"),
        "scope_union_id": None,
        "scope_union_geometry_digest": None,
        "records": [record.to_dict() for record in records],
        "canonical_inventory_count": 2,
        "required_inventory_count": 2,
        "approved_hard_rejection_count": 0,
        "scope_created_or_merged_count": 0,
        "coverage_blocked_count": 2,
        "canonical_inventory_complete": True,
        "required_inventory_complete": True,
        "approved_hard_rejection_complete": True,
        "scope_coverage_complete": False,
        "routing_complete": False,
        "expected_semantic_artifact_id": None,
        "provider_failure_reference_digests": [],
        "stream_failure_reference_digests": [],
        "identity_failure_reference_digests": [],
        "warnings": [],
        "limitations": [],
        "provenance_digest": _sha("prov"),
        "manifest_digest": "",
    }
    digest = frv.frv_digest(
        {key: value for key, value in payload.items() if key != "manifest_digest"}
    )
    payload["manifest_digest"] = digest
    return frv.FRVScopeCoverageManifest.from_dict(payload)


def _blocked_execution_result(source_identity, manifest) -> FrvMembershipExecutionResult:
    result = FrvMembershipExecutionResult.__new__(FrvMembershipExecutionResult)
    object.__setattr__(result, "execution_state", FrvMembershipExecutionState.COVERAGE_BLOCKED)
    object.__setattr__(result, "source_identity", source_identity)
    object.__setattr__(result, "a2_readiness", None)
    object.__setattr__(result, "coverage_manifest", manifest)
    object.__setattr__(result, "membership_artifact", None)
    object.__setattr__(result, "payload_schema", None)
    object.__setattr__(result, "artifact_binding", None)
    object.__setattr__(result, "decision_evidence_artifact", None)
    object.__setattr__(result, "source_proof_digest", _sha("proof"))
    object.__setattr__(result, "hag_proof_digest", None)
    object.__setattr__(result, "geometry_proof_digest", None)
    object.__setattr__(result, "wire_proof_digest", None)
    object.__setattr__(result, "provider_binding_digests", ())
    object.__setattr__(result, "override_digests", ())
    object.__setattr__(result, "failure_records", ())
    object.__setattr__(result, "output_reference", None)
    object.__setattr__(result, "result_digest", result.computed_digest())
    return result


def _write_blocked_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    chain = run_provider_chain(tmp_path)
    las_path, _col, _ground, _reference, _source_config, _configuration, source_result, *_ = chain
    source_identity = source_result.source_identity
    manifest = _blocked_manifest(source_identity)
    result = _blocked_execution_result(source_identity, manifest)
    frv_root = tmp_path / "blocked_frv"
    frv_root.mkdir()
    (frv_root / "execution_result.json").write_text(
        json.dumps(result.to_dict()), "utf-8"
    )
    (frv_root / "coverage_manifest.json").write_text(
        json.dumps(manifest.to_dict()), "utf-8"
    )
    v4_root = tmp_path / "blocked_v4"
    v4_root.mkdir()
    return frv_root, v4_root, las_path


def test_blocked_mode_regions_recorded_never_points(tmp_path: Path) -> None:
    frv_root, v4_root, las_path = _write_blocked_root(tmp_path)
    data = vis.load_frv_v4_input(
        source=las_path, frv_artifact_root=frv_root, v4_input_root=v4_root
    )
    assert data.mode == "blocked"
    assert data.layers == {}
    assert len(data.regions) == 2
    assert all(region.geometry_status == "missing-geometry" for region in data.regions)
    assert all(region.bounds is None for region in data.regions)


def test_blocked_mode_end_to_end_outputs(tmp_path: Path) -> None:
    frv_root, v4_root, las_path = _write_blocked_root(tmp_path)
    vis_result = vis.run_frv_v4_visualization(
        source=las_path,
        frv_artifact_root=frv_root,
        v4_input_root=v4_root,
        output_html=tmp_path / "blocked.html",
        summary_json=tmp_path / "blocked.json",
        review_checklist=tmp_path / "blocked.md",
    )
    assert vis_result.mode == "blocked"
    html = vis_result.html_path.read_text("utf-8")
    assert "V1:00000000000000000001" in html
    assert "plotly" in html.lower()
    summary = vis_result.summary
    assert summary["regions"]["unrecovered_region_count"] == 2
    assert summary["regions"]["region_geometry_status"] == "missing-geometry"
    assert summary["status"]["artifact_status"] == "coverage-blocked"
    assert summary["display_budget"]["total_displayed_point_count"] == 0
    checklist = vis_result.checklist_path.read_text("utf-8")
    assert "DATA_OR_ARTIFACT_BLOCKED" in checklist
