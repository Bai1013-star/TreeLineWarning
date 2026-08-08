"""Decision routing store and FGCCOV1 sort/merge/run-union tests.

Authority: docs/full-resolution-ground-classification-design.md sections 10
and 12, and the frozen Bundle A plan test gate (chunk overlap/gap/duplicate
write; FGCCOV1 ordering/deduplication/connectivity and component-width
boundaries; completion-manifest discovery; orphan LAS non-authority).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from algorithms.ground.decision_store import (
    CoverageRecordAccumulator,
    DecisionConservationError,
    DecisionStore,
    compute_coverage_result,
    merge_coverage_cells,
    run_union_components,
)
from core.contracts.ground_classification import (
    FGC_COVERAGE_FLAG_CONFIRMED,
    FGC_COVERAGE_FLAG_OBSERVED,
    FGC_COVERAGE_RECORD_FORMAT,
    UNROUTED,
    FgcCoverageResult,
    GroundClassificationPolicyConfig,
    GroundDecision,
    pack_coverage_record,
    unpack_coverage_record,
)


def _policy(**overrides) -> GroundClassificationPolicyConfig:
    payload = {**GroundClassificationPolicyConfig().to_dict(), **overrides}
    payload["policy_digest"] = ""  # recompute against the overridden fields
    return GroundClassificationPolicyConfig.from_dict(payload)


# ---------------------------------------------------------------------------
# Decision routing store
# ---------------------------------------------------------------------------


def test_store_initializes_unrouted_and_conserves(tmp_path: Path) -> None:
    store = DecisionStore(10, scratch_dir=tmp_path, policy=_policy())
    assert all(store.peek(i) == GroundDecision.UNROUTED for i in range(10))
    counts = store.total_state_counts()
    assert counts["unrouted"] == 10
    store.route(0, GroundDecision.CONFIRMED_GROUND)
    store.route(1, GroundDecision.NON_GROUND)
    store.route(2, GroundDecision.REVIEW_GROUND)
    with pytest.raises(DecisionConservationError):
        store.assert_no_unrouted()
    for i in range(3, 10):
        store.route(i, GroundDecision.NON_GROUND)
    store.assert_no_unrouted()
    counts = store.total_state_counts()
    assert counts["confirmed_ground"] == 1
    assert counts["non_ground"] == 8
    assert counts["review_ground"] == 1
    assert counts["unrouted"] == 0
    store.cleanup()


def test_store_rejects_duplicate_write(tmp_path: Path) -> None:
    store = DecisionStore(5, scratch_dir=tmp_path, policy=_policy())
    store.route(3, GroundDecision.CONFIRMED_GROUND)
    with pytest.raises(DecisionConservationError, match="duplicate"):
        store.route(3, GroundDecision.REVIEW_GROUND)
    store.cleanup()


def test_store_rejects_out_of_range_and_nonterminal(tmp_path: Path) -> None:
    store = DecisionStore(5, scratch_dir=tmp_path, policy=_policy())
    with pytest.raises(DecisionConservationError):
        store.route(5, GroundDecision.NON_GROUND)
    with pytest.raises(DecisionConservationError):
        store.route(-1, GroundDecision.NON_GROUND)
    with pytest.raises(DecisionConservationError):
        store.route(0, GroundDecision.UNROUTED)
    store.cleanup()


def test_chunks_are_contiguous_and_verified(tmp_path: Path) -> None:
    store = DecisionStore(5, scratch_dir=tmp_path, policy=_policy(decision_chunk_rows=2))
    for i in range(5):
        store.route(i, GroundDecision.CONFIRMED_GROUND)
    chunks = store.write_chunks(tmp_path / "decisions")
    assert [c.start_index for c in chunks] == [0, 2, 4]
    assert [c.end_index for c in chunks] == [2, 4, 5]
    for chunk in chunks:
        assert chunk.size_bytes == chunk.end_index - chunk.start_index
    store.stream_verify(tmp_path / "decisions", chunks)
    store.cleanup()


def test_chunk_state_counts_match_bytes(tmp_path: Path) -> None:
    store = DecisionStore(4, scratch_dir=tmp_path, policy=_policy(decision_chunk_rows=4))
    store.route(0, GroundDecision.CONFIRMED_GROUND)
    store.route(1, GroundDecision.NON_GROUND)
    store.route(2, GroundDecision.REVIEW_GROUND)
    store.route(3, GroundDecision.CONFIRMED_GROUND)
    chunks = store.write_chunks(tmp_path / "decisions")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.state_counts == {
        "non_ground": 1,
        "confirmed_ground": 2,
        "review_ground": 1,
    }
    store.cleanup()


def test_duplicate_write_blocks_chunk_publication(tmp_path: Path) -> None:
    store = DecisionStore(3, scratch_dir=tmp_path, policy=_policy())
    store.route(0, GroundDecision.NON_GROUND)
    store.route(1, GroundDecision.NON_GROUND)
    # Row 2 left UNROUTED -> publication must fail.
    with pytest.raises(DecisionConservationError, match="UNROUTED"):
        store.write_chunks(tmp_path / "decisions")
    store.cleanup()


def test_unrouted_initialization_is_scratch_only(tmp_path: Path) -> None:
    store = DecisionStore(2, scratch_dir=tmp_path, policy=_policy())
    assert store.peek(1) == GroundDecision.UNROUTED
    assert UNROUTED == 255
    store.cleanup()


# ---------------------------------------------------------------------------
# FGCCOV1 external sort / merge / dedup
# ---------------------------------------------------------------------------


def _write_runs(
    tmp_path: Path, rows: list[tuple[int, int, int]], *, run_bytes: int = 64
) -> list[Path]:
    accumulator = CoverageRecordAccumulator(run_dir=tmp_path, run_bytes=run_bytes)
    for iy, ix, flags in rows:
        accumulator.append(ix, iy, flags)
    return accumulator.finish()


def test_run_sort_orders_by_iy_ix_flags(tmp_path: Path) -> None:
    paths = _write_runs(tmp_path, [(0, 2, 1), (1, 0, 1), (0, 1, 3)])
    merged = list(merge_coverage_cells(paths))
    assert merged == [
        (0, 1, 3),
        (0, 2, 1),
        (1, 0, 1),
    ]


def test_merge_or_deduplicates_by_iy_ix(tmp_path: Path) -> None:
    accumulator = CoverageRecordAccumulator(run_dir=tmp_path, run_bytes=16)
    # Two runs each containing duplicate (iy=1, ix=5).
    accumulator.append(5, 1, FGC_COVERAGE_FLAG_OBSERVED)
    accumulator.append(5, 1, FGC_COVERAGE_FLAG_CONFIRMED)
    accumulator.append(6, 1, FGC_COVERAGE_FLAG_OBSERVED)
    paths = accumulator.finish()
    merged = list(merge_coverage_cells(paths))
    assert merged == [
        (1, 5, FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED),
        (1, 6, FGC_COVERAGE_FLAG_OBSERVED),
    ]


def test_merge_multiple_runs_kway(tmp_path: Path) -> None:
    accumulator = CoverageRecordAccumulator(run_dir=tmp_path, run_bytes=8)
    for iy, ix, flags in [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 2, 1),
        (1, 2, 1),
    ]:
        accumulator.append(ix, iy, flags)
    paths = accumulator.finish()
    assert len(paths) > 1
    merged = list(merge_coverage_cells(paths))
    assert merged == [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1), (1, 1, 1), (1, 2, 1)]


def test_coverage_record_roundtrip() -> None:
    record = pack_coverage_record(-3, 7, 0x02)
    assert record == struct.pack(FGC_COVERAGE_RECORD_FORMAT, -3, 7, 0x02)
    assert unpack_coverage_record(record) == (-3, 7, 0x02)


# ---------------------------------------------------------------------------
# Run-union component summaries
# ---------------------------------------------------------------------------


def _observed(ix: int) -> int:
    return FGC_COVERAGE_FLAG_OBSERVED


def _confirmed(ix: int) -> int:
    return FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED


def _sorted(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return sorted(cells, key=lambda item: (item[0], item[1]))


def test_single_interior_component_width(tmp_path: Path) -> None:
    # One unsupported cell at (0,0) with all four cardinal neighbours
    # confirmed (observed) cells, so the unsupported component is interior.
    cells = _sorted(
        [
            (0, -1, _confirmed(-1)),
            (0, 1, _confirmed(1)),
            (-1, 0, _confirmed(-1)),
            (1, 0, _confirmed(1)),
            (0, 0, _observed(0)),  # the unsupported interior cell
        ]
    )
    components = run_union_components(cells)
    assert len(components) == 1
    component = components[0]
    assert component.is_edge is False
    assert component.width_cells() == 1


def test_edge_component_is_edge(tmp_path: Path) -> None:
    # Cell (0,0) is unsupported with confirmed east and north neighbours;
    # west and south are absent from the observed domain, so it is edge.
    cells = _sorted(
        [
            (0, 0, _observed(0)),
            (0, 1, _confirmed(1)),
            (1, 0, _confirmed(1)),
        ]
    )
    components = run_union_components(cells)
    assert len(components) == 1
    assert components[0].is_edge is True


def test_two_components_are_separate(tmp_path: Path) -> None:
    cells = [
        (0, 0, _observed(0)),
        (0, 2, _observed(2)),
        (1, 0, _observed(0)),
        (1, 2, _observed(2)),
    ]
    components = run_union_components(cells)
    assert len(components) == 2


def test_vertical_connection_via_overlapping_x_intervals(tmp_path: Path) -> None:
    # Two unsupported cells vertically adjacent (4-connected).
    cells = [
        (0, 0, _observed(0)),
        (0, 1, _observed(1)),
        (1, 0, _observed(0)),
        (1, 1, _observed(1)),
    ]
    components = run_union_components(cells)
    assert len(components) == 1


def test_diagonal_only_does_not_connect(tmp_path: Path) -> None:
    # Cells at (0,0) and (1,1) are only diagonally adjacent (not 4-connected).
    cells = [
        (0, 0, _observed(0)),
        (1, 1, _observed(1)),
    ]
    components = run_union_components(cells)
    assert len(components) == 2


def test_compute_coverage_result_counts_and_gate(tmp_path: Path) -> None:
    policy = _policy()
    accumulator = CoverageRecordAccumulator(run_dir=tmp_path, run_bytes=16)
    # 10 observed cells, 2 confirmed.
    for ix in range(10):
        accumulator.append(ix, 0, FGC_COVERAGE_FLAG_OBSERVED)
    accumulator.append(0, 0, FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED)
    accumulator.append(1, 0, FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED)
    paths = accumulator.finish()
    result = compute_coverage_result(paths, policy=policy)
    assert result.observed_cell_count == 10
    assert result.confirmed_occupied_cell_count == 2
    assert result.occupancy_fraction == 0.20
    assert result.passes_occupancy(policy=policy)


def test_compute_coverage_result_interior_hole_fails(tmp_path: Path) -> None:
    policy = _policy()
    accumulator = CoverageRecordAccumulator(run_dir=tmp_path, run_bytes=16)
    # A 30 m wide interior unsupported band in row 1 (width > 25 m) fails
    # publication.  The band is flanked on both sides and above/below by
    # observed cells so it is an interior component, not an edge component.
    width = 30
    for ix in range(1, 2 + width):
        accumulator.append(ix, 0, FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED)
        accumulator.append(ix, 2, FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED)
    for ix in range(2, 2 + width):
        accumulator.append(ix, 1, FGC_COVERAGE_FLAG_OBSERVED)
    # Flank the unsupported band with confirmed cells so it is interior.
    accumulator.append(1, 1, FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED)
    accumulator.append(2 + width, 1, FGC_COVERAGE_FLAG_OBSERVED | FGC_COVERAGE_FLAG_CONFIRMED)
    paths = accumulator.finish()
    result = compute_coverage_result(paths, policy=policy)
    over = result.interior_overwidth_components(policy=policy)
    assert len(over) == 1
    assert over[0].width_cells() == width
    assert over[0].is_edge is False


def test_coverage_result_serialization() -> None:
    result = FgcCoverageResult(
        observed_cell_count=5,
        confirmed_occupied_cell_count=1,
        occupancy_fraction=0.2,
        unsupported_components=(),
    )
    restored = FgcCoverageResult.from_dict(result.to_dict())
    assert restored == result
