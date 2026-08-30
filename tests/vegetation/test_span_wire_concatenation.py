"""Focused contracts for multi-span W1 wire-array concatenation.

These fixtures are deliberately small and synthetic.  They exercise the
prefix-array boundaries consumed by ``build_wire_edge_bundle`` as well as the
exact clearance result, without relying on a scene or on the wire
reconstruction pipeline.
"""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.spans.clearance import (
    _concatenate_span_wire_edges,
    run_span_tree_wire_clearance,
)
from algorithms.spans.wire import SpanWireResult
from algorithms.vegetation.tree_wire_clearance import (
    WireEdgeBundle,
    build_wire_edge_bundle,
    exact_point_to_segment_distance,
)
from core.config.fast_baseline import FastBaselineConfig


def _span_wire_result(
    sample_counts: tuple[int, ...],
    *,
    segments_per_wire: tuple[int, ...] | None = None,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> SpanWireResult:
    """Build a complete, compact W1-like result for synthetic tests."""

    if segments_per_wire is None:
        segments_per_wire = (1,) * len(sample_counts)
    if sum(segments_per_wire) != len(sample_counts):
        raise ValueError("segments_per_wire must account for every segment")
    if any(samples < 2 for samples in sample_counts):
        raise ValueError("every synthetic distance-ready segment needs two samples")

    wire_count = len(segments_per_wire)
    segment_count = len(sample_counts)
    wire_segment_offsets = [0]
    for segment_count_for_wire in segments_per_wire:
        wire_segment_offsets.append(wire_segment_offsets[-1] + segment_count_for_wire)

    segment_wire_id: list[int] = []
    segment_index: list[int] = []
    parameter_domain: list[tuple[float, float]] = []
    sampled_xyz: list[np.ndarray] = []
    sampled_parameter: list[np.ndarray] = []
    sampled_curve_offsets = [0]
    segment_position = 0
    for wire_id, segment_count_for_wire in enumerate(segments_per_wire):
        for local_segment_index in range(segment_count_for_wire):
            samples = sample_counts[segment_position]
            start_x = x_origin + 10.0 * segment_position
            y = y_origin + 4.0 * wire_id
            curve = np.column_stack(
                (
                    np.linspace(start_x, start_x + 5.0, samples),
                    np.full(samples, y),
                    np.zeros(samples),
                )
            ).astype(np.float32)
            segment_wire_id.append(wire_id)
            segment_index.append(local_segment_index)
            parameter_domain.append((float(segment_position), float(segment_position + 1)))
            sampled_xyz.append(curve)
            sampled_parameter.append(np.linspace(0.0, 1.0, samples, dtype=np.float32))
            sampled_curve_offsets.append(sampled_curve_offsets[-1] + samples)
            segment_position += 1

    if sampled_xyz:
        all_xyz = np.concatenate(sampled_xyz, axis=0)
        all_parameter = np.concatenate(sampled_parameter, axis=0)
    else:
        all_xyz = np.empty((0, 3), dtype=np.float32)
        all_parameter = np.empty(0, dtype=np.float32)

    arrays = {
        "wire_id": np.arange(wire_count, dtype=np.int32),
        "wire_status": np.zeros(wire_count, dtype=np.uint8),
        "wire_method": np.ones(wire_count, dtype=np.uint8),
        "wire_quality_flags": np.zeros(wire_count, dtype=np.uint16),
        "wire_has_distance_ready_segments": np.ones(wire_count, dtype=np.bool_),
        "wire_continuous_distance_ready": np.ones(wire_count, dtype=np.bool_),
        "wire_segment_offsets": np.asarray(wire_segment_offsets, dtype=np.int64),
        "segment_wire_id": np.asarray(segment_wire_id, dtype=np.int32),
        "segment_index": np.asarray(segment_index, dtype=np.int32),
        "segment_method": np.ones(segment_count, dtype=np.uint8),
        "segment_quality_flags": np.zeros(segment_count, dtype=np.uint16),
        "segment_distance_ready": np.ones(segment_count, dtype=np.bool_),
        "segment_parameter_domain": np.asarray(parameter_domain, dtype=np.float64).reshape(
            segment_count, 2
        ),
        "sampled_curve_xyz": all_xyz,
        "sampled_curve_parameter": all_parameter,
        "sampled_curve_offsets": np.asarray(sampled_curve_offsets, dtype=np.int64),
    }
    return SpanWireResult(
        strands=(),
        w1_arrays=arrays,
        candidate_mask=np.empty(0, dtype=bool),
        cluster_id_by_rep=np.empty(0, dtype=np.int32),
        diagnostics={},
    )


def _edge_bundle_fields(bundle: WireEdgeBundle) -> dict[str, object]:
    return {field.name: getattr(bundle, field.name) for field in fields(WireEdgeBundle)}


def test_empty_and_no_input_spans_have_one_global_boundary() -> None:
    empty = _span_wire_result(())

    for wire_results in ([], [empty], [empty, empty]):
        arrays, distance_ready_segments, total_wires = _concatenate_span_wire_edges(
            wire_results
        )
        np.testing.assert_array_equal(arrays["wire_segment_offsets"], [0])
        np.testing.assert_array_equal(arrays["sampled_curve_offsets"], [0])
        assert distance_ready_segments == 0
        assert total_wires == 0
        assert build_wire_edge_bundle(arrays).edge_count == 0


def test_empty_then_valid_and_valid_then_empty_keep_the_73_sample_prefix() -> None:
    empty = _span_wire_result(())
    valid = _span_wire_result((73,))

    for wire_results in ([empty, valid], [valid, empty]):
        arrays, distance_ready_segments, total_wires = _concatenate_span_wire_edges(
            wire_results
        )
        np.testing.assert_array_equal(arrays["wire_segment_offsets"], [0, 1])
        np.testing.assert_array_equal(arrays["sampled_curve_offsets"], [0, 73])
        assert distance_ready_segments == 1
        assert total_wires == 1
        assert build_wire_edge_bundle(arrays).edge_count == 72


def test_two_nonempty_spans_preserve_global_prefix_shapes_and_wire_membership() -> None:
    first = _span_wire_result((2, 3, 4), segments_per_wire=(2, 1))
    second = _span_wire_result((5, 6), segments_per_wire=(2,), x_origin=100.0)

    arrays, distance_ready_segments, total_wires = _concatenate_span_wire_edges(
        [first, second]
    )
    wire_offsets = arrays["wire_segment_offsets"]
    sampled_offsets = arrays["sampled_curve_offsets"]
    segment_wire_id = arrays["segment_wire_id"]
    wire_id = arrays["wire_id"]

    assert len(wire_offsets) == total_wires + 1 == 4
    assert wire_offsets[0] == 0
    assert wire_offsets[-1] == distance_ready_segments == 5
    assert len(sampled_offsets) == distance_ready_segments + 1 == 6
    assert sampled_offsets[0] == 0
    assert sampled_offsets[-1] == 20
    assert np.all(np.diff(wire_offsets) >= 0)
    assert np.all(np.diff(sampled_offsets) >= 0)
    assert np.all(np.isin(segment_wire_id, wire_id))
    for wire_position, (start, stop) in enumerate(
        zip(wire_offsets[:-1], wire_offsets[1:], strict=True)
    ):
        np.testing.assert_array_equal(segment_wire_id[start:stop], wire_id[wire_position])


def test_wire_segment_offsets_use_segment_cursor_when_wire_and_segment_counts_differ() -> None:
    # The first Span has two wires but three segments.  A wire cursor would
    # rebase the second Span at 2; the required segment cursor rebases it at 3.
    first = _span_wire_result((2, 3, 4), segments_per_wire=(2, 1))
    second = _span_wire_result((5, 6), segments_per_wire=(2,), x_origin=100.0)

    arrays, _, _ = _concatenate_span_wire_edges([first, second])
    np.testing.assert_array_equal(arrays["wire_segment_offsets"], [0, 2, 3, 5])
    assert arrays["wire_segment_offsets"][2] == 3
    assert arrays["wire_segment_offsets"][-1] == 5


def test_single_span_concatenation_preserves_arrays_and_edge_reference() -> None:
    single = _span_wire_result((2, 3, 4), segments_per_wire=(2, 1))
    arrays, distance_ready_segments, total_wires = _concatenate_span_wire_edges([single])

    for key, expected in single.w1_arrays.items():
        np.testing.assert_array_equal(arrays[key], expected)
    assert distance_ready_segments == 3
    assert total_wires == 2
    np.testing.assert_equal(
        _edge_bundle_fields(build_wire_edge_bundle(arrays)),
        _edge_bundle_fields(build_wire_edge_bundle(single.w1_arrays)),
    )


def test_nonempty_offsets_are_validated_but_missing_empty_offsets_are_safe() -> None:
    valid = _span_wire_result((4,))
    malformed = replace(
        valid,
        w1_arrays={
            **valid.w1_arrays,
            "sampled_curve_offsets": np.asarray([0, 3], dtype=np.int64),
        },
    )
    with pytest.raises(ValueError, match="sampled_curve_offsets"):
        _concatenate_span_wire_edges([malformed])

    missing = dict(valid.w1_arrays)
    missing.pop("sampled_curve_offsets")
    with pytest.raises(ValueError, match="sampled_curve_offsets"):
        _concatenate_span_wire_edges([replace(valid, w1_arrays=missing)])

    empty = _span_wire_result(())
    empty_missing = dict(empty.w1_arrays)
    empty_missing.pop("wire_segment_offsets")
    empty_missing.pop("sampled_curve_offsets")
    arrays, distance_ready_segments, total_wires = _concatenate_span_wire_edges(
        [replace(empty, w1_arrays=empty_missing)]
    )
    np.testing.assert_array_equal(arrays["wire_segment_offsets"], [0])
    np.testing.assert_array_equal(arrays["sampled_curve_offsets"], [0])
    assert distance_ready_segments == total_wires == 0


@pytest.mark.parametrize(
    "backend", ["PYTHON_BVH", "NUMPY_DIRECT", "SCIPY_CANDIDATE_EXACT"]
)
def test_multispan_clearance_matches_independent_exact_segment_oracle(
    backend: str,
) -> None:
    first = _span_wire_result((4,), y_origin=0.0)
    second = _span_wire_result((5,), y_origin=8.0)
    wire_results = [first, second]
    points = np.asarray(((2.0, 0.5, 0.5), (2.0, 7.0, 1.0)), dtype=np.float64)
    originals = np.asarray((101, 102), dtype=np.int64)

    # Build the oracle from each local bundle independently.  In particular,
    # it does not inspect the concatenated offsets under test.
    oracle_by_original: dict[int, tuple[float, np.ndarray, int, np.ndarray]] = {}
    wire_cursor = 0
    for result in wire_results:
        local_bundle = build_wire_edge_bundle(result.w1_arrays)
        for point, original in zip(points, originals, strict=True):
            candidates: list[tuple[float, np.ndarray, int, np.ndarray]] = []
            for edge_position in range(local_bundle.edge_count):
                distance, nearest, _ = exact_point_to_segment_distance(
                    point,
                    local_bundle.a_xyz[edge_position],
                    local_bundle.b_xyz[edge_position],
                )
                candidates.append(
                    (
                        distance,
                        nearest,
                        wire_cursor + int(local_bundle.wire_id[edge_position]),
                        point,
                    )
                )
            candidate = min(candidates, key=lambda item: item[0])
            previous = oracle_by_original.get(int(original))
            if previous is None or candidate[0] < previous[0]:
                oracle_by_original[int(original)] = candidate
        wire_cursor += int(result.w1_arrays["wire_id"].shape[0])

    reconciled = SimpleNamespace(
        global_tree_records=(
            SimpleNamespace(
                global_tree_id=7,
                external_name="T001",
                line_id=1,
                span_id=1,
                span_name="S001",
                full_resolution_point_count=1,
            ),
            SimpleNamespace(
                global_tree_id=8,
                external_name="T002",
                line_id=1,
                span_id=2,
                span_name="S002",
                full_resolution_point_count=1,
            ),
        ),
        final_tree_id=np.asarray((7, 8), dtype=np.int32),
        final_xyz_local_m=points,
        final_original_index=originals,
    )
    result = run_span_tree_wire_clearance(
        reconciled,
        wire_results,
        FastBaselineConfig(),
        backend=backend,
    )

    assert result.edge_count == sum(
        build_wire_edge_bundle(item.w1_arrays).edge_count for item in wire_results
    )
    assert result.status == "VALID"
    records = {record.global_tree_id: record for record in result.records}
    assert set(records) == {7, 8}
    expected_distances = []
    for tree_id, original in ((7, 101), (8, 102)):
        expected_distance, expected_nearest, expected_wire, expected_point = oracle_by_original[
            original
        ]
        record = records[tree_id]
        expected_distances.append(expected_distance)
        assert record.nearest_wire_id == expected_wire
        assert record.nearest_tree_original_index == original
        np.testing.assert_allclose(
            record.minimum_clearance_m, expected_distance, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            record.nearest_tree_xyz_local_m, expected_point, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            record.nearest_wire_xyz_local_m, expected_nearest, rtol=0.0, atol=1e-12
        )
    np.testing.assert_allclose(
        result.global_minimum_clearance_m,
        min(expected_distances),
        rtol=0.0,
        atol=1e-12,
    )
