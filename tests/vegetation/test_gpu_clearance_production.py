from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.spans.clearance import _candidate_exact_nearest_edges, _direct_nearest_edges
from algorithms.spans.gpu_clearance import (
    GpuCapability,
    _canonical_order,
    _judge_candidates,
    create_gpu_accelerator,
    memory_plan,
    normalize_gpu_mode,
    probe_gpu,
)
from algorithms.vegetation.tree_wire_clearance import SegmentEdgeBvh, WireEdgeBundle


def _edges() -> WireEdgeBundle:
    return WireEdgeBundle(
        a_xyz=np.asarray(((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0e6, 0.0, 0.0))),
        b_xyz=np.asarray(((10.0, 0.0, 0.0), (10.0, 2.0, 0.0), (1.0e6, 0.0, 0.0))),
        wire_id=np.asarray((5, 2, 7), dtype=np.int32),
        segment_index=np.asarray((1, 0, 0), dtype=np.int32),
        polyline_edge_index=np.asarray((0, 0, 0), dtype=np.int32),
        wire_status=np.zeros(3, dtype=np.uint8),
        segment_quality_flags=np.zeros(3, dtype=np.uint16),
        edge_count=3,
    )


class _UnavailableRuntime:
    @staticmethod
    def getDeviceCount() -> int:
        raise RuntimeError("driver unavailable")


@pytest.fixture
def unavailable_cupy():
    return SimpleNamespace(cuda=SimpleNamespace(runtime=_UnavailableRuntime()))


def test_mode_and_memory_plan_are_device_and_workload_derived():
    assert normalize_gpu_mode(" auto ") == "AUTO"
    assert normalize_gpu_mode("cpu") == "CPU"
    assert normalize_gpu_mode("GPU") == "GPU"
    with pytest.raises(ValueError, match="acceleration_mode"):
        normalize_gpu_mode("CUDA")

    large_chunk, cap = memory_plan(1_000_000, 72, 8_000_000_000)
    small_chunk, small_cap = memory_plan(1_000_000, 72, 4_000)
    assert 0 < small_chunk < large_chunk <= 1_000_000
    assert cap == small_cap
    assert cap == int(np.ceil(np.sqrt(72)))


def test_auto_cpu_fallback_and_explicit_gpu_failure_are_fail_closed(unavailable_cupy):
    capability, module = probe_gpu(unavailable_cupy)
    assert module is unavailable_cupy
    assert capability.available is False
    assert capability.reason.startswith("CUDA_PROBE_FAILED:RuntimeError")

    accelerator, diagnostics = create_gpu_accelerator(
        _edges(), 1e-9, mode="AUTO", cp=unavailable_cupy
    )
    assert accelerator is None
    assert diagnostics["effective_mode"] == "CPU"
    with pytest.raises(RuntimeError, match="GPU_REQUESTED_BUT_UNAVAILABLE"):
        create_gpu_accelerator(_edges(), 1e-9, mode="GPU", cp=unavailable_cupy)


def test_cpu_exact_candidate_judge_matches_full_canonical_oracle():
    edges = _edges()
    points = np.asarray(
        ((5.0, 1.0, 0.0), (5.0, 0.0, 0.0), (1.0e6, 0.0, 0.0)),
        dtype=np.float64,
    )
    order = _canonical_order(edges)
    candidates = np.tile(np.arange(edges.edge_count, dtype=np.int64), (len(points), 1))
    counts = np.full(len(points), edges.edge_count, dtype=np.int64)

    judged = _judge_candidates(points, candidates, counts, order, edges, 1e-9)
    oracle = _direct_nearest_edges(points, edges, 1e-9)

    np.testing.assert_array_equal(judged[0], oracle[0])
    np.testing.assert_allclose(judged[1], oracle[1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(judged[2], oracle[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(judged[3], oracle[3], rtol=0.0, atol=0.0)


def test_auto_runtime_failure_reprocesses_every_chunk_on_cpu(monkeypatch):
    class FailingAccelerator:
        chunk_size = 2
        capability = GpuCapability(True, "TEST_GPU")

        @staticmethod
        def nearest(*_args, **_kwargs):
            raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(
        "algorithms.spans.gpu_clearance.create_gpu_accelerator",
        lambda *_args, **_kwargs: (
            FailingAccelerator(),
            {"requested_mode": "AUTO", "effective_mode": "GPU_CERTIFIED_CPU_EXACT"},
        ),
    )
    from algorithms.spans.clearance import run_span_tree_wire_clearance
    from core.config.fast_baseline import FastBaselineConfig

    from tests.vegetation.test_span_clearance_candidate_backend import _reconciled, _w1

    progress: list[dict] = []
    result = run_span_tree_wire_clearance(
        _reconciled(),
        [_w1()],
        FastBaselineConfig(),
        chunk_size=1,
        acceleration_mode="AUTO",
        progress_callback=progress.append,
    )
    assert result.diagnostics["acceleration"]["effective_mode"] == "CPU"
    assert result.diagnostics["acceleration"]["reason"].startswith(
        "GPU_RUNTIME_FAILED:MemoryError"
    )
    assert progress[-1]["processed_points"] == 3
    assert [item["processed_points"] for item in progress] == [2, 3]


def test_live_gpu_certificate_matches_cpu_oracle_when_cuda_is_available():
    capability, cp = probe_gpu()
    if not capability.available:
        pytest.skip(capability.reason)
    edges = _edges()
    bvh = SegmentEdgeBvh(edges, leaf_cap=2)
    points = np.asarray(
        (
            (5.0, 1.0, 0.0),
            (5.0, 0.0, 0.0),
            (10.0, 2.0, 0.0),
            (1.0e6, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    accelerator, _diagnostics = create_gpu_accelerator(
        edges, 1e-9, mode="GPU", cp=cp
    )
    assert accelerator is not None
    actual = accelerator.nearest(
        points,
        lambda values: _candidate_exact_nearest_edges(
            values, bvh, edges, 1e-9
        ),
    )
    expected = _candidate_exact_nearest_edges(points, bvh, edges, 1e-9)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[3], expected[3], rtol=0.0, atol=0.0)
    assert actual[5]["certificate_success_points"] + actual[5]["fallback_points"] == len(
        points
    )


def test_live_gpu_candidate_overflow_falls_back_exactly_when_cuda_is_available():
    capability, cp = probe_gpu()
    if not capability.available:
        pytest.skip(capability.reason)
    count = 16
    edges = WireEdgeBundle(
        a_xyz=np.zeros((count, 3), dtype=np.float64),
        b_xyz=np.tile((10.0, 0.0, 0.0), (count, 1)),
        wire_id=np.arange(count, dtype=np.int32),
        segment_index=np.zeros(count, dtype=np.int32),
        polyline_edge_index=np.zeros(count, dtype=np.int32),
        wire_status=np.zeros(count, dtype=np.uint8),
        segment_quality_flags=np.zeros(count, dtype=np.uint16),
        edge_count=count,
    )
    bvh = SegmentEdgeBvh(edges, leaf_cap=2)
    points = np.asarray(((5.0, 1.0, 0.0), (5.0, 0.0, 0.0)), dtype=np.float64)
    accelerator, _diagnostics = create_gpu_accelerator(edges, 1e-9, mode="GPU", cp=cp)
    assert accelerator is not None
    actual = accelerator.nearest(
        points,
        lambda values: _candidate_exact_nearest_edges(values, bvh, edges, 1e-9),
    )
    expected = _candidate_exact_nearest_edges(points, bvh, edges, 1e-9)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)
    assert actual[5]["fallback_points"] == len(points)
