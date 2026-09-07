from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.vegetation.tree_wire_clearance import SegmentEdgeBvh, WireEdgeBundle
from scripts.benchmark_gpu_clearance_v1 import (
    _dynamic_chunk_size,
    nearest_edges,
    normalize_mode,
    probe_gpu,
)


def _edges() -> WireEdgeBundle:
    return WireEdgeBundle(
        a_xyz=np.asarray(((0.0, 0.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float64),
        b_xyz=np.asarray(((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)), dtype=np.float64),
        wire_id=np.asarray((0, 1), dtype=np.int32),
        segment_index=np.zeros(2, dtype=np.int32),
        polyline_edge_index=np.zeros(2, dtype=np.int32),
        wire_status=np.zeros(2, dtype=np.uint8),
        segment_quality_flags=np.zeros(2, dtype=np.uint16),
        edge_count=2,
    )


class _UnavailableRuntime:
    @staticmethod
    def getDeviceCount() -> int:
        raise RuntimeError("driver unavailable")


@pytest.fixture
def unavailable_cupy():
    return SimpleNamespace(cuda=SimpleNamespace(runtime=_UnavailableRuntime()))


def test_auto_falls_back_to_the_authoritative_cpu_path(unavailable_cupy):
    edges = _edges()
    bvh = SegmentEdgeBvh(edges, 16)
    points = np.asarray(((0.5, 0.25, 0.0), (0.5, 1.75, 0.0)), dtype=np.float64)

    result = nearest_edges(points, bvh, edges, 1e-9, mode="AUTO", cp=unavailable_cupy)

    assert result.diagnostics["effective_mode"] == "CPU"
    assert result.diagnostics["authoritative"] is True
    np.testing.assert_array_equal(result.edge_position, (0, 1))
    np.testing.assert_allclose(result.distance_m, (0.25, 0.25), rtol=0.0, atol=0.0)


def test_explicit_gpu_fails_closed_when_capability_probe_fails(unavailable_cupy):
    edges = _edges()
    bvh = SegmentEdgeBvh(edges, 16)

    with pytest.raises(RuntimeError, match="GPU_REQUESTED_BUT_UNAVAILABLE"):
        nearest_edges(
            np.asarray(((0.5, 0.25, 0.0),), dtype=np.float64),
            bvh,
            edges,
            1e-9,
            mode="GPU",
            cp=unavailable_cupy,
        )


def test_capability_failure_is_structured_and_does_not_escape(unavailable_cupy):
    capability, module = probe_gpu(unavailable_cupy)

    assert module is unavailable_cupy
    assert capability.available is False
    assert capability.reason.startswith("CUDA_PROBE_FAILED:RuntimeError")


def test_chunk_size_uses_detected_free_memory_and_is_workload_bounded():
    assert _dynamic_chunk_size(10_000, 64 * 400) == 100
    assert _dynamic_chunk_size(50, 64 * 400) == 50
    assert _dynamic_chunk_size(0, 64 * 400) == 0


def test_execution_mode_is_explicit_and_case_insensitive():
    assert normalize_mode(" auto ") == "AUTO"
    assert normalize_mode("cpu") == "CPU"
    assert normalize_mode("GPU") == "GPU"
    with pytest.raises(ValueError, match="mode must be one of"):
        normalize_mode("CUDA")
