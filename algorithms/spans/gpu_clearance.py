"""Certified CUDA screening for authoritative CPU float64 clearance judging.

CUDA never publishes distance science.  It proposes one finite upper-bound
edge, then returns every edge whose segment AABB has a directed-rounding lower
bound within that exact CPU upper bound plus the frozen tie tolerance.  A CPU
float64 judge evaluates that complete candidate set in canonical order.  Any
overflow, uncertainty, allocation error, kernel error, or unavailable runtime
falls back to the existing CPU oracle in AUTO mode.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

GPU_CLEARANCE_SCHEMA = "treeline-certified-gpu-clearance-v1"
GPU_MODES = frozenset({"AUTO", "CPU", "GPU"})
_DEVICE_MEMORY_FRACTION = 0.20


@dataclass(frozen=True, slots=True)
class GpuCapability:
    available: bool
    reason: str
    device_count: int = 0
    device_name: str | None = None
    free_memory_bytes: int | None = None
    total_memory_bytes: int | None = None
    cupy_version: str | None = None
    runtime_version: int | None = None
    driver_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "free_memory_bytes": self.free_memory_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "cupy_version": self.cupy_version,
            "runtime_version": self.runtime_version,
            "driver_version": self.driver_version,
        }


def normalize_gpu_mode(mode: str) -> str:
    selected = str(mode).strip().upper()
    if selected not in GPU_MODES:
        raise ValueError(f"acceleration_mode must be one of {sorted(GPU_MODES)}")
    return selected


def _cupy_module() -> tuple[Any | None, str | None]:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional capability probe
        return None, f"CUPY_IMPORT_FAILED:{type(exc).__name__}:{exc}"
    return cp, None


def probe_gpu(cp: Any | None = None) -> tuple[GpuCapability, Any | None]:
    if cp is None:
        cp, error = _cupy_module()
        if cp is None:
            return GpuCapability(False, str(error)), None
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            return GpuCapability(False, "NO_CUDA_DEVICE", device_count=count), cp
        properties = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = properties.get("name")
        name = raw_name.decode("utf-8", "replace") if isinstance(raw_name, bytes) else str(raw_name)
        free, total = (int(value) for value in cp.cuda.runtime.memGetInfo())
        return (
            GpuCapability(
                True,
                "CUDA_DEVICE_READY",
                device_count=count,
                device_name=name,
                free_memory_bytes=free,
                total_memory_bytes=total,
                cupy_version=str(cp.__version__),
                runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
                driver_version=int(cp.cuda.runtime.driverGetVersion()),
            ),
            cp,
        )
    except Exception as exc:  # noqa: BLE001 - AUTO must fail closed to CPU
        return GpuCapability(False, f"CUDA_PROBE_FAILED:{type(exc).__name__}:{exc}"), cp


def memory_plan(
    point_count: int, edge_count: int, free_memory_bytes: int
) -> tuple[int, int]:
    """Return memory-derived ``(point chunk, candidate cap)`` for this device."""

    if point_count <= 0 or edge_count <= 0 or free_memory_bytes <= 0:
        return 0, 0
    candidate_cap = min(edge_count, max(1, math.ceil(math.sqrt(edge_count))))
    bytes_per_point = 24 + 8 + 8 + 4 + candidate_cap * 4
    budget = max(bytes_per_point, int(free_memory_bytes * _DEVICE_MEMORY_FRACTION))
    return max(1, min(point_count, budget // bytes_per_point)), candidate_cap


_SCREEN_KERNEL = r"""
extern "C" __global__
void nearest_segment_screen(
    const double* points, const long long point_count,
    const double* a, const double* d, const double* denominator,
    const long long edge_count, long long* best_edge)
{
    const long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= point_count) return;
    const double px = points[row * 3];
    const double py = points[row * 3 + 1];
    const double pz = points[row * 3 + 2];
    double best = 1.0 / 0.0;
    long long chosen = -1;
    for (long long edge = 0; edge < edge_count; ++edge) {
        const double ax = a[edge * 3];
        const double ay = a[edge * 3 + 1];
        const double az = a[edge * 3 + 2];
        const double dx = d[edge * 3];
        const double dy = d[edge * 3 + 1];
        const double dz = d[edge * 3 + 2];
        const double denom = denominator[edge];
        double t = 0.0;
        if (denom != 0.0) {
            t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / denom;
            t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
        }
        const double ex = px - (ax + t * dx);
        const double ey = py - (ay + t * dy);
        const double ez = pz - (az + t * dz);
        const double value = sqrt(ex * ex + ey * ey + ez * ez);
        if (chosen < 0 || value < best) { best = value; chosen = edge; }
    }
    best_edge[row] = chosen;
}
"""


_CERTIFICATE_KERNEL = r"""
extern "C" __global__
void conservative_aabb_candidates(
    const double* points, const long long point_count,
    const double* lower, const double* upper,
    const double* exact_upper, const long long edge_count,
    const double tolerance, const int candidate_cap,
    int* candidate_count, int* candidates)
{
    const long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= point_count) return;
    const double px = points[row * 3];
    const double py = points[row * 3 + 1];
    const double pz = points[row * 3 + 2];
    const double threshold = __dadd_ru(exact_upper[row], tolerance);
    int count = 0;
    for (long long edge = 0; edge < edge_count; ++edge) {
        double dx = 0.0, dy = 0.0, dz = 0.0;
        if (px < lower[edge * 3]) dx = __dsub_rd(lower[edge * 3], px);
        else if (px > upper[edge * 3]) dx = __dsub_rd(px, upper[edge * 3]);
        if (py < lower[edge * 3 + 1]) dy = __dsub_rd(lower[edge * 3 + 1], py);
        else if (py > upper[edge * 3 + 1]) dy = __dsub_rd(py, upper[edge * 3 + 1]);
        if (pz < lower[edge * 3 + 2]) dz = __dsub_rd(lower[edge * 3 + 2], pz);
        else if (pz > upper[edge * 3 + 2]) dz = __dsub_rd(pz, upper[edge * 3 + 2]);
        double square = __dmul_rd(dx, dx);
        square = __dadd_rd(square, __dmul_rd(dy, dy));
        square = __dadd_rd(square, __dmul_rd(dz, dz));
        const double bound = __dsqrt_rd(square);
        if (bound <= threshold) {
            if (count < candidate_cap) candidates[row * candidate_cap + count] = (int)edge;
            ++count;
        }
    }
    candidate_count[row] = count;
}
"""


def _canonical_order(edge_bundle: Any) -> np.ndarray:
    return np.lexsort(
        (
            np.asarray(edge_bundle.polyline_edge_index, dtype=np.int64),
            np.asarray(edge_bundle.segment_index, dtype=np.int64),
            np.asarray(edge_bundle.wire_id, dtype=np.int64),
        )
    ).astype(np.int64, copy=False)


def _exact_selected_edges(
    points: np.ndarray, selected_edge: np.ndarray, edge_bundle: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_all = np.asarray(edge_bundle.a_xyz, dtype=np.float64)
    d_all = np.asarray(edge_bundle.b_xyz, dtype=np.float64) - a_all
    denominator_all = np.einsum("ij,ij->i", d_all, d_all)
    edge = np.asarray(selected_edge, dtype=np.int64)
    p = np.asarray(points, dtype=np.float64)
    a = a_all[edge]
    d = d_all[edge]
    denominator = denominator_all[edge]
    safe = np.where(denominator == 0.0, 1.0, denominator)
    parameter = np.einsum("ij,ij->i", p - a, d) / safe
    parameter[denominator == 0.0] = 0.0
    np.clip(parameter, 0.0, 1.0, out=parameter)
    nearest = a + parameter[:, None] * d
    delta = p - nearest
    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return distance, nearest, parameter


def _judge_candidates(
    points: np.ndarray,
    canonical_candidates: np.ndarray,
    candidate_count: np.ndarray,
    canonical_to_edge: np.ndarray,
    edge_bundle: Any,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    count = len(points)
    best_edge = np.full(count, -1, dtype=np.int64)
    best_distance = np.full(count, np.inf, dtype=np.float64)
    best_nearest = np.empty((count, 3), dtype=np.float64)
    best_parameter = np.empty(count, dtype=np.float64)
    evaluated = 0
    for column in range(canonical_candidates.shape[1]):
        rows = np.flatnonzero(candidate_count > column)
        if not len(rows):
            break
        edge = canonical_to_edge[canonical_candidates[rows, column]]
        distance, nearest, parameter = _exact_selected_edges(points[rows], edge, edge_bundle)
        better = distance < best_distance[rows] - tolerance
        targets = rows[better]
        if len(targets):
            best_edge[targets] = edge[better]
            best_distance[targets] = distance[better]
            best_nearest[targets] = nearest[better]
            best_parameter[targets] = parameter[better]
        evaluated += len(rows)
    return best_edge, best_distance, best_nearest, best_parameter, evaluated


class CertifiedGpuClearance:
    """One controlled CUDA context and immutable edge upload per clearance run."""

    def __init__(self, edge_bundle: Any, tolerance: float, capability: GpuCapability, cp: Any):
        self.edge_bundle = edge_bundle
        self.tolerance = float(tolerance)
        self.capability = capability
        self.cp = cp
        self.order = _canonical_order(edge_bundle)
        a = np.asarray(edge_bundle.a_xyz, dtype=np.float64)[self.order]
        b = np.asarray(edge_bundle.b_xyz, dtype=np.float64)[self.order]
        d = b - a
        denominator = np.einsum("ij,ij->i", d, d)
        self.chunk_size, self.candidate_cap = memory_plan(
            2**63 - 1,
            len(self.order),
            int(capability.free_memory_bytes or 0),
        )
        if self.chunk_size <= 0 or self.candidate_cap <= 0:
            raise MemoryError("GPU_MEMORY_PLAN_UNAVAILABLE")
        started = time.perf_counter()
        self.gpu_a = cp.asarray(a)
        self.gpu_d = cp.asarray(d)
        self.gpu_denominator = cp.asarray(denominator)
        self.gpu_lower = cp.asarray(np.minimum(a, b))
        self.gpu_upper = cp.asarray(np.maximum(a, b))
        self.screen_kernel = cp.RawKernel(_SCREEN_KERNEL, "nearest_segment_screen")
        self.certificate_kernel = cp.RawKernel(
            _CERTIFICATE_KERNEL, "conservative_aabb_candidates"
        )
        cp.cuda.Stream.null.synchronize()
        self.diagnostics: dict[str, Any] = {
            "schema": GPU_CLEARANCE_SCHEMA,
            "requested_mode": None,
            "effective_mode": "GPU_CERTIFIED_CPU_EXACT",
            "authoritative": True,
            "certificate": "CUDA_DIRECTED_ROUNDING_SEGMENT_AABB_LOWER_BOUND",
            "capability": capability.to_dict(),
            "edge_setup_h2d_seconds": time.perf_counter() - started,
            "compile_warmup_seconds": 0.0,
            "point_h2d_seconds": 0.0,
            "screen_kernel_seconds": 0.0,
            "upper_d2h_seconds": 0.0,
            "cpu_upper_judge_seconds": 0.0,
            "certificate_kernel_seconds": 0.0,
            "candidate_d2h_seconds": 0.0,
            "cpu_exact_judge_seconds": 0.0,
            "certificate_success_points": 0,
            "fallback_points": 0,
            "candidate_cap": self.candidate_cap,
            "device_chunk_size": self.chunk_size,
        }
        self._warmup()

    def _warmup(self) -> None:
        cp = self.cp
        started = time.perf_counter()
        point = cp.zeros((1, 3), dtype=cp.float64)
        winner = cp.empty(1, dtype=cp.int64)
        self.screen_kernel(
            (1,),
            (1,),
            (
                point,
                np.int64(1),
                self.gpu_a,
                self.gpu_d,
                self.gpu_denominator,
                np.int64(len(self.order)),
                winner,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        self.diagnostics["compile_warmup_seconds"] = time.perf_counter() - started

    def nearest(
        self,
        points: np.ndarray,
        cpu_fallback: Callable[[np.ndarray], tuple[Any, ...]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
        xyz = np.asarray(points, dtype=np.float64)
        cp = self.cp
        output_edge = np.empty(len(xyz), dtype=np.int64)
        output_distance = np.empty(len(xyz), dtype=np.float64)
        output_nearest = np.empty((len(xyz), 3), dtype=np.float64)
        output_parameter = np.empty(len(xyz), dtype=np.float64)
        evaluated_total = 0
        certificate_before = int(self.diagnostics["certificate_success_points"])
        fallback_before = int(self.diagnostics["fallback_points"])
        for start in range(0, len(xyz), self.chunk_size):
            stop = min(len(xyz), start + self.chunk_size)
            block = xyz[start:stop]
            transferred = time.perf_counter()
            gpu_points = cp.asarray(block)
            gpu_winner = cp.empty(len(block), dtype=cp.int64)
            cp.cuda.Stream.null.synchronize()
            self.diagnostics["point_h2d_seconds"] += time.perf_counter() - transferred
            event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
            event_start.record()
            self.screen_kernel(
                ((len(block) + 255) // 256,),
                (256,),
                (
                    gpu_points,
                    np.int64(len(block)),
                    self.gpu_a,
                    self.gpu_d,
                    self.gpu_denominator,
                    np.int64(len(self.order)),
                    gpu_winner,
                ),
            )
            event_end.record()
            event_end.synchronize()
            self.diagnostics["screen_kernel_seconds"] += (
                float(cp.cuda.get_elapsed_time(event_start, event_end)) / 1000.0
            )
            transferred = time.perf_counter()
            canonical_winner = cp.asnumpy(gpu_winner)
            self.diagnostics["upper_d2h_seconds"] += time.perf_counter() - transferred
            judged = time.perf_counter()
            winner_edge = self.order[canonical_winner]
            exact_upper, _nearest, _parameter = _exact_selected_edges(
                block, winner_edge, self.edge_bundle
            )
            self.diagnostics["cpu_upper_judge_seconds"] += time.perf_counter() - judged

            transferred = time.perf_counter()
            gpu_exact_upper = cp.asarray(exact_upper)
            gpu_count = cp.empty(len(block), dtype=cp.int32)
            gpu_candidates = cp.empty((len(block), self.candidate_cap), dtype=cp.int32)
            cp.cuda.Stream.null.synchronize()
            self.diagnostics["point_h2d_seconds"] += time.perf_counter() - transferred
            event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
            event_start.record()
            self.certificate_kernel(
                ((len(block) + 255) // 256,),
                (256,),
                (
                    gpu_points,
                    np.int64(len(block)),
                    self.gpu_lower,
                    self.gpu_upper,
                    gpu_exact_upper,
                    np.int64(len(self.order)),
                    np.float64(self.tolerance),
                    np.int32(self.candidate_cap),
                    gpu_count,
                    gpu_candidates,
                ),
            )
            event_end.record()
            event_end.synchronize()
            self.diagnostics["certificate_kernel_seconds"] += (
                float(cp.cuda.get_elapsed_time(event_start, event_end)) / 1000.0
            )
            transferred = time.perf_counter()
            candidate_count = cp.asnumpy(gpu_count).astype(np.int64, copy=False)
            canonical_candidates = cp.asnumpy(gpu_candidates).astype(np.int64, copy=False)
            self.diagnostics["candidate_d2h_seconds"] += time.perf_counter() - transferred

            certified = (candidate_count > 0) & (candidate_count <= self.candidate_cap)
            certified_rows = np.flatnonzero(certified)
            judged = time.perf_counter()
            if len(certified_rows):
                edge, distance, nearest, parameter, evaluated = _judge_candidates(
                    block[certified_rows],
                    canonical_candidates[certified_rows],
                    candidate_count[certified_rows],
                    self.order,
                    self.edge_bundle,
                    self.tolerance,
                )
                targets = start + certified_rows
                output_edge[targets] = edge
                output_distance[targets] = distance
                output_nearest[targets] = nearest
                output_parameter[targets] = parameter
                evaluated_total += evaluated
            fallback_rows = np.flatnonzero(~certified)
            if len(fallback_rows):
                edge, distance, nearest, parameter, evaluated, _diagnostics = cpu_fallback(
                    block[fallback_rows]
                )
                targets = start + fallback_rows
                output_edge[targets] = edge
                output_distance[targets] = distance
                output_nearest[targets] = nearest
                output_parameter[targets] = parameter
                evaluated_total += evaluated
            self.diagnostics["cpu_exact_judge_seconds"] += time.perf_counter() - judged
            self.diagnostics["certificate_success_points"] += len(certified_rows)
            self.diagnostics["fallback_points"] += len(fallback_rows)
        return (
            output_edge,
            output_distance,
            output_nearest,
            output_parameter,
            evaluated_total,
            {
                "certificate_success_points": int(
                    self.diagnostics["certificate_success_points"]
                )
                - certificate_before,
                "fallback_points": int(self.diagnostics["fallback_points"])
                - fallback_before,
                "candidate_expansion_rounds": 0,
            },
        )


def create_gpu_accelerator(
    edge_bundle: Any,
    tolerance: float,
    *,
    mode: str,
    cp: Any | None = None,
) -> tuple[CertifiedGpuClearance | None, dict[str, Any]]:
    requested = normalize_gpu_mode(mode)
    if requested == "CPU":
        return None, {
            "schema": GPU_CLEARANCE_SCHEMA,
            "requested_mode": requested,
            "effective_mode": "CPU",
            "reason": "EXPLICIT_CPU",
        }
    capability, cp = probe_gpu(cp)
    if not capability.available:
        if requested == "GPU":
            raise RuntimeError(f"GPU_REQUESTED_BUT_UNAVAILABLE:{capability.reason}")
        return None, {
            "schema": GPU_CLEARANCE_SCHEMA,
            "requested_mode": requested,
            "effective_mode": "CPU",
            "reason": capability.reason,
            "capability": capability.to_dict(),
        }
    try:
        accelerator = CertifiedGpuClearance(edge_bundle, tolerance, capability, cp)
        accelerator.diagnostics["requested_mode"] = requested
        return accelerator, dict(accelerator.diagnostics)
    except Exception as exc:
        if requested == "GPU":
            raise RuntimeError(f"GPU_INITIALIZATION_FAILED:{type(exc).__name__}:{exc}") from exc
        return None, {
            "schema": GPU_CLEARANCE_SCHEMA,
            "requested_mode": requested,
            "effective_mode": "CPU",
            "reason": f"GPU_INITIALIZATION_FAILED:{type(exc).__name__}:{exc}",
            "capability": capability.to_dict(),
        }


__all__ = [
    "GPU_CLEARANCE_SCHEMA",
    "GPU_MODES",
    "CertifiedGpuClearance",
    "GpuCapability",
    "create_gpu_accelerator",
    "memory_plan",
    "normalize_gpu_mode",
    "probe_gpu",
]
