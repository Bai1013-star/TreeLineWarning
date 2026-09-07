"""Gate-2 prototype benchmark for GPU-screened, CPU-judged clearance.

This module is deliberately not wired into the production clearance dispatcher.
The GPU proposes one edge per point; NumPy recomputes that candidate in float64.
Only an explicit, full comparison with the current CPU oracle may report
``SCIENTIFIC_EQUIVALENCE=PASS``.  Until a conservative candidate certificate is
implemented, the prototype result is not authoritative science.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.spans.clearance import (  # noqa: E402
    _candidate_exact_nearest_edges,
    _concatenate_span_wire_edges,
    _edge_order_by_key,
)
from algorithms.spans.wire import load_span_wire_result  # noqa: E402
from algorithms.vegetation.tree_wire_clearance import (  # noqa: E402
    SegmentEdgeBvh,
    build_wire_edge_bundle,
)
from core.config.fast_baseline import default_fast_baseline_config  # noqa: E402

PROTOTYPE_SCHEMA = "treeline-gpu-clearance-prototype-v1"
MODES = frozenset({"AUTO", "CPU", "GPU"})
_ESTIMATED_DEVICE_BYTES_PER_POINT = 64
_DEVICE_MEMORY_FRACTION = 0.25


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


@dataclass(slots=True)
class NearestResult:
    edge_position: np.ndarray
    distance_m: np.ndarray
    nearest_xyz: np.ndarray
    parameter: np.ndarray
    diagnostics: dict[str, Any]


def _cupy_module() -> tuple[Any | None, str | None]:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - capability probe must fail closed
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
        capability = GpuCapability(
            True,
            "CUDA_DEVICE_READY",
            device_count=count,
            device_name=name,
            free_memory_bytes=free,
            total_memory_bytes=total,
            cupy_version=str(cp.__version__),
            runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
            driver_version=int(cp.cuda.runtime.driverGetVersion()),
        )
        return capability, cp
    except Exception as exc:  # noqa: BLE001 - AUTO must remain a safe CPU fallback
        return GpuCapability(False, f"CUDA_PROBE_FAILED:{type(exc).__name__}:{exc}"), cp


def normalize_mode(mode: str) -> str:
    selected = str(mode).strip().upper()
    if selected not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    return selected


def _dynamic_chunk_size(point_count: int, free_memory_bytes: int) -> int:
    if point_count <= 0:
        return 0
    budget = max(1, int(free_memory_bytes * _DEVICE_MEMORY_FRACTION))
    return max(1, min(point_count, budget // _ESTIMATED_DEVICE_BYTES_PER_POINT))


_KERNEL_SOURCE = r"""
extern "C" __global__
void nearest_segment_candidate(
    const double* points, const long long point_count,
    const double* a, const double* d, const double* denominator,
    const long long edge_count, const double tolerance,
    long long* best_edge, double* best_distance)
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
        const double qx = ax + t * dx;
        const double qy = ay + t * dy;
        const double qz = az + t * dz;
        const double ex = px - qx;
        const double ey = py - qy;
        const double ez = pz - qz;
        const double value = sqrt(ex * ex + ey * ey + ez * ez);
        // Edges arrive in canonical key order. Retaining the earlier edge for
        // tolerance ties matches the frozen CPU ordering contract.
        if (chosen < 0 || value < best - tolerance) {
            best = value;
            chosen = edge;
        }
    }
    best_edge[row] = chosen;
    best_distance[row] = best;
}
"""


def _cpu_judge(
    points: np.ndarray,
    selected_edge: np.ndarray,
    edge_bundle: Any,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = np.empty(len(points), dtype=np.float64)
    nearest = np.empty((len(points), 3), dtype=np.float64)
    parameter = np.empty(len(points), dtype=np.float64)
    a_all = np.asarray(edge_bundle.a_xyz, dtype=np.float64)
    d_all = np.asarray(edge_bundle.b_xyz, dtype=np.float64) - a_all
    denominator_all = np.einsum("ij,ij->i", d_all, d_all)
    for start in range(0, len(points), max(1, chunk_size)):
        stop = min(len(points), start + max(1, chunk_size))
        edge = selected_edge[start:stop]
        p = points[start:stop]
        a = a_all[edge]
        d = d_all[edge]
        denominator = denominator_all[edge]
        safe = np.where(denominator == 0.0, 1.0, denominator)
        t = np.einsum("ij,ij->i", p - a, d) / safe
        t[denominator == 0.0] = 0.0
        np.clip(t, 0.0, 1.0, out=t)
        q = a + t[:, None] * d
        delta = p - q
        distance[start:stop] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        nearest[start:stop] = q
        parameter[start:stop] = t
    return distance, nearest, parameter


def gpu_screen_cpu_judge(
    points: np.ndarray,
    edge_bundle: Any,
    tolerance: float,
    *,
    cp: Any,
    free_memory_bytes: int,
) -> NearestResult:
    """Run the non-authoritative GPU screen and exact CPU candidate judge."""

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")
    edge_count = int(edge_bundle.edge_count)
    if edge_count <= 0:
        raise ValueError("at least one edge is required")
    if not len(xyz):
        empty = np.empty(0, dtype=np.float64)
        return NearestResult(
            np.empty(0, dtype=np.int64), empty, np.empty((0, 3)), empty,
            {"schema": PROTOTYPE_SCHEMA, "effective_mode": "GPU", "point_count": 0},
        )

    order = _edge_order_by_key(edge_bundle)
    a = np.asarray(edge_bundle.a_xyz, dtype=np.float64)[order]
    d = np.asarray(edge_bundle.b_xyz, dtype=np.float64)[order] - a
    denominator = np.einsum("ij,ij->i", d, d)
    chunk_size = _dynamic_chunk_size(len(xyz), free_memory_bytes)
    kernel = cp.RawKernel(_KERNEL_SOURCE, "nearest_segment_candidate")

    setup_started = time.perf_counter()
    gpu_a = cp.asarray(a)
    gpu_d = cp.asarray(d)
    gpu_denominator = cp.asarray(denominator)
    cp.cuda.Stream.null.synchronize()
    setup_transfer_seconds = time.perf_counter() - setup_started

    # Force JIT before measuring the steady-state pipeline.
    compile_started = time.perf_counter()
    warm_points = cp.asarray(xyz[:1])
    warm_edge = cp.empty(1, dtype=cp.int64)
    warm_distance = cp.empty(1, dtype=cp.float64)
    kernel(
        (1,), (1,),
        (warm_points, np.int64(1), gpu_a, gpu_d, gpu_denominator,
         np.int64(edge_count), np.float64(tolerance), warm_edge, warm_distance),
    )
    cp.cuda.Stream.null.synchronize()
    compile_warmup_seconds = time.perf_counter() - compile_started

    selected_canonical = np.empty(len(xyz), dtype=np.int64)
    gpu_distance = np.empty(len(xyz), dtype=np.float64)
    h2d_seconds = 0.0
    kernel_seconds = 0.0
    d2h_seconds = 0.0
    pipeline_started = time.perf_counter()
    for start in range(0, len(xyz), chunk_size):
        stop = min(len(xyz), start + chunk_size)
        transfer_started = time.perf_counter()
        gpu_points = cp.asarray(xyz[start:stop])
        gpu_edge = cp.empty(stop - start, dtype=cp.int64)
        gpu_best = cp.empty(stop - start, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        h2d_seconds += time.perf_counter() - transfer_started

        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        blocks = ((stop - start + 255) // 256,)
        kernel(
            blocks, (256,),
            (gpu_points, np.int64(stop - start), gpu_a, gpu_d, gpu_denominator,
             np.int64(edge_count), np.float64(tolerance), gpu_edge, gpu_best),
        )
        end.record()
        end.synchronize()
        kernel_seconds += float(cp.cuda.get_elapsed_time(begin, end)) / 1000.0

        transfer_started = time.perf_counter()
        selected_canonical[start:stop] = cp.asnumpy(gpu_edge)
        gpu_distance[start:stop] = cp.asnumpy(gpu_best)
        d2h_seconds += time.perf_counter() - transfer_started

    gpu_pipeline_seconds = time.perf_counter() - pipeline_started
    selected_edge = order[selected_canonical]
    judge_started = time.perf_counter()
    distance, nearest, parameter = _cpu_judge(
        xyz, selected_edge, edge_bundle, chunk_size=min(chunk_size, 262_144)
    )
    cpu_final_judge_seconds = time.perf_counter() - judge_started
    return NearestResult(
        selected_edge,
        distance,
        nearest,
        parameter,
        {
            "schema": PROTOTYPE_SCHEMA,
            "effective_mode": "GPU",
            "authoritative": False,
            "authority_blocker": "GPU_CANDIDATE_SET_HAS_NO_CONSERVATIVE_CERTIFICATE_V1",
            "point_count": len(xyz),
            "edge_count": edge_count,
            "chunk_size": chunk_size,
            "setup_edge_h2d_seconds": setup_transfer_seconds,
            "compile_warmup_seconds": compile_warmup_seconds,
            "point_h2d_seconds": h2d_seconds,
            "kernel_seconds": kernel_seconds,
            "result_d2h_seconds": d2h_seconds,
            "gpu_pipeline_seconds": gpu_pipeline_seconds,
            "cpu_final_judge_seconds": cpu_final_judge_seconds,
            "steady_end_to_end_seconds": gpu_pipeline_seconds + cpu_final_judge_seconds,
            "cold_end_to_end_seconds": (
                setup_transfer_seconds + compile_warmup_seconds
                + gpu_pipeline_seconds + cpu_final_judge_seconds
            ),
            "gpu_vs_cpu_judge_max_abs_distance_m": float(
                np.max(np.abs(gpu_distance - distance))
            ),
        },
    )


def nearest_edges(
    points: np.ndarray,
    bvh: SegmentEdgeBvh,
    edge_bundle: Any,
    tolerance: float,
    *,
    mode: str = "AUTO",
    cp: Any | None = None,
) -> NearestResult:
    requested = normalize_mode(mode)
    capability, cp = probe_gpu(cp)
    if requested == "GPU" and not capability.available:
        raise RuntimeError(f"GPU_REQUESTED_BUT_UNAVAILABLE:{capability.reason}")
    if requested == "CPU" or not capability.available:
        started = time.perf_counter()
        edge, distance, nearest, parameter, evaluated, diagnostics = (
            _candidate_exact_nearest_edges(points, bvh, edge_bundle, tolerance)
        )
        return NearestResult(
            edge,
            distance,
            nearest,
            parameter,
            {
                "schema": PROTOTYPE_SCHEMA,
                "requested_mode": requested,
                "effective_mode": "CPU",
                "reason": "EXPLICIT_CPU" if requested == "CPU" else capability.reason,
                "authoritative": True,
                "wall_seconds": time.perf_counter() - started,
                "evaluated_edge_total": evaluated,
                **diagnostics,
            },
        )
    result = gpu_screen_cpu_judge(
        points,
        edge_bundle,
        tolerance,
        cp=cp,
        free_memory_bytes=int(capability.free_memory_bytes or 0),
    )
    result.diagnostics.update(
        requested_mode=requested,
        capability=capability.to_dict(),
    )
    return result


def _load_workload(
    project_root: Path, run_id: str
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    run_root = project_root / "artifacts" / "project-runs" / "batches" / run_id
    manifest = json.loads((run_root / "batch.json").read_text(encoding="utf-8"))
    wire_results = [
        load_span_wire_result(run_root / span_id / "wire")
        for span_id in manifest.get("span_results", {})
    ]
    arrays, distance_ready_segments, _wire_count = _concatenate_span_wire_edges(wire_results)
    edge_bundle = build_wire_edge_bundle(arrays)
    if distance_ready_segments <= 0 or edge_bundle.edge_count <= 0:
        raise RuntimeError("REAL_WORKLOAD_HAS_NO_DISTANCE_READY_EDGE")
    with np.load(run_root / "tree_segmentation_original.npz", allow_pickle=False) as stored:
        xyz = np.asarray(stored["xyz_local_m"], dtype=np.float64)
        original_index = np.asarray(stored["original_index"], dtype=np.int64)
        tree_id = np.asarray(stored["tree_id"], dtype=np.int32)
    return edge_bundle, xyz, original_index, tree_id


def _tree_summary(
    result: NearestResult,
    tree_id: np.ndarray,
    original_index: np.ndarray,
    edge_bundle: Any,
    tolerance: float,
    danger_threshold_m: float,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for current_tree in sorted(set(int(value) for value in tree_id if value >= 0)):
        rows = np.flatnonzero(tree_id == current_tree)
        distances = result.distance_m[rows]
        minimum = float(np.min(distances))
        candidates = rows[distances <= minimum + tolerance]
        edges = result.edge_position[candidates]
        order = np.lexsort(
            (
                original_index[candidates],
                edge_bundle.polyline_edge_index[edges],
                edge_bundle.segment_index[edges],
                edge_bundle.wire_id[edges],
            )
        )
        selected = int(candidates[int(order[0])])
        edge = int(result.edge_position[selected])
        rows_out.append(
            {
                "tree_id": current_tree,
                "minimum_clearance_m": float(result.distance_m[selected]),
                "nearest_original_index": int(original_index[selected]),
                "nearest_wire_id": int(edge_bundle.wire_id[edge]),
                "nearest_segment_index": int(edge_bundle.segment_index[edge]),
                "nearest_polyline_edge_index": int(edge_bundle.polyline_edge_index[edge]),
                "danger_point_count": int(np.count_nonzero(distances <= danger_threshold_m)),
            }
        )
    return rows_out


def benchmark(project_root: Path, run_id: str, mode: str) -> dict[str, Any]:
    load_started = time.perf_counter()
    edge_bundle, xyz, original_index, tree_id = _load_workload(project_root, run_id)
    load_seconds = time.perf_counter() - load_started
    bvh = SegmentEdgeBvh(
        edge_bundle, default_fast_baseline_config().clearance_bvh_leaf_cap
    )
    tolerance = 1e-9
    capability, cp = probe_gpu()

    cpu = nearest_edges(xyz, bvh, edge_bundle, tolerance, mode="CPU", cp=cp)
    selected_mode = normalize_mode(mode)
    candidate = (
        cpu
        if selected_mode == "CPU"
        else nearest_edges(xyz, bvh, edge_bundle, tolerance, mode=selected_mode, cp=cp)
    )
    edge_equal = bool(np.array_equal(cpu.edge_position, candidate.edge_position))
    distance_delta = float(np.max(np.abs(cpu.distance_m - candidate.distance_m)))
    nearest_delta = float(np.max(np.abs(cpu.nearest_xyz - candidate.nearest_xyz)))
    parameter_delta = float(np.max(np.abs(cpu.parameter - candidate.parameter)))
    cpu_summary = _tree_summary(cpu, tree_id, original_index, edge_bundle, tolerance, 0.5)
    candidate_summary = _tree_summary(
        candidate, tree_id, original_index, edge_bundle, tolerance, 0.5
    )
    equivalent = bool(
        edge_equal
        and distance_delta <= 1e-12
        and nearest_delta <= 1e-12
        and parameter_delta <= 1e-12
        and cpu_summary == candidate_summary
    )
    cpu_seconds = float(cpu.diagnostics["wall_seconds"])
    accelerated_seconds = float(
        candidate.diagnostics.get("steady_end_to_end_seconds", cpu_seconds)
    )
    return {
        "schema": PROTOTYPE_SCHEMA,
        "project_root": str(project_root.resolve()),
        "analysis_run_id": run_id,
        "point_count": len(xyz),
        "edge_count": int(edge_bundle.edge_count),
        "workload_load_seconds": load_seconds,
        "capability": capability.to_dict(),
        "cpu": cpu.diagnostics,
        "candidate": candidate.diagnostics,
        "equivalence": {
            "status": "PASS" if equivalent else "FAIL",
            "edge_position_exact": edge_equal,
            "maximum_distance_delta_m": distance_delta,
            "maximum_nearest_xyz_delta_m": nearest_delta,
            "maximum_parameter_delta": parameter_delta,
            "per_tree_summary_exact": cpu_summary == candidate_summary,
            "tree_count": len(cpu_summary),
        },
        "steady_speedup": cpu_seconds / accelerated_seconds,
        "production_integration_allowed": False,
        "production_blocker": "CONSERVATIVE_GPU_CANDIDATE_CERTIFICATE_NOT_IMPLEMENTED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", default="AUTO", choices=sorted(MODES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = benchmark(args.project_root, args.run_id, args.mode)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output.resolve())
    return 0 if payload["equivalence"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
