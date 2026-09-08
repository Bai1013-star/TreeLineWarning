"""Benchmark production certified-GPU clearance against the CPU oracle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.spans.clearance import _candidate_exact_nearest_edges  # noqa: E402
from algorithms.spans.gpu_clearance import create_gpu_accelerator  # noqa: E402
from algorithms.vegetation.tree_wire_clearance import SegmentEdgeBvh  # noqa: E402
from core.config.fast_baseline import default_fast_baseline_config  # noqa: E402

from scripts.benchmark_gpu_clearance_v1 import (  # noqa: E402
    _load_workload,
    _tree_summary,
)


def benchmark(project_root: Path, run_id: str) -> dict[str, object]:
    load_started = time.perf_counter()
    edges, xyz, original_index, tree_id = _load_workload(project_root, run_id)
    load_seconds = time.perf_counter() - load_started
    config = default_fast_baseline_config()
    tolerance = config.clearance_tie_tolerance_m
    bvh = SegmentEdgeBvh(edges, config.clearance_bvh_leaf_cap)

    cpu_started = time.perf_counter()
    cpu = _candidate_exact_nearest_edges(xyz, bvh, edges, tolerance)
    cpu_seconds = time.perf_counter() - cpu_started

    accelerator, initial = create_gpu_accelerator(edges, tolerance, mode="GPU")
    if accelerator is None:  # pragma: no cover - explicit GPU already fails closed
        raise RuntimeError(f"GPU_ACCELERATOR_UNAVAILABLE:{initial}")
    gpu_started = time.perf_counter()
    gpu = accelerator.nearest(
        xyz,
        lambda values: _candidate_exact_nearest_edges(
            values, bvh, edges, tolerance
        ),
    )
    gpu_seconds = time.perf_counter() - gpu_started
    cpu_result = SimpleNamespace(
        edge_position=cpu[0],
        distance_m=cpu[1],
        nearest_xyz=cpu[2],
        parameter=cpu[3],
    )
    gpu_result = SimpleNamespace(
        edge_position=gpu[0],
        distance_m=gpu[1],
        nearest_xyz=gpu[2],
        parameter=gpu[3],
    )
    cpu_trees = _tree_summary(
        cpu_result, tree_id, original_index, edges, tolerance, 0.5
    )
    gpu_trees = _tree_summary(
        gpu_result, tree_id, original_index, edges, tolerance, 0.5
    )
    equivalence = {
        "edge_position_exact": bool(np.array_equal(cpu[0], gpu[0])),
        "maximum_distance_delta_m": float(np.max(np.abs(cpu[1] - gpu[1]))),
        "maximum_nearest_xyz_delta_m": float(np.max(np.abs(cpu[2] - gpu[2]))),
        "maximum_parameter_delta": float(np.max(np.abs(cpu[3] - gpu[3]))),
        "per_tree_summary_exact": cpu_trees == gpu_trees,
        "tree_count": len(cpu_trees),
    }
    passed = all(
        (
            equivalence["edge_position_exact"],
            equivalence["maximum_distance_delta_m"] == 0.0,
            equivalence["maximum_nearest_xyz_delta_m"] == 0.0,
            equivalence["maximum_parameter_delta"] == 0.0,
            equivalence["per_tree_summary_exact"],
        )
    )
    return {
        "schema": "treeline-certified-gpu-clearance-benchmark-v1",
        "project_root": str(project_root.resolve()),
        "analysis_run_id": run_id,
        "point_count": len(xyz),
        "edge_count": int(edges.edge_count),
        "workload_load_seconds": load_seconds,
        "cpu_exact_seconds": cpu_seconds,
        "gpu_certified_total_seconds": gpu_seconds,
        "speedup": cpu_seconds / gpu_seconds,
        "gpu_diagnostics": dict(accelerator.diagnostics),
        "equivalence": {"status": "PASS" if passed else "FAIL", **equivalence},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = benchmark(args.project_root, args.run_id)
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
