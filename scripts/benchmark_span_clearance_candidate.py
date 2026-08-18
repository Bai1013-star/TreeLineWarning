"""Bounded performance check for the exact streaming clearance backend."""

from __future__ import annotations

import argparse
import time

import numpy as np
from algorithms.spans.clearance import _candidate_exact_nearest_edges
from algorithms.vegetation.tree_wire_clearance import SegmentEdgeBvh, WireEdgeBundle


def _representative_edges() -> WireEdgeBundle:
    edge_count = 307
    y = np.arange(edge_count, dtype=np.float64)
    return WireEdgeBundle(
        a_xyz=np.column_stack((np.zeros(edge_count), y, np.zeros(edge_count))),
        b_xyz=np.column_stack((np.full(edge_count, 10.0), y, np.zeros(edge_count))),
        wire_id=np.arange(edge_count, dtype=np.int32),
        segment_index=np.zeros(edge_count, dtype=np.int32),
        polyline_edge_index=np.zeros(edge_count, dtype=np.int32),
        wire_status=np.zeros(edge_count, dtype=np.uint8),
        segment_quality_flags=np.zeros(edge_count, dtype=np.uint16),
        edge_count=edge_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="reuse an already recorded 100k legacy baseline and time only the new backend",
    )
    args = parser.parse_args(argv)
    edges = _representative_edges()
    bvh = SegmentEdgeBvh(edges, leaf_cap=16)
    rng = np.random.default_rng(20260818)

    def points(count: int) -> np.ndarray:
        return rng.uniform((0.0, 0.0, -3.0), (10.0, 306.0, 3.0), size=(count, 3))

    sample_100k = points(100_000)
    old_seconds: float | None = None
    if not args.new_only:
        started = time.perf_counter()
        for point in sample_100k:
            bvh.nearest_edge(point, 1e-9)
        old_seconds = time.perf_counter() - started
        print(f"OLD_PYTHON_BVH_100K_SECONDS={old_seconds:.6f}")
    for count in (100_000, 1_000_000, 5_000_000):
        sample = sample_100k if count == 100_000 else points(count)
        started = time.perf_counter()
        _edge, _distance, _nearest, _parameter, _visited, diagnostics = (
            _candidate_exact_nearest_edges(sample, bvh, edges, 1e-9)
        )
        elapsed = time.perf_counter() - started
        print(
            f"NEW_CANDIDATE_{count}_SECONDS={elapsed:.6f}; "
            f"POINTS_PER_SECOND={count / elapsed:.1f}; "
            f"FALLBACK={diagnostics['fallback_points']}; "
            f"CERTIFIED={diagnostics['certificate_success_points']}"
        )
    print("OLD_PYTHON_BVH_1M=NOT_RUN_BY_OWNER_POLICY")
    if old_seconds is not None:
        print(f"EXTRAPOLATED_OLD_1M_SECONDS={old_seconds * 10.0:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
