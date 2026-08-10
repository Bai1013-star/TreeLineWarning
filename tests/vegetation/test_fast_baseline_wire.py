"""Synthetic tests for the per-span wire pipeline (Parts 22-31, 45)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from algorithms.poles.topology import SpanRecord
from algorithms.spans.wire import (
    _dbscan,
    _local_pca_features,
    run_span_wire_pipeline,
)
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter
from core.config.fast_baseline import default_fast_baseline_config

_CFG = default_fast_baseline_config()


def _span(start=(0.0, 0.0), end=(50.0, 0.0), top_z=15.0) -> SpanRecord:
    return SpanRecord(
        line_id=0,
        line_name="Line01",
        span_id=0,
        span_name="Line01-Span001",
        start_pole_candidate_id=0,
        end_pole_candidate_id=1,
        start_xyz_absolute_m=(float(start[0]), float(start[1]), 0.0),
        end_xyz_absolute_m=(float(end[0]), float(end[1]), 0.0),
        start_pole_top_z_m=top_z,
        end_pole_top_z_m=top_z,
        length_m=float(np.hypot(end[0] - start[0], end[1] - start[1])),
        direction_xy=(1.0, 0.0, 0.0),
        half_width_m=9.0,
        halo_m=6.0,
        core_bounds_min_xyz=(float(start[0]), -9.0, 0.0),
        core_bounds_max_xyz=(float(end[0]), 9.0, 20.0),
        halo_bounds_min_xyz=(float(start[0]) - 6.0, -9.0, 0.0),
        halo_bounds_max_xyz=(float(end[0]) + 6.0, 9.0, 20.0),
        quality="CONFIRMED_SPAN",
    )


def _run_wire(points: np.ndarray, span: SpanRecord | None = None) -> object:
    with tempfile.TemporaryDirectory() as td:
        writer = _RawRegionWriter(
            Path(td) / "region", origin_xyz=(0.0, 0.0, 0.0)
        )
        writer.append(points, np.arange(len(points)), np.zeros(len(points), dtype=np.uint8))
        region = writer.close()
        return run_span_wire_pipeline(
            span or _span(),
            region,
            _CFG,
            origin_xyz=(0.0, 0.0, 0.0),
            output_dir=Path(td) / "out",
        )


def _line(
    s_start: float, s_stop: float, u: float, z: float, spacing: float = 0.2
) -> np.ndarray:
    s = np.arange(s_start, s_stop + 1e-9, spacing)
    return np.column_stack((s, np.full(len(s), u), np.full(len(s), z)))


def test_one_straight_conductor_yields_one_strand() -> None:
    points = _line(2.0, 48.0, 0.0, 12.0)
    result = _run_wire(points)
    assert result.strand_count == 1
    strand = result.strands[0]
    assert abs(strand.median_u_m - 0.0) < 0.5
    assert strand.s_span_m > 30.0


def test_multiple_parallel_conductors_are_separated() -> None:
    lines = [_line(2.0, 48.0, u, 12.0) for u in (-2.0, 0.5, 3.0)]
    points = np.concatenate(lines, axis=0)
    result = _run_wire(points)
    assert result.strand_count == 3
    median_us = sorted(round(strand.median_u_m, 1) for strand in result.strands)
    assert median_us == [-2.0, 0.5, 3.0]


def test_sagging_conductor_is_reconstructed_as_smooth_curve() -> None:
    s = np.linspace(2.0, 48.0, 120)
    # Parabolic sag in z along the span.
    z = 12.0 - 1.5 * (1.0 - (2.0 * (s - 25.0) / 46.0) ** 2)
    points = np.column_stack((s, np.zeros_like(s), z))
    result = _run_wire(points)
    assert result.strand_count == 1
    curve = result.strands[0].curve_xyz_local_m
    assert np.any(curve[:, 2] < 11.5)  # sag captured
    assert np.isfinite(curve).all()


def test_fragmented_but_continuous_conductor_reconnects() -> None:
    first = _line(2.0, 22.0, 0.0, 12.0)
    second = _line(26.0, 48.0, 0.0, 12.0)
    points = np.concatenate((first, second), axis=0)
    result = _run_wire(points)
    assert result.strand_count == 1
    assert result.strands[0].s_span_m > 40.0


def test_short_tree_branch_noise_does_not_become_a_strand() -> None:
    branch = _line(20.0, 26.0, 1.0, 9.0, spacing=0.4)  # short, below span threshold
    result = _run_wire(branch)
    assert result.strand_count == 0


def test_roof_edge_like_line_is_rejected_by_corridor_prior() -> None:
    # A line-like structure perpendicular to the span direction.
    s = np.linspace(2.0, 40.0, 60)
    u = np.linspace(-8.0, 8.0, 60)  # strong transverse extent
    points = np.column_stack((s * 0.2 + 20.0, u, np.full(60, 10.0)))
    result = _run_wire(points)
    assert result.strand_count == 0


def test_dbscan_is_deterministic() -> None:
    xyz = _line(2.0, 48.0, 0.0, 12.0)
    labels_a = _dbscan(xyz, eps=_CFG.wire_dbscan_eps_m, min_samples=_CFG.wire_dbscan_min_samples)
    labels_b = _dbscan(xyz, eps=_CFG.wire_dbscan_eps_m, min_samples=_CFG.wire_dbscan_min_samples)
    assert np.array_equal(labels_a, labels_b)


def test_local_pca_features_on_a_horizontal_line() -> None:
    xyz = _line(0.0, 10.0, 0.0, 5.0, spacing=0.2)
    features = _local_pca_features(xyz, neighbor_count=8)
    assert np.median(features["linearity"]) > 0.8
    assert np.median(features["main_direction_z"]) < 0.2


def test_deterministic_wire_ids_and_distance_ready_output() -> None:
    points = _line(2.0, 48.0, 0.0, 12.0)
    result = _run_wire(points)
    assert result.strands[0].wire_name == "Line01-Span001-Wire01"
    assert result.w1_arrays["segment_distance_ready"].all()
    assert result.distance_ready_segments == 1
    assert np.isfinite(result.w1_arrays["sampled_curve_xyz"]).all()


def test_wire_curve_publishes_scene_local_z_frame() -> None:
    # Scene origin has a non-zero Z.  The wire region stores scene-local XYZ;
    # the reconstructed curve must also be scene-local (z ~ 12, not ~ 112).
    origin_xyz = (100.0, 100.0, 100.0)
    span = SpanRecord(
        line_id=0,
        line_name="Line01",
        span_id=0,
        span_name="Line01-Span001",
        start_pole_candidate_id=0,
        end_pole_candidate_id=1,
        start_xyz_absolute_m=(100.0, 100.0, 0.0),
        end_xyz_absolute_m=(150.0, 100.0, 0.0),
        start_pole_top_z_m=115.0,
        end_pole_top_z_m=115.0,
        length_m=50.0,
        direction_xy=(1.0, 0.0, 0.0),
        half_width_m=9.0,
        halo_m=6.0,
        core_bounds_min_xyz=(100.0, 91.0, 0.0),
        core_bounds_max_xyz=(150.0, 109.0, 20.0),
        halo_bounds_min_xyz=(94.0, 91.0, 0.0),
        halo_bounds_max_xyz=(156.0, 109.0, 20.0),
        quality="CONFIRMED_SPAN",
    )
    # Scene-local wire points: x 2..48, u 0, z 12 (absolute 102..148, 112).
    points = _line(2.0, 48.0, 0.0, 12.0)
    with tempfile.TemporaryDirectory() as td:
        writer = _RawRegionWriter(Path(td) / "region", origin_xyz=origin_xyz)
        writer.append(
            points, np.arange(len(points)), np.zeros(len(points), dtype=np.uint8)
        )
        region = writer.close()
        result = run_span_wire_pipeline(
            span, region, _CFG, origin_xyz=origin_xyz, output_dir=Path(td) / "out"
        )
    assert result.strand_count == 1
    curve = result.strands[0].curve_xyz_local_m
    # Scene-local z: ~12, not the absolute ~112 the double-origin bug produced.
    assert float(np.median(curve[:, 2])) < 20.0
    assert float(np.median(curve[:, 2])) > 8.0
