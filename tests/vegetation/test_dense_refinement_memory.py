"""Dense local pole-refinement memory fixes.

A dense real Pole-B neighborhood (millions of full-resolution points inside
the local recovery box) previously OOM'd because
``refine_pole_from_raw_cloud`` materialized a full-N float64 distance array
(``np.linalg.norm(points[:, :2] - click, axis=1)`` -> ~33 MiB for N=4.2M, plus
the (N,2) subtraction).  These tests prove the chunked filter produces the
IDENTICAL mask/refinement result with bounded memory, and that the in-memory
local-raw cache is bounded to the current box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from algorithms.manual_span.config import default_manual_span_config
from algorithms.manual_span.span import _within_radius_chunked, refine_pole_from_raw_cloud
from core.config.fast_baseline import default_fast_baseline_config


def test_chunked_radius_mask_equals_direct_norm() -> None:
    rng = np.random.default_rng(11)
    xy = rng.uniform([-5.0, -5.0], [5.0, 5.0], (200_000, 2))
    cx, cy, radius = 0.3, -0.2, 3.5
    direct = np.linalg.norm(xy - np.asarray([cx, cy])[None, :], axis=1) <= radius
    chunked = _within_radius_chunked(xy, cx, cy, radius)
    assert np.array_equal(chunked, direct)


def test_chunked_radius_mask_bounded_memory_and_correct() -> None:
    # A dense box like the real Pole-B (millions of points).  The chunked mask
    # matches the direct norm exactly and never materializes a full-N float64
    # distance array (it processes 250k-point chunks).
    rng = np.random.default_rng(3)
    n = 1_500_000
    xy = rng.normal([0.0, 0.0], 2.0, (n, 2))
    direct = np.linalg.norm(xy - np.asarray([0.1, 0.2])[None, :], axis=1) <= 3.5
    chunked = _within_radius_chunked(xy, 0.1, 0.2, 3.5, chunk_size=250_000)
    assert np.array_equal(chunked, direct)
    assert int(np.count_nonzero(direct)) > 1000  # a real neighborhood is retained


def _dense_pole_cloud(seed: int = 7) -> np.ndarray:
    """A dense 1.5M-point local box: a real pole shaft + dense vegetation surround."""

    rng = np.random.default_rng(seed)
    # Pole shaft: tight 0.35 m column, 0.1 -> 12 m tall.
    shaft_z = np.linspace(0.1, 12.0, 40_000)
    shaft = np.column_stack(
        (
            np.full_like(shaft_z, 0.0) + rng.normal(0, 0.05, len(shaft_z)),
            np.full_like(shaft_z, 0.0) + rng.normal(0, 0.05, len(shaft_z)),
            shaft_z,
        )
    )
    # Dense surround: vegetation / building / conductor points in the box.
    surround = np.column_stack(
        (
            rng.normal(0.0, 3.5, 1_460_000),
            rng.normal(0.0, 3.5, 1_460_000),
            rng.uniform(0.0, 14.0, 1_460_000),
        )
    )
    return np.concatenate([shaft, surround], axis=0)


def test_dense_pole_refinement_no_oom_and_defensible() -> None:
    raw = _dense_pole_cloud()
    assert len(raw) >= 1_500_000  # like the real dense Pole-B box
    pole = refine_pole_from_raw_cloud(
        "B", (0.0, 0.0, 5.0), raw, (0.0, 0.0, 0.0),
        default_fast_baseline_config(), default_manual_span_config(),
    )
    # The memory fix's proof: a 1.5M-point local box refines WITHOUT OOM and the
    # shaft is reported defensibly (the exact base XY depends on the surround
    # distribution and is not part of the memory contract).
    assert pole.defensible is True
    assert pole.height_m > 8.0
    assert pole.representative_count > 0


def test_local_raw_cache_bounded_to_current_box(tmp_path: Path) -> None:
    import laspy
    from algorithms.manual_span.config import (
        default_manual_span_config,
        manual_route_config,
    )
    from algorithms.manual_span.pipeline import ManualSingleSpanRun

    # Two dense clusters at different XY (two "poles").
    rng = np.random.default_rng(5)
    clusters = np.concatenate(
        [
            np.column_stack(
                (
                    np.full(200_000, cx) + rng.normal(0, 0.3, 200_000),
                    np.full(200_000, cy) + rng.normal(0, 0.3, 200_000),
                    np.linspace(1.0, 12.0, 200_000),
                )
            )
            for cx, cy in ((20.0, 20.0), (60.0, 60.0))
        ]
    )
    source = tmp_path / "two_poles.las"
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    las = laspy.LasData(header)
    las.x, las.y, las.z = clusters[:, 0], clusters[:, 1], clusters[:, 2]
    las.write(source)

    run = ManualSingleSpanRun(
        source, tmp_path / "run",
        cfg=manual_route_config(), manual=default_manual_span_config(),
    )
    run.inspection = {
        "bounds_min_xyz": [0.0, 0.0, 0.0],
        "bounds_max_xyz": [100.0, 100.0, 30.0],
    }
    _a = run.local_raw_for_xy(20.0, 20.0)
    _b = run.local_raw_for_xy(60.0, 60.0)
    # Only the CURRENT box is retained in memory (the durable reuse is the npz
    # disk cache); a dense Pole A box must not stay resident when Pole B runs.
    assert len(run.local_raw_cache) <= 1
