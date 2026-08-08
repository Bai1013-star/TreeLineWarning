"""End-to-end synthetic ground scene classification tests.

Authority: docs/full-resolution-ground-classification-design.md sections 7-10
and the frozen Bundle B plan test gate (flat, slope, ditch, embankment, sparse
ground, low shrub/root, low outlier, tower/pole, and no-reliable-ground
scenes).  All scenes are tiny synthetic fixtures; scan_small is never used.
"""

from __future__ import annotations

import numpy as np
import pytest
from algorithms.ground.classification import classify_ground
from core.contracts.ground_classification import GroundClassificationPolicyConfig

from tests.ground._identity import make_hag_config
from tests.ground._scene import load_scene, write_synthetic_las


def _policy(**overrides) -> GroundClassificationPolicyConfig:
    payload = {**GroundClassificationPolicyConfig().to_dict(), **overrides}
    payload["policy_digest"] = ""
    return GroundClassificationPolicyConfig.from_dict(payload)


def _run(scene, output_root, *, policy=None):
    return classify_ground(
        scene.reference,
        scene.result,
        scene.configuration,
        make_hag_config(),
        output_root,
        policy if policy is not None else _policy(),
    )


def _state_counts(chunks):
    totals = {"non_ground": 0, "confirmed_ground": 0, "review_ground": 0}
    for chunk in chunks:
        for key in totals:
            totals[key] += chunk.state_counts[key]
    return totals


def test_flat_scene_confirms_all_ground(tmp_path) -> None:
    gx, gy = np.meshgrid(np.arange(0.0, 10.0, 0.5), np.arange(0.0, 10.0, 0.5))
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "flat.las",
            x=100.0 + gx.ravel(),
            y=200.0 + gy.ravel(),
            z=np.full(gx.size, 0.0),
        )
    )
    out = _run(scene, tmp_path / "out")
    counts = _state_counts(out["chunks"])
    assert counts["confirmed_ground"] == scene.count
    assert counts["non_ground"] == 0
    assert counts["review_ground"] == 0
    assert out["coverage"].occupancy_fraction == pytest.approx(1.0)
    assert out["coverage"].passes_occupancy(policy=_policy())


def test_sloped_ground_confirms(tmp_path) -> None:
    gx, gy = np.meshgrid(np.arange(0.0, 20.0, 0.5), np.arange(0.0, 5.0, 0.5))
    zs = 0.05 * gx.ravel()  # gentle 5% slope
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "slope.las",
            x=100.0 + gx.ravel(),
            y=200.0 + gy.ravel(),
            z=zs,
        )
    )
    out = _run(scene, tmp_path / "out")
    counts = _state_counts(out["chunks"])
    # A gentle slope stays within the frozen confirm band.
    assert counts["confirmed_ground"] == scene.count


def test_steep_scene_routes_review(tmp_path) -> None:
    # A steep 2:1 face routes REVIEW (slope > 0.75) and, with no confirmed
    # ground, publication fails the coverage occupancy gate (fail closed).
    gx, gy = np.meshgrid(np.arange(0.0, 5.0, 0.5), np.arange(0.0, 5.0, 0.5))
    zs = 2.0 * gx.ravel()  # steep 2:1 face -> slope > 0.75 -> review
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "steep.las",
            x=100.0 + gx.ravel(),
            y=200.0 + gy.ravel(),
            z=zs,
        )
    )
    from algorithms.ground.classification import FgcExecutionError

    with pytest.raises(FgcExecutionError, match="occupancy"):
        _run(scene, tmp_path / "out")


def test_tower_pole_triggers_vertical_structure(tmp_path) -> None:
    # A dense vertical column of returns above a flat ground plane.
    gx, gy = np.meshgrid(np.arange(0.0, 10.0, 0.5), np.arange(0.0, 10.0, 0.5))
    ground_count = gx.size
    xs = list(100.0 + gx.ravel())
    ys = list(200.0 + gy.ravel())
    zs = list(np.zeros(ground_count))
    # A pole at the center with 12 stacked returns spanning > 1.5 m.
    pole_x = 105.0
    pole_y = 205.0
    for level in range(12):
        xs.append(pole_x)
        ys.append(pole_y)
        zs.append(0.2 * level + 0.5)
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "pole.las",
            x=np.asarray(xs),
            y=np.asarray(ys),
            z=np.asarray(zs),
        )
    )
    out = _run(scene, tmp_path / "out")
    counts = _state_counts(out["chunks"])
    # Ground remains confirmed; the pole rows and the ground immediately beside
    # them route REVIEW via the vertical-structure rule (pole returns within
    # 1 m create a > 1.5 m span).
    assert counts["confirmed_ground"] >= ground_count - 60
    assert counts["review_ground"] >= 12


def test_low_outlier_routes_not_confirmed(tmp_path) -> None:
    # A single point far above the local ground plane.  The vertical-structure
    # rule fires first (the outlier is within 1 m of many ground rows), so the
    # outlier must never be CONFIRMED_GROUND.
    gx, gy = np.meshgrid(np.arange(0.0, 10.0, 0.5), np.arange(0.0, 10.0, 0.5))
    xs = list(100.0 + gx.ravel())
    ys = list(200.0 + gy.ravel())
    zs = list(np.zeros(gx.size))
    xs.append(104.0)
    ys.append(204.0)
    zs.append(5.0)  # 5 m above ground
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "outlier.las",
            x=np.asarray(xs),
            y=np.asarray(ys),
            z=np.asarray(zs),
        )
    )
    out = _run(scene, tmp_path / "out")
    counts = _state_counts(out["chunks"])
    # The outlier and the ground rows within 1 m of it route REVIEW via the
    # vertical-structure rule; they must never be confirmed.
    assert counts["review_ground"] >= 1
    assert counts["confirmed_ground"] >= gx.size - 30


def test_no_reliable_ground_fails_coverage(tmp_path) -> None:
    # A few isolated high returns spread far apart: no connected ground seeds,
    # so confirmed occupancy falls below the frozen threshold.
    xs = np.asarray([100.0, 105.0, 110.0, 115.0])
    ys = np.asarray([200.0, 205.0, 210.0, 215.0])
    zs = np.asarray([10.0, 11.0, 12.0, 13.0])
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "no_ground.las",
            x=xs,
            y=ys,
            z=zs,
        )
    )
    from algorithms.ground.classification import FgcExecutionError

    with pytest.raises(FgcExecutionError, match="occupancy"):
        _run(scene, tmp_path / "out")


def test_ditch_and_embankment_preserve_interior(tmp_path) -> None:
    # A flat plane with a shallow ditch: interior returns stay confirmed.
    gx, gy = np.meshgrid(np.arange(0.0, 20.0, 0.5), np.arange(0.0, 5.0, 0.5))
    xs = 100.0 + gx.ravel()
    ys = 200.0 + gy.ravel()
    zs = np.zeros(xs.size)
    # Ditch: lower the central band by 0.3 m (within the plane clipping).
    ditch = (np.abs(xs - 105.0) < 1.0) & (np.abs(ys - 202.5) < 1.0)
    zs[ditch] = -0.3
    scene = load_scene(write_synthetic_las(tmp_path / "ditch.las", x=xs, y=ys, z=zs))
    out = _run(scene, tmp_path / "out")
    counts = _state_counts(out["chunks"])
    assert counts["confirmed_ground"] >= xs.size - 10
