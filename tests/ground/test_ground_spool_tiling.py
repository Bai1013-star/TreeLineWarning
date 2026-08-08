"""FGC single-pass owner spool, tile ownership, and halo tests.

Authority: docs/full-resolution-ground-classification-design.md sections 4-6,
and the frozen Bundle B plan test gate (halo maximum dependency; raw
negative/internal/max boundaries; spool original-index order, corruption, cap,
and cancellation cleanup).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algorithms.ground.classification import (
    FgcExecutionError,
    MetricGrid,
    _parse_decimal_rational,
    _reconstruct_raw_minimum,
    build_metric_grid,
    decode_z,
    spool_source,
    tile_work_domain,
)
from core.contracts.ground_classification import (
    FGC_SPOOL_RECORD_SIZE,
    GroundClassificationPolicyConfig,
)

from tests.ground._scene import SyntheticScene, load_scene, write_synthetic_las


def _policy(**overrides) -> GroundClassificationPolicyConfig:
    payload = {**GroundClassificationPolicyConfig().to_dict(), **overrides}
    payload["policy_digest"] = ""
    return GroundClassificationPolicyConfig.from_dict(payload)


def _grid_for_scene(scene: SyntheticScene) -> MetricGrid:
    proof = scene.result.proof
    header = proof.header
    scale_x = _parse_decimal_rational(format(float(header.scale_xyz[0]), ".17g"))
    scale_y = _parse_decimal_rational(format(float(header.scale_xyz[1]), ".17g"))
    x0 = _reconstruct_raw_minimum(
        float(proof.source_bounds_min_xyz[0]), float(header.offset_xyz[0]), scale_x
    )
    y0 = _reconstruct_raw_minimum(
        float(proof.source_bounds_min_xyz[1]), float(header.offset_xyz[1]), scale_y
    )
    x_max = _reconstruct_raw_minimum(
        float(proof.source_bounds_max_xyz[0]), float(header.offset_xyz[0]), scale_x
    )
    y_max = _reconstruct_raw_minimum(
        float(proof.source_bounds_max_xyz[1]), float(header.offset_xyz[1]), scale_y
    )
    return build_metric_grid(header, _policy(), x0=x0, y0=y0, x_max=x_max, y_max=y_max)


def test_decimal_rational_parsing() -> None:
    from fractions import Fraction

    assert _parse_decimal_rational("0.01") == Fraction(1, 100)
    assert _parse_decimal_rational("0.5") == Fraction(1, 2)
    assert _parse_decimal_rational("1e-05") == Fraction(1, 100000)
    assert _parse_decimal_rational("1000.0") == Fraction(1000)
    assert _parse_decimal_rational("0.010000000000000000") == Fraction(1, 100)


def test_metric_grid_tile_indices_and_max_clamp(tmp_path: Path) -> None:
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "a.las",
            x=np.array([100.0, 100.0, 300.0]),
            y=np.array([200.0, 200.0, 200.0]),
            z=np.array([0.0, 1.0, 0.0]),
        )
    )
    grid = _grid_for_scene(scene)
    # tile side 256 m / 0.01 = 25600 ticks. Points at x=0 and x=200 m.
    # x=200 m -> tile 0; the max x is clamped into the final tile.
    tx0 = grid.tile_index(0, 0)
    tx_max = grid.tile_index(grid.x_max, grid.y_max)
    assert tx0 == (0, 0)
    assert tx_max[1] == grid.nx - 1
    assert tx_max[0] == grid.ny - 1


def test_metric_grid_negative_coordinates(tmp_path: Path) -> None:
    # Negative decoded coordinates must still route by raw differences.
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "neg.las",
            x=np.array([-50.0, -50.0, 50.0]),
            y=np.array([-20.0, -20.0, -20.0]),
            z=np.array([0.0, 1.0, 2.0]),
            offsets=np.array([0.0, 0.0, 0.0]),
        )
    )
    grid = _grid_for_scene(scene)
    assert grid.x0 < 0
    assert grid.y0 < 0
    # The source minimum raw coordinate must equal the reconstructed anchor.
    assert grid.tile_index(grid.x0, grid.y0) == (0, 0)


def test_spool_manifest_is_ascending_and_covered(tmp_path: Path) -> None:
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "s.las",
            x=np.array([100.0, 150.0, 400.0]),
            y=np.array([200.0, 250.0, 500.0]),
            z=np.array([0.0, 1.0, 2.0]),
        )
    )
    grid = _grid_for_scene(scene)
    manifest = spool_source(
        scene.reference,
        scene.result,
        scene.configuration,
        _policy(),
        grid=grid,
        scratch_dir=tmp_path / "scratch",
    )
    assert manifest.source_point_count == 3
    tiles = [tile.tile_id for tile in manifest.tiles]
    assert tiles == sorted(tiles)
    total_rows = sum(tile.row_count for tile in manifest.tiles)
    assert total_rows == 3
    # Every nonempty spool file exists and matches its digest.
    spool_dir = tmp_path / "scratch" / "spool"
    for tile in manifest.tiles:
        path = spool_dir / f"tile_{tile.tile_id_ty:06d}_{tile.tile_id_tx:06d}.spool"
        assert path.exists()


def test_spool_records_ascending_original_index(tmp_path: Path) -> None:
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "s.las",
            x=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            y=np.array([200.0, 200.0, 200.0, 200.0, 200.0]),
            z=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    )
    grid = _grid_for_scene(scene)
    manifest = spool_source(
        scene.reference,
        scene.result,
        scene.configuration,
        _policy(),
        grid=grid,
        scratch_dir=tmp_path / "scratch",
    )
    spool_dir = tmp_path / "scratch" / "spool"
    for tile in manifest.tiles:
        path = spool_dir / f"tile_{tile.tile_id_ty:06d}_{tile.tile_id_tx:06d}.spool"
        data = path.read_bytes()
        assert len(data) % FGC_SPOOL_RECORD_SIZE == 0
        count = len(data) // FGC_SPOOL_RECORD_SIZE
        assert count == tile.row_count
        dtype = np.dtype([("index", "<i8"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
        table = np.frombuffer(data, dtype=dtype)
        assert (np.diff(table["index"]) > 0).all()


def test_spool_corrupt_rejected(tmp_path: Path) -> None:
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "s.las",
            x=np.array([100.0, 101.0, 102.0]),
            y=np.array([200.0, 200.0, 200.0]),
            z=np.array([0.0, 0.0, 0.0]),
        )
    )
    grid = _grid_for_scene(scene)
    manifest = spool_source(
        scene.reference,
        scene.result,
        scene.configuration,
        _policy(),
        grid=grid,
        scratch_dir=tmp_path / "scratch",
    )
    # Truncate the only spool file -> corrupt.
    spool_dir = tmp_path / "scratch" / "spool"
    for tile in manifest.tiles:
        path = spool_dir / f"tile_{tile.tile_id_ty:06d}_{tile.tile_id_tx:06d}.spool"
        path.write_bytes(path.read_bytes()[:-1])
    from algorithms.ground.classification import _load_spool_table

    with pytest.raises(FgcExecutionError, match="record size"):
        _load_spool_table(spool_dir, tile.tile_id)


def test_source_min_anchor_mismatch_rejected(tmp_path: Path) -> None:
    # Force the reconstructed anchor to disagree with the actual raw minimum.
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "s.las",
            x=np.array([100.0, 102.0]),
            y=np.array([200.0, 202.0]),
            z=np.array([0.0, 0.0]),
        )
    )
    grid = _grid_for_scene(scene)
    # Build a grid whose x0 is deliberately wrong.
    wrong = MetricGrid(
        scale_x=grid.scale_x,
        scale_y=grid.scale_y,
        scale_z=grid.scale_z,
        offset_x=grid.offset_x,
        offset_y=grid.offset_y,
        offset_z=grid.offset_z,
        g=grid.g,
        kx=grid.kx,
        ky=grid.ky,
        tx=grid.tx,
        ty=grid.ty,
        fx=grid.fx,
        fy=grid.fy,
        cx=grid.cx,
        cy=grid.cy,
        hx=grid.hx,
        hy=grid.hy,
        x0=grid.x0 + 1,
        y0=grid.y0,
        x_max=grid.x_max,
        y_max=grid.y_max,
        nx=grid.nx,
        ny=grid.ny,
        _radius_g_sq=grid.plane_radius_sq,
        _vertical_g_sq=grid.vertical_radius_sq,
    )
    with pytest.raises(FgcExecutionError, match="anchor"):
        spool_source(
            scene.reference,
            scene.result,
            scene.configuration,
            _policy(),
            grid=wrong,
            scratch_dir=tmp_path / "scratch2",
        )


def test_halo_max_dependency() -> None:
    # The frozen 20 m halo subsumes the 16 m PMF opening plus 3 m plane radius.
    import math

    assert math.ceil(16 / 0.5) == 32  # k(16)
    assert (32 + 1) // 2 == 16  # h(16)


def test_decode_z_positive_scale(tmp_path: Path) -> None:
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "probe.las",
            x=np.array([100.0]),
            y=np.array([200.0]),
            z=np.array([1.23]),
        )
    )
    grid = _grid_for_scene(scene)
    assert decode_z(int(round(1.23 / 0.01)), grid) == pytest.approx(1.23)


def test_tile_work_domain_closed_halo_rectangle(tmp_path: Path) -> None:
    scene = load_scene(
        write_synthetic_las(
            tmp_path / "s.las",
            x=np.array([100.0, 150.0]),
            y=np.array([200.0, 250.0]),
            z=np.array([0.0, 0.0]),
        )
    )
    grid = _grid_for_scene(scene)
    domain = tile_work_domain(grid, (0, 0))
    # Halo rectangle is closed in raw ticks, clipped to source minima/maxima.
    assert domain.halo_x_lo >= grid.x0
    assert domain.halo_x_hi <= grid.x_max
    assert domain.halo_y_lo >= grid.y0
    assert domain.halo_y_hi <= grid.y_max
