from pathlib import Path

import laspy
import numpy as np
from desktop.tower_viewer import overview, refine


def _write_las(path: Path, xyz: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.red = np.arange(len(xyz), dtype=np.uint16) * 1000 + 1000
    cloud.green = np.full(len(xyz), 30000, dtype=np.uint16)
    cloud.blue = np.full(len(xyz), 50000, dtype=np.uint16)
    cloud.intensity = np.arange(len(xyz), dtype=np.uint16) * 100
    cloud.write(path)


def test_overview_is_true_xyz_and_reuses_source_bound_display_cache(tmp_path: Path) -> None:
    source = tmp_path / "真实杆塔.las"
    points = np.asarray([[i, i * 2, i * 3 + 10] for i in range(12)], dtype=float)
    _write_las(source, points)
    cache = tmp_path / "viewer_cache"
    identity = {"source_id": "source-a", "source_file_sha256": "a" * 64}

    first = overview(source, maximum=5, cache_dir=cache, source_identity=identity)
    second = overview(source, maximum=5, cache_dir=cache, source_identity=identity)

    assert first["display_point_count"] == 4
    assert len(first["xyz"][0]) == 3
    assert first["xyz"][0][2] == 10
    assert first["viewer_cache"] == "BUILT"
    assert second["viewer_cache"] == "HIT"
    assert second["xyz"] == first["xyz"]
    assert second["diagnostic_timings"]["overview_build_ms"] == 0


def test_overview_cache_identity_invalidates_and_corruption_rebuilds(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    _write_las(source, np.asarray([[0, 0, 1], [1, 1, 2], [2, 2, 3]], dtype=float))
    cache = tmp_path / "cache"

    first = overview(source, cache_dir=cache, source_identity={"source_id": "A"})
    other = overview(source, cache_dir=cache, source_identity={"source_id": "B"})
    assert (
        first["diagnostic_timings"]["overview_cache_identity"]
        != other["diagnostic_timings"]["overview_cache_identity"]
    )
    assert other["viewer_cache"] == "BUILT"

    digest = other["diagnostic_timings"]["overview_cache_identity"]
    target = next(cache.glob(f"overview-{digest}.npz"))
    target.write_bytes(b"broken viewer cache")
    rebuilt = overview(source, cache_dir=cache, source_identity={"source_id": "B"})
    assert rebuilt["viewer_cache"] == "BUILT"
    assert rebuilt["display_point_count"] == 3


def test_true_3d_seed_refines_to_real_source_point_and_uses_chunk_index(tmp_path: Path) -> None:
    source = tmp_path / "tower.las"
    points = np.asarray(
        [[0, 0, 0], [0.1, 0.1, 5], [0.2, 0.2, 12], [0.3, 0.3, 20], [30, 30, 1]],
        dtype=float,
    )
    _write_las(source, points)
    cache = tmp_path / "cache"
    identity = {"source_id": "tower-source"}
    overview(source, cache_dir=cache, source_identity=identity)

    result = refine(
        source, 0.21, 0.21, z=11.8, radius_m=2, cache_dir=cache, source_identity=identity
    )

    assert result["tower"]["picked_source_xyz"] == [0.2, 0.2, 12.0]
    assert result["tower"]["local_point_count"] == 4
    assert result["diagnostic_timings"]["local_index_used"] is True
    assert result["diagnostic_timings"]["local_chunks_read"] == 1
    assert len(result["local"]["xyz"][0]) == 3
