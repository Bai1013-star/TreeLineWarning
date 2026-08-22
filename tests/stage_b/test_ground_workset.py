from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
from algorithms.spans.project_adapter import SpanAnalysisInput
from algorithms.vegetation.fast_weighted_preprocess import build_weighted_working_set
from core.stage_b.ground_workset import (
    adaptive_source_chunk_size,
    ensure_ground_workset,
    ground_region_plan,
)


def _span(identifier: str, start_x: float, end_x: float) -> SpanAnalysisInput:
    return SpanAnalysisInput(
        project_id="project-1",
        source_identity={"source_id": "source-1"},
        group_id="G001",
        span_id=identifier,
        from_tower_id=f"{identifier}-from",
        to_tower_id=f"{identifier}-to",
        from_tower_revision=1,
        to_tower_revision=1,
        from_endpoint_geometry={"base_xyz_absolute_m": [start_x, 0.0, 0.0]},
        to_endpoint_geometry={"base_xyz_absolute_m": [end_x, 0.0, 0.0]},
        span_geometry={
            "core_half_width_m": 2.0,
            "longitudinal_halo_m": 1.0,
        },
        science_configuration_digest="a" * 64,
        risk_configuration_digest="b" * 64,
        validated_source_preparation_references=(),
    )


def _source(path: Path) -> None:
    x, y = np.meshgrid(np.arange(-5.0, 36.0), np.arange(-8.0, 9.0))
    count = x.size
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    header.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float64))
    cloud = laspy.LasData(header)
    cloud.x = x.ravel()
    cloud.y = y.ravel()
    cloud.z = np.linspace(0.0, 1.0, count)
    cloud.classification = np.arange(count, dtype=np.uint8) % 5
    cloud.red = np.arange(count, dtype=np.uint16)
    cloud.NormalX = np.ones(count)
    cloud.write(path)


def test_multi_span_union_is_bounded_unique_and_cache_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    _source(source)
    spans = (_span("S001", 0.0, 10.0), _span("S002", 8.0, 20.0))
    identity = {"source_id": "source-1", "path": str(source)}

    status, result = ensure_ground_workset(
        source,
        tmp_path / "cache",
        identity,
        spans,
        smrf_context_m=1.0,
    )

    assert status == "PASS"
    assert 0 < result.workset_point_count <= result.source_point_count
    assert result.active_region_point_count == result.source_point_count
    assert result.workset_domain == "FULL_SOURCE_COARSE_REPRESENTATIVES"
    assert result.source_read_points == result.source_point_count
    original = np.fromfile(result.original_index_path, dtype="<i8")
    assert len(original) == result.workset_point_count
    assert len(np.unique(original)) == len(original)
    with laspy.open(result.workset_las_path) as reader:
        assert int(reader.header.point_count) == result.workset_point_count
        assert reader.header.point_format.id == 6
        assert "red" not in reader.header.point_format.dimension_names
        assert "NormalX" not in reader.header.point_format.dimension_names

    cached_status, cached = ensure_ground_workset(
        source,
        tmp_path / "cache",
        identity,
        tuple(reversed(spans)),
        smrf_context_m=1.0,
    )
    assert cached_status == "SKIPPED_FROM_CACHE"
    assert cached.region_digest == result.region_digest

    Path(result.original_index_path).write_bytes(b"corrupt")
    rebuilt_status, rebuilt = ensure_ground_workset(
        source,
        tmp_path / "cache",
        identity,
        spans,
        smrf_context_m=1.0,
    )
    assert rebuilt_status == "PASS"
    assert Path(rebuilt.original_index_path).stat().st_size == rebuilt.workset_point_count * 8


def test_region_identity_and_adaptive_memory_are_deterministic() -> None:
    first, _ = ground_region_plan((_span("S001", 0.0, 10.0),), smrf_context_m=16.0)
    second, _ = ground_region_plan((_span("S001", 0.0, 11.0),), smrf_context_m=16.0)
    assert first != second
    assert 65_536 <= adaptive_source_chunk_size(
        60, available_memory_bytes=64 * 1024**2
    ) < 100_000
    assert adaptive_source_chunk_size(60, available_memory_bytes=64 * 1024**3) == 1_000_000


def test_incomplete_manifest_is_never_reused(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    _source(source)
    span = _span("S001", 0.0, 10.0)
    identity = {"source_id": "source-1", "path": str(source)}
    _, result = ensure_ground_workset(
        source, tmp_path / "cache", identity, (span,), smrf_context_m=1.0
    )
    manifest = Path(result.workset_las_path).with_name("ground-workset.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "INCOMPLETE"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    status, _ = ensure_ground_workset(
        source, tmp_path / "cache", identity, (span,), smrf_context_m=1.0
    )
    assert status == "PASS"


def test_bounded_workset_exactly_matches_established_weighted_representatives(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dense-source.las"
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([0.01, 0.02, 0.24, 0.26, 1.01, 1.02, 1.03])
    cloud.y = np.asarray([0.01, 0.02, 0.24, 0.26, 1.01, 1.02, 1.03])
    cloud.z = np.asarray([0.01, 0.02, 0.24, 0.26, 1.01, 1.02, 1.03])
    cloud.classification = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.uint8)
    cloud.write(source)
    identity = {"source_id": "source-dense", "path": str(source)}

    _, bounded = ensure_ground_workset(
        source,
        tmp_path / "bounded",
        identity,
        (_span("S001", 0.0, 10.0),),
        smrf_context_m=1.0,
    )
    _, established = build_weighted_working_set(
        source,
        tmp_path / "established",
        voxel_size_m=0.25,
        artifact_stem="reference",
    )
    bounded_las = laspy.read(bounded.workset_las_path)
    established_las = laspy.read(established.representative_las_path)
    bounded_original = np.fromfile(bounded.original_index_path, dtype="<i8")
    with np.load(established.representatives_npz_path, allow_pickle=False) as stream:
        established_original = np.asarray(stream["original_index"], dtype=np.int64)

    assert np.array_equal(bounded_original, established_original)
    assert np.array_equal(
        np.column_stack((bounded_las.X, bounded_las.Y, bounded_las.Z)),
        np.column_stack((established_las.X, established_las.Y, established_las.Z)),
    )
    assert np.array_equal(
        np.asarray(bounded_las.classification),
        np.asarray(established_las.classification),
    )
