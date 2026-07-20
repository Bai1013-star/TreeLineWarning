from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.io import LasIoError, crop_las_xyz, iter_las_chunks, read_las_header_summary
from core.contracts import CancellationToken, CropBounds, CropResult, PipelineProgress


class InMemoryReporter:
    def __init__(self, token: CancellationToken | None = None) -> None:
        self.events: list[PipelineProgress] = []
        self._token = token

    def update(self, progress: PipelineProgress) -> None:
        self.events.append(progress)
        if self._token is not None and progress.processed_points:
            self._token.cancel()


def write_las(
    path: Path,
    *,
    point_count: int = 7,
    point_format: int = 7,
    with_normals: bool = True,
) -> None:
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    if with_normals:
        header.add_extra_dims(
            [
                laspy.ExtraBytesParams(name="NormalX", type=np.float32),
                laspy.ExtraBytesParams(name="NormalY", type=np.float32),
                laspy.ExtraBytesParams(name="NormalZ", type=np.float32),
            ]
        )

    las = laspy.LasData(header)
    indexes = np.arange(point_count)
    las.x = 100.0 + indexes.astype(np.float64)
    las.y = 200.0 + indexes.astype(np.float64)
    las.z = indexes.astype(np.float64)
    las.intensity = (indexes + 10).astype(np.uint16)
    las.classification = (indexes % 5).astype(np.uint8)
    las.user_data = (indexes + 1).astype(np.uint8)
    if "scan_angle" in tuple(las.point_format.dimension_names):
        las.scan_angle = indexes.astype(np.int16)
    las.point_source_id = (indexes + 20).astype(np.uint16)
    if "gps_time" in tuple(las.point_format.dimension_names):
        las.gps_time = indexes.astype(np.float64) + 0.5
    if {"red", "green", "blue"}.issubset(set(las.point_format.dimension_names)):
        las.red = (indexes + 100).astype(np.uint16)
        las.green = (indexes + 200).astype(np.uint16)
        las.blue = (indexes + 300).astype(np.uint16)
    if with_normals:
        las.NormalX = np.full(point_count, 1.0, dtype=np.float32)
        las.NormalY = np.full(point_count, 0.0, dtype=np.float32)
        las.NormalZ = np.full(point_count, 0.5, dtype=np.float32)
    las.write(path)


def first_output_chunk(path: Path) -> object:
    with laspy.open(path, mode="r") as reader:
        return next(reader.chunk_iterator(100))


def test_chunk_counts_indexes_and_last_flag_for_small_chunk_size(tmp_path: Path) -> None:
    las_path = tmp_path / "source.las"
    write_las(las_path, point_count=7)

    chunks = list(iter_las_chunks(las_path, source_id="source", chunk_size=3))

    assert [chunk.point_count for chunk in chunks] == [3, 3, 1]
    assert [chunk.is_last for chunk in chunks] == [False, False, True]
    all_indexes = np.concatenate([chunk.original_index for chunk in chunks])
    assert all_indexes.tolist() == list(range(7))
    assert len(set(all_indexes.tolist())) == 7
    assert chunks[0].start_index == 0
    assert chunks[1].start_index == 3
    assert chunks[2].start_index == 6


def test_chunk_size_larger_than_total_points_yields_one_last_chunk(tmp_path: Path) -> None:
    las_path = tmp_path / "source.las"
    write_las(las_path, point_count=4)

    chunks = list(iter_las_chunks(las_path, source_id="source", chunk_size=100))

    assert len(chunks) == 1
    assert chunks[0].point_count == 4
    assert chunks[0].is_last is True


def test_chunk_attributes_include_only_existing_fields(tmp_path: Path) -> None:
    las_path = tmp_path / "source.las"
    write_las(las_path, point_count=3, point_format=1, with_normals=False)

    chunk = next(iter(iter_las_chunks(las_path, source_id="source", chunk_size=3)))

    assert "intensity" in chunk.attributes
    assert "classification" in chunk.attributes
    assert "gps_time" in chunk.attributes
    assert "red" not in chunk.attributes
    assert "NormalX" not in chunk.attributes


def test_chunk_cancellation_and_progress_callback(tmp_path: Path) -> None:
    las_path = tmp_path / "source.las"
    write_las(las_path, point_count=7)
    token = CancellationToken()
    reporter = InMemoryReporter()
    iterator = iter_las_chunks(
        las_path,
        source_id="source",
        chunk_size=3,
        reporter=reporter,
        cancellation_token=token,
    )

    first = next(iterator)
    token.cancel()
    remaining = list(iterator)

    assert first.point_count == 3
    assert remaining == []
    assert [event.status for event in reporter.events] == ["running", "cancelled"]


def test_chunk_cancellation_before_iteration_stops(tmp_path: Path) -> None:
    las_path = tmp_path / "source.las"
    write_las(las_path, point_count=4)
    token = CancellationToken()
    token.cancel()

    chunks = list(iter_las_chunks(las_path, source_id="source", cancellation_token=token))

    assert chunks == []


def test_crop_xyz_inclusive_bounds_and_output_count(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "cropped.las"
    write_las(source, point_count=7)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))

    result = crop_las_xyz(source, output, bounds, source_id="source", chunk_size=2)

    assert result.input_point_count == 7
    assert result.output_point_count == 3
    assert output.exists()
    summary = read_las_header_summary(output, source_id="cropped")
    assert summary.point_count == 3
    assert summary.min_xyz == (102.0, 202.0, 2.0)
    assert summary.max_xyz == (104.0, 204.0, 4.0)


def test_crop_local_bounds_use_origin(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "local_crop.las"
    write_las(source, point_count=5)
    bounds = CropBounds(
        min_xyz=(1.0, 1.0, 1.0),
        max_xyz=(2.0, 2.0, 2.0),
        coordinate_mode="local",
        origin_xyz=(100.0, 200.0, 0.0),
    )

    result = crop_las_xyz(source, output, bounds, source_id="source", chunk_size=2)

    assert result.output_point_count == 2


def test_crop_preserves_header_and_approved_attributes(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "cropped.las"
    write_las(source, point_count=5)

    crop_las_xyz(
        source,
        output,
        CropBounds(min_xyz=(100.0, 200.0, 0.0), max_xyz=(104.0, 204.0, 4.0)),
        source_id="source",
        chunk_size=2,
    )
    source_summary = read_las_header_summary(source, source_id="source")
    output_summary = read_las_header_summary(output, source_id="output")
    point_record = first_output_chunk(output)

    assert output_summary.point_format_id == source_summary.point_format_id
    assert output_summary.scale == source_summary.scale
    assert output_summary.offset == source_summary.offset
    assert output_summary.has_rgb is True
    assert output_summary.has_intensity is True
    assert output_summary.has_classification is True
    assert {"NormalX", "NormalY", "NormalZ"}.issubset(set(output_summary.extra_dimensions))
    assert np.asarray(point_record.red).tolist() == [100, 101, 102, 103, 104]
    assert np.asarray(point_record.intensity).tolist() == [10, 11, 12, 13, 14]
    assert np.asarray(point_record.classification).tolist() == [0, 1, 2, 3, 4]
    assert np.asarray(point_record.NormalX).tolist() == [1.0] * 5


def test_crop_rejects_data_raw_output_source_overwrite_and_existing_output(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    write_las(source, point_count=3)
    bounds = CropBounds(min_xyz=(100.0, 200.0, 0.0), max_xyz=(101.0, 201.0, 1.0))

    with pytest.raises(LasIoError) as raw_error:
        crop_las_xyz(
            source,
            raw / "bad.las",
            bounds,
            source_id="source",
            project_root=project,
        )
    assert raw_error.value.error.code == "LAS_CROP_OUTPUT_FORBIDDEN"

    with pytest.raises(LasIoError) as source_error:
        crop_las_xyz(source, source, bounds, source_id="source", project_root=project)
    assert source_error.value.error.code == "LAS_CROP_OUTPUT_IS_SOURCE"

    existing = samples / "existing.las"
    existing.write_bytes(b"exists")
    with pytest.raises(LasIoError) as exists_error:
        crop_las_xyz(source, existing, bounds, source_id="source", project_root=project)
    assert exists_error.value.error.code == "LAS_CROP_OUTPUT_EXISTS"


def test_empty_crop_writes_empty_las_and_serializes_result(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "empty.las"
    write_las(source, point_count=3)
    bounds = CropBounds(min_xyz=(900.0, 900.0, 900.0), max_xyz=(901.0, 901.0, 901.0))

    result = crop_las_xyz(source, output, bounds, source_id="source", chunk_size=2)
    payload = result.to_dict()
    json.dumps(payload)
    restored = CropResult.from_dict(payload)

    assert restored.output_point_count == 0
    assert restored.warnings == ("Crop selected zero points",)
    assert read_las_header_summary(output, source_id="empty").point_count == 0


def test_crop_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "broken.las"
    write_las(source, point_count=3)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("forced write failure")

    monkeypatch.setattr(laspy.laswriter.LasWriter, "write_points", fail_write)

    with pytest.raises(RuntimeError, match="forced write failure"):
        crop_las_xyz(
            source,
            output,
            CropBounds(min_xyz=(100.0, 200.0, 0.0), max_xyz=(101.0, 201.0, 1.0)),
            source_id="source",
            chunk_size=2,
        )

    assert not output.exists()
    assert not (tmp_path / "broken.tmp.las").exists()


def test_crop_cancellation_stops_and_removes_temporary_file(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "cancelled.las"
    write_las(source, point_count=7)
    token = CancellationToken()
    reporter = InMemoryReporter(token)

    with pytest.raises(LasIoError) as error:
        crop_las_xyz(
            source,
            output,
            CropBounds(min_xyz=(100.0, 200.0, 0.0), max_xyz=(106.0, 206.0, 6.0)),
            source_id="source",
            chunk_size=2,
            reporter=reporter,
            cancellation_token=token,
        )

    assert error.value.error.code == "LAS_CROP_CANCELLED"
    assert reporter.events
    assert not output.exists()
    assert not (tmp_path / "cancelled.tmp.las").exists()


def test_no_full_point_read_interfaces_are_called(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "cropped.las"
    write_las(source, point_count=4)

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full point read must not be called")

    monkeypatch.setattr(laspy, "read", fail_read)
    monkeypatch.setattr(laspy.lasreader.LasReader, "read", fail_read)

    chunks = list(iter_las_chunks(source, source_id="source", chunk_size=2))
    result = crop_las_xyz(
        source,
        output,
        CropBounds(min_xyz=(100.0, 200.0, 0.0), max_xyz=(103.0, 203.0, 3.0)),
        source_id="source",
        chunk_size=2,
    )

    assert len(chunks) == 2
    assert result.output_point_count == 4
