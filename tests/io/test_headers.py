from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.io import LasIoError, read_las_header_summary
from core.contracts import LasHeaderSummary


def write_synthetic_las(path: Path, point_count: int = 2, with_extra: bool = True) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    if with_extra:
        header.add_extra_dim(laspy.ExtraBytesParams(name="height_above_ground", type=np.float32))

    las = laspy.LasData(header)
    if point_count:
        las.x = np.array([100.0, 101.5][:point_count])
        las.y = np.array([200.0, 202.5][:point_count])
        las.z = np.array([10.0, 12.0][:point_count])
        las.intensity = np.array([10, 20][:point_count], dtype=np.uint16)
        las.classification = np.array([1, 2][:point_count], dtype=np.uint8)
        las.red = np.array([100, 200][:point_count], dtype=np.uint16)
        las.green = np.array([110, 210][:point_count], dtype=np.uint16)
        las.blue = np.array([120, 220][:point_count], dtype=np.uint16)
        if with_extra:
            las.height_above_ground = np.array([1.5, 2.5][:point_count], dtype=np.float32)
    las.write(path)


def test_read_las_14_header_summary(tmp_path: Path) -> None:
    las_path = tmp_path / "sample.las"
    write_synthetic_las(las_path)

    summary = read_las_header_summary(las_path, source_id="raw_sample")

    assert summary.source_id == "raw_sample"
    assert summary.las_version == "1.4"
    assert summary.point_format_id == 7
    assert summary.point_count == 2
    assert summary.point_record_length_bytes > 0
    assert summary.scale == (0.01, 0.01, 0.01)
    assert summary.offset == (100.0, 200.0, 0.0)
    assert summary.min_xyz == (100.0, 200.0, 10.0)
    assert summary.max_xyz == (101.5, 202.5, 12.0)
    assert summary.has_rgb is True
    assert summary.has_intensity is True
    assert summary.has_classification is True
    assert "height_above_ground" in summary.extra_dimensions
    assert summary.estimated_record_memory_bytes == (
        summary.point_count * summary.point_record_length_bytes
    )


def test_header_summary_json_serialization_restores_path(tmp_path: Path) -> None:
    las_path = tmp_path / "sample.las"
    write_synthetic_las(las_path, point_count=1)

    summary = read_las_header_summary(las_path, source_id="raw_sample")
    payload = summary.to_dict()
    json.dumps(payload)
    restored = LasHeaderSummary.from_dict(payload)

    assert restored.source_path == las_path
    assert restored.metadata["warnings"] == []


def test_missing_file_wrong_extension_and_directory_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.las"
    with pytest.raises(LasIoError) as missing_error:
        read_las_header_summary(missing)
    assert missing_error.value.error.code == "LAS_INPUT_NOT_FOUND"

    wrong_extension = tmp_path / "not_las.txt"
    wrong_extension.write_text("not las", encoding="utf-8")
    with pytest.raises(LasIoError) as extension_error:
        read_las_header_summary(wrong_extension)
    assert extension_error.value.error.code == "LAS_INPUT_EXTENSION_INVALID"

    with pytest.raises(LasIoError) as directory_error:
        read_las_header_summary(tmp_path)
    assert directory_error.value.error.code == "LAS_INPUT_IS_DIRECTORY"


def test_empty_las_warns_without_crashing(tmp_path: Path) -> None:
    las_path = tmp_path / "empty.las"
    write_synthetic_las(las_path, point_count=0, with_extra=False)

    summary = read_las_header_summary(las_path, source_id="empty")

    assert summary.point_count == 0
    assert summary.metadata["warnings"] == ["LAS file contains zero points"]


def test_corrupt_las_returns_clear_header_error(tmp_path: Path) -> None:
    las_path = tmp_path / "corrupt.las"
    las_path.write_bytes(b"not a valid las header")

    with pytest.raises(LasIoError) as error:
        read_las_header_summary(las_path)

    assert error.value.error.code == "LAS_HEADER_READ_FAILED"
    assert error.value.error.details["path"] == str(las_path)


def test_header_summary_does_not_call_full_point_read_interfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    las_path = tmp_path / "sample.las"
    write_synthetic_las(las_path)

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full point read must not be called")

    monkeypatch.setattr(laspy, "read", fail_read)
    monkeypatch.setattr(laspy.lasreader.LasReader, "read", fail_read)

    summary = read_las_header_summary(las_path, source_id="raw_sample")

    assert summary.point_count == 2
