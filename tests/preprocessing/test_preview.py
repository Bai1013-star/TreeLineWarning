from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.io import LasIoError
from algorithms.preprocessing import preview as preview_module
from algorithms.preprocessing.preview import build_pointcloud_preview
from core.contracts import CancellationToken, PipelineProgress


class CancellingReporter:
    def __init__(self) -> None:
        self.events: list[PipelineProgress] = []

    def update(self, progress: PipelineProgress) -> None:
        self.events.append(progress)


class CancelAfterFirstReporter:
    def __init__(self, token: CancellationToken) -> None:
        self.events: list[PipelineProgress] = []
        self.token = token

    def update(self, progress: PipelineProgress) -> None:
        self.events.append(progress)
        self.token.cancel()


class FailingMemoryProcess:
    def memory_info(self) -> object:
        raise RuntimeError("RSS probe unavailable")


def write_las(
    path: Path,
    *,
    point_count: int,
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
    if {"red", "green", "blue"}.issubset(set(las.point_format.dimension_names)):
        las.red = (indexes + 100).astype(np.uint16)
        las.green = (indexes + 200).astype(np.uint16)
        las.blue = (indexes + 300).astype(np.uint16)
    if with_normals:
        las.NormalX = np.full(point_count, 1.0, dtype=np.float32)
        las.NormalY = np.full(point_count, 0.0, dtype=np.float32)
        las.NormalZ = np.full(point_count, 0.5, dtype=np.float32)
    las.write(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def test_preview_continues_when_process_constructor_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_psutil = types.ModuleType("psutil")

    def fail_process() -> object:
        raise RuntimeError("process probe unavailable")

    fake_psutil.Process = fail_process  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert preview_module._current_process() is None


@pytest.mark.parametrize(
    "unavailable_process",
    [None, FailingMemoryProcess()],
    ids=["missing-process", "failing-rss-probe"],
)
def test_preview_preserves_sample_data_when_rss_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unavailable_process: object | None,
) -> None:
    source = tmp_path / "source.las"
    baseline_output = tmp_path / "baseline"
    unavailable_output = tmp_path / "unavailable"
    write_las(source, point_count=10)

    class FixedMemoryProcess:
        def memory_info(self) -> object:
            return types.SimpleNamespace(rss=1234)

    monkeypatch.setattr(preview_module, "_current_process", lambda: FixedMemoryProcess())
    baseline = build_pointcloud_preview(
        source,
        baseline_output,
        source_id="source",
        target_preview_points=4,
        max_preview_points=4,
        chunk_size=3,
        generate_html=False,
    )

    monkeypatch.setattr(preview_module, "_current_process", lambda: unavailable_process)
    unavailable = build_pointcloud_preview(
        source,
        unavailable_output,
        source_id="source",
        target_preview_points=4,
        max_preview_points=4,
        chunk_size=3,
        generate_html=False,
    )

    baseline_data = load_npz(baseline.npz_path)
    unavailable_data = load_npz(unavailable.npz_path)
    assert baseline_data.keys() == unavailable_data.keys()
    for name in baseline_data:
        assert np.array_equal(baseline_data[name], unavailable_data[name])
    assert unavailable.metadata["memory"]["rss_start_bytes"] is None
    assert unavailable.metadata["memory"]["rss_peak_bytes"] is None


def test_preview_streams_multiple_chunks_and_limits_output(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=10)

    result = build_pointcloud_preview(
        source,
        output_dir,
        source_id="source",
        target_preview_points=4,
        max_preview_points=4,
        chunk_size=3,
        generate_html=False,
    )
    data = load_npz(result.npz_path)

    assert result.metadata["preview_point_count"] == 4
    assert len(data["xyz"]) == 4
    assert data["original_index"].tolist() == [0, 3, 6, 9]
    assert result.metadata["downsampling"]["stride"] == 3


def test_preview_deterministic_for_same_parameters(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    write_las(source, point_count=12)

    result_a = build_pointcloud_preview(
        source,
        output_a,
        source_id="source",
        target_preview_points=5,
        max_preview_points=5,
        chunk_size=4,
        generate_html=False,
    )
    result_b = build_pointcloud_preview(
        source,
        output_b,
        source_id="source",
        target_preview_points=5,
        max_preview_points=5,
        chunk_size=4,
        generate_html=False,
    )

    assert np.array_equal(load_npz(result_a.npz_path)["xyz"], load_npz(result_b.npz_path)["xyz"])
    assert np.array_equal(
        load_npz(result_a.npz_path)["original_index"],
        load_npz(result_b.npz_path)["original_index"],
    )


def test_preview_preserves_rgb_intensity_classification_and_normals(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=5)

    result = build_pointcloud_preview(
        source,
        output_dir,
        source_id="source",
        target_preview_points=5,
        max_preview_points=5,
        chunk_size=2,
        generate_html=False,
    )
    data = load_npz(result.npz_path)

    assert data["rgb"].tolist() == [
        [100, 200, 300],
        [101, 201, 301],
        [102, 202, 302],
        [103, 203, 303],
        [104, 204, 304],
    ]
    assert data["intensity"].tolist() == [10, 11, 12, 13, 14]
    assert data["classification"].tolist() == [0, 1, 2, 3, 4]
    assert data["NormalX"].tolist() == [1.0] * 5


def test_preview_handles_missing_optional_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=4, point_format=1, with_normals=False)

    result = build_pointcloud_preview(
        source,
        output_dir,
        source_id="source",
        target_preview_points=4,
        max_preview_points=4,
        generate_html=False,
    )
    data = load_npz(result.npz_path)

    assert "intensity" in data
    assert "classification" in data
    assert "rgb" not in data
    assert "NormalX" not in data


def test_preview_local_coordinates_and_origin(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=3)

    result = build_pointcloud_preview(
        source,
        output_dir,
        source_id="source",
        target_preview_points=3,
        max_preview_points=3,
        generate_html=False,
    )
    data = load_npz(result.npz_path)

    assert data["origin_xyz"].tolist() == [100.0, 200.0, 0.0]
    assert data["local_xyz"][0].tolist() == [0.0, 0.0, 0.0]
    assert np.array_equal(data["xyz"], data["local_xyz"] + data["origin_xyz"])


def test_preview_metadata_json_serializable_and_npz_reloadable(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=3)

    result = build_pointcloud_preview(
        source,
        output_dir,
        source_id="source",
        target_preview_points=3,
        max_preview_points=3,
        generate_html=False,
    )

    json.dumps(result.metadata)
    metadata = json.loads(result.json_path.read_text(encoding="utf-8"))
    data = load_npz(result.npz_path)
    assert metadata["preview_point_count"] == len(data["xyz"])
    assert data["source_id"].item() == "source"


def test_preview_rejects_data_raw_output_and_existing_outputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    raw.mkdir(parents=True)
    source = tmp_path / "source.las"
    write_las(source, point_count=3)

    with pytest.raises(LasIoError) as raw_error:
        build_pointcloud_preview(
            source,
            raw / "previews",
            source_id="source",
            generate_html=False,
            project_root=project,
        )
    assert raw_error.value.error.code == "PREVIEW_OUTPUT_FORBIDDEN"

    output_dir = tmp_path / "previews"
    output_dir.mkdir()
    (output_dir / "source_preview.npz").write_bytes(b"exists")
    with pytest.raises(LasIoError) as exists_error:
        build_pointcloud_preview(source, output_dir, source_id="source", generate_html=False)
    assert exists_error.value.error.code == "PREVIEW_OUTPUT_EXISTS"


def test_preview_cancellation_removes_temporary_files(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=8)
    token = CancellationToken()
    reporter = CancelAfterFirstReporter(token)

    with pytest.raises(LasIoError) as error:
        build_pointcloud_preview(
            source,
            output_dir,
            source_id="source",
            chunk_size=2,
            reporter=reporter,
            cancellation_token=token,
            generate_html=False,
        )

    assert error.value.error.code == "PREVIEW_CANCELLED"
    assert reporter.events
    assert not list(output_dir.glob("*.tmp.*"))
    assert not (output_dir / "source_preview.npz").exists()


def test_preview_failure_removes_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=3)

    def fail_save(path: Path, **kwargs: object) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("forced save failure")

    monkeypatch.setattr(np, "savez_compressed", fail_save)

    with pytest.raises(RuntimeError, match="forced save failure"):
        build_pointcloud_preview(source, output_dir, source_id="source", generate_html=False)

    assert not list(output_dir.glob("*.tmp.*"))
    assert not (output_dir / "source_preview.npz").exists()


def test_empty_las_has_clear_empty_preview_behavior(tmp_path: Path) -> None:
    source = tmp_path / "empty.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=0)

    result = build_pointcloud_preview(
        source,
        output_dir,
        source_id="empty",
        target_preview_points=5,
        max_preview_points=5,
        generate_html=False,
    )
    data = load_npz(result.npz_path)

    assert len(data["xyz"]) == 0
    assert result.metadata["preview_point_count"] == 0
    assert "Source LAS contains zero points" in result.metadata["warnings"]


def test_preview_does_not_call_full_read_interfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.las"
    output_dir = tmp_path / "previews"
    write_las(source, point_count=6)

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full point read must not be called")

    monkeypatch.setattr(laspy, "read", fail_read)
    monkeypatch.setattr(laspy.lasreader.LasReader, "read", fail_read)

    result = build_pointcloud_preview(
        source,
        output_dir,
        source_id="source",
        chunk_size=2,
        generate_html=False,
    )

    assert result.metadata["preview_point_count"] == 6
