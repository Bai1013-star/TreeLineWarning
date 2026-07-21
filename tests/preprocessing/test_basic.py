from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.io import LasIoError
from algorithms.preprocessing.basic import run_basic_preprocessing
from core.contracts import CancellationToken, CropBounds, PipelineProgress, PreprocessingConfig


class RecordingReporter:
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


def write_las(path: Path, xyz: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 0.0])
    header.add_extra_dims(
        [
            laspy.ExtraBytesParams(name="NormalX", type=np.float32),
            laspy.ExtraBytesParams(name="NormalY", type=np.float32),
            laspy.ExtraBytesParams(name="NormalZ", type=np.float32),
        ]
    )
    las = laspy.LasData(header)
    indexes = np.arange(len(xyz))
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = (indexes + 10).astype(np.uint16)
    las.classification = (indexes % 6).astype(np.uint8)
    las.red = (indexes + 100).astype(np.uint16)
    las.green = (indexes + 200).astype(np.uint16)
    las.blue = (indexes + 300).astype(np.uint16)
    las.NormalX = np.full(len(xyz), 1.0, dtype=np.float32)
    las.NormalY = np.full(len(xyz), 0.0, dtype=np.float32)
    las.NormalZ = np.full(len(xyz), 0.5, dtype=np.float32)
    las.write(path)


def make_project(tmp_path: Path, xyz: np.ndarray) -> tuple[Path, Path, Path, Path, Path]:
    project = tmp_path / "project"
    samples = project / "data" / "samples"
    output = project / "outputs" / "pointcloud"
    samples.mkdir(parents=True)
    output.mkdir(parents=True)
    sample = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    write_las(sample, xyz)
    if len(xyz):
        bounds = CropBounds(
            min_xyz=tuple(float(value) for value in xyz.min(axis=0)),
            max_xyz=tuple(float(value) for value in xyz.max(axis=0) + 0.001),
        )
    else:
        bounds = CropBounds(min_xyz=(0.0, 0.0, 0.0), max_xyz=(1.0, 1.0, 1.0))
    np.savez_compressed(
        index_map,
        original_index=np.arange(len(xyz), dtype=np.int64) + 1000,
        source_id=np.asarray("raw_source"),
        crop_bounds=np.asarray(json.dumps(bounds.to_dict())),
        output_point_count=np.asarray(len(xyz), dtype=np.int64),
    )
    header = {
        "source_id": "raw_source",
        "source_path": str(sample),
        "point_count": len(xyz),
        "dimensions": [],
    }
    manifest.write_text(
        json.dumps(
            {
                "source_id": "raw_source",
                "source_path": str(sample),
                "source_file_size": sample.stat().st_size,
                "source_point_count": len(xyz) + 1000,
                "source_header": header,
                "crop_bounds": bounds.to_dict(),
                "coordinate_mode": "source",
                "output_path": str(sample),
                "output_point_count": len(xyz),
                "output_file_size": sample.stat().st_size,
                "index_map_path": str(index_map),
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    return project, sample, index_map, manifest, output


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def test_header_min_local_coordinates_two_pass_voxel_and_traceability(tmp_path: Path) -> None:
    xyz = np.array(
        [
            [100.00, 200.00, 0.00],
            [100.01, 200.00, 0.00],
            [100.04, 200.00, 0.00],
            [100.11, 200.00, 0.00],
        ],
        dtype=np.float64,
    )
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)

    result = run_basic_preprocessing(
        sample,
        index_map,
        manifest,
        output,
        config=PreprocessingConfig(chunk_size=2, voxel_size_m=0.05),
        project_root=project,
    )
    data = load_npz(result.output_npz_path)

    assert result.statistics.input_point_count == 4
    assert result.statistics.downsampled_point_count == 2
    assert data["origin_xyz"].tolist() == [100.0, 200.0, 0.0]
    assert data["sample_index"].tolist() == [1, 3]
    assert data["original_index"].tolist() == [1001, 1003]
    assert data["voxel_key"].tolist() == [[0, 0, 0], [2, 0, 0]]
    assert data["voxel_point_count"].tolist() == [3, 1]
    assert np.isfinite(data["xyz_local"]).all()


def test_explicit_origin_and_floor_voxel_key_rule(tmp_path: Path) -> None:
    xyz = np.array([[99.99, 200.0, 0.0], [100.0, 200.0, 0.0]], dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)

    result = run_basic_preprocessing(
        sample,
        index_map,
        manifest,
        output,
        config=PreprocessingConfig(
            origin_mode="explicit",
            origin_xyz=(100.0, 200.0, 0.0),
            voxel_size_m=0.05,
        ),
        project_root=project,
    )
    data = load_npz(result.output_npz_path)

    assert data["origin_xyz"].tolist() == [100.0, 200.0, 0.0]
    assert data["voxel_key"].tolist() == [[-1, 0, 0], [0, 0, 0]]


def test_attributes_rgb_and_normals_are_preserved(tmp_path: Path) -> None:
    xyz = np.array([[100.0, 200.0, 0.0], [100.1, 200.0, 0.0]], dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)

    result = run_basic_preprocessing(
        sample,
        index_map,
        manifest,
        output,
        config=PreprocessingConfig(voxel_size_m=0.01),
        project_root=project,
    )
    data = load_npz(result.output_npz_path)

    assert data["red"].dtype == np.uint16
    assert data["rgb"].dtype == np.uint16
    assert data["rgb"].tolist() == [[100, 200, 300], [101, 201, 301]]
    assert data["intensity"].tolist() == [10, 11]
    assert data["classification"].tolist() == [0, 1]
    assert "NormalX" in data


def test_output_default_does_not_overwrite_and_json_reloads(tmp_path: Path) -> None:
    xyz = np.array([[100.0, 200.0, 0.0]], dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)
    config = PreprocessingConfig(voxel_size_m=0.01)

    first = run_basic_preprocessing(
        sample,
        index_map,
        manifest,
        output,
        config=config,
        project_root=project,
    )
    payload = json.loads(first.output_json_path.read_text(encoding="utf-8"))

    assert payload["statistics"]["downsampled_point_count"] == 1
    with pytest.raises(LasIoError) as error:
        run_basic_preprocessing(
            sample,
            index_map,
            manifest,
            output,
            config=config,
            project_root=project,
        )
    assert error.value.error.code == "PREPROCESSING_OUTPUT_EXISTS"


def test_max_output_points_exceeding_limit_fails(tmp_path: Path) -> None:
    xyz = np.array([[100.0, 200.0, 0.0], [100.1, 200.0, 0.0]], dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)

    with pytest.raises(LasIoError) as error:
        run_basic_preprocessing(
            sample,
            index_map,
            manifest,
            output,
            config=PreprocessingConfig(voxel_size_m=0.01, max_output_points=1),
            project_root=project,
        )
    assert error.value.error.code == "PREPROCESSING_OUTPUT_POINT_LIMIT_EXCEEDED"


def test_cancelled_run_fails_and_leaves_no_outputs(tmp_path: Path) -> None:
    xyz = np.array([[100.0, 200.0, 0.0], [100.1, 200.0, 0.0]], dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)
    token = CancellationToken()
    reporter = CancelAfterFirstReporter(token)

    with pytest.raises(LasIoError) as error:
        run_basic_preprocessing(
            sample,
            index_map,
            manifest,
            output,
            config=PreprocessingConfig(chunk_size=1, voxel_size_m=0.01),
            reporter=reporter,
            cancellation_token=token,
            project_root=project,
        )

    assert error.value.error.code == "PREPROCESSING_CANCELLED"
    assert not list(output.glob("*.tmp.*"))
    assert not (output / "scene_simple_preprocessed.npz").exists()


def test_empty_input_fails(tmp_path: Path) -> None:
    xyz = np.empty((0, 3), dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)

    with pytest.raises(LasIoError) as error:
        run_basic_preprocessing(sample, index_map, manifest, output, project_root=project)
    assert error.value.error.code == "PREPROCESSING_EMPTY_INPUT"


def test_nan_inf_and_abnormal_coordinates_are_removed_with_monkeypatched_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from algorithms.preprocessing import basic

    project, sample, index_map, manifest, output = make_project(
        tmp_path,
        np.array([[100.0, 200.0, 0.0]], dtype=np.float64),
    )
    np.savez_compressed(
        index_map,
        original_index=np.array([1000, 1001, 1002, 1003], dtype=np.int64),
        source_id=np.asarray("raw_source"),
        crop_bounds=np.asarray(json.dumps(CropBounds((0, 0, 0), (1, 1, 1)).to_dict())),
        output_point_count=np.asarray(4, dtype=np.int64),
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["source_point_count"] = 2000
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    class Header:
        point_count = 4
        min_xyz = (0.0, 0.0, 0.0)

    class Chunk:
        source_id = "scene_simple"
        chunk_id = 0
        start_index = 0
        point_count = 4
        original_index = np.arange(4, dtype=np.int64)
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0],
                [np.inf, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        attributes = {"intensity": np.arange(4, dtype=np.uint16)}

    monkeypatch.setattr(basic, "read_las_header_summary", lambda *_args, **_kwargs: Header())
    monkeypatch.setattr(
        basic,
        "validate_scene_sample_artifacts",
        lambda **_kwargs: {"manifest": {}},
    )
    monkeypatch.setattr(basic, "iter_las_chunks", lambda *_args, **_kwargs: iter([Chunk()]))

    result = run_basic_preprocessing(
        sample,
        index_map,
        manifest,
        output,
        config=PreprocessingConfig(max_abs_local_coord_m=10.0, voxel_size_m=0.05),
        project_root=project,
    )

    assert result.statistics.nan_inf_removed_count == 2
    assert result.statistics.abnormal_coord_removed_count == 1
    assert result.statistics.downsampled_point_count == 1


def test_full_read_interfaces_are_not_called(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xyz = np.array([[100.0, 200.0, 0.0], [100.1, 200.0, 0.0]], dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full point read must not be called")

    monkeypatch.setattr(laspy, "read", fail_read)
    monkeypatch.setattr(laspy.lasreader.LasReader, "read", fail_read)

    result = run_basic_preprocessing(
        sample,
        index_map,
        manifest,
        output,
        config=PreprocessingConfig(chunk_size=1, voxel_size_m=0.01),
        project_root=project,
    )

    assert result.statistics.downsampled_point_count == 2


def test_input_sample_files_are_unchanged(tmp_path: Path) -> None:
    xyz = np.array([[100.0, 200.0, 0.0], [100.1, 200.0, 0.0]], dtype=np.float64)
    project, sample, index_map, manifest, output = make_project(tmp_path, xyz)
    before = {path: path.stat() for path in (sample, index_map, manifest)}

    run_basic_preprocessing(
        sample,
        index_map,
        manifest,
        output,
        config=PreprocessingConfig(voxel_size_m=0.01),
        project_root=project,
    )

    after = {path: path.stat() for path in (sample, index_map, manifest)}
    for path in before:
        assert before[path].st_size == after[path].st_size
        assert before[path].st_mtime == after[path].st_mtime
