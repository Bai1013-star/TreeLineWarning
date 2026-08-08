from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.io import (
    LasIoError,
    boundary_proximity_report,
    create_scene_sample,
    load_and_validate_scene_simple_selection,
    precheck_crop,
    read_las_header_summary,
    validate_scene_sample_artifacts,
)
from algorithms.io.samples import validate_scene_sample_outputs
from algorithms.preprocessing.preview import build_pointcloud_preview
from core.contracts import CropBounds


def write_scene_source(path: Path, *, point_count: int = 8) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
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
    las.classification = (indexes % 4).astype(np.uint8)
    las.return_number = np.ones(point_count, dtype=np.uint8)
    las.number_of_returns = np.ones(point_count, dtype=np.uint8)
    las.gps_time = indexes.astype(np.float64) + 0.25
    las.red = (indexes + 100).astype(np.uint16)
    las.green = (indexes + 200).astype(np.uint16)
    las.blue = (indexes + 300).astype(np.uint16)
    las.NormalX = np.full(point_count, 1.0, dtype=np.float32)
    las.NormalY = np.full(point_count, 0.0, dtype=np.float32)
    las.NormalZ = np.full(point_count, 0.5, dtype=np.float32)
    las.write(path)


def selection_payload(source: Path, bounds: CropBounds) -> dict:
    return {
        "source_id": "source",
        "source_path": str(source),
        "created_at": "2026-07-20T00:00:00+00:00",
        "preview_point_count": 8,
        "selected_preview_point_count": 3,
        "estimated_keep_ratio": 0.375,
        "min_xyz": list(bounds.min_xyz),
        "max_xyz": list(bounds.max_xyz),
        "coordinate_mode": "source",
        "label": "scene_simple",
        "note": "test",
    }


def test_scene_sample_crop_index_map_manifest_and_validation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))
    write_scene_source(source)
    before = source.stat()

    result = create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )

    assert result["output_point_count"] == 3
    assert result["kept_ratio"] == 3 / 8
    assert output.exists()
    assert index_map.exists()
    assert manifest.exists()
    source_after = source.stat()
    assert before.st_size == source_after.st_size
    assert before.st_mtime == source_after.st_mtime
    with np.load(index_map, allow_pickle=False) as data:
        assert data["original_index"].tolist() == [2, 3, 4]
        assert str(data["source_id"]) == "source"
        assert int(data["output_point_count"]) == 3
        assert json.loads(str(data["crop_bounds"]))["coordinate_mode"] == "source"

    output_summary = read_las_header_summary(output, source_id="output")
    source_summary = read_las_header_summary(source, source_id="source")
    assert output_summary.point_count == 3
    assert output_summary.point_format_id == source_summary.point_format_id
    assert output_summary.scale == source_summary.scale
    assert output_summary.offset == source_summary.offset
    assert {"NormalX", "NormalY", "NormalZ"}.issubset(set(output_summary.extra_dimensions))
    assert output_summary.has_rgb is True
    assert output_summary.has_intensity is True
    assert output_summary.has_classification is True

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["source_id"] == "source"
    assert manifest_payload["crop_bounds"]["min_xyz"] == [102.0, 202.0, 2.0]
    assert manifest_payload["output_file_size"] == output.stat().st_size
    assert manifest_payload["validation"]["index_map_unique"] is True
    assert manifest_payload["source_file_unchanged"] is True


def test_precheck_boundary_counts_and_kept_range(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))

    precheck = precheck_crop(source, bounds, source_id="source", chunk_size=2)

    assert precheck.input_point_count == 8
    assert precheck.kept_point_count == 3
    assert precheck.original_index.tolist() == [2, 3, 4]
    assert precheck.boundary_counts["x_min"] == 1
    assert precheck.boundary_counts["x_max"] == 1
    assert precheck.boundary_counts["z_max"] == 1
    assert precheck.z_max_boundary_ratio == pytest.approx(1 / 3)
    assert precheck.kept_min_xyz == (102.0, 202.0, 2.0)
    assert precheck.kept_max_xyz == (104.0, 204.0, 4.0)


def test_selection_validation_requires_approved_source_bounds_and_label(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    bounds = CropBounds(min_xyz=(1.0, 2.0, 3.0), max_xyz=(4.0, 5.0, 6.0))
    selection = tmp_path / "scene_simple_bounds.json"
    selection.write_text(json.dumps(selection_payload(source, bounds)), encoding="utf-8")

    payload, restored = load_and_validate_scene_simple_selection(
        selection,
        expected_source_path=source,
        expected_min_xyz=bounds.min_xyz,
        expected_max_xyz=bounds.max_xyz,
    )

    assert payload["source_id"] == "source"
    assert restored == bounds

    bad_payload = selection_payload(source, bounds)
    bad_payload["coordinate_mode"] = "local"
    selection.write_text(json.dumps(bad_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate_mode"):
        load_and_validate_scene_simple_selection(
            selection,
            expected_source_path=source,
            expected_min_xyz=bounds.min_xyz,
            expected_max_xyz=bounds.max_xyz,
        )


def test_scene_sample_rejects_existing_outputs_and_forbidden_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(100.0, 200.0, 0.0), max_xyz=(101.0, 201.0, 1.0))
    existing = samples / "scene_simple.las"
    existing.write_bytes(b"exists")

    with pytest.raises(LasIoError) as exists_error:
        create_scene_sample(
            source_path=source,
            output_path=existing,
            index_map_path=samples / "scene_simple_index_map.npz",
            manifest_path=samples / "scene_simple_manifest.json",
            bounds=bounds,
            source_id="source",
            selection_record=selection_payload(source, bounds),
            project_root=project,
        )
    assert exists_error.value.error.code == "SCENE_SAMPLE_OUTPUT_EXISTS"

    with pytest.raises(LasIoError) as path_error:
        create_scene_sample(
            source_path=source,
            output_path=project / "outputs" / "bad.las",
            index_map_path=samples / "scene_simple_index_map.npz",
            manifest_path=samples / "scene_simple_manifest.json",
            bounds=bounds,
            source_id="source",
            selection_record=selection_payload(source, bounds),
            project_root=project,
            overwrite=True,
        )
    assert path_error.value.error.code == "SCENE_SAMPLE_OUTPUT_FORBIDDEN"


def test_scene_sample_failure_cleans_generated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))

    def fail_savez(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("forced index failure")

    monkeypatch.setattr(np, "savez_compressed", fail_savez)

    with pytest.raises(RuntimeError, match="forced index failure"):
        create_scene_sample(
            source_path=source,
            output_path=output,
            index_map_path=index_map,
            manifest_path=manifest,
            bounds=bounds,
            source_id="source",
            selection_record=selection_payload(source, bounds),
            chunk_size=2,
            project_root=project,
        )

    assert not output.exists()
    assert not index_map.exists()
    assert not manifest.exists()
    assert not (samples / "scene_simple.tmp.las").exists()
    assert not (samples / "scene_simple_index_map.tmp.npz").exists()


def test_scene_sample_does_not_call_full_read_interfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full point read must not be called")

    monkeypatch.setattr(laspy, "read", fail_read)
    monkeypatch.setattr(laspy.lasreader.LasReader, "read", fail_read)

    create_scene_sample(
        source_path=source,
        output_path=samples / "scene_simple.las",
        index_map_path=samples / "scene_simple_index_map.npz",
        manifest_path=samples / "scene_simple_manifest.json",
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )


def test_validation_detects_manifest_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))
    result = create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )
    broken = json.loads(manifest.read_text(encoding="utf-8"))
    broken["output_point_count"] = 999
    manifest.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        validate_scene_sample_outputs(
            source_path=source,
            output_path=output,
            index_map_path=index_map,
            manifest_path=manifest,
            bounds=bounds,
            source_header=result["source_header"],
            source_snapshot_before=result["source_file_before"],
            source_point_count=result["source_point_count"],
            expected_point_count=result["output_point_count"],
            chunk_size=2,
        )


def test_scene_sample_artifact_consistency_check(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))
    create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )

    validation = validate_scene_sample_artifacts(
        sample_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        chunk_size=2,
    )

    assert validation["sample_header"]["point_count"] == 3
    assert validation["index_map_length"] == 3
    assert validation["index_map_unique"] is True
    assert validation["manifest_matches_files"] is True
    assert validation["points_within_bounds"] is True


def test_scene_sample_artifact_bounds_tolerate_json_float_noise(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    bounds = CropBounds(
        min_xyz=(102.0, 202.0, -3.1609),
        max_xyz=(104.0, 204.0, 4.0),
    )
    write_scene_source(source)
    create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["crop_bounds"]["min_xyz"][2] = -3.1609000000000003
    payload["crop_bounds"]["max_xyz"][0] += 5e-10
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_scene_sample_artifacts(
        sample_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        chunk_size=2,
    )

    assert validation["manifest_matches_files"] is True


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_scene_sample_artifact_bounds_reject_large_or_nonfinite_differences(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))
    write_scene_source(source)
    create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["crop_bounds"]["min_xyz"][0] += 1e-6
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        validate_scene_sample_artifacts(
            sample_path=output,
            index_map_path=index_map,
            manifest_path=manifest,
            bounds=bounds,
            chunk_size=2,
        )

    payload["crop_bounds"] = bounds.to_dict()
    payload["crop_bounds"]["max_xyz"][0] = invalid_value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        validate_scene_sample_artifacts(
            sample_path=output,
            index_map_path=index_map,
            manifest_path=manifest,
            bounds=bounds,
            chunk_size=2,
        )


def test_scene_sample_artifact_bounds_reject_coordinate_mode_change(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))
    write_scene_source(source)
    create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["coordinate_mode"] = "local"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        validate_scene_sample_artifacts(
            sample_path=output,
            index_map_path=index_map,
            manifest_path=manifest,
            bounds=bounds,
            chunk_size=2,
        )


def test_boundary_proximity_report_counts_multiple_tolerances(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))

    report = boundary_proximity_report(source, bounds, source_id="source", chunk_size=3)
    payload = report.to_dict()

    assert payload["point_count"] == 8
    assert payload["counts"]["x_min"]["0.05"] == 1
    assert payload["counts"]["x_max"]["0.10"] == 1
    assert payload["counts"]["z_max"]["0.20"] == 1
    assert payload["z_max_detail"]["count"] == 1
    assert payload["z_max_detail"]["touches_z_max"] is True
    assert payload["z_max_detail"]["classification_counts"] == {"0": 1}


def test_scene_sample_preview_marks_bounds_and_avoids_full_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    previews = project / "outputs" / "previews"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "source.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    write_scene_source(source)
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))
    create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )
    before = output.stat()

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full point read must not be called")

    monkeypatch.setattr(laspy, "read", fail_read)
    monkeypatch.setattr(laspy.lasreader.LasReader, "read", fail_read)

    result = build_pointcloud_preview(
        output,
        previews,
        source_id="scene_simple",
        target_preview_points=3,
        max_preview_points=3,
        chunk_size=2,
        crop_bounds=bounds,
        project_root=project,
    )
    after = output.stat()

    assert result.metadata["preview_point_count"] == 3
    assert result.metadata["crop_bounds"]["coordinate_mode"] == "source"
    assert result.html_path is not None
    assert "Approved crop bounds" in result.html_path.read_text(encoding="utf-8")
    assert before.st_size == after.st_size
    assert before.st_mtime == after.st_mtime


def test_scene_sample_binds_classified_parent_lineage(tmp_path: Path) -> None:
    """The lineage adapter binds the classified parent without crop changes."""
    project = tmp_path / "project"
    raw = project / "data" / "raw"
    samples = project / "data" / "samples"
    raw.mkdir(parents=True)
    samples.mkdir(parents=True)
    source = raw / "classified_parent.las"
    output = samples / "scene_simple.las"
    index_map = samples / "scene_simple_index_map.npz"
    manifest = samples / "scene_simple_manifest.json"
    bounds = CropBounds(min_xyz=(102.0, 202.0, 2.0), max_xyz=(104.0, 204.0, 4.0))
    write_scene_source(source)

    # Without lineage the manifest must not carry the field.
    result_plain = create_scene_sample(
        source_path=source,
        output_path=output,
        index_map_path=index_map,
        manifest_path=manifest,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
    )
    assert result_plain.get("parent_lineage") is None

    output2 = samples / "scene_simple_lineaged.las"
    index_map2 = samples / "scene_simple_lineaged_index_map.npz"
    manifest2 = samples / "scene_simple_lineaged_manifest.json"
    lineage = {
        "classified_parent_source_id": "scene-1--fgc-v1--0123456789abcdef",
        "classified_parent_source_file_sha256": "a" * 64,
        "classified_parent_identity_record_digest": "b" * 64,
        "parent_original_index_map_digest": "c" * 64,
        "sample_las_sha256": "d" * 64,
        "sample_point_count": 3,
    }
    result = create_scene_sample(
        source_path=source,
        output_path=output2,
        index_map_path=index_map2,
        manifest_path=manifest2,
        bounds=bounds,
        source_id="source",
        selection_record=selection_payload(source, bounds),
        chunk_size=2,
        project_root=project,
        parent_lineage=lineage,
    )
    assert result["output_point_count"] == 3  # crop selection unchanged
    assert result["parent_lineage"] == lineage
    payload = json.loads(manifest2.read_text(encoding="utf-8"))
    assert payload["parent_lineage"]["classified_parent_source_id"].startswith("scene-1--fgc-v1--")
    assert payload["parent_lineage"]["parent_original_index_map_digest"] == "c" * 64
