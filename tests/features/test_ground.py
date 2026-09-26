from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.features import ground
from algorithms.features.ground import GroundEstimationError, run_ground_height_estimation
from core.contracts import (
    CancellationToken,
    GroundEstimationConfig,
    GroundEstimationResult,
    GroundQualityFlag,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def make_project(
    tmp_path: Path,
    xyz: np.ndarray,
    *,
    classification: np.ndarray | None = None,
    source_id: str = "scene_simple",
) -> tuple[Path, Path, Path, Path, Path]:
    project = tmp_path / "project"
    pointcloud_dir = project / "outputs" / "pointcloud"
    feature_dir = project / "outputs" / "features"
    pointcloud_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    a1_npz = pointcloud_dir / "scene_simple_preprocessed.npz"
    a1_json = pointcloud_dir / "scene_simple_preprocessed.json"
    a2a_npz = feature_dir / "scene_simple_geometry_features.npz"
    a2a_json = feature_dir / "scene_simple_geometry_features.json"
    points = np.asarray(xyz, dtype=np.float32)
    count = len(points)
    class_values = (
        np.zeros(count, dtype=np.uint8)
        if classification is None
        else np.asarray(classification, dtype=np.uint8)
    )
    original_index = np.arange(1000, 1000 + count, dtype=np.int64)
    sample_index = np.arange(count, dtype=np.int64)
    origin = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    a1_arrays = {
        "source_id": np.asarray(source_id),
        "manifest_source_id": np.asarray("raw_scene"),
        "xyz_local": points,
        "origin_xyz": origin,
        "original_index": original_index,
        "sample_index": sample_index,
        "classification": class_values,
    }
    np.savez_compressed(a1_npz, **a1_arrays)
    a1_payload = {
        "source_id": source_id,
        "manifest_source_id": "raw_scene",
        "output_npz_path": str(a1_npz.resolve()),
        "output_json_path": str(a1_json.resolve()),
        "origin_xyz": origin.tolist(),
        "arrays": {},
    }
    a1_json.write_text(json.dumps(a1_payload), encoding="utf-8")
    a2a_arrays = {
        "source_id": np.asarray(source_id),
        "manifest_source_id": np.asarray("raw_scene"),
        "xyz_local": points.copy(),
        "origin_xyz": origin.copy(),
        "original_index": original_index.copy(),
        "sample_index": sample_index.copy(),
        "planarity": np.full(count, 0.8, dtype=np.float32),
        "verticality": np.zeros(count, dtype=np.float32),
        "scattering": np.full(count, 0.01, dtype=np.float32),
        "normal": np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        "height_m": points[:, 2].copy(),
        "quality_flags": np.zeros(count, dtype=np.uint16),
    }
    np.savez_compressed(a2a_npz, **a2a_arrays)
    a2a_payload = {
        "source_id": source_id,
        "manifest_source_id": "raw_scene",
        "output_npz_path": str(a2a_npz.resolve()),
        "output_json_path": str(a2a_json.resolve()),
        "output_npz_sha256": _sha256(a2a_npz),
        "input_a1_npz_sha256": _sha256(a1_npz),
        "input_a1_json_sha256": _sha256(a1_json),
        "origin_xyz": origin.tolist(),
        "arrays": {},
    }
    a2a_json.write_text(json.dumps(a2a_payload), encoding="utf-8")
    return project, a1_npz, a1_json, a2a_npz, a2a_json


def run_ground(
    tmp_path: Path,
    xyz: np.ndarray,
    *,
    config: GroundEstimationConfig | None = None,
    classification: np.ndarray | None = None,
):
    project, a1_npz, a1_json, a2a_npz, a2a_json = make_project(
        tmp_path, xyz, classification=classification
    )
    result = run_ground_height_estimation(
        a1_npz,
        a1_json,
        a2a_npz,
        a2a_json,
        project / "outputs" / "features" / "hag",
        config=config,
        project_root=project,
    )
    return result, _load_npz(result.output_npz_path)


def _cell_points(x: float, y: float, z: float, count: int) -> np.ndarray:
    offsets = np.linspace(0.02, 0.18, count)
    return np.column_stack((x + offsets, y + offsets, np.full(count, z)))


def test_horizontal_ground_vegetation_and_wire_hag(tmp_path: Path) -> None:
    ground = _cell_points(0.0, 0.0, 1.0, 40)
    vegetation = np.array([[0.3, 0.3, 3.0]])
    wire = np.array([[0.4, 0.4, 7.0]])
    result, data = run_ground(tmp_path, np.vstack((ground, vegetation, wire)))

    assert np.allclose(data["ground_z_m"], 1.0, atol=1e-6)
    assert data["raw_height_agl_m"][-2:].tolist() == pytest.approx([2.0, 6.0])
    assert data["height_agl_m"][-2:].tolist() == pytest.approx([2.0, 6.0])
    assert result.statistics.direct_control_cell_count == 1


def test_sloped_plane_preserves_local_ground_height(tmp_path: Path) -> None:
    rows = []
    for x_cell in range(3):
        for y_cell in range(2):
            for offset in np.linspace(0.05, 0.35, 8):
                x, y = x_cell + offset, y_cell + offset
                rows.append((x, y, 0.2 * x + 0.1 * y))
    xyz = np.asarray(rows)
    config = GroundEstimationConfig(grid_size_m=1.0, min_points_per_cell=4)
    _result, data = run_ground(tmp_path, xyz, config=config)

    assert np.max(np.abs(data["raw_height_agl_m"])) < 0.10


def test_quantile_resists_a_single_extreme_low_noise_point(tmp_path: Path) -> None:
    xyz = np.vstack((_cell_points(0.0, 0.0, 1.0, 40), np.array([[0.3, 0.3, -10.0]])))
    _result, data = run_ground(tmp_path, xyz)

    assert np.median(data["ground_z_m"]) == pytest.approx(1.0)


def test_sparse_empty_idw_requested_k_and_deterministic_cell_ids(tmp_path: Path) -> None:
    left = _cell_points(0.0, 0.0, 1.0, 4)
    right = _cell_points(2.0, 0.0, 3.0, 4)
    sparse = np.array([[1.2, 0.2, 2.0]])
    config = GroundEstimationConfig(
        grid_size_m=1.0,
        min_points_per_cell=4,
        idw_neighbors=99,
        interpolation_max_distance_m=2.0,
    )
    result, data = run_ground(tmp_path, np.vstack((left, right, sparse)), config=config)

    assert result.statistics.sparse_cell_count == 1
    assert data["ground_cell_id"][-1] == 1
    assert data["quality_flags"][-1] & int(GroundQualityFlag.SPARSE_GROUND_CELL)
    assert data["quality_flags"][-1] & int(GroundQualityFlag.INTERPOLATED_GROUND)
    assert data["ground_z_m"][-1] == pytest.approx(2.0, abs=1e-5)
    assert data["ground_distance_m"][-1] == pytest.approx(1.0, abs=1e-5)


def test_extrapolation_keeps_partial_result_and_marks_missing_hag(tmp_path: Path) -> None:
    control = _cell_points(0.0, 0.0, 1.0, 4)
    distant_sparse = np.array([[4.2, 0.2, 4.0]])
    config = GroundEstimationConfig(
        grid_size_m=1.0,
        min_points_per_cell=4,
        interpolation_max_distance_m=1.5,
    )
    _result, data = run_ground(tmp_path, np.vstack((control, distant_sparse)), config=config)

    flag = int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    assert np.isnan(data["ground_z_m"][-1])
    assert np.isnan(data["height_agl_m"][-1])
    assert data["quality_flags"][-1] & flag
    assert data["quality_flags"][-1] & int(GroundQualityFlag.EXTRAPOLATION_TOO_FAR)
    assert np.isfinite(data["ground_z_m"][:-1]).all()


def test_zero_controls_and_unsupported_smoothing_fail_without_output(tmp_path: Path) -> None:
    xyz = np.array([[0.1, 0.1, 1.0], [0.2, 0.2, 1.0]])
    project, a1_npz, a1_json, a2a_npz, a2a_json = make_project(tmp_path, xyz)
    output = project / "outputs" / "features" / "hag"
    with pytest.raises(GroundEstimationError) as error:
        run_ground_height_estimation(
            a1_npz,
            a1_json,
            a2a_npz,
            a2a_json,
            output,
            project_root=project,
        )
    assert error.value.error.code == "GROUND_CONTROL_EMPTY"
    assert not output.exists()
    with pytest.raises(GroundEstimationError, match="smoothing_radius_cells"):
        run_ground_height_estimation(
            a1_npz,
            a1_json,
            a2a_npz,
            a2a_json,
            output,
            config=GroundEstimationConfig(smoothing_radius_cells=1),
            project_root=project,
        )


def test_boundary_quantile_negative_extreme_gradient_and_classification_flags(
    tmp_path: Path,
) -> None:
    left = _cell_points(0.0, 0.0, 1.0, 40)
    right = _cell_points(1.0, 0.0, 5.0, 40)
    special = np.array([[0.4, 0.4, 0.9], [0.45, 0.45, 30.0], [2.0, 0.1, 1.0]])
    classification = np.zeros(len(left) + len(right) + len(special), dtype=np.uint8)
    classification[-2] = 2
    config = GroundEstimationConfig(
        grid_size_m=1.0,
        min_points_per_cell=32,
        high_ground_gradient_threshold_m_per_m=2.0,
    )
    _result, data = run_ground(
        tmp_path, np.vstack((left, right, special)), config=config, classification=classification
    )

    small_negative, extreme, boundary = -3, -2, -1
    assert data["raw_height_agl_m"][small_negative] == pytest.approx(-0.1, abs=1e-5)
    assert data["height_agl_m"][small_negative] == 0.0
    assert not data["quality_flags"][small_negative] & int(GroundQualityFlag.BELOW_GROUND_TOLERANCE)
    assert data["quality_flags"][extreme] & int(GroundQualityFlag.EXTREME_HEIGHT_AGL)
    assert data["quality_flags"][extreme] & int(GroundQualityFlag.SOURCE_CLASSIFICATION_DISAGREES)
    assert data["ground_cell_id"][boundary] == 2
    assert np.any(data["quality_flags"] & int(GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION))


def test_classification_zero_has_no_disagreement_and_indices_outputs_reload(tmp_path: Path) -> None:
    xyz = _cell_points(0.0, 0.0, 1.0, 40)
    result, data = run_ground(tmp_path, xyz)
    payload = json.loads(result.output_json_path.read_text(encoding="utf-8"))
    restored = GroundEstimationResult.from_dict(payload)

    assert not np.any(
        data["quality_flags"] & int(GroundQualityFlag.SOURCE_CLASSIFICATION_DISAGREES)
    )
    assert restored.classification_validation["applicable"] is False
    assert np.array_equal(data["original_index"], np.arange(1000, 1040, dtype=np.int64))
    assert np.array_equal(data["sample_index"], np.arange(40, dtype=np.int64))
    assert restored.output_npz_sha256 == _sha256(result.output_npz_path)


def test_no_overwrite_cancellation_path_escape_no_las_and_inputs_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xyz = _cell_points(0.0, 0.0, 1.0, 40)
    project, a1_npz, a1_json, a2a_npz, a2a_json = make_project(tmp_path, xyz)
    before = {path: _sha256(path) for path in (a1_npz, a1_json, a2a_npz, a2a_json)}
    output = project / "outputs" / "features" / "hag"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("LAS full read must not be called by A2B")

    monkeypatch.setattr(laspy, "read", forbidden)
    result = run_ground_height_estimation(
        a1_npz, a1_json, a2a_npz, a2a_json, output, project_root=project
    )
    with pytest.raises(GroundEstimationError) as exists_error:
        run_ground_height_estimation(
            a1_npz, a1_json, a2a_npz, a2a_json, output, project_root=project
        )
    assert exists_error.value.error.code == "GROUND_OUTPUT_EXISTS"
    token = CancellationToken()
    token.cancel()
    cancelled_output = project / "outputs" / "features" / "cancelled"
    with pytest.raises(GroundEstimationError) as cancelled_error:
        run_ground_height_estimation(
            a1_npz,
            a1_json,
            a2a_npz,
            a2a_json,
            cancelled_output,
            cancellation_token=token,
            project_root=project,
        )
    assert cancelled_error.value.error.code == "GROUND_CANCELLED"
    with pytest.raises(GroundEstimationError) as forbidden_error:
        run_ground_height_estimation(
            a1_npz,
            a1_json,
            a2a_npz,
            a2a_json,
            project / "outside",
            project_root=project,
        )
    assert forbidden_error.value.error.code == "GROUND_OUTPUT_FORBIDDEN"
    assert {path: _sha256(path) for path in before} == before
    assert result.output_npz_path.is_file()


def test_ground_implementation_uses_control_kdtree_not_global_distance_matrix() -> None:
    source = inspect.getsource(ground)

    assert "cKDTree(control_xy)" in source
    assert "distance_matrix" not in source
    assert "cdist(" not in source
