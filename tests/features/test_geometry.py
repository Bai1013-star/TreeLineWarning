from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.features import geometry
from algorithms.features.geometry import (
    GeometryFeatureError,
    load_geometry_features,
    run_geometry_feature_extraction,
)
from core.contracts import (
    CancellationToken,
    FeatureQualityFlag,
    GeometryFeatureConfig,
    PipelineProgress,
)


class CancelAfterFirstReporter:
    def __init__(self, token: CancellationToken) -> None:
        self.token = token
        self.events: list[PipelineProgress] = []

    def update(self, progress: PipelineProgress) -> None:
        self.events.append(progress)
        if progress.status == "running":
            self.token.cancel()


def make_project(
    tmp_path: Path,
    xyz: np.ndarray,
    *,
    source_id: str = "scene_simple",
) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    pointcloud = project / "outputs" / "pointcloud"
    pointcloud.mkdir(parents=True)
    npz_path = pointcloud / "scene_simple_preprocessed.npz"
    json_path = pointcloud / "scene_simple_preprocessed.json"
    xyz = np.asarray(xyz, dtype=np.float32)
    point_count = len(xyz)
    original_index = np.arange(100, 100 + point_count, dtype=np.int64)
    sample_index = np.arange(point_count, dtype=np.int64)
    voxel_key = np.floor(xyz / 0.05).astype(np.int64)
    voxel_count = np.ones(point_count, dtype=np.int32)
    arrays = {
        "xyz_local": xyz,
        "origin_xyz": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "original_index": original_index,
        "sample_index": sample_index,
        "voxel_key": voxel_key,
        "voxel_point_count": voxel_count,
        "source_id": np.asarray(source_id),
        "manifest_source_id": np.asarray("raw_scene"),
        "attribute_names": np.asarray([], dtype="U1"),
    }
    np.savez_compressed(npz_path, **arrays)
    array_metadata = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
        if name
        in {
            "xyz_local",
            "origin_xyz",
            "original_index",
            "sample_index",
            "voxel_key",
            "voxel_point_count",
        }
    }
    array_metadata["attributes"] = {}
    payload = {
        "source_id": source_id,
        "manifest_source_id": "raw_scene",
        "output_npz_path": str(npz_path.resolve()),
        "output_json_path": str(json_path.resolve()),
        "origin_xyz": arrays["origin_xyz"].tolist(),
        "arrays": array_metadata,
        "statistics": {"downsampled_point_count": point_count},
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return project, npz_path, json_path, project / "outputs" / "features"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def run_features(
    tmp_path: Path,
    xyz: np.ndarray,
    *,
    config: GeometryFeatureConfig | None = None,
):
    project, input_npz, input_json, output = make_project(tmp_path, xyz)
    result = run_geometry_feature_extraction(
        input_npz,
        input_json,
        output,
        config=config,
        project_root=project,
    )
    return result, load_npz(result.output_npz_path)


def _published_geometry(tmp_path: Path, xyz: np.ndarray):
    project, input_npz, input_json, output = make_project(tmp_path, xyz)
    return run_geometry_feature_extraction(input_npz, input_json, output, project_root=project)


def _refresh_geometry_integrity(json_path: Path, npz_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["output_npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    payload["output_npz_bytes"] = npz_path.stat().st_size
    json_path.write_text(json.dumps(payload), encoding="utf-8")


def test_line_has_high_linearity_and_principal_direction(tmp_path: Path) -> None:
    x = np.linspace(-5.0, 5.0, 80)
    xyz = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))

    _result, data = run_features(tmp_path, xyz)
    interior = slice(16, -16)

    assert np.median(data["linearity"][interior]) > 0.95
    assert np.median(data["planarity"][interior]) < 0.05
    assert np.median(data["scattering"][interior]) < 0.05
    assert np.median(np.abs(data["principal_direction"][interior, 0])) > 0.99
    assert np.all(data["normal"][interior] == 0.0)


def test_plane_has_high_planarity_and_horizontal_normal(tmp_path: Path) -> None:
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 17), np.linspace(-2.0, 2.0, 17))
    xyz = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    _result, data = run_features(tmp_path, xyz)
    central = (np.abs(xyz[:, 0]) < 1.0) & (np.abs(xyz[:, 1]) < 1.0)

    assert np.median(data["planarity"][central]) > 0.7
    assert np.median(data["scattering"][central]) < 0.05
    assert np.median(np.abs(data["normal"][central, 2])) > 0.99
    assert np.median(data["verticality"][central]) < 0.01


def test_vertical_plane_has_verticality_near_one(tmp_path: Path) -> None:
    y, z = np.meshgrid(np.linspace(-2.0, 2.0, 17), np.linspace(-2.0, 2.0, 17))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))

    _result, data = run_features(tmp_path, xyz)
    central = (np.abs(xyz[:, 1]) < 1.0) & (np.abs(xyz[:, 2]) < 1.0)

    assert np.median(np.abs(data["normal"][central, 0])) > 0.99
    assert np.median(data["verticality"][central]) > 0.99


def test_volumetric_isotropic_cloud_scattering_exceeds_line_and_plane(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    volume = rng.uniform(-1.0, 1.0, size=(400, 3))
    _result, volume_data = run_features(tmp_path / "volume", volume)
    line = np.column_stack((np.linspace(-3.0, 3.0, 80), np.zeros(80), np.zeros(80)))
    _result, line_data = run_features(tmp_path / "line", line)
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 17), np.linspace(-2.0, 2.0, 17))
    plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    _result, plane_data = run_features(tmp_path / "plane", plane)

    assert np.median(volume_data["scattering"]) > np.median(line_data["scattering"])
    assert np.median(volume_data["scattering"]) > np.median(plane_data["scattering"])


def test_scattering_uses_approved_eigenvalue_ratio(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    xyz = rng.uniform(-1.0, 1.0, size=(80, 3))
    _result, data = run_features(tmp_path, xyz)
    eigenvalues = data["eigenvalues"]
    valid = eigenvalues[:, 0] > 1e-12

    expected = eigenvalues[valid, 2] / eigenvalues[valid, 0]

    assert np.allclose(data["scattering"][valid], expected, rtol=1e-6, atol=1e-7)


def test_vertical_line_principal_direction_matches_z_axis(tmp_path: Path) -> None:
    z = np.linspace(-4.0, 4.0, 80)
    xyz = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    _result, data = run_features(tmp_path, xyz)

    assert np.median(np.abs(data["principal_direction"][16:-16, 2])) > 0.99


def test_duplicate_and_identical_coordinates_are_flagged(tmp_path: Path) -> None:
    duplicate_xyz = np.vstack((np.zeros((10, 3)), np.eye(3), -np.eye(3)))
    _result, duplicate_data = run_features(tmp_path / "duplicates", duplicate_xyz)
    assert np.any(
        duplicate_data["quality_flags"] & int(FeatureQualityFlag.DUPLICATE_NEIGHBOR_COORDINATES)
    )

    identical = np.zeros((12, 3))
    _result, identical_data = run_features(tmp_path / "identical", identical)
    flags = identical_data["quality_flags"]
    assert np.all(flags & int(FeatureQualityFlag.DEGENERATE_COVARIANCE))
    assert np.all(flags & int(FeatureQualityFlag.DENSITY_UNDEFINED))
    assert np.all(identical_data["local_density"] == 0.0)


def test_insufficient_neighbors_and_requested_k_larger_than_input_are_safe(tmp_path: Path) -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    _result, data = run_features(tmp_path, xyz)

    assert np.all(data["neighbor_count"] <= 2)
    assert np.all(data["quality_flags"] & int(FeatureQualityFlag.INSUFFICIENT_NEIGHBORS))
    assert np.all(data["quality_flags"] & int(FeatureQualityFlag.DENSITY_UNDEFINED))
    assert np.all(data["height_m"] == 0.0)


def test_query_point_is_excluded_by_index_for_density(tmp_path: Path) -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    config = GeometryFeatureConfig(k_neighbors=1, min_neighbors=1)
    _result, data = run_features(tmp_path, xyz, config=config)

    expected = 1.0 / ((4.0 / 3.0) * np.pi * 2.0**3)
    assert data["neighbor_count"].tolist() == [1, 1]
    assert np.allclose(data["local_density"], expected, rtol=1e-5)


def test_small_negative_eigenvalue_is_clipped_and_large_negative_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xyz = np.column_stack((np.linspace(-3.0, 3.0, 16), np.zeros(16), np.zeros(16)))
    original_eigh = geometry.np.linalg.eigh

    def small_negative(_value: np.ndarray):
        return np.array([-1e-13, 0.1, 1.0]), np.eye(3)

    monkeypatch.setattr(geometry.np.linalg, "eigh", small_negative)
    _result, data = run_features(tmp_path / "small", xyz)
    assert np.any(data["quality_flags"] & int(FeatureQualityFlag.NEGATIVE_EIGENVALUE_CLIPPED))
    assert np.all(data["eigenvalues"] >= 0.0)

    def large_negative(_value: np.ndarray):
        return np.array([-1e-3, 0.1, 1.0]), np.eye(3)

    monkeypatch.setattr(geometry.np.linalg, "eigh", large_negative)
    project, input_npz, input_json, output = make_project(tmp_path / "large", xyz)
    with pytest.raises(GeometryFeatureError) as error:
        run_geometry_feature_extraction(input_npz, input_json, output, project_root=project)
    assert error.value.error.code == "FEATURE_EIGENVALUE_NEGATIVE"
    monkeypatch.setattr(geometry.np.linalg, "eigh", original_eigh)


def test_direction_ambiguity_uses_zero_vector_sentinels(tmp_path: Path) -> None:
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 17), np.linspace(-2.0, 2.0, 17))
    plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    ambiguity_config = GeometryFeatureConfig(direction_gap_relative_tolerance=0.01)
    _result, plane_data = run_features(tmp_path / "plane", plane, config=ambiguity_config)
    assert np.any(
        plane_data["quality_flags"] & int(FeatureQualityFlag.PRINCIPAL_DIRECTION_AMBIGUOUS)
    )

    line = np.column_stack((np.linspace(-3.0, 3.0, 80), np.zeros(80), np.zeros(80)))
    _result, line_data = run_features(tmp_path / "line", line)
    assert np.any(line_data["quality_flags"] & int(FeatureQualityFlag.NORMAL_DIRECTION_AMBIGUOUS))
    assert np.any(np.all(line_data["normal"] == 0.0, axis=1))
    assert np.any(line_data["verticality"] == 0.0)


def test_traceability_xyz_origin_and_outputs_are_preserved_and_finite(tmp_path: Path) -> None:
    xyz = np.column_stack((np.linspace(0.0, 5.0, 64), np.ones(64), np.linspace(2.0, 4.0, 64)))
    project, input_npz, input_json, output = make_project(tmp_path, xyz)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (input_npz, input_json)
    }

    result = run_geometry_feature_extraction(input_npz, input_json, output, project_root=project)
    source = load_npz(input_npz)
    data = load_npz(result.output_npz_path)
    payload = json.loads(result.output_json_path.read_text(encoding="utf-8"))

    for name in ("original_index", "sample_index", "xyz_local", "origin_xyz"):
        assert np.array_equal(data[name], source[name])
    assert all(np.isfinite(value).all() for value in data.values() if value.dtype.kind in "fiu")
    assert (
        payload["output_npz_sha256"]
        == hashlib.sha256(result.output_npz_path.read_bytes()).hexdigest()
    )
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before


def test_output_round_trip_default_no_overwrite_and_forbidden_directory(tmp_path: Path) -> None:
    xyz = np.column_stack((np.linspace(0.0, 5.0, 32), np.zeros(32), np.zeros(32)))
    project, input_npz, input_json, output = make_project(tmp_path, xyz)
    first = run_geometry_feature_extraction(input_npz, input_json, output, project_root=project)
    payload = json.loads(first.output_json_path.read_text(encoding="utf-8"))

    assert payload["statistics"]["output_point_count"] == 32
    with pytest.raises(GeometryFeatureError) as exists_error:
        run_geometry_feature_extraction(input_npz, input_json, output, project_root=project)
    assert exists_error.value.error.code == "FEATURE_OUTPUT_EXISTS"
    with pytest.raises(GeometryFeatureError) as forbidden_error:
        run_geometry_feature_extraction(
            input_npz,
            input_json,
            project / "outputs" / "pointcloud",
            project_root=project,
        )
    assert forbidden_error.value.error.code == "FEATURE_OUTPUT_FORBIDDEN"


def test_cancellation_cleans_temporary_outputs(tmp_path: Path) -> None:
    xyz = np.column_stack((np.linspace(0.0, 10.0, 64), np.zeros(64), np.zeros(64)))
    project, input_npz, input_json, output = make_project(tmp_path, xyz)
    token = CancellationToken()

    with pytest.raises(GeometryFeatureError) as error:
        run_geometry_feature_extraction(
            input_npz,
            input_json,
            output,
            config=GeometryFeatureConfig(query_batch_size=8),
            reporter=CancelAfterFirstReporter(token),
            cancellation_token=token,
            project_root=project,
        )

    assert error.value.error.code == "FEATURE_CANCELLED"
    assert not list(output.glob("*.tmp.*")) if output.exists() else True
    assert not (output / "scene_simple_geometry_features.npz").exists()


def test_queries_are_bounded_and_no_las_read_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xyz = np.column_stack((np.linspace(0.0, 10.0, 37), np.zeros(37), np.zeros(37)))
    project, input_npz, input_json, output = make_project(tmp_path, xyz)
    real_tree = geometry.cKDTree
    query_sizes: list[tuple[int, int]] = []

    class TrackingTree:
        def __init__(self, coordinates: np.ndarray, **kwargs: object) -> None:
            self.tree = real_tree(coordinates, **kwargs)

        def query(self, coordinates: np.ndarray, *, k: int, workers: int):
            query_sizes.append((len(coordinates), k))
            return self.tree.query(coordinates, k=k, workers=workers)

    def fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A2A must not read LAS point records")

    monkeypatch.setattr(geometry, "cKDTree", TrackingTree)
    monkeypatch.setattr(laspy, "read", fail_read)
    monkeypatch.setattr(laspy.lasreader.LasReader, "read", fail_read)
    run_geometry_feature_extraction(
        input_npz,
        input_json,
        output,
        config=GeometryFeatureConfig(query_batch_size=7),
        project_root=project,
    )

    assert query_sizes
    assert max(size for size, _k in query_sizes) <= 7
    assert max(k for _size, k in query_sizes) <= 33


def test_public_loader_reloads_relocated_a2a_and_freezes_arrays(tmp_path: Path) -> None:
    result = _published_geometry(tmp_path, np.column_stack((np.arange(12), np.zeros(12), np.zeros(12))))
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (result.output_npz_path, result.output_json_path)
    }
    loaded, features = load_geometry_features(result.output_npz_path, result.output_json_path)
    assert loaded == result
    assert features.point_count == 12
    assert not features.xyz_local.flags.writeable
    assert not features.eigenvalues.flags.writeable
    with pytest.raises(ValueError):
        features.linearity[0] = np.float32(0.0)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    npz_path, json_path = relocated / result.output_npz_path.name, relocated / result.output_json_path.name
    shutil.copy2(result.output_npz_path, npz_path)
    shutil.copy2(result.output_json_path, json_path)
    relocated_result, relocated_features = load_geometry_features(npz_path, json_path)
    assert relocated_result.output_npz_path == result.output_npz_path
    assert np.array_equal(relocated_features.xyz_local, features.xyz_local)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "version", "hash", "size"])
def test_public_loader_rejects_a2a_json_failures(tmp_path: Path, mutation: str) -> None:
    result = _published_geometry(tmp_path, np.column_stack((np.arange(12), np.zeros(12), np.zeros(12))))
    payload = json.loads(result.output_json_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload.pop("arrays")
    elif mutation == "unknown":
        payload["unexpected"] = True
    elif mutation == "version":
        payload["algorithm_version"] = "unsupported"
    elif mutation == "hash":
        payload["output_npz_sha256"] = "0" * 64
    else:
        payload["output_npz_bytes"] += 1
    result.output_json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GeometryFeatureError):
        load_geometry_features(result.output_npz_path, result.output_json_path)


@pytest.mark.parametrize("mutation", ["missing", "dtype", "shape", "object", "nonfinite"])
def test_public_loader_rejects_invalid_a2a_arrays(tmp_path: Path, mutation: str) -> None:
    result = _published_geometry(tmp_path, np.column_stack((np.arange(12), np.zeros(12), np.zeros(12))))
    arrays = load_npz(result.output_npz_path)
    if mutation == "missing":
        arrays.pop("linearity")
    elif mutation == "dtype":
        arrays["linearity"] = arrays["linearity"].astype(np.float64)
    elif mutation == "shape":
        arrays["linearity"] = arrays["linearity"].reshape((-1, 1))
    elif mutation == "object":
        arrays["linearity"] = np.asarray(["bad"] * len(arrays["linearity"]), dtype=object)
    else:
        arrays["xyz_local"][0, 0] = np.nan
    np.savez_compressed(result.output_npz_path, **arrays)
    _refresh_geometry_integrity(result.output_json_path, result.output_npz_path)
    with pytest.raises(GeometryFeatureError):
        load_geometry_features(result.output_npz_path, result.output_json_path)


def test_public_loader_preserves_contract_allowed_invalid_a2a_features(tmp_path: Path) -> None:
    result = _published_geometry(tmp_path, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32))
    _loaded, features = load_geometry_features(result.output_npz_path, result.output_json_path)
    assert np.any(features.quality_flags & int(FeatureQualityFlag.INSUFFICIENT_NEIGHBORS))
    assert np.isfinite(features.principal_direction).all()


@pytest.mark.parametrize(
    "mutation",
    [
        "shape",
        "dtype",
        "unit",
        "quality_counts",
        "formulas",
        "origin_string",
        "origin_bool",
        "origin_large_int",
    ],
)
def test_public_loader_rejects_tampered_a2a_deterministic_metadata(
    tmp_path: Path, mutation: str
) -> None:
    result = _published_geometry(tmp_path, np.column_stack((np.arange(12), np.zeros(12), np.zeros(12))))
    payload = json.loads(result.output_json_path.read_text(encoding="utf-8"))
    if mutation == "shape":
        payload["arrays"]["linearity"]["shape"] = [99]
    elif mutation == "dtype":
        payload["arrays"]["linearity"]["dtype"] = "float64"
    elif mutation == "unit":
        payload["arrays"]["linearity"]["unit"] = "m"
    elif mutation == "quality_counts":
        payload["quality_flag_counts"]["NONE"] += 1
    elif mutation == "formulas":
        payload["formulas"]["height_m"] = "tampered"
    elif mutation == "origin_string":
        payload["origin_xyz"][0] = "10.0"
    elif mutation == "origin_bool":
        payload["origin_xyz"][0] = True
    else:
        payload["origin_xyz"][0] = 10**400
    result.output_json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GeometryFeatureError):
        load_geometry_features(result.output_npz_path, result.output_json_path)


def test_public_loader_wraps_a2a_integrity_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _published_geometry(tmp_path, np.column_stack((np.arange(12), np.zeros(12), np.zeros(12))))

    def fail_sha(_path: Path) -> str:
        raise OSError("synthetic hash read failure")

    monkeypatch.setattr(geometry, "_sha256_file", fail_sha)
    with pytest.raises(GeometryFeatureError):
        load_geometry_features(result.output_npz_path, result.output_json_path)


@pytest.mark.parametrize("mutation", ["negative_original", "duplicate_sample", "linearity", "density"])
def test_public_loader_rejects_invalid_a2a_identity_and_semantics(
    tmp_path: Path, mutation: str
) -> None:
    result = _published_geometry(tmp_path, np.column_stack((np.arange(12), np.zeros(12), np.zeros(12))))
    arrays = load_npz(result.output_npz_path)
    if mutation == "negative_original":
        arrays["original_index"][0] = -1
    elif mutation == "duplicate_sample":
        arrays["sample_index"][1] = arrays["sample_index"][0]
    elif mutation == "linearity":
        arrays["linearity"][0] = np.float32(1.1)
    else:
        arrays["local_density"][0] = np.float32(-1.0)
    np.savez_compressed(result.output_npz_path, **arrays)
    _refresh_geometry_integrity(result.output_json_path, result.output_npz_path)
    with pytest.raises(GeometryFeatureError):
        load_geometry_features(result.output_npz_path, result.output_json_path)


def test_public_loader_freezes_every_a2a_contract_array(tmp_path: Path) -> None:
    result = _published_geometry(tmp_path, np.column_stack((np.arange(12), np.zeros(12), np.zeros(12))))
    _loaded, features = load_geometry_features(result.output_npz_path, result.output_json_path)
    assert all(
        not getattr(features, name).flags.writeable
        for name in features.__dataclass_fields__
        if isinstance(getattr(features, name), np.ndarray)
    )
