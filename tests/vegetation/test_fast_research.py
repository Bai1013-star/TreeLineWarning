"""Focused safety and correspondence tests for the non-certified fast route."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
from algorithms.vegetation.fast_research import (
    FAST_RESEARCH_SCHEMA_VERSION,
    NON_CERTIFIED_INPUT_MARKER,
    NON_CERTIFIED_MARKER,
    FastGroundResult,
    FastVegetationInputResult,
    _build_fast_handoff,
    _build_fast_source_identity,
    _fast_v4_config,
    build_fast_vegetation_input,
    canonical_digest,
    fast_ground_content_digest,
    sha256_file,
    validate_fast_v4_arrays,
)
from algorithms.vegetation.fast_vegetation_ab import (
    FastVegetationEvidenceConfig,
    apply_fast_vegetation_hard_exclusions,
    classify_plan_b_geometry,
    fast_vegetation_config_digest,
)
from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees

from tests.vegetation.test_v4_tree_segmentation import make_tree_points


def _write_hag_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="HeightAboveGround", type=np.float32))
    data = laspy.LasData(header)
    data.x = np.asarray((0.0, 1.0, 2.0, 3.0), dtype=np.float64)
    data.y = np.zeros(4, dtype=np.float64)
    data.z = np.asarray((0.0, 0.2, 1.5, 2.5), dtype=np.float64)
    data.classification = np.asarray((2, 1, 1, 1), dtype=np.uint8)
    data["HeightAboveGround"] = np.asarray((0.0, 0.2, 1.5, 2.5), dtype=np.float32)
    data.write(path)


def _ground_result(source: Path, output: Path) -> FastGroundResult:
    source_sha = sha256_file(source)
    output_sha = sha256_file(output)
    return FastGroundResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_MARKER,
        source_las_path=str(source),
        source_las_sha256=source_sha,
        output_las_path=str(output),
        output_las_sha256=output_sha,
        algorithm_name="PDAL_SMRF_v1",
        hag_algorithm_name="PDAL_HAG_NN_v1",
        parameters={},
        config_digest="a" * 64,
        source_point_count=4,
        output_point_count=4,
        ground_class_2_count=1,
        non_ground_count=3,
        hag_valid_count=4,
        hag_min_m=0.0,
        hag_max_m=2.5,
        xyz_order_preserved=True,
        source_xyz_sha256="b" * 64,
        output_xyz_sha256="b" * 64,
        elapsed_seconds=0.1,
    )


def test_fast_ground_result_parsing_rejects_certification_marker(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "output.las"
    _write_hag_las(source)
    _write_hag_las(output)
    payload = _ground_result(source, output).to_dict()
    assert FastGroundResult.from_dict(payload).non_certified_marker == NON_CERTIFIED_MARKER
    payload["non_certified_marker"] = "FORMAL"
    try:
        FastGroundResult.from_dict(payload)
    except Exception as exc:
        assert "certification marker" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("formal marker must be rejected")


def test_fast_adapter_preserves_original_index_xyz_and_hag_alignment(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "output.las"
    _write_hag_las(source)
    _write_hag_las(output)
    status, result = build_fast_vegetation_input(
        _ground_result(source, output),
        tmp_path / "fast_input",
        source_id="scene_simple",
        origin_context=(0.0, 0.0, 0.0),
        minimum_hag_m=1.0,
    )
    assert status == "PASS"
    assert result.non_certified_marker == NON_CERTIFIED_INPUT_MARKER
    with np.load(result.output_npz_path, allow_pickle=False) as data:
        assert np.array_equal(data["original_index"], np.asarray((2, 3), dtype=np.int64))
        assert np.array_equal(data["xyz_local_m"][:, 0], np.asarray((2.0, 3.0)))
        assert np.array_equal(data["hag_m"], np.asarray((1.5, 2.5), dtype=np.float32))
        validate_fast_v4_arrays(data["original_index"], data["xyz_local_m"], data["hag_m"])


def test_fast_adapter_cache_reuses_only_valid_output(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "output.las"
    _write_hag_las(source)
    _write_hag_las(output)
    ground = _ground_result(source, output)
    root = tmp_path / "fast_input"
    assert (
        build_fast_vegetation_input(
            ground, root, source_id="scene_simple", origin_context=(0.0, 0.0, 0.0)
        )[0]
        == "PASS"
    )
    assert (
        build_fast_vegetation_input(
            ground, root, source_id="scene_simple", origin_context=(0.0, 0.0, 0.0)
        )[0]
        == "SKIPPED_FROM_CACHE"
    )
    (root / "fast_v4_input.npz").write_bytes(b"corrupt cache")
    assert (
        build_fast_vegetation_input(
            ground, root, source_id="scene_simple", origin_context=(0.0, 0.0, 0.0)
        )[0]
        == "PASS"
    )


def test_fast_configuration_digest_is_deterministic() -> None:
    assert canonical_digest({"a": [1, 2], "b": 3}) == canonical_digest({"b": 3, "a": [1, 2]})
    assert canonical_digest({"a": [1, 2], "b": 3}) != canonical_digest({"a": [1, 3], "b": 3})


def test_plan_b_hard_exclusions_are_exact_and_reach_no_v4_input() -> None:
    original = np.asarray((10, 11, 12, 13), dtype=np.int64)
    selection = apply_fast_vegetation_hard_exclusions(
        classification=np.asarray((2, 1, 1, 1), dtype=np.uint8),
        original_index=original,
        known_wire_original_index=np.asarray((12,), dtype=np.int64),
        vegetation_evidence_mask=np.asarray((True, True, True, False)),
    )
    assert np.array_equal(selection.ground_mask, np.asarray((True, False, False, False)))
    assert np.array_equal(selection.known_wire_mask, np.asarray((False, False, True, False)))
    assert np.array_equal(selection.retained_mask, np.asarray((False, True, False, False)))
    assert not np.any(selection.retained_mask & selection.ground_mask)
    assert not np.any(selection.retained_mask & selection.known_wire_mask)


def test_plan_b_wire_membership_uses_original_index_not_coordinate_proximity() -> None:
    selection = apply_fast_vegetation_hard_exclusions(
        classification=np.asarray((1, 1), dtype=np.uint8),
        original_index=np.asarray((40, 41), dtype=np.int64),
        known_wire_original_index=np.asarray((41,), dtype=np.int64),
        vegetation_evidence_mask=np.asarray((True, True)),
    )
    assert np.array_equal(selection.retained_mask, np.asarray((True, False)))


def test_plan_b_classifier_is_deterministic_and_rejects_invalid_configuration() -> None:
    values = np.asarray((0.059, 0.06, 0.2), dtype=np.float32)
    config = FastVegetationEvidenceConfig(knn=16, minimum_scattering=0.06)
    assert np.array_equal(
        classify_plan_b_geometry(values, config=config), np.asarray((False, True, True))
    )
    try:
        FastVegetationEvidenceConfig(minimum_scattering=1.1)
    except Exception as exc:
        assert "SCATTERING" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("invalid Plan-B configuration must fail")


def test_plan_b_config_digest_invalidates_cache_key_when_geometry_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "output.las"
    _write_hag_las(source)
    _write_hag_las(output)
    candidate_root = tmp_path / "candidate"
    _status, candidate = build_fast_vegetation_input(
        _ground_result(source, output),
        candidate_root,
        source_id="scene_simple",
        origin_context=(0.0, 0.0, 0.0),
    )
    w1_npz = tmp_path / "w1.npz"
    w1_json = tmp_path / "w1.json"
    w1_npz.write_bytes(b"w1-npz")
    w1_json.write_bytes(b"w1-json")
    first = fast_vegetation_config_digest(
        candidate,
        _ground_result(source, output),
        w1_npz=w1_npz,
        w1_json=w1_json,
        reference_a2a_npz=None,
        reference_v1_npz=None,
        config=FastVegetationEvidenceConfig(minimum_scattering=0.06),
    )
    second = fast_vegetation_config_digest(
        candidate,
        _ground_result(source, output),
        w1_npz=w1_npz,
        w1_json=w1_json,
        reference_a2a_npz=None,
        reference_v1_npz=None,
        config=FastVegetationEvidenceConfig(minimum_scattering=0.07),
    )
    assert first != second


def test_fast_ground_content_digest_ignores_timing(tmp_path: Path) -> None:
    source = tmp_path / "source.las"
    output = tmp_path / "output.las"
    _write_hag_las(source)
    _write_hag_las(output)
    first = _ground_result(source, output)
    second = FastGroundResult(**{**first.to_dict(), "elapsed_seconds": 99.0})
    assert fast_ground_content_digest(first) == fast_ground_content_digest(second)


def test_fast_v4_adapter_uses_existing_v4_and_preserves_correspondence(tmp_path: Path) -> None:
    points = make_tree_points(2.0, 2.0) + make_tree_points(9.0, 2.0)
    xyz = np.asarray(points, dtype=np.float64)
    original = np.arange(len(xyz), dtype=np.int64)
    hag = xyz[:, 2].astype(np.float32)
    vegetation = FastVegetationInputResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_INPUT_MARKER,
        ground_result_digest="a" * 64,
        config_digest="b" * 64,
        output_npz_path=str(tmp_path / "input.npz"),
        output_npz_sha256="c" * 64,
        point_count=len(original),
        source_id="scene_simple",
        source_las_path=str(tmp_path / "source.las"),
        source_point_count=len(original),
        source_file_sha256="d" * 64,
        origin_context=(0.0, 0.0, 0.0),
        selected_bounds_min_xyz=None,
        selected_bounds_max_xyz=None,
        minimum_hag_m=0.3,
        admission_voxel_size_m=None,
        original_index_sha256="e" * 64,
        xyz_sha256="f" * 64,
        hag_sha256="0" * 64,
        elapsed_seconds=0.0,
    )
    source = _build_fast_source_identity(
        vegetation,
        source_bounds_min_xyz=(0.0, 0.0, 0.0),
        source_bounds_max_xyz=(14.0, 6.0, 7.0),
    )
    handoff = _build_fast_handoff(
        source,
        original_index=original,
        xyz_local_m=xyz,
        hag_m=hag,
        candidate_reason_flags=np.zeros(len(original), dtype=np.uint32),
        bridge_digest="1" * 64,
    )
    result = segment_v4_trees(handoff, _fast_v4_config(maximum_points=len(original)))
    assert np.array_equal(result.original_index, original)
    assert np.array_equal(result.xyz_local_m, xyz)
    assert np.array_equal(result.hag_m, hag)
    assert result.config.status.value == "CALIBRATION_CANDIDATE"
