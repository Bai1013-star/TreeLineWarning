from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.candidates import wire
from algorithms.candidates.wire import (
    WireCandidateError,
    generate_wire_candidate_preview,
    run_wire_candidate_selection,
)
from core.contracts import (
    CancellationToken,
    FeatureQualityFlag,
    GroundQualityFlag,
    ManualSelectionConfig,
    WireCandidateConfig,
    WireCandidateResult,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {name: stream[name] for name in stream.files}


def make_project(tmp_path: Path) -> tuple[Path, tuple[Path, Path, Path, Path, Path, Path]]:
    project = tmp_path / "project"
    pointcloud = project / "outputs" / "pointcloud"
    features = project / "outputs" / "features"
    pointcloud.mkdir(parents=True)
    features.mkdir()
    count = 9
    xyz = np.column_stack(
        (
            np.arange(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
            np.array([7, 7, 7, 0.5, 7, 7, 7, 7, 7], dtype=np.float32),
        )
    )
    origin = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    original_index = np.arange(100, 100 + count, dtype=np.int64)
    sample_index = np.arange(count, dtype=np.int64)
    source_id, manifest_source_id = "scene_simple", "raw_scene"
    a1_npz = pointcloud / "scene_simple_preprocessed.npz"
    a1_json = pointcloud / "scene_simple_preprocessed.json"
    a2a_npz = features / "scene_simple_geometry_features.npz"
    a2a_json = features / "scene_simple_geometry_features.json"
    a2b_npz = features / "scene_simple_ground_hag.npz"
    a2b_json = features / "scene_simple_ground_hag.json"
    shared = {
        "source_id": np.asarray(source_id),
        "manifest_source_id": np.asarray(manifest_source_id),
        "origin_xyz": origin,
        "original_index": original_index,
        "sample_index": sample_index,
    }
    np.savez_compressed(a1_npz, **shared, xyz_local=xyz)
    a1_json.write_text(
        json.dumps(
            {
                "source_id": source_id,
                "manifest_source_id": manifest_source_id,
                "origin_xyz": origin.tolist(),
                "output_npz_path": str(a1_npz.resolve()),
                "output_json_path": str(a1_json.resolve()),
            }
        ),
        encoding="utf-8",
    )
    geometry_flags = np.zeros(count, dtype=np.uint16)
    np.savez_compressed(
        a2a_npz,
        **shared,
        xyz_local=xyz.copy(),
        linearity=np.array(
            [0.95, 0.50, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95], dtype=np.float32
        ),
        scattering=np.array(
            [0.01, 0.01, 0.30, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32
        ),
        planarity=np.array(
            [0.10, 0.80, 0.30, 0.10, 0.80, 0.10, 0.10, 0.10, 0.10], dtype=np.float32
        ),
        local_density=np.array([500, 500, 500, 500, 500, 500, 500, 500, 20], dtype=np.float32),
        height_m=xyz[:, 2].copy(),
        principal_direction=np.tile(np.array([[1, 0, 0]], dtype=np.float32), (count, 1)),
        quality_flags=geometry_flags,
    )
    a2a_json.write_text(
        json.dumps(
            {
                "source_id": source_id,
                "manifest_source_id": manifest_source_id,
                "origin_xyz": origin.tolist(),
                "output_npz_path": str(a2a_npz.resolve()),
                "output_json_path": str(a2a_json.resolve()),
                "input_a1_npz_sha256": _sha256(a1_npz),
                "input_a1_json_sha256": _sha256(a1_json),
                "output_npz_sha256": _sha256(a2a_npz),
            }
        ),
        encoding="utf-8",
    )
    ground_flags = np.zeros(count, dtype=np.uint16)
    ground_flags[5] = int(GroundQualityFlag.HIGH_LOCAL_GROUND_VARIATION)
    ground_flags[6] = int(GroundQualityFlag.INVALID_GROUND_ESTIMATE)
    ground_flags[7] = int(GroundQualityFlag.BELOW_GROUND_TOLERANCE)
    np.savez_compressed(
        a2b_npz,
        **shared,
        height_agl_m=np.array([7, 7, 7, 0.5, 7, 5, np.nan, 0, 7], dtype=np.float32),
        raw_height_agl_m=np.array([7, 7, 7, 0.5, 7, 5, np.nan, -1, 7], dtype=np.float32),
        quality_flags=ground_flags,
    )
    a2b_json.write_text(
        json.dumps(
            {
                "source_id": source_id,
                "manifest_source_id": manifest_source_id,
                "origin_xyz": origin.tolist(),
                "output_npz_path": str(a2b_npz.resolve()),
                "output_json_path": str(a2b_json.resolve()),
                "input_a1_npz_sha256": _sha256(a1_npz),
                "input_a1_json_sha256": _sha256(a1_json),
                "input_a2a_npz_sha256": _sha256(a2a_npz),
                "input_a2a_json_sha256": _sha256(a2a_json),
                "output_npz_sha256": _sha256(a2b_npz),
            }
        ),
        encoding="utf-8",
    )
    return project, (a1_npz, a1_json, a2a_npz, a2a_json, a2b_npz, a2b_json)


def run_candidate(tmp_path: Path, **kwargs):
    project, paths = make_project(tmp_path)
    result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates",
        project_root=project,
        generate_preview=False,
        **kwargs,
    )
    return project, paths, result, _load(result.output_npz_path)


def test_rule_scores_guard_density_hag_and_reload(tmp_path: Path) -> None:
    _project, _paths, result, data = run_candidate(tmp_path)
    restored = WireCandidateResult.from_dict(json.loads(result.output_json_path.read_text("utf-8")))
    assert data["candidate_mask"][0]
    assert not data["candidate_mask"][1]
    assert not data["candidate_mask"][2]
    assert data["rule_score"][0] == pytest.approx(1.0)
    assert data["linearity_score"][1] == 0.0
    assert data["scattering_score"][2] == 0.0
    assert data["height_score"][5] == pytest.approx(0.93, abs=0.01)
    assert data["height_score"][6] == pytest.approx(1.0)
    assert data["height_score"][7] == pytest.approx(1.0)
    assert data["density_score"][8] == 0.0
    assert restored.output_npz_sha256 == _sha256(result.output_npz_path)
    assert result.statistics.hag_normal_count == 6
    assert result.statistics.hag_mixed_count == 1
    assert result.statistics.hag_fallback_count == 2


def test_normal_ambiguity_has_no_penalty_but_principal_ambiguity_does(tmp_path: Path) -> None:
    project, paths = make_project(tmp_path)
    a2a = _load(paths[2])
    a2a["quality_flags"][0] = int(FeatureQualityFlag.NORMAL_DIRECTION_AMBIGUOUS)
    a2a["quality_flags"][1] = int(FeatureQualityFlag.PRINCIPAL_DIRECTION_AMBIGUOUS)
    np.savez_compressed(paths[2], **a2a)
    payload = json.loads(paths[3].read_text("utf-8"))
    payload["output_npz_sha256"] = _sha256(paths[2])
    paths[3].write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(paths[5].read_text("utf-8"))
    payload["input_a2a_npz_sha256"] = _sha256(paths[2])
    payload["input_a2a_json_sha256"] = _sha256(paths[3])
    paths[5].write_text(json.dumps(payload), encoding="utf-8")
    result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates",
        project_root=project,
        generate_preview=False,
    )
    data = _load(result.output_npz_path)
    assert data["quality_penalty"][0] == 0.0
    assert data["quality_penalty"][1] == pytest.approx(0.20)


def test_manual_modes_exclude_priority_and_numeric_invalid_default(tmp_path: Path) -> None:
    project, paths = make_project(tmp_path)
    a2a = _load(paths[2])
    a2a["quality_flags"][2] = int(FeatureQualityFlag.NUMERIC_INVALID)
    np.savez_compressed(paths[2], **a2a)
    a2a_payload = json.loads(paths[3].read_text("utf-8"))
    a2a_payload["output_npz_sha256"] = _sha256(paths[2])
    paths[3].write_text(json.dumps(a2a_payload), encoding="utf-8")
    a2b_payload = json.loads(paths[5].read_text("utf-8"))
    a2b_payload["input_a2a_npz_sha256"] = _sha256(paths[2])
    a2b_payload["input_a2a_json_sha256"] = _sha256(paths[3])
    paths[5].write_text(json.dumps(a2b_payload), encoding="utf-8")
    manual = ManualSelectionConfig(include_original_index=(101, 102), exclude_original_index=(101,))
    result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates",
        manual_selection=manual,
        mode="union",
        project_root=project,
        generate_preview=False,
    )
    data = _load(result.output_npz_path)
    assert not data["candidate_mask"][1]
    assert not data["candidate_mask"][2]
    assert data["rule_score"][2] == 0.0
    override = ManualSelectionConfig(
        include_original_index=(102,), allow_numeric_invalid_override=True
    )
    result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates" / "override",
        manual_selection=override,
        mode="manual_override",
        project_root=project,
        generate_preview=False,
    )
    data = _load(result.output_npz_path)
    assert data["candidate_mask"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert data["candidate_score"][2] == 1.0


def test_density_piecewise_boxes_intersection_and_candidate_source(tmp_path: Path) -> None:
    config = WireCandidateConfig()
    density = np.array([0, 50, 100, 150, 1500, 2250, 3000], dtype=np.float32)
    assert wire._density_score(density, config).tolist() == pytest.approx(
        [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0]
    )
    project, paths = make_project(tmp_path)
    manual = ManualSelectionConfig(include_boxes=(((0.9, -1.0, 6.0), (1.1, 1.0, 8.0)),))
    result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates",
        manual_selection=manual,
        mode="intersection",
        project_root=project,
        generate_preview=False,
    )
    data = _load(result.output_npz_path)
    assert data["candidate_mask"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert data["evidence_flags"][1] & int(wire.CandidateReasonFlag.MANUAL_INCLUDE)
    assert not data["candidate_mask"][1]
    manual = ManualSelectionConfig(include_boxes=(((0.9, -1.0, 6.0), (1.1, 1.0, 8.0)),))
    result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates" / "union",
        manual_selection=manual,
        mode="union",
        project_root=project,
        generate_preview=False,
    )
    data = _load(result.output_npz_path)
    assert data["candidate_mask"][1]
    assert data["candidate_source"][1] == int(wire.CandidateSource.MANUAL)


def test_mode_requirements_empty_output_and_input_protection(tmp_path: Path) -> None:
    project, paths = make_project(tmp_path)
    before = {path: _sha256(path) for path in paths}
    with pytest.raises(WireCandidateError, match="requires empty"):
        run_wire_candidate_selection(
            *paths,
            project / "outputs" / "candidates",
            manual_selection=ManualSelectionConfig(include_original_index=(100,)),
            project_root=project,
            generate_preview=False,
        )
    with pytest.raises(WireCandidateError, match="requires manual include"):
        run_wire_candidate_selection(
            *paths,
            project / "outputs" / "candidates",
            mode="intersection",
            project_root=project,
            generate_preview=False,
        )
    result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates",
        config=WireCandidateConfig(linearity_guard=0.96, linearity_full=0.99),
        project_root=project,
        generate_preview=False,
    )
    assert result.statistics.candidate_count == 0
    assert {path: _sha256(path) for path in before} == before


def test_output_overwrite_cancellation_path_escape_and_preview_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, paths, result, _data = run_candidate(tmp_path)
    with pytest.raises(WireCandidateError) as exists:
        run_wire_candidate_selection(
            *paths,
            project / "outputs" / "candidates",
            project_root=project,
            generate_preview=False,
        )
    assert exists.value.error.code == "CANDIDATE_OUTPUT_EXISTS"
    token = CancellationToken()
    token.cancel()
    with pytest.raises(WireCandidateError) as cancelled:
        run_wire_candidate_selection(
            *paths,
            project / "outputs" / "candidates" / "cancelled",
            project_root=project,
            cancellation_token=token,
            generate_preview=False,
        )
    assert cancelled.value.error.code == "CANDIDATE_CANCELLED"
    with pytest.raises(WireCandidateError) as escaped:
        run_wire_candidate_selection(
            *paths, project / "outside", project_root=project, generate_preview=False
        )
    assert escaped.value.error.code == "CANDIDATE_OUTPUT_FORBIDDEN"
    monkeypatch.setattr(
        wire,
        "generate_wire_candidate_preview",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("preview")),
    )
    preview_result = run_wire_candidate_selection(
        *paths,
        project / "outputs" / "candidates" / "preview-warning",
        project_root=project,
        generate_preview=True,
    )
    assert preview_result.output_npz_path.exists()
    assert preview_result.warnings == ("preview generation failed: preview",)
    assert result.output_json_path.exists()


def test_preview_sampling_is_deterministic_and_las_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, paths, result, _data = run_candidate(tmp_path)
    first = wire._evenly_spaced_indices(np.arange(100), 11)
    assert np.array_equal(first, wire._evenly_spaced_indices(np.arange(100), 11))
    top = wire._top_candidate_indices(
        np.array([1, 2, 3, 4]), np.array([0, 0.8, 0.9, 0.9, 0.5], dtype=np.float32), 2
    )
    assert top.tolist() == [2, 3]
    preview = generate_wire_candidate_preview(
        paths[0],
        result.output_npz_path,
        project / "outputs" / "previews" / "scene_simple_wire_candidates_preview.html",
        project_root=project,
    )
    assert preview.is_file()
    source = inspect.getsource(wire)
    assert "laspy" not in source
    assert "distance_matrix" not in source
    assert "cdist(" not in source
