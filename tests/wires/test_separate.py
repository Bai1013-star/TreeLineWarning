from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.wires import separate
from algorithms.wires.separate import (
    WireSeparationError,
    _actual_track_direction,
    _confirmation_channel,
    run_wire_separation,
)
from core.contracts import (
    CancellationToken,
    WireConfirmationChannel,
    WireSeedConfig,
    WireSeparationConfig,
    WireSeparationResult,
    WireTrackStatus,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {name: stream[name] for name in stream.files}


def make_a3(
    tmp_path: Path,
    xyz: np.ndarray,
    levels: np.ndarray,
    *,
    original_index: np.ndarray | None = None,
    sample_index: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    directory = root / "outputs" / "candidates"
    directory.mkdir(parents=True)
    npz = directory / "scene_simple_wire_candidates.npz"
    js = directory / "scene_simple_wire_candidates.json"
    count = len(xyz)
    original_index = (
        np.arange(1000, 1000 + count, dtype=np.int64)
        if original_index is None
        else np.asarray(original_index, dtype=np.int64)
    )
    sample_index = (
        np.arange(count, dtype=np.int64)
        if sample_index is None
        else np.asarray(sample_index, dtype=np.int64)
    )
    np.savez_compressed(
        npz,
        source_id=np.asarray("scene_simple"),
        manifest_source_id=np.asarray("raw_scene"),
        xyz_local=np.asarray(xyz, dtype=np.float32),
        origin_xyz=np.zeros(3, dtype=np.float64),
        original_index=original_index,
        sample_index=sample_index,
        candidate_mask=np.ones(count, dtype=bool),
        candidate_score=np.where(levels == 3, 0.95, np.where(levels == 2, 0.70, 0.55)).astype(
            np.float32
        ),
        candidate_level=np.asarray(levels, dtype=np.uint8),
        candidate_source=np.ones(count, dtype=np.uint8),
        evidence_flags=np.zeros(count, dtype=np.uint16),
        quality_flags=np.zeros(count, dtype=np.uint16),
        principal_direction=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
    )
    js.write_text(
        json.dumps(
            {
                "source_id": "scene_simple",
                "manifest_source_id": "raw_scene",
                "output_npz_sha256": _sha(npz),
            }
        ),
        encoding="utf-8",
    )
    return root, npz, js


def config_for(count: int, **overrides: object) -> WireSeparationConfig:
    values: dict[str, object] = {
        "expected_input_point_count": count,
        "min_axis_seed_points": 3,
        "min_axis_span_m": 0.2,
        "weak_cluster_min_points": 1,
        "confirmed_cluster_min_points": 1,
        "medium_seed_min_consecutive_slices": 3,
        "medium_seed_min_span_m": 0.3,
    }
    values.update(overrides)
    return WireSeparationConfig(**values)


def parallel_wires(length_m: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    s = np.arange(int(length_m / 0.1) + 1, dtype=np.float32) * 0.1
    first = np.column_stack((s, np.zeros_like(s), np.full_like(s, 5.0)))
    second = np.column_stack((s, np.full_like(s, 0.55), np.full_like(s, 5.2)))
    return np.vstack((first, second)), np.full(len(s) * 2, 3, dtype=np.uint8)


def test_confirmed_wires_are_separate_and_reload(tmp_path: Path) -> None:
    xyz, levels = parallel_wires()
    root, npz, js = make_a3(tmp_path, xyz, levels)
    result = run_wire_separation(
        npz,
        js,
        root / "outputs" / "wires",
        config=config_for(len(xyz)),
        project_root=root,
        generate_preview=False,
    )
    data = _load(result.output_npz_path)
    restored = WireSeparationResult.from_dict(
        json.loads(result.output_json_path.read_text("utf-8"))
    )
    first_count = len(xyz) // 2
    assert result.statistics.confirmed_wire_count == 2
    assert np.array_equal(data["track_id_by_point"], data["wire_id_by_point"])
    assert len(np.unique(data["wire_id_by_point"][:first_count])) == 1
    assert len(np.unique(data["wire_id_by_point"][first_count:])) == 1
    assert data["wire_id_by_point"][0] != data["wire_id_by_point"][first_count]
    assert np.array_equal(
        data["ordered_track_point_indices"], np.sort(data["ordered_track_point_indices"])
    )
    assert np.array_equal(
        data["ordered_wire_point_indices"], np.sort(data["ordered_wire_point_indices"])
    )
    assert restored.output_npz_sha256 == _sha(result.output_npz_path)


def test_fine_channel_confirms_but_short_track_has_no_wire_id(tmp_path: Path) -> None:
    obvious_s = np.arange(121, dtype=np.float32) * 0.1
    fine_s = np.arange(71, dtype=np.float32) * 0.1
    obvious = np.column_stack((obvious_s, np.zeros_like(obvious_s), np.full_like(obvious_s, 5.0)))
    fine = np.column_stack((fine_s, np.full_like(fine_s, 0.6), np.full_like(fine_s, 4.0)))
    short_s = np.arange(39, dtype=np.float32) * 0.1
    short = np.column_stack((short_s, np.full_like(short_s, 1.2), np.full_like(short_s, 3.5)))
    xyz = np.vstack((obvious, fine, short))
    levels = np.concatenate(
        (
            np.full(len(obvious), 3, dtype=np.uint8),
            np.full(len(fine), 2, dtype=np.uint8),
            np.full(len(short), 2, dtype=np.uint8),
        )
    )
    root, npz, js = make_a3(tmp_path, xyz, levels)
    result = run_wire_separation(
        npz,
        js,
        root / "outputs" / "wires",
        config=config_for(len(xyz)),
        project_root=root,
        generate_preview=False,
    )
    data = _load(result.output_npz_path)
    assert result.statistics.confirmed_wire_count == 2
    assert np.all(data["wire_id_by_point"][: len(obvious)] >= 0)
    assert np.all(data["wire_id_by_point"][len(obvious) : len(obvious) + len(fine)] >= 0)
    short_track = data["track_id_by_point"][-len(short) :]
    assert np.all(short_track >= 0)
    assert np.all(data["wire_id_by_point"][-len(short) :] == -1)
    short_summary = next(item for item in result.summaries if item.track_id == int(short_track[0]))
    assert short_summary.status != WireTrackStatus.CONFIRMED
    assert short_summary.confirmation_channel == WireConfirmationChannel.NONE


def test_confirmation_boundaries_direction_and_numeric_invalid() -> None:
    config = config_for(4)
    base = dict(
        span=10.0,
        coverage=0.8,
        direction_consistency=0.99,
        transverse_residual=0.04,
        curvature_proxy=0.45,
        maximum_gap_m=0.2,
        branch_conflict_count=0,
        axis_angle_deg=2.0,
        mean_candidate_score=0.9,
        high_candidate_fraction=0.95,
        medium_low_fraction=0.05,
        numeric_invalid=False,
        config=config,
    )
    assert _confirmation_channel(**base) == WireConfirmationChannel.OBVIOUS
    assert (
        _confirmation_channel(**(base | {"axis_angle_deg": 2.01})) == WireConfirmationChannel.NONE
    )
    assert (
        _confirmation_channel(**(base | {"curvature_proxy": 0.46})) == WireConfirmationChannel.NONE
    )
    fine = base | {
        "span": 6.0,
        "coverage": 0.55,
        "curvature_proxy": 1.05,
        "mean_candidate_score": 0.63,
        "high_candidate_fraction": 0.0,
        "medium_low_fraction": 0.4,
    }
    assert _confirmation_channel(**fine) == WireConfirmationChannel.FINE
    assert _confirmation_channel(**(fine | {"span": 5.99})) == WireConfirmationChannel.NONE
    assert (
        _confirmation_channel(**(fine | {"numeric_invalid": True})) == WireConfirmationChannel.NONE
    )
    positive, angle, invalid = _actual_track_direction(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.4, 0.0]]),
        np.array([1.0, 0.0, 0.0]),
    )
    negative, reverse_angle, reverse_invalid = _actual_track_direction(
        np.array([[2.0, 0.4, 0.0], [1.0, 0.2, 0.0], [0.0, 0.0, 0.0]]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert not invalid and not reverse_invalid
    assert positive[0] > 0.0 and negative[0] > 0.0
    assert angle == pytest.approx(reverse_angle)
    assert angle > 0.0
    sentinel, _, invalid = _actual_track_direction(np.zeros((3, 3)), np.array([1.0, 0.0, 0.0]))
    assert invalid and np.array_equal(sentinel, np.zeros(3))


def test_track_and_wire_ids_are_stable_after_input_row_reordering(tmp_path: Path) -> None:
    xyz, levels = parallel_wires()
    original_index = np.arange(1000, 1000 + len(xyz), dtype=np.int64)
    sample_index = np.arange(len(xyz), dtype=np.int64)
    first_root, first_npz, first_json = make_a3(
        tmp_path / "first",
        xyz,
        levels,
        original_index=original_index,
        sample_index=sample_index,
    )
    permutation = np.random.default_rng(42).permutation(len(xyz))
    second_root, second_npz, second_json = make_a3(
        tmp_path / "second",
        xyz[permutation],
        levels[permutation],
        original_index=original_index[permutation],
        sample_index=sample_index[permutation],
    )
    first = run_wire_separation(
        first_npz,
        first_json,
        first_root / "outputs" / "wires",
        config=config_for(len(xyz)),
        project_root=first_root,
        generate_preview=False,
    )
    second = run_wire_separation(
        second_npz,
        second_json,
        second_root / "outputs" / "wires",
        config=config_for(len(xyz)),
        project_root=second_root,
        generate_preview=False,
    )
    first_data, second_data = _load(first.output_npz_path), _load(second.output_npz_path)
    first_order = np.argsort(first_data["original_index"])
    second_order = np.argsort(second_data["original_index"])
    assert np.array_equal(
        first_data["track_id_by_point"][first_order],
        second_data["track_id_by_point"][second_order],
    )
    assert np.array_equal(
        first_data["wire_id_by_point"][first_order],
        second_data["wire_id_by_point"][second_order],
    )


def test_manual_seed_preview_and_protection(tmp_path: Path) -> None:
    xyz, levels = parallel_wires()
    root, npz, js = make_a3(tmp_path, xyz, levels)
    before = {path: _sha(path) for path in (npz, js)}
    seed = WireSeedConfig(
        seed_original_indices=(1000,), seed_group_ids=(7,), excluded_original_indices=(1121,)
    )
    result = run_wire_separation(
        npz,
        js,
        root / "outputs" / "wires",
        config=config_for(len(xyz)),
        seeds=seed,
        project_root=root,
    )
    data = _load(result.output_npz_path)
    assert data["track_id_by_point"][121] == -1
    assert data["wire_id_by_point"][121] == -1
    assert {path: _sha(path) for path in before} == before
    preview = root / "outputs" / "previews" / "scene_simple_wire_instances_preview.html"
    html = preview.read_text(encoding="utf-8")
    assert "confirmed" in html and "review tracklets" in html and "legendonly" in html


def test_overwrite_cancel_path_and_no_global_distance_matrix(tmp_path: Path) -> None:
    xyz, levels = parallel_wires()
    root, npz, js = make_a3(tmp_path, xyz, levels)
    config = config_for(len(xyz))
    result = run_wire_separation(
        npz,
        js,
        root / "outputs" / "wires",
        config=config,
        project_root=root,
        generate_preview=False,
    )
    with pytest.raises(WireSeparationError) as exists:
        run_wire_separation(
            npz,
            js,
            root / "outputs" / "wires",
            config=config,
            project_root=root,
            generate_preview=False,
        )
    assert exists.value.error.code == "WIRE_OUTPUT_EXISTS"
    token = CancellationToken()
    token.cancel()
    with pytest.raises(WireSeparationError) as cancelled:
        run_wire_separation(
            npz,
            js,
            root / "outputs" / "wires" / "cancelled",
            config=config,
            project_root=root,
            cancellation_token=token,
            generate_preview=False,
        )
    assert cancelled.value.error.code == "WIRE_CANCELLED"
    with pytest.raises(WireSeparationError) as escaped:
        run_wire_separation(
            npz, js, root / "outside", config=config, project_root=root, generate_preview=False
        )
    assert escaped.value.error.code == "WIRE_OUTPUT_FORBIDDEN"
    assert result.output_npz_path.is_file()
    source = inspect.getsource(separate)
    assert "laspy" not in source
    assert "distance_matrix" not in source
    assert "cdist(" not in source
