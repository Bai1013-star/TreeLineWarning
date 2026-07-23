from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.wires import reconstruct
from algorithms.wires.reconstruct import (
    WireReconstructionError,
    generate_wire_models_preview,
    load_wire_models,
    run_wire_reconstruction,
)
from core.contracts import (
    CancellationToken,
    WireConfirmationChannel,
    WireReconstructionConfig,
    WireSeedConfig,
    WireSeparationConfig,
    WireSeparationResult,
    WireSeparationStatistics,
    WireTrackSource,
    WireTrackStatus,
    WireTrackSummary,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(track_id: int, wire_id: int, xyz: np.ndarray) -> WireTrackSummary:
    return WireTrackSummary(
        track_id=track_id,
        wire_id=wire_id,
        status=WireTrackStatus.CONFIRMED,
        source=WireTrackSource.AUTOMATIC,
        point_count=len(xyz),
        longitudinal_span_m=float(np.ptp(xyz[:, 0])),
        occupied_slice_count=max(1, len(xyz)),
        coverage_ratio=1.0,
        mean_candidate_score=0.9,
        high_candidate_fraction=1.0,
        medium_candidate_fraction=0.0,
        low_candidate_fraction=0.0,
        direction_consistency=1.0,
        transverse_residual=0.0,
        curvature_proxy=0.0,
        maximum_gap_m=0.0,
        branch_conflict_count=0,
        median_u=0.0,
        median_v=0.0,
        first_s=float(xyz[0, 0]),
        actual_main_direction=(1.0, 0.0, 0.0),
        axis_angle_deg=0.0,
        confirmation_channel=WireConfirmationChannel.OBVIOUS,
        start_xyz=tuple(float(value) for value in xyz[0]),
        end_xyz=tuple(float(value) for value in xyz[-1]),
        quality_flags=0,
    )


def make_a4(
    tmp_path: Path, groups: list[tuple[int, np.ndarray]], *, permutation: np.ndarray | None = None
) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    directory = root / "outputs" / "wires"
    directory.mkdir(parents=True)
    xyz = np.vstack([item[1] for item in groups]).astype(np.float32)
    ids = np.concatenate(
        [np.full(len(points), wire_id, dtype=np.int32) for wire_id, points in groups]
    )
    tracks = np.concatenate(
        [np.full(len(points), index, dtype=np.int32) for index, (_, points) in enumerate(groups)]
    )
    if permutation is not None:
        xyz, ids, tracks = xyz[permutation], ids[permutation], tracks[permutation]
    count, track_count = len(xyz), len(groups)
    original = np.arange(1000, 1000 + count, dtype=np.int64)
    if permutation is not None:
        original = original[permutation]
    ordered = np.concatenate(
        [np.flatnonzero(ids == wire_id) for wire_id, _ in sorted(groups)]
    ).astype(np.int64)
    offsets = np.asarray(
        [0, *np.cumsum([np.count_nonzero(ids == wire_id) for wire_id, _ in sorted(groups)])],
        dtype=np.int64,
    )
    npz = directory / "scene_simple_wire_instances.npz"
    np.savez_compressed(
        npz,
        source_id=np.asarray("scene_simple"),
        xyz_local=xyz,
        original_index=original,
        sample_index=np.arange(count, dtype=np.int64),
        track_id_by_point=tracks,
        wire_id_by_point=ids,
        assignment_score=np.full(count, 0.9, dtype=np.float32),
        assignment_quality_flags=np.zeros(count, dtype=np.uint16),
        assignment_reason_flags=np.zeros(count, dtype=np.uint16),
        ordered_wire_point_indices=ordered,
        wire_offsets=offsets,
        track_status=np.full(track_count, int(WireTrackStatus.CONFIRMED), dtype=np.uint8),
        track_wire_id=np.asarray([wire_id for wire_id, _ in groups], dtype=np.int32),
        track_actual_main_direction=np.tile(
            np.array([[1, 0, 0]], dtype=np.float32), (track_count, 1)
        ),
    )
    summaries = tuple(
        _summary(index, wire_id, points) for index, (wire_id, points) in enumerate(groups)
    )
    stats = WireSeparationStatistics(
        count, count, 1, 1, track_count, track_count, track_count, 0, 0, count, count, 0, 0, 0, 0.0
    )
    result = WireSeparationResult(
        "scene_simple",
        "synthetic",
        root / "outputs/candidates/in.npz",
        root / "outputs/candidates/in.json",
        "0" * 64,
        "0" * 64,
        npz,
        directory / "scene_simple_wire_instances.json",
        _sha(npz),
        npz.stat().st_size,
        (0, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        WireSeparationConfig(source_id="scene_simple", expected_input_point_count=count),
        WireSeedConfig(),
        stats,
        {},
        summaries,
        {},
        {},
        {},
        {},
        {},
        (),
        "test-a4",
        "2026-01-01T00:00:00+00:00",
    )
    js = directory / "scene_simple_wire_instances.json"
    js.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    return root, npz, js


def _line(start: float, stop: float, y: float = 0.0) -> np.ndarray:
    x = np.arange(start, stop + 1e-9, 0.1, dtype=np.float32)
    return np.column_stack((x, np.full_like(x, y), 5.0 + 0.02 * (x - x.mean()) ** 2))


def _noisy_thin_wire() -> np.ndarray:
    x = np.arange(0.0, 8.0, 0.1, dtype=np.float32)
    z = 5.0 + 0.03 * np.sin(x)
    z[20::17] += 0.12 * np.array([1, -1, 1, -1], dtype=np.float32)[: len(z[20::17])]
    return np.column_stack((x, np.zeros_like(x), z))


def _sag(
    length: float = 8.0, count: int = 161, noise: float = 0.0, sag_amplitude: float = 0.45
) -> np.ndarray:
    x = np.linspace(0.0, length, count, dtype=np.float32)
    z = 5.0 - sag_amplitude * 4.0 * (x / length) * (1.0 - x / length)
    if noise:
        index = np.arange(count, dtype=np.float32)
        z += noise * (0.65 * np.sin(1.71 * index) + 0.35 * np.cos(2.93 * index))
    return np.column_stack((x, np.zeros_like(x), z))


def _tight_supported_bend() -> np.ndarray:
    x = np.linspace(0.0, 4.0, 801, dtype=np.float32)
    return np.column_stack((x, np.zeros_like(x), 5.0 + 0.85 * np.tanh((x - 2.0) / 0.01)))


def _unsupported_oscillation() -> np.ndarray:
    x = np.arange(0.0, 8.0, 0.1, dtype=np.float32)
    keep = np.ones(len(x), dtype=bool)
    keep[[6, 44, 52, 66, 71]] = False
    x = x[keep]
    index = np.arange(len(x), dtype=np.float32)
    z = 5.0 + 0.02 * np.sin(1.71 * index) + 0.01 * np.cos(2.93 * index)
    for position, delta in ((1.1, -0.3375), (2.5, 0.3625), (3.1, 0.275), (4.6, 0.11875)):
        z[np.argmin(np.abs(x - position))] += delta
    return np.column_stack((x, np.zeros_like(x), z))


def test_confirmed_wires_are_reconstructed_reloaded_and_kept_separate(tmp_path: Path) -> None:
    root, npz, js = make_a4(tmp_path, [(5, _line(0, 2)), (9, _line(0, 2, 0.55))])
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    loaded, arrays = load_wire_models(
        result.output_npz_relpath and root / result.output_npz_relpath,
        root / result.output_json_relpath,
    )
    assert [item.wire_id for item in loaded.wires] == [5, 9]
    assert arrays["wire_id"].tolist() == [5, 9]
    assert all(item.status.name == "CONFIRMED_MODEL" for item in loaded.wires)
    assert all(item.wire_continuous_distance_ready for item in loaded.wires)


def test_large_gap_splits_observed_segments_without_samples_inside_gap(tmp_path: Path) -> None:
    root, npz, js = make_a4(tmp_path, [(4, np.vstack((_line(0, 0.6), _line(3.0, 3.6))))])
    config = WireReconstructionConfig(
        gap_absolute_min_m=0.25, gap_absolute_max_m=0.8, minimum_segment_span_m=0.2
    )
    result = run_wire_reconstruction(
        npz,
        js,
        root / "outputs/wire_models",
        config=config,
        project_root=root,
        generate_preview=False,
    )
    _, arrays = load_wire_models(
        root / result.output_npz_relpath, root / result.output_json_relpath
    )
    assert len(arrays["segment_wire_id"]) == 2
    assert arrays["segment_distance_ready"].all()
    assert bool(arrays["wire_has_distance_ready_segments"][0])
    assert not bool(arrays["wire_continuous_distance_ready"][0])
    samples = arrays["sampled_curve_xyz"][:, 0]
    assert not np.any((samples > 0.8) & (samples < 2.8))
    assert result.wires[0].status.name == "REVIEW_REQUIRED"


def test_scipy_exception_uses_explicit_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _line(0, 1.2))])
    monkeypatch.setattr(
        reconstruct,
        "splprep",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced")),
    )
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    assert result.wires[0].status.name == "FALLBACK_MODEL"
    assert result.wires[0].method.name in {"SMOOTHED_POLYLINE", "PIECEWISE_LINEAR"}
    assert "spline_exception" in result.wires[0].fallback_or_review_reason


def test_support_relative_rule_rejects_unsupported_interpolating_oscillation(
    tmp_path: Path,
) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _unsupported_oscillation())])
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    segment = result.segments[0]
    candidates = segment["spline_candidates"]
    assert candidates[0]["smoothing_value"] == 0.0
    assert not candidates[0]["accepted"]
    assert "turn_angle_spike_detected" in candidates[0]["warnings"]
    assert "unsupported_arc_inflation_detected" in candidates[0]["warnings"]
    assert "unsupported_oscillation" in candidates[0]["rejection_reasons"]
    assert candidates[0]["arc_length_inflation_ratio"] > 1.02
    assert segment["selected_smoothing_value"] == pytest.approx(0.002)
    assert segment["segment_distance_ready"]
    assert result.wires[0].method.name == "CUBIC_BSPLINE"


def test_clean_curve_keeps_least_smoothed_candidate(tmp_path: Path) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _line(0, 2.0))])
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    segment = result.segments[0]
    assert segment["selected_smoothing_value"] == 0.0
    assert segment["spline_candidates"][0]["accepted"]


@pytest.mark.parametrize("sag_amplitude", [0.12, 0.45, 1.0])
def test_clean_sags_keep_least_smoothed_candidate(tmp_path: Path, sag_amplitude: float) -> None:
    root, npz, js = make_a4(tmp_path / str(sag_amplitude), [(0, _sag(sag_amplitude=sag_amplitude))])
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    assert result.segments[0]["selected_smoothing_value"] == 0.0
    assert result.segments[0]["segment_distance_ready"]


def test_support_residual_does_not_depend_on_published_sample_spacing(tmp_path: Path) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _sag(noise=0.02))])
    first = run_wire_reconstruction(
        npz,
        js,
        root / "outputs/wire_models",
        config=WireReconstructionConfig(sample_spacing_m=0.05),
        project_root=root,
        generate_preview=False,
    )
    second = run_wire_reconstruction(
        npz,
        js,
        root / "outputs/wire_models",
        config=WireReconstructionConfig(sample_spacing_m=0.10),
        project_root=root,
        generate_preview=False,
        overwrite=True,
    )
    left, right = first.segments[0], second.segments[0]
    assert left["selected_smoothing_value"] == right["selected_smoothing_value"]
    assert left["support_residual_p95_m"] == pytest.approx(right["support_residual_p95_m"])
    assert left["nearest_sample_residual_stats_m"][2] != pytest.approx(
        right["nearest_sample_residual_stats_m"][2]
    )


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_scale_equivalent_supported_sag_keeps_normalized_decision(
    tmp_path: Path, scale: float
) -> None:
    points = _sag(length=8.0 * scale, count=161, noise=0.02 * scale)
    root, npz, js = make_a4(tmp_path / str(scale), [(0, points)])
    config = WireReconstructionConfig(
        representative_bin_m=0.10 * scale,
        minimum_segment_span_m=0.30 * scale,
        gap_absolute_min_m=0.35 * scale,
        gap_absolute_max_m=1.00 * scale,
        maximum_residual_m=0.15 * scale,
        sample_spacing_m=0.05 * scale,
    )
    result = run_wire_reconstruction(
        npz,
        js,
        root / "outputs/wire_models",
        config=config,
        project_root=root,
        generate_preview=False,
    )
    assert result.segments[0]["selected_smoothing_value"] == 0.0
    assert result.segments[0]["segment_distance_ready"]


def test_supported_tight_bend_is_not_rejected_for_turn_angle_alone(tmp_path: Path) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _tight_supported_bend())])
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    first = result.segments[0]["spline_candidates"][0]
    assert "turn_angle_spike_detected" in first["warnings"]
    assert "unsupported_arc_inflation_detected" not in first["warnings"]
    assert first["accepted"]
    assert result.segments[0]["selected_smoothing_value"] == 0.0


def test_supported_alternating_noise_is_marked_for_review_not_rejected_as_spline_only(
    tmp_path: Path,
) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _noisy_thin_wire())])
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    first = result.segments[0]["spline_candidates"][0]
    assert first["accepted"]
    assert "turn_angle_spike_detected" in first["warnings"]
    assert result.wires[0].status.name == "REVIEW_REQUIRED"


def test_candidate_selection_is_deterministic_after_input_row_reorder(tmp_path: Path) -> None:
    points = _noisy_thin_wire()
    first_root, first_npz, first_json = make_a4(tmp_path / "first", [(0, points)])
    permutation = np.random.default_rng(42).permutation(len(points))
    second_root, second_npz, second_json = make_a4(
        tmp_path / "second", [(0, points)], permutation=permutation
    )
    first = run_wire_reconstruction(
        first_npz,
        first_json,
        first_root / "outputs/wire_models",
        project_root=first_root,
        generate_preview=False,
    )
    second = run_wire_reconstruction(
        second_npz,
        second_json,
        second_root / "outputs/wire_models",
        project_root=second_root,
        generate_preview=False,
    )
    assert (
        first.segments[0]["selected_smoothing_value"]
        == second.segments[0]["selected_smoothing_value"]
    )
    _, first_arrays = load_wire_models(
        first_root / first.output_npz_relpath, first_root / first.output_json_relpath
    )
    _, second_arrays = load_wire_models(
        second_root / second.output_npz_relpath, second_root / second.output_json_relpath
    )
    assert np.array_equal(first_arrays["sampled_curve_xyz"], second_arrays["sampled_curve_xyz"])


def test_support_residual_rejection_is_explicit_and_not_distance_ready(tmp_path: Path) -> None:
    points = _sag(count=241, noise=0.04)
    root, npz, js = make_a4(tmp_path, [(0, points)])
    config = WireReconstructionConfig(maximum_residual_m=0.0001)
    result = run_wire_reconstruction(
        npz,
        js,
        root / "outputs/wire_models",
        config=config,
        project_root=root,
        generate_preview=False,
    )
    assert result.wires[0].status.name == "FAILED"
    assert "spline_rejected_high_residual" in result.wires[0].fallback_or_review_reason


def test_insufficient_two_point_segment_is_piecewise_but_not_ready(tmp_path: Path) -> None:
    points = np.array([[0.0, 0.0, 5.0], [0.5, 0.0, 5.0]], dtype=np.float32)
    root, npz, js = make_a4(tmp_path, [(0, points)])
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    _, arrays = load_wire_models(
        root / result.output_npz_relpath, root / result.output_json_relpath
    )
    assert arrays["segment_method"].tolist() == [2]
    assert not bool(arrays["segment_distance_ready"][0])
    assert result.wires[0].status.name == "FAILED"


def test_cancellation_occurs_before_any_publication(tmp_path: Path) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _line(0, 1.0))])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(WireReconstructionError, match="cancelled"):
        run_wire_reconstruction(
            npz,
            js,
            root / "outputs/wire_models",
            project_root=root,
            cancellation_token=token,
            generate_preview=False,
        )
    assert not (root / "outputs/wire_models").exists()


def test_output_preview_protection_and_relative_metadata(tmp_path: Path) -> None:
    root, npz, js = make_a4(tmp_path, [(0, _line(0, 1.0))])
    with pytest.raises(WireReconstructionError, match="approved directory"):
        run_wire_reconstruction(
            npz, js, root / "outside", project_root=root, generate_preview=False
        )
    result = run_wire_reconstruction(
        npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
    )
    metadata = json.loads((root / result.output_json_relpath).read_text("utf-8"))
    assert "E:" not in json.dumps(metadata) and "D:" not in json.dumps(metadata)
    preview = generate_wire_models_preview(
        root / result.output_npz_relpath,
        root / result.output_json_relpath,
        root / "outputs/previews/preview.html",
        project_root=root,
    )
    assert preview.is_file()
    with pytest.raises(WireReconstructionError):
        run_wire_reconstruction(
            npz, js, root / "outputs/wire_models", project_root=root, generate_preview=False
        )


def test_no_las_a3_or_dense_matrix_api() -> None:
    source = inspect.getsource(reconstruct)
    assert "laspy" not in source
    assert "distance_matrix" not in source
    assert "cdist(" not in source
