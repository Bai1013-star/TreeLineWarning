from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation import instances
from core.contracts import (
    CancellationToken,
    LocalScaleState,
    VegetationCandidateLevel,
    VegetationClusterStatus,
    VegetationInstanceConfig,
    VegetationInstanceReasonFlag,
    VegetationManualActionRecord,
    VegetationManualLog,
)


def _arrays(
    points: np.ndarray, levels: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = len(points)
    if levels is None:
        levels = np.full(n, VegetationCandidateLevel.HIGH, np.uint8)
    v1 = {
        "original_index": np.arange(100, 100 + n, dtype=np.int64),
        "sample_index": np.arange(n, dtype=np.int64),
        "xyz_local": points.astype(np.float32),
        "origin_xyz": np.zeros(3, np.float64),
        "wire_id_by_point": np.full(n, -1, np.int32),
        "vegetation_mask": levels != 0,
        "vegetation_level": levels.astype(np.uint8),
        "automatic_vegetation_level": levels.astype(np.uint8),
        "vegetation_rule_score": np.full(n, 0.8, np.float32),
        "vegetation_confidence": np.full(n, 0.85, np.float32),
        "hag_state": np.zeros(n, np.uint8),
        "vegetation_evidence_flags": np.zeros(n, np.uint16),
        "vegetation_reason_flags": np.zeros(n, np.uint16),
        "vegetation_quality_flags": np.zeros(n, np.uint16),
        "manual_provenance_flags": np.zeros(n, np.uint8),
        "manual_action": np.zeros(n, np.uint8),
    }
    a2a = {
        "linearity": np.full(n, 0.2, np.float32),
        "planarity": np.full(n, 0.2, np.float32),
        "scattering": np.full(n, 0.2, np.float32),
        "verticality": np.full(n, 0.5, np.float32),
        "local_density": np.full(n, 1000.0, np.float32),
    }
    a2b = {
        "height_agl_m": points[:, 2].astype(np.float32),
        "raw_height_agl_m": points[:, 2].astype(np.float32),
        "ground_distance_m": np.zeros(n, np.float32),
    }
    return v1, a2a, a2b


def _volumetric_grid(scale: float = 1.0) -> np.ndarray:
    x, y, z = np.meshgrid(
        np.linspace(-0.12, 0.12, 4),
        np.linspace(-0.12, 0.12, 4),
        np.linspace(0.1, 0.55, 4),
        indexing="ij",
    )
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())) * scale


def _seeded_crown(scale: float, angle: float, density: int) -> np.ndarray:
    """Bounded, non-constant synthetic crown used across scale/rotation cases."""
    rng = np.random.default_rng(731)
    points: list[np.ndarray] = []
    while len(points) < density:
        point = rng.uniform([-0.40, -0.40, 0.0], [0.40, 0.40, 0.80])
        if (point[0] / 0.40) ** 2 + (point[1] / 0.40) ** 2 + ((point[2] - 0.40) / 0.40) ** 2 <= 1:
            points.append(point)
    cloud = np.asarray(points) * scale
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    return cloud @ rotation.T


@pytest.mark.parametrize("scale", (0.5, 1.0, 2.0), ids=("scale-0.5", "scale-1", "scale-2"))
@pytest.mark.parametrize(
    "angle", (0.0, np.pi / 6, 7 * np.pi / 18), ids=("rot-0", "rot-30", "rot-70")
)
@pytest.mark.parametrize("density", (200, 400), ids=("sparse", "dense"))
@pytest.mark.parametrize("hag_state", ("reliable", "uncertain", "unavailable"))
def test_parameterized_crown_generalization_matrix(
    scale: float, angle: float, density: int, hag_state: str
) -> None:
    points = _seeded_crown(scale, angle, density)
    v1, a2a, a2b = _arrays(points)
    a2a["scattering"] = np.clip(0.48 + 0.12 * (points[:, 2] / max(scale, 1e-9)), 0, 1).astype(
        np.float32
    )
    a2a["planarity"] = np.clip(0.18 + 0.05 * np.abs(points[:, 0] / max(scale, 1e-9)), 0, 1).astype(
        np.float32
    )
    a2a["verticality"] = np.clip(
        0.35 + 0.15 * np.abs(points[:, 1] / max(scale, 1e-9)), 0, 1
    ).astype(np.float32)
    v1["vegetation_rule_score"] = np.clip(
        0.72 + 0.16 * (points[:, 2] / max(0.8 * scale, 1e-9)), 0, 1
    ).astype(np.float32)
    v1["vegetation_confidence"] = np.clip(
        0.74 + 0.12 * np.cos(points[:, 0] / max(scale, 1e-9)), 0, 1
    ).astype(np.float32)
    states = {"reliable": 0, "uncertain": 1, "unavailable": 2}
    v1["hag_state"] = np.full(len(points), states[hag_state], np.uint8)
    cfg = VegetationInstanceConfig(
        local_scale_lower_m=0.03 * scale,
        local_scale_upper_m=0.12 * scale,
        core_radius_lower_m=0.035 * scale,
        core_radius_upper_m=0.16 * scale,
        voxel_lower_m=0.08 * scale,
        voxel_upper_m=0.20 * scale,
        min_core_extent_m=0.15 * scale,
    )
    arrays, descriptors, _counts = instances._group_arrays(
        v1, a2a, a2b, cfg, VegetationManualLog(), None, None
    )
    assert descriptors and all(d["status"] != "UNASSIGNED" for d in descriptors)
    assert np.all(
        arrays["distance_ready_by_point"]
        == (arrays["cluster_status_by_point"] == int(VegetationClusterStatus.CONFIRMED))
    )
    if density == 400 and hag_state == "reliable":
        assert any(d["status"] == "CONFIRMED" for d in descriptors)


@pytest.mark.parametrize("family", ("wall", "pole", "compact"))
def test_synthetic_artificial_families_are_never_distance_ready(family: str) -> None:
    if family == "wall":
        x, z = np.meshgrid(np.linspace(-0.45, 0.45, 18), np.linspace(0.0, 0.8, 12))
        points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    elif family == "pole":
        theta, z = np.meshgrid(
            np.linspace(0, 2 * np.pi, 12, endpoint=False), np.linspace(0, 0.9, 24)
        )
        points = np.column_stack(
            (0.035 * np.cos(theta.ravel()), 0.035 * np.sin(theta.ravel()), z.ravel())
        )
    else:
        x, y, z = np.meshgrid(
            np.linspace(-0.45, 0.45, 16), np.linspace(-0.35, 0.35, 14), np.linspace(0.0, 0.18, 4)
        )
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    v1, a2a, a2b = _arrays(points)
    a2a["planarity"] = np.full(len(points), 0.82 if family != "pole" else 0.25, np.float32)
    a2a["scattering"] = np.full(len(points), 0.05, np.float32)
    a2a["verticality"] = np.full(len(points), 0.88 if family == "pole" else 0.25, np.float32)
    arrays, descriptors, _counts = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    assert descriptors and not arrays["distance_ready_by_point"].any()
    assert all(d["status"] == "REVIEW_REQUIRED" for d in descriptors)


def test_quotient_bridge_priority_uses_bounded_general_evidence() -> None:
    points = np.vstack(
        (
            np.column_stack((np.zeros(4), np.arange(4) * 0.01, np.zeros(4))),
            np.column_stack((np.full(4, 0.22), np.arange(4) * 0.01, np.full(4, 0.8))),
        )
    )
    v1, a2a, a2b = _arrays(points)
    v1["hag_state"][:] = 0
    a2b["height_agl_m"][:4] = 0.0
    a2b["height_agl_m"][4:] = 0.8
    rows = np.arange(len(points), dtype=np.int64)
    candidates = instances._bridge_candidates(
        rows,
        [(3, 4, 1.0)],
        rows,
        points,
        np.full(len(points), 0.05, np.float32),
        v1,
        a2a,
        a2b,
        [set() for _ in rows],
        VegetationInstanceConfig(),
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["categories"] >= 3 and 0 <= candidate["evidence"] <= 1
    assert candidate["priority"][-2:] == (103, 104)


def test_vertical_gap_cut_retains_two_supported_sides() -> None:
    lower = _volumetric_grid()
    points = np.vstack((lower, lower + np.array([0.0, 0.0, 1.0])))
    v1, _a2a, a2b = _arrays(points)
    cfg = VegetationInstanceConfig()
    raw, scale, state = instances._local_scales(points, cfg, None)
    rows = np.flatnonzero(state != LocalScaleState.LOCAL_SCALE_UNAVAILABLE)
    edges, neighbours = instances._core_edges(rows, points, scale, v1, a2b, cfg, None)
    children, unsafe = instances._vertical_gap_candidate(
        rows, edges, rows, points, scale, v1, neighbours, cfg
    )
    assert not unsafe and children is not None and len(children) >= 2
    assert all(len(child) >= cfg.min_core_points for child in children)


def test_missing_mixed_region_fields_are_exact_zeros() -> None:
    v1, a2a, a2b = _arrays(_volumetric_grid())
    _arrays_out, descriptors, _counts = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    descriptor = descriptors[0]
    assert descriptor["region_support_crown"] == 0
    assert descriptor["region_support_surface"] == 0
    assert descriptor["normalized_spatial_separation"] == 0
    assert descriptor["morphology_change_score"] == 0


def test_local_scale_support_states_and_cancellation() -> None:
    cfg = VegetationInstanceConfig()
    raw, scale, state = instances._local_scales(_volumetric_grid(), cfg, None)
    assert np.isfinite(raw).all() and np.isfinite(scale).all()
    assert np.all(state != LocalScaleState.LOCAL_SCALE_UNAVAILABLE)
    raw, scale, state = instances._local_scales(
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], float), cfg, None
    )
    assert np.all(state == LocalScaleState.LOCAL_SCALE_UNAVAILABLE)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(instances.VegetationInstanceError):
        instances._local_scales(_volumetric_grid(), cfg, token)


def test_grouping_ids_pca_centres_and_readiness() -> None:
    v1, a2a, a2b = _arrays(_volumetric_grid())
    arrays, descs, counts = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    assert len(descs) == 1
    assert arrays["cluster_id_by_point"].min() == 0
    assert np.all(
        arrays["distance_ready_by_point"]
        == (arrays["cluster_status_by_point"] == int(VegetationClusterStatus.CONFIRMED))
    )
    assert descs[0]["horizontal_length_m"] >= descs[0]["horizontal_width_m"]
    assert np.allclose(descs[0]["centroid_xyz"], _volumetric_grid().mean(axis=0))
    assert counts["confirmed_count"] + counts["review_count"] + counts["provisional_count"] == 1


def test_low_does_not_form_core_or_bridge() -> None:
    points = np.vstack(
        (
            _volumetric_grid()[:16],
            _volumetric_grid()[16:] + [1.0, 0, 0],
            np.array([[0.5, 0, 0.3], [0.55, 0, 0.3]]),
        )
    )
    levels = np.full(len(points), VegetationCandidateLevel.HIGH, np.uint8)
    levels[-2:] = VegetationCandidateLevel.LOW
    v1, a2a, a2b = _arrays(points, levels)
    arrays, _descs, _counts = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    assert np.all(arrays["cluster_id_by_point"][-2:] == -1)


def test_fragmentation_and_deterministic_label_tie() -> None:
    points = _volumetric_grid()
    v1, a2a, a2b = _arrays(points)
    arrays, descs, _ = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    assert descs[0]["fragmentation"] == pytest.approx(0)
    d = {
        "unresolved_bridge": False,
        "fragmentation": 0.0,
        "crown_like_evidence": 0.55,
        "low_shrub_evidence": 0.55,
        "shallow_horizontal_suspicion": 0.55,
        "planar_structure_suspicion": 0.55,
        "linear_structure_suspicion": 0.55,
    }
    assert instances._label(d, VegetationInstanceConfig()).name == "PLANAR_STRUCTURE_SUSPECT"


def test_scale_and_rotation_consistency() -> None:
    statuses = []
    for scale in (0.5, 1.0, 2.0):
        points = _volumetric_grid(scale)
        for angle in (0.0, 0.7):
            rot = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            v1, a2a, a2b = _arrays(points @ rot.T)
            cfg = VegetationInstanceConfig(
                local_scale_lower_m=0.03 * scale,
                local_scale_upper_m=0.12 * scale,
                core_radius_lower_m=0.035 * scale,
                core_radius_upper_m=0.16 * scale,
                voxel_lower_m=0.08 * scale,
                voxel_upper_m=0.2 * scale,
                min_core_extent_m=0.15 * scale,
            )
            _arr, descs, _counts = instances._group_arrays(
                v1, a2a, a2b, cfg, VegetationManualLog(), None, None
            )
            statuses.append(descs[0]["status"])
    assert len(set(statuses)) == 1


def test_relocated_publish_reload(tmp_path: Path) -> None:
    v1, a2a, a2b = _arrays(_volumetric_grid())
    arrays, descs, counts = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    source = tmp_path / "project"
    out = source / "outputs" / "vegetation_instances"
    out.mkdir(parents=True)
    npz, js = (
        out / "scene_simple_vegetation_instances.npz",
        out / "scene_simple_vegetation_instances.json",
    )

    class V:
        source_id = "scene_simple"
        manifest_source_id = "raw"

    stats = instances.VegetationInstanceStatistics(64, 64, 64, 1, 1, 0, 0, 0, 64, 0, 0.1)
    hashes = {
        key: hashlib.sha256(key.encode()).hexdigest()
        for key in ("v1_npz", "v1_json", "a2a_npz", "a2a_json", "a2b_npz", "a2b_json")
    }
    instances._publish(
        arrays,
        descs,
        npz,
        js,
        VegetationInstanceConfig(),
        VegetationManualLog(),
        V(),
        hashes,
        stats,
        source,
    )
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    (relocated / npz.name).write_bytes(npz.read_bytes())
    (relocated / js.name).write_bytes(js.read_bytes())
    result, loaded = instances.load_vegetation_instances(relocated / js.name)
    assert result.output_npz_filename == npz.name and len(loaded["cluster_id_by_point"]) == 64
    payload = result.to_dict()
    payload["schema_version"] = "vegetation-instances-v1"
    with pytest.raises(ValueError):
        instances.VegetationInstanceResult.from_dict(payload)


def test_schema_v2_fixed_slice_arrays_and_ambiguity_readiness() -> None:
    v1, a2a, a2b = _arrays(_volumetric_grid())
    arrays, descs, _counts = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    flat = instances._flat_cluster_arrays([{"members": descs[0]["member_rows"]}], descs)
    assert flat["cluster_slice_valid"].shape == (1, 6)
    assert flat["cluster_slice_horizontal_dimensions_m"].shape == (1, 6, 2)
    assert np.all(
        arrays["distance_ready_by_point"]
        == (arrays["cluster_status_by_point"] == int(VegetationClusterStatus.CONFIRMED))
    )


def test_shallow_cuboid_is_review_not_forced_reject() -> None:
    x, y, z = np.meshgrid(
        np.linspace(-0.6, 0.6, 16),
        np.linspace(-0.5, 0.5, 14),
        np.linspace(0.08, 0.24, 3),
        indexing="ij",
    )
    v1, a2a, a2b = _arrays(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
    arrays, descs, counts = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
    )
    assert counts["rejected_count"] == 0
    assert descs[0]["status"] == VegetationClusterStatus.REVIEW_REQUIRED.name
    assert not arrays["distance_ready_by_point"].any()


def test_manual_confirmation_replay_marks_override() -> None:
    digest = "b" * 64
    log = VegetationManualLog(
        digest,
        digest,
        digest,
        "scene_simple",
        "raw",
        (VegetationManualActionRecord("confirm_cluster", cluster_ids=(0,)),),
    )
    v1, a2a, a2b = _arrays(_volumetric_grid())
    arrays, descs, _ = instances._group_arrays(
        v1, a2a, a2b, VegetationInstanceConfig(), log, None, None
    )
    assert descs[0]["status"] == VegetationClusterStatus.CONFIRMED.name
    assert np.all(
        arrays["assignment_reason_flags"] & int(VegetationInstanceReasonFlag.MANUAL_STATUS_OVERRIDE)
    )


def test_synthetic_scale_density_matrix() -> None:
    baseline = _volumetric_grid()
    outcomes = []
    for scale in (0.5, 1.0, 2.0):
        for density in (1, 2):
            points = baseline * scale
            if density == 2:
                points = np.vstack((points, points + scale * np.array([0.03, 0.02, 0.015])))
            v1, a2a, a2b = _arrays(points)
            cfg = VegetationInstanceConfig(
                local_scale_lower_m=0.03 * scale,
                local_scale_upper_m=0.12 * scale,
                core_radius_lower_m=0.035 * scale,
                core_radius_upper_m=0.16 * scale,
                voxel_lower_m=0.08 * scale,
                voxel_upper_m=0.20 * scale,
                min_core_extent_m=0.15 * scale,
            )
            _output, descs, _counts = instances._group_arrays(
                v1, a2a, a2b, cfg, VegetationManualLog(), None, None
            )
            outcomes.append(descs[0]["status"])
    assert outcomes
    assert set(outcomes) <= {
        VegetationClusterStatus.CONFIRMED.name,
        VegetationClusterStatus.REVIEW_REQUIRED.name,
    }


def _box(shape: tuple[int, int, int], extent: tuple[float, float, float]) -> np.ndarray:
    axes = [
        np.linspace(-size / 2, size / 2, count) for count, size in zip(shape, extent, strict=True)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_conservative_synthetic_family_matrix() -> None:
    crown = _box((4, 4, 5), (0.24, 0.24, 0.65))
    compact = _box((6, 6, 6), (0.45, 0.45, 0.65))
    trunk = _box((3, 3, 14), (0.10, 0.10, 1.4))
    families = {
        "vertical_crown": crown,
        "compact_crown": compact,
        "sparse_canopy": crown[::3],
        "low_shrub": _box((8, 8, 3), (0.50, 0.50, 0.28)),
        "trunk": trunk,
        "shallow_cuboid": _box((12, 10, 3), (1.2, 1.0, 0.18)),
        "planar_wall": _box((12, 2, 9), (1.2, 0.05, 0.9)),
        "pole": _box((3, 3, 15), (0.08, 0.08, 1.5)),
        "fence": _box((16, 2, 4), (1.6, 0.06, 0.5)),
        "touching_crowns": np.vstack((compact - [0.18, 0, 0], compact + [0.18, 0, 0])),
        "narrow_bridge": np.vstack(
            (compact - [0.6, 0, 0], compact + [0.6, 0, 0], _box((5, 1, 1), (0.5, 0.01, 0.01)))
        ),
        "trunk_crown_gap": np.vstack((trunk[:60], compact + [0, 0, 1.0])),
        "vegetation_building_boundary": np.vstack(
            (crown, _box((10, 10, 2), (0.9, 0.9, 0.08)) + [0.35, 0, 0])
        ),
    }
    observed = {}
    for name, points in families.items():
        v1, a2a, a2b = _arrays(points)
        if name == "sparse_canopy":
            v1["hag_state"][::2] = 1
        output, descs, _counts = instances._group_arrays(
            v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
        )
        observed[name] = [d["status"] for d in descs]
        assert not np.any(
            output["distance_ready_by_point"]
            & (output["cluster_status_by_point"] != VegetationClusterStatus.CONFIRMED)
        )
    assert observed["vertical_crown"]
    assert observed["shallow_cuboid"]
    assert observed["trunk"] == [VegetationClusterStatus.REVIEW_REQUIRED.name]
    assert observed["planar_wall"] == [VegetationClusterStatus.REVIEW_REQUIRED.name]
    assert observed["pole"] == [VegetationClusterStatus.REVIEW_REQUIRED.name]
    assert observed["vegetation_building_boundary"] == [
        VegetationClusterStatus.REVIEW_REQUIRED.name
    ]
    rng = np.random.default_rng(19)
    for variant in ("moderate_noise", "uneven_density", "local_gap", "slope", "outlier"):
        points = crown.copy()
        if variant == "moderate_noise":
            points += rng.normal(0, 0.01, points.shape)
        elif variant == "uneven_density":
            points = np.vstack((points[::2], points[: len(points) // 3]))
        elif variant == "local_gap":
            points = points[(points[:, 2] < 0.45) | (points[:, 2] > 0.75)]
        elif variant == "slope":
            points[:, 2] += 0.15 * points[:, 0]
        else:
            points = np.vstack((points, np.array([[3.0, 3.0, 1.0]])))
        v1, a2a, a2b = _arrays(points)
        v1["hag_state"][::5] = 1
        output, _descs, _counts = instances._group_arrays(
            v1, a2a, a2b, VegetationInstanceConfig(), VegetationManualLog(), None, None
        )
        assert not np.any(
            output["distance_ready_by_point"]
            & (output["cluster_status_by_point"] != VegetationClusterStatus.CONFIRMED)
        )
