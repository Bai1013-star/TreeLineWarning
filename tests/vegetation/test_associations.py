from __future__ import annotations

import hashlib
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation import associations
from core.contracts import (
    BLOCKING_EVIDENCE_NAMES,
    VegetationAssociationConfig,
    VegetationAssociationProvenanceFlag,
    VegetationAssociationStatus,
    VegetationCandidateLevel,
)

_FAMILIES = (
    "uniform_crown",
    "density_transition",
    "occlusion_branch",
    "sparse_tall",
    "compact_dense",
    "touching_crowns",
    "crown_wall",
    "crown_pole",
    "vertical_overlap",
    "sloped_hag",
    "low_only_bridge",
    "anisotropic_scanner",
)
_OCCLUSION_NOISE = (
    ("none", 0.0),
    ("none", 0.006),
    ("moderate", 0.0),
    ("moderate", 0.006),
    ("strong", 0.0),
    ("strong", 0.006),
)


def _rotation(degrees: int) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _cloud(
    family: str,
    center: np.ndarray,
    scale: float,
    density: str,
    rotation: int,
    dropout_seed: int,
    occlusion: str,
    noise: float,
    side: int,
) -> np.ndarray:
    counts = {"sparse": 10, "nominal": 18, "dense": 30}
    count = counts[density] + (8 if family == "density_transition" and side else 0)
    rng = np.random.default_rng(10_000 + 101 * dropout_seed + 17 * side)
    cloud = rng.normal(0.0, 1.0, (count, 3))
    extent = np.array([0.18, 0.06, 0.06])
    if family == "sparse_tall":
        extent = np.array([0.05, 0.05, 0.36])
    if family == "occlusion_branch":
        extent = np.array([0.18, 0.035, 0.04])
    if family == "compact_dense":
        extent = np.array([0.05, 0.05, 0.05])
    if family == "crown_wall" and side:
        extent = np.array([0.30, 0.004, 0.22])
    if family == "crown_pole" and side:
        extent = np.array([0.012, 0.012, 0.42])
    if family == "anisotropic_scanner":
        extent = np.array([0.22, 0.025, 0.10])
    cloud *= extent * scale
    cloud = cloud @ _rotation(rotation).T + center
    if occlusion != "none":
        axis = 0 if family != "occlusion_branch" else 2
        fraction = 0.30 if occlusion == "moderate" else 0.55
        cutoff = np.quantile(cloud[:, axis], fraction)
        cloud = cloud[cloud[:, axis] >= cutoff]
    keep = np.random.default_rng(dropout_seed + 71 * side).random(len(cloud)) >= 0.12
    cloud = cloud[keep]
    if len(cloud) < 4:
        cloud = np.vstack(
            (cloud, center + np.eye(3, dtype=float)[: 4 - len(cloud)] * 0.003 * scale)
        )
    if noise:
        cloud = cloud + rng.normal(0.0, noise * scale, cloud.shape)
    return cloud


def _summary(
    cluster_id: int,
    cloud: np.ndarray,
    scale: float,
    density: str,
    *,
    parent: int = -1,
    density_multiplier: float = 1.0,
    reliable_hag: bool = True,
    planar: float = 0.0,
    slender: float = 0.0,
    compact: float = 0.0,
    profile: float = 0.82,
) -> associations._Summary:
    density_value = {"sparse": 700.0, "nominal": 1_000.0, "dense": 1_400.0}[density]
    direction, direction_reliability = associations._stable_direction(cloud)
    original = np.arange(len(cloud), dtype=np.int64) + cluster_id * 10_000
    return associations._Summary(
        cluster_id,
        np.arange(len(cloud), dtype=np.int64),
        cloud,
        original,
        cloud.min(0),
        cloud.max(0),
        cloud.mean(0),
        direction,
        direction_reliability,
        0.10 * scale,
        True,
        0.0,
        density_value * density_multiplier,
        1.0,
        reliable_hag,
        profile,
        0.85,
        planar,
        slender,
        compact,
        0.0,
        parent,
        False,
    )


def _fixture(
    family: str,
    scale: float = 1.0,
    density: str = "nominal",
    rotation: int = 0,
    hag: str = "reliable",
    dropout_seed: int = 0,
    occlusion: str = "none",
    noise: float = 0.0,
) -> tuple[
    associations._Summary, associations._Summary, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    spacing = {
        "uniform_crown": 0.04,
        "density_transition": 0.14,
        "occlusion_branch": 0.17,
        "sparse_tall": 0.16,
        "compact_dense": 0.08,
        "touching_crowns": 0.03,
        "crown_wall": 0.05,
        "crown_pole": 0.06,
        "vertical_overlap": 0.02,
        "sloped_hag": 0.12,
        "low_only_bridge": 1.05,
        "anisotropic_scanner": 0.14,
    }[family] * scale
    left_center = np.array([-spacing / 2, 0.0, 0.0])
    right_center = np.array([spacing / 2, 0.0, 0.0])
    if family == "vertical_overlap":
        right_center[2] = 1.0 * scale
    rotation_matrix = _rotation(rotation)
    left_center = left_center @ rotation_matrix.T
    right_center = right_center @ rotation_matrix.T
    reliable = hag == "reliable"
    parent = 17 if family == "uniform_crown" else -1
    left_cloud = _cloud(
        family, left_center, scale, density, rotation, dropout_seed, occlusion, noise, 0
    )
    right_cloud = _cloud(
        family, right_center, scale, density, rotation, dropout_seed, occlusion, noise, 1
    )
    if family == "uniform_crown":
        right_cloud = left_cloud + (right_center - left_center)
    left = _summary(
        0,
        left_cloud,
        scale,
        density,
        parent=parent,
        reliable_hag=reliable,
    )
    kwargs: dict[str, float | int | bool] = {"parent": parent, "reliable_hag": reliable}
    if family == "compact_dense":
        kwargs["compact"] = 0.95
    if family == "touching_crowns":
        kwargs["profile"] = 0.05
    if family == "crown_wall":
        kwargs["planar"] = 0.95
    if family == "crown_pole":
        kwargs["slender"] = 0.95
    right = _summary(
        1,
        right_cloud,
        scale,
        density,
        density_multiplier=1.5 if family == "density_transition" else 1.0,
        **kwargs,
    )
    xyz = np.vstack((left.xyz, right.xyz)).astype(np.float32)
    cluster = np.array([0] * len(left.xyz) + [1] * len(right.xyz), np.int32)
    levels = np.full(len(xyz), VegetationCandidateLevel.HIGH, np.uint8)
    if family == "low_only_bridge":
        bridge = np.linspace(left.center, right.center, 7)[1:-1].astype(np.float32)
        xyz = np.vstack((xyz, bridge))
        cluster = np.concatenate((cluster, np.full(len(bridge), -1, np.int32)))
        levels = np.concatenate(
            (levels, np.full(len(bridge), VegetationCandidateLevel.LOW, np.uint8))
        )
    return (
        left,
        right,
        {"xyz_local": xyz, "cluster_id_by_point": cluster},
        {"vegetation_level": levels},
    )


def _evaluate(family: str, **factors: object) -> dict[str, object]:
    left, right, v2, v1 = _fixture(family, **factors)
    return associations._evaluate_pair(left, right, v2, v1, {}, {}, VegetationAssociationConfig())


def test_all_statuses_and_conflict_uncertainty_semantics_are_reachable() -> None:
    strong = _evaluate("uniform_crown")
    possible = _evaluate("density_transition")
    blocked = _evaluate("crown_wall")
    left, right, v2, v1 = _fixture("uniform_crown")
    abstained = associations._evaluate_pair(
        left,
        replace(right, scale_valid=False, scale=0.0),
        v2,
        v1,
        {},
        {},
        VegetationAssociationConfig(),
    )
    assert [strong["status"], possible["status"], blocked["status"], abstained["status"]] == [
        int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION),
        int(VegetationAssociationStatus.POSSIBLE_REVIEW_ASSOCIATION),
        int(VegetationAssociationStatus.BLOCKED_BY_CONFLICT),
        int(VegetationAssociationStatus.ABSTAINED),
    ]
    unresolved = associations._evaluate_pair(
        replace(left, unresolved=True), right, v2, v1, {}, {}, VegetationAssociationConfig()
    )
    unavailable_hag = _evaluate("uniform_crown", hag="unavailable")
    assert unresolved["status"] != int(VegetationAssociationStatus.BLOCKED_BY_CONFLICT)
    assert unavailable_hag["status"] != int(VegetationAssociationStatus.BLOCKED_BY_CONFLICT)
    assert unresolved["blocking"][9] == 0.0
    assert unresolved["provenance"] & int(
        VegetationAssociationProvenanceFlag.SPLIT_LINEAGE_UNAVAILABLE
    )


def test_strong_requires_current_continuation_not_parent_alone() -> None:
    left, right, v2, v1 = _fixture("uniform_crown")
    strong = associations._evaluate_pair(left, right, v2, v1, {}, {}, VegetationAssociationConfig())
    without_parent = associations._evaluate_pair(
        replace(left, parent=-1),
        replace(right, parent=-1),
        v2,
        v1,
        {},
        {},
        VegetationAssociationConfig(),
    )
    one_channel = associations._evaluate_pair(
        replace(left, density=0.0, morphology=0.0, profile=0.0),
        replace(right, density=0.0, morphology=0.0, profile=0.0),
        v2,
        v1,
        {},
        {},
        VegetationAssociationConfig(),
    )
    assert strong["status"] == int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION)
    assert without_parent["status"] != int(VegetationAssociationStatus.BLOCKED_BY_CONFLICT)
    assert one_channel["status"] != int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION)
    assert _evaluate("vertical_overlap")["status"] == int(
        VegetationAssociationStatus.BLOCKED_BY_CONFLICT
    )


def test_decision_input_completeness_excludes_optional_history_and_hag() -> None:
    left, right, v2, v1 = _fixture("uniform_crown", hag="unavailable")
    optional_missing = associations._evaluate_pair(
        replace(left, unresolved=True), right, v2, v1, {}, {}, VegetationAssociationConfig()
    )
    invalid = associations._evaluate_pair(
        left,
        replace(right, scale_valid=False, scale=0.0),
        v2,
        v1,
        {},
        {},
        VegetationAssociationConfig(),
    )
    assert optional_missing["decision_input_completeness"] == 1.0
    assert invalid["decision_input_completeness"] == 0.0
    assert BLOCKING_EVIDENCE_NAMES[-1] == "explicit_contradictory_evidence"
    assert optional_missing["blocking"][9] == 0.0
    physical = _evaluate("low_only_bridge")
    assert physical["status"] == int(VegetationAssociationStatus.BLOCKED_BY_CONFLICT)
    assert physical["blocking"][9] == 0.0


def test_fixture_factors_materially_change_deterministic_data() -> None:
    sparse = _fixture("uniform_crown", density="sparse")
    dense = _fixture("uniform_crown", density="dense")
    rotated = _fixture("uniform_crown", rotation=70)
    dropped = _fixture("uniform_crown", dropout_seed=3)
    occluded = _fixture("uniform_crown", occlusion="strong")
    noisy = _fixture("uniform_crown", noise=0.006)
    assert len(sparse[0].xyz) < len(dense[0].xyz)
    assert not np.allclose(
        sparse[0].xyz[: min(len(sparse[0].xyz), len(rotated[0].xyz))],
        rotated[0].xyz[: min(len(sparse[0].xyz), len(rotated[0].xyz))],
    )
    assert len(dropped[0].xyz) != len(sparse[0].xyz) or not np.array_equal(
        dropped[0].xyz, sparse[0].xyz
    )
    assert len(occluded[0].xyz) < len(sparse[0].xyz)
    assert not np.allclose(
        sparse[0].xyz[: min(len(sparse[0].xyz), len(noisy[0].xyz))],
        noisy[0].xyz[: min(len(sparse[0].xyz), len(noisy[0].xyz))],
    )


def test_directional_pca_evidence_is_invariant_and_independent() -> None:
    left, right, v2, v1 = _fixture("uniform_crown")
    base = associations._evaluate_pair(left, right, v2, v1, {}, {}, VegetationAssociationConfig())
    reversed_axis = associations._evaluate_pair(
        left,
        replace(right, direction=-right.direction),
        v2,
        v1,
        {},
        {},
        VegetationAssociationConfig(),
    )
    rotated_left, rotated_right, rotated_v2, rotated_v1 = _fixture("uniform_crown", rotation=70)
    rotated = associations._evaluate_pair(
        rotated_left, rotated_right, rotated_v2, rotated_v1, {}, {}, VegetationAssociationConfig()
    )
    scaled_left, scaled_right, scaled_v2, scaled_v1 = _fixture("uniform_crown", scale=2.0)
    scaled = associations._evaluate_pair(
        scaled_left, scaled_right, scaled_v2, scaled_v1, {}, {}, VegetationAssociationConfig()
    )
    unstable_left, unstable_right, unstable_v2, unstable_v1 = _fixture("compact_dense")
    unstable = associations._evaluate_pair(
        unstable_left,
        unstable_right,
        unstable_v2,
        unstable_v1,
        {},
        {},
        VegetationAssociationConfig(),
    )
    narrow = associations._evaluate_pair(
        left, right, v2, v1, {}, {}, VegetationAssociationConfig(directional_width=0.30)
    )
    wide = associations._evaluate_pair(
        left, right, v2, v1, {}, {}, VegetationAssociationConfig(directional_width=0.80)
    )
    assert base["positive"][2] > 0.5
    assert base["positive"][2] == pytest.approx(reversed_axis["positive"][2])
    assert base["positive"][2] == pytest.approx(rotated["positive"][2], abs=0.08)
    assert base["positive"][2] == pytest.approx(scaled["positive"][2], abs=0.08)
    assert unstable["positive"][2] < base["positive"][2]
    assert narrow["positive"][2] != pytest.approx(wide["positive"][2])
    vertical_left, vertical_right, vertical_v2, vertical_v1 = _fixture("vertical_overlap")
    vertical = associations._evaluate_pair(
        vertical_left,
        vertical_right,
        vertical_v2,
        vertical_v1,
        {},
        {},
        VegetationAssociationConfig(),
    )
    assert vertical["blocking"][1] >= VegetationAssociationConfig().decisive_blocker_threshold
    assert vertical["positive"][2] != vertical["blocking"][1]


def test_profile_width_and_discrete_production_consumers() -> None:
    left, right, v2, v1 = _fixture("uniform_crown")
    profile_pair = replace(right, profile=0.62)
    narrow = associations._evaluate_pair(
        left, profile_pair, v2, v1, {}, {}, VegetationAssociationConfig(profile_width=0.25)
    )
    wide = associations._evaluate_pair(
        left, profile_pair, v2, v1, {}, {}, VegetationAssociationConfig(profile_width=0.80)
    )
    assert narrow["positive"][6] != pytest.approx(wide["positive"][6])
    summaries = {}
    for cid, offset in enumerate((0.0, 0.08, 0.16, 0.24)):
        a, _b, _v2, _v1 = _fixture("uniform_crown")
        summaries[cid] = replace(
            a,
            cluster_id=cid,
            center=a.center + [offset, 0, 0],
            lower=a.lower + [offset, 0, 0],
            upper=a.upper + [offset, 0, 0],
        )
    low_cap = associations._candidate_pairs(
        summaries, VegetationAssociationConfig(per_cluster_candidate_cap=1)
    )
    high_cap = associations._candidate_pairs(
        summaries, VegetationAssociationConfig(per_cluster_candidate_cap=3)
    )
    assert len(low_cap) < len(high_cap) and low_cap == sorted(low_cap)
    boundary_left, boundary_right, _v2, _v1 = _fixture("uniform_crown", density="dense")
    _distance, _vertical, _anisotropic, few = associations._boundary(
        boundary_left, boundary_right, 0.5, 2
    )
    _distance, _vertical, _anisotropic, many = associations._boundary(
        boundary_left, boundary_right, 0.5, 12
    )
    assert len(few) == 2 and len(many) > len(few)
    low_xyz = np.vstack((v2["xyz_local"], np.linspace(left.center, right.center, 5)[1:-1])).astype(
        np.float32
    )
    low_cluster = np.concatenate((v2["cluster_id_by_point"], np.full(3, -1, np.int32)))
    low_levels = np.concatenate(
        (v1["vegetation_level"], np.full(3, VegetationCandidateLevel.LOW, np.uint8))
    )
    low_one = associations._evaluate_pair(
        left,
        right,
        {"xyz_local": low_xyz, "cluster_id_by_point": low_cluster},
        {"vegetation_level": low_levels},
        {},
        {},
        VegetationAssociationConfig(low_support_requirement=1),
    )
    low_many = associations._evaluate_pair(
        left,
        right,
        {"xyz_local": low_xyz, "cluster_id_by_point": low_cluster},
        {"vegetation_level": low_levels},
        {},
        {},
        VegetationAssociationConfig(low_support_requirement=8),
    )
    assert low_one["positive"][7] > low_many["positive"][7]
    assert low_one["status"] != int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION)
    qualifying = associations._evaluate_pair(
        left,
        right,
        v2,
        v1,
        {},
        {},
        VegetationAssociationConfig(required_independent_positive_channels=3),
    )
    strict = associations._evaluate_pair(
        left,
        right,
        v2,
        v1,
        {},
        {},
        VegetationAssociationConfig(required_independent_positive_channels=10),
    )
    assert qualifying["status"] == int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION)
    assert strict["status"] != int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION)


def _smoke_cases(family: str) -> list[tuple[object, ...]]:
    values = []
    for index, (scale, density, rotation) in enumerate(
        product((0.5, 1.0, 2.0), ("sparse", "nominal", "dense"), (0, 30, 70))
    ):
        for slot in range(4):
            occlusion, noise = _OCCLUSION_NOISE[(4 * index + slot) % len(_OCCLUSION_NOISE)]
            hag = ("reliable", "uncertain", "unavailable")[(index + slot) % 3]
            seed = (index + 2 * slot) % 4
            values.append((family, scale, density, rotation, hag, seed, occlusion, noise))
    return values


def test_exact_1296_smoke_coverage_has_all_occlusion_noise_pairs() -> None:
    all_cases = [case for family in _FAMILIES for case in _smoke_cases(family)]
    assert len(all_cases) == 1296
    for family in _FAMILIES:
        cases = _smoke_cases(family)
        assert len(cases) == 108
        assert {case[1] for case in cases} == {0.5, 1.0, 2.0}
        assert {case[2] for case in cases} == {"sparse", "nominal", "dense"}
        assert {case[3] for case in cases} == {0, 30, 70}
        assert {case[4] for case in cases} == {"reliable", "uncertain", "unavailable"}
        assert {case[5] for case in cases} == {0, 1, 2, 3}
        assert {case[6:] for case in cases} == set(_OCCLUSION_NOISE)
        assert all(sum(case[6:] == pair for case in cases) == 18 for pair in _OCCLUSION_NOISE)


def test_full_23328_factor_sensitive_matrix_and_safety() -> None:
    statuses = {family: [] for family in _FAMILIES}
    unsafe = {"touching_crowns", "crown_wall", "crown_pole", "vertical_overlap", "low_only_bridge"}
    cases = 0
    for factors in product(
        _FAMILIES,
        (0.5, 1.0, 2.0),
        ("sparse", "nominal", "dense"),
        (0, 30, 70),
        ("reliable", "uncertain", "unavailable"),
        range(4),
        ("none", "moderate", "strong"),
        (0.0, 0.006),
    ):
        family, scale, density, rotation, hag, seed, occlusion, noise = factors
        result = _evaluate(
            family,
            scale=scale,
            density=density,
            rotation=rotation,
            hag=hag,
            dropout_seed=seed,
            occlusion=occlusion,
            noise=noise,
        )
        statuses[family].append(result["status"])
        if family in unsafe:
            assert result["status"] != int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION)
        cases += 1
    assert cases == 23328
    assert any(
        int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION) in values
        for values in statuses.values()
    )
    assert all(
        int(VegetationAssociationStatus.BLOCKED_BY_CONFLICT) in statuses[name] for name in unsafe
    )


def test_v2a_sensitivity_preserves_unsafe_nonstrong() -> None:
    base = VegetationAssociationConfig()
    settings = [base]
    for name in (
        "search_multiplier",
        "minimum_search_radius_m",
        "maximum_search_radius_m",
        "scale_ratio_width",
        "density_ratio_width",
        "directional_width",
        "profile_width",
        "spatial_possible_threshold",
        "possible_confidence_threshold",
        "strong_confidence_threshold",
        "decisive_blocker_threshold",
    ):
        for factor in (0.8, 0.9, 1.1, 1.2):
            value = getattr(base, name) * factor
            if "threshold" in name:
                value = min(value, 0.99)
            if name == "minimum_search_radius_m":
                value = min(value, base.maximum_search_radius_m)
            if name == "maximum_search_radius_m":
                value = max(value, base.minimum_search_radius_m)
            settings.append(replace(base, **{name: value}))
    for name in (
        "per_cluster_candidate_cap",
        "boundary_sample_cap",
        "required_independent_positive_channels",
        "low_support_requirement",
    ):
        for value in sorted({max(1, getattr(base, name) - 1), getattr(base, name) + 1}):
            settings.append(replace(base, **{name: value}))
    assert len(settings) == 53
    for cfg in settings:
        for family in (
            "touching_crowns",
            "crown_wall",
            "crown_pole",
            "vertical_overlap",
            "low_only_bridge",
        ):
            left, right, v2, v1 = _fixture(family)
            result = associations._evaluate_pair(left, right, v2, v1, {}, {}, cfg)
            assert result["status"] != int(VegetationAssociationStatus.STRONG_REVIEW_ASSOCIATION)
        summaries = {0: _fixture("uniform_crown")[0], 1: _fixture("density_transition")[0]}
        associations._candidate_pairs(summaries, cfg)


def test_real_v2_artifacts_remain_byte_identical_after_v2a_read_only_run() -> None:
    root = Path("outputs")
    v2_npz = root / "vegetation_instances" / "scene_simple_vegetation_instances.npz"
    v2_json = root / "vegetation_instances" / "scene_simple_vegetation_instances.json"
    if not v2_npz.exists() or not v2_json.exists():
        pytest.skip("verified V2 artifacts are not present in this checkout")
    before = (
        hashlib.sha256(v2_npz.read_bytes()).hexdigest(),
        hashlib.sha256(v2_json.read_bytes()).hexdigest(),
    )
    result = associations.load_vegetation_associations(
        root / "vegetation_associations" / "scene_simple_vegetation_associations.json"
    )[0]
    after = (
        hashlib.sha256(v2_npz.read_bytes()).hexdigest(),
        hashlib.sha256(v2_json.read_bytes()).hexdigest(),
    )
    assert result.schema_version == "vegetation-associations-v1" and after == before
