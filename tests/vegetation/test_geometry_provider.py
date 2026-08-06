"""FRV-A2 P-GEOMETRY runtime provider tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from unittest import mock

import core.contracts as contracts
import core.contracts.frv as frv
import core.contracts.frv_geometry as geometry
import laspy
import numpy as np
import pytest
from algorithms.io.source_provider import load_frv_source
from algorithms.vegetation import load_frv_geometry
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2ProviderRole,
)
from core.contracts.frv_geometry import (
    FRV_GEOMETRY_PROVIDER_KEY,
    FrvGeometryEvidenceFlag,
    FrvGeometryEvidenceState,
    FrvGeometryLoadError,
    FrvGeometryProviderConfig,
    frv_geometry_point_content_digest,
    to_frv_a2_geometry_provider_binding_evidence,
    validate_finalized_frv_geometry_provider_result,
)
from core.contracts.frv_source import (
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourceProviderReference,
    FrvSourceProviderResult,
)

from tests.contracts.test_frv_geometry_contracts import _config as _geometry_config
from tests.io.test_source_provider import (
    _configuration,
    _provider_config,
    _source_identity,
)

_VOLUMES = [(4.0 / 3.0) * math.pi * radius**3 for radius in (0.30, 0.75, 1.50)]


def _write_geometry_las(path: Path, points) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    arr = np.asarray(points, dtype=np.float64)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.classification = np.zeros(len(arr), dtype=np.uint8)
    las.write(path)


def _source_reference(path: Path, config, configuration) -> FrvSourceProviderReference:
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    with laspy.open(path) as reader:
        header = reader.header
    count = int(header.point_count)
    mins = tuple(float(value) for value in header.mins)
    maxs = tuple(float(value) for value in header.maxs)
    identity = _source_identity(
        str(path),
        file_sha,
        count,
        ([mins[0], mins[1], mins[2]], [maxs[0], maxs[1], maxs[2]]),
    )
    component = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=file_sha,
        size_bytes=file_size,
    )
    artifact_id = (
        f"frv-a2-ref-v1|source_scan|source_scan|"
        f"source_file,source_file,{file_sha},{file_size}"
    )
    return FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        artifact_reference_id=artifact_id,
        artifact_components=(component,),
        runtime_access_path=path,
        source_identity=identity,
        canonical_point_content_spec=None,
        configuration_digest=configuration.configuration_digest,
        provider_config_digest=config.provider_config_digest,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )


def _configuration_for_geometry(geometry_config):
    base = _configuration().to_digest_dict()
    base["geometry_provider_interface_version"] = geometry_config.interface_version
    base["geometry_provider_config_digest"] = geometry_config.provider_config_digest
    base["configuration_digest"] = frv.frv_digest(base)
    return frv.FrvConfiguration.from_dict(base)


def _run_source(path: Path, *, chunk_size: int = 3, geometry_config=None):
    if geometry_config is None:
        geometry_config = _geometry_config()
    source_config = _provider_config()
    configuration = _configuration_for_geometry(geometry_config)
    reference = _source_reference(path, source_config, configuration)
    session = load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=source_config,
        chunk_size=chunk_size,
    )
    with session:
        list(session.iter_chunks())
        source_result = session.finalize()
    return reference, configuration, source_result


def _run_geometry(
    path: Path,
    *,
    source_chunk_size: int = 3,
    scratch_dir: Path | None = None,
    geometry_config=None,
):
    if geometry_config is None:
        geometry_config = _geometry_config()
    reference, configuration, source_result = _run_source(
        path, chunk_size=source_chunk_size, geometry_config=geometry_config
    )
    scratch = scratch_dir if scratch_dir is not None else path.parent
    session = load_frv_geometry(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=geometry_config,
        scratch_dir=scratch,
        source_chunk_size=source_chunk_size,
    )
    return session, source_result


def _rows(chunks):
    return {
        int(chunk.original_index[r]): (chunk, r)
        for chunk in chunks
        for r in range(chunk.point_count)
    }


def _source_xyz(path: Path, *, chunk_size: int = 1000000):
    """Read the exact xyz the provider sees (LAS round-trip), for the oracle."""
    source_config = _provider_config()
    configuration = _configuration_for_geometry(_geometry_config())
    reference = _source_reference(path, source_config, configuration)
    with load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=source_config,
        chunk_size=chunk_size,
    ) as session:
        chunks = list(session.iter_chunks())
        session.finalize()
    return np.concatenate([chunk.xyz for chunk in chunks], axis=0)


def _grid_points(
    extent: int,
    spacing: float,
    *,
    base: tuple[float, float, float] = (10.0, 20.0, 5.0),
):
    points = []
    for a in range(extent):
        for b in range(extent):
            for c in range(extent):
                points.append(
                    (base[0] + a * spacing, base[1] + b * spacing, base[2] + c * spacing)
                )
    return points


# ---------------------------------------------------------------------------
# Independent full-sort oracle (mirrors the frozen algorithm, no heaps)
# ---------------------------------------------------------------------------


def _sign_canon(vector):
    values = [float(vector[0]), float(vector[1]), float(vector[2])]
    magnitudes = [abs(value) for value in values]
    axis = 0
    for index in (1, 2):
        if magnitudes[index] > magnitudes[axis]:
            axis = index
    if values[axis] < 0.0:
        values = [-value for value in values]
    return tuple(values)


def _oracle(points, config):
    """Compute the frozen per-point geometry features with full sorting.

    Returns a dict keyed by original_index with the exact plan semantics.
    """
    n = len(points)
    result: dict[int, dict] = {}
    for i in range(n):
        x, y, z = points[i]
        all_neighbors = []
        for j in range(n):
            if j == i:
                continue
            xj, yj, zj = points[j]
            dx = xj - x
            dy = yj - y
            dz = zj - z
            d2 = dx * dx + dy * dy + dz * dz
            all_neighbors.append((d2, j, xj, yj, zj))
        scales = []
        medium_direction = (0.0, 0.0, 0.0)
        for scale in range(3):
            radius = config.radii_m[scale]
            candidates = [t for t in all_neighbors if t[0] <= radius * radius]
            raw = len(candidates)
            candidates.sort(key=lambda t: (t[0], t[1]))
            taken = candidates[: config.max_neighbors_per_scale]
            count = len(taken)
            density = raw / _VOLUMES[scale]
            if count == 0:
                spacing = 0.0
            else:
                distances = np.asarray([math.sqrt(t[0]) for t in taken], dtype=np.float64)
                spacing = float(np.median(distances))
            if count >= config.min_neighbor_counts[scale]:
                samples = [(i, x, y, z)] + [(t[1], t[2], t[3], t[4]) for t in taken]
                samples.sort(key=lambda t: t[0])
                m = len(samples)
                mean = [sum(t[k] for t in samples) / m for k in (1, 2, 3)]
                covariance = [[0.0] * 3 for _ in range(3)]
                for t in samples:
                    d = (t[1] - mean[0], t[2] - mean[1], t[3] - mean[2])
                    for a in range(3):
                        for b in range(3):
                            covariance[a][b] += d[a] * d[b]
                matrix = np.asarray(covariance, dtype=np.float64) / m
                eigen, vectors = np.linalg.eigh(matrix)
                eigen = [0.0 if -1e-12 <= float(k) < 0 else float(k) for k in eigen]
                floor = config.eigenvalue_floor_m2
                if eigen[2] <= floor:
                    scales.append(
                        dict(raw=raw, count=count, density=density, spacing=spacing,
                             eigen=tuple(eigen), direction=(0.0, 0.0, 0.0),
                             normal=(0.0, 0.0, 0.0), linearity=0.0, planarity=0.0,
                             scattering=0.0, verticality=0.0, surface_normal_z=0.0)
                    )
                    continue
                linearity = (eigen[2] - eigen[1]) / eigen[2]
                planarity = (eigen[1] - eigen[0]) / eigen[2]
                scattering = eigen[0] / eigen[2]
                if eigen[2] - eigen[1] > floor:
                    direction = _sign_canon(vectors[:, 2])
                    verticality = 1.0 - abs(direction[2])
                else:
                    direction = (0.0, 0.0, 0.0)
                    verticality = 1.0
                if eigen[1] - eigen[0] > floor:
                    normal = _sign_canon(vectors[:, 0])
                else:
                    normal = (0.0, 0.0, 0.0)
                scales.append(
                    dict(raw=raw, count=count, density=density, spacing=spacing,
                         eigen=tuple(eigen), direction=direction, normal=normal,
                         linearity=linearity, planarity=planarity, scattering=scattering,
                         verticality=verticality, surface_normal_z=abs(normal[2]))
                )
                if scale == 1:
                    medium_direction = direction
            else:
                scales.append(
                    dict(raw=raw, count=count, density=density, spacing=spacing,
                         eigen=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 0.0),
                         normal=(0.0, 0.0, 0.0), linearity=0.0, planarity=0.0,
                         scattering=0.0, verticality=0.0, surface_normal_z=0.0)
                )

        # vertical cylinder
        span_min = z
        span_max = z
        above = 0
        below = 0
        slice_bitmap = 0
        for d2, _j, xj, yj, zj in all_neighbors:
            dx = xj - x
            dy = yj - y
            dz = zj - z
            radial_sq = d2 - dz * dz
            if (
                radial_sq <= config.vertical_cylinder_radius_m**2
                and abs(dz) <= config.vertical_half_window_m
            ):
                if dz > 0.05:
                    above += 1
                elif dz < -0.05:
                    below += 1
                slice_id = math.floor(
                    (dz + config.vertical_half_window_m) / config.vertical_slice_height_m
                )
                if slice_id > 11:
                    slice_id = 11
                slice_bitmap |= 1 << slice_id
                span_min = min(span_min, zj)
                span_max = max(span_max, zj)
        vertical_span = span_max - span_min
        occupied_slices = bin(slice_bitmap).count("1")

        medium_at = scales[1]["count"] >= config.min_neighbor_counts[1]
        large_at = scales[2]["count"] >= config.min_neighbor_counts[2]
        if not (medium_at and large_at):
            state = int(FrvGeometryEvidenceState.INSUFFICIENT)
            reason = int(geometry.FrvGeometryReasonFlag.INSUFFICIENT_NEIGHBORS)
        elif (
            scales[1]["eigen"][2] <= config.eigenvalue_floor_m2
            or scales[2]["eigen"][2] <= config.eigenvalue_floor_m2
        ):
            state = int(FrvGeometryEvidenceState.DEGENERATE)
            reason = int(geometry.FrvGeometryReasonFlag.DEGENERATE_COVARIANCE)
        else:
            state = int(FrvGeometryEvidenceState.VALID)
            reason = int(geometry.FrvGeometryReasonFlag.NONE)

        flags = 0
        density_supported = scales[2]["density"] >= config.canopy_density_min_per_m3
        vertical_continuity = (
            vertical_span >= config.woody_vertical_span_min_m
            and occupied_slices >= config.min_vertical_occupied_slices
            and above >= 1
            and below >= 1
        )
        if state == int(FrvGeometryEvidenceState.VALID):
            flags |= int(FrvGeometryEvidenceFlag.NEIGHBORHOOD_SUFFICIENT)
        if density_supported:
            flags |= int(FrvGeometryEvidenceFlag.DENSITY_SUPPORTED)
        if state == int(FrvGeometryEvidenceState.VALID) and density_supported:
            if (
                scales[2]["scattering"] >= config.canopy_scattering_min
                and scales[2]["linearity"] <= config.canopy_linearity_max
            ):
                flags |= int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE)
        if vertical_continuity:
            flags |= int(FrvGeometryEvidenceFlag.VERTICAL_CONTINUITY)
        if state == int(FrvGeometryEvidenceState.VALID) and vertical_continuity:
            if (
                scales[1]["linearity"] >= config.woody_linearity_min
                and scales[1]["verticality"] <= config.woody_verticality_max
            ):
                flags |= int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE)
        if state == int(FrvGeometryEvidenceState.VALID):
            if (
                scales[1]["planarity"] >= config.ground_planarity_min
                and scales[1]["surface_normal_z"] >= config.ground_normal_z_min
            ):
                flags |= int(FrvGeometryEvidenceFlag.GROUND_SURFACE_COMPATIBLE)

        # support candidates (top-16 within 0.30 m by (d2, index))
        support_in_radius = [
            (t[0], t[1]) for t in all_neighbors if t[0] <= config.support_neighbor_radius_m**2
        ]
        support_raw = len(support_in_radius)
        support = sorted(support_in_radius)[: config.support_neighbor_cap]

        result[i] = dict(
            scales=scales,
            vertical_span=vertical_span,
            occupied_slices=occupied_slices,
            above=above,
            below=below,
            state=state,
            reason=reason,
            flags=flags,
            support_candidates=support,
            support_raw=support_raw,
            support_valid=state == int(FrvGeometryEvidenceState.VALID),
            support_canopy=bool(flags & int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE)),
            medium_direction=medium_direction,
        )
    # resolve mutual support edges
    support_sets = {
        i: frozenset(j for _d2, j in data["support_candidates"])
        for i, data in result.items()
    }
    for i, data in result.items():
        edges = []
        for d2, j in data["support_candidates"]:
            if i not in support_sets[j]:
                continue
            if not (data["support_valid"] and result[j]["support_valid"]):
                continue
            dot = sum(
                a * b
                for a, b in zip(
                    data["medium_direction"],
                    result[j]["medium_direction"],
                    strict=True,
                )
            )
            if abs(dot) < config.direction_cosine_min and not (
                data["support_canopy"] and result[j]["support_canopy"]
            ):
                continue
            edges.append((d2, j))
        data["support_edges"] = edges
    return result


# ---------------------------------------------------------------------------
# Two passes, normal flow, one-to-one, digest, invariance
# ---------------------------------------------------------------------------


def test_exactly_two_source_sessions(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    reference, configuration, source_result = _run_source(path, chunk_size=5)
    provider_config = _geometry_config()
    with mock.patch(
        "algorithms.vegetation.geometry_provider.load_frv_source",
        wraps=load_frv_source,
    ) as mocked:
        session = load_frv_geometry(
            reference,
            source_result,
            frv_configuration=configuration,
            provider_config=provider_config,
            scratch_dir=tmp_path,
            source_chunk_size=5,
        )
        with session:
            list(session.iter_chunks())
            session.finalize()
        assert mocked.call_count == 2  # Pass G1 and Pass G2


def test_normal_flow_chunks_and_result(tmp_path: Path) -> None:
    points = _grid_points(3, 0.1) + [(100.0, 100.0, 50.0), (101.0, 101.0, 51.0)]
    path = tmp_path / "s.las"
    _write_geometry_las(path, points)
    session, _ = _run_geometry(path, source_chunk_size=5)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    assert [c.point_count for c in chunks] == [5, 5, 5, 5, 5, 4]
    assert [c.is_last for c in chunks] == [False, False, False, False, False, True]
    all_index = np.concatenate([c.original_index for c in chunks])
    assert all_index.tolist() == list(range(len(points)))
    validate_finalized_frv_geometry_provider_result(result)
    evidence = to_frv_a2_geometry_provider_binding_evidence(result)
    contracts.validate_frv_a2_provider_binding_evidence(evidence)
    assert evidence.provider_key == FRV_GEOMETRY_PROVIDER_KEY
    assert evidence.provider_role is FrvA2ProviderRole.GEOMETRY
    assert evidence.provider_readiness.value == "validated"
    assert result.proof.source_point_count == len(points)
    assert (
        result.proof.valid_count + result.proof.insufficient_count + result.proof.degenerate_count
        == len(points)
    )
    assert result.proof.neighborhood_sufficient_count == result.proof.valid_count


def test_chunk_one_to_one_and_geometry_values(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, source_result = _run_geometry(path, source_chunk_size=7)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    assert result.proof.source_point_count == 27
    # each geometry chunk mirrors its Pass G2 source chunk scalar
    assert result.proof.valid_count > 0
    assert result.proof.insufficient_count >= 0
    for chunk in chunks:
        assert np.isfinite(chunk.density_per_m3).all()
        assert np.isfinite(chunk.eigenvalues_m2).all()
        assert np.isfinite(chunk.principal_direction).all()
        assert (chunk.local_spacing_m >= 0.0).all()
        assert (chunk.support_neighbor_original_index[:, 15] == -1).all()
        geometry.validate_frv_geometry_chunk(chunk, provider_config=_geometry_config())


def test_point_content_digest_matches_oracle(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=5)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    oracle = frv_geometry_point_content_digest(
        source_content_identity_digest=result.proof.source_content_identity_digest,
        source_identity_record_digest=result.proof.source_identity_record_digest,
        provider_config_digest=result.proof.provider_config_digest,
        source_point_count=27,
        chunks=chunks,
    )
    assert oracle == result.proof.geometry_point_content_digest


def test_chunk_size_invariance(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    results = []
    for chunk_size in (2, 5):
        session, _ = _run_geometry(path, source_chunk_size=chunk_size)
        with session:
            list(session.iter_chunks())
            results.append(session.finalize())
    assert results[0].proof == results[1].proof


# ---------------------------------------------------------------------------
# Independent oracle agreement
# ---------------------------------------------------------------------------


def test_features_match_independent_oracle(tmp_path: Path) -> None:
    points = _grid_points(4, 0.1)
    points.extend([(100.0, 100.0, 50.0), (101.0, 101.0, 51.0)])
    path = tmp_path / "s.las"
    _write_geometry_las(path, points)
    config = _geometry_config()
    session, _ = _run_geometry(path, source_chunk_size=8)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    provider_points = [tuple(float(v) for v in row) for row in _source_xyz(path)]
    oracle = _oracle(provider_points, config)
    rows = _rows(chunks)
    assert set(rows) == set(range(len(points)))
    state_counts = {
        int(FrvGeometryEvidenceState.VALID): 0,
        int(FrvGeometryEvidenceState.INSUFFICIENT): 0,
        int(FrvGeometryEvidenceState.DEGENERATE): 0,
    }
    for index in range(len(points)):
        chunk, row = rows[index]
        expected = oracle[index]
        for scale in range(3):
            assert int(chunk.raw_neighbor_count[row, scale]) == expected["scales"][scale]["raw"]
            assert int(chunk.neighbor_count[row, scale]) == expected["scales"][scale]["count"]
            assert float(chunk.density_per_m3[row, scale]) == pytest.approx(
                expected["scales"][scale]["density"], rel=1e-4
            )
            assert float(chunk.local_spacing_m[row, scale]) == pytest.approx(
                expected["scales"][scale]["spacing"], rel=1e-4
            )
            for axis in range(3):
                assert float(chunk.eigenvalues_m2[row, scale, axis]) == pytest.approx(
                    expected["scales"][scale]["eigen"][axis], rel=1e-6
                )
                assert float(chunk.principal_direction[row, scale, axis]) == pytest.approx(
                    expected["scales"][scale]["direction"][axis], abs=1e-4
                )
                assert float(chunk.surface_normal[row, scale, axis]) == pytest.approx(
                    expected["scales"][scale]["normal"][axis], abs=1e-4
                )
            assert float(chunk.linearity[row, scale]) == pytest.approx(
                expected["scales"][scale]["linearity"], abs=1e-4
            )
            assert float(chunk.planarity[row, scale]) == pytest.approx(
                expected["scales"][scale]["planarity"], abs=1e-4
            )
            assert float(chunk.scattering[row, scale]) == pytest.approx(
                expected["scales"][scale]["scattering"], abs=1e-4
            )
            assert float(chunk.verticality[row, scale]) == pytest.approx(
                expected["scales"][scale]["verticality"], abs=1e-4
            )
        assert float(chunk.vertical_span_m[row]) == pytest.approx(
            expected["vertical_span"], abs=1e-4
        )
        assert int(chunk.vertical_occupied_slices[row]) == expected["occupied_slices"]
        assert int(chunk.above_count[row]) == expected["above"]
        assert int(chunk.below_count[row]) == expected["below"]
        assert int(chunk.geometry_evidence_state[row]) == expected["state"]
        assert int(chunk.reason_flags[row]) == expected["reason"]
        assert int(chunk.geometry_evidence_flags[row]) == expected["flags"]
        # mutual support edges must match the oracle exactly
        expected_edges = expected["support_edges"]
        assert int(chunk.support_neighbor_count[row]) == len(expected_edges)
        stored = chunk.support_neighbor_original_index[row][: len(expected_edges)].tolist()
        assert stored == [neighbor for _d2, neighbor in expected_edges]
        state_counts[expected["state"]] += 1
    assert result.proof.valid_count == state_counts[int(FrvGeometryEvidenceState.VALID)]
    assert result.proof.insufficient_count == state_counts[
        int(FrvGeometryEvidenceState.INSUFFICIENT)
    ]
    assert result.proof.degenerate_count == state_counts[
        int(FrvGeometryEvidenceState.DEGENERATE)
    ]


def test_evidence_flags_canopy_woody_ground(tmp_path: Path) -> None:
    # canopy: dense spherical cloud
    canopy_points = []
    radius = 0.5
    step = 0.08
    for dx in np.arange(-radius, radius + step, step):
        for dy in np.arange(-radius, radius + step, step):
            for dz in np.arange(-radius, radius + step, step):
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    canopy_points.append((50.0 + dx, 50.0 + dy, 50.0 + dz))
    path = tmp_path / "canopy.las"
    _write_geometry_las(path, canopy_points)
    config = _geometry_config()
    session, _ = _run_geometry(path, source_chunk_size=60)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    oracle = _oracle(canopy_points, config)
    rows = _rows(chunks)
    canopy_flags = 0
    for index in range(len(canopy_points)):
        chunk, row = rows[index]
        if oracle[index]["state"] == int(FrvGeometryEvidenceState.VALID):
            assert int(chunk.geometry_evidence_flags[row]) == oracle[index]["flags"]
            if oracle[index]["flags"] & int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE):
                canopy_flags += 1
    assert canopy_flags > 0
    assert result.proof.canopy_branching_compatible_count == canopy_flags

    # woody: a tall vertical column of points
    woody_points = [(60.0, 60.0, z) for z in np.arange(0.0, 2.0, 0.05)]
    path = tmp_path / "woody.las"
    _write_geometry_las(path, woody_points)
    session, _ = _run_geometry(path, source_chunk_size=50)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    assert result.proof.woody_column_compatible_count > 0

    # ground: a horizontal planar patch
    ground_points = []
    for dx in np.arange(0.0, 1.0, 0.1):
        for dy in np.arange(0.0, 1.0, 0.1):
            ground_points.append((70.0 + dx, 70.0 + dy, 3.0))
    path = tmp_path / "ground.las"
    _write_geometry_las(path, ground_points)
    session, _ = _run_geometry(path, source_chunk_size=50)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    assert result.proof.ground_surface_compatible_count > 0


def test_support_edges_symmetric_and_capped(tmp_path: Path) -> None:
    points = _grid_points(5, 0.05)  # dense: many neighbors within 0.30 m
    path = tmp_path / "s.las"
    _write_geometry_las(path, points)
    config = _geometry_config()
    session, _ = _run_geometry(path, source_chunk_size=15)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    by_index = {
        int(chunk.original_index[r]): (chunk, r)
        for chunk in chunks
        for r in range(chunk.point_count)
    }
    for index in range(len(points)):
        chunk, row = by_index[index]
        count = int(chunk.support_neighbor_count[row])
        assert count <= config.support_neighbor_cap
        support = chunk.support_neighbor_original_index[row][:count].tolist()
        for neighbor in support:
            nchunk, nrow = by_index[neighbor]
            ncount = int(nchunk.support_neighbor_count[nrow])
            assert index in nchunk.support_neighbor_original_index[nrow][:ncount].tolist()
    # dense points reach the support cap and set the truncation quality flag
    provider_points = [tuple(float(v) for v in row) for row in _source_xyz(path)]
    oracle = _oracle(provider_points, config)
    assert any(
        data["support_raw"] > config.support_neighbor_cap
        for data in oracle.values()
    )
    assert result.proof.support_neighbor_cap_reached_count > 0


def test_support_edges_cross_chunk_halo(tmp_path: Path) -> None:
    # Two clusters split across a source chunk boundary so the mutual support
    # edges must be resolved across chunks and across spool cells.
    cluster_a = _grid_points(3, 0.1, base=(10.0, 10.0, 10.0))
    cluster_b = _grid_points(3, 0.1, base=(10.4, 10.0, 10.0))
    points = cluster_a + cluster_b
    path = tmp_path / "s.las"
    _write_geometry_las(path, points)
    # chunk size 9 splits the inter-cluster gap: cluster A ends at index 26
    # (chunk 2) and cluster B starts at index 27 (chunk 3).
    session, _ = _run_geometry(path, source_chunk_size=9)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    assert len(chunks) == 6  # 6 x 9
    by_index = {
        int(chunk.original_index[r]): (chunk, r)
        for chunk in chunks
        for r in range(chunk.point_count)
    }
    cross_chunk_edges = 0
    for chunk in chunks:
        for row in range(chunk.point_count):
            count = int(chunk.support_neighbor_count[row])
            for neighbor in chunk.support_neighbor_original_index[row][:count].tolist():
                # neighbor in a different source chunk
                if by_index[neighbor][0] is not chunk:
                    cross_chunk_edges += 1
    assert cross_chunk_edges > 0
    validate_finalized_frv_geometry_provider_result(result)


# ---------------------------------------------------------------------------
# Edge-case sources
# ---------------------------------------------------------------------------


def test_one_point_sparse(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, [(10.0, 20.0, 30.0)])
    session, _ = _run_geometry(path, source_chunk_size=1)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    chunk = chunks[0]
    assert int(chunk.geometry_evidence_state[0]) == int(FrvGeometryEvidenceState.INSUFFICIENT)
    assert int(chunk.reason_flags[0]) == int(geometry.FrvGeometryReasonFlag.INSUFFICIENT_NEIGHBORS)
    assert int(chunk.raw_neighbor_count[0, 0]) == 0
    assert float(chunk.local_spacing_m[0, 0]) == 0.0
    assert result.proof.valid_count == 0
    assert result.proof.insufficient_count == 1
    validate_finalized_frv_geometry_provider_result(result)


def test_large_coordinate_source(tmp_path: Path) -> None:
    points = _grid_points(3, 0.1, base=(1_000_000.0, 2_000_000.0, 3_000_000.0))
    path = tmp_path / "s.las"
    _write_geometry_las(path, points)
    config = _geometry_config()
    session, _ = _run_geometry(path, source_chunk_size=5)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    provider_points = [tuple(float(v) for v in row) for row in _source_xyz(path)]
    oracle = _oracle(provider_points, config)
    rows = _rows(chunks)
    for index in range(len(points)):
        chunk, row = rows[index]
        expected = oracle[index]
        assert int(chunk.geometry_evidence_flags[row]) == expected["flags"]
        for scale in range(3):
            assert int(chunk.neighbor_count[row, scale]) == expected["scales"][scale]["count"]
    assert result.proof.valid_count == oracle_valid_count(oracle)


def oracle_valid_count(oracle):
    return sum(
        1
        for data in oracle.values()
        if data["state"] == int(FrvGeometryEvidenceState.VALID)
    )


def test_dense_cell_limit_failure(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(2, 0.1))
    config = _geometry_config(max_index_cell_points=2)
    session, _ = _run_geometry(path, source_chunk_size=4, geometry_config=config)
    with pytest.raises(FrvGeometryLoadError) as excinfo:
        list(session.iter_chunks())
    assert "index_cell" in str(excinfo.value)
    assert "FRV_PROVIDER_INVALID" in str(excinfo.value)
    session.close()


def test_radius_boundary(tmp_path: Path) -> None:
    # Points exactly at the large-scale radius must be included (d2 <= r^2).
    points = [
        (10.0, 20.0, 5.0),
        (11.5, 20.0, 5.0),  # 1.50 m -> exactly on the large radius
        (11.6, 20.0, 5.0),  # 1.60 m -> outside every radius of p0
    ]
    path = tmp_path / "s.las"
    _write_geometry_las(path, points)
    session, _ = _run_geometry(path, source_chunk_size=3)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    rows = _rows(chunks)
    chunk, row = rows[0]
    assert int(chunk.raw_neighbor_count[row, 2]) == 1  # exactly the 1.50 m point
    chunk, row = rows[1]
    assert int(chunk.raw_neighbor_count[row, 2]) == 2  # p0 (1.50 m) and p2 (0.10 m)
    chunk, row = rows[2]
    assert int(chunk.raw_neighbor_count[row, 2]) == 1  # only the 0.10 m point
    validate_finalized_frv_geometry_provider_result(result)


# ---------------------------------------------------------------------------
# Lifecycle, failure mapping, cleanup
# ---------------------------------------------------------------------------


def test_lifecycle_premature_and_repeated(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=5)
    with pytest.raises(FrvGeometryLoadError):
        session.finalize()  # premature (never iterated)
    session.close()

    session2, _ = _run_geometry(path, source_chunk_size=5)
    iterator = session2.iter_chunks()
    next(iterator)  # partial consumption
    with pytest.raises(FrvGeometryLoadError):
        session2.finalize()
    session2.close()
    iterator.close()

    session3, _ = _run_geometry(path, source_chunk_size=5)
    with session3:
        list(session3.iter_chunks())
        session3.finalize()
        with pytest.raises(FrvGeometryLoadError):
            session3.finalize()  # repeated finalize
    # repeated iter_chunks is rejected
    session4, _ = _run_geometry(path, source_chunk_size=5)
    session4.iter_chunks()
    with pytest.raises(FrvGeometryLoadError):
        session4.iter_chunks()
    session4.close()


def test_spool_directory_removed_on_close(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=5, scratch_dir=tmp_path)
    with session:
        list(session.iter_chunks())
        session.finalize()
    assert not any(p.name.startswith("treeline-geometry-") for p in tmp_path.iterdir())


def test_active_exception_not_overridden_by_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=5, scratch_dir=tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with session:
            raise RuntimeError("boom")
    assert not any(p.name.startswith("treeline-geometry-") for p in tmp_path.iterdir())


def test_pass1_load_failure_aborts(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=5)
    with mock.patch(
        "algorithms.vegetation.geometry_provider.load_frv_source",
        side_effect=RuntimeError("load-boom"),
    ):
        with pytest.raises(FrvGeometryLoadError) as excinfo:
            session.iter_chunks()
    assert "FRV_STREAM_INCOMPLETE" in str(excinfo.value)
    assert session._state == "ABORTED"
    session.close()


def test_scratch_preflight_failure(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=5)
    with mock.patch(
        "algorithms.vegetation.geometry_provider.shutil.disk_usage",
        return_value=mock.Mock(free=1024),
    ):
        with pytest.raises(FrvGeometryLoadError) as excinfo:
            list(session.iter_chunks())
    assert "scratch_store" in str(excinfo.value)
    session.close()


def test_original_index_lost_duplicate_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(2, 0.1))

    def tamper(session, mutator):
        original = session._assemble_geometry_chunk

        def builder(data):
            chunk = original(data)
            idx = chunk.original_index.copy()
            mutator(idx)
            idx.setflags(write=False)
            return dataclasses.replace(chunk, original_index=idx)

        session._assemble_geometry_chunk = builder  # type: ignore[method-assign]

    session, _ = _run_geometry(path, source_chunk_size=4)
    tamper(session, lambda idx: idx.__setitem__(0, int(idx[0]) + 5))
    with pytest.raises(FrvGeometryLoadError) as ei:
        list(session.iter_chunks())
    assert "FRV_ORIGINAL_INDEX_INVALID" in str(ei.value)
    session.close()

    session2, _ = _run_geometry(path, source_chunk_size=4)
    tamper(session2, lambda idx: idx.__setitem__(1, int(idx[0])))
    with pytest.raises(FrvGeometryLoadError) as ei:
        list(session2.iter_chunks())
    assert "FRV_ORIGINAL_INDEX_DUPLICATE" in str(ei.value)
    session2.close()

    session3, _ = _run_geometry(path, source_chunk_size=4)
    tamper(session3, lambda idx: idx.__setitem__(0, -1))
    with pytest.raises(FrvGeometryLoadError) as ei:
        list(session3.iter_chunks())
    assert "FRV_ORIGINAL_INDEX_INVALID" in str(ei.value)
    session3.close()


def test_pre_pass_gate_errors(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(2, 0.1))
    reference, configuration, source_result = _run_source(path, chunk_size=4)
    geometry_config = _geometry_config()
    kwargs = dict(
        frv_configuration=configuration,
        provider_config=geometry_config,
        scratch_dir=tmp_path,
        source_chunk_size=4,
    )

    # forged source result (missing private capability)
    forged = object.__new__(FrvSourceProviderResult)
    object.__setattr__(forged, "proof", source_result.proof)
    object.__setattr__(forged, "source_identity", source_result.source_identity)
    object.__setattr__(forged, "_finalized_capability", object())
    with pytest.raises(FrvGeometryLoadError) as ei:
        load_frv_geometry(reference, forged, **kwargs)
    assert "source_result" in str(ei.value)
    assert "FRV_PROVIDER_INVALID" in str(ei.value)

    # wrong-type reference is a TypeError at the loader boundary
    with pytest.raises(TypeError):
        load_frv_geometry(object(), source_result, **kwargs)  # type: ignore[arg-type]

    # invalid configuration
    bad_configuration = object.__new__(frv.FrvConfiguration)
    with pytest.raises(FrvGeometryLoadError) as ei:
        load_frv_geometry(
            reference, source_result, **{**kwargs, "frv_configuration": bad_configuration}
        )
    assert "provider_config" in str(ei.value)

    # geometry config whose interface/digest do not match the configuration
    mismatched = object.__new__(FrvGeometryProviderConfig)
    for field_ in dataclasses.fields(FrvGeometryProviderConfig):
        object.__setattr__(mismatched, field_.name, getattr(geometry_config, field_.name))
    object.__setattr__(mismatched, "provider_config_digest", "b" * 64)
    with pytest.raises(FrvGeometryLoadError) as ei:
        load_frv_geometry(
            reference, source_result, **{**kwargs, "provider_config": mismatched}
        )
    assert "provider_config" in str(ei.value)

    # tampered reference (wrong configuration_digest)
    tampered = dataclasses.replace(reference, configuration_digest="b" * 64)
    with pytest.raises(FrvGeometryLoadError) as ei:
        load_frv_geometry(tampered, source_result, **kwargs)
    assert "reference" in str(ei.value)


def test_source_identity_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(2, 0.1))
    reference, configuration, source_result = _run_source(path, chunk_size=4)
    geometry_config = _geometry_config()
    session = load_frv_geometry(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=geometry_config,
        scratch_dir=tmp_path,
        source_chunk_size=4,
    )
    session._expected_source_id = "other-scene"
    with pytest.raises(FrvGeometryLoadError) as ei:
        list(session.iter_chunks())
    assert "FRV_SOURCE_IDENTITY_FAILED" in str(ei.value)
    session.close()


def test_pass1_binding_mismatch_aborts(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(2, 0.1))
    other = tmp_path / "other.las"
    _write_geometry_las(other, _grid_points(2, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=4)
    _, _, other_result = _run_source(other, chunk_size=4)

    class _FakeSourceSession:
        def __init__(self, result):
            self._result = result

        def iter_chunks(self):
            return iter(())

        def finalize(self):
            return self._result

        def close(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake = _FakeSourceSession(other_result)
    with mock.patch(
        "algorithms.vegetation.geometry_provider.load_frv_source",
        side_effect=[fake],
    ):
        with pytest.raises(FrvGeometryLoadError) as ei:
            session.iter_chunks()
    assert "FRV_ARTIFACT_DIGEST_MISMATCH" in str(ei.value)
    session.close()


def test_bnd1_upstream_triple(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    _write_geometry_las(path, _grid_points(3, 0.1))
    session, _ = _run_geometry(path, source_chunk_size=5)
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    evidence = to_frv_a2_geometry_provider_binding_evidence(result)
    expected = tuple(
        sorted(
            {
                result.proof.geometry_point_content_digest,
                result.proof.source_binding_digest,
                result.proof.source_proof_digest,
            }
        )
    )
    assert evidence.upstream_binding_digests == expected
    assert evidence.binding_digest == contracts.frv_a2_provider_binding_digest(evidence)
