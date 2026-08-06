"""FRV-A2 P-WIRE runtime provider tests (PLAN-FROZEN)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import core.contracts.frv as frv
import laspy
import numpy as np
import pytest
from algorithms.io.source_provider import load_frv_source
from algorithms.vegetation import load_frv_wire
from algorithms.vegetation import wire_provider as wp
from core.contracts.frv import FrvWireProviderState
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2ProviderRole,
)
from core.contracts.frv_source import (
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourceProviderReference,
)
from core.contracts.frv_wire import (
    FRV_WIRE_AUTHORITY_DIGEST_DOMAIN,
    FRV_WIRE_AUTHORITY_SCHEMA_VERSION,
    FRV_WIRE_PROVIDER_KEY,
    FrvWireAuthorityManifest,
    FrvWireAuthorityMode,
    FrvWireAuthorityReference,
    FrvWireLoadError,
    FrvWirePointRelationState,
    frv_wire_geometry_content_digest,
    to_frv_a2_wire_provider_binding_evidence,
    validate_finalized_frv_wire_provider_result,
)
from core.contracts.wire_models import (
    WireModelMethod,
    WireModelStatus,
    WireModelSummary,
    WireReconstructionConfig,
    WireReconstructionResult,
    WireReconstructionStatistics,
)

from tests.contracts.test_frv_wire_contracts import (
    SHA,
    _canon,
)
from tests.contracts.test_frv_wire_contracts import (
    _config as _wire_config,
)
from tests.io.test_source_provider import (
    _configuration,
    _provider_config,
    _source_identity,
)

_WIRE_SCHEMA = "wire-models-v2"
_WIRE_ALGORITHM = "w1-confirmed-wire-centreline-v3"


def _write_wire_las(path: Path, points) -> None:
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


def _configuration_for_wire(wire_config):
    base = _configuration().to_digest_dict()
    base["wire_provider_interface_version"] = wire_config.interface_version
    base["wire_provider_config_digest"] = wire_config.provider_config_digest
    base["configuration_digest"] = frv.frv_digest(base)
    return frv.FrvConfiguration.from_dict(base)


def _run_source(path: Path, *, chunk_size: int = 3, wire_config=None):
    if wire_config is None:
        wire_config = _wire_config()
    source_config = _provider_config()
    configuration = _configuration_for_wire(wire_config)
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


# ---------------------------------------------------------------------------
# W1 artifact helpers
# ---------------------------------------------------------------------------


def _w1_arrays(*, wire_offsets: tuple[float, ...] = (0.0,)) -> dict[str, np.ndarray]:
    """Build a valid W1 NPZ array inventory with one segment per wire."""
    wire_ids = list(range(len(wire_offsets)))
    curves = []
    for offset in wire_offsets:
        x = np.arange(0.0, 1.001, 0.1, dtype=np.float32)
        curve = np.column_stack(
            (x, np.full_like(x, offset), np.full_like(x, 10.0))
        ).astype(np.float32)
        curves.append(curve)
    curve_xyz = np.concatenate(curves, axis=0)
    curve_parameter = np.linspace(0.0, 1.0, len(curve_xyz), dtype=np.float32)
    curve_offsets = [0]
    for curve in curves:
        curve_offsets.append(curve_offsets[-1] + len(curve))
    segment_count = len(wire_ids)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(_WIRE_SCHEMA),
        "wire_id": np.asarray(wire_ids, dtype=np.int32),
        "source_track_id": np.asarray(wire_ids, dtype=np.int32),
        "wire_status": np.zeros(len(wire_ids), dtype=np.uint8),
        "wire_method": np.zeros(len(wire_ids), dtype=np.uint8),
        "wire_quality_flags": np.zeros(len(wire_ids), dtype=np.uint16),
        "wire_has_distance_ready_segments": np.ones(len(wire_ids), dtype=np.bool_),
        "wire_continuous_distance_ready": np.ones(len(wire_ids), dtype=np.bool_),
        "wire_segment_offsets": np.arange(len(wire_ids) + 1, dtype=np.int64),
        "segment_wire_id": np.asarray(wire_ids, dtype=np.int32),
        "segment_index": np.zeros(segment_count, dtype=np.int32),
        "segment_method": np.zeros(segment_count, dtype=np.uint8),
        "segment_quality_flags": np.zeros(segment_count, dtype=np.uint16),
        "segment_distance_ready": np.ones(segment_count, dtype=np.bool_),
        "segment_parameter_domain": np.asarray(
            [[0.0, 1.0]] * segment_count, dtype=np.float64
        ),
        "sampled_curve_xyz": curve_xyz,
        "sampled_curve_parameter": curve_parameter,
        "sampled_curve_offsets": np.asarray(curve_offsets, dtype=np.int64),
    }
    return arrays


def _w1_digest_edges(arrays: dict[str, np.ndarray], origin=(0.0, 0.0, 0.0)):
    """Return the absolute-coordinate edges exactly as the provider builds them."""
    edges = []
    for position in range(len(arrays["wire_id"])):
        seg_start = int(arrays["wire_segment_offsets"][position])
        seg_stop = int(arrays["wire_segment_offsets"][position + 1])
        for segment_position in range(seg_start, seg_stop):
            curve_start = int(arrays["sampled_curve_offsets"][segment_position])
            curve_stop = int(arrays["sampled_curve_offsets"][segment_position + 1])
            segment = int(arrays["segment_index"][segment_position])
            for edge_position in range(curve_start, curve_stop - 1):
                a = arrays["sampled_curve_xyz"][edge_position]
                b = arrays["sampled_curve_xyz"][edge_position + 1]
                edges.append(
                    (
                        int(arrays["wire_id"][position]),
                        segment,
                        edge_position - curve_start,
                        float(a[0]) + origin[0],
                        float(a[1]) + origin[1],
                        float(a[2]) + origin[2],
                        float(b[0]) + origin[0],
                        float(b[1]) + origin[1],
                        float(b[2]) + origin[2],
                    )
                )
    return edges


def _write_w1_pair(tmp_path: Path, arrays: dict[str, np.ndarray]) -> tuple[Path, Path]:
    npz_path = tmp_path / "scene-1_wire_models.npz"
    json_path = tmp_path / "scene-1_wire_models.json"
    np.savez(npz_path, **arrays)
    npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    config = WireReconstructionConfig(source_id="scene-1")
    stats = WireReconstructionStatistics(
        input_point_count=5,
        confirmed_wire_point_count=5,
        wire_count=len(arrays["wire_id"]),
        segment_count=len(arrays["segment_wire_id"]),
        gap_count=0,
        distance_ready_segment_count=int(np.count_nonzero(arrays["segment_distance_ready"])),
        elapsed_seconds=0.1,
    )
    summaries = tuple(
        WireModelSummary(
            wire_id=int(arrays["wire_id"][position]),
            source_track_id=int(arrays["source_track_id"][position]),
            status=WireModelStatus.CONFIRMED_MODEL,
            method=WireModelMethod.CUBIC_BSPLINE,
            quality_flags=0,
            wire_has_distance_ready_segments=True,
            wire_continuous_distance_ready=True,
            observed_segment_count=1,
            point_count=5,
            representative_count=5,
            observed_total_span_m=1.0,
            observed_envelope_span_m=1.0,
            total_unbridged_gap_span_m=0.0,
            observed_coverage_ratio=1.0,
            observed_total_arc_length_m=1.0,
            start_xyz=(0.0, 0.0, 10.0),
            end_xyz=(1.0, 0.0, 10.0),
            main_direction=(1.0, 0.0, 0.0),
        )
        for position in range(len(arrays["wire_id"]))
    )
    result = WireReconstructionResult(
        schema_version=_WIRE_SCHEMA,
        source_id="scene-1",
        manifest_source_id="scene-1",
        input_a4_npz_relpath="a4.npz",
        input_a4_json_relpath="a4.json",
        input_a4_npz_sha256=SHA,
        input_a4_json_sha256=SHA,
        output_npz_relpath="scene-1_wire_models.npz",
        output_json_relpath="scene-1_wire_models.json",
        output_npz_sha256=npz_sha,
        output_npz_bytes=npz_path.stat().st_size,
        config=config,
        statistics=stats,
        arrays={},
        wires=summaries,
        segments=(),
        gaps=(),
        warnings=(),
        algorithm_version=_WIRE_ALGORITHM,
        created_at="2026-01-01T00:00:00+00:00",
    )
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False), "utf-8")
    return npz_path, json_path


def _component(path: Path, key: str, kind: FrvA2ArtifactComponentKind):
    return FrvA2ArtifactComponentBinding(
        component_key=key,
        component_kind=kind,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _build_authority(
    tmp_path: Path,
    source_result,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    authority_mode: FrvWireAuthorityMode = FrvWireAuthorityMode.W1_MODEL_SET,
    geometry_digest: str | None = None,
    source_content_digest: str | None = None,
    source_id: str = "scene-1",
) -> tuple[FrvWireAuthorityReference, Path]:
    if arrays is None:
        arrays = _w1_arrays()
    npz_path, json_path = _write_w1_pair(tmp_path, arrays)
    source = source_result.source_identity
    origin = tuple(source.origin_context)
    edges = _w1_digest_edges(arrays, origin)
    if geometry_digest is None:
        geometry_digest = frv_wire_geometry_content_digest(
            source_content_identity_digest=source.source_content_identity_digest,
            source_identity_record_digest=source.source_identity_record_digest,
            origin_xyz=origin,
            edges=edges,
        )
    content_digest = source_content_digest or source.source_content_identity_digest
    w1_components = (
        _component(json_path, "manifest_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
        _component(npz_path, "payload_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
    )
    fields = {
        "schema_version": FRV_WIRE_AUTHORITY_SCHEMA_VERSION,
        "authority_mode": authority_mode,
        "source_content_identity_digest": content_digest,
        "source_identity_record_digest": source.source_identity_record_digest,
        "source_id": source_id,
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_xyz": origin,
        "transform_declaration": "identity",
        "inventory_complete": True,
        "approved_wire_ids": _approved_ids(arrays, authority_mode),
        "w1_artifact_components": (
            w1_components
            if authority_mode is FrvWireAuthorityMode.W1_MODEL_SET
            else ()
        ),
        "geometry_content_digest": geometry_digest,
        "authority_rule_id": "wire-authority-rule",
        "authority_rule_version": "rule-v1",
        "approver": "approver",
        "approval_timestamp": "2026-01-01T00:00:00Z",
    }
    payload_json = {
        **fields,
        "authority_mode": fields["authority_mode"].value,
        "approved_wire_ids": list(fields["approved_wire_ids"]),
        "w1_artifact_components": [
            component.to_dict() for component in fields["w1_artifact_components"]
        ],
        "origin_xyz": list(fields["origin_xyz"]),
    }
    approval_digest = hashlib.sha256(_canon(payload_json)).hexdigest()
    manifest_payload = {
        "digest_domain": FRV_WIRE_AUTHORITY_DIGEST_DOMAIN,
        **payload_json,
        "approval_digest": approval_digest,
    }
    manifest_digest = hashlib.sha256(_canon(manifest_payload)).hexdigest()
    manifest = FrvWireAuthorityManifest(
        **{
            **fields,
            "approved_wire_ids": tuple(fields["approved_wire_ids"]),
            "w1_artifact_components": tuple(fields["w1_artifact_components"]),
            "origin_xyz": tuple(fields["origin_xyz"]),
            "approval_digest": approval_digest,
            "manifest_digest": manifest_digest,
        }
    )
    manifest_path = tmp_path / "authority_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False), "utf-8")
    authority_component = _component(
        manifest_path, "authority_manifest_json", FrvA2ArtifactComponentKind.MANIFEST_JSON
    )
    if authority_mode is FrvWireAuthorityMode.W1_MODEL_SET:
        w1_json, w1_npz = json_path, npz_path
    else:
        w1_json, w1_npz = None, None
    reference = FrvWireAuthorityReference(
        manifest=manifest,
        authority_manifest_path=manifest_path,
        authority_manifest_component=authority_component,
        w1_json_path=w1_json,
        w1_npz_path=w1_npz,
    )
    return reference, manifest_path


def _approved_ids(
    arrays: dict[str, np.ndarray],
    authority_mode: FrvWireAuthorityMode,
) -> tuple[int, ...]:
    if authority_mode is FrvWireAuthorityMode.EMPTY_INVENTORY:
        return ()
    return tuple(int(value) for value in arrays["wire_id"])


def _run_wire(
    path: Path,
    tmp_path: Path,
    *,
    source_chunk_size: int = 3,
    authority_reference=None,
    wire_config=None,
    arrays=None,
    source_content_digest: str | None = None,
):
    if wire_config is None:
        wire_config = _wire_config()
    reference, configuration, source_result = _run_source(
        path, chunk_size=source_chunk_size, wire_config=wire_config
    )
    if authority_reference is None:
        authority_reference, _ = _build_authority(
            tmp_path,
            source_result,
            arrays=arrays,
            source_content_digest=source_content_digest,
        )
    session = load_frv_wire(
        reference,
        source_result,
        authority_reference,
        frv_configuration=configuration,
        provider_config=wire_config,
        source_chunk_size=source_chunk_size,
    )
    return session, source_result


def _rows(chunks):
    return {
        int(chunk.original_index[r]): (chunk, r)
        for chunk in chunks
        for r in range(chunk.point_count)
    }


_SOURCE_POINTS = [
    (0.0, 0.0, 10.0),
    (0.0, 0.05, 10.0),
    (0.0, 0.30, 10.0),
    (0.0, 1.00, 10.0),
    (0.0, 2.00, 10.0),
]


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------


def test_exactly_one_source_pass_and_relations(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    session, _ = _run_wire(las_path, tmp_path)
    with mock.patch(
        "algorithms.vegetation.wire_provider.load_frv_source",
        wraps=wp.load_frv_source,
    ) as patched:
        with session:
            chunks = list(session.iter_chunks())
            result = session.finalize()
    assert patched.call_count == 1
    rows = _rows(chunks)
    expected = {
        0: (FrvWirePointRelationState.WIRE_BODY_POSSIBLE, 0.0, 0),
        1: (FrvWirePointRelationState.WIRE_BODY_POSSIBLE, 0.05, 0),
        2: (FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT, 0.30, 0),
        3: (FrvWirePointRelationState.WIRE_PROXIMITY_ONLY, 1.00, 0),
        4: (FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN, 0.0, -1),
    }
    for index, (relation, distance, wire_id) in expected.items():
        chunk, row = rows[index]
        assert int(chunk.provider_state[row]) == int(FrvWireProviderState.WIRE_RELATION_VALID)
        assert int(chunk.relation_state[row]) == int(relation)
        assert float(chunk.centerline_distance_m[row]) == pytest.approx(distance)
        assert int(chunk.nearest_wire_id[row]) == wire_id
    indexes = [
        int(chunk.original_index[r])
        for chunk in chunks
        for r in range(chunk.point_count)
    ]
    assert indexes == list(range(5))
    assert [chunk.is_last for chunk in chunks] == [False, True]
    validate_finalized_frv_wire_provider_result(result)
    proof = result.proof
    assert proof.wire_relation_valid_count == 5
    assert proof.outside_influence_domain_count == 1
    assert proof.wire_body_possible_count == 2
    assert proof.wire_vegetation_conflict_count == 1
    assert proof.wire_proximity_only_count == 1
    assert proof.approved_wire_count == 1
    assert proof.minimum_centerline_distance_m == 0.0
    assert proof.maximum_centerline_distance_m == 1.0


def test_bnd1_evidence(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    session, _ = _run_wire(las_path, tmp_path)
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    evidence = to_frv_a2_wire_provider_binding_evidence(result)
    assert evidence.provider_key == FRV_WIRE_PROVIDER_KEY
    assert evidence.provider_role is FrvA2ProviderRole.WIRE
    assert evidence.evidence_class.value == "formal_provider"
    assert evidence.public_module == "algorithms.vegetation.wire_provider"
    assert evidence.public_loader_name == "load_frv_wire"
    assert evidence.domain_kind.value == "full_source"
    assert evidence.provider_readiness.value == "validated"
    assert evidence.quality_flags == 0
    upstream = set(evidence.upstream_binding_digests)
    proof = result.proof
    assert proof.authority_manifest_digest in upstream
    assert proof.wire_geometry_content_digest in upstream
    assert proof.wire_point_content_digest in upstream
    assert len(evidence.artifact_components) == 3


def test_chunk_size_invariance(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    proofs = []
    for chunk_size in (2, 5, 1000000):
        session, _ = _run_wire(las_path, tmp_path, source_chunk_size=chunk_size)
        with session:
            list(session.iter_chunks())
            result = session.finalize()
        proofs.append(result.proof.to_dict())
    assert proofs[0] == proofs[1] == proofs[2]


def test_empty_authority(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    reference, configuration, source_result = _run_source(las_path, chunk_size=3)
    authority, _ = _build_authority(
        tmp_path,
        source_result,
        arrays=_w1_arrays(),
        authority_mode=FrvWireAuthorityMode.EMPTY_INVENTORY,
    )
    session = load_frv_wire(
        reference,
        source_result,
        authority,
        frv_configuration=configuration,
        provider_config=_wire_config(),
        source_chunk_size=3,
    )
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    assert len(chunks) == 2
    for chunk in chunks:
        assert (
            chunk.provider_state
            == int(FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN)
        ).all()
        assert (
            chunk.relation_state
            == int(FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN)
        ).all()
        assert (chunk.nearest_wire_id == -1).all()
        assert (chunk.centerline_distance_m == 0.0).all()
    proof = result.proof
    assert proof.authority_mode is FrvWireAuthorityMode.EMPTY_INVENTORY
    assert proof.wire_not_applicable_proven_count == 5
    assert proof.outside_influence_domain_count == 5
    assert proof.wire_relation_valid_count == 0
    assert proof.minimum_centerline_distance_m == 0.0
    validate_finalized_frv_wire_provider_result(result)


def test_relation_boundaries(tmp_path: Path) -> None:
    # Points at exactly r0, r1, and r2 decide to the closed boundary.
    las_path = tmp_path / "scene-1.las"
    points = [
        (0.0, 0.10, 10.0),
        (0.0, 0.50, 10.0),
        (0.0, 1.50, 10.0),
        (0.0, 1.51, 10.0),
    ]
    _write_wire_las(las_path, points)
    session, _ = _run_wire(las_path, tmp_path)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    rows = _rows(chunks)
    expected_relations = [
        FrvWirePointRelationState.WIRE_BODY_POSSIBLE,
        FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT,
        FrvWirePointRelationState.WIRE_PROXIMITY_ONLY,
        FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN,
    ]
    for index, relation in enumerate(expected_relations):
        chunk, row = rows[index]
        assert int(chunk.relation_state[row]) == int(relation)
    validate_finalized_frv_wire_provider_result(result)


def test_deterministic_tie_prefers_lower_wire_id(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, [(0.0, 0.10, 10.0)])
    arrays = _w1_arrays(wire_offsets=(0.0, 0.20))
    session, _ = _run_wire(las_path, tmp_path, arrays=arrays)
    with session:
        chunks = list(session.iter_chunks())
        result = session.finalize()
    chunk, row = chunks[0], 0
    assert int(chunk.nearest_wire_id[row]) == 0
    assert int(chunk.relation_state[row]) == int(FrvWirePointRelationState.WIRE_BODY_POSSIBLE)
    assert result.proof.approved_wire_count == 2
    validate_finalized_frv_wire_provider_result(result)


def test_geometry_digest_is_bound_to_source_origin(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    arrays = _w1_arrays()
    reference, configuration, source_result = _run_source(las_path, chunk_size=3)
    origin = tuple(source_result.source_identity.origin_context)
    edges = _w1_digest_edges(arrays, origin)
    digest = frv_wire_geometry_content_digest(
        source_content_identity_digest=source_result.source_identity.source_content_identity_digest,
        source_identity_record_digest=source_result.source_identity.source_identity_record_digest,
        origin_xyz=origin,
        edges=edges,
    )
    wrong_origin_digest = frv_wire_geometry_content_digest(
        source_content_identity_digest=source_result.source_identity.source_content_identity_digest,
        source_identity_record_digest=source_result.source_identity.source_identity_record_digest,
        origin_xyz=(1.0, 2.0, 3.0),
        edges=[
            (
                edge[0],
                edge[1],
                edge[2],
                edge[3] + 1,
                edge[4] + 2,
                edge[5] + 3,
                edge[6] + 1,
                edge[7] + 2,
                edge[8] + 3,
            )
            for edge in edges
        ],
    )
    assert digest != wrong_origin_digest


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def _run_wire_raises(las_path: Path, tmp_path: Path, **kwargs):
    session, _ = _run_wire(las_path, tmp_path, **kwargs)
    with session:
        list(session.iter_chunks())
        session.finalize()


def test_failure_geometry_digest_mismatch(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    arrays = _w1_arrays()
    reference, configuration, source_result = _run_source(las_path, chunk_size=3)
    authority, _ = _build_authority(
        tmp_path, source_result, arrays=arrays, geometry_digest="0" * 64
    )
    with pytest.raises(FrvWireLoadError) as excinfo:
        load_frv_wire(
            reference,
            source_result,
            authority,
            frv_configuration=configuration,
            provider_config=_wire_config(),
            source_chunk_size=3,
        )
    assert excinfo.value.failure_record.field_name == "wire_geometry"
    assert excinfo.value.failure_record.error_code.value == "FRV_WIRE_GEOMETRY_UNAVAILABLE"


def test_failure_non_confirmed_wire(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    arrays = _w1_arrays()
    arrays["wire_status"] = np.asarray([2], dtype=np.uint8)  # REVIEW_REQUIRED
    with pytest.raises(FrvWireLoadError) as excinfo:
        _run_wire(las_path, tmp_path, arrays=arrays)
    assert excinfo.value.failure_record.field_name == "wire_geometry"
    assert excinfo.value.quality_flags == 128


def test_failure_identity_mismatch(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    arrays = _w1_arrays()
    with pytest.raises(FrvWireLoadError) as excinfo:
        _run_wire(las_path, tmp_path, arrays=arrays, source_content_digest="0" * 64)
    assert excinfo.value.failure_record.error_code.value == "FRV_WIRE_IDENTITY_MISMATCH"


def test_failure_w1_missing(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    reference, configuration, source_result = _run_source(las_path, chunk_size=3)
    authority, _ = _build_authority(tmp_path, source_result, arrays=_w1_arrays())
    missing = FrvWireAuthorityReference(
        manifest=authority.manifest,
        authority_manifest_path=authority.authority_manifest_path,
        authority_manifest_component=authority.authority_manifest_component,
        w1_json_path=tmp_path / "missing.json",
        w1_npz_path=tmp_path / "missing.npz",
    )
    with pytest.raises(FrvWireLoadError) as excinfo:
        load_frv_wire(
            reference,
            source_result,
            missing,
            frv_configuration=configuration,
            provider_config=_wire_config(),
            source_chunk_size=3,
        )
    assert excinfo.value.failure_record.field_name == "wire_artifact"


def test_failure_edge_cell_limit(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    config = _wire_config(max_edges_per_index_cell=2)
    with pytest.raises(FrvWireLoadError) as excinfo:
        _run_wire(las_path, tmp_path, wire_config=config)
    assert excinfo.value.failure_record.field_name == "wire_index"


def test_failure_authority_component_corruption(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    arrays = _w1_arrays()
    reference, configuration, source_result = _run_source(las_path, chunk_size=3)
    authority, manifest_path = _build_authority(tmp_path, source_result, arrays=arrays)
    manifest_path.write_bytes(b"corrupted bytes")
    with pytest.raises(FrvWireLoadError) as excinfo:
        load_frv_wire(
            reference,
            source_result,
            authority,
            frv_configuration=configuration,
            provider_config=_wire_config(),
            source_chunk_size=3,
        )
    assert excinfo.value.failure_record.field_name == "wire_authority"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_repeated_iterator(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    session, _ = _run_wire(las_path, tmp_path)
    with session:
        list(session.iter_chunks())
        with pytest.raises(FrvWireLoadError) as excinfo:
            list(session.iter_chunks())
        assert excinfo.value.failure_record.field_name == "session_state"


def test_lifecycle_premature_finalize(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    session, _ = _run_wire(las_path, tmp_path)
    with pytest.raises(FrvWireLoadError) as excinfo:
        session.finalize()
    assert excinfo.value.failure_record.error_code.value == "FRV_STREAM_INCOMPLETE"
    session.close()


def test_lifecycle_early_termination_cleans_up(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    session, _ = _run_wire(las_path, tmp_path)
    with session:
        iterator = session.iter_chunks()
        next(iterator)
    # GeneratorExit path aborts and closes the source session without a leak.
    assert session._state in ("ABORTED", "CLOSED")


def test_close_is_idempotent(tmp_path: Path) -> None:
    las_path = tmp_path / "scene-1.las"
    _write_wire_las(las_path, _SOURCE_POINTS)
    session, _ = _run_wire(las_path, tmp_path)
    session.close()
    session.close()
    assert session._state == "CLOSED"


def test_public_loader_rejects_bad_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        load_frv_wire(None, None, None)  # type: ignore[arg-type]
