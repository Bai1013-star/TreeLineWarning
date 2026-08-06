"""FRV-to-V4 input handoff tests (PLAN-FROZEN)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import laspy
import numpy as np
import pytest
from algorithms.vegetation import build_frv_membership, build_v4_input_handoff
from algorithms.vegetation import v4_inputs as v4i
from algorithms.vegetation.frv_artifacts import decode_frv_block
from core.contracts.frv import FrvConfiguration, frv_digest
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
)
from core.contracts.frv_membership import (
    FrvMembershipExecutionResult,
    FrvMembershipExecutionState,
    FrvScopeEvidenceReference,
    default_frv_membership_policy_config,
    default_frv_scope_policy_config,
)
from core.contracts.v4_inputs import (
    V4InputErrorCode,
    V4InputHandoffResult,
    V4InputLoadError,
    V4PrimaryRoute,
)

from tests.contracts.test_frv_membership_contracts import (
    _empty_manifest,
    _source,
)
from tests.vegetation.test_frv_membership import (
    _configuration,
    _geometry_config,
    _hag_config,
    _wire_config,
    build_evidence_pairs,
    run_membership_ready,
    run_provider_chain,
    write_scene_las,
)

# Frozen non-trivial origin used by the CANDIDATE-002 local-XYZ tests.
_NONZERO_ORIGIN = (500000.0, 4200000.0, 125.0)


def _decode_v4_block(
    path: Path, magic: bytes, fields: tuple[tuple[str, str], ...]
) -> dict[str, np.ndarray]:
    order = tuple(name for name, _ in fields)
    dtypes = {name: np.dtype(dtype) for name, dtype in fields}
    row_shapes = {name: () for name, _ in fields}
    row_shapes["xyz_local_m"] = (3,)
    _header, arrays = decode_frv_block(
        path.read_bytes(),
        magic=magic,
        schema_order=order,
        dtypes=dtypes,
        row_shapes=row_shapes,
    )
    return arrays


def _absolute_scene_xyz(las_path: Path) -> np.ndarray:
    with laspy.open(las_path) as reader:
        points = reader.read()
    return np.column_stack(
        (
            np.asarray(points.x, dtype=np.float64),
            np.asarray(points.y, dtype=np.float64),
            np.asarray(points.z, dtype=np.float64),
        )
    )


@pytest.fixture(scope="module")
def ready(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("v4_ready"))
    packed = run_membership_ready(root)
    result = packed[0]
    assert result.execution_state is FrvMembershipExecutionState.READY_FOR_V4
    return packed


def test_ready_handoff_partition_conserves(ready) -> None:
    result, _las, _col, _ground, reference, source_result, configuration, _ref, _auth = ready
    out = result.output_reference.root_path.parent / "v4out"
    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=out,
        source_chunk_size=40,
    )
    assert isinstance(handoff, V4InputHandoffResult)
    report = handoff.coverage_report
    artifact = result.membership_artifact
    total_routed = (
        report.normal_segmentation_count
        + report.measured_point_review_count
        + report.excluded_audit_count
        + report.point_recovery_failed_count
    )
    assert total_routed == artifact.unique_in_scope_point_count
    assert report.frv_unique_in_scope_point_count == artifact.unique_in_scope_point_count
    assert report.segmentation_input_complete is True
    assert report.v4_routing_complete is True
    assert report.measured_clearance_complete is False
    assert report.normal_segmentation_count > 0
    assert report.point_recovery_failed_count == 0
    assert report.unrecovered_region_count == 0
    # Normal and review sets are disjoint and exact.
    recovered = handoff.recovered_vegetation
    unresolved = handoff.unresolved_measured_points
    assert recovered is not None and unresolved is not None
    normal_indexes = set(int(v) for v in recovered.original_index)
    review_indexes = set(int(v) for v in unresolved.original_index)
    assert normal_indexes.isdisjoint(review_indexes)
    assert len(normal_indexes) == report.normal_segmentation_count
    assert len(review_indexes) == report.measured_point_review_count
    # All review points carry the measured-review primary route.
    assert (unresolved.primary_route == int(V4PrimaryRoute.MEASURED_POINT_REVIEW)).all()
    assert handoff.output_reference is not None
    assert handoff.output_reference.root_path.is_dir()
    assert handoff.output_reference.validated_view_json_path.is_file()
    assert handoff.output_reference.coverage_json_path.is_file()
    assert handoff.output_reference.handoff_manifest_path.is_file()
    assert handoff.output_reference.normal_block_paths
    assert handoff.output_reference.review_block_paths


def test_ready_handoff_output_has_no_tree_fields(ready) -> None:
    result, _las, _col, _ground, reference, source_result, configuration, _ref, _auth = ready
    out = result.output_reference.root_path.parent / "v4out_no_tree"
    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=out,
        source_chunk_size=40,
    )
    manifest_text = handoff.output_reference.handoff_manifest_path.read_text("utf-8")
    for forbidden in ("tree_id", "apex", "crown", "chm", "clearance", "slice_segmentation"):
        assert forbidden not in manifest_text.lower()
    # The recovered set is recovered XYZ (source replay), never downsampled.
    xyz = handoff.recovered_vegetation.xyz_local_m
    assert xyz.shape[1] == 3


def _synthetic_result(state: FrvMembershipExecutionState) -> FrvMembershipExecutionResult:
    source = _source()
    manifest = _empty_manifest()
    result = FrvMembershipExecutionResult.__new__(FrvMembershipExecutionResult)
    object.__setattr__(result, "execution_state", state)
    object.__setattr__(result, "source_identity", source)
    object.__setattr__(result, "a2_readiness", None)
    object.__setattr__(result, "coverage_manifest", manifest)
    object.__setattr__(result, "membership_artifact", None)
    object.__setattr__(result, "payload_schema", None)
    object.__setattr__(result, "artifact_binding", None)
    object.__setattr__(result, "decision_evidence_artifact", None)
    object.__setattr__(result, "source_proof_digest", "1" * 64)
    object.__setattr__(result, "hag_proof_digest", None)
    object.__setattr__(result, "geometry_proof_digest", None)
    object.__setattr__(result, "wire_proof_digest", None)
    object.__setattr__(result, "provider_binding_digests", ())
    object.__setattr__(result, "override_digests", ())
    object.__setattr__(result, "failure_records", ())
    object.__setattr__(result, "output_reference", None)
    object.__setattr__(result, "result_digest", result.computed_digest())
    return result


def test_no_scope_result_not_ready(ready) -> None:
    _result, _las, _col, _ground, reference, source_result, configuration, _ref, _auth = ready
    result = _synthetic_result(FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE)

    with pytest.raises(V4InputLoadError) as excinfo:
        build_v4_input_handoff(
            reference,
            source_result,
            result,
            frv_configuration=configuration,
            output_dir=Path("."),
        )
    assert excinfo.value.code is V4InputErrorCode.FRV_RESULT_NOT_READY


def test_coverage_blocked_coverage_only_handoff(ready) -> None:
    _result, _las, _col, _ground, reference, source_result, configuration, _ref, _auth = ready
    result = _synthetic_result(FrvMembershipExecutionState.COVERAGE_BLOCKED)

    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=Path("."),
    )
    assert handoff.validated_membership is None
    assert handoff.recovered_vegetation is None
    assert handoff.unresolved_measured_points is None
    report = handoff.coverage_report
    assert report.frv_routing_complete is False
    assert report.frv_membership_decision_complete is False
    assert report.segmentation_input_complete is False
    assert report.measured_clearance_complete is False
    assert report.v4_routing_complete is True
    assert report.review_required is True
    assert report.normal_segmentation_count == 0
    assert report.measured_point_review_count == 0
    assert handoff.output_reference is None


def test_replay_rejects_fabricated_source_fingerprint(ready) -> None:
    """DC-001: a fabricated expected source fingerprint fails closed before use."""

    _result, _las, _col, _ground, reference, source_result, configuration, _ref, _auth = ready
    from algorithms.vegetation.v4_inputs import _recover_xyz

    with pytest.raises(V4InputLoadError) as excinfo:
        _recover_xyz(
            reference,
            source_result,
            configuration,
            [0],
            40,
            expected_source_proof_digest="1" * 64,
        )
    assert excinfo.value.code is V4InputErrorCode.SOURCE_IDENTITY_FAILED


def test_ready_handoff_rejects_altered_source(tmp_path: Path) -> None:
    """DC-001: replaying a mutated source fails closed with no partial handoff."""

    packed = run_membership_ready(tmp_path)
    result, las_path, _col, _ground, reference, source_result, configuration, _ref, _auth = packed
    data = bytearray(las_path.read_bytes())
    data[-1] ^= 0xFF
    las_path.write_bytes(bytes(data))
    with pytest.raises(V4InputLoadError):
        build_v4_input_handoff(
            reference,
            source_result,
            result,
            frv_configuration=configuration,
            output_dir=tmp_path / "v4out_altered",
            source_chunk_size=40,
        )


def test_real_blocked_result_preserves_rows_and_v4_handoff(tmp_path: Path) -> None:
    """DC-002: a real COVERAGE_BLOCKED result keeps blocked required rows,
    binds the real source fingerprint and drives one unrecovered record per
    blocked row through the V4 coverage-only handoff."""

    las_path = tmp_path / "s.las"
    write_scene_las(las_path)
    hag_config = _hag_config()
    geometry_config = _geometry_config()
    wire_config = _wire_config()
    test_only = _configuration().to_digest_dict()
    test_only["wire_provider_interface_version"] = wire_config.interface_version
    test_only["wire_provider_config_digest"] = wire_config.provider_config_digest
    test_only["configuration_digest"] = frv_digest(test_only)
    configuration = FrvConfiguration.from_dict(test_only)
    chain = run_provider_chain(tmp_path, source_chunk_size=40)
    reference = chain[3]
    source_result = chain[6]
    authority = chain[10]
    scope_ref = build_evidence_pairs(
        tmp_path, las_path=las_path, column_indexes=chain[1], ground_indexes=chain[2]
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = build_frv_membership(
        reference,
        source_result,
        scope_ref,
        authority,
        (),
        frv_configuration=configuration,
        hag_config=hag_config,
        geometry_config=geometry_config,
        wire_config=wire_config,
        scope_policy=default_frv_scope_policy_config(),
        membership_policy=default_frv_membership_policy_config(),
        output_dir=tmp_path / "out",
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.COVERAGE_BLOCKED
    manifest = result.coverage_manifest
    assert manifest.routing_complete is False
    assert manifest.scope_coverage_complete is False
    assert manifest.coverage_blocked_count == manifest.required_inventory_count
    assert manifest.coverage_blocked_count > 0
    assert len(manifest.records) == manifest.canonical_inventory_count
    assert result.source_proof_digest != "0" * 64

    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=tmp_path / "v4out",
    )
    assert handoff.validated_membership is None
    assert handoff.recovered_vegetation is None
    assert handoff.unresolved_measured_points is None
    assert handoff.unrecovered_regions
    assert len(handoff.unrecovered_regions) == manifest.coverage_blocked_count
    report = handoff.coverage_report
    assert report.unrecovered_region_count == len(handoff.unrecovered_regions)
    assert report.v4_routing_complete is True
    assert report.frv_routing_complete is False
    assert report.frv_membership_decision_complete is False
    assert report.segmentation_input_complete is False
    assert report.measured_clearance_complete is False


# ---------------------------------------------------------------------------
# DC-002 test A: invalid scope evidence with known required rows
# ---------------------------------------------------------------------------


def test_dc002_invalid_evidence_known_rows_blocked_and_v4_handoff(
    tmp_path: Path,
) -> None:
    """DC-002-A: invalid scope evidence with at least two known required rows
    produces a formal COVERAGE_BLOCKED manifest that preserves every row and
    drives one V4UnrecoveredRegionRecord per blocked row without opening a
    source replay."""

    las_path = tmp_path / "s.las"
    write_scene_las(las_path)
    chain = run_provider_chain(tmp_path, source_chunk_size=40)
    reference = chain[3]
    configuration = chain[5]
    source_result = chain[6]
    authority = chain[10]
    scope_ref = build_evidence_pairs(
        tmp_path, las_path=las_path, column_indexes=chain[1], ground_indexes=chain[2]
    )
    # Corrupt the A1 NPZ SHA: component verification fails while V1/V2 arrays
    # are still loaded, so every known required row can be enumerated.
    bad_ref = FrvScopeEvidenceReference(
        a1_npz_path=scope_ref.a1_npz_path,
        a1_json_path=scope_ref.a1_json_path,
        v1_npz_path=scope_ref.v1_npz_path,
        v1_json_path=scope_ref.v1_json_path,
        v2_npz_path=scope_ref.v2_npz_path,
        v2_json_path=scope_ref.v2_json_path,
        a1_npz_component=FrvA2ArtifactComponentBinding(
            component_key="a1_npz",
            component_kind=FrvA2ArtifactComponentKind.PAYLOAD_NPZ,
            sha256="0" * 64,
            size_bytes=0,
        ),
        a1_json_component=scope_ref.a1_json_component,
        v1_npz_component=scope_ref.v1_npz_component,
        v1_json_component=scope_ref.v1_json_component,
        v2_npz_component=scope_ref.v2_npz_component,
        v2_json_component=scope_ref.v2_json_component,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = build_frv_membership(
        reference,
        source_result,
        bad_ref,
        authority,
        (),
        frv_configuration=configuration,
        hag_config=_hag_config(),
        geometry_config=_geometry_config(),
        wire_config=_wire_config(),
        scope_policy=default_frv_scope_policy_config(),
        membership_policy=default_frv_membership_policy_config(),
        output_dir=tmp_path / "out",
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.COVERAGE_BLOCKED
    manifest = result.coverage_manifest
    assert len(manifest.records) >= 2
    assert manifest.coverage_blocked_count == manifest.required_inventory_count
    assert manifest.coverage_blocked_count == len(manifest.records)
    assert manifest.routing_complete is False
    assert manifest.scope_coverage_complete is False
    for record in manifest.records:
        assert record.canonical_inventory_key.startswith("V1:")
        assert record.scope_construction_state is not None
        assert record.review_required is True
        assert record.failure_reason_flags != 0
        assert record.safety_completeness_effect is not None

    # Coverage-only V4 handoff: one unrecovered record per blocked row, and no
    # P-SOURCE replay is opened.
    with mock.patch(
        "algorithms.vegetation.v4_inputs.load_frv_source"
    ) as patched:
        handoff = build_v4_input_handoff(
            reference,
            source_result,
            result,
            frv_configuration=configuration,
            output_dir=tmp_path / "v4out",
        )
    patched.assert_not_called()
    assert handoff.validated_membership is None
    assert handoff.recovered_vegetation is None
    assert handoff.unresolved_measured_points is None
    assert len(handoff.unrecovered_regions) == manifest.coverage_blocked_count
    report = handoff.coverage_report
    assert report.unrecovered_region_count == len(handoff.unrecovered_regions)
    assert report.v4_routing_complete is True
    assert report.frv_routing_complete is False
    assert report.frv_membership_decision_complete is False
    assert report.segmentation_input_complete is False
    assert report.measured_clearance_complete is False


# ---------------------------------------------------------------------------
# CANDIDATE-002: V4 local XYZ (xyz_local_m = source_xyz - origin_context)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ready_nonzero_origin(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("v4_nonzero_ready"))
    packed = run_membership_ready(
        root, source_chunk_size=40, offsets=_NONZERO_ORIGIN
    )
    result = packed[0]
    assert result.execution_state is FrvMembershipExecutionState.READY_FOR_V4
    return packed


def test_candidate002_normal_block_local_xyz_round_trip(
    ready_nonzero_origin,
) -> None:
    """CANDIDATE-002-A: replay absolute XYZ becomes canonical local XYZ once;
    the in-memory recovered set and the published normal block both carry local
    XYZ (absolute - origin) and never absolute XYZ."""

    (
        result,
        las_path,
        _col,
        _ground,
        reference,
        source_result,
        configuration,
        _ref,
        _auth,
    ) = ready_nonzero_origin
    out = result.output_reference.root_path.parent / "v4out_norm"
    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=out,
        source_chunk_size=40,
    )
    recovered = handoff.recovered_vegetation
    assert recovered is not None
    assert handoff.output_reference is not None
    origin = np.asarray(_NONZERO_ORIGIN, dtype=np.float64)
    abs_xyz = _absolute_scene_xyz(las_path)
    arrays = _decode_v4_block(
        handoff.output_reference.normal_block_paths[0],
        v4i.V4_NORMAL_BLOCK_MAGIC,
        v4i._NORMAL_FIELDS,
    )
    # In-memory and published normal points are point-identical and local.
    assert np.allclose(arrays["xyz_local_m"], recovered.xyz_local_m)
    # Local is exactly absolute-minus-origin and therefore not absolute.
    for position, index in enumerate(recovered.original_index):
        assert np.allclose(
            recovered.xyz_local_m[position], abs_xyz[int(index)] - origin
        )
    assert not np.allclose(arrays["xyz_local_m"], recovered.xyz_local_m + origin)


def test_candidate002_review_block_local_xyz_round_trip(
    ready_nonzero_origin,
) -> None:
    """CANDIDATE-002-B: the review block and the in-memory unresolved set both
    encode local XYZ under the same origin rule."""

    (
        result,
        las_path,
        _col,
        _ground,
        reference,
        source_result,
        configuration,
        _ref,
        _auth,
    ) = ready_nonzero_origin
    out = result.output_reference.root_path.parent / "v4out_rev"
    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=out,
        source_chunk_size=40,
    )
    unresolved = handoff.unresolved_measured_points
    assert unresolved is not None
    assert handoff.output_reference is not None
    origin = np.asarray(_NONZERO_ORIGIN, dtype=np.float64)
    abs_xyz = _absolute_scene_xyz(las_path)
    arrays = _decode_v4_block(
        handoff.output_reference.review_block_paths[0],
        v4i.V4_REVIEW_BLOCK_MAGIC,
        v4i._REVIEW_FIELDS,
    )
    assert np.allclose(arrays["xyz_local_m"], unresolved.xyz_local_m)
    for position, index in enumerate(unresolved.original_index):
        assert np.allclose(
            unresolved.xyz_local_m[position], abs_xyz[int(index)] - origin
        )
    assert not np.allclose(arrays["xyz_local_m"], unresolved.xyz_local_m + origin)


def test_candidate002_normal_and_review_share_origin_rule(
    ready_nonzero_origin,
) -> None:
    """CANDIDATE-002-C: normal and review blocks coexist under the same origin
    rule and preserve original_index, HAG and route exactly."""

    (
        result,
        las_path,
        _col,
        _ground,
        reference,
        source_result,
        configuration,
        _ref,
        _auth,
    ) = ready_nonzero_origin
    out = result.output_reference.root_path.parent / "v4out_both"
    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=out,
        source_chunk_size=40,
    )
    recovered = handoff.recovered_vegetation
    unresolved = handoff.unresolved_measured_points
    assert recovered is not None and unresolved is not None
    assert recovered.original_index.shape[0] > 0
    assert unresolved.original_index.shape[0] > 0
    assert handoff.output_reference is not None
    origin = np.asarray(_NONZERO_ORIGIN, dtype=np.float64)
    abs_xyz = _absolute_scene_xyz(las_path)
    # Both sets apply the same origin rule against the true absolute scene.
    for position, index in enumerate(recovered.original_index):
        assert np.allclose(recovered.xyz_local_m[position], abs_xyz[int(index)] - origin)
    for position, index in enumerate(unresolved.original_index):
        assert np.allclose(unresolved.xyz_local_m[position], abs_xyz[int(index)] - origin)
    # original_index / HAG / route survive the block round trip unchanged.
    normal = _decode_v4_block(
        handoff.output_reference.normal_block_paths[0],
        v4i.V4_NORMAL_BLOCK_MAGIC,
        v4i._NORMAL_FIELDS,
    )
    review = _decode_v4_block(
        handoff.output_reference.review_block_paths[0],
        v4i.V4_REVIEW_BLOCK_MAGIC,
        v4i._REVIEW_FIELDS,
    )
    assert (normal["original_index"] == recovered.original_index).all()
    assert np.allclose(normal["hag_m"], recovered.hag_m)
    assert (review["original_index"] == unresolved.original_index).all()
    assert (review["primary_route"] == unresolved.primary_route).all()


def test_candidate002_origin_tamper_fails_closed(tmp_path: Path) -> None:
    """CANDIDATE-002-D: an altered origin_context in the published membership
    result is rejected by the handoff loader with no partial formal output."""

    packed = run_membership_ready(
        tmp_path, source_chunk_size=40, offsets=_NONZERO_ORIGIN
    )
    result, _las, _col, _ground, reference, source_result, configuration, _ref, _auth = packed
    assert result.output_reference is not None
    execution_path = result.output_reference.root_path / "execution_result.json"
    payload = json.loads(execution_path.read_text("utf-8"))
    payload["source_identity"]["origin_context"] = [500001.0, 4200001.0, 126.0]
    execution_path.write_text(json.dumps(payload), "utf-8")
    out = tmp_path / "v4out_tamper"
    with pytest.raises((ValueError, V4InputLoadError)):
        build_v4_input_handoff(
            reference,
            source_result,
            result,
            frv_configuration=configuration,
            output_dir=out,
            source_chunk_size=40,
        )
    assert out.exists() is False


def test_candidate002_local_xyz_digest_sensitivity_to_origin() -> None:
    """CANDIDATE-002-E: the same absolute XYZ under different origin contexts
    yields different local payloads and block digests; the expected digest is
    fixed independently (never generated by the production digest function)."""

    absolute = {
        0: np.asarray([500000.0, 4200000.0, 125.0]),
        1: np.asarray([500000.2, 4200000.4, 126.5]),
    }
    origin_a = np.asarray([500000.0, 4200000.0, 125.0])
    origin_b = np.asarray([500010.0, 4200010.0, 135.0])
    payload_rows = {
        index: {
            name: np.asarray(value)
            for name, value in {
                "clamped_hag": 1.5 if index == 0 else 1.7,
                "membership_state": 1,
                "hag_evidence_state": 1,
                "segmentation_hag_readiness": 1,
                "reason_flags": 0,
            }.items()
        }
        for index in (0, 1)
    }
    local_a = {index: absolute[index] - origin_a for index in absolute}
    local_b = {index: absolute[index] - origin_b for index in absolute}
    blocks_a = v4i._encode_normal_blocks([0, 1], local_a, payload_rows)
    blocks_b = v4i._encode_normal_blocks([0, 1], local_b, payload_rows)
    digest_a = hashlib.sha256(blocks_a[0]).hexdigest()
    digest_b = hashlib.sha256(blocks_b[0]).hexdigest()
    assert digest_a != digest_b
    # Independently computed expected digest for the origin-a local payload.
    assert (
        digest_a
        == "cdf519483d1b6b54a057aaba83c315974b35025e296ee7274e791b89bf021d48"
    )
