"""FRV full-resolution membership orchestrator tests (PLAN-FROZEN).

This module also hosts the shared Session B LAS/evidence fixtures used by the
V4 handoff tests (``run_membership_ready`` and friends).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import core.contracts.frv as frv
import laspy
import numpy as np
import pytest
from algorithms.io.source_provider import load_frv_source
from algorithms.vegetation import (
    build_frv_membership,
    load_frv_geometry,
    load_frv_hag,
    load_frv_membership_result,
    load_frv_wire,
)
from algorithms.vegetation.frv_membership import (
    FrvMembershipFailClosedError,
    _wire_clear,
)
from core.contracts.frv import FrvConfiguration, FrvWireProviderState, frv_digest
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2ProviderRole,
)
from core.contracts.frv_membership import (
    FrvMembershipExecutionState,
    FrvScopeEvidenceReference,
    default_frv_membership_policy_config,
    default_frv_scope_policy_config,
)
from core.contracts.frv_source import (
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourcePassThroughDimension,
    FrvSourceProviderReference,
)
from core.contracts.frv_wire import (
    FrvWireAuthorityMode,
    FrvWirePointRelationState,
)

from tests.contracts.test_frv_geometry_contracts import _config as _geometry_config
from tests.contracts.test_frv_hag_contracts import _config as _hag_config
from tests.contracts.test_frv_wire_contracts import _config as _wire_config
from tests.io.test_source_provider import _configuration, _provider_config
from tests.vegetation.test_wire_provider import (
    _build_authority,
    _source_reference,
)

_V1_SCHEMA = "vegetation-candidates-v1"
_V1_ALGORITHM = "v1-explainable-woody-candidates-v1"
_V2_SCHEMA = "vegetation-instances-v2"
_V2_ALGORITHM = "v2-generalized-morphology-safe-vegetation-instances-v2"


# ---------------------------------------------------------------------------
# Scene + A1/V1/V2 fixture builders
# ---------------------------------------------------------------------------


def write_scene_las(
    path: Path,
    *,
    ground_step: float = 0.1,
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[list[int], list[int]]:
    """Write a LAS with a flat ground plane and a vertical woody column.

    Ground points get classification 2 and occupy local x/y in [0, 2]; the
    woody column has classification 5 at (5, 5, z) for z in [0, 2].  ``offsets``
    become the LAS header offsets (``SourceIdentity.origin_context``) so the
    stored points carry large absolute coordinates while the local scene is
    unchanged.  Returns the (column_indexes, ground_indexes) original-index
    lists.
    """

    ground: list[tuple[float, float, float]] = []
    for dx in np.arange(0.0, 2.0 + ground_step, ground_step):
        for dy in np.arange(0.0, 2.0 + ground_step, ground_step):
            ground.append(
                (float(offsets[0] + dx), float(offsets[1] + dy), float(offsets[2]))
            )
    column: list[tuple[float, float, float]] = []
    for z in np.arange(0.0, 2.0 + 0.02, 0.02):
        column.append(
            (float(offsets[0] + 5.0), float(offsets[1] + 5.0), float(offsets[2] + z))
        )
    points = column + ground
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([float(v) for v in offsets])
    las = laspy.LasData(header)
    arr = np.asarray(points, dtype=np.float64)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    classification = np.zeros(len(points), dtype=np.uint8)
    classification[: len(column)] = 5
    classification[len(column) :] = 2
    las.classification = classification
    las.write(path)
    return list(range(len(column))), list(range(len(column), len(points)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component(
    path: Path, key: str, kind: FrvA2ArtifactComponentKind
) -> FrvA2ArtifactComponentBinding:
    return FrvA2ArtifactComponentBinding(
        component_key=key,
        component_kind=kind,
        sha256=_sha(path),
        size_bytes=path.stat().st_size,
    )


def _read_las_xyz(las_path: Path) -> np.ndarray:
    with laspy.open(las_path) as reader:
        points = reader.read()
    return np.column_stack(
        (
            np.asarray(points.x, dtype=np.float64),
            np.asarray(points.y, dtype=np.float64),
            np.asarray(points.z, dtype=np.float64),
        )
    )


def write_a1_pair(
    root: Path,
    *,
    xyz: np.ndarray,
    source_id: str = "scene-1",
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[Path, Path]:
    """Write a minimal A1 JSON+NPZ pair covering the given points."""

    npz_path = root / "a1.npz"
    json_path = root / "a1.json"
    count = int(xyz.shape[0])
    origin = np.asarray([float(v) for v in offsets], dtype=np.float64)
    np.savez_compressed(
        npz_path,
        source_id=np.asarray(source_id),
        manifest_source_id=np.asarray(source_id),
        xyz_local=(xyz - origin).astype(np.float32),
        origin_xyz=origin,
        original_index=np.arange(count, dtype=np.int64),
        sample_index=np.arange(count, dtype=np.int64),
        voxel_key=np.floor((xyz - origin) / 0.05).astype(np.int64),
        voxel_point_count=np.ones(count, dtype=np.int32),
    )
    payload = {
        "source_id": source_id,
        "input_path": "data/samples/scene.las",
        "index_map_path": "data/samples/scene_index_map.npz",
        "manifest_path": "data/samples/scene_manifest.json",
        "output_npz_path": "outputs/pointcloud/scene_preprocessed.npz",
        "output_json_path": "outputs/pointcloud/scene_preprocessed.json",
        "preview_html_path": None,
        "config": {"source_id": source_id},
        "statistics": {
            "input_point_count": count,
            "processed_input_point_count": count,
            "valid_point_count": count,
            "nan_inf_removed_count": 0,
            "abnormal_coord_removed_count": 0,
            "outlier_removed_count": 0,
            "downsampled_point_count": count,
            "downsample_ratio": 1.0,
            "chunk_size": 250000,
            "voxel_size_m": 0.05,
            "elapsed_seconds": 0.1,
        },
        "manifest_source_id": source_id,
        "origin_xyz": [float(v) for v in offsets],
        "arrays": {},
        "warnings": [],
        "created_at": "2026-08-06T00:00:00+00:00",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    return npz_path, json_path


def write_v1_pair(
    root: Path,
    *,
    xyz: np.ndarray,
    levels: np.ndarray,
    a1_npz_sha: str,
    a1_json_sha: str,
    source_id: str = "scene-1",
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[Path, Path]:
    """Write a minimal V1 JSON+NPZ pair (all points unclustered upstream)."""

    npz_path = root / "v1.npz"
    json_path = root / "v1.json"
    count = int(xyz.shape[0])
    levels = np.asarray(levels, dtype=np.uint8)
    origin = np.asarray([float(v) for v in offsets], dtype=np.float64)
    arrays = {
        "source_id": np.asarray(source_id),
        "manifest_source_id": np.asarray(source_id),
        "xyz_local": (xyz - origin).astype(np.float32),
        "origin_xyz": origin,
        "original_index": np.arange(count, dtype=np.int64),
        "sample_index": np.arange(count, dtype=np.int64),
        "wire_id_by_point": np.full(count, -1, dtype=np.int32),
        "hag_state": np.zeros(count, dtype=np.uint8),
        "vegetation_rule_score": np.full(count, 0.8, dtype=np.float32),
        "vegetation_confidence": np.full(count, 0.85, dtype=np.float32),
        "automatic_vegetation_level": levels.copy(),
        "vegetation_level": levels.copy(),
        "vegetation_mask": (levels != 0),
        "vegetation_evidence_flags": np.zeros(count, dtype=np.uint16),
        "vegetation_reason_flags": np.zeros(count, dtype=np.uint16),
        "vegetation_quality_flags": np.zeros(count, dtype=np.uint16),
        "manual_provenance_flags": np.zeros(count, dtype=np.uint8),
        "manual_action": np.zeros(count, dtype=np.uint8),
        "geometry_quality_flags": np.zeros(count, dtype=np.uint16),
        "ground_quality_flags": np.zeros(count, dtype=np.uint16),
        "hag_contribution": np.full(count, 0.4, dtype=np.float32),
        "volumetric_contribution": np.full(count, 0.4, dtype=np.float32),
        "planarity_penalty": np.zeros(count, dtype=np.float32),
        "linearity_penalty": np.zeros(count, dtype=np.float32),
        "density_contribution": np.full(count, 0.3, dtype=np.float32),
        "auxiliary_contribution": np.zeros(count, dtype=np.float32),
        "geometry_confidence": np.ones(count, dtype=np.float32),
        "hag_confidence": np.ones(count, dtype=np.float32),
        "ground_support_confidence": np.ones(count, dtype=np.float32),
        "density_confidence": np.ones(count, dtype=np.float32),
    }
    np.savez_compressed(npz_path, **arrays)
    rejected = int((levels == 0).sum())
    candidate = count - rejected
    low = int((levels == 1).sum())
    medium = int((levels == 2).sum())
    high = int((levels == 3).sum())
    payload = {
        "schema_version": _V1_SCHEMA,
        "source_id": source_id,
        "manifest_source_id": source_id,
        "output_npz_filename": npz_path.name,
        "output_npz_sha256": _sha(npz_path),
        "output_npz_bytes": npz_path.stat().st_size,
        "config": _v1_config().to_dict(),
        "manual_selection": {},
        "input_provenance": {
            "a1_npz": {"logical_path": "a1.npz", "sha256": a1_npz_sha},
            "a1_json": {"logical_path": "a1.json", "sha256": a1_json_sha},
            "a2a_npz": {"logical_path": "a2a.npz", "sha256": "0" * 64},
            "a2a_json": {"logical_path": "a2a.json", "sha256": "0" * 64},
            "a2b_npz": {"logical_path": "a2b.npz", "sha256": "0" * 64},
            "a2b_json": {"logical_path": "a2b.json", "sha256": "0" * 64},
            "a4_npz": {"logical_path": "a4.npz", "sha256": "0" * 64},
            "a4_json": {"logical_path": "a4.json", "sha256": "0" * 64},
        },
        "statistics": {
            "input_point_count": count,
            "candidate_count": candidate,
            "rejected_count": rejected,
            "low_count": low,
            "medium_count": medium,
            "high_count": high,
            "confirmed_wire_exclusion_count": 0,
            "reliable_near_ground_exclusion_count": 0,
            "structural_invalid_count": 0,
            "elapsed_seconds": 0.1,
        },
        "arrays": {},
        "counts": {
            "final_levels": {"REJECTED": rejected, "LOW": low, "MEDIUM": medium, "HIGH": high}
        },
        "formulas": {
            "likelihood": "clip01(...)",
            "confidence": "clip01(...)",
            "rgb_intensity_default_contribution": 0.0,
        },
        "warnings": [],
        "algorithm_version": _V1_ALGORITHM,
        "created_at": "2026-08-06T00:00:00+00:00",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    return npz_path, json_path


def write_v2_pair(
    root: Path,
    *,
    xyz: np.ndarray,
    v1_npz_sha: str,
    source_id: str = "scene-1",
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[Path, Path]:
    """Write a minimal V2 JSON+NPZ pair with zero clusters, all unclustered."""

    npz_path = root / "v2.npz"
    json_path = root / "v2.json"
    count = int(xyz.shape[0])
    origin = np.asarray([float(v) for v in offsets], dtype=np.float64)
    np.savez_compressed(
        npz_path,
        original_index=np.arange(count, dtype=np.int64),
        sample_index=np.arange(count, dtype=np.int64),
        xyz_local=(xyz - origin).astype(np.float32),
        wire_id_by_point=np.full(count, -1, dtype=np.int32),
        vegetation_rule_score=np.full(count, 0.8, dtype=np.float32),
        vegetation_confidence=np.full(count, 0.85, dtype=np.float32),
        automatic_vegetation_level=np.full(count, 3, dtype=np.uint8),
        vegetation_level=np.full(count, 3, dtype=np.uint8),
        hag_state=np.zeros(count, dtype=np.uint8),
        local_scale_raw_m=np.full(count, np.nan, dtype=np.float32),
        local_scale_m=np.full(count, np.nan, dtype=np.float32),
        local_scale_state=np.zeros(count, dtype=np.uint8),
        cluster_id_by_point=np.full(count, -1, dtype=np.int32),
        vegetation_id_by_point=np.full(count, -1, dtype=np.int32),
        cluster_status_by_point=np.zeros(count, dtype=np.uint8),
        distance_ready_by_point=np.zeros(count, dtype=np.bool_),
        assignment_confidence=np.zeros(count, dtype=np.float32),
        assignment_reason_flags=np.zeros(count, dtype=np.uint16),
        assignment_quality_flags=np.zeros(count, dtype=np.uint16),
        manual_provenance_flags=np.zeros(count, dtype=np.uint16),
        origin_xyz=origin,
        cluster_member_offsets=np.array([0], dtype=np.int64),
        ordered_cluster_member_row_index=np.empty(0, dtype=np.int64),
    )
    payload = {
        "schema_version": _V2_SCHEMA,
        "source_id": source_id,
        "manifest_source_id": source_id,
        "origin_xyz": [float(v) for v in offsets],
        "output_npz_filename": npz_path.name,
        "output_npz_sha256": _sha(npz_path),
        "output_npz_bytes": npz_path.stat().st_size,
        "config": _v2_config().to_dict(),
        "manual_log": {
            "v1_npz_sha256": v1_npz_sha,
            "a2a_npz_sha256": "0" * 64,
            "a2b_npz_sha256": "0" * 64,
            "source_id": source_id,
            "manifest_source_id": source_id,
            "actions": [],
        },
        "input_provenance": {
            "v1_npz": {"logical_path": "v1.npz", "sha256": v1_npz_sha},
            "v1_json": {"logical_path": "v1.json", "sha256": "0" * 64},
            "a2a_npz": {"logical_path": "a2a.npz", "sha256": "0" * 64},
            "a2a_json": {"logical_path": "a2a.json", "sha256": "0" * 64},
            "a2b_npz": {"logical_path": "a2b.npz", "sha256": "0" * 64},
            "a2b_json": {"logical_path": "a2b.json", "sha256": "0" * 64},
        },
        "statistics": {
            "input_point_count": count,
            "candidate_count": count,
            "core_point_count": 0,
            "cluster_count": 0,
            "confirmed_count": 0,
            "review_count": 0,
            "provisional_count": 0,
            "rejected_count": 0,
            "assigned_candidate_count": 0,
            "unassigned_candidate_count": count,
            "elapsed_seconds": 0.1,
        },
        "arrays": {},
        "clusters": [],
        "warnings": [],
        "algorithm_version": _V2_ALGORITHM,
        "created_at": "2026-08-06T00:00:00+00:00",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    return npz_path, json_path


def _v1_config() -> Any:
    from core.contracts.vegetation_candidates import VegetationCandidateConfig

    return VegetationCandidateConfig()


def _v2_config() -> Any:
    from core.contracts.vegetation_instances import VegetationInstanceConfig

    return VegetationInstanceConfig()


def build_scope_evidence_reference(
    root: Path, *, a1: tuple[Path, Path], v1: tuple[Path, Path], v2: tuple[Path, Path]
) -> FrvScopeEvidenceReference:
    return FrvScopeEvidenceReference(
        a1_npz_path=a1[0],
        a1_json_path=a1[1],
        v1_npz_path=v1[0],
        v1_json_path=v1[1],
        v2_npz_path=v2[0],
        v2_json_path=v2[1],
        a1_npz_component=_component(a1[0], "a1_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
        a1_json_component=_component(a1[1], "a1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
        v1_npz_component=_component(v1[0], "v1_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
        v1_json_component=_component(v1[1], "v1_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
        v2_npz_component=_component(v2[0], "v2_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ),
        v2_json_component=_component(v2[1], "v2_json", FrvA2ArtifactComponentKind.MANIFEST_JSON),
    )


def build_evidence_pairs(
    tmp_path: Path,
    *,
    las_path: Path,
    column_indexes: list[int],
    ground_indexes: list[int],
    all_candidates: bool = True,
    all_rejected: bool = False,
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FrvScopeEvidenceReference:
    """Build A1/V1/V2 pairs covering the whole scene and return the reference."""

    xyz = _read_las_xyz(las_path)
    a1 = write_a1_pair(tmp_path, xyz=xyz, offsets=offsets)
    if all_rejected:
        levels = np.zeros(len(xyz), dtype=np.uint8)
    elif all_candidates:
        levels = np.full(len(xyz), 3, dtype=np.uint8)
    else:
        levels = np.zeros(len(xyz), dtype=np.uint8)
        for index in column_indexes:
            levels[index] = 3
    v1 = write_v1_pair(
        tmp_path,
        xyz=xyz,
        levels=levels,
        a1_npz_sha=_sha(a1[0]),
        a1_json_sha=_sha(a1[1]),
        offsets=offsets,
    )
    v2 = write_v2_pair(tmp_path, xyz=xyz, v1_npz_sha=_sha(v1[0]), offsets=offsets)
    return build_scope_evidence_reference(tmp_path, a1=a1, v1=v1, v2=v2)


def _source_identity_with_origin(
    path: str,
    file_sha: str,
    count: int,
    bounds: tuple[list[float], list[float]],
    origin: tuple[float, float, float],
) -> frv.SourceIdentity:
    """Like ``test_source_provider._source_identity`` but with a configurable
    ``origin_context`` so the source identity binds the LAS header offsets."""

    relocation_payload = {
        "audit_schema_version": frv.FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "windows",
        "recorded_path_raw": path,
        "resolved_path_raw": path,
        "recorded_path_lexical_normalized": path,
        "resolved_path_lexical_normalized": path,
        "resolved_final_path": None,
        "resolution_method": "exact_match",
        "case_comparison_mode": "case_insensitive",
        "platform_name": "windows",
        "run_id": "run-1",
        "audit_timestamp": "2026-01-01T00:00:00Z",
        "content_identity_result": "matched",
        "warning_flags": 0,
    }
    relocation_payload["audit_digest"] = frv.frv_digest(relocation_payload)
    relocation = frv.FrvRelocationAudit.from_dict(relocation_payload)
    content_payload = {
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": "scene-1",
        "source_file_sha256": file_sha,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "canonical_point_content_spec_version": None,
        "source_point_count": count,
        "source_bounds_min_xyz": list(bounds[0]),
        "source_bounds_max_xyz": list(bounds[1]),
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_context": [float(v) for v in origin],
    }
    content_digest = frv.frv_digest(content_payload)
    record_payload = {
        **content_payload,
        "recorded_source_path": path,
        "resolved_access_path": path,
        "relocation_audit_audit_digest": relocation.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return frv.SourceIdentity.from_dict(
        {
            **content_payload,
            "recorded_source_path": path,
            "resolved_access_path": path,
            "relocation_audit": relocation.to_dict(),
            "source_equivalence_audit": None,
            "source_content_identity_digest": content_digest,
            "source_identity_record_digest": frv.frv_digest(record_payload),
        }
    )


def _source_reference_with_origin(
    path: Path,
    config,
    configuration,
    origin: tuple[float, float, float],
) -> FrvSourceProviderReference:
    """Build a source reference whose identity binds ``origin`` as origin_context."""

    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    with laspy.open(path) as reader:
        header = reader.header
    count = int(header.point_count)
    mins = tuple(float(value) for value in header.mins)
    maxs = tuple(float(value) for value in header.maxs)
    identity = _source_identity_with_origin(
        str(path),
        file_sha,
        count,
        ([mins[0], mins[1], mins[2]], [maxs[0], maxs[1], maxs[2]]),
        tuple(float(value) for value in origin),
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


# ---------------------------------------------------------------------------
# Provider chain and membership helpers
# ---------------------------------------------------------------------------


def run_provider_chain(
    tmp_path: Path,
    *,
    source_chunk_size: int = 40,
    hag_config=None,
    geometry_config=None,
    wire_config=None,
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reviewed: bool = False,
) -> tuple:
    """Write the LAS and run source -> hag -> geometry -> wire (empty authority).

    ``reviewed=True`` builds a REVIEWED_FIXED_POLICY configuration (null
    calibration record) instead of the production-approved default.
    """

    las_path = tmp_path / "s.las"
    column_indexes, ground_indexes = write_scene_las(las_path, offsets=offsets)
    hag_config = hag_config if hag_config is not None else _hag_config()
    geometry_config = geometry_config if geometry_config is not None else _geometry_config()
    wire_config = wire_config if wire_config is not None else _wire_config()
    source_config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    if reviewed:
        configuration = _reviewed_configuration(
            source_config, hag_config, geometry_config, wire_config
        )
    else:
        configuration = _production_configuration(
            source_config, hag_config, geometry_config, wire_config
        )
    if tuple(float(v) for v in offsets) == (0.0, 0.0, 0.0):
        reference = _source_reference(las_path, source_config, configuration)
    else:
        reference = _source_reference_with_origin(
            las_path, source_config, configuration, offsets
        )
    with load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=source_config,
        chunk_size=source_chunk_size,
    ) as session:
        list(session.iter_chunks())
        source_result = session.finalize()
    with load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=hag_config,
        source_chunk_size=source_chunk_size,
    ) as session:
        hag_chunks = list(session.iter_chunks())
        hag_result = session.finalize()
    with load_frv_geometry(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=geometry_config,
        scratch_dir=tmp_path,
        source_chunk_size=source_chunk_size,
    ) as session:
        geometry_chunks = list(session.iter_chunks())
        geometry_result = session.finalize()
    authority, _ = _build_authority(
        tmp_path,
        source_result,
        authority_mode=FrvWireAuthorityMode.EMPTY_INVENTORY,
    )
    with load_frv_wire(
        reference,
        source_result,
        authority,
        frv_configuration=configuration,
        provider_config=wire_config,
        source_chunk_size=source_chunk_size,
    ) as session:
        wire_chunks = list(session.iter_chunks())
        wire_result = session.finalize()
    return (
        las_path,
        column_indexes,
        ground_indexes,
        reference,
        source_config,
        configuration,
        source_result,
        hag_result,
        geometry_result,
        wire_result,
        authority,
        hag_chunks,
        geometry_chunks,
        wire_chunks,
    )


def _production_configuration(
    source_config, hag_config, geometry_config, wire_config
) -> frv.FrvConfiguration:
    base = _configuration().to_digest_dict()
    base["scan_provider_config_digest"] = source_config.provider_config_digest
    base["hag_provider_interface_version"] = hag_config.interface_version
    base["hag_provider_config_digest"] = hag_config.provider_config_digest
    base["geometry_provider_interface_version"] = geometry_config.interface_version
    base["geometry_provider_config_digest"] = geometry_config.provider_config_digest
    base["wire_provider_interface_version"] = wire_config.interface_version
    base["wire_provider_config_digest"] = wire_config.provider_config_digest
    base["scope_policy_digest"] = default_frv_scope_policy_config().policy_digest
    base["feature_policy_digest"] = default_frv_membership_policy_config().policy_digest
    base["configuration_purpose"] = frv.FrvConfigurationPurpose.PRODUCTION_APPROVED
    base["configuration_is_runtime_complete"] = True
    base["formal_completeness_allowed"] = True
    base["membership_artifact_publication_allowed"] = True
    base["calibration_record_digest"] = "c" * 64
    base["configuration_digest"] = frv.frv_digest(base)
    return frv.FrvConfiguration.from_dict(base)


def _reviewed_configuration(
    source_config, hag_config, geometry_config, wire_config
) -> frv.FrvConfiguration:
    """REVIEWED_FIXED_POLICY formal configuration: null calibration record."""
    base = _configuration().to_digest_dict()
    base["scan_provider_config_digest"] = source_config.provider_config_digest
    base["hag_provider_interface_version"] = hag_config.interface_version
    base["hag_provider_config_digest"] = hag_config.provider_config_digest
    base["geometry_provider_interface_version"] = geometry_config.interface_version
    base["geometry_provider_config_digest"] = geometry_config.provider_config_digest
    base["wire_provider_interface_version"] = wire_config.interface_version
    base["wire_provider_config_digest"] = wire_config.provider_config_digest
    base["scope_policy_digest"] = default_frv_scope_policy_config().policy_digest
    base["feature_policy_digest"] = default_frv_membership_policy_config().policy_digest
    base["configuration_purpose"] = frv.FrvConfigurationPurpose.REVIEWED_FIXED_POLICY
    base["configuration_is_runtime_complete"] = True
    base["formal_completeness_allowed"] = True
    base["membership_artifact_publication_allowed"] = True
    base["calibration_record_digest"] = None
    base["configuration_digest"] = frv.frv_digest(base)
    return frv.FrvConfiguration.from_dict(base)


def run_membership_ready(
    root: Path,
    *,
    source_chunk_size: int = 40,
    all_candidates: bool = True,
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reviewed: bool = False,
) -> tuple:
    """Run the full provider chain + evidence pairs + membership publication.

    Returns ``(membership_result, las_path, column_indexes, ground_indexes,
    reference, source_result, configuration, scope_reference, authority)``.
    """

    chain = run_provider_chain(
        root,
        source_chunk_size=source_chunk_size,
        offsets=offsets,
        reviewed=reviewed,
    )
    (
        las_path,
        column_indexes,
        ground_indexes,
        reference,
        source_config,
        configuration,
        source_result,
        hag_result,
        geometry_result,
        wire_result,
        authority,
        *_rest,
    ) = chain
    scope_ref = build_evidence_pairs(
        root,
        las_path=las_path,
        column_indexes=column_indexes,
        ground_indexes=ground_indexes,
        all_candidates=all_candidates,
        offsets=offsets,
    )
    out = root / "membership_out"
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    result = build_frv_membership(
        reference,
        source_result,
        scope_ref,
        authority,
        (),
        frv_configuration=configuration,
        hag_config=_hag_config(),
        geometry_config=_geometry_config(),
        wire_config=_wire_config(),
        scope_policy=default_frv_scope_policy_config(),
        membership_policy=default_frv_membership_policy_config(),
        output_dir=out,
        scratch_dir=scratch,
        source_chunk_size=source_chunk_size,
    )
    return (
        result,
        las_path,
        column_indexes,
        ground_indexes,
        reference,
        source_result,
        configuration,
        scope_ref,
        authority,
    )


# ---------------------------------------------------------------------------
# Membership tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ready(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("membership_ready"))
    result = run_membership_ready(root)
    membership_result = result[0]
    assert membership_result.execution_state is FrvMembershipExecutionState.READY_FOR_V4
    return result


def test_ready_result_state_and_conservation(ready) -> None:
    result = ready[0]
    artifact = result.membership_artifact
    assert artifact is not None
    assert artifact.unique_in_scope_point_count == artifact.evaluated_point_count
    five = (
        artifact.verified_vegetation_candidate_count
        + artifact.excluded_non_candidate_count
        + artifact.review_required_membership_count
        + artifact.manual_include_count
        + artifact.manual_exclude_count
    )
    assert five == artifact.evaluated_point_count
    assert artifact.unasserted_in_scope_count == 0
    assert artifact.verified_vegetation_candidate_count > 0
    assert artifact.excluded_non_candidate_count > 0
    assert artifact.membership_decision_complete is True
    assert result.coverage_manifest.routing_complete is True
    assert result.result_digest == result.computed_digest()


def test_ready_output_files_published(ready) -> None:
    result = ready[0]
    ref = result.output_reference
    assert ref is not None
    assert ref.root_path.is_dir()
    for path in (
        ref.execution_result_path,
        ref.coverage_manifest_path,
        ref.membership_manifest_path,
        ref.binding_path,
        ref.decision_evidence_manifest_path,
        ref.payload_schema_path,
    ):
        assert path.is_file(), f"{path.name} missing"
    assert len(ref.block_paths) == 1
    assert ref.block_paths[0].is_file()
    assert len(ref.decision_evidence_block_paths) == 1
    assert ref.decision_evidence_block_paths[0].is_file()


def test_load_round_trip(ready) -> None:
    result = ready[0]
    reloaded = load_frv_membership_result(result.output_reference.root_path)
    assert reloaded.execution_state is FrvMembershipExecutionState.READY_FOR_V4
    assert (
        reloaded.membership_artifact.manifest_digest == result.membership_artifact.manifest_digest
    )
    assert reloaded.result_digest == result.result_digest
    assert reloaded.coverage_manifest.manifest_digest == result.coverage_manifest.manifest_digest


def test_load_detects_corrupt_block(ready) -> None:
    result = ready[0]
    root = result.output_reference.root_path
    block_path = result.output_reference.block_paths[0]
    original = block_path.read_bytes()
    block_path.write_bytes(original[:-4] + b"\x00\x00\x00\x00")
    try:
        with pytest.raises(ValueError):
            load_frv_membership_result(root)
    finally:
        block_path.write_bytes(original)


def test_load_detects_missing_artifact_file(ready) -> None:
    result = ready[0]
    root = result.output_reference.root_path
    binding_path = root / "binding.json"
    original = binding_path.read_bytes()
    binding_path.unlink()
    try:
        with pytest.raises(ValueError):
            load_frv_membership_result(root)
    finally:
        binding_path.write_bytes(original)


def test_production_policy_rejects_test_only(tmp_path: Path) -> None:
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
    out = tmp_path / "out"
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
        output_dir=out,
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.COVERAGE_BLOCKED
    assert result.membership_artifact is None
    assert result.failure_records
    phases = {failure.phase.value for failure in result.failure_records}
    assert "production_policy" in phases
    assert out.exists() is False


def test_no_relevant_scope_when_all_rejected(tmp_path: Path) -> None:
    las_path = tmp_path / "s.las"
    write_scene_las(las_path)
    chain = run_provider_chain(tmp_path, source_chunk_size=40)
    las_path = chain[0]
    col, ground, reference = chain[1], chain[2], chain[3]
    configuration = chain[5]
    source_result = chain[6]
    authority = chain[10]
    scope_ref = build_evidence_pairs(
        tmp_path,
        las_path=las_path,
        column_indexes=col,
        ground_indexes=ground,
        all_rejected=True,
    )
    out = tmp_path / "out"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = build_frv_membership(
        reference,
        source_result,
        scope_ref,
        authority,
        (),
        frv_configuration=configuration,
        hag_config=_hag_config(),
        geometry_config=_geometry_config(),
        wire_config=_wire_config(),
        scope_policy=default_frv_scope_policy_config(),
        membership_policy=default_frv_membership_policy_config(),
        output_dir=out,
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE
    assert result.membership_artifact is None
    assert result.failure_records == ()


def test_scope_evidence_failure_is_blocked(tmp_path: Path) -> None:
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
    # Corrupt the A1 NPZ SHA so component verification fails.
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
    out = tmp_path / "out"
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
        output_dir=out,
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.COVERAGE_BLOCKED
    assert result.membership_artifact is None
    assert any(
        failure.field_name == "scope_evidence" for failure in result.failure_records
    )


# ---------------------------------------------------------------------------
# DC-002: no residual blocked-empty-manifest bypass
# ---------------------------------------------------------------------------


def test_scope_evidence_failure_without_row_identity_fails_closed(
    tmp_path: Path,
) -> None:
    """DC-002-B: invalid scope evidence with no determinable canonical row
    identity raises the formal fail-closed error and never publishes an
    artifact or output directory."""

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
    # Point the V1 NPZ at a missing file: neither V1 nor V2 arrays can load, so
    # no canonical/required row identity can be established.
    bad_ref = FrvScopeEvidenceReference(
        a1_npz_path=scope_ref.a1_npz_path,
        a1_json_path=scope_ref.a1_json_path,
        v1_npz_path=scope_ref.v1_npz_path.parent / "missing_v1.npz",
        v1_json_path=scope_ref.v1_json_path,
        v2_npz_path=scope_ref.v2_npz_path,
        v2_json_path=scope_ref.v2_json_path,
        a1_npz_component=scope_ref.a1_npz_component,
        a1_json_component=scope_ref.a1_json_component,
        v1_npz_component=scope_ref.v1_npz_component,
        v1_json_component=scope_ref.v1_json_component,
        v2_npz_component=scope_ref.v2_npz_component,
        v2_json_component=scope_ref.v2_json_component,
    )
    out = tmp_path / "out"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(FrvMembershipFailClosedError):
        build_frv_membership(
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
            output_dir=out,
            scratch_dir=scratch,
            source_chunk_size=40,
        )
    assert out.exists() is False


def test_no_reachable_blocked_empty_manifest_or_new_bypass() -> None:
    """DC-002-C: no reachable ``_blocked_empty_manifest`` exists and no
    ``__new__`` path bypasses the coverage manifest validator or the membership
    result constructor/state-completeness validator."""

    from algorithms.vegetation import frv_membership as module

    assert not hasattr(module, "_blocked_empty_manifest")
    source = inspect.getsource(module)
    assert "FRVScopeCoverageManifest.__new__" not in source
    blocked_src = inspect.getsource(module._coverage_blocked_result)
    assert "FrvMembershipFailClosedError" in blocked_src
    assert "validate_membership_result_state_completeness" in blocked_src
    construct_src = inspect.getsource(module._construct_membership_result)
    assert "FrvMembershipExecutionResult(**fields)" in construct_src


# ---------------------------------------------------------------------------
# CANDIDATE-003: formal wire point-relation enum
# ---------------------------------------------------------------------------


class _FakeWireSpool:
    """Minimal spooled wire evidence for the ``_wire_clear`` truth table."""

    def __init__(self, state: int, relation: int) -> None:
        self.wire_state = np.asarray([state], dtype=np.uint8)
        self.wire_relation = np.asarray([relation], dtype=np.uint8)


@pytest.mark.parametrize(
    "relation_value", [int(member) for member in FrvWirePointRelationState]
)
def test_wire_clear_formal_point_relation_truth_table(relation_value: int) -> None:
    """CANDIDATE-003: every formal point-relation member is executable without
    AttributeError and matches the frozen ``wire_clear`` truth table."""

    # WIRE_RELATION_VALID + OUTSIDE_INFLUENCE_DOMAIN is clear; every in-domain
    # relation is NOT clear (wire never produces exclusion in this version).
    assert _wire_clear(
        _FakeWireSpool(
            int(FrvWireProviderState.WIRE_RELATION_VALID), relation_value
        ),
        0,
    ) is (relation_value == int(FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN))
    # WIRE_NOT_APPLICABLE_PROVEN is always clear regardless of relation.
    assert _wire_clear(
        _FakeWireSpool(
            int(FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN), relation_value
        ),
        0,
    ) is True
    # Provider-invalid / relation-invalid states are never clear.
    for invalid_state in (
        0,
        int(FrvWireProviderState.WIRE_PROVIDER_MISSING),
        int(FrvWireProviderState.WIRE_PROVIDER_INVALID),
    ):
        assert _wire_clear(_FakeWireSpool(invalid_state, relation_value), 0) is False


def test_membership_consumer_path_wire_relation_valid_outside(
    tmp_path: Path,
) -> None:
    """CANDIDATE-003: a real membership consumer run with WIRE_RELATION_VALID +
    OUTSIDE_INFLUENCE_DOMAIN points executes to READY without AttributeError."""

    las_path = tmp_path / "s.las"
    column_indexes, ground_indexes = write_scene_las(las_path)
    hag_config = _hag_config()
    geometry_config = _geometry_config()
    wire_config = _wire_config()
    source_config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    configuration = _production_configuration(
        source_config, hag_config, geometry_config, wire_config
    )
    reference = _source_reference(las_path, source_config, configuration)
    with load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=source_config,
        chunk_size=40,
    ) as session:
        list(session.iter_chunks())
        source_result = session.finalize()
    authority, _ = _build_authority(
        tmp_path, source_result, authority_mode=FrvWireAuthorityMode.W1_MODEL_SET
    )
    scope_ref = build_evidence_pairs(
        tmp_path,
        las_path=las_path,
        column_indexes=column_indexes,
        ground_indexes=ground_indexes,
    )
    out = tmp_path / "out"
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
        output_dir=out,
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.READY_FOR_V4
    assert result.wire_proof_digest is not None
    # The W1 authority carries a real wire relation domain, so the membership
    # decision path consumed WIRE_RELATION_VALID outside-domain points.
    assert result.decision_evidence_artifact is not None


# ---------------------------------------------------------------------------
# REVIEWED_FIXED_POLICY formal membership authority
# ---------------------------------------------------------------------------


def _calibration_candidate_configuration(
    source_config, hag_config, geometry_config, wire_config
) -> frv.FrvConfiguration:
    base = _configuration().to_digest_dict()
    base["scan_provider_config_digest"] = source_config.provider_config_digest
    base["hag_provider_interface_version"] = hag_config.interface_version
    base["hag_provider_config_digest"] = hag_config.provider_config_digest
    base["geometry_provider_interface_version"] = geometry_config.interface_version
    base["geometry_provider_config_digest"] = geometry_config.provider_config_digest
    base["wire_provider_interface_version"] = wire_config.interface_version
    base["wire_provider_config_digest"] = wire_config.provider_config_digest
    base["scope_policy_digest"] = default_frv_scope_policy_config().policy_digest
    base["feature_policy_digest"] = default_frv_membership_policy_config().policy_digest
    base["configuration_purpose"] = frv.FrvConfigurationPurpose.CALIBRATION_CANDIDATE
    base["configuration_is_runtime_complete"] = False
    base["formal_completeness_allowed"] = False
    base["membership_artifact_publication_allowed"] = False
    base["calibration_record_digest"] = None
    base["configuration_digest"] = frv.frv_digest(base)
    return frv.FrvConfiguration.from_dict(base)


def test_reviewed_membership_executes_and_publishes(tmp_path: Path) -> None:
    """REVIEWED_FIXED_POLICY runs the full formal chain and publishes READY.

    The artifact's configuration purpose is derived from the validated
    configuration, never from a caller assertion.
    """
    result = run_membership_ready(tmp_path, reviewed=True)[0]
    assert result.execution_state is FrvMembershipExecutionState.READY_FOR_V4
    artifact = result.membership_artifact
    assert artifact is not None
    assert (
        artifact.configuration_purpose
        is frv.FrvConfigurationPurpose.REVIEWED_FIXED_POLICY
    )
    assert result.artifact_binding is not None
    assert result.artifact_binding.semantic_artifact_id == artifact.artifact_id
    assert artifact.manifest_digest == artifact.computed_digest()
    assert artifact.artifact_id == artifact.computed_artifact_id()
    # Reload keeps the propagated authority.
    reloaded = load_frv_membership_result(result.output_reference.root_path)
    assert (
        reloaded.membership_artifact.configuration_purpose
        is frv.FrvConfigurationPurpose.REVIEWED_FIXED_POLICY
    )
    # Conservation stays exact for the reviewed path.
    five = (
        artifact.verified_vegetation_candidate_count
        + artifact.excluded_non_candidate_count
        + artifact.review_required_membership_count
        + artifact.manual_include_count
        + artifact.manual_exclude_count
    )
    assert five == artifact.evaluated_point_count
    assert result.coverage_manifest.routing_complete is True
    assert result.result_digest == result.computed_digest()


def test_calibration_candidate_cannot_formally_publish(tmp_path: Path) -> None:
    """CALIBRATION_CANDIDATE still cannot enter formal publication."""
    chain = run_provider_chain(tmp_path, source_chunk_size=40)
    las_path = chain[0]
    reference = chain[3]
    source_result = chain[6]
    authority = chain[10]
    hag_config = _hag_config()
    geometry_config = _geometry_config()
    wire_config = _wire_config()
    configuration = _calibration_candidate_configuration(
        chain[4], hag_config, geometry_config, wire_config
    )
    scope_ref = build_evidence_pairs(
        tmp_path, las_path=las_path, column_indexes=chain[1], ground_indexes=chain[2]
    )
    out = tmp_path / "out"
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
        output_dir=out,
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.COVERAGE_BLOCKED
    assert result.membership_artifact is None
    phases = {failure.phase.value for failure in result.failure_records}
    assert "production_policy" in phases
    assert out.exists() is False


def test_reviewed_policy_mismatch_rejects(tmp_path: Path) -> None:
    """A policy mismatch rejects the reviewed formal gate (fail closed)."""
    chain = run_provider_chain(tmp_path, source_chunk_size=40)
    las_path = chain[0]
    reference = chain[3]
    source_result = chain[6]
    authority = chain[10]
    hag_config = _hag_config()
    geometry_config = _geometry_config()
    wire_config = _wire_config()
    configuration = _reviewed_configuration(
        chain[4], hag_config, geometry_config, wire_config
    )
    base = configuration.to_digest_dict()
    base["feature_policy_digest"] = "f" * 64
    base["configuration_digest"] = frv.frv_digest(base)
    mismatched = frv.FrvConfiguration.from_dict(base)
    scope_ref = build_evidence_pairs(
        tmp_path, las_path=las_path, column_indexes=chain[1], ground_indexes=chain[2]
    )
    out = tmp_path / "out"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = build_frv_membership(
        reference,
        source_result,
        scope_ref,
        authority,
        (),
        frv_configuration=mismatched,
        hag_config=hag_config,
        geometry_config=geometry_config,
        wire_config=wire_config,
        scope_policy=default_frv_scope_policy_config(),
        membership_policy=default_frv_membership_policy_config(),
        output_dir=out,
        scratch_dir=scratch,
        source_chunk_size=40,
    )
    assert result.execution_state is FrvMembershipExecutionState.COVERAGE_BLOCKED
    assert result.membership_artifact is None
    assert out.exists() is False


def test_reviewed_and_production_decisions_numerically_identical(tmp_path: Path) -> None:
    """The reviewed path produces identical membership decisions to production.

    The same scene, policy, provider, and source data must yield the same
    per-point membership state, routing, and conservation, differing only in the
    artifact's authority/provenance fields.
    """
    from algorithms.vegetation.v4_inputs import _read_artifact_rows

    reviewed_root = tmp_path / "reviewed"
    production_root = tmp_path / "production"
    reviewed_root.mkdir()
    production_root.mkdir()
    reviewed = run_membership_ready(reviewed_root, reviewed=True)[0]
    production = run_membership_ready(production_root)[0]
    assert reviewed.execution_state is FrvMembershipExecutionState.READY_FOR_V4
    assert production.execution_state is FrvMembershipExecutionState.READY_FOR_V4

    reviewed_payload, _ = _read_artifact_rows(reviewed)
    production_payload, _ = _read_artifact_rows(production)
    assert set(reviewed_payload) == set(production_payload)
    decision_fields = (
        "membership_state",
        "reason_flags",
        "input_quality_flags",
        "hag_evidence_state",
        "segmentation_hag_readiness",
    )
    for index in reviewed_payload:
        for field in decision_fields:
            assert int(reviewed_payload[index][field]) == int(
                production_payload[index][field]
            ), (index, field)

    for name in (
        "verified_vegetation_candidate_count",
        "excluded_non_candidate_count",
        "review_required_membership_count",
        "manual_include_count",
        "manual_exclude_count",
        "unique_in_scope_point_count",
        "evaluated_point_count",
        "canonical_inventory_count",
        "required_inventory_count",
        "approved_hard_rejection_count",
    ):
        assert getattr(reviewed.membership_artifact, name) == getattr(
            production.membership_artifact, name
        ), name
    assert reviewed.membership_artifact.manifest_digest != (
        production.membership_artifact.manifest_digest
    )


def test_reviewed_execution_is_deterministic(tmp_path: Path) -> None:
    """Two identical reviewed executions produce identical decisions."""
    from algorithms.vegetation.v4_inputs import _read_artifact_rows

    first_root = tmp_path / "a"
    second_root = tmp_path / "b"
    first_root.mkdir()
    second_root.mkdir()
    first = run_membership_ready(first_root, reviewed=True)[0]
    second = run_membership_ready(second_root, reviewed=True)[0]
    first_payload, _ = _read_artifact_rows(first)
    second_payload, _ = _read_artifact_rows(second)
    for index in first_payload:
        assert int(first_payload[index]["membership_state"]) == int(
            second_payload[index]["membership_state"]
        )
