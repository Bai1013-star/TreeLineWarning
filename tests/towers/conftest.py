"""Shared fixtures/helpers for the V5A2/V5A3 tower pipeline tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import laspy
import numpy as np
import pytest
from core.contracts.frv import frv_digest
from core.contracts.project import (
    PROJECT_SCHEMA_VERSION,
    PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
    ProjectContract,
    ProjectSourceReference,
    SourceIdentityBinding,
    ViewerReference,
)

from tests.contracts.test_project_contracts import PROJECT_ID, TIMESTAMP, _source_identity


def _content_bounds(points: np.ndarray) -> tuple[list[float], list[float]]:
    return [float(value) for value in points.min(axis=0)], [
        float(value) for value in points.max(axis=0)
    ]


def _las_header_bounds(path: Path) -> tuple[list[float], list[float]]:
    with laspy.open(path) as las:
        header = las.header
    return (
        [float(value) for value in header.mins],
        [float(value) for value in header.maxs],
    )


def write_ground_evidence_las(
    path: Path,
    points: np.ndarray,
    *,
    random_seed: int = 7,
) -> np.ndarray:
    """Write a coarse ground/HAG evidence LAS with the FAST representative dims.

    Tower (non-ground) representatives are sampled densely so each tower column
    keeps clean vertical PCA neighbourhoods; ground representatives are sampled
    separately.  Emits ``classification``, ``HeightAboveGround``,
    ``representative_id``, ``source_original_index`` and ``raw_count`` so
    ``algorithms.poles.evidence.load_ground_evidence`` can consume it.  Returns
    the representative rows (n, 5): x, y, z, source_original_index, classification.
    """
    rs = np.random.RandomState(random_seed)
    tower_rows = np.flatnonzero(points[:, 2] >= 0.5)
    ground_rows = np.flatnonzero(points[:, 2] < 0.5)
    selected_rows = np.concatenate((tower_rows[::2], ground_rows[::5]))
    selected = points[selected_rows]
    classification = np.where(selected[:, 2] < 0.5, 2, 0).astype(np.uint8)
    source_original_index = selected_rows.astype(np.int64)
    height_above_ground = np.maximum(selected[:, 2] - 0.0, 0.0).astype(np.float32)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    for name, dtype in (
        ("HeightAboveGround", "<f4"),
        ("representative_id", "<u4"),
        ("source_original_index", "<i8"),
        ("raw_count", "<u4"),
    ):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.dtype(dtype)))
    las = laspy.LasData(header)
    las.x = selected[:, 0]
    las.y = selected[:, 1]
    las.z = selected[:, 2]
    las.classification = classification
    las.HeightAboveGround = height_above_ground
    las.representative_id = np.arange(len(selected), dtype=np.uint32)
    las.source_original_index = source_original_index
    las.raw_count = np.ones(len(selected), dtype=np.uint32) * 4
    las.red = rs.randint(0, 256, len(selected), dtype=np.uint16)
    las.green = rs.randint(0, 256, len(selected), dtype=np.uint16)
    las.blue = rs.randint(0, 256, len(selected), dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)
    return np.column_stack(
        (selected[:, 0], selected[:, 1], selected[:, 2], source_original_index, classification)
    )


def source_identity_for_las(
    path: Path, points: np.ndarray, *, source_id: str = "tower-source"
) -> tuple[object, str]:
    """Build a digest-consistent SourceIdentity bound to an existing LAS file."""
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    # The authoritative source bounds are the LAS header bounds (scaled), which
    # is what the session's fast availability gate compares against.
    min_xyz, max_xyz = _las_header_bounds(path)
    previous = _source_identity()
    content = previous.to_content_digest_dict()
    content["source_id"] = source_id
    content["source_file_sha256"] = sha256
    content["source_point_count"] = int(len(points))
    content["source_bounds_min_xyz"] = min_xyz
    content["source_bounds_max_xyz"] = max_xyz
    content_digest = frv_digest(content)
    record = previous.to_record_digest_dict()
    record["source_id"] = source_id
    record["source_file_sha256"] = sha256
    record["source_point_count"] = int(len(points))
    record["source_bounds_min_xyz"] = min_xyz
    record["source_bounds_max_xyz"] = max_xyz
    record["recorded_source_path"] = str(path)
    record["resolved_access_path"] = str(path)
    record["source_content_identity_digest"] = content_digest
    identity = replace(
        previous,
        source_id=source_id,
        source_file_sha256=sha256,
        source_point_count=int(len(points)),
        source_bounds_min_xyz=tuple(min_xyz),
        source_bounds_max_xyz=tuple(max_xyz),
        source_content_identity_digest=content_digest,
        source_identity_record_digest=frv_digest(record),
        resolved_access_path=str(path),
        recorded_source_path=str(path),
    )
    return identity, sha256


def write_tower_las(
    path: Path,
    *,
    tower_xyz: tuple[float, float, float] = (10.0, 20.0, 0.0),
    tower_height_m: float = 30.0,
    extra_tower_xyz: tuple[float, float, float] | None = (45.0, 50.0, 0.0),
    ground_count: int = 400,
    scatter_m: float = 0.03,
    random_seed: int = 0,
) -> np.ndarray:
    """Write a synthetic RAW LAS with vertical tower column(s) plus scattered ground."""
    rs = np.random.RandomState(random_seed)
    towers: list[np.ndarray] = []
    for txyz in (tower_xyz, extra_tower_xyz):
        if txyz is None:
            continue
        zs = np.arange(0.0, tower_height_m + 0.01, 0.5)
        towers.append(
            np.column_stack(
                (
                    txyz[0] + rs.normal(0.0, scatter_m, len(zs)),
                    txyz[1] + rs.normal(0.0, scatter_m, len(zs)),
                    txyz[2] + zs,
                )
            )
        )
    # Ground is kept away from every tower column so the tower-base
    # representatives keep clear vertical PCA neighbourhoods (the detector input
    # is cleaner and the min-HAG ground-contact gate stays honest).
    ground_xy = np.column_stack(
        (rs.uniform(0.0, 60.0, ground_count * 2), rs.uniform(0.0, 60.0, ground_count * 2))
    )
    tower_centers = [tower_xyz]
    if extra_tower_xyz is not None:
        tower_centers.append(extra_tower_xyz)
    keep = np.ones(len(ground_xy), dtype=bool)
    for center in tower_centers:
        keep &= np.linalg.norm(ground_xy - np.asarray(center[:2])[None, :], axis=1) >= 3.0
    ground_xy = ground_xy[keep][:ground_count]
    ground = np.column_stack(
        (
            ground_xy[:, 0],
            ground_xy[:, 1],
            np.full(len(ground_xy), tower_xyz[2] - 0.1)
            + rs.normal(0.0, 0.05, len(ground_xy)),
        )
    )
    points = np.vstack((*towers, ground))
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)
    return points


def project_for_source(
    source_path: Path,
    points: np.ndarray,
    *,
    project_id: str = PROJECT_ID,
    source_id: str = "tower-source",
) -> tuple[ProjectContract, SourceIdentityBinding, dict]:
    """Build a ProjectContract + binding + access metadata for a synthetic source."""
    identity, sha256 = source_identity_for_las(source_path, points, source_id=source_id)
    binding = SourceIdentityBinding.from_source_identity(identity)
    source_reference = ProjectSourceReference(
        schema_version=PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
        source_identity=identity,
        coordinate_context=binding.coordinate_context,
    )
    project = ProjectContract(
        schema_version=PROJECT_SCHEMA_VERSION,
        project_id=project_id,
        project_name="Tower line A",
        created_at=TIMESTAMP,
        updated_at=TIMESTAMP,
        source_reference=source_reference,
        viewer=ViewerReference.not_built(),
        tower_registry_reference="towers/towers.json",
        span_registry_reference="spans/spans.json",
        run_index_reference="runs/run-index.json",
        next_tower_sequence=1,
        next_span_sequence=1,
    )
    min_xyz, max_xyz = _las_header_bounds(source_path)
    metadata = {
        "policy": "REFERENCE_ORIGINAL_SOURCE_READ_ONLY",
        "current_resolved_source_path": str(source_path),
        "file_size_bytes": source_path.stat().st_size,
        "source_file_sha256": sha256,
        "point_count": int(len(points)),
        "bounds_min_xyz": min_xyz,
        "bounds_max_xyz": max_xyz,
        "header": {"format": "synthetic"},
        "source_available_at_save": True,
        "recorded_at": TIMESTAMP,
    }
    return project, binding, metadata


def build_ground_evidence(tmp_path: Path, points: np.ndarray):
    """Build a GroundEvidence from a synthetic RAW point cloud via the coarse LAS."""
    from algorithms.poles.evidence import load_ground_evidence
    from core.config.fast_baseline import FastBaselineConfig

    path = tmp_path / "ground_evidence.las"
    write_ground_evidence_las(path, points)
    return load_ground_evidence(path, FastBaselineConfig(), origin_xyz=(0.0, 0.0, 0.0))


@pytest.fixture
def tower_las_project(tmp_path):
    """Build a writable project bound to a synthetic tower RAW LAS."""

    def build():
        source_path = tmp_path / "source" / "src.las"
        points = write_tower_las(source_path)
        project, binding, metadata = project_for_source(source_path, points)
        return project, binding, metadata, points

    return build
