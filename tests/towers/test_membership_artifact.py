"""V5A3 exact RAW membership artifact tests.

Covers artifact round-trip integrity, corruption, wrong-source and NaN
rejection, the digest-bound ArtifactReference, and byte-size / member-count /
bounds validation.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from algorithms.towers.membership_artifact import (
    TOWER_MEMBERSHIP_ROLE,
    MembershipArtifactIntegrityError,
    read_membership_artifact,
    write_membership_artifact,
)
from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
from algorithms.towers.segmentation import (
    TowerLocalSegmentationConfig,
    segment_tower_membership,
)
from core.contracts.project import ArtifactReference, SourceIdentityBinding

from tests.towers.conftest import source_identity_for_las, write_tower_las


def _build_membership(tmp_path: Path):
    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=None)
    identity, _ = source_identity_for_las(src, points)
    binding = SourceIdentityBinding.from_source_identity(identity)
    cache = RawTileCache.prepare(
        src, tmp_path / "cache", source_binding=binding, config=RawTileCacheConfig()
    )
    query = cache.query((10.0, 20.0, 0.0), xy_radius_m=3.5)
    result = segment_tower_membership(query, binding, TowerLocalSegmentationConfig())
    assert result.success, result.failure_reason
    return result.membership, binding


def test_artifact_round_trip_preserves_exact_membership(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = write_membership_artifact(project, membership)
    assert record.relative_path.startswith("tower-memberships/")
    assert record.relative_path.endswith(".npz")
    assert record.sha256 == record.relative_path.rsplit("/", 1)[1][:-4]
    assert record.member_count == membership.member_count
    assert record.height_m == membership.height_m
    assert record.byte_size == (project / record.relative_path).stat().st_size

    reference = record.to_artifact_reference()
    assert isinstance(reference, ArtifactReference)
    assert reference.role == TOWER_MEMBERSHIP_ROLE
    assert reference.sha256 == record.sha256
    assert reference.byte_size == record.byte_size

    restored = read_membership_artifact(project, reference, source_binding=binding)
    assert restored.member_count == membership.member_count
    assert np.array_equal(restored.source_index, membership.source_index)
    assert np.array_equal(restored.xyz_absolute_m, membership.xyz_absolute_m)
    assert restored.membership_digest() == membership.membership_digest()
    assert restored.member_bounds_absolute_m == membership.member_bounds_absolute_m
    assert restored.base_xyz_absolute_m == membership.base_xyz_absolute_m
    assert restored.height_m == membership.height_m


def test_artifact_write_is_content_addressed(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    first = write_membership_artifact(project, membership)
    second = write_membership_artifact(project, membership)
    assert first.relative_path == second.relative_path
    assert first.sha256 == second.sha256
    assert (project / first.relative_path).is_file()


def test_corrupt_artifact_fails(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = write_membership_artifact(project, membership)
    path = project / record.relative_path
    data = path.read_bytes()
    path.write_bytes(data[:-10] + bytes([data[-10] ^ 0xFF]) + data[-9:])
    with pytest.raises(MembershipArtifactIntegrityError):
        read_membership_artifact(project, record.to_artifact_reference(), source_binding=binding)


def test_wrong_source_fails(tmp_path):
    membership, binding = _build_membership(tmp_path)
    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=None)
    project = tmp_path / "project"
    project.mkdir()
    record = write_membership_artifact(project, membership)
    other_identity, _ = source_identity_for_las(src, points, source_id="other-source")
    other_binding = SourceIdentityBinding.from_source_identity(other_identity)
    with pytest.raises(MembershipArtifactIntegrityError):
        read_membership_artifact(
            project, record.to_artifact_reference(), source_binding=other_binding
        )


def test_nan_points_fail(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    from algorithms.towers import membership_artifact as ma

    raw = ma._encode_artifact(membership)
    payload = dict(np.load(io.BytesIO(raw), allow_pickle=False))
    payload["xyz_absolute_m"] = payload["xyz_absolute_m"].copy()
    payload["xyz_absolute_m"][0, 0] = np.nan
    buffer = io.BytesIO()
    np.savez(buffer, **payload)
    bad_path = project / "tower-memberships" / ("0" * 64 + ".npz")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(buffer.getvalue())
    with pytest.raises(MembershipArtifactIntegrityError):
        read_membership_artifact(project, str(bad_path), source_binding=binding)


def test_missing_artifact_fails(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = write_membership_artifact(project, membership)
    (project / record.relative_path).unlink()
    with pytest.raises(MembershipArtifactIntegrityError):
        read_membership_artifact(project, record.to_artifact_reference(), source_binding=binding)


def test_wrong_segmentation_config_digest_fails(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = write_membership_artifact(project, membership)
    with pytest.raises(MembershipArtifactIntegrityError):
        read_membership_artifact(
            project,
            record.to_artifact_reference(),
            source_binding=binding,
            segmentation_config_digest="0" * 64,
        )


def test_artifact_rejects_wrong_binding_type(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = write_membership_artifact(project, membership)
    from algorithms.towers.membership_artifact import MembershipArtifactError

    with pytest.raises(MembershipArtifactError):
        read_membership_artifact(project, record.to_artifact_reference(), source_binding=object())


def test_member_count_and_bounds_recorded(tmp_path):
    membership, binding = _build_membership(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = write_membership_artifact(project, membership)
    assert record.member_count == membership.member_count
    assert record.member_bounds_absolute_m == membership.member_bounds_absolute_m
    assert record.segmentation_config_digest == membership.config_digest
    assert record.membership_digest == membership.membership_digest()
