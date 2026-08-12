from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import laspy
import numpy as np
import pytest

from core.contracts.frv import frv_digest
from core.contracts.project import ProjectSourceReference
from core.contracts.project import SourceIdentityBinding
from tests.contracts.test_project_contracts import _project, _source_identity


@pytest.fixture
def tiny_project():
    return _project(SourceIdentityBinding.from_source_identity(_source_identity()))


@pytest.fixture
def deterministic_source_metadata():
    return {
        "policy": "REFERENCE_ORIGINAL_SOURCE_READ_ONLY",
        "current_resolved_source_path": "source/line.las",
        "file_size_bytes": 7,
        "source_file_sha256": "a" * 64,
        "point_count": 10,
        "bounds_min_xyz": [0.0, 0.0, 0.0],
        "bounds_max_xyz": [100.0, 100.0, 50.0],
        "header": {"format": "synthetic", "point_record_length": 1},
        "source_available_at_save": False,
        "recorded_at": "2026-08-12T00:00:00Z",
    }


def _write_tiny_las(path: Path, *, interior_z: float = 5.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    points = laspy.LasData(header)
    points.x = np.array([0.0, 100.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    points.y = np.array([0.0, 100.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0])
    points.z = np.array([0.0, 50.0, interior_z, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    points.write(path)


@pytest.fixture
def tiny_las_project(tiny_project, deterministic_source_metadata):
    """Build a P1 project bound to a tiny header-compatible LAS fixture."""

    def build(project_root: Path):
        source_path = project_root / "source" / "line.las"
        _write_tiny_las(source_path)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        previous_identity = tiny_project.source_reference.source_identity
        content = previous_identity.to_content_digest_dict()
        content["source_file_sha256"] = source_sha256
        content_digest = frv_digest(content)
        record = previous_identity.to_record_digest_dict()
        record["source_file_sha256"] = source_sha256
        record["source_content_identity_digest"] = content_digest
        identity = replace(
            previous_identity,
            source_file_sha256=source_sha256,
            source_content_identity_digest=content_digest,
            source_identity_record_digest=frv_digest(record),
        )
        binding = SourceIdentityBinding.from_source_identity(identity)
        source_reference = ProjectSourceReference(
            schema_version=tiny_project.source_reference.schema_version,
            source_identity=identity,
            coordinate_context=binding.coordinate_context,
        )
        project = replace(tiny_project, source_reference=source_reference)
        metadata = {
            **deterministic_source_metadata,
            "file_size_bytes": source_path.stat().st_size,
            "source_file_sha256": source_sha256,
            "source_available_at_save": True,
        }
        return project, metadata, source_path

    return build


@pytest.fixture
def rewrite_tiny_las():
    def rewrite(path: Path, *, interior_z: float = 13.0) -> None:
        before = path.stat()
        _write_tiny_las(path, interior_z=interior_z)
        # Make the cache-invalidation test deterministic even on a filesystem
        # whose ordinary timestamp resolution is coarser than this test.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    return rewrite
