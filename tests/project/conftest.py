from __future__ import annotations

import pytest

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
