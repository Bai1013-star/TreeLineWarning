from __future__ import annotations

import pytest
from desktop.pdal_runtime import dependency_health


def test_pdal_health_probe_runs_the_real_minimal_operation_when_available() -> None:
    pytest.importorskip("pdal")
    report = dependency_health()
    assert report["found"] is True
    assert report["minimal_operation_pass"] is True
