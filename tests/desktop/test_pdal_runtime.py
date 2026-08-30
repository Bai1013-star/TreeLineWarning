from __future__ import annotations

import pytest
from desktop.pdal_runtime import dependency_health

from desktop import pdal_runtime


def test_pdal_health_probe_runs_the_real_minimal_operation_when_available() -> None:
    pytest.importorskip("pdal")
    report = dependency_health()
    assert report["found"] is True
    assert report["minimal_operation_pass"] is True


def test_frozen_runtime_does_not_use_external_native_overrides(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    external = tmp_path / "external"
    (external / "Library/bin").mkdir(parents=True)
    data = external / "Library/share/proj"
    data.mkdir(parents=True)
    (data / "proj.db").write_bytes(b"external database")
    monkeypatch.setattr(pdal_runtime.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(pdal_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setenv("TREELINE_PDAL_RUNTIME_ENV", str(external))
    monkeypatch.setenv("TREELINE_PDAL_PROJ_DATA_ROOT", str(external))
    monkeypatch.setenv("PROJ_NETWORK", "OFF")

    assert pdal_runtime.bundled_proj_data_dir() is None
    assert pdal_runtime._configure_bundled_runtime() == (None, [])
