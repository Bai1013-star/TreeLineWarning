"""Optional CRS failures must not erase independently readable LAS evidence."""
import laspy
import numpy as np
import pytest
from core.stage_a.capability import inspect_point_cloud, CapabilityState
from desktop.project_management import create_empty_project, import_pointcloud
from core.stage_a import load_project


@pytest.mark.parametrize('error', [ImportError('CRS unavailable'), OSError('CRS DLL unavailable')])
def test_import_preserves_header_when_optional_crs_is_unavailable(tmp_path, monkeypatch, error):
    cloud = laspy.LasData(laspy.LasHeader(point_format=3, version='1.2'))
    cloud.x, cloud.y, cloud.z = np.arange(30), np.arange(30), np.arange(30)
    path = tmp_path / 'original.las'
    cloud.write(path)
    def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr(laspy.LasHeader, 'parse_crs', unavailable)
    report = inspect_point_cloud(path)
    assert report.details['point_count'] == 30
    assert report.capabilities['XYZ'] == CapabilityState.AVAILABLE
    assert report.capabilities['CRS'] == CapabilityState.UNKNOWN
    root = tmp_path / 'project'
    create_empty_project(root.parent, root.name)
    import_pointcloud(root, path)
    scan = load_project(root).scans[0]
    assert scan.capability_report['details']['point_count'] == 30
    assert len(scan.source_identity.sha256) == 64
