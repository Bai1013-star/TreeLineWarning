from pathlib import Path
import laspy
import numpy as np
import pytest
from desktop.source_integrity import PointCloudIntegrityError, validate_las_extent
from desktop.tower_viewer import overview
from desktop.project_management import create_empty_project, import_pointcloud


def source(tmp_path, suffix='.las'):
    cloud = laspy.create(point_format=7, file_version='1.4')
    cloud.x = np.arange(10, dtype=float)
    cloud.y = np.arange(10, dtype=float)
    cloud.z = np.arange(10, dtype=float)
    path = tmp_path / ('原始点云' + suffix)
    cloud.write(path)
    return path


def test_valid_las_and_compressed_laz(tmp_path):
    assert validate_las_extent(source(tmp_path))['status'] == 'COMPLETE_EXTENT'
    assert validate_las_extent(source(tmp_path, '.laz')) is None


def test_truncated_partial_record_is_rejected_before_any_cache(tmp_path):
    path = source(tmp_path)
    path.write_bytes(path.read_bytes()[:-13])
    with pytest.raises(PointCloudIntegrityError, match='这不是断网'):
        overview(path, cache_dir=tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


def test_truncated_exact_records_are_not_silently_accepted(tmp_path):
    path = source(tmp_path)
    path.write_bytes(path.read_bytes()[:-36])
    with pytest.raises(PointCloudIntegrityError, match='9 条完整点记录'):
        validate_las_extent(path)


def test_failed_import_preserves_existing_project(tmp_path):
    root = Path(create_empty_project(tmp_path, '项目')['root'])
    good = source(tmp_path)
    import_pointcloud(root, good)
    files = ['desktop-project.json', 'stage-a-project.json']
    before = {name: (root/name).read_bytes() for name in files}
    bad = tmp_path / '不完整.las'
    bad.write_bytes(good.read_bytes()[:-7])
    with pytest.raises(PointCloudIntegrityError):
        import_pointcloud(root, bad)
    assert before == {name: (root/name).read_bytes() for name in files}
