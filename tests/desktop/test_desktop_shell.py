from pathlib import Path

from desktop.app import resource_root


def test_desktop_resources_and_launcher_are_repository_relative():
    root = resource_root()
    assert isinstance(root, Path)
    assert (Path(__file__).resolve().parents[2] / "desktop" / "app.py").is_file()
    assert (
        Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d" / "desktop.html"
    ).is_file()


def test_desktop_page_uses_existing_local_viewer_and_product_actions():
    page = (
        Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d" / "desktop.html"
    ).read_text(encoding="utf-8")
    assert '<iframe id="viewer" src="/">' in page
    assert 'id="newProject"' in page
    assert 'id="openProject"' in page
    assert 'id="importPointCloud"' in page
    assert 'id="exportWorkOrder"' in page
    assert "导出风险工单" in page
    assert "预测胸径" in page
    assert "预测数据质量" in page
    assert "/prediction" in page
