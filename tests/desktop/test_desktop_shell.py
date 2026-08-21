import json
from pathlib import Path

from desktop.app import resource_root
from desktop.server import _strict_json_value


def test_desktop_resources_and_launcher_are_repository_relative():
    root = resource_root()
    assert isinstance(root, Path)
    assert (Path(__file__).resolve().parents[2] / "desktop" / "app.py").is_file()
    assert (
        Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d" / "desktop.html"
    ).is_file()


def test_launcher_uses_backend_landing_route_for_each_project_format():
    source = (
        Path(__file__).resolve().parents[2] / "desktop" / "app.py"
    ).read_text(encoding="utf-8")

    assert 'url + "desktop.html"' not in source
    assert "\n            url,\n            width=1600," in source


def test_desktop_json_boundary_normalizes_non_finite_diagnostics():
    payload = _strict_json_value(
        {"finite": 4.5, "diagnostics": [float("inf"), -float("inf"), float("nan")]}
    )

    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == {
        "finite": 4.5,
        "diagnostics": [None, None, None],
    }


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
