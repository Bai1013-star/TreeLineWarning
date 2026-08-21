import json
from pathlib import Path
from types import SimpleNamespace

from desktop.app import DesktopApi, resource_root
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
    root = Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d"
    page = (root / "desktop.html").read_text(encoding="utf-8")
    script = (root / "desktop.js").read_text(encoding="utf-8")
    assert '<iframe id="viewer" src="/">' in page
    assert '<iframe id="viewer" src="about:blank"' in page
    assert 'id="newProject"' in page
    assert 'id="openProject"' in page
    assert 'id="importPointCloud"' in page
    assert 'id="exportWorkOrder"' in page
    assert "导出预测输入" in page
    assert "单木结果查看与复核" in page
    assert "/download_prediction" in script


def test_desktop_bridge_exports_required_methods_without_public_object_graph():
    bridge = DesktopApi(SimpleNamespace())
    bridge._window = object()

    required = {
        "choose_project_location",
        "choose_existing_project",
        "create_project",
        "import_point_cloud",
        "import_gis",
    }
    assert all(callable(getattr(bridge, name)) for name in required)
    assert not hasattr(bridge, "window")
    assert not hasattr(bridge, "backend")
    assert all(
        name.startswith("_") or callable(getattr(bridge, name))
        for name in dir(bridge)
    )


def test_frontend_waits_for_required_new_project_bridge_contract():
    script = (
        Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d" / "desktop.js"
    ).read_text(encoding="utf-8")

    assert "pywebviewready" in script
    assert "getNativeBridge(['choose_project_location', 'create_project'])" in script
    assert "await bridge.choose_project_location()" in script
    assert "await bridge.create_project(parent, name.trim())" in script
    assert "已取消新建项目" in script


def test_tower_picker_accepts_a_zero_distance_pointerup_as_a_click():
    """A normal pointer click has zero travel and must request local evidence."""
    script = (
        Path(__file__).resolve().parents[2]
        / "viewer_poc"
        / "giro3d"
        / "tower_picker_3d.js"
    ).read_text(encoding="utf-8")

    assert "Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4" in script
    assert "!start || start.button !== 0" in script
    assert "if (!hits.length) return;" in script


def test_tower_picker_is_true_xyz_webgl_with_orbit_and_world_picking():
    root = Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d"
    page = (root / "desktop.html").read_text(encoding="utf-8")
    shell = (root / "desktop.js").read_text(encoding="utf-8")
    viewer = (root / "tower_picker_3d.js").read_text(encoding="utf-8")

    assert '<script src="/giro3d.bundle.js"></script>' in page
    assert '<script src="/tower-picker-3d.js"></script>' in page
    assert "new this.THREE.PerspectiveCamera" in viewer
    assert "new G.MapControls" in viewer
    assert "mouseButtons.LEFT = this.THREE.MOUSE.ROTATE" in viewer
    assert "mouseButtons.RIGHT = this.THREE.MOUSE.PAN" in viewer
    assert "new this.THREE.Raycaster" in viewer
    assert "intersectObject(this.points, false)" in viewer
    assert "z: point.z + this.origin[2]" in viewer
    assert "sizeAttenuation: false" in viewer
    assert "setMarkers(towers)" in viewer
    assert "id=\"perspectiveTowerView\"" in shell
    assert "id=\"topTowerView\"" in shell
    assert "id=\"sideTowerView\"" in shell
    assert "new window.TreeLineTowerViewer3D($('tower3dCanvas')" in shell
    assert "JSON.stringify(world)" in shell


def test_main_product_copy_is_chinese_and_tree_editor_is_state_gated():
    root = Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d"
    page = (root / "desktop.html").read_text(encoding="utf-8")
    script = (root / "desktop.js").read_text(encoding="utf-8")

    for legacy_copy in (
        "Project Home",
        "Offline Contract",
        "Risk Policy",
        "Start Stage-B Batch",
        "Retry Failed",
        "Finalize Review",
        "Authoritative Folder Pipeline",
    ):
        assert legacy_copy not in page
    assert 'id="startFixture" class="small test-only hidden"' in page
    assert "if (!state.treeReviewReady && !state.finalized)" in script
    assert "loadViewer();" in script
