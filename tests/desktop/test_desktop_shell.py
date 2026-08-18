from pathlib import Path

from desktop.app import resource_root


def test_desktop_resources_and_launcher_are_repository_relative():
    root = resource_root()
    assert isinstance(root, Path)
    assert (Path(__file__).resolve().parents[2] / "desktop" / "app.py").is_file()
    assert (Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d" / "desktop.html").is_file()


def test_desktop_page_uses_existing_local_viewer_and_safe_unimplemented_actions():
    page = (Path(__file__).resolve().parents[2] / "viewer_poc" / "giro3d" / "desktop.html").read_text(
        encoding="utf-8"
    )
    assert '<iframe id="viewer" src="/">' in page
    assert "暂未实现" in page
    assert "/prediction" in page

