from pathlib import Path

import desktop_launcher


def test_bundled_path_uses_source_checkout(monkeypatch):
    monkeypatch.delattr(desktop_launcher.sys, "_MEIPASS", raising=False)
    assert desktop_launcher.bundled_path("web", "templates", "index.html") == (
        Path(desktop_launcher.__file__).resolve().parent / "web" / "templates" / "index.html"
    )


def test_user_data_dir_uses_local_app_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert desktop_launcher.user_data_dir() == tmp_path / "TreeLineWarning"


def test_available_port_returns_a_local_port():
    port = desktop_launcher.available_port(18010)
    assert 18010 <= port < 18030
