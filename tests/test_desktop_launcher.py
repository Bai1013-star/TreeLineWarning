from base64 import b64encode
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


class _SaveDialogWindow:
    def __init__(self, selected_path):
        self.selected_path = selected_path
        self.calls = []

    def create_file_dialog(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.selected_path


def test_csv_template_bridge_saves_selected_path_and_adds_csv_suffix(tmp_path):
    selected_path = tmp_path / "训练示例"
    window = _SaveDialogWindow((str(selected_path),))
    bridge = desktop_launcher.DesktopCsvTemplateBridge(window)

    result = bridge.save_csv_template(b64encode("列名,数值\n".encode()).decode("ascii"))

    destination = selected_path.with_suffix(".csv")
    assert result == {"saved": True, "path": str(destination)}
    assert destination.read_text(encoding="utf-8") == "列名,数值\n"
    assert window.calls[0][1]["save_filename"] == "treeline_history_training_template.csv"


def test_csv_template_bridge_returns_cancelled_when_save_dialog_is_cancelled():
    bridge = desktop_launcher.DesktopCsvTemplateBridge(_SaveDialogWindow(None))
    assert bridge.save_csv_template(b64encode(b"example").decode("ascii")) == {
        "saved": False,
        "cancelled": True,
    }
