from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from desktop.app import DesktopApi


class _SaveWindow:
    def __init__(self, target: Path) -> None:
        self.target = target

    def create_file_dialog(self, *args, **kwargs):
        return [str(self.target)]


class _Workflow:
    def __init__(self, source: Path) -> None:
        self.source = source

    def export_prediction(self) -> Path:
        return self.source

    def export_html_report(self) -> Path:
        return self.source


def test_native_excel_save_supports_unicode_path_and_atomic_copy(tmp_path: Path) -> None:
    source = tmp_path / "generated.xlsx"
    source.write_bytes(b"PK\x03\x04xlsx-fixture")
    target = tmp_path / "中文下载" / "线路结果"
    bridge = DesktopApi(SimpleNamespace(local_app=SimpleNamespace(workflow=_Workflow(source))))
    bridge._window = _SaveWindow(target)

    result = bridge.export_excel()

    saved = Path(result["path"])
    assert result["ok"] is True
    assert saved.name == "线路结果.xlsx"
    assert saved.read_bytes() == source.read_bytes()
    assert not list(saved.parent.glob("*.tmp"))


def test_native_html_save_uses_generated_final_report(tmp_path: Path) -> None:
    source = tmp_path / "generated.html"
    source.write_text("<!doctype html><title>final</title>", encoding="utf-8")
    target = tmp_path / "report.html"
    bridge = DesktopApi(SimpleNamespace(local_app=SimpleNamespace(workflow=_Workflow(source))))
    bridge._window = _SaveWindow(target)

    result = bridge.export_html()

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")
