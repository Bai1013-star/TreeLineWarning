from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
from algorithms.folder_pipeline.user_delivery_excel import (
    DELIVERY_COLUMNS,
    write_delivery_from_prediction,
)
from desktop.app import DesktopApi
from openpyxl import load_workbook


class _SaveWindow:
    def __init__(self, *selections) -> None:
        self.selections = list(selections)
        self.calls = []

    def create_file_dialog(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.selections.pop(0)


class _Workflow:
    def __init__(self, source: Path) -> None:
        self.source = source

    def export_delivery(self) -> Path:
        return self.source

    def export_html_report(self) -> Path:
        return self.source


def test_native_excel_save_supports_unicode_path_and_atomic_copy(tmp_path: Path) -> None:
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "retained", "observation_date": "2026-09-28",
          "dbh_cm": 22.8}],
        {"science_run_id": "fixture"},
    )
    source = write_delivery_from_prediction(internal)
    target = tmp_path / "中文下载" / "线路结果"
    bridge = DesktopApi(SimpleNamespace(local_app=SimpleNamespace(workflow=_Workflow(source))))
    bridge._window = _SaveWindow([str(target)])

    result = bridge.export_excel()

    saved = Path(result["path"])
    assert result["ok"] is True
    assert saved.name == "线路结果.xlsx"
    assert saved.read_bytes() == source.read_bytes()
    assert not list(saved.parent.glob("*.tmp"))
    workbook = load_workbook(saved, read_only=True)
    try:
        assert workbook.sheetnames == ["树木风险结果"]
        assert tuple(next(workbook.active.values)) == DELIVERY_COLUMNS
        assert workbook.active.max_row == 2
    finally:
        workbook.close()


def test_native_html_save_uses_generated_final_report(tmp_path: Path) -> None:
    source = tmp_path / "generated.html"
    source.write_text("<!doctype html><title>final</title>", encoding="utf-8")
    target = tmp_path / "report.html"
    bridge = DesktopApi(SimpleNamespace(local_app=SimpleNamespace(workflow=_Workflow(source))))
    bridge._window = _SaveWindow([str(target)])

    result = bridge.export_html()

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_native_save_accepts_string_and_tuple_paths_and_remembers_directory(tmp_path: Path) -> None:
    source = tmp_path / "generated.html"
    source.write_text("<!doctype html><title>current</title>", encoding="utf-8")
    first = tmp_path / "中文 空格" / "当前报告.html"
    second = first.parent / "下一份.html"
    backend = SimpleNamespace(
        project_root=tmp_path,
        local_app=SimpleNamespace(workflow=_Workflow(source)),
    )
    bridge = DesktopApi(backend)
    window = _SaveWindow(str(first), (str(second),))
    bridge._window = window

    assert Path(bridge.export_html()["path"]) == first.resolve()
    assert first.read_bytes() == source.read_bytes()
    assert window.calls[0]["directory"] == str(tmp_path)
    assert Path(bridge.export_html()["path"]) == second.resolve()
    assert window.calls[1]["directory"] == str(first.parent.resolve())
    assert second.read_bytes() == source.read_bytes()
    assert not (tmp_path / "C.html").exists()


def test_native_save_cancel_does_not_write_or_change_remembered_directory(tmp_path: Path) -> None:
    source = tmp_path / "generated.html"
    source.write_text("content", encoding="utf-8")
    bridge = DesktopApi(SimpleNamespace(
        project_root=tmp_path,
        local_app=SimpleNamespace(workflow=_Workflow(source)),
    ))
    target = tmp_path / "中文 空格" / "existing.html"
    target.parent.mkdir()
    target.write_text("keep", encoding="utf-8")
    bridge._window = _SaveWindow(None, (), str(target))

    assert bridge.export_html() == {"ok": False, "cancelled": True}
    assert bridge.export_html() == {"ok": False, "cancelled": True}
    assert bridge._last_export_dir is None
    assert target.read_text(encoding="utf-8") == "keep"
    assert bridge.export_html()["ok"] is True
    assert target.read_text(encoding="utf-8") == "content"


def test_dialog_path_keeps_full_windows_drive_unicode_and_spaces() -> None:
    for path in (r"C:\Users\Lenovo\中文 空格\结果.xlsx", r"E:\项目 三\报告.html"):
        assert DesktopApi._dialog_path(path) == Path(path)
        assert DesktopApi._dialog_path((path,)) == Path(path)


def test_native_folder_dialog_keeps_full_string_path(tmp_path: Path) -> None:
    folder = tmp_path / "中文 项目"
    bridge = DesktopApi(SimpleNamespace(project_root=tmp_path))
    bridge._window = _SaveWindow(str(folder))
    assert bridge.choose_project_location() == str(folder)


def test_excel_and_html_exports_share_last_chosen_directory(tmp_path: Path) -> None:
    excel = tmp_path / "generated.xlsx"
    html = tmp_path / "generated.html"
    excel.write_bytes(b"xlsx source")
    html.write_text("<!doctype html>", encoding="utf-8")
    workflow = SimpleNamespace(export_delivery=lambda: excel, export_html_report=lambda: html)
    bridge = DesktopApi(SimpleNamespace(
        project_root=tmp_path,
        local_app=SimpleNamespace(workflow=workflow),
    ))
    first = tmp_path / "E盘 中文 空格" / "结果.xlsx"
    second = first.parent / "报告.html"
    window = _SaveWindow(str(first), str(second))
    bridge._window = window

    assert Path(bridge.export_excel()["path"]) == first.resolve()
    assert Path(bridge.export_html()["path"]) == second.resolve()
    assert window.calls[1]["directory"] == str(first.parent.resolve())
    assert first.read_bytes() == b"xlsx source"
    assert second.read_text(encoding="utf-8") == "<!doctype html>"
