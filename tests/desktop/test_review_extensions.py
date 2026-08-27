from __future__ import annotations

from types import SimpleNamespace

import desktop  # noqa: F401 - package import installs the production extensions
from algorithms.folder_pipeline import html as folder_html
from desktop.review_extensions import _result_matches_editor_revision


def _editor(operation_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        session=SimpleNamespace(operations=[{} for _ in range(operation_count)])
    )


def _result(*, editor_schema: bool, operation_count: int | None) -> SimpleNamespace:
    diagnostics = {}
    if editor_schema:
        diagnostics["editor_schema"] = "treeline-stage-b-batch-editor-v1"
        diagnostics["editor_operation_count"] = operation_count
    return SimpleNamespace(
        global_reconciled=SimpleNamespace(diagnostics=diagnostics)
    )


def test_pre_editor_result_is_current_only_before_first_edit() -> None:
    result = _result(editor_schema=False, operation_count=None)

    assert _result_matches_editor_revision(_editor(0), result) is True
    assert _result_matches_editor_revision(_editor(1), result) is False


def test_post_editor_result_revision_must_match_current_edit_log() -> None:
    published = _result(editor_schema=True, operation_count=2)

    assert _result_matches_editor_revision(_editor(2), published) is True
    assert _result_matches_editor_revision(_editor(3), published) is False
    assert _result_matches_editor_revision(_editor(1), published) is False


def test_step2_injects_offline_clearance_controls_before_app_start() -> None:
    page = folder_html.step2_html()

    extension = page.index("treeline-clearance-data")
    start = page.index("window.TreeLineTreeReviewApp.start();")
    assert extension < start
    assert "显示危险点" in page
    assert "clearance-risk-filter" in page
    assert "EXACT_FINAL_CLEARANCE_NEAREST_PAIR" in page
    assert "https://" not in page
    assert "http://" not in page
