from __future__ import annotations

from pathlib import Path


def test_desktop_waits_for_background_finalize_before_reloading_workspace() -> None:
    root = Path(__file__).resolve().parents[2]
    page = (root / "viewer_poc" / "giro3d" / "desktop.html").read_text(encoding="utf-8")

    assert "async function waitForFinalize()" in page
    assert "state.step === 'DONE'" in page
    assert "state.step === 'FINALIZE_FAILED'" in page
    assert "pathname === '/api/desktop/batch/finalize'" in page
    assert page.index("window.fetch = async") < page.index('<script src="/desktop.js"></script>')


def test_desktop_capability_surface_explicitly_lists_optional_inputs() -> None:
    root = Path(__file__).resolve().parents[2]
    page = (root / "viewer_poc" / "giro3d" / "desktop.html").read_text(encoding="utf-8")

    for capability in (
        "GPS_TIME",
        "TRAJECTORY",
        "GNSS",
        "CAMERA",
        "CAMERA_POSE",
        "CALIBRATION",
        "GIS",
    ):
        assert f"{capability}:" in page
    assert "NOT_AVAILABLE: ['不可用'" in page
    assert "REQUIRES_CONFIRMATION: ['待确认'" in page


def test_dependency_health_contract_names_offline_proj_and_runtime_requirements() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "desktop" / "pdal_runtime.py").read_text(encoding="utf-8")

    assert '"pyproj"' in source
    assert '"PROJ"' in source
    assert '"PROJ_NETWORK": "OFF"' in source
    assert '"RUNTIME_NETWORK_REQUIRED": "NO"' in source
    assert 'pyproj.network.set_network_enabled(False)' in source


def test_tree_review_keeps_numeric_selection_ids_but_shows_business_labels() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "desktop" / "tree_display_numbering_extensions.py").read_text(
        encoding="utf-8"
    )

    assert 'label["text"] = identity["display_tree_id"]' in source
    assert 'tree["display_tree_id"] = identity["display_tree_id"]' in source
    assert "tree_id_by_index" not in source
    assert "treeBusinessId(treeId)" in source
    assert "label.text||label.tree_id" in source
    assert "t.display_tree_id||t.tree_id" in source
    assert "DesktopWorkflow.editor_state = editor_state_with_display_ids" in source
    assert "folder_html.step2_html = step2_html_with_display_ids" in source


def test_tree_review_patch_runs_before_app_start_and_survives_state_refresh() -> None:
    import desktop  # noqa: F401, I001 - package import installs production extensions

    from algorithms.folder_pipeline import html as folder_html

    page = folder_html.step2_html()

    patch = page.index("function treeBusinessId(treeId)")
    start = page.index("window.TreeLineTreeReviewApp.start();")
    assert patch < start
    assert "<thead><tr><th>树编号</th>" in page
    assert "treeBusinessId(row.dataset.tid)" in page
    assert "String(label.text||label.tree_id)" in page


def test_display_numbering_is_optional_for_legacy_projection_without_stage_a() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "desktop" / "tree_display_numbering_extensions.py").read_text(
        encoding="utf-8"
    )

    assert "project = self._stage_project()" in source
    assert "except (FileNotFoundError, OSError, TypeError, ValueError):" in source
    assert "return {}" in source


def test_packaged_launch_recovers_from_protected_working_directory() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "desktop" / "__init__.py").read_text(encoding="utf-8")

    assert "_ensure_packaged_writable_cwd" in source
    assert 'getattr(sys, "_MEIPASS", None)' in source
    assert 'for variable in ("LOCALAPPDATA", "APPDATA")' in source
    assert "os.chdir(candidate)" in source
    assert "tempfile.gettempdir()" in source
