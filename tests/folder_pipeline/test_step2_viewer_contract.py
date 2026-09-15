"""Step-5 Giro3D / Three.js viewer and editor payload regressions."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from algorithms.folder_pipeline.html import step2_html
from algorithms.folder_pipeline.server import FolderServer
from tests.folder_pipeline.test_restart_hydration import _build_completed_workspace

ROOT = Path(__file__).resolve().parents[2]
VIEWER_JS = ROOT / "viewer_poc" / "giro3d" / "tree_review_3d.js"
APP_JS = ROOT / "viewer_poc" / "giro3d" / "tree_review_app.js"


def _viewer_js() -> str:
    return VIEWER_JS.read_text(encoding="utf-8")


def _app_js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_editor_data_schema_is_unambiguous(tmp_path: Path) -> None:
    """The table list and bounded 3D cloud remain separate contracts."""

    _workspace_root, workspace = _build_completed_workspace(tmp_path)
    payload = FolderServer(workspace)._build_editor_display()
    assert set(payload) == {"trees", "trees3d", "background", "wire", "wires3d", "tower"}
    assert isinstance(payload["trees"], list) and payload["trees"]
    assert all("tree_id" in tree and "xyz" not in tree for tree in payload["trees"])
    trees3d = payload["trees3d"]
    assert set(trees3d) == {"xyz", "color", "tree_id_by_index", "labels"}
    assert len(trees3d["xyz"]) == len(trees3d["color"]) == len(trees3d["tree_id_by_index"])
    table_ids = {int(tree["tree_id"]) for tree in payload["trees"]}
    assert {int(label["tree_id"]) for label in trees3d["labels"]} == table_ids
    for key in ("background", "wire", "wires3d", "tower"):
        assert isinstance(payload[key], dict)
        assert isinstance(payload[key]["xyz"], list)


def test_tree_labels_refresh_after_delete_merge_and_undo(tmp_path: Path) -> None:
    _workspace_root, workspace = _build_completed_workspace(tmp_path)
    server = FolderServer(workspace)

    def ids() -> tuple[set[int], set[int]]:
        payload = server._build_editor_display()
        return (
            {int(tree["tree_id"]) for tree in payload["trees"]},
            {int(label["tree_id"]) for label in payload["trees3d"]["labels"]},
        )

    assert ids() == ({0, 1}, {0, 1})
    server.orchestrator.editor_select([0])
    server.orchestrator.editor_delete()
    assert ids() == ({1}, {1})
    server.orchestrator.editor_undo()
    server.orchestrator.editor_select([0, 1])
    server.orchestrator.editor_merge()
    assert ids() == ({2}, {2})
    server.orchestrator.editor_undo()
    assert ids() == ({0, 1}, {0, 1})


def test_step2_uses_only_offline_giro3d_threejs_assets() -> None:
    html = step2_html()
    assert '<script src="/giro3d.bundle.js"></script>' in html
    assert '<script src="/tree-review-3d.js"></script>' in html
    assert '<script src="/tree-review-app.js"></script>' in html
    assert "/plotly.js" not in html
    assert "Plotly" not in html
    assert "http://" not in html and "https://" not in html
    assert "setAttribute('aria-label', 'Giro3D 单木点云复核画布')" in _viewer_js()


def test_step2_uses_one_typed_tree_buffer_and_explicit_ids() -> None:
    viewer = _viewer_js()
    app = _app_js()
    assert "new this.THREE.BufferGeometry()" in viewer
    assert "new this.THREE.Points(" in viewer
    assert "new this.THREE.BufferAttribute" in viewer
    assert "new this.THREE.Uint32BufferAttribute(new Uint32Array(capacity), 1)" in viewer
    assert "geometry.setDrawRange(0, rows.length)" in viewer
    assert "this.treeIds[sourceIndex]" in viewer
    assert "treeIdToMetadata: new Map()" in app
    assert "treeIdToMetadata.set(treeId, tree)" in app
    assert "treeId === index" not in viewer + app


def test_selection_and_preview_are_frontend_local_buffer_updates() -> None:
    app = _app_js()
    viewer = _viewer_js()
    select_tree = re.search(
        r"function selectTree\(treeId, toggle = true, scroll = false\) \{([\s\S]*?)\n  \}",
        app,
    )
    assert select_tree
    assert "updatePlotSelection()" in select_tree.group(1)
    assert "post(" not in select_tree.group(1)
    assert "fetch(" not in select_tree.group(1)
    assert "setSelection(selectionHighlightIds())" in app
    assert "colorAttr.needsUpdate = true" in viewer
    preview = re.search(r"setPreview\(points, treeId\) \{([\s\S]*?)\n    \}", viewer)
    assert preview
    assert "new this.THREE.Points" not in preview.group(1)


def test_camera_is_local_and_supports_rotate_pan_zoom_views() -> None:
    viewer = _viewer_js()
    assert "new G.MapControls(this.camera, this.canvas)" in viewer
    assert "mouseButtons.LEFT = this.THREE.MOUSE.ROTATE" in viewer
    assert "mouseButtons.MIDDLE = this.THREE.MOUSE.PAN" in viewer
    assert "mouseButtons.RIGHT = this.THREE.MOUSE.PAN" in viewer
    assert "enableDamping = true" in viewer
    assert "resetView()" in viewer and "topView()" in viewer and "sideView()" in viewer
    controls = re.search(
        r"this\.controls = new G\.MapControls\([\s\S]*?this\.raycaster =",
        viewer,
    )
    assert controls
    assert "fetch(" not in controls.group(0)
    assert "post(" not in controls.group(0)


def test_lasso_overlay_restores_camera_before_exact_request() -> None:
    app = _app_js()
    assert "state.viewer.setControlsEnabled(false)" in app
    assert "const displayCount = displayLassoPreview" in app
    assert app.index("const displayCount = displayLassoPreview") < app.index(
        "await post('/api/trim/lasso-preview'"
    )
    assert "finally {\n        restoreCameraControls();" in app
    assert "window.addEventListener('error', restoreCameraControls)" in app
    assert "window.addEventListener('unhandledrejection', restoreCameraControls)" in app
    assert "event.key === 'Escape'" in app and "void cancelTrim()" in app


def test_lasso_exact_contract_is_current_tree_matrix_polygon_only() -> None:
    app = _app_js()
    assert "tree_id: treeId, view_projection: matrix, polygon_ndc: polygon" in app
    assert "displayLassoPreview(treeId, matrix, polygon)" in app
    assert "source.las" not in app.lower()
    assert "full source" not in app.lower()


def test_preview_recolors_base_buffer_without_duplicate_layer() -> None:
    viewer = _viewer_js()
    app = _app_js()
    assert "this.previewKeys.has(this.pointKey(point))" in viewer
    assert "color = red" in viewer
    assert "previewPoints" not in viewer
    assert "updatePreviewTrace([], null)" in app
    assert "if (!keepLocalDisplayPreview) updatePreviewTrace" in app


def test_display_budget_and_frame_adaptation_are_display_only() -> None:
    app = _app_js()
    viewer = _viewer_js()
    assert "return 12000" in app and "return 30000" in app and "return 50000" in app
    assert "SLOW_FRAME_MS = 27" in viewer
    assert "this.qualityFraction * 0.72" in viewer
    assert "rebuildTrees()" in viewer
    assert "original_index_by_index" in viewer
    assert "applyDelta(delta)" in viewer
    assert "EDITOR_DISPLAY_DELTA_REVISION_MISMATCH" in viewer
    assert "display_budget=${displayBudget()}" in app


def test_tree_review_javascript_syntax() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    for path in (VIEWER_JS, APP_JS):
        result = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        assert result.returncode == 0, result.stderr


def test_step2_exposes_full_review_controls_and_noncontiguous_labels() -> None:
    html = step2_html()
    for text in (
        "删除单木",
        "合并选中",
        "剔除误分团块",
        "套索剔除",
        "撤销",
        "确认剔除",
        "取消",
        "显示树编号",
        "重置",
        "俯视",
        "侧视",
    ):
        assert text in html
    app = _app_js()
    assert "[...state.treeMap.keys()].sort((a, b) => a - b)" in app
    assert "row.dataset.tid = String(treeId)" in app


def test_span_scoped_review_is_not_overwritten_by_global_status_poll() -> None:
    app = _app_js()
    assert "if (current.tree_snapshot && !reviewSpanId)" in app
    assert "span_id=${encodeURIComponent(reviewSpanId)}" in app


def test_finalized_project_reopens_giro3d_review_without_plotly_redirect() -> None:
    app = _app_js()
    assert "redirectOnDone: false" in app
    assert "workflowDone: false" in app
    assert "if (state.redirectOnDone)" in app
    assert "state.redirectOnDone = true" in app
    assert "if (!state.workflowDone) state.refreshTimer" in app
    assert "复核结果已生成，可继续查看或编辑单木" in app
    old_unconditional_redirect = (
        "if (current.step === 'DONE') {\n"
        "        clearInterval(state.refreshTimer);\n"
        "        window.location.href = '/step3';"
    )
    assert old_unconditional_redirect not in app


def test_viewer_initialization_failure_is_never_a_silent_blank_canvas() -> None:
    app = _app_js()
    assert "3D复核视图初始化失败，请查看详情。" in app
    assert "error.details || error.stack || error.message" in app


def test_slow_editor_display_read_does_not_use_mutation_timeout() -> None:
    app = _app_js()
    assert "const MUTATION_REQUEST_TIMEOUT_MS = 15000;" in app
    assert "const DISPLAY_REQUEST_TIMEOUT_MS = 180000;" in app
    assert "DISPLAY_REQUEST_TIMEOUT_MS," in app
    assert "首次加载可能需要几十秒" in app
    assert "单木复核数据加载失败，请刷新视图重试。" in app


def test_interaction_timings_are_exposed_for_release_acceptance() -> None:
    app = _app_js()
    assert "dataset.lastTreeClickMs = elapsed.toFixed(3)" in app
    assert "dataset.lastLassoDisplayMs = elapsed.toFixed(3)" in app
    assert "dataset.lastLassoExactMs = exactElapsed.toFixed(3)" in app
