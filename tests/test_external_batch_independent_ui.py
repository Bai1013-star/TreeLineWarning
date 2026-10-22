from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_batch_is_independent_and_file_first() -> None:
    html = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    start = html.index("async function initializeWorkspace()")
    end = html.index("window.addEventListener('message'", start)
    initialize = html[start:end]

    assert initialize.index("requested==='batch'") < initialize.index(
        "await loadIntegratedMode()"
    )
    assert "长时表格预测（独立导入）" in html
    assert 'id="batchFile"' in html
    assert 'id="loadBatchData"' in html
    assert '解析并预览表格</button>' in html
    assert html.index('id="loadBatchData"') < html.index('id="batchFile"')
    assert html.index('id="batchPreview"') < html.index('id="externalContextHolder"')
    assert 'id="downloadExternalBatchTemplate"' in html
    assert "表中不要求年月日、经纬度或天气" in html
    assert "const externalRequired=['tree_id','height','clearance_distance']" in html
    assert 'id="externalDefaultSpecies"' in html
    assert 'id="externalObservationDate"' in html
    assert 'id="externalContextHolder"' in html


def test_desktop_ribbon_opens_the_external_batch_as_a_distinct_workspace() -> None:
    desktop_html = (ROOT / "viewer_poc" / "giro3d" / "desktop.html").read_text(
        encoding="utf-8"
    )
    desktop_js = (ROOT / "viewer_poc" / "giro3d" / "desktop.js").read_text(
        encoding="utf-8"
    )

    assert "树木预测（1～多棵）" in desktop_html
    assert 'id="ribbonBatchPrediction"' not in desktop_html
    assert "openWarningWorkspace('batch')" in desktop_js
    assert "_workspace_seq" in desktop_js
    assert "workspace: state.warningWorkspace" in desktop_js


def test_unified_prediction_has_two_visible_sources_and_one_count() -> None:
    html = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="suiteBatch"' not in html
    assert 'id="projectBatchMode"' not in html
    assert 'id="singleMode" type="button" role="tab" aria-selected="true" hidden' not in html
    assert html.count('id="integratedTreeCount"') == 1
    assert 'id="manualExtraTrees"' in html
    assert "selectedManualTrees()" in html
    assert "presentTreeForecasts(results" in html


def test_back_navigation_preserves_prediction_frame_and_validates_message_source() -> None:
    html = (ROOT / 'web/templates/index.html').read_text(encoding='utf-8')
    desktop_html = (ROOT / 'viewer_poc/giro3d/desktop.html').read_text(encoding='utf-8')
    desktop_js = (ROOT / 'viewer_poc/giro3d/desktop.js').read_text(encoding='utf-8')
    assert 'id="navigateBack"' in desktop_html
    assert 'id="auxiliaryFrame"' in desktop_html
    assert 'function navigateBack()' in desktop_js
    assert 'event.source!==$(\'moduleFrame\').contentWindow' in desktop_js
    assert 'treeline-restore-view' in html and 'treeline-restore-view' in desktop_js
    assert 'view.revision!==resultRevision' in html
    assert 'loadWorkspace(true)' in desktop_js
