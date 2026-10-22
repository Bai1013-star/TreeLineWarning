import re
from pathlib import Path
from types import SimpleNamespace

from algorithms.folder_pipeline.html import step2_html
from desktop.workflow import DesktopWorkflow

ROOT = Path(__file__).resolve().parents[1]


def test_embedded_review_allows_operator_confirmation_dialogs():
    html = (ROOT / 'viewer_poc/giro3d/desktop.html').read_text(encoding='utf8')
    frame = re.search(r'<iframe id="viewer"[^>]*sandbox="([^"]+)"', html)
    assert frame is not None
    assert 'allow-modals' in frame.group(1).split()


def test_lasso_uses_stable_svg_coordinates_and_visible_commit():
    script = (ROOT / 'viewer_poc/giro3d/tree_review_app.js').read_text(encoding='utf8')
    lasso = script.split('function beginLasso()', 1)[1].split('async function cancelTrim', 1)[0]
    assert 'event.offsetX' not in lasso and 'event.offsetY' not in lasso
    assert 'event.clientX - rect.left' in lasso
    assert "line.setAttribute('pointer-events', 'none')" in lasso
    assert "panel.scrollTop += button.bottom - container.bottom + 12" in script
    assert '确认删除圈内点' in script


def test_windows_navigation_disconnect_is_not_analysis_failure():
    from desktop.server import _DesktopHandler
    handler = object.__new__(_DesktopHandler)
    handler.path = '/api/final'
    def disconnected(path):
        raise ConnectionAbortedError(10053, 'User navigated away')
    handler._handle_get = disconnected
    handler._exception_error = lambda error: (_ for _ in ()).throw(AssertionError('False analysis failure'))
    handler.do_GET()


def test_native_picker_does_not_expire_before_operator_finishes():
    html = (ROOT / 'web/templates/index.html').read_text(encoding='utf8')
    method = html.split('async function callNativeExport(method,args){', 1)[1].split('function openProcessingWorkspace', 1)[0]
    assert 'setTimeout(' not in method
    assert 'nativeExportPending.set(id' in method


def test_integrated_cold_load_is_shared_and_refresh_does_not_falsely_succeed():
    html = (ROOT / 'web/templates/index.html').read_text(encoding='utf8')
    assert 'if(integratedContextLoad)return integratedContextLoad' in html
    assert "fetch('/integrated/context',{cache:'no-store',signal:AbortSignal.timeout(180000)})" in html
    assert "if(loaded)$('processedDataStatus').textContent=`刷新完成：" in html
    assert "integratedContext.status==='ERROR'?integratedContext.message" in html


def test_production_review_contains_every_directly_bound_control():
    html = step2_html()
    script = (ROOT / 'viewer_poc/giro3d/tree_review_app.js').read_text(encoding='utf8')
    binding = script.split('function bindControls()', 1)[1].split('async function start()', 1)[0]
    ids = re.findall(r"byId\('([^']+)'\)\.(?:onclick|onchange|textContent|title)", binding)
    for control in ids:
        assert f'id="{control}"' in html, control
    assert html.count('id="btn-split"') == 1
    async_paths = script.split('const ASYNC_MUTATION_PATHS = new Set([', 1)[1].split(']);', 1)[0]
    assert "'/api/split'" in async_paths


def test_control_binding_errors_are_caught_by_startup():
    script = (ROOT / 'viewer_poc/giro3d/tree_review_app.js').read_text(encoding='utf8')
    startup = script.split('async function start()', 1)[1].split('window.TreeLineTreeReviewApp =', 1)[0]
    assert startup.index('try {') < startup.index('bindControls();')
    assert 'showFailure(error)' in startup and '复核加载失败' in startup


def test_disk_failure_is_not_masked_by_completed_science_manifest():
    workflow = object.__new__(DesktopWorkflow)
    workflow._workflow = {'last_batch_status': 'FAILED', 'last_error': 'OSError: No space left on device'}
    workflow._batch_error = None
    workflow._stage_project = lambda: SimpleNamespace(spans=[])
    workflow._latest_batch_manifest = lambda: {'status': 'COMPLETE'}
    workflow._risk_policy = lambda: SimpleNamespace(to_dict=lambda: {})
    result = workflow.batch_projection()
    assert result['status'] == 'FAILED'
    assert 'No space' in result['error']


def test_hidden_auxiliary_frames_override_default_iframe_display():
    css = (ROOT / 'viewer_poc/giro3d/desktop.css').read_text(encoding='utf8')
    assert '.module-frame-wrap iframe.hidden{display:none}' in css


def test_review_uses_separate_fitted_wires_not_tower_schematic():
    workflow = (ROOT / 'desktop/workflow.py').read_text(encoding='utf8')
    projection = workflow.split('    def editor_data(', 1)[1].split('    @editor_action("SELECT")', 1)[0]
    assert 'current_wire_curves(editor.result)' in projection
    assert '"segments": wire_segments' in projection
    assert 'start.position[2] +' not in projection
    script = (ROOT / 'viewer_poc/giro3d/tree_review_3d.js').read_text(encoding='utf8')
    assert 'fittedSegments.forEach(segment => this.addWireCurve(xyzOf(segment)))' in script
    html = step2_html()
    assert '分割参考高(m)' in html


def test_reload_display_recovers_persisted_undo_journal():
    source = (ROOT / 'desktop/workflow.py').read_text(encoding='utf8')
    projection = source.split('    def editor_data(', 1)[1].split('    @editor_action("SELECT")', 1)[0]
    assert 'display["operations"]' in projection
    assert 'display["tree_edit_count"] = len(editor.session.operations)' in projection


def test_reviewing_data_can_export_draft_excel_without_bypassing_final_gate():
    source = (ROOT / 'web/templates/index.html').read_text(encoding='utf8')
    export = source.split('async function downloadProcessedExcel(){', 1)[1].split('function processedRiskText', 1)[0]
    assert "tree.publication_state!=='FINALIZED'" in export
    assert '复核中.xlsx' in export and "'/exports/table'" in export
    assert "new URL('/download_excel'" in export


def test_twin_missing_measurements_and_calendar_are_not_zero_or_timezone_shifted():
    js = (ROOT / 'viewer_poc/giro3d/future_preview.js').read_text(encoding='utf8')
    assert "if (item === null || item === undefined || item === '') return null;" in js
    assert 'setUTCDate(startDate.getUTCDate() + targetDay)' in js
    html = (ROOT / 'viewer_poc/giro3d/future_preview.html').read_text(encoding='utf8')
    assert 'id="growthPlot"' in html and 'id="clearancePlot"' in html
    growth = js.split("plot.call(window.Plotly, growthPlot", 1)[1].split(
        "plot.call(window.Plotly, clearancePlot", 1
    )[0]
    assert '项目净空标准' not in growth and "yaxis2:{title:'胸径 (cm)'" in growth


def test_species_prediction_is_in_short_term_workspace_not_model_prediction_panel():
    html = (ROOT / 'web/templates/index.html').read_text(encoding='utf8')
    integrated = html.split('id="integratedEntry"', 1)[1].split(
        'id="integratedWeatherDialog"', 1
    )[0]
    training = html.split('id="trainingPage"', 1)[1]
    assert 'id="predictIntegratedSpecies"' in integrated
    assert 'id="integratedRecognitionPhotos"' in integrated
    assert 'id="integratedRecognitionVideos"' in integrated
    assert 'id="integratedRecognitionPointCloud"' in integrated
    assert '前 3 个候选' in integrated
    assert 'id="predictRecognition"' not in training
    assert '.slice(0,3)' in html


def test_busy_twin_is_retryable_but_unreviewed_project_is_not(monkeypatch):
    from desktop import future_preview as preview

    workflow = SimpleNamespace(_lock=SimpleNamespace(acquire=lambda **kwargs: False))
    monkeypatch.setattr(preview, 'current_finalized_gate', lambda *a, **k: {'status': 'PASS'})
    busy = preview.build_future_preview(workflow)
    assert busy['status'] == 'LOCKED' and busy['retryable'] is True
    assert '请先完成' not in busy['message']
    monkeypatch.setattr(preview, 'current_finalized_gate', lambda *a, **k: {'status': 'LOCKED'})
    unreviewed = preview.build_future_preview(workflow)
    assert unreviewed['retryable'] is False


def test_twin_gate_accepts_current_processed_tree_cloud_before_final_review(monkeypatch):
    from desktop import future_preview as preview
    import numpy as np

    reconciled = SimpleNamespace(
        final_tree_id=np.asarray([7, 7, 7], dtype=np.int32),
        final_xyz_local_m=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 1.3], [0.2, 0.1, 3.0]],
            dtype=np.float64,
        ),
        global_tree_records=(SimpleNamespace(global_tree_id=7),),
    )
    editor = SimpleNamespace(
        session=SimpleNamespace(operations=[{"operation": "TRIM"}]),
        final_reconciled=lambda: reconciled,
    )
    result = SimpleNamespace(analysis_run_id="review-ready-run")
    workflow = SimpleNamespace(
        _ensure_editor=lambda: editor,
        _current_result=lambda: result,
    )
    monkeypatch.setattr(
        preview,
        "current_finalized_gate",
        lambda *a, **k: {"status": "LOCKED", "reasons": ["not published"]},
    )

    gate = preview.preview_gate(workflow)

    assert gate["status"] == "PASS"
    assert gate["publication_state"] == "REVIEW_READY"
    assert gate["provisional"] is True
    assert gate["tree_count"] == 1
    assert gate["editor_operation_count"] == 1


def test_dbh_prepares_missing_or_stale_evidence_but_rejects_corruption(tmp_path, monkeypatch):
    from desktop import dbh_review as adapter

    pointer = tmp_path / 'dbh_final.json'
    monkeypatch.setattr(adapter.review.v1, '_run_root', lambda *a: tmp_path)
    generated = []
    monkeypatch.setattr(adapter.v3, 'finalize_dbh', lambda *a: generated.append(a))
    monkeypatch.setattr(adapter.v2, '_load_base', lambda *a: ('proof', {}))
    result = SimpleNamespace(analysis_run_id='test-run')
    assert adapter._load_review_base(tmp_path, result, object())[0] == 'proof'
    assert len(generated) == 1
    pointer.touch()

    outcomes = iter([ValueError('DBH_V2_STALE_BASELINE'), ('updated', {})])
    def stale_then_valid(*args):
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(adapter.v2, '_load_base', stale_then_valid)
    assert adapter._load_review_base(tmp_path, result, object())[0] == 'updated'
    assert len(generated) == 2

    def corrupt(*args):
        raise ValueError('DBH_PROOF_CORRUPTED')
    monkeypatch.setattr(adapter.v2, '_load_base', corrupt)
    import pytest
    with pytest.raises(ValueError, match='DBH_PROOF_CORRUPTED'):
        adapter._load_review_base(tmp_path, result, object())
    assert len(generated) == 2


def test_dbh_review_resumes_valid_but_incomplete_raw_ground_recovery(tmp_path, monkeypatch):
    from desktop import dbh_review as adapter

    (tmp_path / 'dbh_final.json').touch()
    monkeypatch.setattr(adapter.review.v1, '_run_root', lambda *a: tmp_path)
    generated = []
    monkeypatch.setattr(adapter.v3, 'finalize_dbh', lambda *a: generated.append(a))
    outcomes = iter([({'recovery_version': 'V1'}, {}),
                     ({'recovery_version': adapter.v3.VERSION}, {})])
    monkeypatch.setattr(adapter.v2, '_load_base', lambda *a: next(outcomes))
    base, _ = adapter._load_review_base(tmp_path, SimpleNamespace(analysis_run_id='test'), object())
    assert base['recovery_version'] == adapter.v3.VERSION
    assert len(generated) == 1


def test_dbh_review_existing_measurement_remains_recheckable(tmp_path, monkeypatch):
    """An automatic DBH is a protected baseline, not a disabled UI state."""
    from desktop import dbh_review as adapter

    row = {
        'persistent_tree_id': 'tree-1',
        'global_tree_id': 1,
        'dbh_cm': 18.4,
        'dbh_method': 'RAW_GROUND_BH_SECTION',
        'dbh_fit_status': 'PASS',
    }
    result = SimpleNamespace(analysis_run_id='test-run')
    editor = object()
    workflow = SimpleNamespace(_log=lambda *a, **k: None)
    monkeypatch.setattr(adapter, '_context', lambda *a: (tmp_path, result, object(), editor))
    monkeypatch.setattr(adapter, '_load_review_base', lambda *a: ({'trees': [row]}, {}))
    monkeypatch.setattr(adapter, '_current_display_identities', lambda *a: {1: {'display_tree_id': 'T-1-001'}})

    listed = adapter.list_trees(workflow)
    assert listed['trees'][0]['raw_original_las_review_read_only'] is False
    assert listed['trees'][0]['raw_original_las_recheck_existing'] is True

    prepared = {
        'persistent_tree_id': 'tree-1',
        'global_tree_id': 1,
        'current_dbh_cm': 18.4,
        'read_only': True,  # Simulates a pending session from an older build.
        'current_crown': {'label': 'tree-1'},
    }
    monkeypatch.setattr(adapter.review, '_row', lambda *a, **k: row)
    monkeypatch.setattr(adapter.review, 'prepare_review', lambda *a, **k: dict(prepared))
    payload = adapter.prepare(workflow, {'persistent_tree_id': 'tree-1'})
    assert payload['review']['read_only'] is False
    assert payload['review']['recheck_existing_result'] is True


def test_parallel_startup_requests_initialize_one_database(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time
    from api import prediction_api

    created = []
    database = object()
    def create(path):
        created.append(path)
        time.sleep(.03)
        return database
    monkeypatch.setattr(prediction_api, 'create_database', create)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        database=None, database_path='test.sqlite', database_init_lock=Lock())))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: prediction_api._database_for(request), range(8)))
    assert results == [database] * 8
    assert created == ['test.sqlite']

