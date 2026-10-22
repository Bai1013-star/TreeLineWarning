"""Isolated UI verification server. No user projects or databases are touched."""
from io import BytesIO
from pathlib import Path
import os

import uvicorn
from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from openpyxl import Workbook

from api.prediction_api import create_app, predict_forecast
from algorithms.folder_pipeline.export_excel import write_prediction_input_excel

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get('AUDIT_FIXTURE_ROOT', str(ROOT / "verification_unified_prediction")))
ARTIFACTS.mkdir(exist_ok=True)
app = create_app(
    database_path=ARTIFACTS / "ui_test.db",
    web_index_path=ROOT / "web/templates/index.html",
    species_growth_model_path=ROOT / "models/species_growth.json",
    temporal_model_path=ROOT / "models/temporal_attention.npz",
    training_output_dir=ARTIFACTS / "training",
)
TREES = [
    dict(tree_key=f"fixture-{index}", tree_id=9000 + index, tree_code=f"TEST-{index:03}",
         persistent_tree_id=f"fixture-{index}", height=8 + index, diameter=18 + index,
         crown_width=2.5, horizontal_distance=4, vertical_distance=3,
         clearance_distance=5, publication_state="FINALIZED", prediction_ready=True,
         span_label="档距1" if index <= 2 else "档距2", line_id="TEST-LINE",
         measurement_source="UI test fixture, not a LAS measurement", quality_state="VALID")
    for index in range(1, 4)
]


@app.get('/desktop.html')
@app.get('/desktop.js')
@app.get('/desktop.css')
def desktop_assets(request: Request):
    return FileResponse(ROOT / 'viewer_poc/giro3d' / request.url.path.lstrip('/'))


@app.get('/api/desktop/workspace')
def desktop_fixture():
    # Navigation fixture only: no LAS processing or user project files.
    return dict(ui_state='TOPOLOGY_READY', tree_review_ready=True, finalized=False,
                can_predict=True, test_mode=False,
                project=dict(root='UI-NAVIGATION-FIXTURE', project_name='导航验证项目',
                             lines=1, towers=2, spans=1, analysis='AVAILABLE'),
                capability=None, batch=dict(status='COMPLETE', spans=[],
                                           risk_policy=dict(warning_clearance_threshold_m=.5)))


@app.get('/step2')
@app.get('/step3')
@app.get('/dbh-review')
@app.get('/future-preview.html')
def processing_navigation_fixture(request: Request):
    # Isolate navigation from scientific computation; real warning APIs stay active.
    name = request.url.path
    return HTMLResponse(f'<h1 id="fixtureTitle">{name}</h1>'
                        '<a href="/dbh-review" id="fixtureDbh">胸径复核</a>'
                        '<a href="/step3" id="fixtureReport">报告</a>'
                        '<a href="/step2" id="fixtureReview">复核</a>'
                        '<input id="fixtureInput">')


@app.get("/integrated/context")
def fixture_context():
    return dict(status="READY", trees=TREES, message="UI fixture: 3 trees")


@app.post("/integrated/forecast")
def fixture_forecast(request: Request, body: dict):
    tree = next(tree for tree in TREES if tree["tree_key"] == body["tree_key"])
    data = {**tree, "species": body["species"], "age": body["age"],
            "observation_date": body["observation_date"], "voltage_level_kv": 110,
            "line_name": "测试线路", "tower_section": tree["span_label"],
            "selected_fields": ["tree_id", "tree_code", "species", "age", "height",
                                "diameter", "crown_width", "horizontal_distance",
                                "vertical_distance", "clearance_distance", "voltage_level_kv",
                                "line_name", "tower_section"]}
    for key in body["weather"]["selected_fields"]:
        data[key] = body["weather"][key]
    result = predict_forecast(request, {**body, "tree_data": data})
    result["integrated_source"] = {"tree_key": tree["tree_key"]}
    return result


@app.get("/verification/fixture.xlsx")
def fixture_excel():
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["tree_id", "species", "age", "height", "diameter", "crown_width",
                  "voltage_level_kv", "horizontal_distance", "vertical_distance", "clearance_distance"])
    sheet.append([8001, "杨树", 20, 12, 21, 3, 110, 4, 3, 5])
    sheet.append([8002, "松树", 30, 13, 22, 4, 110, 4, 3, 5])
    stream = BytesIO()
    workbook.save(stream)
    return Response(stream.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/verification/exported_fixture.xlsx")
def exported_fixture_excel():
    output = write_prediction_input_excel(
        ARTIFACTS / "actual_export_format.xlsx",
        [dict(persistent_tree_id="test-tree-004", display_tree_id="T-1-004",
              height_m=9.26, diameter_cm=16.92, crown_width_m=4.42,
              minimum_clearance_m=5, horizontal_clearance_m=4, vertical_clearance_m=3)],
        {"science_run_id": "UI export-format verification"},
    )
    return Response(output.read_bytes(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if os.environ.get('AUDIT_MISSING_MEASUREMENTS') == '1':
    from copy import deepcopy
    import desktop.warning_bridge as bridge
    from desktop.manual_dbh import apply_measurements as apply_dbh
    from desktop.manual_measurements import apply_measurements as apply_geometry
    TREES[0].update(height=None, diameter=None, crown_width=None, horizontal_distance=None,
                    clearance_distance=None, prediction_ready=False)
    def missing_context(_):
        rows = deepcopy(TREES)
        apply_dbh(rows, ARTIFACTS, 'missing-measurements-fixture')
        apply_geometry(rows, ARTIFACTS, 'missing-measurements-fixture')
        return dict(status='READY', project_root=str(ARTIFACTS), analysis_run_id='missing-measurements-fixture',
                    trees=rows, source='UI_TEST_FIXTURE_NOT_FIELD_DATA')
    bridge.integrated_context = missing_context
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, 'path', '') not in
                           ('/integrated/context', '/integrated/forecast')]
    bridge.register_warning_bridge(app, processing_backend=object(), project_root=ARTIFACTS)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get('AUDIT_PORT', '18765')))
