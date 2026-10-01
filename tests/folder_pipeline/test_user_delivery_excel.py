from __future__ import annotations

import json
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace

import pytest
from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
from algorithms.folder_pipeline.server import FolderServer
from algorithms.folder_pipeline.state import FolderWorkflowState
from algorithms.folder_pipeline.user_delivery_excel import (
    DELIVERY_COLUMNS,
    DELIVERY_FILENAME,
    _measurement_reason,
    _unmeasured_reason_code,
    migrate_legacy_prediction,
    write_delivery_from_prediction,
)
from desktop.server import _DesktopHandler
from desktop.workflow import DesktopWorkflow
from openpyxl import load_workbook

CSV_HEADER = (
    "tree_id,species,observation_date,observation_year,observation_month,"
    "observation_day,height,age,diameter,vertical_distance,horizontal_distance,"
    "clearance_distance,wind_offset_distance,voltage_level_kv,maximum_sag_winter,"
    "maximum_sag_summer,wind_speed,rainfall,humidity,temperature,sunlight_hours,"
    "risk_score,risk_level"
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("LOCAL_BASE_HEIGHT_AMBIGUOUS", "多个可能地面高度"),
        ("LOCAL_BASE_HEIGHT_UNVERIFIED", "地面尚未确认"),
        ("MULTIPLE_STEM_AXES_REVIEW_REQUIRED", "多根候选树干轴"),
        ("PER_TREE_STEM_EVIDENCE_REQUIRED", "本树可核验的树干"),
        ("SEARCH_BUDGET_EXHAUSTED", "搜索达到预算上限"),
        ("SEARCH_BUDGET_LIMIT", "搜索达到预算上限"),
    ],
)
def test_unmeasured_diameter_explains_review_action(status, expected):
    assert expected in _measurement_reason(status)
    assert _unmeasured_reason_code({
        "dbh_fit_status": status,
        "dbh_pre_reference_failure_reason": "NO_ABSOLUTE_CANDIDATE",
    }) == status


def test_delivery_matches_23_column_example_without_inventing_missing_inputs(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{
            "persistent_tree_id": "tree-17",
            "display_tree_id": "T-001",
            "height_m": 12.4,
            "dbh_cm": 31.2,
            "diameter_cm": 31.2,
            "vertical_clearance_m": 4.1,
            "horizontal_clearance_m": 3.2,
            "minimum_clearance_m": 2.9,
            "current_risk": "WARNING",
            "observation_date": "2026-09-28",
        }],
        {"science_run_id": "fixture"},
    )
    output = write_delivery_from_prediction(internal)
    workbook = load_workbook(output, read_only=True, data_only=True)
    try:
        assert workbook.sheetnames == ["树木风险结果"]
        sheet = workbook.active
        assert sheet.max_row == 2
        assert tuple(cell.value for cell in sheet[1]) == tuple(CSV_HEADER.split(","))
        assert DELIVERY_COLUMNS == tuple(CSV_HEADER.split(","))
        row = dict(zip(DELIVERY_COLUMNS, (cell.value for cell in sheet[2]), strict=True))
        assert row["tree_id"] == "T-001"
        assert row["observation_date"] == "2026/9/28"
        assert (
            row["observation_year"], row["observation_month"], row["observation_day"]
        ) == (2026, 9, 28)
        assert (row["height"], row["diameter"]) == (12.4, 31.2)
        assert row["risk_level"] == "中风险"
        assert all(row[key] is None for key in (
            "species", "age", "wind_offset_distance", "voltage_level_kv",
            "maximum_sag_winter", "maximum_sag_summer", "wind_speed",
            "rainfall", "humidity", "temperature", "sunlight_hours", "risk_score",
        ))
    finally:
        workbook.close()


def test_delivery_has_one_row_per_retained_tree_and_maps_both_risk_vocabularies(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [
            {"persistent_tree_id": "retained-1", "display_tree_id": "T-1-001",
             "current_risk": "DANGEROUS", "observation_date": "2026-09-28",
             "dbh_cm": 23.0},
            {"persistent_tree_id": "retained-2", "display_tree_id": "T-1-002",
             "current_risk": "DANGER", "observation_date": "2026-09-28",
             "dbh_cm": 25.0},
        ],
        {"science_run_id": "fixture"},
    )
    delivery = write_delivery_from_prediction(internal)
    workbook = load_workbook(delivery, read_only=True)
    try:
        rows = list(workbook.active.values)
    finally:
        workbook.close()
    assert len(rows) == 3
    assert [row[0] for row in rows[1:]] == ["T-1-001", "T-1-002"]
    assert [row[-1] for row in rows[1:]] == ["高风险", "高风险"]


def test_delivery_is_readable_and_diameter_comments_preserve_evidence(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [
            {
                "persistent_tree_id": "estimate",
                "observation_date": "2026-09-28",
                "dbh_cm": 16.2,
                "dbh_method": "RECOVERED_STEM",
                "dbh_quality": "ESTIMATE",
                "dbh_uncertainty_cm": 1.35,
                "dbh_model_version": "DBH_RECOVERY_V3_PARTIAL_ARC",
            },
            {
                "persistent_tree_id": "reference",
                "observation_date": "2026-09-28",
                "dbh_cm": 22.8,
                "dbh_method": "REFERENCE_ESTIMATE",
                "dbh_quality": "ESTIMATE",
                "dbh_pre_reference_failure_reason": "EDITOR_REMOVED_BH_EVIDENCE",
                "dbh_reference_estimate_json": json.dumps({
                    "basis": "PROJECT_TRUSTED_MEASUREMENTS",
                    "reference_count": 2,
                    "reference_mean_cm": 22.8,
                    "reference_min_cm": 20.0,
                    "reference_max_cm": 25.6,
                    "reference_std_cm": 2.8,
                    "reference_tree_persistent_ids": ["tree-a", "tree-b"],
                }),
            },
        ],
        {"science_run_id": "fixture"},
    )
    workbook = load_workbook(write_delivery_from_prediction(internal))
    try:
        sheet = workbook.active
        assert all(
            sheet.column_dimensions[cell.column_letter].width >= len(cell.value) + 3
            for cell in sheet[1]
        )
        assert sheet.row_dimensions[1].height >= 30
        diameter = sheet["I2"]
        assert diameter.number_format == "0.00"
        assert diameter.comment is not None
        assert "ESTIMATE" in diameter.comment.text
        assert "RECOVERED_STEM" in diameter.comment.text
        assert "DBH_RECOVERY_V3_PARTIAL_ARC" in diameter.comment.text
        assert "估算不确定度约 1.35 cm（非置信区间）" in diameter.comment.text
        reference = sheet["I3"]
        assert reference.value is None
        assert reference.comment is not None
        assert "旧项目参考估算已撤销" in reference.comment.text
        assert "胸高处点已由人工删点操作剔除" in reference.comment.text
    finally:
        workbook.close()


def test_user_reference_comment_identifies_assumption_without_claiming_measurement(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{
            "persistent_tree_id": "user-estimate",
            "observation_date": "2026-09-28",
            "dbh_cm": 31.5,
            "dbh_method": "REFERENCE_ESTIMATE",
            "dbh_quality": "ESTIMATE",
            "dbh_pre_reference_failure_reason": "NO_CROWN_TO_STEM_IDENTITY",
            "dbh_reference_estimate_json": json.dumps({
                "basis": "USER_PROVIDED_REFERENCE",
                "estimator": "USER_VALUE",
                "value_cm": 31.5,
                "reference_count": 0,
                "reference_tree_ids": [],
            }),
        }],
        {"science_run_id": "fixture"},
    )
    workbook = load_workbook(write_delivery_from_prediction(internal))
    try:
        cell = workbook.active["I2"]
        assert cell.value is None
        assert "旧项目参考估算已撤销" in cell.comment.text
        assert "未找到与当前树冠可靠对应的树干" in cell.comment.text
    finally:
        workbook.close()


def test_delivery_distinguishes_single_side_bh_fit_from_upper_stem_extrapolation(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [
            {"persistent_tree_id": "single-side", "observation_date": "2026-09-28",
             "dbh_cm": 20.1, "dbh_method": "RECOVERED_STEM", "dbh_quality": "ESTIMATE",
             "dbh_model_version": "DBH_RECOVERY_V3_UPPER_PARTIAL_ARC"},
            {"persistent_tree_id": "upper-stem", "observation_date": "2026-09-28",
             "dbh_cm": 21.2, "dbh_method": "RECOVERED_STEM", "dbh_quality": "ESTIMATE",
             "dbh_model_version": "DBH_RECOVERY_V3_UPPER_CIRCULAR"},
        ],
        {"science_run_id": "fixture"},
    )
    workbook = load_workbook(write_delivery_from_prediction(internal))
    try:
        assert "本树胸高点云拟合（单侧估算）" in workbook.active["I2"].comment.text
        assert "本树上部树干外推估算" in workbook.active["I3"].comment.text
    finally:
        workbook.close()


@pytest.mark.parametrize("missing", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_delivery_keeps_missing_or_nonpositive_diameter_blank_with_reason(
    tmp_path, missing
):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "T-1-003", "observation_date": "2026-09-28",
          "dbh_cm": missing, "dbh_fit_status": "REFERENCE_REQUIRED"}],
        {"science_run_id": "fixture"},
    )
    output = write_delivery_from_prediction(internal)
    assert internal.is_file()
    workbook = load_workbook(output)
    try:
        assert workbook.active["I2"].value is None
        assert "REFERENCE_REQUIRED" in workbook.active["I2"].comment.text
    finally:
        workbook.close()


def test_delivery_never_uses_legacy_diameter_when_current_dbh_is_missing(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "current", "observation_date": "2026-09-28",
          "diameter_cm": 25.0, "dbh_cm": None, "dbh_fit_status": "REFERENCE_REQUIRED"}],
        {"science_run_id": "fixture"},
    )
    workbook = load_workbook(write_delivery_from_prediction(internal))
    try:
        assert workbook.active["I2"].value is None
        assert workbook.active["I2"].comment is not None
    finally:
        workbook.close()


def test_delivery_uses_legacy_diameter_only_with_measured_source(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [
            {"persistent_tree_id": "measured", "observation_date": "2026-09-28",
             "diameter_cm": 25.0, "diameter_source": "FOLDER_BH_WIDTH_MEASUREMENT"},
            {"persistent_tree_id": "unknown", "observation_date": "2026-09-28",
             "diameter_cm": 23.5},
        ],
        {"science_run_id": "fixture"},
    )
    workbook = load_workbook(write_delivery_from_prediction(internal))
    try:
        assert workbook.active["I2"].value == 25.0
        assert workbook.active["I3"].value is None
    finally:
        workbook.close()


def test_desktop_delivery_rebuilds_stale_file_even_when_its_timestamp_is_newer(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "current", "observation_date": "2026-09-28",
          "dbh_cm": 22.8}],
        {"science_run_id": "fixture"},
    )
    delivery = tmp_path / "exports" / DELIVERY_FILENAME
    delivery.parent.mkdir()
    delivery.write_bytes(b"stale export")
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._lock = threading.RLock()
    workflow._workflow = {}
    workflow._save_workflow = lambda: None
    workflow.export_prediction = lambda: internal
    assert workflow.export_delivery() == delivery
    workbook = load_workbook(delivery, read_only=True)
    try:
        assert workbook.sheetnames == ["树木风险结果"]
        assert workbook.active.max_row == 2
    finally:
        workbook.close()


def test_desktop_missing_diameter_replaces_stale_delivery_with_blank_comment(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "current", "observation_date": "2026-09-28",
          "dbh_fit_status": "REFERENCE_REQUIRED"}],
        {"science_run_id": "fixture"},
    )
    delivery = tmp_path / "exports" / DELIVERY_FILENAME
    delivery.parent.mkdir()
    delivery.write_bytes(b"stale export")
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._lock = threading.RLock()
    workflow._workflow = {"dbh_delivery_status": "REFERENCE_REQUIRED", "dbh_reference_cm": None}
    workflow._save_workflow = lambda: None
    workflow.export_prediction = lambda: internal
    assert workflow.export_delivery() == delivery
    workbook = load_workbook(delivery)
    try:
        assert workbook.active["I2"].value is None
        assert "REFERENCE_REQUIRED" in workbook.active["I2"].comment.text
    finally:
        workbook.close()
    assert not list((tmp_path / "artifacts").glob("树木风险结果.legacy-*.xlsx"))


def test_folder_missing_diameter_downloads_blank_comment_and_replaces_stale_delivery(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "02_tree_analysis" / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "current", "observation_date": "2026-09-28",
          "dbh_fit_status": "REFERENCE_REQUIRED"}],
        {"science_run_id": "fixture"},
    )
    export_dir = tmp_path / "03_export"
    export_dir.mkdir()
    delivery = export_dir / DELIVERY_FILENAME
    delivery.write_bytes(b"stale export")
    app = FolderServer.__new__(FolderServer)
    app.workspace = SimpleNamespace(
        excel_path=internal,
        export_dir=export_dir,
        tree_analysis_dir=internal.parent,
    )
    app.state = FolderWorkflowState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.handler_factory())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/download_excel", timeout=5
        ) as response:
            assert response.status == 200
            workbook = load_workbook(BytesIO(response.read()))
        try:
            assert workbook.active["I2"].value is None
            assert "REFERENCE_REQUIRED" in workbook.active["I2"].comment.text
        finally:
            workbook.close()
        assert delivery.is_file()
        assert delivery.read_bytes() != b"stale export"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_folder_download_regenerates_stale_delivery_from_current_prediction(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "02_tree_analysis" / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "current", "observation_date": "2026-09-28",
          "dbh_cm": 22.8}],
        {"science_run_id": "fixture"},
    )
    export_dir = tmp_path / "03_export"
    export_dir.mkdir()
    legacy = export_dir / "Prediction_Input.xlsx"
    internal.replace(legacy)
    (export_dir / DELIVERY_FILENAME).write_bytes(b"stale export")
    app = FolderServer.__new__(FolderServer)
    app.workspace = SimpleNamespace(excel_path=internal, export_dir=export_dir)
    app.state = FolderWorkflowState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.handler_factory())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/download_excel", timeout=5
        ) as response:
            assert response.status == 200
            assert "filename*=UTF-8''" in response.headers["Content-Disposition"]
            body = response.read()
        workbook = load_workbook(BytesIO(body), read_only=True)
        try:
            assert workbook.sheetnames == ["树木风险结果"]
            assert tuple(workbook.active.values)[1][0] == "current"
        finally:
            workbook.close()
        assert internal.is_file()
        assert not legacy.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_desktop_download_aliases_serve_delivery_workbook(tmp_path):
    internal = write_prediction_input_excel(
        tmp_path / "Prediction_Input.xlsx",
        [{"persistent_tree_id": "retained", "observation_date": "2026-09-28",
          "dbh_cm": 22.8}],
        {"science_run_id": "fixture"},
    )
    workflow = DesktopWorkflow.__new__(DesktopWorkflow)
    workflow._root = tmp_path
    workflow._lock = threading.RLock()
    workflow._workflow = {}
    workflow._save_workflow = lambda: None
    workflow.export_prediction = lambda: internal
    app = SimpleNamespace(workflow=workflow)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_DesktopHandler, application=app)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for route in ("/download_prediction", "/download_excel"):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}{route}", timeout=5
            ) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                body = response.read()
            workbook = load_workbook(BytesIO(body), read_only=True)
            try:
                assert workbook.sheetnames == ["树木风险结果"]
                rows = list(workbook.active.values)
                assert rows[0] == DELIVERY_COLUMNS
                assert len(rows) == 2
            finally:
                workbook.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_legacy_internal_workbook_moves_out_of_user_exports_without_data_loss(tmp_path):
    legacy = tmp_path / "exports" / "Prediction_Input.xlsx"
    private = tmp_path / "artifacts" / "Prediction_Input.xlsx"
    legacy.parent.mkdir()
    legacy.write_bytes(b"old internal")
    assert migrate_legacy_prediction(legacy, private) == private
    assert private.read_bytes() == b"old internal"
    assert not legacy.exists()

    legacy.write_bytes(b"second old internal")
    assert migrate_legacy_prediction(legacy, private) == private
    assert private.read_bytes() == b"old internal"
    archived = list(private.parent.glob("Prediction_Input.legacy-*.xlsx"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"second old internal"
    assert not legacy.exists()

    legacy.write_bytes(b"pre-finalization internal")
    migrate_legacy_prediction(legacy, private, replacing=True)
    archived = list(private.parent.glob("Prediction_Input.legacy-*.xlsx"))
    assert sorted(path.read_bytes() for path in archived) == [
        b"pre-finalization internal", b"second old internal"
    ]
    assert private.read_bytes() == b"old internal"
