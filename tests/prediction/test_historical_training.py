from io import BytesIO

import pytest
from algorithms.prediction.historical_training import (
    fine_tune_with_history,
    parse_history_file,
    parse_history_text,
    training_template_csv,
)
from openpyxl import Workbook

HEADERS = (
    "tree_id\tspecies\tobservation_date\theight\tdiameter\t"
    "vertical_distance\thorizontal_distance\tclearance_distance\t"
    "wind_offset_distance\tvoltage_level_kv\tmaximum_sag_winter\t"
    "maximum_sag_summer\twind_speed\trainfall\thumidity\t"
    "temperature\tsunlight_hours\trisk_score"
)
ROWS = (
    "1\t杨树\t2026-01-15\t17.2\t33.2\t6.4\t6.2\t5.4\t0.8\t110\t"
    "3.5\t5\t6\t20\t60\t5\t5\t32\n"
    "1\t杨树\t2026-02-15\t17.22\t33.24\t6.39\t6.19\t5.39\t0.8\t110\t"
    "3.5\t5\t7\t28\t62\t8\t6\t35\n"
    "1\t杨树\t2026-03-15\t17.26\t33.31\t6.37\t6.18\t5.37\t0.9\t110\t"
    "3.5\t5\t9\t42\t66\t14\t7\t43"
)


def test_pasted_excel_table_is_normalized_and_trainable():
    parsed = parse_history_text(f"{HEADERS}\n{ROWS}", "tsv")

    assert parsed["source_format"] == "tsv"
    assert parsed["valid_row_count"] == 3
    assert parsed["invalid_row_count"] == 0
    assert parsed["tree_count"] == 1
    assert parsed["trainable_sequence_count"] == 2
    assert parsed["trainable"] is True
    assert parsed["records"][0]["species"] == "杨树"
    assert parsed["records"][0]["observation_month"] == 1
    assert parsed["records"][2]["risk_score"] == 43


def test_chinese_headers_and_risk_level_are_supported():
    text = (
        "树木编号,树种,观测日期,树高,胸径,垂直距离,水平距离,净空距离,"
        "风偏距离,风速,降雨量,湿度,温度,日照时长,风险等级\n"
        "9,松树,2026-01-01,12,24,8,8,7,1,5,10,55,2,4,低风险"
    )
    parsed = parse_history_text(text, "csv")

    assert parsed["valid_row_count"] == 1
    assert parsed["records"][0]["risk_score"] == 20
    assert parsed["records"][0]["voltage_level_kv"] == 110


def test_xlsx_file_upload_uses_first_sheet_with_data():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "月度巡检"
    sheet.append(HEADERS.split("\t"))
    for row in ROWS.splitlines():
        sheet.append(row.split("\t"))
    stream = BytesIO()
    workbook.save(stream)

    parsed = parse_history_file("历史巡检.xlsx", stream.getvalue())

    assert parsed["source_format"] == "xlsx"
    assert parsed["sheet_name"] == "月度巡检"
    assert parsed["valid_row_count"] == 3
    assert parsed["trainable"] is True


def test_invalid_rows_are_reported_without_hiding_valid_rows():
    text = (
        f"{HEADERS}\n"
        f"{ROWS.splitlines()[0]}\n"
        "2\t柳树\t错误日期\t18\t30\t6\t6\t5\t1\t110\t3.5\t5\t8\t20\t"
        "120\t20\t8\t55"
    )
    parsed = parse_history_text(text, "tsv")

    assert parsed["valid_row_count"] == 1
    assert parsed["invalid_row_count"] == 1
    assert parsed["errors"][0]["row"] == 3
    assert parsed["trainable"] is False


def test_template_can_be_reimported():
    parsed = parse_history_file(
        "treeline_history_training_template.csv",
        training_template_csv().encode("utf-8"),
    )
    assert parsed["valid_row_count"] == 3
    assert parsed["trainable_sequence_count"] == 2


@pytest.mark.parametrize("filename", ["history.xls", "history.xlsx.zip", "history.exe"])
def test_unsupported_file_types_are_rejected(filename):
    with pytest.raises(ValueError, match="支持"):
        parse_history_file(filename, b"not a supported workbook")


def test_fine_tuning_requires_an_independent_explicit_output_path(tmp_path):
    source_path = tmp_path / "input-model.npz"
    with pytest.raises(ValueError, match="output_path"):
        fine_tune_with_history([], model_path=source_path)
    with pytest.raises(ValueError, match="model_path"):
        fine_tune_with_history([], model_path=source_path, output_path=source_path)
    assert not source_path.exists()
