from types import SimpleNamespace

import pytest
from core.stage_b.measurement_fields import clearance_components, dbh_fields


def pair(tree=(0, 0, 0), wire=(3, 4, 12), minimum=13, status="VALID"):
    return SimpleNamespace(
        nearest_tree_xyz_local_m=tree,
        nearest_wire_xyz_local_m=wire,
        minimum_clearance_m=minimum,
        status=status,
    )


def test_components_use_same_authoritative_nearest_pair():
    fields = clearance_components(pair())
    assert fields["horizontal_clearance_m"] == 5
    assert fields["vertical_clearance_m"] == 12
    assert fields["clearance_component_source"] == "EXACT_3D_NEAREST_PAIR_COMPONENTS"
    zero = clearance_components(pair(tree=(3, 4, 12), minimum=0))
    assert zero["horizontal_clearance_m"] == zero["vertical_clearance_m"] == 0


@pytest.mark.parametrize(
    "record",
    [
        None,
        pair(wire=None),
        pair(minimum=1),
        pair(status="BLOCKED"),
        pair(wire=(float("nan"), 0, 0)),
        pair(minimum=float("inf")),
    ],
)
def test_missing_or_inconsistent_proof_is_never_zero(record):
    assert all(value is None for value in clearance_components(record).values())


def test_bound_dbh_recovery_survives_missing_ordinary_metrics():
    fields = dbh_fields(
        None,
        {
            "dbh_cm": 31.2,
            "dbh_method": "RECOVERED_STEM",
            "dbh_quality": "MEASURED",
            "dbh_fit_status": "PASS",
        },
    )
    assert fields["dbh_cm"] == fields["diameter_cm"] == 31.2
    assert fields["dbh_method"] == "RECOVERED_STEM"
    missing = dbh_fields(None, None)
    assert missing["dbh_cm"] is None
    assert missing["dbh_fit_status"] == "GROUND_CONTEXT_UNAVAILABLE"


def test_workbook_and_report_keep_measurements_and_missing_values(tmp_path):
    from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
    from desktop.current_report import _current_row
    from openpyxl import load_workbook

    row = {
        "persistent_tree_id": "measured-tree",
        **clearance_components(pair()),
        **dbh_fields(None, {"dbh_cm": 31.2, "dbh_method": "RECOVERED_STEM"}),
    }
    missing = {
        "persistent_tree_id": "unmeasured-tree",
        **clearance_components(None),
        **dbh_fields(None, None),
    }
    assert _current_row(row)["horizontal_clearance_m"] == 5
    assert _current_row(row)["vertical_clearance_m"] == 12
    path = write_prediction_input_excel(tmp_path / "measurements.xlsx", [row, missing], {})
    book = load_workbook(path, read_only=True)
    try:
        values = list(book["Trees"].values)
        measured, unavailable = [dict(zip(values[0], v, strict=True)) for v in values[1:]]
        assert measured["dbh_cm"] == 31.2
        assert measured["horizontal_clearance_m"] == 5
        assert measured["vertical_clearance_m"] == 12
        assert unavailable["dbh_cm"] is None
        assert unavailable["horizontal_clearance_m"] is None
        assert unavailable["vertical_clearance_m"] is None
        assert unavailable["dbh_fit_status"] == "GROUND_CONTEXT_UNAVAILABLE"
    finally:
        book.close()
