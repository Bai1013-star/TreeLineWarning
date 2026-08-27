from __future__ import annotations

from types import SimpleNamespace

from core.stage_b.danger_points import (
    build_clearance_evidence_display,
    build_danger_point_display,
    danger_points_path,
    load_final_danger_points,
    persist_final_danger_points,
)


def _record(
    *,
    tree_id: int = 1,
    clearance: float = 0.25,
    danger_count: int = 1,
    status: str = "VALID",
) -> SimpleNamespace:
    return SimpleNamespace(
        global_tree_id=tree_id,
        external_name=f"Tree-{tree_id}",
        persistent_tree_id=f"tree-{tree_id}",
        line_id=1,
        span_id=1,
        span_name="S001",
        nearest_wire_id=4,
        minimum_clearance_m=clearance,
        nearest_tree_original_index=100 + tree_id,
        nearest_tree_xyz_local_m=(1.0 + tree_id, 2.0, 3.0),
        nearest_wire_xyz_local_m=(1.0 + tree_id, 2.0, 3.0 + clearance),
        tree_point_count=50,
        evaluated_edge_count=10,
        status=status,
        danger_point_count=danger_count,
    )


def _danger(tree_id: int = 1, original_index: int = 101) -> SimpleNamespace:
    return SimpleNamespace(
        global_tree_id=tree_id,
        persistent_tree_id=f"tree-{tree_id}",
        original_index=original_index,
        xyz_local_m=(2.0, 2.0, 3.0),
    )


def _result(clearance, *, run_id: str = "analysis-danger") -> SimpleNamespace:
    tree_records = tuple(
        SimpleNamespace(global_tree_id=record.global_tree_id, review=False)
        for record in clearance.records
    )
    return SimpleNamespace(
        analysis_run_id=run_id,
        final_clearance=clearance,
        global_reconciled=SimpleNamespace(global_tree_records=tree_records),
        risk_policy_snapshot={
            "risk_policy_version": "risk-policy-v1",
            "warning_clearance_threshold_m": 0.5,
        },
    )


def test_persist_and_reload_exact_danger_points(tmp_path) -> None:
    clearance = SimpleNamespace(
        status="VALID",
        global_minimum_clearance_m=0.25,
        records=(_record(),),
        danger_points=(_danger(),),
    )

    path = persist_final_danger_points(tmp_path, "analysis-danger", clearance)
    arrays = load_final_danger_points(tmp_path, "analysis-danger")

    assert path is not None and path.is_file()
    assert arrays is not None
    original_index, tree_id, xyz = arrays
    assert original_index.tolist() == [101]
    assert tree_id.tolist() == [1]
    assert xyz.tolist() == [[2.0, 2.0, 3.0]]


def test_none_clearance_invalidates_persisted_danger_sidecar(tmp_path) -> None:
    clearance = SimpleNamespace(
        status="VALID",
        global_minimum_clearance_m=0.25,
        records=(_record(),),
        danger_points=(_danger(),),
    )
    path = persist_final_danger_points(tmp_path, "analysis-danger", clearance)
    assert path is not None and path.is_file()

    persisted = persist_final_danger_points(tmp_path, "analysis-danger", None)

    assert persisted is None
    assert load_final_danger_points(tmp_path, "analysis-danger") is None
    assert not danger_points_path(tmp_path, "analysis-danger").exists()
    assert not danger_points_path(tmp_path, "analysis-danger").with_suffix(".json").exists()


def test_current_zero_danger_never_resurrects_old_sidecar(tmp_path) -> None:
    old_clearance = SimpleNamespace(
        status="VALID",
        global_minimum_clearance_m=0.25,
        records=(_record(),),
        danger_points=(_danger(),),
    )
    persist_final_danger_points(tmp_path, "analysis-danger", old_clearance)

    current = SimpleNamespace(
        status="VALID",
        global_minimum_clearance_m=0.8,
        records=(_record(clearance=0.8, danger_count=0),),
        danger_points=(),
    )
    display = build_danger_point_display(
        tmp_path,
        _result(current),
    )

    assert display["full_point_count"] == 0
    assert display["xyz"] == []
    assert display["authority"] == "EXACT_FINAL_CLEARANCE_MEMORY"


def test_sidecar_count_mismatch_fails_closed(tmp_path) -> None:
    old_clearance = SimpleNamespace(
        status="VALID",
        global_minimum_clearance_m=0.25,
        records=(_record(),),
        danger_points=(_danger(),),
    )
    persist_final_danger_points(tmp_path, "analysis-danger", old_clearance)

    hydrated_without_points = SimpleNamespace(
        status="VALID",
        global_minimum_clearance_m=0.2,
        records=(_record(clearance=0.2, danger_count=2),),
        danger_points=(),
    )
    display = build_danger_point_display(
        tmp_path,
        _result(hydrated_without_points),
    )

    assert display["authority"] == "COUNT_MISMATCH"
    assert display["full_point_count"] == 0


def test_clearance_evidence_uses_exact_nearest_pair_and_risk_policy() -> None:
    clearance = SimpleNamespace(
        records=(
            _record(tree_id=1, clearance=0.25, danger_count=1),
            _record(tree_id=2, clearance=0.75, danger_count=0),
            _record(tree_id=3, clearance=0.0, danger_count=1),
        )
    )

    payload = build_clearance_evidence_display(_result(clearance))
    by_id = {row["global_tree_id"]: row for row in payload["records"]}

    assert payload["authority"] == "EXACT_FINAL_CLEARANCE_NEAREST_PAIR"
    assert by_id[1]["display_class"] == "WARNING"
    assert by_id[1]["tree_xyz"] == [2.0, 2.0, 3.0]
    assert by_id[1]["wire_xyz"] == [2.0, 2.0, 3.25]
    assert by_id[2]["display_class"] == "SAFE"
    # Exact zero is not itself CONTACT evidence in the current contract.
    assert by_id[3]["display_class"] == "REVIEW_REQUIRED"
    assert by_id[3]["contact_state"] == "CONTACT_UNCERTAIN"
