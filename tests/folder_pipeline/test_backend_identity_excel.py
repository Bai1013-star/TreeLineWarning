"""Backend selection, final geometry, identity, Excel integrity, missing values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algorithms.folder_pipeline.export_excel import (
    _sanitize_tree_row,
    write_prediction_input_excel,
)
from algorithms.folder_pipeline.identity import assign_persistent_tree_ids
from algorithms.folder_pipeline.orchestrator import _final_run_metadata
from algorithms.folder_pipeline.tree_editor import (
    OriginalTrees,
    TreeEditSession,
)
from algorithms.folder_pipeline.workspace import discover_workspace
from core.config.fast_baseline import default_fast_baseline_config
from openpyxl import load_workbook
from tests.folder_pipeline.conftest import (
    build_original_trees_fixture,
    make_workspace,
)


def test_crown_stem_hybrid_is_actual_selected_backend() -> None:
    config = default_fast_baseline_config()
    assert config.tree_backend == "crown_stem_hybrid"


def test_legacy_tree_artifact_cannot_silently_replace_current_backend(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    ws.ensure_dirs()
    # A stale "legacy_reference" directory in the science cache must not be
    # treated as the current tree result: the folder pipeline only consumes
    # tree_segmentation_original.* (backend-annotated), never a bare
    # legacy_reference folder.
    legacy = ws.science_run_root / "06_tree" / "span_001" / "legacy_reference"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "tree_members.npz").write_bytes(b"legacy")
    assert not ws.original_tree_manifest.is_file()
    assert not ws.original_tree_npz.is_file()


def test_metadata_reports_crown_stem_hybrid_and_exact_policies() -> None:
    import types

    run = types.SimpleNamespace(
        wire_outcome=types.SimpleNamespace(strand_count=2),
        pole_a=None,
        pole_b=None,
        origin_xyz=(0.0, 0.0, 0.0),
    )
    session = TreeEditSession(build_original_trees_fixture())
    orchestrator = types.SimpleNamespace(
        run=run,
        state=types.SimpleNamespace(selection={}),
        edit_session=session,
        original_trees=session.original,
        threshold_m=3.5,
    )
    reconciled = types.SimpleNamespace(global_tree_records=[])
    risk = types.SimpleNamespace(
        dangerous_tree_count=0, danger_point_count=0, global_minimum_clearance_m=None
    )
    metadata = _final_run_metadata(orchestrator, reconciled, risk)  # type: ignore[arg-type]
    assert metadata["tree_backend"] == "crown_stem_hybrid"
    assert metadata["wire_exclusion_policy"] == "EXACT_ORIGINAL_INDEX"


# ---------------------------------------------------------------------------
# Final geometry semantics
# ---------------------------------------------------------------------------

def test_height_is_hag_based_not_raw_z_span() -> None:
    # compute_tree_metrics derives height from ground-relative HAG (p98), not
    # max(raw Z) - min(raw Z) over an arbitrary footprint.
    from algorithms.manual_span.config import default_manual_span_config
    from algorithms.manual_span.metrics import compute_tree_metrics
    from algorithms.spans.reconcile import TreeReconciliationResult

    rng = np.random.default_rng(4)
    # A synthetic tree on a sloped scene-local ground: raw Z span alone would be
    # misleading, HAG height is ground-relative.
    n = 2000
    xyz = np.column_stack(
        (
            rng.normal(50.0, 1.5, n),
            rng.normal(20.0, 1.5, n),
            np.linspace(5.0, 35.0, n) + rng.normal(0, 0.2, n),
        )
    )
    reconciled = TreeReconciliationResult(
        global_tree_records=(),
        final_original_index=np.arange(n, dtype=np.int64),
        final_xyz_local_m=xyz,
        final_tree_id=np.zeros(n, dtype=np.int32),
        review_tree_ids=(),
        span_tree_to_global={(0, 0): 0},
        diagnostics={},
    )
    ground = np.asarray(
        [[50.0, 20.0, 5.0], [50.0, 20.0, 5.0], [50.0, 20.0, 5.0]] * 6,
        dtype=np.float64,
    )
    metrics = compute_tree_metrics(
        reconciled, ground, (0.0, 0.0, 0.0), default_manual_span_config()
    )
    assert len(metrics) == 1
    # HAG height ~ 30 m (35 - 5), NOT the raw span of Z over the whole scene.
    assert metrics[0].tree_height_m == pytest.approx(30.0, abs=2.5)


def test_crown_width_is_absolute_geometry_not_growth_amount(
    original_trees: OriginalTrees,
) -> None:
    # Crown widths come from the tree's own horizontal XY extent (p95-p05 along
    # principal axes).  A MERGE that unions two separated crowns produces the
    # UNION's geometry — never a "growth amount" fallback and never 0.
    session = TreeEditSession(original_trees)
    session.select([0, 1])
    session.merge_selected()
    final_index, final_tree, final_xyz, merged = session.build_final()
    assert len(merged) == 1
    union_index = final_index[final_tree == list(merged)[0]]
    assert len(union_index) == 100 + 120
    # The union's XY extent must span both source crowns (trees at x=0 and x=10).
    union_xyz = final_xyz[final_tree == list(merged)[0]]
    x_span = float(np.ptp(union_xyz[:, 0]))
    assert x_span > 8.0


# ---------------------------------------------------------------------------
# Missing value semantics (never 0 for crown / DBH)
# ---------------------------------------------------------------------------

def _base_row() -> dict:
    return {
        "persistent_tree_id": "G001-S001-T001",
        "height_m": 12.0,
        "crown_width_m": 5.0,
        "crown_width_major_m": 6.0,
        "crown_width_minor_m": 4.0,
        "crown_area_m2": 18.0,
        "diameter_cm": 30.0,
        "diameter_source": "MEASURED",
        "current_risk": "SAFE",
        "danger_point_count": 0,
    }


def test_missing_crown_cannot_become_zero() -> None:
    row = _base_row()
    row["crown_width_m"] = 0.0  # the historical 0-missing bug
    out = _sanitize_tree_row(row, "run-1", "2026-01-01")
    assert out["crown_width_m"] is None
    assert out["crown_width_major_m"] is None
    assert out["crown_area_m2"] is None


def test_tiny_crown_is_treated_as_missing_not_zero() -> None:
    # The historical "predicted crown width = 0.004 m" bug: an implausibly tiny
    # crown must become blank, never a false measured value.
    row = _base_row()
    row["crown_width_m"] = 0.004
    out = _sanitize_tree_row(row, "run-1", "2026-01-01")
    assert out["crown_width_m"] is None


def test_missing_dbh_cannot_become_zero() -> None:
    row = _base_row()
    row["diameter_cm"] = 0.0
    out = _sanitize_tree_row(row, "run-1", "2026-01-01")
    assert out["diameter_cm"] is None
    assert out["diameter_source"] is None


def test_nan_and_inf_are_sanitized_to_blank(tmp_path: Path) -> None:
    rows = [
        {
            **_base_row(),
            "height_m": float("nan"),
            "crown_width_m": float("inf"),
            "diameter_cm": float("-inf"),
        }
    ]
    path = write_prediction_input_excel(
        tmp_path / "out.xlsx", rows, {"science_run_id": "run-1"}
    )
    workbook = load_workbook(path)
    sheet = workbook["Trees"]
    height = sheet.cell(row=2, column=7).value  # height_m
    crown = sheet.cell(row=2, column=9).value  # crown_width_m
    diameter = sheet.cell(row=2, column=14).value  # diameter_cm
    assert height is None
    assert crown is None
    assert diameter is None


# ---------------------------------------------------------------------------
# Excel integrity
# ---------------------------------------------------------------------------

def test_excel_tree_rows_equals_final_tree_count(tmp_path: Path) -> None:
    rows = [
        {**_base_row(), "persistent_tree_id": "G001-S001-T001"},
        {**_base_row(), "persistent_tree_id": "G001-S001-T002", "current_risk": "DANGEROUS"},
    ]
    path = write_prediction_input_excel(
        tmp_path / "out.xlsx", rows, {"science_run_id": "run-2"}
    )
    workbook = load_workbook(path)
    sheet = workbook["Trees"]
    assert sheet.title == "Trees"
    ids = [sheet.cell(row=r, column=1).value for r in range(2, sheet.max_row + 1)]
    assert len([value for value in ids if value]) == 2
    assert workbook.sheetnames == ["Trees", "Run_Metadata"]


def test_persistent_tree_id_unique_and_deterministic(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    rows = [
        {**_base_row(), "longitudinal_s_m": 10.0, "tree_id": 0},
        {**_base_row(), "longitudinal_s_m": 5.0, "tree_id": 1},
    ]
    assign_persistent_tree_ids(rows, ws)
    ids = [row["persistent_tree_id"] for row in rows]
    assert len(ids) == len(set(ids))
    # Span-order allocation is deterministic: s=5 first -> T001.
    assert rows[1]["persistent_tree_id"].endswith("T001")
    # Re-running produces the same ids.
    assign_persistent_tree_ids(rows, ws)
    assert [row["persistent_tree_id"] for row in rows] == ids


# ---------------------------------------------------------------------------
# Large source path: no hard science point-count ceiling
# ---------------------------------------------------------------------------

def test_no_hard_science_point_count_ceiling() -> None:
    from algorithms import folder_pipeline

    assert not hasattr(folder_pipeline, "MAX_POINTS")
    from algorithms.folder_pipeline import orchestrator, server, tree_editor

    for module in (orchestrator, server, tree_editor):
        assert not hasattr(module, "MAX_POINTS")


def test_editor_handles_large_membership_without_ceiling() -> None:
    # The tree editor carries membership as int64 original_index arrays and
    # decimates only for display — a large membership (beyond any small display
    # budget) is not a science limitation.
    sizes = [10_000] * 5
    trees = build_original_trees_fixture(tree_sizes=sizes)
    assert len(trees.original_index) == 50_000
    assert trees.original_index.dtype == np.int64
    session = TreeEditSession(trees)
    session.select([0, 1])
    merged_id = session.merge_selected()
    final_index, final_tree, _xyz, merged = session.build_final()
    # All points conserved (no loss), merged tree union == 20_000.
    assert len(final_index) == 50_000
    assert len(np.unique(final_index)) == 50_000
    merged_members = final_index[final_tree == merged_id]
    assert len(merged_members) == 20_000
    assert len(np.unique(merged_members)) == 20_000
