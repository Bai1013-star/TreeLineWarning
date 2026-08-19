"""Focused tests for the pre-real-run semantic fixes.

Covers: tower-core provenance honesty, span-independent persistent tree id,
separate span_id export, canonical danger threshold authority, no new
Folder-pipeline hard-coded threshold, and source-identity-validated ground
cache reuse.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from algorithms.folder_pipeline import orchestrator as orch
from algorithms.folder_pipeline.identity import assign_persistent_tree_ids
from algorithms.folder_pipeline.orchestrator import (
    FolderOrchestrator,
    _canonical_risk_threshold_m,
    _find_prior_ground_cache,
)
from algorithms.folder_pipeline.workspace import discover_workspace
from algorithms.spans.project_adapter import RISK_THRESHOLD_M
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter
from tests.folder_pipeline.conftest import make_workspace


def _write_region(root: Path, xyz: np.ndarray, indices: np.ndarray):
    writer = _RawRegionWriter(root, origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.asarray(xyz, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
        np.zeros(len(xyz), dtype=np.uint8),
    )
    return writer.close()


class _StubRawRegion:
    def __init__(self, count: int) -> None:
        self._idx = np.arange(count, dtype=np.int64)
        self.count = count

    def original_index(self):
        return self._idx


def _stub_run():
    def local_raw_for_xy(x, y, radius=None):  # noqa: ANN001, ANN202
        del radius
        xyz = np.asarray([[x, y, 5.0], [x + 0.1, y, 9.0], [x + 2.0, y, 9.0]])
        return {
            "xyz_absolute": xyz,
            "original_index": np.asarray([100, 101, 500], dtype=np.int64),
        }

    return SimpleNamespace(
        wire_outcome=SimpleNamespace(
            confirmed_member_source_index=np.asarray([0, 1, 2], dtype=np.int64)
        ),
        pole_a=SimpleNamespace(
            base_xyz_absolute_m=(0.0, 0.0, 0.0), top_xyz_absolute_m=(0.0, 0.0, 10.0)
        ),
        pole_b=SimpleNamespace(
            base_xyz_absolute_m=(100.0, 0.0, 0.0), top_xyz_absolute_m=(100.0, 0.0, 10.0)
        ),
        local_raw_for_xy=local_raw_for_xy,
    )


# ---------------------------------------------------------------------------
# 1. Tower exclusion provenance (derived core is NOT full authoritative)
# ---------------------------------------------------------------------------

def test_derived_tower_core_not_labeled_authoritative(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    veg = _write_region(tmp_path / "veg", np.zeros((20, 3)), np.arange(20, dtype=np.int64))
    instance = FolderOrchestrator(ws)
    instance.run = _stub_run()
    _excluded, tower_source = instance._build_tree_input_exclusion(veg, None)
    assert len(tower_source) > 0
    # Honest provenance: derived shaft core, PARTIAL coverage, NOT authoritative.
    assert (
        instance.run_metadata["tower_exclusion_policy"]
        == "DERIVED_POLE_SHAFT_CORE_EXACT_INDICES"
    )
    assert instance.run_metadata["tower_exclusion_coverage"] == "PARTIAL"
    assert instance.run_metadata["tower_membership_source"] == "DERIVED_GEOMETRIC_SHAFT_CORE"
    assert instance.run_metadata["tower_authoritative_membership"] == "NO"
    # Exact original_index subtraction is still used; the derived region is tight.
    assert set(tower_source) == {100, 101}  # z=12 point (500) is outside the shaft


def test_no_authoritative_tower_membership_is_invented(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    # No authoritative artifacts exist for this source -> None (never fabricated).
    assert orch._authoritative_tower_membership(ws) is None


# ---------------------------------------------------------------------------
# 2. persistent_tree_id is span-independent; 3. span_id exported separately
# ---------------------------------------------------------------------------

def _row(longitudinal_s_m: float, tree_id: int) -> dict:
    return {
        "tree_id": tree_id,
        "height_m": 12.0,
        "crown_width_m": 5.0,
        "diameter_cm": 30.0,
        "current_risk": "SAFE",
        "danger_point_count": 0,
        "longitudinal_s_m": longitudinal_s_m,
    }


def test_persistent_tree_id_does_not_embed_span(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    (ws.span_selection_dir).mkdir(parents=True, exist_ok=True)
    (ws.span_selection_path).write_text(
        json.dumps({"group_id": "G001", "span_id": "S001", "project_id": "p"}), encoding="utf-8"
    )
    rows = [_row(10.0, 0), _row(5.0, 1), _row(20.0, 2)]
    assign_persistent_tree_ids(rows, ws)
    for row in rows:
        assert re.fullmatch(r"G001-T\d{3}", row["persistent_tree_id"]) is not None
        assert "S" not in row["persistent_tree_id"]  # span never embedded
        assert row["span_id"] == "S001"  # span stored separately
    ids = [row["persistent_tree_id"] for row in rows]
    assert len(set(ids)) == 3  # unique among final trees


def test_span_id_remains_separately_exported_in_excel(tmp_path: Path) -> None:
    from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
    from openpyxl import load_workbook

    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    (ws.span_selection_dir).mkdir(parents=True, exist_ok=True)
    (ws.span_selection_path).write_text(
        json.dumps({"group_id": "G001", "span_id": "S005", "project_id": "p"}), encoding="utf-8"
    )
    rows = [_row(5.0, 0), _row(15.0, 1)]
    assign_persistent_tree_ids(rows, ws)
    path = write_prediction_input_excel(
        tmp_path / "out.xlsx", rows, {"science_run_id": "run"}
    )
    workbook = load_workbook(path)
    sheet = workbook["Trees"]
    # Column 1 = persistent_tree_id, column 4 = span_id.
    ids = [sheet.cell(row=r, column=1).value for r in range(2, sheet.max_row + 1)]
    spans = [sheet.cell(row=r, column=4).value for r in range(2, sheet.max_row + 1)]
    for tree_id, span_id in zip(ids, spans, strict=True):
        assert re.fullmatch(r"G001-T\d{3}", tree_id)
        assert span_id == "S005"  # separate column, not embedded


def test_merged_tree_receives_one_valid_final_persistent_identity(tmp_path: Path) -> None:
    from algorithms.folder_pipeline.tree_editor import TreeEditSession
    from tests.folder_pipeline.conftest import build_original_trees_fixture

    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    (ws.span_selection_dir).mkdir(parents=True, exist_ok=True)
    (ws.span_selection_path).write_text(
        json.dumps({"group_id": "G001", "span_id": "S001", "project_id": "p"}), encoding="utf-8"
    )
    trees = build_original_trees_fixture()
    session = TreeEditSession(trees)
    session.select([0, 1])
    session.merge_selected()
    final_index, final_tree, _xyz, _merged = session.build_final()
    # Build final rows: one per final tree id.
    final_ids = sorted({int(v) for v in set(final_tree)})
    rows = [
        {**_row(float(i), tid), "longitudinal_s_m": float(i)}
        for i, tid in enumerate(final_ids)
    ]
    assign_persistent_tree_ids(rows, ws)
    assert len(rows) == len({row["persistent_tree_id"] for row in rows})
    assert all(re.fullmatch(r"G001-T\d{3}", row["persistent_tree_id"]) for row in rows)


def test_deleted_trees_are_not_exported(tmp_path: Path) -> None:
    from algorithms.folder_pipeline.tree_editor import TreeEditSession
    from tests.folder_pipeline.conftest import build_original_trees_fixture

    del tmp_path
    trees = build_original_trees_fixture()
    session = TreeEditSession(trees)
    session.select([2])
    session.delete_selected()
    final_ids = set(session.current_tree_ids())
    assert 2 not in final_ids
    assert set(final_ids) == {0, 1, 3}


# ---------------------------------------------------------------------------
# 4. Danger threshold authority; 5. no new hard-coded threshold
# ---------------------------------------------------------------------------

def test_danger_threshold_comes_from_canonical_standard() -> None:
    from algorithms.spans.project_adapter import RISK_CONFIGURATION_DIGEST

    assert _canonical_risk_threshold_m() == RISK_THRESHOLD_M
    assert RISK_CONFIGURATION_DIGEST  # canonical standard is digest-bound


def test_folder_default_threshold_is_canonical(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    instance = FolderOrchestrator(ws)  # no explicit threshold
    assert instance.threshold_m == RISK_THRESHOLD_M


def test_folder_pipeline_owns_no_hardcoded_threshold_default() -> None:
    import inspect

    source = inspect.getsource(orch)
    assert "DEFAULT_RISK_THRESHOLD_M" not in source
    # The only threshold default is the canonical standard import.
    assert "RISK_THRESHOLD_M" in source


# ---------------------------------------------------------------------------
# 6. Ground cache reuse requires matching source identity
# ---------------------------------------------------------------------------

def _make_cache_run(root: Path, name: str, source_sha: str) -> Path:
    run_dir = root / name
    (run_dir / "00_input").mkdir(parents=True, exist_ok=True)
    (run_dir / "02_ground").mkdir(parents=True, exist_ok=True)
    (run_dir / "00_input" / "source_inspection.json").write_text(
        json.dumps({"input_sha256": source_sha}), encoding="utf-8"
    )
    (run_dir / "02_ground" / "ground_hag_result.json").write_text("{}", encoding="utf-8")
    return run_dir / "02_ground"


def test_ground_cache_reuse_requires_matching_source_identity(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    roots = tmp_path / "runs"
    # A cache for a DIFFERENT source must never be reused.
    _make_cache_run(roots, "other_source", "deadbeef" * 8)
    assert _find_prior_ground_cache(ws, roots=[roots]) is None
    # A cache for THIS source (same SHA256) is reused.
    matching = _make_cache_run(roots, "same_source", ws.source_sha256)
    assert _find_prior_ground_cache(ws, roots=[roots]) == matching
    # A cache directory without source inspection is never used.
    (roots / "no_inspection" / "02_ground").mkdir(parents=True, exist_ok=True)
    assert _find_prior_ground_cache(ws, roots=[roots]) == matching
