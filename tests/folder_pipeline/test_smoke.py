"""Folder pipeline smoke tests.

- ``test_orchestrator_to_excel_smoke`` runs the human-edited final chain
  (editor -> final geometry -> exact clearance -> danger -> persistent id ->
  Prediction_Input.xlsx) on synthetic memberships with NO PDAL dependency.
- ``test_full_pipeline_headless_smoke`` runs the complete Folder->Excel chain
  (workspace -> pole clicks -> Wire V2 -> vegetation -> exact exclusion ->
  CROWN_STEM_HYBRID -> editor -> clearance -> Excel) and is skipped when the
  PDAL ground backend is unavailable on this machine.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.folder_pipeline.orchestrator import FolderOrchestrator
from algorithms.folder_pipeline.workspace import discover_workspace
from algorithms.poles.topology import SpanRecord
from algorithms.spans.wire import SpanWireResult
from openpyxl import load_workbook
from tests.folder_pipeline.conftest import (
    build_original_trees_fixture,
    make_workspace,
)

PDAL_AVAILABLE = importlib.util.find_spec("pdal") is not None
requires_pdal = pytest.mark.skipif(not PDAL_AVAILABLE, reason="PDAL ground backend unavailable")


def _make_wire_arrays(curves: list[np.ndarray]) -> dict[str, np.ndarray]:
    """W1-compatible distance-ready arrays (one segment per wire)."""

    wire_count = len(curves)
    wire_id = np.arange(wire_count, dtype=np.int32)
    wire_status = np.zeros(wire_count, dtype=np.uint8)
    wire_method = np.ones(wire_count, dtype=np.uint8)
    wire_quality_flags = np.zeros(wire_count, dtype=np.uint16)
    wire_has_ready = np.ones(wire_count, dtype=np.bool_)
    wire_continuous = np.ones(wire_count, dtype=np.bool_)
    segment_wire_id = np.arange(wire_count, dtype=np.int32)
    segment_index = np.zeros(wire_count, dtype=np.int32)
    segment_method = np.ones(wire_count, dtype=np.uint8)
    segment_quality_flags = np.zeros(wire_count, dtype=np.uint16)
    segment_distance_ready = np.ones(wire_count, dtype=np.bool_)
    segment_parameter_domain = np.asarray([[0.0, 1.0]] * wire_count, dtype=np.float64)
    sampled_curve_xyz = (
        np.concatenate(curves, axis=0).astype(np.float32)
        if wire_count
        else np.empty((0, 3), dtype=np.float32)
    )
    sample_counts = np.asarray([len(curve) for curve in curves], dtype=np.int64)
    sampled_curve_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(sample_counts)), axis=0
    )
    return {
        "wire_id": wire_id,
        "wire_status": wire_status,
        "wire_method": wire_method,
        "wire_quality_flags": wire_quality_flags,
        "wire_has_distance_ready_segments": wire_has_ready,
        "wire_continuous_distance_ready": wire_continuous,
        "wire_segment_offsets": np.arange(wire_count + 1, dtype=np.int64),
        "segment_wire_id": segment_wire_id,
        "segment_index": segment_index,
        "segment_method": segment_method,
        "segment_quality_flags": segment_quality_flags,
        "segment_distance_ready": segment_distance_ready,
        "segment_parameter_domain": segment_parameter_domain,
        "sampled_curve_xyz": sampled_curve_xyz,
        "sampled_curve_parameter": np.linspace(0.0, 1.0, int(sampled_curve_xyz.shape[0])),
        "sampled_curve_offsets": sampled_curve_offsets,
    }


def _fake_span() -> SpanRecord:
    return SpanRecord(
        line_id=0,
        line_name="ManualLine",
        span_id=0,
        span_name="ManualLine-Span001",
        start_pole_candidate_id=-1,
        end_pole_candidate_id=-2,
        start_xyz_absolute_m=(0.0, 0.0, 0.0),
        end_xyz_absolute_m=(100.0, 0.0, 0.0),
        start_pole_top_z_m=12.0,
        end_pole_top_z_m=12.0,
        length_m=100.0,
        direction_xy=(1.0, 0.0, 0.0),
        half_width_m=9.0,
        halo_m=6.0,
        core_bounds_min_xyz=(0.0, -9.0, 0.0),
        core_bounds_max_xyz=(100.0, 9.0, 20.0),
        halo_bounds_min_xyz=(0.0, -9.0, 0.0),
        halo_bounds_max_xyz=(106.0, 9.0, 20.0),
        quality="MANUAL_CONFIRMED_SPAN",
    )


def _fake_run(workspace: Path) -> SimpleNamespace:
    from algorithms.manual_span.config import (
        default_manual_span_config,
    )
    from core.config.fast_baseline import default_fast_baseline_config

    s = np.linspace(0.0, 100.0, 120)
    curve = np.column_stack((s, np.full_like(s, 0.0), 11.0 - 0.02 * (s - 50.0) ** 2))
    return SimpleNamespace(
        origin_xyz=(0.0, 0.0, 0.0),
        span=_fake_span(),
        wire_outcome=SpanWireResult(
            strands=(),
            w1_arrays=_make_wire_arrays([curve]),
            candidate_mask=np.empty(0, dtype=bool),
            cluster_id_by_rep=np.empty(0, dtype=np.int32),
            diagnostics={"wire_candidate_point_count": 0, "wire_cluster_count": 0},
            confirmed_member_source_index=np.arange(200, dtype=np.int64),
        ),
        coarse_ground_xyz_absolute=np.asarray(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]] * 6,
            dtype=np.float64,
        ),
        cfg=default_fast_baseline_config(),
        manual=default_manual_span_config(),
    )


def _persist_selection(workspace: Path) -> None:
    selection_dir = workspace / "01_span_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_id": "folder_smoke",
        "group_id": "G001",
        "span_id": "S001",
        "source_sha256": "smoke",
        "source_identity": "smoke-source",
        "origin_xyz": [0.0, 0.0, 0.0],
        "pole_a": {"base_xyz_absolute_m": [0.0, 0.0, 0.0]},
        "pole_b": {"base_xyz_absolute_m": [100.0, 0.0, 0.0]},
        "created_at": "2026-01-01T00:00:00",
    }
    (selection_dir / "span_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _persist_original_trees(workspace: Path, original_trees) -> None:
    tree_dir = workspace / "02_tree_analysis"
    tree_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        tree_dir / "tree_segmentation_original.npz",
        original_index=original_trees.original_index,
        tree_id=original_trees.tree_id,
        xyz_local_m=original_trees.xyz_local_m,
    )
    records = [
        {
            "tree_id": record.tree_id,
            "external_name": record.external_name,
            "point_count": record.point_count,
            "apex_xy_local_m": (
                None if record.apex_xy_local_m is None else list(record.apex_xy_local_m)
            ),
            "apex_height_m": record.apex_height_m,
        }
        for record in original_trees.records
    ]
    (tree_dir / "tree_segmentation_original.json").write_text(
        json.dumps(
            {"schema": "folder-to-excel-tree-segmentation-original-v1",
             "tree_backend": "crown_stem_hybrid", "trees": records},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_orchestrator_to_excel_smoke(tmp_path: Path) -> None:
    """Editor -> final geometry -> clearance -> risk -> identity -> Excel (no PDAL)."""

    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    original = build_original_trees_fixture()
    _persist_selection(workspace)
    _persist_original_trees(workspace, original)

    orchestrator = FolderOrchestrator(ws, threshold_m=3.5)
    orchestrator.run = _fake_run(workspace)  # inject the science boundary

    # STEP 2: load the editor from the persisted algorithm result.
    session = orchestrator.build_editor()
    assert session.original.count == 4
    # Human edits: delete tree 3, merge trees 0 and 1.
    session.select([3])
    session.delete_selected()
    session.select([0, 1])
    session.merge_selected()
    snapshot = orchestrator.editor_snapshot()
    assert snapshot["current_tree_count"] == 2  # merged + tree 2

    # STEP 3: finalize (clearance + metrics + risk + identity + Excel).
    summary = orchestrator.confirm_trees_and_finalize()
    assert summary["RUN_STATUS"] == "PASS"
    assert summary["FINAL_TREE_COUNT"] == 2
    excel = ws.excel_path
    assert excel.is_file()

    workbook = load_workbook(excel)
    assert workbook.sheetnames == ["Trees", "Run_Metadata"]
    sheet = workbook["Trees"]
    ids = [sheet.cell(row=r, column=1).value for r in range(2, sheet.max_row + 1)]
    ids = [value for value in ids if value]
    assert len(ids) == 2
    assert len(set(ids)) == 2  # PERSISTENT_TREE_ID_DUPLICATES == 0

    # Final membership conserved: no point lost, no duplicate.
    final_index = orchestrator.final_reconciled.final_original_index
    assert len(np.unique(final_index)) == len(final_index)
    # Deleted tree 3's points are absent; merged union is present.
    assert not set(original.members_of(3)).intersection(final_index)

    # Run metadata reports the exact policies.
    meta = orchestrator.run_metadata
    assert meta["tree_backend"] == "crown_stem_hybrid"
    assert meta["wire_exclusion_policy"] == "EXACT_ORIGINAL_INDEX"
    assert meta["algorithm_tree_count"] == 4
    assert meta["deleted_tree_count"] == 1
    assert meta["merge_operation_count"] == 1
    assert meta["final_tree_count"] == 2
    # (wire_tree_input_intersection is a science-stage audit key, verified in
    # test_exclusion.py; this resume-path smoke loads persisted trees instead.)


def test_editor_persists_edits_and_resumes(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    ws = discover_workspace(workspace)
    original = build_original_trees_fixture()
    _persist_selection(workspace)
    _persist_original_trees(workspace, original)

    orchestrator = FolderOrchestrator(ws)
    session = orchestrator.build_editor()
    session.select([1])
    session.delete_selected()
    orchestrator._persist_edits(session)
    assert ws.tree_edits_path.is_file()

    # A fresh orchestrator resumes the same edit state from disk.
    orchestrator2 = FolderOrchestrator(ws)
    session2 = orchestrator2.build_editor()
    assert session2.current_tree_ids() == [0, 2, 3]
    assert session2.operations[0]["type"] == "DELETE"


@requires_pdal
def test_full_pipeline_headless_smoke(tmp_path: Path) -> None:
    """Complete Folder->Excel chain on a synthetic scene (needs PDAL ground)."""

    from algorithms.folder_pipeline.server import FolderServer
    from algorithms.folder_pipeline.state import STEP_DONE
    from tests.vegetation.test_fast_baseline_e2e import _build_synthetic_scene, _write_las

    source = tmp_path / "scene.las"
    _write_las(source, _build_synthetic_scene())
    workspace = make_workspace(tmp_path)
    pointcloud = workspace / "input" / "pointcloud"
    (pointcloud / "source.las").write_bytes(source.read_bytes())

    server = FolderServer(
        discover_workspace(workspace),
        headless_poles=((0.0, -50.0, 6.0), (50.0, -50.0, 6.0)),
    )
    deadline = 600
    elapsed = 0.0
    while elapsed < deadline:
        state = server.state.snapshot()
        if state["step"] == STEP_DONE:
            break
        assert state["step"] != "ERROR", state.get("error")
        __import__("time").sleep(1)
        elapsed += 1
    state = server.state.snapshot()
    assert state["step"] == STEP_DONE
    summary = state["final_summary"]
    assert summary["FINAL_TREE_COUNT"] >= 1
    assert server.workspace.excel_path.is_file()
    workbook = load_workbook(server.workspace.excel_path)
    assert workbook.sheetnames == ["Trees", "Run_Metadata"]
