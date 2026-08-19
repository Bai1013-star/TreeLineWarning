# ruff: noqa: E501
"""Step-2 restart hydration: a completed workspace resumes without re-running science.

Regression for the folder server restart bug where ``GET /api/editor_data``
crashed with ``'NoneType' object has no attribute 'strands'`` because the wire
runtime object was never rehydrated from the persisted 05_wire artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from algorithms.folder_pipeline.orchestrator import (
    _write_editor_store,
)
from algorithms.folder_pipeline.state import STEP_TREE_EDITING
from algorithms.folder_pipeline.workspace import discover_workspace
from tests.folder_pipeline.conftest import make_workspace


def _selection_payload(origin_xyz: tuple[float, float, float]) -> dict:
    """A minimal valid persisted span selection (matches _pole_from_payload)."""

    def pole(label: str, x: float, y: float) -> dict:
        return {
            "label": label,
            "clicked_xyz_absolute_m": [x, y, 9.0],
            "base_xyz_local_m": [x - origin_xyz[0], y - origin_xyz[1], 1.0],
            "base_xyz_absolute_m": [x, y, 0.5],
            "top_xyz_local_m": [x - origin_xyz[0], y - origin_xyz[1], 13.0],
            "top_xyz_absolute_m": [x, y, 13.0],
            "height_m": 12.5,
            "xy_footprint_m": 0.5,
            "median_verticality": 0.8,
            "median_linearity": 0.2,
            "representative_count": 100,
            "ground_support_reps": 50,
            "defensible": True,
            "reason": None,
            "refinement_diagnostics": {},
        }

    pole_a = pole("A", 10.0, 10.0)
    pole_b = pole("B", 60.0, 10.0)
    return {
        "schema_version": "manual-single-span-selection-v1",
        "non_certified_marker": "NON_CERTIFIED_RESEARCH_DEMONSTRATION",
        "origin_xyz": list(origin_xyz),
        "pole_a": pole_a,
        "pole_b": pole_b,
        "span": {
            "length_m": 50.0,
            "direction_xy": [1.0, 0.0, 0.0],
            "half_width_m": 9.0,
            "halo_m": 6.0,
            "start_xyz_absolute_m": [10.0, 10.0, 0.5],
            "end_xyz_absolute_m": [60.0, 10.0, 0.5],
            "start_pole_top_z_m": 13.0,
            "end_pole_top_z_m": 13.0,
            "core_bounds_min_xyz": [10.0, 1.0, 0.0],
            "core_bounds_max_xyz": [60.0, 19.0, 16.0],
            "halo_bounds_min_xyz": [4.0, 1.0, 0.0],
            "halo_bounds_max_xyz": [66.0, 19.0, 16.0],
        },
        "start_tower_id": "T001",
        "end_tower_id": "T002",
    }


def _write_wire_artifacts(workspace_root: Path) -> None:
    science = workspace_root / "source_cache" / "science"
    # 05_wire/span_001 -> wire curves + result
    wire_dir = science / "05_wire" / "span_001"
    wire_dir.mkdir(parents=True, exist_ok=True)
    curve = np.asarray([[10.0, 10.0, 9.0], [35.0, 10.5, 8.5], [60.0, 10.0, 9.0]], dtype=np.float32)
    np.savez(
        wire_dir / "w1_arrays.npz",
        sampled_curve_xyz=curve,
        sampled_curve_offsets=np.asarray([0, 3], dtype=np.int64),
    )
    strand = {
        "wire_id": 0,
        "wire_name": "ManualLine-Span001-Wire01",
        "representative_count": 30,
        "raw_point_count": 100,
        "s_span_m": 50.0,
        "coverage_ratio": 1.0,
        "median_u_m": 0.2,
        "median_hag_m": 8.6,
        "curve_sample_count": 3,
        "residual_stats_m": [0.01, 0.03],
        "approach_start": True,
        "approach_end": True,
        "status": "CONFIRMED",
        "model_type": "ROBUST_QUADRATIC",
        "supported_s_domain_m": [0.0, 50.0],
        "supported_slice_count": 50,
        "supported_slice_ratio": 1.0,
        "maximum_gap_m": 1.0,
        "inlier_ratio": 1.0,
    }
    (wire_dir / "span_wire_result.json").write_text(
        json.dumps(
            {
                "schema": "fast-baseline-span-wire-v1",
                "marker": "FAST_MATURE_SPAN_WIRE",
                "strands": [strand],
                "strand_count": 1,
                "diagnostics": {
                    "fine_wire_representative_count": 30,
                    "wire_candidate_point_count": 30,
                    "confirmed_conductor_member_count": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    # load_span_wire_result reads this npz (candidate_mask + cluster ids +
    # confirmed member source rows).
    np.savez(
        wire_dir / "span_wire_diagnostics.npz",
        candidate_mask=np.asarray([True, True, True], dtype=bool),
        cluster_id_by_rep=np.asarray([0, 0, 0], dtype=np.int32),
        confirmed_member_source_index=np.asarray([100, 101, 102], dtype=np.int64),
    )
    # 04_regions/wire_region -> member display (tiny region)
    from algorithms.vegetation.coarse_to_fine import _RawRegionWriter, _write_region_sidecar

    region_root = science / "04_regions" / "wire_region"
    writer = _RawRegionWriter(region_root, origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.asarray([[10.0, 10.0, 9.0], [35.0, 10.5, 8.5], [60.0, 10.0, 9.0]], dtype=np.float64),
        np.asarray([100, 101, 102], dtype=np.int64),
        np.zeros(3, dtype=np.uint8),
    )
    region = writer.close()
    _write_region_sidecar(region, region_root / "region_sidecar.json")


def _write_display_sample(workspace_root: Path) -> None:
    sample_dir = workspace_root / "source_cache" / "science" / "00_input"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sample_dir / "display_sample_v4.npz",
        xyz_absolute=np.asarray([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0]], dtype=np.float64),
        rgb=np.zeros((2, 3), dtype=np.uint8),
        original_index=np.asarray([0, 1], dtype=np.int64),
        display_source=np.asarray(["SOURCE_RAW_PRE_DENOISE"]),
    )


def _write_editor_artifacts(workspace_root: Path, workspace) -> None:
    tree_dir = workspace_root / "02_tree_analysis"
    tree_dir.mkdir(parents=True, exist_ok=True)
    # Editor input: 2 trees, 5 rows total.
    original_index = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    tree_id = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    xyz = np.asarray([[10.0, 10.0, 5.0], [11.0, 10.0, 6.0], [12.0, 10.0, 7.0],
                      [13.0, 10.0, 6.0], [14.0, 10.0, 7.0]], dtype=np.float64)
    _write_editor_store(workspace, original_index, tree_id, xyz)
    # Compatibility npz (the run_science_if_needed completion marker).
    np.savez(
        tree_dir / "tree_segmentation_original.npz",
        original_index=original_index,
        tree_id=tree_id,
        xyz_local_m=xyz,
    )
    (tree_dir / "tree_segmentation_original.json").write_text(
        json.dumps(
            {
                "schema": "folder-to-excel-tree-segmentation-original-v1",
                "tree_backend": "crown_stem_hybrid",
                "algorithm_tree_count": 2,
                "source_algorithm_tree_count": 3,
                "trees": [
                    {"tree_id": 0, "external_name": "T001", "point_count": 3,
                     "apex_xy_local_m": [12.0, 10.0, 7.0], "apex_height_m": 7.0},
                    {"tree_id": 1, "external_name": "T002", "point_count": 2,
                     "apex_xy_local_m": [14.0, 10.0, 7.0], "apex_height_m": 7.0},
                ],
            }
        ),
        encoding="utf-8",
    )


def _build_completed_workspace(tmp_path: Path):
    """A minimal completed Step-2 workspace (selection + wire + tree + editor)."""

    workspace_root = make_workspace(tmp_path)
    workspace = discover_workspace(workspace_root)
    import laspy

    with laspy.open(workspace.source) as reader:
        origin_xyz = tuple(float(v) for v in reader.header.mins)
    sel_dir = workspace_root / "01_span_selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "span_selection.json").write_text(
        json.dumps(_selection_payload(origin_xyz)), encoding="utf-8"
    )
    _write_display_sample(workspace_root)
    _write_wire_artifacts(workspace_root)
    _write_editor_artifacts(workspace_root, workspace)
    return workspace_root, workspace


def test_completed_workspace_restart_hydrates_step2(tmp_path: Path):
    """A NEW server on a completed workspace reaches TREE_EDITING with editor_data OK."""
    from algorithms.folder_pipeline.server import FolderServer

    workspace_root, workspace = _build_completed_workspace(tmp_path)
    # Simulate a fresh process: a brand-new FolderServer against the workspace.
    server = FolderServer(workspace)
    snap = server.state.snapshot()
    assert snap["step"] == STEP_TREE_EDITING
    assert snap["error"] is None
    assert snap["selection_complete"] is True
    run = server.orchestrator.run
    # Wire runtime hydrated from the persisted artifact (the .strands fix).
    assert run.wire_outcome is not None
    assert len(run.wire_outcome.strands) == 1
    assert run.wire_region_dir is not None
    # Editor original trees hydrated from the memmap store.
    assert server.orchestrator.original_trees is not None
    assert len(server.orchestrator.original_trees.records) == 2

    # GET /api/editor_data payload (the method the handler serializes).
    payload = server._build_editor_display()
    assert set(payload) == {"trees", "trees3d", "background", "wire", "wires3d", "tower"}
    assert len(payload["trees3d"]["xyz"]) > 0          # tree display
    assert len(payload["trees"]) == 2                  # tree list
    assert len(payload["wires3d"]["xyz"]) > 0          # wire curves (confirmed)
    assert len(payload["wire"]["xyz"]) == 3            # wire member display
    assert len(payload["tower"]["xyz"]) == 2           # Pole A + B


def test_editor_store_loads_as_memmap(tmp_path: Path):
    workspace_root, workspace = _build_completed_workspace(tmp_path)
    from algorithms.folder_pipeline.orchestrator import _load_editor_store
    oi, ti, xyz = _load_editor_store(workspace)
    assert isinstance(xyz, np.memmap)
    assert isinstance(oi, np.memmap)
    assert oi.shape == (5,)


def test_fresh_and_partial_workspaces_fall_through_to_full_init(tmp_path: Path):
    """Non-completed workspaces are NOT diverted to the light restore path.

    The completed-workspace fast path must only trigger when the Step-2 editor
    artifact exists; otherwise the original full-initialize flow is preserved.
    """
    workspace_root = make_workspace(tmp_path)
    workspace = discover_workspace(workspace_root)
    # No original_tree_npz -> not "completed" -> the orchestrator still runs the
    # full initialize path (which requires a ground backend/cache; the assertion
    # here is that _prepare did not take the hydrate_completed_workspace branch).
    from algorithms.folder_pipeline.server import FolderServer
    server = FolderServer(workspace)
    assert server.state.step != STEP_TREE_EDITING
