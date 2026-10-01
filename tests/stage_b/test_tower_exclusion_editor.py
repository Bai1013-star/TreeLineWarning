"""Legacy tree reviews retain user edits while tower identities are removed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from algorithms.folder_pipeline.tree_editor import (
    OriginalTree,
    OriginalTrees,
    _encode_indices,
    load_edits_payload,
)
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from algorithms.spans.tower_exclusion import (
    TOWER_TREE_EXCLUSION_VERSION,
    TowerTreeExclusion,
    publish_tower_membership,
)
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter
from core.stage_b.dbh_evidence import editor_eligibility
from core.stage_b.editor import BatchTreeEditor
from core.stage_b.region_edit_session import RegionTreeEditSession
from core.stage_b.tower_exclusion import (
    TOWER_EDITOR_SELECTION_METHOD,
    _span_tower_indices,
    ensure_editor_tower_exclusion,
)


def _session() -> RegionTreeEditSession:
    original = OriginalTrees(
        records=(
            OriginalTree(0, "Tree001", 3, (0.0, 0.0, 10.0), 10.0),
            OriginalTree(1, "Tree002", 2, (3.0, 0.0, 9.0), 9.0),
        ),
        original_index=np.asarray([10, 11, 12, 20, 21], dtype=np.int64),
        tree_id=np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        xyz_local_m=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 5.0], [0.0, 0.0, 10.0], [3.0, 0.0, 5.0], [3.0, 0.0, 9.0]],
            dtype=np.float64,
        ),
    )
    return RegionTreeEditSession(original)


def _project_artifacts(tmp_path):
    project = tmp_path / "project"
    batch = project / "artifacts" / "project-runs" / "batches" / "run"
    span = batch / "span-a"
    source_identity = {"sha256": "a" * 64}
    span.mkdir(parents=True)
    (span / "span-cache-index.json").write_text(
        json.dumps(
            {
                "input": {
                    "source_identity": source_identity,
                    "from_tower_id": "T001",
                    "to_tower_id": "T002",
                    "from_endpoint_geometry": {
                        "base_xyz_absolute_m": [0.0, 0.0, 0.0],
                        "top_xyz_absolute_m": [0.0, 0.0, 12.0],
                    },
                    "to_endpoint_geometry": {
                        "base_xyz_absolute_m": [10.0, 0.0, 0.0],
                        "top_xyz_absolute_m": [10.0, 0.0, 12.0],
                    },
                },
                "source_preparation": {"ground_preparation": {"origin_xyz": [0, 0, 0]}},
            }
        ),
        encoding="utf-8",
    )
    publish_tower_membership(
        TowerTreeExclusion(
            np.asarray([10, 11, 12], dtype=np.int64),
            (
                {
                    "tower_id": "T001",
                    "method": TOWER_TREE_EXCLUSION_VERSION,
                    "excluded_point_count": 3,
                    "coverage": "PARTIAL_CONNECTED_STRUCTURE",
                    "registered_base_xyz_absolute_m": [0.0, 0.0, 0.0],
                    "registered_top_xyz_absolute_m": [0.0, 0.0, 12.0],
                },
                {
                    "tower_id": "T002",
                    "method": TOWER_TREE_EXCLUSION_VERSION,
                    "excluded_point_count": 0,
                    "coverage": "UNAVAILABLE",
                    "registered_base_xyz_absolute_m": [10.0, 0.0, 0.0],
                    "registered_top_xyz_absolute_m": [10.0, 0.0, 12.0],
                },
            ),
            (
                np.asarray([10, 11, 12], dtype=np.int64),
                np.empty(0, dtype=np.int64),
            ),
        ),
        span,
        source_identity=source_identity,
        region_origin_xyz=(0, 0, 0),
    )
    return project, batch


def test_finalization_appends_only_surviving_tower_points_and_is_idempotent(tmp_path):
    project, batch = _project_artifacts(tmp_path)
    session = _session()
    # A prior human edit already removed source row 10.  It must remain first
    # in the journal and must never be restored by the automatic migration.
    human_operation = {
        "type": "REMOVE_POINTS",
        "tree_ids": [0],
        "selection_method": "REGION_LASSO",
        "point_count": 1,
        "removed_indices_b64": _encode_indices(np.asarray([10], dtype=np.int64)),
        "sequence": 1,
        "timestamp": "2026-09-29T00:00:00",
    }
    session.operations.append(human_operation)

    audit = ensure_editor_tower_exclusion(project, batch, ("span-a",), session)
    assert audit is not None
    assert audit["tower_source_index_count"] == 3
    assert audit["removed_from_current_membership"] == 2
    assert session.operations[0] is human_operation
    assert session.operations[1]["selection_method"] == TOWER_EDITOR_SELECTION_METHOD
    final_index = session.build_final()[0]
    np.testing.assert_array_equal(final_index, np.asarray([20, 21], dtype=np.int64))

    again = ensure_editor_tower_exclusion(project, batch, ("span-a",), session)
    assert again is not None and again["removed_from_current_membership"] == 0
    assert len(session.operations) == 2
    session.undo()
    reapplied = ensure_editor_tower_exclusion(project, batch, ("span-a",), session)
    assert reapplied is not None and reapplied["removed_from_current_membership"] == 2
    np.testing.assert_array_equal(session.build_final()[0], final_index)


def test_migrated_journal_survives_save_reopen_replay_without_ghost_tree(tmp_path):
    project, batch = _project_artifacts(tmp_path)
    editor = object.__new__(BatchTreeEditor)
    editor.project_root = project
    editor.artifact_root = batch
    editor.result = SimpleNamespace(analysis_run_id="run", span_statuses={"span-a": "FINALIZED"})
    editor.session = _session()
    editor.original = editor.session.original
    editor.tree_ids_by_span = {"span-a": (0, 1)}
    editor.session.operations.append(
        {
            "type": "REMOVE_POINTS",
            "tree_ids": [1],
            "selection_method": "REGION_LASSO",
            "point_count": 1,
            "removed_indices_b64": _encode_indices(np.asarray([20], dtype=np.int64)),
            "sequence": 1,
            "timestamp": "2026-09-29T00:00:00",
        }
    )
    audit = ensure_editor_tower_exclusion(project, batch, ("span-a",), editor.session)
    assert audit is not None and audit["removed_from_current_membership"] == 3
    editor.persist_edits()
    saved = json.loads((batch / "tree_edits.json").read_text(encoding="utf-8"))
    assert [op["sequence"] for op in saved["operations"]] == [1, 2]

    reopened = _session()
    reopened.operations = load_edits_payload(saved["operations"])
    assert reopened.operations[-1]["selection_method"] == TOWER_EDITOR_SELECTION_METHOD
    assert reopened.snapshot()["current_tree_count"] == 1
    assert [row["tree_id"] for row in reopened.snapshot()["trees"]] == [1]
    assert reopened.current_tree_ids() == [1]
    np.testing.assert_array_equal(reopened.build_final()[0], np.asarray([21], dtype=np.int64))

    original_records = tuple(
        GlobalTreeRecord(
            global_tree_id=tree_id,
            line_id=0,
            span_id=0,
            span_name="span-a",
            external_name=f"Tree{tree_id + 1:03d}",
            primary_span_tree_id=tree_id,
            apex_xy_local_m=(float(tree_id * 3), 0.0, 9.0),
            apex_height_m=9.0,
            full_resolution_point_count=3 if tree_id == 0 else 2,
            core_point_count=3 if tree_id == 0 else 2,
            halo_point_count=0,
            core_fraction=1.0,
            review=False,
            merged_span_tree_keys=((0, tree_id),),
        )
        for tree_id in (0, 1)
    )
    editor.session = reopened
    editor.original_reconciled = TreeReconciliationResult(
        original_records,
        editor.original.original_index,
        editor.original.xyz_local_m,
        editor.original.tree_id,
        (),
        {(0, 0): 0, (0, 1): 1},
        {},
    )
    editor._final_reconciled_cache = None
    reconciled = editor.final_reconciled()
    assert [record.global_tree_id for record in reconciled.global_tree_records] == [1]
    assert reconciled.global_tree_records[0].full_resolution_point_count == 1
    assert reconciled.span_tree_to_global == {(0, 1): 1}

    # DBH's existing journal reader must treat automatic tower removal as an
    # exact current exclusion, irrespective of its selection_method label.
    lineage = [
        {"global_tree_id": 0, "merged_span_tree_keys": [[0, 0]]},
        {"global_tree_id": 1, "merged_span_tree_keys": [[0, 1]]},
    ]
    _, dbh_removed = editor_eligibility(lineage, reopened.operations)
    np.testing.assert_array_equal(dbh_removed, np.asarray([10, 11, 12, 20]))

    repeated = ensure_editor_tower_exclusion(project, batch, ("span-a",), reopened)
    assert repeated is not None and repeated["removed_from_current_membership"] == 0
    assert len(reopened.operations) == 2
    assert [op["sequence"] for op in reopened.operations] == [1, 2]


def test_already_deleted_tower_members_do_not_add_an_operation(tmp_path):
    project, batch = _project_artifacts(tmp_path)
    session = _session()
    session.operations.append(
        {"type": "DELETE", "tree_ids": [0], "sequence": 1, "timestamp": "2026-09-29T00:00:00"}
    )
    audit = ensure_editor_tower_exclusion(project, batch, ("span-a",), session)
    assert audit is not None and audit["removed_from_current_membership"] == 0
    assert len(session.operations) == 1
    np.testing.assert_array_equal(session.build_final()[0], np.asarray([20, 21], dtype=np.int64))


def test_editor_undo_skips_automatic_tower_operation_and_reapplies_it(tmp_path):
    project, batch = _project_artifacts(tmp_path)
    editor = object.__new__(BatchTreeEditor)
    editor.project_root = project
    editor.artifact_root = batch
    editor.result = SimpleNamespace(span_statuses={"span-a": "FINALIZED"})
    editor.session = _session()
    human = {
        "type": "REMOVE_POINTS",
        "tree_ids": [0],
        "selection_method": "REGION_LASSO",
        "point_count": 1,
        "removed_indices_b64": _encode_indices(np.asarray([10], dtype=np.int64)),
        "sequence": 1,
        "timestamp": "2026-09-29T00:00:00",
    }
    editor.session.operations.append(human)
    ensure_editor_tower_exclusion(project, batch, ("span-a",), editor.session)

    undone = editor.undo()
    assert undone is human
    assert len(editor.session.operations) == 1
    assert editor.session.operations[0]["selection_method"] == TOWER_EDITOR_SELECTION_METHOD
    np.testing.assert_array_equal(
        editor.session.build_final()[0], np.asarray([20, 21], dtype=np.int64)
    )


def test_legacy_without_confirmed_wire_region_uses_original_vegetation(tmp_path):
    project, batch = _project_artifacts(tmp_path)
    span = batch / "span-a"
    (span / "tower_membership.json").unlink()
    (span / "tower_membership.npz").unlink()
    rng = np.random.default_rng(901)
    z = rng.uniform(0.0, 11.8, 8_000)
    shaft = np.column_stack(
        (rng.normal(0.7, 0.07, len(z)), rng.normal(0.6, 0.07, len(z)), z)
    )
    tree = np.column_stack(
        (rng.normal(2.8, 0.08, len(z)), rng.normal(0.6, 0.08, len(z)), z)
    )
    points = np.concatenate((shaft, tree))
    writer = _RawRegionWriter(
        span / "regions" / "vegetation", origin_xyz=(0.0, 0.0, 0.0)
    )
    writer.append(
        points,
        np.arange(len(points), dtype=np.int64),
        np.zeros(len(points), dtype=np.uint8),
    )
    writer.close()
    source_index = _span_tower_indices(project, span)
    assert source_index is not None and len(source_index) > 0
    assert np.all(source_index < len(shaft))
    assert (span / "tower_membership.json").is_file()
