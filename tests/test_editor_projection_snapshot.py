"""A concurrent desktop commit must not mix two journals in one read."""
from types import SimpleNamespace

import numpy as np

from algorithms.spans.reconcile import GlobalTreeRecord
from core.stage_b.editor import BatchTreeEditor


def test_projection_uses_one_session_and_one_tree_snapshot():
    editor = object.__new__(BatchTreeEditor)
    xyz = np.array([[0., 0., 1.], [0., 0., 3.]])
    indices = np.array([10, 11], dtype=np.int64)
    ids = np.array([0, 0], dtype=np.int32)
    record = GlobalTreeRecord(
        global_tree_id=0, line_id=1, span_id=1, span_name="test",
        external_name="Tree0", primary_span_tree_id=0,
        apex_xy_local_m=(0., 0., 3.), apex_height_m=3.,
        full_resolution_point_count=2, core_point_count=2,
        halo_point_count=0, core_fraction=1., review=False,
        merged_span_tree_keys=((1, 0),), persistent_tree_id="tree0",
    )
    editor.original_reconciled = SimpleNamespace(global_tree_records=[record], diagnostics={})
    editor.original = SimpleNamespace(original_index=indices, tree_id=ids, xyz_local_m=xyz)
    editor.result = SimpleNamespace(project_id="test", analysis_run_id="test-run")

    class Journal:
        operations = []
        calls = 0

        def build_final(self):
            return indices, ids, xyz, {}

        def snapshot(self):
            self.calls += 1
            # Simulate an atomic commit while a report reads the old journal.
            editor.session = SimpleNamespace(operations=["delete"])
            return {"trees": [{"tree_id": 0, "source_tree_ids": [0]}]}

        def edit_provenance_by_current_tree(self):
            return {}

    journal = Journal()
    editor.session = journal
    result = editor.final_reconciled()
    assert journal.calls == 1
    assert result.final_tree_count == 1
    assert result.global_tree_records[0].full_resolution_point_count == 2
    assert result.diagnostics["editor_operation_count"] == 0
