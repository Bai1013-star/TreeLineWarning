"""Persistent entity identity tests independent of run-local tree ordering."""

from __future__ import annotations

import numpy as np
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from algorithms.spans.tree_identity import assign_persistent_tree_ids
from core.config.fast_baseline import default_fast_baseline_config


def _result(groups: list[list[int]], *, span_id: int = 4) -> TreeReconciliationResult:
    original = np.asarray([value for group in groups for value in group], dtype=np.int64)
    identifiers = np.asarray(
        [gid for gid, group in enumerate(groups) for _ in group], dtype=np.int32
    )
    records = tuple(
        GlobalTreeRecord(
            global_tree_id=gid, line_id=2, span_id=span_id, span_name="S005",
            external_name=f"G002-Span005-Tree{gid + 1:03d}", primary_span_tree_id=gid,
            apex_xy_local_m=(float(gid), 0.0), apex_height_m=10.0,
            full_resolution_point_count=len(group), core_point_count=len(group), halo_point_count=0,
            core_fraction=1.0, review=False, merged_span_tree_keys=((0, gid),),
            span_order_index=gid + 1,
        )
        for gid, group in enumerate(groups)
    )
    return TreeReconciliationResult(
        records, original, np.zeros((len(original), 3)), identifiers, (), {}, {}
    )


def _assign(tmp_path, result, run_id):
    return assign_persistent_tree_ids(
        result,
        default_fast_baseline_config(),
        registry_path=tmp_path / "tree_identity_registry.json",
        membership_path=tmp_path / run_id / "tree_identity_membership.npz",
        group_id="G002", source_key="same-source", run_id=run_id,
    )


def test_same_source_rerun_preserves_ids_and_new_tree_does_not_renumber(tmp_path) -> None:
    first = _assign(tmp_path, _result([[1, 2, 3], [10, 11, 12]]), "run-a")
    assert [record.persistent_tree_id for record in first.global_tree_records] == [
        "G002-T001", "G002-T002"
    ]
    # A missed tree is inserted spatially between the old entities, but IDs are
    # matched by original-index membership and the new one takes the next ID.
    second = _assign(
        tmp_path, _result([[1, 2, 3], [6, 7, 8], [10, 11, 12]]), "run-b"
    )
    assert [record.persistent_tree_id for record in second.global_tree_records] == [
        "G002-T001", "G002-T003", "G002-T002"
    ]


def test_split_and_merge_are_not_silently_inherited(tmp_path) -> None:
    _assign(tmp_path, _result([[1, 2, 3, 4, 5, 6]]), "run-a")
    split = _assign(tmp_path, _result([[1, 2, 3], [4, 5, 6]]), "run-b")
    assert {record.identity_state for record in split.global_tree_records} == {"NEW_ENTITY"}
    # Two established entities observed as one do not auto-merge either.
    merged = _assign(tmp_path, _result([[1, 2, 3, 4, 5, 6]]), "run-c")
    assert merged.global_tree_records[0].identity_state == "MERGE_AMBIGUOUS_REVIEW"
