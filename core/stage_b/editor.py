"""Batch integration for the proven full-resolution TreeEditSession."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from algorithms.folder_pipeline.tree_editor import build_original_trees
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult

from core.config.fast_baseline import FastBaselineConfig, canonical_digest
from core.stage_b.danger_points import persist_final_danger_points
from core.stage_b.region_edit_session import RegionTreeEditSession


class BatchTreeEditorError(RuntimeError):
    """Closed failure for invalid batch-editor state or publication."""


class BatchTreeEditor:
    """Expose DELETE/MERGE/UNDO without mutating raw source or other Spans."""

    def __init__(
        self,
        project_root: Path | str,
        result: Any,
        *,
        tree_ids_by_span: Mapping[str, tuple[int, ...]] | None = None,
        config: FastBaselineConfig | None = None,
    ) -> None:
        reconciled = getattr(result, "global_reconciled", None)
        if reconciled is None:
            raise BatchTreeEditorError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        self.project_root = Path(project_root).expanduser().resolve()
        self.result = result
        self.config = config or FastBaselineConfig()
        self.original_reconciled = reconciled
        self.artifact_root = (
            self.project_root
            / "artifacts"
            / "project-runs"
            / "batches"
            / result.analysis_run_id
        )
        self.original = build_original_trees(
            reconciled.global_tree_records,
            reconciled.final_original_index,
            reconciled.final_xyz_local_m,
            reconciled.final_tree_id,
        )
        self.session = RegionTreeEditSession(self.original)
        self.tree_ids_by_span = {
            str(key): tuple(int(value) for value in values)
            for key, values in (tree_ids_by_span or {}).items()
        }

    def _check_span(self, span_id: str, tree_ids: list[int]) -> None:
        allowed = set(self.tree_ids_by_span.get(span_id, ()))
        if not allowed:
            raise BatchTreeEditorError(f"UNKNOWN_BATCH_SPAN:{span_id}")
        if any(tree_id not in allowed for tree_id in tree_ids):
            raise BatchTreeEditorError("TREE_EDIT_CROSSES_UNRELATED_SPAN")

    def select(self, span_id: str, tree_ids: list[int]) -> None:
        self._check_span(span_id, tree_ids)
        self.session.select(tree_ids)

    def delete(self, span_id: str, tree_ids: list[int]) -> list[int]:
        self.select(span_id, tree_ids)
        return self.session.delete_selected()

    def merge(self, span_id: str, tree_ids: list[int]) -> int:
        self.select(span_id, tree_ids)
        return self.session.merge_selected()

    def undo(self) -> dict[str, Any] | None:
        return self.session.undo()

    def snapshot(self) -> dict[str, Any]:
        return self.session.snapshot()

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # An abandoned temp is never an authoritative journal.

    def _write_npz(self, path: Path, **arrays: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez(temporary, **arrays)
        temporary.replace(path)

    def persist_edits(self) -> None:
        """Atomically persist exact edits without republishing reconstructible finals."""

        self.artifact_root.mkdir(parents=True, exist_ok=True)
        original_path = self.artifact_root / "tree_segmentation_original.npz"
        if not original_path.is_file():
            self._write_npz(
                original_path,
                original_index=np.asarray(self.original.original_index, dtype=np.int64),
                xyz_local_m=np.asarray(self.original.xyz_local_m, dtype=np.float64),
                tree_id=np.asarray(self.original.tree_id, dtype=np.int32),
            )
        edits = self.session.edits_payload()
        edits.update(
            {
                "analysis_run_id": self.result.analysis_run_id,
                "span_tree_ids": {
                    key: list(values)
                    for key, values in sorted(self.tree_ids_by_span.items())
                },
            }
        )
        started = time.perf_counter()
        self._write_json(self.artifact_root / "tree_edits.json", edits)
        self.last_journal_write_seconds = time.perf_counter() - started

    def persist(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
        """Persist clean/original, edit-log, current/final membership and display evidence."""

        self.persist_edits()
        final_index, final_tree, final_xyz, summary = self.session.build_final()
        self._write_npz(
            self.artifact_root / "tree_segmentation_final.npz",
            original_index=final_index,
            xyz_local_m=final_xyz,
            tree_id=final_tree,
        )
        # BatchAnalysisScheduler._finish() already calls this method immediately
        # after exact clearance is produced.  Persist its exact danger XYZ here
        # so a close/reopen before the user first opens Step-2 does not discard
        # the point-level display evidence.  The sidecar is optional UI data;
        # clearance records remain authoritative if the filesystem write fails.
        try:
            persist_final_danger_points(
                self.project_root,
                self.result.analysis_run_id,
                getattr(self.result, "final_clearance", None),
            )
        except (OSError, TypeError, ValueError):
            pass
        return final_index, final_tree, final_xyz, summary

    def final_reconciled(
        self,
        final_membership: tuple[
            np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]
        ]
        | None = None,
    ) -> TreeReconciliationResult:
        """Return the post-editor, unique full-resolution publication result."""

        final_index, final_tree, final_xyz, _summary = (
            final_membership if final_membership is not None else self.session.build_final()
        )
        if len(final_index) != len(np.unique(final_index)):
            raise BatchTreeEditorError("DUPLICATE_FINAL_ORIGINAL_INDEX")
        current = {
            int(tree["tree_id"]): tuple(int(value) for value in tree["source_tree_ids"])
            for tree in self.session.snapshot()["trees"]
        }
        provenance = self.session.edit_provenance_by_current_tree()
        by_id = {
            int(record.global_tree_id): record
            for record in self.original_reconciled.global_tree_records
        }
        records: list[GlobalTreeRecord] = []
        span_tree_to_global: dict[tuple[int, int], int] = {}
        review_ids: list[int] = []
        for final_id in sorted(current):
            source_ids = current[final_id]
            source_records = [by_id[source_id] for source_id in source_ids]
            base = min(source_records, key=lambda record: record.global_tree_id)
            point_count = int(np.count_nonzero(final_tree == final_id))
            member_index = final_index[final_tree == final_id]
            apex_xyz = base.apex_xy_local_m
            apex_height = base.apex_height_m
            height_candidates: list[tuple[float, tuple[float, float, float]]] = []
            for source_record in source_records:
                source_mask = self.original.tree_id == int(source_record.global_tree_id)
                source_index = self.original.original_index[source_mask]
                source_xyz = self.original.xyz_local_m[source_mask]
                remaining_mask = np.isin(source_index, member_index)
                if not np.any(remaining_mask):
                    continue
                remaining_xyz = source_xyz[remaining_mask]
                highest = remaining_xyz[int(np.argmax(remaining_xyz[:, 2]))]
                source_height = source_record.apex_height_m
                if source_height is not None:
                    adjusted_height = (
                        float(source_height)
                        + float(highest[2])
                        - float(np.max(source_xyz[:, 2]))
                    )
                else:
                    adjusted_height = float(highest[2])
                height_candidates.append(
                    (adjusted_height, tuple(float(value) for value in highest))
                )
            if height_candidates:
                apex_height, apex_xyz = max(height_candidates, key=lambda item: item[0])
            merged_keys = tuple(
                sorted(
                    key
                    for record in source_records
                    for key in record.merged_span_tree_keys
                )
            )
            persistent = base.persistent_tree_id
            if len(source_records) > 1:
                persistent = "tree-" + canonical_digest(
                    {
                        "source": self.result.project_id,
                        "analysis_run_id": self.result.analysis_run_id,
                        "source_tree_ids": [
                            int(record.global_tree_id) for record in source_records
                        ],
                    }
                )[:24]
            review = any(record.review for record in source_records)
            record = replace(
                base,
                global_tree_id=final_id,
                external_name=(
                    base.external_name
                    if len(source_records) == 1
                    else f"Merged-Tree{final_id:03d}"
                ),
                full_resolution_point_count=point_count,
                apex_xy_local_m=apex_xyz,
                apex_height_m=apex_height,
                core_point_count=min(base.core_point_count, point_count),
                halo_point_count=max(0, point_count - min(base.core_point_count, point_count)),
                core_fraction=(
                    min(base.core_point_count, point_count) / point_count
                    if point_count
                    else 0.0
                ),
                review=review,
                merged_span_tree_keys=merged_keys,
                persistent_tree_id=persistent,
                identity_state=(
                    "HUMAN_" + "+".join(provenance.get(final_id, ()))
                    if provenance.get(final_id)
                    else "BATCH_EDITOR_FINAL"
                ),
            )
            records.append(record)
            if review:
                review_ids.append(final_id)
            for source_record in source_records:
                for key in source_record.merged_span_tree_keys:
                    span_tree_to_global[key] = final_id
        diagnostics = dict(self.original_reconciled.diagnostics)
        diagnostics.update(
            {
                "editor_schema": "treeline-stage-b-batch-editor-v1",
                "editor_operation_count": len(self.session.operations),
                "duplicate_final_original_index": 0,
            }
        )
        return TreeReconciliationResult(
            global_tree_records=tuple(records),
            final_original_index=final_index,
            final_xyz_local_m=final_xyz,
            final_tree_id=final_tree,
            review_tree_ids=tuple(review_ids),
            span_tree_to_global=span_tree_to_global,
            diagnostics=diagnostics,
        )

    def mark_reviewed(self, span_ids: list[str] | tuple[str, ...] | None = None) -> None:
        """Advance completed Span review states without touching failed Spans."""

        selected = set(span_ids or self.result.span_statuses)
        statuses = dict(self.result.span_statuses)
        from core.stage_b.batch import BatchSpanStatus

        for span_id in selected:
            if statuses.get(span_id) in {
                BatchSpanStatus.AUTO_ANALYZED,
                BatchSpanStatus.REVIEW_REQUIRED,
            }:
                statuses[span_id] = BatchSpanStatus.USER_REVIEWED
        self.result = replace(self.result, span_statuses=statuses)

    def finalize(self, *, progress: Any = None) -> Any:
        """Persist edited memberships and return a finalized exportable result."""

        from core.stage_b.batch import BatchSpanStatus

        if any(status is BatchSpanStatus.FAILED for status in self.result.span_statuses.values()):
            raise BatchTreeEditorError("FINALIZE_REQUIRES_NO_FAILED_SPANS")
        statuses = dict(self.result.span_statuses)
        if any(
            status not in {BatchSpanStatus.USER_REVIEWED, BatchSpanStatus.FINALIZED}
            for status in statuses.values()
        ):
            raise BatchTreeEditorError("FINALIZE_REQUIRES_USER_REVIEW")
        if progress is not None:
            progress("APPLYING_EDITS")
        final_membership = self.persist()
        final = self.final_reconciled(final_membership)
        final_clearance = self.result.final_clearance
        operation_count = len(self.session.operations)
        wire_results = [
            value.wire_result
            for value in self.result.span_results.values()
            if value.wire_result is not None
        ]
        complete_wire_set = bool(wire_results) and len(wire_results) == len(
            self.result.span_statuses
        )
        if complete_wire_set:
            if progress is not None:
                progress("RECALCULATING_CLEARANCE")
            from algorithms.spans.clearance import run_span_tree_wire_clearance

            final_clearance = run_span_tree_wire_clearance(
                final,
                wire_results,
                self.config,
                danger_threshold_m=float(
                    self.result.risk_policy_snapshot["warning_clearance_threshold_m"]
                ),
            )
        elif operation_count > 0:
            # Partial wire geometry cannot be passed into the existing global
            # clearance function: it concatenates supplied wires and evaluates
            # every final tree against them.  Retaining pre-edit clearance would
            # be stale; evaluating a partial set would be scientifically wrong.
            # Fail closed until all required wire geometry is available again.
            final_clearance = None
            diagnostics = dict(final.diagnostics)
            diagnostics.update(
                {
                    "clearance_state": "INVALIDATED_AFTER_TREE_EDIT",
                    "clearance_reason": "INCOMPLETE_WIRE_RESULTS",
                }
            )
            final = replace(final, diagnostics=diagnostics)
            if progress is not None:
                # Keep the established desktop progress contract compatible;
                # the exact invalidation reason is published in diagnostics.
                progress("VALIDATING_CLEARANCE")
        elif progress is not None:
            progress("VALIDATING_CLEARANCE")
        statuses = {span_id: BatchSpanStatus.FINALIZED for span_id in statuses}
        self.result = replace(
            self.result,
            status="COMPLETE",
            span_statuses=statuses,
            global_reconciled=final,
            final_clearance=final_clearance,
        )
        persist_final_danger_points(
            self.project_root,
            self.result.analysis_run_id,
            final_clearance,
        )
        self._write_json(
            self.artifact_root / "batch.json",
            self.result.to_dict(),
        )
        # Additive geometry only: DBH failure must not undo successful Survey
        # finalization or change crown/clearance membership. Reports only read
        # the resulting provenance-bound sidecar; they never trigger recovery.
        from core.stage_b.dbh_identity_recovery import finalize_dbh

        try:
            recovered = finalize_dbh(self.project_root, self.result, final)
            self.dbh_recovery_status = {
                "status": "COMPLETE", "tree_count": len(recovered["trees"])
            }
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.dbh_recovery_status = {"status": "UNAVAILABLE", "reason": str(exc)}
        return self.result


__all__ = ["BatchTreeEditor", "BatchTreeEditorError"]
