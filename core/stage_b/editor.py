"""Batch integration for the proven full-resolution TreeEditSession."""

from __future__ import annotations

import json
import os
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
from core.stage_b.tower_exclusion import (
    TOWER_EDITOR_SELECTION_METHOD,
    ensure_editor_tower_exclusion,
)


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
        original_reconciled: TreeReconciliationResult | None = None,
        config: FastBaselineConfig | None = None,
        acceleration_mode: str = "AUTO",
    ) -> None:
        load_started = time.perf_counter()
        reconciled = getattr(result, "global_reconciled", None)
        if reconciled is None:
            raise BatchTreeEditorError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        self.project_root = Path(project_root).expanduser().resolve()
        self.result = result
        # Keep the analysis publication immutable in memory while successive
        # DELETE/UNDO finalizations update ``self.result``.  This mirrors the
        # original -> edits -> final membership architecture for clearance.
        self.original_clearance = result.final_clearance
        self.config = config or FastBaselineConfig()
        from algorithms.spans.gpu_clearance import normalize_gpu_mode

        self.acceleration_mode = normalize_gpu_mode(acceleration_mode)
        self.original_reconciled = original_reconciled or reconciled
        self.artifact_root = (
            self.project_root
            / "artifacts"
            / "project-runs"
            / "batches"
            / result.analysis_run_id
        )
        self.original = build_original_trees(
            self.original_reconciled.global_tree_records,
            self.original_reconciled.final_original_index,
            self.original_reconciled.final_xyz_local_m,
            self.original_reconciled.final_tree_id,
        )
        self.session = RegionTreeEditSession(self.original)
        self.tree_ids_by_span = {
            str(key): tuple(int(value) for value in values)
            for key, values in (tree_ids_by_span or {}).items()
        }
        self.tree_load_seconds = time.perf_counter() - load_started
        self.last_viewer_update_seconds = 0.0
        self.finalization_performance: dict[str, Any] = {}
        self._final_reconciled_cache: tuple[str, TreeReconciliationResult] | None = None
        # A UI read may populate the reconciliation cache before the matching
        # full-resolution NPZ has been published. Keep disk publication separate.
        self._persisted_final_membership_key: str | None = None
        self._final_membership_disk_mismatch = False

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
        if (
            self.session.operations
            and self.session.operations[-1].get("selection_method")
            == TOWER_EDITOR_SELECTION_METHOD
        ):
            if len(self.session.operations) == 1:
                return None
            self.session.undo()
            undone = self.session.undo()
            ensure_editor_tower_exclusion(
                self.project_root,
                self.artifact_root,
                tuple(self.result.span_statuses),
                self.session,
            )
            return undone
        return self.session.undo()

    def snapshot(self) -> dict[str, Any]:
        return self.session.snapshot()

    def display_identity_records(self) -> tuple[dict[str, Any], ...]:
        """Return the exact fields used by presentation tree numbering.

        Display numbering depends only on the surviving/merged entity graph,
        authoritative span ownership and longitudinal position.  Rebuilding
        full-resolution final arrays here was unnecessary and made every local
        editor action scan millions of rows.
        """

        by_id = {
            int(record.global_tree_id): record
            for record in self.original_reconciled.global_tree_records
        }
        records: list[dict[str, Any]] = []
        for tree in self.snapshot().get("trees", []):
            source_ids = [int(value) for value in tree.get("source_tree_ids", [])]
            if not source_ids:
                continue
            base = min(
                (by_id[source_id] for source_id in source_ids),
                key=lambda item: item.global_tree_id,
            )
            records.append(
                {
                    "global_tree_id": int(tree["tree_id"]),
                    "span_id": int(base.span_id),
                    "longitudinal_s_m": base.longitudinal_s_m,
                }
            )
        return tuple(records)

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
        final_path = self.artifact_root / "tree_segmentation_final.npz"
        from core.stage_b.final_generation import editor_journal_digest

        cache_key = editor_journal_digest(self.session.operations)
        cached_final = (
            self._final_reconciled_cache[1]
            if self._final_reconciled_cache is not None
            and self._final_reconciled_cache[0] == cache_key
            and self._persisted_final_membership_key == cache_key
            and final_path.is_file()
            else None
        )
        if cached_final is not None:
            final_index = cached_final.final_original_index
            final_tree = cached_final.final_tree_id
            final_xyz = cached_final.final_xyz_local_m
            summary = {}
        elif not self.session.operations:
            # Before the first edit the final membership is byte-for-byte the
            # immutable original membership.  Rebuilding three multi-million
            # row arrays and serialising a duplicate NPZ added seconds to every
            # analysis.  A same-directory hard link publishes those exact bytes
            # in O(1); later edited publications use atomic replacement, which
            # leaves the original inode untouched.
            final_index = self.original.original_index
            final_tree = self.original.tree_id
            final_xyz = self.original.xyz_local_m
            summary: dict[int, dict[str, Any]] = {}
            original_path = self.artifact_root / "tree_segmentation_original.npz"
            temporary = final_path.with_name(final_path.name + "." + uuid4().hex + ".tmp")
            try:
                os.link(original_path, temporary)
                os.replace(temporary, final_path)
            except OSError:
                temporary.unlink(missing_ok=True)
                self._write_npz(
                    final_path,
                    original_index=final_index,
                    xyz_local_m=final_xyz,
                    tree_id=final_tree,
                )
        else:
            final_index, final_tree, final_xyz, summary = self.session.build_final()
            self._write_npz(
                final_path,
                original_index=final_index,
                xyz_local_m=final_xyz,
                tree_id=final_tree,
            )
        # Only a completed atomic replacement proves the current journal's
        # membership is on disk. A preview-only cache must never set this key.
        self._persisted_final_membership_key = cache_key
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

        from core.stage_b.final_generation import editor_journal_digest

        cache_key = editor_journal_digest(self.session.operations)
        if self._final_reconciled_cache is not None:
            cached = self._final_reconciled_cache[1]
            same_membership = final_membership is None or (
                final_membership[0] is cached.final_original_index
                and final_membership[1] is cached.final_tree_id
                and final_membership[2] is cached.final_xyz_local_m
            )
            if self._final_reconciled_cache[0] == cache_key and same_membership:
                return cached
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
        from core.stage_b.final_generation import editor_journal_digest, final_generation_id

        generation_id = final_generation_id(
            self.result.analysis_run_id,
            records,
            len(self.session.operations),
        )
        diagnostics = dict(self.original_reconciled.diagnostics)
        diagnostics.update(
            {
                "editor_schema": "treeline-stage-b-batch-editor-v1",
                "editor_operation_count": len(self.session.operations),
                "editor_journal_digest": editor_journal_digest(self.session.operations),
                "duplicate_final_original_index": 0,
                "original_tree_count": len(
                    self.original_reconciled.global_tree_records
                ),
                "global_tree_count": len(records),
                "final_generation_id": generation_id,
            }
        )
        output = TreeReconciliationResult(
            global_tree_records=tuple(records),
            final_original_index=final_index,
            final_xyz_local_m=final_xyz,
            final_tree_id=final_tree,
            review_tree_ids=tuple(review_ids),
            span_tree_to_global=span_tree_to_global,
            diagnostics=diagnostics,
        )
        self._final_reconciled_cache = (cache_key, output)
        return output

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

    def _clearance_matches_membership(
        self,
        clearance: Any | None,
        final: TreeReconciliationResult,
    ) -> bool:
        if clearance is None:
            return False
        final_records = {
            int(record.global_tree_id): record for record in final.global_tree_records
        }
        clearance_records: dict[int, Any] = {}
        for record in clearance.records:
            tree_id = int(record.global_tree_id)
            if tree_id in clearance_records:
                return False
            clearance_records[tree_id] = record
        if set(clearance_records) != set(final_records):
            return False
        return all(
            int(clearance_records[tree_id].tree_point_count)
            == int(record.full_resolution_point_count)
            and getattr(clearance_records[tree_id], "persistent_tree_id", None)
            == getattr(record, "persistent_tree_id", None)
            for tree_id, record in final_records.items()
        )

    def _reuse_delete_only_clearance(self, final: TreeReconciliationResult) -> Any | None:
        """Project existing exact per-tree clearance through whole-tree DELETE edits.

        A DELETE removes complete memberships and changes neither the coordinates
        nor wire geometry of any survivor.  Therefore every surviving per-tree
        record and danger observation is already the exact result the frozen
        clearance algorithm would recompute.  The strict inventory and point
        count checks below fail closed for legacy/incomplete publications.
        """

        clearance = self.original_clearance
        operations = self.session.operations
        if clearance is None or not operations or any(
            operation.get("type") != "DELETE" for operation in operations
        ):
            return None
        original_records = {
            str(record.persistent_tree_id): record
            for record in self.original_reconciled.global_tree_records
        }
        clearance_records: dict[str, Any] = {}
        for record in clearance.records:
            persistent_id = str(record.persistent_tree_id)
            if persistent_id in clearance_records:
                return None
            clearance_records[persistent_id] = record
        if set(clearance_records) != set(original_records):
            return None
        final_records = {
            str(record.persistent_tree_id): record for record in final.global_tree_records
        }
        if not set(final_records) <= set(original_records):
            return None
        for persistent_id, record in final_records.items():
            prior = clearance_records[persistent_id]
            original = original_records[persistent_id]
            if (
                int(record.full_resolution_point_count)
                != int(original.full_resolution_point_count)
                or int(prior.tree_point_count) != int(record.full_resolution_point_count)
                or getattr(prior, "persistent_tree_id", None)
                != getattr(record, "persistent_tree_id", None)
            ):
                return None
        new_tree_id = {
            persistent_id: int(record.global_tree_id)
            for persistent_id, record in final_records.items()
        }
        records = tuple(
            replace(
                clearance_records[persistent_id],
                global_tree_id=new_tree_id[persistent_id],
            )
            for persistent_id in sorted(final_records, key=lambda key: new_tree_id[key])
        )
        danger_points = tuple(
            replace(point, global_tree_id=new_tree_id[str(point.persistent_tree_id)])
            for point in getattr(clearance, "danger_points", ())
            if str(point.persistent_tree_id) in final_records
        )
        minima = [
            float(record.minimum_clearance_m)
            for record in records
            if record.minimum_clearance_m is not None
        ]
        diagnostics = dict(clearance.diagnostics)
        diagnostics.update(
            {
                "tree_count": len(records),
                "evaluated_edge_total": sum(
                    int(record.evaluated_edge_count) for record in records
                ),
                "danger_point_count": len(danger_points),
                "clearance_broad_phase_points": sum(
                    int(record.tree_point_count) for record in records
                ),
                "clearance_exact_points": sum(
                    int(record.tree_point_count) for record in records
                ),
                "final_projection_source": "EXACT_DELETE_ONLY_FILTER",
                "scientific_equivalence": "UNCHANGED_SURVIVOR_MEMBERSHIP_AND_WIRES",
            }
        )
        return replace(
            clearance,
            records=records,
            global_minimum_clearance_m=min(minima) if minima else None,
            diagnostics=diagnostics,
            danger_points=danger_points,
        )

    def finalize(
        self, *, progress: Any = None, reference_dbh_cm: float | None = None
    ) -> Any:
        """Persist edited memberships and return a finalized exportable result."""

        from core.stage_b.batch import BatchSpanStatus

        total_started = time.perf_counter()
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
        tower_exclusion_audit = ensure_editor_tower_exclusion(
            self.project_root,
            self.artifact_root,
            tuple(self.result.span_statuses),
            self.session,
        )
        final_projection_started = time.perf_counter()
        final_membership = self.persist()
        final = self.final_reconciled(final_membership)
        final_projection_seconds = time.perf_counter() - final_projection_started
        final_clearance = self.result.final_clearance
        operation_count = len(self.session.operations)
        wire_load_started = time.perf_counter()
        wire_results = [
            value.wire_result
            for value in self.result.span_results.values()
            if value.wire_result is not None
        ]
        wire_load_seconds = time.perf_counter() - wire_load_started
        complete_wire_set = bool(wire_results) and len(wire_results) == len(
            self.result.span_statuses
        )
        clearance_source = "UNAVAILABLE"
        clearance_started = time.perf_counter()
        current_result_clearance = self.result.final_clearance
        result_generation = (
            getattr(self.result.global_reconciled, "diagnostics", {}) or {}
        ).get("final_generation_id")
        final_generation = final.diagnostics.get("final_generation_id")
        if (
            result_generation
            and result_generation == final_generation
            and self._clearance_matches_membership(current_result_clearance, final)
        ):
            final_clearance = current_result_clearance
            clearance_source = "EXISTING_FINAL_RESULT"
        elif operation_count == 0 and self._clearance_matches_membership(
            self.original_clearance, final
        ):
            final_clearance = self.original_clearance
            clearance_source = "EXISTING_ANALYSIS_RESULT"
        else:
            projected_clearance = self._reuse_delete_only_clearance(final)
            if projected_clearance is not None:
                final_clearance = projected_clearance
                clearance_source = "EXACT_DELETE_ONLY_FILTER"
        if clearance_source == "UNAVAILABLE" and complete_wire_set:
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
                acceleration_mode=self.acceleration_mode,
            )
            clearance_source = "FULL_EXACT_RECALCULATION"
        elif clearance_source == "UNAVAILABLE" and operation_count > 0:
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
        elif clearance_source == "UNAVAILABLE" and operation_count == 0:
            # A prior DELETE publication may have fewer records than an UNDO
            # projection.  Never label that subset as the restored result.
            final_clearance = None
            if progress is not None:
                progress("VALIDATING_CLEARANCE")
        elif progress is not None:
            progress("VALIDATING_CLEARANCE")
        clearance_calc_seconds = time.perf_counter() - clearance_started
        clearance_diagnostics = (
            dict(getattr(final_clearance, "diagnostics", {}) or {})
            if final_clearance is not None
            else {}
        )
        acceleration = dict(clearance_diagnostics.get("acceleration") or {})
        gpu_effective = acceleration.get("effective_mode") == "GPU_CERTIFIED_CPU_EXACT"
        performance = {
            "TREE_LOAD_TIME": float(self.tree_load_seconds),
            "WIRE_LOAD_TIME": float(wire_load_seconds),
            "INDEX_BUILD_TIME": (
                float(clearance_diagnostics.get("index_build_seconds", 0.0) or 0.0)
                if clearance_source == "FULL_EXACT_RECALCULATION"
                else 0.0
            ),
            "CLEARANCE_CALC_TIME": float(clearance_calc_seconds),
            "RISK_UPDATE_TIME": 0.0,
            "VIEWER_UPDATE_TIME": float(self.last_viewer_update_seconds),
            "TOTAL_TIME": float(time.perf_counter() - total_started),
            "FINAL_PROJECTION_TIME": float(final_projection_seconds),
            "CLEARANCE_SOURCE": clearance_source,
            "GPU_USED_FOR": (
                "CERTIFIED_CLEARANCE_CANDIDATE_SCREENING"
                if gpu_effective
                else "NONE"
            ),
            "GPU_NOT_USED_FOR": (
                "GROUND,HAG,WIRE_RECONSTRUCTION,TREE_MEMBERSHIP,DBH,RISK,VIEWER"
            ),
        }
        generation_id = str(final.diagnostics["final_generation_id"])
        statuses = {span_id: BatchSpanStatus.FINALIZED for span_id in statuses}
        batch_diagnostics = dict(self.result.batch_diagnostics)
        batch_diagnostics["finalization_performance"] = performance
        batch_diagnostics["editor_journal_digest"] = final.diagnostics["editor_journal_digest"]
        if tower_exclusion_audit is not None:
            batch_diagnostics["tower_exclusion_final"] = tower_exclusion_audit
        batch_diagnostics["derived_generations"] = {
            "final_generation_id": generation_id,
            "result_generation_id": generation_id,
            "risk_generation_id": generation_id,
            "report_generation_id": generation_id,
        }
        self.result = replace(
            self.result,
            status="COMPLETE",
            span_statuses=statuses,
            global_reconciled=final,
            final_clearance=final_clearance,
            batch_diagnostics=batch_diagnostics,
        )
        self.finalization_performance = performance
        persist_final_danger_points(
            self.project_root,
            self.result.analysis_run_id,
            final_clearance,
        )
        self._write_json(
            self.artifact_root / "batch.json",
            self.result.to_dict(),
        )
        self._final_membership_disk_mismatch = False
        # Additive geometry only: DBH failure must not undo successful Survey
        # finalization or change crown/clearance membership. Reports only read
        # the resulting provenance-bound sidecar; they never trigger recovery.
        from core.stage_b.dbh_identity_recovery import finalize_dbh

        try:
            dbh_options = (
                {"reference_dbh_cm": reference_dbh_cm}
                if reference_dbh_cm is not None else {}
            )
            recovered = finalize_dbh(self.project_root, self.result, final, **dbh_options)
            self.dbh_recovery_status = {
                "status": "COMPLETE", "tree_count": len(recovered["trees"])
            }
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.dbh_recovery_status = {"status": "UNAVAILABLE", "reason": str(exc)}
        return self.result


__all__ = ["BatchTreeEditor", "BatchTreeEditorError"]
