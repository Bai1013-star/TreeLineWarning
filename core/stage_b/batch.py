"""Multi-Span scheduler, global ownership publication, and audit telemetry."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from algorithms.spans.clearance import SpanDangerPoint
from algorithms.spans.project_adapter import (
    ScienceAdapterError,
    SourceSharedPreparation,
    SpanAnalysisInput,
    SpanScienceResult,
    prepare_source_shared,
    run_canonical_span_science,
)
from algorithms.spans.reconcile import (
    GlobalTreeRecord,
    TreeReconciliationResult,
    reconcile_trees,
)

from core.config.fast_baseline import FastBaselineConfig, canonical_digest
from core.integrity import validation_session
from core.stage_a.models import AnalysisRun, AnalysisRunStatus, Project, stable_id
from core.stage_a.persistence import save_project
from core.stage_a.risk import RiskPolicy
from core.stage_b.bridge import ProjectScienceBridge, SpanScienceBinding
from core.stage_b.cache import (
    SpanCacheUnavailable,
    load_exact_span_cache,
    load_span_clearance_payload,
    publish_run_cache_index,
    publish_span_cache_index,
    validate_run_cache_index,
)
from core.stage_b.danger_points import load_final_danger_points
from core.stage_b.failures import structured_failure


class BatchSpanStatus(str, Enum):
    WAITING = "WAITING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    AUTO_ANALYZED = "AUTO_ANALYZED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    USER_REVIEWED = "USER_REVIEWED"
    FINALIZED = "FINALIZED"
    FAILED = "FAILED"


@dataclass(slots=True)
class BatchTelemetry:
    source_point_count: int = 0
    span_count: int = 0
    source_prep_count: int = 0
    source_full_pass_count: int = 0
    ground_prep_count: int = 0
    span_local_read_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    peak_memory_bytes: int | None = None
    total_time_seconds: float = 0.0
    per_span_time_seconds: dict[str, float] | None = None
    stage_time_seconds: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.per_span_time_seconds is None:
            self.per_span_time_seconds = {}
        if self.stage_time_seconds is None:
            self.stage_time_seconds = {}

    def record_span_profile(self, result: SpanScienceResult) -> None:
        """Accumulate canonical-runner diagnostics without changing science."""

        raw = result.diagnostics.get("timings_seconds", {})
        if not isinstance(raw, Mapping):
            return
        for name, value in raw.items():
            try:
                elapsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(elapsed) and elapsed >= 0.0:
                self.stage_time_seconds[str(name)] = (
                    float(self.stage_time_seconds.get(str(name), 0.0)) + elapsed
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "SOURCE_POINT_COUNT": int(self.source_point_count),
            "SPAN_COUNT": int(self.span_count),
            "SOURCE_PREP_COUNT": int(self.source_prep_count),
            "SOURCE_FULL_PASS_COUNT": int(self.source_full_pass_count),
            "GROUND_PREP_COUNT": int(self.ground_prep_count),
            "SPAN_LOCAL_READ_COUNT": int(self.span_local_read_count),
            "LOCAL_SPATIAL_READ_COUNT": int(self.span_local_read_count),
            "CACHE_HIT_COUNT": int(self.cache_hit_count),
            "CACHE_MISS_COUNT": int(self.cache_miss_count),
            "PEAK_MEMORY_BYTES": self.peak_memory_bytes,
            "TOTAL_TIME": float(self.total_time_seconds),
            "PER_SPAN_TIME": dict(self.per_span_time_seconds or {}),
            "STAGE_TIME": dict(self.stage_time_seconds or {}),
        }


@dataclass(frozen=True, slots=True)
class BatchAnalysisResult:
    analysis_run_id: str
    project_id: str
    scan_id: str
    status: str
    span_statuses: Mapping[str, BatchSpanStatus]
    span_results: Mapping[str, SpanScienceResult]
    telemetry: BatchTelemetry
    risk_policy_snapshot: Mapping[str, Any]
    global_reconciled: TreeReconciliationResult | None = None
    final_clearance: Any | None = None
    batch_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    batch_manifest_reference: str | None = None
    prediction_export_reference: str | None = None

    def __post_init__(self) -> None:
        """Restore the authoritative Span order when a result is hydrated.

        Batch manifests sort mapping keys for stable JSON, while the editor's
        clearance publication consumes ``span_results.values()``.  The
        scheduler records the topology-derived order explicitly so an
        in-memory result, a JSON round-trip, and a later editor finalization all
        concatenate the same wire sequence.  Legacy manifests without this
        metadata retain their historical mapping order.
        """

        diagnostics = self.batch_diagnostics
        if not isinstance(diagnostics, Mapping) or "canonical_span_order" not in diagnostics:
            return
        raw_order = diagnostics["canonical_span_order"]
        if not isinstance(raw_order, (list, tuple)):
            raise ValueError("canonical_span_order must be a list or tuple")
        order = tuple(raw_order)
        if any(type(span_id) is not str for span_id in order):
            raise ValueError("canonical_span_order must contain string Span IDs")
        if len(order) != len(set(order)):
            raise ValueError("canonical_span_order must not contain duplicates")
        current = dict(self.span_results)
        unexpected = [span_id for span_id in current if span_id not in order]
        if unexpected:
            raise ValueError(
                "span_results contains IDs absent from canonical_span_order: "
                + ",".join(str(span_id) for span_id in unexpected)
            )
        object.__setattr__(
            self,
            "span_results",
            {span_id: current[span_id] for span_id in order if span_id in current},
        )

    @property
    def duplicate_final_original_index(self) -> int:
        if self.global_reconciled is None:
            return 0
        validated = self.global_reconciled.diagnostics.get(
            "duplicate_final_original_index"
        )
        if type(validated) is int and validated >= 0:
            return validated
        values = np.asarray(self.global_reconciled.final_original_index, dtype=np.int64)
        return int(len(values) - len(np.unique(values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "treeline-stage-batch-result-v1",
            "analysis_run_id": self.analysis_run_id,
            "project_id": self.project_id,
            "scan_id": self.scan_id,
            "status": self.status,
            "span_statuses": {
                key: value.value for key, value in sorted(self.span_statuses.items())
            },
            "span_results": {
                key: {
                    "tree_count": value.tree_count,
                    "minimum_clearance_m": value.minimum_clearance_m,
                    "risk_threshold_m": value.risk_threshold_m,
                    "risk_counts": dict(value.risk_counts),
                    "diagnostics": dict(value.diagnostics),
                }
                for key, value in sorted(self.span_results.items())
            },
            "telemetry": self.telemetry.to_dict(),
            "risk_policy_snapshot": dict(self.risk_policy_snapshot),
            "global_reconciliation": (
                {
                    "tree_count": self.global_reconciled.final_tree_count,
                    "duplicate_final_original_index": self.duplicate_final_original_index,
                    "diagnostics": dict(self.global_reconciled.diagnostics),
                    "trees": [
                        record.to_dict()
                        for record in self.global_reconciled.global_tree_records
                    ],
                }
                if self.global_reconciled is not None
                else None
            ),
            "final_clearance": (
                {
                    "status": self.final_clearance.status,
                    "global_minimum_clearance_m": self.final_clearance.global_minimum_clearance_m,
                    "records": [record.to_dict() for record in self.final_clearance.records],
                    "diagnostics": dict(self.final_clearance.diagnostics),
                }
                if self.final_clearance is not None
                else None
            ),
            "batch_diagnostics": dict(self.batch_diagnostics),
            "batch_manifest_reference": self.batch_manifest_reference,
            "prediction_export_reference": self.prediction_export_reference,
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _hydrate_persisted_danger_points(
    project_root: Path,
    run_id: str,
    clearance: Any | None,
    global_reconciled: TreeReconciliationResult,
) -> Any | None:
    """Attach exact persisted danger XYZ to a restart-hydrated clearance.

    ``batch.json`` intentionally stores bounded clearance records and omits the
    potentially larger point list.  The editor persists that list separately;
    when a record says that points exist, a missing or inconsistent sidecar is
    a cache miss rather than a partially hydrated COMPLETE result.
    """

    if clearance is None:
        return None
    records = tuple(getattr(clearance, "records", ()) or ())
    persistent_by_tree = {
        int(record.global_tree_id): record.persistent_tree_id
        for record in global_reconciled.global_tree_records
    }
    expected_by_tree: dict[int, int] = {}
    try:
        for record in records:
            tree_id = int(record.global_tree_id)
            if tree_id in expected_by_tree:
                return None
            if tree_id not in persistent_by_tree or (
                record.persistent_tree_id != persistent_by_tree[tree_id]
            ):
                return None
            expected_by_tree[tree_id] = int(getattr(record, "danger_point_count", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return None
    if any(count < 0 for count in expected_by_tree.values()):
        return None
    expected_count = sum(expected_by_tree.values())
    arrays = load_final_danger_points(project_root, run_id)
    if expected_count == 0:
        if arrays is not None and len(arrays[0]) != 0:
            return None
        return clearance
    if arrays is None:
        return None
    original_index, tree_id, xyz = arrays
    if len(original_index) != expected_count:
        return None

    final_index = np.asarray(global_reconciled.final_original_index, dtype=np.int64)
    final_tree = np.asarray(global_reconciled.final_tree_id, dtype=np.int32)
    final_xyz = np.asarray(global_reconciled.final_xyz_local_m, dtype=np.float64)
    if (
        final_index.ndim != 1
        or final_tree.shape != final_index.shape
        or final_xyz.shape != (len(final_index), 3)
    ):
        return None
    order = np.argsort(final_index, kind="mergesort")
    sorted_index = final_index[order]
    if np.any(sorted_index[1:] == sorted_index[:-1]):
        return None
    positions = np.searchsorted(sorted_index, original_index)
    if np.any(positions >= len(sorted_index)):
        return None
    if not np.array_equal(sorted_index[positions], original_index):
        return None
    member_positions = order[positions]
    if not np.array_equal(final_tree[member_positions], tree_id):
        return None
    if not np.array_equal(final_xyz[member_positions], xyz):
        return None
    record_by_tree = {int(record.global_tree_id): record for record in records}
    actual_by_tree: dict[int, int] = {}
    points: list[SpanDangerPoint] = []
    for source_index, global_tree_id, point_xyz in zip(
        original_index, tree_id, xyz, strict=True
    ):
        source_key = int(source_index)
        tree_key = int(global_tree_id)
        record = record_by_tree.get(tree_key)
        if record is None:
            return None
        values = tuple(float(value) for value in point_xyz)
        if len(values) != 3 or not np.all(np.isfinite(values)):
            return None
        actual_by_tree[tree_key] = actual_by_tree.get(tree_key, 0) + 1
        points.append(
            SpanDangerPoint(
                global_tree_id=tree_key,
                persistent_tree_id=record.persistent_tree_id,
                original_index=source_key,
                xyz_local_m=values,
            )
        )
    if any(actual_by_tree.get(tree_id, 0) != count for tree_id, count in expected_by_tree.items()):
        return None
    if any(tree_id not in expected_by_tree for tree_id in actual_by_tree):
        return None
    return replace(clearance, danger_points=tuple(points))


class BatchAnalysisScheduler:
    """Execute selected logical Spans over one shared source preparation."""

    def __init__(
        self,
        project_root: Path | str,
        project: Project,
        scan_id: str,
        *,
        risk_policy: RiskPolicy | None = None,
        config: FastBaselineConfig | None = None,
        science_runner: Callable[
            [SpanAnalysisInput, SourceSharedPreparation, Path], SpanScienceResult
        ] | None = None,
        source_preparer: Callable[[Path, SpanAnalysisInput], SourceSharedPreparation] | None = None,
        source_access_enabled: bool = False,
    ) -> None:
        if type(project) is not Project:
            raise TypeError("project must be a Stage-A Project")
        self.project_root = Path(project_root).expanduser().resolve()
        self.project = project
        self.risk_policy = risk_policy or RiskPolicy()
        self.config = config or FastBaselineConfig()
        self.bridge = ProjectScienceBridge(
            project, scan_id, risk_policy=self.risk_policy, config=self.config
        )
        self._uses_default_science_runner = science_runner is None
        self.science_runner = science_runner or self._default_science_runner
        self._uses_default_source_preparer = source_preparer is None
        self.source_preparer = source_preparer or prepare_source_shared
        self.source_access_enabled = source_access_enabled
        self._latest_result: BatchAnalysisResult | None = None

    def _default_science_runner(
        self, input_data: SpanAnalysisInput, preparation: SourceSharedPreparation, output_root: Path
    ) -> SpanScienceResult:
        return run_canonical_span_science(
            input_data,
            preparation,
            output_root,
            executor=self._tracked_executor,
        )

    def _tracked_executor(
        self,
        input_data: SpanAnalysisInput,
        preparation: SourceSharedPreparation,
        output_root: Path,
    ) -> SpanScienceResult:
        from algorithms.spans.canonical_executor import run_tracked_canonical_span_science

        return run_tracked_canonical_span_science(
            input_data,
            preparation,
            output_root,
            config=self.config,
        )

    def _new_analysis_run(self, span_ids: tuple[str, ...], run_id: str) -> AnalysisRun:
        return AnalysisRun(
            analysis_run_id=run_id,
            scan_id=self.bridge.scan.scan_id,
            span_ids=span_ids,
            algorithm_versions={
                "ground": "PDAL_SMRF_HAG_NN_FAST",
                "wire": "FAST_MATURE_SPAN_WIRE",
                "vegetation": "FULL_RESOLUTION_ORIGINAL_INDEX",
                "tree": "CROWN_STEM_HYBRID",
                "reconciliation": "GLOBAL_ORIGINAL_INDEX_CORE_OWNERSHIP",
            },
            configuration_snapshot={
                "tree_backend": self.config.tree_backend,
                "wire_entrypoint": "algorithms.spans.wire.run_span_wire_pipeline",
                "tree_entrypoint": "algorithms.spans.tree_runner.run_span_tree_pipeline",
                "offline_runtime": True,
            },
            risk_policy_snapshot=self.risk_policy_snapshot,
            status=AnalysisRunStatus.RUNNING,
        )

    @property
    def risk_policy_snapshot(self) -> dict[str, Any]:
        snapshot = self.risk_policy.to_dict()
        snapshot["configuration_digest"] = canonical_digest(snapshot)
        return snapshot

    def _persist_run(self, run: AnalysisRun) -> None:
        current = [
            item
            for item in self.project.analysis_runs
            if item.analysis_run_id != run.analysis_run_id
        ]
        self.project = replace(self.project, analysis_runs=tuple((*current, run)))
        if (self.project_root / "stage-a-project.json").is_file():
            save_project(self.project, self.project_root)

    def _source_access_metrics(self, telemetry: BatchTelemetry) -> None:
        """Build the legacy diagnostic chunk index only when explicitly requested.

        Production Span analysis consumes the role-specific tile indexes bound
        by ``prepare_source_shared``.  The generic ``SharedSourceAccess`` index
        is not referenced by that executor, so scanning a large source merely
        to populate diagnostic counters is deliberately opt-in.
        """

        from core.stage_a.shared_source import SharedSourceAccess

        source_identity = self.bridge.source_identity
        source_value = source_identity.get("resolved_access_path") or source_identity.get("path")
        if not self.source_access_enabled or not isinstance(source_value, str):
            return
        source = Path(source_value)
        if not source.is_file() or source.suffix.lower() not in {".las", ".laz"}:
            return
        access = SharedSourceAccess(
            source,
            cache_root=(
                self.project_root
                / "artifacts"
                / "project-runs"
                / "source-shared"
                / "spatial-index"
            ),
        )
        access.prepare()
        telemetry.source_full_pass_count += access.metrics.full_source_read_count
        telemetry.cache_hit_count += access.metrics.cache_reuse_count
        telemetry.cache_miss_count += access.metrics.cache_prep_count

    def _load_previous(self, run_id: str) -> dict[str, Any] | None:
        path = self.project_root / "artifacts" / "project-runs" / "batches" / run_id / "batch.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _assert_previous_resume_allowed(
        self, run_id: str, previous: Mapping[str, Any] | None
    ) -> None:
        """Reject immutable/editor-edited runs before any scheduler writes."""

        manifest_path = (
            self.project_root
            / "artifacts"
            / "project-runs"
            / "batches"
            / run_id
            / "batch.json"
        )
        if previous is None and manifest_path.is_file():
            raise ScienceAdapterError("BATCH_RUN_MANIFEST_INVALID")
        if previous is not None:
            if (
                previous.get("schema") != "treeline-stage-batch-result-v1"
                or previous.get("analysis_run_id") != run_id
            ):
                raise ScienceAdapterError("BATCH_RUN_MANIFEST_INVALID")
            statuses = previous.get("span_statuses")
            manifest_status = previous.get("status")
            valid_manifest_statuses = {
                "COMPLETE",
                "PARTIAL_FAILURE",
                "FAILED",
                BatchSpanStatus.FINALIZED.value,
            }
            if (
                not isinstance(manifest_status, str)
                or manifest_status not in valid_manifest_statuses
            ):
                raise ScienceAdapterError("BATCH_RUN_MANIFEST_INVALID")
            valid_span_statuses = {status.value for status in BatchSpanStatus}
            if not isinstance(statuses, Mapping) or any(
                not isinstance(value, str) or value not in valid_span_statuses
                for value in statuses.values()
            ):
                raise ScienceAdapterError("BATCH_RUN_MANIFEST_INVALID")
            if isinstance(statuses, Mapping) and any(
                value == BatchSpanStatus.FINALIZED.value for value in statuses.values()
            ):
                raise ScienceAdapterError(
                    "BATCH_RUN_IMMUTABLE_FINALIZED_REQUIRES_WORKFLOW_REHYDRATION"
                )
            if previous.get("status") == BatchSpanStatus.FINALIZED.value:
                raise ScienceAdapterError(
                    "BATCH_RUN_IMMUTABLE_FINALIZED_REQUIRES_WORKFLOW_REHYDRATION"
                )

        journal_path = (
            self.project_root
            / "artifacts"
            / "project-runs"
            / "batches"
            / run_id
            / "tree_edits.json"
        )
        if journal_path.is_file():
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ScienceAdapterError(
                    "BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID"
                ) from exc
            if not isinstance(journal, Mapping):
                raise ScienceAdapterError("BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID")
            if journal.get("schema") != "folder-to-excel-tree-edits-v1":
                raise ScienceAdapterError("BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID")
            journal_run_id = journal.get("analysis_run_id")
            if journal_run_id is not None and journal_run_id != run_id:
                raise ScienceAdapterError("BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID")
            operations = journal.get("operations")
            if not isinstance(operations, list):
                raise ScienceAdapterError("BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID")
            if operations:
                raise ScienceAdapterError(
                    "BATCH_RUN_IMMUTABLE_EDITED_REQUIRES_WORKFLOW_REHYDRATION"
                )

        if previous is None:
            return
        global_payload = previous.get("global_reconciliation")
        diagnostics = (
            global_payload.get("diagnostics")
            if isinstance(global_payload, Mapping)
            else None
        )
        if not isinstance(diagnostics, Mapping):
            return
        editor_schema = diagnostics.get("editor_schema")
        if editor_schema is not None and editor_schema != (
            "treeline-stage-b-batch-editor-v1"
        ):
            raise ScienceAdapterError("BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID")
        if editor_schema == "treeline-stage-b-batch-editor-v1" and (
            "editor_operation_count" not in diagnostics
        ):
            raise ScienceAdapterError("BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID")
        operation_count = diagnostics.get("editor_operation_count")
        if operation_count is None:
            return
        if type(operation_count) is not int or operation_count < 0:
            raise ScienceAdapterError("BATCH_RUN_IMMUTABLE_EDIT_JOURNAL_INVALID")
        if operation_count:
            raise ScienceAdapterError(
                "BATCH_RUN_IMMUTABLE_EDITED_REQUIRES_WORKFLOW_REHYDRATION"
            )

    @staticmethod
    def _telemetry_from_manifest(payload: Mapping[str, Any]) -> BatchTelemetry | None:
        raw = payload.get("telemetry")
        if not isinstance(raw, Mapping):
            return None
        try:
            return BatchTelemetry(
                source_point_count=int(raw.get("SOURCE_POINT_COUNT", 0) or 0),
                span_count=int(raw.get("SPAN_COUNT", 0) or 0),
                source_prep_count=int(raw.get("SOURCE_PREP_COUNT", 0) or 0),
                source_full_pass_count=int(raw.get("SOURCE_FULL_PASS_COUNT", 0) or 0),
                ground_prep_count=int(raw.get("GROUND_PREP_COUNT", 0) or 0),
                span_local_read_count=int(raw.get("SPAN_LOCAL_READ_COUNT", 0) or 0),
                cache_hit_count=int(raw.get("CACHE_HIT_COUNT", 0) or 0),
                cache_miss_count=int(raw.get("CACHE_MISS_COUNT", 0) or 0),
                peak_memory_bytes=(
                    None
                    if raw.get("PEAK_MEMORY_BYTES") is None
                    else int(raw["PEAK_MEMORY_BYTES"])
                ),
                total_time_seconds=float(raw.get("TOTAL_TIME", 0.0) or 0.0),
                per_span_time_seconds=dict(raw.get("PER_SPAN_TIME") or {}),
                stage_time_seconds=dict(raw.get("STAGE_TIME") or {}),
            )
        except (TypeError, ValueError):
            return None

    def _resume_existing_complete(
        self,
        run_id: str,
        selected: tuple[str, ...],
        bindings: Mapping[str, SpanScienceBinding],
        previous: Mapping[str, Any] | None,
    ) -> BatchAnalysisResult | None:
        """Hydrate a clean COMPLETE run without rewriting its artifacts."""

        if previous is None or previous.get("status") != "COMPLETE":
            return None
        if previous.get("project_id") != self.project.project_id:
            return None
        if previous.get("scan_id") != self.bridge.scan.scan_id:
            return None
        statuses_raw = previous.get("span_statuses")
        if not isinstance(statuses_raw, Mapping) or set(statuses_raw) != set(selected):
            return None
        if any(value == BatchSpanStatus.FAILED.value for value in statuses_raw.values()):
            return None
        canonical_order = self._canonical_span_order(selected, bindings)
        previous_diagnostics = previous.get("batch_diagnostics")
        if not isinstance(previous_diagnostics, Mapping):
            return None
        previous_order = previous_diagnostics.get("canonical_span_order")
        if not isinstance(previous_order, list) or tuple(previous_order) != canonical_order:
            return None
        previous_policy = previous.get("risk_policy_snapshot")
        if not isinstance(previous_policy, Mapping):
            return None
        if previous_policy.get("configuration_digest") != self.risk_policy_snapshot.get(
            "configuration_digest"
        ):
            return None
        run_proof_valid, _run_proof_reason = validate_run_cache_index(
            self.project_root, run_id
        )
        if not run_proof_valid:
            return None
        telemetry = self._telemetry_from_manifest(previous)
        if telemetry is None:
            return None

        exact_results: dict[str, SpanScienceResult] = {}
        for span_id in canonical_order:
            exact, _reason = load_exact_span_cache(
                self.project_root,
                run_id,
                bindings[span_id],
                self.config,
            )
            if exact is None:
                return None
            exact_results[span_id] = exact

        global_reconciled = self._load_persisted_global(run_id)
        if global_reconciled is None:
            return None
        global_payload = previous.get("global_reconciliation")
        if not isinstance(global_payload, Mapping):
            return None
        try:
            if int(global_payload.get("tree_count")) != global_reconciled.final_tree_count:
                return None
        except (TypeError, ValueError):
            return None
        final_clearance = load_span_clearance_payload(previous.get("final_clearance"))
        if previous.get("final_clearance") is not None and final_clearance is None:
            return None
        if final_clearance is not None:
            final_clearance = _hydrate_persisted_danger_points(
                self.project_root,
                run_id,
                final_clearance,
                global_reconciled,
            )
            if final_clearance is None:
                return None
        try:
            statuses = {
                span_id: BatchSpanStatus(str(statuses_raw[span_id]))
                for span_id in canonical_order
            }
        except (KeyError, TypeError, ValueError):
            return None
        return BatchAnalysisResult(
            analysis_run_id=run_id,
            project_id=self.project.project_id,
            scan_id=self.bridge.scan.scan_id,
            status="COMPLETE",
            span_statuses=statuses,
            span_results=exact_results,
            telemetry=telemetry,
            risk_policy_snapshot=dict(previous_policy),
            global_reconciled=global_reconciled,
            final_clearance=final_clearance,
            batch_diagnostics=dict(previous_diagnostics),
            batch_manifest_reference=(
                f"artifacts/project-runs/batches/{run_id}/batch.json"
            ),
            prediction_export_reference=previous.get("prediction_export_reference"),
        )

    def _complete_telemetry(
        self,
        result: BatchAnalysisResult,
        telemetry: BatchTelemetry,
        started: float,
    ) -> BatchAnalysisResult:
        """Publish telemetry after all reconciliation/publication work ends."""

        telemetry.total_time_seconds = time.perf_counter() - started
        if result.batch_manifest_reference:
            _write_json(
                self.project_root / Path(result.batch_manifest_reference),
                result.to_dict(),
            )
        return result

    def _load_persisted_global(self, run_id: str) -> TreeReconciliationResult | None:
        """Rehydrate valid global ownership for restart without science rerun."""

        batch_root = self.project_root / "artifacts" / "project-runs" / "batches" / run_id
        reconciliation_path = batch_root / "batch-reconciliation.json"
        membership_path = batch_root / "tree_segmentation_original.npz"
        if not reconciliation_path.is_file() or not membership_path.is_file():
            return None
        try:
            payload = json.loads(reconciliation_path.read_text(encoding="utf-8"))
            records = tuple(
                GlobalTreeRecord(
                    global_tree_id=int(raw["global_tree_id"]),
                    line_id=int(raw["line_id"]),
                    span_id=int(raw["span_id"]),
                    span_name=str(raw["span_name"]),
                    external_name=str(raw["external_name"]),
                    primary_span_tree_id=int(raw["primary_span_tree_id"]),
                    apex_xy_local_m=tuple(float(value) for value in raw["apex_xy_local_m"]),
                    apex_height_m=float(raw["apex_height_m"]),
                    full_resolution_point_count=int(raw["full_resolution_point_count"]),
                    core_point_count=int(raw["core_point_count"]),
                    halo_point_count=int(raw["halo_point_count"]),
                    core_fraction=float(raw["core_fraction"]),
                    review=bool(raw["review"]),
                    merged_span_tree_keys=tuple(
                        (int(item[0]), int(item[1]))
                        for item in raw.get("merged_span_tree_keys", [])
                    ),
                    persistent_tree_id=raw.get("persistent_tree_id"),
                    identity_state=str(raw.get("identity_state", "RUN_LOCAL_ONLY")),
                    candidate_persistent_tree_ids=tuple(
                        str(value) for value in raw.get("candidate_persistent_tree_ids", [])
                    ),
                    longitudinal_s_m=raw.get("longitudinal_s_m"),
                    lateral_n_m=raw.get("lateral_n_m"),
                    span_order_index=raw.get("span_order_index"),
                )
                for raw in payload.get("trees", [])
            )
            with np.load(membership_path, allow_pickle=False) as arrays:
                original_index = np.asarray(arrays["original_index"], dtype=np.int64)
                final_xyz = np.asarray(arrays["xyz_local_m"], dtype=np.float64)
                final_tree = np.asarray(arrays["tree_id"], dtype=np.int32)
            if len(original_index) != len(final_xyz) or len(original_index) != len(final_tree):
                return None
            if len(original_index) != len(np.unique(original_index)):
                return None
            mapping = {
                key: record.global_tree_id
                for record in records
                for key in record.merged_span_tree_keys
            }
            return TreeReconciliationResult(
                global_tree_records=records,
                final_original_index=original_index,
                final_xyz_local_m=final_xyz,
                final_tree_id=final_tree,
                review_tree_ids=tuple(
                    record.global_tree_id for record in records if record.review
                ),
                span_tree_to_global=mapping,
                diagnostics=dict(payload.get("diagnostics") or {}),
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    def _reused_result(payload: Mapping[str, Any], span_id: str) -> SpanScienceResult | None:
        raw = (payload.get("span_results") or {}).get(span_id)
        if not isinstance(raw, Mapping):
            return None
        try:
            return SpanScienceResult(
                tree_count=int(raw.get("tree_count", 0)),
                minimum_clearance_m=raw.get("minimum_clearance_m"),
                risk_threshold_m=float(raw.get("risk_threshold_m", 0.5)),
                risk_counts=dict(raw.get("risk_counts") or {}),
                diagnostics=dict(raw.get("diagnostics") or {}),
            )
        except (TypeError, ValueError):
            return None

    def _global_reconcile(
        self,
        bindings: Mapping[str, SpanScienceBinding],
        results: Mapping[str, SpanScienceResult],
    ) -> TreeReconciliationResult | None:
        available = [result for result in results.values() if result.tree_outcome is not None]
        if not available:
            return None
        # The canonical Span executor has already reconciled its one outcome.
        # Re-running the same full-membership union for a single-Span batch is
        # mathematically identical but expensive on tens of millions of rows.
        # Multi-Span batches still execute the authoritative global union.
        reused_single_span = len(available) == 1 and available[0].reconciled is not None
        if reused_single_span:
            reconciled = available[0].reconciled
        else:
            outcomes = [result.tree_outcome for result in available]
            reconciled = reconcile_trees(
                outcomes,
                self.config,
                origin_xyz=tuple(
                    float(item)
                    for item in self.bridge.source_identity.get(
                        "origin_context",
                        self.bridge.source_identity.get(
                            "source_bounds_min_xyz", (0.0, 0.0, 0.0)
                        ),
                    )
                ),
            )
        # Persistent IDs are lineage-derived, not volatile result-list order.
        source_digest = canonical_digest(self.bridge.source_identity)
        records = []
        for record in reconciled.global_tree_records:
            merged = tuple(sorted(record.merged_span_tree_keys))
            identity_material = {
                "source": source_digest,
                "line_id": record.line_id,
                "primary_span_id": record.span_id,
                "primary_span_tree_id": record.primary_span_tree_id,
                "merged_span_tree_keys": [list(item) for item in merged],
            }
            persistent_id = "tree-" + canonical_digest(identity_material)[:24]
            records.append(
                replace(
                    record,
                    persistent_tree_id=persistent_id,
                    identity_state="BATCH_LINEAGE_DERIVED",
                )
            )
        diagnostics = dict(reconciled.diagnostics)
        validated_duplicate_final = diagnostics.get("duplicate_final_original_index")
        final_index = np.asarray(reconciled.final_original_index, dtype=np.int64)
        strictly_ordered = not len(final_index) or bool(
            np.all(final_index[1:] > final_index[:-1])
        )
        if type(validated_duplicate_final) is int and strictly_ordered:
            duplicate_final = validated_duplicate_final
        else:
            duplicate_final = int(len(final_index) - len(np.unique(final_index)))
        diagnostics["duplicate_final_original_index"] = duplicate_final
        if duplicate_final != 0:
            raise ScienceAdapterError("DUPLICATE_FINAL_ORIGINAL_INDEX")
        return replace(reconciled, global_tree_records=tuple(records), diagnostics=diagnostics)

    @staticmethod
    def _canonical_span_order(
        selected: tuple[str, ...], bindings: Mapping[str, SpanScienceBinding]
    ) -> tuple[str, ...]:
        """Return selected product Span IDs in ProjectScienceBridge order."""

        if len(selected) != len(set(selected)):
            raise ScienceAdapterError("SPAN_SELECTION_CONTAINS_DUPLICATES")
        missing = [span_id for span_id in selected if span_id not in bindings]
        if missing:
            raise ScienceAdapterError(
                "SPAN_BINDING_MISSING_FOR_CANONICAL_ORDER:" + ",".join(missing)
            )
        ordinals: dict[str, int] = {}
        for span_id in selected:
            internal = bindings[span_id].internal_span_id
            if (
                type(internal) is not str
                or len(internal) < 2
                or internal[0] != "S"
                or not internal[1:].isdigit()
            ):
                raise ScienceAdapterError("INVALID_INTERNAL_SPAN_ORDER_ID")
            ordinals[span_id] = int(internal[1:])
        if len(ordinals) != len(set(ordinals.values())):
            raise ScienceAdapterError("DUPLICATE_INTERNAL_SPAN_ORDER_ID")
        return tuple(sorted(selected, key=ordinals.__getitem__))

    def _tree_ids_by_span(
        self, result: BatchAnalysisResult
    ) -> dict[str, tuple[int, ...]]:
        """Map product Span IDs to global IDs without using display labels."""

        if result.global_reconciled is None:
            return {}
        mapping: dict[str, tuple[int, ...]] = {}
        for binding in self.bridge.bind_spans(tuple(result.span_statuses)):
            internal_span_id = int(binding.science_input.span_id[1:]) - 1
            mapping[binding.product_span.span_id] = tuple(
                record.global_tree_id
                for record in result.global_reconciled.global_tree_records
                if record.span_id == internal_span_id
                or any(
                    key[0] == internal_span_id
                    for key in record.merged_span_tree_keys
                )
            )
        return mapping

    def open_editor(self, result: BatchAnalysisResult | None = None) -> Any:
        """Open the proven editor over the batch's global result."""

        from core.stage_b.editor import BatchTreeEditor

        selected = result or self._latest_result
        if selected is None:
            raise ScienceAdapterError("TREE_EDITOR_REQUIRES_BATCH_RESULT")
        return BatchTreeEditor(
            self.project_root,
            selected,
            tree_ids_by_span=self._tree_ids_by_span(selected),
            config=self.config,
        )

    @validation_session()
    def run(
        self,
        span_ids: tuple[str, ...] | list[str] | None = None,
        *,
        analysis_run_id: str | None = None,
    ) -> BatchAnalysisResult:
        started = time.perf_counter()
        bindings_tuple = self.bridge.bind_spans(span_ids)
        if not bindings_tuple:
            raise ValueError("batch analysis requires at least one Span")
        selected = tuple(binding.product_span.span_id for binding in bindings_tuple)
        run_id = analysis_run_id or stable_id("analysis")
        previous = self._load_previous(run_id)
        bindings = {
            binding.product_span.span_id: binding.with_analysis_run(run_id)
            for binding in bindings_tuple
        }
        self._assert_previous_resume_allowed(run_id, previous)
        resumed = self._resume_existing_complete(run_id, selected, bindings, previous)
        if resumed is not None:
            self._latest_result = resumed
            return resumed
        statuses = {span_id: BatchSpanStatus.WAITING for span_id in selected}
        results: dict[str, SpanScienceResult] = {}
        telemetry = BatchTelemetry(span_count=len(selected))
        source_point_count = self.bridge.source_identity.get("source_point_count")
        if isinstance(source_point_count, int):
            telemetry.source_point_count = source_point_count

        self._persist_run(self._new_analysis_run(selected, run_id))
        preparation: SourceSharedPreparation | None = None
        try:
            statuses = {span_id: BatchSpanStatus.PREPARING for span_id in selected}
            first_input = bindings[selected[0]].science_input
            if self._uses_default_source_preparer:
                # Source-shared Wire routing includes the reviewed span-aligned
                # rescue path.  Give it the complete project topology so a
                # cache prepared by a subset/retry is still valid for every
                # logical Span in this project.
                all_span_inputs = tuple(
                    binding.science_input
                    for binding in self.bridge.bind_spans(None)
                )
                preparation = prepare_source_shared(
                    self.project_root,
                    first_input,
                    span_inputs=all_span_inputs,
                )
            else:
                # Preserve the established two-argument test/integration seam.
                preparation = self.source_preparer(self.project_root, first_input)
            if telemetry.source_point_count <= 0:
                try:
                    preparation_payload = json.loads(
                        (self.project_root / preparation.manifest_reference).read_text(
                            encoding="utf-8"
                        )
                    )
                    prepared_point_count = preparation_payload.get("source_point_count")
                    if type(prepared_point_count) is int and prepared_point_count > 0:
                        telemetry.source_point_count = prepared_point_count
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            telemetry.source_prep_count = 0 if preparation.telemetry.get("cache_hit_count") else 1
            telemetry.cache_hit_count += int(preparation.telemetry.get("cache_hit_count", 0))
            telemetry.cache_miss_count += int(preparation.telemetry.get("cache_miss_count", 0))
            telemetry.ground_prep_count = int(preparation.telemetry.get("ground_prep_count", 0))
            telemetry.source_full_pass_count += int(
                preparation.telemetry.get("source_full_pass_count", 0)
            )
            if preparation.ground_preparation and not preparation.ground_preparation.get(
                "coarse_ground_reference"
            ):
                raise ScienceAdapterError("GROUND_REFERENCE_NONE")
            self._source_access_metrics(telemetry)
            statuses = {span_id: BatchSpanStatus.WAITING for span_id in selected}
        except Exception as exc:
            failure = structured_failure(exc, phase="SOURCE_PREPARATION")
            for span_id in selected:
                statuses[span_id] = BatchSpanStatus.FAILED
            for span_id in selected:
                results[span_id] = SpanScienceResult(
                    tree_count=0,
                    minimum_clearance_m=None,
                    risk_threshold_m=self.risk_policy.warning_clearance_threshold_m,
                    risk_counts={},
                    diagnostics={"failure": failure["technical_message"], **failure},
                )
            result = self._finish(run_id, selected, statuses, results, telemetry)
            result = self._complete_telemetry(result, telemetry, started)
            if result.status == "COMPLETE":
                try:
                    publish_run_cache_index(self.project_root, run_id)
                except SpanCacheUnavailable:
                    pass
            self._latest_result = result
            return result

        previous_statuses = (previous or {}).get("span_statuses") or {}
        for span_id in selected:
            input_data = replace(
                bindings[span_id].science_input,
                validated_source_preparation_references=(preparation.manifest_reference,),
            )
            previous_status = previous_statuses.get(span_id)
            reusable_status = previous_status in {
                BatchSpanStatus.AUTO_ANALYZED.value,
                BatchSpanStatus.REVIEW_REQUIRED.value,
                BatchSpanStatus.USER_REVIEWED.value,
            }
            exact_cached = None
            cache_miss_reason: str | None = None
            if reusable_status:
                exact_cached, cache_miss_reason = load_exact_span_cache(
                    self.project_root,
                    run_id,
                    bindings[span_id],
                    self.config,
                    preparation=preparation,
                )
                if exact_cached is None:
                    telemetry.cache_miss_count += 1
            if exact_cached is not None:
                previous_status = previous_statuses.get(span_id)
                statuses[span_id] = BatchSpanStatus(str(previous_status))
                results[span_id] = exact_cached
                telemetry.cache_hit_count += 1
                continue
            statuses[span_id] = BatchSpanStatus.RUNNING
            span_started = time.perf_counter()
            try:
                output_root = (
                    self.project_root
                    / "artifacts"
                    / "project-runs"
                    / "batches"
                    / run_id
                    / span_id
                )
                result = self.science_runner(input_data, preparation, output_root)
                if type(result) is not SpanScienceResult:
                    raise ScienceAdapterError("SCIENCE_EXECUTOR_RETURNED_INVALID_RESULT")
                if cache_miss_reason is not None:
                    result = replace(
                        result,
                        diagnostics={
                            **dict(result.diagnostics),
                            "span_cache": {
                                "status": "MISS_RECOMPUTED",
                                "reason": cache_miss_reason,
                                "authoritative": False,
                            },
                        },
                    )
                try:
                    publish_span_cache_index(
                        self.project_root,
                        run_id,
                        bindings[span_id],
                        input_data,
                        preparation,
                        self.config,
                        result,
                        output_root,
                    )
                except SpanCacheUnavailable as exc:
                    if self._uses_default_science_runner:
                        raise ScienceAdapterError(
                            f"SPAN_CACHE_PUBLICATION_FAILED:{exc}"
                        ) from exc
                    diagnostics = dict(result.diagnostics)
                    span_cache = dict(diagnostics.get("span_cache") or {})
                    if "reason" in span_cache:
                        span_cache["publication_reason"] = str(exc)
                    else:
                        span_cache["reason"] = str(exc)
                    span_cache.update({"status": "UNAVAILABLE", "authoritative": False})
                    diagnostics["span_cache"] = span_cache
                    result = replace(
                        result,
                        diagnostics=diagnostics,
                    )
                results[span_id] = result
                telemetry.record_span_profile(result)
                telemetry.span_local_read_count += 1
                statuses[span_id] = (
                    BatchSpanStatus.REVIEW_REQUIRED
                    if result.diagnostics.get("review_required")
                    else BatchSpanStatus.AUTO_ANALYZED
                )
            except Exception as exc:
                failure = structured_failure(exc, phase="SPAN_ANALYSIS")
                results[span_id] = SpanScienceResult(
                    tree_count=0,
                    minimum_clearance_m=None,
                    risk_threshold_m=self.risk_policy.warning_clearance_threshold_m,
                    risk_counts={},
                    diagnostics={"failure": failure["technical_message"], **failure},
                )
                statuses[span_id] = BatchSpanStatus.FAILED
            telemetry.per_span_time_seconds[span_id] = time.perf_counter() - span_started

        result = self._finish(run_id, selected, statuses, results, telemetry, bindings)
        result = self._complete_telemetry(result, telemetry, started)
        if result.status == "COMPLETE":
            try:
                publish_run_cache_index(self.project_root, run_id)
            except SpanCacheUnavailable:
                pass
        self._latest_result = result
        return result

    def _finish(
        self,
        run_id: str,
        selected: tuple[str, ...],
        statuses: Mapping[str, BatchSpanStatus],
        results: Mapping[str, SpanScienceResult],
        telemetry: BatchTelemetry,
        bindings: Mapping[str, SpanScienceBinding] | None = None,
    ) -> BatchAnalysisResult:
        bindings = bindings or {
            binding.product_span.span_id: binding
            for binding in self.bridge.bind_spans(selected)
        }
        canonical_span_order = self._canonical_span_order(selected, bindings)
        unexpected_results = [span_id for span_id in results if span_id not in canonical_span_order]
        if unexpected_results:
            raise ScienceAdapterError(
                "RESULT_SPAN_NOT_SELECTED:"
                + ",".join(str(span_id) for span_id in unexpected_results)
            )
        ordered_results = {
            span_id: results[span_id]
            for span_id in canonical_span_order
            if span_id in results
        }
        statuses = dict(statuses)
        successful = [status is not BatchSpanStatus.FAILED for status in statuses.values()]
        failed = any(status is BatchSpanStatus.FAILED for status in statuses.values())
        global_reconciled = None
        batch_diagnostics: dict[str, Any] = {}
        if not failed:
            reconciliation_started = time.perf_counter()
            try:
                global_reconciled = self._global_reconcile(bindings, ordered_results)
                if global_reconciled is None:
                    global_reconciled = self._load_persisted_global(run_id)
                if global_reconciled is not None and global_reconciled.review_tree_ids:
                    for span_id, span_status in statuses.items():
                        if span_status is BatchSpanStatus.AUTO_ANALYZED:
                            statuses[span_id] = BatchSpanStatus.REVIEW_REQUIRED
            except Exception as exc:
                failed = True
                for span_id, status in statuses.items():
                    if status is BatchSpanStatus.AUTO_ANALYZED:
                        statuses[span_id] = BatchSpanStatus.REVIEW_REQUIRED
                results = dict(results)
                batch_diagnostics["global_reconciliation_failure"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                telemetry.stage_time_seconds["tree_reconciliation"] = (
                    time.perf_counter() - reconciliation_started
                )
        status = (
            "PARTIAL_FAILURE"
            if failed and any(successful)
            else "FAILED"
            if failed
            else "COMPLETE"
        )
        final_clearance = None
        if global_reconciled is not None:
            wire_results = [
                ordered_results[span_id].wire_result
                for span_id in canonical_span_order
                if (
                    span_id in ordered_results
                    and ordered_results[span_id].wire_result is not None
                )
            ]
            if wire_results and len(wire_results) == len(selected):
                clearance_publish_started = time.perf_counter()
                try:
                    # A one-Span global result has the exact same membership,
                    # wire geometry and risk threshold as the result already
                    # evaluated by the canonical executor.  Persistent IDs do
                    # not participate in distance science, so reuse is exact.
                    only = (
                        ordered_results.get(canonical_span_order[0])
                        if len(selected) == 1
                        else None
                    )
                    if only is not None and only.clearance_result is not None:
                        final_clearance = only.clearance_result
                        batch_diagnostics["final_clearance_reuse"] = (
                            "EXACT_SINGLE_SPAN_RESULT"
                        )
                    else:
                        from algorithms.spans.clearance import run_span_tree_wire_clearance

                        final_clearance = run_span_tree_wire_clearance(
                            global_reconciled,
                            wire_results,
                            self.config,
                            danger_threshold_m=self.risk_policy.warning_clearance_threshold_m,
                        )
                except Exception as exc:  # noqa: BLE001 - final review can retry
                    batch_diagnostics["final_clearance_failure"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    telemetry.stage_time_seconds["clearance_publish"] = (
                        time.perf_counter() - clearance_publish_started
                    )
        batch_diagnostics["canonical_span_order"] = list(canonical_span_order)
        result = BatchAnalysisResult(
            analysis_run_id=run_id,
            project_id=self.project.project_id,
            scan_id=self.bridge.scan.scan_id,
            status=status,
            span_statuses=dict(statuses),
            span_results=ordered_results,
            telemetry=telemetry,
            risk_policy_snapshot=self.risk_policy_snapshot,
            global_reconciled=global_reconciled,
            final_clearance=final_clearance,
            batch_diagnostics=batch_diagnostics,
        )
        batch_root = self.project_root / "artifacts" / "project-runs" / "batches" / run_id
        manifest_started = time.perf_counter()
        manifest = result.to_dict()
        manifest_path = batch_root / "batch.json"
        _write_json(manifest_path, manifest)
        telemetry.stage_time_seconds["manifest_write"] = (
            time.perf_counter() - manifest_started
        )
        if global_reconciled is not None:
            reconciliation_write_started = time.perf_counter()
            _write_json(
                batch_root / "batch-reconciliation.json",
                {
                    "schema": "treeline-stage-batch-reconciliation-v1",
                    "duplicate_final_original_index": result.duplicate_final_original_index,
                    "trees": [record.to_dict() for record in global_reconciled.global_tree_records],
                    "diagnostics": dict(global_reconciled.diagnostics),
                },
            )
            telemetry.stage_time_seconds["reconciliation_json_write"] = (
                time.perf_counter() - reconciliation_write_started
            )
            membership_started = time.perf_counter()
            _write_npz(
                batch_root / "tree_segmentation_original.npz",
                original_index=np.asarray(global_reconciled.final_original_index, dtype=np.int64),
                xyz_local_m=np.asarray(global_reconciled.final_xyz_local_m, dtype=np.float64),
                tree_id=np.asarray(global_reconciled.final_tree_id, dtype=np.int32),
            )
            telemetry.stage_time_seconds["membership_serialization_npz_write"] = (
                time.perf_counter() - membership_started
            )
            viewer_started = time.perf_counter()
            editor = self.open_editor(result)
            editor.persist()
            telemetry.stage_time_seconds["viewer_artifact_build"] = (
                time.perf_counter() - viewer_started
            )
            if final_clearance is not None:
                clearance_write_started = time.perf_counter()
                _write_json(
                    batch_root / "batch-clearance.json",
                    {
                        "schema": "treeline-stage-batch-clearance-v1",
                        "status": final_clearance.status,
                        "global_minimum_clearance_m": final_clearance.global_minimum_clearance_m,
                        "records": [record.to_dict() for record in final_clearance.records],
                        "diagnostics": dict(final_clearance.diagnostics),
                    },
                )
                telemetry.stage_time_seconds["clearance_risk_json_write"] = (
                    time.perf_counter() - clearance_write_started
                )
        final_status = (
            AnalysisRunStatus.COMPLETE
            if status == "COMPLETE"
            else AnalysisRunStatus.FAILED
        )
        self._persist_run(
            AnalysisRun(
                analysis_run_id=run_id,
                scan_id=self.bridge.scan.scan_id,
                span_ids=selected,
                algorithm_versions={
                    "ground": "PDAL_SMRF_HAG_NN_FAST",
                    "wire": "FAST_MATURE_SPAN_WIRE",
                    "vegetation": "FULL_RESOLUTION_ORIGINAL_INDEX",
                    "tree": "CROWN_STEM_HYBRID",
                    "reconciliation": "GLOBAL_ORIGINAL_INDEX_CORE_OWNERSHIP",
                },
                configuration_snapshot={"tree_backend": self.config.tree_backend},
                risk_policy_snapshot=self.risk_policy_snapshot,
                status=final_status,
                artifact_references={
                    "batch_manifest": manifest_path.relative_to(self.project_root).as_posix()
                },
            )
        )
        return replace(
            result,
            batch_manifest_reference=manifest_path.relative_to(self.project_root).as_posix(),
        )

    def latest_result(self) -> BatchAnalysisResult | None:
        return self._latest_result


__all__ = [
    "BatchAnalysisResult",
    "BatchAnalysisScheduler",
    "BatchSpanStatus",
    "BatchTelemetry",
]
