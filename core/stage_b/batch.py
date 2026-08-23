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
from core.stage_a.models import AnalysisRun, AnalysisRunStatus, Project, stable_id
from core.stage_a.persistence import save_project
from core.stage_a.risk import RiskPolicy
from core.stage_b.bridge import ProjectScienceBridge, SpanScienceBinding
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
            arrays = np.load(membership_path, allow_pickle=False)
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
        if reused_single_span and type(validated_duplicate_final) is int:
            duplicate_final = validated_duplicate_final
        else:
            final_index = np.asarray(reconciled.final_original_index, dtype=np.int64)
            duplicate_final = int(len(final_index) - len(np.unique(final_index)))
        diagnostics["duplicate_final_original_index"] = duplicate_final
        if duplicate_final != 0:
            raise ScienceAdapterError("DUPLICATE_FINAL_ORIGINAL_INDEX")
        return replace(reconciled, global_tree_records=tuple(records), diagnostics=diagnostics)

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
            self._latest_result = result
            return result

        previous_statuses = (previous or {}).get("span_statuses") or {}
        for span_id in selected:
            reused = (
                previous_statuses.get(span_id)
                in {
                    BatchSpanStatus.AUTO_ANALYZED.value,
                    BatchSpanStatus.REVIEW_REQUIRED.value,
                    BatchSpanStatus.FINALIZED.value,
                }
                and self._reused_result(previous or {}, span_id) is not None
            )
            if reused:
                previous_status = previous_statuses.get(span_id)
                statuses[span_id] = (
                    BatchSpanStatus(previous_status)
                    if previous_status in {
                        BatchSpanStatus.USER_REVIEWED.value,
                        BatchSpanStatus.FINALIZED.value,
                    }
                    else BatchSpanStatus.AUTO_ANALYZED
                )
                results[span_id] = self._reused_result(previous or {}, span_id)  # type: ignore[assignment]
                continue
            statuses[span_id] = BatchSpanStatus.RUNNING
            span_started = time.perf_counter()
            try:
                input_data = replace(
                    bindings[span_id].science_input,
                    validated_source_preparation_references=(preparation.manifest_reference,),
                )
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
        statuses = dict(statuses)
        successful = [status is not BatchSpanStatus.FAILED for status in statuses.values()]
        failed = any(status is BatchSpanStatus.FAILED for status in statuses.values())
        global_reconciled = None
        batch_diagnostics: dict[str, Any] = {}
        if not failed:
            reconciliation_started = time.perf_counter()
            try:
                global_reconciled = self._global_reconcile(bindings, results)
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
                value.wire_result
                for value in results.values()
                if value.wire_result is not None
            ]
            if wire_results and len(wire_results) == len(selected):
                clearance_publish_started = time.perf_counter()
                try:
                    # A one-Span global result has the exact same membership,
                    # wire geometry and risk threshold as the result already
                    # evaluated by the canonical executor.  Persistent IDs do
                    # not participate in distance science, so reuse is exact.
                    only = next(iter(results.values())) if len(selected) == 1 else None
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
        result = BatchAnalysisResult(
            analysis_run_id=run_id,
            project_id=self.project.project_id,
            scan_id=self.bridge.scan.scan_id,
            status=status,
            span_statuses=dict(statuses),
            span_results=dict(results),
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
