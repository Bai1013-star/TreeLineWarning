"""Stage-C desktop workflow orchestration.

This module is deliberately a product adapter, not a second science stack.  It
keeps the Stage-A sidecar authoritative, invokes the Stage-B bridge/scheduler,
and exposes only bounded review samples to the desktop viewer.  The fixture
runner exists solely for offline UI acceptance and is always labelled as such.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
from algorithms.folder_pipeline.tree_editor import load_edits_payload
from algorithms.spans.project_adapter import SourceSharedPreparation, SpanScienceResult
from core.config.fast_baseline import canonical_digest
from core.stage_a import (
    CapabilityState,
    CoordinateDeclaration,
    Line,
    LineRole,
    Project,
    Route,
    Scan,
    Span,
    Tower,
    UnitStatus,
    generate_spans_from_ordered_towers,
    load_project,
    save_project,
)
from core.stage_a.models import stable_id
from core.stage_a.risk import RiskPolicy
from core.stage_b import BatchAnalysisResult, BatchAnalysisScheduler, BatchSpanStatus
from core.stage_b.failures import failure_details
from core.stage_b.measurement_fields import clearance_components

from desktop.editor_tasks import editor_action, editor_tasks
from desktop.project_management import (
    describe_project,
)
from desktop.tower_viewer import overview as tower_overview
from desktop.tower_viewer import refine as refine_tower_pick
from desktop.source_integrity import PointCloudIntegrityError, validate_las_extent

WORKFLOW_SCHEMA = "treeline-desktop-workflow-v2"
WORKFLOW_MANIFEST = "desktop-workflow.json"
# This ceiling is display-only.  Exact tree membership always remains in the
# editor's original_index arrays and is never sampled for science or edits.
_MAX_DISPLAY_POINTS = 60_000
_DEFAULT_DISPLAY_POINTS = 30_000
_MAX_EDIT_PREVIEW_POINTS = 12_000


def derive_workspace_state(
    project: Mapping[str, Any],
    capability: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    tree_review_ready: bool,
    finalized: bool,
) -> str:
    """Derive product navigation from persisted project/science state only."""

    if project.get("is_desktop_home"):
        return "NO_PROJECT"
    if project.get("pointcloud") == "NOT_IMPORTED":
        return "PROJECT_CREATED"
    if not capability.get("unit_confirmed"):
        return "SOURCE_IMPORTED_UNIT_PENDING"
    if batch.get("status") == "RUNNING":
        return "ANALYSIS_RUNNING"
    if finalized:
        return "FINALIZED"
    if tree_review_ready:
        return "TREE_REVIEW_READY"
    if int(project.get("spans") or 0) > 0:
        return "TOPOLOGY_READY"
    return "TOWER_SELECTION"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid4().hex}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _dynamic_risk_snapshot(policy: RiskPolicy) -> dict[str, Any]:
    """Return the versioned product-policy snapshot, separate from science."""

    snapshot = policy.to_dict()
    snapshot["configuration_digest"] = canonical_digest(snapshot)
    return snapshot


def _risk_counts_from_saved_clearance(
    records: Any,
    policy: RiskPolicy,
) -> tuple[dict[str, int], int]:
    """Reclassify saved per-tree minima without invoking clearance science."""

    from core.stage_b.contact import ContactState, contact_evidence_from_clearance

    counts = {"DANGER": 0, "WARNING": 0, "SAFE": 0}
    unavailable = 0
    for record in records or ():
        value = (
            record.get("minimum_clearance_m")
            if isinstance(record, Mapping)
            else getattr(record, "minimum_clearance_m", None)
        )
        status = (
            record.get("status", "UNKNOWN")
            if isinstance(record, Mapping)
            else getattr(record, "status", "UNKNOWN")
        )
        explicit_contact = (
            record.get("contact_state")
            if isinstance(record, Mapping)
            else getattr(record, "contact_state", None)
        )
        if value is None:
            unavailable += 1
            continue
        evidence = contact_evidence_from_clearance(
            value,
            clearance_status=str(status),
        )
        contact = (
            True
            if explicit_contact == ContactState.CONTACT_CONFIRMED.value
            else False
            if explicit_contact == ContactState.CONTACT_NOT_CONFIRMED.value
            or evidence.state is ContactState.CONTACT_NOT_CONFIRMED
            else None
        )
        assessment = policy.classify(clearance_m=float(value), contact=contact)
        counts[assessment.risk.value] += 1
    return counts, unavailable


def _load_final_clearance(payload: Mapping[str, Any] | None) -> Any | None:
    """Hydrate the persisted final-clearance publication for restart/export."""

    if not payload:
        return None
    try:
        from algorithms.spans.clearance import SpanClearanceResult, SpanTreeWireRecord

        records = tuple(
            SpanTreeWireRecord(
                global_tree_id=int(item["global_tree_id"]),
                external_name=str(item["external_name"]),
                persistent_tree_id=item.get("persistent_tree_id"),
                line_id=int(item["line_id"]),
                span_id=int(item["span_id"]),
                span_name=str(item["span_name"]),
                nearest_wire_id=(
                    None if item.get("nearest_wire_id") is None else int(item["nearest_wire_id"])
                ),
                minimum_clearance_m=(
                    None
                    if item.get("minimum_clearance_m") is None
                    else float(item["minimum_clearance_m"])
                ),
                nearest_tree_original_index=(
                    None
                    if item.get("nearest_tree_original_index") is None
                    else int(item["nearest_tree_original_index"])
                ),
                nearest_tree_xyz_local_m=(
                    None
                    if item.get("nearest_tree_xyz_local_m") is None
                    else tuple(float(value) for value in item["nearest_tree_xyz_local_m"])
                ),
                nearest_wire_xyz_local_m=(
                    None
                    if item.get("nearest_wire_xyz_local_m") is None
                    else tuple(float(value) for value in item["nearest_wire_xyz_local_m"])
                ),
                tree_point_count=int(item.get("tree_point_count", 0)),
                evaluated_edge_count=int(item.get("evaluated_edge_count", 0)),
                status=str(item.get("status", "UNKNOWN")),
                danger_point_count=int(item.get("danger_point_count", 0)),
            )
            for item in payload.get("records", [])
        )
        diagnostics = dict(payload.get("diagnostics") or {})
        return SpanClearanceResult(
            records=records,
            status=str(payload.get("status", "UNKNOWN")),
            global_minimum_clearance_m=(
                None
                if payload.get("global_minimum_clearance_m") is None
                else float(payload["global_minimum_clearance_m"])
            ),
            edge_count=int(diagnostics.get("edge_count", 0)),
            diagnostics=diagnostics,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _unit_from_scan(scan: Any) -> UnitStatus:
    try:
        return UnitStatus(scan.capability_report["coordinate_declaration"]["unit_status"])
    except (KeyError, TypeError, ValueError):
        return UnitStatus.REQUIRES_CONFIRMATION


def _coordinate_from_scan(scan: Any) -> CoordinateDeclaration:
    raw = scan.capability_report.get("coordinate_declaration", {})
    try:
        return CoordinateDeclaration(
            raw.get("horizontal_crs"),
            raw.get("vertical_reference"),
            raw.get("linear_unit"),
            UnitStatus(raw.get("unit_status", UnitStatus.REQUIRES_CONFIRMATION.value)),
            raw.get("coordinate_kind", "LOCAL"),
        )
    except (TypeError, ValueError):
        return CoordinateDeclaration.unresolved()


def _empty_display() -> dict[str, Any]:
    return {
        "trees": [],
        "trees3d": {
            "xyz": [],
            "color": [],
            "tree_id_by_index": [],
            "source_tree_id_by_index": [],
            "original_index_by_index": [],
            "visible_by_index": [],
            "labels": [],
        },
        "background": {"xyz": [], "rgb": []},
        "wire": {"xyz": []},
        "wires3d": {"xyz": []},
        "tower": {"xyz": [], "labels": []},
    }


class DesktopWorkflow:
    """Persisted desktop state plus the actual Stage-B scheduler boundary."""

    def __init__(
        self,
        project_root: Path,
        *,
        resource_root: Path | None = None,
        desktop_home: bool = False,
    ) -> None:
        self.resource_root = resource_root or project_root
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._batch_error: str | None = None
        self._editor: Any | None = None
        self._result: BatchAnalysisResult | None = None
        self._editor_snapshot_cache: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        self._editor_display_cache: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        self._editor_display_basis_cache: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        self._finalize_thread: threading.Thread | None = None
        self._finalize_stage = "IDLE"
        self._finalize_message = ""
        self._finalize_error: str | None = None
        self._finalize_detail: str | None = None
        self._final_tree_records: dict[int, dict[str, Any]] = {}
        self._root = project_root.resolve()
        self._desktop_home = desktop_home
        self._logger = logging.getLogger(f"treeline.desktop.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._load_root_state()

    @property
    def project_root(self) -> Path:
        return self._root

    @property
    def log_path(self) -> Path:
        return self._root / "logs" / "application.log"

    def _log(self, event: str, **details: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"time": _now(), "event": event, **details}, ensure_ascii=False)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self._logger.info("%s %s", event, details)

    def _manifest_path(self) -> Path:
        return self._root / WORKFLOW_MANIFEST

    def _load_root_state(self) -> None:
        self._editor = None
        self._result = None
        self._batch_error = None
        self._editor_snapshot_cache = None
        self._editor_display_cache = None
        self._editor_display_basis_cache = None
        self._finalize_thread = None
        self._finalize_stage = "IDLE"
        self._finalize_message = ""
        self._finalize_error = None
        self._finalize_detail = None
        self._final_tree_records = {}
        manifest = _load_json(self._manifest_path()) or {}
        if manifest.get("schema_version") not in {None, WORKFLOW_SCHEMA}:
            self._log("workflow_manifest_ignored", reason="unsupported_schema")
        self._workflow = {
            "schema_version": WORKFLOW_SCHEMA,
            "risk_policy": dict(manifest.get("risk_policy") or RiskPolicy().to_dict()),
            "selected_span_ids": list(manifest.get("selected_span_ids") or []),
            "acceptance_mode": manifest.get("acceptance_mode"),
            "last_batch_id": manifest.get("last_batch_id"),
            "last_batch_status": manifest.get("last_batch_status"),
            "last_error": manifest.get("last_error"),
            "finalize_state": dict(manifest.get("finalize_state") or {}),
            "updated_at": manifest.get("updated_at", _now()),
        }
        self._log("project_open", project_root=str(self._root))

    @editor_action("PROJECT_CHANGE")
    def configure_root(self, project_root: Path) -> dict[str, Any]:
        with self._lock:
            self._root = project_root.resolve()
            self._desktop_home = False
            self._load_root_state()
            return self.project_projection()

    def _save_workflow(self) -> None:
        self._workflow["updated_at"] = _now()
        _atomic_json(self._manifest_path(), self._workflow)

    def _stage_project(self) -> Project:
        with self._lock:
            return load_project(self._root)

    def _scan(self, project: Project | None = None) -> Any | None:
        project = project or self._stage_project()
        return project.scans[-1] if project.scans else None

    def _latest_batch_path(self) -> Path | None:
        root = self._root / "artifacts" / "project-runs" / "batches"
        candidates = [path for path in root.glob("*/batch.json") if path.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def _latest_batch_manifest(self) -> dict[str, Any] | None:
        with self._lock:
            path = self._latest_batch_path()
            return _load_json(path) if path else None

    def _batch_result_from_manifest(
        self, manifest: Mapping[str, Any]
    ) -> BatchAnalysisResult | None:
        try:
            self._final_tree_records = {
                int(item["global_tree_id"]): dict(item)
                for item in ((manifest.get("global_reconciliation") or {}).get("trees") or [])
                if isinstance(item, Mapping) and item.get("global_tree_id") is not None
            }
            project = self._stage_project()
            scan = self._scan(project)
            if scan is None:
                return None
            artifact_root = (
                self._root
                / "artifacts"
                / "project-runs"
                / "batches"
                / str(manifest["analysis_run_id"])
            )
            span_results = {}
            for span_id, raw in (manifest.get("span_results") or {}).items():
                wire_result = None
                try:
                    from algorithms.spans.wire import load_span_wire_result

                    wire_result = load_span_wire_result(artifact_root / span_id / "wire")
                except (OSError, RuntimeError, ValueError, KeyError):
                    pass
                span_results[span_id] = SpanScienceResult(
                    tree_count=int(raw.get("tree_count", 0)),
                    minimum_clearance_m=raw.get("minimum_clearance_m"),
                    risk_threshold_m=float(raw.get("risk_threshold_m", 0.5)),
                    risk_counts=dict(raw.get("risk_counts") or {}),
                    diagnostics=dict(raw.get("diagnostics") or {}),
                    wire_result=wire_result,
                )
            statuses = {
                span_id: BatchSpanStatus(value)
                for span_id, value in (manifest.get("span_statuses") or {}).items()
            }
            telemetry_raw = manifest.get("telemetry") or {}
            from core.stage_b.batch import BatchTelemetry

            source_point_count = int(
                telemetry_raw.get("SOURCE_POINT_COUNT", 0) or 0
            )
            if source_point_count <= 0:
                for raw in (manifest.get("span_results") or {}).values():
                    candidate = (
                        ((raw.get("diagnostics") or {}).get("source_spatial_query") or {}).get(
                            "project_source_points"
                        )
                        if isinstance(raw, Mapping)
                        else None
                    )
                    if type(candidate) is int and candidate > 0:
                        source_point_count = candidate
                        break

            telemetry = BatchTelemetry(
                source_point_count=source_point_count,
                span_count=int(telemetry_raw.get("SPAN_COUNT", len(statuses)) or 0),
                source_prep_count=int(telemetry_raw.get("SOURCE_PREP_COUNT", 0) or 0),
                source_full_pass_count=int(telemetry_raw.get("SOURCE_FULL_PASS_COUNT", 0) or 0),
                ground_prep_count=int(telemetry_raw.get("GROUND_PREP_COUNT", 0) or 0),
                span_local_read_count=int(telemetry_raw.get("SPAN_LOCAL_READ_COUNT", 0) or 0),
                cache_hit_count=int(telemetry_raw.get("CACHE_HIT_COUNT", 0) or 0),
                cache_miss_count=int(telemetry_raw.get("CACHE_MISS_COUNT", 0) or 0),
                total_time_seconds=float(telemetry_raw.get("TOTAL_TIME", 0.0) or 0.0),
                per_span_time_seconds=dict(telemetry_raw.get("PER_SPAN_TIME") or {}),
                stage_time_seconds=dict(telemetry_raw.get("STAGE_TIME") or {}),
            )
            scheduler = BatchAnalysisScheduler(
                self._root, project, scan.scan_id, risk_policy=self._risk_policy()
            )
            global_reconciled = scheduler._load_persisted_global(str(manifest["analysis_run_id"]))
            # The persisted global sidecar deliberately stays at the immutable
            # pre-editor membership so TreeEditSession can replay its journal.
            # Finalization metadata, however, is published in batch.json. Keep
            # revision markers when hydrating so exact post-editor
            # clearance/danger evidence is not incorrectly marked stale.
            published_diagnostics = dict(
                (manifest.get("global_reconciliation") or {}).get("diagnostics") or {}
            )
            if global_reconciled is not None and published_diagnostics.get(
                "editor_schema"
            ) == "treeline-stage-b-batch-editor-v1":
                operation_count = published_diagnostics.get("editor_operation_count")
                if type(operation_count) is int:
                    diagnostics = dict(global_reconciled.diagnostics)
                    diagnostics.update(
                        {
                            "editor_schema": "treeline-stage-b-batch-editor-v1",
                            "editor_operation_count": operation_count,
                        }
                    )
                    protected_digest = published_diagnostics.get(
                        'protected_infrastructure_index_sha256')
                    if (isinstance(protected_digest, str) and len(protected_digest) == 64
                            and all(c in '0123456789abcdef' for c in protected_digest)):
                        diagnostics['protected_infrastructure_index_sha256'] = protected_digest
                    global_reconciled = replace(
                        global_reconciled,
                        diagnostics=diagnostics,
                    )
            return BatchAnalysisResult(
                analysis_run_id=str(manifest["analysis_run_id"]),
                project_id=str(manifest["project_id"]),
                scan_id=str(manifest["scan_id"]),
                status=str(manifest.get("status", "FAILED")),
                span_statuses=statuses,
                span_results=span_results,
                telemetry=telemetry,
                risk_policy_snapshot=dict(manifest.get("risk_policy_snapshot") or {}),
                global_reconciled=global_reconciled,
                final_clearance=_load_final_clearance(manifest.get("final_clearance")),
                batch_diagnostics=dict(manifest.get("batch_diagnostics") or {}),
                batch_manifest_reference=manifest.get("batch_manifest_reference"),
                prediction_export_reference=manifest.get("prediction_export_reference"),
            )
        except (KeyError, TypeError, ValueError, OSError, RuntimeError):
            return None

    def _current_result(self) -> BatchAnalysisResult | None:
        if self._result is not None:
            return self._result
        manifest = self._latest_batch_manifest()
        if manifest is None:
            return None
        self._result = self._batch_result_from_manifest(manifest)
        return self._result

    def _risk_policy(self) -> RiskPolicy:
        raw = self._workflow.get("risk_policy") or {}
        return RiskPolicy(
            risk_policy_version=str(raw.get("risk_policy_version", "risk-policy-v1")),
            warning_clearance_threshold_m=float(raw.get("warning_clearance_threshold_m", 0.5)),
        )

    def project_projection(self) -> dict[str, Any]:
        """Return the product-facing projection used by Project Home."""
        with self._lock:
            descriptor = describe_project(self._root)
            project = None
            try:
                project = self._stage_project()
            except (FileNotFoundError, ValueError):
                pass
            if project is not None:
                scan = self._scan(project)
                descriptor.update(
                    {
                        "capability_report": scan.capability_report if scan else None,
                        "source_identity": scan.source_identity.to_dict() if scan else None,
                        "line_records": [line.to_dict() for line in project.lines],
                        "tower_records": [tower.to_dict() for tower in project.towers],
                        "span_records": [span.to_dict() for span in project.spans],
                        "main_lines": sum(
                            line.line_role is LineRole.MAIN for line in project.lines
                        ),
                        "branch_lines": sum(
                            line.line_role is LineRole.BRANCH for line in project.lines
                        ),
                        "risk_policy": self._risk_policy().to_dict(),
                        "gis_interface_ready": True,
                        "real_gis_format": "PENDING",
                        "real_gis_acceptance": "PENDING_REAL_INPUT",
                        "auto_detected_interface": "RESERVED_ONLY",
                        "blind_auto_tower_detector_implemented": False,
                        "stage_b_scheduler": "AVAILABLE",
                    }
                )
            batch = self._latest_batch_manifest()
            descriptor["workflow"] = dict(self._workflow)
            descriptor["is_desktop_home"] = self._desktop_home
            descriptor["last_batch"] = batch
            descriptor["last_error"] = self._batch_error or self._workflow.get("last_error")
            descriptor["resume_state"] = (
                "RESUMABLE"
                if batch and batch.get("status") in {"COMPLETE", "PARTIAL_FAILURE"}
                else "IN_PROGRESS"
                if self._thread and self._thread.is_alive()
                else "NONE"
            )
            descriptor["real_multi_span_science"] = (
                "PENDING_SUITABLE_INPUT"
                if self._workflow.get("acceptance_mode") == "FIXTURE"
                else "AVAILABLE_OR_PENDING"
            )
            return descriptor

    def workspace_projection(self) -> dict[str, Any]:
        """Return the single derived view state plus its authoritative inputs."""

        project = self.project_projection()
        capability = self.capability_projection()
        batch = self.batch_projection()
        result = self._result
        tree_review_ready = bool(
            result and result.global_reconciled is not None
        ) or batch.get("global_reconciliation_tree_count") is not None
        # Keep the home-screen state aligned with the current selected batch
        # publication.  The project can contain unselected WAITING spans, and
        # process-local finalize flags do not survive a restart; the modern
        # persisted gate covers both cases without rebuilding final arrays.
        from desktop.current_report import current_finalized_gate

        finalized = current_finalized_gate(self, hydrate_editor=False)["status"] == "PASS"
        ui_state = derive_workspace_state(
            project,
            capability,
            batch,
            tree_review_ready=tree_review_ready,
            finalized=finalized,
        )
        return {
            "project": project,
            "capability": capability,
            "batch": batch,
            "ui_state": ui_state,
            "tree_review_ready": tree_review_ready,
            "finalized": finalized,
        }

    def capability_projection(self) -> dict[str, Any]:
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            return {
                "available": False,
                "requires_confirmation": True,
                "message": "请先导入 LAS/LAZ 点云。",
            }
        report = dict(scan.capability_report)
        report["available"] = True
        report["unit_confirmed"] = _unit_from_scan(scan) in {
            UnitStatus.METRE,
            UnitStatus.FOOT,
            UnitStatus.LOCAL_METRIC,
        }
        report["source_identity"] = scan.source_identity.to_dict()
        report["required_for_science"] = ["POINT_CLOUD", "XYZ", "LINEAR_UNIT"]
        return report

    def confirm_unit(self, unit: str) -> dict[str, Any]:
        try:
            declared = UnitStatus(str(unit).upper())
        except ValueError as exc:
            raise ValueError("unit must be METRE, FOOT, or LOCAL_METRIC") from exc
        if declared not in {UnitStatus.METRE, UnitStatus.FOOT, UnitStatus.LOCAL_METRIC}:
            raise ValueError("unit confirmation must be explicit and metric-compatible")
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            raise ValueError("POINT_CLOUD_NOT_IMPORTED")
        coordinate = _coordinate_from_scan(scan)
        updated_coordinate = CoordinateDeclaration(
            coordinate.horizontal_crs,
            coordinate.vertical_reference,
            "ft" if declared is UnitStatus.FOOT else "m",
            declared,
            coordinate.coordinate_kind,
        )
        report = dict(scan.capability_report)
        report["coordinate_declaration"] = updated_coordinate.to_dict()
        report["capabilities"] = {
            **dict(report.get("capabilities") or {}),
            "LINEAR_UNIT": CapabilityState.AVAILABLE.value,
        }
        updated_scan = replace(scan, capability_report=report)
        scans = tuple(
            updated_scan if item.scan_id == scan.scan_id else item for item in project.scans
        )
        save_project(replace(project, scans=scans), self._root)
        self._log("capability_unit_confirmed", unit=declared.value, scan_id=scan.scan_id)
        return self.capability_projection()

    def set_risk_policy(self, threshold_m: float) -> dict[str, Any]:
        started = time.perf_counter()
        threshold = float(threshold_m)
        policy = RiskPolicy(warning_clearance_threshold_m=threshold)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("分析正在运行，完成后才能调整风险阈值。")
            if self._finalize_thread is not None and self._finalize_thread.is_alive():
                raise ValueError("正在生成复核结果，完成后才能调整风险阈值。")

            result = self._current_result()
            records = (
                getattr(getattr(result, "final_clearance", None), "records", ())
                if result is not None
                else ()
            )
            risk_counts, unavailable = _risk_counts_from_saved_clearance(records, policy)
            snapshot = _dynamic_risk_snapshot(policy)
            reclassification = {
                "source": "SAVED_EXACT_MINIMUM_CLEARANCE",
                "risk_counts": risk_counts,
                "reclassified_tree_count": int(sum(risk_counts.values())),
                "unavailable_tree_count": int(unavailable),
                "science_recomputed": False,
                "ground_rerun": False,
                "wire_rerun": False,
                "tree_rerun": False,
                "dbh_rerun": False,
                "clearance_rerun": False,
            }

            updated_result = result
            if result is not None:
                diagnostics = dict(result.batch_diagnostics)
                diagnostics["risk_reclassification"] = dict(reclassification)
                updated_result = replace(
                    result,
                    risk_policy_snapshot=snapshot,
                    batch_diagnostics=diagnostics,
                )

            manifest_path = self._latest_batch_path()
            manifest = _load_json(manifest_path) if manifest_path is not None else None
            if (
                updated_result is not None
                and manifest_path is not None
                and manifest is not None
                and manifest.get("analysis_run_id") == updated_result.analysis_run_id
            ):
                _atomic_json(manifest_path, updated_result.to_dict())

            self._result = updated_result
            if self._editor is not None and updated_result is not None:
                self._editor.result = updated_result
            self._editor_display_cache = None
            self._workflow["risk_policy"] = policy.to_dict()
            self._save_workflow()

        elapsed = time.perf_counter() - started
        self._log(
            "risk_policy_updated",
            threshold_m=threshold,
            elapsed_seconds=elapsed,
            **reclassification,
        )
        return {
            **policy.to_dict(),
            "reclassification": {
                **reclassification,
                "elapsed_seconds": elapsed,
            },
        }

    def save_manual_towers(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            raise ValueError("POINT_CLOUD_NOT_IMPORTED")
        unit = _unit_from_scan(scan)
        if unit not in {UnitStatus.METRE, UnitStatus.FOOT, UnitStatus.LOCAL_METRIC}:
            raise ValueError("UNIT_CONFIRMATION_REQUIRED")
        rows = payload.get("towers")
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError("MANUAL_TOWER_SEQUENCE_REQUIRES_TWO_TOWERS")
        line_name = str(payload.get("line_name") or "Main Line").strip()
        role = LineRole(str(payload.get("line_role", LineRole.MAIN.value)).upper())
        parent_line_id = payload.get("parent_line_id") or None
        if role is LineRole.BRANCH and not parent_line_id:
            raise ValueError("BRANCH_LINE_REQUIRES_PARENT_LINE")
        if role is LineRole.BRANCH and parent_line_id not in {
            line.line_id for line in project.lines
        }:
            raise ValueError("BRANCH_PARENT_LINE_NOT_FOUND")
        coordinate = _coordinate_from_scan(scan)
        towers: list[Tower] = []
        for index, raw in enumerate(rows, start=1):
            if not isinstance(raw, Mapping):
                raise ValueError("tower row must be an object")
            try:
                position = tuple(float(raw[key]) for key in ("x", "y", "z"))
                height = float(raw["height_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("tower coordinates and height must be numeric") from exc
            if height <= 0.0:
                raise ValueError("tower height must be positive")
            label = str(raw.get("display_name") or f"T{index:03d}").strip()
            towers.append(
                Tower(
                    stable_id("tower"),
                    None,
                    label,
                    position,
                    coordinate,
                    "MANUAL_PICK",
                    {"height_m": height, "manual_sequence_index": index},
                )
            )
        line_id = stable_id("line")
        line = Line(
            line_id,
            line_name,
            tuple(tower.tower_id for tower in towers),
            (),
            {"source": "MANUAL_PICK", "desktop_stage": "C"},
            role,
            parent_line_id,
        )
        route = Route(
            stable_id("route"),
            line_id,
            line_name,
            line.ordered_tower_ids,
            (),
            role,
            None,
            {"source": "MANUAL_PICK", "desktop_stage": "C"},
        )
        spans = generate_spans_from_ordered_towers(line)
        retained_lines = tuple(
            item for item in project.lines if item.source_provenance.get("desktop_stage") != "C"
        )
        retained_routes = tuple(
            item for item in project.routes if item.source_provenance.get("desktop_stage") != "C"
        )
        retained_towers = tuple(
            item
            for item in project.towers
            if item.tower_id
            not in {
                old_id
                for old_line in project.lines
                if old_line.source_provenance.get("desktop_stage") == "C"
                for old_id in old_line.ordered_tower_ids
            }
        )
        retained_spans = tuple(
            item
            for item in project.spans
            if item.line_id in {line.line_id for line in retained_lines}
        )
        valid_span_ids = {item.span_id for item in (*retained_spans, *spans)}
        retained_analysis_runs = tuple(
            run
            for run in project.analysis_runs
            if set(run.span_ids).issubset(valid_span_ids)
        )
        updated = replace(
            project,
            lines=(*retained_lines, line),
            routes=(*retained_routes, route),
            towers=(*retained_towers, *towers),
            spans=(*retained_spans, *spans),
            analysis_runs=retained_analysis_runs,
        )
        save_project(updated, self._root)
        self._result = None
        self._editor = None
        self._editor_snapshot_cache = None
        self._editor_display_cache = None
        self._editor_display_basis_cache = None
        self._workflow.update(
            {
                "selected_span_ids": [],
                "last_batch_id": None,
                "last_batch_status": None,
                "last_error": None,
                "finalize_state": {"status": "IDLE"},
            }
        )
        self._save_workflow()
        self._log(
            "manual_tower_sequence_confirmed",
            line_id=line_id,
            tower_count=len(towers),
            span_count=len(spans),
            invalidated_analysis_runs=len(project.analysis_runs) - len(retained_analysis_runs),
        )
        return self.project_projection()

    def tower_viewer_overview(self) -> dict[str, Any]:
        """Bounded viewer-only source sample for the REAL manual picker."""
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None or _unit_from_scan(scan) not in {
            UnitStatus.METRE, UnitStatus.FOOT, UnitStatus.LOCAL_METRIC,
        }:
            raise ValueError("请先导入点云并确认单位。")
        source = Path(scan.point_cloud_source)
        if not source.is_file():
            raise ValueError("原始点云不可读取，请重新选择 LAS/LAZ 文件。")
        self._log("tower_viewer_overview_requested", source=str(source))
        try:
            payload = tower_overview(
                source,
                cache_dir=self._root / "viewer_cache" / "tower_overview",
                source_identity=scan.source_identity.to_dict(),
            )
        except Exception as exc:
            self._log("tower_viewer_overview_failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, PointCloudIntegrityError):
                raise
            raise ValueError("点云显示加载失败，请重试。") from exc
        self._log(
            "tower_viewer_overview_ready",
            points=payload["display_point_count"],
            cache=payload["viewer_cache"],
            timings=payload["diagnostic_timings"],
        )
        return payload

    def refine_tower_pick(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Refine an operator XY seed against a read-only local RAW crop."""
        project = self._stage_project()
        scan = self._scan(project)
        if scan is None:
            raise ValueError("POINT_CLOUD_NOT_IMPORTED")
        try:
            x, y = float(payload["x"]), float(payload["y"])
            z = float(payload["z"]) if payload.get("z") is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("杆塔点击坐标无效") from exc
        try:
            result = refine_tower_pick(
                Path(scan.point_cloud_source),
                x,
                y,
                z=z,
                cache_dir=self._root / "viewer_cache" / "tower_overview",
                source_identity=scan.source_identity.to_dict(),
            )
        except Exception as exc:
            self._log("tower_pick_refine_failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, PointCloudIntegrityError):
                raise
            raise ValueError("杆塔局部点云加载失败，请缩放后重试。") from exc
        self._log(
            "tower_pick_refined",
            x=result["tower"]["x"],
            y=result["tower"]["y"],
            picked_source_xyz=result["tower"]["picked_source_xyz"],
            timings=result["diagnostic_timings"],
        )
        return result

    def _fixture_preparer(self, _root: Path, input_data: Any) -> SourceSharedPreparation:
        return SourceSharedPreparation(
            source_binding_digest="f" * 64,
            configuration_digest=input_data.science_configuration_digest,
            manifest_reference="artifacts/project-runs/source-shared/fixture-preparation.json",
            ground_preparation={"coarse_ground_reference": "FIXTURE_GROUND_REFERENCE"},
            telemetry={"source_prep_count": 1, "ground_prep_count": 1, "cache_miss_count": 1},
        )

    @staticmethod
    def _fixture_runner(
        input_data: Any, _preparation: SourceSharedPreparation, _output_root: Path
    ) -> SpanScienceResult:
        span_number = max(0, int(str(input_data.span_id)[1:]) - 1)
        start = input_data.from_endpoint_geometry.get("base_xyz_absolute_m", [0.0, 0.0, 0.0])
        end = input_data.to_endpoint_geometry.get("base_xyz_absolute_m", [20.0, 0.0, 0.0])
        origin = input_data.source_identity.get("origin_context", [0.0, 0.0, 0.0])
        xyz_abs = np.asarray(
            [
                [
                    float(start[0]) + 0.35 * (float(end[0]) - float(start[0])),
                    float(start[1]) + 1.5,
                    6.0,
                ],
                [
                    float(start[0]) + 0.50 * (float(end[0]) - float(start[0])),
                    float(start[1]) + 1.5,
                    8.0,
                ],
                [
                    float(start[0]) + 0.65 * (float(end[0]) - float(start[0])),
                    float(start[1]) + 1.5,
                    7.0,
                ],
            ],
            dtype=np.float64,
        )
        xyz_local = xyz_abs - np.asarray(origin, dtype=np.float64)
        indices = np.arange(
            span_number * 10_000, span_number * 10_000 + len(xyz_local), dtype=np.int64
        )
        span = SimpleNamespace(
            line_id=1,
            line_name=input_data.group_id,
            span_id=span_number,
            span_name=input_data.span_id,
            start_xyz_absolute_m=tuple(float(v) for v in start),
            end_xyz_absolute_m=tuple(float(v) for v in end),
            length_m=float(input_data.span_geometry.get("length_m", 20.0)),
            half_width_m=8.0,
        )
        outcome = SimpleNamespace(
            span=span,
            raw_original_index=indices,
            raw_xyz_local_m=xyz_local,
            raw_tree_id=np.zeros(len(indices), dtype=np.int32),
            in_core=np.ones(len(indices), dtype=bool),
            tree_records=(
                {
                    "tree_id": 0,
                    "apex_xy_local_m": [float(xyz_local[1, 0]), float(xyz_local[1, 1])],
                    "apex_height_m": 8.0,
                },
            ),
        )
        return SpanScienceResult(
            tree_count=1,
            minimum_clearance_m=0.75,
            risk_threshold_m=0.5,
            risk_counts={"safe": 1},
            diagnostics={
                "fixture_acceptance": True,
                "ground": "fixture-contract-only",
                "wire": "fixture-contract-only",
                "vegetation": "FULL_RESOLUTION_ORIGINAL_INDEX",
                "tree_backend": "CROWN_STEM_HYBRID",
                "reconciliation": "GLOBAL_ORIGINAL_INDEX_CORE_OWNERSHIP",
            },
            tree_outcome=outcome,
        )

    def _run_scheduler(self, selected: tuple[str, ...], mode: str, run_id: str) -> None:
        try:
            # Stage-B persists the Stage-A sidecar at several lifecycle points.
            # Serialize reads/writes in the desktop process so Windows does not
            # observe a reader holding the target while os.replace publishes it.
            with self._lock:
                project = self._stage_project()
                scan = self._scan(project)
                if scan is None:
                    raise ValueError("POINT_CLOUD_NOT_IMPORTED")
                kwargs: dict[str, Any] = {
                    "risk_policy": self._risk_policy(),
                }
                execution_mode = self._validate_execution_mode(
                    self._workflow.get("execution_mode", "AUTO")
                )
                acceleration_mode = self._validate_acceleration_mode(
                    self._workflow.get("acceleration_mode", "AUTO")
                )
                # The released serial scheduler predates the optional mode API.
                # Never depend on an uncommitted parallel scheduler to start a batch.
                if "execution_mode" in inspect.signature(BatchAnalysisScheduler).parameters:
                    kwargs["execution_mode"] = execution_mode
                if "acceleration_mode" in inspect.signature(BatchAnalysisScheduler).parameters:
                    kwargs["acceleration_mode"] = acceleration_mode
                if mode == "FIXTURE":
                    kwargs.update(
                        {
                            "science_runner": self._fixture_runner,
                            "source_preparer": self._fixture_preparer,
                            "source_access_enabled": False,
                        }
                    )
                scheduler = BatchAnalysisScheduler(self._root, project, scan.scan_id, **kwargs)
                self._log("batch_start", mode=mode, run_id=run_id, span_ids=list(selected))
                result = scheduler.run(selected, analysis_run_id=run_id)
                for span_id, span_status in result.span_statuses.items():
                    if span_status is not BatchSpanStatus.FAILED:
                        continue
                    failure = failure_details(result.span_results[span_id].diagnostics)
                    self._log(
                        "span_analysis_failure",
                        run_id=run_id,
                        span_id=span_id,
                        failure_stage=(failure or {}).get("failure_stage"),
                        failure_code=(failure or {}).get("failure_code"),
                        technical_message=(failure or {}).get("technical_message"),
                    )
            with self._lock:
                self._result = result
                self._editor = None
                self._batch_error = None
                self._workflow.update(
                    {
                        "acceptance_mode": mode,
                        "last_batch_id": run_id,
                        "last_batch_status": result.status,
                        "last_error": None,
                        "selected_span_ids": list(selected),
                        "finalize_state": {"run_id": run_id, "status": "IDLE"},
                    }
                )
                self._save_workflow()
            self._log("batch_finish", mode=mode, run_id=run_id, status=result.status)
        except Exception as exc:  # surfaced to the UI and local log
            with self._lock:
                self._batch_error = f"{type(exc).__name__}: {exc}"
                self._workflow.update(
                    {
                        "last_batch_id": run_id,
                        "last_batch_status": "FAILED",
                        "last_error": self._batch_error,
                    }
                )
                self._save_workflow()
            self._log("batch_failure", mode=mode, run_id=run_id, error=self._batch_error)
        finally:
            with self._lock:
                self._thread = None

    def start_batch(
        self,
        span_ids: list[str],
        *,
        mode: str = "REAL",
        execution_mode: str = "AUTO",
        acceleration_mode: str = "AUTO",
    ) -> dict[str, Any]:
        mode = str(mode).upper()
        if mode not in {"REAL", "FIXTURE"}:
            raise ValueError("batch mode must be REAL or FIXTURE")
        execution_mode = self._validate_execution_mode(execution_mode)
        acceleration_mode = self._validate_acceleration_mode(acceleration_mode)
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("BATCH_ALREADY_RUNNING")
            project = self._stage_project()
            scan = self._scan(project)
            if scan is None:
                raise ValueError("POINT_CLOUD_NOT_IMPORTED")
            if mode == "REAL":
                validate_las_extent(Path(scan.point_cloud_source))
            if _unit_from_scan(scan) not in {
                UnitStatus.METRE,
                UnitStatus.FOOT,
                UnitStatus.LOCAL_METRIC,
            }:
                raise ValueError("UNIT_CONFIRMATION_REQUIRED")
            known = {span.span_id for span in project.spans}
            selected = tuple(dict.fromkeys(str(item) for item in span_ids if str(item) in known))
            if not selected:
                raise ValueError("BATCH_REQUIRES_SELECTED_SPAN")
            run_id = stable_id("analysis")
            source_cache_hint = self._source_cache_hint(project, scan)
            source_bytes = Path(scan.point_cloud_source).stat().st_size if scan.point_cloud_source else 0
            required_bytes = max(512 * 1024**2, source_bytes * (1 if source_cache_hint == "REUSE" else 3))
            if shutil.disk_usage(self._root).free < required_bytes:
                raise OSError(28, f"项目盘 {self._root.anchor} 空间不足；本次处理至少预留 {required_bytes / 1024**3:.1f} GB 后重试")
            self._workflow.update(
                {
                    "selected_span_ids": list(selected),
                    "last_batch_id": run_id,
                    "last_batch_status": "RUNNING",
                    "last_error": None,
                    "acceptance_mode": mode,
                    "execution_mode": execution_mode,
                    "acceleration_mode": acceleration_mode,
                    "source_cache_hint": source_cache_hint,
                    "finalize_state": {"run_id": run_id, "status": "IDLE"},
                }
            )
            self._batch_error = None
            self._finalize_stage = "IDLE"
            self._finalize_message = ""
            self._finalize_error = None
            self._finalize_detail = None
            self._save_workflow()
            self._thread = threading.Thread(
                target=self._run_scheduler, args=(selected, mode, run_id), daemon=True
            )
            self._thread.start()
            return self.batch_projection()

    @staticmethod
    def _validate_execution_mode(value: str) -> str:
        mode = str(value).upper()
        if mode not in {"AUTO", "SERIAL", "PARALLEL"}:
            raise ValueError("execution_mode must be AUTO, SERIAL, or PARALLEL")
        if (
            mode == "PARALLEL"
            and "execution_mode" not in inspect.signature(BatchAnalysisScheduler).parameters
        ):
            raise ValueError("PARALLEL_UNAVAILABLE: use AUTO or SERIAL in this release")
        return mode

    @staticmethod
    def _validate_acceleration_mode(value: str) -> str:
        mode = str(value).upper()
        if mode not in {"AUTO", "CPU", "GPU"}:
            raise ValueError("acceleration_mode must be AUTO, CPU, or GPU")
        return mode

    def retry_failed(self, *, execution_mode: str | None = None) -> dict[str, Any]:
        execution_mode = self._validate_execution_mode(
            self._workflow.get("execution_mode", "AUTO")
            if execution_mode is None
            else execution_mode
        )
        manifest = self._latest_batch_manifest()
        if not manifest:
            raise ValueError("NO_BATCH_TO_RETRY")
        results = manifest.get("span_results") or {}
        recoverable_failed = []
        nonrecoverable = []
        for span_id, status in (manifest.get("span_statuses") or {}).items():
            if status != BatchSpanStatus.FAILED.value:
                continue
            failure = failure_details((results.get(span_id) or {}).get("diagnostics"))
            if failure is not None and failure.get("recoverable") is False:
                nonrecoverable.append(span_id)
            else:
                recoverable_failed.append(span_id)
        if not recoverable_failed:
            if nonrecoverable:
                raise ValueError("当前失败由杆塔/档距证据不足导致，请返回杆塔选取重新确认。")
            raise ValueError("NO_FAILED_SPAN_TO_RETRY")
        project = self._stage_project()
        topology_order = [span.span_id for span in project.spans]
        known = set(topology_order)
        original_selected = set((manifest.get("span_statuses") or {}).keys())
        unknown = sorted(original_selected - known)
        if unknown:
            raise ValueError("BATCH_RETRY_UNKNOWN_SPAN:" + ",".join(unknown))
        selected = tuple(span_id for span_id in topology_order if span_id in original_selected)
        if not selected:
            raise ValueError("BATCH_RETRY_REQUIRES_ORIGINAL_SELECTION")
        mode = str(self._workflow.get("acceptance_mode") or "REAL").upper()
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("BATCH_ALREADY_RUNNING")
            run_id = str(manifest["analysis_run_id"])
            self._finalize_stage = "IDLE"
            self._finalize_message = ""
            self._finalize_error = None
            self._finalize_detail = None
            self._workflow.update(
                {
                    "selected_span_ids": list(selected),
                    "last_batch_status": "RUNNING",
                    "execution_mode": execution_mode,
                    "finalize_state": {"run_id": run_id, "status": "IDLE"},
                }
            )
            self._save_workflow()
            self._thread = threading.Thread(
                target=self._run_scheduler, args=(tuple(selected), mode, run_id), daemon=True
            )
            self._thread.start()
        return self.batch_projection()

    def batch_projection(self) -> dict[str, Any]:
        project = self._stage_project()
        manifest = self._latest_batch_manifest()
        batch_status = (manifest or {}).get("status") or self._workflow.get("last_batch_status") or "NOT_STARTED"
        # Science can publish COMPLETE before the desktop/editor artifacts are
        # written. A subsequent disk/write failure must not look like success.
        if self._workflow.get("last_batch_status") in {"FAILED", "RUNNING"}:
            batch_status = self._workflow["last_batch_status"]
        policy = self._risk_policy()
        final_clearance = (manifest or {}).get("final_clearance") or {}
        risk_counts, unavailable_risk_count = _risk_counts_from_saved_clearance(
            final_clearance.get("records") if isinstance(final_clearance, Mapping) else (),
            policy,
        )
        statuses = dict((manifest or {}).get("span_statuses") or {})
        global_reconciliation = (manifest or {}).get("global_reconciliation") or {}
        global_tree_count = global_reconciliation.get("tree_count")
        if type(global_tree_count) is not int or global_tree_count < 0:
            global_tree_count = None
        selected = set(self._workflow.get("selected_span_ids") or [])
        spans = []
        source_preparation_failed = False
        for span in project.spans:
            status = statuses.get(span.span_id, BatchSpanStatus.WAITING.value)
            result = ((manifest or {}).get("span_results") or {}).get(span.span_id) or {}
            failure = failure_details(result.get("diagnostics"))
            source_preparation_failed = source_preparation_failed or bool(
                status == "FAILED"
                and failure
                and failure.get("failure_stage") in {"POINT_CLOUD_PREPARATION", "GROUND"}
            )
            spans.append(
                {
                    "span_id": span.span_id,
                    "display_name": self._span_name(project, span),
                    "line_id": span.line_id,
                    "start_tower_id": span.start_tower_id,
                    "end_tower_id": span.end_tower_id,
                    "selected": span.span_id in selected,
                    "status": status,
                    "stage": failure.get("failure_stage_label", "处理失败")
                    if status == "FAILED" and failure
                    else "地面 · 导线 · 植被 · 单木 · 跨档整理"
                    if status not in {"WAITING", "FAILED"}
                    else ("待运行" if status == "WAITING" else "处理失败"),
                    "retryable": status == BatchSpanStatus.FAILED.value
                    and (failure is None or failure.get("recoverable") is not False),
                    "failure": failure,
                    "summary": result,
                }
            )
        return {
            "ok": True,
            "batch_id": (manifest or {}).get("analysis_run_id")
            or self._workflow.get("last_batch_id"),
            "status": batch_status,
            "acceptance_mode": self._workflow.get("acceptance_mode"),
            "execution_mode": self._workflow.get("execution_mode", "AUTO"),
            "acceleration_mode": self._workflow.get("acceleration_mode", "AUTO"),
            "execution": dict(((manifest or {}).get("telemetry") or {}).get("EXECUTION") or {}),
            "source_preparation": (
                "FAILED" if source_preparation_failed else "DONE" if manifest else "WAITING"
            ),
            "source_cache_hint": self._workflow.get("source_cache_hint", "COLD"),
            "source_cache_message": (
                "已复用点云缓存，正在进行档距分析。"
                if self._workflow.get("source_cache_hint") == "REUSE"
                else "首次处理大型点云，正在建立可复用缓存。"
            ),
            "spans": spans,
            # The per-Span summaries are pre-reconciliation diagnostics.  The
            # dashboard must use the persisted global ownership count instead
            # of adding those summaries together.
            "global_reconciliation_tree_count": global_tree_count,
            "final_tree_count": global_tree_count,
            "tree_count_source": "PERSISTED_GLOBAL_RECONCILIATION",
            "error": self._batch_error or self._workflow.get("last_error"),
            "risk_policy": policy.to_dict(),
            "risk_counts": risk_counts,
            "risk_unavailable_tree_count": unavailable_risk_count,
            "stage_timings_seconds": dict(
                ((manifest or {}).get("telemetry") or {}).get("STAGE_TIME") or {}
            ),
            "duplicate_final_original_index": int(
                ((manifest or {}).get("global_reconciliation") or {}).get(
                    "duplicate_final_original_index", 0
                )
            ),
        }

    def _source_cache_hint(self, project: Project, scan: Scan) -> str:
        """Return a metadata-only UX hint; production validation still fails closed."""

        try:
            from algorithms.spans.project_adapter import (
                PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST,
                wire_role_index_configuration_digest,
            )
            from core.config.fast_baseline import canonical_digest
            from core.stage_b.bridge import ProjectScienceBridge
            from core.stage_b.role_preparation import (
                _global_cache_root,
                _global_role_cache_key,
                _span_admission_digest,
            )

            identity = scan.source_identity.to_dict()
            local = (
                self._root
                / "artifacts"
                / "project-runs"
                / "source-shared"
                / canonical_digest(identity)
                / "preparation.json"
            )
            sha256 = identity.get("sha256")
            if not isinstance(sha256, str) or len(sha256) != 64:
                return "COLD"
            import laspy

            with laspy.open(Path(scan.point_cloud_source), mode="r") as reader:
                point_count = int(reader.header.point_count)
            bridge = ProjectScienceBridge(
                project,
                scan.scan_id,
                risk_policy=self._risk_policy(),
            )
            inputs = tuple(binding.science_input for binding in bridge.bind_spans(None))
            admission_digest = _span_admission_digest(inputs)
            if local.is_file():
                local_payload = _load_json(local)
                if (
                    isinstance(local_payload, Mapping)
                    and local_payload.get("wire_span_admission_digest")
                    == admission_digest
                ):
                    return "REUSE"
            key = _global_role_cache_key(
                identity,
                point_count=point_count,
                admission_digest=admission_digest,
                wire_configuration_digest=wire_role_index_configuration_digest(),
                vegetation_configuration_digest=(
                    PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST
                ),
            )
            if key and (_global_cache_root(self._root) / key / "COMPLETE.json").is_file():
                return "REUSE"
        except Exception:  # noqa: BLE001 - a UX hint must never block science
            pass
        return "COLD"

    @staticmethod
    def _span_name(project: Project, span: Span) -> str:
        lines = {line.line_id: line for line in project.lines}
        towers = {tower.tower_id: tower for tower in project.towers}
        line = lines.get(span.line_id)
        return (
            f"{towers[span.start_tower_id].display_name}-{towers[span.end_tower_id].display_name}"
            if line
            else span.span_id
        )

    def _ensure_editor(self) -> Any | None:
        if self._editor is not None:
            return self._editor
        result = self._current_result()
        if result is None or result.global_reconciled is None:
            return None
        project = self._stage_project()
        scheduler = BatchAnalysisScheduler(
            self._root, project, result.scan_id, risk_policy=self._risk_policy()
        )
        editor = scheduler.open_editor(result)
        edits = _load_json(editor.artifact_root / "tree_edits.json")
        if edits:
            editor.session.operations = load_edits_payload(edits.get("operations", []))
        if self._workflow.get('acceptance_mode') != 'FIXTURE' and project.towers:
            from core.stage_b.infrastructure import protected_infrastructure
            infrastructure = protected_infrastructure(self._root, result, project)
            editor.session.protect_source_rows(infrastructure['original_index'])
            editor.infrastructure = infrastructure
        self._editor = editor
        return self._editor

    @staticmethod
    def _editor_revision_key(editor: Any) -> tuple[Any, ...]:
        """Cheap process-local key for cached UI projections.

        Operation dictionaries are append-only and replaced on undo/reload, so
        their identities distinguish edits without hashing compressed exact
        membership payloads.
        """

        return (id(editor), *(id(operation) for operation in editor.session.operations))

    def _editor_snapshot(self, editor: Any) -> dict[str, Any]:
        key = self._editor_revision_key(editor)
        if self._editor_snapshot_cache is None or self._editor_snapshot_cache[0] != key:
            self._editor_snapshot_cache = (key, editor.snapshot())
        return self._editor_snapshot_cache[1]

    def _editor_display_basis(
        self,
        editor: Any,
        budget: int,
        selected_span: str | None,
    ) -> dict[str, Any]:
        """Return one immutable display LOD shared by all editor revisions."""

        key = (id(editor), int(budget), selected_span)
        cached = getattr(self, "_editor_display_basis_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if callable(getattr(editor.session, "display_basis", None)):
            original_index, source_tree_ids, xyz = editor.session.display_basis(budget)
        else:  # Compatibility for lightweight adapters and older fixtures.
            original_index, source_tree_ids, xyz = editor.session.display_sample(budget)
        if selected_span is not None:
            allowed = np.asarray(editor.tree_ids_by_span[selected_span], dtype=np.int32)
            keep = np.isin(source_tree_ids, allowed)
            original_index = original_index[keep]
            source_tree_ids = source_tree_ids[keep]
            xyz = xyz[keep]
        basis = {
            "base_id": uuid4().hex,
            "original_index": np.asarray(original_index, dtype=np.int64),
            "source_tree_id": np.asarray(source_tree_ids, dtype=np.int32),
            "xyz": np.asarray(xyz, dtype=np.float64),
            "budget": int(budget),
            "span_id": selected_span,
        }
        self._editor_display_basis_cache = (key, basis)
        return basis

    @staticmethod
    def _project_editor_display(
        editor: Any, basis: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        if callable(getattr(editor.session, "project_display_basis", None)):
            return editor.session.project_display_basis(
                basis["original_index"], basis["source_tree_id"]
            )
        tree_ids = np.asarray(basis["source_tree_id"], dtype=np.int32)
        return tree_ids, np.ones(len(tree_ids), dtype=bool)

    @staticmethod
    def _display_labels(
        trees: list[dict[str, Any]],
        tree_ids: np.ndarray,
        xyz: np.ndarray,
        visible: np.ndarray,
    ) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []
        for tree in trees:
            tree_id = int(tree["tree_id"])
            rows = visible & (tree_ids == tree_id)
            if np.any(rows):
                point = np.mean(xyz[rows], axis=0).tolist()
            else:
                apex = tree.get("apex_xy_local_m")
                if not isinstance(apex, (list, tuple)) or len(apex) < 3:
                    continue
                point = [float(value) for value in apex[:3]]
            labels.append({"tree_id": tree_id, "text": str(tree_id), "xyz": point})
        return labels

    def _editor_display_delta(
        self,
        editor: Any,
        snapshot: dict[str, Any],
        *,
        from_revision: int,
    ) -> dict[str, Any]:
        cached = getattr(self, "_editor_display_basis_cache", None)
        if cached is None or cached[0][0] != id(editor):
            return {
                "schema": "treeline-editor-display-delta-v1",
                "available": False,
                "from_revision": int(from_revision),
                "revision": int(editor.session.revision),
            }
        basis = cached[1]
        tree_ids, visible = self._project_editor_display(editor, basis)
        trees = [dict(item) for item in snapshot.get("trees", [])]
        identities: dict[int, dict[str, Any]] = {}
        try:
            from desktop.tree_display_numbering_extensions import _current_display_identities

            identities = _current_display_identities(self, editor)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            pass
        for tree in trees:
            identity = identities.get(int(tree["tree_id"]))
            if identity is not None:
                tree["display_tree_id"] = identity["display_tree_id"]
        labels = self._display_labels(trees, tree_ids, basis["xyz"], visible)
        for label in labels:
            identity = identities.get(int(label["tree_id"]))
            if identity is not None:
                label["text"] = identity["display_tree_id"]
        mapping = editor.session.display_source_mapping()
        # Deleted members project to id=-1. Requiring id>=0 here discards
        # every deletion and leaves its old GPU points visible after commit.
        removed_in_basis = basis["original_index"][~visible]
        result = getattr(editor, "result", None)
        clearance_current = False
        try:
            from desktop.review_extensions import _result_matches_editor_revision

            clearance_current = bool(_result_matches_editor_revision(editor, result))
        except (AttributeError, TypeError, ValueError):
            pass
        return {
            "schema": "treeline-editor-display-delta-v1",
            "available": True,
            "base_id": basis["base_id"],
            "from_revision": int(from_revision),
            "revision": int(editor.session.revision),
            "tree_id_by_source": {str(key): value for key, value in sorted(mapping.items())},
            # A source may feed multiple split children. A source-only map
            # cannot express that ownership: use exact original-index order.
            "tree_id_by_index": tree_ids.tolist(),
            "visible_by_index": visible.tolist(),
            "removed_original_index": removed_in_basis.tolist(),
            "trees": trees,
            "operations": list(snapshot.get("operations", [])),
            "labels": labels,
            "clearance_current": clearance_current,
        }

    def _assert_editor_mutable(self) -> None:
        editor_tasks(self).assert_available()
        batch_thread = getattr(self, "_thread", None)
        workflow_state = getattr(self, "_workflow", {})
        batch_status = (
            workflow_state.get("last_batch_status") if isinstance(workflow_state, Mapping) else None
        )
        if (
            batch_thread is not None
            and batch_thread.is_alive()
            and batch_status
            in {
                None,
                BatchSpanStatus.RUNNING.value,
            }
        ):
            raise ValueError("批量分析正在运行，请完成后再修改单木。")
        finalize_thread = getattr(self, "_finalize_thread", None)
        if finalize_thread is not None and finalize_thread.is_alive():
            raise ValueError("正在生成复核结果，请完成后再修改单木。")

    def editor_state(self) -> dict[str, Any]:
        if self._finalize_stage == "DONE":
            return {
                "step": "DONE",
                "step_label": "复核结果已生成",
                "current_message": self._finalize_message,
                "error": None,
            }
        if self._finalize_stage == "FAILED":
            return {
                "step": "FINALIZE_FAILED",
                "step_label": "复核结果生成失败",
                "current_message": self._finalize_message,
                "error": self._finalize_error,
                "error_details": self._finalize_detail,
            }
        if self._finalize_thread is not None and self._finalize_thread.is_alive():
            return {
                "step": "FINALIZING",
                "step_label": self._finalize_message,
                "current_message": self._finalize_message,
                "error": None,
            }
        editor = self._ensure_editor()
        if editor is None:
            return {
                "step": "WAITING",
                "step_label": "等待批量分析",
                "error": self.batch_projection().get("error"),
                "tree_snapshot": {"trees": [], "current_tree_count": 0, "selected_ids": []},
            }
        snapshot = self._editor_snapshot(editor)
        return {
            "step": "TREE_EDITING",
            "step_label": "STEP 2 · Tree Editor",
            "error": None,
            "tree_edit_count": len(editor.session.operations),
            "tree_snapshot": snapshot,
        }

    def editor_data(
        self,
        *,
        display_budget: int | None = None,
        span_id: str | None = None,
        persist_display_evidence: bool = True,
    ) -> dict[str, Any]:
        # ``persist_display_evidence`` is consumed by the review adapter.  It
        # remains part of the base signature so wrapped and unwrapped desktop
        # workflows share the same explicit read-only projection contract.
        del persist_display_evidence
        editor = self._ensure_editor()
        if editor is None:
            return _empty_display()
        budget = _DEFAULT_DISPLAY_POINTS if display_budget is None else int(display_budget)
        budget = max(1_500, min(_MAX_DISPLAY_POINTS, budget))
        selected_span = str(span_id or "").strip() or None
        if selected_span is not None and selected_span not in editor.tree_ids_by_span:
            raise ValueError("UNKNOWN_BATCH_SPAN")
        key = (*self._editor_revision_key(editor), budget, selected_span)
        if self._editor_display_cache is not None and self._editor_display_cache[0] == key:
            return self._editor_display_cache[1]
        basis = DesktopWorkflow._editor_display_basis(
            self, editor, budget, selected_span
        )
        tree_ids, visible = DesktopWorkflow._project_editor_display(editor, basis)
        display_xyz = basis["xyz"]
        source_tree_ids = basis["source_tree_id"]
        original_index = basis["original_index"]
        snapshot_trees = [dict(item) for item in self._editor_snapshot(editor).get("trees", [])]
        if selected_span is not None:
            source_ids = set(editor.tree_ids_by_span[selected_span])
            visible_ids = {
                int(item["tree_id"])
                for item in snapshot_trees
                if source_ids.intersection(int(value) for value in item.get("source_tree_ids", []))
            }
            snapshot_trees = [
                item for item in snapshot_trees if int(item["tree_id"]) in visible_ids
            ]
        colors = [
            f"hsl({(max(0, int(tree_id)) * 137) % 360},72%,58%)" for tree_id in tree_ids
        ]
        project = self._stage_project()
        tower_xyz: list[list[float]] = []
        labels: list[str] = []
        for tower in project.towers:
            tower_xyz.append(list(tower.position))
            labels.append(tower.display_name)
        wire_xyz: list[list[float]] = []
        selected_spans = (
            {selected_span}
            if selected_span is not None
            else set(self._workflow.get("selected_span_ids") or [])
        )
        from desktop.current_report import current_wire_curves

        fitted_wires = current_wire_curves(editor.result)
        wire_segments = [
            segment for segment in fitted_wires.get('segments', [])
            if not selected_spans or segment.get('span_id') in selected_spans
        ]
        # Never substitute a tower-to-tower schematic for measured wire geometry.
        # Keep curves separate so rendering cannot connect unrelated conductors.
        for segment in wire_segments:
            wire_xyz.extend(segment.get('xyz', []))
        display = _empty_display()
        display["trees"] = snapshot_trees
        # A freshly opened viewer must recover the persisted undo journal, not
        # just the point geometry. Mutation deltas alone are lost on reload.
        display["operations"] = list(self._editor_snapshot(editor).get("operations", []))
        display["tree_edit_count"] = len(editor.session.operations)
        labels_3d = DesktopWorkflow._display_labels(
            snapshot_trees, tree_ids, display_xyz, visible
        )
        display["trees3d"] = {
            "xyz": display_xyz.tolist(),
            "color": colors,
            "tree_id_by_index": tree_ids.tolist(),
            "source_tree_id_by_index": source_tree_ids.tolist(),
            "original_index_by_index": original_index.tolist(),
            "visible_by_index": visible.tolist(),
            "labels": labels_3d,
        }
        display["tower"] = {"xyz": tower_xyz, "labels": labels}
        display["wire"] = {"xyz": wire_xyz}
        display["wires3d"] = {
            "xyz": wire_xyz,
            "segments": wire_segments,
            "source": fitted_wires['source'],
            "diagnostics": fitted_wires.get('diagnostics', []),
        }
        infrastructure = getattr(editor, 'infrastructure', None)
        if infrastructure is not None:
            # Exact identities remain in the editor. The view receives only a
            # bounded, non-editable LOD plus the fixed measured pole axes.
            display['infrastructure'] = {k: v for k, v in infrastructure.items()
                                         if k != 'original_index'}
            if infrastructure['pole_segments']:
                display['tower'] = {
                    'xyz': [pole['xyz'][0] for pole in infrastructure['pole_segments']],
                    'labels': [pole['label'] for pole in infrastructure['pole_segments']],
                }
        display["review_span_id"] = selected_span
        display["editor_revision"] = int(
            getattr(editor.session, "revision", len(editor.session.operations))
        )
        display["display_base_id"] = basis["base_id"]
        self._editor_display_cache = (key, display)
        return display

    @editor_action("SELECT")
    def select_editor_trees(self, tree_ids: list[int]) -> dict[str, Any]:
        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        editor.session.select([int(value) for value in tree_ids])
        return editor.snapshot()

    def _commit_editor_change(self, kind: str, action: Any) -> dict[str, Any]:
        """Prepare on a detached journal; atomic journal replacement is the commit point."""
        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        working = copy.copy(editor)
        working.session = copy.copy(editor.session)
        working.session.operations = list(editor.session.operations)
        working.session.selected_ids = list(editor.session.selected_ids)
        working.session._previews = dict(editor.session._previews)
        working.session._source_rows_cache = dict(editor.session._source_rows_cache)
        working.session._removed_indices_cache = editor.session._removed_indices_cache
        timings = dict.fromkeys(
            [
                "selection_decode",
                "candidate_lookup",
                "geometric_inclusion",
                "membership_mutation",
                "journal_write",
                "viewer_artifact_rebuild",
                "persist",
            ],
            0.0,
        )
        from_revision = int(editor.session.revision)
        started = time.perf_counter()
        action(working)
        timings["membership_mutation"] = time.perf_counter() - started
        started = time.perf_counter()
        snapshot = working.snapshot()
        # The viewer owns a fixed display-only LOD basis.  A commit publishes a
        # revision delta after the durable journal write, so no XYZ/final-array
        # reconstruction is required at this boundary.
        timings["viewer_artifact_rebuild"] = time.perf_counter() - started
        started = time.perf_counter()
        working.persist_edits()
        timings["persist"] = time.perf_counter() - started
        timings["journal_write"] = getattr(
            working, "last_journal_write_seconds", timings["persist"]
        )
        # No fallible rebuild follows the durable commit point.
        editor.session = working.session
        self._editor_snapshot_cache = None
        self._editor_display_cache = None
        delta_started = time.perf_counter()
        try:
            snapshot["display_delta"] = self._editor_display_delta(
                editor, snapshot, from_revision=from_revision
            )
        except Exception:  # noqa: BLE001 - committed edit can recover via full display fetch
            snapshot["display_delta"] = {
                "schema": "treeline-editor-display-delta-v1",
                "available": False,
                "from_revision": from_revision,
                "revision": int(editor.session.revision),
            }
        timings["viewer_artifact_rebuild"] = time.perf_counter() - delta_started
        snapshot["editor_revision"] = int(editor.session.revision)
        snapshot["timings_seconds"] = timings
        try:
            self._log(
                "tree_editor_" + kind.lower(),
                tree_edit_count=len(editor.session.operations),
                timings_seconds=timings,
                state="COMMITTED",
            )
        except Exception:
            pass  # An auxiliary log failure cannot undo a durable journal commit.
        return snapshot

    @editor_action("DELETE")
    def delete_selected(self) -> dict[str, Any]:
        return self._commit_editor_change("DELETE", lambda editor: editor.session.delete_selected())

    @editor_action("MERGE")
    def merge_selected(self) -> dict[str, Any]:
        return self._commit_editor_change("MERGE", lambda editor: editor.session.merge_selected())

    @editor_action("SPLIT")
    def split_selected(self, seed_xyz: list[list[float]] | None = None) -> dict[str, Any]:
        return self._commit_editor_change("SPLIT", lambda editor: editor.session.split_selected(seed_xyz))

    @editor_action("UNDO")
    def undo_editor(self) -> dict[str, Any]:
        return self._commit_editor_change("UNDO", lambda editor: editor.session.undo())

    def _trim_binding(self) -> str:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        return f"{self._root}|{editor.result.analysis_run_id}"

    @editor_action("COMPONENT_PREVIEW")
    def preview_trim_component(self, tree_id: int, seed_xyz: list[float], picks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        if picks is not None:
            preview = editor.session.preview_trim_components(picks, binding=self._trim_binding())
        else:
            preview = editor.session.preview_trim_component(
                int(tree_id), seed_xyz, binding=self._trim_binding()
            )
        preview.update(self.trim_preview_display(preview["preview_id"]))
        preview["preview_xyz"] = preview.pop("xyz")
        return preview

    @editor_action("LASSO_PREVIEW")
    def preview_trim_lasso(
        self, tree_id: int, view_projection: list[list[float]], polygon_ndc: list[list[float]]
    ) -> dict[str, Any]:
        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        preview = editor.session.preview_trim_lasso(
            int(tree_id), view_projection, polygon_ndc, binding=self._trim_binding()
        )
        preview.update(self.trim_preview_display(preview["preview_id"]))
        preview["preview_xyz"] = preview.pop("xyz")
        return preview

    @editor_action("TRIM_COMMIT")
    def commit_trim(self, preview_id: str) -> dict[str, Any]:
        binding = self._trim_binding()
        return self._commit_editor_change(
            "TRIM",
            lambda editor: editor.session.commit_trim(
                preview_id,
                binding=binding,
                minimum_remaining_points=int(editor.config.tree_min_final_members),
            ),
        )

    @editor_action("CANCEL")
    def cancel_trim(self, preview_id: str) -> dict[str, Any]:
        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        editor.session.cancel_trim(preview_id)
        return editor.snapshot()

    def trim_preview_display(self, preview_id: str) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        xyz = editor.session.preview_xyz(preview_id)
        rows = np.linspace(
            0,
            len(xyz) - 1,
            min(len(xyz), _MAX_EDIT_PREVIEW_POINTS),
            dtype=np.int64,
        )
        return {"xyz": xyz[rows].tolist(), "full_point_count": int(len(xyz))}

    def _set_finalize_stage(self, stage: str, message: str) -> None:
        with self._lock:
            self._finalize_stage = stage
            self._finalize_message = message

    def _set_persisted_finalize_state(
        self,
        status: str,
        run_id: str | None,
        *,
        error: str | None = None,
    ) -> None:
        """Persist finalize lifecycle so a restart cannot present stale finals."""

        state: dict[str, Any] = {"status": str(status).upper(), "run_id": run_id}
        if error:
            state["error"] = str(error)
        with self._lock:
            self._workflow["finalize_state"] = state
            self._save_workflow()

    def _finalize_run_id(self, editor: Any | None = None) -> str | None:
        result = self._result
        if result is None and editor is not None:
            result = getattr(editor, "result", None)
        if result is None:
            result = self._current_result()
        value = getattr(result, "analysis_run_id", None) if result is not None else None
        value = value or self._workflow.get("last_batch_id")
        return str(value) if value else None

    def _require_finalize_disk_space(self, editor: Any) -> None:
        """Fail early only when current temp/final artifacts plainly cannot fit."""

        existing = [
            editor.artifact_root / "tree_segmentation_original.npz",
            editor.artifact_root / "tree_segmentation_final.npz",
        ]
        largest = max((path.stat().st_size for path in existing if path.is_file()), default=0)
        required = max(64 * 1024**2, largest * 2 + 64 * 1024**2)
        available = shutil.disk_usage(self._root).free
        if available < required:
            raise OSError(28, "No space left on device")

    def _persist_integrated_current_report_cache(self) -> None:
        """Save the bounded post-review snapshot used by warning and twin pages."""

        try:
            from desktop.current_report import build_current_report
            from desktop.integrated_cache import save_current_report_cache

            report = build_current_report(self)
            saved = save_current_report_cache(self, report)
            if saved is not None:
                self._log("integrated_current_report_cached", path=str(saved))
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            # Final survey publication remains valid even if this optional,
            # regenerable UI acceleration snapshot cannot be written.
            self._log(
                "integrated_current_report_cache_failure",
                error=f"{type(exc).__name__}: {exc}",
            )

    def finalize(self) -> dict[str, Any]:
        """Synchronous compatibility path used by non-UI callers and tests."""

        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        run_id = self._finalize_run_id(editor)
        self._set_persisted_finalize_state("RUNNING", run_id)
        try:
            self._require_finalize_disk_space(editor)
            editor.mark_reviewed()
            self._result = editor.finalize()
            self._log('dbh_recovery_finished', **getattr(editor, 'dbh_recovery_status', {}))
            self._final_tree_records = {
                int(record.global_tree_id): record.to_dict()
                for record in self._result.global_reconciled.global_tree_records
            }
            self._workflow["last_batch_status"] = self._result.status
            self._save_workflow()
            self._set_persisted_finalize_state("DONE", run_id)
            self._persist_integrated_current_report_cache()
            self._log("tree_editor_finalized", analysis_run_id=self._result.analysis_run_id)
            return self.batch_projection()
        except Exception as exc:
            self._set_persisted_finalize_state(
                "FAILED", run_id, error=f"{type(exc).__name__}: {exc}"
            )
            raise

    def _run_finalize(self) -> None:
        try:
            editor = self._ensure_editor()
            if editor is None:
                raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
            self._require_finalize_disk_space(editor)
            editor.mark_reviewed()

            stage_messages = {
                "APPLYING_EDITS": "正在应用人工修改",
                "RECALCULATING_CLEARANCE": "正在重新计算净空",
                "VALIDATING_CLEARANCE": "正在验证已保存的净空结果",
            }

            def editor_progress(stage: str) -> None:
                self._set_finalize_stage(stage, stage_messages[stage])

            self._result = editor.finalize(progress=editor_progress)
            self._log('dbh_recovery_finished', **getattr(editor, 'dbh_recovery_status', {}))
            self._final_tree_records = {
                int(record.global_tree_id): record.to_dict()
                for record in self._result.global_reconciled.global_tree_records
            }
            self._workflow["last_batch_status"] = self._result.status
            self._save_workflow()

            self._log(
                "tree_editor_finalized",
                analysis_run_id=self._result.analysis_run_id,
                warning_handoff="DIRECT_FINALIZED_DATA",
            )
            self._set_persisted_finalize_state("DONE", self._finalize_run_id(editor))
            self._set_finalize_stage("CACHING", "正在保存下次打开可直接使用的项目快照")
            self._persist_integrated_current_report_cache()
            self._set_finalize_stage("DONE", "复核定稿完成，可直接生成预警报告与工单")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            no_space = isinstance(exc, OSError) and getattr(exc, "errno", None) == 28
            message = (
                f"磁盘剩余空间不足，无法完成复核结果生成。请释放项目盘 {self._root.anchor} 空间后重试。"
                if no_space
                else "复核结果生成失败，请查看详情后重试。"
            )
            with self._lock:
                self._finalize_stage = "FAILED"
                self._finalize_message = message
                self._finalize_error = message
                self._finalize_detail = detail
            self._set_persisted_finalize_state("FAILED", self._finalize_run_id(), error=detail)
            self._log("tree_editor_finalize_failure", error=detail)
        finally:
            with self._lock:
                self._finalize_thread = None

    @editor_action("FINALIZE_START")
    def start_finalize(self) -> dict[str, Any]:
        self._assert_editor_mutable()
        with self._lock:
            if self._finalize_thread is not None and self._finalize_thread.is_alive():
                raise ValueError("复核结果正在生成，请稍候。")
            self._finalize_error = None
            self._finalize_detail = None
            self._finalize_stage = "APPLYING_EDITS"
            self._finalize_message = "正在应用人工修改"
            self._set_persisted_finalize_state("RUNNING", self._finalize_run_id())
            self._finalize_thread = threading.Thread(target=self._run_finalize, daemon=True)
            self._finalize_thread.start()
        return self.editor_state()

    def _prediction_export_cache_matches_final(self, path: Path, result: Any) -> bool:
        """Return whether a cached workbook matches the finalized publication."""

        if not self._final_tree_records:
            return False
        from core.stage_b.export import _topology_identity

        final_records = tuple(
            SimpleNamespace(**raw) for raw in self._final_tree_records.values()
        )
        reconciled = SimpleNamespace(global_tree_records=final_records)
        topology = _topology_identity(
            path,
            reconciled,
            project_id=getattr(result, "project_id", None),
        )
        if not topology:
            # Legacy projects/results without a resolvable topology retain the
            # historical cache behavior; export itself remains fail-closed.
            topology = {}
        workbook = None
        try:
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            metadata_sheet = workbook["Run_Metadata"]
            metadata_rows = metadata_sheet.iter_rows(values_only=True)
            metadata_headers = tuple(next(metadata_rows))
            key_index = metadata_headers.index("Key")
            value_index = metadata_headers.index("Value")
            metadata = {
                str(row[key_index]): row[value_index]
                for row in metadata_rows
                if len(row) > max(key_index, value_index)
                and row[key_index] is not None
            }
            if any(
                metadata.get(key) != expected
                for key, expected in {
                    "project_id": result.project_id,
                    "science_run_id": result.analysis_run_id,
                    "science_scope": "FINAL_GLOBAL_RECONCILED_POST_EDITOR",
                }.items()
            ):
                return False

            tree_sheet = workbook["Trees"]
            tree_rows = tree_sheet.iter_rows(values_only=True)
            headers = tuple(next(tree_rows))
            if not {
                "dbh_cm", "dbh_fit_status", "horizontal_clearance_m",
                "vertical_clearance_m", "clearance_component_source",
                "dbh_ground_z_absolute_m", "dbh_measurement_z_absolute_m",
                "dbh_coordinate_source",
            }.issubset(headers):
                return False
            indices = {
                key: headers.index(key)
                for key in (
                    "persistent_tree_id",
                    "project_id",
                    "science_run_id",
                    "analysis_tree_id",
                    "start_tower_id",
                    "end_tower_id",
                )
            }
            data_rows = [
                row for row in tree_rows if any(value is not None for value in row)
            ]
            expected_records = {
                str(raw.get("persistent_tree_id")): raw
                for raw in self._final_tree_records.values()
                if raw.get("persistent_tree_id") is not None
            }
            if len(expected_records) != len(self._final_tree_records):
                # A legacy result without durable IDs cannot support exact
                # cache validation; force the normal exporter instead.
                return False
            if len(data_rows) != len(expected_records):
                return False
            actual_records: dict[str, tuple[Any, ...]] = {}
            for row in data_rows:
                if len(row) <= max(indices.values()):
                    return False
                persistent_id = row[indices["persistent_tree_id"]]
                if persistent_id is None or str(persistent_id) in actual_records:
                    return False
                actual_records[str(persistent_id)] = row
            if set(actual_records) != set(expected_records):
                return False
            for persistent_id, raw in expected_records.items():
                row = actual_records[persistent_id]
                if row[indices["project_id"]] != result.project_id:
                    return False
                if row[indices["science_run_id"]] != result.analysis_run_id:
                    return False
                expected_tree_id = f"analysis-tree-{int(raw['global_tree_id']):06d}"
                if row[indices["analysis_tree_id"]] != expected_tree_id:
                    return False
                if topology:
                    identity = topology.get(int(raw["global_tree_id"]))
                    if identity is None or any(
                        row[indices[key]] != identity[key]
                        for key in ("start_tower_id", "end_tower_id")
                    ):
                        return False
            return True
        except (
            BadZipFile,
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
            OSError,
            ImportError,
        ):
            return False
        finally:
            if workbook is not None:
                workbook.close()

    def export_prediction(self) -> Path:
        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("PREDICTION_EXPORT_REQUIRES_FINALIZED_BATCH")
        if self._result is None:
            self._result = getattr(editor, "result", None)
        result = self._result
        if result is None:
            raise ValueError("PREDICTION_EXPORT_REQUIRES_FINALIZED_BATCH")
        statuses = tuple(result.span_statuses.values())
        if not statuses or any(status is not BatchSpanStatus.FINALIZED for status in statuses):
            raise ValueError("PREDICTION_EXPORT_REQUIRES_FINALIZED_BATCH")
        from desktop.review_extensions import _result_matches_editor_revision

        if not _result_matches_editor_revision(editor, result):
            raise ValueError("人工修改已更新，请先完成复核定稿。")
        path = self._root / "exports" / "Prediction_Input.xlsx"
        batch_path = editor.artifact_root / "batch.json"
        edits_path = editor.artifact_root / "tree_edits.json"
        dbh_path = editor.artifact_root / "dbh_final.json"
        source_times = [
            item.stat().st_mtime_ns
            for item in (batch_path, edits_path, dbh_path) if item.is_file()
        ]
        if (
            batch_path.is_file()
            and edits_path.is_file()
            and edits_path.stat().st_mtime_ns > batch_path.stat().st_mtime_ns
        ):
            raise ValueError("人工修改已更新，请先完成复核定稿。")
        if (
            path.is_file()
            and source_times
            and path.stat().st_mtime_ns >= max(source_times)
            and self._prediction_export_cache_matches_final(path, result)
        ):
            return path
        from core.stage_b.export import export_batch_prediction

        self._require_finalize_disk_space(editor)
        export_batch_prediction(result, path, final_reconciled=editor.final_reconciled())
        self._log("prediction_export", path=str(path))
        return path

    def current_report(self) -> dict[str, Any]:
        """Return the read-only modern current-result projection."""

        from desktop.current_report import build_current_report
        from desktop.integrated_cache import (
            load_current_report_cache,
            save_current_report_cache,
        )

        cached = load_current_report_cache(self)
        if cached is not None:
            return cached
        report = build_current_report(self)
        if report.get("status") == "FINALIZED":
            try:
                save_current_report_cache(self, report)
            except OSError as exc:
                self._log(
                    "integrated_current_report_cache_failure",
                    error=f"{type(exc).__name__}: {exc}",
                )
        return report

    def future_preview(
        self,
        horizon_years: int = 1,
        *,
        horizon_value: int | None = None,
        horizon_unit: str = "years",
    ) -> dict[str, Any]:
        """Build a detached experimental view; never publish current science."""

        from desktop.future_preview import build_future_preview

        return build_future_preview(
            self,
            horizon_years=horizon_years,
            horizon_value=horizon_value,
            horizon_unit=horizon_unit,
        )

    def final_projection(self, *, persist_display_evidence: bool = True) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            return {"trees": [], "display": _empty_display(), "metadata": {}}
        if self._result is None:
            self._result = editor.result
        snapshot = self._editor_snapshot(editor)
        reconciled = self._result.global_reconciled
        records = {
            int(record.global_tree_id): record
            for record in (reconciled.global_tree_records if reconciled is not None else ())
        }
        clearance_by_tree = {
            int(record.global_tree_id): record
            for record in (
                self._result.final_clearance.records if self._result.final_clearance else ()
            )
        }
        rows = []
        for tree in snapshot.get("trees", []):
            tree_id = int(tree.get("tree_id"))
            record = records.get(tree_id)
            persisted_record = self._final_tree_records.get(tree_id, {})
            clearance = clearance_by_tree.get(tree_id)
            minimum = None if clearance is None else clearance.minimum_clearance_m
            current_risk = None
            record_review = bool(persisted_record.get("review", record.review if record else False))
            quality_state = "REVIEW_REQUIRED" if record_review else "VALID"
            if clearance is not None:
                from core.stage_b.contact import contact_evidence_from_clearance

                evidence = contact_evidence_from_clearance(
                    minimum,
                    clearance_status=clearance.status,
                )
                assessment = self._risk_policy().classify(
                    clearance_m=minimum,
                    contact=(
                        False if evidence.state.value == "CONTACT_NOT_CONFIRMED" else None
                    ),
                    review_required=record_review or evidence.review_required,
                )
                current_risk = assessment.risk.value
                if evidence.review_required:
                    quality_state = "REVIEW_REQUIRED"
            rows.append(
                {
                    "persistent_tree_id": (
                        persisted_record.get("persistent_tree_id")
                        or (record.persistent_tree_id if record is not None else f"tree-{tree_id}")
                    ),
                    "height_m": (
                        persisted_record.get("apex_height_m")
                        if persisted_record.get("apex_height_m") is not None
                        else (
                            record.apex_height_m
                            if record is not None
                            else tree.get("apex_height_m")
                        )
                    ),
                    "crown_width_m": None,
                    "minimum_clearance_m": minimum,
                    **clearance_components(clearance),
                    "danger_point_count": (
                        None if clearance is None else clearance.danger_point_count
                    ),
                    "current_risk": current_risk,
                    "quality_state": quality_state,
                    "human_edited": bool(
                        str(
                            persisted_record.get(
                                "identity_state", record.identity_state if record else ""
                            )
                        ).startswith("HUMAN_")
                    ),
                }
            )
        return {
            "trees": rows,
            "display": (
                self.editor_data(persist_display_evidence=False)
                if not persist_display_evidence
                else self.editor_data()
            ),
            "metadata": {
                "tree_backend": "CROWN_STEM_HYBRID",
                "wire_exclusion_policy": "EXACT_ORIGINAL_INDEX",
                "risk_policy": self._risk_policy().to_dict(),
            },
        }


def ensure_desktop_manifest(root: Path, project_name: str | None = None) -> None:
    """Backfill workflow metadata for Stage-A projects created before Stage C."""
    path = root / WORKFLOW_MANIFEST
    if path.is_file():
        return
    _atomic_json(
        path,
        {
            "schema_version": WORKFLOW_SCHEMA,
            "project_name": project_name,
            "risk_policy": RiskPolicy().to_dict(),
            "selected_span_ids": [],
            "acceptance_mode": None,
            "last_batch_id": None,
            "last_batch_status": None,
            "last_error": None,
            "finalize_state": {},
            "updated_at": _now(),
        },
    )


__all__ = ["DesktopWorkflow", "WORKFLOW_MANIFEST", "WORKFLOW_SCHEMA", "ensure_desktop_manifest"]
