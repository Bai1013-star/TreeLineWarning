"""Stage-C desktop workflow orchestration.

This module is deliberately a product adapter, not a second science stack.  It
keeps the Stage-A sidecar authoritative, invokes the Stage-B bridge/scheduler,
and exposes only bounded review samples to the desktop viewer.  The fixture
runner exists solely for offline UI acceptance and is always labelled as such.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
from algorithms.folder_pipeline.tree_editor import load_edits_payload
from algorithms.spans.project_adapter import SourceSharedPreparation, SpanScienceResult
from core.stage_a import (
    CapabilityState,
    CoordinateDeclaration,
    Line,
    LineRole,
    Project,
    Route,
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

from desktop.project_management import (
    describe_project,
)

WORKFLOW_SCHEMA = "treeline-desktop-workflow-v2"
WORKFLOW_MANIFEST = "desktop-workflow.json"
_MAX_DISPLAY_POINTS = 6_000


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
        "trees3d": {"xyz": [], "color": [], "tree_id_by_index": []},
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
            "updated_at": manifest.get("updated_at", _now()),
        }
        self._log("project_open", project_root=str(self._root))

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
            project = self._stage_project()
            scan = self._scan(project)
            if scan is None:
                return None
            span_results = {
                span_id: SpanScienceResult(
                    tree_count=int(raw.get("tree_count", 0)),
                    minimum_clearance_m=raw.get("minimum_clearance_m"),
                    risk_threshold_m=float(raw.get("risk_threshold_m", 0.5)),
                    risk_counts=dict(raw.get("risk_counts") or {}),
                    diagnostics=dict(raw.get("diagnostics") or {}),
                )
                for span_id, raw in (manifest.get("span_results") or {}).items()
            }
            statuses = {
                span_id: BatchSpanStatus(value)
                for span_id, value in (manifest.get("span_statuses") or {}).items()
            }
            telemetry_raw = manifest.get("telemetry") or {}
            from core.stage_b.batch import BatchTelemetry

            telemetry = BatchTelemetry(
                source_point_count=int(telemetry_raw.get("SOURCE_POINT_COUNT", 0) or 0),
                span_count=int(telemetry_raw.get("SPAN_COUNT", len(statuses)) or 0),
                source_prep_count=int(telemetry_raw.get("SOURCE_PREP_COUNT", 0) or 0),
                source_full_pass_count=int(telemetry_raw.get("SOURCE_FULL_PASS_COUNT", 0) or 0),
                ground_prep_count=int(telemetry_raw.get("GROUND_PREP_COUNT", 0) or 0),
                span_local_read_count=int(telemetry_raw.get("SPAN_LOCAL_READ_COUNT", 0) or 0),
                cache_hit_count=int(telemetry_raw.get("CACHE_HIT_COUNT", 0) or 0),
                cache_miss_count=int(telemetry_raw.get("CACHE_MISS_COUNT", 0) or 0),
                total_time_seconds=float(telemetry_raw.get("TOTAL_TIME", 0.0) or 0.0),
                per_span_time_seconds=dict(telemetry_raw.get("PER_SPAN_TIME") or {}),
            )
            scheduler = BatchAnalysisScheduler(
                self._root, project, scan.scan_id, risk_policy=self._risk_policy()
            )
            global_reconciled = scheduler._load_persisted_global(str(manifest["analysis_run_id"]))
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
        result = self._current_result()
        tree_review_ready = bool(result and result.global_reconciled is not None)
        statuses = [str(item.get("status")) for item in batch.get("spans", [])]
        finalized = bool(statuses) and all(
            status == BatchSpanStatus.FINALIZED.value for status in statuses
        )
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
        threshold = float(threshold_m)
        policy = RiskPolicy(warning_clearance_threshold_m=threshold)
        self._workflow["risk_policy"] = policy.to_dict()
        self._save_workflow()
        self._log("risk_policy_updated", threshold_m=threshold)
        return policy.to_dict()

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
                height = float(raw.get("height_m", 12.0))
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
        updated = replace(
            project,
            lines=(*retained_lines, line),
            routes=(*retained_routes, route),
            towers=(*retained_towers, *towers),
            spans=(*retained_spans, *spans),
        )
        save_project(updated, self._root)
        self._log(
            "manual_tower_sequence_confirmed",
            line_id=line_id,
            tower_count=len(towers),
            span_count=len(spans),
        )
        return self.project_projection()

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
                kwargs: dict[str, Any] = {"risk_policy": self._risk_policy()}
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

    def start_batch(self, span_ids: list[str], *, mode: str = "REAL") -> dict[str, Any]:
        mode = str(mode).upper()
        if mode not in {"REAL", "FIXTURE"}:
            raise ValueError("batch mode must be REAL or FIXTURE")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("BATCH_ALREADY_RUNNING")
            project = self._stage_project()
            scan = self._scan(project)
            if scan is None:
                raise ValueError("POINT_CLOUD_NOT_IMPORTED")
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
            self._workflow.update(
                {
                    "selected_span_ids": list(selected),
                    "last_batch_id": run_id,
                    "last_batch_status": "RUNNING",
                    "last_error": None,
                    "acceptance_mode": mode,
                }
            )
            self._batch_error = None
            self._save_workflow()
            self._thread = threading.Thread(
                target=self._run_scheduler, args=(selected, mode, run_id), daemon=True
            )
            self._thread.start()
            return self.batch_projection()

    def retry_failed(self) -> dict[str, Any]:
        manifest = self._latest_batch_manifest()
        if not manifest:
            raise ValueError("NO_BATCH_TO_RETRY")
        selected = [
            span_id
            for span_id, status in (manifest.get("span_statuses") or {}).items()
            if status == BatchSpanStatus.FAILED.value
        ]
        if not selected:
            raise ValueError("NO_FAILED_SPAN_TO_RETRY")
        mode = str(self._workflow.get("acceptance_mode") or "REAL").upper()
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise ValueError("BATCH_ALREADY_RUNNING")
            run_id = str(manifest["analysis_run_id"])
            self._workflow["last_batch_status"] = "RUNNING"
            self._save_workflow()
            self._thread = threading.Thread(
                target=self._run_scheduler, args=(tuple(selected), mode, run_id), daemon=True
            )
            self._thread.start()
        return self.batch_projection()

    def batch_projection(self) -> dict[str, Any]:
        project = self._stage_project()
        manifest = self._latest_batch_manifest()
        statuses = dict((manifest or {}).get("span_statuses") or {})
        selected = set(self._workflow.get("selected_span_ids") or [])
        spans = []
        for span in project.spans:
            status = statuses.get(span.span_id, BatchSpanStatus.WAITING.value)
            result = ((manifest or {}).get("span_results") or {}).get(span.span_id) or {}
            spans.append(
                {
                    "span_id": span.span_id,
                    "display_name": self._span_name(project, span),
                    "line_id": span.line_id,
                    "start_tower_id": span.start_tower_id,
                    "end_tower_id": span.end_tower_id,
                    "selected": span.span_id in selected,
                    "status": status,
                    "stage": "地面 · 导线 · 植被 · 单木 · 跨档整理"
                    if status not in {"WAITING", "FAILED"}
                    else ("待运行" if status == "WAITING" else "处理失败"),
                    "retryable": status == BatchSpanStatus.FAILED.value,
                    "summary": result,
                }
            )
        return {
            "ok": True,
            "batch_id": (manifest or {}).get("analysis_run_id")
            or self._workflow.get("last_batch_id"),
            "status": (manifest or {}).get("status")
            or self._workflow.get("last_batch_status")
            or "NOT_STARTED",
            "acceptance_mode": self._workflow.get("acceptance_mode"),
            "source_preparation": "DONE" if manifest else "WAITING",
            "spans": spans,
            "error": self._batch_error or self._workflow.get("last_error"),
            "risk_policy": self._risk_policy().to_dict(),
            "duplicate_final_original_index": int(
                ((manifest or {}).get("global_reconciliation") or {}).get(
                    "duplicate_final_original_index", 0
                )
            ),
        }

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
        self._editor = scheduler.open_editor(result)
        edits = _load_json(self._editor.artifact_root / "tree_edits.json")
        if edits:
            for operation in load_edits_payload(edits.get("operations", [])):
                self._editor.session.select(operation["tree_ids"])
                if operation["type"] == "DELETE":
                    self._editor.session.delete_selected()
                    continue
                merged_id = self._editor.session.merge_selected()
                if merged_id != operation["merged_id"]:
                    raise ValueError("PERSISTED_TREE_EDIT_MERGE_ID_MISMATCH")
        return self._editor

    def editor_state(self) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            return {
                "step": "WAITING",
                "step_label": "等待批量分析",
                "error": self.batch_projection().get("error"),
                "tree_snapshot": {"trees": [], "current_tree_count": 0, "selected_ids": []},
            }
        snapshot = editor.snapshot()
        return {
            "step": "TREE_EDITING",
            "step_label": "STEP 2 · Tree Editor",
            "error": None,
            "tree_edit_count": len(editor.session.operations),
            "tree_snapshot": snapshot,
        }

    def editor_data(self) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            return _empty_display()
        final_index, final_tree, final_xyz, _summary = editor.session.build_final()
        if len(final_xyz) > _MAX_DISPLAY_POINTS:
            rows = np.linspace(0, len(final_xyz) - 1, _MAX_DISPLAY_POINTS, dtype=np.int64)
        else:
            rows = np.arange(len(final_xyz), dtype=np.int64)
        tree_ids = np.asarray(final_tree, dtype=np.int32)[rows]
        colors = [f"hsl({(int(tree_id) * 137) % 360},72%,58%)" for tree_id in tree_ids]
        project = self._stage_project()
        tower_xyz: list[list[float]] = []
        labels: list[str] = []
        for tower in project.towers:
            tower_xyz.append(list(tower.position))
            labels.append(tower.display_name)
        wire_xyz: list[list[float]] = []
        selected_spans = set(self._workflow.get("selected_span_ids") or [])
        for span in project.spans:
            if selected_spans and span.span_id not in selected_spans:
                continue
            by_id = {tower.tower_id: tower for tower in project.towers}
            start = by_id.get(span.start_tower_id)
            end = by_id.get(span.end_tower_id)
            if start and end:
                wire_xyz.extend(
                    [
                        list(start.position),
                        [
                            start.position[0],
                            start.position[1],
                            start.position[2] + float(start.attributes.get("height_m", 12.0)),
                        ],
                        [
                            end.position[0],
                            end.position[1],
                            end.position[2] + float(end.attributes.get("height_m", 12.0)),
                        ],
                        list(end.position),
                    ]
                )
        display = _empty_display()
        display["trees"] = [dict(item) for item in editor.snapshot().get("trees", [])]
        display["trees3d"] = {
            "xyz": final_xyz[rows].tolist(),
            "color": colors,
            "tree_id_by_index": tree_ids.tolist(),
        }
        display["tower"] = {"xyz": tower_xyz, "labels": labels}
        display["wire"] = {"xyz": wire_xyz}
        display["wires3d"] = {"xyz": wire_xyz}
        return display

    def select_editor_trees(self, tree_ids: list[int]) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        editor.session.select([int(value) for value in tree_ids])
        return editor.snapshot()

    def delete_selected(self) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        editor.session.delete_selected()
        editor.persist()
        self._log("tree_editor_delete", tree_edit_count=len(editor.session.operations))
        return editor.snapshot()

    def merge_selected(self) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        editor.session.merge_selected()
        editor.persist()
        self._log("tree_editor_merge", tree_edit_count=len(editor.session.operations))
        return editor.snapshot()

    def undo_editor(self) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        editor.session.undo()
        editor.persist()
        self._log("tree_editor_undo", tree_edit_count=len(editor.session.operations))
        return editor.snapshot()

    def finalize(self) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        editor.mark_reviewed()
        self._result = editor.finalize()
        self._workflow["last_batch_status"] = self._result.status
        self._save_workflow()
        self._log("tree_editor_finalized", analysis_run_id=self._result.analysis_run_id)
        return self.batch_projection()

    def export_prediction(self) -> Path:
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("PREDICTION_EXPORT_REQUIRES_FINALIZED_BATCH")
        if self._result is None:
            self._result = editor.result
        if any(
            status not in {BatchSpanStatus.USER_REVIEWED, BatchSpanStatus.FINALIZED}
            for status in self._result.span_statuses.values()
        ):
            editor.mark_reviewed()
            self._result = editor.finalize()
        else:
            editor.persist()
        from core.stage_b.export import export_batch_prediction

        path = self._root / "exports" / "Prediction_Input.xlsx"
        export_batch_prediction(self._result, path, final_reconciled=editor.final_reconciled())
        self._log("prediction_export", path=str(path))
        return path

    def final_projection(self) -> dict[str, Any]:
        editor = self._ensure_editor()
        if editor is None:
            return {"trees": [], "display": _empty_display(), "metadata": {}}
        snapshot = editor.snapshot()
        rows = []
        for tree in snapshot.get("trees", []):
            rows.append(
                {
                    "persistent_tree_id": tree.get("persistent_tree_id")
                    or f"fixture-tree-{tree.get('tree_id')}",
                    "height_m": tree.get("apex_height_m"),
                    "crown_width_m": None,
                    "minimum_clearance_m": 0.75,
                    "danger_point_count": 0,
                    "current_risk": "SAFE",
                    "quality_state": "REVIEW_REQUIRED" if tree.get("review") else "VALID",
                    "human_edited": bool(editor.session.operations),
                }
            )
        return {
            "trees": rows,
            "display": self.editor_data(),
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
            "updated_at": _now(),
        },
    )


__all__ = ["DesktopWorkflow", "WORKFLOW_MANIFEST", "WORKFLOW_SCHEMA", "ensure_desktop_manifest"]
