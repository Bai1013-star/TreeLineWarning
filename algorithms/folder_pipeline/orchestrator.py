"""Folder pipeline orchestrator: drives the authoritative science behind STEP 1/2/3.

This module does NOT re-implement Wire, Vegetation, Tree or Clearance.  It
reuses the current authoritative implementations exactly as the manual-span
route does:

    ManualSingleSpanRun.initialize()
        -> preprocess / ground / evidence / display sample
    run.refine_and_build_span(A, B)
        -> two defensible poles -> one SpanRecord
    run._stage_regions()
        -> scene-wide halo recovery -> wire_region + high-recall vegetation_region
    run._stage_wire()
        -> authoritative Wire V2 -> confirmed_member_source_index
    TREE INPUT CLEANING (this pipeline's explicit step)
        FINAL_TREE_INPUT = VEGETATION - CONFIRMED_WIRE - CONFIRMED_TOWER
        (exact full-resolution original_index subtraction, verified)
    run_span_tree_pipeline(...) on the cleaned region
        -> TREE_BACKEND = CROWN_STEM_HYBRID (legacy-crown generated, stem validated)
    reconcile_trees(...)
        -> deterministic global tree ids + exclusive membership (the editor input)

After human edits (DELETE / MERGE / UNDO):

    edited membership -> edited TreeReconciliationResult
    run_manual_clearance(...)   exact full-res tree-wire clearance (post-edits)
    compute_tree_metrics(...)   canonical HAG height / crown / DBH
    apply_risk_threshold(...)   danger classification
    persistent_tree_id          unique per final tree
    Prediction_Input.xlsx       Trees + Run_Metadata sheets

Invariants maintained here: original_index conservation, display points never
enter science, confirmed wire/tower members never remain in the final tree
input, MERGE produces no duplicate original_index, and every final tree gets a
unique persistent id.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.folder_pipeline.state import (
    STEP_DONE,
    STEP_FINALIZING,
    STEP_SELECTING_SPAN,
    STEP_TREE_EDITING,
    FolderWorkflowState,
)
from algorithms.folder_pipeline.tree_editor import (
    OriginalTrees,
    TreeEditSession,
    build_original_trees,
    load_edits_payload,
)
from algorithms.folder_pipeline.workspace import FolderWorkspace

Progress = Callable[[int, str, bool, float | None], None]

_FOLDER_RUN_SCHEMA = "folder-to-excel-run-v1"
_TREE_INPUT_CLEANING_SCHEMA = "folder-to-excel-tree-input-cleaning-v1"

# How risk threshold default flows through (matches the manual-span route).
DEFAULT_RISK_THRESHOLD_M = 3.5


class FolderPipelineError(RuntimeError):
    """Fail-closed Folder pipeline error (Chinese message for the user)."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
    )
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _load_selection_payload(workspace: FolderWorkspace) -> dict[str, Any] | None:
    path = workspace.span_selection_path
    if not path.is_file():
        return None
    payload = json.loads(path.read_text("utf-8"))
    return payload


def _find_prior_ground_cache(workspace: FolderWorkspace) -> Path | None:
    """Reuse a validated PDAL SMRF ground cache for the same source.

    Mirrors the manual-span route: when the Python PDAL binding is unavailable
    but a prior run already executed the frozen PDAL SMRF + HAG-NN ground for
    the SAME source (same SHA), that ground is reused verbatim.  This makes
    repeated runs of a real source fast and PDAL-independent.
    """

    import hashlib

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    source_sha = workspace.source_sha256
    for root in (
        Path("artifacts") / "one_click_runs",
        Path("artifacts") / "manual_single_span_runs",
    ):
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            inspection = candidate / "00_input" / "source_inspection.json"
            if not inspection.is_file():
                continue
            try:
                payload = json.loads(inspection.read_text("utf-8"))
            except Exception:  # noqa: BLE001 - best effort
                continue
            if (
                payload.get("input_sha256") == source_sha
                and (candidate / "02_ground" / "ground_hag_result.json").is_file()
            ):
                return candidate / "02_ground"
    return None


class FolderOrchestrator:
    """One Folder run: owns the shared run + workflow state + edit session."""

    def __init__(
        self,
        workspace: FolderWorkspace,
        *,
        state: FolderWorkflowState | None = None,
        threshold_m: float = DEFAULT_RISK_THRESHOLD_M,
        ground_reuse_dir: Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.state = state if state is not None else FolderWorkflowState()
        self.threshold_m = float(threshold_m)
        self.ground_reuse_dir = ground_reuse_dir
        self.run: Any = None  # ManualSingleSpanRun (set by initialize())
        self.edit_session: TreeEditSession | None = None
        self.original_trees: OriginalTrees | None = None
        self.final_reconciled: Any = None
        self.final_risk: Any = None
        self.final_rows: list[dict[str, Any]] = []
        self.run_metadata: dict[str, Any] = {}
        self._science_lazy_lock: Any = None

    # ------------------------------------------------------------------ setup
    def initialize(self) -> dict[str, Any]:
        """Create and initialize the authoritative manual-span run."""

        from algorithms.folder_pipeline.state import _make_progress_callback
        from algorithms.manual_span.config import (
            default_manual_span_config,
            manual_route_config,
        )
        from algorithms.manual_span.pipeline import ManualSingleSpanRun

        if self.run is not None:
            return {"RUN_STATUS": "INITIALIZED"}
        self.workspace.ensure_dirs()
        reuse = self.ground_reuse_dir or _find_prior_ground_cache(self.workspace)
        self.run = ManualSingleSpanRun(
            self.workspace.source,
            self.workspace.science_run_root,
            cfg=manual_route_config(),
            manual=default_manual_span_config(),
            progress=_make_progress_callback(self.state),
            ground_reuse_dir=reuse,
        )
        self.state.set_step(STEP_SELECTING_SPAN, "[1/3] 正在准备点云...")
        result = self.run.initialize()
        if reuse is not None:
            self.run_metadata["ground_cache_reused"] = str(reuse)
        return result

    def _require_run(self) -> Any:
        if self.run is None:
            self.initialize()
        return self.run

    # ------------------------------------------------------------- selection
    def has_selection(self) -> bool:
        return self.workspace.span_selection_path.is_file()

    def load_selection(self) -> dict[str, Any] | None:
        payload = _load_selection_payload(self.workspace)
        if payload is None:
            return None
        from algorithms.manual_span.span import (
            build_manual_span_record,
        )

        pole_a = _pole_from_payload(payload, "pole_a")
        pole_b = _pole_from_payload(payload, "pole_b")
        run = self._require_run()
        run.pole_a = pole_a
        run.pole_b = pole_b
        run.origin_xyz = tuple(float(v) for v in payload.get("origin_xyz", (0.0, 0.0, 0.0)))
        run.span = build_manual_span_record(
            pole_a,
            pole_b,
            run.cfg,
            run.manual,
            run.origin_xyz,
            span_id=0,
        )
        run.span_digest = _selection_digest(payload)
        run.selection_payload = payload
        self.state.update(selection=payload)
        self.state.set_step(STEP_SELECTING_SPAN)
        return payload

    def apply_selection(
        self,
        click_a: tuple[float, float, float],
        click_b: tuple[float, float, float],
    ) -> dict[str, Any]:
        """Refine two pole clicks, build the span, persist span_selection.json."""

        run = self._require_run()
        if run.span is not None:
            raise FolderPipelineError("档距已确认，请使用重新选择后再试")
        try:
            run.refine_and_build_span(click_a, click_b)
        except Exception as exc:  # PoleRefinementError and friends
            self.state.set_error("杆塔识别失败", str(exc))
            raise FolderPipelineError(f"杆塔识别失败: {exc}") from exc
        payload = _build_selection_payload(self.workspace, run)
        _write_json(self.workspace.span_selection_path, payload)
        self.state.update(selection=payload)
        self.state.set_step(STEP_SELECTING_SPAN)
        return payload

    def reset_selection(self) -> None:
        run = self._require_run()
        if self.workspace.span_selection_path.is_file():
            self.workspace.span_selection_path.unlink()
        for name in ("pole_a", "pole_b", "span", "selection_payload"):
            setattr(run, name, None)
        run.span_digest = None
        self.state.update(
            selection=None, clicks={"A": {}, "B": {}}, pole_preview={},
            selection_mode="IDLE",
        )
        self.state.set_step(STEP_SELECTING_SPAN)

    def refine_pole(self, pole: str, click_xyz: tuple[float, float, float]) -> dict[str, Any]:
        run = self._require_run()
        refined = run._refine_pole(pole, click_xyz)
        return refined.to_dict()

    def local_raw(self, cx: float, cy: float, radius: float) -> dict[str, Any]:
        run = self._require_run()
        raw = run.local_raw_for_xy(cx, cy, radius)
        return {
            "xyz": raw["xyz_absolute"],
            "rgb": raw["rgb"],
            "original_index": raw["original_index"],
        }

    def display_points(self) -> dict[str, Any]:
        run = self._require_run()
        return {
            "xyz": run.display_sample["xyz_absolute"],
            "rgb": run.display_sample["rgb"],
            "source_bounds": run.inspection["bounds_min_xyz"],
            "source_bounds_max": run.inspection["bounds_max_xyz"],
            "origin_xyz": run.origin_xyz,
        }

    # ----------------------------------------------------------------- science
    def run_science(self, progress: Progress | None = None) -> dict[str, Any]:
        """Wire + Vegetation + exact Wire/Tower exclusion + CROWN_STEM_HYBRID.

        Runs once (the outcome is checkpointed inside the authoritative stages).
        Persists the algorithm original tree segmentation so the editor never
        requires re-running science after a server restart.
        """

        run = self._require_run()
        if run.span is None:
            raise FolderPipelineError("尚未确认档距")
        started = time.perf_counter()
        report = progress or (lambda *_: None)

        report(2, "正在重构导线...", False, None)
        wire_region, veg_region, wire_candidate_ids, veg_retained_ids = run._stage_regions()
        regions_digest = _canonical_digest(
            {
                "wire_region_count": wire_region.count,
                "vegetation_region_count": veg_region.count,
                "wire_candidate_reps": int(len(wire_candidate_ids)),
                "vegetation_retained_reps": int(len(veg_retained_ids)),
            }
        )
        run.wire_outcome = run._stage_wire(regions_digest)
        report(2, "正在提取植被...", False, None)

        # --- tree input cleaning (exact original_index subtraction) ----------
        report(2, "正在剔除导线/杆塔...", False, None)
        excluded, tower_source_indices = self._build_tree_input_exclusion(
            veg_region, wire_region
        )
        from algorithms.spans.region import subset_raw_region_excluding_source_indices

        cleaned_dir = self.workspace.source_cache_dir / "tree_input"
        cleaned_region = subset_raw_region_excluding_source_indices(
            veg_region, excluded, cleaned_dir / "cleaned_vegetation"
        )
        audit = _tree_input_audit(veg_region, cleaned_region, excluded, tower_source_indices)
        self.run_metadata.update(audit)
        report(2, "正在执行 CROWN_STEM_HYBRID...", False, None)

        from algorithms.spans.tree_runner import run_span_tree_pipeline

        tree_out_dir = self.workspace.science_run_root / "06_tree" / "span_001"
        run.tree_outcome = run_span_tree_pipeline(
            run.span,
            cleaned_region,
            run.coarse_ground_xyz_absolute,
            run.cfg,
            output_dir=tree_out_dir,
            origin_xyz=run.origin_xyz,
        )
        from algorithms.spans.reconcile import reconcile_trees

        run.reconciled = reconcile_trees(
            [run.tree_outcome], run.cfg, origin_xyz=run.origin_xyz
        )
        self._persist_original_trees(run.reconciled)
        report(2, "人工审核单木就绪", True, time.perf_counter() - started)
        self.state.set_step(STEP_TREE_EDITING, "等待人工审核单木...")
        return {
            "wire_strand_count": run.wire_outcome.strand_count,
            "confirmed_wire_member_count": int(len(run.wire_outcome.confirmed_member_source_index)),
            "tree_count": len(run.reconciled.global_tree_records),
            **audit,
        }

    def run_science_if_needed(self, progress: Progress | None = None) -> dict[str, Any]:
        """Idempotent science: reuses a persisted original tree result if present."""

        if (
            self.workspace.original_tree_npz.is_file()
            and self.workspace.original_tree_manifest.is_file()
        ):
            self._load_persisted_original_trees()
            return {"RESUMED": True}
        return self.run_science(progress=progress)

    # ---------------------------------------------------- tree-input cleaning
    def _build_tree_input_exclusion(
        self, veg_region: Any, wire_region: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (excluded_indices, tower_source_indices) for exact subtraction.

        - Confirmed wire: ``run.wire_outcome.confirmed_member_source_index``
          (the ONLY authoritative wire membership; display points never enter).
        - Confirmed tower: the two manually-confirmed pole shaft cores (exact
          full-res original_index inside a tight shaft radius).  No persistent
          ``TowerLocalMembership`` artifact is fabricated; if one becomes
          available it is used instead.
        """

        run = self._require_run()
        parts: list[np.ndarray] = []
        confirmed_wire = np.asarray(
            run.wire_outcome.confirmed_member_source_index, dtype=np.int64
        )
        parts.append(confirmed_wire)
        wire_excluded_in_veg = np.intersect1d(
            veg_region.original_index(), confirmed_wire, assume_unique=False
        )

        tower_parts: list[np.ndarray] = []
        tower_limit = "AVAILABLE"
        for pole in (run.pole_a, run.pole_b):
            if pole is None:
                continue
            shaft = _pole_shaft_source_indices(run, pole)
            tower_parts.append(shaft)
        tower_source = (
            np.unique(np.concatenate(tower_parts)) if tower_parts else np.empty(0, dtype=np.int64)
        )
        tower_excluded_in_veg = np.intersect1d(
            veg_region.original_index(), tower_source, assume_unique=False
        )
        if len(tower_source) == 0:
            tower_limit = "UNAVAILABLE"
        self.run_metadata["tower_exclusion_policy"] = (
            "EXACT_ORIGINAL_INDEX_POLE_SHAFT_CORE" if tower_limit == "AVAILABLE"
            else "PARTIAL_UNAVAILABLE"
        )
        self.run_metadata["tower_membership_source"] = (
            "MANUAL_POLE_REFINEMENT_SHAFT_CORE"
            if tower_limit == "AVAILABLE"
            else "NONE_AVAILABLE"
        )
        excluded = np.unique(np.concatenate(parts))
        # Audit counts against the FINAL tree input (vegetation region).
        self.run_metadata["confirmed_wire_excluded_count"] = int(len(wire_excluded_in_veg))
        self.run_metadata["confirmed_tower_excluded_count"] = int(len(tower_excluded_in_veg))
        self.run_metadata["tower_limited"] = tower_limit
        return excluded, tower_source

    # ----------------------------------------------------------------- editor
    def build_editor(self) -> TreeEditSession:
        """Build (or resume) the edit session over the algorithm original trees."""

        if self.edit_session is not None:
            return self.edit_session
        if self.original_trees is None:
            self.run_science_if_needed()
        assert self.original_trees is not None
        self.edit_session = TreeEditSession(self.original_trees)
        edits_path = self.workspace.tree_edits_path
        if edits_path.is_file():
            payload = json.loads(edits_path.read_text("utf-8"))
            self.edit_session.operations = load_edits_payload(
                payload.get("operations", [])
            )
            self.edit_session.selected_ids = []
        self.state.update(tree_snapshot=self.edit_session.snapshot())
        self.state.set_step(STEP_TREE_EDITING, "等待人工审核单木...")
        return self.edit_session

    def editor_snapshot(self) -> dict[str, Any]:
        session = self.build_editor()
        return session.snapshot()

    def editor_select(self, tree_ids: list[int]) -> dict[str, Any]:
        session = self.build_editor()
        session.select(tree_ids)
        self._persist_edits(session)
        return session.snapshot()

    def editor_delete(self) -> dict[str, Any]:
        session = self.build_editor()
        session.delete_selected()
        self._persist_edits(session)
        self.state.update(tree_edit_count=len(session.operations))
        return session.snapshot()

    def editor_merge(self) -> dict[str, Any]:
        session = self.build_editor()
        session.merge_selected()
        self._persist_edits(session)
        self.state.update(tree_edit_count=len(session.operations))
        return session.snapshot()

    def editor_undo(self) -> dict[str, Any]:
        session = self.build_editor()
        session.undo()
        self._persist_edits(session)
        self.state.update(tree_edit_count=len(session.operations))
        return session.snapshot()

    def _persist_edits(self, session: TreeEditSession) -> None:
        payload = session.edits_payload()
        payload["source_sha256"] = self.workspace.source_sha256
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(self.workspace.tree_edits_path, payload)

    # ----------------------------------------------------------------- final
    def confirm_trees_and_finalize(self, progress: Progress | None = None) -> dict[str, Any]:
        """Apply human edits -> final geometry -> exact clearance -> danger -> Excel."""

        report = progress or (lambda *_: None)
        self.state.set_step(STEP_FINALIZING, "[3/3] 正在计算最终净空...")
        started = time.perf_counter()
        session = self.build_editor()
        run = self._require_run()

        # 1. final membership from human edits (original_index only, XYZ unchanged)
        final_index, final_tree, final_xyz, merged_summary = session.build_final()

        # 2. build the edited reconciliation result with deterministic global ids
        edited_reconciled = _build_edited_reconciliation(
            run, final_index, final_xyz, final_tree, merged_summary,
            self.original_trees if self.original_trees is not None else None,
            self.workspace,
        )
        self.final_reconciled = edited_reconciled

        # 3. exact final Tree-Wire clearance (AFTER human edits)
        from algorithms.manual_span.clearance_fast import run_manual_clearance

        clearance = run_manual_clearance(
            edited_reconciled, [run.wire_outcome], run.cfg, run.manual
        )

        # 4. canonical HAG-based geometry / crown / DBH
        from algorithms.manual_span.metrics import compute_tree_metrics

        metrics = compute_tree_metrics(
            edited_reconciled, run.coarse_ground_xyz_absolute, run.origin_xyz, run.manual
        )
        metrics_by_tree = {metric.global_tree_id: metric for metric in metrics}

        # 5. danger classification
        from algorithms.manual_span.risk import apply_risk_threshold

        risk = apply_risk_threshold(
            clearance, metrics_by_tree, edited_reconciled, run.origin_xyz, self.threshold_m
        )
        self.final_risk = risk

        # 6. persistent identity + edit provenance + row assembly
        self.final_rows = _assemble_final_rows(
            edited_reconciled, risk, metrics_by_tree,
            self.workspace, self.threshold_m, run.origin_xyz,
        )
        self.run_metadata.update(_final_run_metadata(self, edited_reconciled, risk))

        # 7. persist final tree membership + metadata
        self._persist_final_trees(final_index, final_xyz, edited_reconciled)
        _write_json(self.workspace.final_tree_manifest, {
            "schema": "folder-to-excel-tree-segmentation-final-v1",
            "final_tree_count": len(edited_reconciled.global_tree_records),
            "final_rows": self.final_rows,
            "run_metadata": self.run_metadata,
        })

        # 8. Excel export
        report(3, "正在生成 Excel...", False, None)
        from algorithms.folder_pipeline.export_excel import write_prediction_input_excel

        excel_path = write_prediction_input_excel(
            self.workspace.excel_path,
            self.final_rows,
            self.run_metadata,
        )
        self.run_metadata["excel_path"] = str(excel_path)

        summary = {
            "RUN_STATUS": "PASS",
            "OUTPUT_DIRECTORY": str(self.workspace.export_dir),
            "EXCEL_REPORT": str(excel_path),
            "FINAL_TREE_COUNT": len(edited_reconciled.global_tree_records),
            "DANGEROUS_TREE_COUNT": risk.dangerous_tree_count,
            "DANGER_POINT_COUNT": int(risk.danger_point_count),
            "GLOBAL_MINIMUM_CLEARANCE_M": risk.global_minimum_clearance_m,
            "TOTAL_RUNTIME": time.perf_counter() - started,
            "SPAN_LENGTH_M": run.span.length_m,
            "WIRE_STRAND_COUNT": run.wire_outcome.strand_count,
        }
        self.state.update(final_summary=summary)
        self.state.set_step(STEP_DONE, "完成。")
        report(3, "完成。", True, time.perf_counter() - started)
        return summary

    # ------------------------------------------------------------ persistence
    def _persist_original_trees(self, reconciled: Any) -> None:
        records = [
            {
                "tree_id": int(record.global_tree_id),
                "external_name": record.external_name,
                "point_count": int(record.full_resolution_point_count),
                "apex_xy_local_m": list(record.apex_xy_local_m),
                "apex_height_m": (
                    None if record.apex_height_m is None else float(record.apex_height_m)
                ),
                "longitudinal_s_m": (
                    None if record.longitudinal_s_m is None else float(record.longitudinal_s_m)
                ),
                "lateral_n_m": (
                    None if record.lateral_n_m is None else float(record.lateral_n_m)
                ),
            }
            for record in reconciled.global_tree_records
        ]
        _write_npz(
            self.workspace.original_tree_npz,
            original_index=np.asarray(reconciled.final_original_index, dtype=np.int64),
            tree_id=np.asarray(reconciled.final_tree_id, dtype=np.int32),
            xyz_local_m=np.asarray(reconciled.final_xyz_local_m, dtype=np.float64),
        )
        _write_json(
            self.workspace.original_tree_manifest,
            {
                "schema": "folder-to-excel-tree-segmentation-original-v1",
                "tree_backend": "crown_stem_hybrid",
                "algorithm_tree_count": len(reconciled.global_tree_records),
                "trees": records,
                "run_metadata": self.run_metadata,
            },
        )
        self.original_trees = build_original_trees(
            reconciled.global_tree_records,
            reconciled.final_original_index,
            reconciled.final_xyz_local_m,
            reconciled.final_tree_id,
        )

    def _load_persisted_original_trees(self) -> None:
        manifest = json.loads(self.workspace.original_tree_manifest.read_text("utf-8"))
        records = manifest.get("trees", [])
        self.run_metadata.update(manifest.get("run_metadata", {}))
        with np.load(self.workspace.original_tree_npz, allow_pickle=False) as stream:
            original_index = np.asarray(stream["original_index"], dtype=np.int64)
            tree_id = np.asarray(stream["tree_id"], dtype=np.int32)
            xyz_local_m = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        from algorithms.folder_pipeline.tree_editor import OriginalTree

        built = [
            OriginalTree(
                tree_id=int(item["tree_id"]),
                external_name=str(item["external_name"]),
                point_count=int(item["point_count"]),
                apex_xy_local_m=(
                    tuple(float(v) for v in item["apex_xy_local_m"])
                    if item.get("apex_xy_local_m")
                    else None
                ),
                apex_height_m=(
                    None if item.get("apex_height_m") is None
                    else float(item["apex_height_m"])
                ),
            )
            for item in records
        ]
        self.original_trees = OriginalTrees(
            records=tuple(built),
            original_index=original_index,
            tree_id=tree_id,
            xyz_local_m=xyz_local_m,
        )

    def _persist_final_trees(
        self,
        final_index: np.ndarray,
        final_xyz: np.ndarray,
        reconciled: Any,
    ) -> None:
        _write_npz(
            self.workspace.final_tree_npz,
            original_index=final_index,
            xyz_local_m=final_xyz,
            tree_id=np.asarray(
                [record.global_tree_id for record in reconciled.global_tree_records],
                dtype=np.int32,
            ),
        )


# ------------------------------------------------------------------- helpers
def _pole_from_payload(payload: dict[str, Any], key: str) -> Any:
    from algorithms.manual_span.span import ManualRefinedPole

    data = payload.get(key)
    if not data:
        raise FolderPipelineError(f"span_selection.json 缺少 {key}")
    return ManualRefinedPole(
        label=str(data.get("label", "?")),
        clicked_xyz_absolute_m=tuple(float(v) for v in data["clicked_xyz_absolute_m"]),
        base_xyz_local_m=tuple(float(v) for v in data["base_xyz_local_m"]),
        base_xyz_absolute_m=tuple(float(v) for v in data["base_xyz_absolute_m"]),
        top_xyz_local_m=tuple(float(v) for v in data["top_xyz_local_m"]),
        top_xyz_absolute_m=tuple(float(v) for v in data["top_xyz_absolute_m"]),
        height_m=float(data["height_m"]),
        xy_footprint_m=float(data["xy_footprint_m"]),
        median_verticality=float(data["median_verticality"]),
        median_linearity=float(data["median_linearity"]),
        representative_count=int(data["representative_count"]),
        ground_support_reps=int(data["ground_support_reps"]),
        defensible=bool(data["defensible"]),
        reason=data.get("reason"),
        refinement_diagnostics=dict(data.get("refinement_diagnostics") or {}),
    )


def _selection_digest(payload: dict[str, Any]) -> str:
    from algorithms.manual_span.span import selection_digest

    return selection_digest(payload)


def _build_selection_payload(workspace: FolderWorkspace, run: Any) -> dict[str, Any]:
    from algorithms.manual_span.config import MANUAL_SPAN_SELECTION_SCHEMA
    from algorithms.manual_span.span import manual_span_selection_payload

    payload = manual_span_selection_payload(
        run.pole_a,
        run.pole_b,
        run.span,
        workspace.source_sha256,
        workspace.source_id,
        run.origin_xyz,
        selection_version=MANUAL_SPAN_SELECTION_SCHEMA,
    )
    payload["project_id"] = f"folder_{workspace.source_sha256[:8]}"
    payload["group_id"] = "G001"
    payload["span_id"] = "S001"
    payload["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return payload


def _pole_shaft_source_indices(run: Any, pole: Any) -> np.ndarray:
    """Exact full-res original_index of the confirmed pole shaft core.

    The manual pole refinement already proved these are the pole (dense vertical
    shaft, tight footprint, ground support).  This returns the exact RAW source
    rows inside a TIGHT shaft cylinder (0.6 m radius around the refined base
    centre, from just below base to just above top) — never a large buffer.
    """

    try:
        base = np.asarray(pole.base_xyz_absolute_m, dtype=np.float64)
        top = np.asarray(pole.top_xyz_absolute_m, dtype=np.float64)
        local = run.local_raw_for_xy(float(base[0]), float(base[1]))
        xyz = np.asarray(local["xyz_absolute"], dtype=np.float64)
        idx = np.asarray(local["original_index"], dtype=np.int64)
        if len(xyz) == 0:
            return np.empty(0, dtype=np.int64)
        xy_dist = np.linalg.norm(xyz[:, :2] - base[:2][None, :], axis=1)
        z_lo = float(base[2]) - 0.5
        z_hi = float(top[2]) + 1.0
        mask = (xy_dist <= 0.6) & (xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi)
        return np.unique(idx[mask])
    except Exception:  # noqa: BLE001 - tower exclusion is best effort; never fatal
        return np.empty(0, dtype=np.int64)


def _tree_input_audit(
    veg_region: Any,
    cleaned_region: Any,
    excluded: np.ndarray,
    tower_source_indices: np.ndarray,
) -> dict[str, Any]:
    veg_idx = np.asarray(veg_region.original_index(), dtype=np.int64)
    cleaned_idx = np.asarray(cleaned_region.original_index(), dtype=np.int64)
    wire_left = np.intersect1d(
        cleaned_idx, excluded, assume_unique=False
    )
    tower_left = np.intersect1d(
        cleaned_idx, tower_source_indices, assume_unique=False
    )
    return {
        "vegetation_input_count": int(len(veg_idx)),
        "final_tree_input_count": int(len(cleaned_idx)),
        "wire_tree_input_intersection": int(len(wire_left)),
        "tower_tree_input_intersection": int(len(tower_left)),
    }


def _build_edited_reconciliation(
    run: Any,
    final_index: np.ndarray,
    final_xyz: np.ndarray,
    final_tree: np.ndarray,
    merged_summary: dict[int, dict[str, Any]],
    original_trees: OriginalTrees | None,
    workspace: FolderWorkspace,
) -> Any:
    from algorithms.spans.reconcile import (
        GlobalTreeRecord,
        TreeReconciliationResult,
        _apply_span_order,
        _span_coordinates,
    )

    span = run.span
    by_source_id: dict[int, Any] = (
        {record.tree_id: record for record in original_trees.records}
        if original_trees
        else {}
    )
    ids = np.unique(final_tree)
    id_to_global = {int(tree_id): index for index, tree_id in enumerate(ids)}
    final_global = np.asarray(
        [id_to_global[int(tree_id)] for tree_id in final_tree], dtype=np.int32
    )
    records: list[GlobalTreeRecord] = []
    review_ids: list[int] = []
    for new_id, tree_value in enumerate(ids):
        tree_id = int(tree_value)
        rows = final_tree == tree_id
        total = int(np.count_nonzero(rows))
        merge = merged_summary.get(tree_id)
        if merge is not None:
            apex_xy = _lowest_apex(merge["source_tree_ids"], by_source_id)
            height = None
            review = total < run.cfg.tree_min_final_members
        else:
            src = by_source_id.get(tree_id)
            apex_xy = src.apex_xy_local_m if src else None
            height = src.apex_height_m if src else None
            review = total < run.cfg.tree_min_final_members
        apex_xy = apex_xy if apex_xy is not None else (0.0, 0.0, 0.0)
        longitudinal_s_m, lateral_n_m = _span_coordinates(
            span, apex_xy, run.origin_xyz
        )
        if review:
            review_ids.append(new_id)
        records.append(
            GlobalTreeRecord(
                global_tree_id=new_id,
                line_id=span.line_id,
                span_id=span.span_id,
                span_name=span.span_name,
                external_name=_final_external_name(span, new_id),
                primary_span_tree_id=tree_id,
                apex_xy_local_m=apex_xy,
                apex_height_m=height,
                full_resolution_point_count=total,
                core_point_count=total,
                halo_point_count=0,
                core_fraction=1.0,
                review=review,
                merged_span_tree_keys=(
                    tuple((0, tid) for tid in merge["source_tree_ids"])
                    if merge is not None
                    else ((0, tree_id),)
                ),
                longitudinal_s_m=longitudinal_s_m,
                lateral_n_m=lateral_n_m,
            )
        )
    records = _apply_span_order(records)
    return TreeReconciliationResult(
        global_tree_records=tuple(records),
        final_original_index=np.asarray(final_index, dtype=np.int64),
        final_xyz_local_m=np.asarray(final_xyz, dtype=np.float64),
        final_tree_id=final_global,
        review_tree_ids=tuple(review_ids),
        span_tree_to_global={},
        diagnostics={
            "schema": _FOLDER_RUN_SCHEMA,
            "span_count": 1,
            "global_tree_count": len(records),
            "final_raw_member_count": int(len(final_index)),
            "merge_operation_count": len(merged_summary),
            "workspace": workspace.root.name,
        },
    )


def _lowest_apex(
    source_tree_ids: list[int], by_source_id: dict[int, Any]
) -> tuple[float, float, float] | None:
    for tree_id in sorted(source_tree_ids):
        record = by_source_id.get(tree_id)
        if record is not None and record.apex_xy_local_m is not None:
            return record.apex_xy_local_m
    return None


def _final_external_name(span: Any, new_id: int) -> str:
    return f"{span.line_name}-Span{span.span_id + 1:03d}-Tree{new_id + 1:03d}"


def _assemble_final_rows(
    reconciled: Any,
    risk: Any,
    metrics_by_tree: dict[int, Any],
    workspace: FolderWorkspace,
    threshold_m: float,
    origin_xyz: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """One row per final tree (persistent id assigned deterministically)."""

    from algorithms.folder_pipeline.identity import assign_persistent_tree_ids

    origin = np.asarray(origin_xyz, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for record in reconciled.global_tree_records:
        metric = metrics_by_tree.get(record.global_tree_id)
        risk_record = next(
            (item for item in risk.records if item.global_tree_id == record.global_tree_id),
            None,
        )
        centroid_local = metric.tree_centroid_xyz_local_m if metric else None
        rows.append(
            {
                "persistent_tree_id": None,  # filled below
                "tree_id": record.global_tree_id,
                "external_name": record.external_name,
                "height_m": metric.tree_height_m if metric else None,
                "height_source": "HAG_FINAL_TREE",
                "crown_width_m": metric.crown_mean_width_m if metric else None,
                "crown_width_major_m": metric.crown_major_width_m if metric else None,
                "crown_width_minor_m": metric.crown_minor_width_m if metric else None,
                "crown_area_m2": metric.crown_area_m2 if metric else None,
                "crown_source": "XY_PRINCIPAL_AXES_P95_P05",
                "diameter_cm": (
                    None
                    if (metric is None or metric.dbh.dbh_m is None)
                    else round(float(metric.dbh.dbh_m) * 100.0, 2)
                ),
                "diameter_source": (
                    None
                    if (metric is None or metric.dbh.dbh_quality == "UNAVAILABLE")
                    else metric.dbh.dbh_quality
                ),
                "tree_x_m": (
                    None
                    if centroid_local is None
                    else float(centroid_local[0]) + float(origin[0])
                ),
                "tree_y_m": (
                    None
                    if centroid_local is None
                    else float(centroid_local[1]) + float(origin[1])
                ),
                "longitudinal_s_m": record.longitudinal_s_m,
                "lateral_n_m": record.lateral_n_m,
                "minimum_clearance_m": (
                    risk_record.clearance_3d_m if risk_record else None
                ),
                "danger_point_count": (risk_record.danger_point_count if risk_record else 0),
                "current_risk": _risk_label(risk_record, threshold_m),
                "quality_state": "CONFIRMED_TREE",
                "human_edited": _human_edited(record),
                "edit_provenance": _edit_provenance(record),
            }
        )
    assign_persistent_tree_ids(rows, workspace)
    return rows


def _risk_label(risk_record: Any, threshold_m: float) -> str:
    if risk_record is None:
        return "UNKNOWN"
    if risk_record.dangerous:
        return "DANGEROUS"
    return "SAFE"


def _human_edited(record: Any) -> bool:
    return bool(
        getattr(record, "merged_span_tree_keys", None)
        and len(record.merged_span_tree_keys) > 1
    )


def _edit_provenance(record: Any) -> str:
    if _human_edited(record):
        return "MERGE"
    return "UNEDITED"


def _canonical_digest(payload: dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _final_run_metadata(
    orchestrator: FolderOrchestrator, reconciled: Any, risk: Any
) -> dict[str, Any]:
    run = orchestrator.run
    selection = orchestrator.state.selection or {}
    session = orchestrator.edit_session
    operations = session.operations if session is not None else []
    delete_count = sum(1 for op in operations if op.get("type") == "DELETE")
    merge_count = sum(1 for op in operations if op.get("type") == "MERGE")
    return {
        "tree_backend": "crown_stem_hybrid",
        "wire_exclusion_policy": "EXACT_ORIGINAL_INDEX",
        "algorithm_tree_count": (
            len(orchestrator.original_trees.records)
            if orchestrator.original_trees is not None
            else 0
        ),
        "deleted_tree_count": delete_count,
        "merge_operation_count": merge_count,
        "final_tree_count": len(reconciled.global_tree_records),
        "wire_count": run.wire_outcome.strand_count,
        "danger_tree_count": risk.dangerous_tree_count,
        "danger_point_count": int(risk.danger_point_count),
        "global_minimum_clearance_m": risk.global_minimum_clearance_m,
        "risk_threshold_m": orchestrator.threshold_m,
        "project_id": (selection or {}).get("project_id"),
        "group_id": (selection or {}).get("group_id"),
        "span_id": (selection or {}).get("span_id"),
        "start_tower_id": "T001",  # pole A, deterministic allocation
        "end_tower_id": "T002",  # pole B
        "start_tower_xyz": (
            (selection or {}).get("pole_a", {}).get("base_xyz_absolute_m")
        ),
        "end_tower_xyz": (
            (selection or {}).get("pole_b", {}).get("base_xyz_absolute_m")
        ),
        "science_run_id": time.strftime("folder_%Y%m%d_%H%M%S"),
    }


__all__ = [
    "_FOLDER_RUN_SCHEMA",
    "_TREE_INPUT_CLEANING_SCHEMA",
    "DEFAULT_RISK_THRESHOLD_M",
    "FolderPipelineError",
    "FolderOrchestrator",
]
