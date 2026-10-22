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
from collections.abc import Callable, Iterable
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


def _brief(exc: Exception) -> str:
    """Short, safe exception text for a user-facing error message."""

    text = str(exc)
    if len(text) > 300:
        return text[:300] + "..."
    return text


def _canonical_risk_threshold_m() -> float:
    """Return the existing authoritative project risk threshold (never a new default).

    The Folder pipeline owns NO hard-coded science threshold: the danger /
    clearance threshold is the canonical ``RISK_THRESHOLD_M`` from
    ``algorithms.spans.project_adapter`` (policy ``exact-tree-wire``).
    """

    from algorithms.spans.project_adapter import RISK_THRESHOLD_M

    return float(RISK_THRESHOLD_M)


def _risk_config_digest() -> str:
    """Return the canonical risk-configuration digest (authoritative, never new)."""

    from algorithms.spans.project_adapter import RISK_CONFIGURATION_DIGEST

    return str(RISK_CONFIGURATION_DIGEST)


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


def _find_prior_ground_cache(
    workspace: FolderWorkspace,
    roots: Iterable[Path] | None = None,
) -> Path | None:
    """Reuse a validated PDAL SMRF ground cache for the same source.

    Mirrors the manual-span route: when the Python PDAL binding is unavailable
    but a prior run already executed the frozen PDAL SMRF + HAG-NN ground for
    the SAME source (same SHA), that ground is reused verbatim.  Cache reuse is
    STRICTLY source-identity validated: only a cache whose ``input_sha256``
    matches the workspace source SHA-256 is ever accepted.  An unrelated cache
    is never used.
    """

    import hashlib

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    del _sha256  # source identity is validated via the persisted input_sha256
    source_sha = workspace.source_sha256
    if roots is None:
        roots = (
            Path("artifacts") / "one_click_runs",
            Path("artifacts") / "manual_single_span_runs",
        )
    for root in roots:
        if not Path(root).is_dir():
            continue
        for candidate in sorted(
            Path(root).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
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
    # The workspace's OWN science run_root is the authoritative cache from a
    # prior real run (source_cache/science/02_ground).  It is source-identity
    # validated against source_inspection.json and must never be invalidated.
    local_root = workspace.science_run_root
    local_inspection = local_root / "00_input" / "source_inspection.json"
    local_ground = local_root / "02_ground" / "ground_hag_result.json"
    if local_inspection.is_file() and local_ground.is_file():
        try:
            payload = json.loads(local_inspection.read_text("utf-8"))
        except Exception:  # noqa: BLE001 - best effort
            payload = {}
        if payload.get("input_sha256") == source_sha:
            return local_root / "02_ground"
    return None


class FolderOrchestrator:
    """One Folder run: owns the shared run + workflow state + edit session."""

    def __init__(
        self,
        workspace: FolderWorkspace,
        *,
        state: FolderWorkflowState | None = None,
        threshold_m: float | None = None,
        ground_reuse_dir: Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.state = state if state is not None else FolderWorkflowState()
        # Default threshold is the canonical project standard (never a new
        # Folder-pipeline default).  An explicit caller value is respected.
        self.threshold_m = (
            _canonical_risk_threshold_m()
            if threshold_m is None
            else float(threshold_m)
        )
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
    def initialize(self, *, load_evidence: bool = True) -> dict[str, Any]:
        """Create and initialize the authoritative manual-span run.

        ``load_evidence`` — passed to the run.  A completed Step-2 workspace
        restart skips the heavy coarse-representative evidence load (the editor
        does not need it); science stages load it on demand.
        """

        from algorithms.folder_pipeline.state import _make_progress_callback

        if self.run is not None:
            return {"RUN_STATUS": "INITIALIZED"}
        self._create_run(progress=_make_progress_callback(self.state))
        # Ground preflight: fail with a clear Chinese error BEFORE expensive
        # science when neither PDAL nor a matching ground cache is available.
        # The run already exists (cheap constructor) so STEP-1 display still
        # works even when the ground preflight blocks science.
        self._preflight_ground()
        self.state.set_step(STEP_SELECTING_SPAN, "[1/3] 正在准备点云...")
        reuse = self.ground_reuse_dir or _find_prior_ground_cache(self.workspace)
        result = self.run.initialize(
            load_evidence=load_evidence,
            source_sha256=self.workspace.source_sha256,
        )
        if reuse is not None:
            self.run_metadata["ground_cache_reused"] = str(reuse)
        return result

    def _create_run(self, progress: Any = None) -> Any:
        """Create the authoritative ManualSpan run (constructor only, no science).

        Cheap: only binds the source, run_root, config and ground-reuse cache.
        Neither preprocess nor ground nor display is run here.
        """

        from algorithms.manual_span.config import (
            default_manual_span_config,
            manual_route_config,
        )
        from algorithms.manual_span.pipeline import ManualSingleSpanRun

        if self.run is not None:
            return self.run
        self.workspace.ensure_dirs()
        reuse = self.ground_reuse_dir or _find_prior_ground_cache(self.workspace)
        self.run = ManualSingleSpanRun(
            self.workspace.source,
            self.workspace.science_run_root,
            cfg=manual_route_config(),
            manual=default_manual_span_config(),
            progress=progress,
            ground_reuse_dir=reuse,
        )
        return self.run

    def _ensure_display_run(self) -> Any:
        """Return a run hydrated with the cheap source facts needed by STEP 1.

        Runs the FULL science initialize only when already available; otherwise
        populates the lightweight header facts (source identity + bounds) so the
        display and pole-selection UI work even when ground science is blocked.
        """

        run = self._create_run()
        if not run.inspection:
            run.source_sha256 = self.workspace.source_sha256
            run.source_id = self.workspace.source_id
            run.inspection = run._inspect_source()
            run.origin_xyz = tuple(float(value) for value in run.inspection["bounds_min_xyz"])
        return run

    def _preflight_ground(self) -> None:
        """Select the preferred or offline fallback ground/HAG backend.

        PDAL is preferred.  If Windows enterprise code-integrity policy blocks
        one of its native DLLs, the implemented NumPy/SciPy offline backend is
        selected explicitly instead of making a new LAS project unusable.
        Same-source cache validation remains unchanged.
        """

        try:
            import pdal  # noqa: F401

            pdal_available = True
            pdal_error = None
        except (ImportError, OSError) as exc:
            pdal_available = False
            pdal_error = f"{type(exc).__name__}: {exc}"
        cache = self.ground_reuse_dir or _find_prior_ground_cache(self.workspace)
        # A ground cache may match the source LAS but belong to a DIFFERENT
        # coarse working set (e.g. copied from a one-click run).  The cache is
        # only reusable when it also matches the workspace's own preprocess
        # working set — otherwise recomputation is genuinely required.
        workset_match = _ground_cache_matches_working_set(self.workspace, cache)
        self.run_metadata["GROUND_BACKEND"] = (
            "PDAL_SMRF_HAG_NN" if pdal_available else "NUMPY_SCIPY_OFFLINE_FALLBACK"
        )
        if pdal_error:
            self.run_metadata["PDAL_UNAVAILABLE_REASON"] = pdal_error
        if cache is None:
            self.run_metadata["GROUND_CACHE"] = "NONE"
        elif workset_match:
            self.run_metadata["GROUND_CACHE"] = "REUSED_CANONICAL_CACHE"
            self.run_metadata["GROUND_SOURCE"] = "REUSED_CANONICAL_CACHE"
            self.run_metadata["ground_cache_source_identity_check"] = "PASS"
        else:
            self.run_metadata["GROUND_CACHE"] = "MISMATCHED_WORKING_SET"
        self.run_metadata["ground_cache_working_set_match"] = (
            "PASS" if workset_match else "MISMATCH"
        )
        if not pdal_available:
            self.run_metadata["GROUND_FALLBACK_NOTICE"] = (
                "PDAL 原生库不可用；本次使用离线 NumPy/SciPy 地面与 HAG 后端，"
                "并在结果中保留后端名称供复核。"
            )

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
        # Restore the raw-click session state so the app does not behave as if
        # the owner must select Pole A/B again: the persisted payload records
        # the original clicked absolute coordinates for both poles.
        restored_clicks: dict[str, dict[str, float]] = {"A": {}, "B": {}}
        for label in ("A", "B"):
            clicked = payload.get(f"pole_{label.lower()}") or {}
            clicked_xyz = clicked.get("clicked_xyz_absolute_m")
            if clicked_xyz:
                restored_clicks[label] = {
                    "x": float(clicked_xyz[0]),
                    "y": float(clicked_xyz[1]),
                }
        self.state.update(selection=payload, clicks=restored_clicks)
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

    def _ensure_display_sample(self) -> dict[str, Any]:
        """Hydrate the STEP-1 display sample exactly once (authoritative builder).

        The Folder pipeline reuses the normal ManualSpan display lifecycle:
        ``run._stage_display_sample()`` (a display-only bounded sample, cached at
        ``00_input/display_sample_v4.npz``).  It never triggers downstream
        science — no preprocess / ground / wire / tree / clearance.  If the
        sample already exists it is returned as-is.
        """

        run = self._ensure_display_run()
        existing = getattr(run, "display_sample", None)
        if existing is not None:
            return existing
        try:
            sample = run._stage_display_sample()
        except Exception as exc:  # noqa: BLE001 - controlled failure for the UI
            raise FolderPipelineError(f"点云显示采样准备失败: {_brief(exc)}") from exc
        run.display_sample = sample
        return sample

    def display_points(self) -> dict[str, Any]:
        """Return the bounded STEP-1 display sample (never the full source)."""

        run = self._require_run()
        sample = self._ensure_display_sample()
        inspection = getattr(run, "inspection", None) or {}
        return {
            "xyz": sample["xyz_absolute"],
            "rgb": sample["rgb"],
            "source_bounds": inspection.get("bounds_min_xyz", [0.0, 0.0, 0.0]),
            "source_bounds_max": inspection.get("bounds_max_xyz", [0.0, 0.0, 0.0]),
            "origin_xyz": tuple(getattr(run, "origin_xyz", (0.0, 0.0, 0.0))),
        }

    def source_bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return the authoritative source-LAS absolute bounds (min, max)."""

        run = self._ensure_display_run()
        inspection = getattr(run, "inspection", None) or {}
        minimum = inspection.get("bounds_min_xyz", [0.0, 0.0, 0.0])
        maximum = inspection.get("bounds_max_xyz", [0.0, 0.0, 0.0])
        return (
            (float(minimum[0]), float(minimum[1]), float(minimum[2])),
            (float(maximum[0]), float(maximum[1]), float(maximum[2])),
        )

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
        (
            wire_region,
            veg_region,
            wire_candidate_ids,
            veg_retained_ids,
            _routing_summary,
        ) = run._stage_regions()
        regions_digest = _canonical_digest(
            {
                "wire_region_count": wire_region.count,
                "vegetation_region_count": veg_region.count,
                "wire_candidate_reps": int(len(wire_candidate_ids)),
                "vegetation_retained_reps": int(len(veg_retained_ids)),
                "strict_routed_reps": int(_routing_summary["strict_routed_rep_count"]),
                "rescued_routed_reps": int(_routing_summary["rescued_routed_rep_count"]),
                "union_routed_reps": int(_routing_summary["union_routed_rep_count"]),
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

        ``FINAL_TREE_INPUT = VEGETATION - CONFIRMED_WIRE - CONFIRMED_TOWER``.
        Both exact sets are returned and the caller subtracts them from the
        vegetation region.

        - Confirmed wire: ``run.wire_outcome.confirmed_member_source_index``
          (the ONLY authoritative wire membership; display points never enter).
        - Tower: an authoritative full ``TowerLocalMembership`` artifact bound
          to this source is used FIRST when present (exact original_index).
          Otherwise a DERIVED pole shaft core is used (exact full-res
          original_index inside a tight 0.6 m shaft cylinder around the two
          manually-confirmed poles).  The derived core is NEVER labelled as a
          complete authoritative tower membership and the region is never
          enlarged merely to remove more tower points.
        """

        run = self._require_run()
        parts: list[np.ndarray] = []
        confirmed_wire = np.asarray(
            run.wire_outcome.confirmed_member_source_index, dtype=np.int64
        )
        parts.append(confirmed_wire)
        wire_excluded_in_veg = _region_overlap_count(veg_region, confirmed_wire)

        authoritative = _authoritative_tower_membership(self.workspace)
        if authoritative is not None:
            tower_source = authoritative
            self.run_metadata["tower_exclusion_policy"] = "EXACT_ORIGINAL_INDEX"
            self.run_metadata["tower_exclusion_coverage"] = "FULL"
            self.run_metadata["tower_membership_source"] = "AUTHORITATIVE_TOWER_LOCAL_MEMBERSHIP"
            self.run_metadata["tower_authoritative_membership"] = "YES"
        else:
            tower_parts: list[np.ndarray] = []
            for pole in (run.pole_a, run.pole_b):
                if pole is None:
                    continue
                tower_parts.append(_pole_shaft_source_indices(run, pole))
            tower_source = (
                np.unique(np.concatenate(tower_parts))
                if tower_parts
                else np.empty(0, dtype=np.int64)
            )
            # The derived shaft core covers only the two pole columns; it is
            # partial coverage, never a full authoritative membership.
            self.run_metadata["tower_exclusion_policy"] = "DERIVED_POLE_SHAFT_CORE_EXACT_INDICES"
            self.run_metadata["tower_exclusion_coverage"] = (
                "PARTIAL" if len(tower_source) > 0 else "UNAVAILABLE"
            )
            self.run_metadata["tower_membership_source"] = "DERIVED_GEOMETRIC_SHAFT_CORE"
            self.run_metadata["tower_authoritative_membership"] = "NO"

        tower_excluded_in_veg = _region_overlap_count(veg_region, tower_source)
        # FINAL_TREE_INPUT = VEGETATION - CONFIRMED_WIRE - CONFIRMED_TOWER.
        # The exact tower source indices ARE part of the actual subtraction set
        # (previously they were audit-only, leaving tower points in tree input).
        parts.append(tower_source)
        excluded = np.unique(np.concatenate(parts))
        # Audit counts against the FINAL tree input (vegetation region).
        self.run_metadata["confirmed_wire_excluded_count"] = int(wire_excluded_in_veg)
        self.run_metadata["confirmed_tower_excluded_count"] = int(tower_excluded_in_veg)
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

    def hydrate_completed_workspace(self) -> None:
        """Restore a completed Step-2 workspace from persisted artifacts only.

        This is the restart path for a workspace whose science already finished:
        it never re-runs preprocess / Ground / Wire / Tree, never re-hashes the
        105M-point LAS, and never loads the heavy coarse-representative evidence.
        Everything the Step-2 editor needs (background sample, selection, wire
        runtime, region dirs, original trees) is loaded from the persisted
        artifacts/checkpoints.
        """

        run = self._ensure_display_run()
        # The run's source identity must match the science checkpoints' key so a
        # later (owner-initiated) science re-run still reuses them.
        run.source_sha256 = self.workspace.source_sha256
        run.source_id = f"manual_span_{self.workspace.source_sha256[:16]}"
        if run.inspection:
            run.origin_xyz = tuple(
                float(value) for value in run.inspection["bounds_min_xyz"]
            )
        self._ensure_display_sample()
        self.load_selection()
        self.hydrate_wire_runtime()
        self.build_editor()

    def hydrate_wire_runtime(self) -> None:
        """Load the persisted 05_wire result + region dirs into the runtime model.

        A Folder server restart builds the run from checkpoints but never
        re-runs science, so ``run.wire_outcome`` stays None.  This restores the
        authoritative Wire V2 result (strands + exact confirmed member source
        indices) and the region dirs from the persisted artifacts so Step-2
        display can show the confirmed conductors without recomputation.
        """

        run = self.run
        if run is None:
            return
        if run.wire_outcome is None:
            span_dir = run.run_root / "05_wire" / "span_001"
            if (span_dir / "w1_arrays.npz").is_file() and (
                span_dir / "span_wire_result.json"
            ).is_file():
                from algorithms.spans.wire import load_span_wire_result

                run.wire_outcome = load_span_wire_result(span_dir)
        if run.wire_region_dir is None:
            wire_dir = run.run_root / "04_regions" / "wire_region"
            if (wire_dir / "region_sidecar.json").is_file():
                run.wire_region_dir = wire_dir
        if run.vegetation_region_dir is None:
            veg_dir = run.run_root / "04_regions" / "vegetation_region"
            if (veg_dir / "region_sidecar.json").is_file():
                run.vegetation_region_dir = veg_dir

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

    def _trim_binding(self) -> str:
        run = self._require_run()
        return f"{self.workspace.root}|{self.workspace.source_sha256}|{run.source_id}"

    def editor_preview_trim_component(
        self, tree_id: int, seed_xyz_absolute: list[float]
    ) -> dict[str, Any]:
        session = self.build_editor()
        run = self._require_run()
        seed = np.asarray(seed_xyz_absolute, dtype=np.float64) - np.asarray(
            run.origin_xyz, dtype=np.float64
        )
        preview = session.preview_trim_component(
            int(tree_id), seed, binding=self._trim_binding()
        )
        preview["preview_xyz"] = self._trim_preview_xyz(session, preview["preview_id"])
        return preview

    def editor_preview_trim_lasso(
        self, tree_id: int, view_projection: list[list[float]], polygon_ndc: list[list[float]]
    ) -> dict[str, Any]:
        session = self.build_editor()
        # The matrix is built against absolute coordinates shown by Plotly.
        # Fold the local-to-absolute translation into it without copying or
        # mutating the authoritative full-resolution coordinate array.
        run = self._require_run()
        matrix = np.asarray(view_projection, dtype=np.float64)
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = np.asarray(run.origin_xyz, dtype=np.float64)
        preview = session.preview_trim_lasso(
            int(tree_id), (matrix @ translation).tolist(), polygon_ndc,
            binding=self._trim_binding(),
        )
        preview["preview_xyz"] = self._trim_preview_xyz(session, preview["preview_id"])
        return preview

    def _trim_preview_xyz(self, session: TreeEditSession, preview_id: str) -> list[list[float]]:
        xyz = session.preview_xyz(preview_id)
        rows = np.linspace(0, len(xyz) - 1, min(len(xyz), 60_000), dtype=np.int64)
        origin = np.asarray(self._require_run().origin_xyz, dtype=np.float64)
        return (xyz[rows] + origin).tolist()

    def editor_commit_trim(self, preview_id: str) -> dict[str, Any]:
        session = self.build_editor()
        run = self._require_run()
        session.commit_trim(
            preview_id,
            binding=self._trim_binding(),
            minimum_remaining_points=int(run.cfg.tree_min_final_members),
        )
        self._persist_edits(session)
        self.state.update(tree_edit_count=len(session.operations))
        return session.snapshot()

    def editor_cancel_trim(self, preview_id: str) -> dict[str, Any]:
        session = self.build_editor()
        session.cancel_trim(preview_id)
        return session.snapshot()

    def _persist_edits(self, session: TreeEditSession) -> None:
        payload = session.edits_payload()
        payload["source_sha256"] = self.workspace.source_sha256
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(self.workspace.tree_edits_path, payload)
        # Keep the editor list exposed by /state in sync with the latest edit;
        # otherwise a refresh after delete/merge flashes the stale snapshot
        # (captured at build_editor time) back over the right-hand panel.
        self.state.update(tree_snapshot=session.snapshot())

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
            session.edit_provenance_by_current_tree(),
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

        # 9. persist the STEP 3 final result page (offline copy of /step3).
        from algorithms.folder_pipeline.html import step3_html

        self.workspace.final_html_path.write_text(step3_html(), encoding="utf-8")
        self.run_metadata["final_html_path"] = str(self.workspace.final_html_path)

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
        original_index = np.asarray(reconciled.final_original_index, dtype=np.int64)
        tree_id = np.asarray(reconciled.final_tree_id, dtype=np.int32)
        xyz_local_m = np.asarray(reconciled.final_xyz_local_m, dtype=np.float64)
        _write_npz(
            self.workspace.original_tree_npz,
            original_index=original_index,
            tree_id=tree_id,
            xyz_local_m=xyz_local_m,
        )
        # Memmap-compatible editor store (npz members cannot be memory-mapped in
        # NumPy 2.x): lets a Step-2 restart read the 16M-row editor arrays from
        # disk instead of materializing ~585 MB of ordinary RAM for display.
        _write_editor_store(self.workspace, original_index, tree_id, xyz_local_m)
        _write_json(
            self.workspace.original_tree_manifest,
            {
                "schema": "folder-to-excel-tree-segmentation-original-v1",
                "tree_backend": "crown_stem_hybrid",
                # algorithm_tree_count = EDITOR-ELIGIBLE trees (span-owned after
                # reconcile); source_algorithm_tree_count = the full algorithm
                # CONFIRMED_TREE records including halo-context trees that the
                # reconcile ownership rule does not promote to the editor.
                "algorithm_tree_count": len(reconciled.global_tree_records),
                "source_algorithm_tree_count": (
                    len(self.run.tree_outcome.tree_records)
                    if self.run.tree_outcome is not None
                    else len(reconciled.global_tree_records)
                ),
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
        # Load the original-tree arrays as disk-backed memmaps when the editor
        # store is present (preferred): the editor only relabels membership (XYZ
        # unchanged), so tens of millions of rows never need to sit in ordinary
        # RAM just for Step-2 display.  Falls back to the npz for older
        # workspaces that predate the store.
        store_arrays = _load_editor_store(self.workspace)
        if store_arrays is not None:
            original_index, tree_id, xyz_local_m = store_arrays
        else:
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
    payload["start_tower_id"] = "T001"  # pole A, deterministic allocation
    payload["end_tower_id"] = "T002"  # pole B
    payload["tower_membership_references"] = {
        "pole_a": "MANUAL_POLE_REFINEMENT_SHAFT_CORE",
        "pole_b": "MANUAL_POLE_REFINEMENT_SHAFT_CORE",
    }
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


def _region_overlap_count(region: Any, candidate: np.ndarray) -> int:
    """Count exact ``original_index`` rows of *region* present in *candidate*.

    Streams the region's source-index memmap in bounded chunks (1M rows) so the
    audit never materializes the full 38M-row vegetation index at once — the
    whole index array would otherwise be passed to ``numpy.unique`` (hash-based
    in NumPy >= 2) and exceed the RAM of a 16 GB owner machine.
    """

    candidate = np.unique(np.asarray(candidate, dtype=np.int64))
    if len(candidate) == 0 or region.count == 0:
        return 0
    original = region.original_index()
    count = 0
    chunk_size = 1_000_000
    for start in range(0, region.count, chunk_size):
        stop = min(region.count, start + chunk_size)
        chunk = np.asarray(original[start:stop], dtype=np.int64)
        positions = np.searchsorted(candidate, chunk)
        positions = np.minimum(positions, len(candidate) - 1)
        count += int(np.count_nonzero(candidate[positions] == chunk))
    return count


def _editor_store_dir(workspace: Any) -> Path:
    """Directory holding memmap-compatible copies of the editor input arrays."""
    return Path(workspace.tree_analysis_dir) / "tree_editor_store"


def _write_editor_store(
    workspace: Any,
    original_index: np.ndarray,
    tree_id: np.ndarray,
    xyz_local_m: np.ndarray,
) -> Path:
    """Persist the Step-2 editor arrays as standalone .npy memmap files.

    npz members cannot be memory-mapped in NumPy 2.x, so a completed workspace
    would otherwise reload ~585 MB of editor arrays into RAM on every restart.
    The .npy files are loaded with ``open_memmap(mode="r")`` and keep the editor
    display disk-backed.
    """

    store = _editor_store_dir(workspace)
    store.mkdir(parents=True, exist_ok=True)
    _write_editor_array(store / "original_index.npy", original_index)
    _write_editor_array(store / "tree_id.npy", tree_id)
    _write_editor_array(store / "xyz_local_m.npy", xyz_local_m)
    (store / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "tree-editor-store-v1",
                "count": int(len(original_index)),
                "xyz_dtype": "float64",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return store


def _write_editor_array(path: Path, array: np.ndarray) -> None:
    """Atomic ``np.save`` of a .npy file (memmap-compatible for later reads)."""
    # np.save appends ".npy" unless the target already ends with it.
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def _load_editor_store(workspace: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return memmap views (original_index, tree_id, xyz_local_m) of the editor input.

    ``None`` when the store is absent (older workspace) so the caller falls back
    to loading the npz.
    """

    store = _editor_store_dir(workspace)
    names = ("original_index.npy", "tree_id.npy", "xyz_local_m.npy")
    if not all((store / name).is_file() for name in names):
        return None
    return tuple(
        np.lib.format.open_memmap(store / name, mode="r")
        for name in names
    )  # type: ignore[return-value]


def _tree_input_audit(
    veg_region: Any,
    cleaned_region: Any,
    excluded: np.ndarray,
    tower_source_indices: np.ndarray,
) -> dict[str, Any]:
    wire_left = _region_overlap_count(cleaned_region, excluded)
    tower_left = _region_overlap_count(cleaned_region, tower_source_indices)
    return {
        "vegetation_input_count": int(veg_region.count),
        "final_tree_input_count": int(cleaned_region.count),
        "wire_tree_input_intersection": int(wire_left),
        "tower_tree_input_intersection": int(tower_left),
    }


def _build_edited_reconciliation(
    run: Any,
    final_index: np.ndarray,
    final_xyz: np.ndarray,
    final_tree: np.ndarray,
    merged_summary: dict[int, dict[str, Any]],
    original_trees: OriginalTrees | None,
    workspace: FolderWorkspace,
    edit_provenance: dict[int, tuple[str, ...]] | None = None,
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
                identity_state=(
                    "HUMAN_" + "+".join((edit_provenance or {}).get(tree_id, ()))
                    if (edit_provenance or {}).get(tree_id)
                    else "TREE_EDITOR_FINAL"
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
    return str(getattr(record, "identity_state", "")).startswith("HUMAN_") or bool(
        getattr(record, "merged_span_tree_keys", None)
        and len(record.merged_span_tree_keys) > 1
    )


def _edit_provenance(record: Any) -> str:
    state = str(getattr(record, "identity_state", ""))
    if state.startswith("HUMAN_"):
        return state.removeprefix("HUMAN_")
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
        "risk_threshold_value": orchestrator.threshold_m,
        "risk_threshold_source": "algorithms.spans.project_adapter.RISK_THRESHOLD_M",
        "risk_config_digest": _risk_config_digest(),
        "project_id": (selection or {}).get("project_id"),
        "group_id": (selection or {}).get("group_id"),
        "span_id": (selection or {}).get("span_id"),
        "start_tower_id": (selection or {}).get("start_tower_id", "T001"),
        "end_tower_id": (selection or {}).get("end_tower_id", "T002"),
        "start_tower_xyz": (
            (selection or {}).get("pole_a", {}).get("base_xyz_absolute_m")
        ),
        "end_tower_xyz": (
            (selection or {}).get("pole_b", {}).get("base_xyz_absolute_m")
        ),
        "science_run_id": time.strftime("folder_%Y%m%d_%H%M%S"),
    }


def _ground_cache_matches_working_set(
    workspace: FolderWorkspace, cache_dir: Path | None
) -> bool:
    """A ground cache is reusable only when it matches the workspace working set.

    The coarse ground is computed on the PREPROCESS working set (the
    representative LAS), not the original source.  A cache that matches the
    source LAS but belongs to a different working set (e.g. copied from a
    one-click run) cannot be reused without recomputation.  When there is no
    workspace working set yet (first run) there is nothing to cross-check.
    """

    if cache_dir is None:
        return False
    work_las = workspace.science_run_root / "01_preprocess" / "working_set_representatives.las"
    result_json = Path(cache_dir) / "ground_hag_result.json"
    if not work_las.is_file() or not result_json.is_file():
        return True  # nothing to cross-check yet (first run / partial cache)
    import hashlib

    digest = hashlib.sha256()
    with work_las.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    current_sha = digest.hexdigest()
    try:
        payload = json.loads(result_json.read_text("utf-8"))
        recorded = payload.get("source_las_sha256")
    except Exception:  # noqa: BLE001 - best effort
        return False
    if not recorded:
        return False
    return str(recorded) == current_sha


def _authoritative_tower_membership(workspace: FolderWorkspace) -> np.ndarray | None:
    """Return the exact original_index union of authoritative tower memberships.

    Only artifacts whose source identity digest EXACTLY matches the folder
    source are used (source-identity validated).  When none exists, ``None`` is
    returned and the pipeline falls back to the derived pole shaft cores with
    honest ``PARTIAL`` coverage.  No authoritative membership is ever invented.
    """

    try:
        from scripts.prepare_tower_review_project import build_real_source_identity

        identity = build_real_source_identity(workspace.source)
        expected = str(identity.source_content_identity_digest)
    except Exception:  # noqa: BLE001 - no authoritative artifacts are required
        return None

    parts: list[np.ndarray] = []
    for root in (
        Path("cache") / "tower_ordered_project",
        Path("cache") / "tower_review_project",
        Path("projects"),
        workspace.source_cache_dir,
    ):
        memberships = root / "tower-memberships"
        if not memberships.is_dir():
            continue
        for artifact in sorted(memberships.glob("*.npz")):
            try:
                with np.load(artifact, allow_pickle=False) as stream:
                    digest_field = stream.get("meta_source_content_identity_digest")
                    if digest_field is None or len(digest_field) == 0:
                        continue
                    if str(digest_field[0]) != expected:
                        continue  # source identity mismatch -> never use
                    source_index = np.asarray(stream["source_index"], dtype=np.int64)
                    if len(source_index):
                        parts.append(source_index)
            except Exception:  # noqa: BLE001 - best effort, fail closed to derived
                continue
    if not parts:
        return None
    return np.unique(np.concatenate(parts))


__all__ = [
    "_FOLDER_RUN_SCHEMA",
    "_TREE_INPUT_CLEANING_SCHEMA",
    "FolderPipelineError",
    "FolderOrchestrator",
    "_canonical_risk_threshold_m",
    "_authoritative_tower_membership",
    "_ground_cache_matches_working_set",
]
