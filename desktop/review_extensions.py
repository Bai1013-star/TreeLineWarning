"""Desktop-only review projection extensions.

These wrappers stay at the product adapter boundary: they append exact
clearance evidence to the Giro3D payload, avoid materializing a second giant
XYZ array for region-lasso previews, enrich the final projection with the
canonical post-editor tree metrics, and inject the offline clearance-review UI.
Ground/Wire/Tree/Clearance science is not modified here.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from core.stage_b.danger_points import (
    build_clearance_evidence_display,
    build_danger_point_display,
)
from core.stage_b.tree_metrics import compute_stage_b_tree_metrics
from desktop.review_risk_frontend import TREE_REVIEW_RISK_EXTENSION_JS

_INSTALLED = False


def _stale_layer(authority: str) -> dict[str, Any]:
    return {
        "xyz": [],
        "tree_id_by_index": [],
        "original_index": [],
        "display_point_count": 0,
        "full_point_count": 0,
        "authority": authority,
        "display_only_sampling": False,
    }


def _clearance_is_current(editor: Any, result: Any) -> bool:
    operation_count = (
        len(editor.session.operations)
        if editor is not None and getattr(editor, "session", None) is not None
        else 0
    )
    reconciled = getattr(result, "global_reconciled", None)
    diagnostics = getattr(reconciled, "diagnostics", {}) if reconciled is not None else {}
    post_editor = (
        isinstance(diagnostics, dict)
        and diagnostics.get("editor_schema") == "treeline-stage-b-batch-editor-v1"
    )
    if not post_editor:
        return operation_count == 0
    published_count = diagnostics.get("editor_operation_count")
    return type(published_count) is int and published_count == operation_count


def _install_step2_clearance_ui() -> None:
    """Inject the local-only risk controls before the Step-2 app starts."""

    from algorithms.folder_pipeline import html as folder_html

    original_step2_html = folder_html.step2_html
    if getattr(original_step2_html, "_treeline_risk_extension", False):
        return

    @wraps(original_step2_html)
    def step2_html_with_clearance_ui() -> str:
        page = original_step2_html()
        marker = "<script>\nwindow.TreeLineTreeReviewApp.start();"
        if marker not in page:
            raise RuntimeError("STEP2_RISK_EXTENSION_INJECTION_POINT_MISSING")
        block = f"<script>\n{TREE_REVIEW_RISK_EXTENSION_JS}\n</script>\n"
        return page.replace(marker, block + marker, 1)

    step2_html_with_clearance_ui._treeline_risk_extension = True  # type: ignore[attr-defined]
    folder_html.step2_html = step2_html_with_clearance_ui


def install_review_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from desktop.workflow import DesktopWorkflow

    original_editor_data = DesktopWorkflow.editor_data
    original_preview_trim_lasso = DesktopWorkflow.preview_trim_lasso
    original_final_projection = DesktopWorkflow.final_projection

    @wraps(original_editor_data)
    def editor_data_with_clearance_evidence(
        self: Any,
        *,
        display_budget: int | None = None,
        span_id: str | None = None,
    ) -> dict[str, Any]:
        display = original_editor_data(
            self,
            display_budget=display_budget,
            span_id=span_id,
        )
        editor = getattr(self, "_editor", None)
        result = getattr(editor, "result", None) if editor is not None else None
        if result is None:
            result = getattr(self, "_result", None)
        visible_tree_ids = [
            int(tree["tree_id"])
            for tree in display.get("trees", [])
            if isinstance(tree, dict) and "tree_id" in tree
        ]
        budget = 20_000 if display_budget is None else max(
            1_000,
            min(20_000, int(display_budget) // 2),
        )

        if result is None or not _clearance_is_current(editor, result):
            display["danger_points"] = _stale_layer("STALE_AFTER_TREE_EDIT")
            display["clearance_evidence"] = {
                "records": [],
                "record_count": 0,
                "authority": "STALE_AFTER_TREE_EDIT",
            }
        else:
            display["danger_points"] = build_danger_point_display(
                self.project_root,
                result,
                visible_tree_ids=visible_tree_ids,
                display_budget=budget,
            )
            display["clearance_evidence"] = build_clearance_evidence_display(
                result,
                visible_tree_ids=visible_tree_ids,
            )
        return display

    @wraps(original_preview_trim_lasso)
    def preview_trim_lasso_bounded(
        self: Any,
        tree_id: int,
        view_projection: list[list[float]],
        polygon_ndc: list[list[float]],
    ) -> dict[str, Any]:
        # Keep the established single-tree path byte-for-byte compatible.
        if int(tree_id) >= 0:
            return original_preview_trim_lasso(
                self,
                int(tree_id),
                view_projection,
                polygon_ndc,
            )
        self._assert_editor_mutable()
        editor = self._ensure_editor()
        if editor is None:
            raise ValueError("TREE_EDITOR_REQUIRES_GLOBAL_RECONCILIATION")
        preview = editor.session.preview_trim_lasso(
            int(tree_id),
            view_projection,
            polygon_ndc,
            binding=self._trim_binding(),
        )
        # The browser has already painted a red preview from its bounded
        # display sample. Returning all exact XYZ would defeat the point of a
        # full-resolution region edit, so only exact counts/indices remain
        # authoritative server-side; commit still uses complete membership.
        preview["full_point_count"] = int(preview.get("point_count", 0))
        preview["preview_xyz"] = []
        return preview

    @wraps(original_final_projection)
    def final_projection_with_metrics(self: Any) -> dict[str, Any]:
        projection = original_final_projection(self)
        editor = self._ensure_editor()
        if editor is None:
            return projection
        result = getattr(self, "_result", None) or getattr(editor, "result", None)
        reconciled = (
            getattr(result, "global_reconciled", None) if result is not None else None
        )
        if result is None or reconciled is None:
            return projection
        metrics = compute_stage_b_tree_metrics(
            result,
            reconciled,
            self.project_root / "exports" / "Prediction_Input.xlsx",
            project_root=self.project_root,
        )
        persistent_to_metric = {
            str(record.persistent_tree_id): metrics.get(int(record.global_tree_id))
            for record in reconciled.global_tree_records
        }
        for row in projection.get("trees", []):
            metric = persistent_to_metric.get(str(row.get("persistent_tree_id")))
            if metric is None:
                continue
            row["height_m"] = (
                metric.tree_height_m
                if metric.tree_height_m is not None
                else row.get("height_m")
            )
            row["crown_width_m"] = metric.crown_mean_width_m
            row["crown_width_major_m"] = metric.crown_major_width_m
            row["crown_width_minor_m"] = metric.crown_minor_width_m
            row["crown_area_m2"] = metric.crown_area_m2
            row["diameter_cm"] = (
                None
                if metric.dbh.dbh_m is None
                else float(metric.dbh.dbh_m) * 100.0
            )
            row["diameter_source"] = (
                f"BREAST_HEIGHT_1P30M_{metric.dbh.dbh_quality}"
            )
        projection.setdefault("metadata", {})["tree_metrics_source"] = (
            "CANONICAL_FULLRES_POST_EDITOR"
        )
        return projection

    DesktopWorkflow.editor_data = editor_data_with_clearance_evidence
    DesktopWorkflow.preview_trim_lasso = preview_trim_lasso_bounded
    DesktopWorkflow.final_projection = final_projection_with_metrics
    _install_step2_clearance_ui()
    _INSTALLED = True


__all__ = ["install_review_extensions"]
