"""Small desktop-only review projection extensions.

The desktop workflow itself remains the authoritative source for tree, tower,
wire and editor display data.  This module wraps only ``editor_data`` to append
a bounded red danger-point layer sourced from the exact clearance result (or
its validated post-editor sidecar after restart).  It does not alter science,
editor membership, camera handling, or network behavior.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from core.stage_b.danger_points import build_danger_point_display

_INSTALLED = False


def install_review_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from desktop.workflow import DesktopWorkflow

    original_editor_data = DesktopWorkflow.editor_data

    @wraps(original_editor_data)
    def editor_data_with_danger_points(
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

        # Once edits exist, the pre-editor clearance is stale until finalize
        # recomputes exact clearance.  Never paint stale red points as current.
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
        if operation_count and not post_editor:
            danger = {
                "xyz": [],
                "tree_id_by_index": [],
                "original_index": [],
                "display_point_count": 0,
                "full_point_count": 0,
                "authority": "STALE_AFTER_TREE_EDIT",
                "display_only_sampling": False,
            }
        else:
            danger = build_danger_point_display(
                self.project_root,
                result,
                visible_tree_ids=visible_tree_ids,
                display_budget=budget,
            )
        display["danger_points"] = danger
        return display

    DesktopWorkflow.editor_data = editor_data_with_danger_points
    _INSTALLED = True


__all__ = ["install_review_extensions"]
