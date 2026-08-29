"""Desktop presentation surface for human-facing final tree numbering."""

from __future__ import annotations

from functools import wraps
from typing import Any

from core.stage_b.tree_display_id import TREE_DISPLAY_ID_SCHEMA, assign_tree_display_ids

_INSTALLED = False


def install_tree_display_numbering_extensions() -> None:
    """Expose ``T-<span>-<tree>`` without replacing durable machine identities."""

    global _INSTALLED
    if _INSTALLED:
        return

    from algorithms.folder_pipeline import html as folder_html

    from desktop.workflow import DesktopWorkflow

    original_final_projection = DesktopWorkflow.final_projection
    original_step3_html = folder_html.step3_html

    @wraps(original_final_projection)
    def final_projection_with_display_ids(self: Any) -> dict[str, Any]:
        projection = original_final_projection(self)
        trees = projection.get("trees")
        if not isinstance(trees, list) or not trees:
            return projection
        editor = self._ensure_editor()
        if editor is None:
            return projection
        result = self._result or editor.result
        reconciled = getattr(result, "global_reconciled", None)
        if reconciled is None:
            return projection
        identities = assign_tree_display_ids(
            self._stage_project(),
            reconciled.global_tree_records,
        )
        snapshot = self._editor_snapshot(editor)
        snapshot_trees = snapshot.get("trees", [])
        for row, snapshot_tree in zip(trees, snapshot_trees, strict=False):
            try:
                tree_id = int(snapshot_tree["tree_id"])
            except (KeyError, TypeError, ValueError):
                continue
            identity = identities.get(tree_id)
            if identity is None:
                continue
            row["display_tree_id"] = identity["display_tree_id"]
            row["display_span_number"] = identity["display_span_number"]
            row["display_tree_order"] = identity["display_tree_order"]
            row["numbering_origin_tower_label"] = identity[
                "numbering_origin_tower_label"
            ]
        metadata = projection.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["tree_display_id_schema"] = TREE_DISPLAY_ID_SCHEMA
            metadata["tree_display_id_semantics"] = (
                "T-<span>-<tree>; 001 starts nearest the smaller-numbered tower"
            )
        return projection

    @wraps(original_step3_html)
    def step3_html_with_display_ids() -> str:
        # Keep the legacy HTML implementation intact and only change its final
        # presentation label/value.  The API still carries persistent_tree_id
        # separately for machine identity and Prediction linkage.
        page = original_step3_html()
        page = page.replace("<th>Tree ID</th>", "<th>树编号</th>")
        return page.replace(
            "${t.persistent_tree_id}",
            "${t.display_tree_id||t.persistent_tree_id}",
        )

    DesktopWorkflow.final_projection = final_projection_with_display_ids
    folder_html.step3_html = step3_html_with_display_ids
    _INSTALLED = True


__all__ = ["install_tree_display_numbering_extensions"]
