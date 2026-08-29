"""Desktop presentation surface for human-facing final tree numbering."""

from __future__ import annotations

from functools import wraps
from typing import Any

from core.stage_b.tree_display_id import TREE_DISPLAY_ID_SCHEMA, assign_tree_display_ids

_INSTALLED = False


def _current_display_identities(self: Any, editor: Any) -> dict[int, dict[str, Any]]:
    """Build presentation IDs from the current exact editor membership.

    Legacy/folder-only projections do not own a Stage-A project sidecar.  Tree
    business numbering is optional presentation metadata there, so absence of
    ``stage-a-project.json`` must preserve the established projection instead
    of turning a display enhancement into a hard dependency.
    """

    reconciled = None
    if callable(getattr(editor, "final_reconciled", None)):
        try:
            reconciled = editor.final_reconciled()
        except (RuntimeError, TypeError, ValueError):
            reconciled = None
    if reconciled is None:
        result = getattr(self, "_result", None) or getattr(editor, "result", None)
        reconciled = getattr(result, "global_reconciled", None)
    if reconciled is None:
        return {}
    try:
        project = self._stage_project()
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return {}
    return assign_tree_display_ids(project, reconciled.global_tree_records)


def install_tree_display_numbering_extensions() -> None:
    """Expose ``T-<span>-<tree>`` without replacing durable machine identities."""

    global _INSTALLED
    if _INSTALLED:
        return

    from algorithms.folder_pipeline import html as folder_html

    from desktop.workflow import DesktopWorkflow

    original_editor_data = DesktopWorkflow.editor_data
    original_final_projection = DesktopWorkflow.final_projection
    original_step2_html = folder_html.step2_html
    original_step3_html = folder_html.step3_html

    @wraps(original_editor_data)
    def editor_data_with_display_ids(
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
        editor = self._ensure_editor()
        if editor is None:
            return display
        identities = _current_display_identities(self, editor)
        if not identities:
            return display

        for tree in display.get("trees", []):
            try:
                tree_id = int(tree["tree_id"])
            except (KeyError, TypeError, ValueError):
                continue
            identity = identities.get(tree_id)
            if identity is not None:
                tree["display_tree_id"] = identity["display_tree_id"]

        trees3d = display.get("trees3d", {})
        labels = trees3d.get("labels", []) if isinstance(trees3d, dict) else []
        for label in labels:
            try:
                tree_id = int(label["tree_id"])
            except (KeyError, TypeError, ValueError):
                continue
            identity = identities.get(tree_id)
            if identity is not None:
                # Numeric tree_id remains untouched for selection/edit commands;
                # only the visible label becomes the operator-facing business ID.
                label["text"] = identity["display_tree_id"]
        return display

    @wraps(original_final_projection)
    def final_projection_with_display_ids(self: Any) -> dict[str, Any]:
        projection = original_final_projection(self)
        trees = projection.get("trees")
        if not isinstance(trees, list) or not trees:
            return projection
        editor = self._ensure_editor()
        if editor is None:
            return projection
        identities = _current_display_identities(self, editor)
        if not identities:
            return projection
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

    @wraps(original_step2_html)
    def step2_html_with_display_ids() -> str:
        """Keep numeric edit IDs internal while replacing operator-visible labels."""

        page = original_step2_html()
        patch = r'''
function treeBusinessId(treeId){
  const tree=treeMap.get(Number(treeId));
  return tree?.display_tree_id||treeId;
}
if(window.TreeLineTreeReview3D?.prototype && !window.TreeLineTreeReview3D.prototype._treelineBusinessLabels){
  const proto=window.TreeLineTreeReview3D.prototype;
  const originalRenderLabels=proto.renderLabels;
  proto.renderLabels=function(){
    originalRenderLabels.call(this);
    const labels=Array.isArray(this.source?.trees3d?.labels)?this.source.trees3d.labels:[];
    const visibleById=new Map(labels.map(label=>[Number(label.tree_id),String(label.text||label.tree_id)]));
    this.labelOverlay?.querySelectorAll('.tree-label').forEach(button=>{
      const internalId=Number(button.textContent);
      const visibleId=visibleById.get(internalId);
      if(visibleId){
        button.textContent=visibleId;
        button.title=`选择树 ${visibleId}`;
        button.setAttribute('aria-label',`选择树 ${visibleId}`);
      }
    });
  };
  proto._treelineBusinessLabels=true;
}
'''
        page = page.replace("let treeMap = new Map();", "let treeMap = new Map();\n" + patch, 1)
        page = page.replace("<thead><tr><th>ID</th>", "<thead><tr><th>树编号</th>", 1)
        page = page.replace(
            "tr.innerHTML=`<td>${t.tree_id}</td>",
            "tr.innerHTML=`<td>${t.display_tree_id||t.tree_id}</td>",
            1,
        )
        page = page.replace(
            "button.textContent=String(label.tree_id);",
            "button.textContent=String(label.text||label.tree_id);",
        )
        page = page.replace(
            "button.title=`选择树 ${label.tree_id}`;",
            "button.title=`选择树 ${label.text||label.tree_id}`;",
        )
        page = page.replace(
            "button.setAttribute(\"aria-label\",`选择树 ${label.tree_id}`);",
            "button.setAttribute(\"aria-label\",`选择树 ${label.text||label.tree_id}`);",
        )
        page = page.replace("`当前：树 ${activeTreeId}`", "`当前：树 ${treeBusinessId(activeTreeId)}`")
        page = page.replace('${ids.join("、")}', '${ids.map(treeBusinessId).join("、")}')
        page = page.replace("`删除树 ${ids[0]}`", "`删除树 ${treeBusinessId(ids[0])}`")
        return page

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

    DesktopWorkflow.editor_data = editor_data_with_display_ids
    DesktopWorkflow.final_projection = final_projection_with_display_ids
    folder_html.step2_html = step2_html_with_display_ids
    folder_html.step3_html = step3_html_with_display_ids
    _INSTALLED = True


__all__ = ["install_tree_display_numbering_extensions"]
