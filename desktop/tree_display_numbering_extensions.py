"""Desktop presentation surface for human-facing final tree numbering."""

from __future__ import annotations

import threading
from functools import wraps
from pathlib import Path
from typing import Any

from core.stage_b.tree_display_id import TREE_DISPLAY_ID_SCHEMA, assign_tree_display_ids

_INSTALLED = False


def _current_display_identities(self: Any, editor: Any) -> dict[int, dict[str, Any]]:
    """Reuse unchanged display labels; never rebuild final arrays on every status poll."""
    session = getattr(editor, "session", None)
    root = getattr(self, "project_root", None)
    if session is None or root is None:
        return _compute_display_identities(self, editor)
    manifest = Path(root) / "stage-a-project.json"
    try:
        stat = manifest.stat()
    except OSError:
        return _compute_display_identities(self, editor)
    key = (str(manifest), stat.st_mtime_ns, stat.st_size)
    operations = tuple(session.operations)
    lock = self.__dict__.setdefault("_editor_identity_lock", threading.RLock())
    with lock:
        cached = getattr(self, "_editor_identity_cache", None)
        if (
            cached is not None
            and cached[0] is session
            and cached[2] == key
            and len(cached[1]) == len(operations)
            and all(a is b for a, b in zip(cached[1], operations, strict=True))
        ):
            return cached[3]
        identities = _compute_display_identities(self, editor)
        # Keep operation references, not just len/id values: Undo + a replacement
        # edit must not accidentally reuse an earlier revision's display labels.
        self._editor_identity_cache = (session, operations, key, identities)
        return identities


def _compute_display_identities(self: Any, editor: Any) -> dict[int, dict[str, Any]]:
    """Build presentation IDs from the current exact editor membership.

    Legacy/folder-only projections do not own a Stage-A project sidecar. Tree
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


def _append_display_ids_to_trees(
    trees: Any,
    identities: dict[int, dict[str, Any]],
) -> None:
    if not isinstance(trees, list):
        return
    for tree in trees:
        if not isinstance(tree, dict):
            continue
        try:
            tree_id = int(tree["tree_id"])
        except (KeyError, TypeError, ValueError):
            continue
        identity = identities.get(tree_id)
        if identity is not None:
            tree["display_tree_id"] = identity["display_tree_id"]


_TREE_REVIEW_DISPLAY_PATCH = r"""
(function(){
  const app=window.TreeLineTreeReviewApp;
  if(!app||!app.state)return;
  function treeBusinessId(treeId){
    const t=app.state.treeMap.get(Number(treeId));
    return String((t&&(t.display_tree_id||t.tree_id))??treeId);
  }
  function setText(id,value){
    const node=document.getElementById(id);
    if(node&&node.textContent!==value)node.textContent=value;
  }
  function relabelReview(){
    document.querySelectorAll('#treetable tr[data-tid]').forEach(row=>{
      const cell=row.cells&&row.cells[0];
      const value=treeBusinessId(row.dataset.tid);
      if(cell&&cell.textContent!==value)cell.textContent=value;
    });
    const active=app.state.activeTreeId;
    setText(
      'current-tree',
      active==null?'当前：未选择':`当前：树 ${treeBusinessId(active)}`
    );
    const ids=[...app.state.selected].sort((a,b)=>a-b);
    setText(
      'selection-state',
      ids.length
        ?`已选择 ${ids.length} 棵树：${ids.map(treeBusinessId).join('、')}`
        :'未选择单木'
    );
    setText(
      'btn-delete',
      ids.length===1?`删除树 ${treeBusinessId(ids[0])}`:'删除单木'
    );
  }
  let queued=false;
  const observer=new MutationObserver(()=>{
    if(queued)return;
    queued=true;
    queueMicrotask(()=>{queued=false;relabelReview();});
  });
  observer.observe(document.body,{subtree:true,childList:true,characterData:true});
  const proto=window.TreeLineTreeReview3D?.prototype;
  if(proto&&!proto._treelineBusinessLabels){
    const originalRenderLabels=proto.renderLabels;
    proto.renderLabels=function(){
      originalRenderLabels.call(this);
      const labels=Array.isArray(this.source?.trees3d?.labels)
        ?this.source.trees3d.labels:[];
      const visibleById=new Map(
        labels.map(label=>[
          Number(label.tree_id),
          String(label.text||label.tree_id)
        ])
      );
      this.labelOverlay?.querySelectorAll('.tree-label').forEach(button=>{
        const internalId=Number(button.textContent);
        const visibleId=visibleById.get(internalId);
        if(!visibleId)return;
        button.textContent=visibleId;
        button.title=`选择树 ${visibleId}`;
        button.setAttribute('aria-label',`选择树 ${visibleId}`);
      });
    };
    proto._treelineBusinessLabels=true;
  }
  relabelReview();
})();
"""


def install_tree_display_numbering_extensions() -> None:
    """Expose ``T-<span>-<tree>`` without replacing durable machine identities."""

    global _INSTALLED
    if _INSTALLED:
        return

    from algorithms.folder_pipeline import html as folder_html

    from desktop.workflow import DesktopWorkflow

    original_editor_data = DesktopWorkflow.editor_data
    original_editor_state = DesktopWorkflow.editor_state
    original_final_projection = DesktopWorkflow.final_projection
    original_step2_html = folder_html.step2_html
    original_step3_html = folder_html.step3_html

    @wraps(original_editor_data)
    def editor_data_with_display_ids(
        self: Any,
        *,
        display_budget: int | None = None,
        span_id: str | None = None,
        persist_display_evidence: bool = True,
    ) -> dict[str, Any]:
        display = original_editor_data(
            self,
            display_budget=display_budget,
            span_id=span_id,
            persist_display_evidence=persist_display_evidence,
        )
        editor = self._ensure_editor()
        if editor is None:
            return display
        identities = _current_display_identities(self, editor)
        if not identities:
            return display

        _append_display_ids_to_trees(display.get("trees"), identities)
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

    @wraps(original_editor_state)
    def editor_state_with_display_ids(self: Any) -> dict[str, Any]:
        state = dict(original_editor_state(self))
        snapshot = state.get("tree_snapshot")
        if not isinstance(snapshot, dict):
            return state
        editor = self._ensure_editor()
        if editor is None:
            return state
        identities = _current_display_identities(self, editor)
        if not identities:
            return state
        copied_snapshot = dict(snapshot)
        copied_trees = [
            dict(tree)
            for tree in snapshot.get("trees", [])
            if isinstance(tree, dict)
        ]
        _append_display_ids_to_trees(copied_trees, identities)
        copied_snapshot["trees"] = copied_trees
        state["tree_snapshot"] = copied_snapshot
        return state

    @wraps(original_final_projection)
    def final_projection_with_display_ids(
        self: Any, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        projection = original_final_projection(self, *args, **kwargs)
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
        """Keep numeric edit IDs internal while replacing visible labels."""

        page = original_step2_html()
        page = page.replace(
            "<thead><tr><th>ID</th>",
            "<thead><tr><th>树编号</th>",
            1,
        )
        marker = "window.TreeLineTreeReviewApp.start();"
        if marker not in page:
            raise RuntimeError("TREE_DISPLAY_REVIEW_INJECTION_POINT_MISSING")
        return page.replace(
            marker,
            _TREE_REVIEW_DISPLAY_PATCH + "\n" + marker,
            1,
        )

    @wraps(original_step3_html)
    def step3_html_with_display_ids() -> str:
        # Keep the legacy HTML implementation intact and only change its final
        # presentation label/value. The API still carries persistent_tree_id
        # separately for machine identity and Prediction linkage.
        page = original_step3_html()
        page = page.replace("<th>Tree ID</th>", "<th>树编号</th>")
        return page.replace(
            "${t.persistent_tree_id}",
            "${t.display_tree_id||t.persistent_tree_id}",
        )

    DesktopWorkflow.editor_data = editor_data_with_display_ids
    DesktopWorkflow.editor_state = editor_state_with_display_ids
    DesktopWorkflow.final_projection = final_projection_with_display_ids
    folder_html.step2_html = step2_html_with_display_ids
    folder_html.step3_html = step3_html_with_display_ids
    _INSTALLED = True


__all__ = ["install_tree_display_numbering_extensions"]
