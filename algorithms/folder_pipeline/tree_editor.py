"""Human tree editing model (DELETE / MERGE / UNDO) on top of algorithm trees.

The algorithm tree result is NEVER mutated.  It is the immutable
``original``; human edits are a pure operation log.  The current effective
trees and the final membership are always derived deterministically by
replaying the log over the original.  This keeps three layers distinct:

    1. algorithm original        tree_segmentation_original.*
    2. human edit log            tree_edits.json
    3. final result              tree_segmentation_final.*

Semantics:

- DELETE  -> the tree's original_index members leave the final tree set and
             become unassigned vegetation points (raw source untouched).
- MERGE   -> the selected trees' memberships are joined by exact
             ``original_index`` union (duplicates removed, nothing added or
             lost, XYZ unchanged).
- UNDO    -> drop the last operation and recompute (supports both DELETE and
             MERGE).

``original_index`` is the authoritative full-resolution source row identity;
the editor never carries or copies XYZ — the display path decimates on read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

TREE_EDIT_LOG_SCHEMA = "folder-to-excel-tree-edits-v1"


class TreeEditorError(RuntimeError):
    """Closed failure for an invalid edit operation."""


@dataclass(frozen=True, slots=True)
class OriginalTree:
    """One immutable algorithm tree (from the reconciled science result)."""

    tree_id: int
    external_name: str
    point_count: int
    apex_xy_local_m: tuple[float, float, float] | None
    apex_height_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "external_name": self.external_name,
            "point_count": self.point_count,
            "apex_xy_local_m": (
                None if self.apex_xy_local_m is None else list(self.apex_xy_local_m)
            ),
            "apex_height_m": self.apex_height_m,
        }


@dataclass(frozen=True, slots=True)
class OriginalTrees:
    """Immutable algorithm tree set plus its exclusive full-res membership.

    ``original_index`` and ``tree_id`` are aligned arrays (``tree_id == -1``
    marks unassigned rows, which the editor never touches).  ``xyz_local_m``
    is carried as a numpy memmap-backed reference so final geometry can reuse
    the full-resolution science coordinates without re-copying the source.
    """

    records: tuple[OriginalTree, ...]
    original_index: np.ndarray  # int64 aligned
    tree_id: np.ndarray  # int32 aligned (algorithm per-span or global ids)
    xyz_local_m: np.ndarray  # float64 (N, 3) scene-local full-res science rows

    def members_of(self, tree_id: int) -> np.ndarray:
        return self.original_index[self.tree_id == tree_id]

    @property
    def count(self) -> int:
        return len(self.records)


def build_original_trees(
    global_records: Any,
    final_original_index: np.ndarray,
    final_xyz_local_m: np.ndarray,
    final_tree_id: np.ndarray,
) -> OriginalTrees:
    """Build the immutable original tree set from a reconciled science result."""

    records: list[OriginalTree] = []
    for record in global_records:
        records.append(
            OriginalTree(
                tree_id=int(record.global_tree_id),
                external_name=str(record.external_name),
                point_count=int(record.full_resolution_point_count),
                apex_xy_local_m=(
                    tuple(float(value) for value in record.apex_xy_local_m)
                    if record.apex_xy_local_m is not None
                    else None
                ),
                apex_height_m=(
                    None
                    if record.apex_height_m is None
                    else float(record.apex_height_m)
                ),
            )
        )
    return OriginalTrees(
        records=tuple(records),
        original_index=np.asarray(final_original_index, dtype=np.int64),
        tree_id=np.asarray(final_tree_id, dtype=np.int32),
        xyz_local_m=np.asarray(final_xyz_local_m, dtype=np.float64),
    )


def _validate_tree_ids(original: OriginalTrees, tree_ids: list[int]) -> None:
    known = {record.tree_id for record in original.records}
    for tree_id in tree_ids:
        if tree_id not in known:
            raise TreeEditorError(f"未知的单木 id: {tree_id}")


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Outcome of one MERGE operation (deterministic ids)."""

    merged_id: int
    source_ids: tuple[int, ...]


def _apply_ops(
    original: OriginalTrees,
    operations: list[dict[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, MergeResult]]:
    """Replay the edit log -> per-tree member tree_ids and merge bookkeeping.

    Returns:
        tree_members: map final_tree_id -> list of original tree_ids that form it
        merges: map merged_id -> MergeResult
    """

    alive: set[int] = {record.tree_id for record in original.records}
    merges: dict[int, MergeResult] = {}
    next_id = max(alive, default=0) + 1
    for operation in operations:
        op_type = operation.get("type")
        tree_ids = [int(value) for value in operation.get("tree_ids", [])]
        if op_type == "DELETE":
            for tree_id in tree_ids:
                if tree_id in alive:
                    alive.discard(tree_id)
        elif op_type == "MERGE":
            present = [tree_id for tree_id in tree_ids if tree_id in alive]
            if len(present) >= 2:
                for tree_id in present:
                    alive.discard(tree_id)
                merged_id = int(operation.get("merged_id", next_id))
                next_id = max(next_id, merged_id + 1)
                merges[merged_id] = MergeResult(
                    merged_id=merged_id, source_ids=tuple(present)
                )
                alive.add(merged_id)
        else:
            raise TreeEditorError(f"未知编辑操作类型: {op_type}")
    tree_members: dict[int, list[int]] = {}
    for final_id in alive:
        if final_id in merges:
            tree_members[final_id] = list(merges[final_id].source_ids)
        else:
            tree_members[final_id] = [final_id]
    return tree_members, merges


class TreeEditSession:
    """Thread-safe, log-driven tree edit session over one immutable original."""

    def __init__(self, original: OriginalTrees) -> None:
        self.original = original
        self.operations: list[dict[str, Any]] = []
        self.selected_ids: list[int] = []

    # ------------------------------------------------------------------ state
    def current_tree_ids(self) -> list[int]:
        members, _merges = _apply_ops(self.original, self.operations)
        return sorted(members)

    def snapshot(self) -> dict[str, Any]:
        """Project the current effective trees (records + ids) for the editor UI."""

        members, merges = _apply_ops(self.original, self.operations)
        by_id = {record.tree_id: record for record in self.original.records}
        trees: list[dict[str, Any]] = []
        for final_id in sorted(members):
            source_ids = members[final_id]
            if len(source_ids) == 1:
                record = by_id[source_ids[0]]
                trees.append(
                    {
                        "tree_id": final_id,
                        "external_name": record.external_name,
                        "point_count": record.point_count,
                        "apex_xy_local_m": record.apex_xy_local_m,
                        "apex_height_m": record.apex_height_m,
                        "merged": False,
                        "source_tree_ids": source_ids,
                    }
                )
            else:
                merged_id = final_id
                source_count = sum(by_id[tid].point_count for tid in source_ids)
                apex = _union_apex([by_id[tid] for tid in source_ids])
                trees.append(
                    {
                        "tree_id": merged_id,
                        "external_name": f"Merged-Tree{merged_id:03d}",
                        "point_count": source_count,
                        "apex_xy_local_m": apex,
                        "apex_height_m": None,
                        "merged": True,
                        "source_tree_ids": source_ids,
                    }
                )
        return {
            "schema": TREE_EDIT_LOG_SCHEMA,
            "original_tree_count": self.original.count,
            "current_tree_count": len(trees),
            "trees": trees,
            "selected_ids": list(self.selected_ids),
            "operations": [dict(op) for op in self.operations],
        }

    def edits_payload(self) -> dict[str, Any]:
        return {
            "schema": TREE_EDIT_LOG_SCHEMA,
            "source_sha256": None,  # filled by the orchestrator on persistence
            "operations": [dict(op) for op in self.operations],
        }

    # ---------------------------------------------------------------- editing
    def select(self, tree_ids: list[int]) -> None:
        known = set(self.current_tree_ids())
        for tree_id in tree_ids:
            if tree_id not in known:
                raise TreeEditorError(f"当前不存在单木 id: {tree_id}")
        self.selected_ids = list(tree_ids)

    def delete_selected(self) -> list[int]:
        if not self.selected_ids:
            raise TreeEditorError("请先选择要删除的单木")
        self.operations.append({"type": "DELETE", "tree_ids": list(self.selected_ids)})
        deleted = list(self.selected_ids)
        self.selected_ids = []
        return deleted

    def merge_selected(self) -> int:
        if len(self.selected_ids) < 2:
            raise TreeEditorError("合并至少需要选择两棵单木")
        members, _merges = _apply_ops(self.original, self.operations)
        # Only trees that still exist (and are not themselves merged) may merge.
        valid = [tree_id for tree_id in self.selected_ids if tree_id in members]
        if len(valid) < 2:
            raise TreeEditorError("所选单木中存在已删除或已合并的单木")
        next_id = self._next_merged_id()
        self.operations.append(
            {"type": "MERGE", "tree_ids": list(valid), "merged_id": next_id}
        )
        self.selected_ids = []
        return next_id

    def _next_merged_id(self) -> int:
        members, merges = _apply_ops(self.original, self.operations)
        used = set(members) | set(merges)
        candidate = max(used, default=0) + 1
        return candidate

    def undo(self) -> dict[str, Any] | None:
        if not self.operations:
            return None
        self.selected_ids = []
        return self.operations.pop()

    def reset(self) -> None:
        self.operations = []
        self.selected_ids = []

    # ------------------------------------------------------------- final set
    def build_final(
        self,
        *,
        final_record_builder: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
        """Return the final membership (original_index, tree_id, xyz) after edits.

        Deleted trees become unassigned (dropped from the tree rows); merged
        trees union by exact original_index.  XYZ is unchanged (the editor only
        relabels membership; the coordinates come straight from the immutable
        authoritative science arrays).  The returned ``tree_id`` values are the
        final workflow ids (merged ids and surviving algorithm ids).
        """

        del final_record_builder
        members, merges = _apply_ops(self.original, self.operations)
        original_index = self.original.original_index
        algorithm_tree = self.original.tree_id
        xyz_all = self.original.xyz_local_m

        keep = np.zeros(len(original_index), dtype=bool)
        final_ids = np.full(len(original_index), -1, dtype=np.int32)
        for final_id, source_ids in members.items():
            for source_id in source_ids:
                mask = algorithm_tree == source_id
                keep |= mask
                final_ids[mask] = final_id
        final_index = original_index[keep]
        final_tree = final_ids[keep]
        final_xyz = np.asarray(xyz_all[keep], dtype=np.float64)
        if len(final_index) != len(np.unique(final_index)):
            raise TreeEditorError("MERGE_DUPLICATE_ORIGINAL_INDEX")
        if int(len(final_index)) != int(np.count_nonzero(keep)):
            raise TreeEditorError("FINAL_MEMBERSHIP_COUNT_MISMATCH")
        by_id = {record.tree_id: record for record in self.original.records}
        merged_summary: dict[int, dict[str, Any]] = {}
        for final_id, merge in merges.items():
            if final_id not in members:
                continue
            merged_summary[final_id] = {
                "merged_id": final_id,
                "source_tree_ids": list(merge.source_ids),
                "union_point_count": sum(
                    by_id[tid].point_count for tid in merge.source_ids
                ),
            }
        return final_index, final_tree, final_xyz, merged_summary


def _union_apex(records: list[OriginalTree]) -> tuple[float, float, float] | None:
    """Deterministic merged apex: the lowest-order source tree's apex."""

    ordered = sorted(records, key=lambda record: record.tree_id)
    for record in ordered:
        if record.apex_xy_local_m is not None:
            return record.apex_xy_local_m
    return None


def load_edits_payload(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and return a persisted edit-log operation list."""

    if not isinstance(operations, list):
        raise TreeEditorError("tree_edits.json 的 operations 必须是数组")
    cleaned: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise TreeEditorError("编辑操作必须是对象")
        op_type = operation.get("type")
        if op_type not in {"DELETE", "MERGE"}:
            raise TreeEditorError(f"未知编辑操作类型: {op_type}")
        tree_ids = [int(value) for value in operation.get("tree_ids", [])]
        if not tree_ids:
            raise TreeEditorError("编辑操作缺少 tree_ids")
        item: dict[str, Any] = {"type": op_type, "tree_ids": tree_ids}
        if op_type == "MERGE":
            merged_id = operation.get("merged_id")
            if merged_id is None:
                raise TreeEditorError("MERGE 操作缺少 merged_id")
            item["merged_id"] = int(merged_id)
        cleaned.append(item)
    return cleaned


__all__ = [
    "TREE_EDIT_LOG_SCHEMA",
    "TreeEditorError",
    "OriginalTree",
    "OriginalTrees",
    "build_original_trees",
    "TreeEditSession",
    "load_edits_payload",
]
