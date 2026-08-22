"""Human tree editing model (DELETE / MERGE / REMOVE_POINTS / UNDO).

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

import base64
import time
import uuid
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

TREE_EDIT_LOG_SCHEMA = "folder-to-excel-tree-edits-v1"
_LARGE_REMOVAL_FRACTION = 1.0 / 3.0
_WHOLE_TREE_FRACTION = 0.95
_AUTO_COMPONENT_MAX_FRACTION = 0.75


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
        elif op_type == "REMOVE_POINTS":
            # Point removals do not change the tree entity graph.  They are
            # applied by exact original_index later in ``build_final``.
            continue
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
        self._previews: dict[str, dict[str, Any]] = {}
        self._source_rows_cache: dict[int, np.ndarray] = {}

    @property
    def revision(self) -> int:
        """Monotonic committed membership revision for stale-preview checks."""

        return len(self.operations)

    def _removed_original_indices(self) -> np.ndarray:
        chunks = [
            _decode_indices(operation["removed_indices_b64"])
            for operation in self.operations
            if operation.get("type") == "REMOVE_POINTS"
        ]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(chunks))

    def _source_rows(self, source_id: int) -> np.ndarray:
        rows = self._source_rows_cache.get(source_id)
        if rows is None:
            rows = np.flatnonzero(self.original.tree_id == source_id)
            self._source_rows_cache[source_id] = rows
        return rows

    def current_membership(self, tree_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return authoritative full-resolution (original_index, XYZ) for one tree."""

        members, _merges = _apply_ops(self.original, self.operations)
        source_ids = members.get(int(tree_id))
        if not source_ids:
            raise TreeEditorError(f"当前不存在单木 id: {tree_id}")
        chunks: list[np.ndarray] = []
        for source_id in source_ids:
            # A source-tree membership is discovered once, then all click and
            # lasso work stays on those rows.  This scans the compact final-tree
            # artifact, never the raw LAS/LAZ.
            chunks.append(self._source_rows(source_id))
        rows = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        indices = self.original.original_index[rows]
        removed = self._removed_original_indices()
        keep = np.ones(len(rows), dtype=bool) if not len(removed) else ~np.isin(indices, removed)
        if not np.any(keep):
            raise TreeEditorError(f"当前不存在单木 id: {tree_id}")
        return indices[keep], np.asarray(self.original.xyz_local_m[rows[keep]], dtype=np.float64)

    def edit_provenance_by_current_tree(self) -> dict[int, tuple[str, ...]]:
        """Map current tree ids to human edit methods affecting their members."""

        members, _merges = _apply_ops(self.original, self.operations)
        provenance: dict[int, tuple[str, ...]] = {}
        for final_id, source_ids in members.items():
            methods: set[str] = set()
            source_mask = np.isin(self.original.tree_id, source_ids)
            source_indices = self.original.original_index[source_mask]
            if len(source_ids) > 1:
                methods.add("MERGE")
            for operation in self.operations:
                if operation.get("type") != "REMOVE_POINTS":
                    continue
                removed = _decode_indices(operation["removed_indices_b64"])
                if np.any(np.isin(removed, source_indices)):
                    methods.add(str(operation.get("selection_method") or "TRIM"))
            provenance[int(final_id)] = tuple(sorted(methods))
        return provenance

    # ------------------------------------------------------------------ state
    def current_tree_ids(self) -> list[int]:
        members, _merges = _apply_ops(self.original, self.operations)
        return sorted(members)

    def snapshot(self) -> dict[str, Any]:
        """Project the current effective trees (records + ids) for the editor UI."""

        members, merges = _apply_ops(self.original, self.operations)
        by_id = {record.tree_id: record for record in self.original.records}
        trees: list[dict[str, Any]] = []
        removed = self._removed_original_indices()
        removed_by_source: dict[int, int] = {}
        if len(removed):
            # Count every removed exact member once.  The old per-tree np.isin
            # loop rescanned the compact multi-million-row editor artifact for
            # every tree, making a single historical trim slow all later UI
            # refreshes.  This remains exact and never touches the raw source.
            removed_rows = np.isin(self.original.original_index, removed)
            source_ids, source_counts = np.unique(
                self.original.tree_id[removed_rows], return_counts=True
            )
            removed_by_source = {
                int(source_id): int(count)
                for source_id, count in zip(source_ids, source_counts, strict=True)
            }
        counts: dict[int, int] = {}
        for final_id, source_ids in members.items():
            total = sum(by_id[source_id].point_count for source_id in source_ids)
            total -= sum(removed_by_source.get(source_id, 0) for source_id in source_ids)
            counts[final_id] = total
        for final_id in sorted(members):
            source_ids = members[final_id]
            if len(source_ids) == 1:
                record = by_id[source_ids[0]]
                trees.append(
                    {
                        "tree_id": final_id,
                        "external_name": record.external_name,
                        "point_count": counts.get(final_id, 0),
                        "apex_xy_local_m": record.apex_xy_local_m,
                        "apex_height_m": record.apex_height_m,
                        "merged": False,
                        "source_tree_ids": source_ids,
                    }
                )
            else:
                merged_id = final_id
                source_count = counts.get(final_id, 0)
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
            # Never send compressed (potentially very large) exact membership
            # payloads to the UI.  Persistence still receives them via
            # ``edits_payload``.
            "operations": [_public_operation(op) for op in self.operations],
        }

    def display_sample(self, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a bounded current-membership sample for interactive display only.

        Confirm/final publication still use :meth:`build_final`.  This method
        deliberately samples the compact editor membership without constructing
        multi-million-row final arrays merely to render a few thousand points.
        """

        maximum = max(0, int(limit))
        if maximum == 0 or not len(self.original.original_index):
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.float64),
            )
        members, _merges = _apply_ops(self.original, self.operations)
        source_to_final = {
            int(source_id): int(final_id)
            for final_id, source_ids in members.items()
            for source_id in source_ids
        }
        candidate_count = min(len(self.original.original_index), maximum * 10)
        candidate_rows = np.linspace(
            0, len(self.original.original_index) - 1, candidate_count, dtype=np.int64
        )
        source_ids = self.original.tree_id[candidate_rows]
        final_ids = np.asarray(
            [source_to_final.get(int(source_id), -1) for source_id in source_ids],
            dtype=np.int32,
        )
        keep = final_ids >= 0
        removed = self._removed_original_indices()
        if len(removed):
            keep &= ~np.isin(self.original.original_index[candidate_rows], removed)
        candidate_rows = candidate_rows[keep]
        final_ids = final_ids[keep]
        if len(candidate_rows) > maximum:
            sampled = np.linspace(0, len(candidate_rows) - 1, maximum, dtype=np.int64)
            candidate_rows = candidate_rows[sampled]
            final_ids = final_ids[sampled]
        return (
            np.asarray(self.original.original_index[candidate_rows], dtype=np.int64),
            final_ids,
            np.asarray(self.original.xyz_local_m[candidate_rows], dtype=np.float64),
        )

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

    def preview_trim_component(
        self,
        tree_id: int,
        seed_xyz: list[float] | tuple[float, float, float] | np.ndarray,
        *,
        binding: str = "",
    ) -> dict[str, Any]:
        """Preview the adaptive 3-D component containing a clicked seed."""

        started = time.perf_counter()
        indices, xyz = self.current_membership(int(tree_id))
        seed = np.asarray(seed_xyz, dtype=np.float64)
        if seed.shape != (3,) or not np.all(np.isfinite(seed)):
            raise TreeEditorError("点击位置无效")
        if len(xyz) < 2:
            raise TreeEditorError("当前单木点成员不足，无法分析团块")
        tree = cKDTree(xyz, copy_data=False)
        seed_distance, seed_row = tree.query(seed, k=1, workers=1)
        spacing = _adaptive_spacing(xyz, tree)
        radius = _adaptive_component_radius(xyz, spacing)
        if float(seed_distance) > max(radius * 4.0, np.finfo(float).eps * 100.0):
            raise TreeEditorError("点击位置未命中当前单木，请重新点击")
        component_rows = _seed_component(tree, int(seed_row), radius)
        fraction = len(component_rows) / len(indices)
        if fraction >= _AUTO_COMPONENT_MAX_FRACTION:
            raise TreeEditorError(
                "该区域与树体主体连接较紧密，自动团块剔除可能范围过大，"
                "建议使用‘套索剔除’手动选择。"
            )
        return self._create_preview(
            tree_id=int(tree_id),
            removed_indices=indices[component_rows],
            method="COMPONENT",
            binding=binding,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            extra={
                "adaptive_spacing_m": spacing,
                "connection_radius_m": radius,
                "seed_original_index": int(indices[int(seed_row)]),
            },
        )

    def preview_trim_lasso(
        self,
        tree_id: int,
        view_projection: list[list[float]] | np.ndarray,
        polygon_ndc: list[list[float]] | np.ndarray,
        *,
        binding: str = "",
    ) -> dict[str, Any]:
        """Preview current-tree full-res points inside a screen-space polygon."""

        started = time.perf_counter()
        indices, xyz = self.current_membership(int(tree_id))
        matrix = np.asarray(view_projection, dtype=np.float64)
        polygon = np.asarray(polygon_ndc, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise TreeEditorError("套索相机矩阵无效")
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise TreeEditorError("套索至少需要三个有效顶点")
        if not np.all(np.isfinite(polygon)) or np.any(np.abs(polygon) > 2.0):
            raise TreeEditorError("套索屏幕坐标无效")
        homogeneous = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float64)))
        clip = homogeneous @ matrix.T
        valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > np.finfo(float).eps)
        projected = np.full((len(xyz), 2), np.nan, dtype=np.float64)
        projected[valid] = clip[valid, :2] / clip[valid, 3, None]
        inside = valid & _points_in_polygon(projected, polygon)
        selected = np.flatnonzero(inside)
        if not len(selected):
            raise TreeEditorError("套索内没有当前单木的点")
        return self._create_preview(
            tree_id=int(tree_id),
            removed_indices=indices[selected],
            method="LASSO",
            binding=binding,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _create_preview(
        self,
        *,
        tree_id: int,
        removed_indices: np.ndarray,
        method: str,
        binding: str,
        elapsed_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = np.unique(np.asarray(removed_indices, dtype=np.int64))
        current_indices, _xyz = self.current_membership(tree_id)
        if not len(values) or not np.all(np.isin(values, current_indices)):
            raise TreeEditorError("预览点成员无效")
        fraction = len(values) / len(current_indices)
        if fraction >= _WHOLE_TREE_FRACTION:
            raise TreeEditorError("本次选择接近整棵单木；如需移除整棵树，请使用‘删除单木’。")
        preview_id = uuid.uuid4().hex
        preview = {
            "preview_id": preview_id,
            "tree_id": tree_id,
            "revision": self.revision,
            "binding": str(binding),
            "method": method,
            "removed_indices": values,
            "point_count": int(len(values)),
            "tree_point_count": int(len(current_indices)),
            "fraction": float(fraction),
            "large_removal_warning": bool(fraction > _LARGE_REMOVAL_FRACTION),
            "selection_ms": round(float(elapsed_ms), 3),
        }
        if extra:
            preview.update(extra)
        self._previews = {preview_id: preview}
        return _public_preview(preview)

    def preview_indices(self, preview_id: str) -> np.ndarray:
        preview = self._require_preview(preview_id)
        return np.asarray(preview["removed_indices"], dtype=np.int64)

    def preview_xyz(self, preview_id: str) -> np.ndarray:
        preview = self._require_preview(preview_id)
        indices, xyz = self.current_membership(int(preview["tree_id"]))
        return xyz[np.isin(indices, preview["removed_indices"])]

    def commit_trim(
        self,
        preview_id: str,
        *,
        binding: str = "",
        minimum_remaining_points: int | None = None,
    ) -> dict[str, Any]:
        preview = self._require_preview(preview_id)
        if preview["revision"] != self.revision or preview["binding"] != str(binding):
            self._previews.clear()
            raise TreeEditorError("预览已过期，请重新选择")
        # Revalidate against the current authoritative membership immediately
        # before commit.  No geometry approximation is used for the edit.
        current, _xyz = self.current_membership(int(preview["tree_id"]))
        removed = np.asarray(preview["removed_indices"], dtype=np.int64)
        if not np.all(np.isin(removed, current)):
            self._previews.clear()
            raise TreeEditorError("预览已过期，请重新选择")
        remaining = len(current) - len(removed)
        if minimum_remaining_points is not None and remaining < int(minimum_remaining_points):
            raise TreeEditorError(
                "剔除后该单木无法形成有效结果，请取消本次操作或删除整棵单木。"
            )
        operation = {
            "type": "REMOVE_POINTS",
            "tree_ids": [int(preview["tree_id"])],
            "selection_method": str(preview["method"]),
            "point_count": int(len(removed)),
            "removed_indices_b64": _encode_indices(removed),
            "sequence": self.revision + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.operations.append(operation)
        self.selected_ids = []
        self._previews.clear()
        return _public_operation(operation)

    def cancel_trim(self, preview_id: str) -> bool:
        if preview_id not in self._previews:
            return False
        del self._previews[preview_id]
        return True

    def _require_preview(self, preview_id: str) -> dict[str, Any]:
        if not isinstance(preview_id, str) or preview_id not in self._previews:
            raise TreeEditorError("预览不存在或已过期")
        return self._previews[preview_id]

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
        self._previews.clear()
        return self.operations.pop()

    def reset(self) -> None:
        self.operations = []
        self.selected_ids = []
        self._previews.clear()

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
        removed = self._removed_original_indices()
        if len(removed):
            keep &= ~np.isin(original_index, removed)
            final_ids[~keep] = -1
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
        if op_type not in {"DELETE", "MERGE", "REMOVE_POINTS"}:
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
        elif op_type == "REMOVE_POINTS":
            encoded = operation.get("removed_indices_b64")
            if not isinstance(encoded, str) or not encoded:
                raise TreeEditorError("REMOVE_POINTS 操作缺少精确点成员")
            decoded = _decode_indices(encoded)
            if not len(decoded) or len(decoded) != len(np.unique(decoded)):
                raise TreeEditorError("REMOVE_POINTS 精确点成员无效")
            item.update(
                {
                    "selection_method": str(operation.get("selection_method") or "UNKNOWN"),
                    "point_count": int(operation.get("point_count", len(decoded))),
                    "removed_indices_b64": encoded,
                    "sequence": int(operation.get("sequence", len(cleaned) + 1)),
                    "timestamp": str(operation.get("timestamp") or ""),
                }
            )
            if item["point_count"] != len(decoded):
                raise TreeEditorError("REMOVE_POINTS 点数与精确成员不一致")
        cleaned.append(item)
    return cleaned


def _encode_indices(values: np.ndarray) -> str:
    ordered = np.unique(np.asarray(values, dtype=np.int64))
    deltas = np.diff(np.concatenate((np.asarray([0], dtype=np.int64), ordered)))
    return base64.b64encode(zlib.compress(deltas.astype("<i8").tobytes(), level=9)).decode("ascii")


def _decode_indices(encoded: str) -> np.ndarray:
    try:
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
        if len(raw) % 8:
            raise ValueError("unaligned")
        deltas = np.frombuffer(raw, dtype="<i8")
        values = np.cumsum(deltas, dtype=np.int64)
    except (ValueError, TypeError, zlib.error) as exc:
        raise TreeEditorError("REMOVE_POINTS 精确点成员编码无效") from exc
    if len(values) and (np.any(np.diff(values) <= 0) or np.any(values < 0)):
        raise TreeEditorError("REMOVE_POINTS 精确点成员编码无效")
    return values


def _public_operation(operation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in operation.items() if key != "removed_indices_b64"}


def _public_preview(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in preview.items()
        if key not in {"removed_indices", "binding"}
    }


def _adaptive_spacing(xyz: np.ndarray, tree: cKDTree) -> float:
    rows = np.linspace(0, len(xyz) - 1, min(len(xyz), 10_000), dtype=np.int64)
    distances = tree.query(xyz[rows], k=min(5, len(xyz)), workers=1)[0]
    values = distances[:, -1] if distances.ndim == 2 else distances
    positive = values[np.isfinite(values) & (values > 0)]
    if not len(positive):
        raise TreeEditorError("当前单木点间距无法可靠估计")
    return float(np.median(positive))


def _adaptive_component_radius(xyz: np.ndarray, spacing: float) -> float:
    diagonal = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    numerical_floor = max(diagonal * 1e-7, np.finfo(float).eps * 100.0)
    structural_ceiling = max(numerical_floor, diagonal * 0.08)
    # The fifth-neighbour median represents a local density scale rather than
    # a scanner-specific metric.  Three such scales bridge ordinary sampling
    # gaps inside a crown while the structural ceiling prevents a sparse
    # bridge from swallowing a spatially separate cluster.
    return float(np.clip(spacing * 3.0, numerical_floor, structural_ceiling))


def _seed_component(tree: cKDTree, seed_row: int, radius: float) -> np.ndarray:
    visited = np.zeros(tree.n, dtype=bool)
    visited[seed_row] = True
    queue = [seed_row]
    cursor = 0
    while cursor < len(queue):
        row = queue[cursor]
        cursor += 1
        for neighbour in tree.query_ball_point(tree.data[row], radius):
            neighbour = int(neighbour)
            if not visited[neighbour]:
                visited[neighbour] = True
                queue.append(neighbour)
    return np.asarray(queue, dtype=np.int64)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        denominator = (yj - yi) if yj != yi else np.finfo(float).eps
        boundary_x = (xj - xi) * (y - yi) / denominator + xi
        inside ^= crosses & (x < boundary_x)
        j = i
    return inside


__all__ = [
    "TREE_EDIT_LOG_SCHEMA",
    "TreeEditorError",
    "OriginalTree",
    "OriginalTrees",
    "build_original_trees",
    "TreeEditSession",
    "load_edits_payload",
]
