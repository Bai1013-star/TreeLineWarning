"""Exact full-resolution multi-tree lasso editing for the Stage-B reviewer.

This class extends the proven :class:`TreeEditSession` only at the product
editing boundary.  A negative ``tree_id`` is an internal sentinel meaning
"apply the screen-space polygon to the current selected tree scope".  The
selection is evaluated against every tree's authoritative full-resolution
``original_index`` membership; viewer samples are never used to decide edits.

A region edit is committed as one ``REMOVE_POINTS`` operation so one UNDO
restores the whole lasso action.  Completely covered trees are allowed to
become empty and disappear from the effective tree list.  Partial remnants
must still satisfy the existing minimum-member gate supplied by Stage-B.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

import algorithms.folder_pipeline.tree_editor as _base
from algorithms.folder_pipeline.tree_editor import TreeEditSession, TreeEditorError

REGION_LASSO_METHOD = "REGION_LASSO"
REGION_LASSO_SENTINEL_TREE_ID = -1
# Projection intermediates are display-geometry math over exact membership,
# not science state.  Keep them bounded so a multi-million-point crown cannot
# create one giant (N, 4) homogeneous matrix during a UI edit.
REGION_LASSO_PROJECTION_CHUNK = 250_000


class RegionTreeEditSession(TreeEditSession):
    """TreeEditSession with one exact multi-tree region-lasso extension."""

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        trees = [
            tree
            for tree in snapshot.get("trees", [])
            if int(tree.get("point_count", 0)) > 0
        ]
        snapshot["trees"] = trees
        snapshot["current_tree_count"] = len(trees)
        visible = {int(tree["tree_id"]) for tree in trees}
        snapshot["selected_ids"] = [
            int(tree_id)
            for tree_id in snapshot.get("selected_ids", [])
            if int(tree_id) in visible
        ]
        return snapshot

    def current_tree_ids(self) -> list[int]:
        return [int(tree["tree_id"]) for tree in self.snapshot().get("trees", [])]

    def preview_trim_lasso(
        self,
        tree_id: int,
        view_projection: list[list[float]] | np.ndarray,
        polygon_ndc: list[list[float]] | np.ndarray,
        *,
        binding: str = "",
    ) -> dict[str, Any]:
        if int(tree_id) >= 0:
            return super().preview_trim_lasso(
                int(tree_id),
                view_projection,
                polygon_ndc,
                binding=binding,
            )
        return self._preview_region_lasso(
            view_projection,
            polygon_ndc,
            binding=binding,
        )

    @staticmethod
    def _selected_rows_chunked(
        xyz: np.ndarray,
        matrix: np.ndarray,
        polygon: np.ndarray,
    ) -> np.ndarray:
        """Project exact XYZ in bounded blocks and return exact selected rows."""

        selected: list[np.ndarray] = []
        for start in range(0, len(xyz), REGION_LASSO_PROJECTION_CHUNK):
            stop = min(len(xyz), start + REGION_LASSO_PROJECTION_CHUNK)
            block = np.asarray(xyz[start:stop], dtype=np.float64)
            homogeneous = np.empty((len(block), 4), dtype=np.float64)
            homogeneous[:, :3] = block
            homogeneous[:, 3] = 1.0
            clip = homogeneous @ matrix.T
            valid = np.isfinite(clip).all(axis=1) & (
                clip[:, 3] > np.finfo(float).eps
            )
            if not np.any(valid):
                continue
            projected = np.full((len(block), 2), np.nan, dtype=np.float64)
            projected[valid] = clip[valid, :2] / clip[valid, 3, None]
            inside = valid & _base._points_in_polygon(projected, polygon)
            local_rows = np.flatnonzero(inside)
            if len(local_rows):
                selected.append(local_rows.astype(np.int64, copy=False) + start)
        if not selected:
            return np.empty(0, dtype=np.int64)
        return selected[0] if len(selected) == 1 else np.concatenate(selected)

    def _preview_region_lasso(
        self,
        view_projection: list[list[float]] | np.ndarray,
        polygon_ndc: list[list[float]] | np.ndarray,
        *,
        binding: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        matrix = np.asarray(view_projection, dtype=np.float64)
        polygon = np.asarray(polygon_ndc, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise TreeEditorError("套索相机矩阵无效")
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise TreeEditorError("套索至少需要三个有效顶点")
        if not np.all(np.isfinite(polygon)) or np.any(np.abs(polygon) > 2.0):
            raise TreeEditorError("套索屏幕坐标无效")

        current_ids = set(self.current_tree_ids())
        requested_scope = [
            int(value)
            for value in self.selected_ids
            if int(value) in current_ids
        ]
        scope = requested_scope or sorted(current_ids)
        if not scope:
            raise TreeEditorError("当前没有可进行区域剔除的单木")

        selected_chunks: list[np.ndarray] = []
        per_tree_removed: dict[int, int] = {}
        per_tree_current: dict[int, int] = {}
        for current_tree_id in scope:
            indices, xyz = self.current_membership(current_tree_id)
            per_tree_current[current_tree_id] = int(len(indices))
            rows = self._selected_rows_chunked(xyz, matrix, polygon)
            if not len(rows):
                continue
            chosen = np.asarray(indices[rows], dtype=np.int64)
            selected_chunks.append(chosen)
            per_tree_removed[current_tree_id] = int(len(chosen))

        if not selected_chunks:
            raise TreeEditorError("套索内没有树木点")
        removed = np.unique(np.concatenate(selected_chunks))
        total_scope_points = int(sum(per_tree_current.values()))
        fully_removed = sorted(
            tree_id
            for tree_id, removed_count in per_tree_removed.items()
            if removed_count == per_tree_current[tree_id]
        )
        preview_id = uuid.uuid4().hex
        fraction = len(removed) / total_scope_points if total_scope_points else 0.0
        preview = {
            "preview_id": preview_id,
            "tree_id": REGION_LASSO_SENTINEL_TREE_ID,
            "tree_ids": sorted(per_tree_removed),
            "scope_tree_ids": list(scope),
            "revision": self.revision,
            "binding": str(binding),
            "method": REGION_LASSO_METHOD,
            "removed_indices": removed,
            "point_count": int(len(removed)),
            "tree_point_count": total_scope_points,
            "fraction": float(fraction),
            "large_removal_warning": bool(
                fraction > _base._LARGE_REMOVAL_FRACTION
            ),
            "selection_ms": round(
                (time.perf_counter() - started) * 1000.0,
                3,
            ),
            "affected_tree_count": len(per_tree_removed),
            "per_tree_removed_counts": {
                str(tree_id): count
                for tree_id, count in sorted(per_tree_removed.items())
            },
            "fully_removed_tree_ids": fully_removed,
            "projection_chunk_points": REGION_LASSO_PROJECTION_CHUNK,
        }
        self._previews = {preview_id: preview}
        return _base._public_preview(preview)

    def preview_xyz(self, preview_id: str) -> np.ndarray:
        preview = self._require_preview(preview_id)
        if preview.get("method") != REGION_LASSO_METHOD:
            return super().preview_xyz(preview_id)
        removed = np.asarray(preview["removed_indices"], dtype=np.int64)
        rows = np.isin(self.original.original_index, removed)
        return np.asarray(self.original.xyz_local_m[rows], dtype=np.float64)

    def commit_trim(
        self,
        preview_id: str,
        *,
        binding: str = "",
        minimum_remaining_points: int | None = None,
    ) -> dict[str, Any]:
        preview = self._require_preview(preview_id)
        if preview.get("method") != REGION_LASSO_METHOD:
            return super().commit_trim(
                preview_id,
                binding=binding,
                minimum_remaining_points=minimum_remaining_points,
            )
        if preview["revision"] != self.revision or preview["binding"] != str(binding):
            self._previews.clear()
            raise TreeEditorError("预览已过期，请重新选择")

        removed = np.asarray(preview["removed_indices"], dtype=np.int64)
        affected_ids = [int(value) for value in preview.get("tree_ids", [])]
        if not len(removed) or not affected_ids:
            self._previews.clear()
            raise TreeEditorError("区域剔除预览为空")

        validated_chunks: list[np.ndarray] = []
        for tree_id in affected_ids:
            current, _xyz = self.current_membership(tree_id)
            tree_removed = removed[np.isin(removed, current)]
            expected = int(
                (preview.get("per_tree_removed_counts") or {}).get(
                    str(tree_id),
                    -1,
                )
            )
            if len(tree_removed) != expected:
                self._previews.clear()
                raise TreeEditorError("预览已过期，请重新选择")
            remaining = int(len(current) - len(tree_removed))
            if (
                minimum_remaining_points is not None
                and 0 < remaining < int(minimum_remaining_points)
            ):
                raise TreeEditorError(
                    f"区域剔除后树 {tree_id} 仅剩 {remaining} 个点，"
                    "不足以形成有效单木；请调整套索后重试。"
                )
            validated_chunks.append(tree_removed)

        validated = np.unique(np.concatenate(validated_chunks))
        if len(validated) != len(removed) or not np.array_equal(validated, removed):
            self._previews.clear()
            raise TreeEditorError("预览已过期，请重新选择")

        operation = {
            "type": "REMOVE_POINTS",
            "tree_ids": affected_ids,
            "selection_method": REGION_LASSO_METHOD,
            "point_count": int(len(removed)),
            "removed_indices_b64": _base._encode_indices(removed),
            "sequence": self.revision + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.operations.append(operation)
        self.selected_ids = []
        self._previews.clear()
        return _base._public_operation(operation)


__all__ = [
    "REGION_LASSO_METHOD",
    "REGION_LASSO_PROJECTION_CHUNK",
    "REGION_LASSO_SENTINEL_TREE_ID",
    "RegionTreeEditSession",
]
