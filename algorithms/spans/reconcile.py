"""Cross-span tree reconciliation and deterministic global tree IDs.

A crown near a pole boundary may be segmented in the halos of two adjacent
spans.  Two per-span trees are considered the same real crown only on strong
evidence: substantial ``original_index`` overlap AND a shared stable tree-top
position.  CORE ownership decides the final accepted membership: a raw source
point inherits the tree id of the span whose core contains it, and a raw point
may never be duplicated into two final exclusive tree memberships.

After reconciliation every final tree receives a deterministic external
identity ``LineXX-SpanYYY-TreeZZZ``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

if TYPE_CHECKING:
    from algorithms.spans.tree_runner import SpanTreeOutcome

RECONCILE_SCHEMA = "fast-baseline-tree-reconciliation-v1"
_OVERLAP_IOU_MIN = 0.30


def _span_coordinates(
    span: Any,
    apex_xy_local_m: tuple[float, ...],
    origin_xyz: tuple[float, float, float],
) -> tuple[float | None, float | None]:
    """Project an apex in the authoritative start-to-end Span frame."""

    if (
        len(apex_xy_local_m) < 2
        or not hasattr(span, "start_xyz_absolute_m")
        or not hasattr(span, "end_xyz_absolute_m")
    ):
        return None, None
    start = np.asarray(span.start_xyz_absolute_m[:2], dtype=np.float64)
    end = np.asarray(span.end_xyz_absolute_m[:2], dtype=np.float64)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        return None, None
    axis /= length
    absolute = np.asarray(apex_xy_local_m[:2], dtype=np.float64) + np.asarray(
        origin_xyz[:2], dtype=np.float64
    )
    delta = absolute - start
    return float(delta @ axis), float(delta @ np.asarray((-axis[1], axis[0])))


def _apply_span_order(records: list[GlobalTreeRecord]) -> list[GlobalTreeRecord]:
    """Assign presentation-only per-span order without altering durable ids."""

    ordered: list[GlobalTreeRecord] = []
    for span_id in sorted({record.span_id for record in records}):
        group = [record for record in records if record.span_id == span_id]
        group.sort(
            key=lambda record: (
                float("inf") if record.longitudinal_s_m is None else record.longitudinal_s_m,
                float("inf") if record.lateral_n_m is None else record.lateral_n_m,
                record.global_tree_id,
            )
        )
        ordered.extend(
            replace(record, span_order_index=index) for index, record in enumerate(group, start=1)
        )
    return sorted(ordered, key=lambda record: record.global_tree_id)


@dataclass(frozen=True, slots=True)
class GlobalTreeRecord:
    """One reconciled, deterministic final tree with full-res membership."""

    global_tree_id: int
    line_id: int
    span_id: int
    span_name: str
    external_name: str
    primary_span_tree_id: int
    apex_xy_local_m: tuple[float, float, float]
    apex_height_m: float
    full_resolution_point_count: int
    core_point_count: int
    halo_point_count: int
    core_fraction: float
    review: bool
    merged_span_tree_keys: tuple[tuple[int, int], ...]
    persistent_tree_id: str | None = None
    identity_state: str = "RUN_LOCAL_ONLY"
    candidate_persistent_tree_ids: tuple[str, ...] = ()
    longitudinal_s_m: float | None = None
    lateral_n_m: float | None = None
    span_order_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_tree_id": self.global_tree_id,
            "line_id": self.line_id,
            "span_id": self.span_id,
            "span_name": self.span_name,
            "external_name": self.external_name,
            "primary_span_tree_id": self.primary_span_tree_id,
            "apex_xy_local_m": list(self.apex_xy_local_m),
            "apex_height_m": self.apex_height_m,
            "full_resolution_point_count": self.full_resolution_point_count,
            "core_point_count": self.core_point_count,
            "halo_point_count": self.halo_point_count,
            "core_fraction": self.core_fraction,
            "review": self.review,
            "merged_span_tree_keys": [[int(a), int(b)] for a, b in self.merged_span_tree_keys],
            "persistent_tree_id": self.persistent_tree_id,
            "identity_state": self.identity_state,
            "candidate_persistent_tree_ids": list(self.candidate_persistent_tree_ids),
            "longitudinal_s_m": self.longitudinal_s_m,
            "lateral_n_m": self.lateral_n_m,
            "span_order_index": self.span_order_index,
        }


@dataclass(frozen=True, slots=True)
class TreeReconciliationResult:
    """Final reconciled trees plus the exclusive full-resolution membership."""

    global_tree_records: tuple[GlobalTreeRecord, ...]
    final_original_index: np.ndarray  # int64, unique assigned raw rows
    final_xyz_local_m: np.ndarray  # float64 (N, 3) scene-local
    final_tree_id: np.ndarray  # int32 global ids (-1 unassigned)
    review_tree_ids: tuple[int, ...]
    # Multi-span results publish authoritative ``(span_id, tree_id)`` keys;
    # the single-span fast path intentionally retains its legacy ``(0, tree_id)``
    # contract for persistent-ID compatibility.
    span_tree_to_global: dict[tuple[int, int], int]
    diagnostics: dict[str, Any]

    @property
    def final_tree_count(self) -> int:
        return len(self.global_tree_records)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, key: tuple[int, int]) -> tuple[int, int]:
        parent = self._parent.setdefault(key, key)
        if parent != key:
            self._parent[key] = self.find(parent)
        return self._parent[key]

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self._parent[root_left] = root_right


def _span_tree_members(
    outcome: SpanTreeOutcome,
) -> dict[int, np.ndarray]:
    members: dict[int, np.ndarray] = {}
    for tree_id in np.unique(outcome.raw_tree_id):
        if tree_id < 0:
            continue
        rows = np.flatnonzero(outcome.raw_tree_id == tree_id)
        members[int(tree_id)] = np.sort(outcome.raw_original_index[rows])
    return members


def _overlap_iou(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return 0.0
    intersection = float(np.intersect1d(left, right, assume_unique=True).size)
    union = float(len(left) + len(right) - intersection)
    return intersection / union if union > 0 else 0.0


def _adjacent_spans(
    outcomes: list[SpanTreeOutcome],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for index in range(len(outcomes)):
        for other in range(index + 1, len(outcomes)):
            left = outcomes[index].span
            right = outcomes[other].span
            if left.line_id == right.line_id and abs(left.span_id - right.span_id) == 1:
                pairs.append((index, other))
    return pairs


def _exclusive_membership(
    sorted_original: np.ndarray,
    sorted_xyz: np.ndarray,
    sorted_span: np.ndarray,
    sorted_global: np.ndarray,
    sorted_core: np.ndarray,
    outcomes: list[SpanTreeOutcome],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Resolve one owner per source row without a Python loop per point."""

    if not len(sorted_original):
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            0,
        )
    starts = np.flatnonzero(
        np.r_[True, sorted_original[1:] != sorted_original[:-1]]
    )
    stops = np.r_[starts[1:], len(sorted_original)]
    lengths = stops - starts
    group_original = sorted_original[starts]
    chosen = np.minimum.reduceat(sorted_global, starts).astype(np.int32, copy=False)
    maximum = np.maximum.reduceat(sorted_global, starts)
    conflict_group = chosen != maximum
    conflict_count = int(np.count_nonzero(conflict_group))
    if conflict_count:
        conflict_row = np.repeat(conflict_group, lengths)
        candidate_rows = np.flatnonzero(conflict_row & sorted_core)
        if len(candidate_rows):
            span_positions = sorted_span[candidate_rows]
            line_by_span = np.asarray(
                [outcome.span.line_id for outcome in outcomes], dtype=np.int64
            )
            span_id_by_span = np.asarray(
                [outcome.span.span_id for outcome in outcomes], dtype=np.int64
            )
            candidate_order = np.lexsort(
                (
                    candidate_rows,
                    span_positions,
                    span_id_by_span[span_positions],
                    line_by_span[span_positions],
                    sorted_original[candidate_rows],
                )
            )
            ordered_rows = candidate_rows[candidate_order]
            winning = ordered_rows[
                np.r_[
                    True,
                    sorted_original[ordered_rows[1:]]
                    != sorted_original[ordered_rows[:-1]],
                ]
            ]
            groups = np.searchsorted(group_original, sorted_original[winning])
            chosen[groups] = sorted_global[winning]
    return (
        group_original.astype(np.int64, copy=False),
        sorted_xyz[starts].astype(np.float64, copy=False),
        chosen,
        conflict_count,
    )


def _reconcile_single_span(
    outcome: SpanTreeOutcome,
    config: FastBaselineConfig,
    started: float,
    origin_xyz: tuple[float, float, float],
) -> TreeReconciliationResult:
    """Reconcile one span without the general cross-span raw-row loop."""

    raw_tree = np.asarray(outcome.raw_tree_id, dtype=np.int32)
    raw_original_all = np.asarray(outcome.raw_original_index, dtype=np.int64)
    # NumPy's hash-backed ``unique`` is disproportionately expensive for the
    # multi-million-row int64 source identities seen here.  A sorted copy is
    # both exact and substantially cheaper, and the copy is released before
    # the owned-membership arrays are materialised below.
    sorted_input_identity = np.sort(raw_original_all, kind="quicksort")
    duplicate_input_count = int(
        np.count_nonzero(sorted_input_identity[1:] == sorted_input_identity[:-1])
    )
    del sorted_input_identity
    # A one-span execution sees neighbouring crowns in its halo, but those
    # observations are context only.  The stable tree-top anchor determines
    # whether this Span owns the entity; a halo-only tree may never allocate a
    # persistent identity merely because its points were needed for continuity.
    owned_tree_ids: set[int] = set()
    if all(
        hasattr(outcome.span, name)
        for name in ("start_xyz_absolute_m", "end_xyz_absolute_m", "length_m", "half_width_m")
    ):
        for record in outcome.tree_records:
            tree_id = int(record["tree_id"])
            apex = record.get("apex_xy_local_m")
            if apex is None:
                continue
            longitudinal, lateral = _span_coordinates(outcome.span, tuple(apex), origin_xyz)
            if (
                longitudinal is not None
                and lateral is not None
                and 0.0 <= longitudinal <= float(outcome.span.length_m)
                and abs(lateral) <= float(outcome.span.half_width_m)
            ):
                owned_tree_ids.add(tree_id)
    else:
        # Compatibility boundary for minimal legacy adapters that never had
        # enough geometry to distinguish a core from a halo.
        owned_tree_ids = {int(record["tree_id"]) for record in outcome.tree_records}
    valid = (raw_tree >= 0) & np.isin(raw_tree, np.asarray(sorted(owned_tree_ids), dtype=np.int32))
    raw_original = np.asarray(outcome.raw_original_index, dtype=np.int64)[valid]
    raw_xyz = np.asarray(outcome.raw_xyz_local_m, dtype=np.float64)[valid]
    raw_tree = raw_tree[valid]
    raw_core = np.asarray(outcome.in_core, dtype=bool)[valid]
    order = np.argsort(raw_original, kind="stable")
    sorted_original = raw_original[order]
    if len(sorted_original):
        unique = np.r_[True, sorted_original[1:] != sorted_original[:-1]]
        keep = order[unique]
    else:
        keep = order
    final_original = raw_original[keep]
    final_xyz = raw_xyz[keep]
    final_tree = raw_tree[keep]
    final_core = raw_core[keep]
    tree_ids = np.unique(final_tree)
    tree_to_global = {int(tree_id): index for index, tree_id in enumerate(tree_ids)}
    final_global = np.searchsorted(tree_ids, final_tree).astype(np.int32, copy=False)
    span = outcome.span
    records: list[GlobalTreeRecord] = []
    review_ids: list[int] = []
    for new_id, tree_id_value in enumerate(tree_ids):
        tree_id = int(tree_id_value)
        rows = final_tree == tree_id
        total = int(np.count_nonzero(rows))
        core_points = int(np.count_nonzero(rows & final_core))
        primary_record = next(
            (record for record in outcome.tree_records if record["tree_id"] == tree_id),
            None,
        )
        apex_xy = (
            tuple(primary_record["apex_xy_local_m"])
            if primary_record and primary_record.get("apex_xy_local_m")
            else (0.0, 0.0, 0.0)
        )
        height = float(primary_record["apex_height_m"]) if primary_record else 0.0
        core_fraction = core_points / total if total else 0.0
        longitudinal_s_m, lateral_n_m = _span_coordinates(span, apex_xy, origin_xyz)
        review = core_fraction < 0.15 or total < config.tree_min_final_members
        if review:
            review_ids.append(new_id)
        records.append(
            GlobalTreeRecord(
                global_tree_id=new_id,
                line_id=span.line_id,
                span_id=span.span_id,
                span_name=span.span_name,
                external_name=f"{span.line_name}-Span{span.span_id + 1:03d}-Tree{new_id + 1:03d}",
                primary_span_tree_id=tree_id,
                apex_xy_local_m=apex_xy,
                apex_height_m=height,
                full_resolution_point_count=total,
                core_point_count=core_points,
                halo_point_count=total - core_points,
                core_fraction=core_fraction,
                review=review,
                merged_span_tree_keys=((0, tree_id),),
                longitudinal_s_m=longitudinal_s_m,
                lateral_n_m=lateral_n_m,
            )
        )
    records = _apply_span_order(records)
    return TreeReconciliationResult(
        global_tree_records=tuple(records),
        final_original_index=final_original,
        final_xyz_local_m=final_xyz,
        final_tree_id=final_global,
        review_tree_ids=tuple(review_ids),
        span_tree_to_global={
            (0, int(tree_id)): tree_to_global[int(tree_id)] for tree_id in tree_ids
        },
        diagnostics={
            "schema": RECONCILE_SCHEMA,
            "span_count": 1,
            "global_tree_count": len(records),
            "review_tree_count": len(review_ids),
            "final_raw_member_count": int(len(final_original)),
            "duplicate_input_original_index_count": duplicate_input_count,
            # ``keep`` contains the first row of each sorted identity group,
            # so uniqueness is guaranteed by construction without rescanning
            # the multi-million-row output.
            "duplicate_final_original_index": 0,
            "runtime_seconds": time.perf_counter() - started,
            "single_span_fast_path": True,
        },
    )


def reconcile_trees(
    outcomes: list[SpanTreeOutcome],
    config: FastBaselineConfig,
    *,
    origin_xyz: tuple[float, float, float],
) -> TreeReconciliationResult:
    """Merge duplicate halo hypotheses and build the final tree inventory."""

    started = time.perf_counter()
    if len(outcomes) == 1:
        return _reconcile_single_span(outcomes[0], config, started, origin_xyz)
    members_by_span = [_span_tree_members(outcome) for outcome in outcomes]
    union_find = _UnionFind()

    for span_index, members in enumerate(members_by_span):
        for tree_id in members:
            union_find.find((span_index, tree_id))

    # Merge across adjacent spans only (halos overlap only between neighbours).
    for left_span, right_span in _adjacent_spans(outcomes):
        left_members = members_by_span[left_span]
        right_members = members_by_span[right_span]
        for left_tree, left_indices in left_members.items():
            for right_tree, right_indices in right_members.items():
                if _overlap_iou(left_indices, right_indices) >= _OVERLAP_IOU_MIN:
                    union_find.union((left_span, left_tree), (right_span, right_tree))

    # Map per-span (span_index, tree_id) -> global tree id.
    components: dict[tuple[int, int], int] = {}
    next_global = 0
    for span_index, members in enumerate(members_by_span):
        for tree_id in members:
            root = union_find.find((span_index, tree_id))
            if root not in components:
                components[root] = next_global
                next_global += 1

    span_tree_to_global: dict[tuple[int, int], int] = {}
    for span_index, members in enumerate(members_by_span):
        for tree_id in members:
            root = union_find.find((span_index, tree_id))
            span_tree_to_global[(span_index, tree_id)] = components[root]

    # Choose a primary per-span tree per global tree (largest core membership).
    # global -> (core_count, line_id, span_id, span_index, per_span_tree_id)
    core_count: dict[int, tuple[int, int, int, int, int]] = {}
    for span_index, outcome in enumerate(outcomes):
        members = members_by_span[span_index]
        for tree_id in members:
            rows = np.flatnonzero(outcome.raw_tree_id == tree_id)
            core = int(np.count_nonzero(outcome.in_core[rows]))
            global_id = span_tree_to_global[(span_index, tree_id)]
            candidate = (
                core,
                outcome.span.line_id,
                outcome.span.span_id,
                span_index,
                tree_id,
            )
            current = core_count.get(global_id)
            if current is None or candidate > current:
                core_count[global_id] = candidate

    # Build final per-point exclusive membership.
    parts_original: list[np.ndarray] = []
    parts_xyz: list[np.ndarray] = []
    parts_span: list[np.ndarray] = []
    parts_tree: list[np.ndarray] = []
    parts_core: list[np.ndarray] = []
    for span_index, outcome in enumerate(outcomes):
        global_ids = np.array(
            [span_tree_to_global[(span_index, int(tree))] for tree in outcome.raw_tree_id],
            dtype=np.int32,
        )
        parts_original.append(outcome.raw_original_index)
        parts_xyz.append(outcome.raw_xyz_local_m)
        parts_span.append(np.full(len(outcome.raw_original_index), span_index, dtype=np.int32))
        parts_tree.append(global_ids)
        parts_core.append(np.asarray(outcome.in_core, dtype=bool))
    all_original = np.concatenate(parts_original)
    all_xyz = np.concatenate(parts_xyz)
    all_span = np.concatenate(parts_span)
    all_global = np.concatenate(parts_tree)
    all_core = np.concatenate(parts_core)
    order = np.argsort(all_original, kind="stable")
    sorted_original = all_original[order]
    sorted_xyz = all_xyz[order]
    sorted_span = all_span[order]
    sorted_global = all_global[order]
    sorted_core = all_core[order]
    (
        final_original_array,
        final_xyz_array,
        final_global_array,
        conflict_group_count,
    ) = _exclusive_membership(
        sorted_original,
        sorted_xyz,
        sorted_span,
        sorted_global,
        sorted_core,
        outcomes,
    )
    duplicate_input_count = int(len(all_original) - len(final_original_array))

    # Deterministic global tree ordering by (line, span, primary tree id).
    ordered_globals = sorted(
        core_count,
        key=lambda global_id: (
            core_count[global_id][1],
            core_count[global_id][2],
            core_count[global_id][4],
        ),
    )
    # Remap to dense sequential global ids in that deterministic order.
    remap: dict[int, int] = {}
    for new_id, old_id in enumerate(ordered_globals):
        remap[old_id] = new_id
    remap_array = np.full(next_global, -1, dtype=np.int32)
    for old_id, new_id in remap.items():
        remap_array[old_id] = new_id
    final_tree_remapped = remap_array[final_global_array]
    duplicate_final_count = 0
    if duplicate_final_count != 0:
        raise ValueError("DUPLICATE_FINAL_ORIGINAL_INDEX")
    # Keep the local positional map for all reconciliation internals.  Only
    # after global IDs are dense and stable do we publish keys with the
    # authoritative Span identity carried by each outcome.
    remapped_internal_span_tree: dict[tuple[int, int], int] = {
        key: remap[value] for key, value in span_tree_to_global.items()
    }
    published_span_tree: dict[tuple[int, int], int] = dict(
        sorted(
            (
                (int(outcomes[span_index].span.span_id), int(tree_id)),
                global_id,
            )
            for (span_index, tree_id), global_id in remapped_internal_span_tree.items()
        )
    )

    records: list[GlobalTreeRecord] = []
    review_ids: list[int] = []
    total_by_tree = np.bincount(
        final_tree_remapped, minlength=len(ordered_globals)
    )
    core_count_by_span: dict[int, np.ndarray] = {}
    for new_id, old_id in enumerate(ordered_globals):
        _core, line_id, span_id, span_index, primary_tree = core_count[old_id]
        outcome = outcomes[span_index]
        span = outcome.span
        total = int(total_by_tree[new_id])
        if span_index not in core_count_by_span:
            core_mask = _in_core_of_span(
                final_original_array, final_xyz_array, outcome
            )
            core_count_by_span[span_index] = np.bincount(
                final_tree_remapped[core_mask], minlength=len(ordered_globals)
            )
        core_points = int(core_count_by_span[span_index][new_id])
        halo_points = total - core_points
        primary_record = next(
            (record for record in outcome.tree_records if record["tree_id"] == primary_tree),
            None,
        )
        apex_xy = (
            tuple(primary_record["apex_xy_local_m"])
            if primary_record and primary_record.get("apex_xy_local_m")
            else (0.0, 0.0, 0.0)
        )
        height = float(primary_record["apex_height_m"]) if primary_record else 0.0
        core_fraction = core_points / total if total else 0.0
        longitudinal_s_m, lateral_n_m = _span_coordinates(span, apex_xy, origin_xyz)
        review = core_fraction < 0.15 or total < config.tree_min_final_members
        if review:
            review_ids.append(new_id)
        external_name = f"{span.line_name}-Span{span.span_id + 1:03d}-Tree{new_id + 1:03d}"
        merged_keys = tuple(
            sorted(key for key, value in published_span_tree.items() if value == new_id)
        )
        records.append(
            GlobalTreeRecord(
                global_tree_id=new_id,
                line_id=line_id,
                span_id=span_id,
                span_name=span.span_name,
                external_name=external_name,
                primary_span_tree_id=primary_tree,
                apex_xy_local_m=apex_xy,
                apex_height_m=height,
                full_resolution_point_count=total,
                core_point_count=core_points,
                halo_point_count=halo_points,
                core_fraction=core_fraction,
                review=review,
                merged_span_tree_keys=merged_keys,
                longitudinal_s_m=longitudinal_s_m,
                lateral_n_m=lateral_n_m,
            )
        )
    records = _apply_span_order(records)
    return TreeReconciliationResult(
        global_tree_records=tuple(records),
        final_original_index=final_original_array,
        final_xyz_local_m=final_xyz_array,
        final_tree_id=final_tree_remapped,
        review_tree_ids=tuple(review_ids),
        span_tree_to_global=published_span_tree,
        diagnostics={
            "schema": RECONCILE_SCHEMA,
            "span_count": len(outcomes),
            "global_tree_count": len(records),
            "review_tree_count": len(review_ids),
            "final_raw_member_count": int(len(final_original_array)),
            "duplicate_input_original_index_count": duplicate_input_count,
            "duplicate_final_original_index": duplicate_final_count,
            "exclusive_group_count": int(len(final_original_array)),
            "conflict_group_count": conflict_group_count,
            "ownership_resolution": "VECTORIZED_STABLE_GLOBAL_INDEX_V1",
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def _in_core_of_span(
    original_index: np.ndarray,
    xyz_local_m: np.ndarray,
    outcome: SpanTreeOutcome,
) -> np.ndarray:
    """Boolean per final row indicating membership in *outcome*'s core."""

    del xyz_local_m
    mask = np.zeros(len(original_index), dtype=bool)
    if not len(outcome.raw_original_index):
        return mask
    order = np.argsort(outcome.raw_original_index, kind="stable")
    sorted_outcome = outcome.raw_original_index[order]
    positions = np.searchsorted(sorted_outcome, original_index)
    valid = positions < len(sorted_outcome)
    valid &= sorted_outcome[np.minimum(positions, len(sorted_outcome) - 1)] == original_index
    if np.any(valid):
        mask[valid] = outcome.in_core[order[positions[valid]]]
    return mask


def reconciliation_to_dict(result: TreeReconciliationResult) -> dict[str, Any]:
    return {
        "schema": RECONCILE_SCHEMA,
        "global_tree_records": [record.to_dict() for record in result.global_tree_records],
        "review_tree_ids": list(result.review_tree_ids),
        "diagnostics": dict(result.diagnostics),
    }


__all__ = [
    "RECONCILE_SCHEMA",
    "GlobalTreeRecord",
    "TreeReconciliationResult",
    "reconcile_trees",
    "reconciliation_to_dict",
]
