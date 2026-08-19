"""Risk threshold classification and final per-tree result rows.

After the exact per-tree clearances exist, the SAME local application asks for
the 危险净空阈值 (m).  This module applies it: a tree is dangerous exactly when
``clearance_3d_m < threshold_m``, and every full-resolution raw tree point
whose exact distance to the nearest valid wire is below the threshold is a
danger point (never the 0.10 m representatives).  Nothing upstream is recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from algorithms.manual_span.clearance_fast import ManualClearanceResult
from algorithms.manual_span.config import MANUAL_RISK_SCHEMA
from algorithms.manual_span.metrics import TreeMetrics
from algorithms.spans.clearance import SpanTreeWireRecord
from algorithms.spans.reconcile import TreeReconciliationResult


@dataclass(frozen=True, slots=True)
class TreeRiskRecord:
    """One final per-tree row joining metrics, clearance and risk."""

    global_tree_id: int
    external_name: str
    line_id: int
    span_id: int
    span_name: str
    tree_height_m: float | None
    dbh_m: float | None
    dbh_quality: str
    crown_major_width_m: float
    crown_minor_width_m: float
    crown_mean_width_m: float
    crown_area_m2: float
    fullres_point_count: int
    tree_top_xyz_absolute_m: tuple[float, float, float] | None
    tree_centroid_xyz_absolute_m: tuple[float, float, float] | None
    nearest_wire_id: int | None
    nearest_tree_original_index: int | None
    nearest_tree_xyz_absolute_m: tuple[float, float, float] | None
    nearest_wire_xyz_absolute_m: tuple[float, float, float] | None
    horizontal_clearance_m: float | None
    vertical_clearance_m: float | None
    clearance_3d_m: float | None
    risk_threshold_m: float | None
    danger_point_count: int
    dangerous: bool
    quality_notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.global_tree_id,
            "external_name": self.external_name,
            "line_id": self.line_id,
            "span_id": self.span_id,
            "span_name": self.span_name,
            "tree_height_m": self.tree_height_m,
            "dbh_m": self.dbh_m,
            "dbh_quality": self.dbh_quality,
            "crown_major_width_m": self.crown_major_width_m,
            "crown_minor_width_m": self.crown_minor_width_m,
            "crown_mean_width_m": self.crown_mean_width_m,
            "crown_area_m2": self.crown_area_m2,
            "fullres_point_count": self.fullres_point_count,
            "tree_top_xyz_absolute_m": (
                None if self.tree_top_xyz_absolute_m is None else list(self.tree_top_xyz_absolute_m)
            ),
            "tree_centroid_xyz_absolute_m": (
                None
                if self.tree_centroid_xyz_absolute_m is None
                else list(self.tree_centroid_xyz_absolute_m)
            ),
            "nearest_wire_id": self.nearest_wire_id,
            "nearest_tree_original_index": self.nearest_tree_original_index,
            "nearest_tree_xyz_absolute_m": (
                None
                if self.nearest_tree_xyz_absolute_m is None
                else list(self.nearest_tree_xyz_absolute_m)
            ),
            "nearest_wire_xyz_absolute_m": (
                None
                if self.nearest_wire_xyz_absolute_m is None
                else list(self.nearest_wire_xyz_absolute_m)
            ),
            "horizontal_clearance_m": self.horizontal_clearance_m,
            "vertical_clearance_m": self.vertical_clearance_m,
            "clearance_3d_m": self.clearance_3d_m,
            "risk_threshold_m": self.risk_threshold_m,
            "danger_point_count": self.danger_point_count,
            "dangerous": self.dangerous,
            "quality_notes": self.quality_notes,
        }


@dataclass(frozen=True, slots=True)
class RiskResult:
    """Final risk classification outcome (post-threshold)."""

    records: tuple[TreeRiskRecord, ...]
    threshold_m: float
    dangerous_tree_count: int
    danger_point_count: int
    global_minimum_clearance_m: float | None
    danger_original_index: np.ndarray  # int64
    danger_xyz_local_m: np.ndarray  # float32 (M,3)
    danger_distance_m: np.ndarray  # float64 (M,)
    danger_wire_id: np.ndarray  # int32 (M,)
    danger_nearest_wire_xyz_local_m: np.ndarray  # float32 (M,3)


def _to_absolute(xyz_local: tuple[float, float, float] | None, origin: np.ndarray):
    if xyz_local is None:
        return None
    return tuple(float(value) for value in np.asarray(xyz_local) + origin)


def apply_risk_threshold(
    clearance: ManualClearanceResult,
    metrics_by_tree: dict[int, TreeMetrics],
    reconciliation: TreeReconciliationResult,
    origin_xyz: tuple[float, float, float],
    threshold_m: float,
) -> RiskResult:
    """Classify every tree and mark full-resolution danger points."""

    origin = np.asarray(origin_xyz, dtype=np.float64)
    danger_distance = np.asarray(clearance.danger_distance_m, dtype=np.float64)
    danger_tree = np.asarray(clearance.danger_tree_id, dtype=np.int32)
    danger_mask = danger_distance < threshold_m
    danger_original = np.asarray(clearance.danger_original_index, dtype=np.int64)[danger_mask]
    danger_xyz = np.asarray(clearance.danger_xyz_local_m, dtype=np.float32)[danger_mask]
    danger_dist = danger_distance[danger_mask]
    danger_wire = np.asarray(clearance.danger_wire_id, dtype=np.int32)[danger_mask]
    danger_q = np.asarray(
        clearance.danger_nearest_wire_xyz_local_m, dtype=np.float32
    )[danger_mask]

    # Per-tree danger point counts over the final (post-threshold) set.
    danger_counts: dict[int, int] = {}
    if len(danger_tree):
        unique, counts = np.unique(danger_tree, return_counts=True)
        for uid, count in zip(unique, counts, strict=True):
            danger_counts[int(uid)] = int(count)

    clearance_by_tree = {
        record.global_tree_id: record for record in clearance.clearance.records
    }
    records: list[TreeRiskRecord] = []
    for global_record in reconciliation.global_tree_records:
        tree_id = global_record.global_tree_id
        metric = metrics_by_tree.get(tree_id)
        clear_record = clearance_by_tree.get(tree_id)
        notes: list[str] = []
        dbh_quality = metric.dbh.dbh_quality if metric else "UNAVAILABLE"
        if dbh_quality == "UNAVAILABLE":
            notes.append("DBH 无可用证据")
        elif dbh_quality == "REVIEW":
            notes.append("DBH 需人工复核")
        elif dbh_quality == "ESTIMATED":
            notes.append("DBH 为估算值")
        if clear_record is None or clear_record.minimum_clearance_m is None:
            notes.append("未获得有效净空（无可用导线）")
        clearance_3d = clear_record.minimum_clearance_m if clear_record else None
        dangerous = bool(
            clearance_3d is not None and float(clearance_3d) < threshold_m
        )
        if dangerous:
            notes.append("危险")
        records.append(
            TreeRiskRecord(
                global_tree_id=tree_id,
                external_name=global_record.external_name,
                line_id=global_record.line_id,
                span_id=global_record.span_id,
                span_name=global_record.span_name,
                tree_height_m=metric.tree_height_m if metric else None,
                dbh_m=metric.dbh.dbh_m if metric else None,
                dbh_quality=dbh_quality,
                crown_major_width_m=metric.crown_major_width_m if metric else 0.0,
                crown_minor_width_m=metric.crown_minor_width_m if metric else 0.0,
                crown_mean_width_m=metric.crown_mean_width_m if metric else 0.0,
                crown_area_m2=metric.crown_area_m2 if metric else 0.0,
                fullres_point_count=global_record.full_resolution_point_count,
                tree_top_xyz_absolute_m=_to_absolute(
                    metric.tree_top_xyz_local_m if metric else None, origin
                ),
                tree_centroid_xyz_absolute_m=_to_absolute(
                    metric.tree_centroid_xyz_local_m if metric else None, origin
                ),
                nearest_wire_id=clear_record.nearest_wire_id if clear_record else None,
                nearest_tree_original_index=(
                    clear_record.nearest_tree_original_index if clear_record else None
                ),
                nearest_tree_xyz_absolute_m=_to_absolute(
                    clear_record.nearest_tree_xyz_local_m if clear_record else None,
                    origin,
                ),
                nearest_wire_xyz_absolute_m=_to_absolute(
                    clear_record.nearest_wire_xyz_local_m if clear_record else None,
                    origin,
                ),
                horizontal_clearance_m=(
                    _horizontal_clearance(clear_record) if clear_record else None
                ),
                vertical_clearance_m=(
                    _vertical_clearance(clear_record) if clear_record else None
                ),
                clearance_3d_m=clearance_3d,
                risk_threshold_m=threshold_m,
                danger_point_count=danger_counts.get(tree_id, 0),
                dangerous=dangerous,
                quality_notes="；".join(notes),
            )
        )
    dangerous_count = int(sum(1 for record in records if record.dangerous))
    return RiskResult(
        records=tuple(records),
        threshold_m=float(threshold_m),
        dangerous_tree_count=dangerous_count,
        danger_point_count=int(len(danger_original)),
        global_minimum_clearance_m=clearance.clearance.global_minimum_clearance_m,
        danger_original_index=danger_original,
        danger_xyz_local_m=danger_xyz,
        danger_distance_m=danger_dist,
        danger_wire_id=danger_wire,
        danger_nearest_wire_xyz_local_m=danger_q,
    )


def _horizontal_clearance(record: SpanTreeWireRecord) -> float | None:
    if record.nearest_tree_xyz_local_m is None or record.nearest_wire_xyz_local_m is None:
        return None
    tree = np.asarray(record.nearest_tree_xyz_local_m, dtype=np.float64)
    wire = np.asarray(record.nearest_wire_xyz_local_m, dtype=np.float64)
    return float(np.linalg.norm(tree[:2] - wire[:2]))


def _vertical_clearance(record: SpanTreeWireRecord) -> float | None:
    if record.nearest_tree_xyz_local_m is None or record.nearest_wire_xyz_local_m is None:
        return None
    tree = np.asarray(record.nearest_tree_xyz_local_m, dtype=np.float64)
    wire = np.asarray(record.nearest_wire_xyz_local_m, dtype=np.float64)
    return float(abs(tree[2] - wire[2]))


def risk_threshold_payload(
    source_sha256: str,
    source_identity: str,
    span_digest: str,
    threshold_m: float,
    dangerous_tree_count: int,
    danger_point_count: int,
    global_minimum_clearance_m: float | None,
) -> dict[str, Any]:
    """Digest-bound risk threshold persistence (bound to source and Span)."""

    return {
        "schema_version": MANUAL_RISK_SCHEMA,
        "non_certified_marker": "NON_CERTIFIED_RESEARCH_DEMONSTRATION",
        "source_sha256": source_sha256,
        "source_identity": source_identity,
        "span_digest": span_digest,
        "risk_threshold_m": float(threshold_m),
        "dangerous_tree_count": int(dangerous_tree_count),
        "danger_point_count": int(danger_point_count),
        "global_minimum_clearance_m": (
            None
            if global_minimum_clearance_m is None
            else float(global_minimum_clearance_m)
        ),
    }


__all__ = [
    "MANUAL_RISK_SCHEMA",
    "TreeRiskRecord",
    "RiskResult",
    "apply_risk_threshold",
    "risk_threshold_payload",
]
