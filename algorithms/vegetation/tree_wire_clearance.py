"""TREE_WIRE_DISTANCE_ALGORITHM_V1: full-resolution tree-wire clearance.

The engine reads only validated V4 and W1 artifacts.  Tree memberships are the
full-resolution confirmed/assigned V4 tree points (never visualization
samples, apexes, CHM points or crown centers).  Wire geometry is restricted to
distance-ready W1 polyline segments; candidate edges are built strictly
between consecutive sampled points inside each legitimate segment span.

Distance search uses a deterministic NumPy segment-AABB BVH for candidate
narrowing followed by exact float64 point-to-polyline-edge refinement.  All
ties are resolved with a frozen lexicographic key (wire id, W1 segment index,
polyline edge index, tree original index).

Before any distance is computed the engine hash-verifies the W1 parent A4
artifact under an explicit project root and fails closed on any source or
local-origin mismatch.  Arbitrary coordinate transforms are never performed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.tree_wire_clearance import (
    TREE_WIRE_CLEARANCE_SCHEMA_VERSION,
    TREE_WIRE_DISTANCE_ALGORITHM_V1,
    TreeWireClearanceConfig,
    TreeWireClearanceQualityFlag,
    TreeWireClearanceRecord,
    TreeWireClearanceRecordStatus,
    TreeWireClearanceResult,
    TreeWireClearanceStatus,
    default_tree_wire_clearance_config,
)
from core.contracts.v4_tree_segmentation import (
    V4TreeAssignmentState,
    V4TreeSegmentationError,
    V4TreeSegmentationResult,
)
from core.contracts.wire_models import (
    WireModelQualityFlag,
    WireModelStatus,
    WireReconstructionResult,
)
from core.contracts.wires import WireSeparationResult

from algorithms.vegetation.v4_tree_artifacts import load_v4_tree_segmentation_artifact
from algorithms.wires.reconstruct import WireReconstructionError, load_wire_models


class TreeWireClearanceErrorCode(str, Enum):
    """Closed failure codes for the clearance engine."""

    V4_ARTIFACT_INVALID = "V4_ARTIFACT_INVALID"
    W1_ARTIFACT_INVALID = "W1_ARTIFACT_INVALID"
    PROJECT_ROOT_INVALID = "PROJECT_ROOT_INVALID"
    W1_A4_PROVENANCE_FAILED = "W1_A4_PROVENANCE_FAILED"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    ORIGIN_MISMATCH = "ORIGIN_MISMATCH"
    CONFIG_INVALID = "CONFIG_INVALID"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"


class TreeWireClearanceError(Exception):
    """Fail-closed clearance failure with a stable code and message."""

    def __init__(self, code: TreeWireClearanceErrorCode, message: str) -> None:
        if type(code) is not TreeWireClearanceErrorCode:
            raise TypeError("code must be a TreeWireClearanceErrorCode")
        if type(message) is not str or not message:
            raise ValueError("message must be a non-empty str")
        self.code = code
        super().__init__(f"{code.value}:{message}")


# ---------------------------------------------------------------------------
# Exact geometry helpers (float64)
# ---------------------------------------------------------------------------


def exact_point_to_segment_distance(
    point: Any, a: Any, b: Any
) -> tuple[float, np.ndarray, float]:
    """Return ``(distance, nearest_xyz, t)`` for point ``p`` and edge ``[a, b]``.

    Uses the exact frozen V1 rule in float64: if the squared edge length is
    exactly zero the nearest point is the shared endpoint with ``t = 0``,
    otherwise the interior projection clamped to ``[0, 1]``.
    """

    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    denom = float(np.dot(d, d))
    if denom == 0.0:
        t = 0.0
        q = a
    else:
        t_raw = float(np.dot(p - a, d)) / denom
        t = min(1.0, max(0.0, t_raw))
        q = a + t * d
    distance = float(np.linalg.norm(p - q))
    return distance, q, t


def point_to_aabb_lower_bound(
    point: Any, min_corner: Any, max_corner: Any
) -> float:
    """Return a true lower bound on the distance from *point* to the AABB."""

    p = np.asarray(point, dtype=np.float64)
    mn = np.asarray(min_corner, dtype=np.float64)
    mx = np.asarray(max_corner, dtype=np.float64)
    delta = np.maximum(np.maximum(mn - p, 0.0), np.maximum(p - mx, 0.0))
    return float(np.linalg.norm(delta))


def candidate_better(
    distance: float,
    key: tuple[int, ...],
    best_distance: float | None,
    best_key: tuple[int, ...] | None,
    tolerance: float,
) -> bool:
    """Frozen V1 winner rule: distance first, ties broken lexicographically.

    Distances within *tolerance* are considered tied; the candidate with the
    smaller deterministic key then wins.
    """

    if best_distance is None or best_key is None:
        return True
    if distance < best_distance - tolerance:
        return True
    if distance > best_distance + tolerance:
        return False
    return key < best_key


# ---------------------------------------------------------------------------
# W1 edge bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WireEdgeBundle:
    """Flat, canonical-ordered distance-ready polyline edge arrays."""

    a_xyz: np.ndarray
    b_xyz: np.ndarray
    wire_id: np.ndarray
    segment_index: np.ndarray
    polyline_edge_index: np.ndarray
    wire_status: np.ndarray
    segment_quality_flags: np.ndarray
    edge_count: int


def build_wire_edge_bundle(w1_arrays: Mapping[str, Any]) -> WireEdgeBundle:
    """Build distance-ready edges from validated W1 arrays.

    Edges are created only between consecutive sampled points inside
    ``segment_distance_ready`` segment spans; gaps, non-ready segments,
    separate wires and segment boundaries are never bridged.  The resulting
    edge order is deterministic (ascending segment position, then ascending
    sample position within each segment).
    """

    segment_wire_id = np.asarray(w1_arrays["segment_wire_id"], dtype=np.int32)
    segment_index = np.asarray(w1_arrays["segment_index"], dtype=np.int32)
    segment_ready = np.asarray(w1_arrays["segment_distance_ready"], dtype=bool)
    segment_quality_flags = np.asarray(
        w1_arrays["segment_quality_flags"], dtype=np.uint16
    )
    sampled_curve_xyz = np.asarray(w1_arrays["sampled_curve_xyz"], dtype=np.float32)
    sampled_curve_offsets = np.asarray(
        w1_arrays["sampled_curve_offsets"], dtype=np.int64
    )
    wire_ids = np.asarray(w1_arrays["wire_id"], dtype=np.int32)
    wire_status = np.asarray(w1_arrays["wire_status"], dtype=np.uint8)

    ready_positions = np.flatnonzero(segment_ready)
    if ready_positions.size == 0:
        return WireEdgeBundle(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint16),
            0,
        )

    sample_counts = (
        sampled_curve_offsets[ready_positions + 1]
        - sampled_curve_offsets[ready_positions]
    )
    edge_counts = np.maximum(sample_counts - 1, 0)
    total_edges = int(np.sum(edge_counts, dtype=np.int64))
    if total_edges == 0:
        return WireEdgeBundle(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint16),
            0,
        )

    edge_segment_position = np.repeat(ready_positions, edge_counts).astype(
        np.int64, copy=False
    )
    segment_edge_starts = np.repeat(
        np.cumsum(edge_counts, dtype=np.int64) - edge_counts, edge_counts
    )
    edge_within_segment = np.arange(total_edges, dtype=np.int64) - segment_edge_starts

    sample_starts = sampled_curve_offsets[edge_segment_position]
    edge_a_positions = sample_starts + edge_within_segment
    edge_b_positions = edge_a_positions + 1

    wire_ids_per_edge = segment_wire_id[edge_segment_position]
    wire_positions = np.searchsorted(wire_ids, wire_ids_per_edge)
    if np.any(wire_positions >= wire_ids.shape[0]) or np.any(
        wire_ids[wire_positions] != wire_ids_per_edge
    ):
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_ARTIFACT_INVALID,
            "a distance-ready segment references an unknown wire",
        )

    return WireEdgeBundle(
        a_xyz=sampled_curve_xyz[edge_a_positions].astype(np.float64),
        b_xyz=sampled_curve_xyz[edge_b_positions].astype(np.float64),
        wire_id=wire_ids_per_edge,
        segment_index=segment_index[edge_segment_position],
        polyline_edge_index=edge_within_segment.astype(np.int32),
        wire_status=wire_status[wire_positions],
        segment_quality_flags=segment_quality_flags[edge_segment_position],
        edge_count=total_edges,
    )


# ---------------------------------------------------------------------------
# Deterministic segment-AABB BVH
# ---------------------------------------------------------------------------


class _BvhNode:
    __slots__ = ("leaf", "min_corner", "max_corner", "lo", "hi", "left", "right")

    def __init__(
        self,
        leaf: bool,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
        lo: int,
        hi: int,
        left: _BvhNode | None,
        right: _BvhNode | None,
    ) -> None:
        self.leaf = leaf
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.lo = lo
        self.hi = hi
        self.left = left
        self.right = right


class SegmentEdgeBvh:
    """Deterministic NumPy segment-AABB BVH over distance-ready edges.

    The build uses a frozen split rule (longest AABB axis, then median split
    with stable lexicographic ordering).  Traversal performs branch-and-bound
    AABB lower-bound pruning that never discards an edge that can win under the
    exact tie tolerance; leaves are refined with exact point-to-segment
    distances.
    """

    def __init__(self, edges: WireEdgeBundle, leaf_cap: int) -> None:
        if not isinstance(edges, WireEdgeBundle):
            raise TypeError("edges must be a WireEdgeBundle")
        if type(leaf_cap) is not int or leaf_cap <= 0:
            raise ValueError("leaf_cap must be a positive integer")
        self.edges = edges
        self.leaf_cap = leaf_cap
        self.min_corner = np.minimum(edges.a_xyz, edges.b_xyz)
        self.max_corner = np.maximum(edges.a_xyz, edges.b_xyz)
        self.centroid = 0.5 * (self.min_corner + self.max_corner)
        self.order = np.arange(edges.edge_count, dtype=np.int64)
        self.root = self._build(0, edges.edge_count)

    def _build(self, lo: int, hi: int) -> _BvhNode:
        order = self.order
        node_min = self.min_corner[order[lo:hi]].min(axis=0)
        node_max = self.max_corner[order[lo:hi]].max(axis=0)
        if hi - lo <= self.leaf_cap:
            return _BvhNode(True, node_min, node_max, lo, hi, None, None)
        axis = int(np.argmax(node_max - node_min))
        sub = order[lo:hi]
        centroids = self.centroid[sub, axis]
        permutation = np.lexsort((sub, centroids))
        order[lo:hi] = sub[permutation]
        mid = lo + (hi - lo) // 2
        left = self._build(lo, mid)
        right = self._build(mid, hi)
        return _BvhNode(False, node_min, node_max, lo, hi, left, right)

    def nearest_edge(
        self, point: Any, tolerance: float
    ) -> tuple[int, float, np.ndarray, float, int]:
        """Return the deterministic nearest edge to *point*.

        Returns ``(edge_position, distance, nearest_xyz, t, evaluated_count)``
        where *evaluated_count* is the number of leaf edges refined during the
        search.  The returned edge minimizes ``(distance, wire_id, W1 segment
        index, polyline edge index)`` under the frozen tie rule.
        """

        p = np.asarray(point, dtype=np.float64)
        best_distance: float | None = None
        best_key: tuple[int, int, int] | None = None
        best_edge_position = -1
        best_q: np.ndarray | None = None
        best_t = 0.0
        evaluated = 0
        stack: list[_BvhNode] = [self.root]
        while stack:
            node = stack.pop()
            if best_distance is not None and point_to_aabb_lower_bound(
                p, node.min_corner, node.max_corner
            ) > best_distance + tolerance:
                continue
            if node.leaf:
                order = self.order
                for position in range(node.lo, node.hi):
                    edge_position = int(order[position])
                    distance, q, t = exact_point_to_segment_distance(
                        p, self.edges.a_xyz[edge_position], self.edges.b_xyz[edge_position]
                    )
                    evaluated += 1
                    key = (
                        int(self.edges.wire_id[edge_position]),
                        int(self.edges.segment_index[edge_position]),
                        int(self.edges.polyline_edge_index[edge_position]),
                    )
                    if candidate_better(
                        distance, key, best_distance, best_key, tolerance
                    ):
                        best_distance = distance
                        best_key = key
                        best_edge_position = edge_position
                        best_q = q
                        best_t = t
            else:
                stack.append(node.right)
                stack.append(node.left)
        if best_edge_position < 0:
            raise TreeWireClearanceError(
                TreeWireClearanceErrorCode.W1_ARTIFACT_INVALID,
                "BVH found no edge for a tree point",
            )
        return best_edge_position, best_distance, best_q, best_t, evaluated


# ---------------------------------------------------------------------------
# Provenance / coordinate gate
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_relative_posix(value: str, name: str) -> None:
    if (
        not value
        or ":" in value
        or "\\" in value
        or value.startswith("/")
        or Path(value).is_absolute()
    ):
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_A4_PROVENANCE_FAILED,
            f"{name} must be a relative POSIX path",
        )


def _verify_a4_provenance(
    w1_result: WireReconstructionResult, project_root: Path | str
) -> WireSeparationResult:
    """Resolve and hash-verify the W1 parent A4 artifact under *project_root*."""

    if project_root is None:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.PROJECT_ROOT_INVALID,
            "project_root is required to resolve the W1 parent A4 artifact",
        )
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.PROJECT_ROOT_INVALID,
            "project_root must be an existing directory",
        )
    _require_relative_posix(w1_result.input_a4_npz_relpath, "input_a4_npz_relpath")
    _require_relative_posix(w1_result.input_a4_json_relpath, "input_a4_json_relpath")
    a4_npz = (root / w1_result.input_a4_npz_relpath).resolve()
    a4_json = (root / w1_result.input_a4_json_relpath).resolve()
    for path in (a4_npz, a4_json):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TreeWireClearanceError(
                TreeWireClearanceErrorCode.PROJECT_ROOT_INVALID,
                "A4 parent must resolve inside the project root",
            ) from exc
    if not a4_npz.is_file() or not a4_json.is_file():
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_A4_PROVENANCE_FAILED,
            "W1 parent A4 files are missing under project_root",
        )
    a4_npz_sha = _sha256_file(a4_npz)
    a4_json_sha = _sha256_file(a4_json)
    if a4_npz_sha != w1_result.input_a4_npz_sha256:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_A4_PROVENANCE_FAILED,
            "A4 NPZ SHA-256 does not match W1 provenance",
        )
    if a4_json_sha != w1_result.input_a4_json_sha256:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_A4_PROVENANCE_FAILED,
            "A4 JSON SHA-256 does not match W1 provenance",
        )
    try:
        payload = json.loads(a4_json.read_text("utf-8"))
        a4_result = WireSeparationResult.from_dict(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_A4_PROVENANCE_FAILED,
            "unable to parse the W1 parent A4 JSON",
        ) from exc
    if a4_result.output_npz_sha256 != a4_npz_sha:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_A4_PROVENANCE_FAILED,
            "A4 NPZ SHA-256 does not match the A4 result",
        )
    return a4_result


def _verify_source_and_origin(
    v4_result: V4TreeSegmentationResult,
    w1_result: WireReconstructionResult,
    a4_result: WireSeparationResult,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float], str]:
    """Verify identical source identity and local origin across V4 and A4/W1."""

    v4_source_id = v4_result.source_identity.source_id
    a4_source_id = a4_result.source_id
    if v4_source_id != a4_source_id:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.SOURCE_MISMATCH,
            "V4 and A4 source identifiers differ",
        )
    if w1_result.source_id != a4_source_id:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.SOURCE_MISMATCH,
            "W1 and A4 source identifiers differ",
        )
    if w1_result.manifest_source_id != a4_result.manifest_source_id:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.SOURCE_MISMATCH,
            "W1 and A4 manifest source identifiers differ",
        )
    v4_origin = tuple(float(value) for value in v4_result.source_identity.origin_context)
    a4_origin = tuple(float(value) for value in a4_result.origin_xyz)
    if v4_origin != a4_origin:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.ORIGIN_MISMATCH,
            "V4 and A4/W1 local origins differ",
        )
    return v4_source_id, v4_origin, a4_origin, a4_result.manifest_source_id


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


def _quality_flags(wire_status: int, segment_quality_flags: int) -> int:
    flags = 0
    if wire_status != int(WireModelStatus.CONFIRMED_MODEL):
        flags |= int(TreeWireClearanceQualityFlag.NEAREST_WIRE_NOT_CONFIRMED)
    non_production = (
        int(WireModelQualityFlag.FALLBACK_USED)
        | int(WireModelQualityFlag.RECONSTRUCTION_FAILURE)
        | int(WireModelQualityFlag.HIGH_RESIDUAL)
        | int(WireModelQualityFlag.LARGE_GAP)
        | int(WireModelQualityFlag.ENDPOINT_UNCERTAINTY)
    )
    if int(segment_quality_flags) & non_production:
        flags |= int(TreeWireClearanceQualityFlag.NEAREST_SEGMENT_FALLBACK_OR_REVIEW)
    return flags


def run_tree_wire_clearance(
    v4_root: Path | str,
    w1_npz_path: Path | str,
    w1_json_path: Path | str,
    project_root: Path | str,
    *,
    config: TreeWireClearanceConfig | None = None,
) -> TreeWireClearanceResult:
    """Compute full-resolution tree-wire minimum clearances, fail closed.

    Reads a validated V4 tree-segmentation artifact and a validated W1 wire
    model pair, hash-verifies the W1 parent A4 through *project_root*, and
    returns a digest-bound ``TreeWireClearanceResult``.  No segmentation, wire
    reconstruction, FRV, Ground or prediction step is ever invoked.
    """

    if config is None:
        config = default_tree_wire_clearance_config()
    if type(config) is not TreeWireClearanceConfig:
        raise TypeError("config must be a TreeWireClearanceConfig")

    try:
        loaded = load_v4_tree_segmentation_artifact(Path(v4_root))
    except V4TreeSegmentationError as exc:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.V4_ARTIFACT_INVALID, str(exc)
        ) from exc
    v4_result = loaded.result

    try:
        w1_result, w1_arrays = load_wire_models(Path(w1_npz_path), Path(w1_json_path))
    except WireReconstructionError as exc:
        raise TreeWireClearanceError(
            TreeWireClearanceErrorCode.W1_ARTIFACT_INVALID, str(exc)
        ) from exc

    a4_result = _verify_a4_provenance(w1_result, project_root)
    v4_source_id, v4_origin, a4_origin, a4_manifest_source_id = _verify_source_and_origin(
        v4_result, w1_result, a4_result
    )

    w1_npz_sha256 = w1_result.output_npz_sha256
    w1_json_sha256 = _sha256_file(Path(w1_json_path))
    a4_npz_sha256 = w1_result.input_a4_npz_sha256
    a4_json_sha256 = w1_result.input_a4_json_sha256

    edge_bundle = build_wire_edge_bundle(w1_arrays)
    records: list[TreeWireClearanceRecord] = []

    if edge_bundle.edge_count == 0:
        for tree in v4_result.tree_records:
            positions = np.flatnonzero(
                (v4_result.tree_id_by_point == int(tree.tree_id))
                & (
                    v4_result.assignment_state_by_point
                    == int(V4TreeAssignmentState.ASSIGNED)
                )
            )
            records.append(
                TreeWireClearanceRecord(
                    tree_id=int(tree.tree_id),
                    status=TreeWireClearanceRecordStatus.NO_DISTANCE_READY_EDGE,
                    nearest_wire_id=None,
                    minimum_clearance_m=None,
                    nearest_tree_original_index=None,
                    nearest_tree_xyz_local_m=None,
                    nearest_wire_xyz_local_m=None,
                    w1_segment_index=None,
                    polyline_edge_index=None,
                    edge_interpolation_t=None,
                    tree_point_count=int(positions.shape[0]),
                    candidate_edge_count=0,
                    evaluated_edge_count=0,
                    quality_flags=0,
                    nearest_wire_status=None,
                    nearest_segment_quality_flags=None,
                )
            )
        result = TreeWireClearanceResult(
            schema_version=TREE_WIRE_CLEARANCE_SCHEMA_VERSION,
            algorithm_version=TREE_WIRE_DISTANCE_ALGORITHM_V1,
            status=TreeWireClearanceStatus.NO_DISTANCE_READY_EDGE,
            v4_result_digest=v4_result.result_digest,
            v4_source_id=v4_source_id,
            v4_source_identity_record_digest=v4_result.source_identity.source_identity_record_digest,
            v4_origin_context=v4_origin,
            a4_source_id=a4_result.source_id,
            a4_manifest_source_id=a4_manifest_source_id,
            a4_origin_xyz=a4_origin,
            w1_npz_sha256=w1_npz_sha256,
            w1_json_sha256=w1_json_sha256,
            a4_npz_sha256=a4_npz_sha256,
            a4_json_sha256=a4_json_sha256,
            config_digest=config.config_digest,
            records=tuple(records),
            result_digest="0" * 64,
        )
        return _with_result_digest(result)

    bvh = SegmentEdgeBvh(edge_bundle, config.bvh_leaf_cap)
    for tree in v4_result.tree_records:
        tree_id = int(tree.tree_id)
        positions = np.flatnonzero(
            (v4_result.tree_id_by_point == tree_id)
            & (
                v4_result.assignment_state_by_point
                == int(V4TreeAssignmentState.ASSIGNED)
            )
        )
        tree_point_count = int(positions.shape[0])
        if tree_point_count == 0:
            records.append(
                TreeWireClearanceRecord(
                    tree_id=tree_id,
                    status=TreeWireClearanceRecordStatus.INVALID_TREE_NO_VALID_POINT,
                    nearest_wire_id=None,
                    minimum_clearance_m=None,
                    nearest_tree_original_index=None,
                    nearest_tree_xyz_local_m=None,
                    nearest_wire_xyz_local_m=None,
                    w1_segment_index=None,
                    polyline_edge_index=None,
                    edge_interpolation_t=None,
                    tree_point_count=tree_point_count,
                    candidate_edge_count=edge_bundle.edge_count,
                    evaluated_edge_count=0,
                    quality_flags=0,
                    nearest_wire_status=None,
                    nearest_segment_quality_flags=None,
                )
            )
            continue
        best_distance: float | None = None
        best_key: tuple[int, int, int, int] | None = None
        best_edge_position = -1
        best_q: np.ndarray | None = None
        best_t = 0.0
        best_original_index = -1
        best_point: np.ndarray | None = None
        evaluated_total = 0
        for position in positions:
            point = v4_result.xyz_local_m[int(position)].astype(np.float64)
            edge_position, distance, q, t, evaluated = bvh.nearest_edge(
                point, config.tie_tolerance_m
            )
            evaluated_total += evaluated
            original_index = int(v4_result.original_index[int(position)])
            key = (
                int(edge_bundle.wire_id[edge_position]),
                int(edge_bundle.segment_index[edge_position]),
                int(edge_bundle.polyline_edge_index[edge_position]),
                original_index,
            )
            if candidate_better(
                distance, key, best_distance, best_key, config.tie_tolerance_m
            ):
                best_distance = distance
                best_key = key
                best_edge_position = edge_position
                best_q = q
                best_t = t
                best_original_index = original_index
                best_point = point
        records.append(
            TreeWireClearanceRecord(
                tree_id=tree_id,
                status=TreeWireClearanceRecordStatus.VALID,
                nearest_wire_id=best_key[0],
                minimum_clearance_m=best_distance,
                nearest_tree_original_index=best_original_index,
                nearest_tree_xyz_local_m=tuple(float(value) for value in best_point),
                nearest_wire_xyz_local_m=tuple(float(value) for value in best_q),
                w1_segment_index=best_key[1],
                polyline_edge_index=best_key[2],
                edge_interpolation_t=best_t,
                tree_point_count=tree_point_count,
                candidate_edge_count=edge_bundle.edge_count,
                evaluated_edge_count=evaluated_total,
                quality_flags=_quality_flags(
                    int(edge_bundle.wire_status[best_edge_position]),
                    int(edge_bundle.segment_quality_flags[best_edge_position]),
                ),
                nearest_wire_status=int(edge_bundle.wire_status[best_edge_position]),
                nearest_segment_quality_flags=int(
                    edge_bundle.segment_quality_flags[best_edge_position]
                ),
            )
        )

    status = TreeWireClearanceStatus.VALID
    result = TreeWireClearanceResult(
        schema_version=TREE_WIRE_CLEARANCE_SCHEMA_VERSION,
        algorithm_version=TREE_WIRE_DISTANCE_ALGORITHM_V1,
        status=status,
        v4_result_digest=v4_result.result_digest,
        v4_source_id=v4_source_id,
        v4_source_identity_record_digest=v4_result.source_identity.source_identity_record_digest,
        v4_origin_context=v4_origin,
        a4_source_id=a4_result.source_id,
        a4_manifest_source_id=a4_manifest_source_id,
        a4_origin_xyz=a4_origin,
        w1_npz_sha256=w1_npz_sha256,
        w1_json_sha256=w1_json_sha256,
        a4_npz_sha256=a4_npz_sha256,
        a4_json_sha256=a4_json_sha256,
        config_digest=config.config_digest,
        records=tuple(records),
        result_digest="0" * 64,
    )
    return _with_result_digest(result)


def _with_result_digest(result: TreeWireClearanceResult) -> TreeWireClearanceResult:
    clone = TreeWireClearanceResult.__new__(TreeWireClearanceResult)
    for name in TreeWireClearanceResult.__dataclass_fields__:
        if name == "result_digest":
            object.__setattr__(clone, name, result.computed_result_digest())
        else:
            object.__setattr__(clone, name, getattr(result, name))
    return clone


__all__ = [
    "TreeWireClearanceError",
    "TreeWireClearanceErrorCode",
    "WireEdgeBundle",
    "SegmentEdgeBvh",
    "exact_point_to_segment_distance",
    "point_to_aabb_lower_bound",
    "candidate_better",
    "build_wire_edge_bundle",
    "run_tree_wire_clearance",
]
