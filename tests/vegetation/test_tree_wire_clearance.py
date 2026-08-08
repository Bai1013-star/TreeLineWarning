"""Engine, geometry and BVH tests for TREE_WIRE_DISTANCE_ALGORITHM_V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from algorithms.vegetation.tree_wire_clearance import (
    SegmentEdgeBvh,
    TreeWireClearanceError,
    TreeWireClearanceErrorCode,
    WireEdgeBundle,
    build_wire_edge_bundle,
    candidate_better,
    exact_point_to_segment_distance,
    point_to_aabb_lower_bound,
    run_tree_wire_clearance,
)
from algorithms.vegetation.v4_tree_artifacts import publish_v4_tree_segmentation_artifact
from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees
from algorithms.wires.reconstruct import SCHEMA_VERSION as WIRE_SCHEMA
from core.contracts.tree_wire_clearance import (
    TreeWireClearanceQualityFlag,
    TreeWireClearanceRecordStatus,
    TreeWireClearanceResult,
    TreeWireClearanceStatus,
    default_tree_wire_clearance_config,
)
from core.contracts.wire_models import (
    WireModelMethod,
    WireModelQualityFlag,
    WireModelStatus,
    WireModelSummary,
    WireReconstructionConfig,
    WireReconstructionResult,
    WireReconstructionStatistics,
)
from core.contracts.wires import (
    WireSeedConfig,
    WireSeparationConfig,
    WireSeparationResult,
    WireSeparationStatistics,
)

from tests.vegetation.test_v4_tree_segmentation import (
    make_config,
    make_handoff,
    two_trees_scene,
)

_V4_ASSIGNED = 1


# ---------------------------------------------------------------------------
# V4 artifact fixture
# ---------------------------------------------------------------------------


def _make_v4_artifact(
    tmp_path: Path, scene_points: list[tuple[float, float, float]] | None = None
) -> Path:
    handoff = make_handoff(scene_points if scene_points is not None else two_trees_scene())
    result = segment_v4_trees(handoff, make_config())
    root = tmp_path / "v4"
    publish_v4_tree_segmentation_artifact(result, root, scratch_dir=tmp_path)
    return root


# ---------------------------------------------------------------------------
# W1 + A4 artifact fixtures
# ---------------------------------------------------------------------------


def _w1_arrays(
    segments: list[tuple[int, int, list[tuple[float, float, float]]]],
    *,
    ready: list[bool] | None = None,
    segment_quality: list[int] | None = None,
    wire_statuses: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Build the W1 NPZ array inventory.

    *segments* are ``(wire_id, segment_index, curve_xyz)`` tuples; the arrays
    are placed in canonical (wire id, segment index) order.
    """

    ordered = sorted(segments, key=lambda item: (item[0], item[1]))
    wire_ids = sorted({item[0] for item in ordered})
    segment_count = len(ordered)
    if ready is None:
        ready = [True] * segment_count
    if segment_quality is None:
        segment_quality = [0] * segment_count
    if wire_statuses is None:
        wire_statuses = [int(WireModelStatus.CONFIRMED_MODEL)] * len(wire_ids)
    wire_id_array = np.asarray(wire_ids, dtype=np.int32)
    wire_segment_offsets = np.asarray(
        [0, *np.cumsum([sum(1 for item in ordered if item[0] == wire) for wire in wire_ids])],
        dtype=np.int64,
    )
    curve_xyz = np.concatenate(
        [np.asarray(item[2], dtype=np.float32) for item in ordered], axis=0
    )
    curve_offsets = [0]
    for item in ordered:
        curve_offsets.append(curve_offsets[-1] + len(item[2]))
    segment_wire_id = np.asarray([item[0] for item in ordered], dtype=np.int32)
    segment_index = np.asarray([item[1] for item in ordered], dtype=np.int32)
    return {
        "schema_version": np.asarray(WIRE_SCHEMA),
        "wire_id": wire_id_array,
        "source_track_id": wire_id_array.copy(),
        "wire_status": np.asarray(wire_statuses, dtype=np.uint8),
        "wire_method": np.zeros(len(wire_ids), dtype=np.uint8),
        "wire_quality_flags": np.zeros(len(wire_ids), dtype=np.uint16),
        "wire_has_distance_ready_segments": np.ones(len(wire_ids), dtype=np.bool_),
        "wire_continuous_distance_ready": np.ones(len(wire_ids), dtype=np.bool_),
        "wire_segment_offsets": wire_segment_offsets,
        "segment_wire_id": segment_wire_id,
        "segment_index": segment_index,
        "segment_method": np.zeros(segment_count, dtype=np.uint8),
        "segment_quality_flags": np.asarray(segment_quality, dtype=np.uint16),
        "segment_distance_ready": np.asarray(ready, dtype=np.bool_),
        "segment_parameter_domain": np.zeros((segment_count, 2), dtype=np.float64),
        "sampled_curve_xyz": curve_xyz,
        "sampled_curve_parameter": np.linspace(0.0, 1.0, len(curve_xyz), dtype=np.float32),
        "sampled_curve_offsets": np.asarray(curve_offsets, dtype=np.int64),
    }


def _write_a4_w1(
    tmp_path: Path,
    segments: list[tuple[int, int, list[tuple[float, float, float]]]],
    *,
    ready: list[bool] | None = None,
    segment_quality: list[int] | None = None,
    wire_statuses: list[int] | None = None,
    source_id: str = "scene-1",
    manifest_source_id: str = "scene-1",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    w1_manifest_source_id: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write a W1 pair plus its hash-verified A4 parent under ``tmp_path``.

    Returns ``(w1_npz, w1_json, project_root)``.  The A4 NPZ content is
    arbitrary bytes; only its SHA-256 must match the A4 result and the W1
    provenance fields.
    """

    root = tmp_path / "project"
    wire_dir = root / "outputs" / "wires"
    model_dir = root / "outputs" / "wire_models"
    wire_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    a4_npz = wire_dir / "scene-1_wire_instances.npz"
    a4_json = wire_dir / "scene-1_wire_instances.json"
    np.savez(a4_npz, source_id=np.asarray(source_id))
    a4_npz_sha = hashlib.sha256(a4_npz.read_bytes()).hexdigest()
    a4_result = WireSeparationResult(
        source_id=source_id,
        manifest_source_id=manifest_source_id,
        input_a3_npz_path=root / "outputs/candidates/in.npz",
        input_a3_json_path=root / "outputs/candidates/in.json",
        input_a3_npz_sha256="0" * 64,
        input_a3_json_sha256="0" * 64,
        output_npz_path=a4_npz,
        output_json_path=a4_json,
        output_npz_sha256=a4_npz_sha,
        output_npz_bytes=a4_npz.stat().st_size,
        origin_xyz=tuple(origin),
        center_xyz=(0.0, 0.0, 0.0),
        s_axis=(1.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        config=WireSeparationConfig(source_id=source_id, expected_input_point_count=1),
        seeds=WireSeedConfig(),
        statistics=WireSeparationStatistics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0),
        arrays={},
        summaries=(),
        assignment_reason_counts={},
        assignment_quality_counts={},
        track_quality_counts={},
        confirmation_transition_counts={},
        formulas={},
        warnings=(),
        algorithm_version="test-a4",
        created_at="2026-01-01T00:00:00+00:00",
    )
    a4_json.write_text(json.dumps(a4_result.to_dict()), encoding="utf-8")
    a4_json_sha = hashlib.sha256(a4_json.read_bytes()).hexdigest()

    arrays = _w1_arrays(
        segments,
        ready=ready,
        segment_quality=segment_quality,
        wire_statuses=wire_statuses,
    )
    w1_npz = model_dir / f"{source_id}_wire_models.npz"
    w1_json = model_dir / f"{source_id}_wire_models.json"
    np.savez_compressed(w1_npz, **arrays)
    w1_npz_sha = hashlib.sha256(w1_npz.read_bytes()).hexdigest()
    summaries = tuple(
        WireModelSummary(
            wire_id=int(arrays["wire_id"][position]),
            source_track_id=int(arrays["source_track_id"][position]),
            status=WireModelStatus(int(arrays["wire_status"][position])),
            method=WireModelMethod.CUBIC_BSPLINE,
            quality_flags=0,
            wire_has_distance_ready_segments=True,
            wire_continuous_distance_ready=True,
            observed_segment_count=1,
            point_count=1,
            representative_count=1,
            observed_total_span_m=0.0,
            observed_envelope_span_m=0.0,
            total_unbridged_gap_span_m=0.0,
            observed_coverage_ratio=1.0,
            observed_total_arc_length_m=0.0,
            start_xyz=(0.0, 0.0, 0.0),
            end_xyz=(0.0, 0.0, 0.0),
            main_direction=(1.0, 0.0, 0.0),
        )
        for position in range(len(arrays["wire_id"]))
    )
    w1_result = WireReconstructionResult(
        schema_version=WIRE_SCHEMA,
        source_id=source_id,
        manifest_source_id=(
            manifest_source_id if w1_manifest_source_id is None else w1_manifest_source_id
        ),
        input_a4_npz_relpath=str(a4_npz.relative_to(root)).replace("\\", "/"),
        input_a4_json_relpath=str(a4_json.relative_to(root)).replace("\\", "/"),
        input_a4_npz_sha256=a4_npz_sha,
        input_a4_json_sha256=a4_json_sha,
        output_npz_relpath=str(w1_npz.relative_to(root)).replace("\\", "/"),
        output_json_relpath=str(w1_json.relative_to(root)).replace("\\", "/"),
        output_npz_sha256=w1_npz_sha,
        output_npz_bytes=w1_npz.stat().st_size,
        config=WireReconstructionConfig(source_id=source_id),
        statistics=WireReconstructionStatistics(
            input_point_count=0,
            confirmed_wire_point_count=0,
            wire_count=len(arrays["wire_id"]),
            segment_count=len(arrays["segment_wire_id"]),
            gap_count=0,
            distance_ready_segment_count=int(
                np.count_nonzero(arrays["segment_distance_ready"])
            ),
            elapsed_seconds=0.1,
        ),
        arrays={},
        wires=summaries,
        segments=(),
        gaps=(),
        warnings=(),
        algorithm_version="w1-confirmed-wire-centreline-v3",
        created_at="2026-01-01T00:00:00+00:00",
    )
    w1_json.write_text(json.dumps(w1_result.to_dict()), encoding="utf-8")
    return w1_npz, w1_json, root


def _wire_segment(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> list[tuple[float, float, float]]:
    return [start, end]


# ---------------------------------------------------------------------------
# Independent brute-force reference
# ---------------------------------------------------------------------------


def _better(
    distance: float,
    key: tuple[int, ...],
    best_distance: float | None,
    best_key: tuple[int, ...] | None,
    tolerance: float,
) -> bool:
    if best_distance is None or best_key is None:
        return True
    if distance < best_distance - tolerance:
        return True
    if distance > best_distance + tolerance:
        return False
    return key < best_key


def _brute_force_tree_clearance(
    v4_result, bundle: WireEdgeBundle, tolerance: float
) -> dict[int, tuple]:
    """Return per-tree brute-force winner tuples ``(dist, key, p, q, t, orig)``."""

    results: dict[int, tuple] = {}
    for tree in v4_result.tree_records:
        tree_id = int(tree.tree_id)
        positions = np.flatnonzero(
            (v4_result.tree_id_by_point == tree_id)
            & (v4_result.assignment_state_by_point == _V4_ASSIGNED)
        )
        best_distance: float | None = None
        best_key: tuple[int, int, int, int] | None = None
        best: tuple | None = None
        for position in positions:
            p = np.asarray(v4_result.xyz_local_m[int(position)], dtype=np.float64)
            original_index = int(v4_result.original_index[int(position)])
            for edge_position in range(bundle.edge_count):
                a = bundle.a_xyz[edge_position]
                b = bundle.b_xyz[edge_position]
                d = b - a
                denom = float(np.dot(d, d))
                if denom == 0.0:
                    t = 0.0
                    q = a
                else:
                    t = min(1.0, max(0.0, float(np.dot(p - a, d)) / denom))
                    q = a + t * d
                distance = float(np.linalg.norm(p - q))
                key = (
                    int(bundle.wire_id[edge_position]),
                    int(bundle.segment_index[edge_position]),
                    int(bundle.polyline_edge_index[edge_position]),
                    original_index,
                )
                if _better(distance, key, best_distance, best_key, tolerance):
                    best_distance = distance
                    best_key = key
                    best = (distance, key, p, q, t, original_index)
        results[tree_id] = best
    return results


# ---------------------------------------------------------------------------
# Exact point-to-segment geometry
# ---------------------------------------------------------------------------


def test_point_directly_below_horizontal_segment() -> None:
    distance, q, t = exact_point_to_segment_distance(
        (1.0, 0.0, 0.0), (0.0, 0.0, 2.0), (2.0, 0.0, 2.0)
    )
    assert distance == pytest.approx(2.0)
    assert t == pytest.approx(0.5)
    assert np.allclose(q, (1.0, 0.0, 2.0))


def test_point_nearest_at_segment_interior() -> None:
    distance, q, t = exact_point_to_segment_distance(
        (3.0, 1.0, 0.0), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)
    )
    assert distance == pytest.approx(1.0)
    assert t == pytest.approx(0.75)
    assert np.allclose(q, (3.0, 0.0, 0.0))


def test_point_nearest_at_endpoint() -> None:
    distance, q, t = exact_point_to_segment_distance(
        (-1.0, 5.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)
    )
    assert distance == pytest.approx(np.sqrt(1.0 + 25.0))
    assert t == pytest.approx(0.0)
    assert np.allclose(q, (0.0, 0.0, 0.0))
    distance_end, q_end, t_end = exact_point_to_segment_distance(
        (5.0, 5.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)
    )
    assert t_end == pytest.approx(1.0)
    assert np.allclose(q_end, (2.0, 0.0, 0.0))


def test_point_oblique_3d_segment() -> None:
    distance, q, t = exact_point_to_segment_distance(
        (2.0, 2.0, 0.0), (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)
    )
    assert distance == pytest.approx(np.sqrt(8.0 / 3.0))
    assert t == pytest.approx(1.0 / 6.0)
    assert np.allclose(q, (4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0))


def test_zero_length_edge_uses_endpoint() -> None:
    distance, q, t = exact_point_to_segment_distance(
        (5.0, 1.0, 0.0), (2.0, 3.0, 0.0), (2.0, 3.0, 0.0)
    )
    assert distance == pytest.approx(np.sqrt(9.0 + 4.0))
    assert t == pytest.approx(0.0)
    assert np.allclose(q, (2.0, 3.0, 0.0))


def test_point_to_aabb_lower_bound_inside_is_zero() -> None:
    assert point_to_aabb_lower_bound((0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)) == 0.0
    assert point_to_aabb_lower_bound(
        (3.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
    ) == pytest.approx(2.0)


def test_candidate_better_tie_rule() -> None:
    tolerance = 1e-9
    # Larger wire id loses a within-tolerance tie.
    assert not candidate_better(1.0, (1, 0, 0), 1.0 + 5e-10, (0, 0, 0), tolerance)
    assert candidate_better(1.0, (0, 1, 2), 1.0 + 5e-10, (1, 0, 0), tolerance)
    # Out-of-tolerance distance decides even against a smaller key.
    assert not candidate_better(2.0, (0, 0, 0), 1.0, (0, 0, 0), tolerance)
    assert candidate_better(1.0, (5, 0, 0), 2.0, (0, 0, 0), tolerance)


# ---------------------------------------------------------------------------
# W1 edge bundle construction
# ---------------------------------------------------------------------------


def test_build_wire_edge_bundle_only_ready_segments() -> None:
    arrays = _w1_arrays(
        [
            (0, 0, [(0, 0, 0), (2, 0, 0), (4, 0, 0)]),
            (0, 1, [(0, 10, 0), (4, 10, 0)]),
            (1, 0, [(0, 20, 0), (4, 20, 0)]),
        ],
        ready=[True, False, True],
    )
    bundle = build_wire_edge_bundle(arrays)
    assert bundle.edge_count == 3  # seg0: 2 edges, seg1 non-ready skipped, seg2: 1 edge
    assert bundle.wire_id.tolist() == [0, 0, 1]
    assert bundle.segment_index.tolist() == [0, 0, 0]
    assert bundle.polyline_edge_index.tolist() == [0, 1, 0]


def test_build_wire_edge_bundle_never_bridges_segments() -> None:
    # Adjacent segment endpoints never form an edge across the boundary.
    arrays = _w1_arrays(
        [
            (0, 0, [(0, 0, 0), (2, 0, 0)]),
            (0, 1, [(0, 0, 0), (2, 0, 0)]),
        ]
    )
    bundle = build_wire_edge_bundle(arrays)
    assert bundle.edge_count == 2
    assert bundle.polyline_edge_index.tolist() == [0, 0]


def test_build_wire_edge_bundle_empty_when_no_ready() -> None:
    arrays = _w1_arrays([(0, 0, [(0, 0, 0), (2, 0, 0)])], ready=[False])
    bundle = build_wire_edge_bundle(arrays)
    assert bundle.edge_count == 0
    assert bundle.a_xyz.shape == (0, 3)


def test_build_wire_edge_bundle_single_sample_segment_is_skipped() -> None:
    arrays = _w1_arrays([(0, 0, [(1, 1, 1)])])
    bundle = build_wire_edge_bundle(arrays)
    assert bundle.edge_count == 0


def test_build_wire_edge_bundle_unknown_wire_fails() -> None:
    arrays = _w1_arrays([(0, 0, [(0, 0, 0), (2, 0, 0)])])
    arrays["segment_wire_id"] = np.asarray([7], dtype=np.int32)
    with pytest.raises(TreeWireClearanceError) as excinfo:
        build_wire_edge_bundle(arrays)
    assert excinfo.value.code is TreeWireClearanceErrorCode.W1_ARTIFACT_INVALID


# ---------------------------------------------------------------------------
# BVH
# ---------------------------------------------------------------------------


def _three_wire_bundle() -> WireEdgeBundle:
    arrays = _w1_arrays(
        [
            (0, 0, [(0, 0, 0), (2, 0, 0), (4, 0, 0)]),
            (0, 1, [(0, 10, 0), (4, 10, 0)]),
            (1, 0, [(0, 20, 0), (4, 20, 0)]),
            (2, 0, [(0, 0, 9), (4, 0, 9)]),
        ]
    )
    return build_wire_edge_bundle(arrays)


def test_bvh_matches_brute_force() -> None:
    bundle = _three_wire_bundle()
    bvh = SegmentEdgeBvh(bundle, leaf_cap=2)
    queries = [
        (1.0, 0.0, 0.5),
        (3.0, 10.0, 0.2),
        (2.0, 19.0, 0.0),
        (2.0, 0.0, 8.5),
        (50.0, 50.0, 50.0),
        (1.0, 0.0, 0.0),
    ]
    for query in queries:
        edge_position, distance, q, t, evaluated = bvh.nearest_edge(query, 1e-9)
        key = (
            int(bundle.wire_id[edge_position]),
            int(bundle.segment_index[edge_position]),
            int(bundle.polyline_edge_index[edge_position]),
        )
        best_distance = None
        best_key = None
        for edge in range(bundle.edge_count):
            d, qq, tt = exact_point_to_segment_distance(
                query, bundle.a_xyz[edge], bundle.b_xyz[edge]
            )
            candidate_key = (
                int(bundle.wire_id[edge]),
                int(bundle.segment_index[edge]),
                int(bundle.polyline_edge_index[edge]),
            )
            if _better(d, candidate_key, best_distance, best_key, 1e-9):
                best_distance = d
                best_key = candidate_key
        assert key == best_key
        assert distance == pytest.approx(best_distance, abs=1e-12)
        assert evaluated >= 1


def test_bvh_tie_resolution_is_deterministic() -> None:
    # Two edges from different wires at identical distance: wire 0 must win.
    arrays = _w1_arrays(
        [
            (0, 0, [(0, 0, 2), (4, 0, 2)]),
            (1, 0, [(0, 0, 2), (4, 0, 2)]),
        ]
    )
    bundle = build_wire_edge_bundle(arrays)
    bvh = SegmentEdgeBvh(bundle, leaf_cap=2)
    for _ in range(2):
        edge_position, distance, q, t, _ = bvh.nearest_edge((2.0, 5.0, 2.0), 1e-9)
        assert int(bundle.wire_id[edge_position]) == 0
        assert distance == pytest.approx(5.0)
    # Same query must return the same order index every time.
    first_order = bvh.order.tolist()
    assert first_order == bvh.order.tolist()


def test_bvh_result_is_independent_of_edge_array_order() -> None:
    edges = [
        (0, 0, [(0, 0, 2), (4, 0, 2)]),
        (1, 0, [(0, 10, 2), (4, 10, 2)]),
        (2, 0, [(0, 20, 2), (4, 20, 2)]),
    ]
    first = SegmentEdgeBvh(build_wire_edge_bundle(_w1_arrays(edges)), leaf_cap=1)
    second = SegmentEdgeBvh(
        build_wire_edge_bundle(_w1_arrays(list(reversed(edges)))), leaf_cap=1
    )
    for query in [(2.0, 5.0, 2.0), (2.0, 15.0, 2.0), (2.0, 25.0, 2.0)]:
        e1, d1, _, _, _ = first.nearest_edge(query, 1e-9)
        e2, d2, _, _, _ = second.nearest_edge(query, 1e-9)
        key1 = (int(first.edges.wire_id[e1]), int(first.edges.segment_index[e1]))
        key2 = (int(second.edges.wire_id[e2]), int(second.edges.segment_index[e2]))
        assert key1 == key2
        assert d1 == pytest.approx(d2, abs=1e-12)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


def test_clearance_matches_brute_force_multi_wire(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    loaded = __import__(
        "algorithms.vegetation.v4_tree_artifacts", fromlist=["load_v4_tree_segmentation_artifact"]
    ).load_v4_tree_segmentation_artifact(v4_root)
    v4_result = loaded.result

    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
            (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
            (1, 0, _wire_segment((0.0, 9.0, 1.0), (10.0, 9.0, 1.0))),
        ],
    )
    result = run_tree_wire_clearance(
        v4_root, w1_npz, w1_json, project_root, config=default_tree_wire_clearance_config()
    )
    assert result.status is TreeWireClearanceStatus.VALID
    assert len(result.records) == len(v4_result.tree_records)

    bundle = build_wire_edge_bundle(
        _w1_arrays(
            [
                (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
                (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
                (1, 0, _wire_segment((0.0, 9.0, 1.0), (10.0, 9.0, 1.0))),
            ]
        )
    )
    expected = _brute_force_tree_clearance(v4_result, bundle, 1e-9)
    for record in result.records:
        tree_id = record.tree_id
        assert record.status is TreeWireClearanceRecordStatus.VALID
        dist, key, p, q, t, original_index = expected[tree_id]
        assert record.minimum_clearance_m == pytest.approx(dist, abs=1e-12)
        assert record.nearest_wire_id == key[0]
        assert record.w1_segment_index == key[1]
        assert record.polyline_edge_index == key[2]
        assert record.nearest_tree_original_index == key[3] == original_index
        assert np.allclose(record.nearest_tree_xyz_local_m, p, atol=1e-12)
        assert np.allclose(record.nearest_wire_xyz_local_m, q, atol=1e-12)
        assert record.edge_interpolation_t == pytest.approx(t, abs=1e-12)


def test_clearance_segment_interior_winner(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    # A wire spanning both trees through y = 2 at z = 3: every tree's nearest
    # approach is the perpendicular foot strictly inside a segment, proving
    # segment distance is used (never a wire-vertex-only distance).
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
            (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
        ],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert len(result.records) == 2
    for record in result.records:
        assert record.status is TreeWireClearanceRecordStatus.VALID
        assert 0.0 < record.edge_interpolation_t < 1.0
        assert record.nearest_wire_id == 0


def test_clearance_endpoint_winner(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    # A wire far to the right: the nearest approach is its start endpoint.
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((100.0, 8.0, 3.0), (104.0, 8.0, 3.0))),
        ],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    for record in result.records:
        if record.status is not TreeWireClearanceRecordStatus.VALID:
            continue
        assert record.edge_interpolation_t == pytest.approx(0.0)


def test_zero_length_edge_allowed(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, [(2.0, 2.0, 1.0), (2.0, 2.0, 1.0)]),
        ],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert result.status is TreeWireClearanceStatus.VALID
    for record in result.records:
        assert record.status is TreeWireClearanceRecordStatus.VALID
        assert record.edge_interpolation_t == pytest.approx(0.0)


def test_non_distance_ready_segment_is_excluded(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            # A non-ready segment that would win if it were included.
            (0, 0, _wire_segment((0.0, 2.0, 2.99), (4.0, 2.0, 2.99))),
            # The only ready segment, far away.
            (0, 1, _wire_segment((100.0, 100.0, 1.0), (104.0, 100.0, 1.0))),
        ],
        ready=[False, True],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    for record in result.records:
        if record.status is not TreeWireClearanceRecordStatus.VALID:
            continue
        assert record.w1_segment_index == 1  # the ready segment won, not seg 0


def test_original_index_preserved_and_geometry_consistent(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    loaded = __import__(
        "algorithms.vegetation.v4_tree_artifacts", fromlist=["load_v4_tree_segmentation_artifact"]
    ).load_v4_tree_segmentation_artifact(v4_root)
    v4_result = loaded.result
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
        ],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    index_to_xyz = {
        int(v4_result.original_index[position]): v4_result.xyz_local_m[position]
        for position in range(int(v4_result.original_index.shape[0]))
    }
    for record in result.records:
        assert record.status is TreeWireClearanceRecordStatus.VALID
        expected_point = index_to_xyz[record.nearest_tree_original_index]
        assert np.allclose(record.nearest_tree_xyz_local_m, expected_point, atol=1e-12)


def test_no_distance_ready_edge_result(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
        ],
        ready=[False],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert result.status is TreeWireClearanceStatus.NO_DISTANCE_READY_EDGE
    assert len(result.records) == 2
    assert all(
        record.status is TreeWireClearanceRecordStatus.NO_DISTANCE_READY_EDGE
        for record in result.records
    )
    assert all(record.minimum_clearance_m is None for record in result.records)


def test_clearance_quality_flags_preserved(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
        ],
        wire_statuses=[int(WireModelStatus.FALLBACK_MODEL)],
        segment_quality=[int(WireModelQualityFlag.FALLBACK_USED)],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    for record in result.records:
        assert record.status is TreeWireClearanceRecordStatus.VALID
        assert record.quality_flags & int(
            TreeWireClearanceQualityFlag.NEAREST_WIRE_NOT_CONFIRMED
        )
        assert record.quality_flags & int(
            TreeWireClearanceQualityFlag.NEAREST_SEGMENT_FALLBACK_OR_REVIEW
        )
        assert record.nearest_wire_status == int(WireModelStatus.FALLBACK_MODEL)
        assert record.nearest_segment_quality_flags == int(
            WireModelQualityFlag.FALLBACK_USED
        )


def test_confirmed_wire_has_zero_quality_flags(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    for record in result.records:
        assert record.quality_flags == 0


def test_candidate_and_evaluated_counts(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
            (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
        ],
    )
    result = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    for record in result.records:
        assert record.candidate_edge_count == 2
        assert record.evaluated_edge_count >= 1


def test_result_digest_deterministic_and_json_round_trip(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
            (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
        ],
    )
    first = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    second = run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert first.result_digest == second.result_digest
    assert first.to_dict() == second.to_dict()
    clone = TreeWireClearanceResult.from_dict(first.to_dict())
    assert clone == first
    assert clone.computed_result_digest() == first.result_digest


def test_origin_mismatch_fails_closed(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
        origin=(500_000.0, 4_200_000.0, 125.0),
    )
    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert excinfo.value.code is TreeWireClearanceErrorCode.ORIGIN_MISMATCH


def test_source_mismatch_fails_closed(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
        source_id="other-scene",
        manifest_source_id="other-scene",
    )
    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert excinfo.value.code is TreeWireClearanceErrorCode.SOURCE_MISMATCH


def test_manifest_source_mismatch_fails_closed(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
        w1_manifest_source_id="other-manifest",
    )
    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert excinfo.value.code is TreeWireClearanceErrorCode.SOURCE_MISMATCH


def test_invalid_a4_provenance_fails_closed(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
    )
    # Tamper the A4 NPZ after W1 publication: hashes must not match.
    a4_npz = project_root / "outputs" / "wires" / "scene-1_wire_instances.npz"
    a4_npz.write_bytes(b"tampered")
    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert excinfo.value.code is TreeWireClearanceErrorCode.W1_A4_PROVENANCE_FAILED


def test_project_root_is_required_and_confined(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, _project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
    )
    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(v4_root, w1_npz, w1_json, None)
    assert excinfo.value.code is TreeWireClearanceErrorCode.PROJECT_ROOT_INVALID

    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(v4_root, w1_npz, w1_json, tmp_path / "missing")
    assert excinfo.value.code is TreeWireClearanceErrorCode.PROJECT_ROOT_INVALID


def test_w1_load_failure_fails_closed(tmp_path: Path) -> None:
    v4_root = _make_v4_artifact(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
    )
    w1_npz.write_bytes(b"corrupted")
    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(v4_root, w1_npz, w1_json, project_root)
    assert excinfo.value.code is TreeWireClearanceErrorCode.W1_ARTIFACT_INVALID


def test_v4_artifact_invalid_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "not-an-artifact"
    root.mkdir(exist_ok=True)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
    )
    with pytest.raises(TreeWireClearanceError) as excinfo:
        run_tree_wire_clearance(root, w1_npz, w1_json, project_root)
    assert excinfo.value.code is TreeWireClearanceErrorCode.V4_ARTIFACT_INVALID


def test_bvh_leaf_cap_respected(tmp_path: Path) -> None:
    # Large edge bundle: leaf cap must bound the leaf size.
    curve = [(float(i), 0.0, float(i)) for i in range(200)]
    arrays = _w1_arrays([(0, 0, curve)])
    bundle = build_wire_edge_bundle(arrays)
    assert bundle.edge_count == 199
    bvh = SegmentEdgeBvh(bundle, leaf_cap=8)

    def leaf_sizes(node) -> list[int]:
        if node.leaf:
            return [node.hi - node.lo]
        return leaf_sizes(node.left) + leaf_sizes(node.right)

    assert max(leaf_sizes(bvh.root)) <= 8
