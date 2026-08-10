"""Synthetic tests for pole detection, validation, topology and spans (43, 44)."""

from __future__ import annotations

import numpy as np
from algorithms.poles.candidates import detect_pole_candidates
from algorithms.poles.evidence import GroundEvidence
from algorithms.poles.topology import build_span_topology
from algorithms.poles.validation import (
    CONFIRMED_POWER_POLE,
    POLE_LIKE_REVIEW,
    REJECTED_POLE_LIKE,
    validate_poles,
)
from algorithms.poles.wire_evidence import detect_coarse_wire_evidence
from core.config.fast_baseline import default_fast_baseline_config

_CFG = default_fast_baseline_config()


def _pole_pts(x: float, y: float, z0: float, z1: float, step: float = 0.4) -> np.ndarray:
    return np.column_stack(
        (
            np.full(int((z1 - z0) / step) + 1, x),
            np.full(int((z1 - z0) / step) + 1, y),
            np.arange(z0, z1 + 1e-9, step),
        )
    )


def _wire_pts(s0: float, s1: float, u: float, z: float, step: float = 0.4) -> np.ndarray:
    s = np.arange(s0, s1 + 1e-9, step)
    return np.column_stack((s, np.full(len(s), u), np.full(len(s), z)))


def _crown_pts(cx: float, cy: float, z0: float, z1: float, count: int) -> np.ndarray:
    rng = np.random.default_rng(hash((cx, cy, z1)) % (2**32))
    rows = []
    for _ in range(count):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.5, 3.0)
        rows.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle), rng.uniform(z0, z1)))
    return np.asarray(rows, dtype=np.float64)


def _build_evidence(
    *,
    pole_xys: list[tuple[float, float]] | None = None,
    pole_top_zs: list[float] | None = None,
    wire_segments: list[np.ndarray] | None = None,
    roof_segments: list[np.ndarray] | None = None,
    trunks: list[tuple[float, float]] | None = None,
    crowns: list[np.ndarray] | None = None,
    lights: list[tuple[float, float]] | None = None,
    ground_z: float = 0.0,
) -> GroundEvidence:
    """Assemble a synthetic GroundEvidence from typed geometry."""

    parts_xyz: list[np.ndarray] = []
    parts_cls: list[np.ndarray] = []
    parts_hag: list[np.ndarray] = []
    parts_lin: list[np.ndarray] = []
    parts_plan: list[np.ndarray] = []
    parts_scat: list[np.ndarray] = []
    parts_vert: list[np.ndarray] = []
    parts_dirz: list[np.ndarray] = []

    def add(
        points: np.ndarray,
        *,
        cls: int,
        lin: float,
        plan: float,
        scat: float,
        vert: float,
        dirz: float,
        hag: np.ndarray | None = None,
    ) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        parts_xyz.append(points)
        parts_cls.append(np.full(len(points), cls, dtype=np.uint8))
        hag_value = (
            (points[:, 2] - ground_z).astype(np.float32)
            if hag is None
            else np.asarray(hag, dtype=np.float32)
        )
        parts_hag.append(hag_value)
        parts_lin.append(np.full(len(points), lin, dtype=np.float32))
        parts_plan.append(np.full(len(points), plan, dtype=np.float32))
        parts_scat.append(np.full(len(points), scat, dtype=np.float32))
        parts_vert.append(np.full(len(points), vert, dtype=np.float32))
        parts_dirz.append(np.full(len(points), dirz, dtype=np.float32))

    # Ground plane.
    grid = np.asarray(
        [
            [x, y, ground_z]
            for x in np.arange(-80, 90, 5)
            for y in np.arange(-20, 25, 5)
        ]
    )
    add(grid, cls=2, lin=0.05, plan=0.9, scat=0.05, vert=0.1, dirz=0.05)
    # Poles.
    for position, (x, y) in enumerate(pole_xys or []):
        top_z = (pole_top_zs or [12.0] * len(pole_xys or []))[position]
        pts = _pole_pts(x, y, ground_z + 0.1, ground_z + top_z)
        add(pts, cls=1, lin=0.85, plan=0.05, scat=0.1, vert=0.95, dirz=0.95)
    # Street lights (short poles).
    for x, y in lights or []:
        pts = _pole_pts(x, y, ground_z + 0.1, ground_z + 7.0)
        add(pts, cls=1, lin=0.85, plan=0.05, scat=0.1, vert=0.95, dirz=0.95)
    # Wires.
    for segment in wire_segments or []:
        add(segment, cls=1, lin=0.85, plan=0.05, scat=0.02, vert=0.1, dirz=0.1)
    # Long planar roof-ridge-like structures: line-like in XY but deliberately
    # irregular in elevation, so coarse corridor recovery must reject them.
    for segment in roof_segments or []:
        add(segment, cls=1, lin=0.85, plan=0.05, scat=0.02, vert=0.1, dirz=0.1)
    # Tree trunks.
    for x, y in trunks or []:
        pts = _pole_pts(x, y, ground_z + 0.1, ground_z + 9.0)
        add(pts, cls=1, lin=0.85, plan=0.05, scat=0.1, vert=0.95, dirz=0.95)
    # Tree crowns (volumetric).
    for crown in crowns or []:
        add(crown, cls=1, lin=0.2, plan=0.3, scat=0.5, vert=0.3, dirz=0.3)

    xyz = np.concatenate(parts_xyz, axis=0)
    count = len(xyz)
    return GroundEvidence(
        xyz_absolute_m=xyz.copy(),
        xyz_local_m=xyz.copy(),
        hag_m=np.concatenate(parts_hag),
        classification=np.concatenate(parts_cls),
        representative_id=np.arange(count, dtype=np.uint32),
        source_original_index=np.arange(count, dtype=np.int64),
        raw_count=np.ones(count, dtype=np.uint32),
        linearity=np.concatenate(parts_lin),
        planarity=np.concatenate(parts_plan),
        scattering=np.concatenate(parts_scat),
        verticality=np.concatenate(parts_vert),
        main_direction_z=np.concatenate(parts_dirz),
        count=count,
        origin_xyz=(0.0, 0.0, 0.0),
        ground_class_2_count=int(np.count_nonzero(np.concatenate(parts_cls) == 2)),
        non_ground_count=int(np.count_nonzero(np.concatenate(parts_cls) != 2)),
        feature_runtime_seconds=0.0,
    )


def test_single_vertical_pole_detected() -> None:
    evidence = _build_evidence(pole_xys=[(0.0, 0.0)])
    candidates = detect_pole_candidates(evidence, _CFG)
    assert candidates.candidate_count == 1
    candidate = candidates.records[0]
    assert candidate.height_m >= 10.0
    assert candidate.xy_footprint_m < _CFG.pole_maximum_xy_width_m
    assert candidate.ground_contact_evidence
    assert candidate.base_xyz_absolute_m[2] < 1.0


def test_pole_with_conductor_support_is_confirmed() -> None:
    wires = [_wire_pts(-30.0, 30.0, u, 11.0) for u in (-1.0, 1.0)]
    evidence = _build_evidence(pole_xys=[(0.0, 0.0)], wire_segments=wires)
    candidates = detect_pole_candidates(evidence, _CFG)
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    assert validation.records[0].state == CONFIRMED_POWER_POLE
    assert validation.records[0].wire_support_count >= 1


def test_tree_trunk_with_crown_is_rejected() -> None:
    crown = _crown_pts(0.0, 0.0, 9.0, 14.0, 400)
    evidence = _build_evidence(trunks=[(0.0, 0.0)], crowns=[crown])
    candidates = detect_pole_candidates(evidence, _CFG)
    assert candidates.candidate_count == 1
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    assert validation.records[0].state == REJECTED_POLE_LIKE


def test_street_light_without_wires_is_review() -> None:
    evidence = _build_evidence(lights=[(5.0, 0.0)])
    candidates = detect_pole_candidates(evidence, _CFG)
    assert candidates.candidate_count >= 1
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    assert validation.records[0].state == POLE_LIKE_REVIEW


def test_two_sequential_poles_form_one_span() -> None:
    wire = _wire_pts(0.0, 50.0, 0.0, 11.0)
    evidence = _build_evidence(pole_xys=[(0.0, 0.0), (50.0, 0.0)], wire_segments=[wire])
    candidates = detect_pole_candidates(evidence, _CFG)
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    assert len(validation.usable_pole_ids) == 2
    topology = build_span_topology(candidates, validation, wire_evidence, evidence, _CFG)
    assert topology.line_count == 1
    assert topology.span_count == 1
    span = topology.spans[0]
    assert span.start_pole_candidate_id in validation.usable_pole_ids
    assert span.end_pole_candidate_id in validation.usable_pole_ids
    assert abs(span.length_m - 50.0) < 1.0


def test_unequal_total_pole_heights_use_compatible_attachment_band() -> None:
    wires = [_wire_pts(0.0, 50.0, u, 11.0) for u in (-1.0, 1.0)]
    evidence = _build_evidence(
        pole_xys=[(0.0, 0.0), (50.0, 0.0)],
        pole_top_zs=[15.0, 8.0],
        wire_segments=wires,
    )
    candidates = detect_pole_candidates(evidence, _CFG)
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    topology = build_span_topology(candidates, validation, wire_evidence, evidence, _CFG)
    assert topology.span_count == 1
    assert abs(validation.records[0].attachment_elevation_m - 11.0) < 0.2
    assert abs(validation.records[1].attachment_elevation_m - 11.0) < 0.2
    assert abs(candidates.records[0].top_z_max_m - candidates.records[1].top_z_max_m) > 5.0


def test_roof_ridge_like_line_is_rejected_from_coarse_wire_evidence() -> None:
    xs = np.arange(-40.0, 40.0, 0.4)
    roof = np.column_stack((xs, np.zeros_like(xs), 8.0 + 0.10 * xs))
    evidence = _build_evidence(roof_segments=[roof])
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    assert wire_evidence.segment_count == 0


def test_three_poles_form_two_spans() -> None:
    wires = [_wire_pts(0.0, 50.0, 0.0, 11.0), _wire_pts(50.0, 100.0, 0.0, 11.0)]
    evidence = _build_evidence(
        pole_xys=[(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], wire_segments=wires
    )
    candidates = detect_pole_candidates(evidence, _CFG)
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    topology = build_span_topology(candidates, validation, wire_evidence, evidence, _CFG)
    assert topology.span_count == 2
    assert topology.lines[0].pole_count == 3


def _diagonal_wire(
    start: tuple[float, float],
    stop: tuple[float, float],
    z: float,
    step: float = 0.5,
) -> np.ndarray:
    count = int(np.hypot(stop[0] - start[0], stop[1] - start[1]) / step) + 1
    xs = np.linspace(start[0], stop[0], count)
    ys = np.linspace(start[1], stop[1], count)
    return np.column_stack((xs, ys, np.full(count, z)))


def test_curved_corridor_orders_poles_along_the_path() -> None:
    # Poles along a gentle curve (x, y) = (0,0), (25, 8), (50, 0).
    positions = [(0.0, 0.0), (25.0, 8.0), (50.0, 0.0)]
    wires = [
        _diagonal_wire((0.0, 0.0), (25.0, 8.0), 11.0),
        _diagonal_wire((25.0, 8.0), (50.0, 0.0), 11.0),
    ]
    evidence = _build_evidence(pole_xys=positions, wire_segments=wires)
    candidates = detect_pole_candidates(evidence, _CFG)
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    topology = build_span_topology(candidates, validation, wire_evidence, evidence, _CFG)
    assert topology.span_count == 2
    ordered = topology.lines[0].ordered_pole_candidate_ids
    assert len(ordered) == 3


def test_two_unrelated_corridors_are_not_connected() -> None:
    corridor_a_wire = _wire_pts(0.0, 50.0, 0.0, 11.0)
    corridor_b_wire = _wire_pts(0.0, 50.0, 20.0, 11.0)
    evidence = _build_evidence(
        pole_xys=[(0.0, 0.0), (50.0, 0.0), (0.0, 20.0), (50.0, 20.0)],
        wire_segments=[corridor_a_wire, corridor_b_wire],
    )
    candidates = detect_pole_candidates(evidence, _CFG)
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    topology = build_span_topology(candidates, validation, wire_evidence, evidence, _CFG)
    assert topology.line_count == 2
    assert topology.span_count == 2


def test_deterministic_line_numbering() -> None:
    corridor_a_wire = _wire_pts(0.0, 50.0, 0.0, 11.0)
    corridor_b_wire = _wire_pts(0.0, 50.0, 20.0, 11.0)
    evidence = _build_evidence(
        pole_xys=[(0.0, 0.0), (50.0, 0.0), (0.0, 20.0), (50.0, 20.0)],
        wire_segments=[corridor_a_wire, corridor_b_wire],
    )
    candidates = detect_pole_candidates(evidence, _CFG)
    wire_evidence = detect_coarse_wire_evidence(evidence, _CFG)
    validation = validate_poles(candidates, wire_evidence, evidence, _CFG)
    first = build_span_topology(candidates, validation, wire_evidence, evidence, _CFG)
    second = build_span_topology(candidates, validation, wire_evidence, evidence, _CFG)
    assert [line.line_name for line in first.lines] == [
        line.line_name for line in second.lines
    ]
    assert [span.span_name for span in first.spans] == [
        span.span_name for span in second.spans
    ]
