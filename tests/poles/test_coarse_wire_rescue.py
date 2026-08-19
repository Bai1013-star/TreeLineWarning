"""Dual-path coarse wire router: span-aligned rescue (PATH B).

Regression coverage for the coarse Wire blocker fix:

* A globally contaminated connected component fails the strict PATH A component
  gates and is rejected whole; PATH B recovers the local wire-like subset and
  the existing conservative Wire V2 confirms the real conductor (section 16).
* High-recall rescue may route short linear clutter, but V2 rejects it while
  confirming the full-span conductor (false-positive safety, section 15).
* Multiple parallel conductors are preserved as separate V2 strands
  (multi-conductor, section 17).
* Coarse routing never scans the raw source LAS (performance, section 18).
* Unconfirmed routed candidates never permanently remove vegetation/tree
  membership; only exact FINAL confirmed wire members are subtracted
  (confirmed-only vegetation semantics, section 9).
* ``bounded_supported_slice_ratio`` is always in ``[0, 1]`` while the legacy
  V2 acceptance fields keep their decision semantics (section 12).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from algorithms.poles.evidence import GroundEvidence
from algorithms.poles.topology import SpanRecord
from algorithms.poles.wire_evidence import (
    ROUTE_SPAN_ALIGNED_RESCUE,
    ROUTE_STRICT_COMPONENT,
    detect_coarse_wire_evidence,
    detect_coarse_wire_evidence_with_rescue,
)
from algorithms.spans.wire_v2 import CONFIRMED, reconstruct_slice_tracks
from core.config.fast_baseline import default_fast_baseline_config

_CFG = default_fast_baseline_config()


def _span(start=(0.0, 0.0), end=(50.0, 0.0), top_z=12.0) -> SpanRecord:
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    direction = (
        float((end[0] - start[0]) / length),
        float((end[1] - start[1]) / length),
        0.0,
    )
    return SpanRecord(
        line_id=0,
        line_name="Line01",
        span_id=0,
        span_name="Line01-Span001",
        start_pole_candidate_id=0,
        end_pole_candidate_id=1,
        start_xyz_absolute_m=(float(start[0]), float(start[1]), 0.0),
        end_xyz_absolute_m=(float(end[0]), float(end[1]), 0.0),
        start_pole_top_z_m=top_z,
        end_pole_top_z_m=top_z,
        length_m=length,
        direction_xy=direction,
        half_width_m=9.0,
        halo_m=6.0,
        core_bounds_min_xyz=(float(start[0]), -9.0, 0.0),
        core_bounds_max_xyz=(float(end[0]), 9.0, 20.0),
        halo_bounds_min_xyz=(float(start[0]) - 6.0, -9.0, 0.0),
        halo_bounds_max_xyz=(float(end[0]) + 6.0, 9.0, 20.0),
        quality="CONFIRMED_SPAN",
    )


def _line(s0: float, s1: float, u: float, z: float, step: float = 0.4) -> np.ndarray:
    """A span-aligned line: x=s, y=u, z=z (span runs along x)."""
    s = np.arange(s0, s1 + 1e-9, step)
    return np.column_stack((s, np.full(len(s), u), np.full(len(s), z)))


def _sheet(
    x0: float,
    x1: float,
    u_lo: float,
    u_hi: float,
    z: float,
    step: float = 1.0,
) -> np.ndarray:
    """A horizontal low-scattering sheet (canopy/roof-like contamination)."""
    rows = []
    for x in np.arange(x0, x1 + 1e-9, step):
        for u in np.arange(u_lo, u_hi + 1e-9, step):
            rows.append((x, u, z))
    return np.asarray(rows, dtype=np.float64)


def _build_evidence(*groups: np.ndarray) -> GroundEvidence:
    """Assemble a GroundEvidence from geometry groups (ground + given groups).

    Each group is ``(n, 3)`` XYZ with non-ground class 2 assignment for the
    ground plane.  All non-ground points receive the low-scattering, horizontal
    coarse-wire feature profile; the ground plane is class 2.
    """

    parts_xyz: list[np.ndarray] = []
    parts_cls: list[np.ndarray] = []
    parts_scat: list[np.ndarray] = []
    parts_dirz: list[np.ndarray] = []
    ground = np.asarray(
        [[x, y, 0.0] for x in np.arange(-20, 80, 5) for y in np.arange(-20, 25, 5)]
    )
    parts_xyz.append(ground)
    parts_cls.append(np.full(len(ground), 2, dtype=np.uint8))
    parts_scat.append(np.zeros(len(ground), dtype=np.float32))
    parts_dirz.append(np.zeros(len(ground), dtype=np.float32))
    for group in groups:
        points = np.asarray(group, dtype=np.float64).reshape(-1, 3)
        parts_xyz.append(points)
        parts_cls.append(np.ones(len(points), dtype=np.uint8))
        parts_scat.append(np.full(len(points), 0.03, dtype=np.float32))
        parts_dirz.append(np.full(len(points), 0.10, dtype=np.float32))
    xyz = np.concatenate(parts_xyz, axis=0)
    cls = np.concatenate(parts_cls, axis=0)
    count = len(xyz)
    hag = np.maximum(xyz[:, 2], 0.0).astype(np.float32)
    lin = np.zeros(count, dtype=np.float32)
    plan = np.zeros(count, dtype=np.float32)
    vert = np.zeros(count, dtype=np.float32)
    non_ground = cls != 2
    lin[non_ground] = 0.35
    plan[non_ground] = 0.35
    vert[non_ground] = 0.25
    return GroundEvidence(
        xyz_absolute_m=xyz.copy(),
        xyz_local_m=xyz.copy(),
        hag_m=hag,
        classification=cls,
        representative_id=np.arange(count, dtype=np.uint32),
        source_original_index=np.arange(count, dtype=np.int64),
        raw_count=np.ones(count, dtype=np.uint32),
        linearity=lin,
        planarity=plan,
        scattering=np.concatenate(parts_scat),
        verticality=vert,
        main_direction_z=np.concatenate(parts_dirz),
        count=count,
        origin_xyz=(0.0, 0.0, 0.0),
        ground_class_2_count=int(np.count_nonzero(cls == 2)),
        non_ground_count=int(np.count_nonzero(cls != 2)),
        feature_runtime_seconds=0.0,
    )


def _routed_v2_tracks(evidence: GroundEvidence, span: SpanRecord, mask: np.ndarray):
    """Run the existing V2 slice-track confirmation on routed representatives."""
    rows = np.flatnonzero(mask)
    s, u, z = _span_suz(span, evidence)
    return reconstruct_slice_tracks(
        rows, s, u, z, evidence.raw_count.astype(np.int64), span.length_m, _CFG
    )


def _span_suz(span: SpanRecord, evidence: GroundEvidence):
    from algorithms.poles.topology import span_to_suz

    return span_to_suz(span, evidence.xyz_absolute_m, (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Section 16: contaminated-component rescue + V2 confirmation
# ---------------------------------------------------------------------------

def test_contaminated_component_is_rescued_and_confirmed() -> None:
    span = _span()
    conductor = _line(0.0, 50.0, 0.0, 12.0)
    # A horizontal low-scattering sheet connected to the conductor within the
    # 1 m component grid (z 13.5 is one grid cell above z 12) and spread over
    # |u| <= 8: the component-global transverse width blows past the gate.
    contamination = _sheet(20.0, 30.0, -8.0, 8.0, 13.5)
    evidence = _build_evidence(conductor, contamination)

    strict = detect_coarse_wire_evidence(evidence, _CFG)
    # PATH A must REJECT the whole contaminated component (transverse_width > 8).
    assert strict.segment_count == 0
    routed = detect_coarse_wire_evidence_with_rescue(
        evidence, _CFG, span, origin_xyz=(0.0, 0.0, 0.0)
    )
    assert routed.segment_count == 0  # strict path unchanged
    # PATH B rescues the local wire-like subset spanning the conductor.
    assert int(np.count_nonzero(routed.rescued_mask)) > 0
    s, u, z = _span_suz(span, evidence)
    rescued_s = s[routed.rescued_mask]
    assert float(np.min(rescued_s)) < 2.0
    assert float(np.max(rescued_s)) > 48.0
    # The existing conservative Wire V2 confirms the full-span conductor.
    tracks = _routed_v2_tracks(evidence, span, routed.routed_mask)
    confirmed = [t for t in tracks if t.status == CONFIRMED]
    assert len(confirmed) >= 1
    best = max(confirmed, key=lambda t: t.coverage_ratio)
    assert best.coverage_ratio > 0.8
    assert best.supported_s_max_m - best.supported_s_min_m > 40.0
    # The contamination sheet (|u| up to 8) is NOT published as a conductor.
    wide = [t for t in tracks if abs(t.median_u_m) > 5.0 and t.status == CONFIRMED]
    assert len(wide) == 0


# ---------------------------------------------------------------------------
# Section 15: false-positive safety — short fragment is not confirmed
# ---------------------------------------------------------------------------

def test_short_fragment_and_clutter_are_rejected_while_conductor_confirmed() -> None:
    span = _span()
    conductor = _line(0.0, 50.0, 0.0, 12.0)
    branch = _line(20.0, 26.0, 2.0, 10.0, step=0.4)  # short linear fragment
    clutter = _sheet(20.0, 30.0, -8.0, 8.0, 13.5)
    evidence = _build_evidence(conductor, branch, clutter)

    routed = detect_coarse_wire_evidence_with_rescue(
        evidence, _CFG, span, origin_xyz=(0.0, 0.0, 0.0)
    )
    # High-recall rescue may route extra short linear evidence...
    assert int(np.count_nonzero(routed.rescued_mask)) > 0
    tracks = _routed_v2_tracks(evidence, span, routed.routed_mask)
    confirmed = [t for t in tracks if t.status == CONFIRMED]
    # ...but V2 confirms the full-span conductor and rejects the short clutter.
    assert len(confirmed) == 1
    strand = confirmed[0]
    assert abs(strand.median_u_m) < 1.0
    assert strand.coverage_ratio > 0.8
    for track in tracks:
        if track.status == CONFIRMED:
            assert abs(track.median_u_m) < 1.0
    short = [t for t in tracks if abs(t.median_u_m - 2.0) < 0.5]
    assert all(t.status != CONFIRMED for t in short)
    wide = [t for t in tracks if abs(t.median_u_m) > 5.0]
    assert all(t.status != CONFIRMED for t in wide)


# ---------------------------------------------------------------------------
# Section 17: multi-conductor rescue preserves separation
# ---------------------------------------------------------------------------

def test_multi_conductor_rescue_preserves_parallel_strands() -> None:
    span = _span()
    conductor_a = _line(0.0, 50.0, -1.5, 12.0)
    conductor_b = _line(0.0, 50.0, 1.5, 12.0)
    contamination = _sheet(20.0, 30.0, -8.0, 8.0, 13.5)
    evidence = _build_evidence(conductor_a, conductor_b, contamination)

    strict = detect_coarse_wire_evidence(evidence, _CFG)
    assert strict.segment_count == 0  # contaminated component rejected whole
    routed = detect_coarse_wire_evidence_with_rescue(
        evidence, _CFG, span, origin_xyz=(0.0, 0.0, 0.0)
    )
    tracks = _routed_v2_tracks(evidence, span, routed.routed_mask)
    confirmed = [t for t in tracks if t.status == CONFIRMED]
    # PATH B does NOT collapse the two conductors into one routed component;
    # the existing V2 separates/confirms them as distinct strands.
    median_us = sorted(round(t.median_u_m, 1) for t in confirmed)
    assert len(median_us) == 2
    assert median_us[0] < -1.0
    assert median_us[1] > 1.0


# ---------------------------------------------------------------------------
# Section 18: performance — coarse routing never scans the raw source
# ---------------------------------------------------------------------------

def test_coarse_routing_never_scans_raw_source(monkeypatch) -> None:
    span = _span()
    conductor = _line(0.0, 50.0, 0.0, 12.0)
    contamination = _sheet(20.0, 30.0, -8.0, 8.0, 13.5)
    evidence = _build_evidence(conductor, contamination)

    def _forbid_raw_source(*_args, **_kwargs):  # pragma: no cover - guard
        raise AssertionError("COARSE_ROUTING_SCANNED_RAW_SOURCE")

    monkeypatch.setattr("laspy.open", _forbid_raw_source)
    started = time.perf_counter()
    routed = detect_coarse_wire_evidence_with_rescue(
        evidence, _CFG, span, origin_xyz=(0.0, 0.0, 0.0)
    )
    runtime_seconds = time.perf_counter() - started
    # Representative-level routing only: RAW_SOURCE_POINTS_SCANNED = 0 and the
    # whole routing runs on the already-materialized representative evidence.
    assert int(np.count_nonzero(routed.routed_mask)) > 0
    assert runtime_seconds >= 0.0  # recorded; no arbitrary timing threshold
    # The routing input is the representative set (not the raw source).
    assert evidence.count > 0
    assert routed.rescue_diagnostics["rescue_representative_count"] > 0


# ---------------------------------------------------------------------------
# Section 9: confirmed-only vegetation/tree exclusion semantics
# ---------------------------------------------------------------------------

def test_routed_candidates_are_not_confirmed_wire_members(tmp_path: Path) -> None:
    span = _span()
    conductor = _line(0.0, 50.0, 0.0, 12.0)
    contamination = _sheet(20.0, 30.0, -8.0, 8.0, 13.5)
    evidence = _build_evidence(conductor, contamination)
    routed = detect_coarse_wire_evidence_with_rescue(
        evidence, _CFG, span, origin_xyz=(0.0, 0.0, 0.0)
    )
    prov = routed.route_provenance_by_representative
    strict_ids = set(np.flatnonzero(prov == ROUTE_STRICT_COMPONENT).tolist())
    rescue_ids = set(np.flatnonzero(prov == ROUTE_SPAN_ALIGNED_RESCUE).tolist())
    # Routing provenance is disjoint and covers exactly the routed mask.
    assert strict_ids.isdisjoint(rescue_ids)
    routed_ids = set(np.flatnonzero(routed.routed_mask).tolist())
    assert routed_ids == strict_ids | rescue_ids

    # Only exact FINAL confirmed source members are subtracted from vegetation.
    # Unconfirmed routed candidates (PATH A or PATH B) are NOT subtracted.
    from algorithms.spans.region import subset_raw_region_excluding_source_indices
    from algorithms.vegetation.coarse_to_fine import _RawRegionWriter

    writer = _RawRegionWriter(Path(tmp_path) / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        evidence.xyz_local_m,
        evidence.source_original_index,
        evidence.classification,
    )
    veg_region = writer.close()
    confirmed_wire = evidence.source_original_index[routed.routed_mask][:10]
    cleaned = subset_raw_region_excluding_source_indices(
        veg_region, confirmed_wire, Path(tmp_path) / "cleaned"
    )
    kept = np.asarray(cleaned.original_index(), dtype=np.int64).copy()
    # Confirmed members removed...
    assert len(np.intersect1d(kept, confirmed_wire)) == 0
    # ...but every OTHER routed (unconfirmed) candidate remains.
    routed_src = np.asarray(
        evidence.source_original_index[routed.routed_mask], dtype=np.int64
    )
    remaining_routed = np.setdiff1d(routed_src, confirmed_wire)
    assert len(np.intersect1d(kept, remaining_routed)) == len(remaining_routed)


# ---------------------------------------------------------------------------
# Section 12: bounded supported-slice diagnostic (acceptance semantics frozen)
# ---------------------------------------------------------------------------

def test_bounded_supported_slice_ratio_is_always_in_unit_interval() -> None:
    # A short span-aligned line whose inlier s-domain crosses 4 slice bins but
    # whose (s_max - s_min) undercounts the bins: legacy ratio > 1, bounded <= 1.
    span = _span()
    line = _line(0.6, 3.4, 0.0, 12.0, step=0.2)
    evidence = _build_evidence(line)
    rows = np.flatnonzero(evidence.classification != 2)
    s, u, z = _span_suz(span, evidence)
    tracks = reconstruct_slice_tracks(
        rows, s, u, z, evidence.raw_count.astype(np.int64), span.length_m, _CFG
    )
    assert len(tracks) >= 1
    for track in tracks:
        # New bounded diagnostic is always in [0, 1].
        assert 0.0 <= track.bounded_supported_slice_ratio <= 1.0 + 1e-9
        assert track.unique_supported_slice_count >= 0
        # The legacy fields are still present and retain their decision role.
        assert isinstance(track.supported_slice_count, int)
        assert track.supported_slice_ratio > 0.0
    # The legacy bookkeeping bug (ratio > 1) is preserved for diagnosis while
    # the corrected bounded ratio reports <= 1 on the same track.
    assert any(track.supported_slice_ratio > 1.0 for track in tracks)
    assert all(track.bounded_supported_slice_ratio <= 1.0 for track in tracks)


def test_full_pipeline_v2_confirms_rescued_conductor(tmp_path: Path) -> None:
    """End-to-end: rescue reps -> RawRegion -> run_span_wire_pipeline CONFIRMED."""
    from algorithms.spans.wire import run_span_wire_pipeline
    from algorithms.vegetation.coarse_to_fine import _RawRegionWriter

    span = _span()
    conductor = _line(0.0, 50.0, 0.0, 12.0)
    contamination = _sheet(20.0, 30.0, -8.0, 8.0, 13.5)
    evidence = _build_evidence(conductor, contamination)
    routed = detect_coarse_wire_evidence_with_rescue(
        evidence, _CFG, span, origin_xyz=(0.0, 0.0, 0.0)
    )
    rows = np.flatnonzero(routed.routed_mask)
    writer = _RawRegionWriter(Path(tmp_path) / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        evidence.xyz_local_m[rows],
        evidence.source_original_index[rows],
        evidence.classification[rows],
    )
    region = writer.close()
    result = run_span_wire_pipeline(
        span,
        region,
        _CFG,
        origin_xyz=(0.0, 0.0, 0.0),
        output_dir=Path(tmp_path) / "out",
    )
    assert result.strand_count == 1
    strand = result.strands[0]
    assert abs(strand.median_u_m) < 1.0
    assert strand.coverage_ratio > 0.8
    assert result.confirmed_member_source_index.shape[0] > 0
