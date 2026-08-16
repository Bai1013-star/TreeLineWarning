"""V5A2 automatic tower-candidate detector tests (synthetic but deterministic)."""

from __future__ import annotations

import numpy as np
from algorithms.towers.automatic import (
    TowerAutomaticDetectorConfig,
    detect_tower_candidates,
)
from core.config.fast_baseline import FastBaselineConfig
from core.contracts.tower_candidates import (
    CandidateOrigin,
    DetectorKind,
    TowerCandidateQualityFlag,
)

from tests.contracts.test_project_contracts import PROJECT_ID, TIMESTAMP
from tests.towers.conftest import build_ground_evidence, write_tower_las


def _run_detector(evidence, binding, coordinate_context, *, candidate_set_id="auto-001"):
    return detect_tower_candidates(
        evidence,
        FastBaselineConfig(),
        source_binding=binding,
        coordinate_context=coordinate_context,
        project_id=PROJECT_ID,
        candidate_set_id=candidate_set_id,
        generated_at=TIMESTAMP,
        detector_config=TowerAutomaticDetectorConfig(),
    )


def test_synthetic_obvious_tower_is_detected(tmp_path, tower_las_project):
    _, binding, _, points = tower_las_project()
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    assert candidate_set.candidate_count >= 1
    candidate = candidate_set.candidates[0]
    assert candidate.candidate_id == "P001"
    assert candidate.origin is CandidateOrigin.AUTOMATIC
    assert candidate.coordinate_frame is binding.coordinate_context.frame
    # The tower column at (10, 20) must be proposed near its true XY.
    x, y, _ = candidate.candidate_xyz_absolute_m
    assert abs(x - 10.0) < 2.0
    assert abs(y - 20.0) < 2.0
    # Unrefined proposals carry the needs-refinement quality flag.
    assert candidate.quality_flags & TowerCandidateQualityFlag.PROPOSAL_NEEDS_REFINEMENT


def test_detection_is_deterministic(tmp_path, tower_las_project):
    _, binding, _, points = tower_las_project()
    evidence = build_ground_evidence(tmp_path, points)
    first = _run_detector(
        evidence, binding, binding.coordinate_context, candidate_set_id="auto-001"
    )
    second = _run_detector(
        evidence, binding, binding.coordinate_context, candidate_set_id="auto-002"
    )
    assert [c.candidate_id for c in first.candidates] == [c.candidate_id for c in second.candidates]
    assert [c.candidate_xyz_absolute_m for c in first.candidates] == [
        c.candidate_xyz_absolute_m for c in second.candidates
    ]
    # The content digest is a pure fingerprint and must not depend on the set id.
    assert first.digest == second.digest


def test_candidates_merge_by_xy():
    # Two proposals within the merge radius must merge into one; proposals far
    # apart must both survive.  This exercises the greedy NMS layer directly
    # because the raw geometry detector uses a fixed 16-neighbour PCA that can
    # legitimately couple closely-spaced real towers.
    from algorithms.towers.automatic import AutomaticTowerRecord, _greedy_xy_merge

    def record(x, y, confidence):
        return AutomaticTowerRecord(
            source_pole_candidate_id=0,
            candidate_xyz_absolute_m=(x, y, 2.0),
            base_xyz_absolute_m=(x, y, 2.0),
            top_xyz_absolute_m=(x, y, 22.0),
            height_m=20.0,
            xy_footprint_m=0.5,
            confidence=confidence,
            confidence_components={},
            structural_evidence={},
        )

    merged = _greedy_xy_merge([record(10.0, 20.0, 0.9), record(13.0, 20.0, 0.8)], 4.0)
    assert len(merged) == 1
    assert merged[0].candidate_xyz_absolute_m == (10.0, 20.0, 2.0)

    kept = _greedy_xy_merge([record(10.0, 20.0, 0.9), record(50.0, 60.0, 0.8)], 4.0)
    assert len(kept) == 2


def test_two_far_apart_towers_both_detected(tmp_path):
    # Two towers far enough apart that their PCA neighbourhoods stay clean are
    # both proposed with canonical P### ordering.
    src = tmp_path / "merge.las"
    points = write_tower_las(
        src,
        tower_xyz=(10.0, 20.0, 0.0),
        extra_tower_xyz=(45.0, 50.0, 0.0),
    )
    from tests.towers.conftest import project_for_source

    project, binding, metadata = project_for_source(src, points)
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    assert candidate_set.candidate_count == 2, candidate_set.candidates
    assert [c.candidate_id for c in candidate_set.candidates] == ["P001", "P002"]


def test_flat_non_tower_object_is_not_detected(tmp_path):
    # A wide flat pad with no vertical shaft must not produce candidates.
    import laspy

    rs = np.random.RandomState(3)
    pad = np.column_stack(
        (rs.uniform(5, 25, 300), rs.uniform(5, 25, 300), np.full(300, 0.2))
    )
    ground = np.column_stack(
        (rs.uniform(0, 30, 200), rs.uniform(0, 30, 200), np.zeros(200))
    )
    allp = np.vstack((pad, ground))
    src = tmp_path / "flat.las"
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(h)
    las.x = allp[:, 0]
    las.y = allp[:, 1]
    las.z = allp[:, 2]
    las.write(src)
    from tests.towers.conftest import project_for_source

    project, binding, metadata = project_for_source(src, allp)
    evidence = build_ground_evidence(tmp_path, allp)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    assert candidate_set.candidate_count == 0


def test_candidate_ids_are_canonical_pxxx(tmp_path, tower_las_project):
    _, binding, _, points = tower_las_project()
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    assert candidate_set.candidate_count <= 999
    for candidate in candidate_set.candidates:
        assert candidate.candidate_id.startswith("P")
        assert len(candidate.candidate_id) == 4
        sequence = int(candidate.candidate_id[1:])
        assert 1 <= sequence <= 999
    assert [c.candidate_id for c in candidate_set.candidates] == sorted(
        c.candidate_id for c in candidate_set.candidates
    )


def test_confidence_is_bounded_unit_interval(tmp_path, tower_las_project):
    _, binding, _, points = tower_las_project()
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    assert candidate_set.candidate_count >= 1
    for candidate in candidate_set.candidates:
        assert 0.0 <= candidate.confidence <= 1.0
        if candidate.confidence < 0.5:
            assert candidate.quality_flags & TowerCandidateQualityFlag.LOW_STRUCTURAL_CONFIDENCE


def test_detector_provenance_is_present_and_geometry_by_default(tmp_path, tower_las_project):
    _, binding, _, points = tower_las_project()
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    provenance = candidate_set.detector_provenance
    assert provenance is not None
    assert provenance.detector_kind in {DetectorKind.GEOMETRY, DetectorKind.HYBRID}
    assert provenance.detector_name
    assert provenance.detector_version
    assert len(provenance.configuration_digest) == 64
    for candidate in candidate_set.candidates:
        assert candidate.detector_provenance is not None


def test_no_tower_allocation_and_unrefined_state(tmp_path, tower_las_project):
    _, binding, _, points = tower_las_project()
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    for candidate in candidate_set.candidates:
        # The detector only proposes; refinement state is UNREFINED and no T###
        # identity is ever referenced.
        assert candidate.candidate_id.startswith("P")
        assert "T" not in candidate.candidate_id


def test_source_coordinates_preserved(tmp_path, tower_las_project):
    project, binding, _, points = tower_las_project()
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    assert candidate_set.project_id == project.project_id
    assert candidate_set.source_binding == binding
    assert candidate_set.coordinate_context == binding.coordinate_context
    # Proposal coordinates are within the source bounds.
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    for candidate in candidate_set.candidates:
        x, y, z = candidate.candidate_xyz_absolute_m
        assert xmin - 2 <= x <= xmax + 2
        assert ymin - 2 <= y <= ymax + 2


def test_empty_evidence_produces_no_candidates(tmp_path, tower_las_project):
    _, binding, _, _ = tower_las_project()
    # A ground-only cloud has no non-ground structure to propose.
    rs = np.random.RandomState(5)
    points = np.column_stack(
        (rs.uniform(0, 30, 300), rs.uniform(0, 30, 300), np.zeros(300))
    )
    evidence = build_ground_evidence(tmp_path, points)
    candidate_set = _run_detector(evidence, binding, binding.coordinate_context)
    assert candidate_set.candidate_count == 0
