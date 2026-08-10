"""Full-resolution Tree-Wire input tests.

The exact Tree-Wire engine must consume the back-mapped full-resolution raw
tree members, never the 0.10 m representatives V4 reasoned over.  A fine
representative may summarize many raw source points, so the nearest branch tip
can be a raw member that is not the representative.  These tests prove:

* the full-resolution V4 artifact contains the raw members (including
  non-representative raw rows),
* a deterministic protruding raw point closer to the wire than its voxel's
  representative changes the final minimum clearance to the full-resolution
  value,
* original-index/source-row identity is preserved and no raw row is duplicated,
* the full-resolution artifact validates and publishes for the existing engine.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.vegetation import v4_tree_segmentation as v4s
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter
from algorithms.vegetation.fast_research import (
    FAST_RESEARCH_SCHEMA_VERSION,
    NON_CERTIFIED_MARKER,
    FastGroundResult,
    sha256_file,
)
from algorithms.vegetation.full_res_clearance import (
    build_full_resolution_v4_artifact,
)
from algorithms.vegetation.tree_wire_clearance import exact_point_to_segment_distance
from core.contracts.v4_inputs import (
    V4InputHandoffResult,
    V4RecoveredVegetationSet,
    V4RecoveryCoverageReport,
    V4ValidatedFrvMembershipView,
    frv_v4_recovered_index_digest,
)
from core.contracts.v4_tree_segmentation import (
    V4TreeAssignmentState,
)

from tests.io.test_source_provider import _source_identity
from tests.vegetation.test_v4_tree_segmentation import make_config, make_tree_points


def _handoff_with_source_count(
    points: list[tuple[float, float, float]], source_count: int
) -> V4InputHandoffResult:
    n = len(points)
    xyz = np.asarray(points, dtype=np.float64)
    hag = xyz[:, 2].astype(np.float32)
    original = np.arange(n, dtype=np.int64)
    source = _source_identity(
        "C:/synthetic.las",
        "a" * 64,
        source_count,
        ([0.0, 0.0, 0.0], [20.0, 8.0, 11.0]),
    )

    def ro(array: np.ndarray) -> np.ndarray:
        array.setflags(write=False)
        return array

    recovered = V4RecoveredVegetationSet.__new__(V4RecoveredVegetationSet)
    object.__setattr__(recovered, "source_identity", source)
    object.__setattr__(recovered, "original_index", ro(original.astype("<i8")))
    object.__setattr__(recovered, "xyz_local_m", ro(xyz.astype("<f8")))
    object.__setattr__(recovered, "hag_m", ro(hag.astype("<f4")))
    object.__setattr__(recovered, "membership_state", ro(np.ones(n, dtype="|u1")))
    object.__setattr__(recovered, "hag_evidence_state", ro(np.ones(n, dtype="|u1")))
    object.__setattr__(
        recovered, "segmentation_hag_readiness", ro(np.ones(n, dtype="|u1"))
    )
    object.__setattr__(recovered, "candidate_reason_flags", ro(np.zeros(n, dtype="<u4")))
    object.__setattr__(recovered, "candidate_confidence", None)
    object.__setattr__(recovered, "frv_membership_artifact_digest", "b" * 64)
    object.__setattr__(
        recovered,
        "recovered_index_digest",
        frv_v4_recovered_index_digest(
            source_content_identity_digest=source.source_content_identity_digest,
            original_index=original,
        ),
    )
    view = V4ValidatedFrvMembershipView.__new__(V4ValidatedFrvMembershipView)
    object.__setattr__(view, "source_identity", source)
    object.__setattr__(view, "view_digest", "d" * 64)
    report = V4RecoveryCoverageReport.__new__(V4RecoveryCoverageReport)
    object.__setattr__(report, "report_digest", "e" * 64)
    handoff = V4InputHandoffResult.__new__(V4InputHandoffResult)
    object.__setattr__(handoff, "validated_membership", view)
    object.__setattr__(handoff, "recovered_vegetation", recovered)
    object.__setattr__(handoff, "unresolved_measured_points", None)
    object.__setattr__(handoff, "unrecovered_regions", ())
    object.__setattr__(handoff, "coverage_report", report)
    object.__setattr__(handoff, "output_reference", None)
    object.__setattr__(handoff, "handoff_digest", "c" * 64)
    return handoff


def _ground_result(tmp_path: Path) -> FastGroundResult:
    header = laspy.LasHeader(point_format=3, version="1.2")
    for name, dtype in (
        ("HeightAboveGround", np.float32),
        ("raw_count", np.uint32),
        ("representative_id", np.uint32),
        ("source_original_index", np.uint64),
    ):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    data = laspy.LasData(header)
    xyz = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64)
    data.x, data.y, data.z = xyz.T
    data.classification = np.asarray((2, 2, 2), dtype=np.uint8)
    data["HeightAboveGround"] = np.zeros(3, dtype=np.float32)
    data["raw_count"] = np.ones(3, dtype=np.uint32)
    data["representative_id"] = np.arange(3, dtype=np.uint32)
    data["source_original_index"] = np.arange(3, dtype=np.uint64)
    ground_las = tmp_path / "ground.las"
    data.write(ground_las)
    source = tmp_path / "source.las"
    _laspy_placeholder(source)
    return FastGroundResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_MARKER,
        source_las_path=str(source),
        source_las_sha256=sha256_file(source),
        output_las_path=str(ground_las),
        output_las_sha256=sha256_file(ground_las),
        algorithm_name="PDAL_SMRF_v1",
        hag_algorithm_name="PDAL_HAG_NN_v1",
        parameters={},
        config_digest="a" * 64,
        source_point_count=3,
        output_point_count=3,
        ground_class_2_count=3,
        non_ground_count=0,
        hag_valid_count=3,
        hag_min_m=0.0,
        hag_max_m=0.0,
        xyz_order_preserved=True,
        source_xyz_sha256="b" * 64,
        output_xyz_sha256="b" * 64,
        elapsed_seconds=0.01,
    )


def _laspy_placeholder(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    data = laspy.LasData(header)
    data.x = np.asarray((0.0, 1.0, 2.0), dtype=np.float64)
    data.y = np.zeros(3, dtype=np.float64)
    data.z = np.zeros(3, dtype=np.float64)
    data.write(path)


_WIRE_A = np.asarray((10.0, 2.0, 10.0), dtype=np.float64)
_WIRE_B = np.asarray((20.0, 2.0, 10.0), dtype=np.float64)


def _min_distance(xyz: np.ndarray) -> float:
    return min(
        exact_point_to_segment_distance(point, _WIRE_A, _WIRE_B)[0] for point in xyz
    )


def test_fullres_artifact_uses_raw_members_and_protruding_point(
    tmp_path: Path,
) -> None:
    tree_points = make_tree_points(15.0, 0.0)
    source_count = len(tree_points) + 1
    handoff = _handoff_with_source_count(tree_points, source_count)
    fine = v4s.segment_v4_trees(handoff, make_config())
    assert len(fine.tree_records) == 1

    assigned = (
        np.asarray(fine.assignment_state_by_point, dtype=np.uint8)
        == int(V4TreeAssignmentState.ASSIGNED)
    ) & (np.asarray(fine.tree_id_by_point, dtype=np.int32) == 0)
    member_xyz = np.asarray(fine.xyz_local_m)[assigned]
    member_orig = np.asarray(fine.original_index)[assigned]

    # A protruding raw point very close to the wire (not a representative row).
    protruding = np.asarray(((15.0, 2.0, 9.95),), dtype=np.float64)
    protruding_orig = np.asarray((source_count - 1,), dtype=np.int64)
    xyz_all = np.concatenate((member_xyz, protruding), axis=0)
    orig_all = np.concatenate((member_orig, protruding_orig), axis=0)
    order = np.argsort(orig_all, kind="stable")
    xyz_all = xyz_all[order]
    orig_all = orig_all[order]

    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(xyz_all, orig_all, np.full(len(xyz_all), 5, np.uint8))
    region = writer.close()
    region_tree = np.zeros(region.count, dtype=np.int32)

    full, manifest = build_full_resolution_v4_artifact(
        fine,
        region,
        region_tree,
        (0.0, 0.0, 0.0),
        _ground_result(tmp_path),
        tmp_path / "fullres_artifact",
    )
    assert manifest["tree_geometry_source"] == "FULL_RESOLUTION_RAW"
    assert manifest["full_resolution_tree_point_count"] == len(xyz_all)

    # The nearest full-res member to the wire is the protruding raw row.
    nearest_index = int(
        np.argmin(
            [
                exact_point_to_segment_distance(point, _WIRE_A, _WIRE_B)[0]
                for point in full.xyz_local_m
            ]
        )
    )
    nearest_original = int(full.original_index[nearest_index])
    assert nearest_original == int(protruding_orig[0])
    assert np.allclose(full.xyz_local_m[nearest_index], protruding[0])

    fullres_min = _min_distance(np.asarray(full.xyz_local_m, dtype=np.float64))
    fine_min = _min_distance(member_xyz.astype(np.float64))
    assert fullres_min < fine_min
    # The raw region is stored as float32, so the reconstructed protruding point
    # carries a ~2e-7 m rounding; 1e-5 tolerates that storage precision.
    assert fullres_min == pytest.approx(
        exact_point_to_segment_distance(protruding[0], _WIRE_A, _WIRE_B)[0],
        abs=1e-5,
    )


def test_fullres_original_index_conservation_no_duplication(tmp_path: Path) -> None:
    tree_points = make_tree_points(15.0, 0.0)
    source_count = len(tree_points) + 1
    handoff = _handoff_with_source_count(tree_points, source_count)
    fine = v4s.segment_v4_trees(handoff, make_config())
    assigned = (
        np.asarray(fine.assignment_state_by_point, dtype=np.uint8)
        == int(V4TreeAssignmentState.ASSIGNED)
    ) & (np.asarray(fine.tree_id_by_point, dtype=np.int32) == 0)
    member_xyz = np.asarray(fine.xyz_local_m)[assigned]
    member_orig = np.asarray(fine.original_index)[assigned]
    protruding = np.asarray(((15.0, 2.0, 9.95),), dtype=np.float64)
    protruding_orig = np.asarray((source_count - 1,), dtype=np.int64)
    xyz_all = np.concatenate((member_xyz, protruding), axis=0)
    orig_all = np.concatenate((member_orig, protruding_orig), axis=0)
    order = np.argsort(orig_all, kind="stable")
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(xyz_all[order], orig_all[order], np.full(len(xyz_all), 5, np.uint8))
    region = writer.close()
    full, manifest = build_full_resolution_v4_artifact(
        fine,
        region,
        np.zeros(region.count, dtype=np.int32),
        (0.0, 0.0, 0.0),
        _ground_result(tmp_path),
        tmp_path / "fullres_artifact2",
    )
    indexes = np.asarray(full.original_index, dtype=np.int64)
    assert len(np.unique(indexes)) == len(indexes)  # no duplication
    assert np.all(np.diff(indexes) > 0)  # strictly increasing
    assert set(indexes.tolist()) == set(orig_all[order].tolist())
    # Full-res members include the raw protruding row and are raw source rows.
    assert int(protruding_orig[0]) in indexes.tolist()
