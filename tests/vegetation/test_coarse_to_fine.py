"""Focused tests for the coarse-to-fine source-native transformation.

These cover the owner-mandated safety properties:

* coarse representative -> raw source recovery is exclusive and deterministic
* raw recovery -> fine (0.10 m) vegetation representatives conserve raw_count
  and original_index
* V4 receives fine representatives, never coarse representatives
* V4 density stays raw-weighted
* full-resolution tree back-map is exact
* the source-native wire chain derives A1 from the recovered raw region and
  fails closed (BLOCKED_NO_VALID_W1) when no reliable W1 exists
"""

from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import pytest
from algorithms.vegetation.coarse_to_fine import (
    RawRegion,
    _first_unclaimed_rows,
    _RawRegionWriter,
    build_fine_representatives,
    build_fine_vegetation_v4_input,
    compute_wire_region_bounds,
    fast_wire_separation_config,
    load_raw_region,
    map_region_rows_to_tree,
    recover_vegetation_region_raw,
    run_source_native_wire_chain,
    write_wire_region_a1,
)
from algorithms.vegetation.fast_research import (
    FAST_RESEARCH_SCHEMA_VERSION,
    NON_CERTIFIED_INPUT_MARKER,
    NON_CERTIFIED_MARKER,
    FastGroundResult,
    sha256_file,
    validate_fast_v4_arrays,
)
from algorithms.vegetation.fast_weighted_preprocess import build_weighted_working_set


def _write_source_las(
    path: Path, xyz: np.ndarray, classification: np.ndarray | None = None
) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    data = laspy.LasData(header)
    data.x, data.y, data.z = xyz.T
    data.classification = (
        classification if classification is not None else np.zeros(len(xyz), dtype=np.uint8)
    )
    data.write(path)


def _write_ground_las(
    path: Path,
    xyz: np.ndarray,
    classification: np.ndarray,
    representative_id: np.ndarray,
    source_original_index: np.ndarray,
    raw_count: np.ndarray,
    hag: np.ndarray,
) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray((0.001, 0.001, 0.001))
    header.offsets = np.zeros(3, dtype=np.float64)
    for name, dtype in (
        ("HeightAboveGround", np.float32),
        ("raw_count", np.uint32),
        ("representative_id", np.uint32),
        ("source_original_index", np.uint64),
    ):
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    data = laspy.LasData(header)
    data.x, data.y, data.z = xyz.T
    data.classification = classification
    data["HeightAboveGround"] = hag.astype(np.float32)
    data["raw_count"] = raw_count.astype(np.uint32)
    data["representative_id"] = representative_id.astype(np.uint32)
    data["source_original_index"] = source_original_index.astype(np.uint64)
    data.write(path)


def _ground_result(source: Path, output: Path, **overrides: object) -> FastGroundResult:
    source_sha = sha256_file(source)
    output_sha = sha256_file(output)
    fields: dict[str, object] = {
        "schema_version": FAST_RESEARCH_SCHEMA_VERSION,
        "non_certified_marker": NON_CERTIFIED_MARKER,
        "source_las_path": str(source),
        "source_las_sha256": source_sha,
        "output_las_path": str(output),
        "output_las_sha256": output_sha,
        "algorithm_name": "PDAL_SMRF_v1",
        "hag_algorithm_name": "PDAL_HAG_NN_v1",
        "parameters": {},
        "config_digest": "a" * 64,
        "source_point_count": 1,
        "output_point_count": 1,
        "ground_class_2_count": 1,
        "non_ground_count": 0,
        "hag_valid_count": 1,
        "hag_min_m": 0.0,
        "hag_max_m": 0.0,
        "xyz_order_preserved": True,
        "source_xyz_sha256": "b" * 64,
        "output_xyz_sha256": "b" * 64,
        "elapsed_seconds": 0.01,
    }
    fields.update(overrides)
    return FastGroundResult(**fields)


def _raw_region_from_points(
    root: Path, xyz: np.ndarray, classification: np.ndarray | None = None
) -> RawRegion:
    writer = _RawRegionWriter(root, origin_xyz=(0.0, 0.0, 0.0))
    cls = np.full(len(xyz), 3, np.uint8) if classification is None else classification
    writer.append(np.asarray(xyz, dtype=np.float64), np.arange(len(xyz), dtype=np.int64), cls)
    return writer.close()


def _synthetic_veg_source(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """Ground rows 0..3, vegetation rows 4..18 forming a coarse vegetation region."""
    ground = np.asarray(
        ((0.1, 0.1, 0.0), (0.5, 0.1, 0.0), (1.1, 0.1, 0.0), (2.1, 0.1, 0.0)), dtype=np.float64
    )
    trunk = np.asarray([(3.0, 0.5, z) for z in np.linspace(0.2, 2.0, 8)], dtype=np.float64)
    crown = np.asarray(
        [
            (3.0 + x, 0.5 + dx, 2.0 + dz)
            for x, dx, dz in (
                (0.05, 0.0, 0.1),
                (0.15, 0.1, 0.2),
                (-0.05, -0.1, 0.3),
                (0.1, 0.05, 0.4),
                (-0.1, 0.05, 0.5),
                (0.0, 0.1, 0.6),
            )
        ],
        dtype=np.float64,
    )
    veg = np.concatenate((trunk, crown), axis=0)
    xyz = np.concatenate((ground, veg), axis=0)
    classification = np.concatenate(
        (np.full(len(ground), 2, np.uint8), np.full(len(veg), 5, np.uint8))
    )
    source = tmp_path / "source.las"
    _write_source_las(source, xyz, classification)
    return source, xyz, classification


def _coarse_working_and_ground(
    tmp_path: Path, source: Path, xyz: np.ndarray
) -> tuple[object, FastGroundResult]:
    working_root = tmp_path / "working"
    _status, working = build_weighted_working_set(source, working_root, voxel_size_m=0.5)
    with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
        rep_xyz = np.asarray(stream["xyz_absolute"], dtype=np.float64)
        rep_id = np.asarray(stream["representative_id"], dtype=np.uint32)
        rep_original = np.asarray(stream["original_index"], dtype=np.int64)
        rep_count = np.asarray(stream["raw_count"], dtype=np.uint32)
    # Ground reps are rows whose representative xyz is on the ground plane.
    ground_mask = rep_xyz[:, 2] < 0.5
    classification = np.where(ground_mask, 2, 5).astype(np.uint8)
    hag = np.clip(rep_xyz[:, 2], 0.0, None).astype(np.float32)
    ground_las = tmp_path / "ground.las"
    _write_ground_las(
        ground_las,
        rep_xyz,
        classification,
        rep_id,
        rep_original,
        rep_count,
        hag,
    )
    ground = _ground_result(
        source,
        ground_las,
        source_point_count=len(xyz),
        output_point_count=len(rep_xyz),
        ground_class_2_count=int(np.count_nonzero(ground_mask)),
        non_ground_count=int(np.count_nonzero(~ground_mask)),
        hag_valid_count=len(rep_xyz),
        hag_min_m=float(np.min(hag)),
        hag_max_m=float(np.max(hag)),
    )
    return working, ground


class TestRawRegionRecovery:
    def test_vegetation_raw_recovery_is_exclusive_and_deterministic(self, tmp_path: Path) -> None:
        source, xyz, classification = _synthetic_veg_source(tmp_path)
        working, _ground = _coarse_working_and_ground(tmp_path, source, xyz)
        with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
            rep_original = np.asarray(stream["original_index"], dtype=np.int64)
            rep_id = np.asarray(stream["representative_id"], dtype=np.uint32)
        # Retain the coarse representatives whose source rows are vegetation.
        retained = rep_id[rep_original >= 4]
        region = recover_vegetation_region_raw(source, working, retained, tmp_path / "region")
        recovered = region.original_index()
        assert len(recovered) > 0
        # Recovery is exclusive: every recovered row is a vegetation source row.
        assert np.all(recovered >= 4)
        # Recovery is exact: it recovers every vegetation row.
        expected = np.asarray([i for i in range(4, len(xyz))], dtype=np.int64)
        assert set(recovered.tolist()) == set(expected.tolist())

    def test_recovery_excludes_exact_wire_members(self, tmp_path: Path) -> None:
        source, xyz, classification = _synthetic_veg_source(tmp_path)
        working, _ground = _coarse_working_and_ground(tmp_path, source, xyz)
        with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
            rep_original = np.asarray(stream["original_index"], dtype=np.int64)
            rep_id = np.asarray(stream["representative_id"], dtype=np.uint32)
        retained = rep_id[rep_original >= 4]
        exclude = np.asarray((4, 5), dtype=np.int64)  # exact source-native wire rows
        region = recover_vegetation_region_raw(
            source, working, retained, tmp_path / "region", exclude_original_index=exclude
        )
        recovered = region.original_index()
        assert 4 not in recovered.tolist()
        assert 5 not in recovered.tolist()


def test_recovery_rejects_out_of_range_representative(tmp_path: Path) -> None:
    source, xyz, classification = _synthetic_veg_source(tmp_path)
    working, _ground = _coarse_working_and_ground(tmp_path, source, xyz)
    with pytest.raises(Exception, match="RETAINED_RANGE"):
        recover_vegetation_region_raw(
            source,
            working,
            np.asarray((9999,), dtype=np.uint32),
            tmp_path / "region",
        )


class TestFineRepresentatives:
    def test_first_unclaimed_rows_matches_stable_unique_semantics(self) -> None:
        positions = np.asarray((4, 2, 4, 1, 2, 5, 1), dtype=np.int64)
        claimed = np.asarray((False, False, True, False, False, True), dtype=bool)
        voxel_ids, rows = _first_unclaimed_rows(positions, claimed)
        np.testing.assert_array_equal(voxel_ids, (1, 4))
        np.testing.assert_array_equal(rows, (3, 0))

    def test_raw_count_and_original_index_conservation(self, tmp_path: Path) -> None:
        xyz = np.asarray(
            (
                (0.01, 0.01, 1.00),
                (0.02, 0.02, 1.01),
                (0.03, 0.03, 1.02),
                (0.11, 0.01, 1.03),
                (0.12, 0.02, 1.04),
            ),
            dtype=np.float64,
        )
        region = _raw_region_from_points(tmp_path / "region", xyz)
        fine = build_fine_representatives(region, tmp_path / "fine.npz", voxel_size_m=0.10)
        assert int(np.sum(fine["raw_count"], dtype=np.int64)) == len(xyz)
        assert len(np.unique(fine["original_index"])) == len(fine["original_index"])
        assert len(fine["original_index"]) <= len(xyz)
        # The representative per voxel is the lowest original index in that voxel.
        assert int(fine["original_index"][0]) == int(
            np.min(region.original_index())
        )

    def test_fine_rep_is_first_raw_row_per_voxel(self, tmp_path: Path) -> None:
        xyz = np.asarray(
            (
                (0.01, 0.01, 1.00),
                (0.02, 0.02, 1.01),
                (0.11, 0.01, 1.03),
                (0.12, 0.02, 1.04),
            ),
            dtype=np.float64,
        )
        region = _raw_region_from_points(tmp_path / "region", xyz)
        fine = build_fine_representatives(region, tmp_path / "fine.npz", voxel_size_m=0.10)
        stored = region.xyz().astype("<f8")
        grid_min = np.asarray(fine["grid_min_xyz"])
        shape = np.asarray(fine["grid_shape"])
        cell = np.floor((stored - grid_min) / 0.10).astype(np.int64)
        linear = (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]
        for rep_position, key in enumerate(np.unique(linear)):
            members = region.original_index()[linear == key]
            assert int(fine["original_index"][rep_position]) == int(members.min())


class TestWireRegionBounds:
    def test_region_bounds_are_none_without_candidates(self, tmp_path: Path) -> None:
        source, xyz, classification = _synthetic_veg_source(tmp_path)
        working, ground = _coarse_working_and_ground(tmp_path, source, xyz)
        bounds = compute_wire_region_bounds(
            np.empty(0, dtype=np.uint32), ground, origin_xyz=(0.0, 0.0, 0.0)
        )
        assert bounds is None

    def test_region_bounds_reach_down_to_ground(self, tmp_path: Path) -> None:
        source, xyz, classification = _synthetic_veg_source(tmp_path)
        working, ground = _coarse_working_and_ground(tmp_path, source, xyz)
        bounds = compute_wire_region_bounds(
            np.asarray((0,), dtype=np.uint32), ground, origin_xyz=(0.0, 0.0, 0.0)
        )
        assert bounds is not None
        assert bounds["z_bounds_min"] <= 0.0  # reaches the recovered ground level


class TestWireA1:
    def test_a1_contract_preserves_source_indexes(self, tmp_path: Path) -> None:
        s = np.arange(0, 10.0, 0.05)
        xyz = np.column_stack((s, np.zeros_like(s), np.full_like(s, 8.0)))
        ground = np.asarray(
            [(x, y, 0.05) for x in np.arange(0, 10.5, 0.5) for y in (-1.0, 0.0, 1.0)],
            dtype=np.float64,
        )
        region = _raw_region_from_points(
            tmp_path / "region", np.concatenate((ground, xyz), axis=0)
        )
        npz, js, stats = write_wire_region_a1(
            region,
            tmp_path / "a1",
            source_id="test_source",
            manifest_source_id="test_source",
            voxel_size_m=0.10,
        )
        assert stats["region_raw_point_count"] == region.count
        payload = json.loads(js.read_text("utf-8"))
        assert payload["source_id"] == "test_source"
        assert Path(payload["output_npz_path"]).resolve() == npz.resolve()
        with np.load(npz, allow_pickle=False) as stream:
            original = np.asarray(stream["original_index"], dtype=np.int64)
            sample = np.asarray(stream["sample_index"], dtype=np.int64)
            xyz_local = np.asarray(stream["xyz_local"], dtype=np.float32)
        assert len(np.unique(original)) == len(original)
        assert len(np.unique(sample)) == len(sample)
        assert np.isfinite(xyz_local).all()
        assert payload["statistics"]["downsampled_point_count"] == len(original)

    def test_fast_wire_separation_config_keeps_geometric_guards(self) -> None:
        config = fast_wire_separation_config("test_source", 500)
        assert config.source_id == "test_source"
        assert config.expected_input_point_count == 500
        # Geometric guards are untouched.
        assert config.confirmation_min_span_m == 1.50
        assert config.fine_min_direction_consistency == 0.99
        assert config.fine_max_transverse_residual_m == 0.04
        assert config.fine_max_curvature_proxy_rad == 1.05
        # Route-appropriate relaxations.
        assert config.obvious_min_mean_candidate_score == 0.70
        assert config.fine_min_medium_low_fraction == 1e-6
        assert config.fine_max_axis_angle_deg == 5.0


class TestFineVegetationV4Input:
    def test_v4_receives_fine_representatives_not_coarse(self, tmp_path: Path) -> None:
        source, xyz, classification = _synthetic_veg_source(tmp_path)
        working, ground = _coarse_working_and_ground(tmp_path, source, xyz)
        # Retain the coarse representatives whose source row is vegetation.
        with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
            rep_original = np.asarray(stream["original_index"], dtype=np.int64)
            rep_id = np.asarray(stream["representative_id"], dtype=np.uint32)
        retained = rep_id[rep_original >= 4]
        vegetation, manifest = build_fine_vegetation_v4_input(
            source_las=source,
            working_set=working,
            ground=ground,
            retained_vegetation_representative_ids=retained,
            origin_context=working.source_bounds_min_xyz,
            source_id="test_source",
            voxel_size_m=0.10,
            output_root=tmp_path / "fine",
        )
        assert vegetation.admission_voxel_size_m == 0.10
        assert vegetation.non_certified_marker == NON_CERTIFIED_INPUT_MARKER
        with np.load(vegetation.output_npz_path, allow_pickle=False) as stream:
            original = np.asarray(stream["original_index"], dtype="<i8")
            raw_count = np.asarray(stream["raw_count"], dtype="<u4")
            xyz_local = np.asarray(stream["xyz_local_m"], dtype="<f8")
            hag = np.asarray(stream["hag_m"], dtype="<f4")
        validate_fast_v4_arrays(original, xyz_local, hag)
        assert vegetation.point_count == len(original)
        assert np.all(raw_count >= 1)
        # raw_count is conserved exactly at the fine-representative level: every
        # recovered raw row maps to exactly one fine rep.
        with np.load(manifest["fine_representatives_npz_path"], allow_pickle=False) as stream:
            fine_count = np.asarray(stream["raw_count"], dtype="<u4")
        assert int(np.sum(fine_count, dtype=np.int64)) == manifest["counts"][
            "region_raw_point_count"
        ]
        # The V4 input is the HAG-filtered subset; its raw count is a subset of
        # the recovered region and remains per-rep exact (raw-weighted density).
        assert int(np.sum(raw_count, dtype=np.int64)) <= manifest["counts"][
            "region_raw_point_count"
        ]
        # origin context matches the scene origin.
        assert tuple(vegetation.origin_context) == tuple(working.source_bounds_min_xyz)

    def test_exact_wire_member_exclusion(self, tmp_path: Path) -> None:
        source, xyz, classification = _synthetic_veg_source(tmp_path)
        working, ground = _coarse_working_and_ground(tmp_path, source, xyz)
        with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
            rep_original = np.asarray(stream["original_index"], dtype=np.int64)
            rep_id = np.asarray(stream["representative_id"], dtype=np.uint32)
        retained = rep_id[rep_original >= 4]
        exclude = np.asarray((rep_original[retained][0],), dtype=np.int64)
        with_exclusion, _m = build_fine_vegetation_v4_input(
            source_las=source,
            working_set=working,
            ground=ground,
            retained_vegetation_representative_ids=retained,
            origin_context=working.source_bounds_min_xyz,
            source_id="test_source",
            voxel_size_m=0.10,
            output_root=tmp_path / "excluded",
            wire_exclusion_original_index=exclude,
        )
        with np.load(with_exclusion.output_npz_path, allow_pickle=False) as stream:
            original = np.asarray(stream["original_index"], dtype="<i8")
        assert int(exclude[0]) not in original.tolist()


class TestFullResolutionBackmap:
    def test_region_rows_inherit_fine_rep_tree_ids(self, tmp_path: Path) -> None:
        xyz = np.asarray(
            (
                (0.01, 0.01, 1.00),
                (0.02, 0.02, 1.01),
                (0.03, 0.03, 1.02),
                (0.11, 0.01, 1.03),
                (0.12, 0.02, 1.04),
            ),
            dtype=np.float64,
        )
        region = _raw_region_from_points(tmp_path / "region", xyz)
        fine_npz = tmp_path / "fine.npz"
        fine = build_fine_representatives(region, fine_npz, voxel_size_m=0.10)
        # Fake V4 tree ids on the fine representatives.
        fake_tree = np.arange(len(fine["original_index"]), dtype=np.int32) + 10
        per_row = map_region_rows_to_tree(
            region, fine_npz, fine["original_index"], fake_tree
        )
        stored = region.xyz().astype("<f8")
        grid_min = np.asarray(fine["grid_min_xyz"])
        shape = np.asarray(fine["grid_shape"])
        cell = np.floor((stored - grid_min) / 0.10).astype(np.int64)
        linear = (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]
        keys = np.unique(linear)
        for row, key in enumerate(linear):
            rep_index = int(np.searchsorted(keys, key))
            assert per_row[row] == fake_tree[rep_index]

    def test_backmap_accepts_filtered_v4_subset(self, tmp_path: Path) -> None:
        # The V4 input may be a HAG-filtered subset of the persisted fine reps.
        xyz = np.asarray(
            (
                (0.01, 0.01, 1.00),
                (0.02, 0.02, 1.01),
                (0.03, 0.03, 1.02),
                (1.01, 0.01, 5.00),
                (1.02, 0.02, 5.01),
            ),
            dtype=np.float64,
        )
        region = _raw_region_from_points(tmp_path / "region", xyz)
        fine_npz = tmp_path / "fine.npz"
        fine = build_fine_representatives(region, fine_npz, voxel_size_m=0.10)
        # Keep only the first rep in the V4 input (a filtered subset).
        v4_original = fine["original_index"][:1]
        v4_tree = np.asarray((7,), dtype=np.int32)
        per_row = map_region_rows_to_tree(region, fine_npz, v4_original, v4_tree)
        assigned = np.flatnonzero(per_row >= 0)
        assert len(assigned) > 0  # the kept rep's rows are assigned
        assert set(per_row[assigned].tolist()) == {7}


class TestSourceNativeWireChain:
    def _wire_region(self, root: Path, *, with_wire: bool) -> RawRegion:
        parts: list[np.ndarray] = []
        gx, gy = np.meshgrid(np.arange(0, 12.5, 0.5), np.arange(-2.0, 2.5, 0.5))
        parts.append(np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, 0.05))))
        if with_wire:
            s = np.arange(0, 12.0, 0.05)
            parts.append(np.column_stack((s, np.zeros_like(s), np.full_like(s, 8.0))))
        xyz = np.concatenate(parts, axis=0)
        cls = np.where(xyz[:, 2] < 1.0, 2, 3).astype(np.uint8)
        return _raw_region_from_points(root / "region", xyz, cls)

    def test_chain_produces_valid_w1_for_clean_line(self, tmp_path: Path) -> None:
        region = self._wire_region(tmp_path, with_wire=True)
        a1_npz, a1_json, _stats = write_wire_region_a1(
            region,
            tmp_path / "a1",
            source_id="test_source",
            manifest_source_id="test_source",
            voxel_size_m=0.10,
        )
        chain = run_source_native_wire_chain(
            a1_npz, a1_json, tmp_path / "chain", source_id="test_source"
        )
        assert chain["TREE_WIRE_STATUS"] == "VALID_W1"
        assert chain["w1_distance_ready_segments"] >= 1
        assert chain["w1_polyline_count"] >= 1
        assert chain["stages"] == {
            "a2a": "PASS",
            "a2b": "PASS",
            "a3": "PASS",
            "a4": "PASS",
            "w1": "PASS",
        }
        assert len(chain["wire_member_original_index"]) > 0
        assert len(chain["w1_continuation_xyz_local"]) > 0

    def test_chain_fails_closed_without_wires(self, tmp_path: Path) -> None:
        region = self._wire_region(tmp_path, with_wire=False)
        a1_npz, a1_json, _stats = write_wire_region_a1(
            region,
            tmp_path / "a1",
            source_id="test_source",
            manifest_source_id="test_source",
            voxel_size_m=0.10,
        )
        chain = run_source_native_wire_chain(
            a1_npz, a1_json, tmp_path / "chain", source_id="test_source"
        )
        assert chain["TREE_WIRE_STATUS"] == "BLOCKED_NO_VALID_W1"
        assert chain["w1_distance_ready_segments"] == 0

    def test_no_stale_w1_is_ever_reused(self, tmp_path: Path) -> None:
        stale = tmp_path / "old_scan_small_w1.npz"
        stale.write_bytes(b"stale bytes")
        region = self._wire_region(tmp_path, with_wire=True)
        a1_npz, a1_json, _stats = write_wire_region_a1(
            region,
            tmp_path / "a1",
            source_id="test_source",
            manifest_source_id="test_source",
            voxel_size_m=0.10,
        )
        chain = run_source_native_wire_chain(
            a1_npz, a1_json, tmp_path / "chain", source_id="test_source"
        )
        assert chain["TREE_WIRE_STATUS"] == "VALID_W1"
        serialized = json.dumps(
            {k: v for k, v in chain.items() if isinstance(v, (str, int, float, list))}
        )
        assert stale.name not in serialized
        assert stale.stem not in serialized


class TestRegionSidecar:
    def test_sidecar_round_trip(self, tmp_path: Path) -> None:

        from algorithms.vegetation.coarse_to_fine import _write_region_sidecar

        xyz = np.asarray(((0.1, 0.1, 1.0), (0.2, 0.2, 1.1)), dtype=np.float64)
        region = _raw_region_from_points(tmp_path / "region", xyz)
        assert region.count == 2
        sidecar = tmp_path / "region" / "region_sidecar.json"
        _write_region_sidecar(region, sidecar)
        loaded = load_raw_region(sidecar)
        assert loaded.count == region.count
        assert loaded.origin_xyz == region.origin_xyz
        assert loaded.xyz().shape == (2, 3)
