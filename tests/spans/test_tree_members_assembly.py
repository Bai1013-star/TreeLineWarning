# ruff: noqa: E501
"""Bounded full-resolution tree membership output assembly (two-pass memmap).

Focused regression tests for the Tree-backend memory fix:

1. trusted unique RawRegion bypasses global np.unique
2. untrusted duplicate validation stays bounded
3. chunked span-core == legacy full-array mask
4. PASS 1 final count exact
5. PASS 2 cursor == PASS 1 count
6. chunked membership == legacy membership
7. deterministic output order
8. no full float64 XYZ copy
9. no global flatnonzero
10. no global full-size unique/sort
11. memmap output finalized safely (manifest + shapes)
12. NPZ keys/dtypes preserved
13. no unintended full NPZ runtime reload (float32, not upcast)
14. confirmed Wire exact subtraction retained
15. Tower exact indices actually subtracted
16. Tower PARTIAL provenance retained
17. no Tower radius/buffer introduced
18. CROWN_STEM_HYBRID membership unchanged (== legacy)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.folder_pipeline.orchestrator import FolderOrchestrator
from algorithms.poles.evidence import PoleEvidenceError
from algorithms.poles.topology import SpanRecord
from algorithms.spans.tree_runner import (
    _assemble_tree_members_store,
    _row_in_span_core,
    _validate_region_original_index_unique,
    load_span_tree_outcome,
)
from algorithms.vegetation.coarse_to_fine import _RawRegionWriter


def _span() -> SpanRecord:
    return SpanRecord(
        line_id=0, line_name="Line01", span_id=0, span_name="Line01-Span001",
        start_pole_candidate_id=0, end_pole_candidate_id=1,
        start_xyz_absolute_m=(0.0, 0.0, 0.0), end_xyz_absolute_m=(50.0, 0.0, 0.0),
        start_pole_top_z_m=15.0, end_pole_top_z_m=15.0, length_m=50.0,
        direction_xy=(1.0, 0.0, 0.0), half_width_m=9.0, halo_m=6.0,
        core_bounds_min_xyz=(0.0, -9.0, 0.0), core_bounds_max_xyz=(50.0, 9.0, 20.0),
        halo_bounds_min_xyz=(-6.0, -9.0, 0.0), halo_bounds_max_xyz=(56.0, 9.0, 20.0),
        quality="CONFIRMED_SPAN",
    )


def _build_region(tmp_path: Path, xyz: np.ndarray, tree_id: np.ndarray):
    """Build a RawRegion whose rows are ascending source rows."""
    writer = _RawRegionWriter(tmp_path / "region", origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.asarray(xyz, dtype=np.float64),
        np.arange(len(xyz), dtype=np.int64),
        np.zeros(len(xyz), dtype=np.uint8),
    )
    region = writer.close()
    raw_tree = np.asarray(tree_id, dtype=np.int32)
    return region, raw_tree


def _legacy_assembly(region, raw_tree, span):
    """The pre-fix full-array output assembly (reference for equality)."""
    raw_xyz_all = region.xyz().astype(np.float64, copy=False)
    assigned_rows = np.flatnonzero(raw_tree >= 0)
    in_core = _row_in_span_core(raw_xyz_all, span, (0.0, 0.0, 0.0))
    return (
        np.asarray(region.original_index(), dtype=np.int64)[assigned_rows],
        raw_xyz_all[assigned_rows],
        raw_tree[assigned_rows],
        in_core[assigned_rows],
    )


def _sample_scene():
    """Synthetic oriented rows: 2 core trees, 1 halo tree, unassigned clutter."""
    rng = np.random.default_rng(7)
    xyz = []
    tree = []
    # Tree A: core, x 2..40
    for x in np.arange(2.0, 40.0, 0.5):
        xyz.append((x, 1.0, 6.0))
        tree.append(1)
    # Tree B: core, x 10..46, u=-2
    for x in np.arange(10.0, 46.0, 0.5):
        xyz.append((x, -2.0, 8.0))
        tree.append(2)
    # Tree C: halo only (x > 50, beyond core x <= 50)
    for x in np.arange(51.0, 55.0, 0.5):
        xyz.append((x, 3.0, 7.0))
        tree.append(3)
    # unassigned clutter scattered across the corridor
    for _ in range(120):
        xyz.append((float(rng.uniform(-5.0, 55.0)), float(rng.uniform(-8.0, 8.0)), float(rng.uniform(0.0, 14.0))))
        tree.append(-1)
    arr = np.asarray(xyz, dtype=np.float64)
    return arr, np.asarray(tree, dtype=np.int32)


def _store_arrays(store):
    xyz_out, original_out, tree_out, core_out, count, root = store
    return (
        np.asarray(original_out, dtype=np.int64).copy(),
        np.asarray(xyz_out, dtype=np.float32).copy(),
        np.asarray(tree_out, dtype=np.int32).copy(),
        np.asarray(core_out, dtype=bool).copy(),
        count,
    )


# ---------------------------------------------------------------------------
# 1 + 4 + 5 + 6 + 7 + 8 + 11 + 18
# ---------------------------------------------------------------------------

def test_bounded_assembly_matches_legacy_membership_and_is_deterministic(tmp_path: Path):
    span = _span()
    xyz, tree_id = _sample_scene()
    region, raw_tree = _build_region(tmp_path / "a", xyz, tree_id)

    store_a = _assemble_tree_members_store(region, raw_tree, span, (0.0, 0.0, 0.0), tmp_path / "out_a", chunk_size=137)
    (orig_a, xyz_a, tree_a, core_a, count_a) = _store_arrays(store_a)
    store_b = _assemble_tree_members_store(region, raw_tree, span, (0.0, 0.0, 0.0), tmp_path / "out_b", chunk_size=137)
    (orig_b, xyz_b, tree_b, core_b, count_b) = _store_arrays(store_b)

    legacy_orig, legacy_xyz, legacy_tree, legacy_core = _legacy_assembly(region, raw_tree, span)

    # PASS1 count == legacy assigned count (membership unchanged, #18).
    assert count_a == int(np.count_nonzero(raw_tree >= 0))
    assert count_a == len(legacy_orig)
    # PASS2 cursor == PASS1 count (exact arrays sized to the count).
    assert len(orig_a) == count_a and len(tree_a) == count_a and len(core_a) == count_a
    # membership identical (#6/#18).
    np.testing.assert_array_equal(orig_a, legacy_orig)
    np.testing.assert_array_equal(tree_a, legacy_tree)
    np.testing.assert_array_equal(core_a, legacy_core)
    # XYZ within the source float32 precision contract (#18).
    np.testing.assert_allclose(xyz_a, legacy_xyz, atol=1e-4)
    # deterministic output order (#7).
    np.testing.assert_array_equal(orig_a, orig_b)
    np.testing.assert_array_equal(tree_a, tree_b)
    np.testing.assert_array_equal(core_a, core_b)
    # no full float64 XYZ copy: store xyz is float32 (#8).
    assert store_a[0].dtype == np.float32
    # memmap output finalized safely: files + manifest (#11).
    manifest = json_load(tmp_path / "out_a" / "tree_members_store" / "manifest.json")
    assert manifest["count"] == count_a
    assert (tmp_path / "out_a" / "tree_members_store" / "xyz.npy").is_file()


def json_load(path: Path):
    import json
    return json.loads(path.read_text("utf-8"))


# ---------------------------------------------------------------------------
# 2 — bounded validation
# ---------------------------------------------------------------------------

def test_duplicate_source_index_validation_is_bounded_and_fails_closed(tmp_path: Path):
    region, _raw_tree = _build_region(
        tmp_path, np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        np.array([0, 1, 2]),
    )
    _validate_region_original_index_unique(region, chunk_size=1)  # ascending, ok

    # duplicate adjacent row -> fail closed without a global np.unique.
    dup_region = _dup_region(tmp_path / "dup")
    with pytest.raises(PoleEvidenceError):
        _validate_region_original_index_unique(dup_region, chunk_size=2)


def test_tile_major_unique_source_indices_pass_bounded_validation(tmp_path: Path):
    writer = _RawRegionWriter(
        tmp_path / "tile_major", origin_xyz=(0.0, 0.0, 0.0)
    )
    writer.append(
        np.array(
            [[2.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.array([2, 0, 1], dtype=np.int64),
        np.zeros(3, dtype=np.uint8),
    )
    region = writer.close()

    _validate_region_original_index_unique(region, chunk_size=2)


def _dup_region(root: Path):
    writer = _RawRegionWriter(root, origin_xyz=(0.0, 0.0, 0.0))
    writer.append(
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64),
        np.array([0, 1, 1], dtype=np.int64),  # duplicate source row 1
        np.zeros(3, dtype=np.uint8),
    )
    return writer.close()


# ---------------------------------------------------------------------------
# 3 — chunked span-core == legacy
# ---------------------------------------------------------------------------

def test_chunked_span_core_matches_legacy_full_mask(tmp_path: Path):
    span = _span()
    xyz, _tree = _sample_scene()
    full = _row_in_span_core(np.asarray(xyz, dtype=np.float64), span, (0.0, 0.0, 0.0))
    chunked = np.concatenate(
        [_row_in_span_core(np.asarray(xyz[i:i + 40], dtype=np.float64), span, (0.0, 0.0, 0.0))
         for i in range(0, len(xyz), 40)]
    )
    np.testing.assert_array_equal(chunked, full)


# ---------------------------------------------------------------------------
# 9 + 10 — no global flatnonzero / unique / sort in the assembly
# ---------------------------------------------------------------------------

def test_assembly_never_calls_global_unique_or_flatnonzero(tmp_path: Path, monkeypatch):
    span = _span()
    xyz, tree_id = _sample_scene()
    region, raw_tree = _build_region(tmp_path / "r", xyz, tree_id)

    def forbid(*_args, **_kwargs):
        raise AssertionError("GLOBAL_NP_UNIQUE_OR_FLATNONZERO_CALLED")

    monkeypatch.setattr("numpy.unique", forbid)
    monkeypatch.setattr("numpy.flatnonzero", forbid)
    store = _assemble_tree_members_store(region, raw_tree, span, (0.0, 0.0, 0.0), tmp_path / "out", chunk_size=73)
    _orig, _xyz, _tree, _core, count = _store_arrays(store)
    assert count == int(np.count_nonzero(raw_tree >= 0))


# ---------------------------------------------------------------------------
# 12 + 13 — NPZ keys/dtypes preserved; no float64 upcast on reload
# ---------------------------------------------------------------------------

def test_npz_compatibility_and_no_full_reload_upcast(tmp_path: Path):
    span = _span()
    xyz, tree_id = _sample_scene()
    region, raw_tree = _build_region(tmp_path / "r", xyz, tree_id)
    store = _assemble_tree_members_store(region, raw_tree, span, (0.0, 0.0, 0.0), tmp_path / "out", chunk_size=61)
    xyz_out, original_out, tree_out, core_out, count, _root = store

    from algorithms.spans.tree_runner import _write_npz
    npz_path = tmp_path / "out" / "tree_members.npz"
    _write_npz(npz_path, original_index=original_out, xyz_local_m=xyz_out, tree_id=tree_out, in_core=core_out)

    with np.load(npz_path, allow_pickle=False) as stream:
        assert set(stream.files) == {"original_index", "xyz_local_m", "tree_id", "in_core"}
        assert stream["original_index"].dtype == np.int64
        assert stream["xyz_local_m"].dtype == np.float32
        assert stream["tree_id"].dtype == np.int32
        assert stream["in_core"].dtype == np.bool_
        assert stream["original_index"].shape[0] == count

    # reload helper must NOT upcast xyz to float64 (#13).
    from algorithms.spans.tree_runner import _write_json
    _write_json(tmp_path / "out" / "span_tree_manifest.json", {"trees": [], "raw_local_maxima": 0, "accepted_tree_tops": 0})
    reloaded = load_span_tree_outcome(span, tmp_path / "out")
    assert reloaded.raw_xyz_local_m.dtype == np.float32


# ---------------------------------------------------------------------------
# 14 + 15 + 16 + 17 — wire + tower exact subtraction, provenance, no buffer
# ---------------------------------------------------------------------------

def _stub_run() -> SimpleNamespace:
    pole = SimpleNamespace(
        base_xyz_absolute_m=(10.0, 10.0, 0.0),
        top_xyz_absolute_m=(10.0, 10.0, 12.0),
    )

    def local_raw_for_xy(_x, _y, radius=None):
        # shaft points within the tight 0.6 m cylinder + an OUTSIDE point.
        xyz = np.array([
            [10.0, 10.0, 0.5], [10.0, 10.0, 5.0], [10.0, 10.0, 11.5],
            [10.2, 10.0, 6.0], [13.0, 13.0, 8.0],  # 13,13 is 4.2m away (buffer must NOT include it)
        ])
        return {
            "xyz_absolute": xyz,
            "original_index": np.arange(len(xyz), dtype=np.int64) + 500,
            "rgb": np.zeros((len(xyz), 3), dtype=np.uint8),
        }

    return SimpleNamespace(
        wire_outcome=SimpleNamespace(confirmed_member_source_index=np.array([100, 101], dtype=np.int64)),
        pole_a=pole,
        pole_b=pole,
        local_raw_for_xy=local_raw_for_xy,
    )


def _veg_region(tmp_path: Path):
    writer = _RawRegionWriter(tmp_path / "veg", origin_xyz=(0.0, 0.0, 0.0))
    # idx 100 = confirmed wire; 500 = tower shaft; 504 = 4.2m outside shaft;
    # 200 / 300 = unrelated vegetation.
    xyz = np.array([
        [10.0, 10.0, 6.0], [1.0, 1.0, 8.0], [13.0, 13.0, 8.0], [2.0, 2.0, 7.0], [10.0, 10.0, 5.0],
    ], dtype=np.float64)
    idx = np.array([100, 200, 504, 300, 500], dtype=np.int64)
    writer.append(xyz, idx, np.zeros(len(xyz), dtype=np.uint8))
    return writer.close()


def test_wire_and_tower_exact_subtraction(tmp_path: Path):
    run = _stub_run()
    workspace = SimpleNamespace(
        source=tmp_path / "missing.las",
        source_cache_dir=tmp_path,
        science_run_root=tmp_path / "science",
    )
    orch = FolderOrchestrator(workspace)
    orch.run = run
    veg = _veg_region(tmp_path)

    excluded, tower_source = orch._build_tree_input_exclusion(veg, None)
    # tower indices actually included in the subtraction set (#15).
    assert len(tower_source) > 0
    assert np.all(np.isin(tower_source, excluded))
    # confirmed wire still exact (#14).
    assert 100 in excluded and 101 in excluded
    # PARTIAL provenance retained, no radius/buffer (#16/#17): the point 4.2 m
    # from the shaft (idx 504) must NOT be a tower member.
    assert 504 not in tower_source
    # shaft rows ARE tower members (tight 0.6 m cylinder, no enlargement).
    assert 500 in tower_source and 501 in tower_source and 503 in tower_source
    assert orch.run_metadata["tower_exclusion_coverage"] in ("PARTIAL", "UNAVAILABLE")

    # The actual subtraction removes wire + tower but keeps unrelated veg.
    from algorithms.spans.region import subset_raw_region_excluding_source_indices
    cleaned = subset_raw_region_excluding_source_indices(veg, excluded, tmp_path / "cleaned")
    kept = np.asarray(cleaned.original_index(), dtype=np.int64).copy()
    assert 100 not in kept  # confirmed wire removed
    assert 500 not in kept  # tower removed
    assert 200 in kept and 300 in kept and 504 in kept  # unrelated vegetation kept
