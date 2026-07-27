"""Deterministic temporary source fixtures for I1 export/reload validation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from algorithms.integration import (
    eligible_vegetation_cluster_ids,
    eligible_wire_segment_keys,
    export_i1_bundle,
    load_i1_bundle,
)
from core.contracts import (
    I1Error,
    VegetationAssociationConfig,
    VegetationAssociationResult,
    VegetationAssociationStatistics,
    VegetationClusterStatus,
    VegetationInstanceConfig,
    VegetationInstanceResult,
    VegetationInstanceStatistics,
    VegetationManualLog,
    WireModelMethod,
    WireModelStatus,
    WireModelSummary,
    WireReconstructionConfig,
    WireReconstructionResult,
    WireReconstructionStatistics,
    WireSeedConfig,
    WireSeparationConfig,
    WireSeparationResult,
    WireSeparationStatistics,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _refresh_output_hash(json_path: Path, npz_path: Path) -> None:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    raw["output_npz_sha256"] = _sha(npz_path)
    raw["output_npz_bytes"] = npz_path.stat().st_size
    _write_json(json_path, raw)


def _a4_json(root: Path, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Path:
    path = root / "a4.json"
    result = WireSeparationResult(
        "scene",
        "manifest",
        Path("a3.npz"),
        Path("a3.json"),
        "0" * 64,
        "0" * 64,
        Path("a4.npz"),
        Path("a4.json"),
        "0" * 64,
        0,
        origin,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        WireSeparationConfig(source_id="scene"),
        WireSeedConfig(),
        WireSeparationStatistics(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0),
        {},
        (),
        {},
        {},
        {},
        {},
        {},
        (),
        "a4-test",
        "2026-01-01T00:00:00+00:00",
    )
    _write_json(path, result.to_dict())
    return path


def _w1_pair(
    root: Path, *, ready: tuple[bool, ...] = (True,), gap: bool = False
) -> tuple[Path, Path]:
    a4 = _a4_json(root)
    segment_count = len(ready)
    points = (
        np.vstack(
            [
                np.array([[float(i), 0, 5], [float(i) + 0.25, 0, 5]], np.float32)
                for i in range(segment_count)
            ]
        )
        if segment_count
        else np.empty((0, 3), np.float32)
    )
    offsets = np.arange(0, 2 * segment_count + 1, 2, dtype=np.int64)
    arrays = {
        "schema_version": np.asarray("wire-models-v2"),
        "wire_id": np.asarray([0], np.int32),
        "source_track_id": np.asarray([0], np.int32),
        "wire_status": np.asarray([int(WireModelStatus.CONFIRMED_MODEL)], np.uint8),
        "wire_method": np.asarray([int(WireModelMethod.PIECEWISE_LINEAR)], np.uint8),
        "wire_quality_flags": np.zeros(1, np.uint16),
        "wire_has_distance_ready_segments": np.asarray([any(ready)], bool),
        "wire_continuous_distance_ready": np.asarray([not gap], bool),
        "wire_segment_offsets": np.asarray([0, segment_count], np.int64),
        "segment_wire_id": np.zeros(segment_count, np.int32),
        "segment_index": np.arange(segment_count, dtype=np.int32),
        "segment_method": np.full(segment_count, int(WireModelMethod.PIECEWISE_LINEAR), np.uint8),
        "segment_quality_flags": np.zeros(segment_count, np.uint16),
        "segment_distance_ready": np.asarray(ready, bool),
        "segment_parameter_domain": np.tile(
            np.asarray([[0.0, 1.0]], np.float64), (segment_count, 1)
        ),
        "sampled_curve_xyz": points,
        "sampled_curve_parameter": np.tile(np.asarray([0.0, 1.0], np.float32), segment_count),
        "sampled_curve_offsets": offsets,
        "gap_wire_id": np.asarray([0] if gap else [], np.int32),
        "gap_left_segment_index": np.asarray([0] if gap else [], np.int32),
        "gap_right_segment_index": np.asarray([1] if gap else [], np.int32),
        "gap_start_xyz": np.asarray([[0.25, 0, 5]] if gap else [], np.float32).reshape((-1, 3)),
        "gap_end_xyz": np.asarray([[1.0, 0, 5]] if gap else [], np.float32).reshape((-1, 3)),
    }
    npz, json_path = root / "w1.npz", root / "w1.json"
    np.savez_compressed(npz, **arrays)
    summary = WireModelSummary(
        0,
        0,
        WireModelStatus.CONFIRMED_MODEL,
        WireModelMethod.PIECEWISE_LINEAR,
        0,
        any(ready),
        not gap,
        segment_count,
        2 * segment_count,
        2 * segment_count,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        (0, 0, 5),
        (1, 0, 5),
        (1, 0, 0),
    )
    result = WireReconstructionResult(
        "wire-models-v2",
        "scene",
        "manifest",
        "a4.npz",
        "a4.json",
        "0" * 64,
        _sha(a4),
        "w1.npz",
        "w1.json",
        _sha(npz),
        npz.stat().st_size,
        WireReconstructionConfig(source_id="scene"),
        WireReconstructionStatistics(
            3, 2 * segment_count, 1, segment_count, int(gap), int(sum(ready)), 0.0
        ),
        {},
        (summary,),
        (),
        (),
        (),
        "w1-confirmed-wire-centreline-v3",
        "2026-01-01T00:00:00+00:00",
    )
    _write_json(json_path, result.to_dict())
    return npz, json_path


def _v2_pair(
    root: Path, statuses: tuple[VegetationClusterStatus, ...] = (VegetationClusterStatus.CONFIRMED,)
) -> tuple[Path, Path]:
    n = max(3 * len(statuses), 1)
    cluster_by = (
        np.repeat(np.arange(len(statuses), dtype=np.int32), 3)
        if statuses
        else np.full(n, -1, np.int32)
    )
    cluster_by = cluster_by[:n]
    status_by = (
        np.concatenate([np.full(3, int(state), np.uint8) for state in statuses])
        if statuses
        else np.full(n, int(VegetationClusterStatus.UNASSIGNED), np.uint8)
    )
    vegetation = (
        np.concatenate(
            [
                np.full(3, i if state is VegetationClusterStatus.CONFIRMED else -1, np.int32)
                for i, state in enumerate(statuses)
            ]
        )
        if statuses
        else np.full(n, -1, np.int32)
    )
    arrays = {
        "original_index": np.arange(100, 100 + n, dtype=np.int64),
        "sample_index": np.arange(n, dtype=np.int64),
        "xyz_local": np.column_stack((np.arange(n), np.zeros(n), np.ones(n))).astype(np.float32),
        "wire_id_by_point": np.full(n, -1, np.int32),
        "vegetation_rule_score": np.full(n, 0.8, np.float32),
        "vegetation_confidence": np.full(n, 0.8, np.float32),
        "automatic_vegetation_level": np.full(n, 3, np.uint8),
        "vegetation_level": np.full(n, 3, np.uint8),
        "hag_state": np.zeros(n, np.uint8),
        "local_scale_raw_m": np.full(n, 0.1, np.float32),
        "local_scale_m": np.full(n, 0.1, np.float32),
        "local_scale_state": np.zeros(n, np.uint8),
        "cluster_id_by_point": cluster_by,
        "vegetation_id_by_point": vegetation,
        "cluster_status_by_point": status_by,
        "distance_ready_by_point": vegetation >= 0,
        "assignment_confidence": np.full(n, 0.8, np.float32),
        "assignment_reason_flags": np.zeros(n, np.uint16),
        "assignment_quality_flags": np.zeros(n, np.uint16),
        "manual_provenance_flags": np.zeros(n, np.uint16),
        "origin_xyz": np.zeros(3, np.float64),
        "cluster_id": np.arange(len(statuses), dtype=np.int32),
        "cluster_status": np.asarray([int(s) for s in statuses], np.uint8),
        "cluster_vegetation_id": np.asarray(
            [i if s is VegetationClusterStatus.CONFIRMED else -1 for i, s in enumerate(statuses)],
            np.int32,
        ),
        "cluster_member_offsets": np.arange(0, 3 * len(statuses) + 1, 3, dtype=np.int64),
        "ordered_cluster_member_row_index": np.arange(3 * len(statuses), dtype=np.int64),
    }
    npz, json_path = root / "v2.npz", root / "v2.json"
    np.savez_compressed(npz, **arrays)
    confirmed = sum(state is VegetationClusterStatus.CONFIRMED for state in statuses)
    statistics = VegetationInstanceStatistics(
        n, n, n, len(statuses), confirmed, len(statuses) - confirmed, 0, 0, n, 0, 0.0
    )
    result = VegetationInstanceResult(
        "vegetation-instances-v2",
        "scene",
        "manifest",
        (0.0, 0.0, 0.0),
        "v2.npz",
        _sha(npz),
        npz.stat().st_size,
        VegetationInstanceConfig(source_id="scene"),
        VegetationManualLog(),
        {},
        statistics,
        {},
        tuple({} for _ in statuses),
        (),
        "v2-generalized-morphology-safe-vegetation-instances-v2",
        "2026-01-01T00:00:00+00:00",
    )
    _write_json(json_path, result.to_dict())
    return npz, json_path


def _v2a_pair(root: Path, v2_npz: Path, v2_json: Path) -> tuple[Path, Path]:
    arrays = {
        "pair_cluster_ids": np.empty((0, 2), np.int32),
        "pair_status": np.empty(0, np.uint8),
        "pair_confidence": np.empty(0, np.float32),
        "pair_positive_evidence": np.empty((0, 10), np.float32),
        "pair_blocking_evidence": np.empty((0, 10), np.float32),
        "pair_reason_flags": np.empty(0, np.uint32),
        "pair_distance_summary_m": np.empty((0, 5), np.float32),
        "pair_scale_density_summary": np.empty((0, 4), np.float32),
        "pair_hag_summary": np.empty((0, 4), np.float32),
        "pair_low_support_counts": np.empty((0, 6), np.int32),
        "pair_provenance_flags": np.empty(0, np.uint32),
        "pair_decision_input_completeness": np.empty(0, np.float32),
    }
    npz, json_path = root / "v2a.npz", root / "v2a.json"
    np.savez_compressed(npz, **arrays)
    result = VegetationAssociationResult(
        "vegetation-associations-v1",
        "v2a-review-level-vegetation-association-v1",
        "scene",
        "manifest",
        (0.0, 0.0, 0.0),
        "v2a.npz",
        _sha(npz),
        npz.stat().st_size,
        VegetationAssociationConfig(source_id="scene"),
        {"v2_npz": {"sha256": _sha(v2_npz)}, "v2_json": {"sha256": _sha(v2_json)}},
        {},
        VegetationAssociationStatistics(0, 0, 0, 0, 0, 0, 0.0),
        (),
        (),
        "2026-01-01T00:00:00+00:00",
    )
    _write_json(json_path, result.to_dict())
    return npz, json_path


def _export(
    tmp_path: Path,
    *,
    statuses: tuple[VegetationClusterStatus, ...] = (VegetationClusterStatus.CONFIRMED,),
    ready: tuple[bool, ...] = (True,),
    gap: bool = False,
    v2a: bool = False,
):
    w1 = _w1_pair(tmp_path, ready=ready, gap=gap)
    v2 = _v2_pair(tmp_path, statuses)
    kwargs = {}
    if v2a:
        kwargs["v2a_npz_path"], kwargs["v2a_json_path"] = _v2a_pair(tmp_path, *v2)
    return export_i1_bundle(*w1, *v2, tmp_path / "bundles", project_root=tmp_path, **kwargs)


def test_valid_bundle_and_deterministic_reuse(tmp_path: Path) -> None:
    first = _export(tmp_path)
    second = _export(tmp_path)
    assert first.root == second.root
    assert eligible_wire_segment_keys(first) == ((0, 0),)
    assert eligible_vegetation_cluster_ids(first) == (0,)


def test_optional_v2a_is_isolated(tmp_path: Path) -> None:
    bundle = _export(tmp_path, v2a=True)
    assert bundle.v2a_result is not None
    assert eligible_vegetation_cluster_ids(bundle) == (0,)


def test_relocation_segments_gaps_and_status_exclusion(tmp_path: Path) -> None:
    bundle = _export(
        tmp_path,
        statuses=(
            VegetationClusterStatus.CONFIRMED,
            VegetationClusterStatus.REVIEW_REQUIRED,
            VegetationClusterStatus.PROVISIONAL,
            VegetationClusterStatus.REJECTED_AS_INSTANCE,
        ),
        ready=(True, False, True),
        gap=True,
    )
    assert eligible_wire_segment_keys(bundle) == ((0, 0), (0, 2))
    assert eligible_vegetation_cluster_ids(bundle) == (0,)
    moved = tmp_path / "moved" / bundle.manifest.bundle_id
    moved.parent.mkdir()
    shutil.move(str(bundle.root), moved)
    assert load_i1_bundle(moved).manifest.bundle_id == bundle.manifest.bundle_id


def test_no_confirmed_records_is_valid_but_empty(tmp_path: Path) -> None:
    bundle = _export(tmp_path, statuses=(VegetationClusterStatus.REVIEW_REQUIRED,), ready=(False,))
    assert not eligible_wire_segment_keys(bundle)
    assert not eligible_vegetation_cluster_ids(bundle)


@pytest.mark.parametrize(
    "mutation", ("hash", "missing", "traversal", "expected", "malformed", "unknown")
)
def test_loader_rejects_integrity_contract_failures(tmp_path: Path, mutation: str) -> None:
    bundle = _export(tmp_path)
    manifest_path = bundle.root / "i1_manifest.json"
    if mutation == "hash":
        (bundle.root / bundle.manifest.w1.npz_relpath).write_bytes(b"changed")
        target = bundle.root
    elif mutation == "missing":
        (bundle.root / bundle.manifest.v2.npz_relpath).unlink()
        target = bundle.root
    elif mutation == "traversal":
        raw = json.loads(manifest_path.read_text())
        raw["w1"]["npz_relpath"] = "../escape.npz"
        _write_json(manifest_path, raw)
        target = bundle.root
    elif mutation == "expected":
        with pytest.raises(I1Error):
            load_i1_bundle(bundle.root, expected_bundle_id="0" * 64)
        return
    elif mutation == "malformed":
        manifest_path.write_text("{", encoding="utf-8")
        target = bundle.root
    else:
        raw = json.loads(manifest_path.read_text())
        raw["unknown"] = True
        _write_json(manifest_path, raw)
        target = bundle.root
    with pytest.raises(I1Error):
        load_i1_bundle(target)


def test_export_rejects_origin_source_and_identity_mismatch(tmp_path: Path) -> None:
    w1 = _w1_pair(tmp_path)
    v2 = _v2_pair(tmp_path)
    (tmp_path / "a4.json").unlink()
    with pytest.raises(I1Error):
        export_i1_bundle(*w1, *v2, tmp_path / "bundles", project_root=tmp_path)


@pytest.mark.parametrize("mutation", ("nan", "offsets", "object"))
def test_export_rejects_invalid_w1_geometry_and_storage(tmp_path: Path, mutation: str) -> None:
    w1 = _w1_pair(tmp_path)
    v2 = _v2_pair(tmp_path)
    with np.load(w1[0], allow_pickle=False) as stream:
        arrays = {name: stream[name] for name in stream.files}
    if mutation == "nan":
        arrays["sampled_curve_xyz"][0, 0] = np.nan
    elif mutation == "offsets":
        arrays["sampled_curve_offsets"][-1] -= 1
    else:
        arrays["unexpected_object"] = np.asarray([{"not": "portable"}], dtype=object)
    np.savez_compressed(w1[0], **arrays)
    _refresh_output_hash(w1[1], w1[0])
    with pytest.raises(I1Error):
        export_i1_bundle(*w1, *v2, tmp_path / "bundles", project_root=tmp_path)


def test_export_rejects_source_identity_origin_and_member_mismatch(tmp_path: Path) -> None:
    w1 = _w1_pair(tmp_path)
    v2 = _v2_pair(tmp_path)
    raw = json.loads(w1[1].read_text(encoding="utf-8"))
    raw["source_id"] = "other-scene"
    _write_json(w1[1], raw)
    with pytest.raises(I1Error):
        export_i1_bundle(*w1, *v2, tmp_path / "bundles", project_root=tmp_path)

    w1 = _w1_pair(tmp_path)
    v2 = _v2_pair(tmp_path)
    _a4_json(tmp_path, origin=(9.0, 0.0, 0.0))
    raw = json.loads(w1[1].read_text(encoding="utf-8"))
    raw["input_a4_json_sha256"] = _sha(tmp_path / "a4.json")
    _write_json(w1[1], raw)
    with pytest.raises(I1Error):
        export_i1_bundle(*w1, *v2, tmp_path / "origin-bundles", project_root=tmp_path)

    w1 = _w1_pair(tmp_path)
    v2 = _v2_pair(tmp_path)
    with np.load(v2[0], allow_pickle=False) as stream:
        arrays = {name: stream[name] for name in stream.files}
    arrays["cluster_id_by_point"][0] = 99
    np.savez_compressed(v2[0], **arrays)
    _refresh_output_hash(v2[1], v2[0])
    with pytest.raises(I1Error):
        export_i1_bundle(*w1, *v2, tmp_path / "member-bundles", project_root=tmp_path)
