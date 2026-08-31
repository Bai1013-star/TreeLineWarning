from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import _encode_indices
from algorithms.spans.role_tiles import ROLE_TILE_RECORD_DTYPE
from core.stage_b import dbh_recovery as recovery
from core.stage_b.dbh_evidence import (
    array_digest,
    editor_eligibility,
    local_raw_query,
    member_binding,
    sha256,
)


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _cylinder():
    angle = np.tile(np.linspace(0, 2 * np.pi, 72, endpoint=False), 6)
    hag = np.repeat(np.linspace(0.925, 1.675, 6), 72)
    return np.column_stack((0.15 * np.cos(angle), 0.15 * np.sin(angle), hag))


def test_anchor_prefers_breast_height_column_only_in_existing_group():
    cfg = recovery.StemGuidedHybridConfig()

    def stem(number, xy, confidence):
        return SimpleNamespace(stem_seed_id=number, xy_local_m=xy, stem_confidence=confidence)

    primary = stem(0, (0.1, 0.1), 0.99)
    lower = stem(1, (0.8, 0.1), 0.8)
    unrelated = stem(2, (5.1, 0.1), 0.95)
    support = {(2, 0): 25, (14, 0): 1000}
    anchor = recovery._breast_height_anchor(primary, [primary, lower, unrelated], support, cfg)
    assert anchor["xy"].tolist() == list(lower.xy_local_m)
    assert anchor["group_raw_stem_ids"] == [0, 1]
    fallback = recovery._breast_height_anchor(primary, [primary, lower, unrelated], {}, cfg)
    assert fallback["xy"].tolist() == list(primary.xy_local_m)
    assert fallback["selection"] == "EXISTING_PRIMARY_STEM_AXIS"


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path
    br = root / "artifacts/project-runs/batches/run"
    source = {"sha256": "a" * 64, "source_point_count": 10000, "file_size_bytes": 1000000}
    ground = np.asarray([[x, y, 0.0] for x in (-2, 0, 2) for y in (-2, 0, 2)])
    prep_path = root / "artifacts/project-runs/source-shared/source/preparation.json"
    prep_path.parent.mkdir(parents=True)
    np.save(prep_path.parent / "ground.npy", ground)
    prep = {
        "schema": "treeline-project-source-shared-preparation-v1",
        "authority": "FULL_RESOLUTION_RAW",
        "tower_membership_used_for_vegetation": False,
        "source_point_count": 10000,
        "coarse_ground_reference": (prep_path.parent / "ground.npy").relative_to(root).as_posix(),
        "origin_xyz": [0.0, 0.0, 0.0],
        "ground_preparation": {"source_identity": source},
        "vegetation_role_tile_index_reference": "raw-cache",
    }
    _json(prep_path, prep)
    _json(
        br / "batch.json",
        {"analysis_run_id": "run", "project_id": "project", "span_statuses": {"span": "FINALIZED"}},
    )
    original = [{"global_tree_id": 0, "merged_span_tree_keys": [[0, 0]]}]
    _json(br / "batch-reconciliation.json", {"trees": original})
    _json(br / "tree_edits.json", {"analysis_run_id": "run", "operations": []})
    _json(br / "span/tree/span_tree_manifest.json", {"span_id": 0})
    angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    xyz = np.column_stack((np.cos(angle), np.sin(angle), np.linspace(4, 5, 100)))
    idx = np.arange(100, dtype=np.int64)
    tree = np.zeros(100, dtype=np.int32)
    np.savez(br / "tree_segmentation_final.npz", original_index=idx, xyz_local_m=xyz, tree_id=tree)
    record = SimpleNamespace(
        global_tree_id=0,
        persistent_tree_id="tree-0",
        merged_span_tree_keys=((0, 0),),
        full_resolution_point_count=100,
    )
    reconciled = SimpleNamespace(
        global_tree_records=(record,),
        final_original_index=idx,
        final_xyz_local_m=xyz,
        final_tree_id=tree,
    )
    result = SimpleNamespace(
        analysis_run_id="run",
        project_id="project",
        scan_id="scan",
        telemetry=SimpleNamespace(source_point_count=10000),
        batch_diagnostics={"source_preparation_reference": prep_path.relative_to(root).as_posix()},
    )
    stem = _cylinder()
    fine = {
        "fine_xyz_local_m": stem,
        "fine_hag_m": stem[:, 2],
        "fine_original_index": np.arange(1000, 1000 + len(stem)),
    }
    span = {
        "fine": fine,
        "anchors": {0: {"xy": np.array([0.0, 0.0]), "confidence": 0.95}},
        "proof": {},
    }
    monkeypatch.setattr(recovery, "_span_evidence", lambda path: span)
    return root, br, result, reconciled, span, source


def test_recovered_persistence_restart_and_non_dbh_membership(project):
    root, br, result, reconciled, _, _ = project
    before = member_binding(reconciled)
    final_hash = sha256(br / "tree_segmentation_final.npz")
    payload = recovery.recover_finalized_dbh(root, result, reconciled)
    row = payload["trees"][0]
    assert row["dbh_method"] == "RECOVERED_STEM"
    assert row["dbh_cm"] == pytest.approx(30, abs=0.1)
    assert row["dbh_band_point_count"] == 432
    assert recovery.load_finalized_dbh(root, result, reconciled)[0]["dbh_cm"] == row["dbh_cm"]
    assert member_binding(reconciled) == before
    assert sha256(br / "tree_segmentation_final.npz") == final_hash
    with np.load(root / payload["evidence_reference"]) as f:
        assert np.array_equal(f["original_index"], np.arange(1000, 1432))


def test_same_raw_stem_in_two_span_halos_cannot_publish_twice(project):
    root, br, result, reconciled, _, _ = project
    second = SimpleNamespace(**vars(reconciled.global_tree_records[0]))
    second.global_tree_id = 1
    second.persistent_tree_id = "tree-1"
    second.merged_span_tree_keys = ((1, 0),)
    reconciled.global_tree_records += (second,)
    reconciled.final_xyz_local_m = np.tile(reconciled.final_xyz_local_m, (2, 1))
    reconciled.final_original_index = np.arange(100, 300)
    reconciled.final_tree_id = np.repeat([0, 1], 100)
    _json(br / "span-two/tree/span_tree_manifest.json", {"span_id": 1})
    _json(
        br / "batch-reconciliation.json",
        {
            "trees": [
                {"global_tree_id": 0, "merged_span_tree_keys": [[0, 0]]},
                {"global_tree_id": 1, "merged_span_tree_keys": [[1, 0]]},
            ]
        },
    )
    with pytest.raises(ValueError, match="CROSS_TREE_EVIDENCE_OWNERSHIP_AMBIGUOUS"):
        recovery.recover_finalized_dbh(root, result, reconciled)
    assert not (br / "dbh_final.json").exists()


def test_lasso_fail_closed_and_undo_restores_exact_evidence(project):
    root, br, result, reconciled, _, _ = project
    baseline = recovery.recover_finalized_dbh(root, result, reconciled)
    op = {
        "type": "REMOVE_POINTS",
        "tree_ids": [0],
        "removed_indices_b64": _encode_indices(np.array([1, 2])),
        "point_count": 2,
    }
    _json(br / "tree_edits.json", {"analysis_run_id": "run", "operations": [op]})
    assert recovery.load_finalized_dbh(root, result, reconciled)[0]["dbh_cm"] is None
    rejected = recovery.recover_finalized_dbh(root, result, reconciled, persist=False)["trees"][0]
    assert rejected["dbh_fit_status"] == "EDITOR_STEM_SCOPE_UNPROVEN"
    assert rejected["editor_filtered_point_count"] == 432
    assert rejected["raw_local_query_used"] is False
    _json(br / "tree_edits.json", {"analysis_run_id": "run", "operations": []})
    assert (
        recovery.load_finalized_dbh(root, result, reconciled)[0]["dbh_cm"]
        == baseline["trees"][0]["dbh_cm"]
    )


def test_delete_merge_and_same_count_revision():
    records = [{"global_tree_id": i, "merged_span_tree_keys": [[0, i]]} for i in range(3)]
    status, removed = editor_eligibility(
        records,
        [
            {"type": "DELETE", "tree_ids": [0]},
            {"type": "MERGE", "tree_ids": [1, 2], "merged_id": 3},
        ],
    )
    assert status == {(0, 0): "DELETED", (0, 1): "ELIGIBLE", (0, 2): "ELIGIBLE"}
    assert len(removed) == 0
    op = {
        "type": "REMOVE_POINTS",
        "tree_ids": [0],
        "removed_indices_b64": _encode_indices(np.array([12])),
    }
    status, removed = editor_eligibility(
        records, [op, {"type": "MERGE", "tree_ids": [0, 1], "merged_id": 3}]
    )
    assert status[(0, 0)] == "EDITOR_STEM_SCOPE_UNPROVEN"
    assert removed.tolist() == [12]


@pytest.mark.parametrize(
    "tamper", ["membership", "journal", "ground", "evidence", "geometry", "source"]
)
def test_provenance_tampering_fails_closed(project, tamper):
    root, br, result, reconciled, _, _ = project
    payload = recovery.recover_finalized_dbh(root, result, reconciled)
    if tamper == "membership":
        reconciled.final_xyz_local_m[0, 0] += 1
    elif tamper == "journal":
        # Even a same-operation-count change cannot reuse a prior binding.
        _json(
            br / "tree_edits.json",
            {"analysis_run_id": "run", "operations": [], "revision_note": "changed"},
        )
    elif tamper == "ground":
        with next((root / "artifacts/project-runs/source-shared").glob("*/ground.npy")).open(
            "ab"
        ) as f:
            f.write(b"changed")
    elif tamper == "source":
        p = next((root / "artifacts/project-runs/source-shared").glob("*/preparation.json"))
        data = json.loads(p.read_text())
        data["ground_preparation"]["source_identity"]["sha256"] = "b" * 64
        _json(p, data)
    else:
        path = (
            root / payload["evidence_reference" if tamper == "evidence" else "published_reference"]
        )
        with path.open("ab") as f:
            f.write(b"changed")
    row = recovery.load_finalized_dbh(root, result, reconciled)[0]
    assert row["dbh_method"] == "UNAVAILABLE"
    assert row["dbh_fit_status"] == "DBH_PROOF_INVALID_OR_STALE"


def test_ambiguous_neighbor_and_merged_stems(project):
    root, _, result, reconciled, span, _ = project
    span["anchors"][1] = {"xy": np.array([0.7, 0.0]), "confidence": 0.95}
    row = recovery.recover_finalized_dbh(root, result, reconciled, persist=False)["trees"][0]
    assert row["dbh_fit_status"] == "NEIGHBOR_STEM_ASSOCIATION_AMBIGUOUS"
    assert row["ambiguous_point_count"] > 0
    reconciled.global_tree_records[0].merged_span_tree_keys = ((0, 0), (0, 1))
    row = recovery.recover_finalized_dbh(root, result, reconciled, persist=False)["trees"][0]
    assert row["dbh_fit_status"] == "MERGED_MULTIPLE_STEMS_AMBIGUOUS"


def _cache(root, source):
    xyz = _cylinder()
    block = np.zeros(len(xyz), dtype=ROLE_TILE_RECORD_DTYPE)
    block["x"], block["y"], block["z"] = xyz.T
    block["source_index"] = np.arange(1000, 1000 + len(xyz))
    p = root / "tiles/0.bin"
    p.parent.mkdir(parents=True)
    block.tofile(p)
    _json(
        root / "tiles/index.json",
        {
            "schema": "treeline-span-role-tile-index-v1",
            "tiles": {
                "0": {
                    "file": "0.bin",
                    "point_count": len(block),
                    "sha256": sha256(p),
                    "min_xyz": xyz.min(0).tolist(),
                    "max_xyz": xyz.max(0).tolist(),
                }
            },
        },
    )
    _json(
        root / "role-tile-cache.json",
        {
            "schema": "treeline-span-role-tile-cache-v1",
            "role": "vegetation",
            "admission_semantics": "HIGH_RECALL",
            "source_identity": source,
            "source_point_count": source["source_point_count"],
            "record_dtype": [list(v) for v in ROLE_TILE_RECORD_DTYPE.descr],
            "artifact_references": {
                "tile_index": {
                    "relative_path": "tiles/index.json",
                    "sha256": sha256(root / "tiles/index.json"),
                }
            },
        },
    )


def test_local_raw_roi_recovery_and_exact_indices(project):
    root, _, result, reconciled, span, source = project
    _cache(root / "raw-cache", source)
    # Persisted vertical stem evidence is present only above breast height.
    span["fine"]["fine_xyz_local_m"] = span["fine"]["fine_xyz_local_m"] + [0, 0, 2]
    span["fine"]["fine_hag_m"] = span["fine"]["fine_hag_m"] + 2
    row = recovery.recover_finalized_dbh(root, result, reconciled, persist=False)["trees"][0]
    assert row["raw_local_query_used"] is True
    assert row["raw_dbh_band_point_count"] == 432
    assert row["dbh_cm"] == pytest.approx(30, abs=0.1)
    assert row["opened_tiles"] == 1
    xyz, idx, query = local_raw_query(root / "raw-cache", source, np.array([0.0, 0.0]), (0.0, 2.0))
    assert idx.tolist() == list(range(1000, 1432))
    assert len(xyz) == query["raw_local_point_count"] == 432
    (root / "raw-cache/tiles/0.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="SIZE_MISMATCH"):
        local_raw_query(root / "raw-cache", source, np.zeros(2), (0.0, 2.0))


def test_old_project_compatibility_and_array_binding(project):
    root, _, result, reconciled, _, _ = project
    assert recovery.load_finalized_dbh(root, result, reconciled) == {}
    assert array_digest(np.array([1, 2])) != array_digest(np.array([2, 1]))


def test_api_and_excel_preserve_estimate_label(tmp_path):
    from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
    from desktop.current_report import _current_row
    from openpyxl import load_workbook

    row = {
        "persistent_tree_id": "tree",
        "crown_width_m": 2.0,
        **recovery.publication_fields(
            {
                **recovery.unavailable("PASS"),
                "dbh_cm": 28.4,
                "dbh_method": "ALLOMETRIC_ESTIMATE",
                "dbh_quality": "ESTIMATED",
                "dbh_model_version": "local-test",
                "dbh_uncertainty_cm": 3.0,
            }
        ),
    }
    api = _current_row(row)
    assert api["dbh_method"] == "ALLOMETRIC_ESTIMATE"
    assert api["diameter_cm"] == api["dbh_cm"] == 28.4
    for name in ["Survey.xlsx", "Prediction_Input.xlsx"]:
        path = write_prediction_input_excel(tmp_path / name, [row], {"science_run_id": "run"})
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            values = list(book["Trees"].values)
            result = dict(zip(values[0], values[1], strict=True))
            assert result["dbh_method"] == result["diameter_source"] == "ALLOMETRIC_ESTIMATE"
            assert result["dbh_cm"] == result["diameter_cm"] == 28.4
        finally:
            book.close()


def test_direct_quality_requires_twelve_retained_inliers_per_slice():
    from algorithms.manual_span.dbh_recovery import fit_stem_dbh

    angle = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    section = np.vstack(
        (
            np.column_stack((0.15 * np.cos(angle), 0.15 * np.sin(angle))),
            [[0.55, 0.55], [-0.55, -0.55]],
        )
    )
    xyz = np.vstack(
        [np.column_stack((section, np.full(12, h))) for h in np.linspace(0.925, 1.675, 6)]
    )
    result = fit_stem_dbh(xyz, xyz[:, 2], method="DIRECT_STEM")
    assert result["dbh_cm"] is None
    assert all(s["reason"] == "INSUFFICIENT_INLIER_POINTS" for s in result["slices"])


def test_ui_displays_estimate_source_and_unavailable_without_zero():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node required for exact UI formatter test")
    source = (
        Path(__file__).resolve().parents[2] / "viewer_poc/giro3d/current_report.js"
    ).read_text(encoding="utf-8")
    snippet = source[source.index("  const esc =") : source.index("  function showStatus")]
    snippet += (
        '\nconsole.log(JSON.stringify(["DIRECT_STEM","RECOVERED_STEM","ALLOMETRIC_ESTIMATE"]'
        ".map(dbh_method => dbhCell({dbh_cm:28.4,dbh_method}))"
        ".concat(dbhCell({dbh_cm:null,diameter_cm:null}))));"
    )
    result = subprocess.run(
        [node, "-e", snippet],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    values = json.loads(result.stdout)
    assert "点云" in values[0]
    assert "恢复" in values[1]
    assert "估算" in values[2]
    assert values[3] == "—"
