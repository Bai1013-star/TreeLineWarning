from __future__ import annotations

import numpy as np
import pytest
from core.stage_b import dbh_identity_recovery as v3
from core.stage_b import dbh_recovery as v1

from tests.manual_span.test_stem_identity import cylinder
from tests.stage_b.test_dbh_raw_recovery import (
    current_project,  # noqa: F401
    v1_project,  # noqa: F401
)


def dense_query(current):
    root, _, result, _, old_query = current
    source = old_query()[2]["source_identity"]
    xyz, hag, _ = cylinder()
    assert np.allclose(xyz[:, 2], hag)
    return lambda *args: (xyz.copy(), np.arange(1000, 1000 + len(xyz)), {"source_identity": source})


def test_current_crown_recovery_editor_exclusion_and_restart(current_project):  # noqa: F811
    root, _, result, members, _ = current_project
    before = v1.member_binding(members)
    payload = v3.recover_finalized_dbh_v3(
        root, result, members, query_provider=dense_query(current_project)
    )
    row = payload["trees"][0]
    assert row["dbh_cm"] == pytest.approx(30, abs=0.15), row["v3_candidates"]
    assert row["dbh_method"] == "RECOVERED_STEM"
    assert row["dbh_source"] == v3.VERSION
    assert row["v3_editor_excluded_point_count"] == 1
    with np.load(root / payload["evidence_reference"], allow_pickle=False) as archive:
        assert 1072 not in archive["original_index"]
        assert len(np.unique(archive["original_index"])) == len(archive["original_index"])
    assert before == v1.member_binding(members)
    assert v1.load_finalized_dbh(root, result, members)[0]["dbh_cm"] == row["dbh_cm"]
    repeated = v3.recover_finalized_dbh_v3(
        root,
        result,
        members,
        persist=False,
        query_provider=lambda *a: pytest.fail("accepted DBH must not be refitted"),
    )
    assert repeated["trees"] == payload["trees"]
    assert payload["allometry"]["model_used"] is False


def test_deleted_breast_height_stem_cannot_be_reintroduced(current_project):  # noqa: F811
    from algorithms.folder_pipeline.tree_editor import _encode_indices

    from tests.stage_b.test_dbh_recovery import _json

    root, br, result, members, _ = current_project
    base, arrays = v3.v2._load_base(root, result, members)
    query = dense_query(current_project)
    raw, idx, _ = query()
    deleted = idx[(raw[:, 2] >= 1.20) & (raw[:, 2] <= 1.45)]
    _json(
        br / "tree_edits.json",
        {
            "analysis_run_id": "run",
            "operations": [
                {
                    "type": "REMOVE_POINTS",
                    "tree_ids": [0],
                    "point_count": len(deleted),
                    "removed_indices_b64": _encode_indices(deleted),
                }
            ],
        },
    )
    base["binding"] = v1._binding(root, result, members)
    v1._persist(root, result, base, arrays)
    payload = v3.recover_finalized_dbh_v3(root, result, members, query_provider=query)
    assert payload["trees"][0]["dbh_cm"] is None
    assert payload["trees"][0]["v3_editor_excluded_point_count"] == len(deleted)


def test_v3_api_and_exports_keep_recovered_semantics(current_project, tmp_path):  # noqa: F811
    from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
    from desktop.current_report import _current_row
    from openpyxl import load_workbook

    root, _, result, members, _ = current_project
    payload = v3.recover_finalized_dbh_v3(
        root, result, members, query_provider=dense_query(current_project)
    )
    row = payload["trees"][0]
    api = _current_row({"persistent_tree_id": "tree-0", **v1.publication_fields(row)})
    assert api["dbh_cm"] == row["dbh_cm"]
    for filename in ("Survey.xlsx", "Prediction_Input.xlsx"):
        path = write_prediction_input_excel(tmp_path / filename, [api], {"science_run_id": "run"})
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            values = list(book["Trees"].values)
            cells = dict(zip(values[0], values[1], strict=True))
            assert cells["dbh_cm"] == pytest.approx(30, abs=0.15)
            assert cells["dbh_method"] == "RECOVERED_STEM"
        finally:
            book.close()


def test_v3_finalize_entry_keeps_existing_member_artifact(current_project, monkeypatch):  # noqa: F811
    root, br, result, members, _ = current_project
    before = v3.sha256(br / "tree_segmentation_final.npz")
    monkeypatch.setattr(v3, "local_raw_query", dense_query(current_project))
    payload = v3.finalize_dbh(root, result, members)
    assert payload["trees"][0]["dbh_source"] == v3.VERSION
    assert payload["trees"][0]["dbh_cm"] == pytest.approx(30, abs=0.15)
    assert v3.sha256(br / "tree_segmentation_final.npz") == before


def test_two_current_crowns_cannot_claim_same_raw_stem(current_project):  # noqa: F811
    import copy

    root, br, result, members, _ = current_project
    base, arrays = v3.v2._load_base(root, result, members)
    other = copy.deepcopy(members.global_tree_records[0])
    other.global_tree_id, other.persistent_tree_id = 1, "neighbor"
    members.global_tree_records = (*members.global_tree_records, other)
    xyz = members.final_xyz_local_m.copy()
    members.final_xyz_local_m = np.vstack((xyz, xyz + [0.005, 0, 0]))
    members.final_tree_id = np.repeat([0, 1], len(xyz)).astype(np.int32)
    members.final_original_index = np.arange(100, 100 + 2 * len(xyz))
    np.savez(
        br / "tree_segmentation_final.npz",
        xyz_local_m=members.final_xyz_local_m,
        tree_id=members.final_tree_id,
        original_index=members.final_original_index,
    )
    base["trees"].append(
        {**base["trees"][0], "global_tree_id": 1, "persistent_tree_id": "neighbor"}
    )
    base["binding"] = v1._binding(root, result, members)
    v1._persist(root, result, base, arrays)
    payload = v3.recover_finalized_dbh_v3(
        root, result, members, query_provider=dense_query(current_project)
    )
    assert all(row["dbh_cm"] is None for row in payload["trees"])
    assert not payload["global_ownership"]["selected"]
    assert any(d["conflicts"] for d in payload["global_ownership"]["decisions"].values())


def test_existing_estimate_is_preserved_without_inventing_stem_reservation(current_project):  # noqa: F811
    root, _, result, members, _ = current_project
    base, arrays = v3.v2._load_base(root, result, members)
    row = base["trees"][0]
    row.update(
        dbh_cm=28.0,
        dbh_method="ALLOMETRIC_ESTIMATE",
        dbh_quality="ESTIMATED",
        dbh_model_version="existing-local-model",
        dbh_fit_status="PASS",
    )
    v1._persist(root, result, base, arrays)
    payload = v3.recover_finalized_dbh_v3(
        root,
        result,
        members,
        persist=False,
        query_provider=lambda *a: pytest.fail("existing estimate must not be refitted"),
    )
    assert payload["trees"][0] == row
    assert payload["global_ownership"]["selected"] == {}
    assert payload["allometry"]["reliable_count"] == 0
