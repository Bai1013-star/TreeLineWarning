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
    # This fixture represents a tree growing at the coarse-ground elevation.
    # Root-height auditing requires actual nearby raw ground evidence.
    angle = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    radii = np.linspace(0.25, 0.55, 8)
    rr, aa = np.meshgrid(radii, angle)
    ground = np.column_stack((rr.ravel() * np.cos(aa.ravel()),
                              rr.ravel() * np.sin(aa.ravel()), np.zeros(rr.size)))
    low_stem = np.vstack([
        np.column_stack((.15 * np.cos(angle), .15 * np.sin(angle),
                         np.full(len(angle), h)))
        for h in (.10, .25, .40)
    ])
    xyz = np.vstack((xyz, ground, low_stem))
    return lambda *args, **kwargs: (
        xyz.copy(), np.arange(1000, 1000 + len(xyz)), {"source_identity": source}
    )


def test_current_crown_recovery_editor_exclusion_and_restart(current_project):  # noqa: F811
    root, _, result, members, _ = current_project
    before = v1.member_binding(members)
    payload = v3.recover_finalized_dbh_v3(
        root, result, members, query_provider=dense_query(current_project)
    )
    row = payload["trees"][0]
    assert row["dbh_cm"] == pytest.approx(30, abs=0.15), row["v3_candidates"][0]["local_base_probe"]
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


def test_recompute_auto_revisits_prior_v3_measurement(current_project):  # noqa: F811
    root, _, result, members, _ = current_project
    first = v3.recover_finalized_dbh_v3(
        root, result, members, query_provider=dense_query(current_project)
    )
    assert first["trees"][0]["dbh_cm"] is not None
    calls = []

    def repeated_query(*args):
        calls.append(1)
        return dense_query(current_project)(*args)

    recomputed = v3.recover_finalized_dbh_v3(
        root, result, members, persist=False, recompute_auto=True,
        query_provider=repeated_query,
    )
    assert calls
    assert recomputed["trees"][0]["dbh_cm"] == pytest.approx(first["trees"][0]["dbh_cm"])


def test_prior_reference_value_is_not_fixed_and_can_be_remeasured(current_project):  # noqa: F811
    root, _, result, members, _ = current_project
    first = v3.recover_finalized_dbh_v3(
        root, result, members, query_provider=dense_query(current_project)
    )
    base, arrays = v3.v2._load_base(root, result, members)
    reference_row = base["trees"][0]
    reference_row.update(v1.unavailable("NO_CROWN_TO_STEM_IDENTITY"))
    reference_row.update(dbh_cm=22.0, dbh_method="REFERENCE_ESTIMATE",
                         dbh_quality="ESTIMATE", dbh_fit_status="REFERENCE_ESTIMATE",
                         dbh_source="DBH_RECOVERY_V3_REFERENCE",
                         dbh_point_count=0, dbh_evidence_point_count=0,
                         dbh_uncertainty_cm=None,
                         dbh_pre_reference_failure_reason="NO_CROWN_TO_STEM_IDENTITY",
                         dbh_reference_estimate={"value_cm": 22.0,
                                                 "basis": "PROJECT_TRUSTED_MEASUREMENTS"})
    empty = {key: value[:0] for key, value in arrays.items()}
    v1._persist(root, result, base, empty)
    assert v1.load_finalized_dbh(root, result, members)[0]["dbh_method"] == "REFERENCE_ESTIMATE"
    calls = []

    def query(*args):
        calls.append(1)
        return dense_query(current_project)(*args)

    recomputed = v3.recover_finalized_dbh_v3(
        root, result, members, persist=False, query_provider=query
    )
    assert calls
    assert recomputed["trees"][0]["dbh_method"] == "RECOVERED_STEM"
    assert recomputed["trees"][0]["dbh_cm"] == pytest.approx(first["trees"][0]["dbh_cm"])
    assert "dbh_reference_estimate" not in recomputed["trees"][0]


def test_current_v3_unavailable_cache_is_reused_until_explicit_recompute(
    current_project, monkeypatch
):  # noqa: F811
    root, _, result, members, _ = current_project
    base, arrays = v3.v2._load_base(root, result, members)
    base["trees"][0].update(
        dbh_source=v3.VERSION,
        dbh_fit_status="NO_CROWN_TO_STEM_IDENTITY",
        v3_candidates=[],
    )
    base["recovery_version"] = v3.VERSION
    v1._persist(root, result, base, arrays)
    calls = []

    def upgraded(*args, **kwargs):
        calls.append(kwargs)
        return {"upgraded": True}

    monkeypatch.setattr(v3, "recover_finalized_dbh_v3", upgraded)
    cached = v3.finalize_dbh(root, result, members)
    assert cached["cache_hit"] is True
    assert cached["trees"][0]["dbh_fit_status"] == "NO_CROWN_TO_STEM_IDENTITY"
    assert calls == []
    assert v3.finalize_dbh(root, result, members, force_recompute=True) == {"upgraded": True}
    assert calls == [{"recompute_auto": True, "reference_dbh_cm": None}]



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
    assert payload["trees"][0]["dbh_source"] == v1.load_finalized_dbh(
        root, result, members
    )[0]["dbh_source"]
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


def test_legacy_complete_ring_is_retired_when_local_root_is_elevated(current_project):  # noqa: F811
    """A clear old V2 circle cannot certify 1.30 m above the wrong ground."""
    from tests.manual_span.test_stem_identity import cylinder

    root, _, result, members, _ = current_project
    base, old_arrays = v3.v2._load_base(root, result, members)
    stem, true_hag, _ = cylinder(offset=(0, 0, 1.0))
    chosen = np.arange(0, len(stem), 7)
    arrays = {
        "xyz_local_m": stem[chosen], "hag_m": true_hag[chosen],
        "original_index": (1000 + chosen).astype(np.int64),
        "global_tree_id": np.zeros(len(chosen), np.int32),
    }
    row = base["trees"][0]
    row.update(dbh_cm=30.0, dbh_method="RECOVERED_STEM", dbh_quality="HIGH",
               dbh_fit_status="PASS", dbh_source="DBH_RECOVERY_V2")
    v1._persist(root, result, base, arrays)
    angle = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    lower = np.vstack([
        np.column_stack((0.15 * np.cos(angle), 0.15 * np.sin(angle),
                         np.full(72, 1 + h)))
        for h in (0.05, 0.20, 0.30)
    ])
    rad = np.linspace(0.25, 0.58, 9)
    rr, aa = np.meshgrid(rad, angle)
    sheet = np.column_stack((rr.ravel() * np.cos(aa.ravel()),
                             rr.ravel() * np.sin(aa.ravel()), np.ones(rr.size)))
    raw = np.vstack((stem, lower, sheet))
    source = dense_query(current_project)()[2]["source_identity"]
    assert len(raw) + 1000 < source["source_point_count"]

    payload = v3.recover_finalized_dbh_v3(
        root, result, members, persist=False,
        query_provider=lambda *args: (
            raw.copy(), np.arange(1000, 1000 + len(raw)), {"source_identity": source}
        ),
    )
    current = payload["trees"][0]
    assert current["dbh_cm"] is None
    assert current["legacy_base_probe"]["status"] == "LOCAL_BASE_HEIGHT_AMBIGUOUS"
    assert current["dbh_stale_prior_geometry"]["value_cm"] == 30.0
