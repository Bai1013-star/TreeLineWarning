from __future__ import annotations

import copy

import numpy as np
import pytest
from algorithms.folder_pipeline.tree_editor import _encode_indices
from core.stage_b import dbh_raw_recovery as v2
from core.stage_b import dbh_recovery as v1

from tests.stage_b.test_dbh_recovery import _json
from tests.stage_b.test_dbh_recovery import project as v1_project  # noqa: F401


def rings(heights, radius=0.15):
    angle = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    return np.vstack(
        [
            np.column_stack(
                (radius * np.cos(angle), radius * np.sin(angle), np.full(len(angle), h))
            )
            for h in heights
        ]
    )


def test_current_mode_rejects_stale_historical_ownership():
    xyz = rings([2.25, 2.75, 3.25, 3.75])
    seeds, proof = v2.current_seeds(xyz, xyz[:, 2], [20, 20])
    assert seeds and all(np.linalg.norm(s["xy"]) < 0.3 for s in seeds)
    assert proof["historical_hint_status"] == "REJECTED"
    assert proof["historical_hint_used_as_owner"] is False


def test_validated_stem_hint_keeps_bounded_roi_when_three_crown_modes_are_full():
    groups = []
    for center, count in ((0.0, 80), (2.0, 70), (4.0, 60), (7.0, 20)):
        groups.append(np.column_stack((
            np.full(count, center), np.zeros(count), np.full(count, 2.5)
        )))
    xyz = np.vstack(groups)
    seeds, proof = v2.current_seeds(xyz, xyz[:, 2], [7.0, 0.0])
    assert len(seeds) == v2.MAX_SEEDS == 4
    assert [seed["source"] for seed in seeds].count("CURRENT_SUPPORTED_HINT") == 1
    assert seeds[-1]["xy"] == [7.0, 0.0]
    assert proof["historical_hint_current_support"] == 20

    surviving = xyz[:-20]
    after_delete, proof = v2.current_seeds(surviving, surviving[:, 2], [7.0, 0.0])
    assert not any(seed["source"] == "CURRENT_SUPPORTED_HINT" for seed in after_delete)
    assert proof["historical_hint_status"] == "REJECTED"


def test_current_fragment_cannot_inherit_a_complete_historical_stem():
    xyz = rings([2.25, 2.30, 2.35])
    proof = v2.axis_connection(xyz, xyz[:, 2], np.array([0, 0, 1.3]), np.array([0, 0, 1]), 30)
    assert proof["status"] == "CURRENT_CONNECTION_UNPROVEN"


def test_raw_bridge_requires_two_current_upper_bins_and_continuous_source():
    raw = rings(np.arange(2.05, 4.51, 0.1))
    current = rings([3.55, 3.65, 4.05, 4.15])
    origin = np.array([0, 0, 1.3])
    direction = np.array([0, 0, 1])
    connected = v2.raw_bridged_axis_connection(
        current, current[:, 2], raw, raw[:, 2], origin, direction, 30
    )
    assert connected["status"] == "PASS"
    assert connected["first_current_upper_bin_hag_m"] == 3.5
    gap = raw[(raw[:, 2] < 2.5) | (raw[:, 2] >= 3.0)]
    assert v2.raw_bridged_axis_connection(
        current, current[:, 2], gap, gap[:, 2], origin, direction, 30
    )["status"] != "PASS"
    only_one_upper_bin = rings([3.55, 3.65])
    assert v2.raw_bridged_axis_connection(
        only_one_upper_bin, only_one_upper_bin[:, 2], raw, raw[:, 2], origin, direction, 30
    )["status"] != "PASS"


def test_raw_gap_between_dbh_and_current_crown_is_rejected():
    xyz = rings([1.375, 1.675, 2.25, 2.75, 3.25])
    proof = v2.raw_bridge(xyz, xyz[:, 2], np.array([0, 0, 1.3]), np.array([0, 0, 1]), 30)
    assert proof["status"] == "RAW_TO_CURRENT_BRIDGE_UNPROVEN"


def test_even_one_selected_point_owned_by_neighbor_is_rejected():
    source = np.array([10, 20, 30])
    owners = np.array([0, 1, 0])
    assert v2.conflicting_current_owner(np.array([10, 20, 100]), source, owners, 0)
    assert not v2.conflicting_current_owner(np.array([10, 30, 100]), source, owners, 0)


def test_delete_after_merge_and_undo_exact_source_exclusion(tmp_path):
    records = [{"global_tree_id": i, "merged_span_tree_keys": [[0, i]]} for i in range(3)]
    np.savez(
        tmp_path / "tree_segmentation_original.npz",
        original_index=[10, 11, 20, 21, 30],
        tree_id=[0, 0, 1, 1, 2],
    )
    remove = {
        "type": "REMOVE_POINTS",
        "tree_ids": [2],
        "removed_indices_b64": _encode_indices(np.array([30])),
        "point_count": 1,
    }
    ops = [
        {"type": "MERGE", "tree_ids": [0, 1], "merged_id": 3},
        {"type": "DELETE", "tree_ids": [3]},
        remove,
    ]
    assert v2.current_exclusions(tmp_path, tmp_path, records, ops).tolist() == [10, 11, 20, 21, 30]
    assert v2.current_exclusions(tmp_path, tmp_path, records, ops[:1]).tolist() == []


def test_two_current_stems_are_ambiguous_but_repeat_roi_is_not():
    candidate = {
        "candidate_id": 0,
        "association_status": "PASS",
        "axis_origin": [0, 0, 1.3],
        "fit": {"dbh_slice_count": 6, "dbh_fit_rmse": 0.002},
    }
    repeat = copy.deepcopy(candidate)
    repeat["candidate_id"] = 1
    repeat["axis_origin"] = [0.02, 0, 1.3]
    assert v2.select_candidate([repeat, candidate])[1] == "PASS"
    repeat["axis_origin"] = [0.4, 0, 1.3]
    assert v2.select_candidate([repeat, candidate]) == (None, "MULTIPLE_CURRENT_STEMS_AMBIGUOUS")


@pytest.fixture
def current_project(v1_project):  # noqa: F811 - imported pytest fixture
    root, br, result, members, _, source = v1_project
    xyz = rings([2.25, 2.75, 3.25, 3.75])
    members.final_xyz_local_m = xyz
    members.final_original_index = np.arange(100, 100 + len(xyz))
    members.final_tree_id = np.zeros(len(xyz), dtype=np.int32)
    members.global_tree_records[0].full_resolution_point_count = len(xyz)
    np.savez(
        br / "tree_segmentation_final.npz",
        xyz_local_m=xyz,
        original_index=members.final_original_index,
        tree_id=members.final_tree_id,
    )
    op = {
        "type": "REMOVE_POINTS",
        "tree_ids": [0],
        "point_count": 1,
        "removed_indices_b64": _encode_indices(np.array([1072])),
    }
    _json(br / "tree_edits.json", {"analysis_run_id": "run", "operations": [op]})
    row = {
        **v1.unavailable("EDITOR_STEM_SCOPE_UNPROVEN"),
        "global_tree_id": 0,
        "persistent_tree_id": "tree-0",
        "height_m": 4.0,
        "stem_axis_xy_local_m": [20, 20],
    }
    base = {
        "schema": "treeline-dbh-evidence-v1",
        "binding": v1._binding(root, result, members),
        "trees": [row],
    }
    empty = {
        "xyz_local_m": np.empty((0, 3)),
        "hag_m": np.empty(0),
        "original_index": np.empty(0, dtype=np.int64),
        "global_tree_id": np.empty(0, dtype=np.int32),
    }
    v1._persist(root, result, base, empty)
    raw = rings(
        [
            0.6,
            0.925,
            1.075,
            1.225,
            1.375,
            1.525,
            1.675,
            1.85,
            2.125,
            2.375,
            2.625,
            2.875,
            3.125,
            3.375,
            3.625,
            3.875,
        ]
    )

    def query(*args, **kwargs):
        return raw.copy(), np.arange(1000, 1000 + len(raw)), {"source_identity": source}

    return root, br, result, members, query


def test_current_raw_reassociation_lasso_exclusion_restart_and_repeat(current_project):
    root, br, result, members, query = current_project
    before = v1.member_binding(members)
    payload = v2.recover_finalized_dbh_v2(root, result, members, query_provider=query)
    row = payload["trees"][0]
    assert row["dbh_cm"] == pytest.approx(30, abs=0.1)
    assert row["dbh_method"] == "RECOVERED_STEM"
    assert row["dbh_source"] == "DBH_RECOVERY_V2"
    assert row["editor_excluded_point_count"] == 1
    assert row["current_geometry"]["historical_hint_status"] == "REJECTED"
    with np.load(root / payload["evidence_reference"], allow_pickle=False) as archive:
        assert 1072 not in archive["original_index"]
    assert v1.member_binding(members) == before
    assert v1.load_finalized_dbh(root, result, members)[0]["dbh_cm"] == row["dbh_cm"]
    second = v2.recover_finalized_dbh_v2(
        root,
        result,
        members,
        persist=False,
        query_provider=lambda *a: pytest.fail("accepted stem refitted"),
    )
    assert second["trees"] == payload["trees"]
    _json(br / "tree_edits.json", {"analysis_run_id": "run", "operations": []})
    with pytest.raises(ValueError, match="STALE_BASELINE"):
        v2.recover_finalized_dbh_v2(root, result, members, persist=False, query_provider=query)


def test_v2_numeric_api_and_workbook_keep_recovered_label(current_project, tmp_path):
    from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
    from desktop.current_report import _current_row
    from openpyxl import load_workbook

    root, _, result, members, query = current_project
    row = v2.recover_finalized_dbh_v2(root, result, members, query_provider=query)["trees"][0]
    api = _current_row({"persistent_tree_id": "tree-0", **v1.publication_fields(row)})
    assert api["dbh_method"] == "RECOVERED_STEM" and api["dbh_source"] == v2.VERSION
    for name in ("Survey.xlsx", "Prediction_Input.xlsx"):
        path = write_prediction_input_excel(tmp_path / name, [api], {"science_run_id": "run"})
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            values = list(book["Trees"].values)
            cells = dict(zip(values[0], values[1], strict=True))
            assert cells["dbh_cm"] == pytest.approx(30, abs=0.1)
            assert cells["dbh_method"] == "RECOVERED_STEM"
            assert cells["diameter_source"] == "RECOVERED_STEM"
        finally:
            book.close()


def test_neighboring_current_tree_connection_fails_closed(current_project):
    root, br, result, members, query = current_project
    base, arrays = v2._load_base(root, result, members)
    neighbor = copy.deepcopy(members.global_tree_records[0])
    neighbor.global_tree_id = 1
    neighbor.persistent_tree_id = "neighbor"
    members.global_tree_records = (*members.global_tree_records, neighbor)
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
    base["binding"] = v1._binding(root, result, members)
    base["trees"].append(
        {**base["trees"][0], "global_tree_id": 1, "persistent_tree_id": "neighbor"}
    )
    v1._persist(root, result, base, arrays)
    recovered = v2.recover_finalized_dbh_v2(root, result, members, query_provider=query)
    assert all(r["dbh_cm"] is None for r in recovered["trees"])
    assert all(
        r["dbh_fit_status"] == "NEIGHBOR_STEM_OWNERSHIP_AMBIGUOUS" for r in recovered["trees"]
    )


def test_additional_deletion_source_proof_is_validated_on_restart(current_project):
    root, _, result, members, query = current_project
    payload = v2.recover_finalized_dbh_v2(root, result, members, query_provider=query)
    with np.load(root / payload["evidence_reference"], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    proof = root / "original-members-proof.npz"
    proof.write_bytes(b"original")
    payload["additional_input_hashes"] = [{"reference": proof.name, "sha256": v2.sha256(proof)}]
    v1._persist(root, result, payload, arrays)
    assert v1.load_finalized_dbh(root, result, members)[0]["dbh_cm"] is not None
    proof.write_bytes(b"tampered")
    assert (
        v1.load_finalized_dbh(root, result, members)[0]["dbh_fit_status"]
        == "DBH_PROOF_INVALID_OR_STALE"
    )


def test_axis_summary_counts_stems_not_repeated_pair_proposals():
    axis = {"bh_reference_xyz": [0, 0, 1.3], "direction_xyz": [0, 0, 1]}
    second = {**axis, "bh_reference_xyz": [0.4, 0, 1.3]}
    row = {
        "dbh_cm": None,
        "v2_candidates": [
            {
                "association_status": "AMBIGUOUS_COMPETING_AXES",
                "fit": {"axis": {"plausible_candidates": [axis, axis, second]}},
            }
        ],
    }
    v2.summarize_candidates(row)
    assert row["stem_candidate_count"] == 2
    assert row["plausible_pair_proposal_count"] == 3
    assert row["dbh_fit_status"] == "AMBIGUOUS_COMPETING_AXES"


def test_finalize_entry_uses_current_raw_recovery_without_rewriting_members(
    current_project, monkeypatch
):
    root, br, result, members, query = current_project
    before = v2.sha256(br / "tree_segmentation_final.npz")
    monkeypatch.setattr(v2, "local_raw_query", query)
    payload = v2.finalize_dbh(root, result, members)
    assert payload["trees"][0]["dbh_source"] == v2.VERSION
    assert payload["trees"][0]["dbh_cm"] == pytest.approx(30, abs=0.1)
    assert v2.sha256(br / "tree_segmentation_final.npz") == before
