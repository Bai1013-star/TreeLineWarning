from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from core.stage_b import dbh_review as review

from tests.manual_span.test_stem_identity import cylinder
from tests.stage_b.test_dbh_identity_recovery import dense_query
from tests.stage_b.test_dbh_raw_recovery import current_project, v1_project  # noqa: F401
from tests.stage_b.test_dbh_recovery import _json


def ready(current, *, candidates=True, query=None):
    root, _, result, members, _ = current
    base, arrays = review.v2._load_base(root, result, members)
    _, _, axis = cylinder()
    if candidates:
        base["trees"][0]["v3_candidates"] = [
            {
                "candidate_id": "axis-1",
                "score": 0.5,
                "axis": axis,
                "track": {"status": "CROWN_TO_BH_TRACK_INCOMPLETE", "refined_axis": None},
                "identity_support_count": 100,
            }
        ]
        review.v1._persist(root, result, base, arrays)
    data = review.prepare_review(
        root, result, members, "tree-0", query_provider=query or dense_query(current)
    )
    return data


def request(data, **choice):
    return {"review_id": data["review_id"], "identity_confirmed": True, **choice}


@pytest.mark.parametrize("mode", ["candidate", "seed"])
def test_confirmation_geometry_persistence_restart_and_exports(current_project, tmp_path, mode):  # noqa: F811
    from algorithms.folder_pipeline.export_excel import write_prediction_input_excel
    from desktop.current_report import _current_row
    from openpyxl import load_workbook

    root, br, result, members, _ = current_project
    data = ready(current_project, candidates=mode == "candidate")
    before = review.v1.member_binding(members)
    files = {
        name: review.sha256(br / name)
        for name in ["tree_edits.json", "tree_segmentation_final.npz", "batch.json"]
    }
    choice = (
        {"candidate_id": "axis-1"}
        if mode == "candidate"
        else {"seed_original_index": data["display"]["original_index"][0]}
    )
    answer = review.confirm_review(root, result, members, request(data, **choice))
    assert answer["status"] == "RECOVERED", answer
    assert answer["dbh_cm"] == pytest.approx(30, abs=0.15)
    assert answer["dbh_method"] == review.METHOD
    published = review.v1.load_finalized_dbh(root, result, members)[0]
    assert published["dbh_review"]["identity_confirmed"] is True
    assert 1072 not in published["dbh_review"]["selected_original_index"]
    assert review.v1.member_binding(members) == before
    assert all(review.sha256(br / name) == digest for name, digest in files.items())
    if mode == "candidate":
        # A new interpreter rehydrates exact synthetic members and reads proof
        # from disk; no module cache, monkeypatch, UI choice or fitter is reused.
        snapshot = tmp_path / "restart_members.npz"
        np.savez(
            snapshot,
            xyz=members.final_xyz_local_m,
            idx=members.final_original_index,
            tree=members.final_tree_id,
        )
        meta = tmp_path / "restart.json"
        meta.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "members": str(snapshot),
                    "records": [vars(r) for r in members.global_tree_records],
                    "result": {**vars(result), "telemetry": vars(result.telemetry)},
                }
            ),
            encoding="utf-8",
        )
        code = """
import json,sys,numpy as np
from pathlib import Path
from types import SimpleNamespace as NS
from core.stage_b.dbh_recovery import load_finalized_dbh
d=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
r=NS(**d['result']);r.telemetry=NS(**r.telemetry)
with np.load(d['members'],allow_pickle=False) as a:
 m=NS(final_xyz_local_m=a['xyz'],final_original_index=a['idx'],final_tree_id=a['tree'],
      global_tree_records=tuple(NS(**x) for x in d['records']))
row=load_finalized_dbh(Path(d['root']),r,m)[0]
assert row['dbh_method']=='REVIEW_CONFIRMED_STEM',row
assert abs(row['dbh_cm']-30)<.15
print('FRESH_RESTART_PASS')
"""
        child = subprocess.run(
            [sys.executable, "-B", "-c", code, str(meta)],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == 0, child.stdout + child.stderr
    api = _current_row({"persistent_tree_id": "tree-0", **review.v1.publication_fields(published)})
    for name in ["Survey.xlsx", "Prediction_Input.xlsx"]:
        path = write_prediction_input_excel(tmp_path / name, [api], {"science_run_id": "run"})
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = list(book["Trees"].values)
            cells = dict(zip(rows[0], rows[1], strict=True))
            assert cells["dbh_method"] == review.METHOD
            assert cells["dbh_cm"] == pytest.approx(30, abs=0.15)
        finally:
            book.close()
    with pytest.raises(ValueError, match="STALE|LOCKED|ALREADY"):
        review.confirm_review(root, result, members, request(data, **choice))
    with pytest.raises(ValueError, match="ACCEPTED_VALUE_LOCKED"):
        review.prepare_review(root, result, members, "tree-0")


@pytest.mark.parametrize(
    "bad",
    [
        {"dbh_cm": 30},
        {"seed_original_index": 1072},
        {"seed_original_index": -1},
        {"seed_original_index": True},
        {"seed_xyz": [0, 0, 1.3]},
        {"candidate_id": "unknown"},
        {"candidate_id": "axis-1", "seed_original_index": 1000},
        {"identity_confirmed": False},
    ],
)
def test_invalid_input_cannot_publish(current_project, bad):  # noqa: F811
    root, br, result, members, _ = current_project
    data = ready(current_project)
    before = review.sha256(br / "dbh_final.json")
    with pytest.raises(ValueError):
        review.confirm_review(root, result, members, {**request(data), **bad})
    assert review.sha256(br / "dbh_final.json") == before


@pytest.mark.parametrize("action", ["DELETE", "MERGE", "UNDO"])
def test_any_editor_revision_change_invalidates_session(current_project, action):  # noqa: F811
    root, br, result, members, _ = current_project
    data = ready(current_project)
    journal = review.read_json(br / "tree_edits.json")
    if action == "UNDO":
        journal["operations"].pop()
    else:
        journal["operations"].append({"type": action, "tree_ids": [0], "merged_id": 2})
    _json(br / "tree_edits.json", journal)
    with pytest.raises(ValueError, match="STALE"):
        review.confirm_review(root, result, members, request(data, candidate_id="axis-1"))
    assert review.v1.load_finalized_dbh(root, result, members)[0]["dbh_cm"] is None


def test_confirmation_does_not_override_missing_bh_geometry(current_project):  # noqa: F811
    root, br, result, members, _ = current_project
    query = dense_query(current_project)

    def gap(*args):
        xyz, idx, proof = query(*args)
        keep = (xyz[:, 2] < 1.1) | (xyz[:, 2] > 1.6)
        return xyz[keep], idx[keep], proof

    data = ready(current_project, query=gap)
    before = review.sha256(br / "dbh_final.json")
    answer = review.confirm_review(root, result, members, request(data, candidate_id="axis-1"))
    assert answer["status"] == "UNAVAILABLE" and answer["dbh_cm"] is None
    assert review.sha256(br / "dbh_final.json") == before
    folder = review._session_root(root, result, data["review_id"])
    assert review.read_json(folder / "decision.json")["fit"]["dbh_cm"] is None


def test_neighbor_seed_and_accepted_value_remain_protected(current_project):  # noqa: F811
    root, _, result, members, _ = current_project
    base, arrays = review.v2._load_base(root, result, members)
    old = copy.deepcopy(base["trees"][0])
    old.update(
        global_tree_id=7,
        persistent_tree_id="accepted",
        dbh_cm=19.25,
        dbh_method="RECOVERED_STEM",
        dbh_fit_status="PASS",
    )
    # Direct source-owner mask test independent of additional final identities.
    arrays = {"original_index": np.array([1000]), "global_tree_id": np.array([7])}
    blocked = review._blocked(np.array([1000, 1001]), arrays, members, 0)
    assert blocked.tolist() == [True, False]
    base["trees"].append(old)
    with pytest.raises(ValueError, match="ACCEPTED_VALUE_LOCKED"):
        review._row(base, "accepted")


@pytest.mark.parametrize("tilt", [0, 12])
def test_candidate_tilt_and_seed_still_use_original_hag(tilt):
    xyz, hag, axis = cylinder(tilt, (10, 20, 30))
    original_hag = hag.copy()
    fit, indices = review.fit_review_stem(xyz, hag, candidate_axis=axis)
    assert fit["dbh_cm"] == pytest.approx(30, abs=0.15)
    assert len(indices) > 0 and np.array_equal(hag, original_hag)


def test_two_stems_ambiguous_seed_and_explicit_candidate():
    xyz, hag, axis = cylinder()
    other = xyz + [0.45, 0, 0]
    cloud, heights = np.vstack((xyz, other)), np.tile(hag, 2)
    # Seed is on the near-facing side of one stem; both axes remain in its .6m ROI.
    fit, _ = review.fit_review_stem(cloud, heights, seed_position=0)
    assert fit["dbh_cm"] is None
    fit, _ = review.fit_review_stem(cloud, heights, candidate_axis=axis)
    assert fit["dbh_cm"] == pytest.approx(30, abs=0.15)


def test_session_tampering_fails_closed(current_project):  # noqa: F811
    root, _, result, members, _ = current_project
    data = ready(current_project)
    path = review._session_root(root, result, data["review_id"]) / "session.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        review.confirm_review(root, result, members, request(data, candidate_id="axis-1"))


def test_seven_existing_measurements_and_arrays_are_exact_after_confirmation(current_project):  # noqa: F811
    from types import SimpleNamespace

    root, br, result, members, _ = current_project
    base, arrays = review.v2._load_base(root, result, members)
    angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = np.vstack(
        [
            np.column_stack((0.15 * np.cos(angle), 0.15 * np.sin(angle), np.full(12, h)))
            for h in [0.925, 1.075, 1.225, 1.375, 1.525, 1.675]
        ]
    )
    blocks = []
    for gid in range(1, 8):
        xyz = points + [gid * 3, 0, 0]
        row = {
            **review.v1.unavailable("PASS"),
            "global_tree_id": gid,
            "persistent_tree_id": f"accepted-{gid}",
            "dbh_cm": 15.123456789 + gid,
            "dbh_method": "RECOVERED_STEM",
            "dbh_quality": "HIGH",
        }
        base["trees"].append(row)
        record = SimpleNamespace(**vars(members.global_tree_records[0]))
        record.global_tree_id = gid
        record.persistent_tree_id = row["persistent_tree_id"]
        record.full_resolution_point_count = len(xyz)
        members.global_tree_records += (record,)
        members.final_xyz_local_m = np.vstack((members.final_xyz_local_m, xyz))
        members.final_original_index = np.concatenate(
            (members.final_original_index, np.arange(390 + (gid - 1) * 72, 390 + gid * 72))
        )
        members.final_tree_id = np.concatenate((members.final_tree_id, np.full(72, gid, np.int32)))
        blocks.append(
            {
                "xyz_local_m": xyz,
                "hag_m": points[:, 2],
                "original_index": np.arange(8800 + (gid - 1) * 72, 8800 + gid * 72),
                "global_tree_id": np.full(72, gid, np.int32),
            }
        )
    np.savez(
        br / "tree_segmentation_final.npz",
        xyz_local_m=members.final_xyz_local_m,
        original_index=members.final_original_index,
        tree_id=members.final_tree_id,
    )
    base["binding"] = review.v1._binding(root, result, members)
    arrays = {key: np.concatenate([b[key] for b in blocks]) for key in arrays}
    review.v1._persist(root, result, base, arrays)
    old_rows = copy.deepcopy(base["trees"][1:])
    data = ready(current_project)
    answer = review.confirm_review(root, result, members, request(data, candidate_id="axis-1"))
    assert answer["status"] == "RECOVERED", answer
    current, new_arrays = review.v2._load_base(root, result, members)
    assert current["trees"][1:] == old_rows
    for key in arrays:
        assert np.array_equal(new_arrays[key][new_arrays["global_tree_id"] != 0], arrays[key])
