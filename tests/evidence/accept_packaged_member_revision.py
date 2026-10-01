"""Read-only member-level acceptance against a running desktop project copy.

Example::

    python tests/evidence/accept_packaged_member_revision.py \
      --base-url http://127.0.0.1:52534 --project-root E:/copied-project

The selected project must be a disposable copy. This script only reads API and
project artifact data; it does not start the app, edit trees, or export files.
An optional evidence snapshot is written outside the project when requested.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import urllib.request
import zlib
from pathlib import Path

import numpy as np


def _get(base_url: str, path: str) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"Accept": "application/json", "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("ok") is False:
        raise AssertionError(f"{path}: unsuccessful response: {payload}")
    return payload


def _removed_indices(encoded: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    return np.cumsum(np.frombuffer(raw, dtype="<i8"), dtype=np.int64)


def _sample_indices(count: int, per_tree_budget: int, points: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate((
        np.asarray([np.argmin(points[:, 2]), np.argmax(points[:, 2])]),
        np.linspace(0, count - 1, min(count, per_tree_budget - 2), dtype=int),
    )))


def _digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def check(base_url: str, project_root: Path) -> dict:
    workspace = _get(base_url, "/api/desktop/workspace")
    run_id = str(workspace["batch"]["batch_id"])
    artifact = project_root / "artifacts" / "project-runs" / "batches" / run_id
    edits = json.loads((artifact / "tree_edits.json").read_text(encoding="utf-8"))
    operations = edits["operations"]
    removed = [
        _removed_indices(op["removed_indices_b64"])
        for op in operations if op.get("type") == "REMOVE_POINTS"
    ]
    removed_indices = np.unique(np.concatenate(removed)) if removed else np.empty(0, dtype=np.int64)
    deleted_tree_ids = {int(tree_id) for op in operations if op.get("type") == "DELETE"
                        for tree_id in op.get("tree_ids", [])}

    with np.load(artifact / "tree_segmentation_final.npz", allow_pickle=False) as arrays:
        final_index = arrays["original_index"]
        final_tree = arrays["tree_id"]
        final_xyz = arrays["xyz_local_m"]
    checks: dict[str, dict[str, str]] = {}
    try:
        assert not np.intersect1d(final_index, removed_indices, assume_unique=True).size
        assert deleted_tree_ids.isdisjoint(np.unique(final_tree).tolist())
        checks["final_membership"] = {"status": "PASS"}
    except AssertionError as exc:
        checks["final_membership"] = {"status": "FAIL", "detail": str(exc)}

    review = _get(base_url, "/api/desktop/review?display_budget=30000")
    report = _get(base_url, "/api/current-report")
    visible = np.asarray(review["trees3d"]["visible_by_index"], dtype=bool)
    review_xyz = np.asarray(review["trees3d"]["xyz"])[visible]
    review_tree = np.asarray(review["trees3d"]["tree_id_by_index"])[visible]
    review_original = np.asarray(review["trees3d"]["original_index_by_index"])[visible]
    try:
        assert not np.intersect1d(review_original, removed_indices, assume_unique=True).size
        assert deleted_tree_ids.isdisjoint(np.unique(review_tree).tolist())
        checks["review_visible_membership"] = {"status": "PASS"}
    except AssertionError as exc:
        checks["review_visible_membership"] = {"status": "FAIL", "detail": str(exc)}
    final_ids = set(int(value) for value in np.unique(final_tree))
    review_budget_points = {"30000": len(review_xyz)}
    for budget in (1500, 60000):
        try:
            bounded = _get(base_url, f"/api/desktop/review?display_budget={budget}")
            layer = bounded["trees3d"]
            keep = np.asarray(layer["visible_by_index"], dtype=bool)
            member_index = np.asarray(layer["original_index_by_index"])[keep]
            owner = np.asarray(layer["tree_id_by_index"])[keep]
            assert len(member_index) <= budget
            assert np.isin(member_index, final_index).all()
            assert not np.intersect1d(member_index, removed_indices).size
            assert deleted_tree_ids.isdisjoint(np.unique(owner).tolist())
            review_budget_points[str(budget)] = len(member_index)
            checks[f"review_budget_{budget}"] = {"status": "PASS"}
        except Exception as exc:  # noqa: BLE001 - preserve remaining checks
            checks[f"review_budget_{budget}"] = {
                "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}",
            }
    report_sample_digest = None
    try:
        assert report["status"] == "FINALIZED", report.get("message")
        assert report["gate"]["editor_operation_count"] == len(operations)
        assert report["gate"]["published_editor_operation_count"] == len(operations)
        report_display = report["display"]["trees3d"]
        report_xyz = np.asarray(report_display["xyz"])
        report_tree = np.asarray(report_display["tree_id_by_index"])
        np.testing.assert_array_equal(report_xyz, review_xyz)
        np.testing.assert_array_equal(report_tree, review_tree)
        assert report_display["source_point_count"] == len(review_xyz)
        assert len(report["trees"]) == len(final_ids)
        # The editor review API exposes session tree IDs and display IDs.  The
        # published report/final projection expose persistent IDs.
        assert {str(row["display_tree_id"]) for row in report["trees"]} == {
            str(row["display_tree_id"]) for row in review["trees"]
        }
        final_projection = _get(base_url, "/api/final")
        assert {str(row["persistent_tree_id"]) for row in report["trees"]} == {
            str(row["persistent_tree_id"]) for row in final_projection["trees"]
        }
        report_sample_digest = _digest_arrays(report_tree, report_xyz)
        checks["current_report"] = {"status": "PASS"}
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        checks["current_report"] = {"status": "FAIL", "detail": str(exc)}

    preview_summary = {}
    preview_digests = {}
    for year in (1, 3, 5):
        try:
            preview = _get(base_url, f"/api/future-preview?horizon={year}")
            if not final_ids:
                assert preview["trees"] == []
                assert preview["status"] in {"PASS", "LOCKED"}
                preview_summary[str(year)] = 0
                preview_digests[str(year)] = hashlib.sha256(b"[]").hexdigest()
                checks[f"future_{year}y"] = {"status": "PASS"}
                continue
            assert preview["status"] == "PASS", preview.get("message")
            assert preview["gate"]["final_generation_id"] == report["gate"]["final_generation_id"]
            assert preview["wires"] == report["wires"]
            assert {int(tree["tree_id"]) for tree in preview["trees"]} == final_ids
            budget = preview["display_point_budget"] // len(preview["trees"])
            for tree in preview["trees"]:
                tree_id = int(tree["tree_id"])
                members = final_xyz[final_tree == tree_id]
                keep = _sample_indices(len(members), budget, members)
                np.testing.assert_array_equal(np.asarray(tree["current_xyz"]), members[keep])
                assert len(tree["future_xyz"]) == len(keep)
                assert tree["parent_point_count"] == len(members)
                assert tree["display_point_count"] == len(keep)
            preview_summary[str(year)] = sum(
                tree["display_point_count"] for tree in preview["trees"]
            )
            preview_digests[str(year)] = hashlib.sha256(json.dumps(
                [(tree["tree_id"], tree["current_xyz"]) for tree in preview["trees"]],
                separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            checks[f"future_{year}y"] = {"status": "PASS"}
        except Exception as exc:  # noqa: BLE001 - acceptance must report every horizon
            checks[f"future_{year}y"] = {
                "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}",
            }
    return {
        "run_id": run_id,
        "editor_operations": len(operations),
        "deleted_tree_ids": sorted(deleted_tree_ids),
        "removed_original_indices": len(removed_indices),
        "final_tree_count": len(final_ids),
        "final_member_count": len(final_index),
        "current_report_sample_points": len(review_xyz),
        "review_visible_points_by_budget": review_budget_points,
        "preview_sample_points_by_year": preview_summary,
        "final_membership_sha256": _digest_arrays(final_index, final_tree, final_xyz),
        "current_report_sample_sha256": report_sample_digest,
        "preview_current_sample_sha256_by_year": preview_digests,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--write-snapshot", type=Path,
                        help="Save the read-only result for comparison after reopening the app")
    parser.add_argument("--expect-snapshot", type=Path,
                        help="Require the member evidence to match a previous run")
    args = parser.parse_args()
    result = check(args.base_url, args.project_root)
    if args.expect_snapshot is not None:
        expected = json.loads(args.expect_snapshot.read_text(encoding="utf-8"))
        if result != expected:
            raise AssertionError("current member evidence changed after reopening the app")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.write_snapshot is not None:
        args.write_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.write_snapshot.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if any(check["status"] != "PASS" for check in result["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
