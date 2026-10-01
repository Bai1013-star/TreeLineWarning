"""Exercise DELETE -> finalize -> UNDO -> finalize on a packaged test copy.

This script mutates the served project.  It refuses projects outside the
repository's ``acceptance_evidence`` directory and never starts an application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.parse
import zipfile
from pathlib import Path
from uuid import uuid4

from accept_packaged_member_revision import check as check_members
from accept_packaged_review import Client, workbook_rows


def exact_array_hashes(path: Path) -> dict[str, str]:
    """Hash decompressed NPY members, ignoring ZIP metadata and compression."""

    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".npy"))
        required = {"original_index.npy", "tree_id.npy", "xyz_local_m.npy"}
        if not required.issubset(names):
            raise AssertionError(f"missing final member arrays: {required - set(names)}")
        hashes = {}
        for name in names:
            digest = hashlib.sha256()
            with archive.open(name) as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            hashes[name] = digest.hexdigest()
        return hashes


def editor_revision(client: Client) -> dict:
    return client.json("/api/desktop/review?display_budget=1500")


def wait_mutation(client: Client, request_id: str, deadline: float) -> dict:
    path = "/api/editor/mutation?request_id=" + urllib.parse.quote(request_id)
    while time.monotonic() < deadline:
        status = client.json(path)
        if status.get("request_id") != request_id:
            raise AssertionError(f"mutation request_id mismatch: {status}")
        if status.get("state") == "COMMITTED":
            return status.get("result") or {}
        if status.get("state") == "FAILED":
            raise AssertionError(f"mutation failed: {status.get('error')}")
        time.sleep(0.5)
    raise TimeoutError(f"mutation {request_id} timed out")


def mutate(client: Client, path: str, deadline: float) -> dict:
    request_id = "packaged-delete-undo-" + uuid4().hex
    started = client.post(path, {"async": True, "request_id": request_id})
    if started.get("request_id") != request_id:
        raise AssertionError(f"mutation was not acknowledged: {started}")
    return wait_mutation(client, request_id, deadline)


def current_counts(client: Client) -> tuple[dict, dict, dict]:
    workspace = client.json("/api/desktop/workspace")
    review = editor_revision(client)
    final = client.json("/api/final")
    return workspace, review, final


def assert_count(client: Client, count: int, excluded_persistent_id: str | None = None) -> dict:
    workspace, review, final = current_counts(client)
    counts = {
        "workspace": workspace["batch"]["final_tree_count"],
        "review": len(review.get("trees", [])),
        "final_projection": len(final.get("trees", [])),
    }
    if set(counts.values()) != {count}:
        raise AssertionError(f"tree count mismatch: {counts}")
    review_ids = {str(tree["display_tree_id"]) for tree in review["trees"]}
    final_ids = {str(tree["display_tree_id"]) for tree in final["trees"]}
    if review_ids != final_ids or len(review_ids) != count:
        raise AssertionError("review and final projection IDs differ")
    persistent_ids = {str(tree["persistent_tree_id"]) for tree in final["trees"]}
    if excluded_persistent_id is not None and excluded_persistent_id in persistent_ids:
        raise AssertionError("deleted persistent tree remains in the final projection")
    return {"counts": counts, "ui_state": workspace["ui_state"],
            "review_ids": sorted(review_ids), "final_ids": sorted(final_ids),
            "persistent_ids": sorted(persistent_ids)}


def wait_finalized(client: Client, deadline: float) -> dict:
    while time.monotonic() < deadline:
        state = client.json("/state")
        if state.get("step") == "FINALIZE_FAILED":
            raise AssertionError(f"finalization failed: {state.get('error_details')}")
        if state.get("step") == "DONE":
            workspace = client.json("/api/desktop/workspace")
            report = client.json("/api/current-report")
            if workspace.get("ui_state") == "FINALIZED" and report.get("status") == "FINALIZED":
                return {"step": state["step"], "ui_state": workspace["ui_state"],
                        "report_status": report["status"]}
        time.sleep(1)
    raise TimeoutError("finalization timed out")


def collect_finalized(
    client: Client, project_root: Path, output: Path, phase: str,
    expected_count: int, excluded_persistent_id: str | None = None,
) -> dict:
    summary = assert_count(client, expected_count, excluded_persistent_id)
    if summary["ui_state"] != "FINALIZED":
        raise AssertionError(f"{phase}: workspace is not finalized")
    report = client.json("/api/current-report")
    if report.get("status") != "FINALIZED" or len(report.get("trees", [])) != expected_count:
        raise AssertionError(f"{phase}: current report is not final/current")
    report_by_display = {str(row["display_tree_id"]): row for row in report["trees"]}
    if excluded_persistent_id in {
        str(row["persistent_tree_id"]) for row in report["trees"]
    }:
        raise AssertionError(f"{phase}: deleted tree remains in current report")
    member = check_members(client.base_url, project_root)
    failed = {name: check for name, check in member["checks"].items()
              if check["status"] != "PASS"}
    if failed or member["final_tree_count"] != expected_count:
        raise AssertionError(f"{phase}: member/current/preview checks failed: {failed}")

    downloads = {}
    workbook_values = None
    for name, route in (("prediction", "/download_prediction"), ("excel", "/download_excel")):
        payload = client.get(route)
        (output / f"{phase}_{name}.xlsx").write_bytes(payload)
        rows = workbook_rows(payload)
        by_display = {str(row[0]): row for row in rows[1:]}
        if len(by_display) != expected_count or set(by_display) != set(report_by_display):
            raise AssertionError(f"{phase}: {route} tree IDs differ from current report")
        for display_id, row in by_display.items():
            delivered = row[8]
            report_row = report_by_display[display_id]
            measured = report_row.get("dbh_cm")
            if report_row.get("dbh_method") == "REFERENCE_ESTIMATE":
                measured = None  # retired estimate must remain blank in delivery
            if measured is None:
                if delivered is not None:
                    raise AssertionError(f"{phase}: unmeasured {display_id} has diameter")
            elif not (
                isinstance(delivered, (int, float))
                and math.isclose(float(delivered), float(measured), abs_tol=1e-6)
            ):
                raise AssertionError(f"{phase}: {display_id} Excel/report DBH differs")
        if workbook_values is not None and rows != workbook_values:
            raise AssertionError(f"{phase}: prediction and Excel differ cell by cell")
        workbook_values = rows
        downloads[name] = {
            "tree_count": len(by_display),
            "blank_diameter_count": sum(row[8] is None for row in rows[1:]),
            "columns": len(rows[0]),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {"summary": summary, "member": member, "downloads": downloads,
            "workbook_rows": workbook_values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    evidence_root = Path(__file__).resolve().parents[2] / "acceptance_evidence"
    original_root = Path("E:/1/3").resolve()
    if os.path.normcase(str(project_root)) == os.path.normcase(str(original_root)):
        raise ValueError("Refusing to mutate the original E:/1/3 project")
    if not project_root.is_relative_to(evidence_root.resolve()):
        raise ValueError("Refusing to mutate a project outside acceptance_evidence")
    client = Client(args.base_url)
    served_root = Path(client.json("/api/desktop/project")["project"]["root"]).resolve()
    if os.path.normcase(str(served_root)) != os.path.normcase(str(project_root)):
        raise ValueError(f"served project differs from --project-root: {served_root}")
    output = args.output_dir.resolve()
    if output.is_relative_to(project_root):
        raise ValueError("evidence output must be outside the project copy")
    output.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.timeout_seconds
    baseline_count = len(editor_revision(client).get("trees", []))
    if baseline_count < 2:
        raise AssertionError("acceptance copy needs at least two finalized trees")
    before = collect_finalized(client, project_root, output, "before", baseline_count)
    batch_id = client.json("/api/desktop/workspace")["batch"]["batch_id"]
    artifact = project_root / "artifacts" / "project-runs" / "batches" / str(batch_id)
    journal = artifact / "tree_edits.json"
    members = artifact / "tree_segmentation_final.npz"
    original_operations = json.loads(journal.read_text(encoding="utf-8"))["operations"]
    original_arrays = exact_array_hashes(members)
    review = editor_revision(client)
    target = min(review["trees"], key=lambda row: int(row["point_count"]))
    tree_id = int(target["tree_id"])
    target_display_id = str(target["display_tree_id"])
    before_report = client.json("/api/current-report")
    target_persistent_id = next(
        str(row["persistent_tree_id"]) for row in before_report["trees"]
        if str(row["display_tree_id"]) == target_display_id
    )
    selected = client.post("/api/select", {"tree_ids": [tree_id]})
    if selected.get("selected_ids") != [tree_id]:
        raise AssertionError(f"tree selection failed: {selected.get('selected_ids')}")
    deleted = mutate(client, "/api/delete", deadline)
    if len(deleted.get("trees", [])) != baseline_count - 1:
        raise AssertionError("DELETE did not reduce the tree count by one")
    after_delete = assert_count(client, baseline_count - 1, target_persistent_id)
    if after_delete["ui_state"] != "TREE_REVIEW_READY":
        raise AssertionError(f"parent workspace stayed finalized after DELETE: {after_delete}")
    dirty_report = client.json("/api/current-report")
    if dirty_report.get("status") != "LOCKED" or dirty_report.get("trees"):
        raise AssertionError("dirty report exposed old finalized trees")
    if dirty_report.get("gate", {}).get("status") != "LOCKED":
        raise AssertionError("dirty report gate did not lock")
    deleted_operations = json.loads(journal.read_text(encoding="utf-8"))["operations"]
    if len(deleted_operations) != len(original_operations) + 1:
        raise AssertionError("DELETE was not appended to the edit journal")

    started = client.post("/api/confirm_trees", {})
    if started.get("step") not in {"FINALIZING", "DONE"}:
        raise AssertionError(f"delete finalization did not start: {started}")
    deleted_finalization = wait_finalized(client, deadline)
    deleted_final = collect_finalized(
        client, project_root, output, "deleted", baseline_count - 1, target_persistent_id
    )

    current_operations = json.loads(journal.read_text(encoding="utf-8"))["operations"]
    if current_operations[:len(original_operations)] != original_operations:
        raise AssertionError("delete finalization changed pre-existing edit operations")
    undo_count = len(current_operations) - len(original_operations)
    if undo_count < 1:
        raise AssertionError("delete operation disappeared before UNDO")
    for _ in range(undo_count):
        undone = mutate(client, "/api/undo", deadline)
    if len(undone.get("trees", [])) != baseline_count:
        raise AssertionError("UNDO did not restore the baseline tree count")
    after_undo = assert_count(client, baseline_count)
    if after_undo["review_ids"] != before["summary"]["review_ids"]:
        raise AssertionError("UNDO did not restore the original displayed tree IDs")
    restored_operations = json.loads(journal.read_text(encoding="utf-8"))["operations"]
    if restored_operations != original_operations:
        raise AssertionError("UNDO did not restore the original edit operations")
    started = client.post("/api/confirm_trees", {})
    if started.get("step") not in {"FINALIZING", "DONE"}:
        raise AssertionError(f"finalization did not start: {started}")
    finalization = wait_finalized(client, deadline)
    after = collect_finalized(client, project_root, output, "after", baseline_count)
    restored_arrays = exact_array_hashes(members)
    if restored_arrays != original_arrays:
        raise AssertionError("final NPZ member arrays differ after DELETE and UNDO")
    if after["summary"]["persistent_ids"] != before["summary"]["persistent_ids"]:
        raise AssertionError("persistent tree membership differs after UNDO")
    if after["member"]["final_membership_sha256"] != before["member"]["final_membership_sha256"]:
        raise AssertionError("final member content differs after UNDO")
    evidence = {
        "status": "PASS", "base_url": args.base_url, "project_root": str(project_root),
        "target": {"tree_id": tree_id, "display_tree_id": target_display_id,
                   "persistent_tree_id": target_persistent_id,
                   "point_count": target["point_count"]},
        "baseline_tree_count": baseline_count, "before": before,
        "after_delete": after_delete,
        "dirty_report": {"status": dirty_report["status"],
                         "gate_status": dirty_report["gate"]["status"],
                         "reasons": dirty_report["gate"].get("reasons", [])},
        "deleted_finalization": deleted_finalization, "deleted_final": deleted_final,
        "after_undo": after_undo, "undo_count": undo_count,
        "finalization": finalization, "after": after,
        "journal_operations_restored": True, "final_member_array_sha256": restored_arrays,
    }
    (output / "delete_undo_acceptance.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
