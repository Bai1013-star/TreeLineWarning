"""Explicit DBH-only recovery for an already finalized project; no full LAS pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from core.stage_b.batch import BatchAnalysisResult, BatchSpanStatus, BatchTelemetry
from core.stage_b.dbh_evidence import confined, read_json
from core.stage_b.dbh_recovery import recover_finalized_dbh


def load_finalized_project(root: Path) -> tuple[BatchAnalysisResult, TreeReconciliationResult]:
    """Detached read-only hydration of exact final arrays; never open source LAS."""
    root = root.resolve()
    workflow = read_json(root / "desktop-workflow.json")
    state = workflow.get("finalize_state", {})
    if state.get("status") != "DONE":
        raise ValueError("DBH_PROJECT_NOT_FINALIZED")
    run_id = state["run_id"]
    br = confined(root, f"artifacts/project-runs/batches/{run_id}")
    batch = read_json(br / "batch.json")
    if batch["analysis_run_id"] != run_id or batch["status"] != "COMPLETE":
        raise ValueError("DBH_FINAL_RUN_MISMATCH")
    records = []
    for value in batch["global_reconciliation"]["trees"]:
        item = dict(value)
        item["merged_span_tree_keys"] = tuple(tuple(k) for k in item["merged_span_tree_keys"])
        records.append(GlobalTreeRecord(**item))
    with np.load(br / "tree_segmentation_final.npz", allow_pickle=False) as archive:
        idx, xyz, tree = archive["original_index"], archive["xyz_local_m"], archive["tree_id"]
    if (
        len(np.unique(idx)) != len(idx)
        or len(idx) != len(xyz)
        or len(idx) != len(tree)
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("DBH_FINAL_MEMBERS_INVALID")
    if set(np.unique(tree)) != {r.global_tree_id for r in records}:
        raise ValueError("DBH_FINAL_TREE_IDS_INVALID")
    for record in records:
        if (
            int(np.count_nonzero(tree == record.global_tree_id))
            != record.full_resolution_point_count
        ):
            raise ValueError("DBH_FINAL_TREE_COUNT_MISMATCH")
    reconciled = TreeReconciliationResult(
        global_tree_records=tuple(records),
        final_original_index=idx,
        final_xyz_local_m=xyz,
        final_tree_id=tree,
        review_tree_ids=(),
        span_tree_to_global={
            tuple(k): r.global_tree_id for r in records for k in r.merged_span_tree_keys
        },
        diagnostics=dict(batch["global_reconciliation"]["diagnostics"]),
    )
    # Existing serialization helper is read-only and preserves exact clearance.
    from desktop.workflow import _load_final_clearance

    result = BatchAnalysisResult(
        analysis_run_id=run_id,
        project_id=batch["project_id"],
        scan_id=batch["scan_id"],
        status=batch["status"],
        span_statuses={k: BatchSpanStatus.FINALIZED for k in batch["span_statuses"]},
        span_results={},
        telemetry=BatchTelemetry(source_point_count=batch["telemetry"]["SOURCE_POINT_COUNT"]),
        risk_policy_snapshot=batch["risk_policy_snapshot"],
        global_reconciled=reconciled,
        final_clearance=_load_final_clearance(batch.get("final_clearance")),
        batch_diagnostics=batch.get("batch_diagnostics", {}),
    )
    return result, reconciled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v3"),
        default="v3",
        help="V2/V3 require a valid finalized DBH base; V1 remains available for older projects",
    )
    parser.add_argument(
        "--write", action="store_true", help="Publish a new DBH-only geometry generation"
    )
    parser.add_argument(
        "--export-name", help="New .xlsx filename within project exports; existing files rejected"
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    result, reconciled = load_finalized_project(root)
    output = None
    if args.export_name:
        if (
            not args.write
            or Path(args.export_name).name != args.export_name
            or not args.export_name.endswith(".xlsx")
        ):
            parser.error("--export-name requires --write and a plain .xlsx filename")
        output = confined(root, "exports/" + args.export_name)
        if output.exists():
            parser.error("export destination already exists; preserve it")
    recovery = recover_finalized_dbh
    if args.version == "v2":
        from core.stage_b.dbh_raw_recovery import recover_finalized_dbh_v2

        recovery = recover_finalized_dbh_v2
    elif args.version == "v3":
        from core.stage_b.dbh_identity_recovery import recover_finalized_dbh_v3

        recovery = recover_finalized_dbh_v3
    payload = recovery(root, result, reconciled, persist=args.write)
    if output is not None:
        from core.stage_b.export import export_batch_prediction

        export_batch_prediction(result, output, final_reconciled=reconciled)
    print(
        json.dumps(
            {
                "published_reference": payload.get("published_reference"),
                "trees": [
                    {
                        key: row.get(key)
                        for key in (
                            "persistent_tree_id",
                            "global_tree_id",
                            "dbh_cm",
                            "dbh_method",
                            "dbh_quality",
                            "dbh_point_count",
                            "dbh_fit_rmse",
                            "raw_local_query_used",
                            "dbh_fit_status",
                        )
                    }
                    for row in payload["trees"]
                ],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
