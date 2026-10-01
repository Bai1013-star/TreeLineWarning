"""Desktop adapter for explicit, serialized DBH identity review."""

from __future__ import annotations

from core.stage_b import dbh_raw_recovery as v2
from core.stage_b import dbh_review as review
from core.stage_b.dbh_evidence import read_json, sha256

from desktop.current_report import current_finalized_gate
from desktop.editor_tasks import editor_action
from desktop.tree_display_numbering_extensions import _current_display_identities


def _context(workflow):
    gate = current_finalized_gate(workflow)
    if gate.get("status") != "PASS":
        raise ValueError("DBH_REVIEW_REQUIRES_CURRENT_FINALIZE")
    result = workflow._current_result()
    editor = workflow._ensure_editor()
    if result is None or editor is None:
        raise ValueError("DBH_REVIEW_CURRENT_EDITOR_MISSING")
    return workflow.project_root, result, editor.final_reconciled(), editor


@editor_action("DBH_REVIEW_PREVIEW")
def list_trees(workflow):
    root, result, members, editor = _context(workflow)
    base, _ = v2._load_base(root, result, members)
    identities = _current_display_identities(workflow, editor)
    return {
        "ok": True,
        "human_can_type_dbh": False,
        "reference_input_allowed": False,
        "reference_dbh_cm": None,
        "trees": [
            {
                "persistent_tree_id": row["persistent_tree_id"],
                "global_tree_id": row["global_tree_id"],
                "display_tree_id": identities.get(row["global_tree_id"], {}).get(
                    "display_tree_id", row["persistent_tree_id"]
                ),
                "dbh_cm": None if row.get("dbh_method") == "REFERENCE_ESTIMATE" else row["dbh_cm"],
                "dbh_method": "UNAVAILABLE" if row.get("dbh_method") == "REFERENCE_ESTIMATE" else row["dbh_method"],
                "dbh_quality": "UNAVAILABLE" if row.get("dbh_method") == "REFERENCE_ESTIMATE" else row.get("dbh_quality"),
                "dbh_uncertainty_cm": row.get("dbh_uncertainty_cm"),
                "dbh_fit_status": row.get("dbh_pre_reference_failure_reason", "PER_TREE_STEM_EVIDENCE_REQUIRED")
                if row.get("dbh_method") == "REFERENCE_ESTIMATE" else row.get("dbh_fit_status"),
                "dbh_stale_reference_estimate": row.get("dbh_stale_reference_estimate") or (
                    {"value_cm": row.get("dbh_cm"), "reference": row.get("dbh_reference_estimate"),
                     "status": "RETIRED_NOT_TREE_MEASUREMENT"}
                    if row.get("dbh_method") == "REFERENCE_ESTIMATE" else None
                ),
                "dbh_pre_reference_failure_reason": row.get(
                    "dbh_pre_reference_failure_reason"
                ),
                "dbh_reference_basis": (row.get("dbh_reference_estimate") or {}).get("basis"),
                "dbh_reference_count": (row.get("dbh_reference_estimate") or {}).get(
                    "reference_count"
                ),
                "dbh_reference_mean_cm": (row.get("dbh_reference_estimate") or {}).get(
                    "reference_mean_cm"
                ),
                "dbh_reference_tree_ids": (row.get("dbh_reference_estimate") or {}).get(
                    "reference_tree_persistent_ids"
                ),
            }
            for row in base["trees"]
        ],
    }


@editor_action("DBH_REVIEW_PREVIEW")
def prepare(workflow, request):
    if set(request) != {"persistent_tree_id"}:
        raise ValueError("DBH_REVIEW_INVALID_PREPARE_REQUEST")
    root, result, members, editor = _context(workflow)
    base, _ = v2._load_base(root, result, members)
    review._row(base, request["persistent_tree_id"], allow_accepted=True)
    parent = review.v1._run_root(root, result.analysis_run_id) / "dbh_review_v4"
    data = None
    for path in sorted(
        parent.glob("*/session.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True
    ):
        if (path.parent / "decision.json").exists():
            continue
        try:
            saved = read_json(path)
            if (
                saved.get("schema") == review.VERSION
                and saved.get("review", {}).get("schema") == review.VERSION
                and
                saved["review"]["persistent_tree_id"] == request["persistent_tree_id"]
                and saved["binding"] == base["binding"]
                and saved["baseline_geometry_sha256"] == base["baseline_geometry_sha256"]
                and read_json(path.parent / "proof.json")["session_sha256"] == sha256(path)
                and sha256(path.parent / "raw.npz") == saved["raw_sha256"]
            ):
                data = saved["review"]
                break
        except (ValueError, OSError, KeyError):
            continue
    if data is None:
        data = review.prepare_review(root, result, members, request["persistent_tree_id"])
    identity = _current_display_identities(workflow, editor).get(data["global_tree_id"], {})
    data["display_tree_id"] = identity.get("display_tree_id", data["persistent_tree_id"])
    data["current_crown"]["label"] = data["display_tree_id"]
    return {"ok": True, "review": data}


@editor_action("DBH_REVIEW_CONFIRM")
def confirm(workflow, request):
    root, result, members, _ = _context(workflow)
    outcome = review.confirm_review(root, result, members, request)
    if outcome.get("status") == "RECOVERED" and callable(
        getattr(workflow, "export_prediction", None)
    ):
        workflow.export_prediction()
    if outcome.get("status") == "RECOVERED" and callable(
        getattr(workflow, "export_delivery", None)
    ):
        workflow.export_delivery()
    return outcome


@editor_action("DBH_RECOMPUTE")
def recompute(workflow, request):
    """Re-run current DBH geometry over a verified finalized edit generation."""

    if request:
        raise ValueError("DBH_RECOMPUTE_INVALID_REQUEST")
    root, result, members, _ = _context(workflow)
    from core.stage_b.dbh_identity_recovery import finalize_dbh

    outcome = finalize_dbh(root, result, members, force_recompute=True)
    workbook = workflow.export_prediction()
    available_count = sum(row.get("dbh_cm") is not None for row in outcome["trees"])
    review_required_count = len(outcome["trees"]) - available_count
    if isinstance(getattr(workflow, "_workflow", None), dict):
        workflow._workflow["dbh_delivery_status"] = (
            "READY" if review_required_count == 0 else "PER_TREE_REVIEW_REQUIRED"
        )
        workflow._workflow["dbh_delivery_message"] = (
            None if review_required_count == 0
            else "部分树木缺少本树树干点云证据，请在胸径复核页逐树核对或补测。"
        )
        workflow._save_workflow()
    if callable(getattr(workflow, "export_delivery", None)):
        workflow.export_delivery()
    return {
        "ok": True,
        "status": "RECOMPUTED",
        "tree_count": len(outcome["trees"]),
        "dbh_available_count": available_count,
        "reference_required_count": 0,
        "review_required_count": review_required_count,
        "workbook_updated": workbook.is_file(),
    }
