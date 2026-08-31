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
        "trees": [
            {
                "persistent_tree_id": row["persistent_tree_id"],
                "global_tree_id": row["global_tree_id"],
                "display_tree_id": identities.get(row["global_tree_id"], {}).get(
                    "display_tree_id", row["persistent_tree_id"]
                ),
                "dbh_cm": row["dbh_cm"],
                "dbh_method": row["dbh_method"],
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
    review._row(base, request["persistent_tree_id"])
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
    return review.confirm_review(root, result, members, request)
