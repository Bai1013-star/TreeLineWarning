"""Desktop adapter for explicit, serialized DBH identity review."""

from __future__ import annotations

from core.stage_b import dbh_raw_recovery as v2
from core.stage_b import dbh_identity_recovery as v3
from core.stage_b import dbh_review as review
from core.stage_b.dbh_evidence import read_json, sha256

from desktop.editor_tasks import editor_action
from desktop.tree_display_numbering_extensions import _current_display_identities


def _context(workflow):
    # DBH review is part of the human-review stage, not a post-finalization
    # report.  Requiring a finalized publication here created a circular UI:
    # the operator had to finish review before the 1.30 m raw-LAS evidence
    # could be opened.  The current result + editor are the authoritative
    # provisional membership; confirm/finalize still performs the normal
    # publication checks afterwards.
    result = workflow._current_result()
    editor = workflow._ensure_editor()
    if result is None or editor is None:
        raise ValueError("DBH_REVIEW_REQUIRES_COMPLETED_ANALYSIS")
    return workflow.project_root, result, editor.final_reconciled(), editor


def _load_review_base(root, result, members):
    """Prepare raw-LAS evidence before finalization without publishing a report.

    Removing the old finalized-only gate was insufficient: a new analysis has
    no dbh_final pointer yet. Use the normal geometry/evidence computation for
    the current editor revision, retaining all source and identity guards.
    Corrupt evidence is not silently rebuilt or treated as a valid diameter.
    """
    pointer = review.v1._run_root(root, result.analysis_run_id) / 'dbh_final.json'
    if not pointer.is_file():
        v3.finalize_dbh(root, result, members)
    try:
        base, arrays = v2._load_base(root, result, members)
    except ValueError as exc:
        if str(exc) != 'DBH_V2_STALE_BASELINE':
            raise
        v3.finalize_dbh(root, result, members)
        return v2._load_base(root, result, members)
    # A stop/crash between the baseline and raw-ground recovery can leave a
    # valid V1/V2 pointer. Valid provenance alone does not mean the requested
    # original-LAS 1.30 m calculation has finished. This explicit review
    # action resumes that calculation; read-only reports still never do it.
    if isinstance(base, dict) and base.get('recovery_version') != v3.VERSION:
        v3.finalize_dbh(root, result, members)
        return v2._load_base(root, result, members)
    return base, arrays


@editor_action("DBH_REVIEW_PREVIEW")
def list_trees(workflow):
    root, result, members, editor = _context(workflow)
    base, _ = _load_review_base(root, result, members)
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
                "dbh_fit_status": row["dbh_fit_status"],
                "raw_original_las_review_available": True,
                "raw_original_las_review_read_only": False,
                "raw_original_las_recheck_existing": row["dbh_cm"] is not None,
                "dbh_ground_z_absolute_m": row.get("dbh_ground_z_absolute_m"),
                "dbh_measurement_z_absolute_m": row.get("dbh_measurement_z_absolute_m"),
                "dbh_height_reference": row.get("dbh_height_reference"),
            }
            for row in base["trees"]
        ],
    }


@editor_action("DBH_REVIEW_PREVIEW")
def prepare(workflow, request):
    if set(request) != {"persistent_tree_id"}:
        raise ValueError("DBH_REVIEW_INVALID_PREPARE_REQUEST")
    root, result, members, editor = _context(workflow)
    base, _ = _load_review_base(root, result, members)
    review._row(base, request["persistent_tree_id"], require_unavailable=False)
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
                and saved.get("schema") == review.VERSION
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
    # Pending sessions created by an older build recorded an existing result
    # as read-only.  Do not make users discard project evidence just to regain
    # the controls: confirmation is already fail-safe and only replaces the
    # baseline when the new point-cloud fit passes all quality gates.
    data["read_only"] = False
    data["recheck_existing_result"] = data.get("current_dbh_cm") is not None
    identity = _current_display_identities(workflow, editor).get(data["global_tree_id"], {})
    data["display_tree_id"] = identity.get("display_tree_id", data["persistent_tree_id"])
    data["current_crown"]["label"] = data["display_tree_id"]
    return {"ok": True, "review": data}


@editor_action("DBH_REVIEW_CONFIRM")
def confirm(workflow, request):
    root, result, members, _ = _context(workflow)
    return review.confirm_review(root, result, members, request)
