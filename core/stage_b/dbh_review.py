"""Explicit human stem identity; immutable raw review sessions, unchanged V2 fits.

No API accepts a diameter. A seed is an exact displayed source index, never
an arbitrary XYZ. Human identity replaces automatic association only; geometry,
current editor exclusions, source ownership and publication proof still gate DBH.
"""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from algorithms.manual_span import dbh_raw_stem as fitting
from algorithms.manual_span.dbh_breast_height import fit_raw_breast_height_section
from algorithms.manual_span.metrics import _hag_chunked
from scipy.spatial import cKDTree

from core.stage_b import dbh_identity_recovery as v3
from core.stage_b import dbh_raw_recovery as v2
from core.stage_b import dbh_recovery as v1
from core.stage_b.dbh_evidence import (
    MAX_LOCAL_POINTS,
    array_digest,
    confined,
    local_raw_query,
    raw_ground_reference,
    original_raw_query,
    calibrated_breast_height,
    raw_tree_lowest_reference,
    read_json,
    sha256,
)

VERSION = "DBH_RECOVERY_V5_HUMAN_RAW_1P30_REVIEW"
METHOD = "REVIEW_CONFIRMED_STEM"
DISPLAY_LIMIT = 80_000
_LOCK = threading.RLock()


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _row(base, persistent_id, *, require_unavailable=True):
    rows = [r for r in base["trees"] if r["persistent_tree_id"] == persistent_id]
    if len(rows) != 1:
        raise ValueError("DBH_REVIEW_TREE_NOT_CURRENT")
    if require_unavailable and rows[0]["dbh_cm"] is not None:
        raise ValueError("DBH_REVIEW_ACCEPTED_VALUE_LOCKED")
    return rows[0]


def _measure_at_confirmed_stem_ground(xyz, fit, reference, local_origin, calibrations=()):
    """A human identity choice cannot move the 1.30 m slab to a crown seed.

    Use the original ground plane at the confirmed axis; the first fit locates
    that axis only. Remeasure the actual raw slab, not a diameter at another Z.
    """
    origin = np.asarray(local_origin, dtype=float)
    anchor = np.asarray(fit['breast_height_origin_xyz'], dtype=float)
    direction = np.asarray(fit['unit_direction_xyz'], dtype=float)
    ground = calibrated_breast_height(reference, anchor[:2]+origin[:2], calibrations)
    target = ground['breast_height_z_absolute_m']-origin[2]
    center = anchor[:2] + direction[:2] * ((target-anchor[2])/direction[2])
    hag = xyz[:,2]+origin[2]-ground['ground_z_absolute_m']
    measurement = fit_raw_breast_height_section(xyz, hag, center, fit['dbh_cm']/200)
    corrected = dict(fit)
    corrected.update({k:measurement[k] for k in v1.DBH_FIELDS if k in measurement})
    corrected['dbh_section_measurement'] = measurement
    corrected['breast_height_origin_xyz'] = [*center.tolist(),target]
    selected = np.asarray(measurement['selected_input_indices'], dtype=np.int64)
    return corrected, selected, hag, ground


def _exclusions(root, result):
    br = v1._run_root(root, result.analysis_run_id)
    return v2.current_exclusions(
        root,
        br,
        read_json(br / "batch-reconciliation.json")["trees"],
        read_json(br / "tree_edits.json")["operations"],
    )


def _blocked(indices, arrays, members, gid):
    """Current neighboring members and already published stem evidence reserve IDs."""
    order = np.argsort(members.final_original_index)
    source, owners = members.final_original_index[order], members.final_tree_id[order]
    pos = np.searchsorted(source, indices)
    found = pos < len(source)
    positions = np.flatnonzero(found)
    found[positions] &= source[pos[positions]] == indices[positions]
    conflict = np.zeros(len(indices), dtype=bool)
    conflict[found] = owners[pos[found]] != gid
    reserved = np.sort(arrays["original_index"][arrays["global_tree_id"] != gid])
    return conflict | v2.contains(reserved, indices)


def _sample(length, limit):
    return np.linspace(0, length - 1, min(length, limit), dtype=np.int64)


def _session_root(root, result, review_id):
    if (
        not isinstance(review_id, str)
        or len(review_id) != 32
        or any(c not in "0123456789abcdef" for c in review_id)
    ):
        raise ValueError("DBH_REVIEW_INVALID_SESSION")
    return v1._run_root(root, result.analysis_run_id) / "dbh_review_v4" / review_id


def prepare_review(
    root, result, members, persistent_id, *, query_provider=None, display_tree_id=None
):
    """Prepare artifacts only. No human choice, fit or DBH pointer write occurs."""
    root = Path(root).resolve()
    base, arrays = v2._load_base(root, result, members)
    # All trees can inspect their original evidence. Already accepted values
    # are read-only; only unresolved trees can confirm a new stem identity.
    row = _row(base, persistent_id, require_unavailable=False)
    gid = row["global_tree_id"]
    removed = _exclusions(root, result)
    _, prep, ground, origin = v1._context(root, result)
    origin = np.asarray(origin)
    ground_tree = cKDTree(ground[:, :2])
    crown = members.final_xyz_local_m[members.final_tree_id == gid]
    previous = row.get("v3_queries") or row.get("v2_candidates") or []
    if row.get("dbh_cm") is not None:
        roi_id = row.get("raw_ground_bh_section_estimate", {}).get("roi_id")
        selected = row.get("selected_candidate_id")
        if roi_id is None and selected:
            options = [c for c in row.get("v3_candidates", []) if c["candidate_id"] == selected]
            if options:
                roi_id = options[0]["roi_id"]
        exact = [q for q in previous if q.get("roi_id") == roi_id]
        if exact:
            previous = exact
    seeds = [q["seed"] for q in previous]
    if not seeds:
        hag = _hag_chunked(crown, ground_tree, ground, origin, 250_000)
        seeds, _ = v2.current_seeds(crown, hag)
    if not seeds or len(seeds) > v2.MAX_SEEDS:
        raise ValueError("DBH_REVIEW_NO_BOUNDED_CURRENT_ROI")
    query = query_provider or (lambda cache, source, center, bounds:
                               original_raw_query(root, cache, source, center, bounds))
    source = prep["ground_preparation"]["source_identity"]
    cache = confined(root, prep["vegetation_role_tile_index_reference"])
    blocks, proofs, ground_references = [], [], []
    for seed in seeds:
        center = np.asarray(seed["xy"]) + origin[:2]
        z = float(np.median(ground[ground_tree.query(center, k=min(8, len(ground)))[1], 2]))
        raw, idx, proof = query(cache, source, center, (z - 1, z + 7))
        raw, idx = np.asarray(raw), np.asarray(idx)
        if (
            idx.ndim != 1
            or idx.dtype.kind not in "iu"
            or raw.shape != (len(idx), 3)
            or len(idx) > MAX_LOCAL_POINTS
            or not np.isfinite(raw).all()
            or (idx < 0).any()
            or (idx >= source["source_point_count"]).any()
            or len(np.unique(idx)) != len(idx)
            or v1.strong_source(proof["source_identity"]) != v1.strong_source(source)
        ):
            raise ValueError("DBH_REVIEW_INVALID_RAW_PROVENANCE")
        keep = ~v2.contains(removed, idx)
        deleted_count = int((~keep).sum())
        raw, idx = raw[keep], idx[keep]
        crown_absolute = crown + origin
        ground_reference = raw_tree_lowest_reference(
            crown_absolute, processed_ground_prior_z=z
        )
        if row.get("dbh_cm") is not None:
            # Inspect exactly the ground/target used by the published result.
            ground_reference.update(
                ground_z_absolute_m=row["dbh_ground_z_absolute_m"],
                breast_height_z_absolute_m=row["dbh_measurement_z_absolute_m"],
                raw_minus_processed_ground_m=row["dbh_ground_z_absolute_m"] - z,
            )
        hag = raw[:, 2] - ground_reference["ground_z_absolute_m"]
        raw = raw - origin
        keep = (hag >= -0.20) & (hag <= 5.5)
        blocks.append((raw[keep], idx[keep], hag[keep]))
        proofs.append(
            {
                "seed": seed,
                "query": proof,
                "deleted_count": deleted_count,
                "ground_reference": ground_reference,
            }
        )
        ground_references.append(ground_reference)
    xyz = np.concatenate([b[0] for b in blocks])
    idx = np.concatenate([b[1] for b in blocks])
    hag = np.concatenate([b[2] for b in blocks])
    order = np.argsort(idx, kind="stable")
    duplicate = idx[order][1:] == idx[order][:-1]
    if not np.array_equal(xyz[order][1:][duplicate], xyz[order][:-1][duplicate]):
        raise ValueError("DBH_REVIEW_DUPLICATE_SOURCE_COORDINATE_MISMATCH")
    _, unique = np.unique(idx, return_index=True)
    xyz, idx, hag = xyz[unique], idx[unique], hag[unique]
    if not len(idx) or len(idx) > MAX_LOCAL_POINTS:
        raise ValueError("DBH_REVIEW_EMPTY_OR_OVERSIZED_ROI")
    blocked = _blocked(idx, arrays, members, gid)
    selected = _sample(len(idx), DISPLAY_LIMIT)
    candidates = []
    for c in row.get("v3_candidates", []):
        candidates.append(
            {
                "candidate_id": c["candidate_id"],
                "candidate_score": c["score"],
                "candidate_axis": c["track"].get("refined_axis") or c["axis"],
                "candidate_support": c["identity_support_count"],
                "continuity_status": c["track"]["status"],
                "diagnostics": c,
            }
        )
    review_id = uuid4().hex
    review = {
        "schema": VERSION,
        "review_id": review_id,
        "status": "REVIEW_READY",
        "persistent_tree_id": persistent_id,
        "global_tree_id": gid,
        "display_tree_id": display_tree_id or persistent_id,
        "mode": "CANDIDATE_OR_SEED",
        "current_dbh_cm": row.get("dbh_cm"),
        "current_dbh_method": row.get("dbh_method"),
        # An existing automatic/raw-LAS diameter is a baseline, not a lock.
        # Operators must be able to reopen the original evidence and confirm a
        # better stem identity.  _confirm only persists a replacement after
        # every geometry/ownership quality gate passes, so a failed recheck
        # leaves the existing value untouched.
        "read_only": False,
        "recheck_existing_result": row.get("dbh_cm") is not None,
        "height_reference": "ORIGINAL_RAW_TREE_LOWEST_PLUS_1P30M",
        "ground_references": ground_references,
        "editor_revision": base["binding"]["editor_revision"],
        "candidates": candidates,
        "display": {
            "xyz": xyz[selected].tolist(),
            "original_index": idx[selected].tolist(),
            "hag_m": hag[selected].tolist(),
            "selectable": (~blocked[selected]).tolist(),
        },
        "current_crown": {
            "xyz": crown[_sample(len(crown), 5000)].tolist(),
            "label": display_tree_id or persistent_id,
        },
        "human_can_type_dbh": False,
        "human_selection": None,
        "message": "显示与拟合均来自原始 LAS；胸径截面按该树原始点云最低点以上 1.30 m 计算。页面只确认树干身份，不接受手填胸径。",
    }
    manifest = {
        "schema": VERSION,
        "binding": base["binding"],
        "baseline_geometry_sha256": base["baseline_geometry_sha256"],
        "baseline_geometry_reference": base["baseline_geometry_reference"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review": review,
        "queries": proofs,
        "editor_excluded_index_sha256": array_digest(removed),
    }
    if base["binding"] != v1._binding(root, result, members):
        raise ValueError("DBH_REVIEW_PROJECT_CHANGED")
    folder = _session_root(root, result, review_id)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "raw.npz", xyz=xyz, original_index=idx, hag=hag)
    manifest["raw_sha256"] = sha256(folder / "raw.npz")
    _write(folder / "session.json", manifest)
    # Detached checksum also catches accidental edits of the review's diagnostics.
    _write(folder / "proof.json", {"session_sha256": sha256(folder / "session.json")})
    return review


def fit_review_stem(xyz, hag, *, candidate_axis=None, seed_position=None):
    """Human identity narrows the existing ROI; unchanged V2 axis/V1 circle guards."""
    if (candidate_axis is None) == (seed_position is None):
        raise ValueError("DBH_REVIEW_ONE_IDENTITY_REQUIRED")
    if candidate_axis is not None:
        anchor = np.asarray(candidate_axis["bh_reference_xyz"], dtype=float)
        direction = np.asarray(candidate_axis["direction_xyz"], dtype=float)
        radius = float(candidate_axis.get("radius_m", 0))
        if (
            anchor.shape != (3,)
            or direction.shape != (3,)
            or not np.isfinite(anchor).all()
            or not (
                np.isfinite(direction).all()
                and abs(np.linalg.norm(direction) - 1) < 1e-6
                and 0.025 <= radius <= 0.75
            )
        ):
            raise ValueError("DBH_REVIEW_INVALID_CANDIDATE_AXIS")
        delta = xyz - anchor
        keep = (
            np.linalg.norm(delta - np.outer(delta @ direction, direction), axis=1) <= radius + 0.08
        )
        expected = anchor[:2]
    else:
        seed = xyz[seed_position]
        keep = np.linalg.norm(xyz[:, :2] - seed[:2], axis=1) <= v2.ROI_RADIUS_M
        expected = seed[:2]
    positions = np.flatnonzero(keep)
    fit = fitting.fit_axis_normal_dbh(xyz[keep], hag[keep], expected_xy=expected)
    chosen = positions[np.asarray(fit.pop("selected_input_indices", []), dtype=np.int64)]
    # Human identity cannot bridge a geometry gap at BH. Retain the unchanged
    # circle decisions and require the adjacent normal-plane slices on both
    # sides of the measured true-HAG 1.30 anchor to pass.
    # Actual 1.30 m points, checked below, are mandatory. Older adjacent
    # multisection acceptance must neither replace nor veto that measurement.
    if fit["dbh_cm"] is not None and candidate_axis is not None:
        measured = {
            "bh_reference_xyz": fit["breast_height_origin_xyz"],
            "direction_xyz": fit["unit_direction_xyz"],
        }
        if not fitting._line_candidates_are_equivalent(candidate_axis, measured):
            fit.update(v1.unavailable("DBH_REVIEW_FIT_SWITCHED_STEM"))
    if fit["dbh_cm"] is not None and seed_position is not None:
        delta = xyz[seed_position] - np.asarray(fit["breast_height_origin_xyz"])
        direction = np.asarray(fit["unit_direction_xyz"])
        radial = np.linalg.norm(delta - np.dot(delta, direction) * direction)
        if abs(radial - fit["dbh_cm"] / 200) > 0.08:
            fit.update(v1.unavailable("DBH_REVIEW_SEED_NOT_ON_FITTED_STEM"))
    if fit["dbh_cm"] is not None:
        measurement = fit_raw_breast_height_section(
            xyz, hag, fit["breast_height_origin_xyz"][:2], fit["dbh_cm"] / 200,
        )
        fit["dbh_section_measurement"] = measurement
        if measurement["dbh_cm"] is None:
            fit.update(v1.unavailable(measurement["dbh_fit_status"]))
        else:
            fit.update({key: measurement.get(key) for key in v1.DBH_FIELDS if key in measurement})
            chosen = np.asarray(measurement["selected_input_indices"], dtype=np.int64)
    return fit, chosen


def confirm_review(root, result, members, request):
    """Explicit human action, serialized with editor API; stale sessions fail closed."""
    with _LOCK:
        return _confirm(root, result, members, request)


def _confirm(root, result, members, request):
    if (
        not isinstance(request, dict)
        or set(request) - {"review_id", "identity_confirmed", "candidate_id", "seed_original_index"}
        or request.get("identity_confirmed") is not True
    ):
        raise ValueError("DBH_REVIEW_IDENTITY_ONLY_NO_TYPED_DIAMETER")
    candidate = request.get("candidate_id")
    seed = request.get("seed_original_index")
    if (candidate is None) == (seed is None):
        raise ValueError("DBH_REVIEW_ONE_IDENTITY_REQUIRED")
    root = Path(root).resolve()
    folder = _session_root(root, result, request.get("review_id"))
    manifest = read_json(folder / "session.json")
    if sha256(folder / "session.json") != read_json(folder / "proof.json")["session_sha256"] or (
        sha256(folder / "raw.npz") != manifest["raw_sha256"]
    ):
        raise ValueError("DBH_REVIEW_SESSION_HASH_MISMATCH")
    base, arrays = v2._load_base(root, result, members)
    if base["binding"] != manifest["binding"] or (
        base["baseline_geometry_sha256"] != manifest["baseline_geometry_sha256"]
    ):
        raise ValueError("DBH_REVIEW_STALE_REVISION_REOPEN")
    if (folder / "decision.json").exists():
        raise ValueError("DBH_REVIEW_ALREADY_SUBMITTED_REOPEN")
    review = manifest["review"]
    row = _row(base, review["persistent_tree_id"])
    gid = row["global_tree_id"]
    with np.load(folder / "raw.npz", allow_pickle=False) as archive:
        xyz, idx, hag = archive["xyz"], archive["original_index"], archive["hag"]
    removed = _exclusions(root, result)
    if v2.contains(removed, idx).any():
        raise ValueError("DBH_REVIEW_DELETED_SOURCE_EVIDENCE")
    axis, seed_pos = None, None
    if candidate is not None:
        options = [c for c in review["candidates"] if c["candidate_id"] == candidate]
        if len(options) != 1:
            raise ValueError("DBH_REVIEW_UNKNOWN_CANDIDATE")
        axis = copy.deepcopy(options[0]["candidate_axis"])
        if "radius_m" not in axis:
            axis["radius_m"] = options[0]["diagnostics"]["axis"]["radius_m"]
        roi_id = options[0]["diagnostics"]["roi_id"]
        reference = next((q["ground_reference"] for q in manifest["queries"]
                          if q["seed"] == row["v3_queries"][roi_id]["seed"]), None)
        if reference is None:
            raise ValueError("DBH_REVIEW_ORIGINAL_GROUND_REFERENCE_MISSING")
        _, _, _, local_origin = v1._context(root, result)
        hag = xyz[:, 2] + local_origin[2] - reference["ground_z_absolute_m"]
    else:
        visible = review["display"]
        if type(seed) is not int or seed not in visible["original_index"]:
            raise ValueError("DBH_REVIEW_SEED_NOT_EXACT_DISPLAYED_RAW_POINT")
        display_pos = visible["original_index"].index(seed)
        if not visible["selectable"][display_pos]:
            raise ValueError("DBH_REVIEW_NEIGHBOR_STEM_SEED")
        seed_pos = int(np.searchsorted(idx, seed))
        if seed_pos >= len(idx) or idx[seed_pos] != seed:
            raise ValueError("DBH_REVIEW_SEED_SOURCE_MISMATCH")
        # Multiple search ROIs may have different terrain levels. The selected
        # original point determines one ground reference for this stem fit.
        ground_local = xyz[seed_pos, 2] - hag[seed_pos]
        hag = xyz[:, 2] - ground_local
        _, _, _, local_origin = v1._context(root, result)
        possible = [q['ground_reference'] for q in manifest['queries']
                    if abs(q['ground_reference']['ground_z_absolute_m']
                           -float(local_origin[2])-ground_local) < 1e-6]
        if not possible:
            raise ValueError('DBH_REVIEW_ORIGINAL_GROUND_REFERENCE_MISSING')
        reference = min(possible, key=lambda g: np.linalg.norm(
            np.asarray(g['raw_ground_plane_xy_origin_m'])-xyz[seed_pos,:2]-np.asarray(local_origin[:2])))
    fit, selected = fit_review_stem(xyz, hag, candidate_axis=axis, seed_position=seed_pos)
    stem_ground = None
    if fit['dbh_cm'] is not None:
        fit, selected, hag, stem_ground = _measure_at_confirmed_stem_ground(
            xyz,fit,reference,local_origin,base['trees'])
    if fit["dbh_cm"] is not None:
        origin = np.asarray(fit["breast_height_origin_xyz"])
        direction = np.asarray(fit["unit_direction_xyz"])
        own = members.final_xyz_local_m[members.final_tree_id == gid]
        if cKDTree(own[:, :2]).query(origin[:2])[0] > v2.ROI_RADIUS_M:
            fit.update(v1.unavailable("DBH_REVIEW_OUTSIDE_CURRENT_CROWN"))
        if _blocked(idx[selected], arrays, members, gid).any():
            fit.update(v1.unavailable("DBH_REVIEW_NEIGHBOR_EVIDENCE_CONFLICT"))
        measured = {"bh_reference_xyz": origin.tolist(), "direction_xyz": direction.tolist()}
        for old in base["trees"]:
            if int(old["global_tree_id"]) == int(gid):
                continue
            if old["dbh_cm"] is None or old["dbh_method"] == "ALLOMETRIC_ESTIMATE":
                continue
            fixed = v3._fixed_claim(old, arrays)
            if fitting._line_candidates_are_equivalent(
                measured,
                {
                    "bh_reference_xyz": fixed["axis_origin"],
                    "direction_xyz": fixed["axis_direction"],
                },
            ):
                fit.update(v1.unavailable("DBH_REVIEW_STEM_ALREADY_OWNED"))
    if base["binding"] != v1._binding(root, result, members):
        raise ValueError("DBH_REVIEW_PROJECT_CHANGED_DURING_FIT")
    current = read_json(v1._run_root(root, result.analysis_run_id) / "dbh_final.json")
    if current["geometry_sha256"] != manifest["baseline_geometry_sha256"]:
        raise ValueError("DBH_REVIEW_BASE_CHANGED_DURING_FIT")
    evidence = {
        "schema": VERSION,
        "review_id": review["review_id"],
        "persistent_tree_id": row["persistent_tree_id"],
        "editor_revision": review["editor_revision"],
        "identity_confirmed": True,
        "selection": request,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_original_index": idx[selected].tolist(),
        "selected_original_index_sha256": array_digest(idx[selected]),
        "fit": fit,
    }
    _write(folder / "decision.json", evidence)
    if fit["dbh_cm"] is not None:
        row.update({key: fit.get(key) for key in v1.DBH_FIELDS})
        row.update(
            dbh_method=METHOD,
            dbh_source=VERSION,
            dbh_point_count=len(selected),
            dbh_evidence_point_count=len(selected),
            dbh_review=evidence,
            stem_association_status="HUMAN_CONFIRMED_CURRENT_STEM_IDENTITY",
            dbh_section_measurement=fit["dbh_section_measurement"],
        )
        ground_values = xyz[selected, 2] - hag[selected]
        if float(np.ptp(ground_values)) > 1e-6:
            raise ValueError("DBH_REVIEW_INCONSISTENT_ORIGINAL_GROUND_REOPEN")
        _, _, _, local_origin = v1._context(root, result)
        ground_z = float(np.median(ground_values) + local_origin[2])
        row.update(dbh_ground_z_absolute_m=ground_z,
                   dbh_measurement_z_absolute_m=ground_z + 1.30,
                   dbh_height_reference="ORIGINAL_RAW_TREE_LOWEST_PLUS_1P30M",
                   dbh_ground_reference_method=stem_ground['ground_reference_method'],
                   dbh_site_slope_deg=stem_ground['site_slope_deg'],
                   dbh_common_height_applied=stem_ground['common_height_applied'],
                   dbh_measurement_height_policy=stem_ground['measurement_height_policy'],
                   dbh_calibration_tree_id=stem_ground.get('calibration_tree_id'))
        row["dbh_review_decision_reference"] = (
            (folder / "decision.json").relative_to(root).as_posix()
        )
        row["dbh_review_decision_sha256"] = sha256(folder / "decision.json")
        keep = arrays["global_tree_id"] != gid
        fresh = {
            "xyz_local_m": xyz[selected],
            "hag_m": hag[selected],
            "original_index": idx[selected],
            "global_tree_id": np.full(len(selected), gid, np.int32),
        }
        combined = {key: np.concatenate((arrays[key][keep], fresh[key])) for key in arrays}
        if len(np.unique(combined["original_index"])) != len(combined["original_index"]):
            raise ValueError("DBH_REVIEW_DUPLICATE_OWNERSHIP")
        base["recovery_version"] = VERSION
        proofs = base.setdefault("additional_input_hashes", [])
        for name in ("session.json", "raw.npz", "decision.json"):
            proofs.append(
                {
                    "reference": (folder / name).relative_to(root).as_posix(),
                    "sha256": sha256(folder / name),
                }
            )
        v1._persist(root, result, base, combined)
    return {
        "ok": True,
        "status": "RECOVERED" if fit["dbh_cm"] is not None else "UNAVAILABLE",
        "dbh_cm": fit["dbh_cm"],
        "dbh_method": METHOD if fit["dbh_cm"] is not None else "UNAVAILABLE",
        "dbh_quality": fit["dbh_quality"],
        "dbh_fit_status": fit["dbh_fit_status"],
        "message": "人工确认树干身份，胸径由点云几何计算"
        if fit["dbh_cm"] is not None
        else "该树干点云不足，无法可靠计算胸径",
        "review_id": review["review_id"],
    }
