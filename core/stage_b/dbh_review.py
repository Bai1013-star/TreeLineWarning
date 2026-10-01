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
from algorithms.manual_span.partial_arc_identity import (
    elevated_local_base_probe,
    recover_ground_anchored_bh,
    recover_local_bh_series,
)
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
    read_json,
    sha256,
)
from core.stage_b.dbh_wide_search import current_crown_wide_seeds

VERSION = "DBH_RECOVERY_V4_HUMAN_REVIEW_V2"
METHOD = "REVIEW_CONFIRMED_STEM"
DISPLAY_LIMIT = 80_000
_LOCK = threading.RLock()


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _row(base, persistent_id, *, allow_accepted=False):
    rows = [r for r in base["trees"] if r["persistent_tree_id"] == persistent_id]
    if len(rows) != 1:
        raise ValueError("DBH_REVIEW_TREE_NOT_CURRENT")
    if (not allow_accepted and rows[0]["dbh_cm"] is not None
            and rows[0].get("dbh_method") != "REFERENCE_ESTIMATE"):
        raise ValueError("DBH_REVIEW_ACCEPTED_VALUE_LOCKED")
    return rows[0]


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
    row = _row(base, persistent_id, allow_accepted=True)
    gid = row["global_tree_id"]
    removed = _exclusions(root, result)
    _, prep, ground, origin = v1._context(root, result)
    origin = np.asarray(origin)
    ground_tree = cKDTree(ground[:, :2])
    crown = members.final_xyz_local_m[members.final_tree_id == gid]
    previous = row.get("v3_queries") or row.get("v2_candidates") or []
    seeds = [q["seed"] for q in previous]
    if not seeds:
        hag = _hag_chunked(crown, ground_tree, ground, origin, 250_000)
        seeds, _ = v2.current_seeds(crown, hag)
    query = query_provider or local_raw_query
    source = prep["ground_preparation"]["source_identity"]
    cache = confined(root, prep["vegetation_role_tile_index_reference"])
    wide_proof = None
    if query_provider is None and (
        row.get("dbh_cm") is None or row.get("dbh_method") == "REFERENCE_ESTIMATE"
    ):
        wide, wide_proof = current_crown_wide_seeds(
            cache, source, crown, ground, ground_tree, origin,
            excluded_indices=removed, existing_seeds=seeds,
            limit=max(0, v3.MAX_V3_SEEDS - len(seeds)),
        )
        seeds.extend(wide)
    if not seeds or len(seeds) > v3.MAX_V3_SEEDS:
        raise ValueError("DBH_REVIEW_NO_BOUNDED_CURRENT_ROI")
    blocks, proofs = [], []
    candidate_radii = [float(c.get("candidate_axis", {}).get("radius_m") or
                             c.get("axis", {}).get("radius_m") or 0)
                       for c in row.get("v3_candidates", [])]
    review_radius = min(1.20, max([0.60, *(r + 0.43 for r in candidate_radii)]))
    for seed in seeds:
        center = np.asarray(seed["xy"]) + origin[:2]
        z = float(np.median(ground[ground_tree.query(center, k=min(8, len(ground)))[1], 2]))
        if query_provider is None:
            raw, idx, proof = query(
                cache, source, center, (z - 1, z + 7), radius_m=review_radius
            )
        else:
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
        raw, idx = raw[keep] - origin, idx[keep]
        hag = _hag_chunked(raw, ground_tree, ground, origin, 250_000)
        keep = (hag >= -0.25) & (hag <= 5.5)
        blocks.append((raw[keep], idx[keep], hag[keep]))
        proofs.append({"seed": seed, "query": proof, "deleted_count": deleted_count})
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
        "editor_revision": base["binding"]["editor_revision"],
        "candidates": candidates,
        "wide_raw_stem_search": wide_proof,
        "published_measurement": {
            "dbh_cm": None if row.get("dbh_method") == "REFERENCE_ESTIMATE" else row.get("dbh_cm"),
            "dbh_method": "UNAVAILABLE" if row.get("dbh_method") == "REFERENCE_ESTIMATE" else row.get("dbh_method"),
            "dbh_quality": row.get("dbh_quality"),
            "dbh_fit_status": row.get("dbh_pre_reference_failure_reason", row.get("dbh_fit_status"))
            if row.get("dbh_method") == "REFERENCE_ESTIMATE" else row.get("dbh_fit_status"),
            "dbh_uncertainty_cm": row.get("dbh_uncertainty_cm"),
            "evidence_xyz": arrays["xyz_local_m"][arrays["global_tree_id"] == gid][
                _sample(int(np.count_nonzero(arrays["global_tree_id"] == gid)), 5000)
            ].tolist(),
            "evidence_hag_m": arrays["hag_m"][arrays["global_tree_id"] == gid][
                _sample(int(np.count_nonzero(arrays["global_tree_id"] == gid)), 5000)
            ].tolist(),
        },
        "display": {
            "xyz": xyz[selected].tolist(),
            "original_index": idx[selected].tolist(),
            "hag_m": hag[selected].tolist(),
            "selectable": (~blocked[selected]).tolist(),
            "ground_selectable": ((~blocked[selected]) & (hag[selected] >= -0.25)
                                  & (hag[selected] <= 2.0)).tolist(),
        },
        "current_crown": {
            "xyz": crown[_sample(len(crown), 5000)].tolist(),
            "label": display_tree_id or persistent_id,
        },
        "human_can_type_dbh": False,
        "confirmation_allowed": row.get("dbh_cm") is None or row.get("dbh_method") == "REFERENCE_ESTIMATE",
        "human_selection": None,
        "message": "仅确认树干身份；胸径由点云计算。原始点抽样仅用于显示，拟合使用完整局部证据。",
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


def fit_review_stem(xyz, hag, *, candidate_axis=None, seed_position=None,
                    local_ground=False):
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
    local_series = False
    if local_ground:
        recovered, proof = recover_ground_anchored_bh(
            xyz[keep], hag[keep], np.arange(len(positions), dtype=np.int64)
        )
        if recovered is None:
            fit.update(v1.unavailable(proof["status"]))
        else:
            local_series = True
            fit = recovered["fit"]
            chosen = positions[np.asarray(fit.pop("selected_input_indices"), dtype=np.int64)]
            fit["breast_height_origin_xyz"] = recovered["axis"]["bh_reference_xyz"]
            fit["unit_direction_xyz"] = recovered["axis"]["direction_xyz"]
            fit["local_ground_window_proof"] = proof
    if fit["dbh_cm"] is None and candidate_axis is not None and not local_ground:
        recovered, _ = recover_local_bh_series(
            xyz, hag, np.arange(len(xyz), dtype=np.int64), candidate_axis
        )
        if recovered is not None:
            local_series = True
            fit = recovered["fit"]
            chosen = np.asarray(fit.pop("selected_input_indices"), dtype=np.int64)
            fit["breast_height_origin_xyz"] = recovered["axis"]["bh_reference_xyz"]
            fit["unit_direction_xyz"] = recovered["axis"]["direction_xyz"]
    # Human identity cannot bridge a geometry gap at BH. Retain the unchanged
    # circle decisions and require the adjacent normal-plane slices on both
    # sides of the measured true-HAG 1.30 anchor to pass.
    if fit["dbh_cm"] is not None and not local_series:
        slices = {s["index"]: s for s in fit.get("slices", [])}
        if not all(slices.get(i, {}).get("accepted") is True for i in (2, 3)):
            fit.update(v1.unavailable("DBH_REVIEW_BREAST_HEIGHT_CONTINUITY_GAP"))
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
    if fit["dbh_cm"] is not None and not local_ground:
        axis = {"bh_reference_xyz": fit["breast_height_origin_xyz"],
                "direction_xyz": fit["unit_direction_xyz"],
                "radius_m": fit["dbh_cm"] / 200}
        probe = elevated_local_base_probe(xyz, hag, axis)
        fit["local_base_probe"] = probe
        if probe["status"] in {"LOCAL_BASE_HEIGHT_AMBIGUOUS", "LOCAL_BASE_HEIGHT_UNVERIFIED"}:
            fit.update(v1.unavailable(probe["status"]))
    return fit, chosen


def confirm_review(root, result, members, request):
    """Explicit human action, serialized with editor API; stale sessions fail closed."""
    with _LOCK:
        return _confirm(root, result, members, request)


def _confirm(root, result, members, request):
    if (
        not isinstance(request, dict)
        or set(request) - {"review_id", "identity_confirmed", "candidate_id",
                           "seed_original_index", "ground_original_index"}
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
    if manifest.get("schema") != VERSION or manifest.get("review", {}).get("schema") != VERSION:
        raise ValueError("DBH_REVIEW_STALE_SESSION_SCHEMA")
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
    source = manifest["queries"][0]["query"]["source_identity"]
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
    ground_source = request.get("ground_original_index")
    fit_hag = hag
    ground_anchor = None
    if ground_source is not None:
        visible = review["display"]
        if type(ground_source) is not int or ground_source not in visible["original_index"]:
            raise ValueError("DBH_REVIEW_GROUND_NOT_EXACT_DISPLAYED_RAW_POINT")
        display_pos = visible["original_index"].index(ground_source)
        if not visible["ground_selectable"][display_pos]:
            raise ValueError("DBH_REVIEW_GROUND_POINT_NOT_SELECTABLE")
        ground_pos = int(np.searchsorted(idx, ground_source))
        if ground_pos >= len(idx) or idx[ground_pos] != ground_source:
            raise ValueError("DBH_REVIEW_GROUND_SOURCE_MISMATCH")
        if axis is not None:
            anchor_xy = np.asarray(axis["bh_reference_xyz"][:2])
            direction = np.asarray(axis["direction_xyz"])
            stem_xy = anchor_xy + (
                (xyz[:, 2, None] - axis["bh_reference_xyz"][2])
                * direction[:2] / direction[2]
            )
            stem_radius = float(axis["radius_m"])
        else:
            stem_xy = np.broadcast_to(xyz[seed_pos, :2], (len(xyz), 2))
            stem_radius = None
        radial = np.linalg.norm(xyz[:, :2] - stem_xy, axis=1)
        ground_distance = float(radial[ground_pos])
        ground_inner = stem_radius + max(.08, .20 * stem_radius) if stem_radius else .18
        ground_outer = stem_radius + .43 if stem_radius else .90
        if not ground_inner <= ground_distance <= ground_outer:
            raise ValueError("DBH_REVIEW_GROUND_NOT_ADJACENT_TO_STEM")
        local_level = float(hag[ground_pos])
        ground_z = float(xyz[ground_pos, 2])
        nearby_surface = (np.linalg.norm(xyz[:, :2] - xyz[ground_pos, :2], axis=1) <= 0.15
                          ) & (np.abs(xyz[:, 2] - ground_z) <= 0.10)
        bark = (np.abs(radial - stem_radius) <= max(.05, .25 * stem_radius)
                if stem_radius else radial <= .22)
        contact = bark & (xyz[:, 2] >= ground_z) & (xyz[:, 2] <= ground_z + 0.30)
        if np.count_nonzero(nearby_surface) < 20 or np.count_nonzero(contact) < 50:
            raise ValueError("DBH_REVIEW_GROUND_SURFACE_OR_STEM_CONTACT_UNPROVEN")
        # The selected root point fixes a physical elevation. Subtracting its
        # coarse HAG from every point leaves the coarse DTM's XY slope in the
        # 1.3 m slices and shifts the fitted diameter on sloping ground.
        fit_hag = xyz[:, 2] - ground_z
        ground_anchor = {
            "original_index": ground_source,
            "xyz_local_m": xyz[ground_pos].tolist(),
            "coarse_hag_m": local_level,
            "ground_z_local_m": ground_z,
            "height_reference": "RAW_Z_MINUS_SELECTED_GROUND_Z",
            "source_sha256": source["sha256"],
            "distance_to_stem_axis_m": ground_distance,
            "surface_support_count": int(np.count_nonzero(nearby_surface)),
            "stem_contact_count": int(np.count_nonzero(contact)),
        }
    fit, selected = fit_review_stem(
        xyz, fit_hag, candidate_axis=axis, seed_position=seed_pos,
        local_ground=ground_anchor is not None,
    )
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
            if old["dbh_cm"] is None or old["dbh_method"] not in {
                "DIRECT_STEM", "RECOVERED_STEM", "REVIEW_CONFIRMED_STEM"
            }:
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
        "ground_anchor": ground_anchor,
    }
    _write(folder / "decision.json", evidence)
    if fit["dbh_cm"] is not None:
        if row.get("dbh_method") == "REFERENCE_ESTIMATE":
            row["dbh_stale_reference_estimate"] = {
                "value_cm": row["dbh_cm"],
                "source": row.get("dbh_source"),
                "reference": row.get("dbh_reference_estimate"),
                "status": "SUPERSEDED_BY_TREE_MEASUREMENT",
            }
        row.pop("dbh_reference_estimate", None)
        row.pop("dbh_pre_reference_failure_reason", None)
        row.update({key: fit.get(key) for key in v1.DBH_FIELDS})
        row.update(
            dbh_method=METHOD,
            dbh_source=VERSION,
            dbh_point_count=len(selected),
            dbh_evidence_point_count=len(selected),
            dbh_review=evidence,
            stem_association_status="HUMAN_CONFIRMED_CURRENT_STEM_IDENTITY",
        )
        if ground_anchor is not None:
            row["dbh_ground_anchor"] = ground_anchor
        row["dbh_review_decision_reference"] = (
            (folder / "decision.json").relative_to(root).as_posix()
        )
        row["dbh_review_decision_sha256"] = sha256(folder / "decision.json")
        keep = arrays["global_tree_id"] != gid
        fresh = {
            "xyz_local_m": xyz[selected],
            "hag_m": fit_hag[selected],
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
        "ground_anchor_hag_m": ground_anchor["coarse_hag_m"] if ground_anchor else None,
    }
