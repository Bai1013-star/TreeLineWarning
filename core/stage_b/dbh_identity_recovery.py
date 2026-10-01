"""V3: resolve current crown/stem identity jointly before separate DBH fitting."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from algorithms.manual_span import stem_identity as geometry
from algorithms.manual_span.metrics import _hag_chunked
from algorithms.manual_span.partial_arc_identity import (
    elevated_local_base_probe,
    recover_bh_tube_identity,
    recover_partial_arc_identity,
    recover_upper_partial_arc_identity,
    recover_upper_supported_identity,
)
from algorithms.manual_span.stem_ownership import resolve_stem_ownership
from scipy.spatial import cKDTree

from core.stage_b import dbh_raw_recovery as v2
from core.stage_b import dbh_recovery as v1
from core.stage_b.dbh_evidence import (
    MAX_LOCAL_POINTS, array_digest, confined, local_raw_query, read_json, sha256,
)
from core.stage_b.dbh_reference_estimation import apply_reference_estimates
from core.stage_b.dbh_wide_search import current_crown_wide_seeds

VERSION = "DBH_RECOVERY_V3_PER_TREE_V3"
PRIOR_AUTO_SOURCES = {"DBH_RECOVERY_V3_PER_TREE_V2"}
MAX_V3_SEEDS = 8


def _fixed_claim(row, arrays):
    mask = arrays["global_tree_id"] == row["global_tree_id"]
    xyz, hag = arrays["xyz_local_m"][mask], arrays["hag_m"][mask]
    if not len(xyz):
        raise ValueError("ACCEPTED_STEM_EVIDENCE_MISSING")
    near = np.abs(hag - 1.30) <= 0.15
    center = np.median(xyz[near] if near.any() else xyz, axis=0)
    # Reservation geometry comes from existing accepted evidence, never an
    # invented vertical direction. This does not refit or change its diameter.
    centers, heights = [], []
    for low in np.arange(0.85, 1.75, 0.15):
        band = (hag >= low) & (hag < low + 0.15)
        if band.any():
            centers.append(np.median(xyz[band], axis=0))
            heights.append(float(np.median(hag[band])))
    line = geometry._fit_hag_line(np.asarray(centers), np.asarray(heights), np.arange(len(centers)))
    if line is None:
        raise ValueError("ACCEPTED_STEM_RESERVATION_AXIS_UNPROVEN")
    center, unit, _ = line
    return {
        "tree_id": row["global_tree_id"],
        "candidate_id": f"fixed-{row['global_tree_id']}",
        "score": 1.0,
        "absolute_pass": True,
        "axis_origin": center.tolist(),
        "axis_direction": unit.tolist(),
        "original_index": arrays["original_index"][mask],
    }


def _v2_axes(candidate):
    detail = candidate.get("fit", {}).get("axis", {})
    result = geometry.distinct_axes(detail.get("plausible_candidates", []))
    for axis in result:
        axis["radius_m"] = geometry.axis_radius(axis, detail["preliminary_sections"])
        axis["proposal_source"] = "V2_EXACT_CURRENT_AXIS_PROPOSAL"
    return result


def recover_finalized_dbh_v3(
    root, result, reconciled, *, persist=True, progress=None, query_provider=None, member_hag=None,
    recompute_auto=False, reference_dbh_cm=None,
):
    root = Path(root).resolve()
    base, old_arrays = v2._load_base(root, result, reconciled)
    binding = base["binding"]
    br = v1._run_root(root, result.analysis_run_id)
    operations = read_json(br / "tree_edits.json")["operations"]
    original = read_json(br / "batch-reconciliation.json")["trees"]
    original_hash = (
        sha256(br / "tree_segmentation_original.npz")
        if any(op["type"] == "DELETE" for op in operations)
        else None
    )
    removed = v2.current_exclusions(root, br, original, operations)
    if v2.contains(removed, reconciled.final_original_index).any():
        raise ValueError("CURRENT_MEMBERS_CONTAIN_DELETED_POINTS")
    _, prep, ground, origin = v1._context(root, result)
    origin = np.asarray(origin)
    ground_tree = cKDTree(ground[:, :2])
    if member_hag is None:
        member_hag = _hag_chunked(
            reconciled.final_xyz_local_m, ground_tree, ground, origin, 250_000
        )
    member_hag = np.asarray(member_hag)
    if (
        member_hag.shape != reconciled.final_original_index.shape
        or not np.isfinite(member_hag).all()
    ):
        raise ValueError("INVALID_CURRENT_HAG")
    contexts = {}
    for record in reconciled.global_tree_records:
        mask = reconciled.final_tree_id == record.global_tree_id
        xyz, hag = reconciled.final_xyz_local_m[mask], member_hag[mask]
        contexts[int(record.global_tree_id)] = {
            "xyz": xyz,
            "hag": hag,
            "anchor": geometry.crown_anchor(xyz, hag),
        }
    order = np.argsort(reconciled.final_original_index)
    sorted_members, sorted_owners = (
        reconciled.final_original_index[order],
        reconciled.final_tree_id[order],
    )
    source = prep["ground_preparation"]["source_identity"]
    cache = confined(root, prep["vegetation_role_tile_index_reference"])
    query = query_provider or local_raw_query
    legacy_base_proofs = {}
    for accepted in base["trees"]:
        if (accepted["dbh_cm"] is None
                or accepted.get("dbh_method") not in {"DIRECT_STEM", "RECOVERED_STEM"}
                or accepted.get("dbh_source") in {VERSION, *PRIOR_AUTO_SOURCES}):
            continue
        # V1/V2 could publish a crisp circle at the wrong global HAG when the
        # actual root stands on a local terrace. Audit their existing evidence
        # against *all-source* raw points before reserving or preserving it.
        fixed_axis = _fixed_claim(accepted, old_arrays)
        center = np.asarray(fixed_axis["axis_origin"][:2]) + origin[:2]
        coarse_z = float(np.median(ground[
            ground_tree.query(center, k=min(8, len(ground)))[1], 2
        ]))
        try:
            audit_radius = min(1.20, max(0.60, accepted["dbh_cm"] / 200 + 0.43))
            if query_provider is None:
                raw, idx, proof = query(
                    cache, source, center, (coarse_z - 1, coarse_z + 7),
                    radius_m=audit_radius,
                )
            else:
                raw, idx, proof = query(cache, source, center, (coarse_z - 1, coarse_z + 7))
            raw, idx = np.asarray(raw), np.asarray(idx)
            if (raw.shape != (len(idx), 3) or idx.ndim != 1
                    or len(idx) > MAX_LOCAL_POINTS or (idx < 0).any()
                    or (idx >= source["source_point_count"]).any()
                    or v1.strong_source(proof["source_identity"]) != v1.strong_source(source)):
                raise ValueError("LEGACY_BASE_RAW_SOURCE_PROVENANCE_INVALID")
            keep = ~v2.contains(removed, idx)
            raw = raw[keep] - origin
            hag = _hag_chunked(raw, ground_tree, ground, origin, 250_000)
            legacy_base_proofs[accepted["global_tree_id"]] = elevated_local_base_probe(
                raw, hag, {"bh_reference_xyz": fixed_axis["axis_origin"],
                           "direction_xyz": fixed_axis["axis_direction"],
                           "radius_m": accepted["dbh_cm"] / 200}
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            legacy_base_proofs[accepted["global_tree_id"]] = {
                "status": "LOCAL_BASE_AUDIT_UNAVAILABLE", "reason": str(exc),
            }
    suspect_base_ids = {gid for gid, proof in legacy_base_proofs.items()
                        if proof["status"] != "NO_ELEVATED_BASE_EVIDENCE"}

    def replace_prior_auto(row):
        if row["global_tree_id"] in suspect_base_ids:
            return True
        if row.get("dbh_method") == "REFERENCE_ESTIMATE":
            return True
        return bool(
            recompute_auto
            and row["dbh_cm"] is not None
            and row.get("dbh_method") == "RECOVERED_STEM"
            and row.get("dbh_source") in {VERSION, geometry.VERSION, *PRIOR_AUTO_SOURCES}
        )

    fixed = [
        _fixed_claim(row, old_arrays)
        for row in base["trees"]
        if row["dbh_cm"] is not None
        and not replace_prior_auto(row)
        and row["dbh_method"] in {"DIRECT_STEM", "RECOVERED_STEM", "REVIEW_CONFIRMED_STEM"}
    ]
    rows, jobs, claims = [], {}, []
    for old in sorted(base["trees"], key=lambda row: row["global_tree_id"]):
        row = copy.deepcopy(old)
        gid = row["global_tree_id"]
        rows.append(row)
        if row["dbh_cm"] is not None and not replace_prior_auto(row):
            continue
        if progress:
            progress(f"V3_CURRENT_CROWN:{gid}")
        if old.get("dbh_method") == "REFERENCE_ESTIMATE":
            row["dbh_stale_reference_estimate"] = {
                "value_cm": old.get("dbh_cm"),
                "source": old.get("dbh_source"),
                "reference": old.get("dbh_reference_estimate"),
                "status": "RETIRED_NOT_TREE_MEASUREMENT",
            }
        if gid in suspect_base_ids:
            row["dbh_stale_prior_geometry"] = {
                "value_cm": old.get("dbh_cm"), "method": old.get("dbh_method"),
                "source": old.get("dbh_source"),
                "status": "RETIRED_LOCAL_BASE_NOT_VERIFIED",
            }
        row.update(v1.unavailable("NO_CROWN_TO_STEM_IDENTITY"))
        row.update(
            dbh_source=VERSION,
            v2_failure_reason=old.get("v2_failure_reason", old["dbh_fit_status"]),
            v3_candidates=[],
            v3_queries=[],
            current_editor_revision=binding["editor_revision"],
        )
        if gid in legacy_base_proofs:
            row["legacy_base_probe"] = legacy_base_proofs[gid]
        row.pop("v3_final_fit", None)
        row.pop("selected_candidate_id", None)
        row.pop("dbh_reference_estimate", None)
        row.pop("dbh_pre_reference_failure_reason", None)
        saved_candidates = old.get("v2_candidates", [])
        if not saved_candidates:
            seeds, _ = v2.current_seeds(
                contexts[gid]["xyz"], contexts[gid]["hag"], old.get("stem_axis_xy_local_m")
            )
            saved_candidates = [{"seed": seed} for seed in seeds]
        else:
            # V2 projects saved before the reserved stem-hint ROI existed have
            # three crown-mode seeds. Replay the validated lineage hint as an
            # additional search location only when current members support it.
            seeds, hint_proof = v2.current_seeds(
                contexts[gid]["xyz"], contexts[gid]["hag"], old.get("stem_axis_xy_local_m")
            )
            hint = next(
                (seed for seed in seeds if seed["source"] == "CURRENT_SUPPORTED_HINT"), None
            )
            if hint is not None and all(
                np.linalg.norm(np.asarray(candidate["seed"]["xy"]) - hint["xy"])
                > v2.ROI_RADIUS_M
                for candidate in saved_candidates
            ):
                saved_candidates = [*saved_candidates, {"seed": hint}]
                row["validated_stem_hint_replayed"] = hint_proof
        if query_provider is None and not any(
            c.get("absolute_pass") for c in old.get("v3_candidates", [])
        ):
            wide, wide_proof = current_crown_wide_seeds(
                cache, source, contexts[gid]["xyz"], ground, ground_tree, origin,
                excluded_indices=removed,
                existing_seeds=[candidate["seed"] for candidate in saved_candidates],
                limit=max(0, MAX_V3_SEEDS - len(saved_candidates)),
            )
            saved_candidates.extend({"seed": seed} for seed in wide)
            row["wide_raw_stem_search"] = wide_proof
        if len(saved_candidates) > MAX_V3_SEEDS:
            raise ValueError("V3_ROI_BUDGET_EXCEEDED")
        has_axes = any(
            c.get("fit", {}).get("axis", {}).get("plausible_candidates") for c in saved_candidates
        )
        seen_raw, seen_deleted = [], []
        for roi_id, previous in enumerate(saved_candidates):
            axes = _v2_axes(previous)
            is_hint = previous["seed"].get("source") == "CURRENT_SUPPORTED_HINT"
            is_wide = previous["seed"].get("source") == "ALL_SOURCE_VERTICAL_OCCUPANCY"
            discover = not has_axes or is_hint or is_wide
            if has_axes and not axes and not discover:
                continue
            center = np.asarray(previous["seed"]["xy"]) + origin[:2]
            ground_z = float(
                np.median(ground[ground_tree.query(center, k=min(8, len(ground)))[1], 2])
            )
            query_record = {"roi_id": roi_id, "seed": previous["seed"]}
            row["v3_queries"].append(query_record)
            try:
                raw, idx, proof = query(cache, source, center, (ground_z - 1, ground_z + 7))
                raw, idx = np.asarray(raw), np.asarray(idx)
                if (
                    raw.shape != (len(idx), 3)
                    or idx.ndim != 1
                    or idx.dtype.kind not in "iu"
                    or not np.isfinite(raw).all()
                    or (idx < 0).any()
                    or (idx >= source["source_point_count"]).any()
                    or len(np.unique(idx)) != len(idx)
                    or v1.strong_source(proof["source_identity"]) != v1.strong_source(source)
                ):
                    raise ValueError("RAW_SOURCE_PROVENANCE_INVALID")
                seen_raw.append(idx)
                deleted = v2.contains(removed, idx)
                seen_deleted.append(idx[deleted])
                deleted_bh_count = 0
                if deleted.any():
                    deleted_hag = _hag_chunked(
                        raw[deleted] - origin, ground_tree, ground, origin, 250_000
                    )
                    deleted_bh_count = int(
                        np.count_nonzero(np.abs(deleted_hag - 1.30) <= 0.15)
                    )
                query_record.update(
                    proof=proof,
                    editor_excluded_point_count=int(deleted.sum()),
                    editor_excluded_bh_point_count=deleted_bh_count,
                )
                raw, idx = raw[~deleted] - origin, idx[~deleted]
                hag = _hag_chunked(raw, ground_tree, ground, origin, 250_000)
                low = (hag >= -0.25) & (hag <= 5.5)
                raw, idx, hag = raw[low], idx[low], hag[low]
                if discover:
                    axes, proposal_proof = geometry.upper_axes(raw, hag)
                    query_record["upper_proposals"] = proposal_proof
                partial_arc = None
                if not axes and discover:
                    partial_arc, partial_proof = recover_partial_arc_identity(raw, hag, idx)
                    query_record["partial_arc_probe"] = partial_proof
                    if partial_arc is None:
                        partial_arc, upper_proof = recover_upper_supported_identity(raw, hag, idx)
                        query_record["upper_circular_probe"] = upper_proof
                    if partial_arc is None:
                        partial_arc, upper_arc_proof = recover_upper_partial_arc_identity(
                            raw, hag, idx
                        )
                        query_record["upper_partial_arc_probe"] = upper_arc_proof
                    if partial_arc is None:
                        partial_arc, tube_proof = recover_bh_tube_identity(
                            raw, hag, idx, expected_xy=previous["seed"]["xy"]
                        )
                        query_record["bh_tube_probe"] = tube_proof
                    if partial_arc is not None:
                        axes = [partial_arc["axis"]]
                for axis_number, axis in enumerate(axes):
                    identifier = f"g{gid}-r{roi_id}-a{axis_number}"
                    tracked = (
                        copy.deepcopy(partial_arc["track"])
                        if partial_arc is not None
                        else geometry.track_downward(raw, hag, idx, axis)
                    )
                    support = tracked.pop("original_index")
                    effective = tracked["refined_axis"] or axis
                    axis_origin = np.asarray(effective["bh_reference_xyz"])
                    direction = np.asarray(effective["direction_xyz"])
                    local_base = elevated_local_base_probe(
                        raw, hag, {**effective,
                                   "radius_m": effective.get("radius_m", axis["radius_m"])}
                    )
                    bridge = v2.raw_bridge(raw, hag, axis_origin, direction, axis["radius_m"] * 200)
                    scores = {}
                    for other in sorted(contexts):
                        context = contexts[other]
                        connection = v2.axis_connection(
                            context["xyz"],
                            context["hag"],
                            axis_origin,
                            direction,
                            axis["radius_m"] * 200,
                        )
                        if tracked["status"] == "PASS" and connection["status"] != "PASS":
                            bridged = v2.raw_bridged_axis_connection(
                                context["xyz"], context["hag"], raw, hag,
                                axis_origin, direction, axis["radius_m"] * 200,
                            )
                            connection["raw_bridged_upper_crown"] = bridged
                            if bridged["status"] == "PASS":
                                connection = {
                                    **connection,
                                    "status": "PASS",
                                    "mode": "RAW_BRIDGED_TO_CURRENT_UPPER_CROWN",
                                }
                        scores[other] = geometry.score_crown(
                            tracked, effective, context["anchor"], connection, bridge
                        )
                        if local_base["status"] in {
                            "LOCAL_BASE_HEIGHT_AMBIGUOUS", "LOCAL_BASE_HEIGHT_UNVERIFIED",
                        }:
                            scores[other]["absolute_pass"] = False
                            scores[other]["local_base_gate"] = local_base
                    item = {
                        "candidate_id": identifier,
                        "roi_id": roi_id,
                        "axis": axis,
                        "track": tracked,
                        "crown_scores": {str(key): value for key, value in scores.items()},
                        "identity_support_count": len(support),
                        "identity_support_original_index_sha256": array_digest(support),
                        "editor_excluded_point_count": int(deleted.sum()),
                        "local_base_probe": local_base,
                        **scores[gid],
                    }
                    row["v3_candidates"].append(item)
                    jobs[identifier] = {
                        "raw": raw,
                        "idx": idx,
                        "hag": hag,
                        "axis": effective,
                        "item": item,
                        "source_tree": gid,
                        "editor_excluded_bh_point_count": deleted_bh_count,
                        "prefit": partial_arc["fit"] if partial_arc is not None else None,
                    }
                    for other, score in scores.items():
                        # Always report the source crown; only plausible foreign claims
                        # need to enter the graph. Their full metrics are retained above.
                        if other != gid and not (score["absolute_pass"] and score["score"] >= 0.75):
                            continue
                        foreign_id = identifier if other == gid else f"{identifier}-claim-{other}"
                        claims.append(
                            {
                                "tree_id": other,
                                "candidate_id": foreign_id,
                                "score": score["score"],
                                "absolute_pass": score["absolute_pass"],
                                "axis_origin": axis_origin.tolist(),
                                "axis_direction": direction.tolist(),
                                "original_index": support,
                            }
                        )
                query_record["status"] = "PASS"
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                query_record.update(status="UNAVAILABLE", reason=str(exc))
        row["v3_raw_roi_point_count"] = len(np.unique(np.concatenate(seen_raw))) if seen_raw else 0
        row["v3_editor_excluded_point_count"] = (
            len(np.unique(np.concatenate(seen_deleted))) if seen_deleted else 0
        )
    # Equivalent tracks from overlapping ROIs are one physical option, not a
    # synthetic runner-up. Keep every score in the audit and retain foreign
    # claims so deduplication cannot conceal cross-crown ambiguity.
    duplicates = set()
    for gid in sorted(contexts):
        representatives = []
        for cid in sorted(
            (c for c in jobs if jobs[c]["source_tree"] == gid),
            key=lambda c: (-jobs[c]["item"]["absolute_pass"], -jobs[c]["item"]["score"], c),
        ):
            prior = next(
                (
                    p
                    for p in representatives
                    if geometry._line_candidates_are_equivalent(jobs[cid]["axis"], jobs[p]["axis"])
                ),
                None,
            )
            if prior is None:
                representatives.append(cid)
            else:
                duplicates.add(cid)
                jobs[cid]["item"]["duplicate_track_of"] = prior
    claims = [c for c in claims if c["candidate_id"] not in duplicates]
    # Overlapping ROIs can report one physical stem twice for the same target:
    # once from its own crown and once as a foreign suggestion.  That is one
    # option, not two close-scoring alternatives.  Keep cross-target claims so
    # a stem shared by two *different* trees is still rejected by ownership.
    unique_claims = []
    for claim in sorted(claims, key=lambda c: (
        c["tree_id"],
        "-claim-" in c["candidate_id"],
        -c["score"], c["candidate_id"],
    )):
        same = next((prior for prior in unique_claims
                     if prior["tree_id"] == claim["tree_id"]
                     and geometry._line_candidates_are_equivalent(
                         {"bh_reference_xyz": prior["axis_origin"],
                          "direction_xyz": prior["axis_direction"]},
                         {"bh_reference_xyz": claim["axis_origin"],
                          "direction_xyz": claim["axis_direction"]},
                     )), None)
        if same is None:
            unique_claims.append(claim)
    claims = unique_claims
    resolution = resolve_stem_ownership(claims, fixed)
    # Global resolver output must be identical under a reverse presentation order.
    if resolution != resolve_stem_ownership(list(reversed(claims)), list(reversed(fixed))):
        raise ValueError("GLOBAL_OWNERSHIP_NONDETERMINISTIC")
    blocks = []
    for row in rows:
        gid = row["global_tree_id"]
        if row["dbh_cm"] is not None:
            mask = old_arrays["global_tree_id"] == gid
            blocks.append({key: value[mask] for key, value in old_arrays.items()})
            continue
        for candidate in row["v3_candidates"]:
            candidate["decision"] = resolution["decisions"].get(candidate["candidate_id"])
        chosen_id = resolution["selected"].get(gid)
        row["dbh_fit_status"] = resolution["tree_status"].get(gid, "NO_CROWN_TO_STEM_IDENTITY")
        if row["v3_candidates"] and max(
            row["v3_candidates"], key=lambda c: c["score"]
        ).get("local_base_probe", {}).get("status") in {
            "LOCAL_BASE_HEIGHT_AMBIGUOUS", "LOCAL_BASE_HEIGHT_UNVERIFIED",
        }:
            row["dbh_fit_status"] = max(
                row["v3_candidates"], key=lambda c: c["score"]
            )["local_base_probe"]["status"]
        elif len([c for c in row["v3_candidates"]
                  if c.get("crown_scores", {}).get(str(gid), {}).get(
                      "connection", {}).get("status") == "PASS"
                  and c.get("crown_scores", {}).get(str(gid), {}).get(
                      "raw_bridge", {}).get("status") == "PASS"]) >= 2:
            row["dbh_fit_status"] = "MULTIPLE_STEM_AXES_REVIEW_REQUIRED"
        row["dbh_evidence_point_count"] = row["dbh_band_point_count"] = 0
        if chosen_id is None:
            continue
        if chosen_id not in jobs:
            row["dbh_fit_status"] = "FOREIGN_STEM_CLAIM_NOT_ASSIGNABLE"
            continue
        job = jobs[chosen_id]
        if job["source_tree"] != gid:
            # A foreign claim exists solely to veto duplicated ownership.
            row["dbh_fit_status"] = "FOREIGN_STEM_CLAIM_NOT_ASSIGNABLE"
            continue
        fitted = (
            copy.deepcopy(job["prefit"])
            if job["prefit"] is not None
            else geometry.fit_identified_stem(job["raw"], job["hag"], job["axis"])
        )
        if (
            fitted["dbh_cm"] is not None
            and fitted.get("dbh_quality") == "ESTIMATE"
            and job["editor_excluded_bh_point_count"] >= 12
        ):
            fitted.update(
                dbh_cm=None,
                dbh_method="UNAVAILABLE",
                dbh_quality="UNAVAILABLE",
                dbh_fit_status="EDITOR_REMOVED_BH_EVIDENCE",
            )
        positions = np.asarray(fitted.pop("selected_input_indices"), dtype=np.int64)
        row["v3_final_fit"] = fitted
        row["selected_candidate_id"] = chosen_id
        row["dbh_fit_status"] = fitted["dbh_fit_status"]
        if row.get("v2_failure_reason") == "MULTIPLE_CURRENT_STEMS_AMBIGUOUS":
            row["dbh_fit_status"] = "MULTIPLE_STEM_AXES_REVIEW_REQUIRED"
            continue
        if fitted["dbh_cm"] is None:
            continue
        selected_idx = job["idx"][positions]
        if v2.contains(removed, selected_idx).any() or v2.conflicting_current_owner(
            selected_idx, sorted_members, sorted_owners, gid
        ):
            row["dbh_fit_status"] = "SELECTED_SOURCE_OWNERSHIP_CONFLICT"
            continue
        row.update({key: fitted.get(key) for key in v1.DBH_FIELDS if key != "dbh_source"})
        row.update(
            dbh_source=VERSION,
            dbh_point_count=len(positions),
            dbh_evidence_point_count=len(positions),
            stem_association_status="CURRENT_CROWN_TRACKED",
            stem_axis_xy_local_m=job["axis"]["bh_reference_xyz"][:2],
            dbh_band_point_count=int(
                np.count_nonzero((job["hag"][positions] >= 0.85) & (job["hag"][positions] <= 1.75))
            ),
        )
        blocks.append(
            {
                "xyz_local_m": job["raw"][positions],
                "hag_m": job["hag"][positions],
                "original_index": selected_idx,
                "global_tree_id": np.full(len(positions), gid, np.int32),
            }
        )
    arrays = {
        key: np.concatenate([block[key] for block in blocks]) if blocks else old_arrays[key][:0]
        for key in old_arrays
    }
    if len(np.unique(arrays["original_index"])) != len(arrays["original_index"]):
        raise ValueError("V3_DUPLICATE_FINAL_STEM_OWNERSHIP")
    if v2.contains(removed, arrays["original_index"]).any():
        raise ValueError("V3_DELETED_EVIDENCE_REINTRODUCED")
    reliable = sum(
        r["dbh_cm"] is not None and r["dbh_method"] in {"DIRECT_STEM", "RECOVERED_STEM"}
        for r in rows
    )
    allometry = {"eligible": reliable >= 10, "reliable_count": reliable, "model_used": False}
    # Model evaluation is an explicit post-recovery decision; no automatic estimate.
    if reliable >= 10:
        from algorithms.manual_span.dbh_recovery import estimate_local_dbh

        allometry["evaluation"] = estimate_local_dbh(rows, [r for r in rows if r["dbh_cm"] is None])
    reference_estimation = apply_reference_estimates(rows, reference_dbh_cm=reference_dbh_cm)
    if binding != v1._binding(root, result, reconciled):
        raise ValueError("V3_PROJECT_CHANGED_DURING_RECOVERY")
    if original_hash is not None and sha256(br / "tree_segmentation_original.npz") != original_hash:
        raise ValueError("V3_ORIGINAL_MEMBERS_CHANGED_DURING_RECOVERY")
    payload = {
        "schema": base["schema"],
        "binding": binding,
        "recovery_version": VERSION,
        "baseline_geometry_reference": base["baseline_geometry_reference"],
        "baseline_geometry_sha256": base["baseline_geometry_sha256"],
        "span_proofs": base.get("span_proofs", {}),
        "trees": rows,
        "global_ownership": resolution,
        "allometry": allometry,
        "reference_estimation": reference_estimation,
        "editor_excluded_index_count": len(removed),
        "editor_excluded_index_sha256": array_digest(removed),
        "current_member_hag_sha256": array_digest(member_hag),
        "additional_input_hashes": base.get("additional_input_hashes", [])
        + (
            [
                {
                    "reference": (br / "tree_segmentation_original.npz")
                    .relative_to(root)
                    .as_posix(),
                    "sha256": original_hash,
                }
            ]
            if original_hash is not None
            else []
        ),
    }
    if persist:
        path = v1._persist(root, result, payload, arrays)
        payload["published_reference"] = path.relative_to(root).as_posix()
    return payload


def finalize_dbh(root, result, reconciled, *, force_recompute=False, reference_dbh_cm=None):
    recompute = force_recompute or reference_dbh_cm is not None
    existing = v1.load_finalized_dbh(Path(root), result, reconciled)
    try:
        base, _ = v2._load_base(Path(root), result, reconciled)
    except (OSError, ValueError, KeyError):
        base = None
    current_version = base is not None and base.get("recovery_version") == VERSION
    expected = {
        int(record.global_tree_id): record.persistent_tree_id
        for record in reconciled.global_tree_records
    }
    if not recompute and current_version and set(existing) == set(expected) and all(
        existing[tree_id].get("persistent_tree_id") == persistent_id
        and existing[tree_id].get("dbh_method") != "REFERENCE_ESTIMATE"
        for tree_id, persistent_id in expected.items()
    ):
        # The proof loader revalidates the editor binding, membership hash and
        # every evidence hash. Re-running raw ROI queries for that identical
        # final generation changes no output and cost ~30 s on the 105 M-point
        # production survey, so reuse the already-certified rows.
        return {
            "schema": "treeline-dbh-finalized-cache-v1",
            "cache_hit": True,
            "trees": [existing[tree_id] for tree_id in sorted(existing)],
        }
    if base is None:
        v2.finalize_dbh(root, result, reconciled)
    return recover_finalized_dbh_v3(
        root, result, reconciled, recompute_auto=recompute or not current_version,
        reference_dbh_cm=reference_dbh_cm,
    )
