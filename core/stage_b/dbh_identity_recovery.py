"""V3: resolve current crown/stem identity jointly before separate DBH fitting."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from algorithms.manual_span import stem_identity as geometry
from algorithms.manual_span.dbh_breast_height import (
    fit_original_tree_member_breast_height,
    fit_raw_breast_height_section,
)
from algorithms.manual_span.metrics import _hag_chunked
from algorithms.manual_span.stem_ownership import resolve_stem_ownership
from scipy.spatial import cKDTree

from core.stage_b import dbh_raw_recovery as v2
from core.stage_b import dbh_recovery as v1
from core.stage_b.dbh_evidence import (
    array_digest,
    confined,
    local_raw_query,
    raw_ground_reference,
    raw_tree_lowest_reference,
    original_raw_query,
    ground_reference_at_stem,
    calibrated_breast_height,
    read_json,
    sha256,
)

VERSION = "DBH_RECOVERY_V8_ALL_TREES_RAW_LOWEST_1P30"
BH_TARGET_M = 1.30
BH_SECTION_CENTER_MAX_DISTANCE_M = 0.10
BH_SEED_MAX_DISTANCE_M = 0.30
BH_OWNERSHIP_MARGIN_M = 0.08
BH_MAX_RELATIVE_DIAMETER_SPREAD = 0.25
BH_ESTIMATE_MAX_DIAMETER_M = 0.80
BH_BASELINE_DIAMETER_MAX_RELATIVE_CHANGE = 0.50


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
    measured = row.get("dbh_review", {}).get("fit", {})
    anchor = measured.get("breast_height_origin_xyz")
    direction = measured.get("unit_direction_xyz")
    identified = row.get("v3_final_fit", {}).get("identified_axis", {})
    if anchor is None or direction is None:
        anchor = identified.get("bh_reference_xyz")
        direction = identified.get("direction_xyz")
    line = (np.asarray(anchor), np.asarray(direction), None) if anchor and direction else (
        geometry._fit_hag_line(np.asarray(centers), np.asarray(heights), np.arange(len(centers)))
    )
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


def _breast_height_section_estimate(
    sections,
    seed_xy,
    seed_registry,
    tree_id,
    expected_dbh_cm=None,
    seed_max_distance_m=BH_SEED_MAX_DISTANCE_M,
):
    """Return a conservative raw-ground 1.30 m section estimate.

    This is deliberately an estimate, not a recovered high-confidence stem.
    It is used only when the global crown-to-stem resolver cannot publish a
    direct fit, and requires agreeing circles immediately below and above
    breast height plus unambiguous proximity to the current tree seed.
    """

    below, above = [], []
    for section in sections:
        for proposal in section.get("proposals", []):
            if not proposal.get("proposal_ready"):
                continue
            center_hag = proposal.get("center_hag_m")
            diameter = proposal.get("diameter_m")
            center_x = proposal.get("center_x_m")
            center_y = proposal.get("center_y_m")
            if any(value is None for value in (center_hag, diameter, center_x, center_y)):
                continue
            item = {
                "proposal": proposal,
                "center": np.asarray((center_x, center_y), dtype=np.float64),
                "hag": float(center_hag),
                "diameter_m": float(diameter),
            }
            if item["hag"] <= BH_TARGET_M + 1.0e-9:
                below.append(item)
            if item["hag"] >= BH_TARGET_M - 1.0e-9:
                above.append(item)

    candidates = []
    for lower in below:
        for upper in above:
            if lower["hag"] >= upper["hag"] - 1.0e-9:
                continue
            if not (
                lower["hag"] >= BH_TARGET_M - 0.20
                and upper["hag"] <= BH_TARGET_M + 0.20
            ):
                continue
            center_gap = float(np.linalg.norm(lower["center"] - upper["center"]))
            if center_gap > BH_SECTION_CENTER_MAX_DISTANCE_M:
                continue
            mean_diameter = (lower["diameter_m"] + upper["diameter_m"]) / 2.0
            relative_spread = abs(lower["diameter_m"] - upper["diameter_m"]) / mean_diameter
            if relative_spread > BH_MAX_RELATIVE_DIAMETER_SPREAD:
                continue
            fraction = (BH_TARGET_M - lower["hag"]) / (upper["hag"] - lower["hag"])
            center = lower["center"] + fraction * (upper["center"] - lower["center"])
            diameter = lower["diameter_m"] + fraction * (
                upper["diameter_m"] - lower["diameter_m"]
            )
            if diameter > BH_ESTIMATE_MAX_DIAMETER_M:
                continue
            baseline_difference = None
            if expected_dbh_cm is not None:
                baseline_difference = abs(diameter * 100.0 - expected_dbh_cm) / expected_dbh_cm
                if baseline_difference > BH_BASELINE_DIAMETER_MAX_RELATIVE_CHANGE:
                    continue
            distances = {
                gid: min(float(np.linalg.norm(center - seed)) for seed in seeds)
                for gid, seeds in seed_registry.items()
                if seeds
            }
            own_distance = distances.get(tree_id, float("inf"))
            foreign = min(
                (distance for gid, distance in distances.items() if gid != tree_id),
                default=float("inf"),
            )
            ownership_margin = foreign - own_distance
            if own_distance > seed_max_distance_m:
                continue
            if np.isfinite(foreign) and ownership_margin < BH_OWNERSHIP_MARGIN_M:
                continue
            support = sum(
                int(item["proposal"].get("anchor_inlier_point_count", 0))
                for item in (lower, upper)
            )
            rmse = max(
                float(item["proposal"].get("fit_rmse_m") or 0.0)
                for item in (lower, upper)
            )
            candidates.append(
                {
                    "dbh_cm": float(diameter * 100.0),
                    "center_xy_local_m": center.tolist(),
                    "point_count": support,
                    "fit_rmse_m": rmse,
                    "uncertainty_cm": float(
                        max(0.5, abs(lower["diameter_m"] - upper["diameter_m"]) * 50.0)
                    ),
                    "section_center_distance_m": center_gap,
                    "relative_diameter_spread": relative_spread,
                    "seed_distance_m": own_distance,
                    "ownership_margin_m": None
                    if not np.isfinite(ownership_margin)
                    else ownership_margin,
                    "baseline_relative_difference": baseline_difference,
                    "lower": lower["proposal"],
                    "upper": upper["proposal"],
                }
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["baseline_relative_difference"]
            if item["baseline_relative_difference"] is not None
            else float("inf"),
            item["seed_distance_m"],
            item["relative_diameter_spread"],
            item["section_center_distance_m"],
            -item["point_count"],
        ),
    )


def _matching_proposal_history(root, br, binding):
    """Recover proposals after undo, never diameters or old acceptance claims.

    A journal revision changes on undo even when every member is restored.
    Rebuilding V1/V2 then discards the prior proposal registry. Only snapshots
    with identical source, preparation, ground and tree identities may
    contribute seed locations. Modern snapshots also require exact canonical
    membership; legacy row-order snapshots are proposals only, never proof.
    """
    keys = ("project_id", "analysis_run_id", "source_identity", "preparation_sha256",
            "ground_sha256", "original_lineage_sha256")
    found = {}
    paths = sorted((br / "dbh_v1").glob("*/finalized_geometry.json"),
                   key=lambda p: p.stat().st_mtime_ns, reverse=True)[:128]
    for path in paths:
        path = confined(root, path.relative_to(root).as_posix())
        try:
            snapshot = read_json(path)
            previous = snapshot.get("binding", {})
            if any(previous.get(key) != binding.get(key) for key in keys):
                continue
            old_members, members = previous.get("membership", {}), binding.get("membership", {})
            legacy = old_members != members
            if legacy and (old_members.get("ordering") is not None
                           or not old_members.get("identities")
                           or old_members.get("identities") != members.get("identities")):
                continue
            for row in snapshot.get("trees", []):
                gid = row["global_tree_id"]
                if gid in found or row.get("dbh_cm") is None:
                    continue
                seeds = [copy.deepcopy(q["seed"]) for q in row.get("v3_queries", [])
                         if q.get("seed", {}).get("source") == "BASELINE_ACCEPTED_STEM_AXIS"]
                xy = row.get("stem_axis_xy_local_m")
                if not seeds and xy is not None:
                    seeds = [{"xy": list(xy), "source": "BASELINE_ACCEPTED_STEM_AXIS",
                              "support": int(row.get("dbh_point_count") or 0)}]
                if seeds:
                    found[gid] = {"seed": seeds[0],
                                  "reference": path.relative_to(root).as_posix(),
                                  "sha256": sha256(path), "legacy_ordering_proposal_only": legacy}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def recover_finalized_dbh_v3(
    root, result, reconciled, *, persist=True, progress=None, query_provider=None, member_hag=None
):
    root = Path(root).resolve()
    base, old_arrays = v2._load_base(root, result, reconciled)
    binding = base["binding"]
    br = v1._run_root(root, result.analysis_run_id)
    proposal_history = _matching_proposal_history(root, br, binding)
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
    query = query_provider or (lambda cache, source, center, bounds:
                               original_raw_query(root, cache, source, center, bounds))
    baseline_rows = {}
    baseline_reference = base.get("baseline_geometry_reference")
    baseline_sha256 = base.get("baseline_geometry_sha256")
    if baseline_reference:
        baseline_path = confined(root, baseline_reference)
        if baseline_path.is_file() and (
            not baseline_sha256 or sha256(baseline_path) == baseline_sha256
        ):
            baseline_rows = {
                row["global_tree_id"]: row for row in read_json(baseline_path).get("trees", [])
            }
    # Explicit human-confirmed evidence is retained.  All automatic values,
    # including prior V1/V2 values, are remeasured so none can silently keep
    # the old processed-ground height reference.
    fixed = [
        _fixed_claim(row, old_arrays)
        for row in base["trees"]
        if row["dbh_cm"] is not None
        and row["dbh_method"] == "REVIEW_CONFIRMED_STEM"
        and row.get("dbh_section_measurement", {}).get("diameter_derived_from")
        == "ACTUAL_RAW_1P30_SLAB_CIRCLE_NOT_INTERPOLATED"
    ]
    seed_registry = {}
    saved_by_tree = {}
    raw_axis_proposals = {}
    # Preserve the new stem-first proposal through V1/V2/V5. Dense crown modes
    # must not consume all three ROI slots and push the measured raw shaft out.
    # This is a search position only: current members, raw ground, slab fit and
    # joint ownership are independently checked below, just like other seeds.
    for tree_root in sorted(br.glob("*/tree")):
        manifest_path = tree_root / "span_tree_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("membership_diagnostics", {}).get("version") not in {
            "RAW_SECTIONS_STEM_FIRST_V1", "RAW_SECTIONS_STEM_FIRST_V2_STRUCTURAL_PLANES"
        }:
            continue
        span_proof = v1._span_evidence(tree_root)
        for source_tree, anchor in span_proof["anchors"].items():
            raw_axis_proposals[(int(manifest["span_id"]), int(source_tree))] = anchor
    for old in base["trees"]:
        gid = old["global_tree_id"]
        current_raw_seed = None
        saved = old.get("v2_candidates", [])
        if not saved:
            seeds, _ = v2.current_seeds(contexts[gid]["xyz"], contexts[gid]["hag"])
            saved = [{"seed": seed} for seed in seeds]
        baseline_row = baseline_rows.get(gid, {})
        baseline_xy = baseline_row.get("stem_axis_xy_local_m")
        proposals = [raw_axis_proposals[tuple(link)] for link in old.get("lineage", [])
            if tuple(link) in raw_axis_proposals]
        if proposals and all(np.linalg.norm(p["xy"] - proposals[0]["xy"]) <= .15 for p in proposals):
            current = contexts[gid]
            close = np.linalg.norm(current["xyz"][:, :2] - proposals[0]["xy"], axis=1) <= .30
            low = close & (current["hag"] >= .35) & (current["hag"] <= 3.9)
            levels = np.unique(np.floor(current["hag"][low] / .4))
            if np.count_nonzero(low) >= 24 and len(levels) >= 3:
                seed = {"xy": proposals[0]["xy"].tolist(), "source": "CURRENT_RAW_SHAFT_AXIS",
                    "support": int(np.count_nonzero(low))}
                current_raw_seed = seed
                saved = [{"seed": seed}] + [item for item in saved
                    if np.linalg.norm(np.asarray(item["seed"]["xy"]) - proposals[0]["xy"]) > .05]
        # A repeated V5 calculation uses the current V5 publication as its
        # immediate baseline. Preserve the original proposal (not its fitted
        # diameter) so rerunning identical members cannot silently lose the
        # V2 accepted-axis ROI. Every proposal is still reread and revalidated
        # against the original LAS, original ground and current ownership.
        prior_baseline_seed = next((q.get("seed") for q in old.get("v3_queries", [])
            if q.get("seed", {}).get("source") == "BASELINE_ACCEPTED_STEM_AXIS"), None)
        if prior_baseline_seed is None and gid in proposal_history:
            prior_baseline_seed = proposal_history[gid]["seed"]
        if prior_baseline_seed is not None:
            saved = [{"seed": copy.deepcopy(prior_baseline_seed)}] + [
                item for item in saved if np.linalg.norm(
                    np.asarray(item["seed"]["xy"], dtype=np.float64)
                    - np.asarray(prior_baseline_seed["xy"], dtype=np.float64)) > 0.05
            ]
        if (
            baseline_row.get("dbh_cm") is not None
            and baseline_row.get("dbh_method")
            in {"DIRECT_STEM", "RECOVERED_STEM", "REVIEW_CONFIRMED_STEM"}
            and baseline_xy is not None
        ):
            baseline_seed = {
                "seed": {
                    "xy": list(baseline_xy),
                    "source": "BASELINE_ACCEPTED_STEM_AXIS",
                    "support": int(baseline_row.get("dbh_point_count") or 0),
                }
            }
            saved = [baseline_seed] + [
                item
                for item in saved
                if np.linalg.norm(
                    np.asarray(item["seed"]["xy"], dtype=np.float64)
                    - np.asarray(baseline_xy, dtype=np.float64)
                )
                > 0.05
            ]
        if current_raw_seed is not None:
            # Deterministic current inputs only: a previous fitted centre must
            # not change this backend's proposal order on repeated finalize.
            current_modes, _ = v2.current_seeds(contexts[gid]["xyz"], contexts[gid]["hag"])
            saved = [{"seed": current_raw_seed}] + [{"seed": seed} for seed in current_modes
                if np.linalg.norm(np.asarray(seed["xy"]) - current_raw_seed["xy"]) > .05]
        saved = saved[: v2.MAX_SEEDS]
        saved_by_tree[gid] = saved
        seed_registry[gid] = [np.asarray(item["seed"]["xy"], dtype=np.float64) for item in saved]

    rows, jobs, claims, section_jobs = [], {}, [], {}
    for old in sorted(base["trees"], key=lambda row: row["global_tree_id"]):
        row = copy.deepcopy(old)
        gid = row["global_tree_id"]
        rows.append(row)
        if any(claim["tree_id"] == gid for claim in fixed):
            continue
        if progress:
            progress(f"V3_CURRENT_CROWN:{gid}")
        row.update(v1.unavailable("NO_CROWN_TO_STEM_IDENTITY"))
        row.update(
            dbh_source=VERSION,
            v2_failure_reason=old.get("v2_failure_reason", old["dbh_fit_status"]),
            v3_candidates=[],
            v3_queries=[],
            current_editor_revision=binding["editor_revision"],
        )
        row.pop("v3_final_fit", None)
        row.pop("selected_candidate_id", None)
        saved_candidates = saved_by_tree[gid]
        if len(saved_candidates) > v2.MAX_SEEDS:
            raise ValueError("V3_ROI_BUDGET_EXCEEDED")
        seen_raw, seen_deleted = [], []
        for roi_id, previous in enumerate(saved_candidates):
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
                query_record.update(proof=proof, editor_excluded_point_count=int(deleted.sum()))
                raw, idx = raw[~deleted], idx[~deleted]
                ground_reference = raw_tree_lowest_reference(
                    contexts[gid]["xyz"] + origin,
                    processed_ground_prior_z=ground_z,
                )
                query_record["ground_reference"] = ground_reference
                hag = raw[:, 2] - ground_reference["ground_z_absolute_m"]
                raw = raw - origin
                low = (hag >= 0.5) & (hag <= 5.5)
                raw, idx, hag = raw[low], idx[low], hag[low]
                bh_sections, _, _, _ = geometry._section_proposals(raw, hag)
                query_record["breast_height_proposals"] = {"sections": bh_sections}
                section_estimate = _breast_height_section_estimate(
                    bh_sections,
                    previous["seed"]["xy"],
                    seed_registry,
                    gid,
                    None,  # A processed/legacy diameter must not select the new measurement.
                    0.45
                    if previous["seed"].get("source") == "BASELINE_ACCEPTED_STEM_AXIS"
                    else BH_SEED_MAX_DISTANCE_M,
                )
                if section_estimate is not None:
                    section_ground = calibrated_breast_height(
                        ground_reference, np.asarray(section_estimate["center_xy_local_m"]) + origin[:2],
                        rows,
                    )
                    section_hag = raw[:, 2] + origin[2] - section_ground["ground_z_absolute_m"]
                    section_estimate.update(
                        roi_id=roi_id,
                        ground_reference=section_ground,
                    )
                    measurement = fit_raw_breast_height_section(
                        raw, section_hag, section_estimate["center_xy_local_m"],
                        section_estimate["dbh_cm"] / 200,
                    )
                    query_record["actual_1p30_section_fit"] = measurement
                    if measurement["dbh_cm"] is None:
                        section_estimate = None
                    else:
                        section_estimate.update(
                            dbh_cm=measurement["dbh_cm"], point_count=measurement["dbh_point_count"],
                            fit_rmse_m=measurement["dbh_fit_rmse"],
                            uncertainty_cm=measurement["dbh_uncertainty_cm"],
                            measurement=measurement,
                            evidence=dict(xyz_local_m=raw[measurement["selected_input_indices"]],
                                          hag_m=section_hag[measurement["selected_input_indices"]],
                                          original_index=idx[measurement["selected_input_indices"]]),
                        )
                    prior = section_jobs.get(gid)
                    if section_estimate is not None and (prior is None or (
                        section_estimate["seed_distance_m"],
                        section_estimate["relative_diameter_spread"],
                    ) < (prior["seed_distance_m"], prior["relative_diameter_spread"])):
                        section_jobs[gid] = section_estimate
                axes, proposal_proof = geometry.upper_axes(raw, hag)
                query_record["upper_proposals"] = proposal_proof
                for axis_number, axis in enumerate(axes):
                    identifier = f"g{gid}-r{roi_id}-a{axis_number}"
                    tracked = geometry.track_downward(raw, hag, idx, axis)
                    support = tracked.pop("original_index")
                    effective = tracked["refined_axis"] or axis
                    axis_origin = np.asarray(effective["bh_reference_xyz"])
                    direction = np.asarray(effective["direction_xyz"])
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
                        scores[other] = geometry.score_crown(
                            tracked, effective, context["anchor"], connection, bridge
                        )
                    item = {
                        "candidate_id": identifier,
                        "roi_id": roi_id,
                        "axis": axis,
                        "track": tracked,
                        "crown_scores": {str(key): value for key, value in scores.items()},
                        "identity_support_count": len(support),
                        "identity_support_original_index_sha256": array_digest(support),
                        "editor_excluded_point_count": int(deleted.sum()),
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
                        "ground_reference": ground_reference,
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
        if row["v3_queries"] and all(q.get("status") == "UNAVAILABLE" for q in row["v3_queries"]):
            row["raw_source_failure"] = row["v3_queries"][0]["reason"]
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
    resolution = resolve_stem_ownership(claims, fixed)
    # Global resolver output must be identical under a reverse presentation order.
    if resolution != resolve_stem_ownership(list(reversed(claims)), list(reversed(fixed))):
        raise ValueError("GLOBAL_OWNERSHIP_NONDETERMINISTIC")
    blocks = []
    for row in rows:
        gid = row["global_tree_id"]
        if any(claim["tree_id"] == gid for claim in fixed):
            mask = old_arrays["global_tree_id"] == gid
            blocks.append({key: value[mask] for key, value in old_arrays.items()})
            continue
        for candidate in row["v3_candidates"]:
            candidate["decision"] = resolution["decisions"].get(candidate["candidate_id"])
        chosen_id = resolution["selected"].get(gid)
        row["dbh_fit_status"] = resolution["tree_status"].get(gid, "NO_CROWN_TO_STEM_IDENTITY")
        if row.get("raw_source_failure"):
            row["dbh_fit_status"] = row["raw_source_failure"]
        row["dbh_evidence_point_count"] = row["dbh_band_point_count"] = 0
        if chosen_id is None:
            estimate = section_jobs.get(gid)
            if estimate is not None:
                reference = estimate["ground_reference"]
                evidence = estimate.pop("evidence")
                selected_idx = evidence["original_index"]
                if v2.contains(removed, selected_idx).any() or v2.conflicting_current_owner(
                    selected_idx, sorted_members, sorted_owners, gid
                ):
                    row["dbh_fit_status"] = "RAW_1P30_SECTION_SOURCE_OWNERSHIP_CONFLICT"
                    continue
                blocks.append({**evidence, "global_tree_id": np.full(len(selected_idx), gid, np.int32)})
                row.update(
                    dbh_cm=estimate["dbh_cm"],
                    dbh_method="RAW_GROUND_BH_SECTION",
                    dbh_quality="ESTIMATED",
                    dbh_point_count=estimate["point_count"],
                    dbh_evidence_point_count=estimate["point_count"],
                    dbh_band_point_count=estimate["point_count"],
                    dbh_fit_rmse=estimate["fit_rmse_m"],
                    dbh_slice_count=1,
                    dbh_fit_status="PASS_ACTUAL_RAW_GROUND_1P30_SECTION",
                    dbh_source=VERSION,
                    dbh_model_version=VERSION,
                    dbh_uncertainty_cm=estimate["uncertainty_cm"],
                    stem_association_status="CURRENT_SEED_BH_SECTION_ESTIMATE",
                    stem_axis_xy_local_m=estimate["center_xy_local_m"],
                    dbh_ground_z_absolute_m=reference["ground_z_absolute_m"],
                    dbh_measurement_z_absolute_m=reference["breast_height_z_absolute_m"],
                    dbh_height_reference="ORIGINAL_RAW_TREE_LOWEST_PLUS_1P30M",
                    dbh_ground_reference_method=reference["ground_reference_method"],
                    dbh_ground_reference_point_count=reference[
                        "raw_ground_support_point_count"
                    ],
                    raw_ground_bh_section_estimate=estimate,
                    dbh_section_measurement=estimate["measurement"],
                    dbh_site_slope_deg=reference.get('site_slope_deg'),
                    dbh_common_height_applied=reference.get('common_height_applied', False),
                    dbh_measurement_height_policy=reference.get('measurement_height_policy'),
                    dbh_calibration_tree_id=reference.get('calibration_tree_id'),
                )
            continue
        if chosen_id not in jobs:
            row["dbh_fit_status"] = "FOREIGN_STEM_CLAIM_NOT_ASSIGNABLE"
            continue
        job = jobs[chosen_id]
        if job["source_tree"] != gid:
            # A foreign claim exists solely to veto duplicated ownership.
            row["dbh_fit_status"] = "FOREIGN_STEM_CLAIM_NOT_ASSIGNABLE"
            continue
        fitted = geometry.fit_identified_stem(job["raw"], job["hag"], job["axis"])
        positions = np.asarray(fitted.pop("selected_input_indices"), dtype=np.int64)
        row["v3_final_fit"] = fitted
        row["selected_candidate_id"] = chosen_id
        row["dbh_fit_status"] = fitted["dbh_fit_status"]
        # The 0.85--1.75 m multisection fit is an identity/axis diagnostic only.
        # Once identity is established, measure the actual target slab even
        # when taper/branches make that older pooled diameter unavailable.
        stem_ground = calibrated_breast_height(
            job["ground_reference"], np.asarray(job["axis"]["bh_reference_xyz"][:2]) + origin[:2],
            rows,
        )
        stem_hag = job["raw"][:, 2] + origin[2] - stem_ground["ground_z_absolute_m"]
        measurement = fit_raw_breast_height_section(
            job["raw"], stem_hag, job["axis"]["bh_reference_xyz"][:2],
            fitted["dbh_cm"] / 200 if fitted["dbh_cm"] is not None else job["item"]["axis"]["radius_m"],
        )
        row["dbh_section_measurement"] = measurement
        if measurement["dbh_cm"] is None:
            row["dbh_fit_status"] = measurement["dbh_fit_status"]
            continue
        fitted.update({key: measurement.get(key) for key in v1.DBH_FIELDS if key in measurement})
        positions = np.asarray(measurement["selected_input_indices"], dtype=np.int64)
        job["ground_reference"] = stem_ground
        job["hag"] = stem_hag
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
            dbh_ground_z_absolute_m=job["ground_reference"]["ground_z_absolute_m"],
            dbh_measurement_z_absolute_m=job["ground_reference"][
                "breast_height_z_absolute_m"
            ],
            dbh_height_reference="ORIGINAL_RAW_TREE_LOWEST_PLUS_1P30M",
            dbh_ground_reference_method=job["ground_reference"][
                "ground_reference_method"
            ],
            dbh_ground_reference_point_count=job["ground_reference"][
                "raw_ground_support_point_count"
            ],
            dbh_site_slope_deg=stem_ground.get('site_slope_deg'),
            dbh_common_height_applied=stem_ground.get('common_height_applied', False),
            dbh_measurement_height_policy=stem_ground.get('measurement_height_policy'),
            dbh_calibration_tree_id=stem_ground.get('calibration_tree_id'),
        )
        blocks.append(
            {
                "xyz_local_m": job["raw"][positions],
                "hag_m": job["hag"][positions],
                "original_index": selected_idx,
                "global_tree_id": np.full(len(positions), gid, np.int32),
            }
        )
    # Final source-derived fallback: measure every still-unresolved tree from
    # that tree's full-resolution original LAS membership.  The reconciled
    # arrays retain exact original_index lineage and original coordinates;
    # they are not the downsampled/denoised display cloud.  No crown allometry
    # or placeholder value is permitted here.
    for row in rows:
        if row.get("dbh_cm") is not None:
            continue
        gid = int(row["global_tree_id"])
        member_mask = reconciled.final_tree_id == gid
        raw_tree_xyz = np.asarray(reconciled.final_xyz_local_m)[member_mask]
        raw_tree_index = np.asarray(reconciled.final_original_index)[member_mask]
        measurement = fit_original_tree_member_breast_height(raw_tree_xyz, raw_tree_index)
        row["dbh_section_measurement"] = measurement
        if measurement["dbh_cm"] is None:
            row["dbh_fit_status"] = measurement["dbh_fit_status"]
            continue
        positions = np.asarray(measurement["selected_input_indices"], dtype=np.int64)
        selected_idx = raw_tree_index[positions]
        if v2.contains(removed, selected_idx).any():
            row["dbh_fit_status"] = "ORIGINAL_TREE_1P30_SELECTED_POINT_WAS_DELETED"
            continue
        ground_absolute = float(measurement["ground_local_z_m"] + origin[2])
        measurement_absolute = float(measurement["measurement_local_z_m"] + origin[2])
        row.update(
            dbh_cm=measurement["dbh_cm"],
            dbh_method=measurement["dbh_method"],
            dbh_quality=measurement["dbh_quality"],
            dbh_point_count=measurement["dbh_point_count"],
            dbh_evidence_point_count=measurement["dbh_point_count"],
            dbh_band_point_count=measurement["dbh_point_count"],
            dbh_fit_rmse=measurement["dbh_fit_rmse"],
            dbh_slice_count=1,
            dbh_fit_status=measurement["dbh_fit_status"],
            dbh_source=VERSION,
            dbh_model_version=VERSION,
            dbh_uncertainty_cm=measurement["dbh_uncertainty_cm"],
            stem_association_status="ORIGINAL_TREE_MEMBER_1P30_FALLBACK",
            stem_axis_xy_local_m=measurement["center_xy_local_m"],
            dbh_ground_z_absolute_m=ground_absolute,
            dbh_measurement_z_absolute_m=measurement_absolute,
            dbh_height_reference="ORIGINAL_RAW_TREE_LOWEST_PLUS_1P30M",
            dbh_ground_reference_method="ORIGINAL_RAW_TREE_LOWEST_POINT",
            dbh_ground_reference_point_count=1,
            dbh_site_slope_deg=None,
            dbh_common_height_applied=True,
            dbh_measurement_height_policy="TREE_LOWEST_ORIGINAL_POINT_PLUS_1P30M",
            dbh_calibration_tree_id=gid,
        )
        blocks.append(
            {
                "xyz_local_m": raw_tree_xyz[positions],
                "hag_m": raw_tree_xyz[positions, 2] - measurement["ground_local_z_m"],
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
    if binding != v1._binding(root, result, reconciled):
        raise ValueError("V3_PROJECT_CHANGED_DURING_RECOVERY")
    if original_hash is not None and sha256(br / "tree_segmentation_original.npz") != original_hash:
        raise ValueError("V3_ORIGINAL_MEMBERS_CHANGED_DURING_RECOVERY")
    payload = {
        "schema": base["schema"],
        "binding": binding,
        "recovery_version": VERSION,
        "measurement_policy": "DIRECT_ORIGINAL_LAS_GROUND_PLUS_1P30M_ACTUAL_SLAB",
        "proposal_history_inputs": proposal_history,
        "baseline_geometry_reference": base["baseline_geometry_reference"],
        "baseline_geometry_sha256": base["baseline_geometry_sha256"],
        "span_proofs": base.get("span_proofs", {}),
        "trees": rows,
        "global_ownership": resolution,
        "allometry": allometry,
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


def finalize_dbh(root, result, reconciled):
    try:
        v2._load_base(Path(root), result, reconciled)
    except (OSError, ValueError, KeyError):
        v2.finalize_dbh(root, result, reconciled)
    return recover_finalized_dbh_v3(root, result, reconciled)
