"""V2: new raw-stem association proved against current finalized geometry."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from algorithms.manual_span.dbh_recovery import DBH_RECOVERY_CONFIG, estimate_local_dbh
from algorithms.manual_span.metrics import _hag_chunked
from scipy.spatial import cKDTree

from core.stage_b import dbh_recovery as v1
from core.stage_b.dbh_evidence import (
    array_digest,
    confined,
    editor_eligibility,
    local_raw_query,
    read_json,
    sha256,
)

VERSION = "DBH_RECOVERY_V2"
MAX_SEEDS = 3
SEED_GRID_M = 0.35
ROI_RADIUS_M = 0.60
CONNECTION_BIN_M = 0.50
MIN_CONNECTION_BINS = 3
SAME_AXIS_M = 0.15


def contains(sorted_indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_indices, values)
    present = positions < len(sorted_indices)
    present[present] = sorted_indices[positions[present]] == values[present]
    return present


def conflicting_current_owner(indices, sorted_members, sorted_owners, target):
    present = contains(sorted_members, indices)
    positions = np.searchsorted(sorted_members, indices[present])
    return bool(np.any(sorted_owners[positions] != target))


def current_exclusions(root: Path, br: Path, original: list, operations: list) -> np.ndarray:
    """Current journal only; DELETE after MERGE expands to original source rows."""
    _, removed = editor_eligibility(original, operations)
    alive = {int(r["global_tree_id"]): {int(r["global_tree_id"])} for r in original}
    deleted = set()
    for op in operations:
        if op["type"] == "MERGE":
            alive[op["merged_id"]] = set().union(*(alive.pop(i) for i in op["tree_ids"]))
        elif op["type"] == "DELETE":
            for tid in op["tree_ids"]:
                deleted.update(alive.pop(tid))
    if deleted:
        with np.load(br / "tree_segmentation_original.npz", allow_pickle=False) as archive:
            ids = archive["original_index"][np.isin(archive["tree_id"], sorted(deleted))]
        removed = np.unique(np.concatenate((removed, ids)))
    return removed


def current_seeds(xyz: np.ndarray, hag: np.ndarray, hint: Any = None) -> tuple[list, dict]:
    low = xyz[(hag >= 2.0) & (hag <= 5.5)]
    proof = {
        "current_member_count": len(xyz),
        "current_low_point_count": len(low),
        "current_height_range_m": [float(np.min(hag)), float(np.max(hag))],
        "current_xy_bounds_m": [
            np.min(xyz[:, :2], axis=0).tolist(),
            np.max(xyz[:, :2], axis=0).tolist(),
        ],
        "historical_hint_used_as_owner": False,
    }
    proposals = []
    if len(low):
        cells, counts = np.unique(
            np.floor(low[:, :2] / SEED_GRID_M).astype(np.int64), axis=0, return_counts=True
        )
        for index in np.argsort(-counts, kind="stable"):
            xy = (cells[index] + 0.5) * SEED_GRID_M
            if all(np.linalg.norm(xy - np.asarray(s["xy"])) > ROI_RADIUS_M for s in proposals):
                proposals.append(
                    {
                        "xy": xy.tolist(),
                        "source": "CURRENT_LOW_MEMBER_MODE",
                        "support": int(counts[index]),
                    }
                )
            if len(proposals) == MAX_SEEDS:
                break
    if hint is not None:
        xy = np.asarray(hint, dtype=float)
        support = int(np.count_nonzero(np.linalg.norm(low[:, :2] - xy, axis=1) <= ROI_RADIUS_M))
        proof["historical_hint_current_support"] = support
        proof["historical_hint_status"] = "SUPPORTED_SEARCH_HINT" if support >= 12 else "REJECTED"
        # Current modes always win; a hint can only add a missing supported search location.
        if (
            support >= 12
            and len(proposals) < MAX_SEEDS
            and all(np.linalg.norm(xy - np.asarray(s["xy"])) > ROI_RADIUS_M for s in proposals)
        ):
            proposals.append(
                {"xy": xy.tolist(), "source": "CURRENT_SUPPORTED_HINT", "support": support}
            )
    return proposals, proof


def axis_connection(
    xyz: np.ndarray, hag: np.ndarray, origin: np.ndarray, direction: np.ndarray, diameter_cm: float
) -> dict:
    """Connection to surviving current points, independently of historical labels."""
    delta = xyz - origin
    radial = np.linalg.norm(delta - np.outer(delta @ direction, direction), axis=1)
    # Radius plus two accepted center-residual bounds and one continuity voxel.
    radius = diameter_cm / 200 + 2 * DBH_RECOVERY_CONFIG.maximum_center_track_residual_m + 0.02
    counts = [
        int(
            np.count_nonzero((radial <= radius) & (hag >= lower) & (hag < lower + CONNECTION_BIN_M))
        )
        for lower in np.arange(2.0, 5.5, CONNECTION_BIN_M)
    ]
    longest = run = 0
    for count in counts:
        run = run + 1 if count >= DBH_RECOVERY_CONFIG.minimum_points_per_section else 0
        longest = max(longest, run)
    return {
        "status": "PASS" if longest >= MIN_CONNECTION_BINS else "CURRENT_CONNECTION_UNPROVEN",
        "radius_m": radius,
        "height_bin_counts": counts,
        "longest_consecutive_bins": longest,
    }


def raw_bridge(xyz, hag, origin, direction, diameter_cm):
    delta = xyz - origin
    radial = np.linalg.norm(delta - np.outer(delta @ direction, direction), axis=1)
    radius = diameter_cm / 200 + 2 * DBH_RECOVERY_CONFIG.maximum_center_track_residual_m + 0.02
    counts = [
        int(np.count_nonzero((radial <= radius) & (hag >= low) & (hag < low + 0.25)))
        for low in (1.30, 1.55, 1.80)
    ]
    return {
        "status": "PASS" if min(counts) >= 12 else "RAW_TO_CURRENT_BRIDGE_UNPROVEN",
        "height_bin_counts": counts,
        "radius_m": radius,
    }


def select_candidate(candidates: list[dict]) -> tuple[dict | None, str]:
    accepted = [c for c in candidates if c.get("association_status") == "PASS"]
    if not accepted:
        return None, "NO_CURRENT_ASSOCIATED_STEM"
    accepted.sort(
        key=lambda c: (
            -c["fit"]["dbh_slice_count"],
            c["fit"]["dbh_fit_rmse"],
            tuple(c["axis_origin"]),
            c["candidate_id"],
        )
    )
    best = accepted[0]
    if any(
        np.linalg.norm(np.asarray(c["axis_origin"]) - best["axis_origin"]) > SAME_AXIS_M
        for c in accepted[1:]
    ):
        return None, "MULTIPLE_CURRENT_STEMS_AMBIGUOUS"
    return best, "PASS"


def summarize_candidates(row: dict) -> None:
    """Report physical axis candidates, not repeated pair-line fits or ROIs."""
    from algorithms.manual_span.dbh_raw_stem import _line_candidates_are_equivalent

    axes = []
    pair_count = 0
    candidates = row.get("v2_candidates", [])
    for candidate in candidates:
        detail = candidate.get("fit", {}).get("axis", {})
        for axis in detail.get("plausible_candidates", []):
            pair_count += 1
            if not any(_line_candidates_are_equivalent(axis, prior) for prior in axes):
                axes.append(axis)
    row["stem_candidate_count"] = len(axes)
    row["plausible_pair_proposal_count"] = pair_count
    row["valid_dbh_roi_count"] = sum(c.get("fit", {}).get("dbh_cm") is not None for c in candidates)
    if row.get("dbh_cm") is None:
        statuses = {c.get("association_status") for c in candidates}
        if "AMBIGUOUS_COMPETING_AXES" in statuses:
            row["dbh_fit_status"] = "AMBIGUOUS_COMPETING_AXES"
        elif len(statuses) == 1:
            row["dbh_fit_status"] = next(iter(statuses))


def _load_base(root: Path, result: Any, reconciled: Any) -> tuple[dict, dict]:
    br = v1._run_root(root, result.analysis_run_id)
    pointer = read_json(br / "dbh_final.json")
    path = confined(root, pointer["geometry_reference"])
    if sha256(path) != pointer["geometry_sha256"]:
        raise ValueError("DBH_V2_BASE_HASH_MISMATCH")
    payload = read_json(path)
    if payload["binding"] != v1._binding(root, result, reconciled):
        raise ValueError("DBH_V2_STALE_BASELINE")
    published = v1.load_finalized_dbh(root, result, reconciled)
    if any(r.get("dbh_fit_status") == "DBH_PROOF_INVALID_OR_STALE" for r in published.values()):
        raise ValueError("DBH_V2_INVALID_BASELINE")
    evidence = confined(root, payload["evidence_reference"])
    with np.load(evidence, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    payload["baseline_geometry_sha256"] = pointer["geometry_sha256"]
    payload["baseline_geometry_reference"] = path.relative_to(root).as_posix()
    return payload, arrays


def recover_finalized_dbh_v2(
    root: Path,
    result: Any,
    reconciled: Any,
    *,
    persist=True,
    progress=None,
    query_provider=None,
    member_hag=None,
) -> dict:
    """Explicit V2 pass; successful V1 rows are never refitted or relabeled."""
    from algorithms.manual_span.dbh_raw_stem import fit_axis_normal_dbh

    root = Path(root).resolve()
    base, old_arrays = _load_base(root, result, reconciled)
    binding = base["binding"]
    br = v1._run_root(root, result.analysis_run_id)
    operations = read_json(br / "tree_edits.json")["operations"]
    original = read_json(br / "batch-reconciliation.json")["trees"]
    original_hash = (
        sha256(br / "tree_segmentation_original.npz")
        if any(op["type"] == "DELETE" for op in operations)
        else None
    )
    removed = current_exclusions(root, br, original, operations)
    if contains(removed, np.asarray(reconciled.final_original_index)).any():
        raise ValueError("DBH_V2_FINAL_MEMBERS_CONTAIN_DELETED_POINTS")
    _, prep, ground, local_origin = v1._context(root, result)
    local_origin = np.asarray(local_origin)
    ground_tree = cKDTree(ground[:, :2])
    if member_hag is None:
        member_hag = _hag_chunked(
            reconciled.final_xyz_local_m, ground_tree, ground, local_origin, 250_000
        )
    if len(member_hag) != len(reconciled.final_original_index) or not np.isfinite(member_hag).all():
        raise ValueError("DBH_V2_INVALID_CURRENT_HAG")
    contexts = {
        int(r.global_tree_id): {
            "xyz": reconciled.final_xyz_local_m[reconciled.final_tree_id == r.global_tree_id],
            "hag": member_hag[reconciled.final_tree_id == r.global_tree_id],
        }
        for r in reconciled.global_tree_records
    }
    member_order = np.argsort(reconciled.final_original_index)
    sorted_members = reconciled.final_original_index[member_order]
    sorted_owners = reconciled.final_tree_id[member_order]
    cache_root = confined(root, prep["vegetation_role_tile_index_reference"])
    source = prep["ground_preparation"]["source_identity"]
    query = query_provider or local_raw_query
    successful_ids = [r["global_tree_id"] for r in base["trees"] if r["dbh_cm"] is not None]
    preserved_indices = np.sort(
        old_arrays["original_index"][np.isin(old_arrays["global_tree_id"], successful_ids)]
    )
    rows, blocks = [], []
    for old in sorted(base["trees"], key=lambda r: r["global_tree_id"]):
        row = copy.deepcopy(old)
        gid = row["global_tree_id"]
        if progress:
            progress(f"DBH_V2_TREE:{gid}")
        if row["dbh_cm"] is not None:
            keep = old_arrays["global_tree_id"] == gid
            blocks.append({key: value[keep] for key, value in old_arrays.items()})
            rows.append(row)
            continue
        context = contexts[gid]
        seeds, geometry = current_seeds(
            context["xyz"], context["hag"], row.get("stem_axis_xy_local_m")
        )
        row["v1_failure_reason"] = old["dbh_fit_status"]
        for key in (
            "fit_attempts",
            "dbh_continuity",
            "stem_axis_selection",
            "tiles",
            "raw_points_read",
            "opened_tiles",
            "radius_m",
            "center_xy_absolute_m",
            "z_bounds_absolute_m",
            "index_sha256",
            "source_identity",
            "source_identity_digest",
            "query_replay",
        ):
            row.pop(key, None)
        row.update(v1.unavailable("NO_CURRENT_ASSOCIATED_STEM"))
        row.update(
            dbh_source=VERSION,
            raw_local_query_used=False,
            current_geometry=geometry,
            current_editor_revision=binding["editor_revision"],
            v2_candidates=[],
            editor_excluded_point_count=0,
            raw_roi_point_count=0,
            raw_dbh_band_point_count=0,
            stem_candidate_count=0,
            raw_local_query_attempted=False,
        )
        eligible = {}
        seen_raw, seen_band = [], []
        for number, seed in enumerate(seeds):
            candidate = {"candidate_id": number, "seed": seed}
            row["v2_candidates"].append(candidate)
            try:
                center = np.asarray(seed["xy"]) + local_origin[:2]
                k = min(8, len(ground))
                ground_z = float(np.median(ground[ground_tree.query(center, k=k)[1], 2]))
                row["raw_local_query_attempted"] = True
                raw, indices, proof = query(
                    cache_root, source, center, (ground_z - 1, ground_z + 7)
                )
                raw, indices = np.asarray(raw), np.asarray(indices)
                if (
                    raw.shape != (len(indices), 3)
                    or indices.ndim != 1
                    or indices.dtype.kind not in "iu"
                    or not np.isfinite(raw).all()
                    or np.any(indices < 0)
                    or np.any(indices >= source["source_point_count"])
                    or len(np.unique(indices)) != len(indices)
                    or v1.strong_source(proof["source_identity"]) != v1.strong_source(source)
                ):
                    raise ValueError("DBH_V2_RAW_QUERY_PROVENANCE_INVALID")
                candidate["raw_query"] = proof
                row["raw_local_query_used"] = True
                seen_raw.append(indices)
                excluded = contains(removed, indices)
                candidate["editor_excluded_point_count"] = int(excluded.sum())
                raw, indices = raw[~excluded] - local_origin, indices[~excluded]
                hag = _hag_chunked(raw, ground_tree, ground, local_origin, 250_000)
                keep = (hag >= 0.5) & (hag <= 5.5)
                raw, indices, hag = raw[keep], indices[keep], hag[keep]
                candidate["dbh_band_point_count"] = int(
                    np.count_nonzero((hag >= 0.85) & (hag <= 1.75))
                )
                seen_band.append(indices[(hag >= 0.85) & (hag <= 1.75)])
                fit = fit_axis_normal_dbh(raw, hag, expected_xy=np.asarray(seed["xy"]))
                selected = np.asarray(fit.pop("selected_input_indices", []), dtype=np.int64)
                fit.pop("selected_original_indices", None)
                candidate["fit"] = fit
                if fit.get("dbh_cm") is None:
                    candidate["association_status"] = fit["dbh_fit_status"]
                    continue
                axis_origin = np.asarray(fit["axis"]["breast_height_origin_xyz"], dtype=float)
                direction = np.asarray(fit["axis"]["unit_direction_xyz"], dtype=float)
                candidate["axis_origin"] = axis_origin.tolist()
                candidate["raw_bridge"] = raw_bridge(
                    raw, hag, axis_origin, direction, fit["dbh_cm"]
                )
                if candidate["raw_bridge"]["status"] != "PASS":
                    candidate["association_status"] = "RAW_TO_CURRENT_BRIDGE_UNPROVEN"
                    continue
                candidate["current_connection"] = axis_connection(
                    context["xyz"], context["hag"], axis_origin, direction, fit["dbh_cm"]
                )
                if candidate["current_connection"]["status"] != "PASS":
                    candidate["association_status"] = "CURRENT_CONNECTION_UNPROVEN"
                    continue
                competitors = [
                    other
                    for other, value in contexts.items()
                    if other != gid
                    and axis_connection(
                        value["xyz"], value["hag"], axis_origin, direction, fit["dbh_cm"]
                    )["status"]
                    == "PASS"
                ]
                candidate["competing_current_tree_ids"] = competitors
                owner_conflict = conflicting_current_owner(
                    indices[selected], sorted_members, sorted_owners, gid
                )
                candidate["exact_current_owner_conflict"] = owner_conflict
                if (
                    competitors
                    or owner_conflict
                    or contains(preserved_indices, indices[selected]).any()
                ):
                    candidate["association_status"] = "NEIGHBOR_STEM_OWNERSHIP_AMBIGUOUS"
                    continue
                candidate["association_status"] = "PASS"
                candidate["selected_original_index_sha256"] = array_digest(indices[selected])
                eligible[number] = {
                    "xyz_local_m": raw[selected],
                    "hag_m": hag[selected],
                    "original_index": indices[selected],
                    "global_tree_id": np.full(len(selected), gid, dtype=np.int32),
                }
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                candidate["association_status"] = str(exc)
        unique_raw = (
            np.unique(np.concatenate(seen_raw)) if seen_raw else np.empty(0, dtype=np.int64)
        )
        row["raw_roi_point_count"] = len(unique_raw)
        row["raw_local_point_count"] = len(unique_raw)
        row["editor_excluded_point_count"] = int(contains(removed, unique_raw).sum())
        row["raw_dbh_band_point_count"] = (
            len(np.unique(np.concatenate(seen_band))) if seen_band else 0
        )
        chosen, reason = select_candidate(row["v2_candidates"])
        if chosen is not None:
            row.update(
                {
                    key: chosen["fit"].get(key)
                    for key in v1.DBH_FIELDS
                    if key not in {"dbh_source", "dbh_model_version"}
                }
            )
            row.update(
                dbh_source=VERSION,
                stem_association_status="CURRENT_RAW_REASSOCIATED",
                selected_candidate_id=chosen["candidate_id"],
                stem_axis_xy_local_m=chosen["axis_origin"][:2],
            )
            blocks.append(eligible[chosen["candidate_id"]])
            evidence_hag = eligible[chosen["candidate_id"]]["hag_m"]
            row["dbh_evidence_point_count"] = len(evidence_hag)
            row["dbh_point_count"] = len(evidence_hag)
            row["dbh_model_version"] = VERSION
            row["dbh_band_point_count"] = int(
                np.count_nonzero((evidence_hag >= 0.85) & (evidence_hag <= 1.75))
            )
        else:
            row["dbh_fit_status"] = reason
            row["stem_association_status"] = reason
            row["dbh_evidence_point_count"] = row["dbh_band_point_count"] = 0
        rows.append(row)
    arrays = {
        key: np.concatenate([block[key] for block in blocks]) if blocks else old_arrays[key][:0]
        for key in old_arrays
    }
    if len(np.unique(arrays["original_index"])) != len(arrays["original_index"]):
        raise ValueError("DBH_V2_CROSS_TREE_EVIDENCE_AMBIGUOUS")
    if contains(removed, arrays["original_index"]).any():
        raise ValueError("DBH_V2_DELETED_EVIDENCE_REINTRODUCED")
    estimates = estimate_local_dbh(rows, [r for r in rows if r["dbh_cm"] is None])
    for row in rows:
        if row["dbh_cm"] is None:
            estimate = estimates.get(row["persistent_tree_id"])
            if estimate:
                model = estimate.get("model_validation", {})
                loo = model.get("loo", [])
                errors = [
                    r["predicted_dbh_cm"] - r["observed_dbh_cm"]
                    for r in loo
                    if r.get("predicted_dbh_cm") is not None
                ]
                complete = bool(errors) and len(errors) == model.get("training_count")
                model.update(
                    loo_rmse_cm=float(np.sqrt(np.mean(np.square(errors)))) if complete else None,
                    loo_mae_cm=float(np.mean(np.abs(errors))) if complete else None,
                    loo_bias_cm=float(np.mean(errors)) if complete else None,
                )
            row["estimation_diagnostics"] = estimate
            if estimate and estimate["dbh_method"] == "ALLOMETRIC_ESTIMATE":
                row.update(estimate)
    if binding != v1._binding(root, result, reconciled):
        raise ValueError("DBH_V2_PROJECT_CHANGED_DURING_RECOVERY")
    if original_hash is not None and sha256(br / "tree_segmentation_original.npz") != original_hash:
        raise ValueError("DBH_V2_ORIGINAL_MEMBERS_CHANGED_DURING_RECOVERY")
    for row in rows:
        if row.get("dbh_source") == VERSION:
            summarize_candidates(row)
    payload = {
        "schema": base["schema"],
        "binding": binding,
        "recovery_version": VERSION,
        "baseline_geometry_sha256": base["baseline_geometry_sha256"],
        "baseline_geometry_reference": base["baseline_geometry_reference"],
        "span_proofs": base.get("span_proofs", {}),
        "trees": rows,
        "editor_excluded_index_sha256": array_digest(removed),
        "editor_excluded_index_count": len(removed),
        "additional_input_hashes": (
            [
                {
                    "reference": (br / "tree_segmentation_original.npz")
                    .relative_to(root)
                    .as_posix(),
                    "sha256": original_hash,
                }
            ]
            if any(op["type"] == "DELETE" for op in operations)
            else []
        ),
        "current_member_hag_sha256": array_digest(np.asarray(member_hag)),
    }
    if persist:
        path = v1._persist(root, result, payload, arrays)
        payload["published_reference"] = path.relative_to(root).as_posix()
    return payload


def finalize_dbh(root: Path, result: Any, reconciled: Any) -> dict:
    """Finalize explicitly creates a fresh V1 base if the old revision is stale."""
    try:
        _load_base(Path(root), result, reconciled)
    except (OSError, ValueError, KeyError):
        v1.recover_finalized_dbh(root, result, reconciled)
    return recover_finalized_dbh_v2(root, result, reconciled)
