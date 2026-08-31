"""Deterministic joint resolver for competing raw stem identities.

This module only resolves already-scored geometry candidates.  It does not
compute DBH, infer confidence, or assign crown membership.  All candidates
that pass the fixed absolute score gate participate in one symmetric conflict
graph before any per-tree choice is made.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import acos, degrees, isfinite
from typing import Any

import numpy as np

MIN_IDENTITY_SCORE = 0.75
MIN_SCORE_MARGIN = 0.10
AXIS_SHARED_DISTANCE_M = 0.15
AXIS_SHARED_ANGLE_DEG = 5.0
_EPSILON = 1.0e-12
_UNIT_TOLERANCE = 1.0e-6


def _empty_result() -> dict[str, Any]:
    return {"selected": {}, "decisions": {}, "tree_status": {}}


def _failure_result(records: list[tuple[str, int | None]], reason: str) -> dict[str, Any]:
    result = _empty_result()
    for candidate_id, tree_id in sorted(set(records), key=lambda item: item[0]):
        result["decisions"][candidate_id] = {
            "status": "INVALID",
            "reason": reason,
            "conflicts": [],
        }
        if tree_id is not None:
            result["tree_status"][int(tree_id)] = "INVALID_INPUT"
    return result


def _integer_tree_id(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _indices(value: Any) -> tuple[int, ...] | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or np.issubdtype(array.dtype, np.bool_):
        return None
    if np.issubdtype(array.dtype, np.integer):
        values = tuple(int(item) for item in array.tolist())
        if any(item < 0 for item in values) or len(values) != len(set(values)):
            return None
        return values
    return None


def _vector(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return array


def _normalize_claim(raw: Any, source: str) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    tree_id = _integer_tree_id(raw.get("tree_id"))
    candidate_id = raw.get("candidate_id")
    if tree_id is None or not isinstance(candidate_id, str) or not candidate_id:
        return None
    try:
        score = float(raw.get("score"))
    except (TypeError, ValueError, OverflowError):
        return None
    absolute_pass = raw.get("absolute_pass")
    if (
        not isfinite(score)
        or not 0.0 <= score <= 1.0
        or not isinstance(absolute_pass, (bool, np.bool_))
    ):
        return None
    origin = _vector(raw.get("axis_origin"))
    direction = _vector(raw.get("axis_direction"))
    index_values = _indices(raw.get("original_index"))
    if origin is None or direction is None or index_values is None:
        return None
    direction_norm = float(np.linalg.norm(direction))
    if not isfinite(direction_norm) or abs(direction_norm - 1.0) > _UNIT_TOLERANCE:
        return None
    return {
        "tree_id": tree_id,
        "candidate_id": candidate_id,
        "score": score,
        "absolute_pass": bool(absolute_pass),
        "axis_origin": origin,
        "axis_direction": direction / direction_norm,
        "indices": index_values,
        "source": source,
    }


def _shared_axis(first: dict[str, Any], second: dict[str, Any]) -> bool:
    direction = first["axis_direction"]
    other_direction = second["axis_direction"]
    cosine = float(np.clip(abs(np.dot(direction, other_direction)), -1.0, 1.0))
    angle = degrees(acos(cosine))
    if angle > AXIS_SHARED_ANGLE_DEG + _EPSILON:
        return False
    aligned = other_direction if np.dot(direction, other_direction) >= 0.0 else -other_direction
    mean_direction = direction + aligned
    mean_norm = float(np.linalg.norm(mean_direction))
    if mean_norm <= _EPSILON:
        return False
    mean_direction /= mean_norm
    delta = second["axis_origin"] - first["axis_origin"]
    offset = float(np.linalg.norm(delta - np.dot(delta, mean_direction) * mean_direction))
    return offset <= AXIS_SHARED_DISTANCE_M + _EPSILON


def _shares_stem(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if set(first["indices"]).intersection(second["indices"]):
        return True
    return _shared_axis(first, second)


def _qualified(claim: dict[str, Any]) -> bool:
    return bool(
        claim["source"] == "candidate"
        and claim["absolute_pass"]
        and claim["score"] >= MIN_IDENTITY_SCORE
    )


def resolve_stem_ownership(
    candidates: Iterable[Mapping[str, Any]],
    fixed: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve all qualified candidates jointly and fail closed on ambiguity."""

    try:
        raw_candidates = list(candidates)
        raw_fixed = list(fixed)
    except TypeError:
        return _empty_result()
    records: list[tuple[str, int | None]] = []
    normalized: list[dict[str, Any]] = []
    entries = [(item, "candidate") for item in raw_candidates]
    entries.extend((item, "fixed") for item in raw_fixed)
    for raw, source in entries:
        if isinstance(raw, Mapping):
            candidate_id = raw.get("candidate_id")
            tree_id = _integer_tree_id(raw.get("tree_id"))
            if isinstance(candidate_id, str) and candidate_id:
                records.append((candidate_id, tree_id))
        claim = _normalize_claim(raw, source)
        if claim is None:
            return _failure_result(records, "MALFORMED_CANDIDATE")
        normalized.append(claim)

    ids = [claim["candidate_id"] for claim in normalized]
    if len(ids) != len(set(ids)):
        return _failure_result(
            [(claim["candidate_id"], claim["tree_id"]) for claim in normalized],
            "DUPLICATE_CANDIDATE_ID",
        )
    fixed_tree_ids = [claim["tree_id"] for claim in normalized if claim["source"] == "fixed"]
    if len(fixed_tree_ids) != len(set(fixed_tree_ids)):
        return _failure_result(
            [(claim["candidate_id"], claim["tree_id"]) for claim in normalized],
            "DUPLICATE_FIXED_TREE_ID",
        )

    by_id = {claim["candidate_id"]: claim for claim in normalized}
    conflicts: dict[str, set[str]] = {candidate_id: set() for candidate_id in by_id}
    active = [claim for claim in normalized if _qualified(claim)]
    active.extend(claim for claim in normalized if claim["source"] == "fixed")
    active.sort(key=lambda claim: (claim["tree_id"], claim["candidate_id"]))
    for left_index, first in enumerate(active):
        for second in active[left_index + 1 :]:
            if first["tree_id"] == second["tree_id"] or not _shares_stem(first, second):
                continue
            first_id = first["candidate_id"]
            second_id = second["candidate_id"]
            conflicts[first_id].add(second_id)
            conflicts[second_id].add(first_id)

    result = _empty_result()
    for candidate_id in sorted(by_id):
        claim = by_id[candidate_id]
        claim_conflicts = sorted(conflicts[candidate_id])
        if claim["source"] == "fixed":
            status = "FIXED"
            reason = "FIXED_RESERVATION"
        elif not claim["absolute_pass"]:
            status = "INELIGIBLE_ABSOLUTE_GATE"
            reason = "ABSOLUTE_GATE_FAILED"
        elif claim["score"] < MIN_IDENTITY_SCORE:
            status = "INELIGIBLE_SCORE_GATE"
            reason = "IDENTITY_SCORE_BELOW_MINIMUM"
        elif claim_conflicts:
            status = "INELIGIBLE_CONFLICT"
            reason = "CONFLICTING_STEM"
        else:
            status = "PENDING"
            reason = "PENDING"
        result["decisions"][candidate_id] = {
            "status": status,
            "reason": reason,
            "conflicts": claim_conflicts,
        }

    tree_candidates: dict[int, list[dict[str, Any]]] = {}
    for claim in normalized:
        if claim["source"] == "candidate" and claim["absolute_pass"]:
            tree_candidates.setdefault(claim["tree_id"], []).append(claim)
    fixed_claims = [claim for claim in normalized if claim["source"] == "fixed"]
    fixed_by_tree = {claim["tree_id"]: claim for claim in fixed_claims}
    tree_ids = sorted({claim["tree_id"] for claim in normalized})
    for tree_id in tree_ids:
        if tree_id in fixed_by_tree:
            result["selected"][tree_id] = fixed_by_tree[tree_id]["candidate_id"]
            result["tree_status"][tree_id] = "FIXED_RESERVATION"
            for claim in tree_candidates.get(tree_id, []):
                claim_id = claim["candidate_id"]
                if not conflicts[claim_id]:
                    result["decisions"][claim_id]["status"] = "INELIGIBLE_FIXED_RESERVATION"
                    result["decisions"][claim_id]["reason"] = "FIXED_RESERVATION"
            continue
        absolute_candidates = sorted(
            tree_candidates.get(tree_id, []),
            key=lambda claim: (-claim["score"], claim["candidate_id"]),
        )
        all_for_tree = [
            claim
            for claim in normalized
            if claim["source"] == "candidate" and claim["tree_id"] == tree_id
        ]
        if not absolute_candidates:
            result["tree_status"][tree_id] = (
                "NO_ABSOLUTE_CANDIDATE"
                if not any(claim["absolute_pass"] for claim in all_for_tree)
                else "IDENTITY_SCORE_BELOW_MINIMUM"
            )
            continue
        top = absolute_candidates[0]
        runner = absolute_candidates[1] if len(absolute_candidates) > 1 else None
        if top["score"] < MIN_IDENTITY_SCORE:
            result["tree_status"][tree_id] = "IDENTITY_SCORE_BELOW_MINIMUM"
            continue
        if runner is not None and top["score"] - runner["score"] + _EPSILON < MIN_SCORE_MARGIN:
            result["tree_status"][tree_id] = "SCORE_MARGIN_BELOW_MINIMUM"
            for claim in absolute_candidates:
                candidate_id = claim["candidate_id"]
                if not conflicts[candidate_id] and claim["score"] >= MIN_IDENTITY_SCORE:
                    result["decisions"][candidate_id]["status"] = "INELIGIBLE_SCORE_MARGIN"
                    result["decisions"][candidate_id]["reason"] = "SCORE_MARGIN_BELOW_MINIMUM"
            continue
        top_id = top["candidate_id"]
        if conflicts[top_id]:
            result["tree_status"][tree_id] = "TOP_CANDIDATE_CONFLICT"
            result["decisions"][top_id]["status"] = "INELIGIBLE_CONFLICT"
            result["decisions"][top_id]["reason"] = "CONFLICTING_STEM"
            for claim in absolute_candidates[1:]:
                claim_id = claim["candidate_id"]
                if not conflicts[claim_id]:
                    result["decisions"][claim_id]["status"] = "INELIGIBLE_NOT_SELECTED"
                    result["decisions"][claim_id]["reason"] = "TOP_CANDIDATE_CONFLICT"
            continue
        result["selected"][tree_id] = top_id
        result["tree_status"][tree_id] = "SELECTED"
        result["decisions"][top_id]["status"] = "SELECTED"
        result["decisions"][top_id]["reason"] = "BEST_ABSOLUTE_CANDIDATE"
        for claim in absolute_candidates[1:]:
            claim_id = claim["candidate_id"]
            if conflicts[claim_id]:
                result["decisions"][claim_id]["status"] = "INELIGIBLE_CONFLICT"
                result["decisions"][claim_id]["reason"] = "CONFLICTING_STEM"
            else:
                result["decisions"][claim_id]["status"] = "INELIGIBLE_NOT_SELECTED"
                result["decisions"][claim_id]["reason"] = "LOWER_SCORE"
    return result


__all__ = [
    "AXIS_SHARED_ANGLE_DEG",
    "AXIS_SHARED_DISTANCE_M",
    "MIN_IDENTITY_SCORE",
    "MIN_SCORE_MARGIN",
    "resolve_stem_ownership",
]
