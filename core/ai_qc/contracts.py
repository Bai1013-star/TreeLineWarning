"""Strict advisory result contract for the optional tree AI-QC process."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

AI_QC_STATUSES = frozenset({"PASS", "REVIEW_REQUIRED", "UNCERTAIN"})
AI_QC_ISSUES = frozenset(
    {
        "NONE",
        "MERGED_MULTI_TREE",
        "OVER_SEGMENTED",
        "NEIGHBOR_CONTAMINATION",
        "CROWN_INCOMPLETE",
        "STEM_MISSING",
        "NOISE_CONTAMINATION",
        "OTHER",
    }
)
_MAX_REASON_CODES = 12


@dataclass(frozen=True, slots=True)
class AiQcResult:
    """Sanitized model classification; never authoritative tree data."""

    tree_id: str
    status: str
    issue: str
    confidence: float
    reason_codes: tuple[str, ...]
    parser_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


def _fallback(tree_id: str, reason: str) -> AiQcResult:
    return AiQcResult(
        tree_id=tree_id,
        status="UNCERTAIN",
        issue="OTHER",
        confidence=0.0,
        reason_codes=(reason,),
        parser_fallback=True,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.casefold().startswith("json\n"):
                stripped = stripped[5:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError("AI_QC_RESULT_NOT_OBJECT")
    return payload


def parse_ai_qc_result(text: str, *, expected_tree_id: str) -> AiQcResult:
    """Validate model text and fail closed to ``UNCERTAIN`` on any violation."""

    try:
        payload = _extract_json_object(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _fallback(expected_tree_id, "INVALID_JSON")
    if set(payload) != {"tree_id", "status", "issue", "confidence", "reason_codes"}:
        return _fallback(expected_tree_id, "INVALID_FIELDS")
    if payload["tree_id"] != expected_tree_id:
        return _fallback(expected_tree_id, "TREE_ID_MISMATCH")
    status = payload["status"]
    issue = payload["issue"]
    confidence = payload["confidence"]
    reasons = payload["reason_codes"]
    if status not in AI_QC_STATUSES or issue not in AI_QC_ISSUES:
        return _fallback(expected_tree_id, "UNSUPPORTED_LABEL")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return _fallback(expected_tree_id, "INVALID_CONFIDENCE")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return _fallback(expected_tree_id, "INVALID_CONFIDENCE")
    if not isinstance(reasons, list) or len(reasons) > _MAX_REASON_CODES:
        return _fallback(expected_tree_id, "INVALID_REASON_CODES")
    normalized: list[str] = []
    for reason in reasons:
        if not isinstance(reason, str):
            return _fallback(expected_tree_id, "INVALID_REASON_CODES")
        code = reason.strip().upper()
        if (
            not code
            or len(code) > 64
            or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in code)
        ):
            return _fallback(expected_tree_id, "INVALID_REASON_CODES")
        if code not in normalized:
            normalized.append(code)
    if status == "PASS" and issue != "NONE":
        return _fallback(expected_tree_id, "INCONSISTENT_PASS")
    if status != "PASS" and issue == "NONE":
        return _fallback(expected_tree_id, "INCONSISTENT_ISSUE")
    return AiQcResult(
        tree_id=expected_tree_id,
        status=status,
        issue=issue,
        confidence=confidence,
        reason_codes=tuple(normalized),
    )


def parse_ai_qc_result_v2(text: str, *, expected_tree_id: str) -> AiQcResult:
    """Validate the V2 four-field model contract and fail closed."""

    try:
        payload = _extract_json_object(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _fallback(expected_tree_id, "INVALID_JSON")
    if set(payload) != {"status", "issue", "confidence", "reason_codes"}:
        return _fallback(expected_tree_id, "INVALID_FIELDS")
    status = payload["status"]
    issue = payload["issue"]
    confidence = payload["confidence"]
    reasons = payload["reason_codes"]
    if status not in AI_QC_STATUSES or issue not in AI_QC_ISSUES:
        return _fallback(expected_tree_id, "UNSUPPORTED_LABEL")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return _fallback(expected_tree_id, "INVALID_CONFIDENCE")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return _fallback(expected_tree_id, "INVALID_CONFIDENCE")
    if not isinstance(reasons, list) or len(reasons) > _MAX_REASON_CODES:
        return _fallback(expected_tree_id, "INVALID_REASON_CODES")
    normalized: list[str] = []
    for reason in reasons:
        if not isinstance(reason, str):
            return _fallback(expected_tree_id, "INVALID_REASON_CODES")
        code = reason.strip().upper()
        if (
            not code
            or len(code) > 64
            or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in code)
        ):
            return _fallback(expected_tree_id, "INVALID_REASON_CODES")
        if code not in normalized:
            normalized.append(code)
    if status == "PASS" and issue != "NONE":
        return _fallback(expected_tree_id, "INCONSISTENT_PASS")
    if status != "PASS" and issue == "NONE":
        return _fallback(expected_tree_id, "INCONSISTENT_ISSUE")
    return AiQcResult(
        tree_id=expected_tree_id,
        status=status,
        issue=issue,
        confidence=confidence,
        reason_codes=tuple(normalized),
    )
