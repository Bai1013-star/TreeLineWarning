from __future__ import annotations

import json

from core.ai_qc.contracts import parse_ai_qc_result, parse_ai_qc_result_v2


def _payload(**updates: object) -> str:
    value: dict[str, object] = {
        "tree_id": "T-1-007",
        "status": "REVIEW_REQUIRED",
        "issue": "MERGED_MULTI_TREE",
        "confidence": 0.86,
        "reason_codes": ["MULTIPLE_STEM_LIKE_STRUCTURES", "MULTIPLE_CROWN_CENTERS"],
    }
    value.update(updates)
    return json.dumps(value)


def test_strict_result_parser_accepts_supported_contract() -> None:
    result = parse_ai_qc_result(_payload(), expected_tree_id="T-1-007")

    assert result.status == "REVIEW_REQUIRED"
    assert result.issue == "MERGED_MULTI_TREE"
    assert result.confidence == 0.86
    assert result.parser_fallback is False


def test_invalid_json_fails_closed_to_uncertain() -> None:
    result = parse_ai_qc_result("not json", expected_tree_id="T-1-007")

    assert result.status == "UNCERTAIN"
    assert result.issue == "OTHER"
    assert result.reason_codes == ("INVALID_JSON",)
    assert result.parser_fallback is True


def test_unsupported_label_fails_closed_to_uncertain() -> None:
    result = parse_ai_qc_result(_payload(issue="SPECIES_WRONG"), expected_tree_id="T-1-007")

    assert result.status == "UNCERTAIN"
    assert result.reason_codes == ("UNSUPPORTED_LABEL",)


def test_inconsistent_pass_fails_closed_to_uncertain() -> None:
    result = parse_ai_qc_result(
        _payload(status="PASS", issue="OVER_SEGMENTED"), expected_tree_id="T-1-007"
    )

    assert result.status == "UNCERTAIN"
    assert result.reason_codes == ("INCONSISTENT_PASS",)


def test_v2_four_field_contract_accepts_supported_result() -> None:
    payload = {
        "status": "REVIEW_REQUIRED",
        "issue": "NEIGHBOR_CONTAMINATION",
        "confidence": 0.79,
        "reason_codes": ["CONTEXT_OVERLAP"],
    }

    result = parse_ai_qc_result_v2(json.dumps(payload), expected_tree_id="T-1-007")

    assert result.tree_id == "T-1-007"
    assert result.status == "REVIEW_REQUIRED"
    assert result.parser_fallback is False


def test_v2_rejects_legacy_tree_id_field() -> None:
    result = parse_ai_qc_result_v2(_payload(), expected_tree_id="T-1-007")

    assert result.status == "UNCERTAIN"
    assert result.reason_codes == ("INVALID_FIELDS",)
