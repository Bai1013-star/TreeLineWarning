from __future__ import annotations

import json
from pathlib import Path


def test_v2_reference_manifest_has_balanced_trustworthy_categories() -> None:
    path = Path(__file__).parent / "fixtures" / "reference_manifest_v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest["cases"]

    assert sum(case["reference_category"] == "EXPLICIT_GOOD" for case in cases) >= 8
    assert sum(case["reference_category"] == "EXPLICIT_BAD_VISUAL" for case in cases) >= 8
    assert any(
        case["reference_category"] == "EDITED_BUT_VISUAL_LABEL_UNCERTAIN" for case in cases
    )
    for case in cases:
        assert {
            "tree_id",
            "dataset",
            "pre_edit_revision",
            "post_edit_revision",
            "reference_category",
            "reference_issue",
            "evidence_source",
            "why_label_is_defensible",
        } <= set(case)
