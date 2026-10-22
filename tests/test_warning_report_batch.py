from __future__ import annotations

import json

import pytest

from desktop.warning_bridge import _save_latest_report


def test_latest_report_keeps_ordered_multi_tree_batch(tmp_path):
    _save_latest_report(
        tmp_path,
        {"tree": "T-1-001"},
        batch_id="batch-a",
        batch_index=0,
        batch_count=2,
    )
    _save_latest_report(
        tmp_path,
        {"tree": "T-1-002"},
        batch_id="batch-a",
        batch_index=1,
        batch_count=2,
    )

    saved = json.loads(
        (tmp_path / "runtime" / "warning" / "latest_report.json").read_text("utf-8")
    )
    assert saved["schema"] == "treeline-latest-report-v2"
    assert [item["tree"] for item in saved["reports"]] == ["T-1-001", "T-1-002"]
    assert saved["report"] == saved["reports"][0]


def test_latest_report_rejects_invalid_batch_position(tmp_path):
    with pytest.raises(ValueError, match="批量报告序号"):
        _save_latest_report(
            tmp_path,
            {"tree": "bad"},
            batch_id="batch-a",
            batch_index=-1,
            batch_count=2,
        )
