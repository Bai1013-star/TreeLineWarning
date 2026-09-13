"""Self-contained HTML export of the authoritative current final projection."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


def _text(value: Any, *, digits: int | None = None) -> str:
    if value is None or value == "":
        return "—"
    if digits is not None and isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def render_current_report_html(report: Mapping[str, Any]) -> str:
    """Render one immutable, offline-readable FINALIZED report."""

    if report.get("status") != "FINALIZED":
        raise ValueError("HTML_EXPORT_REQUIRES_CURRENT_FINALIZED_RESULT")
    kpis = report.get("kpis") if isinstance(report.get("kpis"), Mapping) else {}
    rows = report.get("trees") if isinstance(report.get("trees"), list) else []
    body = []
    for row in rows:
        source = row if isinstance(row, Mapping) else {}
        body.append(
            "<tr>"
            + "".join(
                f"<td>{value}</td>"
                for value in (
                    _text(source.get("display_tree_id") or source.get("persistent_tree_id")),
                    _text(source.get("persistent_tree_id")),
                    _text(source.get("height_m"), digits=2),
                    _text(source.get("crown_width_m"), digits=2),
                    _text(source.get("dbh_cm") or source.get("diameter_cm"), digits=2),
                    _text(source.get("minimum_clearance_m"), digits=3),
                    _text(source.get("horizontal_clearance_m"), digits=3),
                    _text(source.get("vertical_clearance_m"), digits=3),
                    _text(source.get("current_risk")),
                    "是" if source.get("human_edited") else "否",
                )
            )
            + "</tr>"
        )
    embedded = html.escape(json.dumps(dict(report), ensure_ascii=False, default=str))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TreeLineWarning 当前最终报告</title><style>
body{{font-family:"Microsoft YaHei",Arial,sans-serif;margin:32px;color:#15202b;
background:#f5f7fa}}h1{{margin-bottom:4px}}
.meta{{color:#52606d}}.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}}
.kpi{{background:white;border:1px solid #d9e2ec;border-radius:8px;padding:14px 18px}}
.kpi b{{display:block;font-size:24px}}table{{border-collapse:collapse;width:100%;
background:white}}th,td{{border:1px solid #d9e2ec;padding:8px;text-align:left}}
th{{background:#e9f2f9}}small{{color:#52606d}}
@media print{{body{{margin:10mm;background:white}}.kpi{{break-inside:avoid}}}}
</style></head><body><h1>TreeLineWarning 当前最终报告</h1>
<p class="meta">FINALIZED · 数据源：EDITOR_FINAL_PROJECTION · 本文件可离线打开</p>
<div class="kpis"><div class="kpi">最终全局单木
<b>{_text(kpis.get('tree_count'), digits=0)}</b></div>
<div class="kpi">全局最小净空 (m)<b>{_text(kpis.get('minimum_clearance_m'), digits=3)}</b></div>
<div class="kpi">预警阈值 (m)<b>{_text(kpis.get('warning_threshold_m'), digits=2)}</b></div></div>
<table><thead><tr><th>树编号</th><th>持久 ID</th><th>高度 (m)</th><th>冠幅 (m)</th>
<th>胸径 (cm)</th><th>最小净空 (m)</th><th>水平分量 (m)</th><th>垂直分量 (m)</th>
<th>风险</th><th>人工编辑</th></tr></thead>
<tbody>{''.join(body) or '<tr><td colspan="10">没有当前最终单木记录</td></tr>'}</tbody></table>
<details><summary>机器可读快照</summary><pre>{embedded}</pre></details></body></html>"""


def write_current_report_html(workflow: Any, output_path: Path | str) -> Path:
    report = workflow.current_report()
    rendered = render_current_report_html(report)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


__all__ = ["render_current_report_html", "write_current_report_html"]
