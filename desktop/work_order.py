"""Printable work-order projection for desktop prediction results."""
# ruff: noqa: E501

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any


def _text(value: Any, fallback: str = "—") -> str:
    if value is None or value == "":
        return fallback
    return escape(str(value))


def _metric(value: Any) -> str:
    return "—" if value is None else _text(f"{float(value):.4f} m")


def _forecast_rows(items: list[dict[str, Any]] | None, *, long_term: bool) -> str:
    rows = []
    for item in items or []:
        period = item.get("horizon_months") if long_term else item.get("period")
        label = f"{period}个月" if long_term else _text(period)
        rows.append(
            "<tr>"
            f"<td>{label}</td><td>{_metric(item.get('predicted_height'))}</td>"
            f"<td>{_metric(item.get('predicted_crown_width'))}</td>"
            f"<td>{_metric(item.get('predicted_clearance_distance'))}</td>"
            f"<td>{_text(item.get('risk_level'))}</td></tr>"
        )
    return "".join(rows) or "<tr><td colspan='5'>未提供</td></tr>"


def render_work_order(payload: dict[str, Any]) -> str:
    """Render a standalone A4 HTML document; only its source is printable."""
    tree, entry, span = payload["tree"], payload["entry"], payload["span"]
    order = entry.get("work_order") or {}
    current = entry.get("current_science") or {}
    assessment = entry.get("current_assessment") or {}
    timeline = entry.get("risk_timeline_analysis") or {}
    decision = entry.get("maintenance_decision") or {}
    advice = entry.get("treatment_advice") or {}
    project = payload["project"]
    generated = order.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    work_order_id = order.get("work_order_id") or payload["work_order_id"]
    mode = entry.get("model_label") or entry.get("prediction_mode")
    status = "待复测" if tree.get("dangerous") else "待执行"
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>TreeLineWarning 风险工单</title><style>
@page {{ size:A4; margin:14mm }} *{{box-sizing:border-box}} body{{font-family:'Microsoft YaHei',Arial,sans-serif;color:#111;background:#fff;font-size:11px;line-height:1.5;margin:0}} h1{{font-size:19px;text-align:center;margin:0}} h2{{font-size:13px;border-left:4px solid #111;padding-left:6px;margin:15px 0 6px}} .subtitle{{text-align:center;margin:2px 0 12px}} table{{width:100%;border-collapse:collapse}} td,th{{border:1px solid #222;padding:4px;vertical-align:top}} th{{background:#eee}} .label{{width:18%;background:#f5f5f5;font-weight:700}} .notice{{font-weight:700}} .screen-only{{margin:12px 0;text-align:center}} button{{padding:7px 14px}} @media print{{.screen-only{{display:none}}}}
</style></head><body><div class='screen-only'><button onclick='window.print()'>打印 / 保存为 PDF</button></div>
<h1>TreeLineWarning</h1><div class='subtitle'>树线隐患智能预警处置工单</div>
<h2>工单信息</h2><table><tr><td class='label'>工单编号</td><td>{_text(work_order_id)}</td><td class='label'>生成时间</td><td>{_text(generated)}</td></tr><tr><td class='label'>项目名称</td><td>{_text(project["project_name"])}</td><td class='label'>Group / Span</td><td>{_text(span["group_id"])} / {_text(span["span_id"])}（{_text(span["from_tower_id"])} → {_text(span["to_tower_id"])}）</td></tr><tr><td class='label'>Tree显示编号</td><td>{_text(tree.get("display_id"))}</td><td class='label'>persistent_tree_id</td><td>{_text(tree.get("persistent_tree_id"))}</td></tr></table>
<h2>当前检测</h2><table><tr><td class='label'>当前树高</td><td>{_metric(tree.get("apex_height_m"))}</td><td class='label'>当前最小净空</td><td>{_metric(current.get("minimum_clearance_m"))} <span class='notice'>（MAIN SCIENCE）</span></td></tr><tr><td class='label'>当前危险状态</td><td>{"危险" if tree.get("dangerous") else "未触发危险"}</td><td class='label'>危险点数量 / 当前风险</td><td>{_text(current.get("danger_point_count"))} / {_text(assessment.get("risk_level") or assessment.get("risk_grade"))}</td></tr></table>
<h2>预测方式</h2><p>{_text(mode)}{"（AI：时序注意力预测（合成数据基线））" if entry.get("prediction_mode") == "ai" else ""}</p>
<h2>短期预测</h2><table><tr><th>周期</th><th>预测树高</th><th>预测冠幅</th><th>预测净空（PREDICTION LAYER）</th><th>预测风险</th></tr>{_forecast_rows(entry.get("short_term_forecast"), long_term=False)}</table>
<h2>长期预测</h2><table><tr><th>周期</th><th>预测树高</th><th>预测冠幅</th><th>预测净空（PREDICTION LAYER）</th><th>预测风险</th></tr>{_forecast_rows(entry.get("long_term_forecast"), long_term=True)}</table>
<h2>风险趋势</h2><table><tr><td class='label'>风险首次明显升高时间</td><td>{_text(timeline.get("first_significant_risk_increase") or timeline.get("first_warning_month"))}</td><td class='label'>风险峰值月份</td><td>{_text(timeline.get("peak_risk_month") or timeline.get("peak_month"))}</td></tr><tr><td class='label'>净空变化趋势</td><td colspan='3'>{_text(timeline.get("clearance_trend") or timeline.get("summary"))}</td></tr></table>
<h2>处置建议</h2><table><tr><td class='label'>建议措施</td><td>{_text(decision.get("recommendation") or advice.get("recommendation") or advice.get("treatment_measure"))}</td></tr><tr><td class='label'>建议完成时间</td><td>{_text(decision.get("recommended_completion_time") or order.get("urgency"))}</td></tr><tr><td class='label'>修剪/处置依据</td><td>{_text(decision.get("basis") or advice.get("reason") or assessment.get("recommendation"))}</td></tr><tr><td class='label'>复测要求</td><td>{_text(advice.get("reinspection_requirement") or decision.get("reinspection_requirement") or "处置后复测当前净空（MAIN SCIENCE）。")}</td></tr></table>
<h2>数据质量与状态</h2><table><tr><td class='label'>树种</td><td>{_text(entry.get("species_fallback") or "未识别 / 其他（模型回退）")}</td><td class='label'>数据质量</td><td>{_text((entry.get("data_quality") or {}).get("status") or entry.get("input_quality"))}</td></tr><tr><td class='label'>状态</td><td colspan='3'>{status}</td></tr></table>
</body></html>"""
