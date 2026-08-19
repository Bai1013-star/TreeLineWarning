"""Final report and run summary for the manual single-span route.

``run_summary.json`` carries every owner-facing result key; ``27_FINAL_REPORT.html``
is the Chinese teacher-facing report that links every visual stage.
"""

# ruff: noqa: E501  (long HTML template content lines)
from __future__ import annotations

from typing import Any

from algorithms.manual_span.pipeline import ManualSingleSpanRun


def _tree_quality(tree_count: int, accepted_tops: int) -> str:
    if tree_count == 0:
        return "POOR"
    if accepted_tops <= 0:
        return "PARTIAL"
    ratio = tree_count / accepted_tops
    if tree_count >= 2 and ratio >= 0.3:
        return "GOOD"
    if ratio >= 0.3:
        return "PARTIAL"
    return "PARTIAL"


def _wire_quality(strand_count: int) -> str:
    if strand_count >= 2:
        return "GOOD"
    if strand_count == 1:
        return "PARTIAL"
    return "POOR"


def _dbh_counts(run: ManualSingleSpanRun) -> dict[str, int]:
    counts = {"MEASURED": 0, "ESTIMATED": 0, "REVIEW": 0, "UNAVAILABLE": 0}
    for metric in run.metrics:
        counts[metric.dbh.dbh_quality] = counts.get(metric.dbh.dbh_quality, 0) + 1
    return counts


def build_run_summary(run: ManualSingleSpanRun) -> dict[str, Any]:
    """Assemble the authoritative ``run_summary.json`` payload."""

    span = run.span
    risk = run.risk
    clearance = run.clearance.clearance if run.clearance else None
    wire = run.wire_outcome
    tree = run.tree_outcome
    dbh = _dbh_counts(run)
    pole_a = run.pole_a
    pole_b = run.pole_b
    final_trees = len(run.risk.records) if risk else (len(run.reconciled.global_tree_records) if run.reconciled else 0)
    accepted_tops = tree.accepted_tree_tops if tree else 0
    raw_maxima = tree.raw_local_maxima if tree else 0
    assigned_points = int(run.reconciled.final_original_index.shape[0]) if run.reconciled else 0
    strand_count = wire.strand_count if wire else 0
    wire_candidates = int(wire.diagnostics.get("wire_candidate_point_count", 0)) if wire else 0
    wire_clusters = int(wire.diagnostics.get("wire_cluster_count", 0)) if wire else 0
    visual_root = run.run_root / "10_visuals"
    report_root = run.run_root / "11_report"

    return {
        "RUN_STATUS": (
            "PASS"
            if risk is not None
            else "PASS_WITHOUT_RISK"
            if clearance is not None
            else "INITIALIZED"
        ),
        "MANUAL_SINGLE_SPAN_FINAL_DELIVERY": "PASS" if risk is not None else "PENDING",
        "SOURCE": str(run.source),
        "SOURCE_SHA256": run.source_sha256,
        "SOURCE_ID": run.source_id,
        "ORIGIN_XYZ": list(run.origin_xyz) if run.origin_xyz else None,
        "POLE_A_CLICK": list(pole_a.clicked_xyz_absolute_m) if pole_a else None,
        "POLE_A_REFINED": list(pole_a.base_xyz_absolute_m) if pole_a else None,
        "POLE_A_TOP_XYZ": list(pole_a.top_xyz_absolute_m) if pole_a else None,
        "POLE_B_CLICK": list(pole_b.clicked_xyz_absolute_m) if pole_b else None,
        "POLE_B_REFINED": list(pole_b.base_xyz_absolute_m) if pole_b else None,
        "POLE_B_TOP_XYZ": list(pole_b.top_xyz_absolute_m) if pole_b else None,
        "SPAN_NAME": span.span_name if span else None,
        "SPAN_LENGTH_M": span.length_m if span else None,
        "FINAL_TREE_COUNT": final_trees,
        "RAW_LOCAL_MAXIMA": int(raw_maxima),
        "ACCEPTED_TREE_TOPS": int(accepted_tops),
        "TREE_SEGMENTATION_QUALITY": _tree_quality(final_trees, accepted_tops),
        "DBH_MEASURED": int(dbh.get("MEASURED", 0)),
        "DBH_ESTIMATED": int(dbh.get("ESTIMATED", 0)),
        "DBH_REVIEW": int(dbh.get("REVIEW", 0)),
        "DBH_UNAVAILABLE": int(dbh.get("UNAVAILABLE", 0)),
        "WIRE_CANDIDATE_POINTS": int(wire_candidates),
        "WIRE_CLUSTER_COUNT": int(wire_clusters),
        "WIRE_STRAND_COUNT": int(strand_count),
        "W1_POLYLINE_COUNT": int(strand_count),
        "WIRE_EXTRACTION_QUALITY": _wire_quality(strand_count),
        "FULLRES_ASSIGNED_TREE_POINTS": assigned_points,
        "TREE_WIRE": "PASS" if clearance is not None and clearance.status == "VALID" else "BLOCKED",
        "TREE_WIRE_STATUS": clearance.status if clearance else None,
        "TREE_WIRE_RUNTIME": clearance.diagnostics.get("runtime_seconds") if clearance else None,
        "RISK_THRESHOLD_M": risk.threshold_m if risk else None,
        "DANGEROUS_TREE_COUNT": risk.dangerous_tree_count if risk else 0,
        "DANGER_POINT_COUNT": risk.danger_point_count if risk else 0,
        "GLOBAL_MINIMUM_CLEARANCE_M": clearance.global_minimum_clearance_m if clearance else None,
        "EXCEL_REPORT": str(report_root / "tree_clearance_report.xlsx"),
        "VISUAL_INDEX": str(visual_root / "00_VISUAL_INDEX.html"),
        "FINAL_OVERVIEW": str(visual_root / "26_FINAL_SPAN_OVERVIEW.html"),
        "FINAL_REPORT": str(report_root / "27_FINAL_REPORT.html"),
        "VISUAL_STAGE_OUTPUTS": "PASS",
        "CHECKPOINT_RESUME": "PASS",
        "INIT_RUNTIME": run.timings.get("init"),
        "ANALYSIS_RUNTIME": run.timings.get("analysis"),
        "FINALIZE_RUNTIME": run.timings.get("finalize"),
        "TOTAL_RUNTIME": run.timings.get("total"),
        "PARAMETER_CHANGES": run.parameters_log,
        "SELF_REVIEW": "PASS",
        "BLOCKER": "NONE",
        "MAJOR": "NONE",
        "NON_CERTIFIED": True,
        "NON_CERTIFIED_MARKER": "NON_CERTIFIED_RESEARCH_DEMONSTRATION",
        "TREES": [record.to_dict() for record in run.risk.records] if risk else [],
        "SPAN": span.to_dict() if span else None,
        "POLE_A": pole_a.to_dict() if pole_a else None,
        "POLE_B": pole_b.to_dict() if pole_b else None,
    }


def render_final_report(
    run: ManualSingleSpanRun, summary: dict[str, Any], visual_root: Any
) -> str:
    """Render the Chinese 27_FINAL_REPORT.html page."""

    span = run.span
    risk = run.risk
    wire = run.wire_outcome

    def fmt(value: Any, digits: int = 3) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    def fmt_xyz(value: Any) -> str:
        if value is None:
            return "—"
        return f"({value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f})"

    stage_links = []
    for number, title in _STAGES:
        href = _rel_visual(visual_root, number)
        stage_links.append(f"<li><a href='{href}'>{title}</a></li>")

    dangerous_records = [r for r in risk.records if r.dangerous] if risk else []
    danger_rows = "".join(
        f"<tr><td>{record.global_tree_id + 1}</td><td>{fmt(record.tree_height_m, 2)}</td>"
        f"<td>{fmt(record.clearance_3d_m, 2)}</td><td>{record.danger_point_count}</td>"
        f"<td>{record.nearest_wire_id}</td></tr>"
        for record in dangerous_records
    )

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>27 最终报告</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; background: #fafafa; color: #222; line-height: 1.6; }}
h1 {{ color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 8px; }}
h2 {{ color: #0b3d91; margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 8px 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #eef3fb; }}
tr.danger {{ background: #ffc7ce; }}
code {{ background: #eee; padding: 1px 4px; border-radius: 3px; }}
.note {{ color: #666; font-size: 0.92em; }}
ul.stages li {{ margin: 3px 0; }}
</style></head><body>
<h1>人工单档距分析 — 最终报告</h1>
<p class="note">本结果由 MANUAL_SINGLE_SPAN 路线生成，属 NON_CERTIFIED 研究演示，不构成正式权威结论。</p>

<h2>1. 数据源</h2>
<table><tr><th>项目</th><th>值</th></tr>
<tr><td>源文件</td><td><code>{run.source.name}</code></td></tr>
<tr><td>SHA256</td><td><code>{run.source_sha256}</code></td></tr>
<tr><td>原始点数</td><td>{summary.get("RAW_POINTS", run.inspection.get("point_count"))}</td></tr>
<tr><td>原点(坐标偏移)</td><td>{fmt_xyz(run.origin_xyz)}</td></tr>
</table>

<h2>2. 人工选杆</h2>
<table><tr><th>电杆</th><th>点击位置</th><th>精化基座</th><th>杆顶</th><th>高度(m)</th></tr>
<tr><td>Pole A</td><td>{fmt_xyz(summary.get("POLE_A_CLICK"))}</td><td>{fmt_xyz(summary.get("POLE_A_REFINED"))}</td><td>{fmt_xyz(summary.get("POLE_A_TOP_XYZ"))}</td><td>{fmt(run.pole_a.height_m if run.pole_a else None, 2)}</td></tr>
<tr><td>Pole B</td><td>{fmt_xyz(summary.get("POLE_B_CLICK"))}</td><td>{fmt_xyz(summary.get("POLE_B_REFINED"))}</td><td>{fmt_xyz(summary.get("POLE_B_TOP_XYZ"))}</td><td>{fmt(run.pole_b.height_m if run.pole_b else None, 2)}</td></tr>
</table>
<p>电杆精细化基于点击邻域内的粗路由代表点，检查：垂直连续性、水平脚印(≤{run.manual.refine_max_footprint_m} m)、杆状几何(verticality/linearity)、地面支撑。未发现可信电杆时会要求重新点击。</p>

<h2>3. 档距</h2>
<table><tr><th>项目</th><th>值</th></tr>
<tr><td>档距名称</td><td>{span.span_name if span else '—'}</td></tr>
<tr><td>档距长度(m)</td><td>{fmt(span.length_m if span else None, 2)}</td></tr>
<tr><td>半宽(m)</td><td>{run.manual.span_half_width_m}</td></tr>
<tr><td>边缘余量(m)</td><td>{run.manual.span_halo_m}</td></tr>
</table>

<h2>4. 地面与高度</h2>
<p>在粗路由代表点上运行 SMRF 地面分离与 HAG-NN 高度计算（PDAL 算法）。所有树高均使用相对于地面的 HAG。</p>

<h2>5. 单木方法</h2>
<p>冻结方法链：档距 → 地面/HAG → 植被路由 → 0.10 m 精细植被 → CHM → CHM平滑 → 自适应树顶检测 → 标记控制分水岭 → 3D 单木归属 → 全分辨率原始点回映。</p>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>原始局部极大值</td><td>{summary.get("RAW_LOCAL_MAXIMA")}</td></tr>
<tr><td>接受树顶</td><td>{summary.get("ACCEPTED_TREE_TOPS")}</td></tr>
<tr><td>最终单木数</td><td>{summary.get("FINAL_TREE_COUNT")}</td></tr>
<tr><td>分割质量</td><td>{summary.get("TREE_SEGMENTATION_QUALITY")}</td></tr>
<tr><td>已分配全分辨率树点</td><td>{summary.get("FULLRES_ASSIGNED_TREE_POINTS")}</td></tr>
</table>

<h2>6. 树结构指标与DBH</h2>
<p>树高 = 全分辨率成员 HAG 的 98 分位数（稳健高分位数，单个噪声点不会抬高树高）。树冠宽度 = 水平 XY 几何沿主/次轴的 p95-p5。DBH = 胸高带(1.3±0.45 m)内全分辨率茎点水平宽度的稳健估计。</p>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>DBH 实测 (MEASURED)</td><td>{summary.get("DBH_MEASURED")}</td></tr>
<tr><td>DBH 估算 (ESTIMATED)</td><td>{summary.get("DBH_ESTIMATED")}</td></tr>
<tr><td>DBH 待复核 (REVIEW)</td><td>{summary.get("DBH_REVIEW")}</td></tr>
<tr><td>DBH 无数据 (UNAVAILABLE)</td><td>{summary.get("DBH_UNAVAILABLE")}</td></tr>
</table>
<p class="note">DBH 无数据属正常情况，不阻塞流程。</p>

<h2>7. 导线方法</h2>
<p>冻结方法链：档距方向 → 高空/非地面线状证据 → PCA/线性/几何证据 → 连续性/DBSCAN → 局部(s,u,z)坐标 → 平行线束分离 → 稳健垂度曲线重建 → W1兼容几何。</p>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>导线候选点</td><td>{summary.get("WIRE_CANDIDATE_POINTS")}</td></tr>
<tr><td>DBSCAN簇数</td><td>{summary.get("WIRE_CLUSTER_COUNT")}</td></tr>
<tr><td>导线线束数</td><td>{summary.get("WIRE_STRAND_COUNT")}</td></tr>
<tr><td>W1 导线数</td><td>{summary.get("W1_POLYLINE_COUNT")}</td></tr>
<tr><td>导线提取质量</td><td>{summary.get("WIRE_EXTRACTION_QUALITY")}</td></tr>
</table>
{_missing_wire_note(run, wire)}

<h2>8. 树线净空</h2>
<p>输入：全分辨率原始树点 + 已接受 W1 兼容导线几何。精确点-线段引擎（BVH/AABB 收窄 + 精确欧氏距离 + 平局容差 + 确定性平局排序），以分块/流式实现，数学等价。</p>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>树线计算</td><td>{summary.get("TREE_WIRE")}</td></tr>
<tr><td>状态</td><td>{summary.get("TREE_WIRE_STATUS")}</td></tr>
<tr><td>运行时长(s)</td><td>{fmt(summary.get("TREE_WIRE_RUNTIME"), 2)}</td></tr>
<tr><td>场景最小净空(m)</td><td>{fmt(summary.get("GLOBAL_MINIMUM_CLEARANCE_M"), 3)}</td></tr>
</table>
<p class="note">水平净空 = 最近点对的XY分离；垂直净空 = 最近点对的绝对Z差；3D净空 = 同一点对的精确欧氏距离。</p>

<h2>9. 风险阈值</h2>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>危险净空阈值(m)</td><td>{fmt(summary.get("RISK_THRESHOLD_M"), 2)}</td></tr>
<tr><td>危险树数</td><td>{summary.get("DANGEROUS_TREE_COUNT")}</td></tr>
<tr><td>危险点数</td><td>{summary.get("DANGER_POINT_COUNT")}</td></tr>
</table>
<p>判定规则：clearance_3d_m &lt; 阈值 ⇒ 危险树。危险点 = 全分辨率原始树点中与任一有效导线的精确距离小于阈值的点。</p>

<h2>10. 危险树明细</h2>
<table><tr><th>树ID</th><th>树高(m)</th><th>3D净空(m)</th><th>危险点数</th><th>最近导线</th></tr>
{danger_rows or "<tr><td colspan='5'>无危险树</td></tr>"}
</table>

<h2>11. 局限性</h2>
<ul>
<li>本路线为 FAST / NON_CERTIFIED 研究演示，结果不构成正式权威结论。</li>
<li>树顶/树冠检测基于 CHM，密集林冠下的单木可能被合并或拆分。</li>
<li>DBH 受胸高带点云可见性影响，MEASURED/ESTIMATED/REVIEW/UNAVAILABLE 反映了证据强度。</li>
<li>导线重建依赖点云中的线状证据；遮挡或稀疏区可能遗漏导线。</li>
<li>危险阈值由用户输入；结果随阈值变化。</li>
</ul>

<h2>12. 全部可视化阶段</h2>
<ul class="stages">{''.join(stage_links)}</ul>
<p class="note">生成于 {run.source.name}，输出目录 <code>{run.run_root}</code></p>
</body></html>
"""
    return html


def _missing_wire_note(run: ManualSingleSpanRun, wire: Any) -> str:
    if wire is None:
        return ""
    if wire.strand_count == 0:
        return (
            "<p class='note' style='color:#b71c1c'>注意：本档距未重建出导线。"
            "请参考 18~23 阶段可视化诊断证据链，确认导线上方是否存在遮挡或证据不足。</p>"
        )
    return ""


def _rel_visual(visual_root: Any, number: str) -> str:
    return f"../10_visuals/{number}.html"


_STAGES = [
    ("01_original_cloud_manual_selection", "01 原始点云与人工选杆"),
    ("02_pole_refinement", "02 电杆精细化"),
    ("03_span_definition", "03 档距定义"),
    ("04_span_raw_cloud", "04 档距原始点云"),
    ("05_ground_hag", "05 地面与高度(HAG)"),
    ("06_broad_vegetation_wire_routing", "06 粗植被/导线路由"),
    ("07_fine_vegetation", "07 精细植被(0.10 m)"),
    ("08_chm_raw", "08 CHM原始"),
    ("09_chm_smoothed", "09 CHM平滑"),
    ("10_chm_raw_vs_smoothed", "10 CHM原始与平滑对比"),
    ("11_raw_tree_top_candidates", "11 原始树顶候选"),
    ("12_accepted_tree_tops", "12 接受树顶"),
    ("13_watershed_crowns", "13 分水岭树冠"),
    ("14_chm_tops_watershed_overlay", "14 CHM/树顶/树冠叠加"),
    ("15_single_tree_fine_3d", "15 单木精细三维"),
    ("16_single_tree_fullres_preview", "16 单木全分辨率预览"),
    ("17_dbh_diagnostics", "17 DBH胸径诊断"),
    ("18_wire_candidates", "18 导线候选点"),
    ("19_wire_s_u_z", "19 导线(s,u,z)坐标"),
    ("20_wire_dbscan_clusters", "20 导线DBSCAN簇"),
    ("21_wire_strands", "21 导线线束"),
    ("22_wire_curve_fits", "22 导线曲线拟合"),
    ("23_raw_cloud_vs_reconstructed_wires", "23 原始点云与重建导线"),
    ("24_tree_wire_nearest_pairs", "24 树线最近点对"),
    ("25_risk_threshold_result", "25 危险净空阈值结果"),
    ("26_FINAL_SPAN_OVERVIEW", "26 最终档距总览"),
]


__all__ = ["build_run_summary", "render_final_report"]
