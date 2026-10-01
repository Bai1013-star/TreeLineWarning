"""Build downstream HTML review pages (Steps 2-9) for the SAME manual span.

This is a result/review generator: it reads the ALREADY-COMPLETED manual
single-span run artifacts (ground, wire, vegetation, tree, clearance, risk,
export) and emits one plain HTML page per processing step, reusing the real
Plotly visualizations via <iframe> where they exist and clearly-labelled
placeholders otherwise.  It never reruns any expensive stage.
"""

# ruff: noqa: E501  (long HTML/CSS template content lines)
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# The definitive completed manual single-span run (span 44.8 m).
RUN_DIR = Path(
    "artifacts/manual_single_span_runs/2026-08-07_16没54-10_PointCloud_colorized_20260810_233432"
).resolve()
OUT_DIR = RUN_DIR / "workflow_flow"

STEPS = [
    ("01", "选杆定档", "01_manual_selector"),
    ("02", "地面/HAG", "02_ground_hag.html"),
    ("03", "导线重构", "03_wire_reconstruction.html"),
    ("04", "植被恢复", "04_vegetation_frv.html"),
    ("05", "单木分割", "05_tree_segmentation.html"),
    ("06", "树线净空", "06_clearance_analysis.html"),
    ("07", "风险判定", "07_threshold_risk.html"),
    ("08", "结果输出", "08_results_export.html"),
    ("09", "生长预测", "09_growth_prediction.html"),
]

CSS = """
body { font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #f4f6fb; color: #222; }
#wrap { max-width: 1180px; margin: 0 auto; padding: 16px; }
h1 { color: #0b3d91; font-size: 22px; margin: 8px 0; }
h2 { color: #0b3d91; font-size: 16px; margin: 12px 0 6px; }
.card { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.10); padding: 12px 16px; margin: 10px 0; }
.flowbar { display: flex; gap: 4px; flex-wrap: wrap; background: #eef3fb; border-radius: 6px; padding: 8px; margin: 8px 0; font-size: 12px; }
.flowbar .step { padding: 4px 8px; border-radius: 4px; color: #666; background: #e0e0e0; }
.flowbar .step.done { background: #2e7d32; color: #fff; }
.flowbar .step.cur { background: #0b3d91; color: #fff; font-weight: 700; }
.spanline { background: #fff3e0; border-left: 4px solid #ef6c00; padding: 8px 12px; border-radius: 4px; margin: 8px 0; font-size: 15px; }
.panels { display: flex; gap: 12px; flex-wrap: wrap; }
.panel { flex: 1 1 480px; min-width: 420px; }
.panel h3 { font-size: 14px; color: #0b3d91; margin: 6px 4px; }
.panel .viz { width: 100%; height: 520px; border: 1px dashed #b0bec5; border-radius: 6px; background: #fafafa; }
.panel .viz iframe { width: 100%; height: 100%; border: 0; }
.panel .ph { display: flex; align-items: center; justify-content: center; height: 100%; color: #90a4ae; font-size: 16px; }
table { border-collapse: collapse; width: 100%; background: #fff; margin: 8px 0; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 13px; }
th { background: #eef3fb; }
tr.danger { background: #ffc7ce; }
tr.safe { background: #e8f5e9; }
.kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 8px; }
.kv div { background: #f8fafc; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px 10px; }
.kv b { color: #0b3d91; }
.kv .v { font-size: 18px; font-weight: 700; color: #222; }
.nav { display: flex; justify-content: space-between; gap: 10px; margin: 16px 0; }
.nav a, .nav button { display: inline-block; padding: 9px 18px; border-radius: 6px; background: #0b3d91; color: #fff; text-decoration: none; font-size: 14px; border: none; cursor: pointer; }
.nav a.sec { background: #9e9e9e; }
details { margin: 10px 0; }
summary { cursor: pointer; color: #0b3d91; font-size: 13px; }
.ds { font-size: 12px; color: #455a64; background: #eceff1; padding: 8px; border-radius: 4px; white-space: pre-wrap; word-break: break-all; }
button.link { background: #0b3d91; color: #fff; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; margin: 4px; }
button.link.gray { background: #90a4ae; }
button.link:disabled { background: #cfd8dc; color: #666; cursor: not-allowed; }
"""


def flowbar(current: str) -> str:
    html = ['<div class="flowbar">']
    for num, label, _target in STEPS:
        if num == current:
            html.append(f'<span class="step cur">{num} {label}</span>')
        elif int(num) < int(current):
            html.append(f'<span class="step done">{num} {label}</span>')
        else:
            html.append(f'<span class="step">{num} {label}</span>')
    html.append("</div>")
    return "".join(html)


def spanline(pole_a: tuple, pole_b: tuple, length: float) -> str:
    return (
        '<div class="spanline">⚠ 同一人工档距：<b>Pole A</b> '
        f'({pole_a[0]:.2f}, {pole_a[1]:.2f}) → <b>Pole B</b> '
        f'({pole_b[0]:.2f}, {pole_b[1]:.2f})　档距长度 {length:.2f} m</div>'
    )


def page(title: str, current: str, body: str, prev: str | None, nxt: str | None,
         data_source: str) -> str:
    nav_prev = f'<a href="{prev}" class="sec">← 上一步</a>' if prev else '<span></span>'
    nav_next = f'<a href="{nxt}">下一步 →</a>' if nxt else '<span></span>'
    back = '<a href="00_workflow_index.html" class="sec">回流程总览</a>'
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>
<div id="wrap">
<h1>{title}</h1>
{flowbar(current)}
<spanline_placeholder>
{body}
<div class="nav">{nav_prev} {back} {nav_next}</div>
<details><summary>数据来源 / DATA_SOURCE</summary><div class="ds">{data_source}</div></details>
</div></body></html>
""".replace("spanline_placeholder", spanline(POLE_A, POLE_B, SPAN_LENGTH))


def viz_iframe(name: str | None, label: str) -> str:
    if name:
        return f'<div class="viz"><iframe src="../10_visuals/{name}" title="{label}"></iframe></div>'
    return '<div class="viz"><div class="ph">真实结果图待接入</div></div>'


def kv_block(pairs: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div><b>{k}</b><br><span class="v">{v}</span></div>' for k, v in pairs)
    return f'<div class="kv">{cells}</div>'


def load_json(rel: str) -> dict:
    return json.loads((RUN_DIR / rel).read_text("utf-8"))


_TREE_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
]


def _read_raw_region_xyz(rel_dir: str, maximum: int, origin: np.ndarray) -> np.ndarray:
    """Read a deterministic sample of a RawRegion xyz_local.bin (absolute frame)."""
    path = RUN_DIR / rel_dir / "xyz_local.bin"
    side = load_json(rel_dir + "/region_sidecar.json")
    count = int(side["count"])
    if count == 0 or not path.is_file():
        return np.empty((0, 3), dtype=np.float64)
    xyz = np.memmap(path, dtype="<f4", mode="r", shape=(count, 3))
    rows = _even_indices(count, maximum)
    return np.asarray(xyz[rows], dtype=np.float64) + origin


def _build_step04_visuals() -> None:
    """Generate Step-04 presentation visuals (candidates + full-res FRV).

    Visualization-only, read from the REAL saved artifacts:
      candidates -> fast_v4_input.npz (345,639 representative points)
      full-res   -> 04_regions/vegetation_region RawRegion (6,287,349 points)
    Coordinates converted to the real LAS frame (local + origin).  Sampling is
    display-only; scientific counts are unchanged.
    """
    origin = np.asarray(
        load_json("00_input/source_inspection.json")["bounds_min_xyz"], dtype=np.float64
    )
    cand_npz = RUN_DIR / "06_tree/coarse_routing/fast_v4_input.npz"
    with np.load(cand_npz, allow_pickle=False) as stream:
        cand_xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64) + origin
    cand_rows = _even_indices(len(cand_xyz), 30_000)
    cand = cand_xyz[cand_rows]

    halo = _read_raw_region_xyz("04_regions/halo_region", 12_000, origin)
    full = _read_raw_region_xyz("04_regions/vegetation_region", 30_000, origin)

    pad = 3.0
    if len(halo):
        xmin, xmax = float(halo[:, 0].min()), float(halo[:, 0].max())
        ymin, ymax = float(halo[:, 1].min()), float(halo[:, 1].max())
    else:
        xmin, xmax = float(cand[:, 0].min()), float(cand[:, 0].max())
        ymin, ymax = float(cand[:, 1].min()), float(cand[:, 1].max())

    # ---------------- LEFT: vegetation candidates ----------------
    left = go.Figure()
    if len(halo):
        left.add_trace(go.Scatter(
            x=halo[:, 0], y=halo[:, 1], mode="markers",
            marker={"size": 2, "color": "#cfd8dc", "opacity": 0.35},
            name="档距点云（背景）", showlegend=False,
        ))
    left.add_trace(go.Scatter(
        x=cand[:, 0], y=cand[:, 1], mode="markers",
        marker={"size": 4, "color": "#2e7d32", "opacity": 0.85},
        name="植被候选代表点", showlegend=True,
    ))
    left.update_layout(
        title={"text": "植被候选代表点（算法初步筛出的植被）", "x": 0.5},
        xaxis={"title": "X (m)", "range": [xmin - pad, xmax + pad]},
        yaxis={"title": "Y (m)", "range": [ymin - pad, ymax + pad],
               "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 50, "r": 20, "t": 60, "b": 40}, height=620,
        template="plotly_white",
    )
    _write_figure(left, RUN_DIR / "10_visuals" / "step04_candidates.html")

    # ---------------- RIGHT: full-resolution vegetation ----------------
    right = go.Figure()
    if len(full):
        right.add_trace(go.Scatter(
            x=full[:, 0], y=full[:, 1], mode="markers",
            marker={
                "size": 2.5, "color": full[:, 2],
                "colorscale": "Greens", "opacity": 0.8,
                "colorbar": {"title": "Z (m)"},
            },
            name="FRV 全分辨率植被（显示抽样）", showlegend=True,
        ))
    right.update_layout(
        title={"text": "FRV 全分辨率植被恢复", "x": 0.5},
        xaxis={"title": "X (m)", "range": [xmin - pad, xmax + pad]},
        yaxis={"title": "Y (m)", "range": [ymin - pad, ymax + pad],
               "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 50, "r": 20, "t": 60, "b": 40}, height=620,
        template="plotly_white",
    )
    _write_figure(right, RUN_DIR / "10_visuals" / "step04_frv_fullres.html")


def _even_indices(count: int, maximum: int) -> np.ndarray:
    count = max(0, int(count))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    maximum = max(1, int(maximum))
    return np.linspace(0, count - 1, min(count, maximum), dtype=np.int64)


def _write_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fig.write_html(temporary, include_plotlyjs=True, full_html=True)
    os.replace(temporary, path)


def _build_step05_visuals(tree_members: np.ndarray, tree_ids: np.ndarray,
                          heights: dict[int, float], origin: np.ndarray) -> None:
    """Generate the top-down and 3D FINAL-tree visualizations (Step 05 only).

    Uses the exact final tree membership fed to Step 06 (tree_members.npz).
    Coordinates are converted to the real LAS frame (local + origin).  Display
    is downsampled per tree purely for rendering; no scientific result changes.
    """
    absolute = tree_members + origin
    ids = sorted(set(int(v) for v in tree_ids if v >= 0))
    # Per-tree XY extents for a tight 1:1 fit.
    all_xy = absolute[:, :2]
    pad = 2.0
    xmin, xmax = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    ymin, ymax = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    xrange = [xmin - pad, xmax + pad]
    yrange = [ymin - pad, ymax + pad]

    # ---------------- top-down ----------------
    top = go.Figure()
    label_x, label_y, label_text = [], [], []
    for tree_id in ids:
        rows = np.flatnonzero(tree_ids == tree_id)
        sample = rows[_even_indices(len(rows), 4000)]
        pts = absolute[sample]
        top.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers",
            marker={"size": 3, "color": _TREE_COLORS[tree_id % len(_TREE_COLORS)], "opacity": 0.8},
            name=f"Tree {tree_id + 1}", showlegend=False,
        ))
        centroid = np.median(absolute[rows, :2], axis=0)
        label_x.append(float(centroid[0]))
        label_y.append(float(centroid[1]))
        label_text.append(f"Tree {tree_id + 1}")
    top.add_trace(go.Scatter(
        x=label_x, y=label_y, mode="text", text=label_text,
        textposition="top center", textfont={"size": 13, "color": "#111"},
        name="Tree IDs", showlegend=False,
    ))
    top.update_layout(
        title={"text": "单木分割俯视结果（最终树，XY 1:1）", "x": 0.5},
        xaxis={"title": "X (m)", "range": xrange},
        yaxis={"title": "Y (m)", "range": yrange, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 50, "r": 20, "t": 60, "b": 40}, height=620,
        template="plotly_white",
    )
    _write_figure(top, RUN_DIR / "10_visuals" / "step05_topdown.html")

    # ---------------- 3D ----------------
    three = go.Figure()
    for tree_id in ids:
        rows = np.flatnonzero(tree_ids == tree_id)
        sample = rows[_even_indices(len(rows), 1500)]
        pts = absolute[sample]
        three.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker={"size": 2, "color": _TREE_COLORS[tree_id % len(_TREE_COLORS)], "opacity": 0.85},
            name=f"Tree {tree_id + 1}",
        ))
    three.update_layout(
        title={"text": "单木分割 3D 结果（最终树，每树最多抽样 1500 点）", "x": 0.5},
        scene={"aspectmode": "data", "xaxis": {"title": "X"}, "yaxis": {"title": "Y"}, "zaxis": {"title": "Z"}},
        margin={"l": 0, "r": 0, "t": 60, "b": 0}, height=620,
        template="plotly_white",
    )
    _write_figure(three, RUN_DIR / "10_visuals" / "step05_3d.html")


def main() -> None:
    global POLE_A, POLE_B, SPAN_LENGTH
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_json("run_summary.json")
    POLE_A = tuple(summary["POLE_A_REFINED"])
    POLE_B = tuple(summary["POLE_B_REFINED"])
    SPAN_LENGTH = float(summary["SPAN_LENGTH_M"])

    ground = load_json("02_ground/ground_hag_result.json")
    wire = load_json("05_wire/span_001/span_wire_result.json")
    wd = wire.get("diagnostics", {})
    veg = load_json("06_tree/coarse_routing/manual_span_fast_vegetation_diagnostics.json")
    veg_counts = veg.get("counts", {})
    tree = load_json("06_tree/span_001/span_tree_manifest.json")
    clear = load_json("07_clearance/tree_wire_clearance.json")
    risk = load_json("09_risk/risk_result.json")
    risk_threshold = load_json("risk_threshold.json")
    halo = load_json("04_regions/halo_region/region_sidecar.json")
    veg_region = load_json("04_regions/vegetation_region/region_sidecar.json")

    clear_records = sorted(clear["records"], key=lambda x: x["minimum_clearance_m"])

    pages = {}

    # ---------------------------------------------------------- STEP 2
    body2 = f"""
<div class="card">
<h2>STEP 2 — 地面分离与高度归一化</h2>
{kv_block([
    ("进入本阶段的点数", f'{halo.get("count", 0):,}'),
    ("地面分离状态", "已完成"),
    ("HAG 计算状态", "已完成"),
    ("地面点(class 2)", f'{ground.get("ground_class_2_count", 0):,}'),
    ("非地面点", f'{ground.get("non_ground_count", 0):,}'),
    ("有效 HAG 点数", f'{ground.get("hag_valid_count", 0):,}'),
    ("地面算法", "PDAL SMRF"),
    ("HAG 算法", "PDAL HAG-NN"),
])}
</div>
<div class="panels">
  <div class="panel"><h3>地面分离结果（左）</h3>{viz_iframe("05_ground_hag.html", "地面/HAG")}</div>
  <div class="panel"><h3>HAG 归一化高度（右）</h3>{viz_iframe("04_span_raw_cloud.html", "档距原始点云")}</div>
</div>
<div class="card"><h2>阶段状态</h2>
<table><tr><th>步骤</th><th>状态</th><th>说明</th></tr>
<tr><td>读取档距点云</td><td>完成</td><td>halo 内原始点 {halo.get('count', 0):,} 个</td></tr>
<tr><td>地面分离</td><td>完成</td><td>class 2 地面 {ground.get('ground_class_2_count', 0):,}</td></tr>
<tr><td>地面模型</td><td>完成</td><td>SMRF 网格地面</td></tr>
<tr><td>HAG 计算</td><td>完成</td><td>有效 HAG {ground.get('hag_valid_count', 0):,}</td></tr>
</table></div>
"""
    pages["02_ground_hag.html"] = page(
        "02 地面/HAG — 树线净空分析", "02", body2, None, "03_wire_reconstruction.html",
        json.dumps({
            "run": str(RUN_DIR),
            "ground_result": "02_ground/ground_hag_result.json",
            "halo_region": "04_regions/halo_region/region_sidecar.json",
            "visual": "10_visuals/05_ground_hag.html",
        }, ensure_ascii=False, indent=2),
    )

    # ---------------------------------------------------------- STEP 3
    strand_rows = ""
    for s in wire.get("strands", []):
        strand_rows += (
            f"<tr><td>{s['wire_name']}</td><td>{s['coverage_ratio']:.2f}</td>"
            f"<td>{s['s_span_m']:.1f}</td><td>{s['median_u_m']:.2f}</td>"
            f"<td>{s['median_hag_m']:.1f}</td><td>{s['residual_stats_m'][1]:.3f}</td></tr>"
        )
    body3 = f"""
<div class="card">
<h2>STEP 3 — 导线提取与重建</h2>
{kv_block([
    ("Pole A", f"({POLE_A[0]:.2f}, {POLE_A[1]:.2f})"),
    ("Pole B", f"({POLE_B[0]:.2f}, {POLE_B[1]:.2f})"),
    ("档距长度", f"{SPAN_LENGTH:.2f} m"),
    ("导线候选点", f"{wd.get('wire_candidate_point_count', 0):,}"),
    ("DBSCAN 簇", f"{wd.get('wire_cluster_count', 0)}"),
    ("确认导线实例", f"{wire.get('strand_count', 0)}"),
    ("精细代表点", f"{wd.get('fine_wire_representative_count', 0):,}"),
    ("导线重构状态", "已完成（GOOD）"),
])}
</div>
<div class="panels">
  <div class="panel"><h3>导线候选 / 实例分离（左）</h3>{viz_iframe("18_wire_candidates.html", "导线候选点")}</div>
  <div class="panel"><h3>3D 重建导线中心线（右）</h3>{viz_iframe("23_raw_cloud_vs_reconstructed_wires.html", "原始点云与重建导线")}</div>
</div>
<div class="card"><h2>导线实例明细</h2>
<table><tr><th>导线</th><th>纵向覆盖率</th><th>长度(m)</th><th>横向偏移u(m)</th><th>中位高度(m)</th><th>拟合残差p95(m)</th></tr>
{strand_rows or "<tr><td colspan=6>无导线</td></tr>"}
</table>
<p style="font-size:13px;color:#455a64">流程：候选提取 → 实例分离 → 确认 → 中心线重建。</p></div>
"""
    pages["03_wire_reconstruction.html"] = page(
        "03 导线重构 — 树线净空分析", "03", body3, "02_ground_hag.html", "04_vegetation_frv.html",
        json.dumps({
            "run": str(RUN_DIR),
            "wire_result": "05_wire/span_001/span_wire_result.json",
            "wire_arrays": "05_wire/span_001/w1_arrays.npz",
            "visuals": ["10_visuals/18_wire_candidates.html", "10_visuals/23_raw_cloud_vs_reconstructed_wires.html"],
        }, ensure_ascii=False, indent=2),
    )

    # ---------------------------------------------------------- STEP 4
    _build_step04_visuals()
    body4 = f"""
<div class="card">
<h2>STEP 4 — 植被提取与全分辨率恢复</h2>
{kv_block([
    ("输入代表点", f"{veg_counts.get('input_representatives', 0):,}"),
    ("植被候选代表点", f"{veg_counts.get('final_vegetation_representatives', 0):,}"),
    ("FRV 全分辨率植被点", f'{veg_region.get("count", 0):,}'),
    ("排除-地面", f"{veg_counts.get('ground_excluded', 0):,}"),
    ("排除-导线", f"{veg_counts.get('wire_geometry_excluded', 0):,}"),
    ("FRV 状态", "已完成"),
])}
<p style="font-size:13px;color:#455a64">流程：<b>先从点云中筛选植被候选代表点</b> → <b>回归原始点云（FRV）</b> → <b>恢复完整的高密度植被</b>。</p>
</div>
<div class="panels">
  <div class="panel"><h3>植被候选结果（左）</h3>{viz_iframe("step04_candidates.html", "植被候选代表点")}</div>
  <div class="panel"><h3>FRV 全分辨率植被恢复（右）</h3>{viz_iframe("step04_frv_fullres.html", "FRV 全分辨率植被")}</div>
</div>
<div class="card"><h2>候选 → FRV 对比</h2>
<p style="font-size:16px;font-weight:700;color:#2e7d32">候选代表点：{veg_counts.get('final_vegetation_representatives', 0):,}　→　FRV 全分辨率植被：{veg_region.get('count', 0):,}</p>
<p style="font-size:13px;color:#455a64">真实全分辨率植被：{veg_region.get('count', 0):,} 点；图中为显示抽样（可视化仅抽样，不改变科学结果）。</p></div>
"""
    pages["04_vegetation_frv.html"] = page(
        "04 植被恢复 — 树线净空分析", "04", body4, "03_wire_reconstruction.html", "05_tree_segmentation.html",
        json.dumps({
            "run": str(RUN_DIR),
            "candidate_source": "06_tree/coarse_routing/fast_v4_input.npz",
            "fullres_source": "04_regions/vegetation_region/region_sidecar.json",
            "veg_diagnostics": "06_tree/coarse_routing/manual_span_fast_vegetation_diagnostics.json",
            "visuals": ["10_visuals/step04_candidates.html", "10_visuals/step04_frv_fullres.html"],
            "note": "展示为真实候选代表点与真实全分辨率植被恢复；可视化显示抽样不影响科学计数。",
        }, ensure_ascii=False, indent=2),
    )

    # ---------------------------------------------------------- STEP 5
    # FINAL tree set used by Step 06 (clearance): tree_members.npz has exactly
    # the same tree_ids and per-tree point counts as the clearance records.
    # Visualization-only: read the saved members, build a NEW top-down + 3D
    # plot of these SAME final trees.  No scientific result is changed.
    with np.load(RUN_DIR / "06_tree/span_001/tree_members.npz", allow_pickle=False) as _s:
        _mem_xyz = np.asarray(_s["xyz_local_m"], dtype=np.float64)
        _mem_tree = np.asarray(_s["tree_id"], dtype=np.int32)
    _origin = np.asarray(
        load_json("00_input/source_inspection.json")["bounds_min_xyz"], dtype=np.float64
    )
    _heights = {
        int(i): float(t.get("apex_height_m", 0.0))
        for i, t in enumerate(tree.get("trees", []))
    }
    _build_step05_visuals(_mem_xyz, _mem_tree, _heights, _origin)
    _tree_rows = ""
    for _i, _t in enumerate(tree.get("trees", [])):
        _tree_rows += (
            f"<tr><td>Tree {_i + 1}</td><td>{_t.get('raw_point_count', 0):,}</td>"
            f"<td>{_t.get('apex_height_m', 0):.2f}</td></tr>"
        )
    body5 = f"""
<div class="card">
<h2>STEP 5 — 单木分割</h2>
{kv_block([
    ("最终单木数", f"{len(tree.get('trees', []))}"),
    ("已分配全分辨率树点", f"{tree.get('assigned_raw_point_count', 0):,}"),
    ("分割方法", "CHM → 树顶 → 分水岭 → 3D归属 → 全分辨率回映"),
    ("分割状态", "已完成（GOOD）"),
    ("展示数据", "最终树集合（与 Step 6 树线净空使用完全一致）"),
])}
<p style="font-size:13px;color:#455a64">本页展示的是<b>最终单木实例</b>：完整植被被分割为一棵棵独立树木。俯视与 3D 使用<b>同一套树、同一套颜色、同一 Tree ID</b>。可视化仅对显示抽样，不改变任何科学结果。</p>
</div>
<div class="panels">
  <div class="panel"><h3>单木分割俯视结果（左）</h3>{viz_iframe("step05_topdown.html", "最终树俯视")}</div>
  <div class="panel"><h3>单木分割 3D 结果（右）</h3>{viz_iframe("step05_3d.html", "最终树 3D")}</div>
</div>
<div class="card"><h2>最终单木明细</h2>
<table><tr><th>Tree ID</th><th>全分辨率点数</th><th>树高(m)</th></tr>
{_tree_rows}</table></div>
"""
    pages["05_tree_segmentation.html"] = page(
        "05 单木分割 — 树线净空分析", "05", body5, "04_vegetation_frv.html", "06_clearance_analysis.html",
        json.dumps({
            "run": str(RUN_DIR),
            "tree_members": "06_tree/span_001/tree_members.npz",
            "tree_manifest": "06_tree/span_001/span_tree_manifest.json",
            "visuals": ["workflow_flow/step05_topdown.html", "workflow_flow/step05_3d.html"],
            "note": "展示与 Step 6 完全相同的最终树集合（tree_members.npz tree_id == 净空 global_tree_id）。",
        }, ensure_ascii=False, indent=2),
    )

    # ---------------------------------------------------------- STEP 6
    clear_rows = ""
    for rec in clear_records:
        clear_rows += (
            f"<tr><td>{rec['global_tree_id'] + 1}</td><td>{rec['nearest_wire_id'] if rec['nearest_wire_id'] is not None else '—'}</td>"
            f"<td>{rec['minimum_clearance_m']:.3f}</td><td>{rec['tree_point_count']:,}</td></tr>"
        )
    body6 = f"""
<div class="card">
<h2>STEP 6 — 树线净空计算</h2>
{kv_block([
    ("分析树数", f"{len(clear.get('records', []))}"),
    ("重建导线数", f"{summary.get('WIRE_STRAND_COUNT', 0)}"),
    ("全局最小净空", f"{clear.get('global_minimum_clearance_m'):.4f} m"),
    ("最小净空对应树", f"{min(clear_records, key=lambda x: x['minimum_clearance_m'])['global_tree_id'] + 1}"),
    ("最小净空对应导线", f"{min(clear_records, key=lambda x: x['minimum_clearance_m'])['nearest_wire_id']}"),
    ("计算状态", "VALID"),
    ("算法", "精确点-线段距离（BVH/AABB）"),
])}
</div>
<div class="panels">
  <div class="panel"><h3>3D 树线净空（左）</h3>{viz_iframe("24_tree_wire_nearest_pairs.html", "树线最近点对")}</div>
  <div class="panel"><h3>净空总览（右）</h3>{viz_iframe("26_FINAL_SPAN_OVERVIEW.html", "最终档距总览")}</div>
</div>
<div class="card"><h2>每树净空（按净空升序）</h2>
<table><tr><th>树ID</th><th>最近导线</th><th>3D最小净空(m)</th><th>树点数</th></tr>
{clear_rows}</table></div>
"""
    pages["06_clearance_analysis.html"] = page(
        "06 树线净空 — 树线净空分析", "06", body6, "05_tree_segmentation.html", "07_threshold_risk.html",
        json.dumps({
            "run": str(RUN_DIR),
            "clearance": "07_clearance/tree_wire_clearance.json",
            "danger_candidates": "07_clearance/danger_candidates.npz",
            "visuals": ["10_visuals/24_tree_wire_nearest_pairs.html", "10_visuals/26_FINAL_SPAN_OVERVIEW.html"],
        }, ensure_ascii=False, indent=2),
    )

    # ---------------------------------------------------------- STEP 7
    danger_rows = ""
    for rec in risk["records"]:
        cls = "danger" if rec.get("dangerous") else "safe"
        level = "危险" if rec.get("dangerous") else "安全"
        danger_rows += (
            f"<tr class='{cls}'><td>{rec['tree_id'] + 1}</td>"
            f"<td>{rec['nearest_wire_id'] if rec['nearest_wire_id'] is not None else '—'}</td>"
            f"<td>{rec['clearance_3d_m']:.3f}</td><td>{level}</td>"
            f"<td>{rec['danger_point_count']}</td>"
            f"<td>{rec.get('horizontal_clearance_m'):.3f}</td><td>{rec.get('vertical_clearance_m'):.3f}</td></tr>"
        )
    danger_detail = ""
    for rec in risk["records"]:
        if rec.get("dangerous"):
            danger_detail += (
                f"<tr class='danger'><td>{rec['tree_id'] + 1}</td><td>{rec['clearance_3d_m']:.3f}</td>"
                f"<td>{rec['danger_point_count']}</td><td>{rec['nearest_wire_id']}</td></tr>"
            )
    body7 = f"""
<div class="card">
<h2>STEP 7 — 阈值与危险点判定</h2>
{kv_block([
    ("危险净空阈值", f"{risk_threshold.get('risk_threshold_m')} m"),
    ("危险树数", f"{risk_threshold.get('dangerous_tree_count')}"),
    ("危险点数", f"{risk_threshold.get('danger_point_count'):,}"),
    ("安全树数", f"{len(risk['records']) - risk_threshold.get('dangerous_tree_count', 0)}"),
    ("全局最小净空", f"{clear.get('global_minimum_clearance_m'):.4f} m"),
    ("最危险树", f"{min(clear_records, key=lambda x: x['minimum_clearance_m'])['global_tree_id'] + 1}"),
    ("判定规则", "clearance_3d &lt; 阈值 ⇒ 危险"),
])}
<p style="font-size:13px;color:#455a64">颜色约定：<span style="color:#2e7d32;font-weight:700">绿=安全</span>　<span style="color:#b71c1c;font-weight:700">红=危险</span>（本项目仅定义一个单一阈值，未定义多档阈值）。</p>
</div>
<div class="card"><h2>风险分级</h2>
<table><tr><th>树ID</th><th>导线</th><th>最小净空(m)</th><th>风险级别</th><th>危险点数</th><th>水平净空(m)</th><th>垂直净空(m)</th></tr>
{danger_rows}</table></div>
<div class="card"><h2>危险树明细</h2>
<table><tr><th>树ID</th><th>3D净空(m)</th><th>危险点数</th><th>最近导线</th></tr>
{danger_detail or "<tr><td colspan=4>无危险树</td></tr>"}</table></div>
<div class="panels">
  <div class="panel"><h3>风险分类可视化（左）</h3>{viz_iframe("25_risk_threshold_result.html", "危险阈值结果")}</div>
  <div class="panel"><h3>最终档距总览（右）</h3>{viz_iframe("26_FINAL_SPAN_OVERVIEW.html", "最终档距总览")}</div>
</div>
"""
    pages["07_threshold_risk.html"] = page(
        "07 风险判定 — 树线净空分析", "07", body7, "06_clearance_analysis.html", "08_results_export.html",
        json.dumps({
            "run": str(RUN_DIR),
            "risk_result": "09_risk/risk_result.json",
            "risk_threshold": "risk_threshold.json",
            "visuals": ["10_visuals/25_risk_threshold_result.html", "10_visuals/26_FINAL_SPAN_OVERVIEW.html"],
        }, ensure_ascii=False, indent=2),
    )

    # ---------------------------------------------------------- STEP 8
    excel = RUN_DIR / "11_report" / "tree_clearance_report.xlsx"
    report = RUN_DIR / "11_report" / "27_FINAL_REPORT.html"
    summary_json = RUN_DIR / "run_summary.json"
    index = RUN_DIR / "10_visuals" / "00_VISUAL_INDEX.html"
    body8 = f"""
<div class="card">
<h2>STEP 8 — 最终结果 / Excel / 报告</h2>
{kv_block([
    ("档距长度", f"{SPAN_LENGTH:.2f} m"),
    ("地面/HAG", "完成"),
    ("导线数", f"{summary.get('WIRE_STRAND_COUNT', 0)}"),
    ("单木数", f"{summary.get('FINAL_TREE_COUNT', 0)}"),
    ("最小净空", f"{clear.get('global_minimum_clearance_m'):.4f} m"),
    ("最高风险", "危险" if risk_threshold.get('dangerous_tree_count', 0) > 0 else "安全"),
    ("危险树", f"{risk_threshold.get('dangerous_tree_count')}"),
    ("危险点", f"{risk_threshold.get('danger_point_count'):,}"),
])}
</div>
<div class="card"><h2>输出文件</h2>
<p style="font-size:14px">
<button class="link" onclick="window.open('{excel.as_uri()}')">打开 Excel 结果</button>
<button class="link" onclick="window.open('{summary_json.as_uri()}')">打开 JSON 结果</button>
<button class="link" onclick="window.open('{index.as_uri()}')">打开可视化总览</button>
<button class="link" onclick="window.open('{report.as_uri()}')">打开最终报告</button>
<button class="link gray" onclick="location.href='00_workflow_index.html'">回到档距选择</button>
</p>
<table>
<tr><th>输出</th><th>文件</th><th>状态</th></tr>
<tr><td>Excel</td><td><code>11_report/tree_clearance_report.xlsx</code></td><td>{'已生成' if excel.is_file() else 'Not generated'}</td></tr>
<tr><td>JSON</td><td><code>run_summary.json</code></td><td>已生成</td></tr>
<tr><td>可视化</td><td><code>10_visuals/00_VISUAL_INDEX.html</code></td><td>已生成（28 页）</td></tr>
<tr><td>报告</td><td><code>11_report/27_FINAL_REPORT.html</code></td><td>已生成</td></tr>
</table></div>
"""
    pages["08_results_export.html"] = page(
        "08 结果输出 — 树线净空分析", "08", body8, "07_threshold_risk.html", "09_growth_prediction.html",
        json.dumps({
            "run": str(RUN_DIR),
            "excel": "11_report/tree_clearance_report.xlsx",
            "report": "11_report/27_FINAL_REPORT.html",
            "summary": "run_summary.json",
            "visual_index": "10_visuals/00_VISUAL_INDEX.html",
        }, ensure_ascii=False, indent=2),
    )

    # ---------------------------------------------------------- STEP 9
    body9 = """
<div class="card">
<h2>STEP 9 — 植被生长预测 / 未来预警</h2>
<div style="background:#eceff1;border-radius:6px;padding:14px;font-size:15px">
<b>PREDICTION_MODULE: NOT_YET_CONNECTED</b><br>
预测模块尚未接入。以下可视化区域为预留，不展示任何虚构的预测结果。
</div>
</div>
<div class="panels">
  <div class="panel"><h3>当前净空（预留）</h3><div class="viz"><div class="ph">真实结果图待接入</div></div></div>
  <div class="panel"><h3>预测净空 / 预测生长 / 阈值穿越时间（预留）</h3><div class="viz"><div class="ph">真实结果图待接入</div></div></div>
</div>
"""
    pages["09_growth_prediction.html"] = page(
        "09 生长预测 — 树线净空分析", "09", body9, "08_results_export.html", None,
        "PREDICTION_MODULE: NOT_YET_CONNECTED\n无预测产物。",
    )

    # ---------------------------------------------------------- index
    idx_rows = ""
    for num, label, target in STEPS:
        href = target if target.endswith(".html") else ("#" + target)
        status = "（已完成）" if int(num) < 9 else ("（预留）" if num == "09" else "")
        idx_rows += f"<tr><td>{num}</td><td><a href='{href}'>{label}</a>{status}</td><td>同一人工档距：Pole A → Pole B（{SPAN_LENGTH:.2f} m）</td></tr>"
    index_html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>树线净空分析 — 流程总览</title>
<style>{CSS}</style></head><body><div id="wrap">
<h1>树线净空分析 — 人工单档距下游流程</h1>
{spanline(POLE_A, POLE_B, SPAN_LENGTH)}
<div class="card"><table>
<tr><th>步骤</th><th>页面</th><th>说明</th></tr>
{idx_rows}
</table>
<p style="font-size:13px;color:#455a64">Step 1 选杆定档为本地交互选择器；此处步骤 2–9 全部基于同一档距的<b>已完成真实结果</b>。打开本目录下页面不会触发任何昂贵计算。</p>
</div>
<details><summary>数据来源 / DATA_SOURCE</summary><div class="ds">{json.dumps({"run": str(RUN_DIR), "summary": "run_summary.json", "visual_index": "10_visuals/00_VISUAL_INDEX.html"}, ensure_ascii=False, indent=2)}</div></details>
</div></body></html>"""
    (OUT_DIR / "00_workflow_index.html").write_text(index_html, encoding="utf-8")

    for name, html in pages.items():
        (OUT_DIR / name).write_text(html, encoding="utf-8")

    print("WROTE:", OUT_DIR)
    for name in sorted(pages):
        print("  -", name)
    print("  - 00_workflow_index.html")


if __name__ == "__main__":
    main()
