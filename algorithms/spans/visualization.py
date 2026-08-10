"""All 16 review visualizations for the pole-first span-based baseline.

Every review page is a bounded, deterministic sample of the persisted
artifacts.  Displayed points are always labelled as sampled; the authoritative
computation is never sampled.  Pages are written atomically so a partially
failed run never leaves a misleading HTML file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _even_rows(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _write_figure(figure: Any, output_html: Path) -> Path:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_html.with_name(output_html.name + ".tmp")
    figure.write_html(temporary, include_plotlyjs=True, full_html=True)
    os.replace(temporary, output_html)
    return output_html


def _scene_axes(title: str) -> dict[str, Any]:
    return {
        "scene": {
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
        },
        "title": title,
        "margin": {"l": 0, "r": 0, "t": 44, "b": 0},
    }


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


# ---------------------------------------------------------------------------
# Pole reviews (01, 02)
# ---------------------------------------------------------------------------


def _pole_review_points(poles: dict[str, Any], stage_dir: Path, number: str) -> None:
    import plotly.graph_objects as go

    evidence = poles["evidence"]
    candidates = poles["candidates"]
    wire_evidence = poles["wire_evidence"]
    validation = poles["validation"]
    origin = np.asarray(evidence.origin_xyz, dtype=np.float64)
    non_ground = evidence.non_ground_mask()
    background_rows = np.flatnonzero(non_ground)[_even_rows(np.count_nonzero(non_ground), 40_000)]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=evidence.xyz_local_m[background_rows, 0] + origin[0],
            y=evidence.xyz_local_m[background_rows, 1] + origin[1],
            z=evidence.xyz_local_m[background_rows, 2] + origin[2],
            mode="markers",
            name="非地面粗代表点（显示点已抽样；计算使用完整分辨率）",
            marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.22},
            hoverinfo="skip",
        )
    )
    for record in candidates.records:
        base = np.asarray(record.base_xyz_absolute_m, dtype=np.float64)
        top = np.asarray(record.top_xyz_absolute_m, dtype=np.float64)
        figure.add_trace(
            go.Scatter3d(
                x=[base[0], top[0]],
                y=[base[1], top[1]],
                z=[base[2], top[2]],
                mode="lines+markers",
                name=f"候选 {record.pole_candidate_id}（h={record.height_m:.1f}m）",
                line={"color": "#f59e0b", "width": 5},
                marker={"size": 3, "color": "#f59e0b"},
            )
        )
    for segment in wire_evidence.segments:
        start = np.asarray(segment.start_xyz_local_m, dtype=np.float64) + origin
        end = np.asarray(segment.end_xyz_local_m, dtype=np.float64) + origin
        figure.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                name=f"粗导线证据 {segment.segment_id}",
                line={"color": "#dc2626", "width": 3},
            )
        )
    figure.update_layout(**_scene_axes(f"电杆候选与粗导线证据（{number}）"))
    _write_figure(figure, stage_dir / f"{number}_pole_candidates_review.html")

    # 02 confirmed poles.
    figure2 = go.Figure()
    figure2.add_trace(
        go.Scatter3d(
            x=evidence.xyz_local_m[background_rows, 0] + origin[0],
            y=evidence.xyz_local_m[background_rows, 1] + origin[1],
            z=evidence.xyz_local_m[background_rows, 2] + origin[2],
            mode="markers",
            name="非地面粗代表点（样本）",
            marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.20},
            hoverinfo="skip",
        )
    )
    state_colors = {
        "CONFIRMED_POWER_POLE": "#16a34a",
        "LIKELY_POWER_POLE": "#2563eb",
        "POLE_LIKE_REVIEW": "#f59e0b",
        "REJECTED_POLE_LIKE": "#6b7280",
    }
    for record in validation.records:
        candidate = candidates.records[record.pole_candidate_id]
        base = np.asarray(candidate.base_xyz_absolute_m, dtype=np.float64)
        top = np.asarray(candidate.top_xyz_absolute_m, dtype=np.float64)
        color = state_colors.get(record.state, "#6b7280")
        figure2.add_trace(
            go.Scatter3d(
                x=[base[0], top[0]],
                y=[base[1], top[1]],
                z=[base[2], top[2]],
                mode="lines+markers",
                name=f"{record.state} #{record.pole_candidate_id}",
                line={"color": color, "width": 6},
                marker={"size": 4, "color": color},
            )
        )
    figure2.update_layout(**_scene_axes("电杆互证状态审查（02_confirmed_poles_review）"))
    _write_figure(figure2, stage_dir / "02_confirmed_poles_review.html")


# ---------------------------------------------------------------------------
# Topology review (03, 04)
# ---------------------------------------------------------------------------


def _topology_review_pages(
    poles: dict[str, Any],
    topology: Any,
    stage_dir: Path,
) -> None:
    import plotly.graph_objects as go

    evidence = poles["evidence"]
    origin = np.asarray(evidence.origin_xyz, dtype=np.float64)
    non_ground = evidence.non_ground_mask()
    background_rows = np.flatnonzero(non_ground)[_even_rows(np.count_nonzero(non_ground), 30_000)]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=evidence.xyz_local_m[background_rows, 0] + origin[0],
            y=evidence.xyz_local_m[background_rows, 1] + origin[1],
            z=evidence.xyz_local_m[background_rows, 2] + origin[2],
            mode="markers",
            name="非地面粗代表点（样本）",
            marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.18},
            hoverinfo="skip",
        )
    )
    line_colors = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#ea580c"]
    for line in topology.lines:
        color = line_colors[line.line_id % len(line_colors)]
        xyz = []
        for pole_id in line.ordered_pole_candidate_ids:
            candidate = poles["candidates"].records[pole_id]
            xyz.append(np.asarray(candidate.top_xyz_absolute_m, dtype=np.float64))
        matrix = np.asarray(xyz, dtype=np.float64)
        figure.add_trace(
            go.Scatter3d(
                x=matrix[:, 0],
                y=matrix[:, 1],
                z=matrix[:, 2],
                mode="lines+markers",
                name=f"{line.line_name}（{line.pole_count} 杆）",
                line={"color": color, "width": 6},
                marker={"size": 5, "color": color},
            )
        )
    figure.update_layout(**_scene_axes("电杆拓扑与线路排序审查（03_pole_topology_review）"))
    _write_figure(figure, stage_dir / "03_pole_topology_review.html")

    figure4 = go.Figure()
    figure4.add_trace(
        go.Scatter3d(
            x=evidence.xyz_local_m[background_rows, 0] + origin[0],
            y=evidence.xyz_local_m[background_rows, 1] + origin[1],
            z=evidence.xyz_local_m[background_rows, 2] + origin[2],
            mode="markers",
            name="非地面粗代表点（样本）",
            marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.16},
            hoverinfo="skip",
        )
    )
    for span in topology.spans:
        corners = _span_box_corners(span)
        figure4.add_trace(
            go.Scatter3d(
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                mode="lines",
                name=f"{span.span_name}（{span.length_m:.0f}m）",
                line={"color": "#2563eb", "width": 3},
                showlegend=span.span_id == 0,
            )
        )
    figure4.update_layout(**_scene_axes("档距划分审查（04_span_partition_review）"))
    _write_figure(figure4, stage_dir / "04_span_partition_review.html")


def _span_box_corners(span: Any) -> np.ndarray:
    minimum = np.asarray(span.core_bounds_min_xyz, dtype=np.float64)
    maximum = np.asarray(span.core_bounds_max_xyz, dtype=np.float64)
    corners = np.asarray(
        [
            [minimum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], minimum[2]],
            [maximum[0], maximum[1], minimum[2]],
            [minimum[0], maximum[1], minimum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [maximum[0], maximum[1], maximum[2]],
            [minimum[0], maximum[1], maximum[2]],
        ],
        dtype=np.float64,
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return corners[list(np.asarray(edges, dtype=np.int64).ravel())]


# ---------------------------------------------------------------------------
# Tree reviews (05, 06, 07, 08)
# ---------------------------------------------------------------------------


def _tree_review_pages(tree_outcomes: list[dict[str, Any]], stage_dir: Path) -> None:
    """Render the four required tree pages (05-08) combining every span."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    span_arrays: list[dict[str, Any]] = []
    for entry in tree_outcomes:
        chm_path = entry["outcome"].artifact_root / "chm_review.npz"
        if not chm_path.is_file():
            continue
        arrays = _load_npz_arrays(chm_path)
        span_arrays.append(
            {
                "span": entry["span"],
                "chm_height": np.asarray(arrays["chm_height_m"], dtype=np.float32),
                "chm_smoothed": np.asarray(arrays["chm_smoothed_m"], dtype=np.float32),
                "occupied": np.asarray(arrays["chm_occupied"], dtype=bool),
                "crown_label": np.asarray(arrays["crown_label"], dtype=np.int32),
                "tops_xy": np.asarray(arrays["accepted_tops_xy"], dtype=np.float64),
                "origin_xy": tuple(float(value) for value in arrays["chm_origin_xy"]),
                "cell_size": float(arrays["chm_cell_size_m"]),
                "members": (
                    _load_npz_arrays(entry["outcome"].artifact_root / "tree_members.npz")
                    if (entry["outcome"].artifact_root / "tree_members.npz").is_file()
                    else None
                ),
            }
        )
    if not span_arrays:
        return

    columns = min(4, len(span_arrays))
    rows_count = int(np.ceil(len(span_arrays) / columns))

    def heatmap_page(values_key: str, title: str, output_name: str) -> None:
        figure = make_subplots(
            rows=rows_count,
            cols=columns,
            subplot_titles=[item["span"].span_name for item in span_arrays],
        )
        for position, item in enumerate(span_arrays):
            row, col = divmod(position, columns)
            values = item[values_key]
            rows_g, cols_g = values.shape
            x_axis = item["origin_xy"][0] + (np.arange(cols_g) + 0.5) * item["cell_size"]
            y_axis = item["origin_xy"][1] + (np.arange(rows_g) + 0.5) * item["cell_size"]
            shown = np.where(item["occupied"], values, np.nan)
            figure.add_trace(
                go.Heatmap(
                    z=shown,
                    x=x_axis,
                    y=y_axis,
                    colorscale="Viridis",
                    colorbar={"title": "HAG (m)"} if position == 0 else None,
                    showscale=position == 0,
                    hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>HAG=%{z:.2f}<extra></extra>",
                ),
                row=row + 1,
                col=col + 1,
            )
            if len(item["tops_xy"]):
                figure.add_trace(
                    go.Scatter(
                        x=item["tops_xy"][:, 0],
                        y=item["tops_xy"][:, 1],
                        mode="markers",
                        name="树顶",
                        marker={"symbol": "x", "size": 8, "color": "#dc2626"},
                    ),
                    row=row + 1,
                    col=col + 1,
                )
        figure.update_layout(
            title=title,
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
        )
        _write_figure(figure, stage_dir / output_name)

    heatmap_page("chm_height", "CHM（HAG 冠层高度模型）", "05_chm_review.html")
    heatmap_page("chm_smoothed", "CHM 平滑", "06_tree_tops_review.html")
    heatmap_page(
        "crown_label",
        "流域分割冠层（marker watershed）",
        "07_watershed_crowns_review.html",
    )

    # 08 full-resolution tree preview (all spans in one 3D scene).
    figure8 = go.Figure()
    for item in span_arrays:
        if item["members"] is None:
            continue
        xyz = np.asarray(item["members"]["xyz_local_m"], dtype=np.float64)
        tree_id = np.asarray(item["members"]["tree_id"], dtype=np.int32)
        for tree in np.unique(tree_id):
            rows = np.flatnonzero(tree_id == tree)
            shown = rows[_even_rows(len(rows), 6_000)]
            figure8.add_trace(
                go.Scatter3d(
                    x=xyz[shown, 0],
                    y=xyz[shown, 1],
                    z=xyz[shown, 2],
                    mode="markers",
                    name=f"{item['span'].span_name} Tree {int(tree)}（{len(rows):,}）",
                    marker={
                        "size": 1.6,
                        "color": _tree_color(int(tree)),
                        "opacity": 0.85,
                    },
                )
            )
    figure8.update_layout(**_scene_axes("单木全分辨率回映预览（显示点已抽样；计算使用完整分辨率）"))
    _write_figure(figure8, stage_dir / "08_tree_segmentation_fullres_preview.html")


def _tree_color(tree_id: int) -> str:
    palette = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
               "#7c3aed", "#059669", "#b45309", "#db2777"]
    return palette[tree_id % len(palette)]


# ---------------------------------------------------------------------------
# Wire reviews (09, 10, 11, 12, 13)
# ---------------------------------------------------------------------------


def _wire_review_pages(
    wire_outcomes: list[dict[str, Any]],
    poles: dict[str, Any],
    stage_dir: Path,
    wire_region_root: Path,
) -> None:
    """Render the required wire pages (09-13) combining every span."""

    import plotly.graph_objects as go

    evidence = poles["evidence"]
    origin = np.asarray(evidence.origin_xyz, dtype=np.float64)
    non_ground = evidence.non_ground_mask()
    background_rows = np.flatnonzero(non_ground)[
        _even_rows(np.count_nonzero(non_ground), 20_000)
    ]
    span_diag: list[dict[str, Any]] = []
    for entry in wire_outcomes:
        diagnostics_path = entry["wire"].diagnostics.get("diagnostics_npz_path")
        diag_arrays: dict[str, np.ndarray] = {}
        if diagnostics_path and Path(diagnostics_path).is_file():
            diag_arrays = _load_npz_arrays(Path(diagnostics_path))
        span_diag.append({"wire": entry["wire"], "arrays": diag_arrays})

    figure9 = go.Figure()
    figure10 = go.Figure()
    for item in span_diag:
        fine_xyz = item["arrays"].get("fine_xyz_local_m")
        candidate = item["arrays"].get("candidate_mask")
        cluster = item["arrays"].get("cluster_id_by_rep")
        if fine_xyz is None or candidate is None:
            continue
        xyz = np.asarray(fine_xyz, dtype=np.float64) + origin
        rows = _even_rows(len(xyz), 12_000)
        figure9.add_trace(
            go.Scatter3d(
                x=xyz[rows, 0],
                y=xyz[rows, 1],
                z=xyz[rows, 2],
                mode="markers",
                name=f"{item['wire'].diagnostics.get('wire_strand_count', '')} 精细点（样本）",
                marker={"size": 1.4, "color": "#9ca3af", "opacity": 0.3},
                hoverinfo="skip",
            )
        )
        cand_rows = np.flatnonzero(candidate)[
            _even_rows(np.count_nonzero(candidate), 15_000)
        ]
        figure9.add_trace(
            go.Scatter3d(
                x=xyz[cand_rows, 0],
                y=xyz[cand_rows, 1],
                z=xyz[cand_rows, 2],
                mode="markers",
                name="导线几何候选",
                marker={"size": 2.2, "color": "#dc2626", "opacity": 0.85},
            )
        )
        if cluster is not None:
            for cluster_id in np.unique(cluster[cluster >= 0]):
                rows_c = np.flatnonzero(cluster == cluster_id)
                shown = rows_c[_even_rows(len(rows_c), 4_000)]
                figure10.add_trace(
                    go.Scatter3d(
                        x=xyz[shown, 0],
                        y=xyz[shown, 1],
                        z=xyz[shown, 2],
                        mode="markers",
                        name=f"簇 {int(cluster_id)}（{len(rows_c):,}）",
                        marker={
                            "size": 2.0,
                            "color": _tree_color(int(cluster_id)),
                            "opacity": 0.8,
                        },
                    )
                )
    figure9.update_layout(**_scene_axes("导线候选审查（显示点已抽样；计算使用完整分辨率）"))
    _write_figure(figure9, stage_dir / "09_wire_candidates_review.html")
    figure10.update_layout(**_scene_axes("导线连续性簇审查（DBSCAN）"))
    _write_figure(figure10, stage_dir / "10_wire_clusters_review.html")

    figure11 = go.Figure()
    figure12 = go.Figure()
    figure13 = go.Figure()
    if background_rows.size:
        for figure in (figure11, figure13):
            figure.add_trace(
                go.Scatter3d(
                    x=evidence.xyz_local_m[background_rows, 0] + origin[0],
                    y=evidence.xyz_local_m[background_rows, 1] + origin[1],
                    z=evidence.xyz_local_m[background_rows, 2] + origin[2],
                    mode="markers",
                    name="粗代表点（样本）",
                    marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.14},
                    hoverinfo="skip",
                )
            )
    for item in span_diag:
        for strand in item["wire"].strands:
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64)
            color = _tree_color(strand.wire_id)
            figure11.add_trace(
                go.Scatter3d(
                    x=curve[:, 0],
                    y=curve[:, 1],
                    z=curve[:, 2],
                    mode="lines+markers",
                    name=f"{strand.wire_name}（u={strand.median_u_m:.1f}m）",
                    line={"color": color, "width": 6},
                    marker={"size": 2, "color": color},
                )
            )
            figure12.add_trace(
                go.Scatter3d(
                    x=curve[:, 0],
                    y=curve[:, 1],
                    z=curve[:, 2],
                    mode="lines+markers",
                    name=f"{strand.wire_name}",
                    line={"color": color, "width": 6},
                    marker={"size": 2, "color": color},
                )
            )
            figure13.add_trace(
                go.Scatter3d(
                    x=curve[:, 0],
                    y=curve[:, 1],
                    z=curve[:, 2],
                    mode="lines",
                    name=f"{strand.wire_name}",
                    line={"color": "#dc2626", "width": 5},
                )
            )
    figure11.update_layout(**_scene_axes("导线根数（strand）审查"))
    _write_figure(figure11, stage_dir / "11_wire_strands_review.html")
    figure12.update_layout(**_scene_axes("导线曲线重建审查（sag 二次曲线）"))
    _write_figure(figure12, stage_dir / "12_wire_reconstruction_review.html")
    figure13.update_layout(**_scene_axes("原始点云与重建导线叠加"))
    _write_figure(figure13, stage_dir / "13_raw_vs_wire_overlay.html")


# ---------------------------------------------------------------------------
# Clearance reviews (14, 15)
# ---------------------------------------------------------------------------


def _clearance_review_pages(
    reconciliation: Any,
    clearance: Any,
    wire_outcomes: list[dict[str, Any]],
    stage_dir: Path,
) -> None:
    import plotly.graph_objects as go

    final_xyz = np.asarray(reconciliation.final_xyz_local_m, dtype=np.float64)
    final_tree = np.asarray(reconciliation.final_tree_id, dtype=np.int32)
    clearance_by_tree = {
        record.global_tree_id: record for record in clearance.records
    }
    figure14 = go.Figure()
    figure15 = go.Figure()
    for tree_id in np.unique(final_tree):
        if tree_id < 0:
            continue
        rows = np.flatnonzero(final_tree == tree_id)
        shown = rows[_even_rows(len(rows), 6_000)]
        record = clearance_by_tree.get(int(tree_id))
        clearance_m = record.minimum_clearance_m if record else None
        label = f"Tree {int(tree_id)}" + (
            f"（最小 {clearance_m:.2f}m）" if clearance_m is not None else ""
        )
        color = _clearance_color(clearance_m)
        figure14.add_trace(
            go.Scatter3d(
                x=final_xyz[shown, 0],
                y=final_xyz[shown, 1],
                z=final_xyz[shown, 2],
                mode="markers",
                name=label,
                marker={"size": 1.6, "color": color, "opacity": 0.85},
            )
        )
        if (
            record
            and record.nearest_tree_xyz_local_m is not None
            and record.nearest_wire_xyz_local_m is not None
        ):
            tree_point = np.asarray(record.nearest_tree_xyz_local_m, dtype=np.float64)
            wire_point = np.asarray(record.nearest_wire_xyz_local_m, dtype=np.float64)
            figure14.add_trace(
                go.Scatter3d(
                    x=[tree_point[0], wire_point[0]],
                    y=[tree_point[1], wire_point[1]],
                    z=[tree_point[2], wire_point[2]],
                    mode="lines",
                    name=f"最近距离线 Tree {int(tree_id)}",
                    line={"color": "#000000", "width": 4},
                )
            )
    for entry in wire_outcomes:
        for strand in entry["wire"].strands:
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64)
            figure14.add_trace(
                go.Scatter3d(
                    x=curve[:, 0],
                    y=curve[:, 1],
                    z=curve[:, 2],
                    mode="lines",
                    name=f"W {strand.wire_name}",
                    line={"color": "#dc2626", "width": 4},
                )
            )
    figure14.update_layout(**_scene_axes("树线距离审查（显示点已抽样；计算使用完整分辨率）"))
    _write_figure(figure14, stage_dir / "14_tree_wire_clearance_review.html")

    # 15 minimum clearance per tree (bar).
    values = [
        record.minimum_clearance_m
        for record in clearance.records
        if record.minimum_clearance_m is not None
    ]
    names = [
        record.external_name
        for record in clearance.records
        if record.minimum_clearance_m is not None
    ]
    figure15.add_trace(
        go.Bar(x=names, y=values, marker={"color": [_clearance_color(value) for value in values]})
    )
    figure15.update_layout(
        title="每棵树的最近导线最小净空（15_minimum_clearance_review）",
        xaxis_title="树",
        yaxis_title="最小净空 (m)",
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    _write_figure(figure15, stage_dir / "15_minimum_clearance_review.html")


def _clearance_color(clearance_m: float | None) -> str:
    if clearance_m is None:
        return "#9ca3af"
    if clearance_m < 2.0:
        return "#dc2626"
    if clearance_m < 4.0:
        return "#f59e0b"
    return "#16a34a"


# ---------------------------------------------------------------------------
# RUN_SUMMARY (16)
# ---------------------------------------------------------------------------


def _run_summary_page(summary: dict[str, Any], report_dir: Path) -> Path:
    links = {
        "01 电杆候选": "03_poles/01_pole_candidates_review.html",
        "02 电杆互证状态": "03_poles/02_confirmed_poles_review.html",
        "03 电杆拓扑": "04_topology/03_pole_topology_review.html",
        "04 档距划分": "04_topology/04_span_partition_review.html",
        "05 CHM": "06_tree/05_chm_review.html",
        "06 树顶": "06_tree/06_tree_tops_review.html",
        "07 流域冠层": "06_tree/07_watershed_crowns_review.html",
        "08 单木全分辨率预览": "06_tree/08_tree_segmentation_fullres_preview.html",
        "09 导线候选": "05_wire/09_wire_candidates_review.html",
        "10 导线簇": "05_wire/10_wire_clusters_review.html",
        "11 导线根数": "05_wire/11_wire_strands_review.html",
        "12 导线重建": "05_wire/12_wire_reconstruction_review.html",
        "13 原始与导线叠加": "05_wire/13_raw_vs_wire_overlay.html",
        "14 树线距离": "07_clearance/14_tree_wire_clearance_review.html",
        "15 最小净空": "07_clearance/15_minimum_clearance_review.html",
    }
    rendered = "".join(
        f'<li><a href="{href}">{label}</a></li>'
        for label, href in links.items()
    )
    rows = "".join(
        f"<tr><th>{_escape(str(key))}</th><td>{_escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key not in {"TREES", "SPANS", "LINES", "CLEARANCE_RECORDS"}
    )
    page = (
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<title>POLE_SPAN_MATURE_BASELINE 运行摘要</title>"
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1100px;"
        "margin:36px auto;padding:0 20px}th{text-align:left;background:#eef2ff}"
        "th,td{padding:7px 10px;border:1px solid #d1d5db;vertical-align:top}"
        "table{border-collapse:collapse;width:100%}</style><body>"
        "<h1>POLE_SPAN_MATURE_BASELINE 运行摘要</h1>"
        "<p><strong>NON-CERTIFIED: TRUE</strong> — 研究演示，非认证结果。</p>"
        f"<h2>主要可视化</h2><ul>{rendered}</ul><h2>运行指标</h2><table>{rows}</table>"
        "</body></html>"
    )
    output = report_dir / "16_RUN_SUMMARY.html"
    report_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(page, encoding="utf-8")
    os.replace(temporary, output)
    return output


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_all_review_pages(
    root: Path,
    summary: dict[str, Any],
    poles: dict[str, Any],
    topology: dict[str, Any],
    wire_outcomes: list[dict[str, Any]],
    tree_outcomes: list[dict[str, Any]],
    reconciliation: Any,
    clearance: Any,
) -> Path:
    """Generate all review pages and the RUN_SUMMARY (Part 39)."""

    _pole_review_points(poles, root / "03_poles", "01")
    _topology_review_pages(poles, topology["topology"], root / "04_topology")
    _tree_review_pages(tree_outcomes, root / "06_tree")
    _wire_review_pages(wire_outcomes, poles, root / "05_wire", root / "05_wire")
    _clearance_review_pages(reconciliation, clearance, wire_outcomes, root / "07_clearance")
    return _run_summary_page(summary, root / "08_report")


def generate_blocked_review_pages(
    root: Path,
    poles: dict[str, Any],
    topology: Any,
) -> Path:
    """Render pole/topology review pages when no spans could be established."""

    _pole_review_points(poles, root / "03_poles", "01")
    _topology_review_pages(poles, topology, root / "04_topology")
    blocked_page = root / "08_report" / "16_RUN_SUMMARY.html"
    blocked_page.parent.mkdir(parents=True, exist_ok=True)
    blocked_page.write_text(
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<title>SPAN_PIPELINE: BLOCKED_INSUFFICIENT_POLES</title><body>"
        "<h1>档距流水线已阻止</h1><p>未能确认足够的可用电杆，未伪造任何档距。</p>"
        "<ul>"
        "<li><a href=\"../03_poles/01_pole_candidates_review.html\">电杆候选审查</a></li>"
        "<li><a href=\"../03_poles/02_confirmed_poles_review.html\">电杆互证状态审查</a></li>"
        "<li><a href=\"../04_topology/03_pole_topology_review.html\">电杆拓扑审查</a></li>"
        "<li><a href=\"../04_topology/04_span_partition_review.html\">档距划分审查</a></li>"
        "</ul></body></html>",
        encoding="utf-8",
    )
    return blocked_page


__all__ = [
    "generate_all_review_pages",
    "generate_blocked_review_pages",
]
