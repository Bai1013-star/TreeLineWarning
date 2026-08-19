"""Offline visual diagnostic chain for the manual single-span route.

Every teacher-facing page is a self-contained offline HTML (plotly.js embedded)
rendered from a deterministic bounded sample — authoritative calculations are
never sampled.  Pages are numbered 01..27 plus ``00_VISUAL_INDEX.html``.  They
are diagnostics, not processing gates; generation never blocks the pipeline.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from algorithms.manual_span.pipeline import ManualSingleSpanRun

_MAX_BACKGROUND = 40_000
_MAX_FINE = 12_000
_MAX_TREE_FULLRES = 6_000
_MAX_DANGER = 40_000

_ProgressSink = Callable[[str], None] | None

_MANUAL_STAGES = [
    ("01_original_cloud_manual_selection", "01_原始点云与人工选杆"),
    ("02_pole_refinement", "02_电杆精细化"),
    ("03_span_definition", "03_档距定义"),
    ("04_span_raw_cloud", "04_档距原始点云"),
    ("05_ground_hag", "05_地面与高度(HAG)"),
    ("06_broad_vegetation_wire_routing", "06_粗植被/导线路由"),
    ("07_fine_vegetation", "07_精细植被(0.10 m)"),
    ("08_chm_raw", "08_CHM原始"),
    ("09_chm_smoothed", "09_CHM平滑"),
    ("10_chm_raw_vs_smoothed", "10_CHM原始与平滑对比"),
    ("11_raw_tree_top_candidates", "11_原始树顶候选"),
    ("12_accepted_tree_tops", "12_接受树顶"),
    ("13_watershed_crowns", "13_分水岭树冠"),
    ("14_chm_tops_watershed_overlay", "14_CHM/树顶/树冠叠加"),
    ("15_single_tree_fine_3d", "15_单木精细三维"),
    ("16_single_tree_fullres_preview", "16_单木全分辨率预览"),
    ("17_dbh_diagnostics", "17_DBH胸径诊断"),
    ("18_wire_candidates", "18_导线候选点"),
    ("19_wire_s_u_z", "19_导线(s,u,z)坐标"),
    ("20_wire_dbscan_clusters", "20_导线DBSCAN簇"),
    ("21_wire_strands", "21_导线线束"),
    ("22_wire_curve_fits", "22_导线曲线拟合"),
    ("23_raw_cloud_vs_reconstructed_wires", "23_原始点云与重建导线"),
    ("24_tree_wire_nearest_pairs", "24_树线最近点对"),
    ("25_risk_threshold_result", "25_危险净空阈值结果"),
    ("26_FINAL_SPAN_OVERVIEW", "26_最终档距总览"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _figure(title: str, *, height: int = 760) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "x": 0.5},
        height=height,
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 90, "b": 40},
        legend={"orientation": "h", "x": 0.0, "y": 1.02},
    )
    fig.update_scenes(
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"
    )
    return fig


def _abs(run: ManualSingleSpanRun, xyz_local: Any) -> np.ndarray:
    return np.asarray(xyz_local, dtype=np.float64) + np.asarray(run.origin_xyz, dtype=np.float64)


def _box_corners(
    min_xyz: tuple[float, float, float],
    max_xyz: tuple[float, float, float],
) -> np.ndarray:
    mn = np.asarray(min_xyz, dtype=np.float64)
    mx = np.asarray(max_xyz, dtype=np.float64)
    return np.array(
        [
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ]
    )


def _add_box(fig: go.Figure, min_xyz: Any, max_xyz: Any, name: str, color: str) -> None:
    corners = _box_corners(min_xyz, max_xyz)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for a, b in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[corners[a, 0], corners[b, 0]],
                y=[corners[a, 1], corners[b, 1]],
                z=[corners[a, 2], corners[b, 2]],
                mode="lines",
                line={"color": color, "width": 2},
                name=name,
                showlegend=(a == 0 and b == 1),
            )
        )


def _add_points(
    fig: go.Figure,
    xyz: Any,
    name: str,
    color: Any = "#1f77b4",
    size: float = 2.0,
    opacity: float = 0.8,
) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker={"size": size, "color": color, "opacity": opacity},
            name=name,
        )
    )


def _add_markers(fig: go.Figure, xyz: Any, name: str, color: str, size: float = 8) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers+text",
            text=[name] * len(xyz),
            textposition="top center",
            marker={"size": size, "color": color, "symbol": "diamond"},
            name=name,
        )
    )


def _add_lines(fig: go.Figure, xyz: Any, name: str, color: str, width: float = 4) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="lines",
            line={"color": color, "width": width},
            name=name,
        )
    )


def _load_region_xyz(run: ManualSingleSpanRun, sidecar_path: Path, maximum: int) -> np.ndarray:
    from algorithms.vegetation.coarse_to_fine import load_raw_region

    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        return np.empty((0, 3), dtype=np.float64)
    region = load_raw_region(sidecar_path)
    if region.count == 0:
        return np.empty((0, 3), dtype=np.float64)
    rows = _even_indices(region.count, maximum)
    xyz = np.asarray(region.xyz()[rows], dtype=np.float64)
    return _abs(run, xyz)


def _load_region_xyz_rgb(run: ManualSingleSpanRun, sidecar_path: Path, maximum: int):
    from algorithms.vegetation.coarse_to_fine import load_raw_region

    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=np.int64)
    region = load_raw_region(sidecar_path)
    if region.count == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=np.int64)
    rows = _even_indices(region.count, maximum)
    xyz = _abs(run, np.asarray(region.xyz()[rows], dtype=np.float64))
    classification = np.asarray(region.classification()[rows], dtype=np.uint8)
    original_index = np.asarray(region.original_index()[rows], dtype=np.int64)
    return xyz, classification, original_index


def _chm_heatmap(
    run: ManualSingleSpanRun, title: str, values: np.ndarray, origin_xy: tuple[float, float],
    cell_size_m: float, colorscale: str = "Viridis",
) -> go.Figure:
    origin = np.asarray(run.origin_xyz, dtype=np.float64)
    rows, cols = values.shape
    x = origin[0] + (np.arange(cols) + 0.5) * cell_size_m
    y = origin[1] + (np.arange(rows) + 0.5) * cell_size_m
    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=values,
            colorscale=colorscale,
            colorbar={"title": "CHM (m)"},
            hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>HAG=%{z:.2f} m<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        height=700,
        template="plotly_white",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        margin={"l": 40, "r": 40, "t": 90, "b": 40},
    )
    return fig


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_tree_fine(run: ManualSingleSpanRun) -> dict[str, np.ndarray] | None:
    path = run.run_root / "06_tree" / "span_001" / "tree_fine.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as stream:
        return {
            "xyz_local_m": np.asarray(stream["xyz_local_m"], dtype=np.float64),
            "hag_m": np.asarray(stream["fine_hag_m"], dtype=np.float32),
            "tree_id": np.asarray(stream["fine_tree_id"], dtype=np.int32),
        }


def _load_tree_members(run: ManualSingleSpanRun) -> dict[str, np.ndarray] | None:
    path = run.run_root / "06_tree" / "span_001" / "tree_members.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as stream:
        return {
            "original_index": np.asarray(stream["original_index"], dtype=np.int64),
            # tree_members.npz xyz is at the source float32 precision; never
            # upcast the full array into RAM when generating visuals.
            "xyz_local_m": np.asarray(stream["xyz_local_m"], dtype=np.float32),
            "tree_id": np.asarray(stream["tree_id"], dtype=np.int32),
            "in_core": np.asarray(stream["in_core"], dtype=bool),
        }


def _load_chm_review(run: ManualSingleSpanRun) -> dict[str, Any] | None:
    path = run.run_root / "06_tree" / "span_001" / "chm_review.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as stream:
        return {
            "chm_height_m": np.asarray(stream["chm_height_m"], dtype=np.float32),
            "chm_smoothed_m": np.asarray(stream["chm_smoothed_m"], dtype=np.float32),
            "chm_occupied": np.asarray(stream["chm_occupied"], dtype=bool),
            "crown_label": np.asarray(stream["crown_label"], dtype=np.int32),
            "chm_origin_xy": tuple(float(v) for v in stream["chm_origin_xy"]),
            "chm_cell_size_m": float(stream["chm_cell_size_m"]),
            "accepted_tops_xy": np.asarray(stream["accepted_tops_xy"], dtype=np.float64),
        }


def _load_wire_diag(run: ManualSingleSpanRun) -> dict[str, np.ndarray] | None:
    path = run.run_root / "05_wire" / "span_001" / "span_wire_diagnostics.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as stream:
        return {
            "fine_xyz_local_m": np.asarray(stream["fine_xyz_local_m"], dtype=np.float64),
            "candidate_mask": np.asarray(stream["candidate_mask"], dtype=bool),
            "cluster_id_by_rep": np.asarray(stream["cluster_id_by_rep"], dtype=np.int32),
            "s_m": np.asarray(stream["s_m"], dtype=np.float64),
            "u_m": np.asarray(stream["u_m"], dtype=np.float64),
            "z_m": np.asarray(stream["z_m"], dtype=np.float64),
        }


def _load_wire_curves(run: ManualSingleSpanRun) -> list[np.ndarray]:
    path = run.run_root / "05_wire" / "span_001" / "w1_arrays.npz"
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=False) as stream:
        curves = np.asarray(stream["sampled_curve_xyz"], dtype=np.float32)
        offsets = np.asarray(stream["sampled_curve_offsets"], dtype=np.int64)
    result = []
    for position in range(len(offsets) - 1):
        segment = curves[int(offsets[position]): int(offsets[position + 1])]
        if len(segment) >= 2:
            result.append(segment)
    return result


def _tree_sizes(run: ManualSingleSpanRun) -> dict[int, int]:
    members = _load_tree_members(run)
    if members is None or len(members["tree_id"]) == 0:
        return {}
    ids, counts = np.unique(members["tree_id"][members["tree_id"] >= 0], return_counts=True)
    return {int(uid): int(count) for uid, count in zip(ids, counts, strict=True)}


def _largest_tree_ids(run: ManualSingleSpanRun, count: int = 6) -> list[int]:
    sizes = _tree_sizes(run)
    ordered = sorted(sizes, key=lambda tid: (-sizes[tid], tid))
    return ordered[:count]


# ---------------------------------------------------------------------------
# Pages 01..04 : selection / refinement / span
# ---------------------------------------------------------------------------


def _page_01(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("01 原始点云与人工选杆（真实坐标）")
    sample = run.display_sample or {}
    xyz = sample.get("xyz_absolute", np.empty((0, 3)))
    rgb = sample.get("rgb", np.empty((0, 3)))
    if len(xyz):
        color = (
            np.clip(rgb.astype(np.float64), 0, 255) / 255.0
            if rgb.shape == xyz.shape and len(rgb)
            else "lightsteelblue"
        )
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker={"size": 2.0, "color": color, "opacity": 0.9},
                name=f"原始点云采样 ({len(xyz)} 点)",
            )
        )
    for label, pole in (("A", run.pole_a), ("B", run.pole_b)):
        if pole is None:
            continue
        click = np.asarray(pole.clicked_xyz_absolute_m, dtype=np.float64)
        base = np.asarray(pole.base_xyz_absolute_m, dtype=np.float64)
        top = np.asarray(pole.top_xyz_absolute_m, dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=[click[0]], y=[click[1]], z=[click[2]],
                mode="markers+text", text=[f"点击 Pole {label}"],
                textposition="top center",
                marker={"size": 10, "color": "orange", "symbol": "circle"},
                name=f"Pole {label} 点击",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[base[0], top[0]], y=[base[1], top[1]], z=[base[2], top[2]],
                mode="lines+markers",
                line={"color": "red", "width": 6},
                marker={"size": 6, "color": "red"},
                name=f"Pole {label} 精细化",
            )
        )
    _write_figure(fig, path)


def _page_02(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("02 电杆精细化（点击邻域 + 精化杆身）")
    origin = np.asarray(run.origin_xyz, dtype=np.float64)
    for label, pole in (("A", run.pole_a), ("B", run.pole_b)):
        if pole is None:
            continue
        click = np.asarray(pole.clicked_xyz_absolute_m, dtype=np.float64)
        if run.evidence is not None:
            local_click = click - origin
            dist = np.linalg.norm(
                run.evidence.xyz_local_m[:, :2] - local_click[:2][None, :], axis=1
            )
            rows = np.flatnonzero(dist <= run.manual.refine_search_radius_m)
            rows = rows[_even_indices(len(rows), 4000)]
            if len(rows):
                nbr = run.evidence.xyz_local_m[rows] + origin
                _add_points(fig, nbr, f"Pole {label} 邻域", "#b0bec5", size=1.5, opacity=0.6)
        base = np.asarray(pole.base_xyz_absolute_m, dtype=np.float64)
        top = np.asarray(pole.top_xyz_absolute_m, dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=[base[0], top[0]], y=[base[1], top[1]], z=[base[2], top[2]],
                mode="lines+markers",
                line={"color": "red", "width": 8},
                marker={"size": 8, "color": "red"},
                name=f"Pole {label} 精化杆身 h={pole.height_m:.1f}m",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[click[0]], y=[click[1]], z=[click[2]],
                mode="markers+text", text=[f"Pole {label}"], textposition="top center",
                marker={"size": 9, "color": "orange", "symbol": "circle"},
                name=f"Pole {label} 点击",
            )
        )
    _write_figure(fig, path)


def _page_03(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("03 档距定义（A→B，核心区+边缘区）")
    if run.span is None:
        _write_figure(fig, path)
        return
    span = run.span
    origin = np.asarray(run.origin_xyz, dtype=np.float64)
    core_min = np.asarray(span.core_bounds_min_xyz) + origin
    core_max = np.asarray(span.core_bounds_max_xyz) + origin
    halo_min = np.asarray(span.halo_bounds_min_xyz) + origin
    halo_max = np.asarray(span.halo_bounds_max_xyz) + origin
    start = np.asarray(span.start_xyz_absolute_m, dtype=np.float64)
    end = np.asarray(span.end_xyz_absolute_m, dtype=np.float64)
    _add_box(fig, halo_min, halo_max, "边缘区 (halo)", "#90a4ae")
    _add_box(fig, core_min, core_max, "核心区 (core)", "#1f77b4")
    _add_lines(fig, np.asarray([start, end]), f"档距 {span.length_m:.1f} m", "red", 6)
    _add_markers(fig, np.asarray([start]), "Pole A", "red")
    _add_markers(fig, np.asarray([end]), "Pole B", "red")
    _write_figure(fig, path)


def _page_04(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("04 档距原始点云（halo 内全分辨率点，采样显示）")
    sidecar = run.run_root / "04_regions" / "halo_region" / "region_sidecar.json"
    xyz, classification, _ = _load_region_xyz_rgb(run, sidecar, _MAX_BACKGROUND)
    if len(xyz):
        ground = classification == 2
        fig.add_trace(
            go.Scatter3d(
                x=xyz[~ground, 0], y=xyz[~ground, 1], z=xyz[~ground, 2],
                mode="markers", marker={"size": 1.8, "color": "#26a69a", "opacity": 0.6},
                name=f"非地面 ({np.count_nonzero(~ground)} 点)",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=xyz[ground, 0], y=xyz[ground, 1], z=xyz[ground, 2],
                mode="markers", marker={"size": 1.6, "color": "#bdbdbd", "opacity": 0.5},
                name=f"地面 ({np.count_nonzero(ground)} 点)",
            )
        )
    _write_figure(fig, path)


# ---------------------------------------------------------------------------
# Pages 05..07 : ground / routing / fine vegetation
# ---------------------------------------------------------------------------


def _page_05(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("05 地面与高度 (HAG)")
    if run.evidence is not None:
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        xyz = run.evidence.xyz_local_m + origin
        ground = run.evidence.classification == 2
        rows_g = np.flatnonzero(ground)[_even_indices(np.count_nonzero(ground), 20000)]
        rows_n = np.flatnonzero(~ground)[_even_indices(np.count_nonzero(~ground), 20000)]
        _add_points(fig, xyz[rows_g], "地面 (class 2)", "#9e9e9e", size=1.8, opacity=0.7)
        fig.add_trace(
            go.Scatter3d(
                x=xyz[rows_n, 0], y=xyz[rows_n, 1], z=xyz[rows_n, 2],
                mode="markers",
                marker={
                    "size": 2.0,
                    "color": run.evidence.hag_m[rows_n],
                    "colorscale": "Jet",
                    "colorbar": {"title": "HAG (m)"},
                    "opacity": 0.85,
                },
                name="非地面点 (按HAG着色)",
            )
        )
    _write_figure(fig, path)


def _page_06(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("06 粗植被 / 导线路由（保留代表点）")
    diag_path = (
        run.run_root
        / "06_tree"
        / "coarse_routing"
        / "manual_span_fast_vegetation_diagnostics.npz"
    )
    origin = np.asarray(run.origin_xyz, dtype=np.float64)
    if diag_path.is_file():
        with np.load(diag_path, allow_pickle=False) as stream:
            state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
            xyz_local = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        retained = state == 7  # VEGETATION_RETAINED
        rows = np.flatnonzero(retained)[_even_indices(np.count_nonzero(retained), 20000)]
        if len(rows):
            _add_points(
                fig, xyz_local[rows] + origin, "保留植被代表点", "#2e7d32",
                size=2.2, opacity=0.8,
            )
    if run.evidence is not None and run.span is not None:
        halo_min = np.asarray(run.span.halo_bounds_min_xyz)
        halo_max = np.asarray(run.span.halo_bounds_max_xyz)
        inside = np.all(run.evidence.xyz_local_m >= halo_min, axis=1) & np.all(
            run.evidence.xyz_local_m <= halo_max, axis=1
        )
        # Show coarse wire candidates inside the halo.
        from algorithms.poles.wire_evidence import detect_coarse_wire_evidence

        wire_ev = detect_coarse_wire_evidence(run.evidence, run.cfg)
        candidate = np.asarray(wire_ev.candidate_mask, dtype=bool) & inside
        rows = np.flatnonzero(candidate)[_even_indices(np.count_nonzero(candidate), 12000)]
        if len(rows):
            _add_points(
                fig, run.evidence.xyz_local_m[rows] + origin, "粗导线候选", "red",
                size=2.0, opacity=0.9,
            )
    _write_figure(fig, path)


def _page_07(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("07 精细植被 (0.10 m 代表点，按HAG着色)")
    fine = _load_tree_fine(run)
    if fine is not None and len(fine["xyz_local_m"]):
        rows = _even_indices(len(fine["xyz_local_m"]), _MAX_FINE)
        xyz = _abs(run, fine["xyz_local_m"][rows])
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker={
                    "size": 1.6,
                    "color": fine["hag_m"][rows],
                    "colorscale": "Viridis",
                    "colorbar": {"title": "HAG (m)"},
                    "opacity": 0.8,
                },
                name="精细植被",
            )
        )
    _write_figure(fig, path)


# ---------------------------------------------------------------------------
# Pages 08..14 : CHM / tops / watershed
# ---------------------------------------------------------------------------


def _page_08(run: ManualSingleSpanRun, path: Path) -> None:
    chm = _load_chm_review(run)
    if chm is None:
        _write_figure(_figure("08 CHM原始"), path)
        return
    _write_figure(_chm_heatmap(run, "08 CHM原始（每格最大HAG）", chm["chm_height_m"],
                               chm["chm_origin_xy"], chm["chm_cell_size_m"]), path)


def _page_09(run: ManualSingleSpanRun, path: Path) -> None:
    chm = _load_chm_review(run)
    if chm is None:
        _write_figure(_figure("09 CHM平滑"), path)
        return
    _write_figure(_chm_heatmap(run, "09 CHM平滑（归一化高斯）", chm["chm_smoothed_m"],
                               chm["chm_origin_xy"], chm["chm_cell_size_m"]), path)


def _page_10(run: ManualSingleSpanRun, path: Path) -> None:
    chm = _load_chm_review(run)
    if chm is None:
        _write_figure(_figure("10 CHM原始与平滑对比"), path)
        return
    raw = _chm_heatmap(
        run, "10a CHM原始", chm["chm_height_m"], chm["chm_origin_xy"], chm["chm_cell_size_m"]
    )
    sm = _chm_heatmap(
        run, "10b CHM平滑", chm["chm_smoothed_m"], chm["chm_origin_xy"], chm["chm_cell_size_m"]
    )
    path_raw = path.with_name("10a_chm_raw_detail.html")
    path_sm = path.with_name("10b_chm_smoothed_detail.html")
    _write_figure(raw, path_raw)
    _write_figure(sm, path_sm)
    fig = _figure("10 CHM原始与平滑对比（含细节页）")
    fig.add_annotation(
        text="原始 CHM：10a_chm_raw_detail.html　|　平滑 CHM：10b_chm_smoothed_detail.html",
        x=0.5, y=0.5, showarrow=False, font={"size": 16},
    )
    _write_figure(fig, path)


def _page_11(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("11 原始树顶候选（局部极大值，全部）")
    chm = _load_chm_review(run)
    if chm is not None and run.tree_outcome is not None:
        raw_count = run.tree_outcome.raw_local_maxima
        fig.add_annotation(
            text=f"原始局部极大值数量：{raw_count}",
            x=0.5, y=0.5, showarrow=False, font={"size": 18},
        )
    _write_figure(fig, path)


def _page_12(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("12 接受树顶（自适应抑制）")
    chm = _load_chm_review(run)
    if chm is not None:
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        tops_xy = chm["accepted_tops_xy"]
        if len(tops_xy):
            tops_abs = tops_xy + origin[:2][None, :]
            if run.tree_outcome is not None and run.tree_outcome.tree_records:
                heights = [
                    float(record.get("apex_height_m", 0.0))
                    for record in run.tree_outcome.tree_records
                ]
            else:
                heights = [1.0] * len(tops_xy)
            fig.add_trace(
                go.Scatter3d(
                    x=tops_abs[:, 0], y=tops_abs[:, 1],
                    z=[np.asarray(run.origin_xyz)[2] + h for h in heights],
                    mode="markers+text",
                    text=[str(i + 1) for i in range(len(tops_abs))],
                    textposition="top center",
                    marker={"size": 8, "color": "red"},
                    name="接受树顶",
                )
            )
    _write_figure(fig, path)


def _page_13(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("13 分水岭树冠（2D 俯视）")
    chm = _load_chm_review(run)
    if chm is not None:
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        labels = chm["crown_label"]
        rows, cols = labels.shape
        x = origin[0] + (np.arange(cols) + 0.5) * chm["chm_cell_size_m"]
        y = origin[1] + (np.arange(rows) + 0.5) * chm["chm_cell_size_m"]
        fig = go.Figure(
            go.Heatmap(
                x=x, y=y, z=np.where(labels > 0, labels, np.nan),
                colorscale="Rainbow",
                colorbar={"title": "树冠ID"},
                hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>Crown=%{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title={"text": "13 分水岭树冠（2D 俯视）", "x": 0.5},
            height=700, template="plotly_white",
            xaxis_title="X (m)", yaxis_title="Y (m)",
            margin={"l": 40, "r": 40, "t": 90, "b": 40},
        )
    _write_figure(fig, path)


def _page_14(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("14 CHM/树顶/树冠叠加（俯视）")
    chm = _load_chm_review(run)
    if chm is not None:
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        rows, cols = chm["chm_height_m"].shape
        x = origin[0] + (np.arange(cols) + 0.5) * chm["chm_cell_size_m"]
        y = origin[1] + (np.arange(rows) + 0.5) * chm["chm_cell_size_m"]
        fig.add_trace(
            go.Heatmap(
                x=x, y=y, z=chm["chm_smoothed_m"],
                colorscale="Viridis",
                colorbar={"title": "CHM (m)"},
                opacity=0.7,
                hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>CHM=%{z:.2f}<extra></extra>",
            )
        )
        tops_xy = chm["accepted_tops_xy"]
        if len(tops_xy):
            tops_abs = tops_xy + origin[:2][None, :]
            fig.add_trace(
                go.Scatter(
                    x=tops_abs[:, 0], y=tops_abs[:, 1],
                    mode="markers+text", text=[str(i + 1) for i in range(len(tops_abs))],
                    textposition="top center",
                    marker={"size": 9, "color": "red", "symbol": "x"},
                    name="接受树顶",
                )
            )
        fig.update_layout(
            title={"text": "14 CHM/树顶/树冠叠加（俯视）", "x": 0.5},
            height=700, template="plotly_white",
            xaxis_title="X (m)", yaxis_title="Y (m)",
            legend={"orientation": "h", "x": 0, "y": 1.02},
            margin={"l": 40, "r": 40, "t": 90, "b": 40},
        )
    _write_figure(fig, path)


# ---------------------------------------------------------------------------
# Pages 15..17 : single tree / fullres / DBH
# ---------------------------------------------------------------------------


def _page_15(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("15 单木精细三维（0.10 m 代表点）")
    fine = _load_tree_fine(run)
    if fine is not None and len(fine["xyz_local_m"]):
        ids = [int(value) for value in np.unique(fine["tree_id"][fine["tree_id"] >= 0])]
        for tree_id in ids[:_largest_tree_count(run, fine)]:
            rows = np.flatnonzero(fine["tree_id"] == tree_id)
            rows = rows[_even_indices(len(rows), 3000)]
            xyz = _abs(run, fine["xyz_local_m"][rows])
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    mode="markers", marker={"size": 1.4, "opacity": 0.75},
                    name=f"Tree {tree_id + 1}",
                )
            )
    _write_figure(fig, path)


def _largest_tree_count(run: ManualSingleSpanRun, fine: dict[str, np.ndarray]) -> int:
    ids = np.unique(fine["tree_id"][fine["tree_id"] >= 0])
    return int(min(6, len(ids)))


def _page_16(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("16 单木全分辨率预览（按树着色）")
    members = _load_tree_members(run)
    if members is not None and len(members["xyz_local_m"]):
        ids = [int(value) for value in np.unique(members["tree_id"][members["tree_id"] >= 0])]
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        for position, tree_id in enumerate(ids[:6]):
            rows = np.flatnonzero(members["tree_id"] == tree_id)
            rows = rows[_even_indices(len(rows), 4000)]
            xyz = _abs(run, members["xyz_local_m"][rows])
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    mode="markers",
                    marker={
                        "size": 1.5,
                        "color": palette[position % len(palette)],
                        "opacity": 0.7,
                    },
                    name=f"Tree {tree_id + 1}",
                )
            )
    _write_figure(fig, path)


def _page_17(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("17 DBH胸径诊断（胸高带茎点）")
    members = _load_tree_members(run)
    if (
        members is not None
        and len(members["xyz_local_m"])
        and run.coarse_ground_xyz_absolute is not None
    ):
        from scipy.spatial import cKDTree

        from algorithms.manual_span.metrics import _hag_chunked

        ground = np.asarray(run.coarse_ground_xyz_absolute, dtype=np.float64)
        ground_tree = cKDTree(ground[:, :2])
        ids = [int(value) for value in np.unique(members["tree_id"][members["tree_id"] >= 0])]
        for tree_id in ids[:8]:
            rows = np.flatnonzero(members["tree_id"] == tree_id)
            sample = rows[_even_indices(len(rows), 8000)]
            pts = members["xyz_local_m"][sample]
            hag = _hag_chunked(pts, ground_tree, ground, float(run.origin_xyz[2]), 100_000)
            band = (hag >= 1.3 - 0.45) & (hag <= 1.3 + 0.45)
            xyz_all = _abs(run, pts)
            fig.add_trace(
                go.Scatter3d(
                    x=xyz_all[:, 0], y=xyz_all[:, 1], z=xyz_all[:, 2],
                    mode="markers", marker={"size": 1.4, "color": "#90a4ae", "opacity": 0.5},
                    name=f"Tree {tree_id + 1} 成员",
                )
            )
            if np.any(band):
                xyz_band = _abs(run, pts[band])
                fig.add_trace(
                    go.Scatter3d(
                        x=xyz_band[:, 0], y=xyz_band[:, 1], z=xyz_band[:, 2],
                        mode="markers", marker={"size": 4, "color": "red"},
                        name=f"Tree {tree_id + 1} 胸高带",
                    )
                )
    _write_figure(fig, path)


# ---------------------------------------------------------------------------
# Pages 18..23 : wire evidence chain
# ---------------------------------------------------------------------------


def _page_18(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("18 导线候选点（低散射/水平证据）")
    diag = _load_wire_diag(run)
    if diag is not None and len(diag["fine_xyz_local_m"]):
        rows = _even_indices(len(diag["fine_xyz_local_m"]), _MAX_FINE)
        xyz = _abs(run, diag["fine_xyz_local_m"][rows])
        candidate = diag["candidate_mask"][rows]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[~candidate, 0], y=xyz[~candidate, 1], z=xyz[~candidate, 2],
                mode="markers", marker={"size": 1.4, "color": "#90a4ae", "opacity": 0.5},
                name=f"非候选 ({np.count_nonzero(~candidate)})",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=xyz[candidate, 0], y=xyz[candidate, 1], z=xyz[candidate, 2],
                mode="markers", marker={"size": 1.8, "color": "red"},
                name=f"导线候选 ({np.count_nonzero(candidate)})",
            )
        )
    _write_figure(fig, path)


def _page_19(run: ManualSingleSpanRun, path: Path) -> None:
    diag = _load_wire_diag(run)
    fig = go.Figure()
    if diag is not None and len(diag["s_m"]):
        rows = _even_indices(len(diag["s_m"]), _MAX_FINE)
        candidate = diag["candidate_mask"][rows]
        s = diag["s_m"][rows]
        u = diag["u_m"][rows]
        z = diag["z_m"][rows]
        fig.add_trace(
            go.Scatter(x=s[~candidate], y=u[~candidate], mode="markers",
                       marker={"size": 1.5, "color": "#90a4ae", "opacity": 0.5},
                       name="s-u 非候选")
        )
        fig.add_trace(
            go.Scatter(x=s[candidate], y=u[candidate], mode="markers",
                       marker={"size": 2.5, "color": "red"}, name="s-u 候选")
        )
        fig.add_trace(
            go.Scatter(x=s[~candidate], y=z[~candidate], mode="markers",
                       marker={"size": 1.5, "color": "#90a4ae", "opacity": 0.5},
                       name="s-z 非候选", xaxis="x2", yaxis="y2")
        )
        fig.add_trace(
            go.Scatter(x=s[candidate], y=z[candidate], mode="markers",
                       marker={"size": 2.5, "color": "orange"}, name="s-z 候选",
                       xaxis="x2", yaxis="y2")
        )
        fig.update_layout(
            title={"text": "19 导线 (s,u,z) 坐标（s-纵向, u-横向, z-高程）", "x": 0.5},
            height=700, template="plotly_white",
            xaxis={"title": "s (m)", "domain": [0, 0.48]},
            yaxis={"title": "u (m)"},
            xaxis2={"title": "s (m)", "domain": [0.52, 1.0], "anchor": "y2"},
            yaxis2={"title": "z (m)", "anchor": "x2"},
            legend={"orientation": "h", "x": 0, "y": 1.05},
            margin={"l": 50, "r": 40, "t": 90, "b": 40},
        )
    _write_figure(fig, path)


def _page_20(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("20 导线 DBSCAN 簇")
    diag = _load_wire_diag(run)
    if diag is not None and len(diag["fine_xyz_local_m"]):
        candidate_rows = np.flatnonzero(diag["candidate_mask"])
        sample = candidate_rows[_even_indices(len(candidate_rows), _MAX_FINE)]
        xyz = _abs(run, diag["fine_xyz_local_m"][sample])
        clusters = diag["cluster_id_by_rep"][sample]
        valid = clusters >= 0
        fig.add_trace(
            go.Scatter3d(
                x=xyz[~valid, 0], y=xyz[~valid, 1], z=xyz[~valid, 2],
                mode="markers", marker={"size": 1.4, "color": "#90a4ae", "opacity": 0.5},
                name="噪声",
            )
        )
        if np.any(valid):
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[valid, 0], y=xyz[valid, 1], z=xyz[valid, 2],
                    mode="markers", marker={"size": 2.0, "color": clusters[valid],
                                            "colorscale": "Rainbow"},
                    name=f"簇 ({np.unique(clusters[valid]).size})",
                )
            )
    _write_figure(fig, path)


def _page_21(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("21 导线线束（DBSCAN 后按 (u,z) 分离）")
    if run.wire_outcome is not None:
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        palette = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]
        for position, strand in enumerate(run.wire_outcome.strands):
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64) + origin
            fig.add_trace(
                go.Scatter3d(
                    x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                    mode="lines", line={"color": palette[position % len(palette)], "width": 5},
                    name=strand.wire_name,
                )
            )
    _write_figure(fig, path)


def _page_22(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("22 导线曲线拟合（稳健二次曲线）")
    if run.wire_outcome is not None:
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        for _position, strand in enumerate(run.wire_outcome.strands):
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64) + origin
            median, p95 = strand.residual_stats_m
            fig.add_trace(
                go.Scatter3d(
                    x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                    mode="lines", line={"width": 6},
                    name=f"{strand.wire_name} (残差 med={median:.2f}, p95={p95:.2f})",
                )
            )
    _write_figure(fig, path)


def _page_23(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("23 原始点云与重建导线叠加")
    diag = _load_wire_diag(run)
    if diag is not None and len(diag["fine_xyz_local_m"]):
        rows = _even_indices(len(diag["fine_xyz_local_m"]), _MAX_FINE)
        xyz = _abs(run, diag["fine_xyz_local_m"][rows])
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers", marker={"size": 1.3, "color": "#cfd8dc", "opacity": 0.5},
                name="导线区原始代表点",
            )
        )
    if run.wire_outcome is not None:
        origin = np.asarray(run.origin_xyz, dtype=np.float64)
        for strand in run.wire_outcome.strands:
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64) + origin
            fig.add_trace(
                go.Scatter3d(
                    x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                    mode="lines", line={"width": 5, "color": "red"},
                    name=strand.wire_name,
                )
            )
    _write_figure(fig, path)


def _page_24(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("24 树线最近点对（每树最小净空连线）")
    origin = np.asarray(run.origin_xyz, dtype=np.float64)
    if run.clearance is not None:
        for record in run.clearance.clearance.records:
            if record.nearest_tree_xyz_local_m is None or record.nearest_wire_xyz_local_m is None:
                continue
            tree_pt = np.asarray(record.nearest_tree_xyz_local_m) + origin
            wire_pt = np.asarray(record.nearest_wire_xyz_local_m) + origin
            fig.add_trace(
                go.Scatter3d(
                    x=[tree_pt[0], wire_pt[0]], y=[tree_pt[1], wire_pt[1]],
                    z=[tree_pt[2], wire_pt[2]],
                    mode="lines", line={"color": "orange", "width": 3},
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[tree_pt[0]], y=[tree_pt[1]], z=[tree_pt[2]],
                    mode="markers", marker={"size": 4, "color": "blue"},
                    showlegend=False,
                )
            )
    if run.wire_outcome is not None:
        for strand in run.wire_outcome.strands:
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64) + origin
            fig.add_trace(
                go.Scatter3d(
                    x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                    mode="lines", line={"width": 4, "color": "red"},
                    name=strand.wire_name,
                )
            )
    _write_figure(fig, path)


# ---------------------------------------------------------------------------
# Pages 25..26 : risk / final overview
# ---------------------------------------------------------------------------


def _page_25(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("25 危险净空阈值结果（红色=危险点/危险树）")
    origin = np.asarray(run.origin_xyz, dtype=np.float64)
    if run.wire_outcome is not None:
        for strand in run.wire_outcome.strands:
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64) + origin
            fig.add_trace(
                go.Scatter3d(
                    x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                    mode="lines", line={"width": 4, "color": "#9e9e9e"},
                    name=strand.wire_name,
                )
            )
    if run.risk is not None:
        danger_xyz = np.asarray(run.risk.danger_xyz_local_m, dtype=np.float64) + origin
        danger_xyz = danger_xyz[_even_indices(len(danger_xyz), _MAX_DANGER)]
        if len(danger_xyz):
            fig.add_trace(
                go.Scatter3d(
                    x=danger_xyz[:, 0], y=danger_xyz[:, 1], z=danger_xyz[:, 2],
                    mode="markers", marker={"size": 3.0, "color": "red"},
                    name=f"危险点 ({run.risk.danger_point_count})",
                )
            )
        # Dangerous trees as translucent red spheres at their centroids.

        for record in run.risk.records:
            if record.dangerous and record.tree_centroid_xyz_absolute_m is not None:
                centroid = np.asarray(record.tree_centroid_xyz_absolute_m, dtype=np.float64)
                fig.add_trace(
                    go.Scatter3d(
                        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                        mode="markers", marker={"size": 12, "color": "rgba(255,0,0,0.35)",
                                                "symbol": "circle"},
                        name=f"危险树 {record.global_tree_id + 1}",
                    )
                )
    _write_figure(fig, path)


def _page_26(run: ManualSingleSpanRun, path: Path) -> None:
    fig = _figure("26 最终档距总览（教师演示）")
    origin = np.asarray(run.origin_xyz, dtype=np.float64)
    if run.span is not None:
        core_min = np.asarray(run.span.core_bounds_min_xyz) + origin
        core_max = np.asarray(run.span.core_bounds_max_xyz) + origin
        _add_box(fig, core_min, core_max, "档距核心区", "#1f77b4")
        start = np.asarray(run.span.start_xyz_absolute_m, dtype=np.float64)
        end = np.asarray(run.span.end_xyz_absolute_m, dtype=np.float64)
        _add_lines(fig, np.asarray([start, end]), f"档距 {run.span.length_m:.1f} m", "black", 4)
        _add_markers(fig, np.asarray([start]), "Pole A", "red")
        _add_markers(fig, np.asarray([end]), "Pole B", "red")
    if run.wire_outcome is not None:
        for strand in run.wire_outcome.strands:
            curve = np.asarray(strand.curve_xyz_local_m, dtype=np.float64) + origin
            fig.add_trace(
                go.Scatter3d(
                    x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                    mode="lines", line={"width": 5, "color": "black"},
                    name=strand.wire_name,
                )
            )
    if run.reconciled is not None:
        final_xyz = run.reconciled.final_xyz_local_m + origin
        final_tree = run.reconciled.final_tree_id
        ids = sorted(set(int(value) for value in final_tree if value >= 0))
        palette = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
                   "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"]
        for position, tree_id in enumerate(ids):
            rows = np.flatnonzero(final_tree == tree_id)
            rows = rows[_even_indices(len(rows), 1500)]
            color = "red" if (run.risk and any(
                r.global_tree_id == tree_id and r.dangerous for r in run.risk.records
            )) else palette[position % len(palette)]
            xyz = final_xyz[rows]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    mode="markers", marker={"size": 1.3, "color": color, "opacity": 0.7},
                    name=f"树 {tree_id + 1}",
                )
            )
    if run.risk is not None:
        danger_xyz = np.asarray(run.risk.danger_xyz_local_m, dtype=np.float64) + origin
        danger_xyz = danger_xyz[_even_indices(len(danger_xyz), 20000)]
        if len(danger_xyz):
            fig.add_trace(
                go.Scatter3d(
                    x=danger_xyz[:, 0], y=danger_xyz[:, 1], z=danger_xyz[:, 2],
                    mode="markers", marker={"size": 2.5, "color": "red"},
                    name="危险点",
                )
            )
    _write_figure(fig, path)


# ---------------------------------------------------------------------------
# 00 index
# ---------------------------------------------------------------------------


def _write_visual_index(run: ManualSingleSpanRun, visual_root: Path) -> None:
    rows = []
    rows.append(
        "<tr><td>00</td><td><a href='00_VISUAL_INDEX.html'>可视化索引</a></td>"
        "<td>本页：全部阶段可视化导航</td></tr>"
    )
    for number, title in _MANUAL_STAGES:
        filename = f"{number}.html"
        rows.append(f"<tr><td>{number[:2]}</td><td><a href='{filename}'>{title}</a></td>"
                    f"<td>阶段 {number[:2]} 可视化</td></tr>")
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>00 可视化索引</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; background: #fafafa;
       color: #222; }}
h1 {{ color: #0b3d91; }} table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
th {{ background: #eef3fb; }}
tr:nth-child(even) {{ background: #f7f9fc; }}
a {{ color: #0b3d91; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
</style></head><body>
<h1>人工单档距分析 — 可视化索引</h1>
<p>源文件：<code>{run.source.name}</code>　SHA256：<code>{run.source_sha256[:16]}…</code></p>
<table><thead><tr><th>编号</th><th>页面</th><th>说明</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>
"""
    _write_text(visual_root / "00_VISUAL_INDEX.html", html)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_all_visuals(
    run: ManualSingleSpanRun,
    visual_root: Path,
    summary: dict[str, Any],
    *,
    progress_sink: _ProgressSink = None,
) -> Path:
    """Generate every stage visual (00..26) under *visual_root*.

    Filenames follow the owner contract exactly: ``01_original_cloud_manual_selection.html``
    through ``26_FINAL_SPAN_OVERVIEW.html``, plus ``00_VISUAL_INDEX.html``.
    """

    visual_root = Path(visual_root)
    visual_root.mkdir(parents=True, exist_ok=True)
    pages = [
        _page_01, _page_02, _page_03, _page_04, _page_05, _page_06, _page_07,
        _page_08, _page_09, _page_10, _page_11, _page_12, _page_13, _page_14,
        _page_15, _page_16, _page_17, _page_18, _page_19, _page_20, _page_21,
        _page_22, _page_23, _page_24, _page_25, _page_26,
    ]
    for (filename, _title), page in zip(_MANUAL_STAGES, pages, strict=True):
        html_name = f"{filename}.html"
        if progress_sink is not None:
            progress_sink(f"生成可视化 {html_name}")
        try:
            page(run, visual_root / html_name)
        except Exception as exc:  # pragma: no cover - diagnostics never gate
            _write_figure(_figure(f"{filename} 生成失败：{exc}"), visual_root / html_name)
    _write_visual_index(run, visual_root)
    return visual_root / "00_VISUAL_INDEX.html"


__all__ = ["generate_all_visuals"]
