"""Streamlit crop bounds selector for the low-density scan_small preview."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIEW_NPZ = PROJECT_ROOT / "outputs" / "previews" / "scan_small_preview.npz"
DEFAULT_PREVIEW_JSON = PROJECT_ROOT / "outputs" / "previews" / "scan_small_preview.json"
DEFAULT_SELECTION_PATH = PROJECT_ROOT / "outputs" / "selections" / "scene_simple_bounds.json"

ColorMode = Literal["增强 RGB", "高度", "Classification", "Intensity"]
DEFAULT_COLOR_MODE: ColorMode = "高度"
DEFAULT_GAMMA = 0.8
WHOLE_SCENE_WARNING_THRESHOLD = 0.8


@dataclass(frozen=True)
class PreviewData:
    xyz: np.ndarray
    local_xyz: np.ndarray
    original_index: np.ndarray
    metadata: dict[str, Any]
    rgb: np.ndarray | None = None
    intensity: np.ndarray | None = None
    classification: np.ndarray | None = None


@dataclass(frozen=True)
class RgbEnhancement:
    colors: list[str] | None
    stats: dict[str, Any]


def load_preview_data(npz_path: Path, json_path: Path) -> PreviewData:
    """Load low-density preview artifacts without touching the original LAS."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Preview NPZ not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Preview JSON not found: {json_path}")

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        local_xyz = np.asarray(data["local_xyz"], dtype=np.float64)
        original_index = np.asarray(data["original_index"], dtype=np.int64)
        rgb = np.asarray(data["rgb"]) if "rgb" in data else None
        intensity = np.asarray(data["intensity"]) if "intensity" in data else None
        classification = np.asarray(data["classification"]) if "classification" in data else None

    return PreviewData(
        xyz=xyz,
        local_xyz=local_xyz,
        original_index=original_index,
        metadata=metadata,
        rgb=rgb,
        intensity=intensity,
        classification=classification,
    )


def validate_xyz_bounds(min_xyz: list[float], max_xyz: list[float]) -> None:
    """Validate each min coordinate is strictly less than its paired max."""
    if len(min_xyz) != 3 or len(max_xyz) != 3:
        raise ValueError("min_xyz and max_xyz must each contain exactly three values")
    for axis, minimum, maximum in zip("XYZ", min_xyz, max_xyz, strict=True):
        if minimum >= maximum:
            raise ValueError(f"{axis} min must be less than {axis} max")


def filter_preview_points(
    xyz: np.ndarray,
    min_xyz: list[float],
    max_xyz: list[float],
) -> np.ndarray:
    """Return an inclusive XYZ bounds mask for preview points."""
    validate_xyz_bounds(min_xyz, max_xyz)
    minimum = np.asarray(min_xyz, dtype=np.float64)
    maximum = np.asarray(max_xyz, dtype=np.float64)
    return np.all((xyz >= minimum) & (xyz <= maximum), axis=1)


def crop_bounds_json(
    min_xyz: list[float],
    max_xyz: list[float],
    label: str = "scene_simple",
) -> dict:
    """Return the copyable CropBounds JSON payload."""
    validate_xyz_bounds(min_xyz, max_xyz)
    return {
        "min_xyz": [float(value) for value in min_xyz],
        "max_xyz": [float(value) for value in max_xyz],
        "coordinate_mode": "source",
        "label": label,
    }


def selection_record(
    preview: PreviewData,
    min_xyz: list[float],
    max_xyz: list[float],
    selected_count: int,
    *,
    note: str,
    label: str = "scene_simple",
) -> dict[str, Any]:
    """Create the persisted selection JSON record."""
    bounds = crop_bounds_json(min_xyz, max_xyz, label=label)
    preview_count = int(len(preview.xyz))
    return {
        "source_id": preview.metadata.get("source_id"),
        "source_path": preview.metadata.get("source_path"),
        "created_at": datetime.now(UTC).isoformat(),
        "preview_point_count": preview_count,
        "selected_preview_point_count": int(selected_count),
        "estimated_keep_ratio": 0.0 if preview_count == 0 else selected_count / preview_count,
        "min_xyz": bounds["min_xyz"],
        "max_xyz": bounds["max_xyz"],
        "coordinate_mode": bounds["coordinate_mode"],
        "label": bounds["label"],
        "note": note,
    }


def save_selection_record(
    output_path: Path,
    record: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    overwrite: bool = False,
) -> None:
    """Save selection JSON under outputs/selections only."""
    output = output_path.resolve(strict=False)
    allowed_root = (project_root / "outputs" / "selections").resolve(strict=False)
    if not _is_relative_to(output, allowed_root):
        raise ValueError("Selection output must be under outputs/selections")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Selection output already exists: {output_path}")
    validate_xyz_bounds(record["min_xyz"], record["max_xyz"])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f"{output.stem}.tmp{output.suffix}")
    try:
        tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(output)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def selection_warning(selected_ratio: float) -> str | None:
    if selected_ratio > WHOLE_SCENE_WARNING_THRESHOLD:
        return "当前范围接近整个场景，不建议作为 scene_simple"
    return None


def enhance_rgb(rgb: np.ndarray | None, gamma: float = DEFAULT_GAMMA) -> RgbEnhancement:
    """Contrast-stretch uint8/uint16 RGB to 0-255 Plotly colors."""
    if rgb is None:
        return RgbEnhancement(None, {"valid": False, "reason": "missing_rgb"})
    values = np.asarray(rgb)
    stats: dict[str, Any] = {
        "valid": False,
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "gamma": gamma,
        "method": "per_channel_1_99_percentile_gamma",
    }
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        stats["reason"] = "invalid_shape_or_empty"
        return RgbEnhancement(None, stats)

    rgb_float = values.astype(np.float64)
    stats["min"] = [float(value) for value in np.nanmin(rgb_float, axis=0)]
    stats["max"] = [float(value) for value in np.nanmax(rgb_float, axis=0)]
    if not np.all(np.isfinite(rgb_float)):
        stats["reason"] = "non_finite_rgb"
        return RgbEnhancement(None, stats)

    low = np.percentile(rgb_float, 1, axis=0)
    high = np.percentile(rgb_float, 99, axis=0)
    dynamic = high - low
    if np.all(dynamic <= 0):
        stats["reason"] = "no_rgb_contrast"
        return RgbEnhancement(None, stats)
    safe_dynamic = np.where(dynamic <= 0, 1.0, dynamic)
    stretched = np.clip((rgb_float - low) / safe_dynamic, 0.0, 1.0)
    gamma_value = max(float(gamma), 0.01)
    adjusted = np.power(stretched, gamma_value)
    rgb_255 = np.rint(adjusted * 255.0).astype(np.uint8)
    stats["valid"] = True
    stats["percentile_1"] = [float(value) for value in low]
    stats["percentile_99"] = [float(value) for value in high]
    stats["output_min"] = [int(value) for value in rgb_255.min(axis=0)]
    stats["output_max"] = [int(value) for value in rgb_255.max(axis=0)]
    colors = [f"rgb({int(red)},{int(green)},{int(blue)})" for red, green, blue in rgb_255]
    return RgbEnhancement(colors, stats)


def figure_for_selection(
    preview: PreviewData,
    mask: np.ndarray,
    color_mode: ColorMode,
    min_xyz: list[float] | None = None,
    max_xyz: list[float] | None = None,
    *,
    point_size: float = 2.0,
    opacity: float = 0.85,
    gamma: float = DEFAULT_GAMMA,
):
    """Build the Plotly 3D preview figure with a crop bounds point subset."""
    import plotly.graph_objects as go

    selected_xyz = preview.xyz[mask]
    selected_local = preview.local_xyz[mask]
    marker = _marker_for_mode(preview, mask, color_mode, point_size, opacity, gamma)

    hover = [f"X={x:.3f} m<br>Y={y:.3f} m<br>Z={z:.3f} m" for x, y, z in selected_xyz]
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=selected_local[:, 0] if len(selected_local) else [],
                y=selected_local[:, 1] if len(selected_local) else [],
                z=selected_local[:, 2] if len(selected_local) else [],
                mode="markers",
                marker=marker,
                text=hover,
                hoverinfo="text",
                name="Selected preview points",
            )
        ]
    )
    if min_xyz is not None and max_xyz is not None:
        figure.add_trace(_bbox_trace(preview, min_xyz, max_xyz))
    figure.update_layout(
        title="预览裁剪范围选择器：仅低密度预览，不是原始全密度 LAS",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font={"color": "#f0f0f0"},
        scene={
            "xaxis_title": "局部 X (m)",
            "yaxis_title": "局部 Y (m)",
            "zaxis_title": "局部 Z (m)",
            "bgcolor": "#151515",
            "xaxis": {"gridcolor": "#666666", "zerolinecolor": "#888888"},
            "yaxis": {"gridcolor": "#666666", "zerolinecolor": "#888888"},
            "zaxis": {"gridcolor": "#666666", "zerolinecolor": "#888888"},
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.9}},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=True,
    )
    return figure


def projection_figure(
    preview: PreviewData,
    mask: np.ndarray,
    plane: Literal["XY", "XZ", "YZ"],
    color_mode: ColorMode,
    min_xyz: list[float],
    max_xyz: list[float],
    *,
    point_size: float = 2.0,
    opacity: float = 0.85,
    gamma: float = DEFAULT_GAMMA,
    max_points: int = 50_000,
):
    """Build a 2D projection figure for selected preview points."""
    import plotly.graph_objects as go

    selected_indices = np.flatnonzero(mask)
    if len(selected_indices) > max_points:
        stride = int(np.ceil(len(selected_indices) / max_points))
        selected_indices = selected_indices[::stride]
    xyz = preview.xyz[selected_indices]
    projection = projection_coordinates(xyz, plane)
    projection_mask = np.zeros(len(preview.xyz), dtype=bool)
    projection_mask[selected_indices] = True
    marker = _marker_for_mode(preview, projection_mask, color_mode, point_size, opacity, gamma)
    figure = go.Figure(
        data=[
            go.Scattergl(
                x=projection[:, 0] if len(projection) else [],
                y=projection[:, 1] if len(projection) else [],
                mode="markers",
                marker=marker,
                name=f"{plane} 预览点",
            )
        ]
    )
    figure.add_trace(_projection_rectangle_trace(plane, min_xyz, max_xyz))
    axis_titles = {
        "XY": ("X (m)", "Y (m)"),
        "XZ": ("X (m)", "Z (m)"),
        "YZ": ("Y (m)", "Z (m)"),
    }
    x_title, y_title = axis_titles[plane]
    figure.update_layout(
        title=f"{plane} 投影视图",
        paper_bgcolor="#111111",
        plot_bgcolor="#151515",
        font={"color": "#f0f0f0"},
        xaxis={"title": x_title, "gridcolor": "#666666", "zerolinecolor": "#888888"},
        yaxis={"title": y_title, "gridcolor": "#666666", "zerolinecolor": "#888888"},
        margin={"l": 20, "r": 10, "t": 35, "b": 30},
        showlegend=True,
    )
    return figure


def projection_coordinates(
    xyz: np.ndarray,
    plane: Literal["XY", "XZ", "YZ"],
) -> np.ndarray:
    if plane == "XY":
        return xyz[:, [0, 1]]
    if plane == "XZ":
        return xyz[:, [0, 2]]
    return xyz[:, [1, 2]]


def _marker_for_mode(
    preview: PreviewData,
    mask: np.ndarray,
    color_mode: ColorMode,
    point_size: float,
    opacity: float,
    gamma: float,
) -> dict[str, Any]:
    selected_xyz = preview.xyz[mask]
    marker: dict[str, Any] = {
        "size": point_size,
        "opacity": opacity,
    }
    if color_mode == "增强 RGB":
        enhanced = enhance_rgb(preview.rgb[mask] if preview.rgb is not None else None, gamma=gamma)
        if enhanced.colors is not None:
            marker["color"] = enhanced.colors
            return marker
    if color_mode == "Classification" and preview.classification is not None:
        marker["color"] = preview.classification[mask]
        marker["colorscale"] = "Turbo"
        marker["colorbar"] = {"title": "Class"}
    elif color_mode == "Intensity" and preview.intensity is not None:
        marker["color"] = preview.intensity[mask]
        marker["colorscale"] = "Viridis"
        marker["colorbar"] = {"title": "Intensity"}
    else:
        marker["color"] = selected_xyz[:, 2] if len(selected_xyz) else []
        marker["colorscale"] = "Turbo"
        marker["colorbar"] = {"title": "Z (m)"}
    return marker


def _bbox_trace(preview: PreviewData, min_xyz: list[float], max_xyz: list[float]):
    import plotly.graph_objects as go

    origin = preview.xyz[0] - preview.local_xyz[0] if len(preview.xyz) else np.zeros(3)
    minimum = np.asarray(min_xyz, dtype=np.float64) - origin
    maximum = np.asarray(max_xyz, dtype=np.float64) - origin
    corners = np.array(
        [
            [minimum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], minimum[2]],
            [maximum[0], maximum[1], minimum[2]],
            [minimum[0], maximum[1], minimum[2]],
            [minimum[0], minimum[1], minimum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [maximum[0], maximum[1], maximum[2]],
            [minimum[0], maximum[1], maximum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [None, None, None],
            [maximum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [None, None, None],
            [maximum[0], maximum[1], minimum[2]],
            [maximum[0], maximum[1], maximum[2]],
            [None, None, None],
            [minimum[0], maximum[1], minimum[2]],
            [minimum[0], maximum[1], maximum[2]],
        ],
        dtype=object,
    )
    return go.Scatter3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=corners[:, 2],
        mode="lines",
        line={"color": "#ffb000", "width": 7},
        name="裁剪包围盒",
        hoverinfo="skip",
    )


def _projection_rectangle_trace(plane: str, min_xyz: list[float], max_xyz: list[float]):
    import plotly.graph_objects as go

    if plane == "XY":
        x_min, y_min = min_xyz[0], min_xyz[1]
        x_max, y_max = max_xyz[0], max_xyz[1]
    elif plane == "XZ":
        x_min, y_min = min_xyz[0], min_xyz[2]
        x_max, y_max = max_xyz[0], max_xyz[2]
    else:
        x_min, y_min = min_xyz[1], min_xyz[2]
        x_max, y_max = max_xyz[1], max_xyz[2]
    return go.Scatter(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        mode="lines",
        line={"color": "#ffb000", "width": 3},
        name="选择范围",
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="树线裁剪范围选择器", layout="wide")
    st.title("低密度点云裁剪范围选择器")
    st.warning("当前只是在预览数据中选择范围，尚未裁剪原始 LAS。")

    preview = load_preview_data(DEFAULT_PREVIEW_NPZ, DEFAULT_PREVIEW_JSON)
    original_range = preview.metadata["original_xyz_range"]
    min_range = [float(value) for value in original_range["min"]]
    max_range = [float(value) for value in original_range["max"]]

    st.subheader("原始 XYZ 范围（source 原始坐标，单位：米）")
    st.json({"min_xyz": min_range, "max_xyz": max_range})

    with st.sidebar:
        st.header("裁剪范围")
        st.caption("建议重点观察场景中间偏右区域，但最终范围由项目组手动确定。")
        x_min = st.number_input("X 最小值", value=min_range[0], step=1.0, format="%.3f")
        x_max = st.number_input("X 最大值", value=max_range[0], step=1.0, format="%.3f")
        y_min = st.number_input("Y 最小值", value=min_range[1], step=1.0, format="%.3f")
        y_max = st.number_input("Y 最大值", value=max_range[1], step=1.0, format="%.3f")
        z_min = st.number_input("Z 最小值", value=min_range[2], step=0.5, format="%.3f")
        z_max = st.number_input("Z 最大值", value=max_range[2], step=0.5, format="%.3f")
        color_mode = st.radio(
            "显示模式",
            ["高度", "增强 RGB", "Classification", "Intensity"],
            index=0,
            horizontal=False,
        )
        point_size = st.slider("点大小", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        opacity = st.slider("透明度", min_value=0.2, max_value=1.0, value=0.85, step=0.05)
        gamma = st.slider("RGB Gamma", min_value=0.5, max_value=1.5, value=DEFAULT_GAMMA, step=0.05)
        allow_overwrite = st.checkbox("允许覆盖已有选择文件", value=False)
        note = st.text_area("备注", value="Candidate range for scene_simple.")

    min_xyz = [x_min, y_min, z_min]
    max_xyz = [x_max, y_max, z_max]
    st.info(
        "选择建议：优先选择多根平行导线较完整的区段；下方至少包含一片植被或两三处树木区域；"
        "第一版建议线路长度约 20～35 米；尽量避开最左侧点云特别密集的杆塔或混杂区域；"
        "不要选择整个场景；不要只选择电线而没有树木。"
    )

    try:
        mask = filter_preview_points(preview.xyz, min_xyz, max_xyz)
        selected_count = int(np.count_nonzero(mask))
        ratio = 0.0 if len(preview.xyz) == 0 else selected_count / len(preview.xyz)
        st.metric("当前选中预览点数", selected_count)
        st.metric("选中点比例", f"{ratio:.2%}")
        warning = selection_warning(ratio)
        if warning:
            st.warning(warning)
        if preview.rgb is not None:
            rgb_stats = enhance_rgb(preview.rgb, gamma=gamma).stats
            st.caption("RGB 原始统计与增强方法")
            st.json(rgb_stats)
        st.json(crop_bounds_json(min_xyz, max_xyz))
        if selected_count == 0:
            st.warning("当前范围未选中任何预览点。")
        st.plotly_chart(
            figure_for_selection(
                preview,
                mask,
                color_mode,
                min_xyz,
                max_xyz,
                point_size=point_size,
                opacity=opacity,
                gamma=gamma,
            ),
            use_container_width=True,
        )
        st.subheader("二维投影视图")
        xy_col, xz_col, yz_col = st.columns(3)
        with xy_col:
            st.plotly_chart(
                projection_figure(
                    preview,
                    mask,
                    "XY",
                    color_mode,
                    min_xyz,
                    max_xyz,
                    point_size=point_size,
                    opacity=opacity,
                    gamma=gamma,
                ),
                use_container_width=True,
            )
        with xz_col:
            st.plotly_chart(
                projection_figure(
                    preview,
                    mask,
                    "XZ",
                    color_mode,
                    min_xyz,
                    max_xyz,
                    point_size=point_size,
                    opacity=opacity,
                    gamma=gamma,
                ),
                use_container_width=True,
            )
        with yz_col:
            st.plotly_chart(
                projection_figure(
                    preview,
                    mask,
                    "YZ",
                    color_mode,
                    min_xyz,
                    max_xyz,
                    point_size=point_size,
                    opacity=opacity,
                    gamma=gamma,
                ),
                use_container_width=True,
            )

        record = selection_record(
            preview,
            min_xyz,
            max_xyz,
            selected_count,
            note=note,
        )
        if st.button("保存裁剪范围"):
            save_selection_record(DEFAULT_SELECTION_PATH, record, overwrite=allow_overwrite)
            st.success(f"已保存：{DEFAULT_SELECTION_PATH}")
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
