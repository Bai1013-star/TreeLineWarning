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

ColorMode = Literal["RGB", "Height", "Classification"]


@dataclass(frozen=True)
class PreviewData:
    xyz: np.ndarray
    local_xyz: np.ndarray
    original_index: np.ndarray
    metadata: dict[str, Any]
    rgb: np.ndarray | None = None
    intensity: np.ndarray | None = None
    classification: np.ndarray | None = None


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


def figure_for_selection(
    preview: PreviewData,
    mask: np.ndarray,
    color_mode: ColorMode,
    min_xyz: list[float] | None = None,
    max_xyz: list[float] | None = None,
):
    """Build the Plotly 3D preview figure with a crop bounds point subset."""
    import plotly.graph_objects as go

    selected_xyz = preview.xyz[mask]
    selected_local = preview.local_xyz[mask]
    marker = {
        "size": 1.8,
        "opacity": 0.85,
    }
    if color_mode == "RGB" and preview.rgb is not None:
        marker["color"] = [
            f"rgb({int(red)},{int(green)},{int(blue)})" for red, green, blue in preview.rgb[mask]
        ]
    elif color_mode == "Classification" and preview.classification is not None:
        marker["color"] = preview.classification[mask]
        marker["colorbar"] = {"title": "Class"}
    else:
        marker["color"] = selected_xyz[:, 2] if len(selected_xyz) else []
        marker["colorbar"] = {"title": "Z (m)"}

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
        title="Preview crop range selector, not original full-density LAS",
        scene={
            "xaxis_title": "Local X (m)",
            "yaxis_title": "Local Y (m)",
            "zaxis_title": "Local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return figure


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
        line={"color": "black", "width": 5},
        name="Crop bounds",
        hoverinfo="skip",
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="TreeLine Crop Selector", layout="wide")
    st.title("Low-density preview crop selector")
    st.warning("This selects bounds on preview data only. It has not cropped the original LAS.")

    preview = load_preview_data(DEFAULT_PREVIEW_NPZ, DEFAULT_PREVIEW_JSON)
    original_range = preview.metadata["original_xyz_range"]
    min_range = [float(value) for value in original_range["min"]]
    max_range = [float(value) for value in original_range["max"]]

    st.subheader("Original XYZ range (source coordinates, meters)")
    st.json({"min_xyz": min_range, "max_xyz": max_range})

    with st.sidebar:
        st.header("Crop Bounds")
        st.caption("Suggested area: inspect the middle-right section, but choose manually.")
        x_min = st.number_input("X min", value=min_range[0], step=1.0, format="%.3f")
        x_max = st.number_input("X max", value=max_range[0], step=1.0, format="%.3f")
        y_min = st.number_input("Y min", value=min_range[1], step=1.0, format="%.3f")
        y_max = st.number_input("Y max", value=max_range[1], step=1.0, format="%.3f")
        z_min = st.number_input("Z min", value=min_range[2], step=0.5, format="%.3f")
        z_max = st.number_input("Z max", value=max_range[2], step=0.5, format="%.3f")
        color_mode = st.radio("Color mode", ["RGB", "Height", "Classification"], horizontal=False)
        note = st.text_area("Note", value="Candidate range for scene_simple.")

    min_xyz = [x_min, y_min, z_min]
    max_xyz = [x_max, y_max, z_max]
    st.info(
        "Selection guidance: choose a 20-35 m line section with multiple intact parallel wires "
        "and some vegetation; avoid the densest left-side pole/mixed area; do not select the "
        "entire scene or only wires without trees."
    )

    try:
        mask = filter_preview_points(preview.xyz, min_xyz, max_xyz)
        selected_count = int(np.count_nonzero(mask))
        ratio = 0.0 if len(preview.xyz) == 0 else selected_count / len(preview.xyz)
        st.metric("Selected preview points", selected_count)
        st.metric("Selected preview ratio", f"{ratio:.2%}")
        st.json(crop_bounds_json(min_xyz, max_xyz))
        if selected_count == 0:
            st.warning("Current bounds select zero preview points.")
        st.plotly_chart(
            figure_for_selection(preview, mask, color_mode, min_xyz, max_xyz),
            use_container_width=True,
        )

        record = selection_record(
            preview,
            min_xyz,
            max_xyz,
            selected_count,
            note=note,
        )
        if st.button("Save scene_simple bounds"):
            save_selection_record(DEFAULT_SELECTION_PATH, record, overwrite=False)
            st.success(f"Saved: {DEFAULT_SELECTION_PATH}")
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
