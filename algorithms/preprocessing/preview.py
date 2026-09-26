"""Streamed low-density point cloud preview generation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import CancellationToken, CropBounds, PipelineProgress, ProgressReporter

from algorithms.io import DEFAULT_CHUNK_SIZE, iter_las_chunks, read_las_header_summary
from algorithms.io.headers import _error

DEFAULT_PREVIEW_POINTS = 200_000
DEFAULT_MAX_PREVIEW_POINTS = 300_000
PREVIEW_ATTRIBUTE_NAMES = (
    "red",
    "green",
    "blue",
    "intensity",
    "classification",
    "NormalX",
    "NormalY",
    "NormalZ",
)


@dataclass(frozen=True)
class PreviewResult:
    """Paths and metadata for a generated preview artifact set."""

    npz_path: Path
    json_path: Path
    html_path: Path | None
    metadata: dict[str, Any]


def build_pointcloud_preview(
    source_path: Path | str,
    output_dir: Path | str,
    *,
    source_id: str,
    target_preview_points: int = DEFAULT_PREVIEW_POINTS,
    max_preview_points: int = DEFAULT_MAX_PREVIEW_POINTS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    generate_html: bool = True,
    crop_bounds: CropBounds | None = None,
    project_root: Path | str | None = None,
) -> PreviewResult:
    """Build deterministic streamed preview NPZ/JSON and optional Plotly HTML."""
    source = Path(source_path)
    output = Path(output_dir)
    _validate_output_dir(output, project_root=project_root)
    _validate_preview_limits(target_preview_points, max_preview_points)

    basename = source.stem
    npz_path = output / f"{basename}_preview.npz"
    json_path = output / f"{basename}_preview.json"
    html_path = output / f"{basename}_preview.html"
    final_paths = [npz_path, json_path]
    if generate_html and _plotly_available():
        final_paths.append(html_path)
    _validate_existing_outputs(final_paths, overwrite)

    started = time.perf_counter()
    process = _current_process()
    start_memory = _rss_bytes(process)
    peak_memory = start_memory
    temp_paths = [_temporary_path(path) for path in (npz_path, json_path, html_path)]

    try:
        output.mkdir(parents=True, exist_ok=True)
        for path in temp_paths:
            if path.exists():
                path.unlink()

        header = read_las_header_summary(source, source_id=source_id)
        total_points = header.point_count
        stride = _deterministic_stride(total_points, target_preview_points, max_preview_points)
        arrays = _PreviewAccumulator(total_points=total_points, stride=stride)

        for chunk in iter_las_chunks(
            source,
            source_id=source_id,
            chunk_size=chunk_size,
            cancellation_token=cancellation_token,
        ):
            if _cancelled(cancellation_token):
                raise _error(
                    code="PREVIEW_CANCELLED",
                    message="Preview generation cancelled before chunk processing",
                    details={"source_path": str(source)},
                )
            selected = (chunk.original_index % stride) == 0
            selected_count = int(np.count_nonzero(selected))
            if selected_count:
                selected_xyz = chunk.xyz[selected]
                arrays.add_xyz(selected_xyz)
                arrays.add_original_index(chunk.original_index[selected])
                for name in PREVIEW_ATTRIBUTE_NAMES:
                    value = chunk.attributes.get(name)
                    if value is not None:
                        arrays.add_attribute(name, np.asarray(value)[selected])

            current_memory = _rss_bytes(process)
            if current_memory is not None:
                peak_memory = (
                    current_memory
                    if peak_memory is None
                    else max(peak_memory, current_memory)
                )
            _report(
                reporter,
                source_id,
                chunk.start_index + chunk.point_count,
                total_points,
                chunk.chunk_id,
            )
            if _cancelled(cancellation_token):
                raise _error(
                    code="PREVIEW_CANCELLED",
                    message="Preview generation cancelled after chunk processing",
                    details={"source_path": str(source)},
                )

        data = arrays.finalize(max_preview_points=max_preview_points)
        origin_xyz = _origin_for(data["xyz"], header.min_xyz)
        data["local_xyz"] = data["xyz"] - origin_xyz
        data["origin_xyz"] = origin_xyz
        data["source_id"] = np.asarray(source_id)

        warnings = []
        if total_points == 0:
            warnings.append("Source LAS contains zero points")
        if len(data["xyz"]) == 0:
            warnings.append("Preview contains zero points")

        elapsed_seconds = time.perf_counter() - started
        metadata = _metadata(
            source=source,
            header=header,
            preview_xyz=data["xyz"],
            preview_count=len(data["xyz"]),
            stride=stride,
            chunk_size=chunk_size,
            elapsed_seconds=elapsed_seconds,
            start_memory=start_memory,
            peak_memory=peak_memory,
            warnings=warnings,
            npz_path=npz_path,
            json_path=json_path,
            html_path=html_path if generate_html and _plotly_available() else None,
            target_preview_points=target_preview_points,
            max_preview_points=max_preview_points,
            crop_bounds=crop_bounds,
        )

        np.savez_compressed(_temporary_path(npz_path), **data)
        _write_json_atomic(_temporary_path(json_path), metadata)
        html_output: Path | None = None
        if generate_html and _plotly_available():
            _write_plotly_html(_temporary_path(html_path), data, metadata, crop_bounds=crop_bounds)
            html_output = html_path

        _replace(_temporary_path(npz_path), npz_path, overwrite=overwrite)
        _replace(_temporary_path(json_path), json_path, overwrite=overwrite)
        if html_output is not None:
            _replace(_temporary_path(html_path), html_path, overwrite=overwrite)

        return PreviewResult(
            npz_path=npz_path,
            json_path=json_path,
            html_path=html_output,
            metadata=metadata,
        )
    except Exception:
        for path in temp_paths + [npz_path, json_path, html_path]:
            if path.exists() and path.name.endswith(".tmp"):
                path.unlink()
        for path in temp_paths:
            if path.exists():
                path.unlink()
        raise


class _PreviewAccumulator:
    def __init__(self, *, total_points: int, stride: int) -> None:
        self.total_points = total_points
        self.stride = stride
        self.xyz_parts: list[np.ndarray] = []
        self.original_index_parts: list[np.ndarray] = []
        self.attribute_parts: dict[str, list[np.ndarray]] = {}

    def add_xyz(self, value: np.ndarray) -> None:
        self.xyz_parts.append(np.asarray(value, dtype=np.float64))

    def add_original_index(self, value: np.ndarray) -> None:
        self.original_index_parts.append(np.asarray(value, dtype=np.int64))

    def add_attribute(self, name: str, value: np.ndarray) -> None:
        self.attribute_parts.setdefault(name, []).append(np.asarray(value))

    def finalize(self, *, max_preview_points: int) -> dict[str, np.ndarray]:
        xyz = _concat_or_empty(self.xyz_parts, shape=(0, 3), dtype=np.float64)
        original_index = _concat_or_empty(self.original_index_parts, shape=(0,), dtype=np.int64)
        if len(xyz) > max_preview_points:
            xyz = xyz[:max_preview_points]
            original_index = original_index[:max_preview_points]

        data = {
            "xyz": xyz,
            "original_index": original_index,
        }
        for name, parts in self.attribute_parts.items():
            values = _concat_or_empty(parts, shape=(0,), dtype=parts[0].dtype)
            data[name] = values[: len(xyz)]
        if {"red", "green", "blue"}.issubset(data):
            data["rgb"] = np.column_stack((data["red"], data["green"], data["blue"])).astype(
                np.uint16,
                copy=False,
            )
        return data


def _deterministic_stride(
    total_points: int,
    target_preview_points: int,
    max_preview_points: int,
) -> int:
    if total_points <= 0:
        return 1
    desired = min(target_preview_points, max_preview_points)
    return max(1, int(np.ceil(total_points / desired)))


def _origin_for(xyz: np.ndarray, fallback_min_xyz: tuple[float, float, float]) -> np.ndarray:
    if len(xyz) == 0:
        return np.asarray(fallback_min_xyz, dtype=np.float64)
    return xyz.min(axis=0).astype(np.float64)


def _metadata(
    *,
    source: Path,
    header: Any,
    preview_xyz: np.ndarray,
    preview_count: int,
    stride: int,
    chunk_size: int,
    elapsed_seconds: float,
    start_memory: int | None,
    peak_memory: int | None,
    warnings: list[str],
    npz_path: Path,
    json_path: Path,
    html_path: Path | None,
    target_preview_points: int,
    max_preview_points: int,
    crop_bounds: CropBounds | None,
) -> dict[str, Any]:
    original_count = int(header.point_count)
    return {
        "source_path": str(source),
        "source_id": header.source_id,
        "original_point_count": original_count,
        "preview_point_count": int(preview_count),
        "sampling_ratio": 0.0 if original_count == 0 else preview_count / original_count,
        "original_xyz_range": {
            "min": list(header.min_xyz),
            "max": list(header.max_xyz),
        },
        "preview_xyz_range": _range_dict(preview_xyz),
        "downsampling": {
            "method": "deterministic_global_stride",
            "stride": stride,
            "target_preview_points": target_preview_points,
            "max_preview_points": max_preview_points,
        },
        "chunk_size": chunk_size,
        "elapsed_seconds": elapsed_seconds,
        "memory": {
            "rss_start_bytes": start_memory,
            "rss_peak_bytes": peak_memory,
        },
        "warnings": warnings,
        "outputs": {
            "npz": str(npz_path),
            "json": str(json_path),
            "html": str(html_path) if html_path is not None else None,
        },
        "plotly_available": html_path is not None,
        "crop_bounds": crop_bounds.to_dict() if crop_bounds is not None else None,
    }


def _range_dict(xyz: np.ndarray) -> dict[str, list[float] | None]:
    if len(xyz) == 0:
        return {"min": None, "max": None}
    return {
        "min": [float(value) for value in xyz.min(axis=0)],
        "max": [float(value) for value in xyz.max(axis=0)],
    }


def _write_json_atomic(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_plotly_html(
    path: Path,
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    crop_bounds: CropBounds | None,
) -> None:
    import plotly.graph_objects as go

    xyz = data["local_xyz"]
    source_xyz = data["xyz"]
    height_marker = {
        "size": 1.8,
        "color": source_xyz[:, 2] if len(source_xyz) else [],
        "colorscale": "Turbo",
        "opacity": 0.85,
        "colorbar": {"title": "Height Z (m)"},
    }

    hover = [
        f"X={x:.3f} m<br>Y={y:.3f} m<br>Z={z:.3f} m"
        for x, y, z in source_xyz
    ]
    traces = [
        go.Scatter3d(
            x=xyz[:, 0] if len(xyz) else [],
            y=xyz[:, 1] if len(xyz) else [],
            z=xyz[:, 2] if len(xyz) else [],
            mode="markers",
            marker=height_marker,
            text=hover,
            hoverinfo="text",
            name="Height",
            visible=True,
        )
    ]
    buttons = [
        {
            "label": "Height",
            "method": "update",
            "args": [{"visible": _visibility(0, 1 + _optional_trace_count(data) + 1)}],
        }
    ]
    if "rgb" in data:
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0] if len(xyz) else [],
                y=xyz[:, 1] if len(xyz) else [],
                z=xyz[:, 2] if len(xyz) else [],
                mode="markers",
                marker={
                    "size": 1.8,
                    "color": _enhanced_rgb_colors(np.asarray(data["rgb"])),
                    "opacity": 0.85,
                },
                text=hover,
                hoverinfo="text",
                name="Enhanced RGB",
                visible=False,
            )
        )
        buttons.append(
            {
                "label": "Enhanced RGB",
                "method": "update",
                "args": [
                    {
                        "visible": _visibility(
                            len(traces) - 1,
                            1 + _optional_trace_count(data) + 1,
                        )
                    }
                ],
            }
        )
    if "classification" in data:
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0] if len(xyz) else [],
                y=xyz[:, 1] if len(xyz) else [],
                z=xyz[:, 2] if len(xyz) else [],
                mode="markers",
                marker={
                    "size": 1.8,
                    "color": np.asarray(data["classification"]),
                    "colorscale": "Turbo",
                    "opacity": 0.85,
                    "colorbar": {"title": "Classification"},
                },
                text=hover,
                hoverinfo="text",
                name="Classification",
                visible=False,
            )
        )
        buttons.append(
            {
                "label": "Classification",
                "method": "update",
                "args": [
                    {
                        "visible": _visibility(
                            len(traces) - 1,
                            1 + _optional_trace_count(data) + 1,
                        )
                    }
                ],
            }
        )
    if "intensity" in data:
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0] if len(xyz) else [],
                y=xyz[:, 1] if len(xyz) else [],
                z=xyz[:, 2] if len(xyz) else [],
                mode="markers",
                marker={
                    "size": 1.8,
                    "color": np.asarray(data["intensity"]),
                    "colorscale": "Viridis",
                    "opacity": 0.85,
                    "colorbar": {"title": "Intensity"},
                },
                text=hover,
                hoverinfo="text",
                name="Intensity",
                visible=False,
            )
        )
        buttons.append(
            {
                "label": "Intensity",
                "method": "update",
                "args": [
                    {
                        "visible": _visibility(
                            len(traces) - 1,
                            1 + _optional_trace_count(data) + 1,
                        )
                    }
                ],
            }
        )
    if crop_bounds is not None:
        traces.append(_crop_bounds_trace(data, crop_bounds))
        for button in buttons:
            button["args"][0]["visible"][-1] = True

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=f"{Path(str(metadata['source_path'])).stem} low-density preview, not full density",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font={"color": "#f0f0f0"},
        scene={
            "xaxis_title": "Local X (m)",
            "yaxis_title": "Local Y (m)",
            "zaxis_title": "Local Z (m)",
            "bgcolor": "#151515",
            "xaxis": {"gridcolor": "#666666", "zerolinecolor": "#888888"},
            "yaxis": {"gridcolor": "#666666", "zerolinecolor": "#888888"},
            "zaxis": {"gridcolor": "#666666", "zerolinecolor": "#888888"},
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.9}},
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
            }
        ],
        annotations=[
            {
                "text": (
                    f"Preview points: {metadata['preview_point_count']} / "
                    f"{metadata['original_point_count']}; default mode: Height"
                ),
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1,
            }
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)


def _optional_trace_count(data: dict[str, np.ndarray]) -> int:
    count = 0
    for name in ("rgb", "classification", "intensity"):
        if name in data:
            count += 1
    return count


def _visibility(active_index: int, trace_count: int) -> list[bool]:
    return [index == active_index for index in range(trace_count)]


def _enhanced_rgb_colors(rgb: np.ndarray, gamma: float = 0.8) -> list[str]:
    values = np.asarray(rgb, dtype=np.float64)
    if len(values) == 0:
        return []
    low = np.percentile(values, 1, axis=0)
    high = np.percentile(values, 99, axis=0)
    dynamic = np.where((high - low) <= 0, 1.0, high - low)
    stretched = np.clip((values - low) / dynamic, 0.0, 1.0)
    adjusted = np.power(stretched, gamma)
    rgb_255 = np.rint(adjusted * 255.0).astype(np.uint8)
    return [f"rgb({int(red)},{int(green)},{int(blue)})" for red, green, blue in rgb_255]


def _crop_bounds_trace(data: dict[str, np.ndarray], bounds: CropBounds) -> Any:
    import plotly.graph_objects as go

    origin = np.asarray(data["origin_xyz"], dtype=np.float64)
    minimum = np.asarray(bounds.min_xyz, dtype=np.float64) - origin
    maximum = np.asarray(bounds.max_xyz, dtype=np.float64) - origin
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
        name="Approved crop bounds",
        hoverinfo="skip",
        visible=True,
    )


def _validate_preview_limits(target_preview_points: int, max_preview_points: int) -> None:
    if target_preview_points <= 0:
        raise _error(
            code="PREVIEW_TARGET_INVALID",
            message="target_preview_points must be positive",
            details={"target_preview_points": target_preview_points},
        )
    if max_preview_points <= 0:
        raise _error(
            code="PREVIEW_MAX_INVALID",
            message="max_preview_points must be positive",
            details={"max_preview_points": max_preview_points},
        )
    if target_preview_points > max_preview_points:
        raise _error(
            code="PREVIEW_TARGET_EXCEEDS_MAX",
            message="target_preview_points cannot exceed max_preview_points",
            details={
                "target_preview_points": target_preview_points,
                "max_preview_points": max_preview_points,
            },
        )


def _validate_output_dir(output_dir: Path, *, project_root: Path | str | None) -> None:
    root = Path(project_root) if project_root is not None else Path.cwd()
    output_resolved = output_dir.resolve(strict=False)
    if _is_relative_to(output_resolved, (root / "data" / "raw").resolve(strict=False)):
        raise _error(
            code="PREVIEW_OUTPUT_FORBIDDEN",
            message="Preview output cannot be written under data/raw",
            details={"output_dir": str(output_dir)},
        )
    previews_root = (root / "outputs" / "previews").resolve(strict=False)
    temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
    if not (
        _is_relative_to(output_resolved, previews_root)
        or _is_relative_to(output_resolved, temp_root)
    ):
        raise _error(
            code="PREVIEW_OUTPUT_FORBIDDEN",
            message="Preview output must be under outputs/previews or pytest temp",
            details={"output_dir": str(output_dir)},
        )


def _validate_existing_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _error(
            code="PREVIEW_OUTPUT_EXISTS",
            message="Preview output already exists and overwrite is disabled",
            details={"paths": existing},
        )


def _concat_or_empty(parts: list[np.ndarray], *, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    if not parts:
        return np.empty(shape, dtype=dtype)
    return np.concatenate(parts)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _replace(temp_path: Path, final_path: Path, *, overwrite: bool) -> None:
    if final_path.exists() and not overwrite:
        raise _error(
            code="PREVIEW_OUTPUT_EXISTS",
            message="Preview output already exists and overwrite is disabled",
            details={"path": str(final_path)},
        )
    if final_path.exists():
        final_path.unlink()
    os.replace(temp_path, final_path)


def _plotly_available() -> bool:
    try:
        import plotly  # noqa: F401
    except ImportError:
        return False
    return True


def _current_process() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process()


def _rss_bytes(process: Any | None) -> int | None:
    if process is None:
        return None
    return int(process.memory_info().rss)


def _report(
    reporter: ProgressReporter | None,
    source_id: str,
    processed_points: int,
    total_points: int,
    chunk_id: int,
) -> None:
    if reporter is None:
        return
    percent = 100.0 if total_points == 0 else processed_points / total_points * 100.0
    reporter.update(
        PipelineProgress(
            task_id=f"{source_id}:preview",
            stage="preview",
            status="running" if processed_points < total_points else "completed",
            processed_points=processed_points,
            total_points=total_points,
            percent=percent,
            message=f"Processed chunk {chunk_id}",
            metadata={"chunk_id": chunk_id},
        )
    )


def _cancelled(cancellation_token: CancellationToken | None) -> bool:
    return cancellation_token is not None and cancellation_token.is_cancelled()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
