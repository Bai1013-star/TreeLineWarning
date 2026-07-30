"""Read-only renderers for inspecting deterministic V3 tree segmentation results."""

from __future__ import annotations

import colorsys
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.tree_inputs import TreeSegmentationInputIdentity, TreeSegmentationInputSet
from core.contracts.vegetation_trees import TreeSegmentationResult
from PIL import Image, ImageDraw
from plotly import graph_objects as go

from algorithms.vegetation.canopy_height_model import CanopyHeightModelResult
from algorithms.vegetation.canopy_segmentation import CanopySegmentationResult
from algorithms.vegetation.treetop_candidates import TreetopCandidateResult

UNASSIGNED_RGB = (128, 128, 128)
_TOPDOWN_MAX_DIMENSION = 1_800
MAXIMUM_3D_POINTS = 200_000
_POINT_ALIGNED_RESULT_ARRAYS = (
    "segment_id_by_point",
    "tree_id_by_point",
    "tree_distance_ready_by_point",
)


def segment_color(segment_id: int) -> tuple[int, int, int]:
    """Return the stable RGB display colour for a formal segment ID."""
    if segment_id < 0:
        return UNASSIGNED_RGB
    hue = ((segment_id * 137.508) % 360.0) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.92)
    return tuple(round(channel * 255) for channel in (red, green, blue))


def deterministic_display_indices(point_count: int, maximum: int) -> np.ndarray:
    """Select evenly distributed source-row indices without random state."""
    if type(point_count) is not int or point_count < 0:
        raise ValueError("point_count must be a non-negative Python int")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("maximum must be a positive Python int")
    displayed = min(point_count, maximum)
    if displayed == 0:
        return np.empty(0, dtype=np.int64)
    if displayed == point_count:
        return np.arange(point_count, dtype=np.int64)
    return np.linspace(0, point_count - 1, displayed, dtype=np.int64)


def validate_maximum_3d_points(value: Any) -> int:
    """Return the fixed display-only 3D cap or reject an invalid request."""
    if not isinstance(value, int | np.integer) or isinstance(value, bool | np.bool_):
        raise ValueError("maximum_3d_points must be an integer from 1 through 200000")
    maximum = int(value)
    if not 1 <= maximum <= MAXIMUM_3D_POINTS:
        raise ValueError("maximum_3d_points must be an integer from 1 through 200000")
    return maximum


def validate_input_result_alignment(
    tree_input: TreeSegmentationInputSet, result: TreeSegmentationResult
) -> int:
    """Reject a result that is not exactly identified and aligned to its V3 input."""
    xyz = tree_input.vegetation_points.xyz_local
    point_count = len(xyz)
    identity = tree_input.identity
    if point_count != identity.point_count:
        raise ValueError("source XYZ count does not match tree input identity")
    if result.statistics.input_point_count != point_count:
        raise ValueError("result source point count does not match source XYZ count")
    if result.source_id != identity.source_id:
        raise ValueError("result source_id does not match tree input identity")
    if result.manifest_source_id != identity.manifest_source_id:
        raise ValueError("result manifest_source_id does not match tree input identity")
    if result.origin_xyz != identity.origin_xyz:
        raise ValueError("result origin_xyz does not exactly match tree input identity")
    provenance_identity = result.input_provenance.get("source_identity")
    try:
        parsed_provenance_identity = TreeSegmentationInputIdentity.from_dict(
            dict(provenance_identity) if isinstance(provenance_identity, Mapping) else {}
        )
    except ValueError as error:
        raise ValueError("result input provenance identity is invalid") from error
    if parsed_provenance_identity != identity:
        raise ValueError("result input provenance identity does not match tree input identity")
    for name in _POINT_ALIGNED_RESULT_ARRAYS:
        if len(getattr(result, name)) != point_count:
            raise ValueError(f"result {name} does not match source XYZ count")
    return point_count


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rgb_hex(color: Sequence[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _point_canvas(xy: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (point_count, 2)")
    if len(xy) == 0:
        return 1, 1, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    bounds_min = np.min(xy, axis=0)
    bounds_max = np.max(xy, axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1e-9)
    scale = min(
        _TOPDOWN_MAX_DIMENSION / float(extent[0]),
        _TOPDOWN_MAX_DIMENSION / float(extent[1]),
    )
    width = max(1, int(np.ceil(extent[0] * scale)) + 1)
    height = max(1, int(np.ceil(extent[1] * scale)) + 1)
    x = np.rint((xy[:, 0] - bounds_min[0]) * scale).astype(np.int64)
    y = np.rint((bounds_max[1] - xy[:, 1]) * scale).astype(np.int64)
    return width, height, x, y


def _topdown_image(
    tree_input: TreeSegmentationInputSet, result: TreeSegmentationResult
) -> tuple[Image.Image, dict[str, int]]:
    validate_input_result_alignment(tree_input, result)
    xy = np.asarray(tree_input.vegetation_points.xyz_local[:, :2], dtype=np.float64)
    segment_ids = result.segment_id_by_point
    width, height, x, y = _point_canvas(xy)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    pixel_ids = y * width + x
    unassigned_pixels = np.unique(pixel_ids[segment_ids < 0])
    assigned = segment_ids >= 0
    assigned_pixel_ids = pixel_ids[assigned]
    assigned_segment_ids = segment_ids[assigned]
    conflict_count = 0
    assigned_pixel_count = 0
    if len(assigned_pixel_ids):
        order = np.lexsort((assigned_segment_ids, assigned_pixel_ids))
        sorted_pixels = assigned_pixel_ids[order]
        sorted_segments = assigned_segment_ids[order]
        pair_start = np.empty(len(sorted_pixels), dtype=np.bool_)
        pair_start[0] = True
        pair_start[1:] = (sorted_pixels[1:] != sorted_pixels[:-1]) | (
            sorted_segments[1:] != sorted_segments[:-1]
        )
        unique_pixels = sorted_pixels[pair_start]
        unique_segments = sorted_segments[pair_start]
        pixel_start = np.empty(len(unique_pixels), dtype=np.bool_)
        pixel_start[0] = True
        pixel_start[1:] = unique_pixels[1:] != unique_pixels[:-1]
        starts = np.flatnonzero(pixel_start)
        segment_count = np.diff(np.append(starts, len(unique_pixels)))
        assigned_pixel_count = len(starts)
        conflict = segment_count >= 2
        conflict_count = int(np.count_nonzero(conflict))
        display_pixels = unique_pixels[starts]
        rows, columns = np.divmod(display_pixels, width)
        for row, column, segment_id, is_conflict in zip(
            rows, columns, unique_segments[starts], conflict, strict=True
        ):
            canvas[int(row), int(column)] = (
                (0, 0, 0) if is_conflict else segment_color(int(segment_id))
            )
    if len(unassigned_pixels):
        rows, columns = np.divmod(unassigned_pixels, width)
        assigned_pixel_mask = np.zeros(height * width, dtype=np.bool_)
        assigned_pixel_mask[assigned_pixel_ids] = True
        unassigned_only = ~assigned_pixel_mask[unassigned_pixels]
        canvas[rows[unassigned_only], columns[unassigned_only]] = UNASSIGNED_RGB
    else:
        unassigned_only = np.empty(0, dtype=np.bool_)
    return Image.fromarray(canvas, mode="RGB"), {
        "occupied_pixel_count": int(len(np.unique(pixel_ids))),
        "assigned_pixel_count": assigned_pixel_count,
        "unassigned_only_pixel_count": int(np.count_nonzero(unassigned_only)),
        "conflict_pixel_count": conflict_count,
    }


def render_topdown_png(
    tree_input: TreeSegmentationInputSet,
    result: TreeSegmentationResult,
    output_path: Path | str,
) -> Path:
    """Render every source point with an order-independent per-pixel policy."""
    path = Path(output_path)
    image, _statistics = _topdown_image(tree_input, result)
    _ensure_parent(path)
    image.save(path, format="PNG")
    return path


def topdown_pixel_statistics(
    tree_input: TreeSegmentationInputSet, result: TreeSegmentationResult
) -> dict[str, int]:
    """Return order-independent top-down occupancy and segment-conflict counts."""
    _image, statistics = _topdown_image(tree_input, result)
    return statistics


def _height_background(height: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    image = np.full((*height.shape, 3), 245, dtype=np.uint8)
    if not np.any(occupied):
        return image
    values = height[occupied].astype(np.float64)
    lower, upper = float(np.min(values)), float(np.max(values))
    normalized = np.full(height.shape, 0.5, dtype=np.float64)
    if upper > lower:
        normalized[occupied] = (height[occupied] - lower) / (upper - lower)
    shade = np.rint(55 + normalized * 180).astype(np.uint8)
    image[occupied] = np.stack(
        (shade[occupied] // 2, shade[occupied], shade[occupied] // 2), axis=1
    )
    return image


def render_diagnostic_png(
    canopy_model: CanopyHeightModelResult,
    candidates: TreetopCandidateResult,
    segmentation: CanopySegmentationResult,
    output_path: Path | str,
) -> Path:
    """Render CHM, candidate-linked labels, and ambiguity cells in one raster."""
    path = Path(output_path)
    height = canopy_model.canopy_height_m
    occupied = canopy_model.occupied_mask
    labels = segmentation.region_label_by_cell
    if height.shape == (0, 0):
        image = Image.new("RGB", (1, 1), (245, 245, 245))
    else:
        pixels = _height_background(height, occupied)
        for label in np.unique(labels[labels > 0]):
            mask = labels == label
            colour = np.asarray(segment_color(int(label) - 1), dtype=np.uint8)
            pixels[mask] = np.rint(pixels[mask] * 0.35 + colour * 0.65).astype(np.uint8)
        pixels[segmentation.ambiguity_mask] = (220, 40, 40)
        image = Image.fromarray(np.flipud(pixels), mode="RGB")
        draw = ImageDraw.Draw(image)
        rows = image.height
        for row, column in zip(candidates.row_index, candidates.column_index, strict=True):
            x, y = int(column), rows - 1 - int(row)
            draw.line((x - 2, y, x + 2, y), fill=(20, 30, 180), width=1)
            draw.line((x, y - 2, x, y + 2), fill=(20, 30, 180), width=1)
        scale = max(1, min(20, 1_000 // max(image.width, image.height)))
        if scale > 1:
            image = image.resize(
                (image.width * scale, image.height * scale), Image.Resampling.NEAREST
            )
    _ensure_parent(path)
    image.save(path, format="PNG")
    return path


def render_3d_html(
    tree_input: TreeSegmentationInputSet,
    result: TreeSegmentationResult,
    output_path: Path | str,
    *,
    maximum_points: int,
) -> tuple[Path, int]:
    """Write an interactive local-XYZ plot using display-only deterministic sampling."""
    path = Path(output_path)
    validate_input_result_alignment(tree_input, result)
    maximum_points = validate_maximum_3d_points(maximum_points)
    xyz = tree_input.vegetation_points.xyz_local
    indices = deterministic_display_indices(len(xyz), maximum_points)
    segment_ids = result.segment_id_by_point[indices]
    colors = [_rgb_hex(segment_color(int(value))) for value in segment_ids]
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[indices, 0],
                y=xyz[indices, 1],
                z=xyz[indices, 2],
                mode="markers",
                marker={"size": 2, "color": colors, "opacity": 0.85},
                customdata=segment_ids,
                hovertemplate="segment_id=%{customdata}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 30},
        title="V3 tree segmentation (local XYZ)",
    )
    _ensure_parent(path)
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return path, len(indices)


def build_summary(
    *,
    scene_name: str,
    source_artifact_paths: Mapping[str, Path | str],
    tree_input: TreeSegmentationInputSet,
    canopy_model: CanopyHeightModelResult,
    candidates: TreetopCandidateResult,
    segmentation: CanopySegmentationResult,
    result: TreeSegmentationResult,
    maximum_3d_points: int,
    displayed_3d_points: int,
    output_paths: Mapping[str, Path | str],
    topdown_statistics: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return bounded scalar metadata for a completed visualization run."""
    source_point_count = validate_input_result_alignment(tree_input, result)
    maximum_3d_points = validate_maximum_3d_points(maximum_3d_points)
    if (
        not isinstance(displayed_3d_points, int | np.integer)
        or isinstance(displayed_3d_points, bool | np.bool_)
        or int(displayed_3d_points) != min(source_point_count, maximum_3d_points)
    ):
        raise ValueError("displayed_3d_points does not match the deterministic display cap")
    audit = result.input_provenance.get("mapping_audit", {})
    if not isinstance(audit, Mapping):
        audit = {}
    segment_ids = result.segment_id_by_point
    assigned_point_count = int(np.count_nonzero(segment_ids >= 0))
    unassigned_point_count = int(np.count_nonzero(segment_ids < 0))
    if assigned_point_count + unassigned_point_count != source_point_count:
        raise ValueError("assigned and unassigned point counts do not cover the source")
    summary = {
        "scene_name": scene_name,
        "source_artifact_paths": {name: str(path) for name, path in source_artifact_paths.items()},
        "source_point_count": source_point_count,
        "assigned_point_count": assigned_point_count,
        "unassigned_point_count": unassigned_point_count,
        "eligible_unassigned_point_count": audit.get("eligible_unassigned_point_count"),
        "candidate_count": int(len(candidates.candidate_id)),
        "segment_record_count": int(len(result.records)),
        "chm_shape": [int(value) for value in canopy_model.canopy_height_m.shape],
        "ambiguous_cell_count": int(segmentation.ambiguous_cell_count),
        "ambiguous_assigned_point_count": audit.get("ambiguous_assigned_point_count"),
        "html_maximum_point_count": maximum_3d_points,
        "html_displayed_point_count": displayed_3d_points,
        "sampling_strategy": "evenly_spaced_source_order",
        "output_paths": {name: str(path) for name, path in output_paths.items()},
    }
    if topdown_statistics is not None:
        summary.update({name: int(value) for name, value in topdown_statistics.items()})
    return summary


def write_summary_json(summary: Mapping[str, Any], output_path: Path | str) -> Path:
    """Persist summary metadata only; callers supply no point or candidate arrays."""
    path = Path(output_path)
    _ensure_parent(path)
    path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "UNASSIGNED_RGB",
    "MAXIMUM_3D_POINTS",
    "build_summary",
    "deterministic_display_indices",
    "render_3d_html",
    "render_diagnostic_png",
    "render_topdown_png",
    "segment_color",
    "topdown_pixel_statistics",
    "validate_input_result_alignment",
    "validate_maximum_3d_points",
    "write_summary_json",
]
