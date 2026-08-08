"""Tree-wire clearance review visualization (read-only).

This module renders a validated ``TreeWireClearanceResult`` together with the
validated V4 and W1 material needed for display.  It never recomputes any
distance: every clearance segment, reported distance and Tree/Wire ID comes
verbatim from the stored clearance result.  Input identity is cross-checked
(the displayed V4 artifact must carry the same result digest and the displayed
W1 pair the same NPZ digest that the clearance result binds), otherwise the
viewer refuses to render.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.tree_wire_clearance import (
    TreeWireClearanceRecordStatus,
    TreeWireClearanceResult,
)
from core.contracts.v4_tree_segmentation import V4TreeAssignmentState

from algorithms.vegetation.v4_tree_artifacts import load_v4_tree_segmentation_artifact
from algorithms.vegetation.v4_tree_visualization import (
    deterministic_tree_sample_positions,
    tree_color,
)
from algorithms.wires.reconstruct import WireReconstructionError, load_wire_models

TOOL_NAME = "tree-wire-clearance-visualization"
SUMMARY_SCHEMA_VERSION = "twc-vis-v1"

DEFAULT_MAX_POINTS_PER_TREE = 10_000
HARD_MAX_POINTS_PER_TREE = 100_000

_WIRE_PALETTE = [
    "#dc2626",
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


class TreeWireClearanceVisualizationError(RuntimeError):
    """Base clearance-viewer failure (invalid or mismatched inputs)."""


class TreeWireClearanceVisualizationOutputError(TreeWireClearanceVisualizationError):
    """Output already exists and overwrite is disabled."""


@dataclass(frozen=True, slots=True)
class TreeWireClearanceVisualizationBudget:
    """Frozen display budget for the clearance viewer."""

    max_points_per_tree: int
    sampling_strategy: str = "per_tree_stratified_even_selection"

    def __post_init__(self) -> None:
        if type(self.max_points_per_tree) is not int or not (
            1 <= self.max_points_per_tree <= HARD_MAX_POINTS_PER_TREE
        ):
            raise TreeWireClearanceVisualizationError(
                "max_points_per_tree must be an integer from 1 through "
                f"{HARD_MAX_POINTS_PER_TREE}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.sampling_strategy,
            "max_points_per_tree": self.max_points_per_tree,
        }


@dataclass(frozen=True, slots=True)
class TreeWireClearanceVisualizationResult:
    """Written output references plus the deterministic summary payload."""

    html_path: Path
    summary_path: Path | None
    summary: dict[str, Any]


def validate_display_budget(max_points_per_tree: int) -> TreeWireClearanceVisualizationBudget:
    return TreeWireClearanceVisualizationBudget(max_points_per_tree=max_points_per_tree)


def load_tree_wire_clearance_result(path: Path | str) -> TreeWireClearanceResult:
    """Load and fully validate a published clearance result JSON."""

    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TreeWireClearanceVisualizationError(
            f"unable to read clearance result JSON: {exc}"
        ) from exc
    try:
        return TreeWireClearanceResult.from_dict(payload)
    except (ValueError, TypeError) as exc:
        raise TreeWireClearanceVisualizationError(
            f"invalid clearance result: {exc}"
        ) from exc


def _check_input_identity(
    clearance: TreeWireClearanceResult,
    v4_result,
    w1_result,
) -> None:
    if v4_result.result_digest != clearance.v4_result_digest:
        raise TreeWireClearanceVisualizationError(
            "displayed V4 artifact does not match the clearance result digest"
        )
    if w1_result.output_npz_sha256 != clearance.w1_npz_sha256:
        raise TreeWireClearanceVisualizationError(
            "displayed W1 artifact does not match the clearance result digest"
        )


def _wire_color(wire_id: int) -> str:
    return _WIRE_PALETTE[wire_id % len(_WIRE_PALETTE)]


def _wire_segment_trace(w1_arrays, *, ready_only: bool) -> Any:
    import plotly.graph_objects as go

    segment_wire_id = np.asarray(w1_arrays["segment_wire_id"], dtype=np.int32)
    segment_ready = np.asarray(w1_arrays["segment_distance_ready"], dtype=bool)
    curve_xyz = np.asarray(w1_arrays["sampled_curve_xyz"], dtype=np.float32)
    curve_offsets = np.asarray(w1_arrays["sampled_curve_offsets"], dtype=np.int64)
    xs: list[Any] = []
    ys: list[Any] = []
    zs: list[Any] = []
    colors: list[str] = []
    customdata: list[Any] = []
    for segment_position in range(int(segment_wire_id.shape[0])):
        if bool(segment_ready[segment_position]) != ready_only:
            continue
        start = int(curve_offsets[segment_position])
        stop = int(curve_offsets[segment_position + 1])
        if stop - start < 2:
            continue
        wire_id = int(segment_wire_id[segment_position])
        color = _wire_color(wire_id)
        samples = curve_xyz[start:stop]
        for sample_position in range(stop - start):
            xs.append(float(samples[sample_position, 0]))
            ys.append(float(samples[sample_position, 1]))
            zs.append(float(samples[sample_position, 2]))
            colors.append(color)
            customdata.append((wire_id, segment_position))
        xs.append(None)
        ys.append(None)
        zs.append(None)
        colors.append(color)
        customdata.append((wire_id, segment_position))
    name = "wire distance-ready segments" if ready_only else "wire non-ready segments"
    dash = "solid" if ready_only else "dash"
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name,
        uid=f"twc-wire-{'ready' if ready_only else 'nonready'}",
        visible=True if ready_only else "legendonly",
        line={"color": colors, "width": 5 if ready_only else 2, "dash": dash},
        customdata=np.asarray(customdata, dtype=np.int64).reshape(-1, 2),
        hovertemplate="wire=%{customdata[0]}<br>segment=%{customdata[1]}<extra></extra>",
    )


def _tree_point_trace(v4_result, tree, positions: np.ndarray) -> Any:
    import plotly.graph_objects as go

    xyz = np.asarray(v4_result.xyz_local_m, dtype=np.float64)[positions]
    original = np.asarray(v4_result.original_index, dtype=np.int64)[positions]
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name=f"tree {int(tree.tree_id)}",
        uid=f"twc-tree-{int(tree.tree_id)}",
        marker={"size": 2.4, "color": tree_color(int(tree.tree_id)), "opacity": 0.85},
        customdata=original.reshape(-1, 1),
        hovertemplate=(
            f"tree={int(tree.tree_id)}<br>original_index=%{{customdata[0]}}<extra></extra>"
        ),
    )


def _clearance_segment_trace(clearance: TreeWireClearanceResult) -> Any:
    import plotly.graph_objects as go

    xs: list[Any] = []
    ys: list[Any] = []
    zs: list[Any] = []
    colors: list[str] = []
    customdata: list[Any] = []
    for record in clearance.records:
        if record.status is not TreeWireClearanceRecordStatus.VALID:
            continue
        if (
            record.nearest_tree_xyz_local_m is None
            or record.nearest_wire_xyz_local_m is None
        ):
            continue
        color = tree_color(record.tree_id)
        tree_xyz = record.nearest_tree_xyz_local_m
        wire_xyz = record.nearest_wire_xyz_local_m
        for point in (tree_xyz, wire_xyz):
            xs.append(float(point[0]))
            ys.append(float(point[1]))
            zs.append(float(point[2]))
            colors.append(color)
            customdata.append(
                (
                    record.tree_id,
                    record.nearest_wire_id if record.nearest_wire_id is not None else -1,
                    record.minimum_clearance_m if record.minimum_clearance_m is not None else -1.0,
                )
            )
        xs.append(None)
        ys.append(None)
        zs.append(None)
        colors.append(color)
        customdata.append(
            (
                record.tree_id,
                record.nearest_wire_id if record.nearest_wire_id is not None else -1,
                record.minimum_clearance_m if record.minimum_clearance_m is not None else -1.0,
            )
        )
    if not xs:
        return go.Scatter3d(
            x=[], y=[], z=[], mode="lines", name="clearance segments", uid="twc-segments",
            visible="legendonly",
        )
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name="clearance segments",
        uid="twc-segments",
        line={"color": colors, "width": 7},
        customdata=np.asarray(customdata, dtype=object),
        hovertemplate=(
            "clearance<br>tree=%{customdata[0]}<br>nearest wire=%{customdata[1]}"
            "<br>minimum clearance=%{customdata[2]:.3f} m<extra></extra>"
        ),
    )


def build_summary(
    *,
    clearance: TreeWireClearanceResult,
    displayed_counts: dict[str, int],
    budget: TreeWireClearanceVisualizationBudget,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    trees = []
    for record in clearance.records:
        trees.append(
            {
                "tree_id": record.tree_id,
                "status": record.status.name,
                "nearest_wire_id": record.nearest_wire_id,
                "minimum_clearance_m": record.minimum_clearance_m,
                "nearest_tree_original_index": record.nearest_tree_original_index,
                "nearest_tree_xyz_local_m": (
                    None
                    if record.nearest_tree_xyz_local_m is None
                    else list(record.nearest_tree_xyz_local_m)
                ),
                "nearest_wire_xyz_local_m": (
                    None
                    if record.nearest_wire_xyz_local_m is None
                    else list(record.nearest_wire_xyz_local_m)
                ),
                "w1_segment_index": record.w1_segment_index,
                "polyline_edge_index": record.polyline_edge_index,
                "edge_interpolation_t": record.edge_interpolation_t,
                "tree_point_count": record.tree_point_count,
                "candidate_edge_count": record.candidate_edge_count,
                "evaluated_edge_count": record.evaluated_edge_count,
                "quality_flags": record.quality_flags,
            }
        )
    return {
        "tool": TOOL_NAME,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "schema_version": clearance.schema_version,
        "algorithm_version": clearance.algorithm_version,
        "status": clearance.status.name,
        "clearance_result_digest": clearance.result_digest,
        "v4_result_digest": clearance.v4_result_digest,
        "v4_source_id": clearance.v4_source_id,
        "v4_origin_context": list(clearance.v4_origin_context),
        "w1_npz_sha256": clearance.w1_npz_sha256,
        "config_digest": clearance.config_digest,
        "displayed_counts": displayed_counts,
        "sampling": budget.to_dict(),
        "trees": trees,
        "outputs": output_paths,
    }


def write_summary_json(summary: Mapping[str, Any], output_path: Path | str) -> Path:
    path = Path(output_path)
    _atomic_write_text(path, json.dumps(dict(summary), indent=2, sort_keys=True) + "\n")
    return path


def render_html(
    clearance: TreeWireClearanceResult,
    v4_result,
    w1_arrays,
    budget: TreeWireClearanceVisualizationBudget,
) -> str:
    import plotly.graph_objects as go

    figure = go.Figure()
    assigned_positions_by_tree = _assigned_positions(v4_result)
    for tree in v4_result.tree_records:
        positions = assigned_positions_by_tree.get(int(tree.tree_id), np.empty(0, dtype=np.int64))
        sampled = deterministic_tree_sample_positions(positions, budget.max_points_per_tree)
        figure.add_trace(_tree_point_trace(v4_result, tree, sampled))
    figure.add_trace(_wire_segment_trace(w1_arrays, ready_only=True))
    figure.add_trace(_wire_segment_trace(w1_arrays, ready_only=False))
    figure.add_trace(_clearance_segment_trace(clearance))
    figure.update_layout(
        title="Tree-Wire Minimum Clearance Review (local XYZ)",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        legend={"itemsizing": "constant"},
    )
    return figure.to_html(
        include_plotlyjs=True, full_html=True, div_id="twc-clearance-review"
    )


def _assigned_positions(v4_result) -> dict[int, np.ndarray]:
    states = np.asarray(v4_result.assignment_state_by_point, dtype=np.uint8)
    tree_ids = np.asarray(v4_result.tree_id_by_point, dtype=np.int32)
    assigned = int(V4TreeAssignmentState.ASSIGNED)
    return {
        int(tree.tree_id): np.flatnonzero((tree_ids == int(tree.tree_id)) & (states == assigned))
        for tree in v4_result.tree_records
    }


def _check_output_target(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists() and not overwrite:
        raise TreeWireClearanceVisualizationOutputError(
            f"{label} exists and overwrite is disabled: {path}"
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def run_tree_wire_clearance_visualization(
    clearance_json: Path | str,
    v4_root: Path | str,
    w1_npz_path: Path | str,
    w1_json_path: Path | str,
    output_html: Path | str,
    summary_json: Path | str | None = None,
    *,
    max_points_per_tree: int = DEFAULT_MAX_POINTS_PER_TREE,
    overwrite: bool = False,
) -> TreeWireClearanceVisualizationResult:
    """Render the read-only clearance review without recomputing any distance."""

    budget = validate_display_budget(max_points_per_tree)
    clearance = load_tree_wire_clearance_result(clearance_json)
    try:
        loaded_v4 = load_v4_tree_segmentation_artifact(Path(v4_root))
    except Exception as exc:  # noqa: BLE001 - fail closed on any invalid V4 input
        raise TreeWireClearanceVisualizationError(f"invalid V4 artifact: {exc}") from exc
    try:
        w1_result, w1_arrays = load_wire_models(Path(w1_npz_path), Path(w1_json_path))
    except WireReconstructionError as exc:
        raise TreeWireClearanceVisualizationError(f"invalid W1 artifact: {exc}") from exc
    _check_input_identity(clearance, loaded_v4.result, w1_result)

    html_path = Path(output_html)
    summary_path = Path(summary_json) if summary_json is not None else None
    _check_output_target(html_path, overwrite=overwrite, label="output_html")
    if summary_path is not None:
        _check_output_target(summary_path, overwrite=overwrite, label="summary_json")

    valid_count = sum(
        record.status is TreeWireClearanceRecordStatus.VALID for record in clearance.records
    )
    displayed_counts = {
        "trees": len(clearance.records),
        "valid_trees": valid_count,
        "clearance_segments": valid_count,
        "assigned_points": int(
            sum(
                min(int(positions.shape[0]), budget.max_points_per_tree)
                for positions in _assigned_positions(loaded_v4.result).values()
            )
        ),
        "wire_segments": int(
            np.count_nonzero(np.asarray(w1_arrays["segment_distance_ready"], dtype=bool))
        ),
    }
    output_paths = {
        "html": str(html_path),
        "summary_json": str(summary_path) if summary_path is not None else None,
    }
    summary = build_summary(
        clearance=clearance,
        displayed_counts=displayed_counts,
        budget=budget,
        output_paths=output_paths,
    )
    html = render_html(clearance, loaded_v4.result, w1_arrays, budget)
    _atomic_write_text(html_path, html)
    if summary_path is not None:
        write_summary_json(summary, summary_path)
    return TreeWireClearanceVisualizationResult(
        html_path=html_path,
        summary_path=summary_path,
        summary=summary,
    )


__all__ = [
    "TOOL_NAME",
    "SUMMARY_SCHEMA_VERSION",
    "DEFAULT_MAX_POINTS_PER_TREE",
    "TreeWireClearanceVisualizationError",
    "TreeWireClearanceVisualizationOutputError",
    "TreeWireClearanceVisualizationBudget",
    "TreeWireClearanceVisualizationResult",
    "validate_display_budget",
    "load_tree_wire_clearance_result",
    "build_summary",
    "write_summary_json",
    "render_html",
    "run_tree_wire_clearance_visualization",
]
