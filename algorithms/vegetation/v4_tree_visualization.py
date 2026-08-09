"""V4 tree-segmentation review visualization (read-only, PLAN-FROZEN patterns).

This module is an independent viewer over a published, fully validated V4
tree-segmentation artifact.  It never executes segmentation, never re-judges
tree memberships, never infers Tree IDs and never modifies any input artifact.

Every Tree ID receives a stable deterministic colour (golden-angle HSV).
Assigned points use bounded per-tree stratified sampling so small trees are not
hidden behind large trees; review, conflict and unassigned layers use bounded
even sampling.  The same artifact always produces the same displayed geometry
and the same deterministic summary JSON.
"""

from __future__ import annotations

import colorsys
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts.v4_tree_segmentation import (
    V4AssignmentReasonFlag,
    V4TreeAssignmentState,
    V4TreeInstanceRecord,
    V4TreeSegmentationError,
    V4TreeSegmentationResult,
    V4TrunkCandidateState,
)

from algorithms.vegetation.tree_visualization import deterministic_display_indices
from algorithms.vegetation.v4_tree_artifacts import load_v4_tree_segmentation_artifact

TOOL_NAME = "v4-tree-visualization"
SUMMARY_SCHEMA_VERSION = "v4-tree-vis-v1"

DEFAULT_MAX_POINTS_PER_TREE = 20_000
DEFAULT_MAX_REVIEW_POINTS = 50_000
DEFAULT_MAX_UNASSIGNED_POINTS = 50_000
DEFAULT_MAX_CONFLICT_POINTS = 20_000
HARD_MAX_POINTS_PER_TREE = 100_000
HARD_MAX_LAYER_POINTS = 100_000

_FOOTPRINT_SEGMENTS = 32

REVIEW_COLOR = "#ffd400"
UNASSIGNED_COLOR = "#9ca3af"
CONFLICT_COLOR = "#d62728"
LAYER_NAMES = {
    "assigned": "assigned tree points",
    "review": "review points",
    "unassigned": "unassigned points",
    "conflict": "conflict points",
    "apex": "apexes",
    "trunk": "accepted trunks",
    "footprint": "summary footprint",
}


class V4TreeVisualizationError(RuntimeError):
    """Base visualization failure."""


class V4TreeVisualizationBudgetError(V4TreeVisualizationError):
    """Invalid display budget; nothing is written."""


class V4TreeVisualizationOutputError(V4TreeVisualizationError):
    """Output already exists and overwrite is disabled."""


@dataclass(frozen=True, slots=True)
class V4TreeVisualizationBudget:
    """Frozen, bounded display budget with deterministic sampling."""

    max_points_per_tree: int
    max_review_points: int
    max_unassigned_points: int
    max_conflict_points: int
    sampling_strategy: str = "per_tree_stratified_even_selection"

    def __post_init__(self) -> None:
        if type(self.max_points_per_tree) is not int or not (
            1 <= self.max_points_per_tree <= HARD_MAX_POINTS_PER_TREE
        ):
            raise V4TreeVisualizationBudgetError(
                "max_points_per_tree must be an integer from 1 through "
                f"{HARD_MAX_POINTS_PER_TREE}"
            )
        for name in (
            "max_review_points",
            "max_unassigned_points",
            "max_conflict_points",
        ):
            value = getattr(self, name)
            if type(value) is not int or not (1 <= value <= HARD_MAX_LAYER_POINTS):
                raise V4TreeVisualizationBudgetError(
                    f"{name} must be an integer from 1 through {HARD_MAX_LAYER_POINTS}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.sampling_strategy,
            "max_points_per_tree": self.max_points_per_tree,
            "max_review_points": self.max_review_points,
            "max_unassigned_points": self.max_unassigned_points,
            "max_conflict_points": self.max_conflict_points,
        }


@dataclass(frozen=True, slots=True)
class V4TreeVisualizationResult:
    """Written output references plus the deterministic summary payload."""

    html_path: Path
    summary_path: Path | None
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Colour and sampling helpers
# ---------------------------------------------------------------------------


def tree_color(tree_id: int) -> str:
    """Return the stable golden-angle HSV hex colour for a Tree ID."""

    if type(tree_id) is not int or tree_id < 0:
        raise ValueError("tree_id must be a non-negative Python int")
    hue = ((tree_id * 137.508) % 360.0) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.92)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def validate_display_budget(
    max_points_per_tree: int,
    max_review_points: int,
    max_unassigned_points: int,
    max_conflict_points: int,
) -> V4TreeVisualizationBudget:
    """Validate and return the display budget (fail closed on invalid caps)."""

    return V4TreeVisualizationBudget(
        max_points_per_tree=max_points_per_tree,
        max_review_points=max_review_points,
        max_unassigned_points=max_unassigned_points,
        max_conflict_points=max_conflict_points,
    )


def deterministic_tree_sample_positions(member_positions: np.ndarray, cap: int) -> np.ndarray:
    """Return per-tree stratified display positions under a bounded cap."""

    if not isinstance(member_positions, np.ndarray):
        raise TypeError("member_positions must be a numpy array")
    if member_positions.ndim != 1:
        raise ValueError("member_positions must be one-dimensional")
    count = int(member_positions.shape[0])
    if count == 0:
        return np.empty(0, dtype=np.int64)
    chosen = deterministic_display_indices(count, cap)
    return member_positions[chosen]


# ---------------------------------------------------------------------------
# Layer extraction (reads validated arrays only)
# ---------------------------------------------------------------------------


def _layer_positions(result: V4TreeSegmentationResult) -> dict[str, np.ndarray]:
    states = np.asarray(result.assignment_state_by_point, dtype=np.uint8)
    assigned = int(V4TreeAssignmentState.ASSIGNED)
    review = int(V4TreeAssignmentState.REVIEW_REQUIRED)
    unassigned = int(V4TreeAssignmentState.UNASSIGNED)
    conflict_mask = (
        np.asarray(result.assignment_reason_flags_by_point, dtype=np.uint32)
        & int(V4AssignmentReasonFlag.REVIEW_CONFLICT)
    ) != 0
    return {
        "assigned": np.flatnonzero(states == assigned),
        "review": np.flatnonzero(states == review),
        "unassigned": np.flatnonzero(states == unassigned),
        "conflict": np.flatnonzero(conflict_mask),
    }


def _tree_assigned_positions(result: V4TreeSegmentationResult) -> dict[int, np.ndarray]:
    states = np.asarray(result.assignment_state_by_point, dtype=np.uint8)
    tree_ids = np.asarray(result.tree_id_by_point, dtype=np.int32)
    assigned = int(V4TreeAssignmentState.ASSIGNED)
    positions_by_tree: dict[int, list[int]] = {}
    for position in range(int(tree_ids.shape[0])):
        tree_id = int(tree_ids[position])
        if tree_id >= 0 and int(states[position]) == assigned:
            positions_by_tree.setdefault(tree_id, []).append(position)
    return {
        int(tree.tree_id): np.asarray(positions_by_tree.get(int(tree.tree_id), []), dtype=np.int64)
        for tree in result.tree_records
    }


def _tree_apexes(result: V4TreeSegmentationResult, tree: V4TreeInstanceRecord) -> list[Any]:
    apex_by_id = {record.candidate_id: record for record in result.apex_records}
    candidates = [tree.primary_apex_candidate_id, *tree.secondary_apex_candidate_ids]
    apexes = []
    for candidate_id in candidates:
        record = apex_by_id.get(candidate_id)
        if record is not None:
            apexes.append(record)
    return apexes


def _tree_accepted_trunks(
    result: V4TreeSegmentationResult, tree: V4TreeInstanceRecord
) -> list[Any]:
    trunk_by_id = {record.candidate_id: record for record in result.trunk_records}
    candidates = [tree.primary_trunk_candidate_id, *tree.secondary_trunk_candidate_ids]
    trunks = []
    for candidate_id in candidates:
        if candidate_id is None:
            continue
        record = trunk_by_id.get(candidate_id)
        if record is not None and record.candidate_state is V4TrunkCandidateState.ACCEPTED:
            trunks.append(record)
    return trunks


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    *,
    result: V4TreeSegmentationResult,
    layer_positions: dict[str, np.ndarray],
    assigned_positions_by_tree: dict[int, np.ndarray],
    displayed_counts: dict[str, int],
    budget: V4TreeVisualizationBudget,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """Build the deterministic review summary payload."""

    trees = []
    for tree in result.tree_records:
        tree_id = int(tree.tree_id)
        apexes = _tree_apexes(result, tree)
        trunks = _tree_accepted_trunks(result, tree)
        slice_by_id = {record.slice_id: record for record in result.slice_records}
        trees.append(
            {
                "tree_id": tree_id,
                "assigned_point_count": int(assigned_positions_by_tree[tree_id].shape[0]),
                "displayed_point_count": min(
                    int(assigned_positions_by_tree[tree_id].shape[0]),
                    budget.max_points_per_tree,
                ),
                "confidence": tree.confidence,
                "quality_flags": tree.quality_flags,
                "reason_flags": tree.reason_flags,
                "crown_summary": tree.crown_summary.to_dict(),
                "geometry_summary": tree.geometry_summary.to_dict(),
                "primary_apex_candidate_id": tree.primary_apex_candidate_id,
                "secondary_apex_candidate_ids": list(tree.secondary_apex_candidate_ids),
                "apex_xyz_local_m": [list(apex.xyz_local_m) for apex in apexes],
                "primary_trunk_candidate_id": tree.primary_trunk_candidate_id,
                "secondary_trunk_candidate_ids": list(tree.secondary_trunk_candidate_ids),
                "accepted_trunk_centroid_xyz_local_m": [
                    list(trunk.centroid_xyz) for trunk in trunks
                ],
                "slice_summaries": [
                    slice_by_id[slice_id].to_dict()
                    for slice_id in tree.slice_ids
                    if slice_id in slice_by_id
                ],
            }
        )

    source = result.source_identity
    return {
        "tool": TOOL_NAME,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "schema_version": result.schema_version,
        "algorithm_version": result.algorithm_version,
        "v4_result_digest": result.result_digest,
        "source": {
            "source_id": source.source_id,
            "source_point_count": source.source_point_count,
            "coordinate_frame": source.coordinate_frame,
            "units": source.units,
            "origin_context": list(source.origin_context),
            "source_bounds_min_xyz": list(source.source_bounds_min_xyz),
            "source_bounds_max_xyz": list(source.source_bounds_max_xyz),
        },
        "layer_counts": {
            name: int(positions.shape[0])
            for name, positions in layer_positions.items()
            if name != "assigned"
        },
        "assigned_point_count": int(
            sum(
                positions.shape[0] for positions in assigned_positions_by_tree.values()
            )
        ),
        "displayed_counts": displayed_counts,
        "sampling": budget.to_dict(),
        "trees": trees,
        "outputs": output_paths,
    }


def write_summary_json(summary: Mapping[str, Any], output_path: Path | str) -> Path:
    """Write the deterministic summary JSON with atomic publication."""

    path = Path(output_path)
    _atomic_write_text(path, json.dumps(dict(summary), indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Plotly HTML
# ---------------------------------------------------------------------------


def _hover_point(customdata: np.ndarray, template: str) -> dict[str, Any]:
    return {"customdata": customdata, "hovertemplate": template}


def _assigned_trace(
    result: V4TreeSegmentationResult, tree: V4TreeInstanceRecord, positions: np.ndarray
) -> Any:
    import plotly.graph_objects as go

    xyz = np.asarray(result.xyz_local_m, dtype=np.float64)[positions]
    original = np.asarray(result.original_index, dtype=np.int64)[positions]
    hag = np.asarray(result.hag_m, dtype=np.float32)[positions]
    confidence = np.asarray(result.assignment_confidence_by_point, dtype=np.float32)[positions]
    customdata = np.column_stack((original, hag, confidence))
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name=f"tree {int(tree.tree_id)}",
        uid=f"v4tree-{int(tree.tree_id)}",
        marker={
            "size": 2.6,
            "color": tree_color(int(tree.tree_id)),
            "opacity": 0.9,
        },
        **_hover_point(
            customdata,
            f"tree={int(tree.tree_id)}<br>original_index=%{{customdata[0]}}"
            f"<br>hag=%{{customdata[1]:.2f}} m<br>confidence=%{{customdata[2]:.3f}}"
            "<extra></extra>",
        ),
    )


def _flat_layer_trace(
    result: V4TreeSegmentationResult,
    positions: np.ndarray,
    *,
    name: str,
    color: str,
    size: float,
    opacity: float,
    visible: bool,
) -> Any:
    import plotly.graph_objects as go

    xyz = np.asarray(result.xyz_local_m, dtype=np.float64)[positions]
    original = np.asarray(result.original_index, dtype=np.int64)[positions]
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name=name,
        uid=f"v4layer-{name}",
        visible=True if visible else "legendonly",
        marker={"size": size, "color": color, "opacity": opacity},
        **_hover_point(
            original.reshape(-1, 1),
            f"{name}<br>original_index=%{{customdata[0]}}<extra></extra>",
        ),
    )


def _apex_trace(result: V4TreeSegmentationResult) -> Any:
    import plotly.graph_objects as go

    rows = []
    colors = []
    customdata = []
    for tree in result.tree_records:
        color = tree_color(int(tree.tree_id))
        for apex in _tree_apexes(result, tree):
            rows.append(apex.xyz_local_m)
            colors.append(color)
            customdata.append((int(tree.tree_id), apex.candidate_id))
    if not rows:
        return go.Scatter3d(
            x=[], y=[], z=[], mode="markers", name=LAYER_NAMES["apex"], uid="v4-apex",
            visible="legendonly",
        )
    xyz = np.asarray(rows, dtype=np.float64)
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name=LAYER_NAMES["apex"],
        uid="v4-apex",
        marker={"size": 6.0, "color": colors, "symbol": "diamond", "opacity": 1.0},
        customdata=np.asarray(customdata, dtype=object),
        hovertemplate="apex<br>tree=%{customdata[0]}<br>candidate_id=%{customdata[1]}<extra></extra>",
    )


def _trunk_trace(result: V4TreeSegmentationResult) -> Any:
    import plotly.graph_objects as go

    rows = []
    colors = []
    customdata = []
    for tree in result.tree_records:
        color = tree_color(int(tree.tree_id))
        for trunk in _tree_accepted_trunks(result, tree):
            rows.append(trunk.centroid_xyz)
            colors.append(color)
            customdata.append((int(tree.tree_id), trunk.candidate_id))
    if not rows:
        return go.Scatter3d(
            x=[], y=[], z=[], mode="markers", name=LAYER_NAMES["trunk"], uid="v4-trunk",
            visible="legendonly",
        )
    xyz = np.asarray(rows, dtype=np.float64)
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name=LAYER_NAMES["trunk"],
        uid="v4-trunk",
        marker={"size": 5.0, "color": colors, "symbol": "square", "opacity": 1.0},
        customdata=np.asarray(customdata, dtype=object),
        hovertemplate="trunk<br>tree=%{customdata[0]}<br>candidate_id=%{customdata[1]}<extra></extra>",
    )


def _footprint_trace(result: V4TreeSegmentationResult) -> Any:
    import plotly.graph_objects as go

    xs: list[Any] = []
    ys: list[Any] = []
    zs: list[Any] = []
    colors: list[str] = []
    customdata: list[Any] = []
    for tree in result.tree_records:
        color = tree_color(int(tree.tree_id))
        radius = float(tree.crown_summary.equivalent_radius_m)
        center = tree.geometry_summary.xy_centroid
        z = float(tree.geometry_summary.robust_hag_m)
        if radius <= 0.0:
            continue
        for segment in range(_FOOTPRINT_SEGMENTS):
            theta = 2.0 * np.pi * segment / _FOOTPRINT_SEGMENTS
            xs.append(center[0] + radius * np.cos(theta))
            ys.append(center[1] + radius * np.sin(theta))
            zs.append(z)
            colors.append(color)
            customdata.append(int(tree.tree_id))
        xs.append(None)
        ys.append(None)
        zs.append(None)
        colors.append(color)
        customdata.append(int(tree.tree_id))
    if not xs:
        return go.Scatter3d(
            x=[], y=[], z=[], mode="lines", name=LAYER_NAMES["footprint"], uid="v4-footprint",
            visible="legendonly",
        )
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=LAYER_NAMES["footprint"],
        uid="v4-footprint",
        line={"color": colors, "width": 3},
        customdata=np.asarray(customdata, dtype=np.int32).reshape(-1, 1),
        hovertemplate="%{customdata[0]} summary footprint<extra></extra>",
    )


def render_html(result: V4TreeSegmentationResult, budget: V4TreeVisualizationBudget) -> str:
    """Render the self-contained Plotly review HTML from validated data."""

    import plotly.graph_objects as go

    layer_positions = _layer_positions(result)
    assigned_positions_by_tree = _tree_assigned_positions(result)
    figure = go.Figure()
    for tree in result.tree_records:
        positions = assigned_positions_by_tree[int(tree.tree_id)]
        sampled = deterministic_tree_sample_positions(positions, budget.max_points_per_tree)
        figure.add_trace(_assigned_trace(result, tree, sampled))
    review_positions = layer_positions["review"]
    unassigned_positions = layer_positions["unassigned"]
    conflict_positions = layer_positions["conflict"]
    figure.add_trace(
        _flat_layer_trace(
            result,
            review_positions[
                deterministic_display_indices(
                    len(review_positions), budget.max_review_points
                )
            ],
            name=LAYER_NAMES["review"],
            color=REVIEW_COLOR,
            size=2.4,
            opacity=0.9,
            visible=True,
        )
    )
    figure.add_trace(
        _flat_layer_trace(
            result,
            unassigned_positions[
                deterministic_display_indices(
                    len(unassigned_positions), budget.max_unassigned_points
                )
            ],
            name=LAYER_NAMES["unassigned"],
            color=UNASSIGNED_COLOR,
            size=1.4,
            opacity=0.5,
            visible=False,
        )
    )
    figure.add_trace(
        _flat_layer_trace(
            result,
            conflict_positions[
                deterministic_display_indices(
                    len(conflict_positions), budget.max_conflict_points
                )
            ],
            name=LAYER_NAMES["conflict"],
            color=CONFLICT_COLOR,
            size=3.0,
            opacity=0.95,
            visible=True,
        )
    )
    figure.add_trace(_apex_trace(result))
    figure.add_trace(_trunk_trace(result))
    figure.add_trace(_footprint_trace(result))
    figure.update_layout(
        title="V4 Tree Segmentation Review (local XYZ)",
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
        include_plotlyjs=True, full_html=True, div_id="v4-tree-review"
    )


# ---------------------------------------------------------------------------
# Atomic publication
# ---------------------------------------------------------------------------


def _check_output_target(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists() and not overwrite:
        raise V4TreeVisualizationOutputError(f"{label} exists and overwrite is disabled: {path}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_v4_tree_visualization(
    v4_root: Path | str,
    output_html: Path | str,
    summary_json: Path | str | None = None,
    *,
    max_points_per_tree: int = DEFAULT_MAX_POINTS_PER_TREE,
    max_review_points: int = DEFAULT_MAX_REVIEW_POINTS,
    max_unassigned_points: int = DEFAULT_MAX_UNASSIGNED_POINTS,
    max_conflict_points: int = DEFAULT_MAX_CONFLICT_POINTS,
    overwrite: bool = False,
) -> V4TreeVisualizationResult:
    """Render a read-only V4 review visualization from a validated artifact."""

    budget = validate_display_budget(
        max_points_per_tree,
        max_review_points,
        max_unassigned_points,
        max_conflict_points,
    )
    try:
        loaded = load_v4_tree_segmentation_artifact(Path(v4_root))
    except V4TreeSegmentationError as exc:
        raise V4TreeVisualizationError(f"invalid V4 artifact: {exc}") from exc
    result = loaded.result

    html_path = Path(output_html)
    summary_path = Path(summary_json) if summary_json is not None else None
    _check_output_target(html_path, overwrite=overwrite, label="output_html")
    if summary_path is not None:
        _check_output_target(summary_path, overwrite=overwrite, label="summary_json")

    layer_positions = _layer_positions(result)
    assigned_positions_by_tree = _tree_assigned_positions(result)
    displayed_counts = {
        "assigned": int(
            sum(
                min(
                    int(positions.shape[0]),
                    budget.max_points_per_tree,
                )
                for positions in assigned_positions_by_tree.values()
            )
        ),
        "review": min(int(layer_positions["review"].shape[0]), budget.max_review_points),
        "unassigned": min(
            int(layer_positions["unassigned"].shape[0]), budget.max_unassigned_points
        ),
        "conflict": min(
            int(layer_positions["conflict"].shape[0]), budget.max_conflict_points
        ),
        "apex": int(
            sum(len(_tree_apexes(result, tree)) for tree in result.tree_records)
        ),
        "trunk": int(
            sum(len(_tree_accepted_trunks(result, tree)) for tree in result.tree_records)
        ),
    }

    output_paths = {
        "html": str(html_path),
        "summary_json": str(summary_path) if summary_path is not None else None,
    }
    summary = build_summary(
        result=result,
        layer_positions=layer_positions,
        assigned_positions_by_tree=assigned_positions_by_tree,
        displayed_counts=displayed_counts,
        budget=budget,
        output_paths=output_paths,
    )
    html = render_html(result, budget)
    _atomic_write_text(html_path, html)
    if summary_path is not None:
        write_summary_json(summary, summary_path)
    return V4TreeVisualizationResult(
        html_path=html_path,
        summary_path=summary_path,
        summary=summary,
    )


__all__ = [
    "TOOL_NAME",
    "SUMMARY_SCHEMA_VERSION",
    "DEFAULT_MAX_POINTS_PER_TREE",
    "DEFAULT_MAX_REVIEW_POINTS",
    "DEFAULT_MAX_UNASSIGNED_POINTS",
    "DEFAULT_MAX_CONFLICT_POINTS",
    "V4TreeVisualizationError",
    "V4TreeVisualizationBudgetError",
    "V4TreeVisualizationOutputError",
    "V4TreeVisualizationBudget",
    "V4TreeVisualizationResult",
    "tree_color",
    "validate_display_budget",
    "deterministic_tree_sample_positions",
    "build_summary",
    "write_summary_json",
    "render_html",
    "run_v4_tree_visualization",
]
