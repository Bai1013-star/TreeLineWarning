"""A4 deterministic sliced wire separation; it reads A3 artifacts only, never LAS."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import (
    CancellationToken,
    CandidateLevel,
    PipelineProgress,
    ProgressReporter,
    UnifiedError,
    WireAssignmentQualityFlag,
    WireAssignmentReasonFlag,
    WireAssignmentSet,
    WireConfirmationChannel,
    WireInstanceSet,
    WireSeedConfig,
    WireSeparationConfig,
    WireSeparationResult,
    WireSeparationStatistics,
    WireTrackQualityFlag,
    WireTrackSource,
    WireTrackStatus,
    WireTrackSummary,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "a4-sliced-hungarian-wire-separation-v2"
_BASE = "scene_simple_wire_instances"
_BIG_COST = 1_000_000.0
_FORMULAS = {
    "primary_slice_id": "clip(floor((s-s_min)/slice_step_m), 0, slice_count-1)",
    "association_cost": (
        "0.30*du + 0.15*dv + 0.20*prediction_error + 0.10*direction_error "
        "+ 0.10*score_penalty + 0.15*gap_penalty"
    ),
    "coverage_ratio": "occupied_slice_count / max(floor(longitudinal_span_m/slice_step_m)+1, 1)",
    "curvature_proxy": "p90 of adjacent center-segment turning angles in radians",
    "maximum_gap_m": "maximum consecutive missing slice count * slice_step_m",
    "axis_angle_deg": "degrees(arccos(clamp(abs(dot(actual_main_direction, s_axis)), 0, 1)))",
    "obvious_confirmation": "all obvious_* thresholds in WireSeparationConfig",
    "fine_confirmation": "all fine_* thresholds in WireSeparationConfig",
}


class WireSeparationError(Exception):
    def __init__(self, error: UnifiedError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True)
class _Cluster:
    primary_indices: np.ndarray
    support_indices: np.ndarray
    slice_id: int
    center_u: float
    center_v: float
    mean_s: float
    score: float
    direction: np.ndarray
    high_count: int
    medium_count: int
    low_count: int
    weak: bool
    manual_group: int | None
    diameter: float


@dataclass
class _Track:
    source: WireTrackSource
    manual_group: int | None
    clusters: list[_Cluster] = field(default_factory=list)
    reasons: list[int] = field(default_factory=list)
    branch_conflicts: int = 0
    initial_slice: int = 0
    pre_confirmation_status: WireTrackStatus | None = None

    @property
    def last(self) -> _Cluster:
        return self.clusters[-1]


def run_wire_separation(
    a3_npz_path: Path | str,
    a3_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: WireSeparationConfig | None = None,
    seeds: WireSeedConfig | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
    generate_preview: bool = True,
) -> WireSeparationResult:
    """Assign A3 candidates to deterministic wire instances without curve fitting."""

    config = config or WireSeparationConfig()
    seeds = seeds or WireSeedConfig()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output_dir = Path(output_dir).resolve()
    _validate_output_dir(output_dir, root / "outputs" / "wires")
    output_npz = output_dir / f"{_BASE}.npz"
    output_json = output_dir / f"{_BASE}.json"
    _validate_outputs((output_npz, output_json), overwrite)
    _check_cancelled(cancellation_token, "before input validation")
    started = time.perf_counter()
    process = _current_process() if config.record_memory else None
    rss_start = _rss(process)
    a3_path, a3_json_path = Path(a3_npz_path).resolve(), Path(a3_json_path).resolve()
    data, metadata, input_hashes = _load_a3(a3_path, a3_json_path, config)
    _report(reporter, config.source_id, "validated_input", 0, len(data["xyz_local"]))
    state = _prepare_state(data, config, seeds)
    _check_cancelled(cancellation_token, "after axis estimation")
    clusters = _build_clusters(state, data, config, cancellation_token)
    _report(reporter, config.source_id, "clustered_slices", len(clusters), len(clusters))
    tracks, unassigned = _associate_clusters(
        state, data, clusters, config, seeds, cancellation_token
    )
    _extend_bidirectionally(state, data, tracks, unassigned, config)
    result, arrays = _finalize(
        data,
        metadata,
        input_hashes,
        state,
        tracks,
        unassigned,
        config,
        seeds,
        a3_path,
        a3_json_path,
        output_npz,
        output_json,
        started,
        rss_start,
        process,
        len(clusters),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _publish(arrays, result, output_npz, output_json, started, process)
    if generate_preview:
        try:
            generate_wire_instances_preview(
                a3_path,
                output_npz,
                root / "outputs" / "previews" / f"{_BASE}_preview.html",
                overwrite=overwrite,
                project_root=root,
            )
        except (
            Exception
        ) as error:  # Core wire artifacts remain valid if visualization is unavailable.
            result = replace(result, warnings=(f"preview generation failed: {error}",))
    _report(
        reporter,
        config.source_id,
        "completed",
        len(data["xyz_local"]),
        len(data["xyz_local"]),
        "completed",
    )
    return result


def generate_wire_instances_preview(
    a3_npz_path: Path | str,
    a4_npz_path: Path | str,
    output_html_path: Path | str,
    *,
    overwrite: bool = False,
    project_root: Path | str | None = None,
    max_background_points: int = 40_000,
    max_candidate_points: int = 50_000,
) -> Path:
    """Write a bounded, deterministic Plotly preview from A3/A4 NPZ artifacts only."""

    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output = Path(output_html_path).resolve()
    _validate_output_dir(output.parent, root / "outputs" / "previews")
    if output.exists() and not overwrite:
        raise _error("WIRE_PREVIEW_EXISTS", "wire preview exists and overwrite is disabled")
    with np.load(Path(a3_npz_path), allow_pickle=False) as stream:
        a3 = {name: stream[name] for name in ("xyz_local", "candidate_mask", "candidate_score")}
    with np.load(Path(a4_npz_path), allow_pickle=False) as stream:
        a4 = {
            name: stream[name]
            for name in (
                "xyz_local",
                "track_id_by_point",
                "wire_id_by_point",
                "assignment_score",
                "track_status",
                "track_source",
                "track_span_m",
                "track_axis_angle_deg",
            )
        }
    if not np.array_equal(a3["xyz_local"], a4["xyz_local"]):
        raise _error("WIRE_PREVIEW_ALIGNMENT", "A3 and A4 XYZ arrays are not aligned")
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise _error(
            "WIRE_PREVIEW_DEPENDENCY", "Plotly is unavailable", cause=str(error)
        ) from error
    xyz, candidate_mask = a3["xyz_local"], a3["candidate_mask"]
    track_id, wire_id = a4["track_id_by_point"], a4["wire_id_by_point"]
    background = _even(np.flatnonzero(~candidate_mask), max_background_points)
    candidates = np.flatnonzero(candidate_mask)
    if len(candidates) > max_candidate_points:
        order = np.lexsort((candidates, -a3["candidate_score"][candidates]))
        candidates = candidates[order[:max_candidate_points]]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[background, 0],
            y=xyz[background, 1],
            z=xyz[background, 2],
            mode="markers",
            name="background",
            marker={"size": 1.2, "color": "#6b7280", "opacity": 0.12},
            hoverinfo="skip",
        )
    )
    unassigned = candidates[track_id[candidates] < 0]
    if len(unassigned):
        figure.add_trace(
            go.Scatter3d(
                x=xyz[unassigned, 0],
                y=xyz[unassigned, 1],
                z=xyz[unassigned, 2],
                mode="markers",
                name="unassigned candidates",
                marker={"size": 1.2, "color": "#9ca3af", "opacity": 0.12},
                hoverinfo="skip",
                visible="legendonly",
            )
        )
    palette = [
        "#dc2626",
        "#2563eb",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
    ]
    for identifier in sorted(int(value) for value in np.unique(wire_id[candidates]) if value >= 0):
        index = candidates[wire_id[candidates] == identifier]
        identifier_track_ids = np.unique(track_id[index])
        if len(identifier_track_ids) != 1:
            raise _error("WIRE_PREVIEW_MAPPING", "confirmed wire maps to multiple tracks")
        identifier_track_id = int(identifier_track_ids[0])
        source = int(a4["track_source"][identifier_track_id])
        span = float(a4["track_span_m"][identifier_track_id])
        angle = float(a4["track_axis_angle_deg"][identifier_track_id])
        figure.add_trace(
            go.Scatter3d(
                x=xyz[index, 0],
                y=xyz[index, 1],
                z=xyz[index, 2],
                mode="markers",
                name=f"wire {identifier}",
                marker={
                    "size": 2.7,
                    "color": palette[identifier % len(palette)],
                    "opacity": 0.9,
                    "symbol": "circle",
                },
                customdata=np.column_stack(
                    (
                        np.full(len(index), identifier_track_id),
                        np.full(len(index), identifier),
                        np.full(len(index), source),
                        a4["assignment_score"][index],
                        np.full(len(index), span),
                        np.full(len(index), angle),
                    )
                ),
                hovertemplate=(
                    "track=%{customdata[0]}<br>wire=%{customdata[1]}"
                    "<br>status=confirmed<br>source=%{customdata[2]}"
                    "<br>assignment=%{customdata[3]:.3f}"
                    "<br>span=%{customdata[4]:.2f} m"
                    "<br>axis angle=%{customdata[5]:.2f} deg<extra></extra>"
                ),
            )
        )
    for status, label, color, size, opacity in (
        (WireTrackStatus.REVIEW_REQUIRED, "review tracklets", "#f59e0b", 1.5, 0.22),
        (WireTrackStatus.PROVISIONAL, "provisional tracklets", "#94a3b8", 1.1, 0.15),
    ):
        status_by_point = np.full(len(track_id), -1, dtype=np.int16)
        assigned = track_id >= 0
        status_by_point[assigned] = a4["track_status"][track_id[assigned]]
        index = candidates[status_by_point[candidates] == int(status)]
        point_track_id = track_id[index]
        figure.add_trace(
            go.Scatter3d(
                x=xyz[index, 0],
                y=xyz[index, 1],
                z=xyz[index, 2],
                mode="markers",
                name=label,
                marker={"size": size, "color": color, "opacity": opacity},
                visible="legendonly",
                customdata=np.column_stack(
                    (
                        point_track_id,
                        wire_id[index],
                        a4["assignment_score"][index],
                        a4["track_span_m"][point_track_id],
                        a4["track_axis_angle_deg"][point_track_id],
                    )
                ),
                hovertemplate=(
                    f"track=%{{customdata[0]}}<br>wire=%{{customdata[1]}}"
                    f"<br>status={label}<br>assignment=%{{customdata[2]:.3f}}"
                    "<br>span=%{customdata[3]:.2f} m"
                    "<br>axis angle=%{customdata[4]:.2f} deg<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Scene Simple A4 Wire Instances",
        scene={
            "xaxis_title": "local X (m)",
            "yaxis_title": "local Y (m)",
            "zaxis_title": "local Z (m)",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(output)
    try:
        figure.write_html(temporary, include_plotlyjs=True, full_html=True)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _prepare_state(
    data: Mapping[str, np.ndarray], config: WireSeparationConfig, seeds: WireSeedConfig
) -> dict[str, Any]:
    input_candidate = np.asarray(data["candidate_mask"], dtype=bool)
    candidate = input_candidate.copy()
    original = data["original_index"]
    excluded = np.isin(original, np.asarray(seeds.excluded_original_indices, dtype=np.int64))
    candidate &= ~excluded
    manual_groups = np.full(len(candidate), -1, dtype=np.int32)
    for index, group in zip(seeds.seed_original_indices, seeds.seed_group_ids, strict=True):
        row = np.flatnonzero(original == index)
        if not len(row):
            raise _error(
                "WIRE_SEED_INDEX", "manual seed original_index is missing", original_index=index
            )
        manual_groups[row[0]] = group
    for group, lower, upper in seeds.local_xyz_seed_boxes:
        within = np.all((data["xyz_local"] >= lower) & (data["xyz_local"] <= upper), axis=1)
        manual_groups[within] = group
    if np.any((manual_groups >= 0) & ~data["candidate_mask"]):
        raise _error("WIRE_SEED_NON_CANDIDATE", "manual seed must select A3 candidate points")
    high = candidate & (data["candidate_level"] == int(CandidateLevel.HIGH))
    directions = data["principal_direction"].astype(np.float64)
    usable = (
        high & np.isfinite(directions).all(axis=1) & (np.linalg.norm(directions, axis=1) > 1e-8)
    )
    if int(usable.sum()) < config.min_axis_seed_points:
        raise _error(
            "WIRE_AXIS_SEEDS", "insufficient high-confidence axis seeds", count=int(usable.sum())
        )
    selected = np.flatnonzero(usable)
    xyz = data["xyz_local"].astype(np.float64)
    scores = data["candidate_score"].astype(np.float64)
    axis = np.array([1.0, 0.0, 0.0])
    center = np.zeros(3)
    for _ in range(config.robust_axis_iterations):
        if len(selected) < config.min_axis_seed_points:
            raise _error("WIRE_AXIS_ROBUST", "robust axis filtering removed too many seeds")
        weights = np.maximum(scores[selected], 1e-6)
        center = np.average(xyz[selected], axis=0, weights=weights)
        relative = xyz[selected] - center
        covariance = (relative * weights[:, None]).T @ relative / weights.sum()
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, np.argmax(values)]
        axis = _signed_unit(axis)
        transverse = np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1)
        median = float(np.median(transverse))
        mad = float(np.median(np.abs(transverse - median)))
        limit = median + config.robust_axis_mad_multiplier * max(1.4826 * mad, 1e-6)
        agreement = np.abs(_unit_rows(directions[selected]) @ axis)
        keep = (transverse <= limit) & (agreement >= config.robust_axis_direction_cosine)
        selected = selected[keep]
    span = float(np.ptp((xyz[selected] - center) @ axis))
    if len(selected) < config.min_axis_seed_points or span < config.min_axis_span_m:
        raise _error("WIRE_AXIS_SPAN", "axis seed longitudinal span is insufficient", span_m=span)
    z_axis = np.array([0.0, 0.0, 1.0])
    u_axis = np.cross(z_axis, axis)
    if np.linalg.norm(u_axis) < 1e-8:
        u_axis = np.cross(np.array([1.0, 0.0, 0.0]), axis)
    if np.linalg.norm(u_axis) < 1e-8:
        u_axis = np.cross(np.array([0.0, 1.0, 0.0]), axis)
    u_axis = _unit(u_axis)
    v_axis = _unit(np.cross(axis, u_axis))
    if float(v_axis @ z_axis) < 0.0:
        u_axis, v_axis = -u_axis, -v_axis
    relative_all = xyz - center
    s, u, v = relative_all @ axis, relative_all @ u_axis, relative_all @ v_axis
    candidate_rows = np.flatnonzero(candidate)
    s_min, s_max = float(s[candidate_rows].min()), float(s[candidate_rows].max())
    slice_count = max(1, int(math.floor((s_max - s_min) / config.slice_step_m)) + 1)
    primary = np.full(len(candidate), -1, dtype=np.int32)
    primary[candidate_rows] = np.minimum(
        np.floor((s[candidate_rows] - s_min) / config.slice_step_m + 1e-9).astype(np.int32),
        slice_count - 1,
    )
    return {
        "candidate": candidate,
        "input_candidate": input_candidate,
        "excluded": excluded,
        "manual_groups": manual_groups,
        "axis": axis,
        "center": center,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "s": s,
        "u": u,
        "v": v,
        "s_min": s_min,
        "slice_count": slice_count,
        "primary": primary,
        "candidate_rows": candidate_rows,
    }


def _build_clusters(
    state: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    config: WireSeparationConfig,
    token: CancellationToken | None,
) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    primary, s, u, v = state["primary"], state["s"], state["u"], state["v"]
    for slice_id in range(state["slice_count"]):
        _check_cancelled(token, "during slice clustering")
        formal = np.flatnonzero(primary == slice_id)
        if not len(formal):
            continue
        center_s = state["s_min"] + (slice_id + 0.5) * config.slice_step_m
        support = state["candidate_rows"][
            np.abs(s[state["candidate_rows"]] - center_s) <= config.slice_length_m / 2.0 + 1e-8
        ]
        components = _components(support, u, v, config)
        for component in components:
            own = component[primary[component] == slice_id]
            if not len(own):
                continue
            for part in _split_large(own, u, v, config):
                high = int(
                    np.count_nonzero(data["candidate_level"][part] == int(CandidateLevel.HIGH))
                )
                medium = int(
                    np.count_nonzero(data["candidate_level"][part] == int(CandidateLevel.MEDIUM))
                )
                low = int(
                    np.count_nonzero(data["candidate_level"][part] == int(CandidateLevel.LOW))
                )
                groups = state["manual_groups"][part]
                manual_group = int(groups[groups >= 0][0]) if np.any(groups >= 0) else None
                if (
                    len(part) < config.weak_cluster_min_points
                    and high == 0
                    and manual_group is None
                ):
                    continue
                direction = _mean_direction(data["principal_direction"][part], state["axis"])
                uv = np.column_stack((u[part], v[part]))
                diameter = (
                    float(np.max(np.linalg.norm(uv - uv.mean(axis=0), axis=1)) * 2.0)
                    if len(part) > 1
                    else 0.0
                )
                clusters.append(
                    _Cluster(
                        part,
                        component,
                        slice_id,
                        float(np.mean(u[part])),
                        float(np.mean(v[part])),
                        float(np.mean(s[part])),
                        float(np.mean(data["candidate_score"][part])),
                        direction,
                        high,
                        medium,
                        low,
                        len(part) < config.confirmed_cluster_min_points,
                        manual_group,
                        diameter,
                    )
                )
    return sorted(
        clusters,
        key=lambda item: (
            item.slice_id,
            item.center_u,
            item.center_v,
            int(data["original_index"][item.primary_indices].min()),
        ),
    )


def _associate_clusters(
    state: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    clusters: list[_Cluster],
    config: WireSeparationConfig,
    seeds: WireSeedConfig,
    token: CancellationToken | None,
) -> tuple[list[_Track], list[_Cluster]]:
    by_slice: dict[int, list[_Cluster]] = {}
    for cluster in clusters:
        by_slice.setdefault(cluster.slice_id, []).append(cluster)
    tracks: list[_Track] = []
    unassigned: list[_Cluster] = []
    manual_tracks: dict[int, _Track] = {}
    for slice_id in range(state["slice_count"]):
        _check_cancelled(token, "during track association")
        current = list(by_slice.get(slice_id, []))
        manual = [cluster for cluster in current if cluster.manual_group is not None]
        current = [cluster for cluster in current if cluster.manual_group is None]
        for cluster in manual:
            track = manual_tracks.get(cluster.manual_group)
            if track is None:
                track = _Track(
                    WireTrackSource.MANUAL_SEED, cluster.manual_group, initial_slice=slice_id
                )
                manual_tracks[cluster.manual_group] = track
                tracks.append(track)
                reason = int(WireAssignmentReasonFlag.MANUAL_SEED)
            else:
                reason = int(
                    WireAssignmentReasonFlag.MANUAL_SEED | WireAssignmentReasonFlag.TRACK_EXTENSION
                )
            track.clusters.append(cluster)
            track.reasons.append(reason)
        active = [
            track
            for track in tracks
            if track.clusters and 0 < slice_id - track.last.slice_id <= config.max_gap_slices
        ]
        costs = np.full((len(active), len(current)), _BIG_COST, dtype=np.float64)
        alternatives: list[int] = [0] * len(active)
        for row, track in enumerate(active):
            legal_costs = []
            for column, cluster in enumerate(current):
                cost = _association_cost(track, cluster, slice_id, config)
                if cost is not None:
                    costs[row, column] = cost
                    legal_costs.append(cost)
            if legal_costs:
                alternatives[row] = (
                    sum(
                        cost <= min(legal_costs) + config.branch_cost_closeness
                        for cost in legal_costs
                    )
                    - 1
                )
        matched_clusters: set[int] = set()
        if len(active) and len(current):
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] > config.max_association_cost:
                    continue
                track, cluster = active[row], current[column]
                track.branch_conflicts += max(0, alternatives[row])
                reason = int(
                    WireAssignmentReasonFlag.TRACK_EXTENSION
                    | WireAssignmentReasonFlag.HUNGARIAN_MATCH
                )
                if slice_id - track.last.slice_id > 1:
                    reason |= int(WireAssignmentReasonFlag.BIDIRECTIONAL_EXTENSION)
                if cluster.low_count and not (cluster.high_count or cluster.medium_count):
                    reason |= int(WireAssignmentReasonFlag.LOW_CONFIDENCE_GAP_FILL)
                track.clusters.append(cluster)
                track.reasons.append(reason)
                if track.source == WireTrackSource.MANUAL_SEED:
                    track.source = WireTrackSource.HYBRID
                matched_clusters.add(column)
        for column, cluster in enumerate(current):
            if column in matched_clusters:
                continue
            if cluster.high_count:
                track = _Track(
                    WireTrackSource.AUTOMATIC,
                    None,
                    [cluster],
                    [
                        int(
                            WireAssignmentReasonFlag.HIGH_CONFIDENCE_SEED
                            | WireAssignmentReasonFlag.PROVISIONAL_TRACK
                        )
                    ],
                    initial_slice=slice_id,
                )
                tracks.append(track)
            elif (
                cluster.medium_count
                and len(cluster.primary_indices) >= config.weak_cluster_min_points
            ):
                track = _Track(
                    WireTrackSource.AUTOMATIC,
                    None,
                    [cluster],
                    [
                        int(
                            WireAssignmentReasonFlag.MEDIUM_CONFIDENCE_SEED
                            | WireAssignmentReasonFlag.PROVISIONAL_TRACK
                        )
                    ],
                    initial_slice=slice_id,
                )
                tracks.append(track)
            else:
                unassigned.append(cluster)
    return tracks, unassigned


def _extend_bidirectionally(
    state: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    tracks: list[_Track],
    unassigned: list[_Cluster],
    config: WireSeparationConfig,
) -> None:
    """Reverse-order pass attaches only eligible orphan low/medium clusters to stable tracks."""

    retained: list[_Cluster] = []
    for cluster in sorted(
        unassigned, key=lambda item: (item.slice_id, item.center_u, item.center_v), reverse=True
    ):
        candidates = [track for track in tracks if len(track.clusters) >= 2]
        scored = [(_endpoint_cost(track, cluster, config), track) for track in candidates]
        scored = [(cost, track) for cost, track in scored if cost is not None]
        if not scored:
            retained.append(cluster)
            continue
        cost, track = min(
            scored, key=lambda item: (item[0], item[1].last.slice_id, item[1].initial_slice)
        )
        if cost > config.max_association_cost or not (cluster.medium_count or cluster.low_count):
            retained.append(cluster)
            continue
        if cluster.low_count and not cluster.medium_count:
            if (
                abs(cluster.center_u - track.last.center_u) > config.low_max_transverse_distance_m
                or abs(cluster.center_v - track.last.center_v) > config.low_max_height_distance_m
            ):
                retained.append(cluster)
                continue
        track.clusters.append(cluster)
        track.reasons.append(
            int(
                WireAssignmentReasonFlag.TRACK_EXTENSION
                | WireAssignmentReasonFlag.BIDIRECTIONAL_EXTENSION
                | (
                    WireAssignmentReasonFlag.LOW_CONFIDENCE_GAP_FILL
                    if cluster.low_count and not cluster.medium_count
                    else WireAssignmentReasonFlag.NONE
                )
            )
        )
    for track in tracks:
        paired = sorted(
            zip(track.clusters, track.reasons, strict=True), key=lambda item: item[0].slice_id
        )
        track.clusters = [cluster for cluster, _ in paired]
        track.reasons = [reason for _, reason in paired]
    unassigned[:] = retained


def _finalize(
    data: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    state: Mapping[str, Any],
    tracks: list[_Track],
    unassigned: list[_Cluster],
    config: WireSeparationConfig,
    seeds: WireSeedConfig,
    a3_path: Path,
    a3_json_path: Path,
    output_npz: Path,
    output_json: Path,
    started: float,
    rss_start: int | None,
    process: Any | None,
    cluster_count: int,
) -> tuple[WireSeparationResult, dict[str, np.ndarray]]:
    summaries_with_tracks = []
    for track in tracks:
        if not track.clusters:
            continue
        summary = _summary(track, data, state, config)
        summaries_with_tracks.append((summary, track))
    summaries_with_tracks.sort(
        key=lambda item: (
            item[0].median_u,
            -item[0].median_v,
            item[0].first_s,
            int(
                data["original_index"][
                    np.concatenate([cluster.primary_indices for cluster in item[1].clusters])
                ].min()
            ),
        )
    )
    track_assignments = np.full(len(data["xyz_local"]), -1, dtype=np.int32)
    wire_assignments = np.full(len(data["xyz_local"]), -1, dtype=np.int32)
    scores = np.zeros(len(track_assignments), dtype=np.float32)
    reasons = np.zeros(len(track_assignments), dtype=np.uint16)
    qualities = np.zeros(len(track_assignments), dtype=np.uint16)
    ordered_tracks: list[np.ndarray] = []
    ordered_wires: list[np.ndarray] = []
    summaries: list[WireTrackSummary] = []
    pre_confirmation_statuses: list[WireTrackStatus] = []
    next_wire_id = 0
    for track_id, (old_summary, track) in enumerate(summaries_with_tracks):
        wire_id = next_wire_id if old_summary.status == WireTrackStatus.CONFIRMED else -1
        if wire_id >= 0:
            next_wire_id += 1
        summary = replace(old_summary, track_id=track_id, wire_id=wire_id)
        summaries.append(summary)
        pre_confirmation_statuses.append(track.pre_confirmation_status or summary.status)
        points = np.unique(np.concatenate([cluster.primary_indices for cluster in track.clusters]))
        points = points[np.argsort(state["s"][points], kind="stable")]
        if np.any(track_assignments[points] >= 0):
            raise _error("WIRE_DUPLICATE_ASSIGNMENT", "a candidate was assigned to multiple tracks")
        track_assignments[points] = track_id
        if wire_id >= 0:
            wire_assignments[points] = wire_id
            ordered_wires.append(points.astype(np.int64))
        ordered_tracks.append(points.astype(np.int64))
        for cluster, reason in zip(track.clusters, track.reasons, strict=True):
            idx = cluster.primary_indices
            scores[idx] = np.clip(
                data["candidate_score"][idx]
                * (1.0 - min(cluster.diameter / config.max_cluster_diameter_m, 1.0) * 0.15),
                0.0,
                1.0,
            )
            reasons[idx] |= reason
            if summary.status != WireTrackStatus.CONFIRMED:
                qualities[idx] |= int(WireAssignmentQualityFlag.TRACK_PROVISIONAL)
            if summary.quality_flags & int(WireTrackQualityFlag.NUMERIC_INVALID):
                qualities[idx] |= int(WireAssignmentQualityFlag.INVALID_INPUT)
            if np.any(data["candidate_level"][idx] == int(CandidateLevel.LOW)):
                qualities[idx[data["candidate_level"][idx] == int(CandidateLevel.LOW)]] |= int(
                    WireAssignmentQualityFlag.LOW_ASSIGNMENT_SCORE
                )
            if cluster.diameter > config.max_cluster_diameter_m * 0.75:
                qualities[idx] |= int(WireAssignmentQualityFlag.LARGE_LOCAL_RESIDUAL)
    reasons[state["excluded"] & state["input_candidate"]] |= int(
        WireAssignmentReasonFlag.MANUAL_EXCLUDE
    )
    reasons[(track_assignments < 0) & state["input_candidate"] & ~state["excluded"]] |= int(
        WireAssignmentReasonFlag.UNASSIGNED_OUTLIER
    )
    track_offsets = _offsets(ordered_tracks)
    wire_offsets = _offsets(ordered_wires)
    ordered_track_indices = _ordered_indices(ordered_tracks)
    ordered_wire_indices = _ordered_indices(ordered_wires)
    assignment_set = WireAssignmentSet(
        str(data["source_id"].item()),
        data["original_index"],
        data["sample_index"],
        data["xyz_local"],
        track_assignments,
        wire_assignments,
        scores,
        reasons,
        qualities,
    )
    instances = WireInstanceSet(
        ordered_track_indices,
        track_offsets,
        ordered_wire_indices,
        wire_offsets,
        tuple(summaries),
    )
    status_counts = {
        status: sum(summary.status == status for summary in summaries) for status in WireTrackStatus
    }
    confirmation_transitions = {
        "pre_confirmation_confirmed_count": int(
            sum(status == WireTrackStatus.CONFIRMED for status in pre_confirmation_statuses)
        ),
        "post_confirmation_confirmed_count": status_counts[WireTrackStatus.CONFIRMED],
        "demoted_confirmed_to_temporary_count": int(
            sum(
                before == WireTrackStatus.CONFIRMED and after.status != WireTrackStatus.CONFIRMED
                for before, after in zip(pre_confirmation_statuses, summaries, strict=True)
            )
        ),
        "promoted_temporary_to_confirmed_count": int(
            sum(
                before != WireTrackStatus.CONFIRMED and after.status == WireTrackStatus.CONFIRMED
                for before, after in zip(pre_confirmation_statuses, summaries, strict=True)
            )
        ),
    }
    stats = WireSeparationStatistics(
        len(track_assignments),
        int(data["candidate_mask"].sum()),
        state["slice_count"],
        cluster_count,
        len(tracks),
        len(summaries),
        status_counts[WireTrackStatus.CONFIRMED],
        status_counts[WireTrackStatus.PROVISIONAL],
        status_counts[WireTrackStatus.REVIEW_REQUIRED],
        int(np.count_nonzero(track_assignments >= 0)),
        int(np.count_nonzero(wire_assignments >= 0)),
        int(np.count_nonzero((track_assignments < 0) & data["candidate_mask"])),
        0,
        0,
        time.perf_counter() - started,
        rss_start,
        _rss(process),
        _rss(process),
    )
    arrays = _arrays(assignment_set, instances, summaries, state)
    result = WireSeparationResult(
        str(data["source_id"].item()),
        str(data["manifest_source_id"].item()),
        a3_path,
        a3_json_path,
        input_hashes["npz"],
        input_hashes["json"],
        output_npz,
        output_json,
        "0" * 64,
        0,
        tuple(float(x) for x in data["origin_xyz"]),
        tuple(float(x) for x in state["center"]),
        tuple(float(x) for x in state["axis"]),
        tuple(float(x) for x in state["u_axis"]),
        tuple(float(x) for x in state["v_axis"]),
        config,
        seeds,
        stats,
        assignment_set.array_metadata()
        | {
            "ordered_track_point_indices": _meta(
                instances.ordered_track_point_indices, "row_index"
            ),
            "track_offsets": _meta(instances.track_offsets, "row_offset"),
            "ordered_wire_point_indices": _meta(instances.ordered_wire_point_indices, "row_index"),
            "wire_offsets": _meta(instances.wire_offsets, "row_offset"),
        },
        tuple(summaries),
        _flag_counts(reasons, WireAssignmentReasonFlag),
        _flag_counts(qualities, WireAssignmentQualityFlag),
        _track_flag_counts(summaries),
        confirmation_transitions,
        _FORMULAS,
        (),
        ALGORITHM_VERSION,
        datetime.now(UTC).isoformat(),
    )
    return result, arrays


def _publish(
    arrays: Mapping[str, np.ndarray],
    result: WireSeparationResult,
    output_npz: Path,
    output_json: Path,
    started: float,
    process: Any | None,
) -> WireSeparationResult:
    temporary_npz, temporary_json = _temporary(output_npz), _temporary(output_json)
    published = False
    try:
        np.savez_compressed(temporary_npz, **arrays)
        rss_values = [
            value
            for value in (
                result.statistics.rss_start_bytes,
                result.statistics.rss_peak_bytes,
                _rss(process),
            )
            if value is not None
        ]
        result = replace(
            result,
            output_npz_sha256=_sha(temporary_npz),
            output_npz_bytes=temporary_npz.stat().st_size,
            statistics=replace(
                result.statistics,
                elapsed_seconds=time.perf_counter() - started,
                rss_peak_bytes=max(rss_values) if rss_values else None,
                rss_end_bytes=_rss(process),
            ),
        )
        temporary_json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_npz, output_npz)
        published = True
        os.replace(temporary_json, output_json)
        return result
    except Exception:
        temporary_npz.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
        if published:
            output_npz.unlink(missing_ok=True)
        raise


def _summary(
    track: _Track,
    data: Mapping[str, np.ndarray],
    state: Mapping[str, Any],
    config: WireSeparationConfig,
) -> WireTrackSummary:
    clusters = sorted(track.clusters, key=lambda item: item.slice_id)
    points = np.unique(np.concatenate([cluster.primary_indices for cluster in clusters]))
    points = points[np.argsort(state["s"][points], kind="stable")]
    s_values, u_values, v_values = state["s"][points], state["u"][points], state["v"][points]
    span = float(np.ptp(s_values)) if len(points) > 1 else 0.0
    occupied = len({cluster.slice_id for cluster in clusters})
    expected = int(math.floor(span / config.slice_step_m + 1e-6)) + 1
    # A global bin phase can touch one extra endpoint bin; coverage remains a ratio.
    coverage = min(occupied / max(expected, 1), 1.0)
    directions = _unit_rows(data["principal_direction"][points])
    consistency = float(np.mean(np.abs(directions @ state["axis"])))
    if len(points) >= 2 and np.ptp(s_values) > 1e-8:
        coefficients = np.polyfit(s_values, u_values, 1)
        residual = float(np.sqrt(np.mean((u_values - np.polyval(coefficients, s_values)) ** 2)))
    else:
        residual = 0.0
    centers = np.array(
        [[cluster.mean_s, cluster.center_u, cluster.center_v] for cluster in clusters]
    )
    if len(centers) >= 3:
        segments = np.diff(centers, axis=0)
        unit_segments = _unit_rows(segments)
        angles = (
            np.arccos(np.clip(np.sum(unit_segments[:-1] * unit_segments[1:], axis=1), -1.0, 1.0))
            if len(unit_segments) > 1
            else np.empty(0)
        )
        curvature = float(np.percentile(angles, 90)) if len(angles) else 0.0
    else:
        curvature = 0.0
    gaps = np.diff(sorted({cluster.slice_id for cluster in clusters})) - 1
    max_gap = int(max(gaps, default=0))
    actual_direction, axis_angle_deg, direction_invalid = _actual_track_direction(
        data["xyz_local"][points], state["axis"]
    )
    flags = 0
    if span < config.confirmation_min_span_m or occupied < config.confirmation_min_occupied_slices:
        flags |= int(WireTrackQualityFlag.PROVISIONAL_SHORT_TRACK)
    if coverage < config.confirmation_min_coverage_ratio:
        flags |= int(WireTrackQualityFlag.LOW_COVERAGE)
    if max_gap > config.max_gap_slices:
        flags |= int(WireTrackQualityFlag.LARGE_GAP)
    if consistency < config.confirmation_min_direction_consistency:
        flags |= int(WireTrackQualityFlag.LOW_DIRECTION_CONSISTENCY)
    if residual > config.high_transverse_residual_m:
        flags |= int(WireTrackQualityFlag.HIGH_TRANSVERSE_RESIDUAL)
    if curvature > config.high_curvature_p90_rad:
        flags |= int(WireTrackQualityFlag.HIGH_CURVATURE)
    if axis_angle_deg > config.fine_max_axis_angle_deg:
        flags |= int(WireTrackQualityFlag.OFF_DOMINANT_AXIS)
    if track.branch_conflicts:
        flags |= int(WireTrackQualityFlag.BRANCH_CONFLICT)
    if direction_invalid:
        flags |= int(WireTrackQualityFlag.NUMERIC_INVALID)
    mean_score = float(np.mean(data["candidate_score"][points]))
    if mean_score < config.confirmation_min_mean_score:
        flags |= int(WireTrackQualityFlag.LOW_MEAN_SCORE)
    if track.source != WireTrackSource.AUTOMATIC:
        flags |= int(WireTrackQualityFlag.MANUAL_SEEDED)
    high_fraction = float(np.mean(data["candidate_level"][points] == int(CandidateLevel.HIGH)))
    medium_fraction = float(np.mean(data["candidate_level"][points] == int(CandidateLevel.MEDIUM)))
    low_fraction = float(np.mean(data["candidate_level"][points] == int(CandidateLevel.LOW)))
    medium_only = high_fraction == 0.0
    consecutive = _max_consecutive(sorted({cluster.slice_id for cluster in clusters}))
    legacy_confirmed = (
        span >= config.confirmation_min_span_m
        and occupied >= config.confirmation_min_occupied_slices
        and coverage >= config.confirmation_min_coverage_ratio
        and consistency >= config.confirmation_min_direction_consistency
        and mean_score >= config.confirmation_min_mean_score
    )
    if medium_only:
        legacy_confirmed &= (
            consecutive >= config.medium_seed_min_consecutive_slices
            and span >= config.medium_seed_min_span_m
            and not track.branch_conflicts
        )
    track.pre_confirmation_status = (
        WireTrackStatus.CONFIRMED
        if legacy_confirmed
        else (
            WireTrackStatus.REVIEW_REQUIRED
            if flags
            & int(WireTrackQualityFlag.BRANCH_CONFLICT | WireTrackQualityFlag.HIGH_CURVATURE)
            else WireTrackStatus.PROVISIONAL
        )
    )
    channel = _confirmation_channel(
        span=span,
        coverage=coverage,
        direction_consistency=consistency,
        transverse_residual=residual,
        curvature_proxy=curvature,
        maximum_gap_m=max_gap * config.slice_step_m,
        branch_conflict_count=track.branch_conflicts,
        axis_angle_deg=axis_angle_deg,
        mean_candidate_score=mean_score,
        high_candidate_fraction=high_fraction,
        medium_low_fraction=medium_fraction + low_fraction,
        numeric_invalid=direction_invalid,
        config=config,
    )
    confirmed = channel != WireConfirmationChannel.NONE
    status = (
        WireTrackStatus.CONFIRMED
        if confirmed
        else (
            WireTrackStatus.REVIEW_REQUIRED
            if flags
            & int(WireTrackQualityFlag.BRANCH_CONFLICT | WireTrackQualityFlag.HIGH_CURVATURE)
            else WireTrackStatus.PROVISIONAL
        )
    )
    return WireTrackSummary(
        0,
        0 if confirmed else -1,
        status,
        track.source,
        len(points),
        span,
        occupied,
        float(coverage),
        mean_score,
        high_fraction,
        medium_fraction,
        low_fraction,
        consistency,
        residual,
        curvature,
        max_gap * config.slice_step_m,
        track.branch_conflicts,
        float(np.median(u_values)),
        float(np.median(v_values)),
        float(np.min(s_values)),
        tuple(float(value) for value in actual_direction),
        axis_angle_deg,
        channel,
        tuple(float(value) for value in data["xyz_local"][points[0]]),
        tuple(float(value) for value in data["xyz_local"][points[-1]]),
        flags,
    )


def _actual_track_direction(xyz: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Return an axis-oriented PCA direction, with an explicit degenerate sentinel."""

    values = np.asarray(xyz, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.zeros(3, dtype=np.float64), 90.0, True
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float64), 90.0, True
    if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 1e-12:
        return np.zeros(3, dtype=np.float64), 90.0, True
    direction = eigenvectors[:, -1]
    if float(direction @ axis) < 0.0:
        direction = -direction
    direction = _unit(direction)
    cosine = float(np.clip(abs(direction @ axis), 0.0, 1.0))
    return direction, float(math.degrees(math.acos(cosine))), False


def _confirmation_channel(
    *,
    span: float,
    coverage: float,
    direction_consistency: float,
    transverse_residual: float,
    curvature_proxy: float,
    maximum_gap_m: float,
    branch_conflict_count: int,
    axis_angle_deg: float,
    mean_candidate_score: float,
    high_candidate_fraction: float,
    medium_low_fraction: float,
    numeric_invalid: bool,
    config: WireSeparationConfig,
) -> WireConfirmationChannel:
    """Apply the approved post-association confirmation rules without ID-specific cases."""

    if numeric_invalid:
        return WireConfirmationChannel.NONE
    obvious = (
        span >= config.obvious_min_span_m
        and coverage >= config.obvious_min_coverage_ratio
        and direction_consistency >= config.obvious_min_direction_consistency
        and transverse_residual <= config.obvious_max_transverse_residual_m
        and curvature_proxy <= config.obvious_max_curvature_proxy_rad
        and maximum_gap_m <= config.obvious_max_gap_m
        and branch_conflict_count <= config.obvious_max_branch_conflicts
        and axis_angle_deg <= config.obvious_max_axis_angle_deg
        and mean_candidate_score >= config.obvious_min_mean_candidate_score
        and high_candidate_fraction >= config.obvious_min_high_candidate_fraction
    )
    if obvious:
        return WireConfirmationChannel.OBVIOUS
    fine = (
        span >= config.fine_min_span_m
        and coverage >= config.fine_min_coverage_ratio
        and direction_consistency >= config.fine_min_direction_consistency
        and transverse_residual <= config.fine_max_transverse_residual_m
        and curvature_proxy <= config.fine_max_curvature_proxy_rad
        and maximum_gap_m <= config.fine_max_gap_m
        and branch_conflict_count <= config.fine_max_branch_conflicts
        and axis_angle_deg <= config.fine_max_axis_angle_deg
        and mean_candidate_score >= config.fine_min_mean_candidate_score
        and medium_low_fraction >= config.fine_min_medium_low_fraction
    )
    return WireConfirmationChannel.FINE if fine else WireConfirmationChannel.NONE


def _association_cost(
    track: _Track, cluster: _Cluster, slice_id: int, config: WireSeparationConfig
) -> float | None:
    last = track.last
    gap = slice_id - last.slice_id - 1
    if gap < 0 or gap > config.max_gap_slices:
        return None
    if len(track.clusters) >= 2:
        previous = track.clusters[-2]
        step = max(last.slice_id - previous.slice_id, 1)
        predicted_u = (
            last.center_u + (last.center_u - previous.center_u) * (slice_id - last.slice_id) / step
        )
        predicted_v = (
            last.center_v + (last.center_v - previous.center_v) * (slice_id - last.slice_id) / step
        )
    else:
        predicted_u, predicted_v = last.center_u, last.center_v
    du = abs(cluster.center_u - last.center_u)
    dv = abs(cluster.center_v - last.center_v)
    prediction = math.hypot(cluster.center_u - predicted_u, cluster.center_v - predicted_v)
    low_only = cluster.low_count and not (cluster.high_count or cluster.medium_count)
    limit_u = config.low_max_transverse_distance_m if low_only else config.max_transverse_distance_m
    limit_v = config.low_max_height_distance_m if low_only else config.max_height_distance_m
    limit_prediction = (
        config.low_max_prediction_distance_m if low_only else config.max_prediction_distance_m
    )
    direction = 1.0 - abs(float(np.dot(cluster.direction, _track_direction(track))))
    if (
        du > limit_u
        or dv > limit_v
        or prediction > limit_prediction
        or (low_only and 1.0 - direction < config.low_min_direction_cosine)
    ):
        return None
    return (
        config.weight_transverse * du / config.max_transverse_distance_m
        + config.weight_height * dv / config.max_height_distance_m
        + config.weight_prediction * prediction / config.max_prediction_distance_m
        + config.weight_direction * direction
        + config.weight_score * (1.0 - cluster.score)
        + config.weight_gap * gap / config.max_gap_slices
    )


def _endpoint_cost(track: _Track, cluster: _Cluster, config: WireSeparationConfig) -> float | None:
    last = track.last
    gap = abs(cluster.slice_id - last.slice_id) - 1
    if gap < 0 or gap > config.max_gap_slices:
        return None
    du, dv = abs(cluster.center_u - last.center_u), abs(cluster.center_v - last.center_v)
    if du > config.max_transverse_distance_m or dv > config.max_height_distance_m:
        return None
    return (
        config.weight_transverse * du / config.max_transverse_distance_m
        + config.weight_height * dv / config.max_height_distance_m
        + config.weight_direction
        * (1.0 - abs(float(np.dot(cluster.direction, _track_direction(track)))))
        + config.weight_score * (1.0 - cluster.score)
    )


def _components(
    rows: np.ndarray, u: np.ndarray, v: np.ndarray, config: WireSeparationConfig
) -> list[np.ndarray]:
    if not len(rows):
        return []
    values = np.column_stack((u[rows] / config.transverse_eps_m, v[rows] / config.vertical_eps_m))
    tree = cKDTree(values)
    parent = np.arange(len(rows))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in sorted(tree.query_pairs(1.0)):
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return [rows[np.asarray(items, dtype=np.int64)] for _, items in sorted(groups.items())]


def _split_large(
    rows: np.ndarray, u: np.ndarray, v: np.ndarray, config: WireSeparationConfig
) -> list[np.ndarray]:
    if len(rows) <= 1:
        return [rows]
    ordered = rows[np.lexsort((rows, v[rows], u[rows]))]
    groups: list[list[int]] = []
    for row in ordered:
        for group in groups:
            center = np.mean(np.column_stack((u[np.asarray(group)], v[np.asarray(group)])), axis=0)
            if (
                np.linalg.norm(np.array([u[row], v[row]]) - center)
                <= config.max_cluster_diameter_m / 2.0
            ):
                group.append(int(row))
                break
        else:
            groups.append([int(row)])
    return [np.asarray(group, dtype=np.int64) for group in groups]


def _mean_direction(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    directions = _unit_rows(values)
    directions[np.dot(directions, axis) < 0] *= -1.0
    mean = directions.mean(axis=0)
    return _unit(mean if np.linalg.norm(mean) > 1e-8 else axis)


def _track_direction(track: _Track) -> np.ndarray:
    return _unit(np.mean([cluster.direction for cluster in track.clusters], axis=0))


def _offsets(groups: list[np.ndarray]) -> np.ndarray:
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    if groups:
        offsets[1:] = np.cumsum([len(group) for group in groups], dtype=np.int64)
    return offsets


def _ordered_indices(groups: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(groups).astype(np.int64) if groups else np.empty(0, dtype=np.int64)


def _arrays(
    assignments: WireAssignmentSet,
    instances: WireInstanceSet,
    summaries: list[WireTrackSummary],
    state: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "source_id": np.asarray(assignments.source_id),
        "original_index": assignments.original_index,
        "sample_index": assignments.sample_index,
        "xyz_local": assignments.xyz_local,
        "track_id_by_point": assignments.track_id_by_point,
        "wire_id_by_point": assignments.wire_id_by_point,
        "assignment_score": assignments.assignment_score,
        "assignment_reason_flags": assignments.assignment_reason_flags,
        "assignment_quality_flags": assignments.assignment_quality_flags,
        "primary_slice_id": state["primary"],
        "s_coordinate_m": state["s"].astype(np.float32),
        "ordered_track_point_indices": instances.ordered_track_point_indices,
        "track_offsets": instances.track_offsets,
        "ordered_wire_point_indices": instances.ordered_wire_point_indices,
        "wire_offsets": instances.wire_offsets,
        "center_xyz": np.asarray(state["center"], dtype=np.float64),
        "s_axis": np.asarray(state["axis"], dtype=np.float64),
        "u_axis": np.asarray(state["u_axis"], dtype=np.float64),
        "v_axis": np.asarray(state["v_axis"], dtype=np.float64),
    }
    scalar = {
        "track_id": np.asarray([item.track_id for item in summaries], dtype=np.int32),
        "track_wire_id": np.asarray([item.wire_id for item in summaries], dtype=np.int32),
        "track_status": np.asarray([int(item.status) for item in summaries], dtype=np.uint8),
        "track_source": np.asarray([int(item.source) for item in summaries], dtype=np.uint8),
        "track_quality_flags": np.asarray(
            [item.quality_flags for item in summaries], dtype=np.uint16
        ),
        "track_point_count": np.asarray([item.point_count for item in summaries], dtype=np.int32),
        "track_span_m": np.asarray(
            [item.longitudinal_span_m for item in summaries], dtype=np.float32
        ),
        "track_coverage_ratio": np.asarray(
            [item.coverage_ratio for item in summaries], dtype=np.float32
        ),
        "track_mean_candidate_score": np.asarray(
            [item.mean_candidate_score for item in summaries], dtype=np.float32
        ),
        "track_actual_main_direction": np.asarray(
            [item.actual_main_direction for item in summaries], dtype=np.float32
        ),
        "track_axis_angle_deg": np.asarray(
            [item.axis_angle_deg for item in summaries], dtype=np.float32
        ),
        "track_confirmation_channel": np.asarray(
            [int(item.confirmation_channel) for item in summaries], dtype=np.uint8
        ),
    }
    values.update(scalar)
    return values


def _load_a3(
    npz_path: Path, json_path: Path, config: WireSeparationConfig
) -> tuple[dict[str, np.ndarray], Mapping[str, Any], dict[str, str]]:
    if not npz_path.is_file() or not json_path.is_file():
        raise _error("WIRE_INPUT_MISSING", "A3 NPZ and JSON inputs are required")
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as stream:
            data = {name: stream[name] for name in stream.files}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise _error("WIRE_INPUT_LOAD", "unable to load A3 input", cause=str(error)) from error
    hashes = {"npz": _sha(npz_path), "json": _sha(json_path)}
    if metadata.get("output_npz_sha256") != hashes["npz"]:
        raise _error("WIRE_INPUT_SHA", "A3 NPZ SHA-256 does not match JSON")
    count = config.expected_input_point_count
    required = {
        "xyz_local": ((count, 3), np.float32),
        "original_index": ((count,), np.int64),
        "sample_index": ((count,), np.int64),
        "candidate_mask": ((count,), np.bool_),
        "candidate_score": ((count,), np.float32),
        "candidate_level": ((count,), np.uint8),
        "candidate_source": ((count,), np.uint8),
        "evidence_flags": ((count,), np.uint16),
        "quality_flags": ((count,), np.uint16),
        "principal_direction": ((count, 3), np.float32),
        "origin_xyz": ((3,), np.float64),
    }
    for name, (shape, dtype) in required.items():
        if name not in data or data[name].shape != shape or data[name].dtype != dtype:
            raise _error("WIRE_INPUT_SHAPE", "A3 array shape or dtype is invalid", array=name)
    if (
        str(data["source_id"].item()) != metadata.get("source_id")
        or str(data["manifest_source_id"].item()) != metadata.get("manifest_source_id")
        or str(data["source_id"].item()) != config.source_id
    ):
        raise _error("WIRE_INPUT_SOURCE", "A3 source metadata is inconsistent")
    if len(np.unique(data["original_index"])) != len(data["original_index"]) or len(
        np.unique(data["sample_index"])
    ) != len(data["sample_index"]):
        raise _error("WIRE_INPUT_INDEX", "A3 indexes must be unique")
    if (
        not np.isfinite(data["xyz_local"]).all()
        or not np.isfinite(data["principal_direction"]).all()
        or not np.isfinite(data["candidate_score"]).all()
        or not np.all((data["candidate_score"] >= 0) & (data["candidate_score"] <= 1))
    ):
        raise _error("WIRE_INPUT_NONFINITE", "A3 coordinates, directions, or scores are invalid")
    if count == 146_126 and (
        int(data["candidate_mask"].sum()) != 17_784
        or [int(np.count_nonzero(data["candidate_level"] == level)) for level in (3, 2, 1)]
        != [8_226, 3_126, 6_432]
    ):
        raise _error("WIRE_INPUT_COUNTS", "A3 candidate count or level distribution is unexpected")
    return data, metadata, hashes


def _validate_output_dir(output: Path, allowed: Path) -> None:
    try:
        output.relative_to(allowed.resolve())
    except ValueError as error:
        raise _error(
            "WIRE_OUTPUT_FORBIDDEN",
            "output must be under the approved directory",
            output=str(output),
        ) from error


def _validate_outputs(paths: tuple[Path, Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise _error(
            "WIRE_OUTPUT_EXISTS", "output exists and overwrite is disabled", paths=existing
        )


def _error(code: str, message: str, **details: Any) -> WireSeparationError:
    cause = details.pop("cause", None)
    return WireSeparationError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="algorithms.wires.separate",
            recoverable=True,
            details=details,
            cause=cause,
        )
    )


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise _error("WIRE_DIRECTION", "cannot normalize zero or invalid direction")
    return np.asarray(value, dtype=np.float64) / norm


def _signed_unit(value: np.ndarray) -> np.ndarray:
    value = _unit(value)
    index = int(np.argmax(np.abs(value)))
    return value if value[index] >= 0 else -value


def _unit_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1)
    valid = norms > 1e-12
    result[valid] /= norms[valid, None]
    result[~valid] = 0.0
    return result


def _max_consecutive(values: list[int]) -> int:
    if not values:
        return 0
    best = current = 1
    for previous, value in zip(values, values[1:], strict=False):
        current = current + 1 if value == previous + 1 else 1
        best = max(best, current)
    return best


def _flag_counts(values: np.ndarray, flag_type: type[Any]) -> dict[str, int]:
    return {
        member.name: int(np.count_nonzero(values & int(member)))
        for member in flag_type
        if int(member)
    }


def _track_flag_counts(summaries: list[WireTrackSummary]) -> dict[str, int]:
    values = np.asarray([summary.quality_flags for summary in summaries], dtype=np.uint16)
    return _flag_counts(values, WireTrackQualityFlag)


def _meta(value: np.ndarray, unit: str) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "unit": unit}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _check_cancelled(token: CancellationToken | None, context: str) -> None:
    if token is not None and token.is_cancelled():
        raise _error("WIRE_CANCELLED", f"A4 wire separation cancelled {context}", context=context)


def _current_process() -> Any | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process()


def _rss(process: Any | None) -> int | None:
    return None if process is None else int(process.memory_info().rss)


def _report(
    reporter: ProgressReporter | None,
    source: str,
    stage: str,
    done: int,
    total: int,
    status: str = "running",
) -> None:
    if reporter is not None:
        reporter.update(
            PipelineProgress(
                task_id=f"{source}:wire-separation",
                stage=stage,
                status=status,
                processed_points=done,
                total_points=total,
                percent=100.0 if total == 0 else 100.0 * done / total,
                message=f"{stage}: {done} of {total}",
            )
        )


def _even(index: np.ndarray, limit: int) -> np.ndarray:
    return (
        index
        if len(index) <= limit
        else index[np.linspace(0, len(index) - 1, limit, dtype=np.int64)]
    )
