"""V2 scale-aware vegetation cluster grouping from V1/A2A/A2B artifacts only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from core.contracts import (
    CancellationToken,
    LocalScaleState,
    PipelineProgress,
    ProgressReporter,
    VegetationCandidateLevel,
    VegetationClusterStatus,
    VegetationHagState,
    VegetationInstanceConfig,
    VegetationInstanceManualFlag,
    VegetationInstancePointSet,
    VegetationInstanceQualityFlag,
    VegetationInstanceReasonFlag,
    VegetationInstanceResult,
    VegetationInstanceStatistics,
    VegetationManualLog,
    VegetationPlausibilityLabel,
)
from core.contracts.errors import UnifiedError
from scipy.spatial import cKDTree

from algorithms.vegetation.candidates import load_vegetation_candidates

SCHEMA_VERSION = "vegetation-instances-v2"
ALGORITHM_VERSION = "v2-generalized-morphology-safe-vegetation-instances-v2"
_BASE = "{source_id}_vegetation_instances"


class VegetationInstanceError(RuntimeError):
    def __init__(self, error: UnifiedError):
        super().__init__(error.message)
        self.error = error


def run_vegetation_instance_grouping(
    v1_npz_path: Path | str,
    v1_json_path: Path | str,
    a2a_npz_path: Path | str,
    a2a_json_path: Path | str,
    a2b_npz_path: Path | str,
    a2b_json_path: Path | str,
    output_dir: Path | str,
    *,
    config: VegetationInstanceConfig | None = None,
    manual_log: VegetationManualLog | None = None,
    overwrite: bool = False,
    reporter: ProgressReporter | None = None,
    cancellation_token: CancellationToken | None = None,
    project_root: Path | str | None = None,
    generate_preview: bool = True,
) -> VegetationInstanceResult:
    """Group verified V1 candidates; no LAS, A3, or raw record access occurs."""
    cfg = config or VegetationInstanceConfig()
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    outdir = Path(output_dir).resolve()
    _validate_output_dir(outdir, root / "outputs" / "vegetation_instances")
    npz_out = outdir / f"{_BASE.format(source_id=cfg.source_id)}.npz"
    json_out = outdir / f"{_BASE.format(source_id=cfg.source_id)}.json"
    _validate_outputs((npz_out, json_out), overwrite)
    _check_cancelled(cancellation_token, "before V2 input validation")
    started = time.perf_counter()
    process = _current_process() if cfg.record_memory else None
    rss_start = _rss(process)
    phase_started = started
    v1_result, v1 = load_vegetation_candidates(v1_npz_path, v1_json_path)
    a2a, a2a_meta = _load_pair(a2a_npz_path, a2a_json_path)
    a2b, a2b_meta = _load_pair(a2b_npz_path, a2b_json_path)
    paths = {
        "v1_npz": Path(v1_npz_path).resolve(),
        "v1_json": Path(v1_json_path).resolve(),
        "a2a_npz": Path(a2a_npz_path).resolve(),
        "a2a_json": Path(a2a_json_path).resolve(),
        "a2b_npz": Path(a2b_npz_path).resolve(),
        "a2b_json": Path(a2b_json_path).resolve(),
    }
    hashes = {name: _sha(path) for name, path in paths.items()}
    _validate_inputs(v1_result, v1, a2a, a2a_meta, a2b, a2b_meta, hashes, cfg)
    validation_seconds = time.perf_counter() - phase_started
    _report(reporter, cfg.source_id, "validated_inputs", 0, len(v1["original_index"]))
    _check_cancelled(cancellation_token, "after V2 input validation")
    log = manual_log or VegetationManualLog()
    if log.actions:
        _validate_manual_log(log, hashes, v1_result.source_id, v1_result.manifest_source_id)
    grouping_started = time.perf_counter()
    arrays, cluster_rows, counts = _group_arrays(
        v1, a2a, a2b, cfg, log, cancellation_token, reporter
    )
    grouping_seconds = time.perf_counter() - grouping_started
    _check_cancelled(cancellation_token, "before V2 publication")
    stats = VegetationInstanceStatistics(
        input_point_count=len(v1["original_index"]),
        candidate_count=int(v1["vegetation_mask"].sum()),
        core_point_count=counts["core_point_count"],
        cluster_count=len(cluster_rows),
        confirmed_count=counts["confirmed_count"],
        review_count=counts["review_count"],
        provisional_count=counts["provisional_count"],
        rejected_count=counts["rejected_count"],
        assigned_candidate_count=counts["assigned_candidate_count"],
        unassigned_candidate_count=counts["unassigned_candidate_count"],
        elapsed_seconds=time.perf_counter() - started,
        rss_start_bytes=rss_start,
        rss_peak_bytes=_rss(process),
        rss_end_bytes=_rss(process),
        phase_timings_seconds={
            "input_validation": validation_seconds,
            **counts["phase_timings_seconds"],
            "grouping_total": grouping_seconds,
        },
        safe_split_counts=counts["safe_split_counts"],
    )
    outdir.mkdir(parents=True, exist_ok=True)
    result = _publish(
        arrays, cluster_rows, npz_out, json_out, cfg, log, v1_result, hashes, stats, root
    )
    if generate_preview:
        try:
            generate_vegetation_instances_preview(
                npz_out,
                json_out,
                root
                / "outputs"
                / "previews"
                / f"{_BASE.format(source_id=cfg.source_id)}_preview.html",
                overwrite=overwrite,
                project_root=root,
            )
        except Exception as exc:  # core result is valid even if optional plotting is unavailable
            result = replace(result, warnings=(f"preview generation failed: {exc}",))
    _report(
        reporter,
        cfg.source_id,
        "completed",
        len(v1["original_index"]),
        len(v1["original_index"]),
        "completed",
    )
    return result


def load_vegetation_instances(
    npz_path: Path | str | None = None, json_path: Path | str | None = None
) -> tuple[VegetationInstanceResult, dict[str, np.ndarray]]:
    """Load a V2 sibling pair without source project paths or upstream artifacts."""
    if json_path is None:
        if npz_path is None or Path(npz_path).suffix.lower() != ".json":
            raise _error("V2_LOAD_PATH", "supply a JSON path or sibling NPZ/JSON paths")
        json_file = Path(npz_path)
        raw = _load_json(json_file)
        npz_file = json_file.with_name(str(raw.get("output_npz_filename", "")))
    else:
        npz_file, json_file = Path(npz_path), Path(json_path)
    if not npz_file.is_file() or not json_file.is_file():
        raise _error("V2_LOAD_MISSING", "V2 NPZ and JSON are both required")
    result = VegetationInstanceResult.from_dict(_load_json(json_file))
    if npz_file.name != result.output_npz_filename or _sha(npz_file) != result.output_npz_sha256:
        raise _error("V2_LOAD_HASH", "V2 NPZ does not match JSON manifest")
    arrays = _load_npz(npz_file)
    _validate_output_arrays(arrays)
    if result.schema_version != SCHEMA_VERSION:
        raise _error("V2_LOAD_SCHEMA", "unsupported V2 schema")
    return result, arrays


def generate_vegetation_instances_preview(
    npz_path: Path | str,
    json_path: Path | str,
    output_html_path: Path | str,
    *,
    overwrite: bool = False,
    project_root: Path | str | None = None,
    point_limit: int = 30_000,
) -> Path:
    """Write a standalone V2 Plotly preview from V2 artifact data only."""
    root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
    output = Path(output_html_path).resolve()
    _validate_output_dir(output.parent, root / "outputs" / "previews")
    if output.exists() and not overwrite:
        raise _error("V2_PREVIEW_EXISTS", "preview exists and overwrite is disabled")
    result, data = load_vegetation_instances(npz_path, json_path)
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise _error("V2_PREVIEW_DEPENDENCY", "Plotly is unavailable", cause=str(exc)) from exc
    xyz, cid, status = (
        data["xyz_local"],
        data["cluster_id_by_point"],
        data["cluster_status_by_point"],
    )
    fig = go.Figure()
    background = _even(np.flatnonzero(cid < 0), point_limit)
    fig.add_trace(
        go.Scatter3d(
            x=xyz[background, 0],
            y=xyz[background, 1],
            z=xyz[background, 2],
            mode="markers",
            name="unassigned / background",
            marker={"size": 1, "color": "#94a3b8", "opacity": 0.12},
            visible="legendonly",
            hoverinfo="skip",
        )
    )
    palette = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c"]
    styles = {
        VegetationClusterStatus.CONFIRMED: ("confirmed vegetation", True, 2.6),
        VegetationClusterStatus.REVIEW_REQUIRED: ("review required", True, 2.2),
        VegetationClusterStatus.PROVISIONAL: ("provisional", "legendonly", 1.5),
        VegetationClusterStatus.REJECTED_AS_INSTANCE: ("rejected as instance", "legendonly", 1.5),
    }
    for state, (name, visible, size) in styles.items():
        ix = np.flatnonzero(status == int(state))
        ix = _even(ix, point_limit)
        text = [
            f"cluster={int(cid[i])}<br>status={state.name}<br>vegetation_id={int(data['vegetation_id_by_point'][i])}<br>score={float(data['vegetation_rule_score'][i]):.3f}<br>assignment={float(data['assignment_confidence'][i]):.3f}"
            for i in ix
        ]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[ix, 0],
                y=xyz[ix, 1],
                z=xyz[ix, 2],
                mode="markers",
                name=name,
                marker={"size": size, "color": palette[int(state) % len(palette)], "opacity": 0.85},
                visible=visible,
                text=text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    wire = np.flatnonzero(data["wire_id_by_point"] >= 0)
    fig.add_trace(
        go.Scatter3d(
            x=xyz[wire, 0],
            y=xyz[wire, 1],
            z=xyz[wire, 2],
            mode="markers",
            name="confirmed wires",
            marker={"size": 2.4, "color": "#dc2626"},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"V2 vegetation instances — {result.source_id}",
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
        fig.write_html(temporary, include_plotlyjs=True, full_html=True)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _group_arrays(
    v1: dict[str, np.ndarray],
    a2a: dict[str, np.ndarray],
    a2b: dict[str, np.ndarray],
    cfg: VegetationInstanceConfig,
    log: VegetationManualLog,
    token: CancellationToken | None,
    reporter: ProgressReporter | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    phase_started = time.perf_counter()
    phase_timings: dict[str, float] = {}
    m = len(v1["original_index"])
    xyz = v1["xyz_local"].astype(np.float64)
    level, wire = v1["vegetation_level"], v1["wire_id_by_point"]
    candidate = v1["vegetation_mask"].copy() & (wire < 0)
    candidate_rows = np.flatnonzero(candidate)
    raw, scale, scale_state = _local_scales(xyz[candidate_rows], cfg, token)
    phase_timings["local_scale"] = time.perf_counter() - phase_started
    raw_m = np.full(m, np.nan, np.float32)
    scale_m = np.full(m, np.nan, np.float32)
    state_m = np.full(m, int(LocalScaleState.LOCAL_SCALE_UNAVAILABLE), np.uint8)
    raw_m[candidate_rows], scale_m[candidate_rows], state_m[candidate_rows] = (
        raw,
        scale,
        scale_state,
    )
    reason = np.zeros(m, np.uint16)
    quality = np.zeros(m, np.uint16)
    manual = np.zeros(m, np.uint16)
    reason[wire >= 0] |= int(VegetationInstanceReasonFlag.CONFIRMED_WIRE_BLOCKED)
    quality[
        (candidate_rows[state_m[candidate_rows] == int(LocalScaleState.LOCAL_SCALE_UNAVAILABLE)])
    ] |= int(VegetationInstanceQualityFlag.LOCAL_SCALE_UNAVAILABLE)
    core_rows = np.flatnonzero(
        candidate
        & np.isin(level, [int(VegetationCandidateLevel.MEDIUM), int(VegetationCandidateLevel.HIGH)])
        & (state_m != int(LocalScaleState.LOCAL_SCALE_UNAVAILABLE))
    )
    _report(reporter, cfg.source_id, "local_scale", len(candidate_rows), len(candidate_rows))
    edges, neighbour_sets = _core_edges(core_rows, xyz, scale_m, v1, a2b, cfg, token)
    _report(reporter, cfg.source_id, "core_edges", len(edges), max(len(core_rows), 1))
    groups = _components(len(core_rows), edges)
    component_by_local = np.full(len(core_rows), -1, np.int32)
    for component_id, local_members in enumerate(groups):
        component_by_local[local_members] = component_id
    component_edges: list[list[tuple[int, int, float]]] = [[] for _ in groups]
    for edge in edges:
        component_id = int(component_by_local[edge[0]])
        if component_id >= 0:
            component_edges[component_id].append(edge)
    # Bridge splitting is conservative and only changes components when both children meet support.
    split_groups: list[tuple[np.ndarray, int, bool]] = []
    split_counts: dict[str, int] = defaultdict(int)
    for parent, local_members in enumerate(groups):
        rows = core_rows[local_members]
        children, unresolved = _safe_split_rows(
            rows,
            component_edges[parent],
            core_rows,
            xyz,
            scale_m,
            v1,
            a2a,
            a2b,
            neighbour_sets,
            cfg,
            split_counts=split_counts,
        )
        for child in children:
            split_groups.append((child, parent, unresolved))
    phase_timings["core_graph_and_safe_splits"] = (
        time.perf_counter() - phase_started - phase_timings["local_scale"]
    )
    accepted: list[dict[str, Any]] = []
    core_position = {int(row): pos for pos, row in enumerate(core_rows)}
    for members, parent, unresolved in split_groups:
        if _meets_support(members, xyz, scale_m, raw_m, level, cfg):
            member_set = set(int(v) for v in members)
            relevant = [
                edge
                for edge in component_edges[parent]
                if int(core_rows[edge[0]]) in member_set and int(core_rows[edge[1]]) in member_set
            ]
            degree = defaultdict(int)
            shared_values = []
            for left, right, _norm in relevant:
                degree[left] += 1
                degree[right] += 1
                shared_values.append(len(neighbour_sets[left] & neighbour_sets[right]))
            local_positions = [core_position[int(row)] for row in members]
            degree_values = [degree[pos] for pos in local_positions]
            states = state_m[members]
            valid = float(np.mean(states == int(LocalScaleState.LOCAL_SCALE_VALID)))
            low = float(np.mean(states == int(LocalScaleState.LOCAL_SCALE_CLAMPED_LOW)))
            high = float(np.mean(states == int(LocalScaleState.LOCAL_SCALE_CLAMPED_HIGH)))
            q_scale = valid + 0.85 * low + 0.70 * high
            median_degree = float(np.median(degree_values)) if degree_values else 0.0
            median_shared = float(np.median(shared_values)) if shared_values else 0.0
            grouping_score = (
                0.30 * _clip((len(members) - 12) / 36)
                + 0.20
                * _clip(
                    (
                        len(
                            np.unique(
                                np.floor(xyz[members] / _voxel_size(members, raw_m, cfg)).astype(
                                    np.int64
                                ),
                                axis=0,
                            )
                        )
                        - 3
                    )
                    / 7
                )
                + 0.20 * _clip((median_degree - 2) / 4)
                + 0.15 * _clip((median_shared - 2) / 3)
                + 0.15 * q_scale
            )
            if not (
                median_degree >= cfg.grouping_min_median_degree
                and median_shared >= cfg.grouping_min_median_shared_neighbors
                and q_scale >= cfg.grouping_min_scale_quality
                and grouping_score >= 0.55
            ):
                reason[members] |= int(VegetationInstanceReasonFlag.UNSUPPORTED_CORE_FRAGMENT)
                continue
            accepted.append(
                {
                    "core": np.sort(members),
                    "parent": parent,
                    "unresolved": unresolved,
                    "median_core_degree": median_degree,
                    "median_shared_neighbors": median_shared,
                    "q_scale": q_scale,
                    "nonunavailable_fraction": float(
                        np.mean(states != int(LocalScaleState.LOCAL_SCALE_UNAVAILABLE))
                    ),
                }
            )
        else:
            reason[members] |= int(VegetationInstanceReasonFlag.UNSUPPORTED_CORE_FRAGMENT)
    accepted.sort(key=lambda item: int(v1["original_index"][item["core"]].min()))
    cluster_by = np.full(m, -1, np.int32)
    for cid, item in enumerate(accepted):
        item["id"] = cid
        item["members"] = item["core"].copy()
        cluster_by[item["core"]] = cid
    _low_expand(
        candidate_rows,
        accepted,
        cluster_by,
        xyz,
        scale_m,
        state_m,
        level,
        v1,
        a2b,
        cfg,
        reason,
        token,
    )
    phase_timings["low_reassignment"] = (
        time.perf_counter()
        - phase_started
        - phase_timings["local_scale"]
        - phase_timings["core_graph_and_safe_splits"]
    )
    # Descriptor/status phase; manual actions apply after automatic membership exists.
    _apply_manual(
        log, accepted, cluster_by, candidate, wire, xyz, v1["original_index"], reason, manual
    )
    cluster_by[:] = -1
    for cid, item in enumerate(
        sorted(accepted, key=lambda q: int(v1["original_index"][q["members"]].min()))
    ):
        item["id"] = cid
        item["members"] = np.unique(item["members"])
        cluster_by[item["members"]] = cid
    descriptors = [
        _describe_cluster(item, xyz, v1, a2a, a2b, raw_m, scale_m, cfg) for item in accepted
    ]
    statuses = _statuses(descriptors, accepted, manual, cfg)
    phase_timings["descriptors_and_status"] = (
        time.perf_counter()
        - phase_started
        - phase_timings["local_scale"]
        - phase_timings["core_graph_and_safe_splits"]
        - phase_timings["low_reassignment"]
    )
    status_by = np.full(m, int(VegetationClusterStatus.UNASSIGNED), np.uint8)
    vegetation_by = np.full(m, -1, np.int32)
    assign = np.zeros(m, np.float32)
    next_vid = 0
    for item, desc, state in zip(accepted, descriptors, statuses, strict=True):
        rows = item["members"]
        status_by[rows] = int(state)
        assign[rows] = np.float32(desc["assessment_confidence"])
        desc["status"] = state.name
        if item.get("manual_status"):
            reason[rows] |= int(VegetationInstanceReasonFlag.MANUAL_STATUS_OVERRIDE)
        if state == VegetationClusterStatus.CONFIRMED:
            vegetation_by[rows] = next_vid
            desc["vegetation_id"] = next_vid
            next_vid += 1
        else:
            desc["vegetation_id"] = -1
        if item["unresolved"]:
            reason[rows] |= int(VegetationInstanceReasonFlag.UNRESOLVED_BRIDGE)
            quality[rows] |= int(VegetationInstanceQualityFlag.BRIDGE_WARNING)
        if desc["fragmentation"] >= cfg.fragmentation_review_threshold:
            quality[rows] |= int(VegetationInstanceQualityFlag.FRAGMENTED)
        for key, flag in (
            ("planar_ambiguity", VegetationInstanceReasonFlag.PLANAR_AMBIGUITY),
            ("isolated_slender_ambiguity", VegetationInstanceReasonFlag.ISOLATED_SLENDER_AMBIGUITY),
            ("compact_layered_ambiguity", VegetationInstanceReasonFlag.COMPACT_LAYERED_AMBIGUITY),
            ("mixed_morphology_conflict", VegetationInstanceReasonFlag.MIXED_MORPHOLOGY_CONFLICT),
            ("unresolved_topology", VegetationInstanceReasonFlag.UNRESOLVED_TOPOLOGY),
        ):
            if desc.get(key):
                reason[rows] |= int(flag)
        if desc["independent_positive_count"] < cfg.minimum_independent_positive_channels:
            reason[rows] |= int(VegetationInstanceReasonFlag.INSUFFICIENT_POSITIVE_EVIDENCE)
        if desc["grouping_support_score"] < cfg.grouping_support_threshold:
            quality[rows] |= int(VegetationInstanceQualityFlag.GROUPING_SUPPORT_DEGRADED)
        if np.count_nonzero(np.asarray(desc["slice_valid"], bool)) < 3:
            quality[rows] |= int(VegetationInstanceQualityFlag.MORPHOLOGY_DEGRADED)
    distance = vegetation_by >= 0
    points = VegetationInstancePointSet(
        v1["original_index"],
        v1["sample_index"],
        v1["xyz_local"],
        wire,
        v1["vegetation_rule_score"],
        v1["vegetation_confidence"],
        v1["automatic_vegetation_level"],
        level,
        v1["hag_state"],
        raw_m,
        scale_m,
        state_m,
        cluster_by,
        vegetation_by,
        status_by,
        distance,
        assign,
        reason,
        quality,
        manual,
    )
    arrays: dict[str, np.ndarray] = {
        name: getattr(points, name) for name in points.__dataclass_fields__
    }
    arrays["origin_xyz"] = v1["origin_xyz"]
    arrays.update(_flat_cluster_arrays(accepted, descriptors))
    counts = {
        "core_point_count": int(len(core_rows)),
        "confirmed_count": int(sum(s == VegetationClusterStatus.CONFIRMED for s in statuses)),
        "review_count": int(sum(s == VegetationClusterStatus.REVIEW_REQUIRED for s in statuses)),
        "provisional_count": int(sum(s == VegetationClusterStatus.PROVISIONAL for s in statuses)),
        "rejected_count": int(
            sum(s == VegetationClusterStatus.REJECTED_AS_INSTANCE for s in statuses)
        ),
        "assigned_candidate_count": int(np.count_nonzero(cluster_by >= 0)),
        "unassigned_candidate_count": int(np.count_nonzero(candidate & (cluster_by < 0))),
        "safe_split_counts": {
            key: int(split_counts.get(key, 0)) for key in ("bridge", "gap", "mixed", "unresolved")
        },
        "phase_timings_seconds": phase_timings,
    }
    return arrays, descriptors, counts


def _local_scales(
    xyz: np.ndarray, cfg: VegetationInstanceConfig, token: CancellationToken | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(xyz)
    raw = np.full(n, np.nan, np.float32)
    scale = np.full(n, np.nan, np.float32)
    state = np.full(n, int(LocalScaleState.LOCAL_SCALE_UNAVAILABLE), np.uint8)
    if not n:
        return raw, scale, state
    _check_cancelled(token, "before local-scale tree")
    dist, _ = cKDTree(xyz).query(xyz, k=min(cfg.local_scale_k + 1, n), workers=1)
    if dist.ndim == 1:
        dist = dist[:, None]
    for i, row in enumerate(dist):
        valid = row[(row > 1e-9) & np.isfinite(row) & (row <= cfg.local_scale_support_radius_m)][
            : cfg.local_scale_k
        ]
        if len(valid) < cfg.local_scale_min_valid_neighbors:
            continue
        value = float(np.median(valid))
        raw[i] = value
        if value < cfg.local_scale_lower_m:
            state[i] = int(LocalScaleState.LOCAL_SCALE_CLAMPED_LOW)
        elif value > cfg.local_scale_upper_m:
            state[i] = int(LocalScaleState.LOCAL_SCALE_CLAMPED_HIGH)
        else:
            state[i] = int(LocalScaleState.LOCAL_SCALE_VALID)
        scale[i] = np.float32(np.clip(value, cfg.local_scale_lower_m, cfg.local_scale_upper_m))
    return raw, scale, state


def _core_edges(
    rows: np.ndarray,
    xyz: np.ndarray,
    scale: np.ndarray,
    v1: dict[str, np.ndarray],
    a2b: dict[str, np.ndarray],
    cfg: VegetationInstanceConfig,
    token: CancellationToken | None,
) -> tuple[list[tuple[int, int, float]], list[set[int]]]:
    if not len(rows):
        return [], []
    _check_cancelled(token, "before core tree")
    local = xyz[rows]
    count = len(rows)
    tree = cKDTree(local)
    # Query a conservative superset then impose our own deterministic ranking.  This
    # avoids cKDTree's unspecified ordering of equidistant regular-grid neighbours.
    # `query_ball_point` around every core point dominated the real-scene profile.
    # A bounded nearest-neighbour superset is sufficient because every accepted
    # edge is subsequently checked against its pair-specific radius; its size is
    # deliberately larger than the expanded cap so stable tie ordering remains
    # under our control rather than depending on cKDTree return order.
    query_cap = min(count, max(cfg.expanded_neighbor_cap + 8, 32))
    query_distances, query_indices = tree.query(local, k=query_cap, workers=1)
    if query_distances.ndim == 1:
        query_distances, query_indices = query_distances[:, None], query_indices[:, None]
    candidate_lists: list[list[tuple[float, int]]] = []
    hag, state = a2b["height_agl_m"], v1["hag_state"]
    for i in range(count):
        if i % 4096 == 0:
            _check_cancelled(token, "during core edge construction")
        candidate = np.asarray(query_indices[i], dtype=np.int32)
        candidate = candidate[(candidate != i) & np.isfinite(query_distances[i])]
        if not len(candidate):
            candidate_lists.append([])
            continue
        a = int(rows[i])
        b = rows[candidate]
        rho = np.clip(
            cfg.core_radius_multiplier * np.sqrt(float(scale[a]) * scale[b]),
            cfg.core_radius_lower_m,
            cfg.core_radius_upper_m,
        )
        delta = xyz[b] - xyz[a]
        horizontal = np.sqrt(np.sum(delta[:, :2] ** 2, axis=1))
        vertical = np.abs(delta[:, 2])
        anisotropic = np.sqrt(horizontal**2 + (vertical / cfg.vertical_scale_multiplier) ** 2)
        keep = (
            (horizontal <= rho)
            & (vertical <= cfg.vertical_scale_multiplier * rho)
            & (anisotropic <= rho)
        )
        if state[a] == int(VegetationHagState.RELIABLE):
            reliable = state[b] == int(VegetationHagState.RELIABLE)
            keep &= ~reliable | (np.abs(hag[a] - hag[b]) <= 0.15 + 1.5 * rho)
        values = list(
            zip((anisotropic[keep] / rho[keep]).tolist(), candidate[keep].tolist(), strict=True)
        )
        values.sort(
            key=lambda value: (
                round(value[0] / cfg.neighbour_tie_epsilon_m)
                if cfg.neighbour_tie_epsilon_m
                else value[0],
                int(v1["original_index"][rows[value[1]]]),
            )
        )
        candidate_lists.append(values)

    primary = [set(j for _, j in values[: cfg.primary_neighbor_cap]) for values in candidate_lists]
    expanded = [
        set(j for _, j in values[: cfg.expanded_neighbor_cap]) for values in candidate_lists
    ]
    edges: list[tuple[int, int, float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, values in enumerate(candidate_lists):
        for norm, j in values[: cfg.primary_neighbor_cap]:
            if j > i and i in primary[j]:
                edges.append((i, j, norm, 0, len(primary[i] & primary[j])))
                seen.add((i, j))

    secondary: list[tuple[float, int, int, int]] = []
    voxel = _voxel_size(rows, scale, cfg)
    keys = np.floor(local / voxel).astype(np.int64)
    for i, values in enumerate(candidate_lists):
        for norm, j in values[: cfg.expanded_neighbor_cap]:
            a, b = sorted((i, j))
            if a == b or (a, b) in seen or i != a:
                continue
            shared = len(primary[a] & primary[b])
            ratio = float(scale[rows[a]] / max(float(scale[rows[b]]), 1e-9))
            if (
                a not in expanded[b]
                or b not in expanded[a]
                or len(primary[a]) < 4
                or len(primary[b]) < 4
                or not (2 / 3 <= ratio <= 3 / 2)
                or norm > cfg.secondary_edge_normalized_distance_max
                or shared < cfg.secondary_edge_min_shared_neighbors
                or np.any(np.abs(keys[a] - keys[b]) > 1)
            ):
                continue
            secondary.append((norm, -shared, a, b))
    secondary.sort(
        key=lambda value: (
            value[0],
            value[1],
            min(
                int(v1["original_index"][rows[value[2]]]), int(v1["original_index"][rows[value[3]]])
            ),
            max(
                int(v1["original_index"][rows[value[2]]]), int(v1["original_index"][rows[value[3]]])
            ),
        )
    )
    secondary_degree = np.zeros(count, np.int16)
    for norm, negative_shared, i, j in secondary:
        if (
            secondary_degree[i] >= cfg.secondary_edge_max_per_point
            or secondary_degree[j] >= cfg.secondary_edge_max_per_point
        ):
            continue
        edges.append((i, j, norm, 1, -negative_shared))
        secondary_degree[i] += 1
        secondary_degree[j] += 1

    def edge_key(value: tuple[int, int, float, int, int]) -> tuple[int, float, float, int, int]:
        i, j, normalized, kind, _shared = value
        oi, oj = int(v1["original_index"][rows[i]]), int(v1["original_index"][rows[j]])
        confidence = min(
            float(v1["vegetation_confidence"][rows[i]]),
            float(v1["vegetation_confidence"][rows[j]]),
        )
        return kind, normalized, -confidence, min(oi, oj), max(oi, oj)

    edges.sort(key=edge_key)
    # Publicly internal callers only need (i, j, normalized); primary sets are
    # returned for bridge shared-neighbour evidence.
    return [(i, j, norm) for i, j, norm, _kind, _shared in edges], primary


def _components(count: int, edges: list[tuple[int, int, float]]) -> list[np.ndarray]:
    parent = np.arange(count, dtype=np.int32)
    size = np.ones(count, np.int32)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i, j, _ in edges:
        a, b = find(i), find(j)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(count):
        groups[find(i)].append(i)
    return [np.asarray(v, np.int64) for _, v in sorted(groups.items())]


def _voxel_size(rows: np.ndarray, raw: np.ndarray, cfg: VegetationInstanceConfig) -> float:
    values = raw[rows]
    values = values[np.isfinite(values)]
    return float(
        np.clip(
            2 * np.median(values) if len(values) else cfg.voxel_lower_m,
            cfg.voxel_lower_m,
            cfg.voxel_upper_m,
        )
    )


def _meets_support(
    rows: np.ndarray,
    xyz: np.ndarray,
    scale: np.ndarray,
    raw: np.ndarray,
    level: np.ndarray,
    cfg: VegetationInstanceConfig,
) -> bool:
    if len(rows) < cfg.min_core_points:
        return False
    if (
        int(np.count_nonzero(level[rows] == int(VegetationCandidateLevel.HIGH)))
        < cfg.min_high_points
        and int(np.count_nonzero(level[rows] == int(VegetationCandidateLevel.MEDIUM)))
        < cfg.min_medium_points
    ):
        return False
    voxel = _voxel_size(rows, raw, cfg)
    keys = np.floor(xyz[rows] / voxel).astype(np.int64)
    if len(np.unique(keys, axis=0)) < cfg.min_core_voxels:
        return False
    if np.max(np.ptp(xyz[rows], axis=0)) < cfg.min_core_extent_m and len(rows) < 24:
        return False
    # The upper local-scale bound is a measurement clamp, not a rejection radius.
    # A connected compact object with CLAMPED_HIGH samples remains supportable.
    values = raw[rows]
    return bool(
        np.count_nonzero(np.isfinite(values) & (values <= cfg.local_scale_support_radius_m))
        >= math.ceil(0.5 * len(rows))
    )


def _partition_core_rows(
    rows: np.ndarray,
    edges: list[tuple[int, int, float]],
    core_rows: np.ndarray,
    removed: set[tuple[int, int]],
) -> list[np.ndarray]:
    """Return deterministic components after removing point-level core edges."""
    index = {int(row): pos for pos, row in enumerate(rows)}
    parent = np.arange(len(rows), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    rowset = set(index)
    for left, right, _ in edges:
        a, b = int(core_rows[left]), int(core_rows[right])
        pair = (min(a, b), max(a, b))
        if a not in rowset or b not in rowset or pair in removed:
            continue
        x, y = find(index[a]), find(index[b])
        if x != y:
            parent[y] = x
    groups: dict[int, list[int]] = defaultdict(list)
    for pos, row in enumerate(rows):
        groups[find(pos)].append(int(row))
    return [
        np.asarray(value, np.int64)
        for _key, value in sorted(groups.items(), key=lambda item: min(item[1]))
    ]


def _bridge_candidates(
    rows: np.ndarray,
    edges: list[tuple[int, int, float]],
    core_rows: np.ndarray,
    xyz: np.ndarray,
    scale: np.ndarray,
    v1: dict[str, np.ndarray],
    a2a: dict[str, np.ndarray],
    a2b: dict[str, np.ndarray],
    neighbours: list[set[int]],
    cfg: VegetationInstanceConfig,
) -> list[dict[str, Any]]:
    """Score quotient-graph bridges without relying on object or scene knowledge."""
    if len(rows) < 2:
        return []
    pos = {int(row): i for i, row in enumerate(core_rows)}
    voxel = _voxel_size(rows, scale, cfg)
    keys = [tuple(v) for v in np.floor(xyz[rows] / voxel).astype(np.int64)]
    node_for = {key: n for n, key in enumerate(sorted(set(keys)))}
    point_node = np.asarray([node_for[key] for key in keys], np.int32)
    row_to_node = {int(row): int(node) for row, node in zip(rows, point_node, strict=True)}
    qedges: dict[tuple[int, int], list[tuple[int, int, float]]] = defaultdict(list)
    rowset = set(int(v) for v in rows)
    for i, j, norm in edges:
        a, b = int(core_rows[i]), int(core_rows[j])
        if a not in rowset or b not in rowset:
            continue
        na, nb = row_to_node[a], row_to_node[b]
        if na != nb:
            qedges[tuple(sorted((na, nb)))].append((a, b, norm))
    if not qedges:
        return []
    graph: dict[int, list[int]] = defaultdict(list)
    for a, b in qedges:
        graph[a].append(b)
        graph[b].append(a)
    # Deterministic iterative DFS bridges: large real components must not use recursion.
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int] = {}
    bridges: list[tuple[int, int]] = []
    clock = 0
    for node in sorted(graph):
        if node in disc:
            continue
        clock += 1
        disc[node] = low[node] = clock
        parent[node] = -1
        stack = [(node, iter(sorted(graph[node])))]
        while stack:
            current, iterator = stack[-1]
            try:
                child = next(iterator)
            except StopIteration:
                stack.pop()
                parent_node = parent[current]
                if parent_node >= 0:
                    low[parent_node] = min(low[parent_node], low[current])
                    if low[current] > disc[parent_node]:
                        bridges.append(tuple(sorted((parent_node, current))))
                continue
            if child == parent[current]:
                continue
            if child in disc:
                low[current] = min(low[current], disc[child])
                continue
            parent[child] = current
            clock += 1
            disc[child] = low[child] = clock
            stack.append((child, iter(sorted(graph[child]))))
    candidates: list[dict[str, Any]] = []
    for edge in sorted(set(bridges)):
        support = qedges[edge]
        shared = []
        for a, b, _ in support:
            ia, ib = pos[a], pos[b]
            shared.append(len(neighbours[ia] & neighbours[ib]))
        removed = {(min(a, b), max(a, b)) for a, b, _ in support}
        # A bridge in the quotient graph has exactly two quotient-side
        # partitions.  Reusing that fact avoids repeating an O(E) point graph
        # traversal for every candidate while still removing every represented
        # point edge when the selected candidate is accepted.
        seen = {edge[0]}
        stack = [edge[0]]
        while stack:
            current = stack.pop()
            for adjacent in graph[current]:
                if tuple(sorted((current, adjacent))) == edge or adjacent in seen:
                    continue
                seen.add(adjacent)
                stack.append(adjacent)
        left = rows[np.isin(point_node, list(seen))]
        right = rows[~np.isin(point_node, list(seen))]
        if not len(left) or not len(right):
            continue
        children = sorted((left, right), key=lambda child: int(v1["original_index"][child].min()))
        left, right = children
        cross_rows = np.asarray([a for a, _, _ in support] + [b for _, b, _ in support], np.int64)
        score_floor = min(
            float(np.median(v1["vegetation_rule_score"][left])),
            float(np.median(v1["vegetation_rule_score"][right])),
        )
        confidence_floor = min(
            float(np.median(v1["vegetation_confidence"][left])),
            float(np.median(v1["vegetation_confidence"][right])),
        )
        score_valley = _clip(
            (score_floor - float(np.median(v1["vegetation_rule_score"][cross_rows])) - 0.12) / 0.20
        )
        confidence_valley = _clip(
            (confidence_floor - float(np.median(v1["vegetation_confidence"][cross_rows])) - 0.08)
            / 0.15
        )
        valley = 0.50 * score_valley + 0.50 * confidence_valley
        reliable = v1["hag_state"] == int(VegetationHagState.RELIABLE)
        if len(left[reliable[left]]) >= 5 and len(right[reliable[right]]) >= 5:
            hag_jump = _clip(
                (
                    abs(
                        float(np.median(a2b["height_agl_m"][left[reliable[left]]]))
                        - float(np.median(a2b["height_agl_m"][right[reliable[right]]]))
                    )
                    - 0.35
                )
                / 0.35
            )
        else:
            hag_jump = 0.0
        geom = max(
            _clip(
                abs(
                    float(np.median(a2a["planarity"][left]))
                    - float(np.median(a2a["planarity"][right]))
                )
                / 0.25
            ),
            _clip(
                abs(
                    float(np.median(a2a["verticality"][left]))
                    - float(np.median(a2a["verticality"][right]))
                )
                / 0.25
            ),
            _clip(
                abs(
                    float(np.median(a2a["scattering"][left]))
                    - float(np.median(a2a["scattering"][right]))
                )
                / 0.15
            ),
        )
        max_norm = max(value[2] for value in support)
        edge_length = _clip((max_norm - cfg.bridge_normalized_length_threshold) / 0.15)
        weak_cross = 1.0 - _clip((len(support) - 1) / 3)
        shared_deficit = 1.0 - _clip(float(np.median(shared)) / 3)
        evidence = (
            0.20 * edge_length
            + 0.15 * weak_cross
            + 0.15 * shared_deficit
            + 0.20 * valley
            + 0.15 * hag_jump
            + 0.15 * geom
        )
        categories = (edge_length, weak_cross, shared_deficit, valley, hag_jump, geom)
        # A quotient bridge that only reflects ordinary voxel tessellation is not
        # a split candidate.  The fixed floor requires multiple, material pieces
        # of general evidence while preserving the deterministic priority below.
        if evidence < 0.45:
            continue
        originals = [int(v1["original_index"][row]) for pair in removed for row in pair]
        candidates.append(
            {
                "removed": removed,
                "children": children,
                "priority": (
                    -sum(value > 0 for value in categories),
                    -evidence,
                    -max_norm,
                    len(support),
                    min(originals),
                    max(originals),
                ),
                "evidence": evidence,
                "categories": int(sum(value > 0 for value in categories)),
            }
        )
    return sorted(candidates, key=lambda candidate: candidate["priority"])


def _vertical_gap_candidate(
    rows: np.ndarray,
    edges: list[tuple[int, int, float]],
    core_rows: np.ndarray,
    xyz: np.ndarray,
    scale: np.ndarray,
    v1: dict[str, np.ndarray],
    neighbours: list[set[int]],
    cfg: VegetationInstanceConfig,
) -> tuple[list[np.ndarray] | None, bool]:
    """Apply the six-slice weak-gap cut; True means a qualifying cut was unsafe."""
    z = xyz[rows, 2]
    z0, z1 = float(z.min()), float(z.max())
    if z1 <= z0:
        return None, False
    dz = (z1 - z0) / cfg.slice_count
    bins = np.minimum(((z - z0) / dz).astype(np.int32), cfg.slice_count - 1)
    voxel = _voxel_size(rows, scale, cfg)
    counts = np.bincount(bins, minlength=cfg.slice_count)
    xy_counts = np.zeros(cfg.slice_count, np.int32)
    for index in range(cfg.slice_count):
        xy_counts[index] = len(
            np.unique(np.floor(xyz[rows[bins == index], :2] / voxel).astype(np.int64), axis=0)
        )
    weak = (counts < cfg.slice_min_points) | (xy_counts < 2)
    runs: list[list[int]] = []
    for index in np.flatnonzero(weak):
        if not runs or index != runs[-1][-1] + 1:
            runs.append([int(index)])
        else:
            runs[-1].append(int(index))
    if not runs:
        return None, False
    run = sorted(runs, key=lambda values: (-len(values), values[0]))[0]
    if len(run) < cfg.bridge_min_weak_slice_run or run[0] == 0 or run[-1] == cfg.slice_count - 1:
        return None, False
    gap_low, gap_high = z0 + run[0] * dz, z0 + (run[-1] + 1) * dz
    regions = np.where(z < gap_low, -1, np.where(z > gap_high, 1, 0))
    if not np.any(regions < 0) or not np.any(regions > 0):
        return None, False
    finite_scale = scale[rows][np.isfinite(scale[rows])]
    separation = float(z[regions > 0].min() - z[regions < 0].max())
    if separation < cfg.bridge_vertical_gap_scale_multiplier * (
        float(np.median(finite_scale)) if len(finite_scale) else voxel
    ):
        return None, False
    position = {int(row): pos for pos, row in enumerate(core_rows)}
    rowset = set(int(value) for value in rows)
    degree: dict[int, int] = defaultdict(int)
    local_voxel = np.floor(xyz[rows] / voxel).astype(np.int64)
    support = {
        tuple(key): int(count)
        for key, count in zip(*np.unique(local_voxel, axis=0, return_counts=True), strict=True)
    }
    for left, right, _ in edges:
        a, b = int(core_rows[left]), int(core_rows[right])
        if a in rowset and b in rowset:
            degree[a] += 1
            degree[b] += 1
    region_by = {int(row): int(region) for row, region in zip(rows, regions, strict=True)}
    key_by = {int(row): tuple(key) for row, key in zip(rows, local_voxel, strict=True)}
    removed: set[tuple[int, int]] = set()
    for left, right, normalized in edges:
        a, b = int(core_rows[left]), int(core_rows[right])
        if a not in rowset or b not in rowset:
            continue
        ra, rb = region_by[a], region_by[b]
        if ra == rb:
            continue
        pair = (min(a, b), max(a, b))
        if {ra, rb} == {-1, 1}:
            removed.add(pair)
            continue
        shared = len(neighbours[position[a]] & neighbours[position[b]])
        weak_edge = (
            normalized >= cfg.bridge_normalized_length_threshold
            or shared <= cfg.bridge_min_shared_neighbors
            or min(degree[a], degree[b]) <= cfg.gap_weak_endpoint_degree_threshold
            or min(support[key_by[a]], support[key_by[b]]) <= cfg.gap_weak_voxel_support_threshold
        )
        if weak_edge:
            removed.add(pair)
    children = _partition_core_rows(rows, edges, core_rows, removed)
    # A retained strong transition that still reaches both sides invalidates the cut.
    if any(
        np.any(np.asarray([region_by[int(row)] for row in child]) < 0)
        and np.any(np.asarray([region_by[int(row)] for row in child]) > 0)
        for child in children
    ):
        return None, True
    valid = [
        child
        for child in children
        if _meets_support(child, xyz, scale, scale, v1["vegetation_level"], cfg)
    ]
    sides = [
        any(np.any(np.asarray([region_by[int(row)] for row in child]) == side) for child in valid)
        for side in (-1, 1)
    ]
    if not all(sides):
        return None, True
    return valid, False


def _mixed_region_candidate(
    rows: np.ndarray,
    edges: list[tuple[int, int, float]],
    core_rows: np.ndarray,
    xyz: np.ndarray,
    scale: np.ndarray,
    v1: dict[str, np.ndarray],
    a2a: dict[str, np.ndarray],
    neighbours: list[set[int]],
    cfg: VegetationInstanceConfig,
) -> tuple[list[np.ndarray] | None, bool]:
    """Use only qualified crown/surface slice runs as an optional split boundary."""
    z = xyz[rows, 2]
    z0, z1 = float(z.min()), float(z.max())
    if z1 <= z0:
        return None, False
    voxel, dz = _voxel_size(rows, scale, cfg), (z1 - z0) / cfg.slice_count
    local_bins = np.minimum(((z - z0) / dz).astype(np.int32), cfg.slice_count - 1)
    median_radius = []
    records: list[tuple[int, np.ndarray, float, float, float, float]] = []
    for index in range(cfg.slice_count):
        local = np.flatnonzero(local_bins == index)
        if len(local) < cfg.slice_min_points:
            continue
        xy = xyz[rows[local], :2]
        if len(np.unique(np.floor(xy / voxel).astype(np.int64), axis=0)) < 2:
            continue
        covariance = np.cov(xy.T, bias=True)
        dimensions = 4 * np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0))
        radius = math.sqrt(max(float(np.prod(dimensions)), 0)) / 2
        median_radius.append(radius)
        support = min(len(local) / 12, 1.0)
        records.append(
            (
                index,
                local,
                support,
                float(np.median(a2a["planarity"][rows[local]])),
                float(np.median(a2a["scattering"][rows[local]])),
                radius,
            )
        )
    if not records:
        return None, False
    radius_floor = 0.50 * float(np.median(median_radius))
    crown = [
        record
        for record in records
        if record[2] >= 0.50
        and record[4] >= 0.35
        and record[3] <= 0.45
        and record[5] >= radius_floor
    ]
    surface = [
        record
        for record in records
        if record[2] >= 0.50 and record[3] >= 0.55 and record[4] <= 0.25
    ]
    crown_indices, surface_indices = {item[0] for item in crown}, {item[0] for item in surface}
    dual = crown_indices & surface_indices
    crown = [item for item in crown if item[0] not in dual]
    surface = [item for item in surface if item[0] not in dual]

    def best_region(
        values: list[tuple[int, np.ndarray, float, float, float, float]],
    ) -> tuple[np.ndarray, float, tuple[float, float, float]] | None:
        by_index = {item[0]: item for item in values}
        runs: list[list[int]] = []
        for index in sorted(by_index):
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        choices = []
        for run in runs:
            mask = np.isin(local_bins, run)
            count = int(mask.sum())
            if count >= max(12, math.ceil(0.20 * len(rows))):
                support = _clip(count / max(0.25 * len(rows), 1)) * float(
                    np.mean([by_index[i][2] for i in run])
                )
                choices.append((count, run[0], np.flatnonzero(mask), support, by_index))
        if not choices:
            return None
        count, _start, local, support, indexed = sorted(
            choices, key=lambda value: (-value[0], value[1])
        )[0]
        indices = [item[0] for item in indexed.values() if item[0] in set(local_bins[local])]
        values_for_region = [indexed[index] for index in indices]
        return (
            local,
            support,
            tuple(
                float(np.median([item[column] for item in values_for_region]))
                for column in (3, 4, 5)
            ),
        )

    cr, sr = best_region(crown), best_region(surface)
    if cr is None or sr is None:
        return None, False
    change = max(
        _clip(abs(cr[2][0] - sr[2][0]) / 0.30),
        _clip(abs(cr[2][1] - sr[2][1]) / 0.25),
        _clip((max(cr[2][2], sr[2][2]) / max(min(cr[2][2], sr[2][2]), 1e-6) - 1) / 0.50),
    )
    if change < 0.60:
        return None, False
    crown_rows, surface_rows = (
        set(int(value) for value in rows[cr[0]]),
        set(int(value) for value in rows[sr[0]]),
    )
    position, rowset = (
        {int(row): pos for pos, row in enumerate(core_rows)},
        set(int(row) for row in rows),
    )
    removed: set[tuple[int, int]] = set()
    for left, right, normalized in edges:
        a, b = int(core_rows[left]), int(core_rows[right])
        if (
            a not in rowset
            or b not in rowset
            or not (
                (a in crown_rows and b in surface_rows) or (b in crown_rows and a in surface_rows)
            )
        ):
            continue
        if (
            normalized >= cfg.secondary_edge_normalized_distance_max
            or len(neighbours[position[a]] & neighbours[position[b]])
            <= cfg.bridge_min_shared_neighbors
        ):
            removed.add((min(a, b), max(a, b)))
    children = _partition_core_rows(rows, edges, core_rows, removed)
    valid = [
        child
        for child in children
        if _meets_support(child, xyz, scale, scale, v1["vegetation_level"], cfg)
    ]
    if len(valid) < 2:
        return None, True
    return valid, False


def _safe_split_rows(
    rows: np.ndarray,
    edges: list[tuple[int, int, float]],
    core_rows: np.ndarray,
    xyz: np.ndarray,
    scale: np.ndarray,
    v1: dict[str, np.ndarray],
    a2a: dict[str, np.ndarray],
    a2b: dict[str, np.ndarray],
    neighbours: list[set[int]],
    cfg: VegetationInstanceConfig,
    depth: int = 0,
    split_counts: dict[str, int] | None = None,
) -> tuple[list[np.ndarray], bool]:
    """Fixed-precedence, stale-candidate-free conservative structural recursion."""
    split_counts = split_counts if split_counts is not None else defaultdict(int)
    bridges = _bridge_candidates(rows, edges, core_rows, xyz, scale, v1, a2a, a2b, neighbours, cfg)
    selected: list[np.ndarray] | None = None
    mechanism = ""
    if bridges:
        candidate = bridges[0]
        valid = [
            child
            for child in candidate["children"]
            if _meets_support(child, xyz, scale, scale, v1["vegetation_level"], cfg)
        ]
        if len(valid) < 2:
            split_counts["unresolved"] += 1
            return [rows], True
        selected, mechanism = valid, "bridge"
    else:
        selected, unsafe = _vertical_gap_candidate(
            rows, edges, core_rows, xyz, scale, v1, neighbours, cfg
        )
        if unsafe:
            split_counts["unresolved"] += 1
            return [rows], True
        if selected is not None:
            mechanism = "gap"
        else:
            selected, unsafe = _mixed_region_candidate(
                rows, edges, core_rows, xyz, scale, v1, a2a, neighbours, cfg
            )
            if unsafe:
                split_counts["unresolved"] += 1
                return [rows], True
            if selected is not None:
                mechanism = "mixed"
    if selected is None:
        return [rows], False
    split_counts[mechanism] += 1
    if depth >= cfg.bridge_max_split_depth or len(selected) > cfg.bridge_max_children_per_parent:
        split_counts["unresolved"] += 1
        return [rows], True
    output: list[np.ndarray] = []
    unresolved = False
    for child in selected:
        branches, child_unresolved = _safe_split_rows(
            child,
            edges,
            core_rows,
            xyz,
            scale,
            v1,
            a2a,
            a2b,
            neighbours,
            cfg,
            depth + 1,
            split_counts,
        )
        output.extend(branches)
        unresolved |= child_unresolved
    return output, unresolved


def _low_expand(
    candidate_rows: np.ndarray,
    clusters: list[dict[str, Any]],
    cluster_by: np.ndarray,
    xyz: np.ndarray,
    scale: np.ndarray,
    state: np.ndarray,
    level: np.ndarray,
    v1: dict[str, np.ndarray],
    a2b: dict[str, np.ndarray],
    cfg: VegetationInstanceConfig,
    reason: np.ndarray,
    token: CancellationToken | None,
) -> None:
    if not clusters:
        return
    core_rows = np.concatenate([item["core"] for item in clusters])
    tree = cKDTree(xyz[core_rows])
    hag_state, hag = v1["hag_state"], a2b["height_agl_m"]
    for low in candidate_rows[level[candidate_rows] == int(VegetationCandidateLevel.LOW)]:
        if state[low] == int(LocalScaleState.LOCAL_SCALE_UNAVAILABLE):
            continue
        _check_cancelled(token, "during LOW expansion")
        ids = tree.query_ball_point(xyz[low], r=cfg.low_search_radius_m)
        ids = ids[: cfg.low_support_cap]
        support: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for pos in ids:
            core = int(core_rows[pos])
            cid = int(cluster_by[core])
            if cid < 0:
                continue
            rho = float(
                np.clip(
                    cfg.low_radius_multiplier
                    * cfg.core_radius_multiplier
                    * math.sqrt(float(scale[low]) * float(scale[core])),
                    0.040,
                    cfg.low_search_radius_m,
                )
            )
            dh = float(np.linalg.norm(xyz[low, :2] - xyz[core, :2]))
            dv = abs(float(xyz[low, 2] - xyz[core, 2]))
            da = math.sqrt(dh * dh + (dv / cfg.vertical_scale_multiplier) ** 2)
            if dh > rho or dv > cfg.vertical_scale_multiplier * rho or da > rho:
                continue
            if (
                hag_state[low] == int(VegetationHagState.RELIABLE)
                and hag_state[core] == int(VegetationHagState.RELIABLE)
                and abs(float(hag[low]) - float(hag[core])) > 0.20 + 2 * rho
            ):
                continue
            support[cid].append((da / rho, float(v1["vegetation_confidence"][core])))
        valid = [(cid, values) for cid, values in support.items() if len(values) >= 2]
        if len(valid) != 1:
            reason[low] |= int(
                VegetationInstanceReasonFlag.LOW_AMBIGUOUS_MULTI_CLUSTER
                if len(valid) > 1
                else VegetationInstanceReasonFlag.LOW_NO_CORE_SUPPORT
            )
            continue
        cid, values = valid[0]
        clusters[cid]["members"] = np.append(clusters[cid]["members"], low)
        cluster_by[low] = cid


def _apply_manual(
    log: VegetationManualLog,
    clusters: list[dict[str, Any]],
    cluster_by: np.ndarray,
    candidate: np.ndarray,
    wire: np.ndarray,
    xyz: np.ndarray,
    original: np.ndarray,
    reason: np.ndarray,
    manual: np.ndarray,
) -> None:
    if not log.actions:
        return
    by_original = {int(v): i for i, v in enumerate(original)}
    for action in log.actions:
        targets = list(action.cluster_ids)
        if action.action in {"confirm_cluster", "reject_cluster"}:
            if len(targets) != 1 or targets[0] >= len(clusters):
                raise _error(
                    "V2_MANUAL_TARGET", "manual status action requires one current cluster"
                )
            item = clusters[targets[0]]
            item["manual_status"] = action.action
            manual[item["members"]] |= int(
                VegetationInstanceManualFlag.CONFIRMED
                if action.action == "confirm_cluster"
                else VegetationInstanceManualFlag.REJECTED
            )
        elif action.action == "merge_clusters":
            if len(targets) < 2 or any(v >= len(clusters) for v in targets):
                raise _error("V2_MANUAL_TARGET", "manual merge requires valid clusters")
            base = clusters[min(targets)]
            base["members"] = np.unique(np.concatenate([clusters[v]["members"] for v in targets]))
            manual[base["members"]] |= int(VegetationInstanceManualFlag.MERGED)
            for v in sorted(targets[1:], reverse=True):
                clusters.pop(v)
        else:
            selected = [by_original[v] for v in action.original_indices if v in by_original]
            for lo, hi in action.boxes:
                selected.extend(np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1)).tolist())
            selected = np.unique(np.asarray(selected, np.int64))
            if np.any(wire[selected] >= 0):
                raise _error("V2_MANUAL_WIRE", "manual operation selected confirmed-wire point")
            if action.action == "exclude_points_or_boxes":
                for item in clusters:
                    item["members"] = np.setdiff1d(item["members"], selected)
                    if not len(item["members"]):
                        raise _error(
                            "V2_MANUAL_EXCLUDE", "manual exclusion may not empty a cluster"
                        )
                manual[selected] |= int(VegetationInstanceManualFlag.EXCLUDED)
                reason[selected] |= int(VegetationInstanceReasonFlag.MANUAL_EXCLUDED)
            elif action.action == "include_unassigned_candidates":
                if (
                    len(targets) != 1
                    or targets[0] >= len(clusters)
                    or np.any(~candidate[selected])
                    or np.any(cluster_by[selected] >= 0)
                ):
                    raise _error(
                        "V2_MANUAL_INCLUDE",
                        "include requires unassigned candidates and target cluster",
                    )
                clusters[targets[0]]["members"] = np.unique(
                    np.append(clusters[targets[0]]["members"], selected)
                )
                manual[selected] |= int(VegetationInstanceManualFlag.INCLUDED)
                reason[selected] |= int(VegetationInstanceReasonFlag.MANUAL_INCLUDED)
            elif action.action.startswith("split_cluster"):
                if len(targets) != 1 or targets[0] >= len(clusters) or not len(selected):
                    raise _error(
                        "V2_MANUAL_SPLIT", "split requires target and selected seed/box points"
                    )
                item = clusters[targets[0]]
                inside = np.intersect1d(item["members"], selected)
                if not len(inside) or len(inside) == len(item["members"]):
                    raise _error("V2_MANUAL_SPLIT", "split selection must divide target cluster")
                item["members"] = np.setdiff1d(item["members"], inside)
                clusters.append(
                    {
                        "core": inside,
                        "members": inside,
                        "parent": item.get("parent", -1),
                        "unresolved": False,
                    }
                )
                manual[inside] |= int(VegetationInstanceManualFlag.SPLIT)


def _describe_cluster(
    item: dict[str, Any],
    xyz: np.ndarray,
    v1: dict[str, np.ndarray],
    a2a: dict[str, np.ndarray],
    a2b: dict[str, np.ndarray],
    raw: np.ndarray,
    scale: np.ndarray,
    cfg: VegetationInstanceConfig,
) -> dict[str, Any]:
    rows = np.unique(item["members"])
    pts = xyz[rows]
    n = len(rows)
    centroid = pts.mean(axis=0)
    weights = np.maximum(v1["vegetation_rule_score"][rows], 0.05)
    representative = np.average(pts, axis=0, weights=weights)
    cov = (pts - centroid).T @ (pts - centroid) / max(n, 1)
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    eig, vec = eig[order], vec[:, order]
    for col in range(3):
        pivot = int(np.argmax(np.abs(vec[:, col])))
        if vec[pivot, col] < 0:
            vec[:, col] *= -1
    xy_cov = (
        (pts[:, :2] - pts[:, :2].mean(axis=0)).T
        @ (pts[:, :2] - pts[:, :2].mean(axis=0))
        / max(n, 1)
    )
    xy_eig = np.sort(np.linalg.eigvalsh(xy_cov))[::-1]
    hlen, hwid = 4 * np.sqrt(np.maximum(xy_eig, 0))
    lower, upper = pts.min(axis=0), pts.max(axis=0)
    extent = upper - lower
    vertical = float(extent[2])
    rmajor, rminor = vertical / max(float(hlen), 0.05), vertical / max(float(hwid), 0.05)
    top = rows[np.lexsort((v1["original_index"][rows], -pts[:, 2]))[0]]
    reliable = v1["hag_state"][rows] == int(VegetationHagState.RELIABLE)
    hags = a2b["height_agl_m"][rows]
    finite = np.isfinite(hags)
    reliable_hag = hags[reliable & finite]
    base_available = len(reliable_hag) >= 5
    if base_available:
        q10, q25, q75 = np.percentile(reliable_hag, [10, 25, 75])
        base_rows = rows[reliable & finite & (hags <= q25)]
        base_xyz = np.average(
            xyz[base_rows], axis=0, weights=np.maximum(v1["vegetation_rule_score"][base_rows], 0.05)
        )
        base_uncertainty = float(q75 - q25 + 0.25 * (1 - reliable.mean()) * vertical)
    else:
        q10, base_xyz, base_uncertainty = 0.0, centroid, 0.0
    hvalid = hags[finite]
    hstats = np.percentile(hvalid, [0, 25, 50, 75, 95, 100]) if len(hvalid) else np.zeros(6)
    normz = (pts[:, 2] - lower[2]) / max(vertical, 1e-9)
    bins = np.minimum((normz * 5).astype(int), 4)
    occupancy = np.bincount(bins, minlength=5) / n
    fractions = [
        float((normz < 1 / 3).mean()),
        float(((normz >= 1 / 3) & (normz < 2 / 3)).mean()),
        float((normz >= 2 / 3).mean()),
    ]
    voxel = _voxel_size(rows, raw, cfg)
    keys3 = np.floor(pts / voxel).astype(np.int64)
    keysxy = keys3[:, :2]
    occ3, counts3 = np.unique(keys3, axis=0, return_counts=True)
    occxy = len(np.unique(keysxy, axis=0))
    regions, largest = _voxel_regions(occ3, counts3)
    planar = a2a["planarity"][rows]
    scattering = a2a["scattering"][rows]
    linearity = a2a["linearity"][rows]
    conf = v1["vegetation_confidence"][rows]
    score = v1["vegetation_rule_score"][rows]
    d = {
        "cluster_id": item["id"],
        "automatic_parent_cluster_id": int(item["parent"]),
        "member_rows": rows,
        "point_count": n,
        "core_point_count": len(item["core"]),
        "median_core_degree": float(item.get("median_core_degree", 0.0)),
        "median_shared_neighbors": float(item.get("median_shared_neighbors", 0.0)),
        "local_scale_quality": float(item.get("q_scale", 0.0)),
        "nonunavailable_scale_fraction": float(item.get("nonunavailable_fraction", 0.0)),
        "centroid_xyz": centroid.tolist(),
        "representative_center_xyz": representative.tolist(),
        "aabb_min_xyz": lower.tolist(),
        "aabb_max_xyz": upper.tolist(),
        "aabb_dimensions_m": extent.tolist(),
        "top_row_index": int(top),
        "top_xyz": xyz[top].tolist(),
        "base_available": bool(base_available),
        "base_hag_m": float(q10),
        "base_xyz": np.asarray(base_xyz).tolist(),
        "base_uncertainty_m": float(base_uncertainty),
        "pca_dimensions_m": (4 * np.sqrt(np.maximum(eig, 0))).tolist(),
        "pca_directions": vec.T.tolist(),
        "horizontal_length_m": float(hlen),
        "horizontal_width_m": float(hwid),
        "vertical_extent_m": vertical,
        "vertical_to_major_horizontal": float(rmajor),
        "vertical_to_minor_horizontal": float(rminor),
        "hag_stats_m": hstats.tolist(),
        "vertical_bin_occupancy": occupancy.tolist(),
        "vertical_portion_fractions": fractions,
        "footprint_proxy_m2": float(occxy * voxel * voxel),
        "occupancy_proxy_m3": float(len(occ3) * voxel**3),
        "voxel_size_m": voxel,
        "occupied_voxel_count": int(len(occ3)),
        "voxel_region_count": regions,
        "largest_voxel_region_point_count": largest,
        "fragmentation": float(1 - largest / n),
        "level_counts": [
            int(np.count_nonzero(v1["vegetation_level"][rows] == v)) for v in (1, 2, 3)
        ],
        "score_stats": np.percentile(score, [0, 25, 50, 75, 95, 100]).tolist(),
        "confidence_stats": np.percentile(conf, [0, 25, 50, 75, 95, 100]).tolist(),
        "reliable_hag_fraction": float(reliable.mean()),
        "uncertain_hag_fraction": float(
            (v1["hag_state"][rows] == int(VegetationHagState.UNCERTAIN)).mean()
        ),
        "planar_fraction": float((planar >= 0.65).mean()),
        "scattered_fraction": float((scattering >= 0.10).mean()),
        "linear_fraction": float((linearity >= 0.85).mean()),
        "planarity_median": float(np.median(planar)),
        "scattering_median": float(np.median(scattering)),
        "unresolved_bridge": bool(item["unresolved"]),
    }
    return _add_morphology(d, pts, rows, v1, a2a, a2b, raw, scale, cfg)


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _add_morphology(
    d: dict[str, Any],
    pts: np.ndarray,
    rows: np.ndarray,
    v1: dict[str, np.ndarray],
    a2a: dict[str, np.ndarray],
    a2b: dict[str, np.ndarray],
    raw: np.ndarray,
    scale: np.ndarray,
    cfg: VegetationInstanceConfig,
) -> dict[str, Any]:
    """Attach fixed-shape slice, morphology, and conservative ambiguity evidence."""
    voxel = float(d["voxel_size_m"])
    n = len(rows)
    z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
    dz = (z1 - z0) / cfg.slice_count
    finite_scale = scale[rows][np.isfinite(scale[rows])]
    median_scale = float(np.median(finite_scale)) if len(finite_scale) else voxel
    slice_valid = np.zeros(cfg.slice_count, np.uint8)
    slice_count = np.zeros(cfg.slice_count, np.int32)
    slice_core = np.zeros(cfg.slice_count, np.int32)
    slice_support = np.zeros(cfg.slice_count, np.float32)
    slice_radius = np.full(cfg.slice_count, np.nan, np.float32)
    slice_dims = np.full((cfg.slice_count, 2), np.nan, np.float32)
    slice_xy_voxels = np.zeros(cfg.slice_count, np.int32)
    slice_planarity = np.full(cfg.slice_count, np.nan, np.float32)
    slice_scattering = np.full(cfg.slice_count, np.nan, np.float32)
    slice_verticality = np.full(cfg.slice_count, np.nan, np.float32)
    slice_hm_fraction = np.zeros(cfg.slice_count, np.float32)
    slice_centroid = np.full((cfg.slice_count, 3), np.nan, np.float32)
    levels = v1["vegetation_level"][rows]
    for i in range(cfg.slice_count):
        lo, hi = z0 + i * dz, z0 + (i + 1) * dz
        use = (pts[:, 2] >= lo) & (
            (pts[:, 2] <= hi) if i == cfg.slice_count - 1 else (pts[:, 2] < hi)
        )
        ix = np.flatnonzero(use)
        slice_count[i] = len(ix)
        if not len(ix):
            continue
        q = pts[ix]
        keys = np.floor(q[:, :2] / voxel).astype(np.int64)
        xyvox = len(np.unique(keys, axis=0))
        slice_xy_voxels[i] = xyvox
        hm = np.isin(
            levels[ix], [int(VegetationCandidateLevel.MEDIUM), int(VegetationCandidateLevel.HIGH)]
        )
        slice_core[i] = int(hm.sum())
        slice_hm_fraction[i] = float(hm.mean())
        confidence = (
            0.50 * min(len(ix) / 12, 1) + 0.30 * min(hm.sum() / 8, 1) + 0.20 * min(xyvox / 3, 1)
        )
        slice_support[i] = confidence
        if len(ix) < cfg.slice_min_points or xyvox < 2 or dz < median_scale:
            continue
        c = q.mean(axis=0)
        cov = (q[:, :2] - c[:2]).T @ (q[:, :2] - c[:2]) / len(q)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        dims = 4 * np.sqrt(np.maximum(eig, 0))
        slice_valid[i] = 1
        slice_centroid[i] = c.astype(np.float32)
        slice_dims[i] = dims.astype(np.float32)
        slice_radius[i] = np.float32(math.sqrt(max(float(dims[0] * dims[1]), 0)) / 2)
        slice_planarity[i] = np.float32(np.median(a2a["planarity"][rows[ix]]))
        slice_scattering[i] = np.float32(np.median(a2a["scattering"][rows[ix]]))
        slice_verticality[i] = np.float32(np.median(a2a["verticality"][rows[ix]]))

    valid = np.flatnonzero(slice_valid)
    valid_fraction = len(valid) / cfg.slice_count
    groups = (valid[(valid <= 1)], valid[(valid >= 2) & (valid <= 3)], valid[valid >= 4])
    radii = [float(np.median(slice_radius[g])) if len(g) else float("nan") for g in groups]
    lower, middle, upper = radii
    if all(np.isfinite(radii)):
        expansion = middle / max(lower, 1e-6)
        contraction = upper / max(middle, 1e-6)
        expansion_score = _clip((expansion - 1.15) / 0.35)
        contraction_score = _clip((1 - contraction) / 0.25)
    else:
        expansion = contraction = float("nan")
        expansion_score = contraction_score = 0.0
    valid_radii = slice_radius[valid]
    radius_cv = (
        float(np.std(valid_radii) / max(float(np.mean(valid_radii)), 1e-6)) if len(valid) else 0.0
    )
    nonconstant = _clip((radius_cv - 0.15) / 0.35)
    profile = (
        0.35 * expansion_score
        + 0.25 * contraction_score
        + 0.20 * nonconstant
        + 0.20 * valid_fraction
    )
    if len(valid) < 3:
        profile = min(profile, 0.40)

    pca = np.asarray(d["pca_dimensions_m"], float)
    p1, p2, p3 = max(float(pca[0]), 1e-6), float(pca[1]), float(pca[2])
    pca_balance = 0.50 * _clip((p2 / p1 - 0.25) / 0.35) + 0.50 * _clip((p3 / p1 - 0.12) / 0.25)
    all_keys = np.floor(pts / voxel).astype(np.int64)
    nxy = len(np.unique(all_keys[:, :2], axis=0))
    nz = len(np.unique(all_keys[:, 2]))
    depth = _clip((nz / max(math.sqrt(nxy), 1) - 0.30) / 0.50)
    med_r = float(np.median(valid_radii)) if len(valid) else float("inf")
    multislice = (
        float(np.count_nonzero(valid_radii >= 0.50 * med_r) / cfg.slice_count)
        if len(valid)
        else 0.0
    )
    volumetric = 0.40 * pca_balance + 0.30 * depth + 0.20 * multislice + 0.10 * valid_fraction

    planar_med = float(d["planarity_median"])
    scatter_med = float(d["scattering_median"])
    scatter_raw = 0.60 * float(d["scattered_fraction"]) + 0.40 * _clip((scatter_med - 0.15) / 0.25)
    planar_penalty = 0.50 * _clip((planar_med - 0.45) / 0.35) + 0.50 * float(d["planar_fraction"])
    scatter = _clip(scatter_raw * (1 - 0.50 * planar_penalty))
    vertical_support = (
        0.45 * valid_fraction
        + 0.35 * (np.count_nonzero(slice_core >= 4) / cfg.slice_count)
        + 0.20 * (float(np.mean(slice_support[valid])) if len(valid) else 0.0)
    )
    hag_quality = _clip(
        float(d["reliable_hag_fraction"]) + 0.50 * float(d["uncertain_hag_fraction"])
    )
    morphology = (
        0.30 * volumetric
        + 0.25 * profile
        + 0.20 * scatter
        + 0.15 * vertical_support
        + 0.10 * hag_quality
    )

    # PCA-aligned grid: the existing basis is deterministic; re-orthogonalise only
    # for rank-deficient numerical cases.
    basis = np.asarray(d["pca_directions"], float).T
    if not np.isfinite(basis).all() or abs(np.linalg.det(basis)) < 1e-8:
        basis = np.eye(3)
    q = (pts - np.asarray(d["centroid_xyz"], float)) @ basis
    pkeys = np.floor(q / voxel + 1e-9).astype(np.int64)
    pocc = len(np.unique(pkeys, axis=0))
    span = np.ptp(pkeys, axis=0) + 1
    pfill = _clip(pocc / max(int(np.prod(span, dtype=np.int64)), 1))
    pcov = (np.unique(pkeys, axis=0).astype(float) - np.mean(np.unique(pkeys, axis=0), axis=0)).T
    pcov = pcov @ pcov.T / max(pocc, 1)
    veig = np.sort(np.linalg.eigvalsh(pcov))[::-1]
    voxel_surface = _clip(
        (0.25 - math.sqrt(max(veig[2], 0)) / max(math.sqrt(max(veig[1], 0)), 1e-6)) / 0.25
    )

    rmajor = float(d["vertical_to_major_horizontal"])
    axis = basis[:, 0]
    alignment = abs(float(axis[2]))
    slender_axis = (
        0.30 * _clip((rmajor - 1.50) / 1.50)
        + 0.25 * _clip((0.40 - p2 / p1) / 0.25)
        + 0.20 * _clip((0.30 - p3 / p1) / 0.20)
        + 0.25 * _clip((alignment - 0.75) / 0.20)
    )
    constancy = 1 - _clip((radius_cv - 0.10) / 0.25)
    crown_association = bool(
        all(np.isfinite(radii))
        and expansion >= 1.35
        and len(groups[1]) >= 2
        and np.count_nonzero(slice_support[np.r_[groups[1], groups[2]]] >= 0.60) >= 2
        and profile >= 0.55
    )
    isolated_slender = slender_axis * constancy * (1 - float(crown_association))

    pca_thin = _clip((0.20 - p3 / max(p2, 1e-6)) / 0.20)
    dims = np.asarray(d["aabb_dimensions_m"], float)
    aabb_thin = _clip((0.20 - float(dims.min()) / max(float(np.median(dims)), 1e-6)) / 0.20)
    planar_feature = 0.50 * float(d["planar_fraction"]) + 0.50 * _clip((planar_med - 0.45) / 0.35)
    scatter_deficit = 1 - _clip((scatter_med - 0.10) / 0.25)
    planar_channels = (
        pca_thin,
        aabb_thin,
        planar_feature,
        scatter_deficit,
        voxel_surface,
        constancy,
    )
    planar = float(np.mean(planar_channels))
    planar_bool = sum(
        v >= (0.70 if idx == 5 else (0.55 if idx == 2 else 0.60))
        for idx, v in enumerate(planar_channels)
    ) >= 3 and (planar_feature >= 0.55 or (pca_thin >= 0.75 and aabb_thin >= 0.75))

    footprint_ratio = math.sqrt(max(float(d["footprint_proxy_m2"]), 0)) / max(
        float(d["vertical_extent_m"]), 0.05
    )
    broad = _clip((footprint_ratio - 1.10) / 0.90)
    layer = 0.0
    if len(valid):
        layer = _clip(
            np.count_nonzero(
                (slice_planarity[valid] >= 0.55)
                & (slice_scattering[valid] <= 0.25)
                & (slice_verticality[valid] <= 0.60)
                & (slice_support[valid] >= 0.50)
            )
            / 3
        )
    fractions = slice_xy_voxels.astype(float) / max(nxy, 1)
    pairs = [
        abs(fractions[i] - fractions[i + 1])
        for i in range(cfg.slice_count - 1)
        if slice_valid[i] and slice_valid[i + 1]
    ]
    occupancy_transition = max(pairs, default=0.0)
    abrupt = _clip((occupancy_transition - 0.15) / 0.35)
    low_hag = (
        _clip((1.50 - float(d["hag_stats_m"][2])) / 1.0)
        if float(d["reliable_hag_fraction"]) >= 0.50
        else None
    )
    compact_channels = [broad, layer, abrupt, 1 - profile, pfill, scatter_deficit] + (
        [] if low_hag is None else [low_hag]
    )
    compact = float(np.mean(compact_channels))
    compact_bool = sum(v >= 0.60 for v in compact_channels) >= 3 and (
        broad >= 0.60 or pfill >= 0.60
    )

    # Slice regions are intentionally bounded; dual slices do not define a split boundary.
    med_radius = float(np.median(valid_radii)) if len(valid) else float("inf")
    crown_slices = [
        i
        for i in valid
        if slice_support[i] >= 0.50
        and slice_scattering[i] >= 0.35
        and slice_planarity[i] <= 0.45
        and slice_radius[i] >= 0.50 * med_radius
    ]
    surface_slices = [
        i
        for i in valid
        if slice_support[i] >= 0.50 and slice_planarity[i] >= 0.55 and slice_scattering[i] <= 0.25
    ]
    dual = set(crown_slices) & set(surface_slices)
    crown_slices = [i for i in crown_slices if i not in dual]
    surface_slices = [i for i in surface_slices if i not in dual]

    def region(
        indices: list[int],
    ) -> tuple[np.ndarray, float, np.ndarray | None, float, float, float]:
        runs = []
        for value in indices:
            if not runs or value != runs[-1][-1] + 1:
                runs.append([value])
            else:
                runs[-1].append(value)
        candidates = []
        for run in runs:
            mask = np.isin(
                np.floor((pts[:, 2] - z0) / max(dz, 1e-9)).clip(0, cfg.slice_count - 1), run
            )
            candidates.append((int(mask.sum()), run, mask))
        if not candidates:
            return np.empty(0, np.int64), 0.0, None, 0.0, 0.0, 0.0
        count, run, mask = sorted(candidates, key=lambda x: (-x[0], x[1][0]))[0]
        if count < max(12, math.ceil(0.20 * n)):
            return np.empty(0, np.int64), 0.0, None, 0.0, 0.0, 0.0
        ix = np.flatnonzero(mask)
        support = _clip(count / max(0.25 * n, 1)) * float(np.mean(slice_support[run]))
        return (
            ix,
            support,
            pts[ix].mean(axis=0),
            float(np.median(slice_planarity[run])),
            float(np.median(slice_scattering[run])),
            float(np.median(slice_radius[run])),
        )

    cr, crsup, crc, crp, crs, crr = region(crown_slices)
    sr, srsup, src, srp, srs, srr = region(surface_slices)
    if len(cr) and len(sr):
        separation = _clip(
            float(np.linalg.norm(crc - src)) / max(float(np.linalg.norm(dims)), 2 * voxel)
        )
        change = max(
            _clip(abs(crp - srp) / 0.30),
            _clip(abs(crs - srs) / 0.25),
            _clip((max(crr, srr) / max(min(crr, srr), 1e-6) - 1) / 0.50),
        )
    else:
        separation = change = 0.0
    high_conflict = morphology >= 0.65 and max(planar, compact) >= 0.55
    mixed = (
        0.30 * min(crsup, srsup) + 0.25 * separation + 0.30 * change + 0.15 * float(high_conflict)
    )
    mixed_bool = mixed >= 0.60 or high_conflict

    positive = sum((volumetric >= 0.60, profile >= 0.55, scatter >= 0.55, vertical_support >= 0.65))
    count_score = _clip((d["core_point_count"] - 12) / 36)
    voxel_score = _clip((d["occupied_voxel_count"] - 3) / 7)
    degree_score = _clip((d["median_core_degree"] - 2) / 4)
    shared_score = _clip((d["median_shared_neighbors"] - 2) / 3)
    grouping = (
        0.30 * count_score
        + 0.20 * voxel_score
        + 0.20 * degree_score
        + 0.15 * shared_score
        + 0.15 * d["local_scale_quality"]
    )
    topology = _clip(1 - 0.45 * d["fragmentation"] - 0.35 * float(d["unresolved_bridge"]))
    completeness = 0.50 * valid_fraction + 0.50 * min(positive / 3, 1)
    assessment = _clip(0.35 * grouping + 0.30 * completeness + 0.20 * topology + 0.15 * hag_quality)
    d.update(
        {
            "slice_valid": slice_valid.tolist(),
            "slice_point_count": slice_count.tolist(),
            "slice_core_hm_count": slice_core.tolist(),
            "slice_support_confidence": slice_support.tolist(),
            "slice_equivalent_radius_m": slice_radius.tolist(),
            "slice_horizontal_dimensions_m": slice_dims.tolist(),
            "slice_xy_voxel_count": slice_xy_voxels.tolist(),
            "slice_planarity_median": slice_planarity.tolist(),
            "slice_scattering_median": slice_scattering.tolist(),
            "slice_verticality_median": slice_verticality.tolist(),
            "slice_hm_support_fraction": slice_hm_fraction.tolist(),
            "expansion_ratio": expansion,
            "contraction_ratio": contraction,
            "slice_radius_cv": radius_cv,
            "pca_box_fill": pfill,
            "voxel_surface_thinness": voxel_surface,
            "volumetric_score": volumetric,
            "profile_score": profile,
            "scatter_score": scatter,
            "vertical_support_score": vertical_support,
            "hag_quality_score": hag_quality,
            "vegetation_morphology_score": morphology,
            "independent_positive_count": int(positive),
            "grouping_support_score": grouping,
            "topology_quality_score": topology,
            "assessment_confidence": assessment,
            "crown_association": crown_association,
            "isolated_slender_ambiguity_score": isolated_slender,
            "isolated_slender_ambiguity": bool(isolated_slender >= 0.60),
            "planar_ambiguity_score": planar,
            "planar_ambiguity": bool(planar_bool),
            "compact_layered_ambiguity_score": compact,
            "compact_layered_ambiguity": bool(compact_bool),
            "mixed_morphology_conflict_score": mixed,
            "mixed_morphology_conflict": bool(mixed_bool),
            "region_support_crown": crsup,
            "region_support_surface": srsup,
            "normalized_spatial_separation": separation,
            "morphology_change_score": change,
            "vertical_gap_score": 0.0,
            "unresolved_topology": bool(d["unresolved_bridge"]),
            "requires_review": False,
        }
    )
    return d


def _voxel_regions(keys: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    lookup = {tuple(v): i for i, v in enumerate(keys)}
    seen: set[int] = set()
    region_weights = []
    offsets = [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ]
    for start in range(len(keys)):
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        total = 0
        while stack:
            i = stack.pop()
            total += int(weights[i])
            key = keys[i]
            for off in offsets:
                j = lookup.get((int(key[0] + off[0]), int(key[1] + off[1]), int(key[2] + off[2])))
                if j is not None and j not in seen:
                    seen.add(j)
                    stack.append(j)
        region_weights.append(total)
    return len(region_weights), max(region_weights, default=0)


def _statuses(
    descs: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    manual: np.ndarray,
    cfg: VegetationInstanceConfig,
) -> list[VegetationClusterStatus]:
    statuses = []
    for d, item in zip(descs, clusters, strict=True):
        hard_morphology = bool(
            d["planar_ambiguity"]
            or d["isolated_slender_ambiguity"]
            or d["compact_layered_ambiguity"]
            or d["mixed_morphology_conflict"]
        )
        hard_topology = bool(
            d["unresolved_topology"] or d["fragmentation"] >= cfg.fragmentation_review_threshold
        )
        positive = int(d["independent_positive_count"])
        morphology = float(d["vegetation_morphology_score"])
        confidence = float(d["assessment_confidence"])
        confirmed = bool(
            d["grouping_support_score"] >= cfg.grouping_support_threshold
            and morphology >= cfg.promotion_morphology_threshold
            and confidence >= cfg.promotion_confidence_threshold
            and positive >= cfg.minimum_independent_positive_channels
            and (d["volumetric_score"] >= 0.60 or d["profile_score"] >= 0.55)
            and d["expansion_ratio"] >= cfg.promotion_min_crown_expansion_ratio
            and d["contraction_ratio"] <= cfg.promotion_max_crown_contraction_ratio
            and not hard_morphology
            and not hard_topology
        )
        review = bool(
            hard_morphology
            or hard_topology
            or (
                confidence >= cfg.review_confidence_threshold
                and morphology >= cfg.review_morphology_threshold
                and not confirmed
            )
            or (positive == 2 and (d["volumetric_score"] >= 0.60 or d["profile_score"] >= 0.55))
        )
        d.update(
            {
                "hard_morphology_review": hard_morphology,
                "hard_topology_review": hard_topology,
                "requires_review": review,
                # Legacy explanatory fields remain, now derived from generalized channels.
                "crown_like_evidence": float(max(d["volumetric_score"], d["profile_score"])),
                "low_shrub_evidence": 0.0,
                "shallow_horizontal_suspicion": float(d["compact_layered_ambiguity_score"]),
                "planar_structure_suspicion": float(d["planar_ambiguity_score"]),
                "linear_structure_suspicion": float(d["isolated_slender_ambiguity_score"]),
                "vegetation_evidence": morphology,
                "artificial_suspicion": float(
                    max(
                        d["planar_ambiguity_score"],
                        d["compact_layered_ambiguity_score"],
                        d["isolated_slender_ambiguity_score"],
                    )
                ),
                "vegetation_plausibility_score": morphology,
                "conflicting_descriptor_evidence": bool(d["mixed_morphology_conflict"]),
            }
        )
        d["plausibility_label"] = _label(d, cfg).name
        override = item.get("manual_status")
        if override == "reject_cluster":
            status = VegetationClusterStatus.REJECTED_AS_INSTANCE
        elif override == "confirm_cluster" and not hard_topology and d["fragmentation"] < 0.50:
            status = VegetationClusterStatus.CONFIRMED
        elif hard_morphology or hard_topology or review:
            status = VegetationClusterStatus.REVIEW_REQUIRED
        elif confirmed:
            status = VegetationClusterStatus.CONFIRMED
        else:
            status = VegetationClusterStatus.PROVISIONAL
        statuses.append(status)
    return statuses


def _label(d: dict[str, Any], cfg: VegetationInstanceConfig) -> VegetationPlausibilityLabel:
    if d["unresolved_bridge"] or d["fragmentation"] >= cfg.fragmentation_review_threshold:
        return VegetationPlausibilityLabel.FRAGMENTED_OR_UNRESOLVED
    if d.get("mixed_morphology_conflict"):
        return VegetationPlausibilityLabel.MIXED_MORPHOLOGY_REVIEW
    if d.get("compact_layered_ambiguity"):
        return VegetationPlausibilityLabel.COMPACT_LAYERED_SUSPECT
    if d.get("isolated_slender_ambiguity"):
        return VegetationPlausibilityLabel.ISOLATED_SLENDER_AMBIGUOUS
    channels = [
        ("planar_structure_suspicion", VegetationPlausibilityLabel.PLANAR_STRUCTURE_SUSPECT),
        ("linear_structure_suspicion", VegetationPlausibilityLabel.LINEAR_STRUCTURE_SUSPECT),
        ("shallow_horizontal_suspicion", VegetationPlausibilityLabel.SHALLOW_HORIZONTAL_SUSPECT),
        ("low_shrub_evidence", VegetationPlausibilityLabel.LOW_SHRUB_LIKE),
        ("crown_like_evidence", VegetationPlausibilityLabel.VERTICALLY_DEVELOPED_CROWN_LIKE),
    ]
    key, label = max(channels, key=lambda p: (d[p[0]], -channels.index(p)))
    return (
        label
        if d[key] >= cfg.label_evidence_threshold
        else VegetationPlausibilityLabel.MIXED_OR_INSUFFICIENT_EVIDENCE
    )


def _flat_cluster_arrays(
    items: list[dict[str, Any]], descs: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    members = [d["member_rows"] for d in descs]
    offsets = np.zeros(len(members) + 1, np.int64)
    offsets[1:] = np.cumsum([len(v) for v in members])

    def arr(key: str, dtype: Any, default: Any = 0):
        return np.asarray([d.get(key, default) for d in descs], dtype=dtype)

    return {
        "cluster_id": arr("cluster_id", np.int32),
        "cluster_status": np.asarray(
            [VegetationClusterStatus[d["status"]].value for d in descs], np.uint8
        ),
        "cluster_vegetation_id": arr("vegetation_id", np.int32, -1),
        "cluster_parent_id": arr("automatic_parent_cluster_id", np.int32),
        "cluster_member_offsets": offsets,
        "ordered_cluster_member_row_index": np.concatenate(members).astype(np.int64)
        if members
        else np.empty(0, np.int64),
        "cluster_centroid_xyz": np.asarray([d["centroid_xyz"] for d in descs], np.float32).reshape(
            (-1, 3)
        ),
        "cluster_representative_center_xyz": np.asarray(
            [d["representative_center_xyz"] for d in descs], np.float32
        ).reshape((-1, 3)),
        "cluster_pca_dimensions_m": np.asarray(
            [d["pca_dimensions_m"] for d in descs], np.float32
        ).reshape((-1, 3)),
        "cluster_horizontal_dimensions_m": np.asarray(
            [[d["horizontal_length_m"], d["horizontal_width_m"]] for d in descs], np.float32
        ).reshape((-1, 2)),
        "cluster_vertical_extent_m": arr("vertical_extent_m", np.float32),
        "cluster_plausibility_score": arr("vegetation_plausibility_score", np.float32),
        "cluster_assessment_confidence": arr("assessment_confidence", np.float32),
        "cluster_fragmentation": arr("fragmentation", np.float32),
        "cluster_plausibility_label": np.asarray(
            [VegetationPlausibilityLabel[d["plausibility_label"]].value for d in descs], np.uint8
        ),
        "cluster_slice_valid": np.asarray([d["slice_valid"] for d in descs], np.uint8).reshape(
            (-1, 6)
        ),
        "cluster_slice_point_count": np.asarray(
            [d["slice_point_count"] for d in descs], np.int32
        ).reshape((-1, 6)),
        "cluster_slice_support_confidence": np.asarray(
            [d["slice_support_confidence"] for d in descs], np.float32
        ).reshape((-1, 6)),
        "cluster_slice_equivalent_radius_m": np.asarray(
            [d["slice_equivalent_radius_m"] for d in descs], np.float32
        ).reshape((-1, 6)),
        "cluster_slice_horizontal_dimensions_m": np.asarray(
            [d["slice_horizontal_dimensions_m"] for d in descs], np.float32
        ).reshape((-1, 6, 2)),
        "cluster_grouping_support_score": arr("grouping_support_score", np.float32),
        "cluster_vegetation_morphology_score": arr("vegetation_morphology_score", np.float32),
        "cluster_volumetric_score": arr("volumetric_score", np.float32),
        "cluster_profile_score": arr("profile_score", np.float32),
        "cluster_planar_ambiguity_score": arr("planar_ambiguity_score", np.float32),
        "cluster_isolated_slender_ambiguity_score": arr(
            "isolated_slender_ambiguity_score", np.float32
        ),
        "cluster_compact_layered_ambiguity_score": arr(
            "compact_layered_ambiguity_score", np.float32
        ),
        "cluster_mixed_morphology_conflict_score": arr(
            "mixed_morphology_conflict_score", np.float32
        ),
        "cluster_independent_positive_count": arr("independent_positive_count", np.uint8),
        "cluster_requires_review": np.asarray([d["requires_review"] for d in descs], np.bool_),
    }


def _validate_inputs(
    v1r: Any,
    v1: dict[str, np.ndarray],
    a2a: dict[str, np.ndarray],
    a2am: dict[str, Any],
    a2b: dict[str, np.ndarray],
    a2bm: dict[str, Any],
    hashes: dict[str, str],
    cfg: VegetationInstanceConfig,
) -> None:
    if (
        v1r.source_id != cfg.source_id
        or str(a2am.get("source_id")) != cfg.source_id
        or str(a2bm.get("source_id")) != cfg.source_id
    ):
        raise _error("V2_SOURCE", "source IDs are inconsistent")
    manifest = v1r.manifest_source_id
    if (
        str(a2am.get("manifest_source_id")) != manifest
        or str(a2bm.get("manifest_source_id")) != manifest
    ):
        raise _error("V2_MANIFEST", "manifest IDs are inconsistent")
    prov = v1r.input_provenance
    for name in ("a2a_npz", "a2a_json", "a2b_npz", "a2b_json"):
        if prov.get(name, {}).get("sha256") != hashes[name]:
            raise _error("V2_PROVENANCE", "V1 provenance does not match supplied input", input=name)
    if (
        a2am.get("output_npz_sha256") != hashes["a2a_npz"]
        or a2bm.get("output_npz_sha256") != hashes["a2b_npz"]
    ):
        raise _error("V2_HASH", "feature JSON does not match supplied NPZ")
    if not str(a2am.get("algorithm_version", "")).startswith("a2a-local-pca-") or not str(
        a2bm.get("algorithm_version", "")
    ).startswith("a2b-grid-quantile-"):
        raise _error("V2_VERSION", "unsupported A2 algorithm version")
    m = len(v1["original_index"])
    if (
        not np.array_equal(v1["xyz_local"], a2a.get("xyz_local"))
        or not np.array_equal(v1["original_index"], a2a.get("original_index"))
        or not np.array_equal(v1["sample_index"], a2a.get("sample_index"))
        or not np.array_equal(v1["original_index"], a2b.get("original_index"))
        or not np.array_equal(v1["sample_index"], a2b.get("sample_index"))
    ):
        raise _error("V2_ALIGNMENT", "V1/A2A/A2B mappings or V1/A2A coordinates differ")
    if (
        len(a2b["height_agl_m"]) != m
        or not np.array_equal(
            np.asarray(v1["origin_xyz"]), np.asarray(a2am.get("origin_xyz"), dtype=np.float64)
        )
        or not np.array_equal(
            np.asarray(v1["origin_xyz"]), np.asarray(a2bm.get("origin_xyz"), dtype=np.float64)
        )
    ):
        raise _error("V2_ALIGNMENT", "A2B length or origin differs")
    for name in ("linearity", "planarity", "scattering", "verticality", "local_density"):
        if (
            a2a.get(name) is None
            or a2a[name].dtype != np.float32
            or len(a2a[name]) != m
            or not np.isfinite(a2a[name]).all()
        ):
            raise _error("V2_A2A", "invalid A2A feature", array=name)
    for name in ("height_agl_m", "raw_height_agl_m", "ground_distance_m"):
        if a2b.get(name) is None or a2b[name].dtype != np.float32 or len(a2b[name]) != m:
            raise _error("V2_A2B", "invalid A2B array", array=name)
    if np.any((v1["wire_id_by_point"] >= 0) & v1["vegetation_mask"]):
        raise _error("V2_WIRE", "V1 confirmed wire must be rejected")


def _load_pair(
    npz: Path | str, json_path: Path | str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return _load_npz(Path(npz)), _load_json(Path(json_path))


def _validate_manual_log(
    log: VegetationManualLog, hashes: dict[str, str], source: str, manifest: str
) -> None:
    if (
        log.source_id != source
        or log.manifest_source_id != manifest
        or log.v1_npz_sha256 != hashes["v1_npz"]
        or log.a2a_npz_sha256 != hashes["a2a_npz"]
        or log.a2b_npz_sha256 != hashes["a2b_npz"]
    ):
        raise _error("V2_MANUAL_STALE", "manual log does not match verified inputs")


def _validate_output_arrays(data: dict[str, np.ndarray]) -> None:
    try:
        VegetationInstancePointSet(
            **{name: data[name] for name in VegetationInstancePointSet.__dataclass_fields__}
        )
    except (KeyError, ValueError) as exc:
        raise _error("V2_OUTPUT_ARRAY", "invalid V2 full-M arrays", cause=str(exc)) from exc
    if data.get("origin_xyz", np.empty(0)).shape != (3,) or data["origin_xyz"].dtype != np.float64:
        raise _error("V2_OUTPUT_ORIGIN", "invalid origin")
    offsets = data.get("cluster_member_offsets")
    members = data.get("ordered_cluster_member_row_index")
    if (
        offsets is None
        or members is None
        or offsets.dtype != np.int64
        or members.dtype != np.int64
        or len(offsets) < 1
        or offsets[0] != 0
        or offsets[-1] != len(members)
        or np.any(np.diff(offsets) < 0)
    ):
        raise _error("V2_OUTPUT_OFFSETS", "invalid cluster member offsets")
    if any(value.dtype == object for value in data.values()):
        raise _error("V2_OUTPUT_OBJECT", "object arrays are forbidden")


def _publish(
    arrays: dict[str, np.ndarray],
    clusters: list[dict[str, Any]],
    npz_out: Path,
    json_out: Path,
    cfg: VegetationInstanceConfig,
    log: VegetationManualLog,
    v1r: Any,
    hashes: dict[str, str],
    stats: VegetationInstanceStatistics,
    root: Path,
) -> VegetationInstanceResult:
    temp_npz, temp_json = _temporary(npz_out), _temporary(json_out)
    published = False
    try:
        np.savez_compressed(temp_npz, **arrays)
        _validate_output_arrays(_load_npz(temp_npz))
        provenance = {
            name: {"logical_path": name, "sha256": digest} for name, digest in hashes.items()
        }
        metadata = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        }
        result = VegetationInstanceResult(
            SCHEMA_VERSION,
            v1r.source_id,
            v1r.manifest_source_id,
            tuple(float(v) for v in arrays["origin_xyz"]),
            npz_out.name,
            _sha(temp_npz),
            temp_npz.stat().st_size,
            cfg,
            log,
            provenance,
            stats,
            metadata,
            tuple(_json_cluster(c) for c in clusters),
            (),
            ALGORITHM_VERSION,
            datetime.now(timezone.utc).isoformat(),
        )
        temp_json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        VegetationInstanceResult.from_dict(_load_json(temp_json))
        os.replace(temp_npz, npz_out)
        published = True
        os.replace(temp_json, json_out)
        return result
    except Exception:
        temp_npz.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)
        if published:
            npz_out.unlink(missing_ok=True)
        raise


def _json_cluster(value: dict[str, Any]) -> dict[str, Any]:
    def safe(item: Any) -> Any:
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, list):
            return [safe(v) for v in item]
        if isinstance(item, tuple):
            return [safe(v) for v in item]
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        return item

    return {k: safe(v) for k, v in value.items() if k not in {"member_rows"}}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as stream:
            data = {name: stream[name] for name in stream.files}
    except (OSError, ValueError) as exc:
        raise _error("V2_NPZ", "unable to load NPZ", path=str(path), cause=str(exc)) from exc
    if any(v.dtype == object for v in data.values()):
        raise _error("V2_NPZ_OBJECT", "object arrays are forbidden", path=str(path))
    return data


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("V2_JSON", "unable to load JSON", path=str(path), cause=str(exc)) from exc
    if not isinstance(data, dict):
        raise _error("V2_JSON", "JSON must be an object")
    return data


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _even(values: np.ndarray, limit: int) -> np.ndarray:
    return (
        values
        if len(values) <= limit
        else values[np.linspace(0, len(values) - 1, limit, dtype=np.int64)]
    )


def _validate_output_dir(output: Path, allowed: Path) -> None:
    try:
        output.resolve().relative_to(allowed.resolve())
    except ValueError as exc:
        raise _error(
            "V2_OUTPUT_FORBIDDEN", "output must remain in approved directory", output=str(output)
        ) from exc


def _validate_outputs(paths: tuple[Path, Path], overwrite: bool) -> None:
    existing = [str(v) for v in paths if v.exists()]
    if existing and not overwrite:
        raise _error("V2_OUTPUT_EXISTS", "output exists and overwrite is disabled", paths=existing)


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _relative_or_name(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _current_process() -> Any | None:
    try:
        import psutil

        return psutil.Process()
    except ImportError:
        return None


def _rss(process: Any | None) -> int | None:
    try:
        return int(process.memory_info().rss) if process is not None else None
    except Exception:
        return None


def _check_cancelled(token: CancellationToken | None, stage: str) -> None:
    if token is not None and token.is_cancelled():
        raise _error("V2_CANCELLED", "V2 operation was cancelled", stage=stage)


def _report(
    reporter: ProgressReporter | None,
    task: str,
    stage: str,
    processed: int,
    total: int,
    status: str = "running",
) -> None:
    if reporter is not None:
        reporter.update(
            PipelineProgress(
                task_id=task,
                stage=stage,
                status=status,
                processed_points=processed,
                total_points=total,
                percent=100 * processed / max(total, 1),
            )
        )


def _error(code: str, message: str, **details: Any) -> VegetationInstanceError:
    cause = details.pop("cause", None)
    return VegetationInstanceError(
        UnifiedError(
            code=code,
            message=message,
            severity="error",
            module="vegetation.instances",
            recoverable=False,
            details=details,
            cause=cause,
        )
    )
