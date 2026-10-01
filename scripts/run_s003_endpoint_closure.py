"""Focused, read-only S003/T004 endpoint evidence visualization.

The diagnostic reads only Shared Source tiles intersecting the final S003
endpoint corridor.  Every plotted observation retains its real source_index;
the fitted catenary is rendered only as a model reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = "analysis-1c8924464e3a4c06bd2c8857a6c2c8ef"
SPAN_DIR = "span-e6eade9ed8c54a4099391b757c0704ec"
SHARED_SOURCE = "e5cfeaff81904abdfab1a06d3716ed23b8cc4fe6b845ebfa4781a72fb10b5fba"
TRACK_ID = 38
DTYPE = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("source_index", "<i8"),
        ("classification", "u1"),
    ]
)


def segment(arrays: np.lib.npyio.NpzFile, stem: str, track_id: int) -> np.ndarray:
    offsets = arrays[f"{stem}_offsets"]
    return arrays[f"{stem}_flat"][offsets[track_id] : offsets[track_id + 1]]


def model(
    evidence: np.lib.npyio.NpzFile, track_id: int, s_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u_coefficients = segment(evidence, "model_u_coefficients", track_id)
    catenary = segment(evidence, "catenary_parameters", track_id)
    z_coefficients = segment(evidence, "model_z_coefficients", track_id)
    u_m = np.polyval(
        u_coefficients[2:], (s_m - u_coefficients[0]) / u_coefficients[1]
    )
    if len(catenary) == 3:
        c_m, log_a, s0_m = catenary
        a_m = np.exp(log_a)
        z_m = c_m + a_m * (
            np.cosh(np.clip((s_m - s0_m) / a_m, -10.0, 10.0)) - 1.0
        )
    else:
        z_m = np.polyval(
            z_coefficients[2:], (s_m - z_coefficients[0]) / z_coefficients[1]
        )
    return u_m, z_m


def _load_intersecting_tiles(
    tile_root: Path, xy_bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, list[str], int]:
    index = json.loads((tile_root / "tiles" / "index.json").read_text("utf-8"))
    xmin, xmax, ymin, ymax = xy_bounds
    selected: list[tuple[str, dict]] = []
    for tile_id, item in index["tiles"].items():
        lo = item["min_xyz"]
        hi = item["max_xyz"]
        if hi[0] >= xmin and lo[0] <= xmax and hi[1] >= ymin and lo[1] <= ymax:
            selected.append((tile_id, item))
    chunks = [
        np.fromfile(tile_root / "tiles" / item["file"], dtype=DTYPE)
        for _, item in selected
    ]
    points = np.concatenate(chunks) if chunks else np.empty(0, dtype=DTYPE)
    return points, [tile_id for tile_id, _ in selected], int(sum(len(c) for c in chunks))


def _frame(
    xy: np.ndarray, start_xy: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    relative = xy - start_xy
    normal = np.asarray([-direction[1], direction[0]])
    return relative @ direction, relative @ normal


def _bounded_sample(rows: np.ndarray, maximum: int) -> np.ndarray:
    if len(rows) <= maximum:
        return rows
    return rows[np.linspace(0, len(rows) - 1, maximum, dtype=int)]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d6dbe2", linewidth=0.5, alpha=0.65)
    ax.set_facecolor("#fbfcfe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--round2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stage_a = json.loads((args.project / "stage-a-project.json").read_text("utf-8"))
    towers = {tower["display_name"]: tower for tower in stage_a["project"]["towers"]}
    start = np.asarray(towers["T003"]["position"], dtype=float)
    end = np.asarray(towers["T004"]["position"], dtype=float)
    delta_xy = end[:2] - start[:2]
    span_length_m = float(np.linalg.norm(delta_xy))
    direction = delta_xy / span_length_m
    normal = np.asarray([-direction[1], direction[0]])

    wire_root = (
        args.project
        / "artifacts"
        / "project-runs"
        / "batches"
        / ANALYSIS
        / SPAN_DIR
        / "wire"
    )
    manifest = json.loads((wire_root / "wire_v2_track_evidence.json").read_text("utf-8"))
    track = manifest["tracks"][TRACK_ID]
    evidence = np.load(wire_root / "wire_v2_track_evidence.npz")
    diagnostics = np.load(wire_root / "span_wire_diagnostics.npz")
    round2 = json.loads(args.round2.read_text("utf-8"))["focus"]["S003"][str(TRACK_ID)]

    support_end_m = float(track["supported_s_domain_m"][1])
    s_min_m = max(0.0, support_end_m - 2.0)
    s_max_m = span_length_m + 1.0
    u_min_m, u_max_m = -9.0, 4.0
    corners = np.asarray(
        [
            start[:2] + s * direction + u * normal
            for s in (s_min_m, s_max_m)
            for u in (u_min_m, u_max_m)
        ]
    )
    xy_bounds = (
        float(corners[:, 0].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].min()),
        float(corners[:, 1].max()),
    )

    shared_root = (
        args.project / "artifacts" / "project-runs" / "source-shared" / SHARED_SOURCE
    )
    preparation = json.loads((shared_root / "preparation.json").read_text("utf-8"))
    all_source_root = shared_root / "vegetation-role-tile-index-high-recall-v3"
    wire_role_root = args.project / preparation["wire_role_tile_index_reference"]
    all_source, source_tiles, local_points_read = _load_intersecting_tiles(
        all_source_root, xy_bounds
    )
    wire_role, wire_tiles, wire_points_read = _load_intersecting_tiles(wire_role_root, xy_bounds)

    source_xy = np.column_stack((all_source["x"], all_source["y"]))
    source_s, source_u = _frame(source_xy, start[:2], direction)
    local_mask = (
        (source_s >= s_min_m)
        & (source_s <= s_max_m)
        & (source_u >= u_min_m)
        & (source_u <= u_max_m)
        & (all_source["z"] >= 4.5)
        & (all_source["z"] <= 12.0)
    )
    source = all_source[local_mask]
    source_s = source_s[local_mask]
    source_u = source_u[local_mask]
    source_z = source["z"].astype(float)

    expected_u, expected_z = model(evidence, TRACK_ID, source_s)
    model_radial = np.hypot(source_u - expected_u, source_z - expected_z)
    accepted_max_s = float(
        np.max(diagnostics["s_m"][np.asarray(round2["accepted_rows"], dtype=int)])
    )
    terminal_model = (
        (source_s >= support_end_m)
        & (source_s <= support_end_m + 1.0)
        & (model_radial <= 0.08)
    )
    terminal_model_max_s = float(np.max(source_s[terminal_model]))
    observation_gap = (
        (source_s > terminal_model_max_s + 1e-9)
        & (source_s < span_length_m - 1.0)
    )
    strict_model = observation_gap & (model_radial <= 0.08)
    broad_model = observation_gap & (model_radial <= 0.25)

    candidate_rows = np.flatnonzero(
        diagnostics["candidate_mask"]
        & (diagnostics["s_m"] >= s_min_m)
        & (diagnostics["s_m"] <= s_max_m)
        & (diagnostics["u_m"] >= u_min_m)
        & (diagnostics["u_m"] <= u_max_m)
        & (diagnostics["z_m"] >= 4.5)
        & (diagnostics["z_m"] <= 12.0)
    )
    target_rows = segment(evidence, "inlier_representative_index", TRACK_ID).astype(int)
    target_rows = target_rows[
        (diagnostics["s_m"][target_rows] >= s_min_m)
        & (diagnostics["s_m"][target_rows] <= s_max_m)
    ]
    accepted_rows = np.asarray(round2["accepted_rows"], dtype=int)
    rejected_rows = np.asarray(round2["rejected_rows"], dtype=int)
    rejected_rows = rejected_rows[
        (diagnostics["s_m"][rejected_rows] >= s_min_m)
        & (diagnostics["s_m"][rejected_rows] <= s_max_m)
        & (diagnostics["u_m"][rejected_rows] >= u_min_m)
        & (diagnostics["u_m"][rejected_rows] <= u_max_m)
    ]

    incompatible_rows: list[np.ndarray] = []
    incompatible_track_ids: list[int] = []
    for track_id in range(len(manifest["tracks"])):
        if track_id == TRACK_ID:
            continue
        rows = segment(evidence, "inlier_representative_index", track_id).astype(int)
        rows = rows[
            (diagnostics["s_m"][rows] >= s_min_m)
            & (diagnostics["s_m"][rows] <= s_max_m)
            & (diagnostics["u_m"][rows] >= u_min_m)
            & (diagnostics["u_m"][rows] <= u_max_m)
            & (diagnostics["z_m"][rows] >= 4.5)
            & (diagnostics["z_m"][rows] <= 12.0)
        ]
        if len(rows):
            incompatible_rows.append(rows)
            incompatible_track_ids.append(track_id)
    incompatible = (
        np.unique(np.concatenate(incompatible_rows)) if incompatible_rows else np.empty(0, int)
    )

    plot_rows = _bounded_sample(np.arange(len(source)), 180_000)
    s_curve = np.linspace(0.0, span_length_m, 600)
    u_curve, z_curve = model(evidence, TRACK_ID, s_curve)
    round2_to_terminal = (
        (source_s > accepted_max_s)
        & (source_s <= terminal_model_max_s)
        & (model_radial <= 0.08)
    )
    terminal_extra_details: list[dict] = []
    source_xyz = np.column_stack((source["x"], source["y"], source["z"]))
    for row in np.flatnonzero(round2_to_terminal):
        distance = np.linalg.norm(source_xyz - source_xyz[row], axis=1)
        terminal_extra_details.append(
            {
                "source_index": int(source["source_index"][row]),
                "s_m": float(source_s[row]),
                "u_m": float(source_u[row]),
                "z_m": float(source_z[row]),
                "model_radial_m": float(model_radial[row]),
                "neighbor_count_within_0_10m": int(np.count_nonzero(distance <= 0.10) - 1),
                "interpretation": "same terminal observation cluster",
            }
        )

    def add_top_layers(ax: plt.Axes) -> None:
        ax.scatter(source_s[plot_rows], source_u[plot_rows], s=0.45, c="#aeb7c4", alpha=0.10)
        ax.scatter(
            diagnostics["s_m"][candidate_rows], diagnostics["u_m"][candidate_rows],
            s=9, c="#7b8797", alpha=0.35, label="all endpoint candidate representatives"
        )
        if len(rejected_rows):
            ax.scatter(
                diagnostics["s_m"][rejected_rows], diagnostics["u_m"][rejected_rows],
                s=18, marker="x", c="#d94841", alpha=0.72, label="Round-2 rejected"
            )
        if len(incompatible):
            ax.scatter(
                diagnostics["s_m"][incompatible], diagnostics["u_m"][incompatible],
                s=28, marker="^", facecolors="none", edgecolors="#e28b16",
                linewidths=0.9, label=f"incompatible tracks {incompatible_track_ids}"
            )
        ax.scatter(
            diagnostics["s_m"][target_rows], diagnostics["u_m"][target_rows],
            s=16, c="#2066d1", label="Track 38 observed real members"
        )
        ax.scatter(
            diagnostics["s_m"][accepted_rows], diagnostics["u_m"][accepted_rows],
            s=62, facecolors="none", edgecolors="#1c9b52", linewidths=1.8,
            label="Round-2 six accepted representatives"
        )
        ax.plot(s_curve, u_curve, color="#6d46c7", linewidth=2.0, label="fitted Track 38 catenary")
        ax.plot(
            s_curve[s_curve >= support_end_m], u_curve[s_curve >= support_end_m],
            color="#6d46c7", linewidth=2.0, linestyle="--", label="predicted continuation only"
        )
        ax.axvline(support_end_m, color="#1457a6", linestyle=":", linewidth=1.8,
                   label="current support endpoint")
        ax.axvspan(span_length_m - 1.0, span_length_m + 0.35, color="#9b6bc1", alpha=0.10,
                   label="T004 structure zone")
        ax.scatter([span_length_m], [0.0], marker="*", s=180, c="#6f268a",
                   edgecolors="#3e0d50", linewidths=0.8, label="T004 base position")
        ax.set_xlim(s_min_m, s_max_m)
        ax.set_ylim(u_min_m, u_max_m)
        ax.set_xlabel("longitudinal s from T003 (m)")
        ax.set_ylabel("transverse u (m)")
        _style_axis(ax)

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    add_top_layers(ax)
    ax.set_title("S003_ENDPOINT_CLOSURE_TOP — real source evidence toward T004")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output / "S003_ENDPOINT_CLOSURE_TOP.png", dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.scatter(source_s[plot_rows], source_z[plot_rows], s=0.45, c="#aeb7c4", alpha=0.10)
    ax.scatter(
        diagnostics["s_m"][candidate_rows], diagnostics["z_m"][candidate_rows],
        s=9, c="#7b8797", alpha=0.35, label="all endpoint candidate representatives"
    )
    if len(rejected_rows):
        ax.scatter(
            diagnostics["s_m"][rejected_rows], diagnostics["z_m"][rejected_rows],
            s=18, marker="x", c="#d94841", alpha=0.72, label="Round-2 rejected"
        )
    if len(incompatible):
        ax.scatter(
            diagnostics["s_m"][incompatible], diagnostics["z_m"][incompatible],
            s=28, marker="^", facecolors="none", edgecolors="#e28b16",
            linewidths=0.9, label="incompatible tracks"
        )
    ax.scatter(
        diagnostics["s_m"][target_rows], diagnostics["z_m"][target_rows],
        s=16, c="#2066d1", label="Track 38 observed real members"
    )
    ax.scatter(
        diagnostics["s_m"][accepted_rows], diagnostics["z_m"][accepted_rows],
        s=62, facecolors="none", edgecolors="#1c9b52", linewidths=1.8,
        label="Round-2 six accepted representatives"
    )
    ax.plot(s_curve, z_curve, color="#6d46c7", linewidth=2.0, label="fitted Track 38 catenary")
    ax.plot(
        s_curve[s_curve >= support_end_m], z_curve[s_curve >= support_end_m],
        color="#6d46c7", linewidth=2.0, linestyle="--", label="predicted continuation only"
    )
    ax.axvline(support_end_m, color="#1457a6", linestyle=":", linewidth=1.8,
               label="current support endpoint")
    ax.plot([span_length_m, span_length_m], [end[2], towers["T004"]["attributes"]["height_m"]],
            color="#6f268a", linewidth=4.0, alpha=0.8, label="T004 vertical extent")
    ax.scatter([span_length_m], [end[2]], marker="*", s=180, c="#6f268a", edgecolors="#3e0d50")
    ax.set_xlim(s_min_m, s_max_m)
    ax.set_ylim(4.5, 12.0)
    ax.set_xlabel("longitudinal s from T003 (m)")
    ax.set_ylabel("height z (m)")
    ax.set_title("S003_ENDPOINT_CLOSURE_SIDE — observed support, model, and T004")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(args.output / "S003_ENDPOINT_CLOSURE_SIDE.png", dpi=210)
    plt.close(fig)

    fig = plt.figure(figsize=(13.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    rows_3d = _bounded_sample(np.arange(len(source)), 90_000)
    ax.scatter(source_s[rows_3d], source_u[rows_3d], source_z[rows_3d],
               s=0.35, c="#aeb7c4", alpha=0.08)
    ax.scatter(
        diagnostics["s_m"][candidate_rows], diagnostics["u_m"][candidate_rows],
        diagnostics["z_m"][candidate_rows], s=8, c="#7b8797", alpha=0.30
    )
    if len(rejected_rows):
        ax.scatter(
            diagnostics["s_m"][rejected_rows], diagnostics["u_m"][rejected_rows],
            diagnostics["z_m"][rejected_rows], s=16, marker="x", c="#d94841", alpha=0.65
        )
    if len(incompatible):
        ax.scatter(
            diagnostics["s_m"][incompatible], diagnostics["u_m"][incompatible],
            diagnostics["z_m"][incompatible], s=25, marker="^", facecolors="none",
            edgecolors="#e28b16", linewidths=0.8
        )
    ax.scatter(
        diagnostics["s_m"][target_rows], diagnostics["u_m"][target_rows],
        diagnostics["z_m"][target_rows], s=14, c="#2066d1"
    )
    ax.scatter(
        diagnostics["s_m"][accepted_rows], diagnostics["u_m"][accepted_rows],
        diagnostics["z_m"][accepted_rows], s=58, facecolors="none", edgecolors="#1c9b52",
        linewidths=1.7
    )
    ax.plot(s_curve, u_curve, z_curve, color="#6d46c7", linewidth=2.3)
    ax.plot([span_length_m, span_length_m], [0.0, 0.0],
            [end[2], towers["T004"]["attributes"]["height_m"]], color="#6f268a", linewidth=4)
    ax.scatter([span_length_m], [0.0], [end[2]], marker="*", s=160, c="#6f268a")
    ax.set_xlim(s_min_m, s_max_m)
    ax.set_ylim(u_min_m, u_max_m)
    ax.set_zlim(4.5, 12.0)
    ax.set_xlabel("s (m)")
    ax.set_ylabel("u (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=24, azim=-58)
    ax.set_title("S003_ENDPOINT_CLOSURE_3D — full-resolution local source context")
    fig.tight_layout()
    fig.savefig(args.output / "S003_ENDPOINT_CLOSURE_3D.png", dpi=210)
    plt.close(fig)

    model_bins: list[dict] = []
    bin_edges = np.arange(np.floor(accepted_max_s * 4) / 4, span_length_m + 0.2501, 0.25)
    for lo_m, hi_m in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        rows = (source_s >= lo_m) & (source_s < hi_m)
        model_bins.append(
            {
                "s_interval_m": [float(lo_m), float(hi_m)],
                "all_height_points": int(np.count_nonzero(rows)),
                "within_0_08m_of_model": int(np.count_nonzero(rows & (model_radial <= 0.08))),
                "within_0_25m_of_model": int(np.count_nonzero(rows & (model_radial <= 0.25))),
            }
        )

    report = {
        "schema": "s003-endpoint-autonomous-closure-diagnostic-v1",
        "source_identity_semantics": "REAL Shared Source source_index only",
        "full_source_rescan": False,
        "local_region": {
            "s_m": [s_min_m, s_max_m],
            "u_m": [u_min_m, u_max_m],
            "z_m": [4.5, 12.0],
            "xy_bounds_m": list(xy_bounds),
        },
        "source_tiles": source_tiles,
        "local_points_read": local_points_read,
        "local_height_points_retained": int(len(source)),
        "wire_role_tiles": wire_tiles,
        "wire_role_points_read": wire_points_read,
        "span_length_m": span_length_m,
        "support_end_m": support_end_m,
        "round2_accepted_max_s_m": accepted_max_s,
        "unresolved_after_round2_m": span_length_m - accepted_max_s,
        "observed_terminal_cluster_max_s_m": terminal_model_max_s,
        "observation_gap_before_tower_m": (span_length_m - 1.0) - terminal_model_max_s,
        "track_38": track,
        "round2": {
            "accepted_rows": accepted_rows.tolist(),
            "accepted_original_index": round2["accepted_original_index"],
            "accepted_already_target_inliers": round2["accepted_already_target_inliers"],
            "after_coverage": round2["after_coverage"],
            "after_endpoint_flags": round2["after_endpoint_flags"],
        },
        "endpoint_candidates": {
            "all_representatives": int(len(candidate_rows)),
            "round2_rejected": int(len(rejected_rows)),
            "incompatible_track_ids": incompatible_track_ids,
            "incompatible_representatives": int(len(incompatible)),
        },
        "round2_to_observed_terminal_cluster": {
            "real_points_within_0_08m": int(np.count_nonzero(round2_to_terminal)),
            "details": terminal_extra_details,
        },
        "unresolved_model_tube_before_tower": {
            "s_m": [terminal_model_max_s, span_length_m - 1.0],
            "real_points_within_0_08m": int(np.count_nonzero(strict_model)),
            "real_points_within_0_25m": int(np.count_nonzero(broad_model)),
            "source_indices_within_0_08m": source["source_index"][strict_model]
            .astype(int)
            .tolist(),
        },
        "classification_distribution": {
            str(int(value)): int(count)
            for value, count in zip(
                *np.unique(source["classification"], return_counts=True), strict=True
            )
        },
        "model_tube_quarter_metre_bins": model_bins,
        "figures": {
            "top": str(args.output / "S003_ENDPOINT_CLOSURE_TOP.png"),
            "side": str(args.output / "S003_ENDPOINT_CLOSURE_SIDE.png"),
            "view3d": str(args.output / "S003_ENDPOINT_CLOSURE_3D.png"),
        },
    }
    (args.output / "S003_ENDPOINT_CLOSURE.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), "utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
