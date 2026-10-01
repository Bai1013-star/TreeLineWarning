"""Read-only Round-2 model-guided Wire endpoint evidence experiment.

This script consumes persisted real representative evidence.  It does not invoke
the production pipeline, alter confirmation thresholds, or synthesize identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_wire_gap_rescue_round1 import SPAN_DIRS, load, model, segment

FOCUS = {"S001": (8, 30), "S002": (41,), "S003": (38,)}
MODEL_RADIAL_MAX_M = 0.08
COMPONENT_MAX_M = 0.06
SEARCH_MAX_EXTENSION_M = 5.0
MIN_GROUP_REPRESENTATIVES = 3
MIN_GROUP_LONGITUDINAL_SPAN_M = 0.20
MAX_LOCAL_GAP_M = 0.25


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _span_length(track: dict) -> float:
    lo, hi = track["supported_s_domain_m"]
    return (hi - lo) / max(float(track["coverage_ratio"]), 1e-12)


def _trigger(track: dict) -> tuple[bool, list[str]]:
    maximum_gap = track.get("maximum_gap_m")
    median_residual = track.get("median_residual_m")
    p95_residual = track.get("p95_residual_m")
    checks = {
        "review_only": track["status"] == "REVIEW_ONLY",
        "catenary": track["model_type"] == "CATENARY",
        "endpoint_only_reason": track["reasons"]
        == ["weak_endpoint_requires_stronger_interior_coverage"],
        "one_endpoint_missing": bool(track["approach_start"]) ^ bool(track["approach_end"]),
        "substantial_interior": track["coverage_ratio"] >= 0.75,
        "supported_slices": track["unique_supported_slice_count"] >= 8,
        "small_internal_gap": maximum_gap is not None and maximum_gap <= 5.0,
        "low_median_residual": median_residual is not None and median_residual <= 0.10,
        "low_p95_residual": p95_residual is not None and p95_residual <= 0.225,
    }
    return all(checks.values()), [key for key, ok in checks.items() if not ok]


def _memberships(data: dict) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for track_id in range(len(data["tracks"])):
        for row in segment(data["evidence"], "inlier_representative_index", track_id).astype(int):
            result.setdefault(int(row), set()).add(track_id)
    return result


def recover(data: dict, track_id: int) -> dict:
    track = data["tracks"][track_id]
    triggered, trigger_failures = _trigger(track)
    result = {
        "track_id": track_id,
        "baseline_status": track["status"],
        "triggered": triggered,
        "trigger_failures": trigger_failures,
        "found": 0,
        "accepted": 0,
        "rejected": 0,
        "accepted_rows": [],
        "rejected_rows": [],
        "new_status": track["status"],
    }
    if not triggered:
        return result

    diag, evidence = data["diag"], data["evidence"]
    span_length = _span_length(track)
    lo, hi = map(float, track["supported_s_domain_m"])
    missing_start = not track["approach_start"]
    edge = lo if missing_start else hi
    region_lo = (
        max(-SEARCH_MAX_EXTENSION_M, edge - SEARCH_MAX_EXTENSION_M) if missing_start else edge
    )
    region_hi = (
        edge
        if missing_start
        else min(span_length + SEARCH_MAX_EXTENSION_M, edge + SEARCH_MAX_EXTENSION_M)
    )
    candidate_rows = np.flatnonzero(
        diag["candidate_mask"] & (diag["s_m"] >= region_lo) & (diag["s_m"] <= region_hi)
    )
    s = diag["s_m"][candidate_rows]
    expected_u, expected_z = model(evidence, track_id, s)
    du = np.abs(diag["u_m"][candidate_rows] - expected_u)
    dz = np.abs(diag["z_m"][candidate_rows] - expected_z)
    radial = np.hypot(du, dz)
    memberships = _memberships(data)
    target_clusters = set(
        int(value)
        for value in diag["cluster_id_by_rep"][
            segment(evidence, "inlier_representative_index", track_id).astype(int)
        ]
        if value >= 0
    )
    prelim = (
        (radial <= MODEL_RADIAL_MAX_M)
        & (du <= COMPONENT_MAX_M)
        & (dz <= COMPONENT_MAX_M)
        & (diag["raw_count"][candidate_rows] >= 2)
        & np.asarray(
            [int(diag["cluster_id_by_rep"][row]) in target_clusters for row in candidate_rows]
        )
        & np.asarray(
            [not (memberships.get(int(row), set()) - {track_id}) for row in candidate_rows]
        )
    )
    prelim_rows = candidate_rows[prelim]
    # Local direction/support gate: accept only a coherent longitudinal run.
    coherent = False
    direction_cosine = None
    if len(prelim_rows) >= MIN_GROUP_REPRESENTATIVES:
        order = prelim_rows[np.argsort(diag["s_m"][prelim_rows])]
        ss = diag["s_m"][order]
        suz = np.column_stack((ss, diag["u_m"][order], diag["z_m"][order]))
        tangent_s = float(np.median(ss))
        eps = 0.02
        u1, z1 = model(evidence, track_id, np.asarray([tangent_s - eps, tangent_s + eps]))
        expected_direction = np.asarray([2 * eps, u1[1] - u1[0], z1[1] - z1[0]])
        actual_direction = np.linalg.svd(suz - suz.mean(axis=0), full_matrices=False)[2][0]
        direction_cosine = float(
            abs(actual_direction @ expected_direction)
            / (np.linalg.norm(actual_direction) * np.linalg.norm(expected_direction))
        )
        coherent = (
            float(np.ptp(ss)) >= MIN_GROUP_LONGITUDINAL_SPAN_M
            and float(np.max(np.diff(ss))) <= MAX_LOCAL_GAP_M
            and direction_cosine >= 0.90
        )
    accepted_rows = prelim_rows if coherent else np.empty(0, dtype=int)
    accepted_set = set(map(int, accepted_rows))
    rejected_rows = np.asarray([row for row in candidate_rows if int(row) not in accepted_set])

    all_target_rows = segment(evidence, "inlier_representative_index", track_id).astype(int)
    recovered_domain_rows = np.unique(np.concatenate((all_target_rows, accepted_rows)))
    recovered_s = diag["s_m"][recovered_domain_rows]
    after_coverage = float(np.ptp(recovered_s) / span_length)
    after_start = bool(track["approach_start"]) or bool(np.min(recovered_s) <= 0.0)
    after_end = bool(track["approach_end"]) or bool(np.max(recovered_s) >= span_length - 5.0)
    existing_confirmation_passes = (
        after_coverage >= 0.80 and after_start and after_end and not track["reasons"][:-1]
    )
    result.update(
        {
            "search_region_m": [region_lo, region_hi],
            "found": int(len(candidate_rows)),
            "accepted": int(len(accepted_rows)),
            "rejected": int(len(rejected_rows)),
            "accepted_rows": accepted_rows.tolist(),
            "rejected_rows": rejected_rows.tolist(),
            "accepted_original_index": diag["fine_original_index"][accepted_rows]
            .astype(int)
            .tolist(),
            "accepted_already_target_inliers": all(
                track_id in memberships.get(int(row), set()) for row in accepted_rows
            ),
            "model_distance_m": radial[prelim].tolist(),
            "transverse_distance_m": du[prelim].tolist(),
            "height_distance_m": dz[prelim].tolist(),
            "direction_cosine": direction_cosine,
            "target_cluster_consistency": True if len(accepted_rows) else None,
            "incompatible_track_conflicts": 0,
            "before_coverage": track["coverage_ratio"],
            "after_coverage": after_coverage,
            "before_endpoint_flags": [track["approach_start"], track["approach_end"]],
            "after_endpoint_flags": [after_start, after_end],
            "supported_slices_after": len(np.unique(np.floor(recovered_s).astype(int))),
            "new_status": "CONFIRMED" if existing_confirmation_passes else track["status"],
            "real_original_index_only": True,
        }
    )
    return result


def _plot(data: dict, recovery: dict, path: Path, title: str) -> None:
    diag, evidence = data["diag"], data["evidence"]
    tid = recovery["track_id"]
    rows = segment(evidence, "inlier_representative_index", tid).astype(int)
    candidates = np.flatnonzero(diag["candidate_mask"])
    accepted = np.asarray(recovery.get("accepted_rows", []), dtype=int)
    rejected = np.asarray(recovery.get("rejected_rows", []), dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, key, ylabel in zip(
        axes, ("u_m", "z_m"), ("transverse u (m)", "height z (m)"), strict=True
    ):
        ax.scatter(diag["s_m"][candidates], diag[key][candidates], s=3, c="#b8b8b8", alpha=0.22)
        ax.scatter(
            diag["s_m"][rows], diag[key][rows], s=12, c="#2563eb", label="existing real inliers"
        )
        if len(rejected):
            ax.scatter(
                diag["s_m"][rejected],
                diag[key][rejected],
                s=15,
                c="#dc2626",
                alpha=0.5,
                label="rejected",
            )
        if len(accepted):
            ax.scatter(
                diag["s_m"][accepted],
                diag[key][accepted],
                s=35,
                facecolors="none",
                edgecolors="#16a34a",
                linewidths=1.5,
                label="accepted endpoint evidence",
            )
        if "search_region_m" in recovery:
            ax.axvspan(
                *recovery["search_region_m"], color="#f59e0b", alpha=0.12, label="bounded search"
            )
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="best", markerscale=1.5)
    axes[1].set_xlabel("longitudinal s (m)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spans = {name: load(args.project, name) for name in SPAN_DIRS}
    focus = {
        name: {tid: recover(data, tid) for tid in tids}
        for name, (data, tids) in ((n, (spans[n], FOCUS[n])) for n in spans)
    }
    review_cases = []
    for span_name, data in spans.items():
        for track in data["tracks"]:
            if track["status"] != "CONFIRMED":
                outcome = recover(data, track["track_id"])
                review_cases.append(
                    {
                        k: outcome[k]
                        for k in (
                            "track_id",
                            "baseline_status",
                            "triggered",
                            "found",
                            "accepted",
                            "new_status",
                        )
                    }
                    | {"span": span_name}
                )
    s3 = focus["S003"][38]
    report = {
        "schema": "wire-endpoint-recovery-round2-diagnostic-v1",
        "production_pipeline_invoked": False,
        "threshold_changed": False,
        "synthetic_points_created": False,
        "mechanism": {
            "trigger": (
                "narrow high-quality catenary review-only track with sole weak-endpoint reason"
            ),
            "search": (
                "missing-end direction, stored support edge to at most 5 m, clipped to span halo"
            ),
            "constraints": {
                "radial_m": MODEL_RADIAL_MAX_M,
                "component_m": COMPONENT_MAX_M,
                "min_raw_support": 2,
                "direction_cosine": 0.90,
            },
            "conflicts": (
                "existing Wire candidate + target cluster + no incompatible-track ownership + "
                "coherent local run"
            ),
        },
        "focus": focus,
        "negative_review_cases": review_cases,
        "new_false_confirmations": sum(
            c["new_status"] == "CONFIRMED" and not (c["span"] == "S003" and c["track_id"] == 38)
            for c in review_cases
        ),
        "s003_real_evidence": "CONFIRMED"
        if s3["accepted"] >= 3 and s3.get("real_original_index_only")
        else "NOT_CONFIRMED",
        "production_gate": "MORE_EVIDENCE_REQUIRED",
        "gate_reason": (
            "Persisted diagnostics lack independent vegetation labels and tower-structure "
            "identities; synthetic guards can bound but cannot close real false-positive gates."
        ),
    }
    (args.output / "round2_results.json").write_text(
        json.dumps(_safe(report), indent=2), encoding="utf-8"
    )
    _plot(
        spans["S003"],
        focus["S003"][38],
        args.output / "S003_ENDPOINT_RECOVERY.png",
        "S003 Track 38: bounded endpoint evidence recovery",
    )
    _plot(
        spans["S002"],
        focus["S002"][41],
        args.output / "S002_CONTROL.png",
        "S002 Track 41: healthy control (recovery not triggered)",
    )
    _plot(
        spans["S001"],
        focus["S001"][8],
        args.output / "S001_NEGATIVE_TRACK8.png",
        "S001 Track 8: negative guard (recovery not triggered)",
    )
    print(json.dumps(_safe(report), indent=2))


if __name__ == "__main__":
    main()
