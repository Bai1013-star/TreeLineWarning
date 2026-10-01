"""Read-only Round-1 Wire gap-rescue experiment on persisted V2 evidence.

This diagnostic never invokes the production pipeline and never synthesizes source
indices.  Outputs are written outside the accepted project workspace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANALYSIS = "analysis-1c8924464e3a4c06bd2c8857a6c2c8ef"
SPAN_DIRS = {
    "S001": "span-244a315f53e44258b464c31b6006a9ea",
    "S002": "span-f8afe381236d4779afd1392328a2f21c",
    "S003": "span-e6eade9ed8c54a4099391b757c0704ec",
}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def segment(z: np.lib.npyio.NpzFile, stem: str, track_id: int) -> np.ndarray:
    offsets = z[f"{stem}_offsets"]
    return z[f"{stem}_flat"][offsets[track_id] : offsets[track_id + 1]]


def model(z: np.lib.npyio.NpzFile, track_id: int, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uc = segment(z, "model_u_coefficients", track_id)
    cat = segment(z, "catenary_parameters", track_id)
    zc = segment(z, "model_z_coefficients", track_id)
    u = np.polyval(uc[2:], (s - uc[0]) / uc[1])
    if len(cat) == 3:
        c, log_a, s0 = cat
        a = np.exp(log_a)
        height = c + a * (np.cosh(np.clip((s - s0) / a, -10, 10)) - 1)
    else:
        height = np.polyval(zc[2:], (s - zc[0]) / zc[1])
    return u, height


def load(project: Path, span: str) -> dict:
    root = project / "artifacts/project-runs/batches" / ANALYSIS / SPAN_DIRS[span] / "wire"
    manifest = json.loads((root / "wire_v2_track_evidence.json").read_text("utf-8"))
    return {
        "root": root,
        "manifest": manifest,
        "tracks": manifest["tracks"],
        "evidence": np.load(root / "wire_v2_track_evidence.npz"),
        "diag": np.load(root / "span_wire_diagnostics.npz"),
        "fine": np.load(root / "fine_representatives.npz"),
        "result": json.loads((root / "span_wire_result.json").read_text("utf-8")),
    }


def pair_evidence(data: dict, left: int, right: int) -> dict:
    z = data["evidence"]
    a, b = data["tracks"][left], data["tracks"][right]
    ad, bd = a["supported_s_domain_m"], b["supported_s_domain_m"]
    overlap = max(0.0, min(ad[1], bd[1]) - max(ad[0], bd[0]))
    gap = max(0.0, max(ad[0], bd[0]) - min(ad[1], bd[1]))
    probe = np.asarray(sorted(set([max(ad[0], bd[0]), min(ad[1], bd[1]) if overlap else (ad[1]+bd[0])/2])))
    ua, za = model(z, left, probe)
    ub, zb = model(z, right, probe)
    du = float(np.median(np.abs(ua - ub)))
    dz = float(np.median(np.abs(za - zb)))
    uca = segment(z, "model_u_coefficients", left)
    ucb = segment(z, "model_u_coefficients", right)
    direction_delta = float(abs(uca[-2] / uca[1] - ucb[-2] / ucb[1]))
    conflict = overlap > 0.5 and du > 0.35
    compatible = gap <= 12.0 and du <= 0.35 and dz <= 0.50 and direction_delta <= 0.12 and not conflict
    return {"tracks": [left, right], "gap_m": gap, "overlap_m": overlap, "transverse_delta_m": du,
            "height_delta_m": dz, "direction_delta": direction_delta, "neighbor_conflict": conflict,
            "compatible": compatible}


def compatible_edges(data: dict) -> list[dict]:
    eligible = [t for t in data["tracks"] if t["unique_supported_slice_count"] >= 3 and t["diagnostic_model_available"]]
    edges = []
    for i, a in enumerate(eligible):
        for b in eligible[i + 1 :]:
            edge = pair_evidence(data, a["track_id"], b["track_id"])
            if edge["compatible"]:
                edges.append(edge)
    return edges


def reassignment(data: dict, track_id: int, radial_limit: float = 0.08) -> dict:
    d, z = data["diag"], data["evidence"]
    rows = np.flatnonzero(d["candidate_mask"])
    s = d["s_m"][rows]
    u0, z0 = model(z, track_id, s)
    residual = np.hypot(d["u_m"][rows] - u0, d["z_m"][rows] - z0)
    accepted = rows[residual <= radial_limit]
    source = np.unique(d["fine_original_index"][accepted])
    ss = d["s_m"][accepted]
    slices = np.unique(np.floor(ss).astype(int)) if len(ss) else np.empty(0, int)
    gaps = np.diff(np.sort(ss)) if len(ss) > 1 else np.asarray([np.inf])
    track = data["tracks"][track_id]
    domain = track["supported_s_domain_m"]
    span_length = (domain[1] - domain[0]) / max(track["coverage_ratio"], 1e-9)
    return {"representative_rows": accepted, "source_indices": source, "s_m": ss,
            "residual_m": residual[residual <= radial_limit], "coverage": float(np.ptp(ss) / max(span_length, 1e-9)) if len(ss) else 0.0,
            "supported_slices": int(len(slices)), "max_gap_m": float(np.max(gaps)),
            "median_residual_m": float(np.median(residual[residual <= radial_limit])) if len(accepted) else None,
            "p95_residual_m": float(np.quantile(residual[residual <= radial_limit], .95)) if len(accepted) else None}


def plot_span(data: dict, path: Path, title: str, focus: list[int], reassigned: dict | None = None, edges: list[dict] | None = None) -> None:
    d, z = data["diag"], data["evidence"]
    fig, (axu, axz) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    candidates = np.flatnonzero(d["candidate_mask"])
    axu.scatter(d["s_m"][candidates], d["u_m"][candidates], s=2, c="#c8c8c8", alpha=.25, label="candidate reps")
    axz.scatter(d["s_m"][candidates], d["z_m"][candidates], s=2, c="#c8c8c8", alpha=.25)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(focus), 1)))
    for color, tid in zip(colors, focus):
        rows = segment(z, "inlier_representative_index", tid).astype(int)
        axu.scatter(d["s_m"][rows], d["u_m"][rows], s=12, color=color, label=f"track {tid} observed")
        axz.scatter(d["s_m"][rows], d["z_m"][rows], s=12, color=color)
        lo, hi = data["tracks"][tid]["supported_s_domain_m"]
        ss = np.linspace(lo, hi, 250); uu, zz = model(z, tid, ss)
        axu.plot(ss, uu, color=color, lw=2); axz.plot(ss, zz, color=color, lw=2)
    if reassigned is not None and len(reassigned["representative_rows"]):
        rows = reassigned["representative_rows"]
        axu.scatter(d["s_m"][rows], d["u_m"][rows], s=20, facecolors="none", edgecolors="black", label="real reassigned")
        axz.scatter(d["s_m"][rows], d["z_m"][rows], s=20, facecolors="none", edgecolors="black")
    if edges:
        axu.text(.01, .02, f"accepted compatibility edges: {len(edges)}", transform=axu.transAxes)
    axu.set_ylabel("transverse u (m)"); axz.set_ylabel("height z (m)"); axz.set_xlabel("longitudinal s (m)")
    axu.legend(loc="best", markerscale=2); fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def summary_track(t: dict) -> dict:
    return {k: t[k] for k in ("track_id", "status", "model_type", "coverage_ratio", "unique_supported_slice_count", "eligible_slice_count", "maximum_gap_m", "median_residual_m", "p95_residual_m", "inlier_source_count", "approach_start", "approach_end", "reasons")}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--project", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    spans = {name: load(args.project, name) for name in SPAN_DIRS}
    focus = {"S001": [8, 30], "S002": [41], "S003": [38]}
    edges = {name: compatible_edges(data) for name, data in spans.items()}
    reassign = {"S001_T30": reassignment(spans["S001"], 30), "S002_T41": reassignment(spans["S002"], 41), "S003_T38": reassignment(spans["S003"], 38)}
    relationship = pair_evidence(spans["S001"], 8, 30)
    s3 = spans["S003"]; t38 = s3["tracks"][38]; r3 = reassign["S003_T38"]
    beyond = r3["s_m"] > t38["supported_s_domain_m"][1]
    endpoint = {"real_reassigned_representatives_beyond_supported_end": int(np.count_nonzero(beyond)),
                "max_reassigned_s_m": float(np.max(r3["s_m"])) if len(r3["s_m"]) else None,
                "physical_endpoint_evidence": "PRESENT" if np.count_nonzero(beyond) >= 3 else "AMBIGUOUS" if np.count_nonzero(beyond) else "ABSENT",
                "algorithm_endpoint_detection": "MISSED" if np.count_nonzero(beyond) >= 3 else "NOT_APPLICABLE"}
    result = {
        "schema": "wire-gap-rescue-round1-diagnostic-v1", "production_pipeline_invoked": False,
        "baseline": {s: {"track_count": d["manifest"]["track_count"], "focus": summary_track(d["tracks"][focus[s][0]])} for s, d in spans.items()},
        "experiment_a": {s: {"mechanism": "predictive multi-condition track association", "changed_associations": len(edges[s]), "edges": edges[s]} for s in spans},
        "experiment_b": {s: {"graph_nodes": d["manifest"]["track_count"], "graph_edges": len(edges[s]), "edges": edges[s], "selected_chain": []} for s, d in spans.items()},
        "experiment_c": {k: {kk: vv for kk, vv in value.items() if kk not in {"representative_rows", "source_indices", "s_m", "residual_m"}} | {"observed_members": len(value["source_indices"]), "real_original_index_only": True} for k, value in reassign.items()},
        "s001_track_8_30": relationship | {"classification": "DIFFERENT_CONDUCTORS" if relationship["neighbor_conflict"] else "INSUFFICIENT_EVIDENCE"},
        "s003_endpoint": endpoint,
        "s003_threshold_sensitivity": {str(x): ("eligible_by_coverage_only" if t38["coverage_ratio"] >= x else "review_only") for x in (.78,.79,.80,.81,.82)},
        "control_preserved": summary_track(spans["S002"]["tracks"][41]),
        "fake_points_created": False,
    }
    result = json_safe(result)
    (args.output / "round1_results.json").write_text(json.dumps(result, indent=2, allow_nan=False), "utf-8")
    plot_span(spans["S001"], args.output/"WIRE_EXP_S001_BASELINE.png", "S001 baseline: observed fragments", [8,30])
    plot_span(spans["S001"], args.output/"WIRE_EXP_S001_A.png", "S001 A: predictive association (fail-closed)", [8,30], edges=edges["S001"])
    plot_span(spans["S001"], args.output/"WIRE_EXP_S001_B.png", "S001 B: compatibility graph + fitted models", [8,30], edges=edges["S001"])
    plot_span(spans["S001"], args.output/"WIRE_EXP_S001_C.png", "S001 C: real-point reassignment to Track 30 model", [30], reassigned=reassign["S001_T30"])
    plot_span(spans["S002"], args.output/"WIRE_EXP_S002_CONTROL.png", "S002 control: Track 41", [41], reassigned=reassign["S002_T41"])
    plot_span(spans["S003"], args.output/"WIRE_EXP_S003_ENDPOINT.png", "S003 endpoint evidence: Track 38", [38], reassigned=reassign["S003_T38"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
