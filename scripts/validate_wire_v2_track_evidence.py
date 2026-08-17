"""Validate additive Wire V2 track evidence against a frozen shadow baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _track_signature(track: dict[str, object]) -> dict[str, object]:
    """All decision-bearing fields, excluding only execution timing/paths."""
    return {
        key: track[key]
        for key in (
            "track_id",
            "status",
            "reasons",
            "model_type",
            "representative_count",
            "raw_support_count",
            "supported_slice_count",
            "supported_slice_ratio",
            "coverage_ratio",
            "maximum_gap_m",
            "supported_s_domain_m",
            "inlier_ratio",
            "median_residual_m",
            "p95_residual_m",
            "median_u_m",
            "u_residual_p95_m",
            "approach_start",
            "approach_end",
        )
    }


def _slice(stream: object, name: str, track_id: int) -> np.ndarray:
    offsets = np.asarray(stream[f"{name}_offsets"], dtype=np.int64)
    flat = np.asarray(stream[f"{name}_flat"])
    return flat[offsets[track_id] : offsets[track_id + 1]]


def _validate_span(baseline: Path, evidence: Path, span_id: str) -> dict[str, object]:
    baseline_summary = _load_json(baseline / span_id / "shadow_summary.json")
    evidence_summary = _load_json(evidence / span_id / "shadow_summary.json")
    for key in (
        "candidate_track_count",
        "confirmed_wire_count",
        "review_only_track_count",
        "rejected_track_count",
        "distance_ready_segment_count",
        "distance_ready_polyline_edge_count",
        "confirmed_member_source_identity",
        "v2_configuration",
    ):
        if baseline_summary[key] != evidence_summary[key]:
            raise ValueError(f"{span_id}: baseline mismatch for {key}")
    old_tracks = baseline_summary["diagnostics"]["v2_tracks"]
    new_tracks = evidence_summary["diagnostics"]["v2_tracks"]
    if [_track_signature(track) for track in old_tracks] != [
        _track_signature(track) for track in new_tracks
    ]:
        raise ValueError(f"{span_id}: decision-bearing V2 diagnostics changed")
    with np.load(baseline / span_id / "w1_arrays.npz", allow_pickle=False) as old, np.load(
        evidence / span_id / "w1_arrays.npz", allow_pickle=False
    ) as new:
        if set(old.files) != set(new.files):
            raise ValueError(f"{span_id}: W1 array names changed")
        for name in old.files:
            if not np.array_equal(old[name], new[name]):
                raise ValueError(f"{span_id}: W1 array changed: {name}")
        w1_curve = np.asarray(new["sampled_curve_xyz"], dtype=np.float32)
        w1_offsets = np.asarray(new["sampled_curve_offsets"], dtype=np.int64)
    with np.load(
        evidence / span_id / "span_wire_diagnostics.npz", allow_pickle=False
    ) as diagnostics, np.load(
        evidence / span_id / "wire_v2_track_evidence.npz", allow_pickle=False
    ) as artifact:
        manifest = _load_json(evidence / span_id / "wire_v2_track_evidence.json")
        tracks = manifest["tracks"]
        source_limit = int(manifest["source_point_count"])
        if len(tracks) != int(manifest["track_count"]):
            raise ValueError(f"{span_id}: track manifest count mismatch")
        if manifest["overlap_report"]["inlier_representative_overlap_pairs"]:
            raise ValueError(f"{span_id}: unexpected inlier representative overlap")
        confirmed_sources: list[np.ndarray] = []
        unmatched_w1_curves = set(range(len(w1_offsets) - 1))
        for track in tracks:
            track_id = int(track["track_id"])
            support = _slice(artifact, "support_source_index", track_id)
            inlier = _slice(artifact, "inlier_source_index", track_id)
            if np.any(support < 0) or np.any(support >= source_limit):
                raise ValueError(f"{span_id}: out-of-range support source identity")
            if len(support) != len(np.unique(support)):
                raise ValueError(f"{span_id}: duplicate support source identity")
            if len(support) != int(track["raw_support_count"]):
                raise ValueError(f"{span_id}: support source count mismatch")
            if bool(track["distance_ready"]) != (track["status"] == "CONFIRMED"):
                raise ValueError(f"{span_id}: distance-ready status mismatch")
            if track["status"] == "CONFIRMED":
                confirmed_sources.append(inlier)
                model_xyz = _slice(artifact, "model_sample_xyz_local_m", track_id)
                matching = [
                    position
                    for position in unmatched_w1_curves
                    if np.array_equal(
                        model_xyz, w1_curve[w1_offsets[position] : w1_offsets[position + 1]]
                    )
                ]
                if len(matching) != 1:
                    raise ValueError(f"{span_id}: confirmed model is not an exact W1 polyline")
                unmatched_w1_curves.remove(matching[0])
        if unmatched_w1_curves:
            raise ValueError(f"{span_id}: W1 polyline has no confirmed track evidence")
        published = np.asarray(diagnostics["confirmed_member_source_index"], dtype=np.int64)
        projected = (
            np.unique(np.concatenate(confirmed_sources))
            if confirmed_sources
            else np.empty(0, dtype=np.int64)
        )
        if not np.array_equal(projected, published):
            raise ValueError(f"{span_id}: confirmed source union mismatch")
    return {
        "span_id": span_id,
        "track_count": len(tracks),
        "confirmed_source_count": int(len(published)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_shadow_root", type=Path)
    parser.add_argument("evidence_shadow_root", type=Path)
    parser.add_argument("--span-id", action="append")
    args = parser.parse_args()
    span_ids = args.span_id or [f"S{position:03d}" for position in range(1, 6)]
    print(json.dumps({"validated": [
        _validate_span(args.baseline_shadow_root, args.evidence_shadow_root, span_id)
        for span_id in span_ids
    ]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
