"""Read-only, versioned persistence for existing Wire V2 track evidence.

This module serializes values supplied by the V2 reconstruction pass.  It must
not fit, associate, classify, or otherwise change a track.  The JSON manifest
is intentionally small and inspectable; the NPZ sidecar contains only flat
numeric arrays with explicit track offsets and never uses pickle.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

WIRE_V2_TRACK_EVIDENCE_SCHEMA = "wire-v2-track-evidence-v1"
WIRE_V2_TRACK_EVIDENCE_MARKER = "WIRE_V2_TRACK_EVIDENCE_NON_AUTHORITATIVE"


def _flat_with_offsets(
    values: Sequence[np.ndarray], *, columns: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-track numeric arrays with zero-copy-safe empty handling."""
    arrays = [np.asarray(value) for value in values]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for position, value in enumerate(arrays):
        offsets[position + 1] = offsets[position] + len(value)
    if columns is None:
        flat = np.concatenate(arrays) if offsets[-1] else np.empty(0, dtype=np.int64)
    else:
        flat = (
            np.concatenate(arrays, axis=0)
            if offsets[-1]
            else np.empty((0, columns), dtype=np.float64)
        )
    return flat, offsets


def _intervals(values: np.ndarray) -> list[list[int]]:
    """Inclusive integer intervals for a sorted-unique slice-id sequence."""
    ordered = np.unique(np.asarray(values, dtype=np.int64))
    if not len(ordered):
        return []
    starts = np.r_[True, np.diff(ordered) != 1]
    ends = np.r_[np.diff(ordered) != 1, True]
    return [
        [int(start), int(end)]
        for start, end in zip(ordered[starts], ordered[ends], strict=True)
    ]


def _missing_intervals(values: np.ndarray) -> list[list[int]]:
    ordered = np.unique(np.asarray(values, dtype=np.int64))
    if len(ordered) < 2:
        return []
    result: list[list[int]] = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        if right > left + 1:
            result.append([int(left + 1), int(right - 1)])
    return result


def _json_number(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def write_track_evidence(
    output_root: Path,
    *,
    span_name: str,
    source_point_count: int | None,
    tracks: Sequence[Any],
    diagnostics: Sequence[dict[str, Any]],
    support_source_indices: Sequence[np.ndarray],
    inlier_source_indices: Sequence[np.ndarray],
    model_s_m: Sequence[np.ndarray],
    model_xyz_local_m: Sequence[np.ndarray],
    support_overlap_pairs: Sequence[dict[str, int]],
    inlier_overlap_pairs: Sequence[dict[str, int]],
) -> dict[str, Any]:
    """Write additive evidence for the exact V2 tracks supplied by the caller."""
    count = len(tracks)
    if not all(
        len(values) == count
        for values in (
            diagnostics,
            support_source_indices,
            inlier_source_indices,
            model_s_m,
            model_xyz_local_m,
        )
    ):
        raise ValueError("WIRE_V2_TRACK_EVIDENCE_LENGTH_MISMATCH")

    support_rep_flat, support_rep_offsets = _flat_with_offsets([track.rows for track in tracks])
    inlier_rep_flat, inlier_rep_offsets = _flat_with_offsets(
        [track.inlier_rows for track in tracks]
    )
    support_source_flat, support_source_offsets = _flat_with_offsets(support_source_indices)
    inlier_source_flat, inlier_source_offsets = _flat_with_offsets(inlier_source_indices)
    support_slice_flat, support_slice_offsets = _flat_with_offsets(
        [track.support_slice_ids for track in tracks]
    )
    inlier_slice_flat, inlier_slice_offsets = _flat_with_offsets(
        [track.inlier_slice_ids for track in tracks]
    )
    sample_s_flat, sample_s_offsets = _flat_with_offsets(model_s_m)
    sample_xyz_flat, sample_xyz_offsets = _flat_with_offsets(model_xyz_local_m, columns=3)
    u_coeff_flat, u_coeff_offsets = _flat_with_offsets(
        [track.model_u_coefficients for track in tracks]
    )
    z_coeff_flat, z_coeff_offsets = _flat_with_offsets(
        [track.model_z_coefficients for track in tracks]
    )
    catenary_flat, catenary_offsets = _flat_with_offsets(
        [track.catenary_parameters for track in tracks]
    )

    manifest_tracks: list[dict[str, Any]] = []
    for track_id, (track, diagnostic, support_source, inlier_source, samples) in enumerate(
        zip(
            tracks,
            diagnostics,
            support_source_indices,
            inlier_source_indices,
            model_s_m,
            strict=True,
        )
    ):
        status = str(diagnostic["status"])
        distance_ready = status == "CONFIRMED"
        model_available = bool(len(samples))
        manifest_tracks.append(
            {
                "track_id": track_id,
                "status": status,
                "distance_ready": distance_ready,
                "diagnostic_model_available": model_available,
                "model_type": track.model_type,
                "reasons": list(diagnostic["reasons"]),
                "representative_count": int(len(track.rows)),
                "raw_support_count": int(track.raw_support_count),
                "support_source_count": int(len(support_source)),
                "inlier_representative_count": int(len(track.inlier_rows)),
                "inlier_source_count": int(len(inlier_source)),
                "supported_slice_count": int(track.supported_slice_count),
                "eligible_slice_count": int(track.eligible_slice_count),
                "supported_slice_ratio": _json_number(track.supported_slice_ratio),
                "unique_supported_slice_count": int(track.unique_supported_slice_count),
                "bounded_supported_slice_ratio": _json_number(
                    track.bounded_supported_slice_ratio
                ),
                "supported_slice_intervals": _intervals(track.inlier_slice_ids),
                "missing_slice_intervals": _missing_intervals(track.inlier_slice_ids),
                "supported_s_domain_m": [
                    _json_number(track.supported_s_min_m),
                    _json_number(track.supported_s_max_m),
                ],
                "coverage_ratio": _json_number(track.coverage_ratio),
                "maximum_gap_m": _json_number(track.maximum_gap_m),
                "inlier_ratio": _json_number(track.inlier_ratio),
                "median_residual_m": _json_number(track.median_residual_m),
                "p95_residual_m": _json_number(track.p95_residual_m),
                "median_u_m": _json_number(track.median_u_m),
                "u_residual_p95_m": _json_number(track.u_residual_p95_m),
                "approach_start": bool(diagnostic["approach_start"]),
                "approach_end": bool(diagnostic["approach_end"]),
                "model_sample_count": int(len(samples)),
                "distance_ready_polyline_edge_count": (
                    int(max(0, len(samples) - 1)) if distance_ready else 0
                ),
            }
        )

    manifest = {
        "schema": WIRE_V2_TRACK_EVIDENCE_SCHEMA,
        "marker": WIRE_V2_TRACK_EVIDENCE_MARKER,
        "span_id": span_name,
        "source_point_count": source_point_count,
        "track_count": count,
        "array_format": "flat-npz-with-track-offsets",
        "track_order": "V2 deterministic track_id ascending",
        "support_identity_semantics": (
            "Each support representative/source identity belongs to the actual "
            "SliceTrack.rows association from this V2 execution."
        ),
        "inlier_identity_semantics": (
            "Each inlier representative/source identity belongs to the actual "
            "SliceTrack.inlier_rows accepted by this V2 execution."
        ),
        "overlap_report": {
            "support_representative_overlap_pairs": list(support_overlap_pairs),
            "inlier_representative_overlap_pairs": list(inlier_overlap_pairs),
        },
        "tracks": manifest_tracks,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "wire_v2_track_evidence.json"
    npz_path = output_root / "wire_v2_track_evidence.npz"
    json_temporary = json_path.with_name(json_path.name + ".tmp")
    npz_temporary = npz_path.with_name(npz_path.name + ".tmp.npz")
    json_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
    )
    np.savez(
        npz_temporary,
        track_id=np.arange(count, dtype=np.int64),
        support_representative_index_flat=np.asarray(support_rep_flat, dtype=np.int64),
        support_representative_index_offsets=support_rep_offsets,
        inlier_representative_index_flat=np.asarray(inlier_rep_flat, dtype=np.int64),
        inlier_representative_index_offsets=inlier_rep_offsets,
        support_source_index_flat=np.asarray(support_source_flat, dtype=np.int64),
        support_source_index_offsets=support_source_offsets,
        inlier_source_index_flat=np.asarray(inlier_source_flat, dtype=np.int64),
        inlier_source_index_offsets=inlier_source_offsets,
        support_slice_id_flat=np.asarray(support_slice_flat, dtype=np.int64),
        support_slice_id_offsets=support_slice_offsets,
        inlier_slice_id_flat=np.asarray(inlier_slice_flat, dtype=np.int64),
        inlier_slice_id_offsets=inlier_slice_offsets,
        model_sample_s_m_flat=np.asarray(sample_s_flat, dtype=np.float64),
        model_sample_s_m_offsets=sample_s_offsets,
        model_sample_xyz_local_m_flat=np.asarray(sample_xyz_flat, dtype=np.float32),
        model_sample_xyz_local_m_offsets=sample_xyz_offsets,
        model_u_coefficients_flat=np.asarray(u_coeff_flat, dtype=np.float64),
        model_u_coefficients_offsets=u_coeff_offsets,
        model_z_coefficients_flat=np.asarray(z_coeff_flat, dtype=np.float64),
        model_z_coefficients_offsets=z_coeff_offsets,
        catenary_parameters_flat=np.asarray(catenary_flat, dtype=np.float64),
        catenary_parameters_offsets=catenary_offsets,
    )
    npz_temporary.replace(npz_path)
    json_temporary.replace(json_path)
    return manifest


__all__ = [
    "WIRE_V2_TRACK_EVIDENCE_MARKER",
    "WIRE_V2_TRACK_EVIDENCE_SCHEMA",
    "write_track_evidence",
]
