"""Inexpensive coarse conductor evidence for pole confirmation and routing.

This stage answers one question per pole candidate: *"does this pole
participate in a plausible long wire corridor?"*  It deliberately produces
incomplete, cheap line-like evidence on the coarse representatives — never
final W1.  Full per-span wire reconstruction happens later inside each
confirmed span.

A coarse wire segment must be high above ground, strongly linear, low in
scattering (not foliage), and predominantly horizontal (not a pole or trunk).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.evidence import GroundEvidence
from algorithms.vegetation.fast_vegetation_v3 import _grid_component_labels

COARSE_WIRE_SCHEMA = "fast-baseline-coarse-wire-evidence-v1"


@dataclass(frozen=True, slots=True)
class CoarseWireSegment:
    """One coarse line-like wire corridor segment (routing evidence only)."""

    segment_id: int
    representative_count: int
    xy_span_m: float
    z_span_m: float
    median_hag_m: float
    start_xyz_local_m: tuple[float, float, float]
    end_xyz_local_m: tuple[float, float, float]
    direction_xyz_local_m: tuple[float, float, float]
    robust_z_span_m: float = 0.0
    z_fit_residual_m: float = 0.0
    transverse_width_m: float = 0.0
    longitudinal_coverage_fraction: float = 0.0
    usage: str = "COARSE_WIRE_EVIDENCE_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "representative_count": self.representative_count,
            "xy_span_m": self.xy_span_m,
            "z_span_m": self.z_span_m,
            "median_hag_m": self.median_hag_m,
            "start_xyz_local_m": list(self.start_xyz_local_m),
            "end_xyz_local_m": list(self.end_xyz_local_m),
            "direction_xyz_local_m": list(self.direction_xyz_local_m),
            "robust_z_span_m": self.robust_z_span_m,
            "z_fit_residual_m": self.z_fit_residual_m,
            "transverse_width_m": self.transverse_width_m,
            "longitudinal_coverage_fraction": self.longitudinal_coverage_fraction,
            "usage": self.usage,
        }


@dataclass(frozen=True, slots=True)
class CoarseWireResult:
    """Deterministic coarse wire evidence set plus per-representative labels."""

    segments: tuple[CoarseWireSegment, ...]
    candidate_mask: np.ndarray  # bool, one per representative
    segment_id_by_representative: np.ndarray  # int32; -1 when not a wire segment member
    component_label_by_representative: np.ndarray  # int32
    diagnostics: dict[str, Any]

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @classmethod
    def from_segments_dict(
        cls,
        payload: dict[str, Any],
        *,
        candidate_mask: np.ndarray,
        segment_id_by_representative: np.ndarray,
        component_label_by_representative: np.ndarray,
    ) -> CoarseWireResult:
        segments = tuple(
            CoarseWireSegment(
                segment_id=int(item["segment_id"]),
                representative_count=int(item["representative_count"]),
                xy_span_m=float(item["xy_span_m"]),
                z_span_m=float(item["z_span_m"]),
                median_hag_m=float(item["median_hag_m"]),
                start_xyz_local_m=tuple(
                    float(value) for value in item["start_xyz_local_m"]
                ),
                end_xyz_local_m=tuple(float(value) for value in item["end_xyz_local_m"]),
                direction_xyz_local_m=tuple(
                    float(value) for value in item["direction_xyz_local_m"]
                ),
                robust_z_span_m=float(item.get("robust_z_span_m", item["z_span_m"])),
                z_fit_residual_m=float(item.get("z_fit_residual_m", item["z_span_m"])),
                transverse_width_m=float(item.get("transverse_width_m", 0.0)),
                longitudinal_coverage_fraction=float(
                    item.get("longitudinal_coverage_fraction", 1.0)
                ),
                usage=str(item.get("usage", "COARSE_WIRE_EVIDENCE_ONLY")),
            )
            for item in payload["segments"]
        )
        return cls(
            segments=segments,
            candidate_mask=candidate_mask,
            segment_id_by_representative=segment_id_by_representative,
            component_label_by_representative=component_label_by_representative,
            diagnostics=dict(payload.get("diagnostics", {})),
        )


def detect_coarse_wire_evidence(
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> CoarseWireResult:
    """Detect coarse line-like wire corridor evidence on the representatives."""

    started = time.perf_counter()
    non_ground = evidence.non_ground_mask()
    # Coarse wire evidence is routing-only.  A single thin conductor is locally
    # linear, but a bundle of parallel conductors is locally *planar* (the two
    # conductors span a horizontal sheet).  The seed gate therefore accepts any
    # high, low-scattering, predominantly horizontal structure; the segment
    # filter then requires a long, thin, horizontal corridor.
    seed = (
        non_ground
        & np.isfinite(evidence.hag_m)
        & (evidence.hag_m >= config.wire_coarse_min_hag_m)
        & (evidence.scattering <= config.wire_coarse_scattering_max)
        & (evidence.main_direction_z <= config.wire_coarse_verticality_max)
    )
    labels, _cell, _origin = _grid_component_labels(
        evidence.xyz_local_m, seed, config.wire_coarse_group_cell_m
    )
    segments: list[CoarseWireSegment] = []
    candidate_mask = np.zeros(evidence.count, dtype=bool)
    segment_id_by_rep = np.full(evidence.count, -1, dtype=np.int32)
    for component_id in np.unique(labels[labels > 0]):
        rows = np.flatnonzero(labels == component_id)
        if len(rows) < config.wire_coarse_min_segment_points:
            continue
        points = evidence.xyz_local_m[rows]
        span = np.ptp(points, axis=0)
        xy_span = float(np.hypot(span[0], span[1]))
        z_span = float(span[2])
        if xy_span < config.wire_coarse_min_segment_span_m:
            continue
        centered = points - points.mean(axis=0)
        _values, vectors = np.linalg.eigh(centered.T @ centered)
        direction = vectors[:, -1]
        if float(abs(direction[2])) > config.wire_coarse_verticality_max:
            continue
        projection = centered @ direction
        start = points.mean(axis=0) + direction * float(np.min(projection))
        end = points.mean(axis=0) + direction * float(np.max(projection))
        axis_xy = np.asarray(direction[:2], dtype=np.float64)
        axis_xy /= max(float(np.linalg.norm(axis_xy)), 1e-12)
        perp_xy = np.asarray((-axis_xy[1], axis_xy[0]), dtype=np.float64)
        relative_xy = points[:, :2] - points.mean(axis=0)[:2]
        longitudinal = relative_xy @ axis_xy
        transverse = relative_xy @ perp_xy
        lower = (1.0 - config.wire_coarse_robust_z_quantile_fraction) / 2.0
        upper = 1.0 - lower
        robust_z_span = float(
            np.quantile(points[:, 2], upper) - np.quantile(points[:, 2], lower)
        )
        fit = np.polyfit(longitudinal, points[:, 2], 1)
        residual = np.abs(points[:, 2] - np.polyval(fit, longitudinal))
        z_fit_residual = float(
            np.quantile(residual, config.wire_coarse_residual_quantile_fraction)
        )
        transverse_width = float(
            np.quantile(transverse, upper) - np.quantile(transverse, lower)
        )
        bins = np.linspace(
            float(np.min(longitudinal)),
            float(np.max(longitudinal)),
            config.wire_coarse_longitudinal_coverage_bins + 1,
        )
        occupied_bins = np.unique(np.clip(np.digitize(longitudinal, bins[1:-1]), 0, None))
        longitudinal_coverage = float(
            len(occupied_bins) / config.wire_coarse_longitudinal_coverage_bins
        )
        if robust_z_span > config.wire_coarse_max_robust_z_span_m:
            continue
        if z_fit_residual > config.wire_coarse_max_z_fit_residual_m:
            continue
        if transverse_width > config.wire_coarse_max_transverse_width_m:
            continue
        if longitudinal_coverage < config.wire_coarse_min_longitudinal_coverage_fraction:
            continue
        segment = CoarseWireSegment(
            segment_id=len(segments),
            representative_count=int(len(rows)),
            xy_span_m=xy_span,
            z_span_m=z_span,
            median_hag_m=float(np.median(evidence.hag_m[rows])),
            start_xyz_local_m=tuple(float(value) for value in start),
            end_xyz_local_m=tuple(float(value) for value in end),
            direction_xyz_local_m=tuple(float(value) for value in direction),
            robust_z_span_m=robust_z_span,
            z_fit_residual_m=z_fit_residual,
            transverse_width_m=transverse_width,
            longitudinal_coverage_fraction=longitudinal_coverage,
        )
        segments.append(segment)
        candidate_mask[rows] = True
        segment_id_by_rep[rows] = len(segments) - 1
    return CoarseWireResult(
        segments=tuple(segments),
        candidate_mask=candidate_mask,
        segment_id_by_representative=segment_id_by_rep,
        component_label_by_representative=labels,
        diagnostics={
            "schema": COARSE_WIRE_SCHEMA,
            "seed_representative_count": int(np.count_nonzero(seed)),
            "component_count": int(np.count_nonzero(np.unique(labels[labels > 0]))),
            "segment_count": len(segments),
            "candidate_representative_count": int(np.count_nonzero(candidate_mask)),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def coarse_wire_to_dict(result: CoarseWireResult) -> dict[str, Any]:
    return {
        "schema": COARSE_WIRE_SCHEMA,
        "segment_count": result.segment_count,
        "segments": [segment.to_dict() for segment in result.segments],
        "diagnostics": dict(result.diagnostics),
    }


__all__ = [
    "COARSE_WIRE_SCHEMA",
    "CoarseWireSegment",
    "CoarseWireResult",
    "detect_coarse_wire_evidence",
    "coarse_wire_to_dict",
]
