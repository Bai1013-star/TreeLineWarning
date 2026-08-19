"""Inexpensive coarse conductor evidence for pole confirmation and routing.

This stage answers one question per pole candidate: *"does this pole
participate in a plausible long wire corridor?"*  It deliberately produces
incomplete, cheap line-like evidence on the coarse representatives — never
final W1.  Full per-span wire reconstruction happens later inside each
confirmed span.

A coarse wire segment must be high above ground, strongly linear, low in
scattering (not foliage), and predominantly horizontal (not a pole or trunk).

Coarse routing is a DUAL-PATH router:

* PATH A (``STRICT_COMPONENT``) — the original component-level gates, unchanged.
* PATH B (``SPAN_ALIGNED_RESCUE``) — a span-aligned, representative-level rescue
  that recovers locally wire-like representatives inside an already-selected
  span corridor from components PATH A rejected on component-global geometry.

PATH B is HIGH-RECALL routing evidence only.  It never promotes a conductor:
the existing conservative per-span Wire V2 confirmation stage is the only
authority for a FINAL confirmed conductor.  Neither path may permanently remove
vegetation/tree membership; only exact FINAL confirmed wire members are
subtracted from tree input downstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.evidence import GroundEvidence
from algorithms.vegetation.fast_vegetation_v3 import _grid_component_labels

COARSE_WIRE_SCHEMA = "fast-baseline-coarse-wire-evidence-v1"
# Semantic version of the coarse wire ROUTER (PATH A + PATH B).  The manual
# span regions stage keys its upstream digest on this so a routing-semantics
# change invalidates ONLY the coarse wire evidence / dependent region routing
# stages — never preprocess, Ground/HAG or the span selection.
COARSE_WIRE_ROUTING_SCHEMA = "fast-baseline-coarse-wire-routing-v2"

# Route provenance codes (per representative; diagnostic only, not a public enum).
ROUTE_NONE = 0
ROUTE_STRICT_COMPONENT = 1
ROUTE_SPAN_ALIGNED_RESCUE = 2


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
    """Deterministic coarse wire evidence set plus per-representative labels.

    ``candidate_mask`` is the PATH A (strict component) result.  When the
    span-aligned PATH B rescue has been attached, ``rescued_mask`` holds the
    rescued representatives and ``routed_mask`` is their union — the coarse
    evidence actually routed toward the per-span Wire V2 confirmation stage.
    """

    segments: tuple[CoarseWireSegment, ...]
    candidate_mask: np.ndarray  # bool, one per representative
    segment_id_by_representative: np.ndarray  # int32; -1 when not a wire segment member
    component_label_by_representative: np.ndarray  # int32
    diagnostics: dict[str, Any]
    rescued_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    route_provenance_by_representative: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int8)
    )
    rescue_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def routed_mask(self) -> np.ndarray:
        """PATH A accepted representatives union PATH B rescued representatives."""
        if len(self.rescued_mask) != len(self.candidate_mask):
            return self.candidate_mask
        return self.candidate_mask | self.rescued_mask

    @classmethod
    def from_segments_dict(
        cls,
        payload: dict[str, Any],
        *,
        candidate_mask: np.ndarray,
        segment_id_by_representative: np.ndarray,
        component_label_by_representative: np.ndarray,
        rescued_mask: np.ndarray | None = None,
        route_provenance_by_representative: np.ndarray | None = None,
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
            rescued_mask=(
                candidate_mask
                if rescued_mask is None
                else np.asarray(rescued_mask, dtype=bool)
            ),
            route_provenance_by_representative=(
                np.zeros(len(candidate_mask), dtype=np.int8)
                if route_provenance_by_representative is None
                else np.asarray(route_provenance_by_representative, dtype=np.int8)
            ),
            rescue_diagnostics=dict(payload.get("rescue_diagnostics", {})),
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


def rescue_span_aligned_wire_evidence(
    evidence: GroundEvidence,
    wire_evidence: CoarseWireResult,
    span: Any,
    config: FastBaselineConfig,
    *,
    origin_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """PATH B: recover locally wire-like reps from PATH A rejected components.

    Only representative points belonging to components PATH A rejected are
    inspected.  The rescue is purely pointwise/local and span-aligned:

        REJECTED COMPONENT
          -> span-coordinate projection
          -> existing span corridor mask
          -> existing pointwise coarse wire seed (already applied by the
             component grid, so the rejected reps are already non-ground /
             HAG-ready / low-scattering / predominantly horizontal)
          -> rescued representative subset

    The rescued subset is HIGH-RECALL routing evidence; the whole contaminated
    component is never admitted.  Conservative Wire V2 confirmation is the only
    authority for a FINAL conductor.  Returns ``(rescued_mask, diagnostics)``.
    """

    labels = wire_evidence.component_label_by_representative
    segment_ids = wire_evidence.segment_id_by_representative
    if len(labels) != evidence.count or len(segment_ids) != evidence.count:
        raise ValueError("COARSE_WIRE_RESCUE_EVIDENCE_MISALIGNED")
    # Components rejected by PATH A (labelled but no accepted segment).
    rejected = (labels > 0) & (segment_ids == -1)
    # Lazy import: algorithms.poles.topology imports CoarseWireResult from this
    # module, so a module-level import would be circular.  Reuse the canonical
    # span coordinate system (s along Pole A->B, u signed transverse offset).
    from algorithms.poles.topology import span_to_suz

    s_values, u_values = span_to_suz(span, evidence.xyz_absolute_m, origin_xyz)[:2]
    corridor = (
        (np.abs(u_values) <= float(span.half_width_m))
        & (s_values >= 0.0)
        & (s_values <= float(span.length_m))
    )
    rescued = rejected & corridor
    rows = np.flatnonzero(rescued)
    if len(rows):
        width = max(float(config.wire_slice_width_m), 0.1)
        slice_ids = np.floor(s_values[rows] / width).astype(np.int64)
        hag = evidence.hag_m[rows]
        diagnostics = {
            "rescue_representative_count": int(len(rows)),
            "s_min_m": float(np.min(s_values[rows])),
            "s_max_m": float(np.max(s_values[rows])),
            "occupied_s_bins": int(len(np.unique(slice_ids))),
            "u_range_m": float(np.ptp(u_values[rows])),
            "hag_min_m": float(np.min(hag)),
            "hag_max_m": float(np.max(hag)),
        }
    else:
        diagnostics = {
            "rescue_representative_count": 0,
            "s_min_m": 0.0,
            "s_max_m": 0.0,
            "occupied_s_bins": 0,
            "u_range_m": 0.0,
            "hag_min_m": 0.0,
            "hag_max_m": 0.0,
        }
    return rescued, diagnostics


def attach_span_aligned_rescue(
    result: CoarseWireResult,
    rescued_mask: np.ndarray,
    diagnostics: dict[str, Any],
) -> CoarseWireResult:
    """Return a new ``CoarseWireResult`` carrying the PATH B rescue provenance."""

    rescued = np.asarray(rescued_mask, dtype=bool)
    if rescued.shape != result.candidate_mask.shape:
        raise ValueError("COARSE_WIRE_RESCUE_MASK_SHAPE_INVALID")
    provenance = np.zeros(len(rescued), dtype=np.int8)
    provenance[result.candidate_mask] = ROUTE_STRICT_COMPONENT
    provenance[rescued] = ROUTE_SPAN_ALIGNED_RESCUE
    return replace(
        result,
        rescued_mask=rescued,
        route_provenance_by_representative=provenance,
        rescue_diagnostics=dict(diagnostics),
    )


def detect_coarse_wire_evidence_with_rescue(
    evidence: GroundEvidence,
    config: FastBaselineConfig,
    span: Any,
    *,
    origin_xyz: tuple[float, float, float],
) -> CoarseWireResult:
    """Run the dual-path coarse wire router: strict PATH A plus PATH B rescue."""

    strict = detect_coarse_wire_evidence(evidence, config)
    rescued, diagnostics = rescue_span_aligned_wire_evidence(
        evidence, strict, span, config, origin_xyz=origin_xyz
    )
    return attach_span_aligned_rescue(strict, rescued, diagnostics)


def coarse_wire_to_dict(result: CoarseWireResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": COARSE_WIRE_SCHEMA,
        "segment_count": result.segment_count,
        "segments": [segment.to_dict() for segment in result.segments],
        "diagnostics": dict(result.diagnostics),
    }
    if result.rescue_diagnostics:
        payload["rescue_diagnostics"] = dict(result.rescue_diagnostics)
    return payload


__all__ = [
    "COARSE_WIRE_SCHEMA",
    "COARSE_WIRE_ROUTING_SCHEMA",
    "ROUTE_NONE",
    "ROUTE_STRICT_COMPONENT",
    "ROUTE_SPAN_ALIGNED_RESCUE",
    "CoarseWireSegment",
    "CoarseWireResult",
    "detect_coarse_wire_evidence",
    "rescue_span_aligned_wire_evidence",
    "attach_span_aligned_rescue",
    "detect_coarse_wire_evidence_with_rescue",
    "coarse_wire_to_dict",
]
