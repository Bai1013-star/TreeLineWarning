"""Pole <-> wire mutual validation with explicit confidence states.

A pole-like candidate is confirmed as a power pole only when both geometries
agree: the object is a tall narrow vertical structure AND it carries evidence
of a plausible long conductor corridor near its upper region.  Wires help
confirm poles; poles help route wires.  Neither subsystem must solve the whole
problem alone.

Confidence states (Part 6):

- CONFIRMED_POWER_POLE: strong geometry + >= ``pole_min_wire_support_segments``
  parallel coarse conductor segments near the top, no broad crown;
- LIKELY_POWER_POLE: real wire support but weaker (or some crown);
- POLE_LIKE_REVIEW: plausible vertical object without wire evidence (street
  light, lamp pole, unknown) — kept, never authoritative for spans;
- REJECTED_POLE_LIKE: tree trunk with a broad volumetric crown above, or a
  tiny isolated vertical object.

Only CONFIRMED + LIKELY poles are usable for span construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.candidates import PoleCandidateResult
from algorithms.poles.evidence import GroundEvidence
from algorithms.poles.wire_evidence import CoarseWireResult

VALIDATION_SCHEMA = "fast-baseline-pole-validation-v1"

CONFIRMED_POWER_POLE = "CONFIRMED_POWER_POLE"
LIKELY_POWER_POLE = "LIKELY_POWER_POLE"
POLE_LIKE_REVIEW = "POLE_LIKE_REVIEW"
REJECTED_POLE_LIKE = "REJECTED_POLE_LIKE"

_STATES = (CONFIRMED_POWER_POLE, LIKELY_POWER_POLE, POLE_LIKE_REVIEW, REJECTED_POLE_LIKE)


@dataclass(frozen=True, slots=True)
class PoleValidationRecord:
    """Per-candidate mutual-validation outcome."""

    pole_candidate_id: int
    state: str
    wire_support_count: int
    wire_support_parallelism: float
    wire_support_segment_ids: tuple[int, ...]
    primary_wire_direction_xy: tuple[float, float, float] | None
    attachment_elevation_m: float | None
    attachment_band_min_m: float | None
    attachment_band_max_m: float | None
    crown_evidence: bool
    crown_volume_proxy: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pole_candidate_id": self.pole_candidate_id,
            "state": self.state,
            "wire_support_count": self.wire_support_count,
            "wire_support_parallelism": self.wire_support_parallelism,
            "wire_support_segment_ids": list(self.wire_support_segment_ids),
            "primary_wire_direction_xy": (
                None
                if self.primary_wire_direction_xy is None
                else list(self.primary_wire_direction_xy)
            ),
            "attachment_elevation_m": self.attachment_elevation_m,
            "attachment_band_min_m": self.attachment_band_min_m,
            "attachment_band_max_m": self.attachment_band_max_m,
            "crown_evidence": self.crown_evidence,
            "crown_volume_proxy": self.crown_volume_proxy,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PoleValidationResult:
    """Explicit pole confidence states aligned with the candidate records."""

    records: tuple[PoleValidationRecord, ...]  # index == pole_candidate_id
    confirmed_pole_ids: tuple[int, ...]
    likely_pole_ids: tuple[int, ...]
    review_pole_ids: tuple[int, ...]
    rejected_pole_ids: tuple[int, ...]
    usable_pole_ids: tuple[int, ...]  # confirmed + likely
    diagnostics: dict[str, Any]

    def state_by_candidate(self) -> dict[int, str]:
        return {record.pole_candidate_id: record.state for record in self.records}

    @classmethod
    def from_records_dict(cls, payload: dict[str, Any]) -> PoleValidationResult:
        records = tuple(
            PoleValidationRecord(
                pole_candidate_id=int(item["pole_candidate_id"]),
                state=str(item["state"]),
                wire_support_count=int(item["wire_support_count"]),
                wire_support_parallelism=float(item["wire_support_parallelism"]),
                wire_support_segment_ids=tuple(
                    int(value) for value in item["wire_support_segment_ids"]
                ),
                primary_wire_direction_xy=(
                    None
                    if item.get("primary_wire_direction_xy") is None
                    else tuple(
                        float(value)
                        for value in item["primary_wire_direction_xy"]
                    )
                ),
                attachment_elevation_m=(
                    None
                    if item.get("attachment_elevation_m") is None
                    else float(item["attachment_elevation_m"])
                ),
                attachment_band_min_m=(
                    None
                    if item.get("attachment_band_min_m") is None
                    else float(item["attachment_band_min_m"])
                ),
                attachment_band_max_m=(
                    None
                    if item.get("attachment_band_max_m") is None
                    else float(item["attachment_band_max_m"])
                ),
                crown_evidence=bool(item["crown_evidence"]),
                crown_volume_proxy=int(item["crown_volume_proxy"]),
                reason=str(item["reason"]),
            )
            for item in payload["records"]
        )
        return cls(
            records=records,
            confirmed_pole_ids=tuple(
                int(value) for value in payload.get("confirmed_pole_ids", [])
            ),
            likely_pole_ids=tuple(int(value) for value in payload.get("likely_pole_ids", [])),
            review_pole_ids=tuple(int(value) for value in payload.get("review_pole_ids", [])),
            rejected_pole_ids=tuple(
                int(value) for value in payload.get("rejected_pole_ids", [])
            ),
            usable_pole_ids=tuple(int(value) for value in payload.get("usable_pole_ids", [])),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


def _pole_wire_support(
    pole_top_xy: np.ndarray,
    pole_top_z: float,
    evidence: GroundEvidence,
    wire_evidence: CoarseWireResult,
    config: FastBaselineConfig,
) -> tuple[int, float, np.ndarray | None, np.ndarray]:
    """Count wire-candidate reps near the pole top zone and their coherence.

    Real conductors are frequently fragmented by vegetation at coarse scale, so
    support is measured directly on wire-candidate representatives near the pole
    top rather than requiring a full-length wire segment.  The coherence is the
    2D line-likeness of the conductor reps over a window twice the support
    radius, so a real conductor's elongation along the corridor is captured even
    when a bundle of parallel wires is locally planar at the pole itself.
    """

    if not wire_evidence.candidate_mask.any():
        return 0, 0.0, None, np.empty(0, dtype=np.int64)
    long_segment_ids = {
        segment.segment_id
        for segment in wire_evidence.segments
        if segment.xy_span_m >= config.pole_min_wire_support_segment_span_m
    }
    if not long_segment_ids:
        return 0, 0.0, None, np.empty(0, dtype=np.int64)
    xy = evidence.xyz_local_m[:, :2]
    z_ok = (
        np.abs(evidence.xyz_local_m[:, 2] - pole_top_z)
        <= config.pole_wire_approach_hag_tolerance_m
    )
    near_strict = (
        np.linalg.norm(xy - pole_top_xy, axis=1) <= config.pole_top_wire_support_radius_m
    )
    rows = np.flatnonzero(near_strict & z_ok & wire_evidence.candidate_mask)
    rows = rows[np.isin(wire_evidence.segment_id_by_representative[rows], list(long_segment_ids))]
    support_count = int(len(rows))
    # Coherence window is twice the strict support radius.
    window = config.pole_top_wire_support_radius_m * 2.0
    near_wide = np.linalg.norm(xy - pole_top_xy, axis=1) <= window
    wide_rows = np.flatnonzero(near_wide & z_ok & wire_evidence.candidate_mask)
    wide_rows = wide_rows[
        np.isin(
            wire_evidence.segment_id_by_representative[wide_rows],
            list(long_segment_ids),
        )
    ]
    if len(wide_rows) < 2:
        return support_count, 0.0, None, rows
    centered = xy[wide_rows] - xy[wide_rows].mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    total = float(np.max(values)) + 1e-12
    linearity = float((values[1] - values[0]) / total)
    direction = np.asarray((vectors[0, 1], vectors[1, 1]), dtype=np.float64)
    return support_count, linearity, direction, rows


def _attachment_zone(
    candidate: Any,
    support_rows: np.ndarray,
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> tuple[float, float, float]:
    """Estimate conductor elevation from a robust support band, not max Z."""

    if len(support_rows) >= 2:
        z = np.asarray(evidence.xyz_local_m[support_rows, 2], dtype=np.float64)
        lower = float(np.quantile(z, config.pole_attachment_lower_quantile))
        upper = float(np.quantile(z, config.pole_attachment_upper_quantile))
        return float(np.median(z)), lower, upper
    top = float(candidate.top_xyz_local_m[2])
    half_band = config.pole_attachment_fallback_half_band_m
    return top, top - half_band, top + half_band


def _crown_volume_proxy(
    pole_top_xy: np.ndarray,
    pole_top_z: float,
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> int:
    """Count volumetric (foliage-like) representatives around the pole top."""

    non_ground = evidence.non_ground_mask()
    within_radius = np.linalg.norm(evidence.xyz_local_m[:, :2] - pole_top_xy, axis=1) <= (
        config.pole_crown_penalty_radius_m
    )
    volumetric = evidence.scattering >= 0.15
    high = evidence.hag_m >= config.pole_crown_penalty_min_hag_m
    near_top = np.abs(evidence.xyz_local_m[:, 2] - pole_top_z) <= 6.0
    return int(np.count_nonzero(non_ground & within_radius & volumetric & high & near_top))


def validate_poles(
    candidates: PoleCandidateResult,
    wire_evidence: CoarseWireResult,
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> PoleValidationResult:
    """Assign explicit confidence states to every accepted pole candidate."""

    started = time.perf_counter()
    records: list[PoleValidationRecord] = []
    for candidate in candidates.records:
        pole_top_xy = np.asarray(candidate.top_xyz_local_m[:2], dtype=np.float64)
        pole_top_z = float(candidate.top_xyz_local_m[2])
        support_count, parallelism, primary_2d, support_rows = _pole_wire_support(
            pole_top_xy,
            pole_top_z,
            evidence,
            wire_evidence,
            config,
        )
        crown_proxy = _crown_volume_proxy(pole_top_xy, pole_top_z, evidence, config)
        crown_evidence = crown_proxy >= 20 and crown_proxy > 3 * max(
            1, candidate.representative_count
        )
        primary_direction = None
        if primary_2d is not None:
            norm = float(np.linalg.norm(primary_2d))
            if norm > 1e-9:
                primary_direction = tuple(
                    float(value) for value in primary_2d / norm
                )
        attachment, attachment_min, attachment_max = _attachment_zone(
            candidate, support_rows, evidence, config
        )
        support_segment_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in wire_evidence.segment_id_by_representative[support_rows]
                    if int(value) >= 0
                }
            )
        )

        if (
            support_count >= config.pole_min_wire_support_reps
            and parallelism >= config.pole_min_wire_support_parallelism
        ):
            # Strong conductor evidence wins over nearby foliage: a power pole
            # commonly stands among trees, so adjacent canopy does not turn a
            # pole with coherent conductor support into a tree trunk.
            state = CONFIRMED_POWER_POLE
            reason = (
                "tall narrow vertical structure with "
                f"{support_count} coherent wire-candidate reps near the top"
            )
        elif support_count >= 2 and not crown_evidence:
            state = LIKELY_POWER_POLE
            reason = (
                f"vertical structure with {support_count} wire-candidate reps; "
                "weaker/less coherent wire evidence"
            )
        elif crown_evidence:
            state = REJECTED_POLE_LIKE
            reason = "tree-trunk-like: broad volumetric crown above, no wire corridor"
        elif candidate.height_m < 12.0:
            state = POLE_LIKE_REVIEW
            reason = "short vertical object without wire evidence (street-light-like)"
        else:
            state = POLE_LIKE_REVIEW
            reason = "vertical object without wire evidence; review required"
        records.append(
            PoleValidationRecord(
                pole_candidate_id=candidate.pole_candidate_id,
                state=state,
                wire_support_count=support_count,
                wire_support_parallelism=parallelism,
                wire_support_segment_ids=support_segment_ids,
                primary_wire_direction_xy=primary_direction,
                attachment_elevation_m=attachment,
                attachment_band_min_m=attachment_min,
                attachment_band_max_m=attachment_max,
                crown_evidence=crown_evidence,
                crown_volume_proxy=crown_proxy,
                reason=reason,
            )
        )
    by_state: dict[str, list[int]] = {state: [] for state in _STATES}
    for record in records:
        by_state[record.state].append(record.pole_candidate_id)
    confirmed = tuple(by_state[CONFIRMED_POWER_POLE])
    likely = tuple(by_state[LIKELY_POWER_POLE])
    return PoleValidationResult(
        records=tuple(records),
        confirmed_pole_ids=confirmed,
        likely_pole_ids=likely,
        review_pole_ids=tuple(by_state[POLE_LIKE_REVIEW]),
        rejected_pole_ids=tuple(by_state[REJECTED_POLE_LIKE]),
        usable_pole_ids=tuple(sorted(confirmed + likely)),
        diagnostics={
            "schema": VALIDATION_SCHEMA,
            "candidate_count": len(records),
            "confirmed_pole_count": len(confirmed),
            "likely_pole_count": len(likely),
            "review_pole_count": len(by_state[POLE_LIKE_REVIEW]),
            "rejected_pole_count": len(by_state[REJECTED_POLE_LIKE]),
            "usable_pole_count": len(confirmed) + len(likely),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def pole_validation_to_dict(result: PoleValidationResult) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "records": [record.to_dict() for record in result.records],
        "confirmed_pole_ids": list(result.confirmed_pole_ids),
        "likely_pole_ids": list(result.likely_pole_ids),
        "review_pole_ids": list(result.review_pole_ids),
        "rejected_pole_ids": list(result.rejected_pole_ids),
        "usable_pole_ids": list(result.usable_pole_ids),
        "diagnostics": dict(result.diagnostics),
    }


__all__ = [
    "VALIDATION_SCHEMA",
    "CONFIRMED_POWER_POLE",
    "LIKELY_POWER_POLE",
    "POLE_LIKE_REVIEW",
    "REJECTED_POLE_LIKE",
    "PoleValidationRecord",
    "PoleValidationResult",
    "validate_poles",
    "pole_validation_to_dict",
]
