"""V5A2 automatic tower-candidate detector.

The automatic detector is deliberately an adapter / filtering / scoring / NMS
layer around the existing proven geometry-first pole detector
(``algorithms.poles.candidates.detect_pole_candidates``) on verified
coarse-representative / HAG input.  It never invents a new detector from
scratch and never reads Giro3D / COPC LOD geometry.

Flow::

    verified representative/HAG input
        -> existing geometry-first pole detection
        -> recall-oriented filter (confirmed / likely / review-worthy)
        -> crown-like / obvious non-pole exclusion
        -> optional coarse wire support (confidence boost; HYBRID only then)
        -> tower structural score / confidence
        -> greedy XY merge / NMS
        -> deterministic absolute-coordinate ordering
        -> P001 ... P999
        -> TowerCandidateSet

Priorities: HIGH RECALL (false positives are acceptable if manageable in
review) while refusing to emit hundreds/thousands of trivial false candidates.
All parameters come from fixed configuration defaults; there are no GPU or
model dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig, canonical_digest
from core.contracts.project import CoordinateContext, SourceIdentityBinding
from core.contracts.tower_candidates import (
    CandidateOrigin,
    DetectorKind,
    DetectorProvenance,
    TowerCandidate,
    TowerCandidateBounds,
    TowerCandidateQualityFlag,
    TowerCandidateSet,
)

from algorithms.poles.candidates import detect_pole_candidates
from algorithms.poles.evidence import GroundEvidence
from algorithms.poles.validation import _pole_wire_support
from algorithms.poles.wire_evidence import CoarseWireResult

TOWER_AUTOMATIC_DETECTOR_NAME = "fast-pole-adapter"
TOWER_AUTOMATIC_DETECTOR_VERSION = "v5a2-automatic-1"
TOWER_AUTOMATIC_DETECTOR_SCHEMA = "treeline-tower-automatic-detector-v1"


@dataclass(frozen=True, slots=True)
class TowerAutomaticDetectorConfig:
    """Fixed automatic-detector adapter parameters (sealed by a config digest)."""

    min_confidence: float = 0.30
    merge_xy_radius_m: float = 4.0
    crown_height_gap_m: float = 6.0
    low_structural_confidence: float = 0.50
    max_candidates: int = 999
    wire_support_min: int = 1

    def __post_init__(self) -> None:
        for name in (
            "min_confidence",
            "low_structural_confidence",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a finite value in [0, 1]")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1]")
        for name in ("merge_xy_radius_m", "crown_height_gap_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 999:
            raise ValueError("max_candidates must be an integer in [1, 999]")
        if type(self.wire_support_min) is not int or self.wire_support_min <= 0:
            raise ValueError("wire_support_min must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TOWER_AUTOMATIC_DETECTOR_SCHEMA,
            "min_confidence": self.min_confidence,
            "merge_xy_radius_m": self.merge_xy_radius_m,
            "crown_height_gap_m": self.crown_height_gap_m,
            "low_structural_confidence": self.low_structural_confidence,
            "max_candidates": self.max_candidates,
            "wire_support_min": self.wire_support_min,
        }

    def configuration_digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class AutomaticTowerRecord:
    """One recall-oriented automatic tower proposal before P### assignment."""

    source_pole_candidate_id: int
    candidate_xyz_absolute_m: tuple[float, float, float]
    base_xyz_absolute_m: tuple[float, float, float]
    top_xyz_absolute_m: tuple[float, float, float]
    height_m: float
    xy_footprint_m: float
    confidence: float
    confidence_components: dict[str, float]
    structural_evidence: dict[str, Any]
    wire_support_count: int = 0
    wire_support_linearity: float = 0.0

    @property
    def xyz_sort_key(self) -> tuple[float, float, float]:
        return self.candidate_xyz_absolute_m


def _record_from_pole(
    record: Any,
    *,
    detector_config: TowerAutomaticDetectorConfig,
) -> AutomaticTowerRecord:
    # The pole record already reports absolute coordinates (local + origin).
    base = tuple(float(value) for value in record.base_xyz_absolute_m)
    top = tuple(float(value) for value in record.top_xyz_absolute_m)
    return AutomaticTowerRecord(
        source_pole_candidate_id=int(record.pole_candidate_id),
        candidate_xyz_absolute_m=base,
        base_xyz_absolute_m=base,
        top_xyz_absolute_m=top,
        height_m=float(record.height_m),
        xy_footprint_m=float(record.xy_footprint_m),
        confidence=float(record.confidence),
        confidence_components=dict(record.confidence_components),
        structural_evidence={
            "pole_candidate_id": int(record.pole_candidate_id),
            "top_z_quantile_m": float(record.top_z_quantile_m),
            "top_z_max_m": float(record.top_z_max_m),
            "median_verticality": float(record.median_verticality),
            "median_linearity": float(record.median_linearity),
            "median_planarity": float(record.median_planarity),
            "median_scattering": float(record.median_scattering),
            "representative_count": int(record.representative_count),
            "raw_count_sum": int(record.raw_count_sum),
            "min_hag_m": float(record.min_hag_m),
            "max_hag_m": float(record.max_hag_m),
            "ground_contact_evidence": bool(record.ground_contact_evidence),
            "top_structure_representative_count": int(
                record.top_structure_representative_count
            ),
        },
    )


def _is_crown_like(record: AutomaticTowerRecord, gap_m: float) -> bool:
    """Exclude objects whose HAG extends far above the shaft (crown-like).

    A real tower's component points stop at its top; a tree/crown carries high
    HAG material well above the shaft core, so ``max_hag_m - height_m`` grows.
    This is a conservative exclusion that preserves recall for genuine towers.
    """
    max_hag = float(record.structural_evidence.get("max_hag_m", record.height_m))
    return max_hag - record.height_m > gap_m


def _apply_wire_support(
    record: AutomaticTowerRecord,
    evidence: GroundEvidence,
    wire_evidence: CoarseWireResult,
    config: FastBaselineConfig,
    detector_config: TowerAutomaticDetectorConfig,
) -> AutomaticTowerRecord:
    """Optionally incorporate real coarse wire support into the score."""
    if not wire_evidence.candidate_mask.any():
        return record
    pole_top_xy = np.asarray(record.top_xyz_absolute_m[:2], dtype=np.float64)
    pole_top_z = float(record.top_xyz_absolute_m[2])
    support_count, linearity, _, _ = _pole_wire_support(
        pole_top_xy - np.asarray(evidence.origin_xyz[:2], dtype=np.float64),
        pole_top_z - evidence.origin_xyz[2],
        evidence,
        wire_evidence,
        config,
    )
    if support_count < detector_config.wire_support_min:
        return record
    boost = min(0.15, 0.05 * support_count)
    confidence = float(min(1.0, record.confidence + boost))
    return AutomaticTowerRecord(
        source_pole_candidate_id=record.source_pole_candidate_id,
        candidate_xyz_absolute_m=record.candidate_xyz_absolute_m,
        base_xyz_absolute_m=record.base_xyz_absolute_m,
        top_xyz_absolute_m=record.top_xyz_absolute_m,
        height_m=record.height_m,
        xy_footprint_m=record.xy_footprint_m,
        confidence=confidence,
        confidence_components={
            **record.confidence_components,
            "wire_support_boost": boost,
        },
        structural_evidence=record.structural_evidence,
        wire_support_count=support_count,
        wire_support_linearity=float(linearity),
    )


def _greedy_xy_merge(
    records: list[AutomaticTowerRecord], merge_radius_m: float
) -> list[AutomaticTowerRecord]:
    """Greedy NMS: highest confidence first; drop proposals within merge radius."""
    ordered = sorted(records, key=lambda item: item.confidence, reverse=True)
    accepted: list[AutomaticTowerRecord] = []
    for record in ordered:
        if any(
            math.hypot(
                record.candidate_xyz_absolute_m[0] - other.candidate_xyz_absolute_m[0],
                record.candidate_xyz_absolute_m[1] - other.candidate_xyz_absolute_m[1],
            )
            <= merge_radius_m
            for other in accepted
        ):
            continue
        accepted.append(record)
    return accepted


def _deterministic_absolute_order(
    records: list[AutomaticTowerRecord],
) -> list[AutomaticTowerRecord]:
    """Stable, documented ordering by absolute X, then Y, then Z."""
    return sorted(records, key=lambda item: item.xyz_sort_key)


def _candidate_id_for_index(index: int) -> str:
    return f"P{index + 1:03d}"


def _candidate_bounds(record: AutomaticTowerRecord) -> TowerCandidateBounds:
    base = record.base_xyz_absolute_m
    top = record.top_xyz_absolute_m
    half = record.xy_footprint_m / 2.0
    return TowerCandidateBounds(
        min_xyz_absolute_m=(
            min(base[0], top[0]) - half,
            min(base[1], top[1]) - half,
            base[2],
        ),
        max_xyz_absolute_m=(
            max(base[0], top[0]) + half,
            max(base[1], top[1]) + half,
            top[2],
        ),
    )


def _detector_configuration_digest(
    detector_config: TowerAutomaticDetectorConfig, config: FastBaselineConfig
) -> str:
    payload = {
        "detector": detector_config.to_dict(),
        "pole": {
            name: getattr(config, name)
            for name in (
                "pole_minimum_height_m",
                "pole_maximum_xy_width_m",
                "pole_verticality_min",
                "pole_linearity_min",
                "pole_planarity_max",
                "pole_min_support_representatives",
                "pole_ground_contact_tolerance_m",
                "pole_group_cell_m",
                "pole_xy_footprint_cap_m",
                "pole_top_wire_support_radius_m",
                "pole_wire_approach_hag_tolerance_m",
            )
        },
    }
    return canonical_digest(payload)


def detect_tower_candidates(
    evidence: GroundEvidence,
    config: FastBaselineConfig,
    *,
    source_binding: SourceIdentityBinding,
    coordinate_context: CoordinateContext,
    project_id: str,
    candidate_set_id: str,
    generated_at: str,
    detector_config: TowerAutomaticDetectorConfig | None = None,
    wire_evidence: CoarseWireResult | None = None,
) -> TowerCandidateSet:
    """Run the automatic detector and return one deterministic candidate set.

    No ``T###`` is ever allocated here.  Candidate IDs are assigned canonically
    in deterministic absolute-coordinate order (P001..P999); the same scientific
    inputs always produce the same ordered proposal content.
    """
    if type(evidence) is not GroundEvidence:
        raise ValueError("evidence must be a GroundEvidence")
    if type(config) is not FastBaselineConfig:
        raise ValueError("config must be a FastBaselineConfig")
    if type(source_binding) is not SourceIdentityBinding:
        raise ValueError("source_binding must be a SourceIdentityBinding")
    if type(coordinate_context) is not CoordinateContext:
        raise ValueError("coordinate_context must be a CoordinateContext")
    if coordinate_context != source_binding.coordinate_context:
        raise ValueError("coordinate_context must match source_binding")
    if wire_evidence is not None and type(wire_evidence) is not CoarseWireResult:
        raise ValueError("wire_evidence must be a CoarseWireResult or null")
    if detector_config is None:
        detector_config = TowerAutomaticDetectorConfig()
    if type(detector_config) is not TowerAutomaticDetectorConfig:
        raise ValueError("detector_config must be a TowerAutomaticDetectorConfig")

    pole = detect_pole_candidates(evidence, config)

    records: list[AutomaticTowerRecord] = []
    for record in pole.records:
        proposal = _record_from_pole(record, detector_config=detector_config)
        if proposal.confidence < detector_config.min_confidence:
            continue
        if _is_crown_like(proposal, detector_config.crown_height_gap_m):
            continue
        records.append(proposal)

    wire_used = False
    if wire_evidence is not None:
        boosted: list[AutomaticTowerRecord] = []
        for proposal in records:
            with_support = _apply_wire_support(
                proposal, evidence, wire_evidence, config, detector_config
            )
            if with_support.wire_support_count > 0:
                wire_used = True
            boosted.append(with_support)
        records = boosted

    merged = _greedy_xy_merge(records, detector_config.merge_xy_radius_m)
    ordered = _deterministic_absolute_order(merged)
    ordered = ordered[: detector_config.max_candidates]

    configuration_digest = _detector_configuration_digest(detector_config, config)
    detector_kind = DetectorKind.HYBRID if wire_used else DetectorKind.GEOMETRY
    # Deterministic evidence only: the immutable candidate-set content digest
    # must never depend on wall-clock runtime.
    pole_diagnostics = {
        key: value for key, value in pole.diagnostics.items() if key != "runtime_seconds"
    }
    detector_provenance = DetectorProvenance(
        detector_kind=detector_kind,
        detector_name=TOWER_AUTOMATIC_DETECTOR_NAME,
        detector_version=TOWER_AUTOMATIC_DETECTOR_VERSION,
        configuration_digest=configuration_digest,
        detector_evidence={
            "schema": TOWER_AUTOMATIC_DETECTOR_SCHEMA,
            "pole_detector_diagnostics": pole_diagnostics,
            "filtered_pole_candidate_count": len(records),
            "nms_merge_xy_radius_m": detector_config.merge_xy_radius_m,
            "wire_support_incorporated": wire_used,
            "candidate_count": len(ordered),
        },
    )

    candidates: list[TowerCandidate] = []
    for index, proposal in enumerate(ordered):
        quality_flags = TowerCandidateQualityFlag.PROPOSAL_NEEDS_REFINEMENT
        if proposal.confidence < detector_config.low_structural_confidence:
            quality_flags |= TowerCandidateQualityFlag.LOW_STRUCTURAL_CONFIDENCE
        evidence_payload: dict[str, Any] = {
            "structural": dict(proposal.structural_evidence),
            "confidence_components": dict(proposal.confidence_components),
            "height_m": proposal.height_m,
            "xy_footprint_m": proposal.xy_footprint_m,
        }
        if proposal.wire_support_count > 0:
            evidence_payload["wire"] = {
                "support_count": proposal.wire_support_count,
                "support_linearity": proposal.wire_support_linearity,
            }
        candidates.append(
            TowerCandidate(
                candidate_id=_candidate_id_for_index(index),
                origin=CandidateOrigin.AUTOMATIC,
                coordinate_frame=coordinate_context.frame,
                coordinate_unit=coordinate_context.units,
                candidate_xyz_absolute_m=proposal.candidate_xyz_absolute_m,
                candidate_bounds_absolute_m=_candidate_bounds(proposal),
                confidence=proposal.confidence,
                quality_flags=quality_flags,
                detector_provenance=detector_provenance,
                evidence=evidence_payload,
            )
        )

    return TowerCandidateSet.create(
        candidate_set_id=candidate_set_id,
        project_id=project_id,
        generated_at=generated_at,
        source_binding=source_binding,
        coordinate_context=coordinate_context,
        detector_provenance=detector_provenance,
        candidates=candidates,
        corridor_bounds_absolute_m=None,
    )


__all__ = [
    "TOWER_AUTOMATIC_DETECTOR_NAME",
    "TOWER_AUTOMATIC_DETECTOR_SCHEMA",
    "TOWER_AUTOMATIC_DETECTOR_VERSION",
    "AutomaticTowerRecord",
    "TowerAutomaticDetectorConfig",
    "detect_tower_candidates",
]
