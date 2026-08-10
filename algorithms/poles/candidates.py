"""FAST mature pole-like candidate detection on coarse representatives.

The detector finds *pole-like vertical objects* — vertical, tall, narrow,
near-ground-continuous structures — and explicitly does NOT yet claim any of
them is a power pole.  Confirmation happens later through pole-wire mutual
validation.  This stage reports rich per-candidate diagnostics so tree trunks,
street lights and signs can be separated from real utility poles downstream.

Geometry evidence (all computed on the coarse routing representatives):

- strong vertical orientation (PDAL Verticality);
- narrow / cylindrical structure (Linearity high, Planarity low);
- high vertical extent (height);
- small XY footprint;
- continuity from near-ground toward upper levels (26-connected voxel column);
- meaningful HAG and sufficient representative support.

A candidate is accepted only when all gates pass; the confidence is a
transparent combination of the individual evidence components.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.poles.evidence import GroundEvidence
from algorithms.vegetation.fast_vegetation_v3 import _grid_component_labels

POLE_CANDIDATE_SCHEMA = "fast-baseline-pole-candidates-v1"


@dataclass(frozen=True, slots=True)
class PoleCandidateRecord:
    """One pole-like vertical-object candidate with full diagnostics."""

    pole_candidate_id: int
    base_xyz_local_m: tuple[float, float, float]
    base_xyz_absolute_m: tuple[float, float, float]
    top_xyz_local_m: tuple[float, float, float]
    top_xyz_absolute_m: tuple[float, float, float]
    top_z_quantile_m: float
    top_z_max_m: float
    height_m: float
    xy_footprint_m: float
    median_verticality: float
    median_linearity: float
    median_planarity: float
    median_scattering: float
    representative_count: int
    raw_count_sum: int
    min_hag_m: float
    max_hag_m: float
    ground_contact_evidence: bool
    top_structure_representative_count: int
    confidence: float
    confidence_components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pole_candidate_id": self.pole_candidate_id,
            "base_xyz_local_m": list(self.base_xyz_local_m),
            "base_xyz_absolute_m": list(self.base_xyz_absolute_m),
            "top_xyz_local_m": list(self.top_xyz_local_m),
            "top_xyz_absolute_m": list(self.top_xyz_absolute_m),
            "top_z_quantile_m": self.top_z_quantile_m,
            "top_z_max_m": self.top_z_max_m,
            "height_m": self.height_m,
            "xy_footprint_m": self.xy_footprint_m,
            "median_verticality": self.median_verticality,
            "median_linearity": self.median_linearity,
            "median_planarity": self.median_planarity,
            "median_scattering": self.median_scattering,
            "representative_count": self.representative_count,
            "raw_count_sum": self.raw_count_sum,
            "min_hag_m": self.min_hag_m,
            "max_hag_m": self.max_hag_m,
            "ground_contact_evidence": self.ground_contact_evidence,
            "top_structure_representative_count": self.top_structure_representative_count,
            "confidence": self.confidence,
            "confidence_components": dict(self.confidence_components),
        }


@dataclass(frozen=True, slots=True)
class PoleCandidateResult:
    """Deterministic pole-like candidate set plus per-representative labels."""

    records: tuple[PoleCandidateRecord, ...]
    candidate_id_by_representative: np.ndarray  # int32; -1 when not pole-like seed
    seed_mask: np.ndarray  # bool
    component_label_by_representative: np.ndarray  # int32; 0 = no pole component
    diagnostics: dict[str, Any]

    @property
    def candidate_count(self) -> int:
        return len(self.records)

    @classmethod
    def from_records_dict(
        cls,
        payload: dict[str, Any],
        *,
        candidate_id_by_representative: np.ndarray,
        seed_mask: np.ndarray,
        component_label_by_representative: np.ndarray,
    ) -> PoleCandidateResult:
        records = tuple(
            PoleCandidateRecord(
                pole_candidate_id=int(item["pole_candidate_id"]),
                base_xyz_local_m=tuple(float(value) for value in item["base_xyz_local_m"]),
                base_xyz_absolute_m=tuple(
                    float(value) for value in item["base_xyz_absolute_m"]
                ),
                top_xyz_local_m=tuple(float(value) for value in item["top_xyz_local_m"]),
                top_xyz_absolute_m=tuple(
                    float(value) for value in item["top_xyz_absolute_m"]
                ),
                top_z_quantile_m=float(
                    item.get("top_z_quantile_m", item["top_xyz_local_m"][2])
                ),
                top_z_max_m=float(
                    item.get("top_z_max_m", item["top_xyz_local_m"][2])
                ),
                height_m=float(item["height_m"]),
                xy_footprint_m=float(item["xy_footprint_m"]),
                median_verticality=float(item["median_verticality"]),
                median_linearity=float(item["median_linearity"]),
                median_planarity=float(item["median_planarity"]),
                median_scattering=float(item["median_scattering"]),
                representative_count=int(item["representative_count"]),
                raw_count_sum=int(item["raw_count_sum"]),
                min_hag_m=float(item["min_hag_m"]),
                max_hag_m=float(item["max_hag_m"]),
                ground_contact_evidence=bool(item["ground_contact_evidence"]),
                top_structure_representative_count=int(
                    item["top_structure_representative_count"]
                ),
                confidence=float(item["confidence"]),
                confidence_components=dict(item["confidence_components"]),
            )
            for item in payload["candidates"]
        )
        return cls(
            records=records,
            candidate_id_by_representative=candidate_id_by_representative,
            seed_mask=seed_mask,
            component_label_by_representative=component_label_by_representative,
            diagnostics=dict(payload.get("diagnostics", {})),
        )


def _component_stats(
    xyz_local: np.ndarray,
    hag: np.ndarray,
    linearity: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
    verticality: np.ndarray,
    raw_count: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    points = xyz_local[rows]
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    center = points.mean(axis=0)
    footprint = float(np.max(np.linalg.norm(points[:, :2] - center[:2], axis=1)))
    # ``order`` indexes into the local ``points`` array (0..len-1).
    order = np.argsort(points[:, 2], kind="stable")
    if len(order) > 2:
        top_positions = order[-max(1, len(order) // 8):]
        base_positions = order[: max(1, len(order) // 8)]
    else:
        top_positions = order
        base_positions = order
    base_xyz = points[base_positions].mean(axis=0)
    top_xyz = points[top_positions].mean(axis=0)
    return {
        "z_min": z_min,
        "z_max": z_max,
        "height_m": z_max - z_min,
        "top_z_quantile_m": float(np.quantile(points[:, 2], 0.90)),
        "top_z_max_m": z_max,
        "xy_footprint_m": footprint,
        "base_xyz": base_xyz,
        "top_xyz": top_xyz,
        "center_xyz": center,
        "min_hag_m": float(np.min(hag[rows])),
        "max_hag_m": float(np.max(hag[rows])),
        "median_verticality": float(np.median(verticality[rows])),
        "median_linearity": float(np.median(linearity[rows])),
        "median_planarity": float(np.median(planarity[rows])),
        "median_scattering": float(np.median(scattering[rows])),
        "representative_count": int(len(rows)),
        "raw_count_sum": int(np.sum(raw_count[rows], dtype=np.int64)),
    }


def detect_pole_candidates(
    evidence: GroundEvidence,
    config: FastBaselineConfig,
) -> PoleCandidateResult:
    """Detect pole-like vertical-object candidates on the coarse evidence.

    The candidate mask marks the *seed* (pole-shaft) representatives belonging
    to each accepted candidate.  Confidence components are reported explicitly
    and are never silently blended into a single opaque score.
    """

    started = time.perf_counter()
    non_ground = evidence.non_ground_mask()
    seed = (
        non_ground
        & np.isfinite(evidence.hag_m)
        & (evidence.hag_m >= 0.0)
        & (evidence.main_direction_z >= config.pole_verticality_min)
        & (evidence.linearity >= config.pole_linearity_min)
        & (evidence.planarity <= config.pole_planarity_max)
    )
    labels, _cell, _origin = _grid_component_labels(
        evidence.xyz_local_m, seed, config.pole_group_cell_m
    )
    records: list[PoleCandidateRecord] = []
    candidate_id_by_rep = np.full(evidence.count, -1, dtype=np.int32)
    component_ids = np.unique(labels[labels > 0])
    accepted_components: list[int] = []
    for component_id in component_ids:
        rows = np.flatnonzero(labels == component_id)
        if len(rows) < config.pole_min_support_representatives:
            continue
        stats = _component_stats(
            evidence.xyz_local_m,
            evidence.hag_m,
            evidence.linearity,
            evidence.planarity,
            evidence.scattering,
            evidence.verticality,
            evidence.raw_count,
            rows,
        )
        if stats["height_m"] < config.pole_minimum_height_m:
            continue
        if stats["xy_footprint_m"] > config.pole_maximum_xy_width_m:
            continue
        if stats["xy_footprint_m"] > config.pole_xy_footprint_cap_m:
            continue
        if stats["min_hag_m"] > config.pole_ground_contact_tolerance_m:
            continue
        accepted_components.append(int(component_id))

    # top-structure evidence: non-ground representatives within the support
    # radius around each accepted pole top.
    if accepted_components:
        top_rows = []
        for _component_id in accepted_components:
            rows = np.flatnonzero(labels == _component_id)
            stats = _component_stats(
                evidence.xyz_local_m,
                evidence.hag_m,
                evidence.linearity,
                evidence.planarity,
                evidence.scattering,
                evidence.verticality,
                evidence.raw_count,
                rows,
            )
            top_rows.append((int(_component_id), rows, stats))
        for _position, (_component_id, rows, stats) in enumerate(top_rows):
            top_xyz = stats["top_xyz"]
            support = np.flatnonzero(
                non_ground
                & (
                    np.linalg.norm(
                        evidence.xyz_local_m[:, :2] - top_xyz[:2], axis=1
                    )
                    <= config.pole_top_wire_support_radius_m
                )
            )
            top_structure = int(
                np.count_nonzero(
                    evidence.xyz_local_m[support, 2]
                    >= stats["top_xyz"][2] - config.pole_wire_approach_hag_tolerance_m
                )
            )
            base_xyz_local = stats["base_xyz"]
            top_xyz_local = stats["top_xyz"]
            verticality_evidence = float(
                min(1.0, max(0.0, stats["median_verticality"] / config.pole_verticality_min))
            )
            linearity_evidence = float(
                min(1.0, max(0.0, stats["median_linearity"] / config.pole_linearity_min))
            )
            height_evidence = float(
                min(1.0, max(0.0, stats["height_m"] / (1.6 * config.pole_minimum_height_m)))
            )
            narrowness_evidence = float(
                max(0.0, min(1.0, 1.0 - stats["xy_footprint_m"] / config.pole_maximum_xy_width_m))
            )
            ground_contact = bool(stats["min_hag_m"] <= config.pole_ground_contact_tolerance_m)
            ground_contact_evidence = 1.0 if ground_contact else 0.0
            support_evidence = float(
                min(
                    1.0,
                    stats["representative_count"]
                    / (3.0 * config.pole_min_support_representatives),
                )
            )
            components = {
                "verticality_evidence": verticality_evidence,
                "linearity_evidence": linearity_evidence,
                "height_evidence": height_evidence,
                "narrowness_evidence": narrowness_evidence,
                "ground_contact_evidence": ground_contact_evidence,
                "support_evidence": support_evidence,
            }
            weights = {
                "verticality_evidence": 0.25,
                "linearity_evidence": 0.20,
                "height_evidence": 0.20,
                "narrowness_evidence": 0.15,
                "ground_contact_evidence": 0.10,
                "support_evidence": 0.10,
            }
            confidence = float(
                sum(weights[name] * components[name] for name in weights)
            )
            origin = np.asarray(evidence.origin_xyz, dtype=np.float64)
            records.append(
                PoleCandidateRecord(
                    pole_candidate_id=_position,
                    base_xyz_local_m=tuple(float(value) for value in base_xyz_local),
                    base_xyz_absolute_m=tuple(float(value) for value in base_xyz_local + origin),
                    top_xyz_local_m=tuple(float(value) for value in top_xyz_local),
                    top_xyz_absolute_m=tuple(float(value) for value in top_xyz_local + origin),
                    top_z_quantile_m=float(stats["top_z_quantile_m"] + origin[2]),
                    top_z_max_m=float(stats["top_z_max_m"] + origin[2]),
                    height_m=stats["height_m"],
                    xy_footprint_m=stats["xy_footprint_m"],
                    median_verticality=stats["median_verticality"],
                    median_linearity=stats["median_linearity"],
                    median_planarity=stats["median_planarity"],
                    median_scattering=stats["median_scattering"],
                    representative_count=stats["representative_count"],
                    raw_count_sum=stats["raw_count_sum"],
                    min_hag_m=stats["min_hag_m"],
                    max_hag_m=stats["max_hag_m"],
                    ground_contact_evidence=ground_contact,
                    top_structure_representative_count=top_structure,
                    confidence=confidence,
                    confidence_components=components,
                )
            )
            candidate_id_by_rep[rows] = _position
    return PoleCandidateResult(
        records=tuple(records),
        candidate_id_by_representative=candidate_id_by_rep,
        seed_mask=seed,
        component_label_by_representative=labels,
        diagnostics={
            "schema": POLE_CANDIDATE_SCHEMA,
            "seed_representative_count": int(np.count_nonzero(seed)),
            "component_count": int(len(component_ids)),
            "candidate_count": len(records),
            "accepted_component_count": len(accepted_components),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def pole_candidates_to_dict(result: PoleCandidateResult) -> dict[str, Any]:
    return {
        "schema": POLE_CANDIDATE_SCHEMA,
        "candidate_count": result.candidate_count,
        "candidates": [record.to_dict() for record in result.records],
        "diagnostics": dict(result.diagnostics),
    }


__all__ = [
    "POLE_CANDIDATE_SCHEMA",
    "PoleCandidateRecord",
    "PoleCandidateResult",
    "detect_pole_candidates",
    "pole_candidates_to_dict",
]
