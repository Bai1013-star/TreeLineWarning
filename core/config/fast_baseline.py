"""Single source of truth for the pole-first / span-based FAST mature baseline.

All tunable parameters for the ``POLE_SPAN_MATURE_BASELINE`` route live in one
frozen, digest-bound structure.  The philosophy (owner decision) is defensible
first-pass defaults chosen by the implementation writer; aggressive parameter
sweeps are deferred to a later Codex tuning stage.  Every parameter has a
finite, validated value so downstream stages never read magic constants.

This module is part of the FAST NON_CERTIFIED research route and never claims
Formal authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any

FAST_BASELINE_SCHEMA_VERSION = "pole-span-mature-baseline-v1"
NON_CERTIFIED_MARKER = "NON_CERTIFIED_RESEARCH_DEMONSTRATION"

_ROUTE_MODE_POLE_SPAN = "POLE_SPAN_MATURE_BASELINE"
_ROUTE_MODE_EXPERIMENTAL_V4 = "EXPERIMENTAL_V4"


def canonical_digest(payload: dict[str, Any]) -> str:
    """Stable SHA-256 over sorted canonical JSON (same convention as FAST route)."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _positive_float(value: float, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive float")
    return value


def _non_negative_float(value: float, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    return value


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _fraction(value: float, name: str) -> float:
    value = _non_negative_float(value, name)
    if value > 1.0:
        raise ValueError(f"{name} must be at most 1.0")
    return value


@dataclass(frozen=True, slots=True)
class FastBaselineConfig:
    """Central configuration for the pole-first span-based FAST baseline.

    Binds every tunable parameter of the pipeline and carries a canonical
    ``config_digest`` that checkpoint validity keys on.  No downstream stage
    may scatter its own magic constants; it must read this structure.
    """

    # --- shared representation scales --------------------------------------
    vegetation_fine_voxel_size_m: float = 0.10
    wire_fine_voxel_size_m: float = 0.10

    # --- pole candidate detection ------------------------------------------
    pole_minimum_height_m: float = 5.0
    pole_maximum_xy_width_m: float = 1.6
    pole_verticality_min: float = 0.60
    pole_linearity_min: float = 0.35
    pole_planarity_max: float = 0.55
    pole_min_support_representatives: int = 6
    pole_ground_contact_tolerance_m: float = 1.0
    pole_group_cell_m: float = 1.0
    pole_xy_footprint_cap_m: float = 2.0

    # --- coarse wire evidence ----------------------------------------------
    wire_coarse_min_hag_m: float = 5.0
    wire_coarse_linearity_min: float = 0.72
    wire_coarse_scattering_max: float = 0.12
    wire_coarse_verticality_max: float = 0.45
    wire_coarse_group_cell_m: float = 1.0
    wire_coarse_min_segment_span_m: float = 12.0
    wire_coarse_min_segment_points: int = 6
    wire_coarse_robust_z_quantile_fraction: float = 0.90
    wire_coarse_residual_quantile_fraction: float = 0.95
    wire_coarse_max_robust_z_span_m: float = 4.0
    wire_coarse_max_z_fit_residual_m: float = 2.0
    wire_coarse_max_transverse_width_m: float = 8.0
    wire_coarse_min_longitudinal_coverage_fraction: float = 0.70
    wire_coarse_longitudinal_coverage_bins: int = 8

    # --- pole-wire mutual validation ---------------------------------------
    pole_top_wire_support_radius_m: float = 5.0
    pole_wire_approach_hag_tolerance_m: float = 4.0
    pole_min_wire_support_reps: int = 6
    pole_min_wire_support_parallelism: float = 0.35
    pole_attachment_lower_quantile: float = 0.10
    pole_attachment_upper_quantile: float = 0.90
    pole_attachment_fallback_half_band_m: float = 1.5
    pole_min_wire_support_segment_span_m: float = 30.0
    pole_crown_penalty_radius_m: float = 4.0
    pole_crown_penalty_min_hag_m: float = 6.0

    # --- topology / ordering ------------------------------------------------
    pole_min_span_m: float = 15.0
    pole_max_span_m: float = 220.0
    topology_direction_tolerance_rad: float = 0.61  # ~35 deg
    topology_heading_alignment_min: float = 0.80  # cos >= this to continue a line
    topology_attachment_height_tolerance_m: float = 2.5
    topology_min_corridor_continuity: float = 0.50
    topology_corridor_gap_max_m: float = 4.0
    topology_corridor_gap_fraction: float = 0.08
    span_half_width_m: float = 9.0
    span_halo_m: float = 6.0
    span_z_ground_clearance_m: float = 1.2
    span_z_top_margin_m: float = 3.0
    span_min_length_m: float = 10.0

    # --- CHM / tree pipeline -----------------------------------------------
    chm_resolution_m: float = 0.50
    chm_minimum_hag_m: float = 1.50
    chm_smooth_sigma_m: float = 0.60
    chm_smooth_kernel_radius_m: float = 1.50
    tree_minimum_hag_m: float = 2.0
    tree_top_local_radius_m: float = 0.60
    tree_top_min_prominence_m: float = 1.2
    tree_top_min_prominence_fraction: float = 0.10
    tree_top_prominence_annulus_outer_m: float = 2.50
    tree_top_min_support_points: int = 4
    # height -> suppression radius: radius = min(radius_cap, radius0 + slope*height)
    tree_top_radius0_m: float = 1.20
    tree_top_radius_per_meter: float = 0.10
    tree_top_radius_cap_m: float = 5.0
    # A lower peak inside a taller crown's adaptive window is not discarded
    # blindly.  It may be a neighbouring tree when its height/support and the
    # CHM saddle between the two peaks provide independent-crown evidence.
    tree_top_suppressed_peak_min_height_ratio: float = 0.55
    tree_top_suppressed_peak_min_saddle_drop_m: float = 1.20
    tree_top_suppressed_peak_min_path_occupancy_fraction: float = 0.80
    tree_min_crown_area_m2: float = 1.0
    watershed_mask_min_occupancy: int = 1
    tree_membership_hag_tolerance_m: float = 1.20
    tree_min_final_members: int = 6

    # --- per-span wire pipeline --------------------------------------------
    wire_min_hag_m: float = 3.0
    wire_linearity_min: float = 0.72
    wire_scattering_max: float = 0.12
    wire_verticality_max: float = 0.50
    wire_pca_neighbor_count: int = 16
    wire_dbscan_eps_m: float = 0.80
    wire_dbscan_min_samples: int = 5
    wire_strand_u_separation_m: float = 0.60
    wire_strand_z_separation_m: float = 0.60
    wire_min_longitudinal_coverage_frac: float = 0.55
    wire_min_longitudinal_span_m: float = 10.0
    wire_min_strand_points: int = 8
    wire_residual_tolerance_m: float = 0.45
    wire_curve_sample_spacing_m: float = 0.50
    wire_max_curve_samples: int = 1200
    wire_approach_pole_radius_m: float = 5.0
    wire_endpoint_hag_tolerance_m: float = 4.0

    # --- clearance ----------------------------------------------------------
    clearance_bvh_leaf_cap: int = 16
    clearance_tie_tolerance_m: float = 1e-9

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_m") and field.name not in {
                "clearance_tie_tolerance_m",
            }:
                if field.name.endswith("_frac"):
                    object.__setattr__(self, field.name, _fraction(value, field.name))
                elif field.name in {"chm_smooth_sigma_m", "tree_top_radius0_m"}:
                    object.__setattr__(
                        self, field.name, _non_negative_float(value, field.name)
                    )
                else:
                    object.__setattr__(self, field.name, _positive_float(value, field.name))
            elif field.name in {
                "pole_min_support_representatives",
                "pole_min_wire_support_reps",
                "wire_coarse_min_segment_points",
                "wire_coarse_longitudinal_coverage_bins",
                "wire_pca_neighbor_count",
                "wire_dbscan_min_samples",
                "wire_min_strand_points",
                "wire_max_curve_samples",
                "tree_top_min_support_points",
                "tree_min_final_members",
                "watershed_mask_min_occupancy",
                "clearance_bvh_leaf_cap",
            }:
                object.__setattr__(self, field.name, _positive_int(value, field.name))
        if self.tree_top_radius0_m < 0.0 or not math.isfinite(self.tree_top_radius0_m):
            raise ValueError("tree_top_radius0_m must be finite and non-negative")
        if self.tree_top_radius_cap_m < self.tree_top_radius0_m:
            raise ValueError("tree_top_radius_cap_m must be >= tree_top_radius0_m")
        object.__setattr__(
            self,
            "tree_top_min_prominence_fraction",
            _fraction(self.tree_top_min_prominence_fraction, "tree_top_min_prominence_fraction"),
        )
        for name in {
            "wire_coarse_robust_z_quantile_fraction",
            "tree_top_suppressed_peak_min_height_ratio",
            "tree_top_suppressed_peak_min_path_occupancy_fraction",
            "wire_coarse_residual_quantile_fraction",
            "wire_coarse_min_longitudinal_coverage_fraction",
            "pole_attachment_lower_quantile",
            "pole_attachment_upper_quantile",
            "topology_min_corridor_continuity",
            "topology_corridor_gap_fraction",
        }:
            object.__setattr__(self, name, _fraction(getattr(self, name), name))
        if self.pole_attachment_lower_quantile >= self.pole_attachment_upper_quantile:
            raise ValueError("pole attachment quantiles must be strictly ordered")
        if (
            self.topology_direction_tolerance_rad <= 0.0
            or self.topology_direction_tolerance_rad >= math.pi
        ):
            raise ValueError("topology_direction_tolerance_rad must be in (0, pi)")
        if not (0.0 <= self.topology_heading_alignment_min <= 1.0):
            raise ValueError("topology_heading_alignment_min must be in [0, 1]")
        if self.span_halo_m <= 0.0:
            raise ValueError("span_halo_m must be positive")
        if self.span_half_width_m <= 0.0:
            raise ValueError("span_half_width_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": FAST_BASELINE_SCHEMA_VERSION,
            "non_certified_marker": NON_CERTIFIED_MARKER,
        }
        for field in fields(self):
            payload[field.name] = getattr(self, field.name)
        payload["config_digest"] = self.computed_digest()
        return payload

    def to_digest_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": FAST_BASELINE_SCHEMA_VERSION,
            "non_certified_marker": NON_CERTIFIED_MARKER,
        }
        for field in fields(self):
            payload[field.name] = getattr(self, field.name)
        return payload

    def computed_digest(self) -> str:
        return canonical_digest(self.to_digest_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FastBaselineConfig:
        allowed = {field.name for field in fields(cls)}
        values: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"schema_version", "non_certified_marker", "config_digest"}:
                continue
            if key not in allowed:
                raise ValueError(f"unknown FastBaselineConfig field: {key}")
            values[key] = value
        return cls(**values)

    def route(self) -> str:
        """Return the route mode marker used by the one-click runner."""

        return _ROUTE_MODE_POLE_SPAN


def default_fast_baseline_config(**overrides: Any) -> FastBaselineConfig:
    """Return the canonical first-pass baseline config with deterministic defaults."""

    base = FastBaselineConfig()
    if not overrides:
        return base
    values = {field.name: getattr(base, field.name) for field in fields(base)}
    values.update(overrides)
    return FastBaselineConfig(**values)


def height_to_suppression_radius_m(config: FastBaselineConfig, height_m: float) -> float:
    """Adaptive tree-top suppression window radius from a peak height.

    Shorter trees -> smaller expected crown -> smaller radius; taller trees ->
    larger expected crown -> larger radius, capped for very tall trees.  This
    is the single centrally-configured height -> window mapping.
    """

    if not math.isfinite(height_m) or height_m < 0.0:
        raise ValueError("height_m must be a finite non-negative height")
    radius = config.tree_top_radius0_m + config.tree_top_radius_per_meter * height_m
    return float(min(radius, config.tree_top_radius_cap_m))


__all__ = [
    "FAST_BASELINE_SCHEMA_VERSION",
    "NON_CERTIFIED_MARKER",
    "_ROUTE_MODE_POLE_SPAN",
    "_ROUTE_MODE_EXPERIMENTAL_V4",
    "canonical_digest",
    "FastBaselineConfig",
    "default_fast_baseline_config",
    "height_to_suppression_radius_m",
]
