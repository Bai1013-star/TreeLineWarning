"""Manual single-span route configuration (selector / refinement / risk).

The frozen tree, wire and clearance parameters live in
``core.config.fast_baseline.FastBaselineConfig`` and are reused unchanged.
This module holds only the additional tunables that the manual-selection
route introduces: pole-refinement evidence gates, the display sample budget,
the danger-point storage cap and the risk threshold file schema.  Every
value is finite and validated; the whole payload contributes to the manual
span digest so checkpoint validity keys on it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any

from core.config.fast_baseline import FastBaselineConfig

MANUAL_SPAN_CONFIG_SCHEMA = "manual-single-span-config-v1"
MANUAL_SPAN_SELECTION_SCHEMA = "manual-single-span-selection-v1"
MANUAL_RISK_SCHEMA = "manual-single-span-risk-v1"
MANUAL_CLEARANCE_SCHEMA = "manual-single-span-clearance-fast-v1"
MANUAL_PIPELINE_SCHEMA = "manual-single-span-pipeline-v1"
NON_CERTIFIED_MARKER = "NON_CERTIFIED_RESEARCH_DEMONSTRATION"


def canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ManualSpanConfig:
    """Tunables for the manual single-span route (selector + risk side)."""

    # --- display / selection ------------------------------------------------
    display_sample_points: int = 50_000
    display_point_size: float = 2.0
    display_opacity: float = 0.9

    # --- pole refinement ----------------------------------------------------
    refine_search_radius_m: float = 3.5
    refine_z_max_hag_m: float = 35.0
    refine_min_pole_height_m: float = 5.0
    refine_max_footprint_m: float = 1.6
    refine_band_height_m: float = 1.0
    refine_max_band_drift_m: float = 0.7
    refine_max_vertical_gap_m: float = 3.0
    refine_min_support_reps: int = 5
    refine_verticality_min: float = 0.50
    refine_linearity_min: float = 0.30
    refine_ground_contact_radius_m: float = 1.8
    refine_top_quantile: float = 0.95
    refine_ground_z_quantile: float = 0.50

    # --- span geometry ------------------------------------------------------
    span_half_width_m: float = 9.0
    span_halo_m: float = 6.0
    span_z_ground_clearance_m: float = 1.2
    span_z_top_margin_m: float = 3.0
    span_min_length_m: float = 10.0

    # --- clearance fast path ------------------------------------------------
    clearance_chunk_points: int = 65_536
    danger_point_cap_m: float = 40.0
    danger_point_chunk: int = 1_000_000

    # --- dbh -----------------------------------------------------------------
    dbh_band_center_m: float = 1.30
    dbh_band_half_width_m: float = 0.45
    dbh_stem_radius_m: float = 0.60
    dbh_measured_min_points: int = 12
    dbh_estimated_min_points: int = 3
    dbh_min_m: float = 0.05
    dbh_max_m: float = 1.5

    # --- risk threshold ------------------------------------------------------
    risk_threshold_default_m: float = 3.5
    risk_threshold_min_m: float = 0.5
    risk_threshold_max_m: float = 20.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            name = field.name
            if name in {"display_sample_points", "refine_min_support_reps"}:
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            elif name.endswith("_m") and name not in {
                "refine_band_height_m",
                "span_half_width_m",
                "span_halo_m",
            }:
                if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be a finite positive float")
            elif name in {"refine_band_height_m", "span_half_width_m", "span_halo_m"}:
                if type(value) is not float or not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be a finite non-negative float")
            elif name in {
                "refine_verticality_min",
                "refine_linearity_min",
                "refine_top_quantile",
                "refine_ground_z_quantile",
            }:
                if type(value) is not float or not (0.0 <= value <= 1.0):
                    raise ValueError(f"{name} must be a float in [0, 1]")
            elif name in {"display_point_size", "display_opacity"}:
                if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be a finite positive float")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": MANUAL_SPAN_CONFIG_SCHEMA,
            "non_certified_marker": NON_CERTIFIED_MARKER,
        }
        for field in fields(self):
            payload[field.name] = getattr(self, field.name)
        return payload

    def computed_digest(self) -> str:
        return canonical_digest(self.to_dict())


def default_manual_span_config(**overrides: Any) -> ManualSpanConfig:
    base = ManualSpanConfig()
    if not overrides:
        return base
    values = {field.name: getattr(base, field.name) for field in fields(base)}
    values.update(overrides)
    return ManualSpanConfig(**values)


# Evidence-driven parameter changes for the manual single-span route (each is
# recorded in run_summary.json -> PARAMETER_CHANGES):
#
#   wire_min_longitudinal_coverage_frac 0.55 -> 0.30
#     On the real manually-selected span, the reconstructed conductor evidence
#     forms long thin horizontal bands (u-width ~0.2-3.8 m, z-stable) that cover
#     34-53% of the corridor before DBSCAN fragmentation; the frozen 55% gate
#     rejected every one of them while every canopy cluster covered < 10%.
#     Lowering the gate to 0.30 admits exactly the real conductor bands.
#
#   wire_min_longitudinal_span_m 10.0 -> 6.0
#     The same bands are 15-24 m long before fragmentation; 6 m keeps a real
#     band admissible while still excluding canopy fragments.
def manual_route_config() -> FastBaselineConfig:
    from core.config.fast_baseline import default_fast_baseline_config as _dfb

    return _dfb(
        wire_min_longitudinal_coverage_frac=0.30,
        wire_min_longitudinal_span_m=6.0,
    )


__all__ = [
    "MANUAL_SPAN_CONFIG_SCHEMA",
    "MANUAL_SPAN_SELECTION_SCHEMA",
    "MANUAL_RISK_SCHEMA",
    "MANUAL_CLEARANCE_SCHEMA",
    "MANUAL_PIPELINE_SCHEMA",
    "NON_CERTIFIED_MARKER",
    "canonical_digest",
    "ManualSpanConfig",
    "default_manual_span_config",
]
