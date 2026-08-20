"""GIS ↔ point-cloud registration state machine.

Only two anchors can establish a local 2-D rigid transform.  One anchor is
explicitly treated as assisted matching, never as a unique alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RegistrationState(str, Enum):
    UNREGISTERED = "UNREGISTERED"
    GEOREFERENCED = "GEOREFERENCED"
    ASSISTED_ONE_ANCHOR = "ASSISTED_ONE_ANCHOR"
    REGISTERED_TWO_ANCHOR = "REGISTERED_TWO_ANCHOR"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class AnchorCorrespondence:
    point_cloud_xyz: tuple[float, float, float]
    gis_xyz: tuple[float, float, float]
    gis_asset_id: str | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.point_cloud_xyz, "point_cloud_xyz"), (self.gis_xyz, "gis_xyz")):
            if len(value) != 3 or not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must contain three finite values")
        if self.gis_asset_id is not None and (
            not isinstance(self.gis_asset_id, str) or not self.gis_asset_id.strip()
        ):
            raise ValueError("gis_asset_id must be a non-empty string when provided")


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    state: RegistrationState
    transform_2d: dict[str, Any] | None
    anchor_count: int
    unique_rigid_transform: bool
    diagnostic: str
    fallback_manual_tower_selection: bool = False

    def __post_init__(self) -> None:
        if type(self.state) is not RegistrationState:
            raise ValueError("state must be a RegistrationState")
        if self.anchor_count < 0:
            raise ValueError("anchor_count must be non-negative")
        if type(self.unique_rigid_transform) is not bool:
            raise ValueError("unique_rigid_transform must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "transform_2d": self.transform_2d,
            "anchor_count": self.anchor_count,
            "unique_rigid_transform": self.unique_rigid_transform,
            "diagnostic": self.diagnostic,
            "fallback_manual_tower_selection": self.fallback_manual_tower_selection,
        }


def georeferenced_registration(
    *, point_cloud_crs: str | None, gis_crs: str | None
) -> RegistrationResult:
    if not point_cloud_crs or not gis_crs:
        return RegistrationResult(
            RegistrationState.REQUIRES_CONFIRMATION,
            None,
            0,
            False,
            "both CRS declarations are required",
        )
    if point_cloud_crs != gis_crs:
        return RegistrationResult(
            RegistrationState.REQUIRES_CONFIRMATION,
            None,
            0,
            False,
            "CRS transformation is not silently inferred",
        )
    return RegistrationResult(
        RegistrationState.GEOREFERENCED,
        {"type": "identity", "crs": point_cloud_crs},
        0,
        True,
        "compatible known CRS",
    )


def register_one_anchor(anchor: AnchorCorrespondence) -> RegistrationResult:
    return RegistrationResult(
        RegistrationState.ASSISTED_ONE_ANCHOR,
        {
            "type": "translation_candidate",
            "dx": anchor.gis_xyz[0] - anchor.point_cloud_xyz[0],
            "dy": anchor.gis_xyz[1] - anchor.point_cloud_xyz[1],
        },
        1,
        False,
        "one anchor reduces candidate space but cannot determine unique rotation",
    )


def register_two_anchors(
    first: AnchorCorrespondence,
    second: AnchorCorrespondence,
    *,
    relative_scale_tolerance: float = 0.01,
) -> RegistrationResult:
    if relative_scale_tolerance <= 0.0 or not math.isfinite(relative_scale_tolerance):
        raise ValueError("relative_scale_tolerance must be positive and finite")
    source_dx = second.point_cloud_xyz[0] - first.point_cloud_xyz[0]
    source_dy = second.point_cloud_xyz[1] - first.point_cloud_xyz[1]
    target_dx = second.gis_xyz[0] - first.gis_xyz[0]
    target_dy = second.gis_xyz[1] - first.gis_xyz[1]
    source_distance = math.hypot(source_dx, source_dy)
    target_distance = math.hypot(target_dx, target_dy)
    if source_distance <= 0.0 or target_distance <= 0.0:
        return RegistrationResult(
            RegistrationState.INVALID, None, 2, False, "two anchors must be distinct in 2-D"
        )
    scale_error = abs(target_distance / source_distance - 1.0)
    if scale_error > relative_scale_tolerance:
        return RegistrationResult(
            RegistrationState.REQUIRES_CONFIRMATION,
            None,
            2,
            False,
            "inter-anchor distances imply scale; dataset was not silently scaled",
        )
    theta = math.atan2(target_dy, target_dx) - math.atan2(source_dy, source_dx)
    cosine, sine = math.cos(theta), math.sin(theta)
    tx = first.gis_xyz[0] - (cosine * first.point_cloud_xyz[0] - sine * first.point_cloud_xyz[1])
    ty = first.gis_xyz[1] - (sine * first.point_cloud_xyz[0] + cosine * first.point_cloud_xyz[1])
    return RegistrationResult(
        RegistrationState.REGISTERED_TWO_ANCHOR,
        {"type": "rigid_2d", "rotation_rad": theta, "translation_xy": [tx, ty], "scale": 1.0},
        2,
        True,
        "two-anchor local 2-D rigid transform accepted",
    )


def no_gis_registration() -> RegistrationResult:
    return RegistrationResult(
        RegistrationState.UNREGISTERED,
        None,
        0,
        False,
        "GIS absent; manual Tower selection remains available",
        True,
    )


__all__ = [
    "AnchorCorrespondence",
    "RegistrationResult",
    "RegistrationState",
    "georeferenced_registration",
    "no_gis_registration",
    "register_one_anchor",
    "register_two_anchors",
]
