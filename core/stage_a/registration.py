"""GIS ↔ point-cloud registration state machine.

Known CRS pairs use an explicit offline transform into the point-cloud frame.
Unknown/local frames use operator anchors. Two anchors establish a unique 2-D
rigid transform without scale; a later checkpoint is required to promote that
transform to VERIFIED for the assisted workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Sequence

from core.stage_a.crs_transform import (
    CRSTransformPlan,
    CRSTransformUnavailable,
    build_anchor_metric_transform,
    build_crs_transform,
    transform_xy,
)


class RegistrationState(str, Enum):
    UNREGISTERED = "UNREGISTERED"
    GEOREFERENCED = "GEOREFERENCED"
    ASSISTED_ONE_ANCHOR = "ASSISTED_ONE_ANCHOR"
    REGISTERED_TWO_ANCHOR = "REGISTERED_TWO_ANCHOR"
    VERIFIED = "VERIFIED"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class AnchorCorrespondence:
    point_cloud_xyz: tuple[float, float, float]
    gis_xyz: tuple[float, float, float]
    gis_asset_id: str | None = None

    def __post_init__(self) -> None:
        pairs = (
            (self.point_cloud_xyz, "point_cloud_xyz"),
            (self.gis_xyz, "gis_xyz"),
        )
        for value, name in pairs:
            if len(value) != 3 or not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must contain three finite values")
        if self.gis_asset_id is not None and (
            not isinstance(self.gis_asset_id, str) or not self.gis_asset_id.strip()
        ):
            raise ValueError("gis_asset_id must be a non-empty string when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_cloud_xyz": [float(item) for item in self.point_cloud_xyz],
            "gis_xyz": [float(item) for item in self.gis_xyz],
            "gis_asset_id": self.gis_asset_id,
        }


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    state: RegistrationState
    transform_2d: dict[str, Any] | None
    anchor_count: int
    unique_rigid_transform: bool
    diagnostic: str
    fallback_manual_tower_selection: bool = False
    checkpoint_residual_m: float | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if type(self.state) is not RegistrationState:
            raise ValueError("state must be a RegistrationState")
        if self.anchor_count < 0:
            raise ValueError("anchor_count must be non-negative")
        if type(self.unique_rigid_transform) is not bool:
            raise ValueError("unique_rigid_transform must be a boolean")
        if type(self.verified) is not bool:
            raise ValueError("verified must be a boolean")
        residual = self.checkpoint_residual_m
        if residual is not None and (
            not math.isfinite(float(residual)) or float(residual) < 0.0
        ):
            raise ValueError("checkpoint_residual_m must be finite and non-negative")

    @property
    def ready_for_science(self) -> bool:
        return self.verified and self.state in {
            RegistrationState.GEOREFERENCED,
            RegistrationState.VERIFIED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "transform_2d": self.transform_2d,
            "anchor_count": self.anchor_count,
            "unique_rigid_transform": self.unique_rigid_transform,
            "diagnostic": self.diagnostic,
            "fallback_manual_tower_selection": self.fallback_manual_tower_selection,
            "checkpoint_residual_m": self.checkpoint_residual_m,
            "verified": self.verified,
            "ready_for_science": self.ready_for_science,
        }


def registration_result_from_dict(value: Any) -> RegistrationResult:
    """Hydrate the persisted JSON-safe registration result."""

    if not isinstance(value, dict):
        raise ValueError("registration result must be a mapping")
    return RegistrationResult(
        RegistrationState(str(value["state"])),
        dict(value["transform_2d"]) if isinstance(value.get("transform_2d"), dict) else None,
        int(value.get("anchor_count", 0)),
        bool(value.get("unique_rigid_transform", False)),
        str(value.get("diagnostic") or "persisted registration"),
        bool(value.get("fallback_manual_tower_selection", False)),
        (
            None
            if value.get("checkpoint_residual_m") is None
            else float(value["checkpoint_residual_m"])
        ),
        bool(value.get("verified", False)),
    )


def georeferenced_registration(
    *, point_cloud_crs: str | None, gis_crs: str | None
) -> RegistrationResult:
    """Register GIS coordinates into the point-cloud horizontal CRS."""

    if not point_cloud_crs or not gis_crs:
        return RegistrationResult(
            RegistrationState.REQUIRES_CONFIRMATION,
            None,
            0,
            False,
            "both CRS declarations are required",
        )
    try:
        plan = build_crs_transform(gis_crs, point_cloud_crs)
    except CRSTransformUnavailable as exc:
        return RegistrationResult(
            RegistrationState.REQUIRES_CONFIRMATION,
            None,
            0,
            False,
            str(exc),
        )
    diagnostic = (
        "compatible explicit CRS"
        if plan.identity
        else "explicit offline CRS transform accepted"
    )
    return RegistrationResult(
        RegistrationState.GEOREFERENCED,
        {**plan.to_dict(), "direction": "GIS_TO_POINT_CLOUD"},
        0,
        True,
        diagnostic,
        checkpoint_residual_m=0.0 if plan.identity else None,
        verified=True,
    )


def register_one_anchor(anchor: AnchorCorrespondence) -> RegistrationResult:
    """Return a translation hint only; one anchor never establishes rotation."""

    return RegistrationResult(
        RegistrationState.ASSISTED_ONE_ANCHOR,
        {
            "type": "translation_candidate",
            "direction": "POINT_CLOUD_TO_GIS",
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
    """Establish a no-scale point-cloud→GIS-metric 2-D rigid transform."""

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
            RegistrationState.INVALID,
            None,
            2,
            False,
            "two anchors must be distinct in 2-D",
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
    source_x, source_y = first.point_cloud_xyz[:2]
    tx = first.gis_xyz[0] - (cosine * source_x - sine * source_y)
    ty = first.gis_xyz[1] - (sine * source_x + cosine * source_y)
    return RegistrationResult(
        RegistrationState.REGISTERED_TWO_ANCHOR,
        {
            "type": "rigid_2d",
            "direction": "POINT_CLOUD_TO_GIS_METRIC",
            "rotation_rad": theta,
            "translation_xy": [tx, ty],
            "scale": 1.0,
        },
        2,
        True,
        "two-anchor rigid transform established; checkpoint validation required",
    )


def _metric_anchor(
    anchor: AnchorCorrespondence,
    plan: CRSTransformPlan,
) -> AnchorCorrespondence:
    x, y = transform_xy(plan, (anchor.gis_xyz,))[0]
    return AnchorCorrespondence(
        anchor.point_cloud_xyz,
        (x, y, float(anchor.gis_xyz[2])),
        anchor.gis_asset_id,
    )


def register_one_anchor_geospatial(
    anchor: AnchorCorrespondence,
    *,
    gis_crs: str,
) -> RegistrationResult:
    """Create a one-anchor hint in a local metre GIS frame."""

    plan = build_anchor_metric_transform(gis_crs, anchor.gis_xyz)
    metric = _metric_anchor(anchor, plan)
    result = register_one_anchor(metric)
    transform = dict(result.transform_2d or {})
    transform.update(
        {
            "direction": "POINT_CLOUD_TO_GIS_METRIC",
            "gis_source_crs": plan.source_crs,
            "gis_metric_crs": plan.target_crs,
        }
    )
    return replace(result, transform_2d=transform)


def register_two_anchors_geospatial(
    first: AnchorCorrespondence,
    second: AnchorCorrespondence,
    *,
    gis_crs: str,
    relative_scale_tolerance: float = 0.01,
) -> RegistrationResult:
    """Fit anchors after converting GIS coordinates into a local metre frame."""

    plan = build_anchor_metric_transform(gis_crs, first.gis_xyz)
    result = register_two_anchors(
        _metric_anchor(first, plan),
        _metric_anchor(second, plan),
        relative_scale_tolerance=relative_scale_tolerance,
    )
    if result.transform_2d is None:
        return result
    transform = dict(result.transform_2d)
    transform.update(
        {
            "gis_source_crs": plan.source_crs,
            "gis_metric_crs": plan.target_crs,
            "gis_metric_origin_source_xy": [
                float(first.gis_xyz[0]),
                float(first.gis_xyz[1]),
            ],
        }
    )
    return replace(result, transform_2d=transform)


def _finite_xy(value: Sequence[float]) -> tuple[float, float]:
    if len(value) < 2:
        raise ValueError("coordinate must contain x and y")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("coordinate x/y must be finite")
    return x, y


def _rigid_forward(
    transform: dict[str, Any], xy: tuple[float, float]
) -> tuple[float, float]:
    theta = float(transform["rotation_rad"])
    tx, ty = (float(item) for item in transform["translation_xy"])
    cosine, sine = math.cos(theta), math.sin(theta)
    x, y = xy
    return cosine * x - sine * y + tx, sine * x + cosine * y + ty


def _rigid_inverse(
    transform: dict[str, Any], xy: tuple[float, float]
) -> tuple[float, float]:
    theta = float(transform["rotation_rad"])
    tx, ty = (float(item) for item in transform["translation_xy"])
    cosine, sine = math.cos(theta), math.sin(theta)
    shifted_x, shifted_y = xy[0] - tx, xy[1] - ty
    return (
        cosine * shifted_x + sine * shifted_y,
        -sine * shifted_x + cosine * shifted_y,
    )


def _gis_metric_plan(transform: dict[str, Any]) -> CRSTransformPlan | None:
    source = transform.get("gis_source_crs")
    target = transform.get("gis_metric_crs")
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    return build_crs_transform(source, target)


def _gis_to_metric(
    transform: dict[str, Any],
    xy: tuple[float, float],
) -> tuple[float, float]:
    plan = _gis_metric_plan(transform)
    return xy if plan is None else transform_xy(plan, (xy,))[0]


def _metric_to_gis(
    transform: dict[str, Any],
    xy: tuple[float, float],
) -> tuple[float, float]:
    plan = _gis_metric_plan(transform)
    if plan is None:
        return xy
    inverse = build_crs_transform(plan.target_crs, plan.source_crs)
    return transform_xy(inverse, (xy,))[0]


def apply_registration_xy(
    registration: RegistrationResult,
    xy: Sequence[float],
    *,
    direction: str,
) -> tuple[float, float]:
    """Apply a resolved registration in an explicit direction."""

    if type(registration) is not RegistrationResult:
        raise ValueError("registration must be a RegistrationResult")
    if registration.transform_2d is None:
        raise ValueError("registration has no transform")
    value = _finite_xy(xy)
    requested = str(direction).upper()
    if requested not in {"GIS_TO_POINT_CLOUD", "POINT_CLOUD_TO_GIS"}:
        raise ValueError("direction must be GIS_TO_POINT_CLOUD or POINT_CLOUD_TO_GIS")
    transform = registration.transform_2d
    transform_type = transform.get("type")
    if transform_type == "identity":
        return value
    if transform_type == "crs_transform":
        plan = CRSTransformPlan(
            str(transform["source_crs"]),
            str(transform["target_crs"]),
            False,
            str(transform.get("description") or "explicit offline PROJ transform"),
        )
        if requested == "POINT_CLOUD_TO_GIS":
            plan = build_crs_transform(plan.target_crs, plan.source_crs)
        return transform_xy(plan, (value,))[0]
    if transform_type == "rigid_2d":
        if requested == "GIS_TO_POINT_CLOUD":
            return _rigid_inverse(transform, _gis_to_metric(transform, value))
        return _metric_to_gis(transform, _rigid_forward(transform, value))
    raise ValueError("registration transform is not uniquely applicable")


def validate_registration_checkpoint(
    registration: RegistrationResult,
    checkpoint: AnchorCorrespondence,
    *,
    maximum_residual_m: float = 2.0,
) -> RegistrationResult:
    """Validate an established two-anchor rigid transform with a third tower."""

    if maximum_residual_m <= 0.0 or not math.isfinite(maximum_residual_m):
        raise ValueError("maximum_residual_m must be positive and finite")
    if (
        registration.state is not RegistrationState.REGISTERED_TWO_ANCHOR
        or registration.transform_2d is None
        or registration.transform_2d.get("type") != "rigid_2d"
    ):
        raise ValueError("checkpoint validation requires a two-anchor rigid registration")
    transform = registration.transform_2d
    predicted_metric = _rigid_forward(transform, _finite_xy(checkpoint.point_cloud_xyz))
    expected_metric = _gis_to_metric(transform, _finite_xy(checkpoint.gis_xyz))
    residual = math.hypot(
        predicted_metric[0] - expected_metric[0],
        predicted_metric[1] - expected_metric[1],
    )
    if residual <= maximum_residual_m:
        return replace(
            registration,
            state=RegistrationState.VERIFIED,
            diagnostic=(
                f"checkpoint residual {residual:.3f} m accepted within "
                f"{maximum_residual_m:.3f} m"
            ),
            checkpoint_residual_m=residual,
            verified=True,
        )
    return replace(
        registration,
        state=RegistrationState.REQUIRES_CONFIRMATION,
        diagnostic=(
            f"checkpoint residual {residual:.3f} m exceeds "
            f"{maximum_residual_m:.3f} m; operator confirmation required"
        ),
        checkpoint_residual_m=residual,
        verified=False,
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
    "apply_registration_xy",
    "georeferenced_registration",
    "no_gis_registration",
    "register_one_anchor",
    "register_one_anchor_geospatial",
    "register_two_anchors",
    "register_two_anchors_geospatial",
    "registration_result_from_dict",
    "validate_registration_checkpoint",
]
