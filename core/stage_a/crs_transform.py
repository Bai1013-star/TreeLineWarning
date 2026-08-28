"""Explicit, offline-only horizontal CRS transformation helpers.

TreeLineWarning never guesses a CRS and never enables PROJ network access at
runtime. This module only transforms coordinates when both source and target
CRS declarations are explicit. Vertical datums are deliberately out of scope
for V0.1 corridor registration; Z values are preserved unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

# Enforce the product runtime contract before pyproj is imported anywhere in
# this process. Individual transformer construction also disables the pyproj
# network switch so an ambient user setting cannot silently re-enable it.
os.environ["PROJ_NETWORK"] = "OFF"


class CRSTransformUnavailable(ValueError):
    """Raised when an explicit offline CRS transform cannot be constructed."""


@dataclass(frozen=True, slots=True)
class CRSTransformPlan:
    source_crs: str
    target_crs: str
    identity: bool
    description: str

    def __post_init__(self) -> None:
        pairs = (
            (self.source_crs, "source_crs"),
            (self.target_crs, "target_crs"),
        )
        for value, name in pairs:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty explicit CRS")
        if type(self.identity) is not bool:
            raise ValueError("identity must be a boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "identity" if self.identity else "crs_transform",
            "source_crs": self.source_crs,
            "target_crs": self.target_crs,
            "always_xy": True,
            "allow_ballpark": False,
            "only_best": True,
            "network_required": False,
            "vertical_transform": False,
            "description": self.description,
        }


def _explicit_crs(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CRSTransformUnavailable(f"{name} must be explicitly declared")
    return value.strip()


def _offline_pyproj() -> Any:
    try:
        import pyproj
    except ImportError as exc:  # pragma: no cover - required product dependency
        raise CRSTransformUnavailable(
            "pyproj is required for explicit CRS transformation"
        ) from exc
    os.environ["PROJ_NETWORK"] = "OFF"
    pyproj.network.set_network_enabled(False)
    return pyproj


def offline_proj_status() -> dict[str, Any]:
    """Return the packaged PROJ state used by offline acceptance tests."""

    pyproj = _offline_pyproj()
    return {
        "network_enabled": bool(pyproj.network.is_network_enabled()),
        "data_dir": str(pyproj.datadir.get_data_dir()),
        "proj_network_env": os.environ.get("PROJ_NETWORK"),
    }


def _transformer(source_crs: str, target_crs: str) -> Any:
    pyproj = _offline_pyproj()
    try:
        return pyproj.Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True,
            allow_ballpark=False,
            only_best=True,
        )
    except (pyproj.exceptions.CRSError, pyproj.exceptions.ProjError) as exc:
        raise CRSTransformUnavailable(
            f"offline CRS transform unavailable: {source_crs} -> {target_crs}"
        ) from exc


def build_crs_transform(
    source_crs: str | None,
    target_crs: str | None,
) -> CRSTransformPlan:
    """Validate an explicit horizontal transform without guessing either CRS."""

    source = _explicit_crs(source_crs, "source_crs")
    target = _explicit_crs(target_crs, "target_crs")
    if source == target:
        return CRSTransformPlan(source, target, True, "explicit identical CRS")
    transformer = _transformer(source, target)
    description = str(getattr(transformer, "description", "offline PROJ transform"))
    unavailable = "unavailable until proj_trans is called"
    if not description.strip() or description.strip().lower() == unavailable:
        description = "explicit offline PROJ transform"
    return CRSTransformPlan(source, target, False, description)


def _xy(value: Sequence[float]) -> tuple[float, float]:
    if len(value) < 2:
        raise ValueError("coordinate must contain x and y")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("coordinate x/y must be finite")
    return x, y


def transform_xy(
    plan: CRSTransformPlan,
    coordinates: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Transform XY coordinates with strict offline error checking."""

    if type(plan) is not CRSTransformPlan:
        raise ValueError("plan must be a CRSTransformPlan")
    values = tuple(_xy(item) for item in coordinates)
    if plan.identity:
        return values
    if not values:
        return ()
    transformer = _transformer(plan.source_crs, plan.target_crs)
    xs, ys = zip(*values, strict=True)
    try:
        tx, ty = transformer.transform(xs, ys, errcheck=True)
    except Exception as exc:
        raise CRSTransformUnavailable(
            f"offline CRS transform failed: {plan.source_crs} -> {plan.target_crs}"
        ) from exc
    result = tuple((float(x), float(y)) for x, y in zip(tx, ty, strict=True))
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in result):
        raise CRSTransformUnavailable(
            "offline CRS transform returned non-finite coordinates"
        )
    return result


def transform_xyz(
    plan: CRSTransformPlan,
    coordinates: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """Transform horizontal coordinates while preserving Z exactly."""

    values: list[tuple[float, float, float]] = []
    for item in coordinates:
        if len(item) != 3:
            raise ValueError("XYZ coordinate must contain exactly three values")
        x, y = _xy(item)
        z = float(item[2])
        if not math.isfinite(z):
            raise ValueError("coordinate z must be finite")
        values.append((x, y, z))
    transformed = transform_xy(plan, values)
    return tuple(
        (xy[0], xy[1], values[index][2])
        for index, xy in enumerate(transformed)
    )


__all__ = [
    "CRSTransformPlan",
    "CRSTransformUnavailable",
    "build_crs_transform",
    "offline_proj_status",
    "transform_xy",
    "transform_xyz",
]
