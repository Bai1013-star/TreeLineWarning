"""Identity-preserving confirmation helpers for GIS-seeded Towers.

A local candidate is presentation assistance only.  This module records the
operator-confirmed RAW refinement on the existing Tower object so persistent
``tower_id`` and ``external_asset_id`` never change during corridor alignment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from core.stage_a.models import CoordinateDeclaration, Project, Tower

GIS_CONFIRMED_POSITION_SOURCE = "GIS_SEED_LOCAL_CANDIDATE_CONFIRMED"
MANUAL_CONFIRMED_POSITION_SOURCES = frozenset({"MANUAL", "MANUAL_PICK"})


def _xyz(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("confirmed_position must contain exactly x, y, z")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("confirmed_position must contain finite values")
    return result  # type: ignore[return-value]


def is_tower_confirmed(tower: Tower) -> bool:
    """Return whether a Tower has an operator-confirmed LiDAR-frame position."""

    if type(tower) is not Tower:
        raise ValueError("tower must be a Tower")
    return tower.position_source == GIS_CONFIRMED_POSITION_SOURCE or (
        tower.position_source in MANUAL_CONFIRMED_POSITION_SOURCES
    )


def confirm_gis_tower(
    project: Project,
    tower_id: str,
    confirmed_position: Sequence[float],
    point_cloud_crs: CoordinateDeclaration,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> Project:
    """Confirm one GIS seed while preserving its persistent identity."""

    if type(project) is not Project:
        raise ValueError("project must be a Project")
    if type(point_cloud_crs) is not CoordinateDeclaration:
        raise ValueError("point_cloud_crs must be a CoordinateDeclaration")
    position = _xyz(confirmed_position)
    target = next((tower for tower in project.towers if tower.tower_id == tower_id), None)
    if target is None:
        raise ValueError("TOWER_NOT_FOUND")
    if target.external_asset_id is None:
        raise ValueError("GIS_TOWER_EXTERNAL_ID_REQUIRED")
    if not (
        target.position_source.startswith("GIS:")
        or target.position_source == GIS_CONFIRMED_POSITION_SOURCE
    ):
        raise ValueError("ONLY_GIS_SEEDED_TOWERS_CAN_USE_GIS_CONFIRMATION")

    attributes = dict(target.attributes)
    attributes.setdefault("gis_original_position", list(target.position))
    attributes.setdefault("gis_original_position_crs", target.position_crs.to_dict())
    confirmation = {
        "authority": "OPERATOR_CONFIRMED_RAW_LOCAL_REFINEMENT",
        "point_cloud_position": list(position),
        "point_cloud_crs": point_cloud_crs.to_dict(),
        "evidence": dict(evidence or {}),
    }
    attributes["tower_confirmation_v1"] = confirmation
    updated_tower = replace(
        target,
        position=position,
        position_crs=point_cloud_crs,
        position_source=GIS_CONFIRMED_POSITION_SOURCE,
        attributes=attributes,
    )
    towers = tuple(
        updated_tower if tower.tower_id == tower_id else tower for tower in project.towers
    )
    return replace(project, towers=towers)


def confirmed_tower_ids(project: Project) -> frozenset[str]:
    """Return persistent Tower IDs that are safe as Span endpoints."""

    if type(project) is not Project:
        raise ValueError("project must be a Project")
    return frozenset(tower.tower_id for tower in project.towers if is_tower_confirmed(tower))


__all__ = [
    "GIS_CONFIRMED_POSITION_SOURCE",
    "MANUAL_CONFIRMED_POSITION_SOURCES",
    "confirm_gis_tower",
    "confirmed_tower_ids",
    "is_tower_confirmed",
]
