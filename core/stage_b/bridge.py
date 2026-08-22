"""One explicit Stage-A product -> existing science adapter.

The Stage-A sidecar remains the source of truth for product identity and
topology.  The existing ``SpanAnalysisInput`` remains the source of truth for
the validated science executor.  This module only translates between them and
retains the product identity in an auditable envelope; it does not persist a
parallel science Project or duplicate physical Tower records.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from algorithms.spans.project_adapter import SpanAnalysisInput

from core.config.fast_baseline import FastBaselineConfig, canonical_digest
from core.stage_a.models import Project, Span, Tower
from core.stage_a.risk import RiskPolicy

_BRIDGE_SCHEMA = "treeline-project-science-bridge-v1"


def _xyz(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numeric values")
    values = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite numeric values")
    return values


def _tower_geometry(tower: Tower) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Read optional refined geometry without changing the Stage-A Tower record.

    Stage A permits a manual Tower seed before refinement.  For the science
    compatibility view, a declared ``top_xyz_absolute_m`` or ``height_m`` is
    preferred.  A one-metre compatibility top is used only when no refined
    geometry exists, and the binding records that fallback explicitly.
    """

    base = tower.position
    top_value = tower.attributes.get("top_xyz_absolute_m")
    if top_value is not None:
        return base, _xyz(top_value, f"Tower[{tower.tower_id}].top_xyz_absolute_m")
    height = tower.attributes.get("height_m", 1.0)
    try:
        height = float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Tower[{tower.tower_id}].height_m must be numeric") from exc
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError(f"Tower[{tower.tower_id}].height_m must be positive")
    return base, (base[0], base[1], base[2] + height)


def _bounds(
    start: tuple[float, float, float],
    axis: tuple[float, float],
    length: float,
    half_width: float,
    before: float,
    after: float,
    z_min: float,
    z_max: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    perp = (-axis[1], axis[0])
    corners = []
    for longitudinal in (-before, length + after):
        for lateral in (-half_width, half_width):
            xy = (
                start[0] + axis[0] * longitudinal + perp[0] * lateral,
                start[1] + axis[1] * longitudinal + perp[1] * lateral,
            )
            corners.extend(((xy[0], xy[1], z_min), (xy[0], xy[1], z_max)))
    return (
        tuple(min(point[index] for point in corners) for index in range(3)),
        tuple(max(point[index] for point in corners) for index in range(3)),
    )


def _span_geometry(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_top: tuple[float, float, float],
    end_top: tuple[float, float, float],
    config: FastBaselineConfig,
) -> dict[str, Any]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError("Span endpoints must not be coincident")
    axis = (dx / length, dy / length)
    z_min = min(start[2], end[2]) - config.span_z_ground_clearance_m
    z_max = max(start_top[2], end_top[2]) + config.span_z_top_margin_m
    core_min, core_max = _bounds(
        start, axis, length, config.span_half_width_m, 0.0, 0.0, z_min, z_max
    )
    halo_min, halo_max = _bounds(
        start,
        axis,
        length,
        config.span_half_width_m,
        config.span_halo_m,
        config.span_halo_m,
        z_min,
        z_max,
    )
    return {
        "length_m": length,
        "direction_xy": list(axis),
        "core_half_width_m": config.span_half_width_m,
        "longitudinal_halo_m": config.span_halo_m,
        "core_bounds_min_xyz": list(core_min),
        "core_bounds_max_xyz": list(core_max),
        "halo_bounds_min_xyz": list(halo_min),
        "halo_bounds_max_xyz": list(halo_max),
        "configuration_digest": canonical_digest(
            {
                "schema": "treeline-stage-b-span-geometry-v1",
                "core_half_width_m": config.span_half_width_m,
                "halo_m": config.span_halo_m,
                "z_ground_margin_m": config.span_z_ground_clearance_m,
                "z_top_margin_m": config.span_z_top_margin_m,
            }
        ),
    }


@dataclass(frozen=True, slots=True)
class BridgeIdentity:
    """Product identity carried into a science run without name-based IDs."""

    project_id: str
    scan_id: str
    line_id: str
    tower_id: str
    span_id: str
    analysis_run_id: str | None
    source_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("project_id", "scan_id", "line_id", "tower_id", "span_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a stable non-empty identity")
        if self.analysis_run_id is not None and (
            type(self.analysis_run_id) is not str or not self.analysis_run_id.strip()
        ):
            raise ValueError("analysis_run_id must be a stable identity or None")
        object.__setattr__(self, "source_identity", dict(self.source_identity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "scan_id": self.scan_id,
            "line_id": self.line_id,
            "tower_id": self.tower_id,
            "span_id": self.span_id,
            "analysis_run_id": self.analysis_run_id,
            "source_identity": dict(self.source_identity),
        }


@dataclass(frozen=True, slots=True)
class SpanScienceBinding:
    """Adapter result: product identity plus the existing science input."""

    identity: BridgeIdentity
    science_input: SpanAnalysisInput
    product_span: Span
    product_start_tower: Tower
    product_end_tower: Tower
    internal_line_id: str
    internal_span_id: str

    def with_analysis_run(self, analysis_run_id: str) -> SpanScienceBinding:
        identity = replace(self.identity, analysis_run_id=analysis_run_id)
        science_input = replace(
            self.science_input,
            analysis_run_id=analysis_run_id,
            product_identity=identity.to_dict(),
        )
        return replace(self, identity=identity, science_input=science_input)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _BRIDGE_SCHEMA,
            "identity": self.identity.to_dict(),
            "science_input": self.science_input.to_dict(),
            "product_span": self.product_span.to_dict(),
            "internal_line_id": self.internal_line_id,
            "internal_span_id": self.internal_span_id,
        }


class ProjectScienceBridge:
    """Translate one Stage-A Project aggregate into existing science inputs."""

    def __init__(
        self,
        project: Project,
        scan_id: str,
        *,
        risk_policy: RiskPolicy | None = None,
        config: FastBaselineConfig | None = None,
    ) -> None:
        if type(project) is not Project:
            raise TypeError("project must be the Stage-A Project aggregate")
        self.project = project
        self.scan = next((scan for scan in project.scans if scan.scan_id == scan_id), None)
        if self.scan is None:
            raise ValueError(f"unknown scan_id: {scan_id}")
        self.policy = risk_policy or RiskPolicy()
        self.config = config or FastBaselineConfig()
        self._towers = {tower.tower_id: tower for tower in project.towers}
        self._lines = {line.line_id: line for line in project.lines}
        self._spans = {span.span_id: span for span in project.spans}
        self._source_identity = {
            **self.scan.source_identity.to_dict(),
            "scan_id": self.scan.scan_id,
            "resolved_access_path": self.scan.point_cloud_source,
            "source_point_count": self.scan.capability_report.get("point_count")
            or self.scan.coverage.get("point_count"),
        }
        for key in ("origin_context", "source_bounds_min_xyz"):
            if key in self.scan.coverage:
                self._source_identity[key] = self.scan.coverage[key]
        ordered_line_ids = tuple(sorted(self._lines))
        self._line_order = {
            line_id: index for index, line_id in enumerate(ordered_line_ids, start=1)
        }
        ordered_tower_ids: list[str] = []
        ordered_span_ids: list[str] = []
        for line_id in ordered_line_ids:
            line = self._lines[line_id]
            for tower_id in line.ordered_tower_ids:
                if tower_id not in ordered_tower_ids:
                    ordered_tower_ids.append(tower_id)
            for start_tower_id, end_tower_id in zip(
                line.ordered_tower_ids,
                line.ordered_tower_ids[1:],
                strict=False,
            ):
                span = next(
                    (
                        item
                        for item in self._spans.values()
                        if item.line_id == line_id
                        and item.start_tower_id == start_tower_id
                        and item.end_tower_id == end_tower_id
                    ),
                    None,
                )
                if span is not None and span.span_id not in ordered_span_ids:
                    ordered_span_ids.append(span.span_id)
        ordered_tower_ids.extend(
            tower_id for tower_id in sorted(self._towers) if tower_id not in ordered_tower_ids
        )
        ordered_span_ids.extend(
            span_id for span_id in sorted(self._spans) if span_id not in ordered_span_ids
        )
        self._tower_order = {
            tower_id: index for index, tower_id in enumerate(ordered_tower_ids, start=1)
        }
        self._span_order = {
            span_id: index for index, span_id in enumerate(ordered_span_ids, start=1)
        }

    @property
    def source_identity(self) -> Mapping[str, Any]:
        return dict(self._source_identity)

    @property
    def risk_policy_snapshot(self) -> dict[str, Any]:
        snapshot = self.policy.to_dict()
        snapshot["configuration_digest"] = canonical_digest(snapshot)
        return snapshot

    def bind_span(self, span_id: str, *, analysis_run_id: str | None = None) -> SpanScienceBinding:
        try:
            span = self._spans[span_id]
            self._lines[span.line_id]
            start_tower = self._towers[span.start_tower_id]
            end_tower = self._towers[span.end_tower_id]
        except KeyError as exc:
            raise ValueError(f"unknown product topology identity: {span_id}") from exc
        start, start_top = _tower_geometry(start_tower)
        end, end_top = _tower_geometry(end_tower)
        geometry = _span_geometry(start, end, start_top, end_top, self.config)
        internal_line_id = f"G{self._line_order[span.line_id]:03d}"
        internal_span_id = f"S{self._span_order[span.span_id]:03d}"
        identity = BridgeIdentity(
            project_id=self.project.project_id,
            scan_id=self.scan.scan_id,
            line_id=span.line_id,
            tower_id=f"{span.start_tower_id}->{span.end_tower_id}",
            span_id=span.span_id,
            analysis_run_id=analysis_run_id,
            source_identity=self.source_identity,
        )
        science_input = SpanAnalysisInput(
            project_id=self.project.project_id,
            source_identity=self.source_identity,
            group_id=internal_line_id,
            span_id=internal_span_id,
            from_tower_id=f"T{self._tower_order[span.start_tower_id]:03d}",
            to_tower_id=f"T{self._tower_order[span.end_tower_id]:03d}",
            from_tower_revision=1,
            to_tower_revision=1,
            from_endpoint_geometry={
                "base_xyz_absolute_m": list(start),
                "top_xyz_absolute_m": list(start_top),
            },
            to_endpoint_geometry={
                "base_xyz_absolute_m": list(end),
                "top_xyz_absolute_m": list(end_top),
            },
            span_geometry=geometry,
            science_configuration_digest=geometry["configuration_digest"],
            risk_configuration_digest=self.risk_policy_snapshot["configuration_digest"],
            validated_source_preparation_references=(),
            source_point_count=self._source_identity.get("source_point_count"),
            product_identity=identity.to_dict(),
            risk_policy_snapshot=self.risk_policy_snapshot,
            analysis_run_id=analysis_run_id,
        )
        return SpanScienceBinding(
            identity=identity,
            science_input=science_input,
            product_span=span,
            product_start_tower=start_tower,
            product_end_tower=end_tower,
            internal_line_id=internal_line_id,
            internal_span_id=internal_span_id,
        )

    def bind_spans(
        self,
        span_ids: tuple[str, ...] | list[str] | None = None,
        *,
        analysis_run_id: str | None = None,
    ) -> tuple[SpanScienceBinding, ...]:
        selected = tuple(span_ids) if span_ids is not None else tuple(self._spans)
        if len(selected) != len(set(selected)):
            raise ValueError("span selection must not contain duplicate identities")
        return tuple(
            self.bind_span(span_id, analysis_run_id=analysis_run_id)
            for span_id in selected
        )


def bridge_project_to_science(
    project: Project,
    scan_id: str,
    *,
    span_ids: tuple[str, ...] | list[str] | None = None,
    risk_policy: RiskPolicy | None = None,
    config: FastBaselineConfig | None = None,
) -> tuple[SpanScienceBinding, ...]:
    """Convenience entrypoint used by the Stage-B scheduler and tests."""

    return ProjectScienceBridge(
        project, scan_id, risk_policy=risk_policy, config=config
    ).bind_spans(span_ids)


__all__ = [
    "BridgeIdentity",
    "ProjectScienceBridge",
    "SpanScienceBinding",
    "bridge_project_to_science",
]
