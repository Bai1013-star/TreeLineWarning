"""Persistent Stage-A product records.

These records are intentionally small and JSON-friendly.  Stable opaque IDs
are used for persistence; display names remain presentation-only.  Existing
``core.contracts.project`` records remain the authoritative compatibility
boundary for old single-span workspaces.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import uuid4

STAGE_A_SCHEMA_VERSION = "treeline-survey-v01-stage-a-v1"
_ID_RE = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")


class UnitStatus(str, Enum):
    METRE = "METRE"
    FOOT = "FOOT"
    UNKNOWN = "UNKNOWN"
    LOCAL_METRIC = "LOCAL_METRIC"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"


class AnalysisRunStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    STALE = "STALE"


class LineRole(str, Enum):
    MAIN = "MAIN"
    BRANCH = "BRANCH"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str) -> str:
    """Return a stable opaque identifier, never derived from a display label."""

    return f"{prefix}-{uuid4().hex}"


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _id(value: Any, name: str) -> str:
    value = _text(value, name)
    if _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lower-case opaque identifier")
    return value


def _timestamp(value: Any, name: str) -> str:
    value = _text(value, name)
    if not value.endswith("Z") or "T" not in value:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 UTC timestamp") from exc
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _xyz(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numeric values")
    return tuple(_finite(item, f"{name}[]") for item in value)  # type: ignore[return-value]


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(_text(item, f"{name}[]") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _metadata(value: Mapping[str, Any] | None, name: str = "metadata") -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON mapping with string keys")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-safe values") from exc
    return dict(value)


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


@dataclass(frozen=True, slots=True)
class CoordinateDeclaration:
    """Explicit CRS/unit declaration used before metric risk calculations."""

    horizontal_crs: str | None
    vertical_reference: str | None
    linear_unit: str | None
    unit_status: UnitStatus
    coordinate_kind: str = "LOCAL"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "horizontal_crs", _optional_text(self.horizontal_crs, "horizontal_crs")
        )
        object.__setattr__(
            self,
            "vertical_reference",
            _optional_text(self.vertical_reference, "vertical_reference"),
        )
        object.__setattr__(self, "linear_unit", _optional_text(self.linear_unit, "linear_unit"))
        if type(self.unit_status) is not UnitStatus:
            raise ValueError("unit_status must be a UnitStatus")
        kind = _text(self.coordinate_kind, "coordinate_kind").upper()
        if kind not in {"LOCAL", "GEOREFERENCED"}:
            raise ValueError("coordinate_kind must be LOCAL or GEOREFERENCED")
        object.__setattr__(self, "coordinate_kind", kind)
        if self.unit_status is UnitStatus.METRE and self.linear_unit not in {
            None,
            "m",
            "metre",
            "meter",
        }:
            raise ValueError("METRE declarations must identify metre units")
        if self.unit_status is UnitStatus.LOCAL_METRIC and (
            self.horizontal_crs or self.linear_unit not in {None, "m", "metre", "meter"}
        ):
            raise ValueError("LOCAL_METRIC is only valid for an explicitly local metre system")

    @property
    def metric_safe(self) -> bool:
        return self.unit_status in {UnitStatus.METRE, UnitStatus.LOCAL_METRIC}

    @classmethod
    def local_metric(cls) -> CoordinateDeclaration:
        return cls(None, None, "m", UnitStatus.LOCAL_METRIC, "LOCAL")

    @classmethod
    def unresolved(cls, *, horizontal_crs: str | None = None) -> CoordinateDeclaration:
        return cls(
            horizontal_crs,
            None,
            None,
            UnitStatus.REQUIRES_CONFIRMATION,
            "GEOREFERENCED" if horizontal_crs else "LOCAL",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizontal_crs": self.horizontal_crs,
            "vertical_reference": self.vertical_reference,
            "linear_unit": self.linear_unit,
            "unit_status": self.unit_status.value,
            "coordinate_kind": self.coordinate_kind,
        }


@dataclass(frozen=True, slots=True)
class MetricDistance:
    """A distance converted to metres with its unit evidence retained."""

    value_m: float
    source_unit: UnitStatus
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_m", _finite(self.value_m, "value_m"))
        if type(self.source_unit) is not UnitStatus:
            raise ValueError("source_unit must be a UnitStatus")
        _text(self.evidence, "evidence")


def to_metric_distance(value: float, declaration: CoordinateDeclaration) -> MetricDistance:
    """Convert a declared distance to metres; refuse unknown units."""

    if type(declaration) is not CoordinateDeclaration:
        raise ValueError("declaration must be a CoordinateDeclaration")
    value = _finite(value, "value")
    evidence = (
        f"unit_status={declaration.unit_status.value};"
        f"linear_unit={declaration.linear_unit or 'UNSPECIFIED'};"
        f"horizontal_crs={declaration.horizontal_crs or 'LOCAL'}"
    )
    if declaration.unit_status in {UnitStatus.METRE, UnitStatus.LOCAL_METRIC}:
        return MetricDistance(value, declaration.unit_status, evidence)
    if declaration.unit_status is UnitStatus.FOOT:
        return MetricDistance(value * 0.3048, UnitStatus.FOOT, evidence)
    raise ValueError("distance unit requires explicit metre/foot confirmation")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    source_id: str
    path: str
    format: str
    file_size_bytes: int | None = None
    sha256: str | None = None
    read_only: bool = True

    def __post_init__(self) -> None:
        _id(self.source_id, "source_id")
        _text(self.path, "path")
        if PurePath(self.path).is_absolute() is False:
            raise ValueError("source path must be absolute to prevent ambiguous source identity")
        _text(self.format, "format")
        if self.file_size_bytes is not None and (
            type(self.file_size_bytes) is not int or self.file_size_bytes < 0
        ):
            raise ValueError("file_size_bytes must be a non-negative integer")
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if type(self.read_only) is not bool:
            raise ValueError("read_only must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "format": self.format,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "read_only": self.read_only,
        }


@dataclass(frozen=True, slots=True)
class Line:
    line_id: str
    display_name: str
    ordered_tower_ids: tuple[str, ...] = ()
    route_geometry: tuple[tuple[float, float, float], ...] = ()
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    line_role: LineRole = LineRole.MAIN
    parent_line_id: str | None = None

    def __post_init__(self) -> None:
        _id(self.line_id, "line_id")
        _text(self.display_name, "display_name")
        object.__setattr__(
            self, "ordered_tower_ids", _strings(self.ordered_tower_ids, "ordered_tower_ids")
        )
        object.__setattr__(
            self,
            "route_geometry",
            tuple(_xyz(item, "route_geometry[]") for item in self.route_geometry),
        )
        object.__setattr__(
            self, "source_provenance", _metadata(self.source_provenance, "source_provenance")
        )
        if type(self.line_role) is not LineRole:
            raise ValueError("line_role must be a LineRole")
        object.__setattr__(
            self, "parent_line_id", _optional_text(self.parent_line_id, "parent_line_id")
        )
        if self.line_role is LineRole.MAIN and self.parent_line_id is not None:
            raise ValueError("MAIN lines cannot have a parent_line_id")
        if self.line_role is LineRole.BRANCH and self.parent_line_id is None:
            raise ValueError("BRANCH lines require a parent_line_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "display_name": self.display_name,
            "ordered_tower_ids": list(self.ordered_tower_ids),
            "route_geometry": [list(item) for item in self.route_geometry],
            "source_provenance": dict(self.source_provenance),
            "line_role": self.line_role.value,
            "parent_line_id": self.parent_line_id,
        }


@dataclass(frozen=True, slots=True)
class Route:
    route_id: str
    line_id: str
    display_name: str
    ordered_tower_ids: tuple[str, ...] = ()
    route_geometry: tuple[tuple[float, float, float], ...] = ()
    route_role: LineRole = LineRole.MAIN
    parent_route_id: str | None = None
    source_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id(self.route_id, "route_id")
        _id(self.line_id, "line_id")
        _text(self.display_name, "display_name")
        object.__setattr__(
            self, "ordered_tower_ids", _strings(self.ordered_tower_ids, "ordered_tower_ids")
        )
        object.__setattr__(
            self,
            "route_geometry",
            tuple(_xyz(item, "route_geometry[]") for item in self.route_geometry),
        )
        if type(self.route_role) is not LineRole:
            raise ValueError("route_role must be a LineRole")
        object.__setattr__(
            self, "parent_route_id", _optional_text(self.parent_route_id, "parent_route_id")
        )
        if self.route_role is LineRole.MAIN and self.parent_route_id is not None:
            raise ValueError("MAIN routes cannot have a parent_route_id")
        if self.route_role is LineRole.BRANCH and self.parent_route_id is None:
            raise ValueError("BRANCH routes require a parent_route_id")
        object.__setattr__(
            self, "source_provenance", _metadata(self.source_provenance, "source_provenance")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "line_id": self.line_id,
            "display_name": self.display_name,
            "ordered_tower_ids": list(self.ordered_tower_ids),
            "route_geometry": [list(item) for item in self.route_geometry],
            "route_role": self.route_role.value,
            "parent_route_id": self.parent_route_id,
            "source_provenance": dict(self.source_provenance),
        }


@dataclass(frozen=True, slots=True)
class Tower:
    tower_id: str
    external_asset_id: str | None
    display_name: str
    position: tuple[float, float, float]
    position_crs: CoordinateDeclaration
    position_source: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id(self.tower_id, "tower_id")
        object.__setattr__(
            self, "external_asset_id", _optional_text(self.external_asset_id, "external_asset_id")
        )
        _text(self.display_name, "display_name")
        object.__setattr__(self, "position", _xyz(self.position, "position"))
        if type(self.position_crs) is not CoordinateDeclaration:
            raise ValueError("position_crs must be a CoordinateDeclaration")
        _text(self.position_source, "position_source")
        object.__setattr__(self, "attributes", _metadata(self.attributes, "attributes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tower_id": self.tower_id,
            "external_asset_id": self.external_asset_id,
            "display_name": self.display_name,
            "position": list(self.position),
            "position_crs": self.position_crs.to_dict(),
            "position_source": self.position_source,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class Span:
    span_id: str
    line_id: str
    start_tower_id: str
    end_tower_id: str
    core_tower_ids: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source_partition: Mapping[str, Any] = field(
        default_factory=lambda: {
            "kind": "LOGICAL",
            "access_mode": "SHARED_SOURCE_LOCAL",
            "full_source_copy": False,
        }
    )

    def __post_init__(self) -> None:
        for value, name in (
            (self.span_id, "span_id"),
            (self.line_id, "line_id"),
            (self.start_tower_id, "start_tower_id"),
            (self.end_tower_id, "end_tower_id"),
        ):
            _id(value, name)
        if self.start_tower_id == self.end_tower_id:
            raise ValueError("a Span must connect two distinct towers")
        object.__setattr__(self, "core_tower_ids", _strings(self.core_tower_ids, "core_tower_ids"))
        object.__setattr__(self, "attributes", _metadata(self.attributes, "attributes"))
        partition = _metadata(self.source_partition, "source_partition")
        if partition.get("kind", "LOGICAL") != "LOGICAL":
            raise ValueError("Stage-A Span source_partition kind must be LOGICAL")
        if partition.get("access_mode", "SHARED_SOURCE_LOCAL") != "SHARED_SOURCE_LOCAL":
            raise ValueError("Stage-A Span source access must use SHARED_SOURCE_LOCAL")
        if partition.get("full_source_copy", False) is not False:
            raise ValueError("Stage-A Span source partition cannot copy the full source")
        partition.setdefault("kind", "LOGICAL")
        partition.setdefault("access_mode", "SHARED_SOURCE_LOCAL")
        partition.setdefault("full_source_copy", False)
        object.__setattr__(self, "source_partition", partition)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "line_id": self.line_id,
            "start_tower_id": self.start_tower_id,
            "end_tower_id": self.end_tower_id,
            "core_tower_ids": list(self.core_tower_ids),
            "attributes": dict(self.attributes),
            "source_partition": dict(self.source_partition),
        }


@dataclass(frozen=True, slots=True)
class Scan:
    scan_id: str
    source_identity: SourceDescriptor
    point_cloud_source: str
    capability_report: Mapping[str, Any]
    registration_status: str = "UNREGISTERED"
    coverage: Mapping[str, Any] = field(default_factory=dict)
    imported_at: str = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id(self.scan_id, "scan_id")
        if type(self.source_identity) is not SourceDescriptor:
            raise ValueError("source_identity must be a SourceDescriptor")
        _text(self.point_cloud_source, "point_cloud_source")
        if not isinstance(self.capability_report, Mapping):
            raise ValueError("capability_report must be a mapping")
        _text(self.registration_status, "registration_status")
        object.__setattr__(self, "coverage", _metadata(self.coverage, "coverage"))
        object.__setattr__(self, "metadata", _metadata(self.metadata, "metadata"))
        object.__setattr__(self, "imported_at", _timestamp(self.imported_at, "imported_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "source_identity": self.source_identity.to_dict(),
            "point_cloud_source": self.point_cloud_source,
            "capability_report": dict(self.capability_report),
            "registration_status": self.registration_status,
            "coverage": dict(self.coverage),
            "imported_at": self.imported_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    analysis_run_id: str
    scan_id: str
    span_ids: tuple[str, ...]
    algorithm_versions: Mapping[str, str]
    configuration_snapshot: Mapping[str, Any]
    risk_policy_snapshot: Mapping[str, Any]
    status: AnalysisRunStatus = AnalysisRunStatus.NOT_STARTED
    artifact_references: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _id(self.analysis_run_id, "analysis_run_id")
        _id(self.scan_id, "scan_id")
        object.__setattr__(self, "span_ids", _strings(self.span_ids, "span_ids"))
        object.__setattr__(
            self,
            "algorithm_versions",
            {
                str(k): _text(v, f"algorithm_versions[{k}]")
                for k, v in self.algorithm_versions.items()
            },
        )
        object.__setattr__(
            self,
            "configuration_snapshot",
            _metadata(self.configuration_snapshot, "configuration_snapshot"),
        )
        object.__setattr__(
            self,
            "risk_policy_snapshot",
            _metadata(self.risk_policy_snapshot, "risk_policy_snapshot"),
        )
        if type(self.status) is not AnalysisRunStatus:
            raise ValueError("status must be an AnalysisRunStatus")
        object.__setattr__(
            self, "artifact_references", _metadata(self.artifact_references, "artifact_references")
        )
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_run_id": self.analysis_run_id,
            "scan_id": self.scan_id,
            "span_ids": list(self.span_ids),
            "algorithm_versions": dict(self.algorithm_versions),
            "configuration_snapshot": dict(self.configuration_snapshot),
            "risk_policy_snapshot": dict(self.risk_policy_snapshot),
            "status": self.status.value,
            "artifact_references": dict(self.artifact_references),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class Project:
    project_id: str
    schema_version: str
    name: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    lines: tuple[Line, ...] = ()
    towers: tuple[Tower, ...] = ()
    spans: tuple[Span, ...] = ()
    scans: tuple[Scan, ...] = ()
    analysis_runs: tuple[AnalysisRun, ...] = ()
    basemap_policy: str = "NO_BASEMAP"
    routes: tuple[Route, ...] = ()

    def __post_init__(self) -> None:
        _id(self.project_id, "project_id")
        if self.schema_version != STAGE_A_SCHEMA_VERSION:
            raise ValueError("unsupported Stage-A project schema version")
        _text(self.name, "name")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        for collection, name in (
            (self.lines, "lines"),
            (self.towers, "towers"),
            (self.spans, "spans"),
            (self.scans, "scans"),
            (self.analysis_runs, "analysis_runs"),
            (self.routes, "routes"),
        ):
            if not isinstance(collection, tuple):
                raise ValueError(f"{name} must be a tuple")
            ids = [
                getattr(item, f"{name[:-1]}_id", getattr(item, "analysis_run_id", None))
                for item in collection
            ]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} must not contain duplicate IDs")
        line_ids = {line.line_id for line in self.lines}
        route_ids = {route.route_id for route in self.routes}
        tower_ids = {tower.tower_id for tower in self.towers}
        span_ids = {span.span_id for span in self.spans}
        scan_ids = {scan.scan_id for scan in self.scans}
        for line in self.lines:
            if any(tower_id not in tower_ids for tower_id in line.ordered_tower_ids):
                raise ValueError("Line references an unknown Tower")
            if line.parent_line_id is not None and line.parent_line_id not in line_ids:
                raise ValueError("Line references an unknown parent Line")
            if line.parent_line_id == line.line_id:
                raise ValueError("Line cannot be its own parent")
        for route in self.routes:
            if route.line_id not in line_ids:
                raise ValueError("Route references an unknown Line")
            if any(tower_id not in tower_ids for tower_id in route.ordered_tower_ids):
                raise ValueError("Route references an unknown Tower")
            if route.parent_route_id is not None and route.parent_route_id not in route_ids:
                raise ValueError("Route references an unknown parent Route")
            if route.parent_route_id == route.route_id:
                raise ValueError("Route cannot be its own parent")
        for span in self.spans:
            if (
                span.line_id not in line_ids
                or span.start_tower_id not in tower_ids
                or span.end_tower_id not in tower_ids
            ):
                raise ValueError("Span references an unknown Line or Tower")
            if any(tower_id not in tower_ids for tower_id in span.core_tower_ids):
                raise ValueError("Span core_tower_ids contains an unknown Tower")
        for run in self.analysis_runs:
            if run.scan_id not in scan_ids or any(
                span_id not in span_ids for span_id in run.span_ids
            ):
                raise ValueError("AnalysisRun references an unknown Scan or Span")
        policy = _text(self.basemap_policy, "basemap_policy").upper()
        if policy not in {"NO_BASEMAP", "OFFLINE_BASEMAP"}:
            raise ValueError("basemap_policy must be NO_BASEMAP or OFFLINE_BASEMAP")
        object.__setattr__(self, "basemap_policy", policy)

    @classmethod
    def new(cls, name: str, *, metadata: Mapping[str, Any] | None = None) -> Project:
        return cls(stable_id("project"), STAGE_A_SCHEMA_VERSION, name, utc_now(), metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "lines": [item.to_dict() for item in self.lines],
            "towers": [item.to_dict() for item in self.towers],
            "spans": [item.to_dict() for item in self.spans],
            "scans": [item.to_dict() for item in self.scans],
            "analysis_runs": [item.to_dict() for item in self.analysis_runs],
            "basemap_policy": self.basemap_policy,
            "routes": [item.to_dict() for item in self.routes],
        }


__all__ = [
    "STAGE_A_SCHEMA_VERSION",
    "AnalysisRun",
    "AnalysisRunStatus",
    "CoordinateDeclaration",
    "Line",
    "LineRole",
    "MetricDistance",
    "Project",
    "Route",
    "Scan",
    "SourceDescriptor",
    "Span",
    "Tower",
    "UnitStatus",
    "stable_id",
    "to_metric_distance",
    "utc_now",
]
