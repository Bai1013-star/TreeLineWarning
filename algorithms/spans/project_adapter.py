"""Tracked project-to-single-Span science adapter.

The adapter is deliberately a narrow boundary: project contracts supply the
authoritative endpoints and dependency digests, while the existing canonical
single-Span algorithms remain the science implementation.  A caller supplies
the concrete science executor (production launchers can bind the existing
wire/tree/clearance pipeline); this module never treats TowerLocalMembership
as vegetation authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config.fast_baseline import FastBaselineConfig, canonical_digest
from core.contracts.project import ArtifactReference, SpanContract, TowerContract


class ScienceAdapterError(RuntimeError):
    """Honest failure from prerequisite validation or the bound science runner."""


# The project adapter must not maintain a second, hidden corridor geometry.
# These are the existing authoritative FAST span corridor and halo values.
_TRACKED_SPAN_REGION_CONFIG = FastBaselineConfig()
PROJECT_SPAN_CORE_HALF_WIDTH_M = _TRACKED_SPAN_REGION_CONFIG.span_half_width_m
PROJECT_SPAN_LONGITUDINAL_HALO_M = _TRACKED_SPAN_REGION_CONFIG.span_halo_m
PROJECT_SPAN_GEOMETRY_CONFIGURATION_DIGEST = canonical_digest(
    {
        "adapter": "project-span-geometry-v1",
        "core_half_width_m": PROJECT_SPAN_CORE_HALF_WIDTH_M,
        "halo_m": PROJECT_SPAN_LONGITUDINAL_HALO_M,
        "z_ground_margin_m": 1.2,
        "z_top_margin_m": 5.0,
    }
)
# Vegetation role admission digest: HIGH RECALL (no broad wire-role exclusion).
# The OLD vegetation role index was built with the span-geometry digest AND the
# entire wire-role source set excluded, which systematically dropped
# conductor-adjacent vegetation.  This new digest identity makes stale caches
# fail closed under the corrected admission semantics.
PROJECT_SPAN_VEGETATION_ROLE_CONFIGURATION_DIGEST = canonical_digest(
    {
        "adapter": "project-source-vegetation-role-v3",
        "source_index_schema": "treeline-span-role-tile-index-v1",
        "admission": "HIGH_RECALL_NO_BROAD_WIRE_EXCLUSION",
    }
)
RISK_POLICY = "exact-tree-wire"
RISK_THRESHOLD_M = 2.0
RISK_CONFIGURATION_DIGEST = canonical_digest(
    {"risk_threshold_m": RISK_THRESHOLD_M, "policy": RISK_POLICY}
)


def wire_role_index_configuration_digest(
    config: FastBaselineConfig | None = None,
) -> str:
    """Digest only dependencies that can alter source-wide wire-role contents.

    The role index is a spatial representation of the already-admitted wire
    source rows.  It must not inherit tree, clearance, viewer or per-span
    geometry settings through the monolithic FAST configuration digest.
    """

    tracked = config or FastBaselineConfig()
    values = tracked.to_digest_dict()
    wire_values = {
        name: value
        for name, value in values.items()
        if name.startswith("wire_") or name == "non_certified_marker"
    }
    return canonical_digest(
        {
            "schema": "treeline-source-wire-role-index-dependencies-v1",
            "role_index_schema": "treeline-span-role-tile-index-v1",
            "role": "wire",
            "wire_configuration": wire_values,
        }
    )


def span_region_configuration() -> dict[str, Any]:
    """Return the auditable wire/vegetation region provenance mapping."""

    tracked = FastBaselineConfig()
    return {
        "wire_region": {
            "half_width_m": tracked.span_half_width_m,
            "longitudinal_halo_m": tracked.span_halo_m,
            "source": "tracked FastBaselineConfig",
            "configuration_digest": tracked.computed_digest(),
        },
        "vegetation_search_region": {
            "half_width_m": PROJECT_SPAN_CORE_HALF_WIDTH_M,
            "longitudinal_halo_m": PROJECT_SPAN_LONGITUDINAL_HALO_M,
            "source": "tracked FastBaselineConfig span corridor",
            "configuration_digest": PROJECT_SPAN_GEOMETRY_CONFIGURATION_DIGEST,
        },
        "risk": {
            "threshold_m": RISK_THRESHOLD_M,
            "policy": RISK_POLICY,
            "configuration_digest": RISK_CONFIGURATION_DIGEST,
        },
    }


@dataclass(frozen=True, slots=True)
class SpanAnalysisInput:
    project_id: str
    source_identity: Mapping[str, Any]
    group_id: str
    span_id: str
    from_tower_id: str
    to_tower_id: str
    from_tower_revision: int
    to_tower_revision: int
    from_endpoint_geometry: Mapping[str, Any]
    to_endpoint_geometry: Mapping[str, Any]
    span_geometry: Mapping[str, Any]
    science_configuration_digest: str
    risk_configuration_digest: str
    validated_source_preparation_references: tuple[str, ...]
    source_point_count: int | None = None

    def __post_init__(self) -> None:
        if not self.project_id or not self.group_id or not self.span_id:
            raise ValueError("project, group, and span identities are required")
        if not self.from_tower_id or not self.to_tower_id or self.from_tower_id == self.to_tower_id:
            raise ValueError("Span endpoints must be distinct")
        if type(self.from_tower_revision) is not int or self.from_tower_revision <= 0:
            raise ValueError("from_tower_revision must be positive")
        if type(self.to_tower_revision) is not int or self.to_tower_revision <= 0:
            raise ValueError("to_tower_revision must be positive")
        for name, value in (
            ("science_configuration_digest", self.science_configuration_digest),
            ("risk_configuration_digest", self.risk_configuration_digest),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_identity", dict(self.source_identity))
        object.__setattr__(self, "from_endpoint_geometry", dict(self.from_endpoint_geometry))
        object.__setattr__(self, "to_endpoint_geometry", dict(self.to_endpoint_geometry))
        object.__setattr__(self, "span_geometry", dict(self.span_geometry))
        object.__setattr__(
            self,
            "validated_source_preparation_references",
            tuple(self.validated_source_preparation_references),
        )
        if self.source_point_count is not None and (
            type(self.source_point_count) is not int or self.source_point_count < 0
        ):
            raise ValueError("source_point_count must be a non-negative integer or None")

    @classmethod
    def from_project(
        cls, project_id: str, span: SpanContract, towers: Mapping[str, TowerContract]
    ) -> SpanAnalysisInput:
        if span.group_id is None or span.within_group_order is None:
            raise ScienceAdapterError("SPAN_GROUP_METADATA_MISSING")
        try:
            from_tower = towers[span.from_tower_id]
            to_tower = towers[span.to_tower_id]
        except KeyError as exc:
            raise ScienceAdapterError("MISSING_TOWER_ENDPOINT") from exc
        if (
            from_tower.revision != span.from_tower_revision
            or to_tower.revision != span.to_tower_revision
        ):
            raise ScienceAdapterError("TOWER_REVISION_MISMATCH")
        source_identity = span.source_binding.to_dict()
        risk_digest = canonical_digest({"risk_threshold_m": 2.0, "policy": "exact-tree-wire"})
        return cls(
            project_id=project_id,
            source_identity=source_identity,
            group_id=span.group_id,
            span_id=span.span_id,
            from_tower_id=span.from_tower_id,
            to_tower_id=span.to_tower_id,
            from_tower_revision=span.from_tower_revision,
            to_tower_revision=span.to_tower_revision,
            from_endpoint_geometry=from_tower.refined_geometry.to_dict(),
            to_endpoint_geometry=to_tower.refined_geometry.to_dict(),
            span_geometry=span.geometry.to_dict(),
            science_configuration_digest=span.geometry.configuration_digest,
            risk_configuration_digest=risk_digest,
            validated_source_preparation_references=(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "source_identity": dict(self.source_identity),
            "group_id": self.group_id,
            "span_id": self.span_id,
            "from_tower_id": self.from_tower_id,
            "to_tower_id": self.to_tower_id,
            "from_tower_revision": self.from_tower_revision,
            "to_tower_revision": self.to_tower_revision,
            "from_endpoint_geometry": dict(self.from_endpoint_geometry),
            "to_endpoint_geometry": dict(self.to_endpoint_geometry),
            "span_geometry": dict(self.span_geometry),
            "science_configuration_digest": self.science_configuration_digest,
            "risk_configuration_digest": self.risk_configuration_digest,
            "validated_source_preparation_references": list(
                self.validated_source_preparation_references
            ),
            "source_point_count": self.source_point_count,
        }


@dataclass(frozen=True, slots=True)
class SourceSharedPreparation:
    source_binding_digest: str
    configuration_digest: str
    manifest_reference: str
    authority: str = "FULL_RESOLUTION_RAW"
    tower_membership_used_for_vegetation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_binding_digest": self.source_binding_digest,
            "configuration_digest": self.configuration_digest,
            "manifest_reference": self.manifest_reference,
            "authority": self.authority,
            "tower_membership_used_for_vegetation": self.tower_membership_used_for_vegetation,
        }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_source_shared(
    project_root: Path, input_data: SpanAnalysisInput
) -> SourceSharedPreparation:
    """Create/reuse only a digest-bound preparation manifest, never global RAW copies."""

    source_digest = canonical_digest(dict(input_data.source_identity))
    configuration_digest = input_data.science_configuration_digest
    relative = (
        Path("artifacts") / "project-runs" / "source-shared" / source_digest / "preparation.json"
    )
    path = project_root / relative
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("source_binding_digest") == source_digest
            and payload.get("configuration_digest") == configuration_digest
        ):
            # Bind only the explicit role-index directories.  In particular,
            # ``wire-tile-cache``/``vegetation-tile-cache`` are not accepted:
            # those names were used by the tower RAW cache and carry no role
            # semantics or classification contract.
            changed = False
            for role in ("wire", "vegetation"):
                role_index = path.parent / f"{role}-role-tile-index"
                key = f"{role}_role_tile_index_reference"
                existing = payload.get(key)
                if isinstance(existing, str) and (project_root / existing).is_dir():
                    continue
                if role_index.is_dir():
                    reference = role_index.relative_to(project_root).as_posix()
                    if payload.get(key) != reference:
                        payload[key] = reference
                        changed = True
            if changed:
                _write_json_atomic(path, payload)
            return SourceSharedPreparation(source_digest, configuration_digest, relative.as_posix())
    payload = {
        "schema": "treeline-project-source-shared-preparation-v1",
        "source_binding_digest": source_digest,
        "configuration_digest": configuration_digest,
        "authority": "FULL_RESOLUTION_RAW",
        "tower_membership_used_for_vegetation": False,
        "raw_to_representative_mapping": None,
        "ground_hag_reference": None,
        "coarse_ground_reference": None,
        "region_configuration": span_region_configuration(),
        "source_spatial_query": (
            "role-specific source tile indexes are required for production Span-local analysis"
        ),
    }
    for role in ("wire", "vegetation"):
        tile_cache = path.parent / f"{role}-role-tile-index"
        if tile_cache.is_dir():
            payload[f"{role}_role_tile_index_reference"] = tile_cache.relative_to(
                project_root
            ).as_posix()
    if input_data.source_point_count is not None:
        payload["source_point_count"] = input_data.source_point_count
    _write_json_atomic(path, payload)
    return SourceSharedPreparation(source_digest, configuration_digest, relative.as_posix())


@dataclass(frozen=True, slots=True)
class SpanScienceResult:
    tree_count: int
    minimum_clearance_m: float | None
    risk_threshold_m: float
    risk_counts: Mapping[str, int]
    artifact_files: Mapping[str, str] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    # Runtime-only terminal identity handoff.  It is deliberately absent from
    # serialized artifacts so tentative IDs cannot be published as canonical.
    identity_terminal_preparation: Any | None = None
    identity_terminal_publish: Callable[[Any], SpanScienceResult] | None = None


ScienceExecutor = Callable[[SpanAnalysisInput, SourceSharedPreparation, Path], SpanScienceResult]


def run_canonical_span_science(
    input_data: SpanAnalysisInput,
    preparation: SourceSharedPreparation,
    output_root: Path,
    *,
    executor: ScienceExecutor | None = None,
) -> SpanScienceResult:
    """Invoke the existing canonical single-Span science through an explicit seam."""

    if (
        preparation.authority != "FULL_RESOLUTION_RAW"
        or preparation.tower_membership_used_for_vegetation
    ):
        raise ScienceAdapterError("INVALID_SOURCE_AUTHORITY")
    if executor is None:
        raise ScienceAdapterError("SCIENCE_EXECUTOR_NOT_BOUND")
    result = executor(input_data, preparation, output_root)
    if type(result) is not SpanScienceResult:
        raise ScienceAdapterError("SCIENCE_EXECUTOR_RETURNED_INVALID_RESULT")
    if result.tree_count < 0 or result.risk_threshold_m <= 0.0:
        raise ScienceAdapterError("SCIENCE_RESULT_INVALID")
    return result


def _artifact_reference(project_root: Path, path: Path, role: str) -> ArtifactReference:
    resolved = path.resolve()
    relative = resolved.relative_to(project_root.resolve()).as_posix()
    data = resolved.read_bytes()
    return ArtifactReference(
        relative_path=relative,
        role=role,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        schema_version="treeline-span-artifact-v1",
    )


def publish_terminal_artifact_index(
    project_root: Path,
    input_data: SpanAnalysisInput,
    preparation: SourceSharedPreparation,
    result: SpanScienceResult,
    output_root: Path,
) -> ArtifactReference:
    """Write the compact immutable per-Span terminal index and return its reference."""

    references: dict[str, dict[str, Any]] = {}
    for role, value in dict(result.artifact_files).items():
        path = Path(value)
        if not path.is_absolute():
            path = output_root / path
        if not path.is_file():
            raise ScienceAdapterError(f"SCIENCE_ARTIFACT_MISSING:{role}")
        references[role] = _artifact_reference(project_root, path, role).to_dict()
    missing_roles = {"wire", "tree", "clearance"} - set(references)
    if missing_roles:
        raise ScienceAdapterError("SCIENCE_ARTIFACT_MISSING:" + ",".join(sorted(missing_roles)))
    relative = (
        Path("artifacts")
        / "project-runs"
        / "groups"
        / input_data.group_id
        / input_data.span_id
        / output_root.name
        / "terminal-artifact-index.json"
    )
    path = project_root / relative
    payload = {
        "schema": "treeline-span-terminal-artifact-index-v1",
        "project_id": input_data.project_id,
        "group_id": input_data.group_id,
        "span_id": input_data.span_id,
        "run_id": output_root.name,
        "status": "COMPLETE",
        "summary": {
            "tree_count": result.tree_count,
            "minimum_clearance_m": result.minimum_clearance_m,
            "risk_threshold_m": result.risk_threshold_m,
            "risk_counts": dict(result.risk_counts),
        },
        "provenance": {
            "input": input_data.to_dict(),
            "source_preparation": preparation.to_dict(),
            "source_authority": "FULL_RESOLUTION_RAW",
            "tower_membership_used_for_vegetation": False,
        },
        "artifacts": references,
        "diagnostics": dict(result.diagnostics),
    }
    _write_json_atomic(path, payload)
    return _artifact_reference(project_root, path, "terminal-artifact-index")


__all__ = [
    "ScienceAdapterError",
    "ScienceExecutor",
    "SourceSharedPreparation",
    "SpanAnalysisInput",
    "SpanScienceResult",
    "span_region_configuration",
    "wire_role_index_configuration_digest",
    "prepare_source_shared",
    "publish_terminal_artifact_index",
    "run_canonical_span_science",
]
