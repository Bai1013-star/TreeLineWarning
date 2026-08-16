"""Authoritative ordered TowerGroup -> project Span synchronization.

This module owns only topology projection.  It never infers adjacency from
global Tower numbering and never mutates an existing Span's identity or run
history.  A legacy group-less Span may be enriched in place when its ordered
endpoint pair is unambiguous; all other historical topology drift fails closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from core.config.fast_baseline import canonical_digest
from core.contracts.project import (
    AnalysisStage,
    AnalysisStatus,
    SpanAnalysis,
    SpanContract,
    SpanDependencySnapshot,
    SpanGeometry,
    SpanTopologyState,
    derive_span_geometry_digest,
)
from core.project import persistence
from core.project.persistence import RunIndex, SpanRegistry
from core.project.registry import SpanDraft, create_span
from core.project.session import WritableProjectSession

# The endpoint/bounds geometry is unchanged.  This revision records the
# corrected project-to-science coordinate-frame contract so existing runs are
# invalidated through the normal Span dependency snapshot.
SPAN_GEOMETRY_DEFINITION_REVISION = 2


class SpanTopologyError(RuntimeError):
    """Base error for a fail-closed ordered topology projection."""


class TopologyDriftError(SpanTopologyError):
    """Raised when existing historical Span topology is no longer authoritative."""

    code = "TOPOLOGY_DRIFT"


class MissingTowerEndpointError(SpanTopologyError):
    """Raised when a group references a Tower absent from the Tower registry."""

    code = "MISSING_TOWER_ENDPOINT"


@dataclass(frozen=True, slots=True)
class DesiredSpan:
    group_id: str
    within_group_order: int
    from_tower_id: str
    to_tower_id: str


@dataclass(frozen=True, slots=True)
class SpanSynchronizationResult:
    created_span_ids: tuple[str, ...]
    preserved_span_ids: tuple[str, ...]
    migrated_span_ids: tuple[str, ...]
    spans: tuple[SpanContract, ...]


def desired_group_spans(session: WritableProjectSession) -> tuple[DesiredSpan, ...]:
    """Project adjacent pairs from the canonical group order only."""

    loaded = session.loaded_project
    desired: list[DesiredSpan] = []
    seen_towers: set[str] = set()
    for group in loaded.tower_group_registry.groups:
        for tower_id in group.ordered_tower_ids:
            if tower_id in seen_towers:
                raise TopologyDriftError(
                    f"{TopologyDriftError.code}: Tower appears in multiple groups: {tower_id}"
                )
            seen_towers.add(tower_id)
            try:
                loaded.tower_registry.get(tower_id)
            except KeyError as exc:
                raise MissingTowerEndpointError(
                    f"{MissingTowerEndpointError.code}: {group.group_id} references {tower_id}"
                ) from exc
        for order, (from_tower, to_tower) in enumerate(
            zip(group.ordered_tower_ids, group.ordered_tower_ids[1:], strict=False), 1
        ):
            desired.append(DesiredSpan(group.group_id, order, from_tower, to_tower))
    return tuple(desired)


def _span_geometry(session: WritableProjectSession, item: DesiredSpan) -> SpanGeometry:
    from_tower = session.tower_registry.get(item.from_tower_id).current
    to_tower = session.tower_registry.get(item.to_tower_id).current
    start = from_tower.refined_geometry.base_xyz_absolute_m
    end = to_tower.refined_geometry.base_xyz_absolute_m
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise TopologyDriftError(
            f"{TopologyDriftError.code}: coincident endpoint geometry for "
            f"{item.from_tower_id}->{item.to_tower_id}"
        )
    axis = (dx / length, dy / length)
    perp = (-axis[1], axis[0])
    width = 20.0
    halo = 30.0
    z_min = min(start[2], end[2]) - 1.2
    z_max = max(from_tower.refined_geometry.top_xyz_absolute_m[2],
                to_tower.refined_geometry.top_xyz_absolute_m[2]) + 5.0
    config_digest = canonical_digest(
        {
            "adapter": "project-span-geometry-v1",
            "core_half_width_m": width,
            "halo_m": halo,
            "z_ground_margin_m": 1.2,
            "z_top_margin_m": 5.0,
        }
    )

    def bounds(
        s0: float, s1: float, half_width: float
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        corners = []
        for s in (s0, s1):
            for u in (-half_width, half_width):
                corners.append((start[0] + axis[0] * s + perp[0] * u,
                                start[1] + axis[1] * s + perp[1] * u,
                                z_min))
                corners.append((start[0] + axis[0] * s + perp[0] * u,
                                start[1] + axis[1] * s + perp[1] * u,
                                z_max))
        return tuple(min(row[index] for row in corners) for index in range(3)), tuple(
            max(row[index] for row in corners) for index in range(3)
        )

    core_min, core_max = bounds(-0.5, length + 0.5, width)
    halo_min, halo_max = bounds(-halo, length + halo, width)
    source_binding = from_tower.source_binding
    geometry_without_digest = SpanGeometry(
        schema_version="treeline-span-geometry-v1",
        definition_revision=SPAN_GEOMETRY_DEFINITION_REVISION,
        coordinate_context=source_binding.coordinate_context,
        length_m=length,
        direction_xy=axis,
        core_bounds_min_xyz=core_min,
        core_bounds_max_xyz=core_max,
        halo_bounds_min_xyz=halo_min,
        halo_bounds_max_xyz=halo_max,
        configuration_digest=config_digest,
        geometry_digest="0" * 64,
    )
    geometry_digest = derive_span_geometry_digest(
        source_binding=source_binding,
        from_tower=from_tower.revision_reference,
        to_tower=to_tower.revision_reference,
        geometry=geometry_without_digest,
    )
    return replace(geometry_without_digest, geometry_digest=geometry_digest)


def _publish_span_registry(
    session: WritableProjectSession, loaded: Any, registry: SpanRegistry
) -> None:
    published = persistence._publish_generation(
        loaded.project_root,
        loaded.project,
        source_sidecar=loaded.source_sidecar,
        tower_registry=loaded.tower_registry,
        span_registry=registry,
        run_index=loaded.run_index,
        candidate_store=loaded.candidate_store,
        tower_group_registry=loaded.tower_group_registry,
    )
    session._accept_published_generation(published)


def _publish_span_state(
    session: WritableProjectSession,
    loaded: Any,
    *,
    span_registry: SpanRegistry,
    run_index: RunIndex,
) -> None:
    published = persistence._publish_generation(
        loaded.project_root,
        loaded.project,
        source_sidecar=loaded.source_sidecar,
        tower_registry=loaded.tower_registry,
        span_registry=span_registry,
        run_index=run_index,
        candidate_store=loaded.candidate_store,
        tower_group_registry=loaded.tower_group_registry,
    )
    session._accept_published_generation(published)


def synchronize_group_spans(session: WritableProjectSession) -> SpanSynchronizationResult:
    """Synchronize missing adjacent group pairs, preserving compatible history."""

    session.require_strong_source_identity()
    loaded = session._revalidate_authoritative_state()
    desired = desired_group_spans(session)
    desired_by_pair = {(item.from_tower_id, item.to_tower_id): item for item in desired}
    existing_pairs: dict[tuple[str, str], SpanContract] = {}
    migrated: list[SpanContract] = []
    dependency_migrations: set[str] = set()
    preserved: list[str] = []
    for span in loaded.span_registry.records:
        if span.topology_state is not SpanTopologyState.CONFIRMED:
            continue
        key = (span.from_tower_id, span.to_tower_id)
        item = desired_by_pair.get(key)
        if item is None:
            raise TopologyDriftError(
                f"{TopologyDriftError.code}: historical Span {span.span_id} is not in group order"
            )
        if key in existing_pairs:
            raise TopologyDriftError(f"{TopologyDriftError.code}: duplicate pair {key}")
        desired_geometry = _span_geometry(session, item)
        if span.geometry.definition_revision != desired_geometry.definition_revision:
            migrated_geometry = replace(
                span.geometry,
                definition_revision=desired_geometry.definition_revision,
                geometry_digest=derive_span_geometry_digest(
                    source_binding=span.source_binding,
                    from_tower=span.from_tower_reference,
                    to_tower=span.to_tower_reference,
                    geometry=replace(
                        span.geometry,
                        definition_revision=desired_geometry.definition_revision,
                    ),
                ),
            )
            migrated_dependencies = SpanDependencySnapshot(
                source_binding=span.source_binding,
                from_tower=span.from_tower_reference,
                to_tower=span.to_tower_reference,
                geometry_digest=migrated_geometry.geometry_digest,
                configuration_digest=migrated_geometry.configuration_digest,
            )
            analysis = span.analysis
            if (
                analysis.run_dependencies is not None
                and analysis.run_dependencies != migrated_dependencies
            ):
                history = list(analysis.historical_run_ids)
                if analysis.active_run_id is not None and analysis.active_run_id not in history:
                    history.append(analysis.active_run_id)
                analysis = replace(
                    analysis,
                    state=AnalysisStatus.STALE,
                    stage_statuses={stage: AnalysisStatus.STALE for stage in AnalysisStage},
                    historical_run_ids=tuple(history),
                )
            migrated_span = replace(
                span,
                geometry=migrated_geometry,
                analysis=analysis,
            )
            migrated.append(migrated_span)
            dependency_migrations.add(span.span_id)
            span = migrated_span
        existing_pairs[key] = span
        if span.group_id is not None and (
            span.group_id != item.group_id or span.within_group_order != item.within_group_order
        ):
            raise TopologyDriftError(
                f"{TopologyDriftError.code}: {span.span_id} has incompatible group position"
            )
        if span.group_id is None:
            migrated.append(
                replace(span, group_id=item.group_id, within_group_order=item.within_group_order)
            )
        preserved.append(span.span_id)

    if migrated:
        migrated_by_id = {span.span_id: span for span in migrated}
        stale_run_ids: set[str] = set()
        for entry in loaded.run_index.entries:
            migrated_span = migrated_by_id.get(entry.span_id)
            if (
                migrated_span is not None
                and entry.span_id in dependency_migrations
                and entry.status is not AnalysisStatus.STALE
                and entry.span_dependencies != migrated_span.current_dependencies
            ):
                stale_run_ids.add(entry.run_id)
        run_index = RunIndex(
            loaded.run_index.schema_version,
            loaded.run_index.generation_id,
            loaded.run_index.project_id,
            entries=tuple(
                replace(entry, status=AnalysisStatus.STALE)
                if entry.run_id in stale_run_ids
                else entry
                for entry in loaded.run_index.entries
            ),
        )
        stale_by_id: dict[str, SpanContract] = {}
        for span_id in dependency_migrations:
            migrated_span = migrated_by_id[span_id]
            analysis = migrated_span.analysis
            if (
                analysis.run_dependencies is not None
                and analysis.run_dependencies != migrated_span.current_dependencies
            ):
                history = list(analysis.historical_run_ids)
                if analysis.active_run_id is not None and analysis.active_run_id not in history:
                    history.append(analysis.active_run_id)
                analysis = replace(
                    analysis,
                    state=AnalysisStatus.STALE,
                    stage_statuses={stage: AnalysisStatus.STALE for stage in AnalysisStage},
                    historical_run_ids=tuple(history),
                )
                stale_by_id[span_id] = replace(migrated_span, analysis=analysis)
        migrated_by_id.update(stale_by_id)
        registry = SpanRegistry(
            loaded.span_registry.schema_version,
            loaded.span_registry.generation_id,
            loaded.span_registry.project_id,
            reserved_sequences=loaded.span_registry.reserved_sequences,
        records=tuple(
            migrated_by_id.get(span.span_id, span)
            for span in loaded.span_registry.records
        ),
        )
        _publish_span_state(
            session,
            loaded,
            span_registry=registry,
            run_index=run_index,
        )

    created: list[str] = []
    for item in desired:
        if (item.from_tower_id, item.to_tower_id) in existing_pairs:
            continue
        from_tower = session.tower_registry.get(item.from_tower_id).current
        to_tower = session.tower_registry.get(item.to_tower_id).current
        geometry = _span_geometry(session, item)
        draft = SpanDraft(
            schema_version="treeline-span-v1",
            display_name=f"{item.group_id} {item.from_tower_id} -> {item.to_tower_id}",
            from_tower_id=item.from_tower_id,
            to_tower_id=item.to_tower_id,
            from_tower_revision=from_tower.revision,
            to_tower_revision=to_tower.revision,
            source_binding=from_tower.source_binding,
            topology_state=SpanTopologyState.CONFIRMED,
            geometry=geometry,
            analysis=SpanAnalysis.not_started(),
            group_id=item.group_id,
            within_group_order=item.within_group_order,
        )
        span = create_span(session, draft)
        created.append(span.span_id)
        existing_pairs[(item.from_tower_id, item.to_tower_id)] = span
    current = tuple(sorted(session.span_registry.records, key=lambda span: int(span.span_id[1:])))
    return SpanSynchronizationResult(
        created_span_ids=tuple(created),
        preserved_span_ids=tuple(preserved),
        migrated_span_ids=tuple(span.span_id for span in migrated),
        spans=current,
    )


__all__ = [
    "DesiredSpan",
    "MissingTowerEndpointError",
    "SpanSynchronizationResult",
    "SpanTopologyError",
    "TopologyDriftError",
    "desired_group_spans",
    "synchronize_group_spans",
]
