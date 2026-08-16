"""V5A3 tower-group services (ordered tower groups + legacy bootstrap).

The group registry is generation-bound and authoritative: the EXPLICIT ordered
tower list inside each ``G###`` is the ONLY future order authority.  Different
groups are independent and never imply adjacency across their boundaries.
Tower IDs remain globally unique; a group is never a second tower namespace.

Mutations publish successor generations through the existing copy-on-write
publisher, so allocating a ``G###`` (or appending a confirmed ``T###`` to a
group) is generation-consistent with the Tower registry and candidate store.
``GROUP_BOOTSTRAP_PENDING`` legacy group-less generations are resolved to a
successor generation carrying the default ``G001`` (区域 1) whose ordered list
is the existing confirmed towers in ``T###`` (confirmation) order.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from core.contracts.tower_groups import (
    TOWER_GROUP_REGISTRY_SCHEMA_VERSION,
    TOWER_GROUP_SCHEMA_VERSION,
    TowerGroup,
    TowerGroupRegistry,
    group_display_id_for_sequence,
)
from core.project import persistence
from core.project.persistence import PostPointerDurabilityError
from core.project.session import (
    ProjectSessionClosedError,
    ProjectSessionError,
    WritableProjectSession,
)

DEFAULT_BOOTSTRAP_GROUP_DISPLAY_NAME = "区域 1"


class TowerGroupError(RuntimeError):
    """Base error for tower-group mutations."""


class InvalidTowerGroupState(TowerGroupError, ValueError):
    """Raised when a group request violates a frozen invariant."""


class TowerGroupPublicationUncertainError(TowerGroupError):
    """Raised after pointer replacement when root durability is uncertain."""

    def __init__(self, message: str, *, original: PostPointerDurabilityError) -> None:
        super().__init__(message)
        self.original = original
        self.pointer_replaced = original.pointer_replaced


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _timestamp_at_or_after(value: str) -> str:
    now = datetime.now(UTC)
    try:
        previous = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        previous = now
    if now < previous:
        now = previous
    return now.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _updated_project(project: Any, **changes: Any) -> Any:
    return replace(project, updated_at=_timestamp_at_or_after(project.updated_at), **changes)


def _require_writable(session: WritableProjectSession) -> Any:
    if type(session) is not WritableProjectSession:
        raise InvalidTowerGroupState("tower-group mutation requires a writable session")
    try:
        return session._begin_registry_mutation()
    except (ProjectSessionClosedError, ProjectSessionError):
        raise
    except (AttributeError, TypeError) as exc:
        raise InvalidTowerGroupState("tower-group mutation received a fabricated session") from exc


def _publish(
    session: WritableProjectSession,
    loaded: Any,
    *,
    tower_group_registry: TowerGroupRegistry,
) -> Any:
    try:
        published = persistence._publish_generation(
            loaded.project_root,
            _updated_project(loaded.project),
            source_sidecar=loaded.source_sidecar,
            tower_registry=loaded.tower_registry,
            span_registry=loaded.span_registry,
            run_index=loaded.run_index,
            candidate_store=loaded.candidate_store,
            tower_group_registry=tower_group_registry,
        )
    except PostPointerDurabilityError as exc:
        try:
            session._revalidate_authoritative_state()
        except BaseException as revalidation_error:
            raise TowerGroupPublicationUncertainError(
                "tower-group publication is uncertain and authoritative revalidation "
                f"failed: {revalidation_error}",
                original=exc,
            ) from exc
        raise TowerGroupPublicationUncertainError(
            "tower-group publication is uncertain; authoritative state was reopened "
            "under the writer lock",
            original=exc,
        ) from exc
    session._accept_published_generation(published)
    return published


def _confirmed_tower_ids_in_order(loaded: Any) -> tuple[str, ...]:
    """Confirmed towers in T### (confirmation) order from the authoritative state."""
    tower_ids = {entry.tower_id for entry in loaded.tower_registry.records}
    confirmation_tower_ids = {
        confirmation.confirmed_tower.tower_id
        for confirmation in loaded.candidate_store.confirmations
    }
    ordered = sorted(
        tower_ids, key=lambda tower_id: int(tower_id[1:])
    )
    # Prefer the confirmation-derived set when it is non-empty; otherwise use the
    # full Tower registry (ordered workflow towers are always confirmed).
    base = ordered if not confirmation_tower_ids else sorted(
        confirmation_tower_ids, key=lambda tower_id: int(tower_id[1:])
    )
    return tuple(base)


def ensure_ordered_group(session: WritableProjectSession) -> TowerGroupRegistry:
    """Resolve a LEGACY group-less generation to G001 (区域 1) via a successor generation.

    No-op when the project already carries a group registry.  The default group
    contains the existing confirmed towers in T### (confirmation) order; Tower
    IDs and artifacts are unchanged and historical generations are untouched.
    """
    loaded = _require_writable(session)
    registry = loaded.tower_group_registry
    if not registry.bootstrap_pending:
        return registry
    tower_ids = _confirmed_tower_ids_in_order(loaded)
    group = TowerGroup(
        schema_version=TOWER_GROUP_SCHEMA_VERSION,
        group_id="G001",
        display_name=DEFAULT_BOOTSTRAP_GROUP_DISPLAY_NAME,
        ordered_tower_ids=tower_ids,
        created_at=_utc_timestamp(),
    )
    new_registry = TowerGroupRegistry(
        schema_version=TOWER_GROUP_REGISTRY_SCHEMA_VERSION,
        generation_id=loaded.generation_id,
        project_id=loaded.project.project_id,
        next_group_sequence=2,
        groups=(group,),
        bootstrap_pending=False,
    )
    _publish(session, loaded, tower_group_registry=new_registry)
    return new_registry


def create_tower_group(
    session: WritableProjectSession,
    *,
    display_name: str,
) -> TowerGroup:
    """Allocate the next canonical ``G###`` and persist a successor generation.

    Group IDs are monotonic and never reused; ``display_name`` is initialized at
    creation (rename is deferred).  No deletion/reuse is supported.
    """
    if type(session) is not WritableProjectSession:
        raise InvalidTowerGroupState("create_tower_group requires a writable session")
    loaded = _require_writable(session)
    registry = loaded.tower_group_registry
    if registry.bootstrap_pending:
        ensure_ordered_group(session)
        loaded = session._begin_registry_mutation()
        registry = loaded.tower_group_registry
    sequence = registry.next_group_sequence
    group_id = group_display_id_for_sequence(sequence)
    group = TowerGroup(
        schema_version=TOWER_GROUP_SCHEMA_VERSION,
        group_id=group_id,
        display_name=display_name,
        ordered_tower_ids=(),
        created_at=_utc_timestamp(),
    )
    new_registry = TowerGroupRegistry(
        schema_version=TOWER_GROUP_REGISTRY_SCHEMA_VERSION,
        generation_id=loaded.generation_id,
        project_id=loaded.project.project_id,
        next_group_sequence=sequence + 1,
        groups=(*registry.groups, group),
    )
    _publish(session, loaded, tower_group_registry=new_registry)
    return group


def append_confirmed_tower_to_group(
    session: WritableProjectSession,
    *,
    group_id: str,
    tower_id: str,
) -> TowerGroup:
    """Atomically append one confirmed ``T###`` to a group's explicit ordered list.

    The append is generation-consistent with the Tower registry / candidate
    store / confirmation in the same successor publication.  A tower already in
    another active group is rejected (exclusive membership); a missing tower or
    group fails closed.
    """
    if type(session) is not WritableProjectSession:
        raise InvalidTowerGroupState("append_confirmed_tower_to_group requires a writable session")
    loaded = _require_writable(session)
    registry = loaded.tower_group_registry
    if registry.bootstrap_pending:
        raise InvalidTowerGroupState("append requires a bootstrapped group registry")
    try:
        group = registry.get(group_id)
    except KeyError as exc:
        raise InvalidTowerGroupState(f"group {group_id} is absent from the project") from exc
    try:
        loaded.tower_registry.get(tower_id)
    except KeyError as exc:
        raise InvalidTowerGroupState(f"tower {tower_id} is absent from the project") from exc
    for other in registry.groups:
        if tower_id in other.ordered_tower_ids:
            raise InvalidTowerGroupState(
                f"tower {tower_id} already belongs to group {other.group_id}"
            )
    new_group = TowerGroup(
        schema_version=group.schema_version,
        group_id=group.group_id,
        display_name=group.display_name,
        ordered_tower_ids=(*group.ordered_tower_ids, tower_id),
        created_at=group.created_at,
    )
    updated_groups = tuple(
        new_group if item.group_id == group.group_id else item for item in registry.groups
    )
    new_registry = TowerGroupRegistry(
        schema_version=registry.schema_version,
        generation_id=loaded.generation_id,
        project_id=loaded.project.project_id,
        next_group_sequence=registry.next_group_sequence,
        groups=updated_groups,
    )
    _publish(session, loaded, tower_group_registry=new_registry)
    return new_group


def build_registry_with_tower(
    registry: TowerGroupRegistry,
    *,
    group_id: str,
    tower_id: str,
) -> TowerGroupRegistry:
    """Return a successor registry with one tower appended to a group (no publication).

    Pure in-memory construction used by the atomic ordered CONFIRM path so the
    ``T###`` allocation, candidate confirmation, Tower registry update and group
    membership append publish in ONE generation.  Fails closed on a missing
    group or a tower already present in another active group.
    """
    if type(registry) is not TowerGroupRegistry:
        raise InvalidTowerGroupState("registry must be a TowerGroupRegistry")
    try:
        group = registry.get(group_id)
    except KeyError as exc:
        raise InvalidTowerGroupState(f"group {group_id} is absent from the project") from exc
    for other in registry.groups:
        if tower_id in other.ordered_tower_ids:
            raise InvalidTowerGroupState(
                f"tower {tower_id} already belongs to group {other.group_id}"
            )
    new_group = TowerGroup(
        schema_version=group.schema_version,
        group_id=group.group_id,
        display_name=group.display_name,
        ordered_tower_ids=(*group.ordered_tower_ids, tower_id),
        created_at=group.created_at,
    )
    updated_groups = tuple(
        new_group if item.group_id == group.group_id else item for item in registry.groups
    )
    return TowerGroupRegistry(
        schema_version=registry.schema_version,
        generation_id=registry.generation_id,
        project_id=registry.project_id,
        next_group_sequence=registry.next_group_sequence,
        groups=updated_groups,
    )


def tower_group_state(session: Any) -> dict[str, Any]:
    """Authoritative group state projection (registry is authoritative)."""
    registry = session.tower_group_registry
    groups: list[dict[str, Any]] = []
    for group in registry.groups:
        tower_rows: list[dict[str, Any]] = []
        for tower_id in group.ordered_tower_ids:
            try:
                entry = session.tower_registry.get(tower_id)
            except KeyError:
                continue
            revision = entry.current
            geometry = revision.refined_geometry
            tower_rows.append(
                {
                    "tower_id": tower_id,
                    "revision": revision.revision,
                    "display_xyz_absolute_m": list(geometry.base_xyz_absolute_m),
                    "top_xyz_absolute_m": list(geometry.top_xyz_absolute_m),
                    "height_m": geometry.height_m,
                }
            )
        groups.append(
            {
                "group_id": group.group_id,
                "display_name": group.display_name,
                "created_at": group.created_at,
                "ordered_tower_ids": list(group.ordered_tower_ids),
                "tower_count": len(group.ordered_tower_ids),
                "next_group_position": len(group.ordered_tower_ids) + 1,
                "towers": tower_rows,
            }
        )
    return {
        "groups": groups,
        "next_group_sequence": registry.next_group_sequence,
        "global_next_tower_id": f"T{session.project.next_tower_sequence:03d}",
        "bootstrap_pending": registry.bootstrap_pending,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_GROUP_DISPLAY_NAME",
    "InvalidTowerGroupState",
    "TowerGroupError",
    "TowerGroupPublicationUncertainError",
    "append_confirmed_tower_to_group",
    "build_registry_with_tower",
    "create_tower_group",
    "ensure_ordered_group",
    "tower_group_state",
]
