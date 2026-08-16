"""V5A3 ordered tower groups (G001, G002, ...) contracts.

An ordered tower group is an explicit, project-scoped, monotonic list of
confirmed towers (``T###``) in the Owner's chosen order.  The EXPLICIT ordered
list is the ONLY future order authority; physical/corridor order is NEVER
inferred from numeric ``T###`` adjacency.  Different groups are independent and
never imply adjacency across their boundaries.

Tower IDs remain globally unique across the whole project and independent of
group identity (``T###`` never restarts inside a group, and a group is not a
second tower namespace).  One Tower belongs to at most one active ordered group.

This module performs no file access and no persistence; it only validates and
serializes the frozen group contracts.  A ``bootstrap_pending`` registry is the
in-memory representation of a LEGACY group-less generation (derived at load
time from the sealed generation manifest, never persisted); the service layer
resolves it to a successor generation with the default ``G001``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

TOWER_GROUP_SCHEMA_VERSION = "treeline-tower-group-v1"
TOWER_GROUP_REGISTRY_SCHEMA_VERSION = "treeline-tower-group-registry-v1"

# Frozen G### display domain: literal G followed by exactly three decimal digits
# (G001..G999).  No extra leading zeros, no lowercase, no signs.
_GROUP_ID_RE = re.compile(r"G([0-9]{3})\Z")
# Tower IDs are the existing project-global display domain (T###).
_TOWER_ID_RE = re.compile(r"T([0-9]{3,})\Z")
_OPAQUE_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_MAX_DISPLAY_NAME_LENGTH = 64
_MAX_GROUP_COUNT = 999


def group_display_id_for_sequence(sequence: int) -> str:
    """Return the immutable project-scoped display ID for a group sequence value."""
    sequence = _positive_int(sequence, "sequence")
    if sequence > _MAX_GROUP_COUNT:
        raise ValueError(
            f"sequence must not exceed the G001-G999 group domain ({_MAX_GROUP_COUNT})"
        )
    return f"G{sequence:03d}"


# ---------------------------------------------------------------------------
# Private scalar validation helpers (mirror the repo contract conventions)
# ---------------------------------------------------------------------------


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _exact_string(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a non-empty built-in string without surrounding whitespace"
        )
    return value


def _timestamp(value: Any, name: str) -> str:
    text = _exact_string(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 UTC timestamp") from exc
    return text


def _group_id(value: Any) -> str:
    text = _exact_string(value, "group_id")
    matched = _GROUP_ID_RE.fullmatch(text)
    if matched is None or int(matched.group(1)) <= 0 or int(matched.group(1)) > _MAX_GROUP_COUNT:
        raise ValueError("group_id must be a canonical G001-G999 group ID")
    return text


def _tower_id(value: Any, name: str = "ordered_tower_ids") -> str:
    if type(value) is not str or value != value.strip():
        raise ValueError(f"{name} must contain canonical T### tower IDs")
    matched = _TOWER_ID_RE.fullmatch(value)
    if matched is None or int(matched.group(1)) <= 0:
        raise ValueError(f"{name} must contain canonical T### tower IDs")
    return value


def _display_name(value: Any) -> str:
    text = _exact_string(value, "display_name")
    if len(text) > _MAX_DISPLAY_NAME_LENGTH:
        raise ValueError(
            f"display_name must not exceed {_MAX_DISPLAY_NAME_LENGTH} characters"
        )
    if any(ord(char) < 0x20 for char in text):
        raise ValueError("display_name must not contain control characters")
    return text


def _freeze_tuple(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


# ---------------------------------------------------------------------------
# TowerGroup
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TowerGroup:
    """One explicit ordered list of confirmed towers in one project group."""

    schema_version: str
    group_id: str
    display_name: str
    ordered_tower_ids: tuple[str, ...]
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_GROUP_SCHEMA_VERSION:
            raise ValueError("unsupported tower-group schema version")
        _group_id(self.group_id)
        object.__setattr__(self, "display_name", _display_name(self.display_name))
        ordered = _freeze_tuple(self.ordered_tower_ids, "ordered_tower_ids")
        for tower_id in ordered:
            _tower_id(tower_id)
        if len(set(ordered)) != len(ordered):
            raise ValueError("ordered_tower_ids must not contain duplicate tower IDs")
        object.__setattr__(self, "ordered_tower_ids", ordered)
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "display_name": self.display_name,
            "ordered_tower_ids": list(self.ordered_tower_ids),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TowerGroup:
        if type(data) is not dict:
            raise ValueError("TowerGroup must be a JSON object")
        fields = set(cls.__dataclass_fields__)
        require_fields(data, fields, "TowerGroup")
        reject_unknown_fields(data, fields, "TowerGroup")
        values = dict(data)
        values["ordered_tower_ids"] = _freeze_tuple(
            values["ordered_tower_ids"], "ordered_tower_ids"
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# TowerGroupRegistry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TowerGroupRegistry:
    """Immutable project group registry with a monotonic, never-reused sequence.

    ``bootstrap_pending`` is a transient in-memory marker for a LEGACY
    group-less generation (it is never persisted; it is derived at load time
    from the sealed generation manifest).  The service layer resolves it to a
    successor generation carrying the default ``G001``.
    """

    schema_version: str
    generation_id: str
    project_id: str
    next_group_sequence: int
    groups: tuple[TowerGroup, ...] = ()
    bootstrap_pending: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_GROUP_REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported tower-group-registry schema version")
        if type(self.generation_id) is not str or not self.generation_id:
            raise ValueError("generation_id must be a non-empty string")
        if type(self.project_id) is not str or not self.project_id:
            raise ValueError("project_id must be a non-empty string")
        if type(self.bootstrap_pending) is not bool:
            raise ValueError("bootstrap_pending must be a boolean")
        groups = _freeze_tuple(self.groups, "groups")
        if any(type(item) is not TowerGroup for item in groups):
            raise ValueError("groups must contain TowerGroup values")
        group_ids = tuple(item.group_id for item in groups)
        for group_id in group_ids:
            _group_id(group_id)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("tower group registry contains duplicate group identities")
        if group_ids != tuple(sorted(group_ids, key=lambda value: int(value[1:]))):
            raise ValueError("tower group registry groups must be in sequence order")
        sequence = _positive_int(self.next_group_sequence, "next_group_sequence")
        represented = tuple(int(group_id[1:]) for group_id in group_ids)
        if sequence <= max(represented, default=0):
            raise ValueError("next_group_sequence is not above every represented group")
        # Exclusive tower membership: one Tower in at most one active group.
        seen_towers: set[str] = set()
        for group in groups:
            for tower_id in group.ordered_tower_ids:
                if tower_id in seen_towers:
                    raise ValueError(
                        f"Tower {tower_id} appears in more than one active tower group"
                    )
                seen_towers.add(tower_id)
        object.__setattr__(self, "groups", groups)

    def get(self, group_id: str) -> TowerGroup:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(group_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "project_id": self.project_id,
            "next_group_sequence": self.next_group_sequence,
            "groups": [group.to_dict() for group in self.groups],
        }

    @classmethod
    def from_dict(cls, data: Any) -> TowerGroupRegistry:
        if type(data) is not dict:
            raise ValueError("TowerGroupRegistry must be a JSON object")
        fields = set(cls.__dataclass_fields__)
        require_fields(data, fields - {"bootstrap_pending"}, "TowerGroupRegistry")
        reject_unknown_fields(data, fields, "TowerGroupRegistry")
        values = dict(data)
        values["groups"] = tuple(
            TowerGroup.from_dict(item)
            for item in _freeze_tuple(values["groups"], "groups")
        )
        values["bootstrap_pending"] = False
        return cls(**values)


def require_fields(data: dict[str, Any], fields: set[str], name: str) -> None:
    missing = fields - set(data)
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")


def reject_unknown_fields(data: dict[str, Any], fields: set[str], name: str) -> None:
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")


__all__ = [
    "TOWER_GROUP_REGISTRY_SCHEMA_VERSION",
    "TOWER_GROUP_SCHEMA_VERSION",
    "TowerGroup",
    "TowerGroupRegistry",
    "group_display_id_for_sequence",
]
