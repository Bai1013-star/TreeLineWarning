"""Transactional persistent identities for reconciled tree observations."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.spans.reconcile import TreeReconciliationResult

TREE_IDENTITY_SCHEMA = "treeline-persistent-tree-registry-v1"
TREE_IDENTITY_MEMBERSHIP_SCHEMA = "treeline-persistent-tree-membership-v2"
_LOCK_POLL_SECONDS = 0.05
_Published = TypeVar("_Published")


class TreeIdentityError(RuntimeError):
    """Raised when durable tree identity cannot be safely finalized."""


@dataclass(frozen=True, slots=True)
class TreeIdentityPreparation:
    """Unpublished identity work that is safe to carry through downstream work."""

    result: TreeReconciliationResult
    config: FastBaselineConfig
    registry_path: Path
    membership_path: Path
    group_id: str
    source_key: str
    run_id: str


def _empty_registry() -> dict[str, Any]:
    return {"schema": TREE_IDENTITY_SCHEMA, "next_sequence_by_group": {}, "entities": {}}


def _project_root(registry_path: Path) -> Path:
    return registry_path.parent.resolve()


def _require_relative_to(path: Path, root: Path, code: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TreeIdentityError(code) from exc
    return path


def _validate_registry(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema") != TREE_IDENTITY_SCHEMA:
        raise TreeIdentityError("TREE_IDENTITY_REGISTRY_SCHEMA_INVALID")
    sequences = data.get("next_sequence_by_group")
    entities = data.get("entities")
    if not isinstance(sequences, dict) or not isinstance(entities, dict):
        raise TreeIdentityError("TREE_IDENTITY_REGISTRY_INVALID")
    for group_id, sequence in sequences.items():
        if not isinstance(group_id, str) or type(sequence) is not int or sequence < 1:
            raise TreeIdentityError("TREE_IDENTITY_REGISTRY_SEQUENCE_INVALID")
    persistent_ids: set[str] = set()
    for entity_id, entity in entities.items():
        if not isinstance(entity_id, str) or not isinstance(entity, dict):
            raise TreeIdentityError("TREE_IDENTITY_REGISTRY_ENTITY_INVALID")
        if entity.get("persistent_tree_id") != entity_id:
            raise TreeIdentityError("TREE_IDENTITY_REGISTRY_ENTITY_ID_INVALID")
        if entity_id in persistent_ids:
            raise TreeIdentityError("TREE_IDENTITY_REGISTRY_DUPLICATE_ID")
        persistent_ids.add(entity_id)
        if not isinstance(entity.get("group_id"), str) or not isinstance(
            entity.get("source_key"), str
        ):
            raise TreeIdentityError("TREE_IDENTITY_REGISTRY_ENTITY_INVALID")
        lineage = entity.get("lineage", [])
        if not isinstance(lineage, list):
            raise TreeIdentityError("TREE_IDENTITY_REGISTRY_LINEAGE_INVALID")
        for item in lineage:
            if not isinstance(item, dict):
                raise TreeIdentityError("TREE_IDENTITY_REGISTRY_LINEAGE_INVALID")
            parent = item.get("persistent_tree_id")
            if (
                item.get("event") != "SPLIT_FROM"
                or not isinstance(parent, str)
                or parent not in entities
            ):
                raise TreeIdentityError("TREE_IDENTITY_REGISTRY_LINEAGE_INVALID")
    return data


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_registry()
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TreeIdentityError("TREE_IDENTITY_REGISTRY_UNREADABLE") from exc
    return _validate_registry(data)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
        )
        temporary.replace(path)
    except OSError as exc:
        raise TreeIdentityError("TREE_IDENTITY_REGISTRY_WRITE_FAILED") from exc


@contextmanager
def _project_registry_lock(registry_path: Path, timeout_seconds: float) -> Iterator[None]:
    """Hold a bounded project-scoped inter-process lock without stale deletion."""

    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise TreeIdentityError("TREE_IDENTITY_LOCK_TIMEOUT_INVALID")
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(timeout_seconds)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        acquired = False
        while not acquired:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TreeIdentityError("TREE_IDENTITY_LOCK_TIMEOUT") from exc
                time.sleep(_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_preparation_paths(registry_path: Path, membership_path: Path) -> None:
    root = _project_root(registry_path)
    _require_relative_to(
        membership_path.resolve(), root, "TREE_IDENTITY_MEMBERSHIP_OUTSIDE_PROJECT"
    )


def write_identity_membership(
    path: Path,
    result: TreeReconciliationResult,
    *,
    source_key: str,
    group_id: str,
) -> None:
    """Persist sorted source-row membership with a verifiable source contract."""

    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(result.final_original_index, kind="stable")
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez(
            temporary,
            schema=np.asarray(TREE_IDENTITY_MEMBERSHIP_SCHEMA),
            source_key=np.asarray(source_key),
            group_id=np.asarray(group_id),
            original_index=np.asarray(result.final_original_index[order], dtype=np.int64),
            global_tree_id=np.asarray(result.final_tree_id[order], dtype=np.int32),
        )
        temporary.replace(path)
    except OSError as exc:
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_WRITE_FAILED") from exc


def _scalar_text(data: Any, name: str) -> str:
    value = np.asarray(data[name])
    if value.size != 1:
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_INVALID")
    return str(value.reshape(-1)[0])


def _load_membership(entity: dict[str, Any], registry_path: Path) -> np.ndarray:
    root = _project_root(registry_path)
    reference = entity.get("last_membership_reference")
    global_tree_id = entity.get("last_global_tree_id")
    if not isinstance(reference, str) or not isinstance(global_tree_id, int):
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_REFERENCE_INVALID")
    ref_path = Path(reference)
    if ref_path.is_absolute():
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_OUTSIDE_PROJECT")
    artifact = _require_relative_to(
        (root / ref_path).resolve(), root, "TREE_IDENTITY_MEMBERSHIP_OUTSIDE_PROJECT"
    )
    if not artifact.is_file():
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_MISSING")
    try:
        with np.load(artifact, allow_pickle=False) as data:
            required = {"schema", "source_key", "group_id", "original_index", "global_tree_id"}
            if not required.issubset(data.files):
                raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_SCHEMA_INVALID")
            if _scalar_text(data, "schema") != TREE_IDENTITY_MEMBERSHIP_SCHEMA:
                raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_SCHEMA_INVALID")
            if _scalar_text(data, "source_key") != entity["source_key"]:
                raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_SOURCE_MISMATCH")
            if _scalar_text(data, "group_id") != entity["group_id"]:
                raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_GROUP_MISMATCH")
            original = np.asarray(data["original_index"])
            tree = np.asarray(data["global_tree_id"])
    except (OSError, ValueError) as exc:
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_UNREADABLE") from exc
    if (
        original.ndim != 1
        or tree.ndim != 1
        or len(original) != len(tree)
        or not np.issubdtype(original.dtype, np.integer)
        or not np.issubdtype(tree.dtype, np.integer)
        or np.any(original < 0)
        or len(original) != len(np.unique(original))
    ):
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_INVALID")
    members = np.asarray(original[tree == global_tree_id], dtype=np.int64)
    if not len(members):
        raise TreeIdentityError("TREE_IDENTITY_MEMBERSHIP_ENTITY_MISSING")
    return np.sort(members)


def _members_by_tree(original_index: np.ndarray, tree_id: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(identifier): np.sort(
            np.unique(np.asarray(original_index[tree_id == identifier], dtype=np.int64))
        )
        for identifier in np.unique(tree_id)
        if int(identifier) >= 0
    }


def _overlap(left: np.ndarray, right: np.ndarray) -> tuple[int, float, float, float]:
    overlap = int(np.intersect1d(left, right, assume_unique=True).size)
    if not overlap:
        return 0, 0.0, 0.0, 0.0
    return (
        overlap,
        overlap / (len(left) + len(right) - overlap),
        overlap / len(left),
        overlap / len(right),
    )


def _next_id(registry: dict[str, Any], group_id: str) -> str:
    sequences = registry["next_sequence_by_group"]
    sequence = int(sequences.get(group_id, 1))
    while f"{group_id}-T{sequence:03d}" in registry["entities"]:
        sequence += 1
    sequences[group_id] = sequence + 1
    return f"{group_id}-T{sequence:03d}"


def _split_children(
    old_members: np.ndarray,
    current_members: dict[int, np.ndarray],
    config: FastBaselineConfig,
) -> list[int]:
    children: list[int] = []
    covered: list[np.ndarray] = []
    for global_tree_id, members in sorted(current_members.items()):
        overlap, _iou, parent_fraction, child_purity = _overlap(old_members, members)
        if (
            len(members) >= config.identity_split_min_child_members
            and overlap >= config.identity_split_min_child_members
            and parent_fraction >= config.identity_split_min_child_parent_fraction
            and child_purity >= config.identity_split_min_child_purity
        ):
            children.append(global_tree_id)
            covered.append(np.intersect1d(old_members, members, assume_unique=True))
    if len(children) < 2:
        return []
    union = np.unique(np.concatenate(covered))
    if len(union) / len(old_members) < config.identity_split_min_union_parent_coverage:
        return []
    return children


def _assign_from_registry(
    result: TreeReconciliationResult,
    config: FastBaselineConfig,
    registry: dict[str, Any],
    *,
    registry_path: Path,
    membership_path: Path,
    group_id: str,
    source_key: str,
    run_id: str,
) -> TreeReconciliationResult:
    current_members = _members_by_tree(result.final_original_index, result.final_tree_id)
    previous: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for entity_id, entity in sorted(registry["entities"].items()):
        if entity.get("group_id") != group_id or entity.get("source_key") != source_key:
            continue
        members = _load_membership(entity, registry_path)
        if entity.get("current_status") != "SUPERSEDED_BY_SPLIT":
            previous[entity_id] = (entity, members)

    links_by_old: dict[str, list[int]] = {}
    links_by_new: dict[int, list[str]] = {}
    for entity_id, (_entity, old_members) in previous.items():
        for gid, new_members in current_members.items():
            _count, iou, old_coverage, _purity = _overlap(old_members, new_members)
            if (
                iou >= config.identity_same_source_min_iou
                and old_coverage >= config.identity_same_source_min_coverage_fraction
            ):
                links_by_old.setdefault(entity_id, []).append(gid)
                links_by_new.setdefault(gid, []).append(entity_id)

    split_by_old = {
        entity_id: _split_children(old_members, current_members, config)
        for entity_id, (_entity, old_members) in previous.items()
    }
    split_by_old = {entity_id: children for entity_id, children in split_by_old.items() if children}
    split_children = {child for children in split_by_old.values() for child in children}
    merge_by_new = {
        gid: sorted(parents) for gid, parents in links_by_new.items() if len(parents) > 1
    }
    inherited: dict[int, str] = {}
    for entity_id, children in links_by_old.items():
        if len(children) == 1 and entity_id not in split_by_old:
            gid = children[0]
            if gid not in merge_by_new and gid not in split_children:
                inherited[gid] = entity_id
    for entity_id in split_by_old:
        registry["entities"][entity_id]["current_status"] = "SUPERSEDED_BY_SPLIT"
        registry["entities"][entity_id]["last_seen_run"] = run_id

    root = _project_root(registry_path)
    membership_reference = membership_path.resolve().relative_to(root).as_posix()
    split_parents_by_child = {
        child: sorted(parent for parent, children in split_by_old.items() if child in children)
        for child in split_children
    }
    records = []
    for record in result.global_tree_records:
        gid = int(record.global_tree_id)
        merge_parents = merge_by_new.get(gid)
        if merge_parents:
            records.append(
                replace(
                    record,
                    persistent_tree_id=None,
                    identity_state="MERGE_AMBIGUOUS_REVIEW",
                    candidate_persistent_tree_ids=tuple(merge_parents),
                )
            )
            continue
        entity_id = inherited.get(gid)
        state = "SAME_ENTITY" if entity_id else "NEW_ENTITY"
        lineage: list[dict[str, str]] = []
        parents = split_parents_by_child.get(gid, [])
        if parents:
            state = "SPLIT_CHILD_NEW_ENTITY"
            lineage = [{"event": "SPLIT_FROM", "persistent_tree_id": parent} for parent in parents]
        if entity_id is None:
            entity_id = _next_id(registry, group_id)
        entity = registry["entities"].setdefault(
            entity_id,
            {
                "persistent_tree_id": entity_id,
                "group_id": group_id,
                "source_key": source_key,
                "first_seen_run": run_id,
                "lineage": lineage,
            },
        )
        entity.update(
            {
                "last_seen_run": run_id,
                "current_status": state,
                "current_span_id": record.span_name,
                "current_global_tree_id": gid,
                "last_global_tree_id": gid,
                "last_membership_reference": membership_reference,
                "identity_confidence": "HIGH" if state == "SAME_ENTITY" else "NEW",
            }
        )
        records.append(
            replace(
                record,
                persistent_tree_id=entity_id,
                external_name=entity_id,
                identity_state=state,
                candidate_persistent_tree_ids=(),
            )
        )
    return replace(result, global_tree_records=tuple(records))


def prepare_tree_identity(
    result: TreeReconciliationResult,
    config: FastBaselineConfig,
    *,
    registry_path: Path,
    membership_path: Path,
    group_id: str,
    source_key: str,
    run_id: str,
) -> TreeIdentityPreparation:
    """Create an unpublished identity transaction; this never mutates registry state."""

    _validate_preparation_paths(registry_path, membership_path)
    return TreeIdentityPreparation(
        result=result,
        config=config,
        registry_path=registry_path,
        membership_path=membership_path,
        group_id=group_id,
        source_key=source_key,
        run_id=run_id,
    )


def finalize_tree_identity(
    prepared: TreeIdentityPreparation,
    *,
    publish: Callable[[TreeReconciliationResult], _Published] | None = None,
) -> tuple[TreeReconciliationResult, _Published | None]:
    """Finalize after downstream publish succeeds, under one project lock."""

    with _project_registry_lock(
        prepared.registry_path, prepared.config.identity_registry_lock_timeout_seconds
    ):
        registry = _read_registry(prepared.registry_path)
        assigned = _assign_from_registry(
            prepared.result,
            prepared.config,
            registry,
            registry_path=prepared.registry_path,
            membership_path=prepared.membership_path,
            group_id=prepared.group_id,
            source_key=prepared.source_key,
            run_id=prepared.run_id,
        )
        published = publish(assigned) if publish is not None else None
        write_identity_membership(
            prepared.membership_path,
            assigned,
            source_key=prepared.source_key,
            group_id=prepared.group_id,
        )
        _write_json_atomic(prepared.registry_path, registry)
    return assigned, published


def assign_persistent_tree_ids(
    result: TreeReconciliationResult,
    config: FastBaselineConfig,
    **kwargs: Any,
) -> TreeReconciliationResult:
    """Compatibility helper: prepare and immediately finalize without publish work."""

    assigned, _published = finalize_tree_identity(prepare_tree_identity(result, config, **kwargs))
    return assigned


__all__ = [
    "TREE_IDENTITY_MEMBERSHIP_SCHEMA",
    "TREE_IDENTITY_SCHEMA",
    "TreeIdentityError",
    "TreeIdentityPreparation",
    "assign_persistent_tree_ids",
    "finalize_tree_identity",
    "prepare_tree_identity",
    "write_identity_membership",
]
