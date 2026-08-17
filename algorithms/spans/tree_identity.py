"""Persistent entity identities for reconciled tree observations.

Segmentation ids are intentionally run-local.  This module adds a small,
auditable registry that assigns a non-recycled entity id only after matching a
new observation to the last observation of the same source.  It stores the
full membership *index* in an NPZ beside the run artifact, never in JSON and
never as Python point objects.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.spans.reconcile import TreeReconciliationResult

TREE_IDENTITY_SCHEMA = "treeline-persistent-tree-registry-v1"


def _empty_registry() -> dict[str, Any]:
    return {"schema": TREE_IDENTITY_SCHEMA, "next_sequence_by_group": {}, "entities": {}}


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_registry()
    data = json.loads(path.read_text("utf-8"))
    if data.get("schema") != TREE_IDENTITY_SCHEMA:
        raise ValueError("TREE_IDENTITY_REGISTRY_SCHEMA_INVALID")
    if not isinstance(data.get("entities"), dict):
        raise ValueError("TREE_IDENTITY_REGISTRY_INVALID")
    return data


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8")
    temporary.replace(path)


def write_identity_membership(path: Path, result: TreeReconciliationResult) -> None:
    """Persist sorted source-row membership for future same-source matching."""

    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(result.final_original_index, kind="stable")
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary,
        original_index=np.asarray(result.final_original_index[order], dtype=np.int64),
        global_tree_id=np.asarray(result.final_tree_id[order], dtype=np.int32),
    )
    temporary.replace(path)


def _members_by_tree(original_index: np.ndarray, tree_id: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(identifier): np.asarray(original_index[tree_id == identifier], dtype=np.int64)
        for identifier in np.unique(tree_id)
        if int(identifier) >= 0
    }


def _iou(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    overlap = int(np.intersect1d(left, right, assume_unique=True).size)
    if not overlap:
        return 0.0, 0.0
    # Coverage is of the historical member set.  This permits us to detect a
    # temporary merge (each historic crown is fully contained in one new
    # observation) while its one-to-many graph still prevents auto-inheritance.
    return overlap / (len(left) + len(right) - overlap), overlap / len(left)


def _next_id(registry: dict[str, Any], group_id: str) -> str:
    sequences = registry["next_sequence_by_group"]
    sequence = int(sequences.get(group_id, 1))
    sequences[group_id] = sequence + 1
    return f"{group_id}-T{sequence:03d}"


def assign_persistent_tree_ids(
    result: TreeReconciliationResult,
    config: FastBaselineConfig,
    *,
    registry_path: Path,
    membership_path: Path,
    group_id: str,
    source_key: str,
    run_id: str,
) -> TreeReconciliationResult:
    """Assign durable ids, preserving explicit split/merge ambiguity events.

    A same-source match is accepted only for a strong, one-to-one membership
    overlap.  One historic entity mapping to several new trees is a split: the
    old id is superseded and each child receives a new id.  Several historic
    entities mapping to one new observation is left REVIEW rather than merging
    permanent entities.
    """

    registry = _read_registry(registry_path)
    current_members = _members_by_tree(result.final_original_index, result.final_tree_id)
    previous: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for entity_id, entity in registry["entities"].items():
        if entity.get("group_id") != group_id or entity.get("source_key") != source_key:
            continue
        if entity.get("current_status") == "SUPERSEDED_BY_SPLIT":
            # A known merged historical observation must never silently regain
            # biological identity merely because a later algorithm re-merges it.
            continue
        ref = entity.get("last_membership_reference")
        prior_gid = entity.get("last_global_tree_id")
        if not isinstance(ref, str) or not isinstance(prior_gid, int):
            continue
        artifact = registry_path.parent / Path(ref)
        if not artifact.is_file():
            continue
        with np.load(artifact, allow_pickle=False) as data:
            prior_original = np.asarray(data["original_index"], dtype=np.int64)
            prior_tree = np.asarray(data["global_tree_id"], dtype=np.int32)
        members = prior_original[prior_tree == prior_gid]
        if len(members):
            previous[entity_id] = (entity, members)

    # Candidate graph is deliberately tree-record/membership-index scale.
    links_by_old: dict[str, list[int]] = {}
    links_by_new: dict[int, list[str]] = {}
    for entity_id, (_entity, old_members) in previous.items():
        for gid, new_members in current_members.items():
            iou, coverage = _iou(old_members, new_members)
            if (
                iou >= config.identity_same_source_min_iou
                and coverage >= config.identity_same_source_min_coverage_fraction
            ):
                links_by_old.setdefault(entity_id, []).append(gid)
                links_by_new.setdefault(gid, []).append(entity_id)

    inherited: dict[int, str] = {}
    split_old = {entity_id for entity_id, children in links_by_old.items() if len(children) > 1}
    merge_new = {gid for gid, parents in links_by_new.items() if len(parents) > 1}
    for entity_id, children in links_by_old.items():
        if len(children) == 1 and entity_id not in split_old:
            gid = children[0]
            if gid not in merge_new:
                inherited[gid] = entity_id
    for entity_id in split_old:
        registry["entities"][entity_id]["current_status"] = "SUPERSEDED_BY_SPLIT"
        registry["entities"][entity_id]["last_seen_run"] = run_id

    membership_reference = membership_path.relative_to(registry_path.parent).as_posix()
    records = []
    for record in result.global_tree_records:
        gid = int(record.global_tree_id)
        entity_id = inherited.get(gid)
        state = "SAME_ENTITY" if entity_id else "NEW_ENTITY"
        if gid in merge_new:
            state = "MERGE_AMBIGUOUS_REVIEW"
        if entity_id is None:
            entity_id = _next_id(registry, group_id)
        entity = registry["entities"].setdefault(entity_id, {
            "persistent_tree_id": entity_id,
            "group_id": group_id,
            "source_key": source_key,
            "first_seen_run": run_id,
            "lineage": [],
        })
        if entity_id not in inherited.values() and split_old:
            parents = [old for old, children in links_by_old.items() if gid in children]
            if parents:
                entity["lineage"] = [
                    {"event": "SPLIT_FROM", "persistent_tree_id": parent}
                    for parent in parents
                ]
        entity.update({
            "last_seen_run": run_id,
            "current_status": state,
            "current_span_id": record.span_name,
            "current_global_tree_id": gid,
            "last_global_tree_id": gid,
            "last_membership_reference": membership_reference,
            "identity_confidence": (
                "HIGH" if state == "SAME_ENTITY"
                else "REVIEW" if "AMBIGUOUS" in state else "NEW"
            ),
        })
        records.append(
            replace(
                record,
                persistent_tree_id=entity_id,
                external_name=entity_id,
                identity_state=state,
            )
        )

    write_identity_membership(membership_path, result)
    _write_json_atomic(registry_path, registry)
    return replace(result, global_tree_records=tuple(records))


__all__ = ["TREE_IDENTITY_SCHEMA", "assign_persistent_tree_ids", "write_identity_membership"]
