"""Apply audited registered-tower identities to an existing tree edit journal."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import algorithms.folder_pipeline.tree_editor as _tree_edits
import numpy as np
from algorithms.spans.tower_exclusion import (
    TOWER_TREE_EXCLUSION_VERSION,
    derive_registered_tower_exclusion,
    load_tower_membership,
    publish_tower_membership,
)
from algorithms.vegetation.coarse_to_fine import RawRegion

from core.stage_b.region_edit_session import RegionTreeEditSession

TOWER_EDITOR_SELECTION_METHOD = "REGISTERED_TOWER_CONNECTED_MEMBERS"


def _span_tower_indices(project_root: Path, span_root: Path) -> np.ndarray | None:
    index_path = span_root / "span-cache-index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    input_data = index.get("input")
    preparation = index.get("source_preparation")
    if not isinstance(input_data, dict) or not isinstance(preparation, dict):
        raise ValueError("TOWER_EXCLUSION_SPAN_CACHE_INVALID")
    source_identity = input_data.get("source_identity")
    ground = preparation.get("ground_preparation")
    if not isinstance(source_identity, dict) or not isinstance(ground, dict):
        raise ValueError("TOWER_EXCLUSION_SOURCE_CONTEXT_MISSING")
    origin = ground.get("origin_xyz")
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("TOWER_EXCLUSION_SOURCE_ORIGIN_INVALID")
    towers = []
    for side in ("from", "to"):
        geometry = input_data.get(f"{side}_endpoint_geometry")
        if not isinstance(geometry, dict):
            raise ValueError("TOWER_EXCLUSION_ENDPOINT_MISSING")
        towers.append(
            {
                "tower_id": input_data[f"{side}_tower_id"],
                "base_xyz_absolute_m": geometry["base_xyz_absolute_m"],
                "top_xyz_absolute_m": geometry["top_xyz_absolute_m"],
            }
        )
    existing = load_tower_membership(
        span_root,
        source_identity=source_identity,
        registered_towers=towers,
        region_origin_xyz=origin,
    )
    if existing is not None:
        return existing.source_index
    post_wire = span_root / "regions" / "vegetation_after_confirmed_wire"
    region_root = post_wire if (post_wire / "original_index.bin").is_file() else (
        span_root / "regions" / "vegetation"
    )
    xyz_path = region_root / "xyz_local.bin"
    source_path = region_root / "original_index.bin"
    classification_path = region_root / "classification.bin"
    if not all(path.is_file() for path in (xyz_path, source_path, classification_path)):
        raise ValueError("TOWER_EXCLUSION_RAW_REGION_MISSING")
    count, remainder = divmod(source_path.stat().st_size, 8)
    if (
        remainder
        or xyz_path.stat().st_size != count * 12
        or classification_path.stat().st_size != count
    ):
        raise ValueError("TOWER_EXCLUSION_RAW_REGION_ALIGNMENT_INVALID")
    region = RawRegion(
        xyz_local_m_path=str(xyz_path),
        original_index_path=str(source_path),
        classification_path=str(classification_path),
        count=count,
        bounds_min_xyz=(0.0, 0.0, 0.0),
        bounds_max_xyz=(0.0, 0.0, 0.0),
        origin_xyz=tuple(float(value) for value in origin),
    )
    exclusion = derive_registered_tower_exclusion(region, towers)
    publish_tower_membership(
        exclusion,
        span_root,
        source_identity=source_identity,
        region_origin_xyz=region.origin_xyz,
    )
    return exclusion.source_index


def ensure_editor_tower_exclusion(
    project_root: Path,
    artifact_root: Path,
    span_ids: tuple[str, ...],
    session: RegionTreeEditSession,
) -> dict[str, Any] | None:
    """Append one standard REMOVE_POINTS for still-present tower members.

    Existing human operations stay in their original order.  A repeated call
    sees no remaining tower identities and appends nothing; undoing the tower
    operation causes the next finalize to reapply it.
    """
    parts: list[np.ndarray] = []
    span_manifests: list[str] = []
    for span_id in span_ids:
        span_root = artifact_root / str(span_id)
        source_index = _span_tower_indices(project_root, span_root)
        if source_index is None:
            continue
        parts.append(source_index)
        span_manifests.append(str((span_root / "tower_membership.json").relative_to(project_root)))
    if not parts:
        return None
    tower_source = np.unique(np.concatenate(parts))
    final_index, final_tree, _final_xyz, _summary = session.build_final()
    selected = _tree_edits._isin_sorted(final_index, tower_source)
    removed = np.unique(final_index[selected])
    affected_tree_ids = [int(value) for value in np.unique(final_tree[selected])]
    if len(removed):
        operation = {
            "type": "REMOVE_POINTS",
            "tree_ids": affected_tree_ids,
            "selection_method": TOWER_EDITOR_SELECTION_METHOD,
            "point_count": int(len(removed)),
            "removed_indices_b64": _tree_edits._encode_indices(removed),
            "sequence": session.revision + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        session.operations.append(operation)
        session.selected_ids = []
        session._previews.clear()
    audit = {
        "schema": "treeline-stage-b-final-tower-exclusion-v1",
        "method": TOWER_TREE_EXCLUSION_VERSION,
        "tower_source_index_count": int(len(tower_source)),
        "removed_from_current_membership": int(len(removed)),
        "affected_tree_ids": affected_tree_ids,
        "span_membership_manifests": span_manifests,
        "editor_selection_method": TOWER_EDITOR_SELECTION_METHOD,
    }
    target = artifact_root / "tower_exclusion_final.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return audit


__all__ = ["TOWER_EDITOR_SELECTION_METHOD", "ensure_editor_tower_exclusion"]
