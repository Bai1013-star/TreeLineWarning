"""DBH-only provenance, editor eligibility and bounded exact raw-cache queries.

No crown/clearance membership is changed and no full-source index is built.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from algorithms.folder_pipeline.tree_editor import _decode_indices, load_edits_payload
from algorithms.spans.role_tiles import ROLE_TILE_RECORD_DTYPE

from core.config.fast_baseline import canonical_digest

SCHEMA = "treeline-dbh-evidence-v1"
MAX_LOCAL_POINTS = 2_000_000
MAX_LOCAL_TILES = 9
MAX_TILE_POINTS_READ = 12_000_000
RAW_CHUNK_POINTS = 250_000


def confined(root: Path, reference: str) -> Path:
    if Path(reference).is_absolute():
        raise ValueError("DBH_ABSOLUTE_ARTIFACT_REFERENCE")
    path = (root / reference).resolve()
    path.relative_to(root.resolve())
    return path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DBH_INVALID_JSON_OBJECT")
    return value


def strong_source(identity: dict[str, Any]) -> dict[str, Any]:
    digest = identity.get("sha256")
    count = identity.get("source_point_count")
    size = identity.get("file_size_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest.lower())
        or type(count) is not int
        or count <= 0
        or type(size) is not int
        or size <= 0
    ):
        raise ValueError("DBH_STRONG_SOURCE_IDENTITY_REQUIRED")
    return {"sha256": digest.lower(), "source_point_count": count, "file_size_bytes": size}


def array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def member_binding(reconciled: Any) -> dict[str, Any]:
    records = sorted(reconciled.global_tree_records, key=lambda r: int(r.global_tree_id))
    return {
        "original_index": array_digest(np.asarray(reconciled.final_original_index, dtype="<i8")),
        "xyz": array_digest(np.asarray(reconciled.final_xyz_local_m, dtype="<f8")),
        "tree_id": array_digest(np.asarray(reconciled.final_tree_id, dtype="<i4")),
        "identities": [
            {
                "persistent_tree_id": r.persistent_tree_id,
                "global_tree_id": int(r.global_tree_id),
                "lineage": sorted([list(k) for k in r.merged_span_tree_keys]),
                "point_count": int(r.full_resolution_point_count),
            }
            for r in records
        ],
    }


def editor_eligibility(
    original_records: list[dict[str, Any]],
    operations: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int], str], np.ndarray]:
    """Replay ancestry; legacy point removal invalidates recovered stem ownership."""
    operations = load_edits_payload(operations)
    alive = {
        int(r["global_tree_id"]): {tuple(int(v) for v in k) for k in r["merged_span_tree_keys"]}
        for r in original_records
    }
    initial = set().union(*alive.values()) if alive else set()
    invalid: set[tuple[int, int]] = set()
    removed = []
    for op in operations:
        ids = op["tree_ids"]
        if any(i not in alive for i in ids):
            raise ValueError("DBH_EDITOR_LINEAGE_INVALID")
        if op["type"] == "REMOVE_POINTS":
            invalid.update(set().union(*(alive[i] for i in ids)))
            removed.append(_decode_indices(op["removed_indices_b64"]))
        elif op["type"] == "DELETE":
            for i in ids:
                del alive[i]
        elif op["type"] == "MERGE":
            merged = op["merged_id"]
            if merged in alive:
                raise ValueError("DBH_EDITOR_MERGED_ID_COLLISION")
            lineage = set().union(*(alive.pop(i) for i in ids))
            alive[merged] = lineage
    surviving = set().union(*alive.values()) if alive else set()
    status = {
        key: (
            "DELETED"
            if key not in surviving
            else "EDITOR_STEM_SCOPE_UNPROVEN"
            if key in invalid
            else "ELIGIBLE"
        )
        for key in initial
    }
    indices = np.unique(np.concatenate(removed)) if removed else np.empty(0, dtype=np.int64)
    return status, indices


def local_raw_query(
    cache_root: Path,
    identity: dict[str, Any],
    center_xy: np.ndarray,
    z_bounds: tuple[float, float],
    *,
    radius_m: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate metadata and only intersecting tiles; never call full-cache load.

    Reads existing full-resolution high-recall source rows with genuine source_index.
    A portable strong identity binds this query even after source-path relocation.
    """
    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all() or not 0 < radius_m <= 0.60:
        raise ValueError("DBH_INVALID_LOCAL_ROI")
    if len(z_bounds) != 2 or not np.isfinite(z_bounds).all() or z_bounds[1] <= z_bounds[0]:
        raise ValueError("DBH_INVALID_LOCAL_Z_RANGE")
    root = cache_root.resolve()
    metadata = read_json(root / "role-tile-cache.json")
    source = strong_source(identity)
    if (
        metadata.get("schema") != "treeline-span-role-tile-cache-v1"
        or metadata.get("role") != "vegetation"
        or metadata.get("admission_semantics") != "HIGH_RECALL"
        or metadata.get("source_index_exclusion") is not None
        or strong_source(metadata.get("source_identity", {})) != source
        or metadata.get("source_point_count") != source["source_point_count"]
        or metadata.get("record_dtype") != [list(v) for v in ROLE_TILE_RECORD_DTYPE.descr]
    ):
        raise ValueError("DBH_LOCAL_SOURCE_PROOF_INVALID")
    reference = metadata["artifact_references"]["tile_index"]
    index_path = confined(root, reference["relative_path"])
    if sha256(index_path) != reference["sha256"]:
        raise ValueError("DBH_LOCAL_INDEX_HASH_MISMATCH")
    index = read_json(index_path)
    if index.get("schema") != "treeline-span-role-tile-index-v1":
        raise ValueError("DBH_LOCAL_INDEX_SCHEMA_INVALID")
    selected = []
    for key, item in sorted(index["tiles"].items()):
        lower, upper = np.asarray(item["min_xyz"]), np.asarray(item["max_xyz"])
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all():
            raise ValueError("DBH_LOCAL_TILE_BOUNDS_INVALID")
        if np.any(lower > upper):
            raise ValueError("DBH_LOCAL_TILE_BOUNDS_INVALID")
        if (
            np.all(upper[:2] >= center - radius_m)
            and np.all(lower[:2] <= center + radius_m)
            and upper[2] >= z_bounds[0]
            and lower[2] <= z_bounds[1]
        ):
            selected.append((key, item))
    count = sum(int(item["point_count"]) for _, item in selected)
    if len(selected) > MAX_LOCAL_TILES or count > MAX_TILE_POINTS_READ:
        raise ValueError("DBH_LOCAL_QUERY_BUDGET_EXCEEDED")
    indices, coordinates, proofs = [], [], []
    for key, item in selected:
        path = confined(root / "tiles", item["file"])
        if path.stat().st_size != int(item["point_count"]) * ROLE_TILE_RECORD_DTYPE.itemsize:
            raise ValueError("DBH_LOCAL_TILE_SIZE_MISMATCH")
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError("DBH_LOCAL_TILE_HASH_MISMATCH")
        with path.open("rb") as stream:
            while True:
                block = np.fromfile(stream, dtype=ROLE_TILE_RECORD_DTYPE, count=RAW_CHUNK_POINTS)
                if not len(block):
                    break
                xyz = np.column_stack((block["x"], block["y"], block["z"]))
                idx = block["source_index"]
                if (
                    not np.isfinite(xyz).all()
                    or np.any(idx < 0)
                    or np.any(idx >= source["source_point_count"])
                ):
                    raise ValueError("DBH_LOCAL_RAW_ROWS_INVALID")
                keep = np.linalg.norm(xyz[:, :2] - center, axis=1) <= radius_m
                keep &= (xyz[:, 2] >= z_bounds[0]) & (xyz[:, 2] <= z_bounds[1])
                coordinates.append(xyz[keep])
                indices.append(idx[keep])
                if sum(len(i) for i in indices) > MAX_LOCAL_POINTS:
                    raise ValueError("DBH_LOCAL_RESULT_BUDGET_EXCEEDED")
        proofs.append(
            {
                "tile": key,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "points_read": int(item["point_count"]),
            }
        )
    xyz = np.concatenate(coordinates) if coordinates else np.empty((0, 3), dtype=np.float64)
    idx = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    order = np.argsort(idx, kind="stable")
    if len(idx) != len(np.unique(idx)):
        raise ValueError("DBH_LOCAL_DUPLICATE_ORIGINAL_INDEX")
    return (
        xyz[order],
        idx[order],
        {
            "raw_local_query_used": True,
            "raw_local_point_count": len(idx),
            "raw_points_read": count,
            "opened_tiles": len(selected),
            "tiles": proofs,
            "radius_m": radius_m,
            "center_xy_absolute_m": center.tolist(),
            "z_bounds_absolute_m": list(z_bounds),
            "index_sha256": reference["sha256"],
            "source_identity": source,
            "source_identity_digest": canonical_digest(source),
        },
    )
