"""DBH-only provenance, editor eligibility and bounded exact raw-cache queries.

No crown/clearance membership is changed.  The existing ``v-hr`` index is
accepted here only when its metadata proves that it is a one-to-one index of
*every* point in the imported LAS/LAZ.  Its historical ``vegetation`` role
name must never be mistaken for a ground-removed or tree-only data source.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
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
RAW_GROUND_BIN_M = 0.03
RAW_GROUND_SEARCH_HALF_WINDOW_M = 0.60
RAW_GROUND_MIN_POINTS = 6


def raw_tree_lowest_reference(
    tree_xyz_absolute: np.ndarray,
    *,
    processed_ground_prior_z: float | None = None,
) -> dict[str, Any]:
    """Anchor DBH to the lowest exact raw member of this tree.

    Membership may be decided by the reviewed segmentation, but Z values are
    the original LAS coordinates carried by those members.  This deliberately
    avoids using a ground-removed/display cloud's new lower boundary.
    """

    xyz = np.asarray(tree_xyz_absolute, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("DBH_RAW_TREE_POINTS_REQUIRED")
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if not len(xyz):
        raise ValueError("DBH_RAW_TREE_POINTS_REQUIRED")
    lowest = float(np.min(xyz[:, 2]))
    support = int(np.count_nonzero(xyz[:, 2] <= lowest + RAW_GROUND_BIN_M))
    prior = lowest if processed_ground_prior_z is None else float(processed_ground_prior_z)
    return {
        "ground_z_absolute_m": lowest,
        "tree_lowest_z_absolute_m": lowest,
        "tree_highest_z_absolute_m": float(np.max(xyz[:, 2])),
        "breast_height_z_absolute_m": lowest + 1.30,
        "breast_height_hag_m": 1.30,
        "ground_reference_method": "ORIGINAL_RAW_TREE_LOWEST_POINT",
        "raw_ground_support_point_count": support,
        "processed_ground_prior_z_absolute_m": prior,
        "raw_minus_processed_ground_m": lowest - prior,
        "raw_ground_plane_rmse_m": None,
        "raw_ground_plane_xy_origin_m": np.median(xyz[:, :2], axis=0).tolist(),
        "raw_ground_plane_coefficients": [0.0, 0.0, lowest],
        "site_slope_deg": 0.0,
        "common_height_applied": True,
        "measurement_height_policy": "RAW_TREE_LOWEST_PLUS_1P30M",
    }


def raw_ground_reference(
    xyz_absolute: np.ndarray,
    expected_ground_z: float,
    center_xy: np.ndarray | None = None,
) -> dict[str, Any]:
    """Resolve local ground from original raw returns near a processed prior.

    The processed ground surface is used only to bound the search.  The actual
    breast-height origin is a spatially spread planar layer in the original
    local point cloud. A dense trunk band is not accepted as a ground surface.
    """

    xyz = np.asarray(xyz_absolute, dtype=np.float64)
    expected = float(expected_ground_z)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(expected):
        raise ValueError("DBH_RAW_GROUND_INPUT_INVALID")
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    candidates_xyz = xyz[np.abs(xyz[:, 2] - expected) <= RAW_GROUND_SEARCH_HALF_WINDOW_M]
    candidates = candidates_xyz[:, 2]
    if len(candidates) < RAW_GROUND_MIN_POINTS:
        raise ValueError("DBH_ORIGINAL_GROUND_EVIDENCE_MISSING")
    else:
        bins = np.floor((candidates - expected) / RAW_GROUND_BIN_M).astype(np.int64)
        labels, counts = np.unique(bins, return_counts=True)
        surfaces = []
        target_xy = np.asarray(center_xy) if center_xy is not None else np.median(xyz[:, :2], axis=0)
        for label in labels[counts >= RAW_GROUND_MIN_POINTS]:
            selected = candidates_xyz[np.abs(bins - label) <= 1]
            # A dense lower-stem elevation band is not a ground surface.
            # Require spatially spread, locally planar original returns.
            if np.max(np.ptp(selected[:, :2], axis=0)) < 0.30:
                continue
            xy_origin = np.median(selected[:, :2], axis=0)
            design = np.column_stack((selected[:, :2] - xy_origin, np.ones(len(selected))))
            plane = np.linalg.lstsq(design, selected[:, 2], rcond=None)[0]
            residual = selected[:, 2] - design @ plane
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            if rmse > 0.015 or np.linalg.norm(plane[:2]) > 0.70:
                continue
            elevation = float(np.dot(target_xy - xy_origin, plane[:2]) + plane[2])
            surfaces.append((abs(elevation - expected), -len(selected), elevation, len(selected), rmse,
                             xy_origin.tolist(), plane.tolist()))
        if not surfaces:
            raise ValueError("DBH_ORIGINAL_GROUND_SURFACE_UNPROVEN")
        _, _, ground, support, plane_rmse, plane_xy_origin, plane_coefficients = min(surfaces)
        method = "ORIGINAL_RAW_LOCAL_GROUND_MODE"
        if abs(ground - expected) > RAW_GROUND_SEARCH_HALF_WINDOW_M:
            raise ValueError("DBH_ORIGINAL_GROUND_SURFACE_OUT_OF_RANGE")
    return {
        "ground_z_absolute_m": float(ground),
        "breast_height_z_absolute_m": float(ground + 1.30),
        "breast_height_hag_m": 1.30,
        "ground_reference_method": method,
        "raw_ground_support_point_count": int(support),
        "processed_ground_prior_z_absolute_m": expected,
        "raw_minus_processed_ground_m": float(ground - expected),
        "raw_ground_plane_rmse_m": plane_rmse,
        "raw_ground_plane_xy_origin_m": plane_xy_origin,
        "raw_ground_plane_coefficients": plane_coefficients,
        "site_slope_deg": float(np.degrees(np.arctan(np.linalg.norm(plane_coefficients[:2])))),
    }


def ground_reference_at_stem(reference, stem_xy_absolute):
    """Evaluate the original ground plane at this trunk, not at a crown seed."""
    ground = dict(reference)
    xy = np.asarray(stem_xy_absolute, dtype=float)
    plane = np.asarray(reference["raw_ground_plane_coefficients"], dtype=float)
    origin = np.asarray(reference["raw_ground_plane_xy_origin_m"], dtype=float)
    z = float(np.dot(xy - origin, plane[:2]) + plane[2])
    ground.update(ground_z_absolute_m=z, breast_height_z_absolute_m=z + 1.30,
                  stem_xy_absolute_m=xy.tolist(),
                  raw_minus_processed_ground_m=z - reference["processed_ground_prior_z_absolute_m"])
    return ground


def calibrated_breast_height(reference, stem_xy_absolute, calibrations=()):
    """Share a measured raw-ground level only on demonstrably level terrain.

    A slope under 5 degrees alone does not prove equal elevations across a
    long span (a 4 degree slope rises metres). Validate this trunk's own ground
    too, and never borrow the diameter of another tree. Otherwise retain the
    local ground + 1.30 m policy.
    """
    if reference.get("ground_reference_method") == "ORIGINAL_RAW_TREE_LOWEST_POINT":
        local = dict(reference)
        local["stem_xy_absolute_m"] = np.asarray(stem_xy_absolute, dtype=float).tolist()
        local["common_height_applied"] = True
        local["measurement_height_policy"] = "RAW_TREE_LOWEST_PLUS_1P30M"
        return local
    local = ground_reference_at_stem(reference, stem_xy_absolute)
    slope = float(np.degrees(np.arctan(np.linalg.norm(
        np.asarray(reference['raw_ground_plane_coefficients'])[:2]))))
    local.update(site_slope_deg=slope, common_height_applied=False,
                 measurement_height_policy='LOCAL_RAW_GROUND_PLUS_1P30M')
    if slope >= 5.0:
        return local
    def eligible_reference(row):
        try:
            diameter, ground, target = [float(row[k]) for k in
                ('dbh_cm', 'dbh_ground_z_absolute_m', 'dbh_measurement_z_absolute_m')]
        except (KeyError, TypeError, ValueError):
            return False
        return (np.isfinite([diameter,ground,target]).all() and 5 <= diameter <= 150
                and row.get('dbh_height_reference') == 'ORIGINAL_RAW_GROUND_PLUS_1P30M'
                and abs(target-ground-1.30) < 1e-6
                and abs(ground-local['ground_z_absolute_m']) <= 1e-6)
    eligible = [row for row in calibrations if eligible_reference(row)]
    if not eligible:
        return local
    calibration = min(eligible, key=lambda r: abs(float(r['dbh_ground_z_absolute_m'])-local['ground_z_absolute_m']))
    # A reference from another tree cannot shift this tree's physical ground.
    local.update(common_height_applied=True, calibration_tree_id=calibration.get('global_tree_id'),
                 measurement_height_policy='CALIBRATED_LEVEL_GROUND_PLUS_1P30M',
                 ground_reference_method='ORIGINAL_RAW_VERIFIED_LEVEL_GROUND_CALIBRATION')
    return local


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


@lru_cache(maxsize=8)
def _verified_original_file(path: str, size: int, mtime_ns: int, expected_sha: str):
    # Stat is part of the cache key, so a changed source is checked anew.
    if sha256(Path(path)) != expected_sha:
        raise ValueError("DBH_ORIGINAL_LAS_HASH_MISMATCH")
    return True


def original_raw_query(project_root, cache_root, identity, center_xy, z_bounds):
    """Use the spatial index only for locating source rows; reread actual LAS.

    Ground and diameter use unmodified LAS integer coordinates and scales,
    never float32 processing coordinates or filtered crown membership.
    """
    import laspy

    root = Path(project_root)
    _, indices, proof = local_raw_query(cache_root, identity, center_xy, z_bounds)
    project = read_json(root / "stage-a-project.json")["project"]
    expected = strong_source(identity)
    candidates = [Path(scan["point_cloud_source"]) for scan in project["scans"]
                  if scan.get("source_identity", {}).get("sha256") == expected["sha256"]
                  and scan.get("source_identity", {}).get("file_size_bytes") == expected["file_size_bytes"]]
    paths = [path for path in candidates if path.is_file()]
    if not paths:
        raise ValueError("DBH_ORIGINAL_LAS_MISSING_RELINK_SOURCE")
    path = paths[0].resolve()
    stat = path.stat()
    if stat.st_size != expected["file_size_bytes"]:
        raise ValueError("DBH_ORIGINAL_LAS_SIZE_MISMATCH")
    _verified_original_file(str(path), stat.st_size, stat.st_mtime_ns, expected["sha256"])
    with laspy.open(path) as reader:
        header = reader.header
        if header.point_count != expected["source_point_count"]:
            raise ValueError("DBH_ORIGINAL_LAS_POINT_COUNT_MISMATCH")
        xyz = np.empty((len(indices), 3), dtype=np.float64)
        if not header.are_points_compressed:
            records = np.memmap(path, mode="r", dtype=header.point_format.dtype(),
                                offset=header.offset_to_point_data, shape=(header.point_count,))
            for dimension, name in enumerate(("X", "Y", "Z")):
                xyz[:, dimension] = (records[name][indices].astype(np.float64)
                                     * header.scales[dimension] + header.offsets[dimension])
            del records
        else:
            start = 0
            while start < len(indices):
                stop = start + 1
                while (stop < len(indices) and indices[stop] - indices[start] < 65536
                       and indices[stop] - indices[stop - 1] <= 256):
                    stop += 1
                first = int(indices[start])
                reader.seek(first)
                records = reader.read_points(int(indices[stop - 1]) - first + 1)
                local = indices[start:stop] - first
                xyz[start:stop] = np.column_stack((records.x[local], records.y[local], records.z[local]))
                start = stop
    proof.update(coordinate_source="DIRECT_ORIGINAL_LAS_INTEGER_COORDINATES",
                 original_las_path=str(path), original_las_sha256=expected["sha256"])
    return xyz, indices, proof


def array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def member_binding(reconciled: Any) -> dict[str, Any]:
    records = sorted(reconciled.global_tree_records, key=lambda r: int(r.global_tree_id))
    # Replay/finalize can concatenate restored trees in a different row order.
    # Bind each coordinate and owner to its original LAS index, not to the
    # incidental array ordering, while still detecting any changed membership.
    order = np.argsort(np.asarray(reconciled.final_original_index), kind="stable")
    return {
        "ordering": "ORIGINAL_INDEX_ASCENDING_V1",
        "original_index": array_digest(np.asarray(reconciled.final_original_index, dtype="<i8")[order]),
        "xyz": array_digest(np.asarray(reconciled.final_xyz_local_m, dtype="<f8")[order]),
        "tree_id": array_digest(np.asarray(reconciled.final_tree_id, dtype="<i4")[order]),
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
        elif op['type'] == 'SPLIT':
            if len(ids) != 1:
                raise ValueError('DBH_EDITOR_SPLIT_LINEAGE_INVALID')
            children = op['child_ids']
            if len(children) != 2 or any(child in alive for child in children):
                raise ValueError('DBH_EDITOR_SPLIT_ID_COLLISION')
            lineage = alive.pop(ids[0])
            for child in children:
                alive[child] = set(lineage)
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
    source_count = source["source_point_count"]
    # Portable role caches publish the count as a validated top-level field,
    # independently of their SourceDescriptor. Older descriptors omit it.
    # Require the same count, SHA and size; never overwrite a conflicting count.
    metadata_source = dict(metadata.get("source_identity", {}))
    if metadata_source.get("source_point_count") is None:
        metadata_source["source_point_count"] = metadata.get("source_point_count")
    totals = metadata.get("totals") or {}
    has_full_coverage_metadata = any(
        key in metadata
        for key in (
            "role_record_count",
            "unique_source_index_count",
            "min_source_index",
            "max_source_index",
        )
    )
    if (
        metadata.get("schema") != "treeline-span-role-tile-cache-v1"
        or metadata.get("role") != "vegetation"
        or metadata.get("admission_semantics") != "HIGH_RECALL"
        or metadata.get("source_index_exclusion") is not None
        or strong_source(metadata_source) != source
        or metadata.get("source_point_count") != source_count
        or (
            has_full_coverage_metadata
            and (
                metadata.get("role_record_count") != source_count
                or metadata.get("unique_source_index_count") != source_count
                or metadata.get("min_source_index") != 0
                or metadata.get("max_source_index") != source_count - 1
                or totals.get("point_count") != source_count
                or metadata.get("source_index_validation") != "UNIQUE_SOURCE_INDEXES_V1"
            )
        )
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
            "source_coverage": (
                "ALL_IMPORTED_ORIGINAL_POINTS_ONE_TO_ONE"
                if has_full_coverage_metadata
                else "LEGACY_HIGH_RECALL_SOURCE_CACHE"
            ),
            "source_coverage_point_count": source_count,
        },
    )
