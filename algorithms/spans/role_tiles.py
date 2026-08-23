"""Validated source-shared role-specific spatial tile indexes.

The tower RAW tile cache is intentionally not used here.  Span science needs a
role binding (wire or vegetation), the role's configuration digest, and the
role attributes that survive into the span-local ``RawRegion``.  This module is
the small on-disk adapter for that contract.  It only reads an authoritative
``RawRegion`` while building and only opens tiles intersecting a query box.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import canonical_digest

from algorithms.vegetation.coarse_to_fine import RawRegion

ROLE_TILE_CACHE_SCHEMA = "treeline-span-role-tile-cache-v1"
ROLE_TILE_INDEX_SCHEMA = "treeline-span-role-tile-index-v1"
ROLE_TILE_RECORD_DTYPE = np.dtype(
    [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("source_index", "<i8"), ("classification", "<u1")],
    align=False,
)
ROLE_TILE_CELL_SIZE_M = 5.0
_ROLE_TILE_RECORD_DESCR = [list(item) for item in ROLE_TILE_RECORD_DTYPE.descr]
ROLE_TILE_BOUNDARY_SEMANTICS = "HALF_OPEN_XY_FINAL_EDGE_INCLUSIVE"


def role_tile_configuration_digest(cell_size_m: float) -> str:
    return canonical_digest(
        {
            "schema": "treeline-span-role-tile-grid-v1",
            "cell_size_m": float(cell_size_m),
            "boundary_semantics": ROLE_TILE_BOUNDARY_SEMANTICS,
            "partition_dimensions": "XY",
        }
    )


def _int64_digest(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype="<i8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _canonical_source_index_chunk(values: Any) -> np.ndarray:
    """Convert one source-index chunk without numeric truncation or wrapping."""

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise RoleTileCacheError("ROLE_TILE_SOURCE_INDEX_TYPE_INVALID")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise RoleTileCacheError("ROLE_TILE_SOURCE_INDEX_OVERFLOW")
    if raw.dtype.kind == "i" and raw.dtype.itemsize > np.dtype("<i8").itemsize:
        limits = np.iinfo(np.int64)
        if np.any(raw < limits.min) or np.any(raw > limits.max):
            raise RoleTileCacheError("ROLE_TILE_SOURCE_INDEX_OVERFLOW")
    return np.asarray(raw, dtype="<i8")


def _validate_unique_source_indices(
    region: RawRegion,
    *,
    source_point_count: int,
    chunk_size: int,
    exclusion: np.ndarray | None,
) -> tuple[int, int | None, int | None]:
    """Validate kept source-row identity before any cache artifact is created.

    The boolean identity map is one byte per source row, avoiding a boxed Python
    set while keeping duplicate detection bounded for the current ~105M-row
    source.  Excluded rows are not part of the role index and are therefore
    removed before uniqueness is checked.
    """

    if type(source_point_count) is not int or source_point_count < 0:
        raise ValueError("source_point_count must be a non-negative integer")
    seen = np.zeros(source_point_count, dtype=np.bool_)
    source_index = region.original_index()
    unique_count = 0
    minimum_index: int | None = None
    maximum_index: int | None = None
    for start in range(0, region.count, chunk_size):
        stop = min(region.count, start + chunk_size)
        indices = _canonical_source_index_chunk(source_index[start:stop])
        if np.any(indices < 0) or np.any(indices >= source_point_count):
            raise RoleTileCacheError("ROLE_TILE_SOURCE_INDEX_INVALID")
        if exclusion is not None:
            positions = np.searchsorted(exclusion, indices)
            present = positions < len(exclusion)
            present &= exclusion[np.minimum(positions, len(exclusion) - 1)] == indices
            indices = indices[~present]
        if not len(indices):
            continue
        chunk_min = int(np.min(indices))
        chunk_max = int(np.max(indices))
        minimum_index = chunk_min if minimum_index is None else min(minimum_index, chunk_min)
        maximum_index = chunk_max if maximum_index is None else max(maximum_index, chunk_max)
        ordered = np.sort(indices)
        if np.any(ordered[1:] == ordered[:-1]) or np.any(seen[indices]):
            raise RoleTileCacheError("ROLE_TILE_SOURCE_INDEX_DUPLICATE")
        seen[indices] = True
        unique_count += len(indices)
    return unique_count, minimum_index, maximum_index


class RoleTileCacheError(RuntimeError):
    """Raised when a role tile cache cannot be built or validated."""


def role_source_cache_identity(source_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return a portable, strong identity when the source contract provides one.

    Paths and generated project source IDs are intentionally excluded from the
    portable form.  Without a complete cryptographic identity the legacy full
    binding remains in force, so weak identities can never cross projects.
    """

    identity = dict(source_identity)
    sha256 = (
        identity.get("sha256")
        or identity.get("source_file_sha256")
        or identity.get("source_sha256")
    )
    size = identity.get("file_size_bytes")
    source_format = identity.get("format") or identity.get("source_format")
    if (
        isinstance(sha256, str)
        and len(sha256) == 64
        and all(character in "0123456789abcdefABCDEF" for character in sha256)
        and type(size) is int
        and size >= 0
        and isinstance(source_format, str)
        and source_format.strip()
    ):
        return {
            "schema": "treeline-strong-source-cache-identity-v1",
            "sha256": sha256.lower(),
            "file_size_bytes": size,
            "format": source_format.upper(),
        }
    return {
        "schema": "treeline-project-source-binding-fallback-v1",
        "source_identity": identity,
    }


def role_source_binding_digest(source_identity: Mapping[str, Any]) -> str:
    portable = role_source_cache_identity(source_identity)
    if portable.get("schema") == "treeline-strong-source-cache-identity-v1":
        return canonical_digest(portable)
    # Preserve the exact legacy digest so existing project-local caches remain
    # readable until they are explicitly rebound to a proven strong identity.
    return canonical_digest(dict(source_identity))


@dataclass(frozen=True, slots=True)
class RoleTileQueryResult:
    seed_xyz_absolute_m: tuple[float, float, float]
    xy_radius_m: float
    z_window_min_m: float
    z_window_max_m: float
    tiles: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    source_index: np.ndarray
    classification: np.ndarray
    point_count: int
    total_tiles: int
    intersecting_aabb_tiles: int
    opened_tiles: int
    points_read: int


class _RoleTileWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("wb")
        self.point_count = 0
        self.min_xyz = np.full(3, np.inf, dtype=np.float64)
        self.max_xyz = np.full(3, -np.inf, dtype=np.float64)

    def append(self, xyz: np.ndarray, source_index: np.ndarray, classification: np.ndarray) -> None:
        xyz = np.asarray(xyz, dtype=np.float64)
        source_index = np.asarray(source_index, dtype=np.int64)
        classification = np.asarray(classification, dtype=np.uint8)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise RoleTileCacheError("ROLE_TILE_XYZ_INVALID")
        if source_index.shape != (len(xyz),) or classification.shape != (len(xyz),):
            raise RoleTileCacheError("ROLE_TILE_ATTRIBUTE_ALIGNMENT_INVALID")
        if not len(xyz):
            return
        if not np.isfinite(xyz).all() or np.any(source_index < 0):
            raise RoleTileCacheError("ROLE_TILE_POINT_INVALID")
        block = np.empty(len(xyz), dtype=ROLE_TILE_RECORD_DTYPE)
        block["x"] = xyz[:, 0]
        block["y"] = xyz[:, 1]
        block["z"] = xyz[:, 2]
        block["source_index"] = source_index
        block["classification"] = classification
        block.tofile(self.handle)
        self.point_count += len(xyz)
        self.min_xyz = np.minimum(self.min_xyz, np.min(xyz, axis=0))
        self.max_xyz = np.maximum(self.max_xyz, np.max(xyz, axis=0))

    def close(self) -> None:
        self.handle.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tile_key(xyz: np.ndarray, origin_xy: tuple[float, float], cell_size_m: float) -> np.ndarray:
    cols = np.floor((xyz[:, 0] - origin_xy[0]) / cell_size_m).astype(np.int64)
    rows = np.floor((xyz[:, 1] - origin_xy[1]) / cell_size_m).astype(np.int64)
    return np.column_stack((cols, rows))


def _tile_id(col: int, row: int) -> str:
    return f"t{col}_{row}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_ROLE_TILE_VALIDATION_CACHE: dict[tuple[Any, ...], tuple[int, int, int | None, int | None]] = {}


def _validate_persisted_source_indices(
    root: Path,
    tiles: Mapping[str, Mapping[str, Any]],
    source_point_count: int,
    *,
    chunk_size: int = 65_536,
) -> tuple[int, int, int | None, int | None]:
    """Validate actual persisted identities with bounded NumPy storage."""

    seen = np.zeros(source_point_count, dtype=np.bool_)
    total_points = 0
    unique_points = 0
    minimum_index: int | None = None
    maximum_index: int | None = None
    for item in tiles.values():
        path = root / "tiles" / str(item["file"])
        point_count = int(item["point_count"])
        if point_count == 0:
            continue
        records = np.memmap(
            path,
            dtype=ROLE_TILE_RECORD_DTYPE,
            mode="r",
            shape=(point_count,),
        )
        records_seen = 0
        for start in range(0, point_count, chunk_size):
            stop = min(point_count, start + chunk_size)
            indices = _canonical_source_index_chunk(records[start:stop]["source_index"])
            if np.any(indices < 0) or np.any(indices >= source_point_count):
                raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_RANGE_INVALID")
            if len(indices):
                ordered = np.sort(indices)
                if np.any(ordered[1:] == ordered[:-1]) or np.any(seen[indices]):
                    raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_DUPLICATE")
                seen[indices] = True
                records_seen += len(indices)
                unique_points += len(indices)
                block_min = int(np.min(indices))
                block_max = int(np.max(indices))
                minimum_index = (
                    block_min if minimum_index is None else min(minimum_index, block_min)
                )
                maximum_index = (
                    block_max if maximum_index is None else max(maximum_index, block_max)
                )
        del records
        if records_seen != point_count:
            raise RoleTileCacheError("ROLE_TILE_CACHE_RECORD_COUNT_INVALID")
        total_points += records_seen
    return total_points, unique_points, minimum_index, maximum_index


def _role_tile_validation_key(
    root: Path,
    *,
    role: str,
    source_point_count: int,
    metadata: Mapping[str, Any],
    tiles: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    """Bind process-local validation reuse to the verified persisted content."""

    return (
        str(root),
        role,
        source_point_count,
        metadata.get("source_binding_digest"),
        metadata.get("role_configuration_digest"),
        metadata.get("tile_configuration_digest"),
        metadata.get("artifact_references", {}).get("tile_index", {}).get("sha256"),
        tuple(
            sorted(
                (
                    str(tile_id),
                    str(item.get("file")),
                    int(item.get("point_count", -1)),
                    item.get("sha256"),
                )
                for tile_id, item in tiles.items()
            )
        ),
    )


class RoleSpecificTileCache:
    """A source/config/role-bound tile index for one science role."""

    def __init__(
        self,
        root: Path,
        *,
        role: str,
        source_identity: Mapping[str, Any],
        config_digest: str,
        metadata: Mapping[str, Any],
        index: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if role not in {"wire", "vegetation"}:
            raise ValueError("role must be wire or vegetation")
        self.root = Path(root)
        self.role = role
        self.source_identity = dict(source_identity)
        self.source_binding_digest = role_source_binding_digest(source_identity)
        self.config_digest = config_digest
        self.metadata = dict(metadata)
        self.index = {str(key): dict(value) for key, value in index.items()}

    @classmethod
    def prepare_from_raw_region(
        cls,
        region: RawRegion,
        cache_root: str | Path,
        *,
        role: str,
        source_identity: Mapping[str, Any],
        config_digest: str,
        source_point_count: int,
        cell_size_m: float = ROLE_TILE_CELL_SIZE_M,
        chunk_size: int = 1_000_000,
        excluded_source_indices: np.ndarray | None = None,
        exclusion_role: str | None = None,
        exclusion_source_index_digest: str | None = None,
    ) -> RoleSpecificTileCache:
        """Build one role cache from a persisted, role-authoritative region."""

        if role not in {"wire", "vegetation"}:
            raise ValueError("role must be wire or vegetation")
        if type(source_point_count) is not int or source_point_count < 0:
            raise ValueError("source_point_count must be a non-negative integer")
        if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        # Vegetation admission is HIGH RECALL: no broad wire candidate/search
        # region may be excluded at the source-shared role-index level.  Only the
        # exact final confirmed conductor members are subtracted later, per Span.
        if role == "vegetation" and excluded_source_indices is not None:
            raise ValueError("vegetation role index must not carry a broad wire source exclusion")
        if excluded_source_indices is None:
            if exclusion_role is not None or exclusion_source_index_digest is not None:
                raise ValueError("exclusion metadata requires excluded source indices")
            exclusion = None
        else:
            exclusion = np.unique(_canonical_source_index_chunk(excluded_source_indices))
            if exclusion.ndim != 1 or np.any(exclusion < 0):
                raise ValueError("excluded_source_indices must be a non-negative 1-D array")
            if not exclusion_role or not exclusion_source_index_digest:
                raise ValueError("complete exclusion metadata is required")
            if len(exclusion_source_index_digest) != 64:
                raise ValueError("exclusion_source_index_digest must be SHA-256")
            if not len(exclusion):
                exclusion = None
        (
            validated_unique_count,
            validated_min_source_index,
            validated_max_source_index,
        ) = _validate_unique_source_indices(
            region,
            source_point_count=source_point_count,
            chunk_size=chunk_size,
            exclusion=exclusion,
        )
        root = Path(cache_root).resolve()
        if root.exists() and any(root.iterdir()):
            raise RoleTileCacheError("ROLE_TILE_CACHE_ROOT_NOT_EMPTY")
        root.mkdir(parents=True, exist_ok=True)
        tiles_dir = root / "tiles"
        started = time.perf_counter()
        origin = np.asarray(region.origin_xyz, dtype=np.float64)
        source_min = np.asarray(region.bounds_min_xyz, dtype=np.float64) + origin
        source_max = np.asarray(region.bounds_max_xyz, dtype=np.float64) + origin
        origin_xy = (
            math.floor(float(source_min[0]) / cell_size_m) * cell_size_m,
            math.floor(float(source_min[1]) / cell_size_m) * cell_size_m,
        )
        writers: dict[str, _RoleTileWriter] = {}
        source_index_hasher = hashlib.sha256()
        xyz_local = region.xyz()
        source_index = region.original_index()
        classification = region.classification()
        for start in range(0, region.count, chunk_size):
            stop = min(region.count, start + chunk_size)
            xyz = np.asarray(xyz_local[start:stop], dtype=np.float64) + origin
            indices = _canonical_source_index_chunk(source_index[start:stop])
            if np.any(indices < 0) or np.any(indices >= source_point_count):
                raise RoleTileCacheError("ROLE_TILE_SOURCE_INDEX_INVALID")
            classes = np.asarray(classification[start:stop], dtype=np.uint8)
            source_index_hasher.update(np.asarray(indices, dtype="<i8").tobytes(order="C"))
            if exclusion is not None:
                positions = np.searchsorted(exclusion, indices)
                present = positions < len(exclusion)
                present &= exclusion[np.minimum(positions, len(exclusion) - 1)] == indices
                if np.any(present):
                    keep = ~present
                    xyz = xyz[keep]
                    indices = indices[keep]
                    classes = classes[keep]
            if not len(xyz):
                continue
            cells = _tile_key(xyz, origin_xy, cell_size_m)
            unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
            for cell_no, cell in enumerate(unique_cells):
                mask = inverse == cell_no
                tile_id = _tile_id(int(cell[0]), int(cell[1]))
                writer = writers.get(tile_id)
                if writer is None:
                    writer = _RoleTileWriter(tiles_dir / f"{tile_id}.bin")
                    writers[tile_id] = writer
                writer.append(xyz[mask], indices[mask], classes[mask])
        index: dict[str, dict[str, Any]] = {}
        byte_size = 0
        for tile_id, writer in sorted(writers.items()):
            writer.close()
            path = tiles_dir / f"{tile_id}.bin"
            size = path.stat().st_size
            byte_size += size
            index[tile_id] = {
                "file": f"{tile_id}.bin",
                "point_count": writer.point_count,
                "min_xyz": [float(value) for value in writer.min_xyz],
                "max_xyz": [float(value) for value in writer.max_xyz],
                "sha256": _sha256_file(path),
            }
        index_payload = {"schema": ROLE_TILE_INDEX_SCHEMA, "tiles": index}
        _write_json(tiles_dir / "index.json", index_payload)
        indexed_point_count = sum(int(item["point_count"]) for item in index.values())
        metadata = {
            "schema": ROLE_TILE_CACHE_SCHEMA,
            "role": role,
            "source_identity": dict(source_identity),
            "source_binding_digest": role_source_binding_digest(source_identity),
            "source_point_count": source_point_count,
            "source_index_digest": source_index_hasher.hexdigest(),
            "role_configuration_digest": config_digest,
            "tile_configuration_digest": role_tile_configuration_digest(cell_size_m),
            "tile_grid": {
                "cell_size_m": float(cell_size_m),
                "boundary_semantics": ROLE_TILE_BOUNDARY_SEMANTICS,
                "partition_dimensions": "XY",
            },
            "record_dtype": ROLE_TILE_RECORD_DTYPE.descr,
            "cell_size_m": float(cell_size_m),
            "origin_xy": [float(value) for value in origin_xy],
            "source_bounds": {"min_xyz": source_min.tolist(), "max_xyz": source_max.tolist()},
            "totals": {
                "tile_count": len(index),
                "point_count": int(indexed_point_count),
                "byte_size": byte_size,
            },
            "role_record_count": int(indexed_point_count),
            "unique_source_index_count": int(validated_unique_count),
            "source_index_validation": "UNIQUE_SOURCE_INDEXES_V1",
            "min_source_index": validated_min_source_index,
            "max_source_index": validated_max_source_index,
            "artifact_references": {
                "tile_index": {
                    "relative_path": "tiles/index.json",
                    "sha256": _sha256_file(tiles_dir / "index.json"),
                    "schema": ROLE_TILE_INDEX_SCHEMA,
                }
            },
            "preparation_runtime_seconds": time.perf_counter() - started,
            # Semantic identity of the role admission rule.  VEGETATION is HIGH
            # RECALL (no broad wire exclusion); WIRE is the reconstruction role.
            "admission_semantics": "HIGH_RECALL" if role == "vegetation" else "WIRE_ROLE",
        }
        if exclusion is not None:
            metadata["source_index_exclusion"] = {
                "schema": "treeline-span-role-index-exclusion-v1",
                "rule": "EXCLUDE_EXACT_SOURCE_INDICES",
                "role": exclusion_role,
                "excluded_count": int(len(exclusion)),
                "excluded_index_digest": _int64_digest(exclusion),
                "reference_source_binding_digest": role_source_binding_digest(source_identity),
                "reference_source_index_digest": exclusion_source_index_digest,
            }
        _write_json(root / "role-tile-cache.json", metadata)
        return cls(
            root,
            role=role,
            source_identity=source_identity,
            config_digest=config_digest,
            metadata=metadata,
            index=index,
        )

    @classmethod
    def load(
        cls,
        cache_root: str | Path,
        *,
        role: str,
        source_identity: Mapping[str, Any],
        config_digest: str,
        source_point_count: int | None = None,
        expected_source_index_exclusion: Mapping[str, Any] | None = None,
        expected_cell_size_m: float | None = None,
    ) -> RoleSpecificTileCache:
        root = Path(cache_root).resolve()
        if source_point_count is not None and (
            type(source_point_count) is not int or source_point_count < 0
        ):
            raise ValueError("source_point_count must be a non-negative integer")
        metadata_path = root / "role-tile-cache.json"
        index_path = root / "tiles" / "index.json"
        if not metadata_path.is_file() or not index_path.is_file():
            raise RoleTileCacheError("ROLE_TILE_CACHE_METADATA_MISSING")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RoleTileCacheError("ROLE_TILE_CACHE_METADATA_INVALID") from exc
        if (
            metadata.get("schema") != ROLE_TILE_CACHE_SCHEMA
            or index_payload.get("schema") != ROLE_TILE_INDEX_SCHEMA
        ):
            raise RoleTileCacheError("ROLE_TILE_CACHE_SCHEMA_MISMATCH")
        if metadata.get("role") != role:
            raise RoleTileCacheError("ROLE_TILE_CACHE_ROLE_MISMATCH")
        if metadata.get("record_dtype") != _ROLE_TILE_RECORD_DESCR:
            raise RoleTileCacheError("ROLE_TILE_CACHE_RECORD_SCHEMA_MISMATCH")
        if metadata.get("role_configuration_digest") != config_digest:
            raise RoleTileCacheError("ROLE_TILE_CACHE_CONFIGURATION_MISMATCH")
        source_exclusion = metadata.get("source_index_exclusion")
        if source_exclusion is not None:
            if not isinstance(source_exclusion, dict):
                raise RoleTileCacheError("ROLE_TILE_CACHE_EXCLUSION_INVALID")
            if (
                source_exclusion.get("schema") != "treeline-span-role-index-exclusion-v1"
                or source_exclusion.get("rule") != "EXCLUDE_EXACT_SOURCE_INDICES"
                or source_exclusion.get("reference_source_binding_digest")
                != role_source_binding_digest(source_identity)
                or type(source_exclusion.get("excluded_count")) is not int
                or type(source_exclusion.get("excluded_index_digest")) is not str
                or len(source_exclusion.get("excluded_index_digest")) != 64
                or type(source_exclusion.get("reference_source_index_digest")) is not str
                or len(source_exclusion.get("reference_source_index_digest")) != 64
            ):
                raise RoleTileCacheError("ROLE_TILE_CACHE_EXCLUSION_INVALID")
        # High-recall vegetation admission: a vegetation role index built with
        # the OLD broad wire-role exclusion must FAIL CLOSED instead of silently
        # loading as the new semantics.  Only exact final confirmed conductor
        # members are subtracted later, per Span, from the tree input.
        if role == "vegetation":
            if source_exclusion is not None:
                raise RoleTileCacheError("ROLE_TILE_CACHE_VEGETATION_BROAD_WIRE_EXCLUSION")
            if metadata.get("admission_semantics") != "HIGH_RECALL":
                raise RoleTileCacheError("ROLE_TILE_CACHE_VEGETATION_ADMISSION_MISMATCH")
        if expected_source_index_exclusion is not None:
            if not isinstance(source_exclusion, dict):
                raise RoleTileCacheError("ROLE_TILE_CACHE_EXCLUSION_MISMATCH")
            if source_exclusion.get("role") != expected_source_index_exclusion.get(
                "role"
            ) or source_exclusion.get(
                "reference_source_index_digest"
            ) != expected_source_index_exclusion.get("source_index_digest"):
                raise RoleTileCacheError("ROLE_TILE_CACHE_EXCLUSION_MISMATCH")
        tile_grid = metadata.get("tile_grid")
        if (
            not isinstance(tile_grid, dict)
            or tile_grid.get("boundary_semantics") != ROLE_TILE_BOUNDARY_SEMANTICS
            or tile_grid.get("partition_dimensions") != "XY"
            or metadata.get("tile_configuration_digest")
            != role_tile_configuration_digest(float(tile_grid.get("cell_size_m", 0.0)))
        ):
            raise RoleTileCacheError("ROLE_TILE_CACHE_TILE_CONFIGURATION_MISMATCH")
        if expected_cell_size_m is not None:
            if not math.isfinite(expected_cell_size_m) or expected_cell_size_m <= 0.0:
                raise ValueError("expected_cell_size_m must be positive")
            if float(tile_grid["cell_size_m"]) != float(expected_cell_size_m):
                raise RoleTileCacheError("ROLE_TILE_CACHE_EXPECTED_TILE_SIZE_MISMATCH")
        expected_binding = role_source_binding_digest(source_identity)
        persisted_identity = metadata.get("source_identity")
        if (
            metadata.get("source_binding_digest") != expected_binding
            or not isinstance(persisted_identity, dict)
            or role_source_cache_identity(persisted_identity)
            != role_source_cache_identity(source_identity)
        ):
            raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_MISMATCH")
        if (
            source_point_count is not None
            and metadata.get("source_point_count") != source_point_count
        ):
            raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_COUNT_MISMATCH")
        refs = metadata.get("artifact_references")
        if not isinstance(refs, dict) or refs.get("tile_index", {}).get("sha256") != _sha256_file(
            index_path
        ):
            raise RoleTileCacheError("ROLE_TILE_CACHE_INDEX_INTEGRITY_INVALID")
        tiles = index_payload.get("tiles")
        if not isinstance(tiles, dict):
            raise RoleTileCacheError("ROLE_TILE_CACHE_INDEX_INVALID")
        tiles_root = (root / "tiles").resolve()
        total_points = 0
        for _tile_id, item in tiles.items():
            if not isinstance(item, dict) or not isinstance(item.get("file"), str):
                raise RoleTileCacheError("ROLE_TILE_CACHE_TILE_REFERENCE_INVALID")
            path = root / "tiles" / item["file"]
            try:
                path.resolve().relative_to(tiles_root)
            except ValueError as exc:
                raise RoleTileCacheError("ROLE_TILE_CACHE_TILE_REFERENCE_INVALID") from exc
            expected_size = int(item.get("point_count", -1)) * ROLE_TILE_RECORD_DTYPE.itemsize
            if not path.is_file() or path.stat().st_size != expected_size:
                raise RoleTileCacheError("ROLE_TILE_CACHE_TILE_SIZE_INVALID")
            if item.get("sha256") != _sha256_file(path):
                raise RoleTileCacheError("ROLE_TILE_CACHE_TILE_INTEGRITY_INVALID")
            bounds_min = item.get("min_xyz")
            bounds_max = item.get("max_xyz")
            if (
                not isinstance(bounds_min, list)
                or not isinstance(bounds_max, list)
                or len(bounds_min) != 3
                or len(bounds_max) != 3
            ):
                raise RoleTileCacheError("ROLE_TILE_CACHE_TILE_BOUNDS_INVALID")
            total_points += int(item["point_count"])
        totals = metadata.get("totals")
        if (
            not isinstance(totals, dict)
            or totals.get("tile_count") != len(tiles)
            or totals.get("point_count") != total_points
        ):
            raise RoleTileCacheError("ROLE_TILE_CACHE_TOTALS_INVALID")
        if source_point_count is not None:
            validation_key = _role_tile_validation_key(
                root,
                role=role,
                source_point_count=source_point_count,
                metadata=metadata,
                tiles=tiles,
            )
            actual_validation = _ROLE_TILE_VALIDATION_CACHE.get(validation_key)
            if actual_validation is None:
                actual_validation = _validate_persisted_source_indices(
                    root, tiles, source_point_count
                )
                _ROLE_TILE_VALIDATION_CACHE[validation_key] = actual_validation
            actual_count, actual_unique_count, actual_min, actual_max = actual_validation
            if actual_count != total_points or actual_unique_count != total_points:
                raise RoleTileCacheError("ROLE_TILE_CACHE_RECORD_COUNT_INVALID")
            persisted_min = metadata.get("min_source_index")
            persisted_max = metadata.get("max_source_index")
            if persisted_min is not None and type(persisted_min) is not int:
                raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_RANGE_INVALID")
            if persisted_max is not None and type(persisted_max) is not int:
                raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_RANGE_INVALID")
            if (persisted_min is None) != (persisted_max is None):
                raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_RANGE_INVALID")
            if persisted_min is not None and persisted_max is not None:
                if (
                    total_points == 0
                    or persisted_min < 0
                    or persisted_max < persisted_min
                    or persisted_max >= source_point_count
                    or persisted_min != actual_min
                    or persisted_max != actual_max
                ):
                    raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_RANGE_INVALID")
            else:
                if total_points == 0:
                    if actual_min is not None or actual_max is not None:
                        raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_RANGE_INVALID")
                elif actual_min is None or actual_max is None:
                    raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_RANGE_INVALID")
        role_record_count = metadata.get("role_record_count")
        unique_source_index_count = metadata.get("unique_source_index_count")
        if role_record_count is not None or unique_source_index_count is not None:
            if (
                type(role_record_count) is not int
                or type(unique_source_index_count) is not int
                or role_record_count != total_points
                or unique_source_index_count != role_record_count
                or metadata.get("source_index_validation") != "UNIQUE_SOURCE_INDEXES_V1"
            ):
                raise RoleTileCacheError("ROLE_TILE_CACHE_SOURCE_INDEX_VALIDATION_INVALID")
        return cls(
            root,
            role=role,
            source_identity=source_identity,
            config_digest=config_digest,
            metadata=metadata,
            index=tiles,
        )

    def query(
        self,
        seed_xyz_absolute_m: tuple[float, float, float],
        *,
        xy_radius_m: float,
        z_context_below_m: float,
        z_context_above_m: float,
    ) -> RoleTileQueryResult:
        seed = np.asarray(seed_xyz_absolute_m, dtype=np.float64)
        if (
            seed.shape != (3,)
            or xy_radius_m < 0.0
            or z_context_below_m < 0.0
            or z_context_above_m < 0.0
        ):
            raise ValueError("invalid role tile query")
        z_min = float(seed[2] - z_context_below_m)
        z_max = float(seed[2] + z_context_above_m)
        total_tiles, selected = self.plan_query(
            seed_xyz_absolute_m,
            xy_radius_m=xy_radius_m,
            z_context_below_m=z_context_below_m,
            z_context_above_m=z_context_above_m,
        )
        chunks: list[np.ndarray] = []
        points_read = 0
        for _tile_id, block in self.iter_tile_blocks(selected):
            points_read += len(block)
            chunks.append(block)
        if chunks:
            block = np.concatenate(chunks)
            x = block["x"].copy()
            y = block["y"].copy()
            z = block["z"].copy()
            source_index = block["source_index"].copy()
            classification = block["classification"].copy()
        else:
            x = y = z = np.empty(0, dtype=np.float64)
            source_index = np.empty(0, dtype=np.int64)
            classification = np.empty(0, dtype=np.uint8)
        return RoleTileQueryResult(
            tuple(float(value) for value in seed),
            float(xy_radius_m),
            z_min,
            z_max,
            tuple(selected),
            x,
            y,
            z,
            source_index,
            classification,
            len(x),
            total_tiles,
            len(selected),
            len(selected),
            points_read,
        )

    def plan_query(
        self,
        seed_xyz_absolute_m: tuple[float, float, float],
        *,
        xy_radius_m: float,
        z_context_below_m: float,
        z_context_above_m: float,
    ) -> tuple[int, tuple[str, ...]]:
        """Return metadata-only tile selection for a conservative AABB query."""

        seed = np.asarray(seed_xyz_absolute_m, dtype=np.float64)
        if (
            seed.shape != (3,)
            or xy_radius_m < 0.0
            or z_context_below_m < 0.0
            or z_context_above_m < 0.0
        ):
            raise ValueError("invalid role tile query")
        z_min = float(seed[2] - z_context_below_m)
        z_max = float(seed[2] + z_context_above_m)
        xy_min = seed[:2] - xy_radius_m
        xy_max = seed[:2] + xy_radius_m
        selected: list[str] = []
        for tile_id, item in sorted(self.index.items()):
            minimum = np.asarray(item["min_xyz"], dtype=np.float64)
            maximum = np.asarray(item["max_xyz"], dtype=np.float64)
            if not (
                maximum[0] < xy_min[0]
                or minimum[0] > xy_max[0]
                or maximum[1] < xy_min[1]
                or minimum[1] > xy_max[1]
                or maximum[2] < z_min
                or minimum[2] > z_max
            ):
                selected.append(tile_id)
        return len(self.index), tuple(selected)

    def plan_oriented_query(self, frame: Any, spec: Any) -> tuple[int, tuple[str, ...]]:
        """Return a conservative tile plan for the exact Span OBB/prism."""

        unit = np.asarray(frame.axis_xy, dtype=np.float64)
        perpendicular = np.asarray((-unit[1], unit[0]), dtype=np.float64)
        center = (
            np.asarray(frame.start_xyz_absolute_m[:2], dtype=np.float64)
            + unit
            * (
                -float(spec.longitudinal_before_m)
                + float(frame.length_m)
                + float(spec.longitudinal_after_m)
            )
            / 2.0
        )
        half_longitudinal = (
            float(spec.longitudinal_before_m)
            + float(frame.length_m)
            + float(spec.longitudinal_after_m)
        ) / 2.0
        half_width = float(spec.half_width_m)
        selected: list[str] = []
        for tile_id, item in sorted(self.index.items()):
            minimum = np.asarray(item["min_xyz"], dtype=np.float64)
            maximum = np.asarray(item["max_xyz"], dtype=np.float64)
            if maximum[2] < float(spec.z_min_m) or minimum[2] > float(spec.z_max_m):
                continue
            tile_center = (minimum[:2] + maximum[:2]) / 2.0
            tile_half = (maximum[:2] - minimum[:2]) / 2.0
            delta = np.abs(tile_center - center)
            axes = (
                (
                    unit,
                    half_longitudinal,
                    float(tile_half[0] * abs(unit[0]) + tile_half[1] * abs(unit[1])),
                ),
                (
                    perpendicular,
                    half_width,
                    float(
                        tile_half[0] * abs(perpendicular[0]) + tile_half[1] * abs(perpendicular[1])
                    ),
                ),
                (
                    np.asarray((1.0, 0.0)),
                    float(tile_half[0]),
                    float(half_longitudinal * abs(unit[0]) + half_width * abs(perpendicular[0])),
                ),
                (
                    np.asarray((0.0, 1.0)),
                    float(tile_half[1]),
                    float(half_longitudinal * abs(unit[1]) + half_width * abs(perpendicular[1])),
                ),
            )
            if all(
                abs(float(np.dot(delta, axis))) <= half_a + half_b for axis, half_a, half_b in axes
            ):
                selected.append(tile_id)
        return len(self.index), tuple(selected)

    def iter_tile_blocks(
        self,
        tile_ids: tuple[str, ...],
        *,
        chunk_size: int = 65_536,
    ) -> Any:
        """Stream selected tile records without concatenating the local workset."""

        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for tile_id in tile_ids:
            item = self.index[tile_id]
            path = self.root / "tiles" / item["file"]
            point_count = int(item["point_count"])
            for start in range(0, point_count, chunk_size):
                count = min(chunk_size, point_count - start)
                block = np.fromfile(
                    path,
                    dtype=ROLE_TILE_RECORD_DTYPE,
                    count=count,
                    offset=start * ROLE_TILE_RECORD_DTYPE.itemsize,
                )
                if len(block) != count:
                    raise RoleTileCacheError("ROLE_TILE_CACHE_TILE_READ_TRUNCATED")
                yield tile_id, block


def migrate_role_tile_configuration_digest(
    cache_root: str | Path,
    *,
    role: str,
    source_identity: Mapping[str, Any],
    previous_config_digest: str,
    configuration_digest: str,
    source_point_count: int | None = None,
    expected_source_index_exclusion: Mapping[str, Any] | None = None,
    expected_cell_size_m: float | None = None,
) -> RoleSpecificTileCache:
    """Rebind validated role-index metadata without rebuilding its tile content."""
    cache = RoleSpecificTileCache.load(
        cache_root,
        role=role,
        source_identity=source_identity,
        config_digest=previous_config_digest,
        source_point_count=source_point_count,
        expected_source_index_exclusion=expected_source_index_exclusion,
        expected_cell_size_m=expected_cell_size_m,
    )
    metadata = dict(cache.metadata)
    metadata["role_configuration_digest"] = configuration_digest
    _write_json(cache.root / "role-tile-cache.json", metadata)
    return RoleSpecificTileCache.load(
        cache.root,
        role=role,
        source_identity=source_identity,
        config_digest=configuration_digest,
        source_point_count=source_point_count,
        expected_source_index_exclusion=expected_source_index_exclusion,
        expected_cell_size_m=expected_cell_size_m,
    )


def migrate_role_tile_source_identity(
    cache_root: str | Path,
    *,
    role: str,
    previous_source_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    configuration_digest: str,
    source_point_count: int,
    expected_cell_size_m: float | None = None,
) -> RoleSpecificTileCache:
    """Rebind validated immutable tile content to a proven strong identity.

    This migration changes metadata only.  Loading the old binding first and
    the new binding afterwards validates every index/range/content contract;
    callers remain responsible for independently proving the supplied source
    SHA-256 before invoking the migration.
    """

    portable = role_source_cache_identity(source_identity)
    if portable.get("schema") != "treeline-strong-source-cache-identity-v1":
        raise RoleTileCacheError("ROLE_TILE_CACHE_STRONG_SOURCE_IDENTITY_REQUIRED")
    cache = RoleSpecificTileCache.load(
        cache_root,
        role=role,
        source_identity=previous_source_identity,
        config_digest=configuration_digest,
        source_point_count=source_point_count,
        expected_cell_size_m=expected_cell_size_m,
    )
    metadata = dict(cache.metadata)
    metadata["source_identity"] = dict(source_identity)
    metadata["source_binding_digest"] = role_source_binding_digest(source_identity)
    exclusion = metadata.get("source_index_exclusion")
    if isinstance(exclusion, dict):
        exclusion = dict(exclusion)
        exclusion["reference_source_binding_digest"] = role_source_binding_digest(
            source_identity
        )
        metadata["source_index_exclusion"] = exclusion
    _write_json(cache.root / "role-tile-cache.json", metadata)
    return RoleSpecificTileCache.load(
        cache.root,
        role=role,
        source_identity=source_identity,
        config_digest=configuration_digest,
        source_point_count=source_point_count,
        expected_cell_size_m=expected_cell_size_m,
    )


__all__ = [
    "ROLE_TILE_CACHE_SCHEMA",
    "ROLE_TILE_BOUNDARY_SEMANTICS",
    "ROLE_TILE_CELL_SIZE_M",
    "ROLE_TILE_INDEX_SCHEMA",
    "RoleSpecificTileCache",
    "RoleTileCacheError",
    "RoleTileQueryResult",
    "migrate_role_tile_configuration_digest",
    "migrate_role_tile_source_identity",
    "role_tile_configuration_digest",
    "role_source_cache_identity",
    "role_source_binding_digest",
]
