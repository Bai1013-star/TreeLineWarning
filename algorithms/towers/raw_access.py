"""Persistent RAW spatial tile cache over the protected full-resolution LAS.

The refinement stage must never rescan the complete 105M-point source for a
single Owner ACCEPT.  This module builds a *derived, source-bound* tile cache
with one bounded streaming scan of the authoritative LAS:

- a fixed XY grid (``cell_size_m``) over the source bounds;
- every source point is written to the tile containing its XY, preserving the
  **exact** RAW source XYZ (the float64 values laspy exposes from the sealed
  scaled integers) and the **original source row/index** identity;
- optional RGB storage is supported but never required for refinement;
- ``cache.json`` seals the source binding, grid configuration, digest, totals,
  and preparation time so reloads fail closed on a wrong or mutated source;
- ``tiles/index.json`` records per-tile bounds/counts so a query can skip tiles
  whose Z range cannot intersect the requested window.

The cache is disposable/derived: it never modifies the protected LAS, it is
never committed, and it is never stored inside the protected authoritative run.

Queries are bounded in XY (a radius around a seed) and in Z (an explicit window
around the seed) so a single candidate refinement reads only the intersecting
local tiles and only the local Z context — never the whole source Z range.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import canonical_digest
from core.contracts.project import CoordinateContext, SourceIdentityBinding

RAW_TILE_CACHE_SCHEMA_VERSION = "treeline-tower-raw-tile-cache-v1"
RAW_TILE_STORAGE_VERSION = 1
_TILE_FILE_MAGIC = b"TLWTILE01"
_TILE_HEADER_SIZE = 88
# record layout without RGB: x,y,z (3×float64) + source_index (int64) = 32 bytes.
_STRIDE_BASE = 32
# with RGB: + red,green,blue (3×uint16) = 6 bytes -> 38.
_STRIDE_RGB = 38
_MAX_TILE_BUFFER_POINTS = 2_000_000
_INDEX_SCHEMA_VERSION = "treeline-tower-raw-tile-index-v1"


class RawTileCacheError(RuntimeError):
    """Closed failure raised by the RAW tile cache loader/preparer."""


class RawTileCacheSourceMismatchError(RawTileCacheError):
    """Raised when a cache is bound to a different source identity than requested."""


class RawTileCacheIntegrityError(RawTileCacheError):
    """Raised when a cache directory or tile file is malformed."""


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RawTileCacheConfig:
    """Fixed tile-cache construction parameters (sealed by a configuration digest)."""

    cell_size_m: float = 40.0
    chunk_size: int = 5_000_000
    include_rgb: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.cell_size_m, bool) or not isinstance(self.cell_size_m, float):
            raise ValueError("cell_size_m must be a finite positive float")
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be a finite positive float")
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if type(self.include_rgb) is not bool:
            raise ValueError("include_rgb must be a boolean")

    @property
    def stride_bytes(self) -> int:
        return _STRIDE_RGB if self.include_rgb else _STRIDE_BASE

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_size_m": self.cell_size_m,
            "chunk_size": self.chunk_size,
            "include_rgb": self.include_rgb,
            "storage_version": RAW_TILE_STORAGE_VERSION,
        }

    def configuration_digest(self) -> str:
        return canonical_digest(self.to_dict())


def _tile_id(col: int, row: int) -> str:
    return f"t{col:06d}_{row:06d}"


def _grid_origin(source_min: tuple[float, float, float], cell_size: float) -> tuple[float, float]:
    x0 = math.floor(source_min[0] / cell_size) * cell_size
    y0 = math.floor(source_min[1] / cell_size) * cell_size
    return float(x0), float(y0)


def _col_row(
    xyz: np.ndarray, x0: float, y0: float, cell_size: float
) -> np.ndarray:
    cols = np.floor((xyz[:, 0] - x0) / cell_size).astype(np.int64)
    rows = np.floor((xyz[:, 1] - y0) / cell_size).astype(np.int64)
    return np.column_stack((cols, rows))


def _write_tile_header(
    handle: Any, *, point_count: int, stride: int, flags: int, bounds: tuple[float, ...]
) -> None:
    header = bytearray(_TILE_HEADER_SIZE)
    header[0 : len(_TILE_FILE_MAGIC)] = _TILE_FILE_MAGIC
    header[16:20] = RAW_TILE_STORAGE_VERSION.to_bytes(4, "little")
    header[20:24] = flags.to_bytes(4, "little")
    header[24:28] = (0).to_bytes(4, "little")
    header[28:36] = point_count.to_bytes(8, "little")
    header[36:40] = stride.to_bytes(4, "little")
    for index, value in enumerate(bounds):
        header[40 + index * 8 : 48 + index * 8] = np.float64(value).tobytes()
    handle.write(bytes(header))


class _TileWriter:
    """Appendable per-tile binary writer with an incremental in-memory buffer.

    Point arrays are buffered as whole numpy slices (one slice per flush-cap
    group) so the bulk work stays vectorized; only the small per-chunk bucketing
    touches Python.
    """

    def __init__(self, path: Path, *, stride: int, flags: int) -> None:
        self.path = path
        self.stride = stride
        self.flags = flags
        self.parts: list[np.ndarray] = []
        self._buffered_points = 0
        self.point_count = 0
        self.min_xyz = np.full(3, math.inf, dtype=np.float64)
        self.max_xyz = np.full(3, -math.inf, dtype=np.float64)
        self._initialized = False

    def add_batch(self, xyz: np.ndarray, idx: np.ndarray, rgb: np.ndarray | None) -> None:
        """Append one vectorized slice (xyz float64 (n,3), idx int64 (n,), rgb (n,3) or None)."""
        if len(xyz) == 0:
            return
        self.parts.append(xyz.astype(np.float64, copy=False))
        self.parts.append(idx.astype(np.int64, copy=False))
        if self.flags:
            assert rgb is not None
            self.parts.append(rgb.astype(np.uint16, copy=False))
        self._buffered_points += len(xyz)
        self.min_xyz[0] = min(self.min_xyz[0], float(np.min(xyz[:, 0])))
        self.min_xyz[1] = min(self.min_xyz[1], float(np.min(xyz[:, 1])))
        self.min_xyz[2] = min(self.min_xyz[2], float(np.min(xyz[:, 2])))
        self.max_xyz[0] = max(self.max_xyz[0], float(np.max(xyz[:, 0])))
        self.max_xyz[1] = max(self.max_xyz[1], float(np.max(xyz[:, 1])))
        self.max_xyz[2] = max(self.max_xyz[2], float(np.max(xyz[:, 2])))
        if self._buffered_points >= _MAX_TILE_BUFFER_POINTS:
            self.flush()

    def _open_append(self) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path.open("ab")

    def flush(self) -> None:
        if not self.parts:
            return
        with self._open_append() as handle:
            if not self._initialized:
                placeholder = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                _write_tile_header(
                    handle, point_count=0, stride=self.stride, flags=self.flags, bounds=placeholder
                )
                self._initialized = True
            # self.parts interleaves [xyz, idx, (rgb)] per buffered batch.
            step = 3 if self.flags else 2
            xyz = np.concatenate(self.parts[0::step], axis=0)
            idx = np.concatenate(self.parts[1::step], axis=0)
            if self.flags:
                rgb = np.concatenate(self.parts[2::step], axis=0)
                block = np.empty(
                    len(xyz),
                    dtype=[("a", "<f8"), ("b", "<f8"), ("c", "<f8"),
                           ("d", "<i8"), ("e", "<u2"), ("f", "<u2"), ("g", "<u2")],
                )
                block["a"] = xyz[:, 0]
                block["b"] = xyz[:, 1]
                block["c"] = xyz[:, 2]
                block["d"] = idx
                block["e"] = rgb[:, 0]
                block["f"] = rgb[:, 1]
                block["g"] = rgb[:, 2]
                handle.write(block.tobytes())
            else:
                block = np.empty(
                    len(xyz),
                    dtype=[("a", "<f8"), ("b", "<f8"), ("c", "<f8"), ("d", "<i8")],
                )
                block["a"] = xyz[:, 0]
                block["b"] = xyz[:, 1]
                block["c"] = xyz[:, 2]
                block["d"] = idx
                handle.write(block.tobytes())
        self.point_count += self._buffered_points
        self.parts = []
        self._buffered_points = 0

    def finalize(self) -> None:
        self.flush()
        if self.point_count == 0:
            # An empty tile is never materialized; remove any stray placeholder.
            if self.path.exists():
                self.path.unlink()
            return
        with self.path.open("r+b") as handle:
            bounds = tuple(float(value) for value in np.concatenate((self.min_xyz, self.max_xyz)))
            _write_tile_header(
                handle,
                point_count=self.point_count,
                stride=self.stride,
                flags=self.flags,
                bounds=bounds,
            )


@dataclass(frozen=True, slots=True)
class RawTileQueryResult:
    """Bounded local RAW point context around one seed, with original identity."""

    seed_xyz_absolute_m: tuple[float, float, float]
    xy_radius_m: float
    z_window_min_m: float
    z_window_max_m: float
    tiles: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    source_index: np.ndarray
    point_count: int

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack((self.x, self.y, self.z))

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.point_count == 0:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return (
            (float(np.min(self.x)), float(np.min(self.y)), float(np.min(self.z))),
            (float(np.max(self.x)), float(np.max(self.y)), float(np.max(self.z))),
        )


class RawTileCache:
    """A loaded, source-validated RAW spatial tile cache."""

    def __init__(
        self,
        root: Path,
        *,
        source_binding: SourceIdentityBinding,
        config: RawTileCacheConfig,
        metadata: dict[str, Any],
        index: dict[str, dict[str, Any]],
    ) -> None:
        self.root = Path(root)
        self.source_binding = source_binding
        self.config = config
        self.metadata = metadata
        self.index = index

    # ------------------------------------------------------------------ prep
    @classmethod
    def prepare(
        cls,
        source_las_path: str | Path,
        cache_root: str | Path,
        *,
        source_binding: SourceIdentityBinding,
        config: RawTileCacheConfig | None = None,
    ) -> RawTileCache:
        """Build a derived RAW tile cache from the authoritative LAS with one scan.

        ``source_las_path`` is only ever read.  The returned cache validates its
        own source binding on reload, so a later query against a different source
        fails closed instead of silently using stale tiles.
        """
        import laspy

        if type(source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        if config is None:
            config = RawTileCacheConfig()
        if type(config) is not RawTileCacheConfig:
            raise ValueError("config must be a RawTileCacheConfig")
        source_path = Path(source_las_path).resolve()
        root = Path(cache_root).resolve()
        if not source_path.is_file():
            raise RawTileCacheError(f"RAW source LAS is missing: {source_path}")
        if root.exists() and any(root.iterdir()):
            raise RawTileCacheError(
                f"tile cache root is not empty; refusing to overwrite a cache: {root}"
            )
        root.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            las = laspy.open(source_path)
            header = las.header
        except Exception as exc:  # header parse failure -> not a compatible source
            raise RawTileCacheError(f"cannot read RAW source LAS header: {exc}") from exc
        source_min = tuple(float(value) for value in header.mins)
        source_max = tuple(float(value) for value in header.maxs)
        x0, y0 = _grid_origin(source_min, config.cell_size_m)
        has_rgb = (
            config.include_rgb
            and "red" in header.point_format.dimension_names
            and "green" in header.point_format.dimension_names
            and "blue" in header.point_format.dimension_names
        )
        if config.include_rgb and not has_rgb:
            raise RawTileCacheError("include_rgb requested but the source LAS has no RGB dims")

        writers: dict[str, _TileWriter] = {}
        tiles_dir = root / "tiles"
        total_points = 0
        row_offset = 0
        for chunk in las.chunk_iterator(config.chunk_size):
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64)
            count = len(xyz)
            rows = np.arange(row_offset, row_offset + count, dtype=np.int64)
            row_offset += count
            total_points += count
            cells = _col_row(xyz, x0, y0, config.cell_size_m)
            unique_keys, inverse = np.unique(cells, axis=0, return_inverse=True)
            if config.include_rgb:
                r = np.asarray(chunk.red, dtype=np.uint16)
                g = np.asarray(chunk.green, dtype=np.uint16)
                b = np.asarray(chunk.blue, dtype=np.uint16)
            for key_index in range(len(unique_keys)):
                mask = inverse == key_index
                tid = _tile_id(int(unique_keys[key_index, 0]), int(unique_keys[key_index, 1]))
                writer = writers.get(tid)
                if writer is None:
                    writer = _TileWriter(
                        tiles_dir / f"{tid}.bin",
                        stride=config.stride_bytes,
                        flags=1 if config.include_rgb else 0,
                    )
                    writers[tid] = writer
                writer.add_batch(
                    xyz[mask],
                    rows[mask],
                    np.column_stack((r[mask], g[mask], b[mask])) if config.include_rgb else None,
                )
        elapsed = time.perf_counter() - started
        if total_points != int(header.point_count):
            raise RawTileCacheError(
                "scanned point count does not match the LAS header; source may have "
                f"changed during preparation ({total_points} != {header.point_count})"
            )

        index: dict[str, dict[str, Any]] = {}
        byte_size = 0
        for tid, writer in sorted(writers.items()):
            writer.finalize()
            path = tiles_dir / f"{tid}.bin"
            if not path.exists():
                continue
            size = path.stat().st_size
            byte_size += size
            index[tid] = {
                "file": f"{tid}.bin",
                "point_count": writer.point_count,
                "min_xyz": list(writer.min_xyz),
                "max_xyz": list(writer.max_xyz),
                "sha256": _sha256_bytes(path.read_bytes()),
            }
        _write_json(
            tiles_dir / "index.json",
            {"schema_version": _INDEX_SCHEMA_VERSION, "tiles": index},
        )
        metadata = {
            "schema_version": RAW_TILE_CACHE_SCHEMA_VERSION,
            "cache_id": f"rawtiles-{secrets.token_hex(8)}",
            "created_at": _utc_timestamp(),
            "source": source_binding.to_dict(),
            "config": config.to_dict(),
            "configuration_digest": config.configuration_digest(),
            "source_bounds": {"min_xyz": list(source_min), "max_xyz": list(source_max)},
            "grid": {"x0": x0, "y0": y0},
            "totals": {
                "source_point_count": int(header.point_count),
                "scanned_point_count": total_points,
                "tile_count": len(index),
                "byte_size": byte_size,
            },
            "preparation_runtime_seconds": elapsed,
        }
        _write_json(root / "cache.json", metadata)
        return cls(
            root,
            source_binding=source_binding,
            config=config,
            metadata=metadata,
            index=index,
        )

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        cache_root: str | Path,
        *,
        source_binding: SourceIdentityBinding,
        config: RawTileCacheConfig | None = None,
    ) -> RawTileCache:
        """Load and strictly validate an existing cache against the expected source."""
        root = Path(cache_root).resolve()
        cache_path = root / "cache.json"
        index_path = root / "tiles" / "index.json"
        if not cache_path.is_file() or not index_path.is_file():
            raise RawTileCacheIntegrityError("tile cache metadata is missing")
        try:
            metadata = _read_json(cache_path)
            index_payload = _read_json(index_path)
        except (OSError, ValueError) as exc:
            raise RawTileCacheIntegrityError(f"tile cache metadata is malformed: {exc}") from exc
        if metadata.get("schema_version") != RAW_TILE_CACHE_SCHEMA_VERSION:
            raise RawTileCacheIntegrityError("unsupported tile-cache schema version")
        if index_payload.get("schema_version") != _INDEX_SCHEMA_VERSION:
            raise RawTileCacheIntegrityError("unsupported tile-cache index schema version")
        if config is None:
            config = RawTileCacheConfig(
                cell_size_m=float(metadata["config"]["cell_size_m"]),
                chunk_size=int(metadata["config"]["chunk_size"]),
                include_rgb=bool(metadata["config"]["include_rgb"]),
            )
        if metadata.get("configuration_digest") != config.configuration_digest():
            raise RawTileCacheIntegrityError("tile-cache configuration digest does not match")
        stored_source = metadata.get("source")
        if (
            not isinstance(stored_source, dict)
            or stored_source.get("source_content_identity_digest")
            != source_binding.source_content_identity_digest
            or stored_source.get("source_identity_record_digest")
            != source_binding.source_identity_record_digest
        ):
            raise RawTileCacheSourceMismatchError(
                "tile cache is bound to a different source identity than requested"
            )
        tiles = index_payload.get("tiles")
        if not isinstance(tiles, dict):
            raise RawTileCacheIntegrityError("tile-cache index tiles must be an object")
        index: dict[str, dict[str, Any]] = {}
        for tid, record in tiles.items():
            if not isinstance(record, dict):
                raise RawTileCacheIntegrityError(f"tile index record {tid} is malformed")
            file_name = record.get("file")
            if not isinstance(file_name, str) or file_name != f"{tid}.bin":
                raise RawTileCacheIntegrityError(f"tile index record {tid} file is not canonical")
            if not isinstance(record.get("point_count"), int) or record["point_count"] < 0:
                raise RawTileCacheIntegrityError(f"tile index record {tid} point_count is invalid")
            if not (root / "tiles" / file_name).is_file():
                raise RawTileCacheIntegrityError(f"tile file is missing: {file_name}")
            index[tid] = dict(record)
        totals = metadata.get("totals")
        if not isinstance(totals, dict):
            raise RawTileCacheIntegrityError("tile-cache totals metadata is missing")
        expected_points = sum(int(record["point_count"]) for record in index.values())
        if totals.get("scanned_point_count") != expected_points:
            raise RawTileCacheIntegrityError("tile-cache totals do not match the tile index")
        if totals.get("tile_count") != len(index):
            raise RawTileCacheIntegrityError("tile-cache tile_count does not match the tile index")
        expected_bytes = sum(
            (root / "tiles" / record["file"]).stat().st_size for record in index.values()
        )
        if totals.get("byte_size") != expected_bytes:
            raise RawTileCacheIntegrityError("tile-cache byte_size does not match the tile files")
        return cls(
            root,
            source_binding=source_binding,
            config=config,
            metadata=metadata,
            index=index,
        )

    # ---------------------------------------------------------------- query
    def query(
        self,
        seed_xyz_absolute_m: tuple[float, float, float],
        *,
        xy_radius_m: float,
        z_context_below_m: float = 20.0,
        z_context_above_m: float = 120.0,
    ) -> RawTileQueryResult:
        """Return bounded local RAW points around a seed (intersecting tiles + Z window)."""
        if isinstance(xy_radius_m, bool) or not isinstance(xy_radius_m, int | float):
            raise ValueError("xy_radius_m must be a finite positive number")
        if not math.isfinite(float(xy_radius_m)) or xy_radius_m <= 0:
            raise ValueError("xy_radius_m must be a finite positive number")
        if isinstance(z_context_below_m, bool) or not isinstance(
            z_context_below_m, int | float
        ):
            raise ValueError("z_context_below_m must be a finite non-negative number")
        if not math.isfinite(float(z_context_below_m)) or z_context_below_m < 0:
            raise ValueError("z_context_below_m must be a finite non-negative number")
        if isinstance(z_context_above_m, bool) or not isinstance(
            z_context_above_m, int | float
        ):
            raise ValueError("z_context_above_m must be a finite non-negative number")
        if not math.isfinite(float(z_context_above_m)) or z_context_above_m < 0:
            raise ValueError("z_context_above_m must be a finite non-negative number")
        seed = tuple(float(value) for value in seed_xyz_absolute_m)
        if len(seed) != 3:
            raise ValueError("seed_xyz_absolute_m must contain exactly 3 values")
        x0 = float(self.metadata["grid"]["x0"])
        y0 = float(self.metadata["grid"]["y0"])
        cell = self.config.cell_size_m
        col_lo = int(math.floor((seed[0] - xy_radius_m - x0) / cell))
        col_hi = int(math.floor((seed[0] + xy_radius_m - x0) / cell))
        row_lo = int(math.floor((seed[1] - xy_radius_m - y0) / cell))
        row_hi = int(math.floor((seed[1] + xy_radius_m - y0) / cell))
        z_lo = seed[2] - z_context_below_m
        z_hi = seed[2] + z_context_above_m

        candidates: list[str] = []
        for col in range(col_lo, col_hi + 1):
            for row in range(row_lo, row_hi + 1):
                tid = _tile_id(col, row)
                record = self.index.get(tid)
                if record is None:
                    continue
                t_min_z = float(record["min_xyz"][2])
                t_max_z = float(record["max_xyz"][2])
                if t_max_z < z_lo or t_min_z > z_hi:
                    # This tile's entire Z range lies outside the requested window.
                    continue
                candidates.append(tid)

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        idxs: list[np.ndarray] = []
        used: list[str] = []
        for tid in sorted(candidates):
            record = self.index[tid]
            x, y, z, idx = _read_tile_points(self.root / "tiles" / record["file"], self.config)
            r2 = xy_radius_m * xy_radius_m
            mask = (
                (x - seed[0]) ** 2 + (y - seed[1]) ** 2 <= r2
            ) & (z >= z_lo) & (z <= z_hi)
            if not np.any(mask):
                continue
            xs.append(x[mask])
            ys.append(y[mask])
            zs.append(z[mask])
            idxs.append(idx[mask])
            used.append(tid)
        if not xs:
            empty = np.empty(0, dtype=np.float64)
            empty_idx = np.empty(0, dtype=np.int64)
            return RawTileQueryResult(
                seed_xyz_absolute_m=seed,
                xy_radius_m=xy_radius_m,
                z_window_min_m=z_lo,
                z_window_max_m=z_hi,
                tiles=(),
                x=empty,
                y=empty,
                z=empty,
                source_index=empty_idx,
                point_count=0,
            )
        return RawTileQueryResult(
            seed_xyz_absolute_m=seed,
            xy_radius_m=xy_radius_m,
            z_window_min_m=z_lo,
            z_window_max_m=z_hi,
            tiles=tuple(used),
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            z=np.concatenate(zs),
            source_index=np.concatenate(idxs),
            point_count=int(sum(len(item) for item in xs)),
        )

    @property
    def byte_size(self) -> int:
        return int(self.metadata["totals"]["byte_size"])

    @property
    def tile_count(self) -> int:
        return int(self.metadata["totals"]["tile_count"])

    @property
    def point_count(self) -> int:
        return int(self.metadata["totals"]["scanned_point_count"])

    @property
    def preparation_runtime_seconds(self) -> float:
        return float(self.metadata.get("preparation_runtime_seconds", 0.0))

    @property
    def cache_id(self) -> str:
        return str(self.metadata["cache_id"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RAW_TILE_CACHE_SCHEMA_VERSION,
            "cache_id": self.cache_id,
            "root": str(self.root),
            "source": self.source_binding.to_dict(),
            "configuration_digest": self.config.configuration_digest(),
            "totals": dict(self.metadata["totals"]),
            "preparation_runtime_seconds": self.preparation_runtime_seconds,
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _read_tile_points(
    path: Path, config: RawTileCacheConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < _TILE_HEADER_SIZE or not data.startswith(_TILE_FILE_MAGIC):
        raise RawTileCacheIntegrityError(f"tile file is malformed: {path.name}")
    point_count = int.from_bytes(data[28:36], "little")
    stride = int.from_bytes(data[36:40], "little")
    flags = int.from_bytes(data[20:24], "little")
    expected_stride = config.stride_bytes
    if stride != expected_stride:
        raise RawTileCacheIntegrityError(
            f"tile file stride does not match cache config: {path.name}"
        )
    if point_count < 0 or len(data) != _TILE_HEADER_SIZE + point_count * stride:
        raise RawTileCacheIntegrityError(f"tile file size/count mismatch: {path.name}")
    body = data[_TILE_HEADER_SIZE:]
    if flags:
        block = np.frombuffer(body, dtype=np.dtype(
            [("a", "<f8"), ("b", "<f8"), ("c", "<f8"), ("d", "<i8"),
             ("e", "<u2"), ("f", "<u2"), ("g", "<u2")]
        ))
        return (
            block["a"].copy(),
            block["b"].copy(),
            block["c"].copy(),
            block["d"].copy(),
        )
    block = np.frombuffer(body, dtype=np.dtype(
        [("a", "<f8"), ("b", "<f8"), ("c", "<f8"), ("d", "<i8")]
    ))
    return (
        block["a"].copy(),
        block["b"].copy(),
        block["c"].copy(),
        block["d"].copy(),
    )


def coordinate_context_digest_ok(
    source_binding: SourceIdentityBinding, coordinate_context: CoordinateContext
) -> bool:
    """True when a coordinate context matches a source binding's context."""
    return coordinate_context == source_binding.coordinate_context


__all__ = [
    "RAW_TILE_CACHE_SCHEMA_VERSION",
    "RawTileCache",
    "RawTileCacheConfig",
    "RawTileCacheError",
    "RawTileCacheIntegrityError",
    "RawTileCacheSourceMismatchError",
    "RawTileQueryResult",
]
