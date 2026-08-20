"""Reusable source preparation and bounded local point-cloud access.

The preparation phase scans a source once to build a chunk spatial index.
Span reads then seek only intersecting source chunks.  This is deliberately a
small foundation for Stage B; it does not introduce COPC or replace the
existing FRV source provider.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class CacheState(str, Enum):
    NOT_PREPARED = "NOT_PREPARED"
    PREPARED = "PREPARED"
    STALE = "STALE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class SourceAccessMetrics:
    full_source_read_count: int = 0
    source_inspection_count: int = 0
    cache_prep_count: int = 0
    cache_reuse_count: int = 0
    span_local_access_count: int = 0
    selected_chunk_read_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "full_source_read_count": self.full_source_read_count,
            "source_inspection_count": self.source_inspection_count,
            "cache_prep_count": self.cache_prep_count,
            "cache_reuse_count": self.cache_reuse_count,
            "span_local_access_count": self.span_local_access_count,
            "selected_chunk_read_count": self.selected_chunk_read_count,
        }


@dataclass(frozen=True, slots=True)
class LocalPointChunk:
    xyz: np.ndarray
    original_index: np.ndarray
    source_chunk_start: int


@dataclass(frozen=True, slots=True)
class _ChunkRecord:
    start: int
    count: int
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


def _bounds(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("bounds must be (minimum_xyz, maximum_xyz)")
    minimum, maximum = value
    if len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("bounds must contain two three-value vectors")
    lower = tuple(float(item) for item in minimum)
    upper = tuple(float(item) for item in maximum)
    if any(lower[index] > upper[index] for index in range(3)):
        raise ValueError("bounds minimum must not exceed maximum")
    return lower, upper


class SharedSourceAccess:
    """One prepared LAS/LAZ source shared by future Span analyses."""

    def __init__(
        self,
        source: Path | str,
        *,
        cache_root: Path | str | None = None,
        chunk_size: int = 1_000_000,
    ) -> None:
        self.source = Path(source).expanduser().resolve()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self.cache_root = (
            Path(cache_root).expanduser().resolve() if cache_root is not None else None
        )
        self.state = CacheState.NOT_PREPARED
        self._chunks: tuple[_ChunkRecord, ...] = ()
        self._metrics = SourceAccessMetrics()
        self._source_fingerprint: tuple[int, int] | None = None

    @property
    def metrics(self) -> SourceAccessMetrics:
        return self._metrics

    @property
    def cache_manifest_path(self) -> Path | None:
        return None if self.cache_root is None else self.cache_root / "shared-source-index.json"

    def _replace_metrics(self, **changes: int) -> None:
        values = self._metrics.to_dict()
        values.update(changes)
        self._metrics = SourceAccessMetrics(**values)

    def _fingerprint(self) -> tuple[int, int]:
        stat = self.source.stat()
        return stat.st_size, stat.st_mtime_ns

    def prepare(self) -> SharedSourceAccess:
        """Inspect and index the source once; subsequent calls reuse the index."""

        if not self.source.is_file():
            self.state = CacheState.INVALID
            raise FileNotFoundError(self.source)
        fingerprint = self._fingerprint()
        if self.state is CacheState.PREPARED and self._source_fingerprint == fingerprint:
            self._replace_metrics(cache_reuse_count=self.metrics.cache_reuse_count + 1)
            return self
        self._replace_metrics(source_inspection_count=self.metrics.source_inspection_count + 1)
        if self.cache_manifest_path is not None and self.cache_manifest_path.is_file():
            try:
                payload = json.loads(self.cache_manifest_path.read_text(encoding="utf-8"))
                if (
                    payload.get("source_fingerprint") == list(fingerprint)
                    and payload.get("chunk_size") == self.chunk_size
                ):
                    self._chunks = tuple(
                        _ChunkRecord(
                            int(item["start"]),
                            int(item["count"]),
                            tuple(item["minimum"]),
                            tuple(item["maximum"]),
                        )
                        for item in payload["chunks"]
                    )
                    self._source_fingerprint = fingerprint
                    self.state = CacheState.PREPARED
                    self._replace_metrics(cache_reuse_count=self.metrics.cache_reuse_count + 1)
                    return self
            except (OSError, ValueError, KeyError, TypeError):
                self.state = CacheState.STALE
        import laspy

        records: list[_ChunkRecord] = []
        with laspy.open(self.source, mode="r") as reader:
            while True:
                start = (
                    int(reader.tell())
                    if hasattr(reader, "tell")
                    else sum(item.count for item in records)
                )
                points = reader.read_points(self.chunk_size)
                if len(points) == 0:
                    break
                xyz = np.column_stack(
                    (np.asarray(points.x), np.asarray(points.y), np.asarray(points.z))
                ).astype(np.float64, copy=False)
                records.append(
                    _ChunkRecord(
                        start,
                        int(len(xyz)),
                        tuple(np.min(xyz, axis=0).tolist()),
                        tuple(np.max(xyz, axis=0).tolist()),
                    )
                )
        self._chunks = tuple(records)
        self._source_fingerprint = fingerprint
        self.state = CacheState.PREPARED
        self._replace_metrics(
            full_source_read_count=self.metrics.full_source_read_count + 1,
            cache_prep_count=self.metrics.cache_prep_count + 1,
        )
        if self.cache_manifest_path is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "shared-source-index-v1",
                "source": str(self.source),
                "source_fingerprint": list(fingerprint),
                "chunk_size": self.chunk_size,
                "chunks": [
                    {
                        "start": item.start,
                        "count": item.count,
                        "minimum": list(item.minimum),
                        "maximum": list(item.maximum),
                    }
                    for item in self._chunks
                ],
            }
            temporary = self.cache_manifest_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.cache_manifest_path)
        return self

    @staticmethod
    def _intersects(
        item: _ChunkRecord, lower: tuple[float, float, float], upper: tuple[float, float, float]
    ) -> bool:
        return all(
            item.maximum[index] >= lower[index] and item.minimum[index] <= upper[index]
            for index in range(3)
        )

    def iter_local(
        self, bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    ) -> Iterator[LocalPointChunk]:
        """Yield only points in chunks intersecting *bounds*; no full scan per Span."""

        if self.state is not CacheState.PREPARED:
            raise RuntimeError("SharedSourceAccess.prepare() is required before local access")
        lower, upper = _bounds(bounds)
        selected = tuple(item for item in self._chunks if self._intersects(item, lower, upper))
        self._replace_metrics(
            cache_reuse_count=self.metrics.cache_reuse_count + 1,
            span_local_access_count=self.metrics.span_local_access_count + 1,
        )
        import laspy

        with laspy.open(self.source, mode="r") as reader:
            for item in selected:
                reader.seek(item.start)
                points = reader.read_points(item.count)
                xyz = np.column_stack(
                    (np.asarray(points.x), np.asarray(points.y), np.asarray(points.z))
                ).astype(np.float64, copy=False)
                mask = np.all((xyz >= np.asarray(lower)) & (xyz <= np.asarray(upper)), axis=1)
                if not np.any(mask):
                    continue
                self._replace_metrics(
                    selected_chunk_read_count=self.metrics.selected_chunk_read_count + 1
                )
                yield LocalPointChunk(
                    xyz[mask], np.flatnonzero(mask).astype(np.int64) + item.start, item.start
                )

    def iter_span(
        self, bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    ) -> Iterator[LocalPointChunk]:
        return self.iter_local(bounds)

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "metrics": self.metrics.to_dict(),
            "chunk_count": len(self._chunks),
            "cache_manifest": str(self.cache_manifest_path) if self.cache_manifest_path else None,
        }


__all__ = ["CacheState", "LocalPointChunk", "SharedSourceAccess", "SourceAccessMetrics"]
