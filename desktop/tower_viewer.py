"""Bounded, read-only point-cloud data for the desktop manual tower picker.

The cache in this module is presentation-only. It can always be deleted and
rebuilt from the source LAS/LAZ and is never consumed by scientific stages.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

OVERVIEW_LIMIT = 90_000
LOCAL_LIMIT = 35_000
_CHUNK_SIZE = 1_000_000
_CACHE_SCHEMA = "tower-viewer-overview-v2"


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _rgb(chunk: Any, count: int) -> np.ndarray:
    names = set(chunk.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(names):
        values = np.column_stack((chunk.red, chunk.green, chunk.blue)).astype(np.float64)
        maximum = max(float(np.max(values)), 1.0)
        return np.clip(values * (255.0 / maximum), 0, 255).astype(np.uint8)
    return np.tile(np.asarray((165, 185, 205), dtype=np.uint8), (count, 1))


def _cache_identity(
    source: Path, maximum: int, source_identity: Mapping[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    stat = source.stat()
    facts = {
        "schema": _CACHE_SCHEMA,
        "resolved_source": str(source.resolve()),
        "file_size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "maximum": int(maximum),
        "source_identity": dict(source_identity or {}),
    }
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), facts


def _cache_path(cache_dir: Path, digest: str) -> Path:
    return cache_dir / f"overview-{digest}.npz"


def _load_cache(path: Path | None, digest: str) -> dict[str, np.ndarray] | None:
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["schema"].item()) != _CACHE_SCHEMA:
                return None
            if str(stored["identity_digest"].item()) != digest:
                return None
            required = {
                "xyz", "rgb", "intensity", "source_bounds", "source_point_count",
                "chunk_start", "chunk_count", "chunk_bounds_xy",
            }
            if not required.issubset(stored.files):
                return None
            return {name: np.array(stored[name], copy=True) for name in required}
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(path: Path, digest: str, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(
                stream,
                schema=np.asarray(_CACHE_SCHEMA),
                identity_digest=np.asarray(digest),
                **arrays,
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _payload(arrays: Mapping[str, np.ndarray], timings: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    xyz = arrays["xyz"]
    result = {
        "xyz": xyz.tolist(),
        "rgb": arrays["rgb"].tolist(),
        "intensity": arrays["intensity"].astype(float).tolist(),
        "source_bounds": arrays["source_bounds"].astype(float).tolist(),
        "source_point_count": int(arrays["source_point_count"].item()),
        "display_point_count": int(len(xyz)),
        "display_source": "READ_ONLY_STRIDED_REAL_LAS_OVERVIEW",
        "viewer_cache": "HIT" if timings["overview_cache_hit"] else "BUILT",
    }
    timings["python_payload_build_ms"] = _elapsed_ms(started)
    result["diagnostic_timings"] = timings
    return result


def overview(
    source: Path,
    *,
    maximum: int = OVERVIEW_LIMIT,
    cache_dir: Path | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, bounded real-source XYZ overview without mutation."""
    import laspy

    total_started = time.perf_counter()
    maximum = max(1, int(maximum))
    digest, _facts = _cache_identity(source, maximum, source_identity)
    cache_path = _cache_path(cache_dir, digest) if cache_dir is not None else None
    lookup_started = time.perf_counter()
    cached = _load_cache(cache_path, digest)
    timings: dict[str, Any] = {
        "source_open_ms": 0.0,
        "overview_cache_lookup_ms": _elapsed_ms(lookup_started),
        "overview_build_ms": 0.0,
        "overview_cache_hit": cached is not None,
        "overview_cache_identity": digest,
    }
    if cached is not None:
        timings["first_python_ready_ms"] = _elapsed_ms(total_started)
        return _payload(cached, timings)

    build_started = time.perf_counter()
    open_started = time.perf_counter()
    reader = laspy.open(source)
    timings["source_open_ms"] = _elapsed_ms(open_started)
    try:
        total = int(reader.header.point_count)
        bounds = np.asarray(reader.header.mins.tolist() + reader.header.maxs.tolist())
        stride = max(1, int(np.ceil(total / maximum)))
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        intensity_parts: list[np.ndarray] = []
        chunk_start: list[int] = []
        chunk_count: list[int] = []
        chunk_bounds: list[tuple[float, float, float, float]] = []
        offset = 0
        for chunk in reader.chunk_iterator(_CHUNK_SIZE):
            cx, cy = np.asarray(chunk.x), np.asarray(chunk.y)
            chunk_start.append(offset)
            chunk_count.append(len(chunk))
            chunk_bounds.append(
                (float(cx.min()), float(cy.min()), float(cx.max()), float(cy.max()))
            )
            rows = np.arange(offset, offset + len(chunk), dtype=np.int64)
            selected = (rows % stride) == 0
            if selected.any():
                xyz_parts.append(
                    np.column_stack((cx[selected], cy[selected], np.asarray(chunk.z)[selected]))
                )
                rgb_parts.append(_rgb(chunk, len(chunk))[selected])
                intensity_parts.append(np.asarray(chunk.intensity)[selected])
            offset += len(chunk)
    finally:
        reader.close()
    xyz = np.concatenate(xyz_parts)[:maximum] if xyz_parts else np.empty((0, 3))
    arrays = {
        "xyz": xyz,
        "rgb": (
            np.concatenate(rgb_parts)[:maximum]
            if rgb_parts
            else np.empty((0, 3), dtype=np.uint8)
        ),
        "intensity": np.concatenate(intensity_parts)[:maximum] if intensity_parts else np.empty(0),
        "source_bounds": bounds,
        "source_point_count": np.asarray(total, dtype=np.int64),
        "chunk_start": np.asarray(chunk_start, dtype=np.int64),
        "chunk_count": np.asarray(chunk_count, dtype=np.int64),
        "chunk_bounds_xy": np.asarray(chunk_bounds, dtype=np.float64),
    }
    timings["overview_build_ms"] = _elapsed_ms(build_started)
    if cache_path is not None:
        _write_cache(cache_path, digest, arrays)
    timings["first_python_ready_ms"] = _elapsed_ms(total_started)
    return _payload(arrays, timings)


def _candidate_chunks(
    index: Mapping[str, np.ndarray], x: float, y: float, radius: float
) -> np.ndarray:
    bounds = index["chunk_bounds_xy"]
    return np.flatnonzero(
        (bounds[:, 0] <= x + radius)
        & (bounds[:, 2] >= x - radius)
        & (bounds[:, 1] <= y + radius)
        & (bounds[:, 3] >= y - radius)
    )


def refine(
    source: Path,
    x: float,
    y: float,
    *,
    z: float | None = None,
    radius_m: float = 12.0,
    cache_dir: Path | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a conservative real-source local pick record and 3-D confirmation sample."""
    import laspy

    query_started = time.perf_counter()
    radius = max(2.0, min(float(radius_m), 30.0))
    digest, _facts = _cache_identity(source, OVERVIEW_LIMIT, source_identity)
    cache_path = _cache_path(cache_dir, digest) if cache_dir is not None else None
    index = _load_cache(cache_path, digest)
    parts: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    intensity: list[np.ndarray] = []
    chunks_read = 0
    with laspy.open(source) as reader:
        if index is None:
            iterator = reader.chunk_iterator(_CHUNK_SIZE)
        else:
            candidates = _candidate_chunks(index, x, y, radius)

            def selected_chunks() -> Any:
                for candidate in candidates:
                    reader.seek(int(index["chunk_start"][candidate]))
                    yield reader.read_points(int(index["chunk_count"][candidate]))

            iterator = selected_chunks()
        for chunk in iterator:
            chunks_read += 1
            cx, cy = np.asarray(chunk.x), np.asarray(chunk.y)
            mask = (cx - x) ** 2 + (cy - y) ** 2 <= radius**2
            if mask.any():
                parts.append(np.column_stack((cx[mask], cy[mask], np.asarray(chunk.z)[mask])))
                colors.append(_rgb(chunk, len(chunk))[mask])
                intensity.append(np.asarray(chunk.intensity)[mask])
    if not parts:
        raise ValueError("点击位置附近未找到真实点云，请缩放后重试。")
    xyz = np.concatenate(parts)
    rgb = np.concatenate(colors)
    values = np.concatenate(intensity)
    xy_distance = (xyz[:, 0] - x) ** 2 + (xyz[:, 1] - y) ** 2
    pick_distance = (
        xy_distance if z is None else xy_distance + (xyz[:, 2] - float(z)) ** 2
    )
    nearest = xyz[int(np.argmin(pick_distance))]
    base_z, top_z = np.quantile(xyz[:, 2], (0.05, 0.99))
    height = max(0.1, float(top_z - base_z))
    pick_xy = (
        nearest[:2]
        if float(np.sqrt(xy_distance.min())) <= radius * 0.45
        else np.asarray((x, y))
    )
    if len(xyz) > LOCAL_LIMIT:
        rows = np.linspace(0, len(xyz) - 1, LOCAL_LIMIT, dtype=np.int64)
        display_xyz, display_rgb, display_intensity = xyz[rows], rgb[rows], values[rows]
    else:
        display_xyz, display_rgb, display_intensity = xyz, rgb, values
    query_ms = _elapsed_ms(query_started)
    payload_started = time.perf_counter()
    result = {
        "tower": {
            "x": float(pick_xy[0]), "y": float(pick_xy[1]), "z": float(base_z),
            "height_m": height, "refinement": "LOCAL_REAL_POINT_QUANTILES_V1",
            "local_point_count": int(len(xyz)),
            "picked_source_xyz": [float(value) for value in nearest],
        },
        "local": {"xyz": display_xyz.tolist(), "rgb": display_rgb.tolist(),
                  "intensity": display_intensity.astype(float).tolist(), "radius_m": radius},
    }
    result["diagnostic_timings"] = {
        "local_crop_query_ms": query_ms,
        "local_crop_points": int(len(xyz)),
        "local_chunks_read": chunks_read,
        "local_index_used": index is not None,
        "python_payload_build_ms": _elapsed_ms(payload_started),
    }
    return result
