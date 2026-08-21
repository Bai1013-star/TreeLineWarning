"""Bounded, read-only point-cloud data for the desktop manual tower picker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

OVERVIEW_LIMIT = 90_000
LOCAL_LIMIT = 35_000


def _rgb(chunk: Any, count: int) -> np.ndarray:
    names = set(chunk.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(names):
        values = np.column_stack((chunk.red, chunk.green, chunk.blue)).astype(np.float64)
        maximum = max(float(np.max(values)), 1.0)
        return np.clip(values * (255.0 / maximum), 0, 255).astype(np.uint8)
    return np.tile(np.asarray((165, 185, 205), dtype=np.uint8), (count, 1))


def overview(source: Path, *, maximum: int = OVERVIEW_LIMIT) -> dict[str, Any]:
    """Return a deterministic, bounded real-source overview without mutation."""
    import laspy

    with laspy.open(source) as reader:
        total = int(reader.header.point_count)
        bounds = reader.header.mins.tolist() + reader.header.maxs.tolist()
        stride = max(1, int(np.ceil(total / max(1, maximum))))
        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        intensity_parts: list[np.ndarray] = []
        offset = 0
        for chunk in reader.chunk_iterator(1_000_000):
            rows = np.arange(offset, offset + len(chunk), dtype=np.int64)
            selected = (rows % stride) == 0
            if selected.any():
                xyz_parts.append(
                    np.column_stack((chunk.x[selected], chunk.y[selected], chunk.z[selected]))
                )
                rgb_parts.append(_rgb(chunk, len(chunk))[selected])
                intensity_parts.append(np.asarray(chunk.intensity)[selected])
            offset += len(chunk)
    xyz = np.concatenate(xyz_parts)[:maximum] if xyz_parts else np.empty((0, 3))
    rgb = np.concatenate(rgb_parts)[:maximum] if rgb_parts else np.empty((0, 3), dtype=np.uint8)
    intensity = np.concatenate(intensity_parts)[:maximum] if intensity_parts else np.empty(0)
    return {
        "xyz": xyz.tolist(),
        "rgb": rgb.tolist(),
        "intensity": intensity.astype(float).tolist(),
        "source_bounds": bounds,
        "source_point_count": total,
        "display_point_count": len(xyz),
        "display_source": "READ_ONLY_STRIDED_REAL_LAS_OVERVIEW",
    }


def refine(source: Path, x: float, y: float, *, radius_m: float = 12.0) -> dict[str, Any]:
    """Make a conservative real-source local pick record and 3-D confirmation sample."""
    import laspy

    radius = max(2.0, min(float(radius_m), 30.0))
    parts: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    intensity: list[np.ndarray] = []
    with laspy.open(source) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
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
    distance = (xyz[:, 0] - x) ** 2 + (xyz[:, 1] - y) ** 2
    nearest = xyz[int(np.argmin(distance))]
    base_z, top_z = np.quantile(xyz[:, 2], (0.05, 0.99))
    height = max(0.1, float(top_z - base_z))
    # Preserve the operator's XY seed when the nearest raw point is distant;
    # this is manual picking with local evidence, never auto detection.
    pick_xy = nearest[:2] if float(np.sqrt(distance.min())) <= radius * 0.45 else np.asarray((x, y))
    if len(xyz) > LOCAL_LIMIT:
        rows = np.linspace(0, len(xyz) - 1, LOCAL_LIMIT, dtype=np.int64)
        display_xyz, display_rgb, display_intensity = xyz[rows], rgb[rows], values[rows]
    else:
        display_xyz, display_rgb, display_intensity = xyz, rgb, values
    return {
        "tower": {
            "x": float(pick_xy[0]), "y": float(pick_xy[1]), "z": float(base_z),
            "height_m": height, "refinement": "LOCAL_REAL_POINT_QUANTILES_V1",
            "local_point_count": int(len(xyz)),
        },
        "local": {"xyz": display_xyz.tolist(), "rgb": display_rgb.tolist(),
                  "intensity": display_intensity.astype(float).tolist(), "radius_m": radius},
    }
