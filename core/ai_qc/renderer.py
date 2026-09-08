"""Deterministic six-view point-cloud renderer for advisory AI-QC only."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RENDER_REVISION = "tree-ai-qc-render-v1"
VIEW_NAMES = ("TOP_XY", "FRONT_XZ", "SIDE_YZ", "OBLIQUE_A", "OBLIQUE_B", "CONTEXT")
_IMAGE_SIZE = 512
_MAX_TREE_POINTS = 18_000
_MAX_CONTEXT_POINTS = 14_000
_BACKGROUND = np.asarray((13, 18, 24), dtype=np.uint8)
_TREE_COLOR = np.asarray((255, 190, 48), dtype=np.uint8)
_TREE_CORE = np.asarray((255, 244, 184), dtype=np.uint8)
_CONTEXT_COLOR = np.asarray((68, 126, 139), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class RenderedTreeViews:
    tree_id: str
    render_revision: str
    source_editor_revision: str
    membership_sha256: str
    view_files: tuple[Path, ...]
    view_sha256: tuple[str, ...]
    contact_sheet: Path
    contact_sheet_sha256: str
    elapsed_seconds: float
    metadata_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_digest(original_index: np.ndarray) -> str:
    ordered = np.sort(np.asarray(original_index, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(RENDER_REVISION.encode("ascii"))
    digest.update(ordered.tobytes())
    return digest.hexdigest()


def _sample(xyz: np.ndarray, original_index: np.ndarray, limit: int) -> np.ndarray:
    if len(xyz) <= limit:
        return np.asarray(xyz, dtype=np.float64)
    order = np.argsort(np.asarray(original_index, dtype=np.int64), kind="mergesort")
    positions = np.linspace(0, len(order) - 1, limit, dtype=np.int64)
    return np.asarray(xyz[order[positions]], dtype=np.float64)


def _pixel_cloud(
    uv: np.ndarray,
    *,
    limits: tuple[float, float, float, float],
    color: np.ndarray,
    radius: int,
) -> np.ndarray:
    width = height = _IMAGE_SIZE
    u0, u1, v0, v1 = limits
    span_u, span_v = max(u1 - u0, 1e-9), max(v1 - v0, 1e-9)
    px = np.rint(28 + (uv[:, 0] - u0) * (width - 57) / span_u).astype(np.int32)
    py = np.rint(height - 29 - (uv[:, 1] - v0) * (height - 57) / span_v).astype(np.int32)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x, y = px[valid] + dx, py[valid] + dy
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            layer[y[inside], x[inside], :3] = color
            layer[y[inside], x[inside], 3] = 255
    return layer


def _project(xyz: np.ndarray, view: str) -> np.ndarray:
    centered = xyz.copy()
    if view in {"TOP_XY", "CONTEXT"}:
        return centered[:, [0, 1]]
    if view == "FRONT_XZ":
        return centered[:, [0, 2]]
    if view == "SIDE_YZ":
        return centered[:, [1, 2]]
    angle = np.deg2rad(38.0 if view == "OBLIQUE_A" else 142.0)
    horizontal = centered[:, 0] * np.cos(angle) + centered[:, 1] * np.sin(angle)
    depth = -centered[:, 0] * np.sin(angle) + centered[:, 1] * np.cos(angle)
    return np.column_stack((horizontal, centered[:, 2] + depth * 0.28))


def _limits(uv: np.ndarray) -> tuple[float, float, float, float]:
    low = np.min(uv, axis=0)
    high = np.max(uv, axis=0)
    span = np.maximum(high - low, 0.25)
    margin = span * 0.08
    return (
        float(low[0] - margin[0]),
        float(high[0] + margin[0]),
        float(low[1] - margin[1]),
        float(high[1] + margin[1]),
    )


def _compose_view(
    tree_xyz: np.ndarray,
    context_xyz: np.ndarray,
    view: str,
    tree_id: str,
) -> Image.Image:
    tree_uv = _project(tree_xyz, view)
    context_uv = _project(context_xyz, view) if len(context_xyz) else np.empty((0, 2))
    bounds_source = (
        np.vstack((tree_uv, context_uv)) if view == "CONTEXT" and len(context_uv) else tree_uv
    )
    bounds = _limits(bounds_source)
    canvas = np.broadcast_to(_BACKGROUND, (_IMAGE_SIZE, _IMAGE_SIZE, 3)).copy()
    image = Image.fromarray(canvas, mode="RGB").convert("RGBA")
    if view == "CONTEXT" and len(context_uv):
        image.alpha_composite(
            Image.fromarray(
                _pixel_cloud(context_uv, limits=bounds, color=_CONTEXT_COLOR, radius=1), mode="RGBA"
            )
        )
    image.alpha_composite(
        Image.fromarray(
            _pixel_cloud(tree_uv, limits=bounds, color=_TREE_COLOR, radius=1), mode="RGBA"
        )
    )
    if len(tree_uv):
        high = tree_xyz[:, 2] >= np.quantile(tree_xyz[:, 2], 0.9)
        image.alpha_composite(
            Image.fromarray(
                _pixel_cloud(tree_uv[high], limits=bounds, color=_TREE_CORE, radius=2), mode="RGBA"
            )
        )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, _IMAGE_SIZE - 1, 23), fill=(5, 8, 12, 230))
    draw.text((8, 6), f"{view}  |  {tree_id}", font=font, fill=(244, 247, 250, 255))
    draw.rectangle((0, 0, _IMAGE_SIZE - 1, _IMAGE_SIZE - 1), outline=(92, 106, 117, 255), width=1)
    return image.convert("RGB")


def render_tree_views(
    *,
    tree_id: str,
    tree_xyz: np.ndarray,
    tree_original_index: np.ndarray,
    scene_xyz: np.ndarray,
    scene_tree_id: np.ndarray,
    selected_tree_id: int,
    output_dir: Path | str,
    source_editor_revision: str,
) -> RenderedTreeViews:
    """Render standardized advisory views without modifying source coordinates."""

    started = time.perf_counter()
    tree_xyz = np.asarray(tree_xyz)
    tree_original_index = np.asarray(tree_original_index)
    if tree_xyz.ndim != 2 or tree_xyz.shape[1] != 3 or len(tree_xyz) == 0:
        raise ValueError("AI_QC_TREE_XYZ_INVALID")
    if tree_original_index.shape != (len(tree_xyz),):
        raise ValueError("AI_QC_TREE_INDEX_ALIGNMENT")
    sampled_tree = _sample(tree_xyz, tree_original_index, _MAX_TREE_POINTS)
    center = np.median(sampled_tree, axis=0)
    tree_span = np.maximum(np.max(sampled_tree, axis=0) - np.min(sampled_tree, axis=0), 0.5)
    radius_xy = max(float(tree_span[0]), float(tree_span[1])) * 1.5 + 2.0
    scene_xyz = np.asarray(scene_xyz)
    scene_tree_id = np.asarray(scene_tree_id)
    nearby = (
        (scene_tree_id != int(selected_tree_id))
        & (np.abs(scene_xyz[:, 0] - center[0]) <= radius_xy)
        & (np.abs(scene_xyz[:, 1] - center[1]) <= radius_xy)
        & (scene_xyz[:, 2] >= np.min(sampled_tree[:, 2]) - 2.0)
        & (scene_xyz[:, 2] <= np.max(sampled_tree[:, 2]) + 2.0)
    )
    context_positions = np.flatnonzero(nearby)
    if len(context_positions) > _MAX_CONTEXT_POINTS:
        picks = np.linspace(0, len(context_positions) - 1, _MAX_CONTEXT_POINTS, dtype=np.int64)
        context_positions = context_positions[picks]
    context_xyz = np.asarray(scene_xyz[context_positions], dtype=np.float64)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    hashes: list[str] = []
    images: list[Image.Image] = []
    for view in VIEW_NAMES:
        image = _compose_view(sampled_tree, context_xyz, view, tree_id)
        path = out / f"{view}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        files.append(path)
        hashes.append(_sha256(path))
        images.append(image)
    sheet = Image.new("RGB", (_IMAGE_SIZE * 3, _IMAGE_SIZE * 2), tuple(_BACKGROUND.tolist()))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 3) * _IMAGE_SIZE, (index // 3) * _IMAGE_SIZE))
    sheet_path = out / "CONTACT_SHEET.png"
    sheet.save(sheet_path, format="PNG", optimize=False, compress_level=9)
    sheet_hash = _sha256(sheet_path)
    member_hash = membership_digest(tree_original_index)
    elapsed = time.perf_counter() - started
    metadata_path = out / "render.json"
    payload: dict[str, Any] = {
        "schema": "tree-ai-qc-render-manifest-v1",
        "tree_id": tree_id,
        "render_revision": RENDER_REVISION,
        "source_editor_revision": source_editor_revision,
        "membership_sha256": member_hash,
        "view_files": [path.name for path in files],
        "view_sha256": hashes,
        "contact_sheet": sheet_path.name,
        "contact_sheet_sha256": sheet_hash,
        "tree_source_point_count": int(len(tree_xyz)),
        "tree_render_point_count": int(len(sampled_tree)),
        "context_render_point_count": int(len(context_xyz)),
        "elapsed_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RenderedTreeViews(
        tree_id=tree_id,
        render_revision=RENDER_REVISION,
        source_editor_revision=source_editor_revision,
        membership_sha256=member_hash,
        view_files=tuple(files),
        view_sha256=tuple(hashes),
        contact_sheet=sheet_path,
        contact_sheet_sha256=sheet_hash,
        elapsed_seconds=elapsed,
        metadata_path=metadata_path,
    )
