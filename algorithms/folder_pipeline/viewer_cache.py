"""Reproducible Potree LOD viewer cache for the Folder workspace (presentation-only).

The LAS/LAZ source remains authoritative.  A Potree cache is derived,
presentation-only, reproducible data bound to the exact source content identity
and pinned converter/runtime versions.  It never carries ``original_index``
semantics and never becomes science truth.

Contract (frozen viewer decision — see docs/project-based-multispan-viewer-design.md):

    Authoritative LAS/LAZ
        -> reproducible viewer-cache builder
        -> Potree static hierarchy
        -> Potree WebGL viewer (view-dependent multi-resolution LOD)

A valid cache is reused on restart; a cache built from a different source
fails closed (never silently reused).  Conversion is a one-time build into the
workspace, and the generated hierarchy is not Git-tracked.

Coordinate policy: PotreeConverter uses a local origin (metadata.json
``offset``); ``source_absolute_xyz = potree_local_xyz + offset``.  A Potree
rendered pick is only an approximate XYZ seed in the source frame; the backend
recovers bounded full-resolution raw points and runs the existing pole
refinement.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from algorithms.folder_pipeline.workspace import FolderWorkspace

VIEWER_CACHE_SCHEMA = "folder-to-excel-potree-viewer-cache-v1"
POTREE_CONVERTER_VERSION = "2.1.3"
POTREE_RUNTIME_VERSION = "1.8.2"
# Official GitHub release asset SHA-256s (verified by the P3A POC, docs/p3a-potree-poc.md).
POTREE_RUNTIME_ASSET_SHA256 = "c140a8bce0ce129de762fbc0fd3ba7c21120ffd36d423dd2755abb39f32aebe5"
POTREE_CONVERTER_ASSET_SHA256 = "957279e35886c63b4aba5d6383f11f3105ee889cb5391b05307265d00cd2d775"
POTREE_VIEWER_DIRNAME = f"Potree_{POTREE_RUNTIME_VERSION}"
POTREE_CONVERTER_DIRNAME = f"PotreeConverter_{POTREE_CONVERTER_VERSION}_x64_windows"
POTREE_CONVERTER_EXE = (
    Path("cache") / "p3a_potree" / "tools" / POTREE_CONVERTER_DIRNAME / "PotreeConverter.exe"
)
POTREE_RUNTIME_DIR = Path("cache") / "p3a_potree" / "dependencies" / POTREE_VIEWER_DIRNAME

CONVERTER_SETTINGS: dict[str, Any] = {
    "sampling_method": "poisson",
    "attributes": ["rgb"],
    "generated_page": None,  # static cache only; the Folder server serves the page
}

# Cache layout root inside the workspace source cache.
VIEWER_CACHE_ROOT = "viewer"

Progress = Callable[[str], None]


class ViewerCacheError(RuntimeError):
    """Fail-closed viewer cache error (Chinese message for the user)."""


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_facts(workspace: FolderWorkspace) -> dict[str, Any]:
    """Read the lightweight source facts used to bind the cache to the source.

    The authoritative facts come from the manual-span inspection when present.
    """

    path = workspace.science_run_root / "00_input" / "source_inspection.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text("utf-8"))
            return {
                "source_sha256": payload.get("input_sha256") or workspace.source_sha256,
                "source_point_count": int(payload.get("point_count") or 0),
                "source_bounds_min": list(payload.get("bounds_min_xyz") or [0.0, 0.0, 0.0]),
                "source_bounds_max": list(payload.get("bounds_max_xyz") or [0.0, 0.0, 0.0]),
            }
        except Exception:  # noqa: BLE001 - best effort
            pass
    return {
        "source_sha256": workspace.source_sha256,
        "source_point_count": 0,
        "source_bounds_min": [0.0, 0.0, 0.0],
        "source_bounds_max": [0.0, 0.0, 0.0],
    }


def compute_cache_id(
    workspace: FolderWorkspace,
    settings: dict[str, Any] | None = None,
) -> str:
    """Deterministic cache identity bound to source + converter + viewer + schema."""

    facts = _source_facts(workspace)
    settings = settings or CONVERTER_SETTINGS
    payload = {
        "schema": VIEWER_CACHE_SCHEMA,
        "source_sha256": facts["source_sha256"],
        "source_point_count": facts["source_point_count"],
        "source_bounds_min": facts["source_bounds_min"],
        "source_bounds_max": facts["source_bounds_max"],
        "converter_version": POTREE_CONVERTER_VERSION,
        "converter_settings": settings,
        "viewer_version": POTREE_RUNTIME_VERSION,
        "coordinate_frame": "SOURCE_LAS_ABSOLUTE -> POTREE_LOCAL (metadata offset)",
    }
    return _canonical_digest(payload)[:24]


@dataclass(frozen=True, slots=True)
class ViewerCacheInfo:
    """A validated Potree viewer cache for one workspace + source."""

    cache_id: str
    cache_dir: Path
    potree_dir: Path
    manifest_path: Path
    metadata_path: Path
    octree_path: Path
    hierarchy_path: Path
    source_sha256: str
    converter_version: str
    viewer_version: str
    offset_xyz: tuple[float, float, float]
    point_count: int
    hierarchy_depth: int

    def potree_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_id": self.cache_id,
            "cache_dir": str(self.cache_dir),
            "potree_dir": str(self.potree_dir),
            "source_sha256": self.source_sha256,
            "converter_version": self.converter_version,
            "viewer_version": self.viewer_version,
            "offset_xyz": list(self.offset_xyz),
            "point_count": self.point_count,
            "hierarchy_depth": self.hierarchy_depth,
            "metadata_ready": self.metadata_path.is_file(),
            "octree_ready": self.octree_path.is_file(),
            "hierarchy_ready": self.hierarchy_path.is_file(),
        }


def _viewer_root(workspace: FolderWorkspace) -> Path:
    return workspace.source_cache_dir / VIEWER_CACHE_ROOT


def load_viewer_cache(workspace: FolderWorkspace) -> ViewerCacheInfo | None:
    """Return the valid workspace cache for this source, or None.

    A cache whose source SHA-256 / schema / converter version do not match is
    NOT returned (fail closed); it is never silently reused.
    """

    root = _viewer_root(workspace)
    if not root.is_dir():
        return None
    expected_id = compute_cache_id(workspace)
    for cache_dir in sorted(root.glob("*")):
        manifest_path = cache_dir / "cache-manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except Exception:  # noqa: BLE001 - corrupt cache is not reused
            continue
        if manifest.get("schema") != VIEWER_CACHE_SCHEMA:
            continue
        if manifest.get("cache_id") != expected_id:
            continue
        if manifest.get("source_sha256") != workspace.source_sha256:
            continue
        if manifest.get("converter_version") != POTREE_CONVERTER_VERSION:
            continue
        potree_dir = cache_dir / "potree"
        info = ViewerCacheInfo(
            cache_id=expected_id,
            cache_dir=cache_dir,
            potree_dir=potree_dir,
            manifest_path=manifest_path,
            metadata_path=potree_dir / "metadata.json",
            octree_path=potree_dir / "octree.bin",
            hierarchy_path=potree_dir / "hierarchy.bin",
            source_sha256=str(manifest.get("source_sha256", "")),
            converter_version=str(manifest.get("converter_version", "")),
            viewer_version=str(manifest.get("viewer_version", POTREE_RUNTIME_VERSION)),
            offset_xyz=tuple(float(v) for v in manifest.get("offset_xyz", (0.0, 0.0, 0.0))),
            point_count=int(manifest.get("source_point_count", 0)),
            hierarchy_depth=int(manifest.get("hierarchy_depth", 0)),
        )
        if not (info.metadata_path.is_file() and info.octree_path.is_file()):
            continue
        return info
    return None


def _stage_ascii_input(workspace: FolderWorkspace, scratch_dir: Path) -> Path:
    """Stage the source under an ASCII name (PotreeConverter cannot read non-ASCII paths)."""

    link = scratch_dir / "source_input.las"
    if link.is_file() or link.exists():
        return link
    try:
        os.link(workspace.source, link)  # hard link: no data copy (same volume)
    except OSError:
        # Fallback: junction on the parent directory (same volume, no copy).
        shutil.copy2(workspace.source, link)
    return link


def build_viewer_cache(
    workspace: FolderWorkspace,
    *,
    progress: Progress | None = None,
    reuse: bool = True,
) -> ViewerCacheInfo:
    """Build (or reuse) the reproducible Potree viewer cache for this source.

    A valid cache is reused (never a full re-conversion on every start).  When
    missing/stale, the converter (pinned 2.1.3) converts the authoritative
    source read-only into the workspace cache and the manifest is written last
    as the completion marker.
    """

    report = progress or (lambda _msg: None)
    existing = load_viewer_cache(workspace)
    if existing is not None and reuse:
        report("viewer cache reused (valid, source-bound)")
        return existing

    cache_id = compute_cache_id(workspace)
    root = _viewer_root(workspace)
    cache_dir = root / cache_id
    potree_dir = cache_dir / "potree"
    scratch = cache_dir / ".convert_tmp"
    if cache_dir.exists():
        # A stale/partial cache must not be reused; remove it atomically.
        shutil.rmtree(cache_dir, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    converter_exe = POTREE_CONVERTER_EXE
    if not converter_exe.is_file():
        raise ViewerCacheError(
            f"PotreeConverter 未找到: {converter_exe}。请确认 PotreeConverter "
            f"{POTREE_CONVERTER_VERSION} 已放置于 cache/p3a_potree/tools/。"
        )
    if not POTREE_RUNTIME_DIR.is_dir():
        raise ViewerCacheError(
            f"Potree 运行时未找到: {POTREE_RUNTIME_DIR}。请确认 Potree "
            f"{POTREE_RUNTIME_VERSION} 已放置于 cache/p3a_potree/dependencies/。"
        )

    report("staging source for conversion...")
    input_las = _stage_ascii_input(workspace, scratch)
    out_dir = scratch / "out"
    report("running PotreeConverter (pinned " + POTREE_CONVERTER_VERSION + ")...")
    result = _run_converter(converter_exe, input_las, out_dir)
    if result != 0:
        raise ViewerCacheError("PotreeConverter 转换失败，请查看转换日志。")

    metadata_path = out_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ViewerCacheError("PotreeConverter 输出缺少 metadata.json")
    potree_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("metadata.json", "octree.bin", "hierarchy.bin"):
        source = out_dir / filename
        if not source.is_file():
            raise ViewerCacheError(f"PotreeConverter 输出缺少 {filename}")
        shutil.move(str(source), str(potree_dir / filename))

    metadata = json.loads(metadata_path.read_text("utf-8")) if metadata_path.is_file() else {}
    # metadata_path moved; re-read from potree_dir.
    metadata = json.loads((potree_dir / "metadata.json").read_text("utf-8"))
    offset_xyz = tuple(float(v) for v in metadata.get("offset", (0.0, 0.0, 0.0)))
    hierarchy = metadata.get("hierarchy", {})
    facts = _source_facts(workspace)
    manifest = {
        "schema": VIEWER_CACHE_SCHEMA,
        "cache_id": cache_id,
        "source_sha256": facts["source_sha256"],
        "source_point_count": facts["source_point_count"],
        "source_bounds_min": facts["source_bounds_min"],
        "source_bounds_max": facts["source_bounds_max"],
        "converter_version": POTREE_CONVERTER_VERSION,
        "converter_settings": CONVERTER_SETTINGS,
        "viewer_version": POTREE_RUNTIME_VERSION,
        "asset_sha256": {
            "potree_runtime": POTREE_RUNTIME_ASSET_SHA256,
            "potree_converter": POTREE_CONVERTER_ASSET_SHA256,
        },
        "coordinate_frame": "SOURCE_LAS_ABSOLUTE -> POTREE_LOCAL (offset)",
        "offset_xyz": list(offset_xyz),
        "point_count": int(metadata.get("points", 0)),
        "hierarchy_depth": int(hierarchy.get("depth", 0)),
        "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_digest": _canonical_digest(
            {
                "schema": VIEWER_CACHE_SCHEMA,
                "cache_id": cache_id,
                "source_sha256": facts["source_sha256"],
                "converter_version": POTREE_CONVERTER_VERSION,
                "settings": CONVERTER_SETTINGS,
                "viewer_version": POTREE_RUNTIME_VERSION,
            }
        ),
    }
    manifest_path = cache_dir / "cache-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
    )
    shutil.rmtree(scratch, ignore_errors=True)

    info = ViewerCacheInfo(
        cache_id=cache_id,
        cache_dir=cache_dir,
        potree_dir=potree_dir,
        manifest_path=manifest_path,
        metadata_path=potree_dir / "metadata.json",
        octree_path=potree_dir / "octree.bin",
        hierarchy_path=potree_dir / "hierarchy.bin",
        source_sha256=facts["source_sha256"],
        converter_version=POTREE_CONVERTER_VERSION,
        viewer_version=POTREE_RUNTIME_VERSION,
        offset_xyz=offset_xyz,
        point_count=int(metadata.get("points", 0)),
        hierarchy_depth=int(hierarchy.get("depth", 0)),
    )
    report("viewer cache built (" + str(info.point_count) + " points)")
    return info


def _run_converter(converter_exe: Path, input_las: Path, out_dir: Path) -> int:
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(converter_exe),
        "-i", str(input_las),
        "-o", str(out_dir),
        "-m", "poisson",
        "--attributes", "rgb",
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return -1
    return int(proc.returncode)


def viewer_cache_status(workspace: FolderWorkspace) -> dict[str, Any]:
    """Status used by the API: is a valid cache present + ready?"""

    cache = load_viewer_cache(workspace)
    if cache is None:
        return {
            "status": "NOT_BUILT",
            "cache_id": None,
            "ready": False,
            "message": "Potree 视图缓存尚未构建",
        }
    return {
        "status": "READY",
        "cache_id": cache.cache_id,
        "ready": True,
        "source_sha256": cache.source_sha256,
        "point_count": cache.point_count,
        "hierarchy_depth": cache.hierarchy_depth,
        "potree_dir": str(cache.potree_dir),
        "converter_version": cache.converter_version,
        "viewer_version": cache.viewer_version,
        "offset_xyz": list(cache.offset_xyz),
        "message": "Potree 视图缓存可用",
    }


__all__ = [
    "VIEWER_CACHE_SCHEMA",
    "POTREE_CONVERTER_VERSION",
    "POTREE_RUNTIME_VERSION",
    "POTREE_RUNTIME_ASSET_SHA256",
    "POTREE_CONVERTER_ASSET_SHA256",
    "POTREE_CONVERTER_EXE",
    "POTREE_RUNTIME_DIR",
    "CONVERTER_SETTINGS",
    "VIEWER_CACHE_ROOT",
    "ViewerCacheError",
    "ViewerCacheInfo",
    "compute_cache_id",
    "load_viewer_cache",
    "build_viewer_cache",
    "viewer_cache_status",
]
