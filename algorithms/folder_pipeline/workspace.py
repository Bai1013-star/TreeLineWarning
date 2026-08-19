"""Folder workspace discovery and fail-closed source validation.

A Folder pipeline workspace is a single directory holding exactly one source
LAS/LAZ under ``input/pointcloud/``.  If zero or more than one point cloud is
present the workspace FAILS CLOSED with a clear Chinese error — no science is
ever started against an ambiguous source.

Layout (V1):

    workspace/
      input/pointcloud/source.las | source.laz   (exactly one)
      source_cache/                              science staging (run checkpoints)
      01_span_selection/                         span_selection.json
      02_tree_analysis/                          algorithm original + edit log + final
      03_export/                                 Prediction_Input.xlsx + final html

Nothing here hard-codes an absolute drive path: the workspace is always passed
in by the launcher (``RUN_FOLDER_TO_EXCEL.bat <workspace>``) or resolved to the
repository default.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_SCHEMA_VERSION = "folder-to-excel-workspace-v1"
POINTCLOUD_DIR_NAME = "input"
POINTCLOUD_SUBDIR_NAME = "pointcloud"
SUPPORTED_POINTCLOUD_SUFFIXES = {".las", ".laz"}
SOURCE_CACHE_DIR = "source_cache"
SPAN_SELECTION_DIR = "01_span_selection"
TREE_ANALYSIS_DIR = "02_tree_analysis"
EXPORT_DIR = "03_export"
SCIENCE_SUBDIR = "science"


class FolderWorkspaceError(RuntimeError):
    """Fail-closed workspace or source error (Chinese message for the user)."""


def default_workspace_root(repository_root: Path | None = None) -> Path:
    """Return the repository default workspace (``workspace/`` next to the repo)."""

    root = repository_root
    if root is None:
        root = Path(__file__).resolve().parents[2]
    return root / "workspace"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_pointcloud(input_dir: Path) -> Path:
    if not input_dir.is_dir():
        raise FolderWorkspaceError(f"点云目录不存在: {input_dir}")
    candidates = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_POINTCLOUD_SUFFIXES
            and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FolderWorkspaceError(
            f"未找到点云文件: {input_dir} 下没有 .las / .laz 文件。"
            "请把唯一的源点云放入 workspace\\input\\pointcloud\\ 后再运行。"
        )
    if len(candidates) > 1:
        names = "、".join(path.name for path in candidates)
        raise FolderWorkspaceError(
            f"检测到多个点云文件（{len(candidates)} 个）：{names}。"
            "第一版一个工作区只允许一个源点云，请清理后重试。"
        )
    return candidates[0]


@dataclass(frozen=True, slots=True)
class FolderWorkspace:
    """Validated Folder pipeline workspace with one authoritative source."""

    root: Path
    source: Path
    source_sha256: str
    source_id: str

    @property
    def source_cache_dir(self) -> Path:
        return self.root / SOURCE_CACHE_DIR

    @property
    def science_run_root(self) -> Path:
        return self.source_cache_dir / SCIENCE_SUBDIR

    @property
    def span_selection_dir(self) -> Path:
        return self.root / SPAN_SELECTION_DIR

    @property
    def tree_analysis_dir(self) -> Path:
        return self.root / TREE_ANALYSIS_DIR

    @property
    def export_dir(self) -> Path:
        return self.root / EXPORT_DIR

    @property
    def span_selection_path(self) -> Path:
        return self.span_selection_dir / "span_selection.json"

    @property
    def original_tree_npz(self) -> Path:
        return self.tree_analysis_dir / "tree_segmentation_original.npz"

    @property
    def original_tree_manifest(self) -> Path:
        return self.tree_analysis_dir / "tree_segmentation_original.json"

    @property
    def tree_edits_path(self) -> Path:
        return self.tree_analysis_dir / "tree_edits.json"

    @property
    def final_tree_npz(self) -> Path:
        return self.tree_analysis_dir / "tree_segmentation_final.npz"

    @property
    def final_tree_manifest(self) -> Path:
        return self.tree_analysis_dir / "tree_segmentation_final.json"

    @property
    def excel_path(self) -> Path:
        return self.export_dir / "Prediction_Input.xlsx"

    @property
    def final_html_path(self) -> Path:
        return self.export_dir / "03_final_result.html"

    def ensure_dirs(self) -> None:
        for directory in (
            self.source_cache_dir,
            self.span_selection_dir,
            self.tree_analysis_dir,
            self.export_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "workspace_root": str(self.root),
            "source_file": str(self.source),
            "source_sha256": self.source_sha256,
            "source_id": self.source_id,
        }


def discover_workspace(
    workspace_root: Path | None = None,
    repository_root: Path | None = None,
) -> FolderWorkspace:
    """Validate a workspace and return the authoritative source + identity.

    Fails closed when the input directory contains zero or more than one point
    cloud.  The workspace root (and the repository root used to resolve the
    default workspace) must never appear in science logic as a hard-coded path
    — they are always passed in explicitly.
    """

    if workspace_root is not None:
        root = Path(workspace_root)
    else:
        root = default_workspace_root(repository_root)
    root = root.resolve()
    if not root.is_dir():
        raise FolderWorkspaceError(f"工作区目录不存在: {root}")
    input_dir = root / POINTCLOUD_DIR_NAME / POINTCLOUD_SUBDIR_NAME
    source = _single_pointcloud(input_dir)
    source_sha256 = _sha256_file(source)
    source_id = f"folder_source_{source_sha256[:16]}"
    workspace = FolderWorkspace(
        root=root,
        source=source,
        source_sha256=source_sha256,
        source_id=source_id,
    )
    workspace.ensure_dirs()
    return workspace


def resolve_sources(workspace_root: Path | None = None) -> Iterable[Path]:
    """Compatibility helper: the Folder workspace always owns exactly one source.

    The caller must already have validated the workspace; this returns the
    single authoritative source.  Raises ``FolderWorkspaceError`` otherwise.
    """

    return [discover_workspace(workspace_root).source]


__all__ = [
    "WORKSPACE_SCHEMA_VERSION",
    "POINTCLOUD_DIR_NAME",
    "POINTCLOUD_SUBDIR_NAME",
    "SUPPORTED_POINTCLOUD_SUFFIXES",
    "SOURCE_CACHE_DIR",
    "SPAN_SELECTION_DIR",
    "TREE_ANALYSIS_DIR",
    "EXPORT_DIR",
    "SCIENCE_SUBDIR",
    "FolderWorkspaceError",
    "FolderWorkspace",
    "default_workspace_root",
    "discover_workspace",
    "resolve_sources",
]
