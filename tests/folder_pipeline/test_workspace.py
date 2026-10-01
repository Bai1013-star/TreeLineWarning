"""Folder workspace fail-closed source validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from algorithms.folder_pipeline.workspace import (
    FolderWorkspaceError,
    discover_workspace,
)
from tests.folder_pipeline.conftest import write_synthetic_las


def _pointcloud(workspace: Path) -> Path:
    return workspace / "input" / "pointcloud"


def test_zero_sources_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _pointcloud(workspace).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FolderWorkspaceError) as exc:
        discover_workspace(workspace)
    assert "未找到点云文件" in str(exc.value)


def test_multiple_sources_fails_closed(tmp_path: Path) -> None:
    import shutil

    workspace = tmp_path / "ws"
    pointcloud = _pointcloud(workspace)
    pointcloud.mkdir(parents=True, exist_ok=True)
    rng = __import__("numpy").random.default_rng(1)
    xyz = rng.uniform([0, 0, 0], [100, 100, 30], (100, 3))
    a = write_synthetic_las(pointcloud / "a.las", xyz)
    shutil.copy2(a, pointcloud / "b.laz")  # two point-cloud candidates
    with pytest.raises(FolderWorkspaceError) as exc:
        discover_workspace(workspace)
    assert "只允许一个源点云" in str(exc.value)


def test_missing_workspace_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FolderWorkspaceError) as exc:
        discover_workspace(tmp_path / "does-not-exist")
    assert "工作区目录不存在" in str(exc.value)


def test_single_source_creates_layout_and_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    pointcloud = _pointcloud(workspace)
    pointcloud.mkdir(parents=True, exist_ok=True)
    import numpy as np

    rng = np.random.default_rng(5)
    xyz = rng.uniform([0, 0, 0], [100, 100, 30], (500, 3))
    source = write_synthetic_las(pointcloud / "source.las", xyz)
    ws = discover_workspace(workspace)
    assert ws.source == source
    assert ws.source_sha256 == __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    assert ws.source_id.startswith("folder_source_")
    for directory in (
        ws.source_cache_dir,
        ws.span_selection_dir,
        ws.tree_analysis_dir,
        ws.export_dir,
    ):
        assert directory.is_dir()
    assert ws.excel_path.name == "Prediction_Input.xlsx"
    assert ws.excel_path.parent == ws.tree_analysis_dir


def test_unsupported_extension_ignored(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    pointcloud = _pointcloud(workspace)
    pointcloud.mkdir(parents=True, exist_ok=True)
    (pointcloud / "notes.txt").write_text("not a cloud", encoding="utf-8")
    with pytest.raises(FolderWorkspaceError) as exc:
        discover_workspace(workspace)
    assert "未找到点云文件" in str(exc.value)
