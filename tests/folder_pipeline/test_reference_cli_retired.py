"""Retired cross-tree DBH input must fail before starting a Folder run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from algorithms.folder_pipeline.orchestrator import FolderOrchestrator
from scripts.run_folder_to_excel import main


def test_retired_reference_argument_fails_before_workspace_discovery(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--workspace", "Z:/does-not-exist", "--reference-dbh-cm", "25"])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "--reference-dbh-cm 已停用" in error
    assert "本树点云" in error
    assert "留空" in error


def test_python_api_rejects_cross_tree_reference_before_run() -> None:
    with pytest.raises(ValueError, match="参考胸径参数已停用"):
        FolderOrchestrator(SimpleNamespace(), reference_dbh_cm=25)
