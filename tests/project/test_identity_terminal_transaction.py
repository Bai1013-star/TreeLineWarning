"""Terminal identity commit boundary tests using a temporary project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algorithms.spans import project_orchestrator
from algorithms.spans.clearance import ClearanceCancelled
from algorithms.spans.project_adapter import (
    SourceSharedPreparation,
    SpanScienceResult,
    run_canonical_span_science,
)
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from algorithms.spans.tree_identity import prepare_tree_identity
from core.config.fast_baseline import default_fast_baseline_config
from core.contracts.project import AnalysisStatus
from core.project import create_project_writable

from tests.contracts.test_project_contracts import _tower


def _identity_result() -> TreeReconciliationResult:
    record = GlobalTreeRecord(
        global_tree_id=0,
        line_id=1,
        span_id=0,
        span_name="S001",
        external_name="G001-Span001-Tree001",
        primary_span_tree_id=0,
        apex_xy_local_m=(0.0, 0.0),
        apex_height_m=10.0,
        full_resolution_point_count=3,
        core_point_count=3,
        halo_point_count=0,
        core_fraction=1.0,
        review=False,
        merged_span_tree_keys=((0, 0),),
    )
    return TreeReconciliationResult(
        (record,),
        np.asarray([1, 2, 3], dtype=np.int64),
        np.zeros((3, 3), dtype=np.float64),
        np.zeros(3, dtype=np.int32),
        (),
        {},
        {},
    )


def _install_terminal_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    def prepare(_root, input_data):
        return SourceSharedPreparation(
            "a" * 64,
            input_data.science_configuration_digest,
            "unused-preparation.json",
        )

    def execute(input_data, _preparation, output_root: Path, *, executor=None):
        prepared = prepare_tree_identity(
            _identity_result(),
            default_fast_baseline_config(),
            registry_path=output_root.parents[5] / "tree_identity_registry.json",
            membership_path=output_root / "tree_identity_membership.npz",
            group_id=input_data.group_id,
            source_key="temporary-project-source",
            run_id=output_root.name,
        )

        def terminal(assigned):
            for name in ("wire.json", "tree.json", "clearance.json"):
                (output_root / name).write_text("{}", encoding="utf-8")
            return SpanScienceResult(
                tree_count=assigned.final_tree_count,
                minimum_clearance_m=3.0,
                risk_threshold_m=2.0,
                risk_counts={"trees_evaluated": 1, "below_threshold": 0},
                artifact_files={
                    "wire": str(output_root / "wire.json"),
                    "tree": str(output_root / "tree.json"),
                    "clearance": str(output_root / "clearance.json"),
                },
            )

        return SpanScienceResult(
            tree_count=1,
            minimum_clearance_m=None,
            risk_threshold_m=2.0,
            risk_counts={},
            identity_terminal_preparation=prepared,
            identity_terminal_publish=terminal,
        )

    monkeypatch.setattr(project_orchestrator, "prepare_source_shared", prepare)
    monkeypatch.setattr(project_orchestrator, "run_canonical_span_science", execute)


def _temporary_project(tmp_path: Path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _source = tiny_las_project(root)
    session = create_project_writable(root, project, source_access_metadata=metadata)
    binding = project.source_reference.binding
    session.create_tower(_tower("T001", binding, 0.0))
    session.create_tower(_tower("T002", binding, 10.0))
    session.create_tower_group(display_name="test group")
    session.append_confirmed_tower_to_group(group_id="G001", tower_id="T001")
    session.append_confirmed_tower_to_group(group_id="G001", tower_id="T002")
    return root, session


@pytest.mark.parametrize("failure", ["index", "attachment", "completion"])
def test_terminal_stage_failure_does_not_commit_persistent_registry(
    tmp_path: Path,
    tiny_las_project,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root, session = _temporary_project(tmp_path, tiny_las_project)
    _install_terminal_executor(monkeypatch)
    if failure == "index":
        monkeypatch.setattr(
            project_orchestrator,
            "publish_terminal_artifact_index",
            lambda *_args: (_ for _ in ()).throw(OSError("index failure")),
        )
    elif failure == "attachment":
        monkeypatch.setattr(
            project_orchestrator,
            "attach_span_artifact_index",
            lambda *_args: (_ for _ in ()).throw(OSError("attachment failure")),
        )
    else:
        monkeypatch.setattr(
            project_orchestrator,
            "complete_run",
            lambda *_args: (_ for _ in ()).throw(OSError("completion failure")),
        )
    try:
        outcome = project_orchestrator.ProjectSpanOrchestrator().analyze_span(session, "S001")
        assert outcome.status == "FAILED"
        entry = next(item for item in session.run_index.entries if item.run_id == outcome.run_id)
        assert entry.status is AnalysisStatus.FAILED
        assert not (root / "tree_identity_registry.json").exists()
        assert not (root / "tree_identity_registry.json.pending.json").exists()
    finally:
        session.close()


def test_terminal_completion_commits_matching_registry(
    tmp_path: Path, tiny_las_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, session = _temporary_project(tmp_path, tiny_las_project)
    _install_terminal_executor(monkeypatch)
    try:
        outcome = project_orchestrator.ProjectSpanOrchestrator().analyze_span(session, "S001")
        assert outcome.error is None
        assert outcome.status == "COMPLETE"
        assert (root / "tree_identity_registry.json").is_file()
        assert not (root / "tree_identity_registry.json.pending.json").exists()
    finally:
        session.close()


def test_clearance_cancellation_interrupts_run_without_identity_or_terminal_artifacts(
    tmp_path: Path, tiny_las_project, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, session = _temporary_project(tmp_path, tiny_las_project)
    _install_terminal_executor(monkeypatch)
    try:
        good = project_orchestrator.ProjectSpanOrchestrator().analyze_span(session, "S001")
        assert good.status == "COMPLETE"
        registry_before = (root / "tree_identity_registry.json").read_bytes()
        # Use the real adapter wrapper so the custom science function follows
        # exactly the cancellation route used by the tracked executor.
        monkeypatch.setattr(
            project_orchestrator, "run_canonical_span_science", run_canonical_span_science
        )

        def cancelled(_input, _preparation, _output_root):
            raise ClearanceCancelled("CLEARANCE_CANCELLED")

        interrupted = project_orchestrator.ProjectSpanOrchestrator(
            science_executor=cancelled
        ).analyze_span(
            session,
            "S001",
            force_new_run=True,
            cancellation_requested=lambda: True,
        )

        assert interrupted.status == "INTERRUPTED"
        entry = next(
            item for item in session.run_index.entries if item.run_id == interrupted.run_id
        )
        assert entry.status is AnalysisStatus.INTERRUPTED
        assert (root / "tree_identity_registry.json").read_bytes() == registry_before
        assert not (root / "tree_identity_registry.json.pending.json").exists()
        assert not (
            root
            / "artifacts"
            / "project-runs"
            / "groups"
            / "G001"
            / "S001"
            / interrupted.run_id
            / "clearance.json"
        ).exists()
        assert good.run_id != interrupted.run_id
    finally:
        session.close()
