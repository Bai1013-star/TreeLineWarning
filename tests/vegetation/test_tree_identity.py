"""Persistent identity safety, lineage, and transaction tests."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
from algorithms.spans import tree_identity
from algorithms.spans.reconcile import GlobalTreeRecord, TreeReconciliationResult
from algorithms.spans.tree_identity import (
    TreeIdentityError,
    assign_persistent_tree_ids,
    finalize_tree_identity,
    finalize_tree_identity_terminal,
    prepare_tree_identity,
    recover_tree_identity_terminal_transaction,
)
from core.config.fast_baseline import default_fast_baseline_config


def _result(groups: list[list[int]], *, span_id: int = 4) -> TreeReconciliationResult:
    original = np.asarray([value for group in groups for value in group], dtype=np.int64)
    identifiers = np.asarray(
        [gid for gid, group in enumerate(groups) for _ in group], dtype=np.int32
    )
    records = tuple(
        GlobalTreeRecord(
            global_tree_id=gid,
            line_id=2,
            span_id=span_id,
            span_name=f"S{span_id:03d}",
            external_name=f"G002-Span{span_id:03d}-Tree{gid + 1:03d}",
            primary_span_tree_id=gid,
            apex_xy_local_m=(float(gid), 0.0),
            apex_height_m=10.0,
            full_resolution_point_count=len(group),
            core_point_count=len(group),
            halo_point_count=0,
            core_fraction=1.0,
            review=False,
            merged_span_tree_keys=((0, gid),),
            span_order_index=gid + 1,
        )
        for gid, group in enumerate(groups)
    )
    return TreeReconciliationResult(
        records, original, np.zeros((len(original), 3)), identifiers, (), {}, {}
    )


def _paths(root: Path, run_id: str) -> tuple[Path, Path]:
    return root / "tree_identity_registry.json", root / run_id / "tree_identity_membership.npz"


def _assign(root: Path, result: TreeReconciliationResult, run_id: str) -> TreeReconciliationResult:
    registry, membership = _paths(root, run_id)
    return assign_persistent_tree_ids(
        result,
        default_fast_baseline_config(),
        registry_path=registry,
        membership_path=membership,
        group_id="G002",
        source_key="same-source",
        run_id=run_id,
    )


def _registry(root: Path) -> dict:
    return json.loads((root / "tree_identity_registry.json").read_text("utf-8"))


def _concurrent_assign(root_text: str, members: list[int], run_id: str, queue) -> None:
    """Spawn-safe worker: two processes must serialize registry allocation."""

    result = _assign(Path(root_text), _result([members]), run_id)
    queue.put(result.global_tree_records[0].persistent_tree_id)


def test_same_source_rerun_preserves_ids_and_new_tree_does_not_renumber(tmp_path: Path) -> None:
    first = _assign(tmp_path, _result([[1, 2, 3], [10, 11, 12]]), "run-a")
    assert [record.persistent_tree_id for record in first.global_tree_records] == [
        "G002-T001",
        "G002-T002",
    ]
    second = _assign(
        tmp_path, _result([[1, 2, 3], [6, 7, 8], [10, 11, 12]]), "run-b"
    )
    assert [record.persistent_tree_id for record in second.global_tree_records] == [
        "G002-T001",
        "G002-T003",
        "G002-T002",
    ]
    ownership_changed = _assign(
        tmp_path,
        _result([[1, 2, 3], [6, 7, 8], [10, 11, 12]], span_id=19),
        "run-c",
    )
    assert [record.persistent_tree_id for record in ownership_changed.global_tree_records] == [
        "G002-T001",
        "G002-T003",
        "G002-T002",
    ]


def test_partition_split_supersedes_parent_and_assigns_children(tmp_path: Path) -> None:
    original = _assign(tmp_path, _result([[1, 2, 3, 4, 5, 6]]), "run-a")
    assert original.global_tree_records[0].persistent_tree_id == "G002-T001"
    split = _assign(tmp_path, _result([[1, 2, 3], [4, 5, 6]]), "run-b")
    assert [record.persistent_tree_id for record in split.global_tree_records] == [
        "G002-T002",
        "G002-T003",
    ]
    assert {record.identity_state for record in split.global_tree_records} == {
        "SPLIT_CHILD_NEW_ENTITY"
    }
    assert all(record.candidate_persistent_tree_ids == () for record in split.global_tree_records)
    registry = _registry(tmp_path)
    assert registry["entities"]["G002-T001"]["current_status"] == "SUPERSEDED_BY_SPLIT"
    assert registry["entities"]["G002-T002"]["lineage"] == [
        {"event": "SPLIT_FROM", "persistent_tree_id": "G002-T001"}
    ]
    assert registry["next_sequence_by_group"]["G002"] == 4


def test_ambiguous_merge_is_unbound_and_rerun_does_not_consume_ids(tmp_path: Path) -> None:
    _assign(tmp_path, _result([[1, 2, 3], [4, 5, 6]]), "run-a")
    merged = _assign(tmp_path, _result([[1, 2, 3, 4, 5, 6]]), "run-b")
    record = merged.global_tree_records[0]
    assert record.persistent_tree_id is None
    assert record.identity_state == "MERGE_AMBIGUOUS_REVIEW"
    assert record.candidate_persistent_tree_ids == ("G002-T001", "G002-T002")
    before = _registry(tmp_path)
    repeated = _assign(tmp_path, _result([[1, 2, 3, 4, 5, 6]]), "run-c")
    assert repeated.global_tree_records[0].persistent_tree_id is None
    assert (
        repeated.global_tree_records[0].candidate_persistent_tree_ids
        == record.candidate_persistent_tree_ids
    )
    after = _registry(tmp_path)
    assert after["next_sequence_by_group"] == before["next_sequence_by_group"]
    assert set(after["entities"]) == {"G002-T001", "G002-T002"}


@pytest.mark.parametrize("kind", ["missing", "outside", "malformed", "schema"])
def test_invalid_historical_membership_fails_closed_without_registry_change(
    tmp_path: Path, kind: str
) -> None:
    _assign(tmp_path, _result([[1, 2, 3]]), "run-a")
    registry_path = tmp_path / "tree_identity_registry.json"
    before = registry_path.read_bytes()
    registry = json.loads(before)
    entity = registry["entities"]["G002-T001"]
    membership_path = tmp_path / entity["last_membership_reference"]
    if kind == "missing":
        membership_path.unlink()
    elif kind == "outside":
        entity["last_membership_reference"] = "../outside.npz"
        registry_path.write_text(json.dumps(registry), "utf-8")
        before = registry_path.read_bytes()
    elif kind == "malformed":
        membership_path.write_bytes(b"not an npz")
    else:
        np.savez(
            membership_path,
            schema=np.asarray("wrong-schema"),
            source_key=np.asarray("same-source"),
            group_id=np.asarray("G002"),
            original_index=np.asarray([1, 2, 3], dtype=np.int64),
            global_tree_id=np.asarray([0, 0, 0], dtype=np.int32),
        )
    with pytest.raises(TreeIdentityError):
        _assign(tmp_path, _result([[1, 2, 3], [8, 9, 10]]), "run-b")
    assert registry_path.read_bytes() == before


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("{not json", "REGISTRY"),
        (
            json.dumps(
                {
                    "schema": "unsupported-schema",
                    "next_sequence_by_group": {},
                    "entities": {},
                }
            ),
            "REGISTRY_SCHEMA_INVALID",
        ),
        (
            json.dumps(
                {
                    "schema": "treeline-persistent-tree-registry-v1",
                    "next_sequence_by_group": {"G002": 0},
                    "entities": {},
                }
            ),
            "REGISTRY_SEQUENCE_INVALID",
        ),
        (
            json.dumps(
                {
                    "schema": "treeline-persistent-tree-registry-v1",
                    "next_sequence_by_group": {"G002": 2},
                    "entities": {
                        "G002-T001": {
                            "persistent_tree_id": "G002-T001",
                            "group_id": "G002",
                            "source_key": "same-source",
                            "lineage": [
                                {
                                    "event": "SPLIT_FROM",
                                    "persistent_tree_id": "G002-T404",
                                }
                            ],
                        }
                    },
                }
            ),
            "REGISTRY_LINEAGE_INVALID",
        ),
    ],
)
def test_corrupt_registry_fails_closed(tmp_path: Path, corrupt: str, message: str) -> None:
    registry_path = tmp_path / "tree_identity_registry.json"
    registry_path.write_text(corrupt, "utf-8")
    with pytest.raises(TreeIdentityError, match=message):
        _assign(tmp_path, _result([[1, 2, 3]]), "run-a")
    assert registry_path.read_text("utf-8") == corrupt


@pytest.mark.parametrize(
    "corrupt",
    [
        '{"schema":"treeline-persistent-tree-registry-v1",'
        '"schema":"treeline-persistent-tree-registry-v1",'
        '"next_sequence_by_group":{},"entities":{}}',
        '{"schema":"treeline-persistent-tree-registry-v1",'
        '"next_sequence_by_group":{},"entities":{"G002-T001":{},"G002-T001":{}}}',
        '{"schema":"treeline-persistent-tree-registry-v1",'
        '"next_sequence_by_group":{},"entities":{"G002-T001":'
        '{"persistent_tree_id":"G002-T001","group_id":"G002",'
        '"source_key":"same-source","lineage":[{"event":"SPLIT_FROM",'
        '"event":"SPLIT_FROM","persistent_tree_id":"G002-T001"}]}}}',
    ],
)
def test_duplicate_json_keys_fail_closed_without_registry_mutation(
    tmp_path: Path, corrupt: str
) -> None:
    registry_path = tmp_path / "tree_identity_registry.json"
    registry_path.write_text(corrupt, "utf-8")
    before = registry_path.read_bytes()

    with pytest.raises(TreeIdentityError, match="DUPLICATE_JSON_KEY"):
        _assign(tmp_path, _result([[1, 2, 3]]), "run-a")

    assert registry_path.read_bytes() == before


@pytest.mark.parametrize("failed_terminal_stage", ["index", "attachment", "completion"])
def test_terminal_failure_does_not_consume_identity_sequence(
    tmp_path: Path, failed_terminal_stage: str
) -> None:
    _assign(tmp_path, _result([[1, 2, 3]]), "run-a")
    registry_path, membership_path = _paths(tmp_path, "run-b")
    prepared = prepare_tree_identity(
        _result([[8, 9, 10]]),
        default_fast_baseline_config(),
        registry_path=registry_path,
        membership_path=membership_path,
        group_id="G002",
        source_key="same-source",
        run_id="run-b",
    )
    before = registry_path.read_bytes()

    def fail_terminal(_assigned):
        raise RuntimeError(f"{failed_terminal_stage} failure")

    with pytest.raises(RuntimeError, match=failed_terminal_stage):
        finalize_tree_identity_terminal(
            prepared,
            publish_terminal=fail_terminal,
            is_run_complete=lambda _run_id: False,
        )
    assert registry_path.read_bytes() == before
    assert not (tmp_path / "tree_identity_registry.json.pending.json").exists()

    completed = {"value": False}

    def complete_terminal(_assigned):
        completed["value"] = True
        return "complete"

    assigned, published = finalize_tree_identity_terminal(
        prepared,
        publish_terminal=complete_terminal,
        is_run_complete=lambda _run_id: completed["value"],
    )
    assert published == "complete"
    assert assigned.global_tree_records[0].persistent_tree_id == "G002-T002"


def test_terminal_transaction_recovery_rolls_back_incomplete_and_rolls_forward_complete(
    tmp_path: Path,
) -> None:
    registry_path, membership_path = _paths(tmp_path, "run-a")
    prepared = prepare_tree_identity(
        _result([[1, 2, 3]]),
        default_fast_baseline_config(),
        registry_path=registry_path,
        membership_path=membership_path,
        group_id="G002",
        source_key="same-source",
        run_id="run-a",
    )
    baseline = tree_identity._read_registry(registry_path)
    proposed = tree_identity.deepcopy(baseline)
    assigned = tree_identity._assign_from_registry(
        prepared.result,
        prepared.config,
        proposed,
        registry_path=registry_path,
        membership_path=membership_path,
        group_id=prepared.group_id,
        source_key=prepared.source_key,
        run_id=prepared.run_id,
    )
    assert assigned.global_tree_records[0].persistent_tree_id == "G002-T001"
    journal = tree_identity._transaction_path(registry_path)
    tree_identity._write_json_atomic(
        journal,
        {
            "schema": tree_identity.TREE_IDENTITY_TRANSACTION_SCHEMA,
            "run_id": "run-a",
            "baseline_registry_digest": tree_identity._registry_digest(baseline),
            "baseline_registry": baseline,
            "proposed_registry_digest": tree_identity._registry_digest(proposed),
            "proposed_registry": proposed,
        },
    )
    assert recover_tree_identity_terminal_transaction(
        registry_path,
        prepared.config,
        is_run_complete=lambda _run_id: False,
    )
    assert tree_identity._read_registry(registry_path) == baseline
    assert not journal.exists()

    tree_identity._write_json_atomic(
        journal,
        {
            "schema": tree_identity.TREE_IDENTITY_TRANSACTION_SCHEMA,
            "run_id": "run-a",
            "baseline_registry_digest": tree_identity._registry_digest(baseline),
            "baseline_registry": baseline,
            "proposed_registry_digest": tree_identity._registry_digest(proposed),
            "proposed_registry": proposed,
        },
    )
    assert recover_tree_identity_terminal_transaction(
        registry_path,
        prepared.config,
        is_run_complete=lambda _run_id: True,
    )
    assert _registry(tmp_path) == proposed
    assert not journal.exists()


def test_two_processes_allocate_unique_ids_without_lost_update(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    for round_id in range(2):
        root = tmp_path / f"round-{round_id}"
        queue = context.Queue()
        first = context.Process(
            target=_concurrent_assign, args=(str(root), [1, 2, 3], "run-a", queue)
        )
        second = context.Process(
            target=_concurrent_assign, args=(str(root), [10, 11, 12], "run-b", queue)
        )
        first.start()
        second.start()
        first.join(30)
        second.join(30)
        assert first.exitcode == 0
        assert second.exitcode == 0
        assigned = {queue.get(timeout=5), queue.get(timeout=5)}
        assert assigned == {"G002-T001", "G002-T002"}
        registry = _registry(root)
        assert set(registry["entities"]) == assigned
        assert registry["next_sequence_by_group"]["G002"] == 3


def test_publish_failures_do_not_consume_identity_sequence(tmp_path: Path, monkeypatch) -> None:
    _assign(tmp_path, _result([[1, 2, 3]]), "run-a")
    registry_path, membership_path = _paths(tmp_path, "run-b")
    prepared = prepare_tree_identity(
        _result([[8, 9, 10]]),
        default_fast_baseline_config(),
        registry_path=registry_path,
        membership_path=membership_path,
        group_id="G002",
        source_key="same-source",
        run_id="run-b",
    )
    before = registry_path.read_bytes()
    def fail_publish(_result):
        raise OSError("artifact failure")

    with pytest.raises(OSError):
        finalize_tree_identity(prepared, publish=fail_publish)
    assert registry_path.read_bytes() == before
    assert not membership_path.exists()

    def fail_registry_write(_path, _payload):
        raise TreeIdentityError("TREE_IDENTITY_REGISTRY_WRITE_FAILED")

    monkeypatch.setattr(tree_identity, "_write_json_atomic", fail_registry_write)
    with pytest.raises(TreeIdentityError, match="WRITE_FAILED"):
        finalize_tree_identity(prepared, publish=lambda _result: "published")
    assert registry_path.read_bytes() == before
    monkeypatch.undo()
    successful = _assign(tmp_path, _result([[8, 9, 10]]), "run-c")
    assert successful.global_tree_records[0].persistent_tree_id == "G002-T002"
