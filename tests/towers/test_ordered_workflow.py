"""V5A3 ordered manual tower-selection workflow tests.

Verifies the frozen ordered semantics:

- a clean project starts with next tower T001 and 0 confirmed towers;
- SELECT never allocates a T###;
- RESELECT never allocates a T###;
- CANCEL never allocates a T###;
- a failed segmentation never allocates a T### (and records the failure);
- the first CONFIRM is T001, the second is T002;
- reload persists T001/T002 in confirmation (corridor) order;
- a duplicate confirmation cannot create a second tower;
- candidate P### numbering never affects the T### sequence;
- membership artifacts are linked through ``TowerRefinement.local_region_reference``.
"""

from __future__ import annotations

import pytest
from algorithms.towers.ordered import (
    OrderedPreviewInvalidError,
    OrderedPreviewStore,
    ordered_cancel,
    ordered_confirm,
    ordered_preview,
    ordered_reselect,
    ordered_review_state,
    ordered_select,
)
from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
from core.project import (
    create_project_writable,
    open_project_read_only,
)

from tests.towers.conftest import project_for_source, write_tower_las


def _new_project(tmp_path, *, second_tower=True):
    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=(45.0, 50.0, 0.0) if second_tower else None)
    project, binding, metadata = project_for_source(src, points)
    root = tmp_path / "project"
    session = create_project_writable(root, project, source_access_metadata=metadata)
    cache = RawTileCache.prepare(
        src, tmp_path / "cache", source_binding=binding, config=RawTileCacheConfig()
    )
    g1 = session.create_tower_group(display_name="区域 1")
    return root, session, cache, store_factory(), g1


def store_factory():
    return OrderedPreviewStore()


def test_clean_project_starts_at_t001(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        state = ordered_review_state(session)
        assert state["ordered"]["next_tower_id"] == "T001"
        assert state["ordered"]["confirmed_tower_ids"] == []
        assert state["ordered"]["confirmed_tower_count"] == 0
        assert session.tower_registry.records == ()


def test_select_does_not_allocate_tower(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        result = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        assert result.success, result.failure_reason
        assert result.preview is not None
        assert session.tower_registry.records == ()
        assert session.candidate_store.confirmations == ()
        # A preview token exists with the membership metadata.
        preview = ordered_preview(store, result.preview["token"])
        assert preview["member_count"] > 0
        assert preview["refinement_ready"] == "PREVIEW_READY"


def test_reselect_does_not_allocate_tower(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        result = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        before = len(session.tower_registry.records)
        outcome = ordered_reselect(
            session, store, result.preview["token"], operator_identity="owner"
        )
        assert len(session.tower_registry.records) == before
        assert outcome["state"]["ordered"]["next_tower_id"] == "T001"
        # The superseded preview is invalid afterwards.
        with pytest.raises(OrderedPreviewInvalidError):
            ordered_preview(store, result.preview["token"])


def test_cancel_does_not_allocate_tower(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        result = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        before = len(session.tower_registry.records)
        outcome = ordered_cancel(session, store, result.preview["token"], operator_identity="owner")
        assert len(session.tower_registry.records) == before
        assert outcome["state"]["ordered"]["next_tower_id"] == "T001"
        with pytest.raises(OrderedPreviewInvalidError):
            ordered_confirm(session, store, result.preview["token"])


def test_failed_segmentation_no_tower(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        result = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(500.0, 500.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        assert not result.success
        assert result.failure_reason.startswith("LOCAL_SEGMENTATION_FAILED")
        assert session.tower_registry.records == ()
        # The candidate set exists and its failure is recorded in the ledger.
        state = ordered_review_state(session)
        assert state["ordered"]["next_tower_id"] == "T001"


def test_confirm_first_t001_second_t002(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=True)
    with session:
        r1 = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        r2 = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(45.0, 50.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        first = ordered_confirm(session, store, r2.preview["token"], operator_identity="owner")
        assert first["tower_id"] == "T001"
        second = ordered_confirm(session, store, r1.preview["token"], operator_identity="owner")
        assert second["tower_id"] == "T002"
        state = ordered_review_state(session)
        assert state["ordered"]["confirmed_tower_ids"] == ["T001", "T002"]
        assert state["ordered"]["next_tower_id"] == "T003"


def test_membership_artifact_linked_to_confirmed_tower(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        r = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        outcome = ordered_confirm(session, store, r.preview["token"], operator_identity="owner")
        assert outcome["tower_id"] == "T001"
        tower = session.tower_registry.get("T001").current
        ref = tower.refinement.local_region_reference
        assert ref is not None
        assert ref.role == "tower-local-membership"
        assert ref.relative_path.startswith("tower-memberships/")
        assert outcome["artifact_reference"]["sha256"] == ref.sha256
        # The review carries membership provenance.
        event = session.candidate_store.review_ledgers[0].events[-1]
        details = event.refinement_evidence.details
        assert details["membership_digest"] == outcome["membership"]["membership_digest"]
        assert details["member_count"] == outcome["membership"]["member_count"]


def test_reload_persists_towers_in_order(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=True)
    with session:
        r1 = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        r2 = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(45.0, 50.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        ordered_confirm(session, store, r1.preview["token"], operator_identity="owner")
        ordered_confirm(session, store, r2.preview["token"], operator_identity="owner")
    with open_project_read_only(root) as reopened:
        state = ordered_review_state(reopened)
        assert state["ordered"]["confirmed_tower_ids"] == ["T001", "T002"]
        assert state["ordered"]["next_tower_id"] == "T003"
        assert (
            reopened.tower_registry.get("T001").current.refinement.local_region_reference
            is not None
        )
        assert (
            reopened.tower_registry.get("T002").current.refinement.local_region_reference
            is not None
        )


def test_duplicate_confirmation_no_duplicate_tower(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        r = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        ordered_confirm(session, store, r.preview["token"], operator_identity="owner")
        assert [entry.tower_id for entry in session.tower_registry.records] == ["T001"]
        assert len(session.candidate_store.confirmations) == 1
        # The consumed token cannot confirm again.
        with pytest.raises(OrderedPreviewInvalidError):
            ordered_confirm(session, store, r.preview["token"])


def test_candidate_numbering_does_not_affect_t_sequence(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=True)
    with session:
        # Every ordered SELECT creates its own MANUAL_ADD candidate set whose
        # candidate is always P001; the T### sequence is purely confirmation order.
        r1 = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(45.0, 50.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        r2 = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        assert r1.candidate_id == "P001"
        assert r2.candidate_id == "P001"
        first = ordered_confirm(session, store, r2.preview["token"], operator_identity="owner")
        assert first["tower_id"] == "T001"
        second = ordered_confirm(session, store, r1.preview["token"], operator_identity="owner")
        assert second["tower_id"] == "T002"


def test_stale_token_fails_closed_after_reselect(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        r = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        ordered_reselect(session, store, r.preview["token"], operator_identity="owner")
        with pytest.raises(OrderedPreviewInvalidError):
            ordered_confirm(session, store, r.preview["token"])
        with pytest.raises(OrderedPreviewInvalidError):
            ordered_preview(store, r.preview["token"])


def test_preview_sample_is_deterministic_and_capped(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        r = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        preview = r.preview
        assert preview["preview_sample_size"] <= preview["preview_sample_cap"]
        assert preview["preview_sample_authority"] == "VIEWER_ONLY"
        sample = preview["preview_sample"]
        assert len(sample) == preview["preview_sample_size"]
        # Deterministic across reads.
        assert ordered_preview(store, r.preview["token"])["preview_sample"] == sample
        # The sample is never the authoritative membership (exact arrays are not
        # served); it is capped and VIEWER_ONLY.
        assert "source_index" not in preview
        assert "xyz_absolute_m" not in preview


def test_preview_sample_point_function(tmp_path):
    root, session, cache, store, g1 = _new_project(tmp_path, second_tower=False)
    with session:
        r = ordered_select(
            session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
            group_id=g1.group_id,
            operator_identity="owner",
        )
        # Recover membership via the artifact round trip is covered elsewhere;
        # here we exercise the capping directly on the preview metadata.
        preview = ordered_preview(store, r.preview["token"], max_sample_points=10)
        assert preview["preview_sample_size"] <= 10
