"""V5A3 ordered CONFIRM atomicity tests.

Simulates a failure during the confirming generation publication and verifies
that after the failed publication there is:

- no active T###;
- no active confirmation;
- no active membership reference (the content-addressed artifact is an orphan
  with no immutable generation referencing it);

and that a retry remains valid and produces exactly one T###.
"""

from __future__ import annotations

import core.project.tower_candidates as tower_candidates_mod
import pytest
from algorithms.towers.ordered import (
    OrderedPreviewStore,
    ordered_confirm,
    ordered_select,
)
from algorithms.towers.raw_access import RawTileCache, RawTileCacheConfig
from core.project import create_project_writable

from tests.towers.conftest import project_for_source, write_tower_las


def _setup(tmp_path):
    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=None)
    project, binding, metadata = project_for_source(src, points)
    root = tmp_path / "project"
    session = create_project_writable(root, project, source_access_metadata=metadata)
    cache = RawTileCache.prepare(
        src, tmp_path / "cache", source_binding=binding, config=RawTileCacheConfig()
    )
    store = OrderedPreviewStore()
    g1 = session.create_tower_group(display_name="区域 1")
    r = ordered_select(
        session, cache, store, candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
        group_id=g1.group_id,
        operator_identity="owner",
    )
    assert r.success, r.failure_reason
    return root, session, store, r.preview["token"]


def test_failed_publication_leaves_no_active_state_then_retry_succeeds(tmp_path, monkeypatch):
    root, session, store, token = _setup(tmp_path)

    real_confirm = tower_candidates_mod.confirm_reviewed_tower_candidate

    def boom(*args, **kwargs):
        raise RuntimeError("simulated generation publication failure")

    monkeypatch.setattr(tower_candidates_mod, "confirm_reviewed_tower_candidate", boom)
    with session:
        with pytest.raises(RuntimeError, match="simulated"):
            ordered_confirm(session, store, token, operator_identity="owner")

        # No active T###, no active confirmation.
        assert session.tower_registry.records == ()
        assert session.candidate_store.confirmations == ()

        # The exact membership artifact exists on disk (content-addressed) but no
        # tower references it, so there is no ACTIVE membership reference.
        memberships_dir = (session.project_root / "tower-memberships")
        artifact_files = sorted(memberships_dir.glob("*.npz")) if memberships_dir.exists() else []
        assert len(artifact_files) == 1, "membership artifact written before publication"
        assert session.tower_registry.records == ()

        # Retry remains valid once publication succeeds.
        monkeypatch.setattr(tower_candidates_mod, "confirm_reviewed_tower_candidate", real_confirm)
        outcome = ordered_confirm(session, store, token, operator_identity="owner")
        assert outcome["tower_id"] == "T001"
        assert [entry.tower_id for entry in session.tower_registry.records] == ["T001"]
        assert len(session.candidate_store.confirmations) == 1
        tower = session.tower_registry.get("T001").current
        assert tower.refinement.local_region_reference is not None
        # The retried tower references the SAME content-addressed artifact.
        assert (
            tower.refinement.local_region_reference.relative_path
            == "tower-memberships/" + artifact_files[0].name
        )
