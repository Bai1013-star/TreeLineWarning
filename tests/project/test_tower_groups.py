"""V5A3 tower-group model + persistence + bootstrap tests.

Covers the frozen group semantics:

- default G001 / 区域 1 for a new project;
- G002 creation with monotonic, never-reused IDs;
- canonical G### IDs and display-name validation;
- global Tower uniqueness and exclusive Tower membership;
- ordered membership persistence and reload;
- malformed group graphs fail closed;
- legacy (group-less) bootstrap to G001 with confirmed towers in order;
- the future adjacency projection exposes no implicit cross-group adjacency.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core.contracts.tower_groups import (
    TOWER_GROUP_REGISTRY_SCHEMA_VERSION,
    TOWER_GROUP_SCHEMA_VERSION,
    TowerGroup,
    TowerGroupRegistry,
    group_display_id_for_sequence,
)
from core.project import (
    create_project_writable,
    open_project_read_only,
    open_project_writable,
)

from tests.towers.conftest import project_for_source, write_tower_las


def _project(tmp_path: Path, second_tower: bool = True):
    src = tmp_path / "src.las"
    points = write_tower_las(src, extra_tower_xyz=(45.0, 50.0, 0.0) if second_tower else None)
    project, binding, metadata = project_for_source(src, points)
    root = tmp_path / "project"
    session = create_project_writable(root, project, source_access_metadata=metadata)
    return root, session


def _append_tower(session, tower_id: str) -> None:
    """Confirm a tower into the registry via the ordered confirm path (no group)."""
    from core.config.fast_baseline import canonical_digest
    from core.contracts.project import TowerGeometry
    from core.contracts.tower_candidates import (
        CandidateOrigin,
        DetectorKind,
        DetectorProvenance,
        RefinementState,
        ReviewDecision,
        TowerCandidate,
        TowerCandidateQualityFlag,
        TowerCandidateRawRefinementEvidence,
        TowerCandidateSet,
    )
    from core.project.tower_candidates import (
        append_tower_candidate_review,
        confirm_reviewed_tower_candidate,
        persist_tower_candidate_set,
    )

    binding = session.project.source_reference.binding
    candidate = TowerCandidate(
        candidate_id="P001",
        origin=CandidateOrigin.AUTOMATIC,
        coordinate_frame=binding.coordinate_context.frame,
        coordinate_unit=binding.coordinate_context.units,
        candidate_xyz_absolute_m=(10.0, 20.0, 0.0),
        confidence=0.9,
        quality_flags=TowerCandidateQualityFlag.PROPOSAL_NEEDS_REFINEMENT,
        detector_provenance=DetectorProvenance(
            detector_kind=DetectorKind.GEOMETRY,
            detector_name="test",
            detector_version="1",
            configuration_digest=canonical_digest({"a": 1}),
        ),
    )
    candidate_set = TowerCandidateSet.create(
        candidate_set_id=f"auto-{tower_id.lower()}",
        project_id=session.project.project_id,
        generated_at="2026-08-16T00:00:00Z",
        source_binding=binding,
        coordinate_context=binding.coordinate_context,
        detector_provenance=candidate.detector_provenance,
        candidates=[candidate],
    )
    persist_tower_candidate_set(session, candidate_set)
    geometry = TowerGeometry(
        base_xyz_absolute_m=(10.0, 20.0, 0.0),
        top_xyz_absolute_m=(10.0, 20.0, 20.0),
        height_m=20.0,
    )
    evidence = TowerCandidateRawRefinementEvidence(
        method="test",
        algorithm_version="test-v1",
        source_content_identity_digest=binding.source_content_identity_digest,
        source_identity_record_digest=binding.source_identity_record_digest,
        details={},
    )
    event = append_tower_candidate_review(
        session,
        candidate_set_reference=candidate_set.reference,
        candidate_id="P001",
        decision=ReviewDecision.ACCEPTED,
        occurred_at="2026-08-16T00:00:01Z",
        reason="test",
        refinement_state=RefinementState.RAW_REFINED,
        refined_raw_geometry=geometry,
        refinement_evidence=evidence,
    )
    confirm_reviewed_tower_candidate(
        session,
        candidate_set_reference=candidate_set.reference,
        candidate_id="P001",
        accepted_review_revision=event.review_revision,
    )
    assert tower_id in [entry.tower_id for entry in session.tower_registry.records]


def test_group_id_helpers():
    assert group_display_id_for_sequence(1) == "G001"
    assert group_display_id_for_sequence(999) == "G999"
    with pytest.raises(ValueError):
        group_display_id_for_sequence(0)
    with pytest.raises(ValueError):
        group_display_id_for_sequence(1000)


def test_tower_group_validation():
    group = TowerGroup(
        schema_version=TOWER_GROUP_SCHEMA_VERSION,
        group_id="G001",
        display_name="区域 1",
        ordered_tower_ids=("T001", "T002"),
        created_at="2026-08-16T00:00:00Z",
    )
    assert group.ordered_tower_ids == ("T001", "T002")
    with pytest.raises(ValueError):
        TowerGroup(
            schema_version=TOWER_GROUP_SCHEMA_VERSION,
            group_id="G001",
            display_name="区域 1",
            ordered_tower_ids=("T001", "T001"),
            created_at="2026-08-16T00:00:00Z",
        )
    with pytest.raises(ValueError):
        TowerGroup(
            schema_version=TOWER_GROUP_SCHEMA_VERSION,
            group_id="X001",
            display_name="区域 1",
            ordered_tower_ids=(),
            created_at="2026-08-16T00:00:00Z",
        )
    with pytest.raises(ValueError):
        TowerGroup(
            schema_version=TOWER_GROUP_SCHEMA_VERSION,
            group_id="G001",
            display_name="   ",
            ordered_tower_ids=(),
            created_at="2026-08-16T00:00:00Z",
        )


def test_registry_validation_exclusive_membership_and_monotonic_sequence():
    base = {
        "schema_version": TOWER_GROUP_REGISTRY_SCHEMA_VERSION,
        "generation_id": "g-0000000000000001-abcdef01",
        "project_id": "p",
        "next_group_sequence": 3,
    }
    g1 = TowerGroup(
        schema_version=TOWER_GROUP_SCHEMA_VERSION,
        group_id="G001",
        display_name="区域 1",
        ordered_tower_ids=("T001",),
        created_at="2026-08-16T00:00:00Z",
    )
    g2 = TowerGroup(
        schema_version=TOWER_GROUP_SCHEMA_VERSION,
        group_id="G002",
        display_name="道路右侧",
        ordered_tower_ids=("T002",),
        created_at="2026-08-16T00:00:00Z",
    )
    TowerGroupRegistry(**base, groups=(g1, g2))
    # Duplicate group id.
    with pytest.raises(ValueError):
        TowerGroupRegistry(**base, groups=(g1, g1))
    # Tower in two groups.
    dup = TowerGroup(
        schema_version=TOWER_GROUP_SCHEMA_VERSION,
        group_id="G002",
        display_name="道路右侧",
        ordered_tower_ids=("T001",),
        created_at="2026-08-16T00:00:00Z",
    )
    with pytest.raises(ValueError):
        TowerGroupRegistry(**base, groups=(g1, dup))
    # next_group_sequence not above represented groups.
    with pytest.raises(ValueError):
        TowerGroupRegistry(**{**base, "next_group_sequence": 2}, groups=(g1, g2))
    # groups not in sequence order.
    with pytest.raises(ValueError):
        TowerGroupRegistry(**base, groups=(g2, g1))


def test_create_groups_and_persist_reload(tmp_path):
    root, session = _project(tmp_path)
    with session:
        g1 = session.create_tower_group(display_name="区域 1")
        g2 = session.create_tower_group(display_name="道路右侧")
        assert g1.group_id == "G001"
        assert g2.group_id == "G002"
        assert session.tower_group_registry.next_group_sequence == 3
    with open_project_read_only(root) as reopened:
        registry = reopened.tower_group_registry
        assert [g.group_id for g in registry.groups] == ["G001", "G002"]
        assert registry.get("G001").display_name == "区域 1"
        assert registry.get("G002").display_name == "道路右侧"
        assert registry.next_group_sequence == 3
        assert registry.bootstrap_pending is False


def test_append_confirmed_tower_to_group_and_reload(tmp_path):
    root, session = _project(tmp_path)
    with session:
        session.create_tower_group(display_name="区域 1")
        _append_tower(session, "T001")
        updated = session.append_confirmed_tower_to_group(group_id="G001", tower_id="T001")
        assert updated.ordered_tower_ids == ("T001",)
        # Exclusive membership: cannot append T001 to G002.
        session.create_tower_group(display_name="道路右侧")
        with pytest.raises(ValueError):
            session.append_confirmed_tower_to_group(group_id="G002", tower_id="T001")
    with open_project_read_only(root) as reopened:
        assert reopened.tower_group_registry.get("G001").ordered_tower_ids == ("T001",)


def test_malformed_group_graph_fails_closed(tmp_path):
    import json

    root, session = _project(tmp_path)
    with session:
        session.create_tower_group(display_name="区域 1")
        _append_tower(session, "T001")
        session.append_confirmed_tower_to_group(group_id="G001", tower_id="T001")
    with open_project_read_only(root) as active:
        generation = active.loaded_project.generation_path
        path = generation / "tower-groups" / "tower-groups.json"
        assert path.is_file()
    # Corrupt the group artifact on disk -> reopen must fail closed because the
    # manifest file record (sha256) no longer matches.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["groups"][0]["ordered_tower_ids"] = ["T999"]
    path.write_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    )
    from core.project.persistence import FileIntegrityError, ProjectIntegrityError

    with pytest.raises((ProjectIntegrityError, FileIntegrityError)):
        open_project_read_only(root)


def _make_legacy(loaded) -> None:
    """Strip the group artifact from a generation to simulate a legacy one.

    Removes ``tower-groups/tower-groups.json`` from disk and from the sealed
    manifest, then fixes the root pointer's generation-manifest digest.  This
    reproduces a generation created BEFORE group persistence (which never
    carried the group artifact).
    """
    import json as _json

    from core.project.persistence import file_sha256, published_json_bytes

    group_dir = loaded.loaded_project.generation_path / "tower-groups"
    if group_dir.exists():
        (group_dir / "tower-groups.json").unlink()
        group_dir.rmdir()
    manifest_path = loaded.loaded_project.generation_path / "generation.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        item
        for item in manifest["files"]
        if item["relative_path"] != "tower-groups/tower-groups.json"
    ]
    manifest_path.write_bytes(published_json_bytes(manifest))
    pointer_path = loaded.project_root / "project.json"
    pointer = _json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(manifest_path.read_bytes())
    pointer_path.write_bytes(published_json_bytes(pointer))


def test_legacy_bootstrap_to_default_group(tmp_path):
    """A legacy group-less project with confirmed towers bootstraps G001 in order."""
    from core.project.tower_groups import DEFAULT_BOOTSTRAP_GROUP_DISPLAY_NAME

    root, session = _project(tmp_path)
    with session:
        _append_tower(session, "T001")
        _append_tower(session, "T002")
        _append_tower(session, "T003")
    # Downgrade the active generation to a legacy group-less form.
    with open_project_read_only(root) as active:
        _make_legacy(active)
    with open_project_read_only(root) as legacy_session:
        assert legacy_session.tower_group_registry.bootstrap_pending is True
    with open_project_writable(root) as session:
        registry = session.ensure_ordered_group()
        group = registry.get("G001")
        assert group.display_name == DEFAULT_BOOTSTRAP_GROUP_DISPLAY_NAME
        assert group.ordered_tower_ids == ("T001", "T002", "T003")
        assert registry.next_group_sequence == 2
    with open_project_read_only(root) as reopened:
        assert reopened.tower_group_registry.bootstrap_pending is False
        assert reopened.tower_group_registry.get("G001").ordered_tower_ids == (
            "T001",
            "T002",
            "T003",
        )


def test_cross_group_non_adjacency_projection():
    """The future adjacency projection exposes NO implicit cross-group adjacency."""
    groups = {
        "G001": ["T001", "T002", "T005"],
        "G002": ["T003", "T004"],
    }

    def adjacency_pairs(ordered):
        return list(zip(ordered, ordered[1:], strict=False))

    assert adjacency_pairs(groups["G001"]) == [("T001", "T002"), ("T002", "T005")]
    assert adjacency_pairs(groups["G002"]) == [("T003", "T004")]
    all_pairs = {pair for g in groups.values() for pair in adjacency_pairs(g)}
    assert ("T002", "T003") not in all_pairs
    assert ("T004", "T005") not in all_pairs
