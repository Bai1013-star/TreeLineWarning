from __future__ import annotations

import pytest

import core.project as project_package
from core.project import (
    ProjectSessionClosedError,
    ProjectSessionMode,
    create_project_writable,
    open_project_read_only,
    open_project_writable,
)
from core.project import persistence
from core.project import session as session_module


def _create_project(root, project, metadata):
    writer = create_project_writable(root, project, source_access_metadata=metadata)
    writer.close()


def test_package_surface_has_no_unlocked_authoritative_creator():
    assert "create_project" not in project_package.__all__
    assert not hasattr(project_package, "create_project")


def test_writable_open_acquires_lock_before_authoritative_pointer_load(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    events: list[str] = []
    original_acquire = session_module.acquire_project_writer_lock
    original_open = session_module.persistence.open_project

    def acquire(*args, **kwargs):
        events.append("lock")
        return original_acquire(*args, **kwargs)

    def open_snapshot(*args, **kwargs):
        events.append("pointer")
        assert events[0] == "lock"
        return original_open(*args, **kwargs)

    monkeypatch.setattr(session_module, "acquire_project_writer_lock", acquire)
    monkeypatch.setattr(session_module.persistence, "open_project", open_snapshot)
    writer = session_module.open_project_writable(root)
    try:
        assert events[0] == "lock"
        assert "pointer" in events
    finally:
        writer.close()


def test_stale_pointer_observation_is_discarded_and_writable_session_reloads_under_lock(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    pre_lock_observation = persistence.open_project(root)

    publisher = open_project_writable(root)
    try:
        newest = persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    finally:
        publisher.close()

    writer = open_project_writable(root)
    try:
        assert pre_lock_observation.generation_id != newest.generation_id
        assert writer.generation_id == newest.generation_id
        assert writer.loaded_project.generation_id == newest.generation_id
    finally:
        writer.close()


def test_read_only_snapshot_stays_on_its_selected_immutable_generation(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    reader = open_project_read_only(root)
    original_generation_id = reader.generation_id

    publisher = open_project_writable(root)
    try:
        newest = persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    finally:
        publisher.close()

    fresh_reader = open_project_read_only(root)
    try:
        assert reader.mode is ProjectSessionMode.READ_ONLY
        assert reader.generation_id == original_generation_id
        assert fresh_reader.generation_id == newest.generation_id
        assert fresh_reader.generation_id != reader.generation_id
    finally:
        reader.close()
        fresh_reader.close()


def test_use_after_close_is_rejected_and_close_is_idempotent(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    writer = open_project_writable(root)
    writer.close()
    writer.close()

    with pytest.raises(ProjectSessionClosedError):
        _ = writer.project
    with pytest.raises(ProjectSessionClosedError):
        writer.require_strong_source_identity()


def test_reopen_closes_the_prior_snapshot_and_selects_current_generation(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    reader = open_project_read_only(root)
    old_generation_id = reader.generation_id

    publisher = open_project_writable(root)
    try:
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    finally:
        publisher.close()

    reopened = reader.reopen()
    try:
        assert reopened.generation_id != old_generation_id
        with pytest.raises(ProjectSessionClosedError):
            _ = reader.generation_id
    finally:
        reopened.close()
