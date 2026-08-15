from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from core.contracts.frv import canonical_frv_json_bytes
from core.project import persistence
from core.project.persistence import (
    ACTIVE_POINTER_SCHEMA_VERSION,
    FileIntegrityError,
    PersistenceError,
    canonical_json_bytes,
    logical_json_digest,
    open_project,
    project_from_dict,
    published_json_bytes,
)


def _create(root: Path, project, metadata):
    return persistence._create_project_unlocked_for_tests(
        root, project, source_access_metadata=metadata
    )


def _root_pointer(root: Path) -> dict:
    return json.loads((root / "project.json").read_text(encoding="utf-8"))


def _write_root_pointer(root: Path, pointer: dict) -> None:
    (root / "project.json").write_bytes(published_json_bytes(pointer))


def test_canonical_json_is_deterministic_and_has_distinct_logical_file_digests():
    value = {"z": "线路", "a": [2, 1], "nested": {"b": True, "a": None}}
    assert canonical_json_bytes(value) == (
        '{"a":[2,1],"nested":{"a":null,"b":true},"z":"线路"}'.encode()
    )
    assert published_json_bytes(value).endswith(b"\n")
    assert not canonical_json_bytes(value).endswith(b"\n")
    assert logical_json_digest(value) != persistence.file_sha256(published_json_bytes(value))


def test_canonical_json_parity_with_frv_helper():
    value = {"unicode": "树", "float": 1.25, "tuple": (1, 2)}
    assert canonical_json_bytes(value) == canonical_frv_json_bytes(value)


def test_project_contract_round_trip_is_strict_and_prediction_does_not_import_ml(tiny_project):
    payload = tiny_project.to_dict()
    restored = project_from_dict(json.loads(json.dumps(payload)))
    assert restored == tiny_project
    assert restored.prediction_capability.to_dict() == tiny_project.prediction_capability.to_dict()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["source_reference"]["coordinate_context"].update(
            {"units": "bad"}
        ),
        lambda value: value["source_reference"]["source_identity"].update(
            {"source_file_sha256": "bad"}
        ),
        lambda value: value["prediction_capability"].update({"status": "bad"}),
        lambda value: value.update({"future": True}),
    ],
)
def test_project_strict_decode_rejects_malformed_nested_or_unknown_fields(tiny_project, mutator):
    payload = json.loads(json.dumps(tiny_project.to_dict()))
    mutator(payload)
    with pytest.raises(PersistenceError):
        project_from_dict(payload)


def test_unknown_project_schema_rejected(tiny_project):
    payload = tiny_project.to_dict()
    payload["schema_version"] = "treeline-project-v2"
    with pytest.raises(PersistenceError):
        project_from_dict(payload)


def test_generation_layout_manifest_and_empty_envelopes_round_trip(
    tmp_path, tiny_project, deterministic_source_metadata
):
    loaded = _create(tmp_path / "project", tiny_project, deterministic_source_metadata)
    root = loaded.project_root
    generation = loaded.generation_path
    assert loaded.project.project_id == tiny_project.project_id
    # Every newly published generation is generation-v2, which always carries
    # the candidate store index alongside the five core project files.
    assert loaded.generation_manifest.schema_version == persistence.GENERATION_SCHEMA_VERSION_V2
    assert sorted(
        path.relative_to(generation).as_posix() for path in generation.rglob("*.json")
    ) == [
        "candidates/index.json",
        "generation.json",
        "project.json",
        "runs/run-index.json",
        "source/source-reference.json",
        "spans/spans.json",
        "towers/towers.json",
    ]
    assert not (root / "project.lock").exists()
    assert not (root / "viewer").exists()
    assert loaded.tower_registry.records == ()
    assert loaded.span_registry.records == ()
    assert loaded.run_index.entries == ()
    assert loaded.candidate_store.candidate_sets == ()
    assert loaded.generation_manifest.sealed is True
    assert loaded.generation_manifest.files[-1].relative_path == "candidates/index.json"


def test_root_pointer_selects_exact_generation_and_reopen_preserves_source(
    tmp_path, tiny_project, deterministic_source_metadata
):
    loaded = _create(tmp_path / "project", tiny_project, deterministic_source_metadata)
    reopened = open_project(tmp_path / "project")
    pointer = _root_pointer(tmp_path / "project")
    assert pointer["schema_version"] == ACTIVE_POINTER_SCHEMA_VERSION
    assert pointer["active_generation_id"] == loaded.generation_id
    assert reopened.generation_id == loaded.generation_id
    assert reopened.project.project_id == tiny_project.project_id
    assert reopened.project.source_reference == tiny_project.source_reference
    assert reopened.source_sidecar.project_source_reference == tiny_project.source_reference


def test_prior_generation_is_unchanged_and_successful_private_publish_selects_new_generation(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    first_bytes = (first.generation_path / "project.json").read_bytes()
    second = persistence._publish_generation(
        root,
        tiny_project,
        source_access_metadata=deterministic_source_metadata,
    )
    assert second.generation_id != first.generation_id
    assert open_project(root).generation_id == second.generation_id
    assert (first.generation_path / "project.json").read_bytes() == first_bytes
    assert (first.generation_path / "generation.json").exists()


@pytest.mark.parametrize(
    "failure_point",
    ["staging", "before_manifest", "before_rename", "before_pointer"],
)
def test_interrupted_publication_preserves_prior_active_generation(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch, failure_point
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)

    if failure_point == "staging":
        original = persistence._write_json

        def fail(path, value):
            if ".staging-" in str(path) and str(path).endswith("towers.json"):
                raise OSError("injected staging interruption")
            return original(path, value)

        monkeypatch.setattr(persistence, "_write_json", fail)
    elif failure_point == "before_manifest":
        original = persistence._write_json

        def fail(path, value):
            if ".staging-" in str(path) and str(path).endswith("generation.json"):
                raise OSError("injected pre-seal interruption")
            return original(path, value)

        monkeypatch.setattr(persistence, "_write_json", fail)
    elif failure_point == "before_rename":
        original = persistence.os.replace

        def fail(source, target):
            if Path(source).name.startswith(".staging-"):
                raise OSError("injected pre-publication interruption")
            return original(source, target)

        monkeypatch.setattr(persistence.os, "replace", fail)
    else:
        monkeypatch.setattr(
            persistence,
            "_atomic_replace_root_pointer",
            lambda root_path, pointer: (_ for _ in ()).throw(
                OSError("injected pre-pointer interruption")
            ),
        )

    with pytest.raises((OSError, PersistenceError)):
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert open_project(root).generation_id == first.generation_id


def _directory_failure(path: Path, operation: str = "test-directory-fsync"):
    return persistence.DirectoryDurabilityError(
        f"injected directory durability failure: {path}",
        operation=operation,
        path=path,
    )


def test_supported_child_directory_fsync_failure_aborts_new_generation(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    original = persistence._fsync_directory

    def fail(path):
        if Path(path).name == "source":
            raise _directory_failure(Path(path))
        return original(path)

    monkeypatch.setattr(persistence, "_fsync_directory", fail)
    with pytest.raises(persistence.DirectoryDurabilityError):
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert open_project(root).generation_id == first.generation_id


def test_staging_generation_preseal_directory_failure_preserves_root(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    original = persistence._fsync_directory

    def fail(path):
        if Path(path).name.startswith(".staging-"):
            raise _directory_failure(Path(path), "preseal-staging-directory-fsync")
        return original(path)

    monkeypatch.setattr(persistence, "_fsync_directory", fail)
    with pytest.raises(persistence.DirectoryDurabilityError):
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert open_project(root).generation_id == first.generation_id


def test_staging_generation_postseal_directory_failure_prevents_rename(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    original = persistence._fsync_directory
    staging_calls = 0

    def fail(path):
        nonlocal staging_calls
        if Path(path).name.startswith(".staging-"):
            staging_calls += 1
            if staging_calls == 2:
                raise _directory_failure(Path(path), "postseal-staging-directory-fsync")
        return original(path)

    monkeypatch.setattr(persistence, "_fsync_directory", fail)
    with pytest.raises(persistence.DirectoryDurabilityError):
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert open_project(root).generation_id == first.generation_id
    assert staging_calls == 2


def test_generations_parent_postrename_failure_prevents_root_switch(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    original = persistence._fsync_directory

    def fail(path):
        if Path(path).name == "generations":
            raise _directory_failure(Path(path), "generations-parent-fsync")
        return original(path)

    monkeypatch.setattr(persistence, "_fsync_directory", fail)
    with pytest.raises(persistence.DirectoryDurabilityError):
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert open_project(root).generation_id == first.generation_id


def test_root_pointer_temp_durability_failure_preserves_old_pointer(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    original = persistence._write_json

    def fail(path, value):
        if Path(path).name.startswith(".project.json.tmp-"):
            raise persistence.FileDurabilityError(
                "injected root pointer temp durability failure",
                operation="root-pointer-temp-write-fsync",
                path=Path(path),
            )
        return original(path, value)

    monkeypatch.setattr(persistence, "_write_json", fail)
    with pytest.raises(persistence.RootPointerReplaceError):
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert open_project(root).generation_id == first.generation_id


def test_root_pointer_replace_failure_preserves_old_pointer(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    original = persistence.os.replace

    def fail(source, target):
        if Path(target).name == "project.json":
            raise OSError("injected root pointer replace failure")
        return original(source, target)

    monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(persistence.RootPointerReplaceError):
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert open_project(root).generation_id == first.generation_id


def test_post_pointer_root_directory_failure_is_explicitly_ambiguous(
    tmp_path, tiny_project, deterministic_source_metadata, monkeypatch
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    original = persistence._fsync_directory

    def fail(path):
        if Path(path) == root:
            raise _directory_failure(Path(path), "post-pointer-root-directory-fsync")
        return original(path)

    monkeypatch.setattr(persistence, "_fsync_directory", fail)
    with pytest.raises(persistence.PostPointerDurabilityError) as raised:
        persistence._publish_generation(
            root,
            tiny_project,
            source_access_metadata=deterministic_source_metadata,
        )
    assert raised.value.pointer_replaced is True
    assert "reopen and revalidate" in str(raised.value)
    assert open_project(root).generation_id != first.generation_id


def test_directory_fsync_unsupported_capability_is_explicit(tmp_path):
    result = persistence._fsync_directory(tmp_path)
    if os.name == "nt":
        assert result.supported is False
        assert "unsupported" in result.detail
    else:
        assert result.supported is True


def test_unrelated_directory_open_oserror_is_not_misclassified(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(persistence, "_directory_fsync_supported", lambda: True)

    def fail_open(path, flags):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(persistence.os, "open", fail_open)
    with pytest.raises(persistence.DirectoryDurabilityError) as raised:
        persistence._fsync_directory(tmp_path)
    assert raised.value.operation == "directory-open"


def test_incomplete_unreferenced_staging_is_ignored_but_active_staging_is_rejected(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    first = _create(root, tiny_project, deterministic_source_metadata)
    staging = root / "state" / "generations" / ".staging-orphan"
    staging.mkdir()
    (staging / "generation.json").write_bytes(b"not complete\n")
    assert open_project(root).generation_id == first.generation_id

    pointer = _root_pointer(root)
    pointer["active_generation_id"] = ".staging-orphan"
    pointer["generation_manifest_reference"] = (
        "state/generations/.staging-orphan/generation.json"
    )
    pointer["project_manifest_reference"] = "state/generations/.staging-orphan/project.json"
    _write_root_pointer(root, pointer)
    with pytest.raises(PersistenceError):
        open_project(root)


@pytest.mark.parametrize("target", ["size", "digest", "project", "registry"])
def test_corruption_fails_closed(tmp_path, tiny_project, deterministic_source_metadata, target):
    root = tmp_path / "project"
    loaded = _create(root, tiny_project, deterministic_source_metadata)
    path = loaded.generation_path / {
        "size": "towers/towers.json",
        "digest": "spans/spans.json",
        "project": "project.json",
        "registry": "runs/run-index.json",
    }[target]
    data = path.read_bytes()
    if target == "size":
        path.write_bytes(data + b" ")
    elif target == "digest":
        path.write_bytes(data.replace(b"[]", b"[ ]", 1))
    elif target == "project":
        path.write_bytes(b"{}\n")
    else:
        path.write_bytes(b"{\"broken\":true}\n")
    with pytest.raises((PersistenceError, FileIntegrityError)):
        open_project(root)


def test_mixed_generation_id_and_unsafe_reference_rejected(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    loaded = _create(root, tiny_project, deterministic_source_metadata)
    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_id"] = "g-0000000000000002-deadbeef"
    manifest_path.write_bytes(published_json_bytes(manifest))
    with pytest.raises(PersistenceError):
        open_project(root)

    # Recreate a valid project for the separate path-safety assertion.
    root2 = tmp_path / "project2"
    _create(root2, tiny_project, deterministic_source_metadata)
    pointer = _root_pointer(root2)
    pointer["generation_manifest_reference"] = "../outside/generation.json"
    _write_root_pointer(root2, pointer)
    with pytest.raises(PersistenceError):
        open_project(root2)
