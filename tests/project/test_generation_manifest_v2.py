"""Generation-manifest v1/v2 version-strictness remediation tests."""

from __future__ import annotations

import json

import pytest
from core.project import (
    GENERATION_SCHEMA_VERSION,
    GENERATION_SCHEMA_VERSION_V2,
    create_project_writable,
    open_project_read_only,
    open_project_writable,
    persistence,
)
from core.project.persistence import (
    ProjectIntegrityError,
    file_sha256,
    published_json_bytes,
)

from tests.contracts.test_project_contracts import _tower
from tests.project.test_tower_candidate_persistence import (
    _persist_set,
    _write_true_v1_generation,
)


def _new_project(tmp_path, tiny_las_project):
    root = tmp_path / "project"
    project, metadata, _ = tiny_las_project(root)
    return root, project, metadata


def _rewrite_manifest_only(loaded, mutate):
    """Rewrite only generation.json and the root pointer manifest digest."""
    manifest_path = loaded.generation_path / "generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    data = published_json_bytes(manifest)
    manifest_path.write_bytes(data)
    pointer_path = loaded.project_root / "project.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_manifest_sha256"] = file_sha256(data)
    pointer_path.write_bytes(published_json_bytes(pointer))


# ---------------------------------------------------------------------------
# A. true historical v1 exact five files -> PASS
# ---------------------------------------------------------------------------


def test_true_v1_exact_five_files_loads(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    _write_true_v1_generation(root, project, source_access_metadata=metadata)
    with open_project_read_only(root) as reopened:
        assert reopened.generation_manifest.schema_version == GENERATION_SCHEMA_VERSION
        assert reopened.generation_manifest.files[-1].relative_path == "runs/run-index.json"
        assert reopened.candidate_store.candidate_sets == ()
        assert reopened.candidate_store.review_ledgers == ()
        assert reopened.candidate_store.confirmations == ()


# ---------------------------------------------------------------------------
# B. V1 + extra candidate artifact record -> FAIL
# ---------------------------------------------------------------------------


def test_v1_with_extra_candidate_artifact_record_rejected(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    _write_true_v1_generation(root, project, source_access_metadata=metadata)
    loaded = persistence.open_project(root)
    index_path = loaded.generation_path / "candidates" / "index.json"
    index_path.parent.mkdir()
    index_bytes = published_json_bytes({})
    index_path.write_bytes(index_bytes)

    def mutate(manifest):
        manifest["files"].append(
            {
                "relative_path": "candidates/index.json",
                "byte_size": len(index_bytes),
                "sha256": file_sha256(index_bytes),
            }
        )

    _rewrite_manifest_only(loaded, mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# C. V1 + candidates/index.json (unmanifested) -> FAIL
# ---------------------------------------------------------------------------


def test_v1_with_candidate_index_directory_rejected(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    v1_path = _write_true_v1_generation(root, project, source_access_metadata=metadata)
    index_path = v1_path / "candidates" / "index.json"
    index_path.parent.mkdir()
    index_path.write_bytes(published_json_bytes({}))
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# D. V2 five core + valid empty candidates/index.json -> PASS
# ---------------------------------------------------------------------------


def test_v2_empty_candidate_store_with_index_passes(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        assert session.generation_manifest.schema_version == GENERATION_SCHEMA_VERSION_V2
        assert (session.loaded_project.generation_path / "candidates" / "index.json").exists()
        assert session.candidate_store.candidate_sets == ()
        assert session.candidate_store.review_ledgers == ()
        assert session.candidate_store.confirmations == ()
    with open_project_read_only(root) as reopened:
        assert reopened.generation_manifest.schema_version == GENERATION_SCHEMA_VERSION_V2
        assert reopened.candidate_store.candidate_sets == ()


# ---------------------------------------------------------------------------
# E. V2 missing candidates/index.json -> FAIL
# ---------------------------------------------------------------------------


def test_v2_missing_candidate_index_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        generation_path = session.loaded_project.generation_path
    (generation_path / "candidates" / "index.json").unlink()
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# F. V2 candidate index references missing artifact -> FAIL
# ---------------------------------------------------------------------------


def test_v2_index_referencing_missing_artifact_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _persist_set(session, project.source_reference.binding)
        generation_path = session.loaded_project.generation_path
    (generation_path / "candidates" / "candidate-sets" / "cset-001.json").unlink()
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# G. V2 manifest missing referenced candidate artifact record -> FAIL
# ---------------------------------------------------------------------------


def test_v2_manifest_missing_artifact_record_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _persist_set(session, project.source_reference.binding)
    loaded = persistence.open_project(root)

    def mutate(manifest):
        manifest["files"] = [
            record
            for record in manifest["files"]
            if record["relative_path"] != "candidates/candidate-sets/cset-001.json"
        ]

    _rewrite_manifest_only(loaded, mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# H. V2 artifact digest mismatch -> FAIL
# ---------------------------------------------------------------------------


def test_v2_artifact_digest_mismatch_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata) as session:
        _persist_set(session, project.source_reference.binding)
        generation_path = session.loaded_project.generation_path
    path = generation_path / "candidates" / "candidate-sets" / "cset-001.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# I. unsupported generation schema version -> FAIL
# ---------------------------------------------------------------------------


def test_unsupported_generation_schema_version_fails_closed(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    with create_project_writable(root, project, source_access_metadata=metadata):
        pass
    loaded = persistence.open_project(root)

    def mutate(manifest):
        manifest["schema_version"] = "treeline-project-generation-v3"

    _rewrite_manifest_only(loaded, mutate)
    with pytest.raises(ProjectIntegrityError):
        persistence.open_project(root)


# ---------------------------------------------------------------------------
# J. V1 project successor publication writes V2
# ---------------------------------------------------------------------------


def test_v1_project_successor_publication_writes_v2(tmp_path, tiny_las_project):
    root, project, metadata = _new_project(tmp_path, tiny_las_project)
    v1_path = _write_true_v1_generation(root, project, source_access_metadata=metadata)
    with open_project_read_only(root) as legacy:
        assert legacy.generation_manifest.schema_version == GENERATION_SCHEMA_VERSION
        first_generation = legacy.generation_id
        assert first_generation == v1_path.name

    with open_project_writable(root) as session:
        session.create_tower(_tower("T001", project.source_reference.binding, 0.0))
        assert session.generation_manifest.schema_version == GENERATION_SCHEMA_VERSION_V2
        assert session.generation_id != first_generation
        assert (session.loaded_project.generation_path / "candidates" / "index.json").exists()
        assert session.candidate_store.candidate_sets == ()
        assert session.candidate_store.review_ledgers == ()
        assert session.candidate_store.confirmations == ()
        successor_path = session.loaded_project.generation_path

    # The historical V1 generation remains unchanged and readable.
    assert v1_path == successor_path.parent / first_generation
    assert (v1_path / "generation.json").exists()
    with open_project_read_only(root) as reopened:
        assert reopened.generation_manifest.schema_version == GENERATION_SCHEMA_VERSION_V2
        assert reopened.generation_id != first_generation
        assert reopened.tower_registry.get("T001").current_revision == 1
