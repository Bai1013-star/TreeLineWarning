"""Focused tests for the production P-WIRE authority publisher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from algorithms.vegetation.wire_provider import (
    FrvWireAuthorityError,
    publish_frv_wire_authority,
)
from core.contracts.frv_wire import (
    FrvWireAuthorityMode,
    validate_frv_wire_authority_manifest,
)

from tests.contracts.test_frv_wire_contracts import _config as _wire_config
from tests.vegetation.test_frv_membership import write_scene_las
from tests.vegetation.test_tree_wire_clearance import (
    _wire_segment,
    _write_a4_w1,
)
from tests.vegetation.test_wire_provider import (
    _configuration_for_wire,
    _run_source,
)


def _source_result(tmp_path: Path):
    las_path = tmp_path / "scene.las"
    write_scene_las(las_path)
    reference, configuration, source_result = _run_source(las_path, chunk_size=40)
    return reference, configuration, source_result


def _publish(
    tmp_path: Path,
    source_result,
    *,
    segments=None,
    **kwargs,
):
    if segments is None:
        segments = [
            (0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0))),
            (0, 1, _wire_segment((5.0, 2.0, 3.0), (13.0, 2.0, 3.0))),
            (1, 0, _wire_segment((0.0, 9.0, 1.0), (10.0, 9.0, 1.0))),
        ]
    w1_npz, w1_json, project_root = _write_a4_w1(tmp_path, segments)
    return publish_frv_wire_authority(
        w1_npz_path=w1_npz,
        w1_json_path=w1_json,
        project_root=project_root,
        source_result=source_result,
        output_dir=tmp_path / "authority",
        approver="test-approver",
        approval_timestamp="2026-01-01T00:00:00Z",
        **kwargs,
    )


def test_publisher_writes_valid_authority_and_reference(tmp_path: Path) -> None:
    _reference, _configuration, source_result = _source_result(tmp_path)
    authority = _publish(tmp_path, source_result)
    assert authority.manifest.authority_mode is FrvWireAuthorityMode.W1_MODEL_SET
    assert authority.manifest.approved_wire_ids == (0, 1)
    validate_frv_wire_authority_manifest(authority.manifest)
    assert authority.authority_manifest_path.is_file()
    assert authority.w1_npz_path is not None
    assert authority.w1_json_path is not None
    # The written manifest re-reads to the same digest.
    reloaded = json.loads(authority.authority_manifest_path.read_text("utf-8"))
    assert reloaded["manifest_digest"] == authority.manifest.manifest_digest


def test_published_authority_is_usable_by_load_frv_wire(tmp_path: Path) -> None:
    from algorithms.vegetation.wire_provider import load_frv_wire

    reference, configuration, source_result = _source_result(tmp_path)
    authority = _publish(tmp_path, source_result)
    wire_config = _wire_config()
    session = load_frv_wire(
        reference,
        source_result,
        authority,
        frv_configuration=_configuration_for_wire(wire_config),
        provider_config=wire_config,
        source_chunk_size=40,
    )
    assert session is not None


def test_publisher_fails_on_wrong_source_identity(tmp_path: Path) -> None:
    _reference, _configuration, source_result = _source_result(tmp_path)
    from tests.vegetation.test_tree_wire_clearance import _write_a4_w1

    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
        source_id="other-scene",
        manifest_source_id="other-scene",
    )
    with pytest.raises(FrvWireAuthorityError, match="source identifiers"):
        publish_frv_wire_authority(
            w1_npz_path=w1_npz,
            w1_json_path=w1_json,
            project_root=project_root,
            source_result=source_result,
            output_dir=tmp_path / "authority",
            approver="test-approver",
            approval_timestamp="2026-01-01T00:00:00Z",
        )


def test_publisher_fails_on_origin_mismatch(tmp_path: Path) -> None:
    _reference, _configuration, source_result = _source_result(tmp_path)
    from tests.vegetation.test_tree_wire_clearance import _write_a4_w1

    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path,
        [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))],
        origin=(500_000.0, 4_200_000.0, 125.0),
    )
    with pytest.raises(FrvWireAuthorityError, match="origins differ"):
        publish_frv_wire_authority(
            w1_npz_path=w1_npz,
            w1_json_path=w1_json,
            project_root=project_root,
            source_result=source_result,
            output_dir=tmp_path / "authority",
            approver="test-approver",
            approval_timestamp="2026-01-01T00:00:00Z",
        )


def test_publisher_fails_on_tampered_a4_parent(tmp_path: Path) -> None:
    _reference, _configuration, source_result = _source_result(tmp_path)
    from tests.vegetation.test_tree_wire_clearance import _write_a4_w1

    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path, [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))]
    )
    a4_npz = project_root / "outputs" / "wires" / "scene-1_wire_instances.npz"
    a4_npz.write_bytes(b"tampered")
    with pytest.raises(FrvWireAuthorityError, match="A4"):
        publish_frv_wire_authority(
            w1_npz_path=w1_npz,
            w1_json_path=w1_json,
            project_root=project_root,
            source_result=source_result,
            output_dir=tmp_path / "authority",
            approver="test-approver",
            approval_timestamp="2026-01-01T00:00:00Z",
        )


def test_publisher_fails_on_missing_w1_artifact(tmp_path: Path) -> None:
    _reference, _configuration, source_result = _source_result(tmp_path)
    with pytest.raises(FrvWireAuthorityError, match="W1"):
        publish_frv_wire_authority(
            w1_npz_path=tmp_path / "missing.npz",
            w1_json_path=tmp_path / "missing.json",
            project_root=tmp_path,
            source_result=source_result,
            output_dir=tmp_path / "authority",
            approver="test-approver",
            approval_timestamp="2026-01-01T00:00:00Z",
        )


def test_publisher_fails_on_corrupt_w1_digest(tmp_path: Path) -> None:
    _reference, _configuration, source_result = _source_result(tmp_path)
    w1_npz, w1_json, project_root = _write_a4_w1(
        tmp_path, [(0, 0, _wire_segment((0.0, 2.0, 3.0), (4.0, 2.0, 3.0)))]
    )
    w1_npz.write_bytes(b"corrupted")
    with pytest.raises(FrvWireAuthorityError, match="W1"):
        publish_frv_wire_authority(
            w1_npz_path=w1_npz,
            w1_json_path=w1_json,
            project_root=project_root,
            source_result=source_result,
            output_dir=tmp_path / "authority",
            approver="test-approver",
            approval_timestamp="2026-01-01T00:00:00Z",
        )


def test_publisher_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    _reference, _configuration, source_result = _source_result(tmp_path)
    _publish(tmp_path, source_result)
    with pytest.raises(FrvWireAuthorityError, match="exists"):
        _publish(tmp_path, source_result)
    authority = _publish(tmp_path, source_result, overwrite=True)
    assert authority.manifest.manifest_digest
