"""V4 tree-segmentation CLI tests (PLAN-FROZEN consolidated plan).

The CLI only accepts formal digest-bound v2 V4 input roots; v1 roots, manifest
or digest mismatches, corrupt blocks, missing configs and existing output roots
fail closed with a nonzero exit code and no partial output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.run_v4_tree_segmentation as cli
from core.contracts.v4_tree_segmentation import V4ConfigStatus

from tests.vegetation.test_v4_inputs import (
    build_v4_input_handoff,
    run_membership_ready,
)
from tests.vegetation.test_v4_tree_segmentation import make_config


@pytest.fixture(scope="module")
def v2_root(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("cli_v2"))
    packed = run_membership_ready(root)
    result, _las, _col, _ground, reference, source_result, configuration, _ref, _auth = packed
    out = root / "v4-input-root"
    handoff = build_v4_input_handoff(
        reference,
        source_result,
        result,
        frv_configuration=configuration,
        output_dir=out,
        source_chunk_size=40,
    )
    assert handoff.output_reference is not None
    return handoff.output_reference.root_path


@pytest.fixture()
def config_json(tmp_path: Path) -> Path:
    config = make_config(status=V4ConfigStatus.CALIBRATION_CANDIDATE)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), "utf-8")
    return path


def _run_cli(*argv: str) -> int:
    return cli.main(list(argv))


def test_cli_success(v2_root: Path, config_json: Path, tmp_path: Path) -> None:
    output = tmp_path / "result"
    code = _run_cli(
        "--v4-input-root", str(v2_root),
        "--config-json", str(config_json),
        "--output-root", str(output),
    )
    assert code == 0
    assert (output / "v4_tree_segmentation.json").is_file()
    assert (output / "v4_tree_segmentation.npz").is_file()
    assert (output / "v4_tree_segmentation_manifest.json").is_file()


def test_cli_rejects_v1_root(v2_root: Path, config_json: Path, tmp_path: Path) -> None:
    import shutil

    v1_root = tmp_path / "v1-root"
    shutil.copytree(v2_root, v1_root)
    manifest_path = v1_root / "handoff_manifest.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["schema_version"] = "v4-input-handoff-v1"
    for key in (
        "source_identity_record_digest",
        "origin_context",
        "normal_row_count",
        "review_row_count",
        "normal_block_paths",
        "review_block_paths",
        "normal_block_digests",
        "review_block_digests",
        "normal_set_digest",
        "review_set_digest",
        "handoff_digest",
        "manifest_digest",
        "source_proof_digest",
        "hag_proof_digest",
        "geometry_proof_digest",
        "wire_proof_digest",
    ):
        payload.pop(key)
    manifest_path.write_text(json.dumps(payload), "utf-8")
    output = tmp_path / "result-v1"
    code = _run_cli(
        "--v4-input-root", str(v1_root),
        "--config-json", str(config_json),
        "--output-root", str(output),
    )
    assert code == 1
    assert not output.exists()


def test_cli_rejects_corrupt_block(v2_root: Path, config_json: Path, tmp_path: Path) -> None:
    import shutil

    corrupt_root = tmp_path / "corrupt-root"
    shutil.copytree(v2_root, corrupt_root)
    block = next((corrupt_root / "normal").glob("*.v4blk"))
    data = bytearray(block.read_bytes())
    data[-1] ^= 0xFF
    block.write_bytes(bytes(data))
    output = tmp_path / "result-corrupt"
    code = _run_cli(
        "--v4-input-root", str(corrupt_root),
        "--config-json", str(config_json),
        "--output-root", str(output),
    )
    assert code == 1
    assert not output.exists()


def test_cli_rejects_existing_output(v2_root: Path, config_json: Path, tmp_path: Path) -> None:
    output = tmp_path / "result-existing"
    output.mkdir()
    code = _run_cli(
        "--v4-input-root", str(v2_root),
        "--config-json", str(config_json),
        "--output-root", str(output),
    )
    assert code == 1


def test_cli_rejects_missing_config(v2_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "result-noconfig"
    code = _run_cli(
        "--v4-input-root", str(v2_root),
        "--config-json", str(tmp_path / "missing.json"),
        "--output-root", str(output),
    )
    assert code == 1
    assert not output.exists()


def test_cli_rejects_test_only_config(v2_root: Path, tmp_path: Path) -> None:
    config = make_config(status=V4ConfigStatus.TEST_ONLY)
    config_path = tmp_path / "test-only.json"
    config_path.write_text(json.dumps(config.to_dict()), "utf-8")
    output = tmp_path / "result-testonly"
    code = _run_cli(
        "--v4-input-root", str(v2_root),
        "--config-json", str(config_path),
        "--output-root", str(output),
    )
    assert code == 1
    assert not output.exists()


def test_cli_no_partial_output_on_failure(
    v2_root: Path, config_json: Path, tmp_path: Path
) -> None:
    output = tmp_path / "result-nested" / "out"
    code = _run_cli(
        "--v4-input-root", str(v2_root),
        "--config-json", str(config_json),
        "--output-root", str(output),
    )
    assert code == 1
    assert not output.exists()
