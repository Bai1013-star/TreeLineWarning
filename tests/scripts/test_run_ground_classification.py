"""Bundle D operator CLI tests (PLAN-FROZEN section 6).

The CLI requires a formal P-SOURCE reference built from a raw LAS; it fails
closed on malformed arguments, missing/non-frozen policy, unreadable sources,
and machine hard failures.  A successful run reports
READY_FOR_OWNER_HUMAN_GATE_REVIEW and never auto-approves the Ground Human Gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scripts.run_ground_classification as cli
from algorithms.ground.classification import FgcExecutionError
from core.contracts.ground_classification import (
    FGC_COMPLETION_MANIFEST_PATH,
    GroundClassificationPolicyConfig,
)

from tests.ground._scene import write_synthetic_las


def _write_scene(path: Path, *, side_m: float = 8.0) -> None:
    gx, gy = np.meshgrid(
        np.arange(0.0, side_m, 0.5),
        np.arange(0.0, side_m, 0.5),
    )
    write_synthetic_las(
        path,
        x=100.0 + gx.ravel(),
        y=200.0 + gy.ravel(),
        z=np.zeros(gx.size),
    )


def _transaction_root(output_root: Path, source_id: str, sha_prefix: str) -> Path:
    return (
        output_root
        / "ground-classification"
        / source_id
        / sha_prefix
    )


def test_cli_runs_and_reports_ready(tmp_path: Path, capsys) -> None:
    source = tmp_path / "scene.las"
    _write_scene(source)
    output_root = tmp_path / "out"
    code = cli.main(
        ["--source-las", str(source), "--output-root", str(output_root)]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "FGC_CLI_RESULT: READY_FOR_OWNER_HUMAN_GATE_REVIEW" in captured.out
    # The publication root is discoverable and the review material exists.
    sha = cli._file_sha256(source)
    source_id = cli.stable_source_id(source, source.parent)
    root = _transaction_root(output_root, source_id, sha[:16])
    assert (root / FGC_COMPLETION_MANIFEST_PATH).is_file()
    assert (root / "diagnostics" / "fgc_ground_review.html").is_file()
    assert (root / "diagnostics" / "fgc_ground_review_summary.json").is_file()


def test_cli_structured_result_is_json_safe(tmp_path: Path, capsys) -> None:
    source = tmp_path / "scene.las"
    _write_scene(source)
    output_root = tmp_path / "out"
    assert cli.main(["--source-las", str(source), "--output-root", str(output_root)]) == 0
    capsys.readouterr()
    sha = cli._file_sha256(source)
    source_id = cli.stable_source_id(source, source.parent)
    root = _transaction_root(output_root, source_id, sha[:16])
    summary = json.loads(
        (root / "diagnostics" / "fgc_ground_review_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "READY_FOR_OWNER_HUMAN_GATE_REVIEW"
    assert summary["owner_decision"] == "PENDING"
    assert summary["counts"]["total_points"] > 0
    assert "PASS_READY_FOR_FRV_RUNTIME" not in json.dumps(summary)


def test_cli_honors_custom_review_output(tmp_path: Path, capsys) -> None:
    source = tmp_path / "scene.las"
    _write_scene(source)
    output_root = tmp_path / "out"
    review_dir = tmp_path / "review_dir"
    code = cli.main(
        [
            "--source-las",
            str(source),
            "--output-root",
            str(output_root),
            "--review-output",
            str(review_dir),
        ]
    )
    capsys.readouterr()
    assert code == 0
    assert (review_dir / "fgc_ground_review.html").is_file()
    assert (review_dir / "fgc_ground_review_summary.json").is_file()


def test_cli_missing_source_blocked(tmp_path: Path, capsys) -> None:
    code = cli.main(
        [
            "--source-las",
            str(tmp_path / "missing.las"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "FGC_RUNTIME_BLOCKED:SOURCE_MISSING" in captured.err


def test_cli_laz_rejected(tmp_path: Path, capsys) -> None:
    source = tmp_path / "scene.laz"
    source.write_bytes(b"not really laz")
    code = cli.main(
        ["--source-las", str(source), "--output-root", str(tmp_path / "out")]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "FGC_RUNTIME_BLOCKED:SOURCE_FORMAT_REJECTED" in captured.err


def test_cli_malformed_arguments_exit_usage(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    capsys.readouterr()


def test_cli_non_frozen_policy_blocked(tmp_path: Path, capsys) -> None:
    source = tmp_path / "scene.las"
    _write_scene(source)
    policy_path = tmp_path / "policy.json"
    payload = dict(GroundClassificationPolicyConfig().to_dict())
    payload["tile_side_m"] = 512.0
    payload["policy_digest"] = ""
    policy_path.write_text(
        json.dumps(GroundClassificationPolicyConfig.from_dict(payload).to_dict()),
        encoding="utf-8",
    )
    code = cli.main(
        [
            "--source-las",
            str(source),
            "--output-root",
            str(tmp_path / "out"),
            "--policy-json",
            str(policy_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "FGC_RUNTIME_BLOCKED:FGC_POLICY_NOT_REVIEWED" in captured.err


def test_cli_invalid_policy_json_blocked(tmp_path: Path, capsys) -> None:
    source = tmp_path / "scene.las"
    _write_scene(source)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{ not json", encoding="utf-8")
    code = cli.main(
        [
            "--source-las",
            str(source),
            "--output-root",
            str(tmp_path / "out"),
            "--policy-json",
            str(policy_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "FGC_RUNTIME_BLOCKED:FGC_POLICY_INVALID" in captured.err


def test_cli_machine_hard_failure_blocked(tmp_path: Path, capsys) -> None:
    # A no-reliable-ground scene fails the coverage occupancy gate.
    xs = np.asarray([100.0, 105.0, 110.0, 115.0])
    ys = np.asarray([200.0, 205.0, 210.0, 215.0])
    zs = np.asarray([10.0, 11.0, 12.0, 13.0])
    source = tmp_path / "no_ground.las"
    write_synthetic_las(source, x=xs, y=ys, z=zs)
    code = cli.main(
        ["--source-las", str(source), "--output-root", str(tmp_path / "out")]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "FGC_RUNTIME_BLOCKED:GROUND_COVERAGE_GAP" in captured.err
    # No completion manifest is published after a fail-closed run.
    assert not list((tmp_path / "out").rglob(FGC_COMPLETION_MANIFEST_PATH))


def test_cli_policy_json_frozen_equivalent_accepted(tmp_path: Path, capsys) -> None:
    source = tmp_path / "scene.las"
    _write_scene(source)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(GroundClassificationPolicyConfig().to_dict()), encoding="utf-8"
    )
    code = cli.main(
        [
            "--source-las",
            str(source),
            "--output-root",
            str(tmp_path / "out"),
            "--policy-json",
            str(policy_path),
        ]
    )
    capsys.readouterr()
    assert code == 0


def test_cli_cancellation_token_is_honored(tmp_path: Path) -> None:
    source = tmp_path / "scene.las"
    _write_scene(source)
    identity = cli._build_raw_source_identity(
        source, cli._file_sha256(source), _header_of(source)
    )
    provider_config = cli._build_provider_config()
    configuration = cli._build_configuration()
    reference = cli._build_reference(
        source, identity, provider_config, configuration
    )
    result = cli._load_source_result(reference, provider_config, configuration)

    class Cancelled:
        def is_cancelled(self) -> bool:
            return True

    from core.contracts.ground_classification import GroundClassificationPolicyConfig

    with pytest.raises(FgcExecutionError) as exc:
        cli.classify_ground(
            reference,
            result,
            configuration,
            cli._build_hag_config(1.0),
            tmp_path / "out",
            GroundClassificationPolicyConfig(),
            cancellation=Cancelled(),
        )
    assert exc.value.code == "SESSION_CANCELLED"


def _header_of(path: Path):
    import laspy

    with laspy.open(path, mode="r") as reader:
        return reader.header


def test_cli_cancellation_token_starts_uncancelled() -> None:
    with cli._CancellationToken() as token:
        assert token.is_cancelled() is False
