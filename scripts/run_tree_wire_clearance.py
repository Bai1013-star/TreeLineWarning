"""TREE_WIRE_DISTANCE_ALGORITHM_V1 clearance CLI (FINAL-TREE-DELIVERY).

Usage::

    E:\\Miniforge\\envs\\treeline\\python.exe scripts/run_tree_wire_clearance.py \\
        --v4-root <published-v4-artifact-root> \\
        --w1-npz <w1-wire-models.npz> \\
        --w1-json <w1-wire-models.json> \\
        --project-root <explicit-project-root> \\
        --output-json <clearance-result.json> \\
        [--config-json <clearance-config.json>] \\
        [--overwrite]

The engine reads only validated V4 and W1 artifacts, hash-verifies the W1
parent A4 through ``--project-root`` and fails closed on any source, origin or
provenance mismatch.  It never invokes FRV, V4 segmentation, W1
reconstruction, Ground or prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from algorithms.vegetation.tree_wire_clearance import (
    TreeWireClearanceError,
    TreeWireClearanceErrorCode,
    run_tree_wire_clearance,
)
from core.contracts.tree_wire_clearance import (
    TreeWireClearanceConfig,
    default_tree_wire_clearance_config,
)

_EXIT_OK = 0
_EXIT_FAIL_CLOSED = 1
_EXIT_USAGE = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_tree_wire_clearance",
        description="Compute full-resolution tree-wire minimum clearances.",
    )
    parser.add_argument("--v4-root", required=True, type=Path,
                        help="Published validated V4 tree-segmentation artifact root.")
    parser.add_argument("--w1-npz", required=True, type=Path,
                        help="W1 wire-model NPZ output.")
    parser.add_argument("--w1-json", required=True, type=Path,
                        help="W1 wire-model JSON result.")
    parser.add_argument("--project-root", required=True, type=Path,
                        help="Explicit project root used to resolve the W1 parent A4.")
    parser.add_argument("--output-json", required=True, type=Path,
                        help="Digest-bound clearance result JSON output path.")
    parser.add_argument("--config-json", type=Path, default=None,
                        help="Optional reviewed clearance config JSON (else defaults).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing output file.")
    return parser.parse_args(argv)


def _load_config(path: Path | None) -> TreeWireClearanceConfig:
    if path is None:
        return default_tree_wire_clearance_config()
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise TreeWireClearanceError(
            code=TreeWireClearanceErrorCode.CONFIG_INVALID,
            message=f"unable to read config JSON: {exc}",
        ) from exc
    try:
        return TreeWireClearanceConfig.from_dict(payload)
    except (ValueError, TypeError) as exc:
        raise TreeWireClearanceError(
            code=TreeWireClearanceErrorCode.CONFIG_INVALID,
            message=f"invalid clearance config: {exc}",
        ) from exc


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = _load_config(args.config_json)
        result = run_tree_wire_clearance(
            args.v4_root,
            args.w1_npz,
            args.w1_json,
            args.project_root,
            config=config,
        )
        output = Path(args.output_json)
        if output.exists() and not args.overwrite:
            raise TreeWireClearanceError(
                code=TreeWireClearanceErrorCode.OUTPUT_EXISTS,
                message="output exists and overwrite is disabled",
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output)
        print(f"Tree-wire clearance published: {output}")
        print(
            f"status={result.status.name} trees={len(result.records)} "
            f"result_digest={result.result_digest}"
        )
        return _EXIT_OK
    except TreeWireClearanceError as exc:
        print(f"TREE_WIRE_CLEARANCE_FAILED: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except (ValueError, TypeError, OSError) as exc:
        print(f"TREE_WIRE_CLEARANCE_INVALID: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED


if __name__ == "__main__":
    sys.exit(main())
