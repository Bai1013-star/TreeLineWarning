"""V4 single-tree segmentation CLI (V4-BUNDLE-A, PLAN-FROZEN).

Usage::

    E:\\Miniforge\\envs\\treeline\\python.exe scripts/run_v4_tree_segmentation.py \\
        --v4-input-root <published-v2-v4-input-root> \\
        --config-json <reviewed-v4-config.json> \\
        --output-root <new-result-root>

The CLI accepts only a formal digest-bound v2 V4 input root (loaded through
``load_v4_input_handoff``); a v1 root, a manifest/digest mismatch, a source or
origin mismatch, corrupted blocks, an incomplete inventory or an existing
output root when overwrite is prohibited all fail closed with a nonzero exit
code and no partial output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from algorithms.vegetation.v4_inputs import load_v4_input_handoff
from algorithms.vegetation.v4_tree_artifacts import publish_v4_tree_segmentation_artifact
from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees
from core.contracts.v4_inputs import V4InputLoadError
from core.contracts.v4_tree_segmentation import (
    V4ConfigStatus,
    V4TreeSegmentationConfig,
    V4TreeSegmentationError,
    V4TreeSegmentationErrorCode,
)

_EXIT_OK = 0
_EXIT_FAIL_CLOSED = 1
_EXIT_USAGE = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_v4_tree_segmentation",
        description="Run formal V4 single-tree segmentation on a v2 input root.",
    )
    parser.add_argument(
        "--v4-input-root",
        required=True,
        type=Path,
        help="Published digest-bound v2 V4 input handoff root.",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        type=Path,
        help="Reviewed V4 segmentation config JSON.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New result artifact root (must not already exist).",
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> V4TreeSegmentationConfig:
    if not path.is_file():
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.CONFIG_INVALID, field_name="config_json"
        )
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.CONFIG_INVALID, field_name="config_json"
        ) from exc
    if payload.get("status") not in (
        V4ConfigStatus.CALIBRATION_CANDIDATE.value,
        V4ConfigStatus.PRODUCTION_APPROVED.value,
    ):
        raise V4TreeSegmentationError(
            code=V4TreeSegmentationErrorCode.CONFIG_STATUS_UNSUPPORTED,
            field_name="status",
        )
    return V4TreeSegmentationConfig.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if not args.v4_input_root.is_dir():
            raise V4TreeSegmentationError(
                code=V4TreeSegmentationErrorCode.INPUT_ROOT_INVALID,
                field_name="v4_input_root",
            )
        config = _load_config(args.config_json)
        if args.output_root.exists():
            raise V4TreeSegmentationError(
                code=V4TreeSegmentationErrorCode.PUBLICATION_FAILED,
                field_name="output_root",
            )
        handoff = load_v4_input_handoff(args.v4_input_root)
        result = segment_v4_trees(handoff, config)
        reference = publish_v4_tree_segmentation_artifact(
            result, args.output_root, scratch_dir=args.output_root.parent
        )
        print(f"V4 segmentation published: {reference.root_path}")
        print(
            f"trees={len(result.tree_records)} "
            f"total_points={result.audit.total_points} "
            f"result_digest={result.result_digest}"
        )
        return _EXIT_OK
    except V4TreeSegmentationError as exc:
        print(f"V4_TREE_SEGMENTATION_FAILED: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except V4InputLoadError as exc:
        print(f"V4_INPUT_LOAD_FAILED: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except (ValueError, TypeError, OSError) as exc:
        print(f"V4_TREE_SEGMENTATION_INVALID: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED


if __name__ == "__main__":
    sys.exit(main())
