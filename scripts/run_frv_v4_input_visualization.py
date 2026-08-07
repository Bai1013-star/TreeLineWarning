"""FRV-to-V4 input visualization CLI (FRV-VIS-A1, PLAN-FROZEN).

Renders a published READY FRV membership artifact together with its published
V4 input handoff into a self-contained interactive 3D Plotly HTML document
plus an optional summary JSON and human acceptance checklist.

The CLI never re-runs the FRV or V4 builders and never changes any input
artifact.  Only the exact frozen entry point is supported:

    python scripts/run_frv_v4_input_visualization.py \\
        --source <source.las> \\
        --frv-artifact-root <published-frv-root> \\
        --v4-input-root <published-v4-root> \\
        --output <output.html> \\
        --summary-json <summary.json> \\
        --review-checklist <checklist.md> \\
        --max-total-points 250000 \\
        --max-background-points 50000 \\
        --max-points-per-layer 100000 \\
        --sampling-seed 0 \\
        [--show-original-background] [--show-excluded] [--show-review]

``--scene-simple`` is an explicit convenience alias for
``--source data/samples/scene_simple.las``; it never searches the current
directory for ad-hoc results, and ``--frv-artifact-root`` / ``--v4-input-root``
remain explicit required arguments.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.vegetation.frv_v4_visualization import (  # noqa: E402
    DEFAULT_BACKGROUND_CAP,
    DEFAULT_LAYER_CAP,
    DEFAULT_TOTAL_CAP,
    FrvV4BudgetError,
    FrvV4OutputError,
    FrvV4ValidationError,
    run_frv_v4_visualization,
    validate_display_budget,
)

_SCENE_SIMPLE_SOURCE = _REPOSITORY_ROOT / "data" / "samples" / "scene_simple.las"


class FrvV4CliError(RuntimeError):
    """Expected user-facing invocation failure for the FRV-VIS-A1 runner."""


def _parse_budget(args: argparse.Namespace) -> Any:
    try:
        max_total = int(args.max_total_points)
        max_layer = int(args.max_points_per_layer)
        max_background = int(args.max_background_points)
        seed = int(args.sampling_seed)
    except ValueError as exc:
        raise FrvV4CliError(f"invalid numeric display budget: {exc}") from exc
    try:
        return validate_display_budget(max_total, max_layer, max_background, seed)
    except ValueError as exc:
        raise FrvV4CliError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path, help="source LAS file")
    source_group.add_argument(
        "--scene-simple",
        action="store_true",
        help=(
            "explicit convenience alias for "
            f"--source {_SCENE_SIMPLE_SOURCE}"
        ),
    )
    parser.add_argument("--frv-artifact-root", type=Path, required=True)
    parser.add_argument("--v4-input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--review-checklist", type=Path)
    parser.add_argument("--max-total-points", default=str(DEFAULT_TOTAL_CAP))
    parser.add_argument("--max-background-points", default=str(DEFAULT_BACKGROUND_CAP))
    parser.add_argument("--max-points-per-layer", default=str(DEFAULT_LAYER_CAP))
    parser.add_argument("--sampling-seed", default="0")
    parser.add_argument("--show-original-background", action="store_true")
    parser.add_argument("--show-excluded", action="store_true")
    parser.add_argument("--show-review", action="store_true")
    args = parser.parse_args(argv)
    if args.scene_simple:
        args.source = _SCENE_SIMPLE_SOURCE
    args.budget = _parse_budget(args)
    return args


def run(args: argparse.Namespace) -> Any:
    """Invoke the frozen visualization runner and return the result object."""
    return run_frv_v4_visualization(
        source=args.source,
        frv_artifact_root=args.frv_artifact_root,
        v4_input_root=args.v4_input_root,
        output_html=args.output,
        summary_json=args.summary_json,
        review_checklist=args.review_checklist,
        max_total_points=args.budget.max_total_points,
        max_points_per_layer=args.budget.max_points_per_layer,
        max_background_points=args.budget.max_background_points,
        sampling_seed=args.budget.sampling_seed,
        show_original_background=args.show_original_background,
        show_excluded=args.show_excluded,
        show_review=args.show_review,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
    except (
        FrvV4CliError,
        FrvV4BudgetError,
        FrvV4ValidationError,
        FrvV4OutputError,
        OSError,
    ) as error:
        print(f"FRV-VIS-A1 visualization failed: {error}", file=sys.stderr)
        return 2
    print(f"mode: {result.mode}")
    print(f"html: {result.html_path}")
    if result.summary_path is not None:
        print(f"summary_json: {result.summary_path}")
    if result.checklist_path is not None:
        print(f"review_checklist: {result.checklist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
