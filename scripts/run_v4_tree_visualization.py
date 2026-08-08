"""V4 tree-segmentation review visualization CLI (FINAL-TREE-DELIVERY).

Usage::

    E:\\Miniforge\\envs\\treeline\\python.exe scripts/run_v4_tree_visualization.py \\
        --v4-root <published-v4-artifact-root> \\
        --output-html <review.html> \\
        [--summary-json <review.json>] \\
        [--max-points-per-tree 20000] \\
        [--max-review-points 50000] \\
        [--max-unassigned-points 50000] \\
        [--max-conflict-points 20000] \\
        [--overwrite]

The viewer reads only the validated V4 artifact and never invokes FRV, V4
segmentation, W1 reconstruction, Ground or prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from algorithms.vegetation.v4_tree_visualization import (
    DEFAULT_MAX_CONFLICT_POINTS,
    DEFAULT_MAX_POINTS_PER_TREE,
    DEFAULT_MAX_REVIEW_POINTS,
    DEFAULT_MAX_UNASSIGNED_POINTS,
    V4TreeVisualizationError,
    V4TreeVisualizationOutputError,
    run_v4_tree_visualization,
)

_EXIT_OK = 0
_EXIT_FAIL_CLOSED = 1
_EXIT_USAGE = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_v4_tree_visualization",
        description="Render a read-only V4 tree-segmentation review visualization.",
    )
    parser.add_argument("--v4-root", required=True, type=Path,
                        help="Published validated V4 tree-segmentation artifact root.")
    parser.add_argument("--output-html", required=True, type=Path,
                        help="Self-contained Plotly review HTML output path.")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="Deterministic review summary JSON output path.")
    parser.add_argument("--max-points-per-tree", type=int,
                        default=DEFAULT_MAX_POINTS_PER_TREE)
    parser.add_argument("--max-review-points", type=int, default=DEFAULT_MAX_REVIEW_POINTS)
    parser.add_argument("--max-unassigned-points", type=int,
                        default=DEFAULT_MAX_UNASSIGNED_POINTS)
    parser.add_argument("--max-conflict-points", type=int,
                        default=DEFAULT_MAX_CONFLICT_POINTS)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing output files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = run_v4_tree_visualization(
            args.v4_root,
            args.output_html,
            args.summary_json,
            max_points_per_tree=args.max_points_per_tree,
            max_review_points=args.max_review_points,
            max_unassigned_points=args.max_unassigned_points,
            max_conflict_points=args.max_conflict_points,
            overwrite=args.overwrite,
        )
        print(f"V4 tree visualization published: {output.html_path}")
        if output.summary_path is not None:
            print(f"summary: {output.summary_path}")
        print(
            f"trees={len(output.summary.get('trees', []))} "
            f"displayed_assigned={output.summary.get('displayed_counts', {}).get('assigned', 0)}"
        )
        return _EXIT_OK
    except V4TreeVisualizationOutputError as exc:
        print(f"V4_VISUALIZATION_OUTPUT_EXISTS: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except V4TreeVisualizationError as exc:
        print(f"V4_VISUALIZATION_FAILED: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except (ValueError, TypeError, OSError) as exc:
        print(f"V4_VISUALIZATION_INVALID: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED


if __name__ == "__main__":
    sys.exit(main())
