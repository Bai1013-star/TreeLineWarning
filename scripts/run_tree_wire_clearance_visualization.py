"""Tree-wire clearance review visualization CLI (FINAL-TREE-DELIVERY).

Usage::

    E:\\Miniforge\\envs\\treeline\\python.exe scripts/run_tree_wire_clearance_visualization.py \\
        --clearance-json <clearance-result.json> \\
        --v4-root <published-v4-artifact-root> \\
        --w1-npz <w1-wire-models.npz> \\
        --w1-json <w1-wire-models.json> \\
        --output-html <review.html> \\
        [--summary-json <review.json>] \\
        [--max-points-per-tree 10000] \\
        [--overwrite]

The viewer reads only a validated clearance result plus the validated V4/W1
material needed for display and never recomputes any distance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from algorithms.vegetation.tree_wire_clearance_visualization import (
    DEFAULT_MAX_POINTS_PER_TREE,
    TreeWireClearanceVisualizationError,
    TreeWireClearanceVisualizationOutputError,
    run_tree_wire_clearance_visualization,
)

_EXIT_OK = 0
_EXIT_FAIL_CLOSED = 1
_EXIT_USAGE = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_tree_wire_clearance_visualization",
        description="Render a read-only tree-wire clearance review visualization.",
    )
    parser.add_argument("--clearance-json", required=True, type=Path,
                        help="Published digest-bound clearance result JSON.")
    parser.add_argument("--v4-root", required=True, type=Path,
                        help="Published validated V4 tree-segmentation artifact root.")
    parser.add_argument("--w1-npz", required=True, type=Path,
                        help="W1 wire-model NPZ output.")
    parser.add_argument("--w1-json", required=True, type=Path,
                        help="W1 wire-model JSON result.")
    parser.add_argument("--output-html", required=True, type=Path,
                        help="Self-contained Plotly review HTML output path.")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="Deterministic review summary JSON output path.")
    parser.add_argument("--max-points-per-tree", type=int,
                        default=DEFAULT_MAX_POINTS_PER_TREE)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing output files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = run_tree_wire_clearance_visualization(
            args.clearance_json,
            args.v4_root,
            args.w1_npz,
            args.w1_json,
            args.output_html,
            args.summary_json,
            max_points_per_tree=args.max_points_per_tree,
            overwrite=args.overwrite,
        )
        print(f"Tree-wire clearance visualization published: {output.html_path}")
        if output.summary_path is not None:
            print(f"summary: {output.summary_path}")
        print(
            f"status={output.summary.get('status', '')} "
            f"trees={output.summary.get('displayed_counts', {}).get('trees', 0)}"
        )
        return _EXIT_OK
    except TreeWireClearanceVisualizationOutputError as exc:
        print(f"CLEARANCE_VISUALIZATION_OUTPUT_EXISTS: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except TreeWireClearanceVisualizationError as exc:
        print(f"CLEARANCE_VISUALIZATION_FAILED: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED
    except (ValueError, TypeError, OSError) as exc:
        print(f"CLEARANCE_VISUALIZATION_INVALID: {exc}", file=sys.stderr)
        return _EXIT_FAIL_CLOSED


if __name__ == "__main__":
    sys.exit(main())
