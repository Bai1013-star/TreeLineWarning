"""Run the completed V3 pipeline and emit read-only scene visualization files."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.vegetation.canopy_height_model import (  # noqa: E402
    CanopyHeightModelError,
    build_canopy_height_model,
)
from algorithms.vegetation.canopy_segmentation import (  # noqa: E402
    CanopySegmentationError,
    segment_canopy_regions,
)
from algorithms.vegetation.tree_inputs import (  # noqa: E402
    TreeSegmentationInputError,
    load_tree_segmentation_input_set,
)
from algorithms.vegetation.tree_segmentation import (  # noqa: E402
    TreeResultConstructionError,
    build_tree_segmentation_result,
)
from algorithms.vegetation.tree_visualization import (  # noqa: E402
    build_summary,
    render_3d_html,
    render_diagnostic_png,
    render_topdown_png,
    topdown_pixel_statistics,
    validate_input_result_alignment,
    validate_maximum_3d_points,
    write_summary_json,
)
from algorithms.vegetation.treetop_candidates import (  # noqa: E402
    TreetopDetectionError,
    detect_treetop_candidates,
)


class VisualizationCliError(RuntimeError):
    """Expected user-facing invocation failure for the V3 visualization runner."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve(strict=False)
    visualization_root = (_REPOSITORY_ROOT / "outputs" / "visualizations").resolve(strict=False)
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=False)
    if not (
        _is_relative_to(resolved, visualization_root) or _is_relative_to(resolved, temporary_root)
    ):
        raise VisualizationCliError(
            "--output-dir must be under outputs/visualizations or the system temporary directory"
        )


def _parse_maximum_3d_points(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise VisualizationCliError(
            "--max-3d-points must be an integer from 1 through 200000"
        ) from error
    try:
        return validate_maximum_3d_points(parsed)
    except ValueError as error:
        raise VisualizationCliError(f"--max-3d-points {error}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-npz", type=Path, required=True)
    parser.add_argument("--v2-json", type=Path, required=True)
    parser.add_argument("--a2a-npz", type=Path, required=True)
    parser.add_argument("--a2a-json", type=Path, required=True)
    parser.add_argument("--a2b-npz", type=Path, required=True)
    parser.add_argument("--a2b-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-3d-points", default="200000")
    args = parser.parse_args(argv)
    args.max_3d_points = _parse_maximum_3d_points(args.max_3d_points)
    return args


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Run V3B1 through V3B4B exactly once, then write visualization-only outputs."""
    _validate_output_dir(args.output_dir)
    try:
        maximum_3d_points = validate_maximum_3d_points(args.max_3d_points)
    except ValueError as error:
        raise VisualizationCliError(f"--max-3d-points {error}") from error
    tree_input = load_tree_segmentation_input_set(
        args.v2_npz, args.v2_json, args.a2a_npz, args.a2a_json, args.a2b_npz, args.a2b_json
    )
    canopy_model = build_canopy_height_model(tree_input)
    candidates = detect_treetop_candidates(canopy_model)
    segmentation = segment_canopy_regions(canopy_model, candidates)
    result = build_tree_segmentation_result(tree_input, canopy_model, candidates, segmentation)
    try:
        validate_input_result_alignment(tree_input, result)
    except ValueError as error:
        raise VisualizationCliError(str(error)) from error

    output_dir = args.output_dir
    paths = {
        "topdown_png": output_dir / "tree_segmentation_topdown.png",
        "diagnostic_png": output_dir / "tree_segmentation_diagnostic.png",
        "interactive_html": output_dir / "tree_segmentation_3d.html",
        "summary_json": output_dir / "tree_segmentation_summary.json",
    }
    try:
        render_topdown_png(tree_input, result, paths["topdown_png"])
        topdown_statistics = topdown_pixel_statistics(tree_input, result)
    except ValueError as error:
        raise VisualizationCliError(str(error)) from error
    render_diagnostic_png(canopy_model, candidates, segmentation, paths["diagnostic_png"])
    _, displayed_count = render_3d_html(
        tree_input, result, paths["interactive_html"], maximum_points=maximum_3d_points
    )
    source_paths = {
        "v2_npz": args.v2_npz,
        "v2_json": args.v2_json,
        "a2a_npz": args.a2a_npz,
        "a2a_json": args.a2a_json,
        "a2b_npz": args.a2b_npz,
        "a2b_json": args.a2b_json,
    }
    summary = build_summary(
        scene_name=output_dir.name,
        source_artifact_paths=source_paths,
        tree_input=tree_input,
        canopy_model=canopy_model,
        candidates=candidates,
        segmentation=segmentation,
        result=result,
        maximum_3d_points=maximum_3d_points,
        displayed_3d_points=displayed_count,
        output_paths=paths,
        topdown_statistics=topdown_statistics,
    )
    write_summary_json(summary, paths["summary_json"])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        paths = run(args)
    except (
        VisualizationCliError,
        TreeSegmentationInputError,
        CanopyHeightModelError,
        TreetopDetectionError,
        CanopySegmentationError,
        TreeResultConstructionError,
        OSError,
    ) as error:
        print(f"V3 visualization failed: {error}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
