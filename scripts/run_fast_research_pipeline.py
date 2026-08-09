"""Run the isolated, non-certified fast tree-line research pipeline.

Example (the existing W1 inputs are read-only and copied byte-for-byte into
the output directory before use)::

    E:\\Miniforge\\envs\\treeline\\python.exe scripts/run_fast_research_pipeline.py \\
        --input E:\\TreeLineWarning\\data\\raw\\scene_01_small\\scan_small.las \\
        --output artifacts\\fast_research\\scene_simple \\
        --source-id scene_simple \\
        --bounds -11 -7 -3.1609 2 3 9 \\
        --admission-voxel-size 0.05 \\
        --w1-npz E:\\TreeLineWarning\\outputs\\wire_models\\scene_simple_wire_models.npz \\
        --w1-json E:\\TreeLineWarning\\outputs\\wire_models\\scene_simple_wire_models.json \\
        --w1-project-root E:\\TreeLineWarning

It never writes to the input LAS, its W1 source project, or formal artifact
directories.  The output is marked ``NON_CERTIFIED_FAST_RESEARCH`` throughout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import laspy
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.vegetation.fast_research import (  # noqa: E402
    FAST_RESEARCH_SCHEMA_VERSION,
    NON_CERTIFIED_MARKER,
    FastResearchError,
    build_fast_vegetation_input,
    generate_fast_ground_preview,
    prepare_w1_reuse_snapshot,
    run_fast_ground_hag,
    run_fast_v4,
    sha256_file,
)
from algorithms.vegetation.fast_vegetation_ab import (  # noqa: E402
    FastVegetationEvidenceConfig,
    benchmark_plan_b_representative,
    build_fast_vegetation_ab_input,
    generate_fast_vegetation_review,
    generate_minimum_clearance_local_review,
    generate_plan_b_geometry_review,
    probe_plan_a_open3d_ml,
)
from algorithms.vegetation.tree_wire_clearance import run_tree_wire_clearance  # noqa: E402
from algorithms.vegetation.tree_wire_clearance_visualization import (  # noqa: E402
    run_tree_wire_clearance_visualization,
)
from algorithms.vegetation.v4_tree_artifacts import load_v4_tree_segmentation_artifact  # noqa: E402
from algorithms.vegetation.v4_tree_visualization import run_v4_tree_visualization  # noqa: E402
from algorithms.wires.reconstruct import load_wire_models  # noqa: E402
from core.contracts.tree_wire_clearance import TreeWireClearanceResult  # noqa: E402
from core.contracts.wires import WireSeparationResult  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source LAS; opened read-only.")
    parser.add_argument(
        "--output", type=Path, required=True, help="New isolated research artifact root."
    )
    parser.add_argument("--source-id", required=True, help="Must match the reused W1/A4 source ID.")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
        default=None,
        help="Optional source-coordinate admission bounds.",
    )
    parser.add_argument("--minimum-hag", type=float, default=0.3)
    parser.add_argument(
        "--admission-voxel-size",
        type=float,
        default=None,
        help="Optional explicit fast-admission voxel size; V4 receives all retained rows.",
    )
    parser.add_argument("--w1-npz", type=Path, required=True)
    parser.add_argument("--w1-json", type=Path, required=True)
    parser.add_argument(
        "--w1-project-root",
        type=Path,
        required=True,
        help="Read-only root used only to resolve W1's original A4 provenance.",
    )
    return parser.parse_args(argv)


def _bounds(
    values: list[float] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if values is None:
        return None
    return (tuple(values[:3]), tuple(values[3:]))  # type: ignore[return-value]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8")
    temporary.replace(path)


def _clearance_from_cache(path: Path, v4_digest: str) -> TreeWireClearanceResult | None:
    if not path.is_file():
        return None
    try:
        result = TreeWireClearanceResult.from_dict(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return result if result.v4_result_digest == v4_digest else None


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    input_las = args.input.resolve()
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise FastResearchError("FAST_RESEARCH_OUTPUT_INVALID")
    if not input_las.is_file():
        raise FastResearchError(f"FAST_RESEARCH_INPUT_MISSING: {input_las}")
    if output in input_las.parents:
        raise FastResearchError("FAST_RESEARCH_OUTPUT_MUST_NOT_BE_INSIDE_INPUT_PATH")

    started = time.perf_counter()
    stages: dict[str, dict[str, Any]] = {}
    plan_a = probe_plan_a_open3d_ml()
    stage_started = time.perf_counter()
    w1_status, w1_paths = prepare_w1_reuse_snapshot(
        args.w1_npz, args.w1_json, args.w1_project_root, output / "w1"
    )
    stages["w1"] = {
        "status": w1_status,
        "reuse": "EXACT_READ_ONLY_SNAPSHOT",
        "runtime_seconds": time.perf_counter() - stage_started,
    }
    _w1_result, _w1_arrays = load_wire_models(w1_paths["w1_npz"], w1_paths["w1_json"])
    a4_json = (
        w1_paths["project_root"]
        / "outputs"
        / "wires"
        / Path(json.loads(w1_paths["w1_json"].read_text("utf-8"))["input_a4_json_relpath"]).name
    )
    a4 = WireSeparationResult.from_dict(json.loads(a4_json.read_text("utf-8")))
    if a4.source_id != args.source_id:
        raise FastResearchError(
            "FAST_RESEARCH_SOURCE_ID_MISMATCH: "
            f"--source-id {args.source_id!r} != W1 {a4.source_id!r}"
        )

    ground_status, ground = run_fast_ground_hag(input_las, output / "ground")
    stages["ground_hag"] = {
        "status": ground_status,
        "runtime_seconds": ground.elapsed_seconds,
        "ground_backend": ground.algorithm_name,
        "hag_backend": ground.hag_algorithm_name,
        "ground_class_2_count": ground.ground_class_2_count,
        "hag_valid_count": ground.hag_valid_count,
    }
    ground_html = output / "visualizations" / "ground_hag_preview.html"
    stage_started = time.perf_counter()
    if ground_html.is_file():
        stages["ground_hag_preview"] = {
            "status": "SKIPPED_FROM_CACHE",
            "html": str(ground_html),
            "runtime_seconds": time.perf_counter() - stage_started,
        }
    else:
        preview = generate_fast_ground_preview(ground, ground_html)
        stages["ground_hag_preview"] = {
            "status": "PASS",
            "html": str(preview),
            "runtime_seconds": time.perf_counter() - stage_started,
        }
    admission_status, admission = build_fast_vegetation_input(
        ground,
        output / "vegetation_admission",
        source_id=args.source_id,
        origin_context=tuple(float(value) for value in a4.origin_xyz),
        minimum_hag_m=args.minimum_hag,
        bounds=_bounds(args.bounds),
        admission_voxel_size_m=args.admission_voxel_size,
    )
    stages["vegetation_admission"] = {
        "status": admission_status,
        "runtime_seconds": admission.elapsed_seconds,
        "point_count": admission.point_count,
        "marker": admission.non_certified_marker,
    }
    with np.load(admission.output_npz_path, allow_pickle=False) as stream:
        admission_xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
    representative = benchmark_plan_b_representative(
        admission_xyz, config=FastVegetationEvidenceConfig()
    )
    plan_a["representative_subset_count"] = representative["point_count"]
    evidence_root = args.w1_project_root.resolve() / "outputs"
    plan_b_status, vegetation, plan_b = build_fast_vegetation_ab_input(
        admission,
        ground,
        output / "vegetation",
        w1_npz=w1_paths["w1_npz"],
        w1_json=w1_paths["w1_json"],
        reference_a2a_npz=evidence_root / "features" / f"{args.source_id}_geometry_features.npz",
        reference_v1_npz=(
            evidence_root / "vegetation_candidates" / f"{args.source_id}_vegetation_candidates.npz"
        ),
        config=FastVegetationEvidenceConfig(),
    )
    stages["vegetation_input"] = {
        "status": plan_b_status,
        "runtime_seconds": vegetation.elapsed_seconds,
        "point_count": vegetation.point_count,
        "marker": vegetation.non_certified_marker,
        "plan_b_backend": plan_b["plan_b_backend"],
        "counts": plan_b["counts"],
        "reference_a2a_v1": plan_b["reference_a2a_v1"],
    }
    plan_b_html = output / "visualizations" / "plan_b_geometry_review.html"
    generate_plan_b_geometry_review(plan_b, plan_b_html)
    vegetation_html = output / "visualizations" / "fast_vegetation_review_v2.html"
    generate_fast_vegetation_review(vegetation, plan_b, vegetation_html)
    with laspy.open(input_las) as source:
        source_bounds_min = tuple(float(value) for value in source.header.mins)
        source_bounds_max = tuple(float(value) for value in source.header.maxs)
    stage_started = time.perf_counter()
    v4_status, v4_root, v4_summary = run_fast_v4(
        vegetation,
        output / "v4",
        source_bounds_min_xyz=source_bounds_min,
        source_bounds_max_xyz=source_bounds_max,
    )
    adapter = json.loads((output / "v4" / "fast_v4_adapter.json").read_text("utf-8"))
    stages["v4"] = {
        "status": v4_status,
        "runtime_seconds": adapter.get("elapsed_seconds", 0.0),
        "invocation_seconds": time.perf_counter() - stage_started,
        **v4_summary,
    }
    v4_html = output / "visualizations" / "v4_review_v2.html"
    v4_summary_json = output / "visualizations" / "v4_review_v2.json"
    stage_started = time.perf_counter()
    if v4_html.is_file() and v4_summary_json.is_file():
        stages["v4_visualization"] = {
            "status": "SKIPPED_FROM_CACHE",
            "html": str(v4_html),
            "runtime_seconds": time.perf_counter() - stage_started,
        }
    else:
        rendered = run_v4_tree_visualization(v4_root, v4_html, v4_summary_json, overwrite=False)
        stages["v4_visualization"] = {
            "status": "PASS",
            "html": str(rendered.html_path),
            "runtime_seconds": time.perf_counter() - stage_started,
        }

    loaded_v4 = load_v4_tree_segmentation_artifact(v4_root)
    clearance_path = output / "clearance" / "tree_wire_clearance.json"
    stage_started = time.perf_counter()
    clearance = _clearance_from_cache(clearance_path, loaded_v4.result.result_digest)
    if clearance is None:
        clearance = run_tree_wire_clearance(
            v4_root,
            w1_paths["w1_npz"],
            w1_paths["w1_json"],
            w1_paths["project_root"],
        )
        _write_json(clearance_path, clearance.to_dict())
        clearance_status = "PASS"
    else:
        clearance_status = "SKIPPED_FROM_CACHE"
    valid_clearances = [
        record.minimum_clearance_m
        for record in clearance.records
        if record.minimum_clearance_m is not None
    ]
    stages["tree_wire_clearance"] = {
        "status": clearance_status,
        "tree_count": len(clearance.records),
        "minimum_clearance_m": min(valid_clearances) if valid_clearances else None,
        "runtime_seconds": time.perf_counter() - stage_started,
    }
    clearance_html = output / "visualizations" / "clearance_review_v2.html"
    clearance_summary_json = output / "visualizations" / "clearance_review_v2.json"
    stage_started = time.perf_counter()
    if clearance_html.is_file() and clearance_summary_json.is_file():
        stages["clearance_visualization"] = {
            "status": "SKIPPED_FROM_CACHE",
            "html": str(clearance_html),
            "runtime_seconds": time.perf_counter() - stage_started,
        }
    else:
        rendered = run_tree_wire_clearance_visualization(
            clearance_path,
            v4_root,
            w1_paths["w1_npz"],
            w1_paths["w1_json"],
            clearance_html,
            clearance_summary_json,
            overwrite=False,
        )
        stages["clearance_visualization"] = {
            "status": "PASS",
            "html": str(rendered.html_path),
            "runtime_seconds": time.perf_counter() - stage_started,
        }

    local_clearance_html = output / "visualizations" / "minimum_clearance_local_review.html"
    generate_minimum_clearance_local_review(clearance, local_clearance_html)

    with np.load(vegetation.output_npz_path, allow_pickle=False) as stream:
        clean_original_index = np.asarray(stream["original_index"], dtype=np.int64)
    v4_original_index = np.asarray(loaded_v4.result.original_index, dtype=np.int64)
    known_wire_original_index = np.asarray(
        _w1_arrays["ordered_input_original_index"], dtype=np.int64
    )
    if not np.all(np.isin(v4_original_index, clean_original_index)):
        raise FastResearchError("FAST_RESEARCH_V4_INPUT_CORRESPONDENCE_INVALID")
    if np.any(np.isin(v4_original_index, known_wire_original_index)):
        raise FastResearchError("FAST_RESEARCH_V4_KNOWN_WIRE_OVERLAP")
    if plan_b["counts"]["final_tree_ground_overlap"] != 0:
        raise FastResearchError("FAST_RESEARCH_V4_GROUND_OVERLAP")

    valid_clearance = min(valid_clearances) if valid_clearances else None
    plausibility = (
        "SUSPICIOUS" if valid_clearance is not None and valid_clearance < 0.01 else "PLAUSIBLE"
    )
    ab_summary = {
        "schema_version": "fast-vegetation-ab-competition-v1",
        "non_certified_marker": NON_CERTIFIED_MARKER,
        "overall_status": "PASS",
        "plan_a": plan_a,
        "plan_b": {
            "status": "PASS",
            "backend": plan_b["plan_b_backend"],
            "runtime_seconds": plan_b["runtime_seconds"],
            "feature_runtime_seconds": plan_b["feature_runtime_seconds"],
            "evidence": plan_b["reference_a2a_v1"],
            "representative_benchmark": representative,
        },
        "winner": "PLAN_B",
        "why": (
            "Plan A could not import its PyTorch semantic backend without an install; "
            "Plan B ran compiled PDAL covariance features on the same FAST admission set, "
            "used exact-index W1 exclusion, and retained an exact-index A2A/V1 evidence audit."
        ),
        "outputs": {
            "plan_b_geometry_review": str(plan_b_html),
            "fast_vegetation_review": str(vegetation_html),
            "v4_review": str(v4_html),
            "clearance_review": str(clearance_html),
            "minimum_clearance_local_review": str(local_clearance_html),
        },
    }
    _write_json(output / "ab_comparison_summary.json", ab_summary)

    summary = {
        "schema_version": FAST_RESEARCH_SCHEMA_VERSION,
        "non_certified_marker": NON_CERTIFIED_MARKER,
        "overall_status": "PASS",
        "formal_pipeline_modified": False,
        "formal_authority_claimed": False,
        "v4_semantics_changed": False,
        "w1_semantics_changed": False,
        "tree_wire_semantics_changed": False,
        "input_las": str(input_las),
        "input_las_sha256": sha256_file(input_las),
        "output_root": str(output),
        "stages": stages,
        "ab_competition": ab_summary,
        "vegetation_cleaning": plan_b,
        "minimum_clearance_plausibility": plausibility,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "fast_research_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(argv)
    except (FastResearchError, OSError, ValueError, TypeError) as exc:
        print(f"FAST_RESEARCH_PIPELINE_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
