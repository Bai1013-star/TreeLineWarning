"""Run the isolated FAST vegetation V3 admission path from a V2 evidence root.

The command reads formal assets and the supplied V2 root only.  It writes a
new non-certified V3 root, keeps V4 and TreeWire untouched, and never claims
formal FGC/FRV authority.
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
    NON_CERTIFIED_MARKER,
    FastGroundResult,
    FastResearchError,
    FastVegetationInputResult,
    prepare_w1_reuse_snapshot,
    run_fast_v4,
    sha256_file,
)
from algorithms.vegetation.fast_vegetation_v3 import (  # noqa: E402
    FAST_V3_SCHEMA_VERSION,
    FastV3DiagnosticState,
    build_fast_vegetation_v3_input,
    generate_fast_vegetation_review_v3,
    generate_minimum_clearance_local_review_v3,
)
from algorithms.vegetation.tree_wire_clearance import run_tree_wire_clearance  # noqa: E402
from algorithms.vegetation.tree_wire_clearance_visualization import (  # noqa: E402
    run_tree_wire_clearance_visualization,
)
from algorithms.vegetation.v4_tree_artifacts import load_v4_tree_segmentation_artifact  # noqa: E402
from algorithms.vegetation.v4_tree_visualization import run_v4_tree_visualization  # noqa: E402
from algorithms.wires.reconstruct import load_wire_models  # noqa: E402
from core.contracts.tree_wire_clearance import TreeWireClearanceResult  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Read-only source LAS.")
    parser.add_argument("--output", type=Path, required=True, help="New isolated V3 artifact root.")
    parser.add_argument(
        "--v2-root", type=Path, required=True, help="Read-only V2 FAST artifact root."
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--w1-npz", type=Path, required=True)
    parser.add_argument("--w1-json", type=Path, required=True)
    parser.add_argument("--w1-project-root", type=Path, required=True)
    return parser.parse_args(argv)


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


def _load_v2_context(v2_root: Path) -> tuple[FastGroundResult, FastVegetationInputResult, Path]:
    try:
        ground = FastGroundResult.from_dict(
            json.loads((v2_root / "ground" / "ground_hag_result.json").read_text("utf-8"))
        )
        admission = FastVegetationInputResult.from_dict(
            json.loads((v2_root / "vegetation_admission" / "fast_v4_input.json").read_text("utf-8"))
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FastResearchError("FAST_V3_V2_CONTEXT_INVALID") from exc
    feature_path = v2_root / "vegetation" / "plan_b_geometry_features.npz"
    if not feature_path.is_file():
        raise FastResearchError("FAST_V3_V2_FEATURE_MISSING")
    return ground, admission, feature_path


def _assert_pre_v4_gate(manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    gate = manifest["machine_pre_v4_gate"]
    if (
        gate.get("status") != "PASS"
        or counts["final_v3_vegetation_retained"] <= 0
        or gate["positive_v1_retained_fraction"] < 0.5
        or any(
            counts[name] != 0
            for name in (
                "ground_overlap",
                "exact_known_wire_overlap",
                "wire_geometry_overlap",
                "rigid_linear_overlap",
                "rigid_surface_overlap",
            )
        )
    ):
        raise FastResearchError("FAST_V3_PRE_V4_MACHINE_GATE_BLOCKED")


def _v4_excluded_overlap(diagnostics_npz: Path, v4_original: np.ndarray) -> dict[str, int]:
    with np.load(diagnostics_npz, allow_pickle=False) as stream:
        original = np.asarray(stream["original_index"], dtype=np.int64)
        state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
    positions = np.searchsorted(original, v4_original)
    if np.any(positions >= len(original)) or not np.array_equal(original[positions], v4_original):
        raise FastResearchError("FAST_V3_V4_ORIGINAL_INDEX_CORRESPONDENCE_INVALID")
    return {
        "ground": int(
            np.count_nonzero(state[positions] == int(FastV3DiagnosticState.GROUND_EXCLUDED))
        ),
        "exact_wire": int(
            np.count_nonzero(state[positions] == int(FastV3DiagnosticState.EXACT_WIRE_EXCLUDED))
        ),
        "wire_geometry": int(
            np.count_nonzero(state[positions] == int(FastV3DiagnosticState.WIRE_GEOMETRY_EXCLUDED))
        ),
        "rigid_linear": int(
            np.count_nonzero(state[positions] == int(FastV3DiagnosticState.RIGID_LINEAR_EXCLUDED))
        ),
        "rigid_surface": int(
            np.count_nonzero(state[positions] == int(FastV3DiagnosticState.RIGID_SURFACE_EXCLUDED))
        ),
    }


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    input_las, output, v2_root = args.input.resolve(), args.output.resolve(), args.v2_root.resolve()
    if not input_las.is_file() or not v2_root.is_dir() or output.exists() and not output.is_dir():
        raise FastResearchError("FAST_V3_INPUT_OR_OUTPUT_INVALID")
    started = time.perf_counter()
    stages: dict[str, dict[str, Any]] = {}
    ground, admission, v2_feature = _load_v2_context(v2_root)
    if admission.source_id != args.source_id or Path(admission.source_las_path) != input_las:
        raise FastResearchError("FAST_V3_SOURCE_ID_OR_INPUT_MISMATCH")
    if sha256_file(input_las) != admission.source_file_sha256:
        raise FastResearchError("FAST_V3_INPUT_SHA_MISMATCH")

    stage_started = time.perf_counter()
    w1_status, w1_paths = prepare_w1_reuse_snapshot(
        args.w1_npz, args.w1_json, args.w1_project_root, output / "w1"
    )
    w1_result, _w1_arrays = load_wire_models(w1_paths["w1_npz"], w1_paths["w1_json"])
    if w1_result.source_id != args.source_id:
        raise FastResearchError("FAST_V3_W1_SOURCE_ID_MISMATCH")
    stages["w1"] = {
        "status": w1_status,
        "reuse": "EXACT_READ_ONLY_SNAPSHOT",
        "runtime_seconds": time.perf_counter() - stage_started,
    }
    stage_started = time.perf_counter()
    vegetation_status, vegetation, diagnostics = build_fast_vegetation_v3_input(
        admission,
        ground,
        output / "vegetation",
        v2_feature_npz=v2_feature,
        w1_npz=w1_paths["w1_npz"],
        w1_json=w1_paths["w1_json"],
        reference_a2a_npz=(
            args.w1_project_root.resolve()
            / "outputs"
            / "features"
            / f"{args.source_id}_geometry_features.npz"
        ),
        reference_v1_npz=(
            args.w1_project_root.resolve()
            / "outputs"
            / "vegetation_candidates"
            / f"{args.source_id}_vegetation_candidates.npz"
        ),
    )
    stages["vegetation_v3"] = {
        "status": vegetation_status,
        "runtime_seconds": diagnostics["runtime_seconds"],
        "invocation_seconds": time.perf_counter() - stage_started,
        "counts": diagnostics["counts"],
    }
    preview = generate_fast_vegetation_review_v3(
        diagnostics, output / "visualizations" / "fast_vegetation_review_v3.html"
    )
    stages["pre_v4_review"] = {"status": "PASS", "html": str(preview)}
    _assert_pre_v4_gate(diagnostics)

    with laspy.open(input_las) as source:
        source_min = tuple(float(value) for value in source.header.mins)
        source_max = tuple(float(value) for value in source.header.maxs)
    stage_started = time.perf_counter()
    v4_status, v4_root, v4_summary = run_fast_v4(
        vegetation,
        output / "v4",
        source_bounds_min_xyz=source_min,
        source_bounds_max_xyz=source_max,
    )
    adapter = json.loads((output / "v4" / "fast_v4_adapter.json").read_text("utf-8"))
    stages["v4"] = {
        "status": v4_status,
        "runtime_seconds": adapter.get("elapsed_seconds", 0.0),
        "invocation_seconds": time.perf_counter() - stage_started,
        **v4_summary,
        "unassigned_point_count": v4_summary["point_count"] - v4_summary["assigned_point_count"],
    }
    v4_html = output / "visualizations" / "v4_review_v3.html"
    v4_summary_json = output / "visualizations" / "v4_review_v3.json"
    if not v4_html.is_file() or not v4_summary_json.is_file():
        run_v4_tree_visualization(v4_root, v4_html, v4_summary_json, overwrite=False)

    loaded_v4 = load_v4_tree_segmentation_artifact(v4_root)
    overlap = _v4_excluded_overlap(
        Path(diagnostics["diagnostics_npz_path"]),
        np.asarray(loaded_v4.result.original_index, dtype=np.int64),
    )
    if any(overlap.values()):
        raise FastResearchError("FAST_V3_EXCLUDED_STATE_REACHED_V4")

    clearance_path = output / "clearance" / "tree_wire_clearance.json"
    stage_started = time.perf_counter()
    clearance = _clearance_from_cache(clearance_path, loaded_v4.result.result_digest)
    if clearance is None:
        clearance = run_tree_wire_clearance(
            v4_root, w1_paths["w1_npz"], w1_paths["w1_json"], w1_paths["project_root"]
        )
        _write_json(clearance_path, clearance.to_dict())
        clearance_status = "PASS"
    else:
        clearance_status = "SKIPPED_FROM_CACHE"
    valid = [
        record.minimum_clearance_m
        for record in clearance.records
        if record.minimum_clearance_m is not None
    ]
    minimum = min(valid) if valid else None
    stages["tree_wire_clearance"] = {
        "status": clearance_status,
        "tree_count": len(clearance.records),
        "minimum_clearance_m": minimum,
        "runtime_seconds": time.perf_counter() - stage_started,
    }
    clearance_html = output / "visualizations" / "clearance_review_v3.html"
    clearance_summary_json = output / "visualizations" / "clearance_review_v3.json"
    if not clearance_html.is_file() or not clearance_summary_json.is_file():
        run_tree_wire_clearance_visualization(
            clearance_path,
            v4_root,
            w1_paths["w1_npz"],
            w1_paths["w1_json"],
            clearance_html,
            clearance_summary_json,
            overwrite=False,
        )
    local_html, minimum_diagnostic = generate_minimum_clearance_local_review_v3(
        clearance, diagnostics, output / "visualizations" / "minimum_clearance_local_review_v3.html"
    )
    diagnostics["minimum_clearance_diagnostic"] = minimum_diagnostic
    diagnostics["v4_excluded_state_overlap"] = overlap
    _write_json(output / "vegetation" / "fast_v3_diagnostics.json", diagnostics)
    summary = {
        "schema_version": FAST_V3_SCHEMA_VERSION,
        "non_certified_marker": NON_CERTIFIED_MARKER,
        "overall_status": "PASS",
        "formal_pipeline_modified": False,
        "formal_authority_claimed": False,
        "v4_semantics_changed": False,
        "w1_semantics_changed": False,
        "tree_wire_semantics_changed": False,
        "input_las": str(input_las),
        "input_las_sha256": sha256_file(input_las),
        "v2_root": str(v2_root),
        "output_root": str(output),
        "stages": stages,
        "counts": diagnostics["counts"],
        "v4_excluded_state_overlap": overlap,
        "minimum_clearance_diagnostic": minimum_diagnostic,
        "minimum_clearance_status": "SUSPICIOUS"
        if minimum is not None and minimum < 0.01
        else "PLAUSIBLE",
        "outputs": {
            "fast_vegetation_review_v3": str(preview),
            "v4_review_v3": str(v4_html),
            "clearance_review_v3": str(clearance_html),
            "minimum_clearance_local_review_v3": str(local_html),
            "fast_v3_diagnostics": str(output / "vegetation" / "fast_v3_diagnostics.json"),
        },
        "total_runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output / "fast_v3_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(argv)
    except (FastResearchError, OSError, ValueError, TypeError) as exc:
        print(f"FAST_V3_PIPELINE_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
