"""Run active project spans through the canonical non-authoritative V2 shadow path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from algorithms.poles.topology import SpanRecord

from scripts.run_wire_v2_shadow import (
    ShadowExecutionError,
    _load_wire_region,
    _validate_shadow_artifacts,
    execute_shadow_run,
)


def _latest_generation(project_root: Path) -> Path:
    generations = sorted((project_root / "state" / "generations").iterdir())
    if not generations:
        raise RuntimeError("project has no persisted generation")
    return generations[-1]


def _source_point_count(project_root: Path) -> int:
    shared = project_root / "artifacts" / "project-runs" / "source-shared"
    preparations = sorted(shared.glob("*/preparation.json"))
    if len(preparations) != 1:
        raise RuntimeError(f"expected one shared preparation artifact, found {len(preparations)}")
    return int(json.loads(preparations[0].read_text(encoding="utf-8"))["source_point_count"])


def _span_context(
    project_root: Path, group: str, record: dict[str, object]
) -> tuple[Path, SpanRecord, tuple[float, float, float]]:
    span_id = str(record["span_id"])
    run_root = (
        project_root
        / "artifacts"
        / "project-runs"
        / "groups"
        / group
        / span_id
        / str(record["active_run_id"])
    )
    if not run_root.is_dir():
        raise FileNotFoundError(f"active run does not exist: {run_root}")
    generation = _latest_generation(project_root)
    spans = json.loads((generation / "spans" / "spans.json").read_text(encoding="utf-8"))["records"]
    span_record = next(item for item in spans if item["span_id"] == span_id)
    towers = json.loads((generation / "towers" / "towers.json").read_text(encoding="utf-8"))[
        "records"
    ]
    tower_by_id = {item["tower_id"]: item["revisions"][-1]["refined_geometry"] for item in towers}
    start_tower = tower_by_id[span_record["from_tower_id"]]
    end_tower = tower_by_id[span_record["to_tower_id"]]
    start = np.asarray(start_tower["base_xyz_absolute_m"], dtype=np.float64)
    end = np.asarray(end_tower["base_xyz_absolute_m"], dtype=np.float64)
    direction = np.asarray(span_record["geometry"]["direction_xy"], dtype=np.float64)
    with np.load(run_root / "wire" / "span_wire_diagnostics.npz", allow_pickle=False) as stream:
        fine = np.asarray(stream["fine_xyz_local_m"], dtype=np.float64)
        s = np.asarray(stream["s_m"], dtype=np.float64)
        u = np.asarray(stream["u_m"], dtype=np.float64)
        z = np.asarray(stream["z_m"], dtype=np.float64)
    perpendicular = np.asarray((-direction[1], direction[0]))
    local_start = np.median(
        fine[:, :2] - s[:, None] * direction - u[:, None] * perpendicular, axis=0
    )
    origin = (
        float(start[0] - local_start[0]),
        float(start[1] - local_start[1]),
        float(np.median(z - fine[:, 2])),
    )
    region = _load_wire_region(run_root)
    span = SpanRecord(
        line_id=0,
        line_name="Shadow",
        span_id=0,
        span_name=span_id,
        start_pole_candidate_id=0,
        end_pole_candidate_id=1,
        start_xyz_absolute_m=tuple(start),
        end_xyz_absolute_m=tuple(end),
        start_pole_top_z_m=float(start_tower["top_xyz_absolute_m"][2]),
        end_pole_top_z_m=float(end_tower["top_xyz_absolute_m"][2]),
        length_m=float(span_record["geometry"]["length_m"]),
        direction_xy=(float(direction[0]), float(direction[1]), 0.0),
        half_width_m=9.0,
        halo_m=6.0,
        core_bounds_min_xyz=region.bounds_min_xyz,
        core_bounds_max_xyz=region.bounds_max_xyz,
        halo_bounds_min_xyz=region.bounds_min_xyz,
        halo_bounds_max_xyz=region.bounds_max_xyz,
        quality="SHADOW_ONLY",
    )
    return run_root, span, origin


def _shadow_span(
    project_root: Path,
    shadow_root: Path,
    group: str,
    record: dict[str, object],
) -> dict[str, object]:
    span_id = str(record["span_id"])
    started = perf_counter()
    try:
        run_root, span, origin = _span_context(project_root, group, record)
        old = json.loads((run_root / "clearance.json").read_text(encoding="utf-8"))
        summary = execute_shadow_run(
            run_root=run_root,
            final_output_dir=shadow_root / span_id,
            span=span,
            origin_xyz=origin,
            authoritative_source_point_count=_source_point_count(project_root),
        )
        return {
            "span_id": span_id,
            "execution_status": "COMPLETE",
            "source_run": str(run_root),
            "old_distance_ready_wire_count": int(old["diagnostics"]["distance_ready_segments"]),
            "old_minimum_clearance_m": old["global_minimum_clearance_m"],
            "runtime_s": perf_counter() - started,
            "v2": summary,
        }
    except ShadowExecutionError as exc:
        return {
            "span_id": span_id,
            "execution_status": "FAILED",
            "runtime_s": perf_counter() - started,
            "failure": {
                "exception_type": type(exc.reason).__name__,
                "exception_message": str(exc.reason),
                "temporary_output_dir": str(exc.temporary_dir),
            },
        }
    except Exception as exc:
        return {
            "span_id": span_id,
            "execution_status": "FAILED",
            "runtime_s": perf_counter() - started,
            "failure": {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "temporary_output_dir": None,
            },
        }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _project_summary_status(results: list[dict[str, object]]) -> str:
    statuses = {str(item["execution_status"]) for item in results}
    if statuses == {"COMPLETE"}:
        return "COMPLETE"
    if "FAILED" in statuses:
        return "FAILED"
    return "RUNNING"


def _write_project_progress(shadow_root: Path, results: list[dict[str, object]]) -> None:
    _atomic_write_json(
        shadow_root / "project_shadow_summary.json",
        {
            "marker": "WIRE_V2_PROJECT_SHADOW_NON_AUTHORITATIVE",
            "execution_status": _project_summary_status(results),
            "span_results": results,
        },
    )


def _published_complete_span(
    project_root: Path,
    shadow_root: Path,
    group: str,
    record: dict[str, object],
) -> dict[str, object] | None:
    """Return a prior atomically-published result, never an incomplete directory."""
    span_id = str(record["span_id"])
    output = shadow_root / span_id
    if not output.is_dir():
        return None
    _validate_shadow_artifacts(output)
    summary = json.loads((output / "shadow_summary.json").read_text(encoding="utf-8"))
    if summary.get("execution_status") != "COMPLETE":
        raise RuntimeError(f"published shadow directory is not COMPLETE: {output}")
    run_root, _, _ = _span_context(project_root, group, record)
    old = json.loads((run_root / "clearance.json").read_text(encoding="utf-8"))
    return {
        "span_id": span_id,
        "execution_status": "COMPLETE",
        "source_run": str(run_root),
        "old_distance_ready_wire_count": int(old["diagnostics"]["distance_ready_segments"]),
        "old_minimum_clearance_m": old["global_minimum_clearance_m"],
        "runtime_s": 0.0,
        "resumed_published_output": True,
        "v2": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("cache/tower_ordered_project"))
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument(
        "--span-id",
        action="append",
        help="resume only this span while preserving other journaled results",
    )
    args = parser.parse_args()
    index = json.loads(
        (args.project_root / "artifacts" / "project-runs" / "project-result-index.json").read_text(
            encoding="utf-8"
        )
    )
    records = [
        (group, record)
        for group, group_records in index["groups"].items()
        for record in group_records
    ]
    summary_path = args.shadow_root / "project_shadow_summary.json"
    existing_by_span: dict[str, dict[str, object]] = {}
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_by_span = {str(item["span_id"]): item for item in existing.get("span_results", [])}
    summaries: list[dict[str, object]] = [
        existing_by_span.get(
            str(record["span_id"]),
            {"span_id": str(record["span_id"]), "execution_status": "PENDING"},
        )
        for _, record in records
    ]
    requested = set(args.span_id or [str(record["span_id"]) for _, record in records])
    known = {str(record["span_id"]) for _, record in records}
    unknown = requested - known
    if unknown:
        parser.error(f"unknown span id(s): {', '.join(sorted(unknown))}")
    _write_project_progress(args.shadow_root, summaries)
    for position, (group, record) in enumerate(records):
        if str(record["span_id"]) not in requested:
            continue
        try:
            summary = _published_complete_span(args.project_root, args.shadow_root, group, record)
            summaries[position] = summary or _shadow_span(
                args.project_root, args.shadow_root, group, record
            )
        except Exception as exc:
            summaries[position] = {
                "span_id": str(record["span_id"]),
                "execution_status": "FAILED",
                "failure": {"exception_type": type(exc).__name__, "exception_message": str(exc)},
            }
        _write_project_progress(args.shadow_root, summaries)


if __name__ == "__main__":
    main()
