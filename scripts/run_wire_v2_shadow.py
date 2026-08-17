"""Run V2 wire reconstruction against a read-only existing span run.

The command only reads ``<run-root>/regions/wire`` and writes the result below
the explicitly supplied non-authoritative shadow directory.  It never touches
the project's run index or accepted span artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from algorithms.poles.topology import SpanRecord
from algorithms.spans.clearance import _direct_nearest_edges
from algorithms.spans.wire import run_span_wire_pipeline
from algorithms.vegetation.coarse_to_fine import RawRegion
from algorithms.vegetation.tree_wire_clearance import build_wire_edge_bundle
from core.config.fast_baseline import default_fast_baseline_config

_REQUIRED_SHADOW_ARTIFACTS = (
    "span_wire_result.json",
    "span_wire_diagnostics.npz",
    "fine_representatives.npz",
    "w1_arrays.npz",
    "wire_v2_track_evidence.json",
    "wire_v2_track_evidence.npz",
    "clearance_v2_shadow.json",
    "shadow_summary.json",
)


class ShadowExecutionError(RuntimeError):
    """A shadow run failed before its result could be atomically published."""

    def __init__(self, span_name: str, temporary_dir: Path, reason: Exception) -> None:
        self.span_name = span_name
        self.temporary_dir = temporary_dir
        self.reason = reason
        super().__init__(f"{span_name}: {type(reason).__name__}: {reason}")


def _json_safe(value: object) -> object:
    """Convert diagnostics to portable JSON without inventing non-finite values."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _config_identity(config: object) -> dict[str, object]:
    payload = asdict(config) if is_dataclass(config) else vars(config)
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "type": f"{type(config).__module__}.{type(config).__qualname__}",
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _confirmed_source_identity(output_dir: Path) -> dict[str, object]:
    with np.load(output_dir / "span_wire_diagnostics.npz", allow_pickle=False) as stream:
        source = np.asarray(stream["confirmed_member_source_index"], dtype=np.int64)
    return {
        "count": int(source.size),
        "unique_count": int(np.unique(source).size),
        "minimum": int(source.min()) if source.size else None,
        "maximum": int(source.max()) if source.size else None,
    }


def _validate_shadow_artifacts(output_dir: Path) -> None:
    missing = [name for name in _REQUIRED_SHADOW_ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete shadow result; missing {', '.join(missing)}")


def execute_shadow_run(
    *,
    run_root: Path,
    final_output_dir: Path,
    span: SpanRecord,
    origin_xyz: tuple[float, float, float],
    authoritative_source_point_count: int,
) -> dict[str, object]:
    """Execute one V2 shadow run and atomically publish a complete result.

    ``final_output_dir`` must not already exist.  A failed run retains its
    uniquely named temporary directory with ``execution_failure.json`` for
    forensics, but never creates a final span directory.
    """
    if final_output_dir.exists():
        raise FileExistsError(f"refusing to overwrite shadow result: {final_output_dir}")
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.incomplete-", dir=final_output_dir.parent
        )
    )
    started = perf_counter()
    try:
        config = default_fast_baseline_config()
        region = _load_wire_region(run_root)
        result = run_span_wire_pipeline(
            span,
            region,
            config,
            origin_xyz=origin_xyz,
            output_dir=temporary_dir,
            authoritative_source_point_count=authoritative_source_point_count,
        )
        clearance = _exact_shadow_clearance(run_root, result, config, temporary_dir)
        (temporary_dir / "clearance_v2_shadow.json").write_text(
            json.dumps(_json_safe(clearance), indent=2, sort_keys=True), encoding="utf-8"
        )
        diagnostics = dict(result.diagnostics)
        # The wire pipeline reports the write location it was given.  Replace
        # temporary paths before publication so every persisted reference names
        # the atomically published shadow result rather than a vanished staging
        # directory.
        for key in ("diagnostics_npz_path", "fine_representatives_npz_path"):
            if key in diagnostics:
                diagnostics[key] = str(final_output_dir / Path(str(diagnostics[key])).name)
        tracks = diagnostics["v2_tracks"]
        summary = {
            "marker": "WIRE_V2_SHADOW_NON_AUTHORITATIVE",
            "execution_status": "COMPLETE",
            "span_id": span.span_name,
            "source_run": str(run_root),
            "source_wire_region": str(run_root / "regions" / "wire"),
            "source_point_count": authoritative_source_point_count,
            "v2_configuration": _config_identity(config),
            "runtime_s": perf_counter() - started,
            "candidate_track_count": len(tracks),
            "confirmed_wire_count": result.strand_count,
            "review_only_track_count": sum(item["status"] == "REVIEW_ONLY" for item in tracks),
            "rejected_track_count": sum(item["status"] == "REJECTED" for item in tracks),
            # A V2 W1 "segment" is a supported conductor model span.  Its
            # clearance geometry is the consecutive-point polyline, whose
            # actual edge count is returned by the unchanged exact engine.
            "distance_ready_segment_count": result.distance_ready_segments,
            "distance_ready_polyline_edge_count": clearance.get("distance_ready_edges", 0),
            "confirmed_member_source_identity": _confirmed_source_identity(temporary_dir),
            "diagnostics": diagnostics,
            "clearance": clearance,
        }
        (temporary_dir / "shadow_summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        _validate_shadow_artifacts(temporary_dir)
        try:
            temporary_dir.replace(final_output_dir)
        except PermissionError:
            # Windows reports an occupied destination as PermissionError for a
            # directory rename.  If another V2 shadow worker won the race and
            # published a complete result, consume that verified result rather
            # than converting it into a false failure.
            if final_output_dir.is_dir():
                _validate_shadow_artifacts(final_output_dir)
                published = json.loads(
                    (final_output_dir / "shadow_summary.json").read_text(encoding="utf-8")
                )
                if published.get("execution_status") == "COMPLETE":
                    return _json_safe(published)
            raise
        return _json_safe(summary)
    except Exception as exc:
        failure = {
            "marker": "WIRE_V2_SHADOW_NON_AUTHORITATIVE",
            "execution_status": "FAILED",
            "span_id": span.span_name,
            "source_run": str(run_root),
            "runtime_s": perf_counter() - started,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        (temporary_dir / "execution_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise ShadowExecutionError(span.span_name, temporary_dir, exc) from exc


def _xyz(value: str) -> tuple[float, float, float]:
    parts = tuple(float(item) for item in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return parts


def _load_wire_region(run_root: Path) -> RawRegion:
    root = run_root / "regions" / "wire"
    index_path = root / "original_index.bin"
    count = index_path.stat().st_size // np.dtype("<i8").itemsize
    xyz = np.memmap(root / "xyz_local.bin", dtype="<f4", mode="r", shape=(count, 3))
    bounds_min = tuple(float(item) for item in np.min(xyz, axis=0))
    bounds_max = tuple(float(item) for item in np.max(xyz, axis=0))
    del xyz
    return RawRegion(
        xyz_local_m_path=str(root / "xyz_local.bin"),
        original_index_path=str(index_path),
        classification_path=str(root / "classification.bin"),
        count=int(count),
        bounds_min_xyz=bounds_min,
        bounds_max_xyz=bounds_max,
        origin_xyz=(0.0, 0.0, 0.0),
    )


def _exact_shadow_clearance(
    run_root: Path, result: object, config: object, shadow_root: Path
) -> dict[str, object]:
    """Reuse the unchanged exact edge engine against existing raw tree members."""
    arrays = result.w1_arrays
    edge_bundle = build_wire_edge_bundle(arrays)
    if edge_bundle.edge_count == 0:
        return {"status": "NO_DISTANCE_READY_EDGE", "records": []}
    with np.load(run_root / "tree" / "tree_members.npz", allow_pickle=False) as stream:
        source_index = np.asarray(stream["original_index"], dtype=np.int64)
        xyz = np.asarray(stream["xyz_local_m"], dtype=np.float64)
        tree_id = np.asarray(stream["tree_id"], dtype=np.int32)
    old = json.loads((run_root / "clearance.json").read_text(encoding="utf-8"))
    names = {int(item["global_tree_id"]): item["external_name"] for item in old["records"]}
    records: list[dict[str, object]] = []
    for identifier in np.unique(tree_id):
        rows = np.flatnonzero(tree_id == identifier)
        positions, distances, nearest_xyz, _, evaluated = _direct_nearest_edges(
            xyz[rows], edge_bundle, config.clearance_tie_tolerance_m
        )
        best = int(np.argmin(distances))
        edge_position = int(positions[best])
        row = int(rows[best])
        records.append(
            {
                "global_tree_id": int(identifier),
                "external_name": names.get(int(identifier), f"Tree{identifier}"),
                "minimum_clearance_m": float(distances[best]),
                "nearest_tree_original_index": int(source_index[row]),
                "nearest_tree_xyz_local_m": [float(value) for value in xyz[row]],
                "nearest_wire_id": int(edge_bundle.wire_id[edge_position]),
                "nearest_edge_index": edge_position,
                "nearest_wire_xyz_local_m": [float(value) for value in nearest_xyz[best]],
                "tree_point_count": int(len(rows)),
                "evaluated_edge_count": int(evaluated),
            }
        )
    records.sort(key=lambda item: (float(item["minimum_clearance_m"]), int(item["global_tree_id"])))
    return {
        "status": "VALID",
        "distance_ready_edges": int(edge_bundle.edge_count),
        "global_minimum_clearance_m": float(records[0]["minimum_clearance_m"]),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("shadow_root", type=Path)
    parser.add_argument("--span-name", required=True)
    parser.add_argument("--start", type=_xyz, required=True)
    parser.add_argument("--end", type=_xyz, required=True)
    parser.add_argument("--start-pole-top-z", type=float, required=True)
    parser.add_argument("--end-pole-top-z", type=float, required=True)
    parser.add_argument(
        "--origin",
        type=_xyz,
        required=True,
        help="existing raw-region local-to-absolute coordinate origin",
    )
    parser.add_argument("--source-point-count", type=int, required=True)
    args = parser.parse_args()
    start, end = np.asarray(args.start, dtype=np.float64), np.asarray(args.end, dtype=np.float64)
    direction_xy = end[:2] - start[:2]
    length = float(np.linalg.norm(direction_xy))
    direction_xy /= length
    region = _load_wire_region(args.run_root)
    span = SpanRecord(
        line_id=0,
        line_name="Shadow",
        span_id=0,
        span_name=args.span_name,
        start_pole_candidate_id=0,
        end_pole_candidate_id=1,
        start_xyz_absolute_m=tuple(start),
        end_xyz_absolute_m=tuple(end),
        start_pole_top_z_m=args.start_pole_top_z,
        end_pole_top_z_m=args.end_pole_top_z,
        length_m=length,
        direction_xy=(float(direction_xy[0]), float(direction_xy[1]), 0.0),
        half_width_m=9.0,
        halo_m=6.0,
        core_bounds_min_xyz=region.bounds_min_xyz,
        core_bounds_max_xyz=region.bounds_max_xyz,
        halo_bounds_min_xyz=region.bounds_min_xyz,
        halo_bounds_max_xyz=region.bounds_max_xyz,
        quality="SHADOW_ONLY",
    )
    # ``region`` only provides persisted bounds for the Shadow SpanRecord;
    # execution itself reloads the same read-only region through the canonical path.
    execute_shadow_run(
        run_root=args.run_root,
        final_output_dir=args.shadow_root,
        span=span,
        origin_xyz=args.origin,
        authoritative_source_point_count=args.source_point_count,
    )


if __name__ == "__main__":
    main()
