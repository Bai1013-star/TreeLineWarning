"""MANUAL_SINGLE_SPAN orchestrator: one real span from two pole clicks.

This is the single integration layer the owner-facing workflow drives.  It
reuses the frozen FAST mature stages directly:

    source LAS
      -> coarse working set (raw-to-rep map, routing representatives)
      -> Ground/HAG (SMRF + HAG-NN on representatives)
      -> [ UI ] two approximate pole clicks
      -> automatic local pole refinement  ->  ONE manual SpanRecord
      -> halo-box raw region (ONE full-resolution stream)
      -> wire region / vegetation region split by representative membership
      -> per-span wire reconstruction (frozen) + tree segmentation (frozen)
      -> reconcile (single span) + optimized exact Tree-Wire clearance
      -> per-tree metrics + DBH (single full-res HAG pass)
      -> [ UI ] risk threshold  ->  danger classification + danger points
      -> 28 offline visuals + Excel + Chinese report

Every stage is digest-checkpointed (source SHA-256, source identity, config
digest, manual-span config digest, schema version, upstream digest) so a late
failure in Tree-Wire / risk / HTML / Excel / report never forces recomputing
Ground, tree segmentation, wire reconstruction or full-resolution membership.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from core.config.fast_baseline import FastBaselineConfig

from algorithms.manual_span.clearance_fast import ManualClearanceResult, run_manual_clearance
from algorithms.manual_span.config import (
    MANUAL_PIPELINE_SCHEMA,
    NON_CERTIFIED_MARKER,
    ManualSpanConfig,
    canonical_digest,
    default_manual_span_config,
)
from algorithms.manual_span.metrics import TreeMetrics, compute_tree_metrics
from algorithms.manual_span.regions import recover_halo_box_region, split_halo_region
from algorithms.manual_span.risk import RiskResult, apply_risk_threshold, risk_threshold_payload
from algorithms.manual_span.span import (
    ManualRefinedPole,
    build_manual_span_record,
    manual_span_selection_payload,
    refine_pole_from_click,
    selection_digest,
)
from algorithms.poles.evidence import GroundEvidence, load_ground_evidence
from algorithms.poles.topology import SpanRecord
from algorithms.poles.wire_evidence import (
    COARSE_WIRE_ROUTING_SCHEMA,
    detect_coarse_wire_evidence_with_rescue,
)
from algorithms.spans.checkpoint import read_checkpoint, write_checkpoint
from algorithms.spans.clearance import SpanTreeWireRecord
from algorithms.spans.reconcile import TreeReconciliationResult, reconcile_trees
from algorithms.spans.tree_runner import (
    SpanTreeOutcome,
    load_span_tree_outcome,
    run_span_tree_pipeline,
)
from algorithms.spans.wire import (
    SpanWireResult,
    load_span_wire_result,
    run_span_wire_pipeline,
)
from algorithms.vegetation.coarse_to_fine import (
    load_coarse_ground_representatives,
    load_raw_region,
)
from algorithms.vegetation.fast_research import (
    FastGroundResult,
    fast_ground_content_digest,
    run_fast_ground_hag,
    sha256_file,
)
from algorithms.vegetation.fast_vegetation_v3 import FastV3DiagnosticState
from algorithms.vegetation.fast_weighted_preprocess import (
    WeightedWorkingSet,
    build_weighted_fast_vegetation,
    build_weighted_working_set,
)

_Progress = Callable[[int, str, bool, float | None], None]

STAGE_LABELS = {
    1: "读取点云并建立粗点云",
    2: "地面分离与高度计算",
    3: "电杆点击与精细化",
    4: "恢复档距原始点云",
    5: "识别和重建导线",
    6: "单木分割与全分辨率回映",
    7: "精确树线净空计算",
    8: "树结构指标与DBH",
    9: "风险分级与危险点",
    10: "生成可视化/Excel/报告",
}


def _report(
    progress: _Progress | None,
    stage: int,
    message: str,
    done: bool,
    elapsed: float | None,
):
    if progress is not None:
        progress(stage, message, done, elapsed)


def _normalize_source_rgb(rgb16: np.ndarray) -> np.ndarray:
    """Deterministically normalize LAS RGB to 0..255 uint8.

    The source is a LAS 1.4 point-format-7 cloud whose RGB fields are uint16.
    Many colorizers store 8-bit values scaled by 256 into the 16-bit field
    (so every value is a multiple of 256); others store true 16-bit.  This
    inspects the actual values and converts correctly instead of truncating
    (truncation ``uint16 -> uint8`` destroys 16-bit colours and renders the
    cloud black).
    """

    rgb16 = np.asarray(rgb16, dtype=np.uint16)
    if rgb16.ndim == 1:
        rgb16 = rgb16[:, None]
    if rgb16.shape[0] == 0 or rgb16.shape[1] == 0:
        return rgb16.astype(np.uint8, copy=False)
    sample = rgb16.ravel()
    maximum = int(sample.max())
    if maximum <= 255:
        return rgb16.astype(np.uint8, copy=False)
    if maximum > 0 and bool(np.all(sample % 256 == 0)):
        # 8-bit values scaled into the 16-bit field.
        return (rgb16 >> 8).astype(np.uint8)
    # True 16-bit: scale linearly to 0..255.
    scaled = rgb16.astype(np.float64) * (255.0 / 65535.0)
    return scaled.astype(np.uint8)


def _sample_source_cloud(
    source: Path,
    maximum_points: int,
    *,
    chunk_size: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Deterministic spatially-aware sample of the RAW / pre-denoisE source LAS.

    The sample is taken directly from the ORIGINAL source cloud (before any
    denoising, ground removal or representative preprocess).  For each small XY
    display voxel we keep the structural TOP (max-Z) point, and additionally the
    BASE (min-Z) point whenever the cell spans a tall vertical structure, so
    narrow poles stay visible as base+top marks instead of being thinned away.
    Coordinates stay in the real LAS frame; RGB is normalized to 0..255.
    """

    import laspy

    _TALL_Z_RANGE_M = 6.0
    with laspy.open(source) as reader:
        has_color = all(
            name in reader.header.point_format.dimension_names
            for name in ("red", "green", "blue")
        )
        bounds_min = np.asarray(reader.header.mins[:2], dtype=np.float64)
        bounds_max = np.asarray(reader.header.maxs[:2], dtype=np.float64)
        area_m2 = float(np.prod(bounds_max - bounds_min))
        cell_size_m = max(0.75, float(np.sqrt(area_m2 / max(1, maximum_points))))

        cell_top: dict[tuple[int, int], dict[str, Any]] = {}  # key -> point data
        cell_base: dict[tuple[int, int], dict[str, Any]] = {}
        cell_zrange: dict[tuple[int, int], float] = {}
        offset = 0
        for chunk in reader.chunk_iterator(chunk_size):
            count = len(chunk)
            xyz_abs = np.column_stack(
                (np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z))
            ).astype(np.float64, copy=False)
            keys = np.floor(xyz_abs[:, :2] / cell_size_m).astype(np.int64)
            order = np.lexsort((np.arange(count), keys[:, 0], keys[:, 1]))
            sorted_keys = keys[order]
            sorted_z = xyz_abs[order, 2]
            boundaries = np.flatnonzero(
                np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
            ) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [count]))
            rgb = None
            if has_color:
                rgb = np.column_stack(
                    (
                        np.asarray(chunk.red, dtype=np.uint16),
                        np.asarray(chunk.green, dtype=np.uint16),
                        np.asarray(chunk.blue, dtype=np.uint16),
                    )
                )
            for group in range(len(starts)):
                key = (int(sorted_keys[starts[group], 0]), int(sorted_keys[starts[group], 1]))
                z_range = float(sorted_z[ends[group] - 1] - sorted_z[starts[group]])
                top_pos = int(order[ends[group] - 1])
                base_pos = int(order[starts[group]])
                # NOTE: the kept rows are COPIED, never stored as numpy views.
                # A view keeps the whole per-chunk XYZ/RGB array alive for the
                # life of the dict, so a multi-chunk source (e.g. 105M points)
                # would retain tens of GB and exhaust memory.
                top_data = {
                    "xyz": xyz_abs[top_pos].copy(),
                    "rgb16": rgb[top_pos].copy() if rgb is not None else None,
                    "index": offset + top_pos,
                }
                if key not in cell_top:
                    cell_top[key] = top_data
                    cell_zrange[key] = z_range
                elif z_range > cell_zrange[key]:
                    cell_top[key] = top_data
                    cell_zrange[key] = z_range
                if z_range > _TALL_Z_RANGE_M and key not in cell_base:
                    cell_base[key] = {
                        "xyz": xyz_abs[base_pos].copy(),
                        "rgb16": rgb[base_pos].copy() if rgb is not None else None,
                        "index": offset + base_pos,
                    }
            if len(cell_top) >= maximum_points:
                break
            offset += count

    if not cell_top:
        return {
            "xyz_absolute": np.empty((0, 3), dtype=np.float64),
            "rgb": np.empty((0, 3), dtype=np.uint8),
            "original_index": np.empty(0, dtype=np.int64),
        }
    kept_xyz = [point["xyz"] for point in cell_top.values()] + [
        point["xyz"] for point in cell_base.values()
    ]
    kept_rgb16 = [
        point["rgb16"] for point in cell_top.values() if point["rgb16"] is not None
    ] + [
        point["rgb16"] for point in cell_base.values() if point["rgb16"] is not None
    ]
    kept_index = [point["index"] for point in cell_top.values()] + [
        point["index"] for point in cell_base.values()
    ]
    xyz = np.asarray(kept_xyz, dtype=np.float64)
    rgb16 = (
        np.asarray(kept_rgb16, dtype=np.uint16)
        if kept_rgb16
        else np.empty((0, 3), dtype=np.uint16)
    )
    return {
        "xyz_absolute": xyz,
        "rgb": _normalize_source_rgb(rgb16),
        "original_index": np.asarray(kept_index, dtype=np.int64),
    }


def recover_local_raw_box(
    source: Path,
    bounds_absolute: tuple[tuple[float, float, float], tuple[float, float, float]],
    output_root: Path,
    *,
    cache_key: str,
    chunk_size: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Recover every RAW source point inside a local box (full resolution).

    Streams the ORIGINAL source LAS and keeps the full-resolution points whose
    absolute XYZ lie inside *bounds_absolute*.  This powers the dense local view
    and raw-based pole refinement; it never alters the source.  The result is
    cached as an npz keyed by the caller-supplied rounded *cache_key*.
    """

    import laspy

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    npz_path = output_root / f"local_raw_{cache_key}.npz"
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as stream:
            return {
                "xyz_absolute": np.asarray(stream["xyz_absolute"], dtype=np.float64),
                "rgb": np.asarray(stream["rgb"], dtype=np.uint8),
                "original_index": np.asarray(stream["original_index"], dtype=np.int64),
            }
    minimum = np.asarray(bounds_absolute[0], dtype=np.float64)
    maximum = np.asarray(bounds_absolute[1], dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum < minimum):
        raise ValueError("MANUAL_SPAN_LOCAL_BOX_INVALID")
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    with laspy.open(source) as reader:
        has_color = all(
            name in reader.header.point_format.dimension_names
            for name in ("red", "green", "blue")
        )
        offset = 0
        for chunk in reader.chunk_iterator(chunk_size):
            count = len(chunk)
            xyz = np.column_stack(
                (np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z))
            ).astype(np.float64, copy=False)
            mask = np.all(xyz >= minimum, axis=1) & np.all(xyz <= maximum, axis=1)
            if np.any(mask):
                xyz_parts.append(xyz[mask])
                idx_parts.append(
                    np.arange(offset, offset + count, dtype=np.int64)[mask]
                )
                if has_color:
                    rgb_parts.append(
                        np.column_stack(
                            (
                                np.asarray(chunk.red, dtype=np.uint16),
                                np.asarray(chunk.green, dtype=np.uint16),
                                np.asarray(chunk.blue, dtype=np.uint16),
                            )
                        )[mask]
                    )
            offset += count
    xyz = np.concatenate(xyz_parts, axis=0) if xyz_parts else np.empty((0, 3), dtype=np.float64)
    rgb16 = (
        np.concatenate(rgb_parts, axis=0)
        if rgb_parts
        else np.empty((0, 3), dtype=np.uint16)
    )
    original_index = (
        np.concatenate(idx_parts, axis=0) if idx_parts else np.empty(0, dtype=np.int64)
    )
    result = {
        "xyz_absolute": xyz,
        "rgb": _normalize_source_rgb(rgb16),
        "original_index": original_index,
    }
    _write_npz(
        npz_path,
        xyz_absolute=result["xyz_absolute"],
        rgb=result["rgb"],
        original_index=result["original_index"],
    )
    return result


def _spatial_display_rows(xyz: np.ndarray, maximum_points: int) -> np.ndarray:
    """Choose a deterministic 3-D-aware display subset.

    The weighted preprocess keeps one source-bound representative per voxel, but
    a later browser cap must not collapse a vertical pole to one XY point.  A
    small height band is therefore part of the display key.  This is display
    assistance only; the returned rows still refer to the source-bound
    representatives and are never used by the analysis stages.
    """

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) <= maximum_points:
        return np.arange(len(xyz), dtype=np.int64)
    if maximum_points <= 0:
        return np.empty(0, dtype=np.int64)
    minimum = np.min(xyz, axis=0)
    maximum = np.max(xyz, axis=0)
    xy_span = np.maximum(maximum[:2] - minimum[:2], 1e-6)
    xy_cell = max(0.25, float(np.sqrt(np.prod(xy_span) / maximum_points)))
    z_span = max(float(maximum[2] - minimum[2]), 1e-6)
    z_cell = max(1.0, z_span / 48.0)
    keys = np.column_stack(
        (
            np.floor((xyz[:, 0] - minimum[0]) / xy_cell).astype(np.int64),
            np.floor((xyz[:, 1] - minimum[1]) / xy_cell).astype(np.int64),
            np.floor((xyz[:, 2] - minimum[2]) / z_cell).astype(np.int64),
        )
    )
    _unique, first = np.unique(keys, axis=0, return_index=True)
    rows = np.sort(first.astype(np.int64, copy=False))
    if len(rows) > maximum_points:
        positions = np.linspace(0, len(rows) - 1, maximum_points, dtype=np.int64)
        rows = rows[positions]
    return rows


def _source_rgb_for_indices(
    source: Path,
    original_index: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Recover RGB for source-row indices without loading the full LAS."""

    import laspy

    requested = np.asarray(original_index, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError("display original_index must be one-dimensional")
    if len(requested) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    if np.any(requested < 0):
        raise ValueError("display original_index contains a negative row")
    order = np.argsort(requested, kind="stable")
    sorted_requested = requested[order]
    recovered = np.zeros((len(requested), 3), dtype=np.uint16)
    with laspy.open(source) as reader:
        has_color = all(
            name in reader.header.point_format.dimension_names
            for name in ("red", "green", "blue")
        )
        if not has_color:
            # A visible neutral palette is preferable to a black scene when the
            # source has no RGB dimensions.
            return np.tile(np.asarray((160, 180, 210), dtype=np.uint8), (len(requested), 1))
        offset = 0
        for chunk in reader.chunk_iterator(chunk_size):
            stop = offset + len(chunk)
            left = int(np.searchsorted(sorted_requested, offset, side="left"))
            right = int(np.searchsorted(sorted_requested, stop, side="left"))
            if left < right:
                positions = sorted_requested[left:right] - offset
                recovered[order[left:right], 0] = np.asarray(chunk.red)[positions]
                recovered[order[left:right], 1] = np.asarray(chunk.green)[positions]
                recovered[order[left:right], 2] = np.asarray(chunk.blue)[positions]
            offset = stop
            if left == len(sorted_requested):
                break
    return _normalize_source_rgb(recovered)


def _sample_preprocessed_cloud(
    source: Path,
    working: WeightedWorkingSet,
    maximum_points: int,
) -> dict[str, np.ndarray]:
    """Build a browser display from source-bound weighted representatives.

    The representatives are the existing coarse/preprocessed geometry, not a
    semantic pole/tree/wire result.  RGB is joined back through each
    representative's original LAS row so clicks remain in the source frame.
    """

    with np.load(working.representatives_npz_path, allow_pickle=False) as stream:
        xyz = np.asarray(stream["xyz_absolute"], dtype=np.float64)
        original_index = np.asarray(stream["original_index"], dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(original_index):
        raise ValueError("preprocessed display representation is inconsistent")
    rows = _spatial_display_rows(xyz, maximum_points)
    selected_xyz = xyz[rows]
    selected_original = original_index[rows]
    return {
        "xyz_absolute": selected_xyz,
        "rgb": _source_rgb_for_indices(source, selected_original),
        "original_index": selected_original,
        "display_source": np.asarray(["PREPROCESSED_WEIGHTED_REPRESENTATIVES"]),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
    )
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _remove_tree(path: Path) -> None:
    """Best-effort recursive removal of a stale stage output directory."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)


class ManualSingleSpanRun:
    """One manual single-span run: init -> selection -> analysis -> finalize."""

    def __init__(
        self,
        source: Path | str,
        run_root: Path | str,
        *,
        cfg: FastBaselineConfig | None = None,
        manual: ManualSpanConfig | None = None,
        progress: _Progress | None = None,
        ground_reuse_dir: Path | str | None = None,
    ) -> None:
        self.source = Path(source).resolve()
        self.run_root = Path(run_root).resolve()
        from algorithms.manual_span.config import manual_route_config

        self.cfg = cfg or manual_route_config()
        self.manual = manual or default_manual_span_config()
        self.progress = progress
        self.local_raw_cache: dict[str, dict[str, np.ndarray]] = {}
        self.local_raw_dir = Path(run_root) / "00_input" / "local_raw"
        import threading as _threading

        self._local_raw_lock = _threading.Lock()
        self.ground_reuse_dir = (
            None if ground_reuse_dir is None else Path(ground_reuse_dir).resolve()
        )
        self.source_sha256: str = ""
        self.source_id: str = ""
        self.inspection: dict[str, Any] = {}
        self.working: WeightedWorkingSet | None = None
        self.ground: FastGroundResult | None = None
        self.evidence: GroundEvidence | None = None
        self.origin_xyz: tuple[float, float, float] | None = None
        self.coarse_ground_xyz_absolute: np.ndarray | None = None
        self.pole_a: ManualRefinedPole | None = None
        self.pole_b: ManualRefinedPole | None = None
        self.span: SpanRecord | None = None
        self.selection_payload: dict[str, Any] | None = None
        self.span_digest: str = ""
        self.wire_region_dir: Path | None = None
        self.vegetation_region_dir: Path | None = None
        self.wire_outcome: SpanWireResult | None = None
        self.tree_outcome: SpanTreeOutcome | None = None
        self.reconciled: TreeReconciliationResult | None = None
        self.clearance: ManualClearanceResult | None = None
        self.metrics: tuple[TreeMetrics, ...] = ()
        self.metrics_by_tree: dict[int, TreeMetrics] = {}
        self.risk: RiskResult | None = None
        self.parameters_log: list[dict[str, Any]] = []
        self.display_sample: dict[str, np.ndarray] | None = None
        self.timings: dict[str, float] = {}
        self._run_started = time.perf_counter()

    # ------------------------------------------------------------------ init
    def initialize(self) -> dict[str, Any]:
        """Inspect the source, build the coarse working set and Ground/HAG.

        All three are checkpointed and cached so a resumed run never re-reads
        the 105M-point source or re-runs ground.
        """

        started = time.perf_counter()
        from algorithms.manual_span.pdal_compat import ensure_pdal_compat

        ensure_pdal_compat()
        self.run_root.mkdir(parents=True, exist_ok=True)
        for stage in ("00_input", "01_preprocess", "02_ground", "03_selection",
                      "04_regions", "05_wire", "06_tree", "07_clearance",
                      "08_metrics", "09_risk", "10_visuals", "11_report"):
            (self.run_root / stage).mkdir(parents=True, exist_ok=True)

        self.source_sha256 = sha256_file(self.source)
        self.source_id = f"manual_span_{self.source_sha256[:16]}"
        self.inspection = self._inspect_source()
        self.origin_xyz = tuple(float(value) for value in self.inspection["bounds_min_xyz"])
        _write_json(self.run_root / "00_input" / "source_inspection.json", self.inspection)

        base_upstream = canonical_digest(
            {
                "schema": MANUAL_PIPELINE_SCHEMA,
                "source_sha256": self.source_sha256,
                "source_id": self.source_id,
                "config_digest": self.cfg.computed_digest(),
                "manual_config_digest": self.manual.computed_digest(),
            }
        )

        _report(self.progress, 1, STAGE_LABELS[1], False, None)
        started_stage = time.perf_counter()
        self.working = self._stage_preprocess(base_upstream)
        _report(self.progress, 1, STAGE_LABELS[1], True, time.perf_counter() - started_stage)

        _report(self.progress, 2, STAGE_LABELS[2], False, None)
        started_stage = time.perf_counter()
        self.ground = self._stage_ground(
            upstream_digest=self.working.config_digest
        )
        _report(self.progress, 2, STAGE_LABELS[2], True, time.perf_counter() - started_stage)
        self.coarse_ground_xyz_absolute = load_coarse_ground_representatives(self.ground)
        self.evidence = self._load_evidence()
        self.display_sample = self._stage_display_sample()
        elapsed = time.perf_counter() - started
        self.timings["init"] = elapsed
        self.parameters_log = [
            {
                "parameter": "wire_min_longitudinal_coverage_frac",
                "old": 0.55,
                "new": self.cfg.wire_min_longitudinal_coverage_frac,
                "reason": (
                    "真实手动档距中导线证据为长而窄的水平条带(覆盖34-53%)，"
                    "被冻结的55%阈值全部拒绝；冠层簇覆盖均<10%，降至0.30只放行真实导线。"
                ),
                "observed_effect": "重建出真实导线线束，冠层仍被排除。",
            },
            {
                "parameter": "wire_min_longitudinal_span_m",
                "old": 10.0,
                "new": self.cfg.wire_min_longitudinal_span_m,
                "reason": (
                    "真实导线条带在DBSCAN分段前为15-24 m；6 m下限保留真实条带并排除冠层碎片。"
                ),
                "observed_effect": "真实导线条带进入线束候选。",
            },
        ]
        return {
            "RUN_STATUS": "INITIALIZED",
            "MANUAL_SINGLE_SPAN": "INITIALIZED",
            "INPUT_FILE": str(self.source),
            "INPUT_SHA256": self.source_sha256,
            "SOURCE_ID": self.source_id,
            "OUTPUT_DIRECTORY": str(self.run_root),
            "RAW_POINTS": int(self.inspection["point_count"]),
            "COARSE_REPRESENTATIVES": self.working.representative_count,
            "ORIGIN_XYZ": list(self.origin_xyz),
            "INIT_RUNTIME": time.perf_counter() - started,
        }

    def _inspect_source(self) -> dict[str, Any]:
        import laspy

        with laspy.open(self.source) as reader:
            header = reader.header
            payload = {
                "input_file": str(self.source),
                "input_sha256": self.source_sha256,
                "source_id": self.source_id,
                "point_count": int(header.point_count),
                "bounds_min_xyz": [float(value) for value in header.mins],
                "bounds_max_xyz": [float(value) for value in header.maxs],
                "scales": [float(value) for value in header.scales],
                "offsets": [float(value) for value in header.offsets],
                "point_format": int(header.point_format.id),
                "version": str(header.version),
            }
        if payload["point_count"] <= 0:
            raise RuntimeError("MANUAL_SPAN_EMPTY_SOURCE")
        return payload

    def _stage_preprocess(self, upstream_digest: str) -> WeightedWorkingSet:
        from algorithms.vegetation.fast_baseline_pipeline import (
            choose_representative_voxel_size,
        )

        stage_dir = self.run_root / "01_preprocess"
        cached = read_checkpoint(
            stage_dir, "preprocess",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream_digest,
        )
        if cached is not None:
            _summary, _digest = cached
            working = self._reload_working_set(stage_dir)
            if working is not None:
                return working
        xy_area = (
            (self.inspection["bounds_max_xyz"][0] - self.inspection["bounds_min_xyz"][0])
            * (self.inspection["bounds_max_xyz"][1] - self.inspection["bounds_min_xyz"][1])
        )
        voxel_size = choose_representative_voxel_size(
            int(self.inspection["point_count"]), xy_area_m2=float(xy_area)
        )
        _status, working = build_weighted_working_set(
            self.source,
            stage_dir,
            voxel_size_m=voxel_size,
            artifact_stem="manual_span",
        )
        write_checkpoint(
            stage_dir, "preprocess",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream_digest,
            summary={
                "status": _status,
                "representative_count": working.representative_count,
                "source_point_count": working.source_point_count,
            },
            output_digest=working.config_digest,
        )
        return working

    def _reload_working_set(self, stage_dir: Path) -> WeightedWorkingSet | None:
        reps_npz = stage_dir / "working_set_representatives.npz"
        mapping = stage_dir / "raw_to_representative_id.npy"
        if not reps_npz.is_file() or not mapping.is_file():
            return None
        try:
            with np.load(reps_npz, allow_pickle=False) as stream:
                representative_count = int(len(stream["representative_id"]))
        except (OSError, ValueError):
            return None
        from algorithms.vegetation.fast_baseline_pipeline import (
            choose_representative_voxel_size,
        )

        voxel_size = choose_representative_voxel_size(
            int(self.inspection["point_count"]),
            xy_area_m2=float(
                (self.inspection["bounds_max_xyz"][0] - self.inspection["bounds_min_xyz"][0])
                * (self.inspection["bounds_max_xyz"][1] - self.inspection["bounds_min_xyz"][1])
            ),
        )
        return WeightedWorkingSet(
            schema_version="coarse-weighted-preprocess-v1",
            non_certified_marker=NON_CERTIFIED_MARKER,
            source_las_path=str(self.source),
            source_las_sha256=self.source_sha256,
            source_point_count=int(self.inspection["point_count"]),
            source_bounds_min_xyz=self.inspection["bounds_min_xyz"],
            source_bounds_max_xyz=self.inspection["bounds_max_xyz"],
            source_scales=self.inspection["scales"],
            source_offsets=self.inspection["offsets"],
            voxel_size_m=float(voxel_size),
            representative_count=representative_count,
            raw_count_sum=int(self.inspection["point_count"]),
            representatives_npz_path=str(reps_npz),
            representative_las_path=str(stage_dir / "working_set_representatives.las"),
            raw_to_representative_npy_path=str(mapping),
            classification_counts={},
            elapsed_seconds=0.0,
            config_digest=canonical_digest(
                {
                    "schema_version": "coarse-weighted-preprocess-v1",
                    "source_las_sha256": self.source_sha256,
                    "voxel_size_m": float(voxel_size),
                }
            ),
        )

    def _stage_ground(self, upstream_digest: str) -> FastGroundResult:
        stage_dir = self.run_root / "02_ground"
        cached = read_checkpoint(
            stage_dir, "ground",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream_digest,
        )
        if cached is not None:
            payload = json.loads((stage_dir / "ground_hag_result.json").read_text("utf-8"))
            return FastGroundResult.from_dict(payload)
        self._reuse_ground_cache_if_possible(stage_dir)
        _status, ground = run_fast_ground_hag(
            Path(self.working.representative_las_path), stage_dir
        )
        write_checkpoint(
            stage_dir, "ground",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream_digest,
            summary={
                "status": _status,
                "ground_class_2_count": ground.ground_class_2_count,
                "hag_valid_count": ground.hag_valid_count,
            },
            output_digest=fast_ground_content_digest(ground),
        )
        return ground

    def _reuse_ground_cache_if_possible(self, stage_dir: Path) -> None:
        """Seed the ground stage from a validated PDAL SMRF cache for this source.

        When the Python PDAL binding is unavailable on this machine, a prior run
        that already executed the frozen PDAL SMRF + HAG-NN ground for the SAME
        source and SAME coarse representatives is reused verbatim (the output
        would be byte-identical).  ``run_fast_ground_hag`` still validates the
        cache key (source SHA + config digest + output SHA) before accepting it.
        """

        if self.ground_reuse_dir is None:
            return
        if (stage_dir / "classified_hag.las").is_file():
            return
        source_las = self.ground_reuse_dir / "classified_hag.las"
        source_json = self.ground_reuse_dir / "ground_hag_result.json"
        if not source_las.is_file() or not source_json.is_file():
            return
        import shutil

        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_las, stage_dir / "classified_hag.las")
        payload = json.loads(source_json.read_text("utf-8"))
        payload["output_las_path"] = str((stage_dir / "classified_hag.las").resolve())
        _write_json(stage_dir / "ground_hag_result.json", payload)

    def _load_evidence(self) -> GroundEvidence:
        return load_ground_evidence(
            Path(self.ground.output_las_path),
            self.cfg,
            origin_xyz=self.origin_xyz,
        )

    def _stage_display_sample(self) -> dict[str, np.ndarray]:
        """Deterministic RAW / pre-denoise / pre-ground display sample.

        The primary selectable surface always uses the ORIGINAL source LAS
        points (before denoising and before ground removal) so thin poles stay
        visible.  The sample is display assistance only; selection coordinates
        and downstream source data are never changed.
        """

        npz_path = self.run_root / "00_input" / "display_sample_v4.npz"
        if npz_path.is_file():
            with np.load(npz_path, allow_pickle=False) as stream:
                return {
                    "xyz_absolute": np.asarray(stream["xyz_absolute"], dtype=np.float64),
                    "rgb": np.asarray(stream["rgb"], dtype=np.uint8),
                    "original_index": np.asarray(stream["original_index"], dtype=np.int64),
                    "display_source": np.asarray(
                        stream["display_source"], dtype="U64"
                    ),
                }
        sample = _sample_source_cloud(self.source, self.manual.display_sample_points)
        sample["display_source"] = np.asarray(["SOURCE_RAW_PRE_DENOISE"])
        _write_npz(
            npz_path,
            xyz_absolute=sample["xyz_absolute"],
            rgb=sample["rgb"],
            original_index=sample["original_index"],
            display_source=sample["display_source"],
        )
        return sample

    # ------------------------------------------------------------- selection
    def local_raw_for_xy(
        self, x: float, y: float, radius: float | None = None
    ) -> dict[str, np.ndarray]:
        """Recover (and cache) the FULL-RESOLUTION RAW source points near an XY.

        The box covers ``refine_search_radius_m`` around the approximate click
        over the full source Z extent.  The first call for a rounded cell
        streams the source once; later calls reuse the cached npz.
        """

        radius = radius or (self.manual.refine_search_radius_m + 1.0)
        cache_key = f"{int(round(x / 2.0))}_{int(round(y / 2.0))}_{int(round(radius))}"
        if cache_key in self.local_raw_cache:
            return self.local_raw_cache[cache_key]
        with self._local_raw_lock:
            if cache_key in self.local_raw_cache:
                return self.local_raw_cache[cache_key]
            z_min = float(self.inspection["bounds_min_xyz"][2])
            z_max = float(self.inspection["bounds_max_xyz"][2])
            bounds = ((x - radius, y - radius, z_min), (x + radius, y + radius, z_max))
            data = recover_local_raw_box(
                self.source,
                bounds,
                self.local_raw_dir,
                cache_key=cache_key,
            )
            # Bound the in-memory local-raw cache to the CURRENT box: a dense
            # recovery box holds millions of full-resolution points (~100 MB+
            # each), so retaining every Pole A / Pole B box would accumulate
            # until a later refinement OOMs on a limited-RAM machine.  The npz
            # disk cache (keyed identically) is the durable reuse; only the most
            # recent box is kept resident.
            self.local_raw_cache.clear()
            self.local_raw_cache[cache_key] = data
            return data

    def _refine_pole(
        self, label: str, click_xyz: tuple[float, float, float]
    ) -> ManualRefinedPole:
        """Refine one pole from RAW local points (fallback: coarse evidence)."""

        from algorithms.manual_span.span import refine_pole_from_raw_cloud

        local = self.local_raw_for_xy(click_xyz[0], click_xyz[1])
        raw = local.get("xyz_absolute", np.empty((0, 3), dtype=np.float64))
        if len(raw) > 0:
            return refine_pole_from_raw_cloud(
                label, tuple(float(v) for v in click_xyz), raw,
                self.origin_xyz, self.cfg, self.manual,
            )
        return refine_pole_from_click(
            label, tuple(float(v) for v in click_xyz), self.evidence,
            self.cfg, self.manual,
        )

    def refine_and_build_span(
        self,
        click_a_xyz: tuple[float, float, float],
        click_b_xyz: tuple[float, float, float],
    ) -> tuple[ManualRefinedPole, ManualRefinedPole, SpanRecord]:
        """Refine both clicks, build ONE Span, persist the selection.

        Returns ``(pole_a, pole_b, span)``; a refinement that is not defensible
        raises ``PoleRefinementError`` with the pole label + reason so the UI
        can ask for a re-click of exactly that pole.
        """

        if self.evidence is None:
            raise RuntimeError("MANUAL_SPAN_NOT_INITIALIZED")
        pole_a = self._refine_pole("A", tuple(float(value) for value in click_a_xyz))
        pole_b = self._refine_pole("B", tuple(float(value) for value in click_b_xyz))
        if not pole_a.defensible:
            raise PoleRefinementError("A", pole_a.reason, pole_a.refinement_diagnostics)
        if not pole_b.defensible:
            raise PoleRefinementError("B", pole_b.reason, pole_b.refinement_diagnostics)
        span = build_manual_span_record(
            pole_a, pole_b, self.cfg, self.manual, self.origin_xyz
        )
        self.pole_a, self.pole_b, self.span = pole_a, pole_b, span
        self.selection_payload = manual_span_selection_payload(
            pole_a,
            pole_b,
            span,
            self.source_sha256,
            self.source_id,
            self.origin_xyz,
            selection_version=MANUAL_PIPELINE_SCHEMA,
        )
        self.span_digest = selection_digest(self.selection_payload)
        stage_dir = self.run_root / "03_selection"
        _write_json(stage_dir / "manual_span_selection.json", self.selection_payload)
        write_checkpoint(
            stage_dir, "selection",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=self.span_digest,
            summary={"selection_digest": self.span_digest, "span": span.to_dict()},
            output_digest=self.span_digest,
        )
        return pole_a, pole_b, span

    # ------------------------------------------------------------- analysis
    def run_analysis(self) -> dict[str, Any]:
        """Wire/vegetation routing, region recovery, tree+wire, clearance, metrics."""

        if self.span is None:
            raise RuntimeError("MANUAL_SPAN_NO_SELECTION")
        total_started = time.perf_counter()

        _report(self.progress, 4, STAGE_LABELS[4], False, None)
        started_stage = time.perf_counter()
        (
            wire_region,
            veg_region,
            wire_candidate_ids,
            veg_retained_ids,
            _routing_summary,
        ) = self._stage_regions()
        _report(self.progress, 4, STAGE_LABELS[4], True, time.perf_counter() - started_stage)
        regions_digest = canonical_digest(
            {
                "wire_region_count": wire_region.count,
                "vegetation_region_count": veg_region.count,
                "wire_candidate_reps": int(len(wire_candidate_ids)),
                "vegetation_retained_reps": int(len(veg_retained_ids)),
                "strict_routed_reps": int(_routing_summary["strict_routed_rep_count"]),
                "rescued_routed_reps": int(_routing_summary["rescued_routed_rep_count"]),
                "union_routed_reps": int(_routing_summary["union_routed_rep_count"]),
            }
        )

        _report(self.progress, 5, STAGE_LABELS[5], False, None)
        started_stage = time.perf_counter()
        self.wire_outcome = self._stage_wire(regions_digest)
        _report(self.progress, 5, STAGE_LABELS[5], True, time.perf_counter() - started_stage)
        wire_digest = canonical_digest(
            {
                "strand_count": self.wire_outcome.strand_count,
                "wire_candidate_point_count": self.wire_outcome.diagnostics.get(
                    "wire_candidate_point_count", 0
                ),
                "confirmed_member_count": int(
                    self.wire_outcome.confirmed_member_source_index.shape[0]
                ),
            }
        )

        _report(self.progress, 6, STAGE_LABELS[6], False, None)
        started_stage = time.perf_counter()
        self.tree_outcome = self._stage_tree(regions_digest, wire_digest)
        _report(self.progress, 6, STAGE_LABELS[6], True, time.perf_counter() - started_stage)
        tree_digest = canonical_digest({"tree_count": len(self.tree_outcome.tree_records)})

        _report(self.progress, 7, STAGE_LABELS[7], False, None)
        started_stage = time.perf_counter()
        self.reconciled = reconcile_trees(
            [self.tree_outcome], self.cfg, origin_xyz=self.origin_xyz
        )
        self.clearance = self._stage_clearance(wire_digest, tree_digest)
        _report(self.progress, 7, STAGE_LABELS[7], True, time.perf_counter() - started_stage)

        _report(self.progress, 8, STAGE_LABELS[8], False, None)
        started_stage = time.perf_counter()
        self.metrics = compute_tree_metrics(
            self.reconciled, self.coarse_ground_xyz_absolute, self.origin_xyz, self.manual
        )
        self.metrics_by_tree = {metric.global_tree_id: metric for metric in self.metrics}
        metrics_dir = self.run_root / "08_metrics"
        _write_json(
            metrics_dir / "tree_metrics.json",
            {"trees": [metric.to_dict() for metric in self.metrics]},
        )
        _report(self.progress, 8, STAGE_LABELS[8], True, time.perf_counter() - started_stage)

        analysis_runtime = time.perf_counter() - total_started
        self.timings["analysis"] = analysis_runtime
        return {
            "wire_candidate_points": self.wire_outcome.diagnostics.get(
                "wire_candidate_point_count", 0
            ),
            "wire_cluster_count": self.wire_outcome.diagnostics.get("wire_cluster_count", 0),
            "wire_strand_count": self.wire_outcome.strand_count,
            "tree_count": len(self.tree_outcome.tree_records),
            "assigned_fullres_points": int(self.reconciled.final_original_index.shape[0]),
            "clearance_status": self.clearance.clearance.status,
            "global_minimum_clearance_m": self.clearance.clearance.global_minimum_clearance_m,
            "clearance_runtime": self.clearance.clearance.diagnostics.get("runtime_seconds"),
            "analysis_runtime": analysis_runtime,
        }

    def _stage_regions(self) -> tuple[Any, Any, np.ndarray, np.ndarray, dict[str, Any]]:
        stage_dir = self.run_root / "04_regions"
        cached = read_checkpoint(
            stage_dir, "regions",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(),
            upstream_digest=self._regions_upstream_digest(),
        )
        halo_dir = stage_dir / "halo_region"
        wire_sidecar = stage_dir / "wire_region" / "region_sidecar.json"
        veg_sidecar = stage_dir / "vegetation_region" / "region_sidecar.json"
        if cached is not None and wire_sidecar.is_file() and veg_sidecar.is_file():
            wire_region = load_raw_region(wire_sidecar)
            veg_region = load_raw_region(veg_sidecar)
            wire_ids = np.asarray(cached[0].get("wire_candidate_rep_ids", []), dtype=np.uint32)
            veg_ids = np.asarray(cached[0].get("vegetation_retained_rep_ids", []), dtype=np.uint32)
            self.wire_region_dir = stage_dir / "wire_region"
            self.vegetation_region_dir = stage_dir / "vegetation_region"
            routing_summary = {
                "strict_routed_rep_count": int(cached[0].get("strict_routed_rep_count", 0)),
                "rescued_routed_rep_count": int(cached[0].get("rescued_routed_rep_count", 0)),
                "union_routed_rep_count": int(cached[0].get("union_routed_rep_count", 0)),
                "rescue_diagnostics": dict(cached[0].get("rescue_diagnostics", {})),
            }
            return wire_region, veg_region, wire_ids, veg_ids, routing_summary

        # Coarse wire evidence restricted to the span halo.  Dual-path router:
        # PATH A (strict component gates, unchanged) union PATH B (span-aligned
        # representative rescue of components PATH A rejected).
        wire_evidence = detect_coarse_wire_evidence_with_rescue(
            self.evidence,
            self.cfg,
            self.span,
            origin_xyz=self.origin_xyz,
        )
        origin = np.asarray(self.origin_xyz, dtype=np.float64)
        halo_min = np.asarray(self.span.halo_bounds_min_xyz, dtype=np.float64)
        halo_max = np.asarray(self.span.halo_bounds_max_xyz, dtype=np.float64)
        inside = np.all(self.evidence.xyz_local_m >= halo_min, axis=1) & np.all(
            self.evidence.xyz_local_m <= halo_max, axis=1
        )
        strict_mask = np.asarray(wire_evidence.candidate_mask, dtype=bool) & inside
        rescued_mask = np.asarray(wire_evidence.rescued_mask, dtype=bool) & inside
        wire_routed_mask = strict_mask | rescued_mask
        wire_candidate_ids = np.unique(
            self.evidence.representative_id[wire_routed_mask]
        ).astype(np.uint32)
        strict_routed_count = int(np.count_nonzero(strict_mask))
        rescued_routed_count = int(np.count_nonzero(rescued_mask))
        union_routed_count = int(np.count_nonzero(wire_routed_mask))

        # Scene-wide vegetation routing (on routing representatives only).  The
        # unconfirmed coarse wire candidates (PATH A or PATH B) are NEVER passed
        # as a permanent vegetation exclusion: FINAL_TREE_INPUT = VEGETATION -
        # CONFIRMED_WIRE - CONFIRMED_TOWER, so only the exact FULL_RESOLUTION
        # members of a FINAL confirmed conductor may be subtracted from tree
        # input (handled by the tree stage / orchestrator).
        veg_dir = self.run_root / "06_tree" / "coarse_routing"
        if veg_dir.exists() and any(veg_dir.iterdir()):
            # Reaching this point means the regions checkpoint is stale (this
            # stage is recomputing), so the previous vegetation-router outputs
            # belong to an outdated routing input.  Replace them in place rather
            # than failing closed on now-stale outputs.
            for stale in sorted(veg_dir.iterdir()):
                try:
                    if stale.is_file():
                        stale.unlink()
                    elif stale.is_dir():
                        _remove_tree(stale)
                except OSError:
                    continue
        _veg_input, veg_manifest = build_weighted_fast_vegetation(
            self.working,
            {
                "xy_bounds_min": [float(halo_min[0]), float(halo_min[1])],
                "xy_bounds_max": [float(halo_max[0]), float(halo_max[1])],
                "z_bounds_min": float(halo_min[2]),
                "z_bounds_max": float(halo_max[2]),
            },
            self.ground,
            veg_dir,
            source_id=self.source_id,
            artifact_stem="manual_span",
        )
        diag_npz = veg_dir / "manual_span_fast_vegetation_diagnostics.npz"
        with np.load(diag_npz, allow_pickle=False) as stream:
            diag_state = np.asarray(stream["diagnostic_state"], dtype=np.uint8)
            diag_ids = np.asarray(stream["representative_id"], dtype=np.uint32)
        retained = diag_state == int(FastV3DiagnosticState.VEGETATION_RETAINED)
        veg_retained_ids = np.unique(diag_ids[retained]).astype(np.uint32)

        # One full-resolution stream for the halo box, then split by membership.
        halo_region = recover_halo_box_region(
            self.source,
            (
                tuple(float(value) for value in halo_min + origin),
                tuple(float(value) for value in halo_max + origin),
            ),
            halo_dir,
            origin_xyz=self.origin_xyz,
        )
        wire_region, veg_region = split_halo_region(
            halo_region,
            self.working,
            wire_candidate_ids,
            veg_retained_ids,
            stage_dir / "wire_region",
            stage_dir / "vegetation_region",
        )
        summary = {
            "halo_region_count": halo_region.count,
            "wire_region_count": wire_region.count,
            "vegetation_region_count": veg_region.count,
            "wire_candidate_rep_ids": [int(value) for value in wire_candidate_ids],
            "vegetation_retained_rep_ids": [int(value) for value in veg_retained_ids],
            "strict_routed_rep_count": strict_routed_count,
            "rescued_routed_rep_count": rescued_routed_count,
            "union_routed_rep_count": union_routed_count,
            "rescue_diagnostics": dict(wire_evidence.rescue_diagnostics),
        }
        write_checkpoint(
            stage_dir, "regions",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(),
            upstream_digest=self._regions_upstream_digest(),
            summary=summary,
            output_digest=canonical_digest(
                {
                    "wire_region_count": wire_region.count,
                    "vegetation_region_count": veg_region.count,
                    "wire_candidate_reps": int(len(wire_candidate_ids)),
                    "vegetation_retained_reps": int(len(veg_retained_ids)),
                    "strict_routed_rep_count": strict_routed_count,
                    "rescued_routed_rep_count": rescued_routed_count,
                    "union_routed_rep_count": union_routed_count,
                }
            ),
        )
        self.wire_region_dir = stage_dir / "wire_region"
        self.vegetation_region_dir = stage_dir / "vegetation_region"
        routing_summary = {
            "strict_routed_rep_count": strict_routed_count,
            "rescued_routed_rep_count": rescued_routed_count,
            "union_routed_rep_count": union_routed_count,
            "rescue_diagnostics": dict(wire_evidence.rescue_diagnostics),
        }
        return wire_region, veg_region, wire_candidate_ids, veg_retained_ids, routing_summary

    def _regions_upstream_digest(self) -> str:
        """Upstream digest for the region-routing stage.

        The coarse wire ROUTER semantics are part of the upstream provenance so
        a routing change invalidates ONLY the coarse wire evidence / dependent
        region routing (04_regions -> 05_wire -> tree), never preprocess,
        Ground/HAG or the persisted span selection.
        """

        return canonical_digest(
            {
                "span_digest": self.span_digest,
                "ground": fast_ground_content_digest(self.ground),
                "coarse_wire_routing": COARSE_WIRE_ROUTING_SCHEMA,
            }
        )

    def _stage_wire(self, regions_digest: str) -> SpanWireResult:
        stage_dir = self.run_root / "05_wire"
        span_dir = stage_dir / "span_001"
        cached = read_checkpoint(
            span_dir, "wire",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=regions_digest,
        )
        if cached is not None and (span_dir / "w1_arrays.npz").is_file():
            return load_span_wire_result(span_dir)
        wire_region = load_raw_region(self.wire_region_dir / "region_sidecar.json")
        wire_result = run_span_wire_pipeline(
            self.span,
            wire_region,
            self.cfg,
            origin_xyz=self.origin_xyz,
            output_dir=span_dir,
        )
        payload = {
            "strand_count": wire_result.strand_count,
            "strands": [strand.to_dict() for strand in wire_result.strands],
        }
        _write_json(span_dir / "wire_summary.json", payload)
        write_checkpoint(
            span_dir, "wire",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=regions_digest,
            summary=payload,
            output_digest=canonical_digest(
                {"strand_count": wire_result.strand_count}
            ),
        )
        return wire_result

    def _stage_tree(self, regions_digest: str, wire_digest: str) -> SpanTreeOutcome:
        stage_dir = self.run_root / "06_tree"
        span_dir = stage_dir / "span_001"
        upstream_digest = canonical_digest(
            {"regions": regions_digest, "wire": wire_digest}
        )
        cached = read_checkpoint(
            span_dir, "tree",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream_digest,
        )
        if cached is not None and (span_dir / "tree_members.npz").is_file():
            return load_span_tree_outcome(self.span, span_dir)
        veg_region = load_raw_region(self.vegetation_region_dir / "region_sidecar.json")
        cleaned_region = self._tree_input_excluding_confirmed_wire(veg_region, span_dir)
        outcome = run_span_tree_pipeline(
            self.span,
            cleaned_region,
            self.coarse_ground_xyz_absolute,
            self.cfg,
            output_dir=span_dir,
            origin_xyz=self.origin_xyz,
        )
        payload = outcome.to_manifest()
        _write_json(span_dir / "tree_summary.json", payload)
        write_checkpoint(
            span_dir, "tree",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream_digest,
            summary=payload,
            output_digest=canonical_digest(
                {"tree_count": len(outcome.tree_records)}
            ),
        )
        return outcome

    def _tree_input_excluding_confirmed_wire(
        self, veg_region: Any, span_dir: Path
    ) -> Any:
        """Subtract exact FINAL confirmed conductor members from tree input.

        ``FINAL_TREE_INPUT = VEGETATION - CONFIRMED_WIRE - CONFIRMED_TOWER``.
        Only ``wire_outcome.confirmed_member_source_index`` — the exact
        FULL_RESOLUTION source rows of FINAL confirmed conductors — is removed
        here; broad (unconfirmed) coarse wire candidates never exclude
        vegetation/tree membership.
        """

        confirmed = np.asarray(
            self.wire_outcome.confirmed_member_source_index, dtype=np.int64
        )
        if len(confirmed) == 0:
            return veg_region
        from algorithms.spans.region import subset_raw_region_excluding_source_indices

        return subset_raw_region_excluding_source_indices(
            veg_region,
            confirmed,
            span_dir / "tree_input",
        )

    def _stage_clearance(
        self, wire_digest: str, tree_digest: str
    ) -> ManualClearanceResult:
        stage_dir = self.run_root / "07_clearance"
        sidecar = stage_dir / "clearance_sidecar.json"
        danger_npz = stage_dir / "danger_candidates.npz"
        upstream = canonical_digest(
            {"wire": wire_digest, "tree": tree_digest, "span": self.span_digest}
        )
        cached = read_checkpoint(
            stage_dir, "clearance",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream,
        )
        if cached is not None and sidecar.is_file() and danger_npz.is_file():
            return _load_manual_clearance(stage_dir, danger_npz)

        result = run_manual_clearance(
            self.reconciled, [self.wire_outcome], self.cfg, self.manual
        )
        _write_manual_clearance(stage_dir, result, self.origin_xyz)
        write_checkpoint(
            stage_dir, "clearance",
            source_sha256=self.source_sha256, source_id=self.source_id,
            config_digest=self.cfg.computed_digest(), upstream_digest=upstream,
            summary={
                "status": result.clearance.status,
                "global_minimum_clearance_m": result.clearance.global_minimum_clearance_m,
                "edge_count": result.clearance.edge_count,
                "tree_count": len(result.clearance.records),
                "danger_candidate_point_count": int(
                    result.clearance.diagnostics.get("danger_candidate_point_count", 0)
                ),
            },
            output_digest=canonical_digest(
                {
                    "status": result.clearance.status,
                    "records": [record.to_dict() for record in result.clearance.records],
                }
            ),
        )
        return result

    # ------------------------------------------------------------- finalize
    def finalize(
        self,
        threshold_m: float,
        *,
        open_results: bool = True,
        progress_sink: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Apply the risk threshold, classify danger, and produce all outputs."""

        if self.clearance is None:
            raise RuntimeError("MANUAL_SPAN_NO_ANALYSIS")
        _report(self.progress, 9, STAGE_LABELS[9], False, None)
        started = time.perf_counter()
        threshold = float(threshold_m)
        if not (
            self.manual.risk_threshold_min_m
            <= threshold
            <= self.manual.risk_threshold_max_m
        ):
            raise ValueError("MANUAL_SPAN_THRESHOLD_OUT_OF_RANGE")
        self.risk = apply_risk_threshold(
            self.clearance,
            self.metrics_by_tree,
            self.reconciled,
            self.origin_xyz,
            threshold,
        )
        risk_dir = self.run_root / "09_risk"
        _write_json(
            risk_dir / "risk_result.json",
            {
                "schema_version": MANUAL_PIPELINE_SCHEMA,
                "threshold_m": threshold,
                "dangerous_tree_count": self.risk.dangerous_tree_count,
                "danger_point_count": self.risk.danger_point_count,
                "records": [record.to_dict() for record in self.risk.records],
            },
        )
        risk_payload = risk_threshold_payload(
            self.source_sha256,
            self.source_id,
            self.span_digest,
            threshold,
            self.risk.dangerous_tree_count,
            self.risk.danger_point_count,
            self.clearance.clearance.global_minimum_clearance_m,
        )
        _write_json(self.run_root / "risk_threshold.json", risk_payload)
        _report(self.progress, 9, STAGE_LABELS[9], True, time.perf_counter() - started)

        # Visuals / Excel / report.
        _report(self.progress, 10, STAGE_LABELS[10], False, None)
        started = time.perf_counter()
        self.timings["finalize"] = time.perf_counter() - started
        self.timings["total"] = time.perf_counter() - self._run_started
        from algorithms.manual_span.report import build_run_summary, render_final_report

        summary = build_run_summary(self)
        _write_json(self.run_root / "run_summary.json", summary)

        from algorithms.manual_span.excel import write_tree_clearance_excel
        from algorithms.manual_span.visuals import generate_all_visuals

        visual_root = self.run_root / "10_visuals"
        report_root = self.run_root / "11_report"
        generate_all_visuals(
            self, visual_root, summary, progress_sink=progress_sink
        )
        excel_path = write_tree_clearance_excel(self, report_root / "tree_clearance_report.xlsx")
        report_html = render_final_report(self, summary, visual_root)
        _write_text(report_root / "27_FINAL_REPORT.html", report_html)
        _write_text(
            self.run_root / "OPEN_FINAL_RESULTS.bat",
            _open_results_bat(
                visual_root / "00_VISUAL_INDEX.html",
                visual_root / "26_FINAL_SPAN_OVERVIEW.html",
                excel_path,
            ),
        )
        _report(self.progress, 10, STAGE_LABELS[10], True, time.perf_counter() - started)

        if open_results:
            import webbrowser

            for path in (
                visual_root / "00_VISUAL_INDEX.html",
                visual_root / "26_FINAL_SPAN_OVERVIEW.html",
                excel_path,
            ):
                try:
                    webbrowser.open(path.resolve().as_uri())
                except OSError:  # pragma: no cover - best effort
                    pass
        return summary


class PoleRefinementError(RuntimeError):
    """A clicked pole could not be refined; the UI asks for a re-click."""

    def __init__(
        self, label: str, reason: str | None, diagnostics: dict[str, Any] | None = None
    ) -> None:
        super().__init__(f"POLE_{label}_REFINEMENT_FAILED: {reason}")
        self.label = label
        self.reason = reason
        self.refinement_diagnostics = diagnostics or {}


def _write_manual_clearance(
    stage_dir: Path, result: ManualClearanceResult, origin_xyz: tuple[float, float, float]
) -> None:
    _write_json(
        stage_dir / "tree_wire_clearance.json",
        {
            "status": result.clearance.status,
            "global_minimum_clearance_m": result.clearance.global_minimum_clearance_m,
            "records": [record.to_dict() for record in result.clearance.records],
            "edge_count": result.clearance.edge_count,
            "diagnostics": dict(result.clearance.diagnostics),
        },
    )
    _write_json(
        stage_dir / "clearance_sidecar.json",
        {
            "schema_version": MANUAL_PIPELINE_SCHEMA,
            "status": result.clearance.status,
            "global_minimum_clearance_m": result.clearance.global_minimum_clearance_m,
            "records": [record.to_dict() for record in result.clearance.records],
            "edge_count": result.clearance.edge_count,
            "origin_xyz": list(origin_xyz),
        },
    )
    _write_npz(
        stage_dir / "danger_candidates.npz",
        original_index=result.danger_original_index,
        xyz_local_m=result.danger_xyz_local_m,
        distance_m=result.danger_distance_m,
        wire_id=result.danger_wire_id,
        segment_index=result.danger_segment_index,
        edge_index=result.danger_edge_index,
        tree_id=result.danger_tree_id,
        nearest_wire_xyz_local_m=result.danger_nearest_wire_xyz_local_m,
    )


def _load_manual_clearance(stage_dir: Path, danger_npz: Path) -> ManualClearanceResult:
    from algorithms.manual_span.clearance_fast import ManualClearanceResult as _MCR
    from algorithms.spans.clearance import SpanClearanceResult

    sidecar = json.loads((stage_dir / "clearance_sidecar.json").read_text("utf-8"))
    records = tuple(
        SpanTreeWireRecord(
            global_tree_id=int(item["global_tree_id"]),
            external_name=str(item["external_name"]),
            line_id=int(item["line_id"]),
            span_id=int(item["span_id"]),
            span_name=str(item["span_name"]),
            nearest_wire_id=item["nearest_wire_id"],
            minimum_clearance_m=item["minimum_clearance_m"],
            nearest_tree_original_index=item["nearest_tree_original_index"],
            nearest_tree_xyz_local_m=(
                None
                if item["nearest_tree_xyz_local_m"] is None
                else tuple(float(value) for value in item["nearest_tree_xyz_local_m"])
            ),
            nearest_wire_xyz_local_m=(
                None
                if item["nearest_wire_xyz_local_m"] is None
                else tuple(float(value) for value in item["nearest_wire_xyz_local_m"])
            ),
            tree_point_count=int(item["tree_point_count"]),
            evaluated_edge_count=int(item["evaluated_edge_count"]),
            status=str(item["status"]),
        )
        for item in sidecar["records"]
    )
    clearance = SpanClearanceResult(
        records=records,
        status=str(sidecar["status"]),
        global_minimum_clearance_m=sidecar["global_minimum_clearance_m"],
        edge_count=int(sidecar["edge_count"]),
        diagnostics={"schema": MANUAL_PIPELINE_SCHEMA},
    )
    with np.load(danger_npz, allow_pickle=False) as stream:
        return _MCR(
            clearance=clearance,
            danger_original_index=np.asarray(stream["original_index"], dtype=np.int64),
            danger_xyz_local_m=np.asarray(stream["xyz_local_m"], dtype=np.float32),
            danger_distance_m=np.asarray(stream["distance_m"], dtype=np.float64),
            danger_wire_id=np.asarray(stream["wire_id"], dtype=np.int32),
            danger_segment_index=np.asarray(stream["segment_index"], dtype=np.int32),
            danger_edge_index=np.asarray(stream["edge_index"], dtype=np.int32),
            danger_tree_id=np.asarray(stream["tree_id"], dtype=np.int32),
            danger_nearest_wire_xyz_local_m=np.asarray(
                stream["nearest_wire_xyz_local_m"], dtype=np.float32
            ),
        )


def _open_results_bat(index_html: Path, overview_html: Path, excel_path: Path) -> str:
    def _win(path: Path) -> str:
        return str(path.resolve())

    lines = [
        "@echo off",
        "chcp 65001 >nul",
        "echo 正在打开最终结果...",
        f'start "" "{_win(index_html)}"',
        f'start "" "{_win(overview_html)}"',
        f'start "" "{_win(excel_path)}"',
        "exit /b 0",
    ]
    return "\r\n".join(lines) + "\r\n"


__all__ = [
    "MANUAL_PIPELINE_SCHEMA",
    "STAGE_LABELS",
    "ManualSingleSpanRun",
    "PoleRefinementError",
]
