"""Non-certified, reproducible fast-research pipeline building blocks.

This module intentionally lives outside the formal FGC/FRV authority chain.  It
uses PDAL SMRF plus HAG-NN to produce a practical demo input, then adapts an
explicit, deterministic vegetation admission set into the existing V4 runtime.
The compatibility bridge is in-memory only and every persisted result carries a
``NON_CERTIFIED_FAST_RESEARCH`` marker in its sidecar manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts.frv import (
    FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
    FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
    FrvMembershipState,
    FrvRelocationAudit,
    FrvSegmentationHagReadiness,
    SourceIdentity,
    frv_digest,
)
from core.contracts.v4_inputs import (
    V4InputHandoffResult,
    V4RecoveredVegetationSet,
    V4RecoveryCoverageReport,
    V4ValidatedFrvMembershipView,
    frv_v4_recovered_index_digest,
)
from core.contracts.v4_tree_segmentation import (
    V4ConfigStatus,
    V4TreeSegmentationConfig,
)

from algorithms.vegetation.v4_tree_artifacts import (
    load_v4_tree_segmentation_artifact,
    publish_v4_tree_segmentation_artifact,
)
from algorithms.vegetation.v4_tree_segmentation import segment_v4_trees

FAST_RESEARCH_SCHEMA_VERSION = "fast-research-pipeline-v1"
NON_CERTIFIED_MARKER = "NON_CERTIFIED_FAST_RESEARCH"
NON_CERTIFIED_INPUT_MARKER = "NON_CERTIFIED_FAST_RESEARCH_INPUT"
FAST_GROUND_ALGORITHM = "PDAL_SMRF_v1"
FAST_HAG_ALGORITHM = "PDAL_HAG_NN_v1"


class FastResearchError(RuntimeError):
    """A concise failure raised by the non-certified research route."""


@dataclass(frozen=True, slots=True)
class FastGroundResult:
    """Digest-bound, non-certified result of the PDAL Ground/HAG stage."""

    schema_version: str
    non_certified_marker: str
    source_las_path: str
    source_las_sha256: str
    output_las_path: str
    output_las_sha256: str
    algorithm_name: str
    hag_algorithm_name: str
    parameters: dict[str, Any]
    config_digest: str
    source_point_count: int
    output_point_count: int
    ground_class_2_count: int
    non_ground_count: int
    hag_valid_count: int
    hag_min_m: float | None
    hag_max_m: float | None
    xyz_order_preserved: bool
    source_xyz_sha256: str
    output_xyz_sha256: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FastGroundResult:
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise FastResearchError("FAST_GROUND_RESULT_INVALID: unexpected result fields")
        result = cls(**payload)
        if result.schema_version != FAST_RESEARCH_SCHEMA_VERSION:
            raise FastResearchError("FAST_GROUND_RESULT_INVALID: unsupported schema")
        if result.non_certified_marker != NON_CERTIFIED_MARKER:
            raise FastResearchError("FAST_GROUND_RESULT_INVALID: certification marker")
        if not _is_sha(result.source_las_sha256) or not _is_sha(result.output_las_sha256):
            raise FastResearchError("FAST_GROUND_RESULT_INVALID: invalid file digest")
        if result.source_point_count < 0 or result.output_point_count < 0:
            raise FastResearchError("FAST_GROUND_RESULT_INVALID: invalid point count")
        return result


@dataclass(frozen=True, slots=True)
class FastVegetationInputResult:
    """Deterministic V4 adapter input, explicitly outside formal FRV."""

    schema_version: str
    non_certified_marker: str
    ground_result_digest: str
    config_digest: str
    output_npz_path: str
    output_npz_sha256: str
    point_count: int
    source_id: str
    source_las_path: str
    source_point_count: int
    source_file_sha256: str
    origin_context: tuple[float, float, float]
    selected_bounds_min_xyz: tuple[float, float, float] | None
    selected_bounds_max_xyz: tuple[float, float, float] | None
    minimum_hag_m: float
    admission_voxel_size_m: float | None
    original_index_sha256: str
    xyz_sha256: str
    hag_sha256: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("origin_context", "selected_bounds_min_xyz", "selected_bounds_max_xyz"):
            if payload[key] is not None:
                payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FastVegetationInputResult:
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise FastResearchError("FAST_V4_INPUT_INVALID: unexpected result fields")
        values = dict(payload)
        for key in ("origin_context", "selected_bounds_min_xyz", "selected_bounds_max_xyz"):
            if values[key] is not None:
                values[key] = tuple(float(item) for item in values[key])
        result = cls(**values)
        if result.schema_version != FAST_RESEARCH_SCHEMA_VERSION:
            raise FastResearchError("FAST_V4_INPUT_INVALID: unsupported schema")
        if result.non_certified_marker != NON_CERTIFIED_INPUT_MARKER:
            raise FastResearchError("FAST_V4_INPUT_INVALID: certification marker")
        return result


def canonical_digest(payload: dict[str, Any]) -> str:
    """Return the stable SHA-256 used for fast-pipeline configuration keys."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fast_ground_content_digest(result: FastGroundResult) -> str:
    """Hash only reproducibility-relevant Ground/HAG facts, never timing data."""

    return canonical_digest(
        {
            "schema_version": result.schema_version,
            "non_certified_marker": result.non_certified_marker,
            "source_las_sha256": result.source_las_sha256,
            "output_las_sha256": result.output_las_sha256,
            "algorithm_name": result.algorithm_name,
            "hag_algorithm_name": result.hag_algorithm_name,
            "parameters": result.parameters,
            "config_digest": result.config_digest,
            "source_point_count": result.source_point_count,
            "output_point_count": result.output_point_count,
            "ground_class_2_count": result.ground_class_2_count,
            "non_ground_count": result.non_ground_count,
            "hag_valid_count": result.hag_valid_count,
            "hag_min_m": result.hag_min_m,
            "hag_max_m": result.hag_max_m,
            "xyz_order_preserved": result.xyz_order_preserved,
            "source_xyz_sha256": result.source_xyz_sha256,
            "output_xyz_sha256": result.output_xyz_sha256,
        }
    )


def sha256_file(path: Path) -> str:
    """Hash a file without ever changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_fast_ground_hag(
    input_las: Path,
    output_root: Path,
    *,
    smrf_cell_m: float = 8.0,
    smrf_slope: float = 0.2,
    smrf_threshold_m: float = 0.45,
    smrf_window_m: float = 16.0,
    hag_neighbor_count: int = 8,
) -> tuple[str, FastGroundResult]:
    """Run or safely reuse the PDAL SMRF plus HAG-NN research stage."""

    input_las = input_las.resolve()
    output_root = output_root.resolve()
    if not input_las.is_file():
        raise FastResearchError(f"FAST_GROUND_INPUT_MISSING: {input_las}")
    if smrf_cell_m <= 0 or smrf_slope <= 0 or smrf_threshold_m < 0 or smrf_window_m <= 0:
        raise FastResearchError("FAST_GROUND_CONFIG_INVALID")
    if hag_neighbor_count <= 0:
        raise FastResearchError("FAST_HAG_CONFIG_INVALID")

    source_sha = sha256_file(input_las)
    parameters = {
        "smrf_cell_m": float(smrf_cell_m),
        "smrf_slope": float(smrf_slope),
        "smrf_threshold_m": float(smrf_threshold_m),
        "smrf_window_m": float(smrf_window_m),
        "hag_neighbor_count": int(hag_neighbor_count),
        "hag_allow_extrapolation": True,
    }
    config_digest = canonical_digest(
        {
            "schema_version": FAST_RESEARCH_SCHEMA_VERSION,
            "algorithm": FAST_GROUND_ALGORITHM,
            "hag_algorithm": FAST_HAG_ALGORITHM,
            "input_sha256": source_sha,
            "parameters": parameters,
        }
    )
    output_las = output_root / "classified_hag.las"
    result_path = output_root / "ground_hag_result.json"
    cached = _load_fast_ground_cache(result_path, output_las, source_sha, config_digest)
    if cached is not None:
        return "SKIPPED_FROM_CACHE", cached

    try:
        import pdal
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise FastResearchError(
            "FAST_GROUND_BACKEND_UNAVAILABLE: Python PDAL binding missing"
        ) from exc

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_output = output_las.with_suffix(".tmp.las")
    temporary_output.unlink(missing_ok=True)
    pipeline_definition = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_las)},
            {
                "type": "filters.smrf",
                "cell": float(smrf_cell_m),
                "slope": float(smrf_slope),
                "threshold": float(smrf_threshold_m),
                "window": float(smrf_window_m),
            },
            {
                "type": "filters.hag_nn",
                "count": int(hag_neighbor_count),
                "allow_extrapolation": True,
            },
            {
                "type": "writers.las",
                "filename": str(temporary_output),
                "extra_dims": "all",
                "forward": "all",
            },
        ]
    }
    started = time.perf_counter()
    try:
        processed = int(pdal.Pipeline(json.dumps(pipeline_definition)).execute())
        os.replace(temporary_output, output_las)
        result = _inspect_fast_ground_output(
            input_las=input_las,
            output_las=output_las,
            source_sha=source_sha,
            parameters=parameters,
            config_digest=config_digest,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        temporary_output.unlink(missing_ok=True)
        raise FastResearchError(f"FAST_GROUND_FAILED: {exc}") from exc
    if processed != result.output_point_count:
        raise FastResearchError(
            "FAST_GROUND_FAILED: PDAL point count does not match written output"
        )
    _write_json_atomic(result_path, result.to_dict())
    return "PASS", result


def build_fast_vegetation_input(
    ground_result: FastGroundResult,
    output_root: Path,
    *,
    source_id: str,
    origin_context: tuple[float, float, float],
    minimum_hag_m: float = 0.3,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    admission_voxel_size_m: float | None = None,
) -> tuple[str, FastVegetationInputResult]:
    """Create a deterministic, HAG-ready V4 admission set from fast output.

    A positive ``admission_voxel_size_m`` selects one *real measured point* per
    voxel, using the lowest original index.  It is deliberately explicit in the
    manifest rather than being hidden as an authority or a V4 operation.
    """

    if ground_result.non_certified_marker != NON_CERTIFIED_MARKER:
        raise FastResearchError("FAST_V4_INPUT_REQUIRES_NON_CERTIFIED_GROUND")
    if not source_id:
        raise FastResearchError("FAST_V4_INPUT_INVALID_SOURCE_ID")
    if not math.isfinite(minimum_hag_m):
        raise FastResearchError("FAST_V4_INPUT_INVALID_MINIMUM_HAG")
    if admission_voxel_size_m is not None and admission_voxel_size_m <= 0:
        raise FastResearchError("FAST_V4_INPUT_INVALID_VOXEL_SIZE")
    source_las = Path(ground_result.source_las_path)
    output_las = Path(ground_result.output_las_path)
    if not source_las.is_file() or not output_las.is_file():
        raise FastResearchError("FAST_V4_INPUT_GROUND_OUTPUT_MISSING")
    if sha256_file(output_las) != ground_result.output_las_sha256:
        raise FastResearchError("FAST_V4_INPUT_GROUND_OUTPUT_SHA_MISMATCH")

    output_root = output_root.resolve()
    normalized_bounds = _normalize_bounds(bounds)
    config_payload = {
        "schema_version": FAST_RESEARCH_SCHEMA_VERSION,
        "ground_result_digest": fast_ground_content_digest(ground_result),
        "source_id": source_id,
        "origin_context": list(origin_context),
        "minimum_hag_m": float(minimum_hag_m),
        "bounds": _bounds_payload(normalized_bounds),
        "admission_voxel_size_m": admission_voxel_size_m,
    }
    config_digest = canonical_digest(config_payload)
    output_npz = output_root / "fast_v4_input.npz"
    result_path = output_root / "fast_v4_input.json"
    cached = _load_fast_vegetation_cache(result_path, output_npz, config_digest)
    if cached is not None:
        return "SKIPPED_FROM_CACHE", cached

    started = time.perf_counter()
    original_index, xyz_local_m, hag_m = _select_fast_vegetation_rows(
        output_las,
        origin_context=origin_context,
        minimum_hag_m=minimum_hag_m,
        bounds=normalized_bounds,
        admission_voxel_size_m=admission_voxel_size_m,
    )
    validate_fast_v4_arrays(original_index, xyz_local_m, hag_m)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_npz_atomic(
        output_npz,
        original_index=original_index,
        xyz_local_m=xyz_local_m,
        hag_m=hag_m,
        candidate_reason_flags=np.zeros(len(original_index), dtype=np.uint32),
    )
    output_sha = sha256_file(output_npz)
    with laspy.open(source_las) as source_reader:
        source_header = source_reader.header
        point_count = int(source_header.point_count)
        tuple(float(value) for value in source_header.mins)
        tuple(float(value) for value in source_header.maxs)
    result = FastVegetationInputResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_INPUT_MARKER,
        ground_result_digest=fast_ground_content_digest(ground_result),
        config_digest=config_digest,
        output_npz_path=str(output_npz),
        output_npz_sha256=output_sha,
        point_count=len(original_index),
        source_id=source_id,
        source_las_path=str(source_las.resolve()),
        source_point_count=point_count,
        source_file_sha256=ground_result.source_las_sha256,
        origin_context=tuple(float(value) for value in origin_context),
        selected_bounds_min_xyz=(normalized_bounds[0] if normalized_bounds else None),
        selected_bounds_max_xyz=(normalized_bounds[1] if normalized_bounds else None),
        minimum_hag_m=float(minimum_hag_m),
        admission_voxel_size_m=admission_voxel_size_m,
        original_index_sha256=_array_digest(original_index),
        xyz_sha256=_array_digest(xyz_local_m),
        hag_sha256=_array_digest(hag_m),
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_json_atomic(result_path, result.to_dict())
    return "PASS", result


def generate_fast_ground_preview(
    ground_result: FastGroundResult, output_html: Path, *, maximum_points: int = 60_000
) -> Path:
    """Render a deterministic ground/HAG inspection view without recomputing it."""

    if maximum_points <= 0:
        raise FastResearchError("FAST_GROUND_PREVIEW_INVALID_MAXIMUM_POINTS")
    output_las = Path(ground_result.output_las_path)
    if not output_las.is_file() or sha256_file(output_las) != ground_result.output_las_sha256:
        raise FastResearchError("FAST_GROUND_PREVIEW_INPUT_INVALID")
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise FastResearchError("FAST_GROUND_PREVIEW_DEPENDENCY_MISSING") from exc
    with laspy.open(output_las) as reader:
        count = int(reader.header.point_count)
        targets = np.linspace(0, count - 1, min(count, maximum_points), dtype=np.int64)
        xyz_parts: list[np.ndarray] = []
        hag_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        offset = 0
        for chunk in reader.chunk_iterator(500_000):
            local = targets[(targets >= offset) & (targets < offset + len(chunk))] - offset
            if len(local):
                xyz_parts.append(
                    np.column_stack(
                        (
                            np.asarray(chunk.x)[local],
                            np.asarray(chunk.y)[local],
                            np.asarray(chunk.z)[local],
                        )
                    )
                )
                hag_parts.append(np.asarray(chunk["HeightAboveGround"], dtype=np.float64)[local])
                class_parts.append(np.asarray(chunk.classification)[local])
            offset += len(chunk)
    xyz = np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3), dtype=np.float64)
    hag = np.concatenate(hag_parts) if hag_parts else np.empty(0, dtype=np.float64)
    classification = np.concatenate(class_parts) if class_parts else np.empty(0, dtype=np.uint8)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            name="HAG (non-ground)",
            marker={"size": 1.8, "color": hag, "colorscale": "Viridis", "opacity": 0.72},
            customdata=np.column_stack((hag, classification)),
            hovertemplate="HAG=%{customdata[0]:.2f} m<br>class=%{customdata[1]}<extra></extra>",
        )
    )
    ground = classification == 2
    figure.add_trace(
        go.Scatter3d(
            x=xyz[ground, 0],
            y=xyz[ground, 1],
            z=xyz[ground, 2],
            mode="markers",
            name="SMRF ground (class 2)",
            marker={"size": 1.7, "color": "#92400e", "opacity": 0.75},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title="Fast research preview — PDAL SMRF ground + HAG-NN (non-certified)",
        scene={"xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": "Z (m)"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_html.with_name(output_html.name + ".tmp")
    figure.write_html(temporary, include_plotlyjs=True, full_html=True)
    os.replace(temporary, output_html)
    return output_html


def validate_fast_v4_arrays(
    original_index: np.ndarray, xyz_local_m: np.ndarray, hag_m: np.ndarray
) -> None:
    """Validate the adapter's full correspondence before V4 can consume it."""

    if original_index.dtype != np.dtype("<i8") or original_index.ndim != 1:
        raise FastResearchError("FAST_V4_INPUT_INVALID: original_index dtype or shape")
    count = len(original_index)
    if xyz_local_m.dtype != np.dtype("<f8") or xyz_local_m.shape != (count, 3):
        raise FastResearchError("FAST_V4_INPUT_INVALID: xyz dtype or shape")
    if hag_m.dtype != np.dtype("<f4") or hag_m.shape != (count,):
        raise FastResearchError("FAST_V4_INPUT_INVALID: HAG dtype or shape")
    if np.any(original_index < 0) or (count > 1 and not np.all(np.diff(original_index) > 0)):
        raise FastResearchError("FAST_V4_INPUT_INVALID: original_index correspondence")
    if not np.isfinite(xyz_local_m).all() or not np.isfinite(hag_m).all():
        raise FastResearchError("FAST_V4_INPUT_INVALID: non-finite input")


def run_fast_v4(
    vegetation_result: FastVegetationInputResult,
    output_root: Path,
    *,
    source_bounds_min_xyz: tuple[float, float, float],
    source_bounds_max_xyz: tuple[float, float, float],
) -> tuple[str, Path, dict[str, Any]]:
    """Run the frozen V4 implementation through a clearly non-formal bridge."""

    input_npz = Path(vegetation_result.output_npz_path)
    if not input_npz.is_file() or sha256_file(input_npz) != vegetation_result.output_npz_sha256:
        raise FastResearchError("FAST_V4_INPUT_ARTIFACT_INVALID")
    output_root = output_root.resolve()
    v4_root = output_root / "v4_artifact"
    bridge_path = output_root / "fast_v4_adapter.json"
    config = _fast_v4_config(maximum_points=max(200_000, vegetation_result.point_count))
    run_digest = canonical_digest(
        {
            "vegetation_input_sha256": vegetation_result.output_npz_sha256,
            "vegetation_config_digest": vegetation_result.config_digest,
            "v4_config_digest": config.config_digest,
            "source_bounds_min_xyz": list(source_bounds_min_xyz),
            "source_bounds_max_xyz": list(source_bounds_max_xyz),
        }
    )
    if v4_root.is_dir() and bridge_path.is_file():
        try:
            bridge = json.loads(bridge_path.read_text("utf-8"))
            loaded = load_v4_tree_segmentation_artifact(v4_root)
            if (
                bridge.get("run_digest") == run_digest
                and bridge.get("non_certified_marker") == NON_CERTIFIED_INPUT_MARKER
            ):
                return "SKIPPED_FROM_CACHE", v4_root, _v4_summary(loaded.result)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    with np.load(input_npz, allow_pickle=False) as stream:
        original_index = np.asarray(stream["original_index"], dtype="<i8")
        xyz_local_m = np.asarray(stream["xyz_local_m"], dtype="<f8")
        hag_m = np.asarray(stream["hag_m"], dtype="<f4")
        candidate_reason_flags = np.asarray(stream["candidate_reason_flags"], dtype="<u4")
        raw_weight = (
            np.asarray(stream["raw_count"], dtype="<u4")
            if "raw_count" in stream.files
            else None
        )
    validate_fast_v4_arrays(original_index, xyz_local_m, hag_m)
    source = _build_fast_source_identity(
        vegetation_result,
        source_bounds_min_xyz=source_bounds_min_xyz,
        source_bounds_max_xyz=source_bounds_max_xyz,
    )
    handoff = _build_fast_handoff(
        source,
        original_index=original_index,
        xyz_local_m=xyz_local_m,
        hag_m=hag_m,
        candidate_reason_flags=candidate_reason_flags,
        raw_weight=raw_weight,
        bridge_digest=run_digest,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if v4_root.exists():
        raise FastResearchError(
            "FAST_V4_OUTPUT_INVALID_CACHE: artifact root exists but does not validate"
        )
    started = time.perf_counter()
    result = segment_v4_trees(handoff, config)
    reference = publish_v4_tree_segmentation_artifact(result, v4_root, scratch_dir=output_root)
    summary = _v4_summary(result)
    bridge_payload = {
        "schema_version": FAST_RESEARCH_SCHEMA_VERSION,
        "non_certified_marker": NON_CERTIFIED_INPUT_MARKER,
        "adapter": "FAST_RESEARCH_V4_ADAPTER",
        "density_source": (
            "RAW_WEIGHTED_COUNT" if raw_weight is not None else "REPRESENTATIVE_COUNT"
        ),
        "authority": "No FRV authority, proof, or acceptance claim is made.",
        "run_digest": run_digest,
        "vegetation_input": vegetation_result.to_dict(),
        "v4_config": config.to_dict(),
        "v4_result_digest": result.result_digest,
        "v4_artifact_root": str(reference.root_path),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }
    _write_json_atomic(bridge_path, bridge_payload)
    return "PASS", v4_root, summary


def prepare_w1_reuse_snapshot(
    w1_npz: Path,
    w1_json: Path,
    source_project_root: Path,
    output_root: Path,
) -> tuple[str, dict[str, Path]]:
    """Copy an existing W1/A4 set byte-for-byte for isolated read-only reuse."""

    from algorithms.wires.reconstruct import load_wire_models

    w1_npz, w1_json = w1_npz.resolve(), w1_json.resolve()
    source_project_root, output_root = source_project_root.resolve(), output_root.resolve()
    result, _arrays = load_wire_models(w1_npz, w1_json)
    parent_npz = source_project_root / Path(result.input_a4_npz_relpath)
    parent_json = source_project_root / Path(result.input_a4_json_relpath)
    if not parent_npz.is_file() or not parent_json.is_file():
        raise FastResearchError("FAST_W1_REUSE_PARENT_MISSING")
    if (
        sha256_file(parent_npz) != result.input_a4_npz_sha256
        or sha256_file(parent_json) != result.input_a4_json_sha256
    ):
        raise FastResearchError("FAST_W1_REUSE_PARENT_SHA_MISMATCH")
    destination_root = output_root / "w1_context"
    files = {
        "w1_npz": destination_root / "outputs" / "wire_models" / w1_npz.name,
        "w1_json": destination_root / "outputs" / "wire_models" / w1_json.name,
        "a4_npz": destination_root / Path(result.input_a4_npz_relpath),
        "a4_json": destination_root / Path(result.input_a4_json_relpath),
        "project_root": destination_root,
    }
    manifest_path = output_root / "w1_reuse.json"
    expected = {
        "w1_npz_sha256": sha256_file(w1_npz),
        "w1_json_sha256": sha256_file(w1_json),
        "a4_npz_sha256": sha256_file(parent_npz),
        "a4_json_sha256": sha256_file(parent_json),
    }
    if manifest_path.is_file() and all(
        path.is_file() for key, path in files.items() if key != "project_root"
    ):
        try:
            cached = json.loads(manifest_path.read_text("utf-8"))
            if (
                cached.get("source_hashes") == expected
                and cached.get("reuse_mode") == "EXACT_READ_ONLY_SNAPSHOT"
            ):
                return "SKIPPED_FROM_CACHE", files
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if destination_root.exists():
        raise FastResearchError("FAST_W1_REUSE_INVALID_CACHE")
    for source, destination in (
        (w1_npz, files["w1_npz"]),
        (w1_json, files["w1_json"]),
        (parent_npz, files["a4_npz"]),
        (parent_json, files["a4_json"]),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": FAST_RESEARCH_SCHEMA_VERSION,
            "non_certified_marker": NON_CERTIFIED_MARKER,
            "reuse_mode": "EXACT_READ_ONLY_SNAPSHOT",
            "authority": "W1 metadata and bytes are copied unchanged; no relabeling occurred.",
            "source_paths": {
                "w1_npz": str(w1_npz),
                "w1_json": str(w1_json),
                "a4_npz": str(parent_npz),
                "a4_json": str(parent_json),
            },
            "source_hashes": expected,
            "snapshot_paths": {key: str(value) for key, value in files.items()},
        },
    )
    return "PASS", files


def _inspect_fast_ground_output(
    *,
    input_las: Path,
    output_las: Path,
    source_sha: str,
    parameters: dict[str, Any],
    config_digest: str,
    elapsed_seconds: float,
) -> FastGroundResult:
    source_xyz_digest = hashlib.sha256()
    output_xyz_digest = hashlib.sha256()
    source_count = output_count = class_2_count = hag_valid_count = 0
    hag_min = float("inf")
    hag_max = float("-inf")
    order_preserved = True
    with laspy.open(input_las) as source, laspy.open(output_las) as output:
        if "HeightAboveGround" not in output.header.point_format.extra_dimension_names:
            raise FastResearchError("FAST_HAG_OUTPUT_INVALID: HeightAboveGround is missing")
        for source_chunk, output_chunk in zip(
            source.chunk_iterator(500_000), output.chunk_iterator(500_000), strict=True
        ):
            source_count += len(source_chunk)
            output_count += len(output_chunk)
            source_xyz = tuple(np.asarray(getattr(source_chunk, name)) for name in ("X", "Y", "Z"))
            output_xyz = tuple(np.asarray(getattr(output_chunk, name)) for name in ("X", "Y", "Z"))
            order_preserved = order_preserved and all(
                np.array_equal(left, right)
                for left, right in zip(source_xyz, output_xyz, strict=True)
            )
            for values in source_xyz:
                source_xyz_digest.update(values.tobytes())
            for values in output_xyz:
                output_xyz_digest.update(values.tobytes())
            classification = np.asarray(output_chunk.classification)
            class_2_count += int(np.count_nonzero(classification == 2))
            hag = np.asarray(output_chunk["HeightAboveGround"], dtype=np.float64)
            finite = np.isfinite(hag)
            hag_valid_count += int(np.count_nonzero(finite))
            if np.any(finite):
                hag_min = min(hag_min, float(np.min(hag[finite])))
                hag_max = max(hag_max, float(np.max(hag[finite])))
    if source_count != output_count:
        raise FastResearchError("FAST_GROUND_OUTPUT_INVALID: point count changed")
    if not order_preserved:
        raise FastResearchError("FAST_GROUND_OUTPUT_INVALID: XYZ row correspondence changed")
    return FastGroundResult(
        schema_version=FAST_RESEARCH_SCHEMA_VERSION,
        non_certified_marker=NON_CERTIFIED_MARKER,
        source_las_path=str(input_las),
        source_las_sha256=source_sha,
        output_las_path=str(output_las),
        output_las_sha256=sha256_file(output_las),
        algorithm_name=FAST_GROUND_ALGORITHM,
        hag_algorithm_name=FAST_HAG_ALGORITHM,
        parameters=parameters,
        config_digest=config_digest,
        source_point_count=source_count,
        output_point_count=output_count,
        ground_class_2_count=class_2_count,
        non_ground_count=output_count - class_2_count,
        hag_valid_count=hag_valid_count,
        hag_min_m=(None if hag_valid_count == 0 else hag_min),
        hag_max_m=(None if hag_valid_count == 0 else hag_max),
        xyz_order_preserved=order_preserved,
        source_xyz_sha256=source_xyz_digest.hexdigest(),
        output_xyz_sha256=output_xyz_digest.hexdigest(),
        elapsed_seconds=elapsed_seconds,
    )


def _select_fast_vegetation_rows(
    output_las: Path,
    *,
    origin_context: tuple[float, float, float],
    minimum_hag_m: float,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    admission_voxel_size_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_original: list[np.ndarray] = []
    selected_xyz: list[np.ndarray] = []
    selected_hag: list[np.ndarray] = []
    representatives: dict[int, tuple[int, float, float, float, float]] = {}
    offset = 0
    with laspy.open(output_las) as reader:
        header_min = np.asarray(reader.header.mins, dtype=np.float64)
        header_max = np.asarray(reader.header.maxs, dtype=np.float64)
        if bounds is not None and (
            np.any(np.asarray(bounds[0]) < header_min) or np.any(np.asarray(bounds[1]) > header_max)
        ):
            raise FastResearchError("FAST_V4_INPUT_BOUNDS_OUTSIDE_SOURCE")
        for chunk in reader.chunk_iterator(500_000):
            count = len(chunk)
            absolute_xyz = np.column_stack(
                (np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z))
            ).astype("<f8", copy=False)
            hag = np.asarray(chunk["HeightAboveGround"], dtype="<f4")
            candidate = (
                (np.asarray(chunk.classification) != 2)
                & np.isfinite(hag)
                & (hag >= np.float32(minimum_hag_m))
                & np.isfinite(absolute_xyz).all(axis=1)
            )
            if bounds is not None:
                candidate &= np.all(absolute_xyz >= np.asarray(bounds[0]), axis=1)
                candidate &= np.all(absolute_xyz <= np.asarray(bounds[1]), axis=1)
            positions = np.flatnonzero(candidate)
            if admission_voxel_size_m is None:
                selected_original.append((offset + positions).astype("<i8", copy=False))
                selected_xyz.append(
                    absolute_xyz[positions] - np.asarray(origin_context, dtype=np.float64)
                )
                selected_hag.append(hag[positions])
            elif len(positions):
                candidate_xyz = absolute_xyz[positions]
                voxel = np.floor((candidate_xyz - header_min) / admission_voxel_size_m).astype(
                    np.int64
                )
                extents = (
                    np.floor((header_max - header_min) / admission_voxel_size_m).astype(np.int64)
                    + 2
                )
                key = (voxel[:, 0] * extents[1] + voxel[:, 1]) * extents[2] + voxel[:, 2]
                _keys, first = np.unique(key, return_index=True)
                for local_position in first:
                    position = int(positions[int(local_position)])
                    cell_key = int(key[int(local_position)])
                    if cell_key not in representatives:
                        xyz = absolute_xyz[position]
                        representatives[cell_key] = (
                            offset + position,
                            float(xyz[0]),
                            float(xyz[1]),
                            float(xyz[2]),
                            float(hag[position]),
                        )
            offset += count
    if admission_voxel_size_m is not None:
        ordered = sorted(representatives.values(), key=lambda row: row[0])
        original = np.asarray([row[0] for row in ordered], dtype="<i8")
        xyz_absolute = np.asarray([row[1:4] for row in ordered], dtype="<f8")
        hag_values = np.asarray([row[4] for row in ordered], dtype="<f4")
        return original, xyz_absolute - np.asarray(origin_context, dtype=np.float64), hag_values
    if not selected_original:
        return (
            np.empty(0, dtype="<i8"),
            np.empty((0, 3), dtype="<f8"),
            np.empty(0, dtype="<f4"),
        )
    return (
        np.concatenate(selected_original).astype("<i8", copy=False),
        np.concatenate(selected_xyz).astype("<f8", copy=False),
        np.concatenate(selected_hag).astype("<f4", copy=False),
    )


def _fast_v4_config(*, maximum_points: int) -> V4TreeSegmentationConfig:
    fields: dict[str, Any] = {
        "schema_version": "v4-tree-segmentation-config-v1",
        "status": V4ConfigStatus.CALIBRATION_CANDIDATE,
        "xy_cell_size_m": 0.5,
        "apex_neighborhood_radius_m": 1.0,
        "minimum_apex_prominence_m": 0.15,
        "density_trend_window_m": 1.0,
        "slice_thickness_m": 1.0,
        "slice_overlap_fraction": 0.25,
        "relation_neighborhood_radius_m": 4.0,
        "maximum_points": maximum_points,
        "maximum_grid_cells": 400_000,
        "maximum_fragments": 5_000,
        "maximum_local_edges": 50_000,
        "maximum_path_evaluations": 500_000,
        "maximum_profile_samples": 500_000,
    }
    digest_fields = {
        key: (value.value if isinstance(value, V4ConfigStatus) else value)
        for key, value in fields.items()
    }
    return V4TreeSegmentationConfig(**fields, config_digest=frv_digest(digest_fields))


def _build_fast_source_identity(
    vegetation: FastVegetationInputResult,
    *,
    source_bounds_min_xyz: tuple[float, float, float],
    source_bounds_max_xyz: tuple[float, float, float],
) -> SourceIdentity:
    source_path = str(Path(vegetation.source_las_path).resolve())
    audit_fields = {
        "audit_schema_version": FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "windows",
        "recorded_path_raw": source_path,
        "resolved_path_raw": source_path,
        "recorded_path_lexical_normalized": source_path.replace("\\", "/"),
        "resolved_path_lexical_normalized": source_path.replace("\\", "/"),
        "resolved_final_path": source_path,
        "resolution_method": "FAST_RESEARCH_EXPLICIT_INPUT",
        "case_comparison_mode": "WINDOWS_CASE_INSENSITIVE",
        "platform_name": "windows",
        "run_id": f"fast-research-{vegetation.source_file_sha256[:16]}",
        "audit_timestamp": "source-file-stable-fast-research-v1",
        "content_identity_result": "FAST_RESEARCH_NON_CERTIFIED",
        "warning_flags": 0,
    }
    audit = FrvRelocationAudit(**audit_fields, audit_digest=frv_digest(audit_fields))
    content_fields = {
        "source_identity_schema_version": FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": vegetation.source_id,
        "source_file_sha256": vegetation.source_file_sha256,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "canonical_point_content_spec_version": None,
        "source_point_count": vegetation.source_point_count,
        "source_bounds_min_xyz": list(source_bounds_min_xyz),
        "source_bounds_max_xyz": list(source_bounds_max_xyz),
        "coordinate_frame": "LAS_HEADER_SCALED_XYZ",
        "units": "m",
        "origin_context": list(vegetation.origin_context),
    }
    content_digest = frv_digest(content_fields)
    record_fields = {
        **content_fields,
        "recorded_source_path": source_path,
        "resolved_access_path": source_path,
        "relocation_audit_audit_digest": audit.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return SourceIdentity(
        **content_fields,
        recorded_source_path=source_path,
        resolved_access_path=source_path,
        relocation_audit=audit,
        source_equivalence_audit=None,
        source_content_identity_digest=content_digest,
        source_identity_record_digest=frv_digest(record_fields),
    )


def _build_fast_handoff(
    source: SourceIdentity,
    *,
    original_index: np.ndarray,
    xyz_local_m: np.ndarray,
    hag_m: np.ndarray,
    candidate_reason_flags: np.ndarray,
    bridge_digest: str,
    raw_weight: np.ndarray | None = None,
) -> V4InputHandoffResult:
    """Construct the minimum in-memory bridge used solely by V4 Algorithm-V1.

    The standard V4 runtime deliberately has no separate public research input
    type.  This bridge is never written as a FRV artifact and therefore cannot
    be misread as one; its persisted companion is ``fast_v4_adapter.json``.
    """

    def readonly(values: np.ndarray) -> np.ndarray:
        values.setflags(write=False)
        return values

    if raw_weight is None:
        effective_raw_weight = np.ones(len(original_index), dtype="<u4")
    else:
        effective_raw_weight = np.asarray(raw_weight, dtype="<u4")
        if effective_raw_weight.shape != original_index.shape or np.any(effective_raw_weight == 0):
            raise FastResearchError("FAST_V4_RAW_WEIGHT_INVALID")

    recovered = V4RecoveredVegetationSet.__new__(V4RecoveredVegetationSet)
    object.__setattr__(recovered, "source_identity", source)
    object.__setattr__(recovered, "original_index", readonly(original_index.copy()))
    object.__setattr__(recovered, "xyz_local_m", readonly(xyz_local_m.copy()))
    object.__setattr__(recovered, "hag_m", readonly(hag_m.copy()))
    object.__setattr__(
        recovered,
        "membership_state",
        readonly(
            np.full(
                len(original_index),
                int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE),
                dtype=np.uint8,
            )
        ),
    )
    object.__setattr__(
        recovered, "hag_evidence_state", readonly(np.ones(len(original_index), dtype=np.uint8))
    )
    object.__setattr__(
        recovered,
        "segmentation_hag_readiness",
        readonly(
            np.full(len(original_index), int(FrvSegmentationHagReadiness.READY), dtype=np.uint8)
        ),
    )
    object.__setattr__(recovered, "candidate_reason_flags", readonly(candidate_reason_flags.copy()))
    object.__setattr__(recovered, "candidate_confidence", None)
    object.__setattr__(recovered, "frv_membership_artifact_digest", bridge_digest)
    object.__setattr__(
        recovered,
        "recovered_index_digest",
        frv_v4_recovered_index_digest(
            source_content_identity_digest=source.source_content_identity_digest,
            original_index=original_index,
        ),
    )
    object.__setattr__(recovered, "raw_weight", readonly(effective_raw_weight.copy()))
    view = V4ValidatedFrvMembershipView.__new__(V4ValidatedFrvMembershipView)
    object.__setattr__(view, "source_identity", source)
    object.__setattr__(
        view, "view_digest", canonical_digest({"bridge": bridge_digest, "view": "fast"})
    )
    report = V4RecoveryCoverageReport.__new__(V4RecoveryCoverageReport)
    object.__setattr__(
        report, "report_digest", canonical_digest({"bridge": bridge_digest, "report": "fast"})
    )
    handoff = V4InputHandoffResult.__new__(V4InputHandoffResult)
    object.__setattr__(handoff, "validated_membership", view)
    object.__setattr__(handoff, "recovered_vegetation", recovered)
    object.__setattr__(handoff, "unresolved_measured_points", None)
    object.__setattr__(handoff, "unrecovered_regions", ())
    object.__setattr__(handoff, "coverage_report", report)
    object.__setattr__(handoff, "output_reference", None)
    object.__setattr__(handoff, "handoff_digest", bridge_digest)
    return handoff


def _load_fast_ground_cache(
    result_path: Path, output_las: Path, source_sha: str, config_digest: str
) -> FastGroundResult | None:
    if not result_path.is_file() or not output_las.is_file():
        return None
    try:
        result = FastGroundResult.from_dict(json.loads(result_path.read_text("utf-8")))
    except (FastResearchError, OSError, ValueError, json.JSONDecodeError):
        return None
    if result.source_las_sha256 != source_sha or result.config_digest != config_digest:
        return None
    if (
        result.output_las_path != str(output_las)
        or sha256_file(output_las) != result.output_las_sha256
    ):
        return None
    return result


def _load_fast_vegetation_cache(
    result_path: Path, output_npz: Path, config_digest: str
) -> FastVegetationInputResult | None:
    if not result_path.is_file() or not output_npz.is_file():
        return None
    try:
        result = FastVegetationInputResult.from_dict(json.loads(result_path.read_text("utf-8")))
        if result.config_digest != config_digest or result.output_npz_path != str(output_npz):
            return None
        if sha256_file(output_npz) != result.output_npz_sha256:
            return None
        with np.load(output_npz, allow_pickle=False) as stream:
            validate_fast_v4_arrays(
                np.asarray(stream["original_index"]),
                np.asarray(stream["xyz_local_m"]),
                np.asarray(stream["hag_m"]),
            )
    except (FastResearchError, OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return result


def _normalize_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if bounds is None:
        return None
    minimum = tuple(float(value) for value in bounds[0])
    maximum = tuple(float(value) for value in bounds[1])
    if (
        len(minimum) != 3
        or len(maximum) != 3
        or not all(math.isfinite(value) for value in (*minimum, *maximum))
    ):
        raise FastResearchError("FAST_V4_INPUT_INVALID_BOUNDS")
    if any(left > right for left, right in zip(minimum, maximum, strict=True)):
        raise FastResearchError("FAST_V4_INPUT_INVALID_BOUNDS")
    return minimum, maximum


def _bounds_payload(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
) -> dict[str, list[float]] | None:
    if bounds is None:
        return None
    return {"minimum": list(bounds[0]), "maximum": list(bounds[1])}


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _is_sha(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _v4_summary(result: Any) -> dict[str, Any]:
    return {
        "tree_count": len(result.tree_records),
        "point_count": int(len(result.original_index)),
        "assigned_point_count": int(np.count_nonzero(result.tree_id_by_point >= 0)),
        "result_digest": result.result_digest,
    }
