"""FGC-PMF-RLP-v1 formal producer: ingress, spool, tiling, PMF, seed, plane, routing.

Authority: docs/full-resolution-ground-classification-design.md (DESIGN-FROZEN)
and docs/full-resolution-ground-classification-implementation-plan.md
(PLAN-FROZEN).  This module implements Bundle B of the frozen plan.  It is
pure-deterministic, CPU-first, NumPy/laspy only, and never opens the raw LAS for
mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from core.contracts.frv import FrvConfiguration, frv_digest
from core.contracts.frv_hag import FrvHagProviderConfig
from core.contracts.frv_source import (
    FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
    FrvSourceCanonicalContentPolicy,
    FrvSourceCoordinateConversionPolicy,
    FrvSourceInvalidPointPolicy,
    FrvSourceModel,
    FrvSourceProviderConfig,
    FrvSourceProviderReference,
    FrvSourceProviderResult,
    derive_frv_source_provider_config_digest,
)
from core.contracts.ground_classification import (
    FGC_CLASSIFIED_DIR,
    FGC_COMPLETION_MANIFEST_PATH,
    FGC_COVERAGE_FLAG_CONFIRMED,
    FGC_COVERAGE_FLAG_OBSERVED,
    FGC_DECISION_MANIFEST_PATH,
    FGC_DECISIONS_DIR,
    FGC_PUBLICATION_PROOF_PATH,
    FGC_SPOOL_RECORD_FORMAT,
    FGC_SPOOL_RECORD_SIZE,
    FgcCompletionManifest,
    FgcDecisionChunkRecord,
    FgcDecisionManifest,
    FgcErrorCode,
    FgcPublicationProof,
    FgcSpoolManifest,
    FgcSpoolTileSummary,
    GroundClassificationPolicyConfig,
    GroundDecision,
    fgc_decision_digest,
    fgc_free_space_requirement_bytes,
    fgc_tile_layout_digest,
)

from algorithms.ground.decision_store import (
    CoverageRecordAccumulator,
    DecisionStore,
    compute_coverage_result,
)
from algorithms.io import load_frv_source
from algorithms.io.classified_las import (
    build_classified_source_identity,
    verify_classified_las,
    write_classified_las,
)

# ---------------------------------------------------------------------------
# Error / cancellation / progress contracts
# ---------------------------------------------------------------------------


class FgcExecutionError(Exception):
    """Source-level fail-closed FGC failure with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...


class ProgressReporter(Protocol):
    def update(self, progress: Any) -> None: ...


def _require_not_cancelled(cancellation: CancellationToken | None) -> None:
    if cancellation is not None and cancellation.is_cancelled():
        raise FgcExecutionError("SESSION_CANCELLED", "FGC session cancelled")


# ---------------------------------------------------------------------------
# Exact rational coordinate authority (design section 4)
# ---------------------------------------------------------------------------


def _parse_decimal_rational(text: str) -> Fraction:
    """Parse ``format(value, '.17g')`` output as an exact base-10 rational."""
    body = text.strip()
    exponent = 0
    if "e" in body or "E" in body:
        mantissa, exponent_text = body.split("e", 1) if "e" in body else body.split("E", 1)
        exponent = int(exponent_text)
    else:
        mantissa = body
    if "." in mantissa:
        integer_part, fractional_part = mantissa.split(".")
        digits = integer_part + fractional_part
        denominator = 10 ** len(fractional_part)
        numerator = int(digits) if digits and any(c != "0" for c in digits) else 0
    else:
        numerator = int(mantissa)
        denominator = 1
    if exponent >= 0:
        numerator *= 10**exponent
    else:
        denominator *= 10 ** (-exponent)
    return Fraction(numerator, denominator)


def _gcd_fraction(a: Fraction, b: Fraction) -> Fraction:
    """Greatest common rational divisor of two positive rationals."""
    return Fraction(
        math.gcd(a.numerator * b.denominator, b.numerator * a.denominator),
        a.denominator * b.denominator,
    )


@dataclass(frozen=True)
class MetricGrid:
    """Exact raw-integer tick authority and radii for one source header."""

    scale_x: Fraction
    scale_y: Fraction
    scale_z: float
    offset_x: float
    offset_y: float
    offset_z: float
    g: Fraction
    kx: int  # scale_x / g
    ky: int  # scale_y / g
    tx: int  # tile side raw ticks (x)
    ty: int
    fx: int  # fine cell raw ticks (x)
    fy: int
    cx: int  # coverage cell raw ticks (x)
    cy: int
    hx: int  # halo raw ticks (x)
    hy: int
    x0: int
    y0: int
    x_max: int
    y_max: int
    nx: int
    ny: int
    _radius_g_sq: int  # (3.0/g)^2 for the plane/seed radius
    _vertical_g_sq: int  # (1.0/g)^2

    @property
    def plane_radius_sq(self) -> int:
        return self._radius_g_sq

    @property
    def seed_radius_sq(self) -> int:
        return self._radius_g_sq

    @property
    def vertical_radius_sq(self) -> int:
        return self._vertical_g_sq

    def metric_ticks_dx(self, xq: int, xp: int) -> int:
        return int(xq - xp) * self.kx

    def metric_ticks_dy(self, yq: int, yp: int) -> int:
        return int(yq - yp) * self.ky

    def squared_distance_g(self, xq: int, yq: int, xp: int, yp: int) -> int:
        dx = self.metric_ticks_dx(xq, xp)
        dy = self.metric_ticks_dy(yq, yp)
        return dx * dx + dy * dy

    def tile_index(self, x: int, y: int) -> tuple[int, int]:
        tx = min((int(x) - self.x0) // self.tx, self.nx - 1)
        ty = min((int(y) - self.y0) // self.ty, self.ny - 1)
        if tx < 0 or ty < 0:
            raise FgcExecutionError(
                FgcErrorCode.TILE_OWNERSHIP_FAILED,
                f"raw coordinate below the source-min anchor: x={x}, y={y}",
            )
        return (ty, tx)

    def fine_index(self, x: int, y: int) -> tuple[int, int]:
        ix = (int(x) - self.x0) // self.fx
        iy = (int(y) - self.y0) // self.fy
        return (iy, ix)

    def coverage_index(self, x: int, y: int) -> tuple[int, int]:
        ix = (int(x) - self.x0) // self.cx
        iy = (int(y) - self.y0) // self.cy
        return (iy, ix)


def _length_to_ticks(length_m: float, scale: Fraction, name: str) -> int:
    value = Fraction(length_m) / scale
    if value.denominator != 1 or value.numerator <= 0:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID,
            f"{name} {length_m} m is not a positive integer number of raw ticks",
        )
    return int(value.numerator)


def build_metric_grid(
    header: Any,
    policy: GroundClassificationPolicyConfig,
    *,
    x0: int,
    y0: int,
    x_max: int,
    y_max: int,
) -> MetricGrid:
    """Build the exact raw-integer grid from header scales and source minima."""
    scale_x = _parse_decimal_rational(format(float(header.scale_xyz[0]), ".17g"))
    scale_y = _parse_decimal_rational(format(float(header.scale_xyz[1]), ".17g"))
    if scale_x <= 0 or scale_y <= 0:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID, "header scales must be positive"
        )
    scale_z = float(header.scale_xyz[2])
    if not math.isfinite(scale_z) or scale_z <= 0:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID, "z scale must be positive"
        )
    g = _gcd_fraction(scale_x, scale_y)
    kx = scale_x / g
    ky = scale_y / g
    if kx.denominator != 1 or ky.denominator != 1 or kx.numerator <= 0 or ky.numerator <= 0:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID, "scale/g must be positive integers"
        )
    one_over_g = Fraction(1, 1) / g
    if one_over_g.denominator != 1 or one_over_g.numerator <= 0:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID, "1.0 m / g must be a positive integer"
        )
    if one_over_g.numerator**2 > (1 << 63) - 1:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID, "g squared radius does not fit int64"
        )

    # Commensurability: every frozen policy length must be an integer tick count.
    for length, label in (
        (policy.fine_cell_m, "fine cell"),
        (policy.coverage_cell_m, "coverage cell"),
        (policy.tile_side_m, "tile side"),
        (policy.halo_m, "halo"),
        (policy.plane_radius_m, "plane radius"),
        (policy.vertical_radius_m, "vertical radius"),
    ):
        _length_to_ticks(length, scale_x, f"{label} (x)")
        _length_to_ticks(length, scale_y, f"{label} (y)")

    tx = _length_to_ticks(policy.tile_side_m, scale_x, "tile side")
    ty = _length_to_ticks(policy.tile_side_m, scale_y, "tile side")
    fx = _length_to_ticks(policy.fine_cell_m, scale_x, "fine cell")
    fy = _length_to_ticks(policy.fine_cell_m, scale_y, "fine cell")
    cx = _length_to_ticks(policy.coverage_cell_m, scale_x, "coverage cell")
    cy = _length_to_ticks(policy.coverage_cell_m, scale_y, "coverage cell")
    hx = _length_to_ticks(policy.halo_m, scale_x, "halo")
    hy = _length_to_ticks(policy.halo_m, scale_y, "halo")

    nx = (x_max - x0) // tx + 1
    ny = (y_max - y0) // ty + 1
    if nx <= 0 or ny <= 0:
        raise FgcExecutionError(FgcErrorCode.UNSUPPORTED_COORDINATE_GRID, "empty tile domain")

    radius_g = Fraction(policy.plane_radius_m) / g
    vertical_g = Fraction(policy.vertical_radius_m) / g
    if radius_g.denominator != 1 or vertical_g.denominator != 1:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID,
            "plane/vertical radius must be an integer number of g ticks",
        )

    return MetricGrid(
        scale_x=scale_x,
        scale_y=scale_y,
        scale_z=scale_z,
        offset_x=float(header.offset_xyz[0]),
        offset_y=float(header.offset_xyz[1]),
        offset_z=float(header.offset_xyz[2]),
        g=g,
        kx=int(kx.numerator),
        ky=int(ky.numerator),
        tx=tx,
        ty=ty,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        hx=hx,
        hy=hy,
        x0=x0,
        y0=y0,
        x_max=x_max,
        y_max=y_max,
        nx=nx,
        ny=ny,
        _radius_g_sq=int(radius_g.numerator) ** 2,
        _vertical_g_sq=int(vertical_g.numerator) ** 2,
    )


def _reconstruct_raw_minimum(decoded_min: float, offset: float, scale: Fraction) -> int:
    """Reconstruct the raw integer minimum from a decoded P-SOURCE proof value."""
    delta = _parse_decimal_rational(format(float(decoded_min), ".17g")) - _parse_decimal_rational(
        format(float(offset), ".17g")
    )
    raw = delta / scale
    if raw.denominator != 1:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID,
            "source minimum does not map to an exact raw integer",
        )
    return int(raw.numerator)


def _reconstruct_raw_maximum(decoded_max: float, offset: float, scale: Fraction) -> int:
    """Recover the exact raw integer maximum from a decoded proof value.

    The design (section 4) mandates exact ".17g" reconstruction only for the
    source minima.  The domain maximum is raw-integer authority, but a decoded
    float64 maximum can carry a few ULPs of rounding relative to the exact
    ``raw * scale + offset`` product, so exact rational reconstruction is not
    guaranteed for real scenes.  The raw maximum is an integer LAS coordinate;
    rounding the float quotient recovers it exactly because the fractional
    error is far below 0.5.  This never weakens the raw-integer authority.
    """
    delta = _parse_decimal_rational(format(float(decoded_max), ".17g")) - _parse_decimal_rational(
        format(float(offset), ".17g")
    )
    quotient = delta / scale
    if quotient.denominator == 1:
        return int(quotient.numerator)
    return int(round(quotient))


def decode_z(raw_z: int, grid: MetricGrid) -> float:
    return float(raw_z) * grid.scale_z + grid.offset_z


def _reconstruct_raw_coordinate(decoded: float, offset: float, scale: float) -> int:
    value = (float(decoded) - offset) / scale
    raw = int(round(value))
    if abs((raw * scale + offset) - float(decoded)) > max(1e-6, 1e-6 * abs(float(decoded))):
        raise FgcExecutionError(
            FgcErrorCode.NONFINITE_SOURCE,
            f"decoded coordinate does not round-trip to an exact raw integer: {decoded!r}",
        )
    return raw


# ---------------------------------------------------------------------------
# Single-pass FGCSP1 owner spool (design section 6)
# ---------------------------------------------------------------------------

_SPOOL_COL_INDEX = 0
_SPOOL_COL_X = 1
_SPOOL_COL_Y = 2
_SPOOL_COL_Z = 3


def _provider_config_from_dimensions(
    pass_through_dimensions: tuple[Any, ...],
) -> FrvSourceProviderConfig:
    return FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=pass_through_dimensions,
        provider_config_digest=derive_frv_source_provider_config_digest(
            schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
            source_model=FrvSourceModel.SINGLE_FILE_ONLY,
            coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
            invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
            canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
            pass_through_dimensions=pass_through_dimensions,
        ),
    )


def _provider_config_from_proof(proof: Any) -> FrvSourceProviderConfig:
    return _provider_config_from_dimensions(proof.pass_through_dimensions)


def _classified_replay_provider_config(proof: Any) -> FrvSourceProviderConfig:
    """P-SOURCE config for the classified replay, always passing classification."""
    from core.contracts.frv_source import FrvSourcePassThroughDimension

    dimensions = list(proof.pass_through_dimensions)
    if FrvSourcePassThroughDimension.CLASSIFICATION not in dimensions:
        dimensions.append(FrvSourcePassThroughDimension.CLASSIFICATION)
    return _provider_config_from_dimensions(tuple(dimensions))


def spool_source(
    source_reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    frv_configuration: FrvConfiguration,
    policy: GroundClassificationPolicyConfig,
    *,
    grid: MetricGrid,
    scratch_dir: Path | str,
    progress: ProgressReporter | None = None,
    cancellation: CancellationToken | None = None,
) -> FgcSpoolManifest:
    """Perform the one sequential raw-LAS owner-spool pass (FGCSP1)."""
    scratch = Path(scratch_dir)
    spool_dir = scratch / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)

    identity = source_result.source_identity
    n = identity.source_point_count
    proof = source_result.proof
    provider_config = _provider_config_from_proof(proof)

    handles: dict[tuple[int, int], Any] = {}
    tile_counts: dict[tuple[int, int], int] = {}
    hashers: dict[tuple[int, int], Any] = {}
    observed_min_x: int | None = None
    observed_min_y: int | None = None
    total_written = 0

    def _tile_handle(tile_id: tuple[int, int]):
        if tile_id not in handles:
            path = spool_dir / f"tile_{tile_id[0]:06d}_{tile_id[1]:06d}.spool"
            handle = path.open("wb")
            handles[tile_id] = handle
            tile_counts[tile_id] = 0
            hashers[tile_id] = hashlib.sha256()
        return handles[tile_id]

    session = load_frv_source(
        source_reference,
        frv_configuration=frv_configuration,
        provider_config=provider_config,
    )
    try:
        for chunk in session.iter_chunks():
            _require_not_cancelled(cancellation)
            xyz = np.asarray(chunk.xyz, dtype=np.float64)
            original_index = np.asarray(chunk.original_index, dtype=np.int64)
            count = len(xyz)
            raw_x = np.empty(count, dtype=np.int64)
            raw_y = np.empty(count, dtype=np.int64)
            raw_z = np.empty(count, dtype=np.int64)
            for row in range(count):
                raw_x[row] = _reconstruct_raw_coordinate(
                    float(xyz[row, 0]), grid.offset_x, float(grid.scale_x)
                )
                raw_y[row] = _reconstruct_raw_coordinate(
                    float(xyz[row, 1]), grid.offset_y, float(grid.scale_y)
                )
                raw_z[row] = _reconstruct_raw_coordinate(
                    float(xyz[row, 2]), grid.offset_z, grid.scale_z
                )
            chunk_min_x = int(raw_x.min())
            chunk_min_y = int(raw_y.min())
            observed_min_x = (
                chunk_min_x if observed_min_x is None else min(observed_min_x, chunk_min_x)
            )
            observed_min_y = (
                chunk_min_y if observed_min_y is None else min(observed_min_y, chunk_min_y)
            )
            for row in range(count):
                tile_id = grid.tile_index(int(raw_x[row]), int(raw_y[row]))
                handle = _tile_handle(tile_id)
                record = struct.pack(
                    FGC_SPOOL_RECORD_FORMAT,
                    int(original_index[row]),
                    int(raw_x[row]),
                    int(raw_y[row]),
                    int(raw_z[row]),
                )
                handle.write(record)
                hashers[tile_id].update(record)
                tile_counts[tile_id] += 1
                total_written += 1
                if total_written > policy.max_source_points:
                    raise FgcExecutionError(
                        FgcErrorCode.RESOURCE_LIMIT_EXCEEDED,
                        "source point count exceeds max_source_points",
                    )
            if progress is not None:
                progress.update(
                    _make_progress(
                        "fgc:spool",
                        "spool",
                        processed=total_written,
                        total=n,
                        message="owner spool pass",
                    )
                )
        session.finalize()
    finally:
        for handle in handles.values():
            try:
                handle.flush()
                handle.close()
            except OSError:
                pass
        try:
            session.close()
        except Exception:
            pass

    if total_written != n:
        raise FgcExecutionError(
            FgcErrorCode.SPATIAL_SPOOL_CORRUPT,
            f"spool wrote {total_written} rows but the source domain is {n}",
        )
    if observed_min_x != grid.x0 or observed_min_y != grid.y0:
        raise FgcExecutionError(
            FgcErrorCode.SOURCE_GRID_ANCHOR_MISMATCH,
            "actual raw minimum differs from the reconstructed source anchor",
        )

    tile_summaries: list[FgcSpoolTileSummary] = []
    for tile_id in sorted(tile_counts):
        tile_summaries.append(
            FgcSpoolTileSummary(
                tile_id_ty=tile_id[0],
                tile_id_tx=tile_id[1],
                row_count=tile_counts[tile_id],
                sha256=hashers[tile_id].hexdigest(),
            )
        )
    tile_layout_digest = fgc_tile_layout_digest([tile.tile_id for tile in tile_summaries], n)
    return FgcSpoolManifest(
        source_identity_record_digest=identity.source_identity_record_digest,
        source_point_count=n,
        tile_layout_digest=tile_layout_digest,
        tiles=tuple(tile_summaries),
    )


def _make_progress(task_id: str, stage: str, *, processed: int, total: int, message: str) -> Any:
    from core.contracts.progress import PipelineProgress

    return PipelineProgress(
        task_id=task_id,
        stage=stage,
        status="running",
        processed_points=processed,
        total_points=total,
        percent=100.0 if total == 0 else processed / total * 100.0,
        message=message,
    )


# ---------------------------------------------------------------------------
# Fine-grid construction and sparse PMF (design sections 5, 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileWorkDomain:
    """The fine-cell work domain for one tile with the halo rectangle in ticks."""

    core_x_lo: int
    core_x_hi: int
    core_y_lo: int
    core_y_hi: int
    halo_x_lo: int
    halo_x_hi: int
    halo_y_lo: int
    halo_y_hi: int
    work_ix_lo: int
    work_ix_hi: int
    work_iy_lo: int
    work_iy_hi: int


def tile_work_domain(grid: MetricGrid, tile_id: tuple[int, int]) -> TileWorkDomain:
    ty, tx = tile_id
    core_x_lo = grid.x0 + tx * grid.tx
    core_y_lo = grid.y0 + ty * grid.ty
    core_x_hi = grid.x_max if tx == grid.nx - 1 else grid.x0 + (tx + 1) * grid.tx - 1
    core_y_hi = grid.y_max if ty == grid.ny - 1 else grid.y0 + (ty + 1) * grid.ty - 1
    if core_x_hi < core_x_lo or core_y_hi < core_y_lo:
        raise FgcExecutionError(FgcErrorCode.TILE_OWNERSHIP_FAILED, "empty tile core")
    halo_x_lo = max(core_x_lo - grid.hx, grid.x0)
    halo_x_hi = min(core_x_hi + grid.hx, grid.x_max)
    halo_y_lo = max(core_y_lo - grid.hy, grid.y0)
    halo_y_hi = min(core_y_hi + grid.hy, grid.y_max)

    # Only fine cells whose complete raw-tick cell rectangle is inside the halo
    # rectangle are eligible PMF work cells (design section 5).
    def _ix_lo(lo: int) -> int:
        ix = (lo - grid.x0) // grid.fx
        while grid.x0 + ix * grid.fx < lo:
            ix += 1
        return ix

    def _ix_hi(hi: int) -> int:
        ix = (hi - grid.x0) // grid.fx
        while grid.x0 + (ix + 1) * grid.fx - 1 > hi:
            ix -= 1
        return ix

    def _iy_lo(lo: int) -> int:
        iy = (lo - grid.y0) // grid.fy
        while grid.y0 + iy * grid.fy < lo:
            iy += 1
        return iy

    def _iy_hi(hi: int) -> int:
        iy = (hi - grid.y0) // grid.fy
        while grid.y0 + (iy + 1) * grid.fy - 1 > hi:
            iy -= 1
        return iy

    return TileWorkDomain(
        core_x_lo=core_x_lo,
        core_x_hi=core_x_hi,
        core_y_lo=core_y_lo,
        core_y_hi=core_y_hi,
        halo_x_lo=halo_x_lo,
        halo_x_hi=halo_x_hi,
        halo_y_lo=halo_y_lo,
        halo_y_hi=halo_y_hi,
        work_ix_lo=_ix_lo(halo_x_lo),
        work_ix_hi=_ix_hi(halo_x_hi),
        work_iy_lo=_iy_lo(halo_y_lo),
        work_iy_hi=_iy_hi(halo_y_hi),
    )


@dataclass(frozen=True)
class FineGrid:
    """Exact sparse minimum fine grid G0 over the tile work domain."""

    iy_lo: int
    iy_hi: int
    ix_lo: int
    ix_hi: int
    g0: np.ndarray  # float64, NaN where nodata
    rep_raw_z: np.ndarray  # int32, lexicographic minimum representative raw Z
    rep_index: np.ndarray  # int64, representative original_index
    rep_raw_x: np.ndarray  # int32
    rep_raw_y: np.ndarray  # int32
    valid: np.ndarray  # bool

    @property
    def height(self) -> int:
        return int(self.g0.shape[0])

    @property
    def width(self) -> int:
        return int(self.g0.shape[1])


def build_fine_grid(
    rows: np.ndarray,
    *,
    grid: MetricGrid,
    domain: TileWorkDomain,
    policy: GroundClassificationPolicyConfig,
) -> FineGrid:
    """Build G0: lexicographic minimum (raw Z, original_index) per fine cell.

    ``rows`` is the (halo) source row table with columns
    original_index int64, raw_x int64, raw_y int64, raw_z int64, sorted by
    original_index.
    """
    iy_lo = domain.work_iy_lo
    iy_hi = domain.work_iy_hi
    ix_lo = domain.work_ix_lo
    ix_hi = domain.work_ix_hi
    if iy_hi < iy_lo or ix_hi < ix_lo:
        raise FgcExecutionError(FgcErrorCode.TILE_OWNERSHIP_FAILED, "empty fine work domain")
    height = iy_hi - iy_lo + 1
    width = ix_hi - ix_lo + 1
    if height * width > policy.max_halo_grid_cells:
        raise FgcExecutionError(
            FgcErrorCode.RESOURCE_LIMIT_EXCEEDED,
            f"tile fine grid {height * width} cells exceeds max_halo_grid_cells",
        )
    g0 = np.full((height, width), np.nan, dtype=np.float64)
    rep_raw_z = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)
    rep_index = np.full((height, width), np.iinfo(np.int64).max, dtype=np.int64)
    rep_raw_x = np.zeros((height, width), dtype=np.int32)
    rep_raw_y = np.zeros((height, width), dtype=np.int32)
    valid = np.zeros((height, width), dtype=np.bool_)
    if len(rows):
        ix_all = (rows[:, _SPOOL_COL_X] - grid.x0) // grid.fx
        iy_all = (rows[:, _SPOOL_COL_Y] - grid.y0) // grid.fy
        mask = (ix_all >= ix_lo) & (ix_all <= ix_hi) & (iy_all >= iy_lo) & (iy_all <= iy_hi)
        if np.any(mask):
            sel_ix = (ix_all[mask] - ix_lo).astype(np.int64)
            sel_iy = (iy_all[mask] - iy_lo).astype(np.int64)
            sel_index = rows[mask, _SPOOL_COL_INDEX].astype(np.int64)
            sel_raw_z = rows[mask, _SPOOL_COL_Z].astype(np.int64)
            sel_raw_x = rows[mask, _SPOOL_COL_X].astype(np.int64)
            sel_raw_y = rows[mask, _SPOOL_COL_Y].astype(np.int64)
            cell_key = sel_iy * width + sel_ix
            # lexicographic minimum (raw Z, original_index) per cell
            order = np.lexsort((sel_index, sel_raw_z, cell_key))
            cell_key_sorted = cell_key[order]
            first_mask = np.empty(len(order), dtype=np.bool_)
            first_mask[0] = True
            first_mask[1:] = cell_key_sorted[1:] != cell_key_sorted[:-1]
            unique_ix = sel_ix[order][first_mask]
            unique_iy = sel_iy[order][first_mask]
            unique_z = sel_raw_z[order][first_mask]
            unique_index = sel_index[order][first_mask]
            unique_x = sel_raw_x[order][first_mask]
            unique_y = sel_raw_y[order][first_mask]
            g0[unique_iy, unique_ix] = unique_z.astype(np.float64) * grid.scale_z + grid.offset_z
            rep_raw_z[unique_iy, unique_ix] = unique_z.astype(np.int32)
            rep_index[unique_iy, unique_ix] = unique_index
            rep_raw_x[unique_iy, unique_ix] = unique_x.astype(np.int32)
            rep_raw_y[unique_iy, unique_ix] = unique_y.astype(np.int32)
            valid[unique_iy, unique_ix] = True
    return FineGrid(
        iy_lo=iy_lo,
        iy_hi=iy_hi,
        ix_lo=ix_lo,
        ix_hi=ix_hi,
        g0=g0,
        rep_raw_z=rep_raw_z,
        rep_index=rep_index,
        rep_raw_x=rep_raw_x,
        rep_raw_y=rep_raw_y,
        valid=valid,
    )


def _pmf_footprint_half_width(window_m: float) -> int:
    """n(w) = ceil(w/0.5), forced odd; h = (n-1)/2 (design section 5)."""
    k = int(math.ceil(window_m / 0.5))
    n = k if k % 2 == 1 else k + 1
    return (n - 1) // 2


def _sliding_extreme(
    source: np.ndarray,
    valid: np.ndarray,
    *,
    h: int,
    m: int,
    extreme: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized windowed min or max ignoring nodata, with clipped boundary.

    Returns (extreme_grid, valid_grid) where valid_grid marks cells with at
    least ``m`` valid source cells in the clipped footprint.
    """
    height, width = source.shape
    window = 2 * h + 1
    # padded values: invalid cells do not influence the extreme.
    fill = math.inf if extreme == "min" else -math.inf
    padded = np.full((height + 2 * h, width + 2 * h), fill, dtype=np.float64)
    padded[h : h + height, h : h + width] = np.where(valid, source, fill)
    count_padded = np.zeros((height + 2 * h, width + 2 * h), dtype=np.int64)
    count_padded[h : h + height, h : h + width] = valid.astype(np.int64)
    # sliding window via stride tricks over the padded array
    from numpy.lib.stride_tricks import sliding_window_view

    windowed = sliding_window_view(padded, (window, window))  # (H, W, w, w)
    counts = sliding_window_view(count_padded, (window, window)).sum(axis=(2, 3))
    if extreme == "min":
        result = windowed.min(axis=(2, 3))
    else:
        result = windowed.max(axis=(2, 3))
    result = result.copy()
    result[np.isinf(result)] = np.nan
    out_valid = counts >= m
    result[~out_valid] = np.nan
    return result, out_valid


def pmf_opening(
    source: np.ndarray,
    *,
    valid: np.ndarray,
    h: int,
    m: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Erosion or dilation opening over the G0 grid (design section 7)."""
    return _sliding_extreme(source, valid, h=h, m=m, extreme=mode)


def compute_pmf_mask(
    fine: FineGrid,
    *,
    policy: GroundClassificationPolicyConfig,
) -> np.ndarray:
    """Progressive mask M_5 over the work domain (design section 7)."""
    g0 = fine.g0
    valid = fine.valid
    m = int(policy.pmf_minimum_valid_cells)
    mask = valid.copy()
    for window_m in policy.pmf_windows_m:
        h = _pmf_footprint_half_width(window_m)
        e, e_valid = pmf_opening(g0, valid=valid, h=h, m=m, mode="min")
        o, o_valid = pmf_opening(e, valid=e_valid, h=h, m=m, mode="max")
        tau = float(policy.pmf_tolerance_base_m) + min(
            float(policy.pmf_tolerance_slope) * float(window_m),
            float(policy.pmf_tolerance_cap_m),
        )
        passes = mask & o_valid & (g0 - o <= tau)
        mask = passes
    return mask


# ---------------------------------------------------------------------------
# D3 / D3_plus connected support and lower envelope (design section 7, FGC-RR-001)
# ---------------------------------------------------------------------------


def _d3_offsets(
    grid: MetricGrid, policy: GroundClassificationPolicyConfig
) -> tuple[tuple[int, int], ...]:
    """Fine-cell offsets in D3 (centre excluded) by exact integer cell-centre distance.

    D3 excludes (0,0); D3_plus = D3 union {(0,0)}.  Cell-centre spacing in g
    ticks is fine_cell_m / g on each axis; the radius is seed_radius_m / g.
    """
    cell_g = Fraction(policy.fine_cell_m) / grid.g
    radius_g = Fraction(policy.seed_radius_m) / grid.g
    if cell_g.denominator != 1 or radius_g.denominator != 1:
        raise FgcExecutionError(
            FgcErrorCode.UNSUPPORTED_COORDINATE_GRID,
            "fine cell / seed radius must be integer g ticks",
        )
    cell_g_int = int(cell_g.numerator)
    radius_g_int = int(radius_g.numerator)
    radius_sq = radius_g_int**2
    offsets: list[tuple[int, int]] = []
    max_off = (radius_g_int // cell_g_int) + 1
    for dy in range(-max_off, max_off + 1):
        for dx in range(-max_off, max_off + 1):
            if dx == 0 and dy == 0:
                continue
            if (dx * cell_g_int) ** 2 + (dy * cell_g_int) ** 2 <= radius_sq:
                offsets.append((dy, dx))
    return tuple(offsets)


def _eight_connected_component_containing(
    valid_cells: set[tuple[int, int]],
    center: tuple[int, int],
) -> set[tuple[int, int]]:
    """8-connected component containing *center* within *valid_cells*."""
    if center not in valid_cells:
        return set()
    component = {center}
    stack = [center]
    while stack:
        iy, ix = stack.pop()
        for dyy in (-1, 0, 1):
            for dxx in (-1, 0, 1):
                if dyy == 0 and dxx == 0:
                    continue
                neighbor = (iy + dyy, ix + dxx)
                if neighbor in valid_cells and neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)
    return component


def seed_mask(
    fine: FineGrid,
    *,
    grid: MetricGrid,
    policy: GroundClassificationPolicyConfig,
    pmf_mask: np.ndarray,
) -> np.ndarray:
    """Seed cells: pass all PMF scales, connected support, not isolated-low."""
    offsets = _d3_offsets(grid, policy)
    m = int(policy.seed_connected_support_cells)
    envelope = int(policy.seed_lower_envelope_cells)
    isolated_low = float(policy.seed_isolated_low_m)
    seeds = np.zeros_like(pmf_mask)
    height, width = pmf_mask.shape
    iy_lo, iy_hi = fine.iy_lo, fine.iy_hi
    ix_lo, ix_hi = fine.ix_lo, fine.ix_hi
    valid_global = fine.valid
    g0_global = fine.g0
    for ly in range(height):
        gy = iy_lo + ly
        for lx in range(width):
            gx = ix_lo + lx
            if not pmf_mask[ly, lx]:
                continue
            # Valid G0 cells in c + D3_plus.
            candidates: set[tuple[int, int]] = set()
            for dy, dx in offsets:
                qy = gy + dy
                qx = gx + dx
                if (
                    iy_lo <= qy <= iy_hi
                    and ix_lo <= qx <= ix_hi
                    and valid_global[qy - iy_lo, qx - ix_lo]
                ):
                    candidates.add((qy, qx))
            candidates.add((gy, gx))
            component = _eight_connected_component_containing(candidates, (gy, gx))
            support = component - {(gy, gx)}
            if len(support) < m:
                continue
            # Five lowest keys (G0(q), qy, qx); lower-envelope median = 3rd.
            ordered = sorted(
                support,
                key=lambda q: (
                    float(g0_global[q[0] - iy_lo, q[1] - ix_lo]),
                    q[0],
                    q[1],
                ),
            )
            if len(ordered) < envelope:
                continue
            median = ordered[envelope // 2]  # 3rd of the five lowest
            median_elevation = float(g0_global[median[0] - iy_lo, median[1] - ix_lo])
            if median_elevation - float(g0_global[ly, lx]) > isolated_low:
                continue
            seeds[ly, lx] = True
    return seeds


# ---------------------------------------------------------------------------
# Local terrain plane, clipping, competing support (design section 8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaneFitResult:
    alpha: float
    beta: float
    intercept: float
    residual: float
    slope: float
    rms: float
    support_count: int
    available: bool
    final_x: tuple[float, ...] = ()
    final_y: tuple[float, ...] = ()
    final_z: tuple[float, ...] = ()


def _fit_ols(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, xp: float, yp: float, r: float
) -> tuple[float, float, float] | None:
    """Return (alpha, beta, intercept) or None when the fit is singular/nonfinite."""
    if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(zs).all()):
        return None
    u = (xs - xp) / r
    v = (ys - yp) / r
    h = np.column_stack((u, v, np.ones_like(u)))
    q = h.T @ h
    eigenvalues = np.linalg.eigvalsh(q)
    lambda_max = float(eigenvalues[2])
    lambda_min = float(eigenvalues[0])
    if not (np.isfinite(lambda_max) and np.isfinite(lambda_min)):
        return None
    if not (lambda_max > 0 and lambda_min >= 1e-10 * lambda_max):
        return None
    try:
        theta = np.linalg.solve(q, h.T @ zs)  # design: solve Q theta = H^T z
    except Exception:
        return None
    if not np.isfinite(theta).all():
        return None
    return (float(theta[0]), float(theta[1]), float(theta[2]))


def fit_plane(
    support_x: np.ndarray,
    support_y: np.ndarray,
    support_z: np.ndarray,
    *,
    xp: float,
    yp: float,
    pz: float,
    policy: GroundClassificationPolicyConfig,
) -> PlaneFitResult:
    """Two synchronous clips then final fit; residual = pz - intercept."""
    r = float(policy.plane_radius_m)
    clip_tol = float(policy.clip_tolerance_m)
    clips = int(policy.clip_rounds)
    xs = support_x.astype(np.float64)
    ys = support_y.astype(np.float64)
    zs = support_z.astype(np.float64)
    for _ in range(clips):
        if len(xs) < policy.plane_min_supports:
            return PlaneFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
        fitted = _fit_ols(xs, ys, zs, xp, yp, r)
        if fitted is None:
            return PlaneFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
        alpha, beta, intercept = fitted
        residuals = zs - (alpha * (xs - xp) / r + beta * (ys - yp) / r + intercept)
        keep = np.abs(residuals) <= clip_tol
        xs = xs[keep]
        ys = ys[keep]
        zs = zs[keep]
    if len(xs) < policy.plane_min_supports:
        return PlaneFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
    final = _fit_ols(xs, ys, zs, xp, yp, r)
    if final is None:
        return PlaneFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
    alpha, beta, intercept = final
    a = alpha / r
    b = beta / r
    slope = math.sqrt(a * a + b * b)
    residuals = zs - (alpha * (xs - xp) / r + beta * (ys - yp) / r + intercept)
    rms = math.sqrt(float(np.mean(residuals**2)))
    return PlaneFitResult(
        alpha=alpha,
        beta=beta,
        intercept=intercept,
        residual=float(pz) - intercept,
        slope=slope,
        rms=rms,
        support_count=int(len(xs)),
        available=True,
        final_x=tuple(float(value) for value in xs),
        final_y=tuple(float(value) for value in ys),
        final_z=tuple(float(value) for value in zs),
    )


def competing_support(
    support_x: np.ndarray,
    support_y: np.ndarray,
    support_z: np.ndarray,
    *,
    xp: float,
    yp: float,
    policy: GroundClassificationPolicyConfig,
) -> bool:
    """True when either X or Y partition is valid and competing (design section 8)."""
    r = float(policy.plane_radius_m)
    half_min = int(policy.competing_half_min_supports)
    disagreement = float(policy.competing_disagreement_m)
    xs = support_x.astype(np.float64)
    ys = support_y.astype(np.float64)
    zs = support_z.astype(np.float64)

    def _partition_difference(values: np.ndarray) -> bool:
        negative = values < 0.0
        nonnegative = ~negative
        if (
            int(np.count_nonzero(negative)) < half_min
            or int(np.count_nonzero(nonnegative)) < half_min
        ):
            return False
        neg_fit = _fit_ols(xs[negative], ys[negative], zs[negative], xp, yp, r)
        pos_fit = _fit_ols(xs[nonnegative], ys[nonnegative], zs[nonnegative], xp, yp, r)
        if neg_fit is None or pos_fit is None:
            return False
        return abs(neg_fit[2] - pos_fit[2]) >= disagreement

    if _partition_difference(xs - xp):
        return True
    if _partition_difference(ys - yp):
        return True
    return False


# ---------------------------------------------------------------------------
# Vertical structure (design section 8)
# ---------------------------------------------------------------------------


class _VerticalHash:
    """Tile-local fine-cell vertical hash for vectorized exact checks.

    Halo rows are sorted by linear fine-cell index so every 7 x 7 window maps
    to one contiguous row slice; ``cell_start`` is a prefix array over the cell
    domain.  ``radius_raw_x`` / ``radius_raw_y`` are the exact raw-tick
    bounding-box radii ``floor(radius_g / kx)`` used only to pre-filter rows
    that cannot be within the 1 m circle, never changing the result.
    """

    __slots__ = (
        "idx",
        "x",
        "y",
        "z",
        "cell_start",
        "cell_count",
        "cell_z_min",
        "cell_z_max",
        "cell_min_x",
        "cell_min_y",
        "cell_max_x",
        "cell_max_y",
        "width_cells",
        "height_cells",
        "radius_raw_x",
        "radius_raw_y",
    )

    def __init__(
        self,
        *,
        idx: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cell_start: np.ndarray,
        cell_count: np.ndarray,
        cell_z_min: np.ndarray,
        cell_z_max: np.ndarray,
        cell_min_x: int,
        cell_min_y: int,
        cell_max_x: int,
        cell_max_y: int,
        width_cells: int,
        height_cells: int,
        radius_raw_x: int,
        radius_raw_y: int,
    ) -> None:
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z
        self.cell_start = cell_start
        self.cell_count = cell_count
        self.cell_z_min = cell_z_min
        self.cell_z_max = cell_z_max
        self.cell_min_x = cell_min_x
        self.cell_min_y = cell_min_y
        self.cell_max_x = cell_max_x
        self.cell_max_y = cell_max_y
        self.width_cells = width_cells
        self.height_cells = height_cells
        self.radius_raw_x = radius_raw_x
        self.radius_raw_y = radius_raw_y


def build_vertical_hash(
    halo_rows: np.ndarray,
    *,
    grid: MetricGrid,
) -> _VerticalHash:
    """Build the vectorized tile-local vertical hash (design section 8).

    The returned object holds the halo rows sorted by linear fine-cell index,
    a ``cell_start`` prefix array, and the exact raw-tick bounding radii.  It
    is created once per loaded halo and discarded after the tile; no KDTree or
    global neighbour structure is used.
    """
    radius_g = math.isqrt(grid.vertical_radius_sq)
    if len(halo_rows) == 0:
        return _VerticalHash(
            idx=np.empty(0, dtype=np.int64),
            x=np.empty(0, dtype=np.int64),
            y=np.empty(0, dtype=np.int64),
            z=np.empty(0, dtype=np.float64),
            cell_start=np.zeros(1, dtype=np.int64),
            cell_count=np.zeros(1, dtype=np.int64),
            cell_z_min=np.empty(1, dtype=np.float64),
            cell_z_max=np.empty(1, dtype=np.float64),
            cell_min_x=0,
            cell_min_y=0,
            cell_max_x=-1,
            cell_max_y=-1,
            width_cells=1,
            height_cells=1,
            radius_raw_x=radius_g // grid.kx,
            radius_raw_y=radius_g // grid.ky,
        )
    rx = halo_rows[:, _SPOOL_COL_X].astype(np.int64)
    ry = halo_rows[:, _SPOOL_COL_Y].astype(np.int64)
    cx = (rx - grid.x0) // grid.fx
    cy = (ry - grid.y0) // grid.fy
    cell_min_x = int(cx.min())
    cell_min_y = int(cy.min())
    cell_max_x = int(cx.max())
    cell_max_y = int(cy.max())
    width_cells = cell_max_x - cell_min_x + 1
    height_cells = cell_max_y - cell_min_y + 1
    linear = (cy - cell_min_y) * width_cells + (cx - cell_min_x)
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    total_cells = width_cells * height_cells
    counts = np.bincount(sorted_linear, minlength=total_cells)
    cell_start = np.zeros(total_cells + 1, dtype=np.int64)
    np.cumsum(counts, out=cell_start[1:])
    z_sorted = (
        halo_rows[order, _SPOOL_COL_Z].astype(np.float64) * float(grid.scale_z)
        + grid.offset_z
    )
    cell_z_min = np.full(total_cells, np.inf, dtype=np.float64)
    cell_z_max = np.full(total_cells, -np.inf, dtype=np.float64)
    nonempty = counts > 0
    if np.any(nonempty):
        starts = cell_start[:-1][nonempty]
        cell_z_min[nonempty] = np.minimum.reduceat(z_sorted, starts)
        cell_z_max[nonempty] = np.maximum.reduceat(z_sorted, starts)
    return _VerticalHash(
        idx=halo_rows[order, _SPOOL_COL_INDEX].astype(np.int64),
        x=halo_rows[order, _SPOOL_COL_X].astype(np.int64),
        y=halo_rows[order, _SPOOL_COL_Y].astype(np.int64),
        z=z_sorted,
        cell_start=cell_start,
        cell_count=counts,
        cell_z_min=cell_z_min,
        cell_z_max=cell_z_max,
        cell_min_x=cell_min_x,
        cell_min_y=cell_min_y,
        cell_max_x=cell_max_x,
        cell_max_y=cell_max_y,
        width_cells=width_cells,
        height_cells=height_cells,
        radius_raw_x=radius_g // grid.kx,
        radius_raw_y=radius_g // grid.ky,
    )


def vertical_structure(
    point_index: int,
    point_x: int,
    point_y: int,
    point_z: float,
    *,
    vertical_hash: _VerticalHash,
    grid: MetricGrid,
    policy: GroundClassificationPolicyConfig,
) -> bool:
    """Tile-local vertical rule: >= 8 other rows within 1 m, span >= 1.5 m.

    The predicate is monotone in the eligible-neighbour set: adding another row
    within 1 m can only raise the count and only widen the z span, so once
    ``count >= min_other`` and ``span >= threshold`` hold the result is final.
    The vectorized computation evaluates the full 7 x 7 window slice exactly
    (same candidate population, same self exclusion, same exact integer
    ``<=`` predicate, same span over {point} union retained rows) as the
    per-neighbour reference scan.

    The exact circle predicate ``dxg**2 + dyg**2 <= radius_sq`` is evaluated in
    the overflow-free form ``dyg**2 <= radius_sq - dxg**2`` after a raw-tick
    bounding-box pre-filter that never drops a within-radius row.
    """
    vh = vertical_hash
    radius_sq = int(grid.vertical_radius_sq)
    min_other = int(policy.vertical_min_other_rows)
    span = float(policy.vertical_span_m)
    px = int(point_x)
    py = int(point_y)
    pcx = (px - grid.x0) // grid.fx
    pcy = (py - grid.y0) // grid.fy
    if (
        pcx + 3 < vh.cell_min_x
        or pcx - 3 > vh.cell_max_x
        or pcy + 3 < vh.cell_min_y
        or pcy - 3 > vh.cell_max_y
    ):
        return False
    cx_lo = max(vh.cell_min_x, pcx - 3)
    cx_hi = min(vh.cell_max_x, pcx + 3)
    cy_lo = max(vh.cell_min_y, pcy - 3)
    cy_hi = min(vh.cell_max_y, pcy + 3)
    lin_lo = (cy_lo - vh.cell_min_y) * vh.width_cells + (cx_lo - vh.cell_min_x)
    lin_hi = (cy_hi - vh.cell_min_y) * vh.width_cells + (cx_hi - vh.cell_min_x)
    # Exact fast reject: the per-cell global min/max over the window cells
    # over-include out-of-radius rows, so if even that overestimated span is
    # below the threshold the true within-1 m span is below it too -> a
    # definite miss (the reference also returns False because span < threshold).
    over_zmin = float(vh.cell_z_min[lin_lo : lin_hi + 1].min())
    over_zmax = float(vh.cell_z_max[lin_lo : lin_hi + 1].max())
    if max(over_zmax, point_z) - min(over_zmin, point_z) < span:
        return False

    lo = vh.cell_start[lin_lo]
    hi = vh.cell_start[lin_hi + 1]
    if hi <= lo:
        return False

    count = 0
    z_lo = math.inf
    z_hi = -math.inf
    has_nonfinite = not math.isfinite(point_z)

    def _accumulate(n: int, zc: np.ndarray) -> None:
        nonlocal count, z_lo, z_hi, has_nonfinite
        if n <= 0:
            return
        count += n
        if not np.isfinite(zc).all():
            has_nonfinite = True
        zc_min = float(zc.min())
        zc_max = float(zc.max())
        if zc_min < z_lo:
            z_lo = zc_min
        if zc_max > z_hi:
            z_hi = zc_max

    # Own cell first: every row in the same 0.5 m fine cell is within the 1 m
    # radius (max intra-cell metric distance is 0.5 m * sqrt(2) < 1 m), so no
    # radius filter is needed there; only the self row is excluded.  The
    # monotone predicate lets us exit as soon as count and span are satisfied.
    own_in_domain = (
        vh.cell_min_x <= pcx <= vh.cell_max_x and vh.cell_min_y <= pcy <= vh.cell_max_y
    )
    if own_in_domain:
        own_lin = (pcy - vh.cell_min_y) * vh.width_cells + (pcx - vh.cell_min_x)
        own_count = int(vh.cell_count[own_lin])
        own_lo = vh.cell_start[own_lin]
        own_hi = vh.cell_start[own_lin + 1]
        if own_count >= min_other + 1:
            # O(1) own-cell early exit: every row in the same 0.5 m fine cell is
            # within the 1 m radius, and the self row's z equals point_z which is
            # already part of the span set, so the precomputed cell min/max give
            # the exact own-cell span.  The eligible count is at least min_other
            # whether or not the self row is present.
            zmin_c = float(vh.cell_z_min[own_lin])
            zmax_c = float(vh.cell_z_max[own_lin])
            if (max(zmax_c, point_z) - min(zmin_c, point_z)) >= span:
                return True
            count += own_count
            if zmin_c < z_lo:
                z_lo = zmin_c
            if zmax_c > z_hi:
                z_hi = zmax_c
        elif own_count > 0:
            o_idx = vh.idx[own_lo:own_hi]
            o_z = vh.z[own_lo:own_hi]
            self_mask = o_idx != int(point_index)
            _accumulate(int(self_mask.sum()), o_z[self_mask])
            if (
                count >= min_other
                and (max(z_hi, point_z) - min(z_lo, point_z)) >= span
            ):
                return True

    # Remaining window rows (before and after the own cell) with the exact
    # integer radius predicate in its overflow-safe form.
    def _scan(s: int, e: int) -> None:
        if e <= s:
            return
        x = vh.x[s:e]
        y = vh.y[s:e]
        z = vh.z[s:e]
        idx = vh.idx[s:e]
        dx = x - px
        dy = y - py
        keep = (np.abs(dx) <= vh.radius_raw_x) & (np.abs(dy) <= vh.radius_raw_y)
        if not keep.any():
            return
        xk = x[keep]
        yk = y[keep]
        zk = z[keep]
        ik = idx[keep]
        dxg = (xk - px) * grid.kx
        dyg = (yk - py) * grid.ky
        dxs = dxg * dxg
        dys = dyg * dyg
        within = dys <= (radius_sq - dxs)
        mask = within & (ik != int(point_index))
        _accumulate(int(mask.sum()), zk[mask])

    if own_in_domain:
        _scan(lo, own_lo)
        _scan(own_hi, hi)
    else:
        _scan(lo, hi)

    if count < min_other:
        return False
    if has_nonfinite:
        return False
    if point_z > z_hi:
        z_hi = point_z
    if point_z < z_lo:
        z_lo = point_z
    return (z_hi - z_lo) >= span


# ---------------------------------------------------------------------------
# Routing precedence (design section 9)
# ---------------------------------------------------------------------------


def route_point(
    *,
    vertical: bool,
    plane: PlaneFitResult,
    competing: bool,
    policy: GroundClassificationPolicyConfig,
) -> GroundDecision:
    """Apply the exact first-match routing precedence."""
    if vertical:
        return GroundDecision.REVIEW_GROUND
    if not plane.available:
        return GroundDecision.REVIEW_GROUND
    if competing:
        return GroundDecision.REVIEW_GROUND
    if plane.slope > policy.slope_max or plane.rms > policy.rms_max_m:
        return GroundDecision.REVIEW_GROUND
    if policy.residual_confirm_min_m <= plane.residual <= policy.residual_confirm_max_m:
        return GroundDecision.CONFIRMED_GROUND
    if plane.residual > policy.residual_non_ground_above_m:
        return GroundDecision.NON_GROUND
    return GroundDecision.REVIEW_GROUND


# ---------------------------------------------------------------------------
# Public classification entry point (Bundle B scope: decisions + coverage)
# ---------------------------------------------------------------------------


def _load_spool_table(
    spool_dir: Path, tile_id: tuple[int, int], *, expected_sha256: str | None = None
) -> np.ndarray | None:
    """Load one owner spool as int64 rows [index, x, y, z].

    When *expected_sha256* is provided and the spool exists, its digest is
    verified and a mismatch raises SPATIAL_SPOOL_CORRUPT.
    """
    path = spool_dir / f"tile_{tile_id[0]:06d}_{tile_id[1]:06d}.spool"
    if not path.exists():
        if expected_sha256 is not None:
            raise FgcExecutionError(
                FgcErrorCode.SPATIAL_SPOOL_CORRUPT,
                f"declared nonempty owner spool missing for tile {tile_id}",
            )
        return None
    data = path.read_bytes()
    if len(data) % FGC_SPOOL_RECORD_SIZE != 0:
        raise FgcExecutionError(FgcErrorCode.SPATIAL_SPOOL_CORRUPT, "spool record size mismatch")
    if expected_sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise FgcExecutionError(
                FgcErrorCode.SPATIAL_SPOOL_CORRUPT, f"spool SHA mismatch for tile {tile_id}"
            )
    count = len(data) // FGC_SPOOL_RECORD_SIZE
    dtype = np.dtype([("index", "<i8"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
    values = np.frombuffer(data, dtype=dtype)
    table = np.empty((count, 4), dtype=np.int64)
    table[:, 0] = values["index"]
    table[:, 1] = values["x"]
    table[:, 2] = values["y"]
    table[:, 3] = values["z"]
    if count > 1 and not (np.diff(table[:, 0]) > 0).all():
        raise FgcExecutionError(
            FgcErrorCode.SPATIAL_SPOOL_CORRUPT, "spool original_index not ascending"
        )
    if len(np.unique(table[:, 0])) != count:
        raise FgcExecutionError(
            FgcErrorCode.SPATIAL_SPOOL_CORRUPT, "spool contains duplicate original_index"
        )
    return table


def classify_ground(
    source_reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    frv_configuration: FrvConfiguration,
    p_hag_compatibility_config: FrvHagProviderConfig,
    output_root: Path | str,
    policy_config: GroundClassificationPolicyConfig,
    *,
    progress: ProgressReporter | None = None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Run FGC-PMF-RLP-v1 over a validated formal source reference."""
    if type(source_reference) is not FrvSourceProviderReference:
        raise TypeError("source_reference must be a FrvSourceProviderReference")
    if type(source_result) is not FrvSourceProviderResult:
        raise TypeError("source_result must be a FrvSourceProviderResult")
    if type(frv_configuration) is not FrvConfiguration:
        raise TypeError("frv_configuration must be a FrvConfiguration")
    if type(p_hag_compatibility_config) is not FrvHagProviderConfig:
        raise TypeError("p_hag_compatibility_config must be a FrvHagProviderConfig")
    if type(policy_config) is not GroundClassificationPolicyConfig:
        raise TypeError("policy_config must be a GroundClassificationPolicyConfig")

    identity = source_result.source_identity
    proof = source_result.proof
    n = identity.source_point_count
    _require_not_cancelled(cancellation)

    header = proof.header
    scale_x = _parse_decimal_rational(format(float(header.scale_xyz[0]), ".17g"))
    scale_y = _parse_decimal_rational(format(float(header.scale_xyz[1]), ".17g"))
    x0 = _reconstruct_raw_minimum(
        float(proof.source_bounds_min_xyz[0]), float(header.offset_xyz[0]), scale_x
    )
    y0 = _reconstruct_raw_minimum(
        float(proof.source_bounds_min_xyz[1]), float(header.offset_xyz[1]), scale_y
    )
    x_max = _reconstruct_raw_maximum(
        float(proof.source_bounds_max_xyz[0]), float(header.offset_xyz[0]), scale_x
    )
    y_max = _reconstruct_raw_maximum(
        float(proof.source_bounds_max_xyz[1]), float(header.offset_xyz[1]), scale_y
    )
    grid = build_metric_grid(header, policy_config, x0=x0, y0=y0, x_max=x_max, y_max=y_max)

    if n > policy_config.max_source_points:
        raise FgcExecutionError(
            FgcErrorCode.RESOURCE_LIMIT_EXCEEDED,
            f"source point count {n} exceeds max_source_points",
        )
    required_space = fgc_free_space_requirement_bytes(
        source_reference.runtime_access_path.stat().st_size, n, policy_config
    )

    output = Path(output_root)
    transaction = (
        output / "ground-classification" / identity.source_id / identity.source_file_sha256[:16]
    )
    scratch = transaction / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        result = _classify_and_publish(
            source_reference,
            source_result,
            frv_configuration,
            p_hag_compatibility_config,
            policy_config,
            grid=grid,
            transaction=transaction,
            scratch=scratch,
            required_space=required_space,
            progress=progress,
            cancellation=cancellation,
        )
        # Scratch (spool/coverage/routing) is never published authority.
        _remove_tree(scratch)
        return result
    except FgcExecutionError:
        _remove_tree(scratch)
        raise
    except Exception:
        _remove_tree(scratch)
        raise


def _classify_and_publish(
    source_reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    frv_configuration: FrvConfiguration,
    p_hag_compatibility_config: FrvHagProviderConfig,
    policy_config: GroundClassificationPolicyConfig,
    *,
    grid: MetricGrid,
    transaction: Path,
    scratch: Path,
    required_space: int,
    progress: ProgressReporter | None,
    cancellation: CancellationToken | None,
) -> dict[str, Any]:
    """Spool, classify, coverage-gate, and publish inside one transaction."""
    import shutil

    transaction_parent = transaction.parent
    transaction_parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(transaction_parent).free
    if available < required_space:
        raise FgcExecutionError(
            FgcErrorCode.SPATIAL_SPOOL_RESOURCE_FAILED,
            "same-volume free space is below the FGC preflight requirement",
        )

    n = source_result.source_identity.source_point_count
    spool_manifest = spool_source(
        source_reference,
        source_result,
        frv_configuration,
        policy_config,
        grid=grid,
        scratch_dir=scratch,
        progress=progress,
        cancellation=cancellation,
    )

    store = DecisionStore(n, scratch_dir=scratch, policy=policy_config)
    coverage = CoverageRecordAccumulator(
        run_dir=scratch / "coverage",
        run_bytes=int(policy_config.coverage_sort_run_bytes),
    )
    try:
        for tile_id in sorted(tile.tile_id for tile in spool_manifest.tiles):
            _require_not_cancelled(cancellation)
            _classify_tile(
                tile_id,
                scratch=scratch,
                store=store,
                coverage=coverage,
                grid=grid,
                policy=policy_config,
                spool_manifest=spool_manifest,
            )
        store.assert_no_unrouted()
        chunks = store.write_chunks(transaction / "decisions")
        store.stream_verify(transaction / "decisions", chunks)
    finally:
        store.cleanup()

    run_paths = coverage.finish()
    coverage_result = compute_coverage_result(
        run_paths,
        policy=policy_config,
        max_cells=int(policy_config.max_coverage_cells),
    )
    if not coverage_result.passes_occupancy(policy=policy_config):
        raise FgcExecutionError(
            FgcErrorCode.GROUND_COVERAGE_GAP,
            "confirmed ground occupancy is below the frozen publication threshold",
        )
    overwidth = coverage_result.interior_overwidth_components(policy=policy_config)
    if overwidth:
        raise FgcExecutionError(
            FgcErrorCode.GROUND_COVERAGE_GAP,
            f"interior unsupported component width exceeds {policy_config.coverage_hole_width_m} m",
        )

    # ---- Bundle C: classified LAS, lineage, proof, atomic completion root. ----
    result = publish_ground_classification(
        source_reference,
        source_result,
        frv_configuration,
        p_hag_compatibility_config,
        policy_config,
        grid=grid,
        transaction=transaction,
        chunks=chunks,
        coverage_result=coverage_result,
        spool_manifest=spool_manifest,
        cancellation=cancellation,
    )
    result["grid"] = grid
    return result


def _build_classified_reference(
    *,
    classified_path: Path,
    classified_identity: Any,
    provider_config: FrvSourceProviderConfig,
    frv_configuration: FrvConfiguration,
) -> FrvSourceProviderReference:
    """Build a formal P-SOURCE reference over the classified derivative."""
    from core.contracts.frv_a2 import (
        FrvA2ArtifactComponentBinding,
        FrvA2ArtifactComponentKind,
        FrvA2ProviderRole,
    )
    from core.contracts.frv_source import (
        FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
        FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    )

    sha = classified_identity.source_file_sha256
    size = int(classified_path.stat().st_size)
    component = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=sha,
        size_bytes=size,
    )
    artifact_id = (
        f"frv-a2-ref-v1|{FrvA2ProviderRole.SOURCE_SCAN.value}|source_scan|"
        f"source_file,source_file,{sha},{size}"
    )
    return FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        artifact_reference_id=artifact_id,
        artifact_components=(component,),
        runtime_access_path=classified_path,
        source_identity=classified_identity,
        canonical_point_content_spec=None,
        configuration_digest=frv_configuration.configuration_digest,
        provider_config_digest=provider_config.provider_config_digest,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )


def _replay_classified_source(
    classified_reference: FrvSourceProviderReference,
    frv_configuration: FrvConfiguration,
    provider_config: FrvSourceProviderConfig,
) -> Any:
    """Replay the classified derivative through unchanged P-SOURCE."""
    session = load_frv_source(
        classified_reference,
        frv_configuration=frv_configuration,
        provider_config=provider_config,
    )
    try:
        for _ in session.iter_chunks():
            pass
        return session.finalize()
    finally:
        try:
            session.close()
        except Exception:
            pass


def _replay_hag_compatibility(
    classified_reference: FrvSourceProviderReference,
    classified_result: FrvSourceProviderResult,
    frv_configuration: FrvConfiguration,
    p_hag_compatibility_config: FrvHagProviderConfig,
) -> dict[str, Any]:
    """Run the P-HAG compatibility replay (recorded in the proof only)."""
    from algorithms.vegetation.hag_provider import load_frv_hag

    session = load_frv_hag(
        classified_reference,
        classified_result,
        frv_configuration=frv_configuration,
        provider_config=p_hag_compatibility_config,
    )
    try:
        for _ in session.iter_chunks():
            pass
        hag_result = session.finalize()
    finally:
        try:
            session.close()
        except Exception:
            pass
    proof = hag_result.proof
    return {
        "config_digest": p_hag_compatibility_config.provider_config_digest,
        "config_schema_version": p_hag_compatibility_config.schema_version,
        "ground_point_count": int(proof.ground_point_count),
        "supported_point_count": int(proof.supported_point_count),
        "ground_cell_inventory_digest": proof.ground_cell_inventory_digest,
        "hag_valid_positive_count": int(proof.hag_valid_positive_count),
        "hag_missing_count": int(proof.hag_missing_count),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def publish_ground_classification(
    source_reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    frv_configuration: FrvConfiguration,
    p_hag_compatibility_config: FrvHagProviderConfig,
    policy: GroundClassificationPolicyConfig,
    *,
    grid: MetricGrid,
    transaction: Path,
    chunks: tuple[FgcDecisionChunkRecord, ...],
    coverage_result: Any,
    spool_manifest: FgcSpoolManifest,
    cancellation: CancellationToken | None,
) -> dict[str, Any]:
    """Create the classified LAS, lineage, proof, and completion root last.

    Immutable chunks and the classified LAS are published first, then the
    proof and decision manifest, and the completion manifest is atomically
    renamed last (design section 12).  A collision in the target namespace
    fails closed with PUBLICATION_COLLISION.
    """
    identity = source_result.source_identity
    n = identity.source_point_count
    raw_path = source_reference.runtime_access_path
    source_stem = raw_path.stem
    classified_dir = transaction / FGC_CLASSIFIED_DIR
    classified_path = classified_dir / f"{source_stem}.fgc-v1.las"

    # Collision rejection: a valid existing publication is never overwritten.
    existing = [
        transaction / FGC_COMPLETION_MANIFEST_PATH,
        transaction / FGC_DECISION_MANIFEST_PATH,
        transaction / FGC_PUBLICATION_PROOF_PATH,
        classified_path,
    ]
    collisions = [path for path in existing if path.exists()]
    if collisions:
        raise FgcExecutionError(
            FgcErrorCode.PUBLICATION_COLLISION,
            "FGC publication target already contains published components",
        )

    decisions_dir = transaction / FGC_DECISIONS_DIR
    writer_result = write_classified_las(
        raw_path,
        decisions_dir,
        classified_path,
        source_point_count=n,
    )
    verify_classified_las(
        raw_path,
        classified_path,
        decisions_dir,
        source_point_count=n,
    )

    decision_digest = fgc_decision_digest(
        source_identity=identity,
        policy_digest=policy.policy_digest,
        source_point_count=n,
        chunks=chunks,
    )
    classified_identity = build_classified_source_identity(
        classified_path=classified_path,
        raw_identity=identity,
        decision_digest=decision_digest,
        point_count=n,
        bounds_min_xyz=identity.source_bounds_min_xyz,
        bounds_max_xyz=identity.source_bounds_max_xyz,
    )

    classified_replay_config = _classified_replay_provider_config(source_result.proof)
    classified_reference = _build_classified_reference(
        classified_path=classified_path,
        classified_identity=classified_identity,
        provider_config=classified_replay_config,
        frv_configuration=frv_configuration,
    )
    classified_result = _replay_classified_source(
        classified_reference, frv_configuration, classified_replay_config
    )

    p_source_replay = {
        "status": "valid",
        "classified_source_identity_record_digest": (
            classified_result.source_identity.source_identity_record_digest
        ),
        "classified_source_point_count": (
            int(classified_result.source_identity.source_point_count)
        ),
        "classified_source_file_sha256": (classified_result.source_identity.source_file_sha256),
    }
    p_hag_replay = _replay_hag_compatibility(
        classified_reference,
        classified_result,
        frv_configuration,
        p_hag_compatibility_config,
    )

    total_state_counts = {"non_ground": 0, "confirmed_ground": 0, "review_ground": 0}
    for chunk in chunks:
        total_state_counts["non_ground"] += chunk.state_counts["non_ground"]
        total_state_counts["confirmed_ground"] += chunk.state_counts["confirmed_ground"]
        total_state_counts["review_ground"] += chunk.state_counts["review_ground"]

    classification_mapping_counts = dict(writer_result["classification_mapping_counts"])

    decision_manifest = FgcDecisionManifest(
        source_identity=identity,
        raw_source_sha256=identity.source_file_sha256,
        source_point_count=n,
        policy_digest=policy.policy_digest,
        chunks=chunks,
        total_state_counts=total_state_counts,
        tile_summaries=spool_manifest.tiles,
        coverage=coverage_result,
        decision_digest=decision_digest,
    )
    decision_manifest.validated_with_policy(policy)
    decision_manifest_sha = frv_digest(decision_manifest.to_dict())
    decision_manifest_path = transaction / FGC_DECISION_MANIFEST_PATH
    _write_json_atomic(decision_manifest_path, decision_manifest.to_dict())

    proof = FgcPublicationProof(
        raw_source_identity=identity,
        raw_source_sha256=identity.source_file_sha256,
        source_point_count=n,
        policy_digest=policy.policy_digest,
        decision_digest=decision_digest,
        decision_manifest_digest=decision_manifest_sha,
        classification_mapping_counts=classification_mapping_counts,
        classified_source_identity=classified_identity,
        classified_source_sha256=classified_identity.source_file_sha256,
        p_source_replay=p_source_replay,
        p_hag_compatibility=p_hag_replay,
    )
    proof_path = transaction / FGC_PUBLICATION_PROOF_PATH
    _write_json_atomic(proof_path, proof.to_dict())

    completion = FgcCompletionManifest(
        policy_digest=policy.policy_digest,
        raw_source_identity=identity,
        classified_source_identity=classified_identity,
        decision_manifest_path=FGC_DECISION_MANIFEST_PATH,
        decision_manifest_sha256=_file_sha256(decision_manifest_path),
        decision_manifest_size_bytes=int(decision_manifest_path.stat().st_size),
        decision_manifest_digest=decision_manifest_sha,
        publication_proof_path=FGC_PUBLICATION_PROOF_PATH,
        publication_proof_sha256=_file_sha256(proof_path),
        publication_proof_size_bytes=int(proof_path.stat().st_size),
        publication_proof_digest=proof.proof_digest,
        classified_las_path=f"{FGC_CLASSIFIED_DIR}/{classified_path.name}",
        classified_las_sha256=classified_identity.source_file_sha256,
        classified_las_size_bytes=int(classified_path.stat().st_size),
        classified_las_point_count=n,
        decision_chunks=chunks,
    )
    # Publish the completion manifest last, atomically.
    completion_path = transaction / FGC_COMPLETION_MANIFEST_PATH
    _write_json_atomic(completion_path, completion.to_dict())

    return {
        "classified_las_path": classified_path,
        "classified_source_identity": classified_identity,
        "decision_manifest": decision_manifest,
        "decision_manifest_digest": decision_manifest_sha,
        "publication_proof": proof,
        "completion_manifest": completion,
        "p_hag_compatibility": p_hag_replay,
        "spool_manifest": spool_manifest,
        "chunks": chunks,
        "coverage": coverage_result,
    }


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _classify_tile(
    tile_id: tuple[int, int],
    *,
    scratch: Path,
    store: DecisionStore,
    coverage: CoverageRecordAccumulator,
    grid: MetricGrid,
    policy: GroundClassificationPolicyConfig,
    spool_manifest: FgcSpoolManifest | None = None,
) -> None:
    """Classify one core tile using only owner spools plus exact-halo rows."""
    domain = tile_work_domain(grid, tile_id)
    spool_dir = scratch / "spool"

    sha_map: dict[tuple[int, int], str] = {}
    if spool_manifest is not None:
        sha_map = {tile.tile_id: tile.sha256 for tile in spool_manifest.tiles}
    core = _load_spool_table(spool_dir, tile_id, expected_sha256=sha_map.get(tile_id))
    if core is None or len(core) == 0:
        return
    core_x = core[:, _SPOOL_COL_X]
    core_y = core[:, _SPOOL_COL_Y]
    core_owned = (
        (core_x >= domain.core_x_lo)
        & (core_x <= domain.core_x_hi)
        & (core_y >= domain.core_y_lo)
        & (core_y <= domain.core_y_hi)
    )
    core_rows = core[core_owned]
    if len(core_rows) > policy.max_core_tile_points:
        raise FgcExecutionError(
            FgcErrorCode.RESOURCE_LIMIT_EXCEEDED,
            f"tile {tile_id} core rows exceed max_core_tile_points",
        )

    halo_parts: list[np.ndarray] = [core_rows]
    # Adjacent owner spools in (dy, dx) order.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor_id = (tile_id[0] + dy, tile_id[1] + dx)
            neighbor_rows = _load_spool_table(
                spool_dir, neighbor_id, expected_sha256=sha_map.get(neighbor_id)
            )
            if neighbor_rows is None or len(neighbor_rows) == 0:
                continue
            nx = neighbor_rows[:, _SPOOL_COL_X]
            ny = neighbor_rows[:, _SPOOL_COL_Y]
            in_halo = (
                (nx >= domain.halo_x_lo)
                & (nx <= domain.halo_x_hi)
                & (ny >= domain.halo_y_lo)
                & (ny <= domain.halo_y_hi)
            )
            halo_parts.append(neighbor_rows[in_halo])

    halo_combined = (
        np.concatenate(halo_parts, axis=0) if halo_parts else np.empty((0, 4), dtype=np.int64)
    )
    if len(halo_combined) > policy.max_halo_tile_points:
        raise FgcExecutionError(
            FgcErrorCode.RESOURCE_LIMIT_EXCEEDED,
            f"tile {tile_id} halo rows exceed max_halo_tile_points",
        )
    if len(halo_combined):
        halo_combined = halo_combined[np.argsort(halo_combined[:, _SPOOL_COL_INDEX], kind="stable")]

    fine = build_fine_grid(halo_combined, grid=grid, domain=domain, policy=policy)
    pmf_mask = compute_pmf_mask(fine, policy=policy)
    seeds = seed_mask(fine, grid=grid, policy=policy, pmf_mask=pmf_mask)
    # Tile-local vertical hash: built once per loaded halo, discarded after the tile.
    vertical_hash = build_vertical_hash(halo_combined, grid=grid)

    max_supports = int(policy.plane_max_supports)
    min_supports = int(policy.plane_min_supports)
    # Search window: seeds whose representative can be within 3 m of the point.
    # The representative lies inside its cell; add one cell of margin.
    search_cells = int(math.ceil(policy.plane_radius_m / policy.fine_cell_m)) + 1

    for row in core_rows:
        original_index = int(row[_SPOOL_COL_INDEX])
        px = int(row[_SPOOL_COL_X])
        py = int(row[_SPOOL_COL_Y])
        pz = decode_z(int(row[_SPOOL_COL_Z]), grid)
        px_dec = float(px) * float(grid.scale_x) + grid.offset_x
        py_dec = float(py) * float(grid.scale_y) + grid.offset_y

        center_ix = (px - grid.x0) // grid.fx
        center_iy = (py - grid.y0) // grid.fy
        support_x: list[int] = []
        support_y: list[int] = []
        support_z: list[float] = []
        support_idx: list[int] = []
        iy_lo_local = max(0, (center_iy - search_cells) - fine.iy_lo)
        iy_hi_local = min(fine.height - 1, (center_iy + search_cells) - fine.iy_lo)
        ix_lo_local = max(0, (center_ix - search_cells) - fine.ix_lo)
        ix_hi_local = min(fine.width - 1, (center_ix + search_cells) - fine.ix_lo)
        for ly in range(iy_lo_local, iy_hi_local + 1):
            for lx in range(ix_lo_local, ix_hi_local + 1):
                if not seeds[ly, lx]:
                    continue
                sx = int(fine.rep_raw_x[ly, lx])
                sy = int(fine.rep_raw_y[ly, lx])
                if grid.squared_distance_g(sx, sy, px, py) <= grid.plane_radius_sq:
                    support_x.append(sx)
                    support_y.append(sy)
                    support_z.append(float(fine.g0[ly, lx]))
                    support_idx.append(int(fine.rep_index[ly, lx]))

        if support_x:
            order = sorted(
                range(len(support_x)),
                key=lambda i: (
                    grid.squared_distance_g(support_x[i], support_y[i], px, py),
                    support_idx[i],
                ),
            )[:max_supports]
            support_x = [support_x[i] for i in order]
            support_y = [support_y[i] for i in order]
            support_z = [support_z[i] for i in order]

        vertical = vertical_structure(
            original_index,
            px,
            py,
            pz,
            vertical_hash=vertical_hash,
            grid=grid,
            policy=policy,
        )

        if len(support_x) >= min_supports:
            # Support representatives are raw ticks; decode to metres so the
            # plane fit uses consistent units with xp/yp (design section 8).
            sx_arr = np.asarray(support_x, dtype=np.float64) * float(grid.scale_x) + grid.offset_x
            sy_arr = np.asarray(support_y, dtype=np.float64) * float(grid.scale_y) + grid.offset_y
            sz_arr = np.asarray(support_z, dtype=np.float64)
            plane = fit_plane(
                sx_arr,
                sy_arr,
                sz_arr,
                xp=px_dec,
                yp=py_dec,
                pz=pz,
                policy=policy,
            )
            if plane.available:
                competing = competing_support(
                    np.asarray(plane.final_x, dtype=np.float64),
                    np.asarray(plane.final_y, dtype=np.float64),
                    np.asarray(plane.final_z, dtype=np.float64),
                    xp=px_dec,
                    yp=py_dec,
                    policy=policy,
                )
            else:
                competing = False
        else:
            plane = PlaneFitResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
            competing = False

        decision = route_point(
            vertical=vertical,
            plane=plane,
            competing=competing,
            policy=policy,
        )
        store.route(original_index, decision)

        cix, ciy = grid.coverage_index(px, py)
        flags = FGC_COVERAGE_FLAG_OBSERVED
        if decision == GroundDecision.CONFIRMED_GROUND:
            flags |= FGC_COVERAGE_FLAG_CONFIRMED
        coverage.append(cix, ciy, flags)
