"""FRV-to-V4 input visualization viewer (FRV-VIS-A1, PLAN-FROZEN).

This module is an independent, read-only viewer that renders a published
READY FRV membership artifact together with its published V4 input handoff so
an Owner can judge whether the handoff is sufficient for formal V4 tree
segmentation.  It never executes V4 segmentation, never re-judges vegetation,
never calls the FRV or V4 builders, and never modifies any input artifact.

Data-flow contract (frozen plan ``docs/frv-to-v4-visualization-acceptance-plan.md``):

1. The FRV artifact root is loaded through ``load_frv_membership_result``
   (full artifact/schema/binding/decision/block re-validation).  The
   authoritative source identity comes only from the validated execution
   result; the viewer never constructs ``FrvSourceProviderReference``,
   ``FrvConfiguration`` or ``FrvSourceProviderConfig`` and never depends on a
   persisted ``output_reference``.
2. The V4 handoff root is read strictly: ``validated_view.json``,
   ``coverage_report.json``, ``handoff_manifest.json`` and the ``normal/`` /
   ``review/`` block files are parsed with the frozen contracts and
   ``decode_frv_block``.
3. Cross-artifact identity, digest, count, bounds, frame, units, origin,
   block-count and original-index-domain checks must all pass or the viewer
   fails closed without producing any output.
4. Excluded/background coordinates come only from the independent read-only
   LAS display reader; measured coordinates come from the V4 blocks.  Local
   coordinates are always ``float64(source_xyz) - float64(origin_context)``
   with the origin subtracted exactly once.
5. Formal counts and route conservation use the full validated arrays; only
   the Plotly display arrays are deterministically sampled under a fixed
   allocation priority and a fixed seed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts.frv import (
    FrvConfigurationPurpose,
    FrvHagEvidenceState,
    FrvMembershipState,
    FrvQualityFlag,
    FrvReasonFlag,
    FRVScopeCoverageManifest,
    FrvSegmentationHagReadiness,
    SourceIdentity,
)
from core.contracts.frv_membership import (
    FrvMembershipExecutionResult,
    FrvMembershipExecutionState,
)
from core.contracts.frv_wire import FrvWirePointRelationState
from core.contracts.v4_inputs import (
    V4PrimaryRoute,
    V4RecoveryCoverageReport,
    V4ValidatedFrvMembershipView,
)
from plotly import graph_objects as go

from algorithms.vegetation.frv_artifacts import (
    FRV_BLOCK_MAGIC,
    V4_NORMAL_BLOCK_MAGIC,
    V4_REVIEW_BLOCK_MAGIC,
    decode_frv_block,
    sha256_file,
)
from algorithms.vegetation.frv_membership import load_frv_membership_result

# ---------------------------------------------------------------------------
# Frozen display vocabulary and caps
# ---------------------------------------------------------------------------

TOOL_NAME = "frv-v4-input-visualization"
SUMMARY_SCHEMA_VERSION = "frv-vis-a1-summary-v1"
CHECKLIST_SCHEMA_VERSION = "frv-vis-a1-checklist-v1"

DEFAULT_TOTAL_CAP = 250_000
HARD_TOTAL_CAP = 500_000
DEFAULT_LAYER_CAP = 100_000
MAX_LAYER_CAP = 100_000
DEFAULT_BACKGROUND_CAP = 50_000
MAX_BACKGROUND_CAP = 100_000

# Display-only low-HAG threshold (meters, local).  Formal routing is never
# changed by this threshold; it only drives the diagnostic overlay.
LOW_HAG_DISPLAY_M = 0.30

_SOURCE_CHUNK_SIZE = 250_000

# Layer names (frozen in the plan's layer table).
LAYER_NORMAL = "normal_admitted"
LAYER_MEMBERSHIP_REVIEW = "membership_review"
LAYER_HAG_REVIEW = "hag_review"
LAYER_EXCLUDED = "excluded_audit"
LAYER_UNRECOVERED = "unrecovered_regions"
LAYER_BACKGROUND = "original_background"
LAYER_WIRE_OVERLAY = "wire_review_overlay"
LAYER_GROUND_OVERLAY = "ground_low_hag_overlay"

_MEASURED_LAYERS = (
    LAYER_NORMAL,
    LAYER_MEMBERSHIP_REVIEW,
    LAYER_HAG_REVIEW,
    LAYER_EXCLUDED,
)

_ALL_LAYERS = (
    LAYER_NORMAL,
    LAYER_MEMBERSHIP_REVIEW,
    LAYER_HAG_REVIEW,
    LAYER_EXCLUDED,
    LAYER_UNRECOVERED,
    LAYER_BACKGROUND,
    LAYER_WIRE_OVERLAY,
    LAYER_GROUND_OVERLAY,
)

# Fixed display budget allocation priority (highest first); unrecovered audit
# records sit above every point bucket but never consume the point cap.
ALLOCATION_PRIORITY = (
    "unrecovered_audit_records",
    LAYER_MEMBERSHIP_REVIEW,
    LAYER_HAG_REVIEW,
    "wire_or_quality_flagged_points",
    LAYER_EXCLUDED,
    LAYER_NORMAL,
    LAYER_BACKGROUND,
)

# Layer display properties (frozen for the first version).
_LAYER_DISPLAY: dict[str, dict[str, Any]] = {
    LAYER_NORMAL: {
        "display_name": "Normal admitted",
        "color": "#2ca02c",
        "marker_size": 3.0,
        "opacity": 0.9,
        "default_visible": True,
        "route": "NORMAL_SEGMENTATION",
    },
    LAYER_MEMBERSHIP_REVIEW: {
        "display_name": "Membership review",
        "color": "#ffd400",
        "marker_size": 3.0,
        "opacity": 0.9,
        "default_visible": True,
        "route": "MEASURED_POINT_REVIEW",
    },
    LAYER_HAG_REVIEW: {
        "display_name": "HAG review",
        "color": "#ff7f0e",
        "marker_size": 3.0,
        "opacity": 0.9,
        "default_visible": True,
        "route": "MEASURED_POINT_REVIEW",
    },
    LAYER_EXCLUDED: {
        "display_name": "Excluded audit",
        "color": "#8c8c8c",
        "marker_size": 2.0,
        "opacity": 0.7,
        "default_visible": False,
        "route": "EXCLUDED_AUDIT_ONLY",
    },
    LAYER_UNRECOVERED: {
        "display_name": "Unrecovered regions",
        "color": "#d62728",
        "marker_size": 3.0,
        "opacity": 0.9,
        "default_visible": True,
        "route": "UNRECOVERED_REGION_REVIEW",
    },
    LAYER_BACKGROUND: {
        "display_name": "Original background",
        "color": "#c8c8c8",
        "marker_size": 1.5,
        "opacity": 0.15,
        "default_visible": False,
        "route": "SOURCE_BACKGROUND",
    },
    LAYER_WIRE_OVERLAY: {
        "display_name": "Wire review overlay",
        "color": "#e6194b",
        "marker_size": 5.0,
        "opacity": 1.0,
        "default_visible": False,
        "route": "WIRE_REVIEW_OVERLAY",
    },
    LAYER_GROUND_OVERLAY: {
        "display_name": "Ground low-HAG overlay",
        "color": "#4363d8",
        "marker_size": 5.0,
        "opacity": 1.0,
        "default_visible": False,
        "route": "GROUND_LOW_HAG_OVERLAY",
    },
}

# V4 normal/review block field inventories (frozen in ``v4_inputs``).  The
# block decoder requires the exact schema_order/dtypes/row_shapes.
_V4_NORMAL_FIELDS = (
    ("original_index", "<i8"),
    ("xyz_local_m", "<f8"),
    ("hag_m", "<f4"),
    ("membership_state", "|u1"),
    ("hag_evidence_state", "|u1"),
    ("segmentation_hag_readiness", "|u1"),
    ("candidate_reason_flags", "<u4"),
)
_V4_REVIEW_FIELDS = (
    ("original_index", "<i8"),
    ("xyz_local_m", "<f8"),
    ("membership_state", "|u1"),
    ("hag_evidence_state", "|u1"),
    ("segmentation_hag_readiness", "|u1"),
    ("raw_hag_m", "<f8"),
    ("clamped_hag_m", "<f8"),
    ("reason_flags", "<u4"),
    ("quality_flags", "<u4"),
    ("wire_provider_state", "|u1"),
    ("wire_relation_state", "|u1"),
    ("nearest_wire_id", "<i4"),
    ("centerline_distance_m", "<f8"),
    ("primary_route", "|u1"),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FrvV4VisualizationError(RuntimeError):
    """Base failure for the FRV-VIS-A1 viewer; the CLI maps it to exit 2."""


class FrvV4BudgetError(FrvV4VisualizationError):
    """Invalid display-budget request; the run is rejected without output."""


class FrvV4ValidationError(FrvV4VisualizationError):
    """Source/artifact/identity/digest/coordinate validation failure."""


class FrvV4OutputError(FrvV4VisualizationError):
    """Output-path policy failure (for example, refusing to overwrite)."""


# ---------------------------------------------------------------------------
# Display budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DisplayBudget:
    """Validated display budget caps and the fixed sampling seed."""

    max_total_points: int
    max_points_per_layer: int
    max_background_points: int
    sampling_seed: int


def _coerce_cap(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if not (minimum <= integer <= maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return integer


def validate_display_budget(
    max_total_points: Any,
    max_points_per_layer: Any,
    max_background_points: Any,
    sampling_seed: Any = 0,
) -> DisplayBudget:
    """Validate the requested display caps; any invalid cap raises ValueError.

    Frozen ranges: ``max_total_points`` in ``1..500000``,
    ``max_points_per_layer`` and ``max_background_points`` in ``1..100000``.
    The seed must be a non-negative integer.
    """

    total = _coerce_cap(max_total_points, "max_total_points", 1, HARD_TOTAL_CAP)
    layer = _coerce_cap(max_points_per_layer, "max_points_per_layer", 1, MAX_LAYER_CAP)
    background = _coerce_cap(
        max_background_points, "max_background_points", 1, MAX_BACKGROUND_CAP
    )
    if isinstance(sampling_seed, (bool, np.bool_)):
        raise ValueError("sampling_seed must not be a boolean")
    if not isinstance(sampling_seed, (int, np.integer)):
        raise ValueError("sampling_seed must be an integer")
    seed = int(sampling_seed)
    if seed < 0:
        raise ValueError("sampling_seed must be non-negative")
    return DisplayBudget(total, layer, background, seed)


# ---------------------------------------------------------------------------
# Display data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvV4RegionRecord:
    """One unrecovered/blocked region audit record; never a fabricated point."""

    key: str
    scope_id: str | None
    coverage_state: str
    failure_reason_flags: int
    review_required: bool
    diagnostic_reference: str
    geometry_status: str
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None


@dataclass(slots=True)
class _LayerSource:
    """Full per-layer point data aligned by ascending ``original_index``."""

    name: str
    original_index: np.ndarray  # int64, sorted ascending, unique
    membership_state: np.ndarray  # uint8
    hag_evidence_state: np.ndarray  # uint8
    hag_readiness: np.ndarray  # uint8
    reason_flags: np.ndarray  # uint32
    quality_flags: np.ndarray  # uint32
    raw_hag: np.ndarray  # float64
    clamped_hag: np.ndarray  # float64
    wire_relation_state: np.ndarray  # uint8
    nearest_wire_id: np.ndarray  # int32
    centerline_distance_m: np.ndarray  # float64
    priority_flagged: np.ndarray  # bool
    xyz: np.ndarray | None  # float64 (n,3) local coords; None before source read


@dataclass(frozen=True, slots=True)
class LayerBudgetResult:
    """Per-layer display-budget accounting."""

    name: str
    full_count: int
    displayed_count: int
    dropped_display_count: int
    requested_cap: int | None
    applied_cap: int | None
    displayed_index: np.ndarray  # int64, sorted ascending


@dataclass(frozen=True, slots=True)
class DisplayAllocation:
    """Deterministic display-budget allocation for the full point layers."""

    layers: tuple[LayerBudgetResult, ...]
    background: LayerBudgetResult
    total_full_point_count: int
    total_displayed_point_count: int
    total_dropped_display_count: int
    requested_total_cap: int
    applied_total_cap: int
    requested_background_cap: int
    applied_background_cap: int
    sampling_seed: int


@dataclass(slots=True)
class _DisplayLayer:
    """One renderable Plotly trace."""

    name: str
    xyz: np.ndarray
    original_index: np.ndarray
    customdata: np.ndarray  # object (n, k) of pre-formatted hover strings
    default_visible: bool
    is_overlay: bool


@dataclass(frozen=True, slots=True)
class LayerDisplayCounts:
    """Full/displayed counts for every display layer (including overlays)."""

    full_count: int
    displayed_count: int


@dataclass(slots=True)
class FrvV4InputData:
    """Validated, mapped input data for one visualization run."""

    mode: str  # "ready" | "blocked"
    source_identity: SourceIdentity
    coverage_report: V4RecoveryCoverageReport | None
    layers: dict[str, _LayerSource]
    regions: tuple[FrvV4RegionRecord, ...]
    identity_checks: dict[str, bool]
    configuration_purpose: FrvConfigurationPurpose | None = None


# ---------------------------------------------------------------------------
# Enum / flag formatting helpers
# ---------------------------------------------------------------------------


def _enum_name(enum_cls: type, value: int) -> str:
    try:
        return enum_cls(int(value)).name
    except (TypeError, ValueError):
        return str(value)


def _format_flag_names(flags: int, flag_cls: type) -> str:
    if int(flags) == 0:
        return "NONE"
    return "+".join(
        member.name for member in flag_cls if int(flags) & int(member.value)
    )


def _fmt_float(value: float, digits: int = 2) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Deterministic sampling
# ---------------------------------------------------------------------------


def _seeded_uniform_subset(
    indices_sorted: np.ndarray, k: int, seed: int
) -> np.ndarray:
    """Return a deterministic uniform subset of ``k`` sorted indices.

    The selection is a function only of ``(indices_sorted, k, seed)``, so
    changing the published block row order never changes the displayed
    ``original_index`` set for identical inputs, seeds and caps.
    """

    n = int(indices_sorted.shape[0])
    if k <= 0 or n <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return indices_sorted
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n, size=k, replace=False))
    return indices_sorted[chosen]


def _background_display_indices(
    source_point_count: int,
    measured_sorted: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Return ``k`` deterministic original indices outside the measured union.

    The complement (non-measured) candidates are ``[0..source_point_count)``
    minus ``measured_sorted``.  Only the selected indices are materialised, so
    the memory cost is bounded by ``k`` even for very large sources.
    """

    measured = np.asarray(measured_sorted, dtype=np.int64)
    total = source_point_count - int(measured.shape[0])
    if k <= 0 or total <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= total:
        mask = np.ones(source_point_count, dtype=bool)
        mask[measured] = False
        return np.flatnonzero(mask).astype(np.int64)
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(total, size=k, replace=False))
    indices = positions.astype(np.int64, copy=True)
    for _ in range(64):
        updated = positions + np.searchsorted(measured, indices, side="right")
        if np.array_equal(updated, indices):
            break
        indices = updated
    if not np.array_equal(
        indices - np.searchsorted(measured, indices, side="right"), positions
    ):
        mask = np.ones(source_point_count, dtype=bool)
        mask[measured] = False
        complement = np.flatnonzero(mask).astype(np.int64)
        return complement[positions]
    return indices


def allocate_display_budget(
    layers: Mapping[str, _LayerSource],
    *,
    source_point_count: int,
    budget: DisplayBudget,
) -> DisplayAllocation:
    """Allocate the fixed display budget across the point layers.

    Allocation follows the frozen priority list; each bucket inherits the
    remaining global cap from the higher-priority buckets.  A wire/quality
    flagged point keeps its formal base layer and consumes one budget slot
    only.
    """

    layer_remaining = {name: budget.max_points_per_layer for name in _MEASURED_LAYERS}
    global_remaining = budget.max_total_points
    background_remaining = budget.max_background_points

    selected: dict[str, np.ndarray] = {
        name: np.empty(0, dtype=np.int64) for name in _MEASURED_LAYERS
    }
    first_global: dict[str, int] = {}

    def _mark_first(layer_name: str) -> None:
        if layer_name not in first_global and selected[layer_name].shape[0] > 0:
            first_global[layer_name] = global_remaining

    def _select_layer_bucket(layer_name: str) -> None:
        nonlocal global_remaining
        source = layers[layer_name]
        indices = np.asarray(source.original_index, dtype=np.int64)
        already = selected[layer_name]
        if already.shape[0]:
            available = indices[~np.isin(indices, already)]
        else:
            available = indices
        cap = min(layer_remaining[layer_name], global_remaining)
        if cap <= 0:
            take = np.empty(0, dtype=np.int64)
        elif available.shape[0] <= cap:
            take = available
        else:
            take = _seeded_uniform_subset(available, cap, budget.sampling_seed)
        if already.shape[0] or take.shape[0]:
            selected[layer_name] = np.sort(
                np.unique(np.concatenate([already, take]))
            )
        _mark_first(layer_name)
        layer_remaining[layer_name] -= int(take.shape[0])
        global_remaining -= int(take.shape[0])

    # Priority buckets: membership review, then HAG review.
    _select_layer_bucket(LAYER_MEMBERSHIP_REVIEW)
    _select_layer_bucket(LAYER_HAG_REVIEW)

    # Priority bucket: wire/quality flagged points, sorted by
    # (base layer, original_index); they stay in their formal base layer.
    base_order = {name: index for index, name in enumerate(_MEASURED_LAYERS)}
    flagged: list[tuple[int, str, int]] = []
    for layer_name in _MEASURED_LAYERS:
        source = layers[layer_name]
        mask = np.asarray(source.priority_flagged, dtype=bool)
        for index, flag in zip(
            np.asarray(source.original_index, dtype=np.int64),
            mask,
            strict=True,
        ):
            if flag:
                flagged.append((base_order[layer_name], layer_name, int(index)))
    flagged.sort()
    for _base_order, layer_name, index in flagged:
        if global_remaining <= 0:
            break
        if layer_remaining[layer_name] <= 0:
            continue
        current = selected[layer_name]
        if np.any(current == index):
            continue
        selected[layer_name] = np.sort(
            np.append(current, np.asarray([index], dtype=np.int64))
        )
        _mark_first(layer_name)
        layer_remaining[layer_name] -= 1
        global_remaining -= 1

    # Priority buckets: excluded, then normal.
    _select_layer_bucket(LAYER_EXCLUDED)
    _select_layer_bucket(LAYER_NORMAL)

    # Priority bucket: original background (independent background cap too).
    measured_union = np.sort(
        np.unique(
            np.concatenate(
                [
                    np.asarray(layers[name].original_index, dtype=np.int64)
                    for name in _MEASURED_LAYERS
                ]
            )
        )
    )
    background_full_count = source_point_count - int(measured_union.shape[0])
    background_cap = min(background_remaining, global_remaining)
    background_index = _background_display_indices(
        source_point_count, measured_union, background_cap, budget.sampling_seed
    )

    def _budget_result(
        name: str, displayed_index: np.ndarray, applied_cap: int | None
    ) -> LayerBudgetResult:
        source = layers[name]
        full_count = int(np.asarray(source.original_index).shape[0])
        displayed_count = int(displayed_index.shape[0])
        return LayerBudgetResult(
            name=name,
            full_count=full_count,
            displayed_count=displayed_count,
            dropped_display_count=full_count - displayed_count,
            requested_cap=budget.max_points_per_layer,
            applied_cap=applied_cap,
            displayed_index=np.asarray(displayed_index, dtype=np.int64),
        )

    layer_results = tuple(
        _budget_result(
            name,
            selected[name],
            min(budget.max_points_per_layer, first_global.get(name, 0)),
        )
        for name in _MEASURED_LAYERS
    )

    background_applied = min(budget.max_background_points, global_remaining)
    background_result = LayerBudgetResult(
        name=LAYER_BACKGROUND,
        full_count=background_full_count,
        displayed_count=int(background_index.shape[0]),
        dropped_display_count=background_full_count - int(background_index.shape[0]),
        requested_cap=budget.max_background_points,
        applied_cap=background_applied,
        displayed_index=background_index,
    )

    total_full = int(
        background_full_count + sum(result.full_count for result in layer_results)
    )
    total_displayed = int(
        background_result.displayed_count
        + sum(result.displayed_count for result in layer_results)
    )
    return DisplayAllocation(
        layers=layer_results,
        background=background_result,
        total_full_point_count=total_full,
        total_displayed_point_count=total_displayed,
        total_dropped_display_count=total_full - total_displayed,
        requested_total_cap=budget.max_total_points,
        applied_total_cap=budget.max_total_points,
        requested_background_cap=budget.max_background_points,
        applied_background_cap=background_applied,
        sampling_seed=budget.sampling_seed,
    )


# ---------------------------------------------------------------------------
# Independent read-only LAS display reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LasReadResult:
    """Extracted local XYZ plus the source-level validation summary."""

    xyz_local: np.ndarray
    extracted_index: np.ndarray
    header_point_count: int
    header_bounds_min: tuple[float, float, float]
    header_bounds_max: tuple[float, float, float]
    scales: tuple[float, float, float]
    file_sha256: str


def read_las_display(
    source: Path,
    *,
    needed: np.ndarray,
    origin: tuple[float, float, float],
    expected_count: int,
    expected_sha256: str,
    expected_bounds_min: tuple[float, float, float],
    expected_bounds_max: tuple[float, float, float],
) -> _LasReadResult:
    """Independently read the source LAS and extract selected local XYZ.

    The reader is read-only and never runs the formal P-SOURCE provider.  It
    streams the file in original order, requires the full sequence to be
    exactly ``0..source_point_count-1``, and keeps only the requested
    ``needed`` original indices.  Any count, SHA, bounds, order or identity
    mismatch fails closed.
    """

    if not isinstance(source, Path):
        raise FrvV4ValidationError("source must be a pathlib.Path")
    if source.suffix.lower() != ".las":
        raise FrvV4ValidationError(
            f"source must be a .las file; the formal entry point rejects .laz: {source.name}"
        )
    if not source.is_file():
        raise FrvV4ValidationError(f"source file does not exist: {source}")

    file_sha = sha256_file(source)
    if file_sha != expected_sha256:
        raise FrvV4ValidationError(
            "source file SHA-256 does not match the FRV source identity"
        )

    needed_arr = np.asarray(needed, dtype=np.int64)
    if needed_arr.ndim != 1:
        raise FrvV4ValidationError("needed indices must be one-dimensional")
    if needed_arr.size:
        if int(needed_arr.min()) < 0:
            raise FrvV4ValidationError("needed indices must be non-negative")
        if not (np.diff(needed_arr) > 0).all():
            raise FrvV4ValidationError("needed indices must be strictly increasing")

    with laspy.open(source, "r") as reader:
        header = reader.header
        header_count = int(header.point_count)
        scales = tuple(float(value) for value in header.scales)
        header_min = tuple(float(value) for value in header.mins)
        header_max = tuple(float(value) for value in header.maxs)
        if header_count != expected_count:
            raise FrvV4ValidationError(
                "source LAS header point count does not match the source identity"
            )
        if needed_arr.size and int(needed_arr.max()) >= header_count:
            raise FrvV4ValidationError("needed indices exceed the source point count")
        for axis in range(3):
            if abs(header_min[axis] - expected_bounds_min[axis]) > scales[axis]:
                raise FrvV4ValidationError(
                    "source LAS header min bounds do not match the source identity"
                )
            if abs(header_max[axis] - expected_bounds_max[axis]) > scales[axis]:
                raise FrvV4ValidationError(
                    "source LAS header max bounds do not match the source identity"
                )

        out_xyz: list[np.ndarray] = []
        out_index: list[np.ndarray] = []
        streamed_min = np.full(3, np.inf, dtype=np.float64)
        streamed_max = np.full(3, -np.inf, dtype=np.float64)
        start = 0
        for chunk in reader.chunk_iterator(_SOURCE_CHUNK_SIZE):
            point_count = int(chunk.array.shape[0])
            end = start + point_count
            xyz = np.column_stack(
                (
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                )
            )
            if not np.isfinite(xyz).all():
                raise FrvV4ValidationError(
                    "source LAS contains non-finite coordinates"
                )
            streamed_min = np.minimum(streamed_min, xyz.min(axis=0))
            streamed_max = np.maximum(streamed_max, xyz.max(axis=0))
            if needed_arr.size:
                low = int(np.searchsorted(needed_arr, start, side="left"))
                high = int(np.searchsorted(needed_arr, end, side="left"))
                if high > low:
                    rows = needed_arr[low:high] - start
                    if rows.min() < 0 or rows.max() >= point_count:
                        raise FrvV4ValidationError(
                            "source chunk original_index mapping is inconsistent"
                        )
                    out_xyz.append(xyz[rows])
                    out_index.append(needed_arr[low:high])
            start = end
        if start != header_count:
            raise FrvV4ValidationError(
                "source LAS streamed point count does not match the header"
            )
        streamed_min = tuple(float(value) for value in streamed_min)
        streamed_max = tuple(float(value) for value in streamed_max)

    for axis in range(3):
        if (
            streamed_min[axis] != expected_bounds_min[axis]
            or streamed_max[axis] != expected_bounds_max[axis]
        ):
            raise FrvV4ValidationError(
                "streamed source bounds do not match the source identity"
            )

    if out_index:
        extracted_index = np.concatenate(out_index).astype(np.int64)
        extracted_xyz = np.concatenate(out_xyz, axis=0).astype(np.float64)
        if not np.array_equal(extracted_index, needed_arr):
            raise FrvV4ValidationError(
                "source LAS did not contain every requested original index"
            )
    else:
        extracted_index = np.empty(0, dtype=np.int64)
        extracted_xyz = np.empty((0, 3), dtype=np.float64)

    origin_arr = np.asarray([float(value) for value in origin], dtype=np.float64)
    local = extracted_xyz - origin_arr
    local[local == 0.0] = 0.0  # normalize IEEE negative zero

    return _LasReadResult(
        xyz_local=local,
        extracted_index=extracted_index,
        header_point_count=header_count,
        header_bounds_min=header_min,
        header_bounds_max=header_max,
        scales=scales,
        file_sha256=file_sha,
    )


# ---------------------------------------------------------------------------
# Block decoding helpers
# ---------------------------------------------------------------------------


def _decode_block_files(
    paths: Sequence[Path],
    *,
    magic: bytes,
    schema_order: Sequence[str],
    dtypes: Mapping[str, np.dtype],
    row_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    arrays_list: list[dict[str, np.ndarray]] = []
    for path in paths:
        data = path.read_bytes()
        _header, arrays = decode_frv_block(
            data,
            magic=magic,
            schema_order=tuple(schema_order),
            dtypes=dtypes,
            row_shapes=row_shapes,
        )
        arrays_list.append(arrays)
    if not arrays_list:
        result: dict[str, np.ndarray] = {}
        for name in schema_order:
            result[name] = np.empty(0, dtype=dtypes[name])
        return result
    return {
        name: np.concatenate([arrays[name] for arrays in arrays_list])
        for name in schema_order
    }


def _normal_block_dtypes() -> dict[str, np.dtype]:
    return {name: np.dtype(code) for name, code in _V4_NORMAL_FIELDS}


def _normal_block_row_shapes() -> dict[str, tuple[int, ...]]:
    return {
        name: ((3,) if name == "xyz_local_m" else ())
        for name, _code in _V4_NORMAL_FIELDS
    }


def _review_block_dtypes() -> dict[str, np.dtype]:
    return {name: np.dtype(code) for name, code in _V4_REVIEW_FIELDS}


def _review_block_row_shapes() -> dict[str, tuple[int, ...]]:
    return {
        name: ((3,) if name == "xyz_local_m" else ())
        for name, _code in _V4_REVIEW_FIELDS
    }


# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------


def _align_sorted(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort all arrays by ``original_index`` while preserving alignment."""
    order = np.argsort(np.asarray(arrays["original_index"]), kind="stable")
    return {
        name: np.asarray(arrays[name])[order]
        for name in arrays
    }


def _wire_conflict(relation_state: int) -> bool:
    return int(relation_state) in (
        int(FrvWirePointRelationState.WIRE_BODY_POSSIBLE),
        int(FrvWirePointRelationState.WIRE_PROXIMITY_ONLY),
        int(FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT),
    )


def _priority_flagged(
    *,
    reason_flags: np.ndarray,
    quality_flags: np.ndarray,
    wire_relation_state: np.ndarray | None,
) -> np.ndarray:
    """Wire-conflict or any non-zero quality flag drives priority bucket 4."""
    flag = np.zeros(reason_flags.shape[0], dtype=bool)
    flag |= (np.asarray(reason_flags, dtype=np.uint32) & np.uint32(
        int(FrvReasonFlag.WIRE_CONFLICT)
    )) != 0
    flag |= np.asarray(quality_flags, dtype=np.uint32) != 0
    if wire_relation_state is not None:
        flag |= np.array(
            [_wire_conflict(int(value)) for value in wire_relation_state],
            dtype=bool,
        )
    return flag


def _empty_layer_source(name: str) -> _LayerSource:
    return _LayerSource(
        name=name,
        original_index=np.empty(0, dtype=np.int64),
        membership_state=np.empty(0, dtype=np.uint8),
        hag_evidence_state=np.empty(0, dtype=np.uint8),
        hag_readiness=np.empty(0, dtype=np.uint8),
        reason_flags=np.empty(0, dtype=np.uint32),
        quality_flags=np.empty(0, dtype=np.uint32),
        raw_hag=np.empty(0, dtype=np.float64),
        clamped_hag=np.empty(0, dtype=np.float64),
        wire_relation_state=np.empty(0, dtype=np.uint8),
        nearest_wire_id=np.empty(0, dtype=np.int32),
        centerline_distance_m=np.empty(0, dtype=np.float64),
        priority_flagged=np.empty(0, dtype=bool),
        xyz=np.empty((0, 3), dtype=np.float64),
    )


def _build_normal_layer(normal: Mapping[str, np.ndarray]) -> _LayerSource:
    aligned = _align_sorted(normal)
    count = int(aligned["original_index"].shape[0])
    if count == 0:
        return _empty_layer_source(LAYER_NORMAL)
    clamped = np.asarray(aligned["hag_m"], dtype=np.float64)
    flags = np.asarray(aligned["candidate_reason_flags"], dtype=np.uint32)
    return _LayerSource(
        name=LAYER_NORMAL,
        original_index=np.asarray(aligned["original_index"], dtype=np.int64),
        membership_state=np.asarray(aligned["membership_state"], dtype=np.uint8),
        hag_evidence_state=np.asarray(aligned["hag_evidence_state"], dtype=np.uint8),
        hag_readiness=np.asarray(
            aligned["segmentation_hag_readiness"], dtype=np.uint8
        ),
        reason_flags=flags,
        quality_flags=np.zeros(count, dtype=np.uint32),
        raw_hag=clamped,
        clamped_hag=clamped,
        wire_relation_state=np.zeros(count, dtype=np.uint8),
        nearest_wire_id=np.full(count, -1, dtype=np.int32),
        centerline_distance_m=np.zeros(count, dtype=np.float64),
        priority_flagged=_priority_flagged(
            reason_flags=flags,
            quality_flags=np.zeros(count, dtype=np.uint32),
            wire_relation_state=None,
        ),
        xyz=np.asarray(aligned["xyz_local_m"], dtype=np.float64),
    )


def _build_review_layers(review: Mapping[str, np.ndarray]) -> dict[str, _LayerSource]:
    aligned = _align_sorted(review)
    membership = np.asarray(aligned["membership_state"], dtype=np.uint8)
    admitted = (membership == int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE)) | (
        membership == int(FrvMembershipState.MANUAL_INCLUDE)
    )

    def _slice_layer(name: str, mask: np.ndarray) -> _LayerSource:
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            return _empty_layer_source(name)
        reason = np.asarray(aligned["reason_flags"], dtype=np.uint32)[positions]
        quality = np.asarray(aligned["quality_flags"], dtype=np.uint32)[positions]
        wire_relation = np.asarray(
            aligned["wire_relation_state"], dtype=np.uint8
        )[positions]
        return _LayerSource(
            name=name,
            original_index=np.asarray(aligned["original_index"], dtype=np.int64)[
                positions
            ],
            membership_state=membership[positions],
            hag_evidence_state=np.asarray(
                aligned["hag_evidence_state"], dtype=np.uint8
            )[positions],
            hag_readiness=np.asarray(
                aligned["segmentation_hag_readiness"], dtype=np.uint8
            )[positions],
            reason_flags=reason,
            quality_flags=quality,
            raw_hag=np.asarray(aligned["raw_hag_m"], dtype=np.float64)[positions],
            clamped_hag=np.asarray(aligned["clamped_hag_m"], dtype=np.float64)[
                positions
            ],
            wire_relation_state=wire_relation,
            nearest_wire_id=np.asarray(aligned["nearest_wire_id"], dtype=np.int32)[
                positions
            ],
            centerline_distance_m=np.asarray(
                aligned["centerline_distance_m"], dtype=np.float64
            )[positions],
            priority_flagged=_priority_flagged(
                reason_flags=reason,
                quality_flags=quality,
                wire_relation_state=wire_relation,
            ),
            xyz=np.asarray(aligned["xyz_local_m"], dtype=np.float64)[positions],
        )

    return {
        LAYER_MEMBERSHIP_REVIEW: _slice_layer(LAYER_MEMBERSHIP_REVIEW, ~admitted),
        LAYER_HAG_REVIEW: _slice_layer(LAYER_HAG_REVIEW, admitted),
    }


def _build_excluded_layer(payload: Mapping[str, np.ndarray]) -> _LayerSource:
    aligned = _align_sorted(payload)
    membership = np.asarray(aligned["membership_state"], dtype=np.uint8)
    mask = (membership == int(FrvMembershipState.EXCLUDED_NON_CANDIDATE)) | (
        membership == int(FrvMembershipState.MANUAL_EXCLUDE)
    )
    positions = np.flatnonzero(mask)
    count = int(positions.shape[0])
    if count == 0:
        empty = _empty_layer_source(LAYER_EXCLUDED)
        empty.xyz = None
        return empty
    reason = np.asarray(aligned["reason_flags"], dtype=np.uint32)[positions]
    quality = np.asarray(aligned["input_quality_flags"], dtype=np.uint32)[positions]
    source = _LayerSource(
        name=LAYER_EXCLUDED,
        original_index=np.asarray(aligned["original_index"], dtype=np.int64)[
            positions
        ],
        membership_state=membership[positions],
        hag_evidence_state=np.asarray(aligned["hag_evidence_state"], dtype=np.uint8)[
            positions
        ],
        hag_readiness=np.asarray(
            aligned["segmentation_hag_readiness"], dtype=np.uint8
        )[positions],
        reason_flags=reason,
        quality_flags=quality,
        raw_hag=np.asarray(aligned["raw_hag"], dtype=np.float64)[positions],
        clamped_hag=np.asarray(aligned["clamped_hag"], dtype=np.float64)[positions],
        wire_relation_state=np.zeros(count, dtype=np.uint8),
        nearest_wire_id=np.full(count, -1, dtype=np.int32),
        centerline_distance_m=np.zeros(count, dtype=np.float64),
        priority_flagged=_priority_flagged(
            reason_flags=reason,
            quality_flags=quality,
            wire_relation_state=None,
        ),
        xyz=None,
    )
    return source


def _validate_route_consistency(
    payload: Mapping[str, np.ndarray],
    normal: Mapping[str, np.ndarray],
    review: Mapping[str, np.ndarray],
) -> None:
    """Fail-closed per-point cross-validation of V4 rows against the FRV payload.

    The FRV membership payload is the authoritative per-point source of
    membership, HAG, reason, quality and route information.  Every decoded V4
    normal/review row must agree with it on every shared published field, or
    the whole run fails closed.  An index that formally routes to normal can
    never appear in review and vice versa, and neither can ever appear in the
    excluded set.
    """

    aligned = _align_sorted(payload)
    payload_orig = np.asarray(aligned["original_index"], dtype=np.int64)
    payload_membership = np.asarray(aligned["membership_state"], dtype=np.uint8)
    payload_readiness = np.asarray(
        aligned["segmentation_hag_readiness"], dtype=np.uint8
    )
    payload_reason = np.asarray(aligned["reason_flags"], dtype=np.uint32)
    payload_quality = np.asarray(aligned["input_quality_flags"], dtype=np.uint32)
    payload_hag_state = np.asarray(aligned["hag_evidence_state"], dtype=np.uint8)
    payload_raw_hag = np.asarray(aligned["raw_hag"], dtype=np.float64)
    payload_clamped = np.asarray(aligned["clamped_hag"], dtype=np.float64)

    _VERIFIED = int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE)
    _MANUAL_INCLUDE = int(FrvMembershipState.MANUAL_INCLUDE)
    _EXCLUDED = int(FrvMembershipState.EXCLUDED_NON_CANDIDATE)
    _MANUAL_EXCLUDE = int(FrvMembershipState.MANUAL_EXCLUDE)
    _READY = int(FrvSegmentationHagReadiness.READY)
    admitted = (_VERIFIED, _MANUAL_INCLUDE)

    def _positions(originals: np.ndarray) -> np.ndarray:
        if originals.size == 0:
            return np.empty(0, dtype=np.int64)
        if payload_orig.size == 0:
            raise FrvV4ValidationError(
                "a decoded V4 row exists but the authoritative FRV evaluated domain is empty"
            )
        positions = np.searchsorted(payload_orig, originals, side="left")
        safe = np.minimum(positions, payload_orig.size - 1)
        in_domain = (positions < payload_orig.size) & (payload_orig[safe] == originals)
        if not in_domain.all():
            raise FrvV4ValidationError(
                "a decoded V4 row original_index is not in the authoritative "
                "FRV evaluated domain"
            )
        return positions

    def _require_equal(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
        if not np.array_equal(actual, expected):
            raise FrvV4ValidationError(message)

    normal_orig = np.asarray(normal["original_index"], dtype=np.int64)
    normal_pos = _positions(normal_orig)
    if normal_orig.size:
        n_membership = payload_membership[normal_pos]
        n_readiness = payload_readiness[normal_pos]
        n_admitted = (n_membership == _VERIFIED) | (n_membership == _MANUAL_INCLUDE)
        if not (n_admitted & (n_readiness == _READY)).all():
            raise FrvV4ValidationError(
                "a V4 normal row is not formally admitted and HAG-ready"
            )
        if not np.isin(
            np.asarray(normal["membership_state"], dtype=np.uint8), admitted
        ).all():
            raise FrvV4ValidationError(
                "a V4 normal row membership state is not an admitted state"
            )
        if not (np.asarray(normal["segmentation_hag_readiness"], dtype=np.uint8) == _READY).all():
            raise FrvV4ValidationError(
                "a V4 normal row HAG readiness is not formally ready"
            )
        _require_equal(
            np.asarray(normal["membership_state"], dtype=np.uint8),
            n_membership,
            "a V4 normal row membership state disagrees with the FRV payload",
        )
        _require_equal(
            np.asarray(normal["segmentation_hag_readiness"], dtype=np.uint8),
            n_readiness,
            "a V4 normal row HAG readiness disagrees with the FRV payload",
        )
        _require_equal(
            np.asarray(normal["hag_evidence_state"], dtype=np.uint8),
            payload_hag_state[normal_pos],
            "a V4 normal row HAG evidence state disagrees with the FRV payload",
        )
        _require_equal(
            np.asarray(normal["candidate_reason_flags"], dtype=np.uint32),
            payload_reason[normal_pos],
            "a V4 normal row reason flags disagree with the FRV payload",
        )
        if not np.allclose(
            np.asarray(normal["hag_m"], dtype=np.float64),
            payload_clamped[normal_pos],
            rtol=1e-4,
            atol=1e-3,
        ):
            raise FrvV4ValidationError(
                "a V4 normal row HAG value disagrees with the FRV payload"
            )

    review_orig = np.asarray(review["original_index"], dtype=np.int64)
    review_pos = _positions(review_orig)
    if review_orig.size:
        r_membership = payload_membership[review_pos]
        r_readiness = payload_readiness[review_pos]
        r_admitted = (r_membership == _VERIFIED) | (r_membership == _MANUAL_INCLUDE)
        r_excluded = (r_membership == _EXCLUDED) | (r_membership == _MANUAL_EXCLUDE)
        if r_excluded.any():
            raise FrvV4ValidationError(
                "a V4 review row overlaps the authoritative excluded set"
            )
        if (r_admitted & (r_readiness == _READY)).any():
            raise FrvV4ValidationError(
                "a V4 review row is actually a normal (admitted and HAG-ready) point"
            )
        if not (
            np.asarray(review["primary_route"], dtype=np.uint8)
            == int(V4PrimaryRoute.MEASURED_POINT_REVIEW)
        ).all():
            raise FrvV4ValidationError(
                "a V4 review row published primary route is not MEASURED_POINT_REVIEW"
            )
        _require_equal(
            np.asarray(review["membership_state"], dtype=np.uint8),
            r_membership,
            "a V4 review row membership state disagrees with the FRV payload",
        )
        _require_equal(
            np.asarray(review["segmentation_hag_readiness"], dtype=np.uint8),
            r_readiness,
            "a V4 review row HAG readiness disagrees with the FRV payload",
        )
        _require_equal(
            np.asarray(review["hag_evidence_state"], dtype=np.uint8),
            payload_hag_state[review_pos],
            "a V4 review row HAG evidence state disagrees with the FRV payload",
        )
        _require_equal(
            np.asarray(review["reason_flags"], dtype=np.uint32),
            payload_reason[review_pos],
            "a V4 review row reason flags disagree with the FRV payload",
        )
        _require_equal(
            np.asarray(review["quality_flags"], dtype=np.uint32),
            payload_quality[review_pos],
            "a V4 review row quality flags disagree with the FRV payload",
        )
        if not np.allclose(
            np.asarray(review["raw_hag_m"], dtype=np.float64),
            payload_raw_hag[review_pos],
            rtol=1e-4,
            atol=1e-3,
        ):
            raise FrvV4ValidationError(
                "a V4 review row raw HAG disagrees with the FRV payload"
            )
        if not np.allclose(
            np.asarray(review["clamped_hag_m"], dtype=np.float64),
            payload_clamped[review_pos],
            rtol=1e-4,
            atol=1e-3,
        ):
            raise FrvV4ValidationError(
                "a V4 review row clamped HAG disagrees with the FRV payload"
            )


def map_layers(
    *,
    normal: Mapping[str, np.ndarray],
    review: Mapping[str, np.ndarray],
    payload: Mapping[str, np.ndarray],
    source_point_count: int,
) -> dict[str, _LayerSource]:
    """Map the decoded V4 normal/review blocks and FRV payload to formal layers.

    Every measured ``original_index`` lands in exactly one of normal,
    membership review, HAG review or excluded; each decoded V4 row is first
    cross-validated point-by-point against the authoritative FRV payload, then
    route conservation and the original-index domain are verified.
    """

    _validate_route_consistency(payload, normal, review)

    normal_layer = _build_normal_layer(normal)
    review_layers = _build_review_layers(review)
    excluded_layer = _build_excluded_layer(payload)

    normal_set = set(int(value) for value in normal_layer.original_index)
    membership_set = set(
        int(value) for value in review_layers[LAYER_MEMBERSHIP_REVIEW].original_index
    )
    hag_set = set(int(value) for value in review_layers[LAYER_HAG_REVIEW].original_index)
    excluded_set = set(int(value) for value in excluded_layer.original_index)

    sets_by_name = (
        (LAYER_NORMAL, normal_set),
        (LAYER_MEMBERSHIP_REVIEW, membership_set),
        (LAYER_HAG_REVIEW, hag_set),
        (LAYER_EXCLUDED, excluded_set),
    )
    for index, (name, index_set) in enumerate(sets_by_name):
        for other, other_set in sets_by_name[index + 1 :]:
            if not index_set.isdisjoint(other_set):
                raise FrvV4ValidationError(
                    f"route conservation failed: {name} overlaps {other}"
                )

    payload_index = np.asarray(payload["original_index"], dtype=np.int64)
    payload_set = set(int(value) for value in payload_index)
    measured_union = (
        normal_set | membership_set | hag_set | excluded_set
    )
    if not measured_union.issubset(payload_set):
        raise FrvV4ValidationError(
            "measured original indices extend beyond the FRV evaluated domain"
        )
    if measured_union != payload_set:
        raise FrvV4ValidationError(
            "normal + review + excluded do not conserve the evaluated point count"
        )
    if payload_index.size:
        if int(payload_index.min()) < 0 or int(payload_index.max()) >= source_point_count:
            raise FrvV4ValidationError(
                "FRV payload original indices fall outside the source domain"
            )
    for name, index_set in sets_by_name:
        if index_set:
            if int(min(index_set)) < 0 or int(max(index_set)) >= source_point_count:
                raise FrvV4ValidationError(
                    f"{name} original indices exceed the source domain"
                )

    return {
        LAYER_NORMAL: normal_layer,
        LAYER_MEMBERSHIP_REVIEW: review_layers[LAYER_MEMBERSHIP_REVIEW],
        LAYER_HAG_REVIEW: review_layers[LAYER_HAG_REVIEW],
        LAYER_EXCLUDED: excluded_layer,
    }


# ---------------------------------------------------------------------------
# Identity validation and loading
# ---------------------------------------------------------------------------


def _load_validated_view(path: Path) -> V4ValidatedFrvMembershipView:
    """Parse a published ``validated_view.json`` through the frozen contract.

    The contract serializes the two runtime-only block-reference path tuples
    away from ``to_dict`` (they never enter the digest), while ``from_dict``
    requires them.  The viewer supplies empty tuples: it reads the V4 blocks
    directly from the handoff root and never depends on those runtime paths.
    """

    payload = json.loads(path.read_text("utf-8"))
    payload["membership_block_references"] = ()
    payload["decision_evidence_block_references"] = ()
    return V4ValidatedFrvMembershipView.from_dict(payload)


_V1_MANIFEST_FIELDS = {
    "schema_version",
    "source_content_identity_digest",
    "frv_membership_artifact_digest",
    "report_digest",
    "normal_block_count",
    "review_block_count",
}
_V2_MANIFEST_FIELDS = _V1_MANIFEST_FIELDS | {
    "source_identity_record_digest",
    "origin_context",
    "normal_row_count",
    "review_row_count",
    "normal_block_paths",
    "review_block_paths",
    "normal_block_digests",
    "review_block_digests",
    "normal_set_digest",
    "review_set_digest",
    "handoff_digest",
    "manifest_digest",
    "source_proof_digest",
    "hag_proof_digest",
    "geometry_proof_digest",
    "wire_proof_digest",
}


def _read_handoff_manifest(v4_input_root: Path) -> dict[str, Any]:
    path = v4_input_root / "handoff_manifest.json"
    if not path.is_file():
        raise FrvV4ValidationError("V4 handoff manifest is missing")
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") == "v4-input-handoff-v1":
        if set(payload) != _V1_MANIFEST_FIELDS:
            raise FrvV4ValidationError("V4 handoff manifest has an invalid field set")
        return payload
    if payload.get("schema_version") == "v4-input-handoff-v2":
        if set(payload) != _V2_MANIFEST_FIELDS:
            raise FrvV4ValidationError("V4 handoff manifest has an invalid field set")
        return payload
    raise FrvV4ValidationError("unsupported V4 handoff manifest schema version")


def _validate_source_identity_equality(
    checks: dict[str, bool],
    first: SourceIdentity,
    second: SourceIdentity,
) -> None:
    comparisons = {
        "source_content_identity_digest": (
            first.source_content_identity_digest == second.source_content_identity_digest
        ),
        "source_identity_record_digest": (
            first.source_identity_record_digest == second.source_identity_record_digest
        ),
        "source_file_sha256": first.source_file_sha256 == second.source_file_sha256,
        "source_point_count": first.source_point_count == second.source_point_count,
        "source_bounds_min_xyz": first.source_bounds_min_xyz == second.source_bounds_min_xyz,
        "source_bounds_max_xyz": first.source_bounds_max_xyz == second.source_bounds_max_xyz,
        "coordinate_frame": first.coordinate_frame == second.coordinate_frame,
        "units": first.units == second.units,
        "origin_context": first.origin_context == second.origin_context,
    }
    for name, ok in comparisons.items():
        checks[name] = bool(ok)
        if not ok:
            raise FrvV4ValidationError(f"source identity mismatch on {name}")


def _block_paths(v4_input_root: Path, directory: str, prefix: str, count: int) -> list[Path]:
    paths = [
        v4_input_root / directory / f"{prefix}-{ordinal:08d}.v4blk"
        for ordinal in range(count)
    ]
    for path in paths:
        if not path.is_file():
            raise FrvV4ValidationError(f"V4 block file is missing: {path}")
    return paths


def _verify_block_hashes(paths: list[Path], expected_digests: Any) -> None:
    if len(paths) != len(expected_digests):
        raise FrvV4ValidationError("V4 block digest inventory length mismatch")
    for path, expected in zip(paths, expected_digests, strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise FrvV4ValidationError(f"V4 block digest mismatch: {path}")


def _load_ready_blocks(
    result: FrvMembershipExecutionResult,
    frv_artifact_root: Path,
    v4_input_root: Path,
    handoff_manifest: Mapping[str, Any],
    coverage_report: V4RecoveryCoverageReport,
    view: V4ValidatedFrvMembershipView,
    checks: dict[str, bool],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    schema = result.payload_schema
    if schema is None:
        raise FrvV4ValidationError("READY FRV result is missing the payload schema")
    schema_order = tuple(array.field_name for array in schema.arrays)
    payload_dtypes = {array.field_name: np.dtype(array.dtype_code) for array in schema.arrays}
    payload_row_shapes: dict[str, tuple[int, ...]] = {
        array.field_name: () for array in schema.arrays
    }
    artifact = result.membership_artifact
    if artifact is None:
        raise FrvV4ValidationError("READY FRV result is missing the membership artifact")
    payload_paths = [
        frv_artifact_root / block.storage_reference for block in artifact.block_descriptors
    ]
    for path in payload_paths:
        if not path.is_file():
            raise FrvV4ValidationError(f"FRV payload block file is missing: {path}")
    payload = _decode_block_files(
        payload_paths,
        magic=FRV_BLOCK_MAGIC,
        schema_order=schema_order,
        dtypes=payload_dtypes,
        row_shapes=payload_row_shapes,
    )

    normal_count = int(handoff_manifest["normal_block_count"])
    review_count = int(handoff_manifest["review_block_count"])
    normal_paths = _block_paths(v4_input_root, "normal", "v4nrm-block", normal_count)
    review_paths = _block_paths(v4_input_root, "review", "v4rev-block", review_count)
    if "normal_block_digests" in handoff_manifest:
        _verify_block_hashes(normal_paths, handoff_manifest["normal_block_digests"])
    if "review_block_digests" in handoff_manifest:
        _verify_block_hashes(review_paths, handoff_manifest["review_block_digests"])
    normal = _decode_block_files(
        normal_paths,
        magic=V4_NORMAL_BLOCK_MAGIC,
        schema_order=tuple(name for name, _code in _V4_NORMAL_FIELDS),
        dtypes=_normal_block_dtypes(),
        row_shapes=_normal_block_row_shapes(),
    )
    review = _decode_block_files(
        review_paths,
        magic=V4_REVIEW_BLOCK_MAGIC,
        schema_order=tuple(name for name, _code in _V4_REVIEW_FIELDS),
        dtypes=_review_block_dtypes(),
        row_shapes=_review_block_row_shapes(),
    )

    normal_row_count = int(np.asarray(normal["original_index"]).shape[0])
    review_row_count = int(np.asarray(review["original_index"]).shape[0])
    checks["normal_block_count"] = normal_count == len(normal_paths)
    checks["review_block_count"] = review_count == len(review_paths)
    checks["normal_route_count"] = normal_row_count == coverage_report.normal_segmentation_count
    checks["review_route_count"] = review_row_count == coverage_report.measured_point_review_count
    checks["handoff_membership_artifact_digest"] = (
        handoff_manifest["frv_membership_artifact_digest"] == artifact.manifest_digest
    )
    checks["handoff_report_digest"] = (
        handoff_manifest["report_digest"] == coverage_report.report_digest
    )
    checks["coverage_report_membership_artifact_digest"] = (
        coverage_report.frv_membership_artifact_digest == artifact.manifest_digest
    )
    checks["coverage_report_coverage_manifest_digest"] = (
        coverage_report.frv_scope_coverage_manifest_digest
        == result.coverage_manifest.manifest_digest
    )
    checks["view_configuration_digest"] = (
        view.frv_configuration_digest == artifact.configuration_digest
    )
    for name, ok in checks.items():
        if not ok:
            raise FrvV4ValidationError(f"identity/digest check failed on {name}")

    return normal, review, payload


def load_frv_v4_input(
    *,
    source: Path,
    frv_artifact_root: Path,
    v4_input_root: Path,
) -> FrvV4InputData:
    """Load and validate the FRV artifact and V4 handoff for one run."""

    try:
        return _load_frv_v4_input_inner(
            source=source,
            frv_artifact_root=frv_artifact_root,
            v4_input_root=v4_input_root,
        )
    except FrvV4ValidationError:
        raise
    except Exception as exc:
        raise FrvV4ValidationError(
            f"artifact parse/validation failure: {exc}"
        ) from exc


def _load_frv_v4_input_inner(
    *,
    source: Path,
    frv_artifact_root: Path,
    v4_input_root: Path,
) -> FrvV4InputData:
    """Load and validate the FRV artifact and V4 handoff for one run."""

    if not isinstance(frv_artifact_root, Path) or not frv_artifact_root.is_dir():
        raise FrvV4ValidationError("frv artifact root must be an existing directory")
    if not isinstance(v4_input_root, Path) or not v4_input_root.is_dir():
        raise FrvV4ValidationError("v4 input root must be an existing directory")

    execution_path = frv_artifact_root / "execution_result.json"
    if not execution_path.is_file():
        raise FrvV4ValidationError("FRV execution result file is missing")
    execution_payload = json.loads(execution_path.read_text("utf-8"))
    parsed_result = FrvMembershipExecutionResult.from_dict(execution_payload)
    state = parsed_result.execution_state

    checks: dict[str, bool] = {}

    if state is FrvMembershipExecutionState.READY_FOR_V4:
        result = load_frv_membership_result(frv_artifact_root)
        source_identity = result.source_identity
        view_path = v4_input_root / "validated_view.json"
        if not view_path.is_file():
            raise FrvV4ValidationError("V4 validated view file is missing")
        view = _load_validated_view(view_path)
        report_path = v4_input_root / "coverage_report.json"
        if not report_path.is_file():
            raise FrvV4ValidationError("V4 coverage report file is missing")
        coverage_report = V4RecoveryCoverageReport.from_dict(
            json.loads(report_path.read_text("utf-8"))
        )
        handoff_manifest = _read_handoff_manifest(v4_input_root)

        _validate_source_identity_equality(checks, source_identity, view.source_identity)
        checks["coverage_report_content_identity_digest"] = (
            coverage_report.source_content_identity_digest
            == source_identity.source_content_identity_digest
        )
        checks["coverage_report_identity_record_digest"] = (
            coverage_report.source_identity_record_digest
            == source_identity.source_identity_record_digest
        )
        checks["handoff_content_identity_digest"] = (
            handoff_manifest["source_content_identity_digest"]
            == source_identity.source_content_identity_digest
        )
        for name, ok in checks.items():
            if not ok:
                raise FrvV4ValidationError(f"identity/digest check failed on {name}")

        normal, review, payload = _load_ready_blocks(
            result,
            frv_artifact_root,
            v4_input_root,
            handoff_manifest,
            coverage_report,
            view,
            checks,
        )

        layers = map_layers(
            normal=normal,
            review=review,
            payload=payload,
            source_point_count=source_identity.source_point_count,
        )
        excluded_count = int(
            np.asarray(layers[LAYER_EXCLUDED].original_index).shape[0]
        )
        checks["excluded_route_count"] = (
            excluded_count == coverage_report.excluded_audit_count
        )
        if not checks["excluded_route_count"]:
            raise FrvV4ValidationError(
                "excluded point count does not match the coverage report"
            )

        artifact = result.membership_artifact
        assert artifact is not None
        return FrvV4InputData(
            mode="ready",
            source_identity=source_identity,
            coverage_report=coverage_report,
            layers=layers,
            regions=(),
            identity_checks=checks,
            configuration_purpose=artifact.configuration_purpose,
        )

    if state is FrvMembershipExecutionState.COVERAGE_BLOCKED:
        coverage_path = frv_artifact_root / "coverage_manifest.json"
        if not coverage_path.is_file():
            raise FrvV4ValidationError(
                "COVERAGE_BLOCKED FRV result is missing the coverage manifest"
            )
        manifest = FRVScopeCoverageManifest.from_dict(
            json.loads(coverage_path.read_text("utf-8"))
        )
        if manifest.manifest_digest != manifest.computed_digest():
            raise FrvV4ValidationError("coverage manifest digest mismatch")
        if (
            parsed_result.coverage_manifest.manifest_digest
            != manifest.manifest_digest
        ):
            raise FrvV4ValidationError(
                "execution result coverage manifest does not match the manifest file"
            )
        source_identity = parsed_result.source_identity
        regions: list[FrvV4RegionRecord] = []
        for record in manifest.records:
            regions.append(
                FrvV4RegionRecord(
                    key=record.canonical_inventory_key,
                    scope_id=record.scope_id,
                    coverage_state=record.scope_coverage_state.value,
                    failure_reason_flags=int(record.failure_reason_flags),
                    review_required=record.review_required,
                    diagnostic_reference=record.diagnostic_reference,
                    geometry_status="missing-geometry",
                    bounds=None,
                )
            )
        checks["coverage_routing_complete"] = manifest.routing_complete is False
        return FrvV4InputData(
            mode="blocked",
            source_identity=source_identity,
            coverage_report=None,
            layers={},
            regions=tuple(regions),
            identity_checks=checks,
        )

    raise FrvV4ValidationError(
        f"FRV execution state is not visualizable: {state.value}"
    )


# ---------------------------------------------------------------------------
# Display layer assembly
# ---------------------------------------------------------------------------


def _ground_low_hag(source: _LayerSource) -> np.ndarray:
    hag_state = np.asarray(source.hag_evidence_state, dtype=np.uint8)
    clamped = np.asarray(source.clamped_hag, dtype=np.float64)
    quality = np.asarray(source.quality_flags, dtype=np.uint32)
    reason = np.asarray(source.reason_flags, dtype=np.uint32)
    problem_states = (
        int(FrvHagEvidenceState.HAG_MISSING),
        int(FrvHagEvidenceState.HAG_INVALID),
        int(FrvHagEvidenceState.HAG_OUT_OF_DOMAIN),
    )
    mask = np.zeros(hag_state.shape[0], dtype=bool)
    for value in problem_states:
        mask |= hag_state == value
    mask |= clamped < LOW_HAG_DISPLAY_M
    mask |= quality != 0
    hag_reason_mask = np.uint32(
        int(FrvReasonFlag.HAG_MISSING | FrvReasonFlag.HAG_INVALID | FrvReasonFlag.HAG_OUT_OF_DOMAIN)
    )
    mask |= (reason & hag_reason_mask) != 0
    return mask


def _subslice_arrays(source: _LayerSource, positions: np.ndarray) -> _LayerSource:
    """Return a copy of the layer restricted to the given original positions."""
    return _LayerSource(
        name=source.name,
        original_index=np.asarray(source.original_index, dtype=np.int64)[positions],
        membership_state=np.asarray(source.membership_state, dtype=np.uint8)[positions],
        hag_evidence_state=np.asarray(source.hag_evidence_state, dtype=np.uint8)[
            positions
        ],
        hag_readiness=np.asarray(source.hag_readiness, dtype=np.uint8)[positions],
        reason_flags=np.asarray(source.reason_flags, dtype=np.uint32)[positions],
        quality_flags=np.asarray(source.quality_flags, dtype=np.uint32)[positions],
        raw_hag=np.asarray(source.raw_hag, dtype=np.float64)[positions],
        clamped_hag=np.asarray(source.clamped_hag, dtype=np.float64)[positions],
        wire_relation_state=np.asarray(source.wire_relation_state, dtype=np.uint8)[
            positions
        ],
        nearest_wire_id=np.asarray(source.nearest_wire_id, dtype=np.int32)[positions],
        centerline_distance_m=np.asarray(
            source.centerline_distance_m, dtype=np.float64
        )[positions],
        priority_flagged=np.asarray(source.priority_flagged, dtype=bool)[positions],
        xyz=(
            np.asarray(source.xyz, dtype=np.float64)[positions]
            if source.xyz is not None
            else None
        ),
    )


def _build_customdata(source: _LayerSource, route: str) -> np.ndarray:
    """Build a pre-formatted object customdata array for the hover template."""
    n = int(source.original_index.shape[0])
    customdata = np.empty((n, 12), dtype=object)
    customdata[:, 0] = [str(int(value)) for value in source.original_index]
    customdata[:, 1] = route
    customdata[:, 2] = [
        _enum_name(FrvMembershipState, int(value)) for value in source.membership_state
    ]
    customdata[:, 3] = [
        _enum_name(FrvHagEvidenceState, int(value))
        for value in source.hag_evidence_state
    ]
    customdata[:, 4] = [
        _enum_name(FrvSegmentationHagReadiness, int(value))
        for value in source.hag_readiness
    ]
    customdata[:, 5] = [
        _format_flag_names(int(value), FrvReasonFlag) for value in source.reason_flags
    ]
    customdata[:, 6] = [
        _format_flag_names(int(value), FrvQualityFlag) for value in source.quality_flags
    ]
    customdata[:, 7] = [_fmt_float(float(value)) for value in source.clamped_hag]
    customdata[:, 8] = [_fmt_float(float(value)) for value in source.raw_hag]
    customdata[:, 9] = [
        _enum_name(FrvWirePointRelationState, int(value))
        for value in source.wire_relation_state
    ]
    customdata[:, 10] = [str(int(value)) for value in source.nearest_wire_id]
    customdata[:, 11] = [
        _fmt_float(float(value)) for value in source.centerline_distance_m
    ]
    return customdata


def _background_customdata(indices: np.ndarray) -> np.ndarray:
    n = int(indices.shape[0])
    customdata = np.empty((n, 12), dtype=object)
    customdata[:, 0] = [str(int(value)) for value in indices]
    customdata[:, 1] = "SOURCE_BACKGROUND"
    customdata[:, 2:] = "n/a"
    return customdata


_HOVER_TEMPLATE = (
    "original_index=%{customdata[0]}<br>"
    "route=%{customdata[1]}<br>"
    "membership=%{customdata[2]}<br>"
    "hag_evidence=%{customdata[3]}<br>"
    "hag_readiness=%{customdata[4]}<br>"
    "reason=%{customdata[5]}<br>"
    "quality=%{customdata[6]}<br>"
    "clamped_hag=%{customdata[7]} m<br>"
    "raw_hag=%{customdata[8]} m<br>"
    "wire_relation=%{customdata[9]}<br>"
    "nearest_wire_id=%{customdata[10]}<br>"
    "wire_distance=%{customdata[11]} m<extra></extra>"
)


# ---------------------------------------------------------------------------
# Summary and checklist
# ---------------------------------------------------------------------------


def _authority_display(purpose: FrvConfigurationPurpose | None) -> dict[str, Any]:
    """Derive the frozen authority-disclosure section from the upstream purpose.

    ``REVIEWED_FIXED_POLICY`` is shown as a review/evaluation authority with an
    explicit no-calibration claim boundary.  No reviewed-fixed path ever claims
    production calibration, held-out validation, or production certification.
    """

    if purpose is FrvConfigurationPurpose.REVIEWED_FIXED_POLICY:
        return {
            "configuration_purpose": "REVIEWED_FIXED_POLICY",
            "authority_label": (
                "REVIEWED_FIXED_POLICY — REVIEW/EVALUATION AUTHORITY"
            ),
            "review_evaluation_authority": True,
            "production_calibrated": False,
            "data_calibrated": False,
            "held_out_validated": False,
            "production_certified": False,
            "claim_boundary": [
                "NOT DATA CALIBRATED",
                "NOT HELD-OUT VALIDATED",
                "NOT PRODUCTION CERTIFIED",
            ],
        }
    if purpose is FrvConfigurationPurpose.PRODUCTION_APPROVED:
        return {
            "configuration_purpose": "PRODUCTION_APPROVED",
            "authority_label": "PRODUCTION_APPROVED",
            "review_evaluation_authority": False,
            "production_calibrated": False,
            "data_calibrated": False,
            "held_out_validated": False,
            "production_certified": False,
            "claim_boundary": [],
        }
    return {
        "configuration_purpose": None,
        "authority_label": "NO UPSTREAM AUTHORITY DISCLOSED",
        "review_evaluation_authority": False,
        "production_calibrated": False,
        "data_calibrated": False,
        "held_out_validated": False,
        "production_certified": False,
        "claim_boundary": [],
    }


def build_summary(
    *,
    mode: str,
    source_identity: SourceIdentity,
    coverage_report: V4RecoveryCoverageReport | None,
    regions: tuple[FrvV4RegionRecord, ...],
    layers: Mapping[str, _LayerSource],
    allocation: DisplayAllocation,
    display_counts: Mapping[str, LayerDisplayCounts],
    identity_checks: Mapping[str, bool],
    budget: DisplayBudget,
    show_flags: Mapping[str, bool],
    output_paths: Mapping[str, Path],
    authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen FRV-VIS-A1 summary JSON document."""

    by_name = {result.name: result for result in allocation.layers}

    layer_section: dict[str, Any] = {}
    for name in _ALL_LAYERS:
        display = _LAYER_DISPLAY[name]
        counts = display_counts.get(name, LayerDisplayCounts(0, 0))
        if name in by_name:
            result = by_name[name]
            entry: dict[str, Any] = {
                "display_name": display["display_name"],
                "route": display["route"],
                "full_count": result.full_count,
                "displayed_count": result.displayed_count,
                "dropped_display_count": result.dropped_display_count,
                "requested_cap": result.requested_cap,
                "applied_cap": result.applied_cap,
                "is_overlay": False,
                "is_region": False,
            }
        elif name == LAYER_BACKGROUND:
            result = allocation.background
            entry = {
                "display_name": display["display_name"],
                "route": display["route"],
                "full_count": result.full_count,
                "displayed_count": result.displayed_count,
                "dropped_display_count": result.dropped_display_count,
                "requested_cap": result.requested_cap,
                "applied_cap": result.applied_cap,
                "is_overlay": False,
                "is_region": False,
            }
        elif name == LAYER_UNRECOVERED:
            entry = {
                "display_name": display["display_name"],
                "route": display["route"],
                "full_count": counts.full_count,
                "displayed_count": 0,
                "dropped_display_count": 0,
                "requested_cap": None,
                "applied_cap": None,
                "is_overlay": False,
                "is_region": True,
            }
        else:
            entry = {
                "display_name": display["display_name"],
                "route": display["route"],
                "full_count": counts.full_count,
                "displayed_count": counts.displayed_count,
                "dropped_display_count": counts.full_count - counts.displayed_count,
                "requested_cap": None,
                "applied_cap": None,
                "is_overlay": True,
                "is_region": False,
            }
        layer_section[name] = entry

    region_section = {
        "unrecovered_region_count": len(regions),
        "region_geometry_status": "missing-geometry" if regions else "none",
        "regions": [
            {
                "canonical_inventory_key": region.key,
                "scope_id": region.scope_id,
                "coverage_state": region.coverage_state,
                "failure_reason_flags": region.failure_reason_flags,
                "review_required": region.review_required,
                "diagnostic_reference": region.diagnostic_reference,
                "geometry_status": region.geometry_status,
                "bounds": region.bounds,
            }
            for region in regions
        ],
    }

    if coverage_report is not None:
        completeness = {
            "frv_routing_complete": coverage_report.frv_routing_complete,
            "frv_membership_decision_complete": (
                coverage_report.frv_membership_decision_complete
            ),
            "segmentation_input_complete": coverage_report.segmentation_input_complete,
            "v4_routing_complete": coverage_report.v4_routing_complete,
            "xyz_recovery_complete": coverage_report.xyz_recovery_complete,
            "hag_ready_complete": coverage_report.hag_ready_complete,
        }
        routes = {
            "normal_segmentation_count": coverage_report.normal_segmentation_count,
            "measured_point_review_count": coverage_report.measured_point_review_count,
            "excluded_audit_count": coverage_report.excluded_audit_count,
            "unrecovered_region_count": coverage_report.unrecovered_region_count,
            "point_recovery_failed_count": coverage_report.point_recovery_failed_count,
        }
    else:
        completeness = {
            "frv_routing_complete": False,
            "frv_membership_decision_complete": False,
            "segmentation_input_complete": False,
            "v4_routing_complete": False,
            "xyz_recovery_complete": False,
            "hag_ready_complete": False,
        }
        routes = {
            "normal_segmentation_count": 0,
            "measured_point_review_count": 0,
            "excluded_audit_count": 0,
            "unrecovered_region_count": len(regions),
            "point_recovery_failed_count": 0,
        }

    denominator = int(source_identity.source_point_count)
    measured_full = sum(
        int(np.asarray(layers[name].original_index).shape[0])
        for name in _MEASURED_LAYERS
        if name in layers
    )

    return {
        "tool": TOOL_NAME,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "mode": mode,
        "authority": (
            authority
            if authority is not None
            else _authority_display(None)
        ),
        "source": {
            "source_id": source_identity.source_id,
            "source_content_identity_digest": (
                source_identity.source_content_identity_digest
            ),
            "source_identity_record_digest": (
                source_identity.source_identity_record_digest
            ),
            "source_file_sha256": source_identity.source_file_sha256,
            "source_point_count": source_identity.source_point_count,
            "source_bounds_min_xyz": list(source_identity.source_bounds_min_xyz),
            "source_bounds_max_xyz": list(source_identity.source_bounds_max_xyz),
            "coordinate_frame": source_identity.coordinate_frame,
            "units": source_identity.units,
            "origin_context": list(source_identity.origin_context),
        },
        "status": {
            **completeness,
            "artifact_status": "validated" if mode == "ready" else "coverage-blocked",
            "handoff_status": "validated" if coverage_report is not None else "none",
        },
        "identity_validation": dict(identity_checks),
        "routes": routes,
        "layers": layer_section,
        "regions": region_section,
        "display_budget": {
            "requested_total_cap": allocation.requested_total_cap,
            "applied_total_cap": allocation.applied_total_cap,
            "total_full_point_count": allocation.total_full_point_count,
            "total_displayed_point_count": allocation.total_displayed_point_count,
            "total_dropped_display_count": allocation.total_dropped_display_count,
            "requested_layer_cap": budget.max_points_per_layer,
            "requested_background_cap": allocation.requested_background_cap,
            "applied_background_cap": allocation.applied_background_cap,
            "sampling_seed": allocation.sampling_seed,
            "allocation_priority": list(ALLOCATION_PRIORITY),
            "default_total_cap": DEFAULT_TOTAL_CAP,
            "hard_total_cap": HARD_TOTAL_CAP,
            "default_layer_cap": DEFAULT_LAYER_CAP,
            "max_layer_cap": MAX_LAYER_CAP,
            "default_background_cap": DEFAULT_BACKGROUND_CAP,
            "max_background_cap": MAX_BACKGROUND_CAP,
            "rejection_status": "none",
        },
        "denominators": {
            "point_ratio_denominator": denominator,
            "measured_point_ratio_denominator": denominator or 1,
            "measured_point_total": measured_full,
        },
        "show_flags": {
            "show_original_background": bool(show_flags.get("background", False)),
            "show_excluded": bool(show_flags.get("excluded", False)),
            "show_review": bool(show_flags.get("review", False)),
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


_CHECKLIST_ROWS = (
    ("树冠主体", "normal 是否覆盖主体，是否存在明显整片缺失"),
    ("树冠边缘", "是否有环状漏点或只剩内部点"),
    ("树干", "是否被系统性排除或仅落入合理 review"),
    ("导线混入", "导线是否进入绿色 normal/admitted"),
    ("杆塔/建筑", "边缘是否混入 normal"),
    ("地面", "normal 是否大量低 HAG/地面点"),
    ("导线附近枝叶", "真实枝叶是否被整片删除，wire review 是否集中在合理边界"),
    ("review 集中性", "membership/HAG review 是否集中在证据不足或边界位置"),
    (
        "blocked/unrecovered",
        "是否有整片 blocked，是否每条都有可追溯 key/geometry 或明确缺 geometry",
    ),
    ("normal 连续性", "是否存在明显断层、条带缺失或异常空洞"),
    ("树间桥接", "输入污染是否让不同树形成连续 normal 桥接"),
    ("V4 适用性", "是否可继续正式单木分割"),
)

_CONCLUSIONS = (
    "PASS_READY_FOR_V4",
    "PASS_WITH_REVIEW_NOTES",
    "FRV_RULE_ADJUSTMENT_REQUIRED",
    "DATA_OR_ARTIFACT_BLOCKED",
)


def build_checklist_markdown(*, summary: Mapping[str, Any]) -> str:
    """Generate the pre-filled human acceptance checklist markdown."""

    source = summary["source"]
    authority = summary.get("authority", {})
    authority_lines = [f"- Authority: `{authority.get('authority_label')}`"]
    for boundary in authority.get("claim_boundary", []):
        authority_lines.append(f"  - Claim boundary: `{boundary}`")
    lines = [
        "# FRV-VIS-A1 Human Acceptance Checklist",
        "",
        f"- Tool: `{summary['tool']}`",
        f"- Schema version: `{summary['summary_schema_version']}`",
        f"- Mode: `{summary['mode']}`",
        *authority_lines,
        f"- Source id: `{source['source_id']}`",
        f"- Source content identity digest: `{source['source_content_identity_digest']}`",
        f"- Source point count: {source['source_point_count']}",
        f"- Origin context: {source['origin_context']}",
        "",
        "## 检查项",
        "",
        "| # | 检查项 | 关注点 | 结论 | 截图编号 | 证据(original-index/region key) | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, (item, focus) in enumerate(_CHECKLIST_ROWS, start=1):
        lines.append(f"| {index} | {item} | {focus} |  |  |  |  |")
    lines.extend(
        [
            "",
            "## 人工结论",
            "",
            "请勾选唯一结论：",
            "",
        ]
    )
    for conclusion in _CONCLUSIONS:
        lines.append(f"- [ ] `{conclusion}`")
    lines.extend(
        [
            "",
            "## 判定规则",
            "",
            "- `PASS_READY_FOR_V4`：无关键混入/漏检，所有正式 complete flags 满足要求。",
            "- `PASS_WITH_REVIEW_NOTES`：可进入 V4，但存在已记录且不阻断的局部 review。",
            "- `FRV_RULE_ADJUSTMENT_REQUIRED`：误判模式稳定指向 FRV reason/decision 规则。",
            "- `DATA_OR_ARTIFACT_BLOCKED`：身份、artifact、coverage、坐标、资源或 "
            "blocked geometry 无法可信验证。",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotly renderer
# ---------------------------------------------------------------------------


def _trace_name(layer_name: str) -> str:
    return _LAYER_DISPLAY[layer_name]["display_name"]


def _build_trace(display: _DisplayLayer) -> go.Scatter3d:
    cfg = _LAYER_DISPLAY[display.name]
    visible: Any = True if display.default_visible else "legendonly"
    return go.Scatter3d(
        x=display.xyz[:, 0],
        y=display.xyz[:, 1],
        z=display.xyz[:, 2],
        mode="markers",
        name=_trace_name(display.name),
        visible=visible,
        marker={
            "size": cfg["marker_size"],
            "color": cfg["color"],
            "opacity": cfg["opacity"],
        },
        customdata=display.customdata,
        hovertemplate=_HOVER_TEMPLATE,
    )


def _render_blocked_table(
    regions: Sequence[FrvV4RegionRecord], title: str, authority_text: str = ""
) -> str:
    keys = [region.key for region in regions]
    scopes = [region.scope_id if region.scope_id is not None else "n/a" for region in regions]
    states = [region.coverage_state for region in regions]
    flags = [str(region.failure_reason_flags) for region in regions]
    geometry = [region.geometry_status for region in regions]
    figure = go.Figure(
        data=[
            go.Table(
                header={
                    "values": [
                        "canonical_inventory_key",
                        "scope_id",
                        "coverage_state",
                        "failure_reason_flags",
                        "review_required",
                        "geometry_status",
                    ]
                },
                cells={
                    "values": [
                        keys,
                        scopes,
                        states,
                        flags,
                        [str(region.review_required) for region in regions],
                        geometry,
                    ]
                },
            )
        ]
    )
    annotations: list[dict[str, Any]] = [
        {
            "text": (
                "Blocked / unrecovered regions carry no verified geometry. "
                "No point or bounding box is fabricated."
            ),
            "xref": "paper",
            "yref": "paper",
            "x": 0.02,
            "y": 0.95,
            "showarrow": False,
            "font": {"color": "#d62728"},
        }
    ]
    if authority_text:
        annotations.append(
            {
                "text": authority_text,
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.92,
                "showarrow": False,
            }
        )
    figure.update_layout(title={"text": title, "x": 0.5}, annotations=annotations)
    return figure.to_html(include_plotlyjs=True, full_html=True)


def _authority_html_text(authority: dict[str, Any] | None) -> str:
    if not authority:
        return ""
    parts = [str(authority.get("authority_label") or "")]
    parts.extend(str(item) for item in authority.get("claim_boundary", []))
    return " | ".join(part for part in parts if part)


def render_html(
    *,
    title: str,
    display_layers: Sequence[_DisplayLayer],
    regions: Sequence[FrvV4RegionRecord],
    mode: str,
    authority: dict[str, Any] | None = None,
) -> str:
    """Render the self-contained interactive Plotly HTML document."""

    authority_text = _authority_html_text(authority)
    if mode == "blocked" or not display_layers:
        if mode == "blocked" and regions:
            return _render_blocked_table(regions, title, authority_text=authority_text)
        figure = go.Figure()
        annotations: list[dict[str, Any]] = [
            {
                "text": "No measured points to display for this input.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ]
        if authority_text:
            annotations.append(
                {
                    "text": authority_text,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.9,
                    "showarrow": False,
                }
            )
        figure.update_layout(title={"text": title, "x": 0.5}, annotations=annotations)
        return figure.to_html(include_plotlyjs=True, full_html=True)

    traces = [_build_trace(display) for display in display_layers]
    figure = go.Figure(data=traces)

    trace_names = [trace.name for trace in traces]

    def _visible_preset(layer_names: set[str]) -> list[Any]:
        return [
            True if name in layer_names else "legendonly" for name in trace_names
        ]

    measured = {
        _trace_name(LAYER_NORMAL),
        _trace_name(LAYER_MEMBERSHIP_REVIEW),
        _trace_name(LAYER_HAG_REVIEW),
    }
    buttons = [
        {
            "label": "All on",
            "method": "update",
            "args": [{"visible": [True] * len(traces)}],
        },
        {
            "label": "All off",
            "method": "update",
            "args": [{"visible": ["legendonly"] * len(traces)}],
        },
        {
            "label": "Measured",
            "method": "update",
            "args": [{"visible": _visible_preset(measured)}],
        },
        {
            "label": "Reviews",
            "method": "update",
            "args": [
                {
                    "visible": _visible_preset(
                        {
                            _trace_name(LAYER_MEMBERSHIP_REVIEW),
                            _trace_name(LAYER_HAG_REVIEW),
                        }
                    )
                }
            ],
        },
        {
            "label": "Excluded",
            "method": "update",
            "args": [{"visible": _visible_preset({_trace_name(LAYER_EXCLUDED)})}],
        },
        {
            "label": "Background",
            "method": "update",
            "args": [{"visible": _visible_preset({_trace_name(LAYER_BACKGROUND)})}],
        },
        {
            "label": "Overlays",
            "method": "update",
            "args": [
                {
                    "visible": _visible_preset(
                        {
                            _trace_name(LAYER_WIRE_OVERLAY),
                            _trace_name(LAYER_GROUND_OVERLAY),
                        }
                    )
                }
            ],
        },
    ]

    scene: dict[str, Any] = {
        "xaxis_title": "local X (m)",
        "yaxis_title": "local Y (m)",
        "zaxis_title": "local Z (m)",
        "aspectmode": "data",
    }

    layout: dict[str, Any] = {
        "scene": scene,
        "margin": {"l": 0, "r": 0, "b": 0, "t": 40},
        "title": {"text": title, "x": 0.5},
        "legend": {"title": {"text": "Layer"}},
        "updatemenus": [
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.15,
            }
        ],
    }

    annotations: list[dict[str, Any]] = []
    if authority_text:
        annotations.append(
            {
                "text": authority_text,
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.98,
                "showarrow": False,
            }
        )
    if regions:
        annotations.append(
            {
                "text": (
                    "Blocked / unrecovered regions carry no verified geometry; "
                    "they are recorded in the summary audit table and checklist."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.95,
                "showarrow": False,
                "font": {"color": "#d62728"},
            }
        )
    if annotations:
        layout["annotations"] = annotations

    figure.update_layout(**layout)
    return figure.to_html(include_plotlyjs=True, full_html=True)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        raise


def _check_output_target(path: Path, kind: str) -> None:
    if path.exists():
        raise FrvV4OutputError(f"{kind} already exists; refusing to overwrite: {path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvV4VisualizationResult:
    """Paths and summary produced by one visualization run."""

    mode: str
    html_path: Path
    summary_path: Path | None
    checklist_path: Path | None
    summary: dict[str, Any]


def _displayed_subslice(
    data: FrvV4InputData,
    allocation: DisplayAllocation,
    layer_name: str,
) -> _LayerSource:
    source = data.layers[layer_name]
    allocation_result = next(
        item for item in allocation.layers if item.name == layer_name
    )
    displayed = np.asarray(allocation_result.displayed_index, dtype=np.int64)
    if displayed.size == 0:
        return _subslice_arrays(source, np.empty(0, dtype=np.int64))
    positions = np.searchsorted(
        np.asarray(source.original_index, dtype=np.int64), displayed
    )
    return _subslice_arrays(source, positions)


def _assemble_display_layers(
    data: FrvV4InputData,
    allocation: DisplayAllocation,
    las: _LasReadResult,
    *,
    show_original_background: bool,
    show_excluded: bool,
) -> tuple[list[_DisplayLayer], dict[str, LayerDisplayCounts]]:
    """Build the renderable traces and per-layer full/displayed counts."""

    display_layers: list[_DisplayLayer] = []
    counts: dict[str, LayerDisplayCounts] = {}

    def _append(
        name: str,
        xyz: np.ndarray,
        indices: np.ndarray,
        customdata: np.ndarray,
        default_visible: bool,
        is_overlay: bool,
        full_count: int,
        displayed_count: int,
    ) -> None:
        display_layers.append(
            _DisplayLayer(
                name=name,
                xyz=xyz,
                original_index=indices,
                customdata=customdata,
                default_visible=default_visible,
                is_overlay=is_overlay,
            )
        )
        counts[name] = LayerDisplayCounts(full_count, displayed_count)

    for layer_name in (LAYER_NORMAL, LAYER_MEMBERSHIP_REVIEW, LAYER_HAG_REVIEW):
        sub = _displayed_subslice(data, allocation, layer_name)
        source = data.layers[layer_name]
        full_count = int(np.asarray(source.original_index).shape[0])
        displayed_count = int(sub.original_index.shape[0])
        default_visible = bool(_LAYER_DISPLAY[layer_name]["default_visible"])
        _append(
            layer_name,
            np.asarray(sub.xyz, dtype=np.float64),
            np.asarray(sub.original_index, dtype=np.int64),
            _build_customdata(sub, _LAYER_DISPLAY[layer_name]["route"]),
            default_visible=default_visible,
            is_overlay=False,
            full_count=full_count,
            displayed_count=displayed_count,
        )

    # Excluded coordinates come from the source LAS display reader.
    excluded_sub = _displayed_subslice(data, allocation, LAYER_EXCLUDED)
    excluded_full = int(
        np.asarray(data.layers[LAYER_EXCLUDED].original_index).shape[0]
    )
    excluded_displayed = int(excluded_sub.original_index.shape[0])
    if excluded_displayed:
        las_positions = np.searchsorted(
            las.extracted_index, np.asarray(excluded_sub.original_index, dtype=np.int64)
        )
        excluded_xyz = las.xyz_local[las_positions]
        excluded_sub.xyz = excluded_xyz
    else:
        excluded_xyz = np.empty((0, 3), dtype=np.float64)
    _append(
        LAYER_EXCLUDED,
        excluded_xyz,
        np.asarray(excluded_sub.original_index, dtype=np.int64),
        _build_customdata(excluded_sub, _LAYER_DISPLAY[LAYER_EXCLUDED]["route"]),
        default_visible=show_excluded,
        is_overlay=False,
        full_count=excluded_full,
        displayed_count=excluded_displayed,
    )

    # Original background from the source LAS display reader.
    background_index = np.asarray(allocation.background.displayed_index, dtype=np.int64)
    background_displayed = int(background_index.shape[0])
    if background_displayed:
        las_positions = np.searchsorted(las.extracted_index, background_index)
        background_xyz = las.xyz_local[las_positions]
    else:
        background_xyz = np.empty((0, 3), dtype=np.float64)
    _append(
        LAYER_BACKGROUND,
        background_xyz,
        background_index,
        _background_customdata(background_index),
        default_visible=show_original_background,
        is_overlay=False,
        full_count=allocation.background.full_count,
        displayed_count=background_displayed,
    )

    # Unrecovered regions are never points; the trace carries only the name.
    region_full = len(data.regions)
    _append(
        LAYER_UNRECOVERED,
        np.empty((0, 3), dtype=np.float64),
        np.empty(0, dtype=np.int64),
        np.empty((0, 12), dtype=object),
        default_visible=True,
        is_overlay=False,
        full_count=region_full,
        displayed_count=0,
    )

    def _overlay_layer(
        overlay_name: str,
        base_names: tuple[str, ...],
        flag_fn: Any,
    ) -> None:
        pieces: list[_LayerSource] = []
        overlay_full = 0
        for base_name in base_names:
            source = data.layers[base_name]
            overlay_full += int(np.count_nonzero(flag_fn(source)))
            sub = _displayed_subslice(data, allocation, base_name)
            if sub.original_index.size == 0:
                continue
            mask = flag_fn(sub)
            flagged_positions = np.flatnonzero(mask)
            if flagged_positions.size:
                pieces.append(_subslice_arrays(sub, flagged_positions))
        if not pieces:
            _append(
                overlay_name,
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty((0, 12), dtype=object),
                default_visible=False,
                is_overlay=True,
                full_count=overlay_full,
                displayed_count=0,
            )
            return
        combined = _LayerSource(
            name=overlay_name,
            original_index=np.concatenate([p.original_index for p in pieces]),
            membership_state=np.concatenate([p.membership_state for p in pieces]),
            hag_evidence_state=np.concatenate(
                [p.hag_evidence_state for p in pieces]
            ),
            hag_readiness=np.concatenate([p.hag_readiness for p in pieces]),
            reason_flags=np.concatenate([p.reason_flags for p in pieces]),
            quality_flags=np.concatenate([p.quality_flags for p in pieces]),
            raw_hag=np.concatenate([p.raw_hag for p in pieces]),
            clamped_hag=np.concatenate([p.clamped_hag for p in pieces]),
            wire_relation_state=np.concatenate(
                [p.wire_relation_state for p in pieces]
            ),
            nearest_wire_id=np.concatenate([p.nearest_wire_id for p in pieces]),
            centerline_distance_m=np.concatenate(
                [p.centerline_distance_m for p in pieces]
            ),
            priority_flagged=np.concatenate([p.priority_flagged for p in pieces]),
            xyz=np.concatenate([p.xyz for p in pieces]),
        )
        _append(
            overlay_name,
            np.asarray(combined.xyz, dtype=np.float64),
            np.asarray(combined.original_index, dtype=np.int64),
            _build_customdata(combined, _LAYER_DISPLAY[overlay_name]["route"]),
            default_visible=False,
            is_overlay=True,
            full_count=overlay_full,
            displayed_count=int(combined.original_index.shape[0]),
        )

    _overlay_layer(
        LAYER_WIRE_OVERLAY,
        (LAYER_MEMBERSHIP_REVIEW, LAYER_HAG_REVIEW),
        lambda source: np.array(
            [
                _wire_conflict(int(value))
                for value in np.asarray(source.wire_relation_state, dtype=np.uint8)
            ],
            dtype=bool,
        ),
    )
    _overlay_layer(
        LAYER_GROUND_OVERLAY,
        (LAYER_NORMAL, LAYER_MEMBERSHIP_REVIEW, LAYER_HAG_REVIEW),
        _ground_low_hag,
    )

    return display_layers, counts


def run_frv_v4_visualization(
    *,
    source: Path,
    frv_artifact_root: Path,
    v4_input_root: Path,
    output_html: Path,
    summary_json: Path | None = None,
    review_checklist: Path | None = None,
    max_total_points: Any = DEFAULT_TOTAL_CAP,
    max_points_per_layer: Any = DEFAULT_LAYER_CAP,
    max_background_points: Any = DEFAULT_BACKGROUND_CAP,
    sampling_seed: Any = 0,
    show_original_background: bool = False,
    show_excluded: bool = False,
    show_review: bool = False,
) -> FrvV4VisualizationResult:
    """Run one FRV-VIS-A1 visualization end to end.

    Any budget, source, artifact, identity, digest or coordinate failure raises
    before any output file is written; the CLI maps these to a non-zero exit.
    """

    try:
        budget = validate_display_budget(
            max_total_points,
            max_points_per_layer,
            max_background_points,
            sampling_seed,
        )
    except ValueError as exc:
        raise FrvV4BudgetError(str(exc)) from exc

    html = Path(output_html)
    _check_output_target(html, "html output")
    if summary_json is not None:
        _check_output_target(Path(summary_json), "summary JSON output")
    if review_checklist is not None:
        _check_output_target(Path(review_checklist), "review checklist output")

    data = load_frv_v4_input(
        source=source,
        frv_artifact_root=frv_artifact_root,
        v4_input_root=v4_input_root,
    )

    if data.mode == "blocked":
        allocation = DisplayAllocation(
            layers=(),
            background=LayerBudgetResult(
                name=LAYER_BACKGROUND,
                full_count=0,
                displayed_count=0,
                dropped_display_count=0,
                requested_cap=budget.max_background_points,
                applied_cap=0,
                displayed_index=np.empty(0, dtype=np.int64),
            ),
            total_full_point_count=0,
            total_displayed_point_count=0,
            total_dropped_display_count=0,
            requested_total_cap=budget.max_total_points,
            applied_total_cap=budget.max_total_points,
            requested_background_cap=budget.max_background_points,
            applied_background_cap=0,
            sampling_seed=budget.sampling_seed,
        )
        needed = np.empty(0, dtype=np.int64)
    else:
        allocation = allocate_display_budget(
            data.layers,
            source_point_count=data.source_identity.source_point_count,
            budget=budget,
        )
        excluded_allocation = next(
            item for item in allocation.layers if item.name == LAYER_EXCLUDED
        )
        needed = np.sort(
            np.union1d(
                np.asarray(excluded_allocation.displayed_index, dtype=np.int64),
                np.asarray(allocation.background.displayed_index, dtype=np.int64),
            )
        )

    las = read_las_display(
        source,
        needed=needed,
        origin=data.source_identity.origin_context,
        expected_count=data.source_identity.source_point_count,
        expected_sha256=data.source_identity.source_file_sha256,
        expected_bounds_min=data.source_identity.source_bounds_min_xyz,
        expected_bounds_max=data.source_identity.source_bounds_max_xyz,
    )

    if data.mode == "blocked":
        display_layers: list[_DisplayLayer] = []
        display_counts: dict[str, LayerDisplayCounts] = {
            LAYER_UNRECOVERED: LayerDisplayCounts(len(data.regions), 0)
        }
    else:
        display_layers, display_counts = _assemble_display_layers(
            data,
            allocation,
            las,
            show_original_background=show_original_background,
            show_excluded=show_excluded,
        )

    output_paths = {
        "html": html,
        "summary_json": Path(summary_json) if summary_json is not None else None,
        "review_checklist": (
            Path(review_checklist) if review_checklist is not None else None
        ),
    }

    show_flags = {
        "background": show_original_background,
        "excluded": show_excluded,
        "review": show_review,
    }
    authority = _authority_display(data.configuration_purpose)
    summary = build_summary(
        mode=data.mode,
        source_identity=data.source_identity,
        coverage_report=data.coverage_report,
        regions=data.regions,
        layers=data.layers,
        allocation=allocation,
        display_counts=display_counts,
        identity_checks=data.identity_checks,
        budget=budget,
        show_flags=show_flags,
        output_paths={
            name: path for name, path in output_paths.items() if path is not None
        },
        authority=authority,
    )

    title = (
        f"FRV-to-V4 input visualization — {data.source_identity.source_id} "
        f"({data.mode})"
    )
    html_text = render_html(
        title=title,
        display_layers=display_layers,
        regions=data.regions,
        mode=data.mode,
        authority=authority,
    )

    checklist_text: str | None = None
    if review_checklist is not None:
        checklist_text = build_checklist_markdown(summary=summary)

    _atomic_write_text(html, html_text)
    if summary_json is not None:
        _atomic_write_text(
            Path(summary_json),
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
    if review_checklist is not None and checklist_text is not None:
        _atomic_write_text(Path(review_checklist), checklist_text)

    return FrvV4VisualizationResult(
        mode=data.mode,
        html_path=html,
        summary_path=Path(summary_json) if summary_json is not None else None,
        checklist_path=Path(review_checklist) if review_checklist is not None else None,
        summary=summary,
    )


__all__ = [
    "ALLOCATION_PRIORITY",
    "CHECKLIST_SCHEMA_VERSION",
    "DEFAULT_BACKGROUND_CAP",
    "DEFAULT_LAYER_CAP",
    "DEFAULT_TOTAL_CAP",
    "DisplayAllocation",
    "DisplayBudget",
    "FrvV4BudgetError",
    "FrvV4InputData",
    "FrvV4OutputError",
    "FrvV4RegionRecord",
    "FrvV4ValidationError",
    "FrvV4VisualizationError",
    "FrvV4VisualizationResult",
    "HARD_TOTAL_CAP",
    "LAYER_BACKGROUND",
    "LAYER_EXCLUDED",
    "LAYER_GROUND_OVERLAY",
    "LAYER_HAG_REVIEW",
    "LAYER_MEMBERSHIP_REVIEW",
    "LAYER_NORMAL",
    "LAYER_UNRECOVERED",
    "LAYER_WIRE_OVERLAY",
    "LayerBudgetResult",
    "LayerDisplayCounts",
    "LOW_HAG_DISPLAY_M",
    "MAX_BACKGROUND_CAP",
    "MAX_LAYER_CAP",
    "SUMMARY_SCHEMA_VERSION",
    "TOOL_NAME",
    "allocate_display_budget",
    "build_checklist_markdown",
    "build_summary",
    "load_frv_v4_input",
    "map_layers",
    "read_las_display",
    "render_html",
    "run_frv_v4_visualization",
    "validate_display_budget",
]
