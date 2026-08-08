"""Exact-schema contracts for the Full-Resolution Ground Classification producer.

Authority: docs/full-resolution-ground-classification-design.md (DESIGN-FROZEN)
and docs/full-resolution-ground-classification-implementation-plan.md
(PLAN-FROZEN).  This module implements Bundle A of the frozen plan: immutable
FGCPOL1 policy, terminal routing states, spool/coverage/decision/publication
schemas, deterministic digests, and the completion-manifest discovery root.

Nothing in this module depends on P-HAG numerical configuration; the P-HAG
compatibility digest is recorded only in the publication proof.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from core.contracts._json import reject_unknown_fields, require_fields
from core.contracts.frv import (
    SourceIdentity,
    frv_digest,
)

# ---------------------------------------------------------------------------
# Frozen identifiers and schema versions
# ---------------------------------------------------------------------------

FGC_ALGORITHM_ID = "FGC-PMF-RLP-v1"

FGC_POLICY_SCHEMA_VERSION = "fgc-ground-policy-v1"
FGC_POLICY_DIGEST_DOMAIN = "FGCPOL1"

FGC_SPOOL_SCHEMA_VERSION = "FGCSP1"
FGC_SPOOL_RECORD_FORMAT = "<qiii"
FGC_SPOOL_RECORD_SIZE = 20  # int64 original_index + 3 * int32 raw X/Y/Z
FGC_SPOOL_DIGEST_DOMAIN = "FGCSPL1"
FGC_TILE_LAYOUT_DIGEST_DOMAIN = "FGCTIL1"

FGC_COVERAGE_SCHEMA_VERSION = "FGCCOV1"
FGC_COVERAGE_RECORD_FORMAT = "<qqB"
FGC_COVERAGE_RECORD_SIZE = 17  # int64 ix + int64 iy + uint8 flags
FGC_COVERAGE_FLAG_OBSERVED = 0x01
FGC_COVERAGE_FLAG_CONFIRMED = 0x02
FGC_COVERAGE_SORT_RUN_BYTES = 64 * 1024 * 1024

FGC_DECISION_MANIFEST_SCHEMA_VERSION = "fgc-decision-manifest-v1"
FGC_DECISION_DIGEST_DOMAIN = "FGCDEC1"

FGC_PUBLICATION_PROOF_SCHEMA_VERSION = "fgc-publication-proof-v1"
FGC_PROOF_DIGEST_DOMAIN = "FGCPRF1"

FGC_COMPLETION_MANIFEST_SCHEMA_VERSION = "fgc-completion-manifest-v1"

FGC_DECISION_CHUNK_ROWS_DEFAULT = 1_000_000
FGC_DECISION_MANIFEST_PATH = "fgc_decision_manifest.json"
FGC_PUBLICATION_PROOF_PATH = "fgc_publication_proof.json"
FGC_COMPLETION_MANIFEST_PATH = "fgc_completion_manifest.json"
FGC_CLASSIFIED_DIR = "classified"
FGC_DECISIONS_DIR = "decisions"
FGC_DIAGNOSTICS_DIR = "diagnostics"

# Routing state terminal codes (design section 2).
# 255 is the scratch-only UNROUTED sentinel; it must never be published.
# CONFIRMED_GROUND is the only state permitted to become LAS base class 2.
NON_GROUND = 0
CONFIRMED_GROUND = 1
REVIEW_GROUND = 2
UNROUTED = 255


class GroundDecision(IntEnum):
    """Terminal per-row routing state for FGC-PMF-RLP-v1."""

    NON_GROUND = NON_GROUND
    CONFIRMED_GROUND = CONFIRMED_GROUND
    REVIEW_GROUND = REVIEW_GROUND
    UNROUTED = UNROUTED


class FgcErrorCode:
    """Source-level fail-closed failure codes (design section 16)."""

    UNSUPPORTED_COORDINATE_GRID = "UNSUPPORTED_COORDINATE_GRID"
    NONFINITE_SOURCE = "NONFINITE_SOURCE"
    SOURCE_GRID_ANCHOR_MISMATCH = "SOURCE_GRID_ANCHOR_MISMATCH"
    SPATIAL_SPOOL_RESOURCE_FAILED = "SPATIAL_SPOOL_RESOURCE_FAILED"
    SPATIAL_SPOOL_CORRUPT = "SPATIAL_SPOOL_CORRUPT"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    TILE_OWNERSHIP_FAILED = "TILE_OWNERSHIP_FAILED"
    DECISION_CONSERVATION_FAILED = "DECISION_CONSERVATION_FAILED"
    GROUND_COVERAGE_GAP = "GROUND_COVERAGE_GAP"
    UNSUPPORTED_LAS_METADATA = "UNSUPPORTED_LAS_METADATA"
    PUBLICATION_PRESERVATION_FAILED = "PUBLICATION_PRESERVATION_FAILED"
    CLASSIFIED_REPLAY_MISMATCH = "CLASSIFIED_REPLAY_MISMATCH"
    COMPLETION_MANIFEST_FINALIZATION_FAILED = "COMPLETION_MANIFEST_FINALIZATION_FAILED"
    PUBLICATION_COLLISION = "PUBLICATION_COLLISION"


class FgcDiagnosticReason:
    """Per-row/tile diagnostic reasons that route REVIEW_GROUND (design section 16)."""

    PMF_UNRESOLVED_SUPPORT = "PMF_UNRESOLVED_SUPPORT"
    INSUFFICIENT_SEED_SUPPORT = "INSUFFICIENT_SEED_SUPPORT"
    PLANE_DEGENERATE = "PLANE_DEGENERATE"


# ---------------------------------------------------------------------------
# Numeric validation helpers
# ---------------------------------------------------------------------------


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be a boolean")
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a built-in int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a built-in int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exact_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _sha_or_none(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _ascending_floats(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be an ordered sequence")
    values = tuple(_positive_float(item, name) for item in value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(previous >= current for previous, current in zip(values, values[1:], strict=False)):
        raise ValueError(f"{name} must be strictly ascending")
    return values


def _canonical_payload(value: Mapping[str, Any], *, digest_domain: str) -> dict[str, Any]:
    """Return the canonical digest payload with ``digest_domain`` injected."""
    payload = dict(value)
    payload["digest_domain"] = digest_domain
    return payload


# ---------------------------------------------------------------------------
# FGCPOL1 immutable policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundClassificationPolicyConfig:
    """Immutable exact-schema FGCPOL1 ground policy for FGC-PMF-RLP-v1.

    All values are the frozen reviewed defaults from design section 11.  The
    policy digest is computed with ``canonical_frv_json_bytes`` under FGCPOL1.
    P-HAG numerical configuration is deliberately absent: changing P-HAG
    configuration must never change the FGC policy digest or decisions.
    """

    schema_version: str = FGC_POLICY_SCHEMA_VERSION
    algorithm_id: str = FGC_ALGORITHM_ID
    # Grid geometry (design sections 4-5).
    tile_side_m: float = 256.0
    halo_m: float = 20.0
    fine_cell_m: float = 0.5
    coverage_cell_m: float = 1.0
    # PMF (design section 7).
    pmf_windows_m: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    pmf_minimum_valid_cells: int = 5
    pmf_tolerance_base_m: float = 0.10
    pmf_tolerance_slope: float = 0.75
    pmf_tolerance_cap_m: float = 1.50
    # Seed support (design section 7 and FGC-RR-001).
    seed_radius_m: float = 3.0
    seed_connected_support_cells: int = 5
    seed_lower_envelope_cells: int = 5
    seed_isolated_low_m: float = 0.30
    # Local plane (design section 8).
    plane_radius_m: float = 3.0
    plane_min_supports: int = 12
    plane_max_supports: int = 32
    clip_tolerance_m: float = 0.20
    clip_rounds: int = 2
    singular_ratio: float = 1e-10
    slope_max: float = 0.75
    rms_max_m: float = 0.12
    competing_half_min_supports: int = 6
    competing_disagreement_m: float = 0.35
    # Vertical structure rule (design section 8).
    vertical_min_other_rows: int = 8
    vertical_radius_m: float = 1.0
    vertical_span_m: float = 1.50
    # Routing residual bands (design section 9).
    residual_confirm_min_m: float = -0.15
    residual_confirm_max_m: float = 0.20
    residual_review_min_m: float = -0.35
    residual_review_max_m: float = 0.60
    residual_non_ground_above_m: float = 0.60
    # Coverage gate (design section 10).
    coverage_occupancy_fraction: float = 0.20
    coverage_hole_width_m: float = 25.0
    # Resource bounds (design section 11).
    max_source_points: int = 50_000_000
    max_core_tile_points: int = 8_000_000
    max_halo_tile_points: int = 8_000_000
    max_halo_grid_cells: int = 500_000
    max_plane_batch: int = 65_536
    decision_memmap_bytes: int = 64 * 1024 * 1024
    max_coverage_cells: int = 50_000_000
    decision_chunk_rows: int = FGC_DECISION_CHUNK_ROWS_DEFAULT
    coverage_sort_run_bytes: int = FGC_COVERAGE_SORT_RUN_BYTES
    # Digest (last field; computed).
    policy_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != FGC_POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported FGC policy schema version: {self.schema_version}")
        if self.algorithm_id != FGC_ALGORITHM_ID:
            raise ValueError(f"unsupported FGC algorithm id: {self.algorithm_id}")
        for field_name in (
            "tile_side_m",
            "halo_m",
            "fine_cell_m",
            "coverage_cell_m",
        ):
            object.__setattr__(
                self, field_name, _positive_float(getattr(self, field_name), field_name)
            )
        object.__setattr__(
            self,
            "pmf_windows_m",
            _ascending_floats(self.pmf_windows_m, "pmf_windows_m"),
        )
        object.__setattr__(
            self,
            "pmf_minimum_valid_cells",
            _positive_int(self.pmf_minimum_valid_cells, "pmf_minimum_valid_cells"),
        )
        object.__setattr__(
            self,
            "pmf_tolerance_base_m",
            _non_negative_float(self.pmf_tolerance_base_m, "pmf_tolerance_base_m"),
        )
        object.__setattr__(
            self,
            "pmf_tolerance_slope",
            _non_negative_float(self.pmf_tolerance_slope, "pmf_tolerance_slope"),
        )
        object.__setattr__(
            self,
            "pmf_tolerance_cap_m",
            _positive_float(self.pmf_tolerance_cap_m, "pmf_tolerance_cap_m"),
        )
        for field_name in (
            "seed_radius_m",
            "plane_radius_m",
            "clip_tolerance_m",
            "competing_disagreement_m",
            "vertical_radius_m",
            "vertical_span_m",
            "slope_max",
            "rms_max_m",
            "coverage_hole_width_m",
        ):
            object.__setattr__(
                self, field_name, _positive_float(getattr(self, field_name), field_name)
            )
        for field_name in ("seed_connected_support_cells", "seed_lower_envelope_cells"):
            object.__setattr__(
                self, field_name, _positive_int(getattr(self, field_name), field_name)
            )
        object.__setattr__(
            self,
            "seed_isolated_low_m",
            _positive_float(self.seed_isolated_low_m, "seed_isolated_low_m"),
        )
        object.__setattr__(
            self, "plane_min_supports", _positive_int(self.plane_min_supports, "plane_min_supports")
        )
        object.__setattr__(
            self, "plane_max_supports", _positive_int(self.plane_max_supports, "plane_max_supports")
        )
        if self.plane_max_supports < self.plane_min_supports:
            raise ValueError("plane_max_supports must be >= plane_min_supports")
        object.__setattr__(self, "clip_rounds", _positive_int(self.clip_rounds, "clip_rounds"))
        object.__setattr__(
            self, "singular_ratio", _positive_float(self.singular_ratio, "singular_ratio")
        )
        object.__setattr__(
            self,
            "competing_half_min_supports",
            _positive_int(self.competing_half_min_supports, "competing_half_min_supports"),
        )
        object.__setattr__(
            self,
            "vertical_min_other_rows",
            _positive_int(self.vertical_min_other_rows, "vertical_min_other_rows"),
        )
        object.__setattr__(
            self,
            "residual_confirm_min_m",
            _finite_float(self.residual_confirm_min_m, "residual_confirm_min_m"),
        )
        object.__setattr__(
            self,
            "residual_confirm_max_m",
            _finite_float(self.residual_confirm_max_m, "residual_confirm_max_m"),
        )
        object.__setattr__(
            self,
            "residual_review_min_m",
            _finite_float(self.residual_review_min_m, "residual_review_min_m"),
        )
        object.__setattr__(
            self,
            "residual_review_max_m",
            _finite_float(self.residual_review_max_m, "residual_review_max_m"),
        )
        object.__setattr__(
            self,
            "residual_non_ground_above_m",
            _finite_float(self.residual_non_ground_above_m, "residual_non_ground_above_m"),
        )
        if self.residual_confirm_min_m >= self.residual_confirm_max_m:
            raise ValueError("residual_confirm_min_m must be < residual_confirm_max_m")
        if self.residual_review_min_m >= self.residual_review_max_m:
            raise ValueError("residual_review_min_m must be < residual_review_max_m")
        object.__setattr__(
            self,
            "coverage_occupancy_fraction",
            _finite_float(self.coverage_occupancy_fraction, "coverage_occupancy_fraction"),
        )
        if not 0.0 < self.coverage_occupancy_fraction <= 1.0:
            raise ValueError("coverage_occupancy_fraction must be in (0, 1]")
        for field_name in (
            "max_source_points",
            "max_core_tile_points",
            "max_halo_tile_points",
            "max_halo_grid_cells",
            "max_plane_batch",
            "decision_memmap_bytes",
            "max_coverage_cells",
            "decision_chunk_rows",
            "coverage_sort_run_bytes",
        ):
            object.__setattr__(
                self, field_name, _positive_int(getattr(self, field_name), field_name)
            )
        if self.max_halo_tile_points < self.max_core_tile_points:
            raise ValueError("max_halo_tile_points must be >= max_core_tile_points")
        if self.decision_memmap_bytes < self.max_source_points:
            raise ValueError("decision_memmap_bytes must fit max_source_points uint8 rows")
        if not self.policy_digest:
            object.__setattr__(self, "policy_digest", self.computed_policy_digest())
        elif self.policy_digest != self.computed_policy_digest():
            raise ValueError("policy_digest does not match canonical FGCPOL1 fields")

    # ------------------------------------------------------------------
    # Digest and serialization
    # ------------------------------------------------------------------

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "tile_side_m": self.tile_side_m,
            "halo_m": self.halo_m,
            "fine_cell_m": self.fine_cell_m,
            "coverage_cell_m": self.coverage_cell_m,
            "pmf_windows_m": list(self.pmf_windows_m),
            "pmf_minimum_valid_cells": self.pmf_minimum_valid_cells,
            "pmf_tolerance_base_m": self.pmf_tolerance_base_m,
            "pmf_tolerance_slope": self.pmf_tolerance_slope,
            "pmf_tolerance_cap_m": self.pmf_tolerance_cap_m,
            "seed_radius_m": self.seed_radius_m,
            "seed_connected_support_cells": self.seed_connected_support_cells,
            "seed_lower_envelope_cells": self.seed_lower_envelope_cells,
            "seed_isolated_low_m": self.seed_isolated_low_m,
            "plane_radius_m": self.plane_radius_m,
            "plane_min_supports": self.plane_min_supports,
            "plane_max_supports": self.plane_max_supports,
            "clip_tolerance_m": self.clip_tolerance_m,
            "clip_rounds": self.clip_rounds,
            "singular_ratio": self.singular_ratio,
            "slope_max": self.slope_max,
            "rms_max_m": self.rms_max_m,
            "competing_half_min_supports": self.competing_half_min_supports,
            "competing_disagreement_m": self.competing_disagreement_m,
            "vertical_min_other_rows": self.vertical_min_other_rows,
            "vertical_radius_m": self.vertical_radius_m,
            "vertical_span_m": self.vertical_span_m,
            "residual_confirm_min_m": self.residual_confirm_min_m,
            "residual_confirm_max_m": self.residual_confirm_max_m,
            "residual_review_min_m": self.residual_review_min_m,
            "residual_review_max_m": self.residual_review_max_m,
            "residual_non_ground_above_m": self.residual_non_ground_above_m,
            "coverage_occupancy_fraction": self.coverage_occupancy_fraction,
            "coverage_hole_width_m": self.coverage_hole_width_m,
            "max_source_points": self.max_source_points,
            "max_core_tile_points": self.max_core_tile_points,
            "max_halo_tile_points": self.max_halo_tile_points,
            "max_halo_grid_cells": self.max_halo_grid_cells,
            "max_plane_batch": self.max_plane_batch,
            "decision_memmap_bytes": self.decision_memmap_bytes,
            "max_coverage_cells": self.max_coverage_cells,
            "decision_chunk_rows": self.decision_chunk_rows,
            "coverage_sort_run_bytes": self.coverage_sort_run_bytes,
        }

    def computed_policy_digest(self) -> str:
        return frv_digest(
            _canonical_payload(self.to_digest_dict(), digest_domain=FGC_POLICY_DIGEST_DOMAIN)
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "policy_digest": self.policy_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundClassificationPolicyConfig:
        allowed = set(cls.__dataclass_fields__)
        require_fields(data, {"schema_version", "algorithm_id"}, cls.__name__)
        reject_unknown_fields(data, allowed, cls.__name__)
        values = dict(data)
        values["pmf_windows_m"] = tuple(values.get("pmf_windows_m", (1.0, 2.0, 4.0, 8.0, 16.0)))
        return cls(**values)


def validate_ground_classification_policy(config: GroundClassificationPolicyConfig) -> None:
    if type(config) is not GroundClassificationPolicyConfig:
        raise ValueError("policy must be a GroundClassificationPolicyConfig")
    if config.policy_digest != config.computed_policy_digest():
        raise ValueError("policy_digest does not match canonical FGCPOL1 fields")


# ---------------------------------------------------------------------------
# FGCSP1 owner-spool manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FgcSpoolTileSummary:
    """One nonempty owner spool digest summary (scratch integrity evidence)."""

    tile_id_ty: int
    tile_id_tx: int
    row_count: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_id_ty", _non_negative_int(self.tile_id_ty, "tile_id_ty"))
        object.__setattr__(self, "tile_id_tx", _non_negative_int(self.tile_id_tx, "tile_id_tx"))
        object.__setattr__(self, "row_count", _non_negative_int(self.row_count, "row_count"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))

    @property
    def tile_id(self) -> tuple[int, int]:
        return (self.tile_id_ty, self.tile_id_tx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": [self.tile_id_ty, self.tile_id_tx],
            "row_count": self.row_count,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcSpoolTileSummary:
        allowed = {"tile_id", "row_count", "sha256"}
        reject_unknown_fields(data, allowed, cls.__name__)
        tile_id = data["tile_id"]
        if (
            not isinstance(tile_id, Sequence)
            or isinstance(tile_id, str | bytes | bytearray)
            or len(tile_id) != 2
        ):
            raise ValueError("tile_id must be [ty, tx]")
        return cls(int(tile_id[0]), int(tile_id[1]), int(data["row_count"]), data["sha256"])


def fgc_tile_layout_digest(tile_ids: Sequence[tuple[int, int]], source_point_count: int) -> str:
    """Deterministic digest over the ordered nonempty owner tile layout."""
    ordered = sorted((int(ty), int(tx)) for ty, tx in tile_ids)
    return frv_digest(
        _canonical_payload(
            {
                "tile_ids": [[ty, tx] for ty, tx in ordered],
                "source_point_count": _non_negative_int(source_point_count, "source_point_count"),
            },
            digest_domain=FGC_TILE_LAYOUT_DIGEST_DOMAIN,
        )
    )


@dataclass(frozen=True)
class FgcSpoolManifest:
    """FGCSP1 spool manifest: scratch integrity evidence, not publication authority."""

    schema_version: str = FGC_SPOOL_SCHEMA_VERSION
    source_identity_record_digest: str = ""
    source_point_count: int = 0
    tile_layout_digest: str = ""
    record_size_bytes: int = FGC_SPOOL_RECORD_SIZE
    tiles: tuple[FgcSpoolTileSummary, ...] = ()
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != FGC_SPOOL_SCHEMA_VERSION:
            raise ValueError(f"unsupported spool schema version: {self.schema_version}")
        object.__setattr__(
            self,
            "source_identity_record_digest",
            _sha256(self.source_identity_record_digest, "source_identity_record_digest"),
        )
        object.__setattr__(
            self,
            "source_point_count",
            _non_negative_int(self.source_point_count, "source_point_count"),
        )
        object.__setattr__(
            self, "tile_layout_digest", _sha256(self.tile_layout_digest, "tile_layout_digest")
        )
        if self.record_size_bytes != FGC_SPOOL_RECORD_SIZE:
            raise ValueError(f"spool record size must be {FGC_SPOOL_RECORD_SIZE}")
        tiles = tuple(self.tiles)
        for tile in tiles:
            if type(tile) is not FgcSpoolTileSummary:
                raise ValueError("tiles must contain FgcSpoolTileSummary instances")
        if tiles != tuple(sorted(tiles, key=lambda item: item.tile_id)):
            raise ValueError("spool tiles must be ordered by (ty, tx)")
        object.__setattr__(self, "tiles", tiles)
        if not self.manifest_digest:
            object.__setattr__(self, "manifest_digest", self.computed_manifest_digest())
        elif self.manifest_digest != self.computed_manifest_digest():
            raise ValueError("manifest_digest does not match canonical FGCSP1 fields")

    def computed_manifest_digest(self) -> str:
        return frv_digest(
            _canonical_payload(
                {
                    "source_identity_record_digest": self.source_identity_record_digest,
                    "source_point_count": self.source_point_count,
                    "tile_layout_digest": self.tile_layout_digest,
                    "record_size_bytes": self.record_size_bytes,
                    "tiles": [tile.to_dict() for tile in self.tiles],
                },
                digest_domain=FGC_SPOOL_DIGEST_DOMAIN,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_identity_record_digest": self.source_identity_record_digest,
            "source_point_count": self.source_point_count,
            "tile_layout_digest": self.tile_layout_digest,
            "record_size_bytes": self.record_size_bytes,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "manifest_digest": self.manifest_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcSpoolManifest:
        allowed = set(cls.__dataclass_fields__)
        reject_unknown_fields(data, allowed, cls.__name__)
        tiles = tuple(FgcSpoolTileSummary.from_dict(item) for item in data.get("tiles", ()))
        return cls(
            schema_version=data.get("schema_version", FGC_SPOOL_SCHEMA_VERSION),
            source_identity_record_digest=data["source_identity_record_digest"],
            source_point_count=data["source_point_count"],
            tile_layout_digest=data["tile_layout_digest"],
            record_size_bytes=data.get("record_size_bytes", FGC_SPOOL_RECORD_SIZE),
            tiles=tiles,
            manifest_digest=data["manifest_digest"],
        )


# ---------------------------------------------------------------------------
# FGCCOV1 coverage records and run-union component summaries
# ---------------------------------------------------------------------------


def pack_coverage_record(ix: int, iy: int, flags: int) -> bytes:
    """Encode one fixed little-endian FGCCOV1 record ``<qqB``."""
    ix = int(ix)
    iy = int(iy)
    flags = int(flags)
    if not 0 <= flags <= 255:
        raise ValueError("coverage flags must fit uint8")
    return struct.pack(FGC_COVERAGE_RECORD_FORMAT, ix, iy, flags)


def unpack_coverage_record(record: bytes) -> tuple[int, int, int]:
    if len(record) != FGC_COVERAGE_RECORD_SIZE:
        raise ValueError("coverage record must be exactly 17 bytes")
    ix, iy, flags = struct.unpack(FGC_COVERAGE_RECORD_FORMAT, record)
    return int(ix), int(iy), int(flags)


@dataclass(frozen=True)
class FgcCoverageComponent:
    """A finalized 4-connected unsupported-component summary."""

    min_ix: int
    max_ix: int
    min_iy: int
    max_iy: int
    is_edge: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_ix", int(self.min_ix))
        object.__setattr__(self, "max_ix", int(self.max_ix))
        object.__setattr__(self, "min_iy", int(self.min_iy))
        object.__setattr__(self, "max_iy", int(self.max_iy))
        object.__setattr__(self, "is_edge", bool(self.is_edge))
        if self.min_ix > self.max_ix or self.min_iy > self.max_iy:
            raise ValueError("component bounds are inverted")

    def width_cells(self) -> int:
        return max(self.max_ix - self.min_ix + 1, self.max_iy - self.min_iy + 1)

    def width_m(self, *, coverage_cell_m: float) -> float:
        return self.width_cells() * coverage_cell_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_ix": self.min_ix,
            "max_ix": self.max_ix,
            "min_iy": self.min_iy,
            "max_iy": self.max_iy,
            "width_cells": self.width_cells(),
            "is_edge": self.is_edge,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcCoverageComponent:
        allowed = {"min_ix", "max_ix", "min_iy", "max_iy", "width_cells", "is_edge"}
        reject_unknown_fields(data, allowed, cls.__name__)
        component = cls(
            min_ix=int(data["min_ix"]),
            max_ix=int(data["max_ix"]),
            min_iy=int(data["min_iy"]),
            max_iy=int(data["max_iy"]),
            is_edge=bool(data["is_edge"]),
        )
        # ``to_dict`` emits the derived ``width_cells``; round-trip it only if
        # present and verify it matches the component bounds so a tampered value
        # is never silently accepted.
        if "width_cells" in data:
            declared = int(data["width_cells"])
            if declared != component.width_cells():
                raise ValueError("width_cells does not match the component bounds")
        return component


@dataclass(frozen=True)
class FgcCoverageResult:
    """Bounded sparse global coverage summary (design section 10)."""

    observed_cell_count: int
    confirmed_occupied_cell_count: int
    occupancy_fraction: float
    unsupported_components: tuple[FgcCoverageComponent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_cell_count",
            _non_negative_int(self.observed_cell_count, "observed_cell_count"),
        )
        object.__setattr__(
            self,
            "confirmed_occupied_cell_count",
            _non_negative_int(self.confirmed_occupied_cell_count, "confirmed_occupied_cell_count"),
        )
        object.__setattr__(
            self, "occupancy_fraction", _finite_float(self.occupancy_fraction, "occupancy_fraction")
        )
        if self.confirmed_occupied_cell_count > self.observed_cell_count:
            raise ValueError("confirmed occupied cells cannot exceed observed cells")
        for component in self.unsupported_components:
            if type(component) is not FgcCoverageComponent:
                raise ValueError(
                    "unsupported_components must contain FgcCoverageComponent instances"
                )
        object.__setattr__(self, "unsupported_components", tuple(self.unsupported_components))

    def passes_occupancy(self, *, policy: GroundClassificationPolicyConfig) -> bool:
        if self.observed_cell_count == 0:
            return False
        return self.occupancy_fraction >= policy.coverage_occupancy_fraction

    def interior_overwidth_components(
        self, *, policy: GroundClassificationPolicyConfig
    ) -> tuple[FgcCoverageComponent, ...]:
        return tuple(
            component
            for component in self.unsupported_components
            if not component.is_edge
            and component.width_m(coverage_cell_m=policy.coverage_cell_m)
            > policy.coverage_hole_width_m
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_cell_count": self.observed_cell_count,
            "confirmed_occupied_cell_count": self.confirmed_occupied_cell_count,
            "occupancy_fraction": self.occupancy_fraction,
            "unsupported_components": [
                component.to_dict() for component in self.unsupported_components
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcCoverageResult:
        allowed = {
            "observed_cell_count",
            "confirmed_occupied_cell_count",
            "occupancy_fraction",
            "unsupported_components",
        }
        reject_unknown_fields(data, allowed, cls.__name__)
        components = tuple(
            FgcCoverageComponent.from_dict(item) for item in data.get("unsupported_components", ())
        )
        return cls(
            observed_cell_count=int(data["observed_cell_count"]),
            confirmed_occupied_cell_count=int(data["confirmed_occupied_cell_count"]),
            occupancy_fraction=float(data["occupancy_fraction"]),
            unsupported_components=components,
        )


# ---------------------------------------------------------------------------
# Decision chunks and decision manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FgcDecisionChunkRecord:
    """One published contiguous uint8 decision chunk over an ordered range."""

    start_index: int
    end_index: int
    relative_path: str
    sha256: str
    size_bytes: int
    state_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_index", _non_negative_int(self.start_index, "start_index"))
        object.__setattr__(self, "end_index", _non_negative_int(self.end_index, "end_index"))
        if self.end_index <= self.start_index:
            raise ValueError("decision chunk range must be inclusive-exclusive non-empty")
        object.__setattr__(self, "relative_path", _exact_str(self.relative_path, "relative_path"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(self, "size_bytes", _positive_int(self.size_bytes, "size_bytes"))
        if self.size_bytes != self.end_index - self.start_index:
            raise ValueError("decision chunk size must equal its row count")
        counts = dict(self.state_counts)
        for key, value in counts.items():
            if key not in {"non_ground", "confirmed_ground", "review_ground"}:
                raise ValueError(f"unknown decision state count key: {key}")
            counts[key] = _non_negative_int(value, f"state_counts.{key}")
        if sum(counts.values()) != self.size_bytes:
            raise ValueError("decision chunk state counts must sum to the chunk size")
        object.__setattr__(self, "state_counts", counts)

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "state_counts": dict(self.state_counts),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_digest_dict()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcDecisionChunkRecord:
        allowed = {
            "start_index",
            "end_index",
            "relative_path",
            "sha256",
            "size_bytes",
            "state_counts",
        }
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            start_index=int(data["start_index"]),
            end_index=int(data["end_index"]),
            relative_path=data["relative_path"],
            sha256=data["sha256"],
            size_bytes=int(data["size_bytes"]),
            state_counts=dict(data.get("state_counts", {})),
        )


def validate_ordered_chunk_ranges(
    chunks: Sequence[FgcDecisionChunkRecord], *, source_point_count: int
) -> None:
    """Reject overlapping, gapped, or unordered decision chunk ranges.

    The input order is authoritative: chunks must already be ascending and
    contiguous so that reordering is detectable and never silently repaired.
    """
    source_point_count = _non_negative_int(source_point_count, "source_point_count")
    records = list(chunks)
    cursor = 0
    for chunk in records:
        if chunk.start_index != cursor:
            raise ValueError("decision chunk ranges contain a gap or overlap")
        cursor = chunk.end_index
    if cursor != source_point_count:
        raise ValueError("decision chunk ranges do not cover the full source domain")


def fgc_decision_digest(
    *,
    source_identity: SourceIdentity,
    policy_digest: str,
    source_point_count: int,
    chunks: Sequence[FgcDecisionChunkRecord],
) -> str:
    """Decision digest over identity, policy digest, algorithm, N, and chunks.

    Chunk order is part of the digest: reordering chunks must be detectable.
    The caller is responsible for passing chunks in ordered contiguous ranges;
    an unordered sequence is rejected rather than silently reordered.
    """
    _sha256(policy_digest, "policy_digest")
    validate_ordered_chunk_ranges(chunks, source_point_count=source_point_count)
    payload = {
        "source_identity": source_identity.to_dict(),
        "policy_digest": policy_digest,
        "algorithm_id": FGC_ALGORITHM_ID,
        "source_point_count": source_point_count,
        "chunks": [chunk.to_digest_dict() for chunk in chunks],
    }
    return frv_digest(_canonical_payload(payload, digest_domain=FGC_DECISION_DIGEST_DOMAIN))


@dataclass(frozen=True)
class FgcDecisionManifest:
    """fgc-decision-manifest-v1 artifact binding the ordered decision chunks."""

    schema_version: str = FGC_DECISION_MANIFEST_SCHEMA_VERSION
    algorithm_id: str = FGC_ALGORITHM_ID
    source_identity: Any = None
    raw_source_sha256: str = ""
    source_point_count: int = 0
    policy_schema_version: str = FGC_POLICY_SCHEMA_VERSION
    policy_digest: str = ""
    chunks: tuple[FgcDecisionChunkRecord, ...] = ()
    total_state_counts: Mapping[str, int] = field(default_factory=dict)
    tile_summaries: tuple[FgcSpoolTileSummary, ...] = ()
    coverage: Any = None
    decision_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != FGC_DECISION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported decision manifest schema version")
        if self.algorithm_id != FGC_ALGORITHM_ID:
            raise ValueError("unsupported algorithm id")
        if type(self.source_identity) is not SourceIdentity:
            raise ValueError("source_identity must be a SourceIdentity")
        object.__setattr__(
            self, "raw_source_sha256", _sha256(self.raw_source_sha256, "raw_source_sha256")
        )
        object.__setattr__(
            self,
            "source_point_count",
            _non_negative_int(self.source_point_count, "source_point_count"),
        )
        if self.policy_schema_version != FGC_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported policy schema version")
        object.__setattr__(self, "policy_digest", _sha256(self.policy_digest, "policy_digest"))
        chunks = tuple(self.chunks)
        for chunk in chunks:
            if type(chunk) is not FgcDecisionChunkRecord:
                raise ValueError("chunks must contain FgcDecisionChunkRecord instances")
        validate_ordered_chunk_ranges(chunks, source_point_count=self.source_point_count)
        object.__setattr__(self, "chunks", chunks)
        total = dict(self.total_state_counts)
        for key in ("non_ground", "confirmed_ground", "review_ground"):
            total[key] = _non_negative_int(total.get(key, 0), f"total_state_counts.{key}")
        if sum(total.values()) != self.source_point_count:
            raise ValueError("total state counts must sum to source point count")
        object.__setattr__(self, "total_state_counts", total)
        object.__setattr__(self, "tile_summaries", tuple(self.tile_summaries))
        if self.coverage is not None and type(self.coverage) is not FgcCoverageResult:
            raise ValueError("coverage must be null or a FgcCoverageResult")
        object.__setattr__(
            self, "decision_digest", _sha256(self.decision_digest, "decision_digest")
        )

    def validated_with_policy(
        self, policy: GroundClassificationPolicyConfig
    ) -> FgcDecisionManifest:
        """Return *self* after verifying the decision digest against *policy*."""
        expected = fgc_decision_digest(
            source_identity=self.source_identity,
            policy_digest=policy.policy_digest,
            source_point_count=self.source_point_count,
            chunks=self.chunks,
        )
        if self.decision_digest != expected:
            raise ValueError("decision_digest does not match identity/policy/N/ordered chunks")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "source_identity": self.source_identity.to_dict(),
            "raw_source_sha256": self.raw_source_sha256,
            "source_point_count": self.source_point_count,
            "policy_schema_version": self.policy_schema_version,
            "policy_digest": self.policy_digest,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "total_state_counts": dict(self.total_state_counts),
            "tile_summaries": [tile.to_dict() for tile in self.tile_summaries],
            "coverage": self.coverage.to_dict() if self.coverage is not None else None,
            "decision_digest": self.decision_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcDecisionManifest:
        allowed = set(cls.__dataclass_fields__)
        reject_unknown_fields(data, allowed, cls.__name__)
        chunks = tuple(FgcDecisionChunkRecord.from_dict(item) for item in data.get("chunks", ()))
        tiles = tuple(
            FgcSpoolTileSummary.from_dict(item) for item in data.get("tile_summaries", ())
        )
        coverage = (
            FgcCoverageResult.from_dict(data["coverage"])
            if data.get("coverage") is not None
            else None
        )
        return cls(
            schema_version=data.get("schema_version", FGC_DECISION_MANIFEST_SCHEMA_VERSION),
            algorithm_id=data.get("algorithm_id", FGC_ALGORITHM_ID),
            source_identity=SourceIdentity.from_dict(data["source_identity"]),
            raw_source_sha256=data["raw_source_sha256"],
            source_point_count=data["source_point_count"],
            policy_schema_version=data.get("policy_schema_version", FGC_POLICY_SCHEMA_VERSION),
            policy_digest=data["policy_digest"],
            chunks=chunks,
            total_state_counts=data.get("total_state_counts", {}),
            tile_summaries=tiles,
            coverage=coverage,
            decision_digest=data["decision_digest"],
        )


# ---------------------------------------------------------------------------
# fgc-publication-proof-v1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FgcPublicationProof:
    """fgc-publication-proof-v1 binding lineage and P-HAG compatibility."""

    schema_version: str = FGC_PUBLICATION_PROOF_SCHEMA_VERSION
    algorithm_id: str = FGC_ALGORITHM_ID
    raw_source_identity: Any = None
    raw_source_sha256: str = ""
    source_point_count: int = 0
    raw_original_index_equals_derivative: bool = True
    policy_schema_version: str = FGC_POLICY_SCHEMA_VERSION
    policy_digest: str = ""
    decision_digest: str = ""
    decision_manifest_digest: str = ""
    classification_mapping_counts: Mapping[str, int] = field(default_factory=dict)
    classified_source_identity: Any = None
    classified_source_sha256: str = ""
    p_source_replay: Mapping[str, Any] = field(default_factory=dict)
    p_hag_compatibility: Mapping[str, Any] = field(default_factory=dict)
    proof_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != FGC_PUBLICATION_PROOF_SCHEMA_VERSION:
            raise ValueError("unsupported publication proof schema version")
        if self.algorithm_id != FGC_ALGORITHM_ID:
            raise ValueError("unsupported algorithm id")
        if type(self.raw_source_identity) is not SourceIdentity:
            raise ValueError("raw_source_identity must be a SourceIdentity")
        object.__setattr__(
            self, "raw_source_sha256", _sha256(self.raw_source_sha256, "raw_source_sha256")
        )
        object.__setattr__(
            self,
            "source_point_count",
            _non_negative_int(self.source_point_count, "source_point_count"),
        )
        if type(self.raw_original_index_equals_derivative) is not bool:
            raise ValueError("raw_original_index_equals_derivative must be a bool")
        object.__setattr__(self, "policy_digest", _sha256(self.policy_digest, "policy_digest"))
        object.__setattr__(
            self, "decision_digest", _sha256(self.decision_digest, "decision_digest")
        )
        object.__setattr__(
            self,
            "decision_manifest_digest",
            _sha256(self.decision_manifest_digest, "decision_manifest_digest"),
        )
        if type(self.classified_source_identity) is not SourceIdentity:
            raise ValueError("classified_source_identity must be a SourceIdentity")
        object.__setattr__(
            self,
            "classified_source_sha256",
            _sha256(self.classified_source_sha256, "classified_source_sha256"),
        )
        object.__setattr__(
            self,
            "classification_mapping_counts",
            json_safe_mapping_type(
                self.classification_mapping_counts, "classification_mapping_counts"
            ),
        )
        object.__setattr__(
            self, "p_source_replay", json_safe_mapping_type(self.p_source_replay, "p_source_replay")
        )
        object.__setattr__(
            self,
            "p_hag_compatibility",
            json_safe_mapping_type(self.p_hag_compatibility, "p_hag_compatibility"),
        )
        if not self.proof_digest:
            object.__setattr__(self, "proof_digest", self.computed_proof_digest())
        elif self.proof_digest != self.computed_proof_digest():
            raise ValueError("proof_digest does not match canonical proof fields")

    def computed_proof_digest(self) -> str:
        payload = self.to_digest_dict()
        return frv_digest(_canonical_payload(payload, digest_domain=FGC_PROOF_DIGEST_DOMAIN))

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "raw_source_identity": self.raw_source_identity.to_dict(),
            "raw_source_sha256": self.raw_source_sha256,
            "source_point_count": self.source_point_count,
            "raw_original_index_equals_derivative": self.raw_original_index_equals_derivative,
            "policy_schema_version": self.policy_schema_version,
            "policy_digest": self.policy_digest,
            "decision_digest": self.decision_digest,
            "decision_manifest_digest": self.decision_manifest_digest,
            "classification_mapping_counts": dict(self.classification_mapping_counts),
            "classified_source_identity": self.classified_source_identity.to_dict(),
            "classified_source_sha256": self.classified_source_sha256,
            "p_source_replay": dict(self.p_source_replay),
            "p_hag_compatibility": dict(self.p_hag_compatibility),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "proof_digest": self.proof_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcPublicationProof:
        allowed = set(cls.__dataclass_fields__)
        reject_unknown_fields(data, allowed, cls.__name__)
        return cls(
            schema_version=data.get("schema_version", FGC_PUBLICATION_PROOF_SCHEMA_VERSION),
            algorithm_id=data.get("algorithm_id", FGC_ALGORITHM_ID),
            raw_source_identity=SourceIdentity.from_dict(data["raw_source_identity"]),
            raw_source_sha256=data["raw_source_sha256"],
            source_point_count=data["source_point_count"],
            raw_original_index_equals_derivative=data.get(
                "raw_original_index_equals_derivative", True
            ),
            policy_schema_version=data.get("policy_schema_version", FGC_POLICY_SCHEMA_VERSION),
            policy_digest=data["policy_digest"],
            decision_digest=data["decision_digest"],
            decision_manifest_digest=data["decision_manifest_digest"],
            classification_mapping_counts=data.get("classification_mapping_counts", {}),
            classified_source_identity=SourceIdentity.from_dict(data["classified_source_identity"]),
            classified_source_sha256=data["classified_source_sha256"],
            p_source_replay=data.get("p_source_replay", {}),
            p_hag_compatibility=data.get("p_hag_compatibility", {}),
            proof_digest=data["proof_digest"],
        )


def json_safe_mapping_type(value: Any, name: str) -> dict[str, Any]:
    from core.contracts._json import json_safe_mapping

    return json_safe_mapping(value, name)


# ---------------------------------------------------------------------------
# fgc-completion-manifest-v1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FgcCompletionManifest:
    """The sole formal discovery root binding all published FGC components.

    Consumers must exact-schema validate, reject unsafe paths, hash/size-check
    components, validate the decision manifest and all chunks, validate the
    proof, validate the classified SourceIdentity, then verify all cross-digest
    bindings.  Without a valid completion manifest the directory is
    INCOMPLETE_NONAUTHORITATIVE.
    """

    schema_version: str = FGC_COMPLETION_MANIFEST_SCHEMA_VERSION
    algorithm_id: str = FGC_ALGORITHM_ID
    policy_schema_version: str = FGC_POLICY_SCHEMA_VERSION
    policy_digest: str = ""
    raw_source_identity: Any = None
    classified_source_identity: Any = None
    decision_manifest_path: str = FGC_DECISION_MANIFEST_PATH
    decision_manifest_sha256: str = ""
    decision_manifest_size_bytes: int = 0
    decision_manifest_digest: str = ""
    publication_proof_path: str = FGC_PUBLICATION_PROOF_PATH
    publication_proof_sha256: str = ""
    publication_proof_size_bytes: int = 0
    publication_proof_digest: str = ""
    classified_las_path: str = ""
    classified_las_sha256: str = ""
    classified_las_size_bytes: int = 0
    classified_las_point_count: int = 0
    decision_chunks: tuple[FgcDecisionChunkRecord, ...] = ()
    completion_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != FGC_COMPLETION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported completion manifest schema version")
        if self.algorithm_id != FGC_ALGORITHM_ID:
            raise ValueError("unsupported algorithm id")
        if self.policy_schema_version != FGC_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported policy schema version")
        object.__setattr__(self, "policy_digest", _sha256(self.policy_digest, "policy_digest"))
        if type(self.raw_source_identity) is not SourceIdentity:
            raise ValueError("raw_source_identity must be a SourceIdentity")
        if type(self.classified_source_identity) is not SourceIdentity:
            raise ValueError("classified_source_identity must be a SourceIdentity")
        for field_name, value in (
            ("decision_manifest_path", self.decision_manifest_path),
            ("publication_proof_path", self.publication_proof_path),
            ("classified_las_path", self.classified_las_path),
        ):
            _require_safe_relative_path(value, field_name)
        object.__setattr__(
            self,
            "decision_manifest_sha256",
            _sha256(self.decision_manifest_sha256, "decision_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "decision_manifest_size_bytes",
            _non_negative_int(self.decision_manifest_size_bytes, "decision_manifest_size_bytes"),
        )
        object.__setattr__(
            self,
            "decision_manifest_digest",
            _sha256(self.decision_manifest_digest, "decision_manifest_digest"),
        )
        object.__setattr__(
            self,
            "publication_proof_sha256",
            _sha256(self.publication_proof_sha256, "publication_proof_sha256"),
        )
        object.__setattr__(
            self,
            "publication_proof_size_bytes",
            _non_negative_int(self.publication_proof_size_bytes, "publication_proof_size_bytes"),
        )
        object.__setattr__(
            self,
            "publication_proof_digest",
            _sha256(self.publication_proof_digest, "publication_proof_digest"),
        )
        object.__setattr__(
            self,
            "classified_las_sha256",
            _sha256(self.classified_las_sha256, "classified_las_sha256"),
        )
        object.__setattr__(
            self,
            "classified_las_size_bytes",
            _positive_int(self.classified_las_size_bytes, "classified_las_size_bytes"),
        )
        object.__setattr__(
            self,
            "classified_las_point_count",
            _non_negative_int(self.classified_las_point_count, "classified_las_point_count"),
        )
        object.__setattr__(self, "decision_chunks", tuple(self.decision_chunks))
        if not self.completion_digest:
            object.__setattr__(self, "completion_digest", self.computed_completion_digest())
        elif self.completion_digest != self.computed_completion_digest():
            raise ValueError("completion_digest does not match the canonical completion payload")

    def computed_completion_digest(self) -> str:
        payload = self.to_digest_dict()
        return frv_digest(payload)

    def verify_bindings(self, root: Path) -> None:
        """Consumer-side cross-digest binding check against a publication root.

        Hash/size-checks every bound component and confirms the completion
        digest equals the canonical payload digest.  Unsafe paths were already
        rejected during construction.
        """
        if self.completion_digest != self.computed_completion_digest():
            raise ValueError("completion_digest does not match the canonical completion payload")
        for field_name, path_value, expected_sha, expected_size in (
            (
                "decision_manifest",
                self.decision_manifest_path,
                self.decision_manifest_sha256,
                self.decision_manifest_size_bytes,
            ),
            (
                "publication_proof",
                self.publication_proof_path,
                self.publication_proof_sha256,
                self.publication_proof_size_bytes,
            ),
            (
                "classified_las",
                self.classified_las_path,
                self.classified_las_sha256,
                self.classified_las_size_bytes,
            ),
        ):
            component = (root / path_value).resolve()
            if not component.is_file():
                raise ValueError(f"{field_name} component is missing at {path_value}")
            if component.stat().st_size != expected_size:
                raise ValueError(f"{field_name} component size does not match the manifest")
            if _file_sha256(component) != expected_sha:
                raise ValueError(f"{field_name} component SHA-256 does not match the manifest")
        for chunk in self.decision_chunks:
            component = (root / FGC_DECISIONS_DIR / chunk.relative_path).resolve()
            if not component.is_file():
                raise ValueError(f"decision chunk is missing at {chunk.relative_path}")
            if component.stat().st_size != chunk.size_bytes:
                raise ValueError(f"decision chunk size does not match: {chunk.relative_path}")
            if _file_sha256(component) != chunk.sha256:
                raise ValueError(f"decision chunk SHA-256 does not match: {chunk.relative_path}")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "policy_schema_version": self.policy_schema_version,
            "policy_digest": self.policy_digest,
            "raw_source_identity": self.raw_source_identity.to_dict(),
            "classified_source_identity": self.classified_source_identity.to_dict(),
            "decision_manifest_path": self.decision_manifest_path,
            "decision_manifest_sha256": self.decision_manifest_sha256,
            "decision_manifest_size_bytes": self.decision_manifest_size_bytes,
            "decision_manifest_digest": self.decision_manifest_digest,
            "publication_proof_path": self.publication_proof_path,
            "publication_proof_sha256": self.publication_proof_sha256,
            "publication_proof_size_bytes": self.publication_proof_size_bytes,
            "publication_proof_digest": self.publication_proof_digest,
            "classified_las_path": self.classified_las_path,
            "classified_las_sha256": self.classified_las_sha256,
            "classified_las_size_bytes": self.classified_las_size_bytes,
            "classified_las_point_count": self.classified_las_point_count,
            "decision_chunks": [chunk.to_digest_dict() for chunk in self.decision_chunks],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "completion_digest": self.completion_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FgcCompletionManifest:
        allowed = set(cls.__dataclass_fields__)
        reject_unknown_fields(data, allowed, cls.__name__)
        chunks = tuple(
            FgcDecisionChunkRecord.from_dict(item) for item in data.get("decision_chunks", ())
        )
        return cls(
            schema_version=data.get("schema_version", FGC_COMPLETION_MANIFEST_SCHEMA_VERSION),
            algorithm_id=data.get("algorithm_id", FGC_ALGORITHM_ID),
            policy_schema_version=data.get("policy_schema_version", FGC_POLICY_SCHEMA_VERSION),
            policy_digest=data["policy_digest"],
            raw_source_identity=SourceIdentity.from_dict(data["raw_source_identity"]),
            classified_source_identity=SourceIdentity.from_dict(data["classified_source_identity"]),
            decision_manifest_path=data.get("decision_manifest_path", FGC_DECISION_MANIFEST_PATH),
            decision_manifest_sha256=data["decision_manifest_sha256"],
            decision_manifest_size_bytes=data["decision_manifest_size_bytes"],
            decision_manifest_digest=data["decision_manifest_digest"],
            publication_proof_path=data.get("publication_proof_path", FGC_PUBLICATION_PROOF_PATH),
            publication_proof_sha256=data["publication_proof_sha256"],
            publication_proof_size_bytes=data["publication_proof_size_bytes"],
            publication_proof_digest=data["publication_proof_digest"],
            classified_las_path=data["classified_las_path"],
            classified_las_sha256=data["classified_las_sha256"],
            classified_las_size_bytes=data["classified_las_size_bytes"],
            classified_las_point_count=data["classified_las_point_count"],
            decision_chunks=chunks,
            completion_digest=data["completion_digest"],
        )


def _require_safe_relative_path(value: Any, name: str) -> None:
    """Reject absolute paths, traversal, and empty paths."""
    text = _exact_str(value, name)
    path = Path(text)
    if path.is_absolute():
        raise ValueError(f"{name} must be a relative path")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} must be a safe relative path without traversal")


# ---------------------------------------------------------------------------
# Resource preflight (design section 11)
# ---------------------------------------------------------------------------


def fgc_working_set_bytes(policy: GroundClassificationPolicyConfig) -> int:
    """Compute the tile-algorithm working-set bound from the frozen caps."""
    return (
        256 * policy.max_halo_tile_points
        + 128 * policy.max_halo_grid_cells
        + 128 * policy.max_plane_batch
        + 96 * 1024 * 1024  # source/chunk and routing workspace
        + 256 * 1024 * 1024  # fixed safety reserve
    )


def fgc_free_space_requirement_bytes(
    input_file_bytes: int, source_point_count: int, policy: GroundClassificationPolicyConfig
) -> int:
    """Same-volume free-space preflight (design section 11)."""
    n = _non_negative_int(source_point_count, "source_point_count")
    _non_negative_int(input_file_bytes, "input_file_bytes")
    base = (
        2 * input_file_bytes
        + 20 * n  # owner spools
        + 2 * n  # routing memmap plus final chunks
        + 51 * n  # FGCCOV1 external-sort worst case
        + 128 * 1024 * 1024  # manifests, bounded diagnostics, margin
    )
    return math.ceil(1.15 * base)


# ---------------------------------------------------------------------------
# Classification mapping (design sections 2 and 18)
# ---------------------------------------------------------------------------


def map_derivative_class(original_base_class: int, decision: GroundDecision) -> int:
    """Map one raw logical base classification to the derivative value.

    CONFIRMED_GROUND always becomes LAS class 2.  Every other decision preserves
    the input base class except that an existing class 2 becomes class 1
    (unclassified).  All other base classes are preserved.
    """
    original = int(original_base_class)
    if not 0 <= original <= 255:
        raise ValueError("original base class must fit uint8")
    if decision == GroundDecision.CONFIRMED_GROUND:
        return 2
    if original == 2:
        return 1
    return original


def _file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of a file's bytes (streamed)."""
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()
