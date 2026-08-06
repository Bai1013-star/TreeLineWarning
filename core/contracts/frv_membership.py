"""FRV full-resolution membership and scope-routing contracts (PLAN-FROZEN).

This module freezes the scope-evidence reference, the scope-policy and
membership-policy configurations, the execution state/result contracts, the
decision-evidence sidecar artifact, and their pure validators.  It performs no
file access, no scope construction, no routing, and no membership decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from core.contracts.frv import (
    FRVArtifactBindingRecord,
    FRVArtifactPayloadSchema,
    FRVScopeCoverageManifest,
    FullResolutionVegetationMembershipArtifact,
    SourceIdentity,
    frv_digest,
    validate_artifact_binding_record,
    validate_artifact_payload_digest_domain,
    validate_payload_schema,
    validate_scope_inventory_conservation,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
    FrvA2ReadinessRecord,
    validate_frv_a2_artifact_component_binding,
    validate_frv_a2_failure_record,
)
from core.contracts.frv_source import (
    FrvSourceProviderResult,
    validate_finalized_frv_source_provider_result,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

FRV_SCOPE_POLICY_SCHEMA_VERSION = "frv-scope-policy-v1"
FRV_SCOPE_POLICY_DIGEST_DOMAIN = "FRVSCPPOL1"
FRV_SCOPE_GEOMETRY_DIGEST_DOMAIN = "FRVSCPGEO1"
FRV_MEMBERSHIP_POLICY_SCHEMA_VERSION = "frv-membership-policy-v1"
FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN = "FRVMEMPOL1"
FRV_MEMBERSHIP_RESULT_DIGEST_DOMAIN = "FRVMEMRES1"
FRV_DECISION_EVIDENCE_SCHEMA_VERSION = "frv-decision-evidence-v1"
FRV_DECISION_EVIDENCE_DIGEST_DOMAIN = "FRVEVD1"


def frv_source_proof_digest(source_result: FrvSourceProviderResult) -> str:
    """Canonical P-SOURCE proof fingerprint (DC-001-SOURCE-PROOF-REBIND).

    Validates that ``source_result`` is a finalized, structurally valid P-SOURCE
    result, then returns the canonical digest of its success proof.  Non-finalized,
    fabricated or structurally invalid results are rejected before any fingerprint
    is returned, so a binding made from this helper can be re-verified later by
    replaying the source and recomputing the same fingerprint.
    """

    validate_finalized_frv_source_provider_result(source_result)
    return frv_digest(source_result.proof.to_dict())


# Frozen default scope-policy values.
FRV_SCOPE_DEFAULT_VOXEL_SIZE_M = 0.05
FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_XY_CELLS = 30
FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_Z_CELLS = 30
FRV_SCOPE_DEFAULT_TILE_EDGE_CELLS = 60
FRV_SCOPE_DEFAULT_FEATURE_HALO_CELLS = 30
FRV_SCOPE_DEFAULT_BOUNDARY_RULE = "min-inclusive-max-exclusive"
FRV_SCOPE_DEFAULT_COORDINATE_CANONICALIZATION = "binary64-subtract-origin-once"

# Frozen default membership-policy values.
FRV_MEMBERSHIP_DEFAULT_ORDINARY_MIN_HAG_M = 0.30
FRV_MEMBERSHIP_DEFAULT_UPPER_SEED_MIN_HAG_M = 1.00
FRV_MEMBERSHIP_DEFAULT_GROUND_MAX_ABS_RAW_HAG_M = 0.10
FRV_MEMBERSHIP_DEFAULT_PROPAGATION_MIN_HAG_DROP_M = 0.05
FRV_MEMBERSHIP_DEFAULT_MAX_STRUCTURAL_ROUNDS = 12
FRV_MEMBERSHIP_DEFAULT_GROUND_CLASSIFICATION_VALUE = 2
FRV_MEMBERSHIP_DEFAULT_OUTPUT_BLOCK_RECORD_COUNT = 250000


# ---------------------------------------------------------------------------
# Private strict validation helpers
# ---------------------------------------------------------------------------


def _require_exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


def _require_sha(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a positive built-in int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a nonnegative built-in int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


def _require_enum(value: Any, enum_type: type, name: str) -> Any:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _exact_fields(data: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(data)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return dict(data)


def _tuple_from_json(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def _enum_from_value(value: Any, enum_type: type, name: str) -> Any:
    text = _require_exact_str(value, name)
    try:
        return enum_type(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unsupported") from exc


# ---------------------------------------------------------------------------
# FrvScopeEvidenceReference (non-serializable runtime object)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvScopeEvidenceReference:
    """Runtime-only binding of the A1/V1/V2 pair paths and exact components."""

    a1_npz_path: Path
    a1_json_path: Path
    v1_npz_path: Path
    v1_json_path: Path
    v2_npz_path: Path
    v2_json_path: Path
    a1_npz_component: FrvA2ArtifactComponentBinding
    a1_json_component: FrvA2ArtifactComponentBinding
    v1_npz_component: FrvA2ArtifactComponentBinding
    v1_json_component: FrvA2ArtifactComponentBinding
    v2_npz_component: FrvA2ArtifactComponentBinding
    v2_json_component: FrvA2ArtifactComponentBinding

    def __post_init__(self) -> None:
        for name in (
            "a1_npz_path",
            "a1_json_path",
            "v1_npz_path",
            "v1_json_path",
            "v2_npz_path",
            "v2_json_path",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a pathlib.Path instance")
        for name in (
            "a1_npz_component",
            "a1_json_component",
            "v1_npz_component",
            "v1_json_component",
            "v2_npz_component",
            "v2_json_component",
        ):
            value = getattr(self, name)
            if type(value) is not FrvA2ArtifactComponentBinding:
                raise ValueError(f"{name} must be a FrvA2ArtifactComponentBinding")


def validate_frv_scope_evidence_reference(reference: FrvScopeEvidenceReference) -> None:
    if type(reference) is not FrvScopeEvidenceReference:
        raise ValueError("reference must be a FrvScopeEvidenceReference")
    for name in (
        "a1_npz_component",
        "a1_json_component",
        "v1_npz_component",
        "v1_json_component",
        "v2_npz_component",
        "v2_json_component",
    ):
        validate_frv_a2_artifact_component_binding(getattr(reference, name))


# ---------------------------------------------------------------------------
# FrvScopePolicyConfig
# ---------------------------------------------------------------------------


def derive_frv_scope_policy_digest(
    *,
    schema_version: str,
    voxel_size_m: float,
    context_margin_xy_cells: int,
    context_margin_z_cells: int,
    tile_edge_cells: int,
    feature_halo_cells: int,
    boundary_rule: str,
    coordinate_canonicalization: str,
    geometry_digest_domain: str,
) -> str:
    """Return the FRVSCPPOL1 digest for the frozen scope-policy payload."""

    payload = {
        "digest_domain": FRV_SCOPE_POLICY_DIGEST_DOMAIN,
        "schema_version": schema_version,
        "voxel_size_m": voxel_size_m,
        "context_margin_xy_cells": context_margin_xy_cells,
        "context_margin_z_cells": context_margin_z_cells,
        "tile_edge_cells": tile_edge_cells,
        "feature_halo_cells": feature_halo_cells,
        "boundary_rule": boundary_rule,
        "coordinate_canonicalization": coordinate_canonicalization,
        "geometry_digest_domain": geometry_digest_domain,
    }
    return frv_digest(payload)


@dataclass(frozen=True, slots=True)
class FrvScopePolicyConfig:
    """Exact frozen scope-policy configuration (digest domain FRVSCPPOL1)."""

    schema_version: str
    voxel_size_m: float
    context_margin_xy_cells: int
    context_margin_z_cells: int
    tile_edge_cells: int
    feature_halo_cells: int
    boundary_rule: str
    coordinate_canonicalization: str
    geometry_digest_domain: str
    policy_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_SCOPE_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported scope policy schema version")
        object.__setattr__(
            self, "voxel_size_m", _require_finite_float(self.voxel_size_m, "voxel_size_m")
        )
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")
        for name in (
            "context_margin_xy_cells",
            "context_margin_z_cells",
            "tile_edge_cells",
            "feature_halo_cells",
        ):
            object.__setattr__(
                self, name, _require_positive_int(getattr(self, name), name)
            )
        _require_exact_str(self.boundary_rule, "boundary_rule")
        if self.boundary_rule != "min-inclusive-max-exclusive":
            raise ValueError("boundary_rule must be min-inclusive-max-exclusive")
        _require_exact_str(
            self.coordinate_canonicalization, "coordinate_canonicalization"
        )
        if self.coordinate_canonicalization != "binary64-subtract-origin-once":
            raise ValueError(
                "coordinate_canonicalization must be binary64-subtract-origin-once"
            )
        _require_exact_str(self.geometry_digest_domain, "geometry_digest_domain")
        if self.geometry_digest_domain != FRV_SCOPE_GEOMETRY_DIGEST_DOMAIN:
            raise ValueError("geometry_digest_domain must be FRVSCPGEO1")
        _require_sha(self.policy_digest, "policy_digest")
        computed = derive_frv_scope_policy_digest(
            schema_version=self.schema_version,
            voxel_size_m=self.voxel_size_m,
            context_margin_xy_cells=self.context_margin_xy_cells,
            context_margin_z_cells=self.context_margin_z_cells,
            tile_edge_cells=self.tile_edge_cells,
            feature_halo_cells=self.feature_halo_cells,
            boundary_rule=self.boundary_rule,
            coordinate_canonicalization=self.coordinate_canonicalization,
            geometry_digest_domain=self.geometry_digest_domain,
        )
        if self.policy_digest != computed:
            raise ValueError("policy_digest does not match canonical FRVSCPPOL1 fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "voxel_size_m": self.voxel_size_m,
            "context_margin_xy_cells": self.context_margin_xy_cells,
            "context_margin_z_cells": self.context_margin_z_cells,
            "tile_edge_cells": self.tile_edge_cells,
            "feature_halo_cells": self.feature_halo_cells,
            "boundary_rule": self.boundary_rule,
            "coordinate_canonicalization": self.coordinate_canonicalization,
            "geometry_digest_domain": self.geometry_digest_domain,
            "policy_digest": self.policy_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvScopePolicyConfig:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**values)


def default_frv_scope_policy_config() -> FrvScopePolicyConfig:
    policy_digest = derive_frv_scope_policy_digest(
        schema_version=FRV_SCOPE_POLICY_SCHEMA_VERSION,
        voxel_size_m=FRV_SCOPE_DEFAULT_VOXEL_SIZE_M,
        context_margin_xy_cells=FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_XY_CELLS,
        context_margin_z_cells=FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_Z_CELLS,
        tile_edge_cells=FRV_SCOPE_DEFAULT_TILE_EDGE_CELLS,
        feature_halo_cells=FRV_SCOPE_DEFAULT_FEATURE_HALO_CELLS,
        boundary_rule=FRV_SCOPE_DEFAULT_BOUNDARY_RULE,
        coordinate_canonicalization=FRV_SCOPE_DEFAULT_COORDINATE_CANONICALIZATION,
        geometry_digest_domain=FRV_SCOPE_GEOMETRY_DIGEST_DOMAIN,
    )
    return FrvScopePolicyConfig(
        schema_version=FRV_SCOPE_POLICY_SCHEMA_VERSION,
        voxel_size_m=FRV_SCOPE_DEFAULT_VOXEL_SIZE_M,
        context_margin_xy_cells=FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_XY_CELLS,
        context_margin_z_cells=FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_Z_CELLS,
        tile_edge_cells=FRV_SCOPE_DEFAULT_TILE_EDGE_CELLS,
        feature_halo_cells=FRV_SCOPE_DEFAULT_FEATURE_HALO_CELLS,
        boundary_rule=FRV_SCOPE_DEFAULT_BOUNDARY_RULE,
        coordinate_canonicalization=FRV_SCOPE_DEFAULT_COORDINATE_CANONICALIZATION,
        geometry_digest_domain=FRV_SCOPE_GEOMETRY_DIGEST_DOMAIN,
        policy_digest=policy_digest,
    )


def validate_frv_scope_policy_config(config: FrvScopePolicyConfig) -> None:
    if type(config) is not FrvScopePolicyConfig:
        raise ValueError("config must be a FrvScopePolicyConfig")
    computed = derive_frv_scope_policy_digest(
        schema_version=config.schema_version,
        voxel_size_m=config.voxel_size_m,
        context_margin_xy_cells=config.context_margin_xy_cells,
        context_margin_z_cells=config.context_margin_z_cells,
        tile_edge_cells=config.tile_edge_cells,
        feature_halo_cells=config.feature_halo_cells,
        boundary_rule=config.boundary_rule,
        coordinate_canonicalization=config.coordinate_canonicalization,
        geometry_digest_domain=config.geometry_digest_domain,
    )
    if config.policy_digest != computed:
        raise ValueError("policy_digest does not match canonical FRVSCPPOL1 fields")


# ---------------------------------------------------------------------------
# FrvMembershipPolicyConfig
# ---------------------------------------------------------------------------


def derive_frv_membership_policy_digest(
    *,
    schema_version: str,
    algorithm_version: str,
    decision_rule_version: str,
    ordinary_min_hag_m: float,
    upper_seed_min_hag_m: float,
    ground_max_abs_raw_hag_m: float,
    propagation_min_hag_drop_m: float,
    max_structural_rounds: int,
    ground_classification_value: int,
    output_block_record_count: int,
    policy_digest_domain: str,
) -> str:
    """Return the FRVMEMPOL1 digest for the frozen membership-policy payload."""

    payload = {
        "digest_domain": FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN,
        "schema_version": schema_version,
        "algorithm_version": algorithm_version,
        "decision_rule_version": decision_rule_version,
        "ordinary_min_hag_m": ordinary_min_hag_m,
        "upper_seed_min_hag_m": upper_seed_min_hag_m,
        "ground_max_abs_raw_hag_m": ground_max_abs_raw_hag_m,
        "propagation_min_hag_drop_m": propagation_min_hag_drop_m,
        "max_structural_rounds": max_structural_rounds,
        "ground_classification_value": ground_classification_value,
        "output_block_record_count": output_block_record_count,
        "policy_digest_domain": policy_digest_domain,
    }
    return frv_digest(payload)


@dataclass(frozen=True, slots=True)
class FrvMembershipPolicyConfig:
    """Exact frozen membership-policy configuration (domain FRVMEMPOL1)."""

    schema_version: str
    algorithm_version: str
    decision_rule_version: str
    ordinary_min_hag_m: float
    upper_seed_min_hag_m: float
    ground_max_abs_raw_hag_m: float
    propagation_min_hag_drop_m: float
    max_structural_rounds: int
    ground_classification_value: int
    output_block_record_count: int
    policy_digest_domain: str
    policy_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_MEMBERSHIP_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported membership policy schema version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        if self.algorithm_version != "frv-membership-synchronous-v1":
            raise ValueError("algorithm_version must be frv-membership-synchronous-v1")
        _require_exact_str(self.decision_rule_version, "decision_rule_version")
        if self.decision_rule_version != "frv-membership-decision-v1":
            raise ValueError("decision_rule_version must be frv-membership-decision-v1")
        for name in (
            "ordinary_min_hag_m",
            "upper_seed_min_hag_m",
            "ground_max_abs_raw_hag_m",
            "propagation_min_hag_drop_m",
        ):
            object.__setattr__(
                self, name, _require_finite_float(getattr(self, name), name)
            )
        if self.ordinary_min_hag_m <= 0.0:
            raise ValueError("ordinary_min_hag_m must be positive")
        if not (self.ordinary_min_hag_m < self.upper_seed_min_hag_m):
            raise ValueError("upper_seed_min_hag_m must exceed ordinary_min_hag_m")
        if self.ground_max_abs_raw_hag_m <= 0.0:
            raise ValueError("ground_max_abs_raw_hag_m must be positive")
        if self.propagation_min_hag_drop_m <= 0.0:
            raise ValueError("propagation_min_hag_drop_m must be positive")
        object.__setattr__(
            self,
            "max_structural_rounds",
            _require_positive_int(self.max_structural_rounds, "max_structural_rounds"),
        )
        object.__setattr__(
            self,
            "ground_classification_value",
            _require_nonnegative_int(
                self.ground_classification_value, "ground_classification_value"
            ),
        )
        object.__setattr__(
            self,
            "output_block_record_count",
            _require_positive_int(
                self.output_block_record_count, "output_block_record_count"
            ),
        )
        _require_exact_str(self.policy_digest_domain, "policy_digest_domain")
        if self.policy_digest_domain != FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN:
            raise ValueError("policy_digest_domain must be FRVMEMPOL1")
        _require_sha(self.policy_digest, "policy_digest")
        computed = derive_frv_membership_policy_digest(
            schema_version=self.schema_version,
            algorithm_version=self.algorithm_version,
            decision_rule_version=self.decision_rule_version,
            ordinary_min_hag_m=self.ordinary_min_hag_m,
            upper_seed_min_hag_m=self.upper_seed_min_hag_m,
            ground_max_abs_raw_hag_m=self.ground_max_abs_raw_hag_m,
            propagation_min_hag_drop_m=self.propagation_min_hag_drop_m,
            max_structural_rounds=self.max_structural_rounds,
            ground_classification_value=self.ground_classification_value,
            output_block_record_count=self.output_block_record_count,
            policy_digest_domain=self.policy_digest_domain,
        )
        if self.policy_digest != computed:
            raise ValueError("policy_digest does not match canonical FRVMEMPOL1 fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "decision_rule_version": self.decision_rule_version,
            "ordinary_min_hag_m": self.ordinary_min_hag_m,
            "upper_seed_min_hag_m": self.upper_seed_min_hag_m,
            "ground_max_abs_raw_hag_m": self.ground_max_abs_raw_hag_m,
            "propagation_min_hag_drop_m": self.propagation_min_hag_drop_m,
            "max_structural_rounds": self.max_structural_rounds,
            "ground_classification_value": self.ground_classification_value,
            "output_block_record_count": self.output_block_record_count,
            "policy_digest_domain": self.policy_digest_domain,
            "policy_digest": self.policy_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvMembershipPolicyConfig:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        return cls(**values)


def default_frv_membership_policy_config() -> FrvMembershipPolicyConfig:
    policy_digest = derive_frv_membership_policy_digest(
        schema_version=FRV_MEMBERSHIP_POLICY_SCHEMA_VERSION,
        algorithm_version="frv-membership-synchronous-v1",
        decision_rule_version="frv-membership-decision-v1",
        ordinary_min_hag_m=FRV_MEMBERSHIP_DEFAULT_ORDINARY_MIN_HAG_M,
        upper_seed_min_hag_m=FRV_MEMBERSHIP_DEFAULT_UPPER_SEED_MIN_HAG_M,
        ground_max_abs_raw_hag_m=FRV_MEMBERSHIP_DEFAULT_GROUND_MAX_ABS_RAW_HAG_M,
        propagation_min_hag_drop_m=FRV_MEMBERSHIP_DEFAULT_PROPAGATION_MIN_HAG_DROP_M,
        max_structural_rounds=FRV_MEMBERSHIP_DEFAULT_MAX_STRUCTURAL_ROUNDS,
        ground_classification_value=FRV_MEMBERSHIP_DEFAULT_GROUND_CLASSIFICATION_VALUE,
        output_block_record_count=FRV_MEMBERSHIP_DEFAULT_OUTPUT_BLOCK_RECORD_COUNT,
        policy_digest_domain=FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN,
    )
    return FrvMembershipPolicyConfig(
        schema_version=FRV_MEMBERSHIP_POLICY_SCHEMA_VERSION,
        algorithm_version="frv-membership-synchronous-v1",
        decision_rule_version="frv-membership-decision-v1",
        ordinary_min_hag_m=FRV_MEMBERSHIP_DEFAULT_ORDINARY_MIN_HAG_M,
        upper_seed_min_hag_m=FRV_MEMBERSHIP_DEFAULT_UPPER_SEED_MIN_HAG_M,
        ground_max_abs_raw_hag_m=FRV_MEMBERSHIP_DEFAULT_GROUND_MAX_ABS_RAW_HAG_M,
        propagation_min_hag_drop_m=FRV_MEMBERSHIP_DEFAULT_PROPAGATION_MIN_HAG_DROP_M,
        max_structural_rounds=FRV_MEMBERSHIP_DEFAULT_MAX_STRUCTURAL_ROUNDS,
        ground_classification_value=FRV_MEMBERSHIP_DEFAULT_GROUND_CLASSIFICATION_VALUE,
        output_block_record_count=FRV_MEMBERSHIP_DEFAULT_OUTPUT_BLOCK_RECORD_COUNT,
        policy_digest_domain=FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN,
        policy_digest=policy_digest,
    )


def validate_frv_membership_policy_config(config: FrvMembershipPolicyConfig) -> None:
    if type(config) is not FrvMembershipPolicyConfig:
        raise ValueError("config must be a FrvMembershipPolicyConfig")
    computed = derive_frv_membership_policy_digest(
        schema_version=config.schema_version,
        algorithm_version=config.algorithm_version,
        decision_rule_version=config.decision_rule_version,
        ordinary_min_hag_m=config.ordinary_min_hag_m,
        upper_seed_min_hag_m=config.upper_seed_min_hag_m,
        ground_max_abs_raw_hag_m=config.ground_max_abs_raw_hag_m,
        propagation_min_hag_drop_m=config.propagation_min_hag_drop_m,
        max_structural_rounds=config.max_structural_rounds,
        ground_classification_value=config.ground_classification_value,
        output_block_record_count=config.output_block_record_count,
        policy_digest_domain=config.policy_digest_domain,
    )
    if config.policy_digest != computed:
        raise ValueError("policy_digest does not match canonical FRVMEMPOL1 fields")


# ---------------------------------------------------------------------------
# FrvMembershipExecutionState
# ---------------------------------------------------------------------------


class FrvMembershipExecutionState(str, Enum):
    READY_FOR_V4 = "ready_for_v4"
    NO_RELEVANT_FRV_SCOPE = "no_relevant_frv_scope"
    COVERAGE_BLOCKED = "coverage_blocked"


# ---------------------------------------------------------------------------
# FrvMembershipOutputReference (runtime-only access information)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvMembershipOutputReference:
    """Runtime access information for a published membership execution."""

    root_path: Path
    execution_result_path: Path
    payload_schema_path: Path | None
    coverage_manifest_path: Path
    membership_manifest_path: Path | None
    binding_path: Path | None
    block_paths: tuple[Path, ...]
    decision_evidence_manifest_path: Path | None
    decision_evidence_block_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        for name in ("root_path", "coverage_manifest_path"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a pathlib.Path instance")
        if not isinstance(self.execution_result_path, Path):
            raise ValueError("execution_result_path must be a pathlib.Path instance")
        for name in (
            "payload_schema_path",
            "membership_manifest_path",
            "binding_path",
            "decision_evidence_manifest_path",
        ):
            if getattr(self, name) is not None and not isinstance(
                getattr(self, name), Path
            ):
                raise ValueError(f"{name} must be null or a pathlib.Path instance")
        for name in ("block_paths", "decision_evidence_block_paths"):
            value = getattr(self, name)
            if type(value) is not tuple or any(
                not isinstance(item, Path) for item in value
            ):
                raise ValueError(f"{name} must be a tuple of pathlib.Path values")


# ---------------------------------------------------------------------------
# FrvDecisionEvidenceArtifact
# ---------------------------------------------------------------------------


def _check_decision_evidence_artifact(
    artifact: FrvDecisionEvidenceArtifact,
) -> None:
    _require_exact_str(artifact.schema_version, "schema_version")
    if artifact.schema_version != FRV_DECISION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported decision evidence schema version")
    _require_sha(
        artifact.source_content_identity_digest, "source_content_identity_digest"
    )
    _require_sha(
        artifact.source_identity_record_digest, "source_identity_record_digest"
    )
    _require_sha(
        artifact.membership_artifact_manifest_digest,
        "membership_artifact_manifest_digest",
    )
    _require_sha(artifact.hag_proof_digest, "hag_proof_digest")
    _require_sha(artifact.geometry_proof_digest, "geometry_proof_digest")
    _require_sha(artifact.wire_proof_digest, "wire_proof_digest")
    _require_sha(
        artifact.original_index_domain_digest, "original_index_domain_digest"
    )
    object.__setattr__(
        artifact,
        "record_count",
        _require_nonnegative_int(artifact.record_count, "record_count"),
    )
    block_digests = _tuple_from_json(artifact.block_digests, "block_digests")
    for digest in block_digests:
        _require_sha(digest, "block_digests[]")
    if len(set(block_digests)) != len(block_digests):
        raise ValueError("block_digests must be unique")
    if tuple(sorted(block_digests)) != block_digests:
        raise ValueError("block_digests must be sorted by block ordinal")
    object.__setattr__(artifact, "block_digests", block_digests)
    _require_sha(artifact.evidence_digest, "evidence_digest")


@dataclass(frozen=True, slots=True)
class FrvDecisionEvidenceArtifact:
    """Read-only evidence sidecar bound to the membership artifact digest."""

    schema_version: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    membership_artifact_manifest_digest: str
    hag_proof_digest: str
    geometry_proof_digest: str
    wire_proof_digest: str
    original_index_domain_digest: str
    record_count: int
    block_digests: tuple[str, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        _check_decision_evidence_artifact(self)
        computed = self.computed_digest()
        if self.evidence_digest != computed:
            raise ValueError(
                "evidence_digest does not match canonical FRVEVD1 fields"
            )

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": FRV_DECISION_EVIDENCE_DIGEST_DOMAIN,
            "schema_version": self.schema_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "membership_artifact_manifest_digest": self.membership_artifact_manifest_digest,
            "hag_proof_digest": self.hag_proof_digest,
            "geometry_proof_digest": self.geometry_proof_digest,
            "wire_proof_digest": self.wire_proof_digest,
            "original_index_domain_digest": self.original_index_domain_digest,
            "record_count": self.record_count,
            "block_digests": list(self.block_digests),
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["evidence_digest"] = self.evidence_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvDecisionEvidenceArtifact:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["block_digests"] = _tuple_from_json(values["block_digests"], "block_digests")
        return cls(**values)


def validate_frv_decision_evidence_artifact(
    artifact: FrvDecisionEvidenceArtifact,
) -> None:
    if type(artifact) is not FrvDecisionEvidenceArtifact:
        raise ValueError("artifact must be a FrvDecisionEvidenceArtifact")
    _check_decision_evidence_artifact(artifact)
    if artifact.evidence_digest != artifact.computed_digest():
        raise ValueError("evidence_digest does not match canonical FRVEVD1 fields")


# ---------------------------------------------------------------------------
# FrvMembershipExecutionResult
# ---------------------------------------------------------------------------


def _check_execution_result(result: FrvMembershipExecutionResult) -> None:
    _require_enum(result.execution_state, FrvMembershipExecutionState, "execution_state")
    if type(result.source_identity) is not SourceIdentity:
        raise ValueError("source_identity must be a SourceIdentity")
    if result.a2_readiness is not None:
        if type(result.a2_readiness) is not FrvA2ReadinessRecord:
            raise ValueError("a2_readiness must be null or a FrvA2ReadinessRecord")
    if type(result.coverage_manifest) is not FRVScopeCoverageManifest:
        raise ValueError("coverage_manifest must be a FRVScopeCoverageManifest")
    if result.membership_artifact is not None and type(
        result.membership_artifact
    ) is not FullResolutionVegetationMembershipArtifact:
        raise ValueError(
            "membership_artifact must be null or a FullResolutionVegetationMembershipArtifact"
        )
    if result.payload_schema is not None and type(
        result.payload_schema
    ) is not FRVArtifactPayloadSchema:
        raise ValueError("payload_schema must be null or a FRVArtifactPayloadSchema")
    if result.artifact_binding is not None and type(
        result.artifact_binding
    ) is not FRVArtifactBindingRecord:
        raise ValueError("artifact_binding must be null or a FRVArtifactBindingRecord")
    if result.decision_evidence_artifact is not None and type(
        result.decision_evidence_artifact
    ) is not FrvDecisionEvidenceArtifact:
        raise ValueError(
            "decision_evidence_artifact must be null or a FrvDecisionEvidenceArtifact"
        )
    _require_sha(result.source_proof_digest, "source_proof_digest")
    for name in (
        "hag_proof_digest",
        "geometry_proof_digest",
        "wire_proof_digest",
    ):
        value = getattr(result, name)
        if value is not None:
            _require_sha(value, name)
    provider_bindings = _tuple_from_json(
        result.provider_binding_digests, "provider_binding_digests"
    )
    for digest in provider_bindings:
        _require_sha(digest, "provider_binding_digests[]")
    if tuple(sorted(set(provider_bindings))) != provider_bindings:
        raise ValueError("provider_binding_digests must be sorted and unique")
    object.__setattr__(result, "provider_binding_digests", provider_bindings)
    override_digests = _tuple_from_json(result.override_digests, "override_digests")
    for digest in override_digests:
        _require_sha(digest, "override_digests[]")
    if tuple(sorted(set(override_digests))) != override_digests:
        raise ValueError("override_digests must be sorted and unique")
    object.__setattr__(result, "override_digests", override_digests)
    failures = _tuple_from_json(result.failure_records, "failure_records")
    for failure in failures:
        validate_frv_a2_failure_record(failure)
    if tuple(sorted(failures, key=_failure_key)) != failures:
        raise ValueError("failure_records must use the A2 total order")
    object.__setattr__(result, "failure_records", failures)
    if result.output_reference is not None:
        if type(result.output_reference) is not FrvMembershipOutputReference:
            raise ValueError("output_reference must be null or a FrvMembershipOutputReference")
    _require_sha(result.result_digest, "result_digest")


def _failure_key(failure: FrvA2FailureRecord) -> tuple[int, int, str, str, str, int]:
    phase_order = {
        member: index for index, member in enumerate(FrvA2FailurePhase)
    }
    role_order = {
        member: index for index, member in enumerate(FrvA2ProviderRole)
    }
    return (
        phase_order[failure.phase],
        role_order[failure.provider_role],
        failure.provider_key,
        failure.field_name,
        failure.error_code.value,
        failure.reason_flags,
    )


@dataclass(frozen=True, slots=True)
class FrvMembershipExecutionResult:
    """Immutable membership execution result; paths never enter the digest."""

    execution_state: FrvMembershipExecutionState
    source_identity: SourceIdentity
    a2_readiness: FrvA2ReadinessRecord | None
    coverage_manifest: FRVScopeCoverageManifest
    membership_artifact: FullResolutionVegetationMembershipArtifact | None
    payload_schema: FRVArtifactPayloadSchema | None
    artifact_binding: FRVArtifactBindingRecord | None
    decision_evidence_artifact: FrvDecisionEvidenceArtifact | None
    source_proof_digest: str
    hag_proof_digest: str | None
    geometry_proof_digest: str | None
    wire_proof_digest: str | None
    provider_binding_digests: tuple[str, ...]
    override_digests: tuple[str, ...]
    failure_records: tuple[FrvA2FailureRecord, ...]
    output_reference: FrvMembershipOutputReference | None
    result_digest: str

    def __post_init__(self) -> None:
        _check_execution_result(self)
        computed = self.computed_digest()
        if self.result_digest != computed:
            raise ValueError("result_digest does not match canonical FRVMEMRES1 fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": FRV_MEMBERSHIP_RESULT_DIGEST_DOMAIN,
            "execution_state": self.execution_state.value,
            "source_identity": self.source_identity.to_dict(),
            "a2_readiness": (
                self.a2_readiness.to_dict() if self.a2_readiness is not None else None
            ),
            "coverage_manifest": self.coverage_manifest.to_dict(),
            "membership_artifact": (
                self.membership_artifact.to_dict()
                if self.membership_artifact is not None
                else None
            ),
            "payload_schema": (
                self.payload_schema.to_dict() if self.payload_schema is not None else None
            ),
            "artifact_binding": (
                self.artifact_binding.to_dict()
                if self.artifact_binding is not None
                else None
            ),
            "decision_evidence_artifact": (
                self.decision_evidence_artifact.to_dict()
                if self.decision_evidence_artifact is not None
                else None
            ),
            "source_proof_digest": self.source_proof_digest,
            "hag_proof_digest": self.hag_proof_digest,
            "geometry_proof_digest": self.geometry_proof_digest,
            "wire_proof_digest": self.wire_proof_digest,
            "provider_binding_digests": list(self.provider_binding_digests),
            "override_digests": list(self.override_digests),
            "failure_records": [
                failure.to_dict() for failure in self.failure_records
            ],
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["result_digest"] = self.result_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvMembershipExecutionResult:
        allowed = set(cls.__dataclass_fields__) - {"output_reference"}
        values = _exact_fields(data, allowed, cls.__name__)
        values["execution_state"] = _enum_from_value(
            values["execution_state"], FrvMembershipExecutionState, "execution_state"
        )
        values["source_identity"] = SourceIdentity.from_dict(
            _mapping(values["source_identity"], "source_identity")
        )
        values["a2_readiness"] = (
            FrvA2ReadinessRecord.from_dict(
                _mapping(values["a2_readiness"], "a2_readiness")
            )
            if values["a2_readiness"] is not None
            else None
        )
        values["coverage_manifest"] = FRVScopeCoverageManifest.from_dict(
            _mapping(values["coverage_manifest"], "coverage_manifest")
        )
        values["membership_artifact"] = (
            FullResolutionVegetationMembershipArtifact.from_dict(
                _mapping(values["membership_artifact"], "membership_artifact")
            )
            if values["membership_artifact"] is not None
            else None
        )
        values["payload_schema"] = (
            FRVArtifactPayloadSchema.from_dict(
                _mapping(values["payload_schema"], "payload_schema")
            )
            if values["payload_schema"] is not None
            else None
        )
        values["artifact_binding"] = (
            FRVArtifactBindingRecord.from_dict(
                _mapping(values["artifact_binding"], "artifact_binding")
            )
            if values["artifact_binding"] is not None
            else None
        )
        values["decision_evidence_artifact"] = (
            FrvDecisionEvidenceArtifact.from_dict(
                _mapping(
                    values["decision_evidence_artifact"],
                    "decision_evidence_artifact",
                )
            )
            if values["decision_evidence_artifact"] is not None
            else None
        )
        values["provider_binding_digests"] = _tuple_from_json(
            values["provider_binding_digests"], "provider_binding_digests"
        )
        values["override_digests"] = _tuple_from_json(
            values["override_digests"], "override_digests"
        )
        values["failure_records"] = tuple(
            FrvA2FailureRecord.from_dict(item)
            for item in _tuple_from_json(values["failure_records"], "failure_records")
        )
        values["output_reference"] = None
        return cls(**values)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_frv_membership_execution_result(
    result: FrvMembershipExecutionResult,
) -> None:
    _check_execution_result(result)
    validate_source_content_identity_digest(result.source_identity)
    validate_source_identity_record_digest(result.source_identity)
    validate_scope_inventory_conservation(result.coverage_manifest)
    if result.membership_artifact is not None:
        _check_artifact_domain_bindings(result)
    if result.result_digest != result.computed_digest():
        raise ValueError("result_digest does not match canonical FRVMEMRES1 fields")


def _check_artifact_domain_bindings(result: FrvMembershipExecutionResult) -> None:
    artifact = result.membership_artifact
    assert artifact is not None
    if result.payload_schema is not None:
        validate_payload_schema(result.payload_schema)
        validate_artifact_payload_digest_domain(
            artifact, payload_schema=result.payload_schema
        )
    if result.artifact_binding is not None:
        validate_artifact_binding_record(
            result.artifact_binding,
            artifact,
            coverage_manifest=result.coverage_manifest,
        )


def validate_membership_result_state_completeness(
    result: FrvMembershipExecutionResult,
) -> None:
    """Enforce the frozen READY / BLOCKED / NO_SCOPE truth tables."""

    _check_execution_result(result)
    state = result.execution_state
    artifact = result.membership_artifact
    if state is FrvMembershipExecutionState.READY_FOR_V4:
        if artifact is None:
            raise ValueError("READY_FOR_V4 requires a membership artifact")
        if result.payload_schema is None:
            raise ValueError("READY_FOR_V4 requires a payload schema")
        if result.artifact_binding is None:
            raise ValueError("READY_FOR_V4 requires an artifact binding")
        if result.decision_evidence_artifact is None:
            raise ValueError("READY_FOR_V4 requires a decision evidence artifact")
        if result.output_reference is None:
            raise ValueError("READY_FOR_V4 requires an output reference")
        if result.failure_records:
            raise ValueError("READY_FOR_V4 cannot contain failure records")
        if result.coverage_manifest.routing_complete is not True:
            raise ValueError("READY_FOR_V4 requires routing complete")
        for name in (
            "canonical_inventory_complete",
            "required_inventory_complete",
            "approved_hard_rejection_complete",
            "scope_coverage_complete",
            "routing_complete",
            "membership_decision_complete",
        ):
            if getattr(artifact, name) is not True:
                raise ValueError(f"READY_FOR_V4 requires {name}")
    elif state is FrvMembershipExecutionState.COVERAGE_BLOCKED:
        if (
            artifact is not None
            or result.payload_schema is not None
            or result.artifact_binding is not None
            or result.decision_evidence_artifact is not None
            or result.output_reference is not None
        ):
            raise ValueError("COVERAGE_BLOCKED forbids artifact/output fields")
        if not result.failure_records:
            raise ValueError("COVERAGE_BLOCKED requires failure records")
        if result.coverage_manifest.routing_complete is not False:
            raise ValueError("COVERAGE_BLOCKED requires routing incomplete")
    else:  # NO_RELEVANT_FRV_SCOPE
        if artifact is not None:
            raise ValueError("NO_RELEVANT_FRV_SCOPE forbids a membership artifact")
        if result.failure_records:
            raise ValueError("NO_RELEVANT_FRV_SCOPE forbids failure records")
        if result.coverage_manifest.routing_complete is not True:
            raise ValueError("NO_RELEVANT_FRV_SCOPE requires routing complete")


__all__ = [
    "FRV_SCOPE_POLICY_SCHEMA_VERSION",
    "FRV_SCOPE_POLICY_DIGEST_DOMAIN",
    "FRV_SCOPE_GEOMETRY_DIGEST_DOMAIN",
    "FRV_MEMBERSHIP_POLICY_SCHEMA_VERSION",
    "FRV_MEMBERSHIP_POLICY_DIGEST_DOMAIN",
    "FRV_MEMBERSHIP_RESULT_DIGEST_DOMAIN",
    "FRV_DECISION_EVIDENCE_SCHEMA_VERSION",
    "FRV_DECISION_EVIDENCE_DIGEST_DOMAIN",
    "FRV_SCOPE_DEFAULT_VOXEL_SIZE_M",
    "FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_XY_CELLS",
    "FRV_SCOPE_DEFAULT_CONTEXT_MARGIN_Z_CELLS",
    "FRV_SCOPE_DEFAULT_TILE_EDGE_CELLS",
    "FRV_SCOPE_DEFAULT_FEATURE_HALO_CELLS",
    "FRV_SCOPE_DEFAULT_BOUNDARY_RULE",
    "FRV_SCOPE_DEFAULT_COORDINATE_CANONICALIZATION",
    "FRV_MEMBERSHIP_DEFAULT_ORDINARY_MIN_HAG_M",
    "FRV_MEMBERSHIP_DEFAULT_UPPER_SEED_MIN_HAG_M",
    "FRV_MEMBERSHIP_DEFAULT_GROUND_MAX_ABS_RAW_HAG_M",
    "FRV_MEMBERSHIP_DEFAULT_PROPAGATION_MIN_HAG_DROP_M",
    "FRV_MEMBERSHIP_DEFAULT_MAX_STRUCTURAL_ROUNDS",
    "FRV_MEMBERSHIP_DEFAULT_GROUND_CLASSIFICATION_VALUE",
    "FRV_MEMBERSHIP_DEFAULT_OUTPUT_BLOCK_RECORD_COUNT",
    "FrvScopeEvidenceReference",
    "FrvScopePolicyConfig",
    "FrvMembershipPolicyConfig",
    "FrvMembershipExecutionState",
    "FrvMembershipOutputReference",
    "FrvDecisionEvidenceArtifact",
    "FrvMembershipExecutionResult",
    "derive_frv_scope_policy_digest",
    "derive_frv_membership_policy_digest",
    "default_frv_scope_policy_config",
    "default_frv_membership_policy_config",
    "validate_frv_scope_evidence_reference",
    "validate_frv_scope_policy_config",
    "validate_frv_membership_policy_config",
    "validate_frv_decision_evidence_artifact",
    "validate_frv_membership_execution_result",
    "validate_membership_result_state_completeness",
    "frv_source_proof_digest",
]
