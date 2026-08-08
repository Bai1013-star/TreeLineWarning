"""Shared test-only SourceIdentity / P-HAG config factories for FGC tests.

Synthetic only; never imported by production code.
"""

from __future__ import annotations

from typing import Any

import core.contracts.frv as frv
from core.contracts.frv_a2 import FrvA2ProviderRole
from core.contracts.frv_hag import (
    FRV_HAG_ALGORITHM_VERSION,
    FRV_HAG_CONFIG_SCHEMA_VERSION,
    FRV_HAG_PROVIDER_INTERFACE_VERSION,
    FRV_HAG_PROVIDER_KEY,
    FrvHagProviderConfig,
    derive_frv_hag_provider_config_digest,
)

SHA = "a" * 64


def make_hag_config(*, cell_size_m: float = 1.0) -> FrvHagProviderConfig:
    return FrvHagProviderConfig(
        schema_version=FRV_HAG_CONFIG_SCHEMA_VERSION,
        provider_key=FRV_HAG_PROVIDER_KEY,
        provider_role=FrvA2ProviderRole.HAG,
        interface_version=FRV_HAG_PROVIDER_INTERFACE_VERSION,
        algorithm_version=FRV_HAG_ALGORITHM_VERSION,
        cell_size_m=cell_size_m,
        support_radius_m=3.0,
        min_support_cells=1,
        max_support_cells=16,
        negative_hag_tolerance_m=0.1,
        classification_dimension="classification",
        provider_config_digest=derive_frv_hag_provider_config_digest(
            schema_version=FRV_HAG_CONFIG_SCHEMA_VERSION,
            provider_key=FRV_HAG_PROVIDER_KEY,
            provider_role=FrvA2ProviderRole.HAG,
            interface_version=FRV_HAG_PROVIDER_INTERFACE_VERSION,
            algorithm_version=FRV_HAG_ALGORITHM_VERSION,
            cell_size_m=cell_size_m,
            support_radius_m=3.0,
            min_support_cells=1,
            max_support_cells=16,
            negative_hag_tolerance_m=0.1,
            classification_dimension="classification",
        ),
    )


def make_source_identity(
    *,
    source_id: str = "scene-1",
    path: str = "C:/data/scan.las",
    file_sha: str = SHA,
    count: int = 7,
) -> frv.SourceIdentity:
    relocation_payload = {
        "audit_schema_version": frv.FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "windows",
        "recorded_path_raw": path,
        "resolved_path_raw": path,
        "recorded_path_lexical_normalized": path,
        "resolved_path_lexical_normalized": path,
        "resolved_final_path": None,
        "resolution_method": "exact_match",
        "case_comparison_mode": "case_insensitive",
        "platform_name": "windows",
        "run_id": "run-1",
        "audit_timestamp": "2026-01-01T00:00:00Z",
        "content_identity_result": "matched",
        "warning_flags": 0,
    }
    relocation_payload["audit_digest"] = frv.frv_digest(relocation_payload)
    relocation = frv.FrvRelocationAudit.from_dict(relocation_payload)
    content_payload: dict[str, Any] = {
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": source_id,
        "source_file_sha256": file_sha,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "canonical_point_content_spec_version": None,
        "source_point_count": count,
        "source_bounds_min_xyz": [0.0, 0.0, 0.0],
        "source_bounds_max_xyz": [100.0, 100.0, 20.0],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_context": [0.0, 0.0, 0.0],
    }
    content_digest = frv.frv_digest(content_payload)
    record_payload = {
        **content_payload,
        "recorded_source_path": path,
        "resolved_access_path": path,
        "relocation_audit_audit_digest": relocation.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return frv.SourceIdentity.from_dict(
        {
            **content_payload,
            "recorded_source_path": path,
            "resolved_access_path": path,
            "relocation_audit": relocation.to_dict(),
            "source_equivalence_audit": None,
            "source_content_identity_digest": content_digest,
            "source_identity_record_digest": frv.frv_digest(record_payload),
        }
    )
