"""Evidence-only production runtime profile; no science is executed here."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STAGE_ORDER = (
    "RAW_IMPORT", "LAS_LAZ_DECODE", "PDAL_PROCESSING", "CACHE_BUILD",
    "GROUND", "HAG", "WIRE", "VEGETATION", "TREE_SEGMENTATION",
    "SPAN_ORGANIZATION", "CLEARANCE", "RISK", "REPORT_BUILD",
    "VIEWER_PAYLOAD", "TOTAL",
)


def production_runtime_profile(
    result: Any,
    *,
    source_preparation_seconds: float | None,
    source_cache_hit: bool | None,
) -> dict[str, Any]:
    telemetry = result.telemetry.to_dict()
    timings = dict(telemetry.get("STAGE_TIME") or {})
    span_results = list((getattr(result, "span_results", {}) or {}).values())
    span_diagnostics = [
        value.diagnostics for value in span_results
        if isinstance(value.diagnostics, Mapping)
    ]
    source_queries = [
        item.get("source_spatial_query") for item in span_diagnostics
        if isinstance(item.get("source_spatial_query"), Mapping)
    ]
    clearance_details = [
        item.get("clearance") for item in span_diagnostics
        if isinstance(item.get("clearance"), Mapping)
    ]
    accelerations = [
        item.get("acceleration") for item in clearance_details
        if isinstance(item.get("acceleration"), Mapping)
    ]
    gpu = next((item for item in accelerations if item.get("effective_mode") == "GPU_CERTIFIED_CPU_EXACT"), None)
    capability = gpu.get("capability", {}) if isinstance(gpu, Mapping) else {}
    source_points = int(telemetry.get("SOURCE_POINT_COUNT", 0) or 0)
    vegetation_points = sum(int(item.get("vegetation_after_confirmed_wire_exclusion", 0) or 0) for item in source_queries)
    tree_member_points = sum(int(item.get("tree_associated_raw_points", 0) or 0) for item in source_queries)
    tree_count = len(result.global_reconciled.global_tree_records) if result.global_reconciled is not None else 0
    elapsed = {
        "CACHE_BUILD": source_preparation_seconds,
        "WIRE": timings.get("wire"),
        "VEGETATION": timings.get("vegetation_role_query"),
        "TREE_SEGMENTATION": timings.get("tree"),
        "SPAN_ORGANIZATION": timings.get("tree_reconciliation"),
        "CLEARANCE": timings.get("clearance"),
        "RISK": timings.get("clearance_risk_json_write"),
        "REPORT_BUILD": timings.get("manifest_write"),
        "VIEWER_PAYLOAD": timings.get("viewer_artifact_build"),
        "TOTAL": telemetry.get("TOTAL_TIME"),
    }
    backends = {
        "RAW_IMPORT": "IO", "LAS_LAZ_DECODE": "IO", "PDAL_PROCESSING": "CPU",
        "CACHE_BUILD": "IO", "GROUND": "CPU", "HAG": "CPU", "WIRE": "CPU",
        "VEGETATION": "IO", "TREE_SEGMENTATION": "CPU", "SPAN_ORGANIZATION": "CPU",
        "CLEARANCE": "MIXED" if gpu is not None else "CPU", "RISK": "CPU",
        "REPORT_BUILD": "CPU", "VIEWER_PAYLOAD": "CPU", "TOTAL": "MIXED",
    }
    inputs = {
        "RAW_IMPORT": source_points, "LAS_LAZ_DECODE": source_points,
        "PDAL_PROCESSING": source_points, "CACHE_BUILD": source_points,
        "GROUND": source_points, "HAG": source_points,
        "WIRE": sum(int(item.get("wire_region_points", 0) or 0) for item in source_queries),
        "VEGETATION": vegetation_points, "TREE_SEGMENTATION": vegetation_points,
        "SPAN_ORGANIZATION": tree_member_points, "CLEARANCE": tree_member_points,
        "RISK": tree_count, "REPORT_BUILD": tree_count,
        "VIEWER_PAYLOAD": tree_member_points, "TOTAL": source_points,
    }
    outputs = {
        "WIRE": sum(int((item.get("wire") or {}).get("confirmed_conductor_member_count", 0) or 0) for item in span_diagnostics if isinstance(item.get("wire"), Mapping)),
        "VEGETATION": vegetation_points, "TREE_SEGMENTATION": tree_count,
        "SPAN_ORGANIZATION": tree_count,
        "CLEARANCE": len(getattr(result.final_clearance, "records", ()) or ()),
        "RISK": tree_count, "REPORT_BUILD": tree_count,
        "VIEWER_PAYLOAD": tree_count, "TOTAL": tree_count,
    }
    cache_stages = {"LAS_LAZ_DECODE", "PDAL_PROCESSING", "CACHE_BUILD", "GROUND", "HAG"}
    stages = []
    for name in STAGE_ORDER:
        duration = elapsed.get(name)
        stages.append({
            "STAGE": name,
            "BACKEND": backends[name],
            "START_TIME": "UNKNOWN",
            "END_TIME": "UNKNOWN",
            "ELAPSED_SECONDS": "UNKNOWN" if duration is None else float(duration),
            "INPUT_POINTS": inputs.get(name, "UNKNOWN"),
            "OUTPUT_POINTS": outputs.get(name, "UNKNOWN"),
            "CACHE_HIT": bool(source_cache_hit) if name in cache_stages and source_cache_hit is not None else "UNKNOWN",
            "CACHE_MISS": (not bool(source_cache_hit)) if name in cache_stages and source_cache_hit is not None else "UNKNOWN",
            "GPU_DEVICE": capability.get("device_name", "UNKNOWN") if name == "CLEARANCE" else "NONE",
            "GPU_MEMORY_USED": "UNKNOWN",
            "GPU_UTILIZATION_EVIDENCE": ({
                "effective_mode": gpu.get("effective_mode"),
                "screen_kernel_seconds": gpu.get("screen_kernel_seconds"),
                "certificate_kernel_seconds": gpu.get("certificate_kernel_seconds"),
                "certificate_success_points": gpu.get("certificate_success_points"),
                "fallback_points": gpu.get("fallback_points"),
            } if name == "CLEARANCE" and gpu is not None else "NONE"),
        })
    return {
        "schema": "treeline-production-runtime-profile-v1",
        "stages": stages,
        "GPU_ACCELERATED_STAGES": ["CLEARANCE_CANDIDATE_SCREENING"] if gpu else [],
        "CPU_ONLY_STAGES": ["GROUND", "HAG", "WIRE", "TREE_SEGMENTATION", "SPAN_ORGANIZATION", "RISK", "REPORT_BUILD", "VIEWER_PAYLOAD"],
        "IO_BOUND_STAGES": ["RAW_IMPORT", "LAS_LAZ_DECODE", "CACHE_BUILD", "VEGETATION"],
        "MIXED_STAGES": ["CLEARANCE"] if gpu else [],
        "GPU_FALLBACK_STAGES": [],
        "GPU_FALLBACK_REASON": "NONE" if gpu else "NO_PRODUCTION_GPU_EVIDENCE",
        "PEAK_RAM": telemetry.get("PEAK_MEMORY_BYTES") or "UNKNOWN",
        "PAGEFILE_USAGE": "UNKNOWN", "GPU_PEAK_VRAM": "UNKNOWN",
        "DISK_READ_VOLUME": "UNKNOWN", "DISK_WRITE_VOLUME": "UNKNOWN",
    }


__all__ = ["STAGE_ORDER", "production_runtime_profile"]
