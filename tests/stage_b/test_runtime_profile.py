from types import SimpleNamespace

from core.stage_b.runtime_profile import STAGE_ORDER, production_runtime_profile


def test_runtime_profile_reports_complete_stage_order_and_real_gpu_evidence() -> None:
    acceleration = {
        "effective_mode": "GPU_CERTIFIED_CPU_EXACT",
        "capability": {"device_name": "Test CUDA Device"},
        "screen_kernel_seconds": 0.12,
        "certificate_kernel_seconds": 0.34,
        "certificate_success_points": 1234,
        "fallback_points": 0,
    }
    span = SimpleNamespace(
        diagnostics={
            "source_spatial_query": {
                "vegetation_after_confirmed_wire_exclusion": 2000,
                "tree_associated_raw_points": 1500,
                "wire_region_points": 300,
            },
            "wire": {"confirmed_conductor_member_count": 100},
            "clearance": {"acceleration": acceleration},
        }
    )
    telemetry = SimpleNamespace(
        to_dict=lambda: {
            "SOURCE_POINT_COUNT": 5000,
            "TOTAL_TIME": 9.0,
            "STAGE_TIME": {"wire": 1.0, "tree": 2.0, "clearance": 3.0},
        }
    )
    result = SimpleNamespace(
        telemetry=telemetry,
        span_results={"S1": span},
        global_reconciled=SimpleNamespace(global_tree_records=(object(), object())),
        final_clearance=SimpleNamespace(records=(object(), object())),
    )

    profile = production_runtime_profile(
        result, source_preparation_seconds=4.0, source_cache_hit=True
    )

    assert [stage["STAGE"] for stage in profile["stages"]] == list(STAGE_ORDER)
    clearance = next(stage for stage in profile["stages"] if stage["STAGE"] == "CLEARANCE")
    assert clearance["BACKEND"] == "MIXED"
    assert clearance["GPU_DEVICE"] == "Test CUDA Device"
    assert clearance["GPU_UTILIZATION_EVIDENCE"]["fallback_points"] == 0
    assert profile["GPU_ACCELERATED_STAGES"] == ["CLEARANCE_CANDIDATE_SCREENING"]

