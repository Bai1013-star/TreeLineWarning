from pathlib import Path

import pytest
from algorithms.prediction.local_provider import (
    LONG_HORIZONS_MONTHS,
    SHORT_HORIZONS_DAYS,
    SPECIES_MODEL_SHA256,
    TEMPORAL_MODEL_SHA256,
    LocalPredictionProvider,
    build_prediction_input,
    verify_model_assets,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES_AVAILABLE = (ROOT / "models" / "species_growth.json").is_file() and (
    ROOT / "models" / "temporal_attention.npz"
).is_file()


pytestmark = pytest.mark.skipif(not MODEL_FILES_AVAILABLE, reason="deployment model assets absent")


def _package():
    return build_prediction_input(
        project_id="project-1",
        group_id="G002",
        span_id="S005",
        run_id="s005-fixture",
        span_context={"tower_pair": "T006-T007", "span_length_m": 52.7},
        trees=[
            {
                "persistent_tree_id": "G002-T001",
                "global_tree_id": 0,
                "category": "CORE",
                "prediction_identity": {"eligible_for_long_term_history": True},
                "apex_height_m": 12.4,
                "minimum_clearance_m": 0.05,
                "dangerous": True,
                "danger_point_count": 8,
                "clearance": {"nearest_wire_id": 2},
            }
        ],
    )


def test_deployment_model_hashes_match_recovered_assets():
    verified = verify_model_assets()
    assert verified["species_model_sha256"] == SPECIES_MODEL_SHA256
    assert verified["temporal_model_sha256"] == TEMPORAL_MODEL_SHA256


@pytest.mark.parametrize("mode", ["formula", "ai"])
def test_forecast_modes_preserve_persistent_identity_and_horizons(mode):
    result = LocalPredictionProvider().run(_package(), prediction_mode=mode)

    assert result.status == "COMPLETE"
    assert result.failed_tree_ids == ()
    assert result.entries[0]["persistent_tree_id"] == "G002-T001"
    assert result.entries[0]["science_run_id"] == "s005-fixture"
    assert result.horizons == tuple(
        [f"{days}天" for days in SHORT_HORIZONS_DAYS]
        + [f"{months}个月" for months in LONG_HORIZONS_MONTHS]
    )
    assert result.entries[0]["uses_temporal_model"] is (mode == "ai")
    assert result.entries[0]["species_fallback"] == "未识别 / 其他（模型回退）"
