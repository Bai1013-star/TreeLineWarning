from algorithms.prediction.local_provider import LocalPredictionProvider, build_prediction_input


def test_local_provider_runs_shipped_rule_based_baseline():
    package = build_prediction_input(
        project_id="project-1",
        group_id="G002",
        span_id="S005",
        run_id="s005-run",
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

    assert package.observations[0]["persistent_tree_id"] == "G002-T001"
    assert package.observations[0]["current_safety"]["minimum_clearance_m"] == 0.05
    result = LocalPredictionProvider().run(package)
    assert result.status == "COMPLETE"
    assert result.horizons == ("current-baseline",)
    assert result.entries[0]["persistent_tree_id"] == "G002-T001"
    assert result.entries[0]["science_run_id"] == "s005-run"
    assert result.entries[0]["model_label"] == "Baseline / Rule-based"
    assert result.entries[0]["current_science"]["minimum_clearance_m"] == 0.05
    assert result.entries[0]["horizons"][0]["future_clearance_m"] is None
    assert "species:未知" in result.entries[0]["core_default_flags"]
    assert "voltage_level_kv" in result.entries[0]["missing_feature_flags"]
