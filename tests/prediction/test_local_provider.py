from algorithms.prediction.local_provider import LocalPredictionProvider, build_prediction_input


def test_local_provider_preserves_terminal_fields_and_blocks_without_core(monkeypatch):
    monkeypatch.delenv("TREELINE_WARNING_PREDICTION_ENTRYPOINT", raising=False)
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
    assert result.status == "BLOCKED"
    assert result.diagnostic == "PREDICTION_CORE_NOT_CONFIGURED"
    assert result.entries == ()

