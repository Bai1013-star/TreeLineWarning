from desktop.work_order import render_work_order


def test_printable_work_order_keeps_science_and_prediction_boundaries() -> None:
    html = render_work_order(
        {
            "project": {"project_name": "示例项目"},
            "span": {
                "group_id": "G002",
                "span_id": "S005",
                "from_tower_id": "T006",
                "to_tower_id": "T007",
            },
            "tree": {
                "display_id": "G002-T002",
                "persistent_tree_id": "G002-T002",
                "apex_height_m": 12.0,
                "dangerous": True,
            },
            "entry": {
                "prediction_mode": "ai",
                "model_label": "时序注意力预测（合成数据基线）",
                "species_fallback": "未识别 / 其他（模型回退）",
                "current_science": {"minimum_clearance_m": 2.0, "danger_point_count": 3},
                "short_term_forecast": [
                    {"period": "1天", "predicted_clearance_distance": 1.9, "risk_level": "预警"}
                ],
                "long_term_forecast": [
                    {
                        "horizon_months": 12,
                        "predicted_height": 12.4,
                        "predicted_crown_width": 4.1,
                        "predicted_clearance_distance": 1.4,
                        "risk_level": "高风险",
                    }
                ],
            },
            "work_order_id": "TLW-TEST",
        }
    )

    assert "persistent_tree_id" in html
    assert "MAIN SCIENCE" in html
    assert "PREDICTION LAYER" in html
    assert "打印 / 保存为 PDF" in html
    assert "时序注意力预测（合成数据基线）" in html
