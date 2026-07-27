"""
Prediction Pipeline Tests
"""


from algorithms.prediction.pipeline import (
    PredictionPipeline
)

from algorithms.prediction.i1_prediction import (
    I1TreeData
)



def test_pipeline_high_risk():

    trees = [

        I1TreeData(
            tree_id=1,
            species="松树",
            height=25,
            age=40,
            diameter=50,
            distance_to_wire=3,
            wind_speed=20,
            rainfall=60,
            humidity=80,
            slope=30
        )

    ]


    result = PredictionPipeline().run(trees)


    report = result["report"]


    assert report["total_tree"] == 1

    assert report["high_risk_count"] == 1

    assert 1 in report["high_risk_tree_ids"]



def test_pipeline_low_risk():

    trees = [

        I1TreeData(
            tree_id=2,
            species="杉树",
            height=5,
            age=5,
            diameter=10,
            distance_to_wire=30,
            wind_speed=2,
            rainfall=0,
            humidity=40,
            slope=5
        )

    ]


    result = PredictionPipeline().run(trees)


    report = result["report"]


    assert report["total_tree"] == 1

    assert report["low_risk_count"] == 1