from algorithms.prediction.service import PredictionService
from algorithms.prediction.batch import BatchPredictionService
from algorithms.prediction.report import PredictionReport


def test_single_tree_prediction():

    service = PredictionService()

    result = service.predict_tree(
        1,
        "松树",
        20,
        30,
        45,
        15,
        50,
        70,
        20,
    )

    assert result["tree_id"] == 1
    assert result["result"].risk_level in [
        "低风险",
        "中风险",
        "高风险",
    ]


def test_batch_prediction():

    batch = BatchPredictionService()

    trees = [
        {
            "tree_id": 1,
            "species": "松树",
            "height": 20,
            "age": 30,
            "diameter": 45,
            "wind_speed": 15,
            "rainfall": 50,
            "humidity": 70,
            "slope": 20,
        },
        {
            "tree_id": 2,
            "species": "杉树",
            "height": 35,
            "age": 50,
            "diameter": 60,
            "wind_speed": 25,
            "rainfall": 90,
            "humidity": 90,
            "slope": 40,
        },
    ]

    results = batch.predict_batch(trees)

    assert len(results) == 2


def test_report_generation():

    report = PredictionReport()

    results = [
        {
            "tree_id": 1,
            "result": type(
                "R",
                (),
                {"risk_level": "低风险"}
            )(),
        },
        {
            "tree_id": 2,
            "result": type(
                "R",
                (),
                {"risk_level": "高风险"}
            )(),
        },
    ]

    output = report.generate(results)

    assert output["total_tree"] == 2
    assert output["high_risk_count"] == 1