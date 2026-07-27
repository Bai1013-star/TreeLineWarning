"""
Prediction service

统一预测入口
"""

from .adapter import from_tree_data
from .predictor import TreeRiskPredictor


class PredictionService:
    """
    风险预测服务
    """

    def __init__(self):
        self.predictor = TreeRiskPredictor()


    def predict_tree(
        self,
        tree_id: int,
        species: str,
        height: float,
        age: int,
        diameter: float,
        wind_speed: float,
        rainfall: float,
        humidity: float,
        slope: float,
    ):
        """
        输入树木和环境数据
        返回风险预测结果
        """

        prediction_input = from_tree_data(
            tree_id,
            species,
            height,
            age,
            diameter,
            wind_speed,
            rainfall,
            humidity,
            slope,
        )

        result = self.predictor.predict(
            prediction_input.tree_input
        )

        return {
            "tree_id": prediction_input.tree_id,
            "result": result,
        }