"""
TreeLineWarning Prediction Engine

综合树木风险预测模型

输入:
    树种
    树木参数
    天气参数
    地形参数
    电线距离

输出:
    风险等级
    风险分数
    建议
"""


from dataclasses import dataclass

from algorithms.prediction.species import get_species_factor
from algorithms.prediction.tree_factor import calculate_tree_factor
from algorithms.prediction.wire_factor import calculate_wire_factor



@dataclass
class TreeInput:
    """
    树木输入数据
    """

    species: str

    height: float
    age: int
    diameter: float

    wind_speed: float
    rainfall: float
    humidity: float

    slope: float

    distance_to_wire: float = 999



@dataclass
class PredictionResult:
    """
    预测结果
    """

    risk_level: str
    risk_score: float
    suggestion: str



class TreeRiskPredictor:
    """
    综合风险预测器
    """

    def predict(self, tree: TreeInput):

        score = 0


        # ----------------------
        # 1 天气风险
        # ----------------------

        if tree.wind_speed > 15:
            score += 30

        elif tree.wind_speed > 10:
            score += 15



        if tree.rainfall > 50:
            score += 20



        if tree.slope > 25:
            score += 20



        # ----------------------
        # 2 树体风险
        # ----------------------

        score += calculate_tree_factor(
            tree.height,
            tree.age,
            tree.diameter
        )



        # ----------------------
        # 3 电线距离风险
        # ----------------------

        score += calculate_wire_factor(
            tree.distance_to_wire
        )



        # ----------------------
        # 4 树种影响
        # ----------------------

        factor = get_species_factor(
            tree.species
        )


        score = score * factor



        # 限制最高分

        if score > 100:
            score = 100



        # ----------------------
        # 风险等级
        # ----------------------

        if score >= 70:

            level = "高风险"
            suggestion = "建议立即巡检"


        elif score >= 40:

            level = "中风险"
            suggestion = "建议加强监测"


        else:

            level = "低风险"
            suggestion = "正常维护"



        return PredictionResult(
            risk_level=level,
            risk_score=score,
            suggestion=suggestion
        )