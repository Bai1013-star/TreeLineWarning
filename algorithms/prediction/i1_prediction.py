"""
I1 Prediction Adapter

作用:
将 I1 数据转换为 TreeInput
然后调用风险预测模型
"""


from dataclasses import dataclass

from algorithms.prediction.predictor import (
    TreeInput,
    TreeRiskPredictor
)



@dataclass
class I1TreeData:
    """
    I1中的单棵树数据
    """

    tree_id: int

    species: str

    height: float

    age: int

    diameter: float

    distance_to_wire: float

    wind_speed: float = 0

    rainfall: float = 0

    humidity: float = 0

    slope: float = 0



class I1PredictionService:
    """
    I1预测服务
    """


    def __init__(self):

        self.predictor = TreeRiskPredictor()



    def predict_tree(self, tree:I1TreeData):


        input_data = TreeInput(

            species=tree.species,

            height=tree.height,

            age=tree.age,

            diameter=tree.diameter,

            wind_speed=tree.wind_speed,

            rainfall=tree.rainfall,

            humidity=tree.humidity,

            slope=tree.slope,

            distance_to_wire=tree.distance_to_wire
        )


        result = self.predictor.predict(input_data)


        return {

            "tree_id": tree.tree_id,

            "result": result

        }