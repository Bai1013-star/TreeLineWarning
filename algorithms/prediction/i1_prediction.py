"""
I1 Prediction Adapter

作用:
将 I1 数据转换为 TreeInput
调用风险预测模型
"""


from dataclasses import asdict, dataclass
from typing import Any

from algorithms.prediction.predictor import TreeInput, TreeRiskPredictor


@dataclass
class I1TreeData:

    tree_id: int

    species: str

    height: float

    age: int

    diameter: float

    distance_to_wire: float
    vertical_distance: float = 999
    horizontal_distance: float = 999
    clearance_distance: float = 999
    wind_offset_distance: float = 0
    voltage_level_kv: float = 110
    maximum_sag_winter: float = 3.5
    maximum_sag_summer: float = 5.0
    observation_month: int = 7

    wind_speed: float = 0

    rainfall: float = 0

    humidity: float = 0
    temperature: float = 0
    sunlight_hours: float = 0

    slope: float = 0
    standards: dict[str, Any] | None = None




class I1PredictionService:


    def __init__(self):

        self.predictor = TreeRiskPredictor()



    def predict_tree(self, tree):


        input_data = TreeInput(

            species=tree.species,

            height=tree.height,

            age=tree.age,

            diameter=tree.diameter,

            wind_speed=tree.wind_speed,

            rainfall=tree.rainfall,

            humidity=tree.humidity,
            temperature=tree.temperature,
            sunlight_hours=tree.sunlight_hours,

            slope=tree.slope,

            distance_to_wire=tree.distance_to_wire,
            vertical_distance=tree.vertical_distance,
            horizontal_distance=tree.horizontal_distance,
            clearance_distance=tree.clearance_distance,
            wind_offset_distance=tree.wind_offset_distance,
            voltage_level_kv=tree.voltage_level_kv,
            maximum_sag_winter=tree.maximum_sag_winter,
            maximum_sag_summer=tree.maximum_sag_summer,
            observation_month=tree.observation_month,
            standards=tree.standards,

        )


        result = self.predictor.predict(
            input_data
        )


        # ==========================
        # 关键修改
        # dataclass 转 dict
        # ==========================

        result_dict = asdict(result)



        return {


            "tree_id": tree.tree_id,


            "result": result_dict,


            # 保存数据库需要

            "species": tree.species,

            "height": tree.height,

            "age": tree.age,

            "diameter": tree.diameter,

            "wind_speed": tree.wind_speed,

            "rainfall": tree.rainfall,

            "humidity": tree.humidity,
            "temperature": tree.temperature,
            "sunlight_hours": tree.sunlight_hours,

            "slope": tree.slope,

            "distance_to_wire": tree.distance_to_wire,
            "vertical_distance": tree.vertical_distance,
            "horizontal_distance": tree.horizontal_distance,
            "clearance_distance": tree.clearance_distance,
            "wind_offset_distance": tree.wind_offset_distance,
            "voltage_level_kv": tree.voltage_level_kv,
            "maximum_sag_winter": tree.maximum_sag_winter,
            "maximum_sag_summer": tree.maximum_sag_summer,
            "observation_month": tree.observation_month,
            "standards": tree.standards,


        }
