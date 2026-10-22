"""
Prediction adapter

负责把 I1 / 植被数据转换为预测模型输入
"""

from dataclasses import dataclass

from .predictor import TreeInput


@dataclass
class PredictionInput:
    """
    预测输入标准格式
    """

    tree_id: int
    tree_input: TreeInput


def from_tree_data(
    tree_id: int,
    species: str,
    height: float,
    age: int,
    diameter: float,
    wind_speed: float,
    rainfall: float,
    humidity: float,
    slope: float,
) -> PredictionInput:
    """
    将系统内部数据转换为预测输入
    """

    tree = TreeInput(
        species=species,
        height=height,
        age=age,
        diameter=diameter,
        wind_speed=wind_speed,
        rainfall=rainfall,
        humidity=humidity,
        slope=slope,
    )

    return PredictionInput(
        tree_id=tree_id,
        tree_input=tree,
    )