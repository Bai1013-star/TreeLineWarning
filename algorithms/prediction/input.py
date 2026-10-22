"""
Prediction input parser

负责把外部输入转换成标准 TreeInput
"""

from .predictor import TreeInput


def create_tree_input(data: dict) -> TreeInput:
    """
    将字典数据转换为预测输入对象
    """

    return TreeInput(
        species=data.get("species", "unknown"),
        height=float(data.get("height", 0)),
        age=int(data.get("age", 0)),
        diameter=float(data.get("diameter", 0)),

        wind_speed=float(data.get("wind_speed", 0)),
        rainfall=float(data.get("rainfall", 0)),
        humidity=float(data.get("humidity", 0)),

        slope=float(data.get("slope", 0)),
    )