"""
I1 prediction adapter

负责把 I1 中的树木信息转换成预测输入
"""

from dataclasses import dataclass


@dataclass
class I1TreeRecord:
    """
    I1中的树木记录
    """

    tree_id: int
    species: str
    height: float
    age: int
    diameter: float
    distance_to_wire: float



def convert_i1_tree(record: I1TreeRecord):
    """
    将I1树木记录转换为预测需要的数据
    """

    return {
        "tree_id": record.tree_id,
        "species": record.species,
        "height": record.height,
        "age": record.age,
        "diameter": record.diameter,

        # 当前天气数据以后从天气接口接入
        "wind_speed": 0,
        "rainfall": 0,
        "humidity": 0,
        "slope": 0,

        "distance_to_wire": record.distance_to_wire,
    }