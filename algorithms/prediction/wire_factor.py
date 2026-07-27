"""
输电线路距离风险计算模块

根据树木距离电线的距离
计算线路影响风险
"""


def calculate_wire_factor(distance_to_wire: float) -> int:
    """
    计算树木靠近电线的风险分数

    参数:
        distance_to_wire:
            树木距离电线距离(m)

    返回:
        风险分数
    """


    # 非法距离
    if distance_to_wire < 0:
        raise ValueError("distance_to_wire cannot be negative")


    # 非常危险
    if distance_to_wire <= 5:
        return 40


    # 较危险
    elif distance_to_wire <= 10:
        return 25


    # 需要关注
    elif distance_to_wire <= 15:
        return 10


    # 安全范围
    else:
        return 0