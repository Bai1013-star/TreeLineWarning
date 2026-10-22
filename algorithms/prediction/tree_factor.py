"""
Tree factor calculation

计算树木自身结构风险
"""


def calculate_tree_factor(
    height: float,
    age: int,
    diameter: float
) -> float:
    """
    返回树体风险分
    """

    score = 0


    # 树高因素
    if height > 30:
        score += 20
    elif height > 20:
        score += 10


    # 树龄因素
    if age > 80:
        score += 20
    elif age > 50:
        score += 10


    # 胸径因素
    # 太细风险增加
    if diameter < 30:
        score += 20
    elif diameter < 50:
        score += 10


    return score