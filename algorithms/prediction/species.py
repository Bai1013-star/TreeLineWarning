"""
树种风险参数
"""


SPECIES_RISK = {

    "松树": 1.2,

    "杨树": 1.3,

    "杉树": 1.0,

    "柳树": 1.1,

    "柏树": 1.1,

    "未知": 1.0,
}


def get_species_factor(species: str) -> float:
    """
    获取树种风险系数
    """

    return SPECIES_RISK.get(
        species,
        SPECIES_RISK["未知"]
    )
