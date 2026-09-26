"""Species-related engineering priors used by the warning workflow."""

from algorithms.prediction.beijing_species import (
    BEIJING_TREE_PROFILES,
    GROWTH_GROUPS,
)

_RISK_BY_GROWTH_GROUP = {"fast": 1.2, "medium": 1.05, "slow": 1.0, "default": 1.0}

# This coefficient is only an engineering prior in the multi-factor warning
# score.  It must not be interpreted as a standalone tree-safety assessment.
SPECIES_RISK = {
    name: _RISK_BY_GROWTH_GROUP[profile[3]] for name, profile in BEIJING_TREE_PROFILES.items()
}
SPECIES_RISK.update({"松树": 1.1, "杉树": 1.0, "柳树": 1.2, "柏树": 1.0, "未知": 1.0})

# 丰台地区现场接触风险页使用的年增高参考值（m/年）。前三项来自
# 项目方给定的本地经验；国槐、白蜡保留为可在页面中覆盖的工程初始值。
SPECIES_ANNUAL_HEIGHT_GROWTH = {
    name: dict(GROWTH_GROUPS[profile[3]]) for name, profile in BEIJING_TREE_PROFILES.items()
}
SPECIES_ANNUAL_HEIGHT_GROWTH.update(
    {
        "法国梧桐": {
            "minimum": 3.0,
            "maximum": 3.0,
            "reference": 3.0,
            "source": "丰台现场经验",
        },
        "杨树": {
            "minimum": 3.0,
            "maximum": 3.0,
            "reference": 3.0,
            "source": "丰台现场经验",
        },
        "榆树": {
            "minimum": 1.0,
            "maximum": 2.0,
            "reference": 1.5,
            "source": "丰台现场经验",
        },
        "国槐": {
            "minimum": 0.6,
            "maximum": 1.0,
            "reference": 0.8,
            "source": "工程初始值，可现场校准",
        },
        "白蜡": {
            "minimum": 0.8,
            "maximum": 1.2,
            "reference": 1.0,
            "source": "工程初始值，可现场校准",
        },
    }
)


def annual_height_growth(species: str) -> dict[str, float | str]:
    """Return the editable local annual height-growth reference."""
    return dict(
        SPECIES_ANNUAL_HEIGHT_GROWTH.get(
            species,
            dict(GROWTH_GROUPS["default"]),
        )
    )


def get_species_factor(species: str) -> float:
    """
    获取树种风险系数
    """

    return SPECIES_RISK.get(species, SPECIES_RISK["未知"])
