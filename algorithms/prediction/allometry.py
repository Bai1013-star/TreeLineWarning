"""Traceable domestic tree growth and biomass formula catalog.

The catalog deliberately keeps regional scope, source and proxy status beside
each formula.  It provides engineering estimates, not a replacement for local
re-measurement or electrical-safety decisions.
"""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log1p
from typing import Any


@dataclass(frozen=True)
class FormulaModel:
    model_id: str
    species: str
    scientific_name: str
    target: str
    formula: str
    formula_kind: str
    parameters: dict[str, float]
    unit: str
    source_ref: str
    source_grade: str
    region: str
    context: str
    d_min: float | None = None
    d_max: float | None = None
    age_min: float | None = None
    age_max: float | None = None
    proxy: bool = False
    notes: str = ""


SPECIES_SCIENTIFIC_NAMES = {
    "悬铃木": "Platanus × acerifolia",
    "杨树": "Populus spp.",
    "榆树": "Ulmus pumila",
    "国槐": "Styphnolobium japonicum",
    "白蜡": "Fraxinus chinensis",
    "油松": "Pinus tabuliformis",
    "白皮松": "Pinus bungeana",
    "侧柏": "Platycladus orientalis",
    "毛白杨": "Populus tomentosa",
    "臭椿": "Ailanthus altissima",
    "旱柳": "Salix matsudana",
    "栾树": "Koelreuteria paniculata",
    "楸树": "Catalpa bungei",
    "银杏": "Ginkgo biloba",
    "元宝枫": "Acer truncatum",
    "丁香": "Syringa spp.",
    "刺槐": "Robinia pseudoacacia",
    "龙爪槐": "Styphnolobium japonicum f. pendula",
    "樱花": "Cerasus spp.",
    "海棠": "Malus spectabilis",
    "碧桃": "Prunus persica f. duplex",
    "紫叶李": "Prunus cerasifera f. atropurpurea",
    "大叶白蜡": "Fraxinus rhynchophylla",
    "欧洲白榆": "Ulmus laevis",
    "夏橡": "Quercus robur",
    "樟子松": "Pinus sylvestris",
    "垂柳": "Salix babylonica",
    "复叶槭": "Acer negundo",
    "火炬树": "Rhus typhina",
    "暴马丁香": "Syringa reticulata",
    "山楂": "Crataegus pinnatifida",
    "山桃": "Amygdalus davidiana",
}

ALIASES = {
    "法国梧桐": "悬铃木",
    "二球悬铃木": "悬铃木",
    "英桐": "悬铃木",
    "泡桐": "悬铃木",
    "白蜡属": "白蜡",
}


# 首页树龄填写使用的工程经验参数。这里的估算明确同时使用胸径 D（cm）和
# 树高 H（m），以便在没有种植档案或年轮样本时，给巡检记录提供一个可直接
# 填入的初始树龄。年增高优先采用本项目已确认的丰台经验，胸径增量为城市绿化
# 工程初始值，后续可用本地连续复测数据校正。
EMPIRICAL_AGE_PROFILES: dict[str, dict[str, float | str]] = {
    "悬铃木": {"height_increment": 3.0, "dbh_increment": 2.2, "source": "丰台现场经验 + 工程胸径增量初始值"},
    "杨树": {"height_increment": 3.0, "dbh_increment": 2.0, "source": "丰台现场经验 + 工程胸径增量初始值"},
    "榆树": {"height_increment": 1.5, "dbh_increment": 1.2, "source": "丰台现场经验 + 工程胸径增量初始值"},
    "国槐": {"height_increment": 0.8, "dbh_increment": 1.0, "source": "工程初始值，可用本地复测校准"},
    "白蜡": {"height_increment": 1.0, "dbh_increment": 1.3, "source": "工程初始值，可用本地复测校准"},
}
DEFAULT_EMPIRICAL_AGE_PROFILE: dict[str, float | str] = {
    "height_increment": 0.7,
    "dbh_increment": 0.8,
    "source": "通用城市乔木工程初始值，建议用本地复测校准",
}


def _d_power(
    model_id: str,
    species: str,
    a: float,
    b: float,
    *,
    source_ref: str,
    source_grade: str,
    region: str,
    context: str,
    d_min: float | None = None,
    d_max: float | None = None,
    proxy: bool = False,
    notes: str = "",
    target: str = "aboveground_biomass",
    unit: str = "kg/株",
) -> FormulaModel:
    return FormulaModel(
        model_id,
        species,
        SPECIES_SCIENTIFIC_NAMES.get(species, "未标注"),
        target,
        f"W = {a} × D^{b}",
        "d_power",
        {"a": a, "b": b},
        unit,
        source_ref,
        source_grade,
        region,
        context,
        d_min,
        d_max,
        proxy=proxy,
        notes=notes,
    )


def _d2h_power(
    model_id: str,
    species: str,
    a: float,
    b: float,
    *,
    source_ref: str,
    source_grade: str,
    region: str,
    context: str,
    d_min: float | None = None,
    d_max: float | None = None,
    proxy: bool = False,
    notes: str = "",
    target: str = "aboveground_biomass",
    unit: str = "kg/株",
) -> FormulaModel:
    return FormulaModel(
        model_id,
        species,
        SPECIES_SCIENTIFIC_NAMES.get(species, "未标注"),
        target,
        f"W = {a} × (D² × H)^{b}",
        "d2h_power",
        {"a": a, "b": b},
        unit,
        source_ref,
        source_grade,
        region,
        context,
        d_min,
        d_max,
        proxy=proxy,
        notes=notes,
    )


BEIJING_SOURCE = "DB11/T 2468-2025《城市绿地碳汇计量监测技术规范》"
SHIHEZI_SOURCE = "Zhang et al., PLOS One, 2026（石河子城市树总生物量）"
JILIN_SOURCE = "温雪香等，吉林省10个主要树种（组）单木胸径生长模型，2025"
GB_SOURCE = "GB/T 43648-2024《主要树种立木生物量模型与碳计量参数》"


BEIJING_FORMULAS = (
    _d_power(
        "BJ2025_YOUSONG_AGB_D",
        "油松",
        0.1138,
        2.2461,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=4.5,
        d_max=34.5,
    ),
    _d2h_power(
        "BJ2025_YOUSONG_AGB_D2H",
        "油松",
        0.1179,
        0.8150,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=4.5,
        d_max=34.5,
    ),
    _d_power(
        "BJ2025_BAIPISONG_AGB_D",
        "白皮松",
        0.1650,
        2.0780,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.6,
        d_max=28.7,
    ),
    _d2h_power(
        "BJ2025_BAIPISONG_AGB_D2H",
        "白皮松",
        0.2410,
        0.7270,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.6,
        d_max=28.7,
    ),
    _d_power(
        "BJ2025_CEBAI_AGB_D",
        "侧柏",
        0.2300,
        1.9190,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=3.3,
        d_max=33.0,
    ),
    _d2h_power(
        "BJ2025_CEBAI_AGB_D2H",
        "侧柏",
        0.1740,
        0.7290,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=3.3,
        d_max=33.0,
    ),
    _d_power(
        "BJ2025_MAOBIYANG_AGB_D",
        "毛白杨",
        0.1079,
        2.4104,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=9.1,
        d_max=42.0,
    ),
    _d2h_power(
        "BJ2025_MAOBIYANG_AGB_D2H",
        "毛白杨",
        0.0552,
        0.9040,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=9.1,
        d_max=42.0,
        notes="适用说明包含杨属乔木；普通“杨树”使用时应标记为同组代理。",
    ),
    _d_power(
        "BJ2025_CHOUCHUN_AGB_D",
        "臭椿",
        0.1420,
        2.3027,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=9.6,
        d_max=41.4,
    ),
    _d2h_power(
        "BJ2025_CHOUCHUN_AGB_D2H",
        "臭椿",
        0.0469,
        0.9520,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=9.6,
        d_max=41.4,
    ),
    _d_power(
        "BJ2025_HANLIU_AGB_D",
        "旱柳",
        0.0071,
        3.1094,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=11.8,
        d_max=48.3,
    ),
    _d2h_power(
        "BJ2025_HANLIU_AGB_D2H",
        "旱柳",
        0.0038,
        1.1815,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=11.8,
        d_max=48.3,
    ),
    _d_power(
        "BJ2025_LUANSHU_AGB_D",
        "栾树",
        0.1843,
        2.2285,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=8.4,
        d_max=32.4,
    ),
    _d2h_power(
        "BJ2025_LUANSHU_AGB_D2H",
        "栾树",
        0.0797,
        0.9172,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=8.4,
        d_max=32.4,
    ),
    _d_power(
        "BJ2025_QIUSHU_AGB_D",
        "楸树",
        0.0918,
        2.3453,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=10.3,
        d_max=30.3,
    ),
    _d2h_power(
        "BJ2025_QIUSHU_AGB_D2H",
        "楸树",
        0.0442,
        0.9411,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=10.3,
        d_max=30.3,
    ),
    _d_power(
        "BJ2025_GUOHUAI_AGB_D",
        "国槐",
        0.0758,
        2.5904,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=11.2,
        d_max=41.0,
    ),
    _d2h_power(
        "BJ2025_GUOHUAI_AGB_D2H",
        "国槐",
        0.0122,
        1.1390,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=11.2,
        d_max=41.0,
    ),
    _d_power(
        "BJ2025_YINXING_AGB_D",
        "银杏",
        0.1170,
        2.2118,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=9.8,
        d_max=30.6,
    ),
    _d2h_power(
        "BJ2025_YINXING_AGB_D2H",
        "银杏",
        0.01821,
        1.0166,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=9.8,
        d_max=30.6,
    ),
    _d_power(
        "BJ2025_XUANLINGMU_AGB_D",
        "悬铃木",
        0.0356,
        2.7610,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=12.9,
        d_max=34.1,
    ),
    _d2h_power(
        "BJ2025_XUANLINGMU_AGB_D2H",
        "悬铃木",
        0.0093,
        1.1273,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=12.9,
        d_max=34.1,
    ),
    _d_power(
        "BJ2025_YUANBAOFENG_AGB_D",
        "元宝枫",
        0.1268,
        2.3146,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=8.6,
        d_max=23.3,
    ),
    _d2h_power(
        "BJ2025_YUANBAOFENG_AGB_D2H",
        "元宝枫",
        0.0355,
        1.0186,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=8.6,
        d_max=23.3,
    ),
    _d_power(
        "BJ2025_DINGXIANG_AGB_D",
        "丁香",
        1.0860,
        1.2240,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.9,
        d_max=10.5,
    ),
    _d2h_power(
        "BJ2025_DINGXIANG_AGB_D2H",
        "丁香",
        0.8450,
        0.4840,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.9,
        d_max=10.5,
    ),
    _d_power(
        "BJ2025_CIHUAI_AGB_D",
        "刺槐",
        0.4280,
        1.9730,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=10.5,
        d_max=45.2,
    ),
    _d2h_power(
        "BJ2025_CIHUAI_AGB_D2H",
        "刺槐",
        0.1704,
        0.8020,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=10.5,
        d_max=45.2,
    ),
    _d_power(
        "BJ2025_LONGZHAOHUAI_AGB_D",
        "龙爪槐",
        0.0420,
        2.5230,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=6.6,
        d_max=14.5,
    ),
    _d2h_power(
        "BJ2025_LONGZHAOHUAI_AGB_D2H",
        "龙爪槐",
        0.0480,
        0.9950,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=6.6,
        d_max=14.5,
    ),
    _d_power(
        "BJ2025_YINGHUA_AGB_D",
        "樱花",
        0.2060,
        1.5650,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=5.1,
        d_max=21.5,
    ),
    _d2h_power(
        "BJ2025_YINGHUA_AGB_D2H",
        "樱花",
        0.1050,
        0.7260,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=5.1,
        d_max=21.5,
    ),
    _d_power(
        "BJ2025_HAITANG_AGB_D",
        "海棠",
        0.8760,
        1.4780,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=1.8,
        d_max=8.2,
    ),
    _d2h_power(
        "BJ2025_HAITANG_AGB_D2H",
        "海棠",
        0.7290,
        0.5620,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=1.8,
        d_max=8.2,
    ),
    _d_power(
        "BJ2025_BITAO_AGB_D",
        "碧桃",
        0.1093,
        2.2870,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.4,
        d_max=13.3,
    ),
    _d2h_power(
        "BJ2025_BITAO_AGB_D2H",
        "碧桃",
        0.0453,
        0.9949,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.4,
        d_max=13.3,
    ),
    _d_power(
        "BJ2025_ZIYELI_AGB_D",
        "紫叶李",
        0.0474,
        2.6510,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.1,
        d_max=13.7,
    ),
    _d2h_power(
        "BJ2025_ZIYELI_AGB_D2H",
        "紫叶李",
        0.0335,
        1.0475,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=2.1,
        d_max=13.7,
    ),
    _d2h_power(
        "BJ2025_BAILASHU_PROXY_AGB_D2H",
        "白蜡",
        0.1704,
        0.8020,
        source_ref=BEIJING_SOURCE,
        source_grade="A2",
        region="北京市",
        context="城市绿地",
        d_min=10.5,
        d_max=45.2,
        proxy=True,
        notes="北京标准中的刺槐同组代理，非 Fraxinus chinensis 专属式。",
    ),
)

SHIHEZI_FORMULAS = tuple(
    _d2h_power(
        f"SHZ2026_{species}_TOTAL",
        species,
        a,
        b,
        source_ref=SHIHEZI_SOURCE,
        source_grade="B",
        region="新疆石河子",
        context="城市绿地",
        target="total_biomass",
        unit="kg/株",
        notes="干旱大陆性城市样本；跨地区使用应以本地数据校准。",
    )
    for species, a, b in (
        ("大叶白蜡", 0.153, 0.921),
        ("白蜡", 0.389, 0.810),
        ("欧洲白榆", 0.088, 0.911),
        ("夏橡", 0.282, 0.818),
        ("榆树", 0.101, 0.935),
        ("樟子松", 0.322, 0.798),
        ("垂柳", 0.105, 0.928),
        ("刺槐", 0.163, 0.877),
        ("复叶槭", 0.499, 0.774),
        ("火炬树", 0.086, 0.961),
        ("暴马丁香", 0.262, 0.882),
        ("山楂", 0.359, 0.824),
        ("山桃", 0.065, 1.008),
        ("海棠", 0.627, 0.734),
    )
)

JILIN_AGE_FORMULAS = tuple(
    FormulaModel(
        f"JL2025_{species}_AGE_DBH",
        species,
        SPECIES_SCIENTIFIC_NAMES.get(species, "树种组"),
        "dbh_from_age",
        "D(t) = a × [1 - exp(-b × t)]^c",
        "richards_age",
        {"a": a, "b": b, "c": c},
        "cm",
        JILIN_SOURCE,
        "B",
        "吉林省",
        "古树/大树调查",
        d_min=d_min,
        d_max=d_max,
        age_min=age_min,
        age_max=age_max,
        notes="树龄—胸径区域模型，不应自动外推到其他气候区或速生无性系。",
    )
    for species, a, b, c, age_min, age_max, d_min, d_max in (
        ("云杉", 160, 0.002276, 1.0090, 6, 327, 2.2, 86.9),
        ("冷杉", 370, 0.000464, 0.8150, 10, 490, 1.8, 108.3),
        ("红松", 160, 0.000628, 0.5595, 4, 485, 1.5, 73.8),
        ("落叶松", 220, 0.002493, 1.1821, 6, 320, 1.6, 108.2),
        ("蒙古栎", 160, 0.001034, 0.6498, 5, 1012, 1.7, 145.2),
        ("桦木", 150, 0.000970, 0.5297, 6, 640, 1.7, 115.9),
        ("榆树", 150, 0.003145, 0.9120, 7, 750, 2.5, 173.0),
        ("椴树", 330, 0.000271, 0.6260, 4, 650, 1.6, 129.0),
        ("杨树", 220, 0.003984, 1.2312, 7, 480, 2.0, 246.8),
        ("水胡黄", 110, 0.005301, 1.3038, 100, 412, 15.3, 120.0),
    )
)

GB_POPLAR_FORMULAS = tuple(
    FormulaModel(
        f"GB43648_POPLAR_{code}_AGB_D2H",
        "杨树",
        "Populus spp.",
        "aboveground_biomass",
        f"M_A = {a2} × D^{b2} × H^{c2}",
        "d_h_power",
        {"a": a2, "b": b2, "c": c2},
        "kg/株",
        GB_SOURCE,
        "A1",
        region,
        context,
        d_min=5.0,
        notes="适用于对应区域与天然/人工总体，D≥5 cm。",
    )
    for code, region, context, a2, b2, c2 in (
        ("I", "黑吉辽蒙", "天然山杨", 0.04502, 2.12223, 0.59163),
        ("II", "陕甘青宁新", "天然山杨", 0.04607, 2.14892, 0.59163),
        ("III", "川滇藏", "天然山杨", 0.05843, 2.05186, 0.59163),
        ("IV", "黑吉辽蒙", "人工杨树", 0.05559, 2.00861, 0.59163),
        ("V", "鲁豫蒙冀晋京", "人工杨树", 0.04703, 2.12487, 0.59163),
        ("VI", "苏徽鄂湘渝贵", "人工杨树", 0.02933, 2.27634, 0.59163),
    )
)

FORMULA_CATALOG = BEIJING_FORMULAS + SHIHEZI_FORMULAS + JILIN_AGE_FORMULAS + GB_POPLAR_FORMULAS


def canonical_species(species: str) -> str:
    name = str(species or "").strip()
    return ALIASES.get(name, name)


def available_species() -> list[str]:
    return sorted({model.species for model in FORMULA_CATALOG} | set(ALIASES))


def _matches_species(model: FormulaModel, selected: str) -> bool:
    if model.species == selected:
        return True
    return selected == "杨树" and model.species == "毛白杨"


def _estimate(
    model: FormulaModel, diameter_cm: float | None, height_m: float | None, age_years: float | None
) -> float | None:
    p = model.parameters
    if model.formula_kind == "d_power" and diameter_cm is not None:
        return p["a"] * diameter_cm ** p["b"]
    if model.formula_kind == "d2h_power" and diameter_cm is not None and height_m is not None:
        return p["a"] * (diameter_cm**2 * height_m) ** p["b"]
    if model.formula_kind == "d_h_power" and diameter_cm is not None and height_m is not None:
        return p["a"] * diameter_cm ** p["b"] * height_m ** p["c"]
    if model.formula_kind == "richards_age" and age_years is not None:
        return p["a"] * (1 - exp(-p["b"] * age_years)) ** p["c"]
    return None


def _warnings(model: FormulaModel, diameter_cm: float | None, age_years: float | None) -> list[str]:
    warnings: list[str] = []
    if model.proxy:
        warnings.append("该模型为同组代理，不是该物种的专属拟合式。")
    if diameter_cm is not None and model.d_min is not None and diameter_cm < model.d_min:
        warnings.append(f"胸径 {diameter_cm:g} cm 小于原模型下限 {model.d_min:g} cm，属于外推。")
    if diameter_cm is not None and model.d_max is not None and diameter_cm > model.d_max:
        warnings.append(f"胸径 {diameter_cm:g} cm 大于原模型上限 {model.d_max:g} cm，属于外推。")
    if age_years is not None and model.age_min is not None and age_years < model.age_min:
        warnings.append(f"树龄 {age_years:g} 年小于原模型下限 {model.age_min:g} 年，属于外推。")
    if age_years is not None and model.age_max is not None and age_years > model.age_max:
        warnings.append(f"树龄 {age_years:g} 年大于原模型上限 {model.age_max:g} 年，属于外推。")
    if model.notes:
        warnings.append(model.notes)
    return warnings


def calculate_species_parameters(
    *,
    species: str,
    diameter_cm: float | None = None,
    height_m: float | None = None,
    age_years: float | None = None,
    region: str = "北京丰台",
) -> dict[str, Any]:
    """Evaluate applicable catalog formulas and return fully traceable results."""
    selected = canonical_species(species)
    if not selected:
        raise ValueError("树种不能为空")
    for label, value in (("胸径", diameter_cm), ("树高", height_m), ("树龄", age_years)):
        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError(f"{label}必须为大于 0 的有限数值")
    matches = [model for model in FORMULA_CATALOG if _matches_species(model, selected)]
    results: list[dict[str, Any]] = []
    for model in matches:
        estimate = _estimate(model, diameter_cm, height_m, age_years)
        row = asdict(model)
        row["input_complete"] = estimate is not None
        row["estimate"] = round(estimate, 3) if estimate is not None else None
        row["warnings"] = _warnings(model, diameter_cm, age_years)
        row["extrapolated"] = any("外推" in warning for warning in row["warnings"])
        results.append(row)
    region_prefix = "北京市" if "北京" in region else ""
    results.sort(
        key=lambda item: (
            not item["input_complete"],
            item["proxy"],
            not item["region"].startswith(region_prefix) if region_prefix else False,
            item["model_id"],
        )
    )
    return {
        "species": species,
        "canonical_species": selected,
        "scientific_name": SPECIES_SCIENTIFIC_NAMES.get(selected, "未标注"),
        "region": region,
        "inputs": {"diameter_cm": diameter_cm, "height_m": height_m, "age_years": age_years},
        "formula_count": len(results),
        "calculable_count": sum(item["input_complete"] for item in results),
        "results": results,
        "method_note": "公式用于尺寸/生物量估算；树线接触高风险仍由现场接触与超线硬规则优先决定。",
    }


def estimate_tree_age(
    *,
    species: str,
    diameter_cm: float,
    height_m: float | None = None,
    region: str = "北京丰台",
) -> dict[str, Any]:
    """Return a directly usable screening age from DBH and height.

    The primary result is a transparent engineering fusion of the tree's
    diameter and height growth ages.  Traceable DBH-only Richards curves, when
    available, are returned as a reference rather than replacing the combined
    estimate.
    """
    selected = canonical_species(species)
    if not selected:
        raise ValueError("树种不能为空")
    for label, value in (("胸径", diameter_cm), ("树高", height_m)):
        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError(f"{label}必须为大于 0 的有限数值")

    profile = EMPIRICAL_AGE_PROFILES.get(selected, DEFAULT_EMPIRICAL_AGE_PROFILE)
    height_increment = float(profile["height_increment"])
    dbh_increment = float(profile["dbh_increment"])
    empirical_result: dict[str, Any] | None = None
    if height_m is not None:
        diameter_age = diameter_cm / dbh_increment
        height_age = height_m / height_increment
        empirical_age = 0.60 * diameter_age + 0.40 * height_age
        empirical_result = {
            "model_id": f"EMPIRICAL_{selected}_DBH_HEIGHT_AGE",
            "formula": "t = 0.60 × (D / g_D) + 0.40 × (H / g_H)",
            "formula_explanation": (
                f"D={diameter_cm:g} cm，g_D={dbh_increment:g} cm/年；"
                f"H={height_m:g} m，g_H={height_increment:g} m/年"
            ),
            "source_ref": str(profile["source"]),
            "region": region,
            "estimate_age_years": round(empirical_age, 1),
            "calculable": True,
            "extrapolated": False,
            "warnings": [
                "经验树龄用于巡检初始填写；种植档案、年轮实测或本地连续胸径资料优先。"
            ],
        }

    models = [
        model
        for model in FORMULA_CATALOG
        if _matches_species(model, selected) and model.formula_kind == "richards_age"
    ]
    height_note = (
        "树高和胸径已同时参与经验树龄：胸径年龄权重 60%，树高年龄权重 40%。"
        if height_m is not None
        else "未输入树高；暂不能生成树高 + 胸径融合的经验树龄。"
    )
    results: list[dict[str, Any]] = [empirical_result] if empirical_result else []
    for model in models:
        p = model.parameters
        warnings = _warnings(model, diameter_cm, None)
        asymptote = p["a"]
        if diameter_cm >= asymptote:
            results.append(
                {
                    "model_id": model.model_id,
                    "formula": model.formula,
                    "source_ref": model.source_ref,
                    "region": model.region,
                    "estimate_age_years": None,
                    "calculable": False,
                    "warnings": warnings
                    + [
                        f"胸径 {diameter_cm:g} cm 不小于模型渐近胸径 {asymptote:g} cm，"
                        "该曲线不能稳定反推树龄。"
                    ],
                }
            )
            continue
        age = -log1p(-(diameter_cm / asymptote) ** (1 / p["c"])) / p["b"]
        extrapolated = bool(
            (model.d_min is not None and diameter_cm < model.d_min)
            or (model.d_max is not None and diameter_cm > model.d_max)
        )
        if extrapolated:
            warnings.append("输入胸径超出原样本范围，树龄结果属于外推，仅供筛查参考。")
        results.append(
            {
                "model_id": model.model_id,
                "formula": model.formula,
                "inverse_formula": "t = -ln[1 - (D/a)^(1/c)] / b",
                "source_ref": model.source_ref,
                "region": model.region,
                "estimate_age_years": round(age, 1),
                "calculable": True,
                "extrapolated": extrapolated,
                "warnings": warnings,
            }
        )
    reference_estimates = [
        item["estimate_age_years"]
        for item in results
        if item["calculable"] and item["model_id"].startswith("JL2025_")
    ]
    estimate = (
        empirical_result["estimate_age_years"]
        if empirical_result is not None
        else reference_estimates[0]
        if reference_estimates
        else None
    )
    return {
        "species": species,
        "canonical_species": selected,
        "region": region,
        "inputs": {"diameter_cm": diameter_cm, "height_m": height_m},
        "status": "estimated" if estimate is not None else "requires_height",
        "estimate_age_years": estimate,
        "results": results,
        "height_note": height_note,
        "message": (
            "已按树高 + 胸径经验融合公式生成可直接填写的树龄；"
            "该值不等同年轮实测，城市修剪、立地和树种亚型会带来偏差。"
            if estimate is not None
            else "请同时输入树高和胸径后再估算；种植档案和年轮实测优先。"
        ),
    }


def formula_catalog_summary() -> dict[str, Any]:
    return {
        "species": available_species(),
        "formula_count": len(FORMULA_CATALOG),
        "sources": [BEIJING_SOURCE, SHIHEZI_SOURCE, JILIN_SOURCE, GB_SOURCE],
    }
