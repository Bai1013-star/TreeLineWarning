"""Engineering screening estimate for wind-driven crown contact.

This is deliberately a scenario model, not a structural safety certificate.
It combines the measured tree/line clearance with species-dependent green-wood
stiffness and a cantilever branch approximation.  The result is reported as a
gust threshold and Beaufort force with an uncertainty band so the UI never
presents a species label as an exact material test.
"""

from __future__ import annotations

import math
from typing import Any


# Representative green-wood longitudinal modulus ranges (GPa), grouped for
# Chinese operator-facing species names.  Values are intentionally ranges: the
# USDA Wood Handbook stresses that within-species variability is substantial.
_SPECIES_E_GPA: dict[str, tuple[float, float, str]] = {
    "柳树": (5.5, 8.0, "柔软枝条"),
    "杨树": (6.5, 9.5, "较柔软枝条"),
    "泡桐": (5.0, 7.5, "较柔软枝条"),
    "桉树": (9.0, 13.0, "中等偏硬枝条"),
    "松树": (7.0, 11.0, "中等枝条"),
    "杉树": (7.0, 10.5, "中等枝条"),
    "柏树": (7.5, 11.5, "中等枝条"),
    "槐树": (10.0, 14.0, "较硬枝条"),
    "榆树": (8.5, 12.0, "中等偏硬枝条"),
    "樟树": (8.0, 12.0, "中等偏硬枝条"),
    "枫树": (8.5, 13.0, "中等偏硬枝条"),
    "栎树": (9.5, 14.0, "较硬枝条"),
    "橡树": (9.5, 14.0, "较硬枝条"),
    "其他": (7.0, 12.0, "未知树种区间"),
}


def beaufort_force(speed_m_s: float) -> int:
    """Return the standard 0--12 Beaufort force for a 10 m wind speed."""

    limits = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7)
    for force, upper in enumerate(limits):
        if speed_m_s < upper:
            return force
    return 12


def _species_profile(species: str) -> tuple[float, float, str, str]:
    raw = str(species or "其他").strip()
    # Botanical UI names such as 油松/白皮松 do not contain the literal 松树.
    # These are explicitly broad screening groups, not species material tests.
    for fragment, group in (("松", "松树"), ("杉", "杉树"), ("柏", "柏树"),
                            ("槐", "槐树"), ("柳", "柳树"), ("杨", "杨树")):
        if fragment in raw:
            values = _SPECIES_E_GPA[group]
            return values[0], values[1], values[2], group
    for name, values in _SPECIES_E_GPA.items():
        if name != "其他" and (name in raw or raw in name):
            return values[0], values[1], values[2], name
    values = _SPECIES_E_GPA["其他"]
    return values[0], values[1], values[2], "其他"


def estimate_wind_contact(
    *,
    species: str,
    clearance_m: float,
    height_m: float,
    crown_width_m: float | None,
    diameter_cm: float | None,
) -> dict[str, Any]:
    """Estimate the gust at which crown deflection consumes the clearance.

    A representative lateral branch is treated as a tapered cantilever.  Crown
    reconfiguration and dynamic amplification are represented by a wide speed
    interval rather than a false single exact value.
    """

    clearance = float(clearance_m)
    height = float(height_m)
    if not math.isfinite(clearance) or clearance < 0 or not math.isfinite(height) or height <= 0:
        raise ValueError("风致触线估算需要有效树高和非负净空")
    missing = [label for label, value in (("冠幅", crown_width_m), ("原始点云胸径或人工实测胸径", diameter_cm))
               if value is None or not math.isfinite(float(value)) or float(value) <= 0]
    if missing and clearance > 0:
        explanation = "缺少" + "、".join(missing) + "，不能用默认胸径补算接触风级。"
        return dict(status="INSUFFICIENT_EVIDENCE", species=str(species or "其他"),
                    missing_fields=missing, interpretation=explanation, explanation=explanation,
                    limitations="补充有效实测数据后再计算；未用树高公式代替胸径。")
    # Zero clearance requires no stiffness calculation and no fabricated size.
    if clearance == 0:
        return dict(status="ALREADY_CONTACT", species=str(species or "其他"),
                    beaufort_force=0, estimated_contact_gust_m_s=0.0,
                    beaufort_force_range=[0, 0], estimated_contact_gust_range_m_s=[0.0, 0.0],
                    interpretation="当前或目标情景净空为零，无需再用风速消耗净空。",
                    limitations="零净空为几何/预测情景结果；实际是否接触仍需现场复核。")
    crown = max(0.6, float(crown_width_m))
    dbh_m = max(0.04, float(diameter_cm) / 100.0)
    e_low, e_high, stiffness_label, matched = _species_profile(species)

    # Representative flexible crown branch, bounded so a very large trunk DBH
    # does not incorrectly turn the whole crown into one solid cantilever.
    length = min(max(crown * 0.55, 0.6), max(height * 0.45, 0.8), 5.0)
    if clearance >= length:
        explanation = "所需偏移已达到或超过代表枝条长度，超出当前树冠弯曲模型适用范围。"
        return dict(status="OUTSIDE_MODEL_RANGE", species=str(species or "其他"),
                    required_crown_deflection_m=round(clearance, 3),
                    representative_branch_length_m=round(length, 3),
                    interpretation=explanation, explanation=explanation,
                    limitations="不输出不合理的极端接触风速；树体倾倒、断枝或导线运动须单独现场评估。")
    branch_diameter = min(max(dbh_m * 0.12, 0.014), 0.045)
    inertia = math.pi * branch_diameter**4 / 64.0
    frontal_area = max(0.15, min(crown * length * 0.22, 5.0))
    rho_air, drag_coefficient = 1.225, 0.9
    load_coefficient = 0.5 * rho_air * drag_coefficient * frontal_area

    def speed_for(e_gpa: float, dynamic_factor: float) -> float:
        if clearance <= 0:
            return 0.0
        numerator = clearance * 3.0 * e_gpa * 1e9 * inertia
        denominator = max(load_coefficient * length**3 * dynamic_factor, 1e-12)
        return math.sqrt(numerator / denominator)

    # Gust dynamics/reconfiguration and uncertain attachment geometry dominate
    # precision.  The band spans a compliant/high-amplification case through a
    # stiff/lower-amplification case.
    lower = speed_for(e_low, 1.65)
    upper = speed_for(e_high, 0.80)
    central = speed_for((e_low + e_high) / 2.0, 1.10)
    lower = min(lower, central, upper)
    upper = max(lower, central, upper)
    force = beaufort_force(central)
    low_force = beaufort_force(lower)
    high_force = beaufort_force(upper)
    return {
        "status": "ALREADY_CONTACT" if clearance <= 0 else "SCREENING_ESTIMATE",
        "species": str(species or "其他"),
        "matched_species_group": matched,
        "stiffness_label": stiffness_label,
        "green_wood_modulus_gpa_range": [e_low, e_high],
        "required_crown_deflection_m": round(clearance, 3),
        "estimated_contact_gust_m_s": round(central, 1),
        "estimated_contact_gust_range_m_s": [round(lower, 1), round(upper, 1)],
        "beaufort_force": force,
        "beaufort_force_range": [low_force, high_force],
        "beaufort_force_basis": "GUST_SPEED_EQUIVALENT_ONLY",
        "parameter_basis": "树种组代表参数为工程假设区间，未逐树实测或按本地树种独立标定。",
        "basis": (
            f"约 {low_force}～{high_force} 级的阵风速度情景（{lower:.1f}～{upper:.1f} m/s）；"
            "风级仅用于说明该阵风速度量级，不等于气象台的平均风力等级。"
            "枝条粗细、风向和树冠变形未现场标定，不能保证该风速必定触线。"
        ),
        "references": [
            {"title": "Wood Handbook Chapter 5 Mechanical properties of wood",
             "url": "https://research.fs.usda.gov/treesearch/62244",
             "use": "木材力学性质的变异性；不直接标定本地各树种枝条参数"},
            {"title": "How we measure wind",
             "url": "https://weather.metoffice.gov.uk/guides/observations/how-we-measure-wind",
             "use": "区分平均风速和阵风；风级只作速度量级对照"},
        ],
        "interpretation": (
            "当前已接触或净空为零"
            if clearance <= 0
            else f"工程情景估算：阵风约 {central:.1f} m/s（速度量级约 {force} 级）可能使树冠扫入当前净空"
        ),
        "method": "SPECIES_GREEN_WOOD_CANTILEVER_CROWN_GUST_SCREENING_V1",
        "limitations": "未实测枝条直径、含水率、风向、根系约束和阵风谱；仅作巡检筛查，须结合现场复测。",
    }


def attach_wind_contact_estimates(forecast: dict[str, Any], observed: dict[str, Any]) -> None:
    """Use the same evidence policy for manual, external and project forecasts."""
    selected = observed.get("selected_fields")
    def measured(name: str) -> Any:
        return observed.get(name) if not isinstance(selected, list) or name in selected else None
    normalized = forecast.get("input_snapshot") or forecast.get("current_snapshot") or {}
    args = dict(species=str(observed.get("species") or "其他"),
                clearance_m=float(observed.get("clearance_distance", observed.get("distance_to_wire", normalized.get("clearance_distance", 999)))),
                height_m=float(observed["height"]),
                crown_width_m=measured("crown_width"), diameter_cm=measured("diameter"))
    forecast["wind_contact_estimate"] = estimate_wind_contact(**args)
    target = forecast.get("selected_target_prediction") or {}
    if isinstance(target.get("predicted_clearance_distance"), (int, float)):
        forecast["target_wind_contact_estimate"] = estimate_wind_contact(
            species=args["species"], clearance_m=float(target["predicted_clearance_distance"]),
            height_m=float(target.get("predicted_height") or args["height_m"]),
            crown_width_m=target.get("predicted_crown_width") if args["crown_width_m"] else None,
            diameter_cm=target.get("predicted_diameter") if args["diameter_cm"] else None,
        )


__all__ = ["beaufort_force", "estimate_wind_contact", "attach_wind_contact_estimates"]
