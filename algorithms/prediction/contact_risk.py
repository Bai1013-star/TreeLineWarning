"""Hard safety rules for observed branch/line relations and clearance.

These rules are deliberately independent of the statistical and comprehensive
risk scores.  A confirmed contact or a tree taller than the conductor is an
observable field condition and must not be downgraded by a model score.
"""

from __future__ import annotations

from typing import Any

CONTACT_TARGETS = {
    "未碰到": "none",
    "未接触（按距离判断）": "none",
    "无接触": "none",
    "none": "none",
    "no_contact": "none",
    "树枝接近线路（按距离判断）": "near_line",
    "树枝接近线路": "near_line",
    "接近线路": "near_line",
    "near_line": "near_line",
    "碰到电线": "contact_line",
    "接触电线": "contact_line",
    "树枝接触线": "contact_line",
    "树枝接触线路": "contact_line",
    "接触线路": "contact_line",
    "导线": "contact_line",
    "wire": "contact_line",
    "contact_line": "contact_line",
    "树枝穿过线": "through_line",
    "树枝穿过线路": "through_line",
    "穿过线路": "through_line",
    "through": "through_line",
    "through_line": "through_line",
    "碰到螺旋避雷针": "lightning_arrester",
    "接触螺旋避雷针": "lightning_arrester",
    "螺旋避雷针": "lightning_arrester",
    "避雷针": "lightning_arrester",
    "lightning_arrester": "lightning_arrester",
    "碰到接地线": "ground_wire",
    "接触接地线": "ground_wire",
    "接地线": "ground_wire",
    "ground_wire": "ground_wire",
}


def normalize_contact_target(value: Any) -> str:
    """Return the stable contact target code used in forecast records."""
    key = str(value or "").strip().lower()
    return CONTACT_TARGETS.get(key, "none")


def contact_risk(
    *,
    contact_target: Any = "none",
    tree_height: float,
    conductor_height: float,
    clearance_distance: float | None = None,
    general_risk_distance: float = 1.0,
    safe_distance: float = 5.0,
) -> dict[str, Any]:
    """Classify branch/line risk with physical relation taking precedence.

    ``conductor_height`` is optional in older data.  It only activates the
    over-line rule when it is a positive measured/design height.
    """
    target = normalize_contact_target(contact_target)
    height = float(tree_height)
    line_height = float(conductor_height)
    general_limit = float(general_risk_distance)
    safe_limit = float(safe_distance)
    clearance = None if clearance_distance is None else max(0.0, float(clearance_distance))
    if general_limit < 0 or safe_limit < 0:
        raise ValueError("树线距离阈值不能小于 0")
    if safe_limit <= general_limit:
        raise ValueError("安全距离必须大于一般风险距离")
    triggers: list[str] = []

    common = {
        "contact_target": target,
        "tree_over_line": line_height > 0 and height > line_height,
        "clearance_distance": clearance,
        "thresholds": {
            "general_risk_max": general_limit,
            "safe_min": safe_limit,
        },
    }

    if target == "through_line":
        return {
            **common,
            "code": "danger",
            "level": "危险",
            "requires_immediate_pruning": True,
            "action": "立即设置安全隔离并由具备资质的人员处置，禁止擅自靠近或修剪",
            "basis": "已确认树枝穿过线路，物理关系优先于距离判定",
        }

    if target == "contact_line":
        return {
            **common,
            "code": "high",
            "level": "高风险",
            "requires_immediate_pruning": True,
            "action": "立即设置安全隔离，并由具备资质的人员现场复核后处置",
            "basis": "已确认树枝接触线路，物理关系优先于距离判定",
        }

    if target == "lightning_arrester":
        triggers.append("树木已接触螺旋避雷针/避雷装置")
    elif target == "ground_wire":
        triggers.append("树木已接触接地线")
    if line_height > 0 and height > line_height:
        triggers.append(f"树高 {height:g} m 已超过导线高度 {line_height:g} m")

    if triggers:
        return {
            **common,
            "code": "high",
            "level": "高风险",
            "requires_immediate_pruning": True,
            "action": "立即修剪，并由具备资质的人员现场复核后处置",
            "basis": "；".join(triggers),
        }
    if clearance is not None and clearance < safe_limit:
        close_basis = (
            f"当前净空 {clearance:g} m ≤ 一般风险距离 {general_limit:g} m，已接近线路"
            if clearance <= general_limit
            else f"当前净空 {clearance:g} m 尚未达到安全距离 {safe_limit:g} m"
        )
        return {
            **common,
            "code": "general",
            "level": "一般风险",
            "requires_immediate_pruning": False,
            "action": "安排巡检复核，并持续监测树枝与线路的净空距离",
            "basis": close_basis,
        }
    return {
        **common,
        "code": "safe",
        "level": "安全",
        "requires_immediate_pruning": False,
        "action": "未发现接触，保持常规巡检",
        "basis": (
            f"当前净空 {clearance:g} m ≥ 安全距离 {safe_limit:g} m，且未发现穿线或接触"
            if clearance is not None
            else "未发现穿线或接触；未提供净空距离，保持常规巡检"
        ),
    }
