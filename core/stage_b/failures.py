"""Small persisted failure contract for Stage-B desktop analysis."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,}\b")


def _code(technical_message: str) -> str:
    match = _CODE_PATTERN.search(technical_message)
    return match.group(0) if match else "UNEXPECTED_ANALYSIS_FAILURE"


def structured_failure(exc: Exception, *, phase: str) -> dict[str, Any]:
    """Translate a technical exception into persisted technical and user fields."""

    technical_message = f"{type(exc).__name__}: {exc}"
    code = _code(str(exc))
    upper = f"{code} {exc}".upper()

    if "GROUND" in upper or "HAG" in upper:
        stage, stage_label = "GROUND", "地面处理"
    elif "WIRE" in upper:
        stage, stage_label = "WIRE", "导线识别"
    elif "VEGETATION" in upper or "FAST_V4" in upper:
        stage, stage_label = "VEGETATION", "植被提取"
    elif "TREE" in upper:
        stage, stage_label = "TREE", "单木分割"
    elif "CLEARANCE" in upper:
        stage, stage_label = "CLEARANCE", "净空计算"
    elif "RISK" in upper:
        stage, stage_label = "RISK", "风险分析"
    elif "RECONCIL" in upper:
        stage, stage_label = "RECONCILIATION", "跨档整理"
    elif phase == "SOURCE_PREPARATION":
        stage, stage_label = "POINT_CLOUD_PREPARATION", "点云准备"
    else:
        stage, stage_label = "SPAN_ANALYSIS", "档距分析"

    if code == "FAST_GROUND_BACKEND_UNAVAILABLE":
        user_message = "地面处理组件未正确安装，当前档距尚未进入科学分析。"
        recommended_action = "请安装包含地面处理组件的新版软件后，点击“重试失败任务”。"
        recoverable = True
    elif code == "FAST_GROUND_FAILED" and "BAD ALLOCATION" in upper:
        user_message = "点云规模超过当前可用内存，地面处理未能完成。"
        recommended_action = "请关闭其他占用内存的程序后重试；若仍失败，请在更大内存的设备上处理。"
        recoverable = True
    elif "INPUT_MISSING" in upper or "SOURCE_MISSING" in upper:
        user_message = "原始点云文件当前不可读取，无法准备分析数据。"
        recommended_action = "请确认原始 LAS/LAZ 文件仍在导入位置，然后重试。"
        recoverable = True
    elif "WIRE" in upper and any(
        marker in upper for marker in ("EMPTY", "MISSING", "NO_VALID", "INSUFFICIENT")
    ):
        user_message = "当前档距未获得可确认的导线证据，分析保持未完成。"
        recommended_action = "请检查两座杆塔是否属于同一线路且为相邻杆塔，再返回杆塔选取重新确认。"
        recoverable = False
    elif any(marker in upper for marker in ("GEOMETRY", "ZERO_LENGTH", "SPAN_INPUT")):
        stage, stage_label = "SPAN_INPUT", "杆塔/档距输入"
        user_message = "当前两座杆塔之间的线路几何无法支持有效档距分析。"
        recommended_action = "请返回杆塔选取，确认两座杆塔属于同一连续线路且彼此相邻。"
        recoverable = False
    elif phase == "SOURCE_PREPARATION":
        user_message = "点云准备未能完成，当前档距尚未进入导线与树木分析。"
        recommended_action = "请点击“重试失败任务”；若仍失败，请查看详情中的故障代码。"
        recoverable = True
    else:
        user_message = "档距分析遇到软件运行错误，未生成可用结果。"
        recommended_action = "请点击“重试失败任务”；若仍失败，请查看详情中的故障代码。"
        recoverable = True

    return {
        "failure_stage": stage,
        "failure_stage_label": stage_label,
        "failure_code": code,
        "user_message": user_message,
        "technical_message": technical_message,
        "recoverable": recoverable,
        "recommended_action": recommended_action,
    }


def failure_details(diagnostics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Read a structured failure while remaining compatible with old manifests."""

    if not isinstance(diagnostics, Mapping) or not diagnostics.get("failure"):
        return None
    if all(
        key in diagnostics
        for key in (
            "failure_stage",
            "failure_code",
            "user_message",
            "technical_message",
            "recoverable",
            "recommended_action",
        )
    ):
        return {
            key: diagnostics.get(key)
            for key in (
                "failure_stage",
                "failure_stage_label",
                "failure_code",
                "user_message",
                "technical_message",
                "recoverable",
                "recommended_action",
            )
        }
    technical = str(diagnostics.get("failure"))
    return structured_failure(RuntimeError(technical), phase="SPAN_ANALYSIS")


__all__ = ["failure_details", "structured_failure"]
