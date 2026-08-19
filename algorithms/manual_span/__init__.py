"""MANUAL_SINGLE_SPAN: one real span, selected by two human pole clicks.

This package is the owner-facing one-pass delivery route.  It reuses the
frozen FAST mature methods (coarse routing, Ground/HAG, span-local tree and
wire pipelines, exact Tree-Wire mathematics) behind a single local
interactive application.  A teacher double-clicks ``RUN_MANUAL_SINGLE_SPAN.bat``,
clicks two poles, presses 开始分析, later enters a clearance threshold and
presses 生成最终结果.  Everything else is automatic and span-local.

It is FAST / NON_CERTIFIED research demonstration; it never claims Formal
authority.
"""

from __future__ import annotations

MANUAL_SPAN_PACKAGE = "manual-single-span-v1"

__all__ = ["MANUAL_SPAN_PACKAGE"]
