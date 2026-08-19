"""Shared run/session state across the three browser steps (/step1 /step2 /step3).

The browser is always a projection of one thread-safe ``FolderWorkflowState``
so the three pages share a single run/session.  Persisted checkpoints
(span_selection.json / tree_edits.json / tree_segmentation_original.*) let a
restarted server resume without making the user re-select or re-edit.
"""

from __future__ import annotations

import json
import threading
from typing import Any

# Workflow step names (drives which page the browser shows).
STEP_SELECTING_SPAN = "SELECTING_SPAN"
STEP_SCIENCE_RUNNING = "SCIENCE_RUNNING"
STEP_TREE_EDITING = "TREE_EDITING"
STEP_FINALIZING = "FINALIZING"
STEP_DONE = "DONE"
STEP_ERROR = "ERROR"

FOLDER_STATE_SCHEMA = "folder-to-excel-state-v1"

_STEP_LABELS = {
    STEP_SELECTING_SPAN: "STEP 1 人工选择档距",
    STEP_SCIENCE_RUNNING: "STEP 2 正在执行科学分析...",
    STEP_TREE_EDITING: "STEP 2 人工审核单木",
    STEP_FINALIZING: "STEP 3 正在计算最终净空/危险并导出...",
    STEP_DONE: "STEP 3 最终结果",
    STEP_ERROR: "失败",
}


class FolderWorkflowState:
    """Thread-safe live workflow authority shared by every route handler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.step = STEP_SELECTING_SPAN
        self.stages: list[dict[str, Any]] = []
        self.current_message = ""
        self.error: str | None = None
        self.error_detail: str | None = None
        # STEP 1
        self.selection_mode = "IDLE"  # IDLE | SELECT_A | SELECT_B
        self.clicks: dict[str, dict[str, float]] = {"A": {}, "B": {}}
        self.pole_preview: dict[str, Any] = {}
        self.selection: dict[str, Any] | None = None
        # STEP 2
        self.tree_snapshot: dict[str, Any] | None = None
        self.tree_edit_count: int = 0
        # STEP 3
        self.final_summary: dict[str, Any] | None = None

    def update(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def set_step(self, step: str, message: str = "") -> None:
        with self._lock:
            self.step = step
            self.current_message = message

    def append_stage(self, stage: int, message: str, done: bool) -> None:
        with self._lock:
            self.stages = self.stages + [
                {"stage": stage, "message": message, "done": done}
            ]

    def set_error(self, message: str, detail: str | None = None) -> None:
        with self._lock:
            self.step = STEP_ERROR
            self.error = message
            self.error_detail = detail
            self.current_message = message

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            selection_complete = bool(self.clicks["A"] and self.clicks["B"])
            distance_xy = None
            if selection_complete:
                import math

                distance_xy = math.dist(
                    (self.clicks["A"]["x"], self.clicks["A"]["y"]),
                    (self.clicks["B"]["x"], self.clicks["B"]["y"]),
                )
            return {
                "schema": FOLDER_STATE_SCHEMA,
                "step": self.step,
                "step_label": _STEP_LABELS.get(self.step, self.step),
                "stage_total": 3,
                "stages": [dict(item) for item in self.stages],
                "current_message": self.current_message,
                "error": self.error,
                "error_detail": self.error_detail,
                "selection_mode": self.selection_mode,
                "clicks": json.loads(json.dumps(self.clicks)),
                "selection_distance_xy_m": distance_xy,
                "selection_complete": selection_complete,
                "pole_preview": json.loads(json.dumps(self.pole_preview)),
                "selection": self.selection,
                "tree_snapshot": self.tree_snapshot,
                "tree_edit_count": self.tree_edit_count,
                "final_summary": self.final_summary,
            }


def _make_progress_callback(state: FolderWorkflowState):
    def _progress(stage: int, message: str, done: bool, elapsed: float | None) -> None:
        del elapsed  # V1 keeps the message simple
        state.append_stage(stage, message, done)
        if done:
            state.update(current_message="")
        else:
            state.update(current_message=f"[{stage}/3] {message}")

    return _progress


__all__ = [
    "STEP_SELECTING_SPAN",
    "STEP_SCIENCE_RUNNING",
    "STEP_TREE_EDITING",
    "STEP_FINALIZING",
    "STEP_DONE",
    "STEP_ERROR",
    "FOLDER_STATE_SCHEMA",
    "FolderWorkflowState",
    "_make_progress_callback",
]
