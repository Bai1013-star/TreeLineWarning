"""One bounded editor task per workflow; no science or server-wide executor."""

from __future__ import annotations

import threading
import time
from functools import wraps
from typing import Any
from uuid import uuid4

_CREATION_LOCK = threading.Lock()


class EditorTaskError(RuntimeError):
    pass


class EditorTasks:
    def __init__(self, report: Any = None) -> None:
        self.lock = threading.RLock()
        self.owner: int | None = None
        self.record: dict[str, Any] = {"state": "IDLE", "status": "EDIT_IDLE"}
        self.requests: list[str] = []
        self.report = report

    def _report(self, record: dict[str, Any]) -> None:
        if callable(self.report):
            try:
                self.report(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"result", "started"}
                    }
                )
            except Exception:
                pass  # Diagnostics must never reverse a committed journal.

    def assert_available(self) -> None:
        with self.lock:
            if self.record["state"] == "RUNNING" and self.owner != threading.get_ident():
                raise EditorTaskError(
                    "EDIT_RUNNING: 当前编辑尚未提交，请等待；撤销不能取消运行中操作。"
                )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.record)
            if "started" in result:
                started = result.pop("started")
                if result["state"] == "RUNNING":
                    result["elapsed_seconds"] = time.perf_counter() - started
            return result

    def _claim(self, kind: str, request_id: str | None) -> dict[str, Any]:
        with self.lock:
            # Even the owning thread cannot submit a second background task.
            if self.record["state"] == "RUNNING":
                raise EditorTaskError("EDIT_RUNNING: 当前编辑正在运行，不能重复提交或撤销。")
            if request_id and request_id in self.requests:
                raise EditorTaskError("EDIT_DUPLICATE: 此编辑请求已提交，请查询其状态。")
            if request_id:
                self.requests = (self.requests + [request_id])[-128:]
            self.record = {
                "state": "RUNNING",
                "status": "EDIT_RUNNING",
                "kind": kind,
                "job_id": uuid4().hex,
                "request_id": request_id,
                "started": time.perf_counter(),
            }
            return self.snapshot()

    def _execute(self, kind: str, callback: Any) -> Any:
        with self.lock:
            self.owner = threading.get_ident()
        try:
            result = callback()
            state = "COMMITTED"
            status = "EDIT_COMMITTED"
            if "PREVIEW" in kind or kind in {"SELECT", "FINALIZE_START", "PROJECT_CHANGE"}:
                state, status = "IDLE", "EDIT_PREVIEW_READY" if "PREVIEW" in kind else "EDIT_IDLE"
            if kind == "CANCEL":
                state, status = "CANCELLED", "EDIT_CANCELLED"
            with self.lock:
                self.record.update(state=state, status=status, result=result)
                self.record["elapsed_seconds"] = time.perf_counter() - self.record["started"]
                self.record["timings_seconds"] = (
                    result.get("timings_seconds", {}) if isinstance(result, dict) else {}
                )
                self.owner = None
                completed = dict(self.record)
            self._report(completed)
            return result
        except Exception as exc:
            with self.lock:
                self.record.update(
                    state="FAILED",
                    status="EDIT_FAILED",
                    error={"message": str(exc), "details": f"{type(exc).__name__}: {exc}"},
                )
                self.record["elapsed_seconds"] = time.perf_counter() - self.record["started"]
                self.owner = None
                completed = dict(self.record)
            self._report(completed)
            raise

    def run(self, kind: str, callback: Any) -> Any:
        with self.lock:
            owned = self.record["state"] == "RUNNING" and self.owner == threading.get_ident()
            if not owned:
                self._claim(kind, None)
        if owned:
            return callback()
        return self._execute(kind, callback)

    def submit(self, kind: str, callback: Any, request_id: str | None) -> dict[str, Any]:
        result = self._claim(kind, request_id)

        def worker() -> None:
            try:
                self._execute(kind, callback)
            except Exception:
                # The structured failure remains available to every HTTP request.
                pass

        try:
            threading.Thread(target=worker, name="treeline-editor-task", daemon=True).start()
        except Exception as exc:
            with self.lock:
                self.record.update(
                    state="FAILED", status="EDIT_FAILED", error={"message": str(exc)}
                )
            raise
        return result


def editor_tasks(workflow: Any) -> EditorTasks:
    with _CREATION_LOCK:
        if not hasattr(workflow, "_edit_tasks"):
            workflow._edit_tasks = EditorTasks(
                lambda record: workflow._log("tree_editor_task", **record)
            )
        return workflow._edit_tasks


def editor_action(kind: str):
    def decorate(function):
        @wraps(function)
        def guarded(self, *args, **kwargs):
            return editor_tasks(self).run(kind, lambda: function(self, *args, **kwargs))

        return guarded

    return decorate
