"""Run one persisted desktop project through retry, review finalization, and report gates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from desktop.workflow import DesktopWorkflow


TERMINAL_BATCH_STATES = {"COMPLETE", "FAILED", "PARTIAL"}
TERMINAL_FINALIZE_STEPS = {"DONE", "FINALIZE_FAILED"}


def _emit(event: str, **details: object) -> None:
    print(json.dumps({"event": event, **details}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    workflow = DesktopWorkflow(project_root, resource_root=Path.cwd().resolve())
    initial = workflow.batch_projection()
    _emit("initial", status=initial.get("status"), batch_id=initial.get("batch_id"))
    if not args.finalize_only:
        if initial.get("status") == "FAILED":
            workflow.retry_failed(execution_mode="SERIAL")
        elif initial.get("status") != "RUNNING":
            selected = [
                item["span_id"] for item in initial.get("spans", ()) if item.get("selected")
            ]
            workflow.start_batch(selected, mode="REAL", execution_mode="SERIAL")

    deadline = time.monotonic() + args.timeout
    previous = None
    while True:
        batch = workflow.batch_projection()
        status = str(batch.get("status") or "")
        stages = {item.get("span_id"): item.get("stage") for item in batch.get("spans", ())}
        marker = (status, tuple(stages.items()))
        if marker != previous:
            _emit("batch", status=status, stages=stages)
            previous = marker
        if status in TERMINAL_BATCH_STATES:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("analysis did not reach a terminal state")
        time.sleep(2.0)
    if status != "COMPLETE":
        _emit("failed", batch=batch)
        return 2

    workflow.start_finalize()
    while True:
        editor = workflow.editor_state()
        step = str(editor.get("step") or "")
        _emit("finalize", step=step, message=editor.get("current_message"))
        if step in TERMINAL_FINALIZE_STEPS:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("review finalization did not reach a terminal state")
        time.sleep(2.0)
    if step != "DONE":
        _emit("failed", editor=editor)
        return 3

    report = workflow.current_report()
    preview = workflow.future_preview(1)
    _emit(
        "complete",
        batch=workflow.batch_projection(),
        report_status=report.get("status"),
        report_tree_count=len(report.get("trees") or ()),
        future_preview_status=preview.get("status"),
        future_preview_tree_count=len(preview.get("trees") or ()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
