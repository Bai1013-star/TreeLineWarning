"""Prepare and optionally confirm one DBH review on a disposable project copy.

Example::

    python tests/evidence/accept_packaged_dbh_review.py \
      --base-url http://127.0.0.1:52534 --project-root E:/copy/project3 \
      --tree-id tree-8 --seed-original-index 12345 \
      --ground-original-index 12346 --confirm

Without --confirm the script prepares evidence and prints selectable indices.
Preparation writes a review session, so point the app at a disposable copy.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4


def _request(base_url: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        payload = json.load(exc)
        raise RuntimeError(f"{path}: HTTP {exc.code}: {payload}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: invalid response")
    return payload


def _mutation(base_url: str, path: str, body: dict) -> dict:
    request_id = f"dbh-evidence-{uuid4().hex}"
    accepted = _request(base_url, path, {**body, "async": True, "request_id": request_id})
    if accepted.get("request_id") != request_id:
        raise RuntimeError(f"{path}: request_id mismatch")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        record = _request(base_url, f"/api/editor/mutation?request_id={request_id}")
        if record.get("request_id") != request_id:
            raise RuntimeError(f"{path}: polled request_id mismatch")
        if record.get("state") in {"RUNNING", "PENDING", "QUEUED"}:
            time.sleep(0.5)
            continue
        if record.get("state") == "FAILED":
            raise RuntimeError(f"{path}: {record}")
        result = record.get("result")
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError(f"{path}: unsuccessful terminal result: {record}")
        return result
    raise TimeoutError(f"{path}: no terminal result within 600 seconds; do not retry blindly")


def _select(review: dict, *, candidate_id: str | None,
            seed_index: int | None, ground_index: int | None) -> dict:
    if review.get("confirmation_allowed") is False:
        raise ValueError("已发布胸径可查看证据，但不可重复人工确认")
    display = review.get("display") or {}
    original = display.get("original_index") or []
    selectable = display.get("selectable") or []
    ground_selectable = display.get("ground_selectable") or [False] * len(original)
    if not (len(original) == len(selectable) == len(ground_selectable)):
        raise ValueError("复核点与选择标志长度不一致")
    body: dict = {"review_id": review["review_id"], "identity_confirmed": True}
    if candidate_id is not None:
        candidates = {str(row.get("candidate_id")) for row in review.get("candidates") or []}
        if candidate_id not in candidates:
            raise ValueError(f"候选轴 {candidate_id} 不在本次复核证据中")
        body["candidate_id"] = candidate_id
    elif seed_index is not None:
        try:
            row = original.index(seed_index)
        except ValueError as exc:
            raise ValueError("树干点不在本次显示的精确原始点中") from exc
        if selectable[row] is not True:
            raise ValueError("树干点未被后台标记为可选")
        body["seed_original_index"] = seed_index
    else:
        raise ValueError("必须指定候选轴或树干原始点")
    if ground_index is not None:
        try:
            row = original.index(ground_index)
        except ValueError as exc:
            raise ValueError("树根地面点不在本次显示的精确原始点中") from exc
        if ground_selectable[row] is not True:
            raise ValueError("树根地面点未被后台标记为可选")
        body["ground_original_index"] = ground_index
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tree-id", required=True, help="persistent_tree_id")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--candidate-id")
    selector.add_argument("--seed-original-index", type=int)
    parser.add_argument("--ground-original-index", type=int)
    parser.add_argument("--confirm", action="store_true", help="明确提交所选原始证据")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if project_root == Path("E:/1/3").resolve():
        parser.error("原始项目 E:/1/3 只读；请先在应用中打开验收副本")
    workspace = _request(args.base_url, "/api/desktop/workspace")
    active_root = Path(workspace["project"]["root"]).resolve()
    if active_root != project_root:
        parser.error(f"应用当前项目为 {active_root}，与验收副本 {project_root} 不一致")
    trees = _request(args.base_url, "/api/dbh-review/trees")["trees"]
    if not any(str(row.get("persistent_tree_id")) == args.tree_id for row in trees):
        parser.error(f"当前项目不存在持久树 ID {args.tree_id}")
    prepared = _mutation(args.base_url, "/api/dbh-review/prepare", {
        "persistent_tree_id": args.tree_id,
    })
    review = prepared["review"]
    display = review.get("display") or {}
    original = display.get("original_index") or []
    seed = [
        index for index, allowed in zip(
            original, display.get("selectable") or [], strict=True
        ) if allowed
    ]
    ground = [
        index for index, allowed in zip(
            original, display.get("ground_selectable") or [], strict=True
        ) if allowed
    ]
    evidence = {
        "project_root": str(project_root),
        "persistent_tree_id": args.tree_id,
        "review_id": review.get("review_id"),
        "confirmation_allowed": review.get("confirmation_allowed"),
        "candidate_ids": [row.get("candidate_id") for row in review.get("candidates") or []],
        "selectable_seed_count": len(seed),
        "selectable_seed_sample": seed[:20],
        "selectable_ground_count": len(ground),
        "selectable_ground_sample": ground[:20],
        "wide_raw_stem_search": review.get("wide_raw_stem_search"),
    }
    if args.confirm:
        try:
            selection = _select(review, candidate_id=args.candidate_id,
                                seed_index=args.seed_original_index,
                                ground_index=args.ground_original_index)
        except ValueError as exc:
            parser.error(str(exc))
        before = project_root / "exports" / "树木风险结果.xlsx"
        before_mtime = before.stat().st_mtime_ns if before.is_file() else None
        outcome = _mutation(args.base_url, "/api/dbh-review/confirm", selection)
        evidence["confirm"] = outcome
        rows = _request(args.base_url, "/api/dbh-review/trees")["trees"]
        current = next(row for row in rows if str(row.get("persistent_tree_id")) == args.tree_id)
        evidence["published_tree"] = {
            "dbh_cm": current.get("dbh_cm"), "dbh_method": current.get("dbh_method"),
            "dbh_fit_status": current.get("dbh_fit_status"),
        }
        evidence["delivery_excel_refreshed"] = (
            before.is_file() and (before_mtime is None or before.stat().st_mtime_ns > before_mtime)
        )
        if outcome.get("status") == "RECOVERED":
            if current.get("dbh_cm") is None or current.get("dbh_method") == "REFERENCE_ESTIMATE":
                raise AssertionError("已恢复胸径未发布到当前树列表")
            if not evidence["delivery_excel_refreshed"]:
                raise AssertionError("已恢复胸径后交付 Excel 未刷新")
    print(json.dumps(evidence, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
