"""FOLDER_TO_EXCEL V1 entrypoint.

Usage::

    python scripts/run_folder_to_excel.py [--workspace <workspace>]
        [--port 0] [--no-open] [--threshold <m>]
        [--pole-a X Y Z --pole-b X Y Z]   (headless smoke, no browser)

    The danger threshold defaults to the canonical project risk standard
    (algorithms.spans.project_adapter.RISK_THRESHOLD_M); it is never a
    Folder-pipeline-specific hard-coded default.

A workspace must contain exactly one point cloud under
``input/pointcloud/``; otherwise it fails closed with a clear Chinese error.

Interactive (default): starts the local three-step browser workflow
(STEP 1 select span -> STEP 2 review trees -> STEP 3 Excel).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("需要 X Y Z 三个坐标值")
    return (parts[0], parts[1], parts[2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOLDER_TO_EXCEL 文件夹到 Excel V1")
    parser.add_argument(
        "--workspace", type=Path, default=None,
        help="工作区目录（默认：仓库下的 workspace）",
    )
    parser.add_argument("--port", type=int, default=0, help="服务器端口（0=自动）")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="危险净空阈值（m；默认取权威标准 algorithms.spans.project_adapter.RISK_THRESHOLD_M）",
    )
    parser.add_argument(
        "--reference-dbh-cm", type=float, default=None,
        help="已停用；跨树均值和统一参考值不能补齐逐树胸径",
    )
    parser.add_argument("--pole-a", type=_parse_xyz, default=None,
                        help="headless：杆塔 A 坐标（X Y Z）")
    parser.add_argument("--pole-b", type=_parse_xyz, default=None,
                        help="headless：杆塔 B 坐标（X Y Z）")
    args = parser.parse_args(argv)
    if args.reference_dbh_cm is not None:
        parser.error(
            "--reference-dbh-cm 已停用：胸径须依据本树点云测量或复核；"
            "无法测得的树会在 Excel 中留空并注明原因，不能用跨树均值或统一参考值补齐。"
        )

    from algorithms.folder_pipeline.state import STEP_DONE, STEP_ERROR
    from algorithms.folder_pipeline.workspace import (
        FolderWorkspaceError,
        default_workspace_root,
        discover_workspace,
    )

    try:
        workspace = discover_workspace(args.workspace, repository_root=_REPOSITORY_ROOT)
    except FolderWorkspaceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        default = default_workspace_root(_REPOSITORY_ROOT)
        print(f"提示：请把唯一的源点云放入 {default}\\input\\pointcloud\\ 目录。", file=sys.stderr)
        return 2

    # The default danger threshold is the canonical project risk standard
    # (never a new Folder-pipeline default); None routes through it.
    threshold = None if args.threshold is None else float(args.threshold)
    headless = args.pole_a is not None and args.pole_b is not None

    from algorithms.folder_pipeline.orchestrator import _canonical_risk_threshold_m

    resolved_threshold = _canonical_risk_threshold_m() if threshold is None else threshold

    print("FOLDER_TO_EXCEL V1")
    print(f"  工作区    : {workspace.root}")
    print(f"  源文件    : {workspace.source}")
    print(f"  源 SHA256 : {workspace.source_sha256[:16]}...")
    print(f"  危险阈值  : {resolved_threshold} m（来源：canonical project risk standard）")

    from algorithms.folder_pipeline.server import FolderServer

    folder_server = FolderServer(
        workspace,
        threshold_m=threshold,
        reference_dbh_cm=args.reference_dbh_cm,
        headless_poles=(args.pole_a, args.pole_b) if headless else None,
    )
    httpd = __import__(
        "http.server", fromlist=["ThreadingHTTPServer"]
    ).ThreadingHTTPServer(("127.0.0.1", args.port), folder_server.handler_factory())
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"  本地地址  : {url}")

    if headless:
        print("  headless 模式：自动选择档距并运行完整流水线...")
    elif not args.no_open:
        import webbrowser

        try:
            webbrowser.open(url)
        except OSError:  # pragma: no cover - best effort
            pass

    if headless:
        # Headless: do not block on serve_forever; poll the workflow state.
        deadline = time.time() + 7200
        while time.time() < deadline:
            state = folder_server.state.snapshot()
            if state.get("step") in (STEP_DONE, STEP_ERROR):
                break
            time.sleep(1.0)
        httpd.server_close()
        if state.get("step") == STEP_DONE and state.get("final_summary"):
            print("\n=== FOLDER_TO_EXCEL 运行摘要 ===")
            for key in (
                "FINAL_TREE_COUNT",
                "WIRE_STRAND_COUNT",
                "DANGEROUS_TREE_COUNT",
                "DANGER_POINT_COUNT",
                "GLOBAL_MINIMUM_CLEARANCE_M",
                "SPAN_LENGTH_M",
            ):
                print(f"  {key}: {state['final_summary'].get(key)}")
            print(f"  输出目录: {workspace.export_dir}")
            return 0
        if state.get("step") == STEP_ERROR:
            print(f"ERROR: {state.get('error')}", file=sys.stderr)
            if state.get("error_detail"):
                print(state.get("error_detail"), file=sys.stderr)
            return 1
        print("ERROR: 运行未完成（headless 超时）", file=sys.stderr)
        return 1

    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
