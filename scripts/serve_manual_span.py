"""Lightweight local UI re-open for the manual single-span route.

Resumes an EXISTING initialized/completed run (all available checkpoints are
reused) and serves the interactive selector without starting downstream
analysis.  Selection is intentionally empty on server start: the Owner must
explicitly choose Pole A and Pole B again.  This prevents a reset selection
from being resurrected by a browser or server refresh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.manual_span.config import (  # noqa: E402
    default_manual_span_config,
    manual_route_config,
)
from algorithms.manual_span.pipeline import ManualSingleSpanRun  # noqa: E402
from algorithms.manual_span.server import (  # noqa: E402
    WorkflowState,
    _make_handler_factory,
    _make_progress_callback,
)

from scripts.run_manual_single_span import _find_prior_ground  # noqa: E402


def _latest_run(runs_root: Path) -> Path:
    runs = sorted(runs_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"没有找到已完成的运行目录: {runs_root}")
    return runs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重新打开人工单档距本地界面")
    parser.add_argument("--run-root", type=Path, default=None,
                        help="已完成的运行目录（默认使用最新）")
    parser.add_argument("--port", type=int, default=0, help="服务器端口（0=自动）")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)

    runs_root = Path("artifacts/manual_single_span_runs")
    run_root = args.run_root.resolve() if args.run_root else _latest_run(runs_root)
    source = run_root / "00_input" / "source_inspection.json"
    if not source.is_file():
        raise SystemExit(f"运行目录缺少源信息: {run_root}")
    with open(source, encoding="utf-8") as stream:
        inspection = json.load(stream)
    source_las = Path(inspection["input_file"])
    if not source_las.is_file():
        raise SystemExit(f"源 LAS 不存在: {source_las}")

    print(f"  复用运行目录: {run_root}")
    print(f"  源文件     : {source_las.name}")
    state = WorkflowState()
    run = ManualSingleSpanRun(
        source_las,
        run_root,
        cfg=manual_route_config(),
        manual=default_manual_span_config(),
        progress=_make_progress_callback(state),
        ground_reuse_dir=_find_prior_ground(source_las, runs_root),
    )
    run.initialize()  # resume from checkpoints (fast, no expensive re-run)

    print("  选择器已就绪：请按“选择杆 A”，再按“选择杆 B”，在俯视图中点击两根真实电杆。")

    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler_factory(run, state))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"  本地界面地址: {url}")

    import webbrowser

    if not args.no_open:
        try:
            webbrowser.open(url)
        except OSError:  # pragma: no cover - best effort
            pass

    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
