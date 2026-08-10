"""Windows-friendly CLI behind ``RUN_NEW_LAS_DEMO.bat``.

The batch file supplies the exact reviewed Python interpreter.  This script
only discovers a new LAS/LAZ, creates a fresh output namespace, emits concise
Chinese progress, and leaves a detailed ``run.log`` on every run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
import webbrowser
from collections.abc import Callable
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.vegetation.new_las_demo import (  # noqa: E402
    SUPPORTED_SUFFIXES,
    create_isolated_run_dir,
    discover_demo_inputs,
    run_new_las_demo,
)

_NO_INPUT_MESSAGE = "请先将 LAS/LAZ 点云文件放入 input_las 文件夹，然后重新双击运行。"
_ENV_INPUT_DIR = "TREE_LINE_DEMO_INPUT_DIR"
_ENV_RUNS_ROOT = "TREE_LINE_DEMO_RUNS_ROOT"
_ENV_NO_OPEN_RESULTS = "TREE_LINE_DEMO_NO_OPEN_RESULTS"
_ENV_NON_INTERACTIVE = "TREE_LINE_DEMO_NON_INTERACTIVE"


def _environment_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else fallback


def _environment_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def configure_windows_console() -> None:
    """Emit UTF-8 from Python after the ASCII launcher selects code page 65001."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="New LAS one-click FAST demonstration runner")
    parser.add_argument("--input", type=Path, help="Explicit LAS/LAZ path for an automated test.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_environment_path(_ENV_INPUT_DIR, _REPOSITORY_ROOT / "input_las"),
        help="Folder containing one or more LAS/LAZ files.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=_environment_path(
            _ENV_RUNS_ROOT,
            _REPOSITORY_ROOT / "artifacts" / "one_click_runs",
        ),
        help="Parent directory for isolated outputs.",
    )
    parser.add_argument(
        "--no-open-results",
        action="store_true",
        help="Do not open local HTML pages; used by automated tests.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail rather than show a numbered choice for multiple files.",
    )
    parser.add_argument(
        "--open-latest-results",
        action="store_true",
        help="Open the newest completed one-click result without starting a run.",
    )
    parser.add_argument(
        "--route",
        choices=("POLE_SPAN_MATURE_BASELINE", "EXPERIMENTAL_V4"),
        default="POLE_SPAN_MATURE_BASELINE",
        help="Default reviewed route (pole-first span-based) or the legacy V4 route.",
    )
    return parser.parse_args(argv)


def select_input(
    input_dir: Path,
    *,
    explicit_input: Path | None = None,
    interactive: bool = True,
) -> Path | None:
    """Select one source without silently choosing among multiple files."""

    if explicit_input is not None:
        candidate = explicit_input.resolve()
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不是有效的 LAS/LAZ 文件: {candidate}")
        return candidate
    files = discover_demo_inputs(input_dir)
    if not files:
        print(_NO_INPUT_MESSAGE, flush=True)
        return None
    if len(files) == 1:
        print(f"检测到输入点云: {files[0]}", flush=True)
        return files[0]
    if not interactive:
        raise ValueError("input_las 中存在多个 LAS/LAZ 文件；自动测试不允许任意选择。")
    print("检测到多个 LAS/LAZ 文件，请输入编号后按回车：", flush=True)
    for position, path in enumerate(files, start=1):
        print(f"  {position}. {path.name}", flush=True)
    while True:
        answer = input("请选择: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            selected = 0
        if 1 <= selected <= len(files):
            return files[selected - 1]
        print("输入无效，请重新输入列表中的编号。", flush=True)


def _baseline_progress(current_stage: list[str]) -> Callable[[int, str, bool, float | None], None]:
    def progress(stage: int, message: str, done: bool, elapsed: float | None) -> None:
        current_stage[0] = message
        prefix = f"[{stage}/8]"
        if done and elapsed is not None:
            print(f"{prefix} {message}，耗时 {elapsed:.1f} 秒", flush=True)
        else:
            print(f"{prefix} {message}", flush=True)

    return progress


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.open_latest_results:
        return open_latest_results(args.runs_root)
    run_root: Path | None = None
    current_stage = "输入文件发现"
    try:
        source = select_input(
            args.input_dir,
            explicit_input=args.input,
            interactive=not (args.non_interactive or _environment_enabled(_ENV_NON_INTERACTIVE)),
        )
        if source is None:
            return 2
        run_root = create_isolated_run_dir(args.runs_root, source)
        _configure_log(run_root / "run.log")
        logging.info("new source selected: %s", source)

        if args.route == "POLE_SPAN_MATURE_BASELINE":
            return _run_baseline_cli(args, source, run_root, current_stage)

        def progress(stage: int, message: str, done: bool, elapsed: float | None) -> None:
            nonlocal current_stage
            current_stage = message
            prefix = f"[{stage}/7]"
            if done and elapsed is not None:
                print(f"{prefix} {message}，耗时 {elapsed:.1f} 秒", flush=True)
            else:
                print(f"{prefix} {message}", flush=True)

        result = run_new_las_demo(source, run_root, progress=progress)
        print("", flush=True)
        print("=" * 48, flush=True)
        print("处理完成", flush=True)
        print("=" * 48, flush=True)
        print(f"输入点云: {source}", flush=True)
        print(f"识别单木: {result.tree_count}", flush=True)
        print(f"树线距离状态: {result.tree_wire_status}", flush=True)
        minimum_text = (
            "不可用（无有效 W1）"
            if result.minimum_clearance_m is None
            else str(result.minimum_clearance_m)
        )
        print("最小距离: " + minimum_text, flush=True)
        print(f"结果目录: {result.run_root}", flush=True)
        logging.info("run completed: %s", result.summary_path)
        if not (args.no_open_results or _environment_enabled(_ENV_NO_OPEN_RESULTS)):
            _open_results(result)
        return 0
    except Exception as exc:  # pragma: no cover - exercised through focused CLI checks
        if run_root is not None:
            logging.exception("new LAS demo failed")
        output = str(run_root) if run_root is not None else "尚未创建"
        print("", flush=True)
        print("处理失败", flush=True)
        print(f"失败阶段: {current_stage}", flush=True)
        print(f"异常类型: {type(exc).__name__}", flush=True)
        print(f"异常信息: {exc}", flush=True)
        print(f"输出目录: {output}", flush=True)
        if run_root is not None:
            _append_traceback(run_root / "run.log")
        return 1


def _configure_log(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )


def _append_traceback(path: Path) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n" + traceback.format_exc() + "\n")


def _open_results(result) -> None:
    for path in (
        result.summary_html_path,
        result.run_root / "05_v4" / "v4_tree_segmentation_clean.html",
    ):
        if path.is_file():
            webbrowser.open(path.resolve().as_uri())


def _baseline_summary_paths(run_root: Path) -> Path | None:
    candidates = (
        run_root / "08_report" / "16_RUN_SUMMARY.html",
        run_root / "08_report" / "RUN_SUMMARY.html",
    )
    return next((path for path in candidates if path.is_file()), None)


def _run_baseline_cli(
    args: argparse.Namespace,
    source: Path,
    run_root: Path,
    initial_stage: str,
) -> int:
    """Run the POLE_SPAN_MATURE_BASELINE route with [N/8] progress."""

    from algorithms.vegetation.fast_baseline_pipeline import (
        run_fast_baseline_pipeline,
    )

    current_stage: list[str] = [initial_stage]
    try:
        summary = run_fast_baseline_pipeline(
            source,
            run_root,
            progress=_baseline_progress(current_stage),
        )
    except Exception as exc:  # pragma: no cover - exercised through focused CLI checks
        logging.exception("pole-span baseline failed")
        print("", flush=True)
        print("处理失败", flush=True)
        print(f"失败阶段: {current_stage[0]}", flush=True)
        print(f"异常类型: {type(exc).__name__}", flush=True)
        print(f"异常信息: {exc}", flush=True)
        print(f"输出目录: {run_root}", flush=True)
        if (run_root / "run.log").is_file():
            with (run_root / "run.log").open("a", encoding="utf-8") as stream:
                stream.write("\n" + traceback.format_exc() + "\n")
        return 1
    print("", flush=True)
    print("=" * 48, flush=True)
    print("处理完成", flush=True)
    print("=" * 48, flush=True)
    print(f"输入点云: {source}", flush=True)
    print(f"确认电杆: {summary.get('CONFIRMED_POLES', 0)}", flush=True)
    print(f"线路数: {summary.get('LINE_COUNT', 0)}", flush=True)
    print(f"档距数: {summary.get('SPAN_COUNT', 0)}", flush=True)
    print(f"最终单木: {summary.get('FINAL_TREE_COUNT', 0)}", flush=True)
    print(f"导线根数: {summary.get('WIRE_STRAND_COUNT', 0)}", flush=True)
    print(f"树线距离状态: {summary.get('TREE_WIRE_STATUS', 'UNKNOWN')}", flush=True)
    minimum = summary.get("GLOBAL_MINIMUM_CLEARANCE")
    print(
        "最小距离: " + (str(minimum) if minimum is not None else "不可用"),
        flush=True,
    )
    print(f"结果目录: {run_root}", flush=True)
    logging.info("pole-span baseline completed: %s", run_root)
    if not (args.no_open_results or _environment_enabled(_ENV_NO_OPEN_RESULTS)):
        summary_html = _baseline_summary_paths(run_root)
        if summary_html is not None:
            webbrowser.open(summary_html.resolve().as_uri())
    return 0


def open_latest_results(runs_root: Path) -> int:
    """Open only a completed report; never infer a result from a partial run directory."""

    if not runs_root.is_dir():
        print("尚未找到一键运行结果文件夹。", flush=True)
        return 1
    completed = sorted(
        (
            directory
            for directory in runs_root.iterdir()
            if directory.is_dir()
            and (
                (directory / "07_report" / "RUN_SUMMARY.html").is_file()
                or (directory / "08_report" / "16_RUN_SUMMARY.html").is_file()
            )
        ),
        key=lambda directory: (directory.stat().st_mtime_ns, directory.name),
        reverse=True,
    )
    if not completed:
        print("尚未找到已完成的一键运行结果。", flush=True)
        return 1
    summary = _baseline_summary_paths(completed[0])
    if summary is None:
        summary = completed[0] / "07_report" / "RUN_SUMMARY.html"
    if _environment_enabled(_ENV_NO_OPEN_RESULTS):
        print(f"最新结果: {summary}", flush=True)
    else:
        webbrowser.open(summary.resolve().as_uri())
        print(f"已打开最新结果: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    configure_windows_console()
    raise SystemExit(run_cli())
