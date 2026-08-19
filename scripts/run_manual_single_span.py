"""MANUAL_SINGLE_SPAN launcher.

Interactive (default): starts the local browser application and lets the
teacher click two poles and enter the risk threshold.

Headless: ``--pole-a X Y Z --pole-b X Y Z --threshold 3.5`` drives the same
pipeline without the browser (used by automated acceptance runs and tests).

Usage::

    python scripts/run_manual_single_span.py --input <las>
    python scripts/run_manual_single_span.py --input <las> \\
        --pole-a 500000.0 3.3e6 30.0 --pole-b 500130.0 3.3e6 30.0 --threshold 3.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from algorithms.manual_span.config import (  # noqa: E402
    default_manual_span_config,
    manual_route_config,
)
from algorithms.manual_span.pipeline import ManualSingleSpanRun, PoleRefinementError  # noqa: E402


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("需要 X Y Z 三个坐标值")
    return (parts[0], parts[1], parts[2])


def _create_run_dir(source: Path, runs_root: Path) -> Path:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    stem = source.stem.replace(" ", "_")[:40]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return runs_root / f"{stem}_{stamp}"


def _find_prior_ground(source: Path, runs_root: Path) -> Path | None:
    """Locate a validated PDAL SMRF ground cache for the same source.

    Used when the Python PDAL binding is unavailable: the frozen PDAL SMRF +
    HAG-NN ground for the same source and same coarse representatives is
    reused verbatim (see ``ManualSingleSpanRun._reuse_ground_cache_if_possible``).
    """

    import hashlib

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    source_sha = sha256_file(source)
    for candidate in sorted(
        (runs_root / ".." / "one_click_runs").glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        inspection = candidate / "00_input" / "source_inspection.json"
        if not inspection.is_file():
            continue
        try:
            payload = json.loads(inspection.read_text("utf-8"))
        except Exception:  # noqa: BLE001 - best effort
            continue
        if payload.get("input_sha256") == source_sha and (
            candidate / "02_ground" / "ground_hag_result.json"
        ).is_file():
            return candidate / "02_ground"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MANUAL_SINGLE_SPAN 人工单档距分析")
    parser.add_argument("--input", required=True, type=Path, help="输入 LAS 文件")
    parser.add_argument("--pole-a", type=_parse_xyz, default=None,
                        help="电杆A 点击坐标（真实坐标系，X Y Z）")
    parser.add_argument("--pole-b", type=_parse_xyz, default=None,
                        help="电杆B 点击坐标（真实坐标系，X Y Z）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="危险净空阈值（m）")
    parser.add_argument("--runs-root", type=Path,
                        default=Path("artifacts/manual_single_span_runs"))
    parser.add_argument("--no-open-results", action="store_true",
                        help="完成后不自动打开结果")
    parser.add_argument("--port", type=int, default=0,
                        help="交互模式服务器端口（0=自动）")
    parser.add_argument("--resume", type=Path, default=None,
                        help="(预留) 从已有运行目录恢复，当前未启用")
    args = parser.parse_args(argv)

    source = args.input.resolve()
    if not source.is_file():
        print(f"ERROR: 输入文件不存在: {source}", file=sys.stderr)
        return 2
    run_root = _create_run_dir(source, args.runs_root)

    headless = args.pole_a is not None and args.pole_b is not None
    if args.threshold is None:
        if headless:
            parser.error("headless 模式必须提供 --threshold")
        threshold = None
    else:
        threshold = float(args.threshold)

    print("MANUAL_SINGLE_SPAN")
    print(f"  源文件 : {source}")
    print(f"  运行目录: {run_root}")

    ground_reuse_dir = _find_prior_ground(source, args.runs_root)
    if ground_reuse_dir is not None:
        print(f"  复用既有 PDAL SMRF 地面缓存: {ground_reuse_dir}")

    if headless:
        run = ManualSingleSpanRun(
            source, run_root, cfg=manual_route_config(),
            manual=default_manual_span_config(),
            ground_reuse_dir=ground_reuse_dir,
        )
        run.initialize()
        print("  已初始化：点云 / 地面 / HAG 就绪。")
        try:
            run.refine_and_build_span(args.pole_a, args.pole_b)
        except PoleRefinementError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        print(f"  档距已建立: {run.span.span_name} 长度 {run.span.length_m:.2f} m")
        analysis = run.run_analysis()
        print(f"  分析完成: 导线 {analysis['wire_strand_count']}，"
              f"树 {analysis['tree_count']}，树线状态 {analysis['clearance_status']}")
        summary = run.finalize(threshold, open_results=not args.no_open_results)
        _print_summary(summary)
        return 0 if summary.get("RUN_STATUS") == "PASS" else 1

    # Interactive local application.
    from algorithms.manual_span.server import run_interactive

    summary = run_interactive(
        source,
        run_root,
        cfg=manual_route_config(),
        manual=default_manual_span_config(),
        ground_reuse_dir=ground_reuse_dir,
        port=args.port,
        open_browser=True,
    )
    _print_summary(summary)
    return 0 if summary.get("RUN_STATUS") == "PASS" else 1


def _print_summary(summary: dict) -> None:
    keys = [
        ("RUN_STATUS", "运行状态"),
        ("SPAN_LENGTH_M", "档距长度(m)"),
        ("FINAL_TREE_COUNT", "最终单木数"),
        ("TREE_SEGMENTATION_QUALITY", "单木分割质量"),
        ("DBH_MEASURED", "DBH实测"),
        ("DBH_ESTIMATED", "DBH估算"),
        ("DBH_REVIEW", "DBH复核"),
        ("DBH_UNAVAILABLE", "DBH无数据"),
        ("WIRE_CANDIDATE_POINTS", "导线候选点"),
        ("WIRE_CLUSTER_COUNT", "导线簇数"),
        ("WIRE_STRAND_COUNT", "导线根数"),
        ("W1_POLYLINE_COUNT", "W1导线数"),
        ("WIRE_EXTRACTION_QUALITY", "导线提取质量"),
        ("FULLRES_ASSIGNED_TREE_POINTS", "已分配全分辨率树点"),
        ("TREE_WIRE", "树线计算"),
        ("TREE_WIRE_RUNTIME", "树线时长(s)"),
        ("RISK_THRESHOLD_M", "危险净空阈值(m)"),
        ("DANGEROUS_TREE_COUNT", "危险树数"),
        ("DANGER_POINT_COUNT", "危险点数"),
        ("GLOBAL_MINIMUM_CLEARANCE_M", "场景最小净空(m)"),
        ("EXCEL_REPORT", "Excel报告"),
        ("VISUAL_INDEX", "可视化索引"),
        ("FINAL_OVERVIEW", "最终总览"),
        ("FINAL_REPORT", "最终报告"),
        ("TOTAL_RUNTIME", "总运行时长(s)"),
    ]
    print("\n=== MANUAL_SINGLE_SPAN 运行摘要 ===")
    for key, label in keys:
        value = summary.get(key)
        if isinstance(value, float):
            value = f"{value:.3f}"
        print(f"  {label}: {value}")
    print(f"  输出目录: {summary.get('OUTPUT_DIRECTORY')}")


if __name__ == "__main__":
    sys.exit(main())
