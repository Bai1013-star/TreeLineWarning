"""One-off disk cleanup planner for E: (report only, deletes nothing).

Enumerates candidate paths, measures real size, flags protected extensions,
and writes a per-item markdown report.
"""
from __future__ import annotations

import os
import json
import datetime
from pathlib import Path

ROOT = Path(r"E:\TreeLineWarning-fast")
PROTECTED = {".las", ".laz", ".copc.laz", ".onnx", ".pth", ".pt", ".ckpt"}


def measure(p: Path) -> dict:
    total = 0
    files = 0
    prot_files = 0
    prot_bytes = 0
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(p, onerror=lambda e: None):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                st = os.stat(fp, follow_symlinks=False)
            except OSError:
                continue
            total += st.st_size
            files += 1
            if st.st_mtime > newest:
                newest = st.st_mtime
            ext = os.path.splitext(fn)[1].lower()
            if ext in PROTECTED:
                prot_files += 1
                prot_bytes += st.st_size
    return {
        "bytes": total,
        "files": files,
        "prot_files": prot_files,
        "prot_bytes": prot_bytes,
        "newest": datetime.datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else "-",
    }


def gb(n: int) -> str:
    return f"{n / 1024 ** 3:.3f} GiB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.1f} MiB"


A_TARGETS = [
    (ROOT / "artifacts/final-product-dist-20260929-r4", "旧构建 dist (r4)"),
    (ROOT / "artifacts/final-product-work-20260929-r4", "旧构建 work (r4)"),
    (ROOT / "artifacts/final-product-dist-20260930-r5", "旧构建 dist (r5)"),
    (ROOT / "artifacts/final-product-work-20260930-r5", "旧构建 work (r5)"),
    (ROOT / "artifacts/TreeLineWarning-20260930-r5.zip", "旧发布 ZIP (r5)"),
    (ROOT / "artifacts/TreeLineWarning-Complete-20260929.zip", "旧发布 ZIP (9/29)"),
    (ROOT / "artifacts/final-product-work-20260929-r2", "旧构建 work (r2)"),
    (ROOT / "artifacts/final-product-work-20260929-r3", "旧构建 work (r3)"),
    (ROOT / "artifacts/dbh-format-release-work-20260928", "旧构建 work (dbh 9/28)"),
    (ROOT / "artifacts/dbh-format-release-work-final-20260928", "旧构建 work (dbh final)"),
    (ROOT / "artifacts/dbh-format-release-work-v3-20260928", "旧构建 work (dbh v3)"),
    (ROOT / "artifacts/dbh-investigation-20260928", "旧调查目录 (dbh 9/28)"),
    (ROOT / "artifacts/review-load-fix-acceptance-20260915", "旧验收 (9/15)"),
    (ROOT / "artifacts/review-load-fix-acceptance-20260915-final", "旧验收 (9/15 final)"),
    (ROOT / "artifacts/review-load-fix-final-smoke-20260915", "旧冒烟 (9/15)"),
    (ROOT / "artifacts/review-load-fix-final-smoke2-20260915", "旧冒烟 (9/15 b)"),
    (ROOT / "artifacts/review-load-fix-proj-data-20260915", "旧构建数据 (9/15)"),
    (ROOT / "artifacts/review-load-fix-work-20260915", "旧构建 work (9/15)"),
    (ROOT / "artifacts/review-fix-native-runtime-20260915", "旧原生运行时 (9/15)"),
    (ROOT / "artifacts/dbh-format-native-runtime-20260928", "旧原生运行时 (9/28)"),
    (ROOT / "artifacts/dbh-format-native-runtime-final-20260928", "旧原生运行时 (9/28 final)"),
    (ROOT / "artifacts/dbh-format-native-runtime-v3-20260928", "旧原生运行时 (9/28 v3)"),
    (ROOT / "artifacts/real-closure-build-work-20260913-606127c", "旧构建 work (9/13)"),
    (ROOT / "artifacts/real-closure-release-20260913-606127c", "旧发布目录 (9/13)"),
    (ROOT / "artifacts/final-acceptance-20260913", "旧验收 (9/13)"),
    (ROOT / "artifacts/final-product-20260930", "旧汇总目录 (9/30)"),
    (ROOT / "artifacts/final-product-20260929", "旧汇总目录 (9/29)"),
    (Path(r"E:\2026.8.31-2\2026.08.31-2.zip"), "旧解压源 ZIP (8/31)"),
    (ROOT / "actual", "零字节垃圾文件"),
    (ROOT / "bounded", "零字节垃圾文件"),
    (ROOT / "T001", "零字节垃圾文件"),
    (ROOT / "63", "未知无扩展名文件"),
    (ROOT / ".wire_endpoint_recovery_round2_run.log", "旧运行日志 (8/21)"),
]

B_TARGETS = [
    (ROOT / "cache/tower_ordered_project", "tower 有序项目缓存 (8/18)"),
    (ROOT / "cache/tower_raw_tiles", "tower 原始瓦片 (8/15)"),
    (ROOT / "cache/tower_review_project", "tower 复核项目 (8/15)"),
    (ROOT / "cache/mamba_root", "mamba/conda 包缓存 (8/14)"),
    (ROOT / "cache/giro3d_pdal_env", "giro3d PDAL 环境 (8/14)"),
    (ROOT / "cache/giro3d_pdal_control", "giro3d PDAL 对照 (8/14)"),
    (ROOT / "cache/giro3d_full_pdal", "giro3d 全量 PDAL (8/15)"),
    (ROOT / "cache/giro3d_full_source", "giro3d 全量源 (8/14)"),
    (ROOT / "cache/p3a_giro3d", "p3a giro3d 试验 (8/14)"),
    (ROOT / "cache/wire_v2_shadow", "wire v2 影子 (8/17)"),
    (ROOT / "cache/wire_v2_shadow_hardened_20260817", "wire v2 加固 (8/17)"),
    (ROOT / "cache/wire_v2_shadow_hardened_20260817_track_evidence", "wire v2 轨迹证据 (8/17)"),
    (ROOT / "cache/wire_v2_shadow_hardened_20260817_track_evidence_exact_polyline", "wire v2 精确折线 (8/17)"),
    (ROOT / "cache/debug", "调试输出 (8/18)"),
    (ROOT / "artifacts/fast_research", "8/9-8/10 研究点云派生结果"),
    (Path(r"E:\TreeLineWarning-source-cache"), "角色索引瓦片缓存 (可再生成)"),
]

# ---------------------------------------------------------------- concurrent
CONCURRENT = [
    (ROOT / "artifacts/one_click_runs", "8/10 一键运行产物 (派生中间 LAS)"),
    (ROOT / "artifacts/manual_single_span_runs", "8/10 人工单档运行产物 (派生中间 LAS)"),
    (ROOT / "artifacts/dbh-format-20260928", "9/28 DBH 格式验证 (含 0 字节 pytest fixture)"),
    (ROOT / "workspace/survey_v01_gate_b_20260830", "8/30 gate-B 验收工作区 (派生中间 LAS)"),
    (ROOT / "acceptance_evidence/final-product-20260928", "9/28 旧验收证据 (派生中间 LAS)"),
    (Path(r"E:\2026.8.31-2\2026.08.31-2"), "8/31 旧发布树 (派生中间 LAS)"),
]

# ------------------------------------------------------------------- excludes
KEEP = [
    (ROOT / "core", "当前源码 (含 9/28-9/30 未提交改动)"),
    (ROOT / "algorithms", "当前源码"),
    (ROOT / "desktop", "当前源码"),
    (ROOT / "scripts", "当前源码"),
    (ROOT / "tests", "当前测试"),
    (ROOT / "viewer_poc", "当前前端"),
    (ROOT / "docs", "当前文档"),
    (ROOT / ".git", "Git 仓库 (含唯一未推送提交 64ff438)"),
    (ROOT / "input_las", "真点云源数据 (不可再生)"),
    (ROOT / ".venv311", "构建虚拟环境"),
    (ROOT / ".venv", "虚拟环境"),
    (ROOT / "artifacts/TreeLineWarning-20260930-r6.zip", "当前最新交付 ZIP"),
    (ROOT / "artifacts/final-product-dist-20260930-r6", "当前最新交付 dist"),
    (ROOT / "artifacts/final-product-work-20260930-r6", "当前最新 build work"),
    (ROOT / "cache/p3a_potree", "当前代码引用的 PotreeConverter.exe + 运行时"),
    (ROOT / "acceptance_evidence/product-fixes-20260929", "9/29-9/30 当前验收证据"),
    (ROOT / "workspace/input", "工作区入口 (空)"),
    (ROOT / "workspace/01_span_selection", "工作区结构"),
    (ROOT / "workspace/02_tree_analysis", "工作区结构"),
    (ROOT / "workspace/03_export", "工作区结构"),
    (Path(r"E:\TreeLineWarning-review-fix-build-20260915"), "承载唯一本地提交 64ff438 的构建检出"),
]

SCATTERED = [
    ".test-tmp*", ".final-test-tmp*", ".wire_*", ".build-*",
    ".pytest_cache", ".pytest_tmp", ".ruff_cache", ".pt2", "tmp-check",
    "TreeLineWarning-fast.test-tmp", "treeline_warning.egg-info",
    ".test-tmp-20260823-performance",
]

RECYCLE = Path(r"E:\$RECYCLE.BIN")


def main() -> None:
    lines: list[str] = []
    w = lines.append

    w("# E 盘清理执行方案（逐项清单）")
    w("")
    w(f"- 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S} +08:00")
    w("- 状态：**仅扫描，未删除任何文件**")
    w("- 容量口径：文件逻辑大小，GiB = 1024³")
    w("- 保护扩展名：`.las .laz .copc.laz .onnx .pth .pt .ckpt`")
    w(f"- 源点云锚点：`input_las\\*.las`（保留，因此全部派生中间 LAS 均可再生）")
    w("")

    def section(title: str, note: str, targets, decision: str):
        w(f"## {title}")
        w("")
        w(note)
        w("")
        w("| # | 路径 | 大小 | 文件数 | 受保护文件 | 最新修改 | 说明 |")
        w("|---:|---|---:|---:|---:|---|---|")
        tot = 0
        tot_files = 0
        rows = []
        for i, (p, desc) in enumerate(targets, 1):
            if not p.exists():
                rows.append((i, str(p), "缺失", 0, "-", "-", desc))
                continue
            if p.is_file():
                st = p.stat()
                m = {"bytes": st.st_size, "files": 1, "prot_files": 0, "prot_bytes": 0,
                     "newest": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")}
            else:
                m = measure(p)
            tot += m["bytes"]
            tot_files += m["files"]
            prot = f"{m['prot_files']} / {gb(m['prot_bytes'])}" if m["prot_files"] else "0"
            rows.append((i, str(p), gb(m["bytes"]), m["files"], prot, m["newest"], desc))
        for r in rows:
            w(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
        w("")
        w(f"**小计：{gb(tot)} / {tot_files:,} 个文件** —— 决策：{decision}")
        w("")
        return tot

    a = section(
        "A 级 — 已明确被取代，可直接删除",
        "全部为被 `r6`（2026-09-30 12:08）取代的旧构建、旧发布 ZIP、旧验收输出及零字节垃圾。",
        A_TARGETS,
        "**删**",
    )
    b = section(
        "B 级 — 可再生成缓存，可直接删除",
        "`cache\\` 下 8/13–8/18 的 giro3d / potree / tower 试验环境与瓦片。"
        "注意：`cache\\p3a_potree` **不在删除清单**，因为当前代码 `viewer_cache.py` 引用其中的 "
        "`PotreeConverter.exe` 与 Potree 运行时。",
        B_TARGETS,
        "**删**（`cache\\tower_ordered_project` 删后下次 tower 流程会重建，耗时较长）",
    )
    c = section(
        "待确认 — 含派生中间 LAS（体积最大）",
        "以下目录含 `.las` 中间产物（`classified_hag.las`、`working_set_representatives.las` 等）。"
        "经核对，**这些全部是 `input_las` 源点云的派生结果**，源文件保留即可再生；"
        "但旧审计报告曾将其列为「受保护测试数据」，故单独列出由你决定。",
        CONCURRENT,
        "**需你确认**",
    )

    # scattered temp dirs
    w("## A 级附 — 仓库内散落的临时测试目录")
    w("")
    found = []
    for pat in SCATTERED:
        for d in sorted(ROOT.glob(pat)):
            if d.is_dir():
                found.append(d)
    seen = set()
    uniq = []
    for d in found:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    scat_bytes = 0
    scat_files = 0
    scat_detail = []
    for d in uniq:
        m = measure(d)
        scat_bytes += m["bytes"]
        scat_files += m["files"]
        scat_detail.append((d.name, m["bytes"]))
    scat_detail.sort(key=lambda x: -x[1])
    w(f"命中 **{len(uniq)} 个目录**，合计 **{gb(scat_bytes)} / {scat_files:,} 个文件**。")
    w("")
    w("体积最大的 20 个：")
    w("")
    w("| 目录 | 大小 |")
    w("|---|---:|")
    for name, sz in scat_detail[:20]:
        w(f"| `{name}` | {gb(sz)} |")
    w("")
    w("决策：**删**（均为 pytest/构建临时输出，非发布输入）")
    w("")

    # scattered loose files
    w("## A 级附 — 仓库根目录散落文件")
    w("")
    loose = [
        ("actual", "零字节"),
        ("bounded", "零字节"),
        ("T001", "零字节"),
        ("63", "无扩展名"),
        (".wire_endpoint_recovery_round2_run.log", "旧日志"),
        ("_analysis_gates.py", "workspace 内的分析脚本"),
    ]
    w("| 文件 | 大小 | 判断 |")
    w("|---|---:|---|")
    for name, desc in loose:
        p = ROOT / name if (ROOT / name).exists() else ROOT / "workspace" / name
        if p.exists():
            w(f"| `{p}` | {p.stat().st_size:,} B | {desc} |")
    w("")

    # recycle bin
    w("## A 级附 — 回收站")
    w("")
    if RECYCLE.exists():
        m = measure(RECYCLE)
        w(f"- `E:\\$RECYCLE.BIN`：**{gb(m['bytes'])}** / {m['files']:,} 个文件（40 个回收站条目）")
        w("- 已知最大条目：`1`（3.98 GiB，与 `E:\\1\\3` 重复）、`litchfield.zip`（2.17 GiB）、"
          "`artifacts`（3.71 GiB），其余为 `*.crdownload` 中断下载残片")
    w("- 决策：**清空回收站**")
    w("")

    # keep list
    w("## 保留清单（本次绝不触碰）")
    w("")
    w("| 路径 | 原因 |")
    w("|---|---|")
    for p, why in KEEP:
        size = ""
        if p.exists():
            if p.is_file():
                size = f" ({gb(p.stat().st_size)})"
            else:
                m = measure(p)
                size = f" ({gb(m['bytes'])})"
        w(f"| `{p}`{size} | {why} |")
    w("")

    w("## 汇总")
    w("")
    w("| 级别 | 可释放 | 决策 |")
    w("|---|---:|---|")
    w(f"| A 级（含散落临时目录 + 回收站） | ≈ {gb(a + scat_bytes + measure(RECYCLE)['bytes'])} | 直接删 |")
    w(f"| B 级（可再生成缓存） | ≈ {gb(b)} | 直接删 |")
    w(f"| 待确认（派生中间 LAS） | ≈ {gb(c)} | 需你点头 |")
    w("")

    out = ROOT / "DISK_CLEANUP_PLAN_20261001.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {out}  ({out.stat().st_size:,} bytes)")
    print(f"A={gb(a)}  scattered={gb(scat_bytes)}  recycle={gb(measure(RECYCLE)['bytes'])}  B={gb(b)}  CONFIRM={gb(c)}")


if __name__ == "__main__":
    main()
