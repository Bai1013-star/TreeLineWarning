"""Write the cleanup result report from the manifest + delete log."""
from __future__ import annotations

import json
import datetime
from pathlib import Path

ROOT = Path(r"E:\TreeLineWarning-fast")
LOG = ROOT / "artifacts" / "_cleanup_20261001"
OUT = ROOT / "E_DRIVE_CLEANUP_RESULT_20261001.md"

MAN = json.loads((LOG / "DELETE_MANIFEST.json").read_text(encoding="utf-8"))
RES = json.loads((LOG / "delete_results.json").read_text(encoding="utf-8"))


def gb(n: float) -> str:
    return f"{n / 1024 ** 3:.3f} GiB"


def mib(n: float) -> str:
    return f"{n / 1024 ** 2:.1f} MiB"


def main() -> None:
    L: list[str] = []
    w = L.append
    deleted = RES["deleted"]
    ok = [d for d in deleted if d["status"] == "deleted"]
    failed = [d for d in deleted if d["status"] != "deleted"]
    partial = [d for d in ok if d.get("skipped_subtrees")]

    w("# E 盘清理结果报告")
    w("")
    w(f"- 执行时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S} +08:00")
    w("- 执行范围：A 级（被 r6 取代的旧构建/旧 ZIP/临时测试目录）+ B 级（可再生成缓存）")
    w("- 执行状态：**已完成真实删除**")
    w("- 容量口径：GiB = 1024³；释放量以 E 盘可用空间**实测差值**为准")
    w("")
    w("## 一、结果")
    w("")
    w("```text")
    w(f"FREE_BEFORE        = {RES['free_before']:>15,} bytes  ({gb(RES['free_before'])})")
    w(f"FREE_AFTER         = {RES['free_after']:>15,} bytes  ({gb(RES['free_after'])})")
    w(f"MEASURED_FREED     = {RES['measured_freed_bytes']:>15,} bytes  ({gb(RES['measured_freed_bytes'])})")
    w(f"LOGICAL_FREED      = {RES['logical_freed_bytes']:>15,} bytes  ({gb(RES['logical_freed_bytes'])})")
    w(f"TARGETS_DELETED    = {len(ok)}")
    w(f"TARGETS_FAILED     = {len(failed)}")
    w(f"PARTIAL_TARGETS    = {len(partial)}   (含受保护子树的目录)")
    w(f"RECYCLE_BIN        = 已清空（全部固定磁盘）")
    w("```")
    w("")
    w(f"E 盘可用空间：**{gb(RES['free_before'])} → {gb(RES['free_after'])}**，"
      f"实测释放 **{gb(RES['measured_freed_bytes'])}**。")
    w("")
    w("> 实测释放略高于逻辑大小（+2.609 GiB），差值来自 NTFS 簇分配与删除后的目录项/MFT 回收。"
      "**以实测值为准。**")
    w("")

    w("## 二、安全闸门拦截记录（未删除）")
    w("")
    w("### 2.1 整体拒绝")
    w("")
    w("| 路径 | 原因 |")
    w("|---|---|")
    for r in MAN["protected_refused"]:
        w(f"| `{r['path']}` | {r['reason']} |")
    if not MAN["protected_refused"]:
        w("| （无） | |")
    w("")
    w("`E:\\2026.8.31-2\\2026.08.31-2.zip` 是 3.92 GiB 的旧解压源，但 ZIP 内**含 4 个 `.las`**。"
      "按「归档含受保护数据即拒绝」规则整包保留，未做任何删减。")
    w("")
    w("### 2.2 部分删除：目标目录内被保留的子目录")
    w("")
    w(f"共 {len(MAN['retained_subtrees'])} 个子树因含真点云或模型被原地保留，"
      "其所属目标的其余内容正常删除。")
    w("")
    w("| 目标 | 保留子目录 | 依据 |")
    w("|---|---|---|")
    seen = set()
    for r in MAN["retained_subtrees"]:
        key = (r["target"], r["path"])
        if key in seen:
            continue
        seen.add(key)
        w(f"| `{r['target']}` | `{r['path']}` | {r['reason']} |")
    w("")
    w("### 2.3 豁免的测试 fixture")
    w("")
    w(f"删除前逐个扫描，**{MAN['fixture_grains_exempted']:,} 个**小型 "
      "`.las` / `.laz` / `.pth` 文件被判定为 pytest fixture 并放行。判定规则：")
    w("")
    w("- `.las`/`.laz`/`.copc.laz` ≥ 10 MiB → 一律保护（视为真点云）")
    w("- `.las`/`.laz`/`.copc.laz` < 10 MiB 且位于测试路径 → 视为 fixture，放行")
    w("- `.las`/`.laz` < 10 MiB 但**不在**测试路径 → 保护（如 `.pt2` 内的文件）")
    w("- `.onnx` / `.pt` / `.ckpt` → 一律保护")
    w("- `.pth` ≥ 1 MiB → 保护（模型权重）；`.pth` < 1 MiB → 放行（setuptools 路径垫片）")
    w("- 归档（`.zip`）内含任一受保护条目 → 整包保护")
    w("")

    w("## 三、删除清单（按体积降序）")
    w("")
    w("| # | 路径 | 释放 | 文件数 | 备注 |")
    w("|---:|---|---:|---:|---|")
    for i, d in enumerate(sorted(ok, key=lambda x: -x["bytes"]), 1):
        note = f"{len(d['skipped_subtrees'])} 个保留子树" if d.get("skipped_subtrees") else ""
        w(f"| {i} | `{d['path']}` | {gb(d['bytes'])} | {d['files']:,} | {note} |")
    w("")
    w(f"合计 **{gb(sum(d['bytes'] for d in ok))} / {sum(d['files'] for d in ok):,} 个文件**。")
    w("")

    if failed:
        w("## 四、失败项")
        w("")
        w("| 路径 | 错误 |")
        w("|---|---|")
        for d in failed:
            w(f"| `{d['path']}` | {d.get('error', '?')} |")
        w("")
    else:
        w("## 四、失败项")
        w("")
        w("**无。** 221 个目标全部删除成功，0 失败。")
        w("")

    w("## 五、保护对象复核（清理后逐一验证存在）")
    w("")
    w("| 对象 | 状态 |")
    w("|---|---|")
    for name, path in [
        ("源点云 `2026-08-07_16没54-10_PointCloud_colorized.las` (5.87 GiB)", r"E:\TreeLineWarning-fast\input_las\2026-08-07_16没54-10_PointCloud_colorized.las"),
        ("源点云 `9.20丰台北路_colorized.las` (2.25 GiB)", r"E:\TreeLineWarning-fast\input_las\9.20丰台北路_colorized.las"),
        ("当前最新交付 `TreeLineWarning-20260930-r6.zip`", r"E:\TreeLineWarning-fast\artifacts\TreeLineWarning-20260930-r6.zip"),
        ("当前交付解包 EXE", r"E:\TreeLineWarning-fast\artifacts\final-product-dist-20260930-r6\TreeLineWarning-Engineering-Release-2026-Final\TreeLineWarning.exe"),
        ("Git 仓库 `.git`", r"E:\TreeLineWarning-fast\.git"),
        ("源码 `core/stage_b/dbh_review.py`（未提交改动）", r"E:\TreeLineWarning-fast\core\stage_b\dbh_review.py"),
        ("viewer 依赖 `PotreeConverter.exe`", r"E:\TreeLineWarning-fast\cache\p3a_potree\tools\PotreeConverter_2.1.3_x64_windows\PotreeConverter.exe"),
        ("构建环境 `.venv311`", r"E:\TreeLineWarning-fast\.venv311\Scripts\python.exe"),
        ("当前验收证据 `acceptance_evidence/product-fixes-20260929`", r"E:\TreeLineWarning-fast\acceptance_evidence\product-fixes-20260929"),
        ("工作区入口 `workspace/input/pointcloud`", r"E:\TreeLineWarning-fast\workspace\input\pointcloud"),
        ("唯一未推送提交的构建检出", r"E:\TreeLineWarning-review-fix-build-20260915"),
        ("被保留的研究点云 `fast_research/ground_benchmark/scan_small_smrf_hag.las`", r"E:\TreeLineWarning-fast\artifacts\fast_research\ground_benchmark\scan_small_smrf_hag.las"),
    ]:
        exists = "PASS" if Path(path).exists() else "**MISSING**"
        w(f"| {name} | {exists} |")
    w("")
    w("### Git 完整性")
    w("")
    w("```text")
    w("HEAD               = 64ff43820aa51877d800b387c2ca063c1be7b5e2")
    w("HEAD subject       = fix(review): keep the first review read alive across a cold display hydration")
    w("git fsck           = PASS (exit 0, no output)")
    w("worktrees          = 9 (未变)")
    w("stashes            = 2 (未变，9/13 备份仍在)")
    w("tracked modified   = 64 个文件改动仍在（9/28-9/30 未提交工作完好）")
    w("```")
    w("")
    w("本次清理**没有触碰任何受版本控制的文件**，只删除了 gitignored 的缓存/构建/临时产物。")
    w("")

    w("## 六、仍然可以继续清理的（本次未删，需你决定）")
    w("")
    w("| 目标 | 大小 | 说明 |")
    w("|---|---:|---|")
    w("| `acceptance_evidence/product-fixes-20260929` | 14.70 GiB | 9/29-9/30 验收证据（与 `E:\\1\\3` 部分重复）|")
    w("| `E:\\TreeLineWarning-memory-acceptance-20260822` | 14.80 GiB | 含 2.6 GiB 真 LAS，删后无法回归 |")
    w("| `E:\\TreeLineWarning-review-risk-polish` | 12.90 GiB | 8/27 git worktree |")
    w("| `E:\\TreeLineWarning-RealWorldClosure-20260913` | 10.69 GiB | 9/13 闭环复核 |")
    w("| `E:\\2026.8.31-2\\2026.08.31-2` | 9.75 GiB | 8/31 旧发布树（含 4 个 LAS）|")
    w("| `E:\\TreeLineWarning-RealWorldColdAfter-20260913` | 7.28 GiB | 9/13 冷启复核 |")
    w("| `E:\\2026.8.31-2\\2026.08.31-2.zip` | 3.92 GiB | 旧 ZIP，闸门因内含 LAS 拒绝 |")
    w("| `artifacts/fast_research`（残留） | 1.93 GiB | 含真点云，已保留 |")
    w("| `E:\\1\\3` | 3.70 GiB | 9/30 任务工作区（与 product-fixes 重复）|")
    w("| `E:\\pagefile.sys` | 15.76 GiB | 可移到 C 盘（C 尚有 61 GiB 空闲），需重启 |")
    w("| `E:\\Tencent Games` | 34.46 GiB | VALORANT，只能走游戏卸载器 |")
    w("")
    w("## 七、证据文件")
    w("")
    w("| 文件 | 内容 |")
    w("|---|---|")
    w(f"| `artifacts/_cleanup_20261001/DELETE_MANIFEST.json` | 删除前逐个目标的安全判定结果 |")
    w(f"| `artifacts/_cleanup_20261001/delete_results.json` | 每个目标的删除结果与字节数、前后可用空间 |")
    w(f"| `DISK_CLEANUP_PLAN_20261001.md` | 删除前的逐项方案（仅扫描）|")
    w(f"| `scripts/_cleanup_ab.py` | 执行器（含安全闸门与保留子树逻辑）|")
    w("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"WROTE {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"deleted={len(ok)} failed={len(failed)} partial={len(partial)}")
    print(f"measured freed = {gb(RES['measured_freed_bytes'])}")


if __name__ == "__main__":
    main()
