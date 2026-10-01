# E 盘清理执行方案（逐项清单）

- 生成时间：2026-10-01 01:07:50 +08:00
- 状态：**仅扫描，未删除任何文件**
- 容量口径：文件逻辑大小，GiB = 1024³
- 保护扩展名：`.las .laz .copc.laz .onnx .pth .pt .ckpt`
- 源点云锚点：`input_las\*.las`（保留，因此全部派生中间 LAS 均可再生）

## A 级 — 已明确被取代，可直接删除

全部为被 `r6`（2026-09-30 12:08）取代的旧构建、旧发布 ZIP、旧验收输出及零字节垃圾。

| # | 路径 | 大小 | 文件数 | 受保护文件 | 最新修改 | 说明 |
|---:|---|---:|---:|---:|---|---|
| 1 | `E:\TreeLineWarning-fast\artifacts\final-product-dist-20260929-r4` | 2.103 GiB | 2861 | 0 | 2026-09-29 01:02 | 旧构建 dist (r4) |
| 2 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260929-r4` | 78.7 MiB | 18 | 0 | 2026-09-29 01:02 | 旧构建 work (r4) |
| 3 | `E:\TreeLineWarning-fast\artifacts\final-product-dist-20260930-r5` | 2.103 GiB | 2861 | 0 | 2026-09-30 11:36 | 旧构建 dist (r5) |
| 4 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260930-r5` | 78.8 MiB | 18 | 0 | 2026-09-30 11:36 | 旧构建 work (r5) |
| 5 | `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-20260930-r5.zip` | 1.473 GiB | 1 | 0 | 2026-09-30 11:45 | 旧发布 ZIP (r5) |
| 6 | `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-Complete-20260929.zip` | 1.469 GiB | 1 | 0 | 2026-09-29 01:05 | 旧发布 ZIP (9/29) |
| 7 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260929-r2` | 78.6 MiB | 18 | 0 | 2026-09-29 00:19 | 旧构建 work (r2) |
| 8 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260929-r3` | 78.7 MiB | 18 | 0 | 2026-09-29 00:44 | 旧构建 work (r3) |
| 9 | `E:\TreeLineWarning-fast\artifacts\dbh-format-release-work-20260928` | 78.6 MiB | 18 | 0 | 2026-09-28 22:40 | 旧构建 work (dbh 9/28) |
| 10 | `E:\TreeLineWarning-fast\artifacts\dbh-format-release-work-final-20260928` | 78.6 MiB | 18 | 0 | 2026-09-28 22:46 | 旧构建 work (dbh final) |
| 11 | `E:\TreeLineWarning-fast\artifacts\dbh-format-release-work-v3-20260928` | 78.6 MiB | 18 | 0 | 2026-09-28 23:01 | 旧构建 work (dbh v3) |
| 12 | `E:\TreeLineWarning-fast\artifacts\dbh-investigation-20260928` | 8.4 MiB | 653 | 0 | 2026-09-29 00:31 | 旧调查目录 (dbh 9/28) |
| 13 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-acceptance-20260915` | 0.0 MiB | 4 | 0 | 2026-09-15 20:54 | 旧验收 (9/15) |
| 14 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-acceptance-20260915-final` | 0.0 MiB | 4 | 0 | 2026-09-15 21:10 | 旧验收 (9/15 final) |
| 15 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-final-smoke-20260915` | 0.0 MiB | 1 | 0 | 2026-09-15 21:11 | 旧冒烟 (9/15) |
| 16 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-final-smoke2-20260915` | 0.0 MiB | 9 | 0 | 2026-09-15 21:42 | 旧冒烟 (9/15 b) |
| 17 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-proj-data-20260915` | 缺失 | 0 | - | - | 旧构建数据 (9/15) |
| 18 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-work-20260915` | 78.5 MiB | 18 | 0 | 2026-09-15 20:51 | 旧构建 work (9/15) |
| 19 | `E:\TreeLineWarning-fast\artifacts\review-fix-native-runtime-20260915` | 112.4 MiB | 70 | 0 | 2026-09-15 20:50 | 旧原生运行时 (9/15) |
| 20 | `E:\TreeLineWarning-fast\artifacts\dbh-format-native-runtime-20260928` | 112.4 MiB | 70 | 0 | 2026-09-28 22:38 | 旧原生运行时 (9/28) |
| 21 | `E:\TreeLineWarning-fast\artifacts\dbh-format-native-runtime-final-20260928` | 112.4 MiB | 72 | 0 | 2026-09-29 13:48 | 旧原生运行时 (9/28 final) |
| 22 | `E:\TreeLineWarning-fast\artifacts\dbh-format-native-runtime-v3-20260928` | 112.4 MiB | 69 | 0 | 2026-09-15 20:49 | 旧原生运行时 (9/28 v3) |
| 23 | `E:\TreeLineWarning-fast\artifacts\real-closure-build-work-20260913-606127c` | 78.5 MiB | 18 | 0 | 2026-09-13 18:44 | 旧构建 work (9/13) |
| 24 | `E:\TreeLineWarning-fast\artifacts\real-closure-release-20260913-606127c` | 11.7 MiB | 12 | 0 | 2026-09-13 19:03 | 旧发布目录 (9/13) |
| 25 | `E:\TreeLineWarning-fast\artifacts\final-acceptance-20260913` | 0.0 MiB | 1 | 0 | 2026-09-13 14:02 | 旧验收 (9/13) |
| 26 | `E:\TreeLineWarning-fast\artifacts\final-product-20260930` | 0.0 MiB | 1 | 0 | 2026-09-30 12:30 | 旧汇总目录 (9/30) |
| 27 | `E:\TreeLineWarning-fast\artifacts\final-product-20260929` | 174.9 MiB | 12713 | 780 / 32.9 MiB | 2026-09-29 12:56 | 旧汇总目录 (9/29) |
| 28 | `E:\2026.8.31-2\2026.08.31-2.zip` | 3.922 GiB | 1 | 0 | 2026-08-31 17:03 | 旧解压源 ZIP (8/31) |
| 29 | `E:\TreeLineWarning-fast\actual` | 0.0 MiB | 1 | 0 | 2026-08-15 21:58 | 零字节垃圾文件 |
| 30 | `E:\TreeLineWarning-fast\bounded` | 0.0 MiB | 1 | 0 | 2026-08-15 21:58 | 零字节垃圾文件 |
| 31 | `E:\TreeLineWarning-fast\T001` | 0.0 MiB | 1 | 0 | 2026-08-15 21:58 | 零字节垃圾文件 |
| 32 | `E:\TreeLineWarning-fast\63` | 缺失 | 0 | - | - | 未知无扩展名文件 |
| 33 | `E:\TreeLineWarning-fast\.wire_endpoint_recovery_round2_run.log` | 0.0 MiB | 1 | 0 | 2026-08-21 20:49 | 旧运行日志 (8/21) |

**小计：12.391 GiB / 19,570 个文件** —— 决策：**删**

## B 级 — 可再生成缓存，可直接删除

`cache\` 下 8/13–8/18 的 giro3d / potree / tower 试验环境与瓦片。注意：`cache\p3a_potree` **不在删除清单**，因为当前代码 `viewer_cache.py` 引用其中的 `PotreeConverter.exe` 与 Potree 运行时。

| # | 路径 | 大小 | 文件数 | 受保护文件 | 最新修改 | 说明 |
|---:|---|---:|---:|---:|---|---|
| 1 | `E:\TreeLineWarning-fast\cache\tower_ordered_project` | 33.312 GiB | 6060 | 0 | 2026-08-18 21:38 | tower 有序项目缓存 (8/18) |
| 2 | `E:\TreeLineWarning-fast\cache\tower_raw_tiles` | 3.131 GiB | 50 | 0 | 2026-08-15 18:37 | tower 原始瓦片 (8/15) |
| 3 | `E:\TreeLineWarning-fast\cache\tower_review_project` | 2.1 MiB | 382 | 0 | 2026-08-15 20:49 | tower 复核项目 (8/15) |
| 4 | `E:\TreeLineWarning-fast\cache\mamba_root` | 5.099 GiB | 24619 | 0 | 2026-08-14 18:35 | mamba/conda 包缓存 (8/14) |
| 5 | `E:\TreeLineWarning-fast\cache\giro3d_pdal_env` | 3.864 GiB | 20581 | 0 | 2026-08-17 13:43 | giro3d PDAL 环境 (8/14) |
| 6 | `E:\TreeLineWarning-fast\cache\giro3d_pdal_control` | 6.1 MiB | 1 | 1 / 6.1 MiB | 2026-08-14 18:37 | giro3d PDAL 对照 (8/14) |
| 7 | `E:\TreeLineWarning-fast\cache\giro3d_full_pdal` | 847.9 MiB | 2 | 1 / 847.9 MiB | 2026-08-15 20:45 | giro3d 全量 PDAL (8/15) |
| 8 | `E:\TreeLineWarning-fast\cache\giro3d_full_source` | 512.1 MiB | 2 | 1 / 512.1 MiB | 2026-08-14 17:51 | giro3d 全量源 (8/14) |
| 9 | `E:\TreeLineWarning-fast\cache\p3a_giro3d` | 183.2 MiB | 9177 | 2 / 4.6 MiB | 2026-08-14 16:59 | p3a giro3d 试验 (8/14) |
| 10 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow` | 49.7 MiB | 27 | 0 | 2026-08-17 14:15 | wire v2 影子 (8/17) |
| 11 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow_hardened_20260817` | 67.3 MiB | 47 | 0 | 2026-08-17 14:46 | wire v2 加固 (8/17) |
| 12 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow_hardened_20260817_track_evidence` | 88.2 MiB | 61 | 0 | 2026-08-17 15:34 | wire v2 轨迹证据 (8/17) |
| 13 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow_hardened_20260817_track_evidence_exact_polyline` | 64.2 MiB | 45 | 0 | 2026-08-17 15:47 | wire v2 精确折线 (8/17) |
| 14 | `E:\TreeLineWarning-fast\cache\debug` | 24.6 MiB | 7 | 0 | 2026-08-18 16:25 | 调试输出 (8/18) |
| 15 | `E:\TreeLineWarning-fast\artifacts\fast_research` | 2.138 GiB | 165 | 10 / 1.761 GiB | 2026-08-10 01:05 | 8/9-8/10 研究点云派生结果 |
| 16 | `E:\TreeLineWarning-source-cache` | 7.019 GiB | 4020 | 0 | 2026-09-30 12:36 | 角色索引瓦片缓存 (可再生成) |

**小计：56.364 GiB / 65,246 个文件** —— 决策：**删**（`cache\tower_ordered_project` 删后下次 tower 流程会重建，耗时较长）

## 待确认 — 含派生中间 LAS（体积最大）

以下目录含 `.las` 中间产物（`classified_hag.las`、`working_set_representatives.las` 等）。经核对，**这些全部是 `input_las` 源点云的派生结果**，源文件保留即可再生；但旧审计报告曾将其列为「受保护测试数据」，故单独列出由你决定。

| # | 路径 | 大小 | 文件数 | 受保护文件 | 最新修改 | 说明 |
|---:|---|---:|---:|---:|---|---|
| 1 | `E:\TreeLineWarning-fast\artifacts\one_click_runs` | 7.359 GiB | 231 | 14 / 273.8 MiB | 2026-08-10 19:06 | 8/10 一键运行产物 (派生中间 LAS) |
| 2 | `E:\TreeLineWarning-fast\artifacts\manual_single_span_runs` | 11.880 GiB | 125 | 2 / 62.0 MiB | 2026-08-11 03:50 | 8/10 人工单档运行产物 (派生中间 LAS) |
| 3 | `E:\TreeLineWarning-fast\artifacts\dbh-format-20260928` | 1.797 GiB | 13418 | 857 / 127.4 MiB | 2026-09-29 00:23 | 9/28 DBH 格式验证 (含 0 字节 pytest fixture) |
| 4 | `E:\TreeLineWarning-fast\workspace\survey_v01_gate_b_20260830` | 18.647 GiB | 2653 | 4 / 99.6 MiB | 2026-09-08 15:20 | 8/30 gate-B 验收工作区 (派生中间 LAS) |
| 5 | `E:\TreeLineWarning-fast\acceptance_evidence\final-product-20260928` | 3.776 GiB | 968 | 4 / 78.8 MiB | 2026-09-29 01:19 | 9/28 旧验收证据 (派生中间 LAS) |
| 6 | `E:\2026.8.31-2\2026.08.31-2` | 9.748 GiB | 1717 | 4 / 99.6 MiB | 2026-09-05 22:54 | 8/31 旧发布树 (派生中间 LAS) |

**小计：53.207 GiB / 19,112 个文件** —— 决策：**需你确认**

## A 级附 — 仓库内散落的临时测试目录

命中 **176 个目录**，合计 **1.980 GiB / 80,705 个文件**。

体积最大的 20 个：

| 目录 | 大小 |
|---|---:|
| `.final-test-tmp-full-20260913` | 559.9 MiB |
| `.test-tmp-delivery-sol-baseline-20260830` | 344.9 MiB |
| `.test-tmp-clearance-sol-broad-20260830-a` | 231.1 MiB |
| `.test-tmp-testfinish-final` | 175.8 MiB |
| `.test-tmp-testfinish-staging` | 131.1 MiB |
| `.test-tmp-testfinish-broad2` | 60.1 MiB |
| `.test-tmp-giro3d-final-regression-20260823-01` | 56.4 MiB |
| `.test-tmp-review-ux-full-suite-20260822` | 50.8 MiB |
| `.test-tmp-dbh-freeze-20260930` | 41.8 MiB |
| `.test-tmp-dbh-close-all-20260930` | 37.8 MiB |
| `.test-tmp-review-ux-full-first-failure-20260822` | 37.5 MiB |
| `.test-tmp-dbh-freshness-20260930` | 37.1 MiB |
| `.test-tmp-dbh-close-20260930` | 35.8 MiB |
| `.test-tmp-phase-b-broad` | 31.3 MiB |
| `.test-tmp-dbh-fixes-20260929` | 29.8 MiB |
| `.wire_v2_high_recall_isolated` | 17.8 MiB |
| `.test-tmp-views-broad-20260929` | 14.4 MiB |
| `.test-tmp-proj-unicode-20260830-01` | 9.9 MiB |
| `.test-tmp-full-final` | 4.1 MiB |
| `.test-tmp-full` | 4.1 MiB |

决策：**删**（均为 pytest/构建临时输出，非发布输入）

## A 级附 — 仓库根目录散落文件

| 文件 | 大小 | 判断 |
|---|---:|---|
| `E:\TreeLineWarning-fast\actual` | 0 B | 零字节 |
| `E:\TreeLineWarning-fast\bounded` | 0 B | 零字节 |
| `E:\TreeLineWarning-fast\T001` | 0 B | 零字节 |
| `E:\TreeLineWarning-fast\.wire_endpoint_recovery_round2_run.log` | 35,312 B | 旧日志 |
| `E:\TreeLineWarning-fast\workspace\_analysis_gates.py` | 6,226 B | workspace 内的分析脚本 |

## A 级附 — 回收站

- `E:\$RECYCLE.BIN`：**9.998 GiB** / 1,886 个文件（40 个回收站条目）
- 已知最大条目：`1`（3.98 GiB，与 `E:\1\3` 重复）、`litchfield.zip`（2.17 GiB）、`artifacts`（3.71 GiB），其余为 `*.crdownload` 中断下载残片
- 决策：**清空回收站**

## 保留清单（本次绝不触碰）

| 路径 | 原因 |
|---|---|
| `E:\TreeLineWarning-fast\core` (4.8 MiB) | 当前源码 (含 9/28-9/30 未提交改动) |
| `E:\TreeLineWarning-fast\algorithms` (8.4 MiB) | 当前源码 |
| `E:\TreeLineWarning-fast\desktop` (1.0 MiB) | 当前源码 |
| `E:\TreeLineWarning-fast\scripts` (1.7 MiB) | 当前源码 |
| `E:\TreeLineWarning-fast\tests` (20.7 MiB) | 当前测试 |
| `E:\TreeLineWarning-fast\viewer_poc` (4.2 MiB) | 当前前端 |
| `E:\TreeLineWarning-fast\docs` (1.0 MiB) | 当前文档 |
| `E:\TreeLineWarning-fast\.git` (0.0 MiB) | Git 仓库 (含唯一未推送提交 64ff438) |
| `E:\TreeLineWarning-fast\input_las` (8.120 GiB) | 真点云源数据 (不可再生) |
| `E:\TreeLineWarning-fast\.venv311` (1.814 GiB) | 构建虚拟环境 |
| `E:\TreeLineWarning-fast\.venv` (362.2 MiB) | 虚拟环境 |
| `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-20260930-r6.zip` (1.473 GiB) | 当前最新交付 ZIP |
| `E:\TreeLineWarning-fast\artifacts\final-product-dist-20260930-r6` (2.103 GiB) | 当前最新交付 dist |
| `E:\TreeLineWarning-fast\artifacts\final-product-work-20260930-r6` (78.8 MiB) | 当前最新 build work |
| `E:\TreeLineWarning-fast\cache\p3a_potree` (613.9 MiB) | 当前代码引用的 PotreeConverter.exe + 运行时 |
| `E:\TreeLineWarning-fast\acceptance_evidence\product-fixes-20260929` (14.701 GiB) | 9/29-9/30 当前验收证据 |
| `E:\TreeLineWarning-fast\workspace\input` (0.0 MiB) | 工作区入口 (空) |
| `E:\TreeLineWarning-fast\workspace\01_span_selection` (0.0 MiB) | 工作区结构 |
| `E:\TreeLineWarning-fast\workspace\02_tree_analysis` (0.0 MiB) | 工作区结构 |
| `E:\TreeLineWarning-fast\workspace\03_export` (0.0 MiB) | 工作区结构 |
| `E:\TreeLineWarning-review-fix-build-20260915` (21.4 MiB) | 承载唯一本地提交 64ff438 的构建检出 |

## 汇总

| 级别 | 可释放 | 决策 |
|---|---:|---|
| A 级（含散落临时目录 + 回收站） | ≈ 24.369 GiB | 直接删 |
| B 级（可再生成缓存） | ≈ 56.364 GiB | 直接删 |
| 待确认（派生中间 LAS） | ≈ 53.207 GiB | 需你点头 |
