# E 盘清理结果报告

- 执行时间：2026-10-01 15:13:35 +08:00
- 执行范围：A 级（被 r6 取代的旧构建/旧 ZIP/临时测试目录）+ B 级（可再生成缓存）
- 执行状态：**已完成真实删除**
- 容量口径：GiB = 1024³；释放量以 E 盘可用空间**实测差值**为准

## 一、结果

```text
FREE_BEFORE        =  26,615,201,792 bytes  (24.787 GiB)
FREE_AFTER         =  97,537,658,880 bytes  (90.839 GiB)
MEASURED_FREED     =  70,922,457,088 bytes  (66.052 GiB)
LOGICAL_FREED      =  68,121,268,999 bytes  (63.443 GiB)
TARGETS_DELETED    = 221
TARGETS_FAILED     = 0
PARTIAL_TARGETS    = 6   (含受保护子树的目录)
RECYCLE_BIN        = 已清空（全部固定磁盘）
```

E 盘可用空间：**24.787 GiB → 90.839 GiB**，实测释放 **66.052 GiB**。

> 实测释放略高于逻辑大小（+2.609 GiB），差值来自 NTFS 簇分配与删除后的目录项/MFT 回收。**以实测值为准。**

## 二、安全闸门拦截记录（未删除）

### 2.1 整体拒绝

| 路径 | 原因 |
|---|---|
| `E:\2026.8.31-2\2026.08.31-2.zip` | archive contains 2026.08.31-2/artifacts/project-runs/source-shared/965efb8f9f80bb504b600b1434ada1799c6999e105c8530e99bd85e06912aa5b/ground-regions/aa83abefcda9e077ff2da5f62c023cb14171dde87b89deeeede9cd76784ec0d6/classified_hag.las |

`E:\2026.8.31-2\2026.08.31-2.zip` 是 3.92 GiB 的旧解压源，但 ZIP 内**含 4 个 `.las`**。按「归档含受保护数据即拒绝」规则整包保留，未做任何删减。

### 2.2 部分删除：目标目录内被保留的子目录

共 14 个子树因含真点云或模型被原地保留，其所属目标的其余内容正常删除。

| 目标 | 保留子目录 | 依据 |
|---|---|---|
| `E:\TreeLineWarning-fast\cache\giro3d_full_pdal` | `E:\TreeLineWarning-fast\cache\giro3d_full_pdal\dataset` | .laz 847.9 MiB >= 10 MiB (real point cloud) |
| `E:\TreeLineWarning-fast\cache\giro3d_full_source` | `E:\TreeLineWarning-fast\cache\giro3d_full_source\dataset` | .laz 512.1 MiB >= 10 MiB (real point cloud) |
| `E:\TreeLineWarning-fast\cache\p3a_giro3d` | `E:\TreeLineWarning-fast\cache\p3a_giro3d\dataset` | .laz 4.63 MiB outside a test path |
| `E:\TreeLineWarning-fast\cache\p3a_giro3d` | `E:\TreeLineWarning-fast\cache\p3a_giro3d\python\giro3d-venv\Lib\site-packages` | .pth model file |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\ground_benchmark` | .las 447.8 MiB >= 10 MiB (real point cloud) |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\scan_large_multi_tree_roi\compact_ground` | .las 5.20 MiB outside a test path |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\scan_large_multi_tree_roi\selected_computation` | .las 0.30 MiB outside a test path |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\scan_large_multi_tree_roi\selected_ground` | .las 0.35 MiB outside a test path |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\scan_large_preprocess_weighted_v4` | .las 0.67 MiB outside a test path |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\scene_simple\ground` | .las 447.8 MiB >= 10 MiB (real point cloud) |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\scene_simple_ab_v2\ground` | .las 447.8 MiB >= 10 MiB (real point cloud) |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | `E:\TreeLineWarning-fast\artifacts\fast_research\scene_simple_confirmed\ground` | .las 447.8 MiB >= 10 MiB (real point cloud) |
| `E:\TreeLineWarning-fast\.final-test-tmp-full-20260913` | `E:\TreeLineWarning-fast\.final-test-tmp-full-20260913\test_local_raw_cache_bounded_t0` | .las 13.7 MiB >= 10 MiB (real point cloud) |
| `E:\TreeLineWarning-fast\.pt2` | `E:\TreeLineWarning-fast\.pt2\test_empty_las_has_clear_empty0` | .las 0.00 MiB outside a test path |

### 2.3 豁免的测试 fixture

删除前逐个扫描，**3,352 个**小型 `.las` / `.laz` / `.pth` 文件被判定为 pytest fixture 并放行。判定规则：

- `.las`/`.laz`/`.copc.laz` ≥ 10 MiB → 一律保护（视为真点云）
- `.las`/`.laz`/`.copc.laz` < 10 MiB 且位于测试路径 → 视为 fixture，放行
- `.las`/`.laz` < 10 MiB 但**不在**测试路径 → 保护（如 `.pt2` 内的文件）
- `.onnx` / `.pt` / `.ckpt` → 一律保护
- `.pth` ≥ 1 MiB → 保护（模型权重）；`.pth` < 1 MiB → 放行（setuptools 路径垫片）
- 归档（`.zip`）内含任一受保护条目 → 整包保护

## 三、删除清单（按体积降序）

| # | 路径 | 释放 | 文件数 | 备注 |
|---:|---|---:|---:|---|
| 1 | `E:\TreeLineWarning-fast\cache\tower_ordered_project` | 33.312 GiB | 6,060 |  |
| 2 | `E:\TreeLineWarning-source-cache` | 7.019 GiB | 4,020 |  |
| 3 | `E:\TreeLineWarning-fast\cache\mamba_root` | 5.099 GiB | 24,619 |  |
| 4 | `E:\TreeLineWarning-fast\cache\giro3d_pdal_env` | 3.864 GiB | 20,581 |  |
| 5 | `E:\TreeLineWarning-fast\cache\tower_raw_tiles` | 3.131 GiB | 50 |  |
| 6 | `E:\TreeLineWarning-fast\artifacts\final-product-dist-20260930-r5` | 2.103 GiB | 2,861 |  |
| 7 | `E:\TreeLineWarning-fast\artifacts\final-product-dist-20260929-r4` | 2.103 GiB | 2,861 |  |
| 8 | `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-20260930-r5.zip` | 1.473 GiB | 1 |  |
| 9 | `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-Complete-20260929.zip` | 1.469 GiB | 1 |  |
| 10 | `E:\TreeLineWarning-fast\.final-test-tmp-full-20260913` | 0.520 GiB | 15,747 | 4 个保留子树 |
| 11 | `E:\TreeLineWarning-fast\.test-tmp-delivery-sol-baseline-20260830` | 0.337 GiB | 12,513 |  |
| 12 | `E:\TreeLineWarning-fast\.test-tmp-clearance-sol-broad-20260830-a` | 0.226 GiB | 8,617 |  |
| 13 | `E:\TreeLineWarning-fast\artifacts\fast_research` | 0.209 GiB | 125 | 14 个保留子树 |
| 14 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-final` | 0.172 GiB | 1,249 |  |
| 15 | `E:\TreeLineWarning-fast\artifacts\final-product-20260929` | 0.171 GiB | 12,713 |  |
| 16 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-staging` | 0.128 GiB | 114 |  |
| 17 | `E:\TreeLineWarning-fast\artifacts\dbh-format-native-runtime-final-20260928` | 0.110 GiB | 72 |  |
| 18 | `E:\TreeLineWarning-fast\artifacts\review-fix-native-runtime-20260915` | 0.110 GiB | 70 |  |
| 19 | `E:\TreeLineWarning-fast\artifacts\dbh-format-native-runtime-20260928` | 0.110 GiB | 70 |  |
| 20 | `E:\TreeLineWarning-fast\artifacts\dbh-format-native-runtime-v3-20260928` | 0.110 GiB | 69 |  |
| 21 | `E:\TreeLineWarning-fast\cache\p3a_giro3d` | 0.098 GiB | 6,022 | 372 个保留子树 |
| 22 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow_hardened_20260817_track_evidence` | 0.086 GiB | 61 |  |
| 23 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260930-r5` | 0.077 GiB | 18 |  |
| 24 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260929-r4` | 0.077 GiB | 18 |  |
| 25 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260929-r3` | 0.077 GiB | 18 |  |
| 26 | `E:\TreeLineWarning-fast\artifacts\final-product-work-20260929-r2` | 0.077 GiB | 18 |  |
| 27 | `E:\TreeLineWarning-fast\artifacts\dbh-format-release-work-final-20260928` | 0.077 GiB | 18 |  |
| 28 | `E:\TreeLineWarning-fast\artifacts\dbh-format-release-work-v3-20260928` | 0.077 GiB | 18 |  |
| 29 | `E:\TreeLineWarning-fast\artifacts\dbh-format-release-work-20260928` | 0.077 GiB | 18 |  |
| 30 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-work-20260915` | 0.077 GiB | 18 |  |
| 31 | `E:\TreeLineWarning-fast\artifacts\real-closure-build-work-20260913-606127c` | 0.077 GiB | 18 |  |
| 32 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow_hardened_20260817` | 0.066 GiB | 47 |  |
| 33 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow_hardened_20260817_track_evidence_exact_polyline` | 0.063 GiB | 45 |  |
| 34 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-broad2` | 0.059 GiB | 3,662 |  |
| 35 | `E:\TreeLineWarning-fast\.test-tmp-giro3d-final-regression-20260823-01` | 0.055 GiB | 2,463 |  |
| 36 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-full-suite-20260822` | 0.050 GiB | 6,714 |  |
| 37 | `E:\TreeLineWarning-fast\cache\wire_v2_shadow` | 0.049 GiB | 27 |  |
| 38 | `E:\TreeLineWarning-fast\.test-tmp-dbh-freeze-20260930` | 0.041 GiB | 537 |  |
| 39 | `E:\TreeLineWarning-fast\.test-tmp-dbh-close-all-20260930` | 0.037 GiB | 501 |  |
| 40 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-full-first-failure-20260822` | 0.037 GiB | 858 |  |
| 41 | `E:\TreeLineWarning-fast\.test-tmp-dbh-freshness-20260930` | 0.036 GiB | 373 |  |
| 42 | `E:\TreeLineWarning-fast\.test-tmp-dbh-close-20260930` | 0.035 GiB | 495 |  |
| 43 | `E:\TreeLineWarning-fast\.test-tmp-phase-b-broad` | 0.031 GiB | 2,206 |  |
| 44 | `E:\TreeLineWarning-fast\.test-tmp-dbh-fixes-20260929` | 0.029 GiB | 340 |  |
| 45 | `E:\TreeLineWarning-fast\cache\debug` | 0.024 GiB | 7 |  |
| 46 | `E:\TreeLineWarning-fast\.wire_v2_high_recall_isolated` | 0.017 GiB | 515 |  |
| 47 | `E:\TreeLineWarning-fast\.test-tmp-views-broad-20260929` | 0.014 GiB | 1,899 |  |
| 48 | `E:\TreeLineWarning-fast\artifacts\real-closure-release-20260913-606127c` | 0.011 GiB | 12 |  |
| 49 | `E:\TreeLineWarning-fast\.test-tmp-proj-unicode-20260830-01` | 0.010 GiB | 20 |  |
| 50 | `E:\TreeLineWarning-fast\artifacts\dbh-investigation-20260928` | 0.008 GiB | 653 |  |
| 51 | `E:\TreeLineWarning-fast\.test-tmp-full-final` | 0.004 GiB | 320 |  |
| 52 | `E:\TreeLineWarning-fast\.test-tmp-full` | 0.004 GiB | 320 |  |
| 53 | `E:\TreeLineWarning-fast\.wire_endpoint_closure` | 0.004 GiB | 4 |  |
| 54 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-broad-20260822` | 0.003 GiB | 352 |  |
| 55 | `E:\TreeLineWarning-fast\.test-tmp-tower-finalv2b` | 0.003 GiB | 121 |  |
| 56 | `E:\TreeLineWarning-fast\.test-tmp-r6-editorfix` | 0.003 GiB | 1,107 |  |
| 57 | `E:\TreeLineWarning-fast\.test-tmp-r5-editorfix` | 0.003 GiB | 1,096 |  |
| 58 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260929c` | 0.003 GiB | 385 |  |
| 59 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260929a` | 0.003 GiB | 374 |  |
| 60 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928h` | 0.003 GiB | 338 |  |
| 61 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928g` | 0.003 GiB | 326 |  |
| 62 | `E:\TreeLineWarning-fast\.test-tmp-phase-b-final` | 0.003 GiB | 630 |  |
| 63 | `E:\TreeLineWarning-fast\.test-tmp-tower-finalv2` | 0.003 GiB | 121 |  |
| 64 | `E:\TreeLineWarning-fast\.test-tmp-dbh-final-20260929` | 0.003 GiB | 298 |  |
| 65 | `E:\TreeLineWarning-fast\.test-tmp-dbh-ref2-20260929` | 0.003 GiB | 286 |  |
| 66 | `E:\TreeLineWarning-fast\.test-tmp-dbh-ref4-20260929` | 0.003 GiB | 286 |  |
| 67 | `E:\TreeLineWarning-fast\.test-tmp-dbh-broad-20260929` | 0.002 GiB | 272 |  |
| 68 | `E:\TreeLineWarning-fast\.test-tmp-dbh-ref5-20260929` | 0.002 GiB | 184 |  |
| 69 | `E:\TreeLineWarning-fast\.test-tmp-dbh-ref-20260929` | 0.002 GiB | 172 |  |
| 70 | `E:\TreeLineWarning-fast\.test-tmp-reference-closure-20260929` | 0.002 GiB | 211 |  |
| 71 | `E:\TreeLineWarning-fast\.test-tmp-reference-final-20260929` | 0.002 GiB | 211 |  |
| 72 | `E:\TreeLineWarning-fast\.test-tmp-reference-broad-20260929` | 0.002 GiB | 199 |  |
| 73 | `E:\TreeLineWarning-fast\.test-tmp-reference-20260929` | 0.002 GiB | 171 |  |
| 74 | `E:\TreeLineWarning-fast\.pytest_tmp` | 0.002 GiB | 1,147 |  |
| 75 | `E:\TreeLineWarning-fast\.test-tmp-ruff-review-20260929` | 0.002 GiB | 189 |  |
| 76 | `E:\TreeLineWarning-fast\.test-tmp-dbh-20260929` | 0.002 GiB | 158 |  |
| 77 | `E:\TreeLineWarning-fast\.test-tmp-views-native2-20260929` | 0.002 GiB | 73 |  |
| 78 | `E:\TreeLineWarning-fast\.test-tmp-tower-ambiguous` | 0.002 GiB | 25 |  |
| 79 | `E:\TreeLineWarning-fast\cache\tower_review_project` | 0.002 GiB | 382 |  |
| 80 | `E:\TreeLineWarning-fast\.test-tmp-tower-final20260929` | 0.002 GiB | 118 |  |
| 81 | `E:\TreeLineWarning-fast\.test-tmp-reference-focused-20260929` | 0.002 GiB | 48 |  |
| 82 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928k` | 0.002 GiB | 65 |  |
| 83 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928e` | 0.002 GiB | 37 |  |
| 84 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928f` | 0.002 GiB | 37 |  |
| 85 | `E:\TreeLineWarning-fast\.test-tmp-tower-reopen` | 0.002 GiB | 41 |  |
| 86 | `E:\TreeLineWarning-fast\.test-tmp-tower-reopen2` | 0.002 GiB | 41 |  |
| 87 | `E:\TreeLineWarning-fast\.test-tmp-tower-generality` | 0.002 GiB | 35 |  |
| 88 | `E:\TreeLineWarning-fast\.wire_gap_rescue_round1` | 0.001 GiB | 18 |  |
| 89 | `E:\TreeLineWarning-fast\.test-tmp-clearance-sol-native-bounded-20260830-b` | 0.001 GiB | 800 |  |
| 90 | `E:\TreeLineWarning-fast\.test-tmp-clearance-sol-native-bounded-20260830-a` | 0.001 GiB | 796 |  |
| 91 | `E:\TreeLineWarning-fast\.test-tmp-clearance-broad-02` | 0.001 GiB | 800 |  |
| 92 | `E:\TreeLineWarning-fast\.test-tmp-views-cli-20260929` | 0.001 GiB | 31 |  |
| 93 | `E:\TreeLineWarning-fast\.test-tmp-lineage-sol-regression-20260830-b` | 0.001 GiB | 778 |  |
| 94 | `E:\TreeLineWarning-fast\.test-tmp-views-close-20260929` | 0.001 GiB | 288 |  |
| 95 | `E:\TreeLineWarning-fast\.wire_endpoint_recovery_round2` | 0.001 GiB | 4 |  |
| 96 | `E:\TreeLineWarning-fast\.test-tmp-editor-replay-broad2-20260930` | 0.001 GiB | 285 |  |
| 97 | `E:\TreeLineWarning-fast\.test-tmp-editor-replay-broad-20260930` | 0.001 GiB | 285 |  |
| 98 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-target` | 0.001 GiB | 269 |  |
| 99 | `E:\TreeLineWarning-fast\.test-tmp-editor-persist-broad-20260930` | 0.001 GiB | 274 |  |
| 100 | `E:\TreeLineWarning-fast\.test-tmp-delivery-lifecycle-20260929` | 0.001 GiB | 189 |  |
| 101 | `E:\TreeLineWarning-fast\.test-tmp-dbh-ref3-20260929` | 0.001 GiB | 104 |  |
| 102 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel7` | 0.001 GiB | 285 |  |
| 103 | `E:\TreeLineWarning-fast\.test-tmp-20260823-performance` | 0.001 GiB | 156 |  |
| 104 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-focused4-20260822` | 0.001 GiB | 156 |  |
| 105 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-focused7-20260822` | 0.001 GiB | 156 |  |
| 106 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-focused6-20260822` | 0.001 GiB | 156 |  |
| 107 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-focused5-20260822` | 0.001 GiB | 155 |  |
| 108 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-focused3-20260822` | 0.001 GiB | 155 |  |
| 109 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-focused2-20260822` | 0.001 GiB | 155 |  |
| 110 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-focused-20260822` | 0.001 GiB | 155 |  |
| 111 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-20260822` | 0.001 GiB | 137 |  |
| 112 | `E:\TreeLineWarning-fast\.test-tmp-integration-revision-20260929` | 0.001 GiB | 67 |  |
| 113 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel5` | 0.001 GiB | 238 |  |
| 114 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel6` | 0.001 GiB | 238 |  |
| 115 | `E:\TreeLineWarning-fast\.test-tmp-tower-exclusion-final` | 0.001 GiB | 43 |  |
| 116 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928c` | 0.001 GiB | 209 |  |
| 117 | `E:\TreeLineWarning-fast\.test-tmp-trace-state` | 0.001 GiB | 109 |  |
| 118 | `E:\TreeLineWarning-fast\.test-tmp-review-ux-js-20260822` | 0.000 GiB | 106 |  |
| 119 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel4` | 0.000 GiB | 234 |  |
| 120 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel3` | 0.000 GiB | 234 |  |
| 121 | `E:\TreeLineWarning-fast\.test-tmp-final-ux` | 0.000 GiB | 106 |  |
| 122 | `E:\TreeLineWarning-fast\.test-tmp-ux-tests` | 0.000 GiB | 106 |  |
| 123 | `E:\TreeLineWarning-fast\TreeLineWarning-fast.test-tmp` | 0.000 GiB | 106 |  |
| 124 | `E:\TreeLineWarning-fast\.test-tmp` | 0.000 GiB | 106 |  |
| 125 | `E:\TreeLineWarning-fast\.test-tmp-tower-editor` | 0.000 GiB | 14 |  |
| 126 | `E:\TreeLineWarning-fast\.test-tmp-tower-exclusion` | 0.000 GiB | 6 |  |
| 127 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928d` | 0.000 GiB | 112 |  |
| 128 | `E:\TreeLineWarning-fast\.test-tmp-views-cli2-20260929` | 0.000 GiB | 23 |  |
| 129 | `E:\TreeLineWarning-fast\.test-tmp-delivery-sol-export-review-20260830` | 0.000 GiB | 185 |  |
| 130 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel2` | 0.000 GiB | 166 |  |
| 131 | `E:\TreeLineWarning-fast\.test-tmp-folder-revision-20260929` | 0.000 GiB | 31 |  |
| 132 | `E:\TreeLineWarning-fast\.test-tmp-views-generic-20260929` | 0.000 GiB | 128 |  |
| 133 | `E:\TreeLineWarning-fast\.test-tmp-phase-b-regression` | 0.000 GiB | 480 |  |
| 134 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-reasons2` | 0.000 GiB | 38 |  |
| 135 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-reasons` | 0.000 GiB | 38 |  |
| 136 | `E:\TreeLineWarning-fast\.final-test-tmp-baseline` | 0.000 GiB | 206 |  |
| 137 | `E:\TreeLineWarning-fast\.test-tmp-delivery-revision-20260929` | 0.000 GiB | 48 |  |
| 138 | `E:\TreeLineWarning-fast\.final-test-tmp-focused-02` | 0.000 GiB | 123 |  |
| 139 | `E:\TreeLineWarning-fast\.final-test-tmp-focused-03` | 0.000 GiB | 123 |  |
| 140 | `E:\TreeLineWarning-fast\.final-test-tmp-focused-01` | 0.000 GiB | 122 |  |
| 141 | `E:\TreeLineWarning-fast\.test-tmp-giro3d-final-timing-20260823-01` | 0.000 GiB | 67 |  |
| 142 | `E:\TreeLineWarning-fast\.test-tmp-giro3d-closure-20260823-01` | 0.000 GiB | 67 |  |
| 143 | `E:\TreeLineWarning-fast\.test-tmp-giro3d-closure-20260823-02` | 0.000 GiB | 67 |  |
| 144 | `E:\TreeLineWarning-fast\.test-tmp-giro3d-closure-20260823-03` | 0.000 GiB | 67 |  |
| 145 | `E:\TreeLineWarning-fast\.test-tmp-native-viewer-closure-20260823-02` | 0.000 GiB | 33 |  |
| 146 | `E:\TreeLineWarning-fast\.test-tmp-native-viewer-closure-20260823-03` | 0.000 GiB | 33 |  |
| 147 | `E:\TreeLineWarning-fast\.test-tmp-tower-stage-b` | 0.000 GiB | 35 |  |
| 148 | `E:\TreeLineWarning-fast\.test-tmp-ux` | 0.000 GiB | 18 |  |
| 149 | `E:\TreeLineWarning-fast\.test-tmp-editor-persist-20260930` | 0.000 GiB | 33 |  |
| 150 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928b` | 0.000 GiB | 33 |  |
| 151 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-broad-cache2` | 0.000 GiB | 4 |  |
| 152 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-broad-cache` | 0.000 GiB | 5 |  |
| 153 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928a` | 0.000 GiB | 22 |  |
| 154 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260929b` | 0.000 GiB | 22 |  |
| 155 | `E:\TreeLineWarning-fast\.ruff_cache` | 0.000 GiB | 80 |  |
| 156 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928i` | 0.000 GiB | 81 |  |
| 157 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel` | 0.000 GiB | 28 |  |
| 158 | `E:\TreeLineWarning-fast\.test-tmp-tower-wire-admission` | 0.000 GiB | 37 |  |
| 159 | `E:\TreeLineWarning-fast\.test-tmp-editor-equal-20260930` | 0.000 GiB | 11 |  |
| 160 | `E:\TreeLineWarning-fast\.wire_endpoint_recovery_round2_run.log` | 0.000 GiB | 1 |  |
| 161 | `E:\TreeLineWarning-fast\.test-tmp-dbh-close-retry-20260930` | 0.000 GiB | 22 |  |
| 162 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-build` | 0.000 GiB | 79 |  |
| 163 | `E:\TreeLineWarning-fast\.test-tmp-views-final-20260929` | 0.000 GiB | 84 |  |
| 164 | `E:\TreeLineWarning-fast\.test-tmp-views-20260929` | 0.000 GiB | 74 |  |
| 165 | `E:\TreeLineWarning-fast\.test-tmp-views-20260929b` | 0.000 GiB | 74 |  |
| 166 | `E:\TreeLineWarning-fast\.test-tmp-views-20260929c` | 0.000 GiB | 74 |  |
| 167 | `E:\TreeLineWarning-fast\.test-tmp-r6-neighbor` | 0.000 GiB | 22 |  |
| 168 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-final-smoke2-20260915` | 0.000 GiB | 9 |  |
| 169 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-final-cache` | 0.000 GiB | 4 |  |
| 170 | `E:\TreeLineWarning-fast\.test-tmp-r6-editorfix-cache` | 0.000 GiB | 4 |  |
| 171 | `E:\TreeLineWarning-fast\.test-tmp-dbh-close-api-20260930` | 0.000 GiB | 10 |  |
| 172 | `E:\TreeLineWarning-fast\.test-tmp-r5-editorfix-cache` | 0.000 GiB | 4 |  |
| 173 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-acceptance-20260915` | 0.000 GiB | 4 |  |
| 174 | `E:\TreeLineWarning-fast\artifacts\final-acceptance-20260913` | 0.000 GiB | 1 |  |
| 175 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-cache` | 0.000 GiB | 5 |  |
| 176 | `E:\TreeLineWarning-fast\artifacts\final-product-20260930` | 0.000 GiB | 1 |  |
| 177 | `E:\TreeLineWarning-fast\.test-tmp-views-close-cache` | 0.000 GiB | 4 |  |
| 178 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-target-cache` | 0.000 GiB | 4 |  |
| 179 | `E:\TreeLineWarning-fast\.test-tmp-views-gate3-20260929` | 0.000 GiB | 80 |  |
| 180 | `E:\TreeLineWarning-fast\.test-tmp-views-gaterecovery-20260929` | 0.000 GiB | 77 |  |
| 181 | `E:\TreeLineWarning-fast\.test-tmp-views-20260929d` | 0.000 GiB | 70 |  |
| 182 | `E:\TreeLineWarning-fast\tmp-check` | 0.000 GiB | 4 |  |
| 183 | `E:\TreeLineWarning-fast\treeline_warning.egg-info` | 0.000 GiB | 5 |  |
| 184 | `E:\TreeLineWarning-fast\.test-tmp-tower-editor-undo` | 0.000 GiB | 12 |  |
| 185 | `E:\TreeLineWarning-fast\.test-tmp-views-gate3-cache` | 0.000 GiB | 4 |  |
| 186 | `E:\TreeLineWarning-fast\.test-tmp-views-gaterecovery-cache` | 0.000 GiB | 4 |  |
| 187 | `E:\TreeLineWarning-fast\.test-tmp-metrics-path-20260929` | 0.000 GiB | 32 |  |
| 188 | `E:\TreeLineWarning-fast\.test-tmp-metrics-path-final-20260929` | 0.000 GiB | 32 |  |
| 189 | `E:\TreeLineWarning-fast\.test-tmp-views-contract-20260929` | 0.000 GiB | 33 |  |
| 190 | `E:\TreeLineWarning-fast\.test-tmp-views-tower-20260929` | 0.000 GiB | 33 |  |
| 191 | `E:\TreeLineWarning-fast\.test-tmp-views-ui2-20260929` | 0.000 GiB | 33 |  |
| 192 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-reasons-cache` | 0.000 GiB | 4 |  |
| 193 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-reasons2-cache` | 0.000 GiB | 4 |  |
| 194 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel-cache` | 0.000 GiB | 5 |  |
| 195 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-acceptance-20260915-final` | 0.000 GiB | 4 |  |
| 196 | `E:\TreeLineWarning-fast\.test-tmp-views-contract-cache` | 0.000 GiB | 4 |  |
| 197 | `E:\TreeLineWarning-fast\.test-tmp-views-tower-cache` | 0.000 GiB | 4 |  |
| 198 | `E:\TreeLineWarning-fast\.test-tmp-views-ui2-cache` | 0.000 GiB | 4 |  |
| 199 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-20260928j` | 0.000 GiB | 36 |  |
| 200 | `E:\TreeLineWarning-fast\.test-tmp-r6-neighbor-cache` | 0.000 GiB | 4 |  |
| 201 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-staging-cache` | 0.000 GiB | 4 |  |
| 202 | `E:\TreeLineWarning-fast\.test-tmp-views-cli-cache` | 0.000 GiB | 5 |  |
| 203 | `E:\TreeLineWarning-fast\cache\giro3d_full_pdal` | 0.000 GiB | 1 | 1 个保留子树 |
| 204 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel3-cache` | 0.000 GiB | 5 |  |
| 205 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-build-cache` | 0.000 GiB | 4 |  |
| 206 | `E:\TreeLineWarning-fast\.test-tmp-views-cli2-cache` | 0.000 GiB | 4 |  |
| 207 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel4-cache` | 0.000 GiB | 5 |  |
| 208 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel5-cache` | 0.000 GiB | 5 |  |
| 209 | `E:\TreeLineWarning-fast\cache\giro3d_full_source` | 0.000 GiB | 1 | 1 个保留子树 |
| 210 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel7-cache` | 0.000 GiB | 4 |  |
| 211 | `E:\TreeLineWarning-fast\.test-tmp-testfinish-parallel6-cache` | 0.000 GiB | 4 |  |
| 212 | `E:\TreeLineWarning-fast\.test-tmp-review-sync-pytest` | 0.000 GiB | 5 |  |
| 213 | `E:\TreeLineWarning-fast\.test-tmp-views-gaterecovery2-cache` | 0.000 GiB | 4 |  |
| 214 | `E:\TreeLineWarning-fast\.test-tmp-views-native-cache` | 0.000 GiB | 4 |  |
| 215 | `E:\TreeLineWarning-fast\.test-tmp-views-native2-cache` | 0.000 GiB | 4 |  |
| 216 | `E:\TreeLineWarning-fast\artifacts\review-load-fix-final-smoke-20260915` | 0.000 GiB | 1 |  |
| 217 | `E:\TreeLineWarning-fast\actual` | 0.000 GiB | 1 |  |
| 218 | `E:\TreeLineWarning-fast\bounded` | 0.000 GiB | 1 |  |
| 219 | `E:\TreeLineWarning-fast\T001` | 0.000 GiB | 1 |  |
| 220 | `E:\TreeLineWarning-fast\.pytest_cache` | 0.000 GiB | 0 |  |
| 221 | `E:\TreeLineWarning-fast\.pt2` | 0.000 GiB | 0 | 1 个保留子树 |

合计 **63.443 GiB / 162,318 个文件**。

## 四、失败项

**无。** 221 个目标全部删除成功，0 失败。

## 五、保护对象复核（清理后逐一验证存在）

| 对象 | 状态 |
|---|---|
| 源点云 `2026-08-07_16没54-10_PointCloud_colorized.las` (5.87 GiB) | PASS |
| 源点云 `9.20丰台北路_colorized.las` (2.25 GiB) | PASS |
| 当前最新交付 `TreeLineWarning-20260930-r6.zip` | PASS |
| 当前交付解包 EXE | PASS |
| Git 仓库 `.git` | PASS |
| 源码 `core/stage_b/dbh_review.py`（未提交改动） | PASS |
| viewer 依赖 `PotreeConverter.exe` | PASS |
| 构建环境 `.venv311` | PASS |
| 当前验收证据 `acceptance_evidence/product-fixes-20260929` | PASS |
| 工作区入口 `workspace/input/pointcloud` | PASS |
| 唯一未推送提交的构建检出 | PASS |
| 被保留的研究点云 `fast_research/ground_benchmark/scan_small_smrf_hag.las` | PASS |

### Git 完整性

```text
HEAD               = 64ff43820aa51877d800b387c2ca063c1be7b5e2
HEAD subject       = fix(review): keep the first review read alive across a cold display hydration
git fsck           = PASS (exit 0, no output)
worktrees          = 9 (未变)
stashes            = 2 (未变，9/13 备份仍在)
tracked modified   = 64 个文件改动仍在（9/28-9/30 未提交工作完好）
```

本次清理**没有触碰任何受版本控制的文件**，只删除了 gitignored 的缓存/构建/临时产物。

## 六、仍然可以继续清理的（本次未删，需你决定）

| 目标 | 大小 | 说明 |
|---|---:|---|
| `acceptance_evidence/product-fixes-20260929` | 14.70 GiB | 9/29-9/30 验收证据（与 `E:\1\3` 部分重复）|
| `E:\TreeLineWarning-memory-acceptance-20260822` | 14.80 GiB | 含 2.6 GiB 真 LAS，删后无法回归 |
| `E:\TreeLineWarning-review-risk-polish` | 12.90 GiB | 8/27 git worktree |
| `E:\TreeLineWarning-RealWorldClosure-20260913` | 10.69 GiB | 9/13 闭环复核 |
| `E:\2026.8.31-2\2026.08.31-2` | 9.75 GiB | 8/31 旧发布树（含 4 个 LAS）|
| `E:\TreeLineWarning-RealWorldColdAfter-20260913` | 7.28 GiB | 9/13 冷启复核 |
| `E:\2026.8.31-2\2026.08.31-2.zip` | 3.92 GiB | 旧 ZIP，闸门因内含 LAS 拒绝 |
| `artifacts/fast_research`（残留） | 1.93 GiB | 含真点云，已保留 |
| `E:\1\3` | 3.70 GiB | 9/30 任务工作区（与 product-fixes 重复）|
| `E:\pagefile.sys` | 15.76 GiB | 可移到 C 盘（C 尚有 61 GiB 空闲），需重启 |
| `E:\Tencent Games` | 34.46 GiB | VALORANT，只能走游戏卸载器 |

## 七、证据文件

| 文件 | 内容 |
|---|---|
| `artifacts/_cleanup_20261001/DELETE_MANIFEST.json` | 删除前逐个目标的安全判定结果 |
| `artifacts/_cleanup_20261001/delete_results.json` | 每个目标的删除结果与字节数、前后可用空间 |
| `DISK_CLEANUP_PLAN_20261001.md` | 删除前的逐项方案（仅扫描）|
| `scripts/_cleanup_ab.py` | 执行器（含安全闸门与保留子树逻辑）|
