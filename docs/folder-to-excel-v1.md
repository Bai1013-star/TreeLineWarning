# FOLDER_TO_EXCEL V1 — 文件夹到 Excel 流水线

一次完成：**Folder → Manual Span → Science → Editable Trees → Clearance → Excel**。

## 最终用户流程

1. 把一个 `LAS` / `LAZ` 放到 `workspace/input/pointcloud/`。
2. 双击 `RUN_FOLDER_TO_EXCEL.bat`（可带参数：`RUN_FOLDER_TO_EXCEL.bat <workspace>`）。
3. 浏览器打开 **STEP 1**：人工选择一个档距（俯视图点击杆塔 A / B）。
4. 后台自动执行：导线（Wire V2）→ 植被 → 导线/杆塔精确剔除 → CROWN_STEM_HYBRID 单木分割。
5. 浏览器进入 **STEP 2**：人工审核单木（删除 / 多选合并 / 撤销）。
6. 点击「确认单木并继续」。
7. 后台执行：最终单木几何 → 精确净空 → 危险 → 持久身份。
8. 浏览器进入 **STEP 3**：查看最终结果。
9. 得到 `03_export/Prediction_Input.xlsx`（`Trees` + `Run_Metadata` 两个 Sheet）。

三个步骤是同一个本地服务器下的三个 route（`/step1` `/step2` `/step3`），共享同一个
运行/会话状态，不是三个互不相干的静态 HTML 程序。

## 工作区布局

```
workspace/
├─ input/pointcloud/source.las | source.laz    （必须恰好一个，否则 fail closed）
├─ source_cache/                                科学暂存（预处理/地面/区域缓存）
│  └─ science/                                  ManualSingleSpanRun 的 run_root
├─ 01_span_selection/span_selection.json        确认后的档距选择
├─ 02_tree_analysis/
│  ├─ tree_segmentation_original.{npz,json}     算法单木原始结果（绝不覆盖）
│  ├─ tree_edits.json                           人工编辑日志（DELETE/MERGE）
│  └─ tree_segmentation_final.{npz,json}        最终单木结果
└─ 03_export/
   ├─ Prediction_Input.xlsx
   └─ 03_final_result.html
```

## 复用（不重写）

| 科学步骤 | 调用 |
|---|---|
| 地面 / HAG | `ManualSingleSpanRun.initialize()`（PDAL SMRF + HAG-NN，可复用同源地面缓存） |
| Wire V2 | `run._stage_wire()` → `run_span_wire_pipeline`，确认导线成员 = `confirmed_member_source_index` |
| 高召回植被 | `run._stage_regions()` → `build_weighted_fast_vegetation` |
| 导线/杆塔精确剔除 | `subset_raw_region_excluding_source_indices`（原始索引相减，非距离缓冲） |
| 单木分割 | `run_span_tree_pipeline`（`TREE_BACKEND = CROWN_STEM_HYBRID`，在剔除后的输入上运行） |
| 协调/全局 ID | `reconcile_trees` |
| 最终几何 | `compute_tree_metrics`（HAG 高度 / 主轴冠幅 / DBH） |
| 精确净空 | `run_manual_clearance`（人工编辑之后） |
| 危险 | `apply_risk_threshold` |

## 关键不变量

- `FINAL_TREE_INPUT = VEGETATION − CONFIRMED_WIRE − CONFIRMED_TOWER`（仅按 full-resolution
  `original_index` 精确相减；显示采样绝不进入科学）。
- 接近导线的植被不会仅因距离被剔除——只有确认导线成员会被剔除。
- 算法单木原始结果与人工编辑日志分层保存（original / edits / final）。
- MERGE = 精确 `original_index` 并集，无重复、无丢失、XYZ 不变。
- 每棵最终单木一个唯一 `persistent_tree_id`；`UNIQUE_PERSISTENT_TREE_IDS == FINAL_TREE_COUNT`。
- 缺失值语义：`crown_width_m` / `diameter_cm` 缺失时写空（blank），绝不写 `0`；
  NaN / Inf 一律清洗为空。

## 环境说明

- 地面阶段需要 PDAL SMRF。若本机缺少 PDAL，但该源点云存在历史地面缓存
  （`artifacts/one_click_runs/` 或 `artifacts/manual_single_span_runs/` 中同 SHA 的
  `02_ground/ground_hag_result.json`），会自动复用；否则以清晰中文错误失败。
- 服务器仅监听 `127.0.0.1`。
