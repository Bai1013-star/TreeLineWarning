# Survey V0.1 REAL Multi-Span E2E Gate — 实测报告

> 最终续验（第 11 节）：**FINAL_VERDICT = FAIL；首个新失败 = EXCEL_EXPORT_MISSING_TOWER_IDENTITIES（Step 11）**。两次 REAL lineage/净空、真实 DELETE/UNDO、finalize 与 post-editor 净空均 PASS；正式 Excel 的 16 行均缺少起止杆塔 ID，故停止，未执行 finalized restart、未 commit/push。以下原第 1–10 节全文保留，其中“最新”“本轮”等措辞均属当时历史。

日期：2026-08-30（Asia/Shanghai）。Gate：`SURVEY_V01_GATE_B_REAL_MULTISPAN_E2E`。

**最新续验：POST_FIX_RESULT = FAIL；首个新 authoritative failure layer = CLEARANCE_PUBLICATION_OR_ORDER_DEPENDENCE。** 候选 `c32cc489344d2bfe16005ba6951dbf221d522e4a` 的两次新 REAL 运行已通过成员、persistent ID、merged keys 和 Span 映射比较，但净空仍随提交顺序改变。详见第 10 节；未进入人工编辑/finalize/Excel，未推送。

**以下第 1–9 节保留修复前历史：PRE_FIX_RESULT = FAIL；PRE_FIX_FAILURE = BATCH_LINEAGE_USES_INPUT_POSITION；当时首个 authoritative failure layer = GLOBAL_RECONCILIATION。** 第 1–9 节中的“本轮”“当前”和“未修复”均指该历史验收，不是第 10 节的修复后结论。

两次真实生产 batch 均返回 `COMPLETE`，但不满足 Gate 的持久身份和跨档关联不变量。保持同一 Project、Scan、source、Tower topology、science configuration 和 risk policy，仅反转两个 Span 的提交顺序：3,766,354 个最终自动成员的 original_index、XYZ、tree_id 分配完全相同，17/17 棵树的 persistent_tree_id 却改变；产品的 Span→tree 关联从每档 10 棵变成每档均关联全部 17 棵。这不是显示编号变化，也不是输入不足导致的 BLOCKED。

**按用户要求停止在该层。没有修改算法、科学阈值或 CONTACT 规则，没有 commit/push，没有执行下游人工编辑、finalize 或正式 Excel 导出。** 本文中的 17 棵是 automatic global reconciliation 输出，不是 post-editor finalized tree set。不得据此宣称整体 Gate 或下游验收通过。

## 1. 基线与目标版本

首次实际执行指定四条命令：

```text
git status --short
  多个既有未跟踪目录/文件；无 tracked modification（完整状态已在任务工具输出记录）
git branch --show-current
  feature/fast-research-pipeline
git rev-parse HEAD
  e81505da62c91f0e1fb1c9e871fea42c80d5d23c
git rev-list --left-right --count origin/feature/fast-research-pipeline...HEAD
  0  0
```

最初的 `0/0` 使用的是过期本地 remote-tracking ref。`git ls-remote origin refs/heads/feature/fast-research-pipeline` 返回指定 `610c0f3178aa2632d2f7638216638b21b28436aa`。执行 fetch 后为 `1/0`，安全 `git merge --ff-only` 到目标 HEAD；该唯一新提交仅新增正式 Gate 文档，未改变算法。无 reset、clean、prune、stash、提交或 push；所有既有未跟踪内容保留。

实际运行基线：`feature/fast-research-pipeline` / `610c0f3178aa2632d2f7638216638b21b28436aa` / `0 0`。已完整阅读 `docs/SURVEY_V01_REAL_MULTISPAN_E2E_GATE.md`。

状态留档：`acceptance_evidence/survey_v01_gate_b_20260830/git_status_before_report.txt` 与 `git_status_final.txt`。新增内容只有本报告及独立验收脚本/证据/项目；正式源码无修改。

## 2. Runtime 与 REAL 路径审查

运行 Python：`E:\TreeLineWarning-fast\.venv311\Scripts\python.exe`，CPython 3.11.9；NumPy 2.4.6、SciPy 1.17.1、laspy 2.7.0、PDAL 3.5.4。

两个现有虚拟环境均未直接提供 Python PDAL binding。使用本机已存在、ABI 匹配的 `dist/TreeLineWarning-rc-20260823f/TreeLineWarning/_internal/pdal` 及 `pdal_runtime` 原生依赖，通过 PDAL/PROJ 健康检查。仅借用原生依赖；DesktopWorkflow、BatchAnalysisScheduler 及所有 Python science 代码仍来自指定 HEAD，不运行旧打包产品逻辑。健康检查自身的小样本只用于验证依赖可用性，**没有进入验收项目、source、Ground 或其他 science 输入**。

审查并实际调用：

```text
DesktopWorkflow.start_batch(span_ids, mode="REAL")
 -> DesktopWorkflow._run_scheduler
 -> BatchAnalysisScheduler（不传入替代 runner/preparer）
 -> prepare_source_shared / ensure_ground_preparation / ensure_role_tile_indexes
 -> run_canonical_span_science / run_tracked_canonical_span_science
 -> run_span_wire_pipeline（Wire V2）
 -> HIGH_RECALL vegetation + exact confirmed-conductor source exclusion
 -> run_span_tree_pipeline（tree_backend=crown_stem_hybrid）
 -> per-Span results
 -> BatchAnalysisScheduler._global_reconcile
```

`desktop/workflow.py:845` 的 `_fixture_runner` / `_fixture_preparer` 注入仅发生于 `mode == "FIXTURE"`；本轮两次均为 REAL，未注入任何测试 runner、preparer 或 synthetic outcome。`legacy_reference` 是当前冻结 hybrid 实现自己的候选中间产物目录，并非本次切换到旧树算法；两档正式 manifest 均记录 `crown_stem_hybrid`。

证据：`run_gate.py`、`runtime.json`、两个 `run_*.log`、两个 `*_batch_result.json`（首次为 `batch_result.json`）、项目 `logs/application.log` 及 `stage-a-project.json` 的 algorithm_versions。

## 3. 真实数据搜索与场景确认

只读搜索 `input_las`、`workspace`、`artifacts`、保留的 `acceptance_evidence/archived_workspace_evidence_20260827`、历史配置和原项目 `E:/TreeLineWarning/data/raw`。识别到如下原始数据；未选择派生 Ground LAS、展示采样或测试数据冒充真实源：

| 原始数据 | bytes | point count |
|---|---:|---:|
| `E:\TreeLineWarning-fast\input_las\2026-08-07_16没54-10_PointCloud_colorized.las` | 6,303,322,159 | 105,055,350 |
| `E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las` | 414,332,205 | 6,905,520 |
| `E:\TreeLineWarning\data\raw\scene_02_large\scan_large.las` | 2,404,477,459 | 40,074,605 |

选择此前已执行过单档 Ground/Wire/Tree 科学链的大型源：

- SOURCE_PATH：`E:\TreeLineWarning-fast\input_las\2026-08-07_16没54-10_PointCloud_colorized.las`
- SOURCE_FILE_SIZE：6,303,322,159 bytes
- SOURCE_SHA256：`6d7ae65ff91d22e62ba97321b975949adf9e361596b733d7c799574877aae596`（本轮生产导入重新计算；结束后独立再次计算一致）
- SOURCE_POINT_COUNT：105,055,350；LAS 1.4 / point format 7。
- CRS：`LOCAL_CS["Arbitrary CRS", ... UNIT["metre",1] ...]`；无可靠水平/垂直参考声明，未虚构 GIS。
- 通过 `DesktopWorkflow.confirm_unit("METRE")` 显式确认米。
- 历史依据：归档 `real_e2e_01` 单档科学产物，以及 `real_multispan_acceptance_01` 的人工杆塔坐标与原始源截图；只复用人工确认的真实坐标，不复用历史科学结论。

三根连续杆塔（source/local XYZ，米）如下：

| Tower | 本次稳定 Tower ID | XYZ (m) | 人工确认 height (m) |
|---|---|---|---:|
| T001 | `tower-cf07856cbc23494a90097b2c725ccfc5` | 6.7566939, 40.8786994, 0.0969000 | 8.401995 |
| T002 | `tower-6b702832f2964f7c880e1dbb938b6848` | -25.4004061, 20.3610994, -0.0027000 | 8.272960 |
| T003 | `tower-e4d433aae29e4ecabd03be472c60a40d` | -56.4427061, 0.2432994, 0.0873000 | 8.933950 |

独立完整扫描源 LAS 再核对三根塔：0.6m 半径塔柱分别包含 18,909 / 60,719 / 63,486 个原始点，Z 证据从基部连续覆盖到约 9.05 / 9.09 / 9.70m。两档 ±8m 检查走廊分别有 2,071,619 / 3,553,651 个原始点。两档均有树和导线返回点，公共塔附近存在跨档树冠观察。

原始可视证据：`raw_two_span_evidence.html`、`raw_two_span_evidence.png`、`raw_scene_check.json`。密集背景仅用于显示抽样；稀疏中心走廊高处返回点完整保留以核对导线。这些展示数据从未用于科学 membership 或编辑。

输入质量保留：T001→T002 导线断续，中心候选 track 30 有 389 个 raw support、12 个支持切片，但纵向 coverage=0.2906519437，正式 Wire V2 以 `low_longitudinal_coverage` 拒绝；这本身不能据以声称算法缺陷，也未放宽约束。T002→T003 track 41 确认 611 个成员、35 个切片、coverage=0.9650874505。此场景确实有三杆两档真实输入，不能报告 `NO_SUITABLE_REAL_MULTISPAN_SCENE`。

## 4. 独立项目与两次真实运行

- PROJECT：`E:\TreeLineWarning-fast\workspace\survey_v01_gate_b_20260830`
- project_id：`project-97ee181d6e6049c789e42f91cd79b2f3`
- scan_id：`scan-c9b48975a8ca417480fa236875a6d413`
- S001 = T001→T002：`span-da0578ed1fc34c71837476070ab1a816`，约 38.145m。
- S002 = T002→T003：`span-8abc94f8a2464430b623b699be20fc6f`，约 36.991m。
- 两个 Span 同一次 batch 提交、同一 Scan/source identity、同一公共 Tower；导入/建拓扑后没有任何 LAS 副本。后续生成的是正式共享派生 Ground workset 和角色索引，无每档完整源 LAS 复制。

| 运行 | batch ID | 提交顺序 | mode | 产品状态 | 用途 |
|---|---|---|---|---|---|
| R1 | `analysis-bb9677112b164a0e93ca72347a39058c` | S001, S002 | REAL | COMPLETE / 两档 REVIEW_REQUIRED | 冷启动独立验收 |
| R2 | `analysis-32fe7e9b723149ba8db5cb77cd15e355` | S002, S001 | REAL | COMPLETE / 两档 REVIEW_REQUIRED | 同输入反序复现首个失败层 |

R1 进程正常结束后，新进程重新打开同一项目执行 R2。两次运行目录各自保留，未覆盖 R1 authoritative artifacts；R2 不是 fixture 或手工生成树结果。

| Telemetry | R1 COLD | R2 REUSE |
|---|---:|---:|
| `SOURCE_PREP_COUNT` | 1 | 0 |
| `SOURCE_FULL_PASS_COUNT` | 4 | 0 |
| `GROUND_PREP_COUNT` | 1 | 0 |
| `SPAN_LOCAL_READ_COUNT` | 2 | 2 |
| `CACHE_HIT_COUNT` | 0 | 1 |
| `CACHE_MISS_COUNT` | 1 | 0 |
| `TOTAL_TIME` | 421.8671902999922 | 109.52017189998878 |

Per-Span：R1 S001=16.916573s、S002=31.923474s；R2 S002=31.614132s、S001=16.166982s。

完整 STAGE_TIME 保存在两次 batch.json 和文末机器摘要。计时没有机器型号阈值；`role_query` / `raw_restoration` 等是重叠诊断口径，不能相加得到总时间。PEAK_MEMORY_BYTES 在产品 telemetry 中为 null，未编造峰值。

Ground：源级流式代表点 workset 579,975 点，PDAL_SMRF_v1 + PDAL_HAG_NN_v1，ground=114,667；源 original_index 映射及 hash 留档。`GROUND_PREP_COUNT=1` 是产品的 source Ground preparation 计数；冷启动角色准备另有正式 role-ground 产物，**不把该计数误称为整个进程只调用过一次 PDAL**。R2 复用有效 Ground/role cache，source full pass 与 Ground preparation 均为 0；仍重新执行每档 science 和 global reconciliation 来验证反序，不是把必须重算的工作跳过以制造缓存命中。

## 5. 首个权威失败：全局 lineage 使用输入列表位置

精确验收错误标签（本验收证据命名，非伪造产品抛出的异常）：

```text
RESULT = FAIL
FAILURE_LAYER = GLOBAL_RECONCILIATION
FAILURE_CODE = BATCH_LINEAGE_USES_INPUT_POSITION
PERSISTENT_TREE_ID_CHANGED_FOR_IDENTICAL_MEMBERSHIP = 17 / 17
SPAN_TREE_MAP_CHANGED = true
```

| 不变量 | 两次真实输出比较 |
|---|---|
| source、Scan、Project、topology、science/risk 配置 | 相同 |
| 自动 global tree count | 17 / 17 |
| exact original_index | 3,766,354 个完全相同 |
| exact XYZ、tree_id 分配 | 完全相同 |
| DUPLICATE_FINAL_ORIGINAL_INDEX | 0 / 0 |
| persistent_tree_id | **17/17 改变** |
| Span→tree map | **每档 10 棵变成每档全部 17 棵** |

具体例子：global_tree_id=1，106,330 个成员完全未变，membership SHA-256=`a9243069b144d7ed6037271002e244e643b79ed967dec629556b74bff08264a6`，primary span 始终是 S001。

| 字段 | R1 | R2 |
|---|---|---|
| merged_span_tree_keys | `[[0, 2]]` | `[[1, 2]]` |
| persistent_tree_id | `tree-910ed9034062b8914545e60a` | `tree-9bf9a9959d878d00b910c514` |

R2 中列表位置 1 实际代表 S001，消费方却把它解释为 S002 的内部 Span 索引。因此没有跨档观察的 S001 tree 也被加进 S002 的 tree 集合。没有偷偷删除重复点或去重修正；本轮 exact 点唯一性检查确实通过，**失败的是持久身份与业务 Span 关联，而非声称发生了重复 original_index**。

根因与涉及文件：

1. `core/stage_b/bridge.py:345` 保留调用者提交顺序。
2. `core/stage_b/batch.py:440` 按结果字典顺序构造 outcomes。
3. `algorithms/spans/reconcile.py:331` / `:506` 以 enumerate 的 `span_index` 构成并发布 merged_span_tree_keys；它表示列表位置，而非稳定 Span 身份。
4. `core/stage_b/batch.py:469` 将上述 keys 直接写入 persistent_tree_id hash 材料。
5. `core/stage_b/batch.py:512` 又将 `key[0]` 与真实内部 Span ID 比较，造成错误关联。

输入/输出证据：两个独立 batch 的 `tree_segmentation_original.npz`、`batch.json`、`batch-reconciliation.json`；验收侧 `authoritative_failure_reproduction.json` 为每棵树保存成员 SHA、两个 ID、lineage 和 Span map。`compare_real_batches.py` 对保存的两次完整真实运行做逐数组比较，并实际调用生产 `_tree_ids_by_span` 验证业务关联差异，没有构造 synthetic outcomes。

可直接复验证据（不重跑 science、不改最终 membership）：

```powershell
Set-Location E:\TreeLineWarning-fast
& .\.venv311\Scripts\python.exe .\acceptance_evidence\survey_v01_gate_b_20260830\compare_real_batches.py
```

第一轮只读反序审查的 `reconciliation_audit.log` 记录了 lineage 比较断言失败。单看列表型 keys 的变化还不充分；最终结论以随后 **第二次完整 REAL batch** 及产品 Span map 的实测变化为准。

## 6. 最小修复建议（仅建议，未实施）

修复边界限于 **Stage-B 多档全局整理/发布的顺序与 lineage 身份绑定**：在进入 multi-span reconciliation 前建立明确的稳定 Span 顺序/身份映射，并让 persistent ID 与 Span 关联消费同一套稳定 keys。优先评估在 Stage-B 边界规范化顺序并转换 positional keys 的最小修复；不要为此更换分割、修改精确成员、放宽 Wire 判据或重新设计 CONTACT。

配套验证只需覆盖：同一批真实 outcomes 的正/反序、非从 S001 开始的连续子集、跨档合并树与非跨档树，以及原有单档回归。必须证明 identical membership 对应相同 persistent_tree_id，且不存在无关 Span 关联。**不能只排序已生成的 keys**，因为其中的数字已经失去真实 Span 含义。

已验收单档影响：本次零源码修改，因此当前单档行为完全未被更改。建议修复应保留现有 `_reconcile_single_span` 及单档 science/clearance 路径；但若改变已有持久 ID 的构成，会有 registry/旧项目兼容风险，不能在未设计兼容规则和回归验证前声称零风险。

## 7. 停止边界、下游与已观察风险

根据首个失败层停止规则：Tree Editor DELETE/MERGE/UNDO、post-editor finalize、正式 final clearance/risk、Prediction_Input.xlsx、finalized restart/revision attribution **均未执行/未验收**。不把产品自动 `COMPLETE` 或预编辑 artifacts 冒充 `FINAL_GLOBAL_RECONCILED_POST_EDITOR`。

两次默认 batch 内自动执行的 clearance 还呈现顺序差异：R1 为 `NO_DISTANCE_READY_EDGE`（diagnostics 仍显示 distance_ready_segments=1），R2 为 `VALID`、minimum=4.5437216265m。这里只保留已经发生的观察；由于 global lineage 是更早的权威失败层，没有进行下游修复、final clearance 验收或借 R2 的 VALID 宣称风险通过。该后续问题需在上游修复后重新验收。

CONTACT detector 保持 `PENDING_REVIEWED_PHYSICAL_DETECTOR`。风险 threshold 始终为 0.5m；没有把零距离/tie-level 提升为 DANGER，没有对 CONTACT_UNCERTAIN/REVIEW_REQUIRED 作任何临时放宽。

非 finalized 项目确实被新进程重新打开以复现失败，但这**不替代**第九步要求的 finalized restart 验收。第二次 cache reuse 已有实测证据，也**不替代**最终编辑 revision 相关科学计算的缓存验收。

## 8. 数据完整性与交付

10 个历史关键 manifest/checkpoint（归档多档项目和单档真实科学链）在运行前后 path/size/mtime/SHA-256 全部一致；覆盖范围明确为这 10 个文件，不声称对全部历史大文件做过 hash。原始 LAS 最终 SHA-256 和 size 再次验证一致。历史 authoritative artifacts 未覆盖；无大型原始点云复制。

证据目录：`E:\TreeLineWarning-fast\acceptance_evidence\survey_v01_gate_b_20260830`。两次完整项目产物：`E:\TreeLineWarning-fast\workspace\survey_v01_gate_b_20260830\artifacts\project-runs\batches`。验收仅新增脚本、日志、原始可视证据、正式报告及独立派生项目数据。

## 9. 机器可读摘要

```text
GATE_ID = SURVEY_V01_GATE_B_REAL_MULTISPAN_E2E
RESULT = FAIL
FAILURE_LAYER = GLOBAL_RECONCILIATION
FAILURE_CODE = BATCH_LINEAGE_USES_INPUT_POSITION
BRANCH = feature/fast-research-pipeline
HEAD = 610c0f3178aa2632d2f7638216638b21b28436aa
RUNTIME = E:\TreeLineWarning-fast\.venv311\Scripts\python.exe | CPython 3.11.9 | PDAL 3.5.4
SOURCE = E:\TreeLineWarning-fast\input_las\2026-08-07_16没54-10_PointCloud_colorized.las
SOURCE_FILE_SIZE = 6303322159
SOURCE_SHA256 = 6d7ae65ff91d22e62ba97321b975949adf9e361596b733d7c799574877aae596
SOURCE_POINT_COUNT = 105055350
TOWER_COUNT = 3
SPAN_COUNT = 2
SELECTED_SPANS = ["span-da0578ed1fc34c71837476070ab1a816", "span-8abc94f8a2464430b623b699be20fc6f"]
ACCEPTANCE_MODE = REAL
SOURCE_CACHE_HINT = COLD
SOURCE_PREP_COUNT = 1
GROUND_PREP_COUNT = 1
SOURCE_FULL_PASS_COUNT = 4
SPAN_LOCAL_READ_COUNT = 2
CACHE_HIT_COUNT = 0
CACHE_MISS_COUNT = 1
TOTAL_TIME = 421.8671902999922
PER_SPAN_TIME = {"span-da0578ed1fc34c71837476070ab1a816": 16.916572500020266, "span-8abc94f8a2464430b623b699be20fc6f": 31.92347370000789}
STAGE_TIME = {"role_index_load": 14.035084500035737, "wire_role_query": 0.11197100000572391, "vegetation_role_query": 2.1842113999882713, "role_query": 2.296182399993995, "raw_restoration": 2.296182399993995, "wire": 0.8817679999920074, "tree": 13.110240900015924, "clearance": 6.938840800023172, "total": 48.827657500019995, "tree_reconciliation": 32.0501246000058, "clearance_publish": 0.00038270000368356705, "manifest_write": 0.014921599999070168, "reconciliation_json_write": 0.004849000019021332, "membership_serialization_npz_write": 0.15396619998500682, "viewer_artifact_build": 2.7782499000022653, "clearance_risk_json_write": 0.005473500001244247}
FIRST_BATCH_ID = analysis-bb9677112b164a0e93ca72347a39058c
SECOND_BATCH_ID = analysis-32fe7e9b723149ba8db5cb77cd15e355
SECOND_SELECTED_SPANS = ["span-8abc94f8a2464430b623b699be20fc6f", "span-da0578ed1fc34c71837476070ab1a816"]
SECOND_SOURCE_CACHE_HINT = REUSE
SECOND_SOURCE_PREP_COUNT = 0
SECOND_GROUND_PREP_COUNT = 0
SECOND_SOURCE_FULL_PASS_COUNT = 0
SECOND_SPAN_LOCAL_READ_COUNT = 2
SECOND_CACHE_HIT_COUNT = 1
SECOND_CACHE_MISS_COUNT = 0
SECOND_TOTAL_TIME = 109.52017189998878
SECOND_STAGE_TIME = {"role_index_load": 13.950925700046355, "wire_role_query": 0.11367679995601065, "vegetation_role_query": 2.144056300021475, "role_query": 2.257733099977486, "raw_restoration": 2.257733099977486, "wire": 0.8808874000096694, "tree": 12.55007320002187, "clearance": 6.957059799984563, "total": 47.770458800019696, "tree_reconciliation": 31.955952499993145, "clearance_publish": 11.270255800016457, "manifest_write": 0.013802400004351512, "reconciliation_json_write": 0.005519099999219179, "membership_serialization_npz_write": 0.14614639998762868, "viewer_artifact_build": 2.5725927999883424, "clearance_risk_json_write": 0.006789299979573116}
AUTO_GLOBAL_TREE_COUNT = 17
FINAL_TREE_COUNT = NOT_FINALIZED
DUPLICATE_FINAL_ORIGINAL_INDEX = 0
EXACT_AUTO_GLOBAL_MEMBER_COUNT = 3766354
PERSISTENT_TREE_ID_CHANGE_COUNT = 17
SPAN_TREE_MAP_CHANGED = true
EDITOR_OPERATIONS = NOT_EXECUTED_UPSTREAM_FAILURE
RESTART_STATE = FINALIZED_RESTART_NOT_TESTED_UPSTREAM_FAILURE
FINAL_CLEARANCE_STATUS = POST_EDITOR_NOT_EXECUTED_UPSTREAM_FAILURE
EXCEL_PATH = NOT_GENERATED_UPSTREAM_FAILURE
FINAL_GLOBAL_RECONCILED_POST_EDITOR = NOT_PRODUCED
CONTACT_DETECTOR = PENDING_REVIEWED_PHYSICAL_DETECTOR
HISTORICAL_KEY_ARTIFACT_CHANGES = 0 / 10 checked
SOURCE_INTEGRITY = UNCHANGED_SHA256_SIZE_MTIME
ALGORITHM_MODIFICATIONS = NONE
COMMIT = NONE
PUSH = NONE
BLOCKERS = GLOBAL_RECONCILIATION_ORDER_DEPENDENT_PERSISTENT_ID_AND_SPAN_ASSOCIATION
```

## 10. 修复后 REAL 续验（2026-08-30）

### 10.1 候选、角色及执行边界

监督/独立审查：GPT-5.6 Sol High；REAL 执行及证据提取：GPT-5.6 Luna Max。

```text
PRE_FIX_RESULT = FAIL
PRE_FIX_FAILURE = BATCH_LINEAGE_USES_INPUT_POSITION
POST_FIX_COMMIT = c32cc489344d2bfe16005ba6951dbf221d522e4a
POST_FIX_REAL_RERUN = COMPLETED_TWO_REAL_BATCHES
POST_FIX_RESULT = FAIL
FAILURE_LAYER = CLEARANCE_PUBLICATION_OR_ORDER_DEPENDENCE
BRANCH = feature/fast-research-pipeline
TRACKED_WORKTREE = CLEAN
REMOTE_HEAD = 610c0f3178aa2632d2f7638216638b21b28436aa
REMOTE_PUSH = NO
```

执行前按要求核对 branch、HEAD、完整 `git status --short`、`git show --stat --oneline HEAD`。相对旧基线仅有已批准的 lineage 发布修复及其两个测试文件；唯一生产改动仍在 `algorithms/spans/reconcile.py`，`core/stage_b/batch.py` 未改。本次续验没有再改任何生产或测试源码，没有 reset/fetch/merge/commit/push。既有未跟踪内容保留。

复用原项目 `E:\TreeLineWarning-fast\workspace\survey_v01_gate_b_20260830`、原 source、原人工确认拓扑和有效缓存。没有重新创建项目、导入/复制 raw LAS、修改 Tower/Span、强制 COLD 或重做实现回归。

运行 Python 为 `.venv311\Scripts\python.exe` / 3.11.9，NumPy 2.4.6、SciPy 1.17.1、laspy 2.7.0、PDAL 3.5.4。与历史验收相同，仅借用 `dist/TreeLineWarning-rc-20260823f/TreeLineWarning/_internal` 中 ABI 匹配的 PDAL 原生依赖。PDAL/PROJ 最小健康检查通过，`PROJ_NETWORK=OFF`；健康小样本不进入验收 science。Sol 独立检查 `canonical_executor`、`reconcile`、`batch`、`workflow` 模块路径，全部来自当前 checkout，并非旧打包产品代码。

执行路径为 `DesktopWorkflow.start_batch(selected, mode="REAL")` → 默认 `BatchAnalysisScheduler` → 当前生产 science。没有注入 runner/preparer/outcome，没有 FIXTURE fallback。产品 `logs/application.log` 同时记录两次 `batch_start` 的 REAL 模式、精确 Span 顺序及匹配的 `batch_finish`。

### 10.2 两次新 REAL 运行与 telemetry

| 项目 | F3：修复后正序 | R4：修复后反序 |
|---|---|---|
| analysis_run_id | `analysis-1456fc40f0314e02818a4c80cea7a949` | `analysis-0ab93088034140d28b887a7723ae115d` |
| Span 顺序 | S001 → S002 | S002 → S001 |
| 产品日志时间（Asia/Shanghai） | 10:44:26 → 10:46:04 | 10:50:07 → 10:51:56 |
| REAL 执行进程 PID | 58948 | 68440 |
| acceptance_mode | REAL | REAL |
| source_cache_hint | REUSE | REUSE |
| 产品状态 | COMPLETE | COMPLETE |
| SOURCE_PREP_COUNT | 0 | 0 |
| GROUND_PREP_COUNT | 0 | 0 |
| SOURCE_FULL_PASS_COUNT | 0 | 0 |
| SPAN_LOCAL_READ_COUNT | 2 | 2 |
| CACHE_HIT_COUNT / CACHE_MISS_COUNT | 1 / 0 | 1 / 0 |
| TOTAL_TIME（秒，产品 telemetry） | 98.05657399998745 | 108.70430460001808 |
| S001 时间（秒） | 16.331094900000608 | 15.97153020001133 |
| S002 时间（秒） | 30.843054199998733 | 30.988396000000648 |
| tree_reconciliation（秒） | 31.690261499985354 | 31.492399999988265 |
| clearance_publish（秒） | 0.0003486999776214361 | 11.04195020001498 |
| 自动 global tree count | 17 | 17 |
| exact 自动成员数 | 3,766,354 | 3,766,354 |
| duplicate_final_original_index | 0 | 0 |

完整 `PER_SPAN_TIME`、`STAGE_TIME`、逐树 persistent ID、merged keys、成员 hash、owner、产品映射均保存在两份 `*_run_evidence.json`、`*_batch_result.json` 及正式批次产物中。上述 TOTAL_TIME 不包含独立 source/hash 验证和证据恢复所花的时间；文件哈希读取不冒充 production source preparation/full-pass 计数。

两次 source point count 均为 **105,055,350**。两次 guard 的 topology 摘要同为 `0e10aef77bd99f366cdb27a32f84ffcb7c3563ca8efadfe6e61e00dc6719b89c`，完整 FastBaselineConfig 摘要同为 `b8a490935f2d7dbb162cb9eef7e43570bf88016ac9c6d535d22c1b5c307601bd`，risk 摘要同为 `bed8816b885e9ec2fdc55772e48f405415a99e40b49899b6f0f3c8c1a376b446`。没有在两次运行之间改变输入或科学配置。

### 10.3 REAL lineage 修复通过

Luna 比较输出后，Sol 另用独立脚本直接读取正式 `tree_segmentation_original.npz`、`batch-reconciliation.json`、每档 `tree/tree_members.npz` 和 `wire/w1_arrays.npz`，没有重跑或替换 science。结论：

- 两次 final original_index、XYZ、tree_id 数组逐项完全相等；也与旧 R1 的三个数组完全相等。
- 17/17 棵树的 exact membership、persistent_tree_id、identity_state、merged_span_tree_keys、primary Span/primary tree 均相等。
- 每档原始树成员数组和每档全部 W1 数组在两次新运行之间完全相等。
- 公开 lineage 不仅彼此相等，还与每档正式原始成员的 original_index → global tree 归属核对一致。
- 直接调用产品 `_tree_ids_by_span()`，其结果与原始成员独立推导的 Span 关联一致，无无关独占树跨档关联。

两次实际产品映射相同：

| 产品 Span | global tree IDs |
|---|---|
| S001 / `span-da0578ed1fc34c71837476070ab1a816` | `[0,1,2,3,4,5,6,7,8,12]` |
| S002 / `span-8abc94f8a2464430b623b699be20fc6f` | `[6,8,9,10,11,12,13,14,15,16]` |

三棵跨档观察树为 6、8、12；其余每侧 7 棵仅关联自身 Span。每棵树的稳定主归属不因顺序改变。

三数组原始内存字节 SHA-256（两次新运行与旧 R1 均相同）：

```text
original_index = 45768915315d44ba6f994746e55b03f994a8051a971b8fc775deab9efababad1
xyz_local_m = c965755d473656cdbec1492d3e602dbc421f8425ad7bab635a7fe30b3b6ce495
tree_id = 54fc24a63a5e5f003464e0e9ba9b26f6f6f2ac4d5a1d41de597d7eb00e6bd055
```

因此：`REAL_FORWARD_REVERSE_MEMBERSHIP`、`REAL_FORWARD_REVERSE_PERSISTENT_ID`、`REAL_FORWARD_REVERSE_MERGED_KEYS`、`REAL_FORWARD_REVERSE_SPAN_MAP` 全部 **PASS**。这只证明上游 lineage 修复，不把它扩展为整体 Gate PASS。

### 10.4 首个新权威失败：净空发布的顺序依赖

| 净空检查 | F3 正序 | R4 反序 |
|---|---|---|
| automatic clearance status | NO_DISTANCE_READY_EDGE | VALID |
| distance_ready_segments | 1 | 1 |
| 根据保存 W1 调用原有构边函数得到的边数 | 0 | 72 |
| global minimum clearance（m） | null | 4.543721626531224 |
| 有完整最近树 original_index/XYZ、wire XYZ 证据的树 | 0 / 17 | 17 / 17 |
| per-tree danger_point_count 合计 | 0（无有效距离计算的默认值） | 0（已计算） |
| global danger_point_count diagnostic | 缺失 | 0 |
| danger-point JSON / NPZ | 均存在，0 点，状态 NO_DISTANCE_READY_EDGE | 均存在，0 点，状态 VALID |

不能把正序“没有距离证据时的默认 0 点”解释为已评估安全；Luna 比较中的 `danger_point_count_exact=false` 对应的是 **缺失诊断字段与已计算的 0**，不是声称两份 per-tree 合计数不同。两个 danger sidecar 的产品 authority 标签也不替代 post-editor finalize 验收。

R4 的最小净空记录：global_tree_id=12，persistent_tree_id=`tree-835e4a62bad8a76799cfff91`，树成员 407,202；nearest_tree_original_index=79,179,041，nearest_wire_id=0。

```text
nearest_tree_xyz_local_m = [-31.484405517578125, 21.73150634765625, 7.708400249481201]
nearest_wire_xyz_local_m = [-28.98086835638627, 18.037923280027545, 6.850990181622474]
minimum_clearance_m = 4.543721626531224
```

这些是自动阶段观测，不是 post-editor final clearance 或正式风险验收。

### 10.5 最小可复现范围（只诊断，没有修复）

**最小疑似修复范围：`algorithms/spans/clearance.py::_concatenate_span_wire_edges` 的跨档 offset 拼接/发布边界。** 不是 Ground、Wire V2 识别、Vegetation/FRV、CROWN_STEM_HYBRID、树成员或 CONTACT 科学。

同一组保存的 REAL W1 输入中，S001 没有 ready wire/segment，`sampled_curve_offsets=[0]`；S002 有一个 ready segment、73 个采样点，`sampled_curve_offsets=[0,73]`。Sol 的只读诊断把保存的原始 W1 数组交给现有拼接及构边函数，未构造替代 science outcomes，也未写回项目：

```text
S001 → S002: sampled_curve_offsets = [0,0,73], wire_segment_offsets = [0,0,1]
             segment_distance_ready = [true], sampled_curve_xyz_count = 73
             build_wire_edge_bundle.edge_count = 0

S002 → S001: sampled_curve_offsets = [0,73,73], wire_segment_offsets = [0,1,1]
             segment_distance_ready = [true], sampled_curve_xyz_count = 73
             build_wire_edge_bundle.edge_count = 72
```

`clearance.py:211–212` 对每档的完整 offsets（包含边界项）直接 extend。前置空档的 `[0]` 留下额外边界，使 `tree_wire_clearance.py:202` 按 ready segment 位置 0 取到样本区间 `[0,0)`；反序则取到 `[0,73)`。这与正式输出的 0/72 条边和 null/有效净空一致，足以将本次新失败定位到该聚合边界。`core/stage_b/batch.py` 按 results 顺序传递 wire_results，因此顺序差异到达该边界。

这不是本轮实施方案；没有改 offsets、排序生产输入、替换距离计算或重跑修补后的结果。任何后续修复须作为独立任务，保留精确距离科学和当前风险/CONTACT 规则。

### 10.6 取证异常、完整性与停止点

两项验收脚本问题已透明保留：

1. 启动 UI 投影可能暂时带有上一个持久化 batch ID。脚本在真正执行前经 Sol 审查，改用新分配的 workflow `last_batch_id`，并与正式完成结果、项目目录及产品日志核对。`*_batch_start_projection.json` 中的旧 projected ID 不被当作新运行身份。
2. F3 science 已返回 COMPLETE 后，取证脚本因要求 source-shared 缓存 **mtime** 也绝对不变而报错。唯一变化是 `preparation.json` 的 mtime；其 4,007 bytes、SHA-256 `d131b1efae87d5a23f2a01761e170f44deec261877bd0c7a289812f34017c4fb` 完全不变，没有新增缓存文件。脚本改为核对 cache path/size/content hash，保留旧批次的严格指纹检查。`forward_driver_failure.json` 原样保留。F3 仅从已完成的正式 manifest/NPZ 恢复取证（提取 PID=62196），**没有再次运行正序 science**。

这两个取证问题均不是新的产品 scientific failure，也未用它们绕过真正的净空失败。

Sol 在执行前、F3 后、R4 后独立计算两次旧批次目录及旧验收目录中 164 个文件的 path/size/mtime/SHA-256（排除 `__pycache__`）；整体指纹三次完全相同：`D746E18A622318632EFD6C495FD4448ED811B0B76EF3BE38269FDF7105A3BFE1`。原 COLD/REUSE 证据和两个失败批次均未覆盖。

原始 LAS 的 source guards 在前后完全相同；Sol 再次独立计算最终 SHA-256，仍为 `6d7ae65ff91d22e62ba97321b975949adf9e361596b733d7c799574877aae596`，大小 6,303,322,159 bytes。没有 raw mutation、原始 LAS 复制或 frozen science 修改。

**停止在用户 Step 6。** 未选择 downstream authoritative post-fix run，未执行 Tree Editor DELETE/MERGE/UNDO、finalize、post-editor clearance/risk、Excel 或 finalized restart。两个新批次的自动 `tree_edits.json` 均为 `operations=[]`；自动存在的 `tree_segmentation_final.npz` 不被宣称为已人工编辑并 finalized 的产物。候选提交不推送，远端仍为旧基线。

### 10.7 新证据与最终摘要

新证据目录：`E:\TreeLineWarning-fast\acceptance_evidence\survey_v01_gate_b_postfix_20260830`。

- 实际执行/提取脚本：`real_postfix_driver.py`；执行模式依次为 `preflight`、`forward`、`recover-forward analysis-1456fc40f0314e02818a4c80cea7a949`、`reverse`、`compare`，forward/reverse 是不同进程。`real_postfix_run.py` 为准备阶段备选草稿，**未用于本次两次正式运行**。
- `preflight.json`、`prior_analysis_run_history.json`、两份 `*_start_guard.json`、`*_run_evidence.json`、`*_source_after.json`、`*_historical_after.json`：source、runtime、拓扑、历史、逐树和 telemetry 证据。
- `forward_batch_result.json`、`reverse_batch_result.json`、`postfix_comparison.json`：Luna 提取及比较结果。
- `sol_verify_postfix.py`、`sol_independent_review.json`：Sol 独立逐数组、逐树、原始成员关联和 W1 聚合诊断。
- 原项目 `logs/application.log`，以及两个新 `artifacts/project-runs/batches/<run_id>` 目录：正式生产结果。第 1–9 节原有证据仍留在原目录。

上述命令是执行记录，不应原样再次执行来覆盖同名证据。若只需独立复核，可调用 `sol_verify_postfix.py <forward_run_id> <reverse_run_id>` 输出到 stdout；不触发 science，也不写正式产物。

```text
SURVEY_V01_GATE_B_REAL_MULTISPAN_E2E_POST_FIX
LOCAL_FIX_COMMIT = c32cc489344d2bfe16005ba6951dbf221d522e4a
REMOTE_PUSH = NO
FORWARD_RUN_ID = analysis-1456fc40f0314e02818a4c80cea7a949
REVERSE_RUN_ID = analysis-0ab93088034140d28b887a7723ae115d
AUTHORITATIVE_POST_FIX_RUN_ID = NOT_SELECTED_STEP_6_FAILED
ACCEPTANCE_MODE = REAL
SOURCE_CACHE_HINT = REUSE / REUSE
SOURCE_POINT_COUNT = 105055350
REAL_FORWARD_REVERSE_MEMBERSHIP = PASS
REAL_FORWARD_REVERSE_PERSISTENT_ID = PASS
REAL_FORWARD_REVERSE_MERGED_KEYS = PASS
REAL_FORWARD_REVERSE_SPAN_MAP = PASS
REAL_CLEARANCE_ORDER_INDEPENDENCE = FAIL
AUTO_GLOBAL_TREE_COUNT = 17 / 17
EXACT_AUTO_GLOBAL_MEMBER_COUNT = 3766354 / 3766354
DUPLICATE_FINAL_ORIGINAL_INDEX = 0 / 0
REAL_TREE_EDITOR = NOT_EXECUTED_UPSTREAM_FAILURE
UNDO = NOT_EXECUTED_UPSTREAM_FAILURE
FINALIZE = NOT_EXECUTED_UPSTREAM_FAILURE
FINALIZED_TREE_COUNT = NOT_FINALIZED
FINAL_POST_EDITOR_CLEARANCE = NOT_EXECUTED_UPSTREAM_FAILURE
CONTACT_DETECTOR = PENDING_REVIEWED_PHYSICAL_DETECTOR
EXCEL = NOT_EXECUTED_UPSTREAM_FAILURE
EXCEL_PATH = NOT_GENERATED
FINALIZED_RESTART = NOT_EXECUTED_UPSTREAM_FAILURE
NO_FIXTURE_FALLBACK = PASS
FROZEN_SCIENCE_CHANGED = NO
RAW_DATA_MUTATED = NO
REPORT_PATH = E:\TreeLineWarning-fast\docs\SURVEY_V01_REAL_MULTISPAN_E2E_REPORT.md
REMOTE_HEAD = 610c0f3178aa2632d2f7638216638b21b28436aa
REMAINING_BLOCKERS = CLEARANCE_PUBLICATION_OR_ORDER_DEPENDENCE; DOWNSTREAM_GATE_NOT_EXECUTED
FINAL_VERDICT = FAIL
```

## 11. Final post-fix REAL acceptance — 27f42e19

本节为 2026-08-30 最终续验。Sol 独立审查并作 Gate 判定；Luna Max 执行 REAL 和下游机械验收。**整体 FAIL，首个新 authoritative failure 为 Step 11 的 `EXCEL_EXPORT_MISSING_TOWER_IDENTITIES`。** 不将文件成功生成误判为 Excel 字段验收通过；按要求停止，未进入 Step 12，未提交或推送。没有在验收期间修改生产代码。

历史链保持完整：

| 阶段 | 提交/结果 |
|---|---|
| PRE_FIX_1 | FAIL = `BATCH_LINEAGE_USES_INPUT_POSITION`，第 1–9 节 |
| FIX_1 | `c32cc489344d2bfe16005ba6951dbf221d522e4a` |
| PRE_FIX_2 | FAIL = `CLEARANCE_PUBLICATION_OR_ORDER_DEPENDENCE`，第 10 节 |
| FIX_2 / 本次运行 HEAD | `27f42e19b5206a5912613f199165347ba59bdd4c` |
| 本次最终续验 | lineage、clearance order、editor/undo/finalize/post-editor clearance PASS；Excel FAIL |

### 11.1 基线、来源与执行边界

`git branch --show-current` = `feature/fast-research-pipeline`；`git rev-parse HEAD` = `27f42e19b5206a5912613f199165347ba59bdd4c`；`git status --short` 仍有既有未跟踪文件，tracked worktree/index 干净。`git log -3 --oneline` 依次为 27f42e1、c32cc48、610c0f3。运行前和停止后的 live remote HEAD 均为 `610c0f3178aa2632d2f7638216638b21b28436aa`，没有 commit/push。

- 原始 LAS：`E:\TreeLineWarning-fast\input_las\2026-08-07_16没54-10_PointCloud_colorized.las`。
- 点数 `105055350`；大小 `6303322159` bytes；SHA-256 `6d7ae65ff91d22e62ba97321b975949adf9e361596b733d7c799574877aae596`；最终独立重算相同，mtime UTC 仍为 `2026-08-10T01:20:06Z`。未复制或修改原始 LAS。
- 同一 Project `project-97ee181d6e6049c789e42f91cd79b2f3`，同一 Scan `scan-c9b48975a8ca417480fa236875a6d413`，仍为三 Towers、两 Spans。
- 使用仓库 `.venv311` CPython 3.11.9，numpy 2.4.6、scipy 1.17.1、laspy 2.7.0；native PDAL 3.5.4 来自 `dist/TreeLineWarning-rc-20260823f/TreeLineWarning/_internal`。PDAL/PROJ minimal-operation 通过，`PROJ_NETWORK=OFF`。产品/科学 Python 模块路径来自当前 checkout。
- 冻结 FastBaselineConfig digest `b8a490935f2d7dbb162cb9eef7e43570bf88016ac9c6d535d22c1b5c307601bd`；risk digest `bed8816b885e9ec2fdc55772e48f405415a99e40b49899b6f0f3c8c1a376b446`；warning threshold `0.5 m`。没有修改拓扑、杆塔坐标或科学/风险配置。
- 正反两次均走默认 `DesktopWorkflow.start_batch(..., mode="REAL")`，没有 fixture、synthetic source、runner/preparer injection 或 fallback。REUSE 两次均为 cache hit 1、cache miss 0、source prep 0、ground prep 0、source full pass 0、Span-local read 2；没有强制 cold run。

实际执行脚本为新证据目录的 `final_real_driver.py`（精简适配器，显式重绑定旧 driver 的输出目录、HEAD、四个历史 batch ID）。预检曾使用较长草稿，后被精简适配器替换；`preflight.json` 保留，但草稿源码未完整留存，`final_real_driver_draft_20260830.py` 仅是说明标记，不是该草稿源代码。所有正式 REAL 执行使用留存的精简适配器。此限制不涉及四个历史失败 batch 或旧失败证据的覆盖。

### 11.2 两次新的 REAL automatic 结果

| 项目 | Forward S001 → S002 | Reverse S002 → S001 |
|---|---|---|
| analysis_run_id | `analysis-056f8909ad564949b2c9768180832e7a` | `analysis-53f4c1ac401249d78c7e8a25a438d9c9` |
| 执行 PID | 4960 | 12128 |
| 产品 batch 开始 UTC | 03:59:42.008174Z | 04:04:10.574910Z |
| 产品 batch 完成 UTC | 04:01:31.137120Z | 04:05:58.117503Z |
| 状态 / Span 状态 | COMPLETE / 两档 REVIEW_REQUIRED | COMPLETE / 两档 REVIEW_REQUIRED |
| automatic tree count | 17 | 17 |
| exact member count | 3766354 | 3766354 |
| duplicate_final_original_index | 0 | 0 |
| clearance status | VALID | VALID |
| distance_ready_segments / total_wires | 1 / 1 | 1 / 1 |
| edge_count | 72 | 72 |
| minimum_clearance_m | 4.543721626531224 | 4.543721626531224 |
| evaluated_edge_total | 60261664 | 60261664 |
| danger_point_count | 0 | 0 |

Sol 直接读取各正式 batch 的 original NPZ、reconciliation、逐 Span 原始成员和 W1、clearance/danger JSON/NPZ，独立验证：

- 全部 `original_index`、XYZ、`tree_id` 数组精确相同；逐树成员、persistent IDs、merged keys、归属及 Span map 精确相同，并由原始 Span 成员反查验证 lineage/map，而非仅比较两份派生 JSON。
- S001 → `[0,1,2,3,4,5,6,7,8,12]`；S002 → `[6,8,9,10,11,12,13,14,15,16]`。两份 manifest 均记录相同 canonical S001/S002 publication order。
- 所有树的 nearest original_index、nearest wire ID、成员数量及 evaluated-edge count 精确匹配；nearest tree/wire XYZ、距离和全部语义诊断使用真实冻结 `clearance_tie_tolerance_m=1e-9`，`rtol=0`。只排除 `runtime_seconds`、`index_build_seconds`、`distance_evaluation_seconds` 三项耗时；danger 元数据仅剔除经各自 run ID 校验的运行归属字段，数组精确比较。
- 每个最近树点均能回查到该树的精确原始成员，其 XYZ 与距离关系也经独立检查。最小值对应 global tree 12、persistent `tree-835e4a62bad8a76799cfff91`、nearest original index `79179041`、wire ID 0。

automatic 三数组 SHA-256 与之前有效 membership 一致：original index `45768915315d44ba6f994746e55b03f994a8051a971b8fc775deab9efababad1`；XYZ `c965755d473656cdbec1492d3e602dbc421f8425ad7bab635a7fe30b3b6ce495`；tree ID `54fc24a63a5e5f003464e0e9ba9b26f6f6f2ac4d5a1d41de597d7eb00e6bd055`。

结论：`REAL_LINEAGE_ORDER_INDEPENDENCE=PASS`，`REAL_CLEARANCE_ORDER_INDEPENDENCE=PASS`。证据 `sol_automatic_review.json`。独立检查器最初把 sidecar 的外层 schema 字段也要求存在于嵌入 manifest 中，已在只读取证脚本中改为先验证该 schema、再比较相同语义载荷；没有更改产物或重跑 science。

### 11.3 单一权威 run 的真实编辑、UNDO 和 finalize

唯一 downstream authority 为 Reverse `analysis-53f4c1ac401249d78c7e8a25a438d9c9`；没有混用 Forward 的树和 Reverse 的净空。Forward 仅作独立对照并封存。

选择真实 gid 0，persistent ID `tree-eba45fa479a3ebd7c49cf42e`，S001 单档成员键 `(0,0)`，精确成员 16037。执行 DELETE → UNDO → 最终 DELETE；这是用户授权的工作流验收操作，不是判定该树在物理场景中应删除。

通过现有 workflow editor API 执行，未使用 viewer samples 作为 authority。第一次 DELETE 精确去除目标成员，17 → 16；UNDO 恢复全部 3766354 个 original_index/tree_id/XYZ，三数组逐值与 SHA-256 均与编辑前相同，17 棵恢复，journal 清空、revision 0。最终 DELETE 留下唯一 `{"type":"DELETE","tree_ids":[0]}`，revision/operation count = 1/1。其余树的精确成员与 persistent IDs 不变。

随后在独立进程 PID 66276 中，仅调用同步 `wf.finalize()`，不使用会自动导出的 `start_finalize()`。正式 `batch.json` 的两 Span 状态均为 FINALIZED，batch 状态 COMPLETE；final tree count 16、exact final members 3750317、duplicate 0，publication 为 `FINAL_GLOBAL_RECONCILED_POST_EDITOR`，editor schema 为 `treeline-stage-b-batch-editor-v1`。

finalized 三数组 SHA-256：

- original index：`202ef58cea6c4bb29a964f644e9e50a0c533b8da7d78a9f7445728d7d582da21`。
- tree ID：`1ad3e7ee51aec8d0ef737f03edaae7ee174b70d293da8aac582db222c22f364d`。
- XYZ：`d55d9bc5e6a6495e6b0ee82a75ee7d67bafffa0cfb08ee821339af3a88cf1556`。
- journal SHA-256：`e3558dfc162fba9704b6bf614a5a68060181974e9e99a6e41e5745eee5bf972f`。

finalize 前审查修正了验收脚本的返回值解包数量及 source_before 的取证时机；没有修改生产实现，也没有因该草稿问题执行失败的 finalize 或重复 finalize。

### 11.4 Post-editor clearance / risk

Sol 独立检查实际 finalized NPZ 等于 immutable original 去掉 gid 0 的精确子集，并逐树确认剩余身份、索引及最近点证据。正式 final clearance 为 VALID、16 records、72 edges、1 ready segment、exact points **3750317**，evaluated-edge total **60005072**，danger points 0；min 仍为 `4.543721626531224 m`。

这是在减少后的精确成员上重新计算，不是以“最小值没变”推定成功：exact-point count 已从 3766354 变为 3750317，evaluated-edge total 从 60261664 变为 60005072。所有剩余最近源点仍属于对应 final tree，nearest wire/tree geometry 和 untouched-tree clearance 与原科学结果相符。

实际 16 棵均为正净空且大于 0.5m；调用冻结产品 contact/risk policy 核验均为 SAFE / CONTACT_NOT_CONFIRMED，实际导出行也一致。部分 tree quality 仍为 REVIEW_REQUIRED，不冒充物理 CONTACT 审查完成。本场景没有 zero/tie-level 观测；没有将零距离推定 CONTACT，没有补造 CONTACT evidence。物理 detector 仍为 `PENDING_REVIEWED_PHYSICAL_DETECTOR`。`FINAL_POST_EDITOR_CLEARANCE=PASS`，证据 `sol_finalized_review.json`。

### 11.5 Step 11 authoritative failure：正式 Excel 缺少已知杆塔身份

PID 61784 的独立进程恢复同一 finalized run，确认 journal revision 1、3750317 个成员后，**只调用默认 `wf.export_prediction()`**，没有 identity override、再次 finalize、自制工作簿或补写字段。

实际文件：`E:\TreeLineWarning-fast\workspace\survey_v01_gate_b_20260830\exports\Prediction_Input.xlsx`；12273 bytes；SHA-256 `4586f52e97a54d660bbd7f614a18a1d44fb837bed4e5e47df1b2621a806417f1`。导出前文件不存在；生成前后 journal hash 相同。`export_evidence.json` 的 `verdict=PASS` 仅表示该命令成功生成文件，**不是 Excel semantic Gate PASS**。

按 Spreadsheets 技能对实际文件只读导入、查看全部工作表，并直接读取原始 OOXML 交叉核验，未重导出、改格式或补数据：

- `Trees` 16 行与 finalized persistent tree inventory 精确一致；analysis IDs、science_run_id、Span 名称、净空、SAFE/contact 状态、0.5m 阈值相符。`Run_Metadata` 的 science_run_id 正确，science_scope 为 `FINAL_GLOBAL_RECONCILED_POST_EDITOR`。
- 所需列标题存在，但 **16/16 行 start_tower_id 为空，16/16 行 end_tower_id 为空**，直接 OOXML 也确认 `Trees!E2:F17` 共 32 个单元格没有值。这是已知产品身份丢失，不是可合法留空的不可用科学几何。
- 源项目拓扑明确存在：S001 的 T001→T002 为 `tower-cf07856cbc23494a90097b2c725ccfc5` → `tower-6b702832f2964f7c880e1dbb938b6848`；S002 的 T002→T003 为 `tower-6b702832f2964f7c880e1dbb938b6848` → `tower-e4d433aae29e4ecabd03be472c60a40d`。
- 同一只读取证确认 16 个 DBH 单元格为空且 `diameter_source=BREAST_HEIGHT_1P30M_UNAVAILABLE`，这属于正确的 unavailable/fail-closed 行为，与缺失杆塔身份不同。其他科学几何不因本次局部检查被宣称完成全部验收。
- 原生导出还存在长标题/身份文本列宽裁切；未修饰。首个决定性 failure 仍是必需的杆塔身份字段缺失，不以视觉问题代替语义判断。

有界代码检查与实际证据一致：`core/stage_b/export.py` 从 `extra.get("start_tower_id")` / `extra.get("end_tower_id")` 取值；默认 automatic display identity 只产生展示编号等字段，默认 workflow 导出没有补充这些 topology identities。本轮没有修复、注入 identity_by_tree 或人工补写。

`EXCEL=FAIL`；`FAILURE_LAYER=EXCEL_EXPORT_MISSING_TOWER_IDENTITIES`。证据 `excel_qa/inspection.json` 与独立标准库 OOXML 检查 `sol_excel_gate_failure.json`。**停止在 Step 11，未执行 Step 12 finalized restart**。为 finalize/export 恢复过新进程，不等于已验证“导出后新进程完整恢复及 Excel revision provenance”的 Step 12；不混淆两者。

### 11.6 历史保护、已知非 Gate 回归与 Git closure

Sol 在开始、Forward 后、Reverse 后以及停止后独立核对四个旧 batch 目录和两个旧证据目录，310 个文件的路径/大小/mtime/SHA-256（排除 `__pycache__`）aggregate 始终为 `35C13054C027F7D39201E87CBD43016DFD39CBEC2BD9DEA62300B32B74FBC2C6`。新 Forward 在下游前后也封存一致：63 files，aggregate `6926208B6753921841C8A1B9D6B09AE46BFB328CDF3346D4B35758A7AFD4CBEF`。缓存允许 preparation.json 的已知 mtime-only refresh，但 path/size/content 不变。原始 LAS 最终独立 hash/size/mtime 一致。

九个 V4 golden mismatch 仅记录为 `PREEXISTING_NON_GATE_REGRESSION`；它们已在之前 c32 baseline 独立重现。本次没有重跑、调查或修改 V4 算法、golden、FRV/V4 thresholds，不以其影响此次首个 REAL failure 的归因。

新证据目录：`E:\TreeLineWarning-fast\acceptance_evidence\survey_v01_gate_b_final_20260830`。保留 preflight、两份 run evidence/batch result/source/history guards、`sol_automatic_review.json`、editor undo/delete、finalize、`sol_finalized_review.json`、export 和 `sol_excel_gate_failure.json`。正式文件留在同一项目；失败 workbook 原样保存。报告新增本节，保留原第 1–10 节正文。未删除历史失败产物。

实际入口命令均使用新目录脚本，按步骤单次执行；不应照抄重跑以覆盖证据：

```text
.venv311\Scripts\python.exe -B acceptance_evidence/survey_v01_gate_b_final_20260830/final_real_driver.py forward --allow-real
.venv311\Scripts\python.exe -B acceptance_evidence/survey_v01_gate_b_final_20260830/final_real_driver.py reverse --allow-real
.venv311\Scripts\python.exe -B acceptance_evidence/survey_v01_gate_b_final_20260830/run_editor_phase.py undo-check analysis-53f4c1ac401249d78c7e8a25a438d9c9 --allow-edit
.venv311\Scripts\python.exe -B acceptance_evidence/survey_v01_gate_b_final_20260830/run_editor_phase.py delete-final analysis-53f4c1ac401249d78c7e8a25a438d9c9 --allow-edit
.venv311\Scripts\python.exe -B acceptance_evidence/survey_v01_gate_b_final_20260830/finalize_only.py analysis-53f4c1ac401249d78c7e8a25a438d9c9 --allow-finalize
.venv311\Scripts\python.exe -B acceptance_evidence/survey_v01_gate_b_final_20260830/export_only.py analysis-53f4c1ac401249d78c7e8a25a438d9c9 --allow-export
```

由于 Gate FAIL，不提交、不推送；保留两次批准的 fix commits，tracked worktree/index 仍干净，报告和本地验收证据保持未跟踪。live remote 未改变。

### 11.7 最终要求字段

```text
SURVEY_V01_GATE_B_REAL_MULTISPAN_E2E_FINAL
LOCAL_HEAD = 27f42e19b5206a5912613f199165347ba59bdd4c
REMOTE_HEAD_BEFORE = 610c0f3178aa2632d2f7638216638b21b28436aa
REMOTE_PUSH = NO
FORWARD_RUN_ID = analysis-056f8909ad564949b2c9768180832e7a
REVERSE_RUN_ID = analysis-53f4c1ac401249d78c7e8a25a438d9c9
AUTHORITATIVE_POST_FIX_RUN_ID = analysis-53f4c1ac401249d78c7e8a25a438d9c9
SOURCE_POINT_COUNT = 105055350
SOURCE_CACHE_HINT = REUSE / REUSE
REAL_LINEAGE_ORDER_INDEPENDENCE = PASS
REAL_CLEARANCE_ORDER_INDEPENDENCE = PASS
FORWARD_CLEARANCE_STATUS = VALID
REVERSE_CLEARANCE_STATUS = VALID
FORWARD_EDGE_COUNT = 72
REVERSE_EDGE_COUNT = 72
FORWARD_MINIMUM_CLEARANCE_M = 4.543721626531224
REVERSE_MINIMUM_CLEARANCE_M = 4.543721626531224
AUTO_GLOBAL_TREE_COUNT = 17 / 17
EXACT_AUTO_GLOBAL_MEMBER_COUNT = 3766354 / 3766354
DUPLICATE_FINAL_ORIGINAL_INDEX = 0 / 0; FINAL=0
REAL_TREE_EDITOR = PASS
UNDO = PASS
FINALIZE = PASS
FINALIZED_TREE_COUNT = 16
FINAL_POST_EDITOR_CLEARANCE = PASS
EXCEL = FAIL
EXCEL_PATH = E:\TreeLineWarning-fast\workspace\survey_v01_gate_b_20260830\exports\Prediction_Input.xlsx
FINALIZED_RESTART = NOT_EXECUTED_STEP_11_FAILED
NO_FIXTURE_FALLBACK = PASS
FROZEN_SCIENCE_CHANGED = NO
RAW_DATA_MUTATED = NO
CONTACT_DETECTOR = PENDING_REVIEWED_PHYSICAL_DETECTOR
PREEXISTING_V4_GOLDEN_MISMATCH = 9
REPORT_PATH = E:\TreeLineWarning-fast\docs\SURVEY_V01_REAL_MULTISPAN_E2E_REPORT.md
PUSH_COMMIT = NONE
REMOTE_HEAD_AFTER = 610c0f3178aa2632d2f7638216638b21b28436aa
REMAINING_BLOCKERS = EXCEL_EXPORT_MISSING_TOWER_IDENTITIES; FINALIZED_RESTART_NOT_EXECUTED; CONTACT_DETECTOR_PENDING_SEPARATE
FINAL_VERDICT = FAIL
```
