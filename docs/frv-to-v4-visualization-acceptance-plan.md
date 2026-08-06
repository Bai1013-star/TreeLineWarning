# FRV-to-V4 最小可视化与人工验收实施计划

文档状态：`PLAN-FROZEN`
阶段：`FRV-VIS-A1`
Implementation：`NOT STARTED`
Formal V4 segmentation：`NOT STARTED`
本文件只冻结实施范围、输入、验收和验证方式；本轮不实现代码、不修改测试、不暂存、不提交、不推送。

## 1. Baseline

- Repository：`E:\TreeLineWarning`
- Branch：`feature/algorithm`
- HEAD：`92277722b6094a03629fcc4b25e37db9f4ffc79a`
- Latest commit：`feat: add reviewed FRV-to-V4 handoff bundle`
- Remote divergence：`origin/feature/algorithm...HEAD = 0 0`
- Cached：empty
- 工作区：除明确排除且不读取、不处理的 `.claude/` 外无项目变更。

基线命令：

```powershell
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
git rev-list --left-right --count origin/feature/algorithm...HEAD
git status --short
git diff --cached --name-status
```

## 2. 目标、边界和完成定义

本阶段只制作一个独立、可重复运行的查看器，用于人工判断 FRV membership result 与 V4 input handoff 是否足以进入正式 V4 单木分割。它不执行 V4 segmentation，也不重新判定植被。

首版必须回答：

1. normal/admitted 点是否完整；
2. review、excluded、wire/ground 冲突是否可解释；
3. 树冠边缘、树干和导线附近枝叶是否被过度排除；
4. blocked/unrecovered 是否被完整记录且没有伪造成点；
5. 当前输入是否可进入 V4。

人工结论固定为：`PASS_READY_FOR_V4`、`PASS_WITH_REVIEW_NOTES`、`FRV_RULE_ADJUSTMENT_REQUIRED`、`DATA_OR_ARTIFACT_BLOCKED`。首版不冻结百分比阈值，先使用真实场景，再根据记录的误差模式决定是否增加量化门槛。

## 3. Existing implementation findings

### 3.1 可复用入口

- `algorithms/vegetation/frv_membership.py::load_frv_membership_result(root_path)` 会重新验证 `execution_result.json`、coverage manifest、payload schema、membership manifest、binding、decision evidence、FRV block digest 与 artifact digest；READY 结果的正式输入应从此入口开始。
- `algorithms/vegetation/frv_artifacts.py::decode_frv_block` 是 FRV/V4 block 的正式解码入口；不得自行按 NPZ 或私有二进制格式猜测字段。
- `algorithms/vegetation/tree_visualization.py` 已有 `deterministic_display_indices`、Plotly `write_html(include_plotlyjs=True, full_html=True)`、父目录创建和 bounded summary 模式，可复用其原则，不复用 V3 的业务字段和单层图层。
- `algorithms/io/source_provider.py::load_frv_source` 是正式 P-SOURCE replay 入口，但它要求完整的 `FrvSourceProviderReference`、`FrvConfiguration` 和 `FrvSourceProviderConfig`；当前 published roots 没有足够字段让查看器安全构造这三个 runtime 对象。

### 3.2 已冻结的字段与路由

- FRV 五态：`VERIFIED_VEGETATION_CANDIDATE`、`EXCLUDED_NON_CANDIDATE`、`REVIEW_REQUIRED_MEMBERSHIP`、`MANUAL_INCLUDE`、`MANUAL_EXCLUDE`。
- V4 四路：`NORMAL_SEGMENTATION`、`MEASURED_POINT_REVIEW`、`UNRECOVERED_REGION_REVIEW`、`EXCLUDED_AUDIT_ONLY`。
- READY handoff 的 normal block 字段含 `original_index`、`xyz_local_m`、`hag_m`、membership state、HAG state、HAG readiness、reason flags；review block 另含 raw/clamped HAG、quality flags、wire provider/relation、nearest wire、distance 和 primary route。
- FRV membership block 另含全部 evaluated point 的 `original_index`、membership state、reason/quality、HAG、scope references；evidence block 含 geometry/wire evidence。excluded 坐标必须由 source 按 original index 审计恢复，不能从 V4 normal/review 猜测。
- `V4RecoveryCoverageReport` 同时提供 source count、各类 point count、`unrecovered_region_count` 以及 routing/readiness 完成标志。`unrecovered_region_count` 永远不是 point count。

### 3.3 现实限制

当前 V4 没有公开的 handoff-directory loader；handoff 目录仅发布 block、`validated_view.json`、`coverage_report.json` 和 `handoff_manifest.json`。查看器计划在独立模块内严格读取并验证这些正式文件，必要时复用现有 contract/decoder，不修改 V4 runtime。

当前 `COVERAGE_BLOCKED` live result 只生成 coverage-only handoff，`output_reference` 为 null；coverage record 只有 canonical key、scope/coverage state、reason flags 和 `diagnostic_reference`，没有可绘制的 bbox/region vertices。首版不得把 source 全局范围冒充 blocked region。若没有可验证的 region geometry resolver，HTML 只能显示独立 blocked-region 审计表/标记状态，并将“缺少几何”置为 `DATA_OR_ARTIFACT_BLOCKED`；不得伪造点或范围。

### 3.4 F-001：SOURCE REPLAY BOOTSTRAP（冻结方案）

本阶段选择方案 C：查看器不调用 `load_frv_source`，不构造或猜测 `FrvSourceProviderReference`、`FrvConfiguration`、`FrvSourceProviderConfig`，也不以独立 LAS 读取取代正式 P-SOURCE 语义。原因是 published JSON 没有完整 runtime reference/configuration/provider-config 的可反序列化字段；强行本地重建会改变正式身份绑定。

READY 模式的唯一只读加载顺序为：

1. 从 `<frv-artifact-root>/execution_result.json` 读取 `source_identity`、`a2_readiness`、`source_proof_digest`、execution state 及其他正式可序列化字段；用 `FrvMembershipExecutionResult.from_dict` 解析正式字段，再调用现有 `load_frv_membership_result(Path(frv_artifact_root))` 完成 artifact、manifest、binding、decision evidence、block digest 和 artifact digest 的正式校验。`FrvMembershipExecutionResult.from_dict` 不恢复 runtime `output_reference`；查看器不得依赖持久化 `output_reference`。artifact root 的唯一来源是 CLI 显式传入的 `--frv-artifact-root`，V4 root 的唯一来源是 CLI 显式传入的 `--v4-input-root`；查看器不从 JSON 猜测、重建或替代 runtime output reference。
2. 从 `<frv-artifact-root>/coverage_manifest.json` 用 `FRVScopeCoverageManifest.from_dict` 解析并验证 `manifest_digest`；从 `membership_manifest.json`、`binding.json` 和 `payload_schema.json` 取得并交叉验证 membership artifact digest、configuration digest、payload schema 和 block descriptors。
3. 从 `<v4-input-root>/validated_view.json` 用 `V4ValidatedFrvMembershipView.from_dict` 解析，从 `<v4-input-root>/coverage_report.json` 用 `V4RecoveryCoverageReport.from_dict` 解析，并读取 `<v4-input-root>/handoff_manifest.json` 的 `source_content_identity_digest`、`frv_membership_artifact_digest`、`report_digest`、`normal_block_count`、`review_block_count`。V4 normal/review blocks 只通过 `decode_frv_block` 读取。
4. 正式 source identity 的唯一权威字段来自已验证的 FRV `execution_result.json` 的 `source_identity`；V4 view、coverage report、handoff manifest、membership manifest 和 binding 只允许与该 identity 相等，不得产生第二个 identity。`a2_readiness.configuration_digest` 与各 provider evidence 的 `provider_config_digest` 仅作为已发布 provenance 摘要校验，不用于构造 runtime config。

查看器必须精确比较：`source_content_identity_digest`、`source_identity_record_digest`、`source_file_sha256`、`source_point_count`、`source_bounds_min_xyz`、`source_bounds_max_xyz`、`coordinate_frame`、`units`、`origin_context`；并比较 membership artifact digest、coverage/report digest、handoff digest、normal/review block 数量和 original-index 域。source 文件的 `source_file_sha256` 由查看器对传入文件按字节计算；LAS header 的 point count/min/max 只用于与 identity 的 `source_point_count`/bounds 比较，绝不用于推导或替代 `origin_context`。

excluded/background 的唯一坐标读取策略是独立、只读 LAS display reader：按文件原始顺序逐 chunk 读取，要求每一行的 `original_index == chunk.start_index + row`，要求完整序列恰为 `0..source_point_count-1`，只保存 FRV excluded index 或 deterministic background index 的 XYZ；selected index 集合必须与 FRV payload 的 excluded 集合精确相等。随后只做一次 `local_xyz = float64(source_xyz) - float64(source_identity.origin_context)`。该 reader 不发布 source result、不计算新的 membership、不调用 builder，也不改变正式 digest/统计；任一 index、count、bounds、file SHA 或 identity 比较失败即不生成 HTML。该策略不新增 CLI 参数，也不接受 `--source-reference-json`、`--frv-configuration-json` 或 `--source-provider-config-json`。

## 4. Exact visualization inputs

### 4.1 READY 场景

正式输入为：

1. `--source <source.las>`：同一 source 文件，用于 excluded 坐标和可选背景；
2. `--frv-artifact-root <published-frv-root>`：已发布 READY FRV membership artifact root；
3. `--v4-input-root <published-v4-root>`：已发布 V4 handoff root；
4. 不新增 reference/config/provider-config JSON 参数；source identity 固定取已校验的 FRV `execution_result.json`，其余 published JSON 只做交叉验证。

查看器先调用 FRV loader 和上述独立 handoff 解析流程，核对 source content identity、source record digest、file digest、artifact digest、handoff digest、point count、bounds、frame、units、origin、block count、original-index 域和各 route 守恒；任何不一致立即拒绝输出。

### 4.2 blocked/unrecovered 场景

由于当前正式 API 不持久化 `COVERAGE_BLOCKED` 的完整 result，首版 CLI 不接受一个不存在的“blocked artifact root”并假装可绘制。若真实运行提供 `execution_result.json`/`coverage_manifest.json`，则以 contract `from_dict` 加载并只显示已验证的 coverage rows；没有 region geometry 时进入上述 fail-closed 状态。新增持久化 blocked artifact 不属于本阶段可视化范围，需另立 contract 变更。

### 4.3 背景 source

背景只用于漏检观察，可关闭。使用上述独立只读 LAS display reader 按 `original_index` 和 chunk 进行确定性降采样；它必须先通过 artifact/source identity、file SHA、count、bounds、frame、units 和 origin 校验。背景采样不得参与正式统计、路由或 digest，也不得把 LAS header min 当作 origin。当前 P-SOURCE provider 只接受 `.las`，CLI 应明确拒绝正式入口不支持的 `.laz`，而不是隐式绕过 provider。

## 5. Formal layer mapping

| 图层 | 正式来源与成员 | 点/区域语义 | 默认显示 |
|---|---|---|---|
| `normal_admitted` | V4 normal blocks；`NORMAL_SEGMENTATION` | admitted 且 HAG ready 的 measured point | 是 |
| `membership_review` | V4 review blocks 中 primary route 为 measured review 且 membership 仍需复核的点 | measured point，未进入正式 normal | 是 |
| `hag_review` | V4 review blocks 中 membership admitted 但 HAG readiness 为 review 的点 | measured point，不得进入 normal | 是 |
| `excluded_audit` | FRV payload 中 `EXCLUDED_NON_CANDIDATE`/`MANUAL_EXCLUDE`，source replay 按 original index 恢复 XYZ | 明确排除但可审计；不计入 V4 point input | 否，可开关 |
| `unrecovered_regions` | V4 coverage report/coverage records 的 `UNRECOVERED_REGION_REVIEW` | 区域记录，不是 point；有 geometry 才绘制 bbox/marker，否则只显示审计表 | 是 |
| `original_background` | source LAS 的确定性显示采样 | 漏检背景，不改变任何正式结果 | 否，可开关 |
| `wire_review_overlay` | review evidence 中 wire provider/relation、nearest wire、centerline distance | 诊断 overlay；首版可用颜色/符号叠加，不创建新的 wire 判定 | 否，可开关 |
| `ground_low_hag_overlay` | HAG state/reason/quality flags | 诊断 overlay；不改变 route | 否，可开关 |

图层互斥规则：每个 measured `original_index` 只能落入 normal、review 或 excluded 之一；excluded 不得同时出现在 V4 measured layers；blocked region 不能转成 point。显式记录 primary route，不能仅凭颜色推导 route。

## 6. Coordinate semantics

- FRV/V4 正式点坐标为 `xyz_local_m = source_xyz - SourceIdentity.origin_context`，使用 float64 且 origin 只减一次。
- source loader 的 XYZ 是 source provider 读取的原始 measured XYZ；背景显示转为相同 local frame 后才可与 V4 layers 叠加。
- hover 必须显示 `original_index`、route/state、reason flags、HAG/raw HAG/clamped HAG、wire relation/distance、quality flags；显示的 `xyz_local_m` 不得被误标为绝对坐标。
- 非零 origin 用 dedicated test 覆盖：一个点的 local XYZ 必须等于绝对 XYZ 减 identity origin，禁止使用 header min、scope min 或再次减 origin。
- source/artifact identity、coordinate frame、units、origin、count、bounds 任一不一致时 fail closed，不尝试平移“修正”。

## 7. Sampling and resource strategy

硬原则：正式计数、route 守恒、digest 和 acceptance flags 使用全量 validated arrays；只有 Plotly 展示数组采样。`unrecovered_regions` 是 region audit record，不是 point，永远不进入 point budget。

冻结配额：默认 `--max-total-points=250000`，绝对硬上限为 `500000`；默认 `--max-points-per-layer=100000`，默认 `--max-background-points=50000`。三个 requested cap 都必须为正整数；`max_total_points` 的合法范围为 `1..500000`，`max_points_per_layer` 与 `max_background_points` 的合法范围均为 `1..100000`。用户请求的 total cap 超过 `500000`，或任一 cap 小于 `1`，CLI 立即失败并且不生成 HTML。global `max_total_points` 优先于任何单层 cap：每个点必须同时满足 global remaining、所属 base-layer remaining，以及 background 点专属的 background remaining；global remaining 为零时立即停止。

每次运行使用固定 allocation priority（从高到低）：

1. `unrecovered audit records`：只写完整 region audit table/geometry status，不占 point cap；
2. `membership_review` points；
3. `hag_review` points；
4. 带 wire conflict 或任一非零 quality flag 的 points；这些点保留在其正式 base layer，不能重复生成第二个 point；
5. `excluded_audit` points；
6. `normal_admitted` points；
7. `original_background` points。

分配算法固定为：先按上列 priority bucket，再按 `(base_layer, original_index)` 排序；每个 bucket 先全量保留仍同时落在 layer cap 和 global cap 内的小类别，超出可用配额时对按 `original_index` 排序的 bucket 使用固定 seed 的均匀确定性索引选择。进入下一 bucket 时继承剩余 global cap 和各 layer remaining；因此 wire/quality 点只占一次名额，且不会重复计数。相同输入、相同 seed、相同 caps 即使 published block 行顺序改变，也必须得到相同的 `original_index` 集合。默认总 cap 下 normal 与 background 只使用前序层保留后的剩余配额；normal 优先于 background，background 还受独立 `50000` cap 限制。小类别在配额足够时全量保留。

cap 语义固定为：用户请求的 `max_total_points` 超过绝对硬上限 `500000` 时拒绝运行；full point count 超过用户请求的 `max_total_points` 时不失败，而是按上述 priority 和 deterministic sampling 缩减；实现不得自动放宽 requested cap，也不得生成 displayed point total 超过 requested cap 或 `500000` 的 HTML。summary 必须逐层记录 `full_count`、`displayed_count`、`requested_layer_cap`、`applied_cap`、`dropped_display_count`，并在全局记录 `requested_total_cap`、`applied_total_cap`、`total_full_point_count`、`total_displayed_point_count`、`total_dropped_display_count`、`requested_background_cap`、`sampling_seed`、完整 `allocation_priority`、`default_total_cap`、`hard_total_cap` 和拒绝/成功状态。

HTML 使用 self-contained Plotly；输出目录只写 HTML、JSON 和可选 checklist，不写 artifact root，不覆盖任何输入。

## 8. CLI design

建议最终入口：

```powershell
E:\Miniforge\envs\treeline\python.exe `
  scripts/run_frv_v4_input_visualization.py `
  --source <source.las> `
  --frv-artifact-root <frv-root> `
  --v4-input-root <v4-root> `
  --output <output.html> `
  --summary-json <summary.json> `
  --review-checklist <checklist.md> `
  --max-total-points 250000 `
  --max-background-points 50000 `
  --max-points-per-layer 100000 `
  --sampling-seed 0 `
  --show-original-background `
  --show-excluded `
  --show-review
```

参数规则：

- `--source`、`--frv-artifact-root`、`--v4-input-root`、`--output` 为 READY 模式必需；路径必须显式，禁止 scene 名称隐式发现。
- `--scene-simple` 只允许成为对 `data/samples/scene_simple.las` 及其正式 artifact/root 的显式便利别名，不能在当前目录搜索任意结果。
- `--max-total-points`、`--max-background-points`、`--max-points-per-layer`、`--sampling-seed`、三个显示开关、`--summary-json`、`--review-checklist` 可选。`--max-total-points` 默认 `250000` 且合法范围为 `1..500000`；`--max-points-per-layer` 默认 `100000`、`--max-background-points` 默认 `50000`，二者合法范围均为 `1..100000`。full data 超过任一 requested cap 时执行确定性缩减；只有参数值非法时才拒绝运行。
- `--output` 的父目录可创建；目标 HTML 已存在时默认拒绝覆盖。输入 root 在任何情况下只读。
- CLI 不接受“重新运行 FRV”的参数，不调用 membership builder，不调用 V4 handoff builder，不接收新的植被阈值。

## 9. Output artifacts

首版输出：

1. 一个自包含 `<output>.html`，含 3D 旋转/缩放/平移、Plotly 图层开关、hover 和截图/浏览器另存截图能力；
2. 可选 `<summary>.json`，包含 source/artifact/handoff identity 摘要、各 layer 全量 count、显示 count、比例分母、unrecovered region count、完整性 flags、采样策略和 artifact status；采样字段至少包括 `requested_total_cap`、`applied_total_cap`、`total_full_point_count`、`total_displayed_point_count`、`total_dropped_display_count`、`requested_layer_cap`、`requested_background_cap`、`sampling_seed` 和 `allocation_priority`；
3. 可选 `<checklist>.md`，预填本文件第 10 节的人工检查表和结论枚举。

Summary 的 point 比例分母只使用 `source_point_count`；region 比例若存在只单独以 region count 表示，禁止把 region count 加入 point total。至少输出 `frv_routing_complete`、`frv_membership_decision_complete`、`segmentation_input_complete`、`v4_routing_complete`、`xyz_recovery_complete`、`hag_ready_complete` 和 `artifact/result status`。

## 10. Human acceptance checklist

每个真实场景保存一份 checklist，逐项填 `PASS / FAIL / N/A`、截图编号、original-index/region key 证据和备注：

| 检查项 | 关注点 |
|---|---|
| 树冠主体 | normal 是否覆盖主体，是否存在明显整片缺失 |
| 树冠边缘 | 是否有环状漏点或只剩内部点 |
| 树干 | 是否被系统性排除或仅落入合理 review |
| 导线混入 | 导线是否进入绿色 normal/admitted |
| 杆塔/建筑 | 边缘是否混入 normal |
| 地面 | normal 是否大量低 HAG/地面点 |
| 导线附近枝叶 | 真实枝叶是否被整片删除，wire review 是否集中在合理边界 |
| review 集中性 | membership/HAG review 是否集中在证据不足或边界位置 |
| blocked/unrecovered | 是否有整片 blocked，是否每条都有可追溯 key/geometry 或明确缺 geometry |
| normal 连续性 | 是否存在明显断层、条带缺失或异常空洞 |
| 树间桥接 | 输入污染是否让不同树形成连续 normal 桥接 |
| V4 适用性 | 是否可继续正式单木分割 |

结论判定：

- `PASS_READY_FOR_V4`：无关键混入/漏检，所有正式 complete flags 满足要求；
- `PASS_WITH_REVIEW_NOTES`：可进入 V4，但存在已记录且不阻断的局部 review；
- `FRV_RULE_ADJUSTMENT_REQUIRED`：误判模式稳定指向 FRV reason/decision 规则；
- `DATA_OR_ARTIFACT_BLOCKED`：身份、artifact、coverage、坐标、资源或 blocked geometry 无法可信验证。

## 11. Exact implementation file scope

代码检查后的首版精确清单：

- `algorithms/vegetation/frv_v4_visualization.py`：只读 loader、formal layer data model、route/identity/coordinate validation、确定性分层采样、summary schema、Plotly renderer、输出清理。
- `scripts/run_frv_v4_input_visualization.py`：参数解析、输入模式、输出路径策略、错误映射、调用上述模块；不包含 FRV/V4 决策。
- `tests/vegetation/test_frv_v4_visualization.py`：数据模型、loader、映射、采样、统计、HTML/JSON 单元测试。
- `tests/scripts/test_run_frv_v4_input_visualization.py`：CLI 参数、路径和 scene_simple smoke test。

不修改：`core/contracts/*`、`algorithms/vegetation/frv_membership.py`、`frv_artifacts.py`、`v4_inputs.py`、现有 V3 visualization/script/tests、`pyproject.toml`、binary fixture、`.claude/`。首版不需修改 `algorithms/vegetation/__init__.py`；CLI 可直接导入新模块。输出文件只落在临时目录或明确的 visualization output 目录，不提交仓库。

## 12. Test matrix

测试动态生成小型数据，不依赖人工打开浏览器：

1. 每个正式 V4 route 映射到正确图层；
2. normal/review/excluded 互斥且 route 守恒；
3. unrecovered region 永不生成 point；
4. 全量统计不随 display sampling 改变；
5. 相同 seed 采样稳定；
6. 改变输入行顺序仍得到相同 `original_index` 显示集合；
7. per-layer cap 的 0、1、恰好 cap、超 cap 边界；
8. `max-total-points=0` 拒绝；
9. `max-total-points=1` 的全局 priority 分配；
10. `max-total-points=250000` 默认边界；
11. `max-total-points=500000` 允许；
12. `max-total-points=500001` 拒绝且不生成 HTML；
13. 多层 full count 合计超过 global cap 时严格按 allocation priority 分配；
14. 相同输入、相同 seed、不同 block 行顺序得到相同 displayed `original_index` 集合；
15. `total_displayed_point_count` 永不超过 `max_total_points`；
16. 空 normal；
17. 空 review；
18. 只有 blocked regions；
19. 非零 origin 的 local XYZ；
20. source/artifact identity mismatch 拒绝；
21. corrupt FRV/V4 block、manifest、digest 拒绝；
22. HTML 是完整 self-contained 文档且包含各 layer 名称/hover 字段；
23. JSON summary schema、全局/逐层 cap 字段、比例分母和 region/point 分离；
24. `max-total-points=0`、`max-points-per-layer=0`、`max-background-points=0` 均被拒绝；
25. `max-total-points=500001`、`max-points-per-layer=100001`、`max-background-points=100001` 均被拒绝；
26. 任一非法 cap 后 HTML、summary JSON、checklist 和临时文件均不存在，且不产生 partial output；
27. 合法 requested cap 下 full data 超过 total cap 时正常生成 deterministic sampling 输出，且 displayed total 不超过 requested total cap；
28. CLI 缺参、非法 seed、冲突路径参数失败；
29. 输出目录创建，artifact/identity/digest/coordinate 失败后全部临时文件清理；
30. 输入 artifact 不被覆盖或改写；
31. 运行前后正式 result/输入 bytes 和 digest 不变；
32. `scene_simple` 显式路径 smoke test。

Browser 不作为测试通过条件；HTML 测试至少检查 Plotly 文档、layer trace、customdata/hover 字段和 summary 内容。

## 13. Directed verification commands

实现阶段完成后执行以下定向检查；本计划阶段不执行这些会产生输出的命令：

```powershell
python -m pytest tests/vegetation/test_frv_v4_visualization.py tests/scripts/test_run_frv_v4_input_visualization.py -q
python -m pytest tests/vegetation/test_tree_visualization.py tests/scripts/test_run_v3_scene_visualization.py -q
python -m ruff check algorithms/vegetation/frv_v4_visualization.py scripts/run_frv_v4_input_visualization.py tests/vegetation/test_frv_v4_visualization.py tests/scripts/test_run_frv_v4_input_visualization.py
E:\Miniforge\envs\treeline\python.exe scripts/run_frv_v4_input_visualization.py --help
E:\Miniforge\envs\treeline\python.exe scripts/run_frv_v4_input_visualization.py --source data/samples/scene_simple.las --frv-artifact-root <frv-root> --v4-input-root <v4-root> --output <temp>\scene_simple.html --summary-json <temp>\scene_simple.json --review-checklist <temp>\scene_simple.md --max-total-points 250000 --max-background-points 50000 --max-points-per-layer 100000 --sampling-seed 0
git diff --check
git status --short
git diff --name-status
git diff --stat
git diff --cached --name-status
```

真实场景命令中的 `<frv-root>`、`<v4-root>` 必须由已完成的正式 FRV/V4 handoff 提供，不能由查看器现场生成。

## 14. Split implementation and human acceptance gates

### A. Implementation Acceptance Gate

该 gate 只判断工具代码是否完成，不要求真实人工结论。必须同时满足：

1. 四个精确实施文件完成，且没有超出第 11 节范围；
2. focused tests 通过；
3. `tests/vegetation/test_tree_visualization.py` 与 `tests/scripts/test_run_v3_scene_visualization.py` regression 通过；
4. Ruff 通过；
5. CLI `--help` 通过；
6. 测试动态生成的小型 formal FRV artifact/V4 handoff smoke test 通过，覆盖 normal/review/excluded、非零 origin、identity 校验和 deterministic budget；
7. 不调用 FRV/V4 builder，不改正式结果、不改输入 artifact、不新增依赖。

通过后，工具可以进入代码 review 和 Git closure 授权；即使真实 artifact roots 尚未准备好，也不阻塞工具代码实现和提交授权。

### B. Human Scene Acceptance Gate

代码工具闭环后单独执行，不与 Implementation Gate 合并。必须使用 `scene_simple` 或当前可用正式场景，并至少包含一个真实输电走廊场景；每个场景必须提供已经完成并可校验的 FRV artifact root 和 V4 input root，生成 HTML、summary JSON 和 checklist，由 Owner 按第 10 节逐项检查并给出四态结论：

- `PASS_READY_FOR_V4`
- `PASS_WITH_REVIEW_NOTES`
- `FRV_RULE_ADJUSTMENT_REQUIRED`
- `DATA_OR_ARTIFACT_BLOCKED`

只有前两种结论可以授权正式 V4 segmentation；后两种结论保持未授权。若真实 roots 尚未准备好，Human Scene Gate 处于 pending，但不倒推阻塞 Implementation Gate。blocked row 没有可信 geometry 时必须显示 audit table 和 `missing-geometry` 状态，不得伪造 bbox 或 point；该场景结论固定为 `DATA_OR_ARTIFACT_BLOCKED`。

## 15. Git closure strategy

本轮 Git 写操作为零。实施时保持本文件和代码范围内的单次实现、一次综合审查、一次提交推送节奏；输出 HTML/JSON/checklist 只放临时或 ignored visualization 目录。提交前应确认只包含上述四个新代码/测试文件及本计划文档（若本轮执行的是计划 review，则只包含本计划文档），暂存区无意外文件；不得处理 `.claude/`。是否提交、推送以及最终提交信息留到后续授权轮次。

## 16. Risks and fail-closed behavior

| 风险 | 处理 |
|---|---|
| V4 handoff root 缺公开 loader | 新模块按正式 JSON/block schema 读取并调用既有 decoder；字段不全即拒绝 |
| excluded 没有 handoff XYZ | 从已校验 FRV artifact 取 original index，再用 source 读取；identity 不符即拒绝 |
| source path 与 artifact 不同 | 比较 source file/content/record digest、count、bounds、frame、units、origin；失败即拒绝 |
| blocked record 没有 bbox | 不从全局 bounds 猜 region；显示独立 blocked 审计信息并标记 `DATA_OR_ARTIFACT_BLOCKED` |
| 非法 display cap | 任一 cap <= 0、`max_total_points > 500000`、`max_points_per_layer > 100000` 或 `max_background_points > 100000` 均在创建任何输出前返回 `CLI_REJECTED` 非零状态；不生成 HTML、summary JSON、checklist，不留下临时文件或 partial output |
| 合法 cap 下 full data 超过 requested cap | 不失败；按固定 allocation priority 和 deterministic sampling 缩减，最终 displayed total <= requested total cap |
| 大点云导致浏览器崩溃 | 仅使用合法 cap；artifact/identity/digest/coordinate validation 失败时 fail closed，正式输出为零且临时文件全部清理 |
| Plotly/浏览器版本差异 | 依赖使用现有 `plotly`；测试只依赖静态 HTML 结构，不要求在线服务 |
| 新实现意外重跑 FRV/V4 | 代码审查禁止调用 builder、阈值参数和新植被判断；输入仅为已发布结果 |
| 输出覆盖输入或已有结果 | 输出路径独立、已存在目标拒绝覆盖、临时目录失败清理 |

## 17. Explicit non-goals

- 正式 V4 单木分割、树顶检测、多树顶归并、冠层分水岭、单木编号；
- 冠幅/树高计算、树线距离、风险预警；
- 修改 FRV 分类/route/reason 规则或 V4 input contract/runtime；
- 图形化人工编辑点云、大型桌面 GUI、Streamlit 服务、常驻 Web 平台、数据标注平台；
- 为可视化重新执行另一套植被判断；
- 新增深度学习/GPU 依赖、网络服务或 binary fixture。

第一版只负责查看、统计完整性和人工验收。
