# V4 单木分割合并版综合实施计划

文档状态：`PLAN-REVIEW-CANDIDATE`

计划架构：`V4-BUNDLE-A + V4-BUNDLE-B`

Repository：`E:\TreeLineWarning`

Branch：`feature/algorithm`

调查基线 HEAD：`8f8042943aff03038707dd91963cf0410ee0f899`

调查基线提交：`feat: add reviewed FRV-VIS-A1 visualization`

Remote divergence：`origin/feature/algorithm...HEAD = 0 0`

Formal V4 segmentation：`NOT STARTED`

FRV-VIS-A1 Implementation Gate：`CLOSED`

FRV-VIS-A1 Human Scene Gate：`PENDING`

V4 seed architecture：`JOINT_TRUNK_APEX_WITH_CSP_STYLE_PATH_AND_ROTATING_PROFILE_PRIOR`

本文件是 V4 第一版单木分割和结果查看的唯一综合实施计划。本轮只修订本计划，不实现
代码、不修改测试、不执行 Git 写操作，也不开始正式 V4 segmentation。`.claude/` 保持
`EXCLUDED_AND_IGNORED`，不读取、不处理、不纳入项目范围。

## 1. 权威、目标和授权边界

解释优先级如下；下级只能细化实现，不能改写上级冻结语义：

1. `docs/full-resolution-vegetation-membership-design.md`：FRV membership、scope、HAG、
   provider、override 和上游 completeness 权威；
2. `docs/v4-tree-segmentation-design.md`：V4 recovery、同步二维 claim、三维切片、最终
   original-point assignment、安全 hand-off 和 clearance 边界权威；
3. `docs/frv-a2-to-v4-handoff-plan.md`：当前已实现 FRV-to-V4 handoff 的精确 contract、
   block 和 runtime 边界；
4. `docs/frv-to-v4-visualization-acceptance-plan.md`：当前 FRV-VIS-A1 输入查看器、Human
   Scene Gate 和真实场景授权边界；
5. 本文件：不改变上述语义，只把原设计中的多个 V4-A/V4-B 小阶段压缩成两次正式
   implementation bundle。

目标是在本综合计划冻结后不再编写 V4-A1、A2、B1、B2 等子计划：Bundle A 的代码、contracts、
artifact I/O、CLI 与 synthetic/focused/regression tests 可按一次合并实现立即推进；FRV-VIS Human
Scene Gate 只继续约束真实/calibration scene 的 formal V4 segmentation execution 与真实 result
publication。整体仍只包含：

- `V4-BUNDLE-A — Single-Tree Segmentation Core`：从正式 V4 handoff 到
  `original_index -> tree_id`、可解释 tree records 和完整 review/unassigned 守恒；
- `V4-BUNDLE-B — Result Visualization & Acceptance`：只读取 Bundle A 正式结果，生成
  self-contained HTML、JSON summary 和可选 checklist。

计划现在可以冻结；计划冻结后 Bundle A **code implementation** 可由 Owner 单独授权立即开始。
FRV-VIS-A1 Human Scene Gate 继续作为 **formal real-scene execution/publication** 的前置门禁。
Gate pending 不产生第三个计划，也不把 Bundle A 拆成更多正式阶段。

## 2. Repository 调查结论

### 2.1 当前正式 V4 input

现有 `core/contracts/v4_inputs.py` 已提供：

- `V4ValidatedFrvMembershipView`；
- `V4RecoveredVegetationSet`；
- `V4UnresolvedMeasuredPointSet`；
- `V4UnrecoveredRegionRecord`；
- `V4RecoveryCoverageReport`；
- `V4InputHandoffResult`、`V4InputOutputReference` 和 closed error enums。

现有 `algorithms/vegetation/v4_inputs.py::build_v4_input_handoff` 发布：

```text
<v4-input-root>/validated_view.json
<v4-input-root>/coverage_report.json
<v4-input-root>/handoff_manifest.json
<v4-input-root>/normal/v4nrm-block-XXXXXXXX.v4blk
<v4-input-root>/review/v4rev-block-XXXXXXXX.v4blk
```

normal block 的正式 segmentation 字段为：

| 字段 | dtype/shape | 语义 |
|---|---|---|
| `original_index` | `<i8[N]` | 唯一 measured source-row identity |
| `xyz_local_m` | `<f8[N,3]` | `source_xyz - SourceIdentity.origin_context`，origin 只减一次 |
| `hag_m` | `<f4[N]` | FRV 已提供并已验证的 clamped HAG cast，不在 V4 重算 |
| `membership_state` | `u1[N]` | 只能是 `VERIFIED_VEGETATION_CANDIDATE` 或 `MANUAL_INCLUDE` |
| `hag_evidence_state` | `u1[N]` | 保留 FRV HAG evidence state |
| `segmentation_hag_readiness` | `u1[N]` | normal 中必须全部为 `READY` |
| `candidate_reason_flags` | `<u4[N]` | 上游原因，不是 tree label |

normal block 通过所在 block 和 coverage report 固定 route 为
`NORMAL_SEGMENTATION`。它没有单独的上游 quality array；Bundle A 不得发明 FRV quality，
只能新增明确命名的 V4 segmentation quality/reason。review block 才含 raw/clamped HAG、
quality、wire relation、distance 和显式 `primary_route`，但这些行不进入 segmentation。

正式进入 Bundle A 的 original indexes 恰为：

```text
membership_state in {VERIFIED_VEGETATION_CANDIDATE, MANUAL_INCLUDE}
AND segmentation_hag_readiness == READY
AND exact XYZ recovery succeeded
AND primary route == NORMAL_SEGMENTATION
```

以下集合必须保持隔离：

- `REVIEW_REQUIRED_MEMBERSHIP` 和 admitted-but-HAG-not-ready measured points：只在
  `V4UnresolvedMeasuredPointSet`；
- `EXCLUDED_NON_CANDIDATE` 和 `MANUAL_EXCLUDE`：只在 excluded/audit route；
- coverage-blocked 或 recovery-failed region：只在 `V4UnrecoveredRegionRecord`；
- 上述三类均不得创建 apex、hypothesis、tree ID 或 normal member。

Bundle A 开始前必须同时验证：

```text
frv_canonical_inventory_complete == true
frv_required_inventory_complete == true
frv_approved_hard_rejection_complete == true
frv_scope_coverage_complete == true
frv_routing_complete == true
frv_membership_decision_complete == true
source_identity_complete == true
xyz_recovery_complete == true
segmentation_input_complete == true
v4_routing_complete == true
```

`measured_clearance_complete` 在当前 handoff 合法地为 false，因为 clearance 不属于本计划。
review point 的存在不阻塞其余 normal population segmentation，只要上述 routing 和 normal
input completeness 成立。

### 2.2 当前 handoff loader 缺口及 Bundle A 内解决方式

仓库当前没有 public handoff-directory loader。FRV-VIS-A1 在
`algorithms/vegetation/frv_v4_visualization.py` 内部用私有流程读取 view/report/manifest，
并通过 `decode_frv_block` 解码 normal/review blocks；该查看器不是 segmentation API。

另一个经实际代码确认的跨进程完整性缺口是：`V4InputHandoffResult.handoff_digest`、
`normal_set_digest` 和 `review_set_digest` 在 builder 运行时计算，但当前 v1
`handoff_manifest.json` 只持久化 block count 和三个上层 digest，没有持久化 block digest
inventory 或最终 handoff digest。结构损坏通常可被 decoder 拒绝，但一个被重新编码成合法
结构的 block 没有发布期 digest 可供正式跨进程比较。

这不是拆分新阶段的理由。Bundle A 的第一项内部工作一次完成：

1. 在 `core/contracts/v4_inputs.py` 增加严格的 digest-bound
   `V4InputArtifactManifest` v2 contract；
2. v2 manifest 持久化 canonical normal/review block relative paths、每块 SHA-256、
   `normal_set_digest`、`review_set_digest` 和 `handoff_digest`；
3. `build_v4_input_handoff` 在 atomic publish 前计算并写入这些字段；
4. 新增 public `load_v4_input_handoff(root_path)`，只从 root 内 canonical inventory 读取，
   验证所有 bytes、schemas、counts、digests、routes、identity 和 conservation 后重建现有
   `V4InputHandoffResult`；
5. segmentation 的正式跨进程入口只接受 v2。旧 v1 保留给已有 FRV-VIS-A1 只读兼容，
   但因缺持久化 handoff digest，formal Bundle A CLI 对 v1 fail closed；
6. Bundle A 真实运行前用升级后的既有 handoff builder 重新发布 v2 root。这个动作不运行
   segmentation、不改变 FRV membership，也不另立计划；
7. FRV-VIS-A1 loader 兼容 v1/v2；遇到 v2 时额外验证所有 block hashes。

这项扩展只增强冻结设计 D02/D09 的 artifact integrity，不改变 membership、route、HAG、
point identity 或 completeness 语义。

### 2.3 V3 可复用与不可复用边界

实际检查了 `canopy_height_model.py`、`treetop_candidates.py`、
`canopy_segmentation.py`、`tree_segmentation.py`、V3 contracts/artifacts/tests。

可复用的是算法和工程原则：

- 网格化 CPU-first spatial indexing 和 resource preflight；
- deterministic local-maximum mask、equal-height plateau collapse；
- median-annulus prominence、稳定 `lexsort`/tie rule；
- 多 marker 在共同 domain 中竞争，而不是逐树 flood-fill；
- bottleneck/path continuity、ambiguity fixed point 的确定性思想；
- canonical read-only arrays、chunked validation、ragged membership；
- manifest/SHA/inventory/corruption-test 和 no-partial-output 模式。

以下代码/contract 绝不能直接作为 V4 正式语义复用：

| 现有对象 | 禁止直接复用原因 |
|---|---|
| `TreeSegmentationInputSet` | V2/downsampled row aligned、cluster status/wire filter 语义，不是 FRV normal original-index population |
| `CanopyHeightModelResult` | identity 绑定 V3 input；当前 CHM 机械 cell max、V2 support bits 和 V3 eligibility filter 不符合 V4 robust CHM authority |
| `TreetopCandidateResult` | apex 是 grid cell marker，缺 measured `original_index`，且没有 multi-apex/tree-hypothesis 语义 |
| `CanopySegmentationResult` | 每格只有一个 owner；score 主要是 height bottleneck/path distance；没有 retained secondary claim、density/HAG trend、multi-height slice 或 later merge |
| `build_tree_segmentation_result` | 把同一二维 cell 的所有 V2 rows直接映射为 segment；输出全部 provisional，未做三维同步 tree comparison |
| `TreeInstanceRecord` | 强依赖 V2 source clusters/readiness；只有单 apex summary；不表达 secondary apices、relation evidence、slice continuity 或 V4 review state |
| `TreeSegmentationResult` | authoritative identity 是 V2 row index，不是 `original_index`；无 `ASSIGNED/UNASSIGNED/REVIEW_REQUIRED` dense conservation |
| V3 artifact schema/loader | 只接受 V3 schema，且仓库没有 V3 publisher；不能承载 V4 evidence |
| FRV-VIS-A1 viewer | 只看 segmentation 之前的 route，不读取或解释单木结果 |

因此 V4 允许复刻或抽取经过测试的纯算法原则，但不得把 V4 normal points 包装成伪 V3
cluster，也不得调用 V3 result constructor 获得正式 V4 tree IDs。

### 2.4 Pre-Bundle-A Human Scene Gate Bootstrap

这是 Human Gate 的执行准备，不是第三个 implementation bundle，不产生 production code、计划、
Git commit 或 Git write。只读调查在 baseline HEAD 确认了以下真实入口：

| 角色 | 当前仓库权威入口 |
|---|---|
| FRV source/runtime | `algorithms/io/source_provider.py::load_frv_source` 及已经审核的 HAG/geometry/wire provider sessions |
| FRV membership builder | `algorithms/vegetation/frv_membership.py::build_frv_membership` |
| FRV artifact loader/publisher | `load_frv_membership_result`；builder 内通过 `algorithms/vegetation/frv_artifacts.py::publish_directory` atomic publish |
| current V4 handoff builder | `algorithms/vegetation/v4_inputs.py::build_v4_input_handoff`，当前发布 v1 root |
| FRV-VIS runtime/CLI | `algorithms/vegetation/frv_v4_visualization.py::run_frv_v4_visualization`；`scripts/run_frv_v4_input_visualization.py` |

`outputs/` 当前没有任何 `execution_result.json`，因此没有可直接交给 viewer 的已发布 formal FRV/
V4 roots；这正是 bootstrap 必须先运行、而不是让 viewer 隐式补建输入的原因。

仓库当前没有 FRV membership 或 V4 handoff builder CLI；不能伪称 viewer CLI 会补建输入，也不能
复制 test helper 当正式运行器。Bootstrap 的输入边界改为“Owner 只提供正式 artifact 路径、已审核
配置记录和审批记录；runtime objects 必须在同一个正式进程内由现有 typed contracts/public loaders
构造”，不得让 Owner 手工拼 runtime object、dict、JSON 或 test fixture。

#### 2.4.1 Pre-Bundle-A Bootstrap Exact Runtime Construction

以下调用链是本计划的正式 bootstrap authority。它不新增 production bootstrap script，也不新增
第三个 Bundle；实际执行可由一次临时、非 Git 的 Python runtime session 完成。

1. **Source identity/reference**：从与当前 source LAS 绑定的正式 A1/preprocessing artifact 读取并
   验证 `SourceIdentity`；为 raw LAS 计算 exact SHA-256/size，构造现有
   `FrvA2ArtifactComponentBinding(component_key="source_file", component_kind=SOURCE_FILE, ...)`。
   `artifact_reference_id` 严格使用既有 A2 v1 派生公式：

   ```text
   "frv-a2-ref-v1|" + provider_role.value + "|" + provider_key + "|" +
   canonical component tuple
   ```

   然后使用现有 frozen `FrvSourceProviderReference` typed constructor，字段必须来自已验证
   `SourceIdentity`、source component、reviewed configuration/provider-config digests 和固定
   source provider interface/schema/algorithm versions；`runtime_access_path` 只能是显式 source
   LAS path。禁止从 filename/source_id 猜路径。

2. **Finalized source result**：唯一 public source loader 为：

   ```python
   with load_frv_source(
       reference,
       frv_configuration=frv_configuration,
       provider_config=source_provider_config,
       chunk_size=source_chunk_size,
   ) as source_session:
       for _chunk in source_session.iter_chunks():
           pass
       source_result = source_session.finalize()
   ```

   不允许从 LAS 旁路构造 `FrvSourceProviderResult`，也不允许复用未 finalized session。

3. **Provider configs**：必须使用现有 exact typed configs：
   `FrvSourceProviderConfig`、`FrvHagProviderConfig`、`FrvGeometryProviderConfig`、
   `FrvWireProviderConfig`。每个 config 的 interface/config digest 必须与同一个
   `FrvConfiguration` 精确匹配；不得使用 filename/scene-specific hidden defaults。
   `FrvConfiguration` 必须满足正式 publication 的 production gate；TEST_ONLY 或
   CALIBRATION_CANDIDATE 不能伪装成正式 FRV publication config。

4. **Scope evidence**：`FrvScopeEvidenceReference` 直接使用现有 frozen typed constructor，
   输入只能是当前 source 对应的 A1、V1、V2 三个 JSON/NPZ pair 的六个显式绝对路径，以及
   对这些六个 bytes 逐一计算并验证得到的六个 `FrvA2ArtifactComponentBinding`。构造后必须由
   `build_frv_membership` 的现有 scope prevalidation 再次执行：`PreprocessingResult.from_dict`、
   `load_vegetation_candidates`、`load_vegetation_instances`、六 component SHA/size、A1/V1/V2
   source/origin/index/cluster mapping 全一致性验证。没有匹配的正式 A1/V1/V2 artifact 时，
   bootstrap 直接记录 `DATA_OR_ARTIFACT_BLOCKED`；禁止以 test fixture 或手工生成 pair 替代。

5. **Wire authority**：使用现有 `FrvWireAuthorityManifest` + `FrvWireAuthorityReference` typed
   contracts。`W1_MODEL_SET` 必须绑定已经批准且全部 confirmed/distance-ready 的 W1 JSON/NPZ；
   `EMPTY_INVENTORY` 只有在存在明确审批并能证明 authoritative inventory 为空时才合法。authority
   manifest/component/W1 pair 的 SHA、size、source identity、origin 和 geometry digest 必须全部
   通过 P-WIRE prevalidation；缺批准 authority 时记录 `DATA_OR_ARTIFACT_BLOCKED`，不得把“没有
   文件”解释成 EMPTY authority。

6. **Scope/membership policies and overrides**：使用现有 exact typed
   `FrvScopePolicyConfig` 与 `FrvMembershipPolicyConfig`；其 digest 必须分别等于同一
   `FrvConfiguration` 中冻结的 policy digest。`approved_overrides` 只能是已经通过现有
   `FRVManualOverrideArtifact` validator 的 tuple；没有批准 override 时使用空 tuple，不创建
   synthetic override。所有 config/policy/override 都必须显式记录 digest。

7. **P-HAG / P-GEOMETRY / P-WIRE execution ownership**：bootstrap **不得在
   `build_frv_membership` 外另跑一套 provider 结果再塞回 membership**。现有
   `build_frv_membership` orchestrator 本身按已冻结实现调用正式 provider runtime；其底层 public
   APIs 分别是 `load_frv_hag(...)`、`load_frv_geometry(...)`、`load_frv_wire(...)`，并在同一
   execution 中 finalize/交叉验证 provider results。bootstrap 只准备其正式输入 reference、
   `source_result`、scope evidence、wire authority、configs/policies/overrides。

8. **FRV -> V4 -> viewer**：完成上述 typed construction 后，严格按下列 public API 顺序执行；
   viewer 始终只读已发布 roots：

```python
membership = build_frv_membership(
    reference,
    source_result,
    scope_evidence,
    wire_authority,
    approved_overrides,
    frv_configuration=frv_configuration,
    hag_config=hag_config,
    geometry_config=geometry_config,
    wire_config=wire_config,
    scope_policy=scope_policy,
    membership_policy=membership_policy,
    output_dir=frv_artifact_root,
    scratch_dir=scratch_dir,
    source_chunk_size=source_chunk_size,
)
reloaded_membership = load_frv_membership_result(frv_artifact_root)
assert reloaded_membership.execution_state is READY_FOR_V4

handoff = build_v4_input_handoff(
    reference,
    source_result,
    reloaded_membership,
    frv_configuration=frv_configuration,
    output_dir=v4_v1_input_root,
    source_chunk_size=source_chunk_size,
)
assert handoff.output_reference is not None
```

Bootstrap preflight 必须在创建 FRV/V4 output root 前确认：source/A1/V1/V2/W1-or-approved-empty
authority/config/policy/override inputs 全部存在且与同一 `SourceIdentity`/configuration digests
一致。缺失任一正式 upstream artifact、批准 authority 或 production config 时，不创建 partial
FRV/V4 root，Human Gate 结果只能是 `DATA_OR_ARTIFACT_BLOCKED`（或真实走廊缺失时额外记录
`REAL_CORRIDOR_SCENE: OWNER_INPUT_REQUIRED`）。这属于输入/审批可用性，不是要求新增 production
bootstrap API 的 architecture gap。

两次 `output_dir` 必须是新的 sibling roots；`publish_directory` 拒绝 overwrite。API session 结束后，
viewer 只接收已经发布并重新加载验证的 roots，绝不重跑、修补或 fallback 到 builder。最短 formal
artifact chain 冻结为：

```text
source LAS
  -> approved formal providers/scope/wire authority
  -> build_frv_membership
  -> published formal FRV artifact root + successful load_frv_membership_result
  -> build_v4_input_handoff
  -> published current V4 v1 handoff root
  -> existing FRV-VIS-A1 CLI
  -> self-contained HTML + summary JSON + review checklist
```

使用唯一固定解释器；`<run-root>` 是 owner 指定的非 Git artifact 目录，不能位于 tests fixture
目录。scene_simple 的 viewer 命令为：

```powershell
E:\Miniforge\envs\treeline\python.exe scripts/run_frv_v4_input_visualization.py `
  --scene-simple `
  --frv-artifact-root <run-root>\scene_simple\frv `
  --v4-input-root <run-root>\scene_simple\v4-v1 `
  --output <run-root>\scene_simple\frv-vis-a1.html `
  --summary-json <run-root>\scene_simple\frv-vis-a1-summary.json `
  --review-checklist <run-root>\scene_simple\frv-vis-a1-checklist.md `
  --show-review
```

`data/samples/scene_simple.las` 是 formal source，不是 test fixture；其 manifest 把它绑定到
`data/raw/scene_01_small/scan_small.las` 的 immutable crop、source identity、index map、point count
和 pre/post validation。它先验证上述整条工具链，但不能单独满足真实走廊 Human Gate。

只读调查还确认两个当前 real-corridor source candidates：

```text
E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las   (414,332,205 bytes)
E:\TreeLineWarning\data\raw\scene_02_large\scan_large.las (2,404,477,459 bytes)
```

至少对 `scan_small.las` 按同一 formal API chain 发布独立 FRV/v1 roots，并把 viewer 命令中的
`--scene-simple` 换成 `--source E:\TreeLineWarning\data\raw\scene_01_small\scan_small.las`；
`scan_large.las` 是第二个规模候选，不替代 first real-corridor run。Human reviewer 必须确认该源
确为输电走廊；若两候选在运行时均不可用或经 owner 判定不是走廊，则记录
`REAL_CORRIDOR_SCENE: OWNER_INPUT_REQUIRED`，但仍先完成 scene_simple formal chain 来验证工具链。

禁止手工伪造/编辑 JSON、block、digest 或 execution state，禁止 test fixture 冒充 Human artifact，
禁止 viewer 搜索 ad-hoc roots。每个 scene 记录 source/config/artifact digests、命令、解释器、HEAD、
输出 paths 和 checklist。Gate outcome 只能是：

```text
PASS_READY_FOR_V4
PASS_WITH_REVIEW_NOTES
FRV_RULE_ADJUSTMENT_REQUIRED
DATA_OR_ARTIFACT_BLOCKED
```

Human Gate 的授权边界拆成 **code implementation** 与 **formal real-scene execution** 两层：

- 综合计划冻结后，Bundle A 的 production code + synthetic/focused/regression tests 可以一次性开始实现，
  不需要等待真实 artifact bootstrap；这不构成 formal V4 segmentation run，也不得发布真实 scene result；
- 只有 `PASS_READY_FOR_V4` 或 `PASS_WITH_REVIEW_NOTES` 才允许 Bundle A 使用真实 FRV/V4 artifact
  执行 formal/calibration scene segmentation、发布真实 result artifact 或据此进入 Bundle B Human V4 gate；
- `FRV_RULE_ADJUSTMENT_REQUIRED` 或 `DATA_OR_ARTIFACT_BLOCKED` 不否定已经通过自动测试的 Bundle A
  代码，但继续阻止任何 formal real-scene execution；
- Bootstrap 使用 current v1 handoff 只服务 FRV-VIS Human Gate；正式 Bundle A real-scene run 仍必须
  按 2.2 升级并重新发布 digest-bound v2 handoff。

这样 Human Gate 继续保护真实数据语义，同时不让 artifact 可用性阻塞 V4 核心代码开发。

## 3. 合并后的总体架构

### 3.1 唯一正式 Bundle

```text
validated, digest-bound V4 normal input
  -> V4-BUNDLE-A: one implementation/review/Git closure
  -> V4TreeSegmentationResult + self-contained result artifact
  -> V4-BUNDLE-B: one implementation/review/Git closure
  -> HTML + JSON + optional checklist
  -> Human V4 acceptance gate
```

Bundle A 内部有算法 pass，但没有内部 authorization、plan、commit 或独立 Git closure。
Bundle B 不调用任何 segmentation builder，不改变 Bundle A result。

### 3.2 LiDAR-only 首版

首版只使用：`original_index`、measured `xyz_local_m`、FRV HAG、局部点密度、局部几何、
二维 topology、三维 slice continuity，以及 **显式但非强制的 measured trunk candidate /
trunk–apex association evidence**。树干与树顶在同一 Bundle A 中联合检测、联合建种子，但
“检测到树干”不是生成 Tree ID 的必要条件。只使用 Python standard library、现有 NumPy 和
现有仓库代码；不修改 `pyproject.toml`。

首版冻结的联合种子原则为：

```text
1 trunk <-> 1 apex
    = 最强、最简单的单木 seed case

1 trunk <-> N apex
    = 合法 multi-apex tree 候选，后续 3D evidence 决定

N trunk <-> 1 apex/crown
    = 合法 multi-stem tree 候选，不强制拆树

0 trunk <-> 1/N apex
    = 合法 apex-only fallback，不因树干遮挡而拒绝 tree

trunk without compatible apex
    = 保留 UNMATCHED_TRUNK diagnostic，不单独创建 final Tree ID
```

因此 **strict one-trunk-one-apex 不是 invariant**。树干是强结构锚点而非硬依赖；任何最终
merge/separate/assignment 仍须经过二维责任区、三维 slice/fragment continuity 和 final
relation evidence。

### 3.1A 文献启发但项目内独立冻结的两项增强

本计划额外吸收两类已有 LiDAR 单木分割思想，但 **不直接复制论文实现、参数或数据假设**：

1. **CSP-style trunk-rooted structural path evidence**：参考 TLS 单木分割中“先检测树干，再比较
   树干到冠层点的最短结构路径”的思路；本项目不建立全点 Dijkstra 图，而是在已经授权的
   bounded voxel/fragment graph 上计算局部、确定性的 trunk-rooted sparse shortest path，
   仅作为 `TrunkCandidate <-> ApexCandidate`、slice continuity 和 conflict resolution 的结构证据。
2. **Rotating-profile crown boundary evidence**：参考机载 LiDAR 中“以树顶为中心，从多个方向
   旋转垂直剖面，寻找冠缘/交叉谷点”的思路；本项目把它改造成 deterministic radial-profile
   prior，只读取正式 normal points 的 HAG/density/occupied support，用来辅助二维
   responsibility transition stop 和 crown-boundary diagnostics，不直接产生 Tree ID。

两项增强都必须保持：

```text
original_index authority
CPU-first bounded neighborhood
no global point O(N^2)
no mandatory trunk requirement
no direct literature parameter copying
no new implementation bundle
```

论文只提供算法思想依据；下面的所有阈值、normalization、tie rule、availability 和 fail-closed
语义均由本项目 Algorithm-V1 独立冻结，并由 synthetic fixed-vector tests 验证。

首版明确不使用 RGB/video、SAM、YOLO、PointNet++、SparseConv、GNN、任何 deep-learning
inference、GPU 或网络服务。结果 contract 可保留一个空的、版本化
`future_visual_evidence_reference=None`，但任何首版 decision/config/digest 不依赖它。

## 4. V4-BUNDLE-A — Single-Tree Segmentation Core

### 4.1 Bundle A 完成定义

一次实现必须覆盖：formal input validation、robust CHM、apex detection/prominence、
**trunk candidate detection、trunk candidate quality、trunk–apex association graph、
CSP-style trunk-rooted sparse structural path、rotating-profile crown boundary prior**、
multi-apex/multi-stem relationship、同步二维责任区、density/HAG/topology/profile stop、
reversible provisional groups、3D candidate domain、multi-height slices、slice boundaries、
adjacent continuity、fragment/hypothesis conflict graph、merge/keep/review、唯一 point assignment、
deterministic tree ID、conservation、fail-closed、tree geometry summaries、focused tests 和
full regression。

成功输出必须直接包含：

```text
original_index -> tree_id  (ASSIGNED points)
original_index -> REVIEW_REQUIRED
original_index -> UNASSIGNED
tree_id -> primary apex + secondary apices + optional primary/secondary trunks + members + geometry + evidence
```

不存在“apex 结果先提交，以后再合并”或“先二维分割，三维以后做”的中间正式完成态。

### 4.2 内部执行顺序

以下是一次 Bundle A 调用内的 pass，不是项目阶段：

1. 加载并验证 v2 V4 handoff，按 `original_index` canonicalize normal rows；
2. 建立 bounded XY/XYZ grid buckets、HAG bins 和 cell/column support summaries；
3. 从同一 normal full-resolution population **并行构造** robust CHM/apex proposals 与
   measured trunk candidates；“并行”表示同一算法阶段的独立 evidence extraction，不要求线程并行；
4. 分别计算 apex support/prominence 与 trunk candidate quality，保留 rejected/suppressed/
   unmatched diagnostics；
5. 建立 bounded `TrunkCandidate <-> ApexCandidate` association graph；不做强制 1:1 matching，
   保留 one-to-one、one-trunk-multi-apex、multi-trunk-one-apex 和 apex-only fallback；
6. 对合法 local trunk–apex pairs 在 bounded voxel corridor 运行 deterministic CSP-style sparse
   structural path，并把 path efficiency 按 B2.2A 融入 `TVc`；禁止全点图；
7. 对每个 accepted apex 生成固定 24-ray rotating-profile evidence，冻结 valley/gap/boundary prior；
8. 将 trunk–apex/path evidence 注入 early apex/hypothesis relation graph，只形成可逆的
   provisional hypothesis groups；
9. 所有独立 apex identities、可选 trunk anchors 及其 provisional groups 在 XY cell graph 中
   同步扩张，保留 primary/secondary/competing claims；trunk anchor 只增强 seed evidence，不直接
   产生 cell ownership；
10. 由 density/HAG/gap/topology/rival pressure + profile-confirmed conjunction 形成可解释 stop evidence；
11. 从多 claim responsibility regions 建立可重叠 3D candidate domains；
12. 按配置执行 overlapping height slices，拟合 measured-support boundaries；
13. 建立 adjacent-slice continuity graph 和 fragment graph，同时重新评估 trunk-to-crown vertical
    continuity 与 multi-stem/multi-apex compatibility；
14. 用完整 trunk/apex/path/profile/slice/geometry evidence 完成 relation 为 `MERGE/KEEP_SEPARATE/REVIEW`；
15. 在全部 2D、3D slice、fragment continuity 和 final relation evidence 完成后，只执行一次
    constraint-aware `FINAL UNION`，再选择每棵树 primary/secondary apices 与 optional
    primary/secondary trunks；
16. 对每个 normal point 同步比较所有合法 candidate trees；
17. 生成 `ASSIGNED/UNASSIGNED/REVIEW_REQUIRED`、tree records、summaries 和 diagnostics；
18. 分配 deterministic tree IDs；
19. 执行 point/ragged/tree/trunk/apex/association/path/profile/evidence/result digest 守恒；
20. 以 atomic directory publish 正式 result artifact；任一失败不留 partial output。

### 4.2A V4 Algorithm Version 1 Deterministic Scoring Specification

本节是实现权威，不把 normalization、权重或 decision semantics 留给实现者。Algorithm-V1
固定使用 binary64 计算；所有输入先按 `original_index`、cell ID、apex canonical key 或 edge
canonical key 排序，固定通道顺序并用 `math.fsum` reduction。只在 artifact 序列化边界按 contract
转换 dtype。禁止 Python `hash()`、set/dict iteration order、并行完成顺序或 source row order 参与
评分。所有 normalized values 在组合前 clamp，因而分数 bounded；只做局部 bucket、路径摘要和
fixed-length vector 运算，禁止 dense point/apex matrix。

令以下五个已经存在的、finite 且严格大于零的物理 scale 为：

```text
Lxy = xy_cell_size_m
La  = apex_neighborhood_radius_m
Ld  = density_trend_window_m
Ls  = slice_thickness_m
Lr  = relation_neighborhood_radius_m
```

Algorithm-V1 公共函数精确定义为：

```text
c01(x)        = min(1.0, max(0.0, x))
near(d, L)    = 1.0 - c01(abs(d) / L)
similar(d, L) = 1.0 - c01(abs(d) / L)
signed(q)     = 2.0 * c01(q) - 1.0
clamp11(x)    = min(1.0, max(-1.0, x))
rho_floor     = 1.0 / (Lxy * Lxy) points/m^2
safe_ratio(a, b) = c01(a / max(b, rho_floor))

Ltr_xy = max(2.0 * Lxy, 0.35 * La)
Ltr_z  = max(2.0 * Ls, La)
Lta    = max(La, 2.0 * Lxy)

Lpath_xy = max(Ltr_xy, 0.5 * La)
Lprofile = Lr

STRUCTURAL_PATH_DOWNWARD_WEIGHT = 0.25
STRUCTURAL_PATH_DENSITY_WEIGHT  = 0.25

ROTATING_PROFILE_RAY_COUNT      = 24
ROTATING_PROFILE_STEP_M         = Lxy
ROTATING_PROFILE_HALF_WIDTH_M   = 0.5 * Lxy
ROTATING_PROFILE_VALLEY_MIN     = 0.65
ROTATING_PROFILE_DROP_MIN       = 0.25
ROTATING_PROFILE_REBOUND_MIN    = 0.25
ROTATING_PROFILE_DENSITY_RATIO  = 0.65
```

`near/similar` 的 physical boundary vector 固定为 `d={0,L/2,L,2L}` ->
`{1,0.5,0,0}`；`safe_ratio` 的 boundary vector固定为 `a={0,b/2,b,2b}` 且
`b>=rho_floor` -> `{0,0.5,1,1}`。`-0.0` 在 contract boundary canonicalize 为 `+0.0`。

#### A. Responsibility claim score

对 hypothesis/apex `h` 到 occupied XY cell `c` 的一条 canonical predecessor path，八个通道均
为 support polarity，范围 `[0,1]`，值越大越支持该 claim：

| 通道 | physical input / units | Algorithm-V1 normalization | required / missing semantics | weight |
|---|---|---|---|---:|
| apex support `A` | apex prominence `m`；apex-cell 与 `La` 圆邻域 median density `points/m^2` | `0.60*c01(prominence/La) + 0.40*safe_ratio(rho_apex,rho_apex_neighborhood)` | required；accepted apex 缺值使该 apex suppressed 并把受影响 population 置 review；nonfinite hard fail | 0.16 |
| path bottleneck `B` | path 各 cell HAG q75 与 apex HAG q75，`m` | non-seed 为 `min_path c01(hag_q75_cell/max(hag_q75_apex,Ls))`；seed path 明确定义为 `1` | required；路径摘要缺失为 localized review，nonfinite hard fail | 0.16 |
| local density trend `D` | cell 前向 `Ld` 窗口 median density 与上一内侧 `Ld` 窗口 median density，`points/m^2` | non-seed 为 `safe_ratio(rho_forward,rho_inner)`；seed 明确定义为 `1` | required；窗口不足但可精确定位时 review，nonfinite hard fail | 0.14 |
| HAG continuity `H` | path 相邻 cell HAG q75 delta，`m` | `min_path similar(delta_hag,max(Ls,Lxy))`；seed 为 `1` | required；缺失 localized review，nonfinite hard fail | 0.14 |
| distance decay `R` | apex 到 cell 的 canonical path length，`m` | `near(path_length,Lr)` | required；缺失/nonfinite hard fail；仅为 prior，不能单独 stop | 0.10 |
| crown connectivity `C` | path 每一步在 measured occupied 8-neighbour graph 上是否有支持，dimensionless | `supported_path_edges / evaluated_path_edges`；seed 为 `1` | required；graph 未完整求值 hard fail；不可达为 `0` 并触发 hard stop | 0.10 |
| empty-gap support `G` | path 上最长 consecutive unsupported run，`m` | `near(longest_empty_run,Ld)` | required；未测得不是空 gap，无法区分时 localized review；nonfinite hard fail | 0.10 |
| neighbouring pressure `P` | 当前与 best rival 的 pre-pressure score difference，dimensionless | 见下方 deterministic two-pass；无合法 rival 是观测到的 `P=1`，不是 missing | required；rival inventory 不完整 hard fail | 0.10 |

先按固定顺序计算不含 pressure 的：

```text
R0(h,c) = fsum(0.16*A, 0.16*B, 0.14*D, 0.14*H,
                 0.10*R, 0.10*C, 0.10*G) / 0.90
P(h,c)  = 1.0                                                   if no legal rival
          c01(0.5 + (R0(h,c)-max_rival_R0(c)) / (2.0*0.08))    otherwise
responsibility_score = fsum(0.16*A,0.16*B,0.14*D,0.14*H,
                            0.10*R,0.10*C,0.10*G,0.10*P)
```

`responsibility_score` 范围严格为 `[0,1]`。Algorithm-V1 常量为
`RESPONSIBILITY_MIN=0.55`、`RESPONSIBILITY_REVIEW_MARGIN=0.08`，不在 scene config 中可调。
每个 cell 将所有完整 claims 按 `(score desc, hypothesis canonical key asc)` 排序；best 必须
`>=0.55`。secondary 仅在其自身 `>=0.55` 且 `best-secondary <=0.08` 时保留，并触发该 cell 的
responsibility ambiguity/review evidence；差值 `>0.08` 才是 clear best。

stop 先于 score eligibility 执行，且记录 bit flags：measured occupied graph 不可达、
`longest_empty_run >= Ld`、连续两个完整 `Ld` 窗口同时满足 `D<0.35 AND H<0.35`，或
4.4A 的 `profile_confirmed_stop` 是 hard transition stop；这些 stop 不能被高总分抵消。
`PROFILE_NOT_AVAILABLE` 不能 hard-stop，只加 diagnostic/quality。无 hard stop 时，
score `<0.55` 停止该 claim，
`>=0.55` 才可传播；rival clear win 只通过 `P` 和 best/secondary margin 生效，不另加顺序相关
shortcut。单次空 cell、单次密度/HAG 波动或仅达到固定半径不构成 hard stop。

#### B. Tree relation MERGE / KEEP_SEPARATE score

每条 canonical apex/hypothesis edge 产生八个 signed evidence `e_i in [-1,1]`；正值支持
`MERGE`，负值支持 `KEEP_SEPARATE`，零为 neutral：

| 通道 | physical input / units | exact signed normalization | required / missing semantics | weight |
|---|---|---|---|---:|
| `horizontal_distance` | apex XY distance `m` | `1 - 2*c01(d_xy/Lr)` | required；缺 measurement localized edge REVIEW；nonfinite hard fail | 0.14 |
| `height_difference` | apex HAG difference `m` | `1 - 2*c01(abs(delta_hag)/La)` | required；缺 measurement REVIEW；nonfinite hard fail | 0.10 |
| `saddle_drop` | lower apex HAG minus measured path saddle HAG，clamped `>=0 m` | `1 - 2*c01(saddle_drop/La)` | required；无完整 saddle path REVIEW；nonfinite hard fail | 0.16 |
| `xy_crown_connectivity` | two responsibility regions 间 measured occupied boundary-link fraction | `signed(linked_boundary_edges/evaluated_boundary_edges)` | required；零 evaluated edges 为 `-1` only when separation was completely measured，否则 REVIEW | 0.14 |
| `empty_gap_evidence` | regions 间最短 measured unsupported run `m` | `1 - 2*c01(gap/Ld)` | required；occluded/unknown 不是 gap，记 REVIEW；nonfinite hard fail | 0.14 |
| `slice_continuity` | jointly supported adjacent-slice links / all evaluated links | `signed(continuous_links/evaluated_links)` | final required；early stage 唯一允许 `NOT_AVAILABLE -> e=0`；final 缺失 REVIEW | 0.16 |
| `local_point_geometry` | `abs` principal-direction cosine 与对称 density ratio，dimensionless | `q=0.5*abs(dot(u,v))+0.5*min(safe_ratio(rho_a,rho_b),safe_ratio(rho_b,rho_a)); e=signed(q)` | final required；局部支持不足 REVIEW；nonfinite hard fail | 0.10 |
| `optional_trunk_evidence` | 由 B2 的 trunk–apex association graph 派生的 shared-trunk vs distinct-exclusive-trunks evidence | `e_trunk = clamp11(shared_trunk_support - distinct_trunk_support)`，精确定义见 B2 | optional；无 trunk / 无可靠 association 为 `NOT_OBSERVED -> e=0`；**absence 永不成为负证据**；只有“两个不同 trunk 对两个 apex 均有强且互斥 association”才可产生负 evidence | 0.06 |

按上表顺序计算：

```text
merge_support    = fsum(w_i * max(e_i, 0.0) for i in fixed_channel_order)
separate_support = fsum(w_i * max(-e_i,0.0) for i in fixed_channel_order)

MERGE
  iff merge_support >= 0.45
  and merge_support - separate_support >= 0.12

KEEP_SEPARATE
  iff separate_support >= 0.45
  and separate_support - merge_support >= 0.12

otherwise REVIEW
```

两种 support 均在 `[0,1]`；`RELATION_MIN=0.45` 和
`RELATION_REVIEW_MARGIN=0.12` 是 Algorithm-V1 常量。early graph 用完全相同的公式，但
`slice_continuity=NOT_AVAILABLE/e=0`，输出名称只能是 `PROVISIONAL_MERGE`、
`PROVISIONAL_KEEP_SEPARATE` 或 `PROVISIONAL_REVIEW`，不具有 final union 权威。final graph 在
slice evidence 就绪后从原始 edge evidence 重新完整计算，不能继承 early decision 当作证据。

#### B2. Trunk candidate + Trunk–Apex association score

Trunk 与 Apex 使用同一 formal normal population 独立提取，然后在 bounded local graph 中关联。
任何 trunk evidence 都是 **optional structural anchor**：没有 trunk 不会降低 apex-only tree 的
资格，也不会触发 `KEEP_SEPARATE`、`REVIEW_REQUIRED` 或 hard fail。

##### B2.1 TrunkCandidate detection

TrunkCandidate construction itself属于 Algorithm-V1 authority，不能只冻结 `TV/TA/TS/TC` 后把
candidate inventory 留给实现者。所有步骤按 canonical binary64 + `original_index` order 执行。

Algorithm-V1 immutable construction constants/derived scales：

```text
Btr = max(Lxy, 0.5 * Ls)             # trunk HAG-bin width, metres
TRUNK_BIN_MIN_POINTS       = 3
TRUNK_COMPONENT_MIN_POINTS = 6
TRUNK_COMPONENT_MIN_BINS   = 3
TRUNK_CORE_RADIUS          = 0.5 * Ltr_xy
PCA_REL_GAP_MIN            = 1e-6
PCA_ABS_EPS                = 1e-12
```

这些不是 scene config，不能按 filename/source 改写。

**Canonical HAG binning**

- HAG bin origin 永远为 `0.0 m`；
- formal normal `hag_m` 必须 finite 且已由 FRV clamp；负 HAG 若违反 handoff contract 则 hard fail；
- `hag_bin_id = floor(hag_m / Btr)`；
- bin 是半开区间 `[k*Btr, (k+1)*Btr)`；exact upper boundary 属于下一 bin；
- 同一 `(xy_cell_id, hag_bin_id)` 内 points 先按 `original_index asc` canonicalize。

**Canonical support voxel / occupied-bin rule**

一个 `(xy_cell_id, hag_bin_id)` support voxel 只有在：

```text
measured normal point_count >= TRUNK_BIN_MIN_POINTS
```

时为 `OCCUPIED`；1–2 points 为 `SPARSE_OBSERVED` diagnostic，不进入 trunk component；
0 points 不是“观测到空 trunk”，只是无 support voxel，不自动成为 negative evidence。

**Canonical raw component graph**

只在 `OCCUPIED` support voxels 上建 undirected local graph。两个 voxels `u,v` 相连当且仅当：

```text
abs(ix_u - ix_v) <= 1
abs(iy_u - iy_v) <= 1
abs(k_u  - k_v ) <= 1
and not all three deltas are zero
and xy_distance(voxel_centroid_u, voxel_centroid_v) <= Ltr_xy
```

禁止全 voxel dense adjacency。Connected components 用 canonical BFS/union-find 求解，但最终
component identity 不使用 discovery order：每个 raw component 的 key 固定为

```text
(min supporting original_index,
 min hag_bin_id,
 min xy_cell_id,
 supporting original_index digest)
```

并按该 key 排序。

raw component 若：

```text
point_count < TRUNK_COMPONENT_MIN_POINTS
or distinct occupied HAG bins < TRUNK_COMPONENT_MIN_BINS
```

则只产生 `SUPPRESSED_INSUFFICIENT_SUPPORT` diagnostic，不进入 PCA。

**Canonical first-pass PCA**

对 raw component 的全部 measured points，按 `original_index asc` 读取 binary64 XYZ，使用
`math.fsum`/fixed-order reduction 构造 centroid 与对称 `3x3` covariance。对 covariance 做
symmetric eigendecomposition，令：

```text
lambda1 >= lambda2 >= lambda3 >= 0
```

其中小于 `PCA_ABS_EPS` 的负舍入值 clamp 为 0；更小的真实负值/非finite hard fail。

如果：

```text
lambda1 <= PCA_ABS_EPS
or (lambda1 - lambda2) / max(lambda1, PCA_ABS_EPS) < PCA_REL_GAP_MIN
```

则 principal axis 不可辨识，candidate 记录
`PCA_DEGENERATE -> WEAK/NOT_AVAILABLE`，不得任选一个特征向量继续。

principal axis 的 sign canonicalize：

```text
if z < 0: multiply by -1
if z == 0:
    choose sign so first non-zero component among (x, y) is positive
if x == y == z == 0:
    PCA_DEGENERATE
```

**Canonical core supporting-point set**

first-pass axis 确定后，对 raw component 每个 measured point 计算到无限 fitted axis 的
perpendicular distance。core set 精确为：

```text
distance_to_axis <= TRUNK_CORE_RADIUS
```

边界 `==` 包含。core points 按 `original_index asc`。

core set 必须同时满足：

```text
core point_count >= TRUNK_COMPONENT_MIN_POINTS
core distinct occupied HAG bins >= TRUNK_COMPONENT_MIN_BINS
```

否则为 `WEAK_CORE_SUPPORT`，不得成为 accepted trunk。

然后只对 core set 按上面的相同 PCA 规则 **恰好 refit 一次**；不迭代到收敛，避免实现差异。
refit 退化则 `PCA_DEGENERATE -> WEAK/NOT_AVAILABLE`。

**Canonical per-bin summaries**

只对 core set 计算：

- `core_hag_min/max`；
- `observed_vertical_span = core_hag_max - core_hag_min`；
- `k_min/k_max`；
- `evaluated_bins = k_max - k_min + 1`；
- `occupied_required_bins` = `[k_min,k_max]` 中 core point_count
  `>= TRUNK_BIN_MIN_POINTS` 的 bin 数；
- 每个 occupied core bin 的 XY centroid；
- 在该 bin 的 median HAG 处求 refit axis 的 XY intersection；
- `max_bin_centroid_xy_drift` = 所有 occupied core bins 的
  `distance(bin_xy_centroid, axis_xy_at_bin_median_hag)` 最大值；
- `TC = occupied_required_bins / evaluated_bins`。由于 accepted/weak candidate 前已保证
  `evaluated_bins >= 1`，这里不存在零分母；若内部 invariant 仍得到 0，则 hard fail。

每个候选计算四个 `[0,1]` support channel：

| 通道 | exact Algorithm-V1 definition | weight |
|---|---|---:|
| vertical span `TV` | `c01(observed_vertical_span / Ltr_z)` | 0.35 |
| axis verticality `TA` | `vz=abs(dot(refit_principal_axis,[0,0,1]))`；`c01((vz-0.50)/0.50)` | 0.30 |
| XY centroid stability `TS` | `near(max_bin_centroid_xy_drift, Ltr_xy)` | 0.20 |
| vertical bin continuity `TC` | `occupied_required_bins / evaluated_bins` | 0.15 |

```text
trunk_candidate_score = fsum(0.35*TV, 0.30*TA, 0.20*TS, 0.15*TC)
TRUNK_CANDIDATE_MIN   = 0.62
TRUNK_WEAK_MIN        = 0.52
```

- `score >= 0.62`：accepted `TrunkCandidate`；
- `0.52 <= score < 0.62`：保留 weak diagnostic，但不能产生 positive/negative relation evidence；
- `<0.52`：suppressed diagnostic；
- required input nonfinite / inventory incomplete：按影响范围 localized review；不能精确隔离时 hard fail；
- candidate representative `original_index` 只从 final core set 选取，排序 key 为
  `(distance_to_refit_axis asc, original_index asc)`，取第一项；
- accepted/weak candidate IDs 最终按
  `(representative_original_index asc, core_member_digest asc)` 分配；
- raw-component traversal、PCA library eigenvector sign、block row order、thread completion order
  均不得影响 candidate inventory / representative / candidate ID。

##### B2.2 Trunk–Apex association

Trunk–Apex graph 是 **sparse materialized edge graph + total deterministic lookup semantics**。
只对 horizontal distance `<= Lr` 的 pair materialize association edge；但 B2.3 查询任意
`(trunk, apex)` 时必须通过本节的 typed lookup，而不能用“dict 中没有 key -> 0”。

每个 materialized pair 的五个 evidence channel 都有独立 availability：

```text
OBSERVED
NOT_AVAILABLE
```

只有五通道全部 `OBSERVED` 时才计算 `trunk_apex_score`。任一 required channel
`NOT_AVAILABLE`：

```text
association_score_available = false
association_score = 0.0        # canonical serialization placeholder ONLY
association_state = REVIEW_ASSOCIATION
quality += ASSOCIATION_EVIDENCE_INCOMPLETE
```

该 placeholder **禁止**进入 B2.3 的 strong/exclusive 数学函数。

五通道 exact semantics：

| 通道 | exact Algorithm-V1 definition | availability / zero-denominator semantics | weight |
|---|---|---|---:|
| XY alignment `TX` | trunk refit axis 在 apex HAG 高度的 XY 外推点与 apex XY 距离 `d_xy`；`near(d_xy,Lta)` | accepted trunk axis + finite apex HAG 时 `OBSERVED`；axis/HAG 非finite hard fail | 0.28 |
| direction alignment `TD` | trunk representative -> apex 向量与向上 canonical trunk axis 的 cosine，负值 clamp 0，再 `c01(cosine)` | 向量 norm `<=PCA_ABS_EPS` 为 `NOT_AVAILABLE`；否则 `OBSERVED` | 0.20 |
| vertical continuity `TVc` | 先按本行计算 `TVc_bins`：用 B2.1 同一 `Btr`，从 trunk `k_max+1` 到 `floor(apex_hag/Btr)-1` 的 intermediate bins；每 bin 若存在与前一支持 bin 通过 canonical 26-neighbour voxel edge 连通的 OCCUPIED support voxel则为 supported；随后按 B2.2A 与 `structural_path_efficiency` 以 `0.50/0.50` 融合成最终 `TVc` | `evaluated_bins>0` 时计算 `TVc_bins`；`evaluated_bins==0` 时直接相邻=`TVc_bins=1`，明确不相邻=`0`；任一 required local inventory 或 structural path 为 `NOT_AVAILABLE` 时最终 `TVc` 也是 `NOT_AVAILABLE` | 0.22 |
| gap support `TG` | 在上述同一 intermediate-bin corridor 上统计最长连续 **observed unsupported** bin run，`near(gap_length,Ld)` | 只有每个 evaluated bin 的 support status 均由完整 local voxel inventory 判定时 `OBSERVED`；任一 bin 因 resource/local-inventory 缺失无法判定则 `NOT_AVAILABLE`；`evaluated_bins==0` 时 `TG=1` | 0.18 |
| local crown connectivity `TR` | trunk upper support voxels 到 apex `La` neighborhood occupied component 的 canonical boundary links / evaluated candidate boundary links | denominator `>0` 正常；denominator `==0` 且两侧 support sets 都完整存在时为 `OBSERVED TR=0`；任一 support set / boundary inventory 不完整为 `NOT_AVAILABLE` | 0.12 |

**Occlusion/absence rule**

Algorithm-V1 不从“没有 normal point”自动推断 sensor-visible empty space。只有在 local voxel
inventory 完整、对应 corridor/boundary evaluation 已执行时，缺 supporting voxel 才是
`observed unsupported`。任何因资源、局部构造失败、缺 support inventory 而无法区分
“无点”与“未评估”的情况必须 `NOT_AVAILABLE`，不得当 0 分或 exclusivity。

apex 必须位于 trunk representative HAG 之上；若 `apex_hag <= trunk_representative_hag` 且两者
finite，则这是完整几何下的 `OBSERVED_UNSUPPORTED_GEOMETRY`：

```text
association_score_available = true
trunk_apex_score = 0.0
association_state = UNSUPPORTED_ASSOCIATION
```

完整五通道时：

```text
trunk_apex_score = fsum(0.28*TX, 0.20*TD, 0.22*TVc, 0.18*TG, 0.12*TR)

TRUNK_APEX_STRONG_MIN = 0.65
TRUNK_APEX_REVIEW_MIN = 0.55
```

- `score >= 0.65`：`STRONG_ASSOCIATION`；
- `0.55 <= score < 0.65`：`REVIEW_ASSOCIATION`；
- `<0.55`：`UNSUPPORTED_ASSOCIATION`；
- 不做 Hungarian/greedy one-to-one assignment；
- 允许 `1 trunk -> N apex`、`N trunk -> 1 apex`；
- `0 trunk -> apex` 继续走 apex-only hypothesis；
- trunk 没有兼容 apex 时只记录 `UNMATCHED_TRUNK`，首版不允许 trunk-only 直接生成 final Tree ID。

**Total cross-score lookup**

B2.3 不得直接访问 sparse edge dict。统一函数语义：

```text
lookup(t, a) -> one of

EVALUATED(score, state)
    if d_xy <= Lr and materialized edge has all five channels OBSERVED

NOT_AVAILABLE
    if d_xy <= Lr but any required channel is NOT_AVAILABLE

OBSERVED_OUT_OF_RADIUS
    if finite measured d_xy > Lr
```

`OBSERVED_OUT_OF_RADIUS` 是由完整 XY geometry 明确得到的“不在 Algorithm-V1 association
domain”，不是 missing；它可以作为 exclusivity evidence。`NOT_AVAILABLE` 永远不能当 0 分、
unsupported 或 out-of-radius。

##### B2.2A CSP-style trunk-rooted sparse structural path evidence

本节把“树干到冠层点的比较最短路径”思想改造成适用于当前 full-resolution/CPU-first V4 的
**局部 sparse structural path evidence**。它不是论文 TLS point graph 的直接复制，也禁止构建
全场景 point graph。

对每个已 materialize 的 accepted `TrunkCandidate t` 与 `ApexCandidate a` pair，只在以下
canonical local domain 建图：

```text
horizontal distance from voxel center
to the XY segment (trunk representative -> apex)
<= Lpath_xy

AND

voxel HAG in
[min(trunk_core_hag_min, apex_hag) - Btr,
 max(trunk_core_hag_max, apex_hag) + Btr]
```

graph node 只能是 B2.1 已验证的 measured occupied support voxel；node key 固定为：

```text
(xy_cell_id, hag_bin_id, minimum_supporting_original_index)
```

两个 node 只有满足 canonical 26-neighbour adjacency 时才连边。禁止为了“补路径”创建 synthetic
voxel 或跨越未知 occlusion 建虚边。

每条 edge `u -> v` 的 binary64 cost：

```text
step = euclidean_distance(center_u, center_v)

downward_fraction =
    c01(max(0, hag_u - hag_v) / max(Btr, Ls))

density_similarity =
    min(safe_ratio(rho_u, rho_v),
        safe_ratio(rho_v, rho_u))

edge_cost =
    step * (
        1.0
        + STRUCTURAL_PATH_DOWNWARD_WEIGHT * downward_fraction
        + STRUCTURAL_PATH_DENSITY_WEIGHT  * (1.0 - density_similarity)
    )
```

cost 必须 finite 且 `>= step > 0`。从 trunk upper-core support nodes 到 apex-neighbourhood support
nodes 运行 deterministic Dijkstra：

```text
heap key =
(total_cost,
 path_hop_count,
 predecessor_node_key,
 current_node_key)
```

相同 total cost 的 tie 不能依赖 insertion/discovery order。

若存在 path：

```text
straight_length =
    euclidean_distance(trunk_upper_anchor_xyz, apex_xyz)

weighted_path_length =
    sum(edge_cost along canonical shortest path)

structural_path_efficiency =
    c01(straight_length /
        max(weighted_path_length, straight_length, Ls))
```

若 `straight_length <= Ls`，只要 trunk upper support 与 apex-neighbourhood support 在 canonical
graph 中同一 node/component，定义 `structural_path_efficiency=1`；否则按上式。

若 local graph inventory 完整但 trunk 与 apex 不可达：

```text
structural_path_availability = OBSERVED
structural_path_efficiency   = 0.0
```

若 graph/resource/local inventory 不完整而无法区分“不连通”和“未评估”：

```text
structural_path_availability = NOT_AVAILABLE
```

禁止把 `NOT_AVAILABLE` 当 0。

**与既有五通道 association 的融合方式固定，不新增第六个 association weight：**

原 `TVc` 先计算 B2.2 的 vertical-bin continuity 为 `TVc_bins`。若 structural path
`OBSERVED`：

```text
TVc =
    0.50 * TVc_bins
    + 0.50 * structural_path_efficiency
```

若 structural path `NOT_AVAILABLE`：

```text
TVc availability = NOT_AVAILABLE
```

因此 incomplete structural path 不会偷偷降分。

`TG` 继续表示 measured gap support，`TR` 继续表示 crown endpoint connectivity；三者是不同
证据，禁止将同一 shortest-path number 重复计入三个通道。

**CSP-style competitor evidence**

当一个 crown/apex 同时存在两个以上 accepted trunk strong associations 时，保存每个 trunk 的：

```text
weighted_path_length
structural_path_efficiency
```

但 early 阶段只作为 association diagnostic 和 `shared/distinct trunk` evidence 的组成来源，
不得用“最短 trunk path winner”直接把整棵冠层一刀切。最终 point assignment 仍由完整
responsibility + slices + fragment continuity + relation + assignment scorer 决定。

这保留“树干到冠层结构路径”的价值，同时避免 TLS 场景中树干完全可见这一前提被错误移植到
机载/走廊 LiDAR。

##### B2.3 Relation 中的 signed trunk evidence

对待比较的两个 apex/final-hypothesis endpoints `a,b`，所有 trunk lookup 先走 B2.2 的 total
typed lookup。Algorithm-V1 不再使用旧的 `c01((s-0.65)/0.35)`，因为 inclusive strong threshold
`0.65` 必须产生 **正** strong support。

定义：

```text
strong01(lookup) =
    0.5 + 0.5*c01((score - 0.65) / 0.35)
        if lookup == EVALUATED(score, STRONG_ASSOCIATION)
    0.0
        if lookup == EVALUATED(score, REVIEW_ASSOCIATION)
        or lookup == EVALUATED(score, UNSUPPORTED_ASSOCIATION)
        or lookup == OBSERVED_OUT_OF_RADIUS
    NOT_AVAILABLE
        if lookup == NOT_AVAILABLE

exclusive01(lookup) =
    1.0
        if lookup == OBSERVED_OUT_OF_RADIUS
    1.0 - c01(score / 0.55)
        if lookup == EVALUATED(score, UNSUPPORTED_ASSOCIATION)
    0.0
        if lookup == EVALUATED(score, REVIEW_ASSOCIATION)
        or lookup == EVALUATED(score, STRONG_ASSOCIATION)
    NOT_AVAILABLE
        if lookup == NOT_AVAILABLE
```

因此 strong classification 的 exact reachable boundary：

```text
score = 0.65 -> strong01 = 0.5   (>0)
score = 1.00 -> strong01 = 1.0
```

review boundary `score=0.55` 不产生 strong 或 exclusivity support：

```text
score = 0.55 -> REVIEW_ASSOCIATION
strong01 = 0
exclusive01 = 0
```

shared support：

```text
shared_trunk_support =
    max over accepted trunk t where both lookups are not NOT_AVAILABLE:
        min(strong01(lookup(t,a)),
            strong01(lookup(t,b)))
```

若没有这样的 trunk，`shared_trunk_support=0`。

distinct support：

```text
distinct_trunk_support =
    max over distinct accepted trunks t1 != t2
    where all four lookups below are not NOT_AVAILABLE:
        min(strong01(lookup(t1,a)),
            strong01(lookup(t2,b)),
            exclusive01(lookup(t1,b)),
            exclusive01(lookup(t2,a)))
```

**任何一个 cross lookup 为 `NOT_AVAILABLE`，该 `(t1,t2)` distinct candidate pair 必须从 max
候选集中排除；禁止把 missing cross edge 当 score=0 / exclusive=1。**

最终：

```text
e_trunk =
    clamp11(shared_trunk_support - distinct_trunk_support)
```

若所有可用 trunk evidence 都缺失或不存在：

```text
shared_trunk_support = 0
distinct_trunk_support = 0
e_trunk = 0
trunk_relation_availability = NOT_AVAILABLE_OR_NOT_OBSERVED
```

这保持 trunk optional/neutral。`NOT_AVAILABLE` 不自行强制最终 relation REVIEW，也不能产生
MERGE/KEEP_SEPARATE；它必须记录 quality flag，使 Bundle B 可见。其他 7 个完整 relation
channels 仍可独立决定 final relation。

如果一个 trunk pair 的 direct associations 是 strong，但决定 distinct exclusivity 所需的
cross lookup 为 `NOT_AVAILABLE`，只允许：

```text
distinct_trunk_support = no contribution from that pair
```

不能推断 exclusive。这样不会产生 false `KEEP_SEPARATE`。

语义保证：

- 同一 trunk 对两个 apex 都是 strong -> positive MERGE support，且 threshold `0.65` 本身已为正；
- 两个不同 trunk 分别 strong 对应两个 apex，且两个 cross lookup 都是
  `OBSERVED_OUT_OF_RADIUS` 或完整 `UNSUPPORTED_ASSOCIATION` -> negative KEEP_SEPARATE support；
- multi-stem、review-band cross-link 或 evidence conflict -> neutral/review-quality evidence；
- **trunk absence / NOT_AVAILABLE 永远不等于负 evidence**。

`1 trunk <-> 1 apex` 是最高置信的常见 seed case，但不是强制 cardinality。trunk association
只增强/约束 provisional hypothesis search；最终 tree relation 仍需完整 2D + 3D
slice/fragment evidence，且只有 4.7 的单次 `FINAL UNION` 有最终权威。

#### C. Final point-to-tree assignment score

每个 point/tree candidate claim 使用 signed `e_i in [-1,1]`；`+1/-1/0` 分别是支持、反对、
neutral。所有 candidate trees 的 base score 先完整计算，之后才计算 competitor channel：

| 通道 | physical input / units | exact signed normalization | required / missing semantics | weight |
|---|---|---|---|---:|
| slice measured support | point 是否为链接到 tree 的 measured fragment member | member `e=+1`；完整求值且不是 member `e=-1` | required；fragment inventory incomplete -> point REVIEW 或不能隔离时 hard fail | 0.18 |
| adjacent-slice continuity | supporting adjacent links / evaluated adjacent links | `signed(fraction)`；合法 single-slice tree 无 adjacent edge时是 observed neutral `e=0` | required；缺求值 REVIEW | 0.14 |
| XY prior | 对该 tree 的 responsibility score | `signed(responsibility_score)` | required；没有合法 claim 是 measured `e=-1`；claim inventory incomplete hard fail | 0.12 |
| apex support | apex channel `A` 与 point-to-apex XY distance `m` | `e=signed(0.5*A+0.5*near(d_xy,Lr))` | required；缺值 REVIEW；nonfinite hard fail | 0.10 |
| local HAG/density | point-vs-fragment median HAG delta `m` 与双向 local density ratio | `q=0.5*similar(delta_hag,Ls)+0.5*min(safe_ratio(rho_p,rho_f),safe_ratio(rho_f,rho_p)); e=signed(q)` | required；support 不足 REVIEW；nonfinite hard fail | 0.12 |
| structure direction | point/local 与 fragment principal direction 的 `abs` cosine | `e=signed(abs(dot(u,v)))` | optional；PCA support 不足严格 `NOT_AVAILABLE -> e=0` | 0.10 |
| gap/occlusion | point 到 fragment core 的 measured unsupported run `m` | measured 时 `e=1-2*c01(gap/Ld)`；只有 occlusion 且无法测 gap 时 `NOT_AVAILABLE -> e=0` 并加 quality flag | gap measurement required when visible；unknown occlusion neutral并强制 REVIEW flag，nonfinite hard fail | 0.08 |
| competitor margin | 本 claim base minus best other-tree base | `e=clamp11((base-best_other_base)/0.10)`；无 other candidate 是 observed `e=+1` | required；candidate inventory incomplete hard fail | 0.10 |
| segmentation quality | V4 segmentation quality severity | no flags `e=+1`；only optional-insufficient flag `e=0`；任一 review-required flag `e=-1` 并 mandatory REVIEW | required；unknown flag/nonfinite hard fail | 0.06 |

`base` 使用除 competitor 外的固定通道，除以其权重和 `0.90` 后映射到 `[0,1]`：

```text
base = 0.5 + 0.5 * fsum(w_i*e_i, i != competitor_margin) / 0.90
competitor_e = +1.0                                  if no other legal candidate
               clamp11((base-best_other_base)/0.10) otherwise
assignment_support = 0.5 + 0.5*fsum(w_i*e_i for all nine channels)
```

`assignment_support in [0,1]`。Algorithm-V1 常量为 `ASSIGNMENT_MIN=0.62`、
`ASSIGNMENT_REVIEW_MARGIN=0.10`。按 `(assignment_support desc, final hypothesis key asc)` 排序：

- `ASSIGNED`：无 mandatory review、best `>=0.62`，且无 second 或 `best-second >=0.10`；
- `REVIEW_REQUIRED`：required evidence 局部缺失、mandatory review flag、best `>=0.62` 但
  `best-second <0.10`，或 relation/responsibility conflict 影响该 point；
- `UNASSIGNED`：全部 candidate comparisons 完整、无 mandatory conflict 且 best `<0.62`；无
  candidate 时定义 `best=0`。边界相等时，`0.10` 是 clear margin，不进入 review。

assignment confidence 精确定义为：

```text
ASSIGNED: c01(0.5*best + 0.5*c01((best-second)/0.10)); no second 时 margin term=1
UNASSIGNED: c01(1.0 - best/0.62)
REVIEW_REQUIRED: 0.0
```

required raw XYZ/HAG、identity、inventory 或计算产生 nonfinite 属于运行级 hard fail；仅当缺失
evidence 能精确隔离到 edge/point 且 input contract 完整时才是 localized REVIEW。optional missing
只能使用上表显式 neutral/`NOT_AVAILABLE`，禁止 `dict.get(...,0)`、NaN replacement、truthiness
或 Python 默认参数偷偷转换成支持/反对。

Algorithm-V1 的 assignment decision precedence 冻结为：先检查 hard-fail 条件，再检查所有
mandatory review 条件，最后才允许依据 support/margin 进入 `ASSIGNED` 或 `UNASSIGNED`。因此
`UNASSIGNED` 永远要求 complete comparison 且不存在 mandatory review/conflict；任何
`segmentation_quality=-1`、review-required flag 或其他 mandatory review 条件都优先得到
`REVIEW_REQUIRED`，不能被低 `assignment_support` 覆盖。

#### 固定 score vectors

测试必须同时覆盖 physical normalizer boundary vectors 和下列 scorer vectors；binary64 期望值
以 `abs<=1e-12` 比较，decision 必须 exact enum equality：

| scorer input | expected score/support | expected decision |
|---|---|---|
| responsibility first seven=`1`，no rival (`P=1`) | `1.0` | eligible clear best |
| responsibility first seven=`0.5`，equal rival (`P=0.5`) | `0.5` | stop: below `0.55` |
| responsibility best=`0.70`, second=`0.64` | delta `0.06` | secondary retained / review evidence |
| responsibility best=`0.70`, second=`0.60` | delta `0.10` | best only |
| relation all eight `e=+1` | merge=`1.0`, separate=`0.0` | `MERGE` |
| relation first seven `e=-1`, trunk=`NOT_OBSERVED/e=0` | merge=`0.0`, separate=`0.94` | `KEEP_SEPARATE`；absence 本身仍 neutral |
| relation all `e=0` | merge=`0.0`, separate=`0.0` | `REVIEW` |
| trunk candidate `TV=TA=TS=TC=1` | `trunk_candidate_score=1.0` | accepted `TrunkCandidate` |
| trunk candidate all four=`0.5` | `trunk_candidate_score=0.5` | suppressed diagnostic |
| trunk–apex score exactly `0.55` | `REVIEW_ASSOCIATION`; `strong01=0`; `exclusive01=0` | threshold review is neutral to relation trunk channel |
| trunk–apex score exactly `0.65` | `STRONG_ASSOCIATION`; `strong01=0.5` | inclusive strong threshold gives positive support |
| trunk–apex score `math.nextafter(0.65,+inf)` = `0.65000000000000013` | `strong01=0.50000000000000011` | strictly greater than `0.5` |
| trunk–apex all five=`1` | `trunk_apex_score=1.0`; `strong01=1.0` | `STRONG_ASSOCIATION` |
| in-radius association required channel missing | `lookup=NOT_AVAILABLE` | score placeholder ignored; no strong/exclusive support |
| cross pair finite `d_xy > Lr` | `lookup=OBSERVED_OUT_OF_RADIUS`; `exclusive01=1.0` | observed geometric exclusivity |
| in-radius cross pair score=`0.0`, all channels observed | `UNSUPPORTED_ASSOCIATION`; `exclusive01=1.0` | observed unsupported exclusivity |
| in-radius cross pair score=`0.55` | `REVIEW_ASSOCIATION`; `exclusive01=0.0` | review-band cross edge is not exclusivity |
| shared trunk two apex scores=`0.65,0.65` | `shared=0.5`, `e_trunk>0` absent stronger distinct evidence | positive merge evidence at exact threshold |
| shared trunk two apex scores=`1,1`, no distinct pair | `e_trunk=+1.0` | maximal positive relation evidence |
| two distinct trunks direct scores=`1,1`, cross lookups=`OBSERVED_OUT_OF_RADIUS` | `e_trunk=-1.0` absent shared support | maximal negative relation evidence |
| two distinct trunks direct scores=`1,1`, one cross lookup=`NOT_AVAILABLE` | that distinct pair excluded | must not create false `KEEP_SEPARATE` |
| no trunk candidates | `e_trunk=0.0` | neutral / apex-only fallback |
| assignment all nine `e=+1` | `1.0` | `ASSIGNED`, confidence `1.0` |
| assignment all nine `e=-1` | `0.0` | `REVIEW_REQUIRED`, confidence `0.0`；`segmentation_quality=-1` 触发 mandatory review |
| assignment first eight `e=-1`, `segmentation_quality=+1` | `0.06` | `UNASSIGNED`, confidence `0.9032258064516129`；complete negative evidence, no mandatory review |
| assignment all nine `e=0` | `0.5` | `UNASSIGNED`, confidence `0.1935483870967742` |
| assignment best=`0.70`, second=`0.65`, no other conflict | delta `0.05` | `REVIEW_REQUIRED`, confidence `0.0` |

### 4.3 Robust CHM、ApexCandidate 与 TrunkCandidate 联合检测

CHM 只用 handoff normal full-resolution points。每个 XY cell 至少保留 measured count、
HAG median/upper quantile、support-filtered maximum、local spacing、occupied 状态和选择
apex-support point 的 `original_index`。不得把绝对最高单点无条件当冠高；低支持极值必须
被 support rule 拒绝或标记 review。

Apex proposal 采用 V3 已验证的 circular local-neighborhood、plateau collapse 和
annulus-median prominence 原则，但输入、输出和 tie 均改为 V4 original-index 语义：

- equal-height plateau 只产生一个 proposal；
- proposal 必须绑定一个真实 measured supporting point；
- plateau 内 supporting point 按 `HAG desc, original_index asc` 选择；
- candidate ranking 至少为 `HAG desc, prominence desc, original_index asc`；
- 缺 prominence、edge-limited、sparse support 和 suppression 都保留 reason/quality；
- `ApexCandidate != Tree`，candidate ID 永不直接成为 tree ID。

`N=0` 返回合法空 result。`N>0` 但没有 valid apex 不 hard fail：即使存在 trunk candidates，
首版也不允许 trunk-only 直接生成 Tree ID；全部 normal points 在完整审计后进入
`REVIEW_REQUIRED`，tree count 为零。单一 apex candidate 仍执行 responsibility、slice 和最终
assignment，不走 special case shortcut。

Trunk candidate 与 apex proposal 从同一 canonical normal rows 中独立提取，二者任何一个的
discovery order 都不得影响另一个 candidate ID。Trunk detection 采用 4.2A-B2 的 column/HAG-bin
vertical support、PCA verticality、centroid stability 和 continuity 规则；所有 trunk supporting
members 必须保留原始 `original_index`。检测到 trunk 不能绕过 FRV membership/HAG/route，也不能
把非 normal point 拉回 segmentation。

### 4.4 Trunk–Apex association + Multi-apex/Multi-stem initial relationship

先按 4.2A-B2 建立 bounded trunk–apex association graph，再建立 apex/hypothesis relation graph；
两者都只连接 spatial-bucket 查询得到的局部 candidates，禁止 trunk×apex 或 apex×apex 全局 dense
matrix。

early apex/hypothesis edge 保留：horizontal distance、height difference、CHM saddle drop、
2D occupied connectivity、empty-gap evidence、local density/geometry，以及从 trunk–apex graph
派生的 signed `optional_trunk_evidence`。此时 slice continuity 为 `NOT_AVAILABLE`，不能假装为零
或反证。

cardinality 语义冻结：

- `1 trunk <-> 1 apex`：强 one-to-one seed，但仍为 provisional；
- `1 trunk <-> N apex`：为 multi-apex merge 提供强 positive support，但不得提前 final merge；
- `N trunk <-> 1 apex/crown`：标记 `MULTI_STEM_CANDIDATE`，不得因 trunk 数量自动拆树；
- `0 trunk <-> apex`：合法 apex-only fallback；
- `unmatched trunk`：仅 diagnostic，不产生 tree/hypothesis ownership。

**EARLY MERGE IS PROVISIONAL ONLY.** early `PROVISIONAL_MERGE` connected components 只能形成
`V4ProvisionalHypothesisGroup`，用于共享 responsibility candidate domain 和减少局部搜索空间；
它不是 tree hypothesis 的不可逆 union。直到 final relation 完成，必须逐项保留：

- every apex identity 及其独立 canonical hypothesis reference；
- every original early relation edge、八通道原始/normalized evidence 和 provisional decision；
- provisional group membership；
- 每个 cell/fragment/point 的 competing hypothesis references。

early stage 禁止永久 union apex、删除独立 apex identity、产生 final tree/tree ID，或让 provisional
group 阻止后续 slice evidence 拆分。`PROVISIONAL_KEEP_SEPARATE` 和 `PROVISIONAL_REVIEW` 同样只是
搜索约束/待决记录；final graph 必须在完整 3D evidence 上重新判定。因而 early merge + later
strong `KEEP_SEPARATE` 必须能保留两个 final hypotheses；仍冲突则为 `REVIEW`，不能由 early edge
处理顺序造成过合并。

### 4.4A Rotating-profile crown boundary prior

Rotating-profile 只对 accepted apex / active provisional hypothesis 运行。它借鉴“以树顶为中心
多方向剖面寻找冠缘与交叉谷点”的思想，但在本项目中冻结为 **XY radial evidence extractor**，
不创建 synthetic crown polygon，也不直接分配 tree membership。

对每个 apex，建立 `ROTATING_PROFILE_RAY_COUNT=24` 条 canonical rays：

```text
theta_j = 2*pi*j/24
j = 0..23
```

方向 `theta=0` 固定为 local +X；角度不依赖 PCA/树干朝向，因此不同运行不会旋转 ray inventory。

每条 ray 从 apex XY 向外到 `Lprofile=Lr`，以 `ROTATING_PROFILE_STEP_M=Lxy` 采样。第 `s` 个
sample 只汇总 perpendicular distance `<=0.5*Lxy` 的 formal normal occupied cells。每个 sample
保存：

```text
sample_index
radial_distance_m
occupied / observed-empty / NOT_AVAILABLE
HAG_q75
median_density
supporting_cell_ids
supporting_original_index_digest
```

若该 radial corridor 的 local cell inventory 不完整，sample 必须 `NOT_AVAILABLE`；禁止把未知
当 empty。

为抑制单 cell 噪声，对连续 `OBSERVED` samples 分别对 `HAG_q75` 与 density 做 canonical
3-sample median filter；端点只使用实际存在的 2/3 samples，tie 按 radial index。

对非 seed sample `j`，定义长度约为 `Ld` 的 inner/outward windows，窗口边界使用半开 radial
interval。只有 inner、candidate、outward 三部分全部 `OBSERVED` 才允许产生 valley evidence。

```text
inner_h   = median(smoothed_hag in inner window)
valley_h  = smoothed_hag[j]
outer_h   = median(smoothed_hag in outward window)

inner_rho  = median(smoothed_density in inner window)
valley_rho = smoothed_density[j]
outer_rho  = median(smoothed_density in outward window)

drop =
    c01((inner_h - valley_h) / max(La, Ls))

rebound =
    c01((outer_h - valley_h) / max(La, Ls))

density_valley =
    c01(
        1.0
        - valley_rho /
          max(min(inner_rho, outer_rho), rho_floor)
    )

profile_valley_support =
    min(drop, rebound, density_valley)
```

candidate valley 必须同时满足：

```text
drop    >= ROTATING_PROFILE_DROP_MIN       (=0.25)
rebound >= ROTATING_PROFILE_REBOUND_MIN    (=0.25)

and

valley_rho <=
ROTATING_PROFILE_DENSITY_RATIO *
min(inner_rho, outer_rho)                  (=0.65 ratio)
```

最终：

```text
PROFILE_CONFIRMED_VALLEY
    if profile_valley_support >= 0.65

PROFILE_WEAK_VALLEY
    if candidate valley but support < 0.65

PROFILE_NO_VALLEY
    if all required samples observed and no candidate valley

PROFILE_NOT_AVAILABLE
    if required sample/window inventory incomplete
```

若 ray 上存在连续 measured observed-empty run `>=Ld`，额外记录 `PROFILE_MEASURED_GAP`；
未知/occluded sample 不计入 measured gap。

**Rotating-profile 不新增 responsibility score 权重，避免重新发明第二套 ownership scorer。**
它只改变 transition authority：

```text
profile_confirmed_stop =
    PROFILE_MEASURED_GAP

    OR

    (
      PROFILE_CONFIRMED_VALLEY
      AND
      (existing local D < 0.35 OR existing local H < 0.35)
    )
```

`PROFILE_CONFIRMED_VALLEY` 单独不能 hard-stop；必须与已有 density/HAG decline 至少一项共同出现。
这样 multi-direction profile 能确认“下降 -> 谷底 -> 外侧重新升高”的冠层交界，但不会因单次局部
波动截断冠层。

如果一个 profile-confirmed boundary 正好被另一合法 hypothesis 从外侧以 clear
responsibility margin 支持，则该 boundary 是强 rival transition prior；若两侧 claim 都落在 review
margin，则保留 overlapping responsibility + review evidence，不能按 ray 顺序决定所有权。

每条 ray 的 profile evidence、valley/gap reason、boundary radial index 和 supporting cells 必须
进入 diagnostic record，供 Bundle B 画出 profile rays/valleys；profile derived point/line 只能标记
`NON_MEASURED_DIAGNOSTIC_GEOMETRY`。

### 4.5 同步二维 responsibility regions

二维 domain 是有 measured support 的 XY cells 及经配置允许的短 gap adjacency，不是固定
半径圆。所有 hypotheses 进入同一 multi-source solver：

- cell claim 读取上一轮 immutable scores，下一轮统一提交；等价 priority-queue 实现必须
  证明与 synchronous reference oracle 相同；
- stable processing key 为 `(iteration/path-cost, hypothesis canonical key, cell_id)`，不读取
  source/block row order；
- claim score 同时使用 apex support、path bottleneck、local density trend、HAG continuity、
  distance decay、crown connectivity、empty gap 和 neighbor pressure；
- 每个 cell 保留 best claim，以及 review margin 内的 secondary claims；
- distinct-height evidence 可产生合法 multi-tree overlap，不强制 XY exclusive partition；
- distance 只能作为一项 prior/tie-break，不能独立定义最终边界。

stop/transition 必须由 versioned conjunction 给出 reason：持续 density decline、持续 HAG
decline、supported saddle/valley、超过允许的 measured empty gap、topology disconnect、邻近
hypothesis 以明确 margin 占优，或 4.4A 定义的 `profile_confirmed_stop`。旋转剖面的单个局部低点、
单次 density fluctuation、一个空 cell 或固定 radius 到达不得单独停止。solver 有
max cell/edge/profile evaluation resource gate；耗尽资源 hard fail，不返回截断的“成功”区域。

输出 responsibility boundary 是 topology-checked occupied-cell union/ragged boundary vertices，
可多 ring、可与其他 hypothesis 重叠。它是 strong prior 和 candidate search domain，不是
最终 tree membership。

### 4.6 三维 slicing 和 fragment graph

每个 hypothesis 从其 primary/secondary responsibility claims 建立可重叠 candidate domain。
垂直范围来自该 domain 内 measured HAG support；只允许配置控制的 uncertainty margin。prism
inclusion 不产生 ownership。

slice schedule：

- `stride = slice_thickness_m * (1 - slice_overlap_fraction)`；
- 完整覆盖 measured HAG range，每个 candidate point 至少进入一个 slice；
- vertical extent 大于一个 thickness 时必须产生多个 overlapping slices；
- very low/broad tree 小于 thickness 时一个 slice 合法，但记录该限制；
- 不允许把一个二维 polygon 无限 extrusion 直接当 tree body。

每个 slice 在局部 XY buckets 上形成 measured fragments，并保留 point count、HAG bounds、
centroid、bounds、area、density、boundary rings、supporting apex/hypothesis claims、quality 和
member-original-index digest。边界是 occupied-cell union，不创建 synthetic member points。

相邻 slice graph 只连接相邻/overlap slice 的局部 fragments，比较 boundary overlap、centroid
shift、extent ratio、density continuity、measured gap、branch/taper direction、responsibility
claims，以及已匹配 trunk upper support 到 lower/mid crown fragments 的连续性。明显横向 split
产生 keep-separate evidence；持续连接、合理 taper/branching 产生 merge support；孤立跳变、空
slice 或证据冲突产生 review。

Trunk evidence 在此阶段必须重新验证而不是照搬 early association：对 accepted trunk-apex
association，使用 B2.2A 已保存的 sparse structural path 作为 early reference，但 final revalidation
必须落到真实 slice/fragment graph。如果同一 trunk 的 upper-column support 与其关联 apex/crown
在 3D fragment graph 中不能保持连续，则该 association 降为 review/unsupported；如果两个独立
trunk 分别持续连接两个独立 fragment，则增强 keep-separate；如果多个 trunk 在低层汇入同一连续
crown structure，则允许 multi-stem single-tree，而不是按 trunk count 强制拆分。early Dijkstra
path 不是 final ownership path。

最终 conflict graph 在 fragment/hypothesis level 建立，绝不建立全点 dense adjacency。edge
evidence vector 至少包含：

```text
horizontal_distance
height_difference
saddle_drop
xy_crown_connectivity
empty_gap_evidence
slice_continuity
local_point_geometry
optional_trunk_evidence
```

trunk 是强正 evidence，但 absence 是 `NOT_OBSERVED`，不是 keep-separate、review 或 rejection
的充分条件。没有 trunk evidence 的 tree 可由 crown/slice/geometry 一致性确认。

### 4.7 MERGE / KEEP_SEPARATE / REVIEW 决策

所有 normalization、weights 和 thresholds 严格使用 4.2A-B，不从 scene config 读取。关系不是
强制二元。只有 2D responsibility、3D slicing、fragment continuity 和 final relation evidence
全部完成后，才允许执行一次 constraint-aware `FINAL UNION`：先收集全部 final `MERGE` 和
`KEEP_SEPARATE` constraints，按 canonical edge key 构造完整 constraint set，再一次性求解；禁止
在遍历中增量 permanent union。

若一个 proposed merge component 内存在 final `KEEP_SEPARATE` edge，受冲突影响的 edge/component
整体降为 `REVIEW`，不得用处理顺序决定胜者。later `KEEP_SEPARATE` 明确覆盖 early
`PROVISIONAL_MERGE`，使两 apex 保持两个 final hypotheses；later `MERGE` 且无 constraint conflict
才产生一个 multi-apex final tree。review edge 不自动合并，也不自动拆分；它保留 original early
edge、provisional group、final evidence 和 conflict record，并使共享/冲突 points 在最终比较时
进入 review。`FINAL UNION` 之后禁止第二次 union 或回写 relation decision。

Trunk evidence 不拥有独立 veto 权：shared trunk、distinct trunks 或 multi-stem 只是八通道 relation
中的一个结构通道。一个 trunk 对多个 apex 的强 association 可以显著支持 MERGE，但若 3D slice
明确分裂仍可 `KEEP_SEPARATE/REVIEW`；两个 trunk 分别对应两个 apex 可以支持 KEEP_SEPARATE，
但若低层/中层持续汇入同一 crown structure，仍允许 multi-stem single-tree。

每个 final hypothesis 的 primary apex 从其全部 apices 按以下冻结顺序选择：

```text
HAG descending
local prominence descending
original_index ascending
```

其余全部为 secondary apices，按同一顺序保存。一个 tree 可有零个 secondary apex；一个
apex 不能同时属于两个已确认 final trees。apex ownership 冲突进入 review。

若 final tree 存在一个或多个 accepted trunk candidates，则仅用于记录/解释的
`primary_trunk` 按以下 deterministic key 选择：

```text
association_to_primary_apex_score descending
trunk_candidate_score descending
representative_original_index ascending
```

其余 accepted trunks 作为 ordered secondary/multi-stem refs 保存。一个 tree 可以没有 trunk；
一个 trunk candidate 可以在 provisional 阶段关联多个 apex，但在 final confirmed tree records
中只能归属一个 tree；无法唯一归属时该 trunk association 保留为 review evidence，不强行写入
两个 final trees。

### 4.8 最终同步 point assignment

每个 normal point 的 candidate-tree set 是包含该点的 valid 3D candidate domains、slice
fragments、primary/secondary XY claims 的 canonical union。evaluation 顺序为
`(original_index, hypothesis canonical key)`；同一 point 的所有 tree claims 全部计算完成后
才决策。

每项 claim 精确保留 4.2A-C 冻结的九通道：slice measured support、adjacent-slice continuity、
XY prior、apex support、local HAG/density、structure direction、gap/occlusion、competitor score
margin 和 segmentation quality；上游 input quality 由 formal route validator 先行处理，不另加
隐藏权重。slice evidence 可以推翻 XY primary claim。

状态冻结为：

- `ASSIGNED`、`UNASSIGNED`、`REVIEW_REQUIRED` 只按 4.2A-C 的 `0.62/0.10` exact semantics
  和 mandatory conflict rules 产生；不得在实现中增加 shortcut 或 scene override。

`UNASSIGNED` 只用于完整、无 contract/integrity error 的负结果。prism/slice/claim generation
失败不能伪装为 unassigned；按影响范围进入 point review 或 hard fail。任何 envelope、boundary、
smoothing 或 interpolation 单独都不能创建 assignment。

### 4.9 deterministic Tree ID

primary apex 选择完成、最终 assignments 稳定且空-member hypothesis 移出 normal tree records
后，正式 trees 按以下 key 排序并分配 contiguous `tree_id = 0..K-1`：

```text
(primary_apex_original_index ascending,
 minimum_assigned_member_original_index ascending)
```

primary apex 的 HAG/prominence/order 已由 4.7 节冻结；Tree ID 再以 stable original identity
排序，避免坐标 quantization 或 input/block row order 改变编号。Tree ID 不使用 UUID、不使用
Python `hash()`、不使用 discovery/union iteration order。相同 input bytes、config digest 和
code version 即使内存 row order 改变，也必须得到相同 normalized result digest、tree IDs 和
assignments。

### 4.10 result conservation 和 fail-closed validator

设 `N` 为 formal normal input points、`K` 为 nonempty formal trees、`A` 为 assigned points：

```text
N = assigned_count + review_required_count + unassigned_count
A = assigned_count
```

并必须满足：

- result `original_index` 与 input normal original-index set 精确相等；
- 每个 point 恰有一个 assignment state；
- `tree_id_by_point >= 0` 当且仅当 state 为 `ASSIGNED`；其他为 `-1`；
- 一个 `original_index` 最多属于一个 tree；
- ragged members 严格按 original index 递增，跨 tree 无 duplicate；
- ragged member union 精确等于 dense `ASSIGNED` set；
- 每个 normal tree record 至少一个 member；
- every tree ID contiguous，且 dense/ragged/records 三处一致；
- review/unassigned points 不出现在任何 tree member block；
- apex、trunk、trunk–apex association、slice、relation、boundary references 全部存在且
  identity/digest 对齐；
- accepted trunk supporting `original_index` 必须是 normal input subset；trunk evidence 不创建
  synthetic measured member，也不得把 review/excluded/unrecovered point 拉入 tree；
- final tree 可以有零 trunk；不存在 “tree_count == trunk_count” 或 strict 1:1 invariant；
- `xyz_local_m`、`hag_m` 与正式 handoff input byte-equivalent；
- validator 失败时不发布结果，不静默修复或重排成另一个结果。

## 5. Bundle A 最小正式数据模型

### 5.1 REUSE / EXTEND / NEW

| 决定 | 对象 | 处理 |
|---|---|---|
| REUSE | `SourceIdentity` | 直接复用，不重定义 source/path/origin/digest |
| REUSE | `V4ValidatedFrvMembershipView`、`V4RecoveryCoverageReport`、`V4RecoveredVegetationSet`、`V4InputHandoffResult` | 作为唯一 formal input contract |
| REUSE | `decode_frv_block`、canonical JSON/SHA 和 atomic directory publish 原则 | 用于严格 I/O；不猜 block schema |
| EXTEND | V4 handoff manifest/runtime | 新增 v2 digest-bound manifest 和 public loader；不改变 point/route semantics |
| REUSE-CONCEPT | `TreeCrownSummary`、`TreeGeometrySummary` 字段思想 | bounds/area/equivalent radius、height/vertical extent 概念保留，V3 class 不直接使用 |
| NEW | `V4TreeSegmentationConfig` | 少量物理参数、resource ceilings、version/status/digest |
| NEW | `V4ApexCandidateRecord` | measured apex identity、HAG、prominence、support、reason/quality |
| NEW | `V4TrunkCandidateRecord` | measured trunk representative/support members、axis、vertical span、candidate score、quality |
| NEW | `V4TrunkApexAssociationEvidence` | trunk/apex refs、五通道 evidence、CSP-style structural-path refs、association score/state、diagnostic digest |
| NEW | `V4StructuralPathEvidence` | local sparse graph domain、canonical path node refs、weighted path length、efficiency、availability/digest |
| NEW | `V4RotatingProfileEvidence` | apex/hypothesis ref、24-ray sample inventory、valley/gap/boundary evidence、availability/digest |
| NEW | `V4TreeHypothesisRecord` | primary/secondary apex refs、optional trunk refs、fragment refs、profile refs、state、evidence refs |
| NEW | `V4TreeSliceSummary` | HAG interval、boundary、support、continuity refs、quality |
| NEW | `V4TreeRelationEvidence` | eight-channel evidence（其中 trunk 通道由 association graph 派生）和 `MERGE/KEEP_SEPARATE/REVIEW` |
| NEW | `V4TreeCrownSummary`、`V4TreeGeometrySummary` | V4 original-index/multi-apex/slice-derived geometry |
| NEW | `V4TreeInstanceRecord` | final tree record，不含 V2 cluster prerequisites |
| NEW | `V4TreeSegmentationResult` | original-index keyed dense states + ragged members + evidence |
| NOT ADDED | 独立 `TreeInstanceAssignment` object per point | 避免百万 Python objects；由 result dense arrays 表达同一 contract |

现有 V3 public contracts 保持不变，不 extend `TreeInstanceRecord`/`TreeSegmentationResult`，避免
改变 V3 consumers 和 artifact schema。

### 5.2 Public enums/records

`core/contracts/v4_tree_segmentation.py` 一次定义并冻结：

- `V4TreeAssignmentState = {UNASSIGNED, ASSIGNED, REVIEW_REQUIRED}`；
- `V4TreeRelation = {MERGE, KEEP_SEPARATE, REVIEW}`；
- `V4TrunkCandidateState = {ACCEPTED, WEAK, SUPPRESSED}`；
- `V4TrunkApexAssociationState = {STRONG_ASSOCIATION, REVIEW_ASSOCIATION, UNSUPPORTED_ASSOCIATION}`；
- `V4TrunkEvidenceAvailability = {OBSERVED, NOT_AVAILABLE}`；
- internal/publicly serializable association lookup kind 至少能无歧义表达
  `{EVALUATED, NOT_AVAILABLE, OBSERVED_OUT_OF_RADIUS}`；
- `V4TreeHypothesisState = {ACTIVE, MERGED, CONFIRMED, REVIEW, SUPPRESSED_EMPTY}`；
- closed `V4TreeSegmentationErrorCode` 和 reason/quality `IntFlag`；
- 上表所有 dataclasses、artifact metadata/manifest、digest helpers 和 pure validators。

`V4ApexCandidateRecord` 至少包含：candidate ID、supporting
`original_index`、`xyz_local_m`、HAG、prominence及 availability、plateau/support counts、
score、reason/quality flags 和 diagnostic digest。

`V4TrunkCandidateRecord` 至少包含：deterministic candidate ID、representative
`original_index`、ordered supporting original-index digest、fitted axis、centroid/bounds、observed
HAG span、vertical-bin support、`TV/TA/TS/TC`、candidate score/state、reason/quality flags 和
diagnostic digest。拟合 axis/bounds 可为 derived geometry，但 supporting membership 只能引用
measured normal `original_index`。

`V4TrunkApexAssociationEvidence` 至少包含：trunk ID、apex candidate ID、`TX/TD/TVc/TG/TR`、
**每通道 availability**、`association_score_available`、association score/state、
`structural_path_evidence_reference`、missing/quality flags、early/final validation state 和
diagnostic digest。`association_score=0.0` 在 `score_available=false` 时只是 canonical
serialization placeholder，任何 scorer/lookup 都必须先检查 availability，禁止把 placeholder
当观测零分。

`V4StructuralPathEvidence` 至少包含：trunk/apex refs、local graph domain bounds、node/edge counts、
canonical source/target node keys、path node-key sequence 或 compact predecessor digest、
straight length、weighted path length、`structural_path_efficiency`、availability、resource audit、
reason/quality flags 和 evidence digest。

`V4RotatingProfileEvidence` 至少包含：apex/hypothesis ref、24 个 canonical ray records、每 ray 的
sample count、observed/empty/NOT_AVAILABLE masks、smoothed HAG/density summaries、
`PROFILE_CONFIRMED_VALLEY/WEAK/NO_VALLEY/NOT_AVAILABLE`、measured-gap state、boundary sample
index、supporting-cell/original-index digest 和 evidence digest。

`V4TreeInstanceRecord` 至少包含：tree ID、primary apex reference、ordered secondary apex
references、optional primary trunk reference、ordered secondary/multi-stem trunk references、
trunk–apex association refs、member count/digest、crown/geometry summaries、slice refs、relation
refs、confidence、reason/quality flags 和 `future_visual_evidence_reference=None`。

### 5.3 V4TreeSegmentationResult

正式 result 至少包含：

| 字段 | dtype/shape 或类型 |
|---|---|
| source/config/input bindings | `SourceIdentity`、algorithm/schema version、config/digest、handoff/view/report digests |
| `original_index` | `<i8[N]`，strict increasing/unique |
| `xyz_local_m` | `<f8[N,3]`，从 input 原样保留，供 Bundle B 自包含读取 |
| `hag_m` | `<f4[N]`，从 input 原样保留 |
| `assignment_state_by_point` | `u1[N]` |
| `tree_id_by_point` | `<i4[N]`，非 assigned 为 `-1` |
| `assignment_confidence_by_point` | `<f4[N]` finite `[0,1]` |
| `assignment_reason_flags_by_point` | `<u4[N]` |
| `tree_member_offsets` | `<i8[K+1]` |
| `ordered_tree_member_original_index` | `<i8[A]` |
| record tuples | apex、trunk、trunk–apex association、structural-path、rotating-profile、hypothesis、slice、relation、tree records |
| ragged geometry | responsibility/crown/slice boundary vertex arrays + offsets，明确标记 `non_measured` |
| audit | counts、phase/resource/determinism evidence、warnings、limitations、result digest |

保留 input XYZ/HAG 是正式 result 的 immutable evidence snapshot；它不创建新点，也不改变
assignment。这样 Bundle B 只需读取 Bundle A artifact，不回放 source、不读取 FRV/V4 input、
不重新 segmentation。

### 5.4 Tree geometry summary

每个 formal tree 必须有：primary apex original index/XYZ/HAG、secondary apex count、member
count、min/max/robust HAG、vertical extent、XY centroid/bounds、crown occupied area、equivalent
radius、responsibility area、slice count/valid slice count、lowest/highest supported slice、
`trunk_evidence_state`、accepted trunk count、optional primary trunk representative original index、
multi-stem flag 和 geometry quality flags。树干 axis/association 只作为 measured-support-derived
结构证据；树高/冠幅/member summaries 仍只由 assigned measured points 和已保留 slice evidence
计算，review/unassigned points 不悄悄计入。

## 6. 配置冻结原则

不提供按 filename/source/scene 分支，不把可视化参数放进 segmentation config。algorithm
parameters 控制在以下核心物理集合：

1. `xy_cell_size_m`；
2. `apex_neighborhood_radius_m`；
3. `minimum_apex_prominence_m`；
4. `density_trend_window_m`；
5. `slice_thickness_m`；
6. `slice_overlap_fraction`；
7. `relation_neighborhood_radius_m`。

其中 CHM、gap、adjacency、local-geometry normalizers **以及 trunk detection/association 的
`Ltr_xy/Ltr_z/Lta`** 都从 cell/neighborhood/slice scales 推导，不新增 scene-specific trunk 半径或
“一树一干”阈值。4.2A 的全部 channel weights、`0.55/0.08` responsibility constants、
`0.45/0.12` relation constants、`0.62/0.10` assignment constants，以及 B2 的
`0.62/0.52` trunk candidate gate、`0.65/0.55` trunk–apex association gate 都属于 immutable
Algorithm-V1，不在 config schema 出现，也不得按 scene/source/filename 覆盖。
`minimum_apex_prominence_m` 是唯一单独保留的 apex proposal physical gate，不参与替换 4.2A
的 apex-support normalization。resource ceilings（maximum
points/cells/fragments/edges/
neighborhood evaluations/temporary bytes）属于 operational safety，不算算法调参，也不能被
可视化 CLI 覆盖。

config 是严格 JSON-serializable、immutable、digest-bound，并含
`TEST_ONLY/CALIBRATION_CANDIDATE/PRODUCTION_APPROVED` status。正式 CLI 必须显式接收
`--config-json`，没有隐式 production default。focused tests 使用显式 TEST_ONLY fixture；
首次真实 Human V4 acceptance 使用经 review 的 CALIBRATION_CANDIDATE。只有 Owner 接受后
才能把同一 digest 或后继 digest 标为 PRODUCTION_APPROVED。这个配置确认包含在 Bundle A
综合审查/最小修复内，不另写子计划。

## 7. Review 与 hard-fail 冻结

### 7.1 Hard fail / no output

以下为运行级 hard fail：

- v1/未知 handoff manifest、缺 block digest 或 handoff digest；
- schema/version/identity/source/origin/units/frame/digest mismatch；
- inherited FRV completeness 或 `segmentation_input_complete`/
  `v4_routing_complete` false；
- unsupported route/membership/readiness 出现在 normal block；
- input index duplicate/out-of-range/domain/count mismatch；
- nonfinite required XYZ/HAG/evidence；
- config unknown/unapproved-for-requested-mode 或 digest mismatch；
- spatial index、slice、claim 或 fragment graph resource ceiling exceeded；
- required candidate/prism/slice/claim comparison 未能完整执行且无法精确局部隔离；
- point/ragged/tree/conservation/determinism invariant failure；
- artifact serialization/hash/reload/atomic publication failure。

hard fail 删除 staging directory，正式 output root 保持不存在。

### 7.2 Point/relation REVIEW

输入完整且算法执行完整时，以下保留在结果中而不猜：

- apex tie/conflict 或一个 apex 被多个 final hypotheses claim；
- trunk–apex association 落在 review band、同一 accepted trunk 最终无法唯一归属 tree、或
  shared-trunk 与 distinct-trunk evidence 冲突；
- merge/separate score 落在 review band 或 union constraint conflict；
- XY best/secondary responsibility margin 不足；
- slice split/merge evidence冲突、isolated discontinuity或合法弱支持；
- 多 tree final claim tie/clear advantage 不足；
- localized quality/occlusion/gap anomaly，但 affected points 可精确识别；
- N>0 但无 valid apex，所有 points 进入 review。

review point 无 tree ID，不进入 ragged membership。relation review 必须在 Bundle B 可见。

### 7.3 UNASSIGNED

仅当 formal input、candidate generation、slicing 和 all-candidate comparison 全部成功，且一个
point 对所有 candidates 都缺 sufficient positive support、也没有 mandatory conflict 时使用。
它不是 error catch-all。

## 8. CPU-first 性能设计

- 首版单线程或 deterministic chunk merge；不要求 GPU；
- XY 使用 uniform grid/spatial buckets；XYZ/slice neighbor 使用局部 voxel buckets 和 bounded
  heaps；不新增 SciPy/sklearn dependency；
- apex pairs 只查询 relation radius 邻域；
- trunk candidates 使用 XY column/local HAG bins；trunk–apex pairs 只查询 relation radius 邻域，
  禁止 dense trunk×apex matrix；
- CSP-style structural path 只在单个 trunk-apex pair 的 `Lpath_xy` local voxel corridor 内运行
  deterministic Dijkstra；禁止全场景 all-point graph / all-trunk multi-source dense Dijkstra；
- rotating-profile 固定 24 rays，每 ray 最多 `ceil(Lr/Lxy)+1` samples，使用已有 XY cell summaries，
  禁止按 raw point 做每-ray 全局扫描；
- point-to-tree claims 只对 overlapping candidate domains；
- conflict graph 只在 fragment/hypothesis level；
- 禁止全局 point pair、global `N x N` matrix 和 point-level dense conflict graph；
- 目标复杂度为
  `O(N log L + E_local + E_path_local log V_path + R_profile + E_fragment log F)`，
  其中 `E_path_local/V_path` 仅属于 bounded trunk-apex local corridors，
  `R_profile <= apex_count * 24 * ceil(Lr/Lxy)`；memory 为
  `O(N + cells + fragments + local edges + bounded path/profile diagnostics)`；
- phase resource audit 写入 result；resource preflight 在大分配前执行；
- synthetic stress gate 验证 bounded runtime/memory 和无 quadratic counter growth。

## 9. Bundle A 正式 artifact、API 和 CLI

### 9.1 Runtime API

```python
load_v4_input_handoff(root_path: Path) -> V4InputHandoffResult

segment_v4_trees(
    handoff: V4InputHandoffResult,
    config: V4TreeSegmentationConfig,
) -> V4TreeSegmentationResult

publish_v4_tree_segmentation_artifact(
    result: V4TreeSegmentationResult,
    output_root: Path,
) -> V4TreeSegmentationOutputReference

load_v4_tree_segmentation_artifact(
    root_path: Path,
) -> LoadedV4TreeSegmentationArtifact
```

### 9.2 Artifact inventory

Bundle A output root 精确包含：

```text
v4_tree_segmentation.json
v4_tree_segmentation.npz
v4_tree_segmentation_manifest.json
```

JSON 保存 configs、records、digests、counts、provenance 和 audit；NPZ 保存 dense/ragged point
arrays、boundary vertices和索引表；manifest 固定文件 inventory、size、SHA、array dtype/shape、
semantic result digest。publisher 使用 sibling staging + re-read validation + single directory
rename；不 overwrite existing root。

### 9.3 CLI

```powershell
E:\Miniforge\envs\treeline\python.exe scripts/run_v4_tree_segmentation.py `
  --v4-input-root <published-v2-v4-input-root> `
  --config-json <reviewed-v4-config.json> `
  --output-root <new-result-root>
```

CLI 不接收 FRV threshold、V1/V2 artifacts、source LAS、RGB/video 或 display 参数；不现场重跑
FRV/handoff，不覆盖 input/output。

## 10. Bundle A 精确 implementation scope

### 10.1 New files

```text
core/contracts/v4_tree_segmentation.py
algorithms/vegetation/v4_tree_segmentation.py
algorithms/vegetation/v4_tree_artifacts.py
scripts/run_v4_tree_segmentation.py
tests/contracts/test_v4_tree_segmentation_contracts.py
tests/vegetation/test_v4_tree_segmentation.py
tests/vegetation/test_v4_tree_artifacts.py
tests/scripts/test_run_v4_tree_segmentation.py
```

### 10.2 Modified files

```text
core/contracts/v4_inputs.py
core/contracts/__init__.py
algorithms/vegetation/v4_inputs.py
algorithms/vegetation/__init__.py
algorithms/vegetation/frv_v4_visualization.py
tests/contracts/test_v4_input_contracts.py
tests/vegetation/test_v4_inputs.py
tests/vegetation/test_frv_v4_visualization.py
```

这些 modification 只用于 v2 manifest/public loader、V4 contracts/runtime exports 和
FRV-VIS-A1 v1/v2 compatibility。**TrunkCandidate、Trunk–Apex association、CSP-style structural path、rotating-profile、
multi-stem/multi-apex 逻辑必须落在第 10.1 已授权的新 V4 contract/runtime/artifact/test files 内，
不新增第三个算法模块或新阶段。** 不得顺便重构 FRV/V3。

### 10.3 Fail-closed scope rule

Bundle A 不修改其他文件。若实现时发现完成上述 frozen semantics 必须修改未列出的 project
file，立即停止，不做 opportunistic edit；只允许 Owner 对一个具体新增文件给出 scope 决定。
唯一预先允许的例外是测试拆分：若 `tests/vegetation/test_v4_tree_segmentation.py` 超过 repo
lint/maintainability gate，可在不改变 production scope 的条件下拆成
`test_v4_tree_segmentation_<topic>.py`；拆分文件仍属于同一次 Bundle A review/commit，且必须
在 scope audit 中列明。除此以外无 “implementation may modify”。

明确禁止修改：现有 V3 contracts/algorithms/tests、FRV membership/provider rules、
`pyproject.toml`、data/outputs、binary fixtures、`.claude/` 和四份权威文档。

## 11. Bundle A 自动测试矩阵

`tests/vegetation/test_v4_tree_segmentation.py` 以小型 synthetic full-resolution measured
scenes 为主，不依赖人工浏览器：

1. single isolated tree；
2. two clearly separated trees；
3. two touching crowns；
4. one tree with two apexes；
5. two trees with similar apex heights；
6. low saddle vs high saddle；
7. no trunk evidence -> apex-only fallback；
8. one trunk + one apex -> strong one-to-one seed；
9. one trunk + two apexes -> multi-apex candidate，3D evidence 可合并为一棵；
10. two distinct trunks + two apexes -> strong keep-separate support；
11. two trunks + one apex/crown -> multi-stem candidate，不按 trunk count 强制拆分；
12. unmatched trunk -> diagnostic only，不单独创建 Tree ID；
13. weak/occluded trunk evidence -> neutral/review，不阻塞 apex-only；
14. uneven point density；
15. sparse crown edge；
16. non-zero origin；
17. shuffled input row/block/trunk discovery order；
18. duplicate claim prevention；
19. unassigned conservation；
20. relation/point review state；
21. empty segmentation input；
22. only one apex candidate；
23. no valid apex but trunk candidate exists -> all review, zero trees；
24. very tall narrow tree；
25. broad low crown；
26. adjacent trees with crown overlap；
27. slice split at upper canopy；
28. slice merge lower down；
29. deterministic apex/trunk candidate IDs、tree IDs and normalized result digest；
30. exact original-index dense/ragged conservation；
31. corrupted/inconsistent formal input fail closed；
32. CPU-bounded synthetic stress case。

并额外覆盖：

- v1 manifest formal-loader rejection，v2 fixed digest vectors 和 every-block corruption；
- contract exact fields/dtypes/read-only/no-negative-zero/enum/flag rejection；
- apex primary ordering和 secondary preservation；
- `MERGE/KEEP_SEPARATE/REVIEW` 全 decision branches；
- 4.2A physical normalizer boundaries 和固定 responsibility/relation/assignment score vectors，
  以 `abs<=1e-12` 验证 exact binary64 score 并验证 exact decision enum；
- early `PROVISIONAL_MERGE` + later strong `KEEP_SEPARATE` -> two final trees，且所有 apex/early
  edge/provisional group/competing references 仍可审计；
- early `PROVISIONAL_MERGE` + later `MERGE` support -> one final multi-apex tree；
- early `PROVISIONAL_MERGE` + conflicting slice evidence -> `REVIEW`；
- merge transitivity、single `FINAL UNION` invocation 与 keep-separate constraint conflict 降为 review；
- trunk HAG bin boundary (`0`, `Btr`, exact upper boundary)、occupancy `2/3 points`、
  component min points/bins、core-radius equality、PCA degeneracy/axis-sign canonicalization；
- trunk candidate score 与 trunk–apex association fixed vectors；
- association exact boundaries `0.55`, `0.65`, `math.nextafter(0.65,+inf)`, `1.0`；
- association zero-denominator/direct-adjacency、incomplete corridor、`NOT_AVAILABLE` placeholder-not-used；
- cross lookup `OBSERVED_OUT_OF_RADIUS` vs in-radius `UNSUPPORTED` vs in-radius `NOT_AVAILABLE`；
- same-trunk two-apex at exact `0.65` -> positive relation evidence；
- same-trunk two-apex -> positive relation evidence；
- distinct-exclusive-trunks two-apex -> negative relation evidence；
- distinct trunks with one cross `NOT_AVAILABLE` -> no false distinct-trunk support；
- CSP-style structural path：直连、弯曲但连续、不可达、NOT_AVAILABLE、equal-cost canonical tie；
- 两 trunk 指向同 apex 时 shortest structural path 只作为 association evidence，不直接 winner-take-all；
- structural-path `TVc` 0.50/0.50 fusion fixed vectors；
- rotating-profile 24-ray inventory/order deterministic；
- profile：单次低点不 stop；下降+回升但 density 不降不 stop；
- profile：HAG drop + rebound + density valley + existing D/H decline -> confirmed stop；
- profile：measured gap >= Ld -> confirmed stop；
- profile：NOT_AVAILABLE/occluded samples -> no false gap/no false stop；
- profile boundary 与 rival claim review-margin overlap -> overlapping review，不按 ray 顺序裁决；
- multi-stem one-apex -> 不因 trunk count 自动 `KEEP_SEPARATE`；
- missing trunk / no trunk candidate -> strict neutral，不作为 negative gate；
- trunk candidate/association IDs 与 decisions 对 shuffled row/block order 稳定；
- responsibility solver synchronous reference oracle 和 scan-row independence；
- complete-vs-failed empty candidate set 的 UNASSIGNED/REVIEW 区分；
- artifact manifest、NPZ header、array inventory、hash、extra/missing file、atomic cleanup；
- CLI missing config/input/existing output/nonzero return/no partial output。

不使用 “HTML 看起来正确” 作为 Bundle A test assertion。

## 12. V4-BUNDLE-B — Result Visualization & Acceptance

### 12.1 输入和禁止事项

Bundle B 唯一输入是
`load_v4_tree_segmentation_artifact(<bundle-a-result-root>)` 返回的已验证 Bundle A artifact。
它不读取 V4 input root、FRV artifact、V1/V2、source LAS 或另一个 segmentation config；不调用
`segment_v4_trees`；不重新检测 apex、不重新 merge、不改变 tree ID/assignment/summary。

artifact 已自包含 measured XYZ/HAG、assignments、apex、trunk candidates、trunk–apex
associations、CSP-style structural paths、rotating-profile rays/valleys、boundaries、slices、
relations 和 geometry summaries，因此查看器没有 source replay bootstrap 问题。

### 12.2 必须显示的 layer

- assigned points：每棵 tree 用 deterministic distinct color，hover 显示 tree ID、
  original index、HAG、assignment confidence/reason；
- tree labels/summary table：tree ID、member count、height、crown area/radius、slice count；
- primary apex：单独 marker/label；
- secondary apices：不同 marker，仍显示所属 tree ID；
- accepted/weak trunk candidates：显示 representative point、derived axis、candidate score/state；
- trunk–apex association：显示 STRONG/REVIEW edges、score 和五通道 evidence；明确区分
  one-to-one、one-trunk-multi-apex、multi-stem-one-apex 和 unmatched trunk；
- structural path：只显示 local sparse canonical shortest-path diagnostic（不是 tree member line），
  标出 efficiency/availability；
- rotating profile：可切换显示 24 rays、confirmed/weak valley、measured gap 和 profile boundary
  candidate；derived rays/lines 必须标 `NON_MEASURED_DIAGNOSTIC_GEOMETRY`；
- responsibility/crown boundary：显示 primary/secondary claim boundary 与 final crown boundary，
  清楚标记 derived `non_measured`；
- 3D slices：可切换 slice boundary/centroid/HAG interval；
- relation evidence：MERGE、KEEP_SEPARATE、REVIEW 使用不同 line/symbol，并在 hover/table 展示
  八通道 evidence 和 score/margin/reasons；
- review/conflict points；
- unassigned points；
- tree geometry summary。

页面必须支持旋转/缩放/平移、legend layer toggle 和截图；HTML 用 existing Plotly
`include_plotlyjs=True, full_html=True` 自包含模式。

### 12.3 Deterministic display budget

正式 counts/digests/acceptance summary 始终使用 full arrays；只有 Plotly traces 采样。采样
按 `original_index` 和 fixed seed deterministic，优先级为 review/conflict、apices、small
trees、unassigned、assigned bulk。boundary/slice/relation records 单独设 bounded record cap，
不得把 region count 加入 point count。相同 result 和 display config 必须产生相同 displayed
original-index set；可视化 caps 不进入 segmentation config/digest。

### 12.4 Outputs 和 CLI

```powershell
E:\Miniforge\envs\treeline\python.exe scripts/run_v4_tree_visualization.py `
  --result-root <bundle-a-result-root> `
  --output <result.html> `
  --summary-json <result.json> `
  --review-checklist <result.md> `
  --max-total-points 250000 `
  --max-points-per-tree 100000 `
  --sampling-seed 0
```

输出至少为 self-contained HTML 和 JSON summary；checklist optional。目标文件已存在时拒绝
overwrite；任一 artifact/digest/budget/render error 不留下 partial outputs。

summary 至少包含 source/result/config digests、full/display counts、tree/review/unassigned
counts、multi-apex tree count、multi-stem tree count、accepted/weak/unmatched trunk counts、
trunk–apex STRONG/REVIEW association counts、structural-path observed/unreachable/NOT_AVAILABLE
counts、profile confirmed/weak/gap/NOT_AVAILABLE counts、relation counts、slice quality counts、conservation
PASS/FAIL、sampling policy 和 output status。

### 12.5 Human V4 acceptance

Owner 一眼应能判断：单木是否分开、是否过分割/欠分割、multi-apex 是否被错误拆树、邻树
是否错误合并、**trunk–apex 是否明显错配、multi-stem 是否被错误拆成多树、无 trunk 的树是否仍能
正常走 apex-only fallback**，以及 review/unassigned 是否集中在合理边界。

checklist 逐项记录 `PASS/FAIL/N/A`、tree ID/original-index/relation key、截图和 notes，最终
结论只允许：

- `PASS_READY_FOR_DOWNSTREAM`；
- `PASS_WITH_REVIEW_NOTES`；
- `V4_RULE_ADJUSTMENT_REQUIRED`；
- `DATA_OR_ARTIFACT_BLOCKED`。

前两种允许进入未来 downstream；后两种不得进入 downstream。`V4_RULE_ADJUSTMENT_REQUIRED`
只允许在同一 consolidated scope 内做一次最小 rule/config fix 和重新验收，不倒推创建一组
V4-C1/C2 子计划。结构语义变化仍须按 frozen design change control 处理。

## 13. Bundle B 精确 implementation scope

### 13.1 New files

```text
algorithms/vegetation/v4_tree_visualization.py
scripts/run_v4_tree_visualization.py
tests/vegetation/test_v4_tree_visualization.py
tests/scripts/test_run_v4_tree_visualization.py
```

### 13.2 Modified files

```text
algorithms/vegetation/__init__.py
```

Bundle B 不修改 Bundle A segmentation/contracts/artifact schema；若 renderer 发现正式 result
缺必要 evidence，必须 fail closed 并返回 scope decision，不能从点云重新推断。由于 5.3/5.4
已冻结全部必需 visual evidence，当前没有预期未决文件。

### 13.3 Bundle B tests

- every tree deterministic distinct color and stable trace membership；
- primary/secondary apex mapping；
- trunk candidate / trunk–apex association mapping；
- one-to-one、one-trunk-multi-apex、multi-stem-one-apex、unmatched trunk visual states；
- responsibility/final crown boundaries 分层且标记 non-measured；
- slices、MERGE/KEEP/REVIEW evidence 全部可见；
- review/unassigned 与 assigned mutually exclusive；
- full stats 不受 sampling 改变；
- row order/seed/cap determinism 和 hard cap；
- empty result、one tree、many trees、all review、all unassigned；
- corrupted result artifact fail closed；
- self-contained HTML 含 required layers/hover/table；
- JSON schema/count/conservation/digest；
- checklist exact four outcomes；
- invalid caps/existing output/render failure 无 partial files；
- monkeypatch segmentation entry point 为 forbidden sentinel，证明 Bundle B 不调用 segmentation。

## 14. 验证命令和 Gates

本计划全部 Python/pytest/Ruff 命令唯一允许的解释器是
`E:\Miniforge\envs\treeline\python.exe`；禁止使用 PATH 中的 `python`、`py` 或其他 env。
focused、regression 和 full-repository pytest 必须 PASS；仓库已有且理由明确的合法 skip 可记录，
新增/原因不明 skip 视为 FAIL。

### 14.1 Bundle A focused gate

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_v4_input_contracts.py tests/contracts/test_v4_tree_segmentation_contracts.py tests/vegetation/test_v4_inputs.py tests/vegetation/test_v4_tree_segmentation.py tests/vegetation/test_v4_tree_artifacts.py tests/scripts/test_run_v4_tree_segmentation.py -q

E:\Miniforge\envs\treeline\python.exe -m pytest tests/vegetation/test_frv_v4_visualization.py tests/scripts/test_run_frv_v4_input_visualization.py tests/vegetation/test_canopy_height_model.py tests/vegetation/test_treetop_candidates.py tests/vegetation/test_canopy_segmentation.py tests/vegetation/test_tree_segmentation.py tests/vegetation/test_tree_artifact_io.py tests/vegetation/test_tree_visualization.py tests/scripts/test_run_v3_scene_visualization.py -q

E:\Miniforge\envs\treeline\python.exe -m pytest -q
```

Bundle A 开始实现前，在 untouched baseline HEAD 对 broad scope 运行并保存 machine-readable Ruff
baseline；实现后分别执行 changed-scope 与 broad-scope：

```powershell
E:\Miniforge\envs\treeline\python.exe -m ruff check algorithms core scripts tests --output-format json --output-file <audit-root>\bundle-a-ruff-baseline.json

$BundlePython = @(
  git diff --name-only --diff-filter=ACMR -- '*.py'
  git ls-files --others --exclude-standard -- '*.py'
) | Sort-Object -Unique
E:\Miniforge\envs\treeline\python.exe -m ruff check -- $BundlePython

E:\Miniforge\envs\treeline\python.exe -m ruff check algorithms core scripts tests --output-format json --output-file <audit-root>\bundle-a-ruff-final.json
```

Changed-scope Ruff 对本 Bundle 全部新增/修改 Python files **MUST PASS**，任一 finding 即 Bundle
FAIL。Broad repository Ruff 允许经 baseline JSON 确认、仅位于未修改历史文件中的 pre-existing
findings；必须验证这些 baseline files 对本 Bundle `git diff --quiet -- <file>`，并按
`(relative path, Ruff code, message)` 比较 finding multiset，count/type 不得增加。以下任一情况
FAIL：新 finding、新 code/message、既有 finding 数增加、baseline file 被修改，或 Bundle file
出现 violation。broad command 因完全相同的历史 findings 返回 nonzero 可以记录为
`KNOWN_BASELINE_EXCEPTION`；该标签本身**不是 Bundle failure**，但不得掩盖任何新增 finding。

### 14.2 Bundle A Git scope audit

```powershell
git diff --check
git status --short
git diff --name-status
git diff --stat
git diff --cached --name-status
git ls-files --others --exclude-standard
```

结果必须恰为第 10 节 files（以及永久排除的 `.claude/`），cached 在授权 add 前为空。

### 14.3 Bundle B focused gate

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/vegetation/test_v4_tree_visualization.py tests/scripts/test_run_v4_tree_visualization.py -q

E:\Miniforge\envs\treeline\python.exe -m pytest tests/vegetation/test_frv_v4_visualization.py tests/scripts/test_run_frv_v4_input_visualization.py tests/vegetation/test_tree_visualization.py tests/scripts/test_run_v3_scene_visualization.py -q

E:\Miniforge\envs\treeline\python.exe -m pytest -q
```

Bundle B 使用与 Bundle A 相同的 pre-implementation baseline procedure 和 failure semantics：

```powershell
E:\Miniforge\envs\treeline\python.exe -m ruff check algorithms core scripts tests --output-format json --output-file <audit-root>\bundle-b-ruff-baseline.json

$BundlePython = @(
  git diff --name-only --diff-filter=ACMR -- '*.py'
  git ls-files --others --exclude-standard -- '*.py'
) | Sort-Object -Unique
E:\Miniforge\envs\treeline\python.exe -m ruff check -- $BundlePython

E:\Miniforge\envs\treeline\python.exe -m ruff check algorithms core scripts tests --output-format json --output-file <audit-root>\bundle-b-ruff-final.json
```

Bundle B changed-scope 必须 zero finding；broad scope 只可保留未修改历史文件中的完全相同
baseline multiset。`KNOWN_BASELINE_EXCEPTION` 不等于失败，任何新 finding 必须 FAIL。

随后执行同一组 Git scope commands，结果必须恰为第 13 节 files。Bundle B code review 和
Git closure 通过后，才使用真实 Bundle A artifact 进行 Human V4 acceptance；人工判断不
替代自动测试。

## 15. 压缩实施节奏和 Git closure

正式节奏固定为：

1. 本综合计划一次冻结；
2. 立即一次性实现 Bundle A code + synthetic/focused/regression gates；内部 passes 不单独
   plan/review/commit；同时可并行按 2.4 准备 FRV-VIS Human Scene Gate artifacts；
3. 一次 Bundle A 综合独立 code review；
4. 必要时一次最小修复并重跑完整自动 Gate；
5. 一次 Bundle A Git closure；
6. 在任何真实 Bundle A segmentation run 前，FRV-VIS-A1 Human Scene Gate 必须给出
   `PASS_READY_FOR_V4` 或 `PASS_WITH_REVIEW_NOTES`；
7. Gate PASS 后用升级后的 digest-bound v2 handoff 跑正式/calibration Bundle A scene；
8. 一次 Bundle B 实现；
9. 一次 Bundle B review + 必要最小修复 + Git closure；
10. 真实场景 Human V4 acceptance。

禁止为 apex、2D responsibility、3D slicing、merge、assignment、geometry summary、artifact
各建 commit/authorization。除非发现 frozen contract 无法表达且不能在第 10/13 节范围内
fail-closed 解决的真实 architecture blocker，否则不增加正式阶段。

当前任务无 Git write 权限；本计划不 stage/commit/push。未来每个 bundle 的 add/commit/push
仍需 Owner 明确授权，且每个 bundle 最多一次正式 Git closure。

## 16. 与 FRV-VIS Human Scene Gate 的关系

- 本计划现在可进入独立 review/freeze；
- 当前 `FRV-VIS-A1 Human Scene Gate = PENDING`；
- Gate 前严格按 2.4 使用审核过的 formal runtime/API 生成 FRV root、current V4 v1 root，再由
  现有 viewer CLI 只读生成 HTML/summary/checklist；viewer 永不运行 builder；
- **计划冻结后即可授权 Bundle A code implementation**：只允许 production code、contracts、artifact
  I/O、CLI 和 synthetic/focused/regression tests；不得对真实 scene 执行 formal segmentation；
- 只有 `PASS_READY_FOR_V4` 或 `PASS_WITH_REVIEW_NOTES` 才授权 Bundle A formal/calibration
  real-scene execution 和真实 result publication；
- `FRV_RULE_ADJUSTMENT_REQUIRED` 或 `DATA_OR_ARTIFACT_BLOCKED` 继续阻止真实 scene execution，
  但不要求回退已通过自动 gate 的 Bundle A code；
- Gate pending 不再产生 V4 子计划，也不阻塞 Bundle A 的一次性代码实现；
- 当前 v1 handoff 可用于 FRV-VIS Human Gate；正式 Bundle A real-scene run 必须在代码升级后重新发布
  digest-bound v2 handoff。

## 17. NON-GOALS

- RGB/video fusion；
- SAM、YOLO、PointNet++、SparseConv、GNN 或其他 deep learning；
- GPU dependency；
- growth prediction；
- risk warning；
- tree species recognition；
- manual point editing GUI；
- V4 downstream clearance computation；
- wire-tree distance redesign；
- root/underground geometry；
- FRV membership/routing/HAG rule redesign；
- V2/V3 public contract migration；
- strict `one trunk == one apex == one tree` hard constraint；
- trunk-required tree confirmation；
- production calibration dataset creation outside本 bundle的最小 config acceptance。

## 18. 风险和计划级结论

| 风险 | 已冻结处理 |
|---|---|
| 当前 handoff 无 public loader/持久化 block digest | Bundle A 内升级 v2 manifest + loader；v1 formal segmentation fail closed |
| multi-apex 错拆/过并 | trunk–apex association + early merge 均只 provisional；全部 apex/trunk/edges/groups/competing refs 保留；完整 3D evidence 后只执行一次 FINAL UNION |
| 邻树错误合并 | distinct-exclusive trunk evidence + independent separate support、KEEP constraint、review band |
| 一 trunk 多 apex | 允许 1->N association；shared trunk 是强 merge support，但 final 仍由 3D evidence 决定 |
| 多 trunk 一 crown | 允许 multi-stem candidate；不按 trunk count 强制拆树 |
| 无 trunk/树干遮挡 | absence 严格 neutral；apex-only fallback 保留，其他 crown/slice/geometry 可确认 |
| trunk 误检/错配 | trunk candidate quality + bounded association score + 3D revalidation；unmatched/weak 不创建 Tree ID |
| strong threshold 边界反转 | `strong01` 对 inclusive `0.65` 映射为 `0.5`；`0.55/0.65/nextafter/1.0` fixed vectors 锁定 |
| missing cross-edge 假分树 | sparse edge 使用 total typed lookup；`NOT_AVAILABLE` 永不等于 0/exclusive；只有 observed unsupported/out-of-radius 可给 exclusivity |
| trunk inventory 非确定 | B2.1 冻结 HAG bin、occupancy、component、core、PCA degeneracy/sign、representative/ID canonical rules |
| TLS 最短路径直接移植失真 | 不建全点图；只在 bounded measured voxel corridor 上跑 deterministic Dijkstra；path 只作为 association/continuity evidence，不能直接 winner-take-all |
| 旋转剖面单噪声误截冠 | 24 canonical rays + 3-sample median + drop/rebound+density valley conjunction；profile valley 单独不能 hard-stop |
| 遮挡被误判为空 gap | profile/path 都显式区分 `OBSERVED` 与 `NOT_AVAILABLE`；未知不算 gap、不产生 false KEEP/stop |
| XY overlap | retained secondary/multi-height claims；3D comparison 最终决策 |
| 顺序依赖 | canonical original-index ordering、synchronous solver/reference oracle、stable graph keys |
| point 丢失/重复 | dense state + ragged exact conservation + artifact reload validator |
| 大点云 O(N²) | grid/local buckets、bounded neighbor、fragment graph、resource fail closed |
| 可视化改变结果 | Bundle B 只读 formal result，segmentation function forbidden by test |
| threshold 膨胀 | 7 个核心物理 parameters；全部 score weights/decision constants 在 Algorithm-V1 冻结，normalizers 从共同 scales 推导 |
| Human Gate runtime bootstrap | Owner 只提供正式 artifact/config/approval 路径；runtime objects 由 typed contracts + public loaders 在同一正式进程构造；缺上游 artifact/authority/config 时 `DATA_OR_ARTIFACT_BLOCKED`，viewer 永不补建 |

当前未发现需要拆分第三个 V4 bundle 的 architecture blocker。新增 trunk+apex 联合种子、CSP-style local structural path 和 rotating-profile boundary prior
全部落在现有 Bundle A scope 内：新 record/enum 放入既定 `v4_tree_segmentation.py`，检测、
association、path/profile、relation 和 assignment runtime 放入既定 `v4_tree_segmentation.py`，
不新增正式阶段或 production 文件。现有 handoff integrity gap 可在 Bundle A 指定 scope 内解决；V3/V4 contract 语义差异通过
新的 V4-only contract 隔离，而不是修改 V3。

## 19. 本轮完成状态

```text
Plan status:
PLAN-REVIEW-CANDIDATE

Architecture:
BUNDLE_A_CORE_PLUS_BUNDLE_B_VISUALIZATION

Bundle A:
CONSOLIDATED_SINGLE_TREE_SEGMENTATION_CORE

Bundle B:
CONSOLIDATED_RESULT_VISUALIZATION

Algorithm-V1 deterministic scoring:
FROZEN

Responsibility / relation / assignment scoring:
FROZEN

Trunk candidate / trunk-apex association scoring:
FROZEN

V4-TA-001 inclusive strong threshold:
CLOSED

V4-TA-002 availability / cross lookup:
CLOSED

V4-TA-003 trunk inventory determinism:
CLOSED

Trunk candidate detection:
INCLUDED_IN_BUNDLE_A

Trunk-Apex joint association:
INCLUDED_IN_BUNDLE_A

CSP-style trunk-rooted sparse structural path evidence:
INCLUDED_IN_BUNDLE_A

Global all-point shortest-path graph:
PROHIBITED

Rotating-profile crown boundary prior:
INCLUDED_IN_BUNDLE_A

Rotating-profile direct Tree ID authority:
PROHIBITED

Strict one-trunk-one-apex constraint:
REJECTED_AS_HARD_INVARIANT

Allowed association cardinality:
1_TO_1 + 1_TO_N + N_TO_1 + APEX_ONLY_FALLBACK

Trunk absence:
NEUTRAL_NOT_REQUIRED

Multi-apex:
INCLUDED_IN_BUNDLE_A

Early merge:
PROVISIONAL_ONLY

Final union:
ONCE_AFTER_COMPLETE_3D_EVIDENCE

2D responsibility regions:
INCLUDED_IN_BUNDLE_A

3D slicing:
INCLUDED_IN_BUNDLE_A

Merge/separate/review:
INCLUDED_IN_BUNDLE_A

Unique original-index assignment:
INCLUDED_IN_BUNDLE_A

Deterministic Tree ID:
INCLUDED_IN_BUNDLE_A

RGB/video:
DEFERRED

Implementation:
NOT STARTED

Formal V4 segmentation:
NOT STARTED

F-005 assignment vector consistency:
CLOSED

F-006 exact runtime bootstrap:
CLOSED_WITH_TYPED_RUNTIME_CONSTRUCTION

FRV-VIS Human Scene Gate:
PENDING

Pre-Bundle-A Human Gate bootstrap path:
FROZEN

Bundle A code implementation after plan freeze:
AUTHORIZED_BY_SEPARATE_OWNER_ACTION

Bundle A formal real-scene execution:
BLOCKED_UNTIL_FRV_VIS_HUMAN_GATE_PASS

Git writes:
NONE
```
