# FRV-A2 to V4 Handoff 综合实施计划

状态：`PLAN-FROZEN`

计划标识：`FRV-TO-V4-BUNDLE`

Repository：`E:\TreeLineWarning`

Branch：`feature/algorithm`

Baseline HEAD：`22837c6192b1bb2e9696bf2507bc1b919a1c6fd6`

Remote divergence：`0 0`

Baseline worktree：项目文件 clean；`.claude/` 为 `EXCLUDED_AND_IGNORED`

本文件是从已完成的 P-HAG 到正式 V4 输入就绪之间的唯一综合计划。它一次冻结
P-GEOMETRY、P-WIRE、full-resolution membership、routing、FRV artifact publication、
V4 input handoff 和不授权实现的 multi-apex 设计边界。不得再拆分创建
`frv-a2-p-geometry-implementation-plan.md`、`frv-a2-p-wire-implementation-plan.md` 或
`frv-a2-membership-implementation-plan.md`。

本文件只冻结未来实施合同，不授权本轮修改代码、执行测试造成的项目写入、暂存、
提交或推送。

## 1. 权威、优先级和不重审边界

规范权威按以下顺序解释；下级只能细化上级未给出的实现事实，不能改变上级语义：

1. `docs/full-resolution-vegetation-membership-design.md` 的 `DESIGN-FROZEN` FRV 语义；
2. `docs/v4-tree-segmentation-design.md` 的 `DESIGN-FROZEN` FRV consumer/V4-A 边界；
3. `docs/frv-a2-implementation-plan.md` 的 provider role、BND1、FULL_SOURCE 和 failure phase；
4. `docs/frv-a2-p-source-implementation-plan.md` 与当前 P-SOURCE 合同/实现；
5. `docs/frv-a2-p-hag-implementation-plan.md` 与当前 P-HAG 合同/实现；
6. 本文件对尚未实现部分给出的精确 API、算法、参数、digest、文件和测试冻结。

当前闭环且本计划不得改写或重审的实现是：

- `core/contracts/frv_source.py` 与 `algorithms/io/source_provider.py`；
- `core/contracts/frv_hag.py` 与 `algorithms/vegetation/hag_provider.py`；
- 它们在 `core/contracts/__init__.py`、`algorithms/io/__init__.py`、
  `algorithms/vegetation/__init__.py` 中的现有公开导出；
- P-SOURCE 的一次 session、P-HAG 的两次独立 P-SOURCE pass、它们的 proof、result、
  BND1、failure mapping 和测试结论。

`core/contracts/frv.py` 与 `core/contracts/frv_a2.py` 是已闭环的通用合同权威。本计划的
实现不得改变其 enum 数值、既有 dataclass 字段、digest 语义或 validator 结果。
Session B 只消费这些合同；不在 `core/contracts/frv.py` 内增加新 payload 数组。

## 2. 总体边界和完成定义

### 2.1 FULL_SOURCE 与 EXHAUSTIVE_REGION

四个正式 provider 的域都必须是：

```text
domain_kind = FrvA2DomainKind.FULL_SOURCE
domain_count = SourceIdentity.source_point_count = N
original_index domain = exactly 0, 1, ..., N-1
```

P-SOURCE、P-HAG、P-GEOMETRY、P-WIRE 对每个原始行提供全源证据。它们的
`FULL_SOURCE` 不等于所有原始点都被 FRV 接纳，也不把 scope 外点解释为排除。

生产 v1 的 membership scope 仍为 `FrvScopeMode.EXHAUSTIVE_REGION`。只对冻结的
`scope_context_margin` 三维并集内的唯一 `original_index` 产生五态终态；并集外点没有
FRV membership 行，语义是 `OUT_OF_SCOPE`，不是 `EXCLUDED_NON_CANDIDATE`。

### 2.2 成功结果

一个 source/scope 只有同时满足以下条件才是正式 V4 输入就绪：

1. 四个 provider 的 finalized result、proof、BND1 和 FULL_SOURCE 守恒均通过；
2. canonical/required inventory、scope、coverage 与 point routing 守恒通过；
3. `FRVScopeCoverageManifest.routing_complete is True`；
4. `FullResolutionVegetationMembershipArtifact.membership_decision_complete is True`；
5. artifact、payload schema、block、binding 和 source replay 全部通过；
6. V4 已精确回收所有应回收原始行；
7. 正常 V4 segmentation input 恰好是
   `{VERIFIED_VEGETATION_CANDIDATE, MANUAL_INCLUDE} ∩ {HAG READY} ∩ {XYZ recovered}`；
8. membership review、HAG review 和 recovery failure 没有被静默丢弃。

本阶段不生成 `tree_id`，不运行 CHM、apex detection、冠层分割、树分配或 clearance。

### 2.3 空输入

canonical inventory 为零时，结果为 `NO_RELEVANT_FRV_SCOPE`：coverage manifest 的
空清单规则必须通过，不发布 membership artifact，也不为该 source 创建 V4 normal input。
这是完整 audit route，不是包含零记录的伪 membership artifact。

canonical inventory 非空、scope union 非空但 source 中零点命中 union 时，结果为
`REGION_MAPPING_INCOMPLETE`，不得发布 artifact。scope 内有点而 admitted 集为空时，
membership artifact 合法，`V4RecoveredVegetationSet.N = 0` 合法，全部 review/excluded
记录仍按本计划守恒。

## 3. P-GEOMETRY 精确冻结

### 3.1 输入和公开 API

P-GEOMETRY 只接受四个正式输入和两个 operational 参数：

```python
def load_frv_geometry(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    *,
    frv_configuration: FrvConfiguration,
    provider_config: FrvGeometryProviderConfig,
    scratch_dir: Path,
    source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvGeometryLoadSession
```

`scratch_dir` 和 `source_chunk_size` 不进入任何 formal digest。`scratch_dir` 必须已存在、
是目录、可写且位于调用者明确提供的位置；session 只创建一个随机临时子目录，成功或
失败关闭时删除该子目录。它不接受 LAS path、laspy handle、output artifact path、HAG、
V1/V2 label 或 membership state。

prevalidation 必须要求 config interface/config digest 分别等于
`frv_configuration.geometry_provider_interface_version` 和
`frv_configuration.geometry_provider_config_digest`，并验证 finalized source result、
reference、configuration、source content/record identity、FULL_SOURCE、frame、units、origin
和 identity transform；第一处失败前不得打开 source session。

```python
class FrvGeometryLoadSession(Protocol):
    def iter_chunks(self) -> Iterator[FrvGeometryChunk]: ...
    def finalize(self) -> FrvGeometryProviderResult: ...
    def close(self) -> None: ...
    def __enter__(self) -> FrvGeometryLoadSession: ...
    def __exit__(self, exc_type, exc, traceback) -> Literal[False]: ...
```

状态机严格为
`CREATED -> INDEXING -> INDEX_READY -> ITERATING -> STREAM_COMPLETE -> FINALIZED -> CLOSED`。
任何提前 finalize 进入 `ABORTED`；重复 iterator、重复 finalize、abort 后继续读取均失败。
`close()` 幂等；活动异常不得被 cleanup 异常覆盖。

### 3.2 常量、config 和默认值

```text
FRV_GEOMETRY_PROVIDER_KEY = "geometry_provider"
FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION = "frv-geometry-provider-v1"
FRV_GEOMETRY_PROVIDER_SCHEMA_VERSION = "frv-geometry-provider-result-v1"
FRV_GEOMETRY_CONFIG_SCHEMA_VERSION = "frv-geometry-provider-config-v1"
FRV_GEOMETRY_PROOF_SCHEMA_VERSION = "frv-geometry-provider-proof-v1"
FRV_GEOMETRY_ALGORITHM_VERSION = "frv-geometry-multiscale-pca-grid-v1"
FRV_GEOMETRY_CONFIG_DIGEST_DOMAIN = "FRVGEOCFG1"
FRV_GEOMETRY_POINT_CONTENT_DIGEST_DOMAIN = "FRVGEOPCD1"
FRV_GEOMETRY_PROOF_DIGEST_DOMAIN = "FRVGEOPRF1"
FRV_GEOMETRY_CACHE_BYTES = 536870912
```

```python
@dataclass(frozen=True, slots=True)
class FrvGeometryProviderConfig:
    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    radii_m: tuple[float, float, float]
    min_neighbor_counts: tuple[int, int, int]
    max_neighbors_per_scale: int
    index_cell_size_m: float
    support_neighbor_radius_m: float
    support_neighbor_cap: int
    direction_cosine_min: float
    eigenvalue_floor_m2: float
    canopy_scattering_min: float
    canopy_linearity_max: float
    canopy_density_min_per_m3: float
    woody_linearity_min: float
    woody_verticality_max: float
    woody_vertical_span_min_m: float
    vertical_cylinder_radius_m: float
    vertical_half_window_m: float
    vertical_slice_height_m: float
    min_vertical_occupied_slices: int
    ground_planarity_min: float
    ground_normal_z_min: float
    max_index_cell_points: int
    provider_config_digest: str
```

默认值按字段顺序为：

```text
schema_version = "frv-geometry-provider-config-v1"
provider_key = "geometry_provider"
provider_role = FrvA2ProviderRole.GEOMETRY
interface_version = "frv-geometry-provider-v1"
algorithm_version = "frv-geometry-multiscale-pca-grid-v1"
radii_m = (0.30, 0.75, 1.50)
min_neighbor_counts = (8, 12, 20)
max_neighbors_per_scale = 64
index_cell_size_m = 0.75
support_neighbor_radius_m = 0.30
support_neighbor_cap = 16
direction_cosine_min = 0.75
eigenvalue_floor_m2 = 1e-12
canopy_scattering_min = 0.08
canopy_linearity_max = 0.90
canopy_density_min_per_m3 = 2.0
woody_linearity_min = 0.70
woody_verticality_max = 0.35
woody_vertical_span_min_m = 0.75
vertical_cylinder_radius_m = 0.30
vertical_half_window_m = 1.50
vertical_slice_height_m = 0.25
min_vertical_occupied_slices = 3
ground_planarity_min = 0.70
ground_normal_z_min = 0.85
max_index_cell_points = 1000000
```

所有 float 必须是 finite built-in float，所有 int 必须是 built-in int 且拒绝 bool/NumPy
scalar。三个 radius 严格递增；三个 minimum count 均为正且不超过 64；概率/比率字段在
`[0,1]`；cell size、radius、span、slice height、density、floor 均大于零；
`support_neighbor_radius_m == radii_m[0]`，
`vertical_half_window_m == radii_m[2]`，
`2 * vertical_half_window_m / vertical_slice_height_m == 12` 必须精确成立。

`derive_frv_geometry_provider_config_digest(...)` 对
`{"digest_domain":"FRVGEOCFG1", ...}` 调用现有 `frv_digest`；payload 含上述除
`provider_config_digest` 外全部字段，tuple 按原顺序序列化。path、chunk size、cache、
scratch name、host、mtime、线程数和内存地址不入 digest。

### 3.3 精确 pass、index、tile 和 halo

P-GEOMETRY 恰好创建两次独立 `load_frv_source` session：

1. **Pass G1 — spatial index**：按 source 顺序读取一次，将每行编码为
   `<qddd`（little-endian original_index、x、y、z），追加到 cell spool；
2. **Pass G2 — feature output**：再次按 source chunk 顺序读取一次，每个 source chunk
   一对一产生一个 geometry chunk，同时累计 point-content digest。

每个 pass 都必须自然耗尽、finalize、关闭，并把新 source result 的 exact proof
serialization、SourceIdentity、artifact components、FULL_SOURCE index digest 和 BND1 与
supplied `source_result` 逐项相等比较。G1 比较通过后才进入 G2；G2 比较通过且 geometry
stream 守恒后才允许 P-GEOMETRY finalize。

不得创建第三次 source pass，不得直接打开 LAS，不得复用已消费 session，不得保留完整
source ndarray。

index anchor 是 source proof 的 `source_bounds_min_xyz`。每个点的 cell key 为：

```text
ix = floor((x - source_min_x) / 0.75)
iy = floor((y - source_min_y) / 0.75)
iz = floor((z - source_min_z) / 0.75)
```

cell 为每轴 min-inclusive/max-exclusive。cell 文件中的记录天然按 `original_index` 递增；
formal cell inventory 按 `(ix, iy, iz)` 递增。单 cell 超过 `max_index_cell_points` 是
provider-level resource failure，不允许抽样。

一个 core point 只由其 Pass G2 source chunk 输出一次。最大 feature halo 固定为 1.50 m；
候选 cell 范围是每轴
`floor((coordinate ± radius - source_min) / index_cell_size_m)` 的闭整数范围。LRU 只影响
I/O，固定上限为 536870912 bytes；cache hit/miss 不影响结果。并行实现只允许按 core
chunk 独立计算，合并顺序仍为 source chunk id 和 original_index。

### 3.4 neighborhood、density、covariance/PCA

对每个 query point `p_i` 和每个 `r in (0.30, 0.75, 1.50)`：

1. 从 halo cells 取同 source 的所有点；
2. 排除 `original_index == i`；
3. 用 float64 计算 `d2 = dot(p_j-p_i, p_j-p_i)`；
4. 保留 finite 且 `d2 <= r*r` 的点；
5. 严格按 `(d2, original_index)` 排序；
6. `raw_neighbor_count` 是截断前数量；PCA 只取前 64 个；
7. `neighbor_count = min(raw_neighbor_count, 64)`；
8. `density = raw_neighbor_count / ((4/3) * pi * r**3)`；
9. `local_spacing` 是所取邻点距离的 float64 median；零邻点时为正 `+0.0`。

实现不得物化一个高密度 point 的全部 candidates：按 canonical cell/record order 流式计数，
用容量 64 的 max-heap 保留全序最小 64 项，最终升序排序。该 heap 算法必须与“全候选排序
后取前 64”reference oracle byte-identical。support neighbors 使用容量 16 heap；vertical
span/count/slice bitmap 在扫描时累计。

PCA 样本是 query point 加所取邻点，共 `m = neighbor_count + 1`。当邻点数达到该 scale
的 minimum 时：

```text
mean = sequential float64 sum(samples in original_index order) / m
C = sequential sum((p-mean) outer (p-mean)) / m
(lambda0, lambda1, lambda2), V = numpy.linalg.eigh(C)
```

`lambda0 <= lambda1 <= lambda2`；`[-1e-12,0)` 夹到 `+0.0`，小于 `-1e-12` 是 provider
failure。每个 eigenvector 的最大绝对分量取正；绝对值相同按 x、y、z 选首个分量。

未达到 scale minimum 时，该 scale 的 eigenvalues、directions、normal 和四个 shape
features 全为正 `+0.0`。当 `lambda2 <= 1e-12` 时 eigenvalues 保留、其余值为 `+0.0`。
当 `lambda2-lambda1 <= 1e-12` 时 largest principal direction 固定为零向量且
`verticality=1.0`；当 `lambda1-lambda0 <= 1e-12` 时 surface normal 固定为零向量且
`surface_normal_z=0.0`。只有对应 spectral gap 大于 floor 时才采用符号 canonicalized
eigenvector，避免 repeated-eigenvalue basis 进入 formal digest。

当 `lambda2 > 1e-12`：

```text
linearity  = (lambda2 - lambda1) / lambda2
planarity  = (lambda1 - lambda0) / lambda2
scattering = lambda0 / lambda2
verticality = 1 - abs(V_largest[z])
surface_normal_z = abs(V_smallest[z])
```

结果在数学完成后转换成约定输出 dtype；所有 exact zero 在 chunk 和 digest 前归一成正
`+0.0`。`lambda2 <= 1e-12` 是该 scale 的 degenerate evidence，不产生正结构证据。

### 3.5 vertical structure 和结构证据

vertical cylinder 取所有满足
`sqrt(dx**2+dy**2) <= 0.30` 且 `abs(dz) <= 1.50` 的非自身点，排序仍为
`(d2, original_index)`。定义：

```text
vertical_span_m = max(z including query) - min(z including query)
above_count = count(dz > 0.05)
below_count = count(dz < -0.05)
slice_id = floor((dz + 1.50) / 0.25), clipped only for dz == +1.50 to 11
vertical_occupied_slices = count(unique slice_id)
```

公开 flags：

```text
FrvGeometryEvidenceFlag.NONE = 0
NEIGHBORHOOD_SUFFICIENT = 1
DENSITY_SUPPORTED = 2
CANOPY_BRANCHING_COMPATIBLE = 4
WOODY_COLUMN_COMPATIBLE = 8
GROUND_SURFACE_COMPATIBLE = 16
VERTICAL_CONTINUITY = 32
```

```text
FrvGeometryEvidenceState.VALID = 1
FrvGeometryEvidenceState.INSUFFICIENT = 2
FrvGeometryEvidenceState.DEGENERATE = 3
```

```text
FrvGeometryReasonFlag.NONE = 0
INSUFFICIENT_NEIGHBORS = 1
DEGENERATE_COVARIANCE = 2
```

```text
FrvGeometryQualityFlag.NONE = 0
NEIGHBOR_CAP_REACHED = 1
SUPPORT_NEIGHBOR_CAP_REACHED = 2
```

state 取值规则：medium 和 large 两个 scale 都达到 minimum 且 `lambda2 > floor` 时为
`VALID`；任一未达 minimum 为 `INSUFFICIENT`；数量满足但任一 degenerate 为
`DEGENERATE`。quality cap flag 是确定性截断事实，不把 finalized provider 变成失败。

正 evidence 规则：

```text
NEIGHBORHOOD_SUFFICIENT := state == VALID
DENSITY_SUPPORTED := large_density >= 2.0
CANOPY_BRANCHING_COMPATIBLE := VALID and large_scattering >= 0.08
    and large_linearity <= 0.90 and DENSITY_SUPPORTED
VERTICAL_CONTINUITY := vertical_span_m >= 0.75
    and vertical_occupied_slices >= 3 and above_count >= 1 and below_count >= 1
WOODY_COLUMN_COMPATIBLE := VALID and medium_linearity >= 0.70
    and medium_verticality <= 0.35 and VERTICAL_CONTINUITY
GROUND_SURFACE_COMPATIBLE := VALID and medium_planarity >= 0.70
    and medium_surface_normal_z >= 0.85
```

所有 threshold 比较使用转换为 float32 之前的 float64 值；flag 决策完成后才写 float32
display/evidence arrays。因此 dtype 转换不能改变 boundary branch。

canopy、woody、ground flags 可同时存在；冲突由 membership 处理，P-GEOMETRY 不输出
membership state。

### 3.6 structural-support neighbor

每点先在 0.30 m 邻域取前 16 个 `(d2, original_index)` 候选。无向 support edge 仅当：

1. 两端互相位于各自前 16 候选；
2. 两端 geometry state 都是 `VALID`；
3. `abs(dot(v_largest_i_medium, v_largest_j_medium)) >= 0.75`，或两端都有
   `CANOPY_BRANCHING_COMPATIBLE`；
4. 两端 direct distance 不超过 0.30 m。

edge 不读 HAG、wire、V1/V2 或 final state。每点输出升序
`(distance_squared, neighbor_original_index)` 的最多 16 个邻点，unused slot 为 `-1`。
mutual 规则保证邻接对称；validator 必须在完整 stream digest 验证对称性。

### 3.7 chunk、proof、result、digest 和 BND1

`FrvGeometryChunk` 是 frozen/slots/runtime-only，字段顺序为：

```text
source_id, chunk_id, start_index, point_count, original_index,
raw_neighbor_count, neighbor_count, density_per_m3, local_spacing_m,
eigenvalues_m2, principal_direction, surface_normal,
linearity, planarity, scattering, verticality,
vertical_span_m, vertical_occupied_slices, above_count, below_count,
support_neighbor_count, support_neighbor_original_index,
geometry_evidence_flags, geometry_evidence_state,
reason_flags, quality_flags, is_last
```

数组 dtype/shape：

| Field | dtype | shape |
|---|---|---|
| `original_index` | `<i8` | `[P]` |
| `raw_neighbor_count` | `<u4` | `[P,3]` |
| `neighbor_count` | `<u2` | `[P,3]` |
| `density_per_m3`, `local_spacing_m` | `<f4` | `[P,3]` |
| `eigenvalues_m2` | `<f8` | `[P,3,3]` |
| `principal_direction`, `surface_normal` | `<f4` | `[P,3,3]` |
| `linearity`, `planarity`, `scattering`, `verticality` | `<f4` | `[P,3]` |
| `vertical_span_m` | `<f4` | `[P]` |
| `vertical_occupied_slices`, `above_count`, `below_count` | `<u2` | `[P]` |
| `support_neighbor_count` | `|u1` | `[P]` |
| `support_neighbor_original_index` | `<i8` | `[P,16]` |
| evidence/state | `<u4`, `|u1` | `[P]` |
| reason/quality | `<u4`, `<u4` | `[P]` |

所有数组 C-contiguous、read-only、无 object、无 NaN/Inf、无 negative zero。chunk scalar
与 Pass G2 source chunk 一对一相等；全 stream 严格覆盖 `0..N-1`。

`FRVGEOPCD1` framing：ASCII `FRVGEOPCD1\0`，依次为三个 UTF-8 SHA（source content、
source record、provider config，各以 NUL 结束）、`uint64 N`，然后按 original_index 为
每个字段写 little-endian raw C-order bytes；字段顺序就是上表顺序，排除 source_id、
chunk scalar 和 `is_last`。每个 ndarray 前写 `uint16 field_name_byte_length`、field name、
`uint32 row_byte_length`，防止拼接歧义。

`FrvGeometryProviderProof` 的 exact field order 为：

```text
schema_version
provider_key
provider_role
interface_version
algorithm_version
configuration_digest
provider_config_digest
source_artifact_reference_id
artifact_reference_id
artifact_components
source_content_identity_digest
source_identity_record_digest
source_proof_digest
source_binding_digest
source_point_count
domain_kind
domain_count
original_index_domain_digest
geometry_point_content_digest
valid_count
insufficient_count
degenerate_count
neighborhood_sufficient_count
density_supported_count
canopy_branching_compatible_count
woody_column_compatible_count
ground_surface_compatible_count
vertical_continuity_count
neighbor_cap_reached_count
support_neighbor_cap_reached_count
coordinate_frame
units
origin_xyz
transform_declaration
quality_flags
reason_flags
error_codes
proof_digest
```

三种 state count 之和为 N；每种 evidence count 是对应 flag 的独立计数；success proof
aggregate 的 `quality_flags=0`、`reason_flags=0`、`error_codes=()`。proof digest 对
`{"digest_domain":"FRVGEOPRF1", ...all fields except proof_digest}` 使用 `frv_digest`。

`FrvGeometryProviderResult` 是
`@dataclass(frozen=True, slots=True, init=False)`，exact fields 为 `proof`、
`source_identity`、`_finalized_capability`；capability 不参与 repr/equality/digest/serialization。
public constructor 永远抛 `TypeError`。公开 validators 是
`validate_frv_geometry_provider_config`、`validate_frv_geometry_chunk`、
`validate_frv_geometry_provider_proof`、`validate_finalized_frv_geometry_provider_result` 和
`to_frv_a2_provider_binding_evidence`。公开 error 是：

```python
class FrvGeometryLoadError(Exception):
    failure_record: FrvA2FailureRecord
    quality_flags: int
```

`binding_digest` 只调用 existing `frv_a2_provider_binding_digest`，digest domain 是
`FRVA2BND1`。BND1 固定映射：formal GEOMETRY、module
`algorithms.vegetation.geometry_provider`、type `FrvGeometryProviderResult`、loader
`load_frv_geometry`、FULL_SOURCE、VALIDATED；artifact component 复用 source proof 的
`source_file`；upstream digests 是排序唯一的
`(geometry_point_content_digest, source_binding_digest, source_proof_digest)`。

### 3.8 P-GEOMETRY failure mapping

public error 是 `FrvGeometryLoadError(failure_record, quality_flags)`；detail 严格使用 A2
固定格式，无 path/异常文本。first failure wins。

| Condition | Phase | field | error | reason | quality |
|---|---|---|---|---|---|
| non-finalized/invalid source result or reference | STRUCTURE_INTEGRITY | `source_result` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| invalid geometry config | STRUCTURE_INTEGRITY | `provider_config` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| source identity/frame/unit/transform mismatch | IDENTITY_CONTEXT | `source_identity` | `FRV_SOURCE_IDENTITY_FAILED` | `PROVIDER_UNCERTAINTY` | `SOURCE_IDENTITY_FAILURE` |
| nonfinite XYZ | SEMANTIC_READINESS | `xyz` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| index cell limit exceeded | SEMANTIC_READINESS | `index_cell` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| index/spool read-write-cleanup failure | STRUCTURE_INTEGRITY | `scratch_store` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| lost original index | DOMAIN_ALIGNMENT | `original_index` | `FRV_ORIGINAL_INDEX_INVALID` | `PROVIDER_UNCERTAINTY` | `ORIGINAL_INDEX_INVALID` |
| duplicate original index | DOMAIN_ALIGNMENT | `original_index` | `FRV_ORIGINAL_INDEX_DUPLICATE` | `PROVIDER_UNCERTAINTY` | `ORIGINAL_INDEX_DUPLICATE` |
| out-of-range original index | DOMAIN_ALIGNMENT | `original_index` | `FRV_ORIGINAL_INDEX_INVALID` | `PROVIDER_UNCERTAINTY` | `ORIGINAL_INDEX_INVALID` |
| source pass incomplete/count mismatch | DOMAIN_ALIGNMENT | `source_pass` | `FRV_STREAM_INCOMPLETE` | `PROVIDER_UNCERTAINTY` | `STREAM_INCOMPLETE` |
| nonfinite covariance/eigensystem/feature | SEMANTIC_READINESS | `pca` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| eigenvalue below `-1e-12` | SEMANTIC_READINESS | `eigenvalue` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| source proof/content/BND1 mismatch | CROSS_PROVIDER | `digest` | `FRV_ARTIFACT_DIGEST_MISMATCH` | `PROVIDER_UNCERTAINTY` | `ARTIFACT_DIGEST_MISMATCH` |
| lifecycle misuse | STRUCTURE_INTEGRITY or DOMAIN_ALIGNMENT | `session_state` | `FRV_PROVIDER_INVALID` or `FRV_STREAM_INCOMPLETE` | `PROVIDER_UNCERTAINTY` | matching quality |

邻点不足和 covariance degenerate 是成功的逐点 evidence state，不是 provider failure。

## 4. P-WIRE 精确冻结

### 4.1 正式 authority

现有 W1 不是 formal full-resolution provider。P-WIRE 只接受一个经过审批、source-bound
的 `FrvWireAuthorityManifest`：

```python
class FrvWireAuthorityMode(str, Enum):
    W1_MODEL_SET = "w1_model_set"
    EMPTY_INVENTORY = "empty_inventory"
```

```python
@dataclass(frozen=True, slots=True)
class FrvWireAuthorityManifest:
    schema_version: str
    authority_mode: FrvWireAuthorityMode
    source_content_identity_digest: str
    source_identity_record_digest: str
    source_id: str
    coordinate_frame: str
    units: str
    origin_xyz: tuple[float, float, float]
    transform_declaration: str
    inventory_complete: bool
    approved_wire_ids: tuple[int, ...]
    w1_artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
    geometry_content_digest: str
    authority_rule_id: str
    authority_rule_version: str
    approver: str
    approval_timestamp: str
    approval_digest: str
    manifest_digest: str
```

schema 是 `frv-wire-authority-v1`；frame=`LOCAL_CARTESIAN_XYZ`、units=`m`、transform=
`identity`、`inventory_complete=True`。W1 mode 的 `w1_artifact_components` 必须是 W1
JSON/NPZ exact pair；`approved_wire_ids` 严格递增且与 W1 `WireModelSet.wire_id` 完全相等；
每个 wire 必须 `WireModelStatus.CONFIRMED_MODEL`、
`wire_continuous_distance_ready=True`，其全部 segment 的 `segment_distance_ready=True`。
任一 gap、fallback、review、failed、non-ready segment 使 authority invalid。

EMPTY mode 的 `w1_artifact_components=()` 且 `approved_wire_ids=()`；这是唯一可证明
没有 authoritative wire 的形式。没有 W1 output、没有 wire ID、空检测结果或 unreadable
W1 都不能替代 EMPTY authority。

authority manifest 不包含自身 file SHA，避免 self-reference。`approval_digest` 是审批系统
对 manifest 中除 `approval_digest`、`manifest_digest` 外全部字段 canonical payload 产生的
lowercase SHA；`manifest_digest` 对
`{"digest_domain":"FRVWIRAUTH1", ...all fields except manifest_digest}` 调用
`frv_digest`。runtime reference 精确字段为：

```python
@dataclass(frozen=True, slots=True)
class FrvWireAuthorityReference:
    manifest: FrvWireAuthorityManifest
    authority_manifest_path: Path
    authority_manifest_component: FrvA2ArtifactComponentBinding
    w1_json_path: Path | None
    w1_npz_path: Path | None
```

`authority_manifest_component` 必须 kind=`MANIFEST_JSON`、key=`authority_manifest_json`，
并在打开前后验证 SHA/size。formal P-WIRE component tuple 是该 component 加
`manifest.w1_artifact_components` 后按 component kind 与 key 的 A2 canonical order 排序。
因此 BND1 绑定 authority JSON 和 W1 pair，而 authority JSON bytes 不参与自身 manifest
payload。

W1 `sampled_curve_xyz` 是 local float32 authority。formal absolute comparison coordinate 是
`float64(sampled_curve_xyz) + origin_xyz`。每个 sampled polyline 相邻点构成闭线段；零长度
edge 被拒绝。`FRVWIRGEO1` digest 按 `(wire_id, segment_index, edge_index)` 排序，framing 为
ASCII `FRVWIRGEO1\0`、两个 source SHA+NUL、float64 origin 三元组、uint64 edge count，
再逐 edge 写 `<iii6d`。EMPTY mode 的 edge count 为零。

### 4.2 P-WIRE config 和 API

```text
FRV_WIRE_PROVIDER_KEY = "wire_provider"
FRV_WIRE_PROVIDER_INTERFACE_VERSION = "frv-wire-provider-v1"
FRV_WIRE_PROVIDER_SCHEMA_VERSION = "frv-wire-provider-result-v1"
FRV_WIRE_CONFIG_SCHEMA_VERSION = "frv-wire-provider-config-v1"
FRV_WIRE_PROOF_SCHEMA_VERSION = "frv-wire-provider-proof-v1"
FRV_WIRE_ALGORITHM_VERSION = "frv-wire-polyline-distance-grid-v1"
FRV_WIRE_CONFIG_DIGEST_DOMAIN = "FRVWIRCFG1"
FRV_WIRE_POINT_CONTENT_DIGEST_DOMAIN = "FRVWIRPCD1"
FRV_WIRE_PROOF_DIGEST_DOMAIN = "FRVWIRPRF1"
```

```python
@dataclass(frozen=True, slots=True)
class FrvWireProviderConfig:
    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    wire_body_possible_radius_m: float
    vegetation_conflict_radius_m: float
    proximity_radius_m: float
    spatial_index_cell_size_m: float
    max_edges_per_index_cell: int
    provider_config_digest: str
```

默认 radius 为 `0.10, 0.50, 1.50` m，index cell 为 `1.50` m，
`max_edges_per_index_cell=100000`；三个 radius 必须严格递增。config digest 用
`FRVWIRCFG1` 加除 self-digest 外全部字段。

```python
def load_frv_wire(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    authority_reference: FrvWireAuthorityReference,
    *,
    frv_configuration: FrvConfiguration,
    provider_config: FrvWireProviderConfig,
    source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvWireLoadSession
```

`FrvWireAuthorityReference` 是 runtime-only frozen type，含 authority manifest 和 authority
manifest/W1 pair 的绝对访问 path；path 不进 digest。W1 mode 只通过现有
`load_wire_models` 读取并验证 JSON/NPZ，随后再次核对 component SHA/size、source、origin
和 geometry digest。EMPTY mode 禁止 W1 path。

prevalidation 还要求 provider interface/config digest 分别等于
`frv_configuration.wire_provider_interface_version` 和
`frv_configuration.wire_provider_config_digest`；production 中二者均 nonnull。source result、
reference、configuration、authority 和 SourceIdentity 全部通过后才允许建立 geometry
index 或打开 source。

P-WIRE 恰好建立一次 P-SOURCE session。geometry index 在打开 source 前从已验证 authority
构建；source stream 每行恰好决策一次并一对一产生 `FrvWireChunk`。stream 必须自然耗尽、
finalize、关闭，并把新 source result 的 proof、SourceIdentity、components、index digest 和
BND1 与 supplied source result 完全相等比较后，才允许 P-WIRE finalize。不得第二次 source
pass、不得直接读 LAS、不得从 A4 discrete assignment 推断 full-resolution relation。

### 4.3 distance、relation、state

每条 polyline edge 建立 3D AABB，向每轴扩张 1.50 m 后插入 1.50 m half-open grid。
cell key anchor 是 source proof minimum bounds。单 cell edge 数超过 100000 时 provider
fail closed。

点到 edge `[a,b]` 的 float64 距离：

```text
u = b-a
t = clip(dot(p-a,u)/dot(u,u), 0.0, 1.0)
q = a+t*u
d2 = dot(p-q,p-q)
```

候选按 `(d2, wire_id, segment_index, edge_index)` 排序，第一项是 influence-query nearest。
relation：

```text
d <= 0.10 -> WIRE_BODY_POSSIBLE
0.10 < d <= 0.50 -> WIRE_VEGETATION_CONFLICT
0.50 < d <= 1.50 -> WIRE_PROXIMITY_ONLY
d > 1.50 -> OUTSIDE_INFLUENCE_DOMAIN
```

公开 point relation enum 值为：

```text
OUTSIDE_INFLUENCE_DOMAIN = 0
WIRE_BODY_POSSIBLE = 2
WIRE_PROXIMITY_ONLY = 3
WIRE_VEGETATION_CONFLICT = 4
```

它故意没有 `WIRE_BODY_CONFIRMED`。非空 authority 的 provider state 全部为
`WIRE_RELATION_VALID`；EMPTY authority 全部为 `WIRE_NOT_APPLICABLE_PROVEN`、relation=0、
wire/segment/edge id=-1、distance=+0.0。非空 authority 若 index query 无 candidate，或
best distance >1.50 m，则 relation=OUTSIDE、三个 ID=-1、distance=+0.0；这个显式 sentinel
表示“不在 influence domain”，不声称全局 nearest distance。三种 in-domain relation 才
保存 measured nearest IDs/distance。

### 4.4 P-WIRE contracts、digests 和 BND1

`FrvWireChunk` 字段：source/chunk scalars、`original_index <i8[P]`、
`provider_state |u1[P]`、`relation_state |u1[P]`、`nearest_wire_id <i4[P]`、
`nearest_segment_index <i4[P]`、`nearest_edge_index <i4[P]`、
`centerline_distance_m <f8[P]`、`reason_flags <u4[P]`、`quality_flags <u4[P]`、`is_last`。
数组全部 C-contiguous、read-only、finite，严格覆盖 0..N-1。

`FRVWIRPCD1` 使用与 geometry point digest 相同的 field-name/row-length framing，header
绑定 source content、source record、authority manifest、geometry content、provider config
五个 SHA 和 N；row fields 按上段顺序编码。

`FrvWireProviderProof` 的 exact field order 为：

```text
schema_version
provider_key
provider_role
interface_version
algorithm_version
configuration_digest
provider_config_digest
source_artifact_reference_id
artifact_reference_id
artifact_components
source_content_identity_digest
source_identity_record_digest
source_proof_digest
source_binding_digest
source_point_count
domain_kind
domain_count
original_index_domain_digest
authority_mode
authority_manifest_digest
wire_geometry_content_digest
wire_point_content_digest
approved_wire_count
wire_relation_valid_count
wire_not_applicable_proven_count
outside_influence_domain_count
wire_body_possible_count
wire_proximity_only_count
wire_vegetation_conflict_count
minimum_centerline_distance_m
maximum_centerline_distance_m
coordinate_frame
units
origin_xyz
transform_declaration
quality_flags
reason_flags
error_codes
proof_digest
```

两个 provider state count 之和为 N；四个 relation count 之和为 N；EMPTY authority 只允许
N 个 outside、两个 distance aggregate 都是正 `+0.0`；非空 authority 只允许
relation-valid state，distance aggregates 是所有 in-domain measured distances 的 finite
min/max；没有 in-domain point 时二者都是正 `+0.0`。

`FrvWireProviderResult` 是 private-capability frozen/slots/init=False type，exact fields 为
`proof`、`source_identity`、`_finalized_capability`。公开 validators 是
`validate_frv_wire_authority_manifest`、`validate_frv_wire_provider_config`、
`validate_frv_wire_chunk`、`validate_frv_wire_provider_proof`、
`validate_finalized_frv_wire_provider_result`、`to_frv_a2_provider_binding_evidence`。
`FrvWireLoadSession` 的五个 method 与第 3.1 节 session protocol 同签名和 single-use
lifecycle。public `FrvWireLoadError` exact fields 是 `failure_record` 和 `quality_flags`。

`binding_digest` 只调用 existing `frv_a2_provider_binding_digest`，digest domain 是
`FRVA2BND1`。BND1 是 formal WIRE、VALIDATED、FULL_SOURCE；
artifact components 是 authority reference 的完整 canonical tuple；upstream digest 是
排序唯一的 authority manifest、wire geometry、wire point content、source BND1 和 source
proof digest。provider success aggregate flags 为 NONE/empty。

### 4.5 P-WIRE failure mapping

| Condition | Phase | field | error | reason | quality |
|---|---|---|---|---|---|
| authority manifest/component/layout invalid | STRUCTURE_INTEGRITY | `wire_authority` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| inventory not complete or approval invalid | PRODUCTION_POLICY | `wire_inventory` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| source/source-record/origin/frame mismatch | IDENTITY_CONTEXT | `wire_authority` | `FRV_WIRE_IDENTITY_MISMATCH` | `WIRE_CONFLICT` | `WIRE_IDENTITY_MISMATCH` |
| W1 missing, unreadable, hash mismatch | STRUCTURE_INTEGRITY | `wire_artifact` | `FRV_WIRE_GEOMETRY_UNAVAILABLE` | `PROVIDER_UNCERTAINTY` | `WIRE_GEOMETRY_UNAVAILABLE` |
| non-confirmed/non-continuous/non-ready W1 geometry | SEMANTIC_READINESS | `wire_geometry` | `FRV_WIRE_GEOMETRY_UNAVAILABLE` | `PROVIDER_UNCERTAINTY` | `WIRE_GEOMETRY_UNAVAILABLE` |
| zero/nonfinite edge or geometry digest mismatch | SEMANTIC_READINESS | `wire_geometry` | `FRV_WIRE_GEOMETRY_UNAVAILABLE` | `PROVIDER_UNCERTAINTY` | `WIRE_GEOMETRY_UNAVAILABLE` |
| edge-cell limit exceeded | SEMANTIC_READINESS | `wire_index` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| source stream early/read/count failure | DOMAIN_ALIGNMENT | `source_pass` | `FRV_STREAM_INCOMPLETE` | `PROVIDER_UNCERTAINTY` | `STREAM_INCOMPLETE` |
| lost/out-of-range original index | DOMAIN_ALIGNMENT | `original_index` | `FRV_ORIGINAL_INDEX_INVALID` | `PROVIDER_UNCERTAINTY` | `ORIGINAL_INDEX_INVALID` |
| duplicate original index | DOMAIN_ALIGNMENT | `original_index` | `FRV_ORIGINAL_INDEX_DUPLICATE` | `PROVIDER_UNCERTAINTY` | `ORIGINAL_INDEX_DUPLICATE` |
| nonfinite distance | SEMANTIC_READINESS | `centerline_distance` | `FRV_PROVIDER_INVALID` | `PROVIDER_UNCERTAINTY` | `PROVIDER_INVALID` |
| proof/BND1 mismatch | CROSS_PROVIDER | `digest` | `FRV_ARTIFACT_DIGEST_MISMATCH` | `PROVIDER_UNCERTAINTY` | `ARTIFACT_DIGEST_MISMATCH` |

provider failure 不产生 point relation、proof、result 或 BND1。有效的三种 in-domain point
relation 不是 provider failure；membership 必须把它们路由到 review。

## 5. A2 runtime bundle

Session B 新增 `algorithms/vegetation/frv_inputs.py`，只做 live finalized provider 绑定：

```python
@dataclass(frozen=True, slots=True)
class ValidatedFrvA2Inputs:
    source_result: FrvSourceProviderResult
    hag_result: FrvHagProviderResult
    geometry_result: FrvGeometryProviderResult
    wire_result: FrvWireProviderResult
    readiness_record: FrvA2ReadinessRecord
```

```python
def validate_frv_a2_inputs(
    source_result: FrvSourceProviderResult,
    hag_result: FrvHagProviderResult,
    geometry_result: FrvGeometryProviderResult,
    wire_result: FrvWireProviderResult,
    *,
    frv_configuration: FrvConfiguration,
) -> ValidatedFrvA2Inputs
```

它调用四个 finalized validator 和四个 BND1 adapter，按 `frv_a2.py` 六 phase 生成
readiness。生产配置只接受 `READY_FOR_A3`；任何 blocked/review provider 不返回 partial
bundle。对象只保存原对象引用，无 full-size ndarray copy。它不重开 source/provider，
不选择 path，不构造 scope，不决策 membership。

本 bundle 是同一 execution 内的能力对象；本阶段不增加 P-HAG/P-GEOMETRY/P-WIRE proof
持久化格式或跨进程 provider reload API。正式跨进程边界是随后发布并自验证的 membership
artifact 与 V4 handoff artifact。

## 6. Scope inventory 和 routing 冻结

### 6.1 正式 supporting input

`FrvScopeEvidenceReference` 是 runtime-only frozen type，字段为 A1、V1、V2 三个 JSON/NPZ
pair path，加六个 exact component bindings。它必须：

- 通过 `PreprocessingResult.from_dict` 并严格验证 A1 arrays；
- 通过现有 `load_vegetation_candidates`；
- 通过现有 `load_vegetation_instances`；
- 验证六个 SHA/size；
- 验证 V1 provenance 中 A1 pair SHA；
- 验证 V2 manual log 的 V1 NPZ SHA 和 V1/V2 source、manifest、origin；
- 验证 A1/V1/V2 `original_index`、`sample_index`、XYZ 和 cluster mapping；
- 验证 source ID、origin 与 `SourceIdentity` 完全一致。

A1 v1 scope authority 固定要求 `voxel_size_m == 0.05`、voxel key dtype int64、shape
`[M,3]`、unique original index，且
`voxel_key == floor((source_xyz-origin_xyz)/0.05)`。不允许从 V1/V2 label 扩散 point
membership；A1 voxel 只创建 scope。

### 6.2 canonical inventory

canonical rows 的唯一构造规则：

1. 每个 V2 cluster status 属于 `CONFIRMED`、`PROVISIONAL`、`REVIEW_REQUIRED` 或
   `REJECTED_AS_INSTANCE` 时创建一行，key 为 `V2:{cluster_id:010d}`；
2. 每个 V1 `vegetation_level != REJECTED` 且 `cluster_id_by_point == -1` 的行创建一行，
   key 为 `V1:{original_index:020d}`；
3. 一个 V1 point 已在 V2 cluster 中时不得再创建 V1 行；
4. `UNASSIGNED` 不是 V2 cluster row；
5. current repository 的普通 V1/V2 rejection 不满足 approved hard rejection；
6. 只有调用者提供且通过现有 `FRVApprovedHardRejectionEvidence` 完整校验的 canonical row
   才进入 `APPROVED_HARD_REJECTION`，其余全部进入 `REQUIRED_FRV`。

V2 cluster 的 member row 只从 `cluster_member_offsets` 和
`ordered_cluster_member_row_index` 读取；每个 row index 必须在 V2 point arrays 范围内且
恰好属于对应 `cluster_id_by_point`。V1 unclustered 判定使用 V2 point-aligned
`cluster_id_by_point == -1` 与 V1 `vegetation_level`，不使用 XYZ proximity 修复映射。

canonical record digest 是 `frv_digest` 对 source digest、canonical key、type、status、
排序唯一 member original indexes、A1/V1/V2 component digests 的 exact object。canonical
inventory digest 是按 key 排序的 record digest tuple；required digest 对 required rows
采用相同顺序。每行出现一次且只出现一次。

### 6.3 scope core、margin、union、tile

`FrvScopePolicyConfig` 是 exact frozen/slots dataclass，字段顺序为
`schema_version`、`voxel_size_m`、`context_margin_xy_cells`、
`context_margin_z_cells`、`tile_edge_cells`、`feature_halo_cells`、`boundary_rule`、
`coordinate_canonicalization`、`geometry_digest_domain`、`policy_digest`；
`policy_digest` 对 `{"digest_domain":"FRVSCPPOL1", ...all preceding fields}` 调用
`frv_digest`，并必须等于 `FrvConfiguration.scope_policy_digest`：

```text
schema_version = "frv-scope-policy-v1"
voxel_size_m = 0.05
context_margin_xy_cells = 30
context_margin_z_cells = 30
tile_edge_cells = 60
feature_halo_cells = 30
boundary_rule = "min-inclusive-max-exclusive"
coordinate_canonicalization = "binary64-subtract-origin-once"
geometry_digest_domain = "FRVSCPGEO1"
```

每个 required canonical row 的 core 是其所有 member 对应 A1 voxel key 的排序唯一集合。
任一 member 无唯一 A1 key、越 source index、identity 不合或 core 空，row 为
`REGION_MAPPING_INCOMPLETE` 且不创建 scope。

context margin 是 core key 在 x/y/z 各扩张闭整数 `[-30,+30]`。union 用 run-length
encoding：按 `(ix,iy)` 聚合连续 z cells，记录 `(ix,iy,z_start,z_stop_exclusive)`，排序
严格为 tuple lexicographic，重叠和相邻 run 合并。`FRVSCPGEO1` framing 为 role ASCII+NUL、
source digest+NUL、float64 origin、float64 voxel size、uint64 run count、逐 run `<qqqq`。
CORE、CONTEXT、UNION role 各有不同 role string，因此三个 content digest 不相等。

source point 转 local coordinate 一次：`local = float64(xyz)-float64(origin)`；cell key 为
`floor(local/0.05)`。formal containment 只查询 union runs。tile key 为
`floor(cell_key/60)`；每点只属于一个 core tile。P-GEOMETRY 已用 FULL_SOURCE halo 计算
features，scope tile 不重复计算 geometry；`feature_halo_cells=30` 只验证二者 policy 一致。

每个 in-union point 收集所有覆盖它的 canonical scope key，排序唯一；首个是
`primary_scope_id`，完整 tuple 写 supporting-region ragged table。没有 supporting scope 的
点不可能进入 union。

### 6.4 coverage result

scope producer构造现有 `FRVScopeDefinition`、`FRVScopeCoverageRecord` 和
`FRVScopeCoverageManifest`。第一行成功 scope 为 `SCOPE_CREATED`；后续与既有 union 相交
或重复的 row 为 `SCOPE_MERGED`；两者都为 `COVERED`。blocked row 使用现有 blocked state
和非零 failure flags。

若 P-WIRE authority 在 geometry 完整性建立前失败，无法证明受影响 influence domain，
因此该 source 的全部 required rows fail closed；这满足 frozen design 的“affected domain”
规则，因为 unknown domain 的保守受影响范围是整个 source。valid EMPTY authority 不阻塞。

`frv_routing_complete` 的唯一来源是现有 manifest `routing_complete`：

```text
canonical = required + approved
required = created_or_merged + blocked
frv_routing_complete = (blocked == 0) and both equations validated
```

blocked manifest 不进入 membership publication；独立 source 可继续。

## 7. Membership decision 精确冻结

### 7.1 policy 和 inputs

`FrvMembershipPolicyConfig` 是 exact frozen/slots dataclass，字段顺序为
`schema_version`、`algorithm_version`、`decision_rule_version`、
`ordinary_min_hag_m`、`upper_seed_min_hag_m`、`ground_max_abs_raw_hag_m`、
`propagation_min_hag_drop_m`、`max_structural_rounds`、`ground_classification_value`、
`output_block_record_count`、`policy_digest_domain`、`policy_digest`。
`policy_digest` 对 policy domain 加除 self-digest 外全部字段调用 `frv_digest`，并必须等于
`FrvConfiguration.feature_policy_digest`：

```text
schema_version = "frv-membership-policy-v1"
algorithm_version = "frv-membership-synchronous-v1"
decision_rule_version = "frv-membership-decision-v1"
ordinary_min_hag_m = 0.30
upper_seed_min_hag_m = 1.00
ground_max_abs_raw_hag_m = 0.10
propagation_min_hag_drop_m = 0.05
max_structural_rounds = 12
ground_classification_value = 2
output_block_record_count = 250000
policy_digest_domain = "FRVMEMPOL1"
```

API：

```python
def build_frv_membership(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    scope_evidence: FrvScopeEvidenceReference,
    wire_authority: FrvWireAuthorityReference,
    overrides: tuple[FRVManualOverrideArtifact, ...],
    *,
    frv_configuration: FrvConfiguration,
    hag_config: FrvHagProviderConfig,
    geometry_config: FrvGeometryProviderConfig,
    wire_config: FrvWireProviderConfig,
    scope_policy: FrvScopePolicyConfig,
    membership_policy: FrvMembershipPolicyConfig,
    output_dir: Path,
    scratch_dir: Path,
    source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvMembershipExecutionResult
```

orchestrator 依次运行并 spool P-HAG、P-GEOMETRY、P-WIRE point chunks，finalize 后创建
`ValidatedFrvA2Inputs`。它另创建一次 P-SOURCE pass，把 XYZ、classification 和 formal
scope hit 写入 original-index keyed scratch store。总 source pass 是：P-HAG 2、
P-GEOMETRY 2、P-WIRE 1、membership 1；恰好 6 次。V4 recovery 是后续单独第 7 次。

正式 publication 前必须要求 configuration purpose=`PRODUCTION_APPROVED`、
scope mode=`EXHAUSTIVE_REGION`、`configuration_is_runtime_complete=True`、
`formal_completeness_allowed=True`、`membership_artifact_publication_allowed=True`，且
algorithm/decision/scope policy versions 与本节常量相等。TEST_ONLY 或
CALIBRATION_CANDIDATE 可用于 directed provider tests，但 `build_frv_membership` 必须以
`PRODUCTION_POLICY/FRV_PROVIDER_INVALID` 拒绝正式 artifact 输出。

### 7.2 base evidence 和 structural solver

对每个 in-scope original index，先计算与 final state 无关的 base evidence：

```text
scan_adequate := geometry state VALID and medium/large neighbor minimum met
wire_clear := provider state is WIRE_NOT_APPLICABLE_PROVEN
    or relation is OUTSIDE_INFLUENCE_DOMAIN
geometry_positive := CANOPY_BRANCHING_COMPATIBLE or WOODY_COLUMN_COMPATIBLE
ground_counter_evidence := CANOPY_BRANCHING_COMPATIBLE
    or WOODY_COLUMN_COMPATIBLE or structural_supported
```

upper seed 必须同时满足：

```text
HAG_VALID_POSITIVE and clamped_hag_m >= 1.00
and geometry_positive and DENSITY_SUPPORTED and scan_adequate
and classification != 2 and wire_clear
```

P-GEOMETRY mutual support edges 是唯一 base adjacency；vertex id 恰为 original_index。
round 0 只含 upper seeds。round `k=1..12` 同步读取前一 immutable bitmap；目标点可加入
support 当且仅当：

- 目标尚未 supported；
- 目标为 HAG positive/near-zero/within-negative-tolerance；
- 目标有 `WOODY_COLUMN_COMPATIBLE` 和 `scan_adequate`；
- 目标 classification != 2 且 `wire_clear`；
- 至少一个前一轮 supported 邻点满足
  `neighbor_clamped_hag >= target_clamped_hag + 0.05`。

同一 round 的新点不能支持同一 round 的其他点。round 无新增即稳定停止。第 12 轮后若
仍存在可由第 12 轮新增点继续触达的合法目标，则该 connected affected set 标记 solver
nonconverged，全部进入 review；不扩大 max rounds。

### 7.3 automatic decision 的唯一顺序

provider/source/scope/artifact integrity failure 是 coverage failure，不创建普通 point state。
健康 scope 内每个 unique original_index 正好执行一次以下 ordered decision：

1. 任一 HAG state 为 MISSING/INVALID/OUT_OF_DOMAIN -> `REVIEW_REQUIRED_MEMBERSHIP`；
2. geometry state 非 VALID -> review；
3. P-WIRE relation 为 BODY_POSSIBLE、PROXIMITY_ONLY 或 VEGETATION_CONFLICT -> review；
4. geometry 同时包含 ground-compatible 与 canopy/woody -> review；
5. structural solver nonconverged -> review；
6. classification==2、`abs(raw_hag_m)<=0.10`、ground-compatible、无 canopy/woody、
   无 structural support、scan adequate、wire clear -> `EXCLUDED_NON_CANDIDATE`；
7. HAG positive、`clamped_hag_m>=0.30`、geometry_positive、density-supported、
   scan adequate、classification!=2、wire clear -> `VERIFIED_VEGETATION_CANDIDATE`；
8. HAG 为 near-zero/within-tolerance，或 positive 且 clamped<0.30，且 structural-supported、
   woody-column、scan adequate、classification!=2、wire clear -> verified candidate；
9. 其他情况 -> review。

缺少正证据永不产生 exclusion。wire 在本版本永不产生 exclusion。near-zero HAG 单独永不
产生 ground exclusion。每个 automatic state 同时生成现有 `FrvReasonFlag`：verified 正常
为 NONE；ground exclusion 为 GROUND_CONFLICT；wire 为 WIRE_CONFLICT|REVIEW_REQUIRED；
HAG/geometry不足使用对应 HAG flag 或 INSUFFICIENT_EVIDENCE，再加 REVIEW_REQUIRED；
evidence conflict 使用 VEGETATION_COUNTER_EVIDENCE|REVIEW_REQUIRED；nonconvergence 使用
PROVIDER_UNCERTAINTY|REVIEW_REQUIRED。input quality 只使用现有、与事实对应的 flags。

`segmentation_hag_readiness=READY` 仅当 final membership 是 VERIFIED 或 MANUAL_INCLUDE，
HAG state 是 POSITIVE/NEAR_ZERO/WITHIN_NEGATIVE_TOLERANCE，且 HAG point quality 不含
PROVIDER_INVALID。其他 point 全为 REVIEW_REQUIRED。

payload `input_quality_flags` 是该 point 的 P-HAG `FrvQualityFlag` 原值；健康 P-GEOMETRY/
P-WIRE 的 provider-specific cap/relation flags 不强塞进该 enum。override artifact integrity
failure 另加 `OVERRIDE_ARTIFACT_FAILURE`，但这种 failure 阻止 artifact publication。manual
include/exclude 的 final reason 分别是 automatic reason OR
`MANUAL_INCLUDE_PROVENANCE`/`MANUAL_EXCLUDE_PROVENANCE`；override chain conflict 是
automatic reason OR `OVERRIDE_CHAIN_CONFLICT|REVIEW_REQUIRED`。

### 7.4 override

先计算 automatic state，再验证 existing `FRVManualOverrideArtifact` 中保存的
`automatic_state_before_override` 和 reason 与计算结果一致。override tuple 先调用现有
descriptor/chain validators；geometry request 在本版本不执行，出现 geometry request 时该
request rejected 并保留 automatic state。

对每个 original index：只看显式包含它的 artifacts；root predecessor 为 null；successor
必须逐 digest 连成单链；有效 leaf 是唯一 approved、无 conflict、未被 successor supersede
的 leaf。唯一 leaf include/exclude 分别产生 MANUAL_INCLUDE/MANUAL_EXCLUDE。没有 approved
leaf 或只有 rejected request 保留 automatic state。cycle、missing predecessor、同一
predecessor 多 approved successor、多 effective leaf、artifact digest failure 产生
OVERRIDE_CHAIN_CONFLICT|REVIEW_REQUIRED，不能产生 manual terminal。manual exclude 不对
manual include 具有额外优先级。

### 7.5 守恒、result 和 failure

```text
unique_in_scope_point_count
= VERIFIED + EXCLUDED + REVIEW + MANUAL_INCLUDE + MANUAL_EXCLUDE
evaluated_point_count = unique_in_scope_point_count
unasserted_in_scope_count = 0
```

`FrvMembershipExecutionState(str, Enum)` 的 exact values 是
`"ready_for_v4"`、`"no_relevant_frv_scope"`、`"coverage_blocked"`。

```python
@dataclass(frozen=True, slots=True)
class FrvMembershipOutputReference:
    root_path: Path
    execution_result_path: Path
    payload_schema_path: Path | None
    coverage_manifest_path: Path
    membership_manifest_path: Path | None
    binding_path: Path | None
    block_paths: tuple[Path, ...]
    decision_evidence_manifest_path: Path | None
    decision_evidence_block_paths: tuple[Path, ...]
```

该 reference 是 runtime access information，不进入 semantic digest。正式 result 的 exact
field order 为：

```python
@dataclass(frozen=True, slots=True)
class FrvMembershipExecutionResult:
    execution_state: FrvMembershipExecutionState
    source_identity: SourceIdentity
    a2_readiness: FrvA2ReadinessRecord | None
    coverage_manifest: FRVScopeCoverageManifest
    membership_artifact: FullResolutionVegetationMembershipArtifact | None
    payload_schema: FRVArtifactPayloadSchema | None
    artifact_binding: FRVArtifactBindingRecord | None
    decision_evidence_artifact: FrvDecisionEvidenceArtifact | None
    source_proof_digest: str
    hag_proof_digest: str | None
    geometry_proof_digest: str | None
    wire_proof_digest: str | None
    provider_binding_digests: tuple[str, ...]
    override_digests: tuple[str, ...]
    failure_records: tuple[FrvA2FailureRecord, ...]
    output_reference: FrvMembershipOutputReference | None
    result_digest: str
```

`result_digest` 对 domain `FRVMEMRES1` 加除 `output_reference`、`result_digest` 外全部字段的
canonical projections 调用 `frv_digest`；path 不入 digest。failure records 使用 A2 total
order。result 不持有 writable full-size arrays。

```python
def load_frv_membership_result(root_path: Path) -> FrvMembershipExecutionResult
```

只加载 READY 或 NO_SCOPE published result；重新验证 execution JSON、coverage、membership、
payload schema、binding、decision evidence 和所有 block bytes。BLOCKED result 不发布目录，
只能由 live execution 返回。

READY 必须有 artifact/schema/binding/decision evidence、无 failure，且 existing artifact validator 所有
completeness bool 为 true。BLOCKED 必须 artifact/schema/binding/decision evidence/output 全 null、failure
非空、`frv_routing_complete=False`、`frv_membership_decision_complete=False`。NO_SCOPE 必须
空 manifest 通过、无 artifact、无 failure，routing complete true，但没有 membership
decision claim。

expected coverage failure mapping：

| Condition | coverage state / error |
|---|---|
| A1/V1/V2 source or digest mismatch | `SOURCE_IDENTITY_FAILED` / `FRV_SOURCE_IDENTITY_FAILED` |
| upstream schema, pair or provenance invalid | `UPSTREAM_EVIDENCE_INVALID` / `FRV_PROVIDER_INVALID` |
| missing A1 voxel/member mapping | `REGION_MAPPING_INCOMPLETE` / `FRV_SCOPE_MAPPING_INCOMPLETE` |
| invalid run geometry/overflow/nonfinite boundary | `REGION_GEOMETRY_INVALID` / `FRV_SCOPE_GEOMETRY_INVALID` |
| any formal provider unavailable | `SCOPE_BLOCKED` / 原 provider 的 validated `FrvA2FailureRecord` 原值传递，不改 phase/field/error/reason |
| membership source pass incomplete | `SCOPE_BLOCKED` / `FRV_STREAM_INCOMPLETE` |
| point duplicate | `SCOPE_BLOCKED` / `FRV_ORIGINAL_INDEX_DUPLICATE` |
| point lost/out of range | `SCOPE_BLOCKED` / `FRV_ORIGINAL_INDEX_INVALID` |
| payload block digest failure | `SCOPE_BLOCKED` / `FRV_BLOCK_DIGEST_MISMATCH` |
| membership manifest digest failure | `SCOPE_BLOCKED` / `FRV_MANIFEST_DIGEST_MISMATCH` |
| artifact/binding/decision-evidence digest failure | `SCOPE_BLOCKED` / `FRV_ARTIFACT_DIGEST_MISMATCH` |

blocked coverage-row `failure_reason_flags` 固定为：source/upstream/provider/stream/artifact
类使用 `PROVIDER_UNCERTAINTY|REVIEW_REQUIRED`；mapping/geometry 类使用
`SCOPE_EDGE_CONFLICT|REVIEW_REQUIRED`；wire identity/geometry 类使用
`WIRE_CONFLICT|PROVIDER_UNCERTAINTY|REVIEW_REQUIRED`。成功 row 必须 NONE。

## 8. FRV artifact publication

`FrvDecisionEvidenceArtifact` 是 membership schema 之外、与其 digest 绑定的只读 evidence
sidecar；它不能单独证明 membership。exact fields 为：

```text
schema_version
source_content_identity_digest
source_identity_record_digest
membership_artifact_manifest_digest
hag_proof_digest
geometry_proof_digest
wire_proof_digest
original_index_domain_digest
record_count
block_digests
evidence_digest
```

schema=`frv-decision-evidence-v1`；record_count 等于 membership evaluated count；block digest
tuple 按 block ordinal 排序。每个 evidence block 与 membership block 拥有完全相同的
original indexes，并按顺序存 `original_index <i8`、`geometry_evidence_state |u1`、
`geometry_evidence_flags <u4`、`wire_provider_state |u1`、`wire_relation_state |u1`、
`nearest_wire_id <i4`、`nearest_segment_index <i4`、`centerline_distance_m <f8`。block framing
是 ASCII `FRVEVDBLK1\0` 后接与 FRVBLK1 相同的 canonical header 和 field framing；
`evidence_digest` 对 domain `FRVEVD1` 加除自身外全部 metadata 调用 `frv_digest`。
V4 必须同时验证 membership block indexes、evidence block indexes 和三个 provider proof
digests；sidecar 缺失或不一致是 `FRV_ARTIFACT_INVALID`。

Session B 使用 existing `FRVArtifactPayloadSchema` exact required arrays，并总是包含允许的
optional arrays：`raw_hag`、`clamped_hag`、`hag_quality_flags`、`hag_reason_flags`；
`candidate_confidence` 不发布。reference tables 严格排序唯一；manual table 无有效 override
时为 null。`manual_override_reference=0xffffffff` 表示无引用；其他 u32 是 table index。

`scope_union_reference_table=(scope_union_id,)`，所以每点
`scope_union_id_reference=0`。`primary_scope_reference_table` 是所有 primary scope ID 的
排序唯一 tuple，point value 是对应 table index。为在既有 payload schema 内无损保留任意
supporting-scope 组合，`supporting_region_reference_table` 的每项是完整 scope-ID tuple 的
`canonical_frv_json_bytes` UTF-8 string；table 对这些 strings 排序去重。每点
`supporting_region_offset` 是其 tuple-string 的 table index，
`supporting_region_count=1`。因此一个 reference 可直接恢复完整 contributing regions，
不依赖隐含连续子区间。manual table 非空时保存 override digest strings，point u32 是 table
index。

block 以 original_index 递增分组，每 block 最多 250000 records；range 为首 index 到
末 index+1，block ID 为 `frv-block-{ordinal:08d}`。文件 extension 为 `.frvblk`。

block bytes：

```text
ASCII FRVBLK1\0
uint64 little-endian canonical block-header byte length
canonical_frv_json_bytes(block header)
for every payload array in schema order:
    uint16 little-endian field-name byte length
    UTF-8 field name
    uint64 little-endian raw byte length
    little-endian C-order raw bytes
```

block header exact keys 是 artifact_id、block_id、payload_schema_digest、record_count、
original_index_start、original_index_stop。`block_digest=sha256(complete block bytes)`；
descriptor storage reference 是 `blocks/<block_id>.frvblk`。

artifact `artifact_sha256` 的 `FRVART1` domain：

```text
ASCII FRVART1\0
uint64 canonical declared-header byte length
canonical_frv_json_bytes(declared header)
for each canonical block descriptor:
    uint16 block-id byte length, UTF-8 block-id
    32 raw bytes of block_digest
    32 raw bytes of block_descriptor_digest
```

先在 output sibling temp directory 写完所有 membership/evidence block、payload schema JSON、coverage JSON、
membership manifest JSON、binding JSON、decision-evidence JSON 和 execution-result JSON，重新读取验证全部 bytes 后以单次 directory rename
发布。目标 output_dir 必须不存在；不支持 overwrite。失败删除 temp directory，目标保持
不存在。manifest 自身用 canonical JSON UTF-8、无 BOM、末尾单个 LF；raw physical path、
mtime、host 不进入 semantic digest。

## 9. V4 input handoff

### 9.1 contracts 和 API

新合同模块 `core/contracts/v4_inputs.py` 公开：

```text
V4PointRecoveryState:
  POINT_RECOVERY_FAILED=1
  POINT_RECOVERED_HAG_READY=2
  POINT_RECOVERED_HAG_REVIEW_REQUIRED=3
  POINT_RECOVERED_HAG_MISSING=4
  POINT_RECOVERED_HAG_INVALID=5

V4PrimaryRoute:
  NORMAL_SEGMENTATION=1
  MEASURED_POINT_REVIEW=2
  UNRECOVERED_REGION_REVIEW=3
  EXCLUDED_AUDIT_ONLY=4
```

`V4ValidatedFrvMembershipView` 是 read-only consumer view，exact fields 为
`source_identity`、`frv_configuration_digest`、`coverage_manifest`、
`membership_artifact`、`payload_schema`、`artifact_binding`、
`decision_evidence_artifact`、`membership_block_references`、
`decision_evidence_block_references`、八个 `frv_` inventory/scope/point counts、五个
`frv_` membership state counts、六个 inherited completeness bool 和 `view_digest`。
counts/bools 的 exact names 与下方 coverage report 同名；view digest domain 是 `V4FRVVIEW1`。
它不得重算或放宽 FRV facts。

`V4RecoveredVegetationSet` 字段：

```text
source_identity
original_index <i8[N]
xyz_local_m <f8[N,3]
hag_m <f4[N]
membership_state |u1[N]
hag_evidence_state |u1[N]
segmentation_hag_readiness |u1[N]
candidate_reason_flags <u4[N]
candidate_confidence: None
frv_membership_artifact_digest
recovered_index_digest
```

`V4UnresolvedMeasuredPointSet` 保存 recovered review/HAG-not-ready real points，exact fields
是 `source_identity`、`original_index <i8[R]`、`xyz_local_m <f8[R,3]`、
`membership_state |u1[R]`、`hag_evidence_state |u1[R]`、
`segmentation_hag_readiness |u1[R]`、`raw_hag_m <f8[R]`、
`clamped_hag_m <f8[R]`、`reason_flags <u4[R]`、`quality_flags <u4[R]`、
`wire_provider_state |u1[R]`、`wire_relation_state |u1[R]`、
`nearest_wire_id <i4[R]`、`centerline_distance_m <f8[R]`、
`primary_route |u1[R]`、FRV artifact/provider digest references 和 set digest。
`primary_route` 全为 MEASURED_POINT_REVIEW。不含 tree ID。

`V4UnrecoveredRegionRecord` exact fields 是 `source_content_identity_digest`、
`frv_scope_coverage_manifest_digest`、`canonical_inventory_key`、`scope_id`、
`frv_scope_union_id`、`scope_union_geometry_digest`、`failure_reason_flags`、
`safety_relevance`、`review_required`、`measured_clearance_complete`、
`diagnostic_reference`、`record_digest`。scope/bounds未知时 scope fields 为 null；
`safety_relevance="UNKNOWN"`、review true、clearance false。它不含虚构 point/HAG/tree/
distance。

`V4RecoveryCoverageReport` 包含 design D05 列出的全部 `frv_` counts/bools，以及
source_identity_complete、xyz_recovery_complete、hag_ready_complete、
segmentation_input_complete、v4_routing_complete、measured_clearance_complete、
review_required；本阶段 `measured_clearance_complete` 固定 false，因为 clearance 尚未运行。

其 exact field order 为：

```text
schema_version
source_content_identity_digest
source_identity_record_digest
frv_scope_coverage_manifest_digest
frv_scope_union_id
frv_upstream_inventory_digest
frv_membership_artifact_digest
recovered_index_digest
frv_canonical_inventory_count
frv_required_inventory_count
frv_approved_hard_rejection_count
frv_scope_created_or_merged_count
frv_coverage_blocked_count
frv_unique_in_scope_point_count
frv_evaluated_point_count
frv_unasserted_in_scope_count
frv_verified_vegetation_candidate_count
frv_excluded_non_candidate_count
frv_review_required_membership_count
frv_manual_include_count
frv_manual_exclude_count
normal_segmentation_count
measured_point_review_count
excluded_audit_count
point_recovery_failed_count
unrecovered_region_count
frv_canonical_inventory_complete
frv_required_inventory_complete
frv_approved_hard_rejection_complete
frv_scope_coverage_complete
frv_routing_complete
frv_membership_decision_complete
source_identity_complete
xyz_recovery_complete
hag_ready_complete
segmentation_input_complete
v4_routing_complete
measured_clearance_complete
review_required
failure_codes
report_digest
```

schema 是 `v4-recovery-coverage-v1`；counts 是 nonnegative built-in int；failure_codes 是
按 enum declaration order 排序唯一 tuple。`report_digest` 对 domain `V4COV1` 加除自身外
全部字段调用 `frv_digest`。

`V4InputOutputReference` exact fields 是 root path、validated-view JSON path、coverage JSON
path、normal block paths、review block paths 和 handoff manifest path，全部 runtime-only。
`V4InputHandoffResult` exact fields 是 `validated_membership`、`recovered_vegetation`、
`unresolved_measured_points`、`unrecovered_regions`、`coverage_report`、`output_reference`、
`handoff_digest`；前三项在 coverage-only blocked handoff 中为 null，在 READY handoff 中
全部 nonnull；handoff digest domain 是 `V4INP1`，排除 output reference 和 self-digest。

```python
def build_v4_input_handoff(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    membership_result: FrvMembershipExecutionResult,
    *,
    frv_configuration: FrvConfiguration,
    output_dir: Path,
    source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> V4InputHandoffResult
```

READY membership 路径重新读取并验证所有 FRV files，然后恰好建立一次 P-SOURCE
session，按升序 artifact indexes merge-recover 原始 XYZ；session 必须自然耗尽、finalize
并与 supplied source result 完全相等。不得从 artifact block、A1/V1/V2 XYZ 或 nearest
neighbor 复制 measured XYZ。

COVERAGE_BLOCKED live result 只允许生成 coverage-only handoff：不打开 source、不创建
validated view/normal/measured point set，把每个 blocked required row 一对一转换成
`V4UnrecoveredRegionRecord`；所有 known rows 有 route 时 `v4_routing_complete=True`，但
`frv_routing_complete=False`、`frv_membership_decision_complete=False`、
`segmentation_input_complete=False`、`measured_clearance_complete=False`。NO_SCOPE 不创建
V4 handoff artifact，调用返回 `FRV_RESULT_NOT_READY`。

### 9.2 partition 和 completeness

```text
normal input = exact recovered rows where
  state in {VERIFIED_VEGETATION_CANDIDATE, MANUAL_INCLUDE}
  and segmentation_hag_readiness == READY

measured review = exact recovered rows where
  state == REVIEW_REQUIRED_MEMBERSHIP
  or admitted state with readiness != READY

excluded audit = EXCLUDED_NON_CANDIDATE or MANUAL_EXCLUDE
```

normal 与 measured review disjoint。所有 normal/review target index 必须 exact recovered；
任何缺失/duplicate/out-of-range 使 handoff fail closed。`xyz_local_m = source_xyz -
SourceIdentity.origin_context`，float64；`hag_m` 是 artifact `clamped_hag` 的 exact float32
cast，不重算、不再 clamp。

`recovered_index_digest` framing 为 ASCII `V4RECIDX1\0`、source content SHA+NUL、uint64 N、
逐 index `<Q`。`v4_routing_complete=True` 当且仅当每个 known point/region 恰有一个 primary
route。`segmentation_input_complete=True` 要求 inherited
`frv_membership_decision_complete`、identity/recovery complete 和 partition complete；N=0
本身不使该 bool false。`hag_ready_complete` 表示所有 normal-planned admitted point ready；
存在 HAG review 时为 false，但安全 route 仍可 complete。

V4 normal/review blocks 使用 field-name/byte-length/raw-C-order framing，magic 分别为
`V4NRMBLK1\0`、`V4REVBLK1\0`，每 block 最多 250000 rows。set digest 分别对 block digest
tuple 使用 domain `V4NRM1`、`V4REV1`。handoff digest 的 byte framing 是：

```text
ASCII V4INP1\0
UTF-8 source_content_identity_digest, NUL
UTF-8 frv_membership_artifact_digest or empty, NUL
UTF-8 coverage_report.report_digest, NUL
UTF-8 normal set digest or empty, NUL
UTF-8 review set digest or empty, NUL
uint64 unrecovered_region_count
for each canonical unrecovered record: 32 raw bytes record_digest
```

V4 handoff output 使用与 FRV block 相同的 temp-directory validate/rename discipline。它是
V4 input artifact，不是 segmentation result。公开 result 不含 `tree_id`、apex、crown、
CHM、slice 或 clearance 字段。

### 9.3 V4 handoff failure mapping

`V4InputLoadError` 使用 closed `V4InputErrorCode`：

```text
FRV_RESULT_NOT_READY
FRV_ARTIFACT_INVALID
SOURCE_IDENTITY_FAILED
SOURCE_STREAM_INCOMPLETE
ORIGINAL_INDEX_INVALID
ORIGINAL_INDEX_DUPLICATE
HAG_EVIDENCE_INVALID
ROUTING_CONSERVATION_FAILED
OUTPUT_PUBLICATION_FAILED
```

first failure wins；错误 detail 只含 code、stable field 和 source content digest，不含 path、
host、traceback 或 raw exception。失败不产生 partial `V4RecoveredVegetationSet` 或正式
output directory。

## 10. V4 Multi-Apex Tree Hypothesis（不授权实现）

本附录冻结 V4 后续设计方向，不为 Session A/B 增加代码、API、config、test 或 output。

1. apex candidate 是局部冠层高度证据，不等于一棵树；任何 apex 不得在本阶段生成
   `tree_id`。
2. 一个 future tree hypothesis 允许一个 main apex 和零个或多个 secondary apex。main
   apex 的结构性 tie order 固定为 HAG descending、局部 prominence descending、
   original_index ascending；这只是未来 hypothesis ordering，不产生当前结果。
3. future V4 先做轻度 over-segmentation，节点是 supervoxel 或 crown fragment；随后在
   fragment graph 上做 apex/crown merge。不得直接构建全点 optimization graph。
4. future merge evidence vector 必须同时保留：水平距离、高差、saddle drop、二维连通、
   gap evidence、三维 slice continuity、local geometry、optional trunk evidence。单一通道
   不得独立证明 merge。
5. trunk 只能是 optional strong prior；没有 trunk evidence 不得自动拒绝 tree hypothesis，
   也不得要求每棵树都有可见 trunk。
6. graph edge 只连接空间相邻且有至少一个非 trunk evidence 通道的 fragments；全点只在
   future final assignment 时作为 measured evidence 回映射。
7. future formal output 必须把 fragment/tree assignment 回映射到唯一 original_index，
   保留 secondary claims 和 unresolved route。
8. 本附录没有批准 executable merge score、numeric threshold、tree contract 或 calibration
   record。缺少这些不是可由实现者补齐的自由度，而是对本 bundle 的明确禁止：Session B
   到 V4 input handoff 即停止。

## 11. 依赖和 pyproject 冻结

当前 `pyproject.toml` 已直接声明 `numpy>=1.26,<3` 和 `laspy==2.7.0`。两次实施 session
均不修改 `pyproject.toml`，不新增 lock file。允许的 runtime 依赖仅为 Python standard
library、当前 NumPy、当前 laspy，以及调用 existing W1 loader 所需的仓库代码。

明确禁止新增或间接引入：PDAL、Open3D、scikit-learn/sklearn、SciPy、GPU framework、
network service、database、rasterio、shapely、pyproj、pandas，以及大型二进制 fixture。
测试 fixture 必须在 pytest temp directory 动态创建；单 fixture source 不超过 20000 点，
另设一个 200000 点 streaming/resource test，均不提交 LAS/NPZ/block binary。

production resource gate 固定为 16 GiB 主机上 peak RSS 不超过 14 GiB，保留至少 2 GiB
系统余量。geometry cache 固定 512 MiB；source/provider chunk 逐个释放；structural solver
只持有 in-scope bitmaps、frontier 和 artifact-backed `[N,16]` adjacency view，不构建 Python
object edge graph。scratch preflight 要求 free bytes 至少
`max(2147483648, 640 * source_point_count)`；不足时在打开第一 source pass 前 fail closed。

## 12. 两个连续 implementation sessions

### 12.1 Session A — P-GEOMETRY

Session A 只实现第 3 节并运行 directed tests。结束后保持同一 dirty worktree供 Session B
继续；不进行 Codex review、Git commit/push 或第二次授权。

Modified：

```text
core/contracts/__init__.py
algorithms/vegetation/__init__.py
```

New：

```text
core/contracts/frv_geometry.py
algorithms/vegetation/geometry_provider.py
tests/contracts/test_frv_geometry_contracts.py
tests/contracts/test_frv_geometry_serialization.py
tests/vegetation/test_geometry_provider.py
```

Forbidden：除上述 7 个文件外全部项目文件；尤其是 P-SOURCE、P-HAG、P-WIRE、FRV A1/A2
通用合同、membership、V4、`pyproject.toml`、`.claude/` 和任何 binary fixture。

Session A directed commands：

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_frv_geometry_contracts.py tests/contracts/test_frv_geometry_serialization.py tests/vegetation/test_geometry_provider.py -q
E:\Miniforge\envs\treeline\python.exe -m ruff check core/contracts/frv_geometry.py algorithms/vegetation/geometry_provider.py tests/contracts/test_frv_geometry_contracts.py tests/contracts/test_frv_geometry_serialization.py tests/vegetation/test_geometry_provider.py
```

### 12.2 Session B — P-WIRE + membership/routing + V4 handoff

Session B 从 Session A workspace 原地继续，不 reset、不 stash、不提交、不重新授权。

Modified：

```text
core/contracts/__init__.py
algorithms/vegetation/__init__.py
```

New：

```text
core/contracts/frv_wire.py
core/contracts/frv_membership.py
core/contracts/v4_inputs.py
algorithms/vegetation/wire_provider.py
algorithms/vegetation/frv_inputs.py
algorithms/vegetation/frv_artifacts.py
algorithms/vegetation/frv_membership.py
algorithms/vegetation/v4_inputs.py
tests/contracts/test_frv_wire_contracts.py
tests/contracts/test_frv_wire_serialization.py
tests/contracts/test_frv_membership_contracts.py
tests/contracts/test_frv_membership_serialization.py
tests/contracts/test_v4_input_contracts.py
tests/vegetation/test_wire_provider.py
tests/vegetation/test_frv_inputs.py
tests/vegetation/test_frv_artifacts.py
tests/vegetation/test_frv_membership.py
tests/vegetation/test_v4_inputs.py
```

Forbidden：除 Session A 已有改动和本节 20 个文件外全部项目文件；尤其禁止修改
`core/contracts/frv.py`、`core/contracts/frv_a2.py`、P-SOURCE/P-HAG files、existing V1/V2/
W1/V3 algorithms/contracts/tests、正式 V4 segmentation files、`pyproject.toml`、`.claude/`、
outputs/data 和 binary fixtures。

## 13. 精确测试矩阵

### 13.1 P-GEOMETRY

- exact defaults/type/bool/NumPy-scalar/extra-field rejection；
- `FRVGEOCFG1/PCD1/PRF1` fixed vectors、negative-zero normalization；
- 两个且只有两个 source sessions、不同 chunk size 结果相同；
- radius boundary、distance/index tie、64 cap、raw count/density/median；
- covariance/eigh/eigenvector sign、degenerate、small negative eigen clamp；
- canopy/woody/ground/vertical flags 的每个 boundary；
- mutual support edge symmetry、16 cap、跨 chunk/cell halo；
- one-point、sparse、dense-cell-limit、large-coordinate source；
- FULL_SOURCE index/count/state conservation、BND1、private capability；
- lifecycle、early termination、cleanup 和所有 failure mapping。

### 13.2 P-WIRE

- W1 authority/EMPTY authority exact schema、approval、components、digests；
- rejected fallback/review/gap/non-ready/zero edge/incomplete inventory；
- W1 local-to-source coordinate binding；
- point-to-segment interior/endpoints、radius exact boundaries、deterministic ties；
- BODY_POSSIBLE/CONFLICT/PROXIMITY/OUTSIDE 和 EMPTY N/A counts；
- 一次且只有一次 source pass，不同 chunk size proof 相同；
- edge-cell limit、source mismatch、artifact corruption、failure mapping；
- FULL_SOURCE、BND1、private capability、lifecycle/cleanup。

### 13.3 Scope/membership/routing

- A1/V1/V2 component/provenance/source/origin/index mapping；
- V2 four status rows、unclustered V1 row、ordinary rejection still required；
- approved hard rejection 全字段与缺字段回 required；
- canonical/required ordering 和两个 conservation equations；
- voxel boundary、±30 margin、z-run union、tile uniqueness、overlap supporting refs；
- blocked row、zero inventory、zero-hit scope、independent source；
- four-provider A2 readiness/no-copy/no partial bundle；
- every automatic decision branch and exact priority；
- low-HAG synchronous rounds、schedule/chunk/tile invariance、round-12 nonconvergence；
- valid/rejected/conflicted/cyclic/superseded overrides；
- unique point decision、five-state counts、readiness partition；
- no first-version wire exclusion/body confirmation；
- provider/stream/index/artifact failures cannot become point review；
- 250000 boundary、multi-block ordering、FRVBLK1/FRVART1 fixed vectors；
- corruption of every header/array/block/manifest/binding field and atomic cleanup。

### 13.4 V4 handoff

- exact consumer validation and inherited `frv_` namespace；
- source replay only, altered bytes/bounds/origin/index failure；
- admitted+ready normal set、membership-review/HAG-review measured set、excluded audit；
- N=0 admitted set、no fictional point/HAG/tree；
- all point recovery states and primary-route disjoint conservation；
- V4RECIDX1/V4INP1 vectors、read-only arrays、no downsampled XYZ；
- `v4_routing_complete` independent from `frv_routing_complete`；
- output contains no tree/apex/crown/CHM/clearance fields；
- output corruption and atomic cleanup。

## 14. Session B directed verification

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_frv_wire_contracts.py tests/contracts/test_frv_wire_serialization.py tests/contracts/test_frv_membership_contracts.py tests/contracts/test_frv_membership_serialization.py tests/contracts/test_v4_input_contracts.py tests/vegetation/test_wire_provider.py tests/vegetation/test_frv_inputs.py tests/vegetation/test_frv_artifacts.py tests/vegetation/test_frv_membership.py tests/vegetation/test_v4_inputs.py -q

E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_frv_source_contracts.py tests/contracts/test_frv_source_serialization.py tests/contracts/test_frv_hag_contracts.py tests/contracts/test_frv_hag_serialization.py tests/contracts/test_frv_geometry_contracts.py tests/contracts/test_frv_geometry_serialization.py tests/contracts/test_frv_wire_contracts.py tests/contracts/test_frv_wire_serialization.py tests/contracts/test_frv_membership_contracts.py tests/contracts/test_frv_membership_serialization.py tests/contracts/test_v4_input_contracts.py tests/io/test_source_provider.py tests/vegetation/test_hag_provider.py tests/vegetation/test_geometry_provider.py tests/vegetation/test_wire_provider.py tests/vegetation/test_frv_inputs.py tests/vegetation/test_frv_artifacts.py tests/vegetation/test_frv_membership.py tests/vegetation/test_v4_inputs.py -q

E:\Miniforge\envs\treeline\python.exe -m ruff check core/contracts/frv_wire.py core/contracts/frv_membership.py core/contracts/v4_inputs.py algorithms/vegetation/wire_provider.py algorithms/vegetation/frv_inputs.py algorithms/vegetation/frv_artifacts.py algorithms/vegetation/frv_membership.py algorithms/vegetation/v4_inputs.py tests/contracts/test_frv_wire_contracts.py tests/contracts/test_frv_wire_serialization.py tests/contracts/test_frv_membership_contracts.py tests/contracts/test_frv_membership_serialization.py tests/contracts/test_v4_input_contracts.py tests/vegetation/test_wire_provider.py tests/vegetation/test_frv_inputs.py tests/vegetation/test_frv_artifacts.py tests/vegetation/test_frv_membership.py tests/vegetation/test_v4_inputs.py
```

Session B directed gate 通过后仍不提交，也不进行 intermediate Codex review。

## 15. 唯一一次最终综合验收和 Git 闭环

两个 session 全部完成后只执行一次以下流程：

1. Codex 对 Session A+B combined diff 做综合 final review；
2. `E:\Miniforge\envs\treeline\python.exe -m pytest -q`；
3. `E:\Miniforge\envs\treeline\python.exe -m ruff check algorithms core tests`；
4. read-only Git scope：`git diff --check`、`git status --short`、
   `git diff --name-status`、`git diff --stat`、`git diff --cached --name-status`；
5. 确认改动集合恰为 Session A Modified/New 加 Session B Modified/New，cached 为 empty；
6. 只有未来 Owner 明确授权覆盖 Git 写操作时，执行一次 add、一次 commit、一次 push。

中间不进行 Codex review、Git commit、push、reset、stash、branch switch 或第二次授权。
当前计划任务没有未来实施和 Git 写权限；Owner 的后续单次授权必须明确覆盖两个连续
implementation sessions，并单独明确是否包含最后一次 commit/push。

## 16. 验收门槛

| Gate | PASS |
|---|---|
| Baseline | branch/HEAD/divergence 与本文一致，除 ignored `.claude/` 外无先存改动 |
| Geometry | 两 pass、算法、contracts、digests、BND1、failure/tests 全部精确 |
| Wire | authority、一次 pass、distance/relation、no-confirm/no-exclude、BND1 全部精确 |
| A2 | 四 formal providers 都是 validated FULL_SOURCE，no-copy bundle |
| Scope | canonical/required、0.05 voxel、30-cell margin、union/routing 守恒 |
| Membership | 每 index 一次、唯一优先级、solver、override、五态守恒 |
| Artifact | exact schema、FRVBLK1/FRVART1、atomic publish、replay/corruption |
| V4 input | admitted∩ready∩recovered、review safety route、no segmentation |
| Multi-apex | 附录方向完整且产生零实现文件/输出 |
| Dependencies | pyproject unchanged，无禁止依赖和 binary fixture |
| Packaging | 两个连续 sessions，无 intermediate review/commit/reauthorization |
| Final | combined review、full pytest、broad Ruff、scope review 各执行一次并通过 |
| Git | 当前无写权限；未来只允许一次最终 commit/push 且须显式授权 |

## 17. 计划结论

```text
FRV_TO_V4_BUNDLE_PLAN_RESULT:
REVIEW_CANDIDATE_READY

P-GEOMETRY:
IMPLEMENTABLE

P-WIRE:
IMPLEMENTABLE

Membership/routing:
IMPLEMENTABLE

V4 handoff:
IMPLEMENTABLE

Multi-apex appendix:
COMPLETE

Implementation sessions:
2

Intermediate review:
NOT REQUIRED

Final combined review:
REQUIRED_ONCE

Git writes:
NOT AUTHORIZED
```
