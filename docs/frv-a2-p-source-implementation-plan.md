Status: IMPLEMENTATION-PLAN-FROZEN
Freeze date: 2026-08-03
Independent review result:
PASS_READY_FOR_PLAN_FREEZE

Historical findings:
CLOSED

Current plan blockers:
NONE

Plan executability:
PASS

# FRV-A2 P-SOURCE 正式源点云 Provider 实施计划（第四次修订稿）

## 1. 权限、职责与非授权声明

本文件是 TreeLineWarning 项目 FRV-A2 P-SOURCE 正式源点云 Provider 的实施计划候选稿。它只冻结后续实现必须遵守的公共合同、数据身份、运行时会话、失败语义、A2 单 Provider 证据边界、文件范围和验证门槛。

本计划不授权实现。只有计划经过独立复审、状态正式冻结，并由用户另行明确授权后，Claude Code + DeepSeek 才可实施代码和测试。

当前分工固定为：

- Codex：编写规划和独立审查。
- GPT：修订计划和编写阶段提示词。
- Claude Code + DeepSeek：只执行已授权代码、测试与实现修复。

本计划不授权：

- P-SOURCE 合同、Provider、导出或测试代码。
- P-HAG、P-GEOMETRY、P-WIRE。
- `FrvA2ReadinessRecord`、`FRVA2RDY1`、`ValidatedFrvA2Inputs` 或 FRV-A2 production integration。
- FRV-A3～A6、V4、增长预测、净空和风险模块。
- Git add、commit、push、merge、rebase、tag 或其他 Git 写操作。

## 2. 仓库基线

| 项目 | 冻结基线 |
|---|---|
| 仓库 | `E:\TreeLineWarning` |
| 分支 | `feature/algorithm` |
| HEAD | `9c501f0cce1afe88efe824232d096549c35cd1d5` |
| 最新提交 | `9c501f0 feat: add reviewed FRV-A2 framework` |
| 远程差异 | ahead `0` / behind `0` |
| 初始工作区 | 仅允许 `?? .claude/` 与本计划文件 |

`.claude/` 是本地工具目录，不读取、不修改、不删除、不暂存、不提交，也不计入项目变更。

## 3. 目的、范围与非目标

### 3.1 目的

P-SOURCE 为 FRV-A2 提供唯一正式 `SOURCE_SCAN` 前置 Provider。它必须把单一原始 LAS 文件的原始字节身份、A1 `SourceIdentity`、完整源点域、稳定 `original_index`、坐标声明、流式读取结果和 A2 Provider 证据绑定在一起，并在任何不完整、替换、错配或运行时中断情况下 fail closed。

### 3.2 正式范围

P-SOURCE v1 只支持：

- 一个显式、普通、可读的未压缩 `.las` 文件；
- 一个 raw `SOURCE_FILE` artifact component；
- A1 `SourceIdentity` 结构和运行时字节重验证；
- `FULL_SOURCE` 域；
- 物理点记录顺序 `original_index = 0..N-1`；
- bounded、read-only、single-use streaming session；
- 同一个底层 `BinaryIO` 贯穿 initial hash、header、point stream 和 final hash；
- success-only `FrvSourceProviderResult`；
- 失败统一抛出结构化 `FrvSourceLoadError`；
- 成功后生成单个 `FrvA2ProviderBindingEvidence` 和 `FRVA2BND1`。

### 3.3 明确非目标

P-SOURCE v1 不支持：

- LAZ 成功加载、压缩点数据、自动解压或 `.las`/`.laz` sibling 替换；
- 目录、glob、自动 discovery、多文件 bundle、manifest 或 sidecar；
- CRS 推断、坐标转换、单位转换、origin 扣减或 `xyz_local`；
- HAG、地面、几何、wire、vegetation membership、scope、tile、graph、solver；
- replayable runtime view、随机访问、并发 iterator 或第二次消费；
- BLOCKED Provider result、partial result、partial proof 或 BLOCKED Provider evidence；
- A2 aggregate readiness、`FRVA2RDY1` 或 A3 runtime bundle；
- 全源 `np.arange(N)`、全源 XYZ 复制或严格 raw zero-copy 声明。

## 4. 已冻结仓库合同与复用边界

P-SOURCE 必须复用而不得修改下列既有语义：

| 既有合同 | P-SOURCE 使用方式 |
|---|---|
| `SourceIdentity` | 使用真实字段、A1 content digest 和 record digest；A1 纯 validator 不访问文件系统，P-SOURCE 补充运行时重验证。 |
| `FrvCanonicalPointContentSpec` / `FRVPCD1` | 完整复用 A1 framing 和字段编码；不创建第二套 point-content framing。 |
| `FrvA2ArtifactComponentBinding` | 精确使用一个 `SOURCE_FILE` component。 |
| `FrvA2ProviderBindingEvidence` | 成功 result 映射为单一 `SOURCE_SCAN` formal evidence。 |
| `FrvA2FailureRecord` | 作为规范化失败记录；字段严格按真实 A2 合同。 |
| `frv_a2_index_domain_digest` / `FRVA2IDX1` | 流式实现必须与现有 scalar helper 字节完全相同。 |
| `frv_a2_provider_binding_digest` / `FRVA2BND1` | adapter 生成的 evidence 由既有 helper 计算。 |
| `frv_digest` | UTF-8 canonical JSON、`ensure_ascii=False`、lexicographic keys、compact separators、`allow_nan=False`、tuple→list、enum→value，然后 SHA-256。 |
| `algorithms/io/headers.py`、`chunks.py` | 仅作既有行为和兼容性参考，不作为 formal authority。P-SOURCE Provider 直接使用 laspy 2.7.0。 |
| `discover_las_sources` | 不进入正式加载路径。 |

A1 `SourceIdentity` 精确身份层次必须保持：

1. `source_file_sha256`：原始源文件字节身份。
2. `source_content_identity_digest`：不含访问路径和 audit 的内容与坐标语义身份。
3. `source_identity_record_digest`：包含 recorded/resolved path、relocation audit、equivalence audit 的完整 provenance record 身份。

## 5. Public constants 与严格 enum

下列 public constants 定义于 `core/contracts/frv_source.py`，名称和值都冻结：

```text
FRV_SOURCE_PROVIDER_KEY = "source_scan"
FRV_SOURCE_PROVIDER_INTERFACE_VERSION = "frv-source-provider-v1"
FRV_SOURCE_PROVIDER_SCHEMA_VERSION = "frv-source-provider-result-v1"
FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION = "frv-source-provider-config-v1"
FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION = "frv-source-header-v1"
FRV_SOURCE_PROVIDER_ALGORITHM_VERSION = "frv-source-laspy-single-handle-v1"
FRV_SOURCE_PROVIDER_CONFIG_DIGEST_DOMAIN = "FRVSRCFG1"
FRV_SOURCE_DEFAULT_CHUNK_SIZE = 1000000
```

下列 P-SOURCE 自有 enum 必须是严格 `str, Enum`，不修改 A1/A2 enum，不做大小写转换或字符串 coercion：

```text
FrvSourceModel:
  SINGLE_FILE_ONLY = "single_file_only"

FrvSourceCoordinateConversionPolicy:
  FORBID = "forbid"

FrvSourceInvalidPointPolicy:
  FAIL_CLOSED = "fail_closed"

FrvSourceCanonicalContentPolicy:
  VERIFY_IF_DECLARED = "verify_if_declared"

FrvSourceFormat:
  LAS = "las"

FrvSourceCompressionState:
  UNCOMPRESSED = "uncompressed"

FrvSourceCrsAuthorityState:
  ABSENT = "absent"

FrvSourceVlrPolicyState:
  NO_CRS_AUTHORITY_DETECTED = "no_crs_authority_detected"
```

P-SOURCE pass-through whitelist 的唯一合同权威也位于 `core/contracts/frv_source.py`：

```text
FrvSourcePassThroughDimension declaration order:
  CLASSIFICATION = "classification"
  INTENSITY = "intensity"
  RETURN_NUMBER = "return_number"
  NUMBER_OF_RETURNS = "number_of_returns"
  SCAN_ANGLE = "scan_angle"
  GPS_TIME = "gps_time"
  POINT_SOURCE_ID = "point_source_id"
  RED = "red"
  GREEN = "green"
  BLUE = "blue"
```

配置中的 dimensions 必须唯一并严格采用上述 declaration order。输入乱序、重复或未知值直接拒绝，不自动排序修复。

## 6. `FrvSourceProviderConfig` 与 `FRVSRCFG1`

### 6.1 精确字段

`FrvSourceProviderConfig` 是 frozen、可序列化、exact-schema dataclass：

```text
schema_version: str
source_model: FrvSourceModel
coordinate_conversion_policy: FrvSourceCoordinateConversionPolicy
invalid_point_policy: FrvSourceInvalidPointPolicy
canonical_content_policy: FrvSourceCanonicalContentPolicy
pass_through_dimensions: tuple[FrvSourcePassThroughDimension, ...]
provider_config_digest: str
```

固定 v1 值：

```text
schema_version = "frv-source-provider-config-v1"
source_model = SINGLE_FILE_ONLY
coordinate_conversion_policy = FORBID
invalid_point_policy = FAIL_CLOSED
canonical_content_policy = VERIFY_IF_DECLARED
```

### 6.2 精确 digest payload

`provider_config_digest` 必须等于 `frv_digest(payload)`，payload 精确且仅含：

```json
{
  "digest_domain": "FRVSRCFG1",
  "schema_version": "frv-source-provider-config-v1",
  "source_model": "single_file_only",
  "coordinate_conversion_policy": "forbid",
  "invalid_point_policy": "fail_closed",
  "canonical_content_policy": "verify_if_declared",
  "pass_through_dimensions": []
}
```

实际非空 dimensions 使用 enum declaration order 的 JSON array。payload 不允许额外 key，并排除：

- `provider_config_digest` 自身；
- `chunk_size`；
- runtime path、filename、host、mtime；
-异常文本、进度、缓存地址、随机值。

默认空 dimensions 的 canonical JSON bytes 精确为：

```text
{"canonical_content_policy":"verify_if_declared","coordinate_conversion_policy":"forbid","digest_domain":"FRVSRCFG1","invalid_point_policy":"fail_closed","pass_through_dimensions":[],"schema_version":"frv-source-provider-config-v1","source_model":"single_file_only"}
```

默认 digest 固定为：

```text
2028bcc1c2b342af5f867c7b49c5cfd1a9e0eda0a1d60effef58158e94a50846
```

测试必须使用上述 literal bytes 和 literal digest，不得调用被测函数生成期望值。

### 6.3 配置权威与 FRV 主配置交叉绑定

`load_frv_source` 必须接收真实 `FrvConfiguration` 和真实
`FrvSourceProviderConfig` 对象，先调用既有 validators，再重算并交叉核对全部配置与版本。
调用方不得用独立字符串绕过对象级校验。

必须同时满足：

1. `reference.configuration_digest` 等于传入
   `FrvConfiguration.configuration_digest` 的重算值。
2. `reference.provider_config_digest` 等于传入
   `FrvSourceProviderConfig.provider_config_digest` 的重算值。
3. `provider_config.provider_config_digest` 必须等于
   `frv_configuration.scan_provider_config_digest`。
4. `FRV_SOURCE_PROVIDER_INTERFACE_VERSION` 必须等于
   `frv_configuration.scan_provider_interface_version`。
5. `reference.interface_version` 必须精确等于
   `FRV_SOURCE_PROVIDER_INTERFACE_VERSION`。
6. `reference.schema_version` 必须精确等于
   `FRV_SOURCE_PROVIDER_SCHEMA_VERSION`。
7. `reference.algorithm_version` 必须精确等于
   `FRV_SOURCE_PROVIDER_ALGORITHM_VERSION`。
8. reference 中的 provider role、provider key、版本和两个 digest 都必须由 loader
   重新验证，不能只依赖构造时验证。
9. adapter 只读取 finalized result 中已经验证的值，不接受调用方单独传入
   configuration digest、provider config digest 或版本字符串。
10. 任一交叉绑定不一致均按 `REFERENCE_INVALID` fail closed，不创建可用 session，
    不产生 result 或 evidence。

`chunk_size` 是运行性能参数：built-in `int`、拒绝 `bool`、必须 `> 0`，不进入任何
identity/digest，且不同 chunk size 必须得到相同 proof、`FRVPCD1`、`FRVA2IDX1`
和 `FRVA2BND1`。

配置测试必须包含：

- 正确的 FRV configuration/provider config 交叉绑定；
- `scan_provider_config_digest` 错配；
- `scan_provider_interface_version` 错配；
- reference interface/schema/algorithm version 任一错配；
- reference configuration/provider config digest 任一错配；
- adapter 无独立 digest/version 参数的 signature 检查。

## 7. 输入、artifact 与 runtime reference

### 7.1 正式格式

P-SOURCE v1 正式接受条件必须全部满足：

- `runtime_access_path.suffix.lower() == ".las"`；
- 路径指向一个显式普通文件；
- `reader.header.are_points_compressed is False`；
- 文件可以在不使用 LAZ backend 的情况下读取；
- extension 与 compression state 一致。

明确拒绝：

- `.laz`，无论 backend 是否安装；
- `.las` 文件名但 header/point data 标记为 compressed；
- 非 `.las` 扩展名；
- 自动解压、临时转换和 sibling substitution。

### 7.2 Artifact component

canonical artifact layout 恰好一个 component：

| 字段 | 固定值 |
|---|---|
| `component_key` | `source_file` |
| `component_kind` | `FrvA2ArtifactComponentKind.SOURCE_FILE` |
| `sha256` | raw LAS bytes 的 lower-case SHA-256 |
| `size_bytes` | raw LAS 文件精确字节长度 |

`artifact_reference_id` 使用既有 A2 公式和 validator 重算。它只绑定 provider role、provider key 和 component identity，不包含 path、filename、mtime、host 或 SourceIdentity record audit。

### 7.3 `FrvSourceProviderReference`

该类型是 frozen 但**非序列化 runtime object**，exact fields：

```text
provider_role: FrvA2ProviderRole
provider_key: str
artifact_reference_id: str
artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
runtime_access_path: Path
source_identity: SourceIdentity
canonical_point_content_spec: FrvCanonicalPointContentSpec | None
configuration_digest: str
provider_config_digest: str
interface_version: str
schema_version: str
algorithm_version: str
```

固定规则：

- role 必须 `SOURCE_SCAN`，key 必须 `source_scan`；
- component 必须恰好一个 `source_file/SOURCE_FILE`；
- component SHA 必须等于 `source_identity.source_file_sha256`；
- reference 不调用 `frv_to_dict`，不进入 canonical JSON，也不直接进入任何 digest；
- `runtime_access_path` 仅用于当前进程的显式访问；
- reference validator 必须使用 §8.4 冻结的 P-SOURCE v1 Windows 路径比较键，比较
  `runtime_access_path` 与 `source_identity.resolved_access_path`，不得使用当前工作目录补全相对路径；
- `source_identity.relocation_audit.resolved_path_raw` 必须精确等于
  `source_identity.resolved_access_path`，保持 A1 record 的原始 provenance 字符串；
- `validate_source_identity_record_digest` 与 relocation audit 自身 validator 必须通过；
- 任一路径绑定、reparse-point、audit 或 digest 校验不一致时，即使 raw SHA 和 size 匹配也必须 fail closed；
- reference equality 不等于 artifact identity；
- 不扫描目录、不根据 `source_id` 猜路径、不跟随 moved sibling。

## 8. 路径、SourceIdentity 与 BND1 的精确关系

必须区分三个层次，并把 runtime access path 与 A1 provenance record 强绑定。

### 8.1 Path-free artifact identity

以下在相同 raw bytes 的显式搬迁中可以保持稳定：

- component raw SHA-256；
- component byte size；
- `artifact_reference_id`。

### 8.2 Path-free content identity

`source_content_identity_digest` 复用 A1 exact payload，排除 recorded/resolved path、
relocation audit 和 equivalence audit。相同内容在经过批准的搬迁后可以保持不变。

### 8.3 Provenance-sensitive record identity

`source_identity_record_digest` 复用 A1 exact record payload，包含真实
`recorded_source_path`、`resolved_access_path`、relocation audit 和 equivalence audit
相关记录。它进入真实 `FRVA2BND1` payload。

loader 必须执行下列确定性验证：

1. `runtime_access_path` 和 `source_identity.resolved_access_path` 都必须是绝对 Windows
   drive path 或绝对 UNC path；空字符串、相对路径、drive-relative path、rooted-without-drive
   path、NUL、alternate-data-stream 语法、`.`/`..` component、component 尾随空格或点均拒绝。
2. 两个路径分别通过 §8.4 的纯 lexical comparison-key 算法，comparison key 必须相等；
   不使用当前工作目录，不调用 `Path.resolve()`，不自动跟随 symlink 或 junction。
3. 从根/UNC share 到目标文件逐 component 检查现有 filesystem entry；任一 component
   是 symlink、junction 或 Windows reparse point 时 v1 fail closed，避免同一物理文件的
   未审计 alias。检查只决定是否允许访问，不进入任何 digest。
4. `source_identity.relocation_audit.resolved_path_raw` 必须精确等于
   `source_identity.resolved_access_path`，保留 A1 record 的原始 provenance 字符串。
5. relocation audit 自身 validator 和 `validate_source_identity_record_digest` 必须通过。
6. 不允许只替换 `runtime_access_path` 而复用旧 `SourceIdentity`。
7. 不允许 raw bytes 相同成为绕过 provenance record 更新的理由。
8. relocation 必须先创建并验证新的 A1 `SourceIdentity` record，其中
   `resolved_access_path`、relocation audit、equivalence audit 和
   `source_identity_record_digest` 与新访问位置一致。
9. 新 `SourceIdentity` 必须被放入新的 `FrvSourceProviderReference`。
10. 新 reference 必须重新 load、完整消费、finalize，并生成新的 Provider evidence。
11. 旧 reference、旧 result、旧 evidence 和旧 BND1 均不得复用。

因此：

- `runtime_access_path` 的 `Path` 对象本身不直接序列化，也不直接进入 digest；
- 它必须与 A1 record 中的 `resolved_access_path` 一致；
- artifact reference 和 content digest 可在批准搬迁后保持不变；
- record digest 必须反映新的 provenance；
- 由于 `source_identity_record_digest` 进入 BND1，搬迁后的 BND1 必须重算并通常变化。

### 8.4 P-SOURCE v1 Windows 路径比较键

P-SOURCE v1 冻结一个只用于 runtime/reference 绑定的纯 lexical comparison key。它不是
A1 digest、不是新的 provenance digest，也不改写 `SourceIdentity` 中保存的原始字符串。

输入规则：

1. 输入先由 `os.fspath()` 取得 built-in `str`；`bytes`、自定义 coercion 和空字符串拒绝。
2. 只接受 `PureWindowsPath` 可表示的 absolute drive path（例如 `E:\data\a.las`）或
   absolute UNC path（例如 `\\server\share\a.las`）。
3. 原始 component 中出现 `.`、`..`、空 component（UNC prefix 除外）、尾随空格、尾随点、
   NUL 或 drive colon 以外的 `:` 时拒绝；不先归一化再悄悄接受。
4. comparison key 将 `/` 统一为 `\`，drive letter 统一为大写，UNC server/share 与所有
   path component 使用 Unicode `casefold()`，并去除非根位置的重复 separator。
5. comparison key 不解析 short-name/8.3 alias，不查询 current working directory，不调用
   `realpath()`/`resolve()`，不展开环境变量或 `~`。
6. comparison key 只用于判断 runtime path 与 recorded resolved path 是否指向同一已声明
   lexical location；原始 `resolved_access_path` 仍原样进入 A1 record digest。
7. filesystem 访问前必须额外执行 no-reparse check：Windows 使用 `os.lstat()` 暴露的
   `st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT`（若字段存在），并同时检查
   `os.path.islink()`；任一祖先或目标为 reparse point/symlink/junction 时拒绝。
8. 非 Windows 执行环境不得模拟 Windows path resolution；P-SOURCE v1 formal load 必须以
   `FORMAT_REJECTED`/`REFERENCE_INVALID` fail closed，合同层的纯 comparison-key vector
   仍可跨平台测试。

固定测试必须覆盖：drive-letter case、separator 差异、Unicode casefold、相对路径、`.`、
`..`、UNC、trailing dot/space、ADS、symlink、junction/reparse point，以及 comparison key
相等但 A1 record 原始字符串不同导致 record digest/BND1 不同的情况。

测试必须包含：

- runtime path 与 resolved access path 精确一致的成功例；
- 相同 bytes、只替换 runtime path、复用旧 SourceIdentity 的拒绝例；
- 新 relocation audit + 新 SourceIdentity + 新 reference 的成功例；
- artifact/content identity 稳定而 record digest/BND1 变化的固定向量；
- 旧 BND1 evidence 在新路径上不可复用。

## 9. Header、格式和坐标 authority

### 9.1 `FrvSourceHeaderMetadata` exact fields

该类型是 frozen、可序列化、exact-schema dataclass，字段不得增加或省略：

```text
schema_version: str
source_format: FrvSourceFormat
las_version: str
point_format_id: int
point_record_length_bytes: int
point_count: int
scale_xyz: tuple[float, float, float]
offset_xyz: tuple[float, float, float]
min_xyz: tuple[float, float, float]
max_xyz: tuple[float, float, float]
dimensions: tuple[str, ...]
extra_dimensions: tuple[str, ...]
compression_state: FrvSourceCompressionState
crs_authority_state: FrvSourceCrsAuthorityState
non_crs_vlr_count: int
non_crs_evlr_count: int
vlr_policy_state: FrvSourceVlrPolicyState
```

字段来源固定为同一 laspy `LasReader.header`：

| 字段 | 来源与规范化 |
|---|---|
| `source_format` | loader 已验证 `.las` 且 uncompressed，固定 `LAS` |
| `las_version` | laspy header version 转稳定字符串 |
| `point_format_id` | header point-format ID，built-in int |
| `point_record_length_bytes` | header point format size，built-in int |
| `point_count` | header point count，built-in int |
| `scale_xyz` / `offset_xyz` | header scales/offsets，binary64 finite tuple |
| `min_xyz` / `max_xyz` | header mins/maxs，binary64 finite tuple |
| `dimensions` | 标准 dimension names，lower-case，lexicographic tuple |
| `extra_dimensions` | extra dimension names，lower-case，lexicographic tuple |
| `compression_state` | 来自 `reader.header.are_points_compressed`；必须为 built-in `False`，proof 固定 `UNCOMPRESSED` |
| `crs_authority_state` | 必须 `ABSENT` |
| `non_crs_vlr_count` | 按 §9.2 分类后允许的非 CRS VLR 数量 |
| `non_crs_evlr_count` | 按 §9.2 分类后允许的非 CRS EVLR 数量 |
| `vlr_policy_state` | 必须 `NO_CRS_AUTHORITY_DETECTED` |

`read_las_header_summary` 只作行为参考，不提供 formal metadata authority，也不需要修改。

### 9.2 CRS/VLR/EVLR 确定性规则

坐标 authority 来自 typed reference + A1 `SourceIdentity`，不是从 LAS 文件自动推断。
P-SOURCE v1 采用 fail-closed deny rule，不依赖 `parse_crs() is not None` 作为唯一证据。

v1 只接受：

```text
coordinate_frame = "LOCAL_CARTESIAN_XYZ"
units = "m"
transform_declaration = "identity"
origin_xyz = tuple(SourceIdentity.origin_context)
```

对同一 `LasReader.header` 中的 VLR 与 EVLR 必须按以下固定规则分类：

1. 使用 `laspy.open(..., read_evlrs=True)` 后读取 `header.number_of_evlrs` 和
   `header.evlrs`；任何 EVLR 存在都按 v1 unsupported authority fail closed，
   不需要读取或解释 EVLR payload，`non_crs_evlr_count` 成功时固定为 `0`。
2. 遍历全部普通 VLR。
3. 将 `user_id` 去除 LAS 固定宽度尾部 NUL 后进行精确大小写敏感比较。
4. 任何 `user_id == "LASF_Projection"` 的 VLR 一律拒绝，不区分 record ID，
   不尝试把其中一部分降级为普通 metadata。
5. 下列 laspy 已知 CRS/WKT/GeoTIFF 类型无条件拒绝，即使其 `user_id` 异常：
   - `GeoKeyDirectoryVlr`
   - `GeoDoubleParamsVlr`
   - `GeoAsciiParamsVlr`
   - `WktCoordinateSystemVlr`
   - `WktMathTransformVlr`
6. 调用 `header.parse_crs()`：
   - 返回非 `None`：拒绝；
   - 抛出异常：按 `COORDINATE_AUTHORITY_REJECTED` fail closed；
   - 返回 `None`：仍必须完成第 1～5 项，不能据此宣称无 CRS。
7. 除第 4～5 项之外的普通 VLR允许作为 non-CRS metadata，其数量进入
   `non_crs_vlr_count`；它们不得改变 coordinate frame、units、origin 或 transform。
8. 未知普通 VLR 只要 `user_id == "LASF_Projection"` 就拒绝；其他未知 user ID
   不得通过内容猜测 CRS，也不得被升级为坐标 authority。
9. 不应用 transform、不减 origin、不生成 `xyz_local`、不计算 HAG。

成功时：

```text
crs_authority_state = ABSENT
vlr_policy_state = NO_CRS_AUTHORITY_DETECTED
```

LAS scale/offset 只负责 storage integer→decoded float64，不是 origin。

测试必须覆盖：

- `LASF_Projection` 的任意 record ID VLR 拒绝；
- 任意 EVLR count > 0 拒绝；
- 每个已知 GeoTIFF/WKT laspy VLR 类型拒绝；
- `parse_crs()` 返回 CRS、返回 `None` 和抛异常三条路径；
- 非 CRS 普通 VLR允许并计数，成功 proof 的 `non_crs_evlr_count` 固定为 0；
- 不依赖 raw VLR payload 文本进行启发式分类。

### 9.3 Float 和 bounds 规则

- scale、offset、header bounds、streamed XYZ、streamed bounds 和 SourceIdentity bounds 全部必须 finite；
- canonical proof 中所有 `-0.0` 规范成 `+0.0`；
- N=0 在 bounds 比较前拒绝；
- 每轴 tolerance 固定为：

```text
axis_tolerance = max(abs(scale_axis) * 0.5, 1e-12)
```

对每轴 `a,b`，必须满足 `abs(a-b) <= axis_tolerance`：

1. header min/max vs streamed min/max；
2. SourceIdentity min/max vs header min/max；
3. SourceIdentity min/max vs streamed min/max。

不允许使用未定义的“近似相等”或调用方自选 tolerance。

## 10. A1 canonical point content 与 `FRVPCD1`

P-SOURCE 不创建新 framing。它必须逐字复用 A1：

```text
b"FRVPCD1\0"
+ uint64_le(spec_header_length)
+ canonical_frv_json_bytes(spec.to_digest_dict())
+ ordered canonical row bytes
```

P-SOURCE v1 若 `SourceIdentity` 声明 canonical point content，允许的 exact spec 只有：

| 顺序 | field_name | semantic_role | dtype | byte order | scale | offset | required | null policy |
|---:|---|---|---|---|---|---|---|---|
| 1 | `original_index` | `ORIGINAL_INDEX` | `<i8` | `LITTLE_ENDIAN` | `1.0` | `0.0` | true | `FORBID_NULL` |
| 2 | `x` | `X_COORDINATE` | `<f8` | `LITTLE_ENDIAN` | `1.0` | `0.0` | true | `FORBID_NULL` |
| 3 | `y` | `Y_COORDINATE` | `<f8` | `LITTLE_ENDIAN` | `1.0` | `0.0` | true | `FORBID_NULL` |
| 4 | `z` | `Z_COORDINATE` | `<f8` | `LITTLE_ENDIAN` | `1.0` | `0.0` | true | `FORBID_NULL` |

共同规则：

- `row_order = ORIGINAL_INDEX_ASCENDING`；
- `quantization_policy = EXACT_TYPED_VALUE`；
- `missing_value_policy = EXPLICIT_VALIDITY_BYTE`，但四个 required fields 均禁止 null；
- `scale` 与 `offset` 是 A1 contract 要求的 finite float metadata；本 exact spec 固定为 `1.0` 与 `0.0`，不使用 `None`；
- field bytes 无 padding；
- finite IEEE binary64；
- `-0.0` 编码为 `+0.0`；
- NaN/Inf 在 hashing 前拒绝；
- 不允许额外 field；
- canonical 三元组在 SourceIdentity 中必须全空或全有；全空时 P-SOURCE 不补造 digest。

`FRVPCD1` 中的 `original_index` 是 signed `<i8`。`FRVA2IDX1` 中同一非负 index 依据 A2 规则编码为 unsigned little-endian `uint64`。两种 byte domain 不得混用。

测试参考实现必须仅使用标准库 `json`、`hashlib`、`struct` 和 literal spec/rows 独立组装预期 bytes/digest，不得调用生产 digest helper生成期望值。

## 11. `FULL_SOURCE` 与 original-index domain

正式 evidence 固定：

```text
domain_kind = FrvA2DomainKind.FULL_SOURCE
domain_count = N
original_index = 0,1,...,N-1
sample_index_domain_digest = None
```

规则：

- N 必须同时等于 SourceIdentity count、header count 和实际完整 stream count；
- physical LAS point-record order 是唯一顺序；
- 不过滤、不重排、不抽样、不去重、不跨文件拼接；
- 每个 chunk 的 `start_index` 必须等于此前累计 count；
- 不允许 negative、gap、duplicate、reorder 或 extra point；
- stream digest必须复现既有 `FRVA2IDX1\0`、index role `ORIGINAL_INDEX`、source content digest、`FULL_SOURCE` value、uint64 count 和逐 index uint64 bytes；
- 不创建 N 长度 `np.arange` 或等价全源 index array。

### 11.1 Zero-point source policy

P-SOURCE v1 在 provider 语义层明确拒绝 N=0，即使底层 A2 index helper技术上可表示空 tuple。

当 header `point_count == 0`：

- `load_frv_source` 立即抛 `FrvSourceLoadError`；
- 不返回可用 `FrvSourceLoadSession`；
- 不迭代 chunk；
- 不生成 proof、result、FULL_SOURCE evidence、`FRVA2IDX1` 或 BND1；
- adapter 不可调用。

固定失败映射见第 15 节 `ZERO_POINT_SOURCE`。

## 12. Runtime public contracts

### 12.1 `FrvSourceChunk`

非序列化、frozen runtime contract exact fields：

```text
source_id: str
chunk_id: int
start_index: int
point_count: int
xyz: numpy.ndarray
original_index: numpy.ndarray
attributes: Mapping[str, numpy.ndarray]
is_last: bool
```

数组规则：

- `xyz`: C-contiguous、`<f8[N,3]`、finite、read-only；
- `original_index`: C-contiguous、`<i8[N]`、read-only；
- `attributes`: immutable mapping，key 严格按 config declaration order；
- classification `|u1[N]`；
- intensity `<u2[N]`；
- return_number `|u1[N]`；
- number_of_returns `|u1[N]`；
- scan_angle `<f8[N]`；
- gps_time `<f8[N]` 且 finite；
- point_source_id `<u2[N]`；
- red/green/blue 分别 `<u2[N]`，不提供复合 RGB object；
- selected dimension 不存在或 dtype/shape 无法规范化时 fail closed；
- 未选择 extra dimension 不影响 readiness；
- pass-through attributes 不进入 FRVPCD1；它们通过 provider config digest 绑定选择规则。

实现可以复制当前 chunk 所需数组以保证 read-only ownership，但不得保留可写别名，不得 materialize full source。

### 12.2 `FrvSourceLoadSession` Protocol

定义于合同层，只使用标准库 `Protocol`、`Iterator`、`Literal` 和 context-manager 类型，不导入 laspy 或算法模块：

```python
class FrvSourceLoadSession(Protocol):
    def iter_chunks(self) -> Iterator[FrvSourceChunk]: ...
    def finalize(self) -> FrvSourceProviderResult: ...
    def close(self) -> None: ...
    def __enter__(self) -> "FrvSourceLoadSession": ...
    def __exit__(self, exc_type, exc, traceback) -> Literal[False]: ...
```

具体私有实现位于 `algorithms/io/source_provider.py`。

### 12.3 Loader signature

唯一 public loader：

```python
def load_frv_source(
    reference: FrvSourceProviderReference,
    *,
    frv_configuration: FrvConfiguration,
    provider_config: FrvSourceProviderConfig,
    chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvSourceLoadSession:
    ...
```

不得存在第二种 public signature，不允许 `chunk_size: int | None`，两个 config 参数必须 keyword-only。

## 13. Single-use session 状态机

状态严格为：

```text
CREATED
ITERATING
STREAM_COMPLETE
FINALIZED
ABORTED
CLOSED
```

允许转移：

```text
CREATED -> ITERATING
ITERATING -> STREAM_COMPLETE
STREAM_COMPLETE -> FINALIZED
CREATED/ITERATING/STREAM_COMPLETE -> ABORTED on any failure or incomplete use
FINALIZED -> CLOSED
ABORTED -> CLOSED
FINALIZED + active caller exception -> CLOSED（finalized result 保持有效，异常继续传播）
```

规则：

- `iter_chunks()` 只能调用一次；
- 不允许 replay、restart、并发 iterator、随机 seek 或 use-after-close；
- iterator 必须完整消费到自然结束；
- caller `break`、`GeneratorExit`、取消、异常或 context-manager 异常退出都标记 ABORTED；
- `finalize()` 仅允许在 STREAM_COMPLETE 调用一次；
- finalize before completion、finalize twice 或 finalize after abort 都抛规范化错误；
- FINALIZED 前不存在 `FrvSourceProviderProof`、`FrvSourceProviderResult` 或 A2 formal evidence；
- `close()` 幂等：在 `FINALIZED` 时转为 `CLOSED`；在 `CREATED`、`ITERATING` 或
  `STREAM_COMPLETE` 时先标记 `ABORTED` 再转为 `CLOSED`，不抛出新异常但保存
  `SESSION_ABORTED` 供后续非法 `finalize()` 报告；ABORTED close 不产生 result；
- chunk 仅在当前迭代消费链中有效；调用方不得将其解释为 replayable source view；
- 下游临时派生输出只有在 session finalize 成功后才能发布，finalize 失败必须丢弃。

### 13.1 Context manager 的精确退出语义

`__exit__` 的返回值固定为 `False`，永不吞掉调用方异常。

- 若 block 内存在活动异常：
  - `CREATED`、`ITERATING` 或 `STREAM_COMPLETE` 转为 `ABORTED`；
  - `ABORTED` 保持 `ABORTED`；
  - 若状态已经是 `FINALIZED`，finalization 是不可逆 publication gate：不得改写为
    `ABORTED`，关闭资源并转为 `CLOSED`，已经返回的 finalized result 保持有效；
  - 先安全关闭 reader，再关闭底层 handle；
  - 非 `FINALIZED` 状态不创建 result/evidence；
  - `__exit__` 返回 `False`，保留原异常，不用新的 load error 覆盖它。
- 若 block 正常退出且状态为 `FINALIZED`：
  - 关闭资源并转为 `CLOSED`；
  - 已返回的 `FrvSourceProviderResult` 只含可序列化 proof 和 SourceIdentity，
    不持有 reader、handle、Path、chunk 或 session，因此关闭后仍有效；
  - `__exit__` 返回 `False`。
- 若 block 正常退出但状态为 `CREATED`、`ITERATING` 或 `STREAM_COMPLETE`：
  - 转为 `ABORTED` 并关闭资源；
  - 在关闭后抛出规范化 `FrvSourceLoadError`，错误 key 为 `SESSION_ABORTED`；
  - 这包括完整迭代后忘记调用 `finalize()` 的 `STREAM_COMPLETE` 情况。
- 若进入 `__exit__` 时状态已经为 `ABORTED` 且无活动异常：
  - 幂等关闭；
  - 抛出保存的规范化 `FrvSourceLoadError`。
- `CLOSED` 状态不允许再次进入 context manager；重复 `close()` 本身保持幂等。
- `__enter__()` 只允许在 `CREATED` 状态调用一次，其他状态按
  `FINALIZE_INVALID_STATE` fail closed。

iterator 的自然完成必须由 provider 在内部读取到真实 EOF 且 count 与 header 一致后
设置 `STREAM_COMPLETE`。仅仅销毁 generator、触发 `GeneratorExit` 或调用方停止迭代
不得被视为自然完成。

唯一推荐用法：

```python
with load_frv_source(
    reference,
    frv_configuration=frv_configuration,
    provider_config=provider_config,
    chunk_size=chunk_size,
) as session:
    for chunk in session.iter_chunks():
        consume_into_unpublished_staging(chunk)
    result = session.finalize()
```

## 14. Single-handle resource ownership 与 TOCTOU

### 14.1 所有权

具体 session：

1. 自己以 binary read mode 打开一个底层 `BinaryIO`；
2. session 对该 handle 拥有唯一关闭责任；
3. 精确调用 `laspy.open(handle, mode="r", closefd=False, read_evlrs=True)` 创建一个
   laspy 2.7.0 `LasReader`；不得把 pathname 直接传给 laspy；
4. laspy reader 先关闭，底层 handle 保持可用；
5. final hash 完成后 session 再关闭 handle；
6. 任何错误路径都按 reader→handle 顺序安全关闭。

若实际 laspy 2.7.0 API不能在 `closefd=False` 下保持底层 handle，实施必须立即停止并回到计划评审，不得改成多个独立 `open(path)`。

### 14.2 精确阶段顺序

```text
Phase A  validate reference/config in memory; no path guessing
Phase B  open exactly one BinaryIO from runtime_access_path
Phase C  seek(0); compute initial raw SHA-256 and exact size from this handle
Phase D  seek(0); call laspy.open(handle, mode="r", closefd=False, read_evlrs=True)
Phase E  inspect and validate exact header, format, compression and CRS/VLR policy
Phase F  stream every point once; compute count, bounds, FRVPCD1 when declared, FRVA2IDX1 and chunks
Phase G  require natural stream completion; close laspy reader without closing underlying handle
Phase H  seek(0); compute final raw SHA-256 and exact size from the same handle
Phase I  compare initial/final/reference/component/SourceIdentity values
Phase J  transition STREAM_COMPLETE; only then allow finalize
```

Hashing和 reader 操作共享 file pointer，因此每个阶段开始前必须显式 `seek(0)` 或使用冻结的 reader position。调用方不能访问或移动底层 handle。

### 14.3 权威边界

同一打开 handle 绑定本次实际消费的文件对象。Windows 路径名后续被替换，不会把当前 handle 自动转换到新文件；本次 result 只证明本 session 实际读取的 bytes。它不承诺 `runtime_access_path` 在 finalize 后仍指向同一 bytes。后续任何 reload 必须创建新 session并重新验证。

本计划不声称可阻止所有外部原地写入。initial/final raw hash、完整 stream count和内容 digest用于检测 session 内发生的变化；一旦不一致，不发布 result或 evidence。

## 15. Success-only result 与规范化失败

### 15.1 `FrvSourceProviderProof`

frozen、可序列化、exact-schema success proof：

```text
schema_version: str
provider_key: str
provider_role: FrvA2ProviderRole
interface_version: str
algorithm_version: str
configuration_digest: str
provider_config_digest: str
artifact_reference_id: str
artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
source_content_identity_digest: str
source_identity_record_digest: str
source_point_count: int
source_bounds_min_xyz: tuple[float, float, float]
source_bounds_max_xyz: tuple[float, float, float]
coordinate_frame: str
units: str
origin_xyz: tuple[float, float, float]
transform_declaration: str
domain_kind: FrvA2DomainKind
domain_count: int
original_index_domain_digest: str
sample_index_domain_digest: None
canonical_point_content_digest: str | None
canonical_point_content_spec_digest: str | None
header: FrvSourceHeaderMetadata
pass_through_dimensions: tuple[FrvSourcePassThroughDimension, ...]
```

Proof 只表示完成并通过 finalize 的成功状态，不含 `provider_readiness`、quality flags、reason flags、error codes、Path、handle、runtime arrays、host、mtime 或 raw exception。

### 15.2 `FrvSourceProviderResult` 与不可绕过的 publication capability

`FrvSourceProviderResult` 是非序列化 success wrapper。它的公开只读属性只有：

```text
proof: FrvSourceProviderProof
source_identity: SourceIdentity
```

具体合同冻结为：

- 使用 `@dataclass(frozen=True, slots=True, init=False)` 或等价的无 public constructor 类型；
- public `__init__` 不存在，普通调用方不能通过 `FrvSourceProviderResult(...)` 构造；
- 合同模块拥有 module-private singleton `_FRV_SOURCE_FINALIZED_CAPABILITY`；
- 合同模块拥有 module-private factory
  `_create_finalized_frv_source_provider_result(proof, source_identity, *, capability)`；
- factory 只在 `capability is _FRV_SOURCE_FINALIZED_CAPABILITY` 时构造，并在构造前重验
  proof/source identity、config binding、artifact、domain、digest 和 success-only invariants；
- concrete session 的 `finalize()` 是唯一获准调用该 private factory 的生产路径；
- result 内部保存 private capability identity，但该 slot 不属于 public fields、不序列化、
  不进入 equality、repr、digest、proof 或 BND1；
- public pure validator `validate_finalized_frv_source_provider_result(result)` 必须检查 exact
  type、private capability identity 和全部 proof/source binding；
- `to_frv_a2_provider_binding_evidence()` 在读取任何字段前必须调用该 validator；
- 反射、`object.__new__` 或私自导入 module-private sentinel 属于绕过 Python module contract
  的不受支持行为；公共 API 和普通 dataclass construction 路径必须不可伪造。

result 不持有 replayable view、Path、reader、handle、session 或 chunk。session 关闭后，
result、proof 和 SourceIdentity 仍完整有效。只有完整消费且成功执行 `finalize()` 才能获得
带有效 capability 的 result；手工拼装 proof/source identity 不能通过 adapter gate。

### 15.3 `FrvSourceLoadError`

所有失败统一抛出：

```text
FrvSourceLoadError
  failure_record: FrvA2FailureRecord
  quality_flags: int
```

真实 `FrvA2FailureRecord` exact fields：

```text
phase: FrvA2FailurePhase
provider_role: FrvA2ProviderRole
provider_key: str
field_name: str
error_code: FrvErrorCode
reason_flags: int
detail: str
```

`detail` 必须使用现有 canonical template：

```text
{phase.value}:{provider_role.value}:{provider_key}:{field_name}:{error_code.value}:{reason_flags}
```

`quality_flags` 是异常对象上的非负 int bitmask，不虚构为 failure record 字段。原始异常只可作为 Python `__cause__`，不序列化、不进入 detail/digest。绝对路径、host、stack trace、runtime value和 raw exception message不得进入 canonical failure record。

P-SOURCE 不返回 BLOCKED result、partial result、partial proof 或 BLOCKED evidence。未来 production integration 负责把 failure record纳入 aggregate readiness。

### 15.4 精确失败映射

所有行固定 `provider_role=SOURCE_SCAN`、`provider_key=source_scan`、`reason_flags=int(FrvReasonFlag.PROVIDER_UNCERTAINTY)`：

| Failure key | phase | field_name | error_code | quality_flags |
|---|---|---|---|---|
| `REFERENCE_INVALID` | `STRUCTURE_INTEGRITY` | `reference` | `FRV_PROVIDER_INVALID` | `int(FrvQualityFlag.PROVIDER_INVALID)` |
| `FILE_MISSING` | `STRUCTURE_INTEGRITY` | `runtime_access_path` | `FRV_PROVIDER_MISSING` | `int(FrvQualityFlag.PROVIDER_MISSING)` |
| `FORMAT_REJECTED` | `STRUCTURE_INTEGRITY` | `source_format` | `FRV_PROVIDER_INVALID` | `int(FrvQualityFlag.PROVIDER_INVALID)` |
| `ZERO_POINT_SOURCE` | `DOMAIN_ALIGNMENT` | `source_point_count` | `FRV_STREAM_INCOMPLETE` | `int(FrvQualityFlag.STREAM_INCOMPLETE)` |
| `HEADER_INVALID` | `STRUCTURE_INTEGRITY` | `header` | `FRV_PROVIDER_INVALID` | `int(FrvQualityFlag.PROVIDER_INVALID)` |
| `COMPONENT_DIGEST_MISMATCH` | `STRUCTURE_INTEGRITY` | `artifact_components` | `FRV_ARTIFACT_DIGEST_MISMATCH` | `int(FrvQualityFlag.ARTIFACT_DIGEST_MISMATCH)` |
| `SOURCE_IDENTITY_MISMATCH` | `IDENTITY_CONTEXT` | `source_identity` | `FRV_SOURCE_IDENTITY_FAILED` | `int(FrvQualityFlag.SOURCE_IDENTITY_FAILURE)` |
| `COORDINATE_AUTHORITY_REJECTED` | `IDENTITY_CONTEXT` | `coordinate_frame` | `FRV_SOURCE_IDENTITY_FAILED` | `int(FrvQualityFlag.SOURCE_IDENTITY_FAILURE)` |
| `STREAM_INCOMPLETE` | `DOMAIN_ALIGNMENT` | `point_stream` | `FRV_STREAM_INCOMPLETE` | `int(FrvQualityFlag.STREAM_INCOMPLETE)` |
| `ORIGINAL_INDEX_INVALID` | `DOMAIN_ALIGNMENT` | `original_index` | `FRV_ORIGINAL_INDEX_INVALID` | `int(FrvQualityFlag.ORIGINAL_INDEX_INVALID)` |
| `ORIGINAL_INDEX_DUPLICATE` | `DOMAIN_ALIGNMENT` | `original_index` | `FRV_ORIGINAL_INDEX_DUPLICATE` | `int(FrvQualityFlag.ORIGINAL_INDEX_DUPLICATE)` |
| `NONFINITE_COORDINATE` | `SEMANTIC_READINESS` | `xyz` | `FRV_PROVIDER_INVALID` | `int(FrvQualityFlag.PROVIDER_INVALID)` |
| `SESSION_ABORTED` | `DOMAIN_ALIGNMENT` | `session_state` | `FRV_STREAM_INCOMPLETE` | `int(FrvQualityFlag.STREAM_INCOMPLETE)` |
| `TOCTOU_CHANGED` | `STRUCTURE_INTEGRITY` | `source_file` | `FRV_ARTIFACT_DIGEST_MISMATCH` | `int(FrvQualityFlag.ARTIFACT_DIGEST_MISMATCH)` |
| `FINALIZE_INVALID_STATE` | `STRUCTURE_INTEGRITY` | `session_state` | `FRV_PROVIDER_INVALID` | `int(FrvQualityFlag.PROVIDER_INVALID)` |

不得新增 A1/A2 error/quality/reason enum。若实现发现无法落入该表，立即停止并重新评审。

## 16. A2 success evidence 与 BND1-only 边界

唯一 adapter：

```python
def to_frv_a2_provider_binding_evidence(
    result: FrvSourceProviderResult,
) -> FrvA2ProviderBindingEvidence:
    ...
```

adapter：

- 只接受 finalized success result；
- 首先调用 `validate_finalized_frv_source_provider_result()` 验证 private publication capability；
- 不接受 session、reference、exception、手工构造 wrapper 或 partial proof；
- 不访问文件；
- 不接受独立 digest 字符串；
- 不构造 readiness record、aggregate、runtime bundle 或 RDY1。

### 16.1 精确字段映射

| A2 field | 固定来源/值 |
|---|---|
| `provider_key` | `source_scan` |
| `provider_role` | `FrvA2ProviderRole.SOURCE_SCAN` |
| `evidence_class` | `FrvA2EvidenceClass.FORMAL_PROVIDER` |
| `public_module` | `algorithms.io.source_provider` |
| `public_type_name` | `FrvSourceProviderResult` |
| `public_loader_name` | `load_frv_source` |
| `interface_version` | `FRV_SOURCE_PROVIDER_INTERFACE_VERSION` |
| `schema_version` | `FRV_SOURCE_PROVIDER_SCHEMA_VERSION` |
| `algorithm_version` | `FRV_SOURCE_PROVIDER_ALGORITHM_VERSION` |
| `configuration_digest` | `result.proof.configuration_digest` |
| `provider_config_digest` | `result.proof.provider_config_digest` |
| `artifact_reference_id` | proof value |
| `artifact_components` | proof exact one component |
| `source_content_identity_digest` | proof value |
| `source_identity_record_digest` | proof value；可能因 relocation provenance 改变并改变 BND1 |
| `coordinate_frame` | `LOCAL_CARTESIAN_XYZ` |
| `units` | `m` |
| `origin_xyz` | proof value |
| `transform_declaration` | `identity` |
| `domain_kind` | `FULL_SOURCE` |
| `domain_count` | proof value，必须 > 0 |
| `original_index_domain_digest` | proof value |
| `sample_index_domain_digest` | `None` |
| `provider_readiness` | `FrvA2ProviderReadiness.VALIDATED` |
| `quality_flags` | integer `0` |
| `reason_flags` | integer `0` |
| `error_codes` | `()` |
| `upstream_binding_digests` | `()` |
| `review_codes` | `()` |
| `limitation_codes` | `()` |
| `binding_digest` | 既有 `frv_a2_provider_binding_digest` 重算值 |

`quality_flags` 和 `reason_flags` 是 int bitmask，不得写成 tuple。

### 16.2 明确禁止

P-SOURCE 不生成、不测试、不组装：

- `FrvA2ReadinessRecord`；
- `FRVA2RDY1`；
- `READY_FOR_A3`；
- BLOCKED aggregate；
- `ValidatedFrvA2Inputs`、`validate_frv_a2_inputs`、`load_frv_a2_inputs`。

这些名称只可出现在禁止或延后声明中。

## 17. 实施文件范围

### 17.1 NEW

| 文件 | 唯一职责 |
|---|---|
| `core/contracts/frv_source.py` | constants、enums、config、header、reference、chunk、session Protocol、proof、result、load error、纯 validators、FRVSRCFG1、streaming digest helpers、BND1 adapter。不得访问文件系统或导入 laspy/algorithms。 |
| `algorithms/io/source_provider.py` | laspy 2.7.0 single-handle session、format/header/CRS validation、hash、stream、TOCTOU、resource ownership。 |
| `tests/contracts/test_frv_source_contracts.py` | exact fields、strict types、constants、enums、reference boundary、failure contracts、state protocol。 |
| `tests/contracts/test_frv_source_serialization.py` | FRVSRCFG1、FRVPCD1 reference、FRVA2IDX1、BND1 mapping 和 path/provenance identity。 |
| `tests/io/test_source_provider.py` | 本地 LAS fixtures、session、single-handle、format、header、bounds、TOCTOU、memory和 failure tests。 |

### 17.2 MODIFY

| 文件 | 最小修改 |
|---|---|
| `core/contracts/__init__.py` | 导出已评审 P-SOURCE public contracts。 |
| `algorithms/io/__init__.py` | 导出 `load_frv_source` 和约定的 public result/session symbols。 |

### 17.3 不修改

```text
core/contracts/frv.py
core/contracts/frv_a2.py
core/contracts/pointcloud.py
algorithms/io/headers.py
algorithms/io/chunks.py
algorithms/io/discovery.py
algorithms/vegetation/frv_inputs.py
```

`source_provider.py` 直接使用 laspy。旧 header/chunk API只作参考，不承担 formal authority，也不需扩展。

## 18. 实施子任务

| ID | 目的 | 文件 | 输出 | 默认验证 | 独立审查 |
|---|---|---|---|---|---|
| C1 | constants、enums、config、exact contracts | contract + exports | public shape | Layer 1 focused | YES |
| C2 | FRVSRCFG1、FRVPCD1/IDX1 streaming helpers、reference validators | contract + serialization tests | digest interoperability | Layer 1 focused | YES |
| C3 | failure contract、state Protocol、artifact/identity/BND1 adapter | contract + contract tests | pure mapping | Layer 1 focused | YES |
| C4 | single-handle session、header/format/CRS、bounded chunks | provider + IO tests | streaming session | Layer 1 focused | YES |
| C5 | finalize、initial/final hash、TOCTOU、success result | provider + IO tests | result gate | Layer 1 focused | YES |
| C6 | complete negative/session/resource tests | all planned tests | focused candidate | Layer 1 focused | YES |
| C7 | 分层回归和最终审查 | no new implementation scope | validation records | 仅按单独授权逐层执行 | YES |

任何子任务需要修改未列文件、修改 A1/A2 digest/enum、接受 LAZ、发布 partial result或构造 RDY1 时立即停止。

## 19. 固定测试 fixtures 与 reference vectors

### 19.1 Normal LAS fixture

使用本地 laspy 创建，不下载：

```text
LAS version: 1.2
point format: 3
scale_xyz: (0.01, 0.01, 0.01)
offset_xyz: (1000.0, 2000.0, 50.0)
stored integer XYZ:
  (0, 0, 0)
  (100, -200, 50)
  (-50, 300, -25)
decoded XYZ:
  (1000.0, 2000.0, 50.0)
  (1001.0, 1998.0, 50.5)
  (999.5, 2003.0, 49.75)
point_count: 3
header/stream min: (999.5, 1998.0, 49.75)
header/stream max: (1001.0, 2003.0, 50.5)
axis tolerance: (0.005, 0.005, 0.005)
original_index: (0, 1, 2)
classification: (1, 5, 2)
intensity: (100, 200, 300)
return_number: (1, 1, 2)
number_of_returns: (1, 2, 2)
point_source_id: (7, 7, 8)
gps_time: (1.25, 2.5, 5.0)
red: (10, 20, 30)
green: (40, 50, 60)
blue: (70, 80, 90)
```

### 19.2 Additional fixed fixtures

- Zero-point LAS：相同 LAS version/point format/scale/offset，0 point；必须拒绝。
- Signed-zero canonical vector：一个 row，index 0，XYZ `(-0.0,+0.0,-0.0)`；独立 reference bytes必须与全部 `+0.0` 相同。
- Chunk invariance：同一 3-point fixture 分别使用 chunk size 1、2、3，所有 digests/proof相同。
- Wrong SHA：reference 中将 raw SHA 第一个 hex 改成不同合法 hex。
- Truncated file：创建有效 LAS 后确定性截断最后一个 point record 的一部分。
- `.laz` rejection：创建普通占位/fixture path 使用 `.laz` 后缀，loader 在正式 stream 前拒绝；不依赖 LAZ backend。
- `.las` compressed-state rejection：先用 LAS 1.2 fixture 写出有效未压缩文件，再以
  `bytearray` 将 LAS 1.2 header 的 point-data-record-format byte（固定 offset 104）
  设置 compression high bit `0x80`，保留 `.las` 后缀。loader 必须在正式 point stream
  前根据 header compression state拒绝；测试不依赖 LAZ backend。
- CRS-bearing LAS：分别构造 `user_id="LASF_Projection"` 的普通 VLR、任意 EVLR，
  以及 laspy 已知 GeoTIFF/WKT VLR class，均必须拒绝。
- Source replacement/TOCTOU：`algorithms/io/source_provider.py` 必须提供模块私有、
  生产默认 no-op 的确定性测试 seam：
  `_frv_source_phase_hook(phase: _FrvSourcePhase, handle: BinaryIO, *, chunk_ordinal: int | None) -> None`。
  `_FrvSourcePhase` 固定包含：
  `AFTER_INITIAL_HASH`、`AFTER_HEADER_VALIDATION`、`DURING_STREAM`、
  `BEFORE_FINAL_HASH`。
  精确调用语义：
  - `AFTER_INITIAL_HASH`：initial SHA/size 完成并验证后恰好一次，`chunk_ordinal=None`；
  - `AFTER_HEADER_VALIDATION`：header、format、CRS/VLR、zero-point 校验全部通过后恰好一次，
    `chunk_ordinal=None`；
  - `DURING_STREAM`：每个成功 decode、shape/type/finite/index 校验并计入 count/bounds/
    FRVPCD1/FRVA2IDX1 的 chunk，紧接在 yield 前恰好一次；第一个 chunk ordinal 为 0，
    后续严格递增，不得在空读、失败 chunk 或自然 EOF 额外调用；
  - `BEFORE_FINAL_HASH`：完整流自然结束、count/bounds/domain 验证通过、reader 已关闭但
    provider-owned handle 尚未 seek(0) 做 final hash 时恰好一次，`chunk_ordinal=None`。
  生产调用不传外部 callback；测试只通过 monkeypatch 该私有 no-op helper。hook 抛异常
  必须使 session ABORTED，并规范化为测试所针对的 load failure，不可被吞掉。
  同一模块还必须有私有 `_open_source_binary(path: Path) -> BinaryIO`，生产实现只执行
  read-only binary open；TOCTOU 测试可 monkeypatch 它返回一个以 `bytearray` 为后端的
  seekable `MutableBinaryIO` 测试对象。phase hook只修改该同一对象的后端 bytes，因而
  initial hash、laspy reader 和 final hash仍共享同一 handle，不使用第二个 OS open、
  sleep、线程竞态或额外 public 参数。

### 19.3 可重复性要求

- TOCTOU tests使用私有 phase seam/monkeypatch，不依赖竞态 timing；
- phase seam 的名称、signature、phase enum和值固定为 §19.2 所列私有接口；测试必须断言全局调用顺序、每个一次性 phase 恰好一次、`DURING_STREAM` 恰好等于成功 chunk 数且 ordinal 为 `0..K-1`；
- compressed `.las` fixture固定修改 LAS 1.2 offset 104 的 compression high bit，并断言
  `reader.header.are_points_compressed is True` 后走正式拒绝路径，不用 mock header 代替格式检测；
- partial consumption主动 `break`，随后 finalize 必须失败；
- repeated/concurrent iteration明确拒绝；
- memory bound使用 allocation spy和最大 materialized row counter，不使用系统 RSS；
- no-full-source guard 拦截 N-size `np.arange` 或等价 allocation；
- failure assertions只比较 typed fields和 canonical detail，不比较 raw exception/path；
- FRVSRCFG1使用 literal JSON bytes/digest；
- FRVPCD1/IDX1 使用独立标准库 reference encoder，不调用生产 helper产生期望；
- BND1 test只构造 Provider evidence，不构造 readiness aggregate。

## 20. 测试矩阵

### 20.1 Contracts

- exact field inventory、unknown/missing fields；
- enum declaration/value/order；
- built-in type、bool-as-int、NumPy scalar rejection；
- immutability、Path non-serialization；
- SHA、size、artifact component和 reference ID；
- config digest vector；
- session Protocol signature；
- success-only proof/result；
- public result direct-construction rejection、private capability mismatch、finalized factory success和adapter publication-gate validation；
- failure record/detail和 quality bitmask。

### 20.2 Identity and digest

- same raw bytes different explicit path：只有新 SourceIdentity record/reference 才允许；artifact/content stable，record/BND1按 audit变化；
- runtime path、`resolved_access_path`、relocation audit `resolved_path_raw` 任一错配均拒绝，且 audit/record digest 必须通过；drive-case/separator可按comparison key等价，relative、dot-segment、ADS、trailing dot/space、symlink/junction/reparse point拒绝；
- FRV main configuration 的 scan provider config/interface 与 P-SOURCE config/version 交叉绑定正反例；
- different bytes same filename：拒绝；
- raw SHA/size mismatch；
- SourceIdentity content/record mismatch；
- FRVPCD1 exact framing、finite scale `1.0`/offset `0.0`、null scale/offset rejection、signed zero、finite、chunk invariance；
- FRVA2IDX1 count/order/source/domain binding；
- BND1 success mapping、quality/reason int zero、record digest participation。

### 20.3 Format/header/coordinate

- valid uncompressed `.las`；
- `.laz` reject；
- compressed state reject；
- extension/header mismatch reject；
- zero point reject；
- exact header fields/types/order；
- CRS/WKT/GeoTIFF reject；
- non-CRS VLR allowed and counted；
- scale/offset/bounds finite；
- fixed tolerance comparison；
- frame/units/origin/identity transform mismatch reject。

### 20.4 Session/stream

- complete single-use flow；
- iter twice、concurrent iter、finalize early/twice、use after close reject；
- caller break/cancel/exception => ABORTED；
- `__exit__` active-exception propagation、normal-without-finalize error、FINALIZED close 和 ABORTED close；
- exact count、bounds、is_last、indices；
- read-only arrays and mapping；
- selected dimensions/dtypes/shapes；
- nonfinite XYZ/GPS time reject；
- truncated/partial stream reject；
- bounded memory and no full-source materialization。

### 20.5 TOCTOU/resource

- same BinaryIO used for both hashes and `laspy.open(..., closefd=False, read_evlrs=True)` reader；
- `closefd=False` ownership；
- reader closes before handle；
- every failure closes resources；
- change after initial hash、during stream、before final hash；
- path replacement does not silently switch current handle；
- later reload creates new session and revalidates。

## 21. 六层验证与授权

每层必须由负责人以独立提示词授权。当前计划阶段没有授权任何测试、Ruff 或 Git；未来
Claude Code + DeepSeek 的初始实现任务默认最多获得 Layer 1。上一层 PASS 仅形成下一层
“可申请授权”的前置证据，不自动授权下一层。

| Layer | 前置条件 | 默认授权 | PASS 后允许 | PASS 后仍不允许 |
|---|---|---|---|---|
| 1 focused | 计划已冻结；用户已授权 P-SOURCE C1～C7 实现；工作区仅批准文件 | 未来实现提示词可明确授权 | 形成 focused implementation candidate，申请 Layer 2 | Layer 2～6、Git |
| 2 A1/A2 | Layer 1 PASS；工作区未再次修改或重新跑 Layer 1 | NO，负责人单独授权 | 申请 Layer 3 | Layer 3～6、Git |
| 3 IO | Layer 2 PASS；测试文件名经只读确认 | NO，负责人单独授权 | 申请 Layer 4 | Layer 4～6、Git |
| 4 contracts broad | Layer 3 PASS；无新增范围变化 | NO，负责人单独授权 | 申请 Layer 5 | Layer 5～6、Git |
| 5 full + broad Ruff | Layer 4 PASS；当前工作区与验证记录一致 | NO，负责人单独授权 | 申请 Layer 6A | Layer 6A/6B、Git writes |
| 6A Git scope | Layer 1～5 全部 PASS；实现后未再改文件 | NO，负责人单独授权 Codex 只读审查 | 形成 `PASS_READY_FOR_GIT_AUTHORIZATION` | add/commit/push |
| 6B Git closure | Layer 6A PASS；用户明确给出 exact Git 授权 | NO，必须用户明确授权 | 精确暂存、提交、推送并闭环 | 后续 Provider/A2/A3/V4 |

### Layer 1 — P-SOURCE focused tests and focused Ruff

未来实施任务默认最多授权本层：

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_frv_source_contracts.py tests/contracts/test_frv_source_serialization.py tests/io/test_source_provider.py -q
E:\Miniforge\envs\treeline\python.exe -m ruff check core/contracts/frv_source.py algorithms/io/source_provider.py core/contracts/__init__.py algorithms/io/__init__.py tests/contracts/test_frv_source_contracts.py tests/contracts/test_frv_source_serialization.py tests/io/test_source_provider.py
```

PASS 条件：全部 focused tests 和 Ruff 通过，`git diff --check` 通过，工作区只有批准的
5 NEW + 2 MODIFY 文件及排除的 `.claude/`。PASS 后只能提交独立代码复审并申请 Layer 2。

Layer 1 PASS ≠ Layer 2 authorization。

### Layer 2 — A1/A2 contract regression

前置条件：Layer 1 PASS，focused 结果对应当前未修改工作区。默认未授权，需负责人单独授权：

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_frv_contracts.py tests/contracts/test_frv_serialization.py tests/contracts/test_frv_a2_contracts.py tests/contracts/test_frv_a2_serialization.py -q
```

PASS 后只允许申请 Layer 3；不允许 IO broad、contracts broad、full regression 或 Git。

Layer 2 PASS ≠ Layer 3 authorization。

### Layer 3 — IO regression

前置条件：Layer 2 PASS，目标测试文件通过只读定位确认。默认未授权，需负责人单独授权：

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/io/test_headers.py tests/io/test_chunks_and_crop.py tests/io/test_discovery.py tests/io/test_source_provider.py -q
```

若真实文件名不同，停止并由负责人修订命令，不得擅自扩大测试集。PASS 后只允许申请
Layer 4；不允许 contracts broad、full regression 或 Git。

Layer 3 PASS ≠ Layer 4 authorization。

### Layer 4 — tests/contracts broad regression

前置条件：Layer 3 PASS，工作区无新变化。默认未授权，需负责人单独授权：

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts -q
```

PASS 后只允许申请 Layer 5；不允许 full repository、broad Ruff 或 Git。

Layer 4 PASS ≠ Layer 5 authorization。

### Layer 5 — full repository regression and broad Ruff

前置条件：Layer 4 PASS，验证记录对应当前工作区。默认未授权，需负责人单独授权：

```powershell
E:\Miniforge\envs\treeline\python.exe -m pytest -q
E:\Miniforge\envs\treeline\python.exe -m ruff check core/contracts algorithms/io tests/contracts tests/io
```

PASS 后只允许申请 Layer 6A 独立 Git 范围审查；不允许暂存、提交或推送。

Layer 5 PASS ≠ Layer 6A authorization。

### Layer 6A — 独立只读 Git scope review

前置条件：

- Layer 1～5 的已授权验证全部 PASS；
- 实现后没有再次修改批准文件；
- 工作区除 `.claude/` 外只包含计划批准的 5 NEW + 2 MODIFY 文件；
- index 必须为空，用户尚未授权 Git 写操作；
- 负责人以独立提示词授权 Codex 只读审查。

只读命令：

```powershell
git branch --show-current
git rev-parse HEAD
git rev-list --left-right --count origin/feature/algorithm...HEAD
git status --short
git diff --check
git diff --stat
git diff --name-status
git diff --cached --check
git diff --cached --name-status
git ls-files --others --exclude-standard
```

`git diff --cached --check` 和 `git diff --cached --name-status` 在本层必须无输出，证明没有
提前暂存；它们不是要求 index 非空。

PASS 条件：分支/HEAD/远程差异符合预期；index 为空；没有第八个项目文件；`.claude/`
未纳入；diff 与冻结计划、公共导出和测试范围一致；验证记录对应当前未修改工作区。

Layer 6A PASS 只表示 `PASS_READY_FOR_GIT_AUTHORIZATION`，不允许执行 `git add`、
`git commit` 或 `git push`。

### Layer 6B — 用户明确授权后的精确 Git 闭环

前置条件：Layer 6A PASS，工作区未再修改，用户明确授权 exact files、commit message 和
push target。默认未授权。

仅在用户明确授权后，才允许：

```powershell
git add -- `
  core/contracts/frv_source.py `
  algorithms/io/source_provider.py `
  core/contracts/__init__.py `
  algorithms/io/__init__.py `
  tests/contracts/test_frv_source_contracts.py `
  tests/contracts/test_frv_source_serialization.py `
  tests/io/test_source_provider.py

git diff --cached --check
git diff --cached --name-status
git commit -m "feat: add reviewed FRV P-SOURCE provider"
git push origin feature/algorithm
git fetch origin
git rev-list --left-right --count origin/feature/algorithm...HEAD
git status --short
```

禁止 `git add .`、`git add -A`、`git commit -a`、force push、merge、rebase、stash、
reset、restore、clean、tag 或推送其他分支。暂存后若 cached scope 不是精确 5 NEW +
2 MODIFY，立即停止且不得自行 reset/restore。推送后必须确认远程差异 `0 0`，最终工作区
仅允许排除的 `.claude/`。

Layer 6B PASS 只声明 P-SOURCE Git closure 完成，不自动授权 P-HAG、P-GEOMETRY、
P-WIRE、FRV-A2 production integration、FRV-A3～A6 或 V4。远程拒绝时停止并报告，
不得自行 merge/rebase/force push。

Layer 5 PASS ≠ Layer 6A authorization。
Layer 6A PASS ≠ Layer 6B authorization。
Layer 6B 必须由用户明确授权。

## 22. 验收门槛

P-SOURCE 实现候选只有同时满足下列门槛才可进入独立代码复审：

1. public constants、enums、fields、signatures完全匹配；
2. FRVSRCFG1 literal vector通过；
3. 一个 uncompressed LAS、一个 SOURCE_FILE component；
4. SourceIdentity content/record/path分层正确；
5. zero-point fail closed；
6. same-handle single-use session和资源关闭通过；
7. 完整 stream后才产生 success result；
8. FRVPCD1和FRVA2IDX1流式结果与独立 reference一致；
9. CRS/format/header/bounds规则全部通过；
10. failure mapping使用真实 A1/A2 enums和 exact failure record；
11. BND1 evidence类型和值完全符合A2 validator；
12. 无 RDY1、aggregate、A3或V4越界；
13. Layer 1通过，其他层按授权逐层执行；
14. 工作区只有批准文件和排除的 `.claude/`。

## 23. 停止条件

出现以下任一情况必须停止实现并回到计划评审：

- laspy 2.7.0 无法以 provider-owned同一 BinaryIO和 `closefd=False`完成 reader生命周期；
- 需要多个独立 open 才能完成 initial/stream/final authority；
- 需要接受 LAZ或压缩点数据；
- 需要修改 A1/A2 enum、digest framing或证据字段；
- 需要创建 BLOCKED/partial result或在 finalize前发布证据；
- 不能证明 N>0完整 `0..N-1`域；
- 需要全源 materialization、过滤、重排、去重或坐标转换；
- CRS authority无法被确定性拒绝；
- 失败无法落入第15.4节映射；
- 需要修改未列文件或引入新依赖；
- 任一已授权测试/lint层失败。

## 24. 第四轮审查 finding closure

| Finding | Final decision | 对齐章节 | Implementation consequence | Test consequence |
|---|---|---|---|---|
| F-002 | runtime Path 与 A1 `resolved_access_path` 强绑定；record digest通过provenance进入BND1 | §7.3、§8 | relocation必须新建 A1 record/reference/result/evidence | 复用旧 SourceIdentity + 新 path 必须拒绝 |
| F-003 | provider语义层拒绝N=0 | §11.1、§15.4 | loader直接失败 | zero-point LAS不返回session/result/evidence |
| F-004 | single-use session + same BinaryIO + exact `__exit__` + finalize publication gate | §12～§14 | 删除direct-result loader和replay view | state/partial/context/TOCTOU/resource tests |
| F-005 | success-only result；失败抛typed error | §15 | 无BLOCKED/partial result | exact failure record和quality mask tests |
| F-006 | uncompressed LAS-only、exact header、deterministic VLR/EVLR deny rules | §7、§9 | provider直接使用laspy header | compression/CRS/VLR/EVLR/header/bounds vectors |
| F-010 | 唯一keyword-only loader和session signatures | §12.2～§12.3 | 无第二签名 | inspect/signature contract tests |
| F-011 | 只生成VALIDATED BND1 evidence | §16 | 不触碰readiness aggregate | flags int zero，RDY1 absence tests |
| F-012 | 5 NEW + 2 MODIFY；固定LAS header mutation与private phase seam | §17、§19～§20 | 不修改旧headers/chunks | no download/sleep/RSS tests |
| F-013 | 六个独立授权层，Layer 6A/6B边界完整 | §21 | focused不自动扩大，Git需用户授权 | 每层单独记录和授权 |
| F-NEW-001 | success flags固定int 0 | §16.1 | evidence通过真实A2 validator | reject tuple flags；accept integer zero |
| F-NEW3-001 | runtime path必须等于A1 resolved access path | §7.3、§8.3 | 禁止新path复用旧record | path/provenance正反例 |
| F-NEW3-002 | P-SOURCE config/version与`FrvConfiguration`交叉绑定 | §6.3 | loader拒绝未绑定FRV主配置的reference | config/interface/version错配负例 |
| F-NEW3-003 | 拒绝全部`LASF_Projection` VLR、全部EVLR和已知CRS类型 | §9.2 | 不依赖parse_crs返回None证明无CRS | VLR/EVLR/parse failure固定测试 |
| F-NEW4-001 | result采用无public constructor、private capability/factory和adapter强制validator | §15.2、§16 | direct construction不能通过formal evidence publication gate | direct constructor、伪造wrapper、capability mismatch与finalized success测试 |
| F-NEW4-002 | FRVPCD1四字段scale/offset固定为finite `1.0`/`0.0` | §10 | exact spec可通过真实A1 contract构造 | literal spec、fixed bytes/digest和null scale/offset拒绝测试 |

已通过的 F-001、F-007、F-008、F-009 由 §5、§6、§10、§12 保持，不得回退。

## 25. 当前 blockers 声明

Current plan blockers: NONE。

该声明基于本计划已经冻结并可由当前仓库与 laspy 2.7.0 实施：

- FRVSRCFG1 exact payload/vector；
- `FrvConfiguration.scan_provider_config_digest` 与 P-SOURCE config digest 的交叉绑定；
- `FrvConfiguration.scan_provider_interface_version` 与固定 interface version 的交叉绑定；
- non-serializable runtime reference 与 `SourceIdentity.resolved_access_path` 的冻结 Windows lexical comparison key、absolute/no-reparse 强一致性；
- provenance-sensitive record digest 和 BND1 重绑定；
- zero-point拒绝；
- single-use state machine、FINALIZED active-exception 语义、精确 `__exit__`、自然完成与 abort区分；
- laspy 2.7.0 provider-owned BinaryIO、`closefd=False`、reader→handle关闭顺序；
- success-only result、private finalized capability publication gate和真实failure record字段；
- uncompressed LAS-only；
- 全部 `LASF_Projection` VLR、全部 EVLR 与已知CRS类型的确定性拒绝；
- FRVPCD1 finite scale/offset exact spec与FRVA2IDX1 byte-domain分离；
- exact A2 success mapping；
- 可实施文件范围；
- LAS 1.2 compression-bit fixture、逐chunk计数的private phase seam和固定fixtures；
- 六层授权、Layer 1～5完整前置/PASS边界及 Layer 6A cached-empty/6B Git边界。

实现阶段若出现以下任一事实，才形成新的真实 BLOCKER并立即停止：

- laspy 实际行为与已核对的 `closefd=False`/seek/handle ownership不一致；
- 当前环境或目标平台不能保持同一打开 handle完成 initial hash、reader stream和final hash；
- public result无法以无public constructor/private capability方式实现，或adapter无法验证该capability；
- A1 `FrvCanonicalPointFieldSpec` 不接受本计划固定的 finite scale `1.0`/offset `0.0`；
- A1 raw path/audit/record digest 的真实字段或 validator 与冻结 comparison-key/no-reparse 规则无法一致绑定；
- `FrvConfiguration` 真实字段名或类型与本计划交叉绑定不一致；
- LAS header无法可靠暴露 compression state；
- laspy VLR class、`LASF_Projection` user ID、EVLR count或 `parse_crs()` 无法按§9.2确定性检查；
- 需要修改 A1/A2 enum、digest framing或未列文件。

不得用多个独立 open、放宽 CRS 规则、复用旧 SourceIdentity、跳过配置交叉绑定、
mock-only格式检测或扩大文件范围静默绕过。

## 26. 最终阶段边界

Plan authorizes implementation: NO

P-SOURCE implementation:
NOT AUTHORIZED

P-HAG:
NOT AUTHORIZED

P-GEOMETRY:
NOT AUTHORIZED

P-WIRE:
NOT AUTHORIZED

FRV-A2 production integration:
NOT AUTHORIZED

FRV-A3～A6:
NOT AUTHORIZED

V4:
NOT AUTHORIZED

Freeze authority:

This document is frozen as the reviewed implementation plan for
FRV-A2 P-SOURCE.

Future implementation must conform to this frozen plan.

Any change to public contracts, digest framing, identity semantics,
session lifecycle, failure semantics, artifact layout, A2 evidence
mapping, authorized files, or stage boundaries requires a separate
plan amendment and independent review.

Plan freeze does not authorize implementation.
