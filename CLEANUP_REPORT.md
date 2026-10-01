# TreeLineWarning 历史构建产物清理报告

- 扫描时间：2026-09-13 19:23:53 +08:00
- 扫描范围：`E:\TreeLineWarning-fast`、`E:\TreeLineWarning-Releases`、`E:\`
- 执行状态：**仅扫描、未删除任何文件或目录**
- 当前稳定提交：`606127c3f3a4a8ad355b3053ca7069d25148e1ee`（已用 `git rev-parse HEAD` 核对）
- 当前 Final：`E:\TreeLineWarning-fast\artifacts\real-closure-release-20260913-606127c`
- 容量口径：文件逻辑大小；GiB/MiB 按 1024 进制；目录大小包含其全部普通文件，不跟随目录联接。
- 扫描时 E 盘可用空间：12,853,919,744 bytes（11.97 GiB）。

## 结论摘要

- 全盘命中 139 个 `.zip/.7z/.rar`（合计 15,529,291,964 bytes / 14.46 GiB）及 3,508 个指定名称目录。
- 大量目录命中来自 VS Code、Miniforge、Python/Node 包、已安装软件内部的通用 `build/dist/test/release` 目录。这些不属于 TreeLineWarning 历史产物，全部按 KEEP/超出清理范围处理。
- 已明确识别的 `DELETE_SAFE` 约 20,118,846,834 bytes（18.74 GiB）。这只是报告分类，不构成删除授权。
- `E:\TreeLineWarning-Releases` 内现有发布物均早于稳定提交 `606127c`，但由于用户明确要求保留该发布目录，目录内部的 15.60 GiB 内容整体列入 CONFIRM，而不是直接列入安全删除。
- 当前 Final ZIP 已复核 SHA256：`979E8AB69EF09B91788B92EDEE878F11442E4F8089D39D161741BDE44E0FF02D`。

## KEEP：当前必须保留

### 当前仓库与当前 Final

| 路径 | 大小 | 创建时间 | 保留原因 |
|---|---:|---|---|
| `E:\TreeLineWarning-fast` | 129,455,869,580 B（120.56 GiB） | 2026-08-09 18:19:04 | 当前 Git 仓库；禁止删除根目录。下表只对其内部可再生成的历史产物分类。 |
| `E:\TreeLineWarning-Releases` | 16,747,456,140 B（15.60 GiB） | 2026-09-08 15:54:44 | 用户明确要求保留目录；内部历史内容另列 CONFIRM。 |
| `E:\TreeLineWarning-fast\artifacts\real-closure-release-20260913-606127c` | 1,599,174,619 B（1.49 GiB） | 2026-09-13 18:45:10 | 当前 `606127c` Final Release 及闭环验收报告。 |
| `E:\TreeLineWarning-fast\artifacts\real-closure-release-20260913-606127c\TreeLineWarning-Engineering-Release-2026-Final.zip` | 1,586,943,875 B（1.48 GiB） | 2026-09-13 18:45:10 | 当前 Final ZIP；SHA256 已复核。 |
| `E:\TreeLineWarning-fast\tests` | 21,079,307 B（20.10 MiB） | 2026-08-09 18:19:05 | 仓库测试源码/fixture，不是临时测试输出。 |

### 含点云、模型或测试数据的路径

这些路径不能整目录删除；“受保护文件”统计包括 `.las/.laz/.copc.laz/.onnx/.pth/.pt`，并不表示其他扩展名一定可删。

| 路径 | 大小 | 创建时间 | 受保护文件 | 处理 |
|---|---:|---|---:|---|
| `E:\TreeLineWarning-fast\acceptance_evidence` | 950,328,435 B（906.30 MiB） | 2026-08-21 17:49:08 | 346 / 10,575,950 B | KEEP：验收/测试数据。 |
| `E:\TreeLineWarning-fast\artifacts\fast_research` | 2,295,404,186 B（2.14 GiB） | 2026-08-09 18:21:05 | 10 / 1,890,679,650 B | KEEP：含点云研究数据。 |
| `E:\TreeLineWarning-fast\artifacts\manual_single_span_runs` | 12,755,682,228 B（11.88 GiB） | 2026-08-10 23:06:37 | 2 / 64,959,402 B | KEEP：含点云和人工运行资料。 |
| `E:\TreeLineWarning-fast\artifacts\one_click_runs` | 7,902,149,963 B（7.36 GiB） | 2026-08-10 01:37:13 | 14 / 287,099,862 B | KEEP：含点云/测试数据。 |
| `E:\TreeLineWarning-fast` 下 45 个含受保护文件的 `*test*` 临时目录 | 1,420,054,734 B（1.32 GiB） | 2026-08-13 至 2026-09-13 | 1,520 / 64,703,089 B | KEEP：禁止整批清理测试数据。 |
| `E:\pytest-temp` | 339,552,505 B（323.82 MiB） | 2026-07-29 13:25:28 | 2,434 / 2,502,962 B | KEEP：含大量 LAS 测试数据。 |
| `E:\TreeLineWarning-memory-acceptance-20260822` | 15,894,123,116 B（14.80 GiB） | 2026-08-22 11:48:05 | 11 / 2,798,045,327 B | KEEP：大型验收点云/模型数据。 |
| `E:\TreeLineWarning-RealWorldClosure-20260913` | 11,475,441,424 B（10.69 GiB） | 2026-09-13 18:10:39 | 4 / 104,399,209 B | KEEP：真实世界闭环数据与结果，不可整目录删除。 |
| `E:\TreeLineWarning-RealWorldColdAfter-20260913` | 7,813,376,907 B（7.28 GiB） | 2026-09-13 18:36:59 | 2 / 39,439,807 B | KEEP：冷启动复核数据，不可整目录删除。 |
| `E:\TreeLineWarning-FreshInstall-20260913-4c03e25` | 10,427 B | 2026-09-13 14:03:54 | 1 / 10,427 B | KEEP：此前清理后仅余 `offline-ui-fixture.las`。 |

### 其他明确保留/超出范围

- `E:\TreeLineWarning`、`E:\TreeLineWarning-corridor-validation`、`E:\TreeLineWarning-fast-performance-finalization-v1`、`E:\TreeLineWarning-final-tree`、`E:\TreeLineWarning-pred-integration`、`E:\TreeLineWarning-review-risk-polish`、`E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1`、`E:\TreeLineWarning-v01-final-acceptance`：检测到 Git 标记，且多处含点云/测试资料；本轮不列为安全删除。
- `E:\Miniforge`、`E:\Miniforge1` 及其他 Python/conda 环境：KEEP。
- `E:\下载\litchfield.zip`：2,331,915,728 B（2.17 GiB），创建于 2026-09-09 13:59:05；疑似 LiDAR 数据集，KEEP。
- 企业微信课程/聊天 ZIP、软件安装目录、开发工具内部 `build/dist/test/release`：与本项目清理无关，KEEP。

## DELETE_SAFE：已确认是历史构建/测试产物

> 本节仅表示从扫描证据看可再生成、未发现受保护扩展名、且不是当前 Final。当前步骤仍然**不删除**。

### 当前仓库内的历史构建产物

| 路径 | 大小 | 创建时间 | 判定依据 |
|---|---:|---|---|
| `E:\TreeLineWarning-fast\build` | 86,228,414 B（82.23 MiB） | 2026-08-22 19:51:43 | 历史 PyInstaller/build 工作目录；最后修改早于当前 Final。 |
| `E:\TreeLineWarning-fast\dist` | 11,412,689,119 B（10.63 GiB） | 2026-08-18 17:12:13 | 旧 RC/测量/兼容性发布副本；未发现点云/模型扩展名。 |
| `E:\TreeLineWarning-fast\.build-lazrs-0.8.2-cp311-win-amd64` | 436,884 B | 2026-08-30 20:58:44 | 可再生成构建目录。 |
| `E:\TreeLineWarning-fast\.build-pdal-proj-data-final-20260830` | 17,186,547 B（16.39 MiB） | 2026-08-30 21:58:50 | 已被后续构建替代的运行时打包目录。 |
| `E:\TreeLineWarning-fast\.build-pdal-proj-data-final-20260830-02` | 17,203,135 B（16.41 MiB） | 2026-08-30 22:05:29 | 已被当前 Final 打包替代。 |
| `E:\TreeLineWarning-fast\.build-pdal-runtime-final-20260830` | 117,865,970 B（112.41 MiB） | 2026-08-30 21:04:51 | 历史本机运行时构建目录。 |
| `E:\TreeLineWarning-fast\.build-pdal-runtime-from-rc-h` | 0 B | 2026-08-23 12:52:21 | 空历史构建目录。 |

### 已被 `606127c` Final 替代的旧发布构建

| 路径 | 大小 | 创建时间 |
|---|---:|---|
| `E:\TreeLineWarning-fast\artifacts\final-build-dist-20260913-328837a-r3` | 2,257,947,118 B（2.10 GiB） | 2026-09-13 14:14:56 |
| `E:\TreeLineWarning-fast\artifacts\final-build-dist-20260913-3400bc0` | 1,994,722,026 B（1.86 GiB） | 2026-09-13 13:57:27 |
| `E:\TreeLineWarning-fast\artifacts\final-build-dist-20260913-4c03e25-r2` | 2,257,947,096 B（2.10 GiB） | 2026-09-13 13:59:50 |
| `E:\TreeLineWarning-fast\artifacts\final-build-work-20260913-328837a-r3` | 82,322,793 B（78.51 MiB） | 2026-09-13 14:14:55 |
| `E:\TreeLineWarning-fast\artifacts\final-build-work-20260913-3400bc0` | 82,316,260 B（78.50 MiB） | 2026-09-13 13:57:26 |
| `E:\TreeLineWarning-fast\artifacts\final-build-work-20260913-4c03e25-r2` | 82,322,444 B（78.51 MiB） | 2026-09-13 13:59:49 |
| `E:\TreeLineWarning-fast\artifacts\final-acceptance-20260913` | 14,600 B | 2026-09-13 14:02:06 |
| `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-Engineering-Release-2026-Final.zip` | 1,577,729,320 B（1.47 GiB） | 2026-09-13 14:16:29 |

旧 `Final.zip` 的 SHA256 为 `15419473695CAAA098C7B2878B18D17F0769FF81BE9A616DB992A756FF6F8BF0`，与当前 `606127c` Final ZIP 不同；它是被替代版本，不是当前 Final。

### 明确的临时测试输出（未发现受保护扩展名）

| 路径 | 大小 | 创建时间 |
|---|---:|---|
| `E:\TreeLineWarning-fast\.test-tmp-clearance-sol-native-e2e-20260830-a` | 93,709,581 B（89.37 MiB） | 2026-08-30 11:24:52 |
| `E:\TreeLineWarning-fast\.test-tmp-unicode-proj-repair-20260830-01` | 10,420,233 B（9.94 MiB） | 2026-08-30 22:59:39 |
| `E:\TreeLineWarning-fast\.test-tmp-measurements-final-20260905` | 2,519,184 B（2.40 MiB） | 2026-09-05 22:46:40 |
| `E:\TreeLineWarning-fast` 下其余 87 个零受保护扩展名的顶层 `*test*` 临时目录 | 10,173,096 B（9.70 MiB） | 2026-08-22 至 2026-09-13 |
| `E:\TreeLineWarning_pytest_tmp` | 12,087,180 B（11.53 MiB） | 2026-07-27 14:34:27 |
| `E:\TreeLineWarning-pytest-temp` | 1,513,610 B（1.44 MiB） | 2026-08-28 19:44:55 |
| `E:\TreeLineWarning-ai-qc-pytest-final`、`-pytest-v2`、`-pytest-v2b`、`-test-tmp`、`-test-tmp-2` | 1,492,224 B（1.42 MiB） | 2026-09-08 18:48 至 21:19 |

90 个仓库内临时测试目录的合计为 116,822,094 B（111.41 MiB）。实际删除前必须重新生成精确目录清单，不能用宽泛的 `*test*` 通配符删除。

## CONFIRM：需要人工确认

### 与当前提交相关、但不是 Final 的可再生成副本

| 路径 | 大小 | 创建时间 | 风险/确认点 |
|---|---:|---|---|
| `E:\TreeLineWarning-fast\artifacts\real-closure-build-dist-20260913-606127c` | 2,258,032,219 B（2.10 GiB） | 2026-09-13 18:42:56 | 当前提交的解包构建目录；Final ZIP 已保留，但它仍属于当前闭环过程。 |
| `E:\TreeLineWarning-fast\artifacts\real-closure-build-work-20260913-606127c` | 82,348,498 B（78.53 MiB） | 2026-09-13 18:42:55 | 当前提交的构建中间目录。 |
| `E:\TreeLineWarning-FreshExtract-606127c` | 2,258,032,219 B（2.10 GiB） | 2026-09-13 18:46:50 | 当前 Final 的 fresh-extract 验证副本；可由保留 ZIP 重建，但不是“历史”版本。 |
| `E:\TreeLineWarning-clean-build-606127c` | 22,436,526 B（21.40 MiB） | 2026-09-13 18:42:28 | 当前提交的 Git linked worktree；若清理必须用 Git worktree 流程，不能直接删目录。 |

### `E:\TreeLineWarning-Releases` 内的历史发布档案

该目录现有工程发布的 `VERSION.txt` 显示 `source_head = 5f6f4d1...`，不是当前稳定提交。由于父目录被明确要求保留，以下内容等待人工确认：

| 路径/模式 | 数量 | 大小 | 创建时间范围 | 说明 |
|---|---:|---:|---|---|
| `E:\TreeLineWarning-Releases\*-build` | 6 | 55,920,271 B（53.33 MiB） | 2026-09-08 15:58 至 16:44 | 旧 PyInstaller 工作目录。 |
| `E:\TreeLineWarning-Releases\*-dist` | 5 | 11,272,045,679 B（10.50 GiB） | 2026-09-08 15:58 至 16:18 | 旧解包发布副本。 |
| `E:\TreeLineWarning-Releases\*-smoke` | 5 | 535,174 B | 2026-09-08 16:03 至 16:46 | 旧 smoke 输出。 |
| `E:\TreeLineWarning-Releases\TreeLineWarning-0.1.0-Windows-x64-20260908-5f6f4d1.zip` | 1 | 1,576,627,672 B（1.47 GiB） | 2026-09-08 16:49:40 | 旧版本发布 ZIP。 |
| `E:\TreeLineWarning-Releases\TreeLineWarning-Engineering-Release-2026` | 1 | 2,257,183,092 B（2.10 GiB） | 2026-09-13 11:59:17 | `5f6f4d1` 工程发布解包副本。 |
| `E:\TreeLineWarning-Releases\TreeLineWarning-Engineering-Release-2026.zip` | 1 | 1,577,220,000 B（1.47 GiB） | 2026-09-13 12:20:02 | `5f6f4d1` 工程发布 ZIP；SHA256 `F288ABF7...34CF`。 |
| `E:\TreeLineWarning-Releases\*THIRD_PARTY_NOTICES*` 及两个 ZIP sidecar | 8 | 7,924,252 B（7.56 MiB） | 2026-09-08 至 2026-09-13 | 应与相应旧发布一起保留或删除，不能拆散。 |

整个目录内容合计 16,747,456,140 B（15.60 GiB）。确认时建议按“保留目录本身、清空已被 `606127c` 替代的内容”或“继续作为版本档案保留”二选一。

### 其他大体积待确认项

| 路径 | 大小 | 创建时间 | 风险/确认点 |
|---|---:|---|---|
| `E:\TreeLineWarning-FreshInstall-20260913-328837a` | 3,091,398,393 B（2.88 GiB） | 2026-09-13 14:17:45 | 旧 Final 解压验证副本，但含 10,427 B LAS fixture；只能保留该测试数据后做精确清理。 |
| `E:\TreeLineWarning-source-cache` | 13,936,440,996 B（12.98 GiB） | 2026-08-23 21:02:20 | 体积大且名称像源码缓存，但用途不明；不可直接判删。 |
| `E:\2026.8.31-2\2026.08.31-2.zip` | 4,211,193,594 B（3.92 GiB） | 2026-08-31 17:09:39 | 旧项目归档，父目录含 LAS；确认 ZIP 不再承担备份作用后才可单独清理。 |
| `E:\TreeLineWarning-cleanup-audit` | 134,651,891 B（128.41 MiB） | 2026-09-08 22:19:43 | 审计资料，不是构建目录。 |
| `E:\TreeLineWarning-D-cleanup-audit` | 20,822,991 B（19.86 MiB） | 2026-09-08 22:36:09 | 审计资料，不是构建目录。 |

## 压缩包专项扫描

50 MiB 以上且与本任务最相关的压缩包如下；其余大压缩包主要是用户课程/聊天资料或软件组件，均不属于本项目安全清理范围。

| 分类 | 路径 | 大小 | 创建时间 |
|---|---|---:|---|
| KEEP | `E:\TreeLineWarning-fast\artifacts\real-closure-release-20260913-606127c\TreeLineWarning-Engineering-Release-2026-Final.zip` | 1,586,943,875 B | 2026-09-13 18:45:10 |
| DELETE_SAFE | `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-Engineering-Release-2026-Final.zip` | 1,577,729,320 B | 2026-09-13 14:16:29 |
| CONFIRM | `E:\TreeLineWarning-Releases\TreeLineWarning-Engineering-Release-2026.zip` | 1,577,220,000 B | 2026-09-13 12:20:02 |
| CONFIRM | `E:\TreeLineWarning-Releases\TreeLineWarning-0.1.0-Windows-x64-20260908-5f6f4d1.zip` | 1,576,627,672 B | 2026-09-08 16:49:40 |
| CONFIRM | `E:\2026.8.31-2\2026.08.31-2.zip` | 4,211,193,594 B | 2026-08-31 17:09:39 |
| KEEP | `E:\下载\litchfield.zip` | 2,331,915,728 B | 2026-09-09 13:59:05 |

- 当前 Final、仓库根部旧 Final、Releases 工程 ZIP 三者大小与 SHA256 均不同，不是字节级重复文件。
- 仓库 `dist`、各 build 目录内的 `base_library.zip` 是其父构建产物的内部组成部分，应随父目录分类，不能单独处理。
- 118 个小于 50 MiB 的压缩包合计 358,146,974 B（341.56 MiB）；其中包括测试 fixture、conda recipe 数据、Potree 依赖和软件资源，未一概列为可删。

## 后续删除边界

本报告不执行删除。若进入实际清理阶段，应：

1. 只使用本报告中逐条确认的精确路径，禁止使用 `E:\TreeLineWarning*`、`*test*`、`build`、`dist` 等宽通配符删除。
2. 删除前重新核对 Git HEAD 仍为 `606127c3f3a4a8ad355b3053ca7069d25148e1ee`，并复核当前 Final ZIP 的大小和 SHA256。
3. 对任何目录再次扫描 `.las/.laz/.copc.laz/.onnx/.pth/.pt`，同时人工检查 XLSX/JSON/CSV 等测试基准数据。
4. Git worktree 只能用 Git 感知的流程处理；不得直接删除其目录。
5. 保留 `E:\TreeLineWarning-fast`、`E:\TreeLineWarning-Releases` 两个根目录，且绝不触碰 conda 环境、点云、模型和测试数据。
