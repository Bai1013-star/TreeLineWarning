# 第二轮深度清理结果

- 执行时间：2026-09-13 17:06:01 +08:00
- 执行依据：`DISK_AUDIT_REPORT.md` 与用户明确授权清单
- 执行结果：**主体清理完成；2 个目标因保护/占用未清，2 个目标按保护规则部分清理**
- 安全边界：未删除 TreeLineWarning 主项目、Git 仓库、Final Release、conda env、点云、模型或受保护测试数据。

## 空间结果

| 项目 | 清理前 | 清理后 | 实测增加 |
|---|---:|---:|---:|
| C 盘可用空间 | 52,297,781,248 bytes（48.71 GiB） | 64,916,545,536 bytes（60.46 GiB） | 12,618,764,288 bytes（11.752 GiB） |
| E 盘可用空间 | 26,600,292,352 bytes（24.77 GiB） | 45,849,739,264 bytes（42.70 GiB） | 19,249,446,912 bytes（17.927 GiB） |
| 合计 | — | — | **31,868,211,200 bytes（29.680 GiB）** |

按已删除文件逻辑大小合计释放 **32,202,853,545 bytes（29.991 GiB）**。逻辑大小与盘符可用空间增量存在约 334.64 MB 差异，属于扫描期间临时文件、缓存写入和文件系统空间记账变化。

## 删除路径

### Safe Delete

| 路径 | 释放空间 | 状态 |
|---|---:|---|
| `C:\Users\Lenovo\AppData\Local\pip\Cache` | 1,610,273,692 bytes | 已删除 |
| `C:\Users\Lenovo\AppData\Local\npm-cache` | 2,521,575,477 bytes | 已删除 |
| `C:\Users\Lenovo\AppData\Local\uv\cache` | 599,862,799 bytes | 已删除；其中 `.pth` 为 Python 路径配置文件，不是模型；`.git` 为无效的 0-byte uv 缓存标记，不是 Git 仓库 |
| `C:\Users\Lenovo\AppData\Local\Temp` 中超过 7 天的安全子集 | 35,580,313 bytes，284 files | 已删除；5 个占用/受限文件未删除；额外排除了 TreeLineWarning、测试/pytest/fixture/benchmark/acceptance 路径以及受保护扩展名 |
| `E:\tmp` | 0 | **未删除**：285 个文件均位于 pytest 测试输出中，含 146 个 `.las` 和 1 个 `.laz`；服从“测试数据/点云禁止删除” |

Safe Delete 实际删除逻辑大小：**4,767,292,281 bytes（4.440 GiB）**。

### Confirm Delete

| 路径 | 释放空间 | 状态 |
|---|---:|---|
| `E:\matlab\MATLAB R2025a.rar` | 14,767,688,989 bytes | 已删除 |
| `C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-closure` | 707,050,802 bytes | 已删除 |
| `C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-lasso-fix` | 706,955,775 bytes | 已删除 |
| `C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-lasso-fix-r2` | 706,955,560 bytes | 已删除 |
| `C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-viewer-compat` | 707,016,916 bytes | 已删除 |
| `E:\TreeLineWarning-FreshInstall-20260913-4c03e25` | 3,925,090,231 bytes | **部分清理**：保留 10,427-byte LAS 测试点云及目录链 |
| `C:\ProgramData\NVIDIA Corporation\NVIDIA app\UpdateFramework\ota-artifacts` | 4,741,937,707 bytes | 已删除 |
| `C:\Users\Lenovo\.cache\codex-runtimes` | 0 | **未删除**：当前 Codex/PowerShell 进程正从该目录运行；保留 1,378,785,591 bytes，避免破坏当前任务 |
| `C:\Users\Lenovo\AppData\Local\Lenovo\LZUpdate` | 622,617,524 bytes | 已删除 |
| `E:\LeStoreDownload` | 471,411,408 bytes | **部分清理**：删除 VALORANT 安装器；其余 4 个 EXE 被文件 ACL 的 `BUILTIN\Users Deny Delete` 拒绝，保留 493,422,504 bytes |
| `E:\Miniforge1\Miniforge3-Windows-x86_64.exe` | 78,836,352 bytes | 已删除 |

Confirm Delete 实际删除逻辑大小：**27,435,561,264 bytes（25.551 GiB）**。

### 明确保留的残余

```text
PATH=E:\TreeLineWarning-FreshInstall-20260913-4c03e25\acceptance\interactive-project\data\raw\offline-ui-fixture.las
SIZE=10,427 bytes
REASON=点云和测试数据禁止删除。
```

```text
PATH=E:\tmp
SIZE=606,131 bytes, 285 files
REASON=pytest 测试数据；包含 146 个 LAS 和 1 个 LAZ。
```

```text
PATH=C:\Users\Lenovo\.cache\codex-runtimes
SIZE=1,378,785,591 bytes, 21,745 files
REASON=当前 Codex 与 PowerShell 运行时正在使用；无法在本任务在线期间安全删除。
```

```text
PATH=E:\LeStoreDownload
SIZE=493,422,504 bytes, 4 files
REASON=文件级 ACL 明确拒绝当前用户删除；未强行提权或绕过权限。
```

## 剩余空间

```text
DRIVE=C:\
FREE=64,916,545,536 bytes（60.46 GiB）
USED=约 239.64 GiB
```

```text
DRIVE=E:\
FREE=45,849,739,264 bytes（42.70 GiB）
USED=约 308.54 GiB
```

## TreeLineWarning 完整性检查

| 检查项 | 结果 |
|---|---|
| 主项目目录 `E:\TreeLineWarning-fast` | PASS，目录存在 |
| 发布归档目录 `E:\TreeLineWarning-Releases` | PASS，目录存在；16,747,456,140 bytes，17,251 files，与审计值一致 |
| Git HEAD | PASS：`328837af3cb74792222936150bbf79a242b7697f` |
| Git 对象检查 | PASS：`git fsck --no-dangling` exit 0 |
| 已跟踪工作树差异 | PASS：`git diff --quiet` exit 0 |
| 暂存区差异 | PASS：`git diff --cached --quiet` exit 0 |
| Final Release 文件 | PASS：存在，1,577,729,320 bytes |
| Final Release SHA256 | PASS：`15419473695CAAA098C7B2878B18D17F0769FF81BE9A616DB992A756FF6F8BF0` |
| 主项目受保护扩展名资产 | PASS：2,343 files，10,228,240,896 bytes，与清理前审计一致 |
| 原始仓库 `E:\TreeLineWarning` | PASS，目录和 `.git` 存在 |
| conda envs `E:\Miniforge\envs` | PASS，目录存在且未触碰 |
| 科学冻结/已跟踪项目路径 | PASS，没有已跟踪文件修改；本轮只新增审计/结果 Markdown |

综合状态：**TreeLineWarning Engineering Release 2026 Final 完整，仍满足 `FINAL_STATUS = READY_FOR_EXTERNAL_USE`。**

## 未执行事项

- 未删除 `C:\Users\Lenovo\.cache\codex-runtimes`。如需继续，必须先关闭 Codex，再从独立管理员终端重新确认并清理。
- 未强行修改 `E:\LeStoreDownload` 剩余 4 个文件的 ACL；需要管理员权限或 Lenovo 软件商店自身清理功能。
- 未删除任何点云、模型、Git 仓库、conda 环境、测试数据、Final Release 或 Releases 归档。
