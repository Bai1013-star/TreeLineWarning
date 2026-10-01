# C: / E: 磁盘空间审计报告

- 审计时间：2026-09-13 16:47:41 +08:00
- 审计范围：`C:\`、`E:\`
- 执行状态：**仅扫描、未删除任何文件**
- 容量口径：文件逻辑大小；`GB/MB` 使用十进制换算；未跟随重解析点/目录联接。
- 扫描前空间：C 盘已用 250.82 GB、可用 49.28 GB；E 盘已用 326.43 GB、可用 24.82 GB。
- 一级目录“最近修改时间”取目录自身的 `LastWriteTime`；受权限保护的系统目录可能为部分统计，详见备注。

## 结论摘要

- 可直接清理的可再生成缓存约 **5.19 GB 上限**；其中临时目录只允许清理“超过 7 天、非 TreeLineWarning、非受保护扩展名、且未被占用”的文件，实际可释放量会略低。
- 需要人工确认的高价值候选约 **45 GB**，主要是 MATLAB 安装压缩包、conda 包缓存、NVIDIA/Lenovo 安装缓存、旧版 TreeLineWarning 下载副本和两份 Final FreshInstall 解压验证目录。
- `E:\TreeLineWarning-fast`、`E:\TreeLineWarning-Releases`、当前 Final Release、Git 仓库、conda 环境、点云、模型、测试数据和论文/项目资料均列入 Keep。
- 50 MB 以上 ZIP 共 28 个；按文件大小分组后 **没有重复大小组**，因此没有可判定的完全重复 ZIP。当前 Final ZIP 的 SHA256 已复核匹配。

## Safe Delete

以下内容可再生成。后续真正清理时仍应先关闭使用这些缓存的 Python/Node/Codex 进程。

```text
PATH=C:\Users\Lenovo\AppData\Local\pip\Cache
SIZE=1,610,273,692 bytes (1.61 GB), 1,053 files
CATEGORY=pip cache
REASON=下载/构建缓存，可由 pip 自动重新下载；不包含 conda 环境本体。
```

```text
PATH=C:\Users\Lenovo\AppData\Local\npm-cache
SIZE=2,521,575,477 bytes (2.52 GB), 29,104 files
CATEGORY=npm cache
REASON=npm 可再生成缓存；删除后只会在后续安装时重新下载。
```

```text
PATH=C:\Users\Lenovo\AppData\Local\uv\cache
SIZE=599,862,799 bytes (599.86 MB), 22,819 files
CATEGORY=uv/Python package cache
REASON=可再生成的包缓存；不属于 Python/conda 环境。
```

```text
PATH=C:\Users\Lenovo\AppData\Local\Temp\*（仅 2026-09-06 以前、非 TreeLineWarning、非 *.las/*.laz/*.copc.laz/*.onnx/*.pth/*.pt、且未占用的文件）
SIZE=最多 461,160,064 bytes (461.16 MB), 1,330 files；应用排除规则后会更少
CATEGORY=aged temporary files
REASON=超过 7 天的普通临时文件可清理；整个 Temp 目录为 11.62 GB 且仍在活动，禁止整目录删除。扫描到 TreeLineWarning 临时测试数据和 LAS，必须排除。
```

```text
PATH=E:\tmp
SIZE=606,131 bytes (0.61 MB), 285 files
CATEGORY=temporary files
REASON=体积很小的通用临时目录；后续清理前确认无进程占用即可。
```

## Confirm Delete

以下候选不应直接手工删除；需由用户确认用途，系统/软件缓存优先使用对应清理器。

```text
PATH=E:\Miniforge\pkgs
SIZE=7,058,707,615 bytes (7.06 GB), 60,431 files
CATEGORY=conda package cache
REASON=不是 conda 环境本体，但建议仅在确认后使用 conda clean 清理；E:\Miniforge\envs 必须保留。
```

```text
PATH=C:\ProgramData\NVIDIA Corporation\NVIDIA app\UpdateFramework\ota-artifacts
SIZE=4,741,937,707 bytes (4.74 GB), 1,279 files
CATEGORY=NVIDIA update/download cache
REASON=含约 761 MB 显卡驱动安装包及更新资产；可通过 NVIDIA 应用/系统存储清理，但可能影响离线回滚能力。
```

```text
PATH=C:\Users\Lenovo\AppData\Local\Lenovo\LZUpdate
SIZE=622,617,524 bytes (622.62 MB), 2 files
CATEGORY=Lenovo update installer cache
REASON=含 LZInstall.exe；与已安装目录中的同名同大小安装器相似，但本报告未把 EXE 当作 ZIP 重复项，删除前确认 LegionZone 不需要回滚/修复。
```

```text
PATH=C:\Autodesk
SIZE=773,546,330 bytes (773.55 MB), 47 files
CATEGORY=Autodesk installation source/cache
REASON=可能是安装解压源；建议使用 Autodesk 清理工具或确认软件不再需要修复后处理。
```

```text
PATH=C:\Windows\SoftwareDistribution\Download
SIZE=66,985,630 bytes (66.99 MB), 14 files；其中超过 7 天为 48,941,141 bytes
CATEGORY=Windows Update download cache
REASON=系统管理目录；只应在 Windows 更新空闲时通过“存储/磁盘清理”处理，不建议手工删除。
```

```text
PATH=C:\Users\Lenovo\.cache\codex-runtimes
SIZE=1,378,785,591 bytes (1.38 GB), 21,745 files
CATEGORY=Codex runtime cache
REASON=可再获取，但当前 Codex 任务可能正在使用；只能关闭相关任务/应用并确认后处理。
```

```text
PATH=E:\matlab\MATLAB R2025a.rar
SIZE=14,767,688,989 bytes (14.77 GB), 1 file
CATEGORY=large installer archive
REASON=已存在 E:\matlab 的 MATLAB 安装目录；若确认无需离线重装，可删除此 RAR。不要直接删除 E:\matlab 已安装程序目录。
```

```text
PATH=E:\Miniforge1\Miniforge3-Windows-x86_64.exe
SIZE=78,836,352 bytes (78.84 MB), 1 file
CATEGORY=installer
REASON=Miniforge 安装包副本；确认无需离线重装后可删除，不影响 E:\Miniforge 环境本体。
```

```text
PATH=E:\LeStoreDownload
SIZE=964,833,912 bytes (964.83 MB), 5 files
CATEGORY=downloaded installers
REASON=包含 VALORANT/联想软件商店下载的安装器；确认已安装且无需离线安装后再删。
```

```text
PATH=E:\2026.8.31-2\2026.08.31-2.zip
SIZE=4,211,193,594 bytes (4.21 GB), 1 file
CATEGORY=old archive / possible old release
REASON=父目录同时含 TreeLineWarning 运行产物和 LAS，父目录必须保留；仅此 ZIP 在确认已解压且不再作为归档后可删。
```

```text
PATH=C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Survey-Windows-x64-20260831
SIZE=706,925,919 bytes (706.93 MB), 612 files
CATEGORY=old extracted release
REASON=旧版下载解压副本；不属于当前 Final Release。确认无需回归/对照后可删。
```

```text
PATH=C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-closure
SIZE=707,050,802 bytes (707.05 MB), 616 files
CATEGORY=old extracted release
REASON=旧版下载解压副本；确认不再用于 closure 回归后可删。
```

```text
PATH=C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-lasso-fix
SIZE=706,955,775 bytes (706.96 MB), 613 files
CATEGORY=old extracted release
REASON=旧版下载解压副本；确认不再用于 lasso 修复对照后可删。
```

```text
PATH=C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-lasso-fix-r2
SIZE=706,955,560 bytes (706.96 MB), 612 files
CATEGORY=old extracted release
REASON=旧版下载解压副本；确认不再用于 lasso R2 对照后可删。
```

```text
PATH=C:\Users\Lenovo\Downloads\TreeLineWarning-0.1.0-Windows-x64-20260831-viewer-compat
SIZE=707,016,916 bytes (707.02 MB), 613 files
CATEGORY=old extracted release
REASON=旧版下载解压副本；确认不再用于 viewer compatibility 回归后可删。
```

```text
PATH=C:\Users\Lenovo\Downloads\TreeLineWarning-*-Test-20260831（4 个测试目录）
SIZE=2,418,722 bytes (2.42 MB), 178 files
CATEGORY=old isolated test outputs
REASON=旧版本隔离测试输出；体积很小，因属于测试资料需人工确认。
```

```text
PATH=E:\TreeLineWarning-FreshInstall-20260913-328837a
SIZE=3,091,398,393 bytes (3.09 GB), 2,959 files
CATEGORY=Final release extracted verification copy
REASON=对应发布提交 328837a 的解压回归副本；当前 Final ZIP 和 Releases 目录均受保护，确认不再需要复核 19/19 证据后才可删此副本。
```

```text
PATH=E:\TreeLineWarning-FreshInstall-20260913-4c03e25
SIZE=3,925,100,658 bytes (3.93 GB), 3,031 files
CATEGORY=pre-final extracted verification copy
REASON=较早构建 4c03e25 的解压验证副本；与当前 Final 提交不同，确认无需版本对照后可删。
```

```text
PATH=E:\pytest-temp
SIZE=338,472,354 bytes (338.47 MB), 49,127 files
CATEGORY=pytest temporary output
REASON=名称和结构显示为测试临时输出，未发现受保护扩展名；因“测试数据必须保护”，仍要求人工确认其不是基准测试数据。
```

```text
PATH=E:\TreeLineWarning_pytest_tmp
SIZE=12,087,180 bytes (12.09 MB), 3,672 files
CATEGORY=TreeLineWarning pytest temporary output
REASON=未发现 Git 标记、ZIP 或受保护扩展名；仍属于项目相关测试输出，只能确认后删除。
```

```text
PATH=C:\tmp
SIZE=33,222,578 bytes (33.22 MB), 12,298 files
CATEGORY=TreeLineWarning P3A review temporary outputs
REASON=全部一级子目录均为 p3a review/contract/focused/project 临时结果；属于项目复核证据，需人工确认。
```

```text
PATH=E:\Tencent Games
SIZE=35,345,402,100 bytes (35.35 GB), 465 files
CATEGORY=installed game
REASON=最大非项目占用之一；若已不使用，应通过 Windows/游戏平台卸载，禁止直接删除程序目录。
```

```text
PATH=E:\腾讯电脑管家软件搬家
SIZE=13,489,035,983 bytes (13.49 GB), 27,898 files
CATEGORY=migrated user/application data
REASON=含企业微信聊天文件、课程 ZIP 等用户资料；只有用户确认资料已备份或不再需要时，才可按子目录处理，不能整目录直接删除。
```

## Keep

```text
PATH=E:\TreeLineWarning-fast
SIZE=125,516,168,659 bytes (125.52 GB), 205,896 files
CATEGORY=protected primary Git repository and engineering release workspace
REASON=用户明确禁止删除；HEAD=328837af3cb74792222936150bbf79a242b7697f，包含源代码、模型、测试数据、构建/验收证据及当前 Final Release。扫描到 2,343 个受保护文件，共 10,228,240,896 bytes。
```

```text
PATH=E:\TreeLineWarning-fast\artifacts\TreeLineWarning-Engineering-Release-2026-Final.zip
SIZE=1,577,729,320 bytes (1.58 GB), 1 file
CATEGORY=protected Final Release
REASON=用户明确禁止删除；SHA256=15419473695CAAA098C7B2878B18D17F0769FF81BE9A616DB992A756FF6F8BF0，复核与给定值一致。
```

```text
PATH=E:\TreeLineWarning-Releases
SIZE=16,747,456,140 bytes (16.75 GB), 17,251 files
CATEGORY=protected release archive
REASON=用户明确禁止删除整个发布归档；其中 8 个 ZIP 共 3,162,509,062 bytes，包含工程发布版和历史版本。
```

```text
PATH=E:\TreeLineWarning
SIZE=3,314,087,704 bytes (3.31 GB), 910 files
CATEGORY=Git repository + point-cloud project data
REASON=Git 仓库；包含 3 个受保护文件，共 2,939,318,509 bytes，其中大型原始 LAS 必须保留。
```

```text
PATH=E:\TreeLineWarning-corridor-validation; E:\TreeLineWarning-fast-performance-finalization-v1; E:\TreeLineWarning-final-tree; E:\TreeLineWarning-pred-integration; E:\TreeLineWarning-review-risk-polish; E:\TreeLineWarning-source-baseline-check-5f6f4d1; E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1; E:\TreeLineWarning-v01-final-acceptance
SIZE=合计约 18.44 GB
CATEGORY=protected Git repositories/worktrees
REASON=每个路径均检测到 .git 标记；按要求禁止删除 Git 仓库。部分目录还含 LAS/模型/测试证据。
```

```text
PATH=E:\TreeLineWarning-memory-acceptance-20260822
SIZE=15,894,123,116 bytes (15.89 GB), 2,437 files
CATEGORY=protected memory-acceptance test data
REASON=包含 11 个 LAS/模型类受保护文件，共 2,798,045,327 bytes；属于测试数据和验收资料。
```

```text
PATH=E:\TreeLineWarning-source-cache
SIZE=10,450,215,899 bytes (10.45 GB), 4,799 files
CATEGORY=protected project source cache
REASON=虽未发现 .git 或指定模型/点云扩展名，但名称和位置明确属于 TreeLineWarning 项目源缓存；按项目资产保护规则保留。
```

```text
PATH=E:\2; E:\2026.8.31-2
SIZE=10,240,920,602 bytes + 14,677,773,481 bytes
CATEGORY=project/test outputs containing point clouds
REASON=两目录均发现 TreeLineWarning 运行产物和 LAS；禁止整目录删除。E:\2026.8.31-2 内仅旧 ZIP 单独列入 Confirm Delete。
```

```text
PATH=E:\ForestFormer3D_new
SIZE=231,692,419 bytes (231.69 MB), 88 files
CATEGORY=model project
REASON=包含 epoch_3000_fix.pth（230,154,197 bytes）；模型文件禁止删除。
```

```text
PATH=E:\Miniforge\envs
SIZE=5,774,955,283 bytes (5.77 GB), 69,601 files
CATEGORY=conda environments
REASON=用户明确禁止删除 conda 环境；只允许将 E:\Miniforge\pkgs 包缓存另行确认。
```

```text
PATH=C:\Users\Lenovo\Desktop\树线
SIZE=至少包含 北门_colorized.las 729,628,701 bytes 和 树线.zip 446,524,801 bytes
CATEGORY=project material and point cloud
REASON=LAS 与项目资料禁止删除。
```

```text
PATH=C:\Users\Lenovo\.cache\torch
SIZE=244,408,911 bytes (244.41 MB), 1 file
CATEGORY=model checkpoint cache
REASON=包含 alexnet-owt-7be5be79.pth；用户明确要求模型文件保留，因此不随通用 .cache 清理。
```

```text
PATH=E:\下载\litchfield.zip
SIZE=2,331,915,728 bytes (2.33 GB), 1 file
CATEGORY=possible LiDAR/test dataset archive
REASON=名称疑似点云数据集，无法在只读审计中证明为无用安装包；按测试数据保护规则保留。
```

```text
PATH=E:\腾讯电脑管家软件搬家\C盘清理文件搬家\企业微信聊天文件搬家
SIZE=包含多份课程/聊天 ZIP，父目录总计 13.49 GB
CATEGORY=user documents / course and project materials
REASON=可能包含论文、课程或项目资料；默认保留，不能按“缓存”自动删除。
```

```text
PATH=C:\Windows; C:\Program Files; C:\Program Files (x86); C:\ProgramData; C:\Recovery; C:\System Volume Information; C:\Windows\Installer; C:\ProgramData\Package Cache
SIZE=见一级目录表；Package Cache 单独为 1,035,474,124 bytes
CATEGORY=system and installed-software managed paths
REASON=禁止手工删除。软件应通过卸载器处理，Windows 组件应通过系统存储/磁盘清理处理；Windows Installer 与 Package Cache 关系到修复和卸载。
```

## 受保护扩展名汇总

| 盘符 | 扩展名 | 文件数 | 总大小（bytes） | 处理 |
|---|---:|---:|---:|---|
| C | `.las` | 912 | 745,405,818 | Keep |
| C | `.onnx` | 1,475 | 406,068,746 | Keep |
| C | `.pt` | 14 | 146,100,192 | Keep |
| C | `.pth` | 28 | 2,404,865,915 | Keep |
| E | `.copc.laz` | 9 | 1,440,108,860 | Keep |
| E | `.las` | 6,953 | 14,941,779,793 | Keep |
| E | `.laz`（不含 `.copc.laz`） | 326 | 9,759,314 | Keep |
| E | `.onnx` | 3 | 21,590,639 | Keep |
| E | `.pt` | 1 | 37,094 | Keep |
| E | `.pth` | 15 | 230,156,089 | Keep |

## ZIP / 安装包检查

- 扫描了 C/E 两盘中 50 MB 以上 ZIP，共 28 个。
- 同文件大小的 ZIP 候选组为 0，因此不存在需要进一步哈希确认的同大小重复 ZIP；不同大小的 ZIP 不可能是字节级完全重复。
- 当前 Final ZIP 为 1,577,729,320 bytes，SHA256 已复核匹配。
- `E:\TreeLineWarning-Releases\TreeLineWarning-Engineering-Release-2026.zip`（1,577,220,000 bytes）和 `TreeLineWarning-0.1.0-Windows-x64-20260908-5f6f4d1.zip`（1,576,627,672 bytes）大小不同于 Final，视为不同版本且位于受保护发布目录。
- 最大安装/压缩候选：`E:\matlab\MATLAB R2025a.rar` 14.77 GB、`E:\2026.8.31-2\2026.08.31-2.zip` 4.21 GB、`E:\下载\litchfield.zip` 2.33 GB。

## C:\ 一级目录清单

| 路径 | 大小 | 文件数 | 最近修改时间 | 备注 |
|---|---:|---:|---|---|
| `C:\$Recycle.Bin` | 2,177 B（部分） | 15 | 2025-07-31 14:19:43 | 3 个访问错误；应使用回收站界面 |
| `C:\Autodesk` | 773,546,330 B | 47 | 2025-07-31 14:59:07 | Confirm |
| `C:\db` | 523,008 B | 3 | 2025-11-02 21:08:06 | 未知，小体积 |
| `C:\Documents and Settings` | 未扫描 | 未扫描 | 2025-07-31 14:11:16 | 系统重解析点，Keep |
| `C:\GameHL` | 0 | 0 | 2026-03-09 22:59:54 | 空目录，无实际收益 |
| `C:\inetpub` | 0 | 0 | 2025-07-31 14:47:19 | IIS 系统目录，Keep |
| `C:\KDubaSoftDownloads` | 0 | 0 | 2026-05-02 14:56:19 | 空目录 |
| `C:\KRECYCLE` | 0 | 0 | 2026-03-05 16:17:16 | 软件管理目录，不手工删除 |
| `C:\OneDriveTemp` | 0 | 0 | 2025-07-31 14:51:56 | OneDrive 管理目录 |
| `C:\PerfLogs` | 0（部分） | 0 | 2024-04-01 15:26:06 | 1 个访问错误 |
| `C:\Program Files` | 55,584,547,877 B | 113,721 | 2026-09-12 18:44:50 | Keep；通过卸载器释放空间 |
| `C:\Program Files (x86)` | 23,278,873,309 B | 56,352 | 2026-08-02 20:13:10 | Keep；通过卸载器释放空间 |
| `C:\ProgramData` | 13,027,652,413 B（部分） | 31,930 | 2026-09-12 18:45:02 | 159 个访问错误；Keep/按软件清理 |
| `C:\Recovery` | 0（不可访问） | 0 | 2025-07-04 09:51:15 | Keep |
| `C:\System Volume Information` | 0（不可访问） | 0 | 2026-09-13 13:23:44 | Keep |
| `C:\tmp` | 33,222,578 B | 12,298 | 2026-08-18 00:13:52 | Confirm；TreeLineWarning P3A 临时证据 |
| `C:\Users` | 85,144,190,767 B（部分） | 454,115 | 2026-05-23 21:48:43 | 57 个访问错误；按子项分类 |
| `C:\Windows` | 48,836,634,419 B（部分） | 208,660 | 2026-09-11 21:28:20 | 48 个访问错误；Keep |

## E:\ 一级目录清单

| 路径 | 大小 | 文件数 | 最近修改时间 | 备注 |
|---|---:|---:|---|---|
| `E:\.accelerate` | 0 | 0 | 2026-01-22 11:21:19 | 空目录 |
| `E:\$RECYCLE.BIN` | 299 B | 2 | 2025-07-31 14:51:35 | 使用回收站界面 |
| `E:\1` | 1,982 B | 1 | 2026-08-23 21:25:48 | 小体积 |
| `E:\2` | 10,240,920,602 B | 1,689 | 2026-09-13 12:55:04 | Keep；含 LAS/项目结果 |
| `E:\2026.8.31-2` | 14,677,773,481 B | 1,718 | 2026-08-31 17:09:53 | Keep；仅内部旧 ZIP 为 Confirm |
| `E:\测试` | 0 | 0 | 2026-07-18 15:41:19 | 空目录 |
| `E:\测试1` | 0 | 0 | 2026-07-18 15:49:08 | 空目录 |
| `E:\港服无畏契约` | 0 | 0 | 2026-08-13 15:39:47 | 空目录 |
| `E:\近代史` | 0 | 0 | 2026-07-17 15:41:27 | 空目录 |
| `E:\雷神加速器` | 917,264,087 B | 420 | 2026-08-05 18:05:28 | 已安装软件，卸载器处理 |
| `E:\腾讯电脑管家软件搬家` | 13,489,035,983 B | 27,898 | 2026-01-26 22:13:15 | Confirm/Keep 用户资料 |
| `E:\腾讯应用宝文件管理` | 247 B | 7 | 2026-07-30 18:23:27 | 小体积 |
| `E:\下载` | 2,460,249,438 B | 22 | 2026-09-12 21:59:07 | Keep；疑似数据集压缩包 |
| `E:\小挑` | 74,161,160 B | 7 | 2026-07-17 15:47:48 | 项目资料，Keep |
| `E:\新建文件夹` | 0 | 0 | 2026-08-26 21:32:14 | 空目录 |
| `E:\新建文件夹 (2)` | 0 | 0 | 2026-09-13 12:53:27 | 空目录 |
| `E:\新建文件夹 (3)` | 0 | 0 | 2026-08-31 17:09:12 | 空目录 |
| `E:\新建文件夹 (4)` | 0 | 0 | 2026-08-31 17:09:21 | 空目录 |
| `E:\迅雷下载` | 0 | 0 | 2026-01-22 20:23:19 | 空目录 |
| `E:\迅雷云盘` | 509,575 B | 1 | 2026-01-22 20:42:22 | 小体积 |
| `E:\AweSun` | 146,832,429 B | 466 | 2026-01-26 15:39:38 | 已安装软件 |
| `E:\BaiduNetdiskDownload` | 10,239,568 B | 4 | 2026-09-13 16:29:04 | 用户下载，Confirm |
| `E:\c home work` | 39,324,274 B | 147 | 2026-05-24 20:21:18 | 资料，Keep |
| `E:\ccswitch` | 45,602,248 B | 3 | 2026-07-31 20:29:30 | 应用资料 |
| `E:\chengxushijian` | 0 | 0 | 2026-08-11 19:22:36 | 空目录 |
| `E:\clash for windows` | 0 | 0 | 2026-03-22 17:12:55 | 空目录 |
| `E:\Clash Verge` | 137,804,175 B | 11 | 2026-05-23 16:48:11 | 已安装软件 |
| `E:\cloudcom` | 0 | 0 | 2026-03-07 17:05:51 | 空目录 |
| `E:\cloudcompare` | 1,252,203,429 B | 14,815 | 2026-03-29 15:04:17 | 点云工具，Keep |
| `E:\CloudMusic` | 0 | 0 | 2025-08-09 20:17:08 | 空目录 |
| `E:\Cmake` | 139,166,320 B | 8,584 | 2026-03-28 11:31:49 | 开发工具，Keep |
| `E:\cursor` | 373,838 B | 11 | 2026-05-19 01:02:39 | 应用资料 |
| `E:\diansai` | 2,026,658,776 B | 24,282 | 2026-07-01 21:25:10 | 项目资料，Keep |
| `E:\ForestFormer3D_new` | 231,692,419 B | 88 | 2026-05-12 20:53:18 | Keep；含模型 |
| `E:\git` | 417,670,457 B | 9,302 | 2026-03-22 18:11:54 | 开发工具，Keep |
| `E:\k5` | 676,689,261 B | 1,271 | 2026-05-16 11:44:47 | 未知，Confirm |
| `E:\KDubaSoftDownloads` | 0 | 0 | 2026-05-16 11:37:44 | 空目录 |
| `E:\Ksoftware` | 0 | 0 | 2026-05-16 11:37:45 | 空目录 |
| `E:\LegionZone` | 0 | 0 | 2025-08-14 17:19:32 | 空目录 |
| `E:\LenovoSoftstore` | 0 | 0 | 2026-01-01 16:22:37 | 空目录 |
| `E:\LeStoreDownload` | 964,833,912 B | 5 | 2026-01-31 14:43:10 | Confirm 安装包 |
| `E:\matlab` | 30,663,841,926 B | 15,258 | 2026-05-19 01:07:21 | 已安装软件 + 14.77 GB RAR |
| `E:\Miniforge` | 13,154,363,894 B | 138,262 | 2026-07-19 00:00:44 | Keep 环境；pkgs 为 Confirm |
| `E:\Miniforge1` | 78,836,352 B | 1 | 2026-07-18 23:55:50 | Confirm 安装包 |
| `E:\note js` | 102,501,357 B | 1,838 | 2026-03-22 18:04:36 | 项目/笔记，Keep |
| `E:\Program Files` | 12,078,109,609 B | 1,003 | 2026-05-30 19:31:10 | 已安装软件；卸载器处理 |
| `E:\pytest-temp` | 338,472,354 B | 49,127 | 2026-08-04 19:18:10 | Confirm |
| `E:\Python312` | 123,484,997 B | 4,039 | 2026-05-29 19:26:20 | Python 环境，Keep |
| `E:\Qt` | 3,534,087,803 B | 63,398 | 2026-03-28 12:46:57 | 开发环境，Keep |
| `E:\rrdownload` | 713,680 B | 7 | 2026-01-22 22:35:30 | 小体积下载 |
| `E:\share` | 0 | 0 | 2026-08-22 18:08:02 | 空目录 |
| `E:\Steam++` | 130,788,524 B | 187 | 2026-03-05 16:42:41 | 已安装软件 |
| `E:\stm32` | 0 | 0 | 2026-08-22 18:07:28 | 空目录/项目占位，Keep |
| `E:\System Volume Information` | 0（不可访问） | 0 | 2026-05-24 17:42:38 | Keep |
| `E:\Tencent Games` | 35,345,402,100 B | 465 | 2025-07-31 16:02:42 | 最大卸载候选；只能卸载器处理 |
| `E:\Thunder` | 9,103,008 B | 32 | 2026-07-18 15:48:16 | 已安装软件 |
| `E:\tmp` | 606,131 B | 285 | 2026-08-06 20:09:42 | Safe，收益极小 |
| `E:\TreeLearn` | 6,169,038 B | 70 | 2026-05-16 14:10:21 | 项目资料，Keep |
| `E:\TreeLineWarning` | 3,314,087,704 B | 910 | 2026-08-08 21:59:17 | Keep Git + LAS |
| `E:\TreeLineWarning_pytest_tmp` | 12,087,180 B | 3,672 | 2026-07-28 17:13:16 | Confirm |
| `E:\TreeLineWarning-ai-qc-assets` | 39,482 B | 2 | 2026-09-08 22:26:35 | 项目资料，Keep |
| `E:\TreeLineWarning-ai-qc-benchmark` | 2,297,281 B | 100 | 2026-09-08 18:43:43 | 项目基准，Keep |
| `E:\TreeLineWarning-ai-qc-benchmark-v2` | 6,381,311 B | 164 | 2026-09-08 21:15:33 | 项目基准，Keep |
| `E:\TreeLineWarning-ai-qc-pytest-final` | 332,956 B | 30 | 2026-09-08 21:19:36 | 测试证据，Keep |
| `E:\TreeLineWarning-ai-qc-pytest-v2` | 332,955 B | 30 | 2026-09-08 20:45:31 | 测试证据，Keep |
| `E:\TreeLineWarning-ai-qc-pytest-v2b` | 332,956 B | 30 | 2026-09-08 20:53:55 | 测试证据，Keep |
| `E:\TreeLineWarning-ai-qc-test-tmp` | 246,679 B | 24 | 2026-09-08 18:48:47 | 测试证据，Keep |
| `E:\TreeLineWarning-ai-qc-test-tmp-2` | 246,678 B | 24 | 2026-09-08 18:49:11 | 测试证据，Keep |
| `E:\TreeLineWarning-cleanup-audit` | 134,651,891 B | 27 | 2026-09-08 22:29:50 | 项目审计资料，Keep |
| `E:\TreeLineWarning-corridor-validation` | 1,155,591,474 B | 2,011 | 2026-08-28 22:32:40 | Keep Git + protected files |
| `E:\TreeLineWarning-D-cleanup-audit` | 20,822,991 B | 19 | 2026-09-08 22:44:01 | 项目审计资料，Keep |
| `E:\TreeLineWarning-fast` | 125,516,168,659 B | 205,896 | 2026-09-13 14:23:19 | Keep，明确保护 |
| `E:\TreeLineWarning-fast-baseline-pytest-temp` | 1,343,827 B | 466 | 2026-09-08 13:56:58 | 含受保护文件，Keep |
| `E:\TreeLineWarning-fast-performance-finalization-v1` | 145,074,972 B | 8,652 | 2026-09-08 16:53:28 | Keep Git |
| `E:\TreeLineWarning-final-tree` | 426,959,140 B | 5,067 | 2026-08-09 00:38:39 | Keep Git |
| `E:\TreeLineWarning-FreshInstall-20260913-328837a` | 3,091,398,393 B | 2,959 | 2026-09-13 14:20:10 | Confirm |
| `E:\TreeLineWarning-FreshInstall-20260913-4c03e25` | 3,925,100,658 B | 3,031 | 2026-09-13 14:11:39 | Confirm |
| `E:\TreeLineWarning-geometry-qc-baseline-check` | 10,455 B | 2 | 2026-09-08 21:42:31 | 测试资料，Keep |
| `E:\TreeLineWarning-geometry-qc-regression` | 26,741,108 B | 633 | 2026-09-08 21:41:50 | 含受保护文件，Keep |
| `E:\TreeLineWarning-memory-acceptance-20260822` | 15,894,123,116 B | 2,437 | 2026-08-22 12:46:34 | Keep 点云测试数据 |
| `E:\TreeLineWarning-pred-integration` | 1,537,164 B | 131 | 2026-08-17 17:15:08 | Keep Git |
| `E:\TreeLineWarning-pytest-temp` | 1,513,610 B | 832 | 2026-08-29 18:35:49 | 测试资料，Keep |
| `E:\TreeLineWarning-Releases` | 16,747,456,140 B | 17,251 | 2026-09-13 12:24:13 | Keep，明确保护 |
| `E:\TreeLineWarning-review-risk-polish` | 13,846,388,746 B | 29,975 | 2026-08-27 22:03:40 | Keep Git + protected files |
| `E:\TreeLineWarning-source-baseline-check-5f6f4d1` | 20,428,950 B | 725 | 2026-09-13 14:22:03 | Keep Git |
| `E:\TreeLineWarning-source-cache` | 10,450,215,899 B | 4,799 | 2026-08-23 21:02:20 | Keep 项目源缓存 |
| `E:\TreeLineWarning-tree-numbering-final-pytest-5bb792af61d940e3a9dd0bdb1f6ff38e` | 396,852 B | 17 | 2026-08-29 19:25:51 | 含受保护文件，Keep |
| `E:\TreeLineWarning-tree-numbering-pytest-794d1352c3624eb2b33fb38b773acf73` | 396,849 B | 17 | 2026-08-29 19:14:15 | 含受保护文件，Keep |
| `E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1` | 71,979,353 B | 971 | 2026-09-13 01:06:49 | Keep Git |
| `E:\TreeLineWarning-v01-final-acceptance` | 2,897,978,649 B | 5,960 | 2026-08-29 22:24:45 | Keep Git + test data |
| `E:\uc` | 0 | 0 | 2026-05-12 20:11:33 | 空目录 |
| `E:\visual studio2` | 3,224,597,460 B | 1,417 | 2026-07-23 10:06:12 | 开发/安装资产，卸载器处理 |
| `E:\VScode` | 3,668,614,878 B | 9,106 | 2026-03-28 13:51:28 | 已安装开发工具 |

## 后续执行边界

本报告不构成删除授权。下一步只能根据用户明确选择的 `PATH` 清理；清理前应重新测量目标、检查进程占用，并再次排除：

- `E:\TreeLineWarning-fast`
- `E:\TreeLineWarning-Releases`
- 当前 Final Release ZIP
- 任何 Git 仓库/工作树
- `E:\Miniforge\envs` 及其他 conda 环境
- `*.las`、`*.laz`、`*.copc.laz`、`*.onnx`、`*.pth`、`*.pt`
- 模型、测试数据、论文和项目资料
