# TreeLineWarning Windows 用户指南

本发行版以 **Survey Core** 为验收范围。分析请选择 AUTO 或 SERIAL；Parallel 未启用。
保留的本地未来点云预览是实验性示意功能，未纳入发行验收，不是预测或安全结论。
独立 Windows 计算机验收尚待完成。

## 使用前准备

- 使用 Windows x64 计算机，并准备一个用户有读写权限的本地文件夹。
- 将整个 `TreeLineWarning` 文件夹解压到该文件夹，再运行同目录下的
  `TreeLineWarning.exe`。必须保留 `_internal` 及其全部子目录。
- 用户不需要安装 Python、Conda、PDAL、PyInstaller 或开发工具。应用使用包内的
  Python 扩展、PDAL/GDAL/PROJ 文件和静态网页资源。
- 桌面壳需要 Microsoft Edge WebView2 Runtime 以及 .NET Framework 4.6.2 或更高版本。
  当前包若附带 `webview2_offline_prerequisite/offline_verified/`
  `MicrosoftEdgeWebView2RuntimeInstallerX64.exe`，可由管理员或 IT 人员手动运行；应用
  不会自动安装它，也不会把在线 bootstrapper 当作离线安装包。安装后重新启动应用。
- 不要直接从压缩包、只读网络盘或 `Program Files` 下的无写权限目录运行。项目、缓存和
  诊断文件需要写入用户指定的工作目录。路径包含中文字符或空格时，应移动整个文件夹，
  不要只移动 exe。

发行包中的离线安装程序只接受 `offline_verified` 目录内的 x64 standalone 文件。其
`PACKAGED_ACCEPTANCE_CONTRACT.md` 给出相对路径、字节数和 SHA-256；同目录的
`authenticode.json`、`offline_download.json` 和 `official_webview2_eula.json` 是配套审阅记录。
`offline/` 目录、顶层同名 exe 和其它失败下载不是包输入。

## 基本流程

1. 点击“新建项目”或“打开项目”。项目目录应保持可写；原始 LAS/LAZ 文件可以放在
   其它位置，应用不会用项目编辑覆盖原始点云。
2. 在“点云”步骤导入 LAS/LAZ，查看能力、坐标系和单位后确认。需要线路地理信息时，
   可另行导入 GIS；缺少可验证的坐标或单位时，按界面提示补充确认。
3. 在“杆塔”步骤沿线路方向人工选择杆塔并确认顺序。至少选择两根杆塔后才会生成逻辑
   档距；线路和档距顺序应由现场数据负责人员复核。
4. 在“分析”步骤选择当前档、选中档或全部档距。失败档距可以使用“重试失败任务”；
   页面会保留成功档距并显示失败原因。
5. 在“单木复核”中检查树木、导线和净空结果。需要时使用选择、删除、合并和撤销，
   再点击“完成复核”。完成步骤后，使用按钮 **“打开当前最终报告”** 查看当前报告。
   报告只展示当前观测结果，不进行预测；证据不足的结果会显示需复核或无可用距离，
   不应据此作安全结论。
6. 复核完成且界面显示可导出后，点击“导出预测输入”生成项目 `exports` 目录中的
   `Prediction_Input.xlsx`。该文件供后续业务流程使用；它不表示本程序已经完成风险处置。

报告中的 DBH 缺失时保持空白，不以零代替。物理线路 CONTACT 表示接触状态，不能把
“距离为零”当作 CONTACT；请按现场证据复核这两类结果。

应用的网页和 API 在本机 `127.0.0.1` 上运行，不加载在线底图。项目和缓存保存在本机；
请按组织的信息安全规则备份、传输和删除这些文件。

## 诊断和日志

窗口化启动可能没有控制台输出。为每次诊断指定一个此前不存在的 JSON 路径；应用拒绝
覆盖旧文件。例如：

```powershell
.\TreeLineWarning.exe --dependency-health `
  --diagnostic-output "$env:USERPROFILE\Desktop\treeline-dependency-20260830-a.json"
```

已支持的只读诊断参数包括：

- `--dependency-health`：检查冻结资源、模块来源和 Pillow/PDAL 原生操作；
- `--survey-smoke --project-root <项目目录>`：读取当前 Survey 报告和已保存的 W1 导线
  曲线，不启动分析、不编辑、不完成复核、不导出；
- `--backend-smoke --project-root <项目目录> --diagnostic-output <新 JSON>`：在有限时间内
  发布本地后端地址，配合 `--stop-signal <新文件>` 停止；
- `--diagnostic-output <新 JSON>`（别名 `--diagnostic-path`）：保存诊断结果；
- `--smoke-timeout-seconds <秒>`：为 smoke 诊断设置有限时长。

诊断文件、`build_receipt.json`、`release_file_manifest.json` 和
`PACKAGED_ACCEPTANCE_CONTRACT.md` 应与问题报告一起保留。不要为了启动应用设置
`PYTHONPATH`、Conda 变量、PDAL 路径或开发机的 DLL 路径；这些设置会让诊断失去意义。

## 常见问题

**双击后没有窗口或提示缺少 WebView2。** 确认已完整解压并保留 `_internal`，再检查
WebView2 Runtime 和 .NET Framework。若包中有上述 `offline_verified` 安装程序，由 IT
人员手动安装后再重启；TreeLineWarning 不会自动安装或静默改用 MSHTML。

**项目或缓存无法保存。** 将整个文件夹移到用户可写的本地路径；不要只复制 exe，也不要
从压缩包内运行。重新打开项目后再执行诊断。

**导入点云后显示不可用或需复核。** 先确认 LAS/LAZ 文件可读、单位和坐标系声明正确，
再查看对应档距的失败原因。不要用缺少原生依赖或不完整证据的结果替代现场复核。

**需要提交启动问题。** 用新的诊断 JSON 路径运行 `--dependency-health`，同时提供包内
`VERSION.txt`、`PACKAGED_ACCEPTANCE_CONTRACT.md` 和诊断 JSON。不要覆盖已有诊断文件。

## 交付边界

包内 `VERSION.txt` 和验收合同会标记 `Survey Core`、
`parallel = BOUNDED_DETERMINISTIC_INCLUDED`、可选的 CuPy CUDA 13 认证筛选，以及
`preview = EXPERIMENTAL_NOT_ACCEPTED`。GPU 路径仍由 CPU float64 做最终判定；无兼容
NVIDIA GPU、驱动初始化失败或显存不足时使用 CPU。GPU 用户只需兼容 NVIDIA 驱动，包内
已带运行组件，不要求安装完整 CUDA Toolkit。当前工作机已有 WebView2/.NET；干净计算机安装、启动和完整
现场工作流仍标记为 **NOT_TESTED**，不能从开发机成功运行推断干净机已验收。第三方通知
文件是工程审阅清单，不是法律许可；其中标出的原生二进制来源仍需发行方完成相应审阅。
