# TreeLineWarning Final External Release Acceptance

## Release Information

- Release: `TreeLineWarning Engineering Release 2026 Final`
- Source baseline: `5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9`
- Release commit: `328837af3cb74792222936150bbf79a242b7697f`
- Build UTC: `2026-09-13T06:14:55.865383Z` to `2026-09-13T06:16:11.344155Z`
- PyInstaller elapsed: `74.500 s`
- Source changed during build: `false`
- Tracked source SHA-256: `76653180043e1a8f29dad11e5fe34c08553d2ee90b87da4bfc97f72dcf702139`
- ZIP SHA-256: `15419473695CAAA098C7B2878B18D17F0769FF81BE9A616DB992A756FF6F8BF0`
- ZIP bytes: `1,577,729,320`
- Package bytes: `2,257,947,118`
- EXE SHA-256: `4B4BF21964BF4C0B1705BD8DBC18BF2F3E4A1B737A07065347AEB9CB0754EE75`
- File-manifest SHA-256: `FDC01DD094E823D0A3C4447812BD8833C78E9287186FA9687D09013ABFD275E6`
- Package inventory verification: `PASS (2860/2860 files)` before ZIP and after fresh extraction

## Fixed Issues

### 删除同步

`DELETE_FLOW=POST /api/select -> POST /api/delete -> detached edit session -> atomic tree_edits.json replace -> live session swap -> editor revision delta -> final projection`

`EDIT_STORAGE=artifacts/project-runs/batches/<analysis_run_id>/tree_edits.json; original NPZ remains immutable`

`FINAL_PROJECTION=BatchTreeEditor final membership replay; keyed by persistent_tree_id across renumbering`

`REVIEW_SOURCE=DesktopWorkflow.editor_data -> editor snapshot + stable display basis + revision delta`

`RESULT_SOURCE=DesktopWorkflow.final_projection -> editor.final_reconciled; never pre-edit global reconciliation`

`EXPORT_SOURCE=current finalized gate -> the same final projection/final reconciliation for XLSX and HTML`

`CACHE_SOURCE=editor revision + display basis id; edit commit invalidates snapshot/display caches and resets stale DONE publication state to IDLE`

验证结果：

- DELETE 后 review、result 立即同时减少，旧 current report/XLSX 明确锁定。
- 重新定稿后 review=result=HTML=XLSX；真实批次从 17 棵变为 16 棵，被删除持久 ID 在所有输出中消失。
- UNDO 后 review、result、report 恢复。
- MERGE 后当前树数减少 1，UNDO 恢复。
- 编辑只写 edits 层；未修改 `tree_segmentation_original.npz`。

### 单木复核性能

`PERFORMANCE_ROOT_CAUSE=Whole-tree DELETE entered the same full clearance recalculation path as membership-changing edits; repeated export was also invalidated by post-export performance metadata timestamp publication.`

`OPTIMIZATION_CHANGE=Strict persistent-ID/point-count inventory proof permits exact filtering of already-computed survivor clearance; no KDTree rebuild or distance evaluation. Existing no-edit clearance is reused. Viewer uses revision delta. Workbook timestamp follows performance metadata publication so repeated download uses the verified cache.`

真实 3,766,354 成员点批次：

`BEFORE=CLEARANCE_CALC_TIME=4.967562400s; INDEX_BUILD_TIME=0.000405300s; full exact points=3,766,354`

`AFTER=CLEARANCE_CALC_TIME=0.000146700s; INDEX_BUILD_TIME=0; CLEARANCE_SOURCE=EXACT_DELETE_ONLY_FILTER; DELETE_COMMIT_AND_VIEWER_DELTA=0.039605s`

最终定稿性能记录：

```text
TREE_LOAD_TIME=0.000639500
WIRE_LOAD_TIME=0.000005900
INDEX_BUILD_TIME=0.000000000
CLEARANCE_CALC_TIME=0.000146700
RISK_UPDATE_TIME=0.000182600
VIEWER_UPDATE_TIME=0.002146300
TOTAL_TIME=6.178939500
```

`TOTAL_TIME` 包含 3,766,354 点 final membership NPZ 投影/持久化；净空重算本身已消除。进入复核读取为 `0.023532 s`。

重复 XLSX 下载（修复后 r3 解压包）：`0.019933 s`、`0.011730 s`，文件哈希相同。

```text
GPU_USED_FOR=CERTIFIED_CLEARANCE_CANDIDATE_SCREENING
GPU_NOT_USED_FOR=GROUND,HAG,WIRE_RECONSTRUCTION,TREE_MEMBERSHIP,DBH,RISK,VIEWER
```

GPU 只做有证书的候选筛选；CPU float64 精确判定仍为科学权威。没有为本任务修改 Ground、HAG、Wire reconstruction、Tree membership science、DBH、Clearance definition 或 Risk threshold。

### XLSX / HTML 导出

`XLSX_GENERATION=PASS; authoritative finalized rows; valid OOXML ZIP signature; 47 columns`

`FILE_PATH=Native WebView2 SAVE_DIALOG destination; Unicode/no-extension path verified; atomic temporary-copy + os.replace`

`DOWNLOAD_RESPONSE=HTTP 200 application/vnd.openxmlformats-officedocument.spreadsheetml.sheet with Content-Disposition; stale edit returns HTTP 409`

`WEBVIEW_DOWNLOAD=Native pywebview export_excel/export_html bridge; fetch/blob same-window fallback retained`

`PACKAGED_TEST=PASS from freshly extracted r3 EXE; after-delete workbook row count=1 for 2-tree interaction fixture; real batch 17 -> 16`

HTML 是自包含 UTF-8 最终报告，来自同一 gated final projection。

## Validation

### Release-focused automated tests

- Final release suite: `84 passed` in `7.53 s`.
- GPU production suite: `6 passed`; GPU count `1`; certified CPU exact match `true`.
- JavaScript syntax: desktop, current report, tree review all `PASS`.
- Ruff on changed Python: `PASS`.
- Build contract and release builder tests: `PASS`.

### Broad repository sweep

- Broad sweep result: `4102 passed, 6 skipped, 62 failed`.
- The release-focused set above is green. Broad failures were audited:
  - output containment tests were caused by placing `--basetemp` on `E:` instead of the Windows temp root; rerun from the system temp root passed those tests;
  - developer `.venv311` initially lacked PDAL/Streamlit, while packaged PDAL/PROJ/LAS/LAZ diagnostics pass;
  - 13 failures came from user-owned untracked experimental parallel files, excluded from the 648-file release source snapshot;
  - nine V4 golden-digest assertions also fail unchanged at the supplied source commit `5f6f4d1`; the release diff touches none of the frozen science paths and does not update those goldens.

### Functional regression

```text
START=PASS; fresh r3 EXE dependency health exit 0; GUI window handle/title observed
IMPORT=PASS; packaged endpoint created/imported a valid LAS
ANALYSIS=PASS; AUTO execution interaction fixture completed
VIEWER=PASS; editor data and 3D display sample loaded
DELETE_SYNC=PASS; review=result immediately; report=XLSX=HTML after finalize
UNDO_SYNC=PASS; review=result=report restored
MERGE=PASS; tree count -1; merge undo restored
XLSX=PASS; valid workbook; native-save contract and packaged HTTP download
HTML=PASS; self-contained final-state report
```

### Fresh-install simulation

Fresh extraction root: `E:\TreeLineWarning-FreshInstall-20260913-328837a`

```text
FRESH_INSTALL=PASS; regenerated ZIP extracted outside repository
FIRST_START=PASS_WITH_VISUAL_AUTOMATION_LIMIT; TreeLineWarning window handle and title observed
IMPORT=PASS
AUTO_ANALYSIS=PASS (AUTO execution on packaged interaction fixture)
TREE_REVIEW=PASS
DELETE=PASS
UNDO=PASS
RESULT_VIEW=PASS
XLSX_DOWNLOAD=PASS
HTML_EXPORT=PASS
CPU_MODE=PASS
GPU_MODE=PASS; RTX 5070, CUDA device ready, certificate pass, CPU exact match
```

Additionally, the freshly extracted r3 EXE loaded a copied real analyzed survey batch and passed all current-report checks: finalized gate, canonical rows, persisted W1 separate segments, and fitted-current geometry.

## Known Limitations

- Automated Windows accessibility/screenshot driving was unavailable because the Computer Use trusted RPC service was not configured. The real WebView2 window was observed by native handle/title, while functional UI contracts were exercised through the same packaged loopback backend. Automated visual layout acceptance remains unverified.
- The GUI automation close request did not finish within 20 seconds and the test process was terminated; normal interactive close was not visually driven.
- The interaction AUTO run uses the packaged deterministic acceptance fixture. Real scientific data was validated by loading the existing real analyzed/finalized batch; this stabilization run did not repeat the multi-gigabyte raw scientific analysis.
- Parallel execution remains unavailable in this release UI. AUTO and SERIAL are supported.
- Systems without WebView2 require the included offline installer to be run manually by the user or IT.
- The supplied baseline's pre-existing V4 golden digest mismatch remains unchanged; science code was frozen for this task.

## Release Package

- Package directory: `E:\TreeLineWarning-fast\artifacts\final-build-dist-20260913-328837a-r3\TreeLineWarning-Engineering-Release-2026-Final`
- ZIP: `E:\TreeLineWarning-fast\artifacts\TreeLineWarning-Engineering-Release-2026-Final.zip`
- Fresh-install evidence: `E:\TreeLineWarning-FreshInstall-20260913-328837a\acceptance`

```text
TREELINEWARNING_FINAL_EXTERNAL_RELEASE

SOURCE=5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9
COMMIT=328837af3cb74792222936150bbf79a242b7697f
BUG_FIX_STATUS=PASS
PERFORMANCE_STATUS=PASS
EXPORT_STATUS=PASS
PACKAGE_PATH=E:\TreeLineWarning-fast\artifacts\final-build-dist-20260913-328837a-r3\TreeLineWarning-Engineering-Release-2026-Final
ZIP_PATH=E:\TreeLineWarning-fast\artifacts\TreeLineWarning-Engineering-Release-2026-Final.zip
SHA256=15419473695CAAA098C7B2878B18D17F0769FF81BE9A616DB992A756FF6F8BF0
FINAL_STATUS=READY_FOR_EXTERNAL_USE
```
