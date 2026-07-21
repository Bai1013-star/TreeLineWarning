# T1A Review Fix: feature/io 骨架、最小契约和测试框架

## 当前分支和基线提交

- 当前分支：`feature/io`
- 基线提交：`72f9935 chore: initialize repository and add project documentation`
- 创建方式：从 `main` 的 `72f9935` 创建并切换
- 未创建 Worktree
- 未 commit
- 未 push

## 环境定位完整结果

### 1. Get-Command python -ErrorAction SilentlyContinue

```text
Exit code: 1
Output: <empty>
```

### 2. Get-Command conda -ErrorAction SilentlyContinue

```text
Exit code: 1
Output: <empty>
```

### 3. where.exe python

```text
INFO: Could not find files for the given pattern(s).
```

### 4. where.exe conda

```text
INFO: Could not find files for the given pattern(s).
```

### 5. Test-Path E:\Miniforge

```text
False
```

### 6. Test-Path E:\Miniforge3

```text
True
```

### 7. Get-ChildItem E:\ -Directory | Where-Object { $_.Name -like "Miniforge*" }

```text
目录: E:\

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2026/7/18     23:55                Miniforge1
d-----         2026/7/19      0:00                Miniforge3
```

### Conda 和 treeline 环境检查

```text
Test-Path E:\Miniforge1\Scripts\conda.exe -> False
Test-Path E:\Miniforge3\Scripts\conda.exe -> True
Test-Path E:\Miniforge1\envs\treeline\python.exe -> False
Test-Path E:\Miniforge3\envs\treeline\python.exe -> True
```

实际可用基础环境：

```text
E:\Miniforge3\envs\treeline
```

`E:\Miniforge3\Scripts\conda.exe env list`：

```text
# conda environments:
#
# * -> active
# + -> frozen
base                     E:\Miniforge3
treeline                 E:\Miniforge3\envs\treeline
```

Python 版本：

```text
E:\Miniforge3\envs\treeline\python.exe --version
Python 3.11.15
```

## 实际完成内容

- 保持在 `feature/io` 分支完成 T1A-REVIEW-FIX。
- 创建最小 Python 项目配置：`pyproject.toml`。
- 创建最小 Python 包骨架：
  - `core/`
  - `core/contracts/`
  - `core/config/`
  - `core/pipeline/`
  - `algorithms/`
  - `algorithms/io/`
  - `tests/`
  - `tests/contracts/`
  - `tests/io/`
- 修正 T1A 最小公共契约，使字段和接口名称与审查意见一致。
- 补充契约测试，覆盖 raw 只读规则、序列化、边界校验、chunk 数组一致性、进度和取消接口。
- 未安装、升级或卸载任何依赖。

## 修改文件列表

```text
algorithms/__init__.py
algorithms/io/__init__.py
core/__init__.py
core/config/__init__.py
core/contracts/__init__.py
core/contracts/_json.py
core/contracts/errors.py
core/contracts/pointcloud.py
core/contracts/progress.py
core/pipeline/__init__.py
pyproject.toml
reviews/T1A-review.md
tests/__init__.py
tests/contracts/__init__.py
tests/contracts/test_pointcloud_contracts.py
tests/contracts/test_progress_contracts.py
tests/io/__init__.py
```

## 修改后的契约字段

### PointCloudSource

- `source_id: str`：必填、非空。
- `path: pathlib.Path`：来源路径。
- `role: Literal["raw", "sample", "processed"]`。
- `size_bytes: int | None`：文件大小，单位 byte，非负。
- `readonly_required: bool`：默认 `True`；`role == "raw"` 时必须为 `True`。
- `sha256: str | None`。
- `crs_wkt: str | None`。
- `metadata: Mapping[str, Any]`：必须 JSON-safe。

### LasHeaderSummary

- `source_id: str`：必填、非空。
- `source_path: pathlib.Path`。
- `file_size_bytes: int`：单位 byte，非负。
- `las_version: str`。
- `point_format_id: int`：非负。
- `point_record_length_bytes: int`：必须为正整数。
- `point_count: int`：非负。
- `min_xyz / max_xyz: tuple[float, float, float]`：单位米，`min <= max`。
- `scale / offset: tuple[float, float, float]`：单位米。
- `dimensions / extra_dimensions: tuple[str, ...]`。
- `has_rgb / has_intensity / has_classification: bool`。
- `estimated_record_memory_bytes: int | None`：非负。
- `metadata: Mapping[str, Any]`：必须 JSON-safe。

### PointCloudChunk

- `source_id: str`：必填、非空。
- `chunk_id: int`：非负。
- `start_index: int`：非负。
- `point_count: int`：非负。
- `xyz: Any | None`：如果存在，必须为 `N x 3`。
- `original_index: Any | None`：如果存在，长度必须等于 `point_count`。
- `attributes: Mapping[str, Any]`。
- `bounds_xyz: tuple[Vector3, Vector3] | None`。
- `is_last: bool`。

说明：`original_index` 保留用于后续追溯原始 LAS 点，本阶段不删除、不改写真实点数据。

### CropBounds

- `min_xyz: tuple[float, float, float]`：单位米。
- `max_xyz: tuple[float, float, float]`：单位米。
- `coordinate_mode: Literal["source", "local"] = "source"`。
- `origin_xyz: tuple[float, float, float] | None`。

校验：

- 每个轴 `min_xyz < max_xyz`。
- `coordinate_mode == "local"` 时必须提供 `origin_xyz`。

### CropResult

- `source_id: str`：必填、非空。
- `bounds: CropBounds`。
- `input_point_count: int`：非负。
- `output_point_count: int`：非负，且不得超过 `input_point_count`。
- `output_path: pathlib.Path`。
- `header_preserved: bool`。
- `dimension_names: tuple[str, ...]`。
- `warnings: tuple[str, ...]`。
- `started_at / finished_at: str | None`。
- `metadata: Mapping[str, Any]`：必须 JSON-safe。

### UnifiedError

- `code: str`：必填、非空。
- `message: str`：必填、非空。
- `severity: Literal["info", "warning", "error", "fatal"]`。
- `module: str`：必填、非空。
- `recoverable: bool`。
- `details: Mapping[str, Any]`：必须 JSON-safe。
- `cause: str | None`。

### PipelineProgress

- `task_id: str`：必填、非空。
- `stage: str`：必填、非空。
- `status: Literal["pending", "running", "cancelled", "failed", "completed"]`。
- `processed_points: int | None`：非负。
- `total_points: int | None`：非负。
- `percent: float | None`：存在时为 0 到 100。
- `message: str | None`。
- `error: UnifiedError | None`。
- `metadata: Mapping[str, Any]`：必须 JSON-safe。

校验：

- `processed_points` 不得超过 `total_points`。

### ProgressReporter

```python
ProgressReporter.update(progress: PipelineProgress) -> None
```

### CancellationToken

```python
CancellationToken.is_cancelled() -> bool
CancellationToken.cancel() -> None
```

已移除旧的 `report()` 和属性式 `is_cancelled` 命名，避免后续模块接口混乱。

## 未实现内容

- 未读取 LAS 文件头。
- 未读取完整 LAS。
- 未读取 LAS 点数组。
- 未实现分块加载。
- 未实现真实裁剪。
- 未实现数据库。
- 未实现语义、导线、树木、距离、风险、预测或 UI。
- 未实现 `ManualCorrection` 等后续契约。
- 未创建 `treeline-ml`。
- 未下载模型代码或权重。

## pytest 实际结果

命令：

```powershell
E:\Miniforge3\envs\treeline\python.exe -m pytest
```

完整结果：

```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\TreeLineWarning
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2, dash-4.4.0, cov-7.1.0
collected 17 items

tests\contracts\test_pointcloud_contracts.py ...........                 [ 64%]
tests\contracts\test_progress_contracts.py ......                        [100%]

============================= 17 passed in 0.14s ==============================
```

通过数量：17。

## ruff 实际结果

命令：

```powershell
E:\Miniforge3\envs\treeline\python.exe -m ruff check .
```

完整结果：

```text
All checks passed!
```

说明：曾运行一次 `ruff check . --fix`，仅修复测试文件导入排序；未安装或升级 ruff。

## compileall 实际结果

命令：

```powershell
E:\Miniforge3\envs\treeline\python.exe -m compileall algorithms core tests
```

完整结果：

```text
Listing 'algorithms'...
Compiling 'algorithms\__init__.py'...
Listing 'algorithms\io'...
Compiling 'algorithms\io\__init__.py'...
Listing 'core'...
Listing 'core\config'...
Compiling 'core\config\__init__.py'...
Listing 'core\contracts'...
Listing 'core\pipeline'...
Compiling 'core\pipeline\__init__.py'...
Listing 'tests'...
Listing 'tests\contracts'...
Compiling 'tests\contracts\test_pointcloud_contracts.py'...
Compiling 'tests\contracts\test_progress_contracts.py'...
Listing 'tests\io'...
Compiling 'tests\io\__init__.py'...
```

## git status

命令：

```powershell
git status --short --branch
```

结果：

```text
## feature/io
?? algorithms/
?? core/
?? pyproject.toml
?? reviews/T1A-review.md
?? tests/
```

## git diff --stat

命令：

```powershell
git diff --stat
```

结果：

```text

```

说明：当前所有变更仍为未跟踪新文件，未执行 `git add`，因此 `git diff --stat` 无输出。

补充新增文件清单来自：

```powershell
git ls-files --others --exclude-standard
```

结果：

```text
algorithms/__init__.py
algorithms/io/__init__.py
core/__init__.py
core/config/__init__.py
core/contracts/__init__.py
core/contracts/_json.py
core/contracts/errors.py
core/contracts/pointcloud.py
core/contracts/progress.py
core/pipeline/__init__.py
pyproject.toml
reviews/T1A-review.md
tests/__init__.py
tests/contracts/__init__.py
tests/contracts/test_pointcloud_contracts.py
tests/contracts/test_progress_contracts.py
tests/io/__init__.py
```

## git diff --check

命令：

```powershell
git diff --check
```

结果：

```text

```

说明：当前所有变更仍为未跟踪新文件，未执行 `git add`，因此 `git diff --check` 无输出。

## 测试覆盖摘要

- `PointCloudSource` raw 只读规则。
- Path JSON 安全序列化和恢复。
- `LasHeaderSummary` 非法点数、记录长度和边界。
- `PointCloudChunk` 的 `xyz` 形状校验。
- `original_index` 长度与点数一致校验。
- `CropBounds` 的 `source` / `local` 模式。
- `CropResult` 输入输出点数校验。
- `UnifiedError` 四种 severity。
- `PipelineProgress` 状态、百分比和进度数量。
- `CancellationToken` 取消前后状态。
- `ProgressReporter.update()` 最小假实现。

## 已知问题和风险

- `python` 和 `conda` 不在 PATH；必须使用实际路径 `E:\Miniforge3\envs\treeline\python.exe` 或 `E:\Miniforge3\Scripts\conda.exe`。
- 文档中给出的 `E:\Miniforge\envs\treeline` 在当前机器不可见，实际环境为 `E:\Miniforge3\envs\treeline`。
- `git diff --stat` 和 `git diff --check` 对未跟踪文件无输出；本次未执行 `git add`。
- `PointCloudChunk` 当前只做最小数组形状和长度校验，不执行真实 LAS 加载。

## 原始 LAS 确认

- 未读取完整 LAS。
- 未读取 LAS 点数组。
- 未调用 LAS 解析库。
- 未运行任何裁剪。
- 未向 `data/raw` 写入任何文件。
- 未修改、移动、删除或覆盖 `data/raw` 中任何文件。
- 本次修复没有访问 `data/raw` 内容。

## 是否真正建议 commit

建议进入 GPT 再次审查。若审查通过，建议 commit T1A，因为：

- pytest：17 passed。
- ruff：All checks passed。
- compileall：通过。
- 契约字段和接口名称已按 T1A-REVIEW-FIX 统一。
- 未触碰原始 LAS 和后续算法范围。

## T1A-FINAL-FIX Update

### Scope

Only the approved environment-path documentation fix was made after GPT review.
No contract fields, tests, algorithms, LAS files, or `data/raw` contents were changed.

### Documentation Files

Updated:

```text
docs/requirements.md
docs/codex_tasks.md
```

Checked and already correct:

```text
docs/TreeLineWarning_requirements_v2.3.md
docs/TreeLineWarning_codex_tasks_v2.3.md
```

Path replacement applied:

```text
E:\Miniforge\envs\treeline -> E:\Miniforge3\envs\treeline
E:\Miniforge\envs\treeline-ml -> E:\Miniforge3\envs\treeline-ml
```

The `treeline-ml` line was updated only to remove the stale `E:\Miniforge` prefix from
the documented environment root. No model environment was created.

### Stale Path Search

Command:

```powershell
rg -n "E:\\Miniforge\\envs\\treeline" docs\requirements.md docs\codex_tasks.md docs\TreeLineWarning_requirements_v2.3.md docs\TreeLineWarning_codex_tasks_v2.3.md
```

Result:

```text
Exit code: 1
Output: <empty>
```

Meaning: the wrong path no longer appears in the four project documents.

Confirmation search:

```powershell
rg -n "E:\\Miniforge3\\envs\\treeline" docs\requirements.md docs\codex_tasks.md docs\TreeLineWarning_requirements_v2.3.md docs\TreeLineWarning_codex_tasks_v2.3.md
```

Result:

```text
docs\TreeLineWarning_requirements_v2.3.md:45:E:\Miniforge3\envs\treeline
docs\TreeLineWarning_requirements_v2.3.md:61:E:\Miniforge3\envs\treeline-ml
docs\codex_tasks.md:67:2. 基础环境：`E:\Miniforge3\envs\treeline`；
docs\TreeLineWarning_codex_tasks_v2.3.md:67:2. 基础环境：`E:\Miniforge3\envs\treeline`；
docs\requirements.md:45:E:\Miniforge3\envs\treeline
docs\requirements.md:61:E:\Miniforge3\envs\treeline-ml
```

### Final Checks

Command:

```powershell
E:\Miniforge3\envs\treeline\python.exe -m pytest
```

Result:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\TreeLineWarning
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2, dash-4.4.0, cov-7.1.0
collected 17 items

tests\contracts\test_pointcloud_contracts.py ...........                 [ 64%]
tests\contracts\test_progress_contracts.py ......                        [100%]

============================= 17 passed in 0.12s ==============================
```

Command:

```powershell
E:\Miniforge3\envs\treeline\python.exe -m ruff check .
```

Result:

```text
All checks passed!
```

Command:

```powershell
E:\Miniforge3\envs\treeline\python.exe -m compileall algorithms core tests
```

Result:

```text
Listing 'algorithms'...
Listing 'algorithms\io'...
Listing 'core'...
Listing 'core\config'...
Listing 'core\contracts'...
Listing 'core\pipeline'...
Listing 'tests'...
Listing 'tests\contracts'...
Listing 'tests\io'...
```

Command:

```powershell
git status --short --branch
```

Result:

```text
## feature/io
 M docs/codex_tasks.md
 M docs/requirements.md
?? algorithms/
?? core/
?? pyproject.toml
?? reviews/T1A-review.md
?? tests/
```

Command:

```powershell
git diff --stat
```

Result:

```text
 docs/codex_tasks.md  | 2 +-
 docs/requirements.md | 4 ++--
 2 files changed, 3 insertions(+), 3 deletions(-)
warning: in the working copy of 'docs/codex_tasks.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/requirements.md', LF will be replaced by CRLF the next time Git touches it
```

Command:

```powershell
git diff --check
```

Result:

```text
warning: in the working copy of 'docs/codex_tasks.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/requirements.md', LF will be replaced by CRLF the next time Git touches it
```

Exit code: 0.

### LAS Confirmation

- Did not read any complete LAS.
- Did not read LAS point arrays.
- Did not modify, move, delete, overwrite, or write anything under `data/raw`.

### Commit Recommendation

Recommend commit only after GPT re-review of this final documentation-path fix.
The known blocker about the wrong base environment path is resolved in the four
approved project documents.
