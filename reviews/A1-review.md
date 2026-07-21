# A1 Review - 预处理基础与算法用点云

## 1. 任务编号和名称

- A1：预处理基础与算法用点云
- 分支：`feature/algorithm`
- 基线：`main` 的 `a8edfa7 merge: integrate feature/io`
- 创建分支命令：`git switch -c feature/algorithm`
- 未 commit，未 push，未创建 worktree

## 2. 原始目标

实现第一批很小的 A1：

- `scene_simple` 三件套一致性验证
- 局部坐标
- 质量检查
- NaN/Inf 删除
- 异常局部坐标检查
- 默认关闭的可配置 SOR 接口
- 两遍流式体素降采样
- 保留 `sample_index` 和 raw LAS `original_index`
- 保留 RGB、intensity、classification、NormalX/NormalY/NormalZ 源属性
- 输出 `outputs/pointcloud/*.npz` 和 `*.json`
- 合成测试和完整 `scene_simple` 验证

未实现导线候选、导线分离、几何特征、HAG、单木、净距、风险、预测或模型。

## 3. 实际完成

新增公共契约：

- `PreprocessingConfig`
- `ProcessingStatistics`
- `DownsampledPointCloud`
- `PreprocessingResult`

新增算法入口：

- `run_basic_preprocessing(...)`

输出：

- `outputs/pointcloud/scene_simple_preprocessed.npz`
- `outputs/pointcloud/scene_simple_preprocessed.json`
- smoke 输出位于 `outputs/pointcloud/smoke/`

这些输出命中 `.gitignore: outputs/`，不会进入 Git。

## 4. 修改文件

- `core/contracts/preprocessing.py`
- `core/contracts/__init__.py`
- `algorithms/preprocessing/basic.py`
- `algorithms/preprocessing/__init__.py`
- `tests/contracts/test_preprocessing_contracts.py`
- `tests/preprocessing/test_basic.py`
- `reviews/A1-review.md`

未修改：

- `data/raw/**`
- `data/samples/**`
- `docs/**`
- `algorithms/io/**`
- `app/**`
- 导线、树木、距离、风险、预测、模型相关模块

## 5. 公共契约字段

`PreprocessingConfig`：

- `source_id`
- `chunk_size`
- `origin_mode`
- `origin_xyz`
- `coordinate_dtype`
- `reject_nan_inf`
- `max_abs_local_coord_m`
- `enable_statistical_outlier_filter`
- `sor_k`
- `sor_std_ratio`
- `sor_max_points`
- `voxel_size_m`
- `voxel_representative`
- `max_input_points`
- `max_output_points`
- `record_memory`

`ProcessingStatistics`：

- 输入、处理、有效、过滤、降采样点数
- 降采样比例
- chunk size、voxel size
- 耗时
- RSS start/peak/end
- 估算输入/输出数组字节数
- `sor_skipped_reason`
- warnings

`DownsampledPointCloud`：

- `xyz_local`，shape `(M, 3)`，单位 m
- `origin_xyz`，shape `(3,)`，float64，单位 m
- `original_index`，shape `(M,)`，raw LAS 全局索引
- `sample_index`，shape `(M,)`，`scene_simple.las` 行索引
- `voxel_key`，shape `(M, 3)`，floor 体素键
- `voxel_point_count`，shape `(M,)`
- `attributes`，每个数组首维必须等于 `M`

`PreprocessingResult`：

- 输入路径、index map 路径、manifest 路径
- 输出 NPZ/JSON 路径
- config、statistics
- `manifest_source_id`
- `origin_xyz`
- arrays metadata
- warnings、created_at

大型数组只写入 NPZ，不嵌入 JSON。

## 6. 两遍体素算法说明

第一遍：

- 通过 `iter_las_chunks()` 分块读取
- 将 source 坐标转为局部坐标
- 去除 NaN/Inf 和异常局部坐标
- 按 `floor(local_xyz / voxel_size_m)` 计算 voxel key
- 对每个 voxel 累计点数和坐标和
- 得到精确 voxel 质心

第二遍：

- 再次通过 `iter_las_chunks()` 分块读取
- 重新执行同样的质量过滤和 voxel key 计算
- 对每个 voxel 选择距离精确质心最近的真实源点
- tie-break 使用更小 `sample_index`，保证结果可重复
- 输出该真实点的 `xyz_local`、`sample_index`、raw `original_index` 和源属性

没有保存每个原始点的 Python 容器；只保存每个 voxel 的累计信息和最终候选代表点。

## 7. SOR 默认关闭说明

默认：

- `enable_statistical_outlier_filter=False`
- `sor_skipped_reason="disabled_by_default_to_preserve_thin_wires"`

原因：

- 避免误删细导线点
- 避免在约 200 万点上直接建立大型近邻结构

如果显式启用 SOR：

- 仅允许 `effective_total <= sor_max_points`
- 使用当前环境已有 `scipy.spatial.cKDTree`
- 不安装依赖
- 超过点数或缺少 scipy 时明确失败，不静默跳过

## 8. 合成测试结果

命令：

```text
E:\Miniforge\envs\treeline\python.exe -m pytest tests/contracts/test_preprocessing_contracts.py tests/preprocessing/test_basic.py
```

结果：

```text
15 passed
```

覆盖：

- header_min 局部坐标
- explicit origin
- NaN、Inf 删除
- 异常坐标删除
- 两遍体素算法
- nearest_centroid 代表点
- voxel key 的 floor 规则
- sample_index 与 original_index 追溯
- 属性同步保留
- uint16 RGB 保留
- 空输入明确失败
- max_output_points 超限明确失败
- 输出默认不覆盖
- 取消操作
- NPZ、JSON 可重新加载
- 禁止 full read
- 输入 `data/samples` 风格文件运行前后不变

## 9. 完整测试和质量检查

```text
E:\Miniforge\envs\treeline\python.exe -m pytest
```

结果：

```text
92 passed
```

```text
E:\Miniforge\envs\treeline\python.exe -m ruff check .
```

结果：

```text
All checks passed!
```

```text
E:\Miniforge\envs\treeline\python.exe -m compileall algorithms core tests scripts
```

结果：通过。

```text
git diff --check
```

结果：

```text
warning: in the working copy of 'algorithms/preprocessing/__init__.py', LF will be replaced by CRLF the next time Git touches it
```

无 whitespace error。

```text
E:\Miniforge\envs\treeline\python.exe -m ruff format --check core/contracts/preprocessing.py core/contracts/__init__.py algorithms/preprocessing/basic.py algorithms/preprocessing/__init__.py tests/contracts/test_preprocessing_contracts.py tests/preprocessing/test_basic.py
```

结果：

```text
6 files already formatted
```

用户指定的完整格式检查：

```text
E:\Miniforge\envs\treeline\python.exe -m ruff format --check .
```

结果：失败，但仅涉及本次禁止修改的既有文件：

```text
Would reformat: algorithms\io\samples.py
Would reformat: algorithms\preprocessing\preview.py
Would reformat: core\contracts\pointcloud.py
3 files would be reformatted, 33 files already formatted
```

本次未修改这些文件。

## 10. scene_simple 真实验证

Smoke：

- 参数：`max_input_points=300000`
- 输入总点数：`2,008,464`
- 实际处理点数：`300,000`
- 输出点数：`18,871`
- 降采样比例：`0.06290333333333334`
- 耗时：`2.5261489000004076 s`
- 峰值 RSS：`178,700,288 bytes`

完整运行：

- 输入点数：`2,008,464`
- 实际处理点数：`2,008,464`
- 默认 SOR：关闭
- voxel size：`0.05 m`
- 输出点数：`146,126`
- 降采样比例：`0.07275510041504353`
- 耗时：`11.23110910000014 s`
- RSS start：`55,025,664 bytes`
- RSS peak：`446,177,280 bytes`
- RSS end：`418,717,696 bytes`

验收：

- 输出点数大于 0 且小于 `1,000,000`
- 完整运行小于 180 秒
- 峰值 RSS 小于 1.5GB
- 输出 NPZ、JSON 可重新加载
- `xyz_local` 全部有限

NPZ/JSON 重载验证：

- JSON 点数：`146,126`
- NPZ 点数：`146,126`
- `xyz_local` finite：`True`
- `original_index` 范围：`3..6888864`
- `original_index` 唯一：`True`
- `sample_index` 范围：`3..2008463`
- `sample_index` 唯一：`True`
- 属性：`NormalX`、`NormalY`、`NormalZ`、`red`、`green`、`blue`、`rgb`、`intensity`、`classification`
- `rgb` dtype：`uint16`

## 11. 数据文件运行前后对比

运行后：

```text
E:\TreeLineWarning\data\samples\scene_simple.las           120508845 2026/7/20 22:06:30
E:\TreeLineWarning\data\samples\scene_simple_index_map.npz   3060946 2026/7/20 22:06:31
E:\TreeLineWarning\data\samples\scene_simple_manifest.json      5735 2026/7/20 22:06:34
```

与 IMPLEMENT 前只读检查一致。未修改 `data/samples` 三件套。

## 12. 是否调用 full read

实现只调用：

- `iter_las_chunks()`
- `read_las_header_summary()`
- `validate_scene_sample_artifacts()`

测试中 monkeypatch：

- `laspy.read`
- `laspy.lasreader.LasReader.read`

若 full read 被调用会失败；完整测试通过。

未读取 `scan_large.las` 点记录。

## 13. Git 状态

```text
## feature/algorithm
 M algorithms/preprocessing/__init__.py
 M core/contracts/__init__.py
?? algorithms/preprocessing/basic.py
?? core/contracts/preprocessing.py
?? tests/contracts/test_preprocessing_contracts.py
?? tests/preprocessing/test_basic.py
?? reviews/A1-review.md
```

`outputs/` 产物被 Git 忽略：

```text
.gitignore:4:outputs/ outputs\pointcloud\scene_simple_preprocessed.npz
.gitignore:4:outputs/ outputs\pointcloud\scene_simple_preprocessed.json
.gitignore:4:outputs/ outputs\pointcloud\smoke\scene_simple_preprocessed.npz
.gitignore:4:outputs/ outputs\pointcloud\smoke\scene_simple_preprocessed.json
```

## 14. 已知限制和风险

- A1 未实现 A2 几何特征，现有 NormalX/Y/Z 仅作为源属性保留。
- SOR 默认关闭；显式启用时只支持小规模受控输入。
- 体素 accumulator 使用每 voxel 字典，适合 `scene_simple` 和 A1 验证；更大点云后续需要继续做分块/分区策略。
- `ruff format --check .` 暴露 3 个既有文件格式问题，但本次范围禁止修改它们。
- 没有生成正式 Streamlit 预览界面，A1 只输出算法用 NPZ/JSON。

## 15. 建议 GPT 重点审查

- `algorithms/preprocessing/basic.py` 的两遍体素逻辑
- `original_index` 和 `sample_index` 追溯链
- 输出路径限制和 overwrite 行为
- SOR 默认关闭及显式启用时的失败/保护策略
- `core/contracts/preprocessing.py` 的数组 shape 校验和 JSON 序列化边界

## 16. 是否建议提交

建议在 GPT 审查通过后提交 A1。

建议 commit message：

```text
feat: add basic preprocessing point cloud pipeline
```
