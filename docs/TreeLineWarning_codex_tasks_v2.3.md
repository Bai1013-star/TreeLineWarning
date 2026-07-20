# TreeLineWarning——最终总控与 Codex 多会话分工文档

> 版本：v2.3（八月底交付版）  
> 更新日期：2026-07-20  
> 项目根目录：`E:\TreeLineWarning`  
> Worktree 根目录：`E:\TreeLineWarning-worktrees`

## 1. 协作主体

用户和队友统一视为一个“项目组”，本文件不分配项目组内部个人任务。用户自行决定由谁准备数据、检查结果或整理文档。

协作只分三方：

### 项目组

负责：

- 准备和整理真实点云；
- 确认场景中的导线、树木、建筑和重影情况；
- 决定标注规则、风险阈值和生长率；
- 控制八月底版本范围；
- 人工检查导线数、单木数、距离和预测是否合理；
- 将 Codex 的 PLAN 和 Review 发给 GPT；
- 审查通过后决定 COMMIT、PUSH 和合并；
- 自行安排项目组内部具体工作。

### 总控 Codex 会话

负责：

- 阅读需求和分工文档；
- 检查仓库、环境、数据和 Git 状态；
- 维护公共数据结构和模块接口；
- 把大任务拆成小任务；
- 为子会话生成完整提示词；
- 明确依赖、并行关系、分支和 Worktree；
- 检查子模块接口是否一致；
- 组织集成、模块级 Review 和最终验收；
- 不代替子会话一次性实现全部系统。

### 子 Codex 会话

负责：

- 在批准范围内实现指定模块；
- 编写测试；
- 执行静态检查；
- 输出 Review 包；
- 按审查意见修正；
- 未经授权不得 commit 或 push。

### GPT 审查

负责：

- 审查总控 PLAN；
- 审查子任务范围和接口；
- 审查 Review 包和关键 diff；
- 判断能否 COMMIT 或是否需要返工；
- 帮助项目组控制技术风险和工期。

## 2. 当前项目事实

总控必须以以下事实为准：

1. 项目根目录：`E:\TreeLineWarning`；
2. 基础环境：`E:\Miniforge3\envs\treeline`；
3. 模型环境尚未作为第一步创建；
4. 当前有两份完全不同的 LAS 原始点云；
5. 一份约 300 多 MB；
6. 一份约 2.19GB；
7. 两份均位于 `data\raw` 的不同子目录；
8. `data\samples` 初始为空；
9. 原始 LAS 只读，不得覆盖、删除、移动或原地修改；
10. 第一阶段优先处理较小 LAS；
11. 大 LAS 禁止整体载入内存或 Open3D；
12. 八月底前的正式轻量语义模型优先为 Open3D-ML RandLA-Net；
13. 单木优先为 pointtree 的 TreeXAlgorithm；
14. 多根导线由几何切片、跟踪和 B 样条完成；
15. 第一版预测采用三档线性情景，不强制 LSTM；
16. 项目组内部具体分工由用户决定。

## 3. 四种强制指令

### PLAN ONLY

允许：

- 阅读文件；
- 检查目录和 Git 状态；
- 检查已安装依赖版本；
- 分析现有代码；
- 输出计划、接口、测试和提示词。

禁止：

- 修改、创建或删除项目文件；
- 安装、升级或卸载依赖；
- 修改 Conda、CUDA、Docker 或系统设置；
- commit、push、创建或合并 PR。

### IMPLEMENT

仅按 GPT 已批准计划实施：

- 只能修改批准范围内的文件；
- 必须编写或更新测试；
- 必须运行检查；
- 必须输出 Review 包；
- 不 commit、不 push。

### COMMIT

仅在 Review 通过后：

- 检查暂存内容；
- 一个 commit 一个目的；
- 不混入无关文件；
- 不提交点云、模型、缓存、环境和输出。

### PUSH

仅在模块级 Review 通过后：

- Push 当前功能分支；
- 创建 PR；
- 不直接合并 `main`。

## 4. 强制流程

```text
项目组/GPT给出任务
→ Codex执行PLAN ONLY
→ 项目组把PLAN发给GPT
→ GPT批准或修改
→ Codex执行IMPLEMENT
→ Codex运行测试并输出Review包
→ 项目组把Review包发给GPT
→ GPT审查
→ 项目组发送COMMIT
→ 模块完成后进行模块级Review
→ 项目组发送PUSH
→ Codex创建PR
→ 人工决定是否合并
```

Codex 无法自动把 Review 包发送到当前 GPT 对话，必须保存到 `reviews` 并由项目组复制或上传。

## 5. 总控会话的权限和边界

总控可以修改：

- `docs/`；
- `core/contracts/`；
- `core/config/`；
- `core/pipeline/`；
- 经批准的集成测试和公共配置。

总控不得：

- 在一个任务中实现所有算法；
- 随意修改子模块内部实现；
- 同时创建大量相互覆盖的文件；
- 直接下载模型或权重；
- 未经审查建立复杂 CUDA、WSL 或 Docker 环境；
- 绕过 PLAN/Review 流程。

总控首要工作是定义稳定公共契约，包括：

- `PointCloudSource`；
- `LasHeaderSummary`；
- `PointCloudChunk`；
- `CropBounds`；
- `CropResult`；
- `QualityReport`；
- `ClassificationResult`；
- `WireInstance`；
- `WireResult`；
- `TreeInstance`；
- `TreeResult`；
- `ClearanceRecord`；
- `RiskThresholds`；
- `GrowthScenario`；
- `PredictionResult`；
- `ProjectRecord`；
- `PipelineProgress`；
- 统一错误类型和取消/进度接口。

## 6. 子会话划分

### 会话 1：项目骨架、公共契约与工程管理

负责：

- `pyproject.toml`；
- `.gitignore`；
- `core/contracts/`；
- `core/config/`；
- `core/database/`；
- 基础工程创建、保存和重新打开；
- 最小测试框架。

首个任务应小而明确，不同时实现算法和完整界面。

### 会话 2：LAS 文件头、分块读取和裁剪

负责：

- `algorithms/io/`；
- `algorithms/preprocessing/crop*`；
- 对应测试和脚本。

任务：

1. 扫描 `data/raw` 下 LAS 文件；
2. 使用只读方式读取文件头；
3. 输出点数、版本、Point Format、XYZ、Scale、Offset 和维度；
4. 估算内存；
5. 使用分块迭代；
6. 按 XYZ 范围裁剪；
7. 保留点格式和必要字段；
8. 保存到 `data/samples`；
9. 禁止覆盖原始文件；
10. 为 2.19GB 文件提供只读、分块和取消机制。

### 会话 3：点云预处理、地面与建筑

负责：

- `algorithms/preprocessing/`；
- `algorithms/ground_building/`。

任务：

- 局部坐标；
- 去噪；
- 体素降采样；
- 分块和索引映射；
- 法向量与几何特征；
- SMRF 适配器；
- 无 PDAL 降级方案；
- 建筑大平面基线。

### 会话 4：语义分类和模型适配

负责：

- `algorithms/semantic/`；
- `models/adapters/`；
- `models/randlanet/`；
- 数据转换和模型测试。

分阶段：

#### 4A 规则/人工接口

- 六类标签；
- `SemanticSegmenter` 接口；
- 规则或人工标签导入；
- 无权重回退。

#### 4B RandLA-Net 快速推理验证

- 只在计划批准后创建 `treeline-ml`；
- 检查当前官方兼容环境；
- 跑通小样本推理；
- 记录权重来源和原始类别；
- 不将公开类别直接伪装成项目六类。

#### 4C 自有数据微调

- 六类数据转换；
- 分块训练；
- `batch_size=1`、AMP；
- 类别加权或 Focal Loss；
- 重叠块融合；
- Precision、Recall、F1、IoU；
- 单独报告电力线召回率。

PointNeXt-S 仅作为后续对照，不在八月底前并行建设第二套正式链路。

### 会话 5：多导线实例与曲线重建

负责：

- `algorithms/wires/`。

任务：

- 线性特征过滤；
- PCA 方向；
- 局部坐标；
- 沿线切片；
- 截面聚类；
- 跨切片轨迹关联；
- 断点容错；
- `wire_id`；
- 三次 B 样条；
- 拟合误差、覆盖率和置信度。

### 会话 6：单木实例分割

负责：

- `algorithms/trees/`。

任务：

- 定义 `TreeInstanceSegmenter`；
- 接入 pointtree 的 TreeXAlgorithm；
- 验证输入为地面和植被；
- 输出 `tree_id`、树干位置、树高和树冠范围；
- 处理未分配植被；
- TreeX 不可用时回退到树干种子＋区域生长；
- 提供拆分、合并和重新分配接口。

### 会话 7：距离、风险和预测

负责：

- `algorithms/clearance/`；
- `algorithms/prediction/`。

任务：

- 导线高密度采样；
- 包围盒粗筛；
- KD-Tree 最近点；
- 三维、水平和垂直净距；
- 最近点对；
- 风险阈值；
- 灰色“需复核”；
- 三档生长率线性预测；
- 关注、预警、危险日期；
- 置信度和复测日期。

### 会话 8：Streamlit、报告与人工修正

负责：

- `app/`；
- `reports/`；
- 已批准的工程接口集成。

任务：

- 工程创建和打开；
- LAS 文件头和裁剪页面；
- 点云显示；
- 图层与参数；
- 导线、树木、距离、风险和预测；
- 人工拆分/合并/重新分配入口；
- Excel 和 Word；
- 进度、取消和错误提示。

页面不得直接实现算法。

### 会话 9：测试、性能与交付

负责：

- `tests/`；
- `scripts/`；
- 环境检查；
- 端到端测试；
- 文档和最终验收。

任务：

- 合成三根弧线和多棵树场景；
- 空输入、损坏文件、缺失字段；
- 大 LAS 不整体加载测试；
- 模型无权重回退；
- TreeX 不可用回退；
- 内存、显存和耗时记录；
- 报告字段检查；
- Windows 启动说明；
- 最终演示清单。

## 7. 推荐依赖顺序

```text
总控公共契约
→ 项目骨架
→ LAS文件头和裁剪
→ 预处理
→ 人工/规则标签接口
→ 导线、单木、距离和预测并行
→ Streamlit和报告集成
→ TreeX真实数据适配
→ RandLA-Net独立模型环境
→ 真实数据调参
→ 冻结、测试和演示
```

第一阶段严禁因为模型环境尚未完成而阻塞几何闭环。

## 8. 分支和 Worktree

推荐分支：

```text
main
feature/project-core
feature/las-io-crop
feature/pointcloud-preprocessing
feature/semantic-randlanet
feature/wire-reconstruction
feature/tree-segmentation
feature/clearance-prediction
feature/streamlit-reports
feature/testing-integration
```

推荐 Worktree：

```text
E:\TreeLineWarning-worktrees\core
E:\TreeLineWarning-worktrees\las-io
E:\TreeLineWarning-worktrees\preprocessing
E:\TreeLineWarning-worktrees\semantic
E:\TreeLineWarning-worktrees\wires
E:\TreeLineWarning-worktrees\trees
E:\TreeLineWarning-worktrees\clearance
E:\TreeLineWarning-worktrees\ui
E:\TreeLineWarning-worktrees\testing
```

一个 Worktree 同一时间只交给一个 Codex 会话。多个会话不得直接在同一工作目录写文件。

## 9. 每个任务开始前检查

Codex 必须执行并报告：

```bash
git status --short
git branch --show-current
git diff --stat
python --version
```

涉及环境时只检查，不安装：

```bash
python -c "import <package>; print(<package>.__version__)"
```

必须确认：

- 当前目录和分支；
- 工作区是否干净；
- 是否与其他会话冲突；
- 使用的是 `treeline` 还是 `treeline-ml`；
- 真实 LAS 文件是否只读；
- 任务是否会整体加载大 LAS；
- 是否需要项目组提供坐标范围、标签或阈值。

## 10. IMPLEMENT 后检查

至少运行：

```bash
pytest
ruff check .
ruff format --check .
git status --short
git diff --stat
git diff
```

启用类型检查时：

```bash
mypy <相关目录>
```

LAS/点云任务还必须报告：

- 输入文件大小；
- 是否只读；
- 点数；
- 分块大小；
- 峰值内存；
- 输出路径；
- 是否保持 Point Format、Scale、Offset 和维度；
- 是否存在完整点云重复复制。

模型任务还必须报告：

- 环境；
- PyTorch 和 CUDA；
- 输入块大小；
- 峰值显存；
- 推理时间；
- 权重来源；
- 原始类别与项目类别映射；
- 各类别指标；
- 是否发生回退。

## 11. Review 包

保存到：

```text
reviews\<TASK-ID>-review.md
```

必须包含：

1. 任务编号和名称；
2. 原始目标；
3. 批准计划；
4. 实际完成；
5. 未完成；
6. 修改文件；
7. 公共接口变化；
8. 关键设计决策；
9. 测试命令和结果；
10. Ruff、格式和类型检查；
11. 性能、内存和显存；
12. `git status --short`；
13. `git diff --stat`；
14. 关键代码位置；
15. 真实点云检查结果；
16. 已知限制和风险；
17. 建议 GPT 重点审查位置；
18. 是否建议 commit。

## 12. 八月底工期控制

- 第 1～4 天：骨架、公共契约、LAS 文件头和裁剪；
- 第 5～14 天：人工/规则闭环、多导线、单木降级、距离和预测；
- 第 15～24 天：TreeX、完整界面、工程和报告；
- 第 25～32 天：RandLA-Net 小样本推理和微调；
- 第 33～36 天：三个场景优化和集成；
- 最后 5～7 天：停止大型新功能，修复、测试、文档和演示。

总控必须主动阻止以下情况：

- 同时维护两套正式语义模型；
- 为追求模型精度而推迟完整闭环；
- 在 8 月 24 日后新增大型功能；
- 对 2.19GB LAS 进行整体内存加载；
- 未经 Review 大批量生成或重写代码。
