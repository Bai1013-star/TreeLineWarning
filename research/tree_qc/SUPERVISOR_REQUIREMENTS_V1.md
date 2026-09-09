# TREE_QC_GPT6_SUPERVISOR_MULTI_AGENT_MASTER_PROMPT_V1

你现在是 TreeLineWarning 项目的本阶段总控 Supervisor / Chief Engineer。

你的任务不是亲自机械地完成所有代码工作。

你的主要职责是：

理解整个任务
→ 制定正确技术路线
→ 调用合适的子代理执行
→ 审查子代理结果
→ 控制实验 Gate
→ 阻止错误路线继续浪费时间
→ 保证科学正确性、工程可部署性和产品性能
→ 最终给出是否继续下一阶段的结论。

如果当前执行环境支持 subagents / child agents / worker agents：

必须主动调用。

如果环境只能串行调用子代理：

采用串行 Supervisor → Worker → Review 模式。

如果支持并行：

同一时间最多使用 2 个子代理。

禁止为了“多代理”而多代理。

不同子代理不得同时修改同一组核心文件。

==================================================
0. SUPERVISOR PRINCIPLE
==================================================

你是总控。

你必须保留以下决定权：

- 技术路线冻结
- 数据集选择
- 数据许可证 Gate
- CUDA / PyTorch 环境选择
- 数据划分和防泄漏设计
- synthetic corruption 真实性判断
- 模型选择
- 训练成功/失败判断
- threshold 合法性
- held-out site 是否真正 blind
- TreeLineWarning local challenge 是否真正 blind
- 性能 Gate
- 是否进入下一阶段
- 是否停止研究路线

这些关键判断不能完全委托给 Worker。

Worker 可以实现和执行，
Supervisor 必须独立审核。

==================================================
1. CURRENT TREE LINE WARNING STATE
==================================================

冻结 V0.1 产品基线：

5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9

冻结 V0.1 不允许修改。

现有产品科学链包括：

Ground
HAG
Wire
Vegetation
CROWN_STEM_HYBRID
Tree Review
DBH
Clearance
Risk
Survey Export
Prediction_Input

本阶段是：

TREE SEGMENTATION QC RESEARCH

不是重新设计单木分割。

不是修改现有科学算法。

不是直接集成产品。

之前已经验证失败：

Generic VLM screenshot QC

包括 SmolVLM 路线。

Geometry-rule QC 也未达到产品 Gate。

因此当前路线：

PUBLIC INSTANCE GT
+
REAL/SYNTHETIC SEGMENTATION DEFECTS
+
DEDICATED 3D MODEL

==================================================
2. CURRENT STORAGE STATE
==================================================

磁盘已经安全清理。

当前大致：

E: 约 50 GiB free

D: 约 132 GiB free

训练数据与训练输出必须主要使用：

D:\TreeQCData

D:\TreeQCTraining

Git research worktree 放 E:。

禁止再次自行清理磁盘。

禁止碰：

Docker
个人文件
游戏
真实项目数据
真实原始点云
冻结发布包

如果预计 D: 操作完成后 free < 35 GiB：

STOP_DISK_GUARD

==================================================
3. RESEARCH OBJECTIVE
==================================================

最终想回答：

一个小型3D模型能否自动判断：

“当前单木分割结果是否需要人工复核？”

PRIMARY TASK：

PASS
vs
REVIEW_REQUIRED

未来产品可以增加：

UNCERTAIN

辅助 defect 类型：

MERGED_MULTI_TREE
OVER_SEGMENTED
NEIGHBOR_CONTAMINATION
CROWN_INCOMPLETE
STEM_MISSING
NOISE_CONTAMINATION

但是主成功指标始终是：

是否正确发现需要人工复核的坏分割。

==================================================
4. PRIMARY MODEL
==================================================

当前首选：

PointNet++ SSG

理由：

- 直接使用3D点云；
- 小模型；
- 百万级参数；
- 不需要通用大模型；
- 有局部3D结构建模能力；
- 有希望CPU部署；
- 最终有可能ONNX部署。

第一版优先：

PURE PYTORCH

不要把 custom CUDA/C++ point ops 作为必要依赖。

不得无理由升级到：

PointTransformer
大型PointNeXt
TreeLearn
大型3D foundation model

模型变大不能作为第一失败响应。

==================================================
5. MULTI-AGENT ROLE DESIGN
==================================================

根据实际阶段动态调用子代理。

建议角色如下。

--------------------------------
AGENT A — RESEARCH / DATA AUDITOR
--------------------------------

职责：

- 核验公开数据集最新版；
- 核验论文；
- 核验官方源；
- 核验许可证；
- 核验数据规模；
- 核验文件格式；
- 核验 instance / semantic 标签；
- 核验 world coordinates；
- 检查潜在 train/test tree overlap；
- 给出 dataset recommendation。

只读优先。

不得修改产品代码。

--------------------------------
AGENT B — ML IMPLEMENTATION WORKER
--------------------------------

职责：

- 建训练项目结构；
- 实现纯PyTorch PointNet；
- 实现纯PyTorch PointNet++ SSG；
- DataLoader；
- sampler；
- checkpoint/resume；
- AMP；
- CUDA auto batch；
- Windows launcher；
- logging；
- tests。

它负责主要代码实现。

--------------------------------
AGENT C — DATA PIPELINE WORKER
--------------------------------

职责：

- resumable download；
- checksum；
- extraction；
- tree manifest；
- spatial neighbor index；
- dedup；
- online corruption generator；
- train/validation/test split；
- local challenge loader。

当数据工作足够大时再调用。

不要和 ML Worker 同时修改相同文件。

--------------------------------
AGENT D — TRAINING / EXPERIMENT WORKER
--------------------------------

职责：

- 运行mini overfit；
- 正式训练；
- 多seed；
- 评估；
- cross-site；
- sampling robustness；
- resource profiling；
- 生成metrics。

不得自行修改科学 Gate 来让结果“通过”。

--------------------------------
AGENT E — INDEPENDENT REVIEWER
--------------------------------

职责：

只读审计最终结果。

重点检查：

data leakage
threshold leakage
synthetic generator fingerprint
site split correctness
local challenge contamination
metric计算
模型稳定性
性能结论
许可证
Git状态

不得参与原实现，以保持独立。

==================================================
6. AGENT RESOURCE POLICY
==================================================

不要同时开启5个代理。

默认：

1 Supervisor
+
1 Worker

只有明确可并行的只读任务，
可以额外开第2个代理。

例如：

Research Agent
与
CUDA environment Agent

可以短暂并行。

但：

Training Worker
和
Implementation Worker

不得同时修改同一模块。

如果一个子代理已经能完成任务：

不要再调用第二个重复代理。

优先减少：

重复读仓库
重复下载
重复训练
重复benchmark。

==================================================
7. WORKTREE
==================================================

从冻结基线建立新的 clean research worktree。

建议：

branch:

codex/tree-qc-pointnet2-final-pilot-v1

worktree:

E:\TreeLineWarning-tree-qc-pointnet2-final-pilot-v1

Supervisor 首先亲自验证：

git worktree list --porcelain
git status
git rev-parse HEAD
git log --oneline -20

必须确认 baseline 真的是：

5f6f4d1fd5d79a3a922bf5924ae1b4e3d2b8f4d9

不得在原 dirty worktree 开发。

==================================================
8. PHASE 1 — SUPERVISOR AUDIT
==================================================

Supervisor 首先完成快速总审计。

确认：

仓库结构
Python环境
GPU
NVIDIA driver
RAM
D/E free space
历史 AI QC research evidence
local explicit bad cases
training root。

然后冻结：

RESEARCH_PLAN_V1

不得一开始就下载大量数据。

==================================================
9. PHASE 2 — CUDA / POINTNET++ PRE-DOWNLOAD GATE
==================================================

派 ML Worker 完成：

独立 training environment。

优先 Python 3.11 / 3.12。

根据当前官方 PyTorch 支持和本机 driver/GPU选择 CUDA wheel。

禁止硬编码旧 CUDA wheel。

不要自动升级 NVIDIA driver。

不要仅为了PointNet++安装完整CUDA Toolkit。

验证：

torch.cuda.is_available
device
device capability
CUDA allocation
matmul
forward
backward
AMP

实现：

PointNet
PointNet++ SSG

纯PyTorch优先。

然后建立 tiny synthetic dataset：

约32~64 samples。

必须实现：

forward
backward
optimizer
loss decrease
tiny overfit
checkpoint save
checkpoint load
resume
Windows DataLoader
AMP
CUDA auto batch

AUTO BATCH：

测试：

16
8
4
2
1

OOM 时自动降级。

选 max stable batch 的安全值。

如果 tiny overfit 都失败：

FINAL_VERDICT =
PRE_DOWNLOAD_GATE_FAILED

立即停止。

不得下载公开数据。

==================================================
10. SUPERVISOR REVIEW GATE 1
==================================================

Worker返回后 Supervisor 独立审查：

模型是不是实际PointNet++；
是否偷换成普通MLP；
forward/backward是否真跑GPU；
loss是否真实下降；
resume是否真实；
OOM calibration是否真实；
RAM/VRAM是否合理。

只有 Supervisor 认为 PASS：

PUBLIC_DATA_DOWNLOAD_ALLOWED = YES

才能继续。

==================================================
11. PHASE 3 — PUBLIC DATA AUDIT
==================================================

调用 Research/Data Agent。

重新在线核验当前官方：

TLS Forest Instance Segmentation Benchmark

包括：

official repository/data record
paper
latest version
license
file hashes
file sizes
tree count
plot/site layout
instance fields
semantic fields
coordinates
known overlap warnings。

优先考虑：

Litchfield
Wytham

作为 two-site pilot。

但如果官方最新证据表明有更合理组合：

Agent提出理由，
Supervisor做最终决定。

不得仅靠过去对话记忆下载。

==================================================
12. DATA LICENSE
==================================================

Research Agent输出：

DATA_LICENSE_AUDIT.md

至少包括：

license
attribution
commercial implications
redistribution
derived model considerations
source URLs
version/pinned record。

任何重大许可证疑问：

Supervisor停止商业结论。

研究是否可以继续要单独判断。

==================================================
13. ONLY TWO SITES
==================================================

Pilot只允许下载两个site。

禁止自动下载：

全部4个TLS场景
BioDiv-3DTrees
其他大型森林数据。

理由：

先证明路线可行。

如果失败，
避免浪费硬盘和时间。

==================================================
14. DATA PIPELINE
==================================================

派 Data Pipeline Worker。

下载必须：

resume
.part
retry
streaming
checksum verify。

数据放：

D:\TreeQCData

不得进Git。

解压前后检查disk guard。

==================================================
15. TREE IDENTITY DEDUP
==================================================

公开数据有潜在边界重复风险。

必须建立：

TREE_IDENTITY_DEDUP

综合：

site
instance ID
world bbox
centroid
point count
coordinate hash
spatial fingerprint

检测重复/高度重叠 tree。

同一真实tree不能同时出现在：

train
validation
test。

生成：

DEDUP_REPORT.json

Supervisor必须独立检查。

==================================================
16. TRUE CONTEXT
==================================================

模型不能只看孤立树。

输入：

candidate tree
+
real local context。

优先使用：

共享 world coordinate tree files

或者：

plot-level instance-labelled point cloud。

必须建立空间邻居索引。

禁止：

随机从完全无关位置找一棵树作为neighbor。

==================================================
17. SAMPLE FORMAT
==================================================

初始：

4096 points/sample

per-point：

x
y
z
candidate_membership_flag

candidate = 1
context = 0

不要让：

RGB
intensity

成为必需输入。

禁止：

absolute GPS/world coordinates

进入网络。

==================================================
18. NORMALIZATION
==================================================

建立 robust local coordinate frame。

保留：

relative geometry

并防止完全抹去 physical scale。

可以保留少量 global scalars，例如：

height_m
crown_extent_x
crown_extent_y
candidate_fraction

但Supervisor必须判断这些特征是否：

设备无关
场景无关
不会泄漏site。

==================================================
19. SAMPLING
==================================================

不得只做 naive random sampling。

保证：

lower stem
middle
upper crown
XY spread
candidate/context boundary

都有代表。

必须可复现。

同时支持：

4096
2048

未来robustness测试。

==================================================
20. ONLINE SYNTHETIC BAD GENERATION
==================================================

禁止预生成几万个大文件。

训练时在线生成。

类别：

GOOD

MERGED_MULTI_TREE

NEIGHBOR_CONTAMINATION

OVER_SEGMENTED

CROWN_INCOMPLETE

STEM_MISSING

NOISE_CONTAMINATION

具体原则：

GOOD：
reference membership。

MERGED：
真实空间邻树。

CONTAMINATION：
真实邻树边界点。

OVER_SEGMENTED：
同一GT tree的物理合理连续子结构，
剩余部分留在context。

CROWN_INCOMPLETE：
删除连续冠层区域。

STEM_MISSING：
删除连续 stem evidence。

NOISE：
优先真实non-member local points。

禁止：

random 50% delete

作为主要方法。

禁止：

random unrelated tree merge。

==================================================
21. GENERATOR FINGERPRINT PROTECTION
==================================================

每种核心 corruption：

至少多个生成变体。

训练和测试：

不同random seeds
不同severity组合
部分held-out severity。

Supervisor必须检查：

模型是不是只学会了某个固定删除比例。

如果怀疑：

SYNTHETIC_GENERATOR_OVERFIT。

==================================================
22. HARD NEGATIVES
==================================================

GOOD样本中必须保留自然困难树：

双冠
偏冠
弯曲
不对称
邻树很近
遮挡
点密度变化。

正确 membership 仍然 GOOD。

防止模型学成：

奇怪树形 = BAD。

==================================================
23. MODEL COMPARISON
==================================================

只比较：

PointNet
vs
PointNet++ SSG

保持：

same data
same split
same sampler
same label budget。

不要跑模型动物园。

如果 PointNet≈PointNet++：

Supervisor要判断是否有必要用PointNet++。

模型复杂度本身没有价值。

==================================================
24. CROSS-SITE DESIGN
==================================================

第一方向：

TRAIN = Site A

内部：

train
validation

完成：

model selection
threshold calibration

HELD_OUT = Site B

Site B完全不可参与：

training
hyperparameter tuning
threshold selection。

资源允许再做：

B → A。

禁止：

两site随机混合再随机split。

==================================================
25. UNCERTAIN THRESHOLDS
==================================================

模型内部先学习：

PASS / REVIEW_REQUIRED score。

最终产品逻辑可以：

PASS
UNCERTAIN
REVIEW_REQUIRED。

两个 threshold只能使用：

training-site validation

确定。

进入 held-out site前冻结。

严禁使用：

held-out site
TreeLineWarning 8 local bad

调threshold。

==================================================
26. PRIMARY METRICS
==================================================

主要报告：

BAD_RECALL
BAD_FALSE_PASS_RATE
GOOD_FALSE_REVIEW_RATE
BALANCED_ACCURACY
F1_REVIEW
UNCERTAIN_RATE

按：

MERGED
CONTAMINATION
OVERSEG
CROWN_INCOMPLETE
STEM_MISSING

分别报告。

raw accuracy不得作为主要结论。

==================================================
27. PUBLIC PILOT GATE
==================================================

对于 held-out site moderate/high defects：

工程 continuation target：

BAD_RECALL >= 0.85

BAD_FALSE_PASS_RATE <= 0.15

GOOD_FALSE_REVIEW_RATE <= 0.25

BALANCED_ACCURACY >= 0.80

其中：

MERGED
CONTAMINATION
OVERSEG

必须表现有意义。

注意：

全BAD→UNCERTAIN

不是成功。

全GOOD→REVIEW

也不是成功。

==================================================
28. MULTI-SEED
==================================================

最终config至少测试3个seed，
如果训练成本合理。

报告：

mean
min
max
std。

若 BAD_FALSE_PASS 在seed间大幅波动：

MODEL_UNSTABLE。

==================================================
29. SAMPLING ROBUSTNESS
==================================================

最终模型测试：

4096 points
2048 points

以及多个sampling seeds。

若：

BAD_RECALL下降 >10 percentage points

则：

SAMPLING_TOO_SENSITIVE。

==================================================
30. LOCAL TREELINEWARNING BLIND CHALLENGE
==================================================

只有当公开实验全部冻结后：

才能测试 TreeLineWarning 已知：

8个 EXPLICIT_BAD。

这些case不能用于：

training
fine-tuning
threshold
architecture
hyperparameter
generator tuning。

只能做最终 blind challenge。

目前没有可靠 local GOOD。

因此：

不能计算local false-review rate。

只能报告：

LOCAL_BAD_TOTAL
LOCAL_BAD_REVIEW
LOCAL_BAD_UNCERTAIN
LOCAL_BAD_FALSE_PASS

继续研究期望：

REVIEW + UNCERTAIN >= 6/8

ideally：

REVIEW >= 5/8。

如果多数仍PASS：

DOMAIN_GAP_TOO_LARGE。

下一步应该：

local labels + fine-tuning

而不是模型变大。

==================================================
31. TRAINING PERFORMANCE
==================================================

记录：

training seconds
epoch seconds
samples/sec
peak VRAM
peak RAM
batch
AMP
workers。

不能把整个plot一次加载进RAM。

==================================================
32. FUTURE PRODUCT SPEED IS A HARD CONTRACT
==================================================

这个AI功能不能破坏之前已经完成的 TreeLineWarning 性能优化。

未来产品要求：

NO_FULL_SOURCE_RESCAN = YES

AI QC只能处理：

existing candidate membership
+
small local context。

禁止：

重新扫描105M source
重新Ground
重新Wire
重新Vegetation
重新Tree
重新Clearance。

==================================================
33. FUTURE QC ARCHITECTURE
==================================================

产品未来必须优先采用：

Tree segmentation completes
        ↓
Tree Review immediately available
        ↓
QC runs in background
        ↓
results appear asynchronously

而不是：

Tree segmentation
↓
wait QC
↓
Tree Review。

AI QC 是：

旁路 advisory stage

不是主算法关键路径。

==================================================
34. EDIT INCREMENTAL QC
==================================================

如果用户修改：

T-006

只能：

invalidate/recompute T-006

必要时加少量邻域相关tree。

禁止：

17棵全部重算

更禁止：

105M source重算。

QC cache key未来应该至少绑定：

source identity
tree membership revision/hash
model version
sampler/config version。

==================================================
35. CPU/GPU DEPLOYMENT BENCHMARK
==================================================

最终模型必须测试：

CPU inference

GPU inference。

由于科学 clearance 等计算可能需要GPU：

如果CPU QC足够快：

未来产品优先CPU。

不得为了省几百毫秒抢占科学GPU。

记录：

preprocess ms/tree
inference ms/tree
total ms/tree

以及：

1 tree
17 trees
100 trees。

==================================================
36. PRODUCT PERFORMANCE GATE
==================================================

目标：

17 trees QC <= 3 sec：

EXCELLENT。

3~5 sec：

允许后台异步。

如果同步增加 >约5% one-click critical-path：

禁止同步接入。

如果：

17-tree QC >10 sec

必须先优化，
不得直接产品化。

不能通过削弱原科学算法换AI速度。

==================================================
37. ONNX DEPLOYMENT PRECHECK
==================================================

只有：

PUBLIC PILOT PASS

且：

LOCAL challenge promising

才做ONNX。

优先当前 supported exporter：

torch.onnx.export(..., dynamo=True)

先尝试：

full model

失败：

sampling/grouping留在轻量preprocessing
neural trunk导ONNX。

比较至少100 samples：

PyTorch logits
ONNX logits

记录：

MAX_ABS_ERROR
STATUS_MATCH_RATE。

ONNX失败：

不等于研究失败。

==================================================
38. NO PRODUCT INTEGRATION
==================================================

这次无论结果多好：

PRODUCT_INTEGRATION = NO

禁止：

修改Tree Review UI
修改PyInstaller
bundle model
修改installer
修改冻结ZIP。

成功只能到：

MODEL_WORTH_LOCAL_FINETUNING

因为现在仍缺少可靠 local GOOD。

==================================================
39. SUBAGENT CHECKPOINTS
==================================================

Supervisor必须在以下位置主动停止Worker继续并审核：

GATE A
CUDA + MINI OVERFIT

GATE B
DATASET/LICENSE

GATE C
DEDUP + SPLIT

GATE D
SYNTHETIC GENERATOR REALISM

GATE E
FIRST TRAINING RESULT

GATE F
HELD-OUT SITE

GATE G
MULTI-SEED + ROBUSTNESS

GATE H
LOCAL BLIND CHALLENGE

GATE I
PERFORMANCE + ONNX

每个Gate：

Supervisor读取实际产物和日志。

不得只相信Worker一句：

PASS。

==================================================
40. FAILURE BEHAVIOR
==================================================

如果某路线失败：

Supervisor先分析根因。

不要自动：

扩大模型
下载更多数据
增加epoch
增加GPU依赖

来掩盖失败。

优先判断：

implementation bug
data leakage
dataset mismatch
generator artifact
domain gap
insufficient local labels
sampling problem
threshold problem。

==================================================
41. FINAL INDEPENDENT REVIEW
==================================================

完成后必须调用一个未参与实现的 Reviewer Agent。

Reviewer只读。

审查：

Git
dataset manifest
dedup
split
training config
threshold
held-out evaluation
local challenge
multi-seed
performance
licenses
ONNX。

Reviewer必须给：

PASS
PASS_WITH_LIMITATIONS
FAIL

以及具体问题。

Supervisor不能忽略Reviewer发现的重大问题。

==================================================
42. GIT
==================================================

Research code可以commit。

不要push。

不要commit：

datasets
large model binaries
checkpoints
ONNX
downloads
cache。

.gitignore覆盖。

保留：

source
small manifests
configs
small reports
tests。

==================================================
43. WINDOWS USER AUTOMATION
==================================================

Worker必须生成：

SETUP_TREE_QC_TRAINING.cmd
RUN_TREE_QC_MINI_GATE.cmd
DOWNLOAD_TREE_QC_PILOT_DATA.cmd
START_TREE_QC_PILOT.cmd
RESUME_TREE_QC_PILOT.cmd
EVALUATE_TREE_QC_PILOT.cmd

用户不应该需要手写CUDA命令。

==================================================
44. FINAL VERDICT
==================================================

Supervisor最终只能使用：

PRE_DOWNLOAD_GATE_FAILED

PUBLIC_PILOT_NOT_LEARNABLE

SYNTHETIC_GENERATOR_OVERFIT

MODEL_UNSTABLE

SAMPLING_TOO_SENSITIVE

DOMAIN_GAP_TOO_LARGE

MODEL_WORTH_LOCAL_FINETUNING

禁止：

PRODUCT_READY

==================================================
45. FINAL SUPERVISOR REPORT
==================================================

最终由 GPT-6 Supervisor 自己写：

TREE_QC_GPT6_MULTI_AGENT_SUPERVISOR_FINAL_REPORT

SUPERVISOR_MODEL =
SUBAGENTS_USED =
SUBAGENT_TASKS =
SUBAGENT_FAILURES =

BASELINE_HEAD =
FINAL_HEAD =
BRANCH =
WORKTREE =

GATE_A_CUDA_MINI =
GATE_B_DATA_LICENSE =
GATE_C_DEDUP_SPLIT =
GATE_D_GENERATOR =
GATE_E_TRAINING =
GATE_F_HELDOUT =
GATE_G_REPEATABILITY =
GATE_H_LOCAL_CHALLENGE =
GATE_I_DEPLOYMENT =

DATA_ROOT =
TRAINING_ROOT =
D_FREE_BEFORE =
D_FREE_AFTER =

GPU =
DRIVER =
PYTORCH =
PYTORCH_CUDA =
AUTO_BATCH =
AMP =
PEAK_VRAM =
PEAK_RAM =

PUBLIC_DATASET =
DATASET_VERSION =
LICENSE =
SITES =
DOWNLOAD_SIZE =
UNIQUE_TREES =
DUPLICATES_REMOVED =

POINTNET_PARAMETERS =
POINTNET2_PARAMETERS =
MODEL_SIZE_MB =

POINTNET_RESULTS =
POINTNET2_RESULTS =

CROSS_SITE_A_TO_B =
CROSS_SITE_B_TO_A =

BAD_RECALL =
BAD_FALSE_PASS_RATE =
GOOD_FALSE_REVIEW_RATE =
BALANCED_ACCURACY =
UNCERTAIN_RATE =

MERGED =
CONTAMINATION =
OVERSEG =
CROWN_INCOMPLETE =
STEM_MISSING =

MULTI_SEED =
SAMPLING_ROBUSTNESS =

LOCAL_BAD_TOTAL =
LOCAL_BAD_REVIEW =
LOCAL_BAD_UNCERTAIN =
LOCAL_BAD_FALSE_PASS =
DOMAIN_GAP =

CPU_MS_PER_TREE =
GPU_MS_PER_TREE =
TOTAL_17_TREE_CPU_SEC =
TOTAL_17_TREE_GPU_SEC =

NO_FULL_SOURCE_RESCAN = YES
ASYNC_ARCHITECTURE =
INCREMENTAL_RECOMPUTE =
EXISTING_PERFORMANCE_OPTIMIZATION_PRESERVED =

ONNX =
ONNX_EQUIVALENCE =

PRODUCT_INTEGRATION = NO
FROZEN_PRODUCT_CHANGED = NO
SCIENCE_CHANGED = NO

INDEPENDENT_REVIEWER_VERDICT =

PUSH = NO

NEXT_STEP =
FINAL_VERDICT =