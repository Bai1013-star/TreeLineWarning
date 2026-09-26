# TreeLine Capture 最小采集器与 PC 数据接口（V1）

> 状态：可进入原型开发\
> 建议首发平台：iOS / ARKit\
> 接口版本：`tlc-capture-v1.0`\
> 适用范围：科研验证与短档距现场采集，不作为测绘级成果

## 1. 结论

第一版只做一个轻量的 **TreeLine Capture**，不在手机上做点云分析、树种推理或风险计算。

手机端只负责：

1. 创建采集任务；
2. 录制原始相机画面；
3. 同步保存 ARKit 相机位姿、相机内参、时间戳、GPS、航向和 IMU；
4. 引导用户依次拍清 START、MID、END 三个 AprilTag；
5. 做最基本的采集质量检查；
6. 导出一个可校验的 `SurveyCapture_*.zip`。

TreeLineWarning PC 端负责：

1. 校验并解包采集包；
2. 检测 AprilTag；
3. 将 Tag 观测与 LiDAR 中的 Tag 坐标对应；
4. 求解手机轨迹到 LiDAR 坐标系的变换并评估误差；
5. 把已确认单木投影到视频；
6. 建立视频实例与稳定单木 ID 的关联；
7. 执行视觉树种识别、多模态融合和人工复核。

## 2. 为什么 V1 先做 iOS

iOS 版可直接使用 ARKit 的世界跟踪结果，逐相机帧读取位姿、内参、图像分辨率和时间戳；GPS 与航向由 Core Location 提供。第一版只维护一套设备行为、时间同步和坐标约定，验证成本最低。

Android/ARCore 不应进入第一版，但接口中预留 `platform`、`pose_provider` 和坐标约定字段。将来 Android 只需生成同一套规范化数据，不需要改变 PC 主流程。

## 3. 现场标志物方案

### 3.1 推荐配置

- Tag family：`tagStandard41h12`；
- START：ID `10`；
- MID：ID `15`；
- END：ID `20`；
- 三块 Tag 的黑色有效边长必须实测并记录，建议原型统一为 `0.240 m`；
- 刚性平板安装，避免纸张弯曲；
- 现场每个 Tag 至少正面停留 2 秒，并从轻微不同角度拍摄；
- Tag 不应紧贴强反光表面、被手遮挡或填满整个画面；
- START、MID、END 应保持不同 ID，不依赖出现顺序猜测身份。

AprilTag 官方实现需要已知 Tag 实际尺寸和相机内参才能估计 6DoF 位姿，因此 Tag 尺寸是接口中的必填数据，不能只写在操作手册里。

### 3.2 三个 Tag 的几何含义

在 ARKit/VIO 模式下，重力已经约束竖直方向，START 与 END 可解除两点配准剩余的主要旋转自由度；MID 主要用于发现并约束长距离 VIO 漂移。因此 START、MID、END 沿线路近似共线仍可工作。

在普通 MP4 + COLMAP 备用模式下，三点不能近似共线。此时应使用至少 3 个非共线、最好 4 个以上的控制标志物，并让标志物在走廊左右两侧形成足够横向基线。仅把 MID 放在线路中心线上，不能稳定解除绕长轴旋转的退化。

## 4. 手机端 MVP 功能清单

### 4.1 必须实现（P0）

#### 任务设置

- 自动生成 `capture_id`（UUID）；
- 输入或扫码获得项目编号、线路名称、杆塔区间；
- 显示并允许确认 Tag family、START/MID/END ID 和实际边长；
- 选择固定清晰度档位，V1 默认 1920×1440、H.264、约 30 fps；
- 记录 App、接口、iOS、设备和 ARKit 版本。

#### 采集

- 只允许横屏采集并锁定方向；
- 同一 ARKit `ARFrame` 同时产出视频帧、位姿、时间戳和内参引用；
- 每个真正写入 MP4 的视频帧都必须有一条 pose 记录；
- 记录 ARKit tracking state；
- 记录 GPS、水平/垂直精度、真北/磁北航向及其精度；
- 记录重力、角速度和用户加速度；
- 提供“标记 START / MID / END”按钮，只标记时间窗口，不在 V1 手机端执行 AprilTag 检测；
- 来电、切后台、存储不足、相机中断或 ARKit relocalization 时明确标记采集异常。

#### 现场提示

- 开始前检查相机和定位权限、可用存储空间、AR tracking 状态；
- 提示“请缓慢行走、避免快速甩动、保持树干与树冠有多视角重叠”；
- 三个 Tag 标记均完成后才能显示“可结束”；
- 实时显示录制时长、已写入帧数、当前 tracking 状态和三个 Tag 的人工标记状态；
- 连续出现 `limited`/`notAvailable` 超过阈值时给出醒目提示，但不中断原始数据保存。

#### 结束与导出

- 先安全结束视频写入，再生成元数据与摘要；
- 逐文件计算 SHA-256；
- 生成 `manifest.json` 作为最后写入的完成标记；
- 导出 `.zip`，支持系统分享、AirDrop 或保存到“文件”；
- 导出前在本机重新打开并校验清单、文件大小、摘要和 JSON 可解析性；
- 不完整任务使用 `INCOMPLETE` 状态，禁止伪装成可分析数据包。

### 4.2 可以延后（P1）

- 手机端实时 AprilTag 检测和自动打点；
- START/MID/END 拍摄质量评分；
- 语音提示和震动反馈；
- 轨迹俯视图；
- 导出前低质量段回放；
- 多档距连续采集。

### 4.3 V1 明确不做

- 手机端点云浏览；
- 手机端树木实例分割；
- 手机端树种模型；
- 手机端 DBH、净空、风险和预测；
- 云端账号、云同步和后台上传；
- Android；
- 测绘级 GNSS 或“厘米级定位”承诺。

## 5. 推荐的手机页面

V1 只需 5 个页面：

1. **任务列表**：新建、继续未完成任务、导出或删除本地任务；
2. **采集设置**：项目/档距信息与 Tag 参数；
3. **采集前检查**：权限、空间、相机、AR tracking、定位状态；
4. **采集页面**：取景、录制状态、START/MID/END 按钮、质量警告；
5. **采集摘要**：时长、帧数、轨迹长度、tracking 异常、Tag 标记、ZIP 导出。

手机端不需要登录页、分析页和复杂项目管理。

## 6. `SurveyCapture` ZIP 结构

```text
SurveyCapture_20260908_001.zip
├─ manifest.json
├─ metadata.json
├─ video.mp4
├─ poses.jsonl
├─ intrinsics.json
├─ gps.jsonl
├─ imu.jsonl
├─ anchor_events.json
└─ preview.jpg                 # 可选，不参与计算
```

约束：

- ZIP 根目录直接包含上述文件，不再包一层同名目录；
- 路径一律使用 UTF-8、相对 POSIX 路径，不允许 `..`、绝对路径或反斜杠；
- `manifest.json` 必须最后生成；
- JSON 使用 UTF-8、禁止 NaN/Infinity；
- JSONL 每行是一个完整 JSON 对象，以换行结尾；
- 所有距离为米、角度为度、时间间隔为秒；
- 所有矩阵均为 4×4 **row-major** 数组；
- PC 导入时先校验，再解压到新建的隔离目录，禁止覆盖已有采集。

## 7. 坐标系与时间基准

### 7.1 规范化相机坐标

采集包不直接把 ARKit 原生相机轴约定留给 PC 猜测。V1 统一保存：

- `capture_world`：右手系、米制，`+Y` 为上；原点为本次 ARKit session 原点；
- `camera_cv`：OpenCV 相机坐标，`+X` 向图像右、`+Y` 向图像下、`+Z` 向前；
- `T_capture_world_from_camera_cv`：把 `camera_cv` 中的齐次点变换到 `capture_world`；
- LiDAR：沿用 I1 `LOCAL_CARTESIAN_XYZ`，单位米，并明确 `+Z` 为上。

ARKit 原生相机坐标转换为 `camera_cv` 时采用：

```text
C_arkit_camera_from_camera_cv = diag(1, -1, -1, 1)
T_capture_world_from_camera_cv = T_arkit_world_from_camera × C
```

手机端必须用单元测试固定这项转换，并用已知棋盘/Tag 场景做一次实拍验证。

### 7.2 时间

- `t_capture_s`：与 ARFrame timestamp 同源的单调时间，任务开始帧为零；
- `video_pts_s`：该帧实际写入 MP4 的 presentation timestamp；
- `utc_time`：ISO 8601 UTC，只用于跨设备记录和 GPS 对照，不参与逐帧同步；
- pose 与视频的主键是 `frame_index + video_pts_s`，不能只靠文件帧率反推；
- GPS/IMU 以各自时间戳保存，由 PC 插值到 `t_capture_s`。

## 8. 文件字段定义

### 8.1 `manifest.json`

```json
{
  "schema_version": "tlc-capture-v1.0",
  "capture_id": "5ee55c26-1045-4f7d-a9b5-0c76d7431193",
  "completion_state": "COMPLETE",
  "created_at": "2026-09-08T03:12:45.230Z",
  "files": [
    {
      "path": "video.mp4",
      "media_type": "video/mp4",
      "bytes": 184329002,
      "sha256": "<64 lowercase hex>"
    }
  ]
}
```

`files` 必须列出除 `manifest.json` 自身之外的每个包内文件。PC 拒绝摘要不符、重复路径、未声明文件或缺少必需文件的数据包。

### 8.2 `metadata.json`

```json
{
  "schema_version": "tlc-metadata-v1.0",
  "capture_id": "5ee55c26-1045-4f7d-a9b5-0c76d7431193",
  "project_id": "TLW-DEMO-001",
  "line_name": "示范线路",
  "tower_section": "T001-T002",
  "platform": "ios",
  "pose_provider": "arkit_world_tracking",
  "app_version": "0.1.0",
  "device_model": "iPhone",
  "os_version": "iOS",
  "capture_started_at": "2026-09-08T03:10:00.000Z",
  "capture_duration_s": 165.42,
  "video": {
    "codec": "h264",
    "width_px": 1920,
    "height_px": 1440,
    "nominal_fps": 30.0,
    "orientation": "landscape_right",
    "pixel_format": "yuv420"
  },
  "coordinate_convention": {
    "world_frame": "capture_world_right_handed_y_up",
    "camera_frame": "opencv_x_right_y_down_z_forward",
    "pose_matrix": "T_capture_world_from_camera_cv",
    "matrix_layout": "row_major",
    "length_unit": "m"
  },
  "tags": [
    {"role": "START", "family": "tagStandard41h12", "id": 10, "size_m": 0.24},
    {"role": "MID", "family": "tagStandard41h12", "id": 15, "size_m": 0.24},
    {"role": "END", "family": "tagStandard41h12", "id": 20, "size_m": 0.24}
  ],
  "limitations": ["consumer_gps_not_survey_grade"]
}
```

### 8.3 `poses.jsonl`

每个写入视频的帧对应一行：

```json
{"frame_index":0,"t_capture_s":0.0,"video_pts_s":0.0,"intrinsics_id":"K000","T_capture_world_from_camera_cv":[[1,0,0,0],[0,1,0,1.62],[0,0,1,0],[0,0,0,1]],"tracking_state":"normal","tracking_reason":null}
```

要求：

- `frame_index` 从 0 连续递增；
- 两类时间戳严格单调不减；
- 位姿矩阵最后一行为 `[0,0,0,1]`；
- 旋转部分必须接近正交且行列式接近 `+1`；
- `tracking_state` 为 `normal | limited | not_available`；
- 发生中断或 relocalization 时不能静默拼接两个世界坐标系，必须结束当前 segment 或显式记录 segment 变化。

### 8.4 `intrinsics.json`

```json
{
  "schema_version": "tlc-intrinsics-v1.0",
  "image_coordinate": "video_encoded_pixels_origin_top_left",
  "models": [
    {
      "intrinsics_id": "K000",
      "valid_from_frame": 0,
      "valid_to_frame": 4962,
      "width_px": 1920,
      "height_px": 1440,
      "camera_model": "PINHOLE",
      "fx": 1482.31,
      "fy": 1481.94,
      "cx": 959.41,
      "cy": 719.88,
      "distortion": null
    }
  ]
}
```

`fx/fy/cx/cy` 必须对应 **实际编码视频像素**。如果视频发生缩放、裁剪或旋转，手机端必须同步变换内参；不能直接复制 ARKit 原始图像内参。若内参变化，则新增 `intrinsics_id`，不覆盖旧值。

### 8.5 `gps.jsonl`

```json
{"t_capture_s":0.42,"utc_time":"2026-09-08T03:10:00.420Z","latitude_deg":39.9042,"longitude_deg":116.4074,"altitude_m":48.2,"horizontal_accuracy_m":4.8,"vertical_accuracy_m":8.1,"true_heading_deg":92.4,"magnetic_heading_deg":98.1,"heading_accuracy_deg":7.0}
```

GPS 和航向只用于粗初始化、轨迹方向核验和现场记录。除非以后接入 RTK，不得把消费级 GPS 当作 Tag/LiDAR 精配准约束。

### 8.6 `imu.jsonl`

```json
{"t_capture_s":0.013,"device_frame":"ios_core_motion","gravity":[0.012,-0.981,-0.191],"user_acceleration_mps2":[0.03,0.01,-0.04],"rotation_rate_radps":[0.002,-0.004,0.001]}
```

V1 的几何主输入仍是 ARKit pose；原始 IMU 用于诊断、后续重优化和发现剧烈抖动，不单独进行惯导积分。

### 8.7 `anchor_events.json`

```json
{
  "schema_version": "tlc-anchor-events-v1.0",
  "events": [
    {"role":"START","tag_id":10,"marked_t_capture_s":4.22,"search_window_s":[1.22,7.22]},
    {"role":"MID","tag_id":15,"marked_t_capture_s":81.35,"search_window_s":[78.35,84.35]},
    {"role":"END","tag_id":20,"marked_t_capture_s":159.02,"search_window_s":[156.02,162.02]}
  ]
}
```

人工事件只缩小 PC 检测搜索范围，不是“Tag 已识别”的证据。PC 必须在视频中检测到正确 ID 后才能建立控制点。

## 9. PC 端导入与处理流程

```text
ZIP 安全校验
  → schema / hash / 帧-位姿一致性检查
  → 视频解码与 Tag 搜索
  → 多帧融合每个 Tag 在 capture_world 中的位置/姿态
  → 用户在 LiDAR 中框选对应 Tag 区域并拟合中心/平面
  → 求解 capture_world → LiDAR 的 Sim(3)
  → 残差、尺度、重力、漂移与覆盖率质量门控
  → 变换全部相机位姿
  → 将 finalized tree geometry 投影到视频
  → 跨帧关联视频实例与单木 ID
  → 视觉树种候选 + 点云候选融合
  → 高置信自动写候选，低置信进入人工复核
```

### 9.1 配准求解

#### 三 Tag 推荐路径

1. 对每个 Tag 的多帧观测做鲁棒融合，得到 `P_capture_tag`；
2. 对 LiDAR 框选点拟合 Tag 平面和中心，得到 `P_lidar_tag`；
3. 用加权 Umeyama/非线性最小二乘求 `P_lidar = s R P_capture + t`；
4. ARKit 本身为米制，因此 `s` 应接近 1；明显偏离必须判为异常而不是自动接受；
5. 使用 START/MID/END 分段残差评估 VIO 漂移；
6. 研究版先做单一全局 Sim(3) + 质量门控，长档距再升级为带 VIO 相对位姿边和 Tag 固定因子的 pose-graph 优化。

#### 双 Tag + 重力路径

用两端向量与竖直方向构造正交基，求旋转、尺度和平移。必须检查：

- 两 Tag 间基线足够长；
- 两端向量不接近竖直方向；
- LiDAR `+Z` 的定义可靠；
- 两端 Tag 都有多帧稳定检测；
- 中间轨迹只做插值使用，不宣称无漂移。

#### 普通 MP4 路径

普通视频通过 COLMAP/SfM 恢复相机轨迹，使用至少 3 个非共线控制点，推荐 4 个以上，并单独报告重建失败、控制点退化和尺度不确定性。它是兼容模式，不是 V1 的默认工作流。

### 9.2 第一版质量门槛（原型默认值）

下列值是工程起点，必须用真实样本重新标定：

| 指标 | 通过 | 人工复核 | 拒绝自动绑定 |
| --- | ---: | ---: | ---: |
| 有效 Tag 数 | 3 | 2 | < 2 |
| 每个 Tag 有效帧 | ≥ 10 | 4–9 | < 4 |
| 对齐 RMSE | ≤ 0.30 m | 0.30–0.75 m | > 0.75 m |
| 最大 Tag 残差 | ≤ 0.50 m | 0.50–1.00 m | > 1.00 m |
| 求得尺度 `s` | 0.98–1.02 | 0.95–1.05 | 其余 |
| `tracking_state=normal` 帧占比 | ≥ 95% | 80–95% | < 80% |
| 视频帧与 pose 覆盖率 | 100% | 不适用 | < 100% |

任何“拒绝自动绑定”只禁止自动写入单木关联，不删除视频和中间结果，允许用户查看原因并人工处理。

## 10. 视频树木与单木 ID 的关联

关联顺序必须是“先确定是哪棵树，再识别树种”：

1. 从 V2 finalized/confirmed cluster 读取单木点集、包围盒、中心与顶部；
2. 只投影位于相机前方且落入图像范围的点；
3. 用深度排序和遮挡近似得到每棵树的 2D mask/box；
4. 与视频树木实例的 mask IoU、中心距离、左右顺序、尺度和连续帧轨迹联合打分；
5. 多帧全局匹配，避免逐帧贪心造成 Tree ID 跳变；
6. 低分、遮挡或一对多冲突进入人工复核；
7. 关联一旦确认，视觉证据才写入该树的物种候选记录。

建议输出独立结果包，而不是直接修改 I1/V2 原始产物：

```text
outputs/vision/<capture_id>/
├─ alignment.json
├─ tag_detections.jsonl
├─ frame_tree_projections.jsonl
├─ tree_video_associations.json
├─ species_evidence.json
└─ review_log.json
```

`tree_video_associations.json` 的核心记录：

```json
{
  "association_id": "a4ed...",
  "capture_id": "5ee55c26-1045-4f7d-a9b5-0c76d7431193",
  "source_id": "scene_simple",
  "persistent_tree_id": "T-1-007",
  "v2_cluster_id": 7,
  "video_instance_id": "VI-0014",
  "frame_range": [1220, 1498],
  "association_score": 0.91,
  "status": "CONFIRMED",
  "source": "AUTO_REVIEWED"
}
```

## 11. 与当前 TreeLineWarning 工程的实际衔接

当前仓库已经具备两个可复用基础：

- I1 数据包已有版本号、SHA-256、米制 `LOCAL_CARTESIAN_XYZ` 坐标上下文和不可变清单风格；
- V2 cluster 已有 `cluster_id`、中心、包围盒、顶部、基部和点成员映射，可作为视频投影几何来源。

但当前代码中尚未发现真正跨期稳定的 `persistent_tree_id` 合同；预测侧主要使用整数 `tree_id`，真实 I1 适配器也暂时把 `cluster_id` 当作 `tree_id`。因此在自动视频绑定之前，应先增加一个独立、不可随重新分割而变化的单木身份层，例如：

```text
persistent_tree_id
  ↔ source_id + survey_epoch + v2_cluster_id
  ↔ 后续各期 cluster
  ↔ prediction tree_id/tree_code
```

不能直接把本期 `cluster_id=7` 永久解释为 `T-1-007`，否则重新分割或下一期扫描后证据可能绑定到错误树木。

建议新增模块边界：

```text
core/contracts/capture.py
algorithms/vision/capture_import.py
algorithms/vision/apriltag_detection.py
algorithms/vision/alignment.py
algorithms/vision/projection.py
algorithms/vision/association.py
algorithms/vision/species_fusion.py
api/capture_api.py
tests/vision/
```

这条开发线不应修改现有风险阈值、DBH、生长预测和 GPU 推理主线。

## 12. 开发顺序与验收

### M0：桌面端数据包模拟器（先做）

- 用脚本生成合法/非法的 `SurveyCapture` 测试包；
- 完成 schema、摘要、ZIP 安全和 pose 矩阵校验；
- 用合成轨迹验证坐标转换与 Sim(3)；
- 验收：已知变换回收误差小于数值容差，所有损坏包可被明确拒绝。

### M1：iOS 最小采集器

- 完成 5 个页面、视频/pose/内参/GPS/IMU 同步记录和 ZIP 导出；
- 不做手机端 AprilTag 检测；
- 验收：连续录制 5 分钟无丢失 pose，导出包可被 PC 严格校验并逐帧对应。

### M2：Tag 配准闭环

- PC 检测 START/MID/END；
- LiDAR 中框选 Tag、拟合中心；
- 输出轨迹、变换、残差和质量等级；
- 验收：在有独立检查点的短档距试验中，检查点误差达到项目设定目标，而不只看三个参与拟合的 Tag。

### M3：单木视频绑定

- 投影 V2 单木；
- 完成跨帧实例关联和人工复核；
- 验收：以人工标注对应关系计算准确率、未决率和误绑定率；优先约束误绑定率，而非强求全部自动匹配。

### M4：树种融合

- 将高质量视频帧送入视觉树种模型；
- 与现有点云候选分开保留置信度和来源；
- 人工确认后写最终 species；
- 验收：输出可追溯到原视频帧、模型版本、候选分数和复核人员操作。

## 13. 第一轮外业验证最小样本

不要一开始就拍 200 m 正式档距。建议按以下顺序：

1. 室外 20–30 m、3 个 Tag、3–5 棵树、重复 5 次；
2. 同一路线正向/反向各拍，检查坐标轴和航向符号；
3. 人为制造一次短暂遮挡、快速转身和 tracking limited；
4. 增加到 80–100 m，观察 MID 前后残差；
5. 最后再测试完整档距。

每次都应保留至少 1 个不参与拟合的检查标志物，用它报告真实外部误差。只有三个拟合点上的低 RMSE 不能证明整条轨迹准确。

## 14. V1 的完成定义

当以下条件全部满足时，才算“视频 + Tag + 点云自动对应”V1 完成：

- 手机可稳定导出严格可校验的数据包；
- PC 能自动找到三个正确 Tag ID；
- LiDAR Tag 选择、坐标变换和误差报告可重复；
- 相机轨迹能正确显示在点云中；
- 至少一个真实短档距数据集完成视频树实例到稳定单木 ID 的绑定；
- 错误数据会被拒绝或转人工，不会静默写错 Tree ID；
- 所有结果保留 capture、源点云、算法版本、模型版本和人工复核记录。

## 参考实现依据

- [Apple ARCamera](https://developer.apple.com/documentation/arkit/arcamera)：ARFrame 的 camera 提供图像分辨率、内参与用于相机姿态的变换信息；
- [Apple Core Location](https://developer.apple.com/documentation/corelocation)：提供位置、航向及各自更新与精度信息；
- [AprilRobotics AprilTag 3](https://github.com/AprilRobotics/apriltag)：提供 Tag 检测、已知尺寸和相机参数下的 6DoF pose estimation，并推荐 `tagStandard41h12` family。
