# 胸径与净空输出修复（2026-09-05）

当前结果页、最终单木表与 Prediction_Input.xlsx 新增水平、垂直净空分量。
三者复用已定稿树木与导线的同一个三维最近点对：

- `horizontal_clearance_m = hypot(tree.x - wire.x, tree.y - wire.y)`。
- `vertical_clearance_m = abs(tree.z - wire.z)`。
- `clearance_component_source = EXACT_3D_NEAREST_PAIR_COMPONENTS`。

这两个数是三维最近点对的分量，并非分别求解的水平、垂直方向最小距离。
发布前检查点对坐标有限、状态有效，且其三维距离与已保存净空一致。
缺失或不一致的证据输出空值；真实零保留。

胸径恢复值经过原有来源、成员、证据哈希校验后，即使常规地面指标不可用，仍会发布。
常规胸径也统一输出 `dbh_cm`、质量、点数与方法。未测得的记录保留失败原因，
不会用树冠宽度或常数代替胸径。旧版 Excel 缺少新列时自动重新导出；胸径发布更新也会使缓存失效。

真实项目：`E:/2026.8.31-2/2026.08.31-2/2026.08.31-2`。
当前批次：`analysis-d434dfe375814b2d9b8a1d765a3e4d13`。
12 棵树，4,765,121 个最终成员点。V3 恢复重跑仍为 7 棵有胸径、5 棵未测得：
3 棵 `NO_ABSOLUTE_CANDIDATE`，2 棵 `NO_CROWN_TO_STEM_IDENTITY`。
重跑采用只读检查，已有有效胸径发布证据保留。

针对测量字段、真实空值、Excel/API 一致性、旧导出缓存、DBH 恢复与结果页运行的 79 项测试通过。
实际项目输出和验证记录保存于 `artifacts/measurements_20260905/`。
