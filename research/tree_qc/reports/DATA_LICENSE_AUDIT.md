# TLS Forest Benchmark 数据许可核验

核验日期：2026-09-09。范围：公开元数据与官方代码文本；没有下载数据归档、模型权重或读取真实生产点云。本报告不批准 Gate A 或 Gate B。

## 候选数据及官方声明

候选固定记录：[Zenodo 16875688](https://zenodo.org/records/16875688)，DOI `10.5281/zenodo.16875688`。检索可见旧记录 14615493、15463506 及该较新记录；本机 API 超时，未完成 latest-version 链证明，不能宣称已经确认全局最新版本。

官方记录原文：

> Any use of this dataset should cite the paper above (Creative Commons Attribution 4.0 International Public License).

上述为数据发布者明确的 CC BY 4.0 声明。[来源](https://zenodo.org/records/16875688)

所需论文引用：Cherlet, W. et al. (2026). *Benchmarking tree instance segmentation of terrestrial laser scanning point clouds*. ISPRS Journal of Photogrammetry and Remote Sensing, 231, 230–247. DOI：[10.1016/j.isprsjprs.2025.10.033](https://doi.org/10.1016/j.isprsjprs.2025.10.033)。作者所属机构书目页确认这组书目信息。[UGent 官方书目](https://biblio.ugent.be/publication/01KAVB41FW55319QCJV1MH1VXB)

## 商业含义和边界

Creative Commons 官方许可摘要的授权范围包括原文：

> for any purpose, even commercially.

它同时要求署名、提供许可链接、标明修改，禁止暗示作者背书和施加限制他人许可权利的额外限制。[CC BY 4.0 官方摘要](https://creativecommons.org/licenses/by/4.0/)；法律文本：[CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)。

据此，该许可本身没有 NonCommercial 限制，可以作为商业研究/开发的许可候选；这不是对商业部署中所有权利、模型权重权属或第三方权利的保证。不得把“训练使用数据”与“发布原数据、派生数据、权重、论文图表”混为同一个已审核事项。保留数据 DOI、作者/论文引用、许可 URL 和处理变更记录；具体分发对象确定后，依据对应材料许可决定署名载体。

## 代码、权重与论文分别处理

- 官方评价仓库：[qforestlab/TreeInstSegEval](https://github.com/qforestlab/TreeInstSegEval)，固定提交 `6210682210d2783e6a5e58e4a1d63d2632e20748`。GitHub API 实测 `license: null`；README 仍含数据地址及引用 TODO。没有观察到可用于代码再分发的明确许可证，因此不能用数据 CC BY 自动覆盖该仓库代码。这里只读研究评价方法，未复制代码到项目。
- Zenodo 同一记录包含 TreeLearn/ForAINet 权重；本任务没有下载或采用它们，不推断其适用于当前 PointNet++ 工作流。
- UGent 书目页对论文标记 `No license (in copyright)`；论文开放阅读不等同于图表自由再分发，也不改变 Zenodo 数据声明。

## 核验限制与下一步证据

Zenodo 搜索索引能读官方声明；网页直接获取出现 HTTP 429，本机 `/api/records/16875688`、`/api/records/16875688/files`、`/records/16875688/export/json` 多次超时。因此 API 中精确 license 对象、version 和 latest 链尚未保存。下载 Gate 开放前，应取得完整官方元数据快照，再核对其与上述声明一致。不得将本报告当作训练或商业发布批准。
