# TLS Forest Instance Segmentation Benchmark 数据审计

核验日期：2026-09-09。状态：**元数据候选，下载 Gate 未开放，训练 Gate 未批准**。未请求任何 ZIP/权重下载；未探测数据归档的 Range 内容；未读取生产点云。

## 来源与固定版本

- 数据候选：[Zenodo record 16875688](https://zenodo.org/records/16875688)，DOI `10.5281/zenodo.16875688`。比索引可见的 14615493 / 15463506 更新，但 latest API 链未能访问，因此“最新”仍待证明。
- 论文：[Cherlet et al., ISPRS JPRS 231 (2026), 230–247](https://doi.org/10.1016/j.isprsjprs.2025.10.033)；[作者机构书目](https://biblio.ugent.be/publication/01KAVB41FW55319QCJV1MH1VXB)。
- 官方评价代码：[TreeInstSegEval](https://github.com/qforestlab/TreeInstSegEval/tree/6210682210d2783e6a5e58e4a1d63d2632e20748)，GitHub API 可访问，master SHA `6210682210d2783e6a5e58e4a1d63d2632e20748`，pushed_at `2025-01-14T15:05:07Z`。

## 官方文件索引

以下为 16875688 官方记录的检索索引展示值。MD5 是发布方提供的预期校验值，**不是本机计算结果**。大小为显示值，不是精确字节值。

| 文件 | 显示大小 | 官方 MD5 |
|---|---:|---|
| litchfield.zip | 2.3 GB | 8e374d940d7e651484a84d7d63f18636 |
| wytham.zip | 3.2 GB | 8d7918ed1270d603efc83aae5a8777c1 |
| ofental.zip | 5.6 GB | 9bfc9d05351633f7a6f10e5a8364e78a |
| robsoncreek.zip | 2.9 GB | 4051a58b19c0900a4db2972b2843146c |
| final_litchfield.png | 1.9 MB | ef45024a117b8a7ac9ba006ccb2853e9 |
| final_wytham.png | 2.3 MB | 8ead4414659969373c3b07607a6a47ba |
| final_ofental.png | 1.5 MB | 18cdecdfce1d7151c2d85597482362a3 |
| final_robsoncreek.png | 1.7 MB | 17b03a2fee38ed072bf70de2c2ab5dd0 |
| ForAINet_finetuned_weights_all.pt | 617.6 MB | fab1fb932c1982cce90b2364168c7d08 |
| TreeLearn_finetuned_weights_all.pth | 361.6 MB | 577847cafec23e3a5886a680687f5afd |

来源：[官方记录文件表](https://zenodo.org/records/16875688)。四个 ZIP 的 MD5 与两个旧记录展示值一致。精确 bytes、API download link 和 SHA256 尚未核验，不能虚构。

优先两站的标准记录下载路由候选为：

- `https://zenodo.org/records/16875688/files/litchfield.zip?download=1`
- `https://zenodo.org/records/16875688/files/wytham.zip?download=1`

这些是依照已知固定记录/文件名构造的精确 URL 字符串，**尚未以官方 API links 字段或 HTTP 成功响应确认**；不得把候选 URL 标记为已验证下载来源。其余文件同样需从成功的 API 响应取 links，不依赖猜测。

## site / plot / tree 与字段

官方数据说明：四个样地分别位于 Litchfield（澳大利亚热带稀树草原）、Wytham Woods（英国温带落叶林）、Ofental（德国温带针叶林）、Robson Creek（澳大利亚热带雨林），合计 2,983 棵人工分割树。九名操作人员提取并经一名操作人员二次质量检查。每站一个样地，**两站不等于多样地重复实验**。

各样地含 train / validation / test 区域 PLY，`semantic` 为 0=ground、1=tree；`instance` 从 1 顺序编号，ground=-1。还有对应区域的单树 PLY。不同站点的 instance 数字不能直接视为全球唯一 ID，建议键为 `(record_id, site, plot, canonical_tree_id)`。

官方说明区域 PLY 可合并为完整标注样地；这支持“同一样地共同坐标系”假设。它**没有证明 EPSG/大地坐标、坐标单位、原点、树文件与区域文件逐点对应关系**。共同局部坐标就足够构造真实邻域，不能将未知 CRS 写成已证实 world georeferencing。上述均需获准下载后的 PLY header、bbox、坐标范围/单位和树到样地匹配检查确认。[官方数据说明](https://zenodo.org/records/16875688)

## boundary 重叠与真实邻域

官方明确单树文件按与区域相交关系分配，边缘树可能跨 train/val/test 文件夹重复。不得按单树文件随机切分，再把相邻/相同树算作独立测试。

官方 [README](https://github.com/qforestlab/TreeInstSegEval/blob/6210682210d2783e6a5e58e4a1d63d2632e20748/README.md) 要求预测保持坐标和尺度；评价采用距离匹配。其 [base_evaluation.py](https://github.com/qforestlab/TreeInstSegEval/blob/6210682210d2783e6a5e58e4a1d63d2632e20748/base_evaluation.py) 从 test 区域 bbox 裁剪 GT，并区分 in_plot / out_plot 树。README 的 `trees/test/...` 与代码 `read_gt` 的 `gt_dir/test/...` 有路径差异，实际 ZIP 布局须核验，不能直接照抄。

建议的后续科学检查（不是 Gate 批准）：

1. 以完整样地的原始共同坐标取目标树及实际邻居；先构建邻域，再进行可逆的训练坐标归一化。每样本保留 origin/scale/plot/source IDs。
2. 跨文件夹实例去重，并以空间点重叠或稳定树 ID 检查 leakage；切分应先于扰动和邻域构建。跨边界邻域必须 purge / buffer 或整组同分配。
3. train / val / test 都记录 unique source trees、unique plots/sites、边界排除数量。增加扰动次数只增加样本，不增加独立 GT 树数量。
4. 实际几何邻域不等于同站点随机配树，不能用随机拼接代替真实邻树。

## 可训练 GT 规模

官方公开记录确认四站共 2,983 棵，但本次可读取的官方元数据没有按站或按 split 的精确去重计数。检索出现论文的 703（Litchfield）/876（Wytham）、128/181 evaluation trees 数字，但本次未取得作者机构全文做一次来源确认，因此不将这些计数升级为训练规模证据。作者机构元数据确认总标注面积约 2.7 ha；无法由此推导合法训练树数量。

两站优先方案在文件量及生态差异上适合 pilot 候选，但 unique train / val / test trees、可恢复邻域数量及边界排除后规模当前均为 **unknown**。在这些统计完成前，不承诺 train-size Gate 已满足；也不能将这两个 site 的空间切片结果外推为跨森林类型泛化。

## 容量和可访问性

Litchfield + Wytham 压缩包约 **5.5 GB（十进制、按显示值相加）**；四站约 14 GB。解压 bytes 未知；没有读取 ZIP central directory，因此不得提供伪精确膨胀率。若预算假设解压为压缩的 2–4 倍，两站仅归档+展开约 16.5–27.5 GB；该范围只是规划情景，不是实测或上限。还需转换副本、训练缓存、checkpoint 与临时文件空间。容量 Gate 应以实测元数据/展开清单约束。

实测访问记录：GitHub repo/commit API 与 raw master 文本成功；main 分支 404（真实 default branch 是 master）。Zenodo `/api/records/16875688`、`/api/records/16875688/files`、`/records/16875688/export/json` 及记录页本机多次 12–30 秒 timeout，web 直读记录返回 429。UGent 书目可读，全文入口 web fetch 失败，本机 HTTPS 验证失败；没有禁用证书校验。

待补齐：官方 JSON 快照（version/latest/license/每文件 bytes/links）、精确容量，再由监督者决定是否开放下载 Gate。下载后再核验归档 hash、文件清单、PLY schema、共同坐标、unique 树计数及跨 split 重叠；这些证据通过后才可提交训练 Gate 审阅。
