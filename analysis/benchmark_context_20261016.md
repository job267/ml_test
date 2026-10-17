# KuaiRand 公开基线与本项目成绩的可比性

核查日期：2026-10-16。此文件是公开资料与本地已完成实验的对照记录，不是统一排行榜，也没有修改正在运行的候选池来源消融。

## 可核查的公开参考

KDD 2025《Generalizable Recommender System During Temporal Popularity Distribution Shifts》（TPAB），Table 2 的 KuaiRand 结果：

| 方法 | Recall@20 | NDCG@20 |
|---|---:|---:|
| MF / Vanilla | 0.1111 | 0.0581 |
| LightGCN / Vanilla | 0.1281 | 0.0672 |
| TPAB / LightGCN | 0.1470 | 0.0769 |

论文第 4.1 节采用 22,128 用户、7,076 物品、621,064 条点击交互，按时间排序后以 8:1:1 划分，使用全物品排名，报告五次运行的均值和标准差。因此它是相关的全排序参考，但与本项目的随机曝光新目标、engagement 正例及日期切分不同，不能计算跨协议的算法提升或退化比例。

来源：[作者提供的 KDD 2025 论文，第 4.1 节、Table 2](https://q-rz.github.io/static/kdd25/kdd25-tpab-paper.pdf)；[作者实现](https://github.com/hsyoo32/tpab_)。

另一个协议的例子：IJCAI 2023《Discriminative-Invariant Representation Learning for Unbiased Recommendation》在 KuaiRand-Pure 上报告 MF NDCG@5=0.3271、DIRL NDCG@5=0.3991；采用 click 正负标签。这里 K、标签和处理协议不同，也不能直接与本项目 NDCG@20 比较。

来源：[IJCAI 2023 论文，第 4.1 节及 Table 1](https://www.ijcai.org/proceedings/2023/0252.pdf)。

官方数据说明提供 Pure、1K、27K 三个版本；Pure 保留随机候选池中的 7,583 个视频。数据集名称相同不保证预处理和评价任务相同。

来源：[KuaiRand 官方说明](https://kuairand.com/)。

## 本项目的实际协议

- 数据：KuaiRand-Pure，训练词表 24,524 用户、7,583 物品。
- 训练截止 2022-04-28；验证为 2022-04-29 至 2022-05-03；之后为 test。
- 正例：like / follow / long_view 任一成立；冲突反馈单独处理。
- 验证：随机曝光日志中的新正向用户—物品目标，共 10,005 用户、30,593 对。
- 对每个用户在全物品中排序，屏蔽训练正例。目标仅包含实际记录到的随机曝光正例，不代表已知全物品的完整真实偏好。
- 下面模型数字都是 validation 结果，不是最终 test，也不代表真实点击率。

证据：`data/kuairand_stage1/manifest.json`、`docs/stage1_implementation.md`。

## 同协议内的可比数字

| 方法 | NDCG@20 | Recall@20 |
|---|---:|---:|
| 均匀随机排序精确期望 | 0.0012328483 | 0.0026450879 |
| 训练热门排序 | 0.0040489102 | 0.0086562977 |
| LightGCN / uniform | 0.0044886694 | 0.0099385961 |
| beta=0，80% 定向、50% 证据候选 | 0.0047523572 | 0.0099446665 |
| beta=0.25 的探索配置 | 0.0047827962 | 0.0099920636 |

随机是精确期望，热门为确定性参照；模型为三个种子前 30 轮最佳验证模型均值。beta=0.25 是反复验证集探索中的一个配置，其均值最高不等于增量收益已获独立证实：两个新种子中，它与 beta=0 的平均 NDCG 几乎相同。

最高已完成配置比热门排序 NDCG 高约 18.13%，比 LightGCN / uniform 高约 6.55%。稀疏用户 NDCG 却比 uniform 低约 22.83%，长尾命中极少。当前可作为可追溯的研究基线，但证据不足以称为强竞争性结果或方法改进已成立。

证据：`runs/stage1_v1/validation_references.json`；`runs/stage1_beta_replication_20261016/comparison.json`。正在运行的 evidence_fraction=0 消融不并入未完成的三种子均值。

## 更有价值的下一步验证

1. 在当前固定 prepared 数据与评价器上补 MF/BPR、ItemCF 等同协议基线，比较当前 LightGCN 是否确有优势。
2. 独立复现一个公开全排序协议（如 TPAB 的预处理与 Vanilla LightGCN），对齐代码、标签、划分和候选集合后核查成绩；使用单独数据/运行目录，保留当前协议。
3. 明确比较和选择规则后再做冻结 test，不根据 test 结果反复调参。即使总体指标提高，也需同时交代稀疏用户和长尾结果。

这些是后续验证建议，本次没有启动新增训练、切换当前协议或执行 test。
