# 第一阶段代码阅读指南

本指南帮助阅读和继续修改新增实验代码。原始六个基线模块保留；研究功能通过单独入口调用原模型。

## 1. 按运行顺序阅读

| 文件 | 职责 | 建议先看 |
|---|---|---|
| `code/prepare_kuairand.py` | 原始日志去重、反馈解释、时间切分与准备产物 | `prepare()`，正向/负向信号规则 |
| `code/research_data.py` | 校验产物、加载训练图与负证据 | `FeedbackDataset`、`is_positive()`、`negative_confidence()` |
| `code/research_sampling.py` | 合法候选生成、困难/可靠选择与课程比例 | `FeedbackSampler.rate()`、`select()`、`sample()` |
| `code/experiment.py` | 单次训练、元数据、检查点和恢复 | `make_parser()`、`build_model()`、`train_one_epoch()`、`run()` |
| `code/research_evaluation.py` | 验证/测试的全物品排序与分组统计 | `evaluate()` |
| `scripts/run_stage1.py` | 按方法和种子顺序调度训练 | `command()`、`main()` |
| `scripts/evaluate_stage1.py` | 固定检查点选择后评价整个套件 | `main()` |
| `scripts/summarize_stage1.py` | 均值、标准差、配对差异和 CSV | `summarize()` |
| `scripts/package_project.py` | 打包可上传源码与文档 | `package()` |
| `code/research_common.py` | 文件指纹与 JSON 读写 | 三个小工具函数 |

## 2. 数据流

```text
官方 log_*.csv
  → prepare_kuairand
  → 固定训练/验证/测试文件 + 训练证据 + manifest
  → FeedbackDataset（校验后加载）
  → FeedbackSampler（生成 triplets）
  → 原 LightGCN/MF + 原 BPRLoss（逐 batch 更新）
  → validation（训练中）
  → last.pt / best_val.pt
  → 独立 final test
  → summary.csv / summary.json
```

## 3. 容易误解的命名与行为

- `positive`：按选定反馈协议认可的正向信号，不是宣称真实偏好被完全观测。
- `confidence`：启发式负证据权重，不是概率校准结果。
- `targeted_rate`：使用困难/可靠选择的 triplets 比例；剩余样本均匀随机。
- `evidence_fraction`：定向候选池中来自训练负证据池的比例，不是损失权重。
- `snapshot`：仅用于候选评分的停止梯度表示；BPR 训练表示仍逐 batch 重算。
- `val_known`：验证时期已经发生的全部认可正向交互，仅用于最终测试屏蔽，不进入训练图或采样证据。
- `last` 与 `best_val`：固定轮数最终模型与验证选模两种协议；正式 Test 前统一选择。

## 4. 修改时如何判断影响范围

修改正向规则、时间切分、词表或证据定义：生成新的准备目录，所有对照重跑。

修改候选分布、证据权重或课程：使用新配置与运行目录，只在 validation 比较。

修改模型数学、残差通道或图连接：作为第二阶段独立实验，保留 A0 与单模块/组合对照。

修改普通格式和函数组织：运行现有回归测试；检查点恢复会检测源码变动，因此继续旧实验仍应使用旧源码包。

## 5. 测试在证明什么

`tests/test_research.py` 检查时间隔离、未来词表过滤、图归一化、重复事件、数据指纹、负样本合法性、证据选择、停止梯度、验证屏蔽、CPU 恢复一致性和四组套件完整流程。

`tests/test_smoke.py` 保留原基线基础测试。测试通过不代表真实数据上推荐效果已提升，也不代表在线反馈策略已实现。
