# MF/BPR、ItemCF 与公开 LightGCN 校准

仅使用现有 conda `gcn` 与物理 GPU 0。所有 GPU 子任务串行运行，启动时核验 CUDA UUID 与 `nvidia-smi -i 0` 一致。当前 `code/*.py`、`scripts/*.py` 保持原字节，已有实验目录保持不变。

## 当前协议

- 保留 KuaiRand-Pure engagement 正反馈、固定日期划分、随机曝光新物品目标、全物品排名和原分组定义。prepared manifest SHA256 为 `99601459915dbdde342261197d544d6d1186ad4037b8ed80dc601c9ea2e5870f`。
- MF/BPR 使用原 `LightGCN` 的 `layer=0`：与 `layer=3` 一致的 N(0, 0.1) 初始化、BPR、L2、uniform 采样、64 维、Adam lr=0.001、decay=0.0001、batch=2048。测试验证了它与复制相同权重后的 `PureMF` 的目标和梯度等价。原生 `PureMF` 默认初始化 std=1，故本实验明确标记 matched initialization。
- 3 个种子 2020/2021/2022，分别报告共同前 30、前 100 轮的验证集最佳 NDCG@20。Recall 和所有分组固定取该轮，不能独立选取分组最优轮次。
- LightGCN seed2020 复用原 100 轮逐轮验证记录；seed2021/2022 的原 30 轮目录完整复制后续训至 100，含优化器、采样器和 RNG 状态。新增输出位于 `runs/baseline_calibration_20261016/stage1/`，不改变原目录。
- ItemCF 一次确定性拟合：二值训练正样本构建目标物品与历史物品的余弦相似度；另做 IUF 加权余弦，用户权重 `1/log(1+degree)`，分母使用同一加权内积范数。去除自相似，邻居数 20/50/100/200，共 8 组。相似度相同按 item ID 选择邻居；排名复用原 `research_evaluation.evaluate` 和 `torch.topk`。评分 `X_batch @ S.T`；不使用 held-out 标签构建相似度。
- ItemCF 按总体验证 NDCG 选择一组，展示所有网格结果。它是一次确定性配置搜索，不伪造三个种子的方差，也不宣称与单一配置的搜索预算相等。当前划分测试集不评估。

## 公开协议

来源：[KDD 2025 TPAB 作者仓库](https://github.com/hsyoo32/tpab_)、[作者论文](https://q-rz.github.io/static/kdd25/kdd25-tpab-paper.pdf)。固定 commit `7753efd0a8cf3aaed9b150a3d835dbaa8fe0e50d`（2025-07-28）。`external/tpab_7753efd/` 内 22 个代码/数据文件逐一通过 Git blob SHA1 与字节大小校验，再记录 SHA256。最初一次原始域名下载遇到 SSL EOF；缺少的 `model.py` 经 GitHub connector 读取，仍恢复并校验了完全一致的原文件字节。

作者数据：22,128 用户、7,076 物品；train/val/test 496,142/63,291/61,631 对，共 621,064；跨划分无重复对，时间顺序一致。作者图缓存与独立从训练对重建的归一化图最大差约 2.98e-8。完整审计由队列保存在 `public/protocol_audit.json`。

运行作者原样 vanilla LightGCN 模型、数据加载、uniform 采样、训练损失及验证评测。模型 64 维、3 层、lr=0.001、decay=0.0001、batch=2048；5 个种子 2018–2022，每 5 轮验证，最多 600 轮，保留作者原始 patience 逻辑。作者默认按四舍五入后的 Recall@10 严格提升选模型；并非按 NDCG@20 选。训练脚本示例 `_tester.py` 的 decay=1e-3 与论文文字不一致，此校准预先固定论文文字及 parse 默认的 1e-4，不按测试成绩再选择。

公开实现的已确认问题：`getUserValidItems` 丢弃没有验证记录的用户，`Procedure.Test` 在测试时直接枚举压缩后的列表，导致验证已知物品与用户行错位。作者数据中有 **3,507 名测试用户不在验证用户集合**。因此在训练结束后，对同一验证选定模型只做一次测试打分，同时生成作者原样屏蔽与修正用户对齐屏蔽两套预先声明的指标。两套指标都不能影响模型或配置选择。测试已验证原样路径与固定版本的作者 `Procedure.Test` 结果一致。

与作者 main 的有意差异：加入原子检查点、优化器/RNG 续训、结构化记录；测试从每次验证提升时延后至停止后；省略无用 batch 列表与训练日志 loss 的计算图保留；禁用 tensorboard 时不导入 tensorboardX。沿用用户 gcn 的 Python 3.10/PyTorch 2.6/CUDA 12.6，作者 README 使用 Python 3.7/PyTorch 1.13，因此不声称逐位复现。原始作者文件完全不修改。

论文表 2 Vanilla LightGCN 的 KuaiRand Recall@20=0.1281、NDCG@20=0.0672。校准完成前不能宣称达到这些分数；它们也不能与当前 engagement/random-exposure 协议直接比较。

## 队列与恢复

配置：`configs/baseline_calibration_20261016.json`。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  conda run --no-capture-output -n gcn python -u analysis/baseline_calibration_20261016/run_study.py
```

顺序：ItemCF 8 组 → 公开 seed2018 前 5 轮兼容性检查（正式轨迹，保存状态、不测试）→ MF 3 个种子先各 30 轮 → MF 和 LightGCN 补足共同 100 轮 → 公开 seed2018 从第 5 轮续训 → 其余 4 个公开种子。公开阶段保留首段的全部优化器和 RNG 状态，前 5 轮不构成额外选参实验。

输出：`runs/baseline_calibration_20261016/status.json`、`summary.md`、`comparison.json`、逐任务 `logs/*.log`。每 30 秒更新汇总，完成后自动生成 `completion.json`。没有新增定时通知或后台监控服务。

恢复使用相同命令追加 `--resume`；所有脚本、配置、当前训练源码、原参考检查点、数据和公开审计指纹必须一致。不要在队列运行时改动本目录的 Python 文件。发生错误时只终止本队列自己的子进程组，不结束其他任务。

验证：原项目全部测试和新增 MF 梯度/初始化、ItemCF 数学公式/训练限定/排名屏蔽测试已运行；另验证公开早停边界、优化器/RNG 断点恢复、单次评分两种屏蔽及作者原样评测等价性；GPU 0 小矩阵稀疏评分检查通过。具体最新通过数量与运行状态另记启动记录。
