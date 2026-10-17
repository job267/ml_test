# 2026-10-16：两个新种子的复核

本轮固定 uniform、beta=0、beta=0.25 三种配置，补 seed=2021、2022，各 30 轮、每轮 validation。训练沿用原 `code/experiment.py`，现有 `gcn` 环境和物理 GPU 0；`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=0`，启动时核对 CUDA 与 nvidia-smi 的 UUID 一致。

## 比较规则

- 新训练共六组、180 个完整 epoch；每轮 451,741 个训练样本，模型、学习率、正则、候选来源和定向比例保持原值。
- 最佳模型按第 1–30 轮 validation NDCG@20 选择；Recall、稀疏用户和长尾指标取同一轮，完整保留各分组。
- 已有 seed=2020 仅使用日志中的第 1–30 轮，匹配训练预算。它参与过 beta 的选择，三种子汇总单独标为描述性结果；新种子 2021/2022 的均值、样本标准差和配对差异另报。
- 原候选规则下 beta=0 是 hard 选择对照。分别比较 beta=0 对 uniform、beta=0.25 对 uniform，以及 beta=0.25 对 beta=0。
- 检查最佳轮次是否落在 26–30 轮，输出 `extension_review`。如出现末段改善，应评估统一延长预算；不混合 30 轮与 100 轮的择优窗口。
- 本轮不执行 test，不改动旧套件或旧检查点，不把单种子峰值或两个新种子的均值称为显著提升。

## 产物

总目录：`runs/stage1_beta_replication_20261016/`。

- `status.json`：调度状态、进程、当前 seed 和完成数量。
- `comparison.json` / `comparison.csv`：每组最佳模型、分组指标、长尾命中、负例集中度、两新种子及三种子描述性汇总。
- `replication_plan.json`：配置、数据、原训练源码、调度工具及参考日志指纹和 GPU UUID。
- `seed2021/`、`seed2022/`：沿用已验证的 pilot 驱动，保存每组 last / best 检查点、完整训练日志、validation 指标、候选分数诊断和 GPU 0 负载记录。
- `runs/launch_logs/stage1_beta_replication_20261016.log`：外层启动日志。

GPU 0 当前 UUID 与上一轮记录不同，工具记录本轮实际值，并在启动时核对设备身份。设备仍为 RTX A6000；耗时比较应参考负载记录。

## 启动与恢复

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 conda run --no-capture-output -n gcn python -u analysis/stage1_beta_replication_20261016/replicate.py
```

进程中断且已停止后，用相同源码、配置、数据和 GPU 身份恢复：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 conda run --no-capture-output -n gcn python -u analysis/stage1_beta_replication_20261016/replicate.py --resume
```

默认沙箱看不到 CUDA，实际训练需要在已授权的沙箱外执行。运行期间保留训练、pilot 和复核驱动源码不变。

启动前原 20 项测试、上一轮 2 项诊断检查和本轮 2 项共同预算/分种子汇总检查通过，共 24 项。

## 已核验的实际启动

北京时间 2026-10-16 08:42:02 启动，外层调度 PID=4191377，GPU 0 UUID=`GPU-422501a2-5189-7f5b-5ce6-95464276b27f`。首组 uniform / seed=2021 已完成完整第 1 轮：451,741 个样本，validation NDCG@20=0.004032089，epoch 约 6.19 秒，last.pt 与 best_val.pt 已保存。仅为启动核验，不代表最终收益。
