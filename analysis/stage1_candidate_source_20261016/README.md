# 候选池来源消融：纯随机候选的困难选择

前轮降低全局定向比例没有同时保住整体收益与修复稀疏用户退化。本轮固定 beta=0、80% 定向，单独将 evidence_fraction 从 0.5 改为 0，检查候选提议来源的作用。

## 实验设计

- 新训练三组：seed=2020、2021、2022，各 30 轮、每轮完整 validation。
- 固定三层 LightGCN、64 维、学习率 0.001、正则 0.0001、每轮 451,741 个训练样本、batch=2048、候选数 20、定向比例 0.8、beta=0。
- `evidence0`：定向候选全部从合法未交互物品随机提出，再按模型分数 argmax 选困难负例。
- `evidence05`：已有 50% 证据候选的 beta=0、80% 定向对照，使用相同种子、源码与数据的前 30 轮日志。
- `uniform`：已有随机负例对照，也限制为前 30 轮。
- evidence_fraction=0 仅取消主动从训练证据池提出候选；随机候选仍可能碰到已有训练弱证据或明确拒绝的物品，并非人为排除这些物品。
- 每组按整体 validation NDCG@20 选最佳模型；Recall、稀疏用户、长尾指标及采样诊断取同一轮。
- 汇总逐种子差异、三种子均值/样本标准差、整体与稀疏用户指标的取舍，以及负例重复、弱证据占比。最佳点落在最后五轮则标记统一延长预算检查。

三个种子和 validation 已参与过配置探索，本轮仍属探索性消融，不宣称独立确认、统计显著或测试收益，不执行 test。候选来源变化与困难选择可能相互作用；单个整体指标不能独立证明所有误差的原因。

## 执行与产物

沿用原 `code/experiment.py` 和已有 pilot 驱动，没有修改 `code/`、`scripts/`，旧检查点依赖的 16 个源码 SHA256 保持一致。

使用现成 `gcn` 环境，仅在物理 GPU 0 顺序执行；固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=0`，启动时核对 PyTorch 可见设备 UUID 与 `nvidia-smi -i 0` 一致。各组持续记录 GPU 0 负载；缓存参考结果不参与耗时效率比较。

配置：`configs/stage1_candidate_source_20261016.json`。

运行目录：`runs/stage1_candidate_source_20261016/`。

- `status.json`：进程、当前 seed 与完成数量。
- `comparison.json`、`comparison.csv`、`comparison.md`：自动汇总；新配置未完成三个种子前不显示其三种子均值。
- `plan.json`：配置、源码、驱动、参考运行、数据清单指纹及 GPU 身份。
- `seed2020/` 等：完整训练日志、last / best 检查点、validation 指标、候选分数诊断及 GPU 0 负载。
- 外层日志：`runs/launch_logs/stage1_candidate_source_20261016.log`。

启动命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 conda run --no-capture-output -n gcn python -u analysis/stage1_candidate_source_20261016/run_sources.py
```

中断后确认进程停止，保持同版配置、源码、数据、参考文件及 GPU 身份，追加 `--resume` 恢复。默认沙箱不暴露 CUDA，实际训练通过已授权的沙箱外执行。

启动前 30 项检查通过，包括单因素配置、零证据比例不读取候选证据池，以及逐种子配对汇总检查。六份原参考运行的完整训练配置和所有原训练源码指纹核对一致。

## 实际启动核验

北京时间 2026-10-16 18:45:15 启动，调度 PID=2415477，GPU 0 UUID=`GPU-341a3faf-c619-5dcd-4bc3-2019cb13e921`。seed=2020 已完成完整第 1 轮：实际 evidence_fraction=0、beta=0、targeted_rate=0.8，451,741 个训练样本、361476 个定向样本、7229520 次候选评分，validation NDCG@20=0.004064190，last.pt 与 best_val.pt 均已保存。这仅为启动核验，不是最终实验结论。
