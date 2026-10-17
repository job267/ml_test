# 降低定向采样比例：三种子消融

本轮依据上轮结果：beta=0 的困难选择对照在整体 NDCG 上有重复收益，但稀疏用户退化；固定证据加权未显示额外稳定收益。先只降低定向采样比例，验证收益与分组损失之间的关系。

## 固定实验设计

- 新增定向比例 0.2、0.4，各跑 seed=2020、2021、2022，共六组、各 30 轮，每轮完整 validation。
- beta 固定为 0，候选数 20、证据候选比例 0.5、temperature=1；模型、学习率、正则、训练样本数等保持上轮设置。
- 调用未改动的 `code/experiment.py` 和已有 pilot 调度器，新工具仅负责参数组合、版本核验和汇总。
- 已有 uniform 与 0.8 定向对照从原日志取相同种子的第 1–30 轮，不重跑已完成的同配置。旧产物保持原样。
- 每组按整体 validation NDCG@20 选最佳轮次；Recall、稀疏用户和长尾指标取同一轮。分别报告整体与稀疏用户指标，不用混合总分遮盖退化。
- 对新配置给出逐种子相对 uniform 和 0.8 定向的变化、三种子均值/样本标准差，以及整体 NDCG / 稀疏用户 NDCG 的 Pareto 取舍。
- 已复用这些种子与验证集进行过配置探索，因此这轮仍属探索性消融；不宣称独立确认、显著性或测试收益，不执行 test。
- 记录最佳点是否落在最后五轮，供统一延长预算检查。

本轮仍不能分开证明证据候选池与困难选择各自的贡献。`evidence_fraction=0` 的候选来源消融保留为独立后续实验，避免同时改两个因素。

## 执行与产物

使用现成 `gcn` 环境，固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=0`。启动核验 PyTorch 唯一可见设备的 UUID 与 `nvidia-smi -i 0` 一致。GPU 0 负载记录保存在各组的 `gpu0.jsonl`。

配置：`configs/stage1_targeted_rate_20261016.json`。

运行目录：`runs/stage1_targeted_rate_20261016/`。

- `status.json`：当前任务、进程与完成数量。
- `comparison.json`、`comparison.csv`、`comparison.md`：自动更新的比较结果；配置未跑满三个种子时，不提前报告该配置的三种子均值。
- `plan.json`：训练源码、调度源码、配置、原数据及参考日志指纹、GPU 身份。
- `jobs/rate02-seed2020/` 等：每组完整训练日志、last / best 检查点、验证指标及候选分数诊断。
- `runs/launch_logs/stage1_targeted_rate_20261016.log`：外层启动日志。

第一、三个种子依次运行 0.2→0.4，第二个种子反向运行 0.4→0.2，减少所有种子都固定同一执行顺序的关联。耗时不与缓存对照做算法效率比较。

启动：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 conda run --no-capture-output -n gcn python -u analysis/stage1_targeted_rate_20261016/run_rates.py
```

队列中断并确认进程停止后，保持同版源码、配置、数据、参考文件与 GPU 身份，追加 `--resume` 恢复。默认沙箱不可见 CUDA，实际训练使用已授权的沙箱外执行。

启动前 27 项检查通过，包括原测试、前两轮分析检查，以及本轮的单因素配置、逐种子配对和分组取舍检查。六份参考运行的完整训练配置和 16 个原源码 SHA256 已核对一致。

## 实际启动核验

北京时间 2026-10-16 10:14:52 启动，外层队列 PID=279272，GPU 0 UUID=`GPU-422501a2-5189-7f5b-5ce6-95464276b27f`。首组 rate=0.2、seed=2020 完成完整第 1 轮，451,741 个样本，实际定向样本 90533，候选打分数 1810660，validation NDCG@20=0.004010988，检查点已保存。此项仅为正式启动核验，不代表最终收益。
