# 第二轮：证据权重与早期验证诊断

日期：2026-10-15（北京时间）。本轮继续 LightGCN / KuaiRand-Pure 第一阶段研究。训练只使用现有 `gcn` conda 环境、物理 GPU 0（RTX A6000）。

## 代码版本核对

核对 GitHub `job267/ml_test` 的 main 提交 `212bfb35147ef913c40fb44c390672a6e666ae77`。

- `code/*.py` 的 12 个文件、`scripts/*.py` 的 4 个文件、`configs/stage1.json` 和依赖清单逐字节一致。
- `tests/test_research.py` 只有 CRLF / LF 换行区别，另一个测试文件完全一致。
- 差异主要是 README、去重修复说明、GitHub 工作流程、验证记录和新增的第一轮分析工具。本次已补充相关文档和分析源码。
- 本地还有 GitHub 刻意忽略的数据、12 组完整检查点与训练记录、环境和归档。本目录是上传的源码目录，未包含可用的 Git 元数据；通过 Git blob SHA1 和逐文件 SHA256 核对，而非假定存在共同 Git HEAD。
- 原始比较清单保存在 `analysis/upstream_20261015/comparison.json`，有差异的远端文档快照位于该目录。

所有新 Python 工具放在 `analysis/`，没有修改或增加 `code/`、`scripts/` 文件。旧检查点依赖的 16 个源码指纹完整保留。新运行目录也保存原源码副本及对应指纹。

## 已完成的诊断

使用原数据清单及原训练源码校验后，在 validation 上计算参照：

| 方法 | NDCG@20 | Recall@20 |
|---|---:|---:|
| 均匀随机排序的精确期望 | 0.0012328483 | 0.0026450879 |
| 按唯一训练正向度排序 | 0.0040489102 | 0.0086562977 |
| 原 uniform，三个种子最佳验证模型均值 | 0.0044256024 | 0.0099114513 |

参照结果：`runs/stage1_v1/validation_references.json`。模型与热门参照的差距有限，不能仅凭高于随机就断言方法有效。原 uniform 的第 1000 轮验证 NDCG 均值为 0.0030644157，低于热门参照。

数据包括 24,524 名训练用户、7,583 个物品、451,741 个训练正交互，validation 为 10,005 名用户、30,593 个目标。弱证据 839,652 对，明确负证据仅 680 对、覆盖 343 名用户。29 个 validation 目标曾有训练弱证据，该信息仅用于审计，不用于反向删除训练候选。

从原归档重新运行第一轮分析，产物位于 `analysis/stage1_validation_20261015/results_recheck/`。核对到 12,000 条训练记录及 1,200 次验证。原方法多在前 10–40 轮达到验证峰值，第一轮每 10 轮验证，未观察第 1–9 轮。

固定初始化、seed=2020 的 8,192 行候选诊断中，原始分数标准差约 0.005635。beta=0.05 时相对 beta=0 的选中物品改变比例为 57.86%；该初始候选池中 beta=0.05、0.1、0.25、1 的选择相同，弱反馈约占定向选择的 98.54%。这证明初始化阶段干预较强，但不证明整段训练中的行为相同，也不证明推荐收益下降的唯一原因。8,192 行诊断的不重复比例不能直接与 451,741 条完整 epoch 比较。

## 本轮配置与选择规则

配置：`configs/stage1_beta_pilot.json`。顺序为 uniform、reliable beta=0 / 0.05 / 0.1 / 0.25 / 1，各 seed=2020、100 轮，每轮 validation。

保持原 LightGCN 三层、64 维、学习率 0.001、正则 0.0001、batch=2048、每轮 451,741 个样本。定向组候选数 20、证据候选比例 0.5、定向比例 0.8、temperature=1。beta=0 是同候选来源下的 hard 选择对照，不代表全随机候选的 hard 采样。

本轮仅做单种子诊断；按 validation NDCG@20 选择最佳模型，Recall 和分组指标使用同一轮。验证集寻优成绩不是最终测试收益。发现有希望的配置后需要用 2021、2022 复核，不能用单种子宣称稳定提升。本轮不修改旧套件最终测试选择，也不执行 test。

新工具保留逐轮 last / best_val 检查点，记录源码、数据、配置和驱动工具指纹，使用独占文件锁避免重复启动同一队列。每 30 秒记录 GPU 0 的 UUID、显存与利用率，并更新状态和 validation 汇总；失败时停止后续组。

初始化及每组最佳模型保存训练候选分数诊断，包括分数分布、证据池大小、相对 hard 的选择改变、弱/明确/未知比例和用户—负物品对重复程度。诊断使用独立进程，不消耗训练采样器或优化器的随机状态。

## 运行与查看

GPU 在本会话的默认沙箱里不可见。已通过沙箱外只读检查及 CUDA 运算确认 GPU 0 可用，无需重建环境或重装驱动。

启动：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 conda run --no-capture-output -n gcn python -u analysis/stage1_beta_pilot_20261015/run_pilot.py
```

查看：

```bash
cat runs/stage1_beta_pilot_20261015/status.json
cat runs/stage1_beta_pilot_20261015/validation_summary.csv
tail -n 1 runs/stage1_beta_pilot_20261015/uniform-seed2020/metrics.jsonl
```

进程结束或明确停止后，同版源码和配置下恢复：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 conda run --no-capture-output -n gcn python -u analysis/stage1_beta_pilot_20261015/run_pilot.py --resume
```

运行期间保留同版源码和配置。原第一轮产物保留在 `runs/stage1_v1/`。小预算链路检查使用单独的 `runs/stage1_beta_pilot_smoke_20261015/`，每组仅 512 个样本、一轮，不纳入正式效果比较。

启动前 20 项原测试及 2 项新诊断/汇总检查通过。正式进度以 `status.json` 和各组 `metrics.jsonl` 为准，本说明不把启动记录当成已完成的训练成绩。

## 已核验的启动记录

六组小预算检查于北京时间 19:29:43 全部完成，每组生成 completion、last.pt、best_val.pt、完整 validation 以及候选分数诊断，均未评价 test。

正式后台队列于北京时间 19:30:11 启动，调度进程 PID=2878206，外层 conda 启动进程 PID=2878166。启动日志：`runs/launch_logs/stage1_beta_pilot_20261015.log`。实际 GPU UUID 为 `GPU-341a3faf-c619-5dcd-4bc3-2019cb13e921`；固定 `CUDA_VISIBLE_DEVICES=0`，PyTorch 可见设备数为 1。

第一组 uniform 的完整第 1 轮已核验：451,741 个训练样本，loss=0.559900，validation NDCG@20=0.004058971、Recall@20=0.008519329，epoch 约 23.31 秒；last.pt 和 best_val.pt 均已保存。这仅证明正式预算训练已正常运行，不代表本轮最终结果。
