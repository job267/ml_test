# LightGCN — PyTorch 2.x baseline and stage-one feedback experiments

Clean compatibility-oriented reconstruction of [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch), based on **LightGCN (SIGIR 2020)**. The legacy entrypoint keeps MF/LightGCN, normalized user–item bipartite adjacency, full graph propagation, mean pooling, one-negative BPR optimization, and Recall/Precision/NDCG @ K.

Stage-one research now has a **separate** `code/experiment.py` entrypoint for temporal KuaiRand preparation, uniform/hard/reliable/curriculum samplers, validation-only training, isolated checkpoints with resume, and final test suites. These are experimental offline methods, not an implemented online reinforcement-learning system. Legacy model and training code remain unchanged.

- [最新研究状态与可核查结果（2026-10-17 快照）](analysis/research_status_20261017/README.md)
- [当前阶段的研究推进策略](docs/research_execution_strategy_20261017.md)
- [第一阶段实现与实验协议](docs/stage1_implementation.md)
- [从上传云服务器到开始实验的完整步骤](docs/cloud_stage1_runbook.md)
- [算法研究路线](docs/algorithm_research_roadmap.md)
- [代码阅读指南](docs/code_structure.md) · [本地验证记录与限制](docs/stage1_validation.md)
- [使用 GitHub 管理服务器代码](docs/github_server_workflow.md)
- [第一轮正式实验的验证分析与后续方案](analysis/stage1_validation_20261015/README.md)
- [第二轮 beta 消融、验证参照与 GPU 0 运行记录](analysis/stage1_beta_pilot_20261015/README.md)
- [两新种子的 30 轮复核与配对比较](analysis/stage1_beta_replication_20261016/README.md)
- [降低定向比例：三种子消融](analysis/stage1_targeted_rate_20261016/README.md)
- [候选池来源消融：纯随机候选的困难选择](analysis/stage1_candidate_source_20261016/README.md)
- [MF、ItemCF 与公开 LightGCN 基线校准](analysis/baseline_calibration_20261016/README.md)
- [同数据实现核对与标签／曝光来源诊断](analysis/protocol_alignment_20261017/README.md)
- [准备固定版本的公开参考文件](references/tpab_7753efd/README.md)

## Setup (WSL2 / Linux, Python 3.10)

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python -c 'import torch; print(torch.__version__,torch.cuda.is_available())'
```

For CPU-only systems install the corresponding official CPU PyTorch build instead. CUDA builds require a compatible NVIDIA driver.

## Data

Download the data from the upstream project and copy its `data/{gowalla,yelp2018,amazon-book,lastfm}` directories here; this archive deliberately does not duplicate the upstream datasets. `train.txt` / `test.txt` use one line per user: `USER_ID ITEM_ID ITEM_ID ...` (zero-based). LastFM uses upstream `data1.txt`, `test1.txt` (one-based `user item` plus possible extra column); this loader converts the first two columns automatically, writing normalized files alongside the originals.

## Smoke test

```bash
python -m pytest -q
python code/main.py --dataset lastfm --epochs 2 --tensorboard 0 --testbatch 100
```

## Gowalla reproduction configuration

```bash
python code/main.py --dataset gowalla --model lgn --layer 3 --recdim 64 --lr 0.001 --decay 0.0001 --epochs 1000 --seed 2020 --topks '[20]' --tensorboard 1
```

Checkpoints are stored in `checkpoints/` and can be loaded using `--load 1` (optimizer state is not restored). `--pretrain 1` explicitly errors because the upstream path assumes externally injected numpy arrays. GPU is auto-selected, or set `--device cpu`. The C++ extension has been removed in favor of the upstream Python sampling behavior for portability, which can be slower on large datasets.

## Migration notes

- Removed incompatible hard pins of torch 1.4, pandas 0.24 and scipy 1.3.
- Replaced deprecated sparse tensor constructors with `torch.sparse_coo_tensor`.
- Constructed sparse block adjacency with SciPy `bmat`, and normalized using degree matrices.
- Removed `eval` for top-K parsing (uses `ast.literal_eval`).
- Computes graph embeddings only once per evaluation pass rather than once per user batch.
- Preserved per-mini-batch graph propagation **during training** so gradient semantics match the original.
- Uses portable Python BPR negative sampling; no pybind11/compiler required.
- Adds repeatable miniature unit/smoke tests and explicit checkpoint loading.

This is a reimplementation of the algorithm and workflow, **not** a byte-for-byte mirror of upstream code. The local handover records one successful Gowalla reproduction; its data, weights and logs are not included in this checkout. Current KuaiRand validation findings, incomplete public reproduction and remaining limitations are recorded in the dated research snapshot above; they do not establish final test performance or a confirmed new algorithm improvement. Upstream repo is archived and does not declare a license in its root; check distribution rights before copying any additional upstream files.
