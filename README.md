# LightGCN — PyTorch 2.x compatibility baseline

Clean compatibility-oriented reconstruction of [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch), based on **LightGCN (SIGIR 2020)**. Keeps MF/LightGCN, normalized user–item bipartite adjacency, full graph propagation, mean pooling, one-negative BPR optimization, and Recall/Precision/NDCG @ K. No novel recommendation methods added.

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

This is a reimplementation of the algorithm and workflow, **not** a byte-for-byte mirror of upstream code. Reproduction of upstream published metrics is not yet established; benchmark runs on Gowalla/Yelp2018 must be completed separately. Upstream repo is archived and does not declare a license in its root; check distribution rights before copying any additional upstream files.
