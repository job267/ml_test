"""Train-only ItemCF with the unchanged stage-one full-ranking evaluator."""
import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch
from shared import ROOT, digest, gpu_guard, read, save, utc
from experiment import code_hashes
from research_data import FeedbackDataset
from research_evaluation import evaluate


def cosine_neighbors(train, weighting, max_neighbors):
    """S[target, history]; weighted cosine, no self edges; stable neighbor ties."""
    x = train.astype(np.float64).tocsr(copy=True)
    x.sum_duplicates()
    if len(x.data) and not np.all(x.data == 1):
        raise ValueError("ItemCF requires binary training positives")
    degree = np.diff(x.indptr)
    if weighting == "cosine":
        weights = np.ones(x.shape[0])
    elif weighting == "iuf":
        weights = np.divide(1., np.log1p(degree), out=np.zeros(len(degree)), where=degree > 0)
    else:
        raise ValueError(weighting)
    gram = (x.T @ x.multiply(weights[:, None])).tocsr()
    norms = np.sqrt(gram.diagonal())
    gram.setdiag(0)
    gram.eliminate_zeros()
    indices, values = [], []
    for item in range(x.shape[1]):
        start, end = gram.indptr[item:item + 2]
        ids, counts = gram.indices[start:end], gram.data[start:end]
        denom = norms[item] * norms[ids]
        sims = np.divide(counts, denom, out=np.zeros(len(counts)), where=denom > 0)
        # Sort every nonzero row once. Item ID breaks equal similarity ties.
        order = np.lexsort((ids, -sims))[:max_neighbors]
        indices.append(ids[order].copy())
        values.append(sims[order].astype(np.float32))
    return indices, values


def similarity_matrix(neighbors, k, n_items):
    indices, values = neighbors
    lengths = np.array([min(k, len(i)) for i in indices])
    rows = np.repeat(np.arange(n_items), lengths)
    cols = np.concatenate([i[:k] for i in indices])
    vals = np.concatenate([v[:k] for v in values])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))


class UserRows:
    """Materialize only the evaluator's current user batch, on its CUDA device."""
    def __init__(self, train, device):
        self.train, self.device = train, torch.device(device)

    def __getitem__(self, ids):
        return torch.as_tensor(self.train[ids.cpu().numpy()].toarray(), dtype=torch.float32, device=self.device)


class ItemCF:
    def __init__(self, train, similarity, device):
        self.users = UserRows(train, device)
        coo = similarity.tocoo()
        self.items = torch.sparse_coo_tensor(
            np.stack([coo.row, coo.col]), coo.data, coo.shape, dtype=torch.float32, device=device).coalesce()

    def eval(self):
        return self

    def computer(self):
        # The original evaluator computes X_batch @ S.T without any protocol change.
        return self.users, self.items


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=str)
    args = p.parse_args()
    from pathlib import Path
    output = Path(args.output)
    gpu = gpu_guard()
    if (output / "completion.json").exists():
        raise FileExistsError("Completed ItemCF study is immutable")
    output.mkdir(parents=True, exist_ok=True)
    ds = FeedbackDataset(ROOT / "data/kuairand_stage1", "cuda")
    save(output / "metadata.json", {"created_utc": utc(), "environment": gpu,
         "dataset_sha256": ds.fingerprint, "code_sha256": code_hashes(),
         "driver_sha256": digest(__file__), "grid": {"weighting": ["cosine", "iuf"], "neighbors": [20, 50, 100, 200]},
         "fit_data": "binary train positives only", "selection": "maximum overall validation NDCG@20; same configuration for every group",
         "neighbor_ties": "ascending item ID", "ranking_ties": "original evaluator torch.topk",
         "test_evaluated": False})
    rows = []
    for weighting in ("cosine", "iuf"):
        start = time.perf_counter()
        neighbors = cosine_neighbors(ds.UserItemNet, weighting, 200)
        for k in (20, 50, 100, 200):
            name = f"{weighting}-k{k}"
            sim = similarity_matrix(neighbors, k, ds.m_items)
            result = evaluate(ds, ItemCF(ds.UserItemNet, sim, "cuda"), "val", [20], 100)
            row = {"variant": name, "weighting": weighting, "neighbors": k, "similarity_nnz": sim.nnz,
                   "validation": result, "elapsed_seconds": time.perf_counter() - start}
            sp.save_npz(output / f"{name}.npz", sim)
            rows.append(row)
            save(output / "validation_grid.json", {"rows": rows, "test_evaluated": False})
            print(json_string(row), flush=True)
    best = max(rows, key=lambda row: row["validation"]["groups"]["overall"]["ndcg"][0])
    save(output / "completion.json", {"completed_utc": utc(), "selected": best,
         "grid_size": len(rows), "selected_similarity_sha256": digest(output / f"{best['variant']}.npz"),
         "test_evaluated": False})


def json_string(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    main()
