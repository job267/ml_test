"""Audit the pinned author split/cache without importing executable author modules."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from shared import ROOT, digest, read, save

VENDOR = ROOT / "external/tpab_7753efd"


def audit(output):
    provenance = read(VENDOR / "provenance.json")
    if provenance["commit"] != "7753efd0a8cf3aaed9b150a3d835dbaa8fe0e50d":
        raise ValueError("Unexpected author revision")
    for record in provenance["files"]:
        if digest(VENDOR / record["path"]) != record["sha256"]:
            raise ValueError(f"Author file changed: {record['path']}")
    path = VENDOR / "kuairand"
    # These pickles come only from the explicitly pinned and byte-verified author repository.
    ds = {name: np.load(path / f"{name}_dict.npy", allow_pickle=True).item()
          for name in ("training", "validation", "testing")}
    pairs = {name: {(int(u), int(i)) for u, items in d.items() for i in items} for name, d in ds.items()}
    n_users = 1 + max(u for p in pairs.values() for u, _ in p)
    n_items = 1 + max(i for p in pairs.values() for _, i in p)
    if n_users != 22128 or n_items != 7076:
        raise ValueError("Author vocabulary differs from the paper")
    rows, cols = [], []
    for u, items in ds["training"].items():
        rows.extend([int(u)] * len(items))
        cols.extend(items)
    x = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items))
    a = sp.bmat([[None, x], [x.T, None]], format="csr", dtype=np.float32)
    degree = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1., np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    expected = (sp.diags(inv) @ a @ sp.diags(inv)).tocsr()
    cached = sp.load_npz(path / "s_pre_adj_mat.npz")
    if cached.shape != expected.shape:
        raise ValueError("Author graph cache has the wrong dimensions")
    diff = cached - expected
    max_diff = float(np.max(np.abs(diff.data), initial=0))
    if max_diff > 1e-6 or cached.nnz != expected.nnz:
        raise ValueError(f"Author graph cache is not train-only: {max_diff}")
    times = np.load(path / "interaction_time_dict.npy", allow_pickle=True).item()
    stats = {}
    for name, d in ds.items():
        t = [int(times[u][i]) for u, items in d.items() for i in items]
        stats[name] = {"users": len(d), "nonempty_users": sum(bool(len(v)) for v in d.values()),
                       "interactions": sum(map(len, d.values())), "unique_pairs": len(pairs[name]),
                       "min_timestamp": min(t), "max_timestamp": max(t)}
    overlap = {a + "_" + b: len(pairs[a] & pairs[b]) for a, b in
               (("training", "validation"), ("training", "testing"), ("validation", "testing"))}
    # Report author artifacts faithfully; don't silently rewrite deduplication or held-out masks.
    result = {"repository": provenance["repository"], "commit": provenance["commit"],
              "n_users": n_users, "n_items": n_items, "splits": stats,
              "split_pair_overlaps": overlap,
              "author_test_mask_issue": {"test_users_absent_from_validation": len(set(ds["testing"]) - set(ds["validation"])),
                  "description": "getUserValidItems omits absent users; Procedure.Test enumerates the compressed list and masks the wrong user rows",
                  "reporting": "one scoring pass reports original author mask and corrected per-user validation mask on the same frozen model"},
              "train_graph_cache_max_abs_difference": max_diff,
              "train_graph_nnz": cached.nnz, "source_and_data_sha256":
                  {r["path"]: r["sha256"] for r in provenance["files"]},
              "test_metrics_evaluated": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "protocol_audit.json", result)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(audit(p.parse_args().output))
