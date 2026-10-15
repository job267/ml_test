"""Full ranking on a chosen split with explicit masks and fixed train-degree groups."""

import numpy as np
import torch
from utils import iter_batches, metrics_for_user


@torch.no_grad()
def evaluate(dataset, model, split="val", topks=(20,), batch_size=100):
    if split not in ("val", "test") or batch_size < 1 or not topks or min(topks) < 1:
        raise ValueError("Invalid evaluation arguments")
    truth = dataset.valDict if split == "val" else dataset.testDict
    model.eval()
    users, items = model.computer()
    user_degree = np.diff(dataset.UserItemNet.indptr)
    item_degree = np.asarray(dataset.UserItemNet.sum(axis=0)).ravel()
    user_low, user_high = np.quantile(user_degree[user_degree > 0], [0.25, 0.75])
    item_tail = np.median(item_degree[item_degree > 0])
    groups = {}

    def add(name, predicted, relevant):
        if not relevant:
            return
        record = groups.setdefault(name, {"users": 0, "sums": np.zeros((len(topks), 3))})
        record["users"] += 1
        for index, k in enumerate(topks):
            record["sums"][index] += metrics_for_user(predicted, relevant, k)

    for subset in iter_batches(list(truth), batch_size):
        ids = torch.tensor(subset, device=users.device, dtype=torch.long)
        scores = users[ids] @ items.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite unmasked evaluation scores")
        for row, user in enumerate(subset):
            excluded = set(dataset.allPos[user].tolist())
            if split == "test":
                excluded.update(dataset.val_knownDict.get(user, []))
            if dataset.m_items - len(excluded) < max(topks):
                raise ValueError(f"User {user}: fewer than max(topks) eligible ranking items")
            scores[row, torch.tensor(sorted(excluded), device=scores.device, dtype=torch.long)] = (
                -torch.inf
            )
        if torch.isnan(scores).any() or torch.isposinf(scores).any():
            raise FloatingPointError("Invalid evaluation scores")
        ranks = torch.topk(scores, max(topks), dim=1).indices.cpu().numpy()
        for row, user in enumerate(subset):
            relevant = truth[user]
            add("overall", ranks[row], relevant)
            degree = user_degree[user]
            name = (
                "user_sparse"
                if degree <= user_low
                else "user_active" if degree > user_high else "user_middle"
            )
            add(name, ranks[row], relevant)
            add("item_zero_degree", ranks[row], [i for i in relevant if item_degree[i] == 0])
            add("item_tail", ranks[row], [i for i in relevant if 0 < item_degree[i] <= item_tail])
            add("item_head", ranks[row], [i for i in relevant if item_degree[i] > item_tail])
    if "overall" not in groups:
        raise ValueError(f"No users to evaluate in {split}")
    results = {}
    for name, record in groups.items():
        means = record["sums"] / record["users"]
        results[name] = {
            "users": record["users"],
            "precision": means[:, 0].tolist(),
            "recall": means[:, 1].tolist(),
            "ndcg": means[:, 2].tolist(),
        }
    return {
        "split": split,
        "topks": list(topks),
        "groups": results,
        "group_definition": {
            "user_degree_q25": float(user_low),
            "user_degree_q75": float(user_high),
            "positive_item_degree_median": float(item_tail),
            "item_metrics": "subset ground truth within global ranking",
        },
    }
