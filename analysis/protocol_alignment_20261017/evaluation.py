"""Rank once and evaluate several validation targets without changing predictions."""
import math
import numpy as np
import torch
from common import ROOT
from utils import metrics_for_user


@torch.no_grad()
def rank(model, dataset, users, k=20, batch_size=512):
    model.eval()
    user_factors, item_factors = model.computer()
    result = []
    for start in range(0, len(users), batch_size):
        subset = users[start:start + batch_size]
        ids = torch.tensor(subset, dtype=torch.long, device=user_factors.device)
        scores = user_factors[ids] @ item_factors.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite unmasked validation scores")
        for row, u in enumerate(subset):
            known = dataset.allPos[u]
            if dataset.m_items - len(known) < k:
                raise ValueError("Insufficient legal ranking candidates")
            scores[row, torch.as_tensor(known, dtype=torch.long, device=scores.device)] = -torch.inf
        result.append(scores.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, k), dtype=np.int64)


def summarize_ranks(dataset, users, predictions, truth, k=20):
    """Same metric function and train-degree groups as research_evaluation.evaluate."""
    ud = np.diff(dataset.UserItemNet.indptr)
    ideg = np.asarray(dataset.UserItemNet.sum(axis=0)).ravel()
    low, high = np.quantile(ud[ud > 0], [.25, .75])
    tail = np.median(ideg[ideg > 0])
    accum = {}
    def add(name, predicted, relevant):
        if not relevant:
            return
        record = accum.setdefault(name, {"users": 0, "sums": np.zeros(3)})
        record["users"] += 1
        record["sums"] += metrics_for_user(predicted, relevant, k)
    for u, predicted in zip(users, predictions):
        relevant = truth.get(u, [])
        if not relevant:
            continue
        add("overall", predicted, relevant)
        name = "user_sparse" if ud[u] <= low else "user_active" if ud[u] > high else "user_middle"
        add(name, predicted, relevant)
        add("item_zero_degree", predicted, [i for i in relevant if ideg[i] == 0])
        add("item_tail", predicted, [i for i in relevant if 0 < ideg[i] <= tail])
        add("item_head", predicted, [i for i in relevant if ideg[i] > tail])
    groups = {name: {"users": a["users"], **{metric: [float(value)] for metric, value in
        zip(("precision", "recall", "ndcg"), a["sums"] / a["users"])}} for name, a in accum.items()}
    if "overall" not in groups:
        raise ValueError("No validation users in this view")
    return {"split": "val", "topks": [k], "groups": groups,
            "group_definition": {"user_degree_q25": float(low), "user_degree_q75": float(high),
                                 "positive_item_degree_median": float(tail),
                                 "item_metrics": "subset ground truth within global ranking"}}


def views(model, dataset, batch_size=512):
    users = sorted(dataset.views["all"])
    predictions = rank(model, dataset, users, 20, batch_size)
    common_positions = [j for j, u in enumerate(users) if u in dataset.common_eval_users]
    common_users = [users[j] for j in common_positions]
    common_predictions = predictions[common_positions]
    result = {}
    for source in ("random", "all"):
        result[source] = {"available_users": summarize_ranks(dataset, users, predictions, dataset.views[source]),
                          "common_users": summarize_ranks(dataset, common_users, common_predictions, dataset.views[source])}
    return result, users, predictions


def reference_metrics(dataset, truth):
    """Deterministic train popularity and exact uniform-ranking expectations."""
    degree = np.asarray(dataset.UserItemNet.sum(axis=0)).ravel()
    order = np.lexsort((np.arange(dataset.m_items), -degree))
    users = sorted(truth)
    predicted = []
    recalls, ndcgs = [], []
    discounts = 1 / np.log2(np.arange(2, 22))
    for u in users:
        known = set(dataset.allPos[u].tolist())
        row = []
        for item in order:
            if int(item) not in known:
                row.append(int(item))
            if len(row) == 20:
                break
        if len(row) < 20:
            raise ValueError("Insufficient legal candidates for reference ranking")
        predicted.append(row)
        legal = dataset.m_items - len(known)
        relevant = len(truth[u])
        recalls.append(20 / legal)
        ndcgs.append((relevant / legal) * discounts.sum() / discounts[:min(relevant, 20)].sum())
    return {"popularity": summarize_ranks(dataset, users, np.asarray(predicted), truth),
            "uniform_expected": {"ndcg20": float(np.mean(ndcgs)), "recall20": float(np.mean(recalls))}}
