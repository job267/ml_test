"""Evaluate validation-only random and popularity references on the original data.

Keep this tool under analysis/, not code/ or scripts/: existing checkpoints
require the original training source fingerprint. No training or test evaluation
is performed. The uniform reference is an exact expectation, not a sampled run.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from research_common import read_json, sha256
from research_data import FeedbackDataset
from research_evaluation import evaluate
from world import parse_topks


class PopularityReference:
    """Rank by unique training-positive degree; break ties by ascending item ID."""

    def __init__(self, dataset, device):
        degree = np.asarray(dataset.UserItemNet.sum(axis=0)).ravel().astype(np.float64)
        # A tie-break in (0, 1) cannot reverse two different integer degrees.
        tie_break = (dataset.m_items - np.arange(dataset.m_items)) / (dataset.m_items + 1)
        self.users = torch.ones((dataset.n_users, 1), dtype=torch.float64, device=device)
        self.items = torch.as_tensor((degree + tie_break)[:, None], device=device)

    def eval(self):
        return self

    def computer(self):
        return self.users, self.items


def expected_uniform_validation(dataset, topks):
    """Compute exact expected metrics for a uniform ranking of legal items.

    For N legal candidates and R relevant items, any position is relevant with
    probability R/N. Thus E[Recall@K]=K/N and E[DCG@K]=(R/N)*sum(discounts).
    Group metrics use subgroup relevance inside the same global ranking, exactly
    as the existing evaluation protocol does.
    """
    user_degree = np.diff(dataset.UserItemNet.indptr)
    item_degree = np.asarray(dataset.UserItemNet.sum(axis=0)).ravel()
    low, high = np.quantile(user_degree[user_degree > 0], [0.25, 0.75])
    tail = np.median(item_degree[item_degree > 0])
    discount_sum = np.r_[0.0, np.cumsum(1 / np.log2(np.arange(2, max(topks) + 2)))]
    groups = {}

    def add(name, relevant, legal_count):
        count = len(relevant)
        if count == 0:
            return
        record = groups.setdefault(name, {"users": 0, "sums": np.zeros((len(topks), 3))})
        record["users"] += 1
        for index, k in enumerate(topks):
            expected_precision = count / legal_count
            expected_recall = k / legal_count
            expected_ndcg = expected_precision * discount_sum[k] / discount_sum[min(count, k)]
            record["sums"][index] += expected_precision, expected_recall, expected_ndcg

    for user, relevant in dataset.valDict.items():
        excluded = set(dataset.allPos[user].tolist())
        if excluded.intersection(relevant):
            raise ValueError("Validation target overlaps the ranking mask")
        legal_count = dataset.m_items - len(excluded)
        if legal_count < max(topks):
            raise ValueError("Too few legal candidates for the requested Top-K")
        degree = user_degree[user]
        user_group = "user_sparse" if degree <= low else "user_active" if degree > high else "user_middle"
        add("overall", relevant, legal_count)
        add(user_group, relevant, legal_count)
        add("item_zero_degree", [item for item in relevant if item_degree[item] == 0], legal_count)
        add("item_tail", [item for item in relevant if 0 < item_degree[item] <= tail], legal_count)
        add("item_head", [item for item in relevant if item_degree[item] > tail], legal_count)

    result = {}
    for name, record in groups.items():
        means = record["sums"] / record["users"]
        result[name] = {
            "users": record["users"], "precision": means[:, 0].tolist(),
            "recall": means[:, 1].tolist(), "ndcg": means[:, 2].tolist(),
        }
    return {
        "split": "val", "topks": topks, "groups": result,
        "group_definition": {
            "user_degree_q25": float(low), "user_degree_q75": float(high),
            "positive_item_degree_median": float(tail),
            "item_metrics": "subset ground truth within global ranking",
        },
        "interpretation": "Exact expectation under uniform random ranking of legal candidates; no seed standard deviation.",
    }


def verify_sources(suite):
    """Require the recorded source version before evaluating references."""
    for name, expected in suite["code_sha256"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Original experiment source changed: {name}")


def validation_data_audit(dataset):
    """Describe evidence coverage and validation targets; never filter training."""
    evidence_counts = Counter(dataset.evidence_kinds.values())
    evidence_users = {key // dataset.m_items for key in dataset.evidence_kinds}
    explicit_users = {
        key // dataset.m_items for key, kind in dataset.evidence_kinds.items() if kind == "explicit"
    }
    target_history = Counter()
    target_items = set()
    for user, items in dataset.valDict.items():
        for item in items:
            kind = dataset.evidence_kinds.get(user * dataset.m_items + item, "unknown")
            target_history[kind] += 1
            target_items.add(item)
    return {
        "training_users": dataset.n_users, "training_items": dataset.m_items,
        "training_positive_pairs": dataset.trainDataSize,
        "evidence_pairs": dict(evidence_counts), "users_with_evidence": len(evidence_users),
        "users_with_explicit_evidence": len(explicit_users),
        "validation_users": len(dataset.valDict),
        "validation_positive_pairs": sum(len(items) for items in dataset.valDict.values()),
        "validation_distinct_items": len(target_items),
        "validation_targets_by_prior_training_evidence": dict(target_history),
        "interpretation": "A future positive with earlier negative evidence may reflect ambiguity or preference change; it is not used to exclude training negatives.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Reference output already exists; preserve it or choose another name")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    suite = read_json(args.suite_dir / "suite.json")
    verify_sources(suite)
    dataset_dir = Path(suite["dataset_dir"])
    if sha256(dataset_dir / "manifest.json") != suite["dataset_sha256"]:
        raise ValueError("Prepared dataset differs from the original training suite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    device = torch.device(args.device)
    training = suite["config"]["training"]
    topks = parse_topks(training["topks"])
    dataset = FeedbackDataset(dataset_dir, device=device, weak_confidence=training["weak_confidence"])
    begin = perf_counter()
    uniform = expected_uniform_validation(dataset, topks)
    popularity = evaluate(dataset, PopularityReference(dataset, device), "val", topks, args.batch_size)
    if uniform["group_definition"] != popularity["group_definition"] or {
        key: value["users"] for key, value in uniform["groups"].items()
    } != {key: value["users"] for key, value in popularity["groups"].items()}:
        raise ValueError("Reference group definitions disagree")
    result = {
        "dataset_sha256": suite["dataset_sha256"],
        "source_sha256": suite["code_sha256"],
        "reference_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": str(device), "split": "val", "seconds": perf_counter() - begin,
        "environment": {
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "dataset_manifest": dataset.manifest,
        "data_audit": validation_data_audit(dataset),
        "uniform_expected": uniform, "training_popularity": popularity,
        "limitations": [
            "Validation-only references; no test plan is created or modified.",
            "Popularity uses unique training-positive degree, not validation labels or total exposure count.",
            "Uniform expected NDCG is a mean expectation, not a realized model or a confidence interval.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "uniform_expected": uniform["groups"]["overall"],
        "training_popularity": popularity["groups"]["overall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
