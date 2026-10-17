"""Measure evidence penalties on fixed training-only candidate pools, without updates."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from experiment import build_model, code_hashes, load_checkpoint
from research_common import read_json, sha256
from research_data import FeedbackDataset
from research_sampling import FeedbackSampler
from utils import set_seed


def selection_diagnostics(scores, confidences, pool, users, kinds, item_count, beta, temperature=1.0):
    """Compare score-only and evidence-adjusted choices on exactly the same pool."""
    if not np.isfinite(scores).all() or not np.isfinite(confidences).all() or (confidences <= 0).any():
        raise ValueError("Diagnostics require finite scores and positive finite confidence")
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid beta or temperature")
    penalties = temperature * beta * np.log(confidences)
    hard = scores.argmax(axis=1)
    selected = (scores + penalties).argmax(axis=1)
    rows = np.arange(len(scores))
    items = pool[rows, selected]
    hard_items = pool[rows, hard]
    selected_kinds = Counter(kinds.get(int(u) * item_count + int(i), "unknown") for u, i in zip(users, items))
    return {
        "beta": beta, "temperature": temperature, "candidate_rows": len(scores),
        "changed_item_fraction_vs_hard": float(np.mean(items != hard_items)),
        "mean_raw_score_sacrificed": float(np.mean(scores[rows, hard] - scores[rows, selected])),
        "mean_effective_penalty": float(penalties.mean()),
        "selected_kind_counts": dict(selected_kinds),
        "selected_kind_fractions": {key: selected_kinds[key] / len(scores) for key in ("explicit", "weak", "unknown")},
        "unique_user_negative_pair_fraction": len(np.unique(users * item_count + items)) / len(users),
    }


@torch.no_grad()
def diagnose(dataset, model, config, count=8192, seed=2020):
    sampler = FeedbackSampler(dataset, method="reliable", seed=seed,
                              candidate_size=config["candidate_size"],
                              evidence_fraction=config["evidence_fraction"],
                              unknown_confidence=config["unknown_confidence"])
    users = sampler.rng.choice(dataset.active_users, size=count)
    pool = sampler.candidates(users)
    if dataset.is_positive(users[:, None], pool).any():
        raise ValueError("A diagnostic candidate is a training positive")
    model.eval()
    all_users, all_items = model.computer()
    scores = []
    for start in range(0, count, 1024):
        u = torch.as_tensor(users[start:start + 1024], device=all_users.device)
        i = torch.as_tensor(pool[start:start + 1024], device=all_items.device)
        scores.append((all_users[u, None, :] * all_items[i]).sum(-1).cpu().numpy())
    scores = np.concatenate(scores)
    confidences = dataset.negative_confidence(users[:, None], pool, config["unknown_confidence"])
    evidence_sizes = np.asarray([len(items) for items in dataset.evidence_items])
    quantiles = [0, 0.25, 0.5, 0.75, 0.95, 1]
    return {
        "candidate_seed": seed, "candidate_rows": count,
        "candidate_size": config["candidate_size"], "evidence_fraction": config["evidence_fraction"],
        "raw_score_quantile_levels": quantiles,
        "raw_score_quantiles": np.quantile(scores, quantiles).tolist(),
        "raw_score_std": float(scores.std()),
        "mean_within_pool_score_range": float(np.ptp(scores, axis=1).mean()),
        "evidence_pool_size_quantiles": np.quantile(evidence_sizes, quantiles).tolist(),
        "users_with_evidence": int((evidence_sizes > 0).sum()),
        "choices": [selection_diagnostics(scores, confidences, pool, users, dataset.evidence_kinds,
                                           dataset.m_items, beta, config["temperature"])
                    for beta in (0.0, 0.05, 0.1, 0.25, 1.0)],
        "interpretation": "Training-only fixed candidate pool; no optimizer update or held-out target filtering. Duplicate fraction uses this diagnostic sample size, not the full epoch budget.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/kuairand_stage1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_beta_pilot.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("Candidate row count must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Diagnostics require only physical GPU 0 to be visible")
    pilot = read_json(args.config)
    config = dict(pilot["training"])
    source = code_hashes()
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if checkpoint:
        if checkpoint["code_sha256"] != source:
            raise ValueError("Diagnostic checkpoint source differs")
        config = checkpoint["training_config"]
    set_seed(pilot["seed"])
    dataset = FeedbackDataset(args.dataset_dir, "cuda", config["weak_confidence"])
    if checkpoint and config["dataset_sha256"] != dataset.fingerprint:
        raise ValueError("Diagnostic checkpoint dataset differs")
    model, _ = build_model(config, dataset, torch.device("cuda"))
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    result = diagnose(dataset, model, config, args.count, pilot["seed"])
    result.update(dataset_sha256=dataset.fingerprint, code_sha256=source,
                  checkpoint_sha256=sha256(args.checkpoint) if checkpoint else None,
                  checkpoint_epoch=checkpoint["completed_epochs"] if checkpoint else 0,
                  gpu=torch.cuda.get_device_name(0), cuda_visible_devices="0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "raw_score_std": result["raw_score_std"], "choices": result["choices"]}), flush=True)


if __name__ == "__main__":
    main()
