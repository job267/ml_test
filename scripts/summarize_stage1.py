"""Summarize fixed final tests, seed variance and paired differences; no significance claims."""

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from research_common import read_json, sha256, write_json


def summarize(suite_dir):
    suite_dir = Path(suite_dir)
    suite, plan = read_json(suite_dir / "suite.json"), read_json(suite_dir / "final_test_plan.json")
    rows = []
    metrics_by_method = {}
    expected_topks = None
    for method in suite["config"]["methods"]:
        metrics_by_method[method] = {}
        for seed in suite["config"]["seeds"]:
            name = f"{method}-seed{seed}"
            run_dir = suite_dir / name
            result = read_json(
                run_dir
                / "evaluations"
                / f'final-{plan["checkpoint_choice"]}'
                / "test_evaluation.json"
            )
            if (
                result["checkpoint_sha256"] != plan["checkpoints"][name]
                or result["training_config"]["dataset_sha256"] != plan["dataset_sha256"]
            ):
                raise ValueError(f"Inconsistent final evaluation: {name}")
            if result["checkpoint_sha256"] != sha256(run_dir / f'{plan["checkpoint_choice"]}.pt'):
                raise ValueError(f"Checkpoint changed since final test: {name}")
            expected_topks = expected_topks or result["topks"]
            if expected_topks != result["topks"]:
                raise ValueError("Top-K settings differ between runs")
            train_rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            metrics = {}
            for group, values in result["groups"].items():
                for index, k in enumerate(result["topks"]):
                    for metric in ("recall", "precision", "ndcg"):
                        metrics[f"{group}/{metric}@{k}"] = values[metric][index]
            metrics_by_method[method][seed] = metrics
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "evaluated_epoch": result["completed_epochs"],
                    "training_seconds": sum(r["train_seconds"] for r in train_rows),
                    "sampling_seconds": sum(r["sampler_seconds"] for r in train_rows),
                    "candidate_scores": sum(r["candidate_scores"] for r in train_rows),
                    "peak_cuda_mib": max(r["peak_cuda_mib"] for r in train_rows),
                    **metrics,
                }
            )
    aggregates = {}
    for method, by_seed in metrics_by_method.items():
        common = set.intersection(*(set(value) for value in by_seed.values()))
        aggregates[method] = {}
        for metric in sorted(common):
            values = np.array([value[metric] for value in by_seed.values()])
            aggregates[method][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                "seeds": len(values),
            }
    paired = {}
    for improved, reference in [
        ("hard", "uniform"),
        ("reliable", "hard"),
        ("curriculum", "reliable"),
    ]:
        if improved not in metrics_by_method or reference not in metrics_by_method:
            continue
        metric = f"overall/ndcg@{expected_topks[0]}"
        differences = [
            metrics_by_method[improved][seed][metric] - metrics_by_method[reference][seed][metric]
            for seed in suite["config"]["seeds"]
        ]
        paired[f"{improved}-minus-{reference}"] = {
            "metric": metric,
            "differences": differences,
            "mean": float(np.mean(differences)),
            "std": float(np.std(differences, ddof=1)) if len(differences) > 1 else None,
        }
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (suite_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "checkpoint_choice": plan["checkpoint_choice"],
        "dataset_sha256": plan["dataset_sha256"],
        "aggregates": aggregates,
        "paired_differences": paired,
        "interpretation": "Observed mean/std and paired seed differences; not evidence of statistical significance.",
    }
    write_json(suite_dir / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.suite_dir), ensure_ascii=False, indent=2))
