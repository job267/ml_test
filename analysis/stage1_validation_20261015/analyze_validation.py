"""Audit a stage-one validation archive without extracting it or reading test labels.

This file lives outside code/ and scripts/ so it does not change the source
fingerprint used to evaluate the existing experiment checkpoints.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile


METRICS = ("precision", "recall", "ndcg")
TIMES = ("train_seconds", "sampler_seconds", "snapshot_seconds", "validation_seconds", "epoch_seconds")


def describe(values):
    """Report seed dispersion; this is not a significance test."""
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def metric(row, name="ndcg", group="overall", index=0):
    return row["validation"]["groups"][group][name][index]


def negative_fractions(row, training):
    """Recover evidence-type shares from the recorded weighted mean.

    The sampler assigns explicit=1, weak=w, unknown=u, therefore
    mean = u + (w-u)*weak_share + (1-u)*explicit_share.
    Small numerical error is expected because the logged mean uses float32.
    """
    weak_weight = training["weak_confidence"]
    unknown_weight = training["unknown_confidence"]
    explicit = row["explicit_negative_fraction"]
    if weak_weight == unknown_weight or weak_weight == 0:
        return {"explicit": explicit, "weak": None, "unknown": None}
    weak = (
        row["mean_negative_evidence"] - unknown_weight - (1 - unknown_weight) * explicit
    ) / (weak_weight - unknown_weight)
    unknown = 1 - weak - explicit
    if not (-1e-5 <= weak <= 1 + 1e-5 and -1e-5 <= unknown <= 1 + 1e-5):
        raise ValueError("Logged confidence mean is inconsistent with configured weights")
    return {"explicit": explicit, "weak": weak, "unknown": unknown}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_comparison(expected, project_root):
    result = {}
    for name, digest in expected.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source path in suite metadata")
        path = project_root / name
        result[name] = (
            "missing"
            if not path.is_file()
            else "match" if hashlib.sha256(path.read_bytes()).hexdigest() == digest else "different"
        )
    return result


def read_archive(path):
    """Read only regular JSON/JSONL members; never extract archive paths."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Archive has duplicate member names")
        if sum(member.size for member in members) > 128 * 1024**2:
            raise ValueError("Validation archive exceeds the 128 MiB review limit")
        for member in members:
            relative = PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unexpected archive member: {member.name}")
        suites = [name for name in names if PurePosixPath(name).name == "suite.json"]
        if len(suites) != 1:
            raise ValueError("Expected exactly one suite.json")
        prefix = str(PurePosixPath(suites[0]).parent)
        suite = json.load(archive.extractfile(suites[0]))
        runs = {}
        for method in suite["config"]["methods"]:
            for seed in suite["config"]["seeds"]:
                name = f"{method}-seed{seed}"
                completion = json.load(archive.extractfile(f"{prefix}/{name}/completion.json"))
                rows = [
                    json.loads(line)
                    for line in archive.extractfile(f"{prefix}/{name}/metrics.jsonl")
                ]
                runs[name] = {"method": method, "seed": seed, "completion": completion, "rows": rows}
    return suite, runs


def audit_runs(suite, runs):
    """Check logged protocol consistency, not the unavailable model/data files."""
    training = suite["config"]["training"]
    epochs = training["epochs"]
    samples = set()
    expected_validation = [
        epoch for epoch in range(1, epochs + 1)
        if epoch % training["eval_interval"] == 0 or epoch == epochs
    ]
    reference_validation = None
    for name, run in runs.items():
        rows, completion = run["rows"], run["completion"]
        if [row["epoch"] for row in rows] != list(range(1, epochs + 1)):
            raise ValueError(f"Missing or repeated epochs: {name}")
        if completion["completed_epochs"] != epochs:
            raise ValueError(f"Incomplete run: {name}")
        validation = [row for row in rows if "validation" in row]
        if [row["epoch"] for row in validation] != expected_validation:
            raise ValueError(f"Unexpected validation schedule: {name}")
        best = max(validation, key=metric)
        if completion["best_val_epoch"] != best["epoch"] or not math.isclose(
            completion["best_val_ndcg"], metric(best), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"Best-validation record disagrees with metrics: {name}")
        for row in rows:
            for key in ("train_total_loss", "mean_negative_evidence", "explicit_negative_fraction", *TIMES):
                value = row.get(key, 0)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {key} in {name}, epoch {row['epoch']}")
            samples.add(row["samples"])
            rate = 0 if run["method"] == "uniform" else training["targeted_rate"]
            if run["method"] == "curriculum":
                rate *= min((row["epoch"] - 1) / training["curriculum_epochs"], 1)
            if not math.isclose(rate, row["targeted_rate"], abs_tol=1e-12):
                raise ValueError(f"Wrong curriculum/targeting rate: {name}")
            if not 0 <= row["targeted_samples"] <= row["samples"]:
                raise ValueError(f"Invalid targeted count: {name}")
            if row["candidate_scores"] != row["targeted_samples"] * training["candidate_size"]:
                raise ValueError(f"Candidate score budget mismatch: {name}")
            if not 0 < row["unique_user_negative_pairs"] <= row["samples"]:
                raise ValueError(f"Invalid unique pair count: {name}")
            negative_fractions(row, training)
            if "validation" in row:
                value = row["validation"]
                signature = {
                    "split": value["split"], "topks": value["topks"],
                    "group_definition": value["group_definition"],
                    "users": {group: item["users"] for group, item in value["groups"].items()},
                }
                if reference_validation is None:
                    reference_validation = signature
                if signature != reference_validation or value["split"] != "val":
                    raise ValueError(f"Evaluation population or groups changed: {name}")
                for group in value["groups"].values():
                    for key in METRICS:
                        if len(group[key]) != len(value["topks"]) or any(
                            not math.isfinite(x) or not 0 <= x <= 1 for x in group[key]
                        ):
                            raise ValueError(f"Invalid validation {key}: {name}")
        run["validation"] = validation
        run["best"] = best
    if len(samples) != 1:
        raise ValueError("Samples per epoch differ within this suite")
    return {
        "runs": len(runs), "training_rows": sum(len(run["rows"]) for run in runs.values()),
        "validation_rows": sum(len(run["validation"]) for run in runs.values()),
        "samples_per_epoch": samples.pop(), "validation_protocol": reference_validation,
        "scope": "Log consistency only; no checkpoints, raw data or test results are in the archive.",
    }


def summarize(suite, runs):
    training = suite["config"]["training"]
    run_rows, group_rows, aggregate_rows, epoch_rows = [], [], [], []
    for name, run in runs.items():
        rows, best, last = run["rows"], run["best"], run["rows"][-1]
        summary = {
            "run": name, "method": run["method"], "seed": run["seed"],
            "best_epoch": best["epoch"], "best_ndcg": metric(best), "last_ndcg": metric(last),
            "best_recall": metric(best, "recall"), "last_recall": metric(last, "recall"),
            "ndcg_drop_percent": 100 * (1 - metric(last) / metric(best)),
            "last_loss": last["train_total_loss"],
            "peak_cuda_mib": max(row["peak_cuda_mib"] for row in rows),
        }
        summary.update({f"sum_{key}": sum(row.get(key, 0) for row in rows) for key in TIMES})
        run_rows.append(summary)
        for choice, row in (("best_val", best), ("last", last)):
            for group, values in row["validation"]["groups"].items():
                for index, k in enumerate(row["validation"]["topks"]):
                    hits = values["precision"][index] * values["users"] * k
                    if not math.isclose(hits, round(hits), abs_tol=1e-6):
                        raise ValueError("Group precision does not imply an integer hit count")
                    group_rows.append({
                        "run": name, "method": run["method"], "seed": run["seed"],
                        "checkpoint": choice, "epoch": row["epoch"], "group": group,
                        "users": values["users"], "k": k, "hits": round(hits),
                        **{key: values[key][index] for key in METRICS},
                    })
    for method in suite["config"]["methods"]:
        method_runs = [run for run in runs.values() if run["method"] == method]
        summaries = [row for row in run_rows if row["method"] == method]
        aggregate = {"method": method, "seeds": len(method_runs)}
        for key in ("best_ndcg", "last_ndcg", "best_recall", "last_recall", "ndcg_drop_percent", "sum_epoch_seconds"):
            stats = describe([row[key] for row in summaries])
            aggregate[f"{key}_mean"] = stats["mean"]
            aggregate[f"{key}_std"] = stats["std"]
        aggregate_rows.append(aggregate)
        for epoch in range(1, training["epochs"] + 1):
            rows = [run["rows"][epoch - 1] for run in method_runs]
            item = {"method": method, "epoch": epoch}
            for key in ("train_total_loss", "targeted_rate", "targeted_samples", "candidate_scores", "mean_negative_evidence", "explicit_negative_fraction", "unique_user_negative_pairs", "train_seconds", "sampler_seconds"):
                item[key] = statistics.mean(row[key] for row in rows)
            for kind in ("weak", "unknown", "explicit"):
                shares = [negative_fractions(row, training)[kind] for row in rows]
                item[f"{kind}_fraction"] = None if any(x is None for x in shares) else statistics.mean(shares)
            item["unique_pair_fraction"] = statistics.mean(row["unique_user_negative_pairs"] / row["samples"] for row in rows)
            if "validation" in rows[0]:
                for group in rows[0]["validation"]["groups"]:
                    for key in METRICS:
                        stats = describe([metric(row, key, group) for row in rows])
                        item[f"{group}/{key}"] = stats["mean"]
                        item[f"{group}/{key}_std"] = stats["std"]
            epoch_rows.append(item)
    paired = {}
    for choice, key in (("best_val", "best_ndcg"), ("last", "last_ndcg")):
        paired[choice] = {}
        for treatment, reference in (("hard", "uniform"), ("reliable", "hard"), ("curriculum", "reliable"), ("curriculum", "uniform")):
            left = {row["seed"]: row[key] for row in run_rows if row["method"] == treatment}
            right = {row["seed"]: row[key] for row in run_rows if row["method"] == reference}
            if not left or left.keys() != right.keys():
                continue
            differences = [left[seed] - right[seed] for seed in sorted(left)]
            paired[choice][f"{treatment}-minus-{reference}"] = {
                **describe(differences), "positive_seeds": sum(x > 0 for x in differences),
                "relative_mean_percent": 100 * (statistics.mean(left.values()) / statistics.mean(right.values()) - 1),
            }
    return run_rows, group_rows, aggregate_rows, epoch_rows, paired


def plot_diagnostics(epoch_rows, output):
    """Plot seed means and sample standard deviations, not confidence intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"uniform": "#4477AA", "hard": "#EE7733", "reliable": "#CC3311", "curriculum": "#228833"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method, color in colors.items():
        rows = [row for row in epoch_rows if row["method"] == method]
        validation = [row for row in rows if "overall/ndcg" in row]
        for axis, upper in ((axes[0, 0], 100), (axes[0, 1], math.inf)):
            selected = [row for row in validation if row["epoch"] <= upper]
            x = [row["epoch"] for row in selected]
            y = [row["overall/ndcg"] for row in selected]
            deviation = [row["overall/ndcg_std"] or 0 for row in selected]
            axis.plot(x, y, color=color, label=method, linewidth=1.8)
            axis.fill_between(x, [max(0, a-b) for a,b in zip(y,deviation)], [a+b for a,b in zip(y,deviation)], color=color, alpha=.1)
        axes[1, 0].plot([row["epoch"] for row in rows], [row["weak_fraction"] for row in rows], color=color, label=method)
        axes[1, 1].plot([row["epoch"] for row in rows], [row["unique_pair_fraction"] for row in rows], color=color, label=method)
    titles = ("Validation NDCG@20: first 100 epochs", "Validation NDCG@20: all epochs", "Weak-evidence share among sampled negatives", "Distinct user-negative pairs / sampled triplets")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("NDCG@20")
    axes[0, 1].set_ylabel("NDCG@20")
    axes[1, 0].set_ylabel("Share")
    axes[1, 1].set_ylabel("Fraction")
    fig.savefig(output / "validation_and_sampling.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory to preserve previous reports")
    suite, runs = read_archive(args.archive)
    audit = audit_runs(suite, runs)
    source = source_comparison(suite["code_sha256"], Path(__file__).resolve().parents[2])
    run_rows, group_rows, aggregates, epoch_rows, paired = summarize(suite, runs)
    report = {
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "suite": suite, "audit": audit, "local_source_comparison": source,
        "aggregates": aggregates, "paired_differences": paired,
        "run_summary": run_rows,
        "limitations": [
            "Validation only. No test results, model checkpoints, environment records or prepared manifest were supplied.",
            "Best-validation metrics select each run by its first top-K overall NDCG; they are not test estimates.",
            "Weak/unknown sample shares are algebraic reconstructions of heuristic evidence weights, not true preference probabilities.",
            "epoch_seconds excludes checkpoint writing and process startup; snapshot_seconds is already inside sampler_seconds.",
            "Observed seed differences do not establish statistical significance.",
        ],
    }
    args.output_dir.mkdir(parents=True)
    for name, rows in (("run_summary", run_rows), ("checkpoint_groups", group_rows), ("method_summary", aggregates), ("epoch_means", epoch_rows)):
        write_csv(args.output_dir / f"{name}.csv", rows)
    (args.output_dir / "validation_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot_diagnostics(epoch_rows, args.output_dir)
    print(json.dumps({"audit": audit, "aggregates": aggregates, "paired_differences": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
