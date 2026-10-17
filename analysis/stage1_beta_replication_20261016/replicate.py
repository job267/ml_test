"""Run the two new seeds with the unchanged pilot driver and compare fixed budgets."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "analysis/stage1_beta_pilot_20261015"
sys.path.insert(0, str(ROOT / "code"))
from experiment import code_hashes, load_checkpoint
from research_common import read_json, sha256


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def window_result(rows, horizon):
    """Choose only inside the common window; don't independently maximize subgroups."""
    window = [row for row in rows if row["epoch"] <= horizon]
    if [row["epoch"] for row in window] != list(range(1, horizon + 1)):
        return None
    if any(row["validation"]["split"] != "val" for row in window):
        raise ValueError("Only validation metrics are allowed")
    best = max(window, key=lambda row: row["validation"]["groups"]["overall"]["ndcg"][0])
    groups = best["validation"]["groups"]
    tail = groups["item_tail"]
    return {
        "best_epoch": best["epoch"],
        "ndcg20": groups["overall"]["ndcg"][0],
        "recall20": groups["overall"]["recall"][0],
        "sparse_ndcg20": groups["user_sparse"]["ndcg"][0],
        "tail_ndcg20": tail["ndcg"][0],
        "tail_hit_pairs": round(tail["precision"][0] * tail["users"] * 20),
        "unique_negative_pair_fraction": best["unique_user_negative_pairs"] / best["samples"],
        "groups_at_best": groups,
        "last_ndcg20": window[-1]["validation"]["groups"]["overall"]["ndcg"][0],
        "late_best": best["epoch"] > horizon - 5,
    }


def aggregate(rows, seeds):
    fields = ("ndcg20", "recall20", "sparse_ndcg20", "tail_ndcg20", "unique_negative_pair_fraction")
    result = {}
    for variant in ("uniform", "beta0", "beta025"):
        chosen = [row for row in rows if row["variant"] == variant and row["seed"] in seeds]
        if {row["seed"] for row in chosen} != set(seeds):
            continue
        result[variant] = {field: {
            "mean": statistics.mean(row[field] for row in chosen),
            "sample_std": statistics.stdev(row[field] for row in chosen) if len(chosen) > 1 else None,
        } for field in fields}
    return result


def paired(rows, seeds):
    by_key = {(r["variant"], r["seed"]): r for r in rows}
    result = {}
    for candidate, reference in (("beta0", "uniform"), ("beta025", "uniform"), ("beta025", "beta0")):
        pairs = []
        for seed in seeds:
            if (candidate, seed) not in by_key or (reference, seed) not in by_key:
                continue
            a, b = by_key[candidate, seed], by_key[reference, seed]
            pairs.append({"seed": seed, **{field: a[field] - b[field] for field in
                         ("ndcg20", "recall20", "sparse_ndcg20", "tail_ndcg20")}})
        result[f"{candidate}-minus-{reference}"] = pairs
    return result


def summarize(output, config):
    rows = []
    references = ROOT / config["reference_root"]
    seeds = [config["reference_seed"], *config["seeds"]]
    for seed in seeds:
        for variant in config["variants"]:
            base = references if seed == config["reference_seed"] else output / f"seed{seed}"
            path = base / f'{variant["id"]}-seed{seed}' / "metrics.jsonl"
            if not path.exists():
                continue
            history = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            metrics = window_result(history, config["training"]["epochs"])
            if metrics is not None:
                rows.append({"variant": variant["id"], "seed": seed,
                             "role": "exploratory_reference" if seed == config["reference_seed"] else "new_replication",
                             **metrics})
    new_rows = [r for r in rows if r["seed"] in config["seeds"]]
    report = {
        "updated_utc": utc(), "split": "val", "comparison_horizon": config["training"]["epochs"],
        "completed_new_runs": len(new_rows), "expected_new_runs": len(config["seeds"]) * len(config["variants"]),
        "rows": rows, "new_seed_aggregates": aggregate(rows, config["seeds"]),
        "three_seed_descriptive_aggregates": aggregate(rows, seeds),
        "paired_new_seeds": paired(rows, config["seeds"]), "paired_all_seeds": paired(rows, seeds),
        "extension_review": [{"variant": r["variant"], "seed": r["seed"], "best_epoch": r["best_epoch"]}
                             for r in new_rows if r["late_best"]],
        "interpretation": config["interpretation"] + " Validation only; no significance or test-performance claim.",
    }
    save_json(output / "comparison.json", report)
    flattened = [{k: v for k, v in r.items() if k != "groups_at_best"} for r in rows]
    if flattened:
        temporary = output / "comparison.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
        temporary.replace(output / "comparison.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_beta_replication_20261016.json")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/kuairand_stage1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/stage1_beta_replication_20261016")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Require CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0")
    if Path(sys.prefix).name != "gcn":
        raise RuntimeError("Use the existing gcn conda environment")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")
    gpu0_uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    cuda_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if cuda_uuid.lower() != gpu0_uuid.lower():
        raise RuntimeError("CUDA visible device does not match physical GPU 0")
    config = read_json(args.config)
    if config["seeds"] != [2021, 2022] or [v["id"] for v in config["variants"]] != ["uniform", "beta0", "beta025"]:
        raise ValueError("Unexpected replication seeds or variants")
    reference = ROOT / config["reference_root"]
    old_plan = read_json(reference / "pilot_plan.json")
    if code_hashes() != old_plan["code_sha256"] or sha256(args.dataset_dir / "manifest.json") != old_plan["dataset_sha256"]:
        raise ValueError("Reference source or prepared dataset differs")
    if {k: v for k, v in config["training"].items() if k != "epochs"} != {
            k: v for k, v in old_plan["config"]["training"].items() if k != "epochs"}:
        raise ValueError("Replication must only change seeds and training horizon")
    old_variants = {v["id"]: v for v in old_plan["config"]["variants"]}
    if any(v != old_variants.get(v["id"]) for v in config["variants"]):
        raise ValueError("Replication changed a sampling rule")
    source_paths = sorted(PILOT.glob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    def fingerprints():
        return {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    plan = {
        "config": config, "config_sha256": sha256(args.config), "code_sha256": code_hashes(),
        "driver_sha256": fingerprints(), "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_sha256": old_plan["dataset_sha256"], "reference_plan_sha256": sha256(reference / "pilot_plan.json"),
        "reference_metrics_sha256": {v["id"]: sha256(reference / f'{v["id"]}-seed2020/metrics.jsonl') for v in config["variants"]},
        "python": sys.executable, "torch": str(torch.__version__), "gpu0_uuid": gpu0_uuid,
        "cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID", "test_evaluated": False,
    }
    output = args.output_root.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".replication.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if read_json(output / "replication_plan.json") != plan:
            raise ValueError("Replication source, data, config, reference or GPU identity changed")
    else:
        save_json(output / "replication_plan.json", plan)
        (output / "configs").mkdir()
        for seed in config["seeds"]:
            save_json(output / "configs" / f"seed{seed}.json", {
                "schema_version": 1, "seed": seed, "training": config["training"],
                "variants": config["variants"], "selection": config["selection"], "followup": config["extension_review"],
            })
    state = {"status": "running", "started_utc": utc(), "queue_pid": os.getpid(), "completed_seeds": []}
    save_json(output / "status.json", state)
    child = None
    def stop(signum, frame):
        raise InterruptedError(f"Replication interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def verify():
        if code_hashes() != plan["code_sha256"] or fingerprints() != plan["driver_sha256"] or sha256(args.config) != plan["config_sha256"]:
            raise ValueError("Replication sources or config changed")
        if sha256(args.dataset_dir / "manifest.json") != plan["dataset_sha256"]:
            raise ValueError("Prepared dataset manifest changed")
        for v in config["variants"]:
            if sha256(reference / f'{v["id"]}-seed2020/metrics.jsonl') != plan["reference_metrics_sha256"][v["id"]]:
                raise ValueError("Exploratory reference changed")
    try:
        summarize(output, config)
        for seed in config["seeds"]:
            verify()
            seed_root = output / f"seed{seed}"
            state.update(active_seed=seed, updated_utc=utc())
            save_json(output / "status.json", state)
            finished = (seed_root / "status.json").exists() and read_json(seed_root / "status.json")["status"] == "completed"
            if not finished:
                command = [sys.executable, "-u", str(PILOT / "run_pilot.py"),
                           "--dataset-dir", str(args.dataset_dir.resolve()), "--config", str(output / "configs" / f"seed{seed}.json"),
                           "--output-root", str(seed_root)]
                if seed_root.exists():
                    if not args.resume:
                        raise FileExistsError(seed_root)
                    command.append("--resume")
                with (output / f"seed{seed}.log").open("a", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state["child_pid"] = child.pid
                    save_json(output / "status.json", state)
                    while True:
                        try:
                            returncode = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            report = summarize(output, config)
                            state.update(completed_new_runs=report["completed_new_runs"], updated_utc=utc())
                            save_json(output / "status.json", state)
                    child = None
                if returncode:
                    raise RuntimeError(f"Seed {seed} failed; inspect {seed_root} and its log")
            verify()
            for variant in config["variants"]:
                run = seed_root / f'{variant["id"]}-seed{seed}'
                complete = read_json(run / "completion.json")
                ck = load_checkpoint(run / "last.pt")
                if (complete["last_sha256"] != sha256(run / "last.pt") or complete["test_evaluated"]
                        or ck["completed_epochs"] != config["training"]["epochs"] or ck["code_sha256"] != plan["code_sha256"]):
                    raise ValueError(f"Run integrity mismatch: {run}")
            state["completed_seeds"].append(seed)
            report = summarize(output, config)
            state["completed_new_runs"] = report["completed_new_runs"]
        state.update(status="completed", completed_utc=utc(), active_seed=None, child_pid=None,
                     extension_review=report["extension_review"])
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(status="failed", error=repr(error), failed_utc=utc(), child_pid=None)
        save_json(output / "status.json", state)
        raise
    save_json(output / "status.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
