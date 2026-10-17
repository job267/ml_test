"""Isolate candidate proposal source with the unchanged training/pilot implementation."""

import argparse
import csv
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RATE = ROOT / "analysis/stage1_targeted_rate_20261016"
spec = importlib.util.spec_from_file_location("frozen_rate_study", RATE / "run_rates.py")
rates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rates)
read_json, sha256, save_json, utc = rates.read_json, rates.sha256, rates.save_json, rates.utc
code_hashes, load_checkpoint, window_result = rates.code_hashes, rates.load_checkpoint, rates.window_result


def pilot_config(config, seed):
    return {"schema_version": 1, "seed": seed, "training": config["training"],
            "variants": [{"id": "hard", "sampler": "reliable", "reliability_beta": 0.0}],
            "selection": config["selection"], "followup": config["interpretation"]}


def summarize(output, config, references):
    locations = [(seed, {"uniform": "uniform", "rate08": "evidence05"}[name], ROOT / run)
                 for seed in config["seeds"] for name, run in references["reference_runs"][str(seed)].items()]
    locations += [(seed, "evidence0", output / f"seed{seed}" / f"hard-seed{seed}") for seed in config["seeds"]]
    rows = []
    for seed, variant, run in locations:
        path = run / "metrics.jsonl"
        if not path.exists():
            continue
        history = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(rates.json.loads(line))
            except rates.json.JSONDecodeError:
                break
        result = window_result(history, config["training"]["epochs"])
        if result is None:
            continue
        best = history[result["best_epoch"] - 1]
        explicit = best["explicit_negative_fraction"]
        weak = (best["mean_negative_evidence"] - 0.05 - 0.95 * explicit) / 0.15
        rows.append({"seed": seed, "variant": variant, "role": "new_source_ablation" if variant == "evidence0" else "existing_reference",
                     **result, "weak_negative_fraction": weak, "explicit_negative_fraction": explicit})
    aggregates = {}
    for variant in ("uniform", "evidence05", "evidence0"):
        chosen = [r for r in rows if r["variant"] == variant]
        if {r["seed"] for r in chosen} == set(config["seeds"]):
            aggregates[variant] = {field: {
                "mean": statistics.mean(r[field] for r in chosen),
                "sample_std": statistics.stdev(r[field] for r in chosen),
            } for field in ("ndcg20", "recall20", "sparse_ndcg20", "tail_ndcg20", "unique_negative_pair_fraction", "weak_negative_fraction")}
    by_key = {(r["variant"], r["seed"]): r for r in rows}
    pairs = []
    for row in rows:
        if row["variant"] != "evidence0":
            continue
        for reference in ("uniform", "evidence05"):
            base = by_key[reference, row["seed"]]
            pairs.append({"seed": row["seed"], "reference": reference,
                          **{field + "_relative_percent": 100 * (row[field] / base[field] - 1)
                             for field in ("ndcg20", "recall20", "sparse_ndcg20")}})
    report = {"updated_utc": utc(), "split": "val", "comparison_horizon": config["training"]["epochs"],
              "completed_new_runs": sum(r["variant"] == "evidence0" for r in rows), "expected_new_runs": len(config["seeds"]),
              "rows": rows, "aggregates": aggregates, "paired_differences": pairs,
              "pareto_variants": rates.pareto_variants(aggregates),
              "extension_review": [{"seed": r["seed"], "best_epoch": r["best_epoch"]}
                                   for r in rows if r["variant"] == "evidence0" and r["late_best"]],
              "interpretation": config["interpretation"]}
    save_json(output / "comparison.json", report)
    flat = [{k: v for k, v in row.items() if k != "groups_at_best"} for row in rows]
    temporary = output / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    temporary.replace(output / "comparison.csv")
    lines = ["# 候选池来源：自动汇总", "", f"新实验完成 {report['completed_new_runs']}/{report['expected_new_runs']} 组。",
             "", "均为前 30 轮最佳 validation NDCG 模型的三种子均值；分组取同一轮。",
             "", "| 配置 | NDCG@20 | Recall@20 | 稀疏用户 NDCG@20 | 不重复负例对 | 弱证据负例占比 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, a in aggregates.items():
        lines.append(f"| {name} | {a['ndcg20']['mean']:.7f} | {a['recall20']['mean']:.7f} | {a['sparse_ndcg20']['mean']:.7f} | "
                     f"{a['unique_negative_pair_fraction']['mean']:.2%} | {a['weak_negative_fraction']['mean']:.2%} |")
    lines += ["", "evidence0 / evidence05：beta=0、80% 定向、20 候选，分别主动从证据池提出 0 / 50% 候选。",
              "evidence0 仍执行模型分数 argmax，随机候选可能碰到已有训练证据。uniform 为独立的随机负例对照。",
              "复用验证集和种子的探索性消融，不能解释为统计显著或测试收益。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_candidate_source_20261016.json")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/kuairand_stage1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/stage1_candidate_source_20261016")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or Path(sys.prefix).name != "gcn" or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Require gcn, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0 and exactly one CUDA device")
    gpu0 = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if ("GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")).lower() != gpu0.lower():
        raise RuntimeError("CUDA device does not match physical GPU 0")
    config = read_json(args.config)
    references = read_json(ROOT / config["reference_config"])
    if config["seeds"] != [2020, 2021, 2022] or config["training"] != {**references["training"], "evidence_fraction": 0.0}:
        raise ValueError("Only evidence_fraction may change from the frozen reference configuration")
    previous = read_json(ROOT / "runs/stage1_targeted_rate_20261016/plan.json")
    if code_hashes() != previous["code_sha256"] or sha256(args.dataset_dir / "manifest.json") != previous["dataset_sha256"]:
        raise ValueError("Training source or data differs from references")
    reference_hashes = {}
    for seed in config["seeds"]:
        for name, directory in references["reference_runs"][str(seed)].items():
            metadata = read_json(ROOT / directory / "metadata.json")
            if (metadata["environment"]["code_sha256"] != previous["code_sha256"]
                    or metadata["training_config"] != rates.expected_training(references, seed, 0.8, previous["dataset_sha256"], name == "uniform")):
                raise ValueError(f"Reference mismatch: {directory}")
            reference_hashes[directory] = {n: sha256(ROOT / directory / n) for n in ("metadata.json", "metrics.jsonl", "completion.json")}
    def fingerprints():
        return {str(p.relative_to(ROOT)): sha256(p) for folder in (rates.PILOT, rates.REPLICATION, RATE, Path(__file__).parent)
                for p in sorted(folder.glob("*.py"))}
    plan = {"config": config, "config_sha256": sha256(args.config), "reference_config_sha256": sha256(ROOT / config["reference_config"]),
            "code_sha256": code_hashes(), "driver_sha256": fingerprints(), "reference_sha256": reference_hashes,
            "dataset_sha256": previous["dataset_sha256"], "gpu0_uuid": gpu0, "python": sys.executable,
            "torch": str(torch.__version__), "test_evaluated": False}
    output = args.output_root.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".source_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if read_json(output / "plan.json") != plan:
            raise ValueError("Frozen source, data, references, configuration or GPU identity changed")
    else:
        save_json(output / "plan.json", plan)
        (output / "configs").mkdir()
        for seed in config["seeds"]:
            save_json(output / "configs" / f"seed{seed}.json", pilot_config(config, seed))
    state = {"status": "running", "started_utc": utc(), "queue_pid": os.getpid(), "completed_seeds": []}
    save_json(output / "status.json", state)
    child = None
    def stop(signum, frame):
        raise InterruptedError(f"Candidate-source queue interrupted: {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def verify():
        if (code_hashes() != plan["code_sha256"] or fingerprints() != plan["driver_sha256"]
                or sha256(args.config) != plan["config_sha256"]
                or sha256(ROOT / config["reference_config"]) != plan["reference_config_sha256"]
                or sha256(args.dataset_dir / "manifest.json") != plan["dataset_sha256"]):
            raise ValueError("Frozen source, configuration or dataset changed")
        for directory, hashes in reference_hashes.items():
            if any(sha256(ROOT / directory / n) != h for n, h in hashes.items()):
                raise ValueError("Reference files changed")
    try:
        summarize(output, config, references)
        for seed in config["seeds"]:
            verify()
            seed_root = output / f"seed{seed}"
            cfg_path = output / "configs" / f"seed{seed}.json"
            if read_json(cfg_path) != pilot_config(config, seed):
                raise ValueError("Per-seed configuration changed")
            state.update(active_seed=seed, updated_utc=utc())
            save_json(output / "status.json", state)
            finished = (seed_root / "status.json").exists() and read_json(seed_root / "status.json")["status"] == "completed"
            if not finished:
                command = [sys.executable, "-u", str(rates.PILOT / "run_pilot.py"), "--dataset-dir", str(args.dataset_dir.resolve()),
                           "--config", str(cfg_path), "--output-root", str(seed_root)]
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
                            report = summarize(output, config, references)
                            state.update(completed_new_runs=report["completed_new_runs"], updated_utc=utc())
                            save_json(output / "status.json", state)
                    child = None
                if returncode:
                    raise RuntimeError(f"Seed {seed} failed; inspect its log")
            verify()
            run = seed_root / f"hard-seed{seed}"
            completed, ck = read_json(run / "completion.json"), load_checkpoint(run / "last.pt")
            if (completed["last_sha256"] != sha256(run / "last.pt") or completed["test_evaluated"]
                    or ck["completed_epochs"] != config["training"]["epochs"] or ck["code_sha256"] != plan["code_sha256"]
                    or ck["training_config"] != rates.expected_training(config, seed, 0.8, plan["dataset_sha256"])):
                raise ValueError(f"Checkpoint mismatch: seed={seed}")
            state["completed_seeds"].append(seed)
            report = summarize(output, config, references)
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
    print(rates.json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
