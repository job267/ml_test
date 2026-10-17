"""Change only the targeted fraction, retaining the original training and pilot driver."""

import argparse
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "analysis/stage1_beta_pilot_20261015"
REPLICATION = ROOT / "analysis/stage1_beta_replication_20261016"
sys.path.insert(0, str(ROOT / "code"))
from experiment import TRAINING_KEYS, code_hashes, load_checkpoint
from research_common import read_json, sha256
from world import parse_topks

spec = importlib.util.spec_from_file_location("frozen_replication", REPLICATION / "replicate.py")
replication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replication)
save_json, utc, window_result = replication.save_json, replication.utc, replication.window_result


def variant_id(rate):
    return {0.2: "rate02", 0.4: "rate04"}[rate]


def job_config(config, seed, rate):
    if rate not in (0.2, 0.4) or seed not in config["seeds"]:
        raise ValueError("Unexpected targeted rate or seed")
    training = {**config["training"], "targeted_rate": rate}
    return {"schema_version": 1, "seed": seed, "training": training,
            "variants": [{"id": "hard", "sampler": "reliable", "reliability_beta": 0.0}],
            "selection": config["selection"], "followup": config["interpretation"]}


def expected_training(config, seed, rate, dataset_sha256, uniform=False):
    values = {**config["training"], "seed": seed, "targeted_rate": rate,
              "sampler": "uniform" if uniform else "reliable",
              "reliability_beta": 1.0 if uniform else 0.0}
    values["topks"] = parse_topks(values["topks"])
    return {**{key: values[key] for key in TRAINING_KEYS}, "dataset_sha256": dataset_sha256}


def jobs(config, output):
    for index, seed in enumerate(config["seeds"]):
        # Reverse the order for the second seed to reduce a fixed execution-order association.
        for rate in config["rates"][::(-1 if index % 2 else 1)]:
            name = f"{variant_id(rate)}-seed{seed}"
            yield {"id": name, "seed": seed, "rate": rate,
                   "root": output / "jobs" / name,
                   "config": output / "configs" / f"{name}.json"}


def pareto_variants(aggregates):
    """Report the two objectives separately, without hiding subgroup losses in a score."""
    result = []
    for name, a in aggregates.items():
        dominated = any(
            b["ndcg20"]["mean"] >= a["ndcg20"]["mean"]
            and b["sparse_ndcg20"]["mean"] >= a["sparse_ndcg20"]["mean"]
            and (b["ndcg20"]["mean"] > a["ndcg20"]["mean"]
                 or b["sparse_ndcg20"]["mean"] > a["sparse_ndcg20"]["mean"])
            for other, b in aggregates.items() if other != name)
        if not dominated:
            result.append(name)
    return result


def summarize(output, config):
    rows = []
    locations = [(seed, name, {"uniform": 0.0, "rate08": 0.8}[name], ROOT / run, "existing_reference")
                 for seed in config["seeds"] for name, run in config["reference_runs"][str(seed)].items()]
    locations += [(j["seed"], variant_id(j["rate"]), j["rate"],
                   j["root"] / f'hard-seed{j["seed"]}', "new_rate_ablation") for j in jobs(config, output)]
    for seed, variant, rate, run, role in locations:
        path = run / "metrics.jsonl"
        if not path.exists():
            continue
        history = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                break
        result = window_result(history, config["training"]["epochs"])
        if result is not None:
            rows.append({"seed": seed, "variant": variant, "targeted_rate": rate,
                         "role": role, **result})
    aggregates = {}
    for variant in ("uniform", "rate02", "rate04", "rate08"):
        chosen = [r for r in rows if r["variant"] == variant]
        if {r["seed"] for r in chosen} != set(config["seeds"]):
            continue
        aggregates[variant] = {field: {
            "mean": statistics.mean(r[field] for r in chosen),
            "sample_std": statistics.stdev(r[field] for r in chosen),
        } for field in ("ndcg20", "recall20", "sparse_ndcg20", "tail_ndcg20", "unique_negative_pair_fraction")}
    paired = []
    by_key = {(r["variant"], r["seed"]): r for r in rows}
    for row in rows:
        if row["role"] != "new_rate_ablation":
            continue
        for reference in ("uniform", "rate08"):
            b = by_key[reference, row["seed"]]
            paired.append({"seed": row["seed"], "variant": row["variant"], "reference": reference,
                           **{field + "_relative_percent": 100 * (row[field] / b[field] - 1)
                              for field in ("ndcg20", "recall20", "sparse_ndcg20")}})
    report = {"updated_utc": utc(), "split": "val", "comparison_horizon": config["training"]["epochs"],
              "completed_new_runs": sum(r["role"] == "new_rate_ablation" for r in rows),
              "expected_new_runs": len(config["seeds"]) * len(config["rates"]),
              "rows": rows, "aggregates": aggregates, "paired_differences": paired,
              "pareto_variants": pareto_variants(aggregates),
              "extension_review": [{"seed": r["seed"], "variant": r["variant"], "best_epoch": r["best_epoch"]}
                                   for r in rows if r["role"] == "new_rate_ablation" and r["late_best"]],
              "interpretation": config["interpretation"]}
    save_json(output / "comparison.json", report)
    flat = [{k: v for k, v in r.items() if k != "groups_at_best"} for r in rows]
    temporary = output / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    temporary.replace(output / "comparison.csv")
    lines = ["# 定向比例消融：自动汇总", "", f"新实验完成 {report['completed_new_runs']}/{report['expected_new_runs']} 组。",
             "", "均为前 30 轮中最佳 validation NDCG 对应模型的三种子均值；分组取同一轮。",
             "", "| 配置 | NDCG@20 | Recall@20 | 稀疏用户 NDCG@20 | 不重复负例对比例 |",
             "|---|---:|---:|---:|---:|"]
    for variant, values in aggregates.items():
        lines.append(f"| {variant} | {values['ndcg20']['mean']:.7f} | {values['recall20']['mean']:.7f} | "
                     f"{values['sparse_ndcg20']['mean']:.7f} | {values['unique_negative_pair_fraction']['mean']:.2%} |")
    lines += ["", "本轮只改变 beta=0 的定向比例；候选池仍有一半来自训练证据。",
              "复用过的种子与验证集上的探索性消融，不代表独立确认、统计显著或测试集收益。",
              "整体与稀疏用户指标分别报告；不可用总分掩盖分组损失。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_targeted_rate_20261016.json")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/kuairand_stage1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/stage1_targeted_rate_20261016")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or Path(sys.prefix).name != "gcn" or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Require existing gcn, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0 and one CUDA device")
    gpu0 = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if ("GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")).lower() != gpu0.lower():
        raise RuntimeError("CUDA device does not match physical GPU 0")
    config = read_json(args.config)
    if config["seeds"] != [2020, 2021, 2022] or config["rates"] != [0.2, 0.4]:
        raise ValueError("Unexpected rate ablation")
    previous = read_json(ROOT / "runs/stage1_beta_replication_20261016/replication_plan.json")
    if config["training"] != previous["config"]["training"]:
        raise ValueError("Base training configuration differs from the completed replication")
    if code_hashes() != previous["code_sha256"] or sha256(args.dataset_dir / "manifest.json") != previous["dataset_sha256"]:
        raise ValueError("Original training source or prepared data differs")
    reference_hashes = {}
    for seed in config["seeds"]:
        for name, directory in config["reference_runs"][str(seed)].items():
            run = ROOT / directory
            metadata = read_json(run / "metadata.json")
            training = metadata["training_config"]
            if (metadata["environment"]["code_sha256"] != previous["code_sha256"]
                    or training != expected_training(config, seed, 0.8, previous["dataset_sha256"], name == "uniform")):
                raise ValueError(f"Reference mismatch: {run}")
            reference_hashes[directory] = {n: sha256(run / n) for n in ("metadata.json", "metrics.jsonl", "completion.json")}
    output = args.output_root.resolve()
    def fingerprints():
        return {str(p.relative_to(ROOT)): sha256(p) for folder in (PILOT, REPLICATION, Path(__file__).parent)
                for p in sorted(folder.glob("*.py"))}
    plan = {"config": config, "config_sha256": sha256(args.config), "code_sha256": code_hashes(),
            "driver_sha256": fingerprints(), "reference_sha256": reference_hashes,
            "dataset_sha256": previous["dataset_sha256"], "gpu0_uuid": gpu0,
            "python": sys.executable, "torch": str(torch.__version__), "test_evaluated": False}
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".rate_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if read_json(output / "plan.json") != plan:
            raise ValueError("Source, data, reference, configuration or GPU changed")
    else:
        save_json(output / "plan.json", plan)
        (output / "configs").mkdir()
        for job in jobs(config, output):
            save_json(job["config"], job_config(config, job["seed"], job["rate"]))
    state = {"status": "running", "started_utc": utc(), "queue_pid": os.getpid(), "completed_jobs": []}
    save_json(output / "status.json", state)
    child = None
    def stop(signum, frame):
        raise InterruptedError(f"Rate queue interrupted: signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def verify():
        if (code_hashes() != plan["code_sha256"] or fingerprints() != plan["driver_sha256"]
                or sha256(args.config) != plan["config_sha256"]
                or sha256(args.dataset_dir / "manifest.json") != plan["dataset_sha256"]):
            raise ValueError("Rate queue source, config or data changed")
        for directory, hashes in reference_hashes.items():
            if any(sha256(ROOT / directory / n) != h for n, h in hashes.items()):
                raise ValueError("Reference files changed")
    try:
        summarize(output, config)
        for job in jobs(config, output):
            verify()
            if read_json(job["config"]) != job_config(config, job["seed"], job["rate"]):
                raise ValueError("Per-job config changed")
            state.update(active_job=job["id"], updated_utc=utc())
            save_json(output / "status.json", state)
            finished = (job["root"] / "status.json").exists() and read_json(job["root"] / "status.json")["status"] == "completed"
            if not finished:
                command = [sys.executable, "-u", str(PILOT / "run_pilot.py"), "--dataset-dir", str(args.dataset_dir.resolve()),
                           "--config", str(job["config"]), "--output-root", str(job["root"])]
                if job["root"].exists():
                    if not args.resume:
                        raise FileExistsError(job["root"])
                    command.append("--resume")
                job["root"].parent.mkdir(exist_ok=True)
                with (output / f'{job["id"]}.log').open("a", encoding="utf-8") as log:
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
                    raise RuntimeError(f'Job failed: {job["id"]}; see its log')
            verify()
            run = job["root"] / f'hard-seed{job["seed"]}'
            completed = read_json(run / "completion.json")
            ck = load_checkpoint(run / "last.pt")
            if (completed["last_sha256"] != sha256(run / "last.pt") or completed["test_evaluated"]
                    or ck["completed_epochs"] != config["training"]["epochs"] or ck["code_sha256"] != plan["code_sha256"]
                    or ck["training_config"] != expected_training(config, job["seed"], job["rate"], plan["dataset_sha256"])):
                raise ValueError(f'Job checkpoint mismatch: {job["id"]}')
            state["completed_jobs"].append(job["id"])
            report = summarize(output, config)
            state["completed_new_runs"] = report["completed_new_runs"]
        state.update(status="completed", completed_utc=utc(), active_job=None, child_pid=None,
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
