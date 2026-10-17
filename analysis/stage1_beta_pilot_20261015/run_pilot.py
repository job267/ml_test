"""Run the frozen beta pilot sequentially, using the original training sources."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from experiment import code_hashes, load_checkpoint
from research_common import read_json, sha256


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(output, config):
    """Summarize validation only; the same selected row supplies all group metrics."""
    rows = []
    for variant in config["variants"]:
        run_id = f'{variant["id"]}-seed{config["seed"]}'
        path = output / run_id / "metrics.jsonl"
        history = []
        if path.exists():
            # A concurrently written final line may be incomplete. Never rewrite the training log.
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if "validation" in row:
                    history.append(row)
        if not history:
            continue
        best = max(history, key=lambda row: row["validation"]["groups"]["overall"]["ndcg"][0])
        last = history[-1]
        groups = best["validation"]["groups"]
        rows.append({
            "variant": variant["id"], "seed": config["seed"],
            "completed_epochs": last["epoch"], "best_epoch": best["epoch"],
            "best_val_ndcg20": groups["overall"]["ndcg"][0],
            "best_val_recall20": groups["overall"]["recall"][0],
            "best_sparse_ndcg20": groups["user_sparse"]["ndcg"][0],
            "best_tail_ndcg20": groups["item_tail"]["ndcg"][0],
            "last_val_ndcg20": last["validation"]["groups"]["overall"]["ndcg"][0],
            "mean_negative_evidence_at_best": best["mean_negative_evidence"],
            "explicit_negative_fraction_at_best": best["explicit_negative_fraction"],
            "unique_pair_fraction_at_best": best["unique_user_negative_pairs"] / best["samples"],
            "training_seconds": sum(row["train_seconds"] for row in history),
            "sampling_seconds": sum(row["sampler_seconds"] for row in history),
            "validation_seconds": sum(row["validation_seconds"] for row in history),
        })
    atomic_json(output / "validation_summary.json", {
        "updated_utc": utc(), "split": "val", "rows": rows,
        "interpretation": "Single-seed validation diagnosis; no test evaluation or significance claim.",
    })
    if rows:
        temporary = output / "validation_summary.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "validation_summary.csv")
    return rows


def gpu_snapshot(output):
    command = ["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "GPU 0 monitoring failed")
    with (output / "gpu0.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "gpu0": result.stdout.strip()}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/kuairand_stage1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_beta_pilot.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/stage1_beta_pilot_20261015")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This pilot requires CUDA_VISIBLE_DEVICES=0")
    if Path(sys.prefix).name != "gcn":
        raise RuntimeError("Run this pilot in the existing gcn conda environment")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device (physical GPU 0) is required")
    output, dataset_dir = args.output_root.resolve(), args.dataset_dir.resolve()
    config = read_json(args.config)
    if not config["variants"] or len({v["id"] for v in config["variants"]}) != len(config["variants"]):
        raise ValueError("Pilot variants must be nonempty and unique")
    plan = {
        "schema_version": 1, "config": config,
        "config_sha256": sha256(args.config),
        "dataset_dir": str(dataset_dir), "dataset_sha256": sha256(dataset_dir / "manifest.json"),
        "code_sha256": code_hashes(),
        "driver_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "python": sys.executable, "torch": str(torch.__version__),
        "cuda_visible_devices": "0", "gpu": torch.cuda.get_device_name(0),
        "cpu_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "test_during_training": False,
    }
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if read_json(output / "pilot_plan.json") != plan:
            raise ValueError("Pilot source, data, configuration or runtime changed")
    else:
        atomic_json(output / "pilot_plan.json", plan)
        # Keep a runnable original source copy alongside the independent new experiment.
        for name in plan["code_sha256"]:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    state = {"started_utc": utc(), "queue_pid": os.getpid(), "status": "running", "completed": []}
    atomic_json(output / "status.json", state)
    child = None

    def stop(signum, frame):
        raise InterruptedError(f"Pilot interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def verify():
        if code_hashes() != plan["code_sha256"] or sha256(args.config) != plan["config_sha256"]:
            raise ValueError("Training code or pilot config changed during queue")
        if sha256(dataset_dir / "manifest.json") != plan["dataset_sha256"]:
            raise ValueError("Prepared manifest changed during queue")
        if {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))} != plan["driver_sha256"]:
            raise ValueError("Pilot driver changed during queue")

    try:
        initial_scores = output / "initial_score_diagnostics.json"
        if not initial_scores.exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("diagnose_scores.py")),
                            "--dataset-dir", str(dataset_dir), "--config", str(args.config.resolve()),
                            "--output", str(initial_scores)], cwd=ROOT, check=True)
        for variant in config["variants"]:
            verify()
            run_id = f'{variant["id"]}-seed{config["seed"]}'
            run_dir = output / run_id
            completion_path = run_dir / "completion.json"
            state.update(active_run=run_id, updated_utc=utc())
            atomic_json(output / "status.json", state)
            if not completion_path.exists():
                command = [sys.executable, str(ROOT / "code/experiment.py"),
                           "--dataset-dir", str(dataset_dir), "--output-root", str(output),
                           "--device", "cuda", "--seed", str(config["seed"]),
                           "--sampler", variant["sampler"], "--reliability-beta", str(variant["reliability_beta"])]
                for key, value in config["training"].items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                if run_dir.exists():
                    if not args.resume or not (run_dir / "last.pt").exists():
                        raise RuntimeError(f"Incomplete run has no resumable checkpoint: {run_dir}")
                    command.extend(["--resume", str(run_dir / "last.pt")])
                else:
                    command.extend(["--run-id", run_id])
                gpu_snapshot(output)
                print(json.dumps({"utc": utc(), "launch": command}), flush=True)
                with (output / f"{run_id}.log").open("a", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state["child_pid"] = child.pid
                    atomic_json(output / "status.json", state)
                    while True:
                        try:
                            returncode = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            gpu_snapshot(output)
                            summarize(output, config)
                            state["updated_utc"] = utc()
                            atomic_json(output / "status.json", state)
                    child = None
                if returncode:
                    raise RuntimeError(f"Training failed: {run_id}, exit={returncode}; see its log")
            verify()
            completion = read_json(completion_path)
            checkpoint = load_checkpoint(run_dir / "last.pt")
            if (completion["completed_epochs"] != config["training"]["epochs"]
                    or checkpoint["completed_epochs"] != config["training"]["epochs"]
                    or completion["last_sha256"] != sha256(run_dir / "last.pt")
                    or checkpoint["code_sha256"] != plan["code_sha256"]
                    or checkpoint["training_config"]["dataset_sha256"] != plan["dataset_sha256"]):
                raise ValueError(f"Completed run integrity mismatch: {run_id}")
            state["completed"].append(run_id)
            summarize(output, config)
            diagnostics = run_dir / "best_score_diagnostics.json"
            if not diagnostics.exists():
                subprocess.run([sys.executable, str(Path(__file__).with_name("diagnose_scores.py")),
                                "--dataset-dir", str(dataset_dir), "--config", str(args.config.resolve()),
                                "--checkpoint", str(run_dir / "best_val.pt"),
                                "--output", str(diagnostics)], cwd=ROOT, check=True)
        state.update(status="completed", active_run=None, child_pid=None, completed_utc=utc())
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(status="failed", error=repr(error), failed_utc=utc(), child_pid=None)
        atomic_json(output / "status.json", state)
        raise
    atomic_json(output / "status.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
