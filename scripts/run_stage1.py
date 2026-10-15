"""Sequential reproducible A0--A3 suite; never evaluates test while training."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from experiment import code_hashes, load_checkpoint, make_parser
from research_common import read_json, sha256, write_json


def command(training, dataset_dir, output_root, method, seed, device, run_id=None, resume=None):
    result = [
        sys.executable,
        str(ROOT / "code" / "experiment.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--output-root",
        str(output_root),
        "--sampler",
        method,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    for key, value in training.items():
        result.extend(["--" + key.replace("_", "-"), str(value)])
    if resume:
        result.extend(["--resume", str(resume)])
    else:
        result.extend(["--run-id", run_id])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="One new suite directory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage1.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--methods", choices=["uniform", "hard", "reliable", "curriculum"], nargs="+"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seeds:
        config["seeds"] = args.seeds
    if args.methods:
        config["methods"] = args.methods
    if (
        not config["methods"]
        or not config["seeds"]
        or len(set(config["methods"])) != len(config["methods"])
        or len(set(config["seeds"])) != len(config["seeds"])
    ):
        raise ValueError("Suite methods and seeds must be nonempty and unique")
    allowed = {action.dest for action in make_parser()._actions} - {
        "resume",
        "eval_only",
        "checkpoint",
        "split",
        "run_id",
        "output_root",
        "dataset_dir",
        "sampler",
        "seed",
        "device",
        "tensorboard",
    }
    if not set(config["training"]) <= allowed:
        raise ValueError("Unknown or reserved configuration keys")
    dataset_dir = args.dataset_dir.resolve()
    plan = {
        "schema_version": 1,
        "config": config,
        "dataset_dir": str(dataset_dir),
        "dataset_sha256": sha256(dataset_dir / "manifest.json"),
        "code_sha256": code_hashes(),
    }
    plan_path = args.output_root / "suite.json"
    if args.resume:
        if read_json(plan_path) != plan:
            raise ValueError("Resume suite configuration/data changed; use the original config")
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
        write_json(plan_path, plan)
    for method in config["methods"]:
        for seed in config["seeds"]:
            run_id = f"{method}-seed{seed}"
            run_dir = args.output_root / run_id
            if (run_dir / "completion.json").exists():
                completion = read_json(run_dir / "completion.json")
                if completion["completed_epochs"] != config["training"]["epochs"] or completion[
                    "last_sha256"
                ] != sha256(run_dir / "last.pt"):
                    raise ValueError(f"Invalid completed run: {run_id}")
                print(f"SKIP completed {run_id}", flush=True)
                continue
            resume = run_dir / "last.pt" if (run_dir / "last.pt").exists() else None
            if run_dir.exists() and not resume:
                raise RuntimeError(
                    f"{run_dir} was interrupted before its first checkpoint. "
                    "Keep it for diagnosis and use a new suite directory."
                )
            if resume:
                load_checkpoint(resume)
            cmd = command(
                config["training"],
                dataset_dir,
                args.output_root,
                method,
                seed,
                args.device,
                run_id,
                resume,
            )
            print("START", run_id, flush=True)
            log = args.output_root / f"{run_id}.log"
            with log.open("a", encoding="utf-8") as handle:
                process = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"{run_id} failed with code {process.returncode}; inspect {log}")
            print("DONE", run_id, flush=True)
    print(
        "SUITE TRAINING COMPLETE. Test remains unevaluated; run evaluate_stage1.py after choosing the protocol.",
        flush=True,
    )


if __name__ == "__main__":
    main()
