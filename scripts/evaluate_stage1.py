"""Freeze final checkpoint choices before evaluating a completed suite on test."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from research_common import read_json, sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["last", "best_val"], default="last")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    suite = read_json(args.suite_dir / "suite.json")
    if sha256(Path(suite["dataset_dir"]) / "manifest.json") != suite["dataset_sha256"]:
        raise ValueError("Dataset fingerprint changed")
    checkpoints = {}
    for method in suite["config"]["methods"]:
        for seed in suite["config"]["seeds"]:
            name = f"{method}-seed{seed}"
            run = args.suite_dir / name
            completion = read_json(run / "completion.json")
            if completion["completed_epochs"] != suite["config"]["training"]["epochs"]:
                raise ValueError(f"Incomplete run: {name}")
            if completion["last_sha256"] != sha256(run / "last.pt"):
                raise ValueError(f"Last checkpoint changed: {name}")
            checkpoints[name] = sha256(run / f"{args.checkpoint}.pt")
    plan = {
        "checkpoint_choice": args.checkpoint,
        "dataset_sha256": suite["dataset_sha256"],
        "checkpoints": checkpoints,
    }
    plan_path = args.suite_dir / "final_test_plan.json"
    if plan_path.exists():
        if not args.resume or read_json(plan_path) != plan:
            raise ValueError(
                "Final test plan already exists. Resume the same plan; do not choose checkpoints using test results."
            )
    else:
        write_json(plan_path, plan)
    for name, digest in checkpoints.items():
        run = args.suite_dir / name
        evaluation_id = f"final-{args.checkpoint}"
        output_root = run / "evaluations"
        result_path = output_root / evaluation_id / "test_evaluation.json"
        if result_path.exists():
            if read_json(result_path)["checkpoint_sha256"] != digest:
                raise ValueError(f"Evaluation checksum changed: {name}")
            print("SKIP evaluated", name, flush=True)
            continue
        destination = output_root / evaluation_id
        if destination.exists():
            # A failed inference has no weights; retain diagnostics and use a unique retry folder.
            import uuid

            destination.rename(output_root / f"{evaluation_id}-failed-{uuid.uuid4().hex[:6]}")
        cmd = [
            sys.executable,
            str(ROOT / "code" / "experiment.py"),
            "--dataset-dir",
            suite["dataset_dir"],
            "--output-root",
            str(output_root),
            "--run-id",
            evaluation_id,
            "--eval-only",
            "--checkpoint",
            str(run / f"{args.checkpoint}.pt"),
            "--split",
            "test",
            "--device",
            args.device,
        ]
        with (run / "final_test.log").open("a", encoding="utf-8") as handle:
            subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print("TEST DONE", name, flush=True)
    print("FINAL TEST COMPLETE", flush=True)


if __name__ == "__main__":
    main()
