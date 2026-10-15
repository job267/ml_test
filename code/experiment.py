"""Stage-one research runner. Legacy main.py and model mathematics are unchanged."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
import uuid
import numpy as np
import scipy
import torch

from model import LightGCN, PureMF
from research_common import read_json, sha256, write_json
from research_data import FeedbackDataset
from research_evaluation import evaluate
from research_sampling import FeedbackSampler
from utils import BPRLoss, iter_batches, set_seed
from world import get_device, parse_topks

ROOT = Path(__file__).resolve().parents[1]
TRAINING_KEYS = (
    "model",
    "recdim",
    "layer",
    "lr",
    "decay",
    "seed",
    "sampler",
    "candidate_size",
    "evidence_fraction",
    "weak_confidence",
    "unknown_confidence",
    "reliability_beta",
    "temperature",
    "targeted_rate",
    "curriculum_epochs",
    "score_batch",
    "bpr_batch",
    "samples_per_epoch",
    "eval_interval",
    "topks",
)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "research")
    parser.add_argument("--run-id")
    parser.add_argument("--model", choices=["lgn", "mf"], default="lgn")
    parser.add_argument("--recdim", type=int, default=64)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--sampler", choices=["uniform", "hard", "reliable", "curriculum"], default="uniform"
    )
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--evidence-fraction", type=float, default=0.5)
    parser.add_argument("--weak-confidence", type=float, default=0.2)
    parser.add_argument("--unknown-confidence", type=float, default=0.05)
    parser.add_argument("--reliability-beta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--targeted-rate", type=float, default=0.8)
    parser.add_argument("--curriculum-epochs", type=int, default=100)
    parser.add_argument("--score-batch", type=int, default=8192)
    parser.add_argument("--bpr-batch", type=int, default=2048)
    parser.add_argument("--test-batch", type=int, default=100)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=0, help="0 means number of unique train pairs"
    )
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--topks", default="[20]")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser


def code_hashes():
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
        for folder in ("code", "scripts")
        for p in sorted((ROOT / folder).glob("*.py"))
    }


def environment(device):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = None
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "commit": commit or None,
        "code_sha256": code_hashes(),
    }


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume GPU count differs from checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") != 1:
        raise ValueError("Not a research checkpoint; legacy weights use legacy evaluation")
    return value


def build_model(config, dataset, device):
    """Adapt research CLI names to the unchanged baseline model configuration."""
    baseline_config = {
        "latent_dim_rec": config["recdim"],
        "lightGCN_n_layers": config["layer"],
        "dropout": 0,
        "keep_prob": 1.0,
        "lr": config["lr"],
        "decay": config["decay"],
    }
    model = (LightGCN if config["model"] == "lgn" else PureMF)(baseline_config, dataset).to(device)
    return model, baseline_config


def create_run(args):
    """Create one exclusive run directory; accidental reuse must fail."""
    run_id = (
        args.run_id
        or f"{args.sampler}-{args.seed}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    )
    if (
        not run_id
        or Path(run_id).name != run_id
        or run_id in (".", "..")
        or "\\" in run_id
        or "/" in run_id
    ):
        raise ValueError("run-id must be one directory name")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_one_epoch(model, loss, sampler, *, epoch, sample_count, batch_size, device):
    """Sample once, then update BPR per batch using fresh graph propagation."""
    triplets, sampler_stats = sampler.sample(model, epoch, sample_count or None)
    model.train()
    train_start = perf_counter()
    weighted_loss_sum = 0.0
    for batch in iter_batches(triplets, batch_size):
        users, positives, negatives = torch.as_tensor(
            batch, device=device, dtype=torch.long
        ).unbind(1)
        batch_loss = loss.stageOne(users, positives, negatives)
        if not np.isfinite(batch_loss):
            raise FloatingPointError("Non-finite training objective")
        weighted_loss_sum += batch_loss * len(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "epoch": epoch + 1,
        "train_total_loss": weighted_loss_sum / len(triplets),
        "train_seconds": perf_counter() - train_start,
        **sampler_stats,
    }


def run(argv=None):
    args = make_parser().parse_args(argv)
    if args.recdim < 1 or args.layer < 0 or args.epochs < 1 or args.lr <= 0 or args.decay < 0:
        raise ValueError("Invalid model or training parameters")
    if min(args.bpr_batch, args.test_batch, args.eval_interval) < 1 or args.samples_per_epoch < 0:
        raise ValueError("Invalid batch, interval or sample count")
    if args.resume and (args.eval_only or args.checkpoint or args.run_id):
        raise ValueError(
            "Resume identifies its existing run; do not combine with run-id/eval/checkpoint"
        )
    if args.eval_only != bool(args.checkpoint):
        raise ValueError("--eval-only and --checkpoint must be supplied together")
    device = get_device(args.device)
    set_seed(args.seed)
    config = {k: getattr(args, k) for k in TRAINING_KEYS}
    config["topks"] = parse_topks(args.topks)
    checkpoint = (
        load_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    )
    if args.eval_only:
        config = dict(checkpoint["training_config"])
    dataset = FeedbackDataset(args.dataset_dir, device, config["weak_confidence"])
    config["dataset_sha256"] = dataset.fingerprint
    if checkpoint and checkpoint["training_config"] != config:
        raise ValueError("Checkpoint training configuration or data fingerprint differs")
    if args.eval_only and checkpoint.get("code_sha256") != code_hashes():
        raise ValueError(
            "Evaluation source differs from checkpoint; use its original source package"
        )
    model, baseline_config = build_model(config, dataset, device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if args.eval_only:
        run_dir = create_run(args)
        result = evaluate(dataset, model, args.split, config["topks"], args.test_batch)
        result.update(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256(args.checkpoint),
                "completed_epochs": checkpoint["completed_epochs"],
                "training_config": config,
                "environment": environment(device),
            }
        )
        write_json(run_dir / f"{args.split}_evaluation.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return run_dir
    sampler = FeedbackSampler(
        dataset=dataset,
        method=args.sampler,
        seed=args.seed,
        candidate_size=args.candidate_size,
        evidence_fraction=args.evidence_fraction,
        unknown_confidence=args.unknown_confidence,
        reliability_beta=args.reliability_beta,
        temperature=args.temperature,
        targeted_rate=args.targeted_rate,
        curriculum_epochs=args.curriculum_epochs,
        score_batch=args.score_batch,
    )
    loss = BPRLoss(model, baseline_config)
    start_epoch, best_ndcg, best_epoch = 0, -float("inf"), 0
    current_env = environment(device)
    if args.resume:
        if args.resume.name != "last.pt":
            raise ValueError("Resume requires last.pt; best_val.pt is for evaluation")
        run_dir = args.resume.resolve().parent
        metadata = read_json(run_dir / "metadata.json")
        if metadata["environment"]["code_sha256"] != current_env["code_sha256"]:
            raise ValueError("Source code changed since this run; start a separate experiment")
        loss.opt.load_state_dict(checkpoint["optimizer"])
        # Optimizer tensors must follow model device after loading a CPU checkpoint.
        for state in loss.opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        sampler.load_state_dict(checkpoint["sampler"])
        restore_rng(checkpoint["rng"])
        start_epoch, best_ndcg, best_epoch = (
            checkpoint["completed_epochs"],
            checkpoint["best_ndcg"],
            checkpoint["best_epoch"],
        )
        if args.epochs <= start_epoch:
            raise ValueError("Requested epochs must exceed the completed checkpoint epoch")
        # Truncate metrics written after the latest committed checkpoint (e.g. interrupted save).
        metrics_path = run_dir / "metrics.jsonl"
        lines = (
            [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            if metrics_path.exists()
            else []
        )
        lines = [row for row in lines if row["epoch"] < start_epoch]
        lines.append(checkpoint["metrics_row"])
        metrics_path.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
        (run_dir / "completion.json").unlink(missing_ok=True)
    else:
        run_dir = create_run(args)
        write_json(
            run_dir / "metadata.json",
            {
                "training_config": config,
                "requested_epochs": args.epochs,
                "environment": current_env,
                "data_audit": dataset.manifest["audit"],
                "protocol": dataset.manifest["protocol"],
                "test_during_training": False,
            },
        )
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=str(run_dir / "tensorboard"),
            purge_step=start_epoch + 1 if args.resume else None,
        )
    with (run_dir / "launches.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "resume": bool(args.resume),
                    "start_epoch": start_epoch,
                    "requested_epochs": args.epochs,
                    "environment": current_env,
                }
            )
            + "\n"
        )
    try:
        for epoch in range(start_epoch, args.epochs):
            begin = perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            row = train_one_epoch(
                model,
                loss,
                sampler,
                epoch=epoch,
                sample_count=args.samples_per_epoch,
                batch_size=args.bpr_batch,
                device=device,
            )
            validation = None
            if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
                eval_start = perf_counter()
                validation = evaluate(dataset, model, "val", config["topks"], args.test_batch)
                row["validation"] = validation
                row["validation_seconds"] = perf_counter() - eval_start
            row["epoch_seconds"] = perf_counter() - begin
            row["samples_per_second"] = row["samples"] / max(
                row["train_seconds"] + row["sampler_seconds"], 1e-9
            )
            row["peak_cuda_mib"] = (
                torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
            )
            improved = validation and validation["groups"]["overall"]["ndcg"][0] > best_ndcg
            if improved:
                best_ndcg = validation["groups"]["overall"]["ndcg"][0]
                best_epoch = epoch + 1
            value = {
                "schema_version": 1,
                "training_config": config,
                "code_sha256": current_env["code_sha256"],
                "completed_epochs": epoch + 1,
                "model": model.state_dict(),
                "optimizer": loss.opt.state_dict(),
                "sampler": sampler.state_dict(),
                "rng": rng_state(),
                "best_ndcg": best_ndcg,
                "best_epoch": best_epoch,
                "metrics_row": row,
            }
            if improved:
                save_checkpoint(run_dir / "best_val.pt", value)
            save_checkpoint(run_dir / "last.pt", value)
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if writer:
                writer.add_scalar("train/total_loss", row["train_total_loss"], epoch + 1)
                writer.add_scalar("sampler/targeted_rate", row["targeted_rate"], epoch + 1)
                if validation:
                    for index, k in enumerate(config["topks"]):
                        for metric in ("recall", "precision", "ndcg"):
                            writer.add_scalar(
                                f"val/{metric}@{k}",
                                validation["groups"]["overall"][metric][index],
                                epoch + 1,
                            )
            print("EPOCH", json.dumps(row, ensure_ascii=False), flush=True)
        write_json(
            run_dir / "completion.json",
            {
                "completed_epochs": args.epochs,
                "best_val_epoch": best_epoch,
                "best_val_ndcg": best_ndcg,
                "last_sha256": sha256(run_dir / "last.pt"),
                "test_evaluated": False,
            },
        )
    finally:
        if writer:
            writer.close()
    return run_dir


if __name__ == "__main__":
    run()
