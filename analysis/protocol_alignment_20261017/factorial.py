"""Train two label models on a shared graph vocabulary; evaluate four views."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import CONFIG, HERE, ROOT, OUTPUT, checkpoint, gpu_guard, history, read, save, sha, utc
from dataset import factorial_dataset
from evaluation import views
from alignment import baseline_config
from experiment import code_hashes, restore_rng, rng_state, train_one_epoch
from model import LightGCN
from research_sampling import FeedbackSampler
from utils import BPRLoss, set_seed


def driver_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob("*.py"))}


def study_identity(label, data_manifest, training_config):
    return {"schema_version": "label-source-training-v1", "label": label,
            "config_sha256": sha(CONFIG), "data_manifest_sha256": sha(data_manifest),
            "dataset_sha256": sha(data_manifest.parent / f"common_{label}.npz"),
            "driver_sha256": driver_hashes(), "core_sha256": code_hashes(), "training_configuration": training_config}


def run(label, stop_after, resume=False, *, config=None, output_root=OUTPUT, device="cuda"):
    # CLI always uses GPU 0. Explicit CPU injection is for miniature regression tests.
    c = config or read(CONFIG)["factorial"]
    if label not in c["positive_rules"] or stop_after not in (c["initial_epochs"], c["maximum_epochs"]):
        raise ValueError("Unplanned label/horizon")
    gpu = gpu_guard() if device == "cuda" else {"device": "cpu", "torch": str(torch.__version__)}
    output = Path(output_root) / "factorial" / label
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = Path(output_root) / "data/manifest.json"
    manifest = read(manifest_path)
    for name, expected in manifest["files"].items():
        if sha(manifest_path.parent / name) != expected:
            raise ValueError("Prepared diagnosis file changed: " + name)
    identity = study_identity(label, manifest_path, c)
    if resume:
        if read(output / "metadata.json")["identity"] != identity:
            raise ValueError("Study training identity changed")
    else:
        save(output / "metadata.json", {"identity": identity, "environment": gpu,
             "configuration": c, "samples_per_epoch": manifest["samples_per_epoch"],
             "selection": c["secondary_selection"], "primary": c["primary_evaluation"],
             "evidence": "Uniform sampling; confidence is not consumed by selection or the BPR objective. No negative-evidence weighting is tested here.",
             "test_evaluated": False, "created_utc": utc()})
    ds = factorial_dataset(manifest_path.parent / f"common_{label}.npz", device)
    set_seed(c["seed"])
    net = LightGCN(baseline_config(c), ds).to(device)
    loss = BPRLoss(net, baseline_config(c))
    sampler = FeedbackSampler(ds, method="uniform", seed=c["seed"])
    start, best, best_epoch = 0, -float("inf"), 0
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint differs from frozen study identity")
        net.load_state_dict(state["model"])
        loss.opt.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        restore_rng(state["rng"])
        start, best, best_epoch = state["completed_epochs"], state["best_ndcg"], state["best_epoch"]
        committed = [r for r in history(output / "metrics.jsonl") if r["epoch"] < start] + [state["metrics_row"]]
        (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in committed))
    for epoch in range(start, stop_after):
        if driver_hashes() != identity["driver_sha256"] or code_hashes() != identity["core_sha256"] or sha(CONFIG) != identity["config_sha256"]:
            raise ValueError("Training source/config changed during the run")
        begin = time.perf_counter()
        row = train_one_epoch(net, loss, sampler, epoch=epoch, sample_count=manifest["samples_per_epoch"],
                              batch_size=c["batch_size"], device=torch.device(device))
        eval_start = time.perf_counter()
        validation, users, ranks = views(net, ds, c["eval_batch"])
        row["validation_views"] = validation
        row["validation_seconds"] = time.perf_counter() - eval_start
        score = validation["random"]["common_users"]["groups"]["overall"]["ndcg"][0]
        improved = score > best
        if improved:
            best, best_epoch = score, epoch + 1
        row.update(epoch_seconds=time.perf_counter() - begin, best_epoch=best_epoch,
                   best_common_random_ndcg20=best, samples_per_epoch=manifest["samples_per_epoch"])
        rank_path = output / f"validation_ranks_epoch{epoch + 1:03d}.npz"
        tmp = rank_path.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, users=np.asarray(users, dtype=np.int64), ranks=ranks.astype(np.int32))
        tmp.replace(rank_path)
        row["validation_ranks_sha256"] = sha(rank_path)
        state = {"schema_version": "label-source-checkpoint-v1", "identity": identity,
                 "completed_epochs": epoch + 1, "model": net.state_dict(), "optimizer": loss.opt.state_dict(),
                 "sampler": sampler.state_dict(), "rng": rng_state(), "best_ndcg": best,
                 "best_epoch": best_epoch, "metrics_row": row}
        if improved:
            checkpoint(output / "best_val.pt", state)
        checkpoint(output / "last.pt", state)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        save(output / "status.json", {"status": "running", "completed_epochs": epoch + 1,
             "best_epoch": best_epoch, "updated_utc": utc(), "test_evaluated": False})
        print("FACTORIAL_EPOCH", json.dumps(row), flush=True)
    committed_epochs = max(start, stop_after)
    save(output / "status.json", {"status": "staged", "completed_epochs": committed_epochs,
         "best_epoch": best_epoch, "updated_utc": utc(), "test_evaluated": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", choices=["engagement", "click"], required=True)
    p.add_argument("--stop-after", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args.label, args.stop_after, args.resume)
