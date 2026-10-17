"""Run pinned author LightGCN with durable checkpoints and deferred test evaluation.

The author model, sampler, graph, evaluation and rounded Recall@10 selection are
unchanged. The author main loop is transcribed to add audit/resume and postpone
test calls. PyTorch 2.6 in the user's existing gcn replaces the author's 1.13.
"""
import argparse
import fcntl
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from shared import ROOT, HERE, digest, gpu_guard, history, read, save, utc
from public_protocol import VENDOR
from public_evaluation import evaluate_public_test


def author_stop(epoch_index, recall, best, count):
    """Exact author patience update, including rounded-metric tie handling."""
    if epoch_index > 100:
        count = count + 1 if recall < best else 1
    return count, count >= 20


def capture_rng():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(rng):
    random.setstate(rng["python"])
    n = rng["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(rng["torch"].cpu())
    if len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA device count differs")
    if rng["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])


def checkpoint(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stop-after", type=int, default=600)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.seed not in (2018, 2019, 2020, 2021, 2022) or not 1 <= args.stop_after <= 600:
        raise ValueError("Unplanned public run")
    gpu = gpu_guard()
    output = args.output.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".public.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(output.parent / "protocol_audit.json")
    for name, sha in protocol["source_and_data_sha256"].items():
        if digest(VENDOR / name) != sha:
            raise ValueError("Public input differs from frozen audit: " + name)
    plan = {"seed": args.seed, "environment": gpu, "protocol_audit_sha256": digest(output.parent / "protocol_audit.json"),
            "author_commit": protocol["commit"], "driver_sha256": {n: digest(HERE / n) for n in
                ("shared.py", "public_run.py", "public_protocol.py", "public_evaluation.py")},
            "selection": "maximum rounded validation Recall@10, strict improvement, every 5 epochs",
            "max_epochs": 600, "patience": "author count>=20 evaluated only after zero-based epoch>100",
            "configuration": read(ROOT / "configs/baseline_calibration_20261016.json")["public"],
            "changes_from_author_main": ["atomic model/optimizer/RNG checkpoints and structured logs",
                "test one scoring pass after stopping using the validation-selected checkpoint",
                "predeclare both original author test mask and corrected user-aligned test mask; no test-based choice",
                "omit unused retained list of training batches and accumulate scalar logging loss",
                "use existing gcn Python/PyTorch, no tensorboardX import when tensorboard disabled"],
            "current_protocol_test_evaluated": False}
    if args.resume:
        if read(output / "plan.json") != plan:
            raise ValueError("Public reproduction plan changed")
    else:
        save(output / "plan.json", plan)
    if (output / "completion.json").exists():
        return
    # Author modules use global names world/model/utils; isolate them in this child.
    os.chdir(output)
    sys.path = [str(VENDOR), *[x for x in sys.path if x != str(ROOT / "code")]]
    sys.argv = [str(VENDOR / "main.py"), "--data_path", str(VENDOR / "kuairand"),
                "--dataset", "kuairand", "--model", "lgn", "--algo", "vanilla",
                "--epochs", "600", "--decay", "0.0001", "--lr", "0.001",
                "--recdim", "64", "--layer", "3", "--batch_size", "2048",
                "--topks", "[10,20,50,100]", "--testbatch", "2048", "--gpu", "0",
                "--seed", str(args.seed), "--log_file", "./author.log", "--tensorboard", "0"]
    import world
    import utils
    import dataloader
    import register
    import Procedure
    from model import PopPredictor
    utils.set_seed(world.seed)
    ds = dataloader.DisenData(path=str(VENDOR / "kuairand"))
    loader = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True, num_workers=4)
    net = register.MODELS[world.model_name](world.config, ds).to(world.device)
    predictor = PopPredictor().to(world.device)
    start, best, best_epoch, count, stopped = 0, 0., 0, 0, False
    if args.resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if state["plan"] != plan:
            raise ValueError("Public checkpoint plan differs")
        net.load_state_dict(state["model"])
        net.opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        start, best, best_epoch, count, stopped = [state[k] for k in
            ("completed_epochs", "best_recall10", "best_epoch", "count", "stopped")]
        rows = [r for r in history(output / "metrics.jsonl") if r["epoch"] < start]
        rows.append(state["metrics_row"])
        (output / "metrics.jsonl").write_text("".join(__import__('json').dumps(r) + "\n" for r in rows))
    save(output / "status.json", {"status": "running", "completed_epochs": start, "updated_utc": utc()})
    for epoch in range(start, args.stop_after):
        if stopped:
            break
        begin = time.perf_counter()
        torch.cuda.empty_cache()
        net.train()
        loader.dataset.get_pair_bpr()
        total, batches = 0., 0
        for x in loader:
            u, pos, neg, stage = [v.to(world.device) for v in x[:4]]
            pos_inter = torch.stack(x[4]).to(world.device)
            neg_inter = torch.stack(x[5]).to(world.device)
            loss, _ = net.bpr_loss(u, pos, neg, stage, pos_inter, neg_inter, predictor, world)
            loss = torch.mean(loss)
            value = loss.item()
            if not np.isfinite(value):
                raise FloatingPointError("Non-finite public objective")
            total += value
            batches += 1
            loss.backward()
            net.opt.step()
            net.opt.zero_grad()
        row = {"epoch": epoch + 1, "loss_mean_over_batches": total / batches,
               "samples": len(ds), "training_seconds": time.perf_counter() - begin}
        if (epoch + 1) % 5 == 0:
            result = Procedure.Test(ds, net, predictor, epoch, None, 0, 0)
            row["validation"] = {"topks": [10, 20, 50, 100], "precision": result[0],
                                  "recall": result[1], "ndcg": result[2], "mrr": result[3]}
            if result[1][0] > best:
                best, best_epoch = result[1][0], epoch + 1
                checkpoint(output / "best_val.pt", {"model": net.state_dict(), "epoch": best_epoch,
                           "validation": row["validation"], "plan": plan})
            count, stopped = author_stop(epoch, result[1][0], best, count)
        row.update(seconds=time.perf_counter() - begin, best_epoch=best_epoch, best_recall10=best, patience_count=count)
        checkpoint(output / "last.pt", {"plan": plan, "completed_epochs": epoch + 1,
            "model": net.state_dict(), "optimizer": net.opt.state_dict(), "rng": capture_rng(),
            "best_recall10": best, "best_epoch": best_epoch, "count": count, "stopped": stopped,
            "metrics_row": row})
        with (output / "metrics.jsonl").open("a") as f:
            f.write(__import__('json').dumps(row) + "\n")
        save(output / "status.json", {"status": "running", "completed_epochs": epoch + 1, "updated_utc": utc()})
        print("PUBLIC_EPOCH", __import__('json').dumps(row), flush=True)
        start = epoch + 1
    if not stopped and start < 600:
        save(output / "status.json", {"status": "staged", "completed_epochs": start, "updated_utc": utc(), "test_evaluated": False})
        return
    chosen = torch.load(output / "best_val.pt", map_location="cpu", weights_only=True)
    best_sha = digest(output / "best_val.pt")
    if not (output / "test_evaluation.json").exists():
        net.load_state_dict(chosen["model"])
        result = evaluate_public_test(ds, net, predictor, world, Procedure)
        save(output / "test_evaluation.json", {"split": "author_public_test", "topks": [10, 20, 50, 100],
            **result,
            "selected_epoch": chosen["epoch"], "checkpoint_sha256": best_sha, "evaluated_utc": utc(),
            "current_protocol_test_evaluated": False})
    result = read(output / "test_evaluation.json")
    if result["checkpoint_sha256"] != best_sha:
        raise ValueError("Test result identifies a different checkpoint")
    save(output / "completion.json", {"completed_epochs": start, "selected_epoch": chosen["epoch"],
        "stop_reason": "author_patience" if stopped else "600_epoch_limit", "test": result,
        "last_sha256": digest(output / "last.pt"), "completed_utc": utc(), "current_protocol_test_evaluated": False})
    save(output / "status.json", {"status": "completed", "completed_epochs": start, "updated_utc": utc()})


if __name__ == "__main__":
    main()
