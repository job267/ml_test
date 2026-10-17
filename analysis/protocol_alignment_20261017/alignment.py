"""Compare pinned author and project mathematics on identical inputs, validation only."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
from common import CONFIG, ROOT, VENDOR, OUTPUT, author_definitions, checkpoint, gpu_guard, read, save, sha, utc
from dataset import author_dataset
from model import LightGCN
from research_evaluation import evaluate
from utils import BPRLoss
from experiment import code_hashes
from evaluation import rank


def baseline_config(c):
    return {"latent_dim_rec": c["recdim"], "lightGCN_n_layers": c["layer"],
            "dropout": 0, "keep_prob": 1., "A_split": False, "decay": c["decay"],
            "lr": c["lr"], "period": 8, "algo": "vanilla", "test_u_batch_size": c["eval_batch"]}


def difference(a, b):
    delta = (a.detach() - b.detach()).abs()
    return float(delta.max()) if delta.numel() else 0.


def require_close(name, a, b, atol):
    maximum = difference(a, b)
    if not np.isfinite(maximum) or maximum > atol:
        raise AssertionError(f"{name}: maximum absolute difference {maximum} exceeds {atol}")
    return maximum


def author_loss(net, batch):
    u, p, n = batch.unbind(1)
    loss, _ = net.bpr_loss(u, p, n, None, None, None, None, None)
    return loss.mean()


def project_loss(net, batch, decay):
    loss, reg = net.bpr_loss(*batch.unbind(1))
    return loss + decay * reg


def model_pair(dataset, c, definitions):
    torch.manual_seed(c["seed"])
    author = definitions.model(baseline_config(c), dataset).to(dataset.device)
    torch.manual_seed(c["seed"])
    project = LightGCN(baseline_config(c), dataset).to(dataset.device)
    initialization_equal = all(torch.equal(x, y) for x, y in zip(author.parameters(), project.parameters()))
    project.load_state_dict(author.state_dict())
    return author, project, initialization_equal


def metrics_on_same_factors(ds, model, definitions, c):
    """Execute both actual evaluators with the same frozen factors and user batches."""
    with torch.no_grad():
        factors = tuple(x.detach() for x in model.computer())
    class Factors:
        def eval(self):
            return self
        def computer(self):
            return factors
        def getUsersRating(self, users, *unused):
            return factors[0][users] @ factors[1].T
    shared = Factors()
    definitions.world.topks = [10, 20, 50, 100]
    # Preserve the function body, avoid the author's total_batch assertion bug for exact divisibility.
    batch = c["eval_batch"]
    if len(ds.valDict) % batch == 0:
        batch += 1
    definitions.world.config["test_u_batch_size"] = batch
    author = definitions.evaluate(ds, shared, None, 0, None, 0, 0)
    project = evaluate(ds, shared, "val", [10, 20, 50, 100], batch)
    group = project["groups"]["overall"]
    for index, field in enumerate(("precision", "recall", "ndcg")):
        if [round(float(x), 4) for x in group[field]] != author[index]:
            raise AssertionError("Actual validation evaluators disagree after author rounding: " + field)
    return {"users": len(ds.valDict), "batch_size": batch,
            "author_rounded": {name: author[j] for j, name in enumerate(("precision", "recall", "ndcg", "mrr"))},
            "project_unrounded": group, "matches_after_author_rounding": True,
            "test_evaluated": False}


def run_alignment(output, device="cpu", full=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    c = read(CONFIG)["alignment"]
    tolerance = c["tolerances"]
    environment = gpu_guard() if device == "cuda" else {"python": __import__('sys').executable, "torch": str(torch.__version__), "device": "cpu"}
    ds = author_dataset(device, 0 if full else c["cpu_validation_users"], c["seed"])
    definitions = author_definitions(baseline_config(c), device)
    graph = ds.getSparseGraph().coalesce()
    cached = sp.load_npz(VENDOR / "kuairand/s_pre_adj_mat.npz").tocoo()
    cached_tensor = torch.sparse_coo_tensor(np.stack((cached.row, cached.col)), cached.data, cached.shape,
        dtype=torch.float32, device=device).coalesce()
    if not torch.equal(graph.indices(), cached_tensor.indices()):
        raise AssertionError("Author cache and project train-only graph have different edges")
    graph_diff = require_close("normalized graph", graph.values(), cached_tensor.values(), tolerance["graph_atol"])
    # Use the SAME graph for isolated model/optimizer comparisons; separately record normalization differences.
    ds._graph = cached_tensor
    author, project, init_equal = model_pair(ds, c, definitions)
    if not init_equal:
        raise AssertionError("Identical seed/config did not create identical initial weights")
    np.random.seed(c["seed"])
    definitions.sample(ds)
    triplets = np.stack((ds.user, ds.posItem, ds.negItem), axis=1).astype(np.int64)
    if ds.is_positive(triplets[:, 0], triplets[:, 2]).any():
        raise AssertionError("Pinned sampler produced a known positive negative")
    np.save(output / "probe_triplets.npy", triplets)
    batch = torch.as_tensor(triplets[:c["batch_size"]], dtype=torch.long, device=device)
    ua, ia = author.computer(); up, ip = project.computer()
    checks = {"graph_cache_max_abs_difference": graph_diff, "initial_weights_equal": init_equal,
              "user_embedding_max_abs_difference": require_close("users", ua, up, tolerance["embedding_atol"]),
              "item_embedding_max_abs_difference": require_close("items", ia, ip, tolerance["embedding_atol"])}
    ids = batch[:, 0]
    raw_a, raw_p = ua[ids[:32]] @ ia.T, up[ids[:32]] @ ip.T
    checks["raw_score_max_abs_difference"] = require_close("raw scores", raw_a, raw_p, tolerance["score_atol"])
    la, lp = author_loss(author, batch), project_loss(project, batch, c["decay"])
    checks["objective_max_abs_difference"] = require_close("objective", la, lp, tolerance["objective_atol"])
    la.backward(); lp.backward()
    checks["gradient_max_abs_difference"] = max(require_close("gradient", x.grad, y.grad, tolerance["gradient_atol"])
        for x, y in zip(author.parameters(), project.parameters()))
    with torch.no_grad():
        u, pos, neg = batch.unbind(1)
        delta = (up[u] * (ip[pos] - ip[neg])).sum(1)
        checks["observed_initial_score_margin_range"] = [float(delta.min()), float(delta.max())]
    del ua, ia, up, ip, raw_a, raw_p, la, lp
    # Restart so the probe doesn't include a duplicated update from its gradient check.
    author, project, _ = model_pair(ds, c, definitions)
    project_optimizer = BPRLoss(project, baseline_config(c))
    steps = c["gpu_probe_steps"] if full else c["cpu_probe_steps"]
    losses = []
    for step in range(steps):
        start = (step * c["batch_size"]) % len(triplets)
        batch = torch.as_tensor(triplets[start:start + c["batch_size"]], device=device, dtype=torch.long)
        author.opt.zero_grad()
        native = author_loss(author, batch); native.backward(); author.opt.step(); author.opt.zero_grad()
        value = project_optimizer.stageOne(*batch.unbind(1))
        losses.append({"step": step + 1, "author": float(native.detach()), "project": value})
    checks["probe_steps"] = steps
    checks["probe_parameter_max_abs_difference"] = max(require_close("Adam replay parameters", x, y,
        tolerance["probe_parameter_atol"]) for x, y in zip(author.parameters(), project.parameters()))
    # Evaluate actual author and project ranking implementations on exactly the same factors.
    checks["evaluation"] = metrics_on_same_factors(ds, project, definitions, c)
    notes = ["Author uses -log(sigmoid(margin)+1e-10); project uses stable softplus(-margin).",
             "Author model rating interface is raw dot product; project legacy getUsersRating is sigmoid, but its research evaluator explicitly uses raw computer() factors.",
             "Author metrics round to four decimals; comparison retains project precision and checks the rounded values.",
             "Same seed does not make the two samplers produce the same triplets; both implementations replay the pinned author sampler's exact triplets here."]
    x = torch.tensor([-30.], dtype=torch.float64, requires_grad=True)
    a = -torch.log(torch.sigmoid(x) + 1e-10)
    ga = torch.autograd.grad(a.sum(), x)[0]
    b = torch.nn.functional.softplus(-x)
    gb = torch.autograd.grad(b.sum(), x)[0]
    saturation = {"margin": -30., "author_loss": float(a.detach()), "project_loss": float(b.detach()),
                  "author_gradient": float(ga), "project_gradient": float(gb),
                  "interpretation": "Known objective difference at extreme negative margins; not claimed to be bitwise mathematically identical for all inputs."}
    report = {"status": "probe_passed", "created_utc": utc(), "device": device, "environment": environment,
        "config_sha256": sha(CONFIG), "core_sha256": code_hashes(),
        "driver_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), Path(__file__).parent / "common.py",
            Path(__file__).parent / "dataset.py", Path(__file__).parent / "evaluation.py"]},
        "author_sha256": definitions.verified, "dataset_sha256": ds.fingerprint, "checks": checks,
        "probe_losses": losses, "known_differences": notes, "saturation_diagnostic": saturation,
        "test_evaluated": False}
    save(output / "probe_result.json", report)
    print("ALIGNMENT_PROBE", json.dumps(report), flush=True)
    if full:
        report["replay"] = replay(ds, c, definitions, output)
        report["status"] = "passed"
    else:
        report["status"] = "cpu_preflight_passed"
    save(output / "completion.json", report)
    return report


def replay(ds, c, definitions, output):
    author, project, _ = model_pair(ds, c, definitions)
    optimizer = BPRLoss(project, baseline_config(c))
    np.random.seed(c["seed"])
    shuffle = np.random.default_rng(c["seed"] + 1000)
    records = []
    for epoch in range(1, c["replay_epochs"] + 1):
        definitions.sample(ds)
        triples = np.stack((ds.user, ds.posItem, ds.negItem), axis=1).astype(np.int64)
        triples = triples[shuffle.permutation(len(triples))]
        np.save(output / f"replay_epoch{epoch}_triplets.npy", triples)
        totals = [0., 0.]
        author.train(); project.train()
        for start in range(0, len(triples), c["batch_size"]):
            batch = torch.as_tensor(triples[start:start + c["batch_size"]], dtype=torch.long, device=ds.device)
            author.opt.zero_grad()
            loss = author_loss(author, batch); loss.backward(); author.opt.step(); author.opt.zero_grad()
            value = optimizer.stageOne(*batch.unbind(1))
            totals[0] += float(loss.detach()) * len(batch); totals[1] += value * len(batch)
        results = {}
        for name, net in (("author", author), ("project", project)):
            results[name] = evaluate(ds, net, "val", [20], c["eval_batch"])["groups"]["overall"]
        deltas = {f: abs(results["author"][f][0] - results["project"][f][0]) for f in ("ndcg", "recall")}
        row = {"epoch": epoch, "mean_native_losses": [v / len(triples) for v in totals],
               "validation": results, "absolute_metric_differences": deltas,
               "parameter_max_abs_difference": max(difference(x, y) for x, y in zip(author.parameters(), project.parameters()))}
        records.append(row)
        save(output / "replay_history.json", records)
        print("ALIGNMENT_REPLAY", json.dumps(row), flush=True)
        if deltas["ndcg"] > c["tolerances"]["replay_ndcg20_atol"] or deltas["recall"] > c["tolerances"]["replay_recall20_atol"]:
            raise AssertionError("Native trainer replay diverged beyond predeclared metric tolerances")
    checkpoint(output / "replay_final.pt", {"author": author.state_dict(), "project": project.state_dict(),
        "author_optimizer": author.opt.state_dict(), "project_optimizer": optimizer.opt.state_dict(), "config": c})
    return {"epochs": len(records), "records": records, "native_loss_formulas_preserved": True,
            "scope": "Short controlled implementation replay; not an independent paper-score reproduction"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    run_alignment(args.output, args.device, args.full)
