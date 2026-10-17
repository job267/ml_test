"""Serial, resumable GPU-0 queue for local baselines and public calibration."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
from shared import ROOT, HERE, digest, gpu_guard, history, read, save, utc
from experiment import code_hashes, load_checkpoint, TRAINING_KEYS
from world import parse_topks

OUTPUT = ROOT / "runs/baseline_calibration_20261016"
CONFIG = ROOT / "configs/baseline_calibration_20261016.json"
spec = importlib.util.spec_from_file_location("frozen_baseline_summary", ROOT / "analysis/stage1_beta_replication_20261016/replicate.py")
replication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replication)


def expected(config, seed, layer):
    t = {**config["training"], "seed": seed, "layer": layer}
    t["topks"] = parse_topks(t["topks"])
    return {**{k: t[k] for k in TRAINING_KEYS},
            "dataset_sha256": digest(ROOT / "data/kuairand_stage1/manifest.json")}


def locations(config, output):
    for seed in config["seeds"]:
        yield "mf_bpr_matched_init", seed, output / "stage1" / f"mf-seed{seed}"
        yield "lightgcn_l3", seed, (ROOT / config["references"][str(seed)] if seed == 2020 else
                                    output / "stage1" / f"lgn-seed{seed}")


def summarize(config, output):
    rows = []
    for name, seed, run in locations(config, output):
        if not run.exists() and name == "lightgcn_l3":
            run = ROOT / config["references"][str(seed)]
        for horizon in config["horizons"]:
            result = replication.window_result(history(run / "metrics.jsonl"), horizon)
            if result:
                rows.append({"model": name, "seed": seed, "horizon": horizon,
                             "run": str(run.relative_to(ROOT)), **result})
    aggregates = []
    for horizon in config["horizons"]:
        for name in ("mf_bpr_matched_init", "lightgcn_l3"):
            subset = [r for r in rows if r["model"] == name and r["horizon"] == horizon]
            if {r["seed"] for r in subset} == set(config["seeds"]):
                aggregates.append({"model": name, "horizon": horizon, "n_seeds": len(subset),
                   **{f: {"mean": statistics.mean(r[f] for r in subset),
                          "sample_std": statistics.stdev(r[f] for r in subset)}
                      for f in ("ndcg20", "recall20", "sparse_ndcg20", "tail_ndcg20")}})
    cf_path = output / "stage1/itemcf/completion.json"
    cf = read(cf_path) if cf_path.exists() else None
    public_rows = []
    for seed in config["public"]["seeds"]:
        completion = output / "public" / f"seed{seed}" / "completion.json"
        if completion.exists():
            result = read(completion)
            public_rows.append({"seed": seed, "completed_epochs": result["completed_epochs"],
                                "selected_epoch": result["selected_epoch"],
                                "ndcg20": result["test"]["author_original_mask"]["ndcg"][1],
                                "recall20": result["test"]["author_original_mask"]["recall"][1],
                                "corrected_mask_ndcg20": result["test"]["corrected_user_aligned_mask"]["ndcg"][1],
                                "corrected_mask_recall20": result["test"]["corrected_user_aligned_mask"]["recall"][1]})
    public_aggregate = None
    if len(public_rows) == len(config["public"]["seeds"]):
        public_aggregate = {f: {"mean": statistics.mean(r[f] for r in public_rows),
                               "sample_std": statistics.stdev(r[f] for r in public_rows)} for f in
                                  ("ndcg20", "recall20", "corrected_mask_ndcg20", "corrected_mask_recall20")}
    report = {"updated_utc": utc(), "local_split": "val", "current_protocol_test_evaluated": False,
              "local_rows": rows, "local_aggregates": aggregates, "itemcf": cf,
              "public_rows": public_rows, "public_aggregate": public_aggregate,
              "interpretation": "Local results use the same 30/100 epoch windows and validation selection; ItemCF has one deterministic 8-configuration validation search, not three seeds. Public scores belong to a separate author split and cannot be compared numerically with local scores."}
    save(output / "comparison.json", report)
    lines = ["# 基线与公开协议校准", "", "当前协议只评估验证集；公开协议单独使用作者划分。", "",
             "| 模型 | 共同轮数 | 种子数 | NDCG@20 均值 ± 标准差 | Recall@20 | 稀疏用户 NDCG@20 |",
             "|---|---:|---:|---:|---:|---:|"]
    for a in aggregates:
        lines.append(f"| {a['model']} | {a['horizon']} | {a['n_seeds']} | {a['ndcg20']['mean']:.8f} ± {a['ndcg20']['sample_std']:.8f} | {a['recall20']['mean']:.8f} | {a['sparse_ndcg20']['mean']:.8f} |")
    if cf:
        b = cf["selected"]
        g = b["validation"]["groups"]
        lines += ["", f"ItemCF：验证集选择 {b['variant']}（8 组配置），NDCG@20={g['overall']['ndcg'][0]:.8f}，Recall@20={g['overall']['recall'][0]:.8f}，稀疏用户 NDCG@20={g['user_sparse']['ndcg'][0]:.8f}。"]
    lines += ["", f"公开协议完成 {len(public_rows)}/5 个种子；按作者默认的验证集 Recall@10 选择检查点，每个种子结束后一次评分同时报告原样和修正屏蔽的测试指标。",
              "", "MF/BPR 采用原 LightGCN 的 L=0，初始化与 L=3 一致；各分组指标均来自总体 NDCG 最优的同一轮。详细记录见 comparison.json。"]
    if public_aggregate:
        lines += ["", f"公开测试（作者原样屏蔽）：NDCG@20={public_aggregate['ndcg20']['mean']:.6f}，Recall@20={public_aggregate['recall20']['mean']:.6f}。",
                  f"修正用户行对齐屏蔽：NDCG@20={public_aggregate['corrected_mask_ndcg20']['mean']:.6f}，Recall@20={public_aggregate['corrected_mask_recall20']['mean']:.6f}。",
                  "作者论文参考值：Vanilla LightGCN NDCG@20=0.0672、Recall@20=0.1281；环境版本差异详见计划，不能据此比较当前协议成绩。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def stage_command(config, output, seed, layer, horizon):
    run = output / "stage1" / f"{'mf' if layer == 0 else 'lgn'}-seed{seed}"
    command = [sys.executable, "-u", str(ROOT / "code/experiment.py"), "--dataset-dir",
               str(ROOT / "data/kuairand_stage1"), "--device", "cuda", "--epochs", str(horizon)]
    for k, v in {**config["training"], "seed": seed, "layer": layer}.items():
        command += ["--" + k.replace("_", "-"), str(v)]
    if layer == 3 and not run.exists():
        source = ROOT / config["references"][str(seed)]
        # A full, independent fork preserves source weights, optimizer, sampler and RNG.
        shutil.copytree(source, run)
        save(run / "fork_provenance.json", {"source": str(source), "source_last_sha256": digest(source / "last.pt"),
             "source_completed_epochs": load_checkpoint(source / "last.pt")["completed_epochs"],
             "target_epochs": horizon, "created_utc": utc(), "source_unchanged": True})
    if (run / "last.pt").exists():
        state = load_checkpoint(run / "last.pt")
        if state["training_config"] != expected(config, seed, layer) or state["code_sha256"] != code_hashes():
            raise ValueError("Local checkpoint config/source mismatch: " + str(run))
        if state["completed_epochs"] >= horizon:
            return None
        command += ["--resume", str(run / "last.pt")]
    elif run.exists():
        # A crash before its first checkpoint is intentionally explicit, never overwrite.
        raise RuntimeError("Uncheckpointed local run requires inspection: " + str(run))
    else:
        command += ["--output-root", str(run.parent), "--run-id", run.name]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    gpu = gpu_guard()
    config = read(CONFIG)
    output = OUTPUT
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / ".study.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume:
        from public_protocol import audit
        audit(output / "public")
    source_paths = [*HERE.glob("*.py"), ROOT / "analysis/stage1_beta_replication_20261016/replicate.py"]
    def fingerprints():
        return {str(path.relative_to(ROOT)): digest(path) for path in sorted(source_paths)}
    references = {}
    for seed, location in config["references"].items():
        base = ROOT / location
        meta = read(base / "metadata.json")
        if meta["training_config"] != expected(config, int(seed), 3) or meta["environment"]["code_sha256"] != code_hashes():
            raise ValueError("Frozen LightGCN reference does not match planned control")
        for filename in ("metadata.json", "metrics.jsonl", "completion.json", "last.pt", "best_val.pt"):
            references[str((base / filename).relative_to(ROOT))] = digest(base / filename)
    plan = {"config": config, "config_sha256": digest(CONFIG), "environment": gpu,
            "code_sha256": code_hashes(), "driver_sha256": fingerprints(), "reference_sha256": references,
            "dataset_manifest_sha256": digest(ROOT / "data/kuairand_stage1/manifest.json"),
            "author_provenance_sha256": digest(ROOT / "external/tpab_7753efd/provenance.json"),
            "public_protocol_audit_sha256": digest(output / "public/protocol_audit.json"),
            "current_protocol_test_evaluated": False}
    if args.resume:
        if read(output / "plan.json") != plan:
            raise ValueError("Study plan/source/input changed")
    else:
        save(output / "plan.json", plan)
        (output / "logs").mkdir()
        (output / "stage1").mkdir()
    def verify():
        if fingerprints() != plan["driver_sha256"] or code_hashes() != plan["code_sha256"] or digest(CONFIG) != plan["config_sha256"]:
            raise ValueError("Frozen source/config changed")
        if digest(ROOT / "data/kuairand_stage1/manifest.json") != plan["dataset_manifest_sha256"]:
            raise ValueError("Local dataset changed")
        if digest(output / "public/protocol_audit.json") != plan["public_protocol_audit_sha256"] or digest(ROOT / "external/tpab_7753efd/provenance.json") != plan["author_provenance_sha256"]:
            raise ValueError("Public protocol/provenance changed")
        for name, sha in references.items():
            if digest(ROOT / name) != sha:
                raise ValueError("Original reference changed: " + name)
    # Early public stage verifies the author code in this environment before long jobs.
    queue = [("itemcf", None, None, None), ("public", 2018, None, 5)]
    queue += [("local", seed, 0, 30) for seed in config["seeds"]]
    queue += [("local", 2020, 0, 100), ("local", 2021, 3, 100), ("local", 2021, 0, 100),
              ("local", 2022, 3, 100), ("local", 2022, 0, 100)]
    queue += [("public", seed, None, 600) for seed in config["public"]["seeds"]]
    state = {"status": "running", "queue_pid": os.getpid(), "started_utc": utc(), "finished_stages": [],
             "expected_stages": len(queue), "current_protocol_test_evaluated": False}
    child = None
    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, (kind, seed, layer, horizon) in enumerate(queue):
            verify()
            job_id = f"{index:02d}-{kind}-s{seed}-l{layer}-e{horizon}"
            command = None
            if kind == "itemcf":
                target = output / "stage1/itemcf"
                if not (target / "completion.json").exists():
                    command = [sys.executable, "-u", str(HERE / "itemcf.py"), "--output", str(target)]
            elif kind == "local":
                command = stage_command(config, output, seed, layer, horizon)
            else:
                target = output / "public" / f"seed{seed}"
                already = (target / "last.pt").exists() and torch_completed(target) >= horizon
                if not (target / "completion.json").exists() and not already:
                    command = [sys.executable, "-u", str(HERE / "public_run.py"), "--seed", str(seed),
                               "--output", str(target), "--stop-after", str(horizon)]
                    if (target / "last.pt").exists():
                        command.append("--resume")
                    elif target.exists():
                        raise RuntimeError("Uncheckpointed public run requires inspection")
            state.update(active_stage=job_id, updated_utc=utc())
            save(output / "status.json", state)
            summarize(config, output)
            if command:
                with (output / "logs" / f"{job_id}.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(child_pid=child.pid, command=command)
                    save(output / "status.json", state)
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            summarize(config, output)
                            state["updated_utc"] = utc()
                            save(output / "status.json", state)
                    child = None
                    if code:
                        raise RuntimeError(f"Stage {job_id} failed with exit code {code}; inspect its log")
            verify()
            state["finished_stages"].append(job_id)
            print("FINISHED_STAGE", job_id, flush=True)
            summarize(config, output)
        state.update(status="completed", completed_utc=utc(), active_stage=None, child_pid=None)
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(status="interrupted" if isinstance(e, (KeyboardInterrupt, InterruptedError)) else "failed", error=str(e), updated_utc=utc())
        save(output / "status.json", state)
        summarize(config, output)
        raise
    save(output / "status.json", state)
    save(output / "completion.json", {"completed_utc": utc(), "plan_sha256": digest(output / "plan.json"),
         "comparison_sha256": digest(output / "comparison.json"), "current_protocol_test_evaluated": False})


def torch_completed(target):
    import torch
    return torch.load(target / "last.pt", map_location="cpu", weights_only=True)["completed_epochs"]


if __name__ == "__main__":
    main()
