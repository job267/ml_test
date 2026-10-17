"""After the existing GPU queue finishes, run alignment, then label/source diagnosis."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from common import CONFIG, HERE, OUTPUT, ROOT, history, read, save, sha, utc, verify_author
from experiment import code_hashes
from factorial import driver_hashes


def prerequisite_state(folder):
    """Never use GPU or infer success from a dead process or stale status."""
    folder = Path(folder)
    state = read(folder / "status.json")
    if state["status"] in ("failed", "interrupted"):
        raise RuntimeError("Prerequisite queue did not complete: " + str(state.get("error")))
    if state["status"] == "completed" and (folder / "completion.json").exists():
        return True, state
    return False, state


def audit_prerequisite(folder):
    folder = Path(folder)
    completion = read(folder / "completion.json")
    if sha(folder / "plan.json") != completion["plan_sha256"] or sha(folder / "comparison.json") != completion["comparison_sha256"]:
        raise ValueError("Prerequisite completion fingerprint mismatch")
    report = read(folder / "comparison.json")
    if len(report["public_rows"]) != 5 or report["public_aggregate"] is None:
        raise ValueError("Five public seeds are not fully completed")
    records = []
    for seed in (2018, 2019, 2020, 2021, 2022):
        run = folder / "public" / f"seed{seed}"
        result = read(run / "completion.json")
        test = read(run / "test_evaluation.json")
        if result["test"] != test or sha(run / "best_val.pt") != test["checkpoint_sha256"] or sha(run / "last.pt") != result["last_sha256"]:
            raise ValueError("Public selected checkpoint/result mismatch")
        rows = history(run / "metrics.jsonl")
        if [r["epoch"] for r in rows] != list(range(1, result["completed_epochs"] + 1)):
            raise ValueError("Public epoch history is incomplete")
        selected = max((r for r in rows if "validation" in r), key=lambda r: r["validation"]["recall"][0])
        if selected["epoch"] != test["selected_epoch"]:
            raise ValueError("Public result does not use author validation selection")
        records.append({"seed": seed, "completed_epochs": result["completed_epochs"],
                        "selected_epoch": test["selected_epoch"], "test_sha256": sha(run / "test_evaluation.json")})
    return {"audited_utc": utc(), "completion_sha256": sha(folder / "completion.json"),
            "five_seeds": records, "public_test_results_not_used_to_change_plan": True}


def extension_needed(histories, initial=30):
    if set(histories) != {"engagement", "click"}:
        raise ValueError("Both label models are required for an extension decision")
    reasons = []
    for label, rows in histories.items():
        window = [r for r in rows if r["epoch"] <= initial]
        if [r["epoch"] for r in window] != list(range(1, initial + 1)):
            raise ValueError("Extension decision requires both complete initial windows")
        for source in ("random", "all"):
            best = max(window, key=lambda r: r["validation_views"][source]["common_users"]["groups"]["overall"]["ndcg"][0])
            if best["epoch"] > initial - 5:
                reasons.append({"label": label, "view": source, "best_epoch": best["epoch"]})
    return bool(reasons), reasons


def summarize(output=OUTPUT):
    c = read(CONFIG)
    records, curves = [], {}
    for label in c["factorial"]["positive_rules"]:
        rows = history(output / "factorial" / label / "metrics.jsonl")
        curves[label] = rows
        if not rows:
            continue
        # Fixed-epoch primary results, without independently maximizing each view.
        for epoch in (1, 5, 10, 20, 30, 60):
            match = next((r for r in rows if r["epoch"] == epoch), None)
            if match:
                records.append({"label": label, "epoch": epoch, "role": "same_epoch_primary",
                                "views": match["validation_views"]})
        chosen = max(rows, key=lambda r: r["validation_views"]["random"]["common_users"]["groups"]["overall"]["ndcg"][0])
        records.append({"label": label, "epoch": chosen["epoch"], "role": "random_validation_selected_secondary",
                        "views": chosen["validation_views"]})
    comparison = {"updated_utc": utc(), "split": "validation", "test_evaluated": False,
                  "records": records, "completed_epochs": {k: len(v) for k, v in curves.items()},
                  "interpretation": "Single-seed protocol diagnosis on training-only common users/vocabulary and equal optimizer-update budgets; cross-label scores are task differences, not algorithm gains. Original train-degree subgroup definitions are label-specific; only the overall common-user cohort is fixed across labels."}
    save(output / "comparison.json", comparison)
    status = read(output / "status.json") if (output / "status.json").exists() else {}
    cpu_path = output / "cpu_preflight_final/completion.json"
    cpu = read(cpu_path) if cpu_path.exists() else None
    lines = ["# 实现对齐与标签/曝光诊断", "", f"阶段：{status.get('stage', 'preparing')}；状态：{status.get('status', 'preparing')}。", "",
             "GPU 0 作业等待原公开五种子队列全部完成并通过审计后，才会串行启动。", ""]
    if cpu:
        checks = cpu["checks"]
        lines += [f"CPU 数值预检通过：相同初始权重、相同图和样本下，原始评分最大差 {checks['raw_score_max_abs_difference']:.3g}，梯度最大差 {checks['gradient_max_abs_difference']:.3g}；{checks['evaluation']['users']} 个用户的两套验证评价器在作者四位小数口径下一致。", ""]
    alignment = output / "gpu_alignment/completion.json"
    if alignment.exists():
        a = read(alignment)
        lines += [f"GPU 对齐：{a['status']}；完成 {a.get('replay', {}).get('epochs', 0)} 轮相同样本回放。", ""]
    lines += ["两种标签只训练两个模型，每个模型的同一份排名同时评价 random/all 与固定/完整可评用户。当前研究只使用训练与验证日期，没有读取未来测试反馈。", "",
              "| 标签 | 训练轮数 | 已有共同用户 random NDCG@20 | 已有共同用户 all NDCG@20 |", "|---|---:|---:|---:|"]
    for label, rows in curves.items():
        if rows:
            r = rows[-1]
            lines.append(f"| {label} | {r['epoch']} | {r['validation_views']['random']['common_users']['groups']['overall']['ndcg'][0]:.8f} | {r['validation_views']['all']['common_users']['groups']['overall']['ndcg'][0]:.8f} |")
    lines += ["", "表格为最近已完成轮次；严格相同轮次的主诊断与同一随机验证选中检查点的辅助结果见 comparison.json。分组阈值来自各自标签训练图，跨标签分组不能视为完全相同人群。没有把这里的单种子诊断称为独立确认。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return comparison


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if Path(sys.prefix).name != "gcn" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Use gcn and restrict all upcoming GPU jobs to physical GPU 0")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / ".sequence.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(CONFIG)
    previous = ROOT / config["prerequisite"]
    data_manifest = OUTPUT / "data/manifest.json"
    prepared = read(data_manifest)
    for name, digest in prepared["files"].items():
        if sha(data_manifest.parent / name) != digest:
            raise ValueError("Prepared diagnosis data changed")
    cpu_path = OUTPUT / "cpu_preflight_final/completion.json"
    cpu_report = read(cpu_path)
    if cpu_report["status"] != "cpu_preflight_passed":
        raise ValueError("CPU preflight has not passed")
    if cpu_report["config_sha256"] != sha(CONFIG) or cpu_report["core_sha256"] != code_hashes():
        raise ValueError("CPU preflight source/config differs")
    for name, digest in cpu_report["driver_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("CPU preflight driver changed")
    author_hashes = verify_author(["model.py", "dataloader.py", "Procedure.py", "kuairand/training_dict.npy",
                                   "kuairand/validation_dict.npy", "kuairand/s_pre_adj_mat.npz"])
    plan = {"config": config, "config_sha256": sha(CONFIG), "driver_sha256": driver_hashes(),
            "core_sha256": code_hashes(), "author_sha256": author_hashes,
            "previous_plan_sha256": sha(previous / "plan.json"), "data_manifest_sha256": sha(data_manifest),
            "cpu_preflight_sha256": sha(cpu_path), "python": sys.executable,
            "current_protocol_test_evaluated": False, "gpu": "physical GPU 0 only; no CUDA work before prerequisite completion"}
    if args.resume:
        if read(OUTPUT / "sequence_plan.json") != plan:
            raise ValueError("Sequence plan/input changed")
    elif (OUTPUT / "sequence_plan.json").exists():
        raise FileExistsError("Existing sequence requires --resume")
    else:
        save(OUTPUT / "sequence_plan.json", plan)
        (OUTPUT / "logs").mkdir(exist_ok=False)
    state = {"status": "waiting", "stage": "wait_for_public_completion", "queue_pid": os.getpid(), "started_utc": utc(), "test_evaluated": False}
    child = None
    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    def verify():
        if driver_hashes() != plan["driver_sha256"] or code_hashes() != plan["core_sha256"] or sha(CONFIG) != plan["config_sha256"]:
            raise ValueError("Frozen source/config changed")
        if sha(data_manifest) != plan["data_manifest_sha256"] or sha(previous / "plan.json") != plan["previous_plan_sha256"]:
            raise ValueError("Data/prerequisite identity changed")
    def execute(stage, command):
        nonlocal child
        verify()
        state.update(status="running", stage=stage, command=command, updated_utc=utc())
        save(OUTPUT / "status.json", state); summarize()
        with (OUTPUT / "logs" / f"{stage}.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["child_pid"] = child.pid; save(OUTPUT / "status.json", state)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    verify(); state["updated_utc"] = utc(); save(OUTPUT / "status.json", state); summarize()
            child = None
            if code:
                raise RuntimeError(f"{stage} failed with code {code}; next stages were not started")
        verify(); summarize()
    try:
        while True:
            verify()
            ready, previous_state = prerequisite_state(previous)
            state.update(updated_utc=utc(), prerequisite_stage=previous_state.get("active_stage"),
                         prerequisite_finished_stages=len(previous_state.get("finished_stages", [])))
            save(OUTPUT / "status.json", state); summarize()
            if ready:
                # Completion JSON must be followed by release of the actual previous queue lock.
                old_lock = (previous / ".study.lock").open("a")
                try:
                    fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    old_lock.close(); time.sleep(30); continue
                save(OUTPUT / "prerequisite_audit.json", audit_prerequisite(previous))
                old_lock.close()
                break
            time.sleep(30)
        gpu_result = OUTPUT / "gpu_alignment/completion.json"
        if not gpu_result.exists():
            execute("gpu_alignment", [sys.executable, "-u", str(HERE / "alignment.py"), "--device", "cuda", "--full", "--output", str(gpu_result.parent)])
        if read(gpu_result)["status"] != "passed":
            raise ValueError("Full implementation alignment did not pass; factorial training is gated")
        for label in config["factorial"]["positive_rules"]:
            run = OUTPUT / "factorial" / label
            rows = history(run / "metrics.jsonl")
            if len(rows) < config["factorial"]["initial_epochs"]:
                command = [sys.executable, "-u", str(HERE / "factorial.py"), "--label", label,
                           "--stop-after", str(config["factorial"]["initial_epochs"])]
                if (run / "last.pt").exists():
                    command.append("--resume")
                execute(f"factorial_{label}_30", command)
        histories = {label: history(OUTPUT / "factorial" / label / "metrics.jsonl") for label in config["factorial"]["positive_rules"]}
        extend, reasons = extension_needed(histories, config["factorial"]["initial_epochs"])
        decision = {"extend_both": extend, "reasons": reasons,
                    "horizon": config["factorial"]["maximum_epochs"] if extend else config["factorial"]["initial_epochs"]}
        decision_path = OUTPUT / "extension_decision.json"
        if decision_path.exists() and read(decision_path) != decision:
            raise ValueError("Committed extension decision changed")
        save(decision_path, decision)
        if extend:
            for label in config["factorial"]["positive_rules"]:
                run = OUTPUT / "factorial" / label
                if len(history(run / "metrics.jsonl")) < decision["horizon"]:
                    execute(f"factorial_{label}_60", [sys.executable, "-u", str(HERE / "factorial.py"), "--label", label,
                                                     "--stop-after", str(decision["horizon"]), "--resume"])
        final = summarize()
        for label, epochs in final["completed_epochs"].items():
            if epochs != decision["horizon"]:
                raise ValueError("Final diagnosis horizons differ")
        state.update(status="completed", stage="completed", updated_utc=utc(), child_pid=None)
        save(OUTPUT / "status.json", state); summarize()
        save(OUTPUT / "completion.json", {"completed_utc": utc(), "horizon": decision["horizon"],
             "plan_sha256": sha(OUTPUT / "sequence_plan.json"), "comparison_sha256": sha(OUTPUT / "comparison.json"), "test_evaluated": False})
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        state.update(status="interrupted" if isinstance(e, (InterruptedError, KeyboardInterrupt)) else "failed",
                     error=str(e), updated_utc=utc())
        save(OUTPUT / "status.json", state); summarize()
        raise


if __name__ == "__main__":
    main()
