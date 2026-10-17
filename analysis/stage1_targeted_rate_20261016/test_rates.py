"""Protect the single-factor comparison and separate overall/subgroup objectives."""

import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).with_name("run_rates.py")
    spec = importlib.util.spec_from_file_location("rate_study", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_job_grid_only_changes_rate_and_alternates_order(tmp_path):
    tool = module()
    config = json.loads((tool.ROOT / "configs/stage1_targeted_rate_20261016.json").read_text())
    jobs = list(tool.jobs(config, tmp_path))
    assert [(j["seed"], j["rate"]) for j in jobs] == [
        (2020, 0.2), (2020, 0.4), (2021, 0.4), (2021, 0.2), (2022, 0.2), (2022, 0.4)]
    assert len({j["id"] for j in jobs}) == 6
    for job in jobs:
        new = tool.job_config(config, job["seed"], job["rate"])
        assert new["training"] == {**config["training"], "targeted_rate": job["rate"]}
        assert new["variants"] == [{"id": "hard", "sampler": "reliable", "reliability_beta": 0.0}]


def test_pareto_keeps_sparse_user_tradeoff():
    tool = module()
    def point(ndcg, sparse):
        return {"ndcg20": {"mean": ndcg}, "sparse_ndcg20": {"mean": sparse}}
    points = {"uniform": point(1, 2), "rate02": point(2, 1.8),
              "rate04": point(2, 1.7), "rate08": point(3, 1)}
    assert set(tool.pareto_variants(points)) == {"uniform", "rate02", "rate08"}


def test_summary_pairs_each_seed_with_its_own_references(tmp_path, monkeypatch):
    tool = module()
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    config = {"seeds": [2020, 2021, 2022], "rates": [0.2, 0.4],
              "training": {"epochs": 30}, "reference_runs": {}, "interpretation": "test"}
    def write_history(run, ndcg, sparse):
        run.mkdir(parents=True)
        rows = [{"epoch": epoch, "samples": 100, "unique_user_negative_pairs": 90,
                 "validation": {"split": "val", "groups": {
                     "overall": {"ndcg": [ndcg], "recall": [ndcg]},
                     "user_sparse": {"ndcg": [sparse]},
                     "item_tail": {"ndcg": [0], "precision": [0], "users": 1}}}}
                for epoch in range(1, 31)]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for index, seed in enumerate(config["seeds"], 1):
        config["reference_runs"][str(seed)] = {"uniform": f"ref{seed}/uniform", "rate08": f"ref{seed}/hard"}
        write_history(tmp_path / f"ref{seed}/uniform", index, 2 * index)
        write_history(tmp_path / f"ref{seed}/hard", 2 * index, index)
        write_history(output / "jobs" / f"rate02-seed{seed}" / f"hard-seed{seed}", 1.5 * index, 1.9 * index)
    report = tool.summarize(output, config)
    assert report["completed_new_runs"] == 3
    assert report["aggregates"]["rate02"]["ndcg20"]["mean"] == 3
    assert "rate04" not in report["aggregates"]
    differences = [r for r in report["paired_differences"] if r["reference"] == "uniform"]
    assert len(differences) == 3
    assert all(abs(r["ndcg20_relative_percent"] - 50) < 1e-10 for r in differences)
    assert all(abs(r["sparse_ndcg20_relative_percent"] + 5) < 1e-10 for r in differences)
