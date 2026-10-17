"""Check isolated candidate proposals and source-specific paired reporting."""

import importlib.util
import json
from pathlib import Path
import numpy as np


def module():
    path = Path(__file__).with_name("run_sources.py")
    spec = importlib.util.spec_from_file_location("source_study", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_configuration_changes_only_candidate_evidence_fraction():
    tool = module()
    config = tool.read_json(tool.ROOT / "configs/stage1_candidate_source_20261016.json")
    reference = tool.read_json(tool.ROOT / config["reference_config"])
    assert config["training"] == {**reference["training"], "evidence_fraction": 0.0}
    for seed in config["seeds"]:
        pilot = tool.pilot_config(config, seed)
        assert pilot["training"]["targeted_rate"] == 0.8
        assert pilot["training"]["epochs"] == 30
        assert pilot["variants"] == [{"id": "hard", "sampler": "reliable", "reliability_beta": 0.0}]


def test_zero_evidence_candidate_source_never_reads_evidence_pool():
    module()  # Load the original code directory through the shared analysis helper.
    from research_sampling import FeedbackSampler
    class Dataset:
        m_items = 7
        @property
        def evidence_items(self):
            raise AssertionError("Evidence pool must not be accessed")
        def is_positive(self, users, items):
            return np.zeros(np.broadcast_arrays(users, items)[0].shape, dtype=bool)
    users = np.array([0, 1, 0])
    sampler = FeedbackSampler(Dataset(), "reliable", seed=23, evidence_fraction=0.0, reliability_beta=0.0)
    expected = np.random.default_rng(23).integers(7, size=(3, 20))
    assert np.array_equal(sampler.candidates(users), expected)


def test_source_summary_keeps_seed_matched_controls(tmp_path, monkeypatch):
    tool = module()
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    config = {"seeds": [2020, 2021, 2022], "training": {"epochs": 30}, "interpretation": "test"}
    references = {"reference_runs": {}}
    def history(run, ndcg, sparse):
        run.mkdir(parents=True)
        rows = [{"epoch": epoch, "samples": 100, "unique_user_negative_pairs": 90,
                 "mean_negative_evidence": 0.05, "explicit_negative_fraction": 0.0,
                 "validation": {"split": "val", "groups": {
                     "overall": {"ndcg": [ndcg], "recall": [ndcg]},
                     "user_sparse": {"ndcg": [sparse]},
                     "item_tail": {"ndcg": [0], "precision": [0], "users": 1}}}}
                for epoch in range(1, 31)]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for index, seed in enumerate(config["seeds"], 1):
        references["reference_runs"][str(seed)] = {"uniform": f"ref{seed}/uniform", "rate08": f"ref{seed}/mixed"}
        history(tmp_path / f"ref{seed}/uniform", index / 10, 2 * index / 10)
        history(tmp_path / f"ref{seed}/mixed", 1.2 * index / 10, index / 10)
        history(output / f"seed{seed}" / f"hard-seed{seed}", 1.1 * index / 10, 1.8 * index / 10)
    result = tool.summarize(output, config, references)
    assert result["completed_new_runs"] == 3
    assert set(result["aggregates"]) == {"uniform", "evidence05", "evidence0"}
    pairs = [p for p in result["paired_differences"] if p["reference"] == "uniform"]
    assert len(pairs) == 3
    assert all(abs(p["ndcg20_relative_percent"] - 10) < 1e-10 for p in pairs)
    assert all(abs(p["sparse_ndcg20_relative_percent"] + 10) < 1e-10 for p in pairs)
