"""Check the diagnostic interpretation and selected-row reporting on synthetic data."""

import importlib.util
import json
from pathlib import Path
import numpy as np


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_penalty_choice_and_temperature_coupling():
    diagnose = module("diagnose_scores").selection_diagnostics
    scores = np.array([[0.5, 0.0], [2.0, 0.0]])
    confidence = np.array([[0.05, 1.0], [0.05, 1.0]])
    pool, users = np.array([[1, 2], [1, 2]]), np.array([0, 1])
    kinds = {2: "explicit", 12: "explicit"}
    zero = diagnose(scores, confidence, pool, users, kinds, 10, 0.0)
    assert zero["changed_item_fraction_vs_hard"] == 0
    assert zero["mean_raw_score_sacrificed"] == 0
    mild = diagnose(scores, confidence, pool, users, kinds, 10, 0.25)
    strong = diagnose(scores, confidence, pool, users, kinds, 10, 1.0)
    coupled = diagnose(scores, confidence, pool, users, kinds, 10, 0.125, 2.0)
    assert mild["changed_item_fraction_vs_hard"] == 0.5
    assert strong["changed_item_fraction_vs_hard"] == 1
    assert coupled["selected_kind_counts"] == mild["selected_kind_counts"]
    assert strong["selected_kind_counts"] == {"explicit": 2}


def test_summary_uses_same_best_row_for_recall_and_groups(tmp_path):
    summarize = module("run_pilot").summarize
    path = tmp_path / "uniform-seed2020"
    path.mkdir()
    def row(epoch, ndcg, recall):
        return {"epoch": epoch, "samples": 100, "unique_user_negative_pairs": 80,
                "mean_negative_evidence": 0.05, "explicit_negative_fraction": 0.0,
                "train_seconds": 1, "sampler_seconds": 2, "validation_seconds": 3,
                "validation": {"groups": {"overall": {"ndcg": [ndcg], "recall": [recall]},
                                           "user_sparse": {"ndcg": [ndcg / 2]}, "item_tail": {"ndcg": [0]}}}}
    (path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [row(1, 0.2, 0.1), row(2, 0.1, 0.4)]) + '\n{"epoch":')
    summary = summarize(tmp_path, {"seed": 2020, "variants": [{"id": "uniform"}]})[0]
    assert summary["best_epoch"] == 1
    assert summary["best_val_recall20"] == 0.1
    assert summary["last_val_ndcg20"] == 0.1
    assert summary["best_sparse_ndcg20"] == 0.1
