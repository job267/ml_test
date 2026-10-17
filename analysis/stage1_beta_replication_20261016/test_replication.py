"""Check common-horizon selection and separation of exploratory and new seeds."""

import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).with_name("replicate.py")
    spec = importlib.util.spec_from_file_location("replication", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(epoch, ndcg, recall):
    return {
        "epoch": epoch, "samples": 100, "unique_user_negative_pairs": 60,
        "validation": {"split": "val", "groups": {
            "overall": {"ndcg": [ndcg], "recall": [recall]},
            "user_sparse": {"ndcg": [recall]},
            "item_tail": {"ndcg": [0], "precision": [0.05], "users": 1},
        }},
    }


def test_window_excludes_future_peak_and_preserves_selected_groups():
    select = module().window_result
    rows = [row(i, 0.5 if i == 7 else 0.1, 0.2 if i == 7 else 0.8) for i in range(1, 32)]
    rows[-1] = row(31, 0.9, 0.9)
    result = select(rows, 30)
    assert result["best_epoch"] == 7
    assert result["recall20"] == result["sparse_ndcg20"] == 0.2
    assert result["tail_hit_pairs"] == 1
    assert not result["late_best"]
    assert select(rows[:29], 30) is None


def test_new_seed_aggregation_excludes_selected_reference_and_pairs_by_seed():
    tool = module()
    rows = []
    for seed, reference in [(2020, 100), (2021, 1), (2022, 2)]:
        for variant, delta in [("uniform", 0), ("beta0", 0.1), ("beta025", 0.2)]:
            rows.append({"seed": seed, "variant": variant, **dict.fromkeys(
                ("ndcg20", "recall20", "sparse_ndcg20", "tail_ndcg20", "unique_negative_pair_fraction"), reference + delta)})
    result = tool.aggregate(rows, [2021, 2022])
    assert result["uniform"]["ndcg20"]["mean"] == 1.5
    differences = tool.paired(rows, [2021, 2022])["beta025-minus-uniform"]
    assert [p["seed"] for p in differences] == [2021, 2022]
    assert all(abs(p["ndcg20"] - 0.2) < 1e-12 for p in differences)
    assert not tool.aggregate([r for r in rows if r["seed"] != 2022], [2021, 2022])
