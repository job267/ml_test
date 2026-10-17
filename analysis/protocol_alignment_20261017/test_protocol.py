"""Cross-implementation and data-leakage regressions for the new diagnosis."""
import csv
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from common import read, CONFIG, author_definitions
from dataset import ValidationDataset, factorial_dataset
from alignment import baseline_config, model_pair, author_loss, project_loss, require_close, metrics_on_same_factors
from evaluation import rank, summarize_ranks, views
from prepare_factorial import prepare, values
from research_evaluation import evaluate
from run_sequence import extension_needed, prerequisite_state


def toy():
    return ValidationDataset(3, 110, [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 3], {0: [2], 1: [3], 2: [4]})


def test_pinned_author_and_project_losses_gradients_and_scores():
    c = {**read(CONFIG)["alignment"], "recdim": 8}
    ds = toy(); definitions = author_definitions(baseline_config(c), "cpu")
    a, b, same = model_pair(ds, c, definitions)
    assert same
    x = torch.tensor([[0, 0, 2], [1, 1, 3], [2, 3, 4]])
    la, lb = author_loss(a, x), project_loss(b, x, c["decay"])
    torch.testing.assert_close(la, lb, atol=1e-6, rtol=1e-6)
    la.backward(); lb.backward()
    for u, v in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(u.grad, v.grad, atol=1e-7, rtol=1e-5)
    ua, ia = a.computer(); ub, ib = b.computer()
    torch.testing.assert_close(ua @ ia.T, ub @ ib.T, atol=1e-7, rtol=1e-5)
    with torch.no_grad():
        b.embedding_item.weight[0, 0] += .1
    with pytest.raises(AssertionError):
        require_close("deliberately perturbed weights", a.embedding_item.weight, b.embedding_item.weight, 1e-5)


def test_actual_author_evaluator_matches_project_on_frozen_factors():
    c = {**read(CONFIG)["alignment"], "recdim": 8, "eval_batch": 2}
    ds = toy(); definitions = author_definitions(baseline_config(c), "cpu")
    _, model, _ = model_pair(ds, c, definitions)
    result = metrics_on_same_factors(ds, model, definitions, c)
    assert result["matches_after_author_rounding"] and not result["test_evaluated"]
    with pytest.raises(RuntimeError):
        _ = ds.test_dict


def test_shared_ranking_view_evaluator_matches_original():
    c = {**read(CONFIG)["alignment"], "recdim": 8}
    ds = toy(); definitions = author_definitions(baseline_config(c), "cpu")
    _, net, _ = model_pair(ds, c, definitions)
    users = sorted(ds.valDict)
    predicted = rank(net, ds, users, 20, 2)
    assert summarize_ranks(ds, users, predicted, ds.valDict) == evaluate(ds, net, "val", [20], 2)
    ds.views = {"all": ds.valDict, "random": {0: [2], 1: [3]}}
    ds.common_eval_users = {0, 1}
    result, _, _ = views(net, ds, 2)
    assert result["random"]["common_users"]["groups"]["overall"]["users"] == 2
    assert result["all"]["available_users"]["groups"]["overall"]["users"] == 3


def event(u, i, day=20220420, **flags):
    result = {"user_id": u, "video_id": i, "date": day, "time_ms": 1650000000000 + i,
              "tab": 0, "is_rand": 0, "is_click": 0, "is_like": 0,
              "is_follow": 0, "is_hate": 0, "long_view": 0}
    result.update(flags)
    return result


def example_events():
    rows = [event(10, 0, is_click=1), event(10, 1, is_click=1, is_like=1),
            event(10, 2, long_view=1), event(11, 0, is_click=1, long_view=1),
            event(11, 1, is_click=1, is_like=1), event(12, 0, is_click=1), event(12, 2, is_click=1),
            event(10, 3, 20220430, is_rand=1, is_click=1, long_view=1),
            event(11, 4, 20220430, is_rand=1, is_click=1, long_view=1),
            event(10, 5, 20220430, is_click=1),
            event(11, 6, 20220430, is_click=1, is_like=1, is_hate=1)]
    # Train exposure vocabulary includes items with zero positive graph degree.
    rows += [event(999, i) for i in range(3, 35)]
    rows += [rows[0].copy()]  # Exact duplicate is not another observation.
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_factorial_common_cohort_and_future_invariance(tmp_path):
    raw = tmp_path / "log_pure.csv"
    rows = example_events(); write_csv(raw, rows)
    c = read(CONFIG)["factorial"]
    first = prepare(tmp_path / "first", [raw], c, check_original=False, references=False)
    # Future labels deliberately malformed: they must never be parsed by this study.
    write_csv(raw, rows + [event(777, 500, 20220508, is_click="INVALID")])
    second = prepare(tmp_path / "second", [raw], c, check_original=False, references=False)
    assert second["audit"]["future_rows_skipped_before_feedback_parsing"] == 1
    assert first["audit"]["duplicate_prefix_observations"] == 1
    for name in ["native_engagement", "native_click", "common_engagement", "common_click"]:
        with np.load(tmp_path / "first" / f"{name}.npz") as a, np.load(tmp_path / "second" / f"{name}.npz") as b:
            assert set(a.files) == set(b.files)
            assert all(np.array_equal(a[k], b[k]) for k in a.files)
    a = factorial_dataset(tmp_path / "first/common_engagement.npz")
    b = factorial_dataset(tmp_path / "first/common_click.npz")
    assert a.raw_user_ids.tolist() == b.raw_user_ids.tolist() == [10, 11]
    assert a.raw_item_ids.tolist() == b.raw_item_ids.tolist() == list(range(35))
    assert a.common_eval_users == b.common_eval_users == {0, 1}
    assert 5 not in a.views["all"].get(0, []) and 5 in b.views["all"][0]
    assert 6 not in a.views["all"].get(1, []) and 6 not in b.views["all"].get(1, [])
    for dataset in (a, b):
        for u, targets in dataset.views["random"].items():
            assert set(targets) <= set(dataset.views["all"][u])
    assert first["samples_per_epoch"] == second["samples_per_epoch"]


def test_label_conflicts_have_same_semantics_as_core():
    r = {k: str(v) for k, v in event(0, 0, is_click=1, is_like=1, is_hate=1).items()}
    assert values(r) == (0, 0)
    r["is_hate"] = "0"
    assert values(r) == (1, 1)


def test_queue_requires_explicit_prerequisite_completion(tmp_path):
    import json
    state = tmp_path / "status.json"
    state.write_text(json.dumps({"status": "running", "queue_pid": -999}))
    assert prerequisite_state(tmp_path)[0] is False
    state.write_text(json.dumps({"status": "completed"}))
    assert prerequisite_state(tmp_path)[0] is False
    (tmp_path / "completion.json").write_text("{}")
    assert prerequisite_state(tmp_path)[0] is True
    state.write_text(json.dumps({"status": "failed", "error": "sample failure"}))
    with pytest.raises(RuntimeError):
        prerequisite_state(tmp_path)


def test_extension_is_committed_from_both_validation_windows_only():
    def rows(peak):
        return [{"epoch": e, "validation_views": {s: {"common_users": {"groups": {"overall": {
            "ndcg": [1. / (1 + abs(e - peak))]}}}} for s in ("random", "all")}} for e in range(1, 31)]
    assert extension_needed({"engagement": rows(10), "click": rows(20)}) == (False, [])
    decision, reasons = extension_needed({"engagement": rows(10), "click": rows(28)})
    assert decision and all(r["label"] == "click" and r["best_epoch"] == 28 for r in reasons)
    with pytest.raises(ValueError):
        extension_needed({"engagement": rows(10)})
    with pytest.raises(ValueError):
        extension_needed({"engagement": rows(10), "click": rows(28)[:-1]})


def test_factorial_actual_runner_resume_matches_uninterrupted(tmp_path):
    import shutil
    from factorial import run
    raw = tmp_path / "log_pure.csv"; write_csv(raw, example_events())
    c = {**read(CONFIG)["factorial"], "recdim": 4, "layer": 1, "batch_size": 2,
         "eval_batch": 2, "initial_epochs": 1, "maximum_epochs": 2}
    one, two = tmp_path / "one", tmp_path / "two"
    prepare(one / "data", [raw], c, check_original=False, references=False)
    shutil.copytree(one / "data", two / "data")
    run("engagement", 2, config=c, output_root=one, device="cpu")
    run("engagement", 1, config=c, output_root=two, device="cpu")
    # Emulate an uncommitted trailing metric after a crash; checkpoint remains epoch 1.
    with (two / "factorial/engagement/metrics.jsonl").open("a") as f:
        f.write('{"epoch": 999}\n')
    run("engagement", 2, True, config=c, output_root=two, device="cpu")
    a = torch.load(one / "factorial/engagement/last.pt", map_location="cpu", weights_only=True)
    b = torch.load(two / "factorial/engagement/last.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["sampler"] == b["sampler"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    import json
    rows = [json.loads(l) for l in (two / "factorial/engagement/metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[-1]["validation_views"] == a["metrics_row"]["validation_views"]
