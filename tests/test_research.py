"""Scientific protocol, sampler legality, evaluation and checkpoint regression tests."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_kuairand import prepare
from research_data import FeedbackDataset
from research_common import read_json
from research_sampling import FeedbackSampler
from research_evaluation import evaluate
from model import LightGCN, PureMF
from experiment import run, load_checkpoint

FIELDS = [
    "user_id",
    "video_id",
    "time_ms",
    "date",
    "tab",
    "is_rand",
    "is_click",
    "is_like",
    "is_follow",
    "is_hate",
    "long_view",
]


def event(user, item, day=20220420, **signals):
    record = dict.fromkeys(FIELDS, 0)
    record.update(
        user_id=user,
        video_id=item,
        date=day,
        time_ms=int(
            datetime.strptime(str(day), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        ),
    )
    record.update(signals)
    return record


def write_events(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        event(0, 0, is_like=1),
        event(0, 1, long_view=1),
        event(1, 1, is_like=1),
        event(1, 2, is_follow=1),
        event(2, 2, is_like=1),
        event(2, 3, is_like=1),
        event(0, 4, is_hate=1),
        event(0, 5),
        event(1, 0),
        event(2, 6, is_hate=1),
        event(0, 2, 20220430, is_like=1, is_rand=1),
        event(1, 3, 20220430, is_like=1, is_rand=1),
        event(2, 4, 20220430, is_like=1, is_rand=1),
        event(0, 6, 20220430, is_like=1),
        # A future positive must not remove training-only negative evidence.
        event(0, 4, 20220430, is_like=1, is_rand=1),
        event(0, 3, 20220504, is_like=1, is_rand=1),
        event(1, 4, 20220504, is_like=1, is_rand=1),
        event(2, 5, 20220504, is_like=1, is_rand=1),
        event(99, 3, 20220504, is_like=1, is_rand=1),
        event(0, 999, 20220504, is_like=1, is_rand=1),
    ]
    write_events(raw / "log_standard_pure.csv", rows)
    write_events(raw / "log_copy_pure.csv", [rows[0]])
    output = tmp_path / "prepared"
    prepare(raw, output)
    return output


def conf():
    return {
        "latent_dim_rec": 4,
        "lightGCN_n_layers": 2,
        "dropout": 0,
        "keep_prob": 1.0,
        "lr": 0.001,
        "decay": 0.0001,
    }


def test_temporal_evidence_vocabulary_and_duplicates(prepared):
    ds = FeedbackDataset(prepared)
    assert ds.n_users == 3 and ds.m_items == 7
    assert ds.trainDataSize == 6
    assert ds.negative_confidence(np.array([0]), np.array([4]))[0] == 1
    assert 4 in ds.valDict[0]  # training sampler cannot inspect this future label
    assert ds.manifest["audit"]["duplicates"] == 1
    assert ds.manifest["audit"]["eval_unseen_user"] == 1
    assert ds.manifest["audit"]["eval_unseen_item"] == 1
    assert 6 in ds.val_knownDict[0] and 6 not in ds.valDict[0]
    graph = ds.getSparseGraph().to_dense()
    raw = torch.zeros_like(graph)
    raw[torch.tensor(ds.trainUser), torch.tensor(ds.trainItem + ds.n_users)] = 1
    raw += raw.T.clone()
    degree = raw.sum(1)
    inv = torch.where(degree > 0, degree.rsqrt(), 0)
    assert torch.allclose(graph, inv[:, None] * raw * inv[None, :])
    assert graph[0, ds.n_users + 4] == 0  # evidence is not a graph edge


def test_prepare_refuses_overwrite_and_integrity_failure(prepared):
    with pytest.raises(FileExistsError):
        prepare(prepared.parent / "raw", prepared)
    with (prepared / "train.txt").open("a", encoding="utf-8") as handle:
        handle.write("0 4\n")
    with pytest.raises(ValueError, match="integrity"):
        FeedbackDataset(prepared)


def test_same_time_feedback_collision_preserves_both_and_positive_overrides(prepared):
    raw = prepared.parent / "raw"
    # Same tuple as the earlier explicit negative, but different actual feedback.
    write_events(raw / "log_collision_pure.csv", [event(0, 4, is_like=1)])
    output = prepared.parent / "collision_prepared"
    prepare(raw, output)
    dataset = FeedbackDataset(output)
    assert dataset.manifest["audit"]["coarse_key_collision_rows"] == 1
    assert dataset.manifest["audit"]["coarse_feedback_disagreement_rows"] == 1
    assert dataset.manifest["audit"]["duplicates"] == 1
    assert dataset.is_positive(np.array([0]), np.array([4])).all()
    assert 4 not in dataset.evidence_items[0]
    assert dataset.manifest["protocol"]["deduplication"] == "full_observation_v2"


def test_same_derived_label_with_different_raw_fields_is_not_dropped(prepared):
    raw = prepared.parent / "raw"
    path = raw / "log_details_pure.csv"
    first = dict(event(0, 5), play_time_ms=100)
    second = dict(event(0, 5), play_time_ms=200)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIELDS, "play_time_ms"])
        writer.writeheader()
        writer.writerows([first, second, first])
    output = prepared.parent / "details_prepared"
    prepare(raw, output)
    manifest = read_json(output / "manifest.json")
    assert manifest["audit"]["duplicates"] == 2  # exact duplicate row + original fixture copy
    assert manifest["audit"]["coarse_key_collision_rows"] == 2
    assert manifest["audit"]["coarse_feedback_disagreement_rows"] == 0
    with (output / "evidence.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    record = next(row for row in records if row["user"] == "0" and row["item"] == "5")
    assert int(record["exposures"]) == 3


def test_clicked_observation_blocks_weak_negative_evidence(prepared):
    raw = prepared.parent / "raw"
    # The earlier no-click observation is not sufficient negative evidence once a click exists.
    write_events(raw / "log_click_pure.csv", [event(0, 5, is_click=1)])
    output = prepared.parent / "clicked_prepared"
    prepare(raw, output)
    dataset = FeedbackDataset(output)
    assert not dataset.is_positive(np.array([0]), np.array([5])).any()
    assert 5 not in dataset.evidence_items[0]
    assert dataset.negative_confidence(np.array([0]), np.array([5]))[0] == pytest.approx(0.05)


@pytest.mark.parametrize("method", ["uniform", "hard", "reliable", "curriculum"])
def test_sampler_legality_determinism_and_detached_snapshot(prepared, method):
    ds = FeedbackDataset(prepared)
    net = LightGCN(conf(), ds)
    net.train()
    kwargs = dict(method=method, seed=42, candidate_size=5, targeted_rate=1.0, curriculum_epochs=2)
    a, b = FeedbackSampler(ds, **kwargs), FeedbackSampler(ds, **kwargs)
    triplets, stats = a.sample(net, 2, 80)
    repeated, _ = b.sample(net, 2, 80)
    assert np.array_equal(triplets, repeated)
    assert ds.is_positive(triplets[:, 0], triplets[:, 1]).all()
    assert not ds.is_positive(triplets[:, 0], triplets[:, 2]).any()
    assert all(parameter.grad is None for parameter in net.parameters())
    assert net.training
    state = a.state_dict()
    expected, _ = a.sample(net, 3, 20)
    b.load_state_dict(state)
    actual, _ = b.sample(net, 3, 20)
    assert np.array_equal(expected, actual)
    assert stats["samples"] == 80


def test_reliability_selects_evidence_and_beta_zero_recovers_hard(prepared):
    ds = FeedbackDataset(prepared)
    users = np.array([0])
    pool = np.array([[3, 4, 5]])  # unknown, explicit rejection, weak exposure
    embeddings = (
        torch.ones(3, 1),
        torch.tensor([[0.0], [0.0], [0.0], [2.0], [1.0], [1.0]] + [[0.0]]),
    )
    hard = FeedbackSampler(ds, "hard")
    reliable = FeedbackSampler(ds, "reliable")
    assert hard.select(users, pool, embeddings).tolist() == [3]
    assert reliable.select(users, pool, embeddings).tolist() == [4]
    zero = FeedbackSampler(ds, "reliable", reliability_beta=0)
    assert np.array_equal(
        zero.select(users, pool, embeddings), hard.select(users, pool, embeddings)
    )
    curriculum = FeedbackSampler(ds, "curriculum", targeted_rate=0.8, curriculum_epochs=10)
    assert curriculum.rate(0) == 0 and curriculum.rate(5) == 0.4 and curriculum.rate(100) == 0.8


def test_candidate_pool_equal_between_targeted_methods(prepared):
    ds = FeedbackDataset(prepared)
    users = np.array([0, 1, 2, 0])
    pools = [
        FeedbackSampler(ds, method, seed=123, candidate_size=8).candidates(users)
        for method in ("hard", "reliable", "curriculum")
    ]
    assert all(np.array_equal(pool, pools[0]) for pool in pools)
    assert not ds.is_positive(users[:, None], pools[0]).any()


def test_evaluation_masks_all_validation_known_positives(prepared):
    ds = FeedbackDataset(prepared)
    net = PureMF(conf(), ds)
    with torch.no_grad():
        net.embedding_user.weight.fill_(1)
        net.embedding_item.weight.copy_(torch.arange(7).float()[:, None].expand(7, 4))
    result = evaluate(ds, net, "test", [2], batch_size=2)
    # User 0's unseen items are 3 and 5 after masking 0,1,2,4,6; item 3 must hit.
    assert result["groups"]["overall"]["recall"][0] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="eligible"):
        evaluate(ds, net, "test", [3])


def experiment_args(prepared, output, method="reliable", epochs=2):
    return [
        "--dataset-dir",
        str(prepared),
        "--output-root",
        str(output),
        "--device",
        "cpu",
        "--epochs",
        str(epochs),
        "--sampler",
        method,
        "--recdim",
        "4",
        "--layer",
        "1",
        "--bpr-batch",
        "4",
        "--samples-per-epoch",
        "16",
        "--candidate-size",
        "4",
        "--score-batch",
        "8",
        "--eval-interval",
        "1",
        "--topks",
        "[2]",
        "--seed",
        "12",
    ]


def test_resume_matches_uninterrupted_and_does_not_evaluate_test(prepared, tmp_path):
    continuous = run(experiment_args(prepared, tmp_path / "full") + ["--run-id", "full"])
    interrupted = run(
        experiment_args(prepared, tmp_path / "parts", epochs=1) + ["--run-id", "parts"]
    )
    # Simulate an interrupted append after checkpoint commit.
    with (interrupted / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"epoch": 999}) + "\n")
    resumed = run(
        experiment_args(prepared, tmp_path / "parts") + ["--resume", str(interrupted / "last.pt")]
    )
    a, b = load_checkpoint(continuous / "last.pt"), load_checkpoint(resumed / "last.pt")
    assert all(torch.equal(a["model"][name], b["model"][name]) for name in a["model"])
    assert a["sampler"] == b["sampler"]
    rows = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert all(row["validation"]["split"] == "val" for row in rows)
    assert not list(resumed.rglob("test_evaluation.json"))
    assert not read_json(resumed / "completion.json")["test_evaluated"]
    # Evaluation uses checkpoint configuration, not default CLI model dimensions.
    target = run(
        [
            "--dataset-dir",
            str(prepared),
            "--output-root",
            str(tmp_path / "eval"),
            "--device",
            "cpu",
            "--eval-only",
            "--checkpoint",
            str(resumed / "last.pt"),
        ]
    )
    assert read_json(target / "test_evaluation.json")["completed_epochs"] == 2


def test_evaluation_rejects_changed_dataset(prepared, tmp_path):
    result = run(experiment_args(prepared, tmp_path / "runs", epochs=1) + ["--run-id", "one"])
    # A re-prepared protocol with the same matrices must still be a different fingerprint.
    manifest = read_json(prepared / "manifest.json")
    manifest["protocol"]["annotation"] = "different protocol"
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        run(
            [
                "--dataset-dir",
                str(prepared),
                "--eval-only",
                "--checkpoint",
                str(result / "last.pt"),
                "--device",
                "cpu",
            ]
        )


def test_suite_final_plan_and_summary(prepared, tmp_path, monkeypatch):
    from run_stage1 import main as run_suite
    from evaluate_stage1 import main as eval_suite
    from summarize_stage1 import summarize

    # Exercise real training/evaluation through the suite without repeatedly importing
    # PyTorch in child processes. The separate CLI smoke check covers process launching.
    import subprocess

    original_run = subprocess.run

    def run_training_in_process(command, **options):
        if len(command) > 1 and Path(command[1]).name == "experiment.py":
            run(command[2:])
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **options)

    monkeypatch.setattr(subprocess, "run", run_training_in_process)

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "methods": ["uniform", "hard", "reliable", "curriculum"],
                "seeds": [1],
                "training": {
                    "epochs": 1,
                    "recdim": 4,
                    "layer": 1,
                    "bpr_batch": 4,
                    "samples_per_epoch": 8,
                    "candidate_size": 3,
                    "topks": "[2]",
                    "eval_interval": 1,
                },
            }
        )
    )
    output = tmp_path / "suite"
    args = [
        "--dataset-dir",
        str(prepared),
        "--output-root",
        str(output),
        "--config",
        str(config),
        "--device",
        "cpu",
    ]
    run_suite(args)
    run_suite(args + ["--resume"])
    assert not list(output.rglob("test_evaluation.json"))
    eval_args = ["--suite-dir", str(output), "--device", "cpu"]
    eval_suite(eval_args)
    with pytest.raises(ValueError, match="already exists"):
        eval_suite(eval_args)
    eval_suite(eval_args + ["--resume"])
    report = summarize(output)
    assert set(report["aggregates"]) == {"uniform", "hard", "reliable", "curriculum"}
    assert report["aggregates"]["uniform"]["overall/ndcg@2"]["std"] is None
    assert (output / "summary.csv").exists()


def test_final_evaluation_rejects_changed_source(prepared, tmp_path, monkeypatch):
    result = run(experiment_args(prepared, tmp_path / "runs", epochs=1) + ["--run-id", "one"])
    monkeypatch.setattr("experiment.code_hashes", lambda: {"changed.py": "different"})
    with pytest.raises(ValueError, match="Evaluation source differs"):
        run(
            [
                "--dataset-dir",
                str(prepared),
                "--eval-only",
                "--checkpoint",
                str(result / "last.pt"),
                "--device",
                "cpu",
            ]
        )


def test_evaluation_rejects_nonfinite_embeddings(prepared):
    dataset = FeedbackDataset(prepared)
    model = PureMF(conf(), dataset)
    with torch.no_grad():
        model.embedding_item.weight[0].fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        evaluate(dataset, model, "val", [2])
