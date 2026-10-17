"""Build fixed-cohort label/exposure diagnosis from train+validation dates only."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from common import CONFIG, OUTPUT, ROOT, read, save, sha, utc
from prepare_kuairand import event_signature, flag, SIGNALS
from dataloader import Loader
from dataset import ValidationDataset, pair_dict, factorial_dataset
from evaluation import reference_metrics


def values(row):
    click, like, follow, hate, long_view = [flag(row[name], name) for name in SIGNALS]
    engagement = int(bool(like or follow or long_view))
    return int(engagement and not hate), int(click and not hate)


def arrays(train, all_targets, random_targets, user_ids, item_ids, common_eval_raw):
    um = {u: j for j, u in enumerate(user_ids)}
    im = {i: j for j, i in enumerate(item_ids)}
    def convert(pairs):
        selected = sorted((um[u], im[i]) for u, i in pairs if u in um and i in im)
        a = np.array(selected, dtype=np.int64).reshape(-1, 2)
        return a[:, 0], a[:, 1]
    u, i = convert(train)
    au, ai = convert(all_targets)
    ru, ri = convert(random_targets)
    return {"train_users": u, "train_items": i, "val_all_users": au, "val_all_items": ai,
            "val_random_users": ru, "val_random_items": ri,
            "user_ids": np.asarray(user_ids, dtype=np.int64), "item_ids": np.asarray(item_ids, dtype=np.int64),
            "common_eval_users": np.array(sorted(um[u] for u in common_eval_raw if u in um), dtype=np.int64)}


def native_reference_check(data, root=ROOT):
    """Reconstruct the unchanged engagement/random prepared split exactly."""
    folder = root / "data/kuairand_stage1"
    maps = read(folder / "id_maps.json")
    if data["user_ids"].tolist() != maps["users"] or data["item_ids"].tolist() != maps["items"]:
        raise AssertionError("Engagement training vocabulary differs from the original prepared data")
    for prefix, filename in (("train", "train.txt"), ("val_random", "val.txt")):
        u, i = Loader._load_pairs(folder / filename)
        if not np.array_equal(u, data[prefix + "_users"]) or not np.array_equal(i, data[prefix + "_items"]):
            raise AssertionError("Engagement/random reconstruction differs: " + filename)
    return {"original_manifest_sha256": sha(folder / "manifest.json"),
            "train_pairs_equal": True, "random_validation_pairs_equal": True,
            "vocabulary_equal": True, "test_file_loaded": False}


def describe(a, views=True):
    ud = np.bincount(a["train_users"], minlength=len(a["user_ids"]))
    degree = np.bincount(a["train_items"], minlength=len(a["item_ids"]))
    result = {"n_users": len(a["user_ids"]), "m_items": len(a["item_ids"]), "train_pairs": len(a["train_users"]),
              "minimum_user_training_positives": int(ud.min()), "zero_positive_degree_items": int((degree == 0).sum()),
              "common_eval_users": len(a["common_eval_users"]), "validation": {}}
    for source in ("random", "all"):
        users, items = a[f"val_{source}_users"], a[f"val_{source}_items"]
        degrees = degree[items]
        result["validation"][source] = {"users": len(np.unique(users)), "pairs": len(users),
            "low_degree_1_to_14_target_fraction": float(((degrees > 0) & (degrees <= 14)).mean()),
            "zero_degree_target_fraction": float((degrees == 0).mean()),
            "median_target_training_degree": float(np.median(degrees))}
    return result


def prepare(output, logs=None, c=None, check_original=True, references=True):
    output = Path(output)
    c = c or read(CONFIG)["factorial"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if logs is None:
        original = read(ROOT / "data/kuairand_stage1/manifest.json")
        logs = [ROOT / "data/raw" / r["name"] for r in original["raw_files"]]
        for record, path in zip(original["raw_files"], logs):
            if sha(path) != record["sha256"]:
                raise ValueError("Raw log differs from original prepared manifest")
    paths = [Path(p) for p in logs]
    fingerprints = {str(p.resolve()): sha(p) for p in paths}
    dbpath = output / "prefix_events.sqlite"
    db = sqlite3.connect(dbpath)
    db.execute("CREATE TABLE events (u INTEGER,i INTEGER,day INTEGER,r INTEGER,pe INTEGER,pc INTEGER,signature TEXT PRIMARY KEY)")
    audit = {"rows_read": 0, "future_rows_skipped_before_feedback_parsing": 0,
             "prefix_rows": 0, "duplicate_prefix_observations": 0}
    begin = time.perf_counter()
    try:
        for path in paths:
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                required = {"user_id", "video_id", "time_ms", "date", "tab", "is_rand", *SIGNALS}
                if not required <= set(reader.fieldnames or []):
                    raise ValueError("Missing KuaiRand fields: " + str(path))
                for row in reader:
                    audit["rows_read"] += 1
                    day = int(row["date"])
                    datetime.strptime(str(day), "%Y%m%d")
                    if day > c["val_end"]:
                        audit["future_rows_skipped_before_feedback_parsing"] += 1
                        continue
                    u, i, ts, tab = [int(row[k]) for k in ("user_id", "video_id", "time_ms", "tab")]
                    if min(u, i, tab) < 0 or ts <= 0:
                        raise ValueError("Invalid prefix observation")
                    randomized = flag(row["is_rand"], "is_rand")
                    pe, pc = values(row)
                    inserted = db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)",
                        (u, i, day, randomized, pe, pc, event_signature(row)))
                    audit["prefix_rows"] += 1
                    audit["duplicate_prefix_observations"] += int(not inserted.rowcount)
                    if audit["rows_read"] % 100000 == 0:
                        db.commit()
                        print("FACTORIAL_PREPARE", json.dumps(audit), flush=True)
            db.commit()
        item_ids = [int(x[0]) for x in db.execute("SELECT DISTINCT i FROM events WHERE day<=? ORDER BY i", (c["train_end"],))]
        positives = {"engagement": set(), "click": set()}
        for u, i, pe, pc in db.execute("SELECT u,i,MAX(pe),MAX(pc) FROM events WHERE day<=? GROUP BY u,i ORDER BY u,i", (c["train_end"],)):
            if pe:
                positives["engagement"].add((int(u), int(i)))
            if pc:
                positives["click"].add((int(u), int(i)))
        eligible = {}
        for label in ("engagement", "click"):
            counts = {}
            for u, _ in positives[label]:
                counts[u] = counts.get(u, 0) + 1
            eligible[label] = {u for u, n in counts.items() if n >= c["min_train_positives"]}
        common_train = sorted(eligible["engagement"] & eligible["click"])
        if not common_train or not item_ids:
            raise ValueError("Empty training-only common cohort/vocabulary")
        targets = {}
        item_set = set(item_ids)
        for label, column in (("engagement", "pe"), ("click", "pc")):
            all_targets, random_targets = set(), set()
            for u, i, randomized in db.execute(f"SELECT u,i,MAX(r) FROM events WHERE day>? AND {column}=1 GROUP BY u,i", (c["train_end"],)):
                pair = (int(u), int(i))
                if u not in eligible[label] or pair in positives[label] or i not in item_set:
                    continue
                all_targets.add(pair)
                if randomized:
                    random_targets.add(pair)
            targets[label] = {"all": all_targets, "random": random_targets}
        common_eval = set(common_train)
        for label in targets:
            for source in ("random", "all"):
                common_eval &= {u for u, _ in targets[label][source]}
        if not common_eval:
            raise ValueError("No shared validation users across the four views")
        saved, descriptions = {}, {}
        original_check = None
        for cohort in ("native", "common"):
            for label in ("engagement", "click"):
                ids = sorted(eligible[label]) if cohort == "native" else common_train
                a = arrays(positives[label], targets[label]["all"], targets[label]["random"], ids, item_ids, common_eval)
                name = f"{cohort}_{label}"
                np.savez_compressed(output / f"{name}.npz", **a)
                saved[f"{name}.npz"] = sha(output / f"{name}.npz")
                descriptions[name] = describe(a)
                if check_original and name == "native_engagement":
                    original_check = native_reference_check(a)
        common_a = np.load(output / "common_engagement.npz", allow_pickle=False)
        common_b = np.load(output / "common_click.npz", allow_pickle=False)
        for key in ("user_ids", "item_ids", "common_eval_users"):
            if not np.array_equal(common_a[key], common_b[key]):
                raise AssertionError("Controlled cohorts differ: " + key)
        sample_count = min(len(common_a["train_users"]), len(common_b["train_users"]))
        common_a.close(); common_b.close()
        reference_results = {}
        if references:
            for name in descriptions:
                ds = factorial_dataset(output / f"{name}.npz", "cpu")
                reference_results[name] = {}
                for source in ("random", "all"):
                    truth = ds.views[source]
                    common = {u: values for u, values in truth.items() if u in ds.common_eval_users}
                    reference_results[name][source] = {
                        "available_users": reference_metrics(ds, truth),
                        "common_users": reference_metrics(ds, common)}
            save(output / "validation_references.json", reference_results)
            saved["validation_references.json"] = sha(output / "validation_references.json")
        audit["unique_prefix_events"] = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        manifest = {"schema_version": "validation-factorial-v1", "created_utc": utc(),
            "config": c, "raw_sha256": fingerprints, "core_event_signature_sha256": sha(ROOT / "code/prepare_kuairand.py"),
            "driver_sha256": sha(__file__), "files": saved, "audit": audit,
            "cohorts": descriptions, "samples_per_epoch": sample_count,
            "original_engagement_random_crosscheck": original_check,
            "fit_user_selection": "Training-positive eligibility intersection only; validation common users affect reporting, never training graph/eligibility",
            "native_results": "Native complete-cohort Popular/random references and data diagnostics; neural factorial models train only on the shared training cohort",
            "future_feedback_parsed": False, "test_evaluated": False,
            "seconds": time.perf_counter() - begin}
        db.close(); dbpath.unlink()
        save(output / "manifest.json", manifest)
        print("FACTORIAL_PREPARED", json.dumps({"output": str(output), "cohorts": descriptions, "samples_per_epoch": sample_count, "crosscheck": original_check}), flush=True)
        return manifest
    except BaseException:
        db.close()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT / "data")
    args = p.parse_args()
    prepare(args.output)
