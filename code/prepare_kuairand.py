"""Prepare a temporal, train-vocabulary-only KuaiRand experiment without overwrites."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sqlite3
import shutil
import uuid

from research_common import sha256, write_json


SIGNALS = ("is_click", "is_like", "is_follow", "is_hate", "long_view")


def event_signature(row):
    """Identify equal log observations, not just equal user/item/time tuples.

    KuaiRand documents time_ms as a timestamp, not as a unique event identifier.
    Preserve records with different feedback, exposure provenance, or other fields.
    Equal row contents are duplicate observations; the coarse tuple is only audited.
    """
    if None in row or any(value is None for value in row.values()):
        raise ValueError("CSV row has extra or missing values")
    observation = {key: value.strip() for key, value in row.items()}
    for key in ("user_id", "video_id", "time_ms", "date", "tab"):
        observation[key] = str(int(row[key]))
    for key in (*SIGNALS, "is_rand"):
        observation[key] = str(flag(row[key], key))
    payload = json.dumps(observation, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def store_observation(db, record, signature, audit, source, line):
    """Deduplicate equal observations and audit, rather than discard, key collisions."""
    inserted = db.execute(
        "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", (*record, signature)
    )
    user, item, timestamp, tab, day, positive, hate, weak, randomized = record
    if not inserted.rowcount:
        audit["duplicates"] += 1
        return

    previous_count, disagreement = db.execute(
        "SELECT COUNT(*), MAX(day<>? OR p<>? OR h<>? OR w<>?) FROM events "
        "WHERE u=? AND i=? AND ts=? AND tab=? AND signature<>?",
        (day, positive, hate, weak, user, item, timestamp, tab, signature),
    ).fetchone()
    if previous_count:
        audit["coarse_key_collision_rows"] += 1
        audit["coarse_feedback_disagreement_rows"] += int(bool(disagreement))
        if len(audit["collision_examples"]) < 5:
            audit["collision_examples"].append(
                {
                    "source": source,
                    "line": line,
                    "user": user,
                    "item": item,
                    "time_ms": timestamp,
                    "tab": tab,
                    "feedback_differs": bool(disagreement),
                }
            )


def flag(value, name):
    if value not in ("0", "1", "0.0", "1.0"):
        raise ValueError(f"{name} must be a binary flag, got {value!r}")
    return int(float(value))


def write_pairs(path, pairs):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        previous = None
        items = []
        for user, item in sorted(pairs):
            if user != previous and previous is not None:
                handle.write(f"{previous} " + " ".join(map(str, items)) + "\n")
                items = []
            previous = user
            items.append(item)
        if previous is not None:
            handle.write(f"{previous} " + " ".join(map(str, items)) + "\n")


def prepare(
    raw_root,
    output,
    train_end=20220428,
    val_end=20220503,
    min_train_positives=2,
    positive_rule="engagement",
    eval_source="random",
):
    """Evidence is aggregated only before train_end; future positive labels stay private."""
    raw_root, output = Path(raw_root), Path(output)
    for date in (train_end, val_end):
        datetime.strptime(str(date), "%Y%m%d")
    if train_end >= val_end or min_train_positives < 1:
        raise ValueError("Require train_end < val_end and min_train_positives >= 1")
    if positive_rule not in ("engagement", "click") or eval_source not in ("random", "all"):
        raise ValueError("Invalid positive rule or evaluation source")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new prepared directory")
    logs = sorted(raw_root.rglob("log_*.csv"))
    if not logs:
        raise FileNotFoundError(f"No log_*.csv found under {raw_root}")
    versions = {
        ("pure" if "pure" in p.stem.lower() else "1k" if "1k" in p.stem.lower() else "full")
        for p in logs
    }
    if len(versions) != 1:
        raise ValueError("Do not mix Pure, 1K and full logs in one raw directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f"{output.name}.preparing-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    database_path = staging / "events.sqlite"
    db = sqlite3.connect(database_path)
    audit = {
        "rows_read": 0,
        "duplicates": 0,
        "coarse_key_collision_rows": 0,
        "coarse_feedback_disagreement_rows": 0,
        "collision_examples": [],
        "conflicting_feedback_events": 0,
        "eval_unseen_user": 0,
        "eval_unseen_item": 0,
        "eval_seen_positive": 0,
    }
    raw_files = []
    try:
        db.execute(
            "CREATE TABLE events (u INTEGER,i INTEGER,ts INTEGER,tab INTEGER,"
            "day INTEGER,p INTEGER,h INTEGER,w INTEGER,r INTEGER,"
            "signature TEXT PRIMARY KEY)"
        )
        db.execute("CREATE INDEX event_time_key ON events(u,i,ts,tab)")
        for log in logs:
            raw_files.append({"name": str(log.relative_to(raw_root)), "sha256": sha256(log)})
            with log.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"user_id", "video_id", "time_ms", "date", "tab", "is_rand", *SIGNALS}
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"{log}: missing columns {sorted(missing)}")
                for line, row in enumerate(reader, 2):
                    audit["rows_read"] += 1
                    try:
                        u, i, ts, tab, day = (
                            int(row[k]) for k in ("user_id", "video_id", "time_ms", "tab", "date")
                        )
                        click, like, follow, hate, long_view = (flag(row[k], k) for k in SIGNALS)
                        randomized = flag(row["is_rand"], "is_rand")
                        if min(u, i, tab) < 0 or ts <= 0:
                            raise ValueError("IDs must be nonnegative and time_ms positive")
                        datetime.strptime(str(day), "%Y%m%d")
                        # Engagement combines strong actions and long_view, not an inferred dislike.
                        positive = (
                            int(bool(like or follow or long_view))
                            if positive_rule == "engagement"
                            else click
                        )
                        if positive and hate:
                            audit["conflicting_feedback_events"] += 1
                            positive, hate = 0, 0  # exclude contradictory event from either label
                            weak = 0
                        else:
                            weak = int(not (click or like or follow or long_view or hate))
                        record = (u, i, ts, tab, day, positive, hate, weak, randomized)
                        store_observation(
                            db,
                            record,
                            event_signature(row),
                            audit,
                            source=str(log.relative_to(raw_root)),
                            line=line,
                        )
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"{log}:{line}: {exc}") from exc
                    if audit["rows_read"] % 100000 == 0:
                        db.commit()
                        print(f'PREPARE rows={audit["rows_read"]}', flush=True)
            db.commit()
        db.execute(
            "CREATE TABLE train_pairs AS SELECT u,i,MAX(p) AS p,MAX(h) AS h,"
            "MIN(w) AS w,COUNT(*) AS exposures,MAX(ts) AS last_time_ms "
            "FROM events WHERE day<=? GROUP BY u,i",
            (train_end,),
        )
        active_users = [
            row[0]
            for row in db.execute(
                "SELECT u FROM train_pairs WHERE p=1 GROUP BY u HAVING COUNT(*)>=? ORDER BY u",
                (min_train_positives,),
            )
        ]
        item_ids = [row[0] for row in db.execute("SELECT DISTINCT i FROM train_pairs ORDER BY i")]
        if not active_users or not item_ids:
            raise ValueError("No eligible training users/items; inspect cutoffs and positive rule")
        user_map = {value: index for index, value in enumerate(active_users)}
        item_map = {value: index for index, value in enumerate(item_ids)}
        train = set()
        evidence = []
        for u, i, pos, hate, weak, exposures, last_time in db.execute(
            "SELECT * FROM train_pairs ORDER BY u,i"
        ):
            if u not in user_map:
                continue
            pair = (user_map[u], item_map[i])
            if pos:
                train.add(
                    pair
                )  # any train-period positive overrides negative evidence for this pair
            elif hate or weak:
                evidence.append((*pair, "explicit" if hate else "weak", exposures, last_time))
        val, test = set(), set()
        condition = "AND r=1" if eval_source == "random" else ""
        for u, i, day in db.execute(
            f"SELECT u,i,MIN(day) FROM events WHERE day>? AND p=1 {condition} "
            "GROUP BY u,i ORDER BY u,i",
            (train_end,),
        ):
            if u not in user_map:
                audit["eval_unseen_user"] += 1
                continue
            if i not in item_map:
                audit["eval_unseen_item"] += 1
                continue
            pair = (user_map[u], item_map[i])
            if pair in train:
                audit["eval_seen_positive"] += 1
                continue
            (val if day <= val_end else test).add(pair)
        # A future test event must not repeat a validation positive, even outside the evaluation source.
        val_known = set()
        for u, i in db.execute(
            "SELECT DISTINCT u,i FROM events WHERE day>? AND day<=? AND p=1", (train_end, val_end)
        ):
            if u in user_map and i in item_map:
                val_known.add((user_map[u], item_map[i]))
        removed = test & val_known
        audit["test_previous_val_positive_removed"] = len(removed)
        test -= val_known
        if not val or not test:
            raise ValueError("Empty validation/test targets; adjust cutoffs or evaluation source")
        for name, pairs in [
            ("train.txt", train),
            ("val.txt", val),
            ("test.txt", test),
            ("val_known.txt", val_known),
        ]:
            write_pairs(staging / name, pairs)
        with (staging / "evidence.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["user", "item", "kind", "exposures", "last_time_ms"])
            writer.writerows(evidence)
        write_json(staging / "id_maps.json", {"users": active_users, "items": item_ids})
        audit.update(
            {
                "unique_events": db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "train_pairs": len(train),
                "val_pairs": len(val),
                "test_pairs": len(test),
                "train_users": len(active_users),
                "train_items": len(item_ids),
                "explicit_evidence_pairs": sum(r[2] == "explicit" for r in evidence),
                "weak_evidence_pairs": sum(r[2] == "weak" for r in evidence),
                "train_val_overlap": len(train & val),
                "train_test_overlap": len(train & test),
                "val_test_overlap": len(val & test),
                "zero_positive_degree_items": len(item_ids) - len({i for _, i in train}),
            }
        )
        db.close()
        database_path.unlink()
        files = {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "dataset": "kuairand",
                "raw_files": raw_files,
                "protocol": {
                    "train_end": train_end,
                    "val_end": val_end,
                    "positive_rule": positive_rule,
                    "eval_source": eval_source,
                    "min_train_positives": min_train_positives,
                    "vocabulary": "training_events_only",
                    "test_mask": "train_plus_val_known",
                    "deduplication": "full_observation_v2",
                    "weak_evidence": "all_train_observations_weak",
                },
                "n_users": len(active_users),
                "m_items": len(item_ids),
                "files": files,
                "audit": audit,
            },
        )
        # Publish the manifest last. Avoid directory renames (restricted on some Windows hosts).
        output.mkdir(exist_ok=False)
        publish = sorted(p for p in staging.iterdir() if p.name != "manifest.json")
        publish.append(staging / "manifest.json")
        for source in publish:
            shutil.copyfile(source, output / source.name)
            source.unlink()
        staging.rmdir()
        print(f"PREPARED {output}\nAUDIT {audit}", flush=True)
        return output
    except Exception:
        db.close()
        print(f"Preparation failed; diagnostic staging directory retained: {staging}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=int, default=20220428)
    parser.add_argument("--val-end", type=int, default=20220503)
    parser.add_argument("--min-train-positives", type=int, default=2)
    parser.add_argument("--positive-rule", choices=["engagement", "click"], default="engagement")
    parser.add_argument("--eval-source", choices=["random", "all"], default="random")
    args = parser.parse_args()
    prepare(
        args.raw_root,
        args.output,
        args.train_end,
        args.val_end,
        args.min_train_positives,
        args.positive_rule,
        args.eval_source,
    )


if __name__ == "__main__":
    main()
