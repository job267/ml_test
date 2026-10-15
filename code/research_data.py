"""Audited prepared feedback data; training graph never consumes held-out labels."""

import csv
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

from dataloader import Loader
from research_common import read_json, sha256


def pair_dict(users, items):
    result = {}
    for user, item in zip(users.tolist(), items.tolist()):
        result.setdefault(user, []).append(item)
    return {u: sorted(set(values)) for u, values in result.items()}


class FeedbackDataset(Loader):
    """Prepared training graph plus separate evidence and held-out ranking targets."""

    def __init__(self, path, device="cpu", weak_confidence=0.2):
        if not 0 <= weak_confidence <= 1:
            raise ValueError("weak_confidence must be in [0,1]")
        self.path, self.device = Path(path), torch.device(device)
        self.manifest = read_json(self.path / "manifest.json")
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported prepared dataset schema")
        required = {
            "train.txt",
            "val.txt",
            "test.txt",
            "val_known.txt",
            "evidence.csv",
            "id_maps.json",
        }
        if not required <= set(self.manifest["files"]):
            raise ValueError("Prepared manifest does not cover required files")
        for name, digest in self.manifest["files"].items():
            if Path(name).name != name or sha256(self.path / name) != digest:
                raise ValueError(f"Prepared data integrity failure: {name}")
        self.fingerprint = sha256(self.path / "manifest.json")
        self.n_users, self.m_items = self.manifest["n_users"], self.manifest["m_items"]
        if min(self.n_users, self.m_items) < 1:
            raise ValueError("Empty declared vocabulary")
        pairs = {}
        for split in ("train", "val", "test", "val_known"):
            users, items = self._load_pairs(self.path / f"{split}.txt")
            if len(users) and (users.max() >= self.n_users or items.max() >= self.m_items):
                raise ValueError(f"{split} contains out-of-vocabulary IDs")
            encoded = users * self.m_items + items
            if len(np.unique(encoded)) != len(encoded):
                raise ValueError(f"{split} contains duplicate pairs")
            pairs[split] = set(encoded.tolist())
            setattr(self, f"{split}Dict", pair_dict(users, items))
            if split == "train":
                self.trainUser, self.trainItem = users, items
        if not pairs["train"] or not pairs["val"] or not pairs["test"]:
            raise ValueError("Training, validation and test must be nonempty")
        if pairs["train"] & (pairs["val"] | pairs["test"]) or pairs["val"] & pairs["test"]:
            raise ValueError("train/val/test interaction overlap")
        if not pairs["val"] <= pairs["val_known"] or pairs["test"] & pairs["val_known"]:
            raise ValueError("Invalid validation-known masking targets")
        self.UserItemNet = sp.csr_matrix(
            (np.ones(len(self.trainUser), dtype=np.float32), (self.trainUser, self.trainItem)),
            shape=(self.n_users, self.m_items),
        )
        self.allPos = [self.UserItemNet[u].indices for u in range(self.n_users)]
        self.trainDataSize = len(self.trainUser)
        self.train_keys = np.sort(self.trainUser * self.m_items + self.trainItem)
        self.active_users = np.array(
            [u for u, pos in enumerate(self.allPos) if 0 < len(pos) < self.m_items], dtype=np.int64
        )
        if not len(self.active_users):
            raise ValueError("No users with both positives and legal negatives")
        evidence_rows, evidence_items, evidence_values = [], [], []
        self.evidence_items = [[] for _ in range(self.n_users)]
        self.evidence_kinds = {}
        with (self.path / "evidence.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                u, i = int(row["user"]), int(row["item"])
                if not (0 <= u < self.n_users and 0 <= i < self.m_items):
                    raise ValueError("Evidence ID outside training vocabulary")
                key = u * self.m_items + i
                if key in pairs["train"] or key in self.evidence_kinds:
                    raise ValueError("Evidence overlaps a train positive or repeats a pair")
                kind = row["kind"]
                if kind not in ("explicit", "weak"):
                    raise ValueError(f"Unknown evidence kind: {kind}")
                self.evidence_kinds[key] = kind
                confidence = 1.0 if kind == "explicit" else weak_confidence
                if confidence > 0:
                    evidence_rows.append(u)
                    evidence_items.append(i)
                    evidence_values.append(confidence)
                    self.evidence_items[u].append(i)
        self.evidence_items = [np.asarray(items, dtype=np.int64) for items in self.evidence_items]
        self.confidence = sp.csr_matrix(
            (np.asarray(evidence_values, dtype=np.float32), (evidence_rows, evidence_items)),
            shape=(self.n_users, self.m_items),
        )
        self._graph = None

    def is_positive(self, users, items):
        """Vectorized membership in the training positives, including candidate matrices."""
        keys = np.asarray(users) * self.m_items + np.asarray(items)
        positions = np.searchsorted(self.train_keys, keys)
        clipped = np.minimum(positions, len(self.train_keys) - 1)
        return (positions < len(self.train_keys)) & (self.train_keys[clipped] == keys)

    def negative_confidence(self, users, items, unknown_confidence=0.05):
        """Read only training evidence; unknown pairs receive a declared fallback weight."""
        rows, columns = np.broadcast_arrays(np.asarray(users), np.asarray(items))
        values = np.asarray(self.confidence[rows.ravel(), columns.ravel()]).reshape(columns.shape)
        return np.where(values > 0, values, unknown_confidence)
