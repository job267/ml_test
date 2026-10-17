"""Validation-only adapters with a fixed training graph and separate target views."""
import numpy as np
import scipy.sparse as sp
import torch
from common import ROOT, VENDOR, read, sha, verify_author
from dataloader import Loader


def pair_dict(users, items):
    result = {}
    for u, i in zip(users.tolist(), items.tolist()):
        result.setdefault(int(u), []).append(int(i))
    return result


class ValidationDataset(Loader):
    def __init__(self, n_users, n_items, train_users, train_items, validation, device="cpu"):
        self.n_users, self.m_items = int(n_users), int(n_items)
        self.device = torch.device(device)
        self.trainUser, self.trainItem = np.asarray(train_users, dtype=np.int64), np.asarray(train_items, dtype=np.int64)
        encoded = self.trainUser * self.m_items + self.trainItem
        if len(np.unique(encoded)) != len(encoded):
            raise ValueError("Duplicate training pair")
        self.train_keys = np.sort(encoded)
        self.UserItemNet = sp.csr_matrix((np.ones(len(encoded), dtype=np.float32), (self.trainUser, self.trainItem)), shape=(self.n_users, self.m_items))
        self.allPos = [self.UserItemNet[u].indices for u in range(self.n_users)]
        self._allPos = self.allPos
        self.active_users = np.array([u for u, pos in enumerate(self.allPos) if 0 < len(pos) < self.m_items], dtype=np.int64)
        self.trainDataSize = self.traindataSize = len(self.trainUser)
        self.n_user, self.m_item = self.n_users, self.m_items
        self.valDict = {int(u): sorted(set(map(int, items))) for u, items in validation.items() if len(items)}
        self.valid_dict = self.valDict
        self._graph = None
        self.evidence_items = [np.empty(0, dtype=np.int64) for _ in range(self.n_users)]
        self.evidence_kinds = {}
        self.item_inter = {i: [0] * 10 for i in range(self.m_items)}
        for u, items in self.valDict.items():
            if not 0 <= u < self.n_users or min(items) < 0 or max(items) >= self.m_items:
                raise ValueError("Invalid validation ID")
            if set(items) & set(self.allPos[u]):
                raise ValueError("Validation target overlaps training positive")

    @property
    def testDict(self):
        raise RuntimeError("This study never loads or evaluates task test labels")

    @property
    def test_dict(self):
        raise RuntimeError("This study never loads or evaluates author test labels")

    def getUserValidItems(self, users):
        # Exact author behavior is relevant only to validation here; flag=1 is forbidden.
        return [self.valDict[u] for u in users if u in self.valDict]

    def is_positive(self, users, items):
        keys = np.asarray(users) * self.m_items + np.asarray(items)
        positions = np.searchsorted(self.train_keys, keys)
        clipped = np.minimum(positions, len(self.train_keys) - 1)
        return (positions < len(self.train_keys)) & (self.train_keys[clipped] == keys)

    def negative_confidence(self, users, items, unknown_confidence=.05):
        # Uniform sampling never uses evidence in its selection or objective.
        return np.full(np.broadcast_arrays(users, items)[0].shape, unknown_confidence)


def author_dataset(device="cpu", validation_limit=0, seed=2020):
    names = ["kuairand/training_dict.npy", "kuairand/validation_dict.npy", "kuairand/s_pre_adj_mat.npz"]
    fingerprints = verify_author(names)
    # Trusted only after pinning and checking the author's exact blob bytes.
    train = np.load(VENDOR / names[0], allow_pickle=True).item()
    val = np.load(VENDOR / names[1], allow_pickle=True).item()
    users, items = [], []
    for u, values in train.items():
        users.extend([int(u)] * len(values)); items.extend(values)
    if validation_limit:
        keys = list(val)
        if validation_limit < len(keys):
            subset = set(np.random.default_rng(seed).choice(keys, validation_limit, replace=False).tolist())
            val = {u: values for u, values in val.items() if u in subset}
    ds = ValidationDataset(22128, 7076, users, items, val, device)
    ds.fingerprint = fingerprints
    return ds


def factorial_dataset(path, device="cpu"):
    with np.load(path, allow_pickle=False) as a:
        views = {source: pair_dict(a[f"val_{source}_users"], a[f"val_{source}_items"]) for source in ("random", "all")}
        ds = ValidationDataset(len(a["user_ids"]), len(a["item_ids"]), a["train_users"], a["train_items"], views["all"], device)
        ds.views = views
        ds.common_eval_users = set(a["common_eval_users"].tolist())
        ds.raw_user_ids = a["user_ids"].copy()
        ds.raw_item_ids = a["item_ids"].copy()
    ds.fingerprint = sha(path)
    return ds
