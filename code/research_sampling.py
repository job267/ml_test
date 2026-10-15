"""Fixed-budget candidates and detached epoch snapshots for A0--A3."""

from time import perf_counter
import numpy as np
import torch


def grouped_rows(users):
    """Group row indexes once, avoiding a full array scan for every user."""
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_users)) + 1, len(users)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        yield int(sorted_users[left]), order[left:right]


class FeedbackSampler:
    """One-negative BPR sampler; evidence weights are heuristics, not calibrated probabilities."""

    def __init__(
        self,
        dataset,
        method="uniform",
        seed=2020,
        candidate_size=20,
        evidence_fraction=0.5,
        unknown_confidence=0.05,
        reliability_beta=1.0,
        temperature=1.0,
        targeted_rate=0.8,
        curriculum_epochs=100,
        score_batch=8192,
    ):
        if method not in ("uniform", "hard", "reliable", "curriculum"):
            raise ValueError("Unknown sampler")
        if candidate_size < 1 or score_batch < 1 or curriculum_epochs < 1:
            raise ValueError("Candidate/score batch/ramp length must be positive")
        if not 0 <= evidence_fraction <= 1 or not 0 <= targeted_rate <= 1:
            raise ValueError("Fractions must be in [0,1]")
        if not 0 < unknown_confidence <= 1 or reliability_beta < 0 or temperature <= 0:
            raise ValueError("Invalid confidence, beta or temperature")
        self.dataset, self.method = dataset, method
        self.candidate_size, self.evidence_fraction = candidate_size, evidence_fraction
        self.unknown_confidence = unknown_confidence
        self.reliability_beta = reliability_beta
        self.temperature, self.targeted_rate = temperature, targeted_rate
        self.curriculum_epochs, self.score_batch = curriculum_epochs, score_batch
        self.rng = np.random.default_rng(seed)

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]

    def _legal_random(self, users, width):
        """Draw unobserved training items, with bounded rejection for dense users."""
        ds = self.dataset
        candidates = self.rng.integers(ds.m_items, size=(len(users), width))
        for _ in range(32):
            illegal = ds.is_positive(users[:, None], candidates)
            if not illegal.any():
                return candidates
            candidates[illegal] = self.rng.integers(ds.m_items, size=int(illegal.sum()))
        # Bounded rejection also terminates for users interacting with almost all items.
        illegal = ds.is_positive(users[:, None], candidates)
        for row in np.flatnonzero(illegal.any(axis=1)):
            legal = np.setdiff1d(
                np.arange(ds.m_items), ds.allPos[int(users[row])], assume_unique=True
            )
            candidates[row, illegal[row]] = self.rng.choice(legal, size=int(illegal[row].sum()))
        return candidates

    def candidates(self, users):
        """Use the same candidate-source budget for hard, reliable and curriculum."""
        result = self._legal_random(users, self.candidate_size)
        count = int(self.candidate_size * self.evidence_fraction)
        if count:
            for user, rows in grouped_rows(users):
                pool = self.dataset.evidence_items[user]
                if len(pool):
                    result[rows, :count] = self.rng.choice(
                        pool, size=(len(rows), count), replace=True
                    )
        return result  # draws with replacement, identical candidate rules across A1--A3

    def rate(self, epoch):
        """Fraction of triplets receiving targeted selection at this completed-epoch count."""
        if self.method == "uniform":
            return 0.0
        if self.method != "curriculum":
            return self.targeted_rate
        return self.targeted_rate * min(max(epoch, 0) / self.curriculum_epochs, 1.0)

    def select(self, users, candidates, embeddings):
        """Rank candidates using a detached snapshot; never access evaluation labels."""
        all_users, all_items = embeddings
        output = np.empty(len(users), dtype=np.int64)
        for start in range(0, len(users), self.score_batch):
            stop = start + self.score_batch
            subset, pool = users[start:stop], candidates[start:stop]
            u = torch.as_tensor(subset, device=all_users.device, dtype=torch.long)
            items = torch.as_tensor(pool, device=all_items.device, dtype=torch.long)
            with torch.no_grad():
                scores = (all_users[u, None, :] * all_items[items]).sum(-1).cpu().numpy()
            if not np.isfinite(scores).all():
                raise FloatingPointError("Non-finite sampler scores")
            if self.method in ("reliable", "curriculum"):
                confidences = self.dataset.negative_confidence(
                    np.broadcast_to(subset[:, None], pool.shape), pool, self.unknown_confidence
                )
                # Larger confidence adds a smaller penalty; it does not create a positive graph edge.
                scores = scores / self.temperature + self.reliability_beta * np.log(confidences)
            output[start : start + len(subset)] = pool[
                np.arange(len(subset)), scores.argmax(axis=1)
            ]
        return output

    def sample(self, model, epoch, sample_count=None):
        """Produce legal triplets and diagnostics, preserving the model's training mode."""
        begin = perf_counter()
        ds = self.dataset
        count = ds.trainDataSize if sample_count is None else sample_count
        if count < 1:
            raise ValueError("sample_count must be positive")
        users = self.rng.choice(ds.active_users, size=count)
        positives = np.empty(count, dtype=np.int64)
        for user, rows in grouped_rows(users):
            positives[rows] = self.rng.choice(ds.allPos[user], size=len(rows))
        negatives = self._legal_random(users, 1)[:, 0]
        rate = self.rate(epoch)
        selected = self.rng.random(count) < rate
        scored = 0
        propagation_seconds = 0.0
        if selected.any():
            pool = self.candidates(users[selected])
            previous_mode = model.training
            model.eval()
            snapshot_start = perf_counter()
            try:
                with torch.no_grad():
                    embeddings = tuple(value.detach() for value in model.computer())
                if embeddings[0].is_cuda:
                    torch.cuda.synchronize(embeddings[0].device)
                propagation_seconds = perf_counter() - snapshot_start
                negatives[selected] = self.select(users[selected], pool, embeddings)
            finally:
                model.train(previous_mode)
            scored = int(selected.sum()) * self.candidate_size
        triplets = np.column_stack((users, positives, negatives))
        if ds.is_positive(users, negatives).any():
            raise RuntimeError("Sampler produced a training positive as a negative")
        confidences = ds.negative_confidence(users, negatives, self.unknown_confidence)
        explicit = sum(
            ds.evidence_kinds.get(int(u) * ds.m_items + int(i)) == "explicit"
            for u, i in zip(users, negatives)
        )
        self.rng.shuffle(triplets)
        stats = {
            "sampler_seconds": perf_counter() - begin,
            "snapshot_seconds": propagation_seconds,
            "samples": count,
            "targeted_rate": rate,
            "targeted_samples": int(selected.sum()),
            "candidate_scores": scored,
            "mean_negative_evidence": float(confidences.mean()),
            "explicit_negative_fraction": explicit / count,
            "unique_user_negative_pairs": len(np.unique(users * ds.m_items + negatives)),
        }
        return triplets, stats
