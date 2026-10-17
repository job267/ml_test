"""Meaningful checks: MF gradient equivalence, ItemCF mathematics and masks."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from shared import ROOT
from itemcf import ItemCF, cosine_neighbors, similarity_matrix
from model import LightGCN, PureMF
from research_evaluation import evaluate
from public_run import author_stop, capture_rng, restore_rng
from public_evaluation import validation_mask_rows, evaluate_public_test
from public_protocol import VENDOR


def toy():
    x = sp.csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0]], dtype=np.float32))
    return SimpleNamespace(n_users=3, m_items=4, UserItemNet=x,
        allPos=[x[u].indices for u in range(3)], valDict={0: [2], 1: [1], 2: [3]},
        testDict={0: [3], 1: [3]}, val_knownDict={0: [2], 1: [1]},
        getSparseGraph=lambda: torch.eye(7).to_sparse())


def test_layer_zero_is_mf_with_equal_gradients_and_initialization():
    ds = toy()
    c = {"latent_dim_rec": 5, "lightGCN_n_layers": 0, "dropout": 0, "keep_prob": 1.}
    torch.manual_seed(2020)
    zero = LightGCN(c, ds)
    torch.manual_seed(2020)
    control = LightGCN({**c, "lightGCN_n_layers": 3}, ds)
    mf = PureMF(c, ds)
    mf.load_state_dict(zero.state_dict())
    assert all(torch.equal(x, y) for x, y in zip(zero.parameters(), control.parameters()))
    u, p, n = torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([2, 3])
    for net in (zero, mf):
        loss, reg = net.bpr_loss(u, p, n)
        (loss + .0001 * reg).backward()
    assert torch.allclose(zero.bpr_loss(u, p, n)[0], mf.bpr_loss(u, p, n)[0])
    assert all(torch.allclose(a.grad, b.grad, atol=1e-8) for a, b in zip(zero.parameters(), mf.parameters()))


@pytest.mark.parametrize("weighting", ["cosine", "iuf"])
def test_itemcf_similarity_and_score_match_direct_formula(weighting):
    x = toy().UserItemNet
    neighbors = cosine_neighbors(x, weighting, 4)
    actual = similarity_matrix(neighbors, 4, 4).toarray()
    dense = x.toarray().astype(np.float64)
    w = np.ones(3) if weighting == "cosine" else 1 / np.log1p(dense.sum(1))
    gram = dense.T @ np.diag(w) @ dense
    norms = np.sqrt(np.diag(gram))
    denom = norms[:, None] * norms[None, :]
    expected = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    model = ItemCF(x, sp.csr_matrix(actual), "cpu")
    u, i = model.computer()
    torch.testing.assert_close(u[torch.arange(3)] @ i.T, torch.tensor(dense @ actual.T, dtype=torch.float32))
    # Exact ties in neighbor similarity are resolved by item ID.
    assert neighbors[0][0].tolist() == [1, 2]


def test_itemcf_reuses_original_validation_and_test_masks():
    ds = toy()
    sim = similarity_matrix(cosine_neighbors(ds.UserItemNet, "cosine", 2), 2, 4)
    sparse = ItemCF(ds.UserItemNet, sim, "cpu")
    dense = SimpleNamespace(eval=lambda: None, computer=lambda:
        (torch.tensor(ds.UserItemNet.toarray()), torch.tensor(sim.toarray())))
    for split in ("val", "test"):
        assert evaluate(ds, sparse, split, [1], 2) == evaluate(ds, dense, split, [1], 2)
    changed = copy.deepcopy(ds)
    changed.valDict, changed.testDict = {0: [3]}, {0: [2]}
    a = cosine_neighbors(ds.UserItemNet, "cosine", 2)
    b = cosine_neighbors(changed.UserItemNet, "cosine", 2)
    assert all(np.array_equal(x, y) for aa, bb in zip(a, b) for x, y in zip(aa, bb))


def test_public_patience_matches_author_ties_and_boundary():
    assert author_stop(99, .1, .2, 19) == (19, False)
    assert author_stop(104, .1, .2, 19) == (20, True)
    assert author_stop(104, .2, .2, 19) == (1, False)


def test_public_mask_preserves_author_bug_and_corrects_alignment():
    validation = {5: [2], 7: [4]}
    assert validation_mask_rows([5, 6, 7], validation, True) == [[2], [4]]
    assert validation_mask_rows([5, 6, 7], validation, False) == [[2], [], [4]]


def test_public_two_masks_share_one_score_pass_and_match_legacy_indices():
    # User 1 has no validation record; legacy masks user 2's item on row 1.
    ds = SimpleNamespace(m_item=110, test_dict={0: [1], 1: [2], 2: [3]},
        valid_dict={0: [7], 2: [2]}, item_inter={i: [0.] * 10 for i in range(110)},
        getUserPosItems=lambda users: [[] for _ in users])
    world = SimpleNamespace(device="cpu", topks=[10, 20, 50, 100],
        config={"period": 8, "test_u_batch_size": 4, "algo": "vanilla"})
    calls, predicted = [], []
    class Net:
        def eval(self):
            return self
        def getUsersRating(self, users, *args):
            calls.append(users.tolist())
            return torch.arange(110, dtype=torch.float32).repeat(len(users), 1)
    def metrics(truth, ranks, ks):
        predicted.append(torch.stack(ranks))
        return [0.] * 4, [0.] * 4, [0.] * 4, [0.] * 4
    evaluate_public_test(ds, Net(), None, world, SimpleNamespace(computeTopNAccuracy=metrics))
    assert len(calls) == 1 and len(predicted) == 2
    # Move the masked item into the top100 so the difference is observable.
    ds.valid_dict = {0: [107], 2: [108]}
    calls.clear(); predicted.clear()
    evaluate_public_test(ds, Net(), None, world, SimpleNamespace(computeTopNAccuracy=metrics))
    scores = torch.arange(110, dtype=torch.float32).repeat(3, 1)
    scores[0, 107] = scores[1, 108] = -(1 << 10)
    assert torch.equal(predicted[0], scores.topk(100).indices)
    scores = torch.arange(110, dtype=torch.float32).repeat(3, 1)
    scores[0, 107] = scores[2, 108] = -(1 << 10)
    assert torch.equal(predicted[1], scores.topk(100).indices)
    # Execute the two actual pinned author functions with isolated module globals.
    import ast, math
    source = ast.parse((VENDOR / "Procedure.py").read_text())
    functions = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef)
        and n.name in ("Test", "computeTopNAccuracy")], type_ignores=[])
    namespace = {"torch": torch, "np": np, "math": math, "world": world,
                 "utils": SimpleNamespace(minibatch=lambda users, batch_size: (users[i:i + batch_size]
                    for i in range(0, len(users), batch_size)))}
    exec(compile(functions, "pinned_author_Procedure.py", "exec"), namespace)
    ds.n_user = 3
    ds.getUserValidItems = lambda users: validation_mask_rows(users, ds.valid_dict, True)
    ds.test_dict = {0: [109], 1: [108], 2: [106]}
    actual = evaluate_public_test(ds, Net(), None, world,
        SimpleNamespace(computeTopNAccuracy=namespace["computeTopNAccuracy"]))
    original = namespace["Test"](ds, Net(), None, 0, None, 0, 1)
    for field, value in zip(("precision", "recall", "ndcg", "mrr"), original):
        assert actual["author_original_mask"][field] == value
    assert actual["corrected_user_aligned_mask"]["recall"][0] > actual["author_original_mask"]["recall"][0]


def test_public_rng_roundtrip_cpu():
    # capture_rng includes CUDA only when visible; CPU suite checks RNG serialization.
    state = capture_rng()
    import random
    expected = (random.random(), np.random.random(), torch.rand(4))
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])


def test_public_checkpoint_optimizer_rng_resume_matches_continuation(tmp_path):
    from public_run import checkpoint
    import random
    torch.manual_seed(7); np.random.seed(7); random.seed(7)
    net = torch.nn.Linear(3, 1)
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    def step(model, optimizer):
        x = torch.tensor(np.random.normal(size=(5, 3)), dtype=torch.float32)[torch.randperm(5)]
        loss = (model(x) - random.random()).square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    step(net, opt)
    checkpoint(tmp_path / "last.pt", {"model": net.state_dict(), "optimizer": opt.state_dict(), "rng": capture_rng()})
    step(net, opt)
    resumed = torch.nn.Linear(3, 1)
    resumed_opt = torch.optim.Adam(resumed.parameters(), lr=.001)
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    resumed.load_state_dict(saved["model"]); resumed_opt.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    step(resumed, resumed_opt)
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), resumed.parameters()))
