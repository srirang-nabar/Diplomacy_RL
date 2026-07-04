"""Model unit tests (tasks M2.4): shapes, mask enforcement, log-prob
consistency, Winter slot counts, value head — plus the checkpoint round-trip."""
from __future__ import annotations

import numpy as np
import torch

from triad.engine.orders import VOCAB_SIZE
from triad.env.obs import OBS_DIM
from triad.rl.checkpoint import load_policy, save_checkpoint
from triad.rl.models import MAX_STEPS, TriadPolicy


def _model(seed: int = 0) -> TriadPolicy:
    torch.manual_seed(seed)
    return TriadPolicy()


def _random_masks(rng, B, T, k=20):
    m = torch.zeros(B, T, VOCAB_SIZE, dtype=torch.bool)
    for b in range(B):
        for t in range(T):
            ids = rng.choice(VOCAB_SIZE, size=k, replace=False)
            m[b, t, ids] = True
    return m


def test_forward_shapes():
    m = _model()
    B, T = 5, MAX_STEPS
    obs = torch.rand(B, OBS_DIM)
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    n = torch.tensor([3, 0, 12, 1, 7])
    logits, logprob, entropy, vlog = m.evaluate_actions(obs, ids, n)
    assert logits.shape == (B, T, VOCAB_SIZE)
    assert logprob.shape == (B,) and entropy.shape == (B,) and vlog.shape == (B, 3)
    assert logprob[1].item() == 0.0 and entropy[1].item() == 0.0  # n_steps=0


def test_mask_enforcement_never_samples_illegal():
    m = _model()
    rng = np.random.default_rng(0)
    B, T = 8, 4
    masks = _random_masks(rng, B, T, k=15)
    obs = torch.rand(B, OBS_DIM)
    n = torch.full((B,), T)
    gen = torch.Generator().manual_seed(0)
    for _ in range(125):  # 125 * 8 = 1000 draws
        ids, _, _ = m.act(obs, masks, n, generator=gen)
        for b in range(B):
            for t in range(T):
                assert masks[b, t, ids[b, t]], "sampled an illegal order id"


def test_joint_logprob_is_sum_of_step_logprobs():
    m = _model()
    rng = np.random.default_rng(1)
    B, T = 6, 5
    masks = _random_masks(rng, B, T)
    obs = torch.rand(B, OBS_DIM)
    n = torch.tensor([5, 3, 0, 1, 5, 2])
    gen = torch.Generator().manual_seed(1)
    ids, logprob_act, _ = m.act(obs, masks, n, generator=gen)
    # re-evaluate the same ids teacher-forced with the same masks
    _, logprob_eval, _, _ = m.evaluate_actions(obs, ids, n, masks=masks)
    assert torch.allclose(logprob_act, logprob_eval, atol=1e-4)


def test_greedy_deterministic():
    m = _model()
    rng = np.random.default_rng(2)
    masks = _random_masks(rng, 4, 6)
    obs = torch.rand(4, OBS_DIM)
    n = torch.full((4,), 6)
    a1, _, _ = m.act(obs, masks, n, greedy=True)
    a2, _, _ = m.act(obs, masks, n, greedy=True)
    assert torch.equal(a1, a2)


def test_winter_slot_counts():
    """delta=+2 -> 2 decode steps, -1 -> 1, 0 -> 0 (steps beyond n are pad)."""
    m = _model()
    rng = np.random.default_rng(3)
    masks = _random_masks(rng, 3, MAX_STEPS)
    obs = torch.rand(3, OBS_DIM)
    n = torch.tensor([2, 1, 0])
    ids, logprob, _ = m.act(obs, masks, n, generator=torch.Generator().manual_seed(0))
    assert (ids[0, 2:] == 0).all() and (ids[1, 1:] == 0).all() and (ids[2] == 0).all()
    assert logprob[2].item() == 0.0


def test_exclude_emitted_no_duplicates():
    """Winter decode must not emit the same BUILD/DISBAND twice."""
    m = _model()
    B, T = 4, 3
    masks = torch.zeros(B, T, VOCAB_SIZE, dtype=torch.bool)
    masks[:, :, [10, 11, 12]] = True  # 3 slots, 3 legal ids
    obs = torch.rand(B, OBS_DIM)
    n = torch.full((B,), T)
    gen = torch.Generator().manual_seed(4)
    for _ in range(50):
        ids, _, _ = m.act(obs, masks, n, generator=gen, exclude_emitted=True)
        for b in range(B):
            assert len(set(ids[b].tolist())) == T, "duplicate emitted in slots"


def test_value_head_softmax_sums_to_one():
    m = _model()
    obs = torch.rand(7, OBS_DIM)
    rng = np.random.default_rng(5)
    masks = _random_masks(rng, 7, 1)
    _, _, vals = m.act(obs, masks, torch.ones(7, dtype=torch.long))
    assert vals.shape == (7, 3)
    assert torch.allclose(vals.sum(-1), torch.ones(7), atol=1e-5)
    assert (vals >= 0).all()


def test_checkpoint_round_trip(tmp_path):
    """Save -> load with map_location cpu / weights_only -> identical outputs."""
    m = _model(seed=9)
    path = save_checkpoint(
        m, tmp_path / "ck.pt", train_config={"lr": 1e-3}, seed=9
    )
    m2, payload = load_policy(path, device="cpu")
    assert payload["order_vocab_size"] == VOCAB_SIZE
    assert payload["model_config"] == m.config
    assert payload["train_config"] == {"lr": 1e-3} and payload["seed"] == 9
    obs = torch.rand(3, OBS_DIM)
    ids = torch.randint(0, VOCAB_SIZE, (3, 4))
    n = torch.full((3,), 4)
    with torch.no_grad():
        l1, p1, _, v1 = m.evaluate_actions(obs, ids, n)
        l2, p2, _, v2 = m2.evaluate_actions(obs, ids, n)
    assert torch.equal(l1, l2) and torch.equal(v1, v2) and torch.equal(p1, p2)
