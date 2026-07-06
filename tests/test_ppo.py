"""PPO correctness tests: rollout/update log-prob exactness (the ratio
foundation), effective-mask reconstruction, and a one-update integration run."""
from __future__ import annotations

import numpy as np
import torch

from triad.engine.orders import VOCAB_SIZE, WAIVE_ID
from triad.env.obs import OBS_DIM
from triad.rl.models import MAX_STEPS, TriadPolicy
from triad.rl.ppo import PPOConfig, train_ppo


def test_effective_masks_reconstruct_exclusion():
    m = TriadPolicy()
    torch.manual_seed(0)
    B, T = 3, 4
    masks = torch.zeros(B, T, VOCAB_SIZE, dtype=torch.bool)
    masks[:, :, [5, 6, 7, WAIVE_ID]] = True  # winter-like: 3 builds + WAIVE
    ids = torch.tensor(
        [[5, WAIVE_ID, WAIVE_ID, 0],   # build then waive twice
         [6, 5, 7, 0],                  # three distinct builds
         [WAIVE_ID, WAIVE_ID, 5, 0]]
    )
    n = torch.tensor([3, 3, 3])
    eff = m.effective_masks(masks, ids, n, repeat_ok=WAIVE_ID)
    # row 0: id 5 excluded after step 0; WAIVE never excluded
    assert not eff[0, 1, 5] and not eff[0, 2, 5]
    assert eff[0, 1, WAIVE_ID] and eff[0, 2, WAIVE_ID]
    # row 1: cumulative exclusion of each emitted build
    assert not eff[1, 1, 6] and not eff[1, 2, 6] and not eff[1, 2, 5]
    assert eff[1, 2, 7]  # not yet emitted at step 2's start
    # row 2: waives exclude nothing
    assert eff[2, 2, 5] and eff[2, 2, 6] and eff[2, 2, 7]


def test_act_vs_evaluate_logprob_exact_with_exclusion():
    """The PPO ratio is only valid if evaluate_actions reproduces act()'s
    sampling distribution exactly — including Winter's dynamic exclusion."""
    m = TriadPolicy()
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    B, T = 16, 5
    masks = torch.zeros(B, T, VOCAB_SIZE, dtype=torch.bool)
    for b in range(B):
        legal = rng.choice(VOCAB_SIZE - 1, size=6, replace=False)
        masks[b, :, legal] = True
        masks[b, :, WAIVE_ID] = True
    n = torch.from_numpy(rng.integers(0, T + 1, size=B))
    obs = torch.rand(B, OBS_DIM)
    ids, logp_act, _ = m.act(
        obs, masks, n, generator=torch.Generator().manual_seed(3),
        exclude_emitted=True, repeat_ok=WAIVE_ID,
    )
    _, logp_eval, _, _ = m.evaluate_actions(
        obs, ids, n, masks=masks, exclude_emitted=True, repeat_ok=WAIVE_ID
    )
    assert torch.allclose(logp_act, logp_eval, atol=1e-4)


def test_one_update_integration(tmp_path):
    """Two envs, one tiny update end-to-end: finite losses, checkpoint written."""
    cfg = PPOConfig(
        n_envs=2, rollout_len=8, total_steps=16, update_epochs=1,
        n_minibatches=1, eval_every=0, seed=0,
    )
    model = train_ppo(
        cfg, anchor_path="weights/bc.pt", init_from_anchor=True,
        device="cpu", output_dir=tmp_path, log=False, checkpoint_every=1,
    )
    for p in model.parameters():
        assert torch.isfinite(p).all()
    assert (tmp_path / "ppo_latest.pt").exists()
    assert (tmp_path / "ppo_latest_trainstate.pt").exists()


def test_population_pool_sampling_and_snapshots(tmp_path):
    import numpy as np
    from triad.rl.population import LATEST, PolicyPool

    pool = PolicyPool(tmp_path)
    rng = np.random.default_rng(0)
    # empty pool -> always latest
    assert (pool.sample_seat_sources(3, rng) == LATEST).all()
    m = TriadPolicy()
    pool.add(m, 5, seed=0)
    pool.add(m, 10, seed=0)
    assert len(pool) == 2 and (tmp_path / "pool" / "snap_0005.pt").exists()
    # with p_latest=0: every seat draws from the pool
    src = pool.sample_seat_sources(3, rng, p_latest=0.0)
    assert set(src.tolist()) <= {0, 1}
    # cached load returns a working policy
    assert pool.get(0).config == m.config


def test_population_training_integration(tmp_path):
    """Population run end-to-end: snapshots appear, training stays finite,
    and theta-masking leaves a valid (non-empty) training batch."""
    cfg = PPOConfig(
        n_envs=2, rollout_len=8, total_steps=48, update_epochs=1,
        n_minibatches=1, eval_every=0, seed=0,
        population=True, snapshot_every=1, p_latest=0.5,
    )
    model = train_ppo(
        cfg, anchor_path="weights/bc.pt", init_from_anchor=True,
        device="cpu", output_dir=tmp_path, log=False, checkpoint_every=10,
    )
    for p in model.parameters():
        assert torch.isfinite(p).all()
    assert (tmp_path / "pool").is_dir()
    assert len(list((tmp_path / "pool").glob("snap_*.pt"))) >= 2
