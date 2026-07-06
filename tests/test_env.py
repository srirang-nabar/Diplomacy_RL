"""Env tests: PettingZoo API compliance, mask correctness, reward accounting,
determinism, elimination handling, vectorization (tasks M2.3)."""
from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from triad.map_data import POWERS
from triad.engine.orders import ORDERS, ORDER_INDEX, Order, legal_movement_orders
from triad.engine.state import FALL, SPRING, WINTER, Board
from triad.engine.game import Game
from triad.env.obs import OBS_DIM
from triad.env.triad_env import MAX_UNITS, N_ACTIONS, NOOP_ID, TriadEnv
from triad.env.vec import VecTriadEnv


def _masked_random_actions(env: TriadEnv, infos: dict, rng: np.random.Generator):
    acts = {}
    for pw in env.agents:
        mask = infos[pw]["action_mask"]
        a = np.empty(MAX_UNITS, dtype=np.int64)
        for i in range(MAX_UNITS):
            ids = np.flatnonzero(mask[i])
            a[i] = ids[rng.integers(len(ids))] if len(ids) else NOOP_ID
        acts[pw] = a
    return acts


def _play_masked_episode(env: TriadEnv, seed: int):
    rng = np.random.default_rng(seed)
    obs, infos = env.reset()
    traj = []
    while env.agents:
        acts = _masked_random_actions(env, infos, rng)
        obs, rew, term, trunc, infos = env.step(acts)
        traj.append((dict(rew), dict(term)))
    return env.game, traj


def test_pettingzoo_parallel_api_compliance():
    parallel_api_test(TriadEnv(), num_cycles=200)


def test_spaces():
    env = TriadEnv()
    for pw in POWERS:
        assert env.observation_space(pw).shape == (OBS_DIM,)
        assert env.action_space(pw).shape == (MAX_UNITS,)
        assert env.action_space(pw).nvec.tolist() == [N_ACTIONS] * MAX_UNITS


def test_movement_mask_matches_engine_legality():
    """Masks are OWN-FRAME (post the M4 action-frame fix): row i holds the
    own-frame translation of the i-th own-frame-ordered unit's legal ids."""
    from triad.engine.orders import to_own_frame_ids
    from triad.env.obs import POWER_INDEX, own_frame_unit_order

    env = TriadEnv()
    _, infos = env.reset()
    for pw in POWERS:
        k = POWER_INDEX[pw]
        mask = infos[pw]["action_mask"]
        provs = own_frame_unit_order(env.game.board.units, pw)
        for i, p in enumerate(provs):
            legal = set(
                to_own_frame_ids(
                    legal_movement_orders(env.game.board.units, p), k
                ).tolist()
            )
            assert set(np.flatnonzero(mask[i]).tolist()) == legal
        for i in range(len(provs), MAX_UNITS):
            assert set(np.flatnonzero(mask[i])) == {NOOP_ID}


def test_winter_masks_builds_disbands_and_noop():
    env = TriadEnv()
    obs, infos = env.reset()
    # A grabs B_AB: spring move + fall hold -> delta +1 in winter
    hold_all = lambda pw: np.full(MAX_UNITS, NOOP_ID, dtype=np.int64)  # noqa: E731
    acts = {pw: hold_all(pw) for pw in POWERS}
    acts["A"][2] = ORDER_INDEX[Order("M", "R_A", None, "B_AB")]  # unit order: CAP_A,L_A,R_A
    obs, *_ , infos = env.step(acts)
    obs, *_ , infos = env.step({pw: hold_all(pw) for pw in POWERS})  # fall
    assert env.game.board.phase == WINTER
    mask_a = infos["A"]["action_mask"]
    row0 = set(np.flatnonzero(mask_a[0]))
    build_r_a = ORDER_INDEX[Order("B", "R_A")]
    from triad.engine.orders import WAIVE_ID
    assert build_r_a in row0 and WAIVE_ID in row0
    # delta=0 powers: NOOP-only rows
    mask_b = infos["B"]["action_mask"]
    assert all(set(np.flatnonzero(mask_b[i])) == {NOOP_ID} for i in range(MAX_UNITS))


def test_full_masked_episode_terminates_with_valid_rewards():
    env = TriadEnv()
    game, traj = _play_masked_episode(env, seed=0)
    assert game.over
    final_rewards, final_terms = traj[-1]
    assert all(final_terms.values())
    total = sum(final_rewards.values())
    # terminal rewards are the scores (+ tiny shaping term)
    assert -1.0 <= total <= 2.0


def test_determinism_same_seed_same_trajectory():
    g1, t1 = _play_masked_episode(TriadEnv(), seed=7)
    g2, t2 = _play_masked_episode(TriadEnv(), seed=7)
    assert g1.board.units == g2.board.units and t1 == t2


def test_shaping_reward_at_winter():
    env = TriadEnv(shaping_alpha=0.02)
    env.reset()
    noop = np.full(MAX_UNITS, NOOP_ID, dtype=np.int64)
    acts = {pw: noop.copy() for pw in POWERS}
    acts["A"][2] = ORDER_INDEX[Order("M", "R_A", None, "B_AB")]
    env.step(acts)                                    # spring: A into B_AB
    obs, rew, *_ = env.step({pw: noop.copy() for pw in POWERS})  # fall: capture
    assert env.game.board.phase == WINTER
    obs, rew, term, trunc, infos = env.step({pw: noop.copy() for pw in POWERS})
    assert rew["A"] == pytest.approx(0.02)            # +1 SC * alpha
    assert rew["B"] == pytest.approx(0.0) and rew["C"] == pytest.approx(0.0)


def test_no_shaping_when_alpha_zero():
    env = TriadEnv(shaping_alpha=0.0)
    env.reset()
    noop = np.full(MAX_UNITS, NOOP_ID, dtype=np.int64)
    acts = {pw: noop.copy() for pw in POWERS}
    acts["A"][2] = ORDER_INDEX[Order("M", "R_A", None, "B_AB")]
    env.step(acts)
    env.step({pw: noop.copy() for pw in POWERS})
    obs, rew, *_ = env.step({pw: noop.copy() for pw in POWERS})
    assert rew["A"] == 0.0


def test_eliminated_agent_exits_with_zero_reward():
    env = TriadEnv()
    env.reset()
    # engineer: B eliminated after A occupies all B homes at fall; B's only
    # unit sits on CTR (non-SC), so the capture pass leaves B with 0 SCs
    g = env.game
    g.board.units = {"CAP_B": "A", "L_B": "A", "R_B": "A", "CTR": "B", "CAP_C": "C"}
    g.board.phase = FALL
    env._last_sc = {pw: g.board.sc_count(pw) for pw in POWERS}
    noop = np.full(MAX_UNITS, NOOP_ID, dtype=np.int64)
    env.step({pw: noop.copy() for pw in POWERS})      # fall -> captures
    obs, rew, term, trunc, infos = env.step({pw: noop.copy() for pw in POWERS})  # winter
    assert not env.game.over            # A has 6 SCs < 7: game continues
    assert "B" in env.game.eliminated
    assert term["B"] is True and rew["B"] == 0.0
    assert "B" not in env.agents and "A" in env.agents and "C" in env.agents


def test_solo_terminal_rewards():
    env = TriadEnv(shaping_alpha=0.0)
    env.reset()
    g = env.game
    g.board.units = {"CAP_B": "A", "L_B": "A", "R_B": "A", "B_AB": "A"}
    g.board.phase = FALL
    env._last_sc = {pw: g.board.sc_count(pw) for pw in POWERS}
    noop = np.full(MAX_UNITS, NOOP_ID, dtype=np.int64)
    env.step({pw: noop.copy() for pw in POWERS})      # fall: A -> 7 SCs
    obs, rew, term, trunc, infos = env.step({pw: noop.copy() for pw in POWERS})
    assert env.game.over and all(term.values()) and env.agents == []
    assert rew == {"A": 1.0, "B": 0.0, "C": 0.0}


def test_vec_env_shapes_autoreset_determinism():
    n = 4
    v = VecTriadEnv(n_envs=n, shaping_alpha=0.0)
    obs, mask = v.reset()
    assert obs.shape == (n, 3, OBS_DIM) and mask.shape == (n, 3, MAX_UNITS, N_ACTIONS)
    rng = np.random.default_rng(0)
    dones_seen = 0
    for _ in range(200):
        acts = np.full((n, 3, MAX_UNITS), NOOP_ID, dtype=np.int64)
        for i in range(n):
            for s in range(3):
                for u in range(MAX_UNITS):
                    ids = np.flatnonzero(mask[i, s, u])
                    acts[i, s, u] = ids[rng.integers(len(ids))]
        obs, mask, rew, done, acted, seat_done, infos = v.step(acts)
        assert rew.shape == (n, 3) and done.shape == (n,)
        assert acted.shape == (n, 3) and seat_done.shape == (n, 3)
        for i in range(n):
            if done[i]:
                dones_seen += 1
                assert "result" in infos[i]
                # all seats still alive at game end got their terminal flag
                assert (seat_done[i] | ~acted[i]).all()
                # auto-reset: fresh masks are valid (every live row nonempty)
                assert mask[i].any(axis=-1).all()
    assert dones_seen >= 1, "no episode finished in 200 vec steps"
