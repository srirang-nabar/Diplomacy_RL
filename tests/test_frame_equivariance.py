"""Regression tests for the M4 action-frame bug: the policy must operate
entirely in the acting power's own frame — observations AND action ids.

History: obs were rotated but order ids stayed real-frame, so identical
observations carried seat-dependent legality masks and exact weight sharing
was silently broken. Caught by the seat chi^2 (beta=0 run: p ~ 1e-8) and
pinned by a trajectory-equivariance experiment (rotate the per-seat seed
assignment -> every intermediate board must rotate). These tests encode both
findings permanently, using an UNTRAINED policy so they are training-free.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from triad.map_data import POWERS, ROTATION
from triad.engine.game import Game
from triad.engine.orders import (
    ORDERS,
    VOCAB_PERM,
    VOCAB_PERM_INV,
    VOCAB_SIZE,
    WAIVE_ID,
    rotate_order,
)
from triad.engine.state import Board, FALL, SPRING
from triad.env.triad_env import TriadEnv, NOOP_ID
from triad.rl.models import TriadPolicy
from triad.rl.policy_bot import PolicyBot

NEXT = {"A": "B", "B": "C", "C": "A"}


def test_vocab_perm_is_rotation_of_order_three():
    assert (VOCAB_PERM[0] == np.arange(VOCAB_SIZE)).all()  # k=0 identity
    for k in range(3):
        assert sorted(VOCAB_PERM[k].tolist()) == list(range(VOCAB_SIZE))  # permutation
        assert (VOCAB_PERM_INV[k][VOCAB_PERM[k]] == np.arange(VOCAB_SIZE)).all()
        assert VOCAB_PERM[k][WAIVE_ID] == WAIVE_ID  # fixed point
    # rho^1 applied to an own-frame order equals the table's k=1 mapping
    for own_id in range(0, VOCAB_SIZE, 17):
        assert ORDERS[VOCAB_PERM[1][own_id]] == rotate_order(ORDERS[own_id])


def test_env_masks_identical_across_seats_at_symmetric_start():
    """The whole point of own-frame actions: at the rotation-invariant start
    every seat must see bitwise-identical action masks."""
    env = TriadEnv()
    _, infos = env.reset()
    m = [infos[pw]["action_mask"] for pw in POWERS]
    assert (m[0] == m[1]).all() and (m[1] == m[2]).all()


def test_policybot_orders_rotate_with_the_seat():
    """Same torch seed, symmetric start: seat B's real orders must be the
    exact rotation of seat A's."""
    torch.manual_seed(0)
    model = TriadPolicy()  # untrained: high-entropy, exercises real sampling
    bot = PolicyBot(model, greedy=False)
    b = Board.initial()
    orders = {pw: bot.movement_orders(b, pw, np.random.default_rng(42)) for pw in POWERS}
    for pw in POWERS:
        rot_of_pw = {ROTATION[p]: rotate_order(o) for p, o in orders[pw].items()}
        assert rot_of_pw == orders[NEXT[pw]], f"seat {pw} orders do not rotate"


def _play_traced(model, seeds: dict[str, int]):
    bot = PolicyBot(model, greedy=False)
    rngs = {pw: np.random.default_rng(seeds[pw]) for pw in POWERS}
    g = Game()
    trace = []
    while not g.over and g.movement_phases < 12:  # bounded for test speed
        if g.board.phase in (SPRING, FALL):
            g.step_movement(
                {pw: bot.movement_orders(g.board, pw, rngs[pw]) for pw in g.alive_powers()}
            )
        else:
            g.step_winter(
                {pw: bot.winter_orders(g.board, pw, rngs[pw]) for pw in g.alive_powers()}
            )
        trace.append((dict(g.board.units), dict(g.board.sc_owner)))
    return trace


def test_full_pipeline_trajectory_equivariance():
    """Rotate the per-seat seed assignment -> every intermediate board of the
    full PolicyBot+engine pipeline must be the exact rotation of the original.
    This is the experiment that pinned the bug, promoted to a test."""
    torch.manual_seed(1)
    model = TriadPolicy()
    for trial in range(3):
        base = np.random.default_rng(500 + trial).integers(0, 2**62, 3)
        s1 = {"A": int(base[0]), "B": int(base[1]), "C": int(base[2])}
        s2 = {"B": int(base[0]), "C": int(base[1]), "A": int(base[2])}
        t1 = _play_traced(model, s1)
        t2 = _play_traced(model, s2)
        assert len(t1) == len(t2)
        for i in range(len(t1)):
            ru = {ROTATION[p]: NEXT[pw] for p, pw in t1[i][0].items()}
            rs = {ROTATION[p]: (None if pw is None else NEXT[pw]) for p, pw in t1[i][1].items()}
            assert t2[i][0] == ru, f"trial {trial}: units diverged at step {i}"
            assert t2[i][1] == rs, f"trial {trial}: sc owners diverged at step {i}"
