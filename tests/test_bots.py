"""Bot smoke tests: termination, legality, determinism, tie-break divergence."""
from __future__ import annotations

import numpy as np
import pytest

from triad.map_data import MAX_MOVEMENT_PHASES, POWERS, SUPPLY_CENTERS
from triad.bots import Grabber, RandomLegal, Turtle, play_game
from triad.engine.game import Game
from triad.engine.orders import ORDER_INDEX, legal_movement_orders
from triad.engine.state import FALL, SPRING

MATCHUPS = [
    ("RRR", [RandomLegal, RandomLegal, RandomLegal]),
    ("GGG", [Grabber, Grabber, Grabber]),
    ("TTT", [Turtle, Turtle, Turtle]),
    ("GRT", [Grabber, RandomLegal, Turtle]),
]


@pytest.mark.parametrize("name,classes", MATCHUPS, ids=[m[0] for m in MATCHUPS])
def test_matchup_terminates(name, classes):
    bots = {pw: cls() for pw, cls in zip(POWERS, classes)}
    for seed in (0, 1):
        g = play_game(bots, np.random.default_rng(seed))
        assert g.over and g.result is not None
        assert g.movement_phases <= MAX_MOVEMENT_PHASES
        g.board.validate()


@pytest.mark.parametrize("cls", [Grabber, Turtle], ids=["grabber", "turtle"])
def test_bot_orders_always_legal(cls):
    """Deterministic bots must emit only dynamically legal orders — no
    coerced-HOLD safety net allowed."""
    rng = np.random.default_rng(3)
    bots = {pw: cls() for pw in POWERS}
    g = Game()
    while not g.over:
        if g.board.phase in (SPRING, FALL):
            merged = {}
            for pw in g.alive_powers():
                om = bots[pw].movement_orders(g.board, pw, rng)
                for p, o in om.items():
                    assert g.board.units.get(p) == pw, f"{p} not {pw}'s unit"
                    legal = legal_movement_orders(g.board.units, p)
                    assert ORDER_INDEX[o] in legal, f"illegal {o} from {cls.__name__}"
                merged[pw] = om
            g.step_movement(merged)
        else:
            g.step_winter(
                {pw: bots[pw].winter_orders(g.board, pw, rng) for pw in g.alive_powers()}
            )


def test_same_seed_same_game():
    bots = {pw: Grabber() for pw in POWERS}
    g1 = play_game(bots, np.random.default_rng(11))
    g2 = play_game(bots, np.random.default_rng(11))
    assert g1.board.units == g2.board.units
    assert g1.result == g2.result and g1.movement_phases == g2.movement_phases


def test_different_seeds_diverge():
    """Tie-breaking must actually fire: Grabber-vs-Grabber games under
    different seeds cannot all be the identical replay (CLAUDE.md §4.4)."""
    bots = {pw: Grabber() for pw in POWERS}
    outcomes = set()
    for seed in range(6):
        g = play_game(bots, np.random.default_rng(seed))
        outcomes.add(
            (g.movement_phases, tuple(sorted(g.board.units.items())), str(g.result))
        )
    assert len(outcomes) > 1, "all Grabber games identical - tie-breaking dead"


def test_turtle_never_moves():
    rng = np.random.default_rng(5)
    g = Game()
    bots = {pw: Turtle() for pw in POWERS}
    for _ in range(6):
        if g.board.phase in (SPRING, FALL):
            for pw in POWERS:
                for o in bots[pw].movement_orders(g.board, pw, rng).values():
                    assert o.kind in ("H", "SH")
            g.step_movement(
                {pw: bots[pw].movement_orders(g.board, pw, rng) for pw in POWERS}
            )
        else:
            g.step_winter()
    # nobody moved: everyone still owns exactly their homes
    assert all(g.board.sc_count(pw) == 3 for pw in POWERS)


def test_grabber_captures_neutrals():
    """Grabber must actually grab: within two game-years of GGG play, the
    neutral border SCs should be owned."""
    bots = {pw: Grabber() for pw in POWERS}
    rng = np.random.default_rng(2)
    g = Game()
    steps = 0
    while not g.over and g.board.year <= 2:
        if g.board.phase in (SPRING, FALL):
            g.step_movement(
                {pw: bots[pw].movement_orders(g.board, pw, rng) for pw in g.alive_powers()}
            )
        else:
            g.step_winter(
                {pw: bots[pw].winter_orders(g.board, pw, rng) for pw in g.alive_powers()}
            )
        steps += 1
    owned = sum(1 for sc in SUPPLY_CENTERS if g.board.sc_owner[sc] is not None)
    assert owned >= 10, f"only {owned}/12 SCs owned after 2 grabber years"
