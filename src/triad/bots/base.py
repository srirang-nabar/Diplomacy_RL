"""Bot base class and game runner."""
from __future__ import annotations

import numpy as np

from triad.map_data import HOME_CENTERS
from triad.engine.game import Game
from triad.engine.orders import Order, BUILD, DISBAND
from triad.engine.state import Board, FALL, SPRING


class Bot:
    """A pure policy: (board, power, rng) -> orders. Stateless by contract."""

    def movement_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> dict[str, Order]:
        raise NotImplementedError

    def winter_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> list[Order]:
        """Default Winter policy: build in every vacant owned home SC (random
        order when delta < available), disband uniformly at random."""
        delta = board.sc_count(power) - board.unit_count(power)
        if delta > 0:
            legal = [
                h
                for h in HOME_CENTERS[power]
                if board.sc_owner[h] == power and h not in board.units
            ]
            picks = [legal[i] for i in rng.permutation(len(legal))[: delta]]
            return [Order(BUILD, h) for h in picks]
        if delta < 0:
            mine = board.unit_provinces(power)
            picks = [mine[i] for i in rng.permutation(len(mine))[: -delta]]
            return [Order(DISBAND, p) for p in picks]
        return []


def play_game(
    bots: dict[str, Bot], rng: np.random.Generator, game: Game | None = None
) -> Game:
    """Run a full game with one bot per power. Deterministic given rng state."""
    g = game if game is not None else Game()
    while not g.over:
        if g.board.phase in (SPRING, FALL):
            g.step_movement(
                {
                    pw: bots[pw].movement_orders(g.board, pw, rng)
                    for pw in g.alive_powers()
                }
            )
        else:
            g.step_winter(
                {
                    pw: bots[pw].winter_orders(g.board, pw, rng)
                    for pw in g.alive_powers()
                }
            )
    return g
