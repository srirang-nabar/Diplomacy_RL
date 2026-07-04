"""RandomLegal: uniform over dynamically legal orders per unit."""
from __future__ import annotations

import numpy as np

from triad.map_data import HOME_CENTERS
from triad.bots.base import Bot
from triad.engine.orders import BUILD, DISBAND, ORDERS, Order, legal_movement_orders
from triad.engine.state import Board


class RandomLegal(Bot):
    def movement_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> dict[str, "Order"]:
        out = {}
        for p in board.unit_provinces(power):
            ids = legal_movement_orders(board.units, p)
            out[p] = ORDERS[ids[int(rng.integers(len(ids)))]]
        return out

    def winter_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> list["Order"]:
        delta = board.sc_count(power) - board.unit_count(power)
        if delta > 0:
            legal = [
                h
                for h in HOME_CENTERS[power]
                if board.sc_owner[h] == power and h not in board.units
            ]
            k = int(rng.integers(min(delta, len(legal)) + 1))  # may waive
            picks = [legal[i] for i in rng.permutation(len(legal))[:k]]
            return [Order(BUILD, h) for h in picks]
        if delta < 0:
            mine = board.unit_provinces(power)
            picks = [mine[i] for i in rng.permutation(len(mine))[: -delta]]
            return [Order(DISBAND, p) for p in picks]
        return []
