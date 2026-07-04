"""Turtle: pure defense. Every unit support-holds a random adjacent own unit
when one exists, otherwise holds. Never moves, never expands."""
from __future__ import annotations

import numpy as np

from triad.map_data import ADJACENCY
from triad.bots.base import Bot
from triad.engine.orders import HOLD, SUP_HOLD, Order
from triad.engine.state import Board


class Turtle(Bot):
    def movement_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> dict[str, Order]:
        units = board.units
        out: dict[str, Order] = {}
        for p in board.unit_provinces(power):
            own_neighbors = [n for n in ADJACENCY[p] if units.get(n) == power]
            if own_neighbors:
                t = own_neighbors[int(rng.integers(len(own_neighbors)))]
                out[p] = Order(SUP_HOLD, p, t)
            else:
                out[p] = Order(HOLD, p)
        return out
