"""Grabber: each unit heads for the nearest capturable SC by BFS distance,
with naive support coordination — when several units aim at the same target,
one moves in and the rest (all adjacent by construction) support it.

All ties (equidistant targets, equally good next steps, who gets to be the
mover) break uniformly at random via the rng.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from triad.map_data import ADJACENCY, PROVINCES, SUPPLY_CENTERS
from triad.bots.base import Bot
from triad.engine.orders import HOLD, MOVE, SUP_MOVE, Order
from triad.engine.state import Board

_SC_SET = frozenset(SUPPLY_CENTERS)


def _all_pairs_dist() -> dict[str, dict[str, int]]:
    dist: dict[str, dict[str, int]] = {}
    for src in PROVINCES:
        d = {src: 0}
        q = deque([src])
        while q:
            cur = q.popleft()
            for n in ADJACENCY[cur]:
                if n not in d:
                    d[n] = d[cur] + 1
                    q.append(n)
        dist[src] = d
    return dist


DIST = _all_pairs_dist()


class Grabber(Bot):
    def movement_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> dict[str, Order]:
        units = board.units
        targets = [sc for sc in SUPPLY_CENTERS if board.sc_owner[sc] != power]
        mine = board.unit_provinces(power)
        if not targets:
            return {p: Order(HOLD, p) for p in mine}

        # intent pass: hold on a capturable SC, else step toward the nearest one
        intents: dict[str, str | None] = {}  # unit -> destination (None = hold)
        for p in mine:
            if p in _SC_SET and board.sc_owner[p] != power:
                intents[p] = None  # stand to capture
                continue
            best = min(DIST[p][t] for t in targets)
            steps = [
                n
                for n in ADJACENCY[p]
                if min(DIST[n][t] for t in targets) == best - 1
            ]
            intents[p] = steps[int(rng.integers(len(steps)))]

        # naive support: several own units moving to the same destination ->
        # one mover (random), the rest support it (all are adjacent to dest)
        by_dest: dict[str, list[str]] = {}
        for p, d in intents.items():
            if d is not None:
                by_dest.setdefault(d, []).append(p)

        out: dict[str, Order] = {}
        for p, d in intents.items():
            if d is None:
                out[p] = Order(HOLD, p)
        for d, group in by_dest.items():
            if len(group) == 1:
                p = group[0]
                out[p] = Order(MOVE, p, None, d)
            else:
                mover = group[int(rng.integers(len(group)))]
                out[mover] = Order(MOVE, mover, None, d)
                for p in group:
                    if p != mover:
                        out[p] = Order(SUP_MOVE, p, mover, d)
        return out
