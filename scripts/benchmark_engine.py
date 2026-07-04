#!/usr/bin/env python
"""M1 benchmark: complete random-legal self-play games per second.

Target (CLAUDE.md §7): >= 300 complete games/sec, single process.

Usage:
    uv run python scripts/benchmark_engine.py [--games N] [--seed S] [--smoke]
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np

from triad.map_data import POWERS
from triad.engine.game import Game
from triad.engine.orders import ORDERS, legal_movement_orders
from triad.engine.state import SPRING, FALL


def play_random_game(py_rng: random.Random) -> Game:
    """One complete game of uniform-random legal play (all three seats).

    Uses python's random.Random internally for speed; the seed comes from an
    explicit numpy Generator at the call site so runs stay reproducible.
    """
    g = Game()
    units = g.board.units
    while not g.over:
        if g.board.phase in (SPRING, FALL):
            merged = {
                pw: {
                    p: ORDERS[py_rng.choice(legal_movement_orders(units, p))]
                    for p in g.board.unit_provinces(pw)
                }
                for pw in g.alive_powers()
            }
            g.step_movement(merged)
            units = g.board.units
        else:
            ob = {}
            for pw in g.alive_powers():
                delta = g.winter_delta(pw)
                if delta > 0:
                    ids = g.legal_build_ids(pw)
                    k = py_rng.randint(0, min(delta, len(ids)))
                    ob[pw] = [ORDERS[i] for i in py_rng.sample(ids, k)] if k else []
                elif delta < 0:
                    ids = g.legal_disband_ids(pw)
                    ob[pw] = [ORDERS[i] for i in py_rng.sample(ids, -delta)]
            g.step_winter(ob)
            units = g.board.units
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run (<60s guarantee), exit 0 on success")
    args = ap.parse_args()

    n_games = 5 if args.smoke else args.games
    rng = np.random.default_rng(args.seed)  # explicit seed source (CLAUDE.md §7)
    py_rng = random.Random(int(rng.integers(0, 2**63)))

    phases = 0
    outcomes = {"solo": 0, "draw": 0}
    t0 = time.perf_counter()
    for _ in range(n_games):
        g = play_random_game(py_rng)
        phases += g.movement_phases
        outcomes[g.result["type"]] += 1
    dt = time.perf_counter() - t0

    gps = n_games / dt
    print(f"{n_games} games in {dt:.2f}s  ->  {gps:.0f} games/sec, "
          f"{phases / dt:.0f} movement-phases/sec")
    print(f"outcomes: {outcomes}")
    if not args.smoke and gps < 300:
        print("WARNING: below the 300 games/sec M1 target")


if __name__ == "__main__":
    main()
