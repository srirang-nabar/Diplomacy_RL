"""TrueSkill ratings over 3-player tournament outcomes (CLAUDE.md §5).

Each game is three single-player teams. Ranks: solo winner 0, others 1;
draws rank by final SC count (more SCs = better), equal counts tie.
"""
from __future__ import annotations

import trueskill

from triad.map_data import POWERS
from triad.eval.tournament import GameRecord


def rate(records: list[GameRecord]) -> dict[str, dict]:
    env = trueskill.TrueSkill(draw_probability=0.05)
    ratings: dict[str, trueskill.Rating] = {}
    for r in records:
        names = list(r.lineup)
        if len(set(names)) == 1:
            # mirror games are uninformative for ratings (the same identity
            # wins and loses simultaneously) and only churn mu/sigma — skip
            continue
        groups = [(ratings.get(n, env.create_rating()),) for n in names]
        if r.result == "solo":
            ranks = [0 if pw == r.winner_seat else 1 for pw in POWERS]
        else:  # draw: rank by final SCs (higher = better rank)
            scs = [r.final_scs[pw] for pw in POWERS]
            order = sorted(set(scs), reverse=True)
            ranks = [order.index(s) for s in scs]
        new = env.rate(groups, ranks=ranks)
        for n, (nr,) in zip(names, new):
            ratings[n] = nr
    return {
        n: {"mu": round(rt.mu, 2), "sigma": round(rt.sigma, 2),
            "conservative": round(rt.mu - 3 * rt.sigma, 2)}
        for n, rt in sorted(ratings.items(), key=lambda kv: -(kv[1].mu - 3 * kv[1].sigma))
    }
