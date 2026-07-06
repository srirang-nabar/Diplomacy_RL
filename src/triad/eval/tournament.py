"""Round-robin tournament (CLAUDE.md §5).

Lineup universe = all 3-multisets of the contender roster. Games use SAMPLED
actions with per-game seat rotation and a seeded rng stream — the engine is
deterministic, so policy/bot stochasticity is what makes games i.i.d. and the
Wilson CIs meaningful.

Output: per-game records (lineup, seat assignment, result type, winner,
final SCs) — everything the ratings and the report tables need.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from triad.map_data import POWERS
from triad.bots import Bot, Grabber, RandomLegal, Turtle, play_game
from triad.rl.checkpoint import load_policy
from triad.rl.policy_bot import PolicyBot


def build_roster(weights: dict[str, str] | None = None) -> dict[str, Bot]:
    """The fixed contender set {RandomLegal, Grabber, Turtle, BC, final}.

    weights: name -> checkpoint path for the NN contenders.
    """
    weights = weights or {"bc": "weights/bc.pt", "final": "weights/ppo_final.pt"}
    roster: dict[str, Bot] = {
        "random": RandomLegal(),
        "grabber": Grabber(),
        "turtle": Turtle(),
    }
    for name, path in weights.items():
        model, _ = load_policy(path, device="cpu")
        roster[name] = PolicyBot(model, greedy=False)  # sampled play (§5)
    return roster


@dataclass
class GameRecord:
    lineup: tuple[str, str, str]      # policy name per seat A, B, C
    result: str                       # "solo" | "draw"
    winner_seat: str | None           # power name or None
    winner_policy: str | None
    final_scs: dict[str, int]         # per seat
    phases: int


def all_lineups(names: list[str]) -> list[tuple[str, ...]]:
    """All 3-multisets of the roster (order-independent lineups)."""
    return list(itertools.combinations_with_replacement(sorted(names), 3))


def play_lineup(
    roster: dict[str, Bot],
    lineup: tuple[str, ...],
    n_games: int,
    rng: np.random.Generator,
) -> list[GameRecord]:
    """n_games of one lineup, rotating the seat assignment each game so any
    residual seat effect cancels across the block."""
    records = []
    for g_i in range(n_games):
        rot = g_i % 3
        assignment = tuple(lineup[(s + rot) % 3] for s in range(3))
        bots = {pw: roster[assignment[s]] for s, pw in enumerate(POWERS)}
        g = play_game(bots, rng)
        res = g.result
        winner = res.get("winner")
        records.append(
            GameRecord(
                lineup=assignment,
                result=res["type"],
                winner_seat=winner,
                winner_policy=(
                    assignment[POWERS.index(winner)] if winner else None
                ),
                final_scs={pw: g.board.sc_count(pw) for pw in POWERS},
                phases=g.movement_phases,
            )
        )
    return records


def run_round_robin(
    roster: dict[str, Bot],
    games_per_lineup: int,
    seed: int,
    boosts: dict[tuple[str, ...], int] | None = None,
    log_every: int = 5,
) -> list[GameRecord]:
    """The full tournament. boosts: lineup -> total games (>= games_per_lineup)
    for matchups whose numbers are cited in the report (CLAUDE.md §5: >=2000)."""
    rng = np.random.default_rng(seed)
    boosts = boosts or {}
    records: list[GameRecord] = []
    lineups = all_lineups(list(roster))
    t0 = time.perf_counter()
    for i, lu in enumerate(lineups):
        n = max(games_per_lineup, boosts.get(lu, 0))
        records.extend(play_lineup(roster, lu, n, rng))
        if log_every and (i + 1) % log_every == 0:
            dt = time.perf_counter() - t0
            print(f"[tournament] {i + 1}/{len(lineups)} lineups, "
                  f"{len(records)} games, {dt / 60:.1f} min", flush=True)
    return records


# --- aggregation ----------------------------------------------------------------
def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - hw, c + hw


def policy_table(records: list[GameRecord]) -> dict[str, dict]:
    """Per policy: solo/draw/eliminated rates over every seat it occupied."""
    stats: dict[str, dict] = {}
    for r in records:
        for s, pw in enumerate(POWERS):
            name = r.lineup[s]
            st = stats.setdefault(
                name, {"games": 0, "solo": 0, "draw": 0, "eliminated": 0}
            )
            st["games"] += 1
            if r.result == "draw":
                st["draw"] += 1
            elif r.winner_seat == pw:
                st["solo"] += 1
            if r.final_scs[pw] == 0:
                st["eliminated"] += 1
    for name, st in stats.items():
        n = st["games"]
        p = st["solo"] / n
        lo, hi = wilson(p, n)
        st.update(solo_rate=round(p, 4), ci=[round(lo, 4), round(hi, 4)],
                  draw_rate=round(st["draw"] / n, 4),
                  eliminated_rate=round(st["eliminated"] / n, 4))
    return stats


def matchup_table(
    records: list[GameRecord], focus: str
) -> dict[str, dict]:
    """focus policy vs each unordered opponent pair it faced."""
    out: dict[str, dict] = {}
    for r in records:
        for s, pw in enumerate(POWERS):
            if r.lineup[s] != focus:
                continue
            opps = tuple(sorted(r.lineup[t] for t in range(3) if t != s))
            key = "+".join(opps)
            st = out.setdefault(key, {"games": 0, "solo": 0, "draw": 0})
            st["games"] += 1
            if r.result == "draw":
                st["draw"] += 1
            elif r.winner_seat == pw:
                st["solo"] += 1
    for key, st in out.items():
        n = st["games"]
        p = st["solo"] / n
        lo, hi = wilson(p, n)
        st.update(solo_rate=round(p, 4), ci=[round(lo, 4), round(hi, 4)])
    return out
