#!/usr/bin/env python
"""M5 round-robin tournament (CLAUDE.md §5).

Roster: {random, grabber, turtle, bc, final}. All 35 3-multiset lineups,
sampled actions, seat-rotated, seeded. Cited matchups get >=2000 games
(headline vs-grabber rows and the final-agent seat chi^2). Writes
results/tournament.json.

Usage:
    uv run python scripts/run_tournament.py                 # full (~3-4h CPU)
    uv run python scripts/run_tournament.py --games 150
    uv run python scripts/run_tournament.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

REPO = Path(__file__).resolve().parent.parent

CITED = [  # lineups whose numbers appear in the report -> >=2000 games (§5)
    ("final", "grabber", "grabber"),
    ("bc", "grabber", "grabber"),
    ("final", "final", "final"),   # doubles as the cited seat chi^2
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=150, help="games per lineup (floor)")
    ap.add_argument("--cited-games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/tournament.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from triad.map_data import POWERS
    from triad.eval.ratings import rate
    from triad.eval.tournament import (
        build_roster,
        matchup_table,
        policy_table,
        run_round_robin,
    )

    games, cited_games = (2, 4) if args.smoke else (args.games, args.cited_games)
    out_path = Path("/tmp/tournament_smoke.json") if args.smoke else REPO / args.out

    roster = build_roster()
    boosts = {tuple(sorted(lu)): cited_games for lu in CITED}
    t0 = time.perf_counter()
    records = run_round_robin(roster, games, args.seed, boosts=boosts,
                              log_every=0 if args.smoke else 5)
    wall_min = (time.perf_counter() - t0) / 60

    # cited seat chi^2 from the final-mirror lineup
    mirror = [r for r in records if r.lineup == ("final",) * 3]
    solos = np.array([sum(1 for r in mirror if r.winner_seat == pw) for pw in POWERS])
    chi2 = {
        "n_games": len(mirror),
        "solos": {pw: int(s) for pw, s in zip(POWERS, solos)},
        "draws": int(sum(1 for r in mirror if r.result == "draw")),
        "p": round(float(chisquare(solos).pvalue), 4) if solos.sum() else 1.0,
    }

    payload = {
        "config": {"games_per_lineup": games, "cited_games": cited_games,
                   "seed": args.seed, "n_games": len(records),
                   "wall_min": round(wall_min, 1)},
        "policy_table": policy_table(records),
        "matchups_final": matchup_table(records, "final"),
        "matchups_bc": matchup_table(records, "bc"),
        "trueskill": rate(records),
        "seat_chi2_final_mirror": chi2,
        "records": [
            {"lineup": r.lineup, "result": r.result, "winner_seat": r.winner_seat,
             "winner_policy": r.winner_policy, "final_scs": r.final_scs,
             "phases": r.phases}
            for r in records
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out_path} ({len(records)} games, {wall_min:.1f} min)")
    if args.smoke:
        print("smoke OK")


if __name__ == "__main__":
    main()
