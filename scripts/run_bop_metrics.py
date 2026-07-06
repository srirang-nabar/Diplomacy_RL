#!/usr/bin/env python
"""Balance-of-power metrics runs (CLAUDE.md §5 headline analyses), per beta.

For each config: N traced self-play games -> chance-corrected
attack-the-leader by lead size, lead-conversion curve, cross-power support
rates. Grabber mirror is the scripted baseline. Writes results/bop_metrics.json.

Usage:
    uv run python scripts/run_bop_metrics.py             # full (~2h CPU)
    uv run python scripts/run_bop_metrics.py --games 400
    uv run python scripts/run_bop_metrics.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CONFIGS = [
    # name          kind      source
    ("grabber",     "bot",    None),                              # scripted baseline
    ("bc",          "ckpt",   "weights/bc.pt"),
    ("scratch",     "ckpt",   "runs/m4/scratch_a02/ppo_latest.pt"),
    ("beta0_final", "ckpt",   "weights/ppo_final.pt"),            # kl0_a02
    ("beta01",      "ckpt",   "runs/m4/kl01_a02/ppo_latest.pt"),
    ("beta05",      "ckpt",   "weights/ppo_kl05.pt"),             # kl05_a02
    ("beta20",      "ckpt",   "runs/m4/kl20_a02/ppo_latest.pt"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/bop_metrics.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from triad.bots import Grabber
    from triad.eval.metrics import (
        attack_the_leader,
        cross_support,
        lead_conversion,
        run_traced_selfplay,
    )
    from triad.rl.checkpoint import load_policy
    from triad.rl.policy_bot import PolicyBot

    n_games = 3 if args.smoke else args.games
    out_path = Path("/tmp/bop_smoke.json") if args.smoke else REPO / args.out
    configs = CONFIGS[:2] if args.smoke else CONFIGS

    results = {}
    for name, kind, src in configs:
        if kind == "ckpt" and not (REPO / src).exists():
            print(f"[bop] skip {name}: {src} missing", flush=True)
            continue
        t0 = time.perf_counter()
        if kind == "bot":
            bot = Grabber()
        else:
            model, _ = load_policy(REPO / src, device="cpu")
            bot = PolicyBot(model, greedy=False)
        traces = run_traced_selfplay(bot, n_games, seed=args.seed)
        results[name] = {
            "source": src or "scripted",
            "n_games": n_games,
            "attack_the_leader": attack_the_leader(traces),
            "lead_conversion": lead_conversion(traces),
            "cross_support": cross_support(traces),
            "solo_rate": sum(1 for t in traces if t.winner) / n_games,
            "wall_min": round((time.perf_counter() - t0) / 60, 1),
        }
        print(f"[bop] {name}: {n_games} games in {results[name]['wall_min']} min "
              f"(cross_vs_leader={results[name]['cross_support']['cross_support_vs_leader_rate']})",
              flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_path}")
    if args.smoke:
        print("smoke OK")


if __name__ == "__main__":
    main()
