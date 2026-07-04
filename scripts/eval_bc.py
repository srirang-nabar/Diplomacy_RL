#!/usr/bin/env python
"""M2.5 acceptance evaluation: policy (rotating seat, argmax) vs two copies
of a fixed bot. Acceptance: >80% solo vs random; ~1/3 'each' vs grabber.

Usage:
    uv run python scripts/eval_bc.py --checkpoint runs/bc/bc_final.pt \
        --opponent random --games 500 --seed 0
    uv run python scripts/eval_bc.py --smoke
"""
from __future__ import annotations

import argparse

from triad.rl.bc import evaluate_policy
from triad.rl.checkpoint import load_policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, default="runs/bc/bc_final.pt")
    ap.add_argument("--opponent", type=str, default="random",
                    choices=["random", "grabber", "turtle"])
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--sample", action="store_true",
                    help="temperature-1.0 sampling instead of argmax")
    ap.add_argument("--smoke", action="store_true",
                    help="untrained model, 6 games — exercises the full path")
    args = ap.parse_args()

    if args.smoke:
        from triad.rl.models import TriadPolicy
        stats = evaluate_policy(TriadPolicy(), "random", n_games=6, seed=0)
        print(f"smoke stats: {stats}")
        print("smoke OK")
        return

    model, payload = load_policy(args.checkpoint, device=args.device)
    print(f"loaded {args.checkpoint} (git {payload['git_sha']}, seed {payload['seed']})")
    stats = evaluate_policy(
        model, args.opponent, n_games=args.games, seed=args.seed,
        greedy=not args.sample,
    )
    mode = "sampled" if args.sample else "argmax"
    print(f"vs 2x{args.opponent} ({mode}): {stats}")


if __name__ == "__main__":
    main()
