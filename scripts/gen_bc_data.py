#!/usr/bin/env python
"""Generate the BC dataset: Grabber-teacher decisions from mixed bot games.

Usage:
    uv run python scripts/gen_bc_data.py --games 50000 --seed 0 \
        --out data/bc_dataset.npz
    uv run python scripts/gen_bc_data.py --smoke
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from triad.rl.bc import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/bc_dataset.npz")
    ap.add_argument("--smoke", action="store_true", help="tiny run, tmp output")
    args = ap.parse_args()

    if args.smoke:
        out = Path(tempfile.mkdtemp()) / "bc_smoke.npz"
        generate_dataset(n_games=50, seed=args.seed, out_path=out)
        print("smoke OK")
        return
    generate_dataset(n_games=args.games, seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
