#!/usr/bin/env python
"""Train the BC policy on a generated dataset (CLAUDE.md §4.4 stage 1).

Headless; outputs (TensorBoard logs + checkpoints) under --output-dir.

Usage:
    uv run python scripts/train_bc.py --data data/bc_dataset.npz \
        --epochs 3 --device auto --output-dir runs/bc --seed 0
    uv run python scripts/train_bc.py --smoke
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from triad.rl.bc import generate_dataset, train_bc
from triad.rl.checkpoint import save_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/bc_dataset.npz")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--output-dir", type=str, default="runs/bc")
    ap.add_argument("--smoke", action="store_true",
                    help="inline tiny dataset, 1 epoch, tmp output, checkpoint save+reload")
    args = ap.parse_args()

    if args.smoke:
        outdir = Path(tempfile.mkdtemp())
        data = generate_dataset(n_games=60, seed=0)
        model = train_bc(
            data, epochs=1, batch_size=256, seed=0, device="cpu",
            output_dir=outdir, log=True,
        )
        ck = save_checkpoint(model, outdir / "bc_smoke.pt",
                             train_config={"smoke": True}, seed=0)
        from triad.rl.checkpoint import load_policy
        load_policy(ck, device="cpu")  # full save+reload path
        print("smoke OK")
        return

    data = dict(np.load(args.data))
    train_cfg = {
        "data": args.data, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "n_samples": int(len(data["obs"])),
        "teacher": "grabber-heavy mixed lineup",
    }
    model = train_bc(
        data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, device=args.device, output_dir=args.output_dir,
    )
    ck = save_checkpoint(
        model, Path(args.output_dir) / "bc_final.pt",
        train_config=train_cfg, seed=args.seed,
    )
    print(f"checkpoint: {ck}")


if __name__ == "__main__":
    main()
