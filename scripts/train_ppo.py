#!/usr/bin/env python
"""PPO self-play training (M3). Headless; config from configs/triad.yaml with
CLI overrides; resumable via --resume (safe on preemptible cloud boxes).

Usage:
    uv run python scripts/train_ppo.py --config configs/triad.yaml
    uv run python scripts/train_ppo.py --beta-kl 0 --no-anchor-init \\
        --output-dir runs/ppo_scratch          # PPO-from-scratch ablation
    uv run python scripts/train_ppo.py --resume runs/ppo/ppo_latest_trainstate.pt
    uv run python scripts/train_ppo.py --smoke
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import yaml

from triad.rl.ppo import PPOConfig, train_ppo


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default="configs/triad.yaml")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--beta-kl", type=float, default=None)
    ap.add_argument("--shaping-alpha", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--anchor", type=str, default=None)
    ap.add_argument("--no-anchor-init", action="store_true",
                    help="random init (PPO-from-scratch ablation)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--output-dir", type=str, default="runs/ppo")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2 envs, 3 tiny updates, checkpoint+resume path, <60s")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.smoke:
        outdir = Path(tempfile.mkdtemp())
        cfg = PPOConfig(
            n_envs=2, rollout_len=16, total_steps=96, lr=1e-3,
            update_epochs=2, n_minibatches=2, eval_every=0, seed=0,
        )
        train_ppo(cfg, anchor_path="weights/bc.pt", init_from_anchor=True,
                  device="cpu", output_dir=outdir, checkpoint_every=2)
        # resume path must work (preemptible-cloud safety)
        cfg2 = PPOConfig(n_envs=2, rollout_len=16, total_steps=128, lr=1e-3,
                         update_epochs=2, n_minibatches=2, eval_every=0, seed=0)
        train_ppo(cfg2, anchor_path="weights/bc.pt", init_from_anchor=True,
                  device="cpu", output_dir=outdir,
                  resume=str(outdir / "ppo_latest_trainstate.pt"),
                  checkpoint_every=2)
        from triad.rl.checkpoint import load_policy
        load_policy(outdir / "ppo_latest.pt", device="cpu")
        print("smoke OK")
        return

    y = load_cfg(args.config)
    pop = y.get("population", {})
    cfg = PPOConfig(
        seed=y.get("seed", 0),
        n_envs=y["env"]["n_envs"],
        shaping_alpha=y["env"]["shaping_alpha"],
        eval_every=y["eval"]["every_updates"],
        eval_games=y["eval"]["games"],
        eval_opponent=y["eval"]["opponent"],
        population=pop.get("enabled", False),
        snapshot_every=pop.get("snapshot_every", 5),
        p_latest=pop.get("p_latest", 0.8),
        **{k: v for k, v in y["ppo"].items()},
    )
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.beta_kl is not None:
        cfg.beta_kl = args.beta_kl
    if args.shaping_alpha is not None:
        cfg.shaping_alpha = args.shaping_alpha
    if args.seed is not None:
        cfg.seed = args.seed

    anchor = args.anchor if args.anchor else y["anchor"]["checkpoint"]
    init_from_anchor = y["anchor"]["init_from_anchor"] and not args.no_anchor_init
    if args.no_anchor_init and cfg.beta_kl == 0:
        anchor = None  # true from-scratch: no anchor at all

    print(f"config: {vars(cfg)}")
    print(f"anchor: {anchor} (init_from_anchor={init_from_anchor})")
    train_ppo(cfg, anchor_path=anchor, init_from_anchor=init_from_anchor,
              device=args.device, output_dir=args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
