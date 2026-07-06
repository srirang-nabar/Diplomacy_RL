#!/usr/bin/env python
"""M4 experiment grid: beta sweep + ablation rows, shared runs (CLAUDE.md §5).

8 rows, all at the standard local budget (500k env-steps) with population
play, so rows are comparable. Each run: train -> 500g eval vs 2xGrabber
(sampled) + 300g vs 2xRandom -> best-snapshot selection from the in-training
eval curve (+500g eval of it) -> seat-symmetry chi^2 -> value-calibration
probe -> registry row (results/m4_registry.{json,md}).

Resumable: rows already present in the registry are skipped.

Usage:
    uv run python scripts/run_m4_grid.py            # full grid (~8h local CPU)
    uv run python scripts/run_m4_grid.py --rows kl05_a02 scratch_a02
    uv run python scripts/run_m4_grid.py --smoke    # micro-grid, <3 min
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

ROWS = [
    # name        beta   alpha  init        serves
    ("kl05_a02",  0.05,  0.02,  "anchor"),  # BC+PPO+KL row + sweep + final-agent candidate
    ("kl0_a02",   0.0,   0.02,  "anchor"),  # BC+PPO row + beta sweep point
    ("kl01_a02",  0.01,  0.02,  "anchor"),  # beta sweep point
    ("kl20_a02",  0.20,  0.02,  "anchor"),  # beta sweep point
    ("scratch_a02", 0.0, 0.02,  "scratch"), # PPO-from-scratch ablation (Q2a)
    ("kl05_a00",  0.05,  0.0,   "anchor"),  # shaping ablation column
    ("kl0_a00",   0.0,   0.0,   "anchor"),
    ("scratch_a00", 0.0, 0.0,   "scratch"),
]


def _chi2_selfplay(model, n_games: int, seed: int) -> dict:
    from scipy.stats import chisquare
    from triad.map_data import POWERS
    from triad.bots.base import play_game
    from triad.rl.policy_bot import PolicyBot

    rng = np.random.default_rng(seed)
    me = PolicyBot(model, greedy=False)
    counts = {pw: 0 for pw in POWERS}
    draws = 0
    for _ in range(n_games):
        g = play_game({pw: me for pw in POWERS}, rng)
        if g.result["type"] == "solo":
            counts[g.result["winner"]] += 1
        else:
            draws += 1
    solos = np.array([counts[pw] for pw in POWERS])
    p = float(chisquare(solos).pvalue) if solos.sum() else 1.0
    return {"solos": counts, "draws": draws, "p": round(p, 4)}


def _value_at_start(model) -> list[float]:
    from triad.engine.state import Board
    from triad.env.obs import encode_observation

    o = torch.from_numpy(encode_observation(Board.initial(), "A")).unsqueeze(0)
    with torch.no_grad():
        return [round(float(x), 3) for x in torch.softmax(model.value_logits(o), -1)[0]]


def _best_snapshot(outdir: Path) -> tuple[int, float, Path] | None:
    """Highest in-training eval score whose update has a pool snapshot."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    evs = sorted(outdir.glob("events*"))
    if not evs:
        return None
    acc = EventAccumulator(str(evs[-1]))
    acc.Reload()
    if "eval/solo_vs_grabber" not in acc.Tags()["scalars"]:
        return None
    best = None
    for pt in acc.Scalars("eval/solo_vs_grabber"):
        update = pt.step // 8192  # steps per update at the standard config
        snap = outdir / "pool" / f"snap_{update:04d}.pt"
        if snap.exists() and (best is None or pt.value > best[1]):
            best = (pt.step, float(pt.value), snap)
    return best


def run_row(name: str, beta: float, alpha: float, init: str, *, smoke: bool,
            device: str, registry: dict, reg_path: Path) -> None:
    from triad.rl.bc import evaluate_policy
    from triad.rl.checkpoint import load_policy
    from triad.rl.ppo import PPOConfig, train_ppo

    outdir = REPO / "runs" / ("m4_smoke" if smoke else "m4") / name
    scale = dict(total_steps=96, n_envs=2, rollout_len=16, update_epochs=1,
                 n_minibatches=1, eval_every=0, snapshot_every=1) if smoke else \
            dict(total_steps=500_000, n_envs=64, rollout_len=128,
                 eval_every=10, snapshot_every=5)
    cfg = PPOConfig(seed=0, beta_kl=beta, shaping_alpha=alpha,
                    population=True, p_latest=0.8, **scale)
    anchor = None if (init == "scratch" and beta == 0.0) else str(REPO / "weights/bc.pt")

    print(f"=== {name}: beta={beta} alpha={alpha} init={init} ===", flush=True)
    t0 = time.perf_counter()
    model = train_ppo(cfg, anchor_path=anchor, init_from_anchor=(init == "anchor"),
                      device=device, output_dir=outdir)
    wall_min = (time.perf_counter() - t0) / 60

    n_eval = 4 if smoke else 500
    n_rand = 4 if smoke else 300
    n_chi = 6 if smoke else 150
    ev_g = evaluate_policy(model, "grabber", n_games=n_eval, seed=1, greedy=False)
    ev_r = evaluate_policy(model, "random", n_games=n_rand, seed=0, greedy=True)

    best = _best_snapshot(outdir)
    best_entry = None
    if best is not None:
        b_model, _ = load_policy(best[2], device="cpu")
        ev_b = evaluate_policy(b_model, "grabber", n_games=n_eval, seed=1, greedy=False)
        best_entry = {"step": best[0], "curve_score": round(best[1], 3),
                      "solo_vs_grabber": ev_b["solo_rate"], "path": str(best[2].relative_to(REPO))}

    row = {
        "beta": beta, "alpha": alpha, "init": init,
        "steps": cfg.total_steps, "wall_min": round(wall_min, 1),
        "final_solo_vs_grabber_sampled": ev_g["solo_rate"],
        "final_elim_vs_grabber": ev_g["eliminated_rate"],
        "final_solo_vs_random": ev_r["solo_rate"],
        "best_snapshot": best_entry,
        "chi2": _chi2_selfplay(model, n_chi, seed=11),
        "value_at_start": _value_at_start(model),
        "checkpoint": str((outdir / "ppo_latest.pt").relative_to(REPO)),
    }
    registry[name] = row
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry, indent=2))
    _write_md(registry, reg_path.with_suffix(".md"))
    print(f"--- {name} done in {wall_min:.0f} min: "
          f"grabber {ev_g['solo_rate']:.3f}, random {ev_r['solo_rate']:.3f}, "
          f"chi2 p={row['chi2']['p']}", flush=True)


def _write_md(registry: dict, path: Path) -> None:
    lines = [
        "# M4 run registry",
        "",
        "All runs: 500k env-steps, population play (0.8 latest / 0.2 pool), seed 0.",
        "Evals: 500g sampled vs 2xGrabber, 300g argmax vs 2xRandom, Wilson CIs in the notebook.",
        "",
        "| run | beta | alpha | init | vs grabber | vs random | best-snap grabber | chi2 p | wall (min) |",
        "|-----|------|-------|------|------------|-----------|-------------------|--------|------------|",
    ]
    for name, r in registry.items():
        bs = r.get("best_snapshot")
        bs_s = f"{bs['solo_vs_grabber']:.3f}@{bs['step']}" if bs else "-"
        lines.append(
            f"| {name} | {r['beta']} | {r['alpha']} | {r['init']} | "
            f"{r['final_solo_vs_grabber_sampled']:.3f} | {r['final_solo_vs_random']:.3f} | "
            f"{bs_s} | {r['chi2']['p']} | {r['wall_min']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="*", default=None, help="subset of row names")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="2-row micro-grid, <3 min")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    reg_path = REPO / "results" / ("m4_registry_smoke.json" if args.smoke else "m4_registry.json")
    registry: dict = json.loads(reg_path.read_text()) if reg_path.exists() else {}

    rows = ROWS[:2] if args.smoke else ROWS
    if args.rows:
        rows = [r for r in ROWS if r[0] in args.rows]
    for name, beta, alpha, init in rows:
        if name in registry:
            print(f"skip {name} (already in registry)", flush=True)
            continue
        run_row(name, beta, alpha, init, smoke=args.smoke,
                device=args.device, registry=registry, reg_path=reg_path)
    print("grid complete" + (" (smoke OK)" if args.smoke else ""), flush=True)


if __name__ == "__main__":
    main()
