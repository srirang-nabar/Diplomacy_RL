#!/usr/bin/env python
"""Regenerate the report figures from the committed results files.

Deterministic, seconds to run — anyone reproducing the report runs this after
run_tournament.py / run_bop_metrics.py. Writes report/figs/*.png.

Usage:
    uv run python scripts/make_report_figs.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "report" / "figs"

BLUE, GRAY, RED, PURPLE = "#4878cf", "#9aa7b8", "#c66a6a", "#8a6fb8"


def wilson(p, n, z=1.96):
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - hw, c + hw


def fig_tournament(tourn):
    pt = tourn["policy_table"]
    names = sorted(pt, key=lambda n: -pt[n]["solo_rate"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(names))
    vals = [pt[n]["solo_rate"] for n in names]
    errs = np.array([[v - pt[n]["ci"][0], pt[n]["ci"][1] - v]
                     for v, n in zip(vals, names)]).T
    ax.barh(y, vals, xerr=errs, capsize=3, height=0.55,
            color=[BLUE if n == "final" else GRAY for n in names])
    ax.axvline(1 / 3, ls=":", lw=1, c="gray")
    ax.text(1 / 3 + 0.005, len(names) - 0.5, "1/3", fontsize=8, c="gray")
    ax.set_yticks(y, names)
    ax.set_xlabel("solo win rate, all lineups (95% Wilson)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    for v, yi, n in zip(vals, y, names):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "tournament.png", dpi=160)
    plt.close(fig)


def fig_ablation_curves():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for run, label, c in [("scratch_a02", "PPO from scratch", RED),
                          ("kl0_a02", "BC init, beta=0 (final)", BLUE),
                          ("kl05_a02", "BC init, beta=0.05", PURPLE)]:
        evs = sorted((REPO / "runs" / "m4" / run).glob("events*"))
        if not evs:
            continue
        acc = EventAccumulator(str(evs[-1]))
        acc.Reload()
        pts = acc.Scalars("eval/solo_vs_grabber")
        ax.plot([p.step / 1e6 for p in pts], [p.value for p in pts],
                "o-", ms=3, lw=1.3, label=label, color=c)
    ax.axhline(0.328, ls="--", lw=1, c="gray")
    ax.text(0.005, 0.345, "BC only", fontsize=7, c="gray")
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel("solo vs 2x grabber (sampled)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "ablation_curves.png", dpi=160)
    plt.close(fig)


def fig_attack_the_leader(bop):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for name, c, label in [("grabber", GRAY, "grabber (scripted baseline)"),
                           ("bc", PURPLE, "BC clone"),
                           ("beta0_final", BLUE, "RL final (beta=0)"),
                           ("beta05", RED, "RL beta=0.05")]:
        atl = bop[name]["attack_the_leader"]
        pts = [(int(k), v["excess"]) for k, v in atl.items()
               if v["n_unit_phases"] >= 200]
        if pts:
            ax.plot(*zip(*sorted(pts)), "o-", lw=1.3, label=label, color=c)
    ax.axhline(0, lw=1, c="gray")
    ax.set_xlabel("leader's SC lead k")
    ax.set_ylabel("leader-targeting excess over chance")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "attack_the_leader.png", dpi=160)
    plt.close(fig)


def fig_lead_conversion(bop):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for name, c, label in [("grabber", GRAY, "grabber (scripted baseline)"),
                           ("bc", PURPLE, "BC clone"),
                           ("beta0_final", BLUE, "RL final (beta=0)")]:
        lc = bop[name]["lead_conversion"]
        pts = [(int(k), v["p_solo"]) for k, v in lc.items() if v["n"] >= 30]
        if pts:
            ax.plot(*zip(*sorted(pts)), "o-", lw=1.3, label=label, color=c)
    ax.set_xlabel("max SC lead reached k")
    ax.set_ylabel("P(eventual solo)")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "lead_conversion.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    tourn = json.loads((REPO / "results/tournament.json").read_text())
    bop = json.loads((REPO / "results/bop_metrics.json").read_text())
    fig_tournament(tourn)
    fig_attack_the_leader(bop)
    fig_lead_conversion(bop)
    fig_ablation_curves()
    print(f"wrote {len(list(FIGS.glob('*.png')))} figures to {FIGS}")
    if args.smoke:
        print("smoke OK")


if __name__ == "__main__":
    main()
