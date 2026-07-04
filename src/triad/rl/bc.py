"""Behavior cloning (CLAUDE.md §4.4 stage 1).

Dataset: >= 50k games of mixed bot matchups (Grabber-heavy); one sample per
Grabber decision (movement phase or Winter), conditioned on the rotated
observation. Targets: the teacher's order-id sequence + the game's final
outcome shares (value head, normalized per §4.2).

Storage is compact for a 16 GB box: observations are exact uint8 (all
features are k/12, k/20 or binary — scale 240 represents each exactly),
~250 bytes/sample instead of ~800.

Training: teacher-forced cross-entropy over decode steps (targets are legal
by construction, so no mask is needed at train time; play-time decoding is
always hard-masked) + soft-target CE on the value head.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from triad.map_data import POWERS
from triad.bots import Grabber, RandomLegal, Turtle
from triad.engine.game import Game
from triad.engine.orders import ORDER_INDEX, WAIVE_ID
from triad.engine.state import FALL, SPRING
from triad.env.obs import encode_observation, own_frame_unit_order
from triad.rl.checkpoint import resolve_device, save_checkpoint
from triad.rl.models import MAX_STEPS, TriadPolicy
from triad.rl.policy_bot import PolicyBot

OBS_SCALE = 240  # exact for k/12 (20k) and k/20 (12k) feature values

_BOT_MENU = [Grabber, RandomLegal, Turtle]
_BOT_PROBS = [0.70, 0.15, 0.15]  # Grabber-heavy (CLAUDE.md §4.4)


# --- dataset generation ------------------------------------------------------
def _outcome_shares(game: Game) -> dict[str, np.ndarray]:
    """Per power, the value-head target [self, next, prev] (sums to 1)."""
    res = game.result
    assert res is not None
    if res["type"] == "solo":
        raw = {pw: 1.0 if pw == res["winner"] else 0.0 for pw in POWERS}
    else:
        counts = {pw: game.board.sc_count(pw) for pw in POWERS}
        total = sum(counts.values()) or 1
        raw = {pw: counts[pw] / total for pw in POWERS}
    out = {}
    for k, pw in enumerate(POWERS):
        out[pw] = np.array(
            [raw[POWERS[(k + r) % 3]] for r in range(3)], dtype=np.float16
        )
    return out


def generate_dataset(
    n_games: int, seed: int, out_path: str | Path | None = None
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs_l: list[np.ndarray] = []
    ids_l: list[np.ndarray] = []
    nst_l: list[int] = []
    game_idx: list[int] = []  # sample -> game number (for value backfill)
    seat_l: list[str] = []
    shares_per_game: list[dict[str, np.ndarray]] = []

    t0 = time.perf_counter()
    for gi in range(n_games):
        # sample a Grabber-heavy lineup; force at least one Grabber teacher
        lineup = [
            _BOT_MENU[rng.choice(len(_BOT_MENU), p=_BOT_PROBS)]() for _ in POWERS
        ]
        if not any(isinstance(b, Grabber) for b in lineup):
            lineup[int(rng.integers(3))] = Grabber()
        bots = dict(zip(POWERS, lineup))
        teachers = [pw for pw in POWERS if isinstance(bots[pw], Grabber)]

        g = Game()
        while not g.over:
            if g.board.phase in (SPRING, FALL):
                merged = {}
                for pw in g.alive_powers():
                    om = bots[pw].movement_orders(g.board, pw, rng)
                    if pw in teachers:
                        provs = own_frame_unit_order(g.board.units, pw)
                        ids = np.full(MAX_STEPS, -1, dtype=np.int16)
                        for i, p in enumerate(provs):
                            ids[i] = ORDER_INDEX[om[p]]
                        if provs:
                            obs_l.append(
                                np.round(
                                    encode_observation(g.board, pw) * OBS_SCALE
                                ).astype(np.uint8)
                            )
                            ids_l.append(ids)
                            nst_l.append(len(provs))
                            game_idx.append(gi)
                            seat_l.append(pw)
                    merged[pw] = om
                g.step_movement(merged)
            else:
                ob = {}
                for pw in g.alive_powers():
                    wo = bots[pw].winter_orders(g.board, pw, rng)
                    delta = g.winter_delta(pw)
                    if pw in teachers and delta != 0:
                        n = abs(delta)
                        ids = np.full(MAX_STEPS, -1, dtype=np.int16)
                        for i in range(n):  # unfilled build slots -> WAIVE
                            ids[i] = (
                                ORDER_INDEX[wo[i]] if i < len(wo) else WAIVE_ID
                            )
                        obs_l.append(
                            np.round(
                                encode_observation(g.board, pw) * OBS_SCALE
                            ).astype(np.uint8)
                        )
                        ids_l.append(ids)
                        nst_l.append(n)
                        game_idx.append(gi)
                        seat_l.append(pw)
                    ob[pw] = wo
                g.step_winter(ob)
        shares_per_game.append(_outcome_shares(g))

    values = np.stack(
        [shares_per_game[gi][pw] for gi, pw in zip(game_idx, seat_l)]
    )
    data = {
        "obs": np.stack(obs_l),
        "ids": np.stack(ids_l),
        "n_steps": np.array(nst_l, dtype=np.int8),
        "values": values,
        "meta_games": np.array([n_games]),
        "meta_seed": np.array([seed]),
    }
    dt = time.perf_counter() - t0
    print(
        f"generated {len(obs_l)} samples from {n_games} games in {dt:.1f}s "
        f"({n_games / dt:.0f} games/s, {data['obs'].nbytes / 1e6:.0f} MB obs)"
    )
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)
        print(f"wrote {out_path}")
    return data


# --- training ----------------------------------------------------------------
def train_bc(
    data: dict[str, np.ndarray],
    *,
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3,
    value_coef: float = 0.5,
    seed: int = 0,
    device: str = "auto",
    output_dir: str | Path = "runs/bc",
    log: bool = True,
) -> TriadPolicy:
    dev = resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = None
    if log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(outdir))

    obs = torch.from_numpy(data["obs"].astype(np.float32) / OBS_SCALE)
    ids = torch.from_numpy(data["ids"].astype(np.int64))
    n_steps = torch.from_numpy(data["n_steps"].astype(np.int64))
    values = torch.from_numpy(data["values"].astype(np.float32))
    N = len(obs)
    n_val = max(256, N // 50)
    perm = rng.permutation(N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = TriadPolicy().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    for ep in range(epochs):
        model.train()
        order = tr_idx[rng.permutation(len(tr_idx))]
        ce_sum = 0.0
        nb = 0
        for s in range(0, len(order), batch_size):
            b = order[s : s + batch_size]
            o, i_, n, v = (
                obs[b].to(dev),
                ids[b].to(dev),
                n_steps[b].to(dev),
                values[b].to(dev),
            )
            logits, _, _, vlog = model.evaluate_actions(o, i_, n)
            valid = (
                torch.arange(MAX_STEPS, device=dev)[None, :] < n[:, None]
            )
            ce = F.cross_entropy(
                logits[valid], i_.clamp(min=0)[valid], reduction="mean"
            )
            vloss = -(v * F.log_softmax(vlog, dim=-1)).sum(-1).mean()
            loss = ce + value_coef * vloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ce_sum += float(ce.detach())
            nb += 1
            step += 1
            if writer and step % 100 == 0:
                writer.add_scalar("bc/ce", float(ce.detach()), step)
                writer.add_scalar("bc/value_ce", float(vloss.detach()), step)
        # validation
        model.eval()
        with torch.no_grad():
            o, i_, n = obs[val_idx].to(dev), ids[val_idx].to(dev), n_steps[val_idx].to(dev)
            logits, _, _, _ = model.evaluate_actions(o, i_, n)
            valid = torch.arange(MAX_STEPS, device=dev)[None, :] < n[:, None]
            val_ce = float(F.cross_entropy(logits[valid], i_.clamp(min=0)[valid]))
            acc = float(
                (logits[valid].argmax(-1) == i_.clamp(min=0)[valid]).float().mean()
            )
        print(
            f"epoch {ep + 1}/{epochs}: train_ce={ce_sum / max(nb, 1):.4f} "
            f"val_ce={val_ce:.4f} val_top1={acc:.3f}"
        )
        if writer:
            writer.add_scalar("bc/val_ce", val_ce, ep)
            writer.add_scalar("bc/val_top1", acc, ep)
    if writer:
        writer.close()
    return model


# --- evaluation (M2.5 acceptance) ---------------------------------------------
def evaluate_policy(
    model: TriadPolicy,
    opponent: str,
    n_games: int,
    seed: int,
    greedy: bool = True,
) -> dict[str, float]:
    """Play n_games with the policy in a rotating seat vs two copies of the
    opponent bot. Argmax policy vs stochastic bots is statistically sound
    (CLAUDE.md §5); bot rng supplies the game-to-game variation."""
    from triad.bots.base import play_game

    opp = {"random": RandomLegal, "grabber": Grabber, "turtle": Turtle}[opponent]
    rng = np.random.default_rng(seed)
    me = PolicyBot(model, greedy=greedy)
    solo = draw = eliminated = 0
    for k in range(n_games):
        seat = POWERS[k % 3]
        bots = {pw: (me if pw == seat else opp()) for pw in POWERS}
        g = play_game(bots, rng)
        if g.result["type"] == "solo":
            if g.result["winner"] == seat:
                solo += 1
        else:
            draw += 1
        if seat in g.eliminated:
            eliminated += 1
    return {
        "solo_rate": solo / n_games,
        "draw_rate": draw / n_games,
        "eliminated_rate": eliminated / n_games,
        "n_games": n_games,
    }
