"""PPO self-play with a KL anchor to the BC policy (CLAUDE.md §4.4 stage 2).

Loop structure adapted from CleanRL's ppo.py (MIT) — vendored and modified,
not imported. Changes for Triad:

- 3-seat simultaneous self-play: one shared policy plays every seat through
  rotation-to-own-frame observations; every seat's experience trains.
- Joint action log-prob = sum over autoregressive decode steps (§4.3);
  update-time distributions reconstruct the exact rollout masking
  (including Winter's exclude-emitted rule) so ratios are exact.
- Loss = clipped surrogate + vf loss + entropy bonus
  + beta * KL(pi_theta || pi_BC), pi_BC frozen.
- Per-seat GAE(lambda), gamma = 1.0 (finite episodes, terminal-heavy
  reward); seat_done cuts the bootstrap at elimination or game end.

Documented deviation from §4.4's "value CE": on rollout fragments the 3-way
final-share target is unavailable until the episode ends, so the value loss
is MSE of the value head's *self* probability against the GAE return —
the same target in expectation at gamma = 1. Revisit if value quality lags.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from triad.engine.orders import WAIVE_ID
from triad.env.obs import OBS_DIM
from triad.env.triad_env import MAX_UNITS, NOOP_ID
from triad.env.vec import VecTriadEnv
from triad.rl.checkpoint import load_policy, resolve_device, save_checkpoint
from triad.rl.models import TriadPolicy

N_SEATS = 3
V = 336  # order vocab (mask column NOOP_ID is padding, stripped for the policy)


class PPOConfig:
    """Flat hyperparameter bag; see configs/triad.yaml for the source values."""

    def __init__(self, **kw):
        self.n_envs = 64
        self.rollout_len = 128
        self.total_steps = 2_000_000       # env steps (x3 seats = samples)
        self.lr = 3e-4
        self.anneal_lr = True
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.update_epochs = 4
        self.n_minibatches = 8
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.beta_kl = 0.05                # KL(pi_theta || pi_BC) weight
        self.shaping_alpha = 0.02
        self.eval_every = 10               # updates between quick bot evals
        self.eval_games = 45
        self.eval_opponent = "grabber"
        self.seed = 0
        for k, v in kw.items():
            assert hasattr(self, k), f"unknown ppo config key {k!r}"
            setattr(self, k, v)


def train_ppo(
    cfg: PPOConfig,
    *,
    anchor_path: str | None,
    init_from_anchor: bool = True,
    device: str = "auto",
    output_dir: str | Path = "runs/ppo",
    resume: str | None = None,
    log: bool = True,
    checkpoint_every: int = 25,
) -> TriadPolicy:
    dev = resolve_device(device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = None
    if log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(outdir))

    # policy + frozen anchor
    anchor = None
    if anchor_path is not None:
        anchor, _ = load_policy(anchor_path, device=dev)
        for p in anchor.parameters():
            p.requires_grad_(False)
    if init_from_anchor:
        assert anchor is not None, "init_from_anchor requires an anchor checkpoint"
        model = TriadPolicy(anchor.config)
        model.load_state_dict(anchor.state_dict())
    else:
        model = TriadPolicy()
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    start_update = 0
    global_step = 0
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        start_update = int(state["update"])
        global_step = int(state["global_step"])
        print(f"resumed from {resume} at update {start_update} ({global_step} steps)")

    envs = VecTriadEnv(cfg.n_envs, shaping_alpha=cfg.shaping_alpha)
    obs_np, mask_np = envs.reset()

    T, N = cfg.rollout_len, cfg.n_envs
    steps_per_update = T * N
    n_updates = cfg.total_steps // steps_per_update
    B = N * N_SEATS  # flattened actor batch per env step

    # rollout storage (cpu tensors; model may live on gpu)
    b_obs = torch.zeros(T, N, N_SEATS, OBS_DIM)
    b_mask = torch.zeros(T, N, N_SEATS, MAX_UNITS, V, dtype=torch.bool)
    b_ids = torch.zeros(T, N, N_SEATS, MAX_UNITS, dtype=torch.long)
    b_nst = torch.zeros(T, N, N_SEATS, dtype=torch.long)
    b_logp = torch.zeros(T, N, N_SEATS)
    b_val = torch.zeros(T, N, N_SEATS)
    b_rew = torch.zeros(T, N, N_SEATS)
    b_acted = torch.zeros(T, N, N_SEATS, dtype=torch.bool)
    b_sdone = torch.zeros(T, N, N_SEATS, dtype=torch.bool)

    ep_returns: list[float] = []
    ep_solo = ep_draw = 0
    t_start = time.perf_counter()

    for update in range(start_update, n_updates):
        if cfg.anneal_lr:
            frac = 1.0 - update / n_updates
            opt.param_groups[0]["lr"] = frac * cfg.lr

        # ---- collect rollout -------------------------------------------------
        model.eval()
        for t in range(T):
            obs_t = torch.from_numpy(obs_np).reshape(B, OBS_DIM)
            m336 = torch.from_numpy(mask_np[..., :V]).reshape(B, MAX_UNITS, V)
            nst = m336.any(-1).sum(-1)  # live decode rows are always a prefix
            with torch.no_grad():
                ids, logp, vals = model.act(
                    obs_t.to(dev), m336.to(dev), nst.to(dev),
                    generator=gen if dev.type == "cpu" else None,
                    exclude_emitted=True, repeat_ok=WAIVE_ID,
                )
            ids, logp, vals = ids.cpu(), logp.cpu(), vals.cpu()
            b_obs[t] = obs_t.view(N, N_SEATS, OBS_DIM)
            b_mask[t] = m336.view(N, N_SEATS, MAX_UNITS, V)
            b_ids[t] = ids.view(N, N_SEATS, MAX_UNITS)
            b_nst[t] = nst.view(N, N_SEATS)
            b_logp[t] = logp.view(N, N_SEATS)
            b_val[t] = vals[:, 0].view(N, N_SEATS)  # self component = baseline

            actions = ids.view(N, N_SEATS, MAX_UNITS).numpy()
            obs_np, mask_np, rew, done, acted, sdone, infos = envs.step(actions)
            b_rew[t] = torch.from_numpy(rew)
            b_acted[t] = torch.from_numpy(acted)
            b_sdone[t] = torch.from_numpy(sdone)
            global_step += N
            for inf in infos:
                if inf:
                    ep_returns.append(sum(inf["final_rewards"].values()))
                    if inf["result"]["type"] == "solo":
                        ep_solo += 1
                    else:
                        ep_draw += 1

        # ---- GAE(lambda), per seat, gamma = cfg.gamma ---------------------------
        with torch.no_grad():
            obs_last = torch.from_numpy(obs_np).reshape(B, OBS_DIM).to(dev)
            next_val = F.softmax(model.value_logits(obs_last), -1)[:, 0].cpu().view(N, N_SEATS)
        advantages = torch.zeros_like(b_rew)
        lastgae = torch.zeros(N, N_SEATS)
        for t in reversed(range(T)):
            nonterm = (~b_sdone[t]).float()
            nv = next_val if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + cfg.gamma * nv * nonterm - b_val[t]
            lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterm * lastgae
            advantages[t] = lastgae
        returns = advantages + b_val

        # ---- flatten valid samples ---------------------------------------------
        flat = b_acted.reshape(-1)
        idx = torch.nonzero(flat).flatten()
        f_obs = b_obs.reshape(-1, OBS_DIM)[idx]
        f_mask = b_mask.reshape(-1, MAX_UNITS, V)[idx]
        f_ids = b_ids.reshape(-1, MAX_UNITS)[idx]
        f_nst = b_nst.reshape(-1)[idx]
        f_logp = b_logp.reshape(-1)[idx]
        f_adv = advantages.reshape(-1)[idx]
        f_ret = returns.reshape(-1)[idx]
        n_samples = len(idx)
        mb_size = n_samples // cfg.n_minibatches

        # ---- PPO update ------------------------------------------------------------
        model.train()
        clipfracs, pg_losses, v_losses, ents, kls, approx_kls = [], [], [], [], [], []
        order = torch.from_numpy(rng.permutation(n_samples))
        for _ep in range(cfg.update_epochs):
            order = order[torch.randperm(n_samples, generator=gen)]
            for s in range(0, n_samples - mb_size + 1, mb_size):
                mb = order[s : s + mb_size]
                o = f_obs[mb].to(dev)
                n_ = f_nst[mb].to(dev)
                # decode only to the minibatch's real max step count — exact,
                # and ~2.5x cheaper than always unrolling MAX_UNITS steps
                tmax = max(int(n_.max().item()), 1)
                m = f_mask[mb, :tmax].to(dev)
                i_ = f_ids[mb, :tmax].to(dev)
                lp_steps, newlogp, entropy, vlogits = model.evaluate_actions(
                    o, i_, n_, masks=m, exclude_emitted=True, repeat_ok=WAIVE_ID
                )
                logratio = newlogp - f_logp[mb].to(dev)
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kls.append(((ratio - 1) - logratio).mean().item())
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                adv = f_adv[mb].to(dev)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_self = F.softmax(vlogits, -1)[:, 0]
                v_loss = F.mse_loss(v_self, f_ret[mb].to(dev))

                ent_loss = entropy.mean()

                kl_anchor = torch.zeros((), device=dev)
                if anchor is not None and cfg.beta_kl > 0:
                    with torch.no_grad():
                        a_lp, _, _, _ = anchor.evaluate_actions(
                            o, i_, n_, masks=m, exclude_emitted=True, repeat_ok=WAIVE_ID
                        )
                    valid = (
                        torch.arange(tmax, device=dev)[None, :] < n_[:, None]
                    ).float()
                    p_theta = lp_steps.exp()
                    kl_steps = (p_theta * (lp_steps - a_lp)).sum(-1)  # [B, T]
                    kl_anchor = (kl_steps * valid).sum(-1).mean()

                loss = (
                    pg_loss
                    - cfg.ent_coef * ent_loss
                    + cfg.vf_coef * v_loss
                    + cfg.beta_kl * kl_anchor
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ents.append(ent_loss.item())
                kls.append(float(kl_anchor.detach()))

        # ---- logging -----------------------------------------------------------------
        y_pred, y_true = b_val.reshape(-1)[idx].numpy(), f_ret.numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.perf_counter() - t_start)) if update > start_update else 0
        if writer:
            writer.add_scalar("ppo/pg_loss", np.mean(pg_losses), global_step)
            writer.add_scalar("ppo/value_loss", np.mean(v_losses), global_step)
            writer.add_scalar("ppo/entropy", np.mean(ents), global_step)
            writer.add_scalar("ppo/kl_anchor", np.mean(kls), global_step)
            writer.add_scalar("ppo/approx_kl", np.mean(approx_kls), global_step)
            writer.add_scalar("ppo/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("ppo/explained_variance", explained_var, global_step)
            writer.add_scalar("ppo/lr", opt.param_groups[0]["lr"], global_step)
            if ep_returns:
                writer.add_scalar("selfplay/mean_episode_total_reward",
                                  float(np.mean(ep_returns)), global_step)
                n_eps = ep_solo + ep_draw
                writer.add_scalar("selfplay/solo_rate",
                                  ep_solo / max(n_eps, 1), global_step)
        if update % 5 == 0 or update == n_updates - 1:
            print(
                f"update {update + 1}/{n_updates} step {global_step} "
                f"({sps}/s) pg={np.mean(pg_losses):.4f} v={np.mean(v_losses):.4f} "
                f"ent={np.mean(ents):.2f} klA={np.mean(kls):.4f} ev={explained_var:.2f} "
                f"selfplay_solo={ep_solo / max(ep_solo + ep_draw, 1):.2f}",
                flush=True,
            )
        ep_returns, ep_solo, ep_draw = [], 0, 0

        # ---- periodic eval vs fixed bots -----------------------------------------------
        if writer and cfg.eval_every and (update + 1) % cfg.eval_every == 0:
            from triad.rl.bc import evaluate_policy

            model.eval()
            st = evaluate_policy(
                model, cfg.eval_opponent, n_games=cfg.eval_games,
                seed=cfg.seed + update, greedy=False,
            )
            writer.add_scalar(f"eval/solo_vs_{cfg.eval_opponent}", st["solo_rate"], global_step)
            print(f"  eval vs 2x{cfg.eval_opponent} (sampled, {cfg.eval_games}g): "
                  f"solo {st['solo_rate']:.2f}", flush=True)
            model.train()

        # ---- checkpointing (resumable; trainstate never shipped) ------------------------
        if (update + 1) % checkpoint_every == 0 or update == n_updates - 1:
            save_checkpoint(model, outdir / "ppo_latest.pt",
                            train_config=vars(cfg) | {"update": update + 1},
                            seed=cfg.seed)
            torch.save(
                {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": opt.state_dict(),
                    "update": update + 1,
                    "global_step": global_step,
                },
                outdir / "ppo_latest_trainstate.pt",
            )

    if writer:
        writer.close()
    model.eval()
    return model
