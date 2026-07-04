"""Policy/value network (CLAUDE.md §4.2/§4.3).

Baseline: flat 203-dim observation -> MLP torso (2x256 ReLU); heads:
  - autoregressive order decoder: GRUCell over embeddings of already-emitted
    order ids, logits over the 336-order vocab, hard legality mask at play
    time. One decode step per own unit (movement) or per adjustment slot
    (Winter, |delta| steps).
  - value head: softmax logits over {self, next, prev} outcome shares
    (DipNet-style); the self component doubles as the PPO baseline.

Joint log-prob of an action = sum of per-step log-probs (PPO ratios, §4.3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from triad.engine.orders import VOCAB_SIZE
from triad.env.obs import OBS_DIM

MAX_STEPS = 12
START_TOKEN = VOCAB_SIZE  # embedding index for "no order emitted yet"

DEFAULT_CONFIG: dict = {
    "obs_dim": OBS_DIM,       # 203
    "vocab": VOCAB_SIZE,      # 336
    "torso_hidden": 256,
    "embed_dim": 64,
    "gru_hidden": 256,
}


class TriadPolicy(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = cfg
        h = cfg["torso_hidden"]
        self.torso = nn.Sequential(
            nn.Linear(cfg["obs_dim"], h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.h0 = nn.Linear(h, cfg["gru_hidden"])
        self.embed = nn.Embedding(cfg["vocab"] + 1, cfg["embed_dim"])  # +1 start
        self.cell = nn.GRUCell(cfg["embed_dim"], cfg["gru_hidden"])
        self.order_head = nn.Linear(cfg["gru_hidden"], cfg["vocab"])
        self.value_head = nn.Linear(h, 3)  # {self, next, prev} outcome logits

    # --- shared pieces -------------------------------------------------------
    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.torso(obs)
        return z, self.h0(z)

    def value_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.torso(obs))

    # --- teacher forcing (BC training / PPO evaluation) -----------------------
    def evaluate_actions(
        self,
        obs: torch.Tensor,          # [B, obs_dim]
        ids: torch.Tensor,          # [B, T] long, pad values ignored
        n_steps: torch.Tensor,      # [B] long
        masks: torch.Tensor | None = None,  # [B, T, vocab] bool; None = unmasked
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (step_logits [B,T,vocab], joint_logprob [B], entropy [B],
        value_logits [B,3]). Log-prob/entropy are masked (when masks given)
        and summed over valid steps only — identical recurrence to act()."""
        B, T = ids.shape
        z, h = self._encode(obs)
        prev = torch.full((B,), START_TOKEN, dtype=torch.long, device=obs.device)
        safe_ids = ids.clamp(min=0)
        logits_steps = []
        logprob = torch.zeros(B, device=obs.device)
        entropy = torch.zeros(B, device=obs.device)
        for t in range(T):
            h = self.cell(self.embed(prev), h)
            lg = self.order_head(h)
            logits_steps.append(lg)
            lg_masked = lg if masks is None else lg.masked_fill(~masks[:, t], -1e9)
            lp = F.log_softmax(lg_masked, dim=-1)
            valid = (t < n_steps).to(lp.dtype)
            logprob = logprob + lp.gather(1, safe_ids[:, t : t + 1]).squeeze(1) * valid
            p = lp.exp()
            entropy = entropy + (-(p * lp).sum(-1)) * valid
            prev = safe_ids[:, t]
        return torch.stack(logits_steps, dim=1), logprob, entropy, self.value_head(z)

    # --- sequential decoding (rollouts / eval) ----------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,             # [B, obs_dim]
        masks: torch.Tensor,           # [B, T, vocab] bool
        n_steps: torch.Tensor,         # [B] long
        greedy: bool = False,
        generator: torch.Generator | None = None,
        exclude_emitted: bool = False,  # Winter: mask out already-emitted ids
        repeat_ok: int | None = None,   # id exempt from exclusion (WAIVE)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (ids [B,T], joint_logprob [B], value_probs [B,3])."""
        B, T, V = masks.shape
        z, h = self._encode(obs)
        m = masks.clone()
        prev = torch.full((B,), START_TOKEN, dtype=torch.long, device=obs.device)
        ids = torch.zeros(B, T, dtype=torch.long, device=obs.device)
        logprob = torch.zeros(B, device=obs.device)
        t_max = int(n_steps.max().item()) if B else 0
        for t in range(t_max):
            h = self.cell(self.embed(prev), h)
            lg = self.order_head(h).masked_fill(~m[:, t], -1e9)
            lp = F.log_softmax(lg, dim=-1)
            if greedy:
                a = lp.argmax(dim=-1)
            else:
                a = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1)
            valid = t < n_steps
            ids[:, t] = torch.where(valid, a, torch.zeros_like(a))
            logprob = logprob + lp.gather(1, a.unsqueeze(1)).squeeze(1) * valid.to(lp.dtype)
            if exclude_emitted and t + 1 < T:
                for b in torch.nonzero(valid).flatten().tolist():
                    if int(a[b]) != repeat_ok:
                        m[b, t + 1 :, a[b]] = False
            prev = torch.where(valid, a, prev)
        return ids, logprob, F.softmax(self.value_head(z), dim=-1)
