"""Synchronous vectorized stepping over N TriadEnvs with auto-reset.

Fixed 3-row (seat) tensors regardless of eliminations: eliminated seats get
zero observations, NOOP-only masks and 0 rewards; their actions are ignored.
This keeps PPO rollout buffers rectangular (CLAUDE.md §4.4: 64 envs).
"""
from __future__ import annotations

import numpy as np

from triad.map_data import POWERS
from triad.env.obs import OBS_DIM
from triad.env.triad_env import MAX_UNITS, N_ACTIONS, NOOP_ID, TriadEnv

N_SEATS = len(POWERS)  # 3


class VecTriadEnv:
    def __init__(self, n_envs: int, shaping_alpha: float = 0.02):
        self.n_envs = n_envs
        self.envs = [TriadEnv(shaping_alpha=shaping_alpha) for _ in range(n_envs)]

    def _gather(self, i: int, obs_d: dict, infos_d: dict, out_obs, out_mask):
        for s, pw in enumerate(POWERS):
            if pw in obs_d:
                out_obs[i, s] = obs_d[pw]
                out_mask[i, s] = infos_d[pw]["action_mask"]
            else:  # eliminated / finished seat
                out_mask[i, s, :, NOOP_ID] = True

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs = np.zeros((self.n_envs, N_SEATS, OBS_DIM), dtype=np.float32)
        mask = np.zeros((self.n_envs, N_SEATS, MAX_UNITS, N_ACTIONS), dtype=bool)
        for i, e in enumerate(self.envs):
            o, inf = e.reset()
            self._gather(i, o, inf, obs, mask)
        return obs, mask

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """actions: int array [n_envs, 3, MAX_UNITS].

        Returns (obs, mask, rewards [n_envs,3], done [n_envs], acted [n_envs,3],
        seat_done [n_envs,3], final_infos).
        acted:     seat was alive when this step's actions were taken (its
                   reward/value belong in the rollout buffer).
        seat_done: seat's episode ended on this step (elimination or game end)
                   — the per-seat GAE bootstrap cut.
        Done envs are auto-reset; their obs/mask are the fresh episode's and
        final_infos[i] carries {"result": ..., "final_rewards": ...}.
        """
        assert actions.shape == (self.n_envs, N_SEATS, MAX_UNITS), actions.shape
        obs = np.zeros((self.n_envs, N_SEATS, OBS_DIM), dtype=np.float32)
        mask = np.zeros((self.n_envs, N_SEATS, MAX_UNITS, N_ACTIONS), dtype=bool)
        rewards = np.zeros((self.n_envs, N_SEATS), dtype=np.float32)
        done = np.zeros(self.n_envs, dtype=bool)
        acted = np.zeros((self.n_envs, N_SEATS), dtype=bool)
        seat_done = np.zeros((self.n_envs, N_SEATS), dtype=bool)
        final_infos: list[dict] = [{} for _ in range(self.n_envs)]

        for i, e in enumerate(self.envs):
            acting = list(e.agents)
            acts = {pw: actions[i, s] for s, pw in enumerate(POWERS) if pw in acting}
            o, r, term, trunc, inf = e.step(acts)
            for s, pw in enumerate(POWERS):
                rewards[i, s] = r.get(pw, 0.0)
                acted[i, s] = pw in acting
                seat_done[i, s] = bool(term.get(pw, False))
            if not e.agents:  # episode over -> auto-reset
                done[i] = True
                final_infos[i] = {
                    "result": e.game.result,  # type: ignore[union-attr]
                    "final_rewards": {pw: float(r.get(pw, 0.0)) for pw in POWERS},
                }
                o, inf = e.reset()
            self._gather(i, o, inf, obs, mask)
        return obs, mask, rewards, done, acted, seat_done, final_infos
