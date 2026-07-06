"""PettingZoo ParallelEnv wrapper around the Triad engine (CLAUDE.md §2.3/M2).

Action space (fixed shape — PettingZoo needs one):
    MultiDiscrete([337] * 12): 12 slots, each an order id in [0, 336] where
    336 (= VOCAB_SIZE) is the NOOP/pad id.

    Movement phase: slot i orders this power's i-th unit in canonical
    province order. NOOP or an illegal id is coerced to HOLD by the engine.
    Winter phase:   slot i is the i-th adjustment decision — |delta| slots
    are live (BUILD.../WAIVE when delta > 0, DISBAND... when delta < 0);
    delta = 0 powers have no live slots (auto-WAIVE). Extra slots ignored.

Legality masks: infos[agent]["action_mask"] is a bool [12, 337] array for
the NEXT decision; padded slots allow only NOOP.

Rewards (§4.5): terminal solo 1.0 / elimination 0.0 / cap-draw SC_i/12, plus
optional shaping alpha * delta-SC at each Winter (alpha=0 is an ablation row).
Eliminated agents exit with terminal reward 0.0; the game continues.

The env itself is deterministic: all stochasticity lives in the policies.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from triad.map_data import POWERS
from triad.engine.game import Game
from triad.engine.orders import (
    VOCAB_SIZE,
    WAIVE_ID,
    legal_movement_orders,
    to_own_frame_ids,
    to_real_order,
)
from triad.engine.state import FALL, SPRING, WINTER
from triad.env.obs import OBS_DIM, POWER_INDEX, encode_observation, own_frame_unit_order

MAX_UNITS = 12
NOOP_ID = VOCAB_SIZE          # 336
N_ACTIONS = VOCAB_SIZE + 1    # 337


class TriadEnv(ParallelEnv):
    metadata = {"name": "triad_v0", "render_modes": []}

    def __init__(self, shaping_alpha: float = 0.02):
        self.shaping_alpha = shaping_alpha
        self.possible_agents = list(POWERS)
        self.agents: list[str] = []
        self.game: Game | None = None
        self._last_sc: dict[str, int] = {}

    # --- spaces --------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([N_ACTIONS] * MAX_UNITS)

    # --- helpers -------------------------------------------------------------
    def _action_mask(self, power: str) -> np.ndarray:
        mask = np.zeros((MAX_UNITS, N_ACTIONS), dtype=bool)
        g = self.game
        assert g is not None
        if g.over or power in g.eliminated:
            mask[:, NOOP_ID] = True
            return mask
        k = POWER_INDEX[power]
        if g.board.phase in (SPRING, FALL):
            # own-frame unit order + own-frame order ids: both are required
            # for seat equivariance (identical positions -> identical masks)
            provs = own_frame_unit_order(g.board.units, power)
            for i, p in enumerate(provs):
                real = legal_movement_orders(g.board.units, p)
                mask[i, to_own_frame_ids(real, k)] = True
            for i in range(len(provs), MAX_UNITS):
                mask[i, NOOP_ID] = True
        else:  # WINTER
            delta = g.winter_delta(power)
            if delta > 0:
                ids = to_own_frame_ids(g.legal_build_ids(power), k)
                for i in range(delta):
                    mask[i, ids] = True
                    mask[i, WAIVE_ID] = True  # WAIVE is a rotation fixed point
                live = delta
            elif delta < 0:
                ids = to_own_frame_ids(g.legal_disband_ids(power), k)
                for i in range(-delta):
                    mask[i, ids] = True
                live = -delta
            else:
                live = 0
            for i in range(live, MAX_UNITS):
                mask[i, NOOP_ID] = True
        return mask

    def _obs(self, power: str) -> np.ndarray:
        assert self.game is not None
        return encode_observation(self.game.board, power)

    def _obs_infos(self, agents: list[str]):
        obs = {a: self._obs(a) for a in agents}
        infos = {
            a: {"action_mask": self._action_mask(a), "phase": self.game.board.phase}
            for a in agents
        }
        return obs, infos

    # --- PettingZoo API --------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        # the env is deterministic; seed is accepted for API compliance only
        self.game = Game()
        self.agents = list(self.possible_agents)
        self._last_sc = {pw: self.game.board.sc_count(pw) for pw in POWERS}
        return self._obs_infos(self.agents)

    def step(self, actions: dict[str, np.ndarray]):
        g = self.game
        assert g is not None and self.agents, "call reset() first"
        acting = list(self.agents)

        if g.board.phase in (SPRING, FALL):
            merged: dict[str, dict[str, object]] = {}
            for pw in acting:
                a = np.asarray(actions[pw]).ravel()
                k = POWER_INDEX[pw]
                provs = own_frame_unit_order(g.board.units, pw)
                om = {}
                for i, p in enumerate(provs):
                    aid = int(a[i]) if i < len(a) else NOOP_ID
                    if 0 <= aid < VOCAB_SIZE:
                        om[p] = to_real_order(aid, k)  # own-frame -> real
                    # NOOP / out-of-range -> no order -> engine coerces to HOLD
                merged[pw] = om
            g.step_movement(merged)
            rewards = {pw: 0.0 for pw in acting}
        else:  # WINTER
            ob: dict[str, list[object]] = {}
            for pw in acting:
                a = np.asarray(actions[pw]).ravel()
                k = POWER_INDEX[pw]
                delta = g.winter_delta(pw)
                live = abs(delta)
                ob[pw] = [
                    to_real_order(int(a[i]), k)
                    for i in range(min(live, len(a)))
                    if 0 <= int(a[i]) < VOCAB_SIZE
                ]
            pre_elim = set(g.eliminated)
            g.step_winter(ob)
            rewards = {}
            for pw in acting:
                sc = g.board.sc_count(pw)
                rewards[pw] = self.shaping_alpha * (sc - self._last_sc[pw])
                self._last_sc[pw] = sc
            # newly eliminated agents: terminal reward 0.0 (overrides shaping)
            for pw in g.eliminated - pre_elim:
                rewards[pw] = 0.0

        terminations = {pw: False for pw in acting}
        truncations = {pw: False for pw in acting}
        if g.over:
            scores = g.result["scores"]  # type: ignore[index]
            for pw in acting:
                terminations[pw] = True
                rewards[pw] = rewards.get(pw, 0.0) + float(scores[pw])
            self.agents = []
        else:
            for pw in list(acting):
                if pw in g.eliminated:
                    terminations[pw] = True
            self.agents = [pw for pw in acting if not terminations[pw]]

        obs, infos = self._obs_infos(acting)
        return obs, rewards, terminations, truncations, infos

    def render(self):  # pragma: no cover - not used
        return None

    def close(self):  # pragma: no cover
        return None
