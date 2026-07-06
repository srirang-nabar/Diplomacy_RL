"""A trained TriadPolicy wrapped as a Bot: playable in bot games and
tournaments alongside the heuristics. CPU-friendly by construction (§7.1)."""
from __future__ import annotations

import numpy as np
import torch

from triad.map_data import HOME_CENTERS
from triad.bots.base import Bot
from triad.engine.orders import (
    ORDERS,
    VOCAB_PERM,
    VOCAB_SIZE,
    WAIVE,
    WAIVE_ID,
    Order,
    legal_movement_orders,
    to_own_frame_ids,
)
from triad.engine.state import Board
from triad.env.obs import POWER_INDEX, encode_observation, own_frame_unit_order
from triad.rl.models import MAX_STEPS, TriadPolicy


class PolicyBot(Bot):
    """greedy=True for argmax play; else temperature-1.0 sampling seeded from
    the game rng (deterministic given the rng state)."""

    def __init__(self, model: TriadPolicy, greedy: bool = True, device: str = "cpu"):
        self.model = model.eval()
        self.greedy = greedy
        self.device = torch.device(device)

    def _decode(
        self,
        board: Board,
        power: str,
        step_ids: list[list[int]],  # REAL-frame legal ids per decode step
        rng: np.random.Generator,
        exclude_emitted: bool,
        repeat_ok: int | None = None,
    ) -> list[int]:
        """Decode in the acting power's OWN frame (obs and action ids both
        rotated), then return REAL-frame order ids."""
        n = len(step_ids)
        if n == 0:
            return []
        k = POWER_INDEX[power]
        obs = torch.from_numpy(encode_observation(board, power)).unsqueeze(0).to(self.device)
        masks = torch.zeros(1, MAX_STEPS, VOCAB_SIZE, dtype=torch.bool, device=self.device)
        for t, ids in enumerate(step_ids):
            masks[0, t, torch.from_numpy(to_own_frame_ids(ids, k))] = True
        gen = None
        if not self.greedy:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(rng.integers(2**63)))
        ids, _, _ = self.model.act(
            obs,
            masks,
            torch.tensor([n], device=self.device),
            greedy=self.greedy,
            generator=gen,
            exclude_emitted=exclude_emitted,
            repeat_ok=repeat_ok,
        )
        return [int(VOCAB_PERM[k][i]) for i in ids[0, :n].tolist()]  # own -> real

    def movement_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> dict[str, Order]:
        provs = own_frame_unit_order(board.units, power)  # seat-equivariant order
        step_ids = [legal_movement_orders(board.units, p) for p in provs]
        chosen = self._decode(board, power, step_ids, rng, exclude_emitted=False)
        return {ORDERS[i].src: ORDERS[i] for i in chosen}

    def winter_orders(
        self, board: Board, power: str, rng: np.random.Generator
    ) -> list[Order]:
        delta = board.sc_count(power) - board.unit_count(power)
        if delta > 0:
            from triad.engine.orders import BUILD_IDS
            legal = [
                i
                for i, h in zip(BUILD_IDS[power], HOME_CENTERS[power])
                if board.sc_owner[h] == power and h not in board.units
            ]
            step_ids = [legal + [WAIVE_ID]] * delta
        elif delta < 0:
            from triad.engine.orders import DISBAND_ID
            step_ids = [[DISBAND_ID[p] for p in board.unit_provinces(power)]] * (-delta)
        else:
            return []
        chosen = self._decode(
            board, power, step_ids, rng, exclude_emitted=True, repeat_ok=WAIVE_ID
        )
        return [ORDERS[i] for i in chosen if ORDERS[i].kind != WAIVE]
