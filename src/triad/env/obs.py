"""Rotation-to-own-frame observation builder (CLAUDE.md §4.1).

Before a power acts, the board is rotated so the acting power always sees
itself as "power A": observation slot for province p describes the real
province rho^k(p), where k is the acting power's index and rho is the map's
C3 automorphism. Real powers are encoded relationally as {self, next, prev}.
One shared network can then play all three seats with exact weight sharing
and no power-identity input.

Per-province features (16 x 12):
    0-3   unit owner one-hot   {none, self, next, prev}
    4-7   SC owner one-hot     {not-an-SC / unowned, self, next, prev}
    8     is_SC flag
    9-11  is home center of    {self, next, prev}

Globals (11):
    0-1   season one-hot (spring, fall)     [0,0 in Winter]
    2-3   phase one-hot  (movement, adjustment)
    4     year / 20
    5-7   SC counts   {self, next, prev} / 12
    8-10  unit counts {self, next, prev} / 12
"""
from __future__ import annotations

import numpy as np

from triad.map_data import (
    HOME_CENTERS,
    MAX_MOVEMENT_PHASES,
    POWERS,
    PROVINCES,
    ROTATION,
    SUPPLY_CENTERS,
)
from triad.engine.state import Board, FALL, SPRING, WINTER

N_PROVINCES = len(PROVINCES)          # 16
N_PROV_FEATURES = 12
N_GLOBALS = 11
OBS_DIM = N_PROVINCES * N_PROV_FEATURES + N_GLOBALS  # 203

POWER_INDEX = {pw: i for i, pw in enumerate(POWERS)}
_SC_SET = frozenset(SUPPLY_CENTERS)
N_YEARS = MAX_MOVEMENT_PHASES // 2    # 20


def _rho_k(p: str, k: int) -> str:
    for _ in range(k % 3):
        p = ROTATION[p]
    return p


#: SLOT_REAL[k][i] = the real province whose features fill slot i when power
#: k acts (slot i is canonical province PROVINCES[i] in the rotated frame).
SLOT_REAL: list[list[str]] = [[_rho_k(p, k) for p in PROVINCES] for k in range(3)]

# Static per-slot features: SC-ness and home-of-relative-power flags are
# invariant under rotation, so they are the same for every seat.
_IS_SC = np.array([1.0 if p in _SC_SET else 0.0 for p in PROVINCES], dtype=np.float32)
_HOME_REL = np.zeros((N_PROVINCES, 3), dtype=np.float32)
for r in range(3):
    for h in HOME_CENTERS[POWERS[r]]:
        _HOME_REL[PROVINCES.index(h), r] = 1.0


def encode_observation(board: Board, power: str) -> np.ndarray:
    """Flat float32 observation of length OBS_DIM, in `power`'s own frame."""
    k = POWER_INDEX[power]
    slots = SLOT_REAL[k]

    def rel(q: str) -> int:  # real power -> 0 self / 1 next / 2 prev
        return (POWER_INDEX[q] - k) % 3

    feat = np.zeros((N_PROVINCES, N_PROV_FEATURES), dtype=np.float32)
    feat[:, 8] = _IS_SC
    feat[:, 9:12] = _HOME_REL
    for i, rp in enumerate(slots):
        u = board.units.get(rp)
        feat[i, 0 if u is None else 1 + rel(u)] = 1.0
        owner = board.sc_owner.get(rp) if rp in _SC_SET else None
        feat[i, 4 if owner is None else 5 + rel(owner)] = 1.0

    g = np.zeros(N_GLOBALS, dtype=np.float32)
    if board.phase == SPRING:
        g[0] = 1.0
    elif board.phase == FALL:
        g[1] = 1.0
    g[2] = 0.0 if board.phase == WINTER else 1.0  # movement
    g[3] = 1.0 if board.phase == WINTER else 0.0  # adjustment
    g[4] = board.year / N_YEARS
    for r in range(3):
        q = POWERS[(k + r) % 3]  # relative r -> real power
        g[5 + r] = board.sc_count(q) / 12.0
        g[8 + r] = board.unit_count(q) / 12.0
    return np.concatenate([feat.reshape(-1), g])
