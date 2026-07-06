"""Order vocabulary for Triad: fixed, enumerable, index-stable (CLAUDE.md §3.3).

336 orders total:
    16 HOLD + 60 MOVE + 60 SUPPORT-HOLD + 174 SUPPORT-MOVE
    + 9 BUILD + 16 DISBAND + 1 WAIVE

Enumeration order is canonical (province order from map_data, adjacency-list
order within a province) and must never change once training data exists.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple

from triad.map_data import ADJACENCY, HOME_CENTERS, POWERS, PROVINCES, ROTATION

# Order kinds.
HOLD = "H"
MOVE = "M"
SUP_HOLD = "SH"
SUP_MOVE = "SM"
BUILD = "B"
DISBAND = "D"
WAIVE = "W"
MOVEMENT_KINDS = frozenset({HOLD, MOVE, SUP_HOLD, SUP_MOVE})
ADJUSTMENT_KINDS = frozenset({BUILD, DISBAND, WAIVE})


class Order(NamedTuple):
    """One order. Field use per kind:

    HOLD      src=unit location
    MOVE      src -> dst
    SUP_HOLD  src supports the unit in aux to stay
    SUP_MOVE  src supports the move aux -> dst
    BUILD     src=home SC to build in
    DISBAND   src=unit location to remove
    WAIVE     (no fields)
    """

    kind: str
    src: str | None = None
    aux: str | None = None
    dst: str | None = None


def _enumerate_vocab() -> list[Order]:
    orders = [Order(HOLD, p) for p in PROVINCES]
    orders += [Order(MOVE, s, None, d) for s in PROVINCES for d in ADJACENCY[s]]
    orders += [Order(SUP_HOLD, s, t) for s in PROVINCES for t in ADJACENCY[s]]
    orders += [
        Order(SUP_MOVE, s, u, d)
        for s in PROVINCES
        for d in ADJACENCY[s]
        for u in ADJACENCY[d]
        if u != s
    ]
    orders += [Order(BUILD, h) for pw in POWERS for h in HOME_CENTERS[pw]]
    orders += [Order(DISBAND, p) for p in PROVINCES]
    orders.append(Order(WAIVE))
    return orders


ORDERS: list[Order] = _enumerate_vocab()
ORDER_INDEX: dict[Order, int] = {o: i for i, o in enumerate(ORDERS)}
VOCAB_SIZE: int = len(ORDERS)  # 336

WAIVE_ORDER = Order(WAIVE)
WAIVE_ID = ORDER_INDEX[WAIVE_ORDER]
BUILD_IDS: dict[str, list[int]] = {
    pw: [ORDER_INDEX[Order(BUILD, h)] for h in HOME_CENTERS[pw]] for pw in POWERS
}
DISBAND_ID: dict[str, int] = {p: ORDER_INDEX[Order(DISBAND, p)] for p in PROVINCES}
HOLD_ID: dict[str, int] = {p: ORDER_INDEX[Order(HOLD, p)] for p in PROVINCES}


# --- static legality tables (movement phase) --------------------------------
# For a unit at province p:
#   BASE_MOVEMENT[p]        : always-legal ids (HOLD + all MOVEs out of p)
#   CONDITIONAL_MOVEMENT[p] : q -> ids legal iff a unit currently occupies q
#                             (SUP_HOLD with target q, SUP_MOVE of the unit at q)
def _build_static_tables() -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    base: dict[str, list[int]] = {}
    cond: dict[str, dict[str, list[int]]] = {}
    for p in PROVINCES:
        ids = [HOLD_ID[p]]
        ids += [ORDER_INDEX[Order(MOVE, p, None, d)] for d in ADJACENCY[p]]
        base[p] = ids
        cnd: dict[str, list[int]] = {}
        for t in ADJACENCY[p]:  # support-hold the unit in t
            cnd.setdefault(t, []).append(ORDER_INDEX[Order(SUP_HOLD, p, t)])
        for d in ADJACENCY[p]:  # support a move u -> d
            for u in ADJACENCY[d]:
                if u != p:
                    cnd.setdefault(u, []).append(ORDER_INDEX[Order(SUP_MOVE, p, u, d)])
        cond[p] = cnd
    return base, cond


BASE_MOVEMENT, CONDITIONAL_MOVEMENT = _build_static_tables()

#: All movement-kind order ids whose src is p (ignores occupancy) — the static
#: boolean mask legal[province, order_id] in list form.
STATIC_MOVEMENT: dict[str, list[int]] = {
    p: BASE_MOVEMENT[p] + [i for ids in CONDITIONAL_MOVEMENT[p].values() for i in ids]
    for p in PROVINCES
}


def legal_movement_orders(units: dict[str, str], p: str) -> list[int]:
    """Dynamically legal movement order ids for the unit at p.

    Supports are legal only if the supported province is currently occupied
    (by any power — cross-power support is allowed).
    """
    out = list(BASE_MOVEMENT[p])
    for q, ids in CONDITIONAL_MOVEMENT[p].items():
        if q in units:
            out.extend(ids)
    return out


# --- text form ---------------------------------------------------------------
def format_order(o: Order) -> str:
    if o.kind == HOLD:
        return f"A {o.src} H"
    if o.kind == MOVE:
        return f"A {o.src} - {o.dst}"
    if o.kind == SUP_HOLD:
        return f"A {o.src} S {o.aux}"
    if o.kind == SUP_MOVE:
        return f"A {o.src} S {o.aux} - {o.dst}"
    if o.kind == BUILD:
        return f"BUILD {o.src}"
    if o.kind == DISBAND:
        return f"DISBAND {o.src}"
    if o.kind == WAIVE:
        return "WAIVE"
    raise ValueError(f"unknown order kind {o.kind!r}")


def parse_order(text: str) -> Order:
    """Inverse of format_order. Raises ValueError on malformed text."""
    toks = text.split()
    try:
        if toks == ["WAIVE"]:
            return Order(WAIVE)
        if toks[0] == "BUILD" and len(toks) == 2:
            return Order(BUILD, toks[1])
        if toks[0] == "DISBAND" and len(toks) == 2:
            return Order(DISBAND, toks[1])
        if toks[0] == "A":
            if len(toks) == 3 and toks[2] == "H":
                return Order(HOLD, toks[1])
            if len(toks) == 4 and toks[2] == "-":
                return Order(MOVE, toks[1], None, toks[3])
            if len(toks) == 4 and toks[2] == "S":
                return Order(SUP_HOLD, toks[1], toks[3])
            if len(toks) == 6 and toks[2] == "S" and toks[4] == "-":
                return Order(SUP_MOVE, toks[1], toks[3], toks[5])
    except IndexError:
        pass
    raise ValueError(f"cannot parse order: {text!r}")


# --- rotation ----------------------------------------------------------------
def rotate_order(o: Order) -> Order:
    """Apply the C3 map rotation to every province in the order."""
    rot = lambda p: ROTATION[p] if p is not None else None  # noqa: E731
    return Order(o.kind, rot(o.src), rot(o.aux), rot(o.dst))


def rotate_orders(orders: Iterable[Order]) -> list[Order]:
    return [rotate_order(o) for o in orders]


# --- own-frame action interface (the vocab permutation) -----------------------
# The policy operates ENTIRELY in the acting power's own frame: observations
# are rotated (env/obs.py) and so must the ACTION space be — otherwise the
# same relational decision has a different order id per seat, the legality
# masks become seat-dependent for identical observations, and exact weight
# sharing is silently broken (the second equivariance bug, caught in M4 by
# the seat chi^2 + a trajectory-equivariance experiment).
#
#   VOCAB_PERM[k][own_id]  = real order id   (apply rho^k to the own-frame order)
#   VOCAB_PERM_INV[k][real_id] = own_id
#
# k is the acting power's index; k=0 is the identity. WAIVE is a fixed point.
def _build_vocab_perms() -> list[list[int]]:
    perms = []
    for k in range(3):
        perm = []
        for o in ORDERS:
            r = o
            for _ in range(k):
                r = rotate_order(r)
            perm.append(ORDER_INDEX[r])
        perms.append(perm)
    return perms


import numpy as _np  # noqa: E402

VOCAB_PERM: list[_np.ndarray] = [_np.array(p, dtype=_np.int64) for p in _build_vocab_perms()]
VOCAB_PERM_INV: list[_np.ndarray] = []
for _perm in VOCAB_PERM:
    _inv = _np.empty_like(_perm)
    _inv[_perm] = _np.arange(len(_perm))
    VOCAB_PERM_INV.append(_inv)


def to_own_frame_ids(real_ids: list[int] | _np.ndarray, power_index: int) -> _np.ndarray:
    """Real-frame order ids -> own-frame ids for the acting power."""
    return VOCAB_PERM_INV[power_index][_np.asarray(real_ids, dtype=_np.int64)]


def to_real_order(own_id: int, power_index: int) -> Order:
    """Own-frame order id -> the real Order to submit to the engine."""
    return ORDERS[int(VOCAB_PERM[power_index][own_id])]
