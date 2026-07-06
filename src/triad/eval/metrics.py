"""Balance-of-power metrics (CLAUDE.md §5) — the headline analyses.

All three metrics come from TRACED games (per-phase board + orders):

1. **Attack-the-leader index, chance-corrected.** The leader owns more of the
   board by definition, so a score-blind policy mechanically targets it more.
   We therefore report the EXCESS: observed targeting rate minus the
   target-blind chance rate (the leader's share of each unit's adjacent
   attackable objects). Positive excess growing with the lead = active
   balancing; the raw rate alone proves nothing.

2. **Lead-conversion curve.** P(eventual solo | power reached max SC lead k).
   If trailing powers coalesce, big leads stop converting.

3. **Cross-power support rate.** Supports given by a trailing power to the
   OTHER trailing power's units (the mechanistic coalition signal), split by
   whether the supported action is directed at the leader.

Definitions (fixed here, cited in the report):
- leader at a movement phase: strict argmax of SC count among alive powers
  (ties -> no leader, phase skipped).
- attackable object for a unit at p, from seat pw: an adjacent province q
  that is occupied by a foreign unit OR is an SC owned by a foreign power.
- an order "targets X" if it is MOVE with dst in X's objects or SUPPORT-MOVE
  with dst in X's objects.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from triad.map_data import ADJACENCY, POWERS, SUPPLY_CENTERS
from triad.bots import Bot
from triad.engine.game import Game
from triad.engine.orders import MOVE, SUP_HOLD, SUP_MOVE, Order
from triad.engine.state import Board, FALL, SPRING

_SC_SET = frozenset(SUPPLY_CENTERS)


# --- traced self-play ------------------------------------------------------------
@dataclass
class Trace:
    phases: list[tuple[dict, dict, dict]] = field(default_factory=list)
    # each: (units, sc_owner, orders_by_power {pw: {prov: Order}})
    result: dict | None = None
    max_lead: dict[str, int] = field(default_factory=dict)  # per power
    winner: str | None = None


def play_traced_game(bots: dict[str, Bot], rng: np.random.Generator) -> Trace:
    g = Game()
    tr = Trace(max_lead={pw: 0 for pw in POWERS})
    while not g.over:
        if g.board.phase in (SPRING, FALL):
            orders = {
                pw: bots[pw].movement_orders(g.board, pw, rng)
                for pw in g.alive_powers()
            }
            tr.phases.append(
                (dict(g.board.units), dict(g.board.sc_owner), orders)
            )
            g.step_movement(orders)
        else:
            g.step_winter(
                {pw: bots[pw].winter_orders(g.board, pw, rng) for pw in g.alive_powers()}
            )
            for pw in POWERS:  # track max lead at each Winter
                mine = g.board.sc_count(pw)
                best_other = max(g.board.sc_count(q) for q in POWERS if q != pw)
                tr.max_lead[pw] = max(tr.max_lead[pw], mine - best_other)
    tr.result = g.result
    tr.winner = g.result.get("winner")
    return tr


# --- helpers -----------------------------------------------------------------------
def _leader(units: dict, sc_owner: dict) -> str | None:
    counts = {pw: 0 for pw in POWERS}
    for sc, pw in sc_owner.items():
        if pw is not None:
            counts[pw] += 1
    best = max(counts.values())
    tops = [pw for pw, c in counts.items() if c == best]
    return tops[0] if len(tops) == 1 else None


def _objects_of(power: str, units: dict, sc_owner: dict) -> set[str]:
    """Provinces that 'belong to' power: its units' locations + owned SCs."""
    out = {p for p, pw in units.items() if pw == power}
    out |= {sc for sc, pw in sc_owner.items() if pw == power}
    return out


def _attackable(unit_prov: str, my_power: str, units: dict, sc_owner: dict) -> set[str]:
    out = set()
    for q in ADJACENCY[unit_prov]:
        occ = units.get(q)
        if occ is not None and occ != my_power:
            out.add(q)
        elif q in _SC_SET and sc_owner.get(q) not in (None, my_power):
            out.add(q)
    return out


def _order_target(o: Order) -> str | None:
    """The province an order is directed into (attack semantics)."""
    if o.kind == MOVE or o.kind == SUP_MOVE:
        return o.dst
    return None


# --- metric 1: chance-corrected attack-the-leader ------------------------------------
def attack_the_leader(traces: list[Trace]) -> dict[int, dict]:
    """Per lead size k: observed targeting rate, chance rate, excess, n units.

    Aggregated over both trailing powers' units at every phase with a strict
    leader. Only units with at least one attackable object count.
    """
    acc: dict[int, dict] = defaultdict(lambda: {"obs": 0.0, "chance": 0.0, "n": 0})
    for tr in traces:
        for units, sc_owner, orders in tr.phases:
            ldr = _leader(units, sc_owner)
            if ldr is None:
                continue
            counts = {pw: sum(1 for _, o in sc_owner.items() if o == pw) for pw in POWERS}
            k = counts[ldr] - max(c for pw, c in counts.items() if pw != ldr)
            leader_objs = _objects_of(ldr, units, sc_owner)
            for pw in POWERS:
                if pw == ldr or pw not in orders:
                    continue
                for prov, o in orders[pw].items():
                    att = _attackable(prov, pw, units, sc_owner)
                    if not att:
                        continue
                    ldr_att = att & leader_objs
                    a = acc[k]
                    a["n"] += 1
                    a["chance"] += len(ldr_att) / len(att)
                    tgt = _order_target(o)
                    if tgt is not None and tgt in ldr_att:
                        a["obs"] += 1.0
    out = {}
    for k, a in sorted(acc.items()):
        n = a["n"]
        if n == 0:
            continue
        obs, chance = a["obs"] / n, a["chance"] / n
        out[k] = {
            "observed": round(obs, 4),
            "chance": round(chance, 4),
            "excess": round(obs - chance, 4),
            "n_unit_phases": n,
        }
    return out


# --- metric 2: lead-conversion --------------------------------------------------------
def lead_conversion(traces: list[Trace]) -> dict[int, dict]:
    """Per max-lead k reached: P(that power eventually wins solo)."""
    acc: dict[int, dict] = defaultdict(lambda: {"n": 0, "won": 0})
    for tr in traces:
        for pw in POWERS:
            k = tr.max_lead[pw]
            if k <= 0:
                continue
            acc[k]["n"] += 1
            if tr.winner == pw:
                acc[k]["won"] += 1
    return {
        k: {"p_solo": round(a["won"] / a["n"], 4), "n": a["n"]}
        for k, a in sorted(acc.items())
        if a["n"] > 0
    }


# --- metric 3: cross-power supports ---------------------------------------------------
def cross_support(traces: list[Trace]) -> dict:
    """Among trailing powers' orders (phases with a strict leader):
    - own_support: supports of own units
    - cross_support: supports of the OTHER trailing power's units
    - cross_support_vs_leader: those whose supported action is directed at
      the leader's objects (SUPPORT-MOVE into leader objects), or
      SUPPORT-HOLD of the other trailer's unit while the leader is adjacent
      to it (defensive coalition)."""
    n_orders = own = cross = cross_vs_leader = 0
    for tr in traces:
        for units, sc_owner, orders in tr.phases:
            ldr = _leader(units, sc_owner)
            if ldr is None:
                continue
            leader_objs = _objects_of(ldr, units, sc_owner)
            trailers = [pw for pw in POWERS if pw != ldr]
            for pw in trailers:
                if pw not in orders:
                    continue
                other = trailers[0] if trailers[1] == pw else trailers[1]
                for prov, o in orders[pw].items():
                    n_orders += 1
                    if o.kind not in (SUP_HOLD, SUP_MOVE):
                        continue
                    supported_owner = units.get(o.aux)
                    if supported_owner == pw:
                        own += 1
                    elif supported_owner == other:
                        cross += 1
                        if o.kind == SUP_MOVE and o.dst in leader_objs:
                            cross_vs_leader += 1
                        elif o.kind == SUP_HOLD and any(
                            units.get(q) == ldr for q in ADJACENCY[o.aux]
                        ):
                            cross_vs_leader += 1
    return {
        "n_trailing_orders": n_orders,
        "own_support_rate": round(own / n_orders, 5) if n_orders else 0.0,
        "cross_support_rate": round(cross / n_orders, 5) if n_orders else 0.0,
        "cross_support_vs_leader_rate": (
            round(cross_vs_leader / n_orders, 5) if n_orders else 0.0
        ),
    }


def run_traced_selfplay(
    bot_factory, n_games: int, seed: int
) -> list[Trace]:
    """n_games of 3x-identical-policy self-play with traces."""
    rng = np.random.default_rng(seed)
    bots = {pw: bot_factory() if callable(bot_factory) else bot_factory for pw in POWERS}
    return [play_traced_game(bots, rng) for _ in range(n_games)]
