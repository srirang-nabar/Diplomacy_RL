"""Metric-definition tests on constructed positions (CLAUDE.md §5).

The chance-corrected attack-the-leader index is a headline number; its math
is verified here against hand-worked examples, not just exercised.
"""
from __future__ import annotations

import numpy as np

from triad.map_data import SUPPLY_CENTERS
from triad.engine.orders import Order
from triad.eval.metrics import (
    Trace,
    attack_the_leader,
    cross_support,
    lead_conversion,
)


def _sc_owner(**overrides) -> dict:
    base = {sc: None for sc in SUPPLY_CENTERS}
    base.update(
        {"CAP_A": "A", "L_A": "A", "R_A": "A",
         "CAP_B": "B", "L_B": "B", "R_B": "B",
         "CAP_C": "C", "L_C": "C", "R_C": "C"}
    )
    base.update(overrides)
    return base


def test_attack_the_leader_chance_correction_hand_worked():
    # Leader A: 5 SCs (homes + B_CA + B_BC); C: 4 (homes + B_AB); B: 3
    # -> strict leader A, lead k = 5 - 4 = 1. B_BC is not adjacent to L_B,
    # so the probed unit's neighbourhood is unaffected.
    sc = _sc_owner(B_CA="A", B_BC="A", B_AB="C")
    units = {"L_B": "B"}  # B's only unit
    # L_B adj = CAP_B(own SC), GATE_B(non-SC, empty), B_AB(SC owned C), R_A(SC owned A)
    # attackable = {B_AB, R_A}; leader's share = 1/2 -> chance 0.5
    orders = {"B": {"L_B": Order("M", "L_B", None, "R_A")}}  # targets the leader
    tr = Trace(phases=[(units, sc, orders)], max_lead={"A": 0, "B": 0, "C": 0})
    out = attack_the_leader([tr])
    assert list(out) == [1]
    assert out[1]["observed"] == 1.0
    assert out[1]["chance"] == 0.5
    assert out[1]["excess"] == 0.5
    assert out[1]["n_unit_phases"] == 1


def test_attack_the_leader_skips_units_with_no_targets_and_tied_leaders():
    # unit whose neighbourhood holds nothing foreign -> excluded
    sc = _sc_owner(B_CA="A", B_AB="C")
    units = {"CAP_B": "B"}  # adj: L_B/R_B own SCs, GATE_B non-SC empty
    orders = {"B": {"CAP_B": Order("H", "CAP_B")}}
    tr = Trace(phases=[(units, sc, orders)], max_lead={"A": 0, "B": 0, "C": 0})
    assert attack_the_leader([tr]) == {}
    # tied leaders -> phase skipped entirely
    sc_tied = _sc_owner()  # 3/3/3
    tr2 = Trace(phases=[({"L_B": "B"}, sc_tied, {"B": {}})], max_lead={})
    assert attack_the_leader([tr2]) == {}


def test_attack_the_leader_support_move_counts_hold_does_not():
    sc = _sc_owner(B_CA="A", B_BC="A", B_AB="C")
    units = {"L_B": "B", "GATE_B": "B", "R_A": "A"}
    # GATE_B support-moves L_B -> ...? GATE_B adj: CAP_B, L_B, R_B, CTR; R_A not
    # adjacent to GATE_B so a support INTO R_A from GATE_B is illegal anyway —
    # use L_B itself: MOVE targets leader; GATE_B HOLD contributes chance only.
    orders = {"B": {
        "L_B": Order("SM", "L_B", "GATE_B", "B_AB"),  # support into C's SC: NOT leader
        "GATE_B": Order("H", "GATE_B"),
    }}
    tr = Trace(phases=[(units, sc, orders)], max_lead={"A": 0, "B": 0, "C": 0})
    out = attack_the_leader([tr])
    # L_B: attackable {B_AB(C), R_A(A, occupied+owned)} -> chance .5, obs 0 (dst=B_AB not leader's)
    # GATE_B: attackable {} (CTR non-SC empty, others own) -> excluded
    assert out[1]["n_unit_phases"] == 1
    assert out[1]["observed"] == 0.0
    assert out[1]["chance"] == 0.5
    assert out[1]["excess"] == -0.5


def test_lead_conversion_buckets():
    trs = []
    for lead, win in [(2, True), (2, False), (2, False), (3, True), (0, False)]:
        trs.append(Trace(max_lead={"A": lead, "B": 0, "C": 0},
                         winner="A" if win else None))
    out = lead_conversion(trs)
    assert out[2] == {"p_solo": round(1 / 3, 4), "n": 3}
    assert out[3] == {"p_solo": 1.0, "n": 1}
    assert 0 not in out  # k <= 0 excluded


def test_cross_support_classification():
    sc = _sc_owner(B_CA="A", B_AB="A")  # A leads 5 vs 3/3
    units = {"L_B": "B", "R_A": "A", "L_C": "C", "R_B": "B", "B_BC": "C"}
    orders = {
        "B": {
            # cross support of C's unit moving into A's owned SC B_AB?
            # L_B adj B_AB; supported unit at B_BC?? B_AB not adj B_BC — use
            # a legal shape: L_B supports C's unit at B_BC?? not adjacent...
            # Simplest legal cross-support: R_B S B_BC (support-hold C's unit;
            # R_B adj B_BC), with leader unit adjacent to B_BC? R_A adj B_BC? no.
            "R_B": Order("SH", "R_B", "B_BC"),          # cross, not vs leader
            "L_B": Order("SH", "L_B", "R_A"),           # supporting the LEADER: neither
        },
        "C": {
            "L_C": Order("SH", "L_C", "B_BC"),          # own support (C's unit)
            "B_BC": Order("H", "B_BC"),
        },
    }
    tr = Trace(phases=[(units, sc, orders)], max_lead={"A": 0, "B": 0, "C": 0})
    out = cross_support([tr])
    assert out["n_trailing_orders"] == 4
    assert out["own_support_rate"] == 0.25          # L_C S B_BC
    assert out["cross_support_rate"] == 0.25        # R_B S B_BC (C's unit)
    assert out["cross_support_vs_leader_rate"] == 0.0


def test_cross_support_vs_leader_detected():
    # C's unit at B_BC attacks leader A's R_A?? not adjacent; use support-move:
    # B's R_B supports C's unit at B_BC moving into ... B_BC adj: R_B, L_C, CTR.
    # Make CTR leader-occupied: A unit at CTR; C at B_BC moves CTR, B's R_B?? not
    # adj CTR. L_C adj B_BC? yes but belongs C. Use SUP_HOLD defensive form:
    # B supports C's unit at B_BC while a LEADER unit is adjacent (CTR).
    sc = _sc_owner(B_CA="A", B_AB="A")
    units = {"R_B": "B", "B_BC": "C", "CTR": "A"}
    orders = {"B": {"R_B": Order("SH", "R_B", "B_BC")},
              "C": {"B_BC": Order("H", "B_BC")}}
    tr = Trace(phases=[(units, sc, orders)], max_lead={"A": 0, "B": 0, "C": 0})
    out = cross_support([tr])
    assert out["cross_support_rate"] == 0.5          # 1 of B+C's 2 orders
    assert out["cross_support_vs_leader_rate"] == 0.5  # defensive, leader adjacent
