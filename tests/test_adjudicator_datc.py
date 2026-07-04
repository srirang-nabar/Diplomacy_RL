"""DATC-adapted adjudicator tests + property tests (CLAUDE.md §3.4).

DATC = Diplomacy Adjudicator Test Cases (Kruijswijk). Only armies-only cases
from sections 6.A (basic), 6.C (circular), 6.D (supports/dislodges) and
6.E (head-to-head) apply — fleet/coast/convoy/retreat cases are skipped by
design. Case mapping:

    test name                                   DATC id (adapted)
    ------------------------------------------  -----------------
    test_move_to_non_adjacent_coerced_to_hold    6.A.1
    test_move_to_own_province_coerced_to_hold    6.A.4
    test_order_for_foreign_unit_ignored          6.A.6
    test_support_self_hold_impossible            6.A.8
    test_simple_bounce                           6.A.11
    test_three_way_bounce                        6.A.12
    test_circular_movement                       6.C.1
    test_circular_movement_with_support          6.C.2
    test_disrupted_circular_movement             6.C.3
    test_supported_hold_prevents_dislodgement    6.D.1
    test_move_cuts_support_on_hold               6.D.2
    test_move_cuts_support_on_move               6.D.3
    test_support_to_hold_on_supporting_unit      6.D.4/6.D.5
    test_support_to_hold_on_moving_unit_void     6.D.7 (spirit)
    test_support_to_move_on_holding_unit_void    6.D.9
    test_no_self_dislodgement                    6.D.10
    test_no_self_dislodgement_returning_unit     6.D.11
    test_own_support_cannot_dislodge_own_unit    6.D.12
    test_foreign_support_no_protection           6.D.14
    test_defender_cannot_cut_support_on_itself   6.D.15
    test_dislodged_no_effect_on_attackers_area   6.E.1
    test_no_self_dislodgement_head_to_head       6.E.2
    test_head_to_head_tie_both_fail              (6.E, tie case)
    test_head_to_head_loser_no_prevent           6.E.1 (prevent side)
    test_non_dislodged_loser_still_prevents      6.E.4
    test_beleaguered_garrison                    (classic garrison)
    test_support_of_empty_province_coerced       (legalize rule)
    test_retroactive_support_void                §3.4 step 6
"""
from __future__ import annotations

import numpy as np
import pytest

from triad.map_data import PROVINCES, POWERS, ROTATION, SUPPLY_CENTERS
from triad.engine.adjudicator import BOUNCED, CUT, OK, VOID, resolve
from triad.engine.orders import (
    MOVEMENT_KINDS,
    ORDERS,
    Order,
    legal_movement_orders,
    parse_order,
    rotate_order,
)
from triad.engine.state import Board


# --- helpers ---------------------------------------------------------------------
def board(units: dict[str, str]) -> Board:
    b = Board.initial()
    b.units = dict(units)
    return b


def orders(*texts: str) -> dict[str, Order]:
    """Parse 'A R_A - B_AB' style strings into {unit_province: Order}."""
    out = {}
    for t in texts:
        o = parse_order(t)
        out[o.src] = o
    return out


# --- 6.A basic --------------------------------------------------------------------
def test_move_to_non_adjacent_coerced_to_hold():  # 6.A.1
    b = board({"CAP_A": "A"})
    nb, dis, res = resolve(b, {"CAP_A": Order("M", "CAP_A", None, "CTR")})
    assert nb.units == {"CAP_A": "A"} and not dis
    assert res["CAP_A"] == VOID  # coerced


def test_move_to_own_province_coerced_to_hold():  # 6.A.4
    b = board({"CAP_A": "A"})
    nb, dis, _ = resolve(b, {"CAP_A": Order("M", "CAP_A", None, "CAP_A")})
    assert nb.units == {"CAP_A": "A"} and not dis


def test_order_for_foreign_unit_ignored():  # 6.A.6
    # game.py enforces ownership on merge; at the adjudicator level an order
    # keyed to a province whose order.src mismatches is coerced to HOLD.
    b = board({"B_AB": "B"})
    nb, dis, _ = resolve(b, {"B_AB": Order("M", "R_A", None, "CAP_A")})
    assert nb.units == {"B_AB": "B"} and not dis


def test_support_self_hold_impossible():  # 6.A.8
    b = board({"R_A": "A"})
    nb, _, res = resolve(b, {"R_A": Order("SH", "R_A", "R_A")})
    assert nb.units == {"R_A": "A"}
    assert res["R_A"] == VOID  # coerced to hold


def test_simple_bounce():  # 6.A.11
    b = board({"R_A": "A", "L_B": "B"})
    nb, dis, res = resolve(b, orders("A R_A - B_AB", "A L_B - B_AB"))
    assert nb.units == {"R_A": "A", "L_B": "B"} and not dis
    assert res["R_A"] == BOUNCED and res["L_B"] == BOUNCED
    assert "B_AB" not in nb.units  # standoff over empty province stays empty


def test_three_way_bounce():  # 6.A.12
    b = board({"GATE_A": "A", "GATE_B": "B", "GATE_C": "C"})
    nb, dis, _ = resolve(
        b, orders("A GATE_A - CTR", "A GATE_B - CTR", "A GATE_C - CTR")
    )
    assert nb.units == {"GATE_A": "A", "GATE_B": "B", "GATE_C": "C"} and not dis


# --- 6.C circular movement -----------------------------------------------------
def test_circular_movement():  # 6.C.1
    b = board({"CAP_A": "A", "L_A": "A", "GATE_A": "A"})
    nb, dis, res = resolve(
        b, orders("A CAP_A - L_A", "A L_A - GATE_A", "A GATE_A - CAP_A")
    )
    assert nb.units == {"L_A": "A", "GATE_A": "A", "CAP_A": "A"} and not dis
    assert all(res[p] == OK for p in ("CAP_A", "L_A", "GATE_A"))


def test_circular_movement_with_support():  # 6.C.2
    b = board({"CAP_A": "A", "L_A": "A", "GATE_A": "A", "R_A": "A"})
    nb, dis, _ = resolve(
        b,
        orders(
            "A CAP_A - L_A",
            "A L_A - GATE_A",
            "A GATE_A - CAP_A",
            "A R_A S CAP_A - L_A",
        ),
    )
    assert not dis
    assert nb.units["L_A"] == "A" and nb.units["GATE_A"] == "A" and nb.units["CAP_A"] == "A"


def test_disrupted_circular_movement():  # 6.C.3
    # C attacks L_A with strength 2; the cycle move into L_A is prevented,
    # the whole cycle fails, and the unit standing in L_A is dislodged.
    b = board(
        {"CAP_A": "A", "L_A": "A", "GATE_A": "A", "R_C": "C", "B_CA": "C"}
    )
    nb, dis, res = resolve(
        b,
        orders(
            "A CAP_A - L_A",
            "A L_A - GATE_A",
            "A GATE_A - CAP_A",
            "A R_C - L_A",
            "A B_CA S R_C - L_A",
        ),
    )
    assert dis == {"L_A"}
    assert res["CAP_A"] == BOUNCED and res["L_A"] == BOUNCED and res["GATE_A"] == BOUNCED
    assert nb.units["L_A"] == "C" and nb.units["CAP_A"] == "A" and nb.units["GATE_A"] == "A"


# --- 6.D supports and dislodges ---------------------------------------------------
def test_supported_hold_prevents_dislodgement():  # 6.D.1
    b = board({"R_A": "A", "CTR": "A", "B_AB": "B", "L_B": "B"})
    nb, dis, _ = resolve(
        b, orders("A R_A - B_AB", "A CTR S R_A - B_AB", "A B_AB H", "A L_B S B_AB")
    )
    assert not dis and nb.units["B_AB"] == "B"  # 2 vs 2: attack must EXCEED


def test_move_cuts_support_on_hold():  # 6.D.2
    # same as 6.D.1 but A also attacks the supporter L_B -> support cut -> dislodge
    b = board({"R_A": "A", "CTR": "A", "GATE_B": "A", "B_AB": "B", "L_B": "B"})
    nb, dis, res = resolve(
        b,
        orders(
            "A R_A - B_AB",
            "A CTR S R_A - B_AB",
            "A GATE_B - L_B",
            "A B_AB H",
            "A L_B S B_AB",
        ),
    )
    assert res["L_B"] == CUT
    assert dis == {"B_AB"}
    assert nb.units["B_AB"] == "A"


def test_move_cuts_support_on_move():  # 6.D.3
    # B supports its own attack on R_A; a THIRD A unit cuts the supporter.
    # (The defender R_A itself could not cut — its attack would come out of
    # the province the support is directed into, see 6.D.15.)
    b = board({"B_AB": "B", "L_B": "B", "R_A": "A", "GATE_B": "A"})
    nb, dis, res = resolve(
        b,
        orders(
            "A B_AB - R_A",
            "A L_B S B_AB - R_A",
            "A GATE_B - L_B",
            "A R_A H",
        ),
    )
    assert res["L_B"] == CUT
    assert not dis and nb.units["R_A"] == "A"  # attack 1 vs hold 1: fails


def test_support_to_hold_on_supporting_unit():  # 6.D.4 / 6.D.5
    # a unit giving support can itself receive hold-support
    b = board({"B_AB": "B", "L_B": "B", "R_B": "B", "R_A": "A", "CTR": "A"})
    # A attacks L_B?? -> not adjacent from CTR; instead attack B_AB (supported hold chain):
    # B: L_B S B_AB (holds B_AB), R_B... keep simple: A attacks B_AB with 2,
    # B_AB is support-held by L_B which itself supports — L_B's support-hold works.
    nb, dis, _ = resolve(
        b,
        orders(
            "A R_A - B_AB",
            "A CTR S R_A - B_AB",
            "A B_AB S L_B",      # B_AB supports (not moving) -> can be support-held
            "A L_B S B_AB",
            "A R_B H",
        ),
    )
    assert not dis and nb.units["B_AB"] == "B"


def test_support_to_hold_on_moving_unit_void():  # 6.D.7 spirit
    # hold-support for a unit that ordered a move (even a failed one) is void
    b = board({"B_AB": "B", "L_B": "B", "R_A": "A", "CTR": "A", "GATE_A": "A"})
    # B_AB tries to move to CTR (bounces vs GATE_A -> CTR? make it fail via bounce)
    nb, dis, res = resolve(
        b,
        orders(
            "A B_AB - CTR",
            "A L_B S B_AB",       # void: B_AB is moving
            "A GATE_A - CTR",     # bounce B_AB's move
            "A R_A - B_AB",       # attack 1 + CTR support = 2 vs hold 1
            "A CTR S R_A - B_AB",
        ),
    )
    assert res["L_B"] == VOID
    assert dis == {"B_AB"}
    assert nb.units["B_AB"] == "A"


def test_support_to_move_on_holding_unit_void():  # 6.D.9
    b = board({"R_A": "A", "CTR": "A", "B_AB": "B"})
    # CTR supports a move R_A -> B_AB that R_A never makes (R_A holds)
    nb, dis, res = resolve(
        b, orders("A R_A H", "A CTR S R_A - B_AB", "A B_AB H")
    )
    assert res["CTR"] == VOID
    assert not dis and nb.units["B_AB"] == "B"


def test_no_self_dislodgement():  # 6.D.10
    b = board({"R_A": "A", "CTR": "A", "B_AB": "A"})
    nb, dis, res = resolve(
        b, orders("A R_A - B_AB", "A CTR S R_A - B_AB", "A B_AB H")
    )
    assert not dis and nb.units["B_AB"] == "A" and res["R_A"] == BOUNCED


def test_no_self_dislodgement_returning_unit():  # 6.D.11
    # own unit at B_AB bounces (returns); supported own attack still must fail
    b = board({"R_A": "A", "L_B": "A", "B_AB": "A", "GATE_A": "A", "CTR": "C"})
    nb, dis, res = resolve(
        b,
        orders(
            "A B_AB - CTR",          # bounces off C holding CTR? no - CTR occupied, attack 1 vs hold 1
            "A R_A - B_AB",
            "A L_B S R_A - B_AB",
            "A GATE_A H",
            "A CTR H",
        ),
    )
    assert res["B_AB"] == BOUNCED  # returning
    assert not dis and nb.units["B_AB"] == "A" and res["R_A"] == BOUNCED


def test_own_support_cannot_dislodge_own_unit():  # 6.D.12
    # B supports A's attack on B's own unit: that support must not count
    b = board({"R_A": "A", "B_AB": "B", "L_B": "B"})
    nb, dis, _ = resolve(
        b, orders("A R_A - B_AB", "A L_B S R_A - B_AB", "A B_AB H")
    )
    assert not dis and nb.units["B_AB"] == "B"  # attack 1 (excl B's sup) vs hold 1


def test_foreign_support_no_protection():  # 6.D.14
    # giving support to a foreign unit does not protect the giver
    b = board({"L_B": "B", "B_AB": "B", "R_A": "A", "CAP_A": "A", "GATE_B": "A"})
    # L_B supports B_AB to hold; A dislodges L_B with 2
    nb, dis, _ = resolve(
        b,
        orders(
            "A L_B S B_AB",
            "A B_AB H",
            "A GATE_B - L_B",
            "A CAP_A - R_A",      # irrelevant traffic
            "A R_A - L_B",        # not adjacent? R_A adj L_B: yes (flank edge)
        ),
    )
    # two A attacks bounce each other? GATE_B->L_B and R_A->L_B both strength 1:
    # they stand off; L_B survives. Give one of them support instead:
    b2 = board({"L_B": "B", "B_AB": "B", "R_A": "A", "CAP_B": "A"})
    nb2, dis2, _ = resolve(
        b2,
        orders(
            "A L_B S B_AB",
            "A B_AB H",
            "A R_A - L_B",
            "A CAP_B S R_A - L_B",
        ),
    )
    assert dis2 == {"L_B"}
    assert nb2.units["L_B"] == "A"


def test_defender_cannot_cut_support_on_itself():  # 6.D.15
    # CTR (A) supports the attack into B_AB; the unit AT B_AB attacks CTR —
    # an attack out of the supported-into province does NOT cut.
    b = board({"R_A": "A", "CTR": "A", "B_AB": "B"})
    nb, dis, res = resolve(
        b, orders("A R_A - B_AB", "A CTR S R_A - B_AB", "A B_AB - CTR")
    )
    assert res["CTR"] == OK  # not cut
    assert dis == {"B_AB"}
    assert nb.units["B_AB"] == "A" and nb.units["CTR"] == "A"


def test_support_of_empty_province_coerced():  # legalize rule
    b = board({"R_A": "A"})
    nb, _, res = resolve(b, {"R_A": Order("SH", "R_A", "B_AB")})  # B_AB empty
    assert res["R_A"] == VOID and nb.units == {"R_A": "A"}


def test_retroactive_support_void():  # §3.4 step 6
    # R_A supports L_A into GATE_A. GATE_A's unit dislodges R_A — a move out
    # of the very province R_A's support was directed into (not a cut, per
    # 6.D.15's exclusion, but a dislodgement -> retroactive VOID).
    # With the support:    L_A -> GATE_A at 2 beats CTR -> GATE_A at 1.
    # After the void:      both are 1 -> standoff, GATE_A stays EMPTY.
    # Step 6 therefore flips the final board.
    b = board({"L_A": "A", "R_A": "A", "GATE_A": "B", "CAP_A": "B", "CTR": "C"})
    nb, dis, res = resolve(
        b,
        orders(
            "A L_A - GATE_A",
            "A R_A S L_A - GATE_A",
            "A GATE_A - R_A",          # dislodger, origin == support's target q
            "A CAP_A S GATE_A - R_A",  # strength 2 -> dislodges the supporter
            "A CTR - GATE_A",          # competitor that ties once support voided
        ),
    )
    assert dis == {"R_A"}
    assert res["R_A"] == VOID          # retroactively voided, not merely cut
    assert res["L_A"] == BOUNCED       # collapsed to 1 vs 1 standoff
    assert res["CTR"] == BOUNCED
    assert "GATE_A" not in nb.units    # vacated + standoff over it
    assert nb.units["R_A"] == "B"      # dislodger arrived


# --- 6.E head-to-head ---------------------------------------------------------------
def test_dislodged_no_effect_on_attackers_area():  # 6.E.1
    b = board({"R_A": "A", "CTR": "A", "B_AB": "B", "CAP_A": "A"})
    nb, dis, res = resolve(
        b,
        orders(
            "A R_A - B_AB",
            "A CTR S R_A - B_AB",
            "A B_AB - R_A",       # head-to-head, loses
            "A CAP_A - R_A",      # enters the vacated origin unhindered
        ),
    )
    assert dis == {"B_AB"}
    assert nb.units["B_AB"] == "A" and nb.units["R_A"] == "A"
    assert res["CAP_A"] == OK


def test_no_self_dislodgement_head_to_head():  # 6.E.2
    b = board({"CAP_A": "A", "L_A": "A"})
    nb, dis, res = resolve(b, orders("A CAP_A - L_A", "A L_A - CAP_A"))
    assert not dis and nb.units == {"CAP_A": "A", "L_A": "A"}
    assert res["CAP_A"] == BOUNCED and res["L_A"] == BOUNCED


def test_head_to_head_tie_both_fail():
    b = board({"R_A": "A", "B_AB": "B"})
    nb, dis, res = resolve(b, orders("A R_A - B_AB", "A B_AB - R_A"))
    assert not dis and nb.units == {"R_A": "A", "B_AB": "B"}
    assert res["R_A"] == BOUNCED and res["B_AB"] == BOUNCED


def test_non_dislodged_loser_still_prevents():  # 6.E.4
    # h2h tie: B_AB's unit survives; a third unit cannot sneak into B_AB
    b = board({"R_A": "A", "B_AB": "B", "CTR": "C"})
    nb, dis, res = resolve(
        b, orders("A R_A - B_AB", "A B_AB - R_A", "A CTR - B_AB")
    )
    assert not dis
    assert nb.units["B_AB"] == "B" and res["CTR"] == BOUNCED


def test_beleaguered_garrison():
    # two equal-strength supported attacks on an occupied province: occupant stays
    b = board(
        {
            "CTR": "B",
            "GATE_A": "A", "B_AB": "A",
            "GATE_C": "C", "B_BC": "C",
        }
    )
    nb, dis, res = resolve(
        b,
        orders(
            "A CTR H",
            "A GATE_A - CTR",
            "A B_AB S GATE_A - CTR",
            "A GATE_C - CTR",
            "A B_BC S GATE_C - CTR",
        ),
    )
    assert not dis and nb.units["CTR"] == "B"
    assert res["GATE_A"] == BOUNCED and res["GATE_C"] == BOUNCED


# --- property tests -------------------------------------------------------------------
N_PROFILES = 10_000  # CLAUDE.md: equivariance on >= 10k random order profiles


def _random_profile(rng: np.random.Generator) -> tuple[Board, dict[str, Order]]:
    units: dict[str, str] = {}
    for p in PROVINCES:
        r = rng.integers(0, 6)
        if r < 3:
            units[p] = POWERS[r]
    b = board(units)
    ords: dict[str, Order] = {}
    for p in units:
        if rng.random() < 0.15:
            # arbitrary vocab order: exercises the LEGALIZE path
            ords[p] = ORDERS[int(rng.integers(0, len(ORDERS)))]
        else:
            ids = legal_movement_orders(units, p)
            ords[p] = ORDERS[ids[int(rng.integers(0, len(ids)))]]
    return b, ords


def _rotate_board(b: Board) -> Board:
    return Board(
        units={ROTATION[p]: pw for p, pw in b.units.items()},
        sc_owner={ROTATION[sc]: pw for sc, pw in b.sc_owner.items()},
        phase=b.phase,
        year=b.year,
    )


def test_property_suite_random_profiles():
    """Unit conservation, uniqueness, determinism and C3 equivariance."""
    rng = np.random.default_rng(20260704)
    for i in range(N_PROFILES):
        b, ords = _random_profile(rng)
        nb, dis, res = resolve(b, ords)

        # conservation + uniqueness (dict keys are unique by construction,
        # collisions assert inside resolve)
        assert len(nb.units) == len(b.units) - len(dis)
        assert set(dis) <= set(b.units)

        # determinism
        nb2, dis2, res2 = resolve(b, ords)
        assert nb2.units == nb.units and dis2 == dis and res2 == res

        # C3 equivariance: rotate inputs -> rotated outputs
        rb = _rotate_board(b)
        rords = {ROTATION[p]: rotate_order(o) for p, o in ords.items()}
        rnb, rdis, rres = resolve(rb, rords)
        assert rnb.units == {ROTATION[p]: pw for p, pw in nb.units.items()}, i
        assert rdis == {ROTATION[p] for p in dis}, i
        assert rres == {ROTATION[p]: r for p, r in res.items()}, i


def test_powers_never_cut_own_support():
    # A moves onto its own supporter: support must NOT be cut
    b = board({"R_A": "A", "CTR": "A", "CAP_A": "A", "GATE_A": "A", "B_AB": "B"})
    nb, dis, res = resolve(
        b,
        orders(
            "A R_A - B_AB",
            "A CTR S R_A - B_AB",
            "A GATE_A - CTR",          # own attack on own supporter: no cut
            "A CAP_A H",
            "A B_AB H",
        ),
    )
    assert res["CTR"] == OK
    assert dis == {"B_AB"}
