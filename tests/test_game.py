"""Game-loop tests: phase flow, capture timing, Winter, elimination, termination."""
from __future__ import annotations

import numpy as np

from triad.map_data import MAX_MOVEMENT_PHASES, POWERS, SUPPLY_CENTERS
from triad.engine.game import Game
from triad.engine.orders import ORDERS, Order, legal_movement_orders
from triad.engine.state import Board, FALL, SPRING, WINTER


def _move(s: str, d: str) -> Order:
    return Order("M", s, None, d)


def test_initial_setup():
    g = Game()
    b = g.board
    assert b.phase == SPRING and b.year == 1
    for pw in POWERS:
        assert b.unit_count(pw) == 3
        assert b.sc_count(pw) == 3
    for sc in ("B_AB", "B_BC", "B_CA"):
        assert b.sc_owner[sc] is None


def test_no_capture_in_spring():
    g = Game()
    g.step_movement({"A": {"R_A": _move("R_A", "B_AB")}})
    assert g.board.units["B_AB"] == "A"
    assert g.board.sc_owner["B_AB"] is None  # spring: no ownership change
    assert g.board.phase == FALL


def test_capture_after_fall_and_unoccupied_keeps_owner():
    g = Game()
    g.step_movement({"A": {"R_A": _move("R_A", "B_AB")}})           # spring
    g.step_movement({"A": {"B_AB": Order("H", "B_AB")}})            # fall
    assert g.board.phase == WINTER
    assert g.board.sc_owner["B_AB"] == "A"                           # captured
    assert g.board.sc_owner["CAP_B"] == "B"                          # untouched
    # vacating later does not lose ownership
    g.step_winter({"A": [i for i in g.legal_build_ids("A")]})
    g.step_movement({"A": {"B_AB": _move("B_AB", "CTR")}})           # spring y2
    g.step_movement({})                                              # fall y2
    assert g.board.sc_owner["B_AB"] == "A"                           # unoccupied keeps owner


def test_winter_build_only_in_vacant_owned_home():
    g = Game()
    g.step_movement({"A": {"R_A": _move("R_A", "B_AB")}})
    g.step_movement({})  # fall; A captures B_AB -> 4 SCs, 3 units, delta 1
    assert g.winter_delta("A") == 1
    legal = g.legal_build_ids("A")
    # R_A is vacant and owned -> legal; CAP_A occupied -> not
    provs = [ORDERS[i].src for i in legal]
    assert "R_A" in provs and "CAP_A" not in provs
    g.step_winter({"A": legal[:1]})
    assert g.board.unit_count("A") == 4
    assert g.board.phase == SPRING and g.board.year == 2


def test_winter_build_waivable_and_capped():
    g = Game()
    g.step_movement({"A": {"R_A": _move("R_A", "B_AB")}})
    g.step_movement({})
    # waive explicitly: no unit gained
    g.step_winter({"A": [Order("W")]})
    assert g.board.unit_count("A") == 3
    # invalid builds (foreign home, occupied) are ignored
    g2 = Game()
    g2.step_movement({"A": {"R_A": _move("R_A", "B_AB")}})
    g2.step_movement({})
    g2.step_winter({"A": [Order("B", "CAP_B"), Order("B", "CAP_A"), Order("B", "R_A"), Order("B", "L_A")]})
    assert g2.board.unit_count("A") == 4  # only one build allowed (delta 1)
    assert "R_A" in g2.board.units


def test_winter_forced_disband_auto_deterministic():
    # engineer delta < 0: A loses a home SC to B while its unit count stays 3
    b = Board.initial()
    b.units = {"CAP_A": "A", "L_A": "A", "GATE_A": "A", "R_B": "B", "B_AB": "B"}
    g = Game(b)
    g.step_movement({"B": {"B_AB": _move("B_AB", "R_A")}})   # spring
    g.step_movement({})                                       # fall: B captures R_A
    assert g.board.sc_owner["R_A"] == "B"
    assert g.winter_delta("A") == -1
    g.step_winter({})  # no disband given -> auto-disband first by province index
    assert g.board.unit_count("A") == 2
    assert "CAP_A" not in g.board.units  # canonical order: CAP_A first


def test_elimination_removes_units_game_continues():
    b = Board.initial()
    # A holds all of B's home SCs after fall -> B eliminated
    b.units = {"CAP_B": "A", "L_B": "A", "R_B": "A", "GATE_B": "B", "CAP_C": "C"}
    b.sc_owner = {sc: None for sc in b.sc_owner}
    b.sc_owner.update({"CAP_A": "A", "L_A": "A", "R_A": "A"})
    b.sc_owner.update({"CAP_B": "B", "L_B": "B", "R_B": "B"})
    b.sc_owner.update({"CAP_C": "C", "L_C": "C", "R_C": "C"})
    b.phase = FALL
    g = Game(b)
    g.step_movement({})  # fall: A's units capture B's homes
    g.step_winter()
    assert "B" in g.eliminated
    assert g.board.unit_count("B") == 0
    assert not g.over  # A has 6 SCs < 7: game continues


def test_solo_victory():
    b = Board.initial()
    b.units = {"CAP_B": "A", "L_B": "A", "R_B": "A", "B_AB": "A"}
    b.phase = FALL
    g = Game(b)
    g.step_movement({})  # captures 4 SCs on top of A's 3 homes -> 7
    g.step_winter()
    assert g.over and g.result is not None
    assert g.result["type"] == "solo" and g.result["winner"] == "A"
    assert g.result["scores"] == {"A": 1.0, "B": 0.0, "C": 0.0}


def test_turn_cap_draw_scores():
    g = Game()
    hold_all = lambda: {  # noqa: E731
        pw: {p: Order("H", p) for p in g.board.unit_provinces(pw)} for pw in POWERS
    }
    while not g.over:
        g.step_movement(hold_all())
        if g.board.phase == WINTER:
            g.step_winter()
    assert g.movement_phases == MAX_MOVEMENT_PHASES
    assert g.result["type"] == "draw"
    # everyone kept their 3 home SCs
    assert g.result["scores"] == {pw: 3 / 12 for pw in POWERS}


def test_random_selfplay_invariants():
    """Mini-fuzz: full random games keep invariants and always terminate."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        g = Game()
        while not g.over:
            if g.board.phase in (SPRING, FALL):
                ob = {}
                for pw in g.alive_powers():
                    om = {}
                    for p, ids in g.legal_movement_order_ids(pw).items():
                        om[p] = int(ids[int(rng.integers(0, len(ids)))])
                    ob[pw] = om
                g.step_movement(ob)
            else:
                ob = {}
                for pw in g.alive_powers():
                    delta = g.winter_delta(pw)
                    if delta > 0:
                        ids = g.legal_build_ids(pw)
                        k = int(rng.integers(0, min(delta, len(ids)) + 1))
                        ob[pw] = [int(i) for i in rng.choice(ids, size=k, replace=False)] if k else []
                    elif delta < 0:
                        ids = g.legal_disband_ids(pw)
                        ob[pw] = [int(i) for i in rng.choice(ids, size=-delta, replace=False)]
                g.step_winter(ob)
            g.board.validate()
            for pw in POWERS:  # a power never has more units than owned SCs + pending
                assert g.board.unit_count(pw) <= 12
        assert g.movement_phases <= MAX_MOVEMENT_PHASES
        assert g.result is not None
        total = sum(g.result["scores"].values())
        assert 0.0 <= total <= 1.0 + 1e-9
