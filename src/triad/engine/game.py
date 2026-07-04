"""Game loop for Triad: phases, SC capture, Winter adjustment, termination.

Year structure (CLAUDE.md §3.2):
    Spring movement -> Fall movement -> SC-ownership update -> Winter -> next year

Termination: solo at >= VICTORY_CENTERS SCs after any Winter; hard cap at
MAX_MOVEMENT_PHASES movement phases -> scored draw (SC_i / 12).
"""
from __future__ import annotations

from triad.map_data import (
    HOME_CENTERS,
    MAX_MOVEMENT_PHASES,
    POWERS,
    SUPPLY_CENTERS,
    VICTORY_CENTERS,
)
from triad.engine.adjudicator import resolve
from triad.engine.orders import (
    BUILD,
    BUILD_IDS,
    DISBAND,
    DISBAND_ID,
    ORDERS,
    Order,
    WAIVE,
    legal_movement_orders,
)
from triad.engine.state import Board, FALL, PROVINCE_INDEX, SPRING, WINTER

N_SC = len(SUPPLY_CENTERS)  # 12


class Game:
    """Mutable wrapper advancing a Board through phases until termination."""

    def __init__(self, board: Board | None = None):
        self.board = board.copy() if board is not None else Board.initial()
        self.movement_phases = 0
        self.over = False
        self.result: dict[str, object] | None = None
        self.eliminated: set[str] = set()

    # --- introspection -----------------------------------------------------
    def alive_powers(self) -> list[str]:
        return [pw for pw in POWERS if pw not in self.eliminated]

    def winter_delta(self, power: str) -> int:
        return self.board.sc_count(power) - self.board.unit_count(power)

    def legal_movement_order_ids(self, power: str) -> dict[str, list[int]]:
        """Per-unit legal movement order ids for this power."""
        return {
            p: legal_movement_orders(self.board.units, p)
            for p in self.board.unit_provinces(power)
        }

    def legal_build_ids(self, power: str) -> list[int]:
        """BUILD ids for vacant home SCs still owned by the power."""
        return [
            i
            for i, h in zip(BUILD_IDS[power], HOME_CENTERS[power])
            if self.board.sc_owner[h] == power and h not in self.board.units
        ]

    def legal_disband_ids(self, power: str) -> list[int]:
        return [DISBAND_ID[p] for p in self.board.unit_provinces(power)]

    # --- movement ----------------------------------------------------------
    def step_movement(
        self, orders_by_power: dict[str, dict[str, Order | int]]
    ) -> tuple[set[str], dict[str, str]]:
        """Adjudicate one movement phase. Orders for provinces not holding the
        issuing power's unit are dropped (DATC 6.A.6). Returns (dislodged,
        results)."""
        assert not self.over, "game is over"
        assert self.board.phase in (SPRING, FALL), f"not a movement phase: {self.board.phase}"

        merged: dict[str, Order] = {}
        for pw, om in orders_by_power.items():
            for prov, o in om.items():
                if isinstance(o, int):
                    o = ORDERS[o]
                if self.board.units.get(prov) == pw:
                    merged[prov] = o

        new_board, dislodged, results = resolve(self.board, merged)
        self.movement_phases += 1

        if self.board.phase == SPRING:
            new_board.phase = FALL
        else:  # FALL: capture, then Winter
            for sc in SUPPLY_CENTERS:
                occupant = new_board.units.get(sc)
                if occupant is not None:
                    new_board.sc_owner[sc] = occupant
            new_board.phase = WINTER
        self.board = new_board
        return dislodged, results

    # --- winter ------------------------------------------------------------
    def step_winter(
        self, orders_by_power: dict[str, list[Order | int]] | None = None
    ) -> None:
        """Apply builds/disbands, eliminations, and check termination.

        Build orders beyond delta or invalid (non-home, unowned, occupied,
        duplicate) are ignored (treated as waive). If a power owes disbands
        and provides too few valid ones, the remainder are auto-disbanded in
        canonical province order (deterministic, documented).
        """
        assert not self.over, "game is over"
        assert self.board.phase == WINTER, f"not winter: {self.board.phase}"
        orders_by_power = orders_by_power or {}

        for pw in POWERS:
            if pw in self.eliminated:
                continue
            delta = self.winter_delta(pw)
            given = [ORDERS[o] if isinstance(o, int) else o for o in orders_by_power.get(pw, [])]

            if delta > 0:
                built = 0
                for o in given:
                    if built >= delta:
                        break
                    if o.kind == WAIVE:
                        built += 1  # explicit waive consumes a build slot
                        continue
                    if (
                        o.kind == BUILD
                        and o.src in HOME_CENTERS[pw]
                        and self.board.sc_owner[o.src] == pw
                        and o.src not in self.board.units
                    ):
                        self.board.units[o.src] = pw
                        built += 1
                # unused slots are implicitly waived
            elif delta < 0:
                owed = -delta
                removed: set[str] = set()
                for o in given:
                    if len(removed) >= owed:
                        break
                    if o.kind == DISBAND and self.board.units.get(o.src) == pw and o.src not in removed:
                        removed.add(o.src)
                if len(removed) < owed:  # forced: auto-disband deterministically
                    for p in sorted(self.board.unit_provinces(pw), key=PROVINCE_INDEX.__getitem__):
                        if len(removed) >= owed:
                            break
                        if p not in removed:
                            removed.add(p)
                for p in removed:
                    del self.board.units[p]

        # eliminations: 0 SCs after Winter -> units removed, power out
        for pw in POWERS:
            if pw not in self.eliminated and self.board.sc_count(pw) == 0:
                self.eliminated.add(pw)
                for p in self.board.unit_provinces(pw):
                    del self.board.units[p]

        # termination
        for pw in POWERS:
            if self.board.sc_count(pw) >= VICTORY_CENTERS:
                self.over = True
                self.result = {"type": "solo", "winner": pw, "scores": self._solo_scores(pw)}
                return
        if self.movement_phases >= MAX_MOVEMENT_PHASES:
            self.over = True
            self.result = {"type": "draw", "winner": None, "scores": self._draw_scores()}
            return

        self.board.phase = SPRING
        self.board.year += 1

    # --- scoring (CLAUDE.md §4.5) --------------------------------------------
    def _solo_scores(self, winner: str) -> dict[str, float]:
        return {pw: 1.0 if pw == winner else 0.0 for pw in POWERS}

    def _draw_scores(self) -> dict[str, float]:
        return {
            pw: 0.0 if pw in self.eliminated else self.board.sc_count(pw) / N_SC
            for pw in POWERS
        }
