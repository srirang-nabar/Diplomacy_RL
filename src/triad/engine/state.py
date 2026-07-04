"""Board state for Triad.

The Board is a plain dataclass with no behaviour beyond construction, copying
and invariant checks. All mutation happens in game.py; adjudicator.py treats
boards as immutable inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from triad.map_data import (
    HOME_CENTERS,
    POWERS,
    PROVINCES,
    STARTING_UNITS,
    SUPPLY_CENTERS,
)

# Phase constants (movement seasons + adjustment).
SPRING = "SPRING"
FALL = "FALL"
WINTER = "WINTER"
PHASES = (SPRING, FALL, WINTER)

#: Canonical province -> index (used everywhere determinism needs an order).
PROVINCE_INDEX: dict[str, int] = {p: i for i, p in enumerate(PROVINCES)}


@dataclass
class Board:
    """Full game state at a point in time.

    units:    province -> owning power, at most one unit per province.
    sc_owner: supply center -> owning power or None (neutral/unowned).
    phase:    SPRING | FALL (movement) | WINTER (adjustment).
    year:     1-based game year.
    """

    units: dict[str, str] = field(default_factory=dict)
    sc_owner: dict[str, str | None] = field(default_factory=dict)
    phase: str = SPRING
    year: int = 1

    @staticmethod
    def initial() -> "Board":
        """Standard start: 3 armies on home SCs, home SCs owned, borders unowned."""
        units = {p: pw for pw in POWERS for p in STARTING_UNITS[pw]}
        sc_owner: dict[str, str | None] = {sc: None for sc in SUPPLY_CENTERS}
        for pw in POWERS:
            for h in HOME_CENTERS[pw]:
                sc_owner[h] = pw
        return Board(units=units, sc_owner=sc_owner, phase=SPRING, year=1)

    def copy(self) -> "Board":
        return Board(
            units=dict(self.units),
            sc_owner=dict(self.sc_owner),
            phase=self.phase,
            year=self.year,
        )

    # --- queries -----------------------------------------------------------
    def unit_provinces(self, power: str) -> list[str]:
        """Provinces holding this power's units, in canonical order."""
        return [p for p in PROVINCES if self.units.get(p) == power]

    def unit_count(self, power: str) -> int:
        return sum(1 for pw in self.units.values() if pw == power)

    def sc_count(self, power: str) -> int:
        return sum(1 for pw in self.sc_owner.values() if pw == power)

    def owned_scs(self, power: str) -> list[str]:
        return [sc for sc in SUPPLY_CENTERS if self.sc_owner.get(sc) == power]

    # --- invariants --------------------------------------------------------
    def validate(self) -> None:
        """Assert structural invariants; raises AssertionError on violation."""
        for p, pw in self.units.items():
            assert p in PROVINCE_INDEX, f"unknown province {p!r}"
            assert pw in POWERS, f"unknown power {pw!r} at {p}"
        assert set(self.sc_owner) == set(SUPPLY_CENTERS), "sc_owner keys != SCs"
        for sc, pw in self.sc_owner.items():
            assert pw is None or pw in POWERS, f"bad SC owner {pw!r} at {sc}"
        assert self.phase in PHASES, f"bad phase {self.phase!r}"
        assert self.year >= 1
