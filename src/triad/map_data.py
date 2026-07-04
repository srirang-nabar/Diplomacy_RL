"""Canonical map data for TRIAD: a rotationally symmetric 3-player Diplomacy variant.

Single source of truth. All other code (engine, env, tests) imports from here.
Powers are indexed 0,1,2 (display names A,B,C) with cyclic order 0->1->2->0.
Naming convention per power i:
  CAP_i  : capital, home supply center (SC)
  L_i    : left flank, home SC, borders B_{i-1,i}  (the border shared with the previous power)
  R_i    : right flank, home SC, borders B_{i,i+1} (the border shared with the next power)
  GATE_i : non-SC pass-through, connects the home region to CTR
Neutral provinces:
  B_AB, B_BC, B_CA : neutral SCs, each between two powers
  CTR              : non-SC central crossroads
"""

POWERS = ["A", "B", "C"]

PROVINCES = [
    "CAP_A", "L_A", "R_A", "GATE_A",
    "CAP_B", "L_B", "R_B", "GATE_B",
    "CAP_C", "L_C", "R_C", "GATE_C",
    "B_AB", "B_BC", "B_CA",
    "CTR",
]

ADJACENCY = {
    "CAP_A":  ["L_A", "R_A", "GATE_A"],
    "L_A":    ["CAP_A", "GATE_A", "B_CA", "R_C"],
    "R_A":    ["CAP_A", "GATE_A", "B_AB", "L_B"],
    "GATE_A": ["CAP_A", "L_A", "R_A", "CTR"],
    "CAP_B":  ["L_B", "R_B", "GATE_B"],
    "L_B":    ["CAP_B", "GATE_B", "B_AB", "R_A"],
    "R_B":    ["CAP_B", "GATE_B", "B_BC", "L_C"],
    "GATE_B": ["CAP_B", "L_B", "R_B", "CTR"],
    "CAP_C":  ["L_C", "R_C", "GATE_C"],
    "L_C":    ["CAP_C", "GATE_C", "B_BC", "R_B"],
    "R_C":    ["CAP_C", "GATE_C", "B_CA", "L_A"],
    "GATE_C": ["CAP_C", "L_C", "R_C", "CTR"],
    "B_AB":   ["R_A", "L_B", "CTR"],
    "B_BC":   ["R_B", "L_C", "CTR"],
    "B_CA":   ["R_C", "L_A", "CTR"],
    "CTR":    ["GATE_A", "GATE_B", "GATE_C", "B_AB", "B_BC", "B_CA"],
}

SUPPLY_CENTERS = [
    "CAP_A", "L_A", "R_A",
    "CAP_B", "L_B", "R_B",
    "CAP_C", "L_C", "R_C",
    "B_AB", "B_BC", "B_CA",
]

HOME_CENTERS = {
    "A": ["CAP_A", "L_A", "R_A"],
    "B": ["CAP_B", "L_B", "R_B"],
    "C": ["CAP_C", "L_C", "R_C"],
}

STARTING_UNITS = {p: list(HOME_CENTERS[p]) for p in POWERS}

VICTORY_CENTERS = 7          # strict majority of 12
MAX_MOVEMENT_PHASES = 40     # 20 game-years; SC-proportional draw at cap

# The C3 rotation rho: power A->B->C->A. Applying it to province names must be
# a graph automorphism (verified in tests). Used at runtime to rotate every
# observation into the acting power's frame, enabling exact parameter sharing.
ROTATION = {
    "CAP_A": "CAP_B", "L_A": "L_B", "R_A": "R_B", "GATE_A": "GATE_B",
    "CAP_B": "CAP_C", "L_B": "L_C", "R_B": "R_C", "GATE_B": "GATE_C",
    "CAP_C": "CAP_A", "L_C": "L_A", "R_C": "R_A", "GATE_C": "GATE_A",
    "B_AB": "B_BC", "B_BC": "B_CA", "B_CA": "B_AB",
    "CTR": "CTR",
}
