"""Structural invariants of the Triad map. Any map edit must keep these green."""
from triad.map_data import (PROVINCES, ADJACENCY, SUPPLY_CENTERS, HOME_CENTERS,
                            ROTATION, POWERS, VICTORY_CENTERS)


def test_province_sets_consistent():
    assert set(ADJACENCY) == set(PROVINCES) == set(ROTATION)
    assert len(PROVINCES) == 16


def test_adjacency_symmetric_simple():
    for p, nbrs in ADJACENCY.items():
        assert len(nbrs) == len(set(nbrs)), f"duplicate neighbor at {p}"
        assert p not in nbrs, f"self-loop at {p}"
        for q in nbrs:
            assert p in ADJACENCY[q], f"asymmetric edge {p}->{q}"


def test_connected():
    seen, stack = {"CAP_A"}, ["CAP_A"]
    while stack:
        for n in ADJACENCY[stack.pop()]:
            if n not in seen:
                seen.add(n)
                stack.append(n)
    assert seen == set(PROVINCES)


def test_rotation_is_order3_automorphism():
    for p in PROVINCES:
        assert sorted(ROTATION[n] for n in ADJACENCY[p]) == \
               sorted(ADJACENCY[ROTATION[p]]), f"rotation breaks adjacency at {p}"
    for p in PROVINCES:
        assert ROTATION[ROTATION[ROTATION[p]]] == p


def test_rotation_respects_scs_and_homes():
    assert {ROTATION[s] for s in SUPPLY_CENTERS} == set(SUPPLY_CENTERS)
    assert sorted(ROTATION[h] for h in HOME_CENTERS["A"]) == sorted(HOME_CENTERS["B"])
    assert sorted(ROTATION[h] for h in HOME_CENTERS["B"]) == sorted(HOME_CENTERS["C"])


def test_counts_and_victory():
    assert len(SUPPLY_CENTERS) == 12
    assert VICTORY_CENTERS == 7  # strict majority
    degs = sorted(len(v) for v in ADJACENCY.values())
    # 6 degree-3 (3 capitals + 3 border SCs), 9 degree-4 (6 flanks + 3 gates),
    # CTR degree 6. Flanks border the opposing power's flank directly.
    assert degs == [3] * 6 + [4] * 9 + [6]
    assert sum(len(v) for v in ADJACENCY.values()) // 2 == 30


def test_order_vocab_size():
    degs = {p: len(ADJACENCY[p]) for p in PROVINCES}
    holds = len(PROVINCES)
    moves = sum(degs.values())
    sup_hold = sum(degs.values())
    sup_move = sum((degs[d] - 1) for d in PROVINCES for _ in ADJACENCY[d])
    adjustment = 9 + 16 + 1
    assert holds + moves + sup_hold + sup_move == 310
    assert holds + moves + sup_hold + sup_move + adjustment == 336
