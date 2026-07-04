"""Order vocabulary invariants: counts, index stability, round-trips, tables."""
from triad.map_data import ADJACENCY, PROVINCES, ROTATION
from triad.engine.orders import (
    BASE_MOVEMENT,
    BUILD,
    BUILD_IDS,
    CONDITIONAL_MOVEMENT,
    DISBAND,
    HOLD,
    MOVE,
    ORDERS,
    ORDER_INDEX,
    SUP_HOLD,
    SUP_MOVE,
    VOCAB_SIZE,
    WAIVE,
    Order,
    format_order,
    legal_movement_orders,
    parse_order,
    rotate_order,
)


def test_vocab_counts_by_category():
    by_kind: dict[str, int] = {}
    for o in ORDERS:
        by_kind[o.kind] = by_kind.get(o.kind, 0) + 1
    assert by_kind[HOLD] == 16
    assert by_kind[MOVE] == 60
    assert by_kind[SUP_HOLD] == 60
    assert by_kind[SUP_MOVE] == 174
    assert by_kind[BUILD] == 9
    assert by_kind[DISBAND] == 16
    assert by_kind[WAIVE] == 1
    assert VOCAB_SIZE == 336


def test_vocab_unique_and_index_bijective():
    assert len(set(ORDERS)) == VOCAB_SIZE
    for i, o in enumerate(ORDERS):
        assert ORDER_INDEX[o] == i


def test_all_orders_adjacency_valid():
    for o in ORDERS:
        if o.kind == MOVE:
            assert o.dst in ADJACENCY[o.src]
        elif o.kind == SUP_HOLD:
            assert o.aux in ADJACENCY[o.src]
        elif o.kind == SUP_MOVE:
            assert o.dst in ADJACENCY[o.src]
            assert o.dst in ADJACENCY[o.aux]
            assert o.aux != o.src


def test_text_round_trip_all_336():
    for o in ORDERS:
        assert parse_order(format_order(o)) == o


def test_rotation_is_vocab_permutation_of_order_3():
    ids = set()
    for o in ORDERS:
        r = rotate_order(o)
        assert r in ORDER_INDEX, f"rotation left vocab: {o} -> {r}"
        ids.add(ORDER_INDEX[r])
        assert rotate_order(rotate_order(r)) == o  # rho^3 = id
    assert ids == set(range(VOCAB_SIZE))


def test_static_tables_cover_movement_kinds():
    for p in PROVINCES:
        deg = len(ADJACENCY[p])
        assert len(BASE_MOVEMENT[p]) == 1 + deg  # hold + moves
        n_cond = sum(len(v) for v in CONDITIONAL_MOVEMENT[p].values())
        # supports out of p: deg support-holds + sum over d in adj(p) of (deg(d)-1)
        expect = deg + sum(len(ADJACENCY[d]) - 1 for d in ADJACENCY[p])
        assert n_cond == expect


def test_dynamic_legality_requires_occupancy():
    units = {"B_AB": "B"}  # only B_AB occupied
    ids = legal_movement_orders(units, "R_A")
    orders = [ORDERS[i] for i in ids]
    # hold + 4 moves always present
    assert Order(HOLD, "R_A") in orders
    assert Order(MOVE, "R_A", None, "B_AB") in orders
    # support-hold of occupied B_AB legal; of empty CAP_A not
    assert Order(SUP_HOLD, "R_A", "B_AB") in orders
    assert Order(SUP_HOLD, "R_A", "CAP_A") not in orders
    # support-move of the unit at B_AB legal; of a unit that isn't there, not
    assert Order(SUP_MOVE, "R_A", "B_AB", "L_B") in orders
    assert Order(SUP_MOVE, "R_A", "CTR", "B_AB") not in orders


def test_rotation_consistent_with_map():
    # spot-check the province map underlying rotate_order
    assert ROTATION["R_A"] == "R_B" and ROTATION["B_CA"] == "B_AB"
