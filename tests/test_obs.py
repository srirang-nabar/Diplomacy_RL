"""Observation builder tests — above all, the seat-equivariance guarantee
that makes exact weight sharing sound (CLAUDE.md §4.1)."""
from __future__ import annotations

import numpy as np

from triad.map_data import POWERS, PROVINCES, ROTATION
from triad.engine.state import Board, FALL, SPRING, WINTER
from triad.env.obs import OBS_DIM, encode_observation


def _rotate_board(b: Board) -> Board:
    return Board(
        units={ROTATION[p]: _next(pw) for p, pw in b.units.items()},
        sc_owner={ROTATION[sc]: (None if pw is None else _next(pw)) for sc, pw in b.sc_owner.items()},
        phase=b.phase,
        year=b.year,
    )


def _next(pw: str) -> str:
    return POWERS[(POWERS.index(pw) + 1) % 3]


def _random_board(rng: np.random.Generator) -> Board:
    b = Board.initial()
    b.units = {}
    for p in PROVINCES:
        r = int(rng.integers(0, 6))
        if r < 3:
            b.units[p] = POWERS[r]
    for sc in list(b.sc_owner):
        r = int(rng.integers(0, 4))
        b.sc_owner[sc] = None if r == 3 else POWERS[r]
    b.phase = [SPRING, FALL, WINTER][int(rng.integers(0, 3))]
    b.year = int(rng.integers(1, 21))
    return b


def test_shape_dtype_range():
    obs = encode_observation(Board.initial(), "A")
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_initial_board_self_features():
    obs = encode_observation(Board.initial(), "B")
    feat = obs[: 16 * 12].reshape(16, 12)
    i_cap_a = PROVINCES.index("CAP_A")
    # in B's own frame, slot CAP_A holds B's real capital: unit=self, sc=self
    assert feat[i_cap_a, 1] == 1.0 and feat[i_cap_a, 5] == 1.0
    assert feat[i_cap_a, 9] == 1.0  # home of self


def test_seat_equivariance_exact():
    """encode(rotate_full(b), next(pw)) == encode(b, pw) bitwise.

    rotate_full moves both provinces AND power labels one step (A's position
    is handed to B), so the next power in the rotated board stands exactly
    where pw stood — the own-frame views must be identical arrays.
    """
    rng = np.random.default_rng(20260704)
    for _ in range(500):
        b = _random_board(rng)
        rb = _rotate_board(b)
        for pw in POWERS:
            a = encode_observation(b, pw)
            c = encode_observation(rb, _next(pw))
            assert np.array_equal(a, c), f"seat equivariance broken for {pw}"


def test_identical_position_identical_obs():
    # symmetric start: all three seats must see the exact same observation
    b = Board.initial()
    o = [encode_observation(b, pw) for pw in POWERS]
    assert np.array_equal(o[0], o[1]) and np.array_equal(o[1], o[2])


def test_unit_decode_order_is_seat_equivariant():
    """Regression for the M3 seat-effect bug: the AR decode ORDER of a
    power's units must map under rotation, not just the observation.
    Real-frame canonical order violates this (B_BC < B_CA but their images
    B_CA > B_AB flip); own-frame slot order is equivariant by construction.
    Caught by the self-play chi^2 check (C > A > B across seeds), not by the
    obs-equivariance test — order and content are separate properties."""
    from triad.env.obs import own_frame_unit_order

    rng = np.random.default_rng(3)
    for _ in range(300):
        b = _random_board(rng)
        rb = _rotate_board(b)
        for pw in POWERS:
            order = own_frame_unit_order(b.units, pw)
            r_order = own_frame_unit_order(rb.units, _next(pw))
            assert r_order == [ROTATION[p] for p in order], (pw, order, r_order)
