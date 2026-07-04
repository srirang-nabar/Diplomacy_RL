"""Movement-phase adjudicator for Triad — pure, deterministic (CLAUDE.md §3.4).

Armies only, no convoys, no retreats: dislodgement = removal. Strength
definitions follow Kruijswijk's "The Math of Adjudication". The resolution
is a bounded fixpoint iteration with interval bounds for the two lazily
resolved quantities (hold strength of a vacating occupant, prevent strength
of a head-to-head participant); residual undecided moves are pure movement
cycles and all succeed.

Public API:
    resolve(board, orders) -> (new_board, dislodged, results)
"""
from __future__ import annotations

from triad.map_data import ADJACENCY
from triad.engine.orders import HOLD, MOVE, Order, SUP_HOLD, SUP_MOVE
from triad.engine.state import Board, PROVINCE_INDEX

# results[] outcome strings
OK = "ok"            # move succeeded / support given / hold held
BOUNCED = "bounced"  # move failed
CUT = "cut"          # support cut by an attack
VOID = "void"        # support did not match the supported unit's order,
                     # was coerced from an illegal order, or was
                     # retroactively voided by dislodgement from its target


def resolve(
    board: Board,
    orders: dict[str, Order],
) -> tuple[Board, set[str], dict[str, str]]:
    """Adjudicate one movement phase.

    orders maps unit province -> Order. Missing / illegal orders are coerced
    to HOLD (step 1). Returns the post-movement board (same phase/year — the
    caller advances those), the set of provinces whose units were dislodged
    (removed), and a per-unit-province outcome string.
    """
    units = board.units
    unit_provs = sorted(units, key=PROVINCE_INDEX.__getitem__)

    # ---- step 1: LEGALIZE ---------------------------------------------------
    legal: dict[str, Order] = {}
    coerced: set[str] = set()
    for p in unit_provs:
        o = _legalize(p, orders.get(p), units)
        if orders.get(p) is not None and o != orders[p]:
            coerced.add(p)
        legal[p] = o

    movers = {p: o.dst for p, o in legal.items() if o.kind == MOVE}

    # ---- steps 2-5 with step-6 outer loop (retroactive support void) --------
    voided: set[str] = set()
    n_supports = sum(1 for o in legal.values() if o.kind in (SUP_HOLD, SUP_MOVE))
    for _ in range(n_supports + 1):
        status, dislodged, cut, matched = _resolve_once(units, legal, movers, voided)
        new_voids = _retroactive_voids(legal, movers, status, dislodged, voided)
        if not new_voids:
            break
        voided |= new_voids
    else:  # pragma: no cover - loop is bounded by construction
        raise AssertionError("retroactive-void loop failed to converge")

    # ---- build the new board -------------------------------------------------
    new_units: dict[str, str] = {}
    for p in unit_provs:
        if p in movers and status[p]:
            dst = movers[p]
            assert dst not in new_units, f"two units resolved into {dst}"
            new_units[dst] = units[p]
    for p in unit_provs:
        if (p in movers and status[p]) or p in dislodged:
            continue
        assert p not in new_units, f"stayer at {p} collides with incoming unit"
        new_units[p] = units[p]
    assert len(new_units) == len(units) - len(dislodged), "unit conservation"

    new_board = Board(
        units=new_units, sc_owner=dict(board.sc_owner), phase=board.phase, year=board.year
    )

    # ---- per-unit outcome strings ---------------------------------------------
    results: dict[str, str] = {}
    for p in unit_provs:
        o = legal[p]
        if o.kind == MOVE:
            results[p] = OK if status[p] else BOUNCED
        elif o.kind in (SUP_HOLD, SUP_MOVE):
            if p in voided or p not in matched:
                results[p] = VOID
            elif p in cut:
                results[p] = CUT
            else:
                results[p] = OK
        else:
            results[p] = VOID if p in coerced else OK
    return new_board, dislodged, results


# --- step 1 helpers ------------------------------------------------------------
def _legalize(p: str, o: Order | None, units: dict[str, str]) -> Order:
    """Coerce anything invalid to HOLD (CLAUDE.md §3.4 step 1)."""
    hold = Order(HOLD, p)
    if o is None or o.kind not in (HOLD, MOVE, SUP_HOLD, SUP_MOVE):
        return hold
    if o.src != p:
        return hold
    if o.kind == HOLD:
        return o
    if o.kind == MOVE:
        return o if o.dst in ADJACENCY[p] else hold
    if o.kind == SUP_HOLD:
        # target must be adjacent and occupied (support of empty province is void)
        return o if (o.aux in ADJACENCY[p] and o.aux in units) else hold
    # SUP_MOVE: d adj s, d adj u, u != s, and a unit must exist at u
    if (
        o.dst in ADJACENCY[p]
        and o.aux is not None
        and o.dst in ADJACENCY[o.aux]
        and o.aux != p
        and o.aux in units
    ):
        return o
    return hold


# --- steps 2-5 -------------------------------------------------------------------
def _resolve_once(
    units: dict[str, str],
    legal: dict[str, Order],
    movers: dict[str, str],
    voided: set[str],
) -> tuple[dict[str, bool], set[str], set[str], set[str]]:
    """One full pass of steps 2-5 given the current set of voided supports.

    Returns (move status by origin, dislodged provinces, cut supporters,
    matched supporters).
    """
    unit_provs = sorted(units, key=PROVINCE_INDEX.__getitem__)

    # matched supports (step 3 prerequisites) and the province each support
    # is directed into (step 2)
    directed: dict[str, str] = {}
    atk_sup: dict[str, list[str]] = {m: [] for m in movers}  # mover -> supporters
    hold_sup: dict[str, list[str]] = {}                      # stayer -> supporters
    matched: set[str] = set()
    for p in unit_provs:
        o = legal[p]
        if o.kind == SUP_MOVE:
            directed[p] = o.dst
            if p not in voided and movers.get(o.aux) == o.dst:
                matched.add(p)
                atk_sup[o.aux].append(p)
        elif o.kind == SUP_HOLD:
            directed[p] = o.aux
            # support-hold counts only if the target did not order a move
            if p not in voided and o.aux in units and legal[o.aux].kind != MOVE:
                matched.add(p)
                hold_sup.setdefault(o.aux, []).append(p)

    # step 2: static support cut
    cut: set[str] = set()
    for p, q in directed.items():
        for r, d in movers.items():
            if d == p and r != q and units[r] != units[p]:
                cut.add(p)
                break

    def _uncut(sups: list[str]) -> list[str]:
        return [s for s in sups if s not in cut]

    eff_sup = {m: _uncut(atk_sup[m]) for m in movers}
    hold_static = {
        p: 1 + len(_uncut(hold_sup.get(p, [])))
        for p in unit_provs
        if p not in movers
    }

    def attack_full(m: str) -> int:
        return 1 + len(eff_sup[m])

    def attack_excl(m: str, power: str) -> int:
        """Attack strength excluding supports given by `power` (§3.4 step 3)."""
        return 1 + sum(1 for s in eff_sup[m] if units[s] != power)

    status: dict[str, bool | None] = {m: None for m in movers}
    mover_list = sorted(movers, key=PROVINCE_INDEX.__getitem__)

    def prevent_bounds(m: str) -> tuple[int, int]:
        """(lo, hi) prevent strength; 0 iff m lost its head-to-head."""
        full = attack_full(m)
        d = movers[m]
        # opposing head-to-head move: the unit at m's destination moving to m
        if d in movers and movers[d] == m:
            opp_status = status[d]
            if opp_status is True:
                return (0, 0)
            if opp_status is None:
                return (0, full)
        return (full, full)

    def try_decide(s: str) -> bool | None:
        d = movers[s]
        me = units[s]
        occ = units.get(d)

        # prevent competition (needed in every branch): attack_full(s) must
        # strictly beat every other move into d
        prev_pending = False
        for c in mover_list:
            if c == s or movers[c] != d:
                continue
            lo, hi = prevent_bounds(c)
            if attack_full(s) <= lo:
                return False
            if attack_full(s) <= hi:
                prev_pending = True

        # head-to-head: the occupant of d has a move order to s
        if occ is not None and movers.get(d) == s:
            if occ == me:
                return False  # same-power swap: no self-dislodgement
            atk = attack_excl(s, occ)
            dfd = attack_full(d)
            if atk <= dfd:
                return False
            return None if prev_pending else True

        if occ is None:
            return None if prev_pending else True

        # occupied destination, not head-to-head
        if d in movers:
            occ_stays = None if status[d] is None else (not status[d])
        else:
            occ_stays = True

        if occ == me:  # own unit: may only enter if it actually vacates
            if occ_stays is True:
                return False
            if occ_stays is None:
                return None
            return None if prev_pending else True

        atk_stay = attack_excl(s, occ)
        hold_stay = 1 if d in movers else hold_static[d]
        if occ_stays is True:
            if atk_stay <= hold_stay:
                return False
            return None if prev_pending else True
        if occ_stays is False:
            return None if prev_pending else True
        # occupant's move still unresolved
        if atk_stay > hold_stay:
            # strong enough even if it stays -> outcome independent of vacate
            return None if prev_pending else True
        return None

    # step 4: fixpoint iteration, then circular-movement fallback
    guard = 0
    while True:
        guard += 1
        assert guard < 10 * (len(movers) + 2), "resolution failed to converge"
        changed = False
        for s in mover_list:
            if status[s] is None:
                r = try_decide(s)
                if r is not None:
                    status[s] = r
                    changed = True
        if changed:
            continue
        undecided = [s for s in mover_list if status[s] is None]
        if not undecided:
            break
        cycle = _find_cycle_members(undecided, movers)
        assert cycle, f"undecided moves with no cycle: {undecided}"
        for s in cycle:
            status[s] = True  # circular movement: all succeed

    # sanity: at most one successful move per destination
    seen_dst: set[str] = set()
    for s in mover_list:
        if status[s]:
            assert movers[s] not in seen_dst
            seen_dst.add(movers[s])

    # step 5: dislodgement
    dislodged: set[str] = set()
    for p in unit_provs:
        if p in movers and status[p]:
            continue  # vacated
        if any(status[s] and movers[s] == p for s in mover_list):
            dislodged.add(p)

    final_status = {m: bool(status[m]) for m in movers}
    return final_status, dislodged, cut, matched


def _find_cycle_members(undecided: list[str], movers: dict[str, str]) -> set[str]:
    """Provinces whose moves form cycles (each waits on the next to vacate)."""
    und = set(undecided)
    on_cycle: set[str] = set()
    color: dict[str, int] = {}  # 1 = on current path, 2 = done
    for start in undecided:
        if color.get(start):
            continue
        path: list[str] = []
        cur: str | None = start
        while cur is not None and cur in und and color.get(cur) != 2:
            if color.get(cur) == 1:  # found a cycle: everything from cur onward
                i = path.index(cur)
                on_cycle.update(path[i:])
                break
            color[cur] = 1
            path.append(cur)
            nxt = movers[cur]
            cur = nxt if nxt in und else None
        for p in path:
            color[p] = 2
    return on_cycle


# --- step 6 ---------------------------------------------------------------------
def _retroactive_voids(
    legal: dict[str, Order],
    movers: dict[str, str],
    status: dict[str, bool],
    dislodged: set[str],
    voided: set[str],
) -> set[str]:
    """Supports whose giver was dislodged by a move out of the very province
    the support was directed into (§3.4 step 6)."""
    new: set[str] = set()
    for p in dislodged:
        o = legal[p]
        if o.kind not in (SUP_HOLD, SUP_MOVE) or p in voided:
            continue
        q = o.aux if o.kind == SUP_HOLD else o.dst
        for s, d in movers.items():
            if d == p and status[s] and s == q:
                new.add(p)
                break
    return new
