# TRIAD-RL

Multi-agent reinforcement learning on a custom rotationally symmetric 3-player
no-press Diplomacy variant ("Triad"). Methodology follows Paquette et al.
(NeurIPS 2019, "No Press Diplomacy") at small scale: supervised bootstrap →
self-play RL fine-tuning with a policy anchor.

Owner: Srirang (PGDBA). Target: defensible placement-interview project by
**2026-08-10**. Panels: quant research / AI-DS. Optimize for correctness,
measurable results, and explainability over feature count.

---

## 1. Research question and thesis

**Primary question:** In a symmetric 3-player simultaneous-move game with no
communication, does *balance-of-power behavior* (the two trailing powers
implicitly coalescing against the leader) emerge from self-play RL?

**Secondary questions:**
1. Does the DipNet result reproduce at small scale — i.e., PPO from scratch
   fails or plateaus, while behaviour-cloning bootstrap + PPO with a KL anchor
   to the BC policy succeeds?
2. How does reward shaping (per-SC delta) affect the draw attractor?

**Deliverables:** working codebase, learning curves, tournament tables with
confidence intervals, an attack-the-leader analysis, one ablation grid, a
short report + slide deck.

**Non-goals (do not build):** fleets, convoys, coasts, retreat phases, press /
negotiation, full 7-player Diplomacy, distributed training, DAIDE
compatibility. If a task seems to need one of these, stop and flag it.

---

## 2. Status tracker

Update this section as work completes. Today = W0.

- [x] M0 (Jul 3–5): map designed, canonical data in `src/map_data.py`,
      all structural invariants verified (`src/test_map.py`)
- [ ] M1 (Jul 6–12): game engine + adjudicator + DATC-adapted test suite
      passing; random self-play benchmark recorded
- [ ] M2 (Jul 13–19): PettingZoo env; heuristic bots (RandomLegal, Grabber,
      Turtle); BC dataset (≥50k games); BC policy beats 2×RandomLegal in >80%
      of games
- [ ] M3 (Jul 20–26): PPO self-play with KL anchor training end-to-end;
      learning curves in TensorBoard/W&B
- [ ] M4 (Jul 27–Aug 2): population-based opponent sampling; hyperparameter
      pass; seat-symmetry sanity checks
- [ ] M5 (Aug 3–9): evaluation suite (round-robin, Elo), ablation grid,
      attack-the-leader analysis; optional Pentad (5-player) config demo
- [ ] M6 (Aug 10–16): report, 10-slide deck, interview Q&A prep

If behind schedule, cut in this order: (1) GNN encoder (keep MLP), (2) Pentad
demo, (3) population play (keep latest-checkpoint self-play). Never cut:
adjudicator tests, ablation {BC / PPO-scratch / BC+PPO+KL}, tournament CIs.

---

## 3. Environment specification

### 3.1 Map ("Triad")

Canonical data lives in `triad/map_data.py` — **single source of truth**; all
engine/env/test code imports from it. Never duplicate adjacency literals
elsewhere.

- 16 provinces, 27 undirected edges, 12 supply centers (SCs), armies only.
- 3 powers A, B, C with cyclic order A→B→C→A. Per power `i`:
  - `CAP_i` — capital, home SC
  - `L_i` — left flank, home SC, borders `B_{prev(i),i}`
  - `R_i` — right flank, home SC, borders `B_{i,next(i)}`
  - `GATE_i` — non-SC pass-through, connects home region to `CTR`
- Neutrals: `B_AB`, `B_BC`, `B_CA` (neutral SCs, one per power pair), `CTR`
  (non-SC central crossroads).
- The map has an exact C3 rotational symmetry `ROTATION` (a verified graph
  automorphism mapping A→B→C→A). This is exploited for parameter sharing
  (§4.1) and must be preserved by any map edit; `tests/test_map.py` enforces it.

Adjacency list (canonical):

```
CAP_A:  L_A, R_A, GATE_A          CAP_B:  L_B, R_B, GATE_B          CAP_C:  L_C, R_C, GATE_C
L_A:    CAP_A, GATE_A, B_CA       L_B:    CAP_B, GATE_B, B_AB       L_C:    CAP_C, GATE_C, B_BC
R_A:    CAP_A, GATE_A, B_AB       R_B:    CAP_B, GATE_B, B_BC       R_C:    CAP_C, GATE_C, B_CA
GATE_A: CAP_A, L_A, R_A, CTR      GATE_B: CAP_B, L_B, R_B, CTR      GATE_C: CAP_C, L_C, R_C, CTR
B_AB:   R_A, L_B, CTR
B_BC:   R_B, L_C, CTR
B_CA:   R_C, L_A, CTR
CTR:    GATE_A, GATE_B, GATE_C, B_AB, B_BC, B_CA
```

Adjacency matrix (K = CAP, G = GATE; `1` = adjacent, `·` = not):

|        | K_A | L_A | R_A | G_A | K_B | L_B | R_B | G_B | K_C | L_C | R_C | G_C | B_AB | B_BC | B_CA | CTR |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|-----|
| K_A    | ·   | 1   | 1   | 1   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·    | ·    | ·    | ·   |
| L_A    | 1   | ·   | ·   | 1   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·    | ·    | 1    | ·   |
| R_A    | 1   | ·   | ·   | 1   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | 1    | ·    | ·    | ·   |
| G_A    | 1   | 1   | 1   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·    | ·    | ·    | 1   |
| K_B    | ·   | ·   | ·   | ·   | ·   | 1   | 1   | 1   | ·   | ·   | ·   | ·   | ·    | ·    | ·    | ·   |
| L_B    | ·   | ·   | ·   | ·   | 1   | ·   | ·   | 1   | ·   | ·   | ·   | ·   | 1    | ·    | ·    | ·   |
| R_B    | ·   | ·   | ·   | ·   | 1   | ·   | ·   | 1   | ·   | ·   | ·   | ·   | ·    | 1    | ·    | ·   |
| G_B    | ·   | ·   | ·   | ·   | 1   | 1   | 1   | ·   | ·   | ·   | ·   | ·   | ·    | ·    | ·    | 1   |
| K_C    | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | 1   | 1   | 1   | ·    | ·    | ·    | ·   |
| L_C    | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | 1   | ·   | ·   | 1   | ·    | 1    | ·    | ·   |
| R_C    | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | 1   | ·   | ·   | 1   | ·    | ·    | 1    | ·   |
| G_C    | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | 1   | 1   | 1   | ·   | ·    | ·    | ·    | 1   |
| B_AB   | ·   | ·   | 1   | ·   | ·   | 1   | ·   | ·   | ·   | ·   | ·   | ·   | ·    | ·    | ·    | 1   |
| B_BC   | ·   | ·   | ·   | ·   | ·   | ·   | 1   | ·   | ·   | 1   | ·   | ·   | ·    | ·    | ·    | 1   |
| B_CA   | ·   | 1   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | ·   | 1   | ·   | ·    | ·    | ·    | 1   |
| CTR    | ·   | ·   | ·   | 1   | ·   | ·   | ·   | 1   | ·   | ·   | ·   | 1   | 1    | 1    | 1    | ·   |

Degrees: twelve provinces of degree 3, three gates of degree 4, CTR of
degree 6. Strategic reading: two routes between any pair of powers (flank →
shared border SC, or gate → CTR → enemy gate); CTR is the only province from
which every neutral SC can be supported.

### 3.2 Game flow

- **Setup:** each power starts with 3 armies on its 3 home SCs; home SCs
  owned by their power, border SCs unowned.
- **Year structure:** Spring movement → Fall movement → SC-ownership update →
  Winter adjustment → next year.
- **SC capture:** after the Fall movement only, each SC occupied by a unit
  changes owner to that unit's power. Unoccupied SCs keep their owner.
- **Winter adjustment:** let `delta = |owned SCs| − |units|` per power.
  - `delta > 0`: build up to `delta` armies, each in a *vacant home SC still
    owned by the power*; may waive.
  - `delta < 0`: must disband `|delta|` units of the power's choice.
- **Dislodgement = removal.** There is **no retreat phase** (deliberate
  deviation from standard rules to reduce phase heterogeneity; units are
  recoverable via Winter builds). Document this in the report.
- **Elimination:** a power with 0 SCs after Winter is eliminated; its units
  are removed. Game continues for the rest.
- **Termination:** solo victory when a power owns ≥ 7 SCs after any Winter;
  otherwise hard cap at 40 movement phases (20 years) → scored draw.

### 3.3 Order grammar

Movement-phase orders per army at province `s` (all destinations must satisfy
adjacency; illegal orders are coerced to HOLD at adjudication):

```
HOLD           A s H
MOVE           A s - d            d ∈ adj(s)
SUPPORT-HOLD   A s S t            t ∈ adj(s)          (supports the unit in t to stay)
SUPPORT-MOVE   A s S u - d        d ∈ adj(s), d ∈ adj(u), u ≠ s
```

Adjustment-phase orders: `BUILD h` (h ∈ own vacant owned home SCs),
`DISBAND s` (s ∈ own unit locations), `WAIVE`.

**Global order vocabulary is fixed and enumerable** (precompute once in
`triad/engine/orders.py`, index-stable):

| category      | count | formula                                   |
|---------------|-------|-------------------------------------------|
| HOLD          | 16    | one per province                           |
| MOVE          | 54    | directed edges = Σ deg                     |
| SUPPORT-HOLD  | 54    | ordered adjacent pairs = Σ deg             |
| SUPPORT-MOVE  | 138   | Σ_d deg(d)·(deg(d)−1)                      |
| BUILD/DISBAND/WAIVE | 26 | 9 + 16 + 1                            |
| **total**     | **288** |                                          |

The policy head is a single softmax over this 288-vocabulary with a hard
legality mask (illegal logits → −inf). Legality of an order for a given unit
depends only on (unit location, order); precompute a boolean table
`legal[province, order_id]` once, plus dynamic masks for support targets /
builds where board state matters.

### 3.4 Adjudication algorithm

Armies only and no convoys ⇒ **no convoy paradoxes exist**; a bounded
iterative resolution is exact. Strength definitions follow Kruijswijk's
standard treatment ("The Math of Adjudication"). Implement in
`triad/engine/adjudicator.py` as a pure function
`resolve(board, orders) -> (new_board, dislodged, results)` with no hidden
state, fully deterministic.

```
ADJUDICATE(movement orders):

1. LEGALIZE. Any order that is syntactically invalid, refers to a nonexistent
   unit, or violates adjacency → replace with HOLD.

2. SUPPORT CUT (static pass). A support given by the unit at p, directed into
   province q (q = destination of the supported move, or q = t for
   support-hold), is CUT iff some unit of ANOTHER power orders a move r → p
   with r ≠ q. (A power never cuts its own support. An attack out of q itself
   can only break the support by dislodging the supporter — handled in step 6.
   Failed attacks still cut.)

3. STRENGTHS (with current cut flags):
   attack(m: s→d)  = 1 + |uncut supports of m|
     • if the unit at d belongs to the mover's own power and does not
       successfully vacate, m FAILS outright (no self-dislodgement)
     • supports given by the power that owns the unit at d are EXCLUDED from
       attack(m) when comparing against that unit's hold strength (a power's
       support never helps dislodge its own unit)
   prevent(m: s→d) = 1 + |uncut supports of m|, except 0 if m is the LOSER of
       a head-to-head battle
   defend(m: s→d)  = 1 + |uncut supports of m|   (used only in head-to-head)
   hold(d) = 0 if d empty, or if d's occupant has a move order that succeeds
           = 1 if d's occupant has a move order that fails
           = 1 + |uncut support-holds for it| if occupant holds or supports

4. MOVE RESOLUTION (iterate to fixpoint; hold(d) resolves lazily because it
   depends on whether the occupant vacates):
   move m: s→d SUCCEEDS iff
     • head-to-head case (the unit at d has a move order to s):
         attack(m) > defend(opposing move)  AND
         attack(m) > prevent(m') for every other move m' into d
     • otherwise:
         attack(m) > hold(d)  AND
         attack(m) > prevent(m') for every other move m' into d
   After the fixpoint loop, any remaining set of mutually-waiting moves forms
   a cycle (each waiting for the next to vacate) with no head-to-head inside:
   ALL succeed (circular-movement rule).
   Ties: no unit enters (standoff — including standoffs over empty provinces,
   which stay empty); an occupant attacked by two equal-strength foreign moves
   stays (beleaguered garrison).

5. DISLODGEMENT. The unit at d is dislodged iff some move into d succeeded
   and the unit did not itself successfully vacate. Dislodged units are
   REMOVED from the board (no retreats).

6. RETROACTIVE SUPPORT VOID. If a supporter at p was dislodged by a move
   originating from q (the province its support was directed into), void that
   support and re-run from step 3. Loop until stable; with no convoys this
   converges (bound iterations by the number of support orders and assert).
```

**Correctness protocol (non-negotiable):**
- Adapt the armies-only cases of the DATC (Diplomacy Adjudicator Test Cases)
  sections 6.A (basic), 6.C (circular movement), 6.D (supports and
  dislodges), 6.E (head-to-head) to Triad geography. Skip fleet/convoy/
  retreat/coast cases. Keep a table in `tests/test_adjudicator_datc.py`
  mapping each test to its DATC ID.
- Property tests: unit conservation (units after = units before − dislodged),
  at most one unit per province, determinism (same input → same output),
  C3 equivariance (rotate board+orders by `ROTATION` → rotated outcome).
- The equivariance test is the strongest single check; run it on ≥10k random
  order profiles.

---

## 4. RL design

### 4.1 Observation encoding (rotation-to-own-frame)

Before each power acts, apply `ROTATION^k` so the acting power always sees
itself as "power A". Powers are then encoded relationally as
{self, next, prev}. One shared network plays all three seats with **exact**
weight sharing and no power-identity input. This is the payoff of the
symmetric map — mention prominently in the report.

Per-province features (16 × F matrix, canonical province order after
rotation):
- unit owner: one-hot {none, self, next, prev} (4)
- SC owner: one-hot {not-an-SC/unowned, self, next, prev} (4) + is_SC flag (1)
- is home center of {self, next, prev} (3)

Global features (broadcast or concatenated): season one-hot (2), phase
one-hot (movement/adjustment) (2), year / 20 (1), SC counts and unit counts
for {self, next, prev} scaled by /12 and /12 (6).

### 4.2 Model

- **Baseline (build first):** flatten 16×12 + globals → MLP (2×256, ReLU) →
  shared torso; heads: policy (§4.3) and value.
- **Upgrade (only after PPO works):** 3-layer GAT/GraphConv over `ADJACENCY`
  (edge list from `map_data`), mean-pool + per-province embeddings. Keep the
  MLP as an ablation row.
- **Value head:** softmax over {self, next, prev} predicting win/draw-share
  probability (DipNet-style); train with cross-entropy on final outcome
  shares. Self component doubles as the PPO baseline.

### 4.3 Action decoding (autoregressive with masking)

Units are ordered by canonical province index. Decode one order per own unit
sequentially; condition each step on the torso output + an embedding of
orders already emitted this phase (GRU over emitted order-ids is sufficient).
Each step: logits over the 288-vocab, hard-masked to legal orders for that
unit and phase, sample (training) or argmax (eval). BC uses the same decoder
with teacher forcing. Log-prob of the joint action = sum over units (needed
for PPO ratios).

Why autoregressive: independent per-unit heads cannot coordinate supports
(unit 2's best order depends on whether unit 1 attacks). Know this cold for
interviews.

### 4.4 Training pipeline

**Stage 0 — heuristic bots** (`triad/bots/`): RandomLegal; Grabber (moves
toward nearest capturable SC via BFS distance, naive supports when two units
share a target); Turtle (holds SCs, support-holds neighbors). Bots are pure
policies usable as env opponents and BC teachers.

**Stage 1 — behavior cloning:** generate ≥50k games of mixed bot matchups
(Grabber-heavy); train the policy by cross-entropy on Grabber's orders
conditioned on rotated observations. Acceptance: BC vs 2×RandomLegal solo-win
rate > 80%; BC vs 2×Grabber roughly 1/3 each.

**Stage 2 — PPO self-play with KL anchor:** all three seats played by the
learning policy (or population samples, stage 3). Loss = PPO clipped
surrogate + value CE + entropy bonus + `β · KL(π_θ ‖ π_BC)` with π_BC frozen.
Start β = 0.05, sweep {0, 0.01, 0.05, 0.2}. GAE(λ=0.95), γ=1.0 (finite
episodes, terminal-heavy reward), clip 0.2, lr 3e-4 with anneal. Rollouts
from 64 vectorized envs; PPO implementation adapted from CleanRL (vendor the
file and modify; do not depend on cleanrl as a library).

**Stage 3 — population play:** snapshot every N updates into a pool; per
game sample opponents 80% latest / 20% uniform-past. Prevents self-play
cycling.

### 4.5 Rewards

Per power, at termination only (plus optional shaping):
- solo win: 1.0; all others 0.0
- elimination: 0.0
- turn-cap draw: `SC_i / 12`
- optional shaping: `+α · ΔSC_i` at each Winter, α = 0.02 default; **α = 0 is
  an ablation row** (tests the draw-attractor hypothesis).

---

## 5. Evaluation

- **Round-robin tournaments:** each evaluated policy vs all pairs of fixed
  opponents {RandomLegal, Grabber, Turtle, BC, final}, ≥500 games per
  matchup, greedy (argmax) actions, report solo/draw/eliminated rates with
  Wilson 95% CIs.
- **Elo/TrueSkill** across the checkpoint pool (3-player games → use
  TrueSkill, or Elo on pairwise-reduced results; `trueskill` pip package is
  fine).
- **Symmetry sanity check:** identical policies in all seats ⇒ per-seat solo
  rates must be statistically indistinguishable (χ² test). Any seat effect =
  bug in rotation logic. Free correctness test; report it.
- **Attack-the-leader index (headline analysis):** among the two non-leading
  powers, P(a unit's order targets the leader's units or SCs) as a function
  of the leader's SC lead. Compare final RL agent vs Grabber baseline. Plot.
- **Ablation grid:** {BC only, PPO from scratch, BC+PPO, BC+PPO+KL} ×
  {shaping α=0.02, α=0}. One table, tournament win rates vs the fixed bot
  set.

---

## 6. Repository layout

```
triad-rl/
  CLAUDE.md                  # this file — keep §2 status current
  pyproject.toml             # editable install: pip install -e .
  requirements.txt
  triad/
    __init__.py
    map_data.py              # canonical map (DO NOT duplicate literals)
    engine/
      state.py               # Board dataclass: units, SC owners, phase, year
      orders.py              # order vocab enumeration, legality tables, parsing
      adjudicator.py         # pure resolve() per §3.4
      game.py                # phase loop, captures, adjustments, termination
    env/
      triad_env.py           # PettingZoo ParallelEnv wrapper
      vec.py                 # batched env stepping
      obs.py                 # rotation-to-own-frame observation builder
    bots/
      random_legal.py  grabber.py  turtle.py
    rl/
      models.py              # torso (MLP / GNN), AR decoder, value head
      bc.py                  # dataset generation + BC training loop
      ppo.py                 # vendored/modified CleanRL PPO + KL anchor
      population.py          # snapshot pool + opponent sampling
    eval/
      tournament.py  ratings.py  metrics.py   # incl. attack-the-leader
  tests/
    test_map.py              # invariants (exists, passing)
    test_adjudicator_datc.py # DATC-adapted cases + property tests
    test_env.py
  scripts/
    gen_bc_data.py  train_bc.py  train_ppo.py  run_tournament.py
  configs/
    triad.yaml               # map=triad, all hyperparameters
    pentad.yaml              # stretch goal: 5-power symmetric map, config-only
```

## 7. Engineering conventions

- Python ≥ 3.10, `pip install -e .` (editable). Type hints on public
  functions. `pytest -q` must pass before any training run.
- Determinism: every stochastic component takes an explicit
  `numpy.random.Generator` / torch generator; seeds recorded in run configs.
- The engine is numpy/pure-python and torch-free; the RL layer never touches
  engine internals except through the env API.
- Build order is strict: engine+tests → bots → env → BC → PPO. Do not start
  a later stage while the previous stage's acceptance criterion fails.
- Performance target (M1): ≥ 300 complete random-bot games/sec single
  process. Profile before optimizing; correctness first.
- Log to TensorBoard by default; W&B optional via config flag.

## 8. Dependencies

See `requirements.txt`. Rationale: torch (models), gymnasium + pettingzoo
(env API), networkx (map analysis/plots only — never in the hot path),
trueskill (3-player ratings), matplotlib (figures), tensorboard (logging),
pytest (tests), pyyaml/tqdm (glue). `torch-geometric` is deliberately
excluded from baseline requirements — install only if/when the GNN upgrade
lands (M4+), since the MLP baseline must work first and PyG installs are
platform-fussy. wandb optional.
