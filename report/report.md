# TRIAD-RL: does balance-of-power behaviour emerge from self-play RL in a symmetric 3-player no-press Diplomacy variant?

Srirang Nabar — July 2026

## Abstract

I designed a rotationally symmetric 3-player Diplomacy variant ("Triad") and
trained agents on it with the DipNet recipe (behaviour cloning from scripted
bots, then PPO self-play with a KL anchor to the cloned policy), to ask one
question: with no communication, do the two trailing players learn to gang up
on the leader? **The answer is no — the trained agents learn the opposite.**
Trailing powers target the leader *below* a chance-corrected baseline
(prey-selection: they eat the weaker neighbour instead), leads convert to
victories *more* reliably than under scripted play (snowballing), and
cross-power supports — the mechanistic signature of coalition — are
essentially absent in every competent configuration. The negative result has
a coherent incentive explanation: stopping the leader is a public good
between the trailers, and with no communication or reputation the free-rider
problem wins. Along the way, two secondary results: the DipNet claim that
from-scratch RL fails reproduces at small scale (2.8% vs 92.4% win rate at
equal budget), and imitation anchoring is shown to actively suppress the
coalition machinery — a bias anyone studying emergent cooperation under a BC
prior should expect. Methodologically, a cheap statistical symmetry test
caught two real equivariance bugs that dedicated unit-test suites could not
see; I consider that pipeline the most transferable artefact of the project.

Everything is reproducible: `git clone`, `uv sync`, and the shipped
checkpoints + seeds regenerate every number in this report.

## 1. The game

![Triad map](figs/map.png)

**Triad**: 16 provinces, 30 edges, 12 supply centers (SCs), armies only,
3 powers. Simultaneous moves with the standard Diplomacy order grammar
(hold / move / support-hold / support-move), strict-majority solo victory at
7/12 SCs, hard cap at 20 game-years scored as SC-proportional draw.

Design choices, each load-bearing:

- **Three players** is the minimum for "who do I attack?" to be a strategic
  question — two-player zero-sum games have minimax play and no alliances.
  **No press** (no communication) means any coalition that appears must be
  *implicit*, learned from incentives alone.
- **Exact C3 rotational symmetry.** The map is invariant under a verified
  graph automorphism mapping each power to the next. This buys exact
  fairness across seats and, more importantly, exact weight sharing: every
  observation *and every action* is expressed in the acting player's own
  rotated frame, so one network plays all three seats with no seat-identity
  input. (§5 explains why "and every action" is italicised.)
- **Deliberate simplifications**, documented rather than hidden: no fleets
  or convoys (eliminates the convoy-paradox class — adjudication becomes a
  provably convergent fixpoint), no retreat phase (dislodgement = removal;
  units are recoverable via builds).

## 2. System

- **Engine** (pure Python, torch-free): Kruijswijk-style adjudication
  (support cutting, head-to-head, beleaguered garrison, circular movement,
  retroactive support voiding). Correctness protocol: the armies-only
  DATC cases (sections 6.A/C/D/E) adapted to Triad geography, plus property
  tests — unit conservation, determinism, and **C3 equivariance of the
  adjudicator on 10,000 random positions**. Throughput ~390 complete random
  games/sec single-process, so data generation is never the bottleneck.
- **Model**: 203-dim rotated observation → MLP torso (2×256) → (a) an
  autoregressive order decoder — a GRU emits one order per unit, conditioned
  on the orders already emitted, hard-masked to legal orders over a fixed
  336-order vocabulary; (b) a value head over {self, next, prev} outcome
  shares. Autoregression is necessary: independent per-unit heads cannot
  coordinate supports (unit 2's best order depends on what unit 1 does).
  The joint log-prob is the sum over decode steps — exactly what PPO needs.
- **Training**: behaviour cloning from a scripted "Grabber" bot (1.25M
  decisions from 50k mixed-bot games), then PPO self-play with loss
  `clip-surrogate + value + entropy + β·KL(π‖π_BC)`, γ=1.0, GAE(0.95),
  64 vectorised envs, population-based opponent sampling (80% live policy /
  20% snapshot pool). Standard local budget: 500k env-steps ≈ 30–40 min on
  an 8-core CPU — every experimental row uses the same budget.
- **Reproducibility**: checkpoints are plain state-dict payloads saved on
  CPU and loaded with `weights_only=True` (train anywhere, run anywhere);
  final weights ship in-repo with a SHA256 manifest; CI loads every shipped
  checkpoint and plays a full game; every script has a `--smoke` mode run
  in CI, so nothing reaches long compute untested.

## 3. The pipeline works

Behaviour cloning reaches statistical parity with its own teacher (32.8% vs
two Grabbers; a faithful clone of the same strategy cannot exceed 1/3) and
100% vs random bots. PPO then lifts it far past the teacher:

![ablation curves](figs/ablation_curves.png)

**The DipNet result reproduces at small scale**: from-scratch PPO at the
identical budget reaches 2.8%/13.6% vs two Grabbers across two runs, against
92.4% for the same algorithm started from the BC weights — a 7–30× win-rate
multiplier from the imitation bootstrap. (Honest caveat: the scratch curves
are still creeping upward, so the claim is "fails at this budget", not
"can never learn".)

Tournament over all 35 lineups of {random, grabber, turtle, bc, final},
10,800 games, sampled actions, seat-rotated; matchups cited here use 2000
games:

![tournament](figs/tournament.png)

| policy | solo rate (all lineups) | 95% CI | draws | eliminated |
|---|---|---|---|---|
| final (BC+PPO, β=0) | **0.543** | [0.533, 0.552] | 0.003 | 0.016 |
| bc | 0.379 | [0.366, 0.393] | 0.001 | 0.031 |
| grabber | 0.243 | [0.235, 0.251] | 0.001 | 0.030 |
| random | 0.045 | [0.038, 0.053] | 0.170 | 0.072 |
| turtle | 0.000 | [0.000, 0.001] | 0.266 | 0.000 |

The final agent wins **93.4% vs two Grabbers** (2000 games) and ≥1/3 of
every pairing it appears in. TrueSkill over the same games (mirror lineups
excluded — rating a policy against itself is uninformative) gives the same
ordering. The **seat-symmetry χ² on the final agent's mirror lineup:
p = 0.234 at 2000 games** — the weight-sharing claim holds at citation
sample size.

## 4. The headline: balance of power does not emerge

Three measurements, three levels of the same question, 400 traced self-play
games per configuration.

**(a) Chance-corrected attack-the-leader.** The leader owns more of the
board, so a score-blind policy mechanically hits it more; the raw targeting
rate proves nothing. I therefore report the *excess* over a target-blind
null (the leader's share of each unit's adjacent attackable objects), and
compare against the scripted Grabber baseline, which is score-blind by
construction:

![attack the leader](figs/attack_the_leader.png)

The excess is **negative for every configuration at every lead size** — and
*more* negative for the trained agents (−0.10 at k=1) than for the scripted
baseline (−0.06). Part of the negative level is rational defence-avoidance
(the leader's holdings are better protected), which is exactly why the
baseline comparison matters: relative to it, the RL agents actively
prefer attacking the *weaker* neighbour. Prey-selection, not balancing.

**(b) Lead-conversion.** If trailing powers coalesced, big leads would stop
converting into victories:

![lead conversion](figs/lead_conversion.png)

The opposite: at a 2-SC lead, conversion rises from 0.50 (grabber) and 0.60
(bc) to **0.84 under the final RL agent**. Leads snowball. (Low-k buckets
are partly tautological — winners necessarily pass through high leads — so
the cross-config comparison at fixed k is the informative reading.)

**(c) Cross-power supports** — the direct mechanistic signature of an
implicit coalition (the grammar allows supporting another power's unit):

| config | cross-support / order | …directed at the leader | self-play solo rate |
|---|---|---|---|
| grabber (scripted) | 0.000 | 0.000 | 1.00 |
| bc | 0.0001 | 0.0001 | 1.00 |
| scratch | 0.062 | 0.017 | 0.27 |
| **RL β=0 (final)** | 0.0033 | 0.0005 | 0.99 |
| RL β=0.01 | 0.0001 | 0.000 | 0.99 |
| RL β=0.05 | 0.0001 | 0.000 | 0.84 |
| RL β=0.2 | 0.0001 | 0.0001 | 1.00 |

Two findings live in this table. First, **the KL anchor suppresses the
coalition channel**: the BC teacher never supports foreign units, so the
anchor inherits and enforces that; even β=0.01 keeps the rate at the clone's
floor. This was predicted at planning time and is why every metric here is
computed per β. Second — and decisive for the research question — **even the
unanchored agent barely uses the channel** (0.3% of orders, 0.05% directed
at the leader) despite fully possessing it, and that same agent shows the
most negative leader-targeting of all. The capability exists; the incentives
never select it. (Scratch's high rate is entropy noise from a barely
competent policy — it is the "no learned prior" floor, which makes the
competent zeros more striking.)

### Why: a free-rider story

Stopping the leader is a public good between the two trailing players: the
blocker pays the full cost (units committed, home exposed) while the benefit
(leader slowed) is shared. Attacking the *other trailer* is a private gain.
With no communication, no persistent identity across games, and no
punishment mechanism, there is nothing to stabilise contribution to the
public good — so independent self-interested learners converge on
prey-selection and snowballing. The classical balance-of-power of IR theory
presupposes machinery (signalling, reputation, enforceable agreements) that
this environment deliberately lacks. Consistent with this reading, the one
quasi-balancing regime observed anywhere in the project is the
**draw-attractor**: with shaping *and* anchor pressure both removed
(α=0, β=0), self-play collapses into symmetric mutual defence (119/150
draws) — cooperation by universal stalemate rather than by coalition.

## 5. Methodological result: the χ² that caught two bugs

The plan included a "free correctness test": identical policies in all three
seats must win at statistically indistinguishable rates. This cheap check
caught **two real bugs that dedicated equivariance test suites missed**:

1. **Decode-order bug (M3).** Units were decoded in real-frame canonical
   province order, which the rotation does not preserve (border-SC indices
   permute cyclically). Same relational position → different decode
   *sequence* per seat. Signature: stable C>A>B seat ordering, pooled
   p≈10⁻⁴ across replications.
2. **Action-frame bug (M4).** The order vocabulary itself was never
   rotated: observations were own-frame, action ids real-frame, so identical
   observations carried seat-dependent legality masks — weight sharing was
   structurally broken. Found when the β=0 run (free of the anchor that had
   been masking the effect) failed at p≈10⁻⁸; pinned by a
   trajectory-equivariance experiment (rotate the per-seat seed assignment →
   every board must rotate; 12/12 diverged at move one); fixed with a vocab
   permutation. Fixing it dropped BC's validation loss from 0.73 to 0.38 —
   the network had been asked to learn seat-conditional mappings from
   seatless inputs.

Both bugs are now permanent regression tests, and the final agent's deep
check reads p=0.578 over 600 games (plus the cited p=0.234 at 2000). The
general lesson: *content* equivariance, *outcome* equivariance, and
*interface* equivariance (ordering, action frames) are separate properties;
statistical invariance tests over the full pipeline are the only net that
catches all of them.

## 6. Limitations

- **Budget.** 500k env-steps per run on a laptop CPU. The scratch rows and
  the β sweep might look different at 10–100×; the per-run cost (~35 min)
  and cloud-portable checkpoints make that extension mechanical.
- **Map decisiveness.** Triad's 7-of-12 threshold is reachable before
  opposition can organise (even random play solos ~55% of games). A larger
  or slower map might give balancing dynamics room to appear.
- **Reward design.** Solo-heavy rewards directly price aggression;
  draw-shares price survival only weakly. Reward mixes that price *relative*
  position could change the equilibrium — that is a follow-up, not a flaw,
  but it bounds the claim: the negative result is about *this* incentive
  structure, which is standard Diplomacy scoring.
- **One seed per experimental row** (grid economics). Sweep-level
  conclusions lean on consistency across rows rather than per-row CIs;
  game-level numbers all carry proper CIs.
- **Single-checkpoint evaluation per config** (best-snapshot where the
  curves said it matters). Population-pool archives allow re-evaluation.

## 7. Reproducing this report

```
git clone <repo> && cd Diplomacy_RL
uv sync                                    # installs deps + the triad package
uv run pytest -q                           # 115 tests, incl. shipped-weights checks
uv run python scripts/run_tournament.py    # regenerates results/tournament.json
uv run python scripts/run_bop_metrics.py   # regenerates results/bop_metrics.json
uv run python scripts/make_report_figs.py  # regenerates the figures above
```

Shipped checkpoints and their SHA256s: `weights/MANIFEST.md`. Analysis
notebooks (`notebooks/*.ipynb`) carry the full milestone-by-milestone
narrative, including both bug investigations.
