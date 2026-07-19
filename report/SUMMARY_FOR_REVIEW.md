# Reviewer summary — Strategy Emergence in Diplomacy (Multi-Agent RL)

**One paragraph.** In a symmetric 3-player game with simultaneous moves and no communication, does
*balance-of-power* behavior (trailing players ganging up on the leader) emerge from self-play RL?
The project builds "Triad" — a custom rotationally symmetric 3-player no-press Diplomacy variant —
with a DATC-adapted, property-tested adjudicator (115 tests incl. 10k-position equivariance checks),
then trains behavior-cloning + PPO with a KL anchor and answers the question with chance-corrected
metrics over a 10,800-game tournament.

**Findings:**

| Question | Result |
| -------- | ------ |
| Does balance-of-power emerge? | **No — the agents learn the opposite**: prey-selection (chance-corrected leader-targeting excess −0.10 vs −0.06 scripted baseline), snowballing (lead-conversion 0.84 vs 0.50), cross-power supports ≈ 0. Mechanism: stopping the leader is a public good between trailers — free-riding wins without communication |
| Does the DipNet bootstrap result reproduce at small scale? | **Yes** — PPO-from-scratch fails at equal budget (2.8%/13.6% vs 2×Grabber) while BC+PPO reaches **92.4%**; final agent 54.3% aggregate in round-robin |
| Methodological headline | A cheap **χ² symmetry test** (identical policies ⇒ equal seat win-rates) caught **two bugs invisible to accuracy metrics** (p≈1e-8) — root-caused, guarded by permanent tests, everything retrained |

**How to review quickly (~5 min):**

**Fastest path: `notebooks/00_review_walkthrough.ipynb`** — a single commented, pre-executed notebook backing every resume point, with the asserts inline.

1. Read the headline table in `README.md`, then open `notebooks/m5_results.ipynb` (tournament,
   attack-the-leader and lead-conversion analyses, commented) — `report/report.md` has the full
   write-up and figures.
2. Optional: `uv sync && uv run pytest -q` (engine + checkpoint smoke tests, CPU-only); shipped
   weights in `weights/` load on any machine and reproduce the tournament tables within stated CIs.

**Scope honesty:** armies-only variant (no fleets/retreats/negotiation); 500k-env-step budget per run;
findings are about this game class and budget — the KL anchor demonstrably suppresses coalition orders,
which is analyzed per-β rather than hidden.
