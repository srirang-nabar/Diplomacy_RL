# TRIAD-RL

Multi-agent reinforcement learning on **Triad** — a custom, rotationally
symmetric 3-player no-press Diplomacy variant — asking one question:

> In a symmetric three-player game with simultaneous moves and no
> communication, does *balance-of-power* behaviour (the trailing players
> ganging up on the leader) emerge from self-play RL?

**Answer: no — the trained agents learn the opposite** (prey-selection and
snowballing), and the result comes with a chance-corrected measurement
methodology and a coherent free-rider explanation. Full write-up:
[`report/report.md`](report/report.md).

## Headline numbers

| | |
|---|---|
| Engine correctness | DATC-adapted suite + C3 equivariance on 10k positions; 115 tests |
| BC clone | 32.8% vs 2× its own teacher (= statistical parity), 100% vs random |
| BC+PPO (final agent) | **92.4%** vs 2×Grabber; 54.3% across a 10,800-game round-robin |
| PPO from scratch, equal budget | 2.8% / 13.6% — **the DipNet bootstrap result reproduces** |
| Seat-symmetry χ² (weight-sharing check) | p = 0.234 @ 2000 games |
| Leader-targeting excess (RL vs scripted baseline) | **−0.10 vs −0.06** — agents hunt the *weaker* neighbour |
| Lead conversion at +2 SCs (RL vs baseline) | **0.84 vs 0.50** — leads snowball |
| Cross-power supports (coalition signature) | ≈ 0 in every competent config; suppressed monotonically by the KL anchor |

The project's most transferable artefact is methodological: a cheap
statistical symmetry test (identical policies in all seats ⇒ equal win
rates) caught **two real equivariance bugs** — decode *order*, then action
*frame* — that bitwise content tests could not see. The story is told in
the analysis notebooks and in §5 of [`report/report.md`](report/report.md).

## Reproduce everything

```bash
git clone <this repo> && cd Diplomacy_RL
uv sync                                    # deps + editable install (Python 3.13)
uv run pytest -q                           # 115 tests, incl. SHA256 + load + play
                                           # of every shipped checkpoint (CPU-only)
uv run python scripts/run_tournament.py    # 10,800 games -> results/tournament.json
uv run python scripts/run_bop_metrics.py   # traced metric runs -> results/bop_metrics.json
uv run python scripts/make_report_figs.py  # regenerate the report figures
```

Trained weights ship in-repo (`weights/`, ~2 MB each) with a SHA256
manifest — no downloads, no GPU required for anything above. Training runs
reproduce with `scripts/train_bc.py` / `train_ppo.py` / `run_m4_grid.py`
(seeds and configs recorded in `configs/triad.yaml` and
`results/m4_registry.json`; ~35 min per run on an 8-core CPU).

## Repository map

```
src/triad/
  map_data.py          canonical map: 16 provinces, C3 automorphism (single source of truth)
  engine/              torch-free game engine + Kruijswijk adjudicator
  env/                 rotation-to-own-frame obs/action interface, PettingZoo env
  bots/                RandomLegal / Grabber / Turtle (scripted, rng tie-breaking)
  rl/                  models (AR decoder), BC, PPO+KL, population play, checkpointing
  eval/                tournament, TrueSkill, balance-of-power metrics
tests/                 115 tests: DATC, property/equivariance, metrics on worked examples
notebooks/             executed analyses: map, BC, PPO, ablation grid, final results
results/               tournament + metrics JSONs, M4 run registry
weights/               shipped checkpoints + SHA256 MANIFEST
report/                report, 10-slide deck (Marp), interview Q&A
```

## Method in one paragraph

A 16-province map with an exact C3 rotational symmetry lets one network play
all three seats: every observation *and every action id* is expressed in the
acting player's own rotated frame (both halves matter — see the action-frame
bug in the report). Orders decode autoregressively over a fixed 336-order
vocabulary under hard legality masks. Training follows the DipNet recipe at
small scale: behaviour cloning from a scripted bot (1.25M decisions), then
PPO self-play with a KL anchor to the clone, population-based opponent
sampling, and an 8-run ablation grid ({scratch, β=0, 0.01, 0.05, 0.2} ×
shaping) at a fixed measured budget. Evaluation: 10,800-game round-robin
with Wilson CIs and TrueSkill, plus three levels of balance-of-power
measurement (chance-corrected targeting, lead conversion, cross-power
supports), each per-β because the anchor provably biases the coalition
channel.
