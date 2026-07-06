"""Population-based opponent sampling (CLAUDE.md §4.4 stage 3).

During PPO training, each seat of each episode independently plays either the
live policy theta (probability p_latest, default 0.8) or a frozen snapshot
sampled uniformly from the pool (0.2). Only theta-seats contribute training
samples. Snapshots are ordinary §7.1 checkpoints saved every `snapshot_every`
updates — the pool doubles as the archive for post-hoc snapshot selection
(M3 finding: eval-vs-fixed-bots peaks early; the best checkpoint is often not
the last one).

Purpose: prevents self-play cycling — past strategies stay alive as opponents,
so the policy cannot forget how to beat them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from triad.rl.checkpoint import load_policy, save_checkpoint
from triad.rl.models import TriadPolicy

LATEST = -1  # seat-source value meaning "live policy theta"


class PolicyPool:
    def __init__(self, outdir: str | Path):
        self.dir = Path(outdir) / "pool"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[tuple[int, Path]] = []  # (update, path), append-only
        self._cache: dict[int, TriadPolicy] = {}

    def __len__(self) -> int:
        return len(self.snapshots)

    def add(self, model: TriadPolicy, update: int, *, seed: int | None = None) -> Path:
        path = save_checkpoint(
            model, self.dir / f"snap_{update:04d}.pt",
            train_config={"snapshot_update": update}, seed=seed,
        )
        self.snapshots.append((update, path))
        return path

    def get(self, idx: int) -> TriadPolicy:
        """Snapshot by pool index, loaded once and cached (eval mode, CPU)."""
        if idx not in self._cache:
            model, _ = load_policy(self.snapshots[idx][1], device="cpu")
            self._cache[idx] = model
        return self._cache[idx]

    def sample_seat_sources(
        self, n_seats: int, rng: np.random.Generator, p_latest: float = 0.8
    ) -> np.ndarray:
        """Per-seat policy source for one fresh episode: LATEST with
        probability p_latest, else a uniform pool snapshot. Empty pool ->
        all LATEST (plain self-play)."""
        out = np.full(n_seats, LATEST, dtype=np.int64)
        if not self.snapshots:
            return out
        for s in range(n_seats):
            if rng.random() >= p_latest:
                out[s] = int(rng.integers(len(self.snapshots)))
        return out
