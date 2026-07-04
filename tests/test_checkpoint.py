"""Shipped-weights smoke test (CLAUDE.md §7.1, runs in CI on CPU).

For every checkpoint committed under weights/: verify its SHA256 against
weights/MANIFEST.md, load it CPU-only via the one true load path, and play a
full seeded game against 2x RandomLegal without error. Proves the
clone-and-run promise on every push.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "weights"
MANIFEST = WEIGHTS / "MANIFEST.md"

_checkpoints = sorted(WEIGHTS.glob("*.pt")) if WEIGHTS.is_dir() else []


def _manifest_sha(name: str) -> str:
    text = MANIFEST.read_text()
    m = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*`?([0-9a-f]{{64}})`?", text)
    assert m, f"{name} missing from MANIFEST.md"
    return m.group(1)


@pytest.mark.skipif(not _checkpoints, reason="no shipped weights yet")
def test_manifest_exists():
    assert MANIFEST.is_file(), "weights/ has checkpoints but no MANIFEST.md"


@pytest.mark.parametrize("ck", _checkpoints, ids=[c.name for c in _checkpoints])
def test_shipped_checkpoint(ck: Path):
    # 1. integrity
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    assert sha == _manifest_sha(ck.name), f"SHA256 mismatch for {ck.name}"
    # 2. CPU-only load through the one true path
    from triad.rl.checkpoint import load_policy

    model, payload = load_policy(ck, device="cpu")
    assert payload["order_vocab_size"] == 336
    # 3. plays a full seeded game vs 2x RandomLegal without error
    from triad.map_data import POWERS
    from triad.bots import RandomLegal
    from triad.bots.base import play_game
    from triad.rl.policy_bot import PolicyBot

    bots = {
        "A": PolicyBot(model, greedy=True),
        "B": RandomLegal(),
        "C": RandomLegal(),
    }
    g = play_game(bots, np.random.default_rng(0))
    assert g.over and g.result is not None
