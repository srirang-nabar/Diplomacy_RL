"""Smoke-run protocol (CLAUDE.md §7.2): every script in scripts/ must run
end-to-end with --smoke in CI. A script without a passing smoke test does
not exist. Grows one entry per new script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _run_smoke(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--smoke"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )


def test_benchmark_engine_smoke():
    r = _run_smoke("benchmark_engine.py")
    assert r.returncode == 0, r.stderr
    assert "games/sec" in r.stdout


def test_gen_bc_data_smoke():
    r = _run_smoke("gen_bc_data.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


def test_train_bc_smoke():
    r = _run_smoke("train_bc.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout  # includes checkpoint save+reload


def test_eval_bc_smoke():
    r = _run_smoke("eval_bc.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout
